#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.ndimage import minimum_filter, maximum_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pmf_io import (
    read_sequential_pmf,        # single PMF (new format)
    read_sequential_pmf_blocks, # history PMFs (multi-block)
    read_sequential_counts,     # history counts (multi-block)
    interpolate_pmf             # interpolation
)


class PMFAnalyzer:
    """
    Analysis-only class.

    IO is delegated to pmf_io:
      - history PMFs: read_sequential_pmf_blocks
      - single PMF:   read_sequential_pmf
      - counts:       read_sequential_counts
      - reference:    single PMF, interpolated to history grid
    """

    def __init__(self, pmf_file, count_file,
                 n_recent=4,
                 slope_thresh=1e-3,
                 use_sliding_window=False,
                 count_std_thresh=None,
                 reference_pmf_file=None,
                 rmsd_thresh=None):

        self.pmf_file = pmf_file
        self.count_file = count_file
        self.n_recent = n_recent
        self.slope_thresh = slope_thresh
        self.use_sliding_window = use_sliding_window
        self.count_std_thresh = count_std_thresh
        self.reference_pmf_file = reference_pmf_file
        self.rmsd_thresh = rmsd_thresh

        # ------------------------------------------------------------
        # 1. Read PMFs: history vs single
        # ------------------------------------------------------------
        try:
            if "hist" in self.pmf_file:
                # History file: multiple PMFs in one file
                pmf_blocks = read_sequential_pmf_blocks(self.pmf_file)
            else:
                # Single PMF file: wrap as a 1-element "history"
                coords, pmf = read_sequential_pmf(self.pmf_file)
                pmf_blocks = [(coords, pmf)]
        except Exception as e:
            raise RuntimeError(f"Failed to read PMF file '{self.pmf_file}': {e}")

        if len(pmf_blocks) == 0:
            raise RuntimeError(f"PMF file '{self.pmf_file}' contains no PMFs")

        self.pmfs = pmf_blocks
        self.pmf_coords = self.pmfs[0][0]
        self.pmf_values = [block[1] for block in self.pmfs]

        # ------------------------------------------------------------
        # 2. Read counts (history, multi-block)
        # ------------------------------------------------------------
        try:
            self.counts, self.count_coords = read_sequential_counts(self.count_file)
        except Exception as e:
            raise RuntimeError(f"Failed to read counts file '{self.count_file}': {e}")

        self.normed_counts = self._normalize_counts(self.counts)

        # ------------------------------------------------------------
        # 3. Read reference PMF (single PMF, new format)
        # ------------------------------------------------------------
        self.reference_pmf = None
        if self.reference_pmf_file is not None:
            try:
                ref_coords, ref_pmf = read_sequential_pmf(self.reference_pmf_file)
                self.reference_pmf = interpolate_pmf(ref_coords, ref_pmf, self.pmf_coords)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to read reference PMF '{self.reference_pmf_file}': {e}"
                )

        # ------------------------------------------------------------
        # 4. Compute RMSD time series
        # ------------------------------------------------------------
        try:
            self._compute_rmsd_series()
        except Exception as e:
            raise RuntimeError(f"RMSD/convergence setup failed: {e}")

        # ------------------------------------------------------------
        # 5. Smooth & fit RMSD
        # ------------------------------------------------------------
        self.rmsd_smooth = self._smooth_rmsd(self.rmsd_raw)
        self.params, self.rmsd_fit = self._fit_exp_decay()

        # ------------------------------------------------------------
        # 6. Detect convergence
        # ------------------------------------------------------------
        try:
            self.convergence_idx = self._detect_convergence()
        except Exception as e:
            raise RuntimeError(f"Convergence detection failed: {e}")

    # ============================================================
    # RMSD computation
    # ============================================================

    def _compute_rmsd_series(self):
        """
        Build self.rmsd_raw and self.t depending on:
          - reference_pmf (if provided)
          - use_sliding_window
        """

        def zero_min(arr):
            return arr - np.nanmin(arr)

        # ------------------------------------------------------------
        # 1. RMSD to external reference PMF (single PMF, already
        #    interpolated to history grid in __init__)
        # ------------------------------------------------------------
        if self.reference_pmf is not None:
            ref = zero_min(self.reference_pmf)

            rmsd_vals = []
            for pmf in self.pmf_values:
                pmf0 = zero_min(pmf)
                # use nanmean in case interpolation produced NaNs
                rmsd_vals.append(
                    np.sqrt(np.nanmean((pmf0 - ref) ** 2))
                )

            self.rmsd_raw = np.array(rmsd_vals)
            self.t = np.arange(len(self.rmsd_raw))
            return

        # ------------------------------------------------------------
        # 2. RMSD to final PMF (no sliding window)
        # ------------------------------------------------------------
        if not self.use_sliding_window:
            final = zero_min(self.pmf_values[-1])

            rmsd_vals = []
            for pmf in self.pmf_values:
                pmf0 = zero_min(pmf)
                rmsd_vals.append(
                    np.sqrt(np.mean((pmf0 - final) ** 2))
                )

            self.rmsd_raw = np.array(rmsd_vals)
            self.t = np.arange(len(self.rmsd_raw))
            return

        # ------------------------------------------------------------
        # 3. Sliding-window RMSD
        # ------------------------------------------------------------
        if len(self.pmf_values) <= self.n_recent:
            raise ValueError(
                f"Not enough PMF snapshots ({len(self.pmf_values)}) "
                f"for sliding window of size {self.n_recent}"
            )

        rmsd_vals = []
        for i in range(self.n_recent, len(self.pmf_values)):
            ref = np.mean(self.pmf_values[i - self.n_recent:i], axis=0)
            ref0 = zero_min(ref)

            pmf = self.pmf_values[i]
            pmf0 = zero_min(pmf)

            rmsd_vals.append(
                np.sqrt(np.mean((pmf0 - ref0) ** 2))
            )

        self.rmsd_raw = np.array(rmsd_vals)
        self.t = np.arange(self.n_recent, len(self.pmf_values))

    # ============================================================
    # Normalization
    # ============================================================

    def _normalize_counts(self, counts):
        c_min = min(np.min(c) for c in counts)
        c_max = max(np.max(c) for c in counts)
        return [(c - c_min) / (c_max - c_min + 1e-12) for c in counts]

    # ============================================================
    # RMSD smoothing & fitting
    # ============================================================

    def _smooth_rmsd(self, rmsd, window_length=11, polyorder=3):
        if len(rmsd) < 3:
            return rmsd
        wl = min(window_length, len(rmsd) if len(rmsd) % 2 else len(rmsd) - 1)
        return savgol_filter(rmsd, window_length=wl, polyorder=polyorder)

    def _exp_decay(self, t, A, B, C):
        return A * np.exp(-B * t) + C

    def _fit_exp_decay(self):
        try:
            params, _ = curve_fit(self._exp_decay, self.t, self.rmsd_smooth,
                                  p0=(1, 0.1, 0.01), maxfev=10000)
            return params, self._exp_decay(self.t, *params)
        except Exception:
            # Fit failure is not fatal; we just disable fit-based convergence
            return None, np.full_like(self.t, np.nan)

    # ============================================================
    # Convergence detection
    # ============================================================

    def _detect_convergence(self):
        """
        Convergence logic:
        1) If rmsd_thresh is set: first index where raw RMSD < rmsd_thresh.
        2) Else: use slope of fitted RMSD (optionally combined with sampling).
        """
        # 1) Fixed RMSD threshold (highest priority)
        if self.rmsd_thresh is not None:
            for idx, val in enumerate(self.rmsd_raw):
                if val < self.rmsd_thresh:
                    return self.t[idx]
            return None

        # 2) If exponential fit failed
        if np.isnan(self.rmsd_fit).all():
            return None

        slope = np.gradient(self.rmsd_fit, self.t)

        # 3) Slope-only convergence
        if self.count_std_thresh is None:
            for idx, s in enumerate(slope):
                if abs(s) < self.slope_thresh:
                    return self.t[idx]
            return None

        # 4) Slope + sampling convergence
        for idx, s in enumerate(slope):
            if abs(s) < self.slope_thresh:
                window = self.normed_counts[idx: idx + self.n_recent]
                mean_std = np.mean([np.std(w) for w in window])
                if mean_std < self.count_std_thresh:
                    return self.t[idx]

        return None

    # ============================================================
    # Feature detection (N-D)
    # ============================================================

    def detect_features(self, window=3, grad_thresh=0.1):
        """
        Locate minima, maxima, and plateau regions in the FINAL PMF.
        Works for arbitrary N-D PMFs.
        """
        pmf_grid = self.pmf_values[-1]
        coords = self.pmf_coords

        local_min = (pmf_grid == minimum_filter(pmf_grid, size=window))
        local_max = (pmf_grid == maximum_filter(pmf_grid, size=window))

        minima_idx = np.argwhere(local_min)
        maxima_idx = np.argwhere(local_max)

        grads = np.gradient(pmf_grid, *coords)
        grad_mag = np.sqrt(sum(g ** 2 for g in grads))
        plateau_idx = np.argwhere(grad_mag < grad_thresh)

        self.features = {
            "minima": minima_idx,
            "maxima": maxima_idx,
            "plateaus": plateau_idx
        }

    def annotate_comparison(self, ax, fs=14):
        """
        Annotate the comparison subplot with font-size fs for all labels/arrows.
        For N-D PMFs, plots using the first two dimensions.
        """
        self.detect_features()

        pmf_grid = self.pmf_values[-1]
        coords = self.pmf_coords

        if pmf_grid.ndim == 1:
            x = coords[0]
            y = pmf_grid

            for idx in self.features['minima']:
                xm, ym = x[idx[0]], y[idx[0]]
                ax.plot(xm, ym, 'bo')
                ax.text(xm, ym - 0.5, f"Min\n{xm:.2f}", fontsize=fs, ha='center', color='blue')

            for idx in self.features['maxima']:
                xM, yM = x[idx[0]], y[idx[0]]
                ax.plot(xM, yM, 'ro')
                ax.text(xM, yM + 0.5, f"Max\n{xM:.2f}", fontsize=fs, ha='center', color='red')

            pts = self.features['plateaus']
            if len(pts) > 0:
                sample = pts[::max(1, len(pts) // 5)]
                for idx in sample:
                    xf, yf = x[idx[0]], y[idx[0]]
                    ax.plot(xf, yf, 'go', alpha=0.3)
                    ax.text(xf, yf + 0.5, f"{yf:.2f}", fontsize=fs, ha='center', color='green')

        else:
            X, Y = np.meshgrid(coords[0], coords[1], indexing='ij')
            ax.contourf(X, Y, pmf_grid, levels=30, cmap='viridis')
            ax.set_xlabel('Coord 1')
            ax.set_ylabel('Coord 2')

            for idx in self.features['minima']:
                xm, ym = coords[0][idx[0]], coords[1][idx[1]]
                ax.plot(xm, ym, 'bo')
                ax.text(xm, ym, "Min", fontsize=fs, ha='center', color='blue')

            for idx in self.features['maxima']:
                xM, yM = coords[0][idx[0]], coords[1][idx[1]]
                ax.plot(xM, yM, 'ro')
                ax.text(xM, yM, "Max", fontsize=fs, ha='center', color='red')

            pts = self.features['plateaus']
            if len(pts) > 0:
                sample = pts[::max(1, len(pts) // 20)]
                for idx in sample:
                    xf, yf = coords[0][idx[0]], coords[1][idx[1]]
                    ax.plot(xf, yf, 'go', alpha=0.3)

    # ============================================================
    # Plotting
    # ============================================================

    def plot(self,
             font_size=14,
             annotation_fs=None,
             show_annotations=True,
             save_path=None,
             dpi=300):

        if annotation_fs is None:
            annotation_fs = font_size

        n = len(self.pmf_values)
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.8])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        axc = fig.add_subplot(gs[1, :])

        # --- RMSD convergence ---
        ax0.plot(self.t, self.rmsd_raw, color='gray', alpha=0.6, label='Raw RMSD')
        ax0.plot(self.t, self.rmsd_smooth, color='blue', label='Smoothed')
        if not np.isnan(self.rmsd_fit).all():
            ax0.plot(self.t, self.rmsd_fit, '--', color='red', label='Fit')
        if self.convergence_idx is not None:
            ax0.axvline(self.convergence_idx, color='orange', linestyle='--', label='Converged')
            ax0.axvspan(self.convergence_idx, self.t[-1], color='orange', alpha=0.2)

        ax0.set_title('PMF Convergence', fontsize=font_size)
        ax0.set_xlabel('PMF Snapshot Index', fontsize=font_size)
        ax0.set_ylabel('RMSD', fontsize=font_size)
        ax0.tick_params(labelsize=font_size)
        ax0.legend(fontsize=font_size)
        ax0.grid(True)

        # --- Sequential PMFs ---
        if self.pmf_values[0].ndim == 1:
            x = self.pmf_coords[0]
            for i, pmf in enumerate(self.pmf_values):
                color = 'black' if i == n - 1 else str(0.3 + 0.7 * i / (n - 1))
                lw = 2 if i == n - 1 else 1
                ax1.plot(x, pmf, color=color, linewidth=lw)
            ax1.set_xlabel(r'$\xi$', fontsize=font_size)
            ax1.set_ylabel('PMF', fontsize=font_size)
        else:
            X, Y = np.meshgrid(self.pmf_coords[0], self.pmf_coords[1], indexing='ij')
            for i, pmf in enumerate(self.pmf_values):
                color = 'black' if i == n - 1 else str(0.3 + 0.7 * i / (n - 1))
                ax1.contour(X, Y, pmf, levels=20, colors=[color], linewidths=1)
            ax1.set_xlabel('Coord 1', fontsize=font_size)
            ax1.set_ylabel('Coord 2', fontsize=font_size)

        ax1.set_title('Sequential PMFs', fontsize=font_size)
        ax1.tick_params(labelsize=font_size)
        ax1.grid(True)

        # --- Sampling density evolution ---
        if self.counts[0].ndim == 1:
            x = self.count_coords[0]
            for i, cnt in enumerate(self.normed_counts):
                color = 'black' if i == n - 1 else str(0.3 + 0.7 * i / (n - 1))
                lw = 2 if i == n - 1 else 1
                ax2.plot(x, cnt, color=color, linewidth=lw)
        else:
            X, Y = np.meshgrid(self.count_coords[0], self.count_coords[1], indexing='ij')
            for i, cnt in enumerate(self.normed_counts):
                color = 'black' if i == n - 1 else str(0.3 + 0.7 * i / (n - 1))
                ax2.contour(X, Y, cnt, levels=20, colors=[color], linewidths=1)

        ax2.set_title('Sampling Evolution', fontsize=font_size)
        ax2.set_xlabel(r'$\xi$', fontsize=font_size)
        ax2.set_ylabel('Normalized Count', fontsize=font_size)
        ax2.tick_params(labelsize=font_size)
        ax2.grid(True)

        # --- Post-convergence vs final PMF ---
        if self.convergence_idx is not None:
            idx = next((i for i, t in enumerate(self.t) if t >= self.convergence_idx), None)
            if idx is not None:
                if self.use_sliding_window:
                    pmf_post = self.pmf_values[self.n_recent + idx]
                else:
                    pmf_post = self.pmf_values[idx]
                pmf_final = self.pmf_values[-1]

                if pmf_final.ndim == 1:
                    x = self.pmf_coords[0]
                    axc.plot(x, pmf_post, color='blue', linewidth=2, label='Post-Conv')
                    axc.plot(x, pmf_final, '--', color='black', linewidth=2, label='Final')
                    axc.set_xlabel(r'$\xi$', fontsize=font_size)
                    axc.set_ylabel('PMF', fontsize=font_size)
                else:
                    X, Y = np.meshgrid(self.pmf_coords[0], self.pmf_coords[1], indexing='ij')
                    axc.contourf(X, Y, pmf_final, levels=30, cmap='viridis')
                    axc.contour(X, Y, pmf_post, levels=30, colors='blue', linewidths=1)
                    axc.set_xlabel('Coord 1', fontsize=font_size)
                    axc.set_ylabel('Coord 2', fontsize=font_size)

                axc.set_title('Post-Convergence vs Final', fontsize=font_size)
                axc.tick_params(labelsize=font_size)
                axc.legend(fontsize=font_size)
                axc.grid(True)

                if show_annotations:
                    self.annotate_comparison(axc, fs=annotation_fs)

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

        plt.show()


# ---- CLI ----
def main():
    parser = argparse.ArgumentParser(description='Generate PMF convergence & comparison plots')

    parser.add_argument('pmf_file', help='Path to PMF history or single PMF file')
    parser.add_argument('counts_file', help='Path to sampling counts history file')

    parser.add_argument('--font-size', type=int, default=14)
    parser.add_argument('--annotation-fs', type=int, default=None)
    parser.add_argument('--no-annotations', dest='show_annotations', action='store_false')

    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--dpi', type=int, default=300)

    parser.add_argument('--conv-threshold', type=float, default=0.01)
    parser.add_argument('--rmsd-threshold', type=float, default=None)
    parser.add_argument('--n-recent', type=int, default=10)
    parser.add_argument('--use-sliding-window', action='store_true')
    parser.add_argument('--counts-std-thresh', type=float, default=None)
    parser.add_argument('--reference-pmf', type=str, default=None)

    args = parser.parse_args()

    analyzer = PMFAnalyzer(
        args.pmf_file,
        args.counts_file,
        slope_thresh=args.conv_threshold,
        n_recent=args.n_recent,
        use_sliding_window=args.use_sliding_window,
        count_std_thresh=args.counts_std_thresh,
        reference_pmf_file=args.reference_pmf,
        rmsd_thresh=args.rmsd_threshold
    )

    analyzer.plot(
        font_size=args.font_size,
        annotation_fs=args.annotation_fs,
        show_annotations=args.show_annotations,
        save_path=args.save_path,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()