#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import argparse

# Configure global Matplotlib defaults for LaTeX serif rendering
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

class PMFAnalyzer:
    """
    Analyze a sequence of PMF snapshots and sampling counts, detect convergence,
    and annotate key free‐energy features (minima, maxima, plateaus, barriers).
    """

    def __init__(self, pmf_file, count_file, n_recent=4, slope_thresh=1e-3, use_final_rmsd=False):
        """
        pmf_file: path to a time‐series PMF file (blocks separated by '#')
        count_file: analogous file of sampling counts
        n_recent: number of recent PMFs to include in RMSD reference window
        slope_thresh: slope threshold to declare convergence of RMSD fit
        """
        self.pmf_file     = pmf_file
        self.count_file   = count_file
        self.n_recent     = n_recent
        self.use_final_rmsd = use_final_rmsd
        self.slope_thresh = slope_thresh

        # Read in sequential PMFs and counts
        self.pmfs,         = [self._read_sequential_pmfs()]  # list of (coords, pmf) arrays
        self.counts, self.count_coords = self._read_sequential_counts()

        # Extract coordinate grid and PMF values
        self.pmf_coords = self.pmfs[0][0]
        self.pmf_values = [block[1] for block in self.pmfs]

        # Normalize counts for plotting
        self.normed_counts = self._normalize_counts(self.counts)

        # Compute RMSD convergence diagnostics
        if self.use_final_rmsd:
        # RMSD of each snapshot to the final PMF
            final = self.pmf_values[-1]
            self.rmsd_raw = np.array([
                np.sqrt(np.mean((pmf - final) ** 2))
                for pmf in self.pmf_values])
            # time axis spans all snapshots
            self.t = np.arange(len(self.rmsd_raw))
        else:
            # original “to‐recent‐average” RMSD
            self.rmsd_raw = self._compute_rmsd_to_recent()
            self.t = np.arange(self.n_recent, len(self.pmf_values))
        # smooth & (optional) fit
        self.rmsd_smooth = self._smooth_rmsd(self.rmsd_raw)
        self.params, self.rmsd_fit = self._fit_exp_decay()

        self.convergence_idx = self._detect_convergence()

        # below = np.where(self.rmsd_smooth <= self.slope_thresh)[0]
        # self.convergence_idx = (self.t[below[0]] if below.size else None)

    def _read_sequential_pmfs(self):
        """Parse PMF file into a list of (coords, pmf) arrays."""
        pmfs = []
        with open(self.pmf_file, 'r') as f:
            tmp = []
            for line in f:
                if line.startswith('#'):
                    if tmp:
                        pmfs.append(np.array(tmp, float).T)
                        tmp = []
                    continue
                if line.strip():
                    tmp.append(line.split())
        return pmfs

    def _read_sequential_counts(self):
        """Parse count file into a list of count arrays and a shared coordinate grid."""
        counts = []
        with open(self.count_file, 'r') as f:
            tmp = []
            for line in f:
                if line.startswith('#'):
                    if tmp:
                        block = np.array(tmp, float).T
                        counts.append(block[1])
                        tmp = []
                    continue
                if line.strip():
                    tmp.append(line.split())
        coords = block[0]
        return counts, coords

    def _normalize_counts(self, counts):
        """Scale each count array to [0,1] for plotting."""
        normed = []
        c_min, c_max = np.min(counts), np.max(counts)
        for c in counts:
            normed.append((c - c_min) / (c_max - c_min + 1e-12))
        return normed

    def _compute_rmsd_to_recent(self):
        """Compute RMSD of each PMF to the average of the previous n_recent PMFs."""
        rmsd = []
        for i in range(self.n_recent, len(self.pmf_values)):
            ref = np.mean(self.pmf_values[i-self.n_recent:i], axis=0)
            rmsd_val = np.sqrt(np.mean((self.pmf_values[i] - ref)**2))
            rmsd.append(rmsd_val)
        return np.array(rmsd)

    def _smooth_rmsd(self, rmsd, window_length=11, polyorder=3):
        """Apply Savitzky-Golay filter to smooth the RMSD curve."""
        wl = min(window_length, len(rmsd) if len(rmsd)%2 else len(rmsd)-1)
        return savgol_filter(rmsd, window_length=wl, polyorder=polyorder)

    def _exp_decay(self, t, A, B, C):
        """Exponential decay model A * exp(-B t) + C."""
        return A * np.exp(-B*t) + C

    def _fit_exp_decay(self):
        """Fit the smoothed RMSD to an exponential decay to find convergence behavior."""
        try:
            params, _ = curve_fit(self._exp_decay, self.t, self.rmsd_smooth,
                                  p0=(1,0.1,0.01), maxfev=10000)
            fit_curve = self._exp_decay(self.t, *params)
            return params, fit_curve
        except RuntimeError:
            print("Warning: Exponential fit did not converge.")
            return None, np.full_like(self.t, np.nan)

    def _detect_convergence(self):
        """
        Determine the first index where the slope of the fitted RMSD falls below
        slope_thresh, marking convergence.
        """
        if np.isnan(self.rmsd_fit).all():
            return None
        slope = np.gradient(self.rmsd_fit, self.t)
        for idx, s in enumerate(slope):
            if abs(s) < self.slope_thresh:
                return self.t[idx]
        return None

    def detect_features(self, window=5, grad_thresh=0.1):
        """
        Locate minima, maxima, and plateau regions in the FINAL PMF.
          window: look-back window for argrelextrema
          grad_thresh: threshold on |dF/dx| to call a plateau
        """
        y = self.pmf_values[-1]
        x = self.pmf_coords

        # 1) find flat regions (plateaus) via small gradient
        grad = np.gradient(y, x)
        plateau_idx = np.where(np.abs(grad) < grad_thresh)[0]

        # 2) find all local minima / maxima
        raw_min = argrelextrema(y, np.less_equal,    order=window)[0]
        raw_max = argrelextrema(y, np.greater_equal, order=window)[0]

        # 3) exclude any extremum lying in a plateau region
        filtered_min = [x[i] for i in raw_min if i not in plateau_idx]
        filtered_max = [x[i] for i in raw_max if i not in plateau_idx]

        self.features = {
            'minima' : np.array(filtered_min),
            'maxima' : np.array(filtered_max),
            'plateaus': x[plateau_idx]
        }

    def compute_barriers(self, min_cutoff=0.5):
        """
        For each detected maximum, find the closest minimum and compute ΔE.
        Only barriers with ΔE >= min_cutoff are returned.
        """
        barriers = {}
        x = self.pmf_coords
        y = self.pmf_values[-1]

        for i, x_max in enumerate(self.features.get('maxima', []), start=1):
            # pair with nearest minimum
            x_min = self.features['minima'][np.argmin(np.abs(self.features['minima'] - x_max))]
            ΔE = y[np.argmin(np.abs(x - x_max))] - y[np.argmin(np.abs(x - x_min))]
            if ΔE >= min_cutoff:
                barriers[f'Barrier_{i}'] = {'from': x_min, 'to': x_max, 'dE': ΔE}

        return barriers

    def annotate_comparison(self, ax, min_cutoff=0.5, fs=14):
        """
        Annotate the comparison subplot with font‐size fs for all labels/arrows.
        """
        self.detect_features()
        barriers = self.compute_barriers(min_cutoff=min_cutoff)
        x, y    = self.pmf_coords, self.pmf_values[-1]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        def clamp(val, vmin, vmax):
            pad = 0.05 * (vmax - vmin)
            return max(vmin+pad, min(val, vmax-pad))

        def safe_text(xp, yp, txt, dx=0.0, dy=0.0, color='black'):
            xt = clamp(xp+dx, *xlim)
            yt = clamp(yp+dy, *ylim)
            ax.text(xt, yt, txt, fontsize=fs, ha='center', color=color)

        # Minima (blue)
        for xm in self.features['minima']:
            ym = y[np.argmin(np.abs(x-xm))]
            ax.plot(xm, ym, 'bo')
            safe_text(xm, ym, f"Min\n{xm:.2f}", dy=-1.0, color='blue')

        # Maxima (red)
        for xM in self.features['maxima']:
            yM = y[np.argmin(np.abs(x-xM))]
            ax.plot(xM, yM, 'ro')
            safe_text(xM, yM, f"Max\n{xM:.2f}", dy=+1.0, color='red')

        # Plateaus (green)
        pts = self.features['plateaus']
        sample = pts[::max(1,len(pts)//5)]
        for xf in sample:
            yf = y[np.argmin(np.abs(x-xf))]
            ax.plot(xf, yf, 'go', alpha=0.3)
            safe_text(xf, yf, f"{yf:.2f}", dy=+0.5, color='green')

        # Barriers (grey arrows)
        for lbl, dat in barriers.items():
            xt, yt = dat['to'], y[np.argmin(np.abs(x-dat['to']))]
            xa = clamp(xt+0.5, *xlim)
            ya = clamp(yt+1.0, *ylim)
            ax.annotate(f"{lbl}\ndE={dat['dE']:.2f}",
                        xy=(xt, yt), xytext=(xa, ya),
                        arrowprops=dict(arrowstyle="->", color='gray'),
                        fontsize=fs, ha='center')


    def plot(self,
             font_size=14,
             annotation_fs=None,
             show_annotations=True,
             save_path=None,
             dpi=300):
        """
        Plot convergence, sequential PMFs, sampling density, and
        post-convergence vs final PMF, with optional annotations
        and file export.

        Parameters
        ----------
        font_size : int
            Font size (pt) for titles, axis labels, ticks, legends.
        annotation_fs : int or None
            Font size (pt) for annotation text. If None, uses font_size.
        show_annotations : bool
            Whether to draw minima/maxima/plateau/barrier annotations.
        save_path : str or None
            File path (with extension) to save the figure. If None, skips saving.
        dpi : int
            Resolution in dots per inch for saved figure.
        """
        if annotation_fs is None:
            annotation_fs = font_size

        n = len(self.pmf_values)
        fig = plt.figure(figsize=(18, 8))
        gs  = gridspec.GridSpec(2, 3, height_ratios=[1, 0.8])
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[0,1])
        ax2 = fig.add_subplot(gs[0,2])
        axc = fig.add_subplot(gs[1,:])

        # --- RMSD convergence ---
        ax0.plot(self.t, self.rmsd_raw,   color='gray', alpha=0.6, label='Raw RMSD')
        ax0.plot(self.t, self.rmsd_smooth, color='blue',                    label='Smoothed')
        if not np.isnan(self.rmsd_fit).all():
            ax0.plot(self.t, self.rmsd_fit, '--', color='red',            label='Fit')
        if self.convergence_idx is not None:
            ax0.axvline(self.convergence_idx, color='orange', linestyle='--', label='Converged')
            ax0.axvspan(self.convergence_idx, self.t[-1], color='orange', alpha=0.2)

        ax0.set_title('PMF Convergence',       fontsize=font_size)
        ax0.set_xlabel('PMF Snapshot Index',   fontsize=font_size)
        ax0.set_ylabel('RMSD [Å]',              fontsize=font_size)
        ax0.tick_params(labelsize=font_size)
        ax0.legend(fontsize=font_size)
        ax0.grid(True)

        # --- Sequential PMFs (grayscale shading) ---
        for i, pmf in enumerate(self.pmf_values):
            color = 'black' if i == n-1 else str(0.3 + 0.7 * i/(n-1))
            lw    = 2 if i == n-1 else 1
            ax1.plot(self.pmf_coords, pmf, color=color, linewidth=lw)
        ax1.set_title('Sequential PMFs',       fontsize=font_size)
        ax1.set_xlabel(r'$\xi$',                fontsize=font_size)
        ax1.set_ylabel('PMF [kcal/mol]',        fontsize=font_size)
        ax1.tick_params(labelsize=font_size)
        ax1.grid(True)

        # --- Sampling density evolution ---
        for i, cnt in enumerate(self.normed_counts):
            color = 'black' if i == n-1 else str(0.3 + 0.7 * i/(n-1))
            lw    = 2 if i == n-1 else 1
            ax2.plot(self.count_coords, cnt, color=color, linewidth=lw)
        ax2.set_title('Sampling Evolution',     fontsize=font_size)
        ax2.set_xlabel(r'$\xi$',                fontsize=font_size)
        ax2.set_ylabel('Normalized Count',      fontsize=font_size)
        ax2.tick_params(labelsize=font_size)
        ax2.grid(True)

        # --- Post-convergence vs final PMF + optional annotations ---
        if self.convergence_idx is not None:
            idx = next((i for i, t in enumerate(self.t) if t >= self.convergence_idx), None)
            if idx is not None:
                pmf_post  = self.pmf_values[self.n_recent + idx]
                pmf_final = self.pmf_values[-1]
                axc.plot(self.pmf_coords, pmf_post,  color='blue',  linewidth=2, label='Post-Conv')
                axc.plot(self.pmf_coords, pmf_final, '--', color='black', linewidth=2, label='Final')

                axc.set_title('Post-Convergence vs Final', fontsize=font_size)
                axc.set_xlabel(r'$\xi$',                fontsize=font_size)
                axc.set_ylabel('PMF [kcal/mol]',        fontsize=font_size)
                axc.tick_params(labelsize=font_size)
                axc.legend(fontsize=font_size)
                axc.grid(True)

                if show_annotations:
                    self.annotate_comparison(axc, min_cutoff=1.0, fs=annotation_fs)

        plt.tight_layout()

        # Save to file if requested
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

        plt.show()

# ---- Command-line interface ----
def main():
    parser = argparse.ArgumentParser(description='Generate PMF convergence & comparison plots')

    # positional
    parser.add_argument('pmf_file',
                        help='Path to PMF data file (e.g., pmf.txt)')
    parser.add_argument('counts_file',
                        help='Path to sampling counts file (e.g., cnts.txt)')

    # font & annotations
    parser.add_argument('--font-size',
                        type=int, default=14,
                        help='Base font size (pt)')
    parser.add_argument('--annotation-fs',
                        type=int, default=None,
                        help='Font size for annotation text (pt)')
    parser.add_argument('--no-annotations',
                        dest='show_annotations',
                        action='store_false',
                        help='Disable barrier/minima annotations')

    # saving
    parser.add_argument('--save-path',
                        type=str, default=None,
                        help='File path to save figure (png, pdf)')
    parser.add_argument('--dpi',
                        type=int, default=300,
                        help='DPI for saved file')

    # new: convergence threshold & n_recent
    parser.add_argument('--conv-threshold',
                        type=float, default=0.01,
                        help='RMSD cutoff for convergence detection')
    parser.add_argument('--n-recent',
                        type=int, default=10,
                        help='Snapshots after convergence to compare')
    parser.add_argument('--use-final-rmsd',
                        action = 'store_true',
                        help = 'Determine convergence by RMSD to final PMF')


    args = parser.parse_args()

    analyzer = PMFAnalyzer(
        args.pmf_file,
        args.counts_file,
        slope_thresh=args.conv_threshold,
        n_recent=args.n_recent,
        use_final_rmsd=args.use_final_rmsd

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

