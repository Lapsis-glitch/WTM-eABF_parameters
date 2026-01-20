import argparse

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.ndimage import minimum_filter, maximum_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PMFAnalyzer:
    """
    Analyze a sequence of PMF snapshots and sampling counts, detect convergence,
    and annotate key free-energy features (minima, maxima, plateaus).
    """

    def __init__(self, pmf_file, count_file, n_recent=4,
                 slope_thresh=1e-3, use_final_rmsd=False,
                 count_std_thresh: int = None):
        """
        pmf_file: path to a time-series PMF file (blocks separated by '#')
        count_file: analogous file of sampling counts
        n_recent: number of recent PMFs to include in RMSD reference window
        slope_thresh: slope threshold to declare convergence of RMSD fit
        count_std_thresh: count standard deviation threshold to declare convergence of sampling
        """
        self.pmf_file     = pmf_file
        self.count_file   = count_file
        self.n_recent     = n_recent
        self.use_final_rmsd = use_final_rmsd
        self.slope_thresh = slope_thresh
        self.count_std_thresh = count_std_thresh

        # Read in sequential PMFs and counts
        self.pmfs = self._read_sequential_pmfs()   # list of (coords_tuple, pmf_ndarray)
        self.counts, self.count_coords = self._read_sequential_counts()

        # Extract coordinate grid and PMF values
        self.pmf_coords = self.pmfs[0][0]          # tuple of coordinate arrays
        self.pmf_values = [block[1] for block in self.pmfs]

        # Normalize counts for plotting
        self.normed_counts = self._normalize_counts(self.counts)

        # Compute RMSD convergence diagnostics
        if self.use_final_rmsd:
            final = self.pmf_values[-1]
            self.rmsd_raw = np.array([
                np.sqrt(np.mean((pmf - final) ** 2))
                for pmf in self.pmf_values])
            self.t = np.arange(len(self.rmsd_raw))
        else:
            self.rmsd_raw = self._compute_rmsd_to_recent()
            self.t = np.arange(self.n_recent, len(self.pmf_values))

        # smooth & fit
        self.rmsd_smooth = self._smooth_rmsd(self.rmsd_raw)
        self.params, self.rmsd_fit = self._fit_exp_decay()

        self.convergence_idx = self._detect_convergence()

    # ----------------------------
    # File parsing
    # ----------------------------
    def _read_sequential_pmfs(self):
        """
        Parse PMF file into a list of (coords_tuple, pmf_ndarray).
        Assumes each block is a flattened grid with columns: coord1 coord2 ... coordN value
        """
        pmfs = []
        with open(self.pmf_file, 'r') as f:
            tmp = []
            for line in f:
                if line.startswith('#'):
                    if tmp:
                        arr = np.array(tmp, float)
                        coords = [np.unique(arr[:, i]) for i in range(arr.shape[1]-1)]
                        shape = tuple(len(c) for c in coords)
                        pmf = arr[:, -1].reshape(shape)
                        pmfs.append((tuple(coords), pmf))
                        tmp = []
                    continue
                if line.strip():
                    tmp.append(line.split())
        return pmfs

    def _read_sequential_counts(self):
        """
        Parse count file into a list of count arrays and a shared coordinate grid.
        Assumes same format as PMF file.
        """
        counts = []
        coords = None
        with open(self.count_file, 'r') as f:
            tmp = []
            for line in f:
                if line.startswith('#'):
                    if tmp:
                        arr = np.array(tmp, float)
                        coords = [np.unique(arr[:, i]) for i in range(arr.shape[1]-1)]
                        shape = tuple(len(c) for c in coords)
                        counts.append(arr[:, -1].reshape(shape))
                        tmp = []
                    continue
                if line.strip():
                    tmp.append(line.split())
        return counts, coords

    # ----------------------------
    # Normalization
    # ----------------------------
    def _normalize_counts(self, counts):
        """Scale each count array to [0,1] for plotting."""
        normed = []
        c_min = min(np.min(c) for c in counts)
        c_max = max(np.max(c) for c in counts)
        for c in counts:
            normed.append((c - c_min) / (c_max - c_min + 1e-12))
        return normed

    # ----------------------------
    # RMSD Convergence
    # ----------------------------
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
        if len(rmsd) < 3:
            return rmsd
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
        """Determine the first index where the slope of the fitted RMSD falls below slope_thresh."""
        if np.isnan(self.rmsd_fit).all():
            return None
        slope = np.gradient(self.rmsd_fit, self.t)
        if self.count_file is None or self.count_std_thresh is None:
            for idx, s in enumerate(slope):
                if abs(s) < self.slope_thresh:
                    return self.t[idx]
            return None
        else:
            for idx, s in enumerate(slope):
                if abs(s) < self.slope_thresh:
                    window = self.normed_counts[idx: idx + self.n_recent]
                    mean_std = np.mean([np.std(w) for w in window])
                    if mean_std < self.count_std_thresh:
                        return self.t[idx]
            return None

    # ----------------------------
    # Feature Detection (N-D)
    # ----------------------------
    def detect_features(self, window=3, grad_thresh=0.1):
        """
        Locate minima, maxima, and plateau regions in the FINAL PMF.
        Works for arbitrary N-D PMFs.
        """
        pmf_grid = self.pmf_values[-1]   # N-D array
        coords   = self.pmf_coords       # tuple of coordinate arrays

        # 1) Local minima / maxima via ndimage filters
        local_min = (pmf_grid == minimum_filter(pmf_grid, size=window))
        local_max = (pmf_grid == maximum_filter(pmf_grid, size=window))

        minima_idx = np.argwhere(local_min)
        maxima_idx = np.argwhere(local_max)

        # 2) Plateaus via gradient magnitude
        grads = np.gradient(pmf_grid, *coords)   # returns list of N gradients
        grad_mag = np.sqrt(sum(g**2 for g in grads))
        plateau_idx = np.argwhere(grad_mag < grad_thresh)

        # 3) Store features as index arrays
        self.features = {
            'minima': minima_idx,
            'maxima': maxima_idx,
            'plateaus': plateau_idx
        }

    # ----------------------------
    # Utility: map indices to coordinates
    # ----------------------------
    def indices_to_coords(self, indices):
        """
        Convert index array (like from np.argwhere) into physical coordinates.
        """
        coords_list = []
        for idx in indices:
            coords_list.append(tuple(c[i] for c, i in zip(self.pmf_coords, idx)))
        return np.array(coords_list)

    def annotate_comparison(self, ax, fs=14):
        """
        Annotate the comparison subplot with font-size fs for all labels/arrows.
        For N-D PMFs, plots using the first two dimensions.
        """
        self.detect_features()

        pmf_grid = self.pmf_values[-1]
        coords = self.pmf_coords

        # --- 1D case: line plot with annotations ---
        if pmf_grid.ndim == 1:
            x = coords[0]
            y = pmf_grid

            # Minima (blue)
            for idx in self.features['minima']:
                xm, ym = x[idx[0]], y[idx[0]]
                ax.plot(xm, ym, 'bo')
                ax.text(xm, ym - 0.5, f"Min\n{xm:.2f}", fontsize=fs, ha='center', color='blue')

            # Maxima (red)
            for idx in self.features['maxima']:
                xM, yM = x[idx[0]], y[idx[0]]
                ax.plot(xM, yM, 'ro')
                ax.text(xM, yM + 0.5, f"Max\n{xM:.2f}", fontsize=fs, ha='center', color='red')

            # Plateaus (green, sampled)
            pts = self.features['plateaus']
            sample = pts[::max(1, len(pts) // 5)]
            for idx in sample:
                xf, yf = x[idx[0]], y[idx[0]]
                ax.plot(xf, yf, 'go', alpha=0.3)
                ax.text(xf, yf + 0.5, f"{yf:.2f}", fontsize=fs, ha='center', color='green')

        # --- 2D or higher: contour plot with annotations on first two dims ---
        else:
            X, Y = np.meshgrid(coords[0], coords[1], indexing='ij')
            ax.contourf(X, Y, pmf_grid, levels=30, cmap='viridis')
            ax.set_xlabel('Coord 1')
            ax.set_ylabel('Coord 2')

            # Minima (blue)
            for idx in self.features['minima']:
                xm, ym = coords[0][idx[0]], coords[1][idx[1]]
                ax.plot(xm, ym, 'bo')
                ax.text(xm, ym, "Min", fontsize=fs, ha='center', color='blue')

            # Maxima (red)
            for idx in self.features['maxima']:
                xM, yM = coords[0][idx[0]], coords[1][idx[1]]
                ax.plot(xM, yM, 'ro')
                ax.text(xM, yM, "Max", fontsize=fs, ha='center', color='red')

            # Plateaus (green, sampled)
            pts = self.features['plateaus']
            sample = pts[::max(1, len(pts) // 20)]
            for idx in sample:
                xf, yf = coords[0][idx[0]], coords[1][idx[1]]
                ax.plot(xf, yf, 'go', alpha=0.3)

    def plot(self,
             font_size=14,
             annotation_fs=None,
             show_annotations=True,
             save_path=None,
             dpi=300):
        """
        Plot convergence, sequential PMFs, sampling density, and
        post-convergence vs final PMF, with optional annotations.
        For N-D PMFs, sequential plots use first dimension (line plots),
        and comparison uses first two dimensions (contour).
        """
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
        ax0.set_ylabel('RMSD [kcal/mol]', fontsize=font_size)
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
            ax1.set_ylabel('PMF [kcal/mol]', fontsize=font_size)
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

        # --- Post-convergence vs final PMF + optional annotations ---
        if self.convergence_idx is not None:
            idx = next((i for i, t in enumerate(self.t) if t >= self.convergence_idx), None)
            if idx is not None:
                pmf_post = self.pmf_values[self.n_recent + idx]
                pmf_final = self.pmf_values[-1]

                if pmf_final.ndim == 1:
                    x = self.pmf_coords[0]
                    axc.plot(x, pmf_post, color='blue', linewidth=2, label='Post-Conv')
                    axc.plot(x, pmf_final, '--', color='black', linewidth=2, label='Final')
                    axc.set_xlabel(r'$\xi$', fontsize=font_size)
                    axc.set_ylabel('PMF [kcal/mol]', fontsize=font_size)
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
                        action='store_true',
                        help='Determine convergence by RMSD to final PMF')
    parser.add_argument('--counts-std-thresh',
                        type=float, default=None,
                        help='Determine convergence by std of the sampling')

    args = parser.parse_args()

    analyzer = PMFAnalyzer(
        args.pmf_file,
        args.counts_file,
        slope_thresh=args.conv_threshold,
        n_recent=args.n_recent,
        use_final_rmsd=args.use_final_rmsd,
        count_std_thresh=args.counts_std_thresh

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
