"""
ClayPMFPlotter.py

Separate plotting class for ClayPMF results.
Handles all visualization and plotting functionality for CIP-clay PMF analysis.

Author: R.Swai
Date: April 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, TwoSlopeNorm
import os
from datetime import datetime


def _savgol(y, window, order):
    """
    Apply a Savitzky-Golay filter to *y*.

    Parameters
    ----------
    y : array-like
    window : int
        Filter window length (in samples).  Must be odd and > order;
        automatically incremented by 1 if even.
    order : int
        Polynomial order (e.g. 3).

    Returns
    -------
    numpy.ndarray  (same shape as y, dtype float)
    """
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        import warnings
        warnings.warn(
            "scipy is not installed — smoothing skipped.  "
            "Install with: pip install scipy",
            stacklevel=3,
        )
        return np.asarray(y, dtype=float)

    y = np.asarray(y, dtype=float)
    w = int(window) if int(window) % 2 == 1 else int(window) + 1
    if len(y) < w:
        return y
    return savgol_filter(y, window_length=w, polyorder=int(order))


def _logsumexp(a, axis=None, keepdims=False):
    """Numerically stable log-sum-exp; NaN entries are ignored (treated as -inf).
    All-NaN slices return NaN (expected for unvisited bins; warnings suppressed).
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        a_max = np.nanmax(a, axis=axis, keepdims=True)
        out = np.log(np.nansum(np.exp(a - a_max), axis=axis, keepdims=True))
        out += a_max
    if not keepdims:
        out = out.squeeze(axis=axis)
    return out


class ClayPMFPlotter:
    """
    Plotting class for ClayPMF results.

    This class handles visualization of:
    - PMF(|z|): symmetrised adsorption free energy profile
    - PMF(z):   signed (full-range) free energy profile
    - Overview: two-panel combined view
    - Window sampling: frames per umbrella window
    - Window histograms: histogram overlap across the RC grid

    Parameters
    ----------
    pmf : ClayPMF
        Instance of ClayPMF with run_wham() already called.

    Examples
    --------
    >>> from ClayPMF import ClayPMF
    >>> from ClayPMFPlotter import ClayPMFPlotter
    >>>
    >>> pmf = ClayPMF(umbrella_dir='...', n_bins=1200, xi_max=2.5)
    >>> pmf.load_data().run_wham()
    >>> pmf.bootstrap_errors(200)
    >>>
    >>> plotter = ClayPMFPlotter(pmf)
    >>> plotter.plot_pmf_abs()
    >>> plotter.plot_overview(plot_gmx=True)
    """

    # ------------------------------------------------------------------
    # Default colour palette
    # ------------------------------------------------------------------
    COLORS = {
        'abs':    'mediumseagreen',
        'signed': 'steelblue',
        'gmx':    'tomato',
        'clay':   'saddlebrown',
    }

    def __init__(self, pmf=None, pmf2d=None, pmf3d=None, ensemble=None):
        """
        Initialise plotter.

        Parameters
        ----------
        pmf : ClayPMF, optional
            ClayPMF instance. Can be set later via ``plotter.pmf = pmf``.
        pmf2d : ClayPMF2D, optional
            ClayPMF2D instance. Can be set later via ``plotter.pmf2d = pmf2d``.
        pmf3d : ClayPMF3D, optional
            ClayPMF3D instance. Can be set later via ``plotter.pmf3d = pmf3d``.
        ensemble : ClayPMFNeuralEnsemble, optional
            Ensemble instance. Can be set later via ``plotter.ensemble = ensemble``.
        """
        self.pmf      = pmf
        self.pmf2d    = pmf2d
        self.pmf3d    = pmf3d
        self.ensemble = ensemble
        self.set_default_style()

        self.default_figsize = (8, 5)
        self.default_dpi     = 300

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def set_default_style(self):
        """Set default matplotlib rcParams for publication-quality plots."""
        plt.rcParams['font.size']        = 12
        plt.rcParams['axes.labelsize']   = 12
        plt.rcParams['axes.titlesize']   = 13
        plt.rcParams['xtick.labelsize']  = 11
        plt.rcParams['ytick.labelsize']  = 11
        plt.rcParams['legend.fontsize']  = 10
        plt.rcParams['figure.titlesize'] = 14
        plt.rcParams['lines.linewidth']  = 2.0
        plt.rcParams['axes.linewidth']   = 1.2

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_wham(self):
        if self.pmf is None or self.pmf.pmf_abs is None:
            raise RuntimeError(
                "ClayPMF instance with run_wham() completed is required. "
                "Set plotter.pmf = pmf after running WHAM."
            )

    def _require_wham_2d(self):
        if self.pmf2d is None or self.pmf2d.pmf_2d is None:
            raise RuntimeError(
                "ClayPMF2D instance with run_wham_2d() completed is required. "
                "Set plotter.pmf2d = pmf2d after running 2D WHAM."
            )

    def _require_wham_3d(self):
        if self.pmf3d is None or self.pmf3d.pmf_3d is None:
            raise RuntimeError(
                "ClayPMF3D instance with run_wham_3d() completed is required. "
                "Set plotter.pmf3d = pmf3d after running 3D WHAM."
            )

    def _require_ensemble(self):
        if self.ensemble is None or not self.ensemble.pmf3d_list:
            raise RuntimeError(
                "ClayPMFNeuralEnsemble instance with pmf3d_list populated is required. "
                "Set plotter.ensemble = ensemble after running WHAM on all replicates."
            )

    @staticmethod
    def _reref(arr, zero_at, r_arr=None):
        """Zero-reference a 1-D PMF array.

        pmf_abs layout: index 0 = pore centre (bulk), index N = near clay.
        So 'bulk' reference = first 20% of array (NOT last).
        """
        arr = arr.copy()
        if zero_at == 'min':
            arr -= np.nanmin(arr)
        elif zero_at == 'bulk':
            n_bulk = max(1, int(0.2 * len(arr)))
            arr -= float(np.nanmean(arr[:n_bulk]))   # first 20% = pore centre = bulk
        return arr

    @staticmethod
    def _reref_signed(arr, zero_at):
        """Zero-reference a signed PMF.

        pmf_signed layout: index 0 = z=-r_max (near clay), centre = z=0 (bulk),
        index -1 = z=+r_max (near clay).
        So 'bulk' reference = central 20% of array (z ≈ 0).
        """
        arr = arr.copy()
        if zero_at == 'min':
            arr -= np.nanmin(arr)
        elif zero_at == 'bulk':
            n      = len(arr)
            n_bulk = max(1, int(0.2 * n))
            mid    = n // 2
            lo     = max(0, mid - n_bulk // 2)
            hi     = min(n, mid + n_bulk // 2 + 1)
            arr   -= float(np.nanmean(arr[lo:hi]))   # centre 20% = z≈0 = bulk
        return arr

    # ------------------------------------------------------------------
    # 1. PMF(|z|) — single panel
    # ------------------------------------------------------------------

    def plot_pmf_abs(
        self,
        ax=None,
        unit='kJ/mol',
        zero_at='bulk',
        plot_gmx=False,
        gmx_label='gmx wham',
        color=None,
        color_gmx=None,
        xlim=None,
        ylim=None,
        title=None,
        smooth=False,
        smooth_window=21,
        smooth_order=3,
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        xlabel=None,
        ylabel=None,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        legend_fontsize=10,
        show_grid=True,
        grid_alpha=0.3,
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_abs.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Plot PMF(|z|) with optional bootstrap error band and gmx comparison.

        Parameters
        ----------
        ax : matplotlib Axes or None
        unit : str
            Energy unit label. Default 'kJ/mol'.
        zero_at : {'min', 'bulk'}
            Reference: 'min' = global minimum; 'bulk' = outer 20 % mean.
        plot_gmx : bool
            Overlay gmx wham reference if loaded. Default False.
        gmx_label : str
        color : str or None
            Line colour. Default COLORS['abs'].
        color_gmx : str or None
            gmx line colour. Default COLORS['gmx'].
        xlim, ylim : tuple or None
        title : str

        Returns
        -------
        fig, ax
        """
        self._require_wham()
        pmf = self.pmf

        if ax is None:
            fig, ax = plt.subplots(figsize=self.default_figsize)
        else:
            fig = ax.get_figure()

        color     = color     or self.COLORS['abs']
        color_gmx = color_gmx or self.COLORS['gmx']

        # x-axis: distance from clay surface (r = 0 at clay, r > 0 into solution).
        # bin_centers_abs = 0 → pore centre (bulk); bin_centers_abs = x_shift → clay.
        # Shift: r = x_shift - bin_centers_abs  (sign intentional — clay at zero).
        x_shift = float(getattr(pmf, 'z_clay_surface', None) or 0.0)

        r     = x_shift - pmf.bin_centers_abs   # 0 at clay, positive toward bulk
        pmf_a = self._reref(pmf.pmf_abs, zero_at)
        if smooth:
            pmf_a = _savgol(pmf_a, smooth_window, smooth_order)

        ax.plot(r, pmf_a, color=color, lw=2, label='Python WHAM (sym.)')

        if pmf.pmf_abs_std is not None:
            n   = min(len(r), len(pmf.pmf_abs_std))
            std = pmf.pmf_abs_std[:n]
            if smooth:
                std = _savgol(std, smooth_window, smooth_order)
            ax.fill_between(r[:n], pmf_a[:n] - std, pmf_a[:n] + std,
                            alpha=0.25, color=color, label='±1σ (bootstrap)')

        if plot_gmx and pmf.gmx_z is not None:
            # gmx wham outputs PMF already zeroed at bulk — do NOT re-reference.
            # Take positive-z half (sym.) and apply the same coordinate shift.
            mask = pmf.gmx_z >= 0
            gz   = x_shift - pmf.gmx_z[mask]    # same shift: r = x_shift - |z|
            gp   = pmf.gmx_pmf[mask].copy()
            ax.plot(gz, gp, color=color_gmx, lw=1.5, ls='--', label=gmx_label)
            if pmf.gmx_pmf_std is not None:
                gs = pmf.gmx_pmf_std[mask]
                ax.fill_between(gz, gp - gs, gp + gs,
                                alpha=0.15, color=color_gmx)

        _xlabel = xlabel if xlabel is not None else (
            'Distance from clay surface  (nm)' if x_shift > 0 else '|z| from pore centre  (nm)')
        _ylabel = ylabel if ylabel is not None else f'PMF  ({unit})'
        _title  = title if title is not None else 'PMF(|z|) — Symmetrised'
        ax.set_xlabel(_xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(_ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax.set_title(_title, fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.legend(fontsize=legend_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
        ax.set_xlim(left=0) if xlim is None else ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, ax

    # ------------------------------------------------------------------
    # 2. PMF(z) — single panel, signed
    # ------------------------------------------------------------------

    def plot_pmf_signed(
        self,
        ax=None,
        unit='kJ/mol',
        zero_at='bulk',
        plot_gmx=False,
        gmx_label='gmx wham',
        color=None,
        color_gmx=None,
        xlim=None,
        ylim=None,
        title=None,
        smooth=False,
        smooth_window=21,
        smooth_order=3,
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        xlabel=None,
        ylabel=None,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        legend_fontsize=10,
        show_grid=True,
        grid_alpha=0.3,
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_signed.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Plot the full signed PMF(z) profile.

        Parameters
        ----------
        ax : matplotlib Axes or None
        unit : str
        zero_at : {'min', 'bulk'}
        plot_gmx : bool
        gmx_label : str
        color : str or None
        color_gmx : str or None
        xlim, ylim : tuple or None
        title : str

        Returns
        -------
        fig, ax
        """
        self._require_wham()
        pmf = self.pmf

        if ax is None:
            fig, ax = plt.subplots(figsize=self.default_figsize)
        else:
            fig = ax.get_figure()

        color     = color     or self.COLORS['signed']
        color_gmx = color_gmx or self.COLORS['gmx']

        z     = getattr(pmf, '_bin_centers_signed', pmf.bin_centers)
        pmf_s = self._reref_signed(pmf.pmf_signed, zero_at)
        if smooth:
            pmf_s = _savgol(pmf_s, smooth_window, smooth_order)

        ax.plot(z, pmf_s, color=color, lw=2, label='Python WHAM')

        if pmf.pmf_signed_std is not None:
            std = pmf.pmf_signed_std
            if smooth:
                std = _savgol(std, smooth_window, smooth_order)
            ax.fill_between(
                z,
                pmf_s - std,
                pmf_s + std,
                alpha=0.25, color=color, label='±1σ (bootstrap)',
            )

        if plot_gmx and pmf.gmx_z is not None:
            # gmx wham outputs PMF already zeroed at bulk — do NOT re-reference.
            gp = pmf.gmx_pmf.copy()
            ax.plot(pmf.gmx_z, gp, color=color_gmx, lw=1.5, ls='--',
                    label=gmx_label)
            if pmf.gmx_pmf_std is not None:
                ax.fill_between(pmf.gmx_z, gp - pmf.gmx_pmf_std,
                                gp + pmf.gmx_pmf_std,
                                alpha=0.15, color=color_gmx)

        # Mark clay surface positions at ±z_surface (z=0 is pore centre = bulk)
        x_shift = float(getattr(pmf, 'z_clay_surface', None) or 0.0)
        if x_shift > 0:
            ax.axvline(+x_shift, color=self.COLORS['clay'], lw=1.2, ls='--',
                       label=f'Clay surface  (|z| = {x_shift:.2f} nm)')
            ax.axvline(-x_shift, color=self.COLORS['clay'], lw=1.2, ls='--')
        _xlabel = xlabel if xlabel is not None else 'z  (nm)'
        _ylabel = ylabel if ylabel is not None else f'PMF  ({unit})'
        _title  = title if title is not None else 'PMF(z) — Signed'
        ax.set_xlabel(_xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(_ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax.set_title(_title, fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.legend(fontsize=legend_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, ax

    # ------------------------------------------------------------------
    # 3. Overview — two panels side by side
    # ------------------------------------------------------------------

    def plot_overview(
        self,
        figsize=(13, 5),
        unit='kJ/mol',
        zero_at='bulk',
        plot_gmx=False,
        gmx_label='gmx wham',
        suptitle=None,
        smooth=False,
        smooth_window=21,
        smooth_order=3,
        suptitle_fontsize=14,
        suptitle_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        title_fontsize=14,
        title_fontweight='bold',
        legend_fontsize=10,
        show_grid=True,
        grid_alpha=0.3,
        save_fig=False,
        filename='pmf_overview.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Two-panel overview: PMF(z) on the left, PMF(|z|) on the right.

        Parameters
        ----------
        figsize : tuple
        unit : str
        zero_at : {'min', 'bulk'}
        plot_gmx : bool
        gmx_label : str
        suptitle : str or None
            Overall figure title.

        Returns
        -------
        fig, (ax_signed, ax_abs)
        """
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

        self.plot_pmf_signed(
            ax=ax0, unit=unit, zero_at=zero_at,
            plot_gmx=plot_gmx, gmx_label=gmx_label,
            smooth=smooth, smooth_window=smooth_window, smooth_order=smooth_order,
            label_fontsize=label_fontsize, label_fontweight=label_fontweight,
            tick_fontsize=tick_fontsize, title_fontsize=title_fontsize,
            title_fontweight=title_fontweight, legend_fontsize=legend_fontsize,
            show_grid=show_grid, grid_alpha=grid_alpha,
        )
        self.plot_pmf_abs(
            ax=ax1, unit=unit, zero_at=zero_at,
            plot_gmx=plot_gmx, gmx_label=gmx_label,
            smooth=smooth, smooth_window=smooth_window, smooth_order=smooth_order,
            label_fontsize=label_fontsize, label_fontweight=label_fontweight,
            tick_fontsize=tick_fontsize, title_fontsize=title_fontsize,
            title_fontweight=title_fontweight, legend_fontsize=legend_fontsize,
            show_grid=show_grid, grid_alpha=grid_alpha,
        )

        if suptitle is not None:
            fig.suptitle(suptitle, y=1.01,
                         fontsize=suptitle_fontsize, fontweight=suptitle_fontweight)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, (ax0, ax1)

    # ------------------------------------------------------------------
    # 4. Sampling — n_frames per window
    # ------------------------------------------------------------------

    def plot_sampling(
        self,
        ax=None,
        color='steelblue',
        figsize=(10, 4),
        title=None,
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        xlabel=None,
        ylabel=None,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        legend_fontsize=10,
        show_grid=True,
        grid_alpha=0.3,
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_sampling.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Bar chart showing the number of production frames per window.

        Bars for CIP1 (+z) and CIP2 (|z| reflected) are plotted at their
        respective |centre| positions, coloured separately.

        Parameters
        ----------
        ax : matplotlib Axes or None
        color : str
            Bar colour. Default 'steelblue'.
        figsize : tuple
        title : str

        Returns
        -------
        fig, ax
        """
        self._require_wham()
        pmf = self.pmf

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        nw = pmf.n_windows
        c1_vals = np.array([abs(pmf.window_centers[i][0]) for i in range(nw)])
        c2_vals = np.array([abs(pmf.window_centers[i][1]) for i in range(nw)])
        n1_vals = np.array([len(pmf.z_data[i][0]) for i in range(nw)], dtype=float)
        n2_vals = np.array([len(pmf.z_data[i][1]) for i in range(nw)], dtype=float)

        w = 0.015   # bar half-width in nm
        ax.bar(c1_vals, n1_vals / 1e3, width=w, color='steelblue',
               alpha=0.7, label='CIP1 (+z)')
        ax.bar(c2_vals + w, n2_vals / 1e3, width=w, color='tomato',
               alpha=0.7, label='CIP2 (−z → |z|)')

        _xlabel = xlabel if xlabel is not None else '|z₀|  (nm)'
        _ylabel = ylabel if ylabel is not None else 'Frames  (×10³)'
        _title  = title if title is not None else 'Umbrella window sampling'
        ax.set_xlabel(_xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(_ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax.set_title(_title, fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.legend(fontsize=legend_fontsize)
        if show_grid:
            ax.grid(True, axis='y', alpha=grid_alpha)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, ax

    # ------------------------------------------------------------------
    # 5. Window histograms — sampling overlap on RC grid
    # ------------------------------------------------------------------

    def plot_window_histograms(
        self,
        ax=None,
        figsize=(11, 5),
        max_windows=None,
        cmap='viridis',
        normalise=True,
        title=None,
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        xlabel=None,
        ylabel=None,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        cbar_label_fontsize=11,
        cbar_tick_fontsize=10,
        show_grid=True,
        grid_alpha=0.3,
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_window_histograms.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Overlay all 60 pseudo-window histograms on the |z| RC grid.

        Parameters
        ----------
        ax : matplotlib Axes or None
        figsize : tuple
        max_windows : int or None
            If set, only plot the first N pseudo-windows (for clarity).
        cmap : str
            Colourmap name for window colouring.
        normalise : bool
            If True, normalise each histogram to unit area before plotting.
        title : str

        Returns
        -------
        fig, ax
        """
        self._require_wham()
        pmf = self.pmf

        if pmf.histograms is None:
            raise RuntimeError("Histograms not available. Call run_wham() first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        R     = pmf.histograms.shape[0]
        if max_windows is not None:
            R = min(R, max_windows)

        r    = pmf.bin_centers
        cmap = plt.get_cmap(cmap)
        colours = [cmap(i / max(R - 1, 1)) for i in range(R)]

        for idx in range(R):
            h = pmf.histograms[idx].copy()
            if normalise and h.sum() > 0:
                h = h / (h.sum() * pmf.bin_width)
            ax.plot(r, h, color=colours[idx], lw=0.8, alpha=0.6)

        # Colour bar to show window index
        sm = plt.cm.ScalarMappable(
            cmap=plt.get_cmap(cmap),
            norm=plt.Normalize(vmin=0, vmax=R - 1),
        )
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label('Pseudo-window index', fontsize=cbar_label_fontsize)
        cb.ax.tick_params(labelsize=cbar_tick_fontsize)

        _label_y = 'Normalised count' if normalise else 'Count'
        _xlabel = xlabel if xlabel is not None else '|z|  (nm)'
        _ylabel = ylabel if ylabel is not None else _label_y
        _title  = title if title is not None else 'Umbrella window histograms'
        ax.set_xlabel(_xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(_ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax.set_title(_title, fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, ax

    # ------------------------------------------------------------------
    # 6. PMF(z) and PMF(|z|) — combined two-panel (from ClayPMF.plot_pmf)
    # ------------------------------------------------------------------

    def plot_pmf(
        self,
        figsize=(12, 5),
        unit='kJ/mol',
        zero_at='bulk',
        plot_gmx=False,
        gmx_label='gmx wham',
        color_python='steelblue',
        color_abs='mediumseagreen',
        color_gmx='tomato',
        ax=None,
        # ── Publication-quality font & style controls ──────────────────
        title_fontsize=13,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        legend_fontsize=10,
        show_title=True,
        show_grid=True,
        grid_alpha=0.3,
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_1d.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Plot PMF(z) and PMF(|z|) side by side.

        Parameters
        ----------
        figsize : tuple, default (12, 5)
        unit : str
            Energy unit label for y-axis. Default 'kJ/mol'.
        zero_at : {'min', 'bulk'}
            Reference point.  'min' zeros at the global minimum;
            'bulk' zeros at the average of the outermost 20 % of |z|.
        plot_gmx : bool
            Overlay loaded gmx wham profile on left panel. Default False.
        gmx_label : str
        color_python, color_abs, color_gmx : str
            Line colours.
        ax : array-like of Axes or None
            Pass two Axes objects to plot into existing subplots.

        Returns
        -------
        fig, axes
        """
        self._require_wham()
        self.set_default_style()
        p = self.pmf

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        else:
            fig = ax[0].get_figure()
            axes = ax

        z     = getattr(p, '_bin_centers_signed', p.bin_centers)
        pmf_s = p.pmf_signed.copy()
        r     = p.bin_centers_abs
        pmf_a = p.pmf_abs.copy()

        # Re-reference
        if zero_at == 'bulk':
            n_bulk = max(1, int(0.2 * len(r)))
            bulk_val = float(np.nanmean(pmf_a[-n_bulk:]))
            pmf_a -= bulk_val
            bulk_s = 0.5 * (
                float(np.nanmean(pmf_s[:n_bulk])) +
                float(np.nanmean(pmf_s[-n_bulk:]))
            )
            pmf_s -= bulk_s

        # ---- Left panel: signed z ----
        ax0 = axes[0]
        ax0.plot(z, pmf_s, color=color_python, lw=2, label='Python WHAM')
        if p.pmf_signed_std is not None:
            ax0.fill_between(
                z,
                pmf_s - p.pmf_signed_std,
                pmf_s + p.pmf_signed_std,
                alpha=0.25, color=color_python, label='±1σ (bootstrap)',
            )
        if plot_gmx and p.gmx_z is not None:
            gmx_p = p.gmx_pmf.copy()
            if zero_at == 'min':
                gmx_p -= np.nanmin(gmx_p)
            elif zero_at == 'bulk':
                n_b = max(1, int(0.2 * len(gmx_p)))
                gmx_p -= 0.5 * (np.nanmean(gmx_p[:n_b]) + np.nanmean(gmx_p[-n_b:]))
            ax0.plot(p.gmx_z, gmx_p, color=color_gmx, lw=1.5,
                     ls='--', label=gmx_label)
            if p.gmx_pmf_std is not None:
                ax0.fill_between(
                    p.gmx_z,
                    gmx_p - p.gmx_pmf_std,
                    gmx_p + p.gmx_pmf_std,
                    alpha=0.2, color=color_gmx,
                )
        ax0.axvline(0, color='k', lw=0.8, ls=':', label='Clay surface')
        ax0.set_xlabel('z  (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax0.set_ylabel(f'PMF  ({unit})', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax0.set_title('PMF(z) — Signed', fontsize=title_fontsize, fontweight=title_fontweight)
        ax0.tick_params(axis='both', labelsize=tick_fontsize)
        ax0.legend(fontsize=legend_fontsize)
        if show_grid:
            ax0.grid(True, alpha=grid_alpha)

        # ---- Right panel: |z| ----
        ax1 = axes[1]
        ax1.plot(r, pmf_a, color=color_abs, lw=2, label='Python WHAM (sym.)')
        if p.pmf_abs_std is not None:
            n = min(len(r), len(p.pmf_abs_std))
            ax1.fill_between(
                r[:n],
                pmf_a[:n] - p.pmf_abs_std[:n],
                pmf_a[:n] + p.pmf_abs_std[:n],
                alpha=0.25, color=color_abs, label='±1σ (bootstrap)',
            )
        ax1.set_xlabel('|z|  (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax1.set_ylabel(f'PMF  ({unit})', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax1.set_title('PMF(|z|) — Symmetrised', fontsize=title_fontsize, fontweight=title_fontweight)
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        ax1.legend(fontsize=legend_fontsize)
        if show_grid:
            ax1.grid(True, alpha=grid_alpha)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, axes

    # ------------------------------------------------------------------
    # 2D PMF plotting methods
    # ------------------------------------------------------------------

    def plot_2d_pmf(
        self,
        unit='kJ/mol',
        zero_at='bulk',
        cmap='viridis',
        levels=20,
        vmax=None,
        r_min=None,
        x_coord='dist',
        smooth=False,
        smooth_sigma=1.0,
        show_minimum=True,
        figsize=(8, 6),
        title=None,
        ax=None,
        # ── Publication-quality font & style controls ──────────────────
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        cbar_label_fontsize=11,
        cbar_tick_fontsize=10,
        show_contour_lines=True,
        contour_linewidth=0.3,
        contour_alpha=0.35,
        minimum_markersize=12,
        minimum_legend_fontsize=9,
        show_legend=True,
        legend_fontsize=None,
        legend_fontweight='normal',
        legend_framealpha=0.0,
        show_surface_line=True,
        surface_linewidth=1.2,
        show_title=True,
        xlabel=None,
        ylabel=None,
        bar_width=None,
        # ── Grid ───────────────────────────────────────────────────────
        show_grid=False,
        grid_alpha=0.3,
        grid_linestyle='--',
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_2d_heatmap.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Filled contour map of W(r, θ).

        Parameters
        ----------
        unit : str
            Energy unit: ``'kJ/mol'``, ``'kcal/mol'``, or ``'kT'``.
        zero_at : str
            ``'bulk'`` (first 20 % of r-axis) or ``'min'``.
        cmap : str
            Matplotlib colormap. Default ``'viridis'``.
        levels : int
            Number of contour levels.
        vmax : float or None
            Clip upper end of colour scale. If ``None`` and ``r_min``
            is set, auto-computed from bins beyond ``r_min``.
        r_min : float or None
            Minimum distance from clay surface (nm) used to auto-compute
            ``vmax``, excluding the repulsive-wall region. Ignored when
            ``vmax`` is supplied explicitly.
        x_coord : str
            ``'dist'`` → x = z_clay_surface − r; ``'r'`` → x = r.
        smooth : bool
            Apply Gaussian smoothing before plotting.
        smooth_sigma : float
            Gaussian sigma in grid units.
        show_minimum : bool
            Mark global minimum with a red star.
        figsize : tuple
        title : str or None
        ax : matplotlib Axes or None

        Publication-quality controls
        -----------------------------
        title_fontsize : float        (default 14)
        title_fontweight : str        (default 'bold')
        label_fontsize : float        (default 12)
        label_fontweight : str        (default 'bold')
        tick_fontsize : float         (default 11)
        cbar_label_fontsize : float   (default 11)
        cbar_tick_fontsize : float    (default 10)
        show_contour_lines : bool     (default True)
        contour_linewidth : float     (default 0.3)
        contour_alpha : float         (default 0.35)
        minimum_markersize : float    (default 12)
        minimum_legend_fontsize : float (default 9)
        show_surface_line : bool      (default True)
        surface_linewidth : float     (default 1.2)
        show_grid : bool              (default False)
        grid_alpha : float            (default 0.3)
        grid_linestyle : str          (default '--')
        save_fig : bool               (default False)
        filename : str                (default 'pmf_2d_heatmap.png')
        dpi : int                     (default 300)
        bbox_inches : str             (default 'tight')
        transparent_bg : bool         (default False)

        Returns
        -------
        fig, ax
        """
        self._require_wham_2d()
        self.set_default_style()
        p = self.pmf2d

        from scipy.ndimage import gaussian_filter

        pmf = p._to_unit(p.pmf_2d, unit)

        # Re-reference
        if zero_at == 'bulk':
            n_bulk = max(1, int(0.2 * p.n_r_bins))
            pmf -= float(np.nanmean(pmf[:n_bulk, :]))
        else:
            pmf -= np.nanmin(pmf)

        if smooth:
            pmf = gaussian_filter(np.nan_to_num(pmf), sigma=smooth_sigma)

        # x-axis
        x_shift = p.z_clay_surface or 0.0
        if x_coord == 'dist' and x_shift > 0:
            x       = x_shift - p.r_centers
            _xlabel = 'Distance from clay surface (nm)'
        else:
            x       = p.r_centers
            _xlabel = 'r = |z| (nm)'
        _xlabel = xlabel if xlabel is not None else _xlabel
        _ylabel = ylabel if ylabel is not None else 'Tilt angle θ (degrees)'

        y = p.theta_centers
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Auto-compute vmax from bins beyond r_min
        if vmax is None and r_min is not None:
            dist_mask = x >= r_min
            if dist_mask.any():
                vmax = float(np.nanmax(pmf[dist_mask, :]))

        pmf_plot = pmf.copy()
        if vmax is not None:
            pmf_plot = np.clip(pmf_plot, None, vmax)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        cf = ax.contourf(X, Y, pmf_plot, levels=levels, cmap=cmap)
        if show_contour_lines:
            ax.contour(X, Y, pmf_plot, levels=levels, colors='k',
                       linewidths=contour_linewidth, alpha=contour_alpha)

        _cbar_kw = dict(fraction=bar_width) if bar_width is not None else {}
        cbar = fig.colorbar(cf, ax=ax, **_cbar_kw)
        cbar.set_label(f'PMF ({unit})', fontsize=cbar_label_fontsize)
        cbar.ax.tick_params(labelsize=cbar_tick_fontsize)

        ax.set_xlabel(_xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(_ylabel, fontsize=label_fontsize,
                      fontweight=label_fontweight)
        if show_title:
            ax.set_title(title or f'2D PMF: W(r, θ)  [{unit}]',
                         fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)

        if show_grid:
            ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)

        if show_minimum:
            min_idx = np.unravel_index(np.nanargmin(pmf), pmf.shape)
            ax.plot(
                x[min_idx[0]], y[min_idx[1]],
                'r*', markersize=minimum_markersize,
                label=f'Min at ({x[min_idx[0]]:.2f} nm, {y[min_idx[1]]:.0f}°)',
            )
            if show_legend:
                _leg_fs = legend_fontsize if legend_fontsize is not None else minimum_legend_fontsize
                leg = ax.legend(fontsize=_leg_fs)
                plt.setp(leg.get_texts(), fontweight=legend_fontweight)
                leg.get_frame().set_alpha(legend_framealpha)

        if show_surface_line and x_coord == 'dist' and x_shift > 0:
            ax.axvline(0.0, ls='--', c='w', lw=surface_linewidth, alpha=0.7)

        plt.tight_layout()

        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
            print(f"Figure saved: {filename}")

        return fig, ax

    def plot_2d_marginals(
        self,
        unit='kJ/mol',
        zero_at='bulk',
        compare_1d=True,
        figsize=(12, 5),
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        xlabel_r=None,
        ylabel_r=None,
        xlabel_th=None,
        ylabel_th=None,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        legend_fontsize=10,
        show_grid=True,
        grid_alpha=0.3,
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_2d_marginals.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Side-by-side panels: W(r) marginal and W(θ) marginal.

        Parameters
        ----------
        unit : str
        zero_at : str
            ``'bulk'`` or ``'min'``.
        compare_1d : bool
            Overlay the 1D WHAM result (``self.pmf``) on the W(r) panel
            as a dashed line.  Requires ``self.pmf`` to be set.
        figsize : tuple

        Returns
        -------
        fig, (ax_r, ax_theta)
        """
        self._require_wham_2d()
        self.set_default_style()
        p = self.pmf2d
        compare = None
        if compare_1d and self.pmf is not None and self.pmf.pmf_abs is not None:
            compare = (self.pmf.bin_centers_abs, self.pmf.pmf_abs)

        pmf_r  = p._to_unit(p.pmf_abs, unit)
        pmf_th = p._to_unit(p.pmf_theta, unit)

        if zero_at == 'bulk':
            n_b    = max(1, int(0.2 * len(pmf_r)))
            pmf_r -= float(np.nanmean(pmf_r[:n_b]))
        else:
            pmf_r -= np.nanmin(pmf_r)
        pmf_th -= np.nanmin(pmf_th)

        x_shift    = p.z_clay_surface or 0.0
        r_plot     = (x_shift - p.r_centers) if x_shift > 0 else p.r_centers
        xlabel_r_auto = ('Distance from clay surface (nm)'
                         if x_shift > 0 else 'r = |z| (nm)')

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

        ax0.plot(r_plot, pmf_r, 'b-', lw=2, label='2D WHAM marginal')

        if compare is not None:
            r1d, pmf1d_raw = compare
            pmf1d = p._to_unit(np.asarray(pmf1d_raw), unit)
            if zero_at == 'bulk':
                n_b1d  = max(1, int(0.2 * len(pmf1d)))
                pmf1d -= float(np.nanmean(pmf1d[:n_b1d]))
            else:
                pmf1d -= np.nanmin(pmf1d)
            r1d_plot = (x_shift - r1d) if x_shift > 0 else r1d
            ax0.plot(r1d_plot, pmf1d, 'r--', lw=2, label='1D WHAM')

        _xlabel_r  = xlabel_r  if xlabel_r  is not None else xlabel_r
        _ylabel_r  = ylabel_r  if ylabel_r  is not None else f'PMF ({unit})'
        _xlabel_th = xlabel_th if xlabel_th is not None else 'Tilt angle θ (degrees)'
        _ylabel_th = ylabel_th if ylabel_th is not None else f'PMF ({unit})'
        # xlabel_r auto from coordinate
        _xlabel_r  = xlabel_r  if xlabel_r  is not None else xlabel_r_auto

        ax0.set_xlabel(_xlabel_r,  fontsize=label_fontsize, fontweight=label_fontweight)
        ax0.set_ylabel(_ylabel_r,  fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax0.set_title('W(r) — marginal over θ',
                          fontsize=title_fontsize, fontweight=title_fontweight)
        ax0.tick_params(axis='both', labelsize=tick_fontsize)
        ax0.legend(fontsize=legend_fontsize)
        if show_grid:
            ax0.grid(True, alpha=grid_alpha)
        if x_shift > 0:
            ax0.axvline(0.0, ls='--', c='grey', lw=1.2, alpha=0.6,
                        label='Clay surface')

        ax1.plot(p.theta_centers, pmf_th, 'g-', lw=2)
        ax1.set_xlabel(_xlabel_th, fontsize=label_fontsize, fontweight=label_fontweight)
        ax1.set_ylabel(_ylabel_th, fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax1.set_title('W(θ) — marginal over r',
                          fontsize=title_fontsize, fontweight=title_fontweight)
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        if show_grid:
            ax1.grid(True, alpha=grid_alpha)

        fig.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, (ax0, ax1)

    def plot_2d_coupling(
        self,
        unit='kJ/mol',
        cmap='RdBu_r',
        levels=20,
        vmax=None,
        x_coord='dist',
        figsize=(8, 6),
        title=None,
        ax=None,
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        xlabel=None,
        ylabel=None,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        cbar_label_fontsize=11,
        cbar_tick_fontsize=10,
        bar_width=None,
        show_contour_lines=True,
        contour_linewidth=1.5,
        show_zero_contour=True,
        zero_contour_linewidth=1.5,
        zero_contour_color='k',
        # ── Grid ───────────────────────────────────────────────────────
        show_grid=False,
        grid_alpha=0.3,
        grid_linestyle='--',
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_2d_coupling.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Coupling free energy ΔΔW(r, θ) = W(r,θ) − W(r) − W(θ).

        Negative regions = correlated geometry; positive = anticorrelated.
        The black zero contour marks statistical independence.
        """
        self._require_wham_2d()
        self.set_default_style()
        p = self.pmf2d

        ddW = p._to_unit(p.coupling_free_energy(), unit)

        x_shift = p.z_clay_surface or 0.0
        if x_coord == 'dist' and x_shift > 0:
            x       = x_shift - p.r_centers
            _xlabel = 'Distance from clay surface (nm)'
        else:
            x       = p.r_centers
            _xlabel = 'r = |z| (nm)'
        _xlabel = xlabel if xlabel is not None else _xlabel
        _ylabel = ylabel if ylabel is not None else 'Tilt angle θ (degrees)'

        X, Y    = np.meshgrid(x, p.theta_centers, indexing='ij')
        max_abs = float(np.nanmax(np.abs(ddW)))
        if vmax is not None:
            max_abs = float(vmax)
        lev = np.linspace(-max_abs, max_abs, levels + 1)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        cf = ax.contourf(
            X, Y, ddW, levels=lev, cmap=cmap,
            norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
            extend='neither',
        )
        if show_contour_lines:
            ax.contour(X, Y, ddW, levels=lev, colors='k',
                       linewidths=contour_linewidth, alpha=0.25)
        if show_zero_contour:
            ax.contour(X, Y, ddW, levels=[0.0],
                       colors=zero_contour_color,
                       linewidths=zero_contour_linewidth)

        _cbar_kw = dict(fraction=bar_width) if bar_width is not None else {}
        cbar = fig.colorbar(cf, ax=ax, **_cbar_kw)
        cbar.set_label(f'ΔΔW ({unit})', fontsize=cbar_label_fontsize)
        cbar.ax.tick_params(labelsize=cbar_tick_fontsize)

        ax.set_xlabel(_xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(_ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if show_title:
            ax.set_title(
                title or 'Coupling free energy ΔΔW(r, θ)',
                fontsize=title_fontsize, fontweight=title_fontweight,
            )
        if show_grid:
            ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)

        if x_coord == 'dist' and x_shift > 0:
            ax.axvline(0.0, ls='--', c='grey', lw=1.2, alpha=0.7)

        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)

        return fig, ax

    def plot_2d_conditional(
        self,
        r_indices=None,
        theta_indices=None,
        unit='kJ/mol',
        figsize=(9, 5),
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        xlabel=None,
        ylabel=None,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        legend_fontsize=9,
        line_width=2,
        show_grid=True,
        grid_alpha=0.3,
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_2d_conditional.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Conditional PMF at fixed r-slices or fixed θ-slices.
        Provide exactly one of ``r_indices`` or ``theta_indices``.
        """
        self._require_wham_2d()
        self.set_default_style()
        p = self.pmf2d

        if r_indices is not None:
            fig, ax = plt.subplots(figsize=figsize)
            colors  = plt.cm.viridis(np.linspace(0, 1, len(r_indices)))

            for color, r_idx in zip(colors, r_indices):
                th_c, cond = p.conditional_pmf(r_index=r_idx)
                cond_u     = p._to_unit(cond, unit)
                r_val      = p.r_centers[r_idx]
                label      = f'r = {r_val:.2f} nm'
                if p.z_clay_surface is not None:
                    d_val  = p.z_clay_surface - r_val
                    label += f'  (d = {d_val:.2f} nm from surface)'
                ax.plot(th_c, cond_u, color=color, lw=line_width, label=label)

            _xlabel = xlabel if xlabel is not None else 'Tilt angle θ (degrees)'
            _ylabel = ylabel if ylabel is not None else f'W(θ | r) ({unit})'
            _title  = 'Conditional PMF: orientation at fixed distance'

        elif theta_indices is not None:
            fig, ax  = plt.subplots(figsize=figsize)
            colors   = plt.cm.plasma(np.linspace(0, 1, len(theta_indices)))
            x_shift  = p.z_clay_surface or 0.0

            for color, th_idx in zip(colors, theta_indices):
                r_c, cond = p.conditional_pmf(theta_index=th_idx)
                cond_u    = p._to_unit(cond, unit)
                x         = (x_shift - r_c) if x_shift > 0 else r_c
                th_val    = p.theta_centers[th_idx]
                ax.plot(x, cond_u, color=color, lw=line_width,
                        label=f'θ = {th_val:.0f}°')

            _auto_xlabel = ('Distance from clay surface (nm)'
                            if x_shift > 0 else 'r = |z| (nm)')
            _xlabel = xlabel if xlabel is not None else _auto_xlabel
            _ylabel = ylabel if ylabel is not None else f'W(r | θ) ({unit})'
            _title  = 'Conditional PMF: distance at fixed orientation'
            if x_shift > 0:
                ax.axvline(0.0, ls='--', c='grey', lw=1.2, alpha=0.7)

        else:
            raise ValueError("Provide r_indices or theta_indices.")

        ax.set_xlabel(_xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(_ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if show_title:
            ax.set_title(_title, fontsize=title_fontsize, fontweight=title_fontweight)
        ax.legend(fontsize=legend_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        fig.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, ax

    def plot_2d_pmf_3d(
        self,
        unit='kJ/mol',
        zero_at='bulk',
        cmap='viridis',
        vmax=None,
        r_min=None,
        x_coord='dist',
        smooth=False,
        smooth_sigma=1.0,
        show_minimum=True,
        show_contours=True,
        filled_contours=True,
        contour_fill_alpha=0.75,
        contour_linewidth=1.5,
        contour_levels=8,
        floor_offset=0.0,
        energy_shift=0.0,
        elev=30,
        azim=-60,
        alpha=0.85,
        wall_color=(0.88, 0.88, 0.88, 0.55),
        wall_alpha=None,       # override transparency for wall_color (0–1); None = use colour's own alpha
        figsize=(10, 7),
        title=None,
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        label_pad=8,
        tick_fontsize=10,
        cbar_label_fontsize=11,
        cbar_tick_fontsize=10,
        cbar_shrink=0.55,
        cbar_pad=0.08,
        cbar_x=None,          # shift colorbar left/right (e.g. -0.08 to move left)
        show_legend=True,
        legend_fontsize=9,
        legend_fontweight='normal',
        legend_framealpha=0.0,
        minimum_markersize=80,
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_2d_surface_3d.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        3-D surface landscape of W(r, θ).

        Height and colour both encode the PMF value.  Floor contours give
        spatial context.  All data come from the already-computed
        ``pmf2d.pmf_2d`` — nothing is recalculated.

        Parameters
        ----------
        unit, zero_at, cmap, vmax, r_min, x_coord, smooth,
        smooth_sigma, show_minimum, show_contours, filled_contours,
        contour_levels, elev, azim, alpha, wall_color, figsize, title :
            Forwarded to :meth:`ClayPMF2D.plot_2d_pmf_3d`.

        Returns
        -------
        fig, ax  (``mpl_toolkits.mplot3d.Axes3D``)
        """
        self._require_wham_2d()
        self.set_default_style()
        p = self.pmf2d

        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        from scipy.ndimage import gaussian_filter
        import matplotlib.colors as mcolors

        pmf = p._to_unit(p.pmf_2d, unit)

        # Reference energy
        if zero_at == 'bulk':
            n_bulk = max(1, int(0.2 * p.n_r_bins))
            pmf -= float(np.nanmean(pmf[:n_bulk, :]))
        else:
            pmf -= np.nanmin(pmf)

        if smooth:
            pmf = gaussian_filter(np.nan_to_num(pmf), sigma=smooth_sigma)

        # x-axis
        x_shift = p.z_clay_surface or 0.0
        if x_coord == 'dist' and x_shift > 0:
            x       = x_shift - p.r_centers
            _xlabel = 'Distance from clay surface (nm)'
        else:
            x       = p.r_centers
            _xlabel = 'r = |z| (nm)'
        _xlabel = xlabel if xlabel is not None else _xlabel
        _ylabel = ylabel if ylabel is not None else 'Tilt angle θ (°)'
        # zlabel=None or zlabel='' → hide (no dead space); any non-empty string → show
        _zlabel = zlabel if zlabel else None

        y = p.theta_centers
        X, Y = np.meshgrid(x, y, indexing='ij')   # (n_r, n_theta)

        # Auto-compute vmax
        if vmax is None and r_min is not None:
            dist_mask = x >= r_min
            if dist_mask.any():
                vmax = float(np.nanmax(pmf[dist_mask, :]))

        pmf_plot = pmf.copy() + float(energy_shift)
        vmin_val = float(np.nanmin(pmf_plot))
        vmax_val = vmax if vmax is not None else float(np.nanmax(pmf_plot))
        floor_z  = vmin_val - float(floor_offset)
        if vmax is not None:
            pmf_plot = np.clip(pmf_plot, None, vmax_val)

        pmf_surface = pmf_plot.copy()

        norm = mcolors.Normalize(vmin=vmin_val, vmax=vmax_val)
        _z_for_color = np.where(np.isnan(pmf_surface), vmin_val, pmf_surface)
        facecolors = plt.get_cmap(cmap)(norm(_z_for_color))

        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')

        ax.plot_surface(
            X, Y, pmf_surface,
            facecolors=facecolors,
            alpha=alpha,
            linewidth=0,
            antialiased=True,
            shade=False,
        )

        if show_contours:
            if filled_contours:
                ax.contourf(
                    X, Y, pmf_plot,
                    levels=contour_levels,
                    cmap=cmap,
                    zdir='z',
                    offset=floor_z,
                    alpha=contour_fill_alpha,
                )
                ax.contour(
                    X, Y, pmf_plot,
                    levels=contour_levels,
                    colors='k',
                    zdir='z',
                    offset=floor_z,
                    linewidths=contour_linewidth,
                )
            else:
                ax.contour(
                    X, Y, pmf_plot,
                    levels=contour_levels,
                    cmap=cmap,
                    zdir='z',
                    offset=floor_z,
                    linewidths=contour_linewidth,
                )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=cbar_shrink, pad=cbar_pad)
        cbar.set_label(f'PMF ({unit})', fontsize=cbar_label_fontsize)
        cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
        if cbar_x is not None:
            fig.canvas.draw()          # flush layout so get_position() is accurate
            pos = cbar.ax.get_position()
            cbar.ax.set_position([pos.x0 + cbar_x, pos.y0, pos.width, pos.height])

        ax.set_xlabel(_xlabel, fontsize=label_fontsize, fontweight=label_fontweight,
                      labelpad=label_pad)
        ax.set_ylabel(_ylabel, fontsize=label_fontsize,
                      fontweight=label_fontweight, labelpad=label_pad)
        if _zlabel:
            ax.set_zlabel(_zlabel, fontsize=label_fontsize,
                          fontweight=label_fontweight, labelpad=label_pad)
        else:
            ax.set_zlabel('')
            ax.zaxis.label.set_visible(False)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if show_title:
            ax.set_title(title or f'2D PMF landscape: W(r, θ)  [{unit}]',
                         fontsize=title_fontsize, fontweight=title_fontweight)
        ax.view_init(elev=elev, azim=azim)

        fig.canvas.draw()
        if wall_color is not None:
            _wall_default_alpha = wall_alpha if wall_alpha is not None else 0.55
            def _parse_wall_color(c):
                """Accept hex str, named colour, or (r,g,b[,a]) tuple → (r,g,b,a)."""
                rgba = mcolors.to_rgba(c)
                # wall_alpha overrides everything; otherwise use the colour's own alpha
                a = wall_alpha if wall_alpha is not None else rgba[3]
                return rgba[:3] + (a,)

            # wall_color can be:
            #   single colour  → same colour for both walls
            #   (color_x, color_y) → separate colours per wall
            _is_pair = (
                isinstance(wall_color, (list, tuple))
                and len(wall_color) == 2
                and not isinstance(wall_color[0], (int, float))   # not a plain (r,g,b,a) tuple
            )
            if _is_pair:
                _wc_x = _parse_wall_color(wall_color[0])
                _wc_y = _parse_wall_color(wall_color[1])
            else:
                _wc_x = _wc_y = _parse_wall_color(wall_color)

            for _pane, _wc in ((ax.xaxis.pane, _wc_x), (ax.yaxis.pane, _wc_y)):
                _pane.fill = True
                _pane.set_facecolor(_wc[:3] + (1.0,))
                _pane.set_alpha(_wc[3])
                _pane.set_edgecolor((0, 0, 0, 0))
        ax.zaxis.pane.fill = False
        ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.pane.set_alpha(0.0)

        if show_minimum:
            min_idx = np.unravel_index(np.nanargmin(pmf), pmf.shape)
            xm, ym, zm = x[min_idx[0]], y[min_idx[1]], pmf_surface[min_idx]
            ax.scatter(
                [xm], [ym], [zm],
                color='red', s=minimum_markersize, zorder=5,
                label=f'Min ({xm:.2f} nm, {ym:.0f}°)',
            )
            if show_legend:
                leg = ax.legend(fontsize=legend_fontsize)
                plt.setp(leg.get_texts(), fontweight=legend_fontweight)
                leg.get_frame().set_alpha(legend_framealpha)

        plt.tight_layout()

        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
            print(f"Figure saved: {filename}")

        return fig, ax

    def plot_2d_overview(
        self,
        unit='kJ/mol',
        zero_at='bulk',
        compare_1d=True,
        smooth=False,
        smooth_sigma=1.0,
        suptitle=None,
        figsize=(14, 10),
        # ── Colormap / levels ──────────────────────────────────────────
        cmap='viridis',
        vmax=None,
        r_min=None,
        n_levels=20,
        show_contour_lines=True,
        contour_linewidth=0.3,
        contour_alpha=0.35,
        show_zero_contour=False,
        zero_contour_color='white',
        zero_contour_lw=1.5,
        # ── Coupling panel (panel 1) ────────────────────────────────────
        coupling_cmap='RdBu_r',   # diverging cmap for ΔΔW (default: RdBu_r)
        coupling_vmax=None,       # symmetric clip |ΔΔW| ≤ coupling_vmax; None=auto
        # ── Axis labels ────────────────────────────────────────────────
        xlabel=None,
        ylabel=None,
        # ── Publication-quality font & style controls ──────────────────
        label_fontsize=10,
        label_fontweight='bold',
        tick_fontsize=9,
        title_fontsize=11,
        title_fontweight='bold',
        suptitle_fontsize=14,
        suptitle_fontweight='bold',
        legend_fontsize=9,
        cbar_label_fontsize=None,
        cbar_tick_fontsize=None,
        show_grid=True,
        grid_alpha=0.3,
        # ── Export ─────────────────────────────────────────────────────
        show_individual_figures=False,
        save_individual_figures=False,
        save_combined_figure=False,
        save_fig=False,              # alias for save_combined_figure
        individual_figsize=(6, 5),
        filename='pmf_2d_overview.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        2 × 2 summary figure.

        ┌──────────────────────┬──────────────────────┐
        │  2D PMF  W(r, θ)    │  Coupling  ΔΔW       │
        ├──────────────────────┼──────────────────────┤
        │  Marginal W(r)       │  Marginal W(θ)       │
        └──────────────────────┴──────────────────────┘

        Parameters
        ----------
        unit : str
        zero_at : str
        compare_1d : bool
            Overlay 1D WHAM on W(r) panel (requires ``self.pmf``).
        smooth : bool
            Gaussian smoothing for the 2D PMF panel only.
        smooth_sigma : float
        suptitle : str or None
        figsize : tuple

        Returns
        -------
        fig, axes  (2 × 2 numpy array)
        """
        self._require_wham_2d()
        self.set_default_style()
        p = self.pmf2d
        compare = None
        if compare_1d and self.pmf is not None and self.pmf.pmf_abs is not None:
            compare = (self.pmf.bin_centers_abs, self.pmf.pmf_abs)

        _cbar_lfs = cbar_label_fontsize if cbar_label_fontsize is not None else label_fontsize
        _cbar_tfs = cbar_tick_fontsize  if cbar_tick_fontsize  is not None else tick_fontsize

        # ── Pre-compute panel data ────────────────────────────────────
        x_shift   = p.z_clay_surface or 0.0
        _xl_dist  = xlabel if xlabel is not None else (
            'Distance from clay surface (nm)' if x_shift > 0 else 'r (nm)')
        _yl_angle = ylabel if ylabel is not None else 'θ (degrees)'

        # Coupling ΔΔW(r,θ) = -kT ln[P(r,θ) / (P(r)·P(θ))]
        # Exactly zero when r and θ are independent; no zeroing required.
        # Using the probability ratio avoids the NaN-bin bias that logsumexp
        # marginals suffer when p.pmf_2d has many unvisited (NaN) grid cells.
        _P2d    = p.P_2d                                             # (n_r, n_θ) density
        _Pr     = np.nansum(_P2d, axis=1) * p.theta_width           # (n_r,)  marginal density
        _Pth    = np.nansum(_P2d, axis=0) * p.r_width               # (n_θ,)  marginal density
        _denom  = _Pr[:, np.newaxis] * _Pth[np.newaxis, :]          # (n_r, n_θ)
        _ratio  = np.where((_P2d > 0) & (_denom > 0),
                           _P2d / _denom, np.nan)
        _ddW_kj = np.where(np.isfinite(_ratio) & (_ratio > 0),
                           -np.log(_ratio) / p.beta, np.nan)
        ddW     = p._to_unit(_ddW_kj, unit)

        x_coup = (x_shift - p.r_centers) if x_shift > 0 else p.r_centers
        X, Y   = np.meshgrid(x_coup, p.theta_centers, indexing='ij')
        # Mask poorly-sampled near-surface bins before auto-scaling
        ddW_masked = ddW.copy()
        if r_min is not None:
            ddW_masked[x_coup < r_min, :] = np.nan
        if coupling_vmax is not None:
            _coup_abs = float(coupling_vmax)
        else:
            _finite = np.abs(ddW_masked[np.isfinite(ddW_masked)])
            _coup_abs = float(np.nanpercentile(_finite, 97)) if len(_finite) > 0 \
                        else 1.0
        lev     = np.linspace(-_coup_abs, _coup_abs, 21)

        # W(r) marginal (panel 2)
        pmf_r = p._to_unit(p.pmf_abs, unit)
        if zero_at == 'bulk':
            n_b    = max(1, int(0.2 * len(pmf_r)))
            pmf_r -= float(np.nanmean(pmf_r[:n_b]))
        else:
            pmf_r -= np.nanmin(pmf_r)
        r_plot = (x_shift - p.r_centers) if x_shift > 0 else p.r_centers

        # W(θ) marginal (panel 3)
        pmf_th = p._to_unit(p.pmf_theta, unit)
        pmf_th -= np.nanmin(pmf_th)

        # ── Panel drawing helper ──────────────────────────────────────
        def _draw_panel(ax, idx, fig_ref):
            if idx == 0:
                self.plot_2d_pmf(
                    unit=unit, zero_at=zero_at, smooth=smooth,
                    smooth_sigma=smooth_sigma, x_coord='dist',
                    cmap=cmap, levels=n_levels, vmax=vmax, r_min=r_min,
                    show_contour_lines=show_contour_lines,
                    contour_linewidth=contour_linewidth,
                    contour_alpha=contour_alpha,
                    xlabel=xlabel, ylabel=ylabel,
                    title=f'2D PMF W(r, θ)  [{unit}]', ax=ax,
                    label_fontsize=label_fontsize, label_fontweight=label_fontweight,
                    tick_fontsize=tick_fontsize, title_fontsize=title_fontsize,
                    title_fontweight=title_fontweight,
                    cbar_label_fontsize=_cbar_lfs, cbar_tick_fontsize=_cbar_tfs,
                )
            elif idx == 1:
                cf2 = ax.contourf(
                    X, Y, ddW_masked, levels=lev, cmap=coupling_cmap,
                    norm=TwoSlopeNorm(vmin=-_coup_abs, vcenter=0.0, vmax=_coup_abs),
                )
                ax.contour(X, Y, ddW_masked, levels=[0.0], colors='k', linewidths=1.5)
                _cb2 = fig_ref.colorbar(cf2, ax=ax)
                _cb2.set_label(f'ΔΔW ({unit})', fontsize=_cbar_lfs)
                _cb2.ax.tick_params(labelsize=_cbar_tfs)
                ax.set_xlabel(_xl_dist, fontsize=label_fontsize, fontweight=label_fontweight)
                ax.set_ylabel(_yl_angle, fontsize=label_fontsize, fontweight=label_fontweight)
                ax.set_title(f'Coupling ΔΔW(r, θ)  [{unit}]',
                             fontsize=title_fontsize, fontweight=title_fontweight)
                ax.tick_params(axis='both', labelsize=tick_fontsize)
            elif idx == 2:
                ax.plot(r_plot, pmf_r, 'b-', lw=2, label='2D marginal')
                if compare is not None:
                    r1d, pmf1d_raw = compare
                    pmf1d = p._to_unit(np.asarray(pmf1d_raw), unit)
                    if zero_at == 'bulk':
                        n_b1  = max(1, int(0.2 * len(pmf1d)))
                        pmf1d -= float(np.nanmean(pmf1d[:n_b1]))
                    else:
                        pmf1d -= np.nanmin(pmf1d)
                    r1d_p = (x_shift - r1d) if x_shift > 0 else r1d
                    ax.plot(r1d_p, pmf1d, 'r--', lw=2, label='1D WHAM')
                if p.pmf_abs_std is not None:
                    ax.fill_between(
                        r_plot, pmf_r - p.pmf_abs_std, pmf_r + p.pmf_abs_std,
                        alpha=0.25, color='b', label='±1σ bootstrap',
                    )
                ax.set_xlabel(_xl_dist, fontsize=label_fontsize, fontweight=label_fontweight)
                ax.set_ylabel(f'PMF ({unit})', fontsize=label_fontsize,
                              fontweight=label_fontweight)
                ax.set_title('W(r) — marginal over θ',
                             fontsize=title_fontsize, fontweight=title_fontweight)
                ax.tick_params(axis='both', labelsize=tick_fontsize)
                ax.legend(fontsize=legend_fontsize)
                if show_grid:
                    ax.grid(True, alpha=grid_alpha)
                if x_shift > 0:
                    ax.axvline(0.0, ls='--', c='grey', lw=1.2, alpha=0.6)
            elif idx == 3:
                ax.plot(p.theta_centers, pmf_th, 'g-', lw=2)
                _yl11 = ylabel if ylabel is not None else 'Tilt angle θ (degrees)'
                ax.set_xlabel(_yl11, fontsize=label_fontsize, fontweight=label_fontweight)
                ax.set_ylabel(f'PMF ({unit})', fontsize=label_fontsize,
                              fontweight=label_fontweight)
                ax.set_title('W(θ) — marginal over r',
                             fontsize=title_fontsize, fontweight=title_fontweight)
                ax.tick_params(axis='both', labelsize=tick_fontsize)
                if show_grid:
                    ax.grid(True, alpha=grid_alpha)

        # ── Individual figures (one per panel) ───────────────────────
        _base, _ext = os.path.splitext(filename)
        panel_tags  = ['2D_PMF', 'coupling', 'W_r', 'W_theta']

        if show_individual_figures or save_individual_figures:
            for idx in range(4):
                fig_ind, ax_ind = plt.subplots(1, 1, figsize=individual_figsize)
                _draw_panel(ax_ind, idx, fig_ind)
                fig_ind.tight_layout()
                if save_individual_figures:
                    ind_fname = f"{_base}_{panel_tags[idx]}{_ext}"
                    fig_ind.savefig(ind_fname, dpi=dpi, bbox_inches=bbox_inches,
                                    transparent=transparent_bg)
                    print(f"  Saved: {os.path.abspath(ind_fname)}")
                if show_individual_figures:
                    plt.show()
                else:
                    plt.close(fig_ind)

        # ── Combined 2×2 figure ───────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax_map = {0: axes[0, 0], 1: axes[0, 1], 2: axes[1, 0], 3: axes[1, 1]}
        for idx in range(4):
            _draw_panel(ax_map[idx], idx, fig)

        if suptitle:
            fig.suptitle(suptitle, fontsize=suptitle_fontsize,
                         fontweight=suptitle_fontweight, y=1.01)
        fig.tight_layout()
        if save_fig or save_combined_figure:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, axes

    # ------------------------------------------------------------------
    # 3D PMF plotting methods  (delegates to ClayPMF3D plot methods)
    # ------------------------------------------------------------------

    def plot_3d_marginals(
        self,
        unit='kJ/mol',
        zero_at='bulk',
        figsize=(15, 5),
        title=None,
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        title_fontsize=13,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        suptitle_fontsize=13,
        suptitle_fontweight='bold',
        show_grid=True,
        grid_alpha=0.3,
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_3d_marginals.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Three-panel marginal PMFs: W(r), W(θ), W(n_cat).

        Parameters
        ----------
        unit : str
            Energy unit: ``'kJ/mol'``, ``'kcal/mol'``, or ``'kT'``.
        zero_at : str
            ``'bulk'`` (first 20 % of r-axis) or ``'min'``.
        figsize : tuple
        title : str or None

        Returns
        -------
        fig, axes
        """
        self._require_wham_3d()
        self.set_default_style()
        p = self.pmf3d

        def _prep(arr, r_like=False):
            pmf = p._to_unit(arr, unit)
            if zero_at == 'bulk' and r_like:
                n_b  = max(1, int(0.2 * len(pmf)))
                pmf -= float(np.nanmedian(pmf[:n_b]))
            else:
                pmf -= np.nanmin(pmf)
            return pmf

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # W(r)
        ax = axes[0]
        x_shift = p.z_clay_surface or 0.0
        x = (x_shift - p.r_centers) if x_shift > 0 else p.r_centers
        xlabel = 'Distance from clay surface (nm)' if x_shift > 0 else 'r = |z| (nm)'
        pmf_r = _prep(p.pmf_r, r_like=True)
        ax.plot(x, pmf_r, 'b-', lw=2)
        if p.pmf_r_std is not None:
            pmf_r_std = p._to_unit(p.pmf_r_std, unit)
            ax.fill_between(x, pmf_r - pmf_r_std, pmf_r + pmf_r_std, alpha=0.25)
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'W ({unit})', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax.set_title('W(r) — marginal distance PMF',
                         fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
        if x_shift > 0:
            ax.axvline(0.0, ls='--', c='grey', lw=1, alpha=0.6)

        # W(θ)
        ax = axes[1]
        pmf_th = _prep(p.pmf_theta)
        ax.plot(p.theta_centers, pmf_th, 'g-', lw=2)
        ax.set_xlabel('Tilt angle θ (°)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'W ({unit})', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax.set_title('W(θ) — marginal orientation PMF',
                         fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        # W(n_cat)
        ax = axes[2]
        pmf_nc = _prep(p.pmf_cation)
        ax.bar(p.cation_centers, pmf_nc, width=0.7, alpha=0.7, color='darkorange')
        ax.set_xlabel(f'n_{p._cation_label}', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'W ({unit})', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax.set_title(f'W(n_{p._cation_label}) — cation coordination PMF',
                         fontsize=title_fontsize, fontweight=title_fontweight)
        ax.set_xticks(p.cation_centers.astype(int))
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha, axis='y')

        if title:
            fig.suptitle(title, fontsize=suptitle_fontsize,
                         fontweight=suptitle_fontweight)
        fig.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, axes

    def plot_3d_ensemble_marginals(
        self,
        figsize=(15, 4),
        individual_figsize=(6, 5),
        zero_at='bulk',
        mask_unvisited=True,
        # ── Publication-quality font & style controls ──────────────────
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        title_fontsize=12,
        title_fontweight='bold',
        suptitle_fontsize=13,
        suptitle_fontweight='bold',
        show_grid=True,
        grid_alpha=0.3,
        # ── Legend ─────────────────────────────────────────────────────
        show_legend=True,
        legend_fontsize=9,
        legend_framealpha=0.8,
        legend_loc='best',
        # ── WHAM mean line ─────────────────────────────────────────────
        wham_color='k',
        wham_lw=2.5,
        wham_ls='-',
        # ── Error fill (WHAM ±1σ) ──────────────────────────────────────
        fill_color='k',
        fill_alpha=0.25,
        # ── NN-A line ──────────────────────────────────────────────────
        nn_a_color='steelblue',
        nn_a_lw=2.0,
        nn_a_ls='--',
        # ── NN-B line ──────────────────────────────────────────────────
        nn_b_color='tomato',
        nn_b_lw=2.0,
        nn_b_ls=':',
        # ── Export ─────────────────────────────────────────────────────
        show_individual_figures=False,
        save_individual_figures=False,
        save_combined_figure=False,
        save_fig=False,              # alias for save_combined_figure
        filename='pmf_3d_ensemble_marginals.png',
        dpi=300,
        bbox_inches='tight',
        # ── NN line domain masking ──────────────────────────────────────
        nn_r_min=None,    # surface-distance lower bound (nm): hide NN below this d
                          # Per-NN near-surface masking uses ens.r_min_a / ens.r_min_b
                          # (set automatically by fit_smooth / fit_reweighted).
        # ── Bulk reference fraction ─────────────────────────────────────
        bulk_frac=0.2,    # fraction of first r-bins used as bulk reference (median→0)
        # ── Optional 1-D reference PMF overlay (W(r) panel only) ──────────
        ref_r=None,        # r-coordinates (nm) in same d-axis as W(r) panel
        ref_w=None,        # W values (kJ/mol), bulk-referenced
        ref_color='forestgreen',
        ref_lw=2.0,
        ref_ls='-.',
        ref_label='Reference PMF',
    ):
        """
        W(r), W(θ), W(n_cat) for all replicates: mean ± σ (WHAM) with
        optional NN-A and NN-B overlay.

        Requires ``plotter.ensemble`` to be a ``ClayPMFNeuralEnsemble``
        instance with ``pmf3d_list`` populated and WHAM completed.

        Parameters
        ----------
        figsize          : tuple
        zero_at          : 'bulk' or 'min'
        mask_unvisited   : bool  — passed to ensemble._marginals_from_nn
        nn_r_min         : float or None
            Surface-distance lower bound (nm).  NN lines in the W(r) panel are
            blanked where r_surf < nn_r_min (near-surface / within-clay region).
            Each NN's training r_min (stored as ``ens.r_min_a`` / ``ens.r_min_b``
            by fit_smooth / fit_reweighted) is also applied automatically —
            if r_min was None at training time, no near-surface masking is added.
        bulk_frac        : float
            Fraction of the first r-bins used to compute the bulk median reference
            (default 0.2).  Reduce to e.g. 0.1 if the first 20% contains noisy
            or sparsely-sampled bins.

        Returns
        -------
        fig, axes
        """
        self._require_ensemble()
        self.set_default_style()

        ens = self.ensemble
        p   = ens.pmf3d              # reference pmf3d (grid geometry)
        beta = ens.beta
        kT   = ens.kT

        x_surf = float(getattr(p, 'z_clay_surface', None) or 0.0)
        r_surf = x_surf - p.r_centers
        r_ref  = p.r_centers   # reference r-grid for alignment

        # ── Per-replicate marginals ──────────────────────────────────
        marg_r_list, marg_th_list, marg_n_list = [], [], []
        for pmf3d in ens.pmf3d_list:
            # second bulk shift (same as fit_smooth uses)
            n_b = max(1, int(bulk_frac * pmf3d.n_r_bins))
            sh  = float(np.nanmedian(pmf3d.pmf_3d[:n_b, :, :]))
            W   = pmf3d.pmf_3d - sh

            log_r  = _logsumexp(-beta * W, axis=(1, 2))
            log_th = _logsumexp(-beta * W, axis=(0, 2))
            log_n  = _logsumexp(-beta * W, axis=(0, 1))

            pmf_r_raw = -kT * log_r
            # Each replicate may have a different r_max (data-driven), so its
            # r_centers differ from the reference.  Interpolate onto the
            # reference grid before stacking to avoid index-based misalignment.
            r_rep = pmf3d.r_centers
            if not np.array_equal(r_rep, r_ref):
                valid = np.isfinite(pmf_r_raw)
                if valid.any():
                    pmf_r = np.interp(r_ref, r_rep[valid], pmf_r_raw[valid],
                                      left=np.nan, right=np.nan)
                else:
                    pmf_r = np.full_like(r_ref, np.nan)
            else:
                pmf_r = pmf_r_raw

            if zero_at == 'bulk':
                n_b_r = max(1, int(bulk_frac * len(pmf_r)))
                pmf_r -= float(np.nanmedian(pmf_r[:n_b_r]))
            else:
                pmf_r -= np.nanmin(pmf_r)

            pmf_th = -kT * log_th
            pmf_th -= np.nanmin(pmf_th)

            pmf_n = -kT * log_n
            pmf_n -= np.nanmin(pmf_n)

            marg_r_list.append(pmf_r)
            marg_th_list.append(pmf_th)
            marg_n_list.append(pmf_n)

        with np.errstate(all='ignore'):
            mean_r  = np.nanmean(marg_r_list,  axis=0)
            std_r   = np.nanstd(marg_r_list,   axis=0)
            mean_th = np.nanmean(marg_th_list, axis=0)
            std_th  = np.nanstd(marg_th_list,  axis=0)
            mean_n  = np.nanmean(marg_n_list,  axis=0)
            std_n   = np.nanstd(marg_n_list,   axis=0)

        # ── NN marginals ─────────────────────────────────────────────
        nn_margs = {}
        for tag in ('a', 'b'):
            nn = ens.nn_a if tag == 'a' else ens.nn_b
            if nn is not None:
                pmf_r_nn, pmf_th_nn, pmf_n_nn = ens._marginals_from_nn(
                    tag, mask_unvisited=mask_unvisited
                )
                # _marginals_from_nn zeros W(r) at its minimum (surface);
                # re-reference to bulk (first bulk_frac median) to match WHAM.
                if zero_at == 'bulk':
                    n_b_r      = max(1, int(bulk_frac * len(pmf_r_nn)))
                    bulk_shift = float(np.nanmedian(pmf_r_nn[:n_b_r]))
                    if not np.isfinite(bulk_shift):
                        # Primary bulk bins are all NaN (outside the sampled
                        # WHAM range).  Fall back: shift NN so it matches the
                        # already-bulk-referenced WHAM mean over the valid
                        # overlap region.
                        overlap = np.isfinite(pmf_r_nn) & np.isfinite(mean_r)
                        if overlap.any():
                            bulk_shift = float(
                                np.nanmedian((pmf_r_nn - mean_r)[overlap])
                            )
                        else:
                            bulk_shift = 0.0
                    pmf_r_nn -= bulk_shift
                nn_margs[tag] = (pmf_r_nn, pmf_th_nn, pmf_n_nn)

        # ── Plot ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        specs = [
            (0, r_surf,           mean_r,  std_r,
             'Distance, d (nm)',  'W(r)  (kJ/mol)'),
            (1, p.theta_centers,  mean_th, std_th,
             'Tilt, θ (°)',       'W(θ)  (kJ/mol)'),
            (2, p.cation_centers, mean_n,  std_n,
             '$n_{\\rm cat}$',   'W($n_{\\rm cat}$)  (kJ/mol)'),
        ]

        def _draw_marginal(ax, idx, x, y_mean, y_std, xlabel, ylabel):
            """Render one marginal panel onto ax."""
            ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                            alpha=fill_alpha, color=fill_color,
                            label='WHAM ±1σ')
            ax.plot(x, y_mean, color=wham_color, ls=wham_ls, lw=wham_lw,
                    label='WHAM mean', zorder=3)

            for tag, color, ls, lw, lbl in (
                ('a', nn_a_color, nn_a_ls, nn_a_lw, 'NN-A'),
                ('b', nn_b_color, nn_b_ls, nn_b_lw, 'NN-B'),
            ):
                if tag in nn_margs:
                    nn_vals = np.array(nn_margs[tag][idx], dtype=float)
                    if idx == 0:
                        # W(r) panel: mask NN line outside reliable domain.
                        # np.isfinite(y_mean) handles extreme tails with no WHAM data.
                        nn_valid = np.isfinite(y_mean)
                        # Explicit d-axis lower bound (e.g. hide inside clay)
                        if nn_r_min is not None:
                            nn_valid &= (x >= nn_r_min)
                        # Auto-mask where this NN was not trained: the r_min used
                        # during fit_smooth / fit_reweighted is stored on the ensemble
                        # as r_min_a / r_min_b.  If r_min was None (all data used)
                        # no extra masking is applied — nothing is hidden wrongly.
                        _r_min_train = getattr(ens, 'r_min_a' if tag == 'a' else 'r_min_b', None)
                        if _r_min_train is not None and _r_min_train > 0:
                            # r_centers < _r_min_train was excluded from training.
                            # r_surf = x_surf - r_centers  →  r_centers = x_surf - x
                            # r_centers < _r_min_train  ↔  x > x_surf - _r_min_train
                            # Only apply when r_min > 0; r_min=0.0 means train on
                            # all data and show NN everywhere WHAM has coverage.
                            nn_valid &= (x <= x_surf - _r_min_train)
                        nn_vals = np.where(nn_valid, nn_vals, np.nan)
                    ax.plot(x, nn_vals,
                            color=color, ls=ls, lw=lw, label=lbl)

            if idx == 0 and ref_r is not None and ref_w is not None:
                ax.plot(np.asarray(ref_r), np.asarray(ref_w),
                        color=ref_color, ls=ref_ls, lw=ref_lw,
                        label=ref_label, zorder=2)

            if idx == 2:
                ax.set_xticks(p.cation_centers.astype(int))
            if idx == 0:
                ax.axvline(0.0, color='grey', lw=1, ls='--', alpha=0.7)

            ax.set_xlabel(xlabel, fontsize=label_fontsize,
                          fontweight=label_fontweight)
            ax.set_ylabel(ylabel, fontsize=label_fontsize,
                          fontweight=label_fontweight)
            ax.set_title(ylabel.split('  ')[0],
                         fontsize=title_fontsize, fontweight=title_fontweight)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            if show_legend:
                ax.legend(fontsize=legend_fontsize, loc=legend_loc,
                          framealpha=legend_framealpha)
            if show_grid:
                ax.grid(True, alpha=grid_alpha)

        # ── Individual figures (one per panel) ───────────────────────
        _base, _ext = os.path.splitext(filename)
        panel_tags = ['W_r', 'W_theta', 'W_ncat']

        for idx, x, y_mean, y_std, xlabel, ylabel in specs:
            if show_individual_figures or save_individual_figures:
                fig_ind, ax_ind = plt.subplots(1, 1, figsize=individual_figsize)
                _draw_marginal(ax_ind, idx, x, y_mean, y_std, xlabel, ylabel)
                fig_ind.tight_layout()
                if save_individual_figures:
                    ind_fname = f"{_base}_{panel_tags[idx]}{_ext}"
                    fig_ind.savefig(ind_fname, dpi=dpi, bbox_inches=bbox_inches)
                    print(f"  Saved: {ind_fname}")
                if show_individual_figures:
                    plt.show()
                else:
                    plt.close(fig_ind)

        # ── Combined 3-panel figure ───────────────────────────────────
        for idx, x, y_mean, y_std, xlabel, ylabel in specs:
            _draw_marginal(axes[idx], idx, x, y_mean, y_std, xlabel, ylabel)

        fig.suptitle(
            f'1-D marginals  [{ens.n_replicates} replicates]',
            fontsize=suptitle_fontsize, fontweight=suptitle_fontweight,
            y=1.01,
        )
        fig.tight_layout()

        if save_fig or save_combined_figure:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            print(f"  Saved: {filename}")

        return fig

    def plot_losses(self, figsize=(11, 4)):
        """
        Side-by-side training loss curves for Approach A and B.

        Requires ``plotter.ensemble`` to be a ``ClayPMFNeuralEnsemble``
        instance that has been fitted.

        Returns
        -------
        fig : matplotlib Figure
        """
        self._require_ensemble()
        self.set_default_style()

        ens = self.ensemble
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, losses, label, color in zip(
            axes,
            [ens.losses_a, ens.losses_b],
            [f'Approach A — grid smoother  ({ens.n_replicates} rep.)',
             f'Approach B — reweighted  ({ens.n_replicates} rep.)'],
            ['steelblue', 'tomato'],
        ):
            if losses is None:
                ax.text(0.5, 0.5, 'Not fitted', ha='center', va='center',
                        fontsize=12, transform=ax.transAxes, color='grey')
                ax.set_title(label)
            else:
                ax.semilogy(losses, color=color, lw=1.5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MSE loss  (kJ/mol)²')
                ax.set_title(label)
                ax.grid(True, alpha=0.3)
                ax.text(0.98, 0.95, f'final={losses[-1]:.3e}',
                        ha='right', va='top', transform=ax.transAxes,
                        fontsize=9, color=color)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------------
    # Ensemble coupling map  ΔΔW(r,θ) = W(r,θ) − W(r) − W(θ)
    # -----------------------------------------------------------------------

    def plot_ensemble_coupling(
        self,
        zero_at='bulk',
        r_min=None,
        vmax=None,
        cmap='RdBu_r',
        n_levels=40,
        figsize=(8, 6),
        individual_figsize=(8, 6),
        # ── Publication font & style controls ───────────────────────────
        show_title=True,
        show_grid=False,
        grid_alpha=0.3,
        label_fontsize=11,
        label_fontweight='bold',
        tick_fontsize=10,
        title_fontsize=12,
        title_fontweight='bold',
        suptitle_fontsize=13,
        suptitle_fontweight='bold',
        cbar_label_fontsize=10,
        cbar_tick_fontsize=9,
        colorbar_pad=0.02,
        colorbar_width='4%',
        show_contour_lines=True,
        contour_linewidth=0.8,
        show_zero_contour=True,
        zero_contour_color='k',
        zero_contour_lw=1.5,
        # ── Axis labels ─────────────────────────────────────────────────
        xlabel=None,   # default: 'Distance, d (nm)'
        ylabel=None,   # default: 'Tilt, θ (°)'
        # ── Layout: show mean + sigma side by side ───────────────────────
        show_sigma_panel=True,
        # ── Output ──────────────────────────────────────────────────────
        show_individual_figures=False,
        save_individual_figures=False,
        save_combined_figure=False,
        save_fig=False,
        filename='pmf_ensemble_coupling.png',
        dpi=300,
        bbox_inches='tight',
    ):
        """
        Ensemble coupling free energy ΔΔW(r, θ) = W(r,θ) − W(r) − W(θ)

        Computed from the WHAM mean PMF (marginalized over n_cat).
        Also shows the inter-replicate σ of the coupling map.

        Negative regions = r and θ are correlated (geometry matters).
        Positive regions = anticorrelated.
        Black zero contour = statistical independence.

        Parameters
        ----------
        zero_at          : 'bulk' or 'min'
        r_min            : float or None  — mask distances below this value (nm)
        vmax             : float or None  — explicit symmetric colour scale
        show_sigma_panel : bool  — add a second panel showing σ(ΔΔW)

        Returns
        -------
        fig : matplotlib Figure
        """
        from matplotlib.colors import TwoSlopeNorm
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        self._require_ensemble()
        self.set_default_style()

        ens = self.ensemble
        p   = ens.pmf3d

        x_surf = float(getattr(p, 'z_clay_surface', None) or 0.0)
        r_surf = x_surf - p.r_centers   # distance from surface

        beta = ens.beta
        kT   = ens.kT

        # ── Per-replicate coupling maps ───────────────────────────────────
        coup_maps = []
        for pmf3d in ens.pmf3d_list:
            sh    = ens._bulk_shift_single(pmf3d, zero_at)
            W3d   = pmf3d.pmf_3d - sh          # (n_r, n_theta, n_cat)

            # 2-D marginal: marginalize over n_cat
            W2d = -kT * _logsumexp(-beta * W3d, axis=2)   # (n_r, n_theta)
            W2d -= np.nanmin(W2d)

            # 1-D marginals
            Wr  = -kT * _logsumexp(-beta * W2d, axis=1)   # (n_r,)
            Wth = -kT * _logsumexp(-beta * W2d, axis=0)   # (n_theta,)
            Wr  -= np.nanmin(Wr)
            Wth -= np.nanmin(Wth)

            # coupling
            coup = W2d - Wr[:, np.newaxis] - Wth[np.newaxis, :]
            coup_maps.append(coup)

        stack      = np.stack(coup_maps, axis=0)   # (n_rep, n_r, n_theta)
        mean_coup  = np.nanmean(stack, axis=0)
        std_coup   = np.nanstd(stack,  axis=0)

        # ── r_min mask ────────────────────────────────────────────────────
        if r_min is not None:
            r_mask = r_surf < r_min
            mean_coup[r_mask, :] = np.nan
            std_coup[r_mask, :]  = np.nan

        # ── Colour scales ─────────────────────────────────────────────────
        max_abs = vmax if vmax is not None else float(np.nanmax(np.abs(mean_coup)))
        lev_coup = np.linspace(-max_abs, max_abs, n_levels + 1)
        lev_sig  = np.linspace(0.0, float(np.nanpercentile(std_coup, 98)), n_levels + 1)

        _xl = xlabel if xlabel is not None else 'Distance, d (nm)'
        _yl = ylabel if ylabel is not None else 'Tilt, θ (°)'

        def _draw_coupling(ax, Z, levels, cmap_name, title_str, sigma_mode=False):
            if not sigma_mode:
                norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
                cf = ax.contourf(r_surf, p.theta_centers, Z.T,
                                 levels=levels, cmap=cmap_name,
                                 norm=norm, extend='neither')
                if show_contour_lines:
                    ax.contour(r_surf, p.theta_centers, Z.T,
                               levels=levels, colors='k',
                               linewidths=contour_linewidth, alpha=0.2)
                if show_zero_contour:
                    ax.contour(r_surf, p.theta_centers, Z.T,
                               levels=[0.0], colors=zero_contour_color,
                               linewidths=zero_contour_lw)
            else:
                cf = ax.contourf(r_surf, p.theta_centers, Z.T,
                                 levels=levels, cmap='Reds', extend='max')

            if x_surf > 0:
                ax.axvline(0.0, color='white', lw=1.2, ls='--', alpha=0.8)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size=colorbar_width, pad=colorbar_pad)
            cb  = ax.get_figure().colorbar(cf, cax=cax)
            cb.set_label('ΔΔW  (kJ/mol)' if not sigma_mode else 'σ(ΔΔW)  (kJ/mol)',
                         fontsize=cbar_label_fontsize)
            cb.ax.tick_params(labelsize=cbar_tick_fontsize)

            ax.set_xlabel(_xl, fontsize=label_fontsize, fontweight=label_fontweight)
            ax.set_ylabel(_yl, fontsize=label_fontsize, fontweight=label_fontweight)
            ax.tick_params(labelsize=tick_fontsize)
            if show_grid:
                ax.grid(True, alpha=grid_alpha)
            if show_title:
                ax.set_title(title_str, fontsize=title_fontsize,
                             fontweight=title_fontweight)

        # ── Individual figures ────────────────────────────────────────────
        base, ext = filename.rsplit('.', 1) if '.' in filename else (filename, 'png')
        fig_mean = fig_sig = None

        if show_individual_figures or save_individual_figures:
            fig_mean, ax_m = plt.subplots(figsize=individual_figsize)
            _draw_coupling(ax_m, mean_coup, lev_coup, cmap,
                           f'Coupling  ΔΔW(r,θ)  —  mean of {ens.n_replicates} replicates')
            fig_mean.tight_layout()
            if save_individual_figures:
                fig_mean.savefig(f'{base}_mean.{ext}', dpi=dpi, bbox_inches=bbox_inches)
            if show_individual_figures:
                plt.show()
            else:
                plt.close(fig_mean)

            if show_sigma_panel:
                fig_sig, ax_s = plt.subplots(figsize=individual_figsize)
                _draw_coupling(ax_s, std_coup, lev_sig, 'Reds',
                               f'Coupling uncertainty  σ(ΔΔW)  —  {ens.n_replicates} replicates',
                               sigma_mode=True)
                fig_sig.tight_layout()
                if save_individual_figures:
                    fig_sig.savefig(f'{base}_sigma.{ext}', dpi=dpi, bbox_inches=bbox_inches)
                if show_individual_figures:
                    plt.show()
                else:
                    plt.close(fig_sig)

        # ── Combined figure ───────────────────────────────────────────────
        n_panels = 2 if show_sigma_panel else 1
        w        = figsize[0] * n_panels / (1 if not show_sigma_panel else 1)
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(figsize[0] * n_panels, figsize[1]))
        if n_panels == 1:
            axes = [axes]

        _draw_coupling(axes[0], mean_coup, lev_coup, cmap,
                       f'ΔΔW(r,θ)  —  mean  ({ens.n_replicates} rep.)')
        if show_sigma_panel:
            _draw_coupling(axes[1], std_coup, lev_sig, 'Reds',
                           f'σ(ΔΔW)  —  {ens.n_replicates} rep.',
                           sigma_mode=True)

        fig.suptitle('Ensemble coupling free energy  ΔΔW(r, θ) = W(r,θ) − W(r) − W(θ)',
                     fontsize=suptitle_fontsize, fontweight=suptitle_fontweight)
        fig.tight_layout()

        if save_fig or save_combined_figure:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)

        return fig

    def plot_comparison_slice(
        self,
        n_cat_val=1,
        # ── Layout ─────────────────────────────────────────────────────
        figsize=None,
        individual_figsize=(8, 6),
        # ── Data / contour ─────────────────────────────────────────────
        vmax=None,
        r_min=None,
        cmap='viridis',
        mask_unvisited=True,
        n_levels=40,
        zero_at='bulk',
        show_uncertainty=True,
        # ── Publication font & style controls ──────────────────────────
        show_title=True,
        show_grid=False,
        title_fontsize=12,
        title_fontweight='bold',
        label_fontsize=11,
        label_fontweight='bold',
        tick_fontsize=10,
        cbar_label_fontsize=10,
        cbar_tick_fontsize=9,
        suptitle_fontsize=13,
        suptitle_fontweight='bold',
        colorbar_pad=0.02,
        colorbar_width='4%',
        # ── Axis labels ───────────────────────────────────────────────
        xlabel=None,   # default: 'Distance from surface (nm)'
        ylabel=None,   # default: 'θ  (°)'
        # ── Texture / grain ────────────────────────────────────────────
        noise_type=None,
        noise_amplitude=0.2,
        noise_seed=None,
        # ── Output ─────────────────────────────────────────────────────
        show_individual_figures=False,
        save_individual_figures=False,
        save_combined_figure=False,
        save_fig=False,
        filename='pmf_comparison_slice.png',
        dpi=300,
        bbox_inches='tight',
    ):
        """
        W(r, θ) at fixed n_cat: mean WHAM, NN-A, NN-B, and (optionally) σ.

        Requires ``plotter.ensemble`` to be a ``ClayPMFNeuralEnsemble``
        instance that has been fitted.

        Parameters
        ----------
        n_cat_val             : int
        figsize               : tuple or None  auto-sized if None
        individual_figsize    : tuple          size for each individual panel figure
        vmax                  : float or None  shared colour ceiling (kJ/mol)
        r_min                 : float or None  surface-distance lower cut (nm)
        cmap                  : str
        mask_unvisited        : bool   hide NN predictions outside WHAM coverage
        n_levels              : int    contour levels
        zero_at               : 'bulk' or 'min'
        show_uncertainty      : bool   append inter-replicate σ panel
        noise_type            : None | 'gaussian' | 'perlin'
            None       — no grain (default)
            'gaussian' — independent per-pixel white noise; fast, no deps
            'perlin'   — spatially correlated coherent noise (organic/wispy);
                         requires ``pip install opensimplex``
        noise_amplitude       : float  RMS amplitude of noise in kJ/mol (default 0.2)
        noise_seed            : int or None  RNG seed for reproducibility
        show_title            : bool   show per-panel title
        show_grid             : bool   show axis grid lines
        title_fontsize/weight : panel title style
        label_fontsize/weight : x/y axis label style
        tick_fontsize         : axis tick label size
        cbar_label_fontsize   : colorbar label size
        cbar_tick_fontsize    : colorbar tick size
        suptitle_fontsize/weight : combined figure suptitle style
        colorbar_pad          : float  gap between panel and colorbar
        colorbar_width        : str    colorbar width (e.g. '4%')
        show_individual_figures : bool  display one figure per panel
        save_individual_figures : bool  save each panel as a separate file
                                        (filenames derived from `filename`)
        save_combined_figure  : bool   save the combined multi-panel figure
        save_fig              : bool   alias for save_combined_figure
        filename              : str    path for combined figure (and base for individuals)
        dpi                   : int
        bbox_inches           : str

        Returns
        -------
        fig : matplotlib Figure or None (if n_cat_val has no WHAM data)
        """
        import matplotlib.colors as mcolors
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        self._require_ensemble()
        self.set_default_style()

        ens   = self.ensemble
        p     = ens.pmf3d
        kT    = ens.kT
        beta  = ens.beta

        x_surf = float(getattr(p, 'z_clay_surface', None) or 0.0)
        r_surf = x_surf - p.r_centers

        mean_pmf, std_pmf = ens.mean_wham_pmf(zero_at=zero_at)

        # ── n_cat_val=None → marginalize over all n_cat ──────────────────
        # W(r,θ) = -kT · log( Σ_n exp(-β W(r,θ,n)) )
        # This is the "general" view: the free energy surface irrespective
        # of cation coordination.
        marginalized = (n_cat_val is None)

        if marginalized:
            n_label = 'all'
            n_val   = None    # only used in formatted titles below

            # WHAM marginal: logsumexp over n_cat axis (axis=2)
            with np.errstate(all='ignore'):
                log_z  = _logsumexp(-beta * mean_pmf, axis=2)
                W_mean = -kT * log_z
            if zero_at == 'bulk':
                n_b = max(1, int(0.2 * W_mean.shape[0]))
                W_mean -= float(np.nanmedian(W_mean[:n_b, :]))
            else:
                W_mean -= float(np.nanmin(W_mean))

            # σ marginal: propagate uncertainty as Boltzmann-weighted std
            # (population-weighted combination across n_cat bins)
            with np.errstate(all='ignore'):
                log_z3 = _logsumexp(-beta * mean_pmf, axis=2, keepdims=True)
                w_pop  = np.exp(-beta * mean_pmf - log_z3)   # shape (r, θ, n_cat)
                W_std  = np.sqrt(np.nansum(w_pop**2 * std_pmf**2, axis=2))

            wham_occupied = np.isfinite(W_mean)
        else:
            n_idx = int(np.argmin(np.abs(p.cation_centers - n_cat_val)))
            n_val = p.cation_centers[n_idx]
            n_label = f'{n_val:.0f}'

            W_mean = mean_pmf[:, :, n_idx]
            W_std  = std_pmf[:, :, n_idx]
            wham_occupied = np.isfinite(W_mean)

        if not np.any(np.isfinite(W_mean)):
            print(f"  n_cat={n_label}: no WHAM data in any replicate — skipping.")
            return None

        panels = [('Mean WHAM', W_mean)]

        if ens.nn_a is not None:
            if marginalized:
                # evaluate on full (r, θ, n_cat) grid then marginalize
                R_q, TH_q, N_q = np.meshgrid(
                    p.r_centers, p.theta_centers, p.cation_centers, indexing='ij')
                X_q  = np.column_stack([R_q.ravel(), TH_q.ravel(), N_q.ravel()])
                Xn, _ = ens._normalise(X_q, ens.norm_a)
                Z_a3 = ens.nn_a.predict(Xn).reshape(R_q.shape)
                with np.errstate(all='ignore'):
                    Z_a = -kT * _logsumexp(-beta * Z_a3, axis=2)
                if zero_at == 'bulk':
                    n_b = max(1, int(0.2 * Z_a.shape[0]))
                    Z_a -= float(np.nanmedian(Z_a[:n_b, :]))
                else:
                    Z_a -= float(np.nanmin(Z_a))
                if mask_unvisited:
                    Z_a = np.where(wham_occupied, Z_a, np.nan)
            else:
                R_q, TH_q = np.meshgrid(p.r_centers, p.theta_centers, indexing='ij')
                N_q  = np.full(R_q.shape, n_val)
                X_q  = np.column_stack([R_q.ravel(), TH_q.ravel(), N_q.ravel()])
                Xn, _ = ens._normalise(X_q, ens.norm_a)
                Z_a  = ens.nn_a.predict(Xn).reshape(R_q.shape)
                if mask_unvisited:
                    Z_a = np.where(wham_occupied, Z_a, np.nan)
            panels.append(('NN-A  (smooth)', Z_a))

        if ens.nn_b is not None:
            if marginalized:
                R_q, TH_q, N_q = np.meshgrid(
                    p.r_centers, p.theta_centers, p.cation_centers, indexing='ij')
                X_q  = np.column_stack([R_q.ravel(), TH_q.ravel(), N_q.ravel()])
                Xn, _ = ens._normalise(X_q, ens.norm_b)
                Z_b3 = ens.nn_b.predict(Xn).reshape(R_q.shape)
                with np.errstate(all='ignore'):
                    Z_b = -kT * _logsumexp(-beta * Z_b3, axis=2)
                if zero_at == 'bulk':
                    n_b = max(1, int(0.2 * Z_b.shape[0]))
                    Z_b -= float(np.nanmedian(Z_b[:n_b, :]))
                else:
                    Z_b -= float(np.nanmin(Z_b))
                if mask_unvisited:
                    Z_b = np.where(wham_occupied, Z_b, np.nan)
            else:
                R_q, TH_q = np.meshgrid(p.r_centers, p.theta_centers, indexing='ij')
                N_q  = np.full(R_q.shape, n_val)
                X_q  = np.column_stack([R_q.ravel(), TH_q.ravel(), N_q.ravel()])
                Xn, _ = ens._normalise(X_q, ens.norm_b)
                Z_b  = ens.nn_b.predict(Xn).reshape(R_q.shape)
                if mask_unvisited:
                    Z_b = np.where(wham_occupied, Z_b, np.nan)
            panels.append(('NN-B  (reweighted)', Z_b))

        if show_uncertainty:
            panels.append(('σ  (inter-replicate)', W_std))

        r_mask = None
        if r_min is not None:
            r_mask = (x_surf - p.r_centers) >= r_min

        # ── Noise / grain helper ──────────────────────────────────────
        def _apply_noise(Z):
            """Return a display copy of Z with grain added to finite cells.

            The noise is added to the display array ONLY — the colour
            scale (_vmin/_vmax) is computed from the original Z, so
            peaks, valleys, and region boundaries are unaffected.
            NaN cells (masked/unvisited) stay NaN.
            """
            if noise_type is None:
                return Z
            rng = np.random.default_rng(noise_seed)
            nr, nth = Z.shape
            if noise_type == 'gaussian':
                grain = rng.normal(0.0, noise_amplitude, size=(nr, nth))
            elif noise_type in ('perlin', 'simplex'):
                try:
                    import opensimplex as osx
                    if noise_seed is not None:
                        osx.seed(int(noise_seed))
                    # Evaluate on a normalised [0,4] grid so we get
                    # ~2–3 visible wave cycles across each axis.
                    xs = np.linspace(0.0, 4.0, nr)
                    ys = np.linspace(0.0, 4.0, nth)
                    grain = np.array(
                        [[osx.noise2(x, y) for y in ys] for x in xs],
                        dtype=float,
                    )  # values in [-1, 1]
                    grain *= noise_amplitude   # scale to kJ/mol
                except ImportError:
                    import warnings
                    warnings.warn(
                        "opensimplex not installed — falling back to Gaussian noise. "
                        "Install with: pip install opensimplex",
                        stacklevel=2,
                    )
                    grain = rng.normal(0.0, noise_amplitude, size=(nr, nth))
            else:
                raise ValueError(
                    f"noise_type must be None, 'gaussian', or 'perlin'; got {noise_type!r}"
                )
            # Add grain only where data exist
            return np.where(np.isfinite(Z), Z + grain, Z)

        # ── Helper: render one panel onto a given axes ────────────────
        def _draw_panel(ax, title, Z_plot):
            # ── Colour scale: computed from the FULL data before masking ──
            # vmax uses the r_mask region (avoids near-surface extremes
            # blowing the scale); vmin uses full data so the scale is stable.
            if r_mask is not None and r_mask.any():
                _vmax = (vmax if vmax is not None
                         else float(np.nanpercentile(Z_plot[r_mask, :], 98)))
            else:
                _vmax = (vmax if vmax is not None
                         else float(np.nanpercentile(Z_plot, 98)))
            _vmin = float(np.nanmin(Z_plot))
            if _vmax <= _vmin:
                _vmax = _vmin + 1e-6

            # ── Display mask: blank the physically inaccessible region ──
            # This only affects what's drawn; the colorbar scale is unchanged.
            Z_display = Z_plot.copy()
            if r_mask is not None:
                Z_display[~r_mask, :] = np.nan

            # ── Grain / texture (display copy only) ──────────────────
            # Noise is added AFTER the colour scale is fixed so it cannot
            # shift peaks, move region boundaries, or alter the colorbar.
            Z_display = _apply_noise(Z_display)

            _cnorm  = mcolors.Normalize(vmin=_vmin, vmax=_vmax)
            _levels = np.linspace(_vmin, _vmax, n_levels + 1)
            Zc = np.where(np.isfinite(Z_display),
                          np.clip(Z_display, _vmin, _vmax),
                          np.nan)
            # contourf does NOT treat NaN as missing data — it interpolates
            # across them.  A masked array tells matplotlib to truly blank
            # unvisited cells instead of filling across them.
            Zc = np.ma.masked_invalid(Zc)

            cf = ax.contourf(r_surf, p.theta_centers, Zc.T,
                             levels=_levels, cmap=cmap, norm=_cnorm,
                             extend='neither')
            ax.axvline(0.0, color='white', lw=1.5, ls='--', alpha=0.9)

            _xl = xlabel if xlabel is not None else 'Distance from surface (nm)'
            ax.set_xlabel(_xl, fontsize=label_fontsize, fontweight=label_fontweight)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            if show_grid:
                ax.grid(True, alpha=0.3)
            if show_title:
                ax.set_title(f'{title}\n$n_{{\\rm cat}}={n_label}$',
                             fontsize=title_fontsize, fontweight=title_fontweight)

            # Colorbar with precise width/pad via make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size=colorbar_width, pad=colorbar_pad)
            cb  = plt.colorbar(cf, cax=cax)
            cb.set_label('W  (kJ/mol)', fontsize=cbar_label_fontsize)
            cb.ax.tick_params(labelsize=cbar_tick_fontsize)
            return cf

        # ── Individual figures (one per panel) ───────────────────────
        if show_individual_figures or save_individual_figures:
            base, ext = os.path.splitext(filename)
            ext = ext or '.png'
            for title, Z_plot in panels:
                fig_ind, ax_ind = plt.subplots(figsize=individual_figsize)
                _yl = ylabel if ylabel is not None else 'θ  (°)'
                ax_ind.set_ylabel(_yl, fontsize=label_fontsize, fontweight=label_fontweight)
                _draw_panel(ax_ind, title, Z_plot)
                fig_ind.tight_layout()
                if save_individual_figures:
                    safe = title.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                    ind_fname = f"{base}_{safe}{ext}"
                    fig_ind.savefig(ind_fname, dpi=dpi, bbox_inches=bbox_inches)
                    print(f"  Saved: {ind_fname}")
                if show_individual_figures:
                    plt.show()
                else:
                    plt.close(fig_ind)

        # ── Combined multi-panel figure ───────────────────────────────
        ncols = len(panels)
        if figsize is None:
            figsize = (5 * ncols, 5)
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        if ncols == 1:
            axes = [axes]

        _yl = ylabel if ylabel is not None else 'θ  (°)'
        for i, (ax, (title, Z_plot)) in enumerate(zip(axes, panels)):
            if i == 0:
                ax.set_ylabel(_yl, fontsize=label_fontsize, fontweight=label_fontweight)
            _draw_panel(ax, title, Z_plot)

        fig.suptitle(
            f'W(r, θ) at $n_{{\\rm cat}} = {n_label}$  '
            f'[{ens.n_replicates} replicates]',
            y=1.01,
            fontsize=suptitle_fontsize,
            fontweight=suptitle_fontweight,
        )
        fig.tight_layout()

        if save_fig or save_combined_figure:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            print(f"  Saved: {filename}")

        return fig

    def plot_uncertainty_map(
        self,
        n_cat_val=1,
        figsize=(7, 5),
        cmap='Reds',
        zero_at='bulk',
        n_levels=30,
        vmax=None,
        r_min=None,
        # ── Publication font & style controls ───────────────────────────
        show_title=True,
        show_grid=False,
        grid_alpha=0.3,
        label_fontsize=11,
        label_fontweight='bold',
        tick_fontsize=10,
        title_fontsize=12,
        title_fontweight='bold',
        cbar_label_fontsize=10,
        cbar_tick_fontsize=9,
        colorbar_pad=0.02,
        colorbar_width='4%',
        # ── Axis labels ─────────────────────────────────────────────────
        xlabel=None,   # default: 'Distance, d (nm)'
        ylabel=None,   # default: 'Tilt, θ (°)'
        # ── Output ──────────────────────────────────────────────────────
        save_fig=False,
        filename='uncertainty_map.png',
        dpi=300,
        bbox_inches='tight',
    ):
        """
        Inter-replicate standard deviation σ(W) at fixed n_cat.

        High σ values indicate poorly converged or under-sampled regions
        where the replicates disagree most.

        Parameters
        ----------
        n_cat_val : int
        figsize   : tuple
        cmap      : str
        zero_at   : 'bulk' or 'min'
        n_levels  : int
        vmax      : float or None  — explicit colour-scale ceiling
        r_min     : float or None  — mask distances below this value

        Returns
        -------
        fig : matplotlib Figure
        """
        self._require_ensemble()
        self.set_default_style()

        ens   = self.ensemble
        p     = ens.pmf3d
        n_idx = int(np.argmin(np.abs(p.cation_centers - n_cat_val)))
        n_val = p.cation_centers[n_idx]

        _, std_pmf = ens.mean_wham_pmf(zero_at=zero_at)
        W_std = std_pmf[:, :, n_idx]

        x_surf = float(getattr(p, 'z_clay_surface', None) or 0.0)
        r_surf = x_surf - p.r_centers

        # optional r_min mask
        if r_min is not None:
            r_mask = r_surf < r_min
            W_display = W_std.copy()
            W_display[r_mask, :] = np.nan
        else:
            W_display = W_std

        _vmax   = vmax if vmax is not None else float(np.nanpercentile(W_std, 98))
        _levels = np.linspace(0.0, max(_vmax, 1e-6), n_levels + 1)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, ax = plt.subplots(figsize=figsize)
        cf = ax.contourf(r_surf, p.theta_centers, W_display.T,
                         levels=_levels, cmap=cmap, extend='max')
        ax.axvline(0.0, color='white', lw=1.5, ls='--', alpha=0.9)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=colorbar_width, pad=colorbar_pad)
        cb  = fig.colorbar(cf, cax=cax)
        cb.set_label('σ(W)  (kJ/mol)', fontsize=cbar_label_fontsize)
        cb.ax.tick_params(labelsize=cbar_tick_fontsize)

        _xl = xlabel if xlabel is not None else 'Distance, d (nm)'
        _yl = ylabel if ylabel is not None else 'Tilt, θ (°)'
        ax.set_xlabel(_xl, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(_yl, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.tick_params(labelsize=tick_fontsize)

        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        if show_title:
            ax.set_title(
                f'Inter-replicate uncertainty  σ(W)  at  '
                f'$n_{{\\rm cat}} = {n_val:.0f}$\n'
                f'({ens.n_replicates} replicates)',
                fontsize=title_fontsize,
                fontweight=title_fontweight,
            )

        fig.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        return fig

    def plot_3d_slices(
        self,
        unit='kJ/mol',
        zero_at='bulk',
        slice_axis=2,
        n_slices=None,
        cmap='viridis',
        levels=20,
        vmax=None,
        figsize=None,
        title=None,
        r_min=None,
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        title_fontsize=11,
        title_fontweight='bold',
        label_fontsize=10,
        label_fontweight='bold',
        tick_fontsize=9,
        cbar_label_fontsize=10,
        cbar_tick_fontsize=9,
        suptitle_fontsize=13,
        suptitle_fontweight='bold',
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_3d_slices.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        2-D contourf slices of W(r,θ,n) at fixed values of one axis.

        Parameters
        ----------
        slice_axis : int
            Axis to fix: 0 = r, 1 = θ, 2 = n_cat (default).  Each
            selected bin becomes one panel.
        n_slices : int or None
            Number of panels.  Defaults to all bins along slice_axis.
        unit, zero_at, cmap, levels, figsize, title : as usual.

        Returns
        -------
        fig
        """
        self._require_wham_3d()
        self.set_default_style()
        p = self.pmf3d

        all_centers = [p.r_centers, p.theta_centers, p.cation_centers]
        all_labels  = ['r = |z| (nm)', 'θ (°)', f'n_{p._cation_label}']

        # Apply surface-relative shift so r=0 is at the clay surface
        x_shift = p.z_clay_surface or 0.0
        if x_shift > 0:
            all_centers[0] = x_shift - p.r_centers
            all_labels[0]  = 'Distance from clay surface (nm)'

        fixed_centers = all_centers[slice_axis]
        free_axes     = [a for a in (0, 1, 2) if a != slice_axis]

        if n_slices is None:
            n_slices = len(fixed_centers)
        else:
            n_slices = min(n_slices, len(fixed_centers))

        indices = np.linspace(0, len(fixed_centers) - 1, n_slices, dtype=int)

        ncols = min(n_slices, 4)
        nrows = (n_slices + ncols - 1) // ncols
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        pmf = p._to_unit(p.pmf_3d.copy(), unit)
        if zero_at == 'bulk':
            n_b = max(1, int(0.2 * p.n_r_bins))
            pmf -= float(np.nanmedian(pmf[:n_b, :, :]))
        else:
            pmf -= np.nanmin(pmf)

        # Determine vmax: explicit > auto from r_min > None
        _vmax = vmax
        if _vmax is None and r_min is not None and free_axes[0] == 0:
            r_mask = all_centers[0] >= r_min
            if r_mask.any():
                _vmax = float(np.nanmax(pmf[r_mask, :, :]))

        for panel, idx in enumerate(indices):
            ax = axes_flat[panel]
            slices = [slice(None), slice(None), slice(None)]
            slices[slice_axis] = idx
            slice_data = pmf[tuple(slices)]
            if _vmax is not None:
                slice_data = np.clip(slice_data, None, _vmax)

            x_vals = all_centers[free_axes[0]]
            y_vals = all_centers[free_axes[1]]
            X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

            cf = ax.contourf(X, Y, slice_data, levels=levels, cmap=cmap)
            ax.contour(X, Y, slice_data, levels=levels, colors='k',
                       linewidths=0.3, alpha=0.35)
            _cb = fig.colorbar(cf, ax=ax)
            _cb.set_label(f'W ({unit})', fontsize=cbar_label_fontsize)
            _cb.ax.tick_params(labelsize=cbar_tick_fontsize)
            ax.set_xlabel(all_labels[free_axes[0]],
                          fontsize=label_fontsize, fontweight=label_fontweight)
            ax.set_ylabel(all_labels[free_axes[1]],
                          fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title:
                ax.set_title(
                    f'{all_labels[slice_axis]} = {fixed_centers[idx]:.2g}',
                    fontsize=title_fontsize, fontweight=title_fontweight,
                )
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            # Mark the clay surface on whichever free axis is r
            if x_shift > 0:
                if free_axes[0] == 0:
                    ax.axvline(0.0, ls='--', c='grey', lw=0.8, alpha=0.6)
                elif free_axes[1] == 0:
                    ax.axhline(0.0, ls='--', c='grey', lw=0.8, alpha=0.6)

        for panel in range(len(indices), len(axes_flat)):
            axes_flat[panel].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=suptitle_fontsize,
                         fontweight=suptitle_fontweight)
        fig.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig

    def plot_3d_kd_resolved(
        self,
        z_cut=None,
        unit='kJ/mol',
        figsize=(13, 5),
        title=None,
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        title_fontsize=13,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        suptitle_fontsize=13,
        suptitle_fontweight='bold',
        show_grid=True,
        grid_alpha=0.3,
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_3d_kd_resolved.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Coordination-resolved Kd(n_cat) and orientation-resolved Kd(θ).

        Parameters
        ----------
        z_cut : float or None
            Upper r-boundary of bound region (nm).  Auto-detected if None.
        unit, figsize, title : as usual.

        Returns
        -------
        fig, axes
        """
        self._require_wham_3d()
        self.set_default_style()
        p = self.pmf3d

        cat_centers, kd_cat = p.kd_cation_resolved(z_cut)
        theta_centers, kd_th = p.kd_theta_resolved(z_cut)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Kd vs n_cation
        ax = axes[0]
        ax.bar(cat_centers, kd_cat, width=0.7, alpha=0.7, color='steelblue',
               edgecolor='navy')
        ax.set_xlabel(f'n_{p._cation_label}',
                      fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'Kd(n_{p._cation_label})',
                      fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax.set_title('Coordination-Resolved Kd',
                         fontsize=title_fontsize, fontweight=title_fontweight)
        ax.set_yscale('log')
        ax.set_xticks(cat_centers.astype(int))
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha, axis='y')

        # Kd vs theta
        ax = axes[1]
        ax.plot(theta_centers, kd_th, 'r-o', lw=2, markersize=6)
        ax.set_xlabel('Tilt angle θ (°)',
                      fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel('Kd(θ)',
                      fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax.set_title('Orientation-Resolved Kd',
                         fontsize=title_fontsize, fontweight=title_fontweight)
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        if title:
            fig.suptitle(title, fontsize=suptitle_fontsize,
                         fontweight=suptitle_fontweight)
        fig.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, axes

    def plot_3d_coupling(
        self,
        unit='kJ/mol',
        n_cation_indices=None,
        figsize=None,
        cmap='RdBu_r',
        vmax=None,
        title=None,
        r_min=None,
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        title_fontsize=11,
        title_fontweight='bold',
        label_fontsize=10,
        label_fontweight='bold',
        tick_fontsize=9,
        cbar_label_fontsize=10,
        cbar_tick_fontsize=9,
        suptitle_fontsize=12,
        suptitle_fontweight='bold',
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_3d_coupling.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        ΔΔW(r,θ) coupling panels at fixed n_cat values.

        Negative (blue): orientation and cation coordination are positively
        coupled at that (r, θ) position.  Black zero-contour marks statistical
        independence.

        Parameters
        ----------
        n_cation_indices : list of int or None
            Indices into cation_centers to show.  Defaults to all.
        unit, cmap, figsize, title : as usual.

        Returns
        -------
        fig
        """
        self._require_wham_3d()
        self.set_default_style()
        p = self.pmf3d

        ddW_full = p.coupling_free_energy()
        ddW      = p._to_unit(ddW_full, unit)

        if n_cation_indices is None:
            n_cation_indices = list(range(p.n_cation_bins))
        n_panels = len(n_cation_indices)

        ncols = min(n_panels, 4)
        nrows = (n_panels + ncols - 1) // ncols
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        x_shift = p.z_clay_surface or 0.0
        x = (x_shift - p.r_centers) if x_shift > 0 else p.r_centers

        if vmax is not None:
            max_abs = float(vmax)
        elif r_min is not None:
            r_mask = x >= r_min
            max_abs = float(np.nanmax(np.abs(ddW[r_mask, :, :]))) if r_mask.any() else float(np.nanmax(np.abs(ddW)))
        else:
            max_abs = float(np.nanmax(np.abs(ddW)))
        levels  = np.linspace(-max_abs, max_abs, 21)
        xlabel = 'Distance from clay surface (nm)' if x_shift > 0 else 'r = |z| (nm)'
        X, Y = np.meshgrid(x, p.theta_centers, indexing='ij')

        for panel, nidx in enumerate(n_cation_indices):
            ax = axes_flat[panel]
            _data = np.clip(ddW[:, :, nidx], -max_abs, max_abs)
            cf = ax.contourf(X, Y, _data, levels=levels, cmap=cmap)
            ax.contour(X, Y, _data, levels=levels, colors='k',
                       linewidths=0.3, alpha=0.25)
            _cb = fig.colorbar(cf, ax=ax)
            _cb.set_label(f'ΔΔW ({unit})', fontsize=cbar_label_fontsize)
            _cb.ax.tick_params(labelsize=cbar_tick_fontsize)
            ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
            ax.set_ylabel('θ (°)', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title:
                ax.set_title(f'n_{p._cation_label} = {int(p.cation_centers[nidx])}',
                             fontsize=title_fontsize, fontweight=title_fontweight)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            if x_shift > 0:
                ax.axvline(0.0, ls='--', c='grey', lw=0.8, alpha=0.6)

        for panel in range(n_panels, len(axes_flat)):
            axes_flat[panel].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=suptitle_fontsize,
                         fontweight=suptitle_fontweight)
        else:
            fig.suptitle(
                f'Coupling ΔΔW(r,θ,n_{p._cation_label}) = '
                f'W(r,θ,n) − W(r) − W(θ) − W(n)  [{unit}]',
                fontsize=suptitle_fontsize,
                fontweight=suptitle_fontweight,
            )
        fig.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig

    def plot_3d_conditional(
        self,
        fixed_coords_list,
        unit='kJ/mol',
        zero_at='bulk',
        cmap='viridis',
        levels=20,
        vmax=None,
        figsize=None,
        title=None,
        r_min=None,
        # ── Publication-quality font & style controls ──────────────────
        show_title=True,
        title_fontsize=11,
        title_fontweight='bold',
        label_fontsize=10,
        label_fontweight='bold',
        tick_fontsize=9,
        cbar_label_fontsize=10,
        cbar_tick_fontsize=9,
        suptitle_fontsize=13,
        suptitle_fontweight='bold',
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_3d_conditional.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Parameters
        ----------
        fixed_coords_list : list of dict
            Each dict specifies one panel's fixed coordinate, e.g.
            ``[{'r': 0.3}, {'r': 0.5}, {'n_cation': 1}]``.
            Supported keys: ``'r'``, ``'theta'``, ``'n_cation'``,
            ``'r_index'``, ``'theta_index'``, ``'n_cation_index'``.
        unit, cmap, levels, figsize, title : as usual.
        zero_at : 'bulk' (default) or 'min'
            Reference convention — same as other plot methods.

        Returns
        -------
        fig
        """
        self._require_wham_3d()
        self.set_default_style()
        p = self.pmf3d

        n_panels = len(fixed_coords_list)
        ncols = min(n_panels, 4)
        nrows = (n_panels + ncols - 1) // ncols
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        x_shift = p.z_clay_surface or 0.0

        # ── Bulk reference shift (same convention as plot_3d_slices / plot_3d_marginals) ──
        pmf_3d_u = p._to_unit(p.pmf_3d, unit)
        if zero_at == 'bulk':
            n_b = max(1, int(0.2 * p.n_r_bins))
            _bulk_shift = float(np.nanmedian(pmf_3d_u[:n_b, :, :]))
        else:
            _bulk_shift = float(np.nanmin(pmf_3d_u))

        # ── Pre-pass: compute every panel's PMF array to find a shared colour scale ──
        _panel_data = []
        for fixed in fixed_coords_list:
            pmf_cond, ax_labels, ctrs0, ctrs1, fixed_label = p.conditional_pmf(fixed)
            pmf_plot = p._to_unit(pmf_cond, unit) - _bulk_shift
            is_r_ax0 = (ax_labels[0] == 'r (nm)')
            is_r_ax1 = (ax_labels[1] == 'r (nm)')
            if x_shift > 0:
                if is_r_ax0:
                    ctrs0 = x_shift - ctrs0
                    ax_labels[0] = 'Distance from clay surface (nm)'
                if is_r_ax1:
                    ctrs1 = x_shift - ctrs1
                    ax_labels[1] = 'Distance from clay surface (nm)'
                fixed_key = next(iter(fixed))
                if fixed_key in ('r', 'r_index'):
                    r_abs = (float(p.r_centers[int(fixed[fixed_key])])
                             if fixed_key == 'r_index'
                             else float(p.r_centers[int(np.argmin(np.abs(p.r_centers - fixed[fixed_key])))]))
                    fixed_label = f'Distance from surface = {x_shift - r_abs:.3f} nm'
            _panel_data.append((pmf_plot, ax_labels, ctrs0, ctrs1, fixed_label, is_r_ax0, is_r_ax1))

        _global_vmin = float(min(np.nanmin(d[0]) for d in _panel_data))
        _global_vmax = float(max(np.nanmax(d[0]) for d in _panel_data))
        if vmax is not None:
            _global_vmax = float(vmax)
        n_lev = levels if isinstance(levels, int) else len(levels)
        _levels = np.linspace(_global_vmin, _global_vmax, n_lev + 1)

        # ── Plot pass ────────────────────────────────────────────────────────────────
        for panel, (pmf_plot, ax_labels, ctrs0, ctrs1,
                    fixed_label, is_r_ax0, is_r_ax1) in enumerate(_panel_data):
            ax = axes_flat[panel]
            X, Y = np.meshgrid(ctrs0, ctrs1, indexing='ij')
            cf = ax.contourf(X, Y, pmf_plot, levels=_levels, cmap=cmap)
            ax.contour(X, Y, pmf_plot, levels=_levels, colors='k',
                       linewidths=0.3, alpha=0.35)
            _cb = fig.colorbar(cf, ax=ax)
            _cb.set_label(f'W ({unit})', fontsize=cbar_label_fontsize)
            _cb.ax.tick_params(labelsize=cbar_tick_fontsize)
            ax.set_xlabel(ax_labels[0],
                          fontsize=label_fontsize, fontweight=label_fontweight)
            ax.set_ylabel(ax_labels[1],
                          fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title:
                ax.set_title(fixed_label,
                             fontsize=title_fontsize, fontweight=title_fontweight)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            if x_shift > 0:
                if is_r_ax0:
                    ax.axvline(0.0, ls='--', c='grey', lw=0.8, alpha=0.6)
                if is_r_ax1:
                    ax.axhline(0.0, ls='--', c='grey', lw=0.8, alpha=0.6)

        for panel in range(n_panels, len(axes_flat)):
            axes_flat[panel].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=suptitle_fontsize,
                         fontweight=suptitle_fontweight)
        fig.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig

    def plot_3d_overview(
        self,
        unit='kJ/mol',
        zero_at='bulk',
        z_cut=None,
        compare_1d=True,
        suptitle=None,
        figsize=(18, 10),
        # ── Publication-quality font & style controls ──────────────────
        label_fontsize=10,
        label_fontweight='bold',
        tick_fontsize=9,
        title_fontsize=11,
        title_fontweight='bold',
        suptitle_fontsize=14,
        suptitle_fontweight='bold',
        legend_fontsize=9,
        show_grid=True,
        grid_alpha=0.3,
        # ── Export ─────────────────────────────────────────────────────
        save_fig=False,
        filename='pmf_3d_overview.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        3 × 2 summary figure for a 3-D WHAM result.

        ┌──────────────────────┬──────────────────────┬──────────────────────┐
        │  W(r)  marginal      │  W(θ) marginal       │  W(n_cat) marginal   │
        ├──────────────────────┼──────────────────────┤──────────────────────┤
        │  W(r,θ) 2-D slices  │  Kd(n) / Kd(θ)      │  Coupling ΔΔW panel  │
        └──────────────────────┴──────────────────────┴──────────────────────┘

        Row 0: three marginal PMFs (reuses plot_3d_marginals).
        Row 1 left: W(r,θ) 2-D contourf (n_cat summed via marginalize_to_2d).
        Row 1 centre: Kd(n_cat) bar chart.
        Row 1 right: Kd(θ) line chart.

        Parameters
        ----------
        unit : str
        zero_at : str
        z_cut : float or None
        compare_1d : bool
            Overlay 1D WHAM W(r) on the marginal W(r) panel (requires
            ``self.pmf`` to be set).
        suptitle : str or None
        figsize : tuple

        Returns
        -------
        fig, axes  (2 × 3 numpy array)
        """
        self._require_wham_3d()
        self.set_default_style()
        p = self.pmf3d

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # ── Row 0: marginals ──────────────────────────────────────────
        def _prep_r(arr):
            arr = p._to_unit(arr, unit)
            if zero_at == 'bulk':
                n_b = max(1, int(0.2 * len(arr)))
                arr = arr - float(np.nanmedian(arr[:n_b]))
            else:
                arr = arr - np.nanmin(arr)
            return arr

        def _prep(arr):
            arr = p._to_unit(arr, unit)
            return arr - np.nanmin(arr)

        x_shift = p.z_clay_surface or 0.0
        x_r = (x_shift - p.r_centers) if x_shift > 0 else p.r_centers
        xlabel_r = 'Distance from clay (nm)' if x_shift > 0 else 'r = |z| (nm)'

        # W(r)
        ax = axes[0, 0]
        pmf_r = _prep_r(p.pmf_r)
        ax.plot(x_r, pmf_r, 'b-', lw=2, label='W(r) 3D marginal')
        if compare_1d and self.pmf is not None and self.pmf.pmf_abs is not None:
            x_1d = (x_shift - self.pmf.bin_centers_abs) if x_shift > 0 \
                   else self.pmf.bin_centers_abs
            pmf1d = self.pmf.pmf_abs.copy()
            n_b = max(1, int(0.2 * len(pmf1d)))
            pmf1d -= float(np.nanmedian(pmf1d[:n_b]))
            ax.plot(x_1d, pmf1d, 'k--', lw=1.5, alpha=0.65, label='W(r) 1D WHAM')
        if p.pmf_r_std is not None:
            std = p._to_unit(p.pmf_r_std, unit)
            ax.fill_between(x_r, pmf_r - std, pmf_r + std, alpha=0.25)
        ax.set_xlabel(xlabel_r, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'W ({unit})', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_title('W(r) — marginal', fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.legend(fontsize=legend_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
        if x_shift > 0:
            ax.axvline(0.0, ls='--', c='grey', lw=0.8, alpha=0.6)

        # W(θ)
        ax = axes[0, 1]
        pmf_th = _prep(p.pmf_theta)
        ax.plot(p.theta_centers, pmf_th, 'g-', lw=2)
        ax.set_xlabel('Tilt angle θ (°)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'W ({unit})', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_title('W(θ) — marginal', fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        # W(n_cat)
        ax = axes[0, 2]
        pmf_nc = _prep(p.pmf_cation)
        ax.bar(p.cation_centers, pmf_nc, width=0.7, alpha=0.7,
               color='darkorange', edgecolor='saddlebrown')
        ax.set_xlabel(f'n_{p._cation_label}', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'W ({unit})', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_title(f'W(n) — {p._cation_label} coordination',
                     fontsize=title_fontsize, fontweight=title_fontweight)
        ax.set_xticks(p.cation_centers.astype(int))
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha, axis='y')

        # ── Row 1 left: W(r,θ) 2-D contourf ─────────────────────────
        ax = axes[1, 0]
        pmf_2d, ctrs_r, ctrs_th = p.marginalize_to_2d(keep_axes=(0, 1))
        pmf_2d_u = p._to_unit(pmf_2d, unit)
        if zero_at == 'bulk':
            n_b = max(1, int(0.2 * len(ctrs_r)))
            pmf_2d_u -= float(np.nanmedian(pmf_2d_u[:n_b, :]))
        else:
            pmf_2d_u -= np.nanmin(pmf_2d_u)
        x_2d = (x_shift - ctrs_r) if x_shift > 0 else ctrs_r
        X2, Y2 = np.meshgrid(x_2d, ctrs_th, indexing='ij')
        cf = ax.contourf(X2, Y2, pmf_2d_u, levels=20, cmap='viridis')
        ax.contour(X2, Y2, pmf_2d_u, levels=20, colors='k',
                   linewidths=0.3, alpha=0.35)
        _cb_2d = fig.colorbar(cf, ax=ax)
        _cb_2d.set_label(f'W ({unit})', fontsize=label_fontsize)
        ax.set_xlabel(xlabel_r, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel('θ (°)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_title('W(r, θ) — n integrated out',
                     fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if x_shift > 0:
            ax.axvline(0.0, ls='--', c='grey', lw=0.8, alpha=0.6)

        # ── Row 1 centre: Kd(n_cat) ──────────────────────────────────
        ax = axes[1, 1]
        cat_centers, kd_cat = p.kd_cation_resolved(z_cut)
        ax.bar(cat_centers, kd_cat, width=0.7, alpha=0.7,
               color='steelblue', edgecolor='navy')
        ax.set_xlabel(f'n_{p._cation_label}', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'Kd(n_{p._cation_label})', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_title('Coordination-resolved Kd',
                     fontsize=title_fontsize, fontweight=title_fontweight)
        ax.set_xticks(cat_centers.astype(int))
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if np.any(kd_cat > 0):
            ax.set_yscale('log')
        if show_grid:
            ax.grid(True, alpha=grid_alpha, axis='y')

        # ── Row 1 right: Kd(θ) ───────────────────────────────────────
        ax = axes[1, 2]
        theta_centers, kd_th = p.kd_theta_resolved(z_cut)
        ax.plot(theta_centers, kd_th, 'r-o', lw=2, markersize=5)
        ax.set_xlabel('Tilt angle θ (°)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel('Kd(θ)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_title('Orientation-resolved Kd',
                     fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if np.any(kd_th > 0):
            ax.set_yscale('log')
        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        if suptitle is not None:
            fig.suptitle(suptitle, y=1.01,
                         fontsize=suptitle_fontsize, fontweight=suptitle_fontweight)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, axes

    # ------------------------------------------------------------------
    # Umbrella Integration vs WHAM comparison
    # ------------------------------------------------------------------

    def plot_meanforce_vs_wham(
        self,
        clay_mf,
        pmf3d=None,
        figsize=(10, 7),
        unit='kJ/mol',
        color_rfd='mediumseagreen',
        color_rbf='darkorange',
        color_gpr='mediumpurple',
        color_wham='steelblue',
        color_forces='dimgray',
        show_rfd=True,
        show_rbf=True,
        show_gpr=True,
        show_wham=True,
        show_forces=True,
        show_interface=True,
        interface_percentile=80,
        annotate_well=True,
        shade_alpha=0.20,
        title=None,
        show_title=True,
        xlabel=None,
        ylabel=None,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        legend_fontsize=10,
        show_grid=True,
        grid_alpha=0.3,
        surface_coords=False,
        save_fig=False,
        filename='meanforce_vs_wham.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Compare Umbrella Integration PMF estimators (RFD, RBF, GPR) with
        the WHAM 1-D marginal W(r).

        Two panels:

        - **Upper (large)**: PMF comparison — RFD / RBF / GPR ± error
          shading, optionally overlaid with the WHAM marginal W(r).
        - **Lower (small)**: mean-force ⟨F(r)⟩ ± SEM scatter plot.
          Only drawn when ``show_forces=True`` and forces are loaded.

        Parameters
        ----------
        clay_mf : ClayMeanForce
            Must have ``load_forces()`` called; preferably also
            ``run_rfd()``, ``run_rbf()`` / ``run_gpr()`` and
            ``reference_to_bulk()``.
        pmf3d : ClayPMF3D or None
            Source of WHAM W(r).  Falls back to ``self.pmf3d`` if None.
        figsize : tuple
        unit : str
            ``'kJ/mol'``, ``'kcal/mol'``, or ``'kT'``.
        color_rfd, color_rbf, color_gpr, color_wham, color_forces : str
            Line / marker colours.
        show_rfd, show_rbf, show_gpr : bool
            Toggle individual UI estimators.
        show_wham : bool
            Toggle WHAM overlay.
        show_forces : bool
            Draw the lower mean-force panel.
        show_interface : bool
            Draw a vertical dotted line at the interface boundary
            returned by ``clay_mf._get_interface_boundary()``.
        annotate_well : bool
            Append the adsorption ΔG to each method's legend label.
        shade_alpha : float
            Alpha for ± error / uncertainty shading bands.
        title : str or None
        show_title : bool
        xlabel, ylabel : str or None
        title_fontsize, title_fontweight : int / str
        label_fontsize, label_fontweight : int / str
        tick_fontsize, legend_fontsize : int
        show_grid : bool
        grid_alpha : float
        save_fig : bool
        filename : str
        dpi : int
        bbox_inches : str
        transparent_bg : bool

        Returns
        -------
        fig : matplotlib Figure
        axes : numpy.ndarray of Axes
            Shape ``(2,)`` when the force panel is shown, ``(1,)`` otherwise.
        """
        self.set_default_style()

        # ── Resolve WHAM source ───────────────────────────────────────
        # Priority: explicit pmf3d arg → self.pmf3d → self.pmf (ClayPMF aliases
        # r_centers/pmf_r onto bin_centers_abs/pmf_abs so it quacks the same way)
        wham = pmf3d if pmf3d is not None else (
            self.pmf3d if self.pmf3d is not None else self.pmf
        )
        has_wham = (
            show_wham
            and wham is not None
            and getattr(wham, 'pmf_r', None) is not None
            and getattr(wham, 'r_centers', None) is not None
        )

        # ── Unit conversion factor (PMF; forces always kJ mol⁻¹ nm⁻¹) ──
        if unit == 'kJ/mol':
            ufac = 1.0
        elif unit == 'kcal/mol':
            ufac = 1.0 / 4.184
        elif unit in ('kT', 'kBT'):
            _K_B = 8.314462618e-3          # kJ mol⁻¹ K⁻¹
            _T   = float(getattr(clay_mf, 'T', 298.15))
            ufac = 1.0 / (_K_B * _T)
        else:
            raise ValueError(
                f"Unknown unit '{unit}'. Use 'kJ/mol', 'kcal/mol', or 'kT'."
            )

        # ── Determine whether to draw the force panel ─────────────────
        has_forces = (
            show_forces
            and getattr(clay_mf, 'mean_force', None) is not None
            and getattr(clay_mf, 'r_eval',    None) is not None
        )

        # ── Build subplot layout ──────────────────────────────────────
        if has_forces:
            fig, _axes = plt.subplots(
                2, 1, figsize=figsize,
                gridspec_kw={'height_ratios': [3, 1]},
            )
            ax_pmf, ax_mf = _axes
            axes = _axes
        else:
            fig, ax_pmf = plt.subplots(1, 1, figsize=figsize)
            axes = np.array([ax_pmf])

        ax = ax_pmf

        # ── Guard: r_grid must be set before any PMF is drawn ─────────
        r = getattr(clay_mf, 'r_grid', None)
        if r is None:
            ax.text(
                0.5, 0.5,
                'No PMF computed.\n'
                'Call clay_mf.run_rfd(), .run_rbf(), or .run_gpr() first.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='gray',
            )
            plt.tight_layout()
            if save_fig:
                fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                            transparent=transparent_bg)
            return fig, axes

        # ── Interface boundaries (computed once; reused in both panels) ─
        r_iface_neg, r_iface_pos = None, None
        if show_interface:
            try:
                r_iface_neg, r_iface_pos = clay_mf._get_interface_boundaries(
                    percentile=interface_percentile
                )
            except Exception:
                pass

        # ── True Si clay surface position (from GRO / detect_clay_surface) ──
        # Try: pmf3d.z_clay_surface → self.pmf.z_clay_surface → clay_mf._pmf3d
        _z_surf = None
        for _src in (wham,
                     getattr(self, 'pmf', None),
                     getattr(clay_mf, '_pmf3d', None)):
            if _src is not None:
                _z_surf = getattr(_src, 'z_clay_surface', None)
                if _z_surf is not None:
                    _z_surf = float(_z_surf)
                    break

        # ── Coordinate arrays (full signed-r, both surfaces) ──────────
        # Always show the complete PMF: negative r (bottom clay surface) and
        # positive r (top clay surface).  surface_coords is reserved for a
        # future d = z_surf − |r| representation.
        x_arr          = r
        _pmf_ix        = np.arange(len(r))
        _default_xlabel = 'r  (nm)'

        # ── Uncertainty attribute mapping per method ──────────────────
        _err_attr = {
            'rfd': 'pmf_rfd_err',
            'rbf': 'pmf_rbf_err',
            'gpr': 'pmf_gpr_std',
        }

        # ── UI PMF curves (upper panel) ───────────────────────────────
        method_cfg = [
            ('rfd', show_rfd, color_rfd),
            ('rbf', show_rbf, color_rbf),
            ('gpr', show_gpr, color_gpr),
        ]
        for tag, visible, col in method_cfg:
            if not visible:
                continue
            pmf_arr = getattr(clay_mf, f'pmf_{tag}', None)
            if pmf_arr is None:
                continue
            pmf_v = pmf_arr * ufac

            # Optionally annotate ΔG in the legend label
            lbl = f'UI − {tag.upper()}'
            if annotate_well:
                try:
                    dg, _ = clay_mf.adsorption_energy(
                        tag,
                        r_surface_pos=r_iface_pos,
                        r_surface_neg=r_iface_neg,
                    )
                    lbl += f'  (ΔG = {dg * ufac:.1f} {unit})'
                except Exception:
                    pass

            ax.plot(x_arr, pmf_v[_pmf_ix], color=col, lw=2, label=lbl, zorder=3)

            err_arr = getattr(clay_mf, _err_attr[tag], None)
            if err_arr is not None:
                e = err_arr * ufac
                ax.fill_between(x_arr, (pmf_v - e)[_pmf_ix], (pmf_v + e)[_pmf_ix],
                                alpha=shade_alpha, color=col, zorder=2)

        # ── WHAM overlay ──────────────────────────────────────────────
        if has_wham:
            # wham.r_centers is |z|; mirror to the negative side for the full plot
            wham_on_grid = np.interp(np.abs(r), wham.r_centers, wham.pmf_r) * ufac
            # Zero WHAM at the same bulk region as ClayMeanForce.reference_to_bulk()
            frac = float(getattr(clay_mf, 'bulk_fraction', 0.2))
            r_min_g, r_max_g = r.min(), r.max()
            if r_min_g < -0.1 and r_max_g > 0.1:
                # Symmetric: bulk is the centre ±(frac/2)*r_span
                bulk_mask = np.abs(r) <= frac * (r_max_g - r_min_g) / 2.0
            else:
                bulk_mask = r >= r_max_g - frac * (r_max_g - r_min_g)
            if bulk_mask.any():
                wham_on_grid -= float(np.nanmean(wham_on_grid[bulk_mask]))

            wlbl = 'WHAM W(r)'
            if annotate_well:
                try:
                    _w_mask = np.ones(len(r), dtype=bool)
                    if r_iface_pos is not None and np.any(r <= r_iface_pos):
                        _w_mask &= r <= r_iface_pos
                    if r_iface_neg is not None and np.any(r >= r_iface_neg):
                        _w_mask &= r >= r_iface_neg
                    _w_dg = float(wham_on_grid[_w_mask][np.nanargmin(wham_on_grid[_w_mask])])
                    wlbl += f'  (ΔG = {_w_dg:.1f} {unit})'
                except Exception:
                    pass

            ax.plot(x_arr, wham_on_grid, color=color_wham, lw=2.0, ls='--',
                    label=wlbl, zorder=2)

            pmf_r_std = getattr(wham, 'pmf_r_std', None)
            if pmf_r_std is not None:
                wstd = np.interp(np.abs(r), wham.r_centers, pmf_r_std) * ufac
                ax.fill_between(
                    x_arr,
                    wham_on_grid - wstd,
                    wham_on_grid + wstd,
                    alpha=0.15, color=color_wham, zorder=1,
                )

        # ── W = 0 horizontal reference ────────────────────────────────
        ax.axhline(0.0, color='gray', lw=0.8, ls='--', alpha=0.6, zorder=1)

        # ── Vertical reference lines ──────────────────────────────────
        if show_interface:
            # (1) Sampled-range edge — innermost window with repulsive force
            #     This is NOT the clay surface; it is the edge of the
            #     umbrella-sampling data, near the PMF well minimum.
            for _r_if, _side in ((r_iface_neg, '−'), (r_iface_pos, '+')):
                if _r_if is not None:
                    ax.axvline(
                        _r_if, color='C1', lw=1.0, ls='--', alpha=0.8, zorder=1,
                        label=f'Sampling edge ({_side})  r = {_r_if:.2f} nm',
                    )
            # (2) True Si clay surface from atomic coordinates (GRO)
            if _z_surf is not None:
                for _sign, _side in ((-1.0, '−'), (1.0, '+')):
                    ax.axvline(
                        _sign * _z_surf, color='firebrick', lw=1.2, ls=':',
                        alpha=0.85, zorder=1,
                        label=f'Si surface ({_side})  r = {_sign*_z_surf:.2f} nm',
                    )

        # ── Axis decoration (upper panel) ─────────────────────────────
        _xlabel = xlabel if xlabel is not None else _default_xlabel
        _ylabel = ylabel if ylabel is not None else f'W  ({unit})'
        _title  = (title if title is not None
                   else 'Umbrella Integration PMF vs WHAM W(r)')

        if has_forces:
            # x-tick labels suppressed on upper panel; lower panel carries them
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel(
                _xlabel, fontsize=label_fontsize, fontweight=label_fontweight
            )
        ax.set_ylabel(_ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax.set_title(
                _title, fontsize=title_fontsize, fontweight=title_fontweight
            )
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.legend(fontsize=legend_fontsize)
        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        # ── Lower panel: mean forces ───────────────────────────────────
        if has_forces:
            ax_mf.errorbar(
                clay_mf.r_eval,
                clay_mf.mean_force,
                yerr=(clay_mf.force_err
                      if clay_mf.force_err is not None else None),
                fmt='o', ms=4, lw=1.2, capsize=3, elinewidth=0.8,
                color=color_forces, label='⟨F⟩ ± SEM', zorder=3,
            )
            ax_mf.axhline(0.0, color='gray', lw=0.8, ls='--', alpha=0.6)
            for _r_if in (r_iface_neg, r_iface_pos):
                if _r_if is not None:
                    ax_mf.axvline(
                        _r_if, color='C1', lw=1.0, ls='--', alpha=0.8
                    )
            if _z_surf is not None:
                for _sign in (-1.0, 1.0):
                    ax_mf.axvline(
                        _sign * _z_surf, color='firebrick', lw=1.2, ls=':',
                        alpha=0.85
                    )
            ax_mf.set_xlabel(
                _xlabel, fontsize=label_fontsize, fontweight=label_fontweight
            )
            ax_mf.set_ylabel(
                '⟨F⟩  (kJ mol⁻¹ nm⁻¹)',
                fontsize=label_fontsize - 1, fontweight=label_fontweight,
            )
            ax_mf.tick_params(axis='both', labelsize=tick_fontsize)
            ax_mf.legend(fontsize=legend_fontsize)
            if show_grid:
                ax_mf.grid(True, alpha=grid_alpha)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, axes

    # ------------------------------------------------------------------
    # Thermodynamic decomposition plot
    # ------------------------------------------------------------------

    def plot_thermo_decomposition(
        self,
        clay_thermo,
        pmf_r=None,
        pmf_values=None,
        figsize=(11, 13),
        unit='kJ/mol',
        color_dG='steelblue',
        color_dH='firebrick',
        color_mTdS='darkorange',
        color_clay_lj='#2a7d3f',
        color_clay_coul='#66c266',
        color_ion_lj='#7b3fa0',
        color_ion_coul='#c080e0',
        color_water_lj='#1a6fa3',
        color_water_coul='#6bbfdf',
        color_drug='#b87333',
        shade_alpha=0.18,
        show_decomp=True,
        show_entropy=True,
        show_interface=True,
        r_interface=None,
        title=None,
        show_title=True,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        legend_fontsize=9,
        show_grid=True,
        grid_alpha=0.3,
        save_fig=False,
        filename='thermo_decomposition.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Three-panel thermodynamic decomposition figure.

        **Panel 1 – Thermodynamic components** (ΔG, ΔH, −TΔS vs r):
            ΔG from supplied PMF (interpolated to window r_centers);
            ΔH from ClayThermo; −TΔS = ΔG − ΔH.

        **Panel 2 – Interaction decomposition** (ΔH components vs r):
            Clay-LJ, Clay-Coul, Ion-LJ, Ion-Coul, Water-LJ, Water-Coul,
            Drug-Drug (LJ+Coul combined per pair).

        **Panel 3 – Cumulative decomposition** (stacked area plot showing
            how each component contributes to total ΔH).

        Panel 1 is only drawn when both *pmf_r* / *pmf_values* are
        supplied (or ``clay_thermo.dG_interp`` is already populated).
        If neither PMF source is available, Panel 1 is replaced by a
        placeholder.

        Parameters
        ----------
        clay_thermo : ClayThermo
            Must have ``load_energies()`` and ``compute_enthalpy()`` called.
        pmf_r : array-like or None
            PMF r-coordinates (nm).  Used to call
            ``clay_thermo.compute_entropy()`` if not already done.
        pmf_values : array-like or None
            PMF values (kJ/mol) referenced to zero in bulk.
        figsize : tuple
        unit : str  ``'kJ/mol'``, ``'kcal/mol'``, or ``'kT'``.
        color_dG, color_dH, color_mTdS : str
            Colours for the three thermodynamic quantities.
        color_clay_lj, color_clay_coul : str
            Clay–drug LJ and Coulomb component colours.
        color_ion_lj, color_ion_coul : str
            Ion–drug LJ and Coulomb component colours.
        color_water_lj, color_water_coul : str
            Water–drug LJ and Coulomb component colours.
        color_drug : str
            Drug–drug interaction colour.
        shade_alpha : float
            Alpha for error shading bands.
        show_decomp : bool
            If False, panel 2 is hidden (only panels 1 and 3 shown).
        show_entropy : bool
            If False, −TΔS is not drawn even when available.
        show_interface : bool
            Draw a vertical dotted line at *r_interface* (nm).
        r_interface : float or None
            Interface position in nm.  If None, not drawn.
        title : str or None
            Overall figure suptitle.
        show_title : bool
        title_fontsize, title_fontweight : int / str
        label_fontsize, label_fontweight : int / str
        tick_fontsize, legend_fontsize : int
        show_grid, grid_alpha : bool / float
        save_fig : bool
        filename : str
        dpi : int
        bbox_inches : str
        transparent_bg : bool

        Returns
        -------
        fig : matplotlib Figure
        axes : list of Axes  (length 2 or 3)
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        self.set_default_style()

        # ── Unit conversion ───────────────────────────────────────────
        _unit_factors = {'kJ/mol': 1.0, 'kcal/mol': 1.0 / 4.184, 'kT': 1.0 / 2.479}
        uf = _unit_factors.get(unit, 1.0)
        _ylabel = f'Energy  ({unit})'

        # ── Ensure enthalpy is computed ───────────────────────────────
        if clay_thermo.dH is None:
            raise ValueError("clay_thermo.dH is None – call compute_enthalpy() first.")

        r = clay_thermo.r_centers

        # ── Optionally compute entropy if PMF supplied ────────────────
        if pmf_r is not None and pmf_values is not None and clay_thermo.mTdS is None:
            clay_thermo.compute_entropy(pmf_values, pmf_r)

        has_entropy = (
            show_entropy
            and clay_thermo.mTdS is not None
            and clay_thermo.dG_interp is not None
        )

        # ── Layout ────────────────────────────────────────────────────
        n_panels = 3 if show_decomp else 2
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_panels, 1, figure=fig,
                               hspace=0.35,
                               height_ratios=[1.4, 1.2, 1.0][:n_panels])
        axes = [fig.add_subplot(gs[k]) for k in range(n_panels)]
        ax_thermo = axes[0]
        ax_decomp = axes[1] if show_decomp else None
        ax_stack  = axes[-1]

        _xlabel = r'$r$  (nm)'

        def _iface(ax):
            if show_interface and r_interface is not None:
                ax.axvline(r_interface, color='gray', lw=1.0,
                           ls=':', alpha=0.8, zorder=1)

        def _fmt(ax, ylabel=_ylabel):
            ax.set_xlabel(_xlabel, fontsize=label_fontsize,
                          fontweight=label_fontweight)
            ax.set_ylabel(ylabel, fontsize=label_fontsize,
                          fontweight=label_fontweight)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            ax.axhline(0.0, color='gray', lw=0.8, ls='--', alpha=0.5, zorder=1)
            if show_grid:
                ax.grid(True, alpha=grid_alpha)

        def _plot(ax, r, y, yerr, color, label, lw=2.0, ls='-', zorder=3):
            """Plot line with optional error shading."""
            valid = ~(np.isnan(y) | np.isnan(r))
            rv, yv = r[valid] * uf if False else r[valid], y[valid] * uf
            ax.plot(rv, yv, color=color, lw=lw, ls=ls, label=label, zorder=zorder)
            if yerr is not None:
                ev = yerr[valid] * uf
                ax.fill_between(rv, yv - ev, yv + ev,
                                color=color, alpha=shade_alpha, zorder=zorder - 1)

        # ── Panel 1: ΔG / ΔH / −TΔS ──────────────────────────────────
        if has_entropy:
            _plot(ax_thermo, r, clay_thermo.dG_interp, None,
                  color_dG, 'ΔG (PMF)', lw=2.2)
        _plot(ax_thermo, r, clay_thermo.dH, clay_thermo.dH_err,
              color_dH, 'ΔH ± SEM', lw=2.0)
        if has_entropy:
            _plot(ax_thermo, r, clay_thermo.mTdS, clay_thermo.mTdS_err,
                  color_mTdS, '−TΔS ± SEM', lw=2.0, ls='--')
        _iface(ax_thermo)
        _fmt(ax_thermo)
        ax_thermo.legend(fontsize=legend_fontsize, framealpha=0.9)
        ax_thermo.set_title('Thermodynamic components', fontsize=label_fontsize,
                            fontweight=label_fontweight)

        # ── Panel 2: Interaction decomposition ───────────────────────
        if show_decomp and ax_decomp is not None:
            components = [
                (clay_thermo.dH_clay_lj,    clay_thermo.dH_clay_lj_err,    color_clay_lj,   'Clay LJ'),
                (clay_thermo.dH_clay_coul,  clay_thermo.dH_clay_coul_err,  color_clay_coul, 'Clay Coul'),
                (clay_thermo.dH_ion_lj,     clay_thermo.dH_ion_lj_err,     color_ion_lj,    'Ion LJ'),
                (clay_thermo.dH_ion_coul,   clay_thermo.dH_ion_coul_err,   color_ion_coul,  'Ion Coul'),
                (clay_thermo.dH_water_lj,   clay_thermo.dH_water_lj_err,   color_water_lj,  'Water LJ'),
                (clay_thermo.dH_water_coul, clay_thermo.dH_water_coul_err, color_water_coul,'Water Coul'),
            ]
            # Drug–drug: sum LJ + Coul
            if (clay_thermo.dH_drug_lj is not None and
                    clay_thermo.dH_drug_coul is not None):
                dd_sum = clay_thermo.dH_drug_lj + clay_thermo.dH_drug_coul
                dd_err = np.sqrt(clay_thermo.dH_drug_lj_err**2 +
                                 clay_thermo.dH_drug_coul_err**2)
                components.append((dd_sum, dd_err, color_drug, 'Drug–Drug'))

            for y, yerr, col, lbl in components:
                if y is not None:
                    _plot(ax_decomp, r, y, yerr, col, lbl, lw=1.8)
            _iface(ax_decomp)
            _fmt(ax_decomp)
            ax_decomp.legend(fontsize=legend_fontsize, framealpha=0.9,
                             ncol=2, loc='best')
            ax_decomp.set_title('ΔH interaction decomposition',
                                fontsize=label_fontsize,
                                fontweight=label_fontweight)

        # ── Panel 3: Stacked cumulative bar chart ─────────────────────
        # Use bar chart at r_centers to show additive contributions.
        comp_labels  = ['Clay LJ', 'Clay Coul', 'Ion LJ', 'Ion Coul',
                        'Water LJ', 'Water Coul', 'Drug–Drug']
        comp_colors  = [color_clay_lj, color_clay_coul,
                        color_ion_lj,  color_ion_coul,
                        color_water_lj, color_water_coul, color_drug]
        comp_arrays  = [
            clay_thermo.dH_clay_lj,
            clay_thermo.dH_clay_coul,
            clay_thermo.dH_ion_lj,
            clay_thermo.dH_ion_coul,
            clay_thermo.dH_water_lj,
            clay_thermo.dH_water_coul,
        ]
        if (clay_thermo.dH_drug_lj is not None and
                clay_thermo.dH_drug_coul is not None):
            comp_arrays.append(clay_thermo.dH_drug_lj +
                               clay_thermo.dH_drug_coul)
        else:
            comp_arrays.append(None)

        # bar width = 80 % of median spacing
        valid_r = r[~np.isnan(r)]
        bar_width = 0.80 * float(np.median(np.diff(valid_r))) if len(valid_r) > 1 else 0.08

        pos_accum = np.zeros(len(r))
        neg_accum = np.zeros(len(r))

        for lbl, col, arr in zip(comp_labels, comp_colors, comp_arrays):
            if arr is None:
                continue
            vals = arr * uf
            nan_mask = np.isnan(vals) | np.isnan(r)
            pos_part = np.where(~nan_mask & (vals >= 0), vals, 0.0)
            neg_part = np.where(~nan_mask & (vals < 0),  vals, 0.0)

            ax_stack.bar(r, pos_part, bottom=pos_accum,
                         width=bar_width, color=col, label=lbl,
                         alpha=0.82, zorder=2)
            ax_stack.bar(r, neg_part, bottom=neg_accum,
                         width=bar_width, color=col,
                         alpha=0.82, zorder=2)
            pos_accum += pos_part
            neg_accum += neg_part

        # Overlay total ΔH line
        valid = ~np.isnan(clay_thermo.dH) & ~np.isnan(r)
        ax_stack.plot(r[valid], clay_thermo.dH[valid] * uf,
                      color='black', lw=2.0, ls='-',
                      label='Total ΔH', zorder=4)

        _iface(ax_stack)
        _fmt(ax_stack)
        ax_stack.legend(fontsize=legend_fontsize - 1, framealpha=0.9,
                        ncol=3, loc='best')
        ax_stack.set_title('Cumulative ΔH decomposition',
                           fontsize=label_fontsize,
                           fontweight=label_fontweight)

        # ── Suptitle ─────────────────────────────────────────────────
        if show_title:
            _title = title if title is not None else 'Thermodynamic Decomposition'
            fig.suptitle(_title, fontsize=title_fontsize,
                         fontweight=title_fontweight, y=1.01)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, axes

    # ------------------------------------------------------------------
    # Convergence plots  (Step 7 – wraps ClayConvergence results)
    # ------------------------------------------------------------------

    def plot_pmf_convergence(
        self,
        clay_convergence,
        show_cumulative=True,
        show_blocks=True,
        show_full_pmf=True,
        figsize=None,
        unit='kJ/mol',
        cmap_cumulative='Blues',
        cmap_blocks='Reds',
        color_full='black',
        lw_full=2.5,
        lw_curves=1.4,
        alpha_curve_min=0.35,
        shade_block_std=True,
        block_shade_alpha=0.12,
        show_interface=False,
        r_interface=None,
        lc_interface='grey',
        ls_interface='--',
        title=None,
        show_title=True,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        legend_fontsize=9,
        show_grid=True,
        grid_alpha=0.3,
        save_fig=False,
        filename='pmf_convergence.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Two-panel PMF convergence figure.

        **Left panel – Cumulative**: PMF curves at each time fraction
        (20 %, 40 %, … 100 %), colour-coded from light (early) to dark
        (late) using *cmap_cumulative*.

        **Right panel – Block-split spread**: Independent block PMFs
        coloured by block index; ±1σ band shaded.  Full PMF shown as
        a bold black reference line on both panels.

        Parameters
        ----------
        clay_convergence : ClayConvergence
            An analysed ``ClayConvergence`` instance.
        show_cumulative : bool
            Draw the cumulative panel.  Default True.
        show_blocks : bool
            Draw the block-split panel.  Default True.
        show_full_pmf : bool
            Overlay full-data PMF as a reference.  Default True.
        figsize : tuple or None
            Figure (width, height) in inches.  Auto-set if None.
        unit : str
            ``'kJ/mol'`` (default), ``'kcal/mol'``, or ``'kT'``.
        cmap_cumulative : str
            Matplotlib colormap for cumulative curves.  Default ``'Blues'``.
        cmap_blocks : str
            Matplotlib colormap for block curves.  Default ``'Reds'``.
        color_full : str
            Colour of the full-data reference line.  Default ``'black'``.
        lw_full : float
            Line width of the full-data PMF.  Default 2.5.
        lw_curves : float
            Line width of individual subset PMFs.  Default 1.4.
        alpha_curve_min : float
            Minimum alpha (earliest / block-1 curve).  Default 0.35.
        shade_block_std : bool
            Fill ±1σ uncertainty band around block-mean PMF.  Default True.
        block_shade_alpha : float
            Alpha of the σ band.  Default 0.12.
        show_interface : bool
            Draw a vertical dashed line at *r_interface*.  Default False.
        r_interface : float or None
            Clay-surface r position (nm).  Falls back to
            ``self.r_interface`` if None.
        lc_interface, ls_interface : str
            Line colour / style for the interface marker.
        title : str or None
        show_title, title_fontsize, title_fontweight : display options
        label_fontsize, label_fontweight, tick_fontsize, legend_fontsize
        show_grid, grid_alpha
        save_fig, filename, dpi, bbox_inches, transparent_bg

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : list of Axes
        """
        import matplotlib.cm as mplcm

        cc = clay_convergence
        r  = cc.r_bins   # (n_bins,) nm

        # ── Unit conversion ──────────────────────────────────────────
        if unit == 'kT':
            uf = 1.0 / (cc.K_B * cc.T)
        else:
            uf = {'kJ/mol': 1.0, 'kcal/mol': 1.0 / 4.184}.get(unit, 1.0)
        _ylabels = {'kJ/mol':  'ΔG (kJ mol⁻¹)',
                    'kcal/mol': 'ΔG (kcal mol⁻¹)',
                    'kT':       'ΔG (kT)'}
        ylabel = _ylabels.get(unit, f'ΔG ({unit})')

        # ── Determine panels ─────────────────────────────────────────
        has_cumul  = show_cumulative and (cc.cumulative_pmfs is not None)
        has_blocks = show_blocks     and (cc.block_pmfs      is not None)
        n_panels   = int(has_cumul) + int(has_blocks)
        if n_panels == 0:
            raise ValueError(
                "No convergence data available.  Run run_cumulative() or "
                "run_block_split() on the ClayConvergence object first."
            )

        if figsize is None:
            figsize = (6.0 * n_panels, 4.8)

        fig, ax_arr = plt.subplots(1, n_panels, figsize=figsize,
                                   sharey=True, squeeze=False)
        ax_list = list(ax_arr[0])
        pi = 0   # panel index

        # ── Helper ───────────────────────────────────────────────────
        def _draw(ax_, pmf_, label_, color_, lw_, alpha_, zo=2):
            ax_.plot(r, pmf_ * uf, color=color_, lw=lw_,
                     alpha=alpha_, label=label_, zorder=zo)

        def _iface(ax_):
            r_int = r_interface if r_interface is not None else getattr(
                self, 'r_interface', None)
            if r_int is not None:
                ax_.axvline(r_int, color=lc_interface, ls=ls_interface,
                            lw=1.2, label='Interface')

        # ── Panel A – cumulative ──────────────────────────────────────
        if has_cumul:
            ax = ax_list[pi]; pi += 1
            fracs = cc.cumulative_fracs   # (nc,)
            pmfs  = cc.cumulative_pmfs    # (nc, n_bins)
            nc    = len(fracs)
            cmap_c = mplcm.get_cmap(cmap_cumulative)

            for i, (frac, pmf_i) in enumerate(zip(fracs, pmfs)):
                col_i   = cmap_c(0.30 + 0.70 * frac)
                alpha_i = alpha_curve_min + (1.0 - alpha_curve_min) * (
                    i / max(1, nc - 1))
                _draw(ax, pmf_i, f'{frac:.0%}', col_i, lw_curves, alpha_i)

            if show_full_pmf and cc.pmf_full is not None:
                _draw(ax, cc.pmf_full, 'Full', color_full, lw_full, 1.0, zo=3)
            if show_interface:
                _iface(ax)

            ax.set_xlabel('r (nm)', fontsize=label_fontsize,
                          fontweight=label_fontweight)
            ax.set_ylabel(ylabel, fontsize=label_fontsize,
                          fontweight=label_fontweight)
            ax.set_title('Cumulative convergence',
                         fontsize=label_fontsize, fontweight='bold')
            ax.tick_params(labelsize=tick_fontsize)
            ax.legend(title='data fraction', fontsize=legend_fontsize,
                      title_fontsize=legend_fontsize, loc='best')
            if show_grid:
                ax.grid(True, alpha=grid_alpha)

        # ── Panel B – block-split ────────────────────────────────────
        if has_blocks:
            ax = ax_list[pi]
            nb     = cc.n_blocks
            bpmfs  = cc.block_pmfs       # (nb, n_bins)
            b_mean = cc.pmf_block_mean
            b_std  = cc.pmf_block_std
            cmap_b = mplcm.get_cmap(cmap_blocks)

            for bi, pmf_i in enumerate(bpmfs):
                col_i   = cmap_b(0.30 + 0.70 * bi / max(1, nb - 1))
                alpha_i = alpha_curve_min + (1.0 - alpha_curve_min) * (
                    bi / max(1, nb - 1))
                _draw(ax, pmf_i, f'Block {bi + 1}', col_i, lw_curves, alpha_i)

            if shade_block_std and b_mean is not None and b_std is not None:
                ax.fill_between(
                    r,
                    (b_mean - b_std) * uf,
                    (b_mean + b_std) * uf,
                    color=cmap_b(0.65), alpha=block_shade_alpha,
                    label='±1σ', zorder=1,
                )

            if show_full_pmf and cc.pmf_full is not None:
                _draw(ax, cc.pmf_full, 'Full', color_full, lw_full, 1.0, zo=3)
            if show_interface:
                _iface(ax)

            ax.set_xlabel('r (nm)', fontsize=label_fontsize,
                          fontweight=label_fontweight)
            if n_panels == 1:
                ax.set_ylabel(ylabel, fontsize=label_fontsize,
                              fontweight=label_fontweight)
            ax.set_title(f'Block-split spread  ({nb} blocks)',
                         fontsize=label_fontsize, fontweight='bold')
            ax.tick_params(labelsize=tick_fontsize)
            ax.legend(fontsize=legend_fontsize, loc='best')
            if show_grid:
                ax.grid(True, alpha=grid_alpha)

        # ── Suptitle ─────────────────────────────────────────────────
        if show_title:
            _t = title if title is not None else 'PMF Convergence'
            fig.suptitle(_t, fontsize=title_fontsize,
                         fontweight=title_fontweight)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, ax_list

    def plot_convergence_metrics(
        self,
        clay_convergence,
        figsize=(8, 7),
        unit='kJ/mol',
        color_cumulative='steelblue',
        color_blocks='firebrick',
        color_full='black',
        marker_cumulative='o',
        marker_blocks='s',
        show_cumulative=True,
        show_blocks=True,
        show_full_ref=True,
        show_drift=True,
        title=None,
        show_title=True,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        legend_fontsize=9,
        show_grid=True,
        grid_alpha=0.3,
        save_fig=False,
        filename='convergence_metrics.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Adsorption-energy convergence summary (scalar metrics).

        **Top panel – Cumulative**: ΔG_ads (PMF minimum) vs fraction of
        production data; horizontal dashed line at the full-data value.

        **Middle panel – Block-split**: ΔG_ads per independent block as
        a scatter plot; mean ± σ band overlaid.

        **Bottom panel – PMF drift** (optional): |ΔG_last(r) − ΔG_first(r)|
        vs r, showing where in coordinate space the PMF has changed most
        between the first and last cumulative checkpoints.

        Parameters
        ----------
        clay_convergence : ClayConvergence
        show_cumulative : bool
        show_blocks : bool
        show_full_ref : bool
            Horizontal dashed line at full-data ΔG_ads.  Default True.
        show_drift : bool
            Draw PMF-drift panel when cumulative data are present.
            Default True.
        unit : str
            ``'kJ/mol'`` (default), ``'kcal/mol'``, or ``'kT'``.
        color_cumulative, color_blocks, color_full : str
        marker_cumulative, marker_blocks : str
        title, show_title, title_fontsize, title_fontweight
        label_fontsize, label_fontweight, tick_fontsize, legend_fontsize
        show_grid, grid_alpha
        save_fig, filename, dpi, bbox_inches, transparent_bg

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : list of Axes
        """
        from matplotlib.gridspec import GridSpec

        cc = clay_convergence

        # ── Unit conversion ──────────────────────────────────────────
        if unit == 'kT':
            uf = 1.0 / (cc.K_B * cc.T)
        else:
            uf = {'kJ/mol': 1.0, 'kcal/mol': 1.0 / 4.184}.get(unit, 1.0)
        _ads_labels = {
            'kJ/mol':   'ΔG_ads (kJ mol⁻¹)',
            'kcal/mol': 'ΔG_ads (kcal mol⁻¹)',
            'kT':        'ΔG_ads (kT)',
        }
        ads_label = _ads_labels.get(unit, f'ΔG_ads ({unit})')

        # ── Available panels ─────────────────────────────────────────
        has_cumul  = show_cumulative and (cc.cumulative_pmfs  is not None)
        has_blocks = show_blocks     and (cc.block_pmfs       is not None)
        has_drift  = (show_drift and has_cumul
                      and len(cc.cumulative_pmfs) >= 2)

        n_panels = int(has_cumul) + int(has_blocks) + int(has_drift)
        if n_panels == 0:
            raise ValueError(
                "No convergence data to plot.  "
                "Run run_cumulative() or run_block_split() first."
            )

        ae = cc.adsorption_energies()

        # ── GridSpec ─────────────────────────────────────────────────
        heights = (
            ([1.2] if has_cumul  else []) +
            ([1.0] if has_blocks else []) +
            ([0.9] if has_drift  else [])
        )
        fig = plt.figure(figsize=figsize)
        gs  = GridSpec(len(heights), 1, hspace=0.45,
                       height_ratios=heights)
        axes = []
        pi   = 0

        # ── Panel: cumulative ΔG_ads vs fraction ──────────────────────
        if has_cumul:
            ax = fig.add_subplot(gs[pi]); axes.append(ax); pi += 1

            fracs    = cc.cumulative_fracs * 100   # → %
            ea_cumul = ae['cumulative'] * uf

            ax.plot(fracs, ea_cumul, color=color_cumulative, lw=1.8,
                    marker=marker_cumulative, ms=7, label='Cumulative ΔG_ads')

            if show_full_ref and ae['full'] is not None:
                ea_full = ae['full'] * uf
                ax.axhline(ea_full, color=color_full, ls='--', lw=1.5,
                           label=f'Full: {ea_full:.2f}')

            ax.set_xlabel('Cumulative data fraction (%)',
                          fontsize=label_fontsize, fontweight=label_fontweight)
            ax.set_ylabel(ads_label, fontsize=label_fontsize,
                          fontweight=label_fontweight)
            ax.set_title('Convergence of ΔG_ads vs simulation length',
                         fontsize=label_fontsize, fontweight='bold')
            ax.tick_params(labelsize=tick_fontsize)
            ax.legend(fontsize=legend_fontsize)
            if show_grid:
                ax.grid(True, alpha=grid_alpha)

        # ── Panel: block-split ΔG_ads ─────────────────────────────────
        if has_blocks:
            ax = fig.add_subplot(gs[pi]); axes.append(ax); pi += 1

            blocks  = np.arange(1, cc.n_blocks + 1)
            ea_blks = ae['blocks'] * uf
            b_mean  = ae['blocks_mean'] * uf
            b_std   = ae['blocks_std']  * uf

            ax.scatter(blocks, ea_blks, color=color_blocks,
                       marker=marker_blocks, s=60, zorder=3,
                       label='Block ΔG_ads')
            ax.axhline(b_mean, color=color_blocks, ls='-', lw=1.5,
                       label=f'Mean: {b_mean:.2f}')
            ax.axhspan(b_mean - b_std, b_mean + b_std,
                       color=color_blocks, alpha=0.12,
                       label=f'±1σ = {b_std:.2f}')

            if show_full_ref and ae['full'] is not None:
                ea_full = ae['full'] * uf
                ax.axhline(ea_full, color=color_full, ls='--', lw=1.5,
                           label=f'Full: {ea_full:.2f}')

            ax.set_xlabel('Block index', fontsize=label_fontsize,
                          fontweight=label_fontweight)
            ax.set_ylabel(ads_label, fontsize=label_fontsize,
                          fontweight=label_fontweight)
            ax.set_title(f'Block-split ΔG_ads  ({cc.n_blocks} blocks)',
                         fontsize=label_fontsize, fontweight='bold')
            ax.set_xticks(blocks)
            ax.tick_params(labelsize=tick_fontsize)
            ax.legend(fontsize=legend_fontsize)
            if show_grid:
                ax.grid(True, alpha=grid_alpha, axis='y')

        # ── Panel: per-bin PMF drift ──────────────────────────────────
        if has_drift:
            ax = fig.add_subplot(gs[pi]); axes.append(ax)

            drift = cc.drift_metric()   # (n_bins,) kJ/mol
            if unit == 'kT':
                drift_uf = drift / (cc.K_B * cc.T)
            else:
                drift_uf = drift * uf

            _drift_labels = {
                'kJ/mol':   '|ΔG_last − ΔG_first| (kJ mol⁻¹)',
                'kcal/mol': '|ΔG_last − ΔG_first| (kcal mol⁻¹)',
                'kT':       '|ΔG_last − ΔG_first| (kT)',
            }
            drift_ylabel = _drift_labels.get(unit, f'PMF drift ({unit})')

            ax.plot(cc.r_bins, drift_uf, color=color_cumulative, lw=1.6)
            ax.set_xlabel('r (nm)', fontsize=label_fontsize,
                          fontweight=label_fontweight)
            ax.set_ylabel(drift_ylabel, fontsize=label_fontsize,
                          fontweight=label_fontweight)
            ax.set_title('Per-bin PMF drift  (last − first checkpoint)',
                         fontsize=label_fontsize, fontweight='bold')
            ax.tick_params(labelsize=tick_fontsize)
            if show_grid:
                ax.grid(True, alpha=grid_alpha)

        # ── Suptitle ─────────────────────────────────────────────────
        if show_title:
            _t = title if title is not None else 'Convergence Metrics'
            fig.suptitle(_t, fontsize=title_fontsize,
                         fontweight=title_fontweight, y=1.01)

        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, axes

    # ------------------------------------------------------------------
    # MFEP plots  (Step 10 – wraps ClayPath results)
    # ------------------------------------------------------------------

    def plot_mfep_2d(
        self,
        clay_path,
        x_coord='r',
        unit='kJ/mol',
        cmap='viridis',
        levels=20,
        vmax=None,
        r_min=None,
        smooth=False,
        smooth_sigma=1.0,
        zero_at='bulk',
        # Path appearance
        path_color='white',
        path_lw=2.0,
        path_alpha=1.0,
        path_linestyle='-',
        show_start=True,
        show_end=True,
        show_saddle=True,
        marker_start='o',
        marker_end='*',
        marker_saddle='^',
        ms_start=9,
        ms_end=13,
        ms_saddle=10,
        color_start='lime',
        color_end='red',
        color_saddle='yellow',
        show_saddle_energy=True,
        show_path_legend=True,
        figsize=(8, 6),
        title=None,
        show_title=True,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        cbar_label_fontsize=11,
        cbar_tick_fontsize=10,
        legend_fontsize=9,
        show_grid=False,
        grid_alpha=0.3,
        # ── Contour / label controls (forwarded to plot_2d_pmf) ───────────
        show_contour_lines=True,
        contour_linewidth=0.3,
        contour_alpha=0.35,
        show_zero_contour=False,
        zero_contour_color='white',
        zero_contour_lw=1.5,
        xlabel=None,
        ylabel=None,
        n_levels=None,
        save_fig=False,
        filename='mfep_2d.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        2-D PMF heatmap with the MFEP path overlaid.

        Generates the filled-contour background by calling
        :meth:`plot_2d_pmf` internally, then overlays the converged
        string path from *clay_path*.  Start, end, and saddle points
        are marked with configurable symbols.

        Parameters
        ----------
        clay_path : ClayPath
            A ``ClayPath`` instance after ``run_string_method()`` has
            been called.
        x_coord : str
            ``'r'`` → x-axis = r (nm).  ``'dist'`` → x-axis =
            z_clay_surface − r (distance from surface).  Default ``'r'``.
        unit : str
            Energy unit passed to ``plot_2d_pmf``.  Default ``'kJ/mol'``.
        cmap, levels, vmax, r_min, smooth, smooth_sigma, zero_at
            Forwarded to :meth:`plot_2d_pmf`.
        path_color : str
            Line colour of the MFEP path.  Default ``'white'``.
        path_lw : float
            Path line width.  Default 2.0.
        path_alpha : float
            Path line alpha.  Default 1.0.
        path_linestyle : str
            Path line style.  Default ``'-'``.
        show_start, show_end, show_saddle : bool
            Whether to mark each special point.  All default True.
        marker_start, marker_end, marker_saddle : str
            Matplotlib marker codes.  Defaults ``'o'``, ``'*'``, ``'^'``.
        ms_start, ms_end, ms_saddle : float
            Marker sizes.  Defaults 9, 13, 10.
        color_start, color_end, color_saddle : str
            Marker face colours.  Defaults ``'lime'``, ``'red'``,
            ``'yellow'``.
        show_path_legend : bool
            Include path markers in the legend.  Default True.
        figsize, title, show_title, title/label/tick/cbar fontsize params
        show_grid, grid_alpha
        save_fig, filename, dpi, bbox_inches, transparent_bg

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes
        """
        self._require_wham_2d()
        cp = clay_path
        if cp.pmf_path is None:
            raise RuntimeError(
                "clay_path has no MFEP data.  "
                "Call run_string_method() on the ClayPath object first."
            )

        # ── Draw background heatmap ───────────────────────────────────
        _levels = n_levels if n_levels is not None else levels
        fig, ax = self.plot_2d_pmf(
            unit=unit,
            zero_at=zero_at,
            cmap=cmap,
            levels=_levels,
            vmax=vmax,
            r_min=r_min,
            smooth=smooth,
            smooth_sigma=smooth_sigma,
            x_coord=x_coord,
            figsize=figsize,
            title=title or 'MFEP on 2-D PMF: W(r, θ)',
            show_title=show_title,
            title_fontsize=title_fontsize,
            title_fontweight=title_fontweight,
            label_fontsize=label_fontsize,
            label_fontweight=label_fontweight,
            tick_fontsize=tick_fontsize,
            cbar_label_fontsize=cbar_label_fontsize,
            cbar_tick_fontsize=cbar_tick_fontsize,
            show_grid=show_grid,
            grid_alpha=grid_alpha,
            show_contour_lines=show_contour_lines,
            contour_linewidth=contour_linewidth,
            contour_alpha=contour_alpha,
            xlabel=xlabel,
            ylabel=ylabel,
            show_minimum=False,   # suppress built-in minimum marker
            show_legend=False,
            save_fig=False,
        )

        # ── Optional zero contour ─────────────────────────────────────
        if show_zero_contour:
            import numpy as _np
            p_zc = self.pmf2d
            _pmf_zc = p_zc._to_unit(p_zc.pmf_2d, unit)
            _n_bulk = max(1, int(0.2 * p_zc.n_r_bins))
            _pmf_zc -= float(_np.nanmean(_pmf_zc[:_n_bulk, :]))
            _x_shift_zc = p_zc.z_clay_surface or 0.0
            _x_zc = (_x_shift_zc - p_zc.r_centers
                     if x_coord == 'dist' and _x_shift_zc > 0
                     else p_zc.r_centers)
            _Xzc, _Yzc = np.meshgrid(_x_zc, p_zc.theta_centers, indexing='ij')
            ax.contour(_Xzc, _Yzc, _pmf_zc, levels=[0.0],
                       colors=zero_contour_color, linewidths=zero_contour_lw)

        # ── x-axis transform (match plot_2d_pmf convention) ──────────
        p = self.pmf2d
        x_shift = (p.z_clay_surface or 0.0)
        if x_coord == 'dist' and x_shift > 0:
            x_path = x_shift - cp.r_path
        else:
            x_path = cp.r_path

        y_path = cp.theta_path

        # ── Overlay MFEP path ─────────────────────────────────────────
        ax.plot(x_path, y_path,
                color=path_color, lw=path_lw,
                alpha=path_alpha, ls=path_linestyle,
                zorder=5, label='MFEP')

        # ── Special point markers ─────────────────────────────────────
        def _mark(x_val, y_val, marker, ms, col, label_):
            ax.plot(x_val, y_val,
                    marker=marker, ms=ms,
                    color=col, markeredgecolor='k', markeredgewidth=0.6,
                    zorder=6, label=label_, linestyle='none')

        if show_start:
            _mark(x_path[0],  y_path[0],  marker_start,  ms_start,
                  color_start, 'Start')
        if show_end:
            _mark(x_path[-1], y_path[-1], marker_end,    ms_end,
                  color_end,   'End')
        if show_saddle:
            sad = cp.saddle_point()
            x_sad = (x_shift - sad['r']) if (x_coord == 'dist' and x_shift > 0) \
                    else sad['r']
            _sad_label = (f"Saddle ({sad['pmf']:+.1f})"
                          if show_saddle_energy else "Saddle")
            _mark(x_sad, sad['theta'], marker_saddle, ms_saddle,
                  color_saddle, _sad_label)

        if show_path_legend:
            ax.legend(fontsize=legend_fontsize, loc='best',
                      framealpha=0.6)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, ax

    def plot_mfep_profile(
        self,
        clay_path,
        x_axis='arc_length',
        unit='kJ/mol',
        color_path='steelblue',
        lw_path=2.0,
        show_saddle=True,
        color_saddle='firebrick',
        saddle_lw=1.2,
        show_start_end_lines=True,
        color_endpoints='dimgrey',
        endpoint_lw=1.0,
        fill_below=True,
        fill_color='steelblue',
        fill_alpha=0.12,
        fill_gradient=False,
        fill_gradient_colors=None,
        figsize=(7, 4),
        title=None,
        show_title=True,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        legend_fontsize=9,
        show_grid=True,
        grid_alpha=0.3,
        show_legend=True,
        legend_loc='best',
        legend_bbox_to_anchor=None,
        legend_ncol=1,
        legend_fill_alpha=0.0,
        legend_framealpha=1.0,
        xlabel=None,
        ylabel=None,
        zero_at_start=False,
        zero_reference_npoints=1,
        show_thermal_zone=False,
        n_kT=1.0,
        thermal_color='gold',
        thermal_alpha=0.25,
        # ── Local barrier detection ───────────────────────────────────
        show_local_barriers=False,
        barrier_min_kT=1.0,
        barrier_significant_kT=4.0,
        barrier_significant_color='crimson',
        barrier_significant_ms=8,
        barrier_marginal_color='darkorange',
        barrier_marginal_ms=6,
        show_barrier_annotation=False,
        show_barrier_energy_in_legend=True,
        # ── Saddle label offset (dx, dy) in data units; None = auto ─────
        saddle_label_offset=None,
        saddle_label_ha='right',
        # ── Inflection points ────────────────────────────────────
        show_inflection_descent=False,
        inflection_descent_color='mediumorchid',
        inflection_descent_ms=7,
        show_well_curvature=False,
        well_curvature_color='teal',
        well_curvature_ms=7,
        show_well_curvature_annotation=False,
        inflection_smooth_window=5,
        xlim=None,
        ylim=None,
        y_min_from_mfep=True,
        y_min_margin=0.0,
        save_fig=False,
        filename='mfep_profile.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        1-D free energy profile along the MFEP.

        Plots W(s) — PMF as a function of normalised arc-length (or image
        index) — with the saddle point, activation barriers, and
        start/end reference levels annotated.

        Parameters
        ----------
        clay_path : ClayPath
            A ``ClayPath`` instance after ``run_string_method()``.
        x_axis : str
            ``'arc_length'`` (default) — x = normalised cumulative arc-
            length (0 → total path length, dimensionless).  ``'r'`` — x =
            physical r-coordinate of each image in nm (same axis as the
            2-D PMF).  ``'dist'`` — x = distance from clay surface
            (``z_clay_surface − r``); clay at 0, bulk at max, matching
            the ``x_coord='dist'`` convention in ``plot_mfep_2d``.
            ``'index'`` — x = image index.
        unit : str
            ``'kJ/mol'`` (default), ``'kcal/mol'``, or ``'kT'``.
        color_path : str
            Line colour.  Default ``'steelblue'``.
        lw_path : float
            Line width.  Default 2.0.
        show_saddle : bool
            Draw a vertical dashed line at the saddle point.  Default True.
        color_saddle : str
            Colour of the saddle line.  Default ``'firebrick'``.
        saddle_lw : float
            Saddle line width.  Default 1.2.
        show_start_end_lines : bool
            Draw horizontal dashed lines at PMF(start) and PMF(end).
            Default True.
        color_endpoints : str
            Colour of start/end horizontal lines.  Default ``'dimgrey'``.
        endpoint_lw : float
            Width of start/end lines.  Default 1.0.
        fill_below : bool
            Fill area under the PMF curve.  Default True.
        fill_color, fill_alpha : str, float
            Fill appearance.
        figsize, title, show_title, font/tick sizes
        show_grid, grid_alpha
        save_fig, filename, dpi, bbox_inches, transparent_bg

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes
        """
        cp = clay_path
        if cp.pmf_path is None:
            raise RuntimeError(
                "clay_path has no MFEP data.  "
                "Call run_string_method() on the ClayPath object first."
            )

        # ── Unit conversion ───────────────────────────────────────────
        T_ref = getattr(getattr(cp, 'pmf2d', None) or cp, 'T', 298.0)
        _K_B = 8.314462618e-3   # kJ mol⁻¹ K⁻¹
        if unit == 'kT':
            uf = 1.0 / (_K_B * T_ref)
        else:
            uf = {'kJ/mol': 1.0, 'kcal/mol': 1.0 / 4.184}.get(unit, 1.0)
        kT_in_unit = _K_B * T_ref * uf  # 1 k_BT expressed in the current unit
        _ylabels = {
            'kJ/mol':   'W (kJ mol⁻¹)',
            'kcal/mol': 'W (kcal mol⁻¹)',
            'kT':       'W (kT)',
        }
        _default_ylabel = _ylabels.get(unit, f'W ({unit})')

        pmf_plot = cp.pmf_path * uf
        if zero_at_start:
            _n_ref = max(1, min(int(zero_reference_npoints), len(pmf_plot)))
            pmf_plot = pmf_plot - pmf_plot[:_n_ref].mean()

        # ── x coordinate ─────────────────────────────────────────────
        if x_axis == 'arc_length':
            x               = cp.arc_length
            _default_xlabel = 'Normalised arc-length'
        elif x_axis == 'r':
            x               = cp.r_path
            _default_xlabel = 'r (nm)'
        elif x_axis == 'dist':
            _x_shift = float(
                getattr(getattr(self, 'pmf2d', None), 'z_clay_surface', None)
                or getattr(cp, 'z_clay_surface', None)
                or 0.0
            )
            x               = _x_shift - cp.r_path
            _default_xlabel = 'Distance, d (nm)'
        else:
            x               = np.arange(cp.n_images)
            _default_xlabel = 'Image index'

        # ── Figure ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize)

        _tband_hi = None   # set below if show_thermal_zone; used for ylim

        # ── Thermal zone (drawn first so it sits behind everything) ───
        if show_thermal_zone:
            _tband_lo = float(pmf_plot[0])
            _tband_hi = float(pmf_plot[0]) + n_kT * kT_in_unit
            ax.axhspan(_tband_lo, _tband_hi,
                       color=thermal_color, alpha=thermal_alpha, zorder=1,
                       label=(f'{n_kT:.0f} '
                              r'$k_\mathrm{B}T$'
                              f'  ({n_kT * kT_in_unit:.2f} {unit})'))

        if fill_below:
            if fill_gradient:
                import matplotlib.colors as _mcolors
                from matplotlib.patches import PathPatch
                from matplotlib.path import Path as _MplPath
                _grad_colors = (
                    fill_gradient_colors
                    or ['lightcoral', 'lightyellow', 'lightsteelblue']
                )
                _cmap_fill = _mcolors.LinearSegmentedColormap.from_list(
                    '_fill_grad', _grad_colors, N=256)
                _baseline = float(min(pmf_plot))
                _imin = int(np.argmin(pmf_plot))

                _n = len(x)
                # Polygon: curve (top) → drop to baseline at end → baseline →
                # close.  CLOSEPOLY is zero-length because the last LINETO
                # already returns to x[0]; explicit corner prevents the
                # diagonal shortcut that CLOSEPOLY would otherwise take when
                # its coordinates differ from the MOVETO vertex.
                _verts = (list(zip(x, pmf_plot))
                          + [(x[-1], _baseline),   # drop to baseline (end)
                             (x[0],  _baseline),   # baseline back to start x
                             (x[0],  float(pmf_plot[0]))])  # explicit close
                _codes = ([_MplPath.MOVETO]
                          + [_MplPath.LINETO] * (_n - 1)
                          + [_MplPath.LINETO,
                             _MplPath.LINETO,
                             _MplPath.CLOSEPOLY])
                _clip_patch = PathPatch(
                    _MplPath(_verts, _codes),
                    transform=ax.transData,
                    facecolor='none', edgecolor='none',
                )
                # set_clip_on(False) prevents the axes box from clipping the
                # patch boundary — the full polygon is preserved as clip path.
                _clip_patch.set_clip_on(False)
                ax.add_patch(_clip_patch)
                _x0, _x1 = float(min(x)), float(max(x))
                _im = ax.imshow(
                    np.linspace(0, 1, 256).reshape(1, -1),
                    aspect='auto', origin='lower',
                    cmap=_cmap_fill, alpha=fill_alpha,
                    extent=[_x0, _x1, _baseline, float(max(pmf_plot))],
                    zorder=2,
                )
                _im.set_clip_path(_clip_patch)
            else:
                ax.fill_between(x, pmf_plot, min(pmf_plot),
                                color=fill_color, alpha=fill_alpha)

        ax.plot(x, pmf_plot, color=color_path, lw=lw_path, zorder=3)

        # ── Saddle marker ─────────────────────────────────────────────
        if show_saddle:
            sad      = cp.saddle_point()
            x_saddle = (cp.arc_length[sad['image_index']] if x_axis == 'arc_length'
                        else cp.r_path[sad['image_index']] if x_axis == 'r'
                        else x[sad['image_index']] if x_axis == 'dist'
                        else sad['image_index'])
            _saddle_w = float(pmf_plot[sad['image_index']])  # zero_at_start-aware
            ax.axvline(x_saddle, color=color_saddle, ls='--',
                       lw=saddle_lw, zorder=2,
                       label=f"Saddle  W={_saddle_w:+.2f}")

            # Activation energy annotations
            ae = cp.activation_energy()
            _dx_range = x[-1] - x[0]
            _dy_range = pmf_plot.max() - pmf_plot.min()
            if saddle_label_offset is not None:
                _sl_dx, _sl_dy = saddle_label_offset
            else:
                _sl_dx = -0.05 * _dx_range
                _sl_dy =  0.04 * _dy_range
            ax.annotate(
                f"ΔG‡={ae['forward'] * uf:.1f}",
                xy=(x_saddle, _saddle_w),
                xytext=(x_saddle + _sl_dx, _saddle_w + _sl_dy),
                fontsize=legend_fontsize,
                color=color_saddle,
                ha=saddle_label_ha,
            )

        # ── Local barriers ────────────────────────────────────────────
        if show_local_barriers:
            from scipy.signal import find_peaks as _find_peaks
            _min_prom = barrier_min_kT * kT_in_unit
            _sig_prom = barrier_significant_kT * kT_in_unit
            _peaks, _props = _find_peaks(pmf_plot, prominence=_min_prom)
            _proms = _props['prominences']
            _tier_added = {'marginal': False, 'significant': False}
            for _pk, _prom in zip(_peaks, _proms):
                _is_sig   = _prom >= _sig_prom
                _col      = barrier_significant_color if _is_sig else barrier_marginal_color
                _ms       = barrier_significant_ms    if _is_sig else barrier_marginal_ms
                _tier     = 'significant' if _is_sig else 'marginal'
                _tier_lbl = 'Strong'      if _is_sig else 'Marginal'
                if not _tier_added[_tier]:
                    _lbl = (f'{_tier_lbl} barrier  ΔW={_prom:.2f} {unit}'
                            if show_barrier_energy_in_legend
                            else f'{_tier_lbl} barrier')
                    _tier_added[_tier] = True
                else:
                    _lbl = '_nolegend_'
                ax.plot(x[_pk], pmf_plot[_pk],
                        marker='^', ms=_ms, color=_col,
                        markeredgecolor='k', markeredgewidth=0.5,
                        zorder=7, linestyle='none', label=_lbl)
                if show_barrier_annotation:
                    ax.annotate(
                        f'ΔW={_prom:.1f}',
                        xy=(x[_pk], pmf_plot[_pk]),
                        xytext=(x[_pk],
                                pmf_plot[_pk] + 0.35 * kT_in_unit),
                        fontsize=legend_fontsize - 1,
                        color=_col, ha='center', va='bottom',
                    )

        # ── Inflection — steepest descent after saddle ────────────────
        if show_inflection_descent and show_saddle:
            _i_sad  = cp.saddle_point()['image_index']
            _desc_w = pmf_plot[_i_sad:]
            _desc_x = x[_i_sad:]
            if len(_desc_w) > inflection_smooth_window:
                _sw  = max(3, inflection_smooth_window | 1)  # ensure odd
                _sm  = np.convolve(_desc_w,
                                   np.ones(_sw) / _sw,
                                   mode='same')
                _grad      = np.gradient(_sm, _desc_x)
                _i_steep   = int(np.argmax(np.abs(_grad)))
                _i_global  = _i_sad + _i_steep
                ax.plot(x[_i_global], pmf_plot[_i_global],
                        marker='D', ms=inflection_descent_ms,
                        color=inflection_descent_color,
                        markeredgecolor='k', markeredgewidth=0.5,
                        zorder=7, linestyle='none',
                        label='Steepest descent')

        # ── Inflection — well-curvature pair flanking global minimum ───
        if show_well_curvature:
            _sw2 = max(3, inflection_smooth_window | 1)
            _sm2 = np.convolve(pmf_plot,
                               np.ones(_sw2) / _sw2,
                               mode='same')
            _d2  = np.gradient(np.gradient(_sm2, x), x)
            _i_min = int(np.argmin(pmf_plot))
            # Left zero-crossing of d²W/ds²
            _i_left = None
            for _ii in range(_i_min - 1, 0, -1):
                if _d2[_ii] * _d2[_ii - 1] < 0:
                    _i_left = _ii
                    break
            # Right zero-crossing of d²W/ds²
            _i_right = None
            for _ii in range(_i_min, len(_d2) - 1):
                if _d2[_ii] * _d2[_ii + 1] < 0:
                    _i_right = _ii + 1
                    break
            _wc_added = False
            for _i_wc in [_i_left, _i_right]:
                if _i_wc is None:
                    continue
                _lbl = 'Well inflection' if not _wc_added else '_nolegend_'
                _wc_added = True
                ax.plot(x[_i_wc], pmf_plot[_i_wc],
                        marker='s', ms=well_curvature_ms,
                        color=well_curvature_color,
                        markeredgecolor='k', markeredgewidth=0.5,
                        zorder=7, linestyle='none', label=_lbl)
            if (show_well_curvature_annotation
                    and _i_left is not None
                    and _i_right is not None):
                _k_val = float(_d2[_i_min])
                ax.annotate(
                    f'k={_k_val:.2f}',
                    xy=(x[_i_min], pmf_plot[_i_min]),
                    xytext=(x[_i_min],
                            pmf_plot[_i_min] - 0.6 * kT_in_unit),
                    fontsize=legend_fontsize - 1,
                    color=well_curvature_color,
                    ha='center', va='top',
                )

        # ── Start / end reference lines ───────────────────────────────
        if show_start_end_lines:
            _start_y = 0.0 if zero_at_start else float(pmf_plot[0])
            ax.axhline(_start_y,     color=color_endpoints, ls=':',
                       lw=endpoint_lw, label=f'Start  {_start_y:+.2f}')
            ax.axhline(pmf_plot[-1], color=color_endpoints, ls='-.',
                       lw=endpoint_lw, label=f'End    {pmf_plot[-1]:+.2f}')

        # ── Clamp y-top to thermal band (+ tiny margin) ──────────────
        if _tband_hi is not None:
            _data_max = float(pmf_plot.max())
            ax.set_ylim(top=max(_tband_hi, _data_max) + 0.15 * kT_in_unit)

        ax.set_xlabel(xlabel if xlabel is not None else _default_xlabel,
                      fontsize=label_fontsize,
                      fontweight=label_fontweight)
        ax.set_ylabel(ylabel if ylabel is not None else _default_ylabel,
                      fontsize=label_fontsize,
                      fontweight=label_fontweight)
        ax.tick_params(labelsize=tick_fontsize)

        if show_title:
            _t = title if title is not None else 'MFEP Free Energy Profile'
            ax.set_title(_t, fontsize=title_fontsize,
                         fontweight=title_fontweight)

        if show_legend:
            _leg_kw = dict(fontsize=legend_fontsize, loc=legend_loc, ncol=legend_ncol,
                          framealpha=legend_fill_alpha)
            if legend_bbox_to_anchor is not None:
                _leg_kw['bbox_to_anchor'] = legend_bbox_to_anchor
                _leg_kw['borderaxespad'] = 0.0
            _leg = ax.legend(**_leg_kw)
            # set border (edge) alpha independently of fill alpha
            _ec = _leg.get_frame().get_edgecolor()
            _leg.get_frame().set_edgecolor((*_ec[:3], legend_framealpha))
        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        # ── axis limits (dist: always start at 0, matching plot_2d_pmf) ──
        if xlim is not None:
            ax.set_xlim(xlim)
        elif x_axis == 'dist':
            ax.set_xlim(left=0)
        if ylim is not None:
            ax.set_ylim(ylim)
        elif y_min_from_mfep:
            # Use the MFEP endpoint value (pmf_plot[-1]) as the y-axis bottom,
            # not the global minimum which may include near-wall spikes.
            _mfep_end = float(pmf_plot[-1])
            _cur_top  = ax.get_ylim()[1]
            ax.set_ylim(bottom=_mfep_end - float(y_min_margin), top=_cur_top)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, ax

    # ------------------------------------------------------------------
    # MBAR vs WHAM comparison plot
    # ------------------------------------------------------------------

    def plot_mbar_comparison(
        self,
        clay_mbar,
        clay_mf=None,
        show_wham=True,
        show_mbar=True,
        show_ti=True,
        show_error_wham=True,
        show_error_mbar=True,
        show_error_ti=True,
        ti_method='rfd',
        show_interface=True,
        annotate_well=True,
        figsize=(9, 5),
        unit='kJ/mol',
        color_wham='steelblue',
        color_mbar='darkorange',
        color_ti='mediumseagreen',
        shade_alpha=0.20,
        title=None,
        show_title=True,
        xlabel=None,
        ylabel=None,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        legend_fontsize=10,
        show_grid=True,
        grid_alpha=0.3,
        save_fig=False,
        filename='mbar_comparison.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Overlay WHAM W(r), MBAR W(r), and Thermodynamic Integration (TI /
        Umbrella Integration) W(r) on a single axis for direct comparison.

        All PMFs are referenced to zero at the high-r bulk plateau before
        plotting, consistent with ``ClayPMF.get_adsorption_energy()`` and
        ``ClayMeanForce.reference_to_bulk()``.

        Parameters
        ----------
        clay_mbar : ClayMBAR
            Instance with ``run_mbar()`` already called.
            ``reference_to_bulk()`` should have been called beforehand so
            that both curves share the same zero reference.
        clay_mf : ClayMeanForce or None
            Optional.  Instance with ``load_forces()`` and at least one of
            ``run_rfd()``, ``run_rbf()``, ``run_gpr()`` already called.
            An inline high-r bulk referencing is applied automatically so
            the TI curve is co-registered with WHAM and MBAR even if
            ``reference_to_bulk()`` was not called beforehand.
        show_wham : bool
            Include the WHAM PMF.  Requires ``self.pmf`` with ``run_wham()``
            completed.
        show_mbar : bool
            Include the MBAR PMF from ``clay_mbar``.
        show_ti : bool
            Include the TI/UI PMF from ``clay_mf`` (requires ``clay_mf``).
        show_error_wham : bool
            Draw ± bootstrap std shading for WHAM (requires
            ``pmf.pmf_abs_std`` to be set via ``pmf.bootstrap_errors()``).
        show_error_mbar : bool
            Draw ± analytical uncertainty shading for MBAR (requires
            ``clay_mbar.pmf_mbar_1d_err``).
        show_error_ti : bool
            Draw ± uncertainty shading for TI (uses ``pmf_rfd_err``,
            ``pmf_rbf_err``, or ``pmf_gpr_std`` depending on
            ``ti_method``).
        ti_method : str
            Which TI estimator to plot: ``'rfd'`` (regularised finite
            differences), ``'rbf'`` (radial basis function spline),
            ``'gpr'`` (Gaussian process), or ``'all'`` (all three).
            Default ``'rfd'``.
        show_interface : bool
            Draw a vertical dotted line at ``self.pmf.z_clay_surface`` (if
            set).
        annotate_well : bool
            Append ΔG_ads to each legend entry.
        figsize : tuple
        unit : str
            ``'kJ/mol'``, ``'kcal/mol'``, or ``'kT'``.
        color_wham, color_mbar, color_ti : str
        shade_alpha : float
            Alpha for uncertainty shading bands.
        title, show_title, xlabel, ylabel : str / bool
        title_fontsize, title_fontweight, label_fontsize, label_fontweight,
        tick_fontsize, legend_fontsize : style parameters
        show_grid, grid_alpha : bool / float
        save_fig, filename, dpi, bbox_inches, transparent_bg : save options

        Returns
        -------
        fig : matplotlib Figure
        ax  : matplotlib Axes
        """
        self.set_default_style()

        # ── Unit conversion ───────────────────────────────────────────
        _K_B = 8.314462618e-3   # kJ mol⁻¹ K⁻¹
        if unit == 'kJ/mol':
            ufac = 1.0
        elif unit == 'kcal/mol':
            ufac = 1.0 / 4.184
        elif unit in ('kT', 'kBT'):
            _T   = float(getattr(clay_mbar, 'T', 298.15))
            ufac = 1.0 / (_K_B * _T)
        else:
            raise ValueError(
                f"Unknown unit '{unit}'. Use 'kJ/mol', 'kcal/mol', or 'kT'."
            )

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # ── WHAM PMF ─────────────────────────────────────────────────
        if show_wham:
            has_wham = (
                self.pmf is not None
                and getattr(self.pmf, 'pmf_abs', None) is not None
                and getattr(self.pmf, 'bin_centers_abs', None) is not None
            )
            if has_wham:
                r_w   = self.pmf.bin_centers_abs
                pmf_w = self.pmf.pmf_abs.copy()

                # Reference to bulk (low-r pore-centre plateau → 0)
                # Consistent with ClayMBAR.reference_to_bulk() which uses
                # r <= r_min + frac*(r_max - r_min).
                frac  = float(getattr(clay_mbar, 'bulk_fraction', 0.2))
                r_bth = r_w.min() + frac * (r_w.max() - r_w.min())
                bm    = r_w <= r_bth
                if bm.any():
                    pmf_w -= float(np.nanmean(pmf_w[bm]))

                pmf_w_plot = pmf_w * ufac

                wlbl = 'WHAM W(r)'
                if annotate_well:
                    # PMF is already bulk-referenced to 0, so ΔG = min(W(r))
                    dg_w = float(np.nanmin(pmf_w_plot))
                    wlbl += f'  (ΔG = {dg_w:.1f} {unit})'

                ax.plot(r_w, pmf_w_plot,
                        color=color_wham, lw=2.0,
                        label=wlbl, zorder=3)

                if show_error_wham and self.pmf.pmf_abs_std is not None:
                    n   = min(len(r_w), len(self.pmf.pmf_abs_std))
                    e_w = self.pmf.pmf_abs_std[:n] * ufac
                    ax.fill_between(
                        r_w[:n],
                        pmf_w_plot[:n] - e_w,
                        pmf_w_plot[:n] + e_w,
                        alpha=shade_alpha, color=color_wham, zorder=2,
                    )
            else:
                warnings.warn(
                    "WHAM PMF not available.  Set plotter.pmf and call "
                    "pmf.run_wham() to include the WHAM curve.",
                    RuntimeWarning,
                )

        # ── MBAR PMF ─────────────────────────────────────────────────
        if show_mbar and getattr(clay_mbar, 'pmf_mbar_1d', None) is not None:
            r_m   = clay_mbar.bin_centers_abs
            pmf_m_raw = clay_mbar.pmf_mbar_1d.copy()

            # Inline bulk-referencing (same logic as WHAM above)
            frac_m  = float(getattr(clay_mbar, 'bulk_fraction', 0.2))
            r_bth_m = r_m.min() + frac_m * (r_m.max() - r_m.min())
            bm_m    = r_m <= r_bth_m
            if bm_m.any():
                pmf_m_raw -= float(np.nanmean(pmf_m_raw[bm_m]))

            pmf_m = pmf_m_raw * ufac

            mlbl = 'MBAR W(r)'
            if annotate_well:
                # PMF is already bulk-referenced to 0, so ΔG = min(W(r))
                dg_m = float(np.nanmin(pmf_m))
                mlbl += f'  (ΔG = {dg_m:.1f} {unit})'

            ax.plot(r_m, pmf_m,
                    color=color_mbar, lw=2.0, ls='--',
                    label=mlbl, zorder=3)

            if show_error_mbar and clay_mbar.pmf_mbar_1d_err is not None:
                e_m = clay_mbar.pmf_mbar_1d_err * ufac
                ax.fill_between(
                    r_m, pmf_m - e_m, pmf_m + e_m,
                    alpha=shade_alpha, color=color_mbar, zorder=2,
                )

        # ── TI / Umbrella Integration PMF ────────────────────────────
        if show_ti and clay_mf is not None:
            r_ti = getattr(clay_mf, 'r_grid', None)
            if r_ti is not None:
                # colour map for the 'all' case — avoids clash with MBAR orange
                _ti_all_colors = {
                    'rfd': color_ti,
                    'rbf': 'salmon',
                    'gpr': 'mediumpurple',
                }
                _ti_err_attr = {
                    'rfd': 'pmf_rfd_err',
                    'rbf': 'pmf_rbf_err',
                    'gpr': 'pmf_gpr_std',
                }
                _ti_methods = (
                    ['rfd', 'rbf', 'gpr'] if ti_method == 'all'
                    else [ti_method]
                )
                for tag in _ti_methods:
                    pmf_ti_raw = getattr(clay_mf, f'pmf_{tag}', None)
                    if pmf_ti_raw is None:
                        continue
                    pmf_ti = pmf_ti_raw.copy()

                    # Inline high-r bulk referencing so TI is co-registered
                    # with WHAM and MBAR regardless of whether
                    # clay_mf.reference_to_bulk() was called beforehand.
                    frac_ti  = float(getattr(clay_mf, 'bulk_fraction', 0.2))
                    r_bth_ti = r_ti.max() - frac_ti * (r_ti.max() - r_ti.min())
                    bm_ti    = r_ti >= r_bth_ti
                    if bm_ti.any():
                        pmf_ti -= float(np.nanmean(pmf_ti[bm_ti]))

                    pmf_ti_plot = pmf_ti * ufac
                    col_ti = (
                        _ti_all_colors[tag] if ti_method == 'all'
                        else color_ti
                    )
                    suffix = f' \u2212 {tag.upper()}' if ti_method == 'all' else ''
                    tlbl   = f'TI W(r){suffix}'

                    if annotate_well:
                        try:
                            _r_surf_ti = clay_mf._get_interface_boundary()
                        except Exception:
                            _r_surf_ti = None
                        try:
                            dg_ti, _ = clay_mf.adsorption_energy(
                                tag, r_surface=_r_surf_ti
                            )
                            tlbl += f'  (\u0394G = {dg_ti * ufac:.1f} {unit})'
                        except Exception:
                            pass

                    ax.plot(r_ti, pmf_ti_plot,
                            color=col_ti, lw=2.0, ls='-.',
                            label=tlbl, zorder=3)

                    if show_error_ti:
                        err_ti = getattr(clay_mf, _ti_err_attr[tag], None)
                        if err_ti is not None:
                            e_ti = err_ti * ufac
                            ax.fill_between(
                                r_ti,
                                pmf_ti_plot - e_ti,
                                pmf_ti_plot + e_ti,
                                alpha=shade_alpha, color=col_ti, zorder=2,
                            )
            else:
                warnings.warn(
                    "TI PMF not available on clay_mf.  Call "
                    "clay_mf.run_rfd() (or run_rbf()/run_gpr()) first.",
                    RuntimeWarning,
                )

        # ── W = 0 horizontal reference ───────────────────────────────
        ax.axhline(0.0, color='gray', lw=0.8, ls='--', alpha=0.6, zorder=1)

        # ── Clay surface vertical line ────────────────────────────────
        if show_interface and self.pmf is not None:
            z_surf = getattr(self.pmf, 'z_clay_surface', None)
            if z_surf is not None:
                ax.axvline(
                    z_surf, color='saddlebrown', lw=1.0, ls=':',
                    alpha=0.8, zorder=1,
                    label=f'Clay surface  (r = {z_surf:.2f} nm)',
                )

        # ── Axis decoration ───────────────────────────────────────────
        _xlabel = xlabel if xlabel is not None else 'r  (nm)'
        _ylabel = ylabel if ylabel is not None else f'W  ({unit})'
        _title  = (title if title is not None
                   else ('WHAM vs MBAR vs TI W(r)'
                         if clay_mf is not None else 'WHAM vs MBAR W(r)'))

        ax.set_xlabel(
            _xlabel, fontsize=label_fontsize, fontweight=label_fontweight
        )
        ax.set_ylabel(
            _ylabel, fontsize=label_fontsize, fontweight=label_fontweight
        )
        if show_title:
            ax.set_title(
                _title, fontsize=title_fontsize, fontweight=title_fontweight
            )
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.legend(fontsize=legend_fontsize, loc='best')
        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
        return fig, ax

    # ------------------------------------------------------------------
    # MBAR 2-D PMF visualisation
    # ------------------------------------------------------------------

    def plot_mbar_2d(
        self,
        clay_mbar,
        compare_wham=True,
        unit='kJ/mol',
        x_coord='dist',
        cmap='viridis',
        levels=20,
        vmax=None,
        smooth=False,
        smooth_sigma=1.0,
        show_minimum=True,
        figsize=None,
        title=None,
        # --- Font / style controls ---
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=12,
        label_fontweight='bold',
        tick_fontsize=11,
        cbar_label_fontsize=11,
        cbar_tick_fontsize=10,
        show_contour_lines=True,
        contour_linewidth=0.3,
        contour_alpha=0.35,
        show_zero_contour=False,
        zero_contour_color='black',
        zero_contour_lw=1.5,
        minimum_markersize=12,
        show_legend=True,
        legend_fontsize=9,
        legend_fontcolor=None,
        show_grid=False,
        grid_alpha=0.3,
        show_title=True,
        # --- Axis label overrides ---
        xlabel=None,
        ylabel=None,
        # --- Colorbar placement ---
        colorbar_pad=0.02,
        colorbar_width='4%',
        cbar_x=None,            # shift colorbar left/right after placement (e.g. -0.08)
        # --- Export ---
        save_fig=False,
        filename='pmf_mbar_2d.png',
        dpi=300,
        bbox_inches='tight',
        transparent_bg=False,
    ):
        """
        Filled-contour map of the 2-D MBAR PMF W(r, θ).

        When *compare_wham* is ``True`` (default) and ``self.pmf2d`` has a
        WHAM surface available, the figure shows two panels side-by-side:
        left = WHAM W(r, θ), right = MBAR W(r, θ), sharing the same colour
        scale so the two methods are directly comparable.

        When *compare_wham* is ``False``, a single panel is drawn.

        Parameters
        ----------
        clay_mbar : ClayMBAR
            Instance with ``run_mbar_2d()`` already called.
        compare_wham : bool
            Show WHAM vs MBAR side-by-side (default ``True``).
        unit : str
            ``'kJ/mol'``, ``'kcal/mol'``, or ``'kT'``.
        cmap : str
            Matplotlib colormap.  Default ``'viridis'``.
        levels : int
            Number of contour levels.
        vmax : float or None
            Clip upper end of colour scale.  If ``None``, uses the 95th
            percentile of finite PMF values.
        smooth : bool
            Apply Gaussian smoothing before plotting.
        smooth_sigma : float
            Gaussian sigma (in grid units).
        show_minimum : bool
            Mark the global minimum with a red star on each panel.
        figsize : tuple or None
            Figure size.  Defaults to ``(12, 5)`` for 2-panel and
            ``(7, 5)`` for 1-panel.
        title : str or None
            Overrides auto-generated title.

        Returns
        -------
        fig, axes   (axes is a single Axes when compare_wham=False,
                     or a (1, 2) ndarray of Axes when True)
        """
        if clay_mbar.pmf_mbar_2d is None:
            raise RuntimeError(
                "clay_mbar.pmf_mbar_2d is None.  Call "
                "clay_mbar.run_mbar_2d(theta_per_window) first."
            )

        from scipy.ndimage import gaussian_filter

        self.set_default_style()

        # --- Unit conversion factor ---
        _K_B = 8.314462618e-3   # kJ mol⁻¹ K⁻¹
        _T   = float(getattr(clay_mbar, 'T', None) or 298.0)
        _unit_factors = {'kJ/mol': 1.0, 'kcal/mol': 1.0 / 4.184,
                         'kT': 1.0 / (_K_B * _T)}
        ufac = _unit_factors.get(unit, 1.0)

        # --- MBAR surface -------------------------------------------------
        # r_centers_2d layout: index 0 = small r = pore centre = BULK
        #                       index N = large r = clay surface
        # Bulk reference = first 20% of the r array (low-r end).
        _pmf_mbar_raw = clay_mbar.pmf_mbar_2d.copy()
        # Mask bins that were never directly sampled (MBAR extrapolates to
        # unvisited bins, giving finite but meaningless values there).
        _count2d = getattr(clay_mbar, 'count2d_raw', None)
        if _count2d is not None:
            _pmf_mbar_raw[_count2d == 0] = np.nan
        r_mbar_raw = clay_mbar.r_centers_2d
        th_mbar    = clay_mbar.theta_centers_2d
        _n_bulk_m  = max(1, int(0.2 * len(r_mbar_raw)))
        _shift_m   = float(np.nanmean(_pmf_mbar_raw[:_n_bulk_m, :]))
        if np.isfinite(_shift_m):
            _pmf_mbar_raw -= _shift_m
        pmf_mbar = _pmf_mbar_raw * ufac

        # Apply dist transform: x = z_clay_surface - r  (clay at x=0)
        # pmf is stored as clay_mbar._pmf (not .pmf) — check that first,
        # then fall back to self.pmf2d (ClayPMF2D), which does NOT have
        # z_clay_surface, so use r_mbar_raw.max() as last resort.
        _pmf_obj   = getattr(clay_mbar, '_pmf', None)
        _x_shift_v = getattr(_pmf_obj, 'z_clay_surface', None)
        if _x_shift_v is None:
            _x_shift_v = float(r_mbar_raw.max())
        _x_shift = float(_x_shift_v)
        if x_coord == 'dist' and _x_shift > 0:
            r_mbar = _x_shift - r_mbar_raw
        else:
            r_mbar = r_mbar_raw

        if smooth:
            # NaN-aware Gaussian smoothing: preserves the NaN mask so that
            # unsampled bins remain NaN (blank) after smoothing.
            # np.nan_to_num would convert NaN→0 before filtering, which
            # causes the smooth to spread valid values into unsampled regions
            # and removes all NaN → contourf then colors everything.
            _nan_mask = np.isnan(pmf_mbar)
            _filled   = pmf_mbar.copy()
            _filled[_nan_mask] = 0.0
            _smooth_num   = gaussian_filter(_filled, sigma=smooth_sigma)
            _smooth_denom = gaussian_filter((~_nan_mask).astype(float), sigma=smooth_sigma)
            with np.errstate(invalid='ignore'):
                pmf_mbar = np.where(_smooth_denom > 0,
                                    _smooth_num / _smooth_denom,
                                    np.nan)

        # --- WHAM surface (optional) --------------------------------------
        _has_wham = (
            compare_wham
            and self.pmf2d is not None
            and getattr(self.pmf2d, 'pmf_2d', None) is not None
        )
        if _has_wham:
            p2d      = self.pmf2d
            pmf_wham = p2d._to_unit(p2d.pmf_2d, unit).copy()
            r_wham_raw = p2d.r_centers
            th_wham    = p2d.theta_centers
            # Re-reference WHAM to bulk: first 20% of r array = low-r = bulk
            n_bulk = max(1, int(0.2 * p2d.n_r_bins))
            pmf_wham -= float(np.nanmean(pmf_wham[:n_bulk, :]))
            # Apply dist transform
            _x_shift_w = float(p2d.z_clay_surface or 0.0)
            if x_coord == 'dist' and _x_shift_w > 0:
                r_wham = _x_shift_w - r_wham_raw
            else:
                r_wham = r_wham_raw
            if smooth:
                _nan_mask_w = np.isnan(pmf_wham)
                _filled_w   = pmf_wham.copy()
                _filled_w[_nan_mask_w] = 0.0
                _sn_w = gaussian_filter(_filled_w, sigma=smooth_sigma)
                _sd_w = gaussian_filter((~_nan_mask_w).astype(float), sigma=smooth_sigma)
                with np.errstate(invalid='ignore'):
                    pmf_wham = np.where(_sd_w > 0, _sn_w / _sd_w, np.nan)

        # --- Shared colour scale ------------------------------------------
        if vmax is None:
            _finite_mbar = pmf_mbar[np.isfinite(pmf_mbar)]
            vmax = float(np.percentile(_finite_mbar, 95)) if _finite_mbar.size else 1.0
            if _has_wham:
                _finite_wham = pmf_wham[np.isfinite(pmf_wham)]
                if _finite_wham.size:
                    vmax = max(vmax, float(np.percentile(_finite_wham, 95)))

        vmin_val = None  # let contourf auto-set lower bound

        # --- Figure layout ------------------------------------------------
        n_panels = 2 if _has_wham else 1
        if figsize is None:
            figsize = (13, 5) if n_panels == 2 else (7, 5)

        if n_panels == 2:
            fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        else:
            fig, axes = plt.subplots(figsize=figsize)

        # --- Helper: draw one panel --------------------------------------
        def _draw_panel(ax, pmf_surf, r_arr, th_arr, panel_title):
            pmf_plot = np.ma.masked_invalid(np.clip(pmf_surf, vmin_val, vmax))
            X, Y = np.meshgrid(r_arr, th_arr, indexing='ij')
            cf = ax.contourf(X, Y, pmf_plot, levels=levels, cmap=cmap,
                             vmin=None, vmax=vmax)
            if show_contour_lines:
                ax.contour(X, Y, pmf_plot, levels=levels, colors='k',
                           linewidths=contour_linewidth, alpha=contour_alpha)
            if show_zero_contour:
                ax.contour(X, Y, np.ma.masked_invalid(pmf_surf), levels=[0.0],
                           colors=[zero_contour_color],
                           linewidths=zero_contour_lw)
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size=colorbar_width, pad=colorbar_pad)
            cbar = fig.colorbar(cf, cax=cax)
            cbar.set_label(f'W(r, θ)  ({unit})', fontsize=cbar_label_fontsize)
            cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
            import matplotlib.ticker as _ticker
            cbar.ax.yaxis.set_major_formatter(_ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
            if cbar_x is not None:
                pos = cbar.ax.get_position()
                cbar.ax.set_position([pos.x0 + cbar_x, pos.y0, pos.width, pos.height])
            _default_xlab = 'Distance from clay surface (nm)' if x_coord == 'dist' else 'r  (nm)'
            _default_ylab = 'Tilt angle θ (degrees)'
            ax.set_xlabel(xlabel if xlabel is not None else _default_xlab,
                          fontsize=label_fontsize, fontweight=label_fontweight)
            ax.set_ylabel(ylabel if ylabel is not None else _default_ylab,
                          fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title:
                ax.set_title(panel_title, fontsize=title_fontsize,
                             fontweight=title_fontweight)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            # Always show a tick at x=0; keep all other ticks within the data range.
            # IMPORTANT: ax.get_xticks() includes ticks outside xlim — we must
            # filter first, otherwise ax.set_xticks() expands xlim to fit them all.
            import matplotlib.ticker as _mticker
            _xl, _xr = ax.get_xlim()
            _visible_ticks = [t for t in ax.get_xticks() if _xl <= t <= _xr]
            if 0.0 not in _visible_ticks:
                _visible_ticks = sorted(set(_visible_ticks + [0.0]))
                if _xl > 0.0:
                    ax.set_xlim(left=0.0)   # snap left edge to 0, right unchanged
            ax.set_xticks(_visible_ticks)
            ax.xaxis.set_major_formatter(_mticker.FuncFormatter(
                lambda x, _: '0' if x == 0 else f'{x:.2g}'
            ))
            if show_grid:
                ax.grid(True, alpha=grid_alpha)
            if show_minimum:
                valid = np.isfinite(pmf_surf)
                if valid.any():
                    tmp = pmf_surf.copy()
                    tmp[~valid] = np.inf
                    idx = np.unravel_index(np.argmin(tmp), tmp.shape)
                    ax.plot(
                        r_arr[idx[0]], th_arr[idx[1]],
                        'r*', markersize=minimum_markersize,
                        label=(
                            f'Min: r={r_arr[idx[0]]:.2f} nm, '
                            f'θ={th_arr[idx[1]]:.0f}°'
                        ),
                    )
                    if show_legend:
                        leg = ax.legend(fontsize=legend_fontsize, loc='best')
                        leg.get_frame().set_alpha(0.0)
                        if legend_fontcolor is not None:
                            for _lt in leg.get_texts():
                                _lt.set_color(legend_fontcolor)

        # --- Draw panel(s) -----------------------------------------------
        if _has_wham:
            _draw_panel(axes[0], pmf_wham, r_wham, th_wham,
                        title or 'WHAM  W(r, θ)')
            _draw_panel(axes[1], pmf_mbar, r_mbar, th_mbar,
                        'MBAR  W(r, θ)')
        else:
            _draw_panel(axes, pmf_mbar, r_mbar, th_mbar,
                        title or 'MBAR  W(r, θ)')

        if title is not None and _has_wham:
            fig.suptitle(title, fontsize=title_fontsize + 1,
                         fontweight=title_fontweight)

        plt.tight_layout()
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
            print(f"Figure saved: {filename}")

        return fig, axes

    # ------------------------------------------------------------------
    # Convenience: save figure
    # ------------------------------------------------------------------

    @staticmethod
    def save_figure(fig, path, dpi=300):
        """
        Save a figure to disk.

        Parameters
        ----------
        fig : matplotlib Figure
        path : str
            Output file path (extension determines format).
        dpi : int
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        pmf_status = 'no PMF' if (self.pmf is None or self.pmf.pmf_abs is None) \
                     else f'PMF ready ({len(self.pmf.bin_centers_abs)} bins)'
        pmf2d_status = 'no 2D PMF' \
            if (self.pmf2d is None or self.pmf2d.pmf_2d is None) \
            else (f'2D PMF ready '
                  f'({self.pmf2d.n_r_bins}r × {self.pmf2d.n_theta_bins}θ bins)')
        pmf3d_status = 'no 3D PMF' \
            if (self.pmf3d is None or self.pmf3d.pmf_3d is None) \
            else (f'3D PMF ready '
                  f'({self.pmf3d.n_r_bins}r × {self.pmf3d.n_theta_bins}θ '
                  f'× {self.pmf3d.n_cation_bins}n bins)')
        return f"ClayPMFPlotter({pmf_status} | {pmf2d_status} | {pmf3d_status})"
