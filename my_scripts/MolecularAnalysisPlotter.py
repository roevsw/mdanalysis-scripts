"""
MolecularAnalysisPlotter.py

Separate plotting class for MolecularAnalysis results.
Handles all visualization and plotting functionality.

Author: R.Swai
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import functools
import os
from datetime import datetime


class MolecularAnalysisPlotter:
    """
    Plotting class for MolecularAnalysis results.
    
    This class handles visualization of:
    - Radial distribution functions (RDFs)
    - Ion binding analysis
    - Ion competition analysis
    - Coordination analysis
    - Time series data
    
    Parameters
    ----------
    analysis : MolecularAnalysis
        Instance of MolecularAnalysis class (optional)
    
    Examples
    --------
    >>> from MolecularAnalysis import MolecularAnalysis
    >>> from MolecularAnalysisPlotter import MolecularAnalysisPlotter
    >>> 
    >>> analysis = MolecularAnalysis('system.tpr', 'traj.xtc')
    >>> plotter = MolecularAnalysisPlotter(analysis)
    >>> 
    >>> # Calculate RDF
    >>> rdf_results = analysis.molecular_rdf('resname LIG', 'resname SOL')
    >>> 
    >>> # Plot RDF
    >>> plotter.plot_rdf(rdf_results, title='Ligand-Water RDF')
    """
    
    def __init__(self, analysis=None):
        """
        Initialize plotter
        
        Parameters
        ----------
        analysis : MolecularAnalysis, optional
            MolecularAnalysis instance for accessing system info
        """
        self.analysis = analysis
        
        # Default plotting style
        self.set_default_style()
    
    def set_default_style(self):
        """Set default matplotlib style for publication-quality plots"""
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['figure.titlesize'] = 14
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['axes.linewidth'] = 1.2
    
    # =========================================================================
    # RDF PLOTTING METHODS
    # =========================================================================
    
    def plot_rdf(self, rdf_results, label=None, title='Radial Distribution Function',
                 xlabel='Distance (Å)', ylabel='g(r)', xlim=None, ylim=None,
                 color='blue', linewidth=2, show_peaks=True, peak_threshold=0.1,
                 save_fig=False, filename='rdf.png', dpi=300, figsize=(8, 6)):
        """
        Plot a single RDF
        
        Parameters
        ----------
        rdf_results : object or dict
            RDF results from molecular_rdf() method. Should have .bins and .rdf attributes
            or be a dict with 'bins' and 'rdf' keys
        label : str, optional
            Label for the RDF line
        title : str
            Plot title
        xlabel, ylabel : str
            Axis labels
        xlim, ylim : tuple, optional
            Axis limits
        color : str
            Line color
        linewidth : float
            Line width
        show_peaks : bool
            Whether to mark peak positions
        peak_threshold : float
            Minimum peak height relative to max
        save_fig : bool
            Whether to save figure
        filename : str
            Output filename if saving
        dpi : int
            Resolution for saved figure
        figsize : tuple
            Figure size (width, height) in inches
        
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        """
        
        # Extract bins and rdf values
        if hasattr(rdf_results, 'bins'):
            bins = rdf_results.bins
            rdf = rdf_results.rdf
        elif isinstance(rdf_results, dict):
            bins = rdf_results['bins']
            rdf = rdf_results['rdf']
        else:
            raise ValueError("rdf_results must have .bins and .rdf attributes or be a dict")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot RDF
        ax.plot(bins, rdf, color=color, linewidth=linewidth, label=label)
        
        # Find and mark peaks if requested
        if show_peaks:
            peaks, properties = find_peaks(rdf, height=np.max(rdf) * peak_threshold)
            if len(peaks) > 0:
                ax.plot(bins[peaks], rdf[peaks], 'ro', markersize=8, 
                       label=f'Peaks ({len(peaks)})', zorder=5)
                
                # Annotate first peak
                if len(peaks) > 0:
                    first_peak = peaks[0]
                    ax.annotate(f'{bins[first_peak]:.2f} Å\ng(r)={rdf[first_peak]:.2f}',
                               xy=(bins[first_peak], rdf[first_peak]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=10, ha='left',
                               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Labels and title
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set limits if provided
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        
        # Grid and legend
        ax.grid(True, alpha=0.3, linestyle='--')
        if label or show_peaks:
            ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save if requested
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✓ Figure saved: {filename}")
        
        return fig, ax
    
    def plot_multiple_rdfs(self, rdf_dict, title='RDF Comparison',
                          xlabel='Distance (Å)', ylabel='g(r)',
                          xlim=None, ylim=None, 
                          # Color control
                          colors=None, colormap='tab10',
                          # Line styling
                          linewidth=2, linestyles=None, line_alpha=1.0,
                          markers=None, markersize=6,
                          # Font & text control
                          title_fontsize=14, title_text=None, show_title=True,
                          title_fontweight='normal',
                          label_fontsize=12, label_fontweight='normal',
                          tick_fontsize=10, legend_fontsize=10, legend_fontweight='normal',
                          # Legend control
                          show_legend=True, legend_loc='best', legend_framealpha=0.9,
                          legend_ncol=1, custom_labels=None,
                          # Grid control
                          show_grid=True, grid_alpha=0.3, grid_linestyle='--',
                          # Fill/shading options
                          fill_under_curves=False, fill_alpha=0.2,
                          # Axis formatting
                          show_minor_ticks=False, tick_direction='out',
                          # Figure export control
                          save_fig=False, filename='rdf_comparison.png',
                          dpi=300, figsize=(10, 6), bbox_inches='tight',
                          transparent_bg=False):
        """
        Plot multiple RDFs on the same axes for comparison with extensive formatting control
        
        Parameters
        ----------
        rdf_dict : dict
            Dictionary of RDF results: {label: rdf_results}
        title : str
            Plot title (default: 'RDF Comparison')
        xlabel, ylabel : str
            Axis labels
        xlim, ylim : tuple, optional
            Axis limits (e.g., (0, 10) for xlim)
        
        Color Control
        -------------
        colors : list or dict, optional
            Colors for each RDF. Can be:
            - List: colors applied in order ['blue', 'red', 'green']
            - Dict: colors mapped by label {'Quinolone': 'red', 'Piperazine': 'blue'}
            If None, uses colormap
        colormap : str
            Matplotlib colormap name for auto-generating colors (default: 'tab10')
        
        Line Styling
        ------------
        linewidth : float or list
            Line width(s). Single value or list per RDF (default: 2)
        linestyles : list, optional
            List of line styles: '-', '--', '-.', ':' (default: None, all solid)
        line_alpha : float
            Line transparency 0-1 (default: 1.0)
        markers : list, optional
            List of marker styles: 'o', 's', '^', 'v', 'd' (default: None)
        markersize : float
            Marker size (default: 6)
        
        Font & Text Control
        -------------------
        title_fontsize : float
            Title font size (default: 14)
        title_text : str, optional
            Custom title override (default: None, uses 'title' parameter)
        show_title : bool
            Whether to show title (default: True)
        title_fontweight : str
            Title font weight: 'normal', 'bold', 'light', 'heavy' (default: 'normal')
        label_fontsize : float
            Axis label font size (default: 12)
        label_fontweight : str
            Axis label font weight: 'normal', 'bold', 'light', 'heavy' (default: 'normal')
        tick_fontsize : float
            Tick label font size (default: 10)
        legend_fontsize : float
            Legend font size (default: 10)
        legend_fontweight : str
            Legend font weight: 'normal', 'bold', 'light', 'heavy' (default: 'normal')
        
        Legend Control
        --------------
        show_legend : bool
            Whether to show legend (default: True)
        legend_loc : str
            Legend location: 'best', 'upper right', 'lower left', etc. (default: 'best')
        legend_framealpha : float
            Legend background transparency (default: 0.9)
        legend_ncol : int
            Number of legend columns (default: 1)
        custom_labels : dict, optional
            Custom labels for legend. Maps rdf_dict keys to display names.
            Use mathtext for superscripts: {'quinolone-NA': r'Quinolone-Na$^+$'}
            If None, uses rdf_dict keys as labels (default: None)
        
        Grid Control
        ------------
        show_grid : bool
            Whether to show grid (default: True)
        grid_alpha : float
            Grid transparency (default: 0.3)
        grid_linestyle : str
            Grid line style: '-', '--', '-.', ':' (default: '--')
        
        Fill/Shading Options
        --------------------
        fill_under_curves : bool
            Whether to fill area under RDF curves (default: False)
        fill_alpha : float
            Fill transparency (default: 0.2)
        
        Axis Formatting
        ---------------
        show_minor_ticks : bool
            Whether to show minor tick marks (default: False)
        tick_direction : str
            Tick direction: 'in', 'out', or 'inout' (default: 'out')
        
        Figure Export Control
        ---------------------
        save_fig : bool
            Whether to save figure (default: False)
        filename : str
            Output filename (default: 'rdf_comparison.png')
        dpi : int
            Resolution for saved figure (default: 300)
        figsize : tuple
            Figure size in inches (width, height) (default: (10, 6))
        bbox_inches : str
            Bounding box for saved figure (default: 'tight')
        transparent_bg : bool
            Whether to save with transparent background (default: False)
        
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        
        Examples
        --------
        >>> # Basic usage
        >>> rdf_dict = {'Na-Water': rdf1, 'K-Water': rdf2}
        >>> plotter.plot_multiple_rdfs(rdf_dict)
        
        >>> # Advanced styling
        >>> plotter.plot_multiple_rdfs(
        ...     rdf_dict,
        ...     linestyles=['-', '--', '-.'],
        ...     linewidth=2.5,
        ...     markers=['o', 's', '^'],
        ...     fill_under_curves=True,
        ...     legend_ncol=2,
        ...     save_fig=True,
        ...     filename='publication_rdf.png',
        ...     dpi=600
        ... )
        
        >>> # Custom legend labels with superscripts
        >>> plotter.plot_multiple_rdfs(
        ...     rdf_dict,
        ...     custom_labels={
        ...         'quinolone-NA': r'Quinolone-Na$^+$',
        ...         'quinolone-K': r'Quinolone-K$^+$',
        ...         'piperazine-NA': r'Piperazine-Na$^+$'
        ...     },
        ...     save_fig=True
        ... )
        """
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate colors if not provided
        if colors is None:
            cmap = plt.cm.get_cmap(colormap)
            colors = [cmap(i % cmap.N) for i in range(len(rdf_dict))]
        elif isinstance(colors, dict):
            # Convert dictionary to list in the order of rdf_dict
            # Support partial matching for keys like 'quinolone' matching 'quinolone-water_oxygen'
            converted_colors = []
            for label in rdf_dict.keys():
                # Try exact match first
                if label in colors:
                    converted_colors.append(colors[label])
                else:
                    # Try partial match (check if any color key is in the label)
                    matched = False
                    for color_key, color_val in colors.items():
                        if color_key in label:
                            converted_colors.append(color_val)
                            matched = True
                            break
                    if not matched:
                        converted_colors.append('black')  # Default fallback
            colors = converted_colors
        
        # Prepare line styles and markers
        n_curves = len(rdf_dict)
        if linestyles is None:
            linestyles = ['-'] * n_curves
        elif len(linestyles) < n_curves:
            # Cycle through provided styles
            linestyles = (linestyles * (n_curves // len(linestyles) + 1))[:n_curves]
        
        if markers is not None and len(markers) < n_curves:
            # Cycle through provided markers
            markers = (markers * (n_curves // len(markers) + 1))[:n_curves]
        
        # Handle linewidth as list or scalar
        if isinstance(linewidth, (int, float)):
            linewidths = [linewidth] * n_curves
        else:
            linewidths = linewidth if len(linewidth) >= n_curves else (linewidth * (n_curves // len(linewidth) + 1))[:n_curves]
        
        # Plot each RDF
        for idx, (label, rdf_results) in enumerate(rdf_dict.items()):
            # Extract data
            if hasattr(rdf_results, 'bins'):
                bins = rdf_results.bins
                rdf = rdf_results.rdf
            else:
                bins = rdf_results['bins']
                rdf = rdf_results['rdf']
            
            # Get styling for this curve
            color = colors[idx] if isinstance(colors, (list, np.ndarray)) else colors
            ls = linestyles[idx]
            lw = linewidths[idx]
            marker = markers[idx] if markers is not None else None
            
            # Get display label (use custom if provided)
            display_label = custom_labels.get(label, label) if custom_labels else label
            
            # Plot line
            ax.plot(bins, rdf, label=display_label, color=color, linewidth=lw,
                   linestyle=ls, alpha=line_alpha,
                   marker=marker, markersize=markersize, 
                   markevery=max(1, len(bins)//20) if marker else None)
            
            # Fill under curve if requested
            if fill_under_curves:
                ax.fill_between(bins, rdf, alpha=fill_alpha, color=color)
        
        # Title
        if show_title:
            title_to_use = title_text if title_text is not None else title
            ax.set_title(title_to_use, fontsize=title_fontsize, fontweight=title_fontweight)
        
        # Axis labels
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        
        # Tick parameters
        ax.tick_params(axis='both', labelsize=tick_fontsize, direction=tick_direction)
        if show_minor_ticks:
            ax.minorticks_on()
            ax.tick_params(which='minor', direction=tick_direction)
        
        # Legend
        if show_legend:
            legend = ax.legend(loc=legend_loc, framealpha=legend_framealpha, 
                              fontsize=legend_fontsize, ncol=legend_ncol)
            # Set legend text font weight
            for text in legend.get_texts():
                text.set_fontweight(legend_fontweight)
        
        # Grid
        if show_grid:
            ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
        
        # Axis limits
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        
        plt.tight_layout()
        
        # Save figure
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, 
                       transparent=transparent_bg)
            print(f"✓ Figure saved: {filename}")
        
        return fig, ax
    
    def plot_rdf_grid(self, rdf_dict, ncols=2, title='RDF Collection',
                     xlim=None, ylim=None, save_fig=False, 
                     filename='rdf_grid.png', dpi=300, figsize=None):
        """
        Plot multiple RDFs in a grid layout
        
        Parameters
        ----------
        rdf_dict : dict
            Dictionary of RDF results: {label: rdf_results}
        ncols : int
            Number of columns in grid
        title : str
            Overall title
        xlim, ylim : tuple, optional
            Axis limits for all subplots
        save_fig : bool
            Whether to save figure
        filename : str
            Output filename
        dpi : int
            Resolution
        figsize : tuple, optional
            Figure size (auto-calculated if None)
        
        Returns
        -------
        fig, axes : matplotlib figure and axes array
        """
        
        n_plots = len(rdf_dict)
        nrows = int(np.ceil(n_plots / ncols))
        
        if figsize is None:
            figsize = (6 * ncols, 5 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.array(axes).flatten()  # Ensure axes is always 1D
        
        # Plot each RDF
        for i, (label, rdf_results) in enumerate(rdf_dict.items()):
            ax = axes[i]
            
            # Extract data
            if hasattr(rdf_results, 'bins'):
                bins = rdf_results.bins
                rdf = rdf_results.rdf
            else:
                bins = rdf_results['bins']
                rdf = rdf_results['rdf']
            
            # Plot
            ax.plot(bins, rdf, linewidth=2, color='blue')
            ax.set_xlabel('Distance (Å)', fontsize=11)
            ax.set_ylabel('g(r)', fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✓ Figure saved: {filename}")
        
        return fig, axes
    
    def plot_rdf_with_shells(self, rdf_dict, boundaries,
                            # Layout control
                            ncols=2, figsize_per_plot=(8, 6), figsize=None,
                            # RDF line styling
                            rdf_color='black', rdf_linewidth=2,
                            linestyles=None, markers=None, markersize=6,
                            # Shell visualization
                            shell_alpha=0.4, shell_colors=None, colormap='default',
                            color_shade_style=None, shell_naming_style=None,
                            show_shell_labels=True, shell_label_fontsize=10,
                            show_boundary_lines=True, boundary_linestyle='--',
                            boundary_alpha=0.7, boundary_linewidth=1.5,
                            # RCN overlay control
                            show_rcn=False, rcn_data=None, rcn_color='red',
                            rcn_linewidth=1.5, rcn_linestyle='--',
                            rcn_scale_factor=4.0, rcn_ylabel='RCN(#)',
                            # Font & text control
                            title_fontsize=22, title_fontweight='bold',
                            show_title=True, title_text=None,
                            label_fontsize=22, label_fontweight='bold',
                            tick_fontsize=18,
                            # Grid control
                            show_grid=True, grid_alpha=0.3, grid_linestyle='--',
                            grid_axis='both',
                            # Axis control
                            xlabel='r (Å)', ylabel='g(r)',
                            xlim=None, ylim=None,
                            show_minor_ticks=False, tick_direction='out',
                            # Custom labels
                            custom_labels=None,
                            # Multi-figure control
                            show_individual_figures=False,
                            individual_figsize=(8, 6),
                            save_combined_figure=True,
                            show_combined_figure=True,
                            save_individual_figures=True,
                            # Figure export control
                            save_fig=False, filename='rdf_with_shells.png',
                            dpi=300, bbox_inches='tight', transparent_bg=False):
        '''
        Plot RDF curves with solvation shell boundaries marked.
        
        Similar to plot_all_modified_shells() from EquilibriumAnalysisOptimized,
        but designed for arbitrary RDF data with custom boundaries.
        
        Parameters
        ----------
        rdf_dict : dict
            Dictionary of RDF results: {label: rdf_results}
        boundaries : dict
            Shell boundaries: {label: {'shell_1': (start, end), 'shell_2': (start, end), ...}}
        
        Layout Control
        --------------
        ncols : int
            Number of columns in grid layout (default: 2)
        figsize_per_plot : tuple
            Size of each subplot (width, height) (default: (8, 6))
        figsize : tuple, optional
            Explicit figure size override (width, height). If None, uses figsize_per_plot (default: None)
        
        RDF Line Styling
        ----------------
        rdf_color : str
            Color for RDF line (default: 'black')
        rdf_linewidth : float
            Width of RDF line (default: 2)
        linestyles : list, optional
            Line styles for RDF lines: '-', '--', '-.', ':' (default: None, all solid)
        markers : list, optional
            Marker styles for RDF lines: 'o', 's', '^', 'v', 'd' (default: None, no markers)
        markersize : float
            Marker size (default: 6)
        
        Shell Visualization
        -------------------
        shell_alpha : float
            Transparency of shell regions (default: 0.4)
        shell_colors : list or None
            Colors for shells. If None, uses colormap (default: None)
        colormap : str
            Colormap for shell colors: 'default' (blue gradient), or any matplotlib colormap (default: 'default')
        color_shade_style : str or None
            Color scheme style for ion coordination shells (default: None)
            - None: Original colors (P1=lightcoral, P2=lightblue, P3=lightgreen, P4=lightyellow, Bulk=lightgoldenrodyellow)
            - 'modified': Improved colors (P1=lightcoral, P2=lightgreen, P3=lightyellow, P4=lightblue, Bulk=aliceblue)
        shell_naming_style : str or None
            Override shell naming convention for display (default: None)
            - None: Use original names from boundaries
            - 'shell': Convert P1/P2/P3/P4 to S1/S2/S3/S4
            - 'peak': Convert S1/S2/S3/S4 to P1/P2/P3/P4
        show_shell_labels : bool
            Whether to show shell labels (S1, S2, etc.) (default: True)
        shell_label_fontsize : float
            Font size for shell labels (default: 10)
        show_boundary_lines : bool
            Whether to show vertical lines at boundaries (default: True)
        boundary_linestyle : str
            Line style for boundaries: '-', '--', '-.', ':' (default: '--')
        boundary_alpha : float
            Transparency of boundary lines (default: 0.7)
        boundary_linewidth : float
            Width of boundary lines (default: 1.5)
        
        RCN Overlay Control
        -------------------
        show_rcn : bool
            Whether to show running coordination number overlay (default: False)
        rcn_data : dict or None
            Dictionary of RCN data: {label: {'r': r_array, 'rcn': rcn_array, 'r0': first_min}}
            If None, RCN overlay is skipped (default: None)
        rcn_color : str
            Color for RCN line (default: 'red')
        rcn_linewidth : float
            Width of RCN line (default: 1.5)
        rcn_linestyle : str
            Line style for RCN: '-', '--', '-.', ':' (default: '--')
        rcn_scale_factor : float
            Scale factor for RCN y-axis (y_max = CN_at_r0 * scale_factor) (default: 4.0)
        rcn_ylabel : str
            Label for RCN y-axis (default: 'RCN(#)')
        
        Font & Text Control
        -------------------
        title_fontsize : float
            Title font size (default: 22)
        title_fontweight : str
            Title font weight: 'normal', 'bold', 'light', 'heavy' (default: 'bold')
        show_title : bool
            Whether to show subplot titles (default: True)
        title_text : dict, optional
            Custom title text override: {label: 'Custom Title'} (default: None)
        label_fontsize : float
            Axis label font size (default: 22)
        label_fontweight : str
            Axis label font weight (default: 'bold')
        tick_fontsize : float
            Tick label font size (default: 18)
        
        Grid Control
        ------------
        show_grid : bool
            Whether to show grid (default: True)
        grid_alpha : float
            Grid transparency (default: 0.3)
        grid_linestyle : str
            Grid line style (default: '--')
        grid_axis : str
            Which axes show grid: 'x', 'y', 'both' (default: 'both')
        
        Axis Control
        ------------
        xlabel, ylabel : str
            Axis labels
        xlim, ylim : tuple, optional
            Axis limits
        show_minor_ticks : bool
            Whether to show minor tick marks (default: False)
        tick_direction : str
            Tick direction: 'in', 'out', 'inout' (default: 'out')
        
        Custom Labels
        -------------
        custom_labels : dict, optional
            Custom display names for RDF labels: {label: 'Display Name'}
            Use mathtext for formatting: {'quinolone-OW': r'Q$\rightarrow$H$_2$O'} (default: None)
        
        Multi-Figure Control
        --------------------
        show_individual_figures : bool
            Whether to show individual figures for each RDF (default: False)
        individual_figsize : tuple
            Size for individual figures (width, height) (default: (8, 6))
        save_combined_figure : bool
            Whether to save the combined grid figure (default: True)
        show_combined_figure : bool
            Whether to display the combined grid figure (default: True)
        save_individual_figures : bool
            Whether to save individual figures (default: True)
        
        Figure Export Control
        ---------------------
        save_fig : bool
            Whether to save figure (default: False)
        filename : str
            Output filename (default: 'rdf_with_shells.png')
        dpi : int
            Resolution (default: 300)
        bbox_inches : str
            Bounding box for saved figure (default: 'tight')
        transparent_bg : bool
            Whether to save with transparent background (default: False)
        
        Returns
        -------
        fig, axes : matplotlib figure and axes objects
        
        Examples
        --------
        >>> # Define boundaries interactively
        >>> boundaries = analysis.interactive_rdf_boundary_editor(rdf)
        
        >>> # Plot with boundaries
        >>> plotter.plot_rdf_with_shells(rdf, boundaries, ncols=2)
        
        >>> # Publication quality
        >>> plotter.plot_rdf_with_shells(
        ...     rdf, boundaries,
        ...     figsize_per_plot=(10, 7),
        ...     rdf_linewidth=3,
        ...     title_fontsize=16,
        ...     label_fontsize=14,
        ...     save_fig=True,
        ...     dpi=600
        ... )
        '''
        
        import matplotlib.colors as mcolors
        
        n_plots = len(rdf_dict)
        nrows = int(np.ceil(n_plots / ncols))
        
        # Calculate figure size
        if figsize is not None:
            fig_width, fig_height = figsize
        else:
            fig_width = ncols * figsize_per_plot[0]
            fig_height = nrows * figsize_per_plot[1]
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
        
        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
        elif nrows == 1:
            axes = [axes] if ncols == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        # Convert to list to ensure uniform access
        axes = list(axes) if not isinstance(axes, list) else axes
        
        # Helper function to generate shell colors
        def generate_shell_colors(n_shells, use_colormap='default'):
            '''Generate colors for shells based on colormap choice'''
            if use_colormap == 'default':
                # Blue saturation gradient like in plot_all_modified_shells
                base_rgb = mcolors.hex2color('#00c5ff')
                base_hsv = mcolors.rgb_to_hsv(base_rgb)
                base_hue = base_hsv[0]
                base_saturation = base_hsv[1]
                base_value = base_hsv[2]
                
                if n_shells == 1:
                    saturations = [base_saturation]
                elif n_shells == 2:
                    saturations = [base_saturation, 0.6]
                elif n_shells == 3:
                    saturations = [base_saturation, 0.7, 0.4]
                else:
                    step = (base_saturation - 0.2) / (n_shells - 1)
                    saturations = [base_saturation - (i * step) for i in range(n_shells)]
                
                colors = []
                for sat in saturations:
                    hsv = (base_hue, sat, base_value)
                    rgb = mcolors.hsv_to_rgb(hsv)
                    colors.append(mcolors.to_hex(rgb))
            else:
                # Use matplotlib colormap
                cmap = plt.get_cmap(use_colormap)
                colors = [mcolors.to_hex(cmap(i / max(1, n_shells - 1))) for i in range(n_shells)]
            
            return colors
        
        # Plot each RDF with its boundaries
        for i, (label, rdf_results) in enumerate(rdf_dict.items()):
            ax = axes[i]
            
            # Extract RDF data
            if hasattr(rdf_results, 'bins'):
                r = rdf_results.bins
                g_r = rdf_results.rdf
            else:
                r = rdf_results['bins']
                g_r = rdf_results['rdf']
            
            # Get boundaries for this RDF
            label_boundaries = boundaries.get(label, {})
            
            # Auto-convert shell_1/shell_2/shell_3 to S1/S2/S3 format for display consistency
            # This ensures boundaries from interactive_rdf_boundary_editor work seamlessly
            auto_converted_boundaries = {}
            for shell_name, bounds in label_boundaries.items():
                # Convert shell_1 -> S1, shell_2 -> S2, etc.
                if shell_name.startswith('shell_') and shell_name[6:].isdigit():
                    shell_num = shell_name[6:]
                    new_name = f'S{shell_num}'
                    auto_converted_boundaries[new_name] = bounds
                else:
                    auto_converted_boundaries[shell_name] = bounds
            label_boundaries = auto_converted_boundaries
            
            # Apply shell naming style conversion if requested
            if shell_naming_style is not None and label_boundaries:
                converted_boundaries = {}
                for shell_name, bounds in label_boundaries.items():
                    if shell_naming_style == 'shell':
                        # Convert P1/P2/P3/P4 to S1/S2/S3/S4
                        if shell_name in ['P1', 'P2', 'P3', 'P4']:
                            new_name = shell_name.replace('P', 'S')
                            converted_boundaries[new_name] = bounds
                        else:
                            converted_boundaries[shell_name] = bounds
                    elif shell_naming_style == 'peak':
                        # Convert S1/S2/S3/S4 to P1/P2/P3/P4
                        if shell_name in ['S1', 'S2', 'S3', 'S4']:
                            new_name = shell_name.replace('S', 'P')
                            converted_boundaries[new_name] = bounds
                        else:
                            converted_boundaries[shell_name] = bounds
                    else:
                        converted_boundaries[shell_name] = bounds
                label_boundaries = converted_boundaries
            
            # Calculate RDF max for later use (ylim, label positioning)
            rdf_max = np.max(g_r)
            
            # Plot shell regions first (so they appear behind RDF line)
            if label_boundaries:
                # Detect if this is ion coordination (CIP, SIP, DSIP, FI shells or flexible naming)
                # Check for both individual and combined region names (e.g., "DSIP+FI")
                ion_shell_order = ['CIP', 'SIP', 'DSIP', 'FI', 'P1', 'P2', 'P3', 'P4', 'Shell_1', 'Shell_2', 'Shell_3', 'Shell_4']
                present_shells = set(label_boundaries.keys())
                is_ion = any(
                    any(ion_name == shell_name or ion_name in shell_name.split('+') for ion_name in ion_shell_order)
                    for shell_name in present_shells
                )
                
                # Sort shells by start position
                sorted_shells = sorted(label_boundaries.items(), key=lambda x: x[1][0])
                
                # For ions: treat all shells as regular (including FI with inf)
                # For water: separate bulk from regular shells
                if is_ion:
                    regular_shells = sorted_shells
                    bulk_shells = []
                else:
                    bulk_shells = [(name, bounds) for name, bounds in sorted_shells if np.isinf(bounds[1])]
                    regular_shells = [(name, bounds) for name, bounds in sorted_shells if not np.isinf(bounds[1])]
                
                n_regular_shells = len(regular_shells)
                n_total_shells = len(sorted_shells)
                
                # Generate colors
                if shell_colors is None:
                    if is_ion:
                        # Use ion pairing colors for CIP, SIP, DSIP, FI with hierarchical logic for combined regions
                        # Now supports flexible naming P1/P2/P3/P4 and Shell_1/Shell_2/Shell_3/Shell_4
                        # Support multiple color schemes via color_shade_style parameter
                        
                        if color_shade_style == 'modified':
                            # Modified color scheme for better visual separation
                            ion_colors = {
                                'CIP': 'lightcoral',
                                'SIP': 'lightgreen',      # Changed from lightblue
                                'DSIP': 'lightyellow',    # Changed from lightgreen  
                                'FI': 'lightblue',        # Changed from lightyellow
                                # Flexible naming support - modified scheme
                                'P1': 'lightcoral',       # Keep same as CIP
                                'P2': 'lightgreen',       # Better separation from P1
                                'P3': 'lightyellow',      # Clear distinction from P2
                                'P4': 'lightblue',        # Much better contrast with P3
                                'Shell_1': 'lightcoral',
                                'Shell_2': 'lightgreen',  # Changed from lightblue
                                'Shell_3': 'lightyellow', # Changed from lightgreen
                                'Shell_4': 'lightblue',   # Changed from lightyellow
                                'S1': 'lightcoral',       # Abbreviated shell names
                                'S2': 'lightgreen',       # Changed from lightblue
                                'S3': 'lightyellow',      # Changed from lightgreen
                                'S4': 'lightblue',        # Changed from lightyellow
                                'Bulk': 'aliceblue'       # Very light, clearly different from P4
                            }
                        else:
                            # Original color scheme (default)
                            ion_colors = {
                                'CIP': 'lightcoral',
                                'SIP': 'lightblue',
                                'DSIP': 'lightgreen',
                                'FI': 'lightgoldenrodyellow',     # Swapped with Bulk
                                # Flexible naming support - original scheme
                                'P1': 'lightcoral',       # Same as CIP
                                'P2': 'lightblue',        # Same as SIP  
                                'P3': 'lightgreen',       # Same as DSIP
                                'P4': 'lightgoldenrodyellow',     # Swapped with Bulk
                                'Shell_1': 'lightcoral',
                                'Shell_2': 'lightblue', 
                                'Shell_3': 'lightgreen',
                                'Shell_4': 'lightgoldenrodyellow',  # Swapped with Bulk
                                'S1': 'lightcoral',        # Abbreviated shell names
                                'S2': 'lightblue',         # Same as SIP
                                'S3': 'lightgreen',        # Same as DSIP
                                'S4': 'lightgoldenrodyellow', # Swapped with Bulk
                                'Bulk': 'lightyellow'      # Swapped with P4
                            }
                        
                        # Define order for hierarchical logic
                        region_order_traditional = ['CIP', 'SIP', 'DSIP', 'FI']
                        region_order_peak = ['P1', 'P2', 'P3', 'P4']
                        region_order_shell = ['Shell_1', 'Shell_2', 'Shell_3', 'Shell_4']
                        region_order_abbrev = ['S1', 'S2', 'S3', 'S4']
                        
                        # Create color list matching shell order
                        colors = []
                        for name, _ in regular_shells:
                            if '+' in name:
                                # Combined region - use hierarchical color logic
                                parts = name.split('+')
                                if 'FI' in parts or 'P4' in parts or 'Shell_4' in parts or 'S4' in parts:
                                    color = ion_colors.get('P4', ion_colors.get('FI', 'lightyellow'))
                                elif any(p in region_order_traditional for p in parts):
                                    highest = max([p for p in parts if p in region_order_traditional], 
                                                key=lambda x: region_order_traditional.index(x))
                                    color = ion_colors.get(highest, 'lightgray')
                                elif any(p in region_order_peak for p in parts):
                                    highest = max([p for p in parts if p in region_order_peak], 
                                                key=lambda x: region_order_peak.index(x))
                                    color = ion_colors.get(highest, 'lightgray')
                                elif any(p in region_order_shell for p in parts):
                                    highest = max([p for p in parts if p in region_order_shell], 
                                                key=lambda x: region_order_shell.index(x))
                                    color = ion_colors.get(highest, 'lightgray')
                                elif any(p in region_order_abbrev for p in parts):
                                    highest = max([p for p in parts if p in region_order_abbrev], 
                                                key=lambda x: region_order_abbrev.index(x))
                                    color = ion_colors.get(highest, 'lightgray')
                                else:
                                    color = 'lightgray'
                            else:
                                # Direct name lookup
                                color = ion_colors.get(name, 'lightgray')
                            colors.append(color)
                    else:
                        # Generate enough colors for regular shells + bulk (water shells)
                        colors = generate_shell_colors(n_total_shells if bulk_shells else n_regular_shells, colormap)
                else:
                    colors = shell_colors if len(shell_colors) >= n_total_shells else shell_colors * (n_total_shells // len(shell_colors) + 1)
                
                # Calculate label position (inside figure, 15% above RDF max)
                label_y_position = rdf_max * 1.15
                
                # Plot regular shells
                for j, (shell_name, (start, end)) in enumerate(regular_shells):
                    color = colors[j] if j < len(colors) else colors[-1]
                    
                    # Determine plot end (handle infinite endpoints like FI)
                    if np.isinf(end):
                        plot_end = xlim[1] if xlim is not None else r[-1]
                    else:
                        plot_end = end
                    
                    # Fill shell region
                    ax.axvspan(start, plot_end, alpha=shell_alpha, color=color, zorder=0)
                    
                    # Add boundary lines
                    if show_boundary_lines:
                        ax.axvline(start, color=color, linestyle=boundary_linestyle,
                                 linewidth=boundary_linewidth, alpha=boundary_alpha, zorder=1)
                        # Only draw end line if it's not infinite
                        if not np.isinf(end):
                            ax.axvline(end, color=color, linestyle=boundary_linestyle,
                                     linewidth=boundary_linewidth, alpha=boundary_alpha, zorder=1)
                    
                    # Add shell label
                    if show_shell_labels:
                        mid_point = (start + plot_end) / 2
                        # Use actual shell name for ions (CIP, SIP, DSIP, FI), short format for water (S1, S2, S3)
                        if is_ion:
                            label_text = shell_name
                        else:
                            shell_number = j + 1
                            label_text = f'S{shell_number}'
                        
                        ax.text(mid_point, label_y_position, label_text,
                               ha='center', va='bottom', fontweight='bold',
                               fontsize=shell_label_fontsize, color='black', zorder=3)
                
                # Plot bulk region if it exists
                if bulk_shells:
                    bulk_name, (bulk_start, bulk_end) = bulk_shells[0]
                    # Use the lightest color for bulk (last color in gradient)
                    bulk_color = colors[-1] if len(colors) >= n_total_shells else colors[n_regular_shells]
                    
                    # Determine plot end for bulk (use xlim if set, otherwise use max r)
                    if xlim is not None:
                        bulk_plot_end = xlim[1]
                    else:
                        bulk_plot_end = r[-1]
                    
                    # Fill bulk region
                    ax.axvspan(bulk_start, bulk_plot_end, alpha=shell_alpha, color=bulk_color, zorder=0)
                    
                    # Add boundary line at bulk start
                    if show_boundary_lines:
                        ax.axvline(bulk_start, color=bulk_color, linestyle=boundary_linestyle,
                                 linewidth=boundary_linewidth, alpha=boundary_alpha, zorder=1)
                    
                    # Add bulk label
                    if show_shell_labels:
                        bulk_mid_point = (bulk_start + bulk_plot_end) / 2
                        ax.text(bulk_mid_point, label_y_position, 'Bulk',
                               ha='center', va='bottom', fontweight='bold',
                               fontsize=shell_label_fontsize, color='black', zorder=3)
            
            # Plot RDF line on top
            linestyle = linestyles[i] if linestyles and i < len(linestyles) else '-'
            marker = markers[i] if markers and i < len(markers) else None
            
            ax.plot(r, g_r, color=rdf_color, linewidth=rdf_linewidth,
                   linestyle=linestyle, marker=marker, markersize=markersize,
                   markevery=max(1, len(r)//20) if marker else None, zorder=2)
            
            # Plot RCN overlay if requested
            if show_rcn and rcn_data is not None:
                # Check if rcn_data has this label
                if label in rcn_data:
                    rcn_info = rcn_data[label]
                    r_rcn = rcn_info.get('r', rcn_info.get('bins'))
                    rcn = rcn_info.get('rcn')
                    
                    if r_rcn is not None and rcn is not None:
                        # Create secondary y-axis for RCN
                        ax2 = ax.twinx()
                        
                        # Plot RCN
                        ax2.plot(r_rcn, rcn, color=rcn_color, linewidth=rcn_linewidth,
                               linestyle=rcn_linestyle, zorder=2.5, label='RCN')
                        
                        # Smart y-axis scaling for RCN
                        # Find r₀ (second shell end) from boundaries to get CN at second coordination shell
                        r0 = None
                        if label_boundaries:
                            # Get second shell (second smallest start position)
                            regular_shells = [(name, bounds) for name, bounds in label_boundaries.items() 
                                            if not np.isinf(bounds[1])]
                            if len(regular_shells) >= 2:
                                sorted_regular = sorted(regular_shells, key=lambda x: x[1][0])
                                second_shell_name, (shell_start, shell_end) = sorted_regular[1]
                                r0 = shell_end  # Second minimum = end of second shell
                            elif len(regular_shells) == 1:
                                # Fallback to first shell if only one shell exists
                                sorted_regular = sorted(regular_shells, key=lambda x: x[1][0])
                                first_shell_name, (shell_start, shell_end) = sorted_regular[0]
                                r0 = shell_end
                        
                        if r0 is not None:
                            # Find CN at r₀
                            idx_r0 = np.argmin(np.abs(r_rcn - r0))
                            cn_at_r0 = rcn[idx_r0]
                            
                            # Scale so CN at r₀ appears at 1/rcn_scale_factor of figure height
                            y_max_rcn = cn_at_r0 * rcn_scale_factor
                            ax2.set_ylim(0, y_max_rcn)
                            
                            # Optional: mark CN value at r₀
                            # ax2.axhline(cn_at_r0, color='orange', linestyle=':', linewidth=1, alpha=0.5)
                        else:
                            # Fallback: use max RCN in plot range
                            if xlim is not None:
                                mask = r_rcn <= xlim[1]
                                max_rcn_in_range = np.max(rcn[mask])
                            else:
                                max_rcn_in_range = np.max(rcn)
                            ax2.set_ylim(0, max_rcn_in_range * 1.1)
                        
                        # Set RCN axis label
                        ax2.set_ylabel(rcn_ylabel, fontsize=label_fontsize, 
                                     fontweight=label_fontweight, color=rcn_color)
                        ax2.tick_params(axis='y', labelsize=tick_fontsize, labelcolor=rcn_color)
            
            # Formatting
            ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
            ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
            
            # Title - use custom text if provided
            if show_title:
                if title_text and label in title_text:
                    title_str = title_text[label]
                elif custom_labels and label in custom_labels:
                    title_str = custom_labels[label]
                else:
                    title_str = label
                ax.set_title(title_str, fontsize=title_fontsize, fontweight=title_fontweight)
            
            # Tick formatting
            ax.tick_params(axis='both', labelsize=tick_fontsize, direction=tick_direction)
            if show_minor_ticks:
                ax.minorticks_on()
                ax.tick_params(which='minor', direction=tick_direction)
            
            # Grid
            if show_grid:
                if grid_axis == 'both':
                    ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle, zorder=0)
                elif grid_axis == 'x':
                    ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle, axis='x', zorder=0)
                elif grid_axis == 'y':
                    ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle, axis='y', zorder=0)
            
            # Axis limits - fill all data regions without padding
            if xlim is not None:
                ax.set_xlim(xlim)
            else:
                # Set xlim to data range without padding
                ax.set_xlim(r[0], r[-1])
            
            if ylim is not None:
                ax.set_ylim(ylim)
            else:
                # Automatically adjust ylim to accommodate labels (35% above max for comfort)
                ax.set_ylim(0, rdf_max * 1.35)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save combined figure if requested
        if save_fig and save_combined_figure:
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                       transparent=transparent_bg)
            print(f"✓ Combined figure saved: {filename}")
        
        # Show combined figure
        if show_combined_figure:
            plt.show()
        else:
            plt.close(fig)
        
        # Generate individual figures if requested
        individual_figs = []
        if show_individual_figures or save_individual_figures:
            for label, rdf_results in rdf_dict.items():
                # Create individual figure
                fig_ind, ax_ind = plt.subplots(1, 1, figsize=individual_figsize)
                
                # Extract RDF data
                if hasattr(rdf_results, 'bins'):
                    r_ind = rdf_results.bins
                    g_r_ind = rdf_results.rdf
                else:
                    r_ind = rdf_results['bins']
                    g_r_ind = rdf_results['rdf']
                
                # Get boundaries
                label_boundaries_ind = boundaries.get(label, {})
                
                # Calculate RDF max for later use (ylim, label positioning)
                rdf_max_ind = np.max(g_r_ind)
                
                # Plot shells (same logic as combined)
                if label_boundaries_ind:
                    # Detect if this is ion coordination (same logic as combined)
                    ion_shell_order_ind = ['CIP', 'SIP', 'DSIP', 'FI', 'P1', 'P2', 'P3', 'P4', 'Shell_1', 'Shell_2', 'Shell_3', 'Shell_4']
                    present_shells_ind = set(label_boundaries_ind.keys())
                    is_ion_ind = any(
                        any(ion_name == shell_name or ion_name in shell_name.split('+') for ion_name in ion_shell_order_ind)
                        for shell_name in present_shells_ind
                    )
                    
                    sorted_shells_ind = sorted(label_boundaries_ind.items(), key=lambda x: x[1][0])
                    
                    # For ions: treat all shells as regular (including FI with inf)
                    # For water: separate bulk from regular shells
                    if is_ion_ind:
                        regular_shells_ind = sorted_shells_ind
                        bulk_shells_ind = []
                    else:
                        bulk_shells_ind = [(name, bounds) for name, bounds in sorted_shells_ind if np.isinf(bounds[1])]
                        regular_shells_ind = [(name, bounds) for name, bounds in sorted_shells_ind if not np.isinf(bounds[1])]
                    
                    n_regular_shells_ind = len(regular_shells_ind)
                    n_total_shells_ind = len(sorted_shells_ind)
                    
                    # Generate colors with same logic as combined
                    if shell_colors is None:
                        if is_ion_ind:
                            # Use ion pairing colors with hierarchical logic for combined regions
                            # Support both traditional CIP/SIP/DSIP/FI and flexible P1/P2/P3/P4 naming
                            ion_colors = {
                                'CIP': 'lightcoral',
                                'SIP': 'lightblue', 
                                'DSIP': 'lightgreen',
                                'FI': 'lightyellow',
                                # Flexible naming support
                                'P1': 'lightcoral',    # Same as CIP
                                'P2': 'lightblue',     # Same as SIP  
                                'P3': 'lightgreen',    # Same as DSIP
                                'P4': 'lightyellow',   # Same as FI
                                'Shell_1': 'lightcoral',
                                'Shell_2': 'lightblue', 
                                'Shell_3': 'lightgreen',
                                'Shell_4': 'lightyellow',
                                'Bulk': 'lightgoldenrodyellow'  # Very light yellow for bulk
                            }
                            
                            # Define order for hierarchical logic
                            region_order_traditional = ['CIP', 'SIP', 'DSIP', 'FI']
                            region_order_peak = ['P1', 'P2', 'P3', 'P4']
                            region_order_shell = ['Shell_1', 'Shell_2', 'Shell_3', 'Shell_4']
                            
                            colors_ind = []
                            for name, _ in regular_shells_ind:
                                if '+' in name:
                                    # Handle combined regions (e.g., "P1+P2")
                                    parts = name.split('+')
                                    if 'FI' in parts or 'P4' in parts or 'Shell_4' in parts:
                                        color = ion_colors.get('P4', ion_colors.get('FI', 'lightyellow'))
                                    elif any(p in region_order_traditional for p in parts):
                                        highest = max([p for p in parts if p in region_order_traditional], 
                                                    key=lambda x: region_order_traditional.index(x))
                                        color = ion_colors.get(highest, 'lightgray')
                                    elif any(p in region_order_peak for p in parts):
                                        highest = max([p for p in parts if p in region_order_peak], 
                                                    key=lambda x: region_order_peak.index(x))
                                        color = ion_colors.get(highest, 'lightgray')
                                    elif any(p in region_order_shell for p in parts):
                                        highest = max([p for p in parts if p in region_order_shell], 
                                                    key=lambda x: region_order_shell.index(x))
                                        color = ion_colors.get(highest, 'lightgray')
                                    else:
                                        color = 'lightgray'
                                else:
                                    # Direct name lookup
                                    color = ion_colors.get(name, 'lightgray')
                                colors_ind.append(color)
                        else:
                            colors_ind = generate_shell_colors(n_total_shells_ind if bulk_shells_ind else n_regular_shells_ind, colormap)
                    else:
                        colors_ind = shell_colors
                    
                    # Calculate label position (inside figure, 15% above RDF max)
                    label_y_position_ind = rdf_max_ind * 1.15
                    
                    # Plot regular shells
                    for j, (shell_name, (start, end)) in enumerate(regular_shells_ind):
                        color = colors_ind[j] if j < len(colors_ind) else colors_ind[-1]
                        
                        # Determine plot end (handle infinite endpoints like FI)
                        if np.isinf(end):
                            plot_end = xlim[1] if xlim is not None else r_ind[-1]
                        else:
                            plot_end = end
                        
                        ax_ind.axvspan(start, plot_end, alpha=shell_alpha, color=color, zorder=0)
                        if show_boundary_lines:
                            ax_ind.axvline(start, color=color, linestyle=boundary_linestyle,
                                         linewidth=boundary_linewidth, alpha=boundary_alpha, zorder=1)
                            if not np.isinf(end):
                                ax_ind.axvline(end, color=color, linestyle=boundary_linestyle,
                                             linewidth=boundary_linewidth, alpha=boundary_alpha, zorder=1)
                        if show_shell_labels:
                            mid_point = (start + plot_end) / 2
                            # Use actual shell name for ions, short format for water
                            if is_ion_ind:
                                label_text = shell_name
                            else:
                                label_text = f'S{j+1}'
                            ax_ind.text(mid_point, label_y_position_ind, label_text,
                                       ha='center', va='bottom', fontweight='bold',
                                       fontsize=shell_label_fontsize, color='black', zorder=3)
                    
                    # Plot bulk if exists (water only)
                    if bulk_shells_ind:
                        bulk_name, (bulk_start, bulk_end) = bulk_shells_ind[0]
                        bulk_color = colors_ind[-1] if len(colors_ind) >= n_total_shells_ind else colors_ind[n_regular_shells_ind]
                        bulk_plot_end = xlim[1] if xlim is not None else r_ind[-1]
                        ax_ind.axvspan(bulk_start, bulk_plot_end, alpha=shell_alpha, color=bulk_color, zorder=0)
                        if show_boundary_lines:
                            ax_ind.axvline(bulk_start, color=bulk_color, linestyle=boundary_linestyle,
                                         linewidth=boundary_linewidth, alpha=boundary_alpha, zorder=1)
                        if show_shell_labels:
                            bulk_mid_point = (bulk_start + bulk_plot_end) / 2
                            ax_ind.text(bulk_mid_point, label_y_position_ind, 'Bulk',
                                       ha='center', va='bottom', fontweight='bold',
                                       fontsize=shell_label_fontsize, color='black', zorder=3)
                
                # Plot RDF line
                ax_ind.plot(r_ind, g_r_ind, color=rdf_color, linewidth=rdf_linewidth, zorder=2)
                
                # RCN overlay for individual figure
                if show_rcn and rcn_data is not None and label in rcn_data:
                    rcn_info_ind = rcn_data[label]
                    r_rcn_ind = rcn_info_ind.get('r', rcn_info_ind.get('bins'))
                    rcn_ind = rcn_info_ind.get('rcn')
                    
                    if r_rcn_ind is not None and rcn_ind is not None:
                        ax2_ind = ax_ind.twinx()
                        ax2_ind.plot(r_rcn_ind, rcn_ind, color=rcn_color, linewidth=rcn_linewidth,
                                   linestyle=rcn_linestyle, zorder=2.5)
                        
                        r0_ind = None
                        if label_boundaries_ind:
                            regular_shells_check = [(name, bounds) for name, bounds in label_boundaries_ind.items() 
                                                  if not np.isinf(bounds[1])]
                            if len(regular_shells_check) >= 2:
                                sorted_regular_check = sorted(regular_shells_check, key=lambda x: x[1][0])
                                _, (_, shell_end_ind) = sorted_regular_check[1]
                                r0_ind = shell_end_ind
                            elif len(regular_shells_check) == 1:
                                sorted_regular_check = sorted(regular_shells_check, key=lambda x: x[1][0])
                                _, (_, shell_end_ind) = sorted_regular_check[0]
                                r0_ind = shell_end_ind
                        
                        if r0_ind is not None:
                            idx_r0_ind = np.argmin(np.abs(r_rcn_ind - r0_ind))
                            cn_at_r0_ind = rcn_ind[idx_r0_ind]
                            y_max_rcn_ind = cn_at_r0_ind * rcn_scale_factor
                            ax2_ind.set_ylim(0, y_max_rcn_ind)
                        else:
                            if xlim is not None:
                                mask = r_rcn_ind <= xlim[1]
                                max_rcn_in_range = np.max(rcn_ind[mask])
                            else:
                                max_rcn_in_range = np.max(rcn_ind)
                            ax2_ind.set_ylim(0, max_rcn_in_range * 1.1)
                        
                        ax2_ind.set_ylabel(rcn_ylabel, fontsize=label_fontsize, 
                                         fontweight=label_fontweight, color=rcn_color)
                        ax2_ind.tick_params(axis='y', labelsize=tick_fontsize, labelcolor=rcn_color)
                
                # Formatting for individual figure
                ax_ind.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
                ax_ind.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                
                if show_title:
                    if title_text and label in title_text:
                        title_str_ind = title_text[label]
                    elif custom_labels and label in custom_labels:
                        title_str_ind = custom_labels[label]
                    else:
                        title_str_ind = label
                    ax_ind.set_title(title_str_ind, fontsize=title_fontsize, fontweight=title_fontweight)
                
                ax_ind.tick_params(axis='both', labelsize=tick_fontsize, direction=tick_direction)
                if show_minor_ticks:
                    ax_ind.minorticks_on()
                    ax_ind.tick_params(which='minor', direction=tick_direction)
                
                if show_grid:
                    if grid_axis == 'both':
                        ax_ind.grid(True, alpha=grid_alpha, linestyle=grid_linestyle, zorder=0)
                    elif grid_axis == 'x':
                        ax_ind.grid(True, alpha=grid_alpha, linestyle=grid_linestyle, axis='x', zorder=0)
                    elif grid_axis == 'y':
                        ax_ind.grid(True, alpha=grid_alpha, linestyle=grid_linestyle, axis='y', zorder=0)
                
                if xlim is not None:
                    ax_ind.set_xlim(xlim)
                else:
                    ax_ind.set_xlim(r_ind[0], r_ind[-1])
                
                if ylim is not None:
                    ax_ind.set_ylim(ylim)
                else:
                    ax_ind.set_ylim(0, rdf_max_ind * 1.35)
                
                plt.tight_layout()
                
                # Save individual figure
                if save_individual_figures:
                    # Generate filename
                    base_name = filename.rsplit('.', 1)[0]
                    ext = filename.rsplit('.', 1)[1] if '.' in filename else 'png'
                    safe_label = label.replace('/', '_').replace('\\', '_').replace(' ', '_')
                    ind_filename = f"{base_name}_{safe_label}.{ext}"
                    fig_ind.savefig(ind_filename, dpi=dpi, bbox_inches=bbox_inches,
                                   transparent=transparent_bg)
                    print(f"✓ Individual figure saved: {ind_filename}")
                
                # Show individual figure
                if show_individual_figures:
                    plt.show()
                else:
                    plt.close(fig_ind)
                
                individual_figs.append((label, fig_ind, ax_ind))
        
        return fig, axes, individual_figs if individual_figs else None
    
    # =========================================================================
    # ION BINDING PLOTTING METHODS
    # =========================================================================
    
    def plot_ion_binding_comparison(self, binding_results_dict, 
                                   # Overall plot control
                                   title='Ion Binding Comparison',
                                   subplot_layout='horizontal',  # 'horizontal', 'vertical', or 'single'
                                   # Volume normalization (NEW)
                                   normalize_by_volume=False, plot_peaks=None,
                                   density_units='auto', volume_info_in_title=True,
                                   volume_calculation_method='weighted_average',
                                   # Bar styling
                                   bar_width=0.25, colors=None, colormap='Set2',
                                   hatches=None, edgecolor='black', edgewidth=1.2,
                                   bar_alpha=0.9,
                                   # Value labels on bars
                                   show_values=True, value_fontsize=9, value_format='{:.2f}',
                                   value_offset=0.05,
                                   # Font & text control
                                   title_fontsize=14, title_fontweight='bold', show_title=True,
                                   label_fontsize=12, label_fontweight='normal',
                                   tick_fontsize=10, legend_fontsize=10, legend_fontweight='normal',
                                   # Axis labels
                                   xlabel='Ion Type', ylabel='Average Number of Bound Ions',
                                   # Legend control
                                   show_legend=True, legend_loc='best', legend_framealpha=0.9,
                                   legend_ncol=1, custom_labels=None,
                                   # Grid control
                                   show_grid=True, grid_alpha=0.3, grid_axis='y',
                                   # Axis limits
                                   ylim=None,
                                   # Error bars (optional)
                                   show_errorbars=False, errorbar_capsize=4,
                                   # Figure export control
                                   save_fig=False, filename='ion_binding.png',
                                   dpi=300, figsize=None, bbox_inches='tight',
                                   transparent_bg=False):
        """
        Plot comparison of ion binding across multiple targets in a single figure with grouped bars.
        Enhanced with volume normalization support for peak-specific analysis.
        
        This method creates a grouped bar chart comparing ion binding for different target selections
        (e.g., quinolone, carboxylic_acid, piperazine). Each target is represented by a different
        color/pattern, and ion types (NA, K, etc.) are shown on the x-axis with grouped bars.
        
        Parameters
        ----------
        binding_results_dict : dict
            Dictionary of binding results from batch ion_binding_analysis()
            Format: {target_label: binding_results_for_that_target}
            Each binding_results contains 'cation_binding' and/or 'anion_binding' dicts
        
        Overall Plot Control
        --------------------
        title : str
            Overall plot title (default: 'Ion Binding Comparison')
        subplot_layout : str
            Layout for cations/anions: 'horizontal' (side-by-side), 'vertical' (stacked),
            or 'single' (combined on one plot) (default: 'horizontal')
        
        Volume Normalization (NEW)
        -------------------------
        normalize_by_volume : bool
            Whether to plot volume-normalized densities instead of raw counts (default: False)
        plot_peaks : list or None
            Specific peaks to plot (e.g., ['P1', 'P2']). If None, plots overall binding.
        density_units : str
            Units for density display: 'auto', 'per_A3', 'per_nm3' (default: 'auto')
        volume_info_in_title : bool
            Whether to include volume normalization info in plot title (default: True)
        volume_calculation_method : str
            Method for volume normalization calculations: 'weighted_average' (volume-weighted
            average density across peaks, good for comparisons), 'sum' (sum of individual peak
            densities, good for breakdowns), 'auto' (uses method-appropriate default)
            (default: 'weighted_average' for comparison plots)
        
        Bar Styling
        -----------
        bar_width : float
            Width of each bar (default: 0.25)
        colors : list or dict, optional
            Colors for each target. Can be:
            - List: colors applied in order ['blue', 'red', 'green']
            - Dict: colors mapped by target label {'quinolone': 'red', 'piperazine': 'blue'}
            If None, uses colormap
        colormap : str
            Matplotlib colormap name for auto-generating colors (default: 'Set2')
        hatches : list or dict, optional
            Hatching patterns for each target. Can be:
            - List: patterns applied in order ['///', '\\\\\\', 'xxx', '...', '|||']
            - Dict: patterns mapped by target label {'quinolone': '///', 'piperazine': 'xxx'}
            Useful for black/white printing (default: None)
        edgecolor : str
            Bar edge color (default: 'black')
        edgewidth : float
            Bar edge width (default: 1.2)
        bar_alpha : float
            Bar transparency 0-1 (default: 0.9)
        
        Value Labels on Bars
        --------------------
        show_values : bool
            Whether to show numerical values on top of bars (default: True)
        value_fontsize : float
            Font size for value labels (default: 9)
        value_format : str
            Format string for values (default: '{:.2f}')
        value_offset : float
            Vertical offset for value labels as fraction of ylim (default: 0.05)
        
        Font & Text Control
        -------------------
        title_fontsize : float
            Title font size (default: 14)
        title_fontweight : str
            Title font weight: 'normal', 'bold', 'light', 'heavy' (default: 'bold')
        show_title : bool
            Whether to show overall title (default: True)
        label_fontsize : float
            Axis label font size (default: 12)
        label_fontweight : str
            Axis label font weight (default: 'normal')
        tick_fontsize : float
            Tick label font size (default: 10)
        legend_fontsize : float
            Legend font size (default: 10)
        legend_fontweight : str
            Legend font weight (default: 'normal')
        
        Axis Labels
        -----------
        xlabel : str
            X-axis label (default: 'Ion Type')
        ylabel : str
            Y-axis label (default: 'Average Number of Bound Ions')
        
        Legend Control
        --------------
        show_legend : bool
            Whether to show legend (default: True)
        legend_loc : str
            Legend location: 'best', 'upper right', 'lower left', etc. (default: 'best')
        legend_framealpha : float
            Legend background transparency (default: 0.9)
        legend_ncol : int
            Number of legend columns (default: 1)
        custom_labels : dict, optional
            Custom labels for targets in legend. Maps target keys to display names.
            Use mathtext for superscripts: {'quinolone': r'Quinolone-Na$^+$'}
            If None, uses target keys as labels (default: None)
        
        Grid Control
        ------------
        show_grid : bool
            Whether to show grid (default: True)
        grid_alpha : float
            Grid transparency (default: 0.3)
        grid_axis : str
            Which axis to show grid: 'x', 'y', or 'both' (default: 'y')
        
        Axis Limits
        -----------
        ylim : tuple, optional
            Y-axis limits (e.g., (0, 5))
        
        Error Bars (Optional)
        ---------------------
        show_errorbars : bool
            Whether to show error bars (std dev across frames) (default: False)
        errorbar_capsize : float
            Cap size for error bars (default: 4)
        
        Figure Export Control
        ---------------------
        save_fig : bool
            Whether to save figure (default: False)
        filename : str
            Output filename (default: 'ion_binding.png')
        dpi : int
            Resolution for saved figure (default: 300)
        figsize : tuple, optional
            Figure size in inches (width, height). Auto-calculated if None
        bbox_inches : str
            Bounding box for saved figure (default: 'tight')
        transparent_bg : bool
            Whether to save with transparent background (default: False)
        
        Returns
        -------
        fig, axes : matplotlib figure and axes objects
        
        Examples
        --------
        >>> # Batch binding analysis for multiple targets
        >>> binding_results = analysis.ion_binding_analysis(
        ...     target_sel=['name O*', 'name N*', 'name C1'],
        ...     ion_types=['NA', 'K'],
        ...     cutoff=3.5
        ... )
        
        >>> # Basic grouped bar plot
        >>> plotter.plot_ion_binding_comparison(binding_results)
        
        >>> # Advanced formatting with custom labels and styling
        >>> plotter.plot_ion_binding_comparison(
        ...     binding_results,
        ...     custom_labels={
        ...         'quinolone': r'Quinolone',
        ...         'carboxylic_acid': r'Carboxylic Acid',
        ...         'piperazine': r'Piperazine'
        ...     },
        ...     colors={'quinolone': '#1f77b4', 'carboxylic_acid': '#ff7f0e', 
        ...             'piperazine': '#2ca02c'},
        ...     hatches={'quinolone': '///', 'carboxylic_acid': '\\\\\\', 'piperazine': 'xxx'},
        ...     subplot_layout='horizontal',
        ...     show_values=True,
        ...     show_errorbars=True,
        ...     bar_width=0.22,
        ...     save_fig=True,
        ...     filename='ion_binding_grouped.png',
        ...     dpi=600
        ... )
        """
        
        # Validate input
        if not isinstance(binding_results_dict, dict):
            print("Error: binding_results_dict must be a dictionary of binding results from batch analysis")
            return None, None
        
        if len(binding_results_dict) == 0:
            print("No binding data to plot")
            return None, None
        
        # Extract target labels
        target_labels = list(binding_results_dict.keys())
        n_targets = len(target_labels)
        
        # Collect all unique ion names (cations and anions separately)
        all_cations = set()
        all_anions = set()
        
        for target_label, binding_results in binding_results_dict.items():
            if 'cation_binding' in binding_results and binding_results['cation_binding']:
                all_cations.update(binding_results['cation_binding'].keys())
            if 'anion_binding' in binding_results and binding_results['anion_binding']:
                all_anions.update(binding_results['anion_binding'].keys())
        
        cation_list = sorted(list(all_cations))
        anion_list = sorted(list(all_anions))
        
        if not cation_list and not anion_list:
            print("No binding data to plot")
            return None, None
        
        # Generate colors for targets
        if colors is None:
            cmap = plt.cm.get_cmap(colormap)
            colors = [cmap(i % cmap.N) for i in range(n_targets)]
        elif isinstance(colors, dict):
            colors = [colors.get(label, 'gray') for label in target_labels]
        
        # Generate hatching patterns if not provided
        if hatches is None:
            hatch_options = ['///', '\\\\\\', 'xxx', '...', '|||', '***', 'ooo', '+++']
            hatches = [hatch_options[i % len(hatch_options)] for i in range(n_targets)]
        elif isinstance(hatches, dict):
            # Convert dictionary to list in the order of target_labels
            hatches = [hatches.get(label, '') for label in target_labels]
        elif len(hatches) < n_targets:
            hatches = (hatches * (n_targets // len(hatches) + 1))[:n_targets]
        
        # Determine subplot layout
        if subplot_layout == 'single' or (not cation_list or not anion_list):
            # Single plot with all data
            n_subplots = 1
            subplot_rows, subplot_cols = 1, 1
        elif subplot_layout == 'horizontal':
            n_subplots = 2
            subplot_rows, subplot_cols = 1, 2
        elif subplot_layout == 'vertical':
            n_subplots = 2
            subplot_rows, subplot_cols = 2, 1
        else:
            print(f"Warning: Unknown subplot_layout '{subplot_layout}', using 'horizontal'")
            n_subplots = 2
            subplot_rows, subplot_cols = 1, 2
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            if n_subplots == 1:
                figsize = (max(8, n_targets * len(cation_list + anion_list) * 0.8), 6)
            elif subplot_layout == 'horizontal':
                figsize = (14, 6)
            else:  # vertical
                figsize = (10, 10)
        
        # Create figure and axes
        if n_subplots == 1:
            fig, axes_array = plt.subplots(1, 1, figsize=figsize)
            axes_array = [axes_array]  # Make it a list for consistent indexing
        else:
            fig, axes_array = plt.subplots(subplot_rows, subplot_cols, figsize=figsize)
            axes_array = axes_array.flatten()
        
        # Helper function to plot grouped bars
        def plot_grouped_bars(ax, ion_list, ion_type_str, subplot_title):
            """Plot grouped bars for given ion list"""
            if not ion_list:
                return
            
            n_ions = len(ion_list)
            x_positions = np.arange(n_ions)
            
            # Calculate bar positions for grouped bars
            total_width = bar_width * n_targets
            start_offset = -total_width / 2 + bar_width / 2
            
            # Plot bars for each target
            for target_idx, target_label in enumerate(target_labels):
                binding_results = binding_results_dict[target_label]
                
                # Get data for this target
                avg_values = []
                std_values = []
                
                for ion_name in ion_list:
                    if ion_type_str == 'cation':
                        ion_data = binding_results.get('cation_binding', {}).get(ion_name, None)
                    else:  # anion
                        ion_data = binding_results.get('anion_binding', {}).get(ion_name, None)
                    
                    if ion_data:
                        # Extract values based on normalization mode
                        if normalize_by_volume:
                            if plot_peaks is None:
                                # Calculate density using selected method
                                if 'peak_analysis' in ion_data and ion_data['peak_analysis']:
                                    if volume_calculation_method == 'weighted_average':
                                        overall_density = self._calculate_overall_density(ion_data['peak_analysis'])
                                        avg_values.append(overall_density if overall_density is not None else 0)
                                    elif volume_calculation_method == 'sum':
                                        # Sum all peak densities
                                        total_density = 0
                                        for peak_name, peak_data in ion_data['peak_analysis'].items():
                                            if 'volume_density' in peak_data and peak_data['volume_density'] is not None:
                                                total_density += peak_data['volume_density']
                                        avg_values.append(total_density)
                                    else:  # 'auto' or invalid - use default
                                        overall_density = self._calculate_overall_density(ion_data['peak_analysis'])
                                        avg_values.append(overall_density if overall_density is not None else 0)
                                else:
                                    avg_values.append(0)  # No volume data available
                            else:
                                # Use specific peak density (calculation method depends on parameter)
                                peak_densities = []
                                if 'peak_analysis' in ion_data:
                                    for peak in plot_peaks:
                                        if (peak in ion_data['peak_analysis'] and 
                                            'volume_density' in ion_data['peak_analysis'][peak]):
                                            density = ion_data['peak_analysis'][peak]['volume_density']
                                            if density is not None:
                                                peak_densities.append(density)
                                
                                if volume_calculation_method == 'sum':
                                    avg_values.append(sum(peak_densities) if peak_densities else 0)
                                else:  # weighted_average, auto, or invalid
                                    avg_values.append(np.mean(peak_densities) if peak_densities else 0)
                        else:
                            # Traditional raw counts
                            avg_values.append(ion_data['average_binding'])
                        
                        # Calculate std if available
                        if 'binding_per_frame' in ion_data:
                            std_values.append(np.std(ion_data['binding_per_frame']))
                        else:
                            std_values.append(0)
                    else:
                        avg_values.append(0)
                        std_values.append(0)
                
                # Calculate bar positions
                bar_positions = x_positions + start_offset + target_idx * bar_width
                
                # Get display label
                display_label = custom_labels.get(target_label, target_label) if custom_labels else target_label
                
                # Plot bars
                bars = ax.bar(bar_positions, avg_values, bar_width,
                             label=display_label,
                             color=colors[target_idx],
                             hatch=hatches[target_idx],
                             edgecolor=edgecolor,
                             linewidth=edgewidth,
                             alpha=bar_alpha)
                
                # Add error bars if requested
                if show_errorbars and any(std_values):
                    ax.errorbar(bar_positions, avg_values, yerr=std_values,
                               fmt='none', ecolor='black', elinewidth=2,
                               capsize=errorbar_capsize, capthick=2, alpha=1.0, zorder=10)
                
                # Add value labels on bars
                if show_values:
                    # Calculate dynamic offset based on current y-axis range
                    if ylim:
                        y_range = ylim[1] - ylim[0]
                    else:
                        y_range = max(avg_values) if avg_values else 1
                    dynamic_offset = y_range * value_offset
                    
                    for bar_pos, avg_val in zip(bar_positions, avg_values):
                        if avg_val > 0:  # Only show label if value is non-zero
                            ax.text(bar_pos, avg_val + dynamic_offset,
                                   value_format.format(avg_val),
                                   ha='center', va='bottom',
                                   fontsize=value_fontsize)
            
            # Set x-axis labels (ion names)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(ion_list, fontsize=tick_fontsize)
            
            # Axis labels
            ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
            ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
            
            # Subplot title
            if subplot_title:
                ax.set_title(subplot_title, fontsize=title_fontsize-2, fontweight=title_fontweight)
            
            # Y-axis tick formatting
            ax.tick_params(axis='y', labelsize=tick_fontsize)
            
            # Grid
            if show_grid:
                ax.grid(True, alpha=grid_alpha, axis=grid_axis, linestyle='--')
            
            # Y-axis limits
            if ylim:
                ax.set_ylim(ylim)
            else:
                # Auto-adjust ylim to accommodate value labels
                current_ylim = ax.get_ylim()
                ax.set_ylim(current_ylim[0], current_ylim[1] * 1.1)
            
            # Legend (only on first subplot or if single plot)
            if show_legend:
                legend = ax.legend(loc=legend_loc, framealpha=legend_framealpha,
                                  fontsize=legend_fontsize, ncol=legend_ncol)
                for text in legend.get_texts():
                    text.set_fontweight(legend_fontweight)
        
        # Plot based on layout
        if subplot_layout == 'single':
            # Combine cations and anions on single plot
            combined_ions = cation_list + anion_list
            plot_grouped_bars(axes_array[0], combined_ions, 'combined', None)
        else:
            # Separate cation and anion subplots
            if cation_list:
                subplot_title_cat = 'Cations' if show_title else None
                plot_grouped_bars(axes_array[0], cation_list, 'cation', subplot_title_cat)
            
            if anion_list and len(axes_array) > 1:
                subplot_title_an = 'Anions' if show_title else None
                plot_grouped_bars(axes_array[1], anion_list, 'anion', subplot_title_an)
        
        # Overall title
        if show_title:
            fig.suptitle(title, fontsize=title_fontsize, fontweight=title_fontweight, y=0.98)
        
        plt.tight_layout()
        
        # Save figure
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                       transparent=transparent_bg)
            print(f"✓ Figure saved: {filename}")
        
        return fig, axes_array

    def plot_ion_binding_peak_breakdown(self, binding_results_dict,
                                      # Peak control
                                      peaks_to_show=None, peak_colors='modified',
                                      # Volume normalization (NEW)
                                      normalize_by_volume=False, density_units='auto',
                                      volume_info_in_title=True, volume_calculation_method='sum',
                                      # Overall plot control  
                                      title='Ion Binding Peak Breakdown',
                                      subplot_layout='horizontal',
                                      # Bar styling
                                      bar_width=0.25, moiety_hatches=None, 
                                      edgecolor='black', edgewidth=1.2, bar_alpha=0.9,
                                      # Value labels on bars
                                      show_values=True, show_peak_values=False,
                                      value_fontsize=9, value_format='{:.2f}',
                                      value_offset=0.05,
                                      # Font & text control
                                      title_fontsize=14, title_fontweight='bold', show_title=True,
                                      label_fontsize=12, label_fontweight='normal',
                                      tick_fontsize=10, legend_fontsize=10, legend_fontweight='normal',
                                      # Axis labels
                                      xlabel='Ion Type', ylabel='Average Number of Bound Ions',
                                      # Legend control
                                      show_legend=True, legend_sections='full', legend_loc='best', legend_framealpha=0.9,
                                      legend_ncol=1, custom_labels=None,
                                      # Grid controlplot_ion_selectivity_peak_breakdown
                                      show_grid=True, grid_alpha=0.3, grid_axis='y',
                                      # Axis limits
                                      ylim=None,
                                      # Figure export control
                                      save_fig=False, filename='ion_binding_peaks.png',
                                      dpi=300, figsize=None, bbox_inches='tight',
                                      transparent_bg=False):
        """
        Plot ion binding with peak contribution breakdown using stacked bars.
        
        This method creates stacked bar charts showing the contribution of different coordination
        peaks (P1, P2, P3, P4) to the total ion binding for each target-ion combination.
        Each stack segment represents a coordination shell, colored according to peak colors,
        with moiety-specific hatching patterns for distinction.
        
        Parameters
        ----------
        binding_results_dict : dict
            Dictionary of binding results from ion_binding_analysis() with peak_analysis data
            Format: {target_label: binding_results_with_peak_analysis}
        
        Peak Control
        ------------
        peaks_to_show : list, optional
            Which peaks to include in breakdown ['P1', 'P2', 'P3', 'P4', 'Bulk']
            If None, automatically detects available peaks (default: None)
        peak_colors : str
            Color scheme for peaks: 'modified' or None (original colors) (default: 'modified')
        
        Volume Normalization (NEW)
        -------------------------
        normalize_by_volume : bool
            Whether to plot volume-normalized densities instead of raw counts (default: False)
        density_units : str
            Units for density display: 'auto', 'per_A3', 'per_nm3' (default: 'auto')  
        volume_info_in_title : bool
            Whether to include volume normalization info in plot title (default: True)
        volume_calculation_method : str
            Method for volume normalization calculations: "weighted_average" (volume-weighted
            average density across peaks, good for comparisons), "sum" (sum of individual peak
            densities, good for breakdowns), "auto" (uses method-appropriate default)
            (default: "sum" for breakdown plots)        
        Overall Plot Control
        --------------------
        title : str
            Overall plot title (default: 'Ion Binding Peak Breakdown')
        subplot_layout : str
            Layout: 'horizontal' (cations/anions side-by-side), 'vertical', or 'single' (default: 'horizontal')
        
        Bar Styling  
        -----------
        bar_width : float
            Width of each bar (default: 0.25)
        moiety_hatches : dict, optional
            Hatching patterns for targets {'quinolone': '///', 'piperazine': 'xxx'}
            Applied across all peak segments for target identification (default: None)
        edgecolor : str
            Bar edge color (default: 'black')
        edgewidth : float
            Bar edge width (default: 1.2)
        bar_alpha : float
            Bar transparency 0-1 (default: 0.9)
        
        Value Labels on Bars
        --------------------
        show_values : bool
            Whether to show total values on top of bars (default: True)
        show_peak_values : bool
            Whether to show individual peak values on each segment (default: False)
        value_fontsize : float
            Font size for value labels (default: 9)
        value_format : str
            Format string for values (default: '{:.2f}')
        value_offset : float
            Vertical offset for total value labels (default: 0.05)
        
        Legend Control
        --------------
        show_legend : bool
            Whether to show legend (default: True)
        legend_sections : str
            Which legend sections to show: 'full' (peaks + moieties), 'peaks' (coordination shells only),
            'moieties' (target patterns only), or 'none' (no legend) (default: 'full')
        legend_loc : str
            Legend location (default: 'best')
        legend_framealpha : float
            Legend background transparency 0-1 (default: 0.9)
        legend_ncol : int
            Number of legend columns (default: 1)
        custom_labels : dict, optional
            Custom display labels for targets (default: None)
        
        [Font, Grid, and Export parameters same as plot_ion_binding_comparison]
        
        Returns
        -------
        fig, axes : matplotlib figure and axes objects
        
        Examples
        --------
        >>> # Peak breakdown analysis
        >>> results = analysis.ion_binding_analysis(
        ...     target_sel=[quinolone, piperazine, carboxylic_acid],
        ...     rdf_boundaries=boundaries_refined,
        ...     peaks={'quinolone-NA': ['P2', 'P3', 'P4'], 'piperazine-NA': ['P1', 'P2']}
        ... )
        
        >>> # Basic peak breakdown plot
        >>> plotter.plot_ion_binding_peak_breakdown(results)
        
        >>> # Advanced styling
        >>> plotter.plot_ion_binding_peak_breakdown(
        ...     results,
        ...     peaks_to_show=['P1', 'P2', 'P3', 'P4'],
        ...     peak_colors='modified',
        ...     moiety_hatches={
        ...         'quinolone': '///',
        ...         'carboxylic_acid': '\\\\\\',
        ...         'piperazine': 'xxx'
        ...     },
        ...     custom_labels={'quinolone': 'Quinolone', 'piperazine': 'Piperazine'},
        ...     show_peak_values=True,
        ...     save_fig=True,
        ...     filename='peak_breakdown.png'
        ... )
        """
        
        # Validate input
        if not isinstance(binding_results_dict, dict):
            print("Error: binding_results_dict must be a dictionary of binding results")
            return None, None
        
        if len(binding_results_dict) == 0:
            print("No binding data to plot")
            return None, None
        
        # Check for peak analysis data
        has_peak_data = False
        for target_label, binding_results in binding_results_dict.items():
            for ion_type in ['cation_binding', 'anion_binding']:
                if ion_type in binding_results:
                    for ion_name, ion_data in binding_results[ion_type].items():
                        if 'peak_analysis' in ion_data:
                            has_peak_data = True
                            break
                if has_peak_data:
                    break
            if has_peak_data:
                break
        
        if not has_peak_data:
            print("Error: No peak_analysis data found. Run ion_binding_analysis with rdf_boundaries parameter.")
            return None, None
        
        # Extract target labels
        target_labels = list(binding_results_dict.keys())
        n_targets = len(target_labels)
        
        # Collect all unique ions and available peaks
        all_cations = set()
        all_anions = set()
        all_peaks = set()
        
        for target_label, binding_results in binding_results_dict.items():
            if 'cation_binding' in binding_results:
                for ion_name, ion_data in binding_results['cation_binding'].items():
                    all_cations.add(ion_name)
                    if 'peak_analysis' in ion_data:
                        all_peaks.update(ion_data['peak_analysis'].keys())
            
            if 'anion_binding' in binding_results:
                for ion_name, ion_data in binding_results['anion_binding'].items():
                    all_anions.add(ion_name)
                    if 'peak_analysis' in ion_data:
                        all_peaks.update(ion_data['peak_analysis'].keys())
        
        cation_list = sorted(list(all_cations))
        anion_list = sorted(list(all_anions))
        
        # Determine peaks to show
        if peaks_to_show is None:
            # Auto-detect peaks in logical order
            peak_order = ['P1', 'P2', 'P3', 'P4', 'Bulk']
            peaks_to_show = [p for p in peak_order if p in all_peaks]
        
        if not peaks_to_show:
            print("No peaks found to display")
            return None, None
        
        print(f"Plotting breakdown for peaks: {peaks_to_show}")
        
        # Dynamic ylabel based on volume normalization (MISSING IMPLEMENTATION ADDED)
        if ylabel == 'Average Number of Bound Ions':  # Default value check
            if normalize_by_volume:
                if density_units == 'per_nm3':
                    ylabel = '(ions/frame/nm³)'
                else:  # 'auto' or 'per_A3'
                    ylabel = '(ions/frame/Å³)'
        
        # Update title to include volume normalization info if requested (MISSING IMPLEMENTATION ADDED)
        if volume_info_in_title and normalize_by_volume:
            if 'Volume-Normalized' not in title:
                title += ' (Volume-Normalized)'
        
        # Define peak colors
        if peak_colors == 'modified':
            peak_color_map = {
                'P1': 'lightcoral',
                'P2': 'lightgreen', 
                'P3': 'lightyellow',
                'P4': 'lightblue',
                'Bulk': 'aliceblue'
            }
        else:
            # Original colors
            peak_color_map = {
                'P1': 'lightcoral',
                'P2': 'lightblue',
                'P3': 'lightgreen', 
                'P4': 'lightgoldenrodyellow',
                'Bulk': 'lightyellow'
            }
        
        # Generate hatching patterns for targets
        if moiety_hatches is None:
            hatch_options = ['///', '\\\\\\', 'xxx', '...', '|||', '***', 'ooo', '+++']
            moiety_hatches = {label: hatch_options[i % len(hatch_options)] 
                             for i, label in enumerate(target_labels)}
        
        # Determine subplot layout
        if subplot_layout == 'single' or (not cation_list or not anion_list):
            n_subplots = 1
            subplot_rows, subplot_cols = 1, 1
        elif subplot_layout == 'horizontal':
            n_subplots = 2
            subplot_rows, subplot_cols = 1, 2
        elif subplot_layout == 'vertical':
            n_subplots = 2
            subplot_rows, subplot_cols = 2, 1
        else:
            print(f"Warning: Unknown subplot_layout '{subplot_layout}', using 'horizontal'")
            n_subplots = 2
            subplot_rows, subplot_cols = 1, 2
        
        # Auto-calculate figure size
        if figsize is None:
            if n_subplots == 1:
                figsize = (max(10, n_targets * len(cation_list + anion_list) * 0.8), 7)
            elif subplot_layout == 'horizontal':
                figsize = (16, 7)
            else:  # vertical
                figsize = (10, 12)
        
        # Create figure and axes
        if n_subplots == 1:
            fig, axes_array = plt.subplots(1, 1, figsize=figsize)
            axes_array = [axes_array]
        else:
            fig, axes_array = plt.subplots(subplot_rows, subplot_cols, figsize=figsize)
            axes_array = axes_array.flatten()
        
        # Helper function to plot stacked bars
        def plot_stacked_bars(ax, ion_list, ion_type_str, subplot_title):
            """Plot stacked bars for given ion list"""
            if not ion_list:
                return
            
            n_ions = len(ion_list)
            x_positions = np.arange(n_ions)
            
            # Calculate bar positions for grouped targets
            total_width = bar_width * n_targets
            start_offset = -total_width / 2 + bar_width / 2
            
            # Plot bars for each target
            for target_idx, target_label in enumerate(target_labels):
                binding_results = binding_results_dict[target_label]
                
                # Calculate bar position for this target
                bar_position = x_positions + start_offset + target_idx * bar_width
                
                # Collect peak data for all ions for this target
                ion_peak_data = []
                ion_totals = []
                
                for ion_name in ion_list:
                    # Get ion data
                    if ion_type_str == 'cation':
                        ion_data = binding_results.get('cation_binding', {}).get(ion_name, {})
                    else:  # anion
                        ion_data = binding_results.get('anion_binding', {}).get(ion_name, {})
                    
                    # Extract peak values (with volume normalization support)
                    peak_values = []
                    for peak in peaks_to_show:
                        if 'peak_analysis' in ion_data and peak in ion_data['peak_analysis']:
                            peak_data = ion_data['peak_analysis'][peak]
                            
                            # Use volume-normalized density or raw counts
                            if normalize_by_volume and 'volume_density' in peak_data and peak_data['volume_density'] is not None:
                                peak_values.append(peak_data['volume_density'])
                            else:
                                peak_values.append(peak_data['average_binding'])
                        else:
                            peak_values.append(0)
                    
                    ion_peak_data.append(peak_values)
                    
                    # Calculate total using selected method
                    if normalize_by_volume and volume_calculation_method == 'weighted_average':
                        # Use volume-weighted average calculation like in comparison plot
                        if 'peak_analysis' in ion_data and ion_data['peak_analysis']:
                            overall_density = self._calculate_overall_density(ion_data['peak_analysis'])
                            ion_totals.append(overall_density if overall_density is not None else 0)
                        else:
                            ion_totals.append(0)
                    else:
                        # Use sum of peak values (default for breakdown plots)
                        ion_totals.append(sum(peak_values))
                
                # Convert to arrays for easier handling
                peak_arrays = [np.array([ion_peak_data[i][j] for i in range(n_ions)]) 
                              for j in range(len(peaks_to_show))]
                
                # Plot stacked segments
                bottom = np.zeros(n_ions)
                for peak_idx, peak in enumerate(peaks_to_show):
                    peak_values = peak_arrays[peak_idx]
                    
                    bars = ax.bar(bar_position, peak_values, bar_width,
                                 bottom=bottom,
                                 color=peak_color_map.get(peak, 'gray'),
                                 hatch=moiety_hatches.get(target_label, ''),
                                 edgecolor=edgecolor,
                                 linewidth=edgewidth,
                                 alpha=bar_alpha)
                    
                    # Add individual peak values if requested
                    if show_peak_values:
                        for i, (pos, val, bot) in enumerate(zip(bar_position, peak_values, bottom)):
                            if val > 0:
                                ax.text(pos, bot + val/2, value_format.format(val),
                                       ha='center', va='center', fontsize=value_fontsize-1)
                    
                    bottom += peak_values
                
                # Add total values on top
                if show_values:
                    # Calculate dynamic offset
                    if ylim:
                        y_range = ylim[1] - ylim[0]
                    else:
                        y_range = max(ion_totals) if ion_totals else 1
                    dynamic_offset = y_range * value_offset
                    
                    for pos, total in zip(bar_position, ion_totals):
                        if total > 0:
                            ax.text(pos, total + dynamic_offset, value_format.format(total),
                                   ha='center', va='bottom', fontsize=value_fontsize)
            
            # Set x-axis labels (ion names)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(ion_list, fontsize=tick_fontsize)
            
            # Fix x-axis limits for single ion case to make bar_width effective
            if n_ions == 1:
                # Calculate actual bar position range for all targets
                all_bar_positions = []
                for target_idx in range(n_targets):
                    bar_pos = start_offset + target_idx * bar_width
                    all_bar_positions.append(bar_pos)
                
                # Set axis limits based on actual bar spread with padding
                if all_bar_positions:
                    min_pos = min(all_bar_positions) - bar_width/2
                    max_pos = max(all_bar_positions) + bar_width/2
                    padding = max(0.5, bar_width * 0.5)  # Adaptive padding
                    ax.set_xlim(min_pos - padding, max_pos + padding)
            
            # Axis labels
            ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
            ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
            
            # Subplot title
            if subplot_title:
                ax.set_title(subplot_title, fontsize=title_fontsize-2, fontweight=title_fontweight)
            
            # Y-axis tick formatting
            ax.tick_params(axis='y', labelsize=tick_fontsize)
            
            # Grid
            if show_grid:
                ax.grid(True, alpha=grid_alpha, axis=grid_axis, linestyle='--')
            
            # Y-axis limits
            if ylim:
                ax.set_ylim(ylim)
            else:
                current_ylim = ax.get_ylim()
                ax.set_ylim(current_ylim[0], current_ylim[1] * 1.15)
            
            # Create sectioned legend
            if show_legend and legend_sections != 'none':
                legend_handles = []
                legend_labels = []
                
                # Add peak color section
                if legend_sections in ['full', 'peaks']:
                    for peak in peaks_to_show:
                        peak_patch = plt.Rectangle((0, 0), 1, 1, 
                                                 facecolor=peak_color_map.get(peak, 'gray'),
                                                 edgecolor='black', linewidth=1)
                        legend_handles.append(peak_patch)
                        legend_labels.append(peak)
                
                # Add moiety hatch section  
                if legend_sections in ['full', 'moieties']:
                    # Add a separator for full legend
                    if legend_sections == 'full' and peaks_to_show:
                        # Add invisible separator
                        separator_patch = plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='none')
                        legend_handles.append(separator_patch)
                        legend_labels.append('')  # Empty label for spacing
                    
                    for target_label in target_labels:
                        display_label = custom_labels.get(target_label, target_label) if custom_labels else target_label
                        moiety_patch = plt.Rectangle((0, 0), 1, 1, 
                                                   facecolor='lightgray',
                                                   hatch=moiety_hatches.get(target_label, ''),
                                                   edgecolor='black', linewidth=1)
                        legend_handles.append(moiety_patch)
                        legend_labels.append(display_label)
                
                # Create the legend
                if legend_handles:
                    legend = ax.legend(legend_handles, legend_labels,
                                     loc=legend_loc, framealpha=legend_framealpha,
                                     fontsize=legend_fontsize, ncol=legend_ncol)
                    for text in legend.get_texts():
                        text.set_fontweight(legend_fontweight)
        
        # Plot based on layout
        if subplot_layout == 'single':
            combined_ions = cation_list + anion_list
            plot_stacked_bars(axes_array[0], combined_ions, 'combined', None)
        else:
            if cation_list:
                subplot_title_cat = 'Cations' if show_title else None
                plot_stacked_bars(axes_array[0], cation_list, 'cation', subplot_title_cat)
            
            if anion_list and len(axes_array) > 1:
                subplot_title_an = 'Anions' if show_title else None
                plot_stacked_bars(axes_array[1], anion_list, 'anion', subplot_title_an)
        
        # Overall title
        if show_title:
            fig.suptitle(title, fontsize=title_fontsize, fontweight=title_fontweight, y=0.98)
        
        plt.tight_layout()
        
        # Save figure
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                       transparent=transparent_bg)
            print(f"✓ Figure saved: {filename}")
        
        return fig, axes_array
    
    def plot_ion_binding_timeseries(self, binding_results, 
                                   # Plot mode control
                                   ion_name=None, target_sel=None, plot_mode='overlay',  # 'overlay', 'separate', 'grid'
                                   # Volume normalization (NEW)
                                   normalize_by_volume=False, density_units='auto',
                                   volume_info_in_title=True, volume_calculation_method='weighted_average',
                                   # Overall plot control
                                   title='Ion Binding Time Series',
                                   # Line styling
                                   linewidth=2, colors=None, colormap='tab10',
                                   linestyles=None, line_alpha=1.0, markers=None,
                                   # Font & text control
                                   title_fontsize=14, title_fontweight='bold', show_title=True,
                                   label_fontsize=12, label_fontweight='normal',
                                   tick_fontsize=10, legend_fontsize=10, legend_fontweight='normal',
                                   # Axis labels
                                   xlabel='Frame', ylabel='Number of Bound Ions',
                                   main_ylabel_offset=-0.15,  # For volume-normalized density ylabel positioning
                                   # Legend control
                                   show_legend=True, legend_loc='best', legend_framealpha=0.9,
                                   legend_ncol=1, custom_labels=None,
                                   # Grid control
                                   show_grid=True, grid_alpha=0.3, grid_linestyle='--',
                                   # Axis limits
                                   xlim=None, ylim=None,
                                   # Figure export control
                                   save_fig=False, filename='binding_timeseries.png',
                                   dpi=300, figsize=None, bbox_inches='tight',
                                   transparent_bg=False,
                                   # Multi-ion figure control (when ion_name is a list)
                                   show_individual_figures=False,
                                   individual_figsize=(8, 6),
                                   individual_titles=None,  # Control individual ion plot titles
                                   save_combined_figure=False,
                                   show_combined_figure=True,
                                   save_individual_figures=True,
                                   # NEW: Peak-specific functionality (backward compatible)
                                   plot_type='total',      # 'total', 'peaks', 'stacked', 'comparison'
                                   peaks_to_show='auto',   # 'auto', ['P1', 'P2'], 'all', None
                                   peak_colors='modified', # Consistent with other methods
                                   peak_linestyles=None,   # ['-', '--', '-.'] for different peaks
                                   stack_alpha=0.7,        # For stacked area charts
                                   show_peak_legend=None): # None=auto, True/False=override
        """
        Plot time series of ion binding events with support for batch results
        
        This method handles both single target and batch (multiple targets) binding results.
        For batch results, creates comparison plots showing time evolution for all targets.
        
        Parameters
        ----------
        binding_results : dict
            Results from ion_binding_analysis(). Can be:
            - Single target result: {'cation_binding': {...}, 'anion_binding': {...}, ...}
            - Batch results: {'target1': {...}, 'target2': {...}, ...}
        
        Plot Mode Control
        -----------------
        ion_name : str, list, or None, optional
            Specific ion(s) to plot:
            - None: plots sum of all ions (default)
            - str: plots single ion (e.g., 'NA', 'K')
            - list: plots multiple ions in side-by-side subplots (e.g., ['NA', 'K'])
        target_sel : list of str, optional
            Specific target(s) to include in plot. If None, plots all targets.
            Example: ['quinolone', 'carboxylic_acid', 'piperazine']
            Only applies to batch results (multiple targets).
        plot_mode : str
            How to display multiple targets (ignored when ion_name is a list):
            - 'overlay': All targets on same plot (default)
            - 'separate': Separate subplots (side-by-side)
            - 'grid': Grid layout (auto-arranged)
        
        Volume Normalization (NEW)
        -------------------------
        normalize_by_volume : bool
            Whether to plot volume-normalized densities instead of raw counts (default: False)
            Essential for cross-concentration and cross-ion comparisons
        density_units : str
            Units for density display: "auto", "per_A3", "per_nm3" (default: "auto")
        volume_info_in_title : bool
            Whether to include volume normalization info in plot title (default: True)
        volume_calculation_method : str
            Method for volume normalization calculations: "weighted_average" (volume-weighted
            average density across peaks, good for time series), "sum" (sum of individual peak
            densities), "auto" (uses method-appropriate default)
            (default: "weighted_average" for timeseries)
                Overall Plot Control
        --------------------
        title : str
            Overall plot title (default: 'Ion Binding Time Series')
        
        Line Styling
        ------------
        linewidth : float or list
            Line width(s) (default: 2)
        colors : list or dict, optional
            Colors for each target. Can be:
            - List: colors applied in order
            - Dict: colors mapped by target label
            If None, uses colormap
        colormap : str
            Matplotlib colormap name (default: 'tab10')
        linestyles : list, optional
            Line styles: '-', '--', '-.', ':' (default: None, all solid)
        line_alpha : float
            Line transparency 0-1 (default: 1.0)
        markers : list, optional
            Marker styles: 'o', 's', '^', 'v' (default: None, no markers)
        
        Font & Text Control
        -------------------
        title_fontsize : float
            Title font size (default: 14)
        title_fontweight : str
            Title font weight (default: 'bold')
        show_title : bool
            Whether to show title (default: True)
        label_fontsize : float
            Axis label font size (default: 12)
        label_fontweight : str
            Axis label font weight (default: 'normal')
        tick_fontsize : float
            Tick label font size (default: 10)
        legend_fontsize : float
            Legend font size (default: 10)
        legend_fontweight : str
            Legend font weight (default: 'normal')
        
        Axis Labels
        -----------
        xlabel : str
            X-axis label (default: 'Frame')
        ylabel : str
            Y-axis label (default: 'Number of Bound Ions')
        
        Legend Control
        --------------
        show_legend : bool
            Whether to show legend (default: True)
        legend_loc : str
            Legend location (default: 'best')
        legend_framealpha : float
            Legend background transparency (default: 0.9)
        legend_ncol : int
            Number of legend columns (default: 1)
        custom_labels : dict, optional
            Custom labels for targets in legend
        
        Grid Control
        ------------
        show_grid : bool
            Whether to show grid (default: True)
        grid_alpha : float
            Grid transparency (default: 0.3)
        grid_linestyle : str
            Grid line style (default: '--')
        
        Axis Limits
        -----------
        xlim : tuple, optional
            X-axis limits
        ylim : tuple, optional
            Y-axis limits
        
        Figure Export Control
        ---------------------
        save_fig : bool
            Whether to save combined figure (backward compatibility) (default: False)
        filename : str
            Output filename for combined figure (default: 'binding_timeseries.png')
        dpi : int
            Resolution for all figures (default: 300)
        figsize : tuple, optional
            Figure size for combined figure in inches (auto-calculated if None)
        bbox_inches : str
            Bounding box for saved figure (default: 'tight')
        transparent_bg : bool
            Whether to save with transparent background (default: False)
        
        Multi-Ion Figure Control (when ion_name is a list)
        ---------------------------------------------------
        show_individual_figures : bool
            Whether to display individual figures for each ion (default: False)
        individual_figsize : tuple
            Figure size for individual ion figures (default: (8, 6))
        individual_titles : None, str, dict, or bool
            Control individual ion plot titles (default: None)
            - None: Auto-generated titles (e.g., "NA+", "K+")
            - str: Template string with {ion} placeholder (e.g., "{ion} Binding")
            - dict: Custom titles per ion ({'NA': 'Sodium', 'K': 'Potassium'})
            - False: No titles on individual plots
            - True: Same as None (auto-generated)
        save_combined_figure : bool
            Whether to save the combined figure with all ions (default: True)
        show_combined_figure : bool
            Whether to display the combined figure (default: True)
        save_individual_figures : bool
            Whether to save separate figures for each ion (default: False)
            Filenames auto-generated from base filename (e.g., 'timeseries.png' → 'timeseries_NA.png')
        
        Peak-Specific Plotting (NEW - Backward Compatible)
        ---------------------------------------------------
        plot_type : str
            Type of timeseries plot (default: 'total')
            - 'total': Current behavior - total binding only (backward compatible)
            - 'peaks': Individual peak lines only (P1, P2, P3 separate lines)
            - 'stacked': Stacked area chart showing peak breakdown
            - 'comparison': Side-by-side comparison of total vs stacked peaks
        peaks_to_show : str, list, or None
            Which peaks to include in peak-based plots (default: 'auto')
            - 'auto': Automatically detect available peaks from data
            - 'all': Include all available peaks (P1, P2, P3, P4, S1, S2, S3, S4)
            - list: Specific peaks to show (e.g., ['P1', 'P2'])
            - None: Disable peak plotting (same as plot_type='total')
        peak_colors : str or dict
            Color scheme for peaks (default: 'modified')
            - 'modified': Enhanced peak colors (P1=lightcoral, P2=lightgreen, P3=lightyellow, P4=lightblue)
            - 'original': Original peak colors (P1=lightcoral, P2=lightblue, P3=lightgreen, P4=lightyellow)
            - dict: Custom color mapping (e.g., {'P1': 'red', 'P2': 'blue'})
        peak_linestyles : list or None
            Line styles for different peaks (default: None, all solid)
            Example: ['-', '--', '-.', ':'] for P1, P2, P3, P4 respectively
        stack_alpha : float
            Transparency for stacked area charts (default: 0.7)
            Only used when plot_type='stacked' or 'comparison'
        show_peak_legend : bool or None
            Whether to show legend for peak breakdown (default: None)
            - None: Auto-enable when showing peaks (plot_type != 'total')
            - True/False: Force enable/disable peak legend
        
        Returns
        -------
        fig, axes : matplotlib figure and axes objects
        
        Examples
        --------
        >>> # Single target
        >>> binding = analysis.ion_binding_analysis(target_sel='resname api')
        >>> plotter.plot_ion_binding_timeseries(binding)
        
        >>> # Batch results - overlay mode
        >>> binding_batch = analysis.ion_binding_analysis(
        ...     target_sel=['name O*', 'name N*', 'name C1']
        ... )
        >>> plotter.plot_ion_binding_timeseries(
        ...     binding_batch,
        ...     plot_mode='overlay',
        ...     custom_labels={'quinolone': 'Quinolone', 'piperazine': 'Piperazine'},
        ...     colors={'quinolone': 'red', 'piperazine': 'blue'}
        ... )
        
        >>> # Batch results - separate subplots
        >>> plotter.plot_ion_binding_timeseries(
        ...     binding_batch,
        ...     plot_mode='separate',
        ...     ion_name='NA'  # Plot only Na+ binding
        ... )
        
        >>> # Multiple ions side-by-side
        >>> plotter.plot_ion_binding_timeseries(
        ...     binding_batch,
        ...     ion_name=['NA', 'K'],  # Creates two subplots
        ...     custom_labels={'quinolone': 'Quinolone', 'piperazine': 'Piperazine'},
        ...     colors={'quinolone': 'red', 'piperazine': 'blue'},
        ...     title='Ion Binding Comparison',
        ...     save_fig=True,
        ...     filename='binding_timeseries_NA_K.png'
        ... )
        
        >>> # NEW: Peak breakdown timeseries
        >>> # Stacked area chart showing P1, P2, P3 contributions
        >>> plotter.plot_ion_binding_timeseries(
        ...     binding_results,
        ...     plot_type='stacked',
        ...     peaks_to_show=['P1', 'P2', 'P3'],
        ...     ion_name='NA'
        ... )
        
        >>> # Individual peak lines
        >>> plotter.plot_ion_binding_timeseries(
        ...     binding_results,
        ...     plot_type='peaks',
        ...     peaks_to_show=['P1', 'P2'],
        ...     peak_linestyles=['-', '--'],
        ...     ion_name='K'
        ... )
        
        >>> # Multiple ions with custom titles
        >>> plotter.plot_ion_binding_timeseries(
        ...     binding_results,
        ...     ion_name=['NA', 'K'],
        ...     plot_type='peaks',
        ...     individual_titles={'NA': 'Sodium Binding', 'K': 'Potassium Binding'},
        ...     peaks_to_show=['P1', 'P2']
        ... )
        
        >>> # Template titles for all ions
        >>> plotter.plot_ion_binding_timeseries(
        ...     binding_results,
        ...     ion_name=['NA', 'K'],
        ...     plot_type='stacked', 
        ...     individual_titles='{ion}⁺ Coordination Shells',
        ...     title='Ion Coordination Analysis'
        ... )
        """
        # Dynamic ylabel based on volume normalization (NEW FUNCTIONALITY)
        volume_main_label = None
        if ylabel == "Number of Bound Ions":  # Default value check
            if normalize_by_volume:
                volume_main_label = "Density"
                if density_units == "per_nm3":
                    ylabel = "(ions/frame/nm³)"
                else:  # "auto" or "per_A3"
                    ylabel = "(ions/frame/Å³)"
        
        # Update title to include volume normalization info if requested
        if volume_info_in_title and normalize_by_volume:
            if "Volume-Normalized" not in title:
                title += " (Volume-Normalized)"
                
        # BACKWARD COMPATIBILITY: If plot_type is 'total', use existing behavior
        if plot_type == 'total':
            # Check if batch results (multiple targets) or single result
            is_batch = self._is_batch_binding_result(binding_results)
            
            if is_batch:
                return self._plot_timeseries_batch(
                    binding_results, ion_name=ion_name, target_sel=target_sel, plot_mode=plot_mode,
                    # Volume normalization parameters
                    normalize_by_volume=normalize_by_volume, density_units=density_units,
                    volume_calculation_method=volume_calculation_method,
                    title=title, linewidth=linewidth, colors=colors, colormap=colormap,
                    linestyles=linestyles, line_alpha=line_alpha, markers=markers,
                    title_fontsize=title_fontsize, title_fontweight=title_fontweight,
                    show_title=show_title, label_fontsize=label_fontsize,
                    label_fontweight=label_fontweight, tick_fontsize=tick_fontsize,
                    legend_fontsize=legend_fontsize, legend_fontweight=legend_fontweight,
                    xlabel=xlabel, ylabel=ylabel, main_ylabel_offset=main_ylabel_offset,
                    show_legend=show_legend,
                    legend_loc=legend_loc, legend_framealpha=legend_framealpha,
                    legend_ncol=legend_ncol, custom_labels=custom_labels,
                    show_grid=show_grid, grid_alpha=grid_alpha, grid_linestyle=grid_linestyle,
                    xlim=xlim, ylim=ylim, save_fig=save_fig, filename=filename,
                    dpi=dpi, figsize=figsize, bbox_inches=bbox_inches,
                    transparent_bg=transparent_bg,
                    show_individual_figures=show_individual_figures,
                    individual_figsize=individual_figsize,
                    save_combined_figure=save_combined_figure,
                    show_combined_figure=show_combined_figure,
                    save_individual_figures=save_individual_figures
                )
            else:
                # Single target - use original logic with new formatting
                return self._plot_timeseries_single(
                    binding_results, ion_name=ion_name,
                    # Volume normalization parameters
                    normalize_by_volume=normalize_by_volume, density_units=density_units,
                    volume_calculation_method=volume_calculation_method,
                    _volume_main_label=volume_main_label,
                    title=title,
                    linewidth=linewidth, colors=colors, line_alpha=line_alpha,
                    title_fontsize=title_fontsize, title_fontweight=title_fontweight,
                    show_title=show_title, label_fontsize=label_fontsize,
                    label_fontweight=label_fontweight, tick_fontsize=tick_fontsize,
                    legend_fontsize=legend_fontsize, legend_fontweight=legend_fontweight,
                    xlabel=xlabel, ylabel=ylabel, main_ylabel_offset=main_ylabel_offset,
                    show_legend=show_legend,
                    legend_loc=legend_loc, legend_framealpha=legend_framealpha,
                    show_grid=show_grid, grid_alpha=grid_alpha, grid_linestyle=grid_linestyle,
                    xlim=xlim, ylim=ylim, save_fig=save_fig, filename=filename,
                    dpi=dpi, figsize=figsize, bbox_inches=bbox_inches,
                    transparent_bg=transparent_bg
                )
        
        # NEW: Peak-specific plotting functionality
        else:
            return self._plot_timeseries_with_peaks(
                binding_results, plot_type=plot_type, peaks_to_show=peaks_to_show,
                peak_colors=peak_colors, peak_linestyles=peak_linestyles,
                stack_alpha=stack_alpha, show_peak_legend=show_peak_legend,
                # Volume normalization parameters
                normalize_by_volume=normalize_by_volume, density_units=density_units,
                volume_calculation_method=volume_calculation_method,
                _volume_main_label=volume_main_label,
                ion_name=ion_name, target_sel=target_sel, plot_mode=plot_mode,
                title=title, linewidth=linewidth, colors=colors, colormap=colormap,
                linestyles=linestyles, line_alpha=line_alpha, markers=markers,
                title_fontsize=title_fontsize, title_fontweight=title_fontweight,
                show_title=show_title, label_fontsize=label_fontsize,
                label_fontweight=label_fontweight, tick_fontsize=tick_fontsize,
                legend_fontsize=legend_fontsize, legend_fontweight=legend_fontweight,
                xlabel=xlabel, ylabel=ylabel, main_ylabel_offset=main_ylabel_offset,
                show_legend=show_legend,
                legend_loc=legend_loc, legend_framealpha=legend_framealpha,
                legend_ncol=legend_ncol, custom_labels=custom_labels,
                show_grid=show_grid, grid_alpha=grid_alpha, grid_linestyle=grid_linestyle,
                xlim=xlim, ylim=ylim, save_fig=save_fig, filename=filename,
                dpi=dpi, figsize=figsize, bbox_inches=bbox_inches,
                transparent_bg=transparent_bg,
                show_individual_figures=show_individual_figures,
                individual_figsize=individual_figsize,
                individual_titles=individual_titles,
                save_combined_figure=save_combined_figure,
                show_combined_figure=show_combined_figure,
                save_individual_figures=save_individual_figures
            )
    
    def _is_batch_binding_result(self, binding_results):
        """Check if binding_results is batch (multiple targets) or single"""
        # Batch results have target names as keys, each containing binding dicts
        # Single results have 'cation_binding', 'anion_binding' as direct keys
        if not isinstance(binding_results, dict):
            return False
        
        # If top level has cation_binding or anion_binding, it's single
        if 'cation_binding' in binding_results or 'anion_binding' in binding_results:
            return False
        
        # If top level doesn't have cation_binding/anion_binding, check if values do (batch structure)
        # In batch: {'quinolone': {'cation_binding': {...}}, 'piperazine': {...}}
        # All values should have similar structure
        for key, value in binding_results.items():
            if isinstance(value, dict):
                # Check if this looks like a binding result (has cation_binding or anion_binding)
                if 'cation_binding' in value or 'anion_binding' in value or 'total_binding_per_frame' in value:
                    # This is batch structure
                    return True
        
        # Otherwise assume single
        return False
    
    def _plot_timeseries_with_peaks(self, binding_results, plot_type='stacked', 
                                   peaks_to_show='auto', peak_colors='modified',
                                   peak_linestyles=None, stack_alpha=0.7, 
                                   show_peak_legend=None,
                                   # Volume normalization parameters
                                   normalize_by_volume=False, density_units='auto',
                                   volume_calculation_method='weighted_average',
                                   _volume_main_label=None,
                                   ion_name=None, target_sel=None, plot_mode='overlay',
                                   title='Ion Binding Time Series', linewidth=2, 
                                   colors=None, colormap='tab10', linestyles=None, 
                                   line_alpha=1.0, markers=None,
                                   title_fontsize=14, title_fontweight='bold',
                                   show_title=True, label_fontsize=12,
                                   label_fontweight='normal', tick_fontsize=10,
                                   legend_fontsize=10, legend_fontweight='normal',
                                   xlabel='Frame', ylabel='Number of Bound Ions',
                                   show_legend=True, legend_loc='best', legend_framealpha=0.9,
                                   legend_ncol=1, custom_labels=None,
                                   show_grid=True, grid_alpha=0.3, grid_linestyle='--',
                                   xlim=None, ylim=None, save_fig=False, filename='binding_timeseries.png',
                                   dpi=300, figsize=None, bbox_inches='tight',
                                   transparent_bg=False, show_individual_figures=False,
                                   individual_figsize=(8, 6), individual_titles=None,
                                   save_combined_figure=False, show_combined_figure=True, 
                                   save_individual_figures=True):
        """
        Handle peak-specific timeseries plotting (stacked, peaks, comparison modes)
        """
        
        # Set up peak colors
        if isinstance(peak_colors, dict):
            color_map = peak_colors
        elif peak_colors == 'modified':
            color_map = {
                'P1': 'lightcoral', 'S1': 'lightcoral',
                'P2': 'lightgreen', 'S2': 'lightgreen', 
                'P3': 'lightyellow', 'S3': 'lightyellow',
                'P4': 'lightblue', 'S4': 'lightblue',
                'Bulk': 'aliceblue'
            }
        else:  # original
            color_map = {
                'P1': 'lightcoral', 'S1': 'lightcoral',
                'P2': 'lightblue', 'S2': 'lightblue',
                'P3': 'lightgreen', 'S3': 'lightgreen', 
                'P4': 'lightyellow', 'S4': 'lightyellow',
                'Bulk': 'lightgoldenrodyellow'
            }
        
        # Check if multiple ions requested
        if isinstance(ion_name, list) and len(ion_name) > 1:
            return self._plot_peak_timeseries_multi_ion(
                binding_results, ion_name, plot_type, peaks_to_show, color_map,
                peak_linestyles, stack_alpha, show_peak_legend, target_sel, plot_mode,
                title, linewidth, line_alpha, title_fontsize, title_fontweight,
                show_title, label_fontsize, label_fontweight, tick_fontsize,
                legend_fontsize, legend_fontweight, xlabel, ylabel, show_legend,
                legend_loc, legend_framealpha, show_grid, grid_alpha, grid_linestyle,
                xlim, ylim, save_fig, filename, dpi, figsize, bbox_inches,
                transparent_bg, show_individual_figures, individual_figsize,
                individual_titles, save_combined_figure, show_combined_figure, 
                save_individual_figures, custom_labels
            )
        
        # Check if batch results
        is_batch = self._is_batch_binding_result(binding_results)
        
        if is_batch:
            return self._plot_peak_timeseries_batch(
                binding_results, plot_type, peaks_to_show, color_map, peak_linestyles,
                stack_alpha, show_peak_legend, ion_name, target_sel, plot_mode,
                title, linewidth, colors, colormap, linestyles, line_alpha, markers,
                title_fontsize, title_fontweight, show_title, label_fontsize,
                label_fontweight, tick_fontsize, legend_fontsize, legend_fontweight,
                xlabel, ylabel, show_legend, legend_loc, legend_framealpha,
                legend_ncol, custom_labels, show_grid, grid_alpha, grid_linestyle,
                xlim, ylim, save_fig, filename, dpi, figsize, bbox_inches,
                transparent_bg, show_individual_figures, individual_figsize,
                save_combined_figure, show_combined_figure, save_individual_figures
            )
        else:
            return self._plot_peak_timeseries_single(
                binding_results, plot_type, peaks_to_show, color_map, peak_linestyles,
                stack_alpha, show_peak_legend, ion_name, title, linewidth,
                line_alpha, title_fontsize, title_fontweight, show_title,
                label_fontsize, label_fontweight, tick_fontsize, legend_fontsize,
                legend_fontweight, xlabel, ylabel, show_legend, legend_loc,
                legend_framealpha, show_grid, grid_alpha, grid_linestyle,
                xlim, ylim, save_fig, filename, dpi, figsize, bbox_inches,
                transparent_bg,
                # Volume normalization parameters
                normalize_by_volume=normalize_by_volume, density_units=density_units,
                volume_calculation_method=volume_calculation_method,
                _volume_main_label=_volume_main_label
            )

    def _plot_peak_timeseries_single(self, binding_results, plot_type, peaks_to_show,
                                    color_map, peak_linestyles, stack_alpha, show_peak_legend,
                                    ion_name, title, linewidth, line_alpha, title_fontsize,
                                    title_fontweight, show_title, label_fontsize,
                                    label_fontweight, tick_fontsize, legend_fontsize,
                                    legend_fontweight, xlabel, ylabel, show_legend,
                                    legend_loc, legend_framealpha, show_grid, grid_alpha,
                                    grid_linestyle, xlim, ylim, save_fig, filename,
                                    dpi, figsize, bbox_inches, transparent_bg,
                                    # Volume normalization parameters
                                    normalize_by_volume=False, density_units='auto',
                                    volume_calculation_method='weighted_average',
                                    _volume_main_label=None):
        """Plot peak timeseries for single target"""
        
        # Determine which ion to plot
        if ion_name is None:
            # Plot sum of all ions
            all_ions = list(binding_results.get('cation_binding', {}).keys()) + \
                      list(binding_results.get('anion_binding', {}).keys())
            ions_to_plot = all_ions
        elif isinstance(ion_name, str):
            ions_to_plot = [ion_name]
        else:
            ions_to_plot = ion_name
        
        # Extract peak data for the first available ion
        peak_data = None
        selected_ion = None
        
        for ion in ions_to_plot:
            if ion in binding_results.get('cation_binding', {}):
                ion_data = binding_results['cation_binding'][ion]
            elif ion in binding_results.get('anion_binding', {}):
                ion_data = binding_results['anion_binding'][ion]
            else:
                continue
                
            if 'peak_analysis' in ion_data:
                peak_data = ion_data['peak_analysis']
                selected_ion = ion
                break
        
        if peak_data is None:
            print("⚠️  No peak analysis data found. Falling back to total binding...")
            # Fall back to original method with volume normalization parameters
            return self._plot_timeseries_single(
                binding_results, ion_name=ion_name, title=title,
                normalize_by_volume=normalize_by_volume, density_units=density_units,
                volume_calculation_method=volume_calculation_method,
                linewidth=linewidth, colors=None, line_alpha=line_alpha,
                title_fontsize=title_fontsize, title_fontweight=title_fontweight,
                show_title=show_title, label_fontsize=label_fontsize,
                label_fontweight=label_fontweight, tick_fontsize=tick_fontsize,
                legend_fontsize=legend_fontsize, legend_fontweight=legend_fontweight,
                xlabel=xlabel, ylabel=ylabel, show_legend=show_legend,
                legend_loc=legend_loc, legend_framealpha=legend_framealpha,
                show_grid=show_grid, grid_alpha=grid_alpha, grid_linestyle=grid_linestyle,
                xlim=xlim, ylim=ylim, save_fig=save_fig, filename=filename,
                dpi=dpi, figsize=figsize, bbox_inches=bbox_inches,
                transparent_bg=transparent_bg
            )
        
        # Determine peaks to show
        available_peaks = list(peak_data.keys())
        if peaks_to_show == 'auto':
            peaks = [p for p in ['P1', 'P2', 'P3', 'P4', 'S1', 'S2', 'S3', 'S4'] if p in available_peaks][:3]
        elif peaks_to_show == 'all':
            peaks = available_peaks
        else:
            peaks = [p for p in peaks_to_show if p in available_peaks]
        
        if not peaks:
            print("⚠️  No valid peaks found. Available peaks:", available_peaks)
            return None
            
        print(f"📊 Plotting {plot_type} timeseries for {selected_ion} with peaks: {peaks}")
        
        # Set up figure
        if figsize is None:
            figsize = (10, 6)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get frame indices
        first_peak_data = peak_data[peaks[0]]
        n_frames = len(first_peak_data['binding_events'])
        frames = np.arange(n_frames)
        
        # Plot based on type
        if plot_type == 'stacked':
            # Stacked area chart
            peak_arrays = []
            peak_labels = []
            peak_colors_list = []
            
            for peak in peaks:
                if peak in peak_data:
                    # Extract appropriate data based on volume normalization
                    if normalize_by_volume:
                        data_key = 'volume_density' if 'volume_density' in peak_data[peak] else 'binding_events'
                    else:
                        data_key = 'binding_events'
                    
                    peak_arrays.append(peak_data[peak][data_key])
                    peak_labels.append(peak)
                    peak_colors_list.append(color_map.get(peak, 'gray'))
            
            if peak_arrays:
                ax.stackplot(frames, *peak_arrays, labels=peak_labels, 
                           colors=peak_colors_list, alpha=stack_alpha,
                           linewidth=linewidth)
                
        elif plot_type == 'peaks':
            # Individual peak lines
            for i, peak in enumerate(peaks):
                if peak in peak_data:
                    # Extract appropriate data based on volume normalization
                    if normalize_by_volume:
                        data_key = 'volume_density' if 'volume_density' in peak_data[peak] else 'binding_events'
                    else:
                        data_key = 'binding_events'
                    
                    linestyle = peak_linestyles[i] if peak_linestyles and i < len(peak_linestyles) else '-'
                    ax.plot(frames, peak_data[peak][data_key], 
                           label=peak, color=color_map.get(peak, 'gray'),
                           linewidth=linewidth, linestyle=linestyle, alpha=line_alpha)
        
        elif plot_type == 'comparison':
            # Show both total binding and individual peaks on the same axis
            # First plot total binding
            if selected_ion in binding_results.get('cation_binding', {}):
                total_data = binding_results['cation_binding'][selected_ion]
            elif selected_ion in binding_results.get('anion_binding', {}):
                total_data = binding_results['anion_binding'][selected_ion]
            else:
                total_data = None
                
            if total_data is not None:
                # Extract appropriate data based on volume normalization
                if normalize_by_volume:
                    if 'volume_density' in total_data:
                        total_binding = total_data['volume_density']
                    else:
                        # Fallback calculation or use binding_events
                        total_binding = total_data['binding_events']
                else:
                    total_binding = total_data['binding_events']
                
                ax.plot(frames, total_binding, label='Total', color='black', 
                       linewidth=linewidth + 1, linestyle='-', alpha=line_alpha)
            
            # Then plot individual peaks with different styles
            for i, peak in enumerate(peaks):
                if peak in peak_data:
                    # Extract appropriate data based on volume normalization
                    if normalize_by_volume:
                        data_key = 'volume_density' if 'volume_density' in peak_data[peak] else 'binding_events'
                    else:
                        data_key = 'binding_events'
                    
                    linestyle = peak_linestyles[i] if peak_linestyles and i < len(peak_linestyles) else '--'
                    ax.plot(frames, peak_data[peak][data_key], 
                           label=peak, color=color_map.get(peak, 'gray'),
                           linewidth=linewidth, linestyle=linestyle, alpha=line_alpha * 0.8)
        
        # Formatting with volume normalization handling
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        
        # Handle ylabel with split formatting for volume normalization
        if normalize_by_volume:
            # Determine units
            if density_units == 'per_nm3':
                units = '(ions/frame/nm³)'
            else:  # 'auto' or 'per_A3'
                units = '(ions/frame/Å³)'
            
            # Simple split formatting like selectivity method
            ax.set_ylabel(units,
                         fontsize=tick_fontsize,
                         fontweight='normal')
            ax.text(-0.2, 0.5, 'Density',
                   transform=ax.transAxes,
                   fontsize=label_fontsize,
                   fontweight=label_fontweight,
                   ha='center', va='center', rotation=90)
        else:
            # Standard ylabel
            ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        if show_title:
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        if show_grid:
            ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
        
        if show_legend:
            ax.legend(loc=legend_loc, framealpha=legend_framealpha, 
                     fontsize=legend_fontsize)
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
            
        plt.tight_layout()
        
        # Save figure
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, 
                       transparent=transparent_bg)
            print(f"✓ Peak timeseries saved: {filename}")
        
        # Only show if this is a standalone single plot, not part of multi-ion processing
        # The multi-ion handler manages display logic separately
        return fig, ax

    def _plot_peak_timeseries_batch(self, binding_results, plot_type, peaks_to_show,
                                   color_map, peak_linestyles, stack_alpha, show_peak_legend,
                                   ion_name, target_sel, plot_mode, title, linewidth,
                                   colors, colormap, linestyles, line_alpha, markers,
                                   title_fontsize, title_fontweight, show_title,
                                   label_fontsize, label_fontweight, tick_fontsize,
                                   legend_fontsize, legend_fontweight, xlabel, ylabel,
                                   show_legend, legend_loc, legend_framealpha,
                                   legend_ncol, custom_labels, show_grid, grid_alpha,
                                   grid_linestyle, xlim, ylim, save_fig, filename,
                                   dpi, figsize, bbox_inches, transparent_bg,
                                   show_individual_figures, individual_figsize,
                                   save_combined_figure, show_combined_figure,
                                   save_individual_figures):
        """Plot peak timeseries for batch results (multiple targets)"""
        
        print("📊 Batch peak timeseries plotting not fully implemented yet.")
        print("   Falling back to individual target peak plots...")
        
        # For now, plot first target with peaks
        if target_sel:
            targets = target_sel
        else:
            targets = list(binding_results.keys())
        
        if targets:
            first_target = targets[0]
            target_data = binding_results[first_target]
            
            return self._plot_peak_timeseries_single(
                target_data, plot_type, peaks_to_show, color_map, peak_linestyles,
                stack_alpha, show_peak_legend, ion_name, title, linewidth,
                line_alpha, title_fontsize, title_fontweight, show_title,
                label_fontsize, label_fontweight, tick_fontsize, legend_fontsize,
                legend_fontweight, xlabel, ylabel, show_legend, legend_loc,
                legend_framealpha, show_grid, grid_alpha, grid_linestyle,
                xlim, ylim, save_fig, filename, dpi, figsize, bbox_inches,
                transparent_bg
            )
        
        return None

    def _plot_peak_timeseries_multi_ion(self, binding_results, ion_list, plot_type,
                                       peaks_to_show, color_map, peak_linestyles,
                                       stack_alpha, show_peak_legend, target_sel,
                                       plot_mode, title, linewidth, line_alpha,
                                       title_fontsize, title_fontweight, show_title,
                                       label_fontsize, label_fontweight, tick_fontsize,
                                       legend_fontsize, legend_fontweight, xlabel, ylabel,
                                       show_legend, legend_loc, legend_framealpha,
                                       show_grid, grid_alpha, grid_linestyle,
                                       xlim, ylim, save_fig, filename, dpi, figsize,
                                       bbox_inches, transparent_bg, show_individual_figures,
                                       individual_figsize, individual_titles,
                                       save_combined_figure, show_combined_figure, 
                                       save_individual_figures, custom_labels):
        """
        Handle peak timeseries plotting for multiple ions (e.g., ['NA', 'K'])
        """
        
        print(f"📊 Creating peak timeseries plots for ions: {ion_list}")
        print(f"   Plot type: {plot_type}, Peaks: {peaks_to_show}")
        
        # Check if this is batch results or single target
        is_batch = self._is_batch_binding_result(binding_results)
        
        # Extract base filename for individual files
        base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        extension = filename.rsplit('.', 1)[1] if '.' in filename else 'png'
        
        individual_figures = []
        
        # Create individual plots for each ion-target combination
        if is_batch:
            # For batch results with multiple targets, create individual plots for each ion-target combo
            if target_sel:
                targets = target_sel
            else:
                targets = list(binding_results.keys())
                
            for ion in ion_list:
                for target_label in targets:
                    print(f"   Processing {ion} + {target_label}...")
                    
                    target_data = binding_results[target_label]
                    
                    # Generate individual ion-target title
                    if individual_titles is False:
                        combo_title = ""
                    elif individual_titles is None or individual_titles is True:
                        ion_display = ion
                        if ion == 'NA':
                            ion_display = r'Na$^+$'
                        elif ion == 'K':
                            ion_display = r'K$^+$'
                        elif ion == 'CL':
                            ion_display = r'Cl$^-$'
                        
                        target_display = custom_labels.get(target_label, target_label) if custom_labels else target_label
                        combo_title = f"{ion_display} + {target_display}"
                    elif isinstance(individual_titles, str):
                        combo_title = individual_titles.format(ion=ion, target=target_label)
                    elif isinstance(individual_titles, dict):
                        combo_title = individual_titles.get(f"{ion}_{target_label}", f"{ion}+ + {target_label}")
                    else:
                        combo_title = f"{ion}+ + {target_label}"
                    
                    # Create individual figure for this ion-target combination
                    fig, ax = self._plot_peak_timeseries_single(
                        target_data, plot_type, peaks_to_show, color_map, peak_linestyles,
                        stack_alpha, show_peak_legend, ion, combo_title, linewidth,
                        line_alpha, title_fontsize, title_fontweight, show_title,
                        label_fontsize, label_fontweight, tick_fontsize, legend_fontsize,
                        legend_fontweight, xlabel, ylabel, show_legend, legend_loc,
                        legend_framealpha, show_grid, grid_alpha, grid_linestyle,
                        xlim, ylim, False, filename, dpi, individual_figsize, bbox_inches,
                        transparent_bg
                    )
                    
                    if fig is not None:
                        individual_figures.append((f"{ion}_{target_label}", fig, ax))
                        
                        # Save individual figure if requested
                        if save_individual_figures:
                            individual_filename = f"{base_name}_{ion}_{target_label}.{extension}"
                            fig.savefig(individual_filename, dpi=dpi, bbox_inches=bbox_inches,
                                      transparent=transparent_bg)
                            print(f"✓ Saved individual peak plot: {individual_filename}")
                        
                        # Show or close individual figure based on user preference
                        if show_individual_figures:
                            plt.show()
                        else:
                            plt.close(fig)
        else:
            # Single target - create individual plots for each ion
            for ion in ion_list:
                print(f"   Processing {ion}...")
                
                # Generate individual ion title
                if individual_titles is False:
                    ion_title = ""
                elif individual_titles is None or individual_titles is True:
                    ion_title = f"{ion}+"
                elif isinstance(individual_titles, str):
                    ion_title = individual_titles.format(ion=ion)
                elif isinstance(individual_titles, dict):
                    ion_title = individual_titles.get(ion, f"{ion}+")
                else:
                    ion_title = f"{ion}+"
                
                # Create individual figure for this ion
                fig, ax = self._plot_peak_timeseries_single(
                    binding_results, plot_type, peaks_to_show, color_map, peak_linestyles,
                    stack_alpha, show_peak_legend, ion, ion_title, linewidth,
                    line_alpha, title_fontsize, title_fontweight, show_title,
                    label_fontsize, label_fontweight, tick_fontsize, legend_fontsize,
                    legend_fontweight, xlabel, ylabel, show_legend, legend_loc,
                    legend_framealpha, show_grid, grid_alpha, grid_linestyle,
                    xlim, ylim, False, filename, dpi, individual_figsize, bbox_inches,
                    transparent_bg
                )
                
                if fig is not None:
                    individual_figures.append((ion, fig, ax))
                    
                    # Save individual figure if requested
                    if save_individual_figures:
                        individual_filename = f"{base_name}_{ion}.{extension}"
                        fig.savefig(individual_filename, dpi=dpi, bbox_inches=bbox_inches,
                                  transparent=transparent_bg)
                        print(f"✓ Saved individual peak plot: {individual_filename}")
                    
                    # Show or close individual figure based on user preference
                    if show_individual_figures:
                        plt.show()
                    else:
                        plt.close(fig)
        
        # Create combined figure if requested or if no individual figures shown
        if save_combined_figure or show_combined_figure:
            # For multi-target, create grid layout: ions × targets
            if is_batch and target_sel and len(target_sel) > 1:
                n_ions = len(ion_list)
                n_targets = len(target_sel)
                
                # Grid layout: rows=ions, cols=targets
                if figsize is None:
                    figsize = (6 * n_targets, 6 * n_ions)
                
                combined_fig, axes = plt.subplots(n_ions, n_targets, figsize=figsize)
                if n_ions == 1 and n_targets == 1:
                    axes = [[axes]]  # Make it 2D for consistency
                elif n_ions == 1:
                    axes = [axes]  # Make it 2D for consistency
                elif n_targets == 1:
                    axes = [[ax] for ax in axes]  # Make it 2D for consistency
                
                for i, ion in enumerate(ion_list):
                    for j, target_label in enumerate(target_sel):
                        print(f"   Creating grid subplot for {ion} + {target_label}...")
                        
                        target_data = binding_results[target_label]
                        
                        # Generate subplot title for this ion-target combination
                        ion_display = ion
                        if ion == 'NA':
                            ion_display = r'Na$^+$'
                        elif ion == 'K':
                            ion_display = r'K$^+$'
                        elif ion == 'CL':
                            ion_display = r'Cl$^-$'
                        
                        target_display = custom_labels.get(target_label, target_label) if custom_labels else target_label
                        subplot_title = f"{ion_display} + {target_display}"
                        
                        # Plot data for this specific ion-target combination
                        self._plot_peak_timeseries_on_axis(
                            axes[i][j], target_data, ion, plot_type, peaks_to_show, color_map,
                            peak_linestyles, stack_alpha, show_peak_legend, subplot_title,
                            linewidth, line_alpha, title_fontsize, title_fontweight,
                            True, label_fontsize, label_fontweight, tick_fontsize,  # show_title=True for subplots
                            legend_fontsize, legend_fontweight, xlabel, ylabel,
                            show_legend, legend_loc, legend_framealpha, show_grid,
                            grid_alpha, grid_linestyle, xlim, ylim
                        )
                
                # Set overall title
                if show_title:
                    combined_fig.suptitle(title, fontsize=title_fontsize + 2, 
                                        fontweight=title_fontweight, y=0.98)
                
            else:
                # Original layout: just ions (no multi-target grid)
                if figsize is None:
                    figsize = (6 * len(ion_list), 6)
                
                combined_fig, axes = plt.subplots(1, len(ion_list), figsize=figsize)
                if len(ion_list) == 1:
                    axes = [axes]
                
                for i, ion in enumerate(ion_list):
                    print(f"   Creating combined subplot for {ion}...")
                    
                    if is_batch:
                        if target_sel:
                            targets = target_sel
                        else:
                            targets = list(binding_results.keys())
                            
                        if targets:
                            target_data = binding_results[targets[0]]  # Use first target
                        else:
                            continue
                    else:
                        target_data = binding_results
                    
                    # Generate subplot title (same logic as individual titles)
                    if individual_titles is False:
                        ion_title = ""
                    elif individual_titles is None or individual_titles is True:
                        ion_title = f"{ion}+"
                    elif isinstance(individual_titles, str):
                        ion_title = individual_titles.format(ion=ion)
                    elif isinstance(individual_titles, dict):
                        ion_title = individual_titles.get(ion, f"{ion}+")
                    else:
                        ion_title = f"{ion}+"
                    
                    # Plot peak data for this ion on the subplot
                    self._plot_peak_timeseries_on_axis(
                        axes[i], target_data, ion, plot_type, peaks_to_show, color_map,
                        peak_linestyles, stack_alpha, show_peak_legend, ion_title,
                        linewidth, line_alpha, title_fontsize, title_fontweight,
                        show_title, label_fontsize, label_fontweight, tick_fontsize,
                        legend_fontsize, legend_fontweight, xlabel, ylabel,
                        show_legend, legend_loc, legend_framealpha, show_grid,
                        grid_alpha, grid_linestyle, xlim, ylim
                    )
                
                # Set overall title
                if show_title:
                    combined_fig.suptitle(title, fontsize=title_fontsize + 2, 
                                        fontweight=title_fontweight, y=0.98)
            
            plt.tight_layout()
            
            # Save combined figure if requested
            if save_combined_figure:
                combined_fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                                   transparent=transparent_bg)
                print(f"✓ Saved combined peak plot: {filename}")
            
            # Show combined figure if requested
            if show_combined_figure:
                plt.show()
                return combined_fig, axes
            else:
                plt.close(combined_fig)
        
        return individual_figures if individual_figures else None

    def _plot_peak_timeseries_on_axis(self, ax, target_data, ion, plot_type, peaks_to_show,
                                     color_map, peak_linestyles, stack_alpha, show_peak_legend,
                                     ion_title, linewidth, line_alpha, title_fontsize,
                                     title_fontweight, show_title, label_fontsize,
                                     label_fontweight, tick_fontsize, legend_fontsize,
                                     legend_fontweight, xlabel, ylabel, show_legend,
                                     legend_loc, legend_framealpha, show_grid,
                                     grid_alpha, grid_linestyle, xlim, ylim):
        """
        Plot peak timeseries data on a specific axis (for subplots)
        """
        
        # Get peak data for this ion
        peak_data = None
        
        if ion in target_data.get('cation_binding', {}):
            ion_data = target_data['cation_binding'][ion]
        elif ion in target_data.get('anion_binding', {}):
            ion_data = target_data['anion_binding'][ion]
        else:
            print(f"⚠️  Ion {ion} not found in binding data")
            return
            
        if 'peak_analysis' not in ion_data:
            print(f"⚠️  No peak analysis data for {ion}")
            return
            
        peak_data = ion_data['peak_analysis']
        
        # Determine peaks to show
        available_peaks = list(peak_data.keys())
        if peaks_to_show == 'auto':
            peaks = [p for p in ['P1', 'P2', 'P3', 'P4', 'S1', 'S2', 'S3', 'S4'] if p in available_peaks][:3]
        elif peaks_to_show == 'all':
            peaks = available_peaks
        else:
            peaks = [p for p in peaks_to_show if p in available_peaks]
        
        if not peaks:
            print(f"⚠️  No valid peaks found for {ion}")
            return
        
        # Get frame indices
        first_peak_data = peak_data[peaks[0]]
        n_frames = len(first_peak_data['binding_events'])
        frames = np.arange(n_frames)
        
        # Plot based on type
        if plot_type == 'stacked':
            # Stacked area chart
            peak_arrays = []
            peak_labels = []
            peak_colors_list = []
            
            for peak in peaks:
                if peak in peak_data:
                    peak_arrays.append(peak_data[peak]['binding_events'])
                    peak_labels.append(peak)
                    peak_colors_list.append(color_map.get(peak, 'gray'))
            
            if peak_arrays:
                ax.stackplot(frames, *peak_arrays, labels=peak_labels, 
                           colors=peak_colors_list, alpha=stack_alpha,
                           linewidth=linewidth)
                
        elif plot_type in ['peaks', 'peak']:  # Support both 'peaks' and 'peak'
            # Individual peak lines
            for i, peak in enumerate(peaks):
                if peak in peak_data:
                    linestyle = peak_linestyles[i] if peak_linestyles and i < len(peak_linestyles) else '-'
                    ax.plot(frames, peak_data[peak]['binding_events'], 
                           label=peak, color=color_map.get(peak, 'gray'),
                           linewidth=linewidth, linestyle=linestyle, alpha=line_alpha)
        
        elif plot_type == 'comparison':
            # Show both total binding and individual peaks on the same axis
            # First plot total binding
            if ion in target_data.get('cation_binding', {}):
                total_binding = target_data['cation_binding'][ion]['binding_events']
            elif ion in target_data.get('anion_binding', {}):
                total_binding = target_data['anion_binding'][ion]['binding_events']
            else:
                total_binding = None
                
            if total_binding is not None:
                ax.plot(frames, total_binding, label='Total', color='black', 
                       linewidth=linewidth + 1, linestyle='-', alpha=line_alpha)
            
            # Then plot individual peaks with different styles
            for i, peak in enumerate(peaks):
                if peak in peak_data:
                    linestyle = peak_linestyles[i] if peak_linestyles and i < len(peak_linestyles) else '--'
                    ax.plot(frames, peak_data[peak]['binding_events'], 
                           label=peak, color=color_map.get(peak, 'gray'),
                           linewidth=linewidth, linestyle=linestyle, alpha=line_alpha * 0.8)
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        if show_title:
            ax.set_title(ion_title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        if show_grid:
            ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
        
        if show_legend:
            ax.legend(loc=legend_loc, framealpha=legend_framealpha, 
                     fontsize=legend_fontsize)
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
    
    def _plot_timeseries_single(self, binding_results, ion_name=None, **kwargs):
        """Plot time series for single target"""
        # Extract volume normalization parameters
        normalize_by_volume = kwargs.get('normalize_by_volume', False)
        volume_calculation_method = kwargs.get('volume_calculation_method', 'weighted_average')
        density_units = kwargs.get('density_units', 'ions/frame/Å³')
        
        figsize = kwargs.get('figsize', (12, 6))
        fig, ax = plt.subplots(figsize=figsize)
        
        if ion_name is None:
            # Plot total binding per frame
            if normalize_by_volume:
                # Need to calculate volume-normalized total from individual ions
                total_data = None
                for ion_type in ['cation_binding', 'anion_binding']:
                    if ion_type in binding_results:
                        for ion, ion_data in binding_results[ion_type].items():
                            volume_data = self._extract_volume_normalized_timeseries(
                                ion_data, volume_calculation_method)
                            if volume_data is not None:
                                if total_data is None:
                                    total_data = np.array(volume_data)
                                else:
                                    total_data += np.array(volume_data)
                
                if total_data is None:
                    print("No volume-normalized data available for total binding")
                    return None, None
            else:
                # Use existing total binding data or calculate from raw events
                if 'total_binding_per_frame' in binding_results:
                    total_data = binding_results['total_binding_per_frame']
                else:
                    # Calculate from individual binding events
                    total_data = None
                    for ion_type in ['cation_binding', 'anion_binding']:
                        if ion_type in binding_results:
                            for ion, ion_data in binding_results[ion_type].items():
                                if 'binding_events' in ion_data:
                                    if total_data is None:
                                        total_data = np.array(ion_data['binding_events'])
                                    else:
                                        total_data += np.array(ion_data['binding_events'])
                    
                    if total_data is None:
                        print("No binding data found")
                        return None, None
            
            frames = np.arange(len(total_data))
            ax.plot(frames, total_data, linewidth=kwargs.get('linewidth', 2),
                   alpha=kwargs.get('line_alpha', 1.0), label='Total Binding')
            
        else:
            # Plot specific ion
            found = False
            for ion_type in ['cation_binding', 'anion_binding']:
                if ion_type in binding_results and ion_name in binding_results[ion_type]:
                    ion_data = binding_results[ion_type][ion_name]
                    
                    if normalize_by_volume:
                        # Use volume-normalized density data
                        data = self._extract_volume_normalized_timeseries(
                            ion_data, volume_calculation_method)
                        if data is None:
                            print(f"No volume data available for {ion_name}")
                            return None, None
                    else:
                        # Use raw binding data
                        if 'binding_per_frame' in ion_data:
                            data = ion_data['binding_per_frame']
                        elif 'binding_events' in ion_data:
                            data = ion_data['binding_events']
                        else:
                            print(f"No binding data found for {ion_name}")
                            return None, None
                    
                    frames = np.arange(len(data))
                    ax.plot(frames, data, linewidth=kwargs.get('linewidth', 2),
                           alpha=kwargs.get('line_alpha', 1.0), label=ion_name)
                    found = True
                    break
            
            if not found:
                print(f"Ion '{ion_name}' not found in binding results")
                return None, None
        
        # Apply formatting
        self._apply_timeseries_formatting(ax, **kwargs)
        
        # Add volume normalization ylabel split formatting AFTER regular formatting
        if normalize_by_volume:
            # Put units as ylabel (closer to axis, small font, not bold)
            density_unit = 'nm³' if density_units == 'per_nm3' else 'Å³'
            ax.set_ylabel(f'(ions/frame/{density_unit})',
                         fontsize=kwargs.get('tick_fontsize', 10),
                         fontweight='normal')
            # Put main label as text (further from axis, large font)
            ax.text(main_ylabel_offset, 0.5, 'Density',
                   transform=ax.transAxes,
                   fontsize=kwargs.get('label_fontsize', 12),
                   fontweight=kwargs.get('label_fontweight', 'normal'),
                   ha='center', va='center', rotation=90)
        
        # Title
        if kwargs.get('show_title', True):
            ax.set_title(kwargs.get('title', 'Ion Binding Time Series'),
                        fontsize=kwargs.get('title_fontsize', 14),
                        fontweight=kwargs.get('title_fontweight', 'bold'))
        
        plt.tight_layout()
        
        if kwargs.get('save_fig', False):
            plt.savefig(kwargs.get('filename', 'binding_timeseries.png'),
                       dpi=kwargs.get('dpi', 300),
                       bbox_inches=kwargs.get('bbox_inches', 'tight'),
                       transparent=kwargs.get('transparent_bg', False))
            print(f"✓ Figure saved: {kwargs.get('filename', 'binding_timeseries.png')}")
        
        return fig, ax
    
    def _plot_timeseries_batch(self, binding_results_dict, ion_name=None, target_sel=None, plot_mode='overlay', **kwargs):
        """Plot time series for batch results (multiple targets)"""
        
        # Check if ion_name is a list (multi-ion plotting)
        if isinstance(ion_name, (list, tuple)):
            return self._plot_timeseries_multi_ion(binding_results_dict, ion_name, plot_mode, target_sel=target_sel, **kwargs)
        
        # Filter targets if target_sel provided
        all_targets = list(binding_results_dict.keys())
        if target_sel is not None:
            target_labels = [t for t in target_sel if t in all_targets]
            if not target_labels:
                print(f"Warning: None of the selected targets {target_sel} found in results. Available: {all_targets}")
                return None, None
            missing = [t for t in target_sel if t not in all_targets]
            if missing:
                print(f"Warning: Targets {missing} not found in results. Plotting: {target_labels}")
        else:
            target_labels = all_targets
        n_targets = len(target_labels)
        
        # Generate colors
        colors = kwargs.get('colors')
        if colors is None:
            cmap = plt.cm.get_cmap(kwargs.get('colormap', 'tab10'))
            colors = [cmap(i % cmap.N) for i in range(n_targets)]
        elif isinstance(colors, dict):
            colors = [colors.get(label, 'gray') for label in target_labels]
        
        # Generate linestyles
        linestyles = kwargs.get('linestyles')
        if linestyles is None:
            linestyles = ['-'] * n_targets
        elif len(linestyles) < n_targets:
            linestyles = (linestyles * (n_targets // len(linestyles) + 1))[:n_targets]
        
        # Determine layout
        if plot_mode == 'overlay':
            # All on one plot
            figsize = kwargs.get('figsize', (12, 6))
            fig, ax = plt.subplots(figsize=figsize)
            axes_array = [ax]
            
            for idx, target_label in enumerate(target_labels):
                result = binding_results_dict[target_label]
                display_label = kwargs.get('custom_labels', {}).get(target_label, target_label) if kwargs.get('custom_labels') else target_label
                
                # Extract data
                if ion_name is None:
                    # Sum all ions' binding events to get total binding per frame
                    data = None
                    for ion_type in ['cation_binding', 'anion_binding']:
                        if ion_type in result:
                            for ion, ion_data in result[ion_type].items():
                                if normalize_by_volume:
                                    # Use volume-normalized density data
                                    volume_data = self._extract_volume_normalized_timeseries(
                                        ion_data, volume_calculation_method)
                                    if volume_data is not None:
                                        if data is None:
                                            data = np.array(volume_data)
                                        else:
                                            data += np.array(volume_data)
                                elif 'binding_events' in ion_data:
                                    # Use raw binding events
                                    if data is None:
                                        data = np.array(ion_data['binding_events'])
                                    else:
                                        data += np.array(ion_data['binding_events'])
                    if data is None:
                        continue
                else:
                    # Get specific ion's binding events
                    data = None
                    for ion_type in ['cation_binding', 'anion_binding']:
                        if ion_type in result and ion_name in result[ion_type]:
                            ion_data = result[ion_type][ion_name]
                            if normalize_by_volume:
                                # Use volume-normalized density data
                                data = self._extract_volume_normalized_timeseries(
                                    ion_data, volume_calculation_method)
                            elif 'binding_events' in ion_data:
                                # Use raw binding events
                                data = ion_data['binding_events']
                            break
                    if data is None:
                        continue
                
                frames = np.arange(len(data))
                ax.plot(frames, data, label=display_label, color=colors[idx],
                       linewidth=kwargs.get('linewidth', 2), linestyle=linestyles[idx],
                       alpha=kwargs.get('line_alpha', 1.0))
            
            # Apply formatting
            self._apply_timeseries_formatting(ax, **kwargs)
            
        elif plot_mode == 'separate':
            # Separate subplots (horizontal)
            figsize = kwargs.get('figsize', (6 * n_targets, 5))
            fig, axes_array = plt.subplots(1, n_targets, figsize=figsize, squeeze=False)
            axes_array = axes_array.flatten()
            
            for idx, target_label in enumerate(target_labels):
                ax = axes_array[idx]
                result = binding_results_dict[target_label]
                display_label = kwargs.get('custom_labels', {}).get(target_label, target_label) if kwargs.get('custom_labels') else target_label
                
                # Extract data
                if ion_name is None:
                    # Sum all ions' binding events to get total binding per frame
                    data = None
                    for ion_type in ['cation_binding', 'anion_binding']:
                        if ion_type in result:
                            for ion, ion_data in result[ion_type].items():
                                if normalize_by_volume:
                                    # Use volume-normalized density data
                                    volume_data = self._extract_volume_normalized_timeseries(
                                        ion_data, volume_calculation_method)
                                    if volume_data is not None:
                                        if data is None:
                                            data = np.array(volume_data)
                                        else:
                                            data += np.array(volume_data)
                                elif 'binding_events' in ion_data:
                                    # Use raw binding events
                                    if data is None:
                                        data = np.array(ion_data['binding_events'])
                                    else:
                                        data += np.array(ion_data['binding_events'])
                    if data is None:
                        continue
                else:
                    # Get specific ion's binding events
                    data = None
                    for ion_type in ['cation_binding', 'anion_binding']:
                        if ion_type in result and ion_name in result[ion_type]:
                            ion_data = result[ion_type][ion_name]
                            if normalize_by_volume:
                                # Use volume-normalized density data
                                data = self._extract_volume_normalized_timeseries(
                                    ion_data, volume_calculation_method)
                            elif 'binding_events' in ion_data:
                                # Use raw binding events
                                data = ion_data['binding_events']
                            break
                    if data is None:
                        continue
                
                frames = np.arange(len(data))
                ax.plot(frames, data, color=colors[idx], linewidth=kwargs.get('linewidth', 2),
                       alpha=kwargs.get('line_alpha', 1.0))
                
                # Subplot title
                ax.set_title(display_label, fontsize=kwargs.get('title_fontsize', 14)-2,
                           fontweight=kwargs.get('title_fontweight', 'bold'))
                
                # Apply formatting
                self._apply_timeseries_formatting(ax, show_legend=False, **kwargs)
        
        elif plot_mode == 'grid':
            # Grid layout
            ncols = min(3, n_targets)
            nrows = int(np.ceil(n_targets / ncols))
            figsize = kwargs.get('figsize', (6 * ncols, 5 * nrows))
            fig, axes_array = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
            axes_array = axes_array.flatten()
            
            for idx, target_label in enumerate(target_labels):
                ax = axes_array[idx]
                result = binding_results_dict[target_label]
                display_label = kwargs.get('custom_labels', {}).get(target_label, target_label) if kwargs.get('custom_labels') else target_label
                
                # Extract data
                if ion_name is None:
                    # Sum all ions' binding events to get total binding per frame
                    data = None
                    for ion_type in ['cation_binding', 'anion_binding']:
                        if ion_type in result:
                            for ion, ion_data in result[ion_type].items():
                                if normalize_by_volume:
                                    # Use volume-normalized density data
                                    volume_data = self._extract_volume_normalized_timeseries(
                                        ion_data, volume_calculation_method)
                                    if volume_data is not None:
                                        if data is None:
                                            data = np.array(volume_data)
                                        else:
                                            data += np.array(volume_data)
                                elif 'binding_events' in ion_data:
                                    # Use raw binding events
                                    if data is None:
                                        data = np.array(ion_data['binding_events'])
                                    else:
                                        data += np.array(ion_data['binding_events'])
                    if data is None:
                        continue
                else:
                    # Get specific ion's binding events
                    data = None
                    for ion_type in ['cation_binding', 'anion_binding']:
                        if ion_type in result and ion_name in result[ion_type]:
                            ion_data = result[ion_type][ion_name]
                            if normalize_by_volume:
                                # Use volume-normalized density data
                                data = self._extract_volume_normalized_timeseries(
                                    ion_data, volume_calculation_method)
                            elif 'binding_events' in ion_data:
                                # Use raw binding events
                                data = ion_data['binding_events']
                            break
                    if data is None:
                        continue
                
                frames = np.arange(len(data))
                ax.plot(frames, data, color=colors[idx], linewidth=kwargs.get('linewidth', 2),
                       alpha=kwargs.get('line_alpha', 1.0))
                
                # Subplot title
                ax.set_title(display_label, fontsize=kwargs.get('title_fontsize', 14)-2,
                           fontweight=kwargs.get('title_fontweight', 'bold'))
                
                # Apply formatting
                self._apply_timeseries_formatting(ax, show_legend=False, **kwargs)
            
            # Hide unused subplots
            for idx in range(n_targets, len(axes_array)):
                axes_array[idx].axis('off')
        
        # Overall title
        if kwargs.get('show_title', True):
            title_text = kwargs.get('title', 'Ion Binding Time Series')
            if ion_name:
                title_text += f' ({ion_name})'
            fig.suptitle(title_text, fontsize=kwargs.get('title_fontsize', 14),
                        fontweight=kwargs.get('title_fontweight', 'bold'), y=0.98)
        
        plt.tight_layout()
        
        if kwargs.get('save_fig', False):
            plt.savefig(kwargs.get('filename', 'binding_timeseries.png'),
                       dpi=kwargs.get('dpi', 300),
                       bbox_inches=kwargs.get('bbox_inches', 'tight'),
                       transparent=kwargs.get('transparent_bg', False))
            print(f"✓ Figure saved: {kwargs.get('filename', 'binding_timeseries.png')}")
        
        return fig, axes_array
    
    def _plot_timeseries_multi_ion(self, binding_results_dict, ion_names, plot_mode, target_sel=None, **kwargs):
        """Plot time series for multiple ions side-by-side with optional individual figures"""
        # Extract volume normalization parameters
        normalize_by_volume = kwargs.get('normalize_by_volume', False)
        volume_calculation_method = kwargs.get('volume_calculation_method', 'weighted_average')
        density_units = kwargs.get('density_units', 'ions/frame/Å³')
        
        # Filter targets if target_sel provided
        all_targets = list(binding_results_dict.keys())
        if target_sel is not None:
            target_labels = [t for t in target_sel if t in all_targets]
            if not target_labels:
                print(f"Warning: None of the selected targets {target_sel} found in results. Available: {all_targets}")
                return None, None
            missing = [t for t in target_sel if t not in all_targets]
            if missing:
                print(f"Warning: Targets {missing} not found in results. Plotting: {target_labels}")
        else:
            target_labels = all_targets
        n_targets = len(target_labels)
        n_ions = len(ion_names)
        
        # Generate colors for targets
        colors = kwargs.get('colors')
        if colors is None:
            cmap = plt.cm.get_cmap(kwargs.get('colormap', 'tab10'))
            colors = [cmap(i % cmap.N) for i in range(n_targets)]
        elif isinstance(colors, dict):
            colors = [colors.get(label, 'gray') for label in target_labels]
        
        # Generate linestyles
        linestyles = kwargs.get('linestyles')
        if linestyles is None:
            linestyles = ['-'] * n_targets
        elif len(linestyles) < n_targets:
            linestyles = (linestyles * (n_targets // len(linestyles) + 1))[:n_targets]
        
        # Helper function to plot single ion
        def plot_single_ion(ax, ion_name):
            """Plot all targets for a single ion on given axes"""
            for target_idx, target_label in enumerate(target_labels):
                result = binding_results_dict[target_label]
                display_label = kwargs.get('custom_labels', {}).get(target_label, target_label) if kwargs.get('custom_labels') else target_label
                
                # Extract data for this specific ion
                data = None
                for ion_type in ['cation_binding', 'anion_binding']:
                    if ion_type in result and ion_name in result[ion_type]:
                        ion_data = result[ion_type][ion_name]
                        if normalize_by_volume:
                            # Use volume-normalized density data
                            data = self._extract_volume_normalized_timeseries(
                                ion_data, volume_calculation_method)
                        elif 'binding_events' in ion_data:
                            # Use raw binding events
                            data = ion_data['binding_events']
                        break
                
                if data is None:
                    continue
                
                frames = np.arange(len(data))
                ax.plot(frames, data, label=display_label, color=colors[target_idx],
                       linewidth=kwargs.get('linewidth', 2), linestyle=linestyles[target_idx],
                       alpha=kwargs.get('line_alpha', 1.0))
            
            # Subplot title showing ion name
            ion_display = ion_name
            if ion_name == 'NA':
                ion_display = r'Na$^+$'
            elif ion_name == 'K':
                ion_display = r'K$^+$'
            elif ion_name == 'MG':
                ion_display = r'Mg$^{2+}$'
            elif ion_name == 'CA':
                ion_display = r'Ca$^{2+}$'
            elif ion_name == 'CL':
                ion_display = r'Cl$^-$'
            
            return ion_display
        
        # === CREATE INDIVIDUAL FIGURES (if requested) ===
        if kwargs.get('save_individual_figures', False) or kwargs.get('show_individual_figures', False):
            # Parse base filename to auto-generate individual filenames
            base_filename = kwargs.get('filename', 'binding_timeseries.png')
            base_name, ext = base_filename.rsplit('.', 1) if '.' in base_filename else (base_filename, 'png')
            
            # Strip existing ion names from base_name to avoid duplicates (e.g., file_NA_NA.png)
            for ion in ion_names:
                if base_name.endswith(f'_{ion}'):
                    base_name = base_name[:-len(f'_{ion}')]
                    break
            
            for ion_name in ion_names:
                # Close any existing figures to ensure clean state
                plt.close('all')
                
                # Create individual figure for this ion
                ind_figsize = kwargs.get('individual_figsize', (8, 6))
                fig_ind, ax_ind = plt.subplots(1, 1, figsize=ind_figsize)
                
                # Plot this ion
                ion_display = plot_single_ion(ax_ind, ion_name)
                
                # Apply formatting 
                kwargs_ind = kwargs.copy()
                self._apply_timeseries_formatting(ax_ind, **kwargs_ind)
                
                # Add volume normalization ylabel split formatting AFTER regular formatting
                if normalize_by_volume:
                    # Put units as ylabel (closer to axis, small font, not bold)
                    density_unit = 'nm³' if density_units == 'per_nm3' else 'Å³'
                    ax_ind.set_ylabel(f'(ions/frame/{density_unit})',
                                     fontsize=kwargs.get('tick_fontsize', 10),
                                     fontweight='normal')
                    # Put main label as text (further from axis, large font)
                    main_ylabel_offset = kwargs.get('main_ylabel_offset', -0.15)
                    ax_ind.text(main_ylabel_offset, 0.5, 'Density',
                               transform=ax_ind.transAxes,
                               fontsize=kwargs.get('label_fontsize', 12),
                               fontweight=kwargs.get('label_fontweight', 'normal'),
                               ha='center', va='center', rotation=90)
                
                # Title
                ax_ind.set_title(f'{ion_display} Binding', 
                                fontsize=kwargs.get('title_fontsize', 14),
                                fontweight=kwargs.get('title_fontweight', 'bold'))
                
                # Add legend before tight_layout
                if kwargs.get('show_legend', True):
                    # Use 'upper right' as reliable default instead of 'best' to avoid placement issues
                    ind_legend_loc = kwargs.get('legend_loc', 'upper right') if kwargs.get('legend_loc') != 'best' else 'upper right'
                    legend = ax_ind.legend(loc=ind_legend_loc,
                                          framealpha=kwargs.get('legend_framealpha', 0.9),
                                          fontsize=kwargs.get('legend_fontsize', 10),
                                          ncol=kwargs.get('legend_ncol', 1))
                    for text in legend.get_texts():
                        text.set_fontweight(kwargs.get('legend_fontweight', 'normal'))
                
                # Apply tight_layout after legend
                fig_ind.tight_layout()
                
                # Save individual figure
                if kwargs.get('save_individual_figures', False):
                    ind_filename = f"{base_name}_{ion_name}.{ext}"
                    fig_ind.savefig(ind_filename,
                                   dpi=kwargs.get('dpi', 300),
                                   bbox_inches=kwargs.get('bbox_inches', 'tight'),
                                   transparent=kwargs.get('transparent_bg', False))
                    print(f"✓ Individual figure saved: {ind_filename}")
                
                # Show or close individual figure
                if kwargs.get('show_individual_figures', False):
                    plt.show()
                else:
                    plt.close(fig_ind)
        
        # === CREATE COMBINED FIGURE ===
        # Use same aspect ratio as individual figures (8x6) scaled by number of ions
        ind_figsize = kwargs.get('individual_figsize', (8, 6))
        default_combined_figsize = (ind_figsize[0] * n_ions, ind_figsize[1])
        # Only use user figsize if explicitly provided (not None)
        user_figsize = kwargs.get('figsize')
        figsize = user_figsize if user_figsize is not None else default_combined_figsize
        fig, axes_array = plt.subplots(1, n_ions, figsize=figsize, squeeze=False)
        axes_array = axes_array.flatten()
        
        # Plot each ion in its own subplot for combined figure
        for ion_idx, ion_name in enumerate(ion_names):
            ax = axes_array[ion_idx]
            ion_display = plot_single_ion(ax, ion_name)
            
            ax.set_title(f'{ion_display} Binding', 
                        fontsize=kwargs.get('title_fontsize', 14),
                        fontweight=kwargs.get('title_fontweight', 'bold'))
            
            # Apply formatting but skip legend (we'll add it manually after tight_layout)
            kwargs_no_legend = kwargs.copy()
            kwargs_no_legend['show_legend'] = False
            
            self._apply_timeseries_formatting(ax, **kwargs_no_legend)
            
            # Add volume normalization ylabel split formatting AFTER regular formatting
            if normalize_by_volume:
                # Put units as ylabel (closer to axis, small font, not bold)
                density_unit = 'nm³' if density_units == 'per_nm3' else 'Å³'
                ax.set_ylabel(f'(ions/frame/{density_unit})',
                             fontsize=kwargs.get('tick_fontsize', 10),
                             fontweight='normal')
                # Put main label as text (further from axis, large font)
                main_ylabel_offset = kwargs.get('main_ylabel_offset', -0.15)
                ax.text(main_ylabel_offset, 0.5, 'Density',
                       transform=ax.transAxes,
                       fontsize=kwargs.get('label_fontsize', 12),
                       fontweight=kwargs.get('label_fontweight', 'normal'),
                       ha='center', va='center', rotation=90)
        
        # Overall title
        if kwargs.get('show_title', True):
            title_text = kwargs.get('title', 'Ion Binding Time Series')
            fig.suptitle(title_text, fontsize=kwargs.get('title_fontsize', 14),
                        fontweight=kwargs.get('title_fontweight', 'bold'), y=0.98)
        
        plt.tight_layout()
        
        # Add legend to first subplot after tight_layout
        if kwargs.get('show_legend', True):
            # Use 'upper right' as reliable default instead of 'best'
            combined_legend_loc = kwargs.get('legend_loc', 'upper right') if kwargs.get('legend_loc') != 'best' else 'upper right'
            legend = axes_array[0].legend(loc=combined_legend_loc,
                                         framealpha=kwargs.get('legend_framealpha', 0.9),
                                         fontsize=kwargs.get('legend_fontsize', 10),
                                         ncol=kwargs.get('legend_ncol', 1))
            for text in legend.get_texts():
                text.set_fontweight(kwargs.get('legend_fontweight', 'normal'))
        
        # Save combined figure (controlled by new parameters)
        if kwargs.get('save_fig', False) and kwargs.get('save_combined_figure', True):
            filename = kwargs.get('filename', 'binding_timeseries.png')
            fig.savefig(filename,
                       dpi=kwargs.get('dpi', 300),
                       bbox_inches=kwargs.get('bbox_inches', 'tight'),
                       transparent=kwargs.get('transparent_bg', False))
            print(f"✓ Combined figure saved: {filename}")
        
        # Show combined figure or close it
        if kwargs.get('show_combined_figure', True):
            plt.show()  # Show the combined figure
        else:
            plt.close(fig)
        
        return fig, axes_array
    
    def _apply_timeseries_formatting(self, ax, **kwargs):
        """Apply common formatting to time series axes"""
        # Axis labels
        ax.set_xlabel(kwargs.get('xlabel', 'Frame'),
                     fontsize=kwargs.get('label_fontsize', 12),
                     fontweight=kwargs.get('label_fontweight', 'normal'))
        
        ylabel = kwargs.get('ylabel', 'Number of Bound Ions')
        # Standard ylabel - no complex split logic
        ax.set_ylabel(ylabel,
                     fontsize=kwargs.get('label_fontsize', 12),
                     fontweight=kwargs.get('label_fontweight', 'normal'))
        
        # Tick formatting
        ax.tick_params(axis='both', labelsize=kwargs.get('tick_fontsize', 10))
        
        # Grid
        if kwargs.get('show_grid', True):
            ax.grid(True, alpha=kwargs.get('grid_alpha', 0.3),
                   linestyle=kwargs.get('grid_linestyle', '--'))
        
        # Axis limits
        if kwargs.get('xlim'):
            ax.set_xlim(kwargs.get('xlim'))
        if kwargs.get('ylim'):
            ax.set_ylim(kwargs.get('ylim'))
        
        # Legend
        if kwargs.get('show_legend', True):
            legend = ax.legend(loc=kwargs.get('legend_loc', 'best'),
                              framealpha=kwargs.get('legend_framealpha', 0.9),
                              fontsize=kwargs.get('legend_fontsize', 10),
                              ncol=kwargs.get('legend_ncol', 1))
            for text in legend.get_texts():
                text.set_fontweight(kwargs.get('legend_fontweight', 'normal'))
    
    def plot_ion_selectivity(self, binding_results, 
                            # Overall plot control
                            title='Ion Selectivity',
                            # Metric selection
                            metric='ratio',
                            # Bar styling
                            bar_width=0.25, colors=None, colormap='Set2',
                            hatches=None, edgecolor='black', edgewidth=1.2,
                            bar_alpha=0.9, xlabel_rotation=0,
                            # Value labels
                            show_values=True, value_fontsize=9, value_format='{:.3f}',
                            value_offset=0.05,
                            # Font & text control
                            title_fontsize=14, title_fontweight='bold', show_title=True,
                            label_fontsize=12, label_fontweight='normal',
                            tick_fontsize=10, legend_fontsize=10, legend_fontweight='normal',
                            # Axis labels
                            xlabel='Ion-target Pair', ylabel='Selectivity Index',
                            main_ylabel_offset=-0.15,
                            # Legend control
                            show_legend=True, legend_loc='best', legend_framealpha=0.9,
                            legend_ncol=1, custom_labels=None,
                            # Grid control
                            show_grid=True, grid_alpha=0.3, grid_axis='y',
                            # Axis limits
                            ylim=None,
                            # Figure export control
                            save_fig=False, filename='ion_selectivity.png',
                            dpi=300, figsize=None, bbox_inches='tight',
                            transparent_bg=False):
        """
        Plot ion selectivity indices for single or multiple targets
        
        Automatically detects batch results (multiple targets) and creates grouped bar plot.
        For single target, creates simple bar plot.
        
        Parameters
        ----------
        binding_results : dict
            Results from ion_binding_analysis(). Can be:
            - Single target result: {'selectivity': {...}, ...}
            - Batch results: {'target1': {...}, 'target2': {...}, ...}
        
        Overall Plot Control
        --------------------
        title : str
            Plot title (default: 'Ion Selectivity')
        
        Metric Selection
        ----------------
        metric : str
            Which selectivity metric to plot (default: 'ratio'):
            - 'ratio': ion1/ion2 ratio. >1 means ion1 preferred, <1 means ion2 preferred
            - 'fraction': ion1/(ion1+ion2) fraction. Ranges 0-1, where 0.5 = equal preference
        
        Bar Styling
        -----------
        bar_width : float
            Width of bars (default: 0.25)
        colors : list or dict, optional
            Colors for each target. Can be:
            - List: colors applied in order
            - Dict: colors mapped by target label
            If None, uses colormap
        colormap : str
            Matplotlib colormap name (default: 'Set2')
        hatches : list or dict, optional
            Hatch patterns for bars: '///', '\\\\\\', 'xxx', '...', etc.
        edgecolor : str
            Bar edge color (default: 'black')
        edgewidth : float
            Bar edge width (default: 1.2)
        bar_alpha : float
            Bar transparency 0-1 (default: 0.9)
        
        Value Labels
        ------------
        show_values : bool
            Whether to show values on bars (default: True)
        value_fontsize : float
            Font size for value labels (default: 9)
        value_format : str
            Format string for values (default: '{:.3f}')
        value_offset : float
            Offset of value labels above bars (default: 0.05)
        
        Font & Text Control
        -------------------
        title_fontsize : float
            Title font size (default: 14)
        title_fontweight : str
            Title font weight (default: 'bold')
        show_title : bool
            Whether to show title (default: True)
        label_fontsize : float
            Axis label font size (default: 12)
        label_fontweight : str
            Axis label font weight (default: 'normal')
        tick_fontsize : float
            Tick label font size (default: 10)
        legend_fontsize : float
            Legend font size (default: 10)
        legend_fontweight : str
            Legend font weight (default: 'normal')
        
        Axis Labels
        -----------
        xlabel : str
            X-axis label (default: 'Ion Pair')
        ylabel : str
            Y-axis label (default: 'Selectivity Index')
        main_ylabel_offset : float
            X-coordinate offset for main ylabel positioning in axes coordinates (default: -0.15)
            More negative values move the label further left. Adjust based on y-axis value width:
            - Normal decimals (1.23): -0.15
            - High precision (1.23456): -0.25
            - Large numbers (1234.5): -0.3
            - Scientific notation (1.2e-04): -0.3
        xlabel_rotation : float
            Rotation angle for x-axis tick labels in degrees (default: 0 for horizontal)
        
        Legend Control
        --------------
        show_legend : bool
            Whether to show legend (default: True, only for batch)
        legend_loc : str
            Legend location (default: 'best')
        legend_framealpha : float
            Legend background transparency (default: 0.9)
        legend_ncol : int
            Number of legend columns (default: 1)
        custom_labels : dict, optional
            Custom labels for targets in legend
        
        Grid Control
        ------------
        show_grid : bool
            Whether to show grid (default: True)
        grid_alpha : float
            Grid transparency (default: 0.3)
        grid_axis : str
            Grid axis: 'both', 'x', 'y' (default: 'y')
        
        Axis Limits
        -----------
        ylim : tuple, optional
            Y-axis limits
        
        Figure Export Control
        ---------------------
        save_fig : bool
            Whether to save figure (default: False)
        filename : str
            Output filename (default: 'ion_selectivity.png')
        dpi : int
            Resolution (default: 300)
        figsize : tuple, optional
            Figure size (auto-calculated if None)
        bbox_inches : str
            Bounding box for saved figure (default: 'tight')
        transparent_bg : bool
            Whether to save with transparent background (default: False)
        
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        
        Examples
        --------
        >>> # Single target
        >>> binding = analysis.ion_binding_analysis(target_sel='resname CIP', ion_types=['NA', 'K'])
        >>> plotter.plot_ion_selectivity(binding)
        
        >>> # Batch results (multiple targets)
        >>> binding_batch = analysis.ion_binding_analysis(
        ...     target_sel=['name O*', 'name N*', 'name C1'],
        ...     ion_types=['NA', 'K']
        ... )
        >>> plotter.plot_ion_selectivity(
        ...     binding_batch,
        ...     custom_labels={'quinolone': 'Quinolone', 'carboxylic_acid': 'Carboxylic Acid'},
        ...     colors={'quinolone': 'red', 'carboxylic_acid': 'black'},
        ...     hatches={'quinolone': '///', 'carboxylic_acid': '\\\\\\'},
        ...     bar_width=0.22,
        ...     save_fig=True
        ... )
        """
        
        # Check if batch results (multiple targets) or single result
        is_batch = self._is_batch_binding_result(binding_results)
        
        if is_batch:
            return self._plot_selectivity_batch(
                binding_results, title=title, metric=metric, bar_width=bar_width,
                colors=colors, colormap=colormap, hatches=hatches,
                edgecolor=edgecolor, edgewidth=edgewidth, bar_alpha=bar_alpha,
                xlabel_rotation=xlabel_rotation,
                show_values=show_values, value_fontsize=value_fontsize,
                value_format=value_format, value_offset=value_offset,
                title_fontsize=title_fontsize, title_fontweight=title_fontweight,
                show_title=show_title, label_fontsize=label_fontsize,
                label_fontweight=label_fontweight, tick_fontsize=tick_fontsize,
                legend_fontsize=legend_fontsize, legend_fontweight=legend_fontweight,
                xlabel=xlabel, ylabel=ylabel, main_ylabel_offset=main_ylabel_offset,
                show_legend=show_legend, legend_loc=legend_loc,
                legend_framealpha=legend_framealpha, legend_ncol=legend_ncol,
                custom_labels=custom_labels, show_grid=show_grid,
                grid_alpha=grid_alpha, grid_axis=grid_axis, ylim=ylim,
                save_fig=save_fig, filename=filename, dpi=dpi,
                figsize=figsize, bbox_inches=bbox_inches,
                transparent_bg=transparent_bg
            )
        else:
            return self._plot_selectivity_single(
                binding_results, title=title, metric=metric, colors=colors,
                edgecolor=edgecolor, edgewidth=edgewidth, bar_alpha=bar_alpha,
                xlabel_rotation=xlabel_rotation,
                show_values=show_values, value_fontsize=value_fontsize,
                value_format=value_format, value_offset=value_offset,
                title_fontsize=title_fontsize, title_fontweight=title_fontweight,
                show_title=show_title, label_fontsize=label_fontsize,
                label_fontweight=label_fontweight, tick_fontsize=tick_fontsize,
                xlabel=xlabel, ylabel=ylabel, main_ylabel_offset=main_ylabel_offset,
                show_grid=show_grid, grid_alpha=grid_alpha, grid_axis=grid_axis,
                ylim=ylim, save_fig=save_fig, filename=filename, dpi=dpi,
                figsize=figsize, bbox_inches=bbox_inches,
                transparent_bg=transparent_bg
            )
    
    def _plot_selectivity_single(self, binding_results, **kwargs):
        """Plot selectivity for single target"""
        if 'selectivity' not in binding_results or not binding_results['selectivity']:
            print("No selectivity data to plot")
            return None, None
        
        selectivity = binding_results['selectivity']
        pairs = list(selectivity.keys())
        values = list(selectivity.values())
        
        figsize = kwargs.get('figsize', (10, 6))
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get color
        color = kwargs.get('colors')
        if isinstance(color, (list, tuple)):
            color = color[0] if color else 'steelblue'
        elif color is None:
            color = 'steelblue'
        
        # Create bar plot
        bars = ax.bar(pairs, values, color=color, 
                     edgecolor=kwargs.get('edgecolor', 'black'),
                     linewidth=kwargs.get('edgewidth', 1.2),
                     alpha=kwargs.get('bar_alpha', 0.9))
        
        # Add value labels
        if kwargs.get('show_values', True):
            value_format = kwargs.get('value_format', '{:.3f}')
            value_offset = kwargs.get('value_offset', 0.05)
            y_range = max(values) - min(values) if values else 1
            offset = y_range * value_offset
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                       value_format.format(value), ha='center', va='bottom',
                       fontsize=kwargs.get('value_fontsize', 9))
        
        # Determine formula based on metric
        metric = kwargs.get('metric', 'ratio')
        if metric == 'fraction':
            formula = '(K⁺/(K⁺+Na⁺))'
        else:  # ratio
            formula = '(K⁺/Na⁺)'
        
        # Axis labels
        ax.set_xlabel(kwargs.get('xlabel', 'Ion Pair'),
                     fontsize=kwargs.get('label_fontsize', 12),
                     fontweight=kwargs.get('label_fontweight', 'normal'))
        
        # Put formula as ylabel (closer to axis, small font, not bold)
        ax.set_ylabel(formula,
                     fontsize=kwargs.get('tick_fontsize', 10),
                     fontweight='normal')
        
        # Put main label as text (further from axis, large font)
        ax.text(kwargs.get('main_ylabel_offset', -0.15), 0.5, kwargs.get('ylabel', 'Selectivity Index'),
               transform=ax.transAxes,
               fontsize=kwargs.get('label_fontsize', 12),
               fontweight=kwargs.get('label_fontweight', 'normal'),
               ha='center', va='center', rotation=90)
        
        # Title
        if kwargs.get('show_title', True):
            ax.set_title(kwargs.get('title', 'Ion Selectivity'),
                        fontsize=kwargs.get('title_fontsize', 14),
                        fontweight=kwargs.get('title_fontweight', 'bold'))
        
        # Grid
        if kwargs.get('show_grid', True):
            ax.grid(True, alpha=kwargs.get('grid_alpha', 0.3),
                   axis=kwargs.get('grid_axis', 'y'))
        
        # Tick formatting
        ax.tick_params(axis='both', labelsize=kwargs.get('tick_fontsize', 10))
        rotation = kwargs.get('xlabel_rotation', 0)
        ha = 'right' if rotation > 0 else 'center'
        plt.xticks(rotation=rotation, ha=ha)
        
        # Y-axis limits
        if kwargs.get('ylim'):
            ax.set_ylim(kwargs.get('ylim'))
        
        plt.tight_layout()
        
        if kwargs.get('save_fig', False):
            plt.savefig(kwargs.get('filename', 'ion_selectivity.png'),
                       dpi=kwargs.get('dpi', 300),
                       bbox_inches=kwargs.get('bbox_inches', 'tight'),
                       transparent=kwargs.get('transparent_bg', False))
            print(f"✓ Figure saved: {kwargs.get('filename', 'ion_selectivity.png')}")
        
        return fig, ax
    
    def _plot_selectivity_batch(self, binding_results_dict, **kwargs):
        """Plot selectivity for batch results (multiple targets)"""
        # Extract metric from kwargs
        metric = kwargs.get('metric', 'ratio')
        
        target_labels = list(binding_results_dict.keys())
        n_targets = len(target_labels)
        
        # Collect all selectivity data
        selectivity_data = {}
        all_pairs = set()
        
        for target_label in target_labels:
            result = binding_results_dict[target_label]
            if 'selectivity' in result and result['selectivity']:
                selectivity_data[target_label] = result['selectivity']
                all_pairs.update(result['selectivity'].keys())
        
        if not selectivity_data:
            print("No selectivity data to plot")
            return None, None
        
        pairs = sorted(list(all_pairs))
        n_pairs = len(pairs)
        
        # Generate colors for targets
        colors = kwargs.get('colors')
        if colors is None:
            cmap = plt.cm.get_cmap(kwargs.get('colormap', 'Set2'))
            colors = [cmap(i % cmap.N) for i in range(n_targets)]
        elif isinstance(colors, dict):
            colors = [colors.get(label, 'gray') for label in target_labels]
        
        # Generate hatching patterns
        hatches = kwargs.get('hatches')
        if hatches is None:
            hatches = ['', '///', '\\\\\\', 'xxx', '...', '|||', '---', '+++', 'ooo'][:n_targets]
        elif isinstance(hatches, dict):
            hatches = [hatches.get(label, '') for label in target_labels]
        elif len(hatches) < n_targets:
            hatches = (hatches * (n_targets // len(hatches) + 1))[:n_targets]
        
        # Auto-calculate figure size if not provided
        figsize = kwargs.get('figsize')
        if figsize is None:
            width = max(8, 2 * n_pairs + 2)
            figsize = (width, 6)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate bar positions
        bar_width = kwargs.get('bar_width', 0.25)
        x = np.arange(n_pairs)
        
        # Use fixed spacing for bar positioning (independent of bar_width)
        # This ensures bars don't bunch together when bar_width is reduced
        group_spacing = 0.8  # Total width allocated for each group
        offsets = np.linspace(-group_spacing / 2, group_spacing / 2, n_targets)
        
        # Plot grouped bars
        for target_idx, target_label in enumerate(target_labels):
            display_label = kwargs.get('custom_labels', {}).get(target_label, target_label) if kwargs.get('custom_labels') else target_label
            
            # Get selectivity values for this target
            values = []
            for pair in pairs:
                if target_label in selectivity_data and pair in selectivity_data[target_label]:
                    pair_data = selectivity_data[target_label][pair]
                    # Extract the chosen metric (ratio or fraction)
                    if isinstance(pair_data, dict):
                        values.append(pair_data.get(metric, pair_data.get('ratio', 0)))
                    else:
                        # Backward compatibility: if old format (single value), treat as fraction
                        values.append(pair_data)
                else:
                    values.append(0)
            
            # Plot bars
            bars = ax.bar(x + offsets[target_idx], values, bar_width,
                         label=display_label, color=colors[target_idx],
                         edgecolor=kwargs.get('edgecolor', 'black'),
                         linewidth=kwargs.get('edgewidth', 1.2),
                         alpha=kwargs.get('bar_alpha', 0.9),
                         hatch=hatches[target_idx])
            
            # Add value labels
            if kwargs.get('show_values', True):
                value_format = kwargs.get('value_format', '{:.3f}')
                value_offset = kwargs.get('value_offset', 0.05)
                
                for bar, value in zip(bars, values):
                    if value != 0:
                        height = bar.get_height()
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        offset = y_range * value_offset if y_range > 0 else 0.1
                        
                        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                               value_format.format(value), ha='center', va='bottom',
                               fontsize=kwargs.get('value_fontsize', 9))
        
        # Determine formula based on metric
        metric = kwargs.get('metric', 'ratio')
        if metric == 'fraction':
            formula = '(K⁺/(K⁺+Na⁺))'
        else:  # ratio
            formula = '(K⁺/Na⁺)'
        
        # Axis labels
        ax.set_xlabel(kwargs.get('xlabel', 'Ion Pair'),
                     fontsize=kwargs.get('label_fontsize', 12),
                     fontweight=kwargs.get('label_fontweight', 'normal'))
        
        # Put formula as ylabel (closer to axis, small font, not bold)
        ax.set_ylabel(formula,
                     fontsize=kwargs.get('tick_fontsize', 10),
                     fontweight='normal')
        
        # Put main label as text (further from axis, large font)
        ax.text(kwargs.get('main_ylabel_offset', -0.15), 0.5, kwargs.get('ylabel', 'Selectivity Index'),
               transform=ax.transAxes,
               fontsize=kwargs.get('label_fontsize', 12),
               fontweight=kwargs.get('label_fontweight', 'normal'),
               ha='center', va='center', rotation=90)
        
        # Title
        if kwargs.get('show_title', True):
            ax.set_title(kwargs.get('title', 'Ion Selectivity'),
                        fontsize=kwargs.get('title_fontsize', 14),
                        fontweight=kwargs.get('title_fontweight', 'bold'))
        
        # X-ticks
        ax.set_xticks(x)
        rotation = kwargs.get('xlabel_rotation', 0)
        ha = 'right' if rotation > 0 else 'center'
        ax.set_xticklabels(pairs, rotation=rotation, ha=ha)
        ax.tick_params(axis='both', labelsize=kwargs.get('tick_fontsize', 10))
        
        # Legend
        if kwargs.get('show_legend', True):
            legend = ax.legend(loc=kwargs.get('legend_loc', 'best'),
                             framealpha=kwargs.get('legend_framealpha', 0.9),
                             fontsize=kwargs.get('legend_fontsize', 10),
                             ncol=kwargs.get('legend_ncol', 1))
            for text in legend.get_texts():
                text.set_fontweight(kwargs.get('legend_fontweight', 'normal'))
        
        # Grid
        if kwargs.get('show_grid', True):
            ax.grid(True, alpha=kwargs.get('grid_alpha', 0.3),
                   axis=kwargs.get('grid_axis', 'y'))
        
        # Y-axis limits
        if kwargs.get('ylim'):
            ax.set_ylim(kwargs.get('ylim'))
        
        plt.tight_layout()
        
        if kwargs.get('save_fig', False):
            plt.savefig(kwargs.get('filename', 'ion_selectivity.png'),
                       dpi=kwargs.get('dpi', 300),
                       bbox_inches=kwargs.get('bbox_inches', 'tight'),
                       transparent=kwargs.get('transparent_bg', False))
            print(f"✓ Figure saved: {kwargs.get('filename', 'ion_selectivity.png')}")
        
        return fig, ax
    
    def plot_ion_selectivity_peak_breakdown(self, binding_results_dict,
                                           # Volume normalization (NEW)
                                           normalize_by_volume=False, density_units='auto',
                                           volume_calculation_method='weighted_average',
                                           # Peak control
                                           peaks_to_show=None, peak_colors='modified',
                                           # Selectivity metrics
                                           metric='ratio', ion_pair=None,
                                           # Overall plot control  
                                           title='Ion Selectivity Peak Breakdown',
                                           subplot_layout='horizontal',
                                           # Bar styling
                                           bar_width=0.25, moiety_hatches=None, 
                                           edgecolor='black', edgewidth=1.2, bar_alpha=0.9,
                                           # Value labels on bars
                                           show_values=True, show_peak_values=False,
                                           value_fontsize=9, value_format='{:.3f}',
                                           value_offset=0.05,
                                           # Font & text control
                                           title_fontsize=14, title_fontweight='bold', show_title=True,
                                           label_fontsize=12, label_fontweight='normal',
                                           tick_fontsize=10, legend_fontsize=10, legend_fontweight='normal',
                                           # Axis labels
                                           xlabel='Target Type', ylabel='Selectivity Index',
                                           main_ylabel_offset=-0.15,
                                           # Legend control
                                           show_legend=True, legend_sections='full', legend_loc='best', legend_framealpha=0.9,
                                           legend_ncol=1, custom_labels=None,
                                           # Grid control
                                           show_grid=True, grid_alpha=0.3, grid_axis='y',
                                           # Axis limits
                                           ylim=None,
                                           # Figure export control
                                           save_fig=False, filename='ion_selectivity_peaks.png',
                                           dpi=300, figsize=None, bbox_inches='tight',
                                           transparent_bg=False):
        """
        Plot ion selectivity with peak contribution breakdown using stacked bars.
        
        This method creates stacked bar charts showing how different coordination
        peaks (P1, P2, P3, P4) contribute to the total ion selectivity for each target.
        Each stack segment represents a coordination shell's selectivity, colored 
        according to peak colors, with moiety-specific hatching patterns for distinction.
        
        Parameters
        ----------
        binding_results_dict : dict
            Dictionary of binding results from ion_binding_analysis() with peak_analysis data
            Format: {target_label: binding_results_with_peak_analysis}
        
        Peak Control
        ------------
        peaks_to_show : list, optional
            Which peaks to include in breakdown ['P1', 'P2', 'P3', 'P4', 'Bulk']
            If None, automatically detects available peaks (default: None)
        peak_colors : str
            Color scheme for peaks: 'modified' or None (original colors) (default: 'modified')
        
        Selectivity Metrics
        -------------------
        metric : str
            Selectivity metric to plot: 'ratio' or 'fraction' (default: 'ratio')
        ion_pair : tuple, optional
            Ion pair for selectivity calculation (ion1, ion2). If None, auto-detects (default: None)
        
        Overall Plot Control
        --------------------
        title : str
            Overall plot title (default: 'Ion Selectivity Peak Breakdown')
        subplot_layout : str
            Layout: 'horizontal' (targets side-by-side), 'vertical', or 'single' (default: 'horizontal')
        
        Bar Styling  
        -----------
        bar_width : float
            Width of each bar (default: 0.25)
        moiety_hatches : dict, optional
            Hatching patterns for targets {'quinolone': '///', 'piperazine': 'xxx'}
            Applied across all peak segments for target identification (default: None)
        edgecolor : str
            Bar edge color (default: 'black')
        edgewidth : float
            Bar edge width (default: 1.2)
        bar_alpha : float
            Bar transparency 0-1 (default: 0.9)
        
        Value Labels on Bars
        --------------------
        show_values : bool
            Whether to show total values on top of bars (default: True)
        show_peak_values : bool
            Whether to show individual peak values on each segment (default: False)
        value_fontsize : float
            Font size for value labels (default: 9)
        value_format : str
            Format string for values (default: '{:.3f}')
        value_offset : float
            Vertical offset for total value labels (default: 0.05)
        
        Legend Control
        --------------
        show_legend : bool
            Whether to show legend (default: True)
        legend_sections : str
            Which legend sections to show: 'full' (peaks + moieties), 'peaks' (coordination shells only),
            'moieties' (target patterns only), or 'none' (no legend) (default: 'full')
        legend_loc : str
            Legend location (default: 'best')
        legend_framealpha : float
            Legend background transparency 0-1 (default: 0.9)
        legend_ncol : int
            Number of legend columns (default: 1)
        custom_labels : dict, optional
            Custom display labels for targets (default: None)
        
        Axis Labels
        -----------
        xlabel : str
            X-axis label (default: 'Target Type')
        ylabel : str
            Y-axis label (default: 'Selectivity Index')
        main_ylabel_offset : float
            X-coordinate offset for main ylabel positioning in axes coordinates (default: -0.15)
            More negative values move the label further left. Adjust based on y-axis value width:
            - Normal decimals (1.23): -0.15
            - High precision (1.23456): -0.25
            - Large numbers (1234.5): -0.3
            - Scientific notation (1.2e-04): -0.3
        
        [Font, Grid, and Export parameters same as plot_ion_binding_peak_breakdown]
        
        Returns
        -------
        fig, axes : matplotlib figure and axes objects
        
        Examples
        --------
        >>> # Peak selectivity breakdown analysis
        >>> results = analysis.ion_binding_analysis(
        ...     target_sel=[quinolone, piperazine, carboxylic_acid],
        ...     ion_types=['NA', 'K'],
        ...     rdf_boundaries=boundaries_refined,
        ...     peaks={'quinolone-NA': ['P2', 'P3', 'P4'], 'piperazine-NA': ['P1', 'P2']}
        ... )
        
        >>> # Basic peak selectivity plot
        >>> plotter.plot_ion_selectivity_peak_breakdown(results)
        
        >>> # Advanced styling with ion pair specification
        >>> plotter.plot_ion_selectivity_peak_breakdown(
        ...     results,
        ...     ion_pair=('NA', 'K'),
        ...     metric='ratio',
        ...     peaks_to_show=['P1', 'P2', 'P3', 'P4'],
        ...     peak_colors='modified',
        ...     moiety_hatches={
        ...         'quinolone': '///',
        ...         'carboxylic_acid': '\\\\\\',
        ...         'piperazine': 'xxx'
        ...     },
        ...     legend_sections='full',
        ...     show_peak_values=True,
        ...     save_fig=True,
        ...     filename='selectivity_peak_breakdown.png'
        ... )
        """
        
        # Validate input
        if not isinstance(binding_results_dict, dict):
            print("Error: binding_results_dict must be a dictionary of binding results")
            return None, None
        
        if len(binding_results_dict) == 0:
            print("No binding data to plot")
            return None, None
        
        # Check for peak analysis data
        has_peak_data = False
        for target_label, binding_results in binding_results_dict.items():
            for ion_type in ['cation_binding', 'anion_binding']:
                if ion_type in binding_results:
                    for ion_name, ion_data in binding_results[ion_type].items():
                        if 'peak_analysis' in ion_data:
                            has_peak_data = True
                            break
                if has_peak_data:
                    break
            if has_peak_data:
                break
        
        if not has_peak_data:
            print("Error: No peak_analysis data found. Run ion_binding_analysis with rdf_boundaries parameter.")
            return None, None
        
        # Extract target labels
        target_labels = list(binding_results_dict.keys())
        n_targets = len(target_labels)
        
        # Collect all unique ions and available peaks
        all_cations = set()
        all_anions = set()
        all_peaks = set()
        
        for target_label, binding_results in binding_results_dict.items():
            if 'cation_binding' in binding_results:
                all_cations.update(binding_results['cation_binding'].keys())
                for ion_data in binding_results['cation_binding'].values():
                    if 'peak_analysis' in ion_data:
                        all_peaks.update(ion_data['peak_analysis'].keys())
            
            if 'anion_binding' in binding_results:
                all_anions.update(binding_results['anion_binding'].keys())
                for ion_data in binding_results['anion_binding'].values():
                    if 'peak_analysis' in ion_data:
                        all_peaks.update(ion_data['peak_analysis'].keys())
        
        # Auto-detect ion pair if not specified
        if ion_pair is None:
            all_ions = list(all_cations) + list(all_anions)
            if len(all_ions) < 2:
                print("Error: Need at least 2 ions for selectivity calculation")
                return None, None
            ion_pair = tuple(sorted(all_ions)[:2])
            print(f"Auto-detected ion pair: {ion_pair}")
        
        ion1, ion2 = ion_pair
        
        # Determine peaks to show
        if peaks_to_show is None:
            # Auto-detect peaks in logical order
            peak_order = ['P1', 'P2', 'P3', 'P4', 'Bulk']
            peaks_to_show = [p for p in peak_order if p in all_peaks]
        
        if not peaks_to_show:
            print("No peaks found to display")
            return None, None
        
        print(f"Plotting selectivity breakdown for peaks: {peaks_to_show}")
        print(f"Ion pair: {ion1} vs {ion2}")
        
        # Define peak colors
        if peak_colors == 'modified':
            peak_color_map = {
                'P1': 'lightcoral',
                'P2': 'lightgreen', 
                'P3': 'lightyellow',
                'P4': 'lightblue',
                'Bulk': 'aliceblue'
            }
        else:
            # Original colors
            peak_color_map = {
                'P1': 'lightcoral',
                'P2': 'lightblue',
                'P3': 'lightgreen', 
                'P4': 'lightgoldenrodyellow',
                'Bulk': 'lightyellow'
            }
        
        # Generate hatching patterns for targets
        if moiety_hatches is None:
            hatch_options = ['///', '\\\\\\', 'xxx', '...', '|||', '***', 'ooo', '+++']
            moiety_hatches = {label: hatch_options[i % len(hatch_options)] 
                             for i, label in enumerate(target_labels)}
        
        # Auto-calculate figure size (matching original)
        if figsize is None:
            width = max(8, 2 * 1 + 2)  # Assuming 1 ion pair like original
            figsize = (width, 7)
        
        # Create figure and axes
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Calculate bar positions (exactly like original)
        x = np.arange(1)  # Single ion pair position
        
        # Use fixed spacing for bar positioning (independent of bar_width)
        # This ensures bars don't bunch together when bar_width is reduced
        group_spacing = 0.8  # Total width allocated for each group
        offsets = np.linspace(-group_spacing / 2, group_spacing / 2, n_targets)
        
        # Calculate selectivity for each target and peak
        target_selectivity_data = []
        target_totals = []
        
        for target_label in target_labels:
            binding_results = binding_results_dict[target_label]
            
            # First calculate TOTAL binding for each ion across all peaks
            ion1_total_binding = 0
            ion2_total_binding = 0
            
            for ion_type in ['cation_binding', 'anion_binding']:
                if ion_type in binding_results:
                    if ion1 in binding_results[ion_type] and 'peak_analysis' in binding_results[ion_type][ion1]:
                        for peak_data in binding_results[ion_type][ion1]['peak_analysis'].values():
                            if normalize_by_volume:
                                # Use volume-normalized density
                                if 'volume_density' in peak_data and peak_data['volume_density'] is not None:
                                    ion1_total_binding += peak_data['volume_density']
                            else:
                                # Use raw average binding
                                ion1_total_binding += peak_data['average_binding']
                    
                    if ion2 in binding_results[ion_type] and 'peak_analysis' in binding_results[ion_type][ion2]:
                        for peak_data in binding_results[ion_type][ion2]['peak_analysis'].values():
                            if normalize_by_volume:
                                # Use volume-normalized density
                                if 'volume_density' in peak_data and peak_data['volume_density'] is not None:
                                    ion2_total_binding += peak_data['volume_density']
                            else:
                                # Use raw average binding
                                ion2_total_binding += peak_data['average_binding']
            
            # Calculate overall selectivity for this target
            if metric == 'ratio':
                if ion2_total_binding > 0:
                    total_selectivity = ion1_total_binding / ion2_total_binding
                else:
                    total_selectivity = ion1_total_binding if ion1_total_binding > 0 else 1.0
            elif metric == 'fraction':
                total = ion1_total_binding + ion2_total_binding
                total_selectivity = ion1_total_binding / total if total > 0 else 0.5
            else:
                total_selectivity = 0
            
            # Now calculate the PROPORTION of selectivity contributed by each peak
            peak_selectivity = []
            for peak in peaks_to_show:
                # Get peak binding values for both ions
                ion1_peak_binding = 0
                ion2_peak_binding = 0
                
                for ion_type in ['cation_binding', 'anion_binding']:
                    if ion_type in binding_results:
                        if ion1 in binding_results[ion_type] and 'peak_analysis' in binding_results[ion_type][ion1]:
                            if peak in binding_results[ion_type][ion1]['peak_analysis']:
                                peak_data = binding_results[ion_type][ion1]['peak_analysis'][peak]
                                if normalize_by_volume:
                                    # Use volume-normalized density
                                    if 'volume_density' in peak_data and peak_data['volume_density'] is not None:
                                        ion1_peak_binding += peak_data['volume_density']
                                else:
                                    # Use raw average binding
                                    ion1_peak_binding += peak_data['average_binding']
                        
                        if ion2 in binding_results[ion_type] and 'peak_analysis' in binding_results[ion_type][ion2]:
                            if peak in binding_results[ion_type][ion2]['peak_analysis']:
                                peak_data = binding_results[ion_type][ion2]['peak_analysis'][peak]
                                if normalize_by_volume:
                                    # Use volume-normalized density
                                    if 'volume_density' in peak_data and peak_data['volume_density'] is not None:
                                        ion2_peak_binding += peak_data['volume_density']
                                else:
                                    # Use raw average binding
                                    ion2_peak_binding += peak_data['average_binding']
                
                # Calculate this peak's contribution to total selectivity
                # Proportion based on binding contribution
                if ion1_total_binding + ion2_total_binding > 0:
                    peak_contribution = (ion1_peak_binding + ion2_peak_binding) / (ion1_total_binding + ion2_total_binding)
                    peak_selectivity_value = total_selectivity * peak_contribution
                else:
                    peak_selectivity_value = 0
                
                peak_selectivity.append(peak_selectivity_value)
            
            target_selectivity_data.append(peak_selectivity)
            target_totals.append(total_selectivity)  # This should now equal the original selectivity
        
        # Convert to arrays for easier handling
        peak_arrays = [np.array([target_selectivity_data[i][j] for i in range(n_targets)]) 
                      for j in range(len(peaks_to_show))]
        
        # Plot stacked segments for each target
        for target_idx, target_label in enumerate(target_labels):
            # Calculate bar position for this target (using offsets like original)
            bar_position = x[0] + offsets[target_idx]
            
            bottom = 0
            for peak_idx, peak in enumerate(peaks_to_show):
                peak_value = target_selectivity_data[target_idx][peak_idx]
                
                bars = ax.bar(bar_position, peak_value, bar_width,
                             bottom=bottom,
                             color=peak_color_map.get(peak, 'gray'),
                             hatch=moiety_hatches.get(target_label, ''),
                             edgecolor=edgecolor,
                             linewidth=edgewidth,
                             alpha=bar_alpha)
                
                # Add individual peak values if requested
                if show_peak_values:
                    if peak_value > 0.01:  # Only show if meaningful
                        ax.text(bar_position, bottom + peak_value/2, f'{peak_value:.2f}',
                               ha='center', va='center',
                               fontsize=value_fontsize-2,
                               rotation=90 if peak_value < 0.5 else 0)
                
                bottom += peak_value
            
            # Add total values on top
            if show_values:
                total = target_totals[target_idx]
                if total > 0:
                    # Calculate dynamic offset
                    if ylim:
                        y_range = ylim[1] - ylim[0]
                    else:
                        y_range = max(target_totals) if target_totals else 1
                    dynamic_offset = y_range * value_offset
                    
                    ax.text(bar_position, total + dynamic_offset,
                           value_format.format(total),
                           ha='center', va='bottom',
                           fontsize=value_fontsize)
        
        # Set x-axis labels (single ion pair in center like original)
        ax.set_xticks(x)
        pair_label = f"{ion1}_over_{ion2}"
        ax.set_xticklabels([pair_label], fontsize=tick_fontsize)
        
        # Axis labels (exactly matching original plot_ion_selectivity)
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        
        # Determine formula based on metric and volume normalization
        if normalize_by_volume:
            # Volume-normalized selectivity
            density_unit = 'Å³' if density_units == 'auto' or density_units == 'per_A3' else 'nm³'
            if metric == 'fraction':
                formula = f'({ion1}/{ion1}+{ion2}) (ions/frame/{density_unit})'
            else:  # ratio
                formula = f'({ion1}/{ion2}) (ions/frame/{density_unit})'
        else:
            # Raw count selectivity
            if metric == 'fraction':
                formula = '(K⁺/(K⁺+Na⁺))'
            else:  # ratio
                formula = '(K⁺/Na⁺)'
        
        # Put formula as ylabel (closer to axis, small font, not bold)
        ax.set_ylabel(formula,
                     fontsize=tick_fontsize,
                     fontweight='normal')
        
        # Put main label as text (further from axis, large font)
        ax.text(main_ylabel_offset, 0.5, ylabel,
               transform=ax.transAxes,
               fontsize=label_fontsize,
               fontweight=label_fontweight,
               ha='center', va='center', rotation=90)
        
        # Y-axis tick formatting
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        
        # Grid
        if show_grid:
            ax.grid(True, alpha=grid_alpha, axis=grid_axis, linestyle='--')
        
        # Y-axis limits
        if ylim:
            ax.set_ylim(ylim)
        else:
            current_ylim = ax.get_ylim()
            ax.set_ylim(current_ylim[0], current_ylim[1] * 1.15)
        
        # Create sectioned legend
        if show_legend and legend_sections != 'none':
            legend_handles = []
            legend_labels = []
            
            # Add peak color section
            if legend_sections in ['full', 'peaks']:
                for peak in peaks_to_show:
                    peak_patch = plt.Rectangle((0, 0), 1, 1, 
                                             facecolor=peak_color_map.get(peak, 'gray'),
                                             edgecolor='black', linewidth=1)
                    legend_handles.append(peak_patch)
                    legend_labels.append(peak)
            
            # Add moiety hatch section  
            if legend_sections in ['full', 'moieties'] and n_targets > 1:
                # Add a separator for full legend
                if legend_sections == 'full' and peaks_to_show:
                    # Add invisible separator
                    separator_patch = plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='none')
                    legend_handles.append(separator_patch)
                    legend_labels.append('')  # Empty label for spacing
                
                for target_label in target_labels:
                    display_label = custom_labels.get(target_label, target_label) if custom_labels else target_label
                    moiety_patch = plt.Rectangle((0, 0), 1, 1,
                                               facecolor='white',
                                               hatch=moiety_hatches.get(target_label, ''),
                                               edgecolor='black', linewidth=1)
                    legend_handles.append(moiety_patch)
                    legend_labels.append(display_label)
            
            # Create the legend
            if legend_handles:
                legend = ax.legend(legend_handles, legend_labels,
                                 loc=legend_loc, framealpha=legend_framealpha,
                                 fontsize=legend_fontsize, ncol=legend_ncol)
                for text in legend.get_texts():
                    text.set_fontweight(legend_fontweight)
        
        # Overall title
        if show_title:
            full_title = f'{title} ({ion1} vs {ion2})'
            ax.set_title(full_title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        plt.tight_layout()
        
        # Save figure
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                       transparent=transparent_bg)
            print(f"✓ Figure saved: {filename}")
        
        return fig, ax
    
    # =========================================================================
    # ION COMPETITION CORRELATION PLOTTING
    # =========================================================================
    
    def plot_ion_competition_correlation(self, binding_results,
                                        target_sel=None,
                                        ion_pair=('NA', 'K'),
                                        # Volume normalization parameters
                                        normalize_by_volume=False,
                                        density_units='auto',
                                        volume_calculation_method='weighted_average',
                                        # Peak analysis control
                                        peaks_to_show=['total'],
                                        peak_comparison_mode='same',
                                        # Plot type control
                                        plot_type='combined',
                                        show_marginals=True,
                                        # Color/style control
                                        color_by='density',
                                        colormap='viridis',
                                        point_size=20,
                                        point_alpha=0.3,
                                        bubble_scale=10.0,
                                        show_bubble_labels=True,
                                        show_center_markers=True,
                                        marker_size=20,
                                        marker_color='black',
                                        marker_edgecolor='white',
                                        marker_edgewidth=0.5,
                                        hexbin_gridsize=30,
                                        contour_levels=10,
                                        contour_colors='white',
                                        contour_percentile_range=(0, 100),
                                        contour_spacing='linear',
                                        kde_log_transform=False,
                                        colorbar_log_scale=False,
                                        normalize_colorbar=False,
                                        extend='auto',
                                        # Data point overlay (for contour/combined plots)
                                        show_data_points=False,
                                        data_point_marker='+',
                                        data_point_size=30,
                                        data_point_color='black',
                                        data_point_alpha=0.5,
                                        # Statistical display
                                        show_stats=True,
                                        stats_location='print',
                                        stats_fontsize=10,
                                        show_reference_lines=True,
                                        show_summary=False,
                                        # Font control
                                        title_fontsize=14,
                                        title_fontweight='bold',
                                        label_fontsize=12,
                                        label_fontweight='normal',
                                        tick_fontsize=10,
                                        # Grid layout (for multi-target)
                                        subplot_layout=None,
                                        custom_labels=None,
                                        # Figure control
                                        title=None,
                                        xlabel=None,
                                        ylabel=None,
                                        show_grid=True,
                                        grid_alpha=0.3,
                                        xlim=None,
                                        ylim=None,
                                        figsize=None,
                                        # Multi-target figure control
                                        show_individual_figures=False,
                                        individual_figsize=None,
                                        save_individual_figures=False,
                                        show_combined_figure=True,
                                        save_combined_figure=False,
                                        # Single figure control (legacy)
                                        save_fig=False,
                                        filename='ion_competition_correlation.png',
                                        dpi=300,
                                        bbox_inches='tight',
                                        transparent_bg=False):
        """
        Plot Na vs K binding correlation per frame to reveal competition patterns
        
        Shows 2D scatter/hexbin/contour plot of frame-by-frame binding values:
        - Negative correlation: ions compete for same site
        - No correlation: independent binding sites
        - Positive correlation: cooperative binding
        
        Parameters
        ----------
        binding_results : dict
            Results from ion_binding_analysis() - batch or single target
        
        target_sel : str, list, dict, or None
            Which target(s) to plot with configuration options:
            
            Simple usage (backward compatible):
            - None: plots all targets with global parameters
            - str: single target, e.g., 'carboxylic_acid'
            - list: multiple targets, e.g., ['carboxylic_acid', 'quinolone']
            
            Advanced usage (target-specific configuration):
            - dict: target-specific parameters, e.g.:
              {
                  'carboxylic_acid': {'peaks_to_show': ['P1'], 'ion_pair': ('NA', 'K')},
                  'quinolone': {'peaks_to_show': ['P2'], 'ion_pair': ('K', 'RB')},
                  'piperazine': {'peaks_to_show': ['P1', 'P3']}  # uses global ion_pair
              }
              
            Target-specific parameters override global ones:
            - 'peaks_to_show': list of peaks for this target
            - 'ion_pair': tuple of ions for this target
            - 'plot_type': visualization type for this target
            - 'title': custom title for this target
            - Any other parameter from the method signature
        
        ion_pair : tuple of str
            Pair of ions to compare, e.g., ('NA', 'K'), ('K', 'RB')
            First ion = x-axis, second ion = y-axis
        
        Peak Analysis Control
        ---------------------
        peaks_to_show : list of str
            Which coordination shells to analyze (default: ['total']):
            - 'total': Total binding events (all peaks combined)
            - 'P1': First coordination shell
            - 'P2': Second coordination shell 
            - 'P3': Third coordination shell
            - ['P1', 'P2', 'P3']: Multiple peaks create grid of correlation plots
            - ['total', 'P1', 'P2']: Mix of total and peak-specific correlations
        
        peak_comparison_mode : str
            How to compare peaks between ions (default: 'same'):
            - 'same': Compare same peak between ions (Na_P1 vs K_P1, Na_P2 vs K_P2)
            - 'cross': Compare different peaks (Na_total vs K_P1, Na_P1 vs K_P2)
            - 'total_vs_peaks': Compare total of one ion vs individual peaks of other
        
        Peak Analysis Control
        ---------------------
        peaks_to_show : list of str
            Which coordination shells to analyze (default: ['total']):
            - 'total': Total binding events (all peaks combined)
            - 'P1': First coordination shell
            - 'P2': Second coordination shell 
            - 'P3': Third coordination shell
            - ['P1', 'P2', 'P3']: Multiple peaks create grid of correlation plots
            - ['total', 'P1', 'P2']: Mix of total and peak-specific correlations
        
        peak_comparison_mode : str
            How to compare peaks between ions (default: 'same'):
            - 'same': Compare same peak between ions (Na_P1 vs K_P1, Na_P2 vs K_P2)
            - 'cross': Compare different peaks (Na_total vs K_P1, Na_P1 vs K_P2)
            - 'total_vs_peaks': Compare total of one ion vs individual peaks of other
        
        Plot Type Control
        -----------------
        plot_type : str
            Type of visualization (default: 'combined'):
            - 'scatter': Simple scatter plot
            - 'hexbin': Hexagonal binning (better for many points)
            - 'contour': 2D density contour plot
            - 'combined': Hexbin + contours (recommended)
            - 'bubble': Bubble plot - circle size = state frequency (ideal for discrete data)
        
        show_marginals : bool
            Add marginal distribution histograms on sides (default: True)
        
        Color/Style Control
        -------------------
        color_by : str
            How to color points (default: 'density'):
            - 'density': Color by local point density
            - 'time': Color by frame number (early to late)
            - 'total': Color by total binding (Na + K)
            - 'frequency': Color by state frequency (for bubble plot)
            - 'count': Color by raw coordination counts (contour: avoids huge density values with volume normalization)
        
        colormap : str
            Matplotlib colormap name (default: 'viridis')
        
        point_size : float
            Base size of scatter points (default: 20)
            For bubble plot: multiplier for frequency-based sizing
        
        point_alpha : float
            Transparency of scatter points (default: 0.3)
        
        bubble_scale : float
            Scaling factor for bubble sizes (default: 10.0)
            Larger = bigger bubbles. Only used for bubble plot type
        
        show_bubble_labels : bool
            Show frequency labels on largest bubbles (default: True)
            Labels show exact frame count for top 25% most common states
        
        show_center_markers : bool
            Show small marker at exact center of each bubble (default: True)
            Helps identify precise binding state coordinates when bubbles are large
        
        marker_size : float
            Size of center markers (default: 20)
        
        marker_color : str
            Color of center markers (default: 'black')
        
        marker_edgecolor : str
            Edge color of center markers (default: 'white')
            Creates halo effect for better visibility
        
        marker_edgewidth : float
            Width of marker edge (default: 0.5)
        
        hexbin_gridsize : int
            Grid size for hexbin plot (default: 30)
        
        contour_levels : int or array
            Number of contour levels or specific levels (default: 10)
        
        contour_colors : str
            Color for contour lines (default: 'white')
        
        Data Point Overlay (contour/combined plots only)
        -------------------------------------------------
        show_data_points : bool
            Overlay actual discrete data points on plot (default: False)
            Shows real data locations vs smoothed density
        
        data_point_marker : str
            Marker style: '+', 'x', '.', 'o', etc. (default: '+')
        
        data_point_size : float
            Size of data point markers (default: 30)
        
        data_point_color : str
            Color of data point markers (default: 'black')
        
        data_point_alpha : float
            Transparency of markers, 0-1 (default: 0.5)
        
        Statistical Display
        -------------------
        show_stats : bool
            Display correlation statistics (default: True)
        
        stats_location : str
            Where to show statistics (default: 'print'):
            - 'print': Print as text report after cell execution (clean figure)
            - 'figure': Show as text box on figure (classic style)
            - 'both': Show on figure AND print report
            - 'none': Hide statistics (show_stats=False)
        
        stats_fontsize : int
            Font size for statistics text on figure (default: 10)
            Only used when stats_location='figure' or 'both'
        
        show_reference_lines : bool
            Show diagonal and mean lines (default: True)
        
        Font Control
        ------------
        title_fontsize, label_fontsize, tick_fontsize : int
            Font sizes for different elements
        
        title_fontweight, label_fontweight : str
            Font weights: 'normal', 'bold', 'light', 'heavy'
        
        Grid Layout (Multi-target)
        --------------------------
        subplot_layout : tuple or None
            Manual subplot grid layout (nrows, ncols)
            If None, automatically determines best layout
        
        custom_labels : dict
            Custom labels for targets, e.g., {'quinolone': 'Quinolone Ring'}
        
        Figure Control
        --------------
        title, xlabel, ylabel : str
            Custom axis labels and title
        
        show_grid : bool
            Show grid lines (default: True)
        
        xlim, ylim : tuple
            Axis limits, e.g., (0, 5)
        
        figsize : tuple
            Figure size (width, height) for combined figure. Auto-calculated if None
        
        Multi-Target Figure Control
        ---------------------------
        show_individual_figures : bool
            Display individual figure for each target (default: False)
            Useful when you want to see each target separately
        
        individual_figsize : tuple
            Figure size for individual target figures (default: None = auto)
            Only applies when show_individual_figures=True
        
        save_individual_figures : bool
            Save each target as a separate file (default: False)
            Filenames: '{filename_base}_{target}.png'
        
        show_combined_figure : bool
            Display combined multi-panel figure (default: True)
            Set to False to only show/save individual figures
        
        save_combined_figure : bool
            Save the combined multi-panel figure (default: False)
            Uses the 'filename' parameter
        
        Legacy Single Figure Control
        -----------------------------
        save_fig : bool
            Save figure to file (default: False)
            For single target: saves that figure
            For multi-target: same as save_combined_figure
        
        filename : str
            Output filename (default: 'ion_competition_correlation.png')
            For individual figures: base name (adds _{target}.png)
        
        dpi : int
            Resolution for saved figure (default: 300)
        
        Returns
        -------
        fig, axes : matplotlib figure and axes
        
        Examples
        --------
        >>> # Single target with combined hexbin + contours (traditional total binding)
        >>> plotter.plot_ion_competition_correlation(
        ...     binding_results['carboxylic_acid'],
        ...     plot_type='combined',
        ...     show_marginals=True,
        ...     show_stats=True
        ... )
        
        >>> # Peak-specific correlation analysis (P1 shell only)
        >>> plotter.plot_ion_competition_correlation(
        ...     binding_results,
        ...     target_sel=['quinolone', 'carboxylic_acid'],
        ...     peaks_to_show=['P1'],
        ...     plot_type='combined',
        ...     show_marginals=True
        ... )
        
        >>> # Compare multiple peaks for same ion pair
        >>> plotter.plot_ion_competition_correlation(
        ...     binding_results,
        ...     target_sel=['carboxylic_acid'],
        ...     peaks_to_show=['P1', 'P2', 'P3'],
        ...     peak_comparison_mode='same',
        ...     plot_type='hexbin',
        ...     custom_labels={'carboxylic_acid': 'Carboxylic Acid'}
        ... )
        
        >>> # Traditional usage (backward compatible)
        >>> plotter.plot_ion_competition_correlation(
        ...     binding_results,
        ...     target_sel=['quinolone', 'carboxylic_acid', 'piperazine'],
        ...     plot_type='hexbin',
        ...     custom_labels={'quinolone': 'Quinolone', 
        ...                    'carboxylic_acid': 'Carboxylic Acid'}
        ... )
        
        >>> # Advanced target-specific configuration (NEW!)
        >>> plotter.plot_ion_competition_correlation(
        ...     binding_results,
        ...     target_sel={
        ...         'carboxylic_acid': {'peaks_to_show': ['P1'], 'ion_pair': ('NA', 'K')},
        ...         'quinolone': {'peaks_to_show': ['P2'], 'ion_pair': ('K', 'RB')},
        ...         'piperazine': {'peaks_to_show': ['P1', 'P3'], 'plot_type': 'bubble'}
        ...     }
        ... )
        
        >>> # Mixed configuration with custom parameters per target
        >>> plotter.plot_ion_competition_correlation(
        ...     binding_results,
        ...     target_sel={
        ...         'carboxylic_acid': {
        ...             'peaks_to_show': ['P1'], 
        ...             'ion_pair': ('NA', 'K'),
        ...             'plot_type': 'combined',
        ...             'colormap': 'plasma',
        ...             'title': 'Carboxyl P1 Competition'
        ...         },
        ...         'quinolone': {
        ...             'peaks_to_show': ['P2'], 
        ...             'ion_pair': ('NA', 'K'),
        ...             'plot_type': 'hexbin',
        ...             'colormap': 'viridis'
        ...         }
        ...     }
        ... )
        
        >>> # Color by trajectory time
        >>> plotter.plot_ion_competition_correlation(
        ...     binding_results,
        ...     color_by='time',
        ...     colormap='coolwarm'
        ... )
        """
        
        # Check if batch results (multiple targets) or single result
        is_batch = self._is_batch_binding_result(binding_results)
        
        # Validate and normalize peak parameters
        if isinstance(peaks_to_show, str):
            peaks_to_show = [peaks_to_show]
        
        # Check if peak-specific analysis is requested
        has_peak_analysis = any(peak != 'total' for peak in peaks_to_show)
        
        if has_peak_analysis:
            # Route to peak-specific correlation method
            return self._plot_ion_competition_correlation_with_peaks(
                binding_results, target_sel, ion_pair, peaks_to_show, peak_comparison_mode,
                plot_type, show_marginals, color_by, colormap, point_size, point_alpha,
                bubble_scale, show_bubble_labels, show_center_markers, marker_size,
                marker_color, marker_edgecolor, marker_edgewidth, hexbin_gridsize,
                contour_levels, contour_colors, show_data_points, data_point_marker,
                data_point_size, data_point_color, data_point_alpha, show_stats,
                stats_location, stats_fontsize, show_reference_lines, show_summary,
                title_fontsize, title_fontweight, label_fontsize, label_fontweight,
                tick_fontsize, subplot_layout, custom_labels, title, xlabel, ylabel,
                show_grid, grid_alpha, xlim, ylim, figsize, show_individual_figures,
                individual_figsize, save_individual_figures, show_combined_figure,
                save_combined_figure, save_fig, filename, dpi, bbox_inches, transparent_bg,
                # Volume normalization parameters
                normalize_by_volume, density_units, volume_calculation_method
            )
        
        # Check for advanced target-specific configuration
        if isinstance(target_sel, dict):
            # Advanced mode: target-specific parameters
            return self._plot_ion_competition_correlation_advanced(
                binding_results, target_sel, ion_pair, peaks_to_show, peak_comparison_mode,
                plot_type, show_marginals, color_by, colormap, point_size, point_alpha,
                bubble_scale, show_bubble_labels, show_center_markers, marker_size,
                marker_color, marker_edgecolor, marker_edgewidth, hexbin_gridsize,
                contour_levels, contour_colors, show_data_points, data_point_marker,
                data_point_size, data_point_color, data_point_alpha, show_stats,
                stats_location, stats_fontsize, show_reference_lines, show_summary,
                title_fontsize, title_fontweight, label_fontsize, label_fontweight,
                tick_fontsize, subplot_layout, custom_labels, title, xlabel, ylabel,
                show_grid, grid_alpha, xlim, ylim, figsize, show_individual_figures,
                individual_figsize, save_individual_figures, show_combined_figure,
                save_combined_figure, save_fig, filename, dpi, bbox_inches, transparent_bg,
                # Volume normalization parameters
                normalize_by_volume, density_units, volume_calculation_method
            )
        
        # Determine targets to plot
        if is_batch:
            if target_sel is None:
                targets = list(binding_results.keys())
            elif isinstance(target_sel, str):
                targets = [target_sel]
            else:
                targets = target_sel
        else:
            targets = [None]  # Single target, no key needed
        
        n_targets = len(targets)
        
        # Determine subplot layout
        if n_targets == 1:
            nrows, ncols = 1, 1
            if figsize is None:
                figsize = (8, 8) if show_marginals else (8, 6)
        else:
            if subplot_layout is not None:
                nrows, ncols = subplot_layout
            else:
                # Auto-determine layout
                ncols = min(3, n_targets)
                nrows = (n_targets + ncols - 1) // ncols
            
            if figsize is None:
                width = 6 * ncols if show_marginals else 5 * ncols
                height = 6 * nrows if show_marginals else 5 * nrows
                figsize = (width, height)
        
        # Storage for statistics to print later
        all_stats = []
        
        # Determine figure creation mode
        # For single target: always create one figure
        # For multi-target: can create individual figures, combined figure, or both
        create_combined = (n_targets == 1) or show_combined_figure
        create_individual = (n_targets > 1) and (show_individual_figures or save_individual_figures)
        
        # Storage for individual figures
        individual_figs = []
        
        # Set individual figure size
        if individual_figsize is None:
            individual_figsize = (8, 8) if show_marginals else (8, 6)
        
        # Create combined figure if requested
        combined_fig = None
        combined_axes = None
        
        if create_combined:
            # Create figure
            if show_marginals and n_targets == 1:
                # Use gridspec for marginal plots with title and colorbar space
                # Layout: [title row] [top_hist row] [main + right_hist + colorbar row]
                combined_fig = plt.figure(figsize=figsize)
                gs = combined_fig.add_gridspec(3, 3, 
                                 width_ratios=[4, 0.8, 0.3],  # main, right_hist, colorbar
                                 height_ratios=[0.3, 0.8, 4],  # title, top_hist, main
                                 hspace=0.02, wspace=0.02,
                                 left=0.1, right=0.95, top=0.95, bottom=0.1)
                
                # Main plot
                ax_main = combined_fig.add_subplot(gs[2, 0])
                # Top histogram (aligned with main plot)
                ax_top = combined_fig.add_subplot(gs[1, 0], sharex=ax_main)
                # Right histogram (aligned with main plot)
                ax_right = combined_fig.add_subplot(gs[2, 1], sharey=ax_main)
                # Title space (above top histogram)
                ax_title = combined_fig.add_subplot(gs[0, 0])
                ax_title.axis('off')
                # Colorbar space (right of right histogram)
                ax_cbar = combined_fig.add_subplot(gs[2, 2])
                
                combined_axes = [(ax_main, ax_top, ax_right, ax_title, ax_cbar)]
            else:
                combined_fig, axes_grid = plt.subplots(nrows, ncols, figsize=figsize)
                if n_targets == 1:
                    combined_axes = [axes_grid]
                else:
                    combined_axes = axes_grid.flatten() if n_targets > 1 else [axes_grid]
        
        # Plot each target
        for idx, target_key in enumerate(targets):
            # Prepare axes for combined figure
            if create_combined:
                if show_marginals and n_targets == 1:
                    ax_main, ax_top, ax_right, ax_title, ax_cbar = combined_axes[idx]
                else:
                    ax_main = combined_axes[idx]
                    ax_top, ax_right, ax_title, ax_cbar = None, None, None, None
            else:
                ax_main, ax_top, ax_right, ax_title, ax_cbar = None, None, None, None, None
            
            # Prepare axes for individual figure if requested
            if create_individual:
                ind_fig = plt.figure(figsize=individual_figsize)
                if show_marginals:
                    gs_ind = ind_fig.add_gridspec(3, 3,
                                                  width_ratios=[4, 0.8, 0.3],
                                                  height_ratios=[0.3, 0.8, 4],
                                                  hspace=0.02, wspace=0.02,
                                                  left=0.1, right=0.95, top=0.95, bottom=0.1)
                    ax_ind_main = ind_fig.add_subplot(gs_ind[2, 0])
                    ax_ind_top = ind_fig.add_subplot(gs_ind[1, 0], sharex=ax_ind_main)
                    ax_ind_right = ind_fig.add_subplot(gs_ind[2, 1], sharey=ax_ind_main)
                    ax_ind_title = ind_fig.add_subplot(gs_ind[0, 0])
                    ax_ind_title.axis('off')
                    ax_ind_cbar = ind_fig.add_subplot(gs_ind[2, 2])
                else:
                    ax_ind_main = ind_fig.add_subplot(111)
                    ax_ind_top, ax_ind_right, ax_ind_title, ax_ind_cbar = None, None, None, None
                individual_figs.append((ind_fig, target_key))
            else:
                ax_ind_main, ax_ind_top, ax_ind_right, ax_ind_title, ax_ind_cbar = None, None, None, None, None
            
            # Get data for this target
            if is_batch:
                if target_key not in binding_results:
                    continue
                result = binding_results[target_key]
            else:
                result = binding_results
            
            # Extract per-frame binding data
            ion1, ion2 = ion_pair
            
            if 'cation_binding' in result:
                binding_dict = result['cation_binding']
            elif 'anion_binding' in result:
                binding_dict = result['anion_binding']
            else:
                print(f"Warning: No binding data for {target_key}")
                continue
            
            if ion1 not in binding_dict or ion2 not in binding_dict:
                print(f"Warning: Ion pair {ion_pair} not found in {target_key}")
                continue
            
            # Get per-frame binding values
            if normalize_by_volume:
                # Use volume-normalized data
                if 'volume_density' in binding_dict[ion1]:
                    binding1 = binding_dict[ion1]['volume_density']
                else:
                    # Try to extract using helper method
                    binding1 = self._extract_volume_normalized_timeseries(binding_dict[ion1], volume_calculation_method)
                    if binding1 is None:
                        print(f"Warning: No volume data available for {ion1}, using raw binding events")
                        binding1 = binding_dict[ion1]['binding_events']
                
                if 'volume_density' in binding_dict[ion2]:
                    binding2 = binding_dict[ion2]['volume_density']
                else:
                    # Try to extract using helper method
                    binding2 = self._extract_volume_normalized_timeseries(binding_dict[ion2], volume_calculation_method)
                    if binding2 is None:
                        print(f"Warning: No volume data available for {ion2}, using raw binding events")
                        binding2 = binding_dict[ion2]['binding_events']
            else:
                # Use raw binding events
                binding1 = binding_dict[ion1]['binding_events']  # Array of per-frame values
                binding2 = binding_dict[ion2]['binding_events']
            
            # Extract KDE data for color_by='count' option
            kde_data1, kde_data2 = None, None
            if color_by == 'count' and normalize_by_volume:
                # Use raw binding events for KDE even with volume normalization
                kde_data1 = binding_dict[ion1]['binding_events']
                kde_data2 = binding_dict[ion2]['binding_events']
            
            # Determine target label for stats report and title
            if is_batch:
                if custom_labels and target_key in custom_labels:
                    stats_target_label = custom_labels[target_key]
                else:
                    stats_target_label = target_key
                plot_title = stats_target_label  # Use target name as title in batch mode
            else:
                stats_target_label = title if title else 'Target'
                plot_title = title  # Use provided title in single mode
            
            # Plot on combined figure axes if created
            if create_combined:
                stats = self._plot_single_ion_correlation(
                    ax_main, binding1, binding2, ion1, ion2,
                    ax_top=ax_top, ax_right=ax_right, ax_title=ax_title, ax_cbar=ax_cbar,
                    kde_data1=kde_data1, kde_data2=kde_data2,
                    title=plot_title,
                plot_type=plot_type, color_by=color_by, colormap=colormap,
                point_size=point_size, point_alpha=point_alpha,
                bubble_scale=bubble_scale, show_bubble_labels=show_bubble_labels,
                show_center_markers=show_center_markers, marker_size=marker_size,
                marker_color=marker_color, marker_edgecolor=marker_edgecolor,
                marker_edgewidth=marker_edgewidth,
                hexbin_gridsize=hexbin_gridsize,
                contour_levels=contour_levels, contour_colors=contour_colors,
                contour_percentile_range=contour_percentile_range,
                contour_spacing=contour_spacing,
                kde_log_transform=kde_log_transform,
                colorbar_log_scale=colorbar_log_scale,
                normalize_colorbar=normalize_colorbar,
                extend=extend,
                show_data_points=show_data_points, data_point_marker=data_point_marker,
                data_point_size=data_point_size, data_point_color=data_point_color,
                data_point_alpha=data_point_alpha,
                show_stats=show_stats, stats_location=stats_location,
                stats_fontsize=stats_fontsize,
                show_reference_lines=show_reference_lines,
                show_grid=show_grid, grid_alpha=grid_alpha,
                xlim=xlim, ylim=ylim,
                    label_fontsize=label_fontsize, label_fontweight=label_fontweight,
                    tick_fontsize=tick_fontsize
                )
                
                # Store statistics for printing
                if stats:
                    stats['target'] = stats_target_label
                    if idx == 0 or not all_stats:  # Only store once per target
                        all_stats.append(stats)
                
                # Set labels for combined figure
                if xlabel is None:
                    if normalize_by_volume:
                        # Determine units
                        if density_units == 'per_nm3':
                            units = 'ions/frame/nm³'
                        else:  # 'auto' or 'per_A3'
                            units = 'ions/frame/Å³'
                        xlabel_text = f'{ion1}$^+$ Density ({units})'
                    else:
                        xlabel_text = f'{ion1}$^+$ Binding Events'
                    ax_main.set_xlabel(xlabel_text, 
                                      fontsize=label_fontsize, fontweight=label_fontweight)
                else:
                    ax_main.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
                
                if ylabel is None:
                    if normalize_by_volume:
                        # Determine units
                        if density_units == 'per_nm3':
                            units = 'ions/frame/nm³'
                        else:  # 'auto' or 'per_A3'
                            units = 'ions/frame/Å³'
                        ylabel_text = f'{ion2}$^+$ Density ({units})'
                    else:
                        ylabel_text = f'{ion2}$^+$ Binding Events'
                    ax_main.set_ylabel(ylabel_text,
                                      fontsize=label_fontsize, fontweight=label_fontweight)
                else:
                    ax_main.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                
                # Set title (only on ax_main if no dedicated title axis)
                if ax_title is None:
                    if n_targets > 1:
                        # Multi-target: use target name as title
                        if custom_labels and target_key in custom_labels:
                            target_title = custom_labels[target_key]
                        else:
                            target_title = target_key
                        ax_main.set_title(target_title, fontsize=title_fontsize, fontweight=title_fontweight)
                    elif title is not None:
                        ax_main.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
            
            # Plot on individual figure axes if created
            if create_individual:
                self._plot_single_ion_correlation(
                    ax_ind_main, binding1, binding2, ion1, ion2,
                    ax_top=ax_ind_top, ax_right=ax_ind_right, ax_title=ax_ind_title, ax_cbar=ax_ind_cbar,
                    kde_data1=kde_data1, kde_data2=kde_data2,
                    title=plot_title,
                    plot_type=plot_type, color_by=color_by, colormap=colormap,
                    point_size=point_size, point_alpha=point_alpha,
                    bubble_scale=bubble_scale, show_bubble_labels=show_bubble_labels,
                    show_center_markers=show_center_markers, marker_size=marker_size,
                    marker_color=marker_color, marker_edgecolor=marker_edgecolor,
                    marker_edgewidth=marker_edgewidth,
                    hexbin_gridsize=hexbin_gridsize,
                    contour_levels=contour_levels, contour_colors=contour_colors,
                    contour_percentile_range=contour_percentile_range,
                    contour_spacing=contour_spacing,
                    kde_log_transform=kde_log_transform,
                    colorbar_log_scale=colorbar_log_scale,
                    normalize_colorbar=normalize_colorbar,
                    extend=extend,
                    show_data_points=show_data_points, data_point_marker=data_point_marker,
                    data_point_size=data_point_size, data_point_color=data_point_color,
                    data_point_alpha=data_point_alpha,
                    show_stats=show_stats, stats_location=stats_location,
                    stats_fontsize=stats_fontsize,
                    show_reference_lines=show_reference_lines,
                    show_grid=show_grid, grid_alpha=grid_alpha,
                    xlim=xlim, ylim=ylim,
                    label_fontsize=label_fontsize, label_fontweight=label_fontweight,
                    tick_fontsize=tick_fontsize
                )
                
                # Set labels for individual figure
                if xlabel is None:
                    if normalize_by_volume:
                        # Determine units
                        if density_units == 'per_nm3':
                            units = 'ions/frame/nm³'
                        else:  # 'auto' or 'per_A3'
                            units = 'ions/frame/Å³'
                        xlabel_text = f'{ion1}$^+$ Density ({units})'
                    else:
                        xlabel_text = f'{ion1}$^+$ Binding Events'
                    ax_ind_main.set_xlabel(xlabel_text, 
                                          fontsize=label_fontsize, fontweight=label_fontweight)
                else:
                    ax_ind_main.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
                
                if ylabel is None:
                    if normalize_by_volume:
                        # Determine units
                        if density_units == 'per_nm3':
                            units = 'ions/frame/nm³'
                        else:  # 'auto' or 'per_A3'
                            units = 'ions/frame/Å³'
                        ylabel_text = f'{ion2}$^+$ Density ({units})'
                    else:
                        ylabel_text = f'{ion2}$^+$ Binding Events'
                    ax_ind_main.set_ylabel(ylabel_text,
                                          fontsize=label_fontsize, fontweight=label_fontweight)
                else:
                    ax_ind_main.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                
                # Set title for individual figure
                if ax_ind_title is None:
                    ax_ind_main.set_title(plot_title, fontsize=title_fontsize, fontweight=title_fontweight)
                
                ind_fig.tight_layout()
                
                # Save individual figure if requested
                if save_individual_figures:
                    base_name = filename.rsplit('.', 1)[0]
                    ext = filename.rsplit('.', 1)[1] if '.' in filename else 'png'
                    ind_filename = f"{base_name}_{target_key}.{ext}"
                    ind_fig.savefig(ind_filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent_bg)
                    print(f"✓ Individual figure saved: {ind_filename}")
        
        # Handle combined figure display and saving
        if create_combined and combined_fig is not None:
            # Hide unused subplots
            if n_targets > 1 and combined_axes is not None:
                for idx in range(n_targets, len(combined_axes)):
                    combined_axes[idx].set_visible(False)
            
            combined_fig.tight_layout()
            
            # Save combined figure if requested
            # Logic: (save_fig AND save_combined_figure) OR (for single target, just save_fig)
            should_save_combined = (n_targets == 1 and save_fig) or (save_fig and save_combined_figure)
            if should_save_combined:
                combined_fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent_bg)
                print(f"✓ Combined figure saved: {filename}")
            
            # Show combined figure if requested
            if not show_combined_figure:
                plt.close(combined_fig)
        
        # Handle individual figures display
        if not show_individual_figures and individual_figs:
            for ind_fig, _ in individual_figs:
                plt.close(ind_fig)
        
        # Print complete summary report if requested (includes statistics + binding summary)
        if show_summary:
            import pandas as pd
            
            print("\n" + "="*80)
            print("ION COMPETITION ANALYSIS SUMMARY")
            print("="*80)
            
            # Print correlation statistics first
            if all_stats:
                print("\nCORRELATION STATISTICS:")
                print("-" * 80)
                for stat in all_stats:
                    print(f"\n{stat['target']}:")
                    print(f"  Pearson correlation:  r = {stat['pearson_r']:7.3f}  (p = {stat['pearson_p']:.2e})")
                    print(f"  Spearman correlation: ρ = {stat['spearman_r']:7.3f}  (p = {stat['spearman_p']:.2e})")
                    print(f"\n  Binding State Occupancy:")
                    print(f"    Both {stat['ion1']}⁺ and {stat['ion2']}⁺:  {stat['both_bound']:5d} frames  ({stat['both_pct']:5.1f}%)")
                    print(f"    Only {stat['ion1']}⁺:              {stat['only_ion1']:5d} frames  ({stat['only_ion1_pct']:5.1f}%)")
                    print(f"    Only {stat['ion2']}⁺:              {stat['only_ion2']:5d} frames  ({stat['only_ion2_pct']:5.1f}%)")
                    print(f"    Neither:                {stat['neither']:5d} frames  ({stat['neither_pct']:5.1f}%)")
            
            # Determine which targets to summarize
            if is_batch:
                summary_targets = targets
            else:
                summary_targets = [None]
            
            # Create summary table
            summary_data = []
            for target_key in summary_targets:
                # Get data for this target
                if is_batch:
                    target_data = binding_results[target_key]
                    target_label = custom_labels.get(target_key, target_key) if custom_labels else target_key
                else:
                    target_data = binding_results
                    target_label = 'Single Target'
                
                # Add cation binding data
                if 'cation_binding' in target_data:
                    for ion_type, data in target_data['cation_binding'].items():
                        summary_data.append({
                            'Target': target_label,
                            'Ion': ion_type,
                            'Type': 'Cation',
                            'Avg Binding': f"{data['average_binding']:.2f}",
                            'Max Binding': data['max_binding'],
                            'Occupancy': f"{data['occupancy']*100:.1f}%",
                            'Std Dev': f"{data['std_binding']:.2f}"
                        })
                
                # Add anion binding data
                if 'anion_binding' in target_data:
                    for ion_type, data in target_data['anion_binding'].items():
                        summary_data.append({
                            'Target': target_label,
                            'Ion': ion_type,
                            'Type': 'Anion',
                            'Avg Binding': f"{data['average_binding']:.2f}",
                            'Max Binding': data['max_binding'],
                            'Occupancy': f"{data['occupancy']*100:.1f}%",
                            'Std Dev': f"{data['std_binding']:.2f}"
                        })
            
            if summary_data:
                print("\n" + "-" * 80)
                print("BINDING SUMMARY:")
                print("-" * 80)
                df = pd.DataFrame(summary_data)
                print(df.to_string(index=False))
                
                # Print selectivity for each target
                print("\n" + "-" * 80)
                print("SELECTIVITY:")
                print("-" * 80)
                for target_key in summary_targets:
                    if is_batch:
                        target_data = binding_results[target_key]
                        target_label = custom_labels.get(target_key, target_key) if custom_labels else target_key
                    else:
                        target_data = binding_results
                        target_label = 'Single Target'
                    
                    if target_data.get('selectivity'):
                        print(f"\n{target_label}:")
                        for key, metrics in target_data['selectivity'].items():
                            if isinstance(metrics, dict):
                                print(f"  {key}:")
                                print(f"    Fraction: {metrics['fraction']:.3f}")
                                print(f"    Ratio:    {metrics['ratio']:.3f}")
                            else:
                                print(f"  {key}: {metrics:.3f}")
            
            print("="*80 + "\n")
        
        # Return appropriate figures
        if create_combined and not create_individual:
            return combined_fig, combined_axes
        elif create_individual and not create_combined:
            return individual_figs
        else:
            return combined_fig, combined_axes, individual_figs
    
    def _plot_single_ion_correlation(self, ax, binding1, binding2, ion1, ion2,
                                     ax_top=None, ax_right=None, ax_title=None, ax_cbar=None, 
                                     kde_data1=None, kde_data2=None, **kwargs):
        """Helper method to plot single ion correlation"""
        
        import scipy.stats as stats
        
        plot_type = kwargs.get('plot_type', 'combined')
        color_by = kwargs.get('color_by', 'density')
        colormap = kwargs.get('colormap', 'viridis')
        
        # Calculate statistics
        pearson_r, pearson_p = stats.pearsonr(binding1, binding2)
        spearman_r, spearman_p = stats.spearmanr(binding1, binding2)
        
        # Calculate mutual exclusivity metrics
        both_bound = np.sum((binding1 > 0) & (binding2 > 0))
        only_ion1 = np.sum((binding1 > 0) & (binding2 == 0))
        only_ion2 = np.sum((binding1 == 0) & (binding2 > 0))
        neither = np.sum((binding1 == 0) & (binding2 == 0))
        total = len(binding1)
        
        # Create plot based on type
        if plot_type == 'scatter':
            if color_by == 'density':
                # Calculate 2D density for coloring
                from scipy.stats import gaussian_kde
                try:
                    xy = np.vstack([binding1, binding2])
                    z = gaussian_kde(xy)(xy)
                    scatter = ax.scatter(binding1, binding2, c=z, s=kwargs.get('point_size', 20),
                                       alpha=kwargs.get('point_alpha', 0.3), cmap=colormap,
                                       edgecolors='none')
                    plt.colorbar(scatter, ax=ax, label='Density')
                except:
                    scatter = ax.scatter(binding1, binding2, s=kwargs.get('point_size', 20),
                                       alpha=kwargs.get('point_alpha', 0.3), c='blue',
                                       edgecolors='none')
            elif color_by == 'time':
                frames = np.arange(len(binding1))
                scatter = ax.scatter(binding1, binding2, c=frames, s=kwargs.get('point_size', 20),
                                   alpha=kwargs.get('point_alpha', 0.3), cmap=colormap,
                                   edgecolors='none')
                plt.colorbar(scatter, ax=ax, label='Frame Number')
            elif color_by == 'total':
                total_binding = binding1 + binding2
                scatter = ax.scatter(binding1, binding2, c=total_binding, s=kwargs.get('point_size', 20),
                                   alpha=kwargs.get('point_alpha', 0.3), cmap=colormap,
                                   edgecolors='none')
                plt.colorbar(scatter, ax=ax, label='Total Binding')
        
        elif plot_type == 'bubble':
            # Bubble plot: each discrete state gets a circle sized by frequency
            from collections import Counter
            
            # Count occurrences of each (binding1, binding2) state
            states = list(zip(binding1, binding2))
            state_counts = Counter(states)
            
            # Extract unique states and their frequencies
            unique_states = np.array(list(state_counts.keys()))
            frequencies = np.array(list(state_counts.values()))
            
            x_states = unique_states[:, 0]
            y_states = unique_states[:, 1]
            
            # Calculate bubble sizes (proportional to frequency)
            bubble_scale = kwargs.get('bubble_scale', 10.0)
            sizes = frequencies * bubble_scale
            
            # Color by frequency or other metrics
            if color_by == 'frequency' or color_by == 'density':
                colors_data = frequencies
                color_label = 'Frequency (# frames)'
            elif color_by == 'total':
                colors_data = x_states + y_states
                color_label = 'Total Binding'
            else:
                colors_data = frequencies
                color_label = 'Frequency (# frames)'
            
            scatter = ax.scatter(x_states, y_states, s=sizes, c=colors_data,
                               cmap=colormap, alpha=kwargs.get('point_alpha', 0.6),
                               edgecolors='black', linewidths=0.5)
            
            # Place colorbar in dedicated axis if available, otherwise on main plot
            if ax_cbar is not None:
                cbar = plt.colorbar(scatter, cax=ax_cbar, label=color_label)
            else:
                cbar = plt.colorbar(scatter, ax=ax, label=color_label)
            
            # Add center markers to show exact coordinates
            if kwargs.get('show_center_markers', True):
                ax.scatter(x_states, y_states, 
                          s=kwargs.get('marker_size', 20),
                          c=kwargs.get('marker_color', 'black'),
                          edgecolors=kwargs.get('marker_edgecolor', 'white'),
                          linewidths=kwargs.get('marker_edgewidth', 0.5),
                          alpha=1.0, zorder=10)
            
            # Add text labels for frequencies on larger bubbles
            if kwargs.get('show_bubble_labels', True):
                freq_threshold = np.percentile(frequencies, 75)  # Label top 25%
                for i, (x, y, freq) in enumerate(zip(x_states, y_states, frequencies)):
                    if freq >= freq_threshold:
                        ax.text(x, y, str(int(freq)), ha='center', va='center',
                               fontsize=8, fontweight='bold', color='white')
        
        elif plot_type == 'hexbin':
            hexbin = ax.hexbin(binding1, binding2, gridsize=kwargs.get('hexbin_gridsize', 30),
                             cmap=colormap, mincnt=1, edgecolors='none')
            # Place colorbar in dedicated axis if available
            if ax_cbar is not None:
                plt.colorbar(hexbin, cax=ax_cbar, label='Count')
            else:
                plt.colorbar(hexbin, ax=ax, label='Count')
        
        elif plot_type == 'contour':
            from scipy.stats import gaussian_kde
            try:
                # For color_by='count' with volume normalization, use raw count data for KDE
                if color_by == 'count' and kde_data1 is not None and kde_data2 is not None:
                    kde_input = np.vstack([kde_data1, kde_data2])
                    colorbar_label = 'Count Density'
                else:
                    # Use same data for both axes and KDE
                    kde_input = np.vstack([binding1, binding2])
                    colorbar_label = 'Frequency Density'
                
                # Create grid for contour - extend slightly beyond data for edge coverage
                xmin, xmax = binding1.min(), binding1.max()
                ymin, ymax = binding2.min(), binding2.max()
                
                # Add small margin for KDE bandwidth, but keep tight
                x_range = xmax - xmin if xmax > xmin else 1
                y_range = ymax - ymin if ymax > ymin else 1
                margin_x = x_range * 0.05  # 5% margin
                margin_y = y_range * 0.05
                
                # Grid extends to edges with margin
                xx, yy = np.mgrid[xmin-margin_x:xmax+margin_x:100j, 
                                  ymin-margin_y:ymax+margin_y:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                kernel = gaussian_kde(kde_input)
                f = np.reshape(kernel(positions).T, xx.shape)
                
                # Option 2: Log-transform KDE values if requested
                kde_log_transform_param = kwargs.get('kde_log_transform', False)
                if kde_log_transform_param:
                    # Apply log transform: log10(f + small_offset) to avoid log(0)
                    min_nonzero = f[f > 0].min() if np.any(f > 0) else 1e-10
                    offset = min_nonzero / 100  # Small offset to avoid log(0)
                    f_for_levels = np.log10(f + offset)
                    colorbar_label = f"Log {colorbar_label}"
                else:
                    f_for_levels = f
                
                # Determine contour levels based on spacing method
                contour_levels_param = kwargs.get('contour_levels', 10)
                contour_percentile_range_param = kwargs.get('contour_percentile_range', (0, 100))
                contour_spacing_param = kwargs.get('contour_spacing', 'linear')
                
                if isinstance(contour_levels_param, int):
                    if contour_spacing_param == 'log':
                        # Option 1: Log-spaced contour levels
                        min_val = f_for_levels[f_for_levels > 0].min() if np.any(f_for_levels > 0) else 1e-10
                        max_val = f_for_levels.max()
                        if max_val > min_val and min_val > 0:
                            levels = np.logspace(np.log10(min_val), np.log10(max_val), contour_levels_param)
                        else:
                            # Fallback to linear if log spacing fails
                            levels = np.linspace(f_for_levels.min(), f_for_levels.max(), contour_levels_param)
                    elif contour_spacing_param == 'percentile' or contour_percentile_range_param != (0, 100):
                        # Use percentile-based levels
                        min_pct, max_pct = contour_percentile_range_param
                        percentiles = np.linspace(min_pct, max_pct, contour_levels_param)
                        levels = np.percentile(f_for_levels, percentiles)
                    else:
                        # Use linear levels (original behavior)
                        levels = np.linspace(f_for_levels.min(), f_for_levels.max(), contour_levels_param)
                    
                else:
                    # Use provided levels directly
                    levels = contour_levels_param
                
                # Determine appropriate extend parameter based on data clipping
                extend_param = kwargs.get('extend', 'auto')
                
                if extend_param == 'auto':
                    # Smart extend logic (original behavior)
                    plot_data = f if not kde_log_transform_param else f_for_levels
                    data_min_actual = plot_data.min()
                    data_max_actual = plot_data.max()
                    level_min = levels[0]
                    level_max = levels[-1]
                    
                    # Smart extend logic
                    if data_min_actual >= level_min and data_max_actual <= level_max:
                        extend_param = 'neither'  # No clipping, rectangular colorbar
                    elif data_min_actual >= level_min and data_max_actual > level_max:
                        extend_param = 'max'      # Only top data clipped, arrow at top
                    elif data_min_actual < level_min and data_max_actual <= level_max:
                        extend_param = 'min'      # Only bottom data clipped, arrow at bottom
                    else:
                        extend_param = 'both'     # Both ends clipped, arrows at both ends
                
                # Create contour plots (always use original f for visualization)
                contour = ax.contourf(xx, yy, f if not kde_log_transform_param else f_for_levels, levels=levels,
                                     cmap=colormap, alpha=0.7, extend=extend_param)
                ax.contour(xx, yy, f if not kde_log_transform_param else f_for_levels, levels=levels,
                          colors=kwargs.get('contour_colors', 'white'), linewidths=1)
                
                # Handle colorbar normalization
                normalize_colorbar_param = kwargs.get('normalize_colorbar', False)
                if normalize_colorbar_param:
                    # Normalize colorbar values to 0-1 range for consistent comparison
                    data_for_colorbar = f if not kde_log_transform_param else f_for_levels
                    data_min, data_max = data_for_colorbar.min(), data_for_colorbar.max()
                    if data_max > data_min:
                        # Create normalized data for colorbar
                        data_normalized = (data_for_colorbar - data_min) / (data_max - data_min)
                        # Create new contour with normalized data for colorbar only
                        norm_extend = 'neither' if extend_param == 'auto' else extend_param  # Normalized is 0-1, usually no clipping
                        contour_normalized = ax.contourf(xx, yy, data_normalized, 
                                                        levels=np.linspace(0, 1, len(levels)),
                                                        cmap=colormap, alpha=0.7, extend=norm_extend)
                        colorbar_data = contour_normalized
                        colorbar_label = "Relative Frequency Density"
                    else:
                        # Fallback if no range
                        colorbar_data = contour
                else:
                    colorbar_data = contour
                
                # Overlay actual data points if requested
                if kwargs.get('show_data_points', False):
                    ax.scatter(binding1, binding2, 
                              marker=kwargs.get('data_point_marker', '+'),
                              s=kwargs.get('data_point_size', 30),
                              c=kwargs.get('data_point_color', 'black'),
                              alpha=kwargs.get('data_point_alpha', 0.5),
                              linewidths=1.5, edgecolors='white',
                              zorder=100)
                
                # Set tight axis limits to data range (no extra padding)
                if not kwargs.get('xlim'):
                    ax.set_xlim(xmin-margin_x, xmax+margin_x)
                if not kwargs.get('ylim'):
                    ax.set_ylim(ymin-margin_y, ymax+margin_y)
                
                # Place colorbar in dedicated axis if available
                # Option 3: Log colorbar scale if requested (but not with normalized colorbar)
                colorbar_log_scale_param = kwargs.get('colorbar_log_scale', False)
                normalize_colorbar_param = kwargs.get('normalize_colorbar', False)
                
                # Smart colorbar formatting based on data range and normalization
                def get_colorbar_format(data_range_max, is_normalized=False):
                    if is_normalized:
                        return '%.1f'  # 0.1, 0.5, 1.0 for normalized data
                    elif data_range_max >= 10000:
                        return lambda x, pos: f'{int(round(x, -2))}'  # Round to hundreds: 18200
                    elif data_range_max >= 1000:
                        return lambda x, pos: f'{int(round(x, -1))}'  # Round to tens: 1820  
                    elif data_range_max >= 100:
                        return lambda x, pos: f'{int(round(x))}'      # Round to ones: 182
                    elif data_range_max >= 10:
                        return '%.0f'  # Whole numbers: 18
                    else:
                        return '%.1f'  # Small values: 1.8
                
                # Determine data range for formatting
                original_data = f if not kde_log_transform_param else f_for_levels
                data_max_for_format = original_data.max() if not normalize_colorbar_param else 1.0
                colorbar_format = get_colorbar_format(data_max_for_format, normalize_colorbar_param)
                
                if colorbar_log_scale_param and not kde_log_transform_param and not normalize_colorbar_param:
                    # Use LogNorm for colorbar scaling (only if data wasn't already log-transformed or normalized)
                    from matplotlib.colors import LogNorm
                    # Find positive values for LogNorm
                    original_data = f if not kde_log_transform_param else f_for_levels
                    if original_data.max() > 0 and np.any(original_data > 0):
                        vmin = original_data[original_data > 0].min()
                        vmax = original_data.max()
                        # Use scientific notation for log scale regardless of range
                        log_format = '%.1e' if not normalize_colorbar_param else '%.1f'
                        if ax_cbar is not None:
                            cbar = plt.colorbar(colorbar_data, cax=ax_cbar, label=f"Log {colorbar_label}", 
                                              norm=LogNorm(vmin=vmin, vmax=vmax), format=log_format)
                        else:
                            cbar = plt.colorbar(colorbar_data, ax=ax, label=f"Log {colorbar_label}",
                                              norm=LogNorm(vmin=vmin, vmax=vmax), format=log_format)
                    else:
                        # Fallback to linear if no positive values
                        if ax_cbar is not None:
                            plt.colorbar(colorbar_data, cax=ax_cbar, label=colorbar_label, format=colorbar_format)
                        else:
                            plt.colorbar(colorbar_data, ax=ax, label=colorbar_label, format=colorbar_format)
                else:
                    # Standard linear colorbar
                    if ax_cbar is not None:
                        cbar = plt.colorbar(colorbar_data, cax=ax_cbar, label=colorbar_label, format=colorbar_format)
                    else:
                        cbar = plt.colorbar(colorbar_data, ax=ax, label=colorbar_label, format=colorbar_format)
                    pass  # Standard linear colorbar creation complete
            except:
                # Fallback to scatter if KDE fails
                ax.scatter(binding1, binding2, s=kwargs.get('point_size', 20),
                         alpha=kwargs.get('point_alpha', 0.3), c='blue', edgecolors='none')
        
        elif plot_type == 'combined':
            # Hexbin + contours
            hexbin = ax.hexbin(binding1, binding2, gridsize=kwargs.get('hexbin_gridsize', 30),
                             cmap=colormap, mincnt=1, edgecolors='none', alpha=0.8)
            
            # Add contours on top
            from scipy.stats import gaussian_kde
            try:
                xmin, xmax = binding1.min(), binding1.max()
                ymin, ymax = binding2.min(), binding2.max()
                
                # Add small margin for KDE bandwidth
                x_range = xmax - xmin if xmax > xmin else 1
                y_range = ymax - ymin if ymax > ymin else 1
                margin_x = x_range * 0.05
                margin_y = y_range * 0.05
                
                xx, yy = np.mgrid[xmin-margin_x:xmax+margin_x:100j, 
                                  ymin-margin_y:ymax+margin_y:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                kernel = gaussian_kde(np.vstack([binding1, binding2]))
                f = np.reshape(kernel(positions).T, xx.shape)
                
                ax.contour(xx, yy, f, levels=kwargs.get('contour_levels', 5),
                          colors=kwargs.get('contour_colors', 'white'), linewidths=1.5, alpha=0.8)
                
                # Set tight axis limits
                if not kwargs.get('xlim'):
                    ax.set_xlim(xmin-margin_x, xmax+margin_x)
                if not kwargs.get('ylim'):
                    ax.set_ylim(ymin-margin_y, ymax+margin_y)
            except:
                pass
            
            # Overlay actual data points if requested
            if kwargs.get('show_data_points', False):
                ax.scatter(binding1, binding2, 
                          marker=kwargs.get('data_point_marker', '+'),
                          s=kwargs.get('data_point_size', 30),
                          c=kwargs.get('data_point_color', 'black'),
                          alpha=kwargs.get('data_point_alpha', 0.5),
                          linewidths=1.5, edgecolors='white',
                          zorder=100)
            
            # Place colorbar in dedicated axis if available
            if ax_cbar is not None:
                plt.colorbar(hexbin, cax=ax_cbar, label='Count')
            else:
                plt.colorbar(hexbin, ax=ax, label='Count')
        
        # Reference lines
        if kwargs.get('show_reference_lines', True):
            # Diagonal line (x=y)
            max_val = max(binding1.max(), binding2.max())
            ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1, alpha=0.5, label='Equal binding')
            
            # Mean lines
            ax.axvline(binding1.mean(), color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(binding2.mean(), color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        # Statistics text box (only if requested on figure)
        stats_location = kwargs.get('stats_location', 'print')
        if kwargs.get('show_stats', True) and stats_location in ['figure', 'both']:
            stats_text = (
                f"Pearson r = {pearson_r:.3f} (p={pearson_p:.1e})\n"
                f"Spearman ρ = {spearman_r:.3f} (p={spearman_p:.1e})\n"
                f"\n"
                f"Both: {both_bound} ({both_bound/total*100:.1f}%)\n"
                f"Only {ion1}: {only_ion1} ({only_ion1/total*100:.1f}%)\n"
                f"Only {ion2}: {only_ion2} ({only_ion2/total*100:.1f}%)\n"
                f"Neither: {neither} ({neither/total*100:.1f}%)"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=kwargs.get('stats_fontsize', 10), verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Prepare statistics dict for return (for printing)
        stats_dict = None
        if kwargs.get('show_stats', True):
            stats_dict = {
                'ion1': ion1,
                'ion2': ion2,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'both_bound': both_bound,
                'both_pct': both_bound/total*100,
                'only_ion1': only_ion1,
                'only_ion1_pct': only_ion1/total*100,
                'only_ion2': only_ion2,
                'only_ion2_pct': only_ion2/total*100,
                'neither': neither,
                'neither_pct': neither/total*100,
                'total': total
            }
        
        # Grid
        if kwargs.get('show_grid', True):
            ax.grid(True, alpha=kwargs.get('grid_alpha', 0.3))
        
        # Axis limits
        if kwargs.get('xlim'):
            ax.set_xlim(kwargs.get('xlim'))
        if kwargs.get('ylim'):
            ax.set_ylim(kwargs.get('ylim'))
        
        # Tick formatting
        ax.tick_params(labelsize=kwargs.get('tick_fontsize', 10))
        
        # Marginal distributions
        if ax_top is not None and ax_right is not None:
            # Determine appropriate binning based on data type
            # Check if data appears to be integer counts or continuous densities
            is_integer_data = (np.all(binding1 % 1 == 0) and np.all(binding2 % 1 == 0) and
                             binding1.max() <= 20 and binding2.max() <= 20)
            
            if is_integer_data:
                # Integer binning for binding event counts (0, 1, 2, 3, ...)
                # Bins edges at -0.5, 0.5, 1.5, 2.5, ... so bars center on integers
                max_binding1 = int(np.ceil(binding1.max()))
                max_binding2 = int(np.ceil(binding2.max()))
                bins1 = np.arange(-0.5, max_binding1 + 1.5, 1.0)
                bins2 = np.arange(-0.5, max_binding2 + 1.5, 1.0)
                edge_color = 'black'
            else:
                # Continuous binning for density data
                # Use fewer bins and extend range slightly for better visualization
                n_bins = min(15, max(8, int(np.sqrt(len(binding1)))))
                
                # Add small margins to data range for better bin coverage
                range1 = binding1.max() - binding1.min()
                range2 = binding2.max() - binding2.min()
                margin1 = range1 * 0.02  # 2% margin
                margin2 = range2 * 0.02
                
                bins1 = np.linspace(binding1.min() - margin1, binding1.max() + margin1, n_bins + 1)
                bins2 = np.linspace(binding2.min() - margin2, binding2.max() + margin2, n_bins + 1)
                edge_color = None  # No edges for continuous data to ensure bars touch
            
            # Top histogram (ion1 distribution)
            ax_top.hist(binding1, bins=bins1, color='steelblue', alpha=0.7, 
                       edgecolor=edge_color, linewidth=0.5 if edge_color else 0)
            ax_top.set_ylabel('Count', fontsize=kwargs.get('label_fontsize', 12) - 2)
            ax_top.tick_params(labelbottom=False, labelsize=kwargs.get('tick_fontsize', 10))
            ax_top.axvline(binding1.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            
            # Right histogram (ion2 distribution)
            ax_right.hist(binding2, bins=bins2, orientation='horizontal', 
                         color='steelblue', alpha=0.7, 
                         edgecolor=edge_color, linewidth=0.5 if edge_color else 0)
            ax_right.set_xlabel('Count', fontsize=kwargs.get('label_fontsize', 12) - 2)
            ax_right.tick_params(labelleft=False, labelsize=kwargs.get('tick_fontsize', 10))
            ax_right.axhline(binding2.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax_right.spines['top'].set_visible(False)
            ax_right.spines['right'].set_visible(False)
        
        # Place title in dedicated title axis if available
        if ax_title is not None and 'title' in kwargs and kwargs['title']:
            ax_title.text(0.5, 0.5, kwargs['title'], 
                         ha='center', va='center',
                         fontsize=kwargs.get('title_fontsize', 14),
                         fontweight=kwargs.get('title_fontweight', 'bold'),
                         transform=ax_title.transAxes)
        
        return stats_dict
    
    def _plot_ion_competition_correlation_with_peaks(self, binding_results, target_sel, ion_pair, 
                                                    peaks_to_show, peak_comparison_mode, plot_type,
                                                    show_marginals, color_by, colormap, point_size, point_alpha,
                                                    bubble_scale, show_bubble_labels, show_center_markers,
                                                    marker_size, marker_color, marker_edgecolor, marker_edgewidth,
                                                    hexbin_gridsize, contour_levels, contour_colors, show_data_points,
                                                    data_point_marker, data_point_size, data_point_color,
                                                    data_point_alpha, show_stats, stats_location, stats_fontsize,
                                                    show_reference_lines, show_summary, title_fontsize,
                                                    title_fontweight, label_fontsize, label_fontweight,
                                                    tick_fontsize, subplot_layout, custom_labels, title, xlabel,
                                                    ylabel, show_grid, grid_alpha, xlim, ylim, figsize,
                                                    show_individual_figures, individual_figsize, save_individual_figures,
                                                    show_combined_figure, save_combined_figure, save_fig, filename,
                                                    dpi, bbox_inches, transparent_bg,
                                                    # Volume normalization parameters
                                                    normalize_by_volume, density_units, volume_calculation_method):
        """
        Handle peak-specific ion competition correlation analysis
        """
        
        print(f"📊 Creating peak-specific ion competition correlation plots")
        print(f"   Ion pair: {ion_pair}, Peaks: {peaks_to_show}, Mode: {peak_comparison_mode}")
        
        # Check if batch results or single target
        is_batch = self._is_batch_binding_result(binding_results)
        
        # Determine targets to plot
        if is_batch:
            if target_sel is None:
                targets = list(binding_results.keys())
            elif isinstance(target_sel, str):
                targets = [target_sel]
            else:
                targets = target_sel
        else:
            targets = [None]  # Single target, no key needed
        
        n_targets = len(targets)
        n_peaks = len(peaks_to_show)
        
        # Determine subplot layout
        if peak_comparison_mode == 'same':
            # Grid: targets × peaks (each peak shows ion1_peak vs ion2_peak)
            if n_targets == 1 and n_peaks == 1:
                nrows, ncols = 1, 1
            elif n_targets == 1:
                # Single target, multiple peaks
                ncols = min(3, n_peaks)
                nrows = (n_peaks + ncols - 1) // ncols
            elif n_peaks == 1:
                # Multiple targets, single peak
                ncols = min(3, n_targets)
                nrows = (n_targets + ncols - 1) // ncols
            else:
                # Multiple targets AND peaks: target × peak grid
                if subplot_layout is not None:
                    nrows, ncols = subplot_layout
                else:
                    total_subplots = n_targets * n_peaks
                    ncols = min(4, total_subplots)  # Max 4 columns
                    nrows = (total_subplots + ncols - 1) // ncols
        else:
            # Other modes: implement as needed
            nrows, ncols = 1, 1
        
        # Figure size calculation
        if figsize is None:
            if show_marginals:
                base_size = 6
            else:
                base_size = 5
            figsize = (base_size * ncols, base_size * nrows)
        
        # Storage for statistics and figures
        all_stats = []
        individual_figs = []
        
        # Determine figure creation mode
        create_combined = show_combined_figure or (n_targets == 1 and n_peaks == 1)
        create_individual = (n_targets > 1 or n_peaks > 1) and (show_individual_figures or save_individual_figures)
        
        # Set individual figure size
        if individual_figsize is None:
            individual_figsize = (8, 8) if show_marginals else (8, 6)
        
        # Create combined figure if requested
        combined_fig = None
        combined_axes = None
        
        if create_combined:
            if show_marginals and nrows == 1 and ncols == 1:
                # Single subplot with marginals
                combined_fig = plt.figure(figsize=figsize)
                gs = combined_fig.add_gridspec(3, 3, 
                                 width_ratios=[4, 0.8, 0.3],
                                 height_ratios=[0.3, 0.8, 4],
                                 hspace=0.02, wspace=0.02,
                                 left=0.1, right=0.95, top=0.95, bottom=0.1)
                
                ax_main = combined_fig.add_subplot(gs[2, 0])
                ax_top = combined_fig.add_subplot(gs[1, 0], sharex=ax_main)
                ax_right = combined_fig.add_subplot(gs[2, 1], sharey=ax_main)
                ax_title = combined_fig.add_subplot(gs[0, 0])
                ax_title.axis('off')
                ax_cbar = combined_fig.add_subplot(gs[2, 2])
                
                combined_axes = [(ax_main, ax_top, ax_right, ax_title, ax_cbar)]
            else:
                # Multiple subplots - no marginals for grid layout
                combined_fig, axes_grid = plt.subplots(nrows, ncols, figsize=figsize)
                if nrows == 1 and ncols == 1:
                    combined_axes = [axes_grid]
                elif nrows == 1 or ncols == 1:
                    combined_axes = list(axes_grid.flatten()) if hasattr(axes_grid, 'flatten') else [axes_grid]
                else:
                    combined_axes = list(axes_grid.flatten())
        
        # Plot combinations based on comparison mode
        plot_idx = 0
        
        if peak_comparison_mode == 'same':
            # Compare same peaks between ions (Na_P1 vs K_P1, Na_P2 vs K_P2)
            for target_key in targets:
                for peak in peaks_to_show:
                    # Prepare axes for combined figure
                    if create_combined:
                        if show_marginals and nrows == 1 and ncols == 1:
                            ax_main, ax_top, ax_right, ax_title, ax_cbar = combined_axes[plot_idx]
                        else:
                            ax_main = combined_axes[plot_idx] if plot_idx < len(combined_axes) else None
                            ax_top, ax_right, ax_title, ax_cbar = None, None, None, None
                    else:
                        ax_main, ax_top, ax_right, ax_title, ax_cbar = None, None, None, None, None
                    
                    # Prepare axes for individual figure if requested
                    if create_individual:
                        ind_fig = plt.figure(figsize=individual_figsize)
                        if show_marginals:
                            gs_ind = ind_fig.add_gridspec(3, 3,
                                                          width_ratios=[4, 0.8, 0.3],
                                                          height_ratios=[0.3, 0.8, 4],
                                                          hspace=0.02, wspace=0.02,
                                                          left=0.1, right=0.95, top=0.95, bottom=0.1)
                            ax_ind_main = ind_fig.add_subplot(gs_ind[2, 0])
                            ax_ind_top = ind_fig.add_subplot(gs_ind[1, 0], sharex=ax_ind_main)
                            ax_ind_right = ind_fig.add_subplot(gs_ind[2, 1], sharey=ax_ind_main)
                            ax_ind_title = ind_fig.add_subplot(gs_ind[0, 0])
                            ax_ind_title.axis('off')
                            ax_ind_cbar = ind_fig.add_subplot(gs_ind[2, 2])
                        else:
                            ax_ind_main = ind_fig.add_subplot(111)
                            ax_ind_top, ax_ind_right, ax_ind_title, ax_ind_cbar = None, None, None, None
                        
                        combo_key = f"{target_key}_{peak}" if target_key else peak
                        individual_figs.append((ind_fig, combo_key))
                    else:
                        ax_ind_main, ax_ind_top, ax_ind_right, ax_ind_title, ax_ind_cbar = None, None, None, None, None
                    
                    # Extract peak-specific binding data
                    binding1, binding2, data_label = self._extract_peak_correlation_data(
                        binding_results, target_key, ion_pair, peak, peak, is_batch,
                        normalize_by_volume, volume_calculation_method
                    )
                    
                    # For color_by='count' with volume normalization, also get raw count data for KDE
                    kde_data1, kde_data2 = None, None
                    if color_by == 'count' and normalize_by_volume:
                        kde_data1, kde_data2, _ = self._extract_peak_correlation_data(
                            binding_results, target_key, ion_pair, peak, peak, is_batch,
                            normalize_by_volume=False, volume_calculation_method=volume_calculation_method
                        )
                    
                    if binding1 is None or binding2 is None:
                        print(f"⚠️  Skipping {target_key} + {peak}: data not available")
                        plot_idx += 1
                        continue
                    
                    # Generate subplot title
                    if n_targets > 1 and n_peaks > 1:
                        if custom_labels and target_key in custom_labels:
                            target_display = custom_labels[target_key]
                        else:
                            target_display = target_key if target_key else "Target"
                        subplot_title = f"{target_display} - {peak}"
                    elif n_peaks > 1:
                        subplot_title = peak
                    elif n_targets > 1:
                        if custom_labels and target_key in custom_labels:
                            subplot_title = custom_labels[target_key]
                        else:
                            subplot_title = target_key if target_key else "Target"
                    else:
                        subplot_title = title
                    
                    # Plot on combined figure axes if created
                    if create_combined and ax_main is not None:
                        stats = self._plot_single_peak_correlation(
                            ax_main, binding1, binding2, ion_pair[0], ion_pair[1], peak, peak,
                            kde_data1=kde_data1, kde_data2=kde_data2,
                            ax_top=ax_top, ax_right=ax_right, ax_title=ax_title, ax_cbar=ax_cbar,
                            title=subplot_title, plot_type=plot_type, color_by=color_by,
                            colormap=colormap, point_size=point_size, point_alpha=point_alpha,
                            bubble_scale=bubble_scale, show_bubble_labels=show_bubble_labels,
                            show_center_markers=show_center_markers, marker_size=marker_size,
                            marker_color=marker_color, marker_edgecolor=marker_edgecolor,
                            marker_edgewidth=marker_edgewidth, hexbin_gridsize=hexbin_gridsize,
                            contour_levels=contour_levels, contour_colors=contour_colors,
                            show_data_points=show_data_points, data_point_marker=data_point_marker,
                            data_point_size=data_point_size, data_point_color=data_point_color,
                            data_point_alpha=data_point_alpha, show_stats=show_stats,
                            stats_location=stats_location, stats_fontsize=stats_fontsize,
                            show_reference_lines=show_reference_lines, show_grid=show_grid,
                            grid_alpha=grid_alpha, xlim=xlim, ylim=ylim,
                            title_fontsize=title_fontsize, title_fontweight=title_fontweight,
                            label_fontsize=label_fontsize, label_fontweight=label_fontweight,
                            tick_fontsize=tick_fontsize
                        )
                        
                        if stats:
                            stats['target'] = target_key if target_key else "Single Target"
                            stats['peak'] = peak
                            all_stats.append(stats)
                        
                        # Set axis labels
                        if xlabel is None:
                            peak_label = peak if peak != 'total' else 'Total'
                            ax_main.set_xlabel(f'{ion_pair[0]}$^+$ {peak_label} Binding', 
                                              fontsize=label_fontsize, fontweight=label_fontweight)
                        else:
                            ax_main.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
                        
                        if ylabel is None:
                            peak_label = peak if peak != 'total' else 'Total'
                            ax_main.set_ylabel(f'{ion_pair[1]}$^+$ {peak_label} Binding',
                                              fontsize=label_fontsize, fontweight=label_fontweight)
                        else:
                            ax_main.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                        
                        # Set title if no dedicated title axis
                        if ax_title is None and subplot_title:
                            ax_main.set_title(subplot_title, fontsize=title_fontsize, fontweight=title_fontweight)
                    
                    # Plot on individual figure axes if created
                    if create_individual and ax_ind_main is not None:
                        self._plot_single_peak_correlation(
                            ax_ind_main, binding1, binding2, ion_pair[0], ion_pair[1], peak, peak,
                            kde_data1=kde_data1, kde_data2=kde_data2,
                            ax_top=ax_ind_top, ax_right=ax_ind_right, ax_title=ax_ind_title, ax_cbar=ax_ind_cbar,
                            title=subplot_title, plot_type=plot_type, color_by=color_by,
                            colormap=colormap, point_size=point_size, point_alpha=point_alpha,
                            bubble_scale=bubble_scale, show_bubble_labels=show_bubble_labels,
                            show_center_markers=show_center_markers, marker_size=marker_size,
                            marker_color=marker_color, marker_edgecolor=marker_edgecolor,
                            marker_edgewidth=marker_edgewidth, hexbin_gridsize=hexbin_gridsize,
                            contour_levels=contour_levels, contour_colors=contour_colors,
                            show_data_points=show_data_points, data_point_marker=data_point_marker,
                            data_point_size=data_point_size, data_point_color=data_point_color,
                            data_point_alpha=data_point_alpha, show_stats=show_stats,
                            stats_location=stats_location, stats_fontsize=stats_fontsize,
                            show_reference_lines=show_reference_lines, show_grid=show_grid,
                            grid_alpha=grid_alpha, xlim=xlim, ylim=ylim,
                            title_fontsize=title_fontsize, title_fontweight=title_fontweight,
                            label_fontsize=label_fontsize, label_fontweight=label_fontweight,
                            tick_fontsize=tick_fontsize
                        )
                        
                        # Set axis labels for individual figure
                        if xlabel is None:
                            peak_label = peak if peak != 'total' else 'Total'
                            if normalize_by_volume:
                                # Determine units
                                if density_units == 'per_nm3':
                                    units = 'ions/frame/nm³'
                                else:  # 'auto' or 'per_A3'
                                    units = 'ions/frame/Å³'
                                xlabel_text = f'{ion_pair[0]}$^+$ {peak_label} Density ({units})'
                            else:
                                xlabel_text = f'{ion_pair[0]}$^+$ {peak_label} Binding'
                            ax_ind_main.set_xlabel(xlabel_text, 
                                                  fontsize=label_fontsize, fontweight=label_fontweight)
                        else:
                            ax_ind_main.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
                        
                        if ylabel is None:
                            peak_label = peak if peak != 'total' else 'Total'
                            if normalize_by_volume:
                                # Determine units
                                if density_units == 'per_nm3':
                                    units = 'ions/frame/nm³'
                                else:  # 'auto' or 'per_A3'
                                    units = 'ions/frame/Å³'
                                ylabel_text = f'{ion_pair[1]}$^+$ {peak_label} Density ({units})'
                            else:
                                ylabel_text = f'{ion_pair[1]}$^+$ {peak_label} Binding'
                            ax_ind_main.set_ylabel(ylabel_text,
                                                  fontsize=label_fontsize, fontweight=label_fontweight)
                        else:
                            ax_ind_main.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                        
                        # Set title for individual figure
                        if ax_ind_title is None and subplot_title:
                            ax_ind_main.set_title(subplot_title, fontsize=title_fontsize, fontweight=title_fontweight)
                        
                        ind_fig.tight_layout()
                        
                        # Save individual figure if requested
                        if save_individual_figures:
                            base_name = filename.rsplit('.', 1)[0]
                            ext = filename.rsplit('.', 1)[1] if '.' in filename else 'png'
                            ind_filename = f"{base_name}_{combo_key}.{ext}"
                            ind_fig.savefig(ind_filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent_bg)
                            print(f"✓ Individual correlation plot saved: {ind_filename}")
                    
                    plot_idx += 1
        
        # Handle combined figure display and saving
        if create_combined and combined_fig is not None:
            # Hide unused subplots
            if combined_axes is not None:
                for idx in range(plot_idx, len(combined_axes)):
                    if hasattr(combined_axes[idx], 'set_visible'):
                        combined_axes[idx].set_visible(False)
            
            combined_fig.tight_layout()
            
            # Save combined figure if requested
            should_save_combined = save_fig or save_combined_figure
            if should_save_combined:
                combined_fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent_bg)
                print(f"✓ Combined correlation figure saved: {filename}")
            
            # Show combined figure if requested
            if not show_combined_figure:
                plt.close(combined_fig)
        
        # Handle individual figures display
        if not show_individual_figures and individual_figs:
            for ind_fig, _ in individual_figs:
                plt.close(ind_fig)
        
        # Print statistics if requested
        if show_stats and stats_location in ['print', 'both'] and all_stats:
            print("\n" + "="*80)
            print("PEAK-SPECIFIC ION COMPETITION CORRELATION ANALYSIS")
            print("="*80)
            
            for stat in all_stats:
                target_label = stat.get('target', 'Target')
                peak_label = stat.get('peak', 'Unknown')
                print(f"\n{target_label} - {peak_label} Shell:")
                print(f"  Pearson correlation:  r = {stat['pearson_r']:7.3f}  (p = {stat['pearson_p']:.2e})")
                print(f"  Spearman correlation: ρ = {stat['spearman_r']:7.3f}  (p = {stat['spearman_p']:.2e})")
                print(f"\n  Binding State Occupancy:")
                print(f"    Both {stat['ion1']}⁺ and {stat['ion2']}⁺:  {stat['both_bound']:5d} frames  ({stat['both_pct']:5.1f}%)")
                print(f"    Only {stat['ion1']}⁺:              {stat['only_ion1']:5d} frames  ({stat['only_ion1_pct']:5.1f}%)")
                print(f"    Only {stat['ion2']}⁺:              {stat['only_ion2']:5d} frames  ({stat['only_ion2_pct']:5.1f}%)")
                print(f"    Neither:                {stat['neither']:5d} frames  ({stat['neither_pct']:5.1f}%)")
            print("="*80 + "\n")
        
        # Return appropriate figures
        if create_combined and not create_individual:
            return combined_fig, combined_axes
        elif create_individual and not create_combined:
            return individual_figs
        else:
            return combined_fig, combined_axes, individual_figs
    
    def _extract_peak_correlation_data(self, binding_results, target_key, ion_pair, peak1, peak2, is_batch,
                                      normalize_by_volume=False, volume_calculation_method='weighted_average'):
        """
        Extract peak-specific binding data for correlation analysis with optional volume normalization
        """
        # Get data for this target
        if is_batch:
            if target_key not in binding_results:
                return None, None, None
            result = binding_results[target_key]
        else:
            result = binding_results
        
        ion1, ion2 = ion_pair
        
        # Determine binding dictionary location
        if 'cation_binding' in result:
            binding_dict = result['cation_binding']
        elif 'anion_binding' in result:
            binding_dict = result['anion_binding']
        else:
            return None, None, None
        
        if ion1 not in binding_dict or ion2 not in binding_dict:
            return None, None, None
        
        # Extract binding events for specified peaks
        binding1 = self._get_peak_binding_events(binding_dict[ion1], peak1, normalize_by_volume, volume_calculation_method)
        binding2 = self._get_peak_binding_events(binding_dict[ion2], peak2, normalize_by_volume, volume_calculation_method)
        
        if binding1 is None or binding2 is None:
            return None, None, None
        
        data_label = f"{ion1}_{peak1}_vs_{ion2}_{peak2}"
        return binding1, binding2, data_label
    
    def _get_peak_binding_events(self, ion_data, peak, normalize_by_volume=False, volume_calculation_method='weighted_average'):
        """
        Extract binding events for a specific peak or total with optional volume normalization
        """
        if peak == 'total':
            if normalize_by_volume:
                # Use volume-normalized total binding
                normalized_data = self._extract_volume_normalized_timeseries(ion_data, volume_calculation_method)
                if normalized_data is not None:
                    return normalized_data
                else:
                    print("Warning: No volume data available for total binding, using raw events")
                    return ion_data['binding_events']
            else:
                return ion_data['binding_events']
        else:
            # Check if peak analysis data exists
            if 'peak_analysis' in ion_data and peak in ion_data['peak_analysis']:
                peak_data = ion_data['peak_analysis'][peak]
                if normalize_by_volume:
                    # Create per-frame volume-normalized density
                    if ('volume_data' in peak_data and peak_data['volume_data'] is not None and
                        'binding_events' in peak_data):
                        volume = peak_data['volume_data']['volume']
                        raw_events = peak_data['binding_events']
                        # Convert raw events to per-frame density
                        volume_normalized_events = np.array(raw_events) / volume
                        return volume_normalized_events
                    elif 'volume_density' in peak_data and hasattr(peak_data['volume_density'], '__len__'):
                        # If volume_density is already per-frame array
                        return peak_data['volume_density']
                    else:
                        print(f"Warning: No volume data available for peak {peak}, using raw events")
                        return peak_data['binding_events']
                else:
                    return peak_data['binding_events']
            else:
                return None
    
    def _plot_single_peak_correlation(self, ax, binding1, binding2, ion1, ion2, peak1, peak2, 
                                      kde_data1=None, kde_data2=None, **kwargs):
        """
        Plot correlation for specific peaks with enhanced labeling
        """
        # Use existing correlation plotting logic with peak-aware labels
        stats = self._plot_single_ion_correlation(ax, binding1, binding2, ion1, ion2, 
                                                 kde_data1=kde_data1, kde_data2=kde_data2, **kwargs)
        
        # Enhance statistics with peak information
        if stats:
            stats['peak1'] = peak1
            stats['peak2'] = peak2
        
        return stats
    
    def _plot_ion_competition_correlation_advanced(self, binding_results, target_configs, 
                                                  default_ion_pair, default_peaks_to_show, peak_comparison_mode,
                                                  default_plot_type, show_marginals, color_by, colormap,
                                                  point_size, point_alpha, bubble_scale, show_bubble_labels,
                                                  show_center_markers, marker_size, marker_color, marker_edgecolor,
                                                  marker_edgewidth, hexbin_gridsize, contour_levels, contour_colors,
                                                  show_data_points, data_point_marker, data_point_size,
                                                  data_point_color, data_point_alpha, show_stats, stats_location,
                                                  stats_fontsize, show_reference_lines, show_summary,
                                                  title_fontsize, title_fontweight, label_fontsize,
                                                  label_fontweight, tick_fontsize, subplot_layout, custom_labels,
                                                  default_title, xlabel, ylabel, show_grid, grid_alpha, xlim, ylim,
                                                  figsize, show_individual_figures, individual_figsize,
                                                  save_individual_figures, show_combined_figure, save_combined_figure,
                                                  save_fig, filename, dpi, bbox_inches, transparent_bg,
                                                  # Volume normalization parameters
                                                  normalize_by_volume, density_units, volume_calculation_method):
        """
        Handle advanced target-specific configuration for ion competition correlation
        
        target_configs format:
        {
            'carboxylic_acid': {'peaks_to_show': ['P1'], 'ion_pair': ('NA', 'K'), 'plot_type': 'hexbin'},
            'quinolone': {'peaks_to_show': ['P2'], 'ion_pair': ('K', 'RB')},
            'piperazine': {'peaks_to_show': ['P1', 'P3']}  # inherits default ion_pair
        }
        """
        
        print(f"📊 Creating advanced target-specific ion competition correlation plots")
        print(f"   Target configurations: {list(target_configs.keys())}")
        
        # Check if batch results
        is_batch = self._is_batch_binding_result(binding_results)
        if not is_batch:
            print("⚠️  Advanced configuration requires batch results (multiple targets)")
            return None
        
        # Parse target configurations and expand for multiple peaks per target
        plot_configs = []
        for target_key, config in target_configs.items():
            if target_key not in binding_results:
                print(f"⚠️  Target '{target_key}' not found in binding results")
                continue
            
            # Extract target-specific parameters with defaults
            target_peaks = config.get('peaks_to_show', default_peaks_to_show)
            target_ion_pair = config.get('ion_pair', default_ion_pair)
            target_plot_type = config.get('plot_type', default_plot_type)
            target_title = config.get('title', None)
            
            # Normalize peaks to list
            if isinstance(target_peaks, str):
                target_peaks = [target_peaks]
            
            # Create separate plot config for each peak
            for peak in target_peaks:
                plot_config = {
                    'target_key': target_key,
                    'peak': peak,
                    'ion_pair': target_ion_pair,
                    'plot_type': target_plot_type,
                    'title': target_title,
                    'config': config  # Store original config for additional parameters
                }
                plot_configs.append(plot_config)
        
        if not plot_configs:
            print("⚠️  No valid plot configurations found")
            return None
        
        n_plots = len(plot_configs)
        print(f"   Creating {n_plots} subplot(s)")
        
        # Determine subplot layout
        if subplot_layout is not None:
            nrows, ncols = subplot_layout
        else:
            ncols = min(4, n_plots)  # Max 4 columns
            nrows = (n_plots + ncols - 1) // ncols
        
        # Figure size calculation
        if figsize is None:
            base_size = 6 if show_marginals else 5
            figsize = (base_size * ncols, base_size * nrows)
        
        # Storage for statistics and figures
        all_stats = []
        individual_figs = []
        
        # Determine figure creation mode
        create_combined = show_combined_figure or n_plots == 1
        create_individual = n_plots > 1 and (show_individual_figures or save_individual_figures)
        
        # Set individual figure size
        if individual_figsize is None:
            individual_figsize = (8, 8) if show_marginals else (8, 6)
        
        # Create combined figure if requested
        combined_fig = None
        combined_axes = None
        
        if create_combined:
            if show_marginals and n_plots == 1:
                # Single subplot with marginals
                combined_fig = plt.figure(figsize=figsize)
                gs = combined_fig.add_gridspec(3, 3, 
                                 width_ratios=[4, 0.8, 0.3],
                                 height_ratios=[0.3, 0.8, 4],
                                 hspace=0.02, wspace=0.02,
                                 left=0.1, right=0.95, top=0.95, bottom=0.1)
                
                ax_main = combined_fig.add_subplot(gs[2, 0])
                ax_top = combined_fig.add_subplot(gs[1, 0], sharex=ax_main)
                ax_right = combined_fig.add_subplot(gs[2, 1], sharey=ax_main)
                ax_title = combined_fig.add_subplot(gs[0, 0])
                ax_title.axis('off')
                ax_cbar = combined_fig.add_subplot(gs[2, 2])
                
                combined_axes = [(ax_main, ax_top, ax_right, ax_title, ax_cbar)]
            else:
                # Multiple subplots - no marginals for grid layout
                combined_fig, axes_grid = plt.subplots(nrows, ncols, figsize=figsize)
                if n_plots == 1:
                    combined_axes = [axes_grid]
                elif nrows == 1 or ncols == 1:
                    combined_axes = list(axes_grid.flatten()) if hasattr(axes_grid, 'flatten') else [axes_grid]
                else:
                    combined_axes = list(axes_grid.flatten())
        
        # Plot each configuration
        for plot_idx, plot_config in enumerate(plot_configs):
            target_key = plot_config['target_key']
            peak = plot_config['peak']
            ion_pair = plot_config['ion_pair']
            plot_type = plot_config['plot_type']
            target_title = plot_config['title']
            config = plot_config['config']
            
            print(f"   Processing {target_key} + {peak} (ions: {ion_pair})...")
            
            # Prepare axes for combined figure
            if create_combined:
                if show_marginals and n_plots == 1:
                    ax_main, ax_top, ax_right, ax_title, ax_cbar = combined_axes[plot_idx]
                else:
                    ax_main = combined_axes[plot_idx] if plot_idx < len(combined_axes) else None
                    ax_top, ax_right, ax_title, ax_cbar = None, None, None, None
            else:
                ax_main, ax_top, ax_right, ax_title, ax_cbar = None, None, None, None, None
            
            # Prepare axes for individual figure if requested
            if create_individual:
                ind_fig = plt.figure(figsize=individual_figsize)
                if show_marginals:
                    gs_ind = ind_fig.add_gridspec(3, 3,
                                                  width_ratios=[4, 0.8, 0.3],
                                                  height_ratios=[0.3, 0.8, 4],
                                                  hspace=0.02, wspace=0.02,
                                                  left=0.1, right=0.95, top=0.95, bottom=0.1)
                    ax_ind_main = ind_fig.add_subplot(gs_ind[2, 0])
                    ax_ind_top = ind_fig.add_subplot(gs_ind[1, 0], sharex=ax_ind_main)
                    ax_ind_right = ind_fig.add_subplot(gs_ind[2, 1], sharey=ax_ind_main)
                    ax_ind_title = ind_fig.add_subplot(gs_ind[0, 0])
                    ax_ind_title.axis('off')
                    ax_ind_cbar = ind_fig.add_subplot(gs_ind[2, 2])
                else:
                    ax_ind_main = ind_fig.add_subplot(111)
                    ax_ind_top, ax_ind_right, ax_ind_title, ax_ind_cbar = None, None, None, None
                
                combo_key = f"{target_key}_{peak}_{ion_pair[0]}{ion_pair[1]}"
                individual_figs.append((ind_fig, combo_key))
            else:
                ax_ind_main, ax_ind_top, ax_ind_right, ax_ind_title, ax_ind_cbar = None, None, None, None, None
            
            # Extract peak-specific binding data
            binding1, binding2, data_label = self._extract_peak_correlation_data(
                binding_results, target_key, ion_pair, peak, peak, True,
                normalize_by_volume, volume_calculation_method
            )
            
            if binding1 is None or binding2 is None:
                print(f"⚠️  Skipping {target_key} + {peak} + {ion_pair}: data not available")
                continue
            
            # Generate subplot title
            if target_title and len(target_peaks) == 1:
                # Custom title only applies when there's a single peak
                subplot_title = target_title
            elif target_title and len(target_peaks) > 1:
                # For multiple peaks with custom title, append peak info
                if '{peak}' in target_title:
                    # Title template with {peak} placeholder
                    subplot_title = target_title.format(peak=peak)
                else:
                    # Append peak to custom title
                    subplot_title = f"{target_title} - {peak}"
            elif custom_labels and target_key in custom_labels:
                target_display = custom_labels[target_key]
                subplot_title = f"{target_display} - {peak} ({ion_pair[0]}⁺ vs {ion_pair[1]}⁺)"
            else:
                subplot_title = f"{target_key} - {peak} ({ion_pair[0]}⁺ vs {ion_pair[1]}⁺)"
            
            # Get target-specific parameters or use defaults
            target_specific_params = {
                'title': subplot_title,
                'plot_type': plot_type,
                'color_by': config.get('color_by', color_by),
                'colormap': config.get('colormap', colormap),
                'point_size': config.get('point_size', point_size),
                'point_alpha': config.get('point_alpha', point_alpha),
                'bubble_scale': config.get('bubble_scale', bubble_scale),
                'show_bubble_labels': config.get('show_bubble_labels', show_bubble_labels),
                'show_center_markers': config.get('show_center_markers', show_center_markers),
                'marker_size': config.get('marker_size', marker_size),
                'marker_color': config.get('marker_color', marker_color),
                'marker_edgecolor': config.get('marker_edgecolor', marker_edgecolor),
                'marker_edgewidth': config.get('marker_edgewidth', marker_edgewidth),
                'hexbin_gridsize': config.get('hexbin_gridsize', hexbin_gridsize),
                'contour_levels': config.get('contour_levels', contour_levels),
                'contour_colors': config.get('contour_colors', contour_colors),
                'show_data_points': config.get('show_data_points', show_data_points),
                'data_point_marker': config.get('data_point_marker', data_point_marker),
                'data_point_size': config.get('data_point_size', data_point_size),
                'data_point_color': config.get('data_point_color', data_point_color),
                'data_point_alpha': config.get('data_point_alpha', data_point_alpha),
                'show_stats': config.get('show_stats', show_stats),
                'stats_location': config.get('stats_location', stats_location),
                'stats_fontsize': config.get('stats_fontsize', stats_fontsize),
                'show_reference_lines': config.get('show_reference_lines', show_reference_lines),
                'show_grid': config.get('show_grid', show_grid),
                'grid_alpha': config.get('grid_alpha', grid_alpha),
                'xlim': config.get('xlim', xlim),
                'ylim': config.get('ylim', ylim),
                'title_fontsize': title_fontsize,
                'title_fontweight': title_fontweight,
                'label_fontsize': label_fontsize,
                'label_fontweight': label_fontweight,
                'tick_fontsize': tick_fontsize
            }
            
            # Plot on combined figure axes if created
            if create_combined and ax_main is not None:
                stats = self._plot_single_peak_correlation(
                    ax_main, binding1, binding2, ion_pair[0], ion_pair[1], peak, peak,
                    ax_top=ax_top, ax_right=ax_right, ax_title=ax_title, ax_cbar=ax_cbar,
                    **target_specific_params
                )
                
                if stats:
                    stats['target'] = target_key
                    stats['peak'] = peak
                    stats['ion_pair'] = ion_pair
                    all_stats.append(stats)
                
                # Set axis labels
                target_xlabel = config.get('xlabel', None)
                target_ylabel = config.get('ylabel', None)
                
                if target_xlabel is None:
                    peak_label = peak if peak != 'total' else 'Total'
                    ax_main.set_xlabel(f'{ion_pair[0]}$^+$ {peak_label} Binding', 
                                      fontsize=label_fontsize, fontweight=label_fontweight)
                else:
                    ax_main.set_xlabel(target_xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
                
                if target_ylabel is None:
                    peak_label = peak if peak != 'total' else 'Total'
                    ax_main.set_ylabel(f'{ion_pair[1]}$^+$ {peak_label} Binding',
                                      fontsize=label_fontsize, fontweight=label_fontweight)
                else:
                    ax_main.set_ylabel(target_ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                
                # Set title if no dedicated title axis
                if ax_title is None and subplot_title:
                    ax_main.set_title(subplot_title, fontsize=title_fontsize, fontweight=title_fontweight)
            
            # Plot on individual figure axes if created
            if create_individual and ax_ind_main is not None:
                self._plot_single_peak_correlation(
                    ax_ind_main, binding1, binding2, ion_pair[0], ion_pair[1], peak, peak,
                    ax_top=ax_ind_top, ax_right=ax_ind_right, ax_title=ax_ind_title, ax_cbar=ax_ind_cbar,
                    **target_specific_params
                )
                
                # Set axis labels for individual figure
                target_xlabel = config.get('xlabel', None)
                target_ylabel = config.get('ylabel', None)
                
                if target_xlabel is None:
                    peak_label = peak if peak != 'total' else 'Total'
                    if normalize_by_volume:
                        # Determine units
                        if density_units == 'per_nm3':
                            units = 'ions/frame/nm³'
                        else:  # 'auto' or 'per_A3'
                            units = 'ions/frame/Å³'
                        xlabel_text = f'{ion_pair[0]}$^+$ {peak_label} Density ({units})'
                    else:
                        xlabel_text = f'{ion_pair[0]}$^+$ {peak_label} Binding'
                    ax_ind_main.set_xlabel(xlabel_text, 
                                          fontsize=label_fontsize, fontweight=label_fontweight)
                else:
                    ax_ind_main.set_xlabel(target_xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
                
                if target_ylabel is None:
                    peak_label = peak if peak != 'total' else 'Total'
                    if normalize_by_volume:
                        # Determine units
                        if density_units == 'per_nm3':
                            units = 'ions/frame/nm³'
                        else:  # 'auto' or 'per_A3'
                            units = 'ions/frame/Å³'
                        ylabel_text = f'{ion_pair[1]}$^+$ {peak_label} Density ({units})'
                    else:
                        ylabel_text = f'{ion_pair[1]}$^+$ {peak_label} Binding'
                    ax_ind_main.set_ylabel(ylabel_text,
                                          fontsize=label_fontsize, fontweight=label_fontweight)
                else:
                    ax_ind_main.set_ylabel(target_ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                
                # Set title for individual figure
                if ax_ind_title is None and subplot_title:
                    ax_ind_main.set_title(subplot_title, fontsize=title_fontsize, fontweight=title_fontweight)
                
                ind_fig.tight_layout()
                
                # Save individual figure if requested
                if save_individual_figures:
                    base_name = filename.rsplit('.', 1)[0]
                    ext = filename.rsplit('.', 1)[1] if '.' in filename else 'png'
                    ind_filename = f"{base_name}_{combo_key}.{ext}"
                    ind_fig.savefig(ind_filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent_bg)
                    print(f"✓ Individual advanced correlation plot saved: {ind_filename}")
        
        # Handle combined figure display and saving
        if create_combined and combined_fig is not None:
            # Hide unused subplots
            if combined_axes is not None:
                for idx in range(len(plot_configs), len(combined_axes)):
                    if hasattr(combined_axes[idx], 'set_visible'):
                        combined_axes[idx].set_visible(False)
            
            combined_fig.tight_layout()
            
            # Save combined figure if requested
            should_save_combined = save_fig or save_combined_figure
            if should_save_combined:
                combined_fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent_bg)
                print(f"✓ Combined advanced correlation figure saved: {filename}")
            
            # Show combined figure if requested
            if not show_combined_figure:
                plt.close(combined_fig)
        
        # Handle individual figures display
        if not show_individual_figures and individual_figs:
            for ind_fig, _ in individual_figs:
                plt.close(ind_fig)
        
        # Print statistics if requested
        if show_stats and stats_location in ['print', 'both'] and all_stats:
            print("\n" + "="*80)
            print("ADVANCED TARGET-SPECIFIC ION COMPETITION CORRELATION ANALYSIS")
            print("="*80)
            
            for stat in all_stats:
                target_label = stat.get('target', 'Target')
                peak_label = stat.get('peak', 'Unknown')
                ion_pair = stat.get('ion_pair', ('Ion1', 'Ion2'))
                print(f"\n{target_label} - {peak_label} Shell ({ion_pair[0]}⁺ vs {ion_pair[1]}⁺):") 
                print(f"  Pearson correlation:  r = {stat['pearson_r']:7.3f}  (p = {stat['pearson_p']:.2e})")
                print(f"  Spearman correlation: ρ = {stat['spearman_r']:7.3f}  (p = {stat['spearman_p']:.2e})")
                print(f"\n  Binding State Occupancy:")
                print(f"    Both {stat['ion1']}⁺ and {stat['ion2']}⁺:  {stat['both_bound']:5d} frames  ({stat['both_pct']:5.1f}%)")
                print(f"    Only {stat['ion1']}⁺:              {stat['only_ion1']:5d} frames  ({stat['only_ion1_pct']:5.1f}%)")
                print(f"    Only {stat['ion2']}⁺:              {stat['only_ion2']:5d} frames  ({stat['only_ion2_pct']:5.1f}%)")
                print(f"    Neither:                {stat['neither']:5d} frames  ({stat['neither_pct']:5.1f}%)")
            print("="*80 + "\n")
        
        # Return appropriate figures
        if create_combined and not create_individual:
            return combined_fig, combined_axes
        elif create_individual and not create_combined:
            return individual_figs
        else:
            return combined_fig, combined_axes, individual_figs
    
    # =========================================================================
    # COORDINATION ANALYSIS PLOTTING
    # =========================================================================
    
    def plot_coordination_distribution(self, coordination_results,
                                      # Volume normalization parameters
                                      normalize_by_volume=False,
                                      density_units='auto',  # 'auto', 'per_A3', 'per_nm3'
                                      volume_calculation_method='sum',  # 'sum', 'weighted_average'
                                      # Data source control
                                      solvent_type='OW',  # 'OW' for water, ion name for ions
                                      peak_selection=None,  # List of peaks to include (e.g., ['S1', 'S2'])
                                      # Histogram control
                                      bins=30,
                                      density=True,
                                      align_histograms_to_grids=True,
                                      # Bar width control
                                      bar_width=None,  # Explicit bar width for integer bins (e.g., 0.3, 0.8)
                                      bar_width_scale=1.0,  # Scale factor for bar width (< 1.0 narrows bars)
                                      # Bar styling
                                      bar_color='steelblue',
                                      bar_edgecolor='black',
                                      bar_alpha=0.7,
                                      bar_edgewidth=1.0,
                                      # Value labels on bars
                                      show_values=True,
                                      value_label_fontsize=9,
                                      value_label_fontweight='normal',
                                      value_format='{:.2f}',
                                      value_label_threshold=0.01,
                                      value_offset=0.02,
                                      # Mean line control
                                      show_mean=False,  # Simple on/off control
                                      show_mean_line=True,  # Backward compatibility
                                      mean_line_color='red',
                                      mean_linestyle='--',
                                      mean_linewidth=2,
                                      mean_alpha=1.0,
                                      # Font & text control
                                      title='Coordination Number Distribution',
                                      title_fontsize=14,
                                      title_fontweight='bold',
                                      show_title=True,
                                      label_fontsize=12,
                                      label_fontweight='normal',
                                      tick_fontsize=10,
                                      legend_fontsize=10,
                                      legend_fontweight='normal',
                                      # Axis labels
                                      xlabel='Coordination Number',
                                      ylabel=None,  # Auto-determined based on normalization
                                      # Legend control
                                      show_legend=True,
                                      legend_loc='best',
                                      legend_framealpha=0.9,
                                      # Grid control
                                      show_grid=True,
                                      grid_alpha=0.3,
                                      grid_linestyle='--',
                                      # Axis limits
                                      xlim=None,
                                      ylim=None,
                                      # Figure export control
                                      save_fig=False,
                                      filename='coordination_dist.png',
                                      dpi=300,
                                      figsize=(8, 6),
                                      bbox_inches='tight',
                                      transparent_bg=False):
        """
        Plot distribution of coordination numbers with volume normalization support.
        Compatible with both water_solvation_analysis() and ion_binding_analysis() results.
        
        Parameters
        ----------
        coordination_results : dict
            Results from water_solvation_analysis() or ion_binding_analysis()
            Expected structure:
            - For water: results['water_solvation'][solvent_type]
            - For ions: results['cation_binding'][ion_name] or results['anion_binding'][ion_name]
        
        Volume Normalization Parameters
        -------------------------------
        normalize_by_volume : bool
            Whether to use volume-normalized density data instead of raw coordination counts (default: False)
        density_units : str
            Units for density display (default: 'auto'):
            - 'auto': Automatically choose based on volume magnitudes
            - 'per_A3': Display as "solvents/frame/Å³"
            - 'per_nm3': Display as "solvents/frame/nm³"
        volume_calculation_method : str
            Method for combining multiple peaks (default: 'sum'):
            - 'sum': Sum coordination across all selected peaks
            - 'weighted_average': Volume-weighted average of peak densities
        
        Data Source Control
        -------------------
        solvent_type : str
            Type of solvent/ion to analyze (default: 'OW' for water oxygen)
            For water analysis: 'OW'
            For ion analysis: ion name like 'NA', 'K', 'CL', etc.
        peak_selection : list or None
            List of peaks/shells to include (default: None = all available)
            Examples: ['S1', 'S2'] for water shells, ['P1', 'P2'] for ion peaks
        
        Histogram Control
        -----------------
        bins : int or array
            Number of bins or bin edges (default: 30)
        density : bool
            If True, draw probability density; if False, draw counts (default: True)
        align_histograms_to_grids : bool
            If True, align histogram bars with integer coordination numbers;
            bars center on 0, 1, 2, 3, etc. If False, use standard binning (default: True)
        
        Bar Width Control
        -----------------
        bar_width : float or None
            Explicit bar width for integer-binned histograms (default: None)
            Sets the actual width of bars in data coordinates (e.g., 0.3, 0.8)
            Only applies when align_histograms_to_grids=True
            Example: bar_width=0.5 makes bars half as wide (0.5 units instead of 1.0)
        bar_width_scale : float
            Scale factor for visual bar width via x-axis adjustment (default: 1.0)
            Values < 1.0 zoom in to make bars appear narrower
            Useful when few coordination numbers make bars appear too wide
            Example: bar_width_scale=0.5 zooms in to make bars appear half as wide
        
        Bar Styling
        -----------
        bar_color : str
            Bar fill color (default: 'steelblue')
        bar_edgecolor : str
            Bar edge color (default: 'black')
        bar_alpha : float
            Bar transparency 0-1 (default: 0.7)
        bar_edgewidth : float
            Bar edge width (default: 1.0)
        
        Value Labels on Bars
        --------------------
        show_values : bool
            Whether to show values on top of bars (default: True)
        value_label_fontsize : float
            Font size for value labels on top of bars (default: 9)
        value_label_fontweight : str
            Font weight for value labels: 'normal', 'bold' (default: 'normal')
        value_format : str
            Format string for values (default: '{:.3f}')
        value_label_threshold : float
            Minimum bar height to show label; bars below this value are not labeled (default: 0.01)
        value_offset : float
            Vertical offset as fraction of y-range (default: 0.02)
        
        Mean Line Control
        -----------------
        show_mean : bool
            Whether to show mean coordination line (default: False)
        show_mean_line : bool
            Whether to show mean coordination line (default: True, for backward compatibility)
        mean_line_color : str
            Mean line color (default: 'red')
        mean_linestyle : str
            Mean line style: '-', '--', '-.', ':' (default: '--')
        mean_linewidth : float
            Mean line width (default: 2)
        mean_alpha : float
            Mean line transparency (default: 1.0)
        
        Font & Text Control
        -------------------
        title : str
            Plot title (default: 'Coordination Number Distribution')
        title_fontsize : float
            Title font size (default: 14)
        title_fontweight : str
            Title font weight: 'normal', 'bold' (default: 'bold')
        show_title : bool
            Whether to show title (default: True)
        label_fontsize : float
            Axis label font size (default: 12)
        label_fontweight : str
            Axis label font weight (default: 'normal')
        tick_fontsize : float
            Tick label font size (default: 10)
        legend_fontsize : float
            Legend font size (default: 10)
        legend_fontweight : str
            Legend font weight (default: 'normal')
        
        Axis Labels
        -----------
        xlabel : str
            X-axis label (default: 'Coordination Number')
        ylabel : str or None
            Y-axis label. If None, auto-determined based on normalization and density settings
        
        Legend Control
        --------------
        show_legend : bool
            Whether to show legend (default: True)
        legend_loc : str
            Legend location (default: 'best')
        legend_framealpha : float
            Legend background transparency (default: 0.9)
        
        Grid Control
        ------------
        show_grid : bool
            Whether to show grid (default: True)
        grid_alpha : float
            Grid transparency (default: 0.3)
        grid_linestyle : str
            Grid line style (default: '--')
        
        Axis Limits
        -----------
        xlim : tuple, optional
            X-axis limits (e.g., (0, 10))
        ylim : tuple, optional
            Y-axis limits
        
        Figure Export Control
        ---------------------
        save_fig : bool
            Whether to save figure (default: False)
        filename : str
            Output filename (default: 'coordination_dist.png')
        dpi : int
            Resolution (default: 300)
        figsize : tuple
            Figure size (default: (8, 6))
        bbox_inches : str
            Bounding box for saved figure (default: 'tight')
        transparent_bg : bool
            Whether to save with transparent background (default: False)
        
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        
        Examples
        --------
        >>> # Water solvation distribution (volume-normalized)
        >>> plotter.plot_coordination_distribution(
        ...     water_solvation_results['carboxylic_acid'],
        ...     normalize_by_volume=True,
        ...     volume_calculation_method='sum',
        ...     solvent_type='OW',
        ...     peak_selection=['S1', 'S2'],
        ...     title='Water Solvation Distribution',
        ...     xlabel='Water Coordination',
        ...     save_fig=True
        ... )
        
        >>> # Ion binding distribution (traditional)
        >>> plotter.plot_coordination_distribution(
        ...     ion_binding_results,
        ...     solvent_type='NA',
        ...     normalize_by_volume=False,
        ...     title='Na⁺ Coordination',
        ...     bar_color='navy'
        ... )
        """
        
        # Extract coordination data based on analysis type
        coord_numbers = self._extract_coordination_data(
            coordination_results, solvent_type, peak_selection,
            normalize_by_volume, volume_calculation_method
        )
        
        if coord_numbers is None or len(coord_numbers) == 0:
            print("No coordination data found to plot")
            return None, None
        
        # Auto-determine ylabel based on normalization and density
        if ylabel is None:
            if normalize_by_volume:
                # Determine appropriate density units
                if density_units == 'auto':
                    # Choose units based on data magnitude
                    mean_val = np.mean(coord_numbers)
                    if mean_val > 1.0:
                        units = 'Å³'
                    else:
                        units = 'nm³' if mean_val < 0.001 else 'Å³'
                elif density_units == 'per_nm3':
                    units = 'nm³'
                else:  # 'per_A3'
                    units = 'Å³'
                
                if density:
                    ylabel = f'Probability Density\n(density/frame/{units})'
                else:
                    ylabel = f'Count\n(density/frame/{units})'
                
                # Update xlabel for volume-normalized data
                if 'Coordination Number' in xlabel:
                    solvent_name = 'Water' if solvent_type == 'OW' else solvent_type
                    xlabel = f'{solvent_name} Density (molecules/frame/{units})'
            else:
                if density:
                    ylabel = 'Probability Density'
                else:
                    ylabel = 'Count'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bins aligned with integer coordination numbers if requested
        # For volume-normalized data, use continuous binning
        if align_histograms_to_grids and isinstance(bins, int) and not normalize_by_volume:
            # Integer binning for discrete coordination counts
            # Bin edges at -0.5, 0.5, 1.5, ... so bars center on 0, 1, 2, 3, ...
            # Always use standard 1.0 width bins - bar_width only affects xlim zoom
            max_coord = int(np.ceil(coord_numbers.max()))
            bins = np.arange(-0.5, max_coord + 1.5, 1.0)
        elif isinstance(bins, int):
            # For volume-normalized or continuous data, use regular binning
            bins = np.linspace(coord_numbers.min(), coord_numbers.max(), bins + 1)
        
        # Create histogram
        n, bin_edges, patches = ax.hist(coord_numbers, bins=bins, 
                                        color=bar_color, edgecolor=bar_edgecolor, 
                                        alpha=bar_alpha, density=density,
                                        linewidth=bar_edgewidth)
        
        # Apply x-axis limit adjustments for bar width control
        if xlim is None:
            # Apply bar_width zoom (like plot_ion_binding_peak_breakdown for single ion)
            # The key: smaller bar_width = we want bars to LOOK narrow = zoom OUT
            # This seems counterintuitive but: narrow physical bars + wide view = visually narrow
            if bar_width is not None and align_histograms_to_grids and not normalize_by_volume:
                # Get integer positions where bars exist
                min_coord = int(np.floor(coord_numbers.min()))
                max_coord = int(np.ceil(coord_numbers.max()))
                
                # Calculate xlim exactly like plot_ion_binding_peak_breakdown
                # Bar centers are at integer positions: 0, 1, 2, 3...
                all_bar_positions = list(range(min_coord, max_coord + 1))
                
                # Calculate limits based on bar edges
                min_pos = min(all_bar_positions) - bar_width/2
                max_pos = max(all_bar_positions) + bar_width/2
                
                # Fixed padding regardless of bar_width
                padding = 0.5
                
                xlim = (min_pos - padding, max_pos + padding)
            
            # Apply bar_width_scale x-axis adjustment (zoom effect)
            elif bar_width_scale != 1.0:
                # Calculate the range of actual data
                min_coord = coord_numbers.min()
                max_coord = coord_numbers.max()
                data_range = max_coord - min_coord
                
                # Calculate center of data
                center = (min_coord + max_coord) / 2
                
                # Scale the view range
                scaled_range = data_range * bar_width_scale
                
                # Add padding (at least 0.5 units on each side)
                padding = max(0.5, scaled_range * 0.3)
                
                # Set x-axis limits to zoom in on bars
                xlim = (center - scaled_range/2 - padding, center + scaled_range/2 + padding)
        
        # Add value labels on bars
        if show_values:
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            offset = y_range * value_offset
            
            for i, (height, edge) in enumerate(zip(n, bin_edges[:-1])):
                if height > value_label_threshold:  # Only label bars above threshold
                    bin_center = edge + (bin_edges[i+1] - edge) / 2
                    ax.text(bin_center, height + offset,
                           value_format.format(height),
                           ha='center', va='bottom',
                           fontsize=value_label_fontsize,
                           fontweight=value_label_fontweight)
        
        # Calculate statistics
        mean_coord = np.mean(coord_numbers)
        std_coord = np.std(coord_numbers)
        
        # Add mean line with appropriate label
        # When show_mean is explicitly set (True or False), it takes precedence
        # When show_mean is default False, use show_mean_line for backward compatibility
        if show_mean is not False:  # show_mean explicitly True
            show_mean_final = True
        else:  # show_mean is False (either default or explicit)
            show_mean_final = False
        
        if show_mean_final:
            if normalize_by_volume:
                label_text = f'Mean: {mean_coord:.4f} ± {std_coord:.4f}'
            else:
                label_text = f'Mean: {mean_coord:.2f} ± {std_coord:.2f}'
            
            ax.axvline(mean_coord, color=mean_line_color, linestyle=mean_linestyle,
                      linewidth=mean_linewidth, alpha=mean_alpha,
                      label=label_text)
        
        # Axis labels
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        
        # Title
        if show_title:
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        # Tick formatting
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        # Grid
        if show_grid:
            ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
        
        # Legend (without frame)
        if show_legend and show_mean_final:
            legend = ax.legend(loc=legend_loc, framealpha=0,  # Remove frame
                             fontsize=legend_fontsize, frameon=False)  # Disable frame
            for text in legend.get_texts():
                text.set_fontweight(legend_fontweight)
        
        # Axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, 
                       transparent=transparent_bg)
            print(f"✓ Figure saved: {filename}")
        
        return fig, ax

    def _extract_coordination_data(self, coordination_results, solvent_type, peak_selection,
                                   normalize_by_volume, volume_calculation_method):
        """
        Extract coordination data from water_solvation_analysis() or ion_binding_analysis() results.
        Handles both raw coordination and volume-normalized density data.
        Enhanced to support both water and ion data from water_solvation_analysis().
        """
        
        # Determine data source based on result structure
        data_source = None
        
        # Check for water solvation results
        if 'water_solvation' in coordination_results:
            if solvent_type in coordination_results['water_solvation']:
                data_source = coordination_results['water_solvation'][solvent_type]
                analysis_type = 'water'
            else:
                print(f"Warning: Solvent type '{solvent_type}' not found in water_solvation results")
        
        # Check for ion solvation results (from enhanced water_solvation_analysis)
        if data_source is None and 'ion_solvation' in coordination_results:
            if solvent_type in coordination_results['ion_solvation']:
                data_source = coordination_results['ion_solvation'][solvent_type]
                analysis_type = 'ion'
            else:
                print(f"Warning: Ion type '{solvent_type}' not found in ion_solvation results")
                
        # Check for ion binding results (legacy format)
        if data_source is None and ('cation_binding' in coordination_results or 'anion_binding' in coordination_results):
            analysis_type = 'ion'
            
            # Look for ion in cation or anion binding
            if 'cation_binding' in coordination_results and solvent_type in coordination_results['cation_binding']:
                data_source = coordination_results['cation_binding'][solvent_type]
            elif 'anion_binding' in coordination_results and solvent_type in coordination_results['anion_binding']:
                data_source = coordination_results['anion_binding'][solvent_type]
            else:
                print(f"Warning: Ion type '{solvent_type}' not found in binding results")
                
        # Check for legacy format (direct coordination_numbers)
        if data_source is None and 'coordination_numbers' in coordination_results:
            analysis_type = 'legacy'
            return coordination_results['coordination_numbers']
        
        # Final fallback
        if data_source is None:
            print("Warning: Unrecognized data format")
            return None
        
        if data_source is None:
            print(f"Warning: No data found for {solvent_type}")
            return None
        
        # Extract coordination data based on normalization preference
        if normalize_by_volume and 'peak_analysis' in data_source:
            # Volume-normalized analysis using peak data
            return self._extract_volume_normalized_coordination(
                data_source, peak_selection, volume_calculation_method
            )
        else:
            # Raw coordination data
            if peak_selection is not None and 'peak_analysis' in data_source:
                # Apply peak selection for non-normalized data
                return self._extract_raw_coordination_by_peaks(
                    data_source, peak_selection
                )
            elif 'binding_events' in data_source:
                return data_source['binding_events']
            elif 'coordination_numbers' in data_source:
                return data_source['coordination_numbers']
            else:
                print("Warning: No coordination data found in results")
                return None

    def _extract_volume_normalized_coordination(self, data_source, peak_selection, volume_calculation_method):
        """
        Extract volume-normalized coordination data from peak analysis.
        """
        
        if 'peak_analysis' not in data_source:
            print("Warning: No peak analysis data available for volume normalization")
            return None
        
        peak_analysis = data_source['peak_analysis']
        
        # Determine peaks to include
        if peak_selection is None:
            peaks_to_use = list(peak_analysis.keys())
        else:
            peaks_to_use = [p for p in peak_selection if p in peak_analysis]
        
        if not peaks_to_use:
            print("Warning: No valid peaks found for analysis")
            return None
        
        print(f"   Using peaks: {peaks_to_use}")
        print(f"   Volume calculation method: {volume_calculation_method}")
        
        if volume_calculation_method == 'sum':
            # Sum volume-normalized densities across peaks
            total_density = None
            
            for peak in peaks_to_use:
                peak_data = peak_analysis[peak]
                
                if 'volume_density' in peak_data and peak_data['volume_density'] is not None:
                    if total_density is None:
                        total_density = peak_data['volume_density'].copy()
                    else:
                        total_density += peak_data['volume_density']
                else:
                    print(f"Warning: No volume density data for peak {peak}")
            
            return total_density
            
        elif volume_calculation_method == 'weighted_average':
            # Volume-weighted average of peak densities
            weighted_density = None
            total_volume = 0
            
            for peak in peaks_to_use:
                peak_data = peak_analysis[peak]
                
                if ('volume_density' in peak_data and peak_data['volume_density'] is not None and
                    'volume_data' in peak_data and peak_data['volume_data'] is not None):
                    
                    volume = peak_data['volume_data']['volume']
                    density = peak_data['volume_density']
                    
                    if weighted_density is None:
                        weighted_density = density * volume
                    else:
                        weighted_density += density * volume
                    
                    total_volume += volume
                else:
                    print(f"Warning: Incomplete volume data for peak {peak}")
            
            if total_volume > 0:
                return weighted_density / total_volume
            else:
                print("Warning: No valid volume data found for weighted average")
                return None
        
        else:
            print(f"Warning: Unknown volume calculation method: {volume_calculation_method}")
            return None

    def _extract_raw_coordination_by_peaks(self, data_source, peak_selection):
        """
        Extract raw coordination data (non-normalized) filtered by peak selection.
        """
        
        if 'peak_analysis' not in data_source:
            print("Warning: No peak analysis data available for peak selection")
            return data_source.get('binding_events', data_source.get('coordination_numbers', None))
        
        peak_analysis = data_source['peak_analysis']
        
        # Determine peaks to include
        if peak_selection is None:
            peaks_to_use = list(peak_analysis.keys())
        else:
            peaks_to_use = [p for p in peak_selection if p in peak_analysis]
        
        if not peaks_to_use:
            print("Warning: No valid peaks found for analysis")
            return None
        
        print(f"   Using peaks: {peaks_to_use}")
        
        # Sum raw binding events across selected peaks
        total_coordination = None
        
        for peak in peaks_to_use:
            peak_data = peak_analysis[peak]
            
            if 'binding_events' in peak_data and peak_data['binding_events'] is not None:
                if total_coordination is None:
                    total_coordination = peak_data['binding_events'].copy()
                else:
                    total_coordination += peak_data['binding_events']
            else:
                print(f"Warning: No binding events data for peak {peak}")
        
        return total_coordination

    # =========================================================================
    # SPATIAL BINDING VISUALIZATION
    # =========================================================================
    
    def plot_spatial_binding_3d(self, spatial_results, view_angle=(30, 45),
                               colormap='hot', alpha=0.7, marker_size=None,
                               title=None, show_colorbar=True,
                               save_fig=False, filename='spatial_binding_3d.png',
                               dpi=300, figsize=(12, 10)):
        """
        Plot 3D visualization of per-atom contact frequencies
        
        Displays molecular structure with atoms colored by binding frequency.
        Useful for identifying specific hotspot atoms.
        
        Parameters
        ----------
        spatial_results : dict
            Results from spatial_binding_analysis() with method='per-atom' or 'both'
        view_angle : tuple
            (elevation, azimuth) viewing angle in degrees
        colormap : str
            Matplotlib colormap name
        alpha : float
            Transparency of points (0-1)
        marker_size : float or array, optional
            Marker sizes. If None, scales with contact frequency
        title : str, optional
            Plot title (auto-generated if None)
        show_colorbar : bool
            Whether to show colorbar
        save_fig : bool
            Whether to save figure
        filename : str
            Output filename
        dpi : int
            Resolution
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig, ax : matplotlib figure and 3D axes objects
        """
        
        # Check for per-atom data
        if 'contact_frequency' not in spatial_results:
            raise ValueError("spatial_results must contain 'contact_frequency' (use method='per-atom' or 'both')")
        
        # Extract data
        positions = spatial_results['atom_positions']
        frequencies = spatial_results['contact_frequency']
        atom_names = spatial_results['atom_names']
        
        # Normalize frequencies for color mapping
        freq_normalized = frequencies / frequencies.max() if frequencies.max() > 0 else frequencies
        
        # Auto-scale marker sizes if not provided
        if marker_size is None:
            # Base size + scale with frequency
            marker_size = 100 + 400 * freq_normalized
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Color map
        cmap = plt.get_cmap(colormap)
        colors = cmap(freq_normalized)
        
        # Scatter plot
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=frequencies, cmap=colormap, s=marker_size,
                           alpha=alpha, edgecolors='black', linewidths=0.5)
        
        # Labels
        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        ax.set_zlabel('Z (Å)', fontsize=12)
        
        # Title
        if title is None:
            ion_type = spatial_results.get('ion_type', 'Ion')
            title = f'{ion_type} Binding: Per-Atom Contact Frequency'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set viewing angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Colorbar
        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label('Contact Frequency', fontsize=12)
        
        # Add text annotation for hotspot
        max_idx = frequencies.argmax()
        ax.text(positions[max_idx, 0], positions[max_idx, 1], positions[max_idx, 2],
               f'  {atom_names[max_idx]}\n  ({frequencies[max_idx]:.0f})',
               fontsize=10, color='red', fontweight='bold')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✓ Figure saved: {filename}")
        
        return fig, ax
    
    def plot_spherical_binding_heatmap(self, spatial_results, projection='mollweide',
                                      colormap='hot', show_colorbar=True,
                                      title=None, interpolation='bilinear',
                                      save_fig=False, filename='spherical_binding.png',
                                      dpi=300, figsize=(14, 8)):
        """
        Plot 2D heatmap of spherical angular distribution
        
        Shows angular distribution of ion binding in theta (polar) and phi (azimuthal) angles.
        Uses Mollweide projection (like world map) to minimize distortion.
        
        Parameters
        ----------
        spatial_results : dict
            Results from spatial_binding_analysis() with method='spherical' or 'both'
        projection : str
            Map projection: 'mollweide', 'hammer', 'aitoff', or None (rectangular)
        colormap : str
            Matplotlib colormap name
        show_colorbar : bool
            Whether to show colorbar
        title : str, optional
            Plot title (auto-generated if None)
        interpolation : str
            Interpolation method for imshow
        save_fig : bool
            Whether to save figure
        filename : str
            Output filename
        dpi : int
            Resolution
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        """
        
        # Check for spherical data
        if 'angular_histogram' not in spatial_results:
            raise ValueError("spatial_results must contain 'angular_histogram' (use method='spherical' or 'both')")
        
        # Extract data
        histogram = spatial_results['angular_histogram']
        theta_centers = spatial_results['theta_centers']
        phi_centers = spatial_results['phi_centers']
        
        # Create meshgrid for plotting
        # Convert to longitude/latitude for projection
        # phi: 0 to 2π -> longitude: -π to π
        # theta: 0 to π -> latitude: -π/2 to π/2
        phi_lon = phi_centers - np.pi  # Center at 0
        theta_lat = np.pi/2 - theta_centers  # Convert to latitude
        
        Phi, Theta = np.meshgrid(phi_lon, theta_lat)
        
        # Create figure
        if projection:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection=projection)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        if projection:
            # For projection, use pcolormesh
            im = ax.pcolormesh(Phi, Theta, histogram, cmap=colormap, shading='auto')
            ax.grid(True, alpha=0.3)
        else:
            # For rectangular, use imshow
            extent = [phi_centers[0], phi_centers[-1], 
                     theta_centers[-1], theta_centers[0]]
            im = ax.imshow(histogram, cmap=colormap, aspect='auto',
                          extent=extent, interpolation=interpolation, origin='upper')
            ax.set_xlabel('Phi (Azimuthal Angle)', fontsize=12)
            ax.set_ylabel('Theta (Polar Angle)', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # Title
        if title is None:
            ion_type = spatial_results.get('ion_type', 'Ion')
            title = f'{ion_type} Binding: Angular Distribution'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Colorbar
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, pad=0.05, shrink=0.9)
            cbar.set_label('Contact Frequency', fontsize=12)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✓ Figure saved: {filename}")
        
        return fig, ax
    
    def plot_spatial_binding_combined(self, spatial_results, view_angle=(30, 45),
                                     save_fig=False, filename='spatial_binding_combined.png',
                                     dpi=300, figsize=(16, 7)):
        """
        Plot both 3D and spherical visualizations side-by-side
        
        Requires spatial_results with method='both'
        
        Parameters
        ----------
        spatial_results : dict
            Results from spatial_binding_analysis() with method='both'
        view_angle : tuple
            (elevation, azimuth) for 3D plot
        save_fig : bool
            Whether to save figure
        filename : str
            Output filename
        dpi : int
            Resolution
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig : matplotlib figure
        axes : tuple of (ax_3d, ax_sphere)
        """
        
        # Check for both data types
        if 'contact_frequency' not in spatial_results or 'angular_histogram' not in spatial_results:
            raise ValueError("spatial_results must contain both per-atom and spherical data (use method='both')")
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # 3D subplot
        ax_3d = fig.add_subplot(121, projection='3d')
        
        # Extract per-atom data
        positions = spatial_results['atom_positions']
        frequencies = spatial_results['contact_frequency']
        atom_names = spatial_results['atom_names']
        
        freq_normalized = frequencies / frequencies.max() if frequencies.max() > 0 else frequencies
        marker_size = 100 + 400 * freq_normalized
        
        scatter = ax_3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                              c=frequencies, cmap='hot', s=marker_size,
                              alpha=0.7, edgecolors='black', linewidths=0.5)
        
        ax_3d.set_xlabel('X (Å)', fontsize=11)
        ax_3d.set_ylabel('Y (Å)', fontsize=11)
        ax_3d.set_zlabel('Z (Å)', fontsize=11)
        ax_3d.set_title('Per-Atom Contact Frequency', fontsize=12, fontweight='bold')
        ax_3d.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Colorbar for 3D
        cbar_3d = plt.colorbar(scatter, ax=ax_3d, pad=0.1, shrink=0.7)
        cbar_3d.set_label('Contacts', fontsize=10)
        
        # Spherical subplot
        ax_sphere = fig.add_subplot(122, projection='mollweide')
        
        # Extract spherical data
        histogram = spatial_results['angular_histogram']
        theta_centers = spatial_results['theta_centers']
        phi_centers = spatial_results['phi_centers']
        
        phi_lon = phi_centers - np.pi
        theta_lat = np.pi/2 - theta_centers
        Phi, Theta = np.meshgrid(phi_lon, theta_lat)
        
        im = ax_sphere.pcolormesh(Phi, Theta, histogram, cmap='hot', shading='auto')
        ax_sphere.grid(True, alpha=0.3)
        ax_sphere.set_title('Angular Distribution', fontsize=12, fontweight='bold')
        
        # Colorbar for spherical
        cbar_sphere = plt.colorbar(im, ax=ax_sphere, pad=0.05, shrink=0.7)
        cbar_sphere.set_label('Contacts', fontsize=10)
        
        # Overall title
        ion_type = spatial_results.get('ion_type', 'Ion')
        fig.suptitle(f'{ion_type} Binding: Spatial Distribution Analysis',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✓ Figure saved: {filename}")
        
        return fig, (ax_3d, ax_sphere)
    
    def export_pymol_script(self, spatial_results, output_prefix='binding',
                           structure_file=None, color_scheme='blue_white_red',
                           scale_factor=1.0):
        """
        Export PyMOL script and data for visualizing spatial binding
        
        Generates:
        1. PyMOL script (.pml) with commands to color atoms by binding frequency
        2. Data file with per-atom contact frequencies
        
        Parameters
        ----------
        spatial_results : dict
            Results from spatial_binding_analysis() with method='per-atom' or 'both'
        output_prefix : str
            Prefix for output files (will create prefix.pml and prefix_data.txt)
        structure_file : str, optional
            Path to structure file (PDB/GRO) to load in PyMOL
            If None, assumes structure is already loaded
        color_scheme : str
            PyMOL color scheme: 'blue_white_red', 'rainbow', or 'heat'
        scale_factor : float
            Scale factor for sphere sizes (default 1.0)
        
        Returns
        -------
        script_file : str
            Path to generated PyMOL script
        data_file : str
            Path to data file
        
        Examples
        --------
        >>> plotter.export_pymol_script(results, 'K_binding', 
        ...                            structure_file='system.pdb')
        >>> # Then in PyMOL: @K_binding.pml
        """
        
        # Check for per-atom data
        if 'contact_frequency' not in spatial_results:
            raise ValueError("spatial_results must contain 'contact_frequency'")
        
        # Extract data
        atom_indices = spatial_results['atom_indices']
        frequencies = spatial_results['contact_frequency']
        atom_names = spatial_results['atom_names']
        
        # Normalize frequencies (0-1)
        freq_normalized = frequencies / frequencies.max() if frequencies.max() > 0 else frequencies
        
        # Write data file
        data_file = f"{output_prefix}_data.txt"
        with open(data_file, 'w') as f:
            f.write("# Atom binding frequency data\n")
            f.write("# Index\tName\tFrequency\tNormalized\n")
            for idx, name, freq, norm in zip(atom_indices, atom_names, frequencies, freq_normalized):
                f.write(f"{idx}\t{name}\t{freq:.1f}\t{norm:.4f}\n")
        
        # Generate PyMOL script
        script_file = f"{output_prefix}.pml"
        
        with open(script_file, 'w') as f:
            f.write("# PyMOL script for spatial binding visualization\n")
            f.write(f"# Generated for {spatial_results.get('ion_type', 'Ion')} binding\n\n")
            
            # Load structure if provided
            if structure_file:
                f.write(f"# Load structure\n")
                f.write(f"load {structure_file}, molecule\n\n")
            else:
                f.write("# Assuming structure is already loaded as 'molecule'\n\n")
            
            # Basic setup
            f.write("# Basic visualization setup\n")
            f.write("hide everything\n")
            f.write("show cartoon, molecule\n")
            f.write("show spheres, molecule\n")
            f.write("set sphere_scale, 0.3\n")
            f.write("set sphere_transparency, 0.2\n\n")
            
            # Color scheme setup
            f.write("# Color scheme for binding intensity\n")
            if color_scheme == 'blue_white_red':
                f.write("set_color low_bind, [0.0, 0.0, 1.0]\n")
                f.write("set_color mid_bind, [1.0, 1.0, 1.0]\n")
                f.write("set_color high_bind, [1.0, 0.0, 0.0]\n\n")
            elif color_scheme == 'rainbow':
                f.write("# Using rainbow spectrum\n\n")
            elif color_scheme == 'heat':
                f.write("# Using heat colors\n\n")
            
            # Color atoms by binding frequency
            f.write("# Color atoms by binding frequency\n")
            
            # Bin atoms into color groups
            n_colors = 10
            for i in range(n_colors):
                lower = i / n_colors
                upper = (i + 1) / n_colors
                
                # Find atoms in this range
                mask = (freq_normalized >= lower) & (freq_normalized < upper)
                indices_in_bin = atom_indices[mask]
                
                if len(indices_in_bin) == 0:
                    continue
                
                # Color value (0-1)
                color_val = (i + 0.5) / n_colors
                
                if color_scheme == 'blue_white_red':
                    if color_val < 0.5:
                        # Blue to white
                        r = color_val * 2
                        g = color_val * 2
                        b = 1.0
                    else:
                        # White to red
                        r = 1.0
                        g = 1.0 - (color_val - 0.5) * 2
                        b = 1.0 - (color_val - 0.5) * 2
                    
                    color_name = f"bind_color_{i}"
                    f.write(f"set_color {color_name}, [{r:.3f}, {g:.3f}, {b:.3f}]\n")
                    
                    # Select and color atoms
                    atom_str = "+".join([str(idx) for idx in indices_in_bin])
                    f.write(f"color {color_name}, index {atom_str}\n")
                
                elif color_scheme == 'rainbow':
                    f.write(f"color spectrum, index {'+'.join([str(idx) for idx in indices_in_bin])}\n")
            
            # Highlight hotspot
            max_idx = frequencies.argmax()
            hotspot_atom_idx = atom_indices[max_idx]
            f.write(f"\n# Highlight hotspot atom\n")
            f.write(f"select hotspot, index {hotspot_atom_idx}\n")
            f.write(f"show spheres, hotspot\n")
            f.write(f"set sphere_scale, {0.8 * scale_factor}, hotspot\n")
            f.write(f"color yellow, hotspot\n")
            f.write(f"label hotspot, '{atom_names[max_idx]} ({frequencies[max_idx]:.0f} contacts)'\n\n")
            
            # Final commands
            f.write("# Final view settings\n")
            f.write("set label_color, black\n")
            f.write("set label_size, 20\n")
            f.write("center molecule\n")
            f.write("zoom molecule\n")
            f.write(f"\n# Data saved to: {data_file}\n")
        
        print(f"✓ PyMOL script saved: {script_file}")
        print(f"✓ Data file saved: {data_file}")
        print(f"\nTo use in PyMOL:")
        print(f"  1. Open PyMOL")
        print(f"  2. Run: @{script_file}")
        
        return script_file, data_file
    
    def _build_coordinate_system_plotter(self, positions):
        """
        Build an orthonormal coordinate system from 3+ atom positions using Gram-Schmidt.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of shape (N, 3) with at least 3 atom positions
            
        Returns
        -------
        np.ndarray
            3x3 orthonormal matrix with basis vectors as COLUMNS (consistent with MolecularAnalysis)
            To transform lab → molecular: coord_system.T @ v
            To transform molecular → lab: coord_system @ v
        """
        if len(positions) < 3:
            return np.eye(3)
        
        # Use first 3 atoms to define the coordinate system
        p0, p1, p2 = positions[0], positions[1], positions[2]
        
        # First basis vector: normalized p0 -> p1
        v1 = p1 - p0
        norm1 = np.linalg.norm(v1)
        if norm1 < 1e-10:
            return np.eye(3)
        e1 = v1 / norm1
        
        # Second basis vector: p0 -> p2, orthogonalized against e1
        v2 = p2 - p0
        v2_perp = v2 - np.dot(v2, e1) * e1
        norm2 = np.linalg.norm(v2_perp)
        if norm2 < 1e-10:
            # p0, p1, p2 are collinear - create arbitrary perpendicular
            if abs(e1[0]) < 0.9:
                v2_perp = np.cross(e1, np.array([1, 0, 0]))
            else:
                v2_perp = np.cross(e1, np.array([0, 1, 0]))
            norm2 = np.linalg.norm(v2_perp)
        e2 = v2_perp / norm2
        
        # Third basis vector: cross product to complete right-handed system
        e3 = np.cross(e1, e2)
        
        # Stack as COLUMNS to match MolecularAnalysis._build_coordinate_system()
        # This ensures consistent transformation semantics:
        #   coord_system.T @ v  ->  lab to molecular frame
        #   coord_system @ v    ->  molecular to lab frame
        coord_system = np.column_stack([e1, e2, e3])
        
        return coord_system
    
    def _build_coordinate_system_plotter_pca(self, positions):
        """
        Build orthonormal coordinate system using PCA on multiple atom positions.
        
        Better for planar molecules where 3-atom Gram-Schmidt can give inconsistent orientations.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of shape (N, 3) with N>=3 atom positions
            
        Returns
        -------
        coord_system : np.ndarray or None
            3x3 orthonormal matrix with PC vectors as COLUMNS
        mean_position : np.ndarray or None
            Mean position of atoms (PCA center)
        """
        try:
            from sklearn.decomposition import PCA
            
            if len(positions) < 3:
                return None, None
            
            # Center the positions
            mean_pos = np.mean(positions, axis=0)
            centered = positions - mean_pos
            
            # Perform PCA
            pca = PCA(n_components=3)
            pca.fit(centered)
            
            # Get principal components
            pc1 = pca.components_[0]
            pc2 = pca.components_[1]
            pc3 = pca.components_[2]
            
            # Build coordinate system matrix (columns = basis vectors)
            coord_system = np.column_stack([pc1, pc2, pc3])
            
            # Ensure right-handed system
            if np.linalg.det(coord_system) < 0:
                coord_system[:, 2] *= -1
            
            return coord_system, mean_pos
            
        except (ImportError, Exception) as e:
            return None, None
 

    def plot_spatial_binding_interactive(self, spatial_results, structure_file=None,
                                        universe=None, density_threshold=0.02,
                                        distance_cutoff=None, distance_method='nearest_atom',
                                        sphere_size=0.4, sphere_opacity=0.3,
                                        stick_radius=0.15, ball_scale=0.3,
                                        width=800, height=600,
                                        show_output=True, max_spheres=500,
                                        # Enhanced region selection parameters
                                        plot_regions=None, shell_info_display=True,
                                        show_boundary_spheres=False, boundary_sphere_alpha=0.3,
                                        color_shade_style='modified', boundary_sphere_data_extent=False,
                                        # Aromatic ring visualization parameters
                                        show_aromatic_rings=True, aromatic_ring_color='gold',
                                        aromatic_ring_alpha=0.7, aromatic_ring_scale=0.8,
                                        aromatic_ring_thickness=0.15,
                                        # Triangulation and reconstruction parameters
                                        use_triangulation=True,
                                        reconstruction_method='atom',
                                        # Cluster comparison parameter
                                        density_scale='auto'):
        """
        Create interactive 3D visualization showing molecule + ion binding positions in space.
        
        This shows:
        1. Target molecule as ball-and-stick structure (the actual molecule)
        2. Spheres in 3D space at actual ion binding locations (not at molecule atoms)
        
        Supports shell-specific visualization when spatial_results comes from shell-boundary analysis.
        
        **Requires py3Dmol**: Install with `pip install py3Dmol`
        
        **For publication-quality images:**
        - Right-click on the visualization → "Save Image As..." (in Jupyter)
        - Use browser screenshot tools (Cmd+Shift+4 on Mac, Snipping Tool on Windows)
        - Or use the export_pymol_script() method for PyMOL rendering
        
        Parameters
        ----------
        spatial_results : dict
            Results from spatial_binding_analysis() with return_positions=True
            Must contain 'ion_positions_relative' key with stored ion coordinates
        structure_file : str, optional
            Path to PDB structure file for the target molecule.
            If None, structure will be generated from the universe trajectory.
        universe : MDAnalysis.Universe, required if structure_file is None
            Universe object for generating structure and proper coordinate alignment
        density_threshold : float
            Minimum density to show a binding position sphere (0-1). Default 0.02
        distance_cutoff : float or tuple, optional
            Distance filtering for custom regions. Options:
            - None: Use all regions from shell-boundary analysis (recommended)
            - float: Maximum distance (Å) to show spheres
            - (min, max): Distance range in Angstroms for custom filtering
            WARNING: Custom ranges must overlap with analyzed regions
        distance_method : str
            Method for distance calculation. Options:
            - 'nearest_atom': distance to nearest atom (default, better for elongated molecules)
            - 'com': distance to center of mass (better for spherical molecules)
        sphere_size : float
            Radius of ion position spheres in Angstroms. Default 0.4
        sphere_opacity : float
            Opacity of binding site spheres (0-1). Default 0.3
        stick_radius : float
            Radius of molecular structure sticks. Default 0.15
        ball_scale : float
            Radius scale for molecule atoms. Default 0.3
        width : int
            Viewer width in pixels
        height : int
            Viewer height in pixels
        show_output : bool
            Whether to print detailed visualization information
        max_spheres : int
            Maximum number of binding position spheres to display. Default 500
        plot_regions : list, optional
            Specific shell regions to visualize, e.g., ['P1', 'P3'] or ['P2']. 
            If None, shows all analyzed regions. Only works with shell-boundary analysis.
            Automatically extracts distance ranges for specified regions.
        shell_info_display : bool
            Whether to print shell information in output. Default True
        show_boundary_spheres : bool
            Whether to display transparent spheres marking the outer boundaries of each
            selected region. Useful for visualizing shell limits. Default False
        boundary_sphere_alpha : float
            Transparency level for boundary spheres (0.0-1.0). Higher values = more opaque.
            Default 0.3 (30% opacity)
        color_shade_style : str
            Color scheme to match RDF plots. Options:
            - 'modified': Enhanced colors (P1=lightcoral, P2=lightgreen, P3=lightyellow, P4=lightblue)
            - 'original': Original colors (P1=lightcoral, P2=lightblue, P3=lightgreen, P4=lightgoldenrodyellow)
            - 'vibrant': High-contrast colors (P1=#FF6B6B, P2=#4ECDC4, P3=#45B7D1, P4=#96CEB4)
            Default 'modified'
        boundary_sphere_data_extent : bool
            If True, boundary spheres show only angular regions where ion binding actually occurs.
            If False, shows complete theoretical boundary spheres. Default False
        use_triangulation : bool
            Whether to use triangulation data for precise geometric mapping. Default True.
            When True and triangulation_data is available, uses stored vectors for reconstruction.
        reconstruction_method : str
            Method for reconstructing ion positions from triangulation data. Options:
            - 'atom': Positions relative to specific target atoms (triangulation-based)
            - 'com': Positions relative to molecule center of mass
            - 'spherical': Spherical coordinate reconstruction relative to target atoms
            - 'molecular': Rotation-corrected positions using molecular frame with 3-atom Gram-Schmidt (uses stored distances)
            - 'molecular_pca': Rotation-corrected positions using PCA-based molecular frame (BEST for planar molecules like quinolone)
              Uses 5-10 reference atoms with PCA to get standardized principal axes. Helps ensure consistent
              coordinate frame orientation for flat aromatic rings. Must run analysis with molecular_frame_method='pca'.
            - 'molecular_atom': RECOMMENDED - Combines atom-based triangulation with rotation 
              correction for most accurate binding site visualization (uses stored distances)
            - 'molecular_spherical': Combines rotation correction with COM-based positioning.
              Like molecular_atom but positions relative to target COM instead of specific atoms.
              (uses stored distances)
            Default 'atom'
        density_scale : str or float
            Color scale for cluster comparison. Options:
            - 'auto': Automatically use max density value (default, best for single cluster)
            - 'global': Use fixed scale across all clusters (set manually after seeing max values)
            - float: Explicit maximum value for color scale (e.g., 0.5 for 0-0.5 contacts/frame)
            When comparing clusters, use the same value for all visualizations.
            Example: If cluster 0 max=0.45, use density_scale=0.5 for all clusters.
            Default 'auto'
        
        Returns
        -------
        view : py3Dmol.view
            Interactive 3D viewer object (displays automatically in Jupyter)
        
        Examples
        --------
        >>> # Run analysis with return_positions=True
        >>> spatial_K = analysis.spatial_binding_analysis('resname api', 'K',
        ...                                                cutoff=3.5, return_positions=True)
        >>> view = plotter.plot_spatial_binding_interactive(spatial_K, 'CIP.pdb')
        
        Notes
        -----
        - Molecule shown as ball-and-stick (normal molecular structure)
        - Spheres positioned in 3D space WHERE ions bind (not at molecule atoms)
        - Blue spheres: low-density binding regions
        - Red spheres: high-density binding regions
        - Click and drag to rotate, scroll to zoom, right-click to pan
        """
        
        try:
            import py3Dmol
        except ImportError:
            raise ImportError("py3Dmol not installed. Install with: pip install py3Dmol")
        
        # Import scipy KDTree for distance calculations (always needed)
        from scipy.spatial import cKDTree
        
        # Check for ion positions
        # Check for triangulation data (new preferred method) or fall back to legacy ion positions
        # Honor user's use_triangulation parameter, but require data to be available
        triangulation_available = 'triangulation_data' in spatial_results
        
        if use_triangulation and not triangulation_available:
            if show_output:
                print(f"⚠️  use_triangulation=True but no triangulation_data found, falling back to legacy positions")
            use_triangulation = False
        elif not use_triangulation and triangulation_available:
            if show_output:
                print(f"⚠️  Triangulation data available but use_triangulation=False, using legacy positions")
        
        if use_triangulation:
            triangulation_data = spatial_results['triangulation_data']
            if len(triangulation_data['frame_indices']) == 0:
                raise ValueError("No triangulation data found in spatial_results")
            
            # Filter triangulation data by region labels if plot_regions is specified
            if plot_regions is not None and 'region_labels' in triangulation_data:
                region_labels = triangulation_data['region_labels']
                
                # Create mask for requested regions
                region_mask = np.array([label in plot_regions for label in region_labels])
                n_original = len(region_labels)
                n_filtered = region_mask.sum()
                
                # Apply filter to all arrays in triangulation_data
                if n_filtered > 0:
                    filtered_triangulation = {}
                    for key, value in triangulation_data.items():
                        # Check if value is list or array with matching length
                        if isinstance(value, list):
                            if len(value) == n_original:
                                # Filter list-based data
                                filtered_triangulation[key] = [v for v, include in zip(value, region_mask) if include]
                            else:
                                # Keep lists of different lengths unchanged
                                filtered_triangulation[key] = value
                        elif isinstance(value, np.ndarray):
                            if value.ndim > 0 and len(value) == n_original:
                                # Filter numpy array data
                                filtered_triangulation[key] = value[region_mask]
                            else:
                                # Keep arrays of different lengths or scalars unchanged
                                filtered_triangulation[key] = value
                        else:
                            # Keep non-array data unchanged (booleans, None, etc.)
                            filtered_triangulation[key] = value
                    triangulation_data = filtered_triangulation
            
            if show_output:
                print(f"✓ Using triangulation data for precise geometric mapping")
                print(f"  Ion-atom pairs: {len(triangulation_data['frame_indices'])}")
        elif 'ion_positions_relative' not in spatial_results:
            raise ValueError(
                "spatial_results must contain 'triangulation_data' or 'ion_positions_relative'.\n"
                "Re-run spatial_binding_analysis() with return_positions=True"
            )
        
        # Validate plot_regions and distance_cutoff parameters
        # Pass boundaries_ions_refined if available in the calling environment or spatial_results
        try:
            import inspect
            frame = inspect.currentframe().f_back
            boundaries_ions_refined = frame.f_globals.get('boundaries_ions_refined', {})
            
            # Also check for 'boundaries_with_ion' or 'boundaries_Na' etc. in caller's namespace
            if not boundaries_ions_refined:
                for var_name in ['boundaries_with_ion', 'boundaries_Na', 'boundaries_K', 'boundaries_CA']:
                    var_value = frame.f_globals.get(var_name, None)
                    if var_value:
                        boundaries_ions_refined = var_value
                        break
        except:
            boundaries_ions_refined = {}
        
        # If still not found, try to get from spatial_results
        if not boundaries_ions_refined and 'spatial_binding_boundaries' in spatial_results:
            boundaries_ions_refined = spatial_results['spatial_binding_boundaries']
        
        self._validate_region_parameters(spatial_results, plot_regions, distance_cutoff, show_output, boundaries_ions_refined)
        
        # Determine effective distance filtering parameters
        effective_distance_cutoff, effective_regions = self._determine_distance_filtering(
            spatial_results, plot_regions, distance_cutoff, show_output, boundaries_ions_refined
        )
        
        # Process ion positions based on available data format
        if use_triangulation:
            # Use triangulation data for precise reconstruction
            # We'll reconstruct ion positions later using PDB structure coordinates
            ion_positions = None  # Will be reconstructed below
            triangulation_count = len(triangulation_data['frame_indices'])
        else:
            # Legacy approach: use stored relative positions
            ion_positions_list = spatial_results['ion_positions_relative']
            if len(ion_positions_list) == 0:
                raise ValueError("No ion positions found in spatial_results")
            
            # Flatten list of arrays into single array
            ion_positions = np.vstack(ion_positions_list)
            triangulation_count = len(ion_positions)
        
        if show_output:
            print(f"\n{'='*60}")
            print(f"Spatial Binding Visualization")
            print(f"{'='*60}")
            if use_triangulation:
                print(f"Triangulation data: {triangulation_count} ion-atom pairs")
                print(f"✓ Precise geometric mapping with motion compensation")
            else:
                print(f"Total ion positions recorded: {triangulation_count}")
                print(f"⚠️  Using legacy relative positions (no motion compensation)")
            
            # Display shell information if available
            if shell_info_display and spatial_results.get('use_shell_boundaries', False):
                print(f"\nShell-Boundary Analysis:")
                print(f"  Ion type: {spatial_results.get('ion_type', 'Unknown')}")
                print(f"  Target-ion key: {spatial_results.get('target_ion_key', 'Unknown')}")
                
                analyzed_shells = spatial_results.get('selected_shells', [])
                if analyzed_shells:
                    print(f"  Shells analyzed: {', '.join(analyzed_shells)}")
                    
                    if effective_regions and effective_regions != analyzed_shells:
                        print(f"  Shells displayed: {', '.join(effective_regions)}")
                    else:
                        print(f"  Shells displayed: All analyzed shells")
                else:
                    print(f"  No shell information available")
            elif spatial_results.get('use_shell_boundaries', False) == False:
                print(f"\nDistance-Cutoff Analysis:")
                print(f"  Cutoff: {spatial_results.get('cutoff', 'Unknown')} Å")
        
        # Handle structure source - either from file or universe trajectory
        if structure_file is not None:
            # Use provided PDB file
            with open(structure_file, 'r') as f:
                pdb_string = f.read()
            
            # Parse PDB to get molecule's center of mass
            atom_coords = []
            for line in pdb_string.split('\n'):
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        atom_coords.append([x, y, z])
                    except:
                        continue
            
            if len(atom_coords) == 0:
                raise ValueError("No atoms found in PDB structure")
            
            # Calculate COM of molecule in PDB
            molecule_com = np.mean(atom_coords, axis=0)
            coordinate_source = "PDB file"
            
            # For PDB files, assume ion positions need to be shifted to align
            # Ion positions are relative to trajectory coordinates
            ion_positions_shifted = ion_positions + molecule_com
            
        else:
            # Generate structure from trajectory universe
            if universe is None:
                raise ValueError("Universe is required when structure_file is None")
            
            # Use current frame (allows cluster-specific structures)
            # Note: Caller can position universe to desired frame before calling this method
            current_frame = universe.trajectory.frame
            molecule_atoms = universe.select_atoms("resname api")  # CIP molecule
            
            if len(molecule_atoms) == 0:
                raise ValueError("No molecule atoms found with 'resname api' selection")
            
            # Generate PDB string from trajectory coordinates with proper formatting
            pdb_lines = []
            pdb_lines.append("HEADER    Generated from trajectory")
            pdb_lines.append(f"REMARK    Structure from trajectory frame {current_frame}")
            pdb_lines.append("REMARK    Ciprofloxacin molecule (CIP)")
            
            for i, atom in enumerate(molecule_atoms, 1):
                pos = atom.position
                
                # Get proper element symbol
                if hasattr(atom, 'element') and atom.element:
                    element = atom.element.upper()
                else:
                    # Guess element from atom name
                    atom_name = atom.name.strip()
                    if atom_name.startswith('C'):
                        element = 'C'
                    elif atom_name.startswith('N'):
                        element = 'N'
                    elif atom_name.startswith('O'):
                        element = 'O'
                    elif atom_name.startswith('H'):
                        element = 'H'
                    elif atom_name.startswith('F'):
                        element = 'F'
                    elif atom_name.startswith('S'):
                        element = 'S'
                    elif atom_name.startswith('P'):
                        element = 'P'
                    else:
                        element = atom_name[0].upper()
                
                # Proper PDB format with correct spacing
                pdb_line = (f"ATOM  {i:5d}  {atom.name:<4s}{atom.resname:>3s} A{atom.resid:4d}    "
                          f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00 20.00          {element:>2s}  ")
                pdb_lines.append(pdb_line)
            
            # Add connectivity information if available
            pdb_lines.append("END")
            pdb_string = '\n'.join(pdb_lines)
            
            # Get molecule COM from trajectory
            molecule_com = molecule_atoms.center_of_mass()
            coordinate_source = "trajectory coordinates"
            
            # For trajectory-based coordinates, ion positions are already in correct reference frame
            # They were stored relative to target atoms in the same coordinate system
            # We need to reconstruct absolute positions by adding target atom positions back
            
            # Get target atoms from trajectory (first frame)
            if 'atom_indices' in spatial_results:
                target_indices = spatial_results['atom_indices']
                target_atoms = universe.atoms[target_indices]
                target_positions = target_atoms.positions
                
            # Reconstruct ion positions using available data format
            if use_triangulation:
                # TRIANGULATION METHOD: Reconstruct using vector relationships
                if 'atom_indices' not in spatial_results:
                    raise ValueError("Triangulation data requires atom_indices for reconstruction")
                
                # Get PBC-corrected vectors if available
                if 'atom_to_ion_vectors' in triangulation_data:
                    atom_to_ion_vectors = np.array(triangulation_data['atom_to_ion_vectors'])
                else:
                    raise ValueError("No atom_to_ion_vectors in triangulation data")
                
                # Get target atom positions from current frame for reference
                # (allows cluster-specific structures when caller positions universe appropriately)
                current_frame = universe.trajectory.frame
                target_atoms_current = universe.atoms[target_indices]
                target_positions_current = target_atoms_current.positions
                target_com_current = target_atoms_current.center_of_mass()
                
                # ========== RECONSTRUCTION METHOD DISPATCH ==========
                
                if reconstruction_method == 'atom':
                    # ATOM METHOD: Position relative to specific target atoms at current frame
                    # Uses stored target_atom_positions for consistency
                    
                    # Use stored target atom positions if available, otherwise use frame 0 positions
                    if 'target_atom_positions' in triangulation_data:
                        stored_target_positions = np.array(triangulation_data['target_atom_positions'])
                        ion_positions_shifted = stored_target_positions + atom_to_ion_vectors
                    else:
                        # Fallback: use target atom positions from frame 0
                        target_atom_indices = triangulation_data.get('target_atom_indices', 
                                                                      [0] * len(atom_to_ion_vectors))
                        ion_positions_shifted = []
                        for i, vec in enumerate(atom_to_ion_vectors):
                            atom_idx = target_atom_indices[i] if i < len(target_atom_indices) else 0
                            if atom_idx < len(target_positions_current):
                                ion_positions_shifted.append(target_positions_current[atom_idx] + vec)
                            else:
                                ion_positions_shifted.append(target_com_current + vec)
                        ion_positions_shifted = np.array(ion_positions_shifted)
                
                elif reconstruction_method == 'com':
                    # COM METHOD: Position relative to target COM at current frame
                    
                    if 'COM_to_ion_vectors' in triangulation_data:
                        com_to_ion_vectors = np.array(triangulation_data['COM_to_ion_vectors'])
                        ion_positions_shifted = target_com_current + com_to_ion_vectors
                    else:
                        # Fallback: compute COM vectors from atom vectors
                        stored_target_positions = np.array(triangulation_data['target_atom_positions'])
                        ion_absolute = stored_target_positions + atom_to_ion_vectors
                        ion_positions_shifted = ion_absolute  # Already in absolute coordinates
                
                elif reconstruction_method == 'spherical':
                    # SPHERICAL METHOD: Convert to spherical coords and reconstruct
                    
                    # Use stored target atom positions if available, otherwise use frame 0 positions
                    if 'target_atom_positions' in triangulation_data:
                        stored_target_positions = np.array(triangulation_data['target_atom_positions'])
                        ion_positions_shifted = stored_target_positions + atom_to_ion_vectors
                    else:
                        # Fallback: use target atom positions from frame 0
                        # Each entry maps to a target atom via target_atom_indices
                        target_atom_indices = triangulation_data.get('target_atom_indices', 
                                                                      [0] * len(atom_to_ion_vectors))
                        ion_positions_shifted = []
                        for i, vec in enumerate(atom_to_ion_vectors):
                            atom_idx = target_atom_indices[i] if i < len(target_atom_indices) else 0
                            if atom_idx < len(target_positions_current):
                                ion_positions_shifted.append(target_positions_current[atom_idx] + vec)
                            else:
                                ion_positions_shifted.append(target_com_current + vec)
                        ion_positions_shifted = np.array(ion_positions_shifted)
                
                elif reconstruction_method == 'molecular':
                    # MOLECULAR METHOD: Use molecular reference frame with rotation correction (3-atom Gram-Schmidt)
                    
                    # CRITICAL: Must use the SAME reference atoms that were used during analysis!
                    # The analysis stores reference_frame_established with the selected reference atoms
                    if 'reference_frame_established' in triangulation_data and triangulation_data['reference_frame_established'] is not None:
                        ref_frame = np.array(triangulation_data['reference_frame_established'])
                        reference_coords = self._build_coordinate_system_plotter(ref_frame)
                    elif len(target_positions_current) >= 3:
                        reference_coords = self._build_coordinate_system_plotter(target_positions_current)
                        if show_output:
                            print(f"⚠️  No stored reference frame, using first 3 of {len(target_positions_current)} target atoms")
                    else:
                        reference_coords = np.eye(3)
                        if show_output:
                            print(f"⚠️  Not enough atoms for proper molecular frame, using identity")
                    
                    # Get stored molecular frame vectors if available
                    if 'molecular_frame_vectors' in triangulation_data:
                        molecular_vectors = np.array(triangulation_data['molecular_frame_vectors'])
                        
                        # Reconstruct: transform from molecular frame back to lab frame
                        # molecular_vector was computed as: current_coords.T @ com_to_ion_vec (Lab → Molecular)
                        # To get back to lab frame: reference_coords @ molecular_vector (Molecular → Reference Lab)
                        ion_positions_shifted = np.array([
                            reference_coords @ vec + target_com_current 
                            for vec in molecular_vectors
                        ])
                
                elif reconstruction_method == 'molecular_pca':
                    # MOLECULAR PCA METHOD: Use PCA-based molecular reference frame (better for planar molecules)
                    
                    # CRITICAL: Must use the SAME reference atoms that were used during analysis!
                    if 'reference_frame_established' in triangulation_data and triangulation_data['reference_frame_established'] is not None:
                        ref_frame = np.array(triangulation_data['reference_frame_established'])
                        
                        # Check if PCA was used (more than 3 reference atoms)
                        if len(ref_frame) > 3:
                            reference_coords, ref_mean = self._build_coordinate_system_plotter_pca(ref_frame)
                            if reference_coords is None:
                                if show_output:
                                    print(f"⚠️  PCA coordinate system failed, falling back to Gram-Schmidt")
                                reference_coords = self._build_coordinate_system_plotter(ref_frame[:3])
                            else:
                                if show_output:
                                    print(f"✓  Using PCA coordinate system with {len(ref_frame)} reference atoms")
                        else:
                            # Fallback to Gram-Schmidt if only 3 atoms
                            reference_coords = self._build_coordinate_system_plotter(ref_frame)
                            if show_output:
                                print(f"⚠️  Only 3 reference atoms found, using Gram-Schmidt instead of PCA")
                    elif len(target_positions_current) >= 3:
                        # No stored reference frame - try PCA with current atoms
                        reference_coords, _ = self._build_coordinate_system_plotter_pca(target_positions_current)
                        if reference_coords is None:
                            reference_coords = self._build_coordinate_system_plotter(target_positions_current)
                        if show_output:
                            print(f"⚠️  No stored reference frame, using current target atoms")
                    else:
                        reference_coords = np.eye(3)
                        if show_output:
                            print(f"⚠️  Not enough atoms for proper molecular frame, using identity")
                    
                    # Get stored molecular frame vectors if available
                    if 'molecular_frame_vectors' in triangulation_data:
                        molecular_vectors = np.array(triangulation_data['molecular_frame_vectors'])
                        
                        # Reconstruct: transform from molecular frame back to lab frame
                        ion_positions_shifted = np.array([
                            reference_coords @ vec + target_com_current 
                            for vec in molecular_vectors
                        ])
                        
                        molecular_count = len(ion_positions_shifted)
                        com_fallback_count = 0
                    else:
                        # Fallback to COM vectors
                        if 'COM_to_ion_vectors' in triangulation_data:
                            com_to_ion_vectors = np.array(triangulation_data['COM_to_ion_vectors'])
                            ion_positions_shifted = target_com_current + com_to_ion_vectors
                            molecular_count = 0
                            com_fallback_count = len(ion_positions_shifted)
                        else:
                            stored_target_positions = np.array(triangulation_data['target_atom_positions'])
                            ion_positions_shifted = stored_target_positions + atom_to_ion_vectors
                            molecular_count = 0
                            com_fallback_count = len(ion_positions_shifted)
                
                elif reconstruction_method == 'molecular_atom':
                    # MOLECULAR_ATOM METHOD: Combines atom-based triangulation with rotation correction
                    # This is the RECOMMENDED method for accurate binding site visualization
                    
                    # CRITICAL: Must use the SAME reference atoms that were used during analysis!
                    if 'reference_frame_established' in triangulation_data and triangulation_data['reference_frame_established'] is not None:
                        ref_frame = np.array(triangulation_data['reference_frame_established'])
                        reference_coords = self._build_coordinate_system_plotter(ref_frame)
                    elif len(target_positions_current) >= 3:
                        reference_coords = self._build_coordinate_system_plotter(target_positions_current)
                    else:
                        reference_coords = np.eye(3)
                    
                    # Get reference atom positions if available (for building per-frame coordinate systems)
                    if 'reference_atom_positions' in triangulation_data:
                        ref_atom_positions = triangulation_data['reference_atom_positions']
                    else:
                        ref_atom_positions = [None] * len(atom_to_ion_vectors)
                    
                    # Get target atom indices for each entry
                    target_atom_indices = triangulation_data.get('target_atom_indices', 
                                                                  [0] * len(atom_to_ion_vectors))
                    
                    # Reconstruct positions with rotation correction
                    ion_positions_shifted = []
                    rotation_corrected_count = 0
                    fallback_count = 0
                    
                    for i in range(len(atom_to_ion_vectors)):
                        atom_to_ion_vec = atom_to_ion_vectors[i]
                        target_atom_idx = target_atom_indices[i] if i < len(target_atom_indices) else 0
                        
                        # Get target atom position at frame 0
                        if target_atom_idx < len(target_positions_current):
                            target_atom_pos_frame0 = target_positions_current[target_atom_idx]
                        else:
                            target_atom_pos_frame0 = target_com_current
                        
                        # Try to build current frame coordinate system
                        ref_pos = ref_atom_positions[i] if i < len(ref_atom_positions) else None
                        
                        if ref_pos is not None and len(ref_pos) >= 3:
                            try:
                                current_coords = self._build_coordinate_system_plotter(np.array(ref_pos))
                                
                                # Transform atom_to_ion_vec from current frame to reference frame:
                                # With column-stacked basis vectors:
                                # 1. current_coords.T @ v transforms v from lab to molecular frame
                                # 2. reference_coords @ v_mol transforms from molecular to reference lab frame
                                # Combined: v_ref = reference_coords @ current_coords.T @ v
                                v_unrotated = reference_coords @ current_coords.T @ atom_to_ion_vec
                                
                                # Reconstruct position relative to target atom at frame 0
                                reconstructed_pos = target_atom_pos_frame0 + v_unrotated
                                rotation_corrected_count += 1
                            except:
                                # Fallback: use vector directly without rotation correction
                                reconstructed_pos = target_atom_pos_frame0 + atom_to_ion_vec
                                fallback_count += 1
                        else:
                            # No reference positions - use vector directly
                            reconstructed_pos = target_atom_pos_frame0 + atom_to_ion_vec
                            fallback_count += 1
                        
                        ion_positions_shifted.append(reconstructed_pos)
                    
                    ion_positions_shifted = np.array(ion_positions_shifted)
                
                elif reconstruction_method == 'molecular_spherical':
                    # MOLECULAR_SPHERICAL METHOD: Combines rotation correction with COM-based positioning
                    # Like molecular_atom but positions relative to target COM instead of specific atoms
                    
                    # CRITICAL: Must use the SAME reference atoms that were used during analysis!
                    if 'reference_frame_established' in triangulation_data and triangulation_data['reference_frame_established'] is not None:
                        ref_frame = np.array(triangulation_data['reference_frame_established'])
                        reference_coords = self._build_coordinate_system_plotter(ref_frame)
                    elif len(target_positions_current) >= 3:
                        reference_coords = self._build_coordinate_system_plotter(target_positions_current)
                    else:
                        reference_coords = np.eye(3)
                    
                    # Get reference atom positions if available (for building per-frame coordinate systems)
                    if 'reference_atom_positions' in triangulation_data:
                        ref_atom_positions = triangulation_data['reference_atom_positions']
                    else:
                        ref_atom_positions = [None] * len(atom_to_ion_vectors)
                    
                    # Get stored target atom positions to compute COM offset
                    if 'target_atom_positions' in triangulation_data:
                        stored_target_positions = np.array(triangulation_data['target_atom_positions'])
                    else:
                        stored_target_positions = None
                    
                    # Get target atom indices for each entry
                    target_atom_indices = triangulation_data.get('target_atom_indices', 
                                                                  [0] * len(atom_to_ion_vectors))
                    
                    # Reconstruct positions with rotation correction, placing relative to COM
                    ion_positions_shifted = []
                    rotation_corrected_count = 0
                    fallback_count = 0
                    
                    for i in range(len(atom_to_ion_vectors)):
                        atom_to_ion_vec = atom_to_ion_vectors[i]
                        
                        # Try to build current frame coordinate system
                        ref_pos = ref_atom_positions[i] if i < len(ref_atom_positions) else None
                        
                        if ref_pos is not None and len(ref_pos) >= 3:
                            try:
                                current_coords = self._build_coordinate_system_plotter(np.array(ref_pos))
                                
                                # Transform atom_to_ion_vec from current frame to reference frame:
                                # With column-stacked basis vectors:
                                # 1. current_coords.T @ v transforms v from lab to molecular frame
                                # 2. reference_coords @ v_mol transforms from molecular to reference lab frame
                                # Combined: v_ref = reference_coords @ current_coords.T @ v
                                v_unrotated = reference_coords @ current_coords.T @ atom_to_ion_vec
                                
                                # For spherical: we need to adjust vector to be relative to COM
                                # Original vector was from specific target atom to ion
                                # We want vector from COM to ion
                                if stored_target_positions is not None:
                                    # Get the stored target atom position for this entry
                                    stored_atom_pos = stored_target_positions[i]
                                    # Compute stored COM for this frame's reference atoms
                                    ref_pos_array = np.array(ref_pos)
                                    stored_com = ref_pos_array.mean(axis=0)
                                    # Vector from stored COM to stored atom
                                    com_to_atom = stored_atom_pos - stored_com
                                    # Also rotate this offset vector
                                    com_to_atom_rotated = reference_coords @ current_coords.T @ com_to_atom
                                    # Final vector from frame0 COM to ion
                                    v_from_com = v_unrotated + com_to_atom_rotated
                                else:
                                    # Fallback: use the vector as-is relative to COM
                                    v_from_com = v_unrotated
                                
                                # Reconstruct position relative to target COM at current frame
                                reconstructed_pos = target_com_current + v_from_com
                                rotation_corrected_count += 1
                            except:
                                # Fallback: use vector directly without rotation correction
                                target_atom_idx = target_atom_indices[i] if i < len(target_atom_indices) else 0
                                if target_atom_idx < len(target_positions_current):
                                    target_atom_pos_frame0 = target_positions_current[target_atom_idx]
                                else:
                                    target_atom_pos_frame0 = target_com_current
                                reconstructed_pos = target_atom_pos_frame0 + atom_to_ion_vec
                                fallback_count += 1
                        else:
                            # No reference positions - use vector directly relative to COM
                            reconstructed_pos = target_com_current + atom_to_ion_vec
                            fallback_count += 1
                        
                        ion_positions_shifted.append(reconstructed_pos)
                    
                    ion_positions_shifted = np.array(ion_positions_shifted)
                
                else:
                    raise ValueError(f"Unknown reconstruction_method: {reconstruction_method}. "
                                   f"Use 'atom', 'com', 'spherical', 'molecular', 'molecular_pca', 'molecular_atom', or 'molecular_spherical'")
                    
            else:
                # LEGACY METHOD: Use stored relative positions
                if 'atom_indices' in spatial_results:
                    target_indices = spatial_results['atom_indices']
                    target_atoms = universe.atoms[target_indices]
                    target_positions = target_atoms.positions
                    
                    if show_output:
                        print(f"⚠️  Using legacy atom-based position reconstruction")
                        print(f"✓ Target atoms: {target_atoms.names} at positions:")
                        for i, (name, pos) in enumerate(zip(target_atoms.names, target_positions)):
                            print(f"     {name}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                    
                    # Use center of target atoms (consistent with analysis storage)
                    if len(target_positions) > 1:
                        target_center = target_atoms.center_of_mass()  # Use COM like in analysis
                        reference_label = "center of target atoms"
                    else:
                        # Single target atom
                        target_center = target_positions[0]
                        reference_label = f"target atom {target_atoms.names[0]}"
                    
                    # CRITICAL FIX: Use stored reference COM from spatial analysis if available
                    # This ensures ion positions align with boundary spheres
                    if 'reference_com' in spatial_results:
                        stored_reference_com = spatial_results['reference_com']
                        ion_positions_shifted = ion_positions + stored_reference_com
                        reference_label = f"stored analysis COM"
                        if show_output:
                            print(f"✓ Using stored reference COM: [{stored_reference_com[0]:.2f}, {stored_reference_com[1]:.2f}, {stored_reference_com[2]:.2f}]")
                            print(f"✓ This ensures ion positions align with boundary spheres")
                    else:
                        # Fallback to PDB structure COM (may cause coordinate mismatch)
                        ion_positions_shifted = ion_positions + target_center
                        if show_output:
                            print(f"⚠️  No stored reference COM found, using PDB structure COM")
                            print(f"   This may cause misalignment between boundary spheres and ion positions")
                        
                else:
                    # Fallback: assume ion positions are relative to molecule COM
                    ion_positions_shifted = ion_positions + molecule_com
                    if show_output:
                        print(f"⚠️  No target atom info found, using molecule COM as reference")
            
            # Create atom_coords array for distance calculations (from trajectory coordinates)
            atom_coords = molecule_atoms.positions.tolist()
        
        if show_output:
            print(f"✓ Molecule structure from: {coordinate_source}")
            print(f"✓ Molecule COM: [{molecule_com[0]:.2f}, {molecule_com[1]:.2f}, {molecule_com[2]:.2f}]")
            print(f"✓ Positioned {len(ion_positions_shifted) if ion_positions_shifted is not None else 0} ion positions in 3D space")
        
        # Create viewer
        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_string, 'pdb')
        
        # Style molecule with proper atom coloring and bonding
        # Use element-based coloring for better molecular representation
        view.setStyle({}, {
            'stick': {
                'radius': stick_radius,
                'colorscheme': 'default'  # Use default element colors
            },
            'sphere': {
                'radius': ball_scale,
                'colorscheme': 'default'  # Use default element colors
            }
        })
        
        # Alternative: if default doesn't work well, use specific element colors
        # view.setStyle({'elem': 'C'}, {'stick': {'radius': stick_radius, 'color': 'gray'},
        #                               'sphere': {'radius': ball_scale, 'color': 'gray'}})
        # view.setStyle({'elem': 'N'}, {'stick': {'radius': stick_radius, 'color': 'blue'},
        #                               'sphere': {'radius': ball_scale, 'color': 'blue'}})
        # view.setStyle({'elem': 'O'}, {'stick': {'radius': stick_radius, 'color': 'red'},
        #                               'sphere': {'radius': ball_scale, 'color': 'red'}})
        # view.setStyle({'elem': 'F'}, {'stick': {'radius': stick_radius, 'color': 'green'},
        #                               'sphere': {'radius': ball_scale, 'color': 'green'}})
        
        if show_output:
            print(f"✓ Molecule rendered as ball-and-stick with proper element coloring")
        
        # Add aromatic rings if requested
        if show_aromatic_rings and universe is not None:
            aromatic_rings_added = self._add_aromatic_rings(view, universe, aromatic_ring_color, 
                                                           aromatic_ring_alpha, aromatic_ring_scale, 
                                                           aromatic_ring_thickness, show_output)
            if aromatic_rings_added > 0 and show_output:
                print(f"✓ Added {aromatic_rings_added} aromatic ring indicator(s)")
        
        # Get frame normalization info for density calculations
        n_frames_analyzed = None
        use_absolute_coloring = False
        max_density_scale = None
        
        if 'cluster_metadata' in spatial_results:
            n_frames_analyzed = spatial_results['cluster_metadata'].get('n_cluster_frames')
            if n_frames_analyzed and n_frames_analyzed > 0:
                use_absolute_coloring = True
                
                # Determine color scale - for 'auto', calculate from ALL positions (before region filtering)
                if density_scale == 'auto':
                    # Calculate densities from all positions to get global scale
                    tree_all = cKDTree(ion_positions_shifted)
                    search_radius = 2.0
                    densities_all = np.array([len(tree_all.query_ball_point(pos, search_radius)) 
                                             for pos in ion_positions_shifted])
                    densities_all_normalized = densities_all / n_frames_analyzed
                    max_density_scale = densities_all_normalized.max() * 1.1  # Add 10% headroom
                elif isinstance(density_scale, (int, float)):
                    # User-specified scale (consistent across all regions)
                    max_density_scale = float(density_scale)
                else:  # 'global' or other
                    # Default fallback
                    max_density_scale = 0.5
        
        # IMPORTANT: Distance calculation strategy depends on reconstruction method:
        #
        # For 'atom', 'com', and 'spherical' methods: 
        #   Positions are reconstructed consistently with frame 0 atom positions.
        #   Computing distances from reconstructed positions to frame 0 atoms is CORRECT.
        #   Use KDTree-based distance calculation (original behavior).
        #
        # For 'molecular', 'molecular_atom', and 'molecular_spherical' methods:
        #   Positions are rotated to reference frame orientation, but target atoms are NOT.
        #   Computing distances from reconstructed positions gives WRONG distances.
        #   Must use stored distances_to_target from original frames.
        #
        use_stored_distances = (
            reconstruction_method in ('molecular', 'molecular_atom', 'molecular_spherical') and
            use_triangulation and 
            'distances_to_target' in triangulation_data and
            len(triangulation_data['distances_to_target']) == len(ion_positions_shifted)
        )
        
        if use_stored_distances:
            # MOLECULAR/MOLECULAR_ATOM: Use pre-computed distances from original frames
            # This avoids rotation artifacts where reconstructed positions don't match target atoms
            raw_distances = np.array(triangulation_data['distances_to_target'])
            
            # Check if we have multiple target atoms (entries per ion)
            frame_indices = np.array(triangulation_data['frame_indices'])
            ion_indices = np.array(triangulation_data['ion_indices'])
            
            # Create unique (frame, ion) identifiers
            unique_keys = list(zip(frame_indices, ion_indices))
            
            # Find unique ion observations
            seen_keys = {}
            for i, key in enumerate(unique_keys):
                if key not in seen_keys:
                    seen_keys[key] = []
                seen_keys[key].append(i)
            
            n_unique_ions = len(seen_keys)
            n_total_entries = len(raw_distances)
            entries_per_ion = n_total_entries / n_unique_ions if n_unique_ions > 0 else 1
            
            if entries_per_ion > 1.5:  # Multi-atom target
                # Compute minimum distance for each unique ion across all its target atoms
                distances = np.zeros(n_total_entries)
                for key, indices in seen_keys.items():
                    min_dist = raw_distances[indices].min()
                    for idx in indices:
                        distances[idx] = min_dist
                
                distance_label = "minimum stored distance to target atoms"
            else:
                # Single target atom - use distances directly
                distances = raw_distances
                distance_label = "stored original distance to target"
        elif distance_method == 'nearest_atom':
            # Distance to nearest molecule atom (better for elongated molecules)
            atom_coords_array = np.array(atom_coords)
            mol_tree = cKDTree(atom_coords_array)
            distances, _ = mol_tree.query(ion_positions_shifted)
            distance_label = "nearest atom"
        elif distance_method == 'com':
            # Distance to center of mass (better for spherical molecules)
            distances = np.linalg.norm(ion_positions_shifted - molecule_com, axis=1)
            distance_label = "COM"
        else:
            raise ValueError(f"distance_method must be 'nearest_atom' or 'com', got '{distance_method}'")
        
        # Apply distance filtering based on effective parameters
        # Check if we need discrete region filtering
        use_discrete_regions = (plot_regions is not None and 
                              spatial_results.get('use_shell_boundaries', False) and 
                              boundaries_ions_refined)
        
        # Extract region boundaries for later use (boundary sphere rendering)
        target_species_key = spatial_results.get('target_species_key')
        region_boundaries = boundaries_ions_refined.get(target_species_key, {}) if boundaries_ions_refined else {}
        
        if use_discrete_regions:
            # Discrete region filtering - apply each region separately
            distance_mask = np.zeros_like(distances, dtype=bool)
            region_info_list = []
            cutoff_description_parts = []
            
            for region in plot_regions:
                if region in region_boundaries:
                    r_min, r_max = region_boundaries[region]
                    region_mask = (distances >= r_min) & (distances <= r_max)
                    distance_mask |= region_mask  # Union of all regions
                    region_info_list.append(f"{region}({r_min:.1f}-{r_max:.1f})")
                    cutoff_description_parts.append(f"{r_min:.1f}-{r_max:.1f}")
            
            cutoff_description = " OR ".join(cutoff_description_parts) + " Å (discrete)"
        
        elif isinstance(effective_distance_cutoff, (tuple, list)) and len(effective_distance_cutoff) == 2:
            # Distance range filtering (continuous)
            min_dist, max_dist = effective_distance_cutoff
            distance_mask = (distances >= min_dist) & (distances <= max_dist)
            cutoff_description = f"{min_dist:.1f}-{max_dist:.1f} Å"
        else:
            # Single cutoff filtering
            distance_mask = distances <= effective_distance_cutoff
            cutoff_description = f"≤{effective_distance_cutoff:.1f} Å"
        
        # CRITICAL: Calculate densities AFTER region filtering for independent regional densities
        # This ensures density_threshold applies to each region independently
        
        # Filter positions by distance/region first
        region_filtered_positions = ion_positions_shifted[distance_mask]
        
        if len(region_filtered_positions) == 0:
            filtered_positions = np.array([]).reshape(0, 3)
            filtered_densities = np.array([])
            filtered_densities_raw = np.array([])
        else:
            # Build KDTree from region-filtered positions only
            tree = cKDTree(region_filtered_positions)
            
            # Calculate local density for each position (neighbors within 2 Å)
            search_radius = 2.0
            densities = np.array([len(tree.query_ball_point(pos, search_radius)) 
                                 for pos in region_filtered_positions])
            
            # Frame normalization
            if n_frames_analyzed and n_frames_analyzed > 0:
                # Normalize to contacts per frame
                densities = densities / n_frames_analyzed
            
            # Coloring strategy: use pre-calculated global scale or relative
            if use_absolute_coloring and max_density_scale:
                # For frame-normalized data: use absolute density scale (consistent across regions)
                densities_norm = np.clip(densities / max_density_scale, 0, 1)
            else:
                # For non-normalized data: relative coloring within this dataset
                max_density = densities.max()
                densities_norm = densities / max_density if max_density > 0 else densities
            
            # Filter by density threshold
            density_mask_regional = densities_norm >= density_threshold
            
            # Apply density threshold to already region-filtered positions
            filtered_positions = region_filtered_positions[density_mask_regional]
            filtered_densities = densities_norm[density_mask_regional]
            filtered_densities_raw = densities[density_mask_regional]
        
        # Combine filters
        combined_mask = distance_mask.copy()  # No longer combining - already filtered
        if len(filtered_densities_raw) > 0:
            # Use the filtered positions
            if effective_regions and show_output:
                print(f"Displaying regions: {', '.join(effective_regions)}")
        elif show_output:
            print(f"\n⚠️  No positions remain after density filtering!")
            print(f"  Try lowering density_threshold or check your region definitions")
        
        # Per-region subsampling if using discrete regions, otherwise global subsampling
        if use_discrete_regions and len(plot_regions) > 1 and len(filtered_positions) > 0:
            # Independent subsampling per region - each gets full max_spheres quota
            # Note: We need to re-calculate which filtered position belongs to which region
            
            final_positions = []
            final_densities = []
            
            # Calculate distances from filtered positions to target
            if distance_method == 'stored':
                # Use stored distances that were already calculated for filtered positions
                # We need to map filtered_positions back to their original indices in ion_positions_shifted
                # This is complex, so for now just subsample globally
                if len(filtered_positions) > max_spheres * len(plot_regions):
                    top_indices = np.argsort(filtered_densities)[-max_spheres * len(plot_regions):]
                    filtered_positions = filtered_positions[top_indices]
                    filtered_densities = filtered_densities[top_indices]
                
                if show_output:
                    print(f"Global sampling for multi-region (stored distances): {len(filtered_positions)} spheres")
            else:
                # Calculate distances for filtered positions
                if distance_method == 'nearest_atom':
                    atom_coords_array = np.array(atom_coords)
                    mol_tree = cKDTree(atom_coords_array)
                    filtered_distances, _ = mol_tree.query(filtered_positions)
                else:  # 'com'
                    filtered_distances = np.linalg.norm(filtered_positions - molecule_com, axis=1)
                
                for region in plot_regions:
                    if region in region_boundaries:
                        r_min, r_max = region_boundaries[region]
                        
                        # Find positions in this specific region from filtered set
                        region_mask = (filtered_distances >= r_min) & (filtered_distances <= r_max)
                        region_positions = filtered_positions[region_mask]
                        region_densities = filtered_densities[region_mask]
                        
                        # Subsample this region independently (full quota)
                        if len(region_positions) > max_spheres:
                            top_indices = np.argsort(region_densities)[-max_spheres:]
                            region_positions = region_positions[top_indices]
                            region_densities = region_densities[top_indices]
                        
                        final_positions.append(region_positions)
                        final_densities.append(region_densities)
                        
                        if show_output:
                            print(f"Region {region}: {len(region_positions)} spheres (from {region_mask.sum()} candidates)")
                
                # Combine all regions
                if final_positions and any(len(pos) > 0 for pos in final_positions):
                    valid_positions = [pos for pos in final_positions if len(pos) > 0]
                    valid_densities = [dens for i, dens in enumerate(final_densities) if len(final_positions[i]) > 0]
                    if valid_positions:
                        filtered_positions = np.vstack(valid_positions)
                        filtered_densities = np.concatenate(valid_densities)
                    else:
                        filtered_positions = np.array([]).reshape(0, 3)
                        filtered_densities = np.array([])
                
        elif len(filtered_positions) > max_spheres:
            # Global subsampling for single region or non-discrete filtering
            top_indices = np.argsort(filtered_densities)[-max_spheres:]
            filtered_positions = filtered_positions[top_indices]
            filtered_densities = filtered_densities[top_indices]
        
        # Add spheres at ion binding positions in 3D space
        for pos, density in zip(filtered_positions, filtered_densities):
            # Color gradient: blue (low) -> white (mid) -> red (high)
            if density < 0.5:
                r = int(255 * density * 2)
                g = int(255 * density * 2)
                b = 255
            else:
                r = 255
                g = int(255 * (1 - (density - 0.5) * 2))
                b = int(255 * (1 - (density - 0.5) * 2))
            
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            
            # Add sphere at this position in space (not at molecule atoms)
            view.addSphere({
                'center': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
                'radius': sphere_size,
                'color': hex_color,
                'alpha': sphere_opacity
            })
        
        # Add boundary spheres if requested - MUST use same reference as ion positions
        if show_boundary_spheres and use_discrete_regions:
            # Ensure boundary spheres use the exact same reference point as ion positions
            if 'atom_indices' in spatial_results:
                # Use target center (same as ion position reconstruction)
                target_indices = spatial_results['atom_indices']
                target_atoms = universe.atoms[target_indices]
                if len(target_atoms) > 1:
                    boundary_center = target_atoms.center_of_mass()
                else:
                    boundary_center = target_atoms.positions[0]
            else:
                # Fallback to molecule COM (same as ion position fallback)
                boundary_center = molecule_com
                
            self._add_boundary_spheres(view, plot_regions, region_boundaries, 
                                     boundary_center,
                                     show_output, boundary_sphere_alpha, color_shade_style,
                                     boundary_sphere_data_extent, filtered_positions if boundary_sphere_data_extent else None)
        elif show_boundary_spheres and not use_discrete_regions:
            if show_output:
                print(f"⚠️  Boundary spheres only available for discrete region visualization")

        
        if show_output:
            print(f"\n{'='*60}")
            print(f"✓ Interactive 3D viewer ready!")
            print(f"{'='*60}")
            print(f"Controls:")
            print(f"  • Click and drag to rotate")
            print(f"  • Scroll to zoom")
            print(f"  • Right-click and drag to pan")
            print(f"\nColor scheme:")
            if use_absolute_coloring:
                print(f"  🔵 Blue = 0 contacts/frame (rare binding)")
                print(f"  🟣 White = ~{max_density_scale/2:.4f} contacts/frame (moderate)")
                print(f"  🔴 Red = {max_density_scale:.4f}+ contacts/frame (frequent binding)")
                print(f"  ✓ ABSOLUTE scale - colors comparable across all clusters!")
                print(f"  ✓ Normalized by {n_frames_analyzed} frames")
                if isinstance(density_scale, (int, float)):
                    print(f"  ✓ Using fixed density_scale={density_scale}")
                    print(f"  💡 Use same density_scale value for all clusters to compare!")
                elif density_scale == 'auto':
                    print(f"  ℹ️  Using auto-scaled max from this dataset")
                    print(f"  💡 For cluster comparison, use density_scale={max_density_scale:.4f} for all")
            else:
                print(f"  🔵 Blue spheres = Low-density binding regions")
                print(f"  🟣 Purple/White spheres = Medium-density regions")
                print(f"  🔴 Red spheres = High-density binding regions")
                print(f"  Note: Relative scale within this dataset only")
            print(f"\nMolecule: Ball-and-stick structure (colored by atom type)")
            print(f"\n{'='*60}")
            print(f"💡 TO SAVE HIGH-RESOLUTION IMAGE:")
            print(f"{'='*60}")
            print(f"  Option 1: Right-click viewer → 'Save Image As...'")
            print(f"  Option 2: Use screenshot tool (Cmd+Shift+4 on Mac)")
            print(f"  Option 3: Use export_pymol_script() for PyMOL rendering")
            print(f"{'='*60}")
        
        view.zoomTo()
        return view
    
    # =========================================================================
    # EXPORT UTILITIES
    # =========================================================================
    
    def export_current_figure(self, base_filename='figure', 
                             export_formats=None, **format_overrides):
        """
        Export the currently active matplotlib figure in multiple formats
        
        Parameters
        ----------
        base_filename : str
            Base name for output files
        export_formats : list, optional
            List of format names: 'word', 'journal', 'presentation', 'poster'
            If None, exports 'word' and 'journal'
        **format_overrides : dict
            Override format-specific parameters (e.g., dpi=600)
        
        Returns
        -------
        list
            List of exported filenames
        """
        
        if export_formats is None:
            export_formats = ['word', 'journal']
        
        format_specs = {
            'word': {'dpi': 300, 'extension': 'png'},
            'presentation': {'dpi': 150, 'extension': 'png'},
            'journal': {'dpi': 600, 'extension': 'png'},
            'poster': {'dpi': 300, 'extension': 'png'},
        }
        
        exported_files = []
        
        print(f"\n{'='*60}")
        print(f"EXPORTING FIGURE: {base_filename}")
        print(f"{'='*60}")
        
        for format_name in export_formats:
            if format_name not in format_specs:
                print(f"Warning: Unknown format '{format_name}'. Skipping...")
                continue
            
            specs = format_specs[format_name].copy()
            specs.update(format_overrides)
            
            filename = f"{base_filename}_{format_name}.{specs['extension']}"
            
            print(f"\n  Exporting {format_name} format...")
            print(f"    DPI: {specs['dpi']}")
            print(f"    File: {filename}")
            
            try:
                plt.savefig(filename, dpi=specs['dpi'], bbox_inches='tight',
                          facecolor='white', edgecolor='none')
                exported_files.append(filename)
                print(f"    ✓ Success")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        print(f"\n{'='*60}")
        print(f"Exported {len(exported_files)} file(s)")
        print(f"{'='*60}\n")
        
        return exported_files
    
    def save_all_figures(self, prefix='figure', dpi=300):
        """
        Save all currently open matplotlib figures
        
        Parameters
        ----------
        prefix : str
            Prefix for filenames
        dpi : int
            Resolution
        
        Returns
        -------
        list
            List of saved filenames
        """
        
        saved_files = []
        
        for i in plt.get_fignums():
            fig = plt.figure(i)
            filename = f"{prefix}_{i:02d}.png"
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            saved_files.append(filename)
            print(f"✓ Saved: {filename}")
        
        return saved_files

    # =========================================================================
    # VOLUME NORMALIZATION METHODS (NEW)
    # =========================================================================

    def _calculate_overall_density(self, peak_analysis):
        """
        Calculate overall volume-weighted density across all peaks.
        
        Parameters
        ----------
        peak_analysis : dict
            Peak analysis data containing volume and density information
            
        Returns
        -------
        overall_density : float or None
            Volume-weighted average density across all peaks
        """
        total_weighted_density = 0
        total_volume = 0
        
        for peak_name, peak_data in peak_analysis.items():
            if ('volume_density' in peak_data and peak_data['volume_density'] is not None and
                'volume_data' in peak_data and peak_data['volume_data'] is not None):
                
                volume = peak_data['volume_data']['volume']
                density = peak_data['volume_density']
                
                # Weight by volume
                total_weighted_density += density * volume
                total_volume += volume
        
        if total_volume > 0:
            return total_weighted_density / total_volume
        else:
            return None

    def _extract_volume_normalized_timeseries(self, ion_data, volume_calculation_method='weighted_average'):
        """
        Extract volume-normalized density timeseries data from ion binding results.
        
        For timeseries, we need per-frame density values. This method calculates
        volume-normalized density per frame using peak analysis data.
        
        Parameters
        ----------
        ion_data : dict
            Ion binding data containing 'binding_events' and 'peak_analysis'
        volume_calculation_method : str
            'weighted_average' or 'sum' calculation method
            
        Returns
        -------
        density_per_frame : np.array or None
            Volume-normalized density per frame, or None if no volume data available
        """
        if 'peak_analysis' not in ion_data or 'binding_events' not in ion_data:
            return None
        
        peak_analysis = ion_data['peak_analysis']
        raw_binding_events = np.array(ion_data['binding_events'])
        n_frames = len(raw_binding_events)
        
        # Calculate density conversion factor
        if volume_calculation_method == 'weighted_average':
            # Use overall volume-weighted density conversion
            overall_density = self._calculate_overall_density(peak_analysis)
            if overall_density is None:
                return None
            
            # Calculate average binding for conversion factor
            average_binding = np.mean(raw_binding_events)
            if average_binding == 0:
                return np.zeros(n_frames)
            
            # Convert raw counts to density per frame
            density_conversion_factor = overall_density / average_binding
            density_per_frame = raw_binding_events * density_conversion_factor
            
        elif volume_calculation_method == 'sum':
            # Sum all peak densities and use as conversion
            total_density = 0
            for peak_name, peak_data in peak_analysis.items():
                if 'volume_density' in peak_data and peak_data['volume_density'] is not None:
                    total_density += peak_data['volume_density']
            
            if total_density == 0:
                return np.zeros(n_frames)
            
            # Calculate average binding for conversion factor
            average_binding = np.mean(raw_binding_events)
            if average_binding == 0:
                return np.zeros(n_frames)
            
            # Convert raw counts to density per frame
            density_conversion_factor = total_density / average_binding
            density_per_frame = raw_binding_events * density_conversion_factor
            
        else:
            # Default to weighted average
            overall_density = self._calculate_overall_density(peak_analysis)
            if overall_density is None:
                return None
            
            average_binding = np.mean(raw_binding_events)
            if average_binding == 0:
                return np.zeros(n_frames)
            
            density_conversion_factor = overall_density / average_binding
            density_per_frame = raw_binding_events * density_conversion_factor
        
        return density_per_frame

    def plot_peak_density_comparison(self, binding_results_dict, peaks_to_plot=['P1', 'P2', 'P3'], 
                                    ion_types=None, target_selection=None, save_fig=False, 
                                    filename='peak_density_comparison.png', figsize=(14, 8),
                                    title='Ion Binding Density Comparison by Peak', dpi=300):
        """
        Create grouped bar chart comparing volume-normalized densities across peaks.
        
        Parameters
        ----------
        binding_results_dict : dict
            Dictionary of binding results from ion_binding_analysis with volume normalization
            Format: {target_label: binding_results}
        peaks_to_plot : list
            List of peaks to compare (default: ['P1', 'P2', 'P3'])
        ion_types : list or None
            Specific ion types to include. If None, includes all available ions.
        target_selection : list or None  
            Specific targets to include. If None, includes all available targets.
        save_fig : bool
            Whether to save the figure
        filename : str
            Output filename if saving
        figsize : tuple
            Figure size (width, height)
        title : str
            Plot title
        dpi : int
            Resolution for saved figure
            
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        import numpy as np
        
        # Collect density data
        target_data = {}
        
        for target_label, binding_results in binding_results_dict.items():
            if target_selection is None or target_label in target_selection:
                target_data[target_label] = {}
                
                for ion_type in ['cation_binding', 'anion_binding']:
                    if ion_type in binding_results:
                        for ion_name, data in binding_results[ion_type].items():
                            if ion_types is None or ion_name in ion_types:
                                if 'peak_analysis' in data and data['peak_analysis']:
                                    ion_key = f"{ion_name}"
                                    if ion_key not in target_data[target_label]:
                                        target_data[target_label][ion_key] = {}
                                    
                                    for peak in peaks_to_plot:
                                        if (peak in data['peak_analysis'] and 
                                            'volume_density' in data['peak_analysis'][peak]):
                                            density = data['peak_analysis'][peak]['volume_density']
                                            target_data[target_label][ion_key][peak] = density or 0
                                        else:
                                            target_data[target_label][ion_key][peak] = 0
        
        if not target_data:
            print("No volume-normalized data available for plotting")
            return None, None
        
        # Determine layout
        targets = list(target_data.keys())
        all_ions = set()
        for target_dict in target_data.values():
            all_ions.update(target_dict.keys())
        all_ions = sorted(list(all_ions))
        
        # Create grouped bar chart
        x = np.arange(len(all_ions))
        width = 0.15
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(peaks_to_plot)))
        patterns = ['', '///', '...', '+++', '***'][:len(targets)]
        
        for i, peak in enumerate(peaks_to_plot):
            for j, target in enumerate(targets):
                densities = []
                for ion in all_ions:
                    ion_data = target_data[target].get(ion, {})
                    densities.append(ion_data.get(peak, 0))
                
                offset = (i * len(targets) + j - len(peaks_to_plot) * len(targets) / 2) * width + width/2
                bars = ax.bar(x + offset, densities, width, 
                             label=f'{peak}-{target}' if len(targets) > 1 else peak,
                             color=colors[i], alpha=0.8, 
                             hatch=patterns[j] if len(targets) > 1 else '')
                
                # Add value labels for non-zero values
                for bar, value in zip(bars, densities):
                    if value > 0.001:  # Only show significant densities
                        ax.text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + max(densities) * 0.01,
                               f'{value:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax.set_xlabel('Ion Type', fontsize=12)
        ax.set_ylabel('Volume-Normalized Density (ions/frame/Å³)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_ions)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"📊 Peak density comparison saved: {filename}")
        
        return fig, ax

    def plot_volume_correlation(self, binding_results_dict, save_fig=False, 
                               filename='volume_correlation_analysis.png', figsize=(16, 8),
                               target_selection=None, ion_types=None, dpi=300):
        """
        Create correlation plots between peak volumes and binding metrics.
        
        Parameters
        ----------
        binding_results_dict : dict
            Dictionary of binding results from ion_binding_analysis with volume normalization
        save_fig : bool
            Whether to save the figure
        filename : str
            Output filename if saving
        figsize : tuple
            Figure size (width, height)
        target_selection : list or None
            Specific targets to include in analysis
        ion_types : list or None
            Specific ion types to include in analysis
        dpi : int
            Resolution for saved figure
            
        Returns
        -------
        fig, axes : matplotlib figure and axes
        """
        import numpy as np
        
        volumes = []
        raw_counts = []
        densities = []
        labels = []
        colors = []
        target_labels = []
        
        # Color map for different targets
        target_colors = plt.cm.Set1(np.linspace(0, 1, len(binding_results_dict)))
        target_color_map = {target: target_colors[i] for i, target in enumerate(binding_results_dict.keys())}
        
        # Collect data for correlation
        for target_label, binding_results in binding_results_dict.items():
            if target_selection is None or target_label in target_selection:
                for ion_type in ['cation_binding', 'anion_binding']:
                    if ion_type in binding_results:
                        for ion_name, data in binding_results[ion_type].items():
                            if ion_types is None or ion_name in ion_types:
                                if 'peak_analysis' in data and data['peak_analysis']:
                                    for peak_name, peak_data in data['peak_analysis'].items():
                                        if ('volume_data' in peak_data and peak_data['volume_data'] and
                                            'volume_density' in peak_data and peak_data['volume_density'] is not None):
                                            
                                            volumes.append(peak_data['volume_data']['volume'])
                                            raw_counts.append(peak_data['average_binding'])
                                            densities.append(peak_data['volume_density'])
                                            labels.append(f"{target_label}-{ion_name}-{peak_name}")
                                            target_labels.append(target_label)
                                            
                                            # Color by target
                                            colors.append(target_color_map[target_label])
        
        if len(volumes) < 2:
            print("Insufficient data for correlation plot")
            return None, None
        
        # Create correlation plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Volume vs Raw Counts
        scatter1 = ax1.scatter(volumes, raw_counts, c=colors, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Peak Volume (Å³)', fontsize=12)
        ax1.set_ylabel('Average Ion Count (ions/frame)', fontsize=12)
        ax1.set_title('Volume vs Raw Ion Counts', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add trendline
        if len(volumes) > 1:
            z = np.polyfit(volumes, raw_counts, 1)
            p = np.poly1d(z)
            ax1.plot(volumes, p(volumes), "r--", alpha=0.8, linewidth=2)
            
            # Calculate R²
            correlation = np.corrcoef(volumes, raw_counts)[0,1]
            ax1.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax1.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Volume vs Density
        scatter2 = ax2.scatter(volumes, densities, c=colors, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Peak Volume (Å³)', fontsize=12)
        ax2.set_ylabel('Volume Density (ions/frame/Å³)', fontsize=12)
        ax2.set_title('Volume vs Normalized Density', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add trendline
        if len(volumes) > 1:
            z2 = np.polyfit(volumes, densities, 1)
            p2 = np.poly1d(z2)
            ax2.plot(volumes, p2(volumes), "r--", alpha=0.8, linewidth=2)
            
            # Calculate R²
            correlation2 = np.corrcoef(volumes, densities)[0,1]
            ax2.text(0.05, 0.95, f'R = {correlation2:.3f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add legend for targets
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=target_color_map[target], label=target) 
                          for target in target_color_map.keys()]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"🔗 Volume correlation plot saved: {filename}")
        
        return fig, (ax1, ax2)

    def plot_binding_with_volume_normalization(self, binding_results_dict, normalize_by_volume=True,
                                              plot_peaks=None, density_units='auto',
                                              comparison_mode='targets',  # 'targets', 'ions', 'peaks' 
                                              save_fig=False, filename=None, figsize=(14, 8), dpi=300):
        """
        Enhanced plotting method specifically for volume-normalized binding analysis.
        
        Parameters
        ----------
        binding_results_dict : dict
            Dictionary of binding results with volume normalization
        normalize_by_volume : bool
            Whether to plot densities (True) or raw counts (False)
        plot_peaks : list or None
            Specific peaks to plot. If None, calculates overall binding.
        density_units : str
            Units for density: 'auto', 'per_A3', 'per_nm3'
        comparison_mode : str
            What to emphasize: 'targets' (group by target), 'ions' (group by ion), 'peaks' (group by peak)
        save_fig : bool
            Whether to save the figure
        filename : str or None
            Custom filename. If None, auto-generates based on parameters.
        figsize : tuple
            Figure size
        dpi : int
            Resolution for saved figure
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        
        # Auto-generate filename if not provided
        if filename is None:
            filename = 'binding_analysis'
            if normalize_by_volume:
                filename += '_volume_normalized'
            if plot_peaks:
                filename += f'_peaks_{"_".join(plot_peaks)}'
            filename += f'_{comparison_mode}.png'
        
        # Call the enhanced plot_ion_binding_comparison with volume parameters
        return self.plot_ion_binding_comparison(
            binding_results_dict,
            normalize_by_volume=normalize_by_volume,
            plot_peaks=plot_peaks,
            density_units=density_units,
            volume_info_in_title=True,
            save_fig=save_fig,
            filename=filename,
            figsize=figsize,
            dpi=dpi
        )

    def _validate_region_parameters(self, spatial_results, plot_regions, distance_cutoff, show_output, boundaries_ions_refined=None):
        """
        Validate plot_regions and distance_cutoff parameters for data integrity.
        
        Prevents visualization of regions that weren't actually analyzed,
        which could lead to misleading scientific interpretations.
        """
        is_shell_analysis = spatial_results.get('use_shell_boundaries', False)
        
        # Case 1: Both plot_regions and distance_cutoff specified (conflicting)
        if plot_regions is not None and distance_cutoff is not None:
            raise ValueError(
                "Cannot specify both 'plot_regions' and 'distance_cutoff' simultaneously.\n"
                "Use 'plot_regions=['P1', 'P2']' for shell-boundary analysis OR\n"
                "Use 'distance_cutoff=(2.0, 3.5)' for custom range analysis."
            )
        
        # Case 2: plot_regions with non-shell analysis
        if plot_regions is not None and not is_shell_analysis:
            raise ValueError(
                f"plot_regions={plot_regions} specified but spatial_results uses distance cutoff analysis.\n"
                f"plot_regions only works with shell-boundary analysis.\n"
                f"Use distance_cutoff parameter for custom filtering instead."
            )
        
        # Case 3: Validate plot_regions against analyzed regions
        if plot_regions is not None and is_shell_analysis:
            if not isinstance(plot_regions, list):
                plot_regions = [plot_regions]
            
            # Use correct key name for analyzed regions
            analyzed_regions = spatial_results.get('selected_analysis_regions', [])
            missing_regions = set(plot_regions) - set(analyzed_regions)
            
            if missing_regions:
                raise ValueError(
                    f"Requested regions {list(missing_regions)} were not analyzed.\n"
                    f"Available regions: {analyzed_regions}\n"
                    f"Re-run spatial_binding_analysis() with these regions included."
                )
        
        # Case 4: Validate custom distance_cutoff against analyzed data
        if distance_cutoff is not None and is_shell_analysis:
            # Try to extract region boundaries from spatial_results or reconstruct from target_species_key
            target_species_key = spatial_results.get('target_species_key')
            
            # Extract region boundaries from provided boundaries_ions_refined
            region_boundaries = None
            if target_species_key and boundaries_ions_refined:
                region_boundaries = boundaries_ions_refined.get(target_species_key, {})
            
            # Alternative: check if boundaries are directly stored (future enhancement)
            if not region_boundaries:
                region_boundaries = spatial_results.get('region_boundaries', {})
            
            if region_boundaries:
                # Convert distance_cutoff to range format
                if isinstance(distance_cutoff, (int, float)):
                    custom_range = (0.0, float(distance_cutoff))
                elif isinstance(distance_cutoff, (tuple, list)) and len(distance_cutoff) == 2:
                    custom_range = tuple(distance_cutoff)
                else:
                    raise ValueError(
                        f"distance_cutoff must be a number or (min, max) tuple, got: {distance_cutoff}"
                    )
                
                # Check if custom range has any overlap with analyzed regions
                analyzed_regions = spatial_results.get('selected_analysis_regions', [])
                has_overlap = False
                overlapping_regions = []
                
                for region in analyzed_regions:
                    if region in region_boundaries:
                        r_min, r_max = region_boundaries[region]
                        # Check for overlap: ranges overlap if not (max1 < min2 or min1 > max2)
                        if not (custom_range[1] < r_min or custom_range[0] > r_max):
                            has_overlap = True
                            overlapping_regions.append(f"{region}({r_min:.1f}-{r_max:.1f})")
                
                if not has_overlap:
                    available_ranges = [f"{region}({region_boundaries[region][0]:.1f}-{region_boundaries[region][1]:.1f})" 
                                      for region in analyzed_regions if region in region_boundaries]
                    raise ValueError(
                        f"Custom range {custom_range} has no analyzed data.\n"
                        f"Analyzed regions: {', '.join(available_ranges)}\n"
                        f"Specify a range that overlaps with analyzed regions or use plot_regions instead."
                    )
                
                elif show_output:
                    print(f"✓ Custom range {custom_range} overlaps with: {', '.join(overlapping_regions)}")
            elif show_output:
                print(f"⚠️  Cannot validate custom range - no boundary information available")
    
    def _determine_distance_filtering(self, spatial_results, plot_regions, distance_cutoff, show_output, boundaries_ions_refined=None):
        """
        Determine the effective distance filtering parameters based on inputs.
        
        Returns
        -------
        effective_distance_cutoff : float or tuple or None
            Distance cutoff to use for filtering
        effective_regions : list or None
            Regions to display (for informational purposes)
        """
        is_shell_analysis = spatial_results.get('use_shell_boundaries', False)
        
        if plot_regions is not None and is_shell_analysis:
            # Use shell-boundary regions - need to extract their ranges
            target_species_key = spatial_results.get('target_species_key')
            
            # Extract region boundaries from provided boundaries_ions_refined
            region_boundaries = None
            if target_species_key and boundaries_ions_refined:
                region_boundaries = boundaries_ions_refined.get(target_species_key, {})
            
            # For now, since boundaries aren't directly stored in spatial_results,
            # we'll need to access them from the calling environment
            # This is a limitation that should be fixed by storing boundaries in spatial_results
            
            if region_boundaries:
                # Find min and max distances across all requested regions
                region_ranges = [region_boundaries[region] for region in plot_regions if region in region_boundaries]
                if region_ranges:
                    min_dist = min(r[0] for r in region_ranges)
                    max_dist = max(r[1] for r in region_ranges)
                    effective_distance_cutoff = (min_dist, max_dist)
                    
                    if show_output:
                        region_info = [f"{region}({region_boundaries[region][0]:.1f}-{region_boundaries[region][1]:.1f})" 
                                      for region in plot_regions if region in region_boundaries]
                        print(f"✓ Using shell regions: {', '.join(region_info)}")
                        print(f"✓ Effective distance range: {min_dist:.1f}-{max_dist:.1f} Å")
                else:
                    # Fallback if regions not found in boundaries
                    effective_distance_cutoff = 50.0
                    if show_output:
                        print(f"⚠️  Could not find region boundaries, using large cutoff: {effective_distance_cutoff} Å")
            else:
                # Fallback: use large cutoff if no boundaries available
                effective_distance_cutoff = 50.0
                if show_output:
                    print(f"⚠️  No region boundaries found, using large cutoff: {effective_distance_cutoff} Å")
            
            return effective_distance_cutoff, plot_regions
        
        elif distance_cutoff is not None:
            # Use custom distance cutoff
            if show_output:
                if isinstance(distance_cutoff, (tuple, list)):
                    print(f"✓ Using custom distance range: {distance_cutoff[0]:.1f}-{distance_cutoff[1]:.1f} Å")
                else:
                    print(f"✓ Using custom distance cutoff: ≤{distance_cutoff:.1f} Å")
            
            return distance_cutoff, None
        
        else:
            # No specific filtering requested
            if is_shell_analysis:
                # Use all analyzed regions
                effective_distance_cutoff = 50.0  # Large cutoff to not interfere
                analyzed_regions = spatial_results.get('selected_analysis_regions', [])
                
                if show_output:
                    print(f"✓ Using all analyzed shell regions: {analyzed_regions}")
                    print(f"✓ Distance cutoff set large (50 Å) to not interfere with shell boundaries")
                
                return effective_distance_cutoff, analyzed_regions
            else:
                # Distance-cutoff analysis, need explicit cutoff
                default_cutoff = 6.0
                if show_output:
                    print(f"⚠️  No distance filtering specified, using default: {default_cutoff} Å")
                
                return default_cutoff, None
    
    def _add_boundary_spheres(self, view, plot_regions, region_boundaries, center_point, show_output, boundary_sphere_alpha=0.3, color_shade_style='modified', boundary_sphere_data_extent=False, ion_positions=None):
        """
        Add smooth transparent spheres marking the outer boundaries of each region.
        
        Parameters
        ----------
        view : py3Dmol.view
            The 3D viewer object
        plot_regions : list
            List of regions being plotted (e.g., ['P2', 'P3'])
        region_boundaries : dict
            Dictionary mapping region names to (min, max) boundaries
        center_point : np.array
            Center point for the boundary spheres (usually target atom center)
        show_output : bool
            Whether to print informational output
        boundary_sphere_alpha : float, optional
            Transparency level for boundary spheres (0.0-1.0), default: 0.3
        color_shade_style : str, optional
            Color scheme to match RDF plots ('modified' or 'original'), default: 'modified'
        boundary_sphere_data_extent : bool, optional
            If True, draw partial spheres only where ion binding actually occurs (angular extents),
            If False, draw complete spheres at theoretical boundaries, default: False
        ion_positions : np.array, optional
            Filtered ion positions for calculating actual angular extents, required when boundary_sphere_data_extent=True
        """
        # Use same color scheme as RDF plots for consistency
        if color_shade_style == 'modified':
            # Modified color scheme - improved contrast
            boundary_colors = {
                'P1': 'lightcoral',     # Light red - same as RDF
                'P2': 'lightgreen',     # Light green - matches RDF modified scheme
                'P3': 'lightyellow',    # Light yellow - matches RDF modified scheme
                'P4': 'lightblue',      # Light blue - matches RDF modified scheme
                'Bulk': 'aliceblue'     # Very light blue
            }
        elif color_shade_style == 'vibrant':
            # High-contrast vibrant colors - best for presentations
            boundary_colors = {
                'P1': '#FF6B6B',        # Bright red - high contrast
                'P2': '#4ECDC4',        # Teal - distinct from red
                'P3': '#45B7D1',        # Blue - clear progression
                'P4': '#96CEB4',        # Light green - completes spectrum
                'Bulk': '#FFEAA7'       # Light yellow - neutral
            }
        else:
            # Original color scheme - default
            boundary_colors = {
                'P1': 'lightcoral',              # Light red - same as RDF
                'P2': 'lightblue',               # Light blue - matches RDF original scheme
                'P3': 'lightgreen',              # Light green - matches RDF original scheme
                'P4': 'lightgoldenrodyellow',    # Light golden yellow - matches RDF original scheme
                'Bulk': 'lightyellow'            # Light yellow
            }
        
        boundary_spheres_added = 0
        
        for region in plot_regions:
            if region in region_boundaries:
                r_min, r_max = region_boundaries[region]
                color = boundary_colors.get(region, '#CCCCCC')  # Default gray
                
                if boundary_sphere_data_extent and ion_positions is not None:
                    # Advanced: Draw partial sphere only where data exists
                    self._add_data_extent_sphere(view, region, ion_positions, center_point, r_min, r_max, color, boundary_sphere_alpha, show_output)
                else:
                    # Simple: Draw complete sphere at theoretical boundary
                    view.addSphere({
                        'center': {'x': float(center_point[0]), 'y': float(center_point[1]), 'z': float(center_point[2])},
                        'radius': float(r_max),
                        'color': color,
                        'alpha': boundary_sphere_alpha,  # User-controlled transparency
                        'wireframe': True  # Wireframe for better visibility without blocking ions
                    })
                
                boundary_spheres_added += 1

    def _add_data_extent_sphere(self, view, region, ion_positions, center_point, r_min, r_max, color, alpha, show_output):
        """
        Add partial sphere surface showing only angular regions where ion binding actually occurs.
        
        Parameters
        ----------
        view : py3Dmol.view
            The 3D viewer object
        region : str
            Region name (e.g., 'P2', 'P3')
        ion_positions : np.array
            Array of ion positions for angular analysis
        center_point : np.array
            Center point for spherical coordinate calculation
        r_min, r_max : float
            Region boundary distances
        color : str
            Color for the partial sphere
        alpha : float
            Transparency level
        show_output : bool
            Whether to print informational output
        """
        import numpy as np
        
        # Filter ion positions within this region's boundaries
        distances = np.linalg.norm(ion_positions - center_point, axis=1)
        region_mask = (distances >= r_min) & (distances <= r_max)
        region_ions = ion_positions[region_mask]
        
        if len(region_ions) == 0:
            if show_output:
                print(f"⚠️  No ion data for {region} - skipping data-extent sphere")
            return
        
        # Convert to spherical coordinates relative to center
        relative_positions = region_ions - center_point
        
        # FIXED: Determine actual orientation of ion cluster using PCA
        # Don't assume Z-axis, find the principal axes of the actual data
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(relative_positions)
        
        # Transform coordinates to align with principal axes of actual data
        # This ensures angles are calculated relative to where ions actually cluster
        transformed_positions = pca.transform(relative_positions)
        
        # Now calculate spherical coordinates in the properly oriented coordinate system
        r = np.linalg.norm(transformed_positions, axis=1)
        
        # Use the first principal component as the "polar axis"
        # This aligns with the actual dominant direction of ion clustering
        z_aligned = transformed_positions[:, 0]  # First PC is most dominant
        x_aligned = transformed_positions[:, 1]  # Second PC
        y_aligned = transformed_positions[:, 2]  # Third PC
        
        # Calculate angles in the properly aligned system
        theta = np.arccos(np.clip(z_aligned / r, -1, 1))  # Polar angle relative to actual main axis
        phi = np.arctan2(y_aligned, x_aligned)  # Azimuthal angle in the actual plane
        
        # Use much tighter percentiles to focus on actual core binding locations
        # Start with tight core (25th-75th percentile) and expand only if needed
        theta_core_min = np.percentile(theta, 25)   # 25th percentile - tight core
        theta_core_max = np.percentile(theta, 75)   # 75th percentile - tight core  
        phi_core_min = np.percentile(phi, 25)       # 25th percentile for azimuthal
        phi_core_max = np.percentile(phi, 75)       # 75th percentile for azimuthal
        
        # Check if core is too restrictive (less than 10° range), then expand slightly
        theta_range = theta_core_max - theta_core_min
        phi_range_core = phi_core_max - phi_core_min
        
        if theta_range < np.pi/18:  # Less than 10° range
            # Expand to 15th-85th percentile
            theta_min = np.percentile(theta, 15)
            theta_max = np.percentile(theta, 85)
        else:
            theta_min, theta_max = theta_core_min, theta_core_max
            
        # Handle azimuthal angle wrap-around
        if phi_range_core > np.pi:  # Data spans across -π/π boundary
            phi_unwrapped = np.where(phi < 0, phi + 2*np.pi, phi)
            if theta_range < np.pi/18:
                phi_min = np.percentile(phi_unwrapped, 15) 
                phi_max = np.percentile(phi_unwrapped, 85)
            else:
                phi_min = np.percentile(phi_unwrapped, 25)
                phi_max = np.percentile(phi_unwrapped, 75)
            # Convert back to [-π, π] range
            if phi_min > np.pi:
                phi_min -= 2*np.pi
            if phi_max > np.pi:
                phi_max -= 2*np.pi
        else:
            if phi_range_core < np.pi/18:  # Less than 10° range
                phi_min = np.percentile(phi, 15)
                phi_max = np.percentile(phi, 85)
            else:
                phi_min, phi_max = phi_core_min, phi_core_max
        
        # Minimal buffer for visualization (only 1%)
        theta_buffer = max(0.02, (theta_max - theta_min) * 0.01)  # Min 1.15° buffer
        phi_buffer = max(0.02, (phi_max - phi_min) * 0.01)
        theta_min = max(0, theta_min - theta_buffer)
        theta_max = min(np.pi, theta_max + theta_buffer)
        phi_min = max(-np.pi, phi_min - phi_buffer)
        phi_max = min(np.pi, phi_max + phi_buffer)
        
        # Create partial sphere mesh at outer boundary using the same coordinate transformation
        self._create_partial_sphere_mesh(view, center_point, r_max, theta_min, theta_max, 
                                       phi_min, phi_max, color, alpha, pca_transform=pca)
        
        if show_output:
            theta_deg = (theta_min * 180/np.pi, theta_max * 180/np.pi)
            phi_deg = (phi_min * 180/np.pi, phi_max * 180/np.pi)
            print(f"✓ Added data-extent sphere for {region} (radius: {r_max:.1f} Å)")
            print(f"   θ range: {theta_deg[0]:.1f}° - {theta_deg[1]:.1f}°, φ range: {phi_deg[0]:.1f}° - {phi_deg[1]:.1f}°")
            print(f"   Based on {len(region_ions)} ion binding sites")

    def _create_partial_sphere_mesh(self, view, center, radius, theta_min, theta_max, phi_min, phi_max, color, alpha, pca_transform=None):
        """
        Create a clean wireframe mesh representing the actual data extent.
        Now uses coordinate transformation to match actual ion cluster orientation.
        """
        import numpy as np
        
        # Create precise wireframe mesh following data bounds
        n_theta = max(8, int((theta_max - theta_min) * 180/np.pi / 15))  # ~15° resolution
        n_phi = max(8, int((phi_max - phi_min) * 180/np.pi / 15))
        
        theta_range = np.linspace(theta_min, theta_max, n_theta)
        phi_range = np.linspace(phi_min, phi_max, n_phi)
        
        # Create wireframe lines only - no spheres!
        lines = []
        
        # Theta lines (meridians) - lines at constant phi
        for phi in phi_range:
            line_points = []
            for theta in theta_range:
                # Calculate coordinates in the aligned system (where angles make sense)
                z_aligned = radius * np.cos(theta)  # Along first principal component
                x_aligned = radius * np.sin(theta) * np.cos(phi)  # In the second PC
                y_aligned = radius * np.sin(theta) * np.sin(phi)  # In the third PC
                
                # Transform back to original coordinate system if PCA was used
                if pca_transform is not None:
                    aligned_point = np.array([[z_aligned, x_aligned, y_aligned]])
                    original_point = pca_transform.inverse_transform(aligned_point)[0]
                    x, y, z = original_point + center
                else:
                    # Fallback to standard coordinates if no transformation
                    x = radius * np.sin(theta) * np.cos(phi) + center[0]
                    y = radius * np.sin(theta) * np.sin(phi) + center[1]
                    z = radius * np.cos(theta) + center[2]
                
                line_points.append([float(x), float(y), float(z)])
            if len(line_points) > 1:
                lines.append(line_points)
        
        # Phi lines (parallels) - lines at constant theta  
        for theta in theta_range:
            line_points = []
            for phi in phi_range:
                # Calculate coordinates in the aligned system
                z_aligned = radius * np.cos(theta)
                x_aligned = radius * np.sin(theta) * np.cos(phi)
                y_aligned = radius * np.sin(theta) * np.sin(phi)
                
                # Transform back to original coordinate system
                if pca_transform is not None:
                    aligned_point = np.array([[z_aligned, x_aligned, y_aligned]])
                    original_point = pca_transform.inverse_transform(aligned_point)[0]
                    x, y, z = original_point + center
                else:
                    x = radius * np.sin(theta) * np.cos(phi) + center[0]
                    y = radius * np.sin(theta) * np.sin(phi) + center[1]
                    z = radius * np.cos(theta) + center[2]
                
                line_points.append([float(x), float(y), float(z)])
            if len(line_points) > 1:
                lines.append(line_points)
        
        # Add wireframe lines to viewer
        for line_points in lines:
            try:
                # Add as line segments
                for i in range(len(line_points) - 1):
                    start = line_points[i]
                    end = line_points[i + 1]
                    view.addLine({
                        'start': {'x': start[0], 'y': start[1], 'z': start[2]},
                        'end': {'x': end[0], 'y': end[1], 'z': end[2]},
                        'color': color,
                        'alpha': alpha,
                        'linewidth': 2
                    })
            except:
                # Fallback: Add very small cylinder for line if addLine fails
                pass

    def _add_aromatic_rings(self, view, universe, color, alpha, scale, thickness, show_output):
        """
        Detect and add aromatic ring indicators using actual molecular topology.
        
        Parameters
        ----------
        view : py3Dmol.view
            The 3D viewer object
        universe : MDAnalysis.Universe
            The molecular system with bond topology
        color : str
            Color of the aromatic ring circles
        alpha : float
            Transparency of the aromatic ring circles
        scale : float
            Scale factor for ring size (fraction of ring radius)
        thickness : float
            Thickness of the ring disk
        show_output : bool
            Whether to print informational output
            
        Returns
        -------
        int
            Number of aromatic rings added
        """
        try:
            if show_output:
                # print(f"\n🔍 DEBUG: Starting aromatic ring detection...")
                print(f"   Universe: {universe}")
                
                # Check if molecule exists
                molecule = universe.select_atoms("resname api")
                print(f"   CIP molecule atoms found: {len(molecule)}")
                
                # Check bond information
                try:
                    bonds = molecule.bonds
                    print(f"   Bond information available: {len(bonds)} bonds found")
                except:
                    print(f"   ⚠️  No bond information available in universe")
                
                # Check atom names and elements
                if len(molecule) > 0:
                    print(f"   Sample atoms: {[atom.name for atom in molecule[:5]]}")
                    print(f"   Sample elements: {[getattr(atom, 'element', 'Unknown') for atom in molecule[:5]]}")
            
            # Use MDAnalysis built-in ring detection with topology
            aromatic_rings = self._detect_aromatic_rings_topology(universe)
            
            if show_output:
                print(f"   Aromatic rings detected: {len(aromatic_rings)}")
            
            rings_added = 0
            for ring_atoms in aromatic_rings:
                try:
                    # Calculate ring center and normal using actual atomic positions
                    ring_center, ring_normal, ring_radius = self._calculate_ring_geometry_topology(ring_atoms)
                    
                    # Scale the radius (make it smaller to avoid bond overlap)
                    scaled_radius = ring_radius * scale * 0.6  # Additional 0.6 factor to keep rings inside
                    
                    if show_output:
                        ring_atom_names = [atom.name for atom in ring_atoms]
                        print(f"   Rendering ring {rings_added + 1}: {len(ring_atoms)} atoms ({', '.join(ring_atom_names[:3])}...) "
                              f"center=[{ring_center[0]:.2f}, {ring_center[1]:.2f}, {ring_center[2]:.2f}] "
                              f"radius={ring_radius:.2f}Å → {scaled_radius:.2f}Å")
                    
                    # Position ring in the same plane as the molecule, centered in the ring
                    start_pos = ring_center - ring_normal * thickness/2
                    end_pos = ring_center + ring_normal * thickness/2
                    
                    # Add aromatic ring circle as a cylinder disk
                    view.addCylinder({
                        'start': {
                            'x': float(start_pos[0]),
                            'y': float(start_pos[1]),
                            'z': float(start_pos[2])
                        },
                        'end': {
                            'x': float(end_pos[0]),
                            'y': float(end_pos[1]),
                            'z': float(end_pos[2])
                        },
                        'radius': float(scaled_radius),
                        'color': color,
                        'alpha': alpha
                    })
                    rings_added += 1
                    
                except Exception as e:
                    if show_output:
                        print(f"Warning: Could not render aromatic ring: {e}")
                    continue
            
            if show_output:
                print(f"✓ Successfully added {rings_added} aromatic ring indicator(s)")
            
            return rings_added
            
        except Exception as e:
            if show_output:
                print(f"Warning: Aromatic ring detection failed: {e}")
            return 0

    def _detect_aromatic_rings_topology(self, universe):
        """
        Detect aromatic rings using MDAnalysis topology information.
        
        Parameters
        ----------
        universe : MDAnalysis.Universe
            The molecular system with bond information
            
        Returns
        -------
        list
            List of aromatic rings, each ring is a list of MDAnalysis atoms
        """
        aromatic_rings = []
        
        try:
            # Get the CIP molecule
            molecule = universe.select_atoms("resname api")
            # print(f"🔍 DEBUG: _detect_aromatic_rings_topology called")
            print(f"   Molecule atoms: {len(molecule)}")
            
            if len(molecule) == 0:
                print(f"   ⚠️  No atoms found with 'resname api'")
                # Try alternative residue names
                alternative_names = ['CIP', 'CPXN', 'LIG', 'UNL', 'MOL']
                for resname in alternative_names:
                    alt_molecule = universe.select_atoms(f"resname {resname}")
                    if len(alt_molecule) > 0:
                        print(f"   ✓ Found {len(alt_molecule)} atoms with 'resname {resname}'")
                        molecule = alt_molecule
                        break
                else:
                    print(f"   ⚠️  No molecule found with any of the tried residue names: {alternative_names}")
                    return aromatic_rings
            
            # Check for bond topology
            try:
                bonds = molecule.bonds
                print(f"   Bond topology available: {len(bonds)} bonds")
                if len(bonds) > 0:
                    # Try bond-based detection
                    print(f"   Attempting bond-based ring detection...")
                    aromatic_rings = self._find_rings_from_bonds(molecule)
                    print(f"   Bond-based detection found: {len(aromatic_rings)} rings")
                else:
                    raise ValueError("No bonds found")
            except Exception as e:
                print(f"   Bond topology not available or failed: {e}")
                print(f"   Falling back to CIP-specific detection...")
                aromatic_rings = self._detect_cip_aromatic_rings(molecule)
                print(f"   CIP-specific detection found: {len(aromatic_rings)} rings")
                
        except Exception as e:
            print(f"Warning in topology-based ring detection: {e}")
            
        return aromatic_rings

    def _find_rings_from_bonds(self, molecule):
        """
        Use actual bond topology to find rings in the molecule.
        """
        aromatic_rings = []
        
        try:
            # Get bond information
            bonds = molecule.bonds
            if len(bonds) == 0:
                raise ValueError("No bond information available")
            
            # print(f"🔍 DEBUG: _find_rings_from_bonds called")
            print(f"   Molecule has {len(molecule)} atoms")
            print(f"   Found {len(bonds)} bonds")
            
            # Create mapping from global atom indices to local molecule indices
            global_to_local = {}
            for local_idx, atom in enumerate(molecule):
                global_to_local[atom.ix] = local_idx
            
            print(f"   Global index range in molecule: {min(atom.ix for atom in molecule)} - {max(atom.ix for atom in molecule)}")
            
            # Build adjacency list using local indices
            adjacency = {}
            for local_idx in range(len(molecule)):
                adjacency[local_idx] = []
            
            valid_bonds = 0
            for bond in bonds:
                atom1_global = bond[0].ix
                atom2_global = bond[1].ix
                
                # Map to local indices
                if atom1_global in global_to_local and atom2_global in global_to_local:
                    atom1_local = global_to_local[atom1_global]
                    atom2_local = global_to_local[atom2_global]
                    adjacency[atom1_local].append(atom2_local)
                    adjacency[atom2_local].append(atom1_local)
                    valid_bonds += 1
            
            print(f"   Valid bonds within molecule: {valid_bonds}")
            
            # Find rings using bond topology with local indices
            visited_global = set()
            
            for local_idx in range(len(molecule)):
                if local_idx in visited_global:
                    continue
                    
                # Look for rings starting from this atom
                rings = self._find_rings_dfs(local_idx, adjacency, max_size=8)
                
                for ring_indices in rings:
                    # Filter for likely aromatic rings (5-7 atoms, mostly carbons)
                    ring_atoms = [molecule[ix] for ix in ring_indices]  # Use local indexing
                    print(f"   Found ring candidate: {len(ring_atoms)} atoms")
                    if self._is_likely_aromatic_ring(ring_atoms):
                        print(f"   Ring passes aromatic test")
                        aromatic_rings.append(ring_atoms)
                        visited_global.update(ring_indices)
                    else:
                        print(f"   Ring fails aromatic test")
                        
        except Exception as e:
            print(f"Warning in bond-based ring detection: {e}")
            import traceback
            traceback.print_exc()
            
        return aromatic_rings

    def _find_rings_dfs(self, start_atom, adjacency, max_size=8):
        """
        Find rings using depth-first search on the bond graph.
        """
        rings = []
        
        def dfs(current, path, visited):
            if len(path) > max_size:
                return
                
            if len(path) > 2 and start_atom in adjacency[current]:
                # Found a ring
                ring = path + [current]
                if 5 <= len(ring) <= 7:  # Common aromatic ring sizes
                    rings.append(ring)
                return
            
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                if len(path) > 1 and neighbor == path[-2]:  # Don't go back immediately
                    continue
                    
                new_visited = visited.copy()
                new_visited.add(current)
                dfs(neighbor, path + [current], new_visited)
        
        dfs(start_atom, [], {start_atom})
        return rings

    def _is_likely_aromatic_ring(self, ring_atoms):
        """
        Check if a ring is likely aromatic (quinolone rings only, no piperazine).
        """
        # print(f"🔍 DEBUG: Checking if ring is aromatic ({len(ring_atoms)} atoms)")
        
        if len(ring_atoms) < 5 or len(ring_atoms) > 7:
            print(f"   ❌ Wrong size: {len(ring_atoms)} atoms")
            return False
        
        # Get atom names and elements
        atom_names = [atom.name for atom in ring_atoms]
        atom_elements = []
        for atom in ring_atoms:
            if hasattr(atom, 'element') and atom.element:
                atom_elements.append(atom.element)
            else:
                # Try to guess element from atom name
                name = atom.name.upper()
                if name.startswith('C'):
                    atom_elements.append('C')
                elif name.startswith('N'):
                    atom_elements.append('N')
                elif name.startswith('O'):
                    atom_elements.append('O')
                else:
                    atom_elements.append('C')  # Default to carbon
        
        # print(f"   Atom names: {atom_names}")
        # print(f"   Atom elements: {atom_elements}")
        
        # Exclude piperazine ring (6-membered ring with 2 nitrogens - not aromatic)
        nitrogen_count = sum(1 for elem in atom_elements if elem == 'N')
        carbon_count = sum(1 for elem in atom_elements if elem == 'C')
        
        if len(ring_atoms) == 6 and nitrogen_count == 2 and carbon_count == 4:
            # print(f"   ❌ Piperazine ring detected - not aromatic, skipping")
            return False
        
        # Check for mostly carbon/nitrogen atoms (typical for aromatic rings)
        aromatic_elements = ['C', 'N', 'O', 'S']
        element_count = sum(1 for elem in atom_elements if elem in aromatic_elements)
        
        # print(f"   Aromatic elements: {element_count}/{len(ring_atoms)}")
        
        if element_count < len(ring_atoms) * 0.8:  # At least 80% aromatic elements
            # print(f"   ❌ Not enough aromatic elements")
            return False
            
        # Check planarity (strict for truly aromatic rings)
        positions = np.array([atom.position for atom in ring_atoms])
        is_planar = self._is_planar_ring_simple(positions, relaxed=False)
        
        # print(f"   Planar: {is_planar}")
        
        result = is_planar
        # print(f"   ✓ Ring passes aromatic test: {result}")
        return result

    def _is_planar_ring_simple(self, positions, relaxed=False):
        """
        Simple planarity check using variance in the perpendicular direction.
        """
        if len(positions) < 4:
            return True
            
        # Calculate centroid
        centroid = np.mean(positions, axis=0)
        
        # Get vectors from centroid
        vectors = positions - centroid
        
        # Calculate covariance matrix and get principal components
        cov_matrix = np.cov(vectors.T)
        eigenvals, eigenvects = np.linalg.eigh(cov_matrix)
        
        # The smallest eigenvalue indicates out-of-plane variance
        # For planar rings, this should be small
        planarity_threshold = 1.0 if relaxed else 0.25  # More relaxed for saturated rings
        return eigenvals[0] < planarity_threshold

    def _detect_cip_aromatic_rings(self, molecule):
        """
        CIP-specific aromatic ring detection based on known structure.
        """
        aromatic_rings = []
        
        try:
            # print(f"🔍 DEBUG: _detect_cip_aromatic_rings called")
            print(f"   Molecule has {len(molecule)} atoms")
            
            # For CIP (ciprofloxacin), we know the structure:
            # 1. Quinolone ring system (two fused rings)
            # We'll identify them by carbon atoms in the right positions
            
            carbon_atoms = molecule.select_atoms("name C*")
            print(f"   Carbon atoms found: {len(carbon_atoms)}")
            
            if len(carbon_atoms) == 0:
                # Try alternative carbon naming
                carbon_atoms = molecule.select_atoms("element C")
                print(f"   Carbon atoms (by element): {len(carbon_atoms)}")
                
            if len(carbon_atoms) == 0:
                print(f"   ⚠️  No carbon atoms found")
                return aromatic_rings
            
            print(f"   Sample carbon names: {[atom.name for atom in carbon_atoms[:10]]}")
            
            # Group carbons that are close to each other (ring formation)
            from scipy.spatial import cKDTree
            positions = carbon_atoms.positions
            tree = cKDTree(positions)
            
            # Find connected components of carbons
            visited = set()
            ring_candidates = []
            
            for i, atom in enumerate(carbon_atoms):
                if i in visited:
                    continue
                    
                # Find all carbons within bonding distance
                neighbors = tree.query_ball_point(atom.position, 1.6)  # Typical C-C bond length
                
                # Build a connected component
                component = set()
                to_visit = [i]
                
                while to_visit:
                    current = to_visit.pop()
                    if current in component:
                        continue
                    component.add(current)
                    
                    current_neighbors = tree.query_ball_point(positions[current], 1.6)
                    for neighbor in current_neighbors:
                        if neighbor not in component and neighbor not in visited:
                            to_visit.append(neighbor)
                
                if len(component) >= 5:  # Potential ring
                    ring_atoms = [carbon_atoms[j] for j in component]
                    print(f"   Found component of {len(component)} carbons")
                    if self._is_likely_aromatic_ring(ring_atoms):
                        print(f"   Component passes aromatic test")
                        ring_candidates.append(ring_atoms)
                        visited.update(component)
                    else:
                        print(f"   Component failed aromatic test")
            
            print(f"   Ring candidates found: {len(ring_candidates)}")
            
            # For CIP, limit to the two main aromatic rings
            # Sort by size and take the two largest reasonable rings
            ring_candidates.sort(key=len, reverse=True)
            for i, ring in enumerate(ring_candidates[:2]):  # Maximum 2 rings for CIP
                if 5 <= len(ring) <= 8:  # Reasonable ring size
                    print(f"   Adding ring {i+1}: {len(ring)} atoms")
                    aromatic_rings.append(ring)
                else:
                    print(f"   Skipping ring {i+1}: {len(ring)} atoms (wrong size)")
                    
        except Exception as e:
            print(f"Warning in CIP-specific ring detection: {e}")
            import traceback
            traceback.print_exc()
            
        return aromatic_rings

    def _calculate_ring_geometry_topology(self, ring_atoms):
        """
        Calculate ring geometry using actual atomic positions from topology.
        
        Parameters
        ----------
        ring_atoms : list
            List of MDAnalysis atoms in the ring
            
        Returns
        -------
        tuple
            (center, normal, radius) as numpy arrays and float
        """
        # Get actual positions from the atoms
        positions = np.array([atom.position for atom in ring_atoms])
        
        # Calculate ring center (centroid)
        center = np.mean(positions, axis=0)
        
        # Calculate ring normal using PCA for better accuracy
        centered_positions = positions - center
        
        # Use SVD to find the best-fit plane
        _, _, vh = np.linalg.svd(centered_positions)
        normal = vh[-1]  # Last row is the normal to the best-fit plane
        
        # Ensure normal points in a consistent direction
        if normal[2] < 0:  # Point upward
            normal = -normal
        
        # Calculate average distance from center as radius
        distances = np.linalg.norm(centered_positions, axis=1)
        radius = np.mean(distances)
        
        return center, normal, radius

    def _create_wireframe_extent_markers(self, view, center, radius, theta_min, theta_max, phi_min, phi_max, color, alpha):
        """
        This method should not be used anymore - keeping for compatibility.
        """
        # Just call the mesh method instead
        self._create_partial_sphere_mesh(view, center, radius, theta_min, theta_max, phi_min, phi_max, color, alpha)
