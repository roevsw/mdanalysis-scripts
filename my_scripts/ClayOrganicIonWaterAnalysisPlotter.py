"""
ClayOrganicIonWaterAnalysisPlotter.py

Separate plotting class for ClayOrganicIonWaterAnalysis results.
Handles all visualization and plotting functionality for clay-organic-ion-water systems.

Author: R.Swai
Date: January 2026
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


class ClayOrganicIonWaterAnalysisPlotter:
    """
    Plotting class for ClayOrganicIonWaterAnalysis results.
    
    This class handles visualization of:
    - Multi-component radial distribution functions
    - Competitive adsorption analysis
    - Organic molecule conformations
    - Three-component bridge structures
    - Hydration shell competition
    - Stratified (layered) adsorption profiles
    - Exchange kinetics and residence times
    - Selectivity coefficients
    
    Parameters
    ----------
    analysis : ClayOrganicIonWaterAnalysis
        Instance of ClayOrganicIonWaterAnalysis class (optional)
    
    Examples
    --------
    >>> from ClayOrganicIonWaterAnalysis import ClayOrganicIonWaterAnalysis
    >>> from ClayOrganicIonWaterAnalysisPlotter import ClayOrganicIonWaterAnalysisPlotter
    >>> 
    >>> analysis = ClayOrganicIonWaterAnalysis('traj.xtc', 'topol.tpr', ...)
    >>> analysis.run_full_analysis()
    >>> 
    >>> plotter = ClayOrganicIonWaterAnalysisPlotter(analysis)
    >>> plotter.plot_multi_component_rdfs()
    >>> plotter.plot_competitive_adsorption()
    """
    
    def __init__(self, analysis=None):
        """
        Initialize plotter
        
        Parameters
        ----------
        analysis : ClayOrganicIonWaterAnalysis, optional
            ClayOrganicIonWaterAnalysis instance for accessing results
        """
        self.analysis = analysis
        
        # Default plotting style
        self.set_default_style()
        
        # Default plotting parameters
        self.default_figsize = (12, 8)
        self.default_dpi = 300
        self.default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'olive']
    
    def set_default_style(self):
        """Set default matplotlib style for publication-quality plots"""
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 14
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['axes.linewidth'] = 1.2
    
    def set_analysis(self, analysis):
        """Set or update the analysis object"""
        self.analysis = analysis
    
    def _validate_analysis(self):
        """Check if analysis object is set and has results"""
        if self.analysis is None:
            raise ValueError("No analysis object set. Use set_analysis() or provide analysis in __init__")
        if not hasattr(self.analysis, 'results'):
            raise ValueError("Analysis object has no results. Run analysis first.")
    
    # =========================================================================
    # MULTI-COMPONENT RDF PLOTTING
    # =========================================================================
    
    def plot_multiple_rdfs(self, rdf_dict, title='RDF Comparison',
                          xlabel='Distance (Å)', ylabel='g(r)',
                          xlim=None, ylim=None, 
                          # Color control
                          colors=None, colormap='tab10',
                          # Line styling
                          linewidth=2, linestyles=None, line_alpha=1.0, alphas=None,
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
                          # RDF filtering
                          skip_individual_atoms=True,
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
            - Dict: colors mapped by label {'Ob-OW': 'red', 'Si-OW': 'blue'}
            If None, uses colormap
        colormap : str
            Matplotlib colormap name for auto-generating colors (default: 'tab10')
        
        Line Styling
        ------------
        linewidth : float or list
            Line width(s). Single value or list per RDF (default: 2)
        linestyles : list or dict, optional
            Line styles. Can be:
            - List: styles applied in order ['-', '--', '-.', ':']
            - Dict: styles mapped by label {'Ob-OW': '-', 'Si-OW': '--'}
            (default: None, all solid '-')
        line_alpha : float
            Default line transparency 0-1 for all lines (default: 1.0)
        alphas : dict, optional
            Per-line transparency: {'Ob-OW': 0.5, 'Si-OW': 0.3}
            Overrides line_alpha for specific curves
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
            Use mathtext for superscripts: {'Ob-OW': r'Ob-O$_w$'}
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
        
        RDF Filtering
        -------------
        skip_individual_atoms : bool
            Whether to automatically skip individual atom RDFs (those with _parent_group attribute).
            When True (default), only plots grouped RDFs from store_per_atom=True calculations.
            Set to False to plot all RDFs including individual atoms. (default: True)
        
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
        >>> rdf_dict = {'Ob-OW': rdf1, 'Si-OW': rdf2}
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
        
        >>> # Custom legend labels with formatting
        >>> plotter.plot_multiple_rdfs(
        ...     rdf_dict,
        ...     custom_labels={
        ...         'Ob-OW': r'O$_b$-O$_w$',
        ...         'Si-OW': r'Si-O$_w$',
        ...         'Mgo-OW': r'Mg$_o$-O$_w$'
        ...     },
        ...     save_fig=True
        ... )
        """
        
        # Filter out individual atoms if requested (default behavior)
        if skip_individual_atoms:
            filtered_dict = {}
            for label, rdf_result in rdf_dict.items():
                # Skip RDFs that have _parent_group attribute (individual atoms)
                if not hasattr(rdf_result, '_parent_group'):
                    filtered_dict[label] = rdf_result
            
            # If filtering removed everything, warn user
            if not filtered_dict:
                print("⚠️  Warning: skip_individual_atoms=True removed all RDFs!")
                print("   Set skip_individual_atoms=False to plot all RDFs.")
                filtered_dict = rdf_dict  # Use original dict as fallback
            elif len(filtered_dict) < len(rdf_dict):
                print(f"📊 Filtered: {len(filtered_dict)} grouped RDFs (skipped {len(rdf_dict) - len(filtered_dict)} individual atoms)")
            
            rdf_dict = filtered_dict
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate colors if not provided
        if colors is None:
            cmap = plt.cm.get_cmap(colormap)
            colors = [cmap(i % cmap.N) for i in range(len(rdf_dict))]
        elif isinstance(colors, dict):
            # Convert dictionary to list in the order of rdf_dict
            # Support partial matching for keys
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
        elif isinstance(linestyles, dict):
            # Convert dictionary to list in the order of rdf_dict
            converted_linestyles = []
            for label in rdf_dict.keys():
                if label in linestyles:
                    converted_linestyles.append(linestyles[label])
                else:
                    converted_linestyles.append('-')  # Default to solid
            linestyles = converted_linestyles
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
        
        # Handle alphas as dict or use line_alpha default
        if alphas is None:
            alpha_values = [line_alpha] * n_curves
        else:
            # Convert dict to list in the order of rdf_dict
            alpha_values = []
            for label in rdf_dict.keys():
                if label in alphas:
                    alpha_values.append(alphas[label])
                else:
                    alpha_values.append(line_alpha)  # Use default for unspecified
        
        # Plot each RDF
        for idx, (label, rdf_results) in enumerate(rdf_dict.items()):
            # Extract data - support both object and dict formats
            if hasattr(rdf_results, 'bins'):
                bins = rdf_results.bins
                rdf = rdf_results.rdf
            elif 'bins' in rdf_results:
                bins = rdf_results['bins']
                rdf = rdf_results['rdf']
            else:
                print(f"⚠ Warning: Skipping '{label}' - invalid RDF format")
                continue
            
            # Get styling for this curve
            color = colors[idx] if isinstance(colors, (list, np.ndarray)) else colors
            ls = linestyles[idx]
            lw = linewidths[idx]
            alpha = alpha_values[idx]
            marker = markers[idx] if markers is not None else None
            
            # Get display label (use custom if provided)
            display_label = custom_labels.get(label, label) if custom_labels else label
            
            # Plot line
            ax.plot(bins, rdf, label=display_label, color=color, linewidth=lw,
                   linestyle=ls, alpha=alpha,
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
    
    def plot_multiple_rdfs_with_atoms(self, rdf_dict, 
                                       plot_mode='grouped',
                                       group_colors=None,
                                       clay_linestyles=None,
                                       group_alphas=None,
                                       alphas=None,
                                       grouped_linewidth=None,
                                       individual_linewidth=None,
                                       show_std_band=False,
                                       show_individual_in_legend=True,
                                       dual_legend=False,
                                       dual_legend_layout='auto',
                                       legend_outside=False,
                                       **kwargs):
        """
        Plot RDFs with support for per-atom detail when store_per_atom=True was used.
        
        Parameters
        ----------
        rdf_dict : dict
            Dictionary from molecular_rdf with store_per_atom=True
        plot_mode : str, default='grouped'
            'grouped': Show only grouped RDFs (e.g., 'Ob-piperazine')
            'individual': Show all individual atom RDFs (e.g., 'Ob-N13', 'Ob-N16')
            'both': Show both individual atoms + bold grouped average
        group_colors : dict, optional
            Base colors for functional groups: {'piperazine': 'blue', 'quinolone': 'green'}
            Individual atoms get color gradients from base color
        clay_linestyles : dict, optional
            Linestyles for clay components: {'Ob': '-', 'Si': '--', 'Mgo': '-.', 'H_Ohmg': ':'}
            Applied to all RDFs starting with that clay component (e.g., 'Ob-carboxylic_acid')
            Allows distinguishing clay components when using same colors for functional groups
        group_alphas : dict, optional
            Transparency for functional groups: {'quinolone': 0.3, 'piperazine': 1.0}
            Applied to all atoms in that group. Default: 1.0 (opaque)
        alphas : dict, optional
            Per-line transparency: {'Ob-N6': 0.2, 'Si-N6': 0.4}
            Overrides group_alphas for specific curves
        grouped_linewidth : float, optional
            Linewidth for grouped RDFs when plot_mode='both'. Default: 1.5 * kwargs['linewidth']
        individual_linewidth : float, optional
            Linewidth for individual atom RDFs when plot_mode='both'. Default: 0.7 * kwargs['linewidth']
        show_std_band : bool, default=False
            Show shaded standard deviation band around grouped RDF curves.
            Only works with plot_mode='grouped' or 'both'. Requires individual atom RDFs.
            The band shows mean ± std of all individual atoms in each functional group.
            Useful for visualizing uncertainty/spread in grouped averages.
        show_individual_in_legend : bool, default=True
            Whether to show individual atom RDFs in the legend.
            When False (with plot_mode='both'), only grouped RDFs appear in legend.
            Reduces legend clutter from 88 entries to 12 entries.
        dual_legend : bool, default=False
            Create two separate legends: one for colors (functional groups) and one for
            linestyles (clay components). Reduces 12 entries → 7 entries (3 colors + 4 styles).
            Only works when clay_linestyles is provided.
        dual_legend_layout : str, default='auto'
            Layout for dual legends. Options:
            'auto' - Automatically find best locations for both legends
            'vertical' - Stack legends vertically (one above the other)
            'horizontal' - Place legends side by side
            Only used when dual_legend=True.
        legend_outside : bool, default=False
            If True, place legend outside the plot area on the right.
            If False, legend finds best location inside plot.
        **kwargs : dict
            Additional arguments passed to plot_multiple_rdfs
            
        Returns
        -------
        fig, ax : matplotlib figure and axes
        
        Examples
        --------
        >>> # Calculate with per-atom storage
        >>> rdf_api = analysis.molecular_rdf(
        ...     group1_sel=[surface_oxygen],
        ...     group2_sel=[piperazine, quinolone],
        ...     store_per_atom=True
        ... )
        >>> 
        >>> # Plot individual atoms with auto-generated gradients
        >>> plotter.plot_multiple_rdfs_with_atoms(
        ...     rdf_api,
        ...     plot_mode='individual',
        ...     group_colors={'piperazine': 'blue', 'quinolone': 'green'},
        ...     clay_linestyles={'Ob': '-', 'Si': '--', 'Mgo': '-.', 'H_Ohmg': ':'},
        ...     group_alphas={'quinolone': 0.3, 'piperazine': 1.0},  # Fade quinolone
        ...     alphas={'Ob-N6': 0.5}  # Override specific curve
        ... )
        >>> 
        >>> # Plot both with separate linewidths for clarity
        >>> plotter.plot_multiple_rdfs_with_atoms(
        ...     rdf_api,
        ...     plot_mode='both',
        ...     group_colors={'piperazine': 'blue', 'quinolone': 'green'},
        ...     grouped_linewidth=3.0,  # Thick lines for grouped
        ...     individual_linewidth=1.0,  # Thin lines for individual atoms
        ...     show_std_band=True,  # Show uncertainty bands around grouped
        ...     fill_alpha=0.15  # Control band transparency
        ... )
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.cm import get_cmap
        
        # Separate grouped and individual RDFs
        grouped_rdfs = {}
        individual_rdfs = {}
        parent_mapping = {}  # Maps atom name to parent group
        
        for label, rdf_result in rdf_dict.items():
            # Check if this is an individual atom RDF (has _parent_group attribute)
            if hasattr(rdf_result, '_parent_group'):
                individual_rdfs[label] = rdf_result
                # Extract atom name from label (e.g., 'Ob-N13' -> 'N13')
                atom_name = label.split('-')[-1]
                parent_mapping[label] = rdf_result._parent_group
            else:
                grouped_rdfs[label] = rdf_result
        
        # Determine what to plot based on mode
        if plot_mode == 'grouped':
            plot_dict = grouped_rdfs
            print(f"📊 Plotting {len(plot_dict)} grouped RDFs")
            
            # Process group_colors, clay_linestyles, and group_alphas for grouped mode
            if group_colors:
                # Map grouped labels to colors based on functional group
                colors_dict = {}
                for label in grouped_rdfs.keys():
                    func_group = label.split('-')[-1]  # Extract functional group from label
                    colors_dict[label] = group_colors.get(func_group, 'blue')
                kwargs['colors'] = colors_dict
            
            # Apply clay_linestyles if provided
            if clay_linestyles:
                linestyles_dict = self._generate_clay_linestyles(
                    grouped_rdfs.keys(),
                    clay_linestyles
                )
                kwargs['linestyles'] = linestyles_dict
            
            # Note: group_alphas is NOT applied in 'grouped' mode
            # group_alphas only affects individual atoms in 'individual' or 'both' modes
            # Grouped RDFs are always fully opaque
            # Only apply alphas if user explicitly provided per-line alphas
            if alphas:
                kwargs['alphas'] = alphas
            
            # Apply grouped_linewidth if provided
            if grouped_linewidth is not None:
                kwargs['linewidth'] = grouped_linewidth
            
            # Save original save_fig setting and disable it temporarily
            original_save_fig = kwargs.get('save_fig', False)
            original_filename = kwargs.get('filename', 'rdf_comparison.png')
            original_dpi = kwargs.get('dpi', 300)
            original_bbox_inches = kwargs.get('bbox_inches', 'tight')
            original_transparent_bg = kwargs.get('transparent_bg', False)
            
            # Disable saving in plot_multiple_rdfs if we need to add std bands or reposition legend
            if (show_std_band and individual_rdfs) or legend_outside:
                kwargs['save_fig'] = False
            
            # Plot the RDFs
            fig, ax = self.plot_multiple_rdfs(plot_dict, **kwargs)
            
            # Add standard deviation bands if requested
            if show_std_band and individual_rdfs:
                # Use the colors we just generated
                grouped_colors = kwargs.get('colors', {})
                self._add_std_bands(ax, grouped_rdfs, individual_rdfs, parent_mapping,
                                   grouped_colors, kwargs.get('fill_alpha', 0.2))
            
            # Reposition legend outside if requested
            if legend_outside and kwargs.get('show_legend', True):
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    legend = ax.legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left',
                             fontsize=kwargs.get('legend_fontsize', 10),
                             ncol=kwargs.get('legend_ncol', 1),
                             framealpha=kwargs.get('legend_framealpha', 0.8))
                    # Apply font weight
                    for text in legend.get_texts():
                        text.set_fontweight(kwargs.get('legend_fontweight', 'normal'))
            
            # Save the figure with all modifications (std bands + repositioned legend)
            if original_save_fig:
                import matplotlib.pyplot as plt
                plt.savefig(original_filename, dpi=original_dpi, 
                           bbox_inches=original_bbox_inches,
                           transparent=original_transparent_bg)
                print(f"✓ Figure saved: {original_filename}")
            
            return fig, ax
        
        elif plot_mode == 'individual':
            plot_dict = individual_rdfs
            print(f"📊 Plotting {len(plot_dict)} individual atom RDFs")
            
            # Generate color gradients for each functional group
            if group_colors is None:
                group_colors = self._auto_assign_group_colors(parent_mapping)
            
            colors_dict = self._generate_atom_color_gradients(
                individual_rdfs.keys(),
                parent_mapping,
                group_colors
            )
            
            # Generate alpha values
            alphas_dict = self._generate_alphas(
                individual_rdfs.keys(),
                parent_mapping,
                group_alphas,
                alphas
            )
            
            # Generate linestyles based on clay component
            if clay_linestyles:
                linestyles_dict = self._generate_clay_linestyles(
                    individual_rdfs.keys(),
                    clay_linestyles
                )
                kwargs['linestyles'] = linestyles_dict
            
            # Apply individual_linewidth if specified
            if individual_linewidth is not None:
                kwargs['linewidth'] = individual_linewidth
            
            # Update kwargs with generated colors and alphas
            kwargs['colors'] = colors_dict
            if alphas_dict:
                kwargs['alphas'] = alphas_dict
            
            # Handle dual legend or legend positioning
            if dual_legend and clay_linestyles:
                kwargs['show_legend'] = False  # Disable default legend
            
            fig, ax = self.plot_multiple_rdfs(plot_dict, **kwargs)
            
            # Create dual legend if requested
            if dual_legend and clay_linestyles:
                self._create_dual_legend(ax, group_colors, clay_linestyles, legend_outside, dual_legend_layout, kwargs)
            elif legend_outside and kwargs.get('show_legend', True):
                # Move existing legend outside
                legend = ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left',
                         fontsize=kwargs.get('legend_fontsize', 10),
                         ncol=kwargs.get('legend_ncol', 1),
                         framealpha=kwargs.get('legend_framealpha', 0.8))
                # Apply font weight
                for text in legend.get_texts():
                    text.set_fontweight(kwargs.get('legend_fontweight', 'normal'))
            
            return fig, ax
        
        elif plot_mode == 'both':
            # Plot both individual + grouped with special styling
            print(f"📊 Plotting {len(individual_rdfs)} individual + {len(grouped_rdfs)} grouped RDFs")
            
            # Create combined dictionary with metadata for special styling
            plot_dict = {**individual_rdfs, **grouped_rdfs}
            
            # Generate colors
            if group_colors is None:
                group_colors = self._auto_assign_group_colors(parent_mapping)
            
            atom_colors = self._generate_atom_color_gradients(
                individual_rdfs.keys(),
                parent_mapping,
                group_colors
            )
            
            # Grouped RDFs get bold colors
            grouped_colors = {label: group_colors.get(label.split('-')[-1], 'black') 
                            for label in grouped_rdfs.keys()}
            
            colors_dict = {**atom_colors, **grouped_colors}
            kwargs['colors'] = colors_dict
            
            # Generate alpha values (for individual atoms)
            alphas_dict = self._generate_alphas(
                individual_rdfs.keys(),
                parent_mapping,
                group_alphas,
                alphas
            )
            
            # Grouped RDFs always fully opaque
            for label in grouped_rdfs.keys():
                alphas_dict[label] = 1.0
            
            if alphas_dict:
                kwargs['alphas'] = alphas_dict
            
            # Generate linestyles based on clay component
            if clay_linestyles:
                linestyles_dict = self._generate_clay_linestyles(
                    plot_dict.keys(),
                    clay_linestyles
                )
                kwargs['linestyles'] = linestyles_dict
            
            # Make grouped lines thicker with configurable widths
            base_linewidth = kwargs.get('linewidth', 2)
            
            # Use provided values or defaults
            grouped_lw = grouped_linewidth if grouped_linewidth is not None else base_linewidth * 1.5
            individual_lw = individual_linewidth if individual_linewidth is not None else base_linewidth * 0.7
            
            linewidths = {}
            for label in plot_dict.keys():
                if label in grouped_rdfs:
                    linewidths[label] = grouped_lw
                else:
                    linewidths[label] = individual_lw
            
            # Convert to list in order
            lw_list = [linewidths.get(label, base_linewidth) for label in plot_dict.keys()]
            kwargs['linewidth'] = lw_list
            
            # Handle legend filtering - hide individual atoms from legend if requested
            if not show_individual_in_legend:
                # Store original labels
                original_labels = {}
                for label in individual_rdfs.keys():
                    original_labels[label] = label
                    # Set individual atom labels to empty string to hide from legend
                    if 'custom_labels' not in kwargs:
                        kwargs['custom_labels'] = {}
                    kwargs['custom_labels'][label] = '_nolegend_'
            
            # Handle dual legend
            if dual_legend and clay_linestyles:
                kwargs['show_legend'] = False  # Disable default legend
            
            # Save original save_fig setting and disable it temporarily
            original_save_fig = kwargs.get('save_fig', False)
            original_filename = kwargs.get('filename', 'rdf_comparison.png')
            original_dpi = kwargs.get('dpi', 300)
            original_bbox_inches = kwargs.get('bbox_inches', 'tight')
            original_transparent_bg = kwargs.get('transparent_bg', False)
            
            # Disable saving in plot_multiple_rdfs if we need to add std bands or legends
            if show_std_band or (dual_legend and clay_linestyles) or not show_individual_in_legend or legend_outside:
                kwargs['save_fig'] = False
            
            # Plot the RDFs
            fig, ax = self.plot_multiple_rdfs(plot_dict, **kwargs)
            
            # Add standard deviation bands if requested
            if show_std_band:
                self._add_std_bands(ax, grouped_rdfs, individual_rdfs, parent_mapping, 
                                   grouped_colors, kwargs.get('fill_alpha', 0.2))
            
            # Create dual legend if requested
            if dual_legend and clay_linestyles:
                self._create_dual_legend(ax, group_colors, clay_linestyles, legend_outside, dual_legend_layout, kwargs)
            elif not show_individual_in_legend or legend_outside:
                # Recreate legend with filtered labels or position outside
                handles, labels = ax.get_legend_handles_labels()
                # Filter out '_nolegend_' entries
                filtered = [(h, l) for h, l in zip(handles, labels) if not l.startswith('_nolegend_')]
                if filtered:
                    handles, labels = zip(*filtered)
                    if legend_outside:
                        legend = ax.legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left',
                                 fontsize=kwargs.get('legend_fontsize', 10),
                                 ncol=kwargs.get('legend_ncol', 1),
                                 framealpha=kwargs.get('legend_framealpha', 0.8))
                    else:
                        legend = ax.legend(handles, labels, loc=kwargs.get('legend_loc', 'best'),
                                 fontsize=kwargs.get('legend_fontsize', 10),
                                 ncol=kwargs.get('legend_ncol', 1),
                                 framealpha=kwargs.get('legend_framealpha', 0.8))
                    # Apply font weight
                    for text in legend.get_texts():
                        text.set_fontweight(kwargs.get('legend_fontweight', 'normal'))
            
            # Now save the figure with all modifications (std bands + legends)
            if original_save_fig:
                import matplotlib.pyplot as plt
                plt.savefig(original_filename, dpi=original_dpi, 
                           bbox_inches=original_bbox_inches,
                           transparent=original_transparent_bg)
                print(f"✓ Figure saved: {original_filename}")
            
            return fig, ax
        
        else:
            raise ValueError(f"plot_mode must be 'grouped', 'individual', or 'both', got '{plot_mode}'")
    
    def _auto_assign_group_colors(self, parent_mapping):
        """Auto-assign colors to functional groups"""
        unique_groups = set(parent_mapping.values())
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        return {group: colors[i % len(colors)] for i, group in enumerate(sorted(unique_groups))}
    
    def _create_dual_legend(self, ax, group_colors, clay_linestyles, legend_outside, dual_legend_layout, kwargs):
        """Create two separate legends: one for colors (functional groups), one for linestyles (clay components)"""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        
        # Create color legend (functional groups)
        color_handles = []
        for group, color in sorted(group_colors.items()):
            color_handles.append(Line2D([0], [0], color=color, lw=2, label=group))
        
        # Create linestyle legend (clay components)
        linestyle_handles = []
        for clay, style in sorted(clay_linestyles.items()):
            linestyle_handles.append(Line2D([0], [0], color='black', linestyle=style, lw=2, label=clay))
        
        fontsize = kwargs.get('legend_fontsize', 10)
        framealpha = kwargs.get('legend_framealpha', 0.8)
        fontweight = kwargs.get('legend_fontweight', 'normal')
        
        if legend_outside:
            # Place both legends outside, stacked vertically
            legend1 = ax.legend(handles=color_handles, title='Functional Group',
                              bbox_to_anchor=(1.05, 1.0), loc='upper left',
                              fontsize=fontsize, framealpha=framealpha,
                              title_fontsize=fontsize)
            ax.add_artist(legend1)  # Keep first legend when adding second
            
            legend2 = ax.legend(handles=linestyle_handles, title='Clay Component',
                              bbox_to_anchor=(1.05, 0.5), loc='center left',
                              fontsize=fontsize, framealpha=framealpha,
                              title_fontsize=fontsize)
        else:
            # Determine legend positions based on layout
            if dual_legend_layout == 'vertical':
                # Stack vertically - one above the other on right side
                loc1 = 'upper right'
                loc2 = 'center right'
                legend1 = ax.legend(handles=color_handles, title='Functional Group',
                                  loc=loc1, fontsize=fontsize, framealpha=framealpha,
                                  title_fontsize=fontsize)
                ax.add_artist(legend1)  # Keep first legend when adding second
                
                legend2 = ax.legend(handles=linestyle_handles, title='Clay Component',
                                  loc=loc2, fontsize=fontsize, framealpha=framealpha,
                                  title_fontsize=fontsize)
            elif dual_legend_layout == 'horizontal':
                # Side by side inside the plot
                loc1 = 'upper left'
                loc2 = 'upper right'
                legend1 = ax.legend(handles=color_handles, title='Functional Group',
                                  loc=loc1, fontsize=fontsize, framealpha=framealpha,
                                  title_fontsize=fontsize)
                ax.add_artist(legend1)  # Keep first legend when adding second
                
                legend2 = ax.legend(handles=linestyle_handles, title='Clay Component',
                                  loc=loc2, fontsize=fontsize, framealpha=framealpha,
                                  title_fontsize=fontsize)
            else:  # 'auto' - let matplotlib find best positions
                loc1 = 'best'
                loc2 = 'best'
                legend1 = ax.legend(handles=color_handles, title='Functional Group',
                                  loc=loc1, fontsize=fontsize, framealpha=framealpha,
                                  title_fontsize=fontsize)
                ax.add_artist(legend1)  # Keep first legend when adding second
                
                legend2 = ax.legend(handles=linestyle_handles, title='Clay Component',
                                  loc=loc2, fontsize=fontsize, framealpha=framealpha,
                                  title_fontsize=fontsize)
        
        # Apply font weight to both legend texts AND titles
        for text in legend1.get_texts():
            text.set_fontweight(fontweight)
        for text in legend2.get_texts():
            text.set_fontweight(fontweight)
        legend1.get_title().set_fontweight(fontweight)
        legend2.get_title().set_fontweight(fontweight)
    
    def _generate_atom_color_gradients(self, atom_labels, parent_mapping, group_colors):
        """Generate color gradients for atoms within same functional group"""
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba, to_hex
        import numpy as np
        
        # Group atoms by parent functional group
        groups = {}
        for label in atom_labels:
            parent = parent_mapping.get(label, 'unknown')
            if parent not in groups:
                groups[parent] = []
            groups[parent].append(label)
        
        # Generate gradients for each group
        color_dict = {}
        for parent, labels in groups.items():
            base_color = group_colors.get(parent, 'blue')
            n_atoms = len(labels)
            
            if n_atoms == 1:
                color_dict[labels[0]] = base_color
            else:
                # Create color gradient using lightness variation
                base_rgba = to_rgba(base_color)
                
                # Generate colors from dark to light
                for i, label in enumerate(sorted(labels)):
                    # Vary lightness: darkest (0.6) to lightest (1.4)
                    factor = 0.6 + (0.8 * i / (n_atoms - 1))
                    
                    # Adjust RGB channels
                    adjusted = tuple(min(1.0, c * factor) if j < 3 else c 
                                   for j, c in enumerate(base_rgba))
                    
                    color_dict[label] = adjusted
        
        return color_dict
    
    def _generate_clay_linestyles(self, labels, clay_linestyles):
        """Generate linestyles based on clay component (first part of label)
        
        Parameters
        ----------
        labels : iterable
            RDF labels (e.g., 'Ob-N13', 'Si-carboxylic_acid', 'Mgo-C14')
        clay_linestyles : dict
            Mapping of clay component to linestyle: {'Ob': '-', 'Si': '--', 'Mgo': '-.', 'H_Ohmg': ':'}
        
        Returns
        -------
        dict
            Mapping of label to linestyle
        """
        linestyles_dict = {}
        
        for label in labels:
            # Extract clay component (first part before dash)
            if '-' in label:
                clay_component = label.split('-')[0]
                linestyle = clay_linestyles.get(clay_component, '-')  # Default to solid
                linestyles_dict[label] = linestyle
            else:
                linestyles_dict[label] = '-'  # Default
        
        return linestyles_dict
    
    def _generate_alphas(self, labels, parent_mapping, group_alphas=None, alphas=None):
        """Generate alpha (transparency) values for RDF curves
        
        Parameters
        ----------
        labels : iterable
            RDF labels (e.g., 'Ob-N13', 'Si-C14')
        parent_mapping : dict
            Maps labels to parent functional groups
        group_alphas : dict, optional
            Group-level alpha: {'quinolone': 0.3, 'piperazine': 1.0}
        alphas : dict, optional
            Per-line alpha: {'Ob-N6': 0.5}
            Overrides group_alphas
        
        Returns
        -------
        dict
            Mapping of label to alpha value
        """
        if group_alphas is None and alphas is None:
            return {}  # Use default (1.0)
        
        alphas_dict = {}
        
        # First apply group-level alphas
        if group_alphas:
            for label in labels:
                parent = parent_mapping.get(label)
                if parent in group_alphas:
                    alphas_dict[label] = group_alphas[parent]
        
        # Then override with per-line alphas
        if alphas:
            for label, alpha_val in alphas.items():
                if label in labels:
                    alphas_dict[label] = alpha_val
        
        return alphas_dict
    
    def _add_std_bands(self, ax, grouped_rdfs, individual_rdfs, parent_mapping, 
                      grouped_colors, fill_alpha=0.2):
        """Add standard deviation bands around grouped RDF curves
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to add bands to
        grouped_rdfs : dict
            Dictionary of grouped RDF results
        individual_rdfs : dict
            Dictionary of individual atom RDF results
        parent_mapping : dict
            Maps individual labels to their parent functional groups
        grouped_colors : dict
            Colors for grouped RDFs
        fill_alpha : float
            Transparency of the shaded bands (default: 0.2)
        """
        import numpy as np
        
        # For each grouped RDF, find corresponding individual atoms and compute std
        for grouped_label, grouped_rdf in grouped_rdfs.items():
            # Extract functional group name (e.g., 'Ob-piperazine' -> 'piperazine')
            parts = grouped_label.split('-')
            if len(parts) >= 2:
                clay_component = parts[0]  # e.g., 'Ob'
                functional_group = '-'.join(parts[1:])  # e.g., 'piperazine'
            else:
                continue
            
            # Find all individual atom RDFs for this group and clay component
            matching_atoms = []
            for ind_label, ind_rdf in individual_rdfs.items():
                # Check if this individual belongs to the same group and clay component
                if (hasattr(ind_rdf, '_parent_group') and 
                    ind_rdf._parent_group == functional_group and
                    ind_label.startswith(clay_component + '-')):
                    matching_atoms.append(ind_rdf)
            
            # Need at least 2 atoms to compute std
            if len(matching_atoms) < 2:
                continue
            
            # Extract bins and RDF values
            if hasattr(grouped_rdf, 'bins'):
                bins = grouped_rdf.bins
            elif 'bins' in grouped_rdf:
                bins = grouped_rdf['bins']
            else:
                continue
            
            # Collect all atom RDFs at each distance bin
            n_bins = len(bins)
            rdf_matrix = []
            
            for atom_rdf in matching_atoms:
                if hasattr(atom_rdf, 'rdf'):
                    rdf_values = atom_rdf.rdf
                elif 'rdf' in atom_rdf:
                    rdf_values = atom_rdf['rdf']
                else:
                    continue
                
                # Ensure same number of bins
                if len(rdf_values) == n_bins:
                    rdf_matrix.append(rdf_values)
            
            if len(rdf_matrix) < 2:
                continue
            
            # Convert to numpy array and compute statistics
            rdf_matrix = np.array(rdf_matrix)  # Shape: (n_atoms, n_bins)
            mean_rdf = np.mean(rdf_matrix, axis=0)
            std_rdf = np.std(rdf_matrix, axis=0)
            
            # Get color for this group
            color = grouped_colors.get(grouped_label, 'gray')
            
            # Add shaded band: mean ± std
            ax.fill_between(bins, mean_rdf - std_rdf, mean_rdf + std_rdf,
                           alpha=fill_alpha, color=color, linewidth=0,
                           label=f'{grouped_label} ±σ' if fill_alpha > 0.15 else '')
    
    def plot_multi_component_rdfs(self, pairs=None, save_plots=False, filename='multi_component_rdfs.png',
                                   figsize=None, dpi=None, xlim=None, ylim=None, show_coordination=False):
        """
        Plot multi-component radial distribution functions.
        
        Parameters
        ----------
        pairs : list, optional
            List of component pairs to plot (e.g., ['clay-ion', 'ion-water'])
            If None, plots all available RDF pairs
        save_plots : bool
            Whether to save figure
        filename : str
            Output filename if saving
        figsize : tuple, optional
            Figure size (width, height) in inches
        dpi : int, optional
            Resolution for saved figure
        xlim : tuple, optional
            X-axis limits
        ylim : tuple, optional
            Y-axis limits
        show_coordination : bool
            Whether to show coordination numbers in separate subplot
        
        Returns
        -------
        fig, axes : matplotlib figure and axes objects
        """
        self._validate_analysis()
        
        if not hasattr(self.analysis, 'rdf_results') or not self.analysis.rdf_results:
            raise ValueError("No RDF results found. Run calculate_multi_component_rdfs() first.")
        
        rdf_results = self.analysis.rdf_results
        
        # Filter pairs if specified
        if pairs is not None:
            rdf_results = {k: v for k, v in rdf_results.items() if k in pairs}
        
        if not rdf_results:
            print("No RDF pairs to plot")
            return
        
        # Determine figure layout
        n_pairs = len(rdf_results)
        n_cols = min(3, n_pairs)
        n_rows = int(np.ceil(n_pairs / n_cols))
        
        if show_coordination:
            n_rows *= 2  # Double rows for coordination subplots
        
        figsize = figsize or (6 * n_cols, 4 * n_rows)
        dpi = dpi or self.default_dpi
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        
        # Plot each RDF pair
        for idx, (pair_name, rdf_data) in enumerate(rdf_results.items()):
            row = idx // n_cols
            col = idx % n_cols
            
            if show_coordination:
                row *= 2  # Use every other row for RDF
            
            ax = axes[row, col]
            
            # Plot g(r)
            r = rdf_data['r']
            gr = rdf_data['gr']
            
            ax.plot(r, gr, linewidth=2, color=self.default_colors[idx % len(self.default_colors)])
            ax.set_xlabel('Distance (Å)', fontweight='bold')
            ax.set_ylabel('g(r)', fontweight='bold')
            ax.set_title(f'{pair_name}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
            
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            
            # Find and mark first peak
            peaks, _ = find_peaks(gr, height=1.5)
            if len(peaks) > 0:
                first_peak = peaks[0]
                ax.plot(r[first_peak], gr[first_peak], 'ro', markersize=8, zorder=5)
                ax.text(r[first_peak], gr[first_peak] * 1.1, f'{r[first_peak]:.2f} Å',
                       ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor='white', alpha=0.8))
            
            # Plot coordination number if requested
            if show_coordination:
                ax_coord = axes[row + 1, col]
                cn = rdf_data['coordination_number']
                ax_coord.plot(r, cn, linewidth=2, color=self.default_colors[idx % len(self.default_colors)])
                ax_coord.set_xlabel('Distance (Å)', fontweight='bold')
                ax_coord.set_ylabel('Coordination Number', fontweight='bold')
                ax_coord.set_title(f'CN: {pair_name}', fontweight='bold')
                ax_coord.grid(True, alpha=0.3)
                
                if xlim:
                    ax_coord.set_xlim(xlim)
        
        # Remove empty subplots
        for idx in range(n_pairs, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if show_coordination:
                row *= 2
            fig.delaxes(axes[row, col])
            if show_coordination:
                fig.delaxes(axes[row + 1, col])
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"RDF plot saved to {filename}")
        
        plt.show()
        
        return fig, axes
    
    # =========================================================================
    # COMPETITIVE ADSORPTION PLOTTING
    # =========================================================================
    
    def plot_competitive_adsorption(self, plot_type='grouped_bars', species_to_plot=None,
                                    save_plots=False, filename='competitive_adsorption.png',
                                    figsize=None, dpi=None, show_time_series=True,
                                    colors=None, bar_width=0.25, show_values=True):
        """
        Plot competitive adsorption analysis results with distance-range categorization.
        
        Parameters
        ----------
        plot_type : str, default='grouped_bars'
            Type of plot to create:
            - 'grouped_bars': Grouped bar chart with distance ranges side-by-side
            - 'stacked_bars': Stacked bar chart showing range contributions
            - 'heatmap': Heatmap showing species × distance ranges
            - 'time_series': Time series for all species and ranges
        species_to_plot : dict, optional
            Which species to plot: {'ions': ['Na', 'Mg'], 'organics': ['quinolone', 'piperazine']}
            If None, plots all available species
        save_plots : bool
            Whether to save figure
        filename : str
            Output filename if saving
        figsize : tuple, optional
            Figure size (width, height) in inches
        dpi : int, optional
            Resolution for saved figure
        show_time_series : bool
            Whether to include time series subplots (only for 'grouped_bars' and 'stacked_bars')
        colors : dict, optional
            Colors for distance ranges: {'CIP': 'red', 'SIP': 'blue', 'DSIP': 'green'}
            If None, uses default colormap
        bar_width : float
            Width of bars in grouped bar plot (default: 0.25)
        show_values : bool
            Whether to show numerical values on bars (default: True)
        
        Returns
        -------
        fig, axes : matplotlib figure and axes objects
        
        Examples
        --------
        >>> # Grouped bar chart with time series
        >>> plotter.plot_competitive_adsorption(
        ...     plot_type='grouped_bars',
        ...     show_time_series=True,
        ...     colors={'CIP': 'red', 'SIP': 'orange', 'DSIP': 'yellow'}
        ... )
        
        >>> # Heatmap
        >>> plotter.plot_competitive_adsorption(
        ...     plot_type='heatmap',
        ...     figsize=(12, 8)
        ... )
        
        >>> # Plot specific species only
        >>> plotter.plot_competitive_adsorption(
        ...     species_to_plot={'ions': ['Na'], 'organics': ['quinolone', 'carboxylic_acid']}
        ... )
        """
        self._validate_analysis()
        
        if 'competitive_adsorption' not in self.analysis.results:
            raise ValueError("No competitive adsorption results found. Run analyze_competitive_adsorption() first.")
        
        ca_data = self.analysis.results['competitive_adsorption']
        
        # Check data structure
        if 'ions' not in ca_data or 'organics' not in ca_data:
            raise ValueError("Competitive adsorption data has unexpected structure. "
                           "Expected 'ions' and 'organics' keys with distance range data.")
        
        # Filter species if requested
        if species_to_plot:
            ions_data = {k: v for k, v in ca_data['ions'].items() 
                        if k in species_to_plot.get('ions', [])}
            organics_data = {k: v for k, v in ca_data['organics'].items() 
                            if k in species_to_plot.get('organics', [])}
        else:
            ions_data = ca_data['ions']
            organics_data = ca_data['organics']
        
        # Get distance ranges from metadata or first species
        if 'metadata' in ca_data and 'distance_ranges' in ca_data['metadata']:
            distance_ranges = list(ca_data['metadata']['distance_ranges'].keys())
        else:
            # Infer from first ion - handle nested target structure
            first_ion = next(iter(ions_data.values()))
            # Check if data has target level (ion -> target -> range) or direct (ion -> range)
            first_value = next(iter(first_ion.values()))
            if isinstance(first_value, dict) and 'mean' not in first_value:
                # Has target level: ion -> target -> range -> {mean, std, time_series}
                first_target = first_value
                distance_ranges = list(next(iter(first_target.values())).keys())
            else:
                # Direct: ion -> range -> {mean, std, time_series}
                distance_ranges = list(first_ion.keys())
        
        # Set default colors if not provided
        if colors is None:
            range_colors = plt.cm.Set3(np.linspace(0, 1, len(distance_ranges)))
            colors = {range_name: range_colors[i] for i, range_name in enumerate(distance_ranges)}
        
        dpi = dpi or self.default_dpi
        
        # Route to appropriate plotting method
        if plot_type == 'grouped_bars':
            return self._plot_grouped_bars_competitive(
                ions_data, organics_data, distance_ranges, colors, bar_width,
                show_values, show_time_series, figsize, dpi, save_plots, filename
            )
        elif plot_type == 'stacked_bars':
            return self._plot_stacked_bars_competitive(
                ions_data, organics_data, distance_ranges, colors,
                show_values, show_time_series, figsize, dpi, save_plots, filename
            )
        elif plot_type == 'heatmap':
            return self._plot_heatmap_competitive(
                ions_data, organics_data, distance_ranges,
                figsize, dpi, save_plots, filename
            )
        elif plot_type == 'time_series':
            return self._plot_time_series_competitive(
                ions_data, organics_data, distance_ranges, colors,
                figsize, dpi, save_plots, filename
            )
        else:
            raise ValueError(f"Unknown plot_type: '{plot_type}'. "
                           "Choose from: 'grouped_bars', 'stacked_bars', 'heatmap', 'time_series'")
    
    def plot_ion_competitive_adsorption(self, binding_results=None,
                                       # Data selection
                                       ion_types=None, target_names=None, distance_ranges=None,
                                       # Overall plot control
                                       title='Ion Competitive Adsorption',
                                       # Bar styling
                                       bar_width=0.25, target_hatches=None, edgecolor='black', edgewidth=1.2, bar_alpha=0.8,
                                       # Value labels on bars
                                       show_values=False, value_fontsize=9, value_format='{:.2f}',
                                       value_rotation=0, value_offset=0.05,
                                       # Error bars
                                       show_error_bars=True, errorbar_capsize=3,
                                       # Font & text control
                                       title_fontsize=22, title_fontweight='bold', show_title=True,
                                       label_fontsize=22, label_fontweight='bold',
                                       tick_fontsize=18, legend_fontsize=18, legend_fontweight='bold',
                                       # Axis labels
                                       xlabel='Ion-Target Pair', ylabel='Mean Count',
                                       # Legend control
                                       show_legend=True, show_legend_title=True, legend_title='Distance Range', legend_loc='best',
                                       legend_framealpha=0.9, legend_ncol=1,
                                       # Grid control
                                       show_grid=True, grid_alpha=0.3, grid_axis='y',
                                       # Axis limits
                                       ylim=None,
                                       # Colors
                                       colors=None, colormap='Set3',
                                       # Figure export control
                                       save_fig=False, filename='ion_competitive_adsorption.png',
                                       dpi=300, figsize=None, bbox_inches='tight',
                                       transparent_bg=False):
        """
        Plot ion competitive adsorption as grouped bar charts with full parameter control.
        
        This method creates grouped bar charts showing ion binding to different clay surface
        targets across various distance ranges (CIP, SIP, DSIP). Provides comprehensive
        control over plot styling, similar to MolecularAnalysisPlotter methods.
        
        Parameters
        ----------
        binding_results : dict, optional
            Results from analyze_competitive_adsorption(). If None, uses stored results.
            Structure: {'ions': {ion: {target: {range: {mean, std, time_series}}}}}
        
        Data Selection
        --------------
        ion_types : list of str, optional
            Specific ions to plot (e.g., ['Na', 'Mg']). If None, plots all ions.
        target_names : list of str, optional
            Specific targets to plot (e.g., ['Mgo', 'H_Ohmg']). If None, plots all targets.
        distance_ranges : list of str, optional
            Specific distance ranges to plot (e.g., ['CIP', 'SIP']). If None, plots all ranges.
        
        Overall Plot Control
        --------------------
        title : str
            Plot title (default: 'Ion Competitive Adsorption')
        
        Bar Styling
        -----------
        bar_width : float
            Width of each bar (default: 0.25)
        target_hatches : dict, optional
            Hatching patterns for targets {'Mgo': '///', 'H_Ohmg': '\\\\\\\\\\\\'}
            Applied to distinguish different targets (default: None, auto-generates)
        edgecolor : str
            Bar edge color (default: 'black')
        edgewidth : float
            Bar edge width (default: 1.2)
        bar_alpha : float
            Bar transparency 0-1 (default: 0.8)
        
        Value Labels on Bars
        --------------------
        show_values : bool
            Whether to display value labels on top of bars (default: False)
        value_fontsize : int
            Font size for value labels (default: 9)
        value_format : str
            Format string for value labels (default: '{:.2f}')
        value_rotation : int
            Rotation angle for value labels in degrees (default: 0)
        value_offset : float
            Vertical offset for value labels as fraction of y-range (default: 0.05)
        
        Error Bars
        ----------
        show_error_bars : bool
            Whether to display error bars (default: True)
        errorbar_capsize : float
            Size of error bar caps (default: 3)
        
        Font & Text Control
        -------------------
        title_fontsize : int
            Title font size (default: 22)
        title_fontweight : str
            Title font weight (default: 'bold')
        show_title : bool
            Whether to show title (default: True)
        label_fontsize : int
            Axis label font size (default: 22)
        label_fontweight : str
            Axis label font weight (default: 'bold')
        tick_fontsize : int
            Tick label font size (default: 18)
        legend_fontsize : int
            Legend font size (default: 18)
        legend_fontweight : str
            Legend font weight (default: 'bold')
        
        Axis Labels
        -----------
        xlabel : str
            X-axis label (default: 'Ion-Target Pair')
        ylabel : str
            Y-axis label (default: 'Mean Count')
        
        Legend Control
        --------------
        show_legend : bool
            Whether to show legend (default: True)
        show_legend_title : bool
            Whether to show legend title (default: True)
        legend_title : str
            Legend title (default: 'Distance Range')
        legend_loc : str
            Legend location (default: 'best')
        legend_framealpha : float
            Legend background transparency (default: 0.9)
        legend_ncol : int
            Number of legend columns (default: 1)
        
        Grid Control
        ------------
        show_grid : bool
            Whether to show grid (default: True)
        grid_alpha : float
            Grid transparency (default: 0.3)
        grid_axis : str
            Which axis to show grid: 'both', 'x', 'y' (default: 'y')
        
        Axis Limits
        -----------
        ylim : tuple, optional
            Y-axis limits (default: None, auto-scale)
        
        Colors
        ------
        colors : dict, optional
            Colors for distance ranges: {'CIP': 'lightcoral', 'SIP': 'lightblue'}
            If None, uses default colors from EquilibriumAnalysisOptimized
        colormap : str
            Matplotlib colormap name (default: 'Set3', used as fallback)
        
        Figure Export Control
        ---------------------
        save_fig : bool
            Whether to save figure (default: False)
        filename : str
            Output filename (default: 'ion_competitive_adsorption.png')
        dpi : int
            Resolution (default: 300)
        figsize : tuple, optional
            Figure size (width, height) in inches. If None, auto-calculated.
        bbox_inches : str
            Bounding box for saved figure (default: 'tight')
        transparent_bg : bool
            Whether to save with transparent background (default: False)
        
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        
        Examples
        --------
        >>> # Basic usage with error bars, no value labels
        >>> plotter.plot_ion_competitive_adsorption()
        
        >>> # Custom styling with hatches, values, no error bars
        >>> plotter.plot_ion_competitive_adsorption(
        ...     ion_types=['Na', 'Mg'],
        ...     target_names=['Mgo', 'H_Ohmg', 'Ob', 'Op'],
        ...     target_hatches={'Mgo': '///', 'H_Ohmg': '\\\\\\\\\\\\', 'Ob': 'xxx', 'Op': 'ooo'},
        ...     show_values=True,
        ...     show_error_bars=False,
        ...     value_rotation=0,
        ...     legend_ncol=2,
        ...     legend_framealpha=0.0,
        ...     show_legend_title=False,
        ...     title='Ion Competitive Adsorption to Clay Surface Sites',
        ...     save_fig=True,
        ...     dpi=600
        ... )
        """
        self._validate_analysis()
        
        # Get data
        if binding_results is None:
            if 'competitive_adsorption' not in self.analysis.results:
                raise ValueError("No competitive adsorption results found. Run analyze_competitive_adsorption() first.")
            binding_results = self.analysis.results['competitive_adsorption']
        
        ions_data = binding_results.get('ions', {})
        if not ions_data:
            print("No ion data to plot")
            return None, None
        
        # Filter by ion_types if specified
        if ion_types:
            ions_data = {k: v for k, v in ions_data.items() if k in ion_types}
        
        # Detect structure and get distance ranges
        first_ion = next(iter(ions_data.values()))
        first_value = next(iter(first_ion.values()))
        
        # Check if it's multi-target structure
        # Multi-target: ion -> target -> range -> {mean, std}
        # Simple: ion -> range -> {mean, std}
        if isinstance(first_value, dict):
            # Could be either target dict or range dict
            sample_key = next(iter(first_value.keys()))
            sample_data = first_value[sample_key]
            has_targets = isinstance(sample_data, dict) and 'mean' in sample_data
        else:
            has_targets = False
        
        if has_targets:
            # Multi-target structure: ion -> target -> range
            if distance_ranges is None:
                first_target = first_value
                distance_ranges = list(first_target.keys())
            
            # Filter by target_names if specified
            if target_names:
                for ion in ions_data:
                    ions_data[ion] = {k: v for k, v in ions_data[ion].items() if k in target_names}
            
            # Build ion-target pairs
            ion_target_pairs = []
            for ion_name in ions_data.keys():
                for target_name in ions_data[ion_name].keys():
                    ion_target_pairs.append(f"{ion_name}-{target_name}")
            
            if not ion_target_pairs:
                print("No data after filtering")
                return None, None
            
            x_labels = ion_target_pairs
        else:
            # Simple structure: ion -> range
            if distance_ranges is None:
                distance_ranges = list(first_ion.keys())
            x_labels = list(ions_data.keys())
        
        # Set up colors - use established convention from EquilibriumAnalysisOptimized
        if colors is None:
            # Default color scheme matching EquilibriumAnalysisOptimized
            default_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
            colors = {range_name: default_colors[i % len(default_colors)] 
                     for i, range_name in enumerate(distance_ranges)}
        
        # Generate hatching patterns for targets if not provided
        if target_hatches is None and has_targets:
            hatch_options = ['///', '\\\\\\', 'xxx', '...', '|||', '***', 'ooo', '+++']
            # Get all unique target names
            all_targets = set()
            for ion in ions_data.values():
                all_targets.update(ion.keys())
            target_hatches = {target: hatch_options[i % len(hatch_options)] 
                            for i, target in enumerate(sorted(all_targets))}
        
        # Create figure
        if has_targets:
            # Group bars by ion: x-axis shows ion names, targets/ranges grouped within each ion
            ion_names = list(ions_data.keys())
            target_names_ordered = sorted(list(set(t for ion in ions_data.values() for t in ion.keys())))
            n_ions = len(ion_names)
            n_targets = len(target_names_ordered)
            n_ranges = len(distance_ranges)
            
            figsize = figsize or (max(8, n_ions * n_targets * n_ranges * 0.4), 6)
            fig, ax = plt.subplots(figsize=figsize)
            
            # Calculate bar positions: targets and ranges grouped within each ion
            n_bars_per_ion = n_targets * n_ranges
            
            # Create positions with spacing between ion groups
            group_spacing = n_bars_per_ion * bar_width * 1.3  # Add 30% extra space between groups
            x_positions = np.arange(n_ions) * group_spacing
            
            total_width = bar_width * n_bars_per_ion
            start_offset = -total_width / 2 + bar_width / 2
            
            # Track which bars have been added to legend
            legend_added = set()
            
            # Plot bars grouped by ion
            bar_idx = 0
            for target_name in target_names_ordered:
                for range_name in distance_ranges:
                    means = []
                    stds = []
                    
                    for ion_name in ion_names:
                        if target_name in ions_data[ion_name]:
                            means.append(ions_data[ion_name][target_name][range_name]['mean'])
                            stds.append(ions_data[ion_name][target_name][range_name]['std'])
                        else:
                            means.append(0)
                            stds.append(0)
                    
                    # Calculate bar position for this target-range combo
                    bar_position = x_positions + start_offset + bar_idx * bar_width
                    
                    # Only add to legend once per range
                    label = range_name if range_name not in legend_added else ''
                    if label:
                        legend_added.add(range_name)
                    
                    # Plot bars without error bars in the bar() call
                    bars = ax.bar(bar_position, means, bar_width,
                                 label=label,
                                 color=colors[range_name],
                                 hatch=target_hatches.get(target_name, ''),
                                 edgecolor=edgecolor, linewidth=edgewidth,
                                 alpha=bar_alpha)
                    
                    # Add error bars separately if requested
                    if show_error_bars and any(stds):
                        # Clip lower error bars at zero (asymmetric errors)
                        lower_errors = [min(m, s) for m, s in zip(means, stds)]
                        upper_errors = stds
                        
                        ax.errorbar(bar_position, means, yerr=[lower_errors, upper_errors],
                                   fmt='none', ecolor='black', elinewidth=1.5,
                                   capsize=errorbar_capsize, capthick=1.5, alpha=0.8, zorder=10)
                    
                    # Add value labels
                    if show_values:
                        # Calculate dynamic offset based on y-axis range
                        if ylim:
                            y_range = ylim[1] - ylim[0]
                        else:
                            y_range = max(means) if means else 1
                        dynamic_offset = y_range * value_offset
                        
                        for bar, mean in zip(bars, means):
                            if mean > 0:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2, height + dynamic_offset,
                                       value_format.format(mean),
                                       ha='center', va='bottom', fontsize=value_fontsize,
                                       rotation=value_rotation, fontweight='normal')
                    
                    bar_idx += 1
            
            # Customize plot
            ax.set_xlabel(xlabel if xlabel != 'Ion-Target Pair' else 'Ion Type',
                         fontweight=label_fontweight, fontsize=label_fontsize)
            ax.set_ylabel(ylabel, fontweight=label_fontweight, fontsize=label_fontsize)
            
            if show_title:
                ax.set_title(title, fontweight=title_fontweight, fontsize=title_fontsize)
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(ion_names, fontsize=tick_fontsize)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            
        else:
            # Simple structure without targets
            figsize = figsize or (max(8, len(x_labels) * 0.8), 6)
            fig, ax = plt.subplots(figsize=figsize)
            
            x_pos = np.arange(len(x_labels))
            
            # Plot bars for each distance range
            for i, range_name in enumerate(distance_ranges):
                means = []
                stds = []
                
                for ion_name in ions_data.keys():
                    means.append(ions_data[ion_name][range_name]['mean'])
                    stds.append(ions_data[ion_name][range_name]['std'])
                
                offset = (i - len(distance_ranges)/2 + 0.5) * bar_width
                bars = ax.bar(x_pos + offset, means, bar_width, yerr=stds,
                             label=range_name, color=colors[range_name],
                             edgecolor=edgecolor, linewidth=edgewidth,
                             capsize=3, alpha=bar_alpha)
                
                # Add value labels
                if show_values:
                    for bar, mean in zip(bars, means):
                        if mean > 0:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2, height * value_offset,
                                   value_format.format(mean),
                                   ha='center', va='bottom', fontsize=value_fontsize,
                                   rotation=value_rotation, fontweight=legend_fontweight)
            
            # Customize plot
            ax.set_xlabel(xlabel, fontweight=label_fontweight, fontsize=label_fontsize)
            ax.set_ylabel(ylabel, fontweight=label_fontweight, fontsize=label_fontsize)
            
            if show_title:
                ax.set_title(title, fontweight=title_fontweight, fontsize=title_fontsize)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=tick_fontsize)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        if show_legend:
            # Build legend handles
            if has_targets and target_hatches:
                # Create combined legend: distance ranges (colors) + targets (hatches)
                from matplotlib.patches import Patch
                
                # Distance range handles (colored)
                range_handles = [Patch(facecolor=colors[range_name], edgecolor=edgecolor, 
                                      linewidth=edgewidth, alpha=bar_alpha, label=range_name)
                                for range_name in distance_ranges]
                
                # Target hatch handles (white with hatches for clarity)
                target_handles = [Patch(facecolor='white', edgecolor=edgecolor, 
                                       hatch=target_hatches[target], linewidth=edgewidth, 
                                       alpha=bar_alpha, label=target)
                                 for target in sorted(target_hatches.keys()) 
                                 if target in {t for ion in ions_data.values() for t in ion.keys()}]
                
                # Combine handles and create sectioned legend
                combined_title = None
                if show_legend_title:
                    combined_title = legend_title if legend_title != 'Distance Range' else 'Range / Target'
                
                ax.legend(handles=range_handles + target_handles, 
                         title=combined_title,
                         loc=legend_loc, framealpha=legend_framealpha,
                         ncol=legend_ncol, fontsize=legend_fontsize)
            else:
                # Simple legend for distance ranges only
                ax.legend(title=legend_title if show_legend_title else None,
                         loc=legend_loc, framealpha=legend_framealpha,
                         ncol=legend_ncol, fontsize=legend_fontsize)
        
        if show_grid:
            ax.grid(True, alpha=grid_alpha, axis=grid_axis)
        
        if ylim:
            ax.set_ylim(ylim)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent_bg)
            print(f"✓ Ion competitive adsorption plot saved: {filename}")
        
        plt.show()
        return fig, ax
    
    def plot_organic_competitive_adsorption(self, binding_results=None,
                                           # Data selection
                                           organic_names=None, target_names=None, distance_ranges=None,
                                           show_individual_atoms=False,
                                           # Overall plot control
                                           title='Organic Competitive Adsorption',
                                           # Bar styling
                                           bar_width=0.25, target_hatches=None, edgecolor='black', edgewidth=1.2, bar_alpha=0.8,
                                           # Value labels on bars
                                           show_values=False, value_fontsize=9, value_format='{:.2f}',
                                           value_rotation=0, value_offset=0.05,
                                           # Error bars
                                           show_error_bars=True, errorbar_capsize=3,
                                           # Font & text control
                                           title_fontsize=22, title_fontweight='bold', show_title=True,
                                           label_fontsize=22, label_fontweight='bold',
                                           tick_fontsize=18, legend_fontsize=18, legend_fontweight='bold',
                                           # Axis labels
                                           xlabel='Organic-Target Pair', ylabel='Mean Count',
                                           # Legend control
                                           show_legend=True, show_legend_title=True, legend_title='Distance Range', legend_loc='best',
                                           legend_framealpha=0.9, legend_ncol=1,
                                           # Grid control
                                           show_grid=True, grid_alpha=0.3, grid_axis='y',
                                           # Axis limits
                                           ylim=None,
                                           # Colors
                                           colors=None, colormap='Set3',
                                           # Figure export control
                                           save_fig=False, filename='organic_competitive_adsorption.png',
                                           dpi=300, figsize=None, bbox_inches='tight',
                                           transparent_bg=False):
        """
        Plot organic competitive adsorption as grouped bar charts (mirrors plot_ion_competitive_adsorption).
        
        Shows organic molecule binding to different clay surface targets across distance ranges.
        Bars are grouped by organic type, with hatches distinguishing different targets.
        
        Parameters
        ----------
        binding_results : dict, optional
            Results from analyze_competitive_adsorption(). If None, uses stored results.
        organic_names : list of str, optional
            Specific organics to plot. If None, plots all organics.
        target_names : list of str, optional
            Specific targets to plot. If None, plots all targets.
        distance_ranges : list of str, optional
            Specific distance ranges. If None, plots all ranges.
        show_individual_atoms : bool, default=False
            If True and data contains per-atom information, plots individual atoms.
            If False, plots aggregated data for each organic moiety.
            Only works when analyze_competitive_adsorption was run with store_per_atom_organics=True.
        
        [All other parameters identical to plot_ion_competitive_adsorption - see that method for details]
        
        Examples
        --------
        >>> plotter.plot_organic_competitive_adsorption(
        ...     organic_names=['quinolone', 'piperazine', 'carboxylic_acid'],
        ...     target_hatches={'Mgo': '///', 'H_Ohmg': '\\\\\\\\\\\\', 'Ob': 'xxx'},
        ...     show_values=True,
        ...     show_error_bars=False,
        ...     legend_ncol=2,
        ...     save_fig=True
        ... )
        """
        # Get binding results
        if binding_results is None:
            if not hasattr(self.analysis, 'results') or 'competitive_adsorption' not in self.analysis.results:
                print("No binding results available. Run analyze_competitive_adsorption first.")
                return None
            binding_results = self.analysis.results['competitive_adsorption']
        
        # Validate structure
        if 'organics' not in binding_results:
            print("Error: 'organics' key not found in binding_results")
            return None
        
        organics_data = binding_results['organics']
        
        if not organics_data:
            print("No organic data available. Did you provide organic_parts in analyze_competitive_adsorption?")
            return None
        
        # Filter organics
        if organic_names:
            organics_data = {k: v for k, v in organics_data.items() if k in organic_names}
        
        if not organics_data:
            print("No data available for selected organics")
            return None
        
        # Detect if data is per-atom or aggregated
        first_organic = next(iter(organics_data.values()))
        first_target = next(iter(first_organic.values()))
        first_range = next(iter(first_target.values()))
        
        # Per-atom data: organic -> target -> range -> atom -> {mean, std, time_series}
        # Aggregated data: organic -> target -> range -> {mean, std, time_series}
        is_per_atom_data = isinstance(first_range, dict) and 'mean' not in first_range
        
        # Check if user wants per-atom but data isn't per-atom
        if show_individual_atoms and not is_per_atom_data:
            print("Warning: show_individual_atoms=True but data is aggregated.")
            print("Run analyze_competitive_adsorption with store_per_atom_organics=True to enable per-atom plotting.")
            show_individual_atoms = False
        
        # If data is per-atom but user doesn't want it, aggregate
        if is_per_atom_data and not show_individual_atoms:
            print("Data contains per-atom information. Set show_individual_atoms=True to plot individual atoms.")
            print("Plotting aggregated sums...")
            # Aggregate per-atom data by summing means
            for org_name in organics_data:
                for target_name in organics_data[org_name]:
                    for range_name in organics_data[org_name][target_name]:
                        atom_data = organics_data[org_name][target_name][range_name]
                        if isinstance(atom_data, dict) and 'mean' not in atom_data:
                            # Sum all atom means and stds (approximate aggregation)
                            total_mean = sum(atom['mean'] for atom in atom_data.values())
                            # Std approximation: sqrt(sum of variances)
                            total_std = np.sqrt(sum(atom['std']**2 for atom in atom_data.values()))
                            organics_data[org_name][target_name][range_name] = {
                                'mean': total_mean,
                                'std': total_std
                            }
            is_per_atom_data = False
        
        # Get available targets and ranges
        first_organic = next(iter(organics_data.values()))
        all_targets = set(first_organic.keys())
        first_target_data = next(iter(first_organic.values()))
        all_ranges = set(first_target_data.keys())
        
        if target_names:
            targets_to_plot = [t for t in target_names if t in all_targets]
        else:
            targets_to_plot = sorted(list(all_targets))
        
        if distance_ranges:
            ranges_to_plot = [r for r in distance_ranges if r in all_ranges]
        else:
            ranges_to_plot = sorted(list(all_ranges))
        
        if not targets_to_plot or not ranges_to_plot:
            print("No data available for selected targets/ranges")
            return None
        
        organic_names_ordered = sorted(list(organics_data.keys()))
        target_names_ordered = targets_to_plot
        distance_ranges = ranges_to_plot
        
        n_organics = len(organic_names_ordered)
        n_targets = len(target_names_ordered)
        n_ranges = len(distance_ranges)
        
        # Set up colors
        if colors is None:
            colors = {
                'CIP': 'lightcoral',
                'SIP': 'lightblue',
                'DSIP': 'lightgreen',
                'FI': 'lightyellow'
            }
        
        # Auto-generate target hatches
        if target_hatches is None:
            hatch_patterns = ['///', '\\\\\\\\\\\\', 'xxx', '...', '|||', '***', 'ooo', '+++']
            target_hatches = {}
            for idx, target in enumerate(target_names_ordered):
                target_hatches[target] = hatch_patterns[idx % len(hatch_patterns)]
        
        # Create figure
        figsize = figsize or (max(8, n_organics * n_targets * n_ranges * 0.4), 6)  
        fig, ax = plt.subplots(figsize=figsize)
        
        # Build entity list (organics or atoms)
        if is_per_atom_data:
            # Create list of (organic, atom) pairs
            entity_list = []
            entity_labels = []
            for org_name in organic_names_ordered:
                # Get all atoms for this organic from first target/range
                first_target = target_names_ordered[0]
                first_range = distance_ranges[0]
                atom_dict = organics_data[org_name][first_target][first_range]
                atoms = sorted(atom_dict.keys())
                
                for atom_name in atoms:
                    entity_list.append((org_name, atom_name))
                    entity_labels.append(f"{org_name}\n{atom_name}")
            
            n_entities = len(entity_list)
        else:
            # Regular organic list
            entity_list = [(org_name, None) for org_name in organic_names_ordered]
            entity_labels = organic_names_ordered
            n_entities = n_organics
        
        # Calculate bar positions
        n_bars_per_entity = n_targets * n_ranges
        
        # Create positions with spacing
        group_spacing = n_bars_per_entity * bar_width * 1.3
        x_positions = np.arange(n_entities) * group_spacing
        
        total_width = bar_width * n_bars_per_entity
        start_offset = -total_width / 2 + bar_width / 2
        
        # Track legend
        legend_added = set()
        
        # Plot bars grouped by entity (organic or organic-atom)
        bar_idx = 0
        for target_name in target_names_ordered:
            for range_name in distance_ranges:
                means = []
                stds = []
                
                for entity in entity_list:
                    if is_per_atom_data:
                        org_name, atom_name = entity
                        if target_name in organics_data[org_name]:
                            atom_dict = organics_data[org_name][target_name][range_name]
                            if atom_name in atom_dict:
                                means.append(atom_dict[atom_name]['mean'])
                                stds.append(atom_dict[atom_name]['std'])
                            else:
                                means.append(0)
                                stds.append(0)
                        else:
                            means.append(0)
                            stds.append(0)
                    else:
                        org_name, _ = entity
                        if target_name in organics_data[org_name]:
                            means.append(organics_data[org_name][target_name][range_name]['mean'])
                            stds.append(organics_data[org_name][target_name][range_name]['std'])
                        else:
                            means.append(0)
                            stds.append(0)
                
                # Calculate bar position
                bar_position = x_positions + start_offset + bar_idx * bar_width
                
                # Only add to legend once per range
                label = range_name if range_name not in legend_added else ''
                if label:
                    legend_added.add(range_name)
                
                # Plot bars without error bars
                bars = ax.bar(bar_position, means, bar_width,
                             label=label,
                             color=colors[range_name],
                             hatch=target_hatches.get(target_name, ''),
                             edgecolor=edgecolor, linewidth=edgewidth,
                             alpha=bar_alpha)
                
                # Add error bars separately if requested
                if show_error_bars and any(stds):
                    # Clip lower error bars at zero
                    lower_errors = [min(m, s) for m, s in zip(means, stds)]
                    upper_errors = stds
                    
                    ax.errorbar(bar_position, means, yerr=[lower_errors, upper_errors],
                               fmt='none', ecolor='black', elinewidth=1.5,
                               capsize=errorbar_capsize, capthick=1.5, alpha=0.8, zorder=10)
                
                # Add value labels
                if show_values:
                    # Calculate dynamic offset
                    if ylim:
                        y_range = ylim[1] - ylim[0]
                    else:
                        y_range = max(means) if means else 1
                    dynamic_offset = y_range * value_offset
                    
                    for bar, mean in zip(bars, means):
                        if mean > 0:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2, height + dynamic_offset,
                                   value_format.format(mean),
                                   ha='center', va='bottom', fontsize=value_fontsize,
                                   rotation=value_rotation, fontweight='normal')
                
                bar_idx += 1
        
        # Customize plot
        xlabel_text = xlabel if xlabel == 'Organic-Target Pair' else xlabel
        if xlabel == 'Organic-Target Pair':
            xlabel_text = 'Atom' if is_per_atom_data else 'Organic Type'
        
        ax.set_xlabel(xlabel_text, fontweight=label_fontweight, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontweight=label_fontweight, fontsize=label_fontsize)
        
        if show_title:
            ax.set_title(title, fontweight=title_fontweight, fontsize=title_fontsize)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(entity_labels, fontsize=tick_fontsize, rotation=45 if is_per_atom_data else 0, ha='right' if is_per_atom_data else 'center')
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        # Grid
        if show_grid:
            ax.grid(True, alpha=grid_alpha, axis=grid_axis, linestyle='--', linewidth=0.5)
        
        # Legend
        if show_legend:
            # Create combined legend (ranges + targets)
            range_handles = [plt.Rectangle((0,0),1,1, fc=colors[r], ec=edgecolor, lw=edgewidth, alpha=bar_alpha) 
                           for r in distance_ranges]
            target_handles = [plt.Rectangle((0,0),1,1, fc='white', ec=edgecolor, hatch=target_hatches[t], lw=edgewidth) 
                            for t in target_names_ordered]
            
            range_labels = list(distance_ranges)
            target_labels = target_names_ordered
            
            # Combine handles and labels
            all_handles = range_handles + target_handles
            all_labels = range_labels + target_labels
            
            if show_legend_title:
                # Two-part legend with titles
                legend1 = ax.legend(range_handles, range_labels, title=legend_title,
                                  loc='upper left', framealpha=legend_framealpha,
                                  fontsize=legend_fontsize, title_fontsize=legend_fontsize,
                                  ncol=1)
                legend1.get_title().set_fontweight(legend_fontweight)
                ax.add_artist(legend1)
                
                legend2 = ax.legend(target_handles, target_labels, title='Target',
                                  loc='upper right', framealpha=legend_framealpha,
                                  fontsize=legend_fontsize, title_fontsize=legend_fontsize,
                                  ncol=1)
                legend2.get_title().set_fontweight(legend_fontweight)
            else:
                # Single combined legend
                ax.legend(all_handles, all_labels, loc=legend_loc,
                        framealpha=legend_framealpha, fontsize=legend_fontsize,
                        ncol=legend_ncol)
        
        if ylim:
            ax.set_ylim(ylim)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent_bg)
            print(f"✓ Organic competitive adsorption plot saved: {filename}")
        
        plt.show()
        return fig, ax
    
    def plot_ion_competitive_timeseries(self, binding_results=None,
                                       # Data selection
                                       ion_types=None, target_names=None, distance_ranges=None,
                                       # Overall plot control
                                       title='Ion Competitive Adsorption Time Series',
                                       # Line styling
                                       linewidth=1.5, colors=None, colormap='Set3',
                                       linestyles=None, line_alpha=0.7, markers=None,
                                       marker_every=None,
                                       # Font & text control
                                       title_fontsize=14, title_fontweight='bold', show_title=True,
                                       label_fontsize=12, label_fontweight='normal',
                                       tick_fontsize=10, legend_fontsize=8, legend_fontweight='normal',
                                       # Axis labels
                                       xlabel='Frame', ylabel='Count',
                                       # Legend control
                                       show_legend=True, legend_loc='best', legend_framealpha=0.9,
                                       legend_ncol=2, custom_labels=None,
                                       # Grid control
                                       show_grid=True, grid_alpha=0.3, grid_linestyle='--',
                                       # Axis limits
                                       xlim=None, ylim=None,
                                       # Figure export control
                                       save_fig=False, filename='ion_competitive_timeseries.png',
                                       dpi=300, figsize=None, bbox_inches='tight',
                                       transparent_bg=False,
                                       # Multi-ion/target figure control
                                       show_individual_figures=False,
                                       individual_figsize=(8, 6),
                                       save_combined_figure=False,
                                       show_combined_figure=True,
                                       save_individual_figures=True):
        """
        Plot time series of ion competitive adsorption with full parameter control.
        
        This method creates time series plots showing the dynamics of ion binding to
        clay surface targets across different distance ranges over the simulation trajectory.
        
        Parameters
        ----------
        binding_results : dict, optional
            Results from analyze_competitive_adsorption(). If None, uses stored results.
        
        Data Selection
        --------------
        ion_types : list of str, optional
            Specific ions to plot (e.g., ['Na', 'Mg']). If None, plots all ions.
        target_names : list of str, optional
            Specific targets to plot (e.g., ['Mgo', 'H_Ohmg']). If None, plots all targets.
        distance_ranges : list of str, optional
            Specific distance ranges to plot (e.g., ['CIP', 'SIP']). If None, plots all ranges.
        
        Overall Plot Control
        --------------------
        title : str
            Plot title (default: 'Ion Competitive Adsorption Time Series')
        
        Line Styling
        ------------
        linewidth : float
            Line width (default: 1.5)
        colors : dict, optional
            Colors for distance ranges: {'CIP': 'red', 'SIP': 'orange'}
            If None, uses colormap
        colormap : str
            Matplotlib colormap name (default: 'Set3')
        linestyles : dict or list, optional
            Line styles for different series. Can be dict mapping to ion-target pairs
            or list of styles to cycle through: '-', '--', '-.', ':'
        line_alpha : float
            Line transparency 0-1 (default: 0.7)
        markers : list, optional
            Marker styles: 'o', 's', '^', 'v', None (default: None)
        marker_every : int, optional
            Plot marker every N points to reduce clutter (default: None, no markers)
        
        Multi-Ion/Target Figure Control
        --------------------------------
        show_individual_figures : bool
            Whether to display individual figures for each ion-target combo (default: False)
        individual_figsize : tuple
            Figure size for individual figures in inches (default: (8, 6))
        save_combined_figure : bool
            Whether to save the combined figure with all data (default: False)
        show_combined_figure : bool
            Whether to display the combined figure (default: True)
        save_individual_figures : bool
            Whether to save separate figures for each ion-target (default: True)
            Filenames auto-generated: 'timeseries_Na_Mgo.png', 'timeseries_Mg_H_Ohmg.png', etc.
        
        [Font, Legend, Grid, Axis, and Export parameters similar to plot_ion_competitive_adsorption]
        
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        
        Examples
        --------
        >>> # Basic time series
        >>> plotter.plot_ion_competitive_timeseries()
        
        >>> # Custom styling with specific data selection
        >>> plotter.plot_ion_competitive_timeseries(
        ...     ion_types=['Na'],
        ...     target_names=['Mgo', 'H_Ohmg'],
        ...     distance_ranges=['CIP', 'SIP'],
        ...     colors={'CIP': 'red', 'SIP': 'orange'},
        ...     linewidth=2,
        ...     line_alpha=0.8,
        ...     legend_ncol=3,
        ...     save_fig=True
        ... )
        """
        self._validate_analysis()
        
        # Get data
        if binding_results is None:
            if 'competitive_adsorption' not in self.analysis.results:
                raise ValueError("No competitive adsorption results found. Run analyze_competitive_adsorption() first.")
            binding_results = self.analysis.results['competitive_adsorption']
        
        ions_data = binding_results.get('ions', {})
        if not ions_data:
            print("No ion data to plot")
            return None, None
        
        # Filter by ion_types if specified
        if ion_types:
            ions_data = {k: v for k, v in ions_data.items() if k in ion_types}
        
        # Detect structure and get distance ranges
        first_ion = next(iter(ions_data.values()))
        first_value = next(iter(first_ion.values()))
        has_targets = isinstance(first_value, dict) and 'mean' not in first_value
        
        if has_targets:
            # Multi-target structure
            if distance_ranges is None:
                first_target = first_value
                distance_ranges = list(next(iter(first_target.values())).keys())
            
            # Filter by target_names if specified
            if target_names:
                for ion in ions_data:
                    ions_data[ion] = {k: v for k, v in ions_data[ion].items() if k in target_names}
        else:
            # Simple structure
            if distance_ranges is None:
                distance_ranges = list(first_ion.keys())
        
        # Set up colors - use established convention matching plot_ion_competitive_adsorption
        if colors is None:
            # Default color scheme: CIP=lightcoral, SIP=lightblue, DSIP=lightgreen
            default_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
            colors = {range_name: default_colors[i % len(default_colors)] 
                     for i, range_name in enumerate(distance_ranges)}
        
        # Get list of ions
        ion_list = list(ions_data.keys())
        n_ions = len(ion_list)
        
        # Helper function to plot single ion
        def plot_single_ion_data(ax, ion_name):
            """Plot all targets and ranges for a single ion"""
            line_idx = 0
            if has_targets:
                # Plot each target-range combination for this ion
                for target_name in ions_data[ion_name].keys():
                    for range_name in distance_ranges:
                        ts = ions_data[ion_name][target_name][range_name]['time_series']
                        frames = np.arange(len(ts))
                        
                        label = custom_labels.get(f"{target_name}-{range_name}",
                                                  f'{target_name}-{range_name}') if custom_labels else f'{target_name}-{range_name}'
                        
                        linestyle = linestyles[line_idx % len(linestyles)] if linestyles else '-'
                        marker = markers[line_idx % len(markers)] if markers else None
                        
                        ax.plot(frames, ts, label=label, color=colors[range_name],
                               alpha=line_alpha, linewidth=linewidth, linestyle=linestyle,
                               marker=marker, markevery=marker_every)
                        line_idx += 1
            else:
                # Plot each range for this ion (no targets)
                for range_name in distance_ranges:
                    ts = ions_data[ion_name][range_name]['time_series']
                    frames = np.arange(len(ts))
                    
                    label = custom_labels.get(range_name, range_name) if custom_labels else range_name
                    
                    linestyle = linestyles[line_idx % len(linestyles)] if linestyles else '-'
                    marker = markers[line_idx % len(markers)] if markers else None
                    
                    ax.plot(frames, ts, label=label, color=colors[range_name],
                           alpha=line_alpha, linewidth=linewidth, linestyle=linestyle,
                           marker=marker, markevery=marker_every)
                    line_idx += 1
            
            # Format ion name for display
            ion_display = ion_name
            if ion_name == 'Na':
                ion_display = r'Na$^+$'
            elif ion_name == 'Mg':
                ion_display = r'Mg$^{2+}$'
            elif ion_name == 'K':
                ion_display = r'K$^+$'
            elif ion_name == 'Ca':
                ion_display = r'Ca$^{2+}$'
            elif ion_name == 'Cl':
                ion_display = r'Cl$^-$'
            
            return ion_display
        
        # === CREATE INDIVIDUAL FIGURES (if requested) ===
        if save_individual_figures or show_individual_figures:
            # Parse base filename
            base_filename = filename
            base_name, ext = base_filename.rsplit('.', 1) if '.' in base_filename else (base_filename, 'png')
            
            # Strip existing ion names to avoid duplicates
            for ion in ion_list:
                if base_name.endswith(f'_{ion}'):
                    base_name = base_name[:-len(f'_{ion}')]
                    break
            
            # Create one figure per ion
            for ion_name in ion_list:
                plt.close('all')
                
                fig_ind, ax_ind = plt.subplots(figsize=individual_figsize)
                
                # Plot all data for this ion
                ion_display = plot_single_ion_data(ax_ind, ion_name)
                
                # Formatting
                ax_ind.set_xlabel(xlabel, fontweight=label_fontweight, fontsize=label_fontsize)
                ax_ind.set_ylabel(ylabel, fontweight=label_fontweight, fontsize=label_fontsize)
                ax_ind.set_title(f'{ion_display} Binding', fontweight=title_fontweight, fontsize=title_fontsize)
                ax_ind.tick_params(axis='both', labelsize=tick_fontsize)
                
                if show_legend:
                    legend = ax_ind.legend(loc=legend_loc if legend_loc != 'best' else 'upper right',
                                          framealpha=legend_framealpha,
                                          ncol=legend_ncol, fontsize=legend_fontsize)
                    for text in legend.get_texts():
                        text.set_fontweight(legend_fontweight)
                
                if show_grid:
                    ax_ind.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
                
                if xlim:
                    ax_ind.set_xlim(xlim)
                if ylim:
                    ax_ind.set_ylim(ylim)
                
                fig_ind.tight_layout()
                
                # Save individual figure
                if save_individual_figures:
                    ind_filename = f"{base_name}_{ion_name}.{ext}"
                    fig_ind.savefig(ind_filename, dpi=dpi, bbox_inches=bbox_inches, 
                                   transparent=transparent_bg)
                    print(f"✓ Individual figure saved: {ind_filename}")
                
                # Show or close
                if show_individual_figures:
                    plt.show()
                else:
                    plt.close(fig_ind)
        
        # === CREATE COMBINED FIGURE ===
        if not show_combined_figure and not save_combined_figure and not save_fig:
            # No combined figure needed
            return None, None
        
        # Multi-ion case: create side-by-side subplots
        if n_ions > 1:
            # Calculate figsize based on number of ions
            ind_figsize_tuple = individual_figsize if isinstance(individual_figsize, tuple) else (8, 6)
            default_combined_figsize = (ind_figsize_tuple[0] * n_ions, ind_figsize_tuple[1])
            figsize = figsize if figsize is not None else default_combined_figsize
            
            fig, axes_array = plt.subplots(1, n_ions, figsize=figsize, squeeze=False)
            axes_array = axes_array.flatten()
            
            # Plot each ion in its own subplot
            for ion_idx, ion_name in enumerate(ion_list):
                ax = axes_array[ion_idx]
                ion_display = plot_single_ion_data(ax, ion_name)
                
                # Formatting for this subplot
                ax.set_xlabel(xlabel, fontweight=label_fontweight, fontsize=label_fontsize)
                ax.set_ylabel(ylabel, fontweight=label_fontweight, fontsize=label_fontsize)
                ax.set_title(f'{ion_display} Binding', fontweight=title_fontweight, fontsize=title_fontsize)
                ax.tick_params(axis='both', labelsize=tick_fontsize)
                
                if show_legend:
                    legend = ax.legend(loc=legend_loc if legend_loc != 'best' else 'upper right',
                                      framealpha=legend_framealpha,
                                      ncol=legend_ncol, fontsize=legend_fontsize)
                    for text in legend.get_texts():
                        text.set_fontweight(legend_fontweight)
                
                if show_grid:
                    ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
                
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)
            
            # Overall title
            if show_title:
                fig.suptitle(title, fontsize=title_fontsize, fontweight=title_fontweight, y=0.98)
            
            plt.tight_layout()
        else:
            # Single ion: just one plot
            figsize = figsize or (12, 6)
            fig, ax = plt.subplots(figsize=figsize)
            
            ion_display = plot_single_ion_data(ax, ion_list[0])
            
            # Formatting
            ax.set_xlabel(xlabel, fontweight=label_fontweight, fontsize=label_fontsize)
            ax.set_ylabel(ylabel, fontweight=label_fontweight, fontsize=label_fontsize)
            
            if show_title:
                ax.set_title(f'{title} - {ion_display}', fontweight=title_fontweight, fontsize=title_fontsize)
            
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            
            if show_legend:
                legend = ax.legend(loc=legend_loc, framealpha=legend_framealpha,
                                  ncol=legend_ncol, fontsize=legend_fontsize)
                # Apply fontweight to legend text
                for text in legend.get_texts():
                    text.set_fontweight(legend_fontweight)
            
            if show_grid:
                ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
            
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            
            plt.tight_layout()
        
        # Save combined figure
        # Only save if save_combined_figure=True, ignoring save_fig for combined figure
        # (save_fig is now only for backward compatibility when save_combined_figure is not used)
        if save_combined_figure:
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent_bg)
            print(f"✓ Ion competitive time series plot saved: {filename}")
        
        # Show combined figure
        if show_combined_figure:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, axes_array if n_ions > 1 else (fig, ax)
        
        return fig, ax
    
    def _plot_grouped_bars_competitive(self, ions_data, organics_data, distance_ranges, 
                                       colors, bar_width, show_values, show_time_series,
                                       figsize, dpi, save_plots, filename):
        """Helper: Create grouped bar charts for competitive adsorption"""
        
        figsize = figsize or (16, 10 if show_time_series else 6)
        
        if show_time_series:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            axes = np.array([[axes[0], axes[1]]])  # Reshape for consistent indexing
        
        # Check if data has target level (ion -> target -> range) or direct (ion -> range)
        first_ion = next(iter(ions_data.values())) if ions_data else {}
        has_targets = first_ion and isinstance(next(iter(first_ion.values()), {}), dict) and 'mean' not in next(iter(first_ion.values()), {})
        
        # Plot 1: Ions grouped bars
        ax1 = axes[0, 0]
        
        if has_targets:
            # Multi-target structure: ion -> target -> range
            ion_target_pairs = []
            for ion_name in ions_data.keys():
                for target_name in ions_data[ion_name].keys():
                    ion_target_pairs.append(f"{ion_name}-{target_name}")
            
            x_pos = np.arange(len(ion_target_pairs))
            
            for i, range_name in enumerate(distance_ranges):
                means = []
                stds = []
                for ion_name in ions_data.keys():
                    for target_name in ions_data[ion_name].keys():
                        means.append(ions_data[ion_name][target_name][range_name]['mean'])
                        stds.append(ions_data[ion_name][target_name][range_name]['std'])
                
                offset = (i - len(distance_ranges)/2 + 0.5) * bar_width
                bars = ax1.bar(x_pos + offset, means, bar_width, yerr=stds, 
                              label=range_name, color=colors[range_name], 
                              capsize=3, alpha=0.8)
                
                if show_values:
                    for j, (bar, mean) in enumerate(zip(bars, means)):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                                f'{mean:.1f}', ha='center', va='bottom', fontsize=8, rotation=90)
            
            ax1.set_xlabel('Ion-Target Pair', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Mean Count', fontweight='bold', fontsize=12)
            ax1.set_title('Ion Adsorption by Target and Distance Range', fontweight='bold', fontsize=14)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(ion_target_pairs, rotation=45, ha='right')
            ax1.legend(title='Distance Range', framealpha=0.9)
            ax1.grid(True, alpha=0.3, axis='y')
        else:
            # Simple structure: ion -> range
            ion_names = list(ions_data.keys())
            x_pos = np.arange(len(ion_names))
            
            for i, range_name in enumerate(distance_ranges):
                means = [ions_data[ion][range_name]['mean'] for ion in ion_names]
                stds = [ions_data[ion][range_name]['std'] for ion in ion_names]
                
                offset = (i - len(distance_ranges)/2 + 0.5) * bar_width
                bars = ax1.bar(x_pos + offset, means, bar_width, yerr=stds, 
                              label=range_name, color=colors[range_name], 
                              capsize=3, alpha=0.8)
                
                if show_values:
                    for j, (bar, mean) in enumerate(zip(bars, means)):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                                f'{mean:.1f}', ha='center', va='bottom', fontsize=8, rotation=90)
            
            ax1.set_xlabel('Ion Type', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Mean Count', fontweight='bold', fontsize=12)
            ax1.set_title('Ion Adsorption by Distance Range', fontweight='bold', fontsize=14)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(ion_names, rotation=45, ha='right')
            ax1.legend(title='Distance Range', framealpha=0.9)
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Organics grouped bars (similar logic)
        ax2 = axes[0, 1]
        
        if organics_data:
            first_org = next(iter(organics_data.values()))
            has_targets_org = first_org and isinstance(next(iter(first_org.values()), {}), dict) and 'mean' not in next(iter(first_org.values()), {})
            
            if has_targets_org:
                # Multi-target structure: organic -> target -> range
                org_target_pairs = []
                for org_name in organics_data.keys():
                    for target_name in organics_data[org_name].keys():
                        org_target_pairs.append(f"{org_name}-{target_name}")
                
                x_pos = np.arange(len(org_target_pairs))
                
                for i, range_name in enumerate(distance_ranges):
                    means = []
                    stds = []
                    for org_name in organics_data.keys():
                        for target_name in organics_data[org_name].keys():
                            means.append(organics_data[org_name][target_name][range_name]['mean'])
                            stds.append(organics_data[org_name][target_name][range_name]['std'])
                    
                    offset = (i - len(distance_ranges)/2 + 0.5) * bar_width
                    bars = ax2.bar(x_pos + offset, means, bar_width, yerr=stds, 
                                  label=range_name, color=colors[range_name], 
                                  capsize=3, alpha=0.8)
                    
                    if show_values:
                        for j, (bar, mean) in enumerate(zip(bars, means)):
                            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                                    f'{mean:.1f}', ha='center', va='bottom', fontsize=8, rotation=90)
                
                ax2.set_xlabel('Organic-Target Pair', fontweight='bold', fontsize=12)
                ax2.set_ylabel('Mean Atom Count', fontweight='bold', fontsize=12)
                ax2.set_title('Organic Adsorption by Target and Distance Range', fontweight='bold', fontsize=14)
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(org_target_pairs, rotation=45, ha='right')
                ax2.legend(title='Distance Range', framealpha=0.9)
                ax2.grid(True, alpha=0.3, axis='y')
            else:
                # Simple structure: organic -> range
                org_names = list(organics_data.keys())
                x_pos = np.arange(len(org_names))
                
                for i, range_name in enumerate(distance_ranges):
                    means = [organics_data[org][range_name]['mean'] for org in org_names]
                    stds = [organics_data[org][range_name]['std'] for org in org_names]
                    
                    offset = (i - len(distance_ranges)/2 + 0.5) * bar_width
                    bars = ax2.bar(x_pos + offset, means, bar_width, yerr=stds, 
                                  label=range_name, color=colors[range_name], 
                                  capsize=3, alpha=0.8)
                    
                    if show_values:
                        for j, (bar, mean) in enumerate(zip(bars, means)):
                            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                                    f'{mean:.1f}', ha='center', va='bottom', fontsize=8, rotation=90)
                
                ax2.set_xlabel('Organic Part', fontweight='bold', fontsize=12)
                ax2.set_ylabel('Mean Atom Count', fontweight='bold', fontsize=12)
                ax2.set_title('Organic Adsorption by Distance Range', fontweight='bold', fontsize=14)
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(org_names, rotation=45, ha='right')
                ax2.legend(title='Distance Range', framealpha=0.9)
                ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'No organic data', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Organic Adsorption (No Data)', fontweight='bold', fontsize=14)
        
        if show_time_series:
            # Plot 3: Ions time series
            ax3 = axes[1, 0]
            
            if has_targets:
                for ion_name in ions_data.keys():
                    for target_name in ions_data[ion_name].keys():
                        for range_name in distance_ranges:
                            ts = ions_data[ion_name][target_name][range_name]['time_series']
                            frames = np.arange(len(ts))
                            ax3.plot(frames, ts, label=f'{ion_name}-{target_name}-{range_name}', 
                                    alpha=0.7, color=colors[range_name], linewidth=1.5)
            else:
                ion_names = list(ions_data.keys())
                for ion_name in ion_names:
                    for range_name in distance_ranges:
                        ts = ions_data[ion_name][range_name]['time_series']
                        frames = np.arange(len(ts))
                        ax3.plot(frames, ts, label=f'{ion_name}-{range_name}', 
                                alpha=0.7, color=colors[range_name], linewidth=1.5)
            
            ax3.set_xlabel('Frame', fontweight='bold', fontsize=12)
            ax3.set_ylabel('Count', fontweight='bold', fontsize=12)
            ax3.set_title('Ion Adsorption Time Series', fontweight='bold', fontsize=14)
            ax3.legend(fontsize=8, ncol=2, framealpha=0.9)
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Organics time series
            ax4 = axes[1, 1]
            
            if organics_data:
                if has_targets_org:
                    for org_name in organics_data.keys():
                        for target_name in organics_data[org_name].keys():
                            for range_name in distance_ranges:
                                ts = organics_data[org_name][target_name][range_name]['time_series']
                                frames = np.arange(len(ts))
                                ax4.plot(frames, ts, label=f'{org_name}-{target_name}-{range_name}', 
                                        alpha=0.7, color=colors[range_name], linewidth=1.5)
                else:
                    org_names = list(organics_data.keys())
                    for org_name in org_names:
                        for range_name in distance_ranges:
                            ts = organics_data[org_name][range_name]['time_series']
                            frames = np.arange(len(ts))
                            ax4.plot(frames, ts, label=f'{org_name}-{range_name}', 
                                    alpha=0.7, color=colors[range_name], linewidth=1.5)
                
                ax4.set_xlabel('Frame', fontweight='bold', fontsize=12)
                ax4.set_ylabel('Atom Count', fontweight='bold', fontsize=12)
                ax4.set_title('Organic Adsorption Time Series', fontweight='bold', fontsize=14)
                ax4.legend(fontsize=8, ncol=2, framealpha=0.9)
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No organic data', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=14)
                ax4.set_title('Organic Time Series (No Data)', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✓ Competitive adsorption plot saved: {filename}")
        
        plt.show()
        return fig, axes
    
    def _plot_stacked_bars_competitive(self, ions_data, organics_data, distance_ranges,
                                       colors, show_values, show_time_series,
                                       figsize, dpi, save_plots, filename):
        """Helper: Create stacked bar charts for competitive adsorption"""
        
        figsize = figsize or (14, 10 if show_time_series else 6)
        
        if show_time_series:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            axes = np.array([[axes[0], axes[1]]])
        
        # Plot 1: Ions stacked bars
        ax1 = axes[0, 0]
        ion_names = list(ions_data.keys())
        x_pos = np.arange(len(ion_names))
        
        bottom = np.zeros(len(ion_names))
        for range_name in distance_ranges:
            means = np.array([ions_data[ion][range_name]['mean'] for ion in ion_names])
            bars = ax1.bar(x_pos, means, label=range_name, color=colors[range_name],
                          alpha=0.8, bottom=bottom)
            
            if show_values:
                for i, (bar, mean) in enumerate(zip(bars, means)):
                    if mean > 0.5:  # Only show if segment is large enough
                        ax1.text(bar.get_x() + bar.get_width()/2, 
                                bottom[i] + mean/2,
                                f'{mean:.1f}', ha='center', va='center', 
                                fontsize=9, fontweight='bold')
            
            bottom += means
        
        ax1.set_xlabel('Ion Type', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Total Count', fontweight='bold', fontsize=12)
        ax1.set_title('Ion Adsorption (Stacked)', fontweight='bold', fontsize=14)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(ion_names, rotation=45, ha='right')
        ax1.legend(title='Distance Range', framealpha=0.9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Organics stacked bars
        ax2 = axes[0, 1]
        org_names = list(organics_data.keys())
        x_pos = np.arange(len(org_names))
        
        bottom = np.zeros(len(org_names))
        for range_name in distance_ranges:
            means = np.array([organics_data[org][range_name]['mean'] for org in org_names])
            bars = ax2.bar(x_pos, means, label=range_name, color=colors[range_name],
                          alpha=0.8, bottom=bottom)
            
            if show_values:
                for i, (bar, mean) in enumerate(zip(bars, means)):
                    if mean > 0.5:
                        ax2.text(bar.get_x() + bar.get_width()/2, 
                                bottom[i] + mean/2,
                                f'{mean:.1f}', ha='center', va='center', 
                                fontsize=9, fontweight='bold')
            
            bottom += means
        
        ax2.set_xlabel('Organic Part', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Total Atom Count', fontweight='bold', fontsize=12)
        ax2.set_title('Organic Adsorption (Stacked)', fontweight='bold', fontsize=14)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(org_names, rotation=45, ha='right')
        ax2.legend(title='Distance Range', framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        if show_time_series:
            # Use same time series plots as grouped bars
            ax3 = axes[1, 0]
            for ion_name in ion_names:
                for range_name in distance_ranges:
                    ts = ions_data[ion_name][range_name]['time_series']
                    frames = np.arange(len(ts))
                    ax3.plot(frames, ts, label=f'{ion_name}-{range_name}', 
                            alpha=0.7, color=colors[range_name], linewidth=1.5)
            
            ax3.set_xlabel('Frame', fontweight='bold', fontsize=12)
            ax3.set_ylabel('Count', fontweight='bold', fontsize=12)
            ax3.set_title('Ion Adsorption Time Series', fontweight='bold', fontsize=14)
            ax3.legend(fontsize=8, ncol=2, framealpha=0.9)
            ax3.grid(True, alpha=0.3)
            
            ax4 = axes[1, 1]
            for org_name in org_names:
                for range_name in distance_ranges:
                    ts = organics_data[org_name][range_name]['time_series']
                    frames = np.arange(len(ts))
                    ax4.plot(frames, ts, label=f'{org_name}-{range_name}', 
                            alpha=0.7, color=colors[range_name], linewidth=1.5)
            
            ax4.set_xlabel('Frame', fontweight='bold', fontsize=12)
            ax4.set_ylabel('Atom Count', fontweight='bold', fontsize=12)
            ax4.set_title('Organic Adsorption Time Series', fontweight='bold', fontsize=14)
            ax4.legend(fontsize=8, ncol=2, framealpha=0.9)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✓ Competitive adsorption plot saved: {filename}")
        
        plt.show()
        return fig, axes
    
    def _plot_heatmap_competitive(self, ions_data, organics_data, distance_ranges,
                                  figsize, dpi, save_plots, filename):
        """Helper: Create heatmap for competitive adsorption"""
        
        figsize = figsize or (12, 8)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Ions heatmap
        ion_names = list(ions_data.keys())
        ion_matrix = np.array([[ions_data[ion][range_name]['mean'] 
                               for range_name in distance_ranges]
                              for ion in ion_names])
        
        im1 = ax1.imshow(ion_matrix, aspect='auto', cmap='YlOrRd')
        ax1.set_xticks(np.arange(len(distance_ranges)))
        ax1.set_yticks(np.arange(len(ion_names)))
        ax1.set_xticklabels(distance_ranges)
        ax1.set_yticklabels(ion_names)
        ax1.set_xlabel('Distance Range', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Ion Type', fontweight='bold', fontsize=12)
        ax1.set_title('Ion Adsorption Heatmap', fontweight='bold', fontsize=14)
        
        # Add values to heatmap
        for i in range(len(ion_names)):
            for j in range(len(distance_ranges)):
                text = ax1.text(j, i, f'{ion_matrix[i, j]:.1f}',
                               ha='center', va='center', color='black', fontsize=10)
        
        plt.colorbar(im1, ax=ax1, label='Mean Count')
        
        # Organics heatmap
        org_names = list(organics_data.keys())
        org_matrix = np.array([[organics_data[org][range_name]['mean'] 
                               for range_name in distance_ranges]
                              for org in org_names])
        
        im2 = ax2.imshow(org_matrix, aspect='auto', cmap='YlGnBu')
        ax2.set_xticks(np.arange(len(distance_ranges)))
        ax2.set_yticks(np.arange(len(org_names)))
        ax2.set_xticklabels(distance_ranges)
        ax2.set_yticklabels(org_names, rotation=0, ha='right')
        ax2.set_xlabel('Distance Range', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Organic Part', fontweight='bold', fontsize=12)
        ax2.set_title('Organic Adsorption Heatmap', fontweight='bold', fontsize=14)
        
        # Add values
        for i in range(len(org_names)):
            for j in range(len(distance_ranges)):
                text = ax2.text(j, i, f'{org_matrix[i, j]:.1f}',
                               ha='center', va='center', color='black', fontsize=10)
        
        plt.colorbar(im2, ax=ax2, label='Mean Atom Count')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✓ Competitive adsorption heatmap saved: {filename}")
        
        plt.show()
        return fig, (ax1, ax2)
    
    def _plot_time_series_competitive(self, ions_data, organics_data, distance_ranges,
                                      colors, figsize, dpi, save_plots, filename):
        """Helper: Create time series plots for competitive adsorption"""
        
        figsize = figsize or (14, 10)
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Ions time series
        ax1 = axes[0]
        ion_names = list(ions_data.keys())
        for ion_name in ion_names:
            for range_name in distance_ranges:
                ts = ions_data[ion_name][range_name]['time_series']
                frames = np.arange(len(ts))
                ax1.plot(frames, ts, label=f'{ion_name}-{range_name}', 
                        alpha=0.7, color=colors[range_name], linewidth=2)
        
        ax1.set_xlabel('Frame', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Ion Count', fontweight='bold', fontsize=12)
        ax1.set_title('Ion Adsorption Time Series', fontweight='bold', fontsize=14)
        ax1.legend(fontsize=9, ncol=3, framealpha=0.9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Organics time series
        ax2 = axes[1]
        org_names = list(organics_data.keys())
        for org_name in org_names:
            for range_name in distance_ranges:
                ts = organics_data[org_name][range_name]['time_series']
                frames = np.arange(len(ts))
                ax2.plot(frames, ts, label=f'{org_name}-{range_name}', 
                        alpha=0.7, color=colors[range_name], linewidth=2)
        
        ax2.set_xlabel('Frame', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Atom Count', fontweight='bold', fontsize=12)
        ax2.set_title('Organic Adsorption Time Series', fontweight='bold', fontsize=14)
        ax2.legend(fontsize=9, ncol=3, framealpha=0.9, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✓ Competitive adsorption time series saved: {filename}")
        
        plt.show()
        return fig, axes
    
    # =========================================================================
    # ORGANIC CONFORMATION PLOTTING
    # =========================================================================
    
    def plot_organic_conformations(self, organic_names=None, save_plots=False, 
                                   filename='organic_conformations.png',
                                   figsize=None, dpi=None):
        """
        Plot organic molecule conformation analysis.
        
        Parameters
        ----------
        organic_names : list, optional
            List of organic molecule names to plot. If None, plots all.
        save_plots : bool
            Whether to save figure
        filename : str
            Output filename if saving
        figsize : tuple, optional
            Figure size
        dpi : int, optional
            Resolution for saved figure
        
        Returns
        -------
        fig, axes : matplotlib figure and axes objects
        """
        self._validate_analysis()
        
        if 'organic_conformations' not in self.analysis.results:
            raise ValueError("No organic conformation results found. Run analyze_organic_conformations() first.")
        
        conf_data = self.analysis.results['organic_conformations']
        
        if organic_names is None:
            organic_names = list(conf_data.keys())
        else:
            conf_data = {k: v for k, v in conf_data.items() if k in organic_names}
        
        if not conf_data:
            print("No organic conformations to plot")
            return
        
        figsize = figsize or (14, 6 * len(organic_names))
        dpi = dpi or self.default_dpi
        
        fig, axes = plt.subplots(len(organic_names), 3, figsize=figsize, squeeze=False)
        
        for row, (org_name, data) in enumerate(conf_data.items()):
            # Plot 1: Radius of gyration
            ax1 = axes[row, 0]
            rg_ts = data['radius_of_gyration']['time_series']
            frames = np.arange(len(rg_ts))
            ax1.plot(frames, rg_ts, linewidth=1.5, alpha=0.7, color='blue')
            ax1.axhline(data['radius_of_gyration']['mean'], color='red', 
                       linestyle='--', linewidth=2, label=f"Mean: {data['radius_of_gyration']['mean']:.2f} Å")
            ax1.fill_between(frames,
                            data['radius_of_gyration']['mean'] - data['radius_of_gyration']['std'],
                            data['radius_of_gyration']['mean'] + data['radius_of_gyration']['std'],
                            alpha=0.3, color='red')
            ax1.set_xlabel('Frame', fontweight='bold')
            ax1.set_ylabel('Rg (Å)', fontweight='bold')
            ax1.set_title(f'{org_name}: Radius of Gyration', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Aspect ratio
            ax2 = axes[row, 1]
            ar_ts = data['aspect_ratio']['time_series']
            frames = np.arange(len(ar_ts))
            ax2.plot(frames, ar_ts, linewidth=1.5, alpha=0.7, color='green')
            ax2.axhline(data['aspect_ratio']['mean'], color='red', 
                       linestyle='--', linewidth=2, label=f"Mean: {data['aspect_ratio']['mean']:.2f}")
            ax2.fill_between(frames,
                            data['aspect_ratio']['mean'] - data['aspect_ratio']['std'],
                            data['aspect_ratio']['mean'] + data['aspect_ratio']['std'],
                            alpha=0.3, color='red')
            ax2.set_xlabel('Frame', fontweight='bold')
            ax2.set_ylabel('Aspect Ratio', fontweight='bold')
            ax2.set_title(f'{org_name}: Aspect Ratio', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Distribution histograms
            ax3 = axes[row, 2]
            ax3_twin = ax3.twinx()
            
            # Rg histogram
            ax3.hist(rg_ts, bins=30, alpha=0.5, color='blue', label='Rg', density=True)
            ax3.set_xlabel('Value', fontweight='bold')
            ax3.set_ylabel('Rg Probability Density', fontweight='bold', color='blue')
            ax3.tick_params(axis='y', labelcolor='blue')
            
            # Aspect ratio histogram
            ax3_twin.hist(ar_ts, bins=30, alpha=0.5, color='green', label='Aspect Ratio', density=True)
            ax3_twin.set_ylabel('Aspect Ratio Probability Density', fontweight='bold', color='green')
            ax3_twin.tick_params(axis='y', labelcolor='green')
            
            ax3.set_title(f'{org_name}: Distributions', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Organic conformation plot saved to {filename}")
        
        plt.show()
        
        return fig, axes
    
    # =========================================================================
    # THREE-COMPONENT BRIDGE PLOTTING
    # =========================================================================
    
    def plot_three_component_bridges(self, save_plots=False, filename='three_component_bridges.png',
                                     figsize=None, dpi=None, show_time_series=True):
        """
        Plot three-component bridge analysis.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save figure
        filename : str
            Output filename if saving
        figsize : tuple, optional
            Figure size
        dpi : int, optional
            Resolution for saved figure
        show_time_series : bool
            Whether to show time series data
        
        Returns
        -------
        fig, axes : matplotlib figure and axes objects
        """
        self._validate_analysis()
        
        if 'three_component_bridges' not in self.analysis.results:
            raise ValueError("No three-component bridge results found. Run analyze_three_component_bridges() first.")
        
        bridge_data = self.analysis.results['three_component_bridges']
        
        figsize = figsize or (14, 8 if not show_time_series else 12)
        dpi = dpi or self.default_dpi
        
        if show_time_series:
            fig, axes = plt.subplots(2, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axes = [ax]
        
        # Plot 1: Bar plot of mean bridge counts
        ax1 = axes[0]
        bridge_types = list(bridge_data.keys())
        means = [bridge_data[bt]['mean'] for bt in bridge_types]
        stds = [bridge_data[bt]['std'] for bt in bridge_types]
        maxs = [bridge_data[bt]['max'] for bt in bridge_types]
        
        x_pos = np.arange(len(bridge_types))
        bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                      color=self.default_colors[:len(bridge_types)])
        
        ax1.set_xlabel('Bridge Type', fontweight='bold')
        ax1.set_ylabel('Mean Number of Bridges', fontweight='bold')
        ax1.set_title('Three-Component Bridge Formation', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(bridge_types, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, mean, max_val) in enumerate(zip(bars, means, maxs)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f'{mean:.1f}\n(max: {max_val:.0f})', ha='center', va='bottom', fontsize=9)
        
        if show_time_series:
            # Plot 2: Time series of bridge counts
            ax2 = axes[1]
            for i, bridge_type in enumerate(bridge_types):
                time_series = bridge_data[bridge_type]['time_series']
                frames = np.arange(len(time_series))
                ax2.plot(frames, time_series, label=bridge_type, alpha=0.7,
                        color=self.default_colors[i % len(self.default_colors)], linewidth=1.5)
            
            ax2.set_xlabel('Frame', fontweight='bold')
            ax2.set_ylabel('Number of Bridges', fontweight='bold')
            ax2.set_title('Bridge Formation Dynamics', fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Three-component bridge plot saved to {filename}")
        
        plt.show()
        
        return fig, axes
    
    # =========================================================================
    # HYDRATION SHELL COMPETITION PLOTTING
    # =========================================================================
    
    def plot_hydration_shell_competition(self, shell_cutoffs=None, save_plots=False,
                                         filename='hydration_competition.png',
                                         figsize=None, dpi=None):
        """
        Plot hydration shell competition analysis.
        
        Parameters
        ----------
        shell_cutoffs : list, optional
            List of shell cutoffs to plot (e.g., ['shell_3.5A', 'shell_5.0A'])
            If None, plots all available shells
        save_plots : bool
            Whether to save figure
        filename : str
            Output filename if saving
        figsize : tuple, optional
            Figure size
        dpi : int, optional
            Resolution for saved figure
        
        Returns
        -------
        fig, axes : matplotlib figure and axes objects
        """
        self._validate_analysis()
        
        if 'hydration_shell_competition' not in self.analysis.results:
            raise ValueError("No hydration shell competition results found. Run analyze_hydration_shell_competition() first.")
        
        hyd_data = self.analysis.results['hydration_shell_competition']
        
        # Get available shells
        ion_shells = list(hyd_data['ion_hydration'].keys())
        if shell_cutoffs is None:
            shell_cutoffs = ion_shells
        else:
            shell_cutoffs = [s for s in shell_cutoffs if s in ion_shells]
        
        if not shell_cutoffs:
            print("No hydration shells to plot")
            return
        
        n_shells = len(shell_cutoffs)
        figsize = figsize or (14, 5 * n_shells)
        dpi = dpi or self.default_dpi
        
        fig, axes = plt.subplots(n_shells, 2, figsize=figsize, squeeze=False)
        
        for row, shell_name in enumerate(shell_cutoffs):
            # Plot 1: Ion hydration numbers
            ax1 = axes[row, 0]
            ion_data = hyd_data['ion_hydration'][shell_name]
            ion_names = list(ion_data.keys())
            ion_means = [ion_data[name]['mean'] for name in ion_names]
            ion_stds = [ion_data[name]['std'] for name in ion_names]
            
            x_pos = np.arange(len(ion_names))
            bars = ax1.bar(x_pos, ion_means, yerr=ion_stds, capsize=5, alpha=0.7,
                          color=self.default_colors[:len(ion_names)])
            ax1.set_xlabel('Ion Type', fontweight='bold')
            ax1.set_ylabel('Hydration Number', fontweight='bold')
            ax1.set_title(f'Ion Hydration ({shell_name})', fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(ion_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (bar, mean) in enumerate(zip(bars, ion_means)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
            
            # Plot 2: Organic hydration numbers
            ax2 = axes[row, 1]
            org_data = hyd_data['organic_hydration'][shell_name]
            org_names = list(org_data.keys())
            org_means = [org_data[name]['mean'] for name in org_names]
            org_stds = [org_data[name]['std'] for name in org_names]
            
            x_pos = np.arange(len(org_names))
            bars = ax2.bar(x_pos, org_means, yerr=org_stds, capsize=5, alpha=0.7,
                          color=self.default_colors[:len(org_names)])
            ax2.set_xlabel('Organic Type', fontweight='bold')
            ax2.set_ylabel('Hydration Number', fontweight='bold')
            ax2.set_title(f'Organic Hydration ({shell_name})', fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(org_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (bar, mean) in enumerate(zip(bars, org_means)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Hydration competition plot saved to {filename}")
        
        plt.show()
        
        return fig, axes
    
    # =========================================================================
    # STRATIFIED ADSORPTION (DENSITY PROFILES) PLOTTING
    # =========================================================================
    
    def plot_stratified_adsorption(self, components=None, save_plots=False,
                                   filename='stratified_adsorption.png',
                                   figsize=None, dpi=None, relative_position=True):
        """
        Plot stratified (layered) adsorption density profiles.
        
        Parameters
        ----------
        components : list, optional
            List of components to plot (e.g., ['clay', 'water', 'ion_name'])
            If None, plots all available components
        save_plots : bool
            Whether to save figure
        filename : str
            Output filename if saving
        figsize : tuple, optional
            Figure size
        dpi : int, optional
            Resolution for saved figure
        relative_position : bool
            If True, plot relative to clay surface; if False, plot absolute z position
        
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        """
        self._validate_analysis()
        
        if 'stratified_adsorption' not in self.analysis.results:
            raise ValueError("No stratified adsorption results found. Run analyze_stratified_adsorption() first.")
        
        strat_data = self.analysis.results['stratified_adsorption']
        density_profiles = strat_data['density_profiles']
        
        if components is None:
            components = list(density_profiles.keys())
        else:
            density_profiles = {k: v for k, v in density_profiles.items() if k in components}
        
        if not density_profiles:
            print("No density profiles to plot")
            return
        
        figsize = figsize or (12, 8)
        dpi = dpi or self.default_dpi
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Choose x-axis data
        if relative_position:
            x_data = strat_data['relative_positions']
            xlabel = 'Distance from Clay Surface (Å)'
        else:
            x_data = strat_data['bin_centers']
            xlabel = 'Z Position (Å)'
        
        # Plot each component
        for i, (comp_name, density) in enumerate(density_profiles.items()):
            color = self.default_colors[i % len(self.default_colors)]
            linestyle = '-' if comp_name != 'clay' else '--'
            linewidth = 2.5 if comp_name in ['clay', 'water'] else 2.0
            
            ax.plot(x_data, density, label=comp_name, color=color, 
                   linestyle=linestyle, linewidth=linewidth, alpha=0.8)
        
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=12)
        ax.set_ylabel('Density (counts/bin)', fontweight='bold', fontsize=12)
        ax.set_title('Stratified Adsorption Profiles', fontweight='bold', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Mark clay surface position
        if relative_position:
            ax.axvline(0, color='k', linestyle=':', linewidth=2, alpha=0.7, label='Clay Surface')
        else:
            ax.axvline(strat_data['clay_surface_position'], color='k', linestyle=':', 
                      linewidth=2, alpha=0.7, label='Clay Surface')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Stratified adsorption plot saved to {filename}")
        
        plt.show()
        
        return fig, ax
    
    # =========================================================================
    # EXCHANGE KINETICS PLOTTING
    # =========================================================================
    
    def plot_exchange_kinetics(self, species=None, save_plots=False,
                               filename='exchange_kinetics.png',
                               figsize=None, dpi=None):
        """
        Plot exchange kinetics and residence time analysis.
        
        Parameters
        ----------
        species : list, optional
            List of species to plot (ion/organic names)
            If None, plots all available species
        save_plots : bool
            Whether to save figure
        filename : str
            Output filename if saving
        figsize : tuple, optional
            Figure size
        dpi : int, optional
            Resolution for saved figure
        
        Returns
        -------
        fig, axes : matplotlib figure and axes objects
        """
        self._validate_analysis()
        
        if 'exchange_kinetics' not in self.analysis.results:
            raise ValueError("No exchange kinetics results found. Run analyze_exchange_kinetics() first.")
        
        kin_data = self.analysis.results['exchange_kinetics']
        
        residence_times = kin_data['residence_times']
        exchange_rates = kin_data['exchange_rates']
        
        if species is None:
            species = list(residence_times.keys())
        else:
            residence_times = {k: v for k, v in residence_times.items() if k in species}
            exchange_rates = {k: v for k, v in exchange_rates.items() if k in species}
        
        if not residence_times:
            print("No exchange kinetics to plot")
            return
        
        figsize = figsize or (14, 10)
        dpi = dpi or self.default_dpi
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Mean residence times (bar plot)
        ax1 = axes[0, 0]
        species_names = list(residence_times.keys())
        means = [residence_times[sp]['mean'] for sp in species_names]
        stds = [residence_times[sp]['std'] for sp in species_names]
        
        x_pos = np.arange(len(species_names))
        bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                      color=self.default_colors[:len(species_names)])
        ax1.set_xlabel('Species', fontweight='bold')
        ax1.set_ylabel('Mean Residence Time (ps)', fontweight='bold')
        ax1.set_title('Surface Residence Times', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(species_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f'{mean:.1f} ps', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Exchange rates (bar plot)
        ax2 = axes[0, 1]
        rates = [exchange_rates[sp] for sp in species_names]
        
        bars = ax2.bar(x_pos, rates, alpha=0.7, color=self.default_colors[:len(species_names)])
        ax2.set_xlabel('Species', fontweight='bold')
        ax2.set_ylabel('Exchange Rate (events/ps)', fontweight='bold')
        ax2.set_title('Adsorption-Desorption Exchange Rates', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(species_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f'{rate:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Residence time distributions (histograms)
        ax3 = axes[1, 0]
        for i, sp in enumerate(species_names):
            dist = residence_times[sp]['distribution']
            ax3.hist(dist, bins=30, alpha=0.5, label=sp, 
                    color=self.default_colors[i % len(self.default_colors)], density=True)
        ax3.set_xlabel('Residence Time (ps)', fontweight='bold')
        ax3.set_ylabel('Probability Density', fontweight='bold')
        ax3.set_title('Residence Time Distributions', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Comparison scatter plot (residence time vs exchange rate)
        ax4 = axes[1, 1]
        ax4.scatter(means, rates, s=200, alpha=0.7, 
                   c=self.default_colors[:len(species_names)], edgecolors='black', linewidth=2)
        
        # Add labels for each point
        for i, sp in enumerate(species_names):
            ax4.annotate(sp, (means[i], rates[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
        
        ax4.set_xlabel('Mean Residence Time (ps)', fontweight='bold')
        ax4.set_ylabel('Exchange Rate (events/ps)', fontweight='bold')
        ax4.set_title('Residence Time vs Exchange Rate', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Exchange kinetics plot saved to {filename}")
        
        plt.show()
        
        return fig, axes
    
    # =========================================================================
    # SELECTIVITY COEFFICIENTS PLOTTING
    # =========================================================================
    
    def plot_selectivity_coefficients(self, save_plots=False, filename='selectivity_coefficients.png',
                                      figsize=None, dpi=None, log_scale=False):
        """
        Plot selectivity coefficients for competitive binding.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save figure
        filename : str
            Output filename if saving
        figsize : tuple, optional
            Figure size
        dpi : int, optional
            Resolution for saved figure
        log_scale : bool
            Whether to use log scale for y-axis
        
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        """
        self._validate_analysis()
        
        if 'selectivity_coefficients' not in self.analysis.results:
            raise ValueError("No selectivity coefficients found. Run calculate_selectivity_coefficients() first.")
        
        sel_data = self.analysis.results['selectivity_coefficients']
        
        if not sel_data:
            print("No selectivity coefficients to plot")
            return
        
        figsize = figsize or (12, 8)
        dpi = dpi or self.default_dpi
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Prepare data
        pairs = list(sel_data.keys())
        coefficients = list(sel_data.values())
        
        # Sort by coefficient value
        sorted_indices = np.argsort(coefficients)[::-1]  # Descending order
        pairs = [pairs[i] for i in sorted_indices]
        coefficients = [coefficients[i] for i in sorted_indices]
        
        x_pos = np.arange(len(pairs))
        
        # Color code by selectivity strength
        colors = []
        for coef in coefficients:
            if coef > 2.0:
                colors.append('darkgreen')  # Strong preference
            elif coef > 1.5:
                colors.append('lightgreen')  # Moderate preference
            elif coef > 1.0:
                colors.append('yellow')  # Weak preference
            elif coef > 0.5:
                colors.append('orange')  # Weak reverse preference
            else:
                colors.append('red')  # Strong reverse preference
        
        bars = ax.bar(x_pos, coefficients, alpha=0.7, color=colors, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Species Comparison', fontweight='bold', fontsize=12)
        ax.set_ylabel('Selectivity Coefficient', fontweight='bold', fontsize=12)
        ax.set_title('Competitive Binding Selectivity', fontweight='bold', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pairs, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(1.0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Equal preference')
        
        if log_scale:
            ax.set_yscale('log')
        
        # Add value labels
        for i, (bar, coef) in enumerate(zip(bars, coefficients)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f'{coef:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        # Add legend for color coding
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', alpha=0.7, label='Strong preference (>2.0)'),
            Patch(facecolor='lightgreen', alpha=0.7, label='Moderate preference (1.5-2.0)'),
            Patch(facecolor='yellow', alpha=0.7, label='Weak preference (1.0-1.5)'),
            Patch(facecolor='orange', alpha=0.7, label='Weak reverse (0.5-1.0)'),
            Patch(facecolor='red', alpha=0.7, label='Strong reverse (<0.5)')
        ]
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9, fontsize=9)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Selectivity coefficients plot saved to {filename}")
        
        plt.show()
        
        return fig, ax
    
    # =========================================================================
    # COMPREHENSIVE SUMMARY PLOTTING
    # =========================================================================
    
    def plot_comprehensive_summary(self, save_plots=False, filename='comprehensive_summary.png',
                                   figsize=None, dpi=None):
        """
        Create a comprehensive summary figure with key results from all analyses.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save figure
        filename : str
            Output filename if saving
        figsize : tuple, optional
            Figure size
        dpi : int, optional
            Resolution for saved figure
        
        Returns
        -------
        fig, axes : matplotlib figure and axes objects
        """
        self._validate_analysis()
        
        figsize = figsize or (18, 12)
        dpi = dpi or self.default_dpi
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. System composition (top-left)
        if hasattr(self.analysis, 'ions') and hasattr(self.analysis, 'organics'):
            ax1 = fig.add_subplot(gs[0, 0])
            components = ['Clay', 'Water']
            counts = [len(self.analysis.clay), len(self.analysis.water) // 3]
            
            for ion_name, ion_atoms in self.analysis.ions.items():
                components.append(ion_name)
                counts.append(len(ion_atoms))
            
            for org_name, org_atoms in self.analysis.organics.items():
                components.append(org_name)
                counts.append(len(org_atoms))
            
            ax1.bar(range(len(components)), counts, alpha=0.7, color=self.default_colors[:len(components)])
            ax1.set_xticks(range(len(components)))
            ax1.set_xticklabels(components, rotation=45, ha='right')
            ax1.set_ylabel('Count', fontweight='bold')
            ax1.set_title('System Composition', fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Competitive adsorption (top-middle)
        if 'competitive_adsorption' in self.analysis.results:
            ax2 = fig.add_subplot(gs[0, 1])
            ca_data = self.analysis.results['competitive_adsorption']
            
            # Combine ion and organic contacts
            all_species = []
            all_contacts = []
            
            for ion_name, data in ca_data['ion_surface_contacts'].items():
                all_species.append(ion_name)
                all_contacts.append(data['mean'])
            
            for org_name, data in ca_data['organic_surface_contacts'].items():
                all_species.append(org_name)
                all_contacts.append(data['mean'])
            
            ax2.barh(range(len(all_species)), all_contacts, alpha=0.7, 
                    color=self.default_colors[:len(all_species)])
            ax2.set_yticks(range(len(all_species)))
            ax2.set_yticklabels(all_species)
            ax2.set_xlabel('Surface Contacts', fontweight='bold')
            ax2.set_title('Surface Adsorption', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Bridge types (top-right)
        if 'three_component_bridges' in self.analysis.results:
            ax3 = fig.add_subplot(gs[0, 2])
            bridge_data = self.analysis.results['three_component_bridges']
            
            bridge_types = list(bridge_data.keys())
            bridge_counts = [bridge_data[bt]['mean'] for bt in bridge_types]
            
            # Simplify labels
            short_labels = [bt.replace('-', '\n') for bt in bridge_types]
            
            ax3.bar(range(len(bridge_types)), bridge_counts, alpha=0.7, 
                   color=self.default_colors[:len(bridge_types)])
            ax3.set_xticks(range(len(bridge_types)))
            ax3.set_xticklabels(short_labels, rotation=0, fontsize=9)
            ax3.set_ylabel('Mean Bridges', fontweight='bold')
            ax3.set_title('Bridge Formation', fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4-5. Stratified adsorption (middle row, spans 2 columns)
        if 'stratified_adsorption' in self.analysis.results:
            ax4 = fig.add_subplot(gs[1, :2])
            strat_data = self.analysis.results['stratified_adsorption']
            
            x_data = strat_data['relative_positions']
            for i, (comp_name, density) in enumerate(strat_data['density_profiles'].items()):
                if i < len(self.default_colors):
                    ax4.plot(x_data, density, label=comp_name, linewidth=2, 
                            color=self.default_colors[i], alpha=0.8)
            
            ax4.axvline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
            ax4.set_xlabel('Distance from Clay (Å)', fontweight='bold')
            ax4.set_ylabel('Density', fontweight='bold')
            ax4.set_title('Stratified Adsorption', fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
        
        # 6. Hydration competition (middle-right)
        if 'hydration_shell_competition' in self.analysis.results:
            ax5 = fig.add_subplot(gs[1, 2])
            hyd_data = self.analysis.results['hydration_shell_competition']
            
            # Use first shell
            shell_name = list(hyd_data['ion_hydration'].keys())[0]
            ion_data = hyd_data['ion_hydration'][shell_name]
            
            species = list(ion_data.keys())
            hydration = [ion_data[sp]['mean'] for sp in species]
            
            ax5.bar(range(len(species)), hydration, alpha=0.7, color=self.default_colors[:len(species)])
            ax5.set_xticks(range(len(species)))
            ax5.set_xticklabels(species, rotation=45, ha='right', fontsize=9)
            ax5.set_ylabel('Hydration Number', fontweight='bold')
            ax5.set_title(f'Hydration ({shell_name})', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
        
        # 7. Exchange kinetics residence times (bottom-left)
        if 'exchange_kinetics' in self.analysis.results:
            ax6 = fig.add_subplot(gs[2, 0])
            kin_data = self.analysis.results['exchange_kinetics']
            
            if kin_data['residence_times']:
                species = list(kin_data['residence_times'].keys())
                res_times = [kin_data['residence_times'][sp]['mean'] for sp in species]
                
                ax6.barh(range(len(species)), res_times, alpha=0.7, color=self.default_colors[:len(species)])
                ax6.set_yticks(range(len(species)))
                ax6.set_yticklabels(species, fontsize=9)
                ax6.set_xlabel('Residence Time (ps)', fontweight='bold')
                ax6.set_title('Surface Residence', fontweight='bold')
                ax6.grid(True, alpha=0.3, axis='x')
        
        # 8. Exchange rates (bottom-middle)
        if 'exchange_kinetics' in self.analysis.results:
            ax7 = fig.add_subplot(gs[2, 1])
            
            if kin_data['exchange_rates']:
                species = list(kin_data['exchange_rates'].keys())
                rates = [kin_data['exchange_rates'][sp] for sp in species]
                
                ax7.bar(range(len(species)), rates, alpha=0.7, color=self.default_colors[:len(species)])
                ax7.set_xticks(range(len(species)))
                ax7.set_xticklabels(species, rotation=45, ha='right', fontsize=9)
                ax7.set_ylabel('Exchange Rate', fontweight='bold')
                ax7.set_title('Exchange Rates', fontweight='bold')
                ax7.grid(True, alpha=0.3, axis='y')
        
        # 9. Selectivity coefficients (bottom-right)
        if 'selectivity_coefficients' in self.analysis.results:
            ax8 = fig.add_subplot(gs[2, 2])
            sel_data = self.analysis.results['selectivity_coefficients']
            
            if sel_data:
                # Show top 5 selectivities
                pairs = list(sel_data.keys())[:5]
                coeffs = [sel_data[p] for p in pairs]
                
                short_pairs = [p.replace('_vs_', '\nvs\n') for p in pairs]
                
                ax8.barh(range(len(pairs)), coeffs, alpha=0.7, color=self.default_colors[:len(pairs)])
                ax8.set_yticks(range(len(pairs)))
                ax8.set_yticklabels(short_pairs, fontsize=8)
                ax8.set_xlabel('Selectivity', fontweight='bold')
                ax8.set_title('Top Selectivities', fontweight='bold')
                ax8.axvline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
                ax8.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Clay-Organic-Ion-Water Analysis Summary', fontweight='bold', fontsize=16)
        
        if save_plots:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Comprehensive summary plot saved to {filename}")
        
        plt.show()
        
        return fig
    
    # =========================================================================
    # ELECTRICAL DOUBLE LAYER PLOTTING
    # =========================================================================
    
    def plot_electrical_double_layer(self, 
                                     save_plots=False,
                                     show_electric_field=True,
                                     show_ion_densities=True,
                                     show_stern_layer=True,
                                     show_adsorption_modes=True,
                                     show_gouy_chapman=True,
                                     show_capacitance=False,
                                     # Figure control
                                     figsize=None,
                                     dpi=300,
                                     save_format='png',
                                     filename=None,
                                     # Legend control
                                     show_legend=True,
                                     legend_frameon=False,
                                     legend_framealpha=0.8,
                                     legend_fontsize=10,
                                     legend_loc='best',
                                     # Text/label control
                                     show_title=True,
                                     title_fontsize=14,
                                     label_fontsize=12,
                                     tick_fontsize=11,
                                     subplot_titles=None,
                                     xlabel='Distance from surface (Å)',
                                     # Surface reference line
                                     show_surface_line=True,
                                     surface_line_color='black',
                                     surface_line_style='-',
                                     surface_line_width=2.0,
                                     surface_line_alpha=0.8,
                                     surface_label='Clay surface',
                                     # Stern layer styling
                                     stern_ihp_color='orange',
                                     stern_ihp_style='--',
                                     stern_ihp_width=1.5,
                                     stern_ihp_alpha=0.7,
                                     stern_ohp_color='green',
                                     stern_ohp_style='--',
                                     stern_ohp_width=1.5,
                                     stern_ohp_alpha=0.7,
                                     stern_fill_region=True,
                                     stern_fill_color='yellow',
                                     stern_fill_alpha=0.1,
                                     show_stern_labels=True,
                                     stern_label_fontsize=9,
                                     # Main line styling
                                     charge_density_color='red',
                                     charge_density_linewidth=2,
                                     potential_color='blue',
                                     potential_linewidth=2,
                                     field_color='green',
                                     field_linewidth=2,
                                     # Ion density styling
                                     ion_colors=None,
                                     ion_linestyle='-',
                                     ion_linewidth=1.5,
                                     ion_alpha=0.8,
                                     # Gouy-Chapman styling
                                     gc_linestyle='--',
                                     gc_linewidth=1.5,
                                     gc_alpha=0.6,
                                     # Grid and reference lines
                                     show_grid=True,
                                     grid_alpha=0.3,
                                     show_zero_line=True,
                                     zero_line_color='gray',
                                     zero_line_style=':',
                                     zero_line_alpha=0.5,
                                     zero_line_width=1.0,
                                     # Electric field direction coloring
                                     show_field_direction_colors=True,
                                     field_positive_color='red',
                                     field_negative_color='blue',
                                     field_direction_alpha=0.2,
                                     # Plot range control
                                     x_range=None,
                                     y_ranges=None):
        """
        Plot comprehensive electrical double layer analysis results.
        
        Creates multi-panel figure showing:
        1. Charge density profile with ion contributions
        2. Electrostatic potential with Stern layer markers
        3. Electric field profile (optional)
        4. Ion density profiles with Gouy-Chapman comparison (optional)
        5. Adsorption mode classification (optional)
        6. Differential capacitance (optional)
        
        Parameters
        ----------
        save_plots : bool, default=False
            Save plot to file
        show_electric_field : bool, default=True
            Include electric field subplot
        show_ion_densities : bool, default=True
            Include ion density profiles subplot
        show_stern_layer : bool, default=True
            Mark Stern layer boundaries (IHP, OHP)
        show_adsorption_modes : bool, default=True
            Show ion adsorption mode classification
        show_gouy_chapman : bool, default=True
            Overlay Gouy-Chapman theory predictions on ion densities
        show_capacitance : bool, default=False
            Include capacitance profile subplot
            
        Figure Control
        --------------
        figsize : tuple, optional
            Figure size (width, height). Auto-calculated if None
        dpi : int, default=300
            Resolution for saved plots
        save_format : str, default='png'
            File format for saved plots
        filename : str, optional
            Custom filename for saved plot
            
        Legend Control
        --------------
        show_legend : bool, default=True
            Show legends on subplots
        legend_frameon : bool, default=False
            Frame around legend
        legend_framealpha : float, default=0.8
            Legend background transparency
        legend_fontsize : int, default=10
            Legend text size
        legend_loc : str, default='best'
            Legend location
            
        Text/Label Control
        ------------------
        show_title : bool, default=True
            Show subplot titles
        title_fontsize : int, default=14
            Subplot title size
        label_fontsize : int, default=12
            Axis label size
        tick_fontsize : int, default=11
            Tick label size
        subplot_titles : dict, optional
            Custom subplot titles
        xlabel : str, default='Distance from surface (Å)'
            X-axis label
            
        Surface Reference Line
        ----------------------
        show_surface_line : bool, default=True
            Show vertical line at surface position
        surface_line_color : str, default='black'
            Surface line color
        surface_line_style : str, default='-'
            Surface line style
        surface_line_width : float, default=2.0
            Surface line width
        surface_line_alpha : float, default=0.8
            Surface line transparency
        surface_label : str, default='Clay surface'
            Surface line label
            
        Stern Layer Styling
        -------------------
        stern_ihp_color : str, default='orange'
            Inner Helmholtz Plane color
        stern_ihp_style : str, default='--'
            IHP line style
        stern_ihp_width : float, default=1.5
            IHP line width
        stern_ihp_alpha : float, default=0.7
            IHP transparency
        stern_ohp_color : str, default='green'
            Outer Helmholtz Plane color
        stern_ohp_style : str, default='--'
            OHP line style
        stern_ohp_width : float, default=1.5
            OHP line width
        stern_ohp_alpha : float, default=0.7
            OHP transparency
        stern_fill_region : bool, default=True
            Fill Stern layer region
        stern_fill_color : str, default='yellow'
            Stern region fill color
        stern_fill_alpha : float, default=0.1
            Stern region transparency
        show_stern_labels : bool, default=True
            Annotate Stern layer boundaries
        stern_label_fontsize : int, default=9
            Stern label text size
            
        Main Line Styling
        -----------------
        charge_density_color : str, default='red'
            Charge density line color
        charge_density_linewidth : float, default=2
            Charge density line width
        potential_color : str, default='blue'
            Potential line color
        potential_linewidth : float, default=2
            Potential line width
        field_color : str, default='green'
            Electric field line color
        field_linewidth : float, default=2
            Electric field line width
            
        Ion Density Styling
        -------------------
        ion_colors : dict, optional
            Colors for each ion type {ion_name: color}
        ion_linestyle : str, default='-'
            Ion density line style
        ion_linewidth : float, default=1.5
            Ion density line width
        ion_alpha : float, default=0.8
            Ion density transparency
            
        Gouy-Chapman Styling
        --------------------
        gc_linestyle : str, default='--'
            Gouy-Chapman line style
        gc_linewidth : float, default=1.5
            Gouy-Chapman line width
        gc_alpha : float, default=0.6
            Gouy-Chapman transparency
            
        Grid and Reference Lines
        -------------------------
        show_grid : bool, default=True
            Show grid lines
        grid_alpha : float, default=0.3
            Grid transparency
        show_zero_line : bool, default=True
            Show horizontal y=0 line
        zero_line_color : str, default='gray'
            Zero line color
        zero_line_style : str, default=':'
            Zero line style
        zero_line_alpha : float, default=0.5
            Zero line transparency
        zero_line_width : float, default=1.0
            Zero line width
            
        Electric Field Direction Coloring
        ----------------------------------
        show_field_direction_colors : bool, default=True
            Color-code field direction regions
        field_positive_color : str, default='red'
            Color for E > 0 regions
        field_negative_color : str, default='blue'
            Color for E < 0 regions
        field_direction_alpha : float, default=0.2
            Field direction transparency
            
        Plot Range Control
        ------------------
        x_range : tuple, optional
            X-axis range (min, max)
        y_ranges : dict, optional
            Y-axis ranges for each subplot {subplot_key: (ymin, ymax)}
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """
        
        self._validate_analysis()
        
        if 'edl_analysis' not in self.analysis.results:
            raise ValueError("No EDL analysis results found. Run analyze_electrical_double_layer_complete() first.")
        
        edl = self.analysis.results['edl_analysis']
        
        # Extract data
        z_centers = edl['z_centers']
        surface_pos = edl['surface_position']
        charge_density = edl['charge_density']
        potential = edl['electrostatic_potential']
        electric_field = edl['electric_field']
        ion_densities = edl['ion_densities']
        
        # Adjust z-axis to be relative to surface
        z_rel = z_centers - surface_pos
        
        # Determine number of subplots
        n_subplots = 2  # charge density + potential (baseline)
        if show_electric_field:
            n_subplots += 1
        if show_ion_densities:
            n_subplots += 1
        if show_adsorption_modes and 'adsorption_modes' in edl:
            n_subplots += 1
        if show_capacitance and 'capacitance' in edl:
            n_subplots += 1
        
        # Figure size
        if figsize is None:
            figsize = (14, 4 * n_subplots)
        
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
        if n_subplots == 1:
            axes = [axes]
        
        # Default ion colors
        if ion_colors is None:
            ion_colors = {}
            color_cycle = ['blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'olive']
            for i, ion_name in enumerate(ion_densities.keys()):
                ion_colors[ion_name] = color_cycle[i % len(color_cycle)]
        
        # Default subplot titles
        if subplot_titles is None:
            subplot_titles = {
                'charge_density': 'Charge Density Profile',
                'potential': 'Electrostatic Potential',
                'electric_field': 'Electric Field (E = -dψ/dz)',
                'ion_densities': 'Ion Density Profiles',
                'adsorption_modes': 'Ion Adsorption Modes',
                'capacitance': 'Differential Capacitance'
            }
        
        subplot_idx = 0
        
        # =====================================================================
        # SUBPLOT 1: CHARGE DENSITY
        # =====================================================================
        ax = axes[subplot_idx]
        subplot_idx += 1
        
        ax.plot(z_rel, charge_density, color=charge_density_color, 
               linewidth=charge_density_linewidth, label='Total charge density')
        
        ax.set_ylabel('Charge Density (e/Å³)', fontsize=label_fontsize)
        if show_title:
            ax.set_title(subplot_titles['charge_density'], fontsize=title_fontsize, fontweight='bold')
        
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
        if show_zero_line:
            ax.axhline(0, color=zero_line_color, linestyle=zero_line_style, 
                      alpha=zero_line_alpha, linewidth=zero_line_width)
        if show_surface_line:
            ax.axvline(0, color=surface_line_color, linestyle=surface_line_style,
                      linewidth=surface_line_width, alpha=surface_line_alpha, label=surface_label)
        
        # Add Stern layer markers
        if show_stern_layer and 'stern_layer' in edl:
            stern = edl['stern_layer']
            if stern['ihp_position'] is not None:
                ihp_rel = stern['ihp_position'] - surface_pos
                ax.axvline(ihp_rel, color=stern_ihp_color, linestyle=stern_ihp_style,
                          linewidth=stern_ihp_width, alpha=stern_ihp_alpha, label='IHP')
                if show_stern_labels:
                    y_pos = ax.get_ylim()[1] * 0.9
                    ax.text(ihp_rel, y_pos, 'IHP', fontsize=stern_label_fontsize,
                           rotation=90, va='top', ha='right', color=stern_ihp_color)
            
            if stern['ohp_position'] is not None:
                ohp_rel = stern['ohp_position'] - surface_pos
                ax.axvline(ohp_rel, color=stern_ohp_color, linestyle=stern_ohp_style,
                          linewidth=stern_ohp_width, alpha=stern_ohp_alpha, label='OHP')
                if show_stern_labels:
                    y_pos = ax.get_ylim()[1] * 0.9
                    ax.text(ohp_rel, y_pos, 'OHP', fontsize=stern_label_fontsize,
                           rotation=90, va='top', ha='right', color=stern_ohp_color)
                
                # Fill Stern layer region
                if stern_fill_region and stern['ihp_position'] is not None:
                    ax.axvspan(ihp_rel, ohp_rel, color=stern_fill_color, alpha=stern_fill_alpha, zorder=0)
        
        if show_legend:
            ax.legend(fontsize=legend_fontsize, loc=legend_loc, frameon=legend_frameon, framealpha=legend_framealpha)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # =====================================================================
        # SUBPLOT 2: ELECTROSTATIC POTENTIAL
        # =====================================================================
        ax = axes[subplot_idx]
        subplot_idx += 1
        
        ax.plot(z_rel, potential, color=potential_color, 
               linewidth=potential_linewidth, label='MD Potential')
        
        ax.set_ylabel('Electrostatic Potential (kT/e)', fontsize=label_fontsize)
        if show_title:
            ax.set_title(subplot_titles['potential'], fontsize=title_fontsize, fontweight='bold')
        
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
        if show_zero_line:
            ax.axhline(0, color=zero_line_color, linestyle=zero_line_style, 
                      alpha=zero_line_alpha, linewidth=zero_line_width)
        if show_surface_line:
            ax.axvline(0, color=surface_line_color, linestyle=surface_line_style,
                      linewidth=surface_line_width, alpha=surface_line_alpha)
        
        # Add Stern layer markers
        if show_stern_layer and 'stern_layer' in edl:
            stern = edl['stern_layer']
            if stern['ihp_position'] is not None:
                ihp_rel = stern['ihp_position'] - surface_pos
                ax.axvline(ihp_rel, color=stern_ihp_color, linestyle=stern_ihp_style,
                          linewidth=stern_ihp_width, alpha=stern_ihp_alpha)
            if stern['ohp_position'] is not None:
                ohp_rel = stern['ohp_position'] - surface_pos
                ax.axvline(ohp_rel, color=stern_ohp_color, linestyle=stern_ohp_style,
                          linewidth=stern_ohp_width, alpha=stern_ohp_alpha)
                if stern_fill_region and stern['ihp_position'] is not None:
                    ax.axvspan(ihp_rel, ohp_rel, color=stern_fill_color, alpha=stern_fill_alpha, zorder=0)
        
        # Add Debye length annotation if available
        if 'debye_length' in edl:
            lambda_d = edl['debye_length'].get('lambda_D_fitted') or edl['debye_length'].get('lambda_D_theoretical')
            if lambda_d and lambda_d < 100:
                ax.text(0.98, 0.95, f'λ$_D$ = {lambda_d:.1f} Å', 
                       transform=ax.transAxes, fontsize=label_fontsize,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if show_legend:
            ax.legend(fontsize=legend_fontsize, loc=legend_loc, frameon=legend_frameon, framealpha=legend_framealpha)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # =====================================================================
        # SUBPLOT 3: ELECTRIC FIELD (optional)
        # =====================================================================
        if show_electric_field:
            ax = axes[subplot_idx]
            subplot_idx += 1
            
            ax.plot(z_rel, electric_field, color=field_color, 
                   linewidth=field_linewidth, label='Electric field')
            
            ax.set_ylabel('Electric Field (kT/e/Å)', fontsize=label_fontsize)
            if show_title:
                ax.set_title(subplot_titles['electric_field'], fontsize=title_fontsize, fontweight='bold')
            
            if show_grid:
                ax.grid(True, alpha=grid_alpha)
            if show_zero_line:
                ax.axhline(0, color=zero_line_color, linestyle=zero_line_style, 
                          alpha=zero_line_alpha, linewidth=zero_line_width)
            if show_surface_line:
                ax.axvline(0, color=surface_line_color, linestyle=surface_line_style,
                          linewidth=surface_line_width, alpha=surface_line_alpha)
            
            # Color-code field direction
            if show_field_direction_colors:
                positive_mask = electric_field > 0
                negative_mask = electric_field < 0
                
                if np.any(positive_mask):
                    ax.fill_between(z_rel, 0, electric_field, where=positive_mask,
                                   color=field_positive_color, alpha=field_direction_alpha,
                                   label='E > 0')
                if np.any(negative_mask):
                    ax.fill_between(z_rel, 0, electric_field, where=negative_mask,
                                   color=field_negative_color, alpha=field_direction_alpha,
                                   label='E < 0')
            
            # Add Stern layer markers
            if show_stern_layer and 'stern_layer' in edl:
                stern = edl['stern_layer']
                if stern['ihp_position'] is not None:
                    ihp_rel = stern['ihp_position'] - surface_pos
                    ax.axvline(ihp_rel, color=stern_ihp_color, linestyle=stern_ihp_style,
                              linewidth=stern_ihp_width, alpha=stern_ihp_alpha)
                if stern['ohp_position'] is not None:
                    ohp_rel = stern['ohp_position'] - surface_pos
                    ax.axvline(ohp_rel, color=stern_ohp_color, linestyle=stern_ohp_style,
                              linewidth=stern_ohp_width, alpha=stern_ohp_alpha)
            
            if show_legend:
                ax.legend(fontsize=legend_fontsize, loc=legend_loc, frameon=legend_frameon, framealpha=legend_framealpha)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # =====================================================================
        # SUBPLOT 4: ION DENSITIES (optional)
        # =====================================================================
        if show_ion_densities:
            ax = axes[subplot_idx]
            subplot_idx += 1
            
            for ion_name, density in ion_densities.items():
                if np.max(density) > 0:
                    color = ion_colors.get(ion_name, 'black')
                    ax.plot(z_rel, density, color=color, linestyle=ion_linestyle,
                           linewidth=ion_linewidth, alpha=ion_alpha, label=f'{ion_name} (MD)')
            
            # Add Gouy-Chapman predictions
            if show_gouy_chapman and 'gouy_chapman_comparison' in edl:
                gc_data = edl['gouy_chapman_comparison']
                for ion_name, density_gc in gc_data['gouy_chapman_ion_densities'].items():
                    if np.max(density_gc) > 0:
                        color = ion_colors.get(ion_name, 'black')
                        ax.plot(z_rel, density_gc, color=color, linestyle=gc_linestyle,
                               linewidth=gc_linewidth, alpha=gc_alpha, label=f'{ion_name} (GC)')
            
            ax.set_ylabel('Ion Density (ions/Å³)', fontsize=label_fontsize)
            if show_title:
                ax.set_title(subplot_titles['ion_densities'], fontsize=title_fontsize, fontweight='bold')
            
            if show_grid:
                ax.grid(True, alpha=grid_alpha)
            if show_surface_line:
                ax.axvline(0, color=surface_line_color, linestyle=surface_line_style,
                          linewidth=surface_line_width, alpha=surface_line_alpha)
            
            # Add Stern layer markers
            if show_stern_layer and 'stern_layer' in edl:
                stern = edl['stern_layer']
                if stern['ihp_position'] is not None:
                    ihp_rel = stern['ihp_position'] - surface_pos
                    ax.axvline(ihp_rel, color=stern_ihp_color, linestyle=stern_ihp_style,
                              linewidth=stern_ihp_width, alpha=stern_ihp_alpha)
                if stern['ohp_position'] is not None:
                    ohp_rel = stern['ohp_position'] - surface_pos
                    ax.axvline(ohp_rel, color=stern_ohp_color, linestyle=stern_ohp_style,
                              linewidth=stern_ohp_width, alpha=stern_ohp_alpha)
                    if stern_fill_region and stern['ihp_position'] is not None:
                        ax.axvspan(ihp_rel, ohp_rel, color=stern_fill_color, alpha=stern_fill_alpha, zorder=0)
            
            if show_legend:
                ax.legend(fontsize=legend_fontsize-2, loc=legend_loc, frameon=legend_frameon, 
                         framealpha=legend_framealpha, ncol=2)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # =====================================================================
        # SUBPLOT 5: ADSORPTION MODES (optional)
        # =====================================================================
        if show_adsorption_modes and 'adsorption_modes' in edl:
            ax = axes[subplot_idx]
            subplot_idx += 1
            
            adsorption = edl['adsorption_modes']
            
            ion_names = list(adsorption.keys())
            x_pos = np.arange(len(ion_names))
            width = 0.25
            
            inner_counts = [adsorption[ion]['inner_sphere_count'] for ion in ion_names]
            outer_counts = [adsorption[ion]['outer_sphere_count'] for ion in ion_names]
            diffuse_counts = [adsorption[ion]['diffuse_layer_count'] for ion in ion_names]
            
            ax.bar(x_pos - width, inner_counts, width, label='Inner-sphere', alpha=0.8, color='red')
            ax.bar(x_pos, outer_counts, width, label='Outer-sphere', alpha=0.8, color='orange')
            ax.bar(x_pos + width, diffuse_counts, width, label='Diffuse layer', alpha=0.8, color='blue')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(ion_names, fontsize=tick_fontsize)
            ax.set_ylabel('Average Ion Count', fontsize=label_fontsize)
            if show_title:
                ax.set_title(subplot_titles['adsorption_modes'], fontsize=title_fontsize, fontweight='bold')
            
            if show_grid:
                ax.grid(True, alpha=grid_alpha, axis='y')
            if show_legend:
                ax.legend(fontsize=legend_fontsize, loc=legend_loc, frameon=legend_frameon, framealpha=legend_framealpha)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # =====================================================================
        # SUBPLOT 6: CAPACITANCE (optional)
        # =====================================================================
        if show_capacitance and 'capacitance' in edl:
            ax = axes[subplot_idx]
            subplot_idx += 1
            
            cap_profile = edl['capacitance']['capacitance_profile']
            
            ax.plot(z_rel, cap_profile, color='purple', linewidth=2, label='Differential capacitance')
            
            ax.set_ylabel('Capacitance (F/m²)', fontsize=label_fontsize)
            if show_title:
                ax.set_title(subplot_titles['capacitance'], fontsize=title_fontsize, fontweight='bold')
            
            if show_grid:
                ax.grid(True, alpha=grid_alpha)
            if show_zero_line:
                ax.axhline(0, color=zero_line_color, linestyle=zero_line_style, 
                          alpha=zero_line_alpha, linewidth=zero_line_width)
            if show_surface_line:
                ax.axvline(0, color=surface_line_color, linestyle=surface_line_style,
                          linewidth=surface_line_width, alpha=surface_line_alpha)
            
            if show_legend:
                ax.legend(fontsize=legend_fontsize, loc=legend_loc, frameon=legend_frameon, framealpha=legend_framealpha)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # Set x-label on bottom subplot
        axes[-1].set_xlabel(xlabel, fontsize=label_fontsize)
        
        # Apply x-range if specified
        if x_range is not None:
            for ax in axes:
                ax.set_xlim(x_range)
        
        # Apply y-ranges if specified
        if y_ranges is not None:
            for key, (ymin, ymax) in y_ranges.items():
                if key < len(axes):
                    axes[key].set_ylim(ymin, ymax)
        
        # Overall title
        if show_title:
            fig.suptitle('Electrical Double Layer Analysis', fontsize=title_fontsize+2, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Save plot
        if save_plots:
            if filename is None:
                filename = f'edl_analysis.{save_format}'
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', format=save_format)
            print(f"EDL analysis plot saved to {filename}")
        
        plt.show()
        
        return fig
    
    def plot_cavity_ion_binding_xy(self, ion_types=None, z_slice_centers=None,
                                show_ion_density=True, ion_density_cmap='Purples',
                                ion_density_alpha=0.7, cavity_marker='o',
                                cavity_colormap='hot', cavity_size_range=(50, 500),
                                cavity_alpha=0.8, cavity_edgecolor='black',
                                cavity_linewidth=1.5, show_colorbar=True,
                                show_empty_cavities=True, empty_cavity_color='gray',
                                empty_cavity_alpha=0.3, empty_cavity_size=50,
                                figsize=(15, 5), dpi=300, save_plots=False,
                                filename=None, save_format='png',
                                title_fontsize=14, label_fontsize=12,
                                tick_fontsize=10, colorbar_label_fontsize=11,
                                show_title=True):
        """
        Plot XY heatmaps of ion density with cavity markers overlaid.
        
        Cavity markers are sized and colored by average ion occupancy.
        
        Parameters
        ----------
        ion_types : list of str, optional
            Ion types to plot. If None, plots all analyzed ions.
        z_slice_centers : list of float, optional
            Z-slices to plot. If None, plots all analyzed slices.
        show_ion_density : bool, default=True
            Show background ion density heatmap
        ion_density_cmap : str, default='Purples'
            Colormap for ion density
        ion_density_alpha : float, default=0.7
            Transparency for ion density heatmap
        cavity_marker : str, default='o'
            Marker style for cavities
        cavity_colormap : str, default='hot'
            Colormap for cavity occupancy
        cavity_size_range : tuple, default=(50, 500)
            (min, max) marker size for cavities
        cavity_alpha : float, default=0.8
            Transparency for cavity markers
        cavity_edgecolor : str, default='black'
            Edge color for cavity markers
        cavity_linewidth : float, default=1.5
            Edge line width for cavity markers
        show_colorbar : bool, default=True
            Show colorbar for cavity occupancy
        show_empty_cavities : bool, default=True
            Show cavities with zero occupancy
        empty_cavity_color : str, default='gray'
            Color for empty cavities
        empty_cavity_alpha : float, default=0.3
            Transparency for empty cavities
        empty_cavity_size : float, default=50
            Marker size for empty cavities
        figsize : tuple, default=(15, 5)
            Figure size (width, height) in inches
        dpi : int, default=300
            Resolution for saved figures
        save_plots : bool, default=False
            Save plots to file
        filename : str, optional
            Custom filename for saved plot
        save_format : str, default='png'
            File format ('png', 'pdf', 'svg')
        title_fontsize : int, default=14
            Font size for titles
        label_fontsize : int, default=12
            Font size for axis labels
        tick_fontsize : int, default=10
            Font size for tick labels
        colorbar_label_fontsize : int, default=11
            Font size for colorbar label
        show_title : bool, default=True
            Show subplot titles
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        self._validate_analysis()
        
        if 'cavity_ion_binding' not in self.analysis.results:
            raise ValueError("No cavity ion binding data found. Run analyze_cavity_ion_binding() first.")
        
        cib_data = self.analysis.results['cavity_ion_binding']
        
        # Get ion types
        if ion_types is None:
            ion_types = cib_data['metadata']['ion_types']
        elif isinstance(ion_types, str):
            ion_types = [ion_types]
        
        # Get z-slices
        if z_slice_centers is None:
            z_slice_centers = cib_data['z_slice_centers']
        
        # Get box dimensions from universe
        box_x = self.analysis.u.dimensions[0]
        box_y = self.analysis.u.dimensions[1]
        
        # Create figure
        n_ions = len(ion_types)
        n_slices = len(z_slice_centers)
        
        fig, axes = plt.subplots(n_ions, n_slices, figsize=figsize, 
                                squeeze=False, dpi=dpi)
        
        for ion_idx, ion_type in enumerate(ion_types):
            for slice_idx, z_center in enumerate(z_slice_centers):
                ax = axes[ion_idx, slice_idx]
                
                # Get cavity data for this slice
                if z_center not in cib_data['cavity_data']:
                    ax.text(0.5, 0.5, f'No data\nfor z={z_center:.1f}',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlim(0, box_x)
                    ax.set_ylim(0, box_y)
                    continue
                
                cavity_data = cib_data['cavity_data'][z_center]
                ring_centers = cavity_data['ring_centers']
                
                # Get ion occupancy data
                if z_center not in cib_data['ion_data'][ion_type]:
                    ax.text(0.5, 0.5, f'No {ion_type} data\nfor z={z_center:.1f}',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlim(0, box_x)
                    ax.set_ylim(0, box_y)
                    continue
                
                ion_data = cib_data['ion_data'][ion_type][z_center]
                avg_occupancy = ion_data['avg_occupancy']
                
                # Plot ion density background (optional)
                if show_ion_density:
                    # Calculate ion density from trajectory
                    # This is a simplified version - you may want to integrate with ZDirectionalAnalysis
                    ion_atoms = self.analysis.ions[ion_type]
                    
                    # Collect ion positions in z-slice
                    z_width = cib_data['metadata']['z_slice_width']
                    ion_positions_slice = []
                    
                    for ts in self.analysis.u.trajectory[::10]:  # Sample every 10 frames
                        ion_pos = ion_atoms.positions
                        z_mask = np.abs(ion_pos[:, 2] - z_center) <= z_width
                        ion_positions_slice.extend(ion_pos[z_mask, :2])
                    
                    if len(ion_positions_slice) > 0:
                        ion_positions_slice = np.array(ion_positions_slice)
                        
                        # Create 2D histogram
                        H, xedges, yedges = np.histogram2d(
                            ion_positions_slice[:, 0], 
                            ion_positions_slice[:, 1],
                            bins=50, 
                            range=[[0, box_x], [0, box_y]]
                        )
                        
                        # Plot heatmap
                        extent = [0, box_x, 0, box_y]
                        im = ax.imshow(H.T, origin='lower', extent=extent,
                                      cmap=ion_density_cmap, alpha=ion_density_alpha,
                                      aspect='auto')
                
                # Prepare cavity markers
                if len(ring_centers) > 0:
                    # Normalize occupancy for coloring
                    max_occ = np.max(avg_occupancy) if np.max(avg_occupancy) > 0 else 1.0
                    norm_occupancy = avg_occupancy / max_occ
                    
                    # Size markers by occupancy
                    min_size, max_size = cavity_size_range
                    sizes = min_size + (max_size - min_size) * norm_occupancy
                    
                    # Separate occupied and empty cavities
                    occupied_mask = avg_occupancy > 0
                    
                    # Plot occupied cavities
                    if np.any(occupied_mask):
                        scatter = ax.scatter(
                            ring_centers[occupied_mask, 0],
                            ring_centers[occupied_mask, 1],
                            s=sizes[occupied_mask],
                            c=avg_occupancy[occupied_mask],
                            cmap=cavity_colormap,
                            alpha=cavity_alpha,
                            marker=cavity_marker,
                            edgecolors=cavity_edgecolor,
                            linewidths=cavity_linewidth,
                            vmin=0, vmax=max_occ,
                            zorder=10
                        )
                        
                        # Add colorbar
                        if show_colorbar and ion_idx == 0 and slice_idx == n_slices - 1:
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.1)
                            cbar = plt.colorbar(scatter, cax=cax)
                            cbar.set_label('Avg Ion Occupancy', fontsize=colorbar_label_fontsize)
                    
                    # Plot empty cavities
                    if show_empty_cavities and np.any(~occupied_mask):
                        ax.scatter(
                            ring_centers[~occupied_mask, 0],
                            ring_centers[~occupied_mask, 1],
                            s=empty_cavity_size,
                            c=empty_cavity_color,
                            alpha=empty_cavity_alpha,
                            marker=cavity_marker,
                            edgecolors=cavity_edgecolor,
                            linewidths=cavity_linewidth * 0.5,
                            zorder=9
                        )
                
                # Set axis properties
                ax.set_xlim(0, box_x)
                ax.set_ylim(0, box_y)
                ax.set_aspect('equal')
                
                # Labels
                if ion_idx == n_ions - 1:
                    ax.set_xlabel('X (Å)', fontsize=label_fontsize)
                if slice_idx == 0:
                    ax.set_ylabel('Y (Å)', fontsize=label_fontsize)
                
                # Title
                if show_title:
                    if ion_idx == 0:
                        ax.set_title(f'z = {z_center:.1f} Å', fontsize=title_fontsize)
                    if slice_idx == 0:
                        ax.text(-0.15, 0.5, ion_type, transform=ax.transAxes,
                               fontsize=title_fontsize, fontweight='bold',
                               rotation=90, va='center', ha='center')
                
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        plt.tight_layout()
        
        if save_plots:
            if filename is None:
                filename = f'cavity_ion_binding.{save_format}'
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', format=save_format)
            print(f"Cavity ion binding plot saved to {filename}")
        
        plt.show()
        
        return fig
    
    def plot_cavity_occupancy_timeseries(self, ion_types=None, z_slice_centers=None,
                                        max_cavities_per_plot=10, show_average=True,
                                        avg_linewidth=3, avg_color='red', avg_alpha=0.8,
                                        individual_alpha=0.3, colormap='tab10',
                                        figsize=(14, 8), dpi=300, save_plots=False,
                                        filename=None, save_format='png',
                                        title_fontsize=14, label_fontsize=12,
                                        tick_fontsize=10, legend_fontsize=9,
                                        show_title=True, show_legend=True):
        """
        Plot time series of ion occupancy for each cavity.
        
        Parameters
        ----------
        ion_types : list of str, optional
            Ion types to plot. If None, plots all analyzed ions.
        z_slice_centers : list of float, optional
            Z-slices to plot. If None, plots all analyzed slices.
        max_cavities_per_plot : int, default=10
            Maximum number of individual cavities to show (highest occupancy)
        show_average : bool, default=True
            Show average occupancy across all cavities
        avg_linewidth : float, default=3
            Line width for average occupancy
        avg_color : str, default='red'
            Color for average occupancy line
        avg_alpha : float, default=0.8
            Transparency for average line
        individual_alpha : float, default=0.3
            Transparency for individual cavity lines
        colormap : str, default='tab10'
            Colormap for individual cavity lines
        figsize : tuple, default=(14, 8)
            Figure size (width, height) in inches
        dpi : int, default=300
            Resolution for saved figures
        save_plots : bool, default=False
            Save plots to file
        filename : str, optional
            Custom filename for saved plot
        save_format : str, default='png'
            File format ('png', 'pdf', 'svg')
        title_fontsize : int, default=14
            Font size for titles
        label_fontsize : int, default=12
            Font size for axis labels
        tick_fontsize : int, default=10
            Font size for tick labels
        legend_fontsize : int, default=9
            Font size for legend
        show_title : bool, default=True
            Show subplot titles
        show_legend : bool, default=True
            Show legend
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        self._validate_analysis()
        
        if 'cavity_ion_binding' not in self.analysis.results:
            raise ValueError("No cavity ion binding data found. Run analyze_cavity_ion_binding() first.")
        
        cib_data = self.analysis.results['cavity_ion_binding']
        
        # Get ion types
        if ion_types is None:
            ion_types = cib_data['metadata']['ion_types']
        elif isinstance(ion_types, str):
            ion_types = [ion_types]
        
        # Get z-slices
        if z_slice_centers is None:
            z_slice_centers = cib_data['z_slice_centers']
        
        # Create figure
        n_ions = len(ion_types)
        n_slices = len(z_slice_centers)
        
        fig, axes = plt.subplots(n_ions, n_slices, figsize=figsize,
                                squeeze=False, dpi=dpi)
        
        # Get time axis
        step = cib_data['metadata']['step']
        n_frames = cib_data['metadata']['n_frames']
        dt = self.analysis.u.trajectory.dt  # Time step in ps
        time_array = np.arange(n_frames) * dt * step
        
        # Get colormap
        cmap = plt.get_cmap(colormap)
        
        for ion_idx, ion_type in enumerate(ion_types):
            for slice_idx, z_center in enumerate(z_slice_centers):
                ax = axes[ion_idx, slice_idx]
                
                # Check if data exists
                if z_center not in cib_data['ion_data'][ion_type]:
                    ax.text(0.5, 0.5, f'No {ion_type} data\nfor z={z_center:.1f}',
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                ion_data = cib_data['ion_data'][ion_type][z_center]
                per_cavity_ts = ion_data['per_cavity_timeseries']
                avg_occupancy = ion_data['avg_occupancy']
                
                n_cavities = per_cavity_ts.shape[0]
                
                if n_cavities == 0:
                    ax.text(0.5, 0.5, 'No cavities',
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Select top cavities by average occupancy
                top_indices = np.argsort(avg_occupancy)[::-1][:max_cavities_per_plot]
                
                # Plot individual cavities
                for i, cavity_idx in enumerate(top_indices):
                    color = cmap(i / max(max_cavities_per_plot, 1))
                    ax.plot(time_array, per_cavity_ts[cavity_idx, :],
                           color=color, alpha=individual_alpha, linewidth=1,
                           label=f'Cavity {cavity_idx+1}')
                
                # Plot average across all cavities
                if show_average:
                    avg_ts = np.mean(per_cavity_ts, axis=0)
                    ax.plot(time_array, avg_ts, color=avg_color, 
                           linewidth=avg_linewidth, alpha=avg_alpha,
                           label='Average (all cavities)', zorder=100)
                
                # Set axis properties
                ax.set_xlim(0, time_array[-1])
                ax.set_ylim(bottom=0)
                
                # Labels
                if ion_idx == n_ions - 1:
                    ax.set_xlabel('Time (ps)', fontsize=label_fontsize)
                if slice_idx == 0:
                    ax.set_ylabel('Ion Count', fontsize=label_fontsize)
                
                # Title
                if show_title:
                    title = f'{ion_type}, z={z_center:.1f} Å\n({n_cavities} cavities)'
                    ax.set_title(title, fontsize=title_fontsize)
                
                # Legend (only for first subplot to avoid clutter)
                if show_legend and ion_idx == 0 and slice_idx == 0:
                    ax.legend(fontsize=legend_fontsize, loc='best', frameon=True,
                             framealpha=0.7, ncol=2)
                
                # Grid
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        plt.tight_layout()
        
        if save_plots:
            if filename is None:
                filename = f'cavity_occupancy_timeseries.{save_format}'
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', format=save_format)
            print(f"Cavity occupancy timeseries plot saved to {filename}")
        
        plt.show()
        
        return fig
    
    def plot_preferential_binding_sites(self, ion_types=None, z_slice_centers=None,
                                       top_n=5, marker='*', marker_size=300,
                                       marker_color='gold', marker_edgecolor='red',
                                       marker_linewidth=2, show_labels=True,
                                       label_fontsize=9, label_color='black',
                                       label_bbox=True, figsize=(15, 5), dpi=300,
                                       save_plots=False, filename=None,
                                       save_format='png', title_fontsize=14,
                                       label_fontsize_axis=12, tick_fontsize=10,
                                       show_title=True):
        """
        Plot spatial map of preferential ion binding sites (top-occupied cavities).
        
        Parameters
        ----------
        ion_types : list of str, optional
            Ion types to plot
        z_slice_centers : list of float, optional
            Z-slices to plot
        top_n : int, default=5
            Number of top cavities to highlight
        marker : str, default='*'
            Marker style for preferential sites
        marker_size : float, default=300
            Size of preferential site markers
        marker_color : str, default='gold'
            Color for preferential site markers
        marker_edgecolor : str, default='red'
            Edge color for markers
        marker_linewidth : float, default=2
            Edge line width
        show_labels : bool, default=True
            Show occupancy labels on markers
        label_fontsize : int, default=9
            Font size for labels
        label_color : str, default='black'
            Color for labels
        label_bbox : bool, default=True
            Draw box around labels
        figsize : tuple, default=(15, 5)
            Figure size
        dpi : int, default=300
            Resolution
        save_plots : bool, default=False
            Save plots to file
        filename : str, optional
            Custom filename
        save_format : str, default='png'
            File format
        title_fontsize : int, default=14
            Title font size
        label_fontsize_axis : int, default=12
            Axis label font size
        tick_fontsize : int, default=10
            Tick label font size
        show_title : bool, default=True
            Show titles
        
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        self._validate_analysis()
        
        if 'cavity_ion_binding' not in self.analysis.results:
            raise ValueError("No cavity ion binding data found. Run analyze_cavity_ion_binding() first.")
        
        cib_data = self.analysis.results['cavity_ion_binding']
        pref_sites = cib_data['preferential_sites']
        
        # Get ion types
        if ion_types is None:
            ion_types = cib_data['metadata']['ion_types']
        elif isinstance(ion_types, str):
            ion_types = [ion_types]
        
        # Get z-slices
        if z_slice_centers is None:
            z_slice_centers = cib_data['z_slice_centers']
        
        # Get box dimensions
        box_x = self.analysis.u.dimensions[0]
        box_y = self.analysis.u.dimensions[1]
        
        # Create figure
        n_ions = len(ion_types)
        n_slices = len(z_slice_centers)
        
        fig, axes = plt.subplots(n_ions, n_slices, figsize=figsize,
                                squeeze=False, dpi=dpi)
        
        for ion_idx, ion_type in enumerate(ion_types):
            for slice_idx, z_center in enumerate(z_slice_centers):
                ax = axes[ion_idx, slice_idx]
                
                # Check if data exists
                if z_center not in pref_sites[ion_type]:
                    ax.text(0.5, 0.5, f'No data for z={z_center:.1f}',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlim(0, box_x)
                    ax.set_ylim(0, box_y)
                    continue
                
                site_data = pref_sites[ion_type][z_center]
                positions = site_data['cavity_positions']
                occupancies = site_data['avg_occupancy']
                
                # Plot all cavity positions as background
                all_centers = cib_data['cavity_data'][z_center]['ring_centers']
                ax.scatter(all_centers[:, 0], all_centers[:, 1],
                          s=50, c='lightgray', alpha=0.3, marker='o')
                
                # Plot preferential sites
                if len(positions) > 0:
                    ax.scatter(positions[:, 0], positions[:, 1],
                              s=marker_size, c=marker_color, marker=marker,
                              edgecolors=marker_edgecolor, linewidths=marker_linewidth,
                              alpha=0.9, zorder=10)
                    
                    # Add labels
                    if show_labels:
                        for i, (pos, occ) in enumerate(zip(positions, occupancies)):
                            label = f'{occ:.2f}'
                            bbox_props = dict(boxstyle='round,pad=0.3', 
                                            facecolor='white', alpha=0.7) if label_bbox else None
                            ax.text(pos[0], pos[1] + 1.5, label,
                                   fontsize=label_fontsize, color=label_color,
                                   ha='center', va='bottom', bbox=bbox_props,
                                   zorder=11)
                
                # Set axis properties
                ax.set_xlim(0, box_x)
                ax.set_ylim(0, box_y)
                ax.set_aspect('equal')
                
                # Labels
                if ion_idx == n_ions - 1:
                    ax.set_xlabel('X (Å)', fontsize=label_fontsize_axis)
                if slice_idx == 0:
                    ax.set_ylabel('Y (Å)', fontsize=label_fontsize_axis)
                
                # Title
                if show_title:
                    title = f'{ion_type}, z={z_center:.1f} Å\nTop {len(positions)} Sites'
                    ax.set_title(title, fontsize=title_fontsize)
                
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        plt.tight_layout()
        
        if save_plots:
            if filename is None:
                filename = f'preferential_binding_sites.{save_format}'
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', format=save_format)
            print(f"Preferential binding sites plot saved to {filename}")
        
        plt.show()
        
        return fig
    
    # =========================================================================
    # CLAY STRUCTURE VISUALIZATION
    # =========================================================================
    
    def plot_clay_structure_2d(self, view='top', surface='top', z_range=10.0,
                                show_atoms=None, show_polyhedra=False,
                                polyhedra_alpha=0.3, polyhedra_linewidth=0.8,
                                atom_colors=None, atom_sizes=None,
                                figsize=(14, 12), save_plot=False, 
                                filename='clay_structure_2d.png', dpi=300):
        """
        Plot 2D clay structure with unwrapped coordinates.
        
        Shows clay atoms from different views, colored by type. Uses unwrapping
        logic to create complete layers by adding periodic images.
        
        Parameters
        ----------
        view : str, default='top'
            View type: 'top' (xy-plane, looking down z), 
                      'side' (xz-plane, see both surfaces),
                      'both' (subplot with both views)
        surface : str, default='top'
            Which surface to render: 'top', 'bottom', or 'both'.
            For side view, this is ignored and both surfaces are shown.
        z_range : float, default=10.0
            Z-range (Å) to include around the surface center.
            For side view, full z-range of unwrapped atoms is shown.
        show_atoms : list, optional
            Atom types to show. If None (default), shows all clay atoms (resname MMT).
        show_polyhedra : bool, default=False
            Draw coordination polyhedra (tetrahedra for Si-O, octahedra for Mg/Al-O)
        polyhedra_alpha : float, default=0.3
            Transparency of polyhedra edges
        polyhedra_linewidth : float, default=0.8
            Line width for polyhedra edges
        atom_colors : dict, optional
            Colors for each atom type {atom_name: color}
        atom_sizes : dict, optional
            Sizes for each atom type {atom_name: size}
        figsize : tuple, default=(14, 12)
            Figure size
        save_plot : bool, default=False
            Save plot to file
        filename : str, default='clay_structure_2d.png'
            Filename for saved plot
        dpi : int, default=300
            Resolution for saved plot
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        
        Examples
        --------
        >>> fig, ax = plotter.plot_clay_structure_2d(view='top', surface='top')
        >>> fig, ax = plotter.plot_clay_structure_2d(view='side')  # See both surfaces
        >>> fig, axes = plotter.plot_clay_structure_2d(view='both')  # Top + side views
        >>> fig, ax = plotter.plot_clay_structure_2d(view='side', show_polyhedra=True)  # With crystal structure
        """
        self._validate_analysis()
        
        # Get universe from analysis (fallback for different attribute names)
        if hasattr(self.analysis, 'universe'):
            u = self.analysis.universe
        elif hasattr(self.analysis, 'u'):
            u = self.analysis.u
        else:
            raise AttributeError("Analysis object has no 'universe' or 'u' attribute")
        u.trajectory[0]
        
        # Get box dimensions
        box_x, box_y, box_z = u.dimensions[0], u.dimensions[1], u.dimensions[2]
        box_center = box_z / 2
        
        print("="*80)
        print("2D CLAY STRUCTURE VISUALIZATION")
        print("="*80)
        print(f"\n📦 Box: {box_x:.2f} × {box_y:.2f} × {box_z:.2f} Å³")
        
        # Get all clay atoms
        all_clay = u.select_atoms("resname MMT")
        positions = all_clay.positions.copy()
        names = all_clay.names
        
        # Center z-coordinates
        positions[:, 2] -= box_center
        
        # Unwrap clay structure (add periodic images)
        threshold = 10.0
        top_mask = positions[:, 2] > (box_z/2 - threshold)
        bottom_mask = positions[:, 2] < (-box_z/2 + threshold)
        
        # Create copies
        pos_top_shifted = positions[top_mask].copy()
        pos_top_shifted[:, 2] -= box_z
        names_top = names[top_mask]
        
        pos_bottom_shifted = positions[bottom_mask].copy()
        pos_bottom_shifted[:, 2] += box_z
        names_bottom = names[bottom_mask]
        
        # Keep only non-overlapping periodic images
        top_keep = pos_top_shifted[:, 2] < -box_z/2
        bottom_keep = pos_bottom_shifted[:, 2] > box_z/2
        
        # Combine
        positions_complete = np.vstack([
            positions,
            pos_top_shifted[top_keep],
            pos_bottom_shifted[bottom_keep]
        ])
        names_complete = np.concatenate([
            names,
            names_top[top_keep],
            names_bottom[bottom_keep]
        ])
        
        print(f"🧱 Total atoms (unwrapped): {len(positions_complete)}")
        print(f"   Z-range: {np.min(positions_complete[:, 2]):+.2f} to {np.max(positions_complete[:, 2]):+.2f} Å")
        
        # If show_atoms not specified, show all clay atom types
        if show_atoms is None:
            show_atoms = np.unique(names_complete).tolist()
            print(f"📋 Showing all clay atom types: {show_atoms}")
        
        # Default colors and sizes
        if atom_colors is None:
            atom_colors = {
                'Si': 'gold', 'ST': 'gold',
                'Mgo': 'green', 'MG': 'green',
                'Alo': 'purple', 'AL': 'purple',
                'Ob': 'red', 'OB': 'red',
                'Oa': 'blue', 'OA': 'blue',
                'Oh': 'cyan', 'OH': 'cyan'
            }
        
        if atom_sizes is None:
            atom_sizes = {
                'Si': 150, 'ST': 150,
                'Mgo': 200, 'MG': 200,
                'Alo': 200, 'AL': 200,
                'Ob': 80, 'OB': 80,
                'Oa': 80, 'OA': 80,
                'Oh': 80, 'OH': 80
            }
        
        # Handle different views
        if view == 'both':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
            axes = [ax1, ax2]
            views_to_plot = ['top', 'side']
        else:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
            views_to_plot = [view]
        
        for ax, current_view in zip(axes, views_to_plot):
            if current_view == 'top':
                # TOP VIEW: xy-plane (looking down z-axis)
                print(f"\n🎯 {surface.upper()} surface view (xy-plane)")
                
                # Determine surface center
                if surface == 'top':
                    surf_center = box_z/2 - 3
                elif surface == 'bottom':
                    surf_center = -box_z/2 + 3
                elif surface == 'both':
                    surf_center = 0
                    z_range = box_z
                else:
                    raise ValueError("surface must be 'top', 'bottom', or 'both'")
                
                # Select atoms in z-range
                z_min = surf_center - z_range/2
                z_max = surf_center + z_range/2
                in_range = (positions_complete[:, 2] >= z_min) & (positions_complete[:, 2] <= z_max)
                
                pos_region = positions_complete[in_range]
                names_region = names_complete[in_range]
                
                print(f"   Z-range: {z_min:+.2f} to {z_max:+.2f} Å")
                print(f"   Atoms: {len(pos_region)}")
                
                # Plot each atom type
                for atom_type in show_atoms:
                    mask = np.isin(names_region, [atom_type, atom_type.upper()])
                    if np.any(mask):
                        pos_type = pos_region[mask]
                        color = atom_colors.get(atom_type, 'gray')
                        size = atom_sizes.get(atom_type, 100)
                        
                        ax.scatter(pos_type[:, 0], pos_type[:, 1], 
                                  c=color, s=size, alpha=0.7, 
                                  edgecolors='black', linewidth=0.5,
                                  label=f'{atom_type} ({np.sum(mask)})')
                        
                        if view != 'both':
                            print(f"      {atom_type}: {np.sum(mask)}")
                
                ax.set_xlabel('X (Å)', fontsize=12)
                ax.set_ylabel('Y (Å)', fontsize=12)
                ax.set_title(f'Top View (xy-plane)\nz: {z_min:+.1f} to {z_max:+.1f} Å',
                             fontsize=13, fontweight='bold')
                ax.set_aspect('equal')
                
            else:  # side view
                # SIDE VIEW: xz-plane (looking along y-axis) - see both surfaces
                print(f"\n🎯 SIDE view (xz-plane) - both surfaces")
                
                pos_region = positions_complete
                names_region = names_complete
                
                print(f"   Atoms: {len(pos_region)}")
                
                # Plot each atom type
                for atom_type in show_atoms:
                    mask = np.isin(names_region, [atom_type, atom_type.upper()])
                    if np.any(mask):
                        pos_type = pos_region[mask]
                        color = atom_colors.get(atom_type, 'gray')
                        size = atom_sizes.get(atom_type, 100)
                        
                        ax.scatter(pos_type[:, 0], pos_type[:, 2], 
                                  c=color, s=size, alpha=0.7, 
                                  edgecolors='black', linewidth=0.5,
                                  label=f'{atom_type} ({np.sum(mask)})')
                        
                        if view != 'both':
                            print(f"      {atom_type}: {np.sum(mask)}")
                
                # Mark original box boundaries
                ax.axhline(-box_z/2, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Original box')
                ax.axhline(+box_z/2, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
                ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3, label='Box center')
                
                ax.set_xlabel('X (Å)', fontsize=12)
                ax.set_ylabel('Z (Å)', fontsize=12)
                ax.set_title('Side View (xz-plane)\nBoth Surfaces',
                             fontsize=13, fontweight='bold')
            
            # Draw polyhedra if requested
            if show_polyhedra:
                self._draw_polyhedra_2d(ax, pos_region, names_region, current_view,
                                       polyhedra_alpha, polyhedra_linewidth)
            
            ax.legend(fontsize=9, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"\n📸 Plot saved: {filename}")
        
        print("\n" + "="*80)
        
        plt.show()
        
        return fig, axes[0] if view != 'both' else (fig, axes)
    
    def _draw_polyhedra_2d(self, ax, positions, names, view, alpha, linewidth):
        """
        Draw coordination polyhedra edges on 2D plot.
        
        Parameters
        ----------
        ax : matplotlib axis
            Axis to draw on
        positions : ndarray
            Atomic positions
        names : ndarray
            Atom names
        view : str
            'top' or 'side'
        alpha : float
            Line transparency
        linewidth : float
            Line width
        """
        from scipy.spatial import distance_matrix
        
        print(f"\n🔷 Drawing coordination polyhedra...")
        
        # Define coordination criteria
        tetra_cutoff = 2.0  # Si-O distance (Å)
        octa_cutoff = 2.3   # Mg/Al-O distance (Å)
        
        # Find tetrahedral centers (Si)
        si_mask = np.isin(names, ['Si', 'ST', 'si', 'st'])
        si_positions = positions[si_mask]
        
        # Find octahedral centers (Mg, Al)
        octa_mask = np.isin(names, ['Mgo', 'Alo', 'MG', 'AL', 'mgo', 'alo', 'Mg', 'Al'])
        octa_positions = positions[octa_mask]
        
        # Find oxygen atoms
        o_mask = np.isin(names, ['Ob', 'Oa', 'Oh', 'OB', 'OA', 'OH', 'ob', 'oa', 'oh', 'O', 'o'])
        o_positions = positions[o_mask]
        
        tetra_count = 0
        octa_count = 0
        
        # Draw tetrahedra (Si-O)
        if len(si_positions) > 0 and len(o_positions) > 0:
            for si_pos in si_positions:
                # Calculate distances to all oxygens
                distances = np.linalg.norm(o_positions - si_pos, axis=1)
                coordinated = distances < tetra_cutoff
                
                if np.any(coordinated):
                    o_coord = o_positions[coordinated]
                    
                    # Draw lines from Si to coordinated O
                    for o_pos in o_coord:
                        if view == 'top':
                            ax.plot([si_pos[0], o_pos[0]], [si_pos[1], o_pos[1]],
                                   'b-', alpha=alpha, linewidth=linewidth, zorder=1)
                        else:  # side view
                            ax.plot([si_pos[0], o_pos[0]], [si_pos[2], o_pos[2]],
                                   'b-', alpha=alpha, linewidth=linewidth, zorder=1)
                    
                    tetra_count += 1
        
        # Draw octahedra (Mg/Al-O)
        if len(octa_positions) > 0 and len(o_positions) > 0:
            for octa_pos in octa_positions:
                # Calculate distances to all oxygens
                distances = np.linalg.norm(o_positions - octa_pos, axis=1)
                coordinated = distances < octa_cutoff
                
                if np.any(coordinated):
                    o_coord = o_positions[coordinated]
                    
                    # Draw lines from Mg/Al to coordinated O
                    for o_pos in o_coord:
                        if view == 'top':
                            ax.plot([octa_pos[0], o_pos[0]], [octa_pos[1], o_pos[1]],
                                   'g-', alpha=alpha, linewidth=linewidth, zorder=1)
                        else:  # side view
                            ax.plot([octa_pos[0], o_pos[0]], [octa_pos[2], o_pos[2]],
                                   'g-', alpha=alpha, linewidth=linewidth, zorder=1)
                    
                    octa_count += 1
        
        print(f"   Tetrahedra (Si-O): {tetra_count}")
        print(f"   Octahedra (Mg/Al-O): {octa_count}")
    
    # =========================================================================
    # 3D CLAY STRUCTURE VISUALIZATION WITH PYVISTA
    # =========================================================================
    
    def render_clay_polyhedra(self, surface='top', z_range=15.0, 
                               show_tetrahedra=True, show_octahedra=True,
                               tetrahedra_color='blue', tetrahedra_opacity=0.7,
                               octahedra_color='green', octahedra_opacity=0.7,
                               show_atoms=False, atom_size=0.3,
                               coord_cutoff=2.5, jupyter_backend='static',
                               save_screenshot=False, screenshot_filename='clay_structure.png'):
        """
        Render 3D clay polyhedra structure using PyVista with unwrapped coordinates.
        
        Uses the unwrapping logic to create complete clay layers by adding periodic images,
        then renders tetrahedral (Si) and octahedral (Mg/Al) coordination polyhedra.
        
        Parameters
        ----------
        surface : str, default='top'
            Which surface to render: 'top', 'bottom', or 'both'
        z_range : float, default=15.0
            Z-range (Å) to include around the surface center
        show_tetrahedra : bool, default=True
            Render tetrahedral polyhedra (Si-O)
        show_octahedra : bool, default=True
            Render octahedral polyhedra (Mg/Al-O)
        tetrahedra_color : str, default='blue'
            Color for tetrahedral polyhedra
        tetrahedra_opacity : float, default=0.7
            Opacity for tetrahedral polyhedra (0-1)
        octahedra_color : str, default='green'
            Color for octahedral polyhedra
        octahedra_opacity : float, default=0.7
            Opacity for octahedral polyhedra (0-1)
        show_atoms : bool, default=False
            Show atoms as spheres
        atom_size : float, default=0.3
            Size of atom spheres (if show_atoms=True)
        coord_cutoff : float, default=2.5
            Distance cutoff (Å) for identifying coordinated atoms
        jupyter_backend : str, default='static'
            PyVista backend: 'static' (non-interactive) or 'ipyvtklink' (interactive)
        save_screenshot : bool, default=False
            Save screenshot to file
        screenshot_filename : str, default='clay_structure.png'
            Filename for screenshot
        
        Returns
        -------
        plotter : pyvista.Plotter
            PyVista plotter object (can be used for further customization)
        
        Examples
        --------
        >>> plotter_obj = plotter.render_clay_polyhedra(surface='top', z_range=10)
        >>> plotter_obj = plotter.render_clay_polyhedra(surface='both', show_atoms=True)
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("PyVista not installed. Install with: pip install pyvista")
        
        self._validate_analysis()
        
        print("="*80)
        print("RENDERING CLAY POLYHEDRA STRUCTURE")
        print("="*80)
        
        # Get universe from analysis (fallback for different attribute names)
        if hasattr(self.analysis, 'universe'):
            u = self.analysis.universe
        elif hasattr(self.analysis, 'u'):
            u = self.analysis.u
        else:
            raise AttributeError("Analysis object has no 'universe' or 'u' attribute")
        u.trajectory[0]
        
        # Get box dimensions
        box_x, box_y, box_z = u.dimensions[0], u.dimensions[1], u.dimensions[2]
        box_center = box_z / 2
        
        print(f"\n📦 Box: {box_x:.2f} × {box_y:.2f} × {box_z:.2f} Å³")
        
        # Get all clay atoms
        all_clay = u.select_atoms("resname MMT")
        positions_abs = all_clay.positions.copy()
        names = all_clay.names
        
        # Center coordinates
        positions_abs[:, 2] -= box_center
        
        # Unwrap clay structure (add periodic images)
        threshold = 10.0
        top_mask = positions_abs[:, 2] > (box_z/2 - threshold)
        bottom_mask = positions_abs[:, 2] < (-box_z/2 + threshold)
        
        # Create copies
        pos_top_shifted = positions_abs[top_mask].copy()
        pos_top_shifted[:, 2] -= box_z
        names_top = names[top_mask]
        
        pos_bottom_shifted = positions_abs[bottom_mask].copy()
        pos_bottom_shifted[:, 2] += box_z
        names_bottom = names[bottom_mask]
        
        # Keep only non-overlapping periodic images
        top_keep = pos_top_shifted[:, 2] < -box_z/2
        bottom_keep = pos_bottom_shifted[:, 2] > box_z/2
        
        # Combine
        positions_complete = np.vstack([
            positions_abs,
            pos_top_shifted[top_keep],
            pos_bottom_shifted[bottom_keep]
        ])
        names_complete = np.concatenate([
            names,
            names_top[top_keep],
            names_bottom[bottom_keep]
        ])
        
        print(f"🧱 Total atoms (unwrapped): {len(positions_complete)}")
        print(f"   Z-range: {np.min(positions_complete[:, 2]):+.2f} to {np.max(positions_complete[:, 2]):+.2f} Å")
        
        # Determine surface center(s) to render
        if surface == 'top':
            surface_centers = [box_z/2 - 3]  # Top surface around +26 Å
            print(f"\n🎯 Rendering TOP surface region")
        elif surface == 'bottom':
            surface_centers = [-box_z/2 + 3]  # Bottom surface around -26 Å
            print(f"\n🎯 Rendering BOTTOM surface region")
        elif surface == 'both':
            surface_centers = [box_z/2 - 3, -box_z/2 + 3]
            print(f"\n🎯 Rendering BOTH surfaces")
        else:
            raise ValueError("surface must be 'top', 'bottom', or 'both'")
        
        # Create PyVista plotter
        pv.set_jupyter_backend(jupyter_backend)
        pl = pv.Plotter()
        pl.set_background('white')
        
        # Process each surface
        for surf_center in surface_centers:
            # Select atoms in z-range around surface
            z_min = surf_center - z_range/2
            z_max = surf_center + z_range/2
            in_range = (positions_complete[:, 2] >= z_min) & (positions_complete[:, 2] <= z_max)
            
            pos_region = positions_complete[in_range]
            names_region = names_complete[in_range]
            
            print(f"\n   Surface at z ≈ {surf_center:+.1f} Å:")
            print(f"   Range: {z_min:+.2f} to {z_max:+.2f} Å")
            print(f"   Atoms in range: {len(pos_region)}")
            
            # Separate by atom type
            si_mask = np.isin(names_region, ['Si', 'ST'])
            mg_mask = np.isin(names_region, ['Mgo', 'MG'])
            al_mask = np.isin(names_region, ['Alo', 'AL'])
            o_mask = np.isin(names_region, ['Ob', 'Oa', 'Oh', 'O', 'OB', 'OA', 'OH'])
            
            si_pos = pos_region[si_mask]
            mg_pos = pos_region[mg_mask]
            al_pos = pos_region[al_mask]
            o_pos = pos_region[o_mask]
            oct_pos = pos_region[mg_mask | al_mask]
            
            print(f"      Si: {len(si_pos)}, Mg/Al: {len(oct_pos)}, O: {len(o_pos)}")
            
            # Render tetrahedra (Si-O4)
            if show_tetrahedra and len(si_pos) > 0 and len(o_pos) > 0:
                n_tetra = 0
                for si in si_pos:
                    # Find 4 nearest O atoms
                    distances = np.linalg.norm(o_pos - si, axis=1)
                    nearest_indices = np.argsort(distances)[:4]
                    nearest_o = o_pos[nearest_indices]
                    nearest_dist = distances[nearest_indices]
                    
                    # Only create tetrahedron if all O within cutoff
                    if np.all(nearest_dist < coord_cutoff):
                        # Create convex hull (tetrahedron)
                        points = np.vstack([si, nearest_o])
                        cloud = pv.PolyData(points)
                        hull = cloud.delaunay_3d()
                        pl.add_mesh(hull, color=tetrahedra_color, opacity=tetrahedra_opacity,
                                    show_edges=True, edge_color='darkblue', line_width=1)
                        n_tetra += 1
                
                print(f"      Tetrahedra rendered: {n_tetra}")
            
            # Render octahedra (Mg/Al-O6)
            if show_octahedra and len(oct_pos) > 0 and len(o_pos) > 0:
                n_octa = 0
                for oct in oct_pos:
                    # Find 6 nearest O atoms
                    distances = np.linalg.norm(o_pos - oct, axis=1)
                    nearest_indices = np.argsort(distances)[:6]
                    nearest_o = o_pos[nearest_indices]
                    nearest_dist = distances[nearest_indices]
                    
                    # Only create octahedron if all O within cutoff
                    if np.all(nearest_dist < coord_cutoff):
                        # Create convex hull (octahedron)
                        points = np.vstack([oct, nearest_o])
                        cloud = pv.PolyData(points)
                        hull = cloud.delaunay_3d()
                        pl.add_mesh(hull, color=octahedra_color, opacity=octahedra_opacity,
                                    show_edges=True, edge_color='darkgreen', line_width=1)
                        n_octa += 1
                
                print(f"      Octahedra rendered: {n_octa}")
            
            # Render atoms as spheres
            if show_atoms:
                if len(si_pos) > 0:
                    si_cloud = pv.PolyData(si_pos)
                    pl.add_mesh(si_cloud, color='yellow', point_size=atom_size*10,
                                render_points_as_spheres=True)
                if len(oct_pos) > 0:
                    oct_cloud = pv.PolyData(oct_pos)
                    pl.add_mesh(oct_cloud, color='gray', point_size=atom_size*10,
                                render_points_as_spheres=True)
                if len(o_pos) > 0:
                    o_cloud = pv.PolyData(o_pos)
                    pl.add_mesh(o_cloud, color='red', point_size=atom_size*8,
                                render_points_as_spheres=True)
        
        # Set camera and labels
        pl.add_axes()
        pl.add_text("Clay Structure (Unwrapped)", position='upper_edge', font_size=12)
        pl.show_grid()
        
        # Show or save
        if save_screenshot:
            pl.screenshot(screenshot_filename)
            print(f"\n📸 Screenshot saved: {screenshot_filename}")
        
        print("\n" + "="*80)
        
        pl.show()
        
        return pl

    
    def plot_cavity_occupancy(self, ion_types=None, functional_groups=None, z_slices='all',
                             grid_type='both',  # 'density', 'cavity_occupancy', or 'both'
                             plot_mode='2d',  # '2d', '3d', or 'surface'
                             cavity_interpolation_method='nearest',  # 'weighted' or 'nearest'
                             save_plots=True, figsize=(16, 7), dpi=300,
                             
                             # Individual figure control
                             save_individual_figures=False,
                             show_individual_figures=False,
                             create_combined_figure=False,
                             save_combined_figure=False,
                             show_combined_figure=True,
                             individual_figsize=(8, 7),
                             
                             # Publication settings
                             title_fontsize=14,
                             show_title=True,
                             label_fontsize=12,
                             tick_fontsize=10,
                             colorbar_label_fontsize=11,
                             
                             # 3D histogram parameters
                             bar_alpha=0.7,
                             colormap='viridis',
                             threshold_percentile=1,
                             
                             # 3D surface parameters
                             surface_alpha=0.8,
                             surface_cmap='RdYlBu_r',
                             surface_lighting=True,
                             surface_smoothing=True,
                             surface_smoothing_sigma=1.0,
                             surface_stride_x=1,
                             surface_stride_y=1,
                             surface_linewidth=0,
                             surface_antialiased=True,
                             surface_rstride=1,
                             surface_cstride=1,
                             elevation=None,
                             azimuth=None,
                             
                             # 2D heatmap parameters
                             ion_density_cmap='hot',
                             cavity_occupancy_cmap='viridis',
                             origin='lower',
                             interpolation='gaussian',
                             aspect='equal',
                             
                             # Colorbar control
                             show_colorbar=True,
                             colorbar_pad=0.02,
                             colorbar_width='4%',
                             ion_density_vmin=None,
                             ion_density_vmax=None,
                             cavity_occupancy_vmin=None,
                             cavity_occupancy_vmax=None,
                             
                             # Separate Z-axis and colorbar scaling control (3D only)
                             z_scale_factor=None,
                             z_axis_limit=None,
                             colorbar_vmin=None,
                             colorbar_vmax=None,
                             
                             # Grid and styling
                             show_grid=True,
                             grid_alpha=0.3,
                             
                             # Clay overlay parameters
                             overlay_clay_contours=False,
                             clay_contour_results=None,
                             mg_contour_color='white',
                             mg_contour_alpha=0.3,
                             mg_contour_levels=5,
                             mg_contour_linewidth=1,
                             clay_z_tolerance=5.0,
                             
                             # Si network and Mg atom overlay parameters
                             show_si_network=False,
                             show_mgo_atoms=False,
                             show_si_atoms=False,
                             mg_vdw_scaling=2.0,
                             si_vdw_scaling=1.0,
                             si_connection_threshold=4.5,
                             si_color='orange',
                             si_connection_alpha=0.6,
                             si_connection_linewidth=1.2,
                             si_connection_style='lines',
                             show_hexagonal_pattern=True,
                             si_center_alpha=0.8,
                             si_atom_alpha=0.6,
                             si_radial_fade=True,
                             si_fade_alpha_min=0.1,
                             si_fade_alpha_max=0.8,
                             mgo_color='darkgreen',
                             mgo_atom_alpha=0.6,
                             mgo_radial_fade=True,
                             fade_alpha_min=0.1,
                             fade_alpha_max=0.8,
                             
                             # Cavity center overlay
                             show_cavity_centers=True,
                             cavity_center_color='red',
                             cavity_center_size=20,
                             cavity_center_alpha=0.5,
                             cavity_center_marker='o',
                             cavity_center_edgecolor='white',
                             cavity_center_linewidth=0.5,
                             
                             # Buffer control
                             buffer_2d=False,
                             buffer_3d=True,
                             
                             # Complete clay structure visualization (show top AND bottom simultaneously)
                             show_both_clay_surfaces=False,
                             clay_layer_separation=0.2,  # Separation between main and mirror clay layers (Å)
                             
                             # Surface elevation (shift data surface upward for clarity)
                             shift_surface=False,
                             shift_amount=2.0):  # Amount to shift surface upward (Å)
        """
        Plot cavity ion binding spatial distribution as 2D heatmaps, 3D histograms, or 3D surfaces with clay overlays.
        
        Creates dual-panel or single-panel visualizations showing:
        - Ion density distribution (ions per Ų)
        - Cavity occupancy distribution (mapped to XY space)
        
        Supports 2D and 3D visualization modes with full clay overlay capabilities.
        
        Parameters
        ----------
        ion_types : str or list, default='all'
            Ion types to plot. If 'all', plots all available ions.
        z_slices : str or list, default='all'
            Which z-slices to plot. If 'all', plots all available slices.
        grid_type : str, default='both'
            Which grids to plot: 'density', 'cavity_occupancy', or 'both'
            (density refers to either ion_density or functional_group_density depending on data type)
        plot_mode : str, default='2d'
            Visualization mode: '2d' (heatmaps), '3d' (bar histograms), 'surface' (smooth 3D surfaces)
        cavity_interpolation_method : str, default='weighted'
            Method for cavity occupancy grid visualization:
            - 'weighted': Distance-weighted interpolation from all cavities (smooth gradients,
              shows how occupancy decreases with distance from cavity centers)
            - 'nearest': Nearest-neighbor assignment (discrete regions, each grid point shows
              occupancy of nearest cavity)
            Both methods are pre-computed during analysis and can be plotted without re-running.
        
        Individual figure control:
            save_individual_figures, show_individual_figures, create_combined_figure, 
            save_combined_figure, show_combined_figure, individual_figsize
        
        Publication settings:
            title_fontsize, show_title, label_fontsize, tick_fontsize, colorbar_label_fontsize
        
        3D histogram parameters:
            bar_alpha, colormap, threshold_percentile
        
        3D surface parameters:
            surface_alpha, surface_cmap, surface_lighting, surface_smoothing, surface_smoothing_sigma,
            elevation, azimuth, z_scale_factor, z_axis_limit
        
        2D heatmap parameters:
            ion_density_cmap, cavity_occupancy_cmap, origin, interpolation, aspect
        
        Colorbar control:
            show_colorbar, colorbar_pad, colorbar_width, colorbar_vmin, colorbar_vmax
        
        Clay overlay parameters:
            overlay_clay_contours, clay_contour_results, mg_contour_*, clay_z_tolerance,
            show_si_network, show_mgo_atoms, show_si_atoms, *_radial_fade, *_fade_alpha_*
        
        Cavity center overlay:
            show_cavity_centers, cavity_center_color, cavity_center_size, etc.
        
        Buffer control:
            buffer_2d, buffer_3d - Prevents clay overlay clipping
        
        Returns
        -------
        matplotlib.figure.Figure or None
        
        Examples
        --------
        >>> # 2D heatmap with clay overlays
        >>> plotter.plot_cavity_occupancy(
        ...     ion_types='Na', plot_mode='2d',
        ...     show_si_network=True, show_cavity_centers=True
        ... )
        
        >>> # 3D surface with smooth rendering
        >>> plotter.plot_cavity_occupancy(
        ...     ion_types='Na', plot_mode='surface',
        ...     surface_smoothing=True, elevation=30, azimuth=45
        ... )
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # Determine data source: ion binding or organic binding
        use_organic = functional_groups is not None
        use_ions = ion_types is not None
        
        if use_organic and use_ions:
            print("❌ Cannot specify both ion_types and functional_groups!")
            print("   Use either ion_types= for ion binding data OR functional_groups= for organic binding data")
            return None
        
        if not use_organic and not use_ions:
            # Default to ions if available
            if 'cavity_ion_binding' in self.analysis.results:
                use_ions = True
                ion_types = 'all'
            elif 'cavity_organic_binding' in self.analysis.results:
                use_organic = True
                functional_groups = 'all'
            else:
                print("❌ No cavity binding data found!")
                print("   Please run analyze_cavity_ion_binding() or analyze_cavity_organic_binding() first")
                return None
        
        # Get the appropriate data source
        if use_organic:
            if 'cavity_organic_binding' not in self.analysis.results:
                print("❌ No cavity organic binding data found!")
                print("   Please run analyze_cavity_organic_binding(compute_xy_spatial=True) first")
                return None
            
            cavity_results = self.analysis.results['cavity_organic_binding']
            
            if 'functional_group_data' not in cavity_results:
                print("❌ Missing 'functional_group_data' in cavity results!")
                return None
            
            species_data = cavity_results['functional_group_data']
            species_param_name = 'functional_groups'
            species_display_name = 'functional groups'
            
        else:  # use_ions
            if 'cavity_ion_binding' not in self.analysis.results:
                print("❌ No cavity ion binding data found!")
                print("   Please run analyze_cavity_ion_binding(compute_xy_spatial=True) first")
                return None
            
            cavity_results = self.analysis.results['cavity_ion_binding']
            
            if 'ion_data' not in cavity_results:
                print("❌ Missing 'ion_data' in cavity results!")
                return None
            
            species_data = cavity_results['ion_data']
            species_param_name = 'ion_types'
            species_display_name = 'ion types'
        
        # Determine which species to plot
        available_species = list(species_data.keys())
        
        if not available_species:
            print(f"❌ No {species_display_name} data available!")
            return None
        
        # Get the species list from appropriate parameter
        species_input = functional_groups if use_organic else ion_types
        
        if species_input == 'all':
            species_to_plot = available_species
        elif isinstance(species_input, str):
            species_to_plot = [species_input] if species_input in available_species else []
        else:
            species_to_plot = [s for s in species_input if s in available_species]
        
        if not species_to_plot:
            print(f"❌ No valid {species_display_name} found! Available: {available_species}")
            return None
        
        print(f"\nPlotting cavity {'organic' if use_organic else 'ion'} binding spatial distribution ({plot_mode})")
        print(f"   {species_display_name.capitalize()}: {', '.join(species_to_plot)}")
        print(f"   Grid type: {grid_type}")
        if overlay_clay_contours:
            print(f"   Clay contour overlay: ENABLED")
        if show_si_network or show_mgo_atoms or show_si_atoms:
            print(f"   Clay atom overlays: Si network={show_si_network}, Si atoms={show_si_atoms}, Mg atoms={show_mgo_atoms}")
        print(f"   Buffer settings: 2D={buffer_2d}, 3D={buffer_3d}")
        
        # Get clay atom positions for overlays if needed
        si_positions = None
        mgo_positions = None
        
        # Detect center_box setting from cavity results (already have cavity_results from above)
        center_box_used = False
        if 'metadata' in cavity_results:
            center_box_used = cavity_results['metadata'].get('center_box', False)
        elif 'analysis_parameters' in cavity_results:
            center_box_used = cavity_results['analysis_parameters'].get('center_box', False)
        
        if show_si_network or show_mgo_atoms or show_si_atoms:
            try:
                self.analysis.u.trajectory[0]
                # Load Si atoms directly from universe (more robust)
                si_atoms = self.analysis.u.select_atoms('resname MMT and (name Si or name SI or name Sio or name SIO)')
                if len(si_atoms) > 0:
                    si_positions = si_atoms.positions.copy()
                    # Apply centering if cavity analysis used center_box=True
                    if center_box_used:
                        box_center_z = self.analysis.u.dimensions[2] / 2
                        si_positions[:, 2] -= box_center_z
                    print(f"   Loaded {len(si_atoms)} Si atoms (centered={center_box_used})")
                
                # Load Mg atoms
                mg_atoms = self.analysis.u.select_atoms('resname MMT and (name Mg or name MG or name Mgo or name MGO)')
                if len(mg_atoms) > 0:
                    mgo_positions = mg_atoms.positions.copy()
                    # Apply centering if cavity analysis used center_box=True
                    if center_box_used:
                        box_center_z = self.analysis.u.dimensions[2] / 2
                        mgo_positions[:, 2] -= box_center_z
                    print(f"   Loaded {len(mg_atoms)} Mg atoms (centered={center_box_used})")
            except Exception as e:
                print(f"   Warning: Could not load clay atoms: {e}")
        
        # VdW radii (Angstroms)
        BASE_MG_RADIUS = 0.72
        BASE_SI_RADIUS = 1.11
        
        # COMPREHENSIVE CLAY OVERLAY HELPER FUNCTIONS WITH 3D SUPPORT
        def add_vdw_mg_atoms(ax, z_center, is_3d=False, data_x_centers=None, data_y_centers=None, z_plot=0.0, show_both_surfaces=False):
            """Add VdW-scaled Mg atoms with optional radial fade and 3D support"""
            if not show_mgo_atoms or mgo_positions is None or len(mgo_positions) == 0:
                return
            
            if show_both_surfaces:
                # Determine which side we're on
                is_positive_slice = z_center >= 0
                
                # Get atoms from both surfaces
                top_mask = mgo_positions[:, 2] >= 0
                bottom_mask = mgo_positions[:, 2] < 0
                
                # ALWAYS position clay at/below data surface minimum (cavity on top)
                # z_plot is the data surface minimum - use it as reference
                # TOT structure after cleaving: Mg layers face each other in interlayer
                mg_offset = 0.1  # Å
                
                print(f"       DEBUG Mg atoms: z_plot={z_plot:.3f}, clay_layer_separation={clay_layer_separation:.3f}")
                print(f"       DEBUG: is_positive_slice={is_positive_slice} (TOP side={is_positive_slice})")
                
                if is_positive_slice:
                    # Plotting TOP side: main=top layer (flipped upside down)
                    main_atoms = mgo_positions[top_mask]
                    mirror_atoms = mgo_positions[bottom_mask]
                    # After flipping: Si outer (higher), Mg inner (lower)
                    main_z_offset = z_plot
                    # Mirror: Mg outer (higher), Si inner (lower)
                    mirror_z_offset = z_plot - clay_layer_separation
                    print(f"       DEBUG TOP (FLIPPED): Main Mg at {main_z_offset:.3f}, Mirror Mg at {mirror_z_offset:.3f}")
                else:
                    # Plotting BOTTOM side: main=bottom layer (Si outer, Mg inner)
                    main_atoms = mgo_positions[bottom_mask]
                    mirror_atoms = mgo_positions[top_mask]
                    # Bottom layer: Si OUTER (at z_plot), Mg INNER (below)
                    main_z_offset = z_plot - mg_offset
                    # Top layer (mirror): Mg OUTER, Si INNER (below Mg)
                    mirror_z_offset = z_plot - clay_layer_separation
                    print(f"       DEBUG BOTTOM: Main Mg at {main_z_offset:.3f}, Mirror Mg at {mirror_z_offset:.3f}")
                
                surfaces = [
                    ('main', main_atoms, main_z_offset, mgo_color, mgo_atom_alpha, 'black', 0.5),
                    ('mirror', mirror_atoms, mirror_z_offset, '#404040', mgo_atom_alpha * 0.6, 'white', 1.0)  # Dark gray with white outline
                ]
            else:
                # Original behavior: filter by region and z-tolerance
                is_positive_slice = z_center >= 0
                region_mask = (mgo_positions[:, 2] >= 0) if is_positive_slice else (mgo_positions[:, 2] < 0)
                z_tolerance_mask = np.abs(mgo_positions[:, 2] - z_center) <= clay_z_tolerance
                final_mask = region_mask & z_tolerance_mask
                
                if not np.any(final_mask):
                    print(f"       Warning: No Mg atoms in range for z={z_center:.2f}")
                    return
                
                mg_pos = mgo_positions[final_mask]
                # Use same structure as show_both_surfaces for consistency
                z_offset = z_plot if z_plot is not None and is_3d else 0
                surfaces = [('current', mg_pos, z_offset, mgo_color, mgo_atom_alpha, 'black', 0.5)]
            
            scaled_radius = BASE_MG_RADIUS * mg_vdw_scaling
            print(f"       Adding Mg atoms (radius: {scaled_radius:.2f}Å)")
            
            for surface_name, mg_pos, z_offset, color, alpha, edge_color, edge_width in surfaces:
                for mg in mg_pos:
                    x, y = mg[0], mg[1]
                    
                    # Apply coordinate transformation for 3D mode
                    if is_3d and data_x_centers is not None and data_y_centers is not None:
                        x = x - data_x_centers.min()
                        y = y - data_y_centers.min()
                    
                    # Position near data surface instead of at actual Z
                    z_final = z_offset if show_both_surfaces else z_plot
                    
                    if mgo_radial_fade:
                        n_circles = 5
                        alphas = np.linspace(fade_alpha_max, fade_alpha_min, n_circles)
                        radii = np.linspace(0.3 * scaled_radius, scaled_radius, n_circles)
                        
                        for i, (radius, circle_alpha) in enumerate(zip(radii, alphas)):
                            # Adjust alpha for mirror layer
                            final_alpha = circle_alpha * alpha if show_both_surfaces else circle_alpha
                            if is_3d:
                                ax.scatter(x, y, z_final, s=radius*200, c=color, 
                                         alpha=final_alpha, marker='o', edgecolors=edge_color,
                                         linewidth=edge_width if i == len(radii)-1 else 0.2, zorder=5+i)
                            else:
                                circle = plt.Circle((x, y), radius, color=color, 
                                                  alpha=final_alpha, zorder=5+i)
                                circle.set_clip_box(ax.bbox)
                                ax.add_patch(circle)
                    else:
                        if is_3d:
                            ax.scatter(x, y, z_final, s=scaled_radius*200, c=color,
                                     alpha=alpha, marker='o', edgecolors=edge_color,
                                     linewidth=edge_width, zorder=10)
                        else:
                            circle = plt.Circle((x, y), scaled_radius, color=color,
                                              alpha=alpha, zorder=10)
                            circle.set_clip_box(ax.bbox)
                            ax.add_patch(circle)
        
        def add_si_network(ax, z_center, is_3d=False, data_x_centers=None, data_y_centers=None, z_plot=0.0, show_both_surfaces=False):
            """Add Si network with connections and 3D support"""
            if not show_si_network or si_positions is None or len(si_positions) == 0:
                return
            
            if show_both_surfaces:
                # Determine which side we're on
                is_positive_slice = z_center >= 0
                
                # Get atoms from both surfaces
                top_mask = si_positions[:, 2] >= 0
                bottom_mask = si_positions[:, 2] < 0
                
                # ALWAYS position clay at/below data surface minimum (cavity on top)
                # z_plot is the data surface minimum - use it as reference
                # TOT structure: Si offset logic matches Mg logic
                mg_offset = 0.1  # Å
                
                print(f"       DEBUG Si atoms: z_plot={z_plot:.3f}, clay_layer_separation={clay_layer_separation:.3f}")
                print(f"       DEBUG: is_positive_slice={is_positive_slice} (TOP side={is_positive_slice})")
                
                if is_positive_slice:
                    # Plotting TOP side: main=top layer (flipped upside down)
                    main_atoms = si_positions[top_mask]
                    mirror_atoms = si_positions[bottom_mask]
                    # After flipping: Si outer (higher), Mg inner (lower)
                    main_z_offset = z_plot + mg_offset
                    # Mirror: Mg outer (higher), Si inner (lower)
                    mirror_z_offset = z_plot - clay_layer_separation - mg_offset
                    print(f"       DEBUG TOP (FLIPPED): Main Si at {main_z_offset:.3f}, Mirror Si at {mirror_z_offset:.3f}")
                else:
                    # Plotting BOTTOM side: main=bottom layer (Si outer, Mg inner)
                    main_atoms = si_positions[bottom_mask]
                    mirror_atoms = si_positions[top_mask]
                    # Bottom layer: Si OUTER (above Mg at z_plot-mg_offset)
                    main_z_offset = z_plot
                    # Top layer (mirror): Si INNER (below Mg at z_plot-sep)
                    mirror_z_offset = z_plot - clay_layer_separation - mg_offset
                    print(f"       DEBUG BOTTOM: Main Si at {main_z_offset:.3f}, Mirror Si at {mirror_z_offset:.3f}")
                
                surfaces = [
                    ('main', main_atoms, main_z_offset, si_color, si_connection_alpha, si_center_alpha),
                    ('mirror', mirror_atoms, mirror_z_offset, '#808080', si_connection_alpha * 0.4, si_center_alpha * 0.4)  # Light gray
                ]
            else:
                # Original behavior
                is_positive_slice = z_center >= 0
                region_mask = (si_positions[:, 2] >= 0) if is_positive_slice else (si_positions[:, 2] < 0)
                z_tolerance_mask = np.abs(si_positions[:, 2] - z_center) <= clay_z_tolerance
                final_mask = region_mask & z_tolerance_mask
                
                if not np.any(final_mask):
                    print(f"       Warning: No Si atoms in range for z={z_center:.2f}")
                    return
                
                si_pos = si_positions[final_mask]
                # Use same structure as show_both_surfaces for consistency
                z_offset = z_plot if z_plot is not None and is_3d else 0
                surfaces = [('current', si_pos, z_offset, si_color, si_connection_alpha, si_center_alpha)]
            
            print(f"       Adding Si network, threshold: {si_connection_threshold:.1f}Å)")
            
            for surface_name, si_pos, z_offset, color, conn_alpha, center_alpha in surfaces:
                si_array = si_pos[:, :2].copy()
                
                # Apply coordinate transformation for 3D mode
                if is_3d and data_x_centers is not None and data_y_centers is not None:
                    si_array[:, 0] -= data_x_centers.min()
                    si_array[:, 1] -= data_y_centers.min()
                
                for i in range(len(si_array)):
                    for j in range(i+1, len(si_array)):
                        dist = np.linalg.norm(si_array[i] - si_array[j])
                        if dist <= si_connection_threshold:
                            x_coords = [si_array[i][0], si_array[j][0]]
                            y_coords = [si_array[i][1], si_array[j][1]]
                            
                            # Position near data surface
                            z_coords = [z_offset, z_offset] if show_both_surfaces else [z_plot, z_plot]
                            
                            linestyle = '-' if si_connection_style == 'lines' else ('--' if si_connection_style == 'dashed' else ':')
                            
                            if is_3d:
                                ax.plot(x_coords, y_coords, z_coords,
                                  color=color, alpha=conn_alpha,
                                  linewidth=si_connection_linewidth, 
                                  linestyle=linestyle, zorder=3)
                
                if show_hexagonal_pattern:
                    z_positions = [z_offset]*len(si_array) if show_both_surfaces else [z_plot]*len(si_array)
                    if is_3d:
                        ax.scatter(si_array[:, 0], si_array[:, 1], z_positions,
                                 s=25, c=color, alpha=center_alpha,
                                 marker='o', edgecolors='black', linewidth=0.5, zorder=4)
                    else:
                        scatter = ax.scatter(si_array[:, 0], si_array[:, 1],
                                           s=20, c=color, alpha=center_alpha,
                                           marker='o', edgecolors='black', linewidth=0.5, zorder=4)
                        scatter.set_clip_box(ax.bbox)
        
        def add_vdw_si_atoms(ax, z_center, is_3d=False):
            """Add VdW-scaled Si atoms with optional radial fade and 3D support"""
            if not show_si_atoms or si_positions is None or len(si_positions) == 0:
                return
            
            is_positive_slice = z_center >= 0
            region_mask = (si_positions[:, 2] >= 0) if is_positive_slice else (si_positions[:, 2] < 0)
            z_tolerance_mask = np.abs(si_positions[:, 2] - z_center) <= clay_z_tolerance
            final_mask = region_mask & z_tolerance_mask
            
            if not np.any(final_mask):
                return
            
            si_atom_pos = si_positions[final_mask]
            scaled_radius = BASE_SI_RADIUS * si_vdw_scaling
            
            print(f"       🔷 Adding {len(si_atom_pos)} Si atoms (radius: {scaled_radius:.2f}Å)")
            
            for si in si_atom_pos:
                x, y = si[0], si[1]
                
                if si_radial_fade:
                    n_circles = 5
                    alphas = np.linspace(si_fade_alpha_max, si_fade_alpha_min, n_circles)
                    radii = np.linspace(0.3 * scaled_radius, scaled_radius, n_circles)
                    
                    for i, (radius, alpha) in enumerate(zip(radii, alphas)):
                        if is_3d:
                            ax.scatter(x, y, 0.0, s=radius*200, c=si_color,
                                     alpha=alpha, marker='s', edgecolors='black',
                                     linewidth=0.2, zorder=5+i)
                        else:
                            circle = plt.Circle((x, y), radius, color=si_color,
                                              alpha=alpha, zorder=5+i)
                            circle.set_clip_box(ax.bbox)
                            ax.add_patch(circle)
                else:
                    if is_3d:
                        ax.scatter(x, y, 0.0, s=scaled_radius*200, c=si_color,
                                 alpha=si_atom_alpha, marker='s', edgecolors='black',
                                 linewidth=0.5, zorder=10)
                    else:
                        circle = plt.Circle((x, y), scaled_radius, color=si_color,
                                          alpha=si_atom_alpha, zorder=10)
                        circle.set_clip_box(ax.bbox)
                        ax.add_patch(circle)
        
        def add_clay_contour_overlay(ax, z_center, is_3d=False, data_x_centers=None, data_y_centers=None, contour_z_offset=None, show_both_surfaces=False):
            """Add clay contour lines with proper coordinate alignment
            
            Parameters
            ----------
            ax : matplotlib axes
                Axes to plot on
            z_center : float
                Z-coordinate of slice
            is_3d : bool
                Whether this is a 3D plot
            data_x_centers : array, optional
                X-coordinates of the plotted data (for 3D coordinate alignment)
            data_y_centers : array, optional
                Y-coordinates of the plotted data (for 3D coordinate alignment)
            contour_z_offset : float, optional
                Z-position for 3D contours (default: -0.01 or grid minimum)
            show_both_surfaces : bool, optional
                If True, show contours for both top and bottom surfaces
            """
            if not overlay_clay_contours or clay_contour_results is None:
                return
            
            if not clay_contour_results.get('z_slices'):
                return
            
            if show_both_surfaces:
                # Find both top and bottom clay slices (one from each region)
                clay_slices_to_plot = []
                
                # Find best top slice
                best_top_slice = None
                min_top_dist = float('inf')
                for clay_slice in clay_contour_results['z_slices']:
                    clay_z = clay_slice.get('z_center')
                    if clay_z is not None and clay_z >= 0:
                        if abs(clay_z) < min_top_dist:
                            min_top_dist = abs(clay_z)
                            best_top_slice = clay_slice
                
                # Find best bottom slice
                best_bottom_slice = None
                min_bottom_dist = float('inf')
                for clay_slice in clay_contour_results['z_slices']:
                    clay_z = clay_slice.get('z_center')
                    if clay_z is not None and clay_z < 0:
                        if abs(clay_z) < min_bottom_dist:
                            min_bottom_dist = abs(clay_z)
                            best_bottom_slice = clay_slice
                
                # Determine which is main and which is mirror based on z_center
                is_positive_slice = z_center >= 0
                # Use contour_z_offset as base (data surface minimum)
                base_z = contour_z_offset if contour_z_offset is not None else 0.0
                
                # TOT structure: Mg layers face interlayer after cleaving
                mg_offset = 0.1  # Å
                
                if is_positive_slice:
                    # TOP side: flipped upside down, Mg layers inverted
                    # Contours represent Mg layer positions
                    if best_top_slice:
                        # Top layer (flipped): Mg inner (at reference)
                        clay_slices_to_plot.append(('main', best_top_slice, base_z))
                    if best_bottom_slice:
                        # Bottom layer (mirror): Mg outer (less separation)
                        clay_slices_to_plot.append(('mirror', best_bottom_slice, base_z - clay_layer_separation))
                else:
                    # BOTTOM side: main=bottom layer (Si outer, Mg inner)
                    # Contours represent Mg layer positions
                    if best_bottom_slice:
                        # Bottom layer: Mg INNER (below Si at base_z)
                        clay_slices_to_plot.append(('main', best_bottom_slice, base_z - mg_offset))
                    if best_top_slice:
                        # Top layer (mirror): Mg OUTER (above Si)
                        clay_slices_to_plot.append(('mirror', best_top_slice, base_z - clay_layer_separation))
            else:
                # Original behavior: find best matching slice
                best_clay_slice = None
                min_distance = float('inf')
                
                for clay_slice in clay_contour_results['z_slices']:
                    clay_z = clay_slice.get('z_center')
                    if clay_z is not None:
                        dist = abs(clay_z - z_center)
                        if dist <= clay_z_tolerance and dist < min_distance:
                            min_distance = dist
                            best_clay_slice = clay_slice
                
                if best_clay_slice is None:
                    return
                clay_slices_to_plot = [('current', best_clay_slice, None)]
            
            for surface_info in clay_slices_to_plot:
                if show_both_surfaces:
                    surface_name, clay_slice, z_offset = surface_info
                else:
                    surface_name, clay_slice, _ = surface_info
                    z_offset = None
                
                clay_grid = None
                for key in ['mg_grid', 'combined_grid', 'si_grid']:
                    if key in clay_slice and clay_slice[key] is not None:
                        clay_grid = clay_slice[key]
                        break
                
                if clay_grid is None or np.max(clay_grid) == 0:
                    continue
                
                x_centers_clay = clay_slice['x_centers']
                y_centers_clay = clay_slice['y_centers']
                
                # Match coordinate system used by the data:
                # - 2D mode: use original coordinates (matches imshow extent)
                # - 3D mode: transform using DATA's reference point to ensure alignment
                #   Clay and data may have different grid spacings, so they must use
                #   the SAME origin point to stay aligned despite resolution differences
                if is_3d and data_x_centers is not None and data_y_centers is not None:
                    # Use DATA's minimum as reference point - ensures same coordinate system
                    x_clay_plot = x_centers_clay - data_x_centers.min()
                    y_clay_plot = y_centers_clay - data_y_centers.min()
                elif is_3d:
                    # Fallback if data coordinates not provided
                    x_clay_plot = x_centers_clay - x_centers_clay.min()
                    y_clay_plot = y_centers_clay - y_centers_clay.min()
                else:
                    # 2D: use original coordinates
                    x_clay_plot = x_centers_clay
                    y_clay_plot = y_centers_clay
                
                X, Y = np.meshgrid(x_clay_plot, y_clay_plot)
                
                non_zero = clay_grid[clay_grid > 0]
                if len(non_zero) == 0:
                    continue
                
                levels = np.linspace(np.min(non_zero), np.max(non_zero), mg_contour_levels)
                
                try:
                    if is_3d:
                        # Use z_offset if provided (for both surfaces mode), else use contour_z_offset
                        if show_both_surfaces and z_offset is not None:
                            # Use compacted Z position
                            z_pos = z_offset
                        else:
                            # Original behavior: use contour_z_offset or default
                            z_pos = contour_z_offset if contour_z_offset is not None else -0.01
                        
                        # Adjust alpha and color for mirror layer
                        contour_alpha = mg_contour_alpha * 0.4 if (show_both_surfaces and surface_name == 'mirror') else mg_contour_alpha
                        contour_color = '#606060' if (show_both_surfaces and surface_name == 'mirror') else mg_contour_color  # Gray for mirror
                        
                        # Place contours at specified z-position
                        ax.contour(X, Y, clay_grid, levels=levels,
                                 colors=[contour_color], alpha=contour_alpha,
                                 linewidths=mg_contour_linewidth, zdir='z', offset=z_pos)
                    else:
                        ax.contour(X, Y, clay_grid, levels=levels,
                                 colors=[mg_contour_color], alpha=mg_contour_alpha,
                                 linewidths=mg_contour_linewidth)
                    print(f"        Added clay contours")
                except Exception as e:
                    print(f"        ❌ Error adding contours: {e}")
        
        def add_cavity_centers_overlay(ax, cavity_centers_xy, is_3d=False):
            """Add cavity center positions"""
            if not show_cavity_centers or cavity_centers_xy is None or len(cavity_centers_xy) == 0:
                return
            
            if is_3d:
                ax.scatter(cavity_centers_xy[:, 0], cavity_centers_xy[:, 1], [0.0]*len(cavity_centers_xy),
                          c=cavity_center_color, s=cavity_center_size,
                          alpha=cavity_center_alpha, marker=cavity_center_marker,
                          edgecolors=cavity_center_edgecolor,
                          linewidths=cavity_center_linewidth, zorder=12)
            else:
                ax.scatter(cavity_centers_xy[:, 0], cavity_centers_xy[:, 1],
                          c=cavity_center_color, s=cavity_center_size,
                          alpha=cavity_center_alpha, marker=cavity_center_marker,
                          edgecolors=cavity_center_edgecolor,
                          linewidths=cavity_center_linewidth, zorder=12)
        
        # Store all figures for return
        all_figures = []
        
        # Process each species (ion or functional group)
        for species_idx, species in enumerate(species_to_plot):
            species_z_data = species_data[species]
            
            # Get available z-slices for this species
            available_z_centers = sorted([z for z in species_z_data.keys() if isinstance(z, (int, float))])
            
            if not available_z_centers:
                print(f"   Warning: No z-slices found for {species}")
                continue
            
            # Determine which z-slices to plot
            if z_slices == 'all':
                z_centers_to_plot = available_z_centers
            elif isinstance(z_slices, (list, tuple)):
                z_centers_to_plot = [z for z in z_slices if z in available_z_centers]
            else:
                z_centers_to_plot = [z_slices] if z_slices in available_z_centers else []
            
            if not z_centers_to_plot:
                print(f"   Warning: No valid z-slices for {species}")
                continue
            
            print(f"\n   Processing {species}: {len(z_centers_to_plot)} z-slices")
            
            # MODE-SPECIFIC RENDERING FUNCTION
            def create_single_cavity_plot(ax, grid_data, x_centers_data, y_centers_data, z_center_val, 
                                        grid_name, cavity_centers_xy_data, box_x_val, box_y_val):
                """Create single cavity plot in specified mode (2D, 3D, or surface)"""
                is_3d_mode = (plot_mode == '3d' or plot_mode == 'surface')
                
                # Determine colormap and vmin/vmax based on grid type
                if grid_name == 'density':
                    cmap_to_use = ion_density_cmap
                    vmin_to_use = ion_density_vmin
                    vmax_to_use = ion_density_vmax
                    cbar_label = density_cbar_label
                else:  # cavity_occupancy
                    cmap_to_use = cavity_occupancy_cmap
                    vmin_to_use = cavity_occupancy_vmin
                    vmax_to_use = cavity_occupancy_vmax
                    cbar_label = 'Cavity Occupancy'
                
                if plot_mode == '2d':
                    # 2D HEATMAP MODE
                    # Transform coordinates to start from 0 (match electrostatic method)
                    x_display_min = 0
                    x_display_max = box_x_val
                    y_display_min = 0
                    y_display_max = box_y_val
                    
                    im = ax.imshow(grid_data, origin=origin,
                                  extent=[x_display_min, x_display_max, y_display_min, y_display_max],
                                  cmap=cmap_to_use, aspect=aspect,
                                  interpolation=interpolation,
                                  vmin=vmin_to_use, vmax=vmax_to_use)
                    
                    # Set axis limits before overlays
                    ax.set_xlim(x_display_min, x_display_max)
                    ax.set_ylim(y_display_min, y_display_max)
                    
                    # Add overlays
                    add_clay_contour_overlay(ax, z_center_val, is_3d=False)
                    add_si_network(ax, z_center_val, is_3d=False)
                    add_vdw_si_atoms(ax, z_center_val, is_3d=False)
                    add_vdw_mg_atoms(ax, z_center_val, is_3d=False)
                    add_cavity_centers_overlay(ax, cavity_centers_xy_data, is_3d=False)
                    
                    return im
                
                elif plot_mode == '3d':
                    # 3D BAR HISTOGRAM MODE
                    from scipy.ndimage import gaussian_filter
                    
                    # Create meshgrid
                    x_display = x_centers_data - x_centers_data.min()
                    y_display = y_centers_data - y_centers_data.min()
                    X, Y = np.meshgrid(x_display, y_display)
                    
                    # Apply adaptive Z-scaling for display
                    # z_scale_factor becomes target maximum height
                    if z_scale_factor is not None:
                        data_max = np.max(grid_data)
                        if data_max > 0:
                            adaptive_factor = z_scale_factor / data_max
                            grid_scaled = grid_data * adaptive_factor
                        else:
                            grid_scaled = grid_data
                    else:
                        grid_scaled = grid_data
                    
                    # Flatten for bar3d
                    x_flat = X.flatten()
                    y_flat = Y.flatten()
                    z_flat = np.zeros_like(x_flat)
                    heights = grid_scaled.flatten()
                    
                    # Apply threshold
                    threshold = np.percentile(heights, threshold_percentile)
                    mask = heights > threshold
                    
                    if np.any(mask):
                        dx = x_display[1] - x_display[0] if len(x_display) > 1 else 1.0
                        dy = y_display[1] - y_display[0] if len(y_display) > 1 else 1.0
                        
                        bars = ax.bar3d(x_flat[mask], y_flat[mask], z_flat[mask],
                                    dx, dy, heights[mask], alpha=bar_alpha, cmap=colormap)
                        
                        # Color mapping - use scaled heights for colors when z_scale_factor is applied
                        scaled_heights = grid_scaled.flatten()[mask]
                        vmin_color = colorbar_vmin if colorbar_vmin is not None else (vmin_to_use if vmin_to_use is not None else np.min(scaled_heights))
                        vmax_color = colorbar_vmax if colorbar_vmax is not None else (vmax_to_use if vmax_to_use is not None else np.max(scaled_heights))
                        
                        colors = plt.cm.get_cmap(colormap)(
                            (scaled_heights - vmin_color) / (vmax_color - vmin_color) 
                            if vmax_color > vmin_color else np.zeros_like(original_heights)
                        )
                        bars.set_facecolors(colors)
                    
                    # Set axis limits with transformed coordinates
                    x_display_max = x_display[-1] + (x_display[1] - x_display[0]) if len(x_display) > 1 else box_x_val
                    y_display_max = y_display[-1] + (y_display[1] - y_display[0]) if len(y_display) > 1 else box_y_val
                    ax.set_xlim(0, x_display_max)
                    ax.set_ylim(0, y_display_max)
                    
                    # Add overlays (pass data coordinates for proper alignment)
                    add_clay_contour_overlay(ax, z_center_val, is_3d=True, 
                                            data_x_centers=x_centers_data, 
                                            data_y_centers=y_centers_data,
                                            contour_z_offset=smoothed_grid_scaled.min(),
                                            show_both_surfaces=show_both_clay_surfaces)
                    add_si_network(ax, z_center_val, is_3d=True, data_x_centers=x_centers_data, data_y_centers=y_centers_data, z_plot=smoothed_grid_scaled.min(), show_both_surfaces=show_both_clay_surfaces)
                    add_vdw_si_atoms(ax, z_center_val, is_3d=True)
                    add_vdw_mg_atoms(ax, z_center_val, is_3d=True, data_x_centers=x_centers_data, data_y_centers=y_centers_data, z_plot=smoothed_grid_scaled.min(), show_both_surfaces=show_both_clay_surfaces)
                    add_cavity_centers_overlay(ax, cavity_centers_xy_data, is_3d=True)
                    
                    # Labels and styling
                    ax.set_xlabel('X (Å)', fontsize=label_fontsize, labelpad=15)
                    ax.set_ylabel('Y (Å)', fontsize=label_fontsize, labelpad=15)
                    ax.set_zlabel(cbar_label, fontsize=label_fontsize, labelpad=10)
                    ax.tick_params(axis='y', which='major', pad=10, labelsize=tick_fontsize)
                    ax.tick_params(axis='x', which='major', pad=8, labelsize=tick_fontsize)
                    ax.tick_params(axis='z', which='major', pad=5, labelsize=tick_fontsize)
                    
                    # Set Z-axis limits
                    if show_both_clay_surfaces:
                        # Clay at data minimum (unshifted), surface may be shifted
                        z_min_clay = np.min(smoothed_grid_scaled) - clay_layer_separation - 0.2
                        z_max_surface = np.max(smoothed_grid_shifted)
                        ax.set_zlim(z_min_clay, z_max_surface * 1.1)
                    elif z_axis_limit is not None:
                        ax.set_zlim(0, z_axis_limit)
                    
                    ax.set_box_aspect([1, 1, 0.5])
                    
                    if elevation is not None or azimuth is not None:
                        elev = elevation if elevation is not None else ax.elev
                        azim = azimuth if azimuth is not None else ax.azim
                        ax.view_init(elev=elev, azim=azim)
                    
                    return None
                
                elif plot_mode == 'surface':
                    # 3D SURFACE MODE
                    from scipy.ndimage import gaussian_filter
                    
                    # Apply smoothing
                    if surface_smoothing:
                        smoothed_grid = gaussian_filter(grid_data, sigma=surface_smoothing_sigma)
                    else:
                        smoothed_grid = grid_data
                    
                    # Apply adaptive Z-scaling
                    # z_scale_factor becomes target maximum height
                    if z_scale_factor is not None:
                        data_max = np.max(smoothed_grid)
                        if data_max > 0:
                            adaptive_factor = z_scale_factor / data_max
                            smoothed_grid_scaled = smoothed_grid * adaptive_factor
                        else:
                            smoothed_grid_scaled = smoothed_grid
                    else:
                        smoothed_grid_scaled = smoothed_grid
                    
                    # Create meshgrid
                    x_display = x_centers_data - x_centers_data.min()
                    y_display = y_centers_data - y_centers_data.min()
                    X, Y = np.meshgrid(x_display, y_display)
                    
                    # Apply surface shift if requested (elevate data surface for clarity)
                    # Shift only affects Z position, NOT color mapping
                    if shift_surface:
                        smoothed_grid_shifted = smoothed_grid_scaled + shift_amount
                    else:
                        smoothed_grid_shifted = smoothed_grid_scaled
                    
                    # Colorbar limits - ALWAYS use unshifted scaled data for colors
                    vmin_color = colorbar_vmin if colorbar_vmin is not None else (vmin_to_use if vmin_to_use is not None else np.min(smoothed_grid_scaled))
                    vmax_color = colorbar_vmax if colorbar_vmax is not None else (vmax_to_use if vmax_to_use is not None else np.max(smoothed_grid_scaled))
                    
                    # Create surface with shifted Z but unshifted colors
                    surf = ax.plot_surface(X, Y, smoothed_grid_shifted,
                                        cmap=surface_cmap, alpha=surface_alpha,
                                        vmin=vmin_color, vmax=vmax_color,
                                        facecolors=plt.cm.get_cmap(surface_cmap)((smoothed_grid_scaled - vmin_color) / (vmax_color - vmin_color)),
                                        rstride=surface_rstride, cstride=surface_cstride,
                                        linewidth=surface_linewidth,
                                        antialiased=surface_antialiased,
                                        shade=surface_lighting)
                    
                    # Set axis limits with transformed coordinates
                    x_display_max = x_display[-1] + (x_display[1] - x_display[0]) if len(x_display) > 1 else box_x_val
                    y_display_max = y_display[-1] + (y_display[1] - y_display[0]) if len(y_display) > 1 else box_y_val
                    ax.set_xlim(0, x_display_max)
                    ax.set_ylim(0, y_display_max)
                    
                    # Add overlays
                    add_clay_contour_overlay(ax, z_center_val, is_3d=True, 
                                            data_x_centers=x_centers_data, 
                                            data_y_centers=y_centers_data,
                                            contour_z_offset=smoothed_grid_scaled.min(),
                                            show_both_surfaces=show_both_clay_surfaces)
                    add_si_network(ax, z_center_val, is_3d=True, data_x_centers=x_centers_data, data_y_centers=y_centers_data, z_plot=smoothed_grid_scaled.min(), show_both_surfaces=show_both_clay_surfaces)
                    add_vdw_si_atoms(ax, z_center_val, is_3d=True)
                    add_vdw_mg_atoms(ax, z_center_val, is_3d=True, data_x_centers=x_centers_data, data_y_centers=y_centers_data, z_plot=smoothed_grid_scaled.min(), show_both_surfaces=show_both_clay_surfaces)
                    add_cavity_centers_overlay(ax, cavity_centers_xy_data, is_3d=True)
                    
                    # Apply buffer ONCE after all overlays
                    if buffer_3d:
                        x_min, x_max = ax.get_xlim()
                        y_min, y_max = ax.get_ylim()
                        x_buffer = (x_max - x_min) * 0.02
                        y_buffer = (y_max - y_min) * 0.02
                        ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
                        ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
                    
                    # Invert axes to match electrostatic plotting (center shows max values)
                    ax.invert_xaxis()
                    ax.invert_yaxis()
                    
                    # Transparent panes
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                    ax.xaxis.pane.set_edgecolor('white')
                    ax.yaxis.pane.set_edgecolor('white')
                    ax.zaxis.pane.set_edgecolor('white')
                    ax.xaxis.pane.set_alpha(0)
                    ax.yaxis.pane.set_alpha(0)
                    ax.zaxis.pane.set_alpha(0)
                    
                    # Labels and styling
                    ax.set_xlabel('X (Å)', fontsize=label_fontsize, labelpad=15)
                    ax.set_ylabel('Y (Å)', fontsize=label_fontsize, labelpad=15)
                    ax.set_zlabel(cbar_label, fontsize=label_fontsize, labelpad=10)
                    ax.tick_params(axis='y', which='major', pad=10, labelsize=tick_fontsize)
                    ax.tick_params(axis='x', which='major', pad=8, labelsize=tick_fontsize)
                    ax.tick_params(axis='z', which='major', pad=5, labelsize=tick_fontsize)
                    
                    # Set Z-axis limits
                    if show_both_clay_surfaces:
                        # Clay at data minimum (unshifted), surface may be shifted
                        z_min_clay = np.min(smoothed_grid_scaled) - clay_layer_separation - 0.2
                        z_max_surface = np.max(smoothed_grid_shifted)
                        ax.set_zlim(z_min_clay, z_max_surface * 1.1)
                    elif z_axis_limit is not None:
                        ax.set_zlim(0, z_axis_limit)
                    else:
                        # Auto-set based on surface data
                        z_max_data = np.max(smoothed_grid_shifted) if 'smoothed_grid_shifted' in locals() else 1.0
                        ax.set_zlim(0, z_max_data * 1.1)
                    
                    ax.set_box_aspect([1, 1, 0.6])
                    
                    elev = elevation if elevation is not None else 30
                    azim = azimuth if azimuth is not None else 45
                    ax.view_init(elev=elev, azim=azim)
                    
                    return surf
            
            # PROCESS EACH Z-SLICE (INDIVIDUAL FIGURES)
            for z_idx, z_center in enumerate(z_centers_to_plot):
                if 'xy_spatial' not in species_z_data[z_center]:
                    print(f"     Warning: No xy_spatial data for z={z_center:.1f} Å")
                    continue
                
                xy_data = species_z_data[z_center]['xy_spatial']
                
                # Extract data with appropriate key names
                if use_organic:
                    density_grid_key = 'functional_group_density_grid'
                    density_label = f'{species} Density'
                    density_cbar_label = 'Functional Group Density'
                else:
                    density_grid_key = 'ion_density_grid'
                    density_label = f'{species} Ion Density'
                    density_cbar_label = 'Ion Density (ions/Ų)'
                
                density_grid = xy_data[density_grid_key]
                
                # Select cavity occupancy grid based on interpolation method
                if cavity_interpolation_method == 'weighted':
                    # Try new key first, fallback to old key for backward compatibility
                    if 'cavity_occupancy_grid_weighted' in xy_data:
                        cavity_occupancy_grid = xy_data['cavity_occupancy_grid_weighted']
                    else:
                        cavity_occupancy_grid = xy_data.get('cavity_occupancy_grid')
                elif cavity_interpolation_method == 'nearest':
                    # Try new key first, fallback to old key for backward compatibility
                    if 'cavity_occupancy_grid_nearest' in xy_data:
                        cavity_occupancy_grid = xy_data['cavity_occupancy_grid_nearest']
                    else:
                        cavity_occupancy_grid = xy_data.get('cavity_occupancy_grid')
                else:
                    raise ValueError(f"Invalid cavity_interpolation_method: {cavity_interpolation_method}. Use 'weighted' or 'nearest'")
                
                x_centers = xy_data['x_centers']
                y_centers = xy_data['y_centers']
                box_x, box_y = xy_data['box_dimensions']
                surface_type = xy_data['surface_type']
                cavity_centers_xy = xy_data.get('cavity_centers_xy', None)
                
                # Print grid statistics
                print(f"     z={z_center:.2f} Å ({surface_type}):")
                print(f"        {density_label}: min={np.min(density_grid):.4f}, max={np.max(density_grid):.4f}, mean={np.mean(density_grid):.4f}")
                print(f"        Cavity occupancy ({cavity_interpolation_method}): min={np.min(cavity_occupancy_grid):.4f}, max={np.max(cavity_occupancy_grid):.4f}, mean={np.mean(cavity_occupancy_grid):.4f}")
                
                # Determine number of panels for DISPLAY figure
                if grid_type == 'density':
                    num_panels = 1
                    grids_to_plot = [('density', density_grid)]
                elif grid_type == 'occupancy':
                    num_panels = 1
                    grids_to_plot = [('cavity_occupancy', cavity_occupancy_grid)]
                else:  # both - create combined display with 2 panels side-by-side
                    num_panels = 2
                    grids_to_plot = [('density', density_grid), ('cavity_occupancy', cavity_occupancy_grid)]
                
                # CREATE COMBINED DISPLAY FIGURE (for notebook/window viewing)
                if plot_mode == '2d':
                    # Use individual_figsize for individual figures
                    fig = plt.figure(figsize=(individual_figsize[0]*num_panels, individual_figsize[1]))
                    
                    for panel_idx, (grid_name, grid_data_val) in enumerate(grids_to_plot, start=1):
                        ax = fig.add_subplot(1, num_panels, panel_idx)
                        
                        # Create plot and get image handle
                        im = create_single_cavity_plot(ax, grid_data_val, x_centers, y_centers, 
                                                      z_center, grid_name, cavity_centers_xy, 
                                                      box_x, box_y)
                        
                        # Set title and labels
                        if grid_name == 'density':
                            title = f'{density_label} - z={z_center:.1f} Å ({surface_type})'
                            cbar_label = density_cbar_label
                        else:
                            title = f'{species} Cavity Occupancy - z={z_center:.1f} Å ({surface_type})'
                            cbar_label = 'Cavity Occupancy'
                        
                        ax.set_title(title, fontsize=title_fontsize)
                        ax.set_xlabel('X (Å)', fontsize=label_fontsize, labelpad=15)
                        ax.set_ylabel('Y (Å)', fontsize=label_fontsize, labelpad=15)
                        ax.tick_params(labelsize=tick_fontsize)
                        
                        if show_grid:
                            ax.grid(alpha=grid_alpha, linestyle='--', linewidth=0.5)
                        
                        # Add colorbar
                        if show_colorbar and im is not None:
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size=colorbar_width, pad=colorbar_pad)
                            cbar = plt.colorbar(im, cax=cax)
                            cbar.set_label(cbar_label, fontsize=colorbar_label_fontsize)
                    
                    plt.tight_layout()
                
                else:  # 3D or surface mode
                    from mpl_toolkits.mplot3d import Axes3D
                    
                    # Create figure with 3D subplots - use individual_figsize
                    if num_panels == 1:
                        fig = plt.figure(figsize=individual_figsize)
                        ax = fig.add_subplot(111, projection='3d')
                        axes = [ax]
                    else:
                        fig = plt.figure(figsize=(individual_figsize[0]*num_panels, individual_figsize[1]))
                        axes = []
                        for i in range(num_panels):
                            ax = fig.add_subplot(1, num_panels, i+1, projection='3d')
                            axes.append(ax)
                    
                    # Create plots
                    for panel_idx, (grid_name, grid_data_val) in enumerate(grids_to_plot):
                        ax_curr = axes[panel_idx]
                        
                        # Render 3D plot
                        create_single_cavity_plot(ax_curr, grid_data_val, x_centers, y_centers,
                                                z_center, grid_name, cavity_centers_xy,
                                                box_x, box_y)
                        
                        # Set title
                        if grid_name == 'density':
                            title = f'{density_label} - z={z_center:.1f} Å ({surface_type})'
                        else:
                            title = f'{species} Cavity Occupancy - z={z_center:.1f} Å ({surface_type})'
                        
                        ax_curr.set_title(title, fontsize=title_fontsize, pad=20)
                    
                    # Adjust spacing for 3D
                    if num_panels > 1:
                        plt.subplots_adjust(wspace=0.3)
                
                # SAVE SEPARATE INDIVIDUAL FIGURES (one per grid type)
                if save_individual_figures:
                    for grid_name, grid_data_val in grids_to_plot:
                        # Create separate figure for saving
                        if plot_mode == '2d':
                            save_fig = plt.figure(figsize=individual_figsize)
                            save_ax = save_fig.add_subplot(111)
                            
                            im = create_single_cavity_plot(save_ax, grid_data_val, x_centers, y_centers,
                                                          z_center, grid_name, cavity_centers_xy,
                                                          box_x, box_y)
                            
                            if grid_name == 'ion_density':
                                save_title = f'{species} Ion Density - z={z_center:.1f} Å ({surface_type})'
                                cbar_label = 'Ion Density (ions/Ų)'
                            else:
                                save_title = f'{species} Cavity Occupancy - z={z_center:.1f} Å ({surface_type})'
                                cbar_label = 'Cavity Occupancy'
                            
                            save_ax.set_title(save_title, fontsize=title_fontsize)
                            save_ax.set_xlabel('X (Å)', fontsize=label_fontsize, labelpad=15)
                            save_ax.set_ylabel('Y (Å)', fontsize=label_fontsize, labelpad=15)
                            save_ax.tick_params(labelsize=tick_fontsize)
                            
                            if show_grid:
                                save_ax.grid(alpha=grid_alpha, linestyle='--', linewidth=0.5)
                            
                            if show_colorbar and im is not None:
                                divider = make_axes_locatable(save_ax)
                                cax = divider.append_axes("right", size=colorbar_width, pad=colorbar_pad)
                                cbar = plt.colorbar(im, cax=cax)
                                cbar.set_label(cbar_label, fontsize=colorbar_label_fontsize)
                            
                            plt.tight_layout()
                        
                        else:  # 3D or surface
                            save_fig = plt.figure(figsize=individual_figsize)
                            save_ax = save_fig.add_subplot(111, projection='3d')
                            
                            create_single_cavity_plot(save_ax, grid_data_val, x_centers, y_centers,
                                                    z_center, grid_name, cavity_centers_xy,
                                                    box_x, box_y)
                            
                            if grid_name == 'ion_density':
                                save_title = f'{species} Ion Density - z={z_center:.1f} Å ({surface_type})'
                            else:
                                save_title = f'{species} Cavity Occupancy - z={z_center:.1f} Å ({surface_type})'
                            
                            save_ax.set_title(save_title, fontsize=title_fontsize, pad=20)
                        
                        # Save with specific grid name in filename
                        if grid_name == 'density':
                            filename = f'cavity_{species}_density_z{z_center:.1f}_{plot_mode}.png'
                        else:
                            filename = f'cavity_{species}_occupancy_z{z_center:.1f}_{plot_mode}.png'
                        save_fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                        print(f"     Saved: {filename}")
                        
                        # Close the save figure to avoid display
                        plt.close(save_fig)
                
                # Append display figure to list
                all_figures.append(fig)
                
                # Handle display: show figures individually or close them
                if show_individual_figures:
                    plt.show()
                    # Don't close if showing - let user close manually
                elif not save_individual_figures:
                    # Only close if NOT saving (to prevent memory issues in loops)
                    plt.close(fig)
                # If saving but not showing, keep figures open for notebook display
                # They will be closed at the end or displayed via return value
        
        # OPTIONAL: CREATE COMBINED MULTI-PANEL FIGURE
        if create_combined_figure and len(all_figures) > 0:
            print(f"\n   Creating combined multi-panel figure...")
            
            # Collect all ion types and z-slices for combined view
            combined_data = []
            for ion_idx, ion_type in enumerate(ion_types_to_plot):
                ion_z_data = ion_data[ion_type]
                
                for z_center in z_centers_to_plot:
                    if 'xy_spatial' in ion_z_data[z_center]:
                        combined_data.append({
                            'ion_type': ion_type,
                            'z_center': z_center,
                            'xy_data': ion_z_data[z_center]['xy_spatial']
                        })
            
            if combined_data:
                num_items = len(combined_data)
                
                # Calculate grid layout
                if num_items <= 3:
                    nrows, ncols = 1, num_items
                elif num_items <= 6:
                    nrows, ncols = 2, 3
                elif num_items <= 9:
                    nrows, ncols = 3, 3
                else:
                    nrows = int(np.ceil(np.sqrt(num_items)))
                    ncols = int(np.ceil(num_items / nrows))
                
                # Create combined figure
                if plot_mode == '2d':
                    combined_fig = plt.figure(figsize=(figsize[0]*ncols, figsize[1]*nrows))
                    
                    for idx, data_item in enumerate(combined_data, start=1):
                        ax = combined_fig.add_subplot(nrows, ncols, idx)
                        
                        xy_data = data_item['xy_data']
                        species = data_item['species']
                        z_center = data_item['z_center']
                        
                        # Use appropriate grid based on grid_type
                        if grid_type == 'density':
                            grid_data = xy_data['ion_density_grid']
                            grid_name = 'ion_density'
                        else:  # occupancy or both (use occupancy for combined)
                            grid_data = xy_data['cavity_occupancy_grid']
                            grid_name = 'cavity_occupancy'
                        
                        x_centers = xy_data['x_centers']
                        y_centers = xy_data['y_centers']
                        box_x, box_y = xy_data['box_dimensions']
                        cavity_centers_xy = xy_data.get('cavity_centers_xy', None)
                        
                        # Create plot
                        im = create_single_cavity_plot(ax, grid_data, x_centers, y_centers,
                                                     z_center, grid_name, cavity_centers_xy,
                                                     box_x, box_y)
                        
                        ax.set_title(f'{species} z={z_center:.1f}Å', fontsize=title_fontsize-2)
                        ax.set_xlabel('X (Å)', fontsize=label_fontsize-2)
                        ax.set_ylabel('Y (Å)', fontsize=label_fontsize-2)
                        ax.tick_params(labelsize=tick_fontsize-2)
                    
                    plt.tight_layout()
                
                else:  # 3D or surface mode
                    combined_fig = plt.figure(figsize=(figsize[0]*ncols, figsize[1]*nrows))
                    
                    for idx, data_item in enumerate(combined_data, start=1):
                        ax = combined_fig.add_subplot(nrows, ncols, idx, projection='3d')
                        
                        xy_data = data_item['xy_data']
                        ion_type = data_item['ion_type']
                        z_center = data_item['z_center']
                        
                        # Use appropriate grid
                        if grid_type == 'density':
                            grid_data = xy_data['ion_density_grid']
                            grid_name = 'ion_density'
                        else:
                            grid_data = xy_data['cavity_occupancy_grid']
                            grid_name = 'cavity_occupancy'
                        
                        x_centers = xy_data['x_centers']
                        y_centers = xy_data['y_centers']
                        box_x, box_y = xy_data['box_dimensions']
                        cavity_centers_xy = xy_data.get('cavity_centers_xy', None)
                        
                        # Create 3D plot
                        create_single_cavity_plot(ax, grid_data, x_centers, y_centers,
                                                z_center, grid_name, cavity_centers_xy,
                                                box_x, box_y)
                        
                        ax.set_title(f'{ion_type} z={z_center:.1f}Å', fontsize=title_fontsize-2, pad=10)
                    
                    plt.subplots_adjust(wspace=0.3, hspace=0.3)
                
                # Save combined figure
                if save_combined_figure:
                    combined_filename = f'cavity_occupancy_combined_{plot_mode}_{grid_type}.png'
                    combined_fig.savefig(combined_filename, dpi=dpi, bbox_inches='tight')
                    print(f"     💾 Saved combined: {combined_filename}")
                
                if show_combined_figure:
                    plt.show()
                else:
                    plt.close(combined_fig)
        
        # SUMMARY OUTPUT
        print(f"\n=== Cavity Occupancy XY Spatial Summary ===")
        print(f"  Plot mode: {plot_mode}")
        print(f"  Grid type: {grid_type}")
        print(f"  Species plotted: {', '.join(species_to_plot)}")
        print(f"  Individual figures generated: {len(all_figures)}")
        print(f"  Combined figure: {'Yes' if create_combined_figure else 'No'}")
        print(f"  Clay overlays: Si network={show_si_network}, Mg atoms={show_mgo_atoms}, Si atoms={show_si_atoms}")
        print(f"  Cavity centers overlay: {show_cavity_centers}")
        
        if plot_mode in ['3d', 'surface']:
            print(f"  3D settings:")
            print(f"    - Z-scale factor: {z_scale_factor}")
            print(f"    - View angles: elevation={elevation}, azimuth={azimuth}")
            print(f"    - Buffer (3D): {buffer_3d}")
            if plot_mode == '3d':
                print(f"    - Bar alpha: {bar_alpha}")
                print(f"    - Threshold percentile: {threshold_percentile}")
            else:
                print(f"    - Surface smoothing: {surface_smoothing}")
                print(f"    - Surface alpha: {surface_alpha}")
        
        if si_radial_fade or mgo_radial_fade:
            print(f"  Radial fade: Si={si_radial_fade}, Mg={mgo_radial_fade}")
            print(f"    - Fade range: α={fade_alpha_min} to {fade_alpha_max}")
        
        return all_figures[0] if all_figures else None


# Example usage
if __name__ == "__main__":
    print("ClayOrganicIonWaterAnalysisPlotter class loaded successfully")
    print("\nExample usage:")
    print(">>> from ClayOrganicIonWaterAnalysis import ClayOrganicIonWaterAnalysis")
    print(">>> from ClayOrganicIonWaterAnalysisPlotter import ClayOrganicIonWaterAnalysisPlotter")
    print(">>>")
    print(">>> # Run analysis")
    print(">>> analysis = ClayOrganicIonWaterAnalysis(...)")
    print(">>> analysis.run_full_analysis()")
    print(">>>")
    print(">>> # Create plotter and visualize")
    print(">>> plotter = ClayOrganicIonWaterAnalysisPlotter(analysis)")
    print(">>> plotter.plot_multi_component_rdfs()")
    print(">>> plotter.plot_competitive_adsorption()")
    print(">>> plotter.plot_comprehensive_summary()")

 