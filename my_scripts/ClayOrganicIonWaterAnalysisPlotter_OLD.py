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
        
        # Get available targets and ranges (same logic as ion method)
        first_organic = next(iter(organics_data.values()))
        first_value = next(iter(first_organic.values()))
        if isinstance(first_value, dict):
            sample_key = next(iter(first_value.keys()))
            sample_data = first_value[sample_key]
            has_targets = isinstance(sample_data, dict) and 'mean' in sample_data
        else:
            has_targets = False
        
        # Handle multi-target structure
        if has_targets:
            all_targets = set()
            all_ranges = set()
            
            for organic_data in organics_data.values():
                all_targets.update(organic_data.keys())
                for target_data in organic_data.values():
                    all_ranges.update(target_data.keys())
            
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
            
            # Calculate bar positions
            n_bars_per_organic = n_targets * n_ranges
            
            # Create positions with spacing
            group_spacing = n_bars_per_organic * bar_width * 1.3
            x_positions = np.arange(n_organics) * group_spacing
            
            total_width = bar_width * n_bars_per_organic
            start_offset = -total_width / 2 + bar_width / 2
            
            # Track legend
            legend_added = set()
            
            # Plot bars grouped by organic
            bar_idx = 0
            for target_name in target_names_ordered:
                for range_name in distance_ranges:
                    means = []
                    stds = []
                    
                    for organic_name in organic_names_ordered:
                        if target_name in organics_data[organic_name]:
                            means.append(organics_data[organic_name][target_name][range_name]['mean'])
                            stds.append(organics_data[organic_name][target_name][range_name]['std'])
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
            ax.set_xlabel(xlabel if xlabel != 'Organic-Target Pair' else 'Organic Type',
                         fontweight=label_fontweight, fontsize=label_fontsize)
            ax.set_ylabel(ylabel, fontweight=label_fontweight, fontsize=label_fontsize)
            
            if show_title:
                ax.set_title(title, fontweight=title_fontweight, fontsize=title_fontsize)
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(organic_names_ordered, fontsize=tick_fontsize)
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
        
        else:
            # Simple structure fallback (same as ion method)
            print("Simple structure not yet implemented for organics")
            return None
    
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
                                       transparent_bg=False):
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
        
        # Set up colors
        if colors is None:
            cmap = plt.get_cmap(colormap)
            range_colors = cmap(np.linspace(0, 1, len(distance_ranges)))
            colors = {range_name: range_colors[i] for i, range_name in enumerate(distance_ranges)}
        
        # Create figure
        figsize = figsize or (12, 6)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot time series
        line_idx = 0
        if has_targets:
            for ion_name in ions_data.keys():
                for target_name in ions_data[ion_name].keys():
                    for range_name in distance_ranges:
                        ts = ions_data[ion_name][target_name][range_name]['time_series']
                        frames = np.arange(len(ts))
                        
                        label = custom_labels.get(f"{ion_name}-{target_name}-{range_name}", 
                                                  f'{ion_name}-{target_name}-{range_name}') if custom_labels else f'{ion_name}-{target_name}-{range_name}'
                        
                        linestyle = linestyles[line_idx % len(linestyles)] if linestyles else '-'
                        marker = markers[line_idx % len(markers)] if markers else None
                        
                        ax.plot(frames, ts, label=label, color=colors[range_name],
                               alpha=line_alpha, linewidth=linewidth, linestyle=linestyle,
                               marker=marker, markevery=marker_every)
                        line_idx += 1
        else:
            for ion_name in ions_data.keys():
                for range_name in distance_ranges:
                    ts = ions_data[ion_name][range_name]['time_series']
                    frames = np.arange(len(ts))
                    
                    label = custom_labels.get(f"{ion_name}-{range_name}",
                                             f'{ion_name}-{range_name}') if custom_labels else f'{ion_name}-{range_name}'
                    
                    linestyle = linestyles[line_idx % len(linestyles)] if linestyles else '-'
                    marker = markers[line_idx % len(markers)] if markers else None
                    
                    ax.plot(frames, ts, label=label, color=colors[range_name],
                           alpha=line_alpha, linewidth=linewidth, linestyle=linestyle,
                           marker=marker, markevery=marker_every)
                    line_idx += 1
        
        # Customize plot
        ax.set_xlabel(xlabel, fontweight=label_fontweight, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontweight=label_fontweight, fontsize=label_fontsize)
        
        if show_title:
            ax.set_title(title, fontweight=title_fontweight, fontsize=title_fontsize)
        
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        if show_legend:
            ax.legend(loc=legend_loc, framealpha=legend_framealpha,
                     ncol=legend_ncol, fontsize=legend_fontsize, fontweight=legend_fontweight)
        
        if show_grid:
            ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent_bg)
            print(f"✓ Ion competitive time series plot saved: {filename}")
        
        plt.show()
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
