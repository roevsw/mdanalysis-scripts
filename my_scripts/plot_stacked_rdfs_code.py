    def plot_stacked_rdfs(self,
                         rdf_data: Union[Dict, List[Dict]],
                         vertical_offset: float = 2.0,
                         figsize: Tuple[float, float] = (8, 10),
                         xlim: Optional[Tuple[float, float]] = None,
                         ylim: Optional[Tuple[float, float]] = None,
                         show_title: bool = True,
                         title: str = 'Stacked RDF Comparison',
                         title_fontsize: int = 14,
                         title_fontweight: str = 'bold',
                         xlabel: str = 'r (Å)',
                         ylabel: str = 'g(r) + offset',
                         label_fontsize: int = 12,
                         label_fontweight: str = 'bold',
                         tick_fontsize: int = 10,
                         show_legend: bool = True,
                         legend_fontsize: int = 10,
                         legend_fontweight: str = 'normal',
                         legend_loc: str = 'best',
                         legend_ncol: int = 1,
                         legend_frame_alpha: float = 0.9,
                         legend_frameon: bool = True,
                         colors: Optional[Dict[int, str]] = None,
                         linewidth: float = 2,
                         linestyle: str = '-',
                         alpha: float = 1.0,
                         show_bulk_line: bool = False,
                         bulk_line_color: str = 'gray',
                         bulk_line_style: str = '--',
                         bulk_line_width: float = 1,
                         bulk_line_alpha: float = 0.5,
                         grid: bool = True,
                         grid_alpha: float = 0.3,
                         grid_linestyle: str = '--',
                         shell_boundaries: Optional[Union[List[float], Dict[str, List[float]]]] = None,
                         shell_alpha: float = 0.15,
                         shell_label_fontsize: int = 10,
                         shell_label_style: str = 'complete',
                         shell_label_ha: str = 'center',
                         ion_pairing_gradient: Optional[Union[bool, Dict[str, Union[bool, Dict]]]] = None,
                         ion_pairing_gradient_alpha: float = 0.25,
                         show_cluster_labels: bool = True,
                         cluster_label_fontsize: int = 11,
                         cluster_label_fontweight: str = 'bold',
                         cluster_label_ha: str = 'right',
                         cluster_label_x_position: float = 0.98,
                         save_fig: bool = False,
                         filename: str = 'rdf_stacked.png',
                         dpi: int = 300,
                         bbox_inches: str = 'tight') -> plt.Figure:
        """
        Create XRD-style stacked RDF plots with vertical offset between clusters.
        
        Each cluster's RDF is plotted with its original g(r) values but vertically 
        shifted to prevent overlap, making it easy to compare multiple clusters in
        a single panel.
        
        Parameters
        ----------
        rdf_data : dict or list of dict
            RDF data from analyzer. Can be batch data or single selection data.
            Format: {cluster_id: {'r': array, 'rdf': array, 'selection1': str, ...}}
        
        Stacking control:
        vertical_offset : float, default=2.0
            Vertical spacing between stacked cluster curves
        
        Figure layout:
        figsize : tuple, default=(8, 10)
            Figure size (width, height) in inches
        xlim : tuple, optional
            X-axis limits (r_min, r_max)
        ylim : tuple, optional
            Y-axis limits. If None, auto-calculated from stacked data
        
        Title and labels:
        show_title : bool, default=True
            Show figure title
        title : str, default='Stacked RDF Comparison'
            Figure title
        title_fontsize : int, default=14
            Title font size
        title_fontweight : str, default='bold'
            Title font weight
        xlabel : str, default='r (Å)'
            X-axis label
        ylabel : str, default='g(r) + offset'
            Y-axis label
        label_fontsize : int, default=12
            Axis label font size
        label_fontweight : str, default='bold'
            Axis label font weight
        tick_fontsize : int, default=10
            Tick label font size
        
        Legend:
        show_legend : bool, default=True
            Show legend
        legend_fontsize : int, default=10
            Legend font size
        legend_fontweight : str, default='normal'
            Legend font weight
        legend_loc : str, default='best'
            Legend location
        legend_ncol : int, default=1
            Number of legend columns
        legend_frame_alpha : float, default=0.9
            Legend frame transparency
        legend_frameon : bool, default=True
            Show legend frame
        
        Curve styling:
        colors : dict, optional
            Custom colors for clusters {cluster_id: color}
        linewidth : float, default=2
            Line width
        linestyle : str, default='-'
            Line style
        alpha : float, default=1.0
            Line transparency
        
        Reference lines:
        show_bulk_line : bool, default=False
            Show g(r)=1 reference lines at each offset
        bulk_line_color : str, default='gray'
            Bulk line color
        bulk_line_style : str, default='--'
            Bulk line style
        bulk_line_width : float, default=1
            Bulk line width
        bulk_line_alpha : float, default=0.5
            Bulk line transparency
        
        Grid:
        grid : bool, default=True
            Show grid
        grid_alpha : float, default=0.3
            Grid transparency
        grid_linestyle : str, default='--'
            Grid line style
        
        Shell shading:
        shell_boundaries : list or dict, optional
            Solvation shell boundary positions (same as plot_multiple_rdfs)
        shell_alpha : float, default=0.15
            Shell shading transparency
        shell_label_fontsize : int, default=10
            Shell label font size
        shell_label_style : str, default='complete'
            'complete' or 'short' label style
        shell_label_ha : str, default='center'
            Shell label horizontal alignment
        
        Ion pairing gradient:
        ion_pairing_gradient : bool or dict, optional
            Apply gradient shading (lightcoral → lightyellow → lightgreen → lightblue)
        ion_pairing_gradient_alpha : float, default=0.25
            Gradient transparency
        
        Cluster labels:
        show_cluster_labels : bool, default=True
            Show cluster labels on right side
        cluster_label_fontsize : int, default=11
            Cluster label font size
        cluster_label_fontweight : str, default='bold'
            Cluster label font weight
        cluster_label_ha : str, default='right'
            Cluster label horizontal alignment
        cluster_label_x_position : float, default=0.98
            Cluster label x position (0-1, in axis coordinates)
        
        Save:
        save_fig : bool, default=False
            Save figure
        filename : str, default='rdf_stacked.png'
            Output filename
        dpi : int, default=300
            Resolution
        bbox_inches : str, default='tight'
            Bounding box setting
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        
        Example
        -------
        >>> # Stack surface_o RDFs from multiple clusters
        >>> fig = plotter.plot_stacked_rdfs(
        ...     rdf_surface_o,
        ...     vertical_offset=2.5,
        ...     figsize=(8, 10),
        ...     ion_pairing_gradient=True,
        ...     show_cluster_labels=True,
        ...     save_fig=True
        ... )
        """
        
        # Helper function for parameter resolution
        def resolve_param(param, key):
            if param is None:
                return None
            if isinstance(param, dict):
                return param.get(key, None)
            return param
        
        # Helper function to create gradient colormap
        def create_ion_pairing_gradient_cmap():
            from matplotlib.colors import LinearSegmentedColormap
            colors_list = ['lightcoral', 'lightyellow', 'lightgreen', 'lightblue']
            return LinearSegmentedColormap.from_list('ion_pairing', colors_list, N=256)
        
        # Parse RDF data structure
        cluster_data = {}
        selection_name = None
        reference_name = None
        
        if isinstance(rdf_data, list):
            # Batch RDF data format
            for item in rdf_data:
                cluster_id = item.get('cluster_id')
                if cluster_id is not None:
                    cluster_data[cluster_id] = {
                        'r': item['r'],
                        'rdf': item['rdf'],
                        'n_frames': item.get('n_frames', 'N/A'),
                        'selection1': item.get('selection1', ''),
                        'selection2': item.get('selection2', ''),
                    }
                    if selection_name is None:
                        # Extract selection name from selection1
                        sel1 = item.get('selection1', '')
                        if 'name' in sel1.lower():
                            selection_name = sel1.split()[-1] if sel1.split() else 'Selection'
                        else:
                            selection_name = 'Selection'
                    if reference_name is None:
                        # Extract reference name from selection2
                        sel2 = item.get('selection2', '')
                        if 'name' in sel2.lower() or 'resname' in sel2.lower():
                            parts = sel2.split()
                            if parts:
                                reference_name = parts[-1]
        elif isinstance(rdf_data, dict):
            # Assume format: {cluster_id: {'r': ..., 'rdf': ..., ...}}
            for cluster_id, data in rdf_data.items():
                if isinstance(data, dict) and 'r' in data and 'rdf' in data:
                    cluster_data[cluster_id] = data
                    if selection_name is None and 'selection1' in data:
                        sel1 = data.get('selection1', '')
                        if 'name' in sel1.lower():
                            selection_name = sel1.split()[-1] if sel1.split() else 'Selection'
                    if reference_name is None and 'selection2' in data:
                        sel2 = data.get('selection2', '')
                        if 'name' in sel2.lower() or 'resname' in sel2.lower():
                            parts = sel2.split()
                            if parts:
                                reference_name = parts[-1]
        
        if not cluster_data:
            raise ValueError("No valid RDF data found")
        
        cluster_ids = sorted(cluster_data.keys())
        n_clusters = len(cluster_ids)
        
        # Set default colors
        if colors is None:
            colors = {i: f"C{i}" for i in cluster_ids}
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate pair key for gradient
        pair_key = None
        if selection_name and reference_name:
            pair_key = f"{selection_name}-{reference_name}"
        
        # Check if gradient should be applied
        should_apply_gradient = False
        if ion_pairing_gradient is not None and pair_key:
            gradient_config = resolve_param(ion_pairing_gradient, pair_key)
            if gradient_config is True:
                should_apply_gradient = True
            elif isinstance(gradient_config, dict):
                should_apply_gradient = gradient_config.get('apply', False)
        
        # Apply gradient background if configured and no shell boundaries
        current_shell_boundaries = resolve_param(shell_boundaries, selection_name or 'default')
        if should_apply_gradient and (current_shell_boundaries is None or len(current_shell_boundaries) == 0):
            x_min = xlim[0] if xlim is not None else 0
            x_max = xlim[1] if xlim is not None else 20
            
            # Calculate y extent for gradient
            if ylim is not None:
                y_min, y_max = ylim
            else:
                # Auto-calculate from stacked data
                max_rdf = 0
                for i, cluster_id in enumerate(cluster_ids):
                    offset = i * vertical_offset
                    max_rdf = max(max_rdf, np.max(cluster_data[cluster_id]['rdf']) + offset)
                y_min = 0
                y_max = max_rdf * 1.1
            
            # Create and apply gradient
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            cmap = create_ion_pairing_gradient_cmap()
            extent = [x_min, x_max, y_min, y_max]
            ax.imshow(gradient, aspect='auto', extent=extent, origin='lower',
                     cmap=cmap, alpha=ion_pairing_gradient_alpha, zorder=0,
                     interpolation='bilinear')
        
        # Apply shell shading if configured
        if current_shell_boundaries is not None and len(current_shell_boundaries) > 0:
            n_shells = len(current_shell_boundaries)
            from matplotlib.colors import to_rgba
            
            # Get shell colors
            def get_blue_saturation_colors_from_00c5ff(n_shells):
                base_color = (0, 197/255, 1.0)
                colors_shell = []
                for i in range(n_shells):
                    saturation = 1.0 - (i / n_shells) * 0.7
                    colors_shell.append((base_color[0], base_color[1] * saturation, base_color[2]))
                bulk_sat = max(0.1, 1.0 - ((n_shells) / n_shells) * 0.7)
                bulk_color = (base_color[0], base_color[1] * bulk_sat, base_color[2])
                colors_shell.append(bulk_color)
                return colors_shell
            
            all_colors = get_blue_saturation_colors_from_00c5ff(n_shells)
            shell_colors = all_colors[:-1]
            bulk_color = all_colors[-1]
            
            # Calculate y extent
            if ylim is not None:
                y_min, y_max = ylim
            else:
                max_rdf = 0
                for i, cluster_id in enumerate(cluster_ids):
                    offset = i * vertical_offset
                    max_rdf = max(max_rdf, np.max(cluster_data[cluster_id]['rdf']) + offset)
                y_min = 0
                y_max = max_rdf * 1.1
            
            # Add shell shading
            prev_r = 0.0
            for i, r_max in enumerate(current_shell_boundaries):
                ax.axvspan(prev_r, r_max, alpha=shell_alpha, color=shell_colors[i], zorder=0)
                prev_r = r_max
            
            # Add bulk shading
            bulk_end = xlim[1] if xlim is not None else 20
            ax.axvspan(current_shell_boundaries[-1], bulk_end, alpha=shell_alpha, color=bulk_color, zorder=0)
            
            # Add shell labels
            label_y_pos = y_max * 0.98
            prev_r = 0.0
            for i, r_max in enumerate(current_shell_boundaries):
                if shell_label_ha == 'left':
                    label_x_pos = prev_r
                elif shell_label_ha == 'right':
                    label_x_pos = r_max
                else:
                    label_x_pos = (prev_r + r_max) / 2
                
                label_text = f'S{i+1}' if shell_label_style == 'short' else f'Shell {i+1}'
                ax.text(label_x_pos, label_y_pos, label_text,
                       ha=shell_label_ha, va='top', fontsize=shell_label_fontsize,
                       fontweight='bold', color='black')
                prev_r = r_max
            
            # Bulk label
            if shell_label_ha == 'left':
                bulk_label_x = current_shell_boundaries[-1]
            elif shell_label_ha == 'right':
                bulk_label_x = bulk_end
            else:
                bulk_label_x = (current_shell_boundaries[-1] + bulk_end) / 2
            
            ax.text(bulk_label_x, label_y_pos, 'Bulk',
                   ha=shell_label_ha, va='top', fontsize=shell_label_fontsize,
                   fontweight='bold', color='black')
        
        # Plot stacked RDFs
        for i, cluster_id in enumerate(cluster_ids):
            offset = i * vertical_offset
            data = cluster_data[cluster_id]
            r = data['r']
            g_r = data['rdf']
            n_frames = data.get('n_frames', 'N/A')
            
            color = colors.get(cluster_id, f"C{cluster_id}")
            label = f"Cluster {cluster_id} ({n_frames} frames)"
            
            # Plot with offset
            ax.plot(r, g_r + offset, color=color, linewidth=linewidth,
                   linestyle=linestyle, alpha=alpha, label=label, zorder=2)
            
            # Add bulk reference line if requested
            if show_bulk_line:
                x_max_val = xlim[1] if xlim is not None else np.max(r)
                ax.hlines(1.0 + offset, 0, x_max_val,
                         colors=bulk_line_color, linestyles=bulk_line_style,
                         linewidth=bulk_line_width, alpha=bulk_line_alpha, zorder=1)
            
            # Add cluster label on right side
            if show_cluster_labels:
                # Get y position (middle of the curve at this offset)
                y_label = offset + np.mean(g_r)
                ax.text(cluster_label_x_position, y_label, f"C{cluster_id}",
                       transform=ax.get_yaxis_transform(),
                       fontsize=cluster_label_fontsize,
                       fontweight=cluster_label_fontweight,
                       ha=cluster_label_ha, va='center')
        
        # Set labels and title
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        # Set tick sizes
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # Set axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            # Auto-set ylim based on stacked data
            max_val = max(np.max(cluster_data[cid]['rdf']) + i * vertical_offset 
                         for i, cid in enumerate(cluster_ids))
            ax.set_ylim(0, max_val * 1.1)
        
        # Add grid
        if grid:
            ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
        
        # Add legend
        if show_legend:
            legend = ax.legend(loc=legend_loc, ncol=legend_ncol,
                             fontsize=legend_fontsize,
                             framealpha=legend_frame_alpha,
                             frameon=legend_frameon)
            for text in legend.get_texts():
                text.set_fontweight(legend_fontweight)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            print(f"✓ Saved: {filename} (DPI={dpi})")
        
        return fig
