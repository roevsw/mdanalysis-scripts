
    def plot_coordination_height_surfaces(self, coordination_surface_results=None, ion_types=None,
                                        shell_indices=None,
                                        z_slice_indices=None,
                                        plot_mode='shells',  # 'shells' or 'z_slices'
                                        visualization_mode='3d_surfaces',
                                        surface_style='smooth',
                                        color_scheme='coordination_gradient',
                                        show_colorbar=True,
                                        colorbar_label='Coordination Number',
                                        colorbar_vmin=None,
                                        colorbar_vmax=None,
                                        view_angle=(30, 45),
                                        lighting_style='default',
                                        show_reference_planes=False,
                                        reference_plane_alpha=0.2,
                                        clay_overlay=False,
                                        save_plots=True,
                                        figsize=(15, 12),
                                        dpi=300,
                                        filename_prefix='coordination_height_surfaces',
                                        # Spatial interpolation parameters
                                        fill_missing_data=True,
                                        max_neighbor_cells=3,
                                        interpolation_method='simple_average',
                                        min_neighbors_required=2,
                                        interpolation_max_distance=None,
                                        # Publication settings
                                        title_fontsize=14,
                                        label_fontsize=12,
                                        tick_fontsize=10,
                                        legend_fontsize=12,
                                        colorbar_tick_fontsize=10,
                                        show_title=True,
                                        show_legend=False,
                                        # Publication figure control parameters
                                        save_individual_figures=False,
                                        individual_figsize=(8, 6),
                                        save_combined_figure=False,
                                        show_individual_figures=False,
                                        show_combined_figure=True,
                                        # Colorbar scaling options
                                        colorbar_scaling='individual',
                                        # Surface rendering parameters
                                        surface_alpha=0.8,
                                        surface_cmap='viridis',
                                        surface_lighting=True,
                                        surface_linewidth=0,
                                        surface_antialiased=True,
                                        surface_rstride=1,
                                        surface_cstride=1,
                                        # Surface smoothing parameters
                                        surface_smoothing=False,
                                        surface_smoothing_sigma=1.5,
                                        smoothing_method='gaussian',
                                        grid_upsampling_factor=1,
                                        # Height scaling parameters
                                        height_scale_factor=1.0,
                                        base_height=0.0):
        """
        Plot coordination number surfaces where coordination values are represented as surface height.
        
        This method creates 3D surface plots where the Z-coordinate represents coordination numbers,
        similar to topographical maps but for coordination landscapes. It adapts the infrastructure
        from plot_pmf_height_surfaces() for coordination data.
        
        Parameters
        ----------
        coordination_surface_results : dict, optional
            Results from calculate_solvation_shells_vs_z_detailed() containing coordination_grids.
            If None, attempts to load from analysis results
        ion_types : list, optional
            Ion types to plot. If None, plots all available
        shell_indices : list, optional
            Indices of shells to plot (0=first, 1=second, 2=third). If None, plots all shells
        z_slice_indices : list, optional
            Indices of Z-slices to plot. If None, plots all Z-slices
        plot_mode : str, default='shells'
            Plotting strategy:
            - 'shells': Plot different shells across Z-layers for each ion (RECOMMENDED)
            - 'z_slices': Plot different Z-layers across shells for each ion
        visualization_mode : str, default='3d_surfaces'
            Plotting mode:
            - '3d_surfaces': 3D surface plots
            - '2d_contours': 2D contour projections
            - 'combined': Both 3D and 2D views
        surface_style : str, default='smooth'
            Surface rendering style:
            - 'smooth': Smooth continuous surfaces
            - 'wireframe': Wireframe surfaces
            - 'filled_contour': Filled contour surfaces
        color_scheme : str, default='coordination_gradient'
            Coloring scheme:
            - 'coordination_gradient': Color based on coordination values (RECOMMENDED)
            - 'height_gradient': Color based on surface height values
            - 'adaptive': Use solid colors (single color per surface)
        show_colorbar : bool, default=True
            Show colorbar for coordination/height values
        colorbar_label : str, default='Coordination Number'
            Label for the colorbar
        colorbar_vmin, colorbar_vmax : float, optional
            Colorbar range limits
        view_angle : tuple, default=(30, 45)
            3D viewing angle (elevation, azimuth)
        lighting_style : str, default='default'
            3D lighting setup
        show_reference_planes : bool, default=False
            Show reference planes (z=0, bulk level)
        reference_plane_alpha : float, default=0.2
            Transparency of reference planes
        clay_overlay : bool, default=False
            Overlay clay structure if available
        save_plots : bool, default=True
            Save plots to files
        figsize : tuple, default=(15, 12)
            Figure size
        dpi : int, default=300
            Resolution for saved plots
        filename_prefix : str, default='coordination_height_surfaces'
            Prefix for saved filenames
        fill_missing_data : bool, default=True
            Fill missing/infinite coordination data using spatial interpolation
        max_neighbor_cells : int, default=3
            Maximum number of neighbor cells to consider for averaging
        interpolation_method : str, default='simple_average'
            Method for spatial interpolation:
            - 'simple_average': Simple mean of valid neighbors
            - 'distance_weighted': Distance-weighted averaging
            - 'gaussian_kernel': Gaussian kernel averaging
        min_neighbors_required : int, default=2
            Minimum number of valid neighbors required for interpolation
        interpolation_max_distance : int, optional
            Maximum distance (in grid cells) for neighbor search.
            If None, uses max_neighbor_cells as radius
        colorbar_scaling : str, default='individual'
            Colorbar scaling strategy:
            - 'individual': Each surface gets its own colorbar range
            - 'global': All surfaces per ion use same range
            - 'unified': All ions and surfaces use same global range
        surface_smoothing : bool, default=False
            Whether to apply smoothing to surfaces for better visual appearance
        surface_smoothing_sigma : float, default=1.5
            Gaussian smoothing parameter (higher = more smoothing)
        smoothing_method : str, default='gaussian'
            Smoothing method: 'gaussian', 'median', 'bilateral'
        grid_upsampling_factor : int, default=1
            Factor to increase grid resolution through interpolation before smoothing
        height_scale_factor : float, default=1.0
            Scaling factor for coordination values when converting to height
        base_height : float, default=0.0
            Base height for coordination surfaces
        
        Publication Font Settings:
        title_fontsize : int, default=14
        label_fontsize : int, default=12
        tick_fontsize : int, default=10
        legend_fontsize : int, default=12
        colorbar_tick_fontsize : int, default=10
        show_title : bool, default=True
        show_legend : bool, default=False
        
        Publication Figure Control:
        save_individual_figures : bool, default=False
        individual_figsize : tuple, default=(8, 6)
        save_combined_figure : bool, default=False
        show_individual_figures : bool, default=False
        show_combined_figure : bool, default=True
        
        Surface Rendering Options:
        surface_alpha : float, default=0.8
        surface_cmap : str, default='viridis'
        surface_lighting : bool, default=True
        surface_linewidth : float, default=0
        surface_antialiased : bool, default=True
        surface_rstride, surface_cstride : int, default=1
        
        Returns
        -------
        dict
            Dictionary containing plot objects and metadata
        """
        
        from scipy.ndimage import gaussian_filter, median_filter
        from scipy.interpolate import interp2d
        
        print(f"\n🧭 PLOTTING COORDINATION HEIGHT SURFACES")
        print(f"{'='*60}")
        print(f"Plot mode: {plot_mode}")
        print(f"Visualization mode: {visualization_mode}")
        print(f"Surface style: {surface_style}")
        print(f"Color scheme: {color_scheme}")
        print(f"Colorbar scaling: {colorbar_scaling}")
        if surface_smoothing:
            print(f"Surface smoothing: {smoothing_method} (σ={surface_smoothing_sigma}, upsampling={grid_upsampling_factor}x)")
        
        # Get coordination HEIGHT SURFACE data from analysis results
        if coordination_surface_results is None:
            # First, check if we have height surface data from create_coordination_height_surfaces()
            if (hasattr(self.analysis.results, 'coordination_height_surfaces') and 
                self.analysis.results.coordination_height_surfaces is not None):
                height_surface_data = self.analysis.results.coordination_height_surfaces
                print(f"✓ Found coordination height surface data in: analysis.results.coordination_height_surfaces")
                
                # Use the height surface data directly - this is what we want!
                coordination_surface_results = height_surface_data
            
            # Fallback: If no height surfaces, try to find raw coordination data and create height surfaces
            elif (hasattr(self.analysis.results, 'solvation_shells_xy_spatial') and 
                  self.analysis.results.solvation_shells_xy_spatial is not None):
                print("⚠️ No height surfaces found, but found raw coordination data")
                print("💡 You should run create_coordination_height_surfaces() first to convert coordination numbers to heights")
                
                spatial_data = self.analysis.results.solvation_shells_xy_spatial
                print(f"✓ Found spatial coordination data in: analysis.results.solvation_shells_xy_spatial")
                
                # Convert z_slices format to layer_data format expected by plotting
                if 'z_slices' in spatial_data:
                    layer_data = {}
                    for layer in spatial_data['z_slices']:
                        z_center = layer['z_center']
                        layer_data[z_center] = layer
                    coordination_surface_results = {'layer_data': layer_data}
                    print(f"✓ Converted {len(spatial_data['z_slices'])} z_slices to layer_data format")
                else:
                    coordination_surface_results = spatial_data
            
            # Fallback: Try other possible attribute names for coordination data
            else:
                possible_attributes = [
                    'spatial_analysis_detailed',
                    'spatial_analysis_results',
                    'solvation_shells_detailed',
                    'coordination_analysis',
                    'layer_analysis',
                    'z_layer_analysis'
                ]
                
                coordination_surface_results = None
                for attr_name in possible_attributes:
                    if hasattr(self.analysis.results, attr_name):
                        attr_value = getattr(self.analysis.results, attr_name)
                        if attr_value and isinstance(attr_value, dict) and 'layer_data' in attr_value:
                            coordination_surface_results = attr_value
                            print(f"✓ Found coordination data in: analysis.results.{attr_name}")
                            break
            
            if coordination_surface_results is None:
                # Print available results to help debug
                print("❌ No coordination surface results found.")
                print("🔍 DEBUGGING: Available analysis.results attributes:")
                
                found_data = False
                if hasattr(self.analysis, 'results'):
                    results_attrs = [attr for attr in dir(self.analysis.results) if not attr.startswith('_')]
                    for attr in sorted(results_attrs):
                        value = getattr(self.analysis.results, attr)
                        if value is not None:
                            print(f"  ✓ {attr}: {type(value)}")
                            # Check if this might be coordination data
                            if isinstance(value, dict):
                                if 'layer_data' in value:
                                    print(f"    🎯 Contains 'layer_data' - this IS coordination data!")
                                    coordination_surface_results = value
                                    found_data = True
                                    break
                                elif any(key in str(value).lower() for key in ['layer', 'coord', 'shell']):
                                    print(f"    💡 Might contain coordination data - checking keys: {list(value.keys())}")
                        else:
                            print(f"  ✗ {attr}: None")
                    
                    if not found_data:
                        print(f"\n🔍 MANUAL SEARCH: Looking for any dict with coordination-like data...")
                        for attr in sorted(results_attrs):
                            value = getattr(self.analysis.results, attr)
                            if isinstance(value, dict) and value:
                                keys = list(value.keys())
                                if any('layer' in str(k).lower() or 'coord' in str(k).lower() or 'shell' in str(k).lower() or 'z' in str(k).lower() for k in keys):
                                    print(f"  🤔 {attr} has interesting keys: {keys}")
                                    # Try to access layer-like data
                                    for key in keys:
                                        if 'layer' in str(key).lower() and isinstance(value[key], dict):
                                            print(f"    🔍 {attr}['{key}'] contains: {list(value[key].keys())}")
                                            if 'coordination_grids' in value[key] or any('coord' in str(k).lower() for k in value[key].keys()):
                                                print(f"    ✅ Found coordination data in {attr}['{key}']!")
                                                coordination_surface_results = {'layer_data': value[key]}
                                                found_data = True
                                                break
                                    if found_data:
                                        break
                else:
                    print("  ✗ No analysis.results found")
                
                if coordination_surface_results is None:
                    print(f"\n💡 SOLUTION OPTIONS:")
                    print(f"1. Run the debug method first: plotter.debug_coordination_data_availability()")
                    print(f"2. Pass data directly: coordination_surface_results=your_coordination_data")
                    print(f"3. Re-run: analysis.calculate_solvation_shells_vs_z_detailed()")
                    raise ValueError("No coordination surface results found. Run calculate_solvation_shells_vs_z_detailed() first and ensure the results are stored in analysis.results.")
        
        # Validate data structure - check if we have height surfaces (Step 2) or raw data (Step 1)
        if 'height_surfaces' in coordination_surface_results:
            # We have HEIGHT SURFACE data from create_coordination_height_surfaces() - STEP 2
            print("✅ Using height surface data from create_coordination_height_surfaces() (Step 2)")
            height_surfaces = coordination_surface_results['height_surfaces']
            
            # Extract available ions from height surface data
            available_ions = list(height_surfaces.keys())
            print(f"Available ions from height surfaces: {available_ions}")
            
            # For height surfaces, we don't need to process raw coordination grids
            layer_data = None
            
        elif 'layer_data' in coordination_surface_results:
            # We have raw coordination data from calculate_solvation_shells_vs_z_detailed() - STEP 1
            print("⚠️ Using raw coordination data - you should run create_coordination_height_surfaces() first")
            layer_data = coordination_surface_results['layer_data']
        
        else:
            raise ValueError("Invalid coordination surface results. Must contain 'height_surfaces' (from Step 2) or 'layer_data' (from Step 1).")
        
        # Only process raw coordination data if we don't have height surfaces
        if layer_data is not None:
            # Get available ions and shells from coordination_grids
            available_ions = []
            available_shells = []
            shell_names = ['first_shell', 'second_shell', 'third_shell']
            
            for z_layer, z_data in layer_data.items():
                if 'coordination_grids' in z_data:
                    coord_grids = z_data['coordination_grids']
                    for shell_type in shell_names:
                        if shell_type in coord_grids:
                            available_shells.append(shell_type)
                            for ion_type in coord_grids[shell_type]:
                                if ion_type not in available_ions:
                                    available_ions.append(ion_type)
            
            available_shells = list(set(available_shells))  # Remove duplicates
            
            if not available_ions:
                raise ValueError("No coordination data found in results")
            
            print(f"Available ions: {available_ions}")
            print(f"Available shells: {available_shells}")
            print(f"Available Z-layers: {list(layer_data.keys())}")
        
        # Get ion types to plot
        if ion_types is None:
            ion_types = available_ions
        elif isinstance(ion_types, str):
            ion_types = [ion_types]
        
        # Validate ion types
        invalid_ions = [ion for ion in ion_types if ion not in available_ions]
        if invalid_ions:
            print(f"⚠ Warning: Ion types {invalid_ions} not found in results")
        ion_types = [ion for ion in ion_types if ion in available_ions]
        
        if not ion_types:
            raise ValueError("No valid ion types to plot")
        
        print(f"Plotting ion types: {ion_types}")
        
        # ===== HANDLE HEIGHT SURFACES (STEP 2 DATA) =====
        if layer_data is None:  # We have height surface data from Step 2
            print("🎯 Using pre-computed height surfaces from create_coordination_height_surfaces()")
            
            # Get coordinate data from height surface results (stored at top level)
            x_centers = coordination_surface_results['x_centers']
            y_centers = coordination_surface_results['y_centers']
            
            # Transform coordinates to start from 0 (same as PMF method)
            x_display = x_centers - x_centers.min()
            y_display = y_centers - y_centers.min()
            
            print(f"Coordinate range: X=[0, {x_display[-1]:.1f}], Y=[0, {y_display[-1]:.1f}]")
            
            # Create coordinate meshgrid for surfaces
            X_display, Y_display = np.meshgrid(x_display, y_display)
            
            # Extract surfaces to plot from height surface data
            first_ion = list(height_surfaces.keys())[0]
            surfaces_to_plot = list(height_surfaces[first_ion]['surfaces'].keys())
            print(f"Available surfaces to plot: {surfaces_to_plot}")
            
            # Set plot dimension based on surface names
            if any('shell' in str(surface) for surface in surfaces_to_plot):
                plot_dimension = 'shells'
                available_shells = surfaces_to_plot  # Shell names from height surface data
                surface_labels = surfaces_to_plot  # Use surface names as labels
                print(f"Detected shell-based surfaces: {plot_dimension}")
            else:
                plot_dimension = 'z_layers'
                available_shells = []  # No shells for z-layer mode
                # For numeric z-layers, format as "z=X.X Å"
                surface_labels = [f"z={z:.1f} Å" if isinstance(z, (int, float)) else str(z) for z in surfaces_to_plot]
                print(f"Detected z-layer based surfaces: {plot_dimension}")
            
            # Use existing height surface data
            print(f"Available height surface data for ions: {list(height_surfaces.keys())}")
            
        # ===== PROCESS RAW COORDINATION DATA (STEP 1 DATA) =====
        else:  # We have raw coordination data - process it
            print("⚙️ Processing raw coordination data to create height surfaces...")
            
            # Get shells and Z-slices to plot based on mode
            available_shells = []
            shell_names = ['first_shell', 'second_shell', 'third_shell']
            
            for z_layer, z_data in layer_data.items():
                if 'coordination_grids' in z_data:
                    coord_grids = z_data['coordination_grids']
                    for shell_type in shell_names:
                        if shell_type in coord_grids:
                            available_shells.append(shell_type)
            
            available_shells = list(set(available_shells))  # Remove duplicates
            
            if plot_mode == 'shells':
                # Plot different shells - each surface represents a shell across Z-layers
                if shell_indices is None:
                    surfaces_to_plot = available_shells
                    surface_labels = available_shells
                else:
                    surfaces_to_plot = [available_shells[i] for i in shell_indices if i < len(available_shells)]
                    surface_labels = surfaces_to_plot
                
                plot_dimension = 'shells'
                print(f"Shell-focused mode: plotting {len(surfaces_to_plot)} shells: {surfaces_to_plot}")
                
            elif plot_mode == 'z_slices':
                # Plot different Z-slices - each surface represents a Z-layer across shells
                z_layers = list(layer_data.keys())
                if z_slice_indices is None:
                    surfaces_to_plot = z_layers
                    surface_labels = [f"z={z:.1f}" for z in z_layers]
                else:
                    surfaces_to_plot = [z_layers[i] for i in z_slice_indices if i < len(z_layers)]
                    surface_labels = [f"z={z:.1f}" for z in surfaces_to_plot]
                
                plot_dimension = 'z_layers'
                print(f"Z-slice-focused mode: plotting {len(surfaces_to_plot)} Z-layers: {surface_labels}")
            
            else:
                raise ValueError(f"Invalid plot_mode: {plot_mode}. Use 'shells' or 'z_slices'")
            
            # Get coordinate arrays from the first available layer
            first_layer = list(layer_data.keys())[0]
            first_layer_data = layer_data[first_layer]
            
            if 'x_centers' not in first_layer_data or 'y_centers' not in first_layer_data:
                raise ValueError("Coordinate information (x_centers, y_centers) not found in layer data")
            
            x_centers = first_layer_data['x_centers']
            y_centers = first_layer_data['y_centers']
            
            # Transform coordinates to start from 0 (SAME AS PMF METHOD)
            x_display = x_centers - x_centers.min()  # Start X from 0
            y_display = y_centers - y_centers.min()  # Start Y from 0
            
            print(f"Coordinate transformation: X=[0, {x_display[-1]:.1f}], Y=[0, {y_display[-1]:.1f}]")
            
            # Create coordinate meshgrid for surfaces
            X_display, Y_display = np.meshgrid(x_display, y_display)
            
            # Convert coordination data to height surface format
            print(f"\n📊 Converting coordination data to height surface format...")
            
            height_surfaces = {}
            
            # Define colors for different ions (reuse from existing method)
            ion_colors = {
                'NA': '#1f77b4',  # Blue
                'MG': '#ff7f0e',  # Orange  
                'CA': '#2ca02c',  # Green
                'K': '#d62728',   # Red
                'LI': '#9467bd',  # Purple
                'RB': '#8c564b',  # Brown
                'SR': '#e377c2',  # Pink
                'CL': '#7f7f7f',  # Gray
                'BR': '#bcbd22',  # Olive
                'F': '#17becf'    # Cyan
            }
            
            for ion_type in ion_types:
                print(f"  Processing {ion_type}...")
                
                ion_data = {
                    'surfaces': {},
                    'color': ion_colors.get(ion_type, '#1f77b4'),
                    'alpha': 0.8
                }
                
                for i, surface_key in enumerate(surfaces_to_plot):
                    print(f"    Creating surface for {surface_key}...")
                    
                    # Initialize coordination grid for this surface
                    coord_surface = np.full(X_display.shape, np.nan)
                    
                    if plot_mode == 'shells':
                        # Aggregate across Z-layers for this shell
                        shell_type = surface_key
                        coord_values = []
                        
                        for z_layer, z_data in layer_data.items():
                            if ('coordination_grids' in z_data and 
                                shell_type in z_data['coordination_grids'] and
                                ion_type in z_data['coordination_grids'][shell_type]):
                                
                                shell_grid = z_data['coordination_grids'][shell_type][ion_type]
                                coord_values.append(shell_grid)
                        
                        if coord_values:
                            # Average across Z-layers (you could also use max, median, etc.)
                            coord_surface = np.mean(coord_values, axis=0)
                            surface_label = f"{shell_type.replace('_', ' ').title()}"
                        else:
                            print(f"      ⚠ No data found for {ion_type} {shell_type}")
                            continue
                            
                    elif plot_mode == 'z_slices':
                        # Aggregate across shells for this Z-layer
                        z_layer = surface_key
                        z_data = layer_data[z_layer]
                        
                        if 'coordination_grids' in z_data:
                            coord_grids = z_data['coordination_grids']
                            coord_values = []
                            
                            for shell_type in available_shells:
                                if (shell_type in coord_grids and 
                                    ion_type in coord_grids[shell_type]):
                                    
                                    shell_grid = coord_grids[shell_type][ion_type]
                                    coord_values.append(shell_grid)
                            
                            if coord_values:
                                # Sum across shells for total coordination
                                coord_surface = np.sum(coord_values, axis=0)
                                surface_label = f"z = {z_layer:.1f} Å"
                            else:
                                print(f"      ⚠ No data found for {ion_type} at z = {z_layer}")
                                continue
                        else:
                            print(f"      ⚠ No coordination_grids found for z = {z_layer}")
                            continue
                    
                    # Convert coordination values to height surface
                    coord_height_surface = base_height + coord_surface * height_scale_factor
                    
                    # Store surface data in PMF-like format
                    surface_info = {
                        'surface_data': {
                            'X': X_display,
                            'Y': Y_display, 
                            'Z': coord_height_surface
                        },
                        'pmf_grid': coord_surface,  # Store original coordination data as 'pmf_grid' for compatibility
                        'base_z': base_height,
                        'original_z': surface_key if isinstance(surface_key, (int, float)) else i,
                        'surface_type': surface_key,
                        'surface_label': surface_label
                    }
                    
                    ion_data['surfaces'][surface_key] = surface_info
                    print(f"      ✓ Surface created: coord range = {np.nanmin(coord_surface):.2f} to {np.nanmax(coord_surface):.2f}")
                
                height_surfaces[ion_type] = ion_data
                print(f"    ✓ {ion_type}: {len(ion_data['surfaces'])} surfaces created")
            
            # Create the height surface results structure (compatible with PMF method)
            height_surface_results = {
                'height_surfaces': height_surfaces,
                'x_centers': x_centers,
                'y_centers': y_centers,
                'z_scale_factor': height_scale_factor,
                'base_height': base_height,
                'plot_mode': plot_mode,
                'coordination_metadata': {
                    'available_shells': available_shells,
                    'z_layers': list(layer_data.keys()) if layer_data is not None else [],
                    'surfaces_plotted': surfaces_to_plot
                }
            }
        
        # ===== COMMON PLOTTING SECTION (BOTH HEIGHT SURFACES AND RAW DATA) =====
        # At this point, we have height_surfaces data regardless of whether it came from Step 2 or was processed from raw data
        
        # Prepare final height_surface_results for plotting
        if layer_data is None:  # Height surface data from Step 2
            # Extract metadata from existing height surfaces
            first_ion = list(height_surfaces.keys())[0]
            x_centers = coordination_surface_results['x_centers']  
            y_centers = coordination_surface_results['y_centers']
            
            height_surface_results = {
                'height_surfaces': height_surfaces,
                'x_centers': x_centers,
                'y_centers': y_centers,
                'z_scale_factor': height_scale_factor,
                'base_height': base_height,
                'plot_mode': plot_mode,
                'coordination_metadata': {
                    'surfaces_plotted': list(height_surfaces[first_ion]['surfaces'].keys())
                }
            }
        else:  # Raw data was processed above
            # height_surface_results already created above
            pass
        
        print(f"✅ Data conversion complete!")
        
        # Now call the adapted PMF plotting infrastructure with coordination-specific parameters
        print(f"\n🎨 Rendering coordination height surfaces...")
        
        # Adapt the PMF method parameters for coordination plotting
        adapted_params = {
            'height_surface_results': height_surface_results,
            'ion_types': ion_types,
            'surface_indices': None,  # Plot all converted surfaces
            'visualization_mode': visualization_mode,
            'surface_style': surface_style,
            'color_scheme': color_scheme.replace('coordination_gradient', 'pmf_gradient'),  # Map to PMF method
            'show_colorbar': show_colorbar,
            'colorbar_label': colorbar_label,
            'colorbar_vmin': colorbar_vmin,
            'colorbar_vmax': colorbar_vmax,
            'view_angle': view_angle,
            'lighting_style': lighting_style,
            'show_reference_planes': show_reference_planes,
            'reference_plane_alpha': reference_plane_alpha,
            'clay_overlay': clay_overlay,
            'save_plots': save_plots,
            'figsize': figsize,
            'dpi': dpi,
            'filename_prefix': f"{filename_prefix}_{plot_mode}",
            # Spatial interpolation parameters
            'fill_missing_data': fill_missing_data,
            'max_neighbor_cells': max_neighbor_cells,
            'interpolation_method': interpolation_method,
            'min_neighbors_required': min_neighbors_required,
            'interpolation_max_distance': interpolation_max_distance,
            # Publication settings
            'title_fontsize': title_fontsize,
            'label_fontsize': label_fontsize,
            'tick_fontsize': tick_fontsize,
            'legend_fontsize': legend_fontsize,
            'colorbar_tick_fontsize': colorbar_tick_fontsize,
            'show_title': show_title,
            'show_legend': show_legend,
            # Publication figure control
            'save_individual_figures': save_individual_figures,
            'individual_figsize': individual_figsize,
            'save_combined_figure': save_combined_figure,
            'show_individual_figures': show_individual_figures,
            'show_combined_figure': show_combined_figure,
            # Colorbar scaling
            'colorbar_scaling': colorbar_scaling,
            # Surface rendering
            'surface_alpha': surface_alpha,
            'surface_cmap': surface_cmap,
            'surface_lighting': surface_lighting,
            'surface_linewidth': surface_linewidth,
            'surface_antialiased': surface_antialiased,
            'surface_rstride': surface_rstride,
            'surface_cstride': surface_cstride,
            # Surface smoothing
            'surface_smoothing': surface_smoothing,
            'surface_smoothing_sigma': surface_smoothing_sigma,
            'smoothing_method': smoothing_method,
            'grid_upsampling_factor': grid_upsampling_factor
        }
        
        # Use the existing PMF method infrastructure but with coordination data
        plot_results = self.plot_pmf_height_surfaces(**adapted_params)
        
        # Add coordination-specific metadata to results
        plot_results['coordination_metadata'] = {
            'plot_mode': plot_mode,
            'surfaces_plotted': surfaces_to_plot,
            'surface_labels': surface_labels if 'surface_labels' in locals() else surfaces_to_plot,
            'plot_dimension': plot_dimension,
            'height_scale_factor': height_scale_factor,
            'base_height': base_height,
            'available_shells': available_shells,
            'z_layers': list(layer_data.keys()) if layer_data is not None else []
        }
        
        # Update title information for coordination context
        for ax in plot_results['axes']:
            if show_title:
                current_title = ax.get_title()
                # Replace "PMF" with "Coordination" in titles
                coordination_title = current_title.replace('PMF Height Surfaces', f'Coordination Height Surfaces ({plot_mode.title()})')
                ax.set_title(coordination_title, fontsize=title_fontsize, fontweight='bold')
        
        # Print coordination-specific summary
        print(f"\n📊 Coordination Height Surface Plotting Summary:")
        print(f"  Plot mode: {plot_mode} ({plot_dimension})")
        print(f"  Ion types: {len(ion_types)} ({ion_types})")
        print(f"  Surfaces per ion: {len(surfaces_to_plot)}")
        print(f"  Total surfaces plotted: {sum(len(surfaces) for surfaces in plot_results['surfaces'].values())}")
        print(f"  Height scaling: {height_scale_factor}x + {base_height}")
        print(f"  Coordinate range: X=[0, {x_display[-1]:.1f}], Y=[0, {y_display[-1]:.1f}]")
        
        if plot_mode == 'shells':
            print(f"  💡 Shell-focused: Each surface shows a coordination shell aggregated across Z-layers")
        elif plot_mode == 'z_slices':
            print(f"  💡 Z-slice-focused: Each surface shows total coordination at a Z-layer across all shells")
        
        return plot_results


    def debug_coordination_data_availability(self):
        """
        Debug helper to check what coordination data is available in analysis results
        """
        print("🔍 DEBUGGING COORDINATION DATA AVAILABILITY")
        print("="*60)
        
        if not hasattr(self, 'analysis'):
            print("❌ No analysis object found in plotter")
            return None
        
        if not hasattr(self.analysis, 'results'):
            print("❌ No analysis.results found")
            return None
        
        print("✓ Analysis results found")
        
        # List all available results
        results_attrs = dir(self.analysis.results)
        results_attrs = [attr for attr in results_attrs if not attr.startswith('_')]
        print(f"\n📋 Available results attributes ({len(results_attrs)}):")
        
        coordination_candidates = []
        
        for attr in sorted(results_attrs):
            value = getattr(self.analysis.results, attr)
            if value is not None:
                print(f"  ✓ {attr}: {type(value)}")
                
                # Check if this might be coordination data
                if isinstance(value, dict):
                    if 'layer_data' in value:
                        print(f"    🎯 Contains 'layer_data' - LIKELY COORDINATION DATA!")
                        coordination_candidates.append(attr)
                        
                        # Examine layer_data structure
                        layer_data = value['layer_data']
                        print(f"    📊 Layer data keys: {list(layer_data.keys())}")
                        
                        # Check first layer
                        if layer_data:
                            first_key = list(layer_data.keys())[0]
                            first_layer = layer_data[first_key]
                            print(f"    🔍 First layer ({first_key}) keys: {list(first_layer.keys())}")
                            
                            # Check coordination_grids
                            if 'coordination_grids' in first_layer:
                                coord_grids = first_layer['coordination_grids']
                                print(f"    ✅ Coordination grids found:")
                                for shell_type in coord_grids:
                                    ions = list(coord_grids[shell_type].keys())
                                    print(f"      {shell_type}: {ions}")
                            else:
                                print(f"    ⚠️ No 'coordination_grids' in first layer")
                    
                    elif any(key in str(value).lower() for key in ['coord', 'shell', 'spatial']):
                        print(f"    🤔 Might contain coordination data - checking...")
            else:
                print(f"  ✗ {attr}: None")
        
        print(f"\n🎯 SUMMARY:")
        if coordination_candidates:
            print(f"✅ Found {len(coordination_candidates)} potential coordination data sources:")
            for candidate in coordination_candidates:
                print(f"  - analysis.results.{candidate}")
            print(f"\n💡 Try using: coordination_surface_results=analysis.results.{coordination_candidates[0]}")
            return coordination_candidates[0]
        else:
            print("❌ No coordination data found")
            print("\n🔧 Troubleshooting:")
            print("1. Make sure calculate_solvation_shells_vs_z_detailed() completed successfully")
            print("2. Check that the results were saved to analysis.results")
            print("3. Look for any error messages during calculation")
            return None


    def plot_pmf_with_bootstrap_comparison(self, pmf_results, bootstrap_results, 
                                        comparison_type='original_vs_bootstrap',
                                        show_individual_curves=False,
                                        n_bootstrap_curves=20,
                                        units='kT'):
        """
        Create comparison plots showing original PMF vs bootstrap mean with uncertainty.
        """
        
        print(f"\n📊 Creating PMF Bootstrap Comparison ({comparison_type})")
        
        ion_types = list(pmf_results.keys())
        n_ions = len(ion_types)
        
        # Create figure with proper subplot arrangement
        fig, axes = plt.subplots(n_ions, 2, figsize=(15, 6*n_ions))
        
        # Handle axes indexing correctly for single ion case
        if n_ions == 1:
            axes = axes.reshape(1, -1)
        
        for idx, ion_type in enumerate(ion_types):
            # Get the correct axes for this ion
            if n_ions == 1:
                ax1 = axes[0, 0]
                ax2 = axes[0, 1]
            else:
                ax1 = axes[idx, 0]
                ax2 = axes[idx, 1]
            
            # Check if we have PMF data for this ion
            if ion_type not in pmf_results:
                ax1.text(0.5, 0.5, f'No PMF data for {ion_type}', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax2.text(0.5, 0.5, f'No PMF data for {ion_type}', 
                        ha='center', va='center', transform=ax2.transAxes)
                continue
            
            # Get original PMF data
            pmf_data = pmf_results[ion_type]
            original_z = pmf_data['z_centers']
            original_pmf = pmf_data['pmf']
            
            print(f"   📈 {ion_type} - Original PMF: z range {original_z.min():.1f} to {original_z.max():.1f}, PMF range {original_pmf.min():.3f} to {original_pmf.max():.3f}")
            
            # Bootstrap data with key mapping
            if ion_type in bootstrap_results:
                boot_data = bootstrap_results[ion_type]
                
                print(f"   🔧 {ion_type} - Available bootstrap keys: {list(boot_data.keys())}")
                
                # Get z-coordinates (try both possible keys)
                boot_z = boot_data.get('z_centers', original_z)  # Fallback to original z if not found
                
                # Try mapped keys first, then original keys
                boot_mean = (boot_data.get('bootstrap_mean') or 
                            boot_data.get('mean_pmf'))
                
                boot_std = (boot_data.get('bootstrap_std') or 
                        boot_data.get('pmf_std'))
                
                # Get confidence intervals
                ci_data = boot_data.get('confidence_intervals', {})
                ci_95_lower = None
                ci_95_upper = None
                
                if '95%' in ci_data:
                    ci_95_lower = ci_data['95%']['lower']
                    ci_95_upper = ci_data['95%']['upper']
                elif boot_mean is not None and boot_std is not None:
                    # Calculate 95% CI from mean ± 1.96*std
                    ci_95_lower = boot_mean - 1.96 * boot_std
                    ci_95_upper = boot_mean + 1.96 * boot_std
                
                if boot_mean is None or boot_std is None:
                    print(f"   ❌ Missing bootstrap statistics for {ion_type}")
                    print(f"      Available keys: {list(boot_data.keys())}")
                    ax1.text(0.5, 0.5, f'Invalid bootstrap data for {ion_type}', 
                            ha='center', va='center', transform=ax1.transAxes)
                    ax2.text(0.5, 0.5, f'Invalid bootstrap data for {ion_type}', 
                            ha='center', va='center', transform=ax2.transAxes)
                    continue
                
                print(f"   📊 {ion_type} - Bootstrap: z shape {boot_z.shape}, mean shape {boot_mean.shape}")
                
                # FIXED: Plot original PMF first (make sure it's visible)
                ax1.plot(original_z, original_pmf, 'b-', linewidth=3, 
                        label='Original PMF', alpha=0.8, zorder=3)
                
                # FIXED: Plot bootstrap mean with different color and style
                ax1.plot(boot_z, boot_mean, 'r-', linewidth=2.5, 
                        label='Bootstrap Mean', alpha=0.9, zorder=2)
                
                # FIXED: Plot 95% confidence band
                if ci_95_lower is not None and ci_95_upper is not None:
                    ax1.fill_between(boot_z, ci_95_lower, ci_95_upper, alpha=0.3, 
                                color='red', label='95% CI', zorder=1)
                
                # Show individual bootstrap curves if requested
                if show_individual_curves:
                    bootstrap_pmfs = (boot_data.get('individual_pmfs') or 
                                    boot_data.get('bootstrap_pmfs'))
                    
                    if bootstrap_pmfs is not None:
                        n_curves = min(n_bootstrap_curves, bootstrap_pmfs.shape[0])
                        curve_indices = np.random.choice(bootstrap_pmfs.shape[0], 
                                                    n_curves, replace=False)
                        
                        for i, curve_idx in enumerate(curve_indices):
                            alpha = 0.1 if n_curves > 10 else 0.3
                            ax1.plot(boot_z, bootstrap_pmfs[curve_idx], 'gray', 
                                    alpha=alpha, linewidth=0.5, zorder=0,
                                    label='Bootstrap samples' if i == 0 else "")
                
                # Debug: Print data ranges to verify they're different
                print(f"   🔍 {ion_type} - Original PMF range: {original_pmf.min():.3f} to {original_pmf.max():.3f}")
                print(f"   🔍 {ion_type} - Bootstrap mean range: {boot_mean.min():.3f} to {boot_mean.max():.3f}")
                
            else:
                # Only original PMF available
                ax1.plot(original_z, original_pmf, 'b-', linewidth=3, 
                        label=f'{ion_type} (No bootstrap)', alpha=0.8)
                print(f"   ⚠️  {ion_type} - No bootstrap data available")
            
            # FIXED: Set axis properties and legend
            ax1.set_xlabel('z (Å)')
            ax1.set_ylabel(f'PMF ({units})')
            ax1.set_title(f'{ion_type}: Original vs Bootstrap PMF')
            ax1.legend(loc='best', framealpha=0.8)
            ax1.grid(True, alpha=0.3)
            ax1.axvline(0, color='k', linestyle='--', alpha=0.5)
            
            # Set consistent y-limits for better comparison
            if ion_type in bootstrap_results and boot_mean is not None:
                all_values = np.concatenate([original_pmf, boot_mean])
                if ci_95_lower is not None:
                    all_values = np.concatenate([all_values, ci_95_lower, ci_95_upper])
                y_margin = (all_values.max() - all_values.min()) * 0.1
                ax1.set_ylim(all_values.min() - y_margin, all_values.max() + y_margin)
            
            # Right plot: Uncertainty analysis
            if ion_type in bootstrap_results and boot_std is not None:
                # Plot uncertainty (standard deviation)
                ax2.plot(boot_z, boot_std, 'purple', linewidth=2, 
                        label='Uncertainty (σ)')
                ax2.fill_between(boot_z, 0, boot_std, alpha=0.3, color='purple')
                
                # Plot relative uncertainty on twin axis
                ax2_twin = ax2.twinx()
                # Avoid division by zero
                relative_uncertainty = np.where(np.abs(boot_mean) > 0.01,
                                            boot_std / np.abs(boot_mean) * 100,
                                            0)
                ax2_twin.plot(boot_z, relative_uncertainty, 'orange', 
                            linewidth=2, linestyle='--', 
                            label='Relative uncertainty (%)')
                
                # Highlight high uncertainty regions
                high_uncertainty_threshold = np.mean(boot_std) + np.std(boot_std)
                high_regions = boot_std > high_uncertainty_threshold
                
                if np.any(high_regions):
                    ax2.fill_between(boot_z, 0, boot_std, 
                                where=high_regions, alpha=0.6, 
                                color='red', interpolate=True,
                                label='High uncertainty')
                
                ax2.set_ylabel(f'Uncertainty ({units})')
                ax2_twin.set_ylabel('Relative uncertainty (%)')
                
                # Combine legends
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            else:
                ax2.text(0.5, 0.5, f'No bootstrap data for {ion_type}', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            ax2.set_xlabel('z (Å)')
            ax2.set_title(f'{ion_type}: Uncertainty Analysis')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        filename = f'pmf_bootstrap_comparison_{comparison_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        print(f"✅ Comparison plot saved as: {filename}")
        
        plt.show()
        return fig

           

    def plot_coordination_height_surfaces(self, coordination_surface_results=None, ion_types=None,
                                        shell_indices=None,
                                        z_slice_indices=None,
                                        plot_mode='shells',  # 'shells' or 'z_slices'
                                        visualization_mode='3d_surfaces',
                                        surface_style='smooth',
                                        color_scheme='coordination_gradient',
                                        show_colorbar=True,
                                        colorbar_label='Coordination Number',
                                        colorbar_vmin=None,
                                        colorbar_vmax=None,
                                        view_angle=(30, 45),
                                        lighting_style='default',
                                        show_reference_planes=False,
                                        reference_plane_alpha=0.2,
                                        clay_overlay=False,
                                        save_plots=True,
                                        figsize=(15, 12),
                                        dpi=300,
                                        filename_prefix='coordination_height_surfaces',
                                        # Spatial interpolation parameters
                                        fill_missing_data=True,
                                        max_neighbor_cells=3,
                                        interpolation_method='simple_average',
                                        min_neighbors_required=2,
                                        interpolation_max_distance=None,
                                        # Publication settings
                                        title_fontsize=14,
                                        label_fontsize=12,
                                        tick_fontsize=10,
                                        legend_fontsize=12,
                                        colorbar_tick_fontsize=10,
                                        show_title=True,
                                        show_legend=False,
                                        # Publication figure control parameters
                                        save_individual_figures=False,
                                        individual_figsize=(8, 6),
                                        save_combined_figure=False,
                                        show_individual_figures=False,
                                        show_combined_figure=True,
                                        # Colorbar scaling options
                                        colorbar_scaling='individual',
                                        # Surface rendering parameters
                                        surface_alpha=0.8,
                                        surface_cmap='viridis',
                                        surface_lighting=True,
                                        surface_linewidth=0,
                                        surface_antialiased=True,
                                        surface_rstride=1,
                                        surface_cstride=1,
                                        # Surface smoothing parameters
                                        surface_smoothing=False,
                                        surface_smoothing_sigma=1.5,
                                        smoothing_method='gaussian',
                                        grid_upsampling_factor=1,
                                        adaptive_smoothing=True,
                                        min_data_points_for_smoothing=10,
                                        median_filter_size=3,
                                        bilateral_sigma_spatial=1.0,
                                        bilateral_sigma_intensity=0.1,
                                        savgol_window_length=5,
                                        savgol_polyorder=2,
                                        # NaN handling parameters
                                        handling_nan='default',
                                        nan_color='lightgray',
                                        nan_alpha=0.3,
                                        hatch_pattern='///',
                                        hatch_density=1.0,
                                        # Outlier detection parameters
                                        outlier_detection=False,
                                        outlier_method='median',
                                        outlier_threshold=3.0,
                                        outlier_correction='mean',
                                        outlier_log_corrections=True,
                                        statistical_outlier_method='iqr',
                                        iterative_refinement=False,
                                        iterative_threshold_decay=0.8,
                                        iterative_max_iterations=3,
                                        # Height scaling parameters
                                        height_scale_factor=1.0,
                                        base_height=0.0,
                                        # Grid and display parameters
                                        show_grid=False,
                                        grid_alpha=0.3,
                                        # Contour parameters (for 2d_contours mode)
                                        contour_levels=20,
                                        contour_filled=True,
                                        contour_linewidths=1.0):
        """
        Plot coordination number surfaces where coordination values are represented as surface height.
        
        This method creates 3D surface plots where the Z-coordinate represents coordination numbers,
        similar to topographical maps but for coordination landscapes. It adapts the full functionality
        from plot_pmf_height_surfaces() for coordination data.
        
        Parameters
        ----------
        coordination_surface_results : dict, optional
            Results from create_coordination_height_surfaces() (Step 2) containing height_surfaces.
            Can also accept raw data from calculate_solvation_shells_vs_z_detailed() (Step 1).
            If None, attempts to load from analysis results
        ion_types : list, optional
            Ion types to plot. If None, plots all available
        shell_indices : list, optional
            Indices of shells to plot (0=first, 1=second, 2=third). If None, plots all shells
        z_slice_indices : list, optional
            Indices of Z-slices to plot. If None, plots all Z-slices
        plot_mode : str, default='shells'
            Plotting strategy:
            - 'shells': Plot different shells across Z-layers for each ion (RECOMMENDED)
            - 'z_slices': Plot different Z-layers across shells for each ion
        visualization_mode : str, default='3d_surfaces'
            Plotting mode:
            - '3d_surfaces': 3D surface plots
            - '2d_contours': 2D contour projections
            - 'combined': Both 3D and 2D views
        surface_style : str, default='smooth'
            Surface rendering style:
            - 'smooth': Smooth continuous surfaces
            - 'wireframe': Wireframe surfaces
            - 'filled_contour': Filled contour surfaces
        color_scheme : str, default='coordination_gradient'
            Coloring scheme:
            - 'coordination_gradient': Color based on coordination values (RECOMMENDED)
            - 'height_gradient': Color based on surface height values
            - 'adaptive': Use solid colors (single color per surface)
        show_colorbar : bool, default=True
            Show colorbar for coordination/height values
        colorbar_label : str, default='Coordination Number'
            Label for the colorbar
        colorbar_vmin, colorbar_vmax : float, optional
            Colorbar range limits
        view_angle : tuple, default=(30, 45)
            3D viewing angle (elevation, azimuth)
        lighting_style : str, default='default'
            3D lighting setup
        show_reference_planes : bool, default=False
            Show reference planes (z=0, bulk level)
        reference_plane_alpha : float, default=0.2
            Transparency of reference planes
        clay_overlay : bool, default=False
            Overlay clay structure if available
        save_plots : bool, default=True
            Save plots to files
        figsize : tuple, default=(15, 12)
            Figure size
        dpi : int, default=300
            Resolution for saved plots
        filename_prefix : str, default='coordination_height_surfaces'
            Prefix for saved filenames
        
        Spatial Interpolation Parameters:
        fill_missing_data : bool, default=True
            Fill missing/infinite coordination data using spatial interpolation
        max_neighbor_cells : int, default=3
            Maximum number of neighbor cells to consider for averaging
        interpolation_method : str, default='simple_average'
            Method for spatial interpolation:
            - 'simple_average': Simple mean of valid neighbors
            - 'distance_weighted': Distance-weighted averaging
            - 'gaussian_kernel': Gaussian kernel averaging
        min_neighbors_required : int, default=2
            Minimum number of valid neighbors required for interpolation
        interpolation_max_distance : int, optional
            Maximum distance (in grid cells) for neighbor search.
            If None, uses max_neighbor_cells as radius
        
        Colorbar Scaling:
        colorbar_scaling : str, default='individual'
            Colorbar scaling strategy:
            - 'individual': Each surface gets its own colorbar range
            - 'global': All surfaces per ion use same range
            - 'unified': All ions and surfaces use same global range
        
        Surface Smoothing Parameters:
        surface_smoothing : bool, default=False
            Whether to apply smoothing to surfaces for better visual appearance
        surface_smoothing_sigma : float, default=1.5
            Gaussian smoothing parameter (higher = more smoothing)
        smoothing_method : str, default='gaussian'
            Smoothing method: 'gaussian', 'median', 'bilateral', 'savgol'
        grid_upsampling_factor : int, default=1
            Factor to increase grid resolution through interpolation before smoothing
        adaptive_smoothing : bool, default=True
            Apply adaptive smoothing based on local data density
        min_data_points_for_smoothing : int, default=10
            Minimum number of valid data points required for smoothing
        median_filter_size : int, default=3
            Size of median filter kernel (for median smoothing)
        bilateral_sigma_spatial : float, default=1.0
            Spatial sigma for bilateral filtering
        bilateral_sigma_intensity : float, default=0.1
            Intensity sigma for bilateral filtering
        savgol_window_length : int, default=5
            Window length for Savitzky-Golay filter
        savgol_polyorder : int, default=2
            Polynomial order for Savitzky-Golay filter
        
        NaN Handling Parameters:
        handling_nan : str, default='default'
            How to handle NaN/missing values:
            - 'default': Matplotlib's default (may show artifacts)
            - 'mask': Mask NaN regions with specific color
            - 'transparent': Make NaN regions fully transparent
            - 'hatched': Add hatch pattern to NaN regions
            - 'interpolate': Interpolate through NaN regions
        nan_color : str, default='lightgray'
            Color for NaN regions (when handling_nan='mask')
        nan_alpha : float, default=0.3
            Transparency for NaN regions
        hatch_pattern : str, default='///'
            Hatch pattern for NaN regions (when handling_nan='hatched')
        hatch_density : float, default=1.0
            Density of hatch pattern
        
        Outlier Detection Parameters:
        outlier_detection : bool, default=False
            Enable outlier detection and correction
        outlier_method : str, default='median'
            Method for outlier detection: 'median', 'std', 'iqr'
        outlier_threshold : float, default=3.0
            Threshold for outlier detection (in standard deviations or IQR)
        outlier_correction : str, default='mean'
            How to correct outliers: 'mean', 'median', 'interpolate', 'clip'
        outlier_log_corrections : bool, default=True
            Log outlier corrections to console
        statistical_outlier_method : str, default='iqr'
            Statistical method: 'iqr', 'zscore', 'modified_zscore'
        iterative_refinement : bool, default=False
            Apply iterative outlier detection
        iterative_threshold_decay : float, default=0.8
            Decay factor for threshold in iterative refinement
        iterative_max_iterations : int, default=3
            Maximum number of refinement iterations
        
        Height Scaling Parameters:
        height_scale_factor : float, default=1.0
            Scaling factor for coordination values when converting to height
        base_height : float, default=0.0
            Base height for coordination surfaces
        
        Grid and Display Parameters:
        show_grid : bool, default=False
            Show grid lines on plots
        grid_alpha : float, default=0.3
            Transparency of grid lines
        
        Contour Parameters (for 2d_contours mode):
        contour_levels : int, default=20
            Number of contour levels
        contour_filled : bool, default=True
            Use filled contours
        contour_linewidths : float, default=1.0
            Width of contour lines
        
        Publication Font Settings:
        title_fontsize : int, default=14
        label_fontsize : int, default=12
        tick_fontsize : int, default=10
        legend_fontsize : int, default=12
        colorbar_tick_fontsize : int, default=10
        show_title : bool, default=True
        show_legend : bool, default=False
        
        Publication Figure Control:
        save_individual_figures : bool, default=False
        individual_figsize : tuple, default=(8, 6)
        save_combined_figure : bool, default=False
        show_individual_figures : bool, default=False
        show_combined_figure : bool, default=True
        
        Surface Rendering Options:
        surface_alpha : float, default=0.8
        surface_cmap : str, default='viridis'
        surface_lighting : bool, default=True
        surface_linewidth : float, default=0
        surface_antialiased : bool, default=True
        surface_rstride, surface_cstride : int, default=1
        
        Returns
        -------
        dict
            Dictionary containing plot objects and metadata
        """
        
        from scipy.ndimage import gaussian_filter, median_filter
        from scipy.interpolate import interp2d
        import numpy as np
        
        print(f"\n🧭 PLOTTING COORDINATION HEIGHT SURFACES")
        print(f"{'='*60}")
        print(f"Plot mode: {plot_mode}")
        print(f"Visualization mode: {visualization_mode}")
        print(f"Surface style: {surface_style}")
        print(f"Color scheme: {color_scheme}")
        print(f"Colorbar scaling: {colorbar_scaling}")
        
        # Print smoothing parameters
        if surface_smoothing:
            print(f"Surface smoothing: {smoothing_method} (σ={surface_smoothing_sigma}, upsampling={grid_upsampling_factor}x)")
            if adaptive_smoothing:
                print(f"  Adaptive smoothing: enabled (min_points={min_data_points_for_smoothing})")
            if smoothing_method == 'median':
                print(f"  Median filter size: {median_filter_size}")
            elif smoothing_method == 'bilateral':
                print(f"  Bilateral: spatial_σ={bilateral_sigma_spatial}, intensity_σ={bilateral_sigma_intensity}")
            elif smoothing_method == 'savgol':
                print(f"  Savitzky-Golay: window={savgol_window_length}, order={savgol_polyorder}")
        
        # Print NaN handling parameters
        if handling_nan != 'default':
            print(f"NaN handling: {handling_nan}")
            if handling_nan == 'mask':
                print(f"  NaN color: {nan_color} (α={nan_alpha})")
            elif handling_nan == 'hatched':
                print(f"  Hatch pattern: {hatch_pattern} (density={hatch_density})")
        
        # Print outlier detection parameters
        if outlier_detection:
            print(f"Outlier detection: {outlier_method} (threshold={outlier_threshold})")
            print(f"  Correction method: {outlier_correction}")
            print(f"  Statistical method: {statistical_outlier_method}")
            if iterative_refinement:
                print(f"  Iterative refinement: max_iter={iterative_max_iterations}, decay={iterative_threshold_decay}")
        
        # Print interpolation parameters
        if fill_missing_data:
            print(f"Spatial interpolation: {interpolation_method} (max_cells={max_neighbor_cells}, min_neighbors={min_neighbors_required})")

        
        # Get coordination HEIGHT SURFACE data from analysis results
        if coordination_surface_results is None:
            # First, check if we have height surface data from create_coordination_height_surfaces()
            if (hasattr(self.analysis.results, 'coordination_height_surfaces') and 
                self.analysis.results.coordination_height_surfaces is not None):
                height_surface_data = self.analysis.results.coordination_height_surfaces
                print(f"✓ Found coordination height surface data in: analysis.results.coordination_height_surfaces")
                
                # Use the height surface data directly - this is what we want!
                coordination_surface_results = height_surface_data
            
            # Fallback: If no height surfaces, try to find raw coordination data and create height surfaces
            elif (hasattr(self.analysis.results, 'solvation_shells_xy_spatial') and 
                  self.analysis.results.solvation_shells_xy_spatial is not None):
                print("⚠️ No height surfaces found, but found raw coordination data")
                print("💡 You should run create_coordination_height_surfaces() first to convert coordination numbers to heights")
                
                spatial_data = self.analysis.results.solvation_shells_xy_spatial
                print(f"✓ Found spatial coordination data in: analysis.results.solvation_shells_xy_spatial")
                
                # Convert z_slices format to layer_data format expected by plotting
                if 'z_slices' in spatial_data:
                    layer_data = {}
                    for layer in spatial_data['z_slices']:
                        z_center = layer['z_center']
                        layer_data[z_center] = layer
                    coordination_surface_results = {'layer_data': layer_data}
                    print(f"✓ Converted {len(spatial_data['z_slices'])} z_slices to layer_data format")
                else:
                    coordination_surface_results = spatial_data
            
            # Fallback: Try other possible attribute names for coordination data
            else:
                possible_attributes = [
                    'spatial_analysis_detailed',
                    'spatial_analysis_results',
                    'solvation_shells_detailed',
                    'coordination_analysis',
                    'layer_analysis',
                    'z_layer_analysis'
                ]
                
                coordination_surface_results = None
                for attr_name in possible_attributes:
                    if hasattr(self.analysis.results, attr_name):
                        attr_value = getattr(self.analysis.results, attr_name)
                        if attr_value and isinstance(attr_value, dict) and 'layer_data' in attr_value:
                            coordination_surface_results = attr_value
                            print(f"✓ Found coordination data in: analysis.results.{attr_name}")
                            break
            
            if coordination_surface_results is None:
                # Print available results to help debug
                print("❌ No coordination surface results found.")
                print("🔍 DEBUGGING: Available analysis.results attributes:")
                
                found_data = False
                if hasattr(self.analysis, 'results'):
                    results_attrs = [attr for attr in dir(self.analysis.results) if not attr.startswith('_')]
                    for attr in sorted(results_attrs):
                        value = getattr(self.analysis.results, attr)
                        if value is not None:
                            print(f"  ✓ {attr}: {type(value)}")
                            # Check if this might be coordination data
                            if isinstance(value, dict):
                                if 'layer_data' in value:
                                    print(f"    🎯 Contains 'layer_data' - this IS coordination data!")
                                    coordination_surface_results = value
                                    found_data = True
                                    break
                                elif any(key in str(value).lower() for key in ['layer', 'coord', 'shell']):
                                    print(f"    💡 Might contain coordination data - checking keys: {list(value.keys())}")
                        else:
                            print(f"  ✗ {attr}: None")
                    
                    if not found_data:
                        print(f"\n🔍 MANUAL SEARCH: Looking for any dict with coordination-like data...")
                        for attr in sorted(results_attrs):
                            value = getattr(self.analysis.results, attr)
                            if isinstance(value, dict) and value:
                                keys = list(value.keys())
                                if any('layer' in str(k).lower() or 'coord' in str(k).lower() or 'shell' in str(k).lower() or 'z' in str(k).lower() for k in keys):
                                    print(f"  🤔 {attr} has interesting keys: {keys}")
                                    # Try to access layer-like data
                                    for key in keys:
                                        if 'layer' in str(key).lower() and isinstance(value[key], dict):
                                            print(f"    🔍 {attr}['{key}'] contains: {list(value[key].keys())}")
                                            if 'coordination_grids' in value[key] or any('coord' in str(k).lower() for k in value[key].keys()):
                                                print(f"    ✅ Found coordination data in {attr}['{key}']!")
                                                coordination_surface_results = {'layer_data': value[key]}
                                                found_data = True
                                                break
                                    if found_data:
                                        break
                else:
                    print("  ✗ No analysis.results found")
                
                if coordination_surface_results is None:
                    print(f"\n💡 SOLUTION OPTIONS:")
                    print(f"1. Run the debug method first: plotter.debug_coordination_data_availability()")
                    print(f"2. Pass data directly: coordination_surface_results=your_coordination_data")
                    print(f"3. Re-run: analysis.calculate_solvation_shells_vs_z_detailed()")
                    raise ValueError("No coordination surface results found. Run calculate_solvation_shells_vs_z_detailed() first and ensure the results are stored in analysis.results.")
        
        # Validate data structure - check if we have height surfaces (Step 2) or raw data (Step 1)
        if 'height_surfaces' in coordination_surface_results:
            # We have HEIGHT SURFACE data from create_coordination_height_surfaces() - STEP 2
            print("✅ Using height surface data from create_coordination_height_surfaces() (Step 2)")
            height_surfaces = coordination_surface_results['height_surfaces']
            
            # Extract available ions from height surface data
            available_ions = list(height_surfaces.keys())
            print(f"Available ions from height surfaces: {available_ions}")
            
            # For height surfaces, we don't need to process raw coordination grids
            layer_data = None
            
        elif 'layer_data' in coordination_surface_results:
            # We have raw coordination data from calculate_solvation_shells_vs_z_detailed() - STEP 1
            print("⚠️ Using raw coordination data - you should run create_coordination_height_surfaces() first")
            layer_data = coordination_surface_results['layer_data']
        
        else:
            raise ValueError("Invalid coordination surface results. Must contain 'height_surfaces' (from Step 2) or 'layer_data' (from Step 1).")
        
        # Only process raw coordination data if we don't have height surfaces
        if layer_data is not None:
            # Get available ions and shells from coordination_grids
            available_ions = []
            available_shells = []
            shell_names = ['first_shell', 'second_shell', 'third_shell']
            
            for z_layer, z_data in layer_data.items():
                if 'coordination_grids' in z_data:
                    coord_grids = z_data['coordination_grids']
                    for shell_type in shell_names:
                        if shell_type in coord_grids:
                            available_shells.append(shell_type)
                            for ion_type in coord_grids[shell_type]:
                                if ion_type not in available_ions:
                                    available_ions.append(ion_type)
            
            available_shells = list(set(available_shells))  # Remove duplicates
            
            if not available_ions:
                raise ValueError("No coordination data found in results")
            
            print(f"Available ions: {available_ions}")
            print(f"Available shells: {available_shells}")
            print(f"Available Z-layers: {list(layer_data.keys())}")
        
        # Get ion types to plot
        if ion_types is None:
            ion_types = available_ions
        elif isinstance(ion_types, str):
            ion_types = [ion_types]
        
        # Validate ion types
        invalid_ions = [ion for ion in ion_types if ion not in available_ions]
        if invalid_ions:
            print(f"⚠ Warning: Ion types {invalid_ions} not found in results")
        ion_types = [ion for ion in ion_types if ion in available_ions]
        
        if not ion_types:
            raise ValueError("No valid ion types to plot")
        
        print(f"Plotting ion types: {ion_types}")
        
        # ===== HANDLE HEIGHT SURFACES (STEP 2 DATA) =====
        if layer_data is None:  # We have height surface data from Step 2
            print("🎯 Using pre-computed height surfaces from create_coordination_height_surfaces()")
            
            # Get coordinate data from height surface results (stored at top level)
            x_centers = coordination_surface_results['x_centers']
            y_centers = coordination_surface_results['y_centers']
            
            # Transform coordinates to start from 0 (same as PMF method)
            x_display = x_centers - x_centers.min()
            y_display = y_centers - y_centers.min()
            
            print(f"Coordinate range: X=[0, {x_display[-1]:.1f}], Y=[0, {y_display[-1]:.1f}]")
            
            # Create coordinate meshgrid for surfaces
            X_display, Y_display = np.meshgrid(x_display, y_display)
            
            # Extract surfaces to plot from height surface data
            first_ion = list(height_surfaces.keys())[0]
            surfaces_to_plot = list(height_surfaces[first_ion]['surfaces'].keys())
            print(f"Available surfaces to plot: {surfaces_to_plot}")
            
            # Set plot dimension based on surface names
            if any('shell' in str(surface) for surface in surfaces_to_plot):
                plot_dimension = 'shells'
                available_shells = surfaces_to_plot  # Shell names from height surface data
                surface_labels = surfaces_to_plot  # Use surface names as labels
                print(f"Detected shell-based surfaces: {plot_dimension}")
            else:
                plot_dimension = 'z_layers'
                available_shells = []  # No shells for z-layer mode
                # For numeric z-layers, format as "z=X.X Å"
                surface_labels = [f"z={z:.1f} Å" if isinstance(z, (int, float)) else str(z) for z in surfaces_to_plot]
                print(f"Detected z-layer based surfaces: {plot_dimension}")
            
            # Use existing height surface data
            print(f"Available height surface data for ions: {list(height_surfaces.keys())}")
            
        # ===== PROCESS RAW COORDINATION DATA (STEP 1 DATA) =====
        else:  # We have raw coordination data - process it
            print("⚙️ Processing raw coordination data to create height surfaces...")
            
            # Get shells and Z-slices to plot based on mode
            available_shells = []
            shell_names = ['first_shell', 'second_shell', 'third_shell']
            
            for z_layer, z_data in layer_data.items():
                if 'coordination_grids' in z_data:
                    coord_grids = z_data['coordination_grids']
                    for shell_type in shell_names:
                        if shell_type in coord_grids:
                            available_shells.append(shell_type)
            
            available_shells = list(set(available_shells))  # Remove duplicates
            
            if plot_mode == 'shells':
                # Plot different shells - each surface represents a shell across Z-layers
                if shell_indices is None:
                    surfaces_to_plot = available_shells
                    surface_labels = available_shells
                else:
                    surfaces_to_plot = [available_shells[i] for i in shell_indices if i < len(available_shells)]
                    surface_labels = surfaces_to_plot
                
                plot_dimension = 'shells'
                print(f"Shell-focused mode: plotting {len(surfaces_to_plot)} shells: {surfaces_to_plot}")
                
            elif plot_mode == 'z_slices':
                # Plot different Z-slices - each surface represents a Z-layer across shells
                z_layers = list(layer_data.keys())
                if z_slice_indices is None:
                    surfaces_to_plot = z_layers
                    surface_labels = [f"z={z:.1f}" for z in z_layers]
                else:
                    surfaces_to_plot = [z_layers[i] for i in z_slice_indices if i < len(z_layers)]
                    surface_labels = [f"z={z:.1f}" for z in surfaces_to_plot]
                
                plot_dimension = 'z_layers'
                print(f"Z-slice-focused mode: plotting {len(surfaces_to_plot)} Z-layers: {surface_labels}")
            
            else:
                raise ValueError(f"Invalid plot_mode: {plot_mode}. Use 'shells' or 'z_slices'")
            
            # Get coordinate arrays from the first available layer
            first_layer = list(layer_data.keys())[0]
            first_layer_data = layer_data[first_layer]
            
            if 'x_centers' not in first_layer_data or 'y_centers' not in first_layer_data:
                raise ValueError("Coordinate information (x_centers, y_centers) not found in layer data")
            
            x_centers = first_layer_data['x_centers']
            y_centers = first_layer_data['y_centers']
            
            # Transform coordinates to start from 0 (SAME AS PMF METHOD)
            x_display = x_centers - x_centers.min()  # Start X from 0
            y_display = y_centers - y_centers.min()  # Start Y from 0
            
            print(f"Coordinate transformation: X=[0, {x_display[-1]:.1f}], Y=[0, {y_display[-1]:.1f}]")
            
            # Create coordinate meshgrid for surfaces
            X_display, Y_display = np.meshgrid(x_display, y_display)
            
            # Convert coordination data to height surface format
            print(f"\n📊 Converting coordination data to height surface format...")
            
            height_surfaces = {}
            
            # Define colors for different ions (reuse from existing method)
            ion_colors = {
                'NA': '#1f77b4',  # Blue
                'MG': '#ff7f0e',  # Orange  
                'CA': '#2ca02c',  # Green
                'K': '#d62728',   # Red
                'LI': '#9467bd',  # Purple
                'RB': '#8c564b',  # Brown
                'SR': '#e377c2',  # Pink
                'CL': '#7f7f7f',  # Gray
                'BR': '#bcbd22',  # Olive
                'F': '#17becf'    # Cyan
            }
            
            for ion_type in ion_types:
                print(f"  Processing {ion_type}...")
                
                ion_data = {
                    'surfaces': {},
                    'color': ion_colors.get(ion_type, '#1f77b4'),
                    'alpha': 0.8
                }
                
                for i, surface_key in enumerate(surfaces_to_plot):
                    print(f"    Creating surface for {surface_key}...")
                    
                    # Initialize coordination grid for this surface
                    coord_surface = np.full(X_display.shape, np.nan)
                    
                    if plot_mode == 'shells':
                        # Aggregate across Z-layers for this shell
                        shell_type = surface_key
                        coord_values = []
                        
                        for z_layer, z_data in layer_data.items():
                            if ('coordination_grids' in z_data and 
                                shell_type in z_data['coordination_grids'] and
                                ion_type in z_data['coordination_grids'][shell_type]):
                                
                                shell_grid = z_data['coordination_grids'][shell_type][ion_type]
                                coord_values.append(shell_grid)
                        
                        if coord_values:
                            # Average across Z-layers (you could also use max, median, etc.)
                            coord_surface = np.mean(coord_values, axis=0)
                            surface_label = f"{shell_type.replace('_', ' ').title()}"
                        else:
                            print(f"      ⚠ No data found for {ion_type} {shell_type}")
                            continue
                            
                    elif plot_mode == 'z_slices':
                        # Aggregate across shells for this Z-layer
                        z_layer = surface_key
                        z_data = layer_data[z_layer]
                        
                        if 'coordination_grids' in z_data:
                            coord_grids = z_data['coordination_grids']
                            coord_values = []
                            
                            for shell_type in available_shells:
                                if (shell_type in coord_grids and 
                                    ion_type in coord_grids[shell_type]):
                                    
                                    shell_grid = coord_grids[shell_type][ion_type]
                                    coord_values.append(shell_grid)
                            
                            if coord_values:
                                # Sum across shells for total coordination
                                coord_surface = np.sum(coord_values, axis=0)
                                surface_label = f"z = {z_layer:.1f} Å"
                            else:
                                print(f"      ⚠ No data found for {ion_type} at z = {z_layer}")
                                continue
                        else:
                            print(f"      ⚠ No coordination_grids found for z = {z_layer}")
                            continue
                    
                    # Convert coordination values to height surface
                    coord_height_surface = base_height + coord_surface * height_scale_factor
                    
                    # Store surface data in PMF-like format
                    surface_info = {
                        'surface_data': {
                            'X': X_display,
                            'Y': Y_display, 
                            'Z': coord_height_surface
                        },
                        'pmf_grid': coord_surface,  # Store original coordination data as 'pmf_grid' for compatibility
                        'base_z': base_height,
                        'original_z': surface_key if isinstance(surface_key, (int, float)) else i,
                        'surface_type': surface_key,
                        'surface_label': surface_label
                    }
                    
                    ion_data['surfaces'][surface_key] = surface_info
                    print(f"      ✓ Surface created: coord range = {np.nanmin(coord_surface):.2f} to {np.nanmax(coord_surface):.2f}")
                
                height_surfaces[ion_type] = ion_data
                print(f"    ✓ {ion_type}: {len(ion_data['surfaces'])} surfaces created")
            
            # Create the height surface results structure (compatible with PMF method)
            height_surface_results = {
                'height_surfaces': height_surfaces,
                'x_centers': x_centers,
                'y_centers': y_centers,
                'z_scale_factor': height_scale_factor,
                'base_height': base_height,
                'plot_mode': plot_mode,
                'coordination_metadata': {
                    'available_shells': available_shells,
                    'z_layers': list(layer_data.keys()) if layer_data is not None else [],
                    'surfaces_plotted': surfaces_to_plot
                }
            }
        
        # ===== COMMON PLOTTING SECTION (BOTH HEIGHT SURFACES AND RAW DATA) =====
        # At this point, we have height_surfaces data regardless of whether it came from Step 2 or was processed from raw data
        
        # Prepare final height_surface_results for plotting
        if layer_data is None:  # Height surface data from Step 2
            # Extract metadata from existing height surfaces
            first_ion = list(height_surfaces.keys())[0]
            x_centers = coordination_surface_results['x_centers']  
            y_centers = coordination_surface_results['y_centers']
            
            height_surface_results = {
                'height_surfaces': height_surfaces,
                'x_centers': x_centers,
                'y_centers': y_centers,
                'z_scale_factor': height_scale_factor,
                'base_height': base_height,
                'plot_mode': plot_mode,
                'coordination_metadata': {
                    'surfaces_plotted': list(height_surfaces[first_ion]['surfaces'].keys())
                }
            }
        else:  # Raw data was processed above
            # height_surface_results already created above
            pass
        
        print(f"✅ Data conversion complete!")
        
        # ===== HELPER FUNCTIONS FOR ADVANCED PROCESSING =====
        def _fill_missing_coord_data(coord_grid, method=interpolation_method,
                                     max_cells=max_neighbor_cells,
                                     min_neighbors=min_neighbors_required,
                                     max_distance=interpolation_max_distance):
            """Fill NaN/infinite coordination values using spatial averaging of neighbors"""
            if not fill_missing_data:
                return coord_grid.copy()
            
            filled_grid = coord_grid.copy()
            missing_mask = ~np.isfinite(coord_grid)
            
            if not np.any(missing_mask):
                return filled_grid
            
            missing_count = np.sum(missing_mask)
            ny, nx = coord_grid.shape
            search_dist = max_distance if max_distance is not None else max_cells
            filled_count = 0
            
            for i, j in np.argwhere(missing_mask):
                i_min = max(0, i - search_dist)
                i_max = min(ny, i + search_dist + 1)
                j_min = max(0, j - search_dist)
                j_max = min(nx, j + search_dist + 1)
                
                neighbor_region = coord_grid[i_min:i_max, j_min:j_max]
                valid_neighbors = neighbor_region[np.isfinite(neighbor_region)]
                
                if len(valid_neighbors) < min_neighbors:
                    continue
                
                if method == 'simple_average':
                    fill_value = np.mean(valid_neighbors)
                elif method == 'distance_weighted':
                    distances, values = [], []
                    for di in range(i_min, i_max):
                        for dj in range(j_min, j_max):
                            if np.isfinite(coord_grid[di, dj]):
                                distance = np.sqrt((di - i)**2 + (dj - j)**2)
                                if distance > 0:
                                    distances.append(distance)
                                    values.append(coord_grid[di, dj])
                    if distances:
                        weights = 1.0 / np.array(distances)
                        fill_value = np.average(values, weights=weights)
                    else:
                        fill_value = np.mean(valid_neighbors)
                elif method == 'gaussian_kernel':
                    distances, values = [], []
                    for di in range(i_min, i_max):
                        for dj in range(j_min, j_max):
                            if np.isfinite(coord_grid[di, dj]):
                                distance = np.sqrt((di - i)**2 + (dj - j)**2)
                                distances.append(distance)
                                values.append(coord_grid[di, dj])
                    if distances:
                        sigma = max_cells / 2.0
                        weights = np.exp(-np.array(distances)**2 / (2 * sigma**2))
                        fill_value = np.average(values, weights=weights)
                    else:
                        fill_value = np.mean(valid_neighbors)
                else:
                    fill_value = np.mean(valid_neighbors)
                
                filled_grid[i, j] = fill_value
                filled_count += 1
            
            if filled_count > 0:
                print(f"      🔧 Filled {filled_count}/{missing_count} missing cells")
            
            return filled_grid
        
        def _detect_and_correct_outliers(coord_grid):
            """Detect and correct outliers in coordination data"""
            if not outlier_detection:
                return coord_grid.copy()
            
            corrected_grid = coord_grid.copy()
            finite_mask = np.isfinite(coord_grid)
            finite_values = coord_grid[finite_mask]
            
            if len(finite_values) == 0:
                return corrected_grid
            
            # Detect outliers based on method
            if outlier_method == 'median':
                median = np.median(finite_values)
                mad = np.median(np.abs(finite_values - median))
                outlier_mask = np.abs(coord_grid - median) > (outlier_threshold * mad)
            elif outlier_method == 'std':
                mean = np.mean(finite_values)
                std = np.std(finite_values)
                outlier_mask = np.abs(coord_grid - mean) > (outlier_threshold * std)
            elif outlier_method == 'iqr' or statistical_outlier_method == 'iqr':
                q1, q3 = np.percentile(finite_values, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - (outlier_threshold * iqr)
                upper_bound = q3 + (outlier_threshold * iqr)
                outlier_mask = (coord_grid < lower_bound) | (coord_grid > upper_bound)
            else:
                return corrected_grid
            
            outlier_mask = outlier_mask & finite_mask
            n_outliers = np.sum(outlier_mask)
            
            if n_outliers > 0:
                if outlier_log_corrections:
                    print(f"      🔍 Detected {n_outliers} outliers ({n_outliers/finite_mask.sum()*100:.1f}%)")
                
                # Correct outliers based on correction method
                if outlier_correction == 'mean':
                    correction_value = np.mean(finite_values[~outlier_mask[finite_mask]])
                elif outlier_correction == 'median':
                    correction_value = np.median(finite_values[~outlier_mask[finite_mask]])
                elif outlier_correction == 'clip':
                    if outlier_method == 'iqr':
                        corrected_grid = np.clip(corrected_grid, lower_bound, upper_bound)
                        return corrected_grid
                    else:
                        mean = np.mean(finite_values)
                        std = np.std(finite_values)
                        corrected_grid = np.clip(corrected_grid, mean - outlier_threshold*std, mean + outlier_threshold*std)
                        return corrected_grid
                elif outlier_correction == 'interpolate':
                    # Use spatial interpolation for outliers
                    corrected_grid[outlier_mask] = np.nan
                    corrected_grid = _fill_missing_coord_data(corrected_grid)
                    return corrected_grid
                else:
                    correction_value = np.mean(finite_values)
                
                corrected_grid[outlier_mask] = correction_value
                
                if outlier_log_corrections:
                    print(f"      ✅ Corrected outliers using {outlier_correction} method")
            
            return corrected_grid
        
        def _apply_smoothing(coord_grid, X, Y):
            """Apply smoothing to coordination surface"""
            if not surface_smoothing:
                return coord_grid, X, Y
            
            finite_count = np.sum(np.isfinite(coord_grid))
            if finite_count < min_data_points_for_smoothing:
                if adaptive_smoothing:
                    print(f"      ⚠️ Insufficient data points ({finite_count}) for smoothing, skipping")
                    return coord_grid, X, Y
            
            smoothed_grid = coord_grid.copy()
            
            # Apply grid upsampling if requested
            X_smooth, Y_smooth = X, Y
            if grid_upsampling_factor > 1:
                from scipy.interpolate import RectBivariateSpline
                ny, nx = coord_grid.shape
                ny_new = ny * grid_upsampling_factor
                nx_new = nx * grid_upsampling_factor
                
                x_orig = np.linspace(0, 1, nx)
                y_orig = np.linspace(0, 1, ny)
                x_new = np.linspace(0, 1, nx_new)
                y_new = np.linspace(0, 1, ny_new)
                
                # Interpolate the coordination grid
                valid_mask = np.isfinite(coord_grid)
                if np.any(valid_mask):
                    spline = RectBivariateSpline(y_orig, x_orig, np.nan_to_num(coord_grid, nan=0))
                    smoothed_grid = spline(y_new, x_new)
                    
                    # Update coordinate meshgrids
                    x_display_new = np.linspace(X[0,0], X[0,-1], nx_new)
                    y_display_new = np.linspace(Y[0,0], Y[-1,0], ny_new)
                    X_smooth, Y_smooth = np.meshgrid(x_display_new, y_display_new)
            
            # Apply smoothing method - use the imported functions from outer scope
            if smoothing_method == 'gaussian':
                from scipy.ndimage import gaussian_filter as gauss_filt
                smoothed_grid = gauss_filt(smoothed_grid, sigma=surface_smoothing_sigma)
            elif smoothing_method == 'median':
                from scipy.ndimage import median_filter as med_filt
                smoothed_grid = med_filt(smoothed_grid, size=median_filter_size)
            elif smoothing_method == 'bilateral':
                # Simplified bilateral filter using gaussian
                from scipy.ndimage import gaussian_filter as gauss_filt
                smoothed_grid = gauss_filt(smoothed_grid, sigma=bilateral_sigma_spatial)
            elif smoothing_method == 'savgol':
                from scipy.signal import savgol_filter
                if savgol_window_length <= smoothed_grid.shape[0] and savgol_window_length <= smoothed_grid.shape[1]:
                    smoothed_grid = savgol_filter(savgol_filter(smoothed_grid, savgol_window_length, savgol_polyorder, axis=0),
                                                 savgol_window_length, savgol_polyorder, axis=1)
            
            return smoothed_grid, X_smooth, Y_smooth
        
        # ===== COORDINATION-SPECIFIC PLOTTING IMPLEMENTATION =====
        # Adapted from PMF method but tailored for coordination data
        print(f"\n🎨 Rendering coordination height surfaces with proper coordination coloring...")
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        # Create the figure and subplots
        fig = plt.figure(figsize=figsize)
        plot_results = {'fig': fig, 'axes': [], 'surfaces': {}}
        
        # Create subplot for each ion
        n_ions = len(ion_types)
        if n_ions == 1:
            ax = fig.add_subplot(111, projection='3d')
            plot_results['axes'].append(ax)
        else:
            for i in range(n_ions):
                ax = fig.add_subplot(1, n_ions, i+1, projection='3d')
                plot_results['axes'].append(ax)
        
        # Plot each ion
        for ion_idx, ion_type in enumerate(ion_types):
            ax = plot_results['axes'][ion_idx]
            print(f"\n📊 Plotting coordination surfaces for {ion_type}:")
            
            ion_surfaces_data = height_surfaces[ion_type]['surfaces']
            ion_color = height_surfaces[ion_type]['color']
            
            ion_plot_surfaces = []
            
            # Plot each surface (shell) for this ion
            for shell_name in surfaces_to_plot:
                if shell_name in ion_surfaces_data:
                    surface_info = ion_surfaces_data[shell_name]
                    surface_data = surface_info['surface_data']
                    coord_values = surface_info['pmf_grid']  # This contains coordination numbers
                    
                    X_surf, Y_surf, Z_surf = surface_data['X'], surface_data['Y'], surface_data['Z']
                    
                    # ===== APPLY ADVANCED PROCESSING =====
                    # 1. Fill missing data
                    coord_processed = _fill_missing_coord_data(coord_values)
                    
                    # 2. Detect and correct outliers
                    coord_processed = _detect_and_correct_outliers(coord_processed)
                    
                    # 3. Apply smoothing
                    coord_smoothed, X_plot, Y_plot = _apply_smoothing(coord_processed, X_surf, Y_surf)
                    
                    # Recalculate height surface after processing
                    Z_plot = base_height + coord_smoothed * height_scale_factor
                    
                    # Print coordination statistics for this shell
                    finite_coords = coord_smoothed[np.isfinite(coord_smoothed)]
                    if len(finite_coords) > 0:
                        coord_min, coord_max = np.min(finite_coords), np.max(finite_coords)
                        coord_mean = np.mean(finite_coords)
                        print(f"  {shell_name}: coord range = {coord_min:.1f} to {coord_max:.1f}, mean = {coord_mean:.1f}")
                    else:
                        print(f"  {shell_name}: No valid coordination data")
                        continue
                    
                    # ===== APPLY COLORING SCHEME =====
                    if color_scheme == 'coordination_gradient':
                        # Use actual coordination values for coloring
                        if len(finite_coords) > 0 and coord_max > coord_min:
                            from matplotlib.colors import Normalize
                            
                            # Determine colorbar limits
                            if colorbar_vmin is not None and colorbar_vmax is not None:
                                vmin, vmax = colorbar_vmin, colorbar_vmax
                            elif colorbar_scaling == 'unified':
                                # Will be calculated globally
                                vmin, vmax = coord_min, coord_max  # Placeholder
                            else:
                                vmin, vmax = coord_min, coord_max
                            
                            norm = Normalize(vmin=vmin, vmax=vmax)
                            
                            # Handle NaN values based on handling_nan parameter
                            if handling_nan == 'mask':
                                # Mask NaN regions with specific color
                                coord_for_color = coord_smoothed.copy()
                                nan_mask = ~np.isfinite(coord_for_color)
                                coord_for_color[nan_mask] = vmin  # Set to minimum
                                
                                coord_normalized = norm(coord_for_color)
                                facecolors = plt.cm.get_cmap(surface_cmap)(coord_normalized)
                                
                                # Set NaN regions to nan_color
                                from matplotlib.colors import to_rgba
                                nan_rgba = to_rgba(nan_color, alpha=nan_alpha)
                                facecolors[nan_mask] = nan_rgba
                                
                            elif handling_nan == 'transparent':
                                coord_normalized = norm(coord_smoothed)
                                facecolors = plt.cm.get_cmap(surface_cmap)(coord_normalized)
                                # Set NaN regions to transparent
                                nan_mask = ~np.isfinite(coord_smoothed)
                                facecolors[nan_mask, 3] = 0  # Alpha = 0
                                
                            elif handling_nan == 'interpolate':
                                # Already filled by _fill_missing_coord_data
                                coord_normalized = norm(coord_smoothed)
                                facecolors = plt.cm.get_cmap(surface_cmap)(coord_normalized)
                                
                            else:  # 'default'
                                coord_normalized = norm(coord_smoothed)
                                facecolors = plt.cm.get_cmap(surface_cmap)(coord_normalized)
                            
                            # Plot surface with facecolors
                            if surface_style == 'wireframe':
                                surf = ax.plot_wireframe(X_plot, Y_plot, Z_plot,
                                                        color=ion_color,
                                                        alpha=surface_alpha,
                                                        linewidth=surface_linewidth or 1.0,
                                                        rstride=surface_rstride,
                                                        cstride=surface_cstride)
                            else:  # 'smooth' or 'filled_contour'
                                surf = ax.plot_surface(X_plot, Y_plot, Z_plot, 
                                                     facecolors=facecolors,
                                                     alpha=surface_alpha,
                                                     linewidth=surface_linewidth,
                                                     antialiased=surface_antialiased,
                                                     rstride=surface_rstride,
                                                     cstride=surface_cstride,
                                                     shade=surface_lighting)
                            
                            print(f"    ✅ Applied coordination gradient: {vmin:.1f}-{vmax:.1f}")
                            
                            # Add colorbar if requested
                            if show_colorbar and colorbar_scaling == 'individual':
                                from matplotlib.cm import ScalarMappable
                                sm = ScalarMappable(norm=norm, cmap=surface_cmap)
                                sm.set_array([])
                                cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
                                cbar.set_label(colorbar_label, fontsize=label_fontsize)
                                cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
                        else:
                            # Fallback to solid color
                            surf = ax.plot_surface(X_plot, Y_plot, Z_plot,
                                                 color=ion_color,
                                                 alpha=surface_alpha,
                                                 linewidth=surface_linewidth or 0.1,
                                                 antialiased=surface_antialiased)
                            print(f"    ⚠️ Used solid color (insufficient coord range)")
                    
                    elif color_scheme == 'height_gradient':
                        # Color based on height values
                        from matplotlib.colors import Normalize
                        height_min, height_max = np.nanmin(Z_plot), np.nanmax(Z_plot)
                        norm = Normalize(vmin=height_min, vmax=height_max)
                        
                        if surface_style == 'wireframe':
                            surf = ax.plot_wireframe(X_plot, Y_plot, Z_plot,
                                                    color=ion_color,
                                                    alpha=surface_alpha,
                                                    linewidth=surface_linewidth or 1.0)
                        else:
                            surf = ax.plot_surface(X_plot, Y_plot, Z_plot,
                                                 cmap=surface_cmap,
                                                 alpha=surface_alpha,
                                                 linewidth=surface_linewidth,
                                                 antialiased=surface_antialiased,
                                                 norm=norm,
                                                 shade=surface_lighting)
                        print(f"    ✅ Applied height gradient: {height_min:.1f}-{height_max:.1f} Å")
                    
                    else:  # 'adaptive' - solid color
                        if surface_style == 'wireframe':
                            surf = ax.plot_wireframe(X_plot, Y_plot, Z_plot,
                                                    color=ion_color,
                                                    alpha=surface_alpha,
                                                    linewidth=surface_linewidth or 1.0)
                        else:
                            surf = ax.plot_surface(X_plot, Y_plot, Z_plot,
                                                 color=ion_color,
                                                 alpha=surface_alpha,
                                                 linewidth=surface_linewidth,
                                                 antialiased=surface_antialiased)
                        print(f"    ✅ Applied solid color: {ion_color}")
                    
                    ion_plot_surfaces.append(surf)
            
            # Set up the axis
            ax.set_xlabel(f'X (Å)', fontsize=label_fontsize)
            ax.set_ylabel(f'Y (Å)', fontsize=label_fontsize) 
            ax.set_zlabel(f'Height (Å)', fontsize=label_fontsize)
            ax.set_title(f'{ion_type} Coordination Shells\n({plot_mode.replace("_", " ").title()})', 
                        fontsize=title_fontsize)
            
            # Set view angle
            if view_angle:
                ax.view_init(elev=view_angle[0], azim=view_angle[1])
            
            plot_results['surfaces'][ion_type] = ion_plot_surfaces
        
        # Save the plot
        if save_plots:
            save_path = f"{filename_prefix}_{plot_mode}_coordination.png"
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"💾 Plot saved: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        # Add coordination-specific metadata to results
        plot_results['coordination_metadata'] = {
            'plot_mode': plot_mode,
            'surfaces_plotted': surfaces_to_plot,
            'surface_labels': surface_labels if 'surface_labels' in locals() else surfaces_to_plot,
            'plot_dimension': plot_dimension,
            'height_scale_factor': height_scale_factor,
            'base_height': base_height,
            'available_shells': available_shells,
            'z_layers': list(layer_data.keys()) if layer_data is not None else []
        }
        
        # Update title information for coordination context
        for ax in plot_results['axes']:
            if show_title:
                current_title = ax.get_title()
                # Replace "PMF" with "Coordination" in titles
                coordination_title = current_title.replace('PMF Height Surfaces', f'Coordination Height Surfaces ({plot_mode.title()})')
                ax.set_title(coordination_title, fontsize=title_fontsize, fontweight='bold')
        
        # Print coordination-specific summary
        print(f"\n📊 Coordination Height Surface Plotting Summary:")
        print(f"{'='*60}")
        print(f"  Plot mode: {plot_mode} ({plot_dimension})")
        print(f"  Visualization mode: {visualization_mode}")
        print(f"  Surface style: {surface_style}")
        print(f"  Color scheme: {color_scheme}")
        print(f"  Ion types: {len(ion_types)} ({', '.join(ion_types)})")
        print(f"  Surfaces per ion: {len(surfaces_to_plot)}")
        print(f"  Total surfaces plotted: {sum(len(surfaces) for surfaces in plot_results['surfaces'].values())}")
        print(f"  Height scaling: {height_scale_factor}x + {base_height}")
        print(f"  Coordinate range: X=[0, {x_display[-1]:.1f}], Y=[0, {y_display[-1]:.1f}]")
        
        # Advanced features summary
        if surface_smoothing:
            print(f"\n🎯 Surface Smoothing Applied:")
            print(f"   Method: {smoothing_method}")
            if smoothing_method == 'gaussian':
                print(f"   Gaussian sigma: {surface_smoothing_sigma}")
            elif smoothing_method == 'median':
                print(f"   Median filter size: {median_filter_size}")
            elif smoothing_method == 'bilateral':
                print(f"   Bilateral sigmas: spatial={bilateral_sigma_spatial}, intensity={bilateral_sigma_intensity}")
            elif smoothing_method == 'savgol':
                print(f"   Savitzky-Golay: window={savgol_window_length}, order={savgol_polyorder}")
            
            if grid_upsampling_factor > 1:
                print(f"   Grid upsampling: {grid_upsampling_factor}x")
            
            if adaptive_smoothing:
                print(f"   Adaptive smoothing: enabled (min_points={min_data_points_for_smoothing})")
        else:
            print(f"\n🎯 Surface Smoothing: DISABLED (raw surfaces)")
        
        # NaN handling summary
        if handling_nan != 'default':
            print(f"\n🎯 NaN Handling Applied:")
            print(f"   Method: {handling_nan}")
            if handling_nan == 'mask':
                print(f"   NaN color: {nan_color} (α={nan_alpha})")
            elif handling_nan == 'hatched':
                print(f"   Hatch pattern: {hatch_pattern} (density={hatch_density})")
            elif handling_nan == 'transparent':
                print(f"   NaN regions: fully transparent")
        else:
            print(f"\n🎯 NaN Handling: DEFAULT (matplotlib default)")
        
        # Outlier detection summary
        if outlier_detection:
            print(f"\n🎯 Outlier Detection Applied:")
            print(f"   Method: {outlier_method} (threshold={outlier_threshold})")
            print(f"   Correction: {outlier_correction}")
            print(f"   Statistical method: {statistical_outlier_method}")
            if iterative_refinement:
                print(f"   Iterative refinement: {iterative_max_iterations} iterations (decay={iterative_threshold_decay})")
        
        # Interpolation summary
        if fill_missing_data:
            print(f"\n🎯 Spatial Interpolation Applied:")
            print(f"   Method: {interpolation_method}")
            print(f"   Max neighbor cells: {max_neighbor_cells}")
            print(f"   Min neighbors required: {min_neighbors_required}")
        
        # Colorbar scaling summary
        print(f"\n🎨 Colorbar Scaling: {colorbar_scaling}")
        if colorbar_vmin is not None or colorbar_vmax is not None:
            print(f"   Custom range: [{colorbar_vmin}, {colorbar_vmax}]")
        
        # Plot mode explanation
        if plot_mode == 'shells':
            print(f"\n💡 Shell-focused: Each surface shows a coordination shell aggregated across Z-layers")
        elif plot_mode == 'z_slices':
            print(f"\n💡 Z-slice-focused: Each surface shows total coordination at a Z-layer across all shells")
        
        print(f"{'='*60}")
        
        return plot_results


    def debug_coordination_data_availability(self):
        """
        Debug helper to check what coordination data is available in analysis results
        """
        print("🔍 DEBUGGING COORDINATION DATA AVAILABILITY")
        print("="*60)
        
        if not hasattr(self, 'analysis'):
            print("❌ No analysis object found in plotter")
            return None
        
        if not hasattr(self.analysis, 'results'):
            print("❌ No analysis.results found")
            return None
        
        print("✓ Analysis results found")
        
        # List all available results
        results_attrs = dir(self.analysis.results)
        results_attrs = [attr for attr in results_attrs if not attr.startswith('_')]
        print(f"\n📋 Available results attributes ({len(results_attrs)}):")
        
        coordination_candidates = []
        
        for attr in sorted(results_attrs):
            value = getattr(self.analysis.results, attr)
            if value is not None:
                print(f"  ✓ {attr}: {type(value)}")
                
                # Check if this might be coordination data
                if isinstance(value, dict):
                    if 'layer_data' in value:
                        print(f"    🎯 Contains 'layer_data' - LIKELY COORDINATION DATA!")
                        coordination_candidates.append(attr)
                        
                        # Examine layer_data structure
                        layer_data = value['layer_data']
                        print(f"    📊 Layer data keys: {list(layer_data.keys())}")
                        
                        # Check first layer
                        if layer_data:
                            first_key = list(layer_data.keys())[0]
                            first_layer = layer_data[first_key]
                            print(f"    🔍 First layer ({first_key}) keys: {list(first_layer.keys())}")
                            
                            # Check coordination_grids
                            if 'coordination_grids' in first_layer:
                                coord_grids = first_layer['coordination_grids']
                                print(f"    ✅ Coordination grids found:")
                                for shell_type in coord_grids:
                                    ions = list(coord_grids[shell_type].keys())
                                    print(f"      {shell_type}: {ions}")
                            else:
                                print(f"    ⚠️ No 'coordination_grids' in first layer")
                    
                    elif any(key in str(value).lower() for key in ['coord', 'shell', 'spatial']):
                        print(f"    🤔 Might contain coordination data - checking...")
            else:
                print(f"  ✗ {attr}: None")
        
        print(f"\n🎯 SUMMARY:")
        if coordination_candidates:
            print(f"✅ Found {len(coordination_candidates)} potential coordination data sources:")
            for candidate in coordination_candidates:
                print(f"  - analysis.results.{candidate}")
            print(f"\n💡 Try using: coordination_surface_results=analysis.results.{coordination_candidates[0]}")
            return coordination_candidates[0]
        else:
            print("❌ No coordination data found")
            print("\n🔧 Troubleshooting:")
            print("1. Make sure calculate_solvation_shells_vs_z_detailed() completed successfully")
            print("2. Check that the results were saved to analysis.results")
            print("3. Look for any error messages during calculation")
            return None


    def plot_pmf_with_bootstrap_comparison(self, pmf_results, bootstrap_results, 
                                        comparison_type='original_vs_bootstrap',
                                        show_individual_curves=False,
                                        n_bootstrap_curves=20,
                                        units='kT'):
        """
        Create comparison plots showing original PMF vs bootstrap mean with uncertainty.
        """
        
        print(f"\n📊 Creating PMF Bootstrap Comparison ({comparison_type})")
        
        ion_types = list(pmf_results.keys())
        n_ions = len(ion_types)
        
        # Create figure with proper subplot arrangement
        fig, axes = plt.subplots(n_ions, 2, figsize=(15, 6*n_ions))
        
        # Handle axes indexing correctly for single ion case
        if n_ions == 1:
            axes = axes.reshape(1, -1)
        
        for idx, ion_type in enumerate(ion_types):
            # Get the correct axes for this ion
            if n_ions == 1:
                ax1 = axes[0, 0]
                ax2 = axes[0, 1]
            else:
                ax1 = axes[idx, 0]
                ax2 = axes[idx, 1]
            
            # Check if we have PMF data for this ion
            if ion_type not in pmf_results:
                ax1.text(0.5, 0.5, f'No PMF data for {ion_type}', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax2.text(0.5, 0.5, f'No PMF data for {ion_type}', 
                        ha='center', va='center', transform=ax2.transAxes)
                continue
            
            # Get original PMF data
            pmf_data = pmf_results[ion_type]
            original_z = pmf_data['z_centers']
            original_pmf = pmf_data['pmf']
            
            print(f"   📈 {ion_type} - Original PMF: z range {original_z.min():.1f} to {original_z.max():.1f}, PMF range {original_pmf.min():.3f} to {original_pmf.max():.3f}")
            
            # Bootstrap data with key mapping
            if ion_type in bootstrap_results:
                boot_data = bootstrap_results[ion_type]
                
                print(f"   🔧 {ion_type} - Available bootstrap keys: {list(boot_data.keys())}")
                
                # Get z-coordinates (try both possible keys)
                boot_z = boot_data.get('z_centers', original_z)  # Fallback to original z if not found
                
                # Try mapped keys first, then original keys
                boot_mean = (boot_data.get('bootstrap_mean') or 
                            boot_data.get('mean_pmf'))
                
                boot_std = (boot_data.get('bootstrap_std') or 
                        boot_data.get('pmf_std'))
                
                # Get confidence intervals
                ci_data = boot_data.get('confidence_intervals', {})
                ci_95_lower = None
                ci_95_upper = None
                
                if '95%' in ci_data:
                    ci_95_lower = ci_data['95%']['lower']
                    ci_95_upper = ci_data['95%']['upper']
                elif boot_mean is not None and boot_std is not None:
                    # Calculate 95% CI from mean ± 1.96*std
                    ci_95_lower = boot_mean - 1.96 * boot_std
                    ci_95_upper = boot_mean + 1.96 * boot_std
                
                if boot_mean is None or boot_std is None:
                    print(f"   ❌ Missing bootstrap statistics for {ion_type}")
                    print(f"      Available keys: {list(boot_data.keys())}")
                    ax1.text(0.5, 0.5, f'Invalid bootstrap data for {ion_type}', 
                            ha='center', va='center', transform=ax1.transAxes)
                    ax2.text(0.5, 0.5, f'Invalid bootstrap data for {ion_type}', 
                            ha='center', va='center', transform=ax2.transAxes)
                    continue
                
                print(f"   📊 {ion_type} - Bootstrap: z shape {boot_z.shape}, mean shape {boot_mean.shape}")
                
                # FIXED: Plot original PMF first (make sure it's visible)
                ax1.plot(original_z, original_pmf, 'b-', linewidth=3, 
                        label='Original PMF', alpha=0.8, zorder=3)
                
                # FIXED: Plot bootstrap mean with different color and style
                ax1.plot(boot_z, boot_mean, 'r-', linewidth=2.5, 
                        label='Bootstrap Mean', alpha=0.9, zorder=2)
                
                # FIXED: Plot 95% confidence band
                if ci_95_lower is not None and ci_95_upper is not None:
                    ax1.fill_between(boot_z, ci_95_lower, ci_95_upper, alpha=0.3, 
                                color='red', label='95% CI', zorder=1)
                
                # Show individual bootstrap curves if requested
                if show_individual_curves:
                    bootstrap_pmfs = (boot_data.get('individual_pmfs') or 
                                    boot_data.get('bootstrap_pmfs'))
                    
                    if bootstrap_pmfs is not None:
                        n_curves = min(n_bootstrap_curves, bootstrap_pmfs.shape[0])
                        curve_indices = np.random.choice(bootstrap_pmfs.shape[0], 
                                                    n_curves, replace=False)
                        
                        for i, curve_idx in enumerate(curve_indices):
                            alpha = 0.1 if n_curves > 10 else 0.3
                            ax1.plot(boot_z, bootstrap_pmfs[curve_idx], 'gray', 
                                    alpha=alpha, linewidth=0.5, zorder=0,
                                    label='Bootstrap samples' if i == 0 else "")
                
                # Debug: Print data ranges to verify they're different
                print(f"   🔍 {ion_type} - Original PMF range: {original_pmf.min():.3f} to {original_pmf.max():.3f}")
                print(f"   🔍 {ion_type} - Bootstrap mean range: {boot_mean.min():.3f} to {boot_mean.max():.3f}")
                
            else:
                # Only original PMF available
                ax1.plot(original_z, original_pmf, 'b-', linewidth=3, 
                        label=f'{ion_type} (No bootstrap)', alpha=0.8)
                print(f"   ⚠️  {ion_type} - No bootstrap data available")
            
            # FIXED: Set axis properties and legend
            ax1.set_xlabel('z (Å)')
            ax1.set_ylabel(f'PMF ({units})')
            ax1.set_title(f'{ion_type}: Original vs Bootstrap PMF')
            ax1.legend(loc='best', framealpha=0.8)
            ax1.grid(True, alpha=0.3)
            ax1.axvline(0, color='k', linestyle='--', alpha=0.5)
            
            # Set consistent y-limits for better comparison
            if ion_type in bootstrap_results and boot_mean is not None:
                all_values = np.concatenate([original_pmf, boot_mean])
                if ci_95_lower is not None:
                    all_values = np.concatenate([all_values, ci_95_lower, ci_95_upper])
                y_margin = (all_values.max() - all_values.min()) * 0.1
                ax1.set_ylim(all_values.min() - y_margin, all_values.max() + y_margin)
            
            # Right plot: Uncertainty analysis
            if ion_type in bootstrap_results and boot_std is not None:
                # Plot uncertainty (standard deviation)
                ax2.plot(boot_z, boot_std, 'purple', linewidth=2, 
                        label='Uncertainty (σ)')
                ax2.fill_between(boot_z, 0, boot_std, alpha=0.3, color='purple')
                
                # Plot relative uncertainty on twin axis
                ax2_twin = ax2.twinx()
                # Avoid division by zero
                relative_uncertainty = np.where(np.abs(boot_mean) > 0.01,
                                            boot_std / np.abs(boot_mean) * 100,
                                            0)
                ax2_twin.plot(boot_z, relative_uncertainty, 'orange', 
                            linewidth=2, linestyle='--', 
                            label='Relative uncertainty (%)')
                
                # Highlight high uncertainty regions
                high_uncertainty_threshold = np.mean(boot_std) + np.std(boot_std)
                high_regions = boot_std > high_uncertainty_threshold
                
                if np.any(high_regions):
                    ax2.fill_between(boot_z, 0, boot_std, 
                                where=high_regions, alpha=0.6, 
                                color='red', interpolate=True,
                                label='High uncertainty')
                
                ax2.set_ylabel(f'Uncertainty ({units})')
                ax2_twin.set_ylabel('Relative uncertainty (%)')
                
                # Combine legends
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            else:
                ax2.text(0.5, 0.5, f'No bootstrap data for {ion_type}', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            ax2.set_xlabel('z (Å)')
            ax2.set_title(f'{ion_type}: Uncertainty Analysis')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        filename = f'pmf_bootstrap_comparison_{comparison_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        print(f"✅ Comparison plot saved as: {filename}")
        
        plt.show()
        return fig

   def create_pmf_height_surfaces(self, debug_results, ion_types=None,
                                 z_surface_values=None,
                                 z_spacing=2.0,
                                 colors=None,
                                 alphas=None,
                                 smoothing=True,
                                 smoothing_sigma=1.0,
                                 z_scale_factor=None,
                                 pmf_threshold_range=None,
                                 surface_interpolation='cubic',
                                 grid_upsampling_factor=1,
                                 nan_handling='interpolate',
                                 reference_level='bulk'):
        """
        Create smooth continuous height-based surfaces from PMF data.
        
        This method creates surfaces where PMF values are represented as height (Z-coordinate)
        instead of isosurfaces at fixed energy levels. Similar to electrostatic potential 
        surface plotting but using PMF data.
        
        Parameters
        ----------
        debug_results : dict
            Results from calculate_pmf_vs_z_from_ion_density() method
        ion_types : list, optional
            Ion types to process. If None, uses all available from debug_results
        z_surface_values : list, optional
            Z-slice values to create surfaces for. If None, uses all available slices
        z_spacing : float, default=2.0
            Vertical spacing between surface layers (Å)
        colors : list, optional
            Colors for each ion type. If None, auto-generates colors
        alphas : list, optional
            Transparency values for each ion type. If None, auto-generates
        smoothing : bool, default=True
            Apply Gaussian smoothing to surfaces for continuity
        smoothing_sigma : float, default=1.0
            Gaussian smoothing parameter (higher = more smoothing)
        z_scale_factor : float, optional
            Scale factor for PMF height representation. If None, auto-calculates
        pmf_threshold_range : tuple, optional
            (min_pmf, max_pmf) range to display. Values outside are clipped
        surface_interpolation : str, default='cubic'
            Interpolation method for smooth surfaces: 'linear', 'cubic', 'quintic'
        grid_upsampling_factor : int, default=1
            Factor to increase grid resolution through interpolation
        nan_handling : str, default='interpolate'
            How to handle NaN values: 'interpolate', 'mask', 'zero'
        reference_level : str, default='bulk'
            Reference level for PMF heights: 'bulk' (PMF=0), 'minimum', 'custom'
        
        Returns
        -------
        dict
            Dictionary containing height surface data for plotting
        """
        
        print(f"\n🏔️ CREATING PMF HEIGHT SURFACES")
        print(f"{'='*60}")
        print(f"Surface type: Height-based (PMF as Z-coordinate)")
        print(f"Smoothing: {'Enabled' if smoothing else 'Disabled'}")
        print(f"Reference level: {reference_level}")
        
        # Import required modules
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter
        from scipy.interpolate import interp2d, griddata
        
        # Validate debug results structure
        if not debug_results or 'slab_results' not in debug_results:
            raise ValueError("Invalid debug_results. Must contain 'slab_results' from calculate_pmf_vs_z_from_ion_density()")
        
        slab_results = debug_results['slab_results']
        
        # Extract coordinate arrays
        x_centers = debug_results.get('x_centers')
        y_centers = debug_results.get('y_centers')
        
        if x_centers is None or y_centers is None:
            raise ValueError("Missing coordinate data in debug_results")
        
        print(f"✓ Found {len(slab_results)} z-slabs in debug results")
        print(f"✓ Grid dimensions: {len(x_centers)} × {len(y_centers)}")
        
        # Get available ion types and z-values
        available_ions = set()
        z_values = []
        
        for slab_idx, slab_data in slab_results.items():
            if slab_data:  # Check if slab_data is not empty
                # Get z-value from first available ion in this slab
                first_ion_key = next(iter(slab_data.keys()))
                if 'target_z' in slab_data[first_ion_key]:
                    z_values.append(slab_data[first_ion_key]['target_z'])
                
                # Collect all ion types
                for ion_type in slab_data.keys():
                    available_ions.add(ion_type)
        
        available_ions = list(available_ions)
        z_values = sorted(set(z_values))
        
        if not available_ions:
            raise ValueError("No ion types found in debug results")
        
        # Process ion types parameter
        if ion_types is None:
            ion_types = available_ions
        elif isinstance(ion_types, str):
            ion_types = [ion_types]
        
        # Validate ion types
        invalid_ions = [ion for ion in ion_types if ion not in available_ions]
        if invalid_ions:
            print(f"⚠️ Warning: Ion types {invalid_ions} not found in results")
        ion_types = [ion for ion in ion_types if ion in available_ions]
        
        if not ion_types:
            raise ValueError("No valid ion types to process")
        
        print(f"Processing ion types: {ion_types}")
        
        # Process z-surface values parameter
        if z_surface_values is None:
            z_surface_values = z_values
        else:
            # Find closest matching z-values
            matched_z_values = []
            for target_z in z_surface_values:
                closest_z = min(z_values, key=lambda z: abs(z - target_z))
                matched_z_values.append(closest_z)
            z_surface_values = matched_z_values
        
        print(f"Creating surfaces for z-values: {z_surface_values}")
        
        # Generate colors and alphas if not provided
        if colors is None:
            # Use same color scheme as other methods
            color_cycle = plt.cm.Set1(np.linspace(0, 1, max(8, len(ion_types))))
            colors = {ion_type: color_cycle[i % len(color_cycle)] 
                     for i, ion_type in enumerate(ion_types)}
        elif isinstance(colors, (list, tuple)):
            colors = {ion_type: colors[i % len(colors)] 
                     for i, ion_type in enumerate(ion_types)}
        
        if alphas is None:
            # Use gradual transparency based on number of surfaces
            base_alpha = 0.8
            alphas = {ion_type: base_alpha - 0.1 * i 
                     for i, ion_type in enumerate(ion_types)}
        elif isinstance(alphas, (list, tuple)):
            alphas = {ion_type: alphas[i % len(alphas)] 
                     for i, ion_type in enumerate(ion_types)}
        
        # Initialize results storage
        results = {
            'height_surfaces': {},
            'ion_types': ion_types,
            'z_surface_values': z_surface_values,
            'x_centers': x_centers,
            'y_centers': y_centers,
            'colors': colors,
            'alphas': alphas,
            'smoothing_applied': smoothing,
            'smoothing_sigma': smoothing_sigma if smoothing else None,
            'z_scale_factor': z_scale_factor,
            'reference_level': reference_level,
            'surface_type': 'height_based',
            'creation_timestamp': debug_results.get('timestamp', 'unknown')
        }
        
        # Transform coordinates to start from 0 (same as electrostatic method)
        x_display = x_centers - x_centers.min()
        y_display = y_centers - y_centers.min()
        
        # Create meshgrids for surface creation
        X, Y = np.meshgrid(x_display, y_display)
        
        # Determine PMF scaling
        if z_scale_factor is None:
            # Auto-calculate scale factor based on PMF range
            all_pmfs = []
            for slab_idx, slab_data in slab_results.items():
                for ion_type in ion_types:
                    if ion_type in slab_data and 'pmf_grid' in slab_data[ion_type]:
                        pmf_grid = slab_data[ion_type]['pmf_grid']
                        if pmf_grid is not None:
                            finite_pmfs = pmf_grid[np.isfinite(pmf_grid)]
                            all_pmfs.extend(finite_pmfs)
            
            if all_pmfs:
                pmf_range = np.max(all_pmfs) - np.min(all_pmfs)
                # Scale to reasonable height range (e.g., 10-20 Å)
                target_height_range = 15.0  # Å
                z_scale_factor = target_height_range / pmf_range if pmf_range > 0 else 1.0
            else:
                z_scale_factor = 1.0
        
        results['z_scale_factor'] = z_scale_factor
        print(f"✓ Z-scale factor: {z_scale_factor:.3f} (Å per kJ/mol)")
        
        # Process each ion type
        for ion_idx, ion_type in enumerate(ion_types):
            print(f"\n🔹 Processing {ion_type}:")
            
            ion_surfaces = {}
            ion_color = colors[ion_type]
            ion_alpha = alphas[ion_type]
            
            # Process each z-surface
            for surf_idx, target_z in enumerate(z_surface_values):
                # Find corresponding slab
                slab_idx = None
                for s_idx, slab_data in slab_results.items():
                    if ion_type in slab_data:
                        slab_target_z = slab_data[ion_type].get('target_z')
                        if slab_target_z is not None and abs(slab_target_z - target_z) < 0.1:
                            slab_idx = s_idx
                            break
                
                if slab_idx is None:
                    print(f"  ⚠️ No data found for z = {target_z:.1f} Å")
                    continue
                
                ion_result = slab_results[slab_idx][ion_type]
                pmf_grid = ion_result.get('pmf_grid')  # Fixed: singular 'pmf_grid' not 'pmf_grids'
                
                if pmf_grid is None or pmf_grid.size == 0:
                    print(f"  ⚠️ No PMF grid data for z = {target_z:.1f} Å")
                    continue
                
                # Handle NaN values based on method
                if nan_handling == 'interpolate' and np.any(np.isnan(pmf_grid)):
                    # Interpolate NaN values
                    valid_mask = ~np.isnan(pmf_grid)
                    if np.any(valid_mask):
                        valid_coords = np.column_stack(np.where(valid_mask))
                        valid_values = pmf_grid[valid_mask]
                        
                        # Create coordinate grid for interpolation
                        yi, xi = np.mgrid[0:pmf_grid.shape[0], 0:pmf_grid.shape[1]]
                        
                        try:
                            pmf_grid = griddata(valid_coords, valid_values, (yi, xi), 
                                              method='nearest', fill_value=0.0)
                        except Exception as e:
                            print(f"    ⚠️ Interpolation failed: {e}, using original grid")
                elif nan_handling == 'zero':
                    pmf_grid = np.nan_to_num(pmf_grid, nan=0.0)
                elif nan_handling == 'mask':
                    # Keep NaN values as is (will be handled during plotting)
                    pass
                
                # Apply smoothing if requested
                if smoothing:
                    # Only smooth finite values
                    finite_mask = np.isfinite(pmf_grid)
                    if np.any(finite_mask):
                        pmf_grid_smooth = gaussian_filter(pmf_grid, sigma=smoothing_sigma)
                        # Restore NaN values if nan_handling is 'mask'
                        if nan_handling == 'mask':
                            pmf_grid_smooth[~finite_mask] = np.nan
                        pmf_grid = pmf_grid_smooth
                    else:
                        print(f"    ⚠️ No finite values to smooth for z = {target_z:.1f} Å")
                
                # Apply PMF threshold range if specified
                if pmf_threshold_range is not None:
                    min_thresh, max_thresh = pmf_threshold_range
                    pmf_grid = np.clip(pmf_grid, min_thresh, max_thresh)
                
                # Calculate surface height (base_z + PMF scaling)
                base_z = target_z + surf_idx * z_spacing
                height_surface = base_z + pmf_grid * z_scale_factor
                
                # Apply grid upsampling if requested
                if grid_upsampling_factor > 1:
                    try:
                        # Use RegularGridInterpolator instead of deprecated interp2d
                        from scipy.interpolate import RegularGridInterpolator
                        
                        # Create interpolator
                        interpolator = RegularGridInterpolator(
                            (y_display, x_display), height_surface,
                            method='cubic' if surface_interpolation == 'cubic' else 'linear',
                            bounds_error=False, fill_value=np.nan
                        )
                        
                        # Create higher resolution grids
                        x_hr = np.linspace(x_display.min(), x_display.max(), 
                                         len(x_display) * grid_upsampling_factor)
                        y_hr = np.linspace(y_display.min(), y_display.max(),
                                         len(y_display) * grid_upsampling_factor)
                        
                        X_hr, Y_hr = np.meshgrid(x_hr, y_hr)
                        
                        # Create coordinate pairs for interpolation
                        coords = np.column_stack([Y_hr.ravel(), X_hr.ravel()])
                        height_surface_hr = interpolator(coords).reshape(Y_hr.shape)
                        
                        surface_data = {
                            'X': X_hr, 'Y': Y_hr, 'Z': height_surface_hr,
                            'upsampled': True
                        }
                    except Exception as e:
                        print(f"    ⚠️ Upsampling failed: {e}, using original resolution")
                        surface_data = {
                            'X': X, 'Y': Y, 'Z': height_surface,
                            'upsampled': False
                        }
                else:
                    surface_data = {
                        'X': X, 'Y': Y, 'Z': height_surface,
                        'upsampled': False
                    }
                
                # Store surface information
                surface_info = {
                    'original_z': target_z,
                    'base_z': base_z,
                    'surface_data': surface_data,
                    'pmf_grid': pmf_grid,
                    'color': ion_color,
                    'alpha': ion_alpha,
                    'pmf_stats': {
                        'min': np.nanmin(pmf_grid) if np.any(np.isfinite(pmf_grid)) else np.nan,
                        'max': np.nanmax(pmf_grid) if np.any(np.isfinite(pmf_grid)) else np.nan,
                        'mean': np.nanmean(pmf_grid) if np.any(np.isfinite(pmf_grid)) else np.nan,
                        'finite_count': np.sum(np.isfinite(pmf_grid))
                    },
                    'height_stats': {
                        'min': np.nanmin(height_surface) if np.any(np.isfinite(height_surface)) else np.nan,
                        'max': np.nanmax(height_surface) if np.any(np.isfinite(height_surface)) else np.nan,
                        'range': (np.nanmax(height_surface) - np.nanmin(height_surface)) if np.any(np.isfinite(height_surface)) else 0.0
                    }
                }
                
                ion_surfaces[target_z] = surface_info
                
                finite_points = surface_info['pmf_stats']['finite_count']
                pmf_range = surface_info['pmf_stats']['max'] - surface_info['pmf_stats']['min']
                height_range = surface_info['height_stats']['range']
                
                print(f"  ✓ z = {target_z:5.1f} Å: {finite_points} points, "
                      f"PMF range = {pmf_range:.2f} kJ/mol, height range = {height_range:.2f} Å")
            
            # Store results for this ion type
            results['height_surfaces'][ion_type] = {
                'surfaces': ion_surfaces,
                'color': ion_color,
                'alpha': ion_alpha,
                'n_surfaces': len(ion_surfaces),
                'z_range': (min(ion_surfaces.keys()), max(ion_surfaces.keys())) if ion_surfaces else (0, 0)
            }
            
            print(f"  ✅ Created {len(ion_surfaces)} height surfaces for {ion_type}")
        
        print(f"\n✅ PMF HEIGHT SURFACES COMPLETE!")
        print(f"{'='*60}")
        
        total_surfaces = sum(len(data['surfaces']) for data in results['height_surfaces'].values())
        print(f"📊 Summary:")
        print(f"  Ion types processed: {len(ion_types)}")
        print(f"  Total height surfaces: {total_surfaces}")
        print(f"  Smoothing applied: {'Yes' if smoothing else 'No'}")
        print(f"  Grid upsampling: {grid_upsampling_factor}x")
        print(f"  Z-scale factor: {z_scale_factor:.3f} Å/(kJ/mol)")
        
        print(f"\n💡 Height Surface Interpretation:")
        print(f"  → Surface height = Base Z + (PMF × scale_factor)")
        print(f"  → Higher peaks: Less favorable binding (higher PMF)")
        print(f"  → Lower valleys: More favorable binding (lower PMF)")
        print(f"  → Continuous surface: Smooth spatial variation of binding affinity")
        
        return results




    def plot_pmf_height_surfaces(self, height_surface_results=None, ion_types=None,
                                surface_indices=None,
                                visualization_mode='3d_surfaces',
                                surface_style='smooth',
                                color_scheme='pmf_gradient',
                                show_colorbar=True,
                                colorbar_label='PMF Height (Å)',
                                colorbar_vmin=None,
                                colorbar_vmax=None,
                                view_angle=(30, 45),
                                lighting_style='default',
                                show_reference_planes=False,
                                reference_plane_alpha=0.2,
                                clay_overlay=False,
                                save_plots=True,
                                figsize=(15, 12),
                                dpi=300,
                                filename_prefix='pmf_height_surfaces',
                                # Spatial interpolation parameters
                                fill_missing_data=True,
                                max_neighbor_cells=3,
                                interpolation_method='simple_average',
                                min_neighbors_required=2,
                                interpolation_max_distance=None,
                                # Publication settings
                                title_fontsize=14,
                                label_fontsize=12,
                                tick_fontsize=10,
                                legend_fontsize=12,
                                colorbar_tick_fontsize=10,
                                show_title=True,
                                show_legend=False,
                                # Publication figure control parameters
                                save_individual_figures=False,
                                individual_figsize=(8, 6),
                                save_combined_figure=False,
                                show_individual_figures=False,
                                show_combined_figure=True,
                                # Colorbar scaling options
                                colorbar_scaling='individual',
                                # Surface rendering parameters
                                surface_alpha=0.8,
                                surface_cmap='RdYlBu_r',
                                surface_lighting=True,
                                surface_linewidth=0,
                                surface_antialiased=True,
                                surface_rstride=1,
                                surface_cstride=1,
                                # Surface smoothing parameters
                                surface_smoothing=False,
                                surface_smoothing_sigma=1.5,
                                smoothing_method='gaussian',
                                grid_upsampling_factor=1):
        """
        Plot PMF height surfaces where PMF values are represented as surface height.
        
        This method creates 3D surface plots where the Z-coordinate represents PMF values,
        similar to topographical maps but for energy landscapes.
        
        Parameters
        ----------
        height_surface_results : dict, optional
            Results from create_pmf_height_surfaces(). If None, attempts to load from analysis results
        ion_types : list, optional
            Ion types to plot. If None, plots all available
        surface_indices : list, optional
            Indices of surfaces to plot for each ion. If None, plots all surfaces
        visualization_mode : str, default='3d_surfaces'
            Plotting mode:
            - '3d_surfaces': 3D surface plots
            - '2d_contours': 2D contour projections
            - 'combined': Both 3D and 2D views
        surface_style : str, default='smooth'
            Surface rendering style:
            - 'smooth': Smooth continuous surfaces
            - 'wireframe': Wireframe surfaces
            - 'filled_contour': Filled contour surfaces
        color_scheme : str, default='pmf_gradient'
            Coloring scheme:
            - 'pmf_gradient': Color based on PMF values (RECOMMENDED - shows energy landscape)
            - 'height_gradient': Color based on surface height values
            - 'adaptive': Use solid colors from height surface results (single color per ion)
        show_colorbar : bool, default=True
            Show colorbar for height/PMF values
        colorbar_label : str, default='PMF Height (Å)'
            Label for the colorbar
        colorbar_vmin, colorbar_vmax : float, optional
            Colorbar range limits
        view_angle : tuple, default=(30, 45)
            3D viewing angle (elevation, azimuth)
        lighting_style : str, default='default'
            3D lighting setup
        show_reference_planes : bool, default=True
            Show reference planes (z=0, bulk level)
        reference_plane_alpha : float, default=0.2
            Transparency of reference planes
        clay_overlay : bool, default=False
            Overlay clay structure if available
        save_plots : bool, default=True
            Save plots to files
        figsize : tuple, default=(15, 12)
            Figure size
        dpi : int, default=300
            Resolution for saved plots
        filename_prefix : str, default='pmf_height_surfaces'
            Prefix for saved filenames
        fill_missing_data : bool, default=True
            Fill missing/infinite PMF data using spatial interpolation
        max_neighbor_cells : int, default=3
            Maximum number of neighbor cells to consider for averaging
        interpolation_method : str, default='simple_average'
            Method for spatial interpolation:
            - 'simple_average': Simple mean of valid neighbors
            - 'distance_weighted': Distance-weighted averaging
            - 'gaussian_kernel': Gaussian kernel averaging
        min_neighbors_required : int, default=2
            Minimum number of valid neighbors required for interpolation
        interpolation_max_distance : int, optional
            Maximum distance (in grid cells) for neighbor search.
            If None, uses max_neighbor_cells as radius
        colorbar_scaling : str, default='individual'
            Colorbar scaling strategy:
            - 'individual': Each surface gets its own colorbar range (best for similar PMF values)
            - 'global': All surfaces per ion use same range (previous behavior)
            - 'unified': All ions and surfaces use same global range
        surface_smoothing : bool, default=False
            Whether to apply smoothing to surfaces for better visual appearance
        surface_smoothing_sigma : float, default=1.5
            Gaussian smoothing parameter (higher = more smoothing)
        smoothing_method : str, default='gaussian'
            Smoothing method: 'gaussian', 'median', 'bilateral'
        grid_upsampling_factor : int, default=1
            Factor to increase grid resolution through interpolation before smoothing
        
        Publication Font Settings:
        title_fontsize : int, default=14
            Font size for plot titles
        label_fontsize : int, default=12
            Font size for axis labels and colorbar labels
        tick_fontsize : int, default=10
            Font size for tick labels
        legend_fontsize : int, default=12
            Font size for legends
        colorbar_tick_fontsize : int, default=10
            Font size for colorbar tick labels
        show_title : bool, default=True
            Whether to show plot titles
        show_legend : bool, default=False
            Whether to show legends for surfaces
        
        Publication Figure Control:
        save_individual_figures : bool, default=False
            Save each ion as a separate figure (for multi-ion plots)
        individual_figsize : tuple, default=(8, 6)
            Figure size for individual ion plots
        save_combined_figure : bool, default=True
            Save combined figure with all ions (for multi-ion plots)
        show_individual_figures : bool, default=False
            Display each ion as a separate figure
        show_combined_figure : bool, default=True
            Display combined figure with all ions
        
        Surface Rendering Options:
        surface_alpha : float, default=0.8
            Transparency of surfaces (0=transparent, 1=opaque)
        surface_cmap : str, default='RdYlBu_r'
            Colormap for surface coloring
        surface_lighting : bool, default=True
            Enable 3D lighting effects
        surface_linewidth : float, default=0
            Width of surface wireframe lines (0 for no wireframe)
        surface_antialiased : bool, default=True
            Enable surface antialiasing for smoother appearance
        surface_rstride, surface_cstride : int, default=1
            Stride for surface sampling (higher = lower resolution)
        
        Returns
        -------
        dict
            Dictionary containing plot objects and metadata
        """
        
        from scipy.ndimage import gaussian_filter, median_filter
        from scipy.interpolate import interp2d
        
        print(f"\n🏔️ PLOTTING PMF HEIGHT SURFACES")
        print(f"{'='*60}")
        print(f"Visualization mode: {visualization_mode}")
        print(f"Surface style: {surface_style}")
        print(f"Color scheme: {color_scheme}")
        print(f"Colorbar scaling: {colorbar_scaling}")
        if surface_smoothing:
            print(f"Surface smoothing: {smoothing_method} (σ={surface_smoothing_sigma}, upsampling={grid_upsampling_factor}x)")
        
        # Get height surface data
        if height_surface_results is None:
            if hasattr(self.analysis.results, 'pmf_height_surfaces'):
                height_surface_results = self.analysis.results.pmf_height_surfaces
            else:
                raise ValueError("No height surface results found. Run create_pmf_height_surfaces() first.")
        
        # Validate data structure
        if 'height_surfaces' not in height_surface_results:
            raise ValueError("Invalid height surface results structure. Missing 'height_surfaces' key.")
        
        height_surfaces = height_surface_results['height_surfaces']
        
        # Get ion types to plot
        available_ions = list(height_surfaces.keys())
        if ion_types is None:
            ion_types = available_ions
        elif isinstance(ion_types, str):
            ion_types = [ion_types]
        
        # Validate ion types
        invalid_ions = [ion for ion in ion_types if ion not in available_ions]
        if invalid_ions:
            print(f"⚠ Warning: Ion types {invalid_ions} not found in results")
        ion_types = [ion for ion in ion_types if ion in available_ions]
        
        if not ion_types:
            raise ValueError("No valid ion types to plot")
        
        print(f"Plotting ion types: {ion_types}")
        
        # Calculate unified PMF range if using unified scaling
        unified_pmf_min, unified_pmf_max = None, None
        if colorbar_scaling == 'unified':
            all_pmf_values = []
            for ion_type in ion_types:
                ion_data = height_surfaces[ion_type]
                for z_val, surface_info in ion_data['surfaces'].items():
                    pmf_grid = surface_info['pmf_grid']
                    finite_pmfs = pmf_grid[np.isfinite(pmf_grid)]
                    if len(finite_pmfs) > 0:
                        all_pmf_values.extend(finite_pmfs)
            
            if all_pmf_values:
                unified_pmf_min = np.min(all_pmf_values)
                unified_pmf_max = np.max(all_pmf_values)
                print(f"Unified PMF range: {unified_pmf_min:.3f} to {unified_pmf_max:.3f} kJ/mol")
            else:
                unified_pmf_min, unified_pmf_max = 0, 1
        
        # Transform coordinates to start from 0 (SAME AS ELECTROSTATIC METHOD)
        x_centers = height_surface_results['x_centers']
        y_centers = height_surface_results['y_centers']
        x_display = x_centers - x_centers.min()  # Start X from 0
        y_display = y_centers - y_centers.min()  # Start Y from 0
        
        print(f"Coordinate transformation: X=[0, {x_display[-1]:.1f}], Y=[0, {y_display[-1]:.1f}]")
        
        # Helper function for spatial interpolation
        def _fill_missing_pmf_data(pmf_grid, method=interpolation_method, 
                                  max_cells=max_neighbor_cells,
                                  min_neighbors=min_neighbors_required,
                                  max_distance=interpolation_max_distance):
            """
            Fill NaN/infinite PMF values using spatial averaging of neighbors
            """
            if not fill_missing_data:
                return pmf_grid.copy()
            
            # Create copy to modify
            filled_grid = pmf_grid.copy()
            
            # Find cells with missing data (NaN or infinite)
            missing_mask = ~np.isfinite(pmf_grid)
            
            if not np.any(missing_mask):
                return filled_grid  # No missing data
            
            missing_count = np.sum(missing_mask)
            total_cells = pmf_grid.size
            missing_percentage = (missing_count / total_cells) * 100
            
            print(f"      🔧 Filling {missing_count}/{total_cells} missing cells ({missing_percentage:.1f}%)")
            
            # Get grid dimensions
            ny, nx = pmf_grid.shape
            
            # Set search distance
            search_dist = max_distance if max_distance is not None else max_cells
            
            filled_count = 0
            
            # Process each missing cell
            for i, j in np.argwhere(missing_mask):
                # Define search bounds
                i_min = max(0, i - search_dist)
                i_max = min(ny, i + search_dist + 1)
                j_min = max(0, j - search_dist)
                j_max = min(nx, j + search_dist + 1)
                
                # Extract neighbor region
                neighbor_region = pmf_grid[i_min:i_max, j_min:j_max]
                valid_neighbors = neighbor_region[np.isfinite(neighbor_region)]
                
                # Skip if insufficient neighbors
                if len(valid_neighbors) < min_neighbors:
                    continue
                
                # Apply interpolation method
                if method == 'simple_average':
                    fill_value = np.mean(valid_neighbors)
                    
                elif method == 'distance_weighted':
                    # Calculate distances from center cell
                    distances = []
                    values = []
                    
                    for di in range(i_min, i_max):
                        for dj in range(j_min, j_max):
                            if np.isfinite(pmf_grid[di, dj]):
                                distance = np.sqrt((di - i)**2 + (dj - j)**2)
                                if distance > 0:  # Exclude self
                                    distances.append(distance)
                                    values.append(pmf_grid[di, dj])
                    
                    if distances:
                        weights = 1.0 / np.array(distances)
                        fill_value = np.average(values, weights=weights)
                    else:
                        fill_value = np.mean(valid_neighbors)
                        
                elif method == 'gaussian_kernel':
                    # Gaussian weighted average
                    sigma = search_dist / 3.0  # 3-sigma rule
                    weighted_sum = 0
                    weight_sum = 0
                    
                    for di in range(i_min, i_max):
                        for dj in range(j_min, j_max):
                            if np.isfinite(pmf_grid[di, dj]):
                                distance = np.sqrt((di - i)**2 + (dj - j)**2)
                                if distance <= search_dist:
                                    weight = np.exp(-0.5 * (distance / sigma)**2)
                                    weighted_sum += pmf_grid[di, dj] * weight
                                    weight_sum += weight
                    
                    if weight_sum > 0:
                        fill_value = weighted_sum / weight_sum
                    else:
                        fill_value = np.mean(valid_neighbors)
                
                else:
                    # Default to simple average
                    fill_value = np.mean(valid_neighbors)
                
                # Fill the missing cell
                filled_grid[i, j] = fill_value
                filled_count += 1
            
            fill_percentage = (filled_count / missing_count) * 100 if missing_count > 0 else 0
            print(f"         ✓ Filled {filled_count}/{missing_count} cells ({fill_percentage:.1f}%)")
            
            return filled_grid
        
        # Helper function for surface smoothing
        def _apply_surface_smoothing(surface_data, method=smoothing_method, 
                                   sigma=surface_smoothing_sigma,
                                   upsampling=grid_upsampling_factor):
            """
            Apply smoothing to surface data for better visual appearance
            """
            if not surface_smoothing:
                return surface_data
            
            X, Y, Z = surface_data['X'], surface_data['Y'], surface_data['Z']
            
            # Apply upsampling if requested
            if upsampling > 1:
                try:
                    # Get original grid dimensions
                    ny, nx = Z.shape
                    new_ny, new_nx = ny * upsampling, nx * upsampling
                    
                    # Create new coordinate arrays
                    x_orig = X[0, :]
                    y_orig = Y[:, 0]
                    x_new = np.linspace(x_orig.min(), x_orig.max(), new_nx)
                    y_new = np.linspace(y_orig.min(), y_orig.max(), new_ny)
                    
                    # Interpolate Z values
                    f = interp2d(x_orig, y_orig, Z, kind='cubic', fill_value=np.nan)
                    Z_upsampled = f(x_new, y_new)
                    
                    # Create new meshgrids
                    X_new, Y_new = np.meshgrid(x_new, y_new)
                    X, Y, Z = X_new, Y_new, Z_upsampled
                    
                    print(f"      🔍 Upsampled grid from {nx}x{ny} to {new_nx}x{new_ny}")
                except Exception as e:
                    print(f"      ⚠️ Upsampling failed: {e}, using original grid")
            
            # Apply smoothing
            try:
                if method == 'gaussian':
                    Z_smoothed = gaussian_filter(Z, sigma=sigma)
                    print(f"      🎯 Applied Gaussian smoothing (σ={sigma})")
                    
                elif method == 'median':
                    kernel_size = max(3, int(sigma * 2) + 1)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    Z_smoothed = median_filter(Z, size=kernel_size)
                    print(f"      🎯 Applied median filtering (kernel={kernel_size})")
                    
                elif method == 'bilateral':
                    # Simple bilateral-like filtering using Gaussian on both spatial and intensity
                    Z_smoothed = gaussian_filter(Z, sigma=sigma)
                    print(f"      🎯 Applied bilateral-style smoothing (σ={sigma})")
                    
                else:
                    Z_smoothed = gaussian_filter(Z, sigma=sigma)
                    print(f"      🎯 Applied default Gaussian smoothing (σ={sigma})")
                    
            except Exception as e:
                print(f"      ⚠️ Smoothing failed: {e}, using original data")
                Z_smoothed = Z
            
            return {'X': X, 'Y': Y, 'Z': Z_smoothed}
        
        # Determine subplot layout
        n_ions = len(ion_types)
        if n_ions == 1:
            nrows, ncols = 1, 1
        elif n_ions == 2:
            nrows, ncols = 1, 2
        elif n_ions <= 4:
            nrows, ncols = 2, 2
        else:
            ncols = int(np.ceil(np.sqrt(n_ions)))
            nrows = int(np.ceil(n_ions / ncols))
        
        # Create figure with 3D subplots
        fig = plt.figure(figsize=figsize)
        
        # Initialize plot results
        plot_results = {
            'figure': fig,
            'axes': [],
            'surfaces': {},
            'ion_types': ion_types,
            'visualization_mode': visualization_mode,
            'surface_style': surface_style,
            'metadata': height_surface_results.copy()
        }
        
        # Set up viewing angle
        elev, azim = view_angle
        
        # Plot each ion type
        for idx, ion_type in enumerate(ion_types):
            print(f"\n🔹 Plotting {ion_type}:")
            
            # Create 3D subplot
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
            plot_results['axes'].append(ax)
            
            ion_data = height_surfaces[ion_type]
            ion_surfaces_data = ion_data['surfaces']
            ion_color = ion_data['color']
            ion_alpha = ion_data['alpha']
            
            # Determine which surfaces to plot
            if surface_indices is None:
                surfaces_to_plot = list(ion_surfaces_data.keys())
            else:
                available_indices = list(ion_surfaces_data.keys())
                surfaces_to_plot = [available_indices[i] for i in surface_indices 
                                  if i < len(available_indices)]
            
            print(f"  Plotting {len(surfaces_to_plot)} surfaces: {surfaces_to_plot}")
            
            # Calculate colorbar range based on scaling strategy
            if colorbar_scaling == 'unified':
                # Use unified range for all ions and surfaces
                ion_colorbar_vmin = colorbar_vmin if colorbar_vmin is not None else unified_pmf_min
                ion_colorbar_vmax = colorbar_vmax if colorbar_vmax is not None else unified_pmf_max
                print(f"  Using unified PMF range: {ion_colorbar_vmin:.3f} to {ion_colorbar_vmax:.3f} kJ/mol")
                
            elif colorbar_scaling == 'global':
                # Calculate global range for this ion type (all surfaces)
                ion_pmf_values = []
                for z_val, surface_info in ion_surfaces_data.items():
                    pmf_grid = surface_info['pmf_grid']
                    finite_pmfs = pmf_grid[np.isfinite(pmf_grid)]
                    if len(finite_pmfs) > 0:
                        ion_pmf_values.extend(finite_pmfs)
                
                if ion_pmf_values:
                    ion_pmf_min = np.min(ion_pmf_values)
                    ion_pmf_max = np.max(ion_pmf_values)
                    ion_colorbar_vmin = colorbar_vmin if colorbar_vmin is not None else ion_pmf_min
                    ion_colorbar_vmax = colorbar_vmax if colorbar_vmax is not None else ion_pmf_max
                else:
                    ion_colorbar_vmin = colorbar_vmin if colorbar_vmin is not None else 0
                    ion_colorbar_vmax = colorbar_vmax if colorbar_vmax is not None else 1
                
                print(f"  Global PMF range for {ion_type}: {ion_colorbar_vmin:.3f} to {ion_colorbar_vmax:.3f} kJ/mol")
                
            # For 'individual' scaling, we'll calculate per surface below
            
            ion_plot_surfaces = []
            
            # Plot each surface for this ion
            for surf_z in surfaces_to_plot:
                surface_info = ion_surfaces_data[surf_z]
                surface_data = surface_info['surface_data']
                
                # Calculate individual surface colorbar range if needed
                if colorbar_scaling == 'individual':
                    pmf_grid = surface_info['pmf_grid']
                    finite_pmfs = pmf_grid[np.isfinite(pmf_grid)]
                    
                    if len(finite_pmfs) > 0:
                        surf_pmf_min = np.min(finite_pmfs)
                        surf_pmf_max = np.max(finite_pmfs)
                        surf_colorbar_vmin = colorbar_vmin if colorbar_vmin is not None else surf_pmf_min
                        surf_colorbar_vmax = colorbar_vmax if colorbar_vmax is not None else surf_pmf_max
                    else:
                        surf_colorbar_vmin = colorbar_vmin if colorbar_vmin is not None else 0
                        surf_colorbar_vmax = colorbar_vmax if colorbar_vmax is not None else 1
                    
                    z_label = f"{surf_z:.1f} Å" if isinstance(surf_z, (int, float)) else str(surf_z)
                    print(f"    {z_label} individual PMF range: {surf_colorbar_vmin:.3f} to {surf_colorbar_vmax:.3f} kJ/mol")
                else:
                    # Use ion-level or unified range
                    surf_colorbar_vmin = ion_colorbar_vmin
                    surf_colorbar_vmax = ion_colorbar_vmax
                
                # Use transformed coordinates - reconstruct X,Y with transformed coordinates
                X_orig, Y_orig = surface_data['X'], surface_data['Y']
                X = X_orig - x_centers.min()  # Apply same transformation as x_display
                Y = Y_orig - y_centers.min()  # Apply same transformation as y_display
                Z_original = surface_data['Z']
                
                # Apply spatial interpolation to fill missing data
                if fill_missing_data:
                    z_label = f"{surf_z:.1f} Å" if isinstance(surf_z, (int, float)) else str(surf_z)
                    print(f"    🔧 Processing surface {z_label} for spatial interpolation")
                    
                    # Extract PMF grid from surface height
                    base_z = surface_info['base_z']
                    z_scale = height_surface_results.get('z_scale_factor', 1.0)
                    
                    # Recover PMF grid from height surface
                    if z_scale != 0:
                        pmf_grid_recovered = (Z_original - base_z) / z_scale
                    else:
                        pmf_grid_recovered = Z_original - base_z
                    
                    # Fill missing PMF data
                    filled_pmf_grid = _fill_missing_pmf_data(pmf_grid_recovered)
                    
                    # Recreate height surface from filled PMF data
                    Z = base_z + filled_pmf_grid * z_scale
                else:
                    Z = Z_original
                
                # Apply surface smoothing AFTER spatial interpolation for complete coverage
                if surface_smoothing:
                    z_label = f"{surf_z:.1f} Å" if isinstance(surf_z, (int, float)) else str(surf_z)
                    print(f"    🎨 Applying surface smoothing to complete {z_label} surface")
                    smoothed_data = _apply_surface_smoothing({
                        'X': X, 'Y': Y, 'Z': Z
                    })
                    X, Y, Z = smoothed_data['X'], smoothed_data['Y'], smoothed_data['Z']
                
                # Handle color scheme - COPY THE WORKING PATTERN FROM plot_pmf_3d_spatial_distribution_vs_z()
                if color_scheme == 'coordination_gradient':
                    # Color based on coordination values (NEW COORDINATION-SPECIFIC SCHEME)
                    # Note: In coordination surfaces, coordination data is stored as 'pmf_grid' for compatibility
                    if 'pmf_grid' in surface_info:
                        coordination_for_coloring = surface_info['pmf_grid']  # This contains coordination values
                        print(f"      Using coordination values for coloring: min={np.nanmin(coordination_for_coloring):.3f}, max={np.nanmax(coordination_for_coloring):.3f}")
                    else:
                        # Fallback: estimate coordination from height (height correlates with coordination)
                        # Higher surfaces = higher coordination numbers
                        base_z = surface_info['base_z']
                        height_diff = Z - base_z
                        # Scale height differences to approximate coordination range (0-8 typical)
                        coordination_for_coloring = 2.0 + 6.0 * (height_diff - np.nanmin(height_diff)) / (np.nanmax(height_diff) - np.nanmin(height_diff))
                        print(f"      Estimated coordination from height: min={np.nanmin(coordination_for_coloring):.3f}, max={np.nanmax(coordination_for_coloring):.3f}")
                    
                    # Apply coordination normalization for colormap
                    coord_min = np.nanmin(coordination_for_coloring)
                    coord_max = np.nanmax(coordination_for_coloring)
                    if coord_max > coord_min:
                        from matplotlib.colors import Normalize
                        norm = Normalize(vmin=coord_min, vmax=coord_max)
                        coord_normalized = norm(coordination_for_coloring)
                        
                        # Use coordination-appropriate colormap (viridis good for continuous coordination values)
                        facecolors = plt.cm.get_cmap(surface_cmap)(coord_normalized)
                        color = None
                        print(f"      ✓ Applied coordination gradient coloring (range: {coord_min:.3f} to {coord_max:.3f})")
                    else:
                        facecolors = None
                        color = ion_color
                        print(f"      ⚠️ Coordination range too small, using solid color")
                        
                elif color_scheme == 'pmf_gradient':
                    # Get the PMF data for coloring (SAME AS WORKING METHOD)
                    if fill_missing_data:
                        # Use the filled PMF grid that was calculated above
                        base_z = surface_info['base_z']
                        z_scale = height_surface_results.get('z_scale_factor', 1.0)
                        if z_scale != 0:
                            pmf_for_coloring = (Z - base_z) / z_scale
                        else:
                            pmf_for_coloring = Z - base_z
                    else:
                        # Use original PMF grid
                        pmf_for_coloring = surface_info['pmf_grid']
                    
                    # Apply PMF normalization for colormap (EXACT COPY FROM WORKING METHOD)
                    if surf_colorbar_vmax > surf_colorbar_vmin:
                        from matplotlib.colors import Normalize
                        norm = Normalize(vmin=surf_colorbar_vmin, vmax=surf_colorbar_vmax)
                        pmf_normalized = norm(pmf_for_coloring)
                        
                        # CRITICAL: Use the EXACT same facecolors pattern as the working method
                        facecolors = plt.cm.get_cmap(surface_cmap)(pmf_normalized)
                        color = None
                        print(f"      ✓ Applied PMF gradient coloring (range: {surf_colorbar_vmin:.3f} to {surf_colorbar_vmax:.3f})")
                    else:
                        facecolors = None
                        color = ion_color
                        print(f"      ⚠️ PMF range too small, using solid color")
                        
                elif color_scheme == 'height_gradient':
                    # Color based on height values
                    if surf_colorbar_vmax > surf_colorbar_vmin:
                        from matplotlib.colors import Normalize
                        norm = Normalize(vmin=surf_colorbar_vmin, vmax=surf_colorbar_vmax)
                        height_normalized = norm(Z)
                        facecolors = plt.cm.get_cmap(surface_cmap)(height_normalized)
                        color = None
                        print(f"      ✓ Applied height gradient coloring")
                    else:
                        facecolors = None
                        color = ion_color
                        
                elif color_scheme == 'adaptive':
                    # Use solid color from results
                    facecolors = None
                    color = ion_color
                    print(f"      ✓ Applied solid color: {color}")
                    
                else:
                    # Default to solid color
                    facecolors = None
                    color = ion_color
                
                # Create the surface plot - USE EXACT PATTERN FROM WORKING METHOD
                if surface_style == 'smooth':
                    if facecolors is not None:
                        # EXACT COPY from plot_pmf_3d_spatial_distribution_vs_z() working method
                        surf = ax.plot_surface(X, Y, Z, 
                                             facecolors=facecolors,
                                             alpha=surface_alpha, 
                                             linewidth=surface_linewidth, 
                                             antialiased=surface_antialiased,
                                             label=f'z = {surf_z:.1f} Å' if isinstance(surf_z, (int, float)) else str(surf_z))
                        z_label = f"z = {surf_z:.1f} Å" if isinstance(surf_z, (int, float)) else str(surf_z)
                        print(f"      ✓ Applied facecolors (PMF-based) to surface at {z_label}")
                    else:
                        # Use solid color (fallback when facecolors fails)
                        surf = ax.plot_surface(X, Y, Z,
                                             color=color,
                                             alpha=surface_alpha,
                                             linewidth=surface_linewidth,
                                             antialiased=surface_antialiased,
                                             label=f'z = {surf_z:.1f} Å' if isinstance(surf_z, (int, float)) else str(surf_z))
                        z_label = f"z = {surf_z:.1f} Å" if isinstance(surf_z, (int, float)) else str(surf_z)
                        print(f"      ✓ Applied solid color to surface at {z_label}")
                
                elif surface_style == 'wireframe':
                    surf = ax.plot_wireframe(X, Y, Z,
                                           color=color or ion_color,
                                           alpha=ion_alpha,
                                           rstride=surface_rstride*2,
                                           cstride=surface_cstride*2,
                                           linewidth=1.0)
                
                elif surface_style == 'filled_contour':
                    # Use contour3D for filled contour surfaces
                    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 15)
                    surf = ax.contour3D(X, Y, Z, levels=levels,
                                      cmap=surface_cmap, alpha=ion_alpha)
                
                ion_plot_surfaces.append(surf)
                
                z_label = f"z = {surf_z:.1f} Å" if isinstance(surf_z, (int, float)) else str(surf_z)
                print(f"    ✓ {z_label}: height range = {np.nanmin(Z):.2f} to {np.nanmax(Z):.2f} Å")
            
            # Add reference planes if requested
            if show_reference_planes:
                x_range = [X.min(), X.max()]
                y_range = [Y.min(), Y.max()]
                
                # Bulk reference plane (z = original z-value)
                if len(surfaces_to_plot) > 0:
                    first_surface = ion_surfaces_data[surfaces_to_plot[0]]
                    bulk_z = first_surface['original_z']
                    
                    xx, yy = np.meshgrid(x_range, y_range)
                    zz = np.full_like(xx, bulk_z)
                    ax.plot_surface(xx, yy, zz, color='gray', alpha=reference_plane_alpha,
                                  linewidth=0, antialiased=False)
            
            # Add clay overlay if requested
            if clay_overlay:
                # Try to overlay clay structure if available in analysis results
                if hasattr(self.analysis, 'clay_positions') and self.analysis.clay_positions is not None:
                    clay_data = self.analysis.clay_positions
                    # Transform clay coordinates to match display coordinates
                    if 'x' in clay_data and 'y' in clay_data and 'z' in clay_data:
                        clay_x = clay_data['x'] - x_centers.min()
                        clay_y = clay_data['y'] - y_centers.min()
                        clay_z = clay_data['z']
                        ax.scatter(clay_x, clay_y, clay_z, c='brown', alpha=0.3, s=20, label='Clay atoms')
                        print(f"      ✓ Added clay overlay with {len(clay_x)} atoms")
                    else:
                        print(f"      ⚠️ Clay overlay requested but clay position data format not recognized")
                else:
                    print(f"      ⚠️ Clay overlay requested but no clay position data available")
            
            # Store surface objects
            plot_results['surfaces'][ion_type] = ion_plot_surfaces
            
            # CREATE COMBINED FIGURE FONT SIZES (following working method pattern)
            combined_label_fontsize = 10  # Smaller for combined figure
            combined_tick_fontsize = 8    # Smaller for combined figure
            combined_title_fontsize = 12  # Smaller for combined figure
            combined_colorbar_tick_fontsize = 8  # Smaller for combined figure
            
            # Set axis properties with COMBINED FIGURE font sizes
            ax.set_xlabel('X (Å)', fontsize=combined_label_fontsize)
            ax.set_ylabel('Y (Å)', fontsize=combined_label_fontsize)
            ax.set_zlabel('Z (Å)', fontsize=combined_label_fontsize)
            
            if show_title:
                title = f'{ion_type} PMF Height Surfaces'
                ax.set_title(title, fontsize=combined_title_fontsize, fontweight='bold')
            
            # Set viewing angle
            ax.view_init(elev=elev, azim=azim)
            
            # Set axis limits using transformed coordinates
            ax.set_xlim(x_display[0], x_display[-1])
            ax.set_ylim(y_display[0], y_display[-1])
            
            # Invert axes to match the original method (CRITICAL FIX)
            ax.invert_xaxis()
            ax.invert_yaxis()
            
            # Apply lighting style
            if lighting_style == 'dramatic':
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
            
            # Apply surface lighting settings
            if not surface_lighting:
                # Disable 3D lighting effects for flatter appearance
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                # Set uniform lighting
                ax.xaxis.pane.set_edgecolor('none')
                ax.yaxis.pane.set_edgecolor('none')
                ax.zaxis.pane.set_edgecolor('none')
            
            # Remove gray background fill from 3D plot panes
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_alpha(0)
            ax.yaxis.pane.set_alpha(0)
            ax.zaxis.pane.set_alpha(0)
            
            # Add colorbar for this ion with appropriate scaling
            if show_colorbar and ion_plot_surfaces:
                if colorbar_scaling == 'individual':
                    # For individual scaling, use the range from the last surface (or you could skip colorbar)
                    # Note: Individual scaling makes colorbars less meaningful since each surface has different range
                    # Use the global ion range for colorbar display
                    ion_pmf_values = []
                    for z_val, surface_info in ion_surfaces_data.items():
                        pmf_grid = surface_info['pmf_grid']
                        finite_pmfs = pmf_grid[np.isfinite(pmf_grid)]
                        if len(finite_pmfs) > 0:
                            ion_pmf_values.extend(finite_pmfs)
                    
                    if ion_pmf_values:
                        display_vmin = np.min(ion_pmf_values)
                        display_vmax = np.max(ion_pmf_values)
                    else:
                        display_vmin, display_vmax = 0, 1
                    colorbar_label_text = f'{ion_type} {colorbar_label}\n({display_vmin:.1f} - {display_vmax:.1f})'
                else:
                    # Use the calculated ion/unified range
                    display_vmin = ion_colorbar_vmin
                    display_vmax = ion_colorbar_vmax
                    colorbar_label_text = f'{ion_type} {colorbar_label}\n({display_vmin:.1f} - {display_vmax:.1f})'
                
                norm = plt.Normalize(vmin=display_vmin, vmax=display_vmax)
                sm = plt.cm.ScalarMappable(norm=norm, cmap=surface_cmap)
                sm.set_array([])
                
                cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
                cbar.set_label(colorbar_label_text, fontsize=combined_label_fontsize)
                # CRITICAL FIX: Set colorbar tick font size for combined figure
                cbar.ax.tick_params(labelsize=combined_colorbar_tick_fontsize)
            
            # COMBINED FIGURE TICK FONT SIZE
            ax.tick_params(axis='both', which='major', labelsize=combined_tick_fontsize)
            
            # Add legend if requested and we have labeled surfaces
            if show_legend and ion_plot_surfaces:
                # Create custom legend entries for surfaces
                legend_elements = []
                for i, surf_z in enumerate(surfaces_to_plot):
                    if i < len(ion_plot_surfaces):
                        # Use a proxy artist for legend since 3D surfaces don't work well with legends
                        from matplotlib.patches import Patch
                        legend_elements.append(Patch(facecolor=ion_color, alpha=surface_alpha, 
                                                   label=f'z = {surf_z:.1f} Å' if isinstance(surf_z, (int, float)) else str(surf_z)))
                
                if legend_elements:
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=legend_fontsize, 
                             framealpha=0.8, fancybox=True, shadow=True)
                    print(f"      ✓ Added legend with {len(legend_elements)} surface entries")
        
        # Individual colorbars are now added per ion subplot above
        
        # CRITICAL FIX: Apply generous padding to prevent cropping (copy from working method)
        plt.tight_layout(pad=3.0)  # Increased padding
        
        # Handle individual and combined figure display and saving
        figures_saved = []
        figures_displayed = []
        
        if len(ion_types) > 1:
            # Multiple ions: handle individual vs combined figures
            
            # Save/show individual figures if requested
            if save_individual_figures or show_individual_figures:
                print(f"\n🎯 Processing individual figures for {len(ion_types)} ions...")
                
                individual_results = []
                for idx, ion_type in enumerate(ion_types):
                    print(f"    🔄 Creating individual figure for {ion_type}...")
                    
                    # Create individual figure for this ion
                    individual_fig = plt.figure(figsize=individual_figsize)
                    individual_ax = individual_fig.add_subplot(111, projection='3d')
                    
                    # Get the combined axis for this ion to copy settings and data
                    combined_ax = plot_results['axes'][idx]
                    
                    # Get ion data
                    ion_data = height_surfaces[ion_type]
                    ion_surfaces_data = ion_data['surfaces']
                    ion_color = ion_data['color']
                    ion_alpha = ion_data['alpha']
                    
                    # Determine which surfaces to plot (same logic as main plotting)
                    if surface_indices is None:
                        surfaces_to_plot = list(ion_surfaces_data.keys())
                    else:
                        available_indices = list(ion_surfaces_data.keys())
                        surfaces_to_plot = [available_indices[i] for i in surface_indices 
                                          if i < len(available_indices)]
                    
                    # Replicate the EXACT same processing pipeline as the main plotting
                    for surf_z in surfaces_to_plot:
                        surface_info = ion_surfaces_data[surf_z]
                        surface_data = surface_info['surface_data']
                        
                        # Use transformed coordinates - reconstruct X,Y with transformed coordinates
                        X_orig, Y_orig = surface_data['X'], surface_data['Y']
                        X = X_orig - x_centers.min()  # Apply same transformation as x_display
                        Y = Y_orig - y_centers.min()  # Apply same transformation as y_display
                        Z_original = surface_data['Z']
                        
                        # Apply spatial interpolation to fill missing data (SAME AS MAIN)
                        if fill_missing_data:
                            # Extract PMF grid from surface height
                            base_z = surface_info['base_z']
                            z_scale = height_surface_results.get('z_scale_factor', 1.0)
                            
                            # Recover PMF grid from height surface
                            if z_scale != 0:
                                pmf_grid_recovered = (Z_original - base_z) / z_scale
                            else:
                                pmf_grid_recovered = Z_original - base_z
                            
                            # Fill missing PMF data (using same function as main)
                            filled_pmf_grid = _fill_missing_pmf_data(pmf_grid_recovered)
                            
                            # Recreate height surface from filled PMF data
                            Z = base_z + filled_pmf_grid * z_scale
                        else:
                            Z = Z_original
                        
                        # Apply surface smoothing AFTER spatial interpolation (SAME AS MAIN)
                        if surface_smoothing:
                            smoothed_data = _apply_surface_smoothing({
                                'X': X, 'Y': Y, 'Z': Z
                            })
                            X, Y, Z = smoothed_data['X'], smoothed_data['Y'], smoothed_data['Z']
                        
                        # Calculate colorbar range based on scaling strategy (SAME AS MAIN)
                        if colorbar_scaling == 'unified':
                            surf_colorbar_vmin = ion_colorbar_vmin
                            surf_colorbar_vmax = ion_colorbar_vmax
                        elif colorbar_scaling == 'global':
                            surf_colorbar_vmin = ion_colorbar_vmin
                            surf_colorbar_vmax = ion_colorbar_vmax
                        elif colorbar_scaling == 'individual':
                            pmf_grid = surface_info['pmf_grid']
                            finite_pmfs = pmf_grid[np.isfinite(pmf_grid)]
                            
                            if len(finite_pmfs) > 0:
                                surf_pmf_min = np.min(finite_pmfs)
                                surf_pmf_max = np.max(finite_pmfs)
                                surf_colorbar_vmin = colorbar_vmin if colorbar_vmin is not None else surf_pmf_min
                                surf_colorbar_vmax = colorbar_vmax if colorbar_vmax is not None else surf_pmf_max
                            else:
                                surf_colorbar_vmin = colorbar_vmin if colorbar_vmin is not None else 0
                                surf_colorbar_vmax = colorbar_vmax if colorbar_vmax is not None else 1
                        
                        # Handle color scheme - EXACT COPY FROM MAIN METHOD
                        if color_scheme == 'pmf_gradient':
                            # Get the PMF data for coloring (SAME AS WORKING METHOD)
                            if fill_missing_data:
                                # Use the filled PMF grid that was calculated above
                                base_z = surface_info['base_z']
                                z_scale = height_surface_results.get('z_scale_factor', 1.0)
                                if z_scale != 0:
                                    pmf_for_coloring = (Z - base_z) / z_scale
                                else:
                                    pmf_for_coloring = Z - base_z
                            else:
                                # Use original PMF grid
                                pmf_for_coloring = surface_info['pmf_grid']
                            
                            # Apply PMF normalization for colormap (EXACT COPY FROM WORKING METHOD)
                            if surf_colorbar_vmax > surf_colorbar_vmin:
                                from matplotlib.colors import Normalize
                                norm = Normalize(vmin=surf_colorbar_vmin, vmax=surf_colorbar_vmax)
                                pmf_normalized = norm(pmf_for_coloring)
                                
                                # CRITICAL: Use the EXACT same facecolors pattern as the working method
                                facecolors = plt.cm.get_cmap(surface_cmap)(pmf_normalized)
                                color = None
                            else:
                                facecolors = None
                                color = ion_color
                                
                        elif color_scheme == 'height_gradient':
                            # Color based on height values
                            if surf_colorbar_vmax > surf_colorbar_vmin:
                                from matplotlib.colors import Normalize
                                norm = Normalize(vmin=surf_colorbar_vmin, vmax=surf_colorbar_vmax)
                                height_normalized = norm(Z)
                                facecolors = plt.cm.get_cmap(surface_cmap)(height_normalized)
                                color = None
                            else:
                                facecolors = None
                                color = ion_color
                                
                        elif color_scheme == 'adaptive':
                            # Use solid color from results
                            facecolors = None
                            color = ion_color
                            
                        else:
                            # Default to solid color
                            facecolors = None
                            color = ion_color
                        
                        # Create the surface plot - USE EXACT PATTERN FROM WORKING METHOD
                        if surface_style == 'smooth':
                            if facecolors is not None:
                                # EXACT COPY from plot_pmf_3d_spatial_distribution_vs_z() working method
                                surf = individual_ax.plot_surface(X, Y, Z, 
                                                     facecolors=facecolors,
                                                     alpha=surface_alpha, 
                                                     linewidth=surface_linewidth, 
                                                     antialiased=surface_antialiased,
                                                     label=f'z = {surf_z:.1f} Å')
                            else:
                                # Use solid color (fallback when facecolors fails)
                                surf = individual_ax.plot_surface(X, Y, Z,
                                                     color=color,
                                                     alpha=surface_alpha,
                                                     linewidth=surface_linewidth,
                                                     antialiased=surface_antialiased,
                                                     label=f'z = {surf_z:.1f} Å')
                        
                        elif surface_style == 'wireframe':
                            surf = individual_ax.plot_wireframe(X, Y, Z,
                                                   color=color or ion_color,
                                                   alpha=ion_alpha,
                                                   rstride=surface_rstride*2,
                                                   cstride=surface_cstride*2,
                                                   linewidth=1.0)
                        
                        elif surface_style == 'filled_contour':
                            # Use contour3D for filled contour surfaces
                            levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 15)
                            surf = individual_ax.contour3D(X, Y, Z, levels=levels,
                                                  cmap=surface_cmap, alpha=ion_alpha)
                    
                    # FIXED FONT SIZES FOR INDIVIDUAL FIGURES (copying working pattern)
                    # Use the same pattern as plot_pmf_3d_spatial_distribution_vs_z() method
                    
                    # Set individual axis properties with INDIVIDUAL FIGURE font sizes
                    individual_ax.set_xlabel('X (Å)', fontsize=label_fontsize)
                    individual_ax.set_ylabel('Y (Å)', fontsize=label_fontsize)
                    individual_ax.set_zlabel('Z (Å)', fontsize=label_fontsize)
                    
                    if show_title:
                        title = f'{ion_type} PMF Height Surfaces'
                        individual_ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
                    
                    # Set viewing angle (SAME AS MAIN)
                    individual_ax.view_init(elev=elev, azim=azim)
                    
                    # Set axis limits using transformed coordinates (SAME AS MAIN)
                    individual_ax.set_xlim(x_display[0], x_display[-1])
                    individual_ax.set_ylim(y_display[0], y_display[-1])
                    
                    # Invert axes to match the original method (SAME AS MAIN)
                    individual_ax.invert_xaxis()
                    individual_ax.invert_yaxis()
                    
                    # Apply lighting and remove gray backgrounds (SAME AS MAIN)
                    if lighting_style == 'dramatic':
                        individual_ax.xaxis.pane.fill = False
                        individual_ax.yaxis.pane.fill = False
                        individual_ax.zaxis.pane.fill = False
                    
                    # Apply surface lighting settings (SAME AS MAIN)
                    if not surface_lighting:
                        individual_ax.xaxis.pane.fill = False
                        individual_ax.yaxis.pane.fill = False
                        individual_ax.zaxis.pane.fill = False
                        individual_ax.xaxis.pane.set_edgecolor('none')
                        individual_ax.yaxis.pane.set_edgecolor('none')
                        individual_ax.zaxis.pane.set_edgecolor('none')
                    
                    # Remove gray background fill from 3D plot panes (SAME AS MAIN)
                    individual_ax.xaxis.pane.fill = False
                    individual_ax.yaxis.pane.fill = False
                    individual_ax.zaxis.pane.fill = False
                    individual_ax.xaxis.pane.set_alpha(0)
                    individual_ax.yaxis.pane.set_alpha(0)
                    individual_ax.zaxis.pane.set_alpha(0)
                    
                    # Add colorbar for individual plot (SAME LOGIC AS MAIN)
                    if show_colorbar:
                        if colorbar_scaling == 'individual':
                            # Calculate ion-level PMF range for colorbar display
                            ion_pmf_values = []
                            for z_val, surface_info in ion_surfaces_data.items():
                                pmf_grid = surface_info['pmf_grid']
                                finite_pmfs = pmf_grid[np.isfinite(pmf_grid)]
                                if len(finite_pmfs) > 0:
                                    ion_pmf_values.extend(finite_pmfs)
                            
                            if ion_pmf_values:
                                display_vmin = np.min(ion_pmf_values)
                                display_vmax = np.max(ion_pmf_values)
                            else:
                                display_vmin, display_vmax = 0, 1
                            colorbar_label_text = f'{ion_type} {colorbar_label}\n({display_vmin:.1f} - {display_vmax:.1f})'
                        else:
                            # Use the calculated ion/unified range
                            display_vmin = ion_colorbar_vmin
                            display_vmax = ion_colorbar_vmax
                            colorbar_label_text = f'{ion_type} {colorbar_label}\n({display_vmin:.1f} - {display_vmax:.1f})'
                        
                        norm = plt.Normalize(vmin=display_vmin, vmax=display_vmax)
                        sm = plt.cm.ScalarMappable(norm=norm, cmap=surface_cmap)
                        sm.set_array([])
                        
                        cbar = individual_fig.colorbar(sm, ax=individual_ax, shrink=0.6, aspect=20)
                        cbar.set_label(colorbar_label_text, fontsize=label_fontsize)
                        # CRITICAL FIX: Set colorbar tick font size (copy from working method)
                        cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
                    
                    # INDIVIDUAL FIGURE TICK FONT SIZE (copy from working method)
                    individual_ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    
                    # CRITICAL FIX: Use generous padding (copy from working method pattern)
                    plt.tight_layout(pad=3.0)  # Increased padding to prevent label cropping
                    
                    # Save individual figure if requested
                    individual_filename = None
                    if save_individual_figures:
                        individual_filename = f"{filename_prefix}_{ion_type}_{visualization_mode}_individual.png"
                        individual_fig.savefig(individual_filename, dpi=dpi, bbox_inches='tight', 
                                             facecolor='white', pad_inches=0.3)  # Increased padding
                        figures_saved.append(individual_filename)
                        print(f"    💾 Saved individual: {individual_filename}")
                    
                    # Store individual figure info
                    individual_results.append({
                        'ion_type': ion_type,
                        'figure': individual_fig,
                        'axis': individual_ax,
                        'filename': individual_filename
                    })
                    
                    # Show individual figure if requested (do this AFTER storing)
                    if show_individual_figures:
                        # Individual figure will remain open for viewing
                        figures_displayed.append(f"{ion_type}_individual")
                        print(f"    🖼️ Displayed individual: {ion_type}")
                    else:
                        # CRITICAL FIX: Close individual figures if not showing them (copy from working method)
                        plt.close(individual_fig)
                
                plot_results['individual_figures'] = individual_results
            
            # Handle combined figure
            if save_combined_figure:
                combined_filename = f"{filename_prefix}_{visualization_mode}_combined.png"
                fig.savefig(combined_filename, dpi=dpi, bbox_inches='tight', facecolor='white')
                figures_saved.append(combined_filename)
                print(f"\n💾 Saved combined: {combined_filename}")
            
            if show_combined_figure:
                fig.show()
                figures_displayed.append("combined")
                print(f"\n🖼️ Displayed combined figure")
        
        else:
            # Single ion: handle individual figures AND standard behavior
            
            # Save/show individual figures if requested (SAME LOGIC AS MULTI-ION)
            if save_individual_figures or show_individual_figures:
                print(f"\n🎯 Processing individual figures for single ion...")
                
                individual_results = []
                for idx, ion_type in enumerate(ion_types):
                    print(f"    🔄 Creating individual figure for {ion_type}...")
                    
                    # Create individual figure for this ion
                    individual_fig = plt.figure(figsize=individual_figsize)
                    individual_ax = individual_fig.add_subplot(111, projection='3d')
                    
                    # Get the combined axis for this ion to copy settings and data
                    combined_ax = plot_results['axes'][idx]
                    
                    # Get ion data
                    ion_data = height_surfaces[ion_type]
                    ion_surfaces_data = ion_data['surfaces']
                    ion_color = ion_data['color']
                    ion_alpha = ion_data['alpha']
                    
                    # CRITICAL FIX: Calculate colorbar range for single ion (same as main loop)
                    if colorbar_scaling == 'unified':
                        # Use unified range for all ions and surfaces
                        ion_colorbar_vmin = colorbar_vmin if colorbar_vmin is not None else unified_pmf_min
                        ion_colorbar_vmax = colorbar_vmax if colorbar_vmax is not None else unified_pmf_max
                        
                    elif colorbar_scaling == 'global':
                        # Calculate global range for this ion type (all surfaces)
                        ion_pmf_values = []
                        for z_val, surface_info in ion_surfaces_data.items():
                            pmf_grid = surface_info['pmf_grid']
                            finite_pmfs = pmf_grid[np.isfinite(pmf_grid)]
                            if len(finite_pmfs) > 0:
                                ion_pmf_values.extend(finite_pmfs)
                        
                        if ion_pmf_values:
                            ion_pmf_min = np.min(ion_pmf_values)
                            ion_pmf_max = np.max(ion_pmf_values)
                            ion_colorbar_vmin = colorbar_vmin if colorbar_vmin is not None else ion_pmf_min
                            ion_colorbar_vmax = colorbar_vmax if colorbar_vmax is not None else ion_pmf_max
                        else:
                            ion_colorbar_vmin = colorbar_vmin if colorbar_vmin is not None else 0
                            ion_colorbar_vmax = colorbar_vmax if colorbar_vmax is not None else 1
                    
                    # For 'individual' scaling, we'll calculate per surface below
                    
                    # Determine which surfaces to plot (same logic as main plotting)
                    if surface_indices is None:
                        surfaces_to_plot = list(ion_surfaces_data.keys())
                    else:
                        available_indices = list(ion_surfaces_data.keys())
                        surfaces_to_plot = [available_indices[i] for i in surface_indices 
                                          if i < len(available_indices)]
                    
                    # Replicate the EXACT same processing pipeline as the main plotting
                    for surf_z in surfaces_to_plot:
                        surface_info = ion_surfaces_data[surf_z]
                        surface_data = surface_info['surface_data']
                        
                        # Use transformed coordinates - reconstruct X,Y with transformed coordinates
                        X_orig, Y_orig = surface_data['X'], surface_data['Y']
                        X = X_orig - x_centers.min()  # Apply same transformation as x_display
                        Y = Y_orig - y_centers.min()  # Apply same transformation as y_display
                        Z_original = surface_data['Z']
                        
                        # Apply spatial interpolation to fill missing data (SAME AS MAIN)
                        if fill_missing_data:
                            # Extract PMF grid from surface height
                            base_z = surface_info['base_z']
                            z_scale = height_surface_results.get('z_scale_factor', 1.0)
                            
                            # Recover PMF grid from height surface
                            if z_scale != 0:
                                pmf_grid_recovered = (Z_original - base_z) / z_scale
                            else:
                                pmf_grid_recovered = Z_original - base_z
                            
                            # Fill missing PMF data (using same function as main)
                            filled_pmf_grid = _fill_missing_pmf_data(pmf_grid_recovered)
                            
                            # Recreate height surface from filled PMF data
                            Z = base_z + filled_pmf_grid * z_scale
                        else:
                            Z = Z_original
                        
                        # Apply surface smoothing AFTER spatial interpolation (SAME AS MAIN)
                        if surface_smoothing:
                            smoothed_data = _apply_surface_smoothing({
                                'X': X, 'Y': Y, 'Z': Z
                            })
                            X, Y, Z = smoothed_data['X'], smoothed_data['Y'], smoothed_data['Z']
                        
                        # Calculate colorbar range based on scaling strategy (SAME AS MAIN)
                        if colorbar_scaling == 'unified':
                            surf_colorbar_vmin = ion_colorbar_vmin
                            surf_colorbar_vmax = ion_colorbar_vmax
                        elif colorbar_scaling == 'global':
                            surf_colorbar_vmin = ion_colorbar_vmin
                            surf_colorbar_vmax = ion_colorbar_vmax
                        elif colorbar_scaling == 'individual':
                            pmf_grid = surface_info['pmf_grid']
                            finite_pmfs = pmf_grid[np.isfinite(pmf_grid)]
                            
                            if len(finite_pmfs) > 0:
                                surf_pmf_min = np.min(finite_pmfs)
                                surf_pmf_max = np.max(finite_pmfs)
                                surf_colorbar_vmin = colorbar_vmin if colorbar_vmin is not None else surf_pmf_min
                                surf_colorbar_vmax = colorbar_vmax if colorbar_vmax is not None else surf_pmf_max
                            else:
                                surf_colorbar_vmin = colorbar_vmin if colorbar_vmin is not None else 0
                                surf_colorbar_vmax = colorbar_vmax if colorbar_vmax is not None else 1
                        
                        # Handle color scheme - EXACT COPY FROM MAIN METHOD
                        if color_scheme == 'pmf_gradient':
                            # Get the PMF data for coloring (SAME AS WORKING METHOD)
                            if fill_missing_data:
                                # Use the filled PMF grid that was calculated above
                                base_z = surface_info['base_z']
                                z_scale = height_surface_results.get('z_scale_factor', 1.0)
                                if z_scale != 0:
                                    pmf_for_coloring = (Z - base_z) / z_scale
                                else:
                                    pmf_for_coloring = Z - base_z
                            else:
                                # Use original PMF grid
                                pmf_for_coloring = surface_info['pmf_grid']
                            
                            # Apply PMF normalization for colormap (EXACT COPY FROM WORKING METHOD)
                            if surf_colorbar_vmax > surf_colorbar_vmin:
                                from matplotlib.colors import Normalize
                                norm = Normalize(vmin=surf_colorbar_vmin, vmax=surf_colorbar_vmax)
                                pmf_normalized = norm(pmf_for_coloring)
                                
                                # CRITICAL: Use the EXACT same facecolors pattern as the working method
                                facecolors = plt.cm.get_cmap(surface_cmap)(pmf_normalized)
                                color = None
                            else:
                                facecolors = None
                                color = ion_color
                                
                        elif color_scheme == 'height_gradient':
                            # Color based on height values
                            if surf_colorbar_vmax > surf_colorbar_vmin:
                                from matplotlib.colors import Normalize
                                norm = Normalize(vmin=surf_colorbar_vmin, vmax=surf_colorbar_vmax)
                                height_normalized = norm(Z)
                                facecolors = plt.cm.get_cmap(surface_cmap)(height_normalized)
                                color = None
                            else:
                                facecolors = None
                                color = ion_color
                                
                        elif color_scheme == 'adaptive':
                            # Use solid color from results
                            facecolors = None
                            color = ion_color
                            
                        else:
                            # Default to solid color
                            facecolors = None
                            color = ion_color
                        
                        # Create the surface plot - USE EXACT PATTERN FROM WORKING METHOD
                        if surface_style == 'smooth':
                            if facecolors is not None:
                                # EXACT COPY from plot_pmf_3d_spatial_distribution_vs_z() working method
                                surf = individual_ax.plot_surface(X, Y, Z, 
                                                     facecolors=facecolors,
                                                     alpha=surface_alpha, 
                                                     linewidth=surface_linewidth, 
                                                     antialiased=surface_antialiased,
                                                     label=f'z = {surf_z:.1f} Å')
                            else:
                                # Use solid color (fallback when facecolors fails)
                                surf = individual_ax.plot_surface(X, Y, Z,
                                                     color=color,
                                                     alpha=surface_alpha,
                                                     linewidth=surface_linewidth,
                                                     antialiased=surface_antialiased,
                                                     label=f'z = {surf_z:.1f} Å')
                        
                        elif surface_style == 'wireframe':
                            surf = individual_ax.plot_wireframe(X, Y, Z,
                                                   color=color or ion_color,
                                                   alpha=ion_alpha,
                                                   rstride=surface_rstride*2,
                                                   cstride=surface_cstride*2,
                                                   linewidth=1.0)
                        
                        elif surface_style == 'filled_contour':
                            # Use contour3D for filled contour surfaces
                            levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 15)
                            surf = individual_ax.contour3D(X, Y, Z, levels=levels,
                                                  cmap=surface_cmap, alpha=ion_alpha)
                    
                    # FIXED FONT SIZES FOR INDIVIDUAL FIGURES (copying working pattern)
                    # Use the same pattern as plot_pmf_3d_spatial_distribution_vs_z() method
                    
                    # Set individual axis properties with INDIVIDUAL FIGURE font sizes
                    individual_ax.set_xlabel('X (Å)', fontsize=label_fontsize)
                    individual_ax.set_ylabel('Y (Å)', fontsize=label_fontsize)
                    individual_ax.set_zlabel('Z (Å)', fontsize=label_fontsize)
                    
                    if show_title:
                        title = f'{ion_type} PMF Height Surfaces'
                        individual_ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
                    
                    # Set viewing angle (SAME AS MAIN)
                    individual_ax.view_init(elev=elev, azim=azim)
                    
                    # Set axis limits using transformed coordinates (SAME AS MAIN)
                    individual_ax.set_xlim(x_display[0], x_display[-1])
                    individual_ax.set_ylim(y_display[0], y_display[-1])
                    
                    # Invert axes to match the original method (SAME AS MAIN)
                    individual_ax.invert_xaxis()
                    individual_ax.invert_yaxis()
                    
                    # Apply lighting and remove gray backgrounds (SAME AS MAIN)
                    if lighting_style == 'dramatic':
                        individual_ax.xaxis.pane.fill = False
                        individual_ax.yaxis.pane.fill = False
                        individual_ax.zaxis.pane.fill = False
                    
                    # Apply surface lighting settings (SAME AS MAIN)
                    if not surface_lighting:
                        individual_ax.xaxis.pane.fill = False
                        individual_ax.yaxis.pane.fill = False
                        individual_ax.zaxis.pane.fill = False
                        individual_ax.xaxis.pane.set_edgecolor('none')
                        individual_ax.yaxis.pane.set_edgecolor('none')
                        individual_ax.zaxis.pane.set_edgecolor('none')
                    
                    # Remove gray background fill from 3D plot panes (SAME AS MAIN)
                    individual_ax.xaxis.pane.fill = False
                    individual_ax.yaxis.pane.fill = False
                    individual_ax.zaxis.pane.fill = False
                    individual_ax.xaxis.pane.set_alpha(0)
                    individual_ax.yaxis.pane.set_alpha(0)
                    individual_ax.zaxis.pane.set_alpha(0)
                    
                    # Add colorbar for individual plot (SAME LOGIC AS MAIN)
                    if show_colorbar:
                        if colorbar_scaling == 'individual':
                            # Calculate ion-level PMF range for colorbar display
                            ion_pmf_values = []
                            for z_val, surface_info in ion_surfaces_data.items():
                                pmf_grid = surface_info['pmf_grid']
                                finite_pmfs = pmf_grid[np.isfinite(pmf_grid)]
                                if len(finite_pmfs) > 0:
                                    ion_pmf_values.extend(finite_pmfs)
                            
                            if ion_pmf_values:
                                display_vmin = np.min(ion_pmf_values)
                                display_vmax = np.max(ion_pmf_values)
                            else:
                                display_vmin, display_vmax = 0, 1
                            colorbar_label_text = f'{ion_type} {colorbar_label}\n({display_vmin:.1f} - {display_vmax:.1f})'
                        else:
                            # Use the calculated ion/unified range
                            display_vmin = ion_colorbar_vmin
                            display_vmax = ion_colorbar_vmax
                            colorbar_label_text = f'{ion_type} {colorbar_label}\n({display_vmin:.1f} - {display_vmax:.1f})'
                        
                        norm = plt.Normalize(vmin=display_vmin, vmax=display_vmax)
                        sm = plt.cm.ScalarMappable(norm=norm, cmap=surface_cmap)
                        sm.set_array([])
                        
                        cbar = individual_fig.colorbar(sm, ax=individual_ax, shrink=0.6, aspect=20)
                        cbar.set_label(colorbar_label_text, fontsize=label_fontsize)
                        # CRITICAL FIX: Set colorbar tick font size (copy from working method)
                        cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
                    
                    # INDIVIDUAL FIGURE TICK FONT SIZE (copy from working method)
                    individual_ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    
                    # CRITICAL FIX: Use generous padding (copy from working method pattern)
                    plt.tight_layout(pad=3.0)  # Increased padding to prevent label cropping
                    
                    # Save individual figure if requested
                    individual_filename = None
                    if save_individual_figures:
                        individual_filename = f"{filename_prefix}_{ion_type}_{visualization_mode}_individual.png"
                        individual_fig.savefig(individual_filename, dpi=dpi, bbox_inches='tight', 
                                             facecolor='white', pad_inches=0.3)  # Increased padding
                        figures_saved.append(individual_filename)
                        print(f"    💾 Saved individual: {individual_filename}")
                    
                    # Store individual figure info
                    individual_results.append({
                        'ion_type': ion_type,
                        'figure': individual_fig,
                        'axis': individual_ax,
                        'filename': individual_filename
                    })
                    
                    # Show individual figure if requested (do this AFTER storing)
                    if show_individual_figures:
                        # Individual figure will remain open for viewing
                        figures_displayed.append(f"{ion_type}_individual")
                        print(f"    🖼️ Displayed individual: {ion_type}")
                    else:
                        # CRITICAL FIX: Close individual figures if not showing them (copy from working method)
                        plt.close(individual_fig)
                
                plot_results['individual_figures'] = individual_results
            
            # Standard behavior for single ion
            if save_plots:
                filename = f"{filename_prefix}_{visualization_mode}.png"
                fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
                figures_saved.append(filename)
                print(f"\n💾 Saved plot: {filename} (dpi={dpi})")
            
            # Show figure (avoid duplicate display)
            if not (save_individual_figures or show_individual_figures):
                fig.show()
                figures_displayed.append("single")
        
        # Store figure management info in results
        plot_results['figures_saved'] = figures_saved
        plot_results['figures_displayed'] = figures_displayed
        
        # Print summary
        print(f"\n📊 PMF Height Surface Plotting Summary:")
        print(f"  Ion types: {len(ion_types)}")
        total_surfaces = sum(len(surfaces) for surfaces in plot_results['surfaces'].values())
        print(f"  Total surfaces plotted: {total_surfaces}")
        print(f"  Visualization mode: {visualization_mode}")
        print(f"  Surface style: {surface_style}")
        print(f"  Color scheme: {color_scheme}")
        print(f"  Colorbar scaling: {colorbar_scaling}")
        
        if colorbar_scaling == 'individual':
            print(f"  💡 Individual scaling: Each surface uses its own PMF range for optimal contrast")
        elif colorbar_scaling == 'global':
            print(f"  💡 Global scaling: All surfaces per ion use the same PMF range")
        elif colorbar_scaling == 'unified':
            print(f"  💡 Unified scaling: All ions and surfaces use the same global PMF range")
        
        # Report saved figures
        if figures_saved:
            print(f"  Plots saved: {len(figures_saved)} files")
            for saved_file in figures_saved:
                print(f"    📁 {saved_file}")
        
        if figures_displayed:
            print(f"  Plots displayed: {len(figures_displayed)} figures")
        
        return plot_results


    def plot_rdf_vs_z(self, save_plots=True, figsize=None, filename=None, 
                    ion_pairs=None, swap_axes=False, cmap='hot',
                    # Publication settings
                    title_fontsize=16,
                    show_title=False,
                    label_fontsize=14,
                    tick_fontsize=10,
                    colorbar_pad=0.08,
                    colorbar_width='3%',
                    colorbar_tick_fontsize=10,
                    # Publication figure control parameters
                    save_individual_figures=True,
                    individual_figsize=(8, 6),
                    save_combined_figure=False,
                    show_individual_figures=False,
                    show_combined_figure=True,
                    dpi=300,
                    # Shell boundary parameters
                    show_solvation_boundary=False,
                    show_pairing_boundary=False,
                    ion_solvation_radii=None,
                    ion_pair_radii=None,
                    boundary_line_color='white',
                    boundary_line_style='--',
                    boundary_line_width=1.5,
                    boundary_line_alpha=0.7):
        '''
        Plot RDF vs z-position for ion pairs with publication-quality control
        
        Parameters
        ----------
        save_plots : bool, default=True
            Whether to save plots (deprecated - use save_individual_figures and save_combined_figure)
        figsize : tuple, optional
            Figure size for combined plot (width, height). Auto-calculated if None.
        filename : str, optional
            Output filename for combined plot (auto-generated if None)
        ion_pairs : list, optional
            Specific ion pairs to plot. If None, plots all available pairs.
        swap_axes : bool, default=False
            If True, plot with z on x-axis and r on y-axis
            If False, plot with r on x-axis and z on y-axis (standard orientation)
        cmap : str, default='hot'
            Colormap for heatmap (e.g., 'hot', 'Blues', 'viridis', 'plasma', 'RdBu_r')
        
        Publication settings
        --------------------
        title_fontsize : int, default=16
            Font size for subplot titles
        show_title : bool, default=False
            Whether to show titles on plots
        label_fontsize : int, default=14
            Font size for axis labels
        tick_fontsize : int, default=10
            Font size for axis tick labels
        colorbar_pad : float, default=0.08
            Padding between plot and colorbar
        colorbar_width : str, default='3%'
            Width of colorbar
        colorbar_tick_fontsize : int, default=10
            Font size for colorbar tick labels
        
        Figure control parameters
        -------------------------
        save_individual_figures : bool, default=True
            Save individual figure for each ion pair
        individual_figsize : tuple, default=(8, 6)
            Figure size for individual plots (width, height)
        save_combined_figure : bool, default=False
            Save combined figure with all ion pairs
        show_individual_figures : bool, default=False
            Display individual figures
        show_combined_figure : bool, default=True
            Display combined figure
        dpi : int, default=300
            DPI for saved figures
        
        Shell boundary parameters
        -------------------------
        show_solvation_boundary : bool, default=False
            Show vertical/horizontal lines at solvation shell boundaries (ion-water pairs)
        show_pairing_boundary : bool, default=False
            Show vertical/horizontal lines at ion pairing boundaries (ion-ion pairs)
        ion_solvation_radii : dict, optional
            Dictionary of solvation shell radii for each ion type
            Example: {'MG': [2.88, 5.03, 7.25], 'NA': [3.18, 5.53, 7.58]}
        ion_pair_radii : dict, optional
            Dictionary of contact ion pair radii for each pair
            Example: {'NA-CL': [3.50, 5.90, 8.20], 'MG-CL': [3.50, 5.40, 7.70]}
        boundary_line_color : str, default='white'
            Color of boundary lines
        boundary_line_style : str, default='--'
            Line style for boundaries ('--', '-', '-.', ':')
        boundary_line_width : float, default=1.5
            Width of boundary lines
        boundary_line_alpha : float, default=0.7
            Transparency of boundary lines (0=transparent, 1=opaque)
        '''
        
        self._validate_analysis()
        
        if not hasattr(self.analysis.results, 'rdf_vs_z'):
            print("No RDF vs z data available. Run calculate_rdf_vs_z_detailed() first.")
            return
        
        rdf_data = self.analysis.results.rdf_vs_z
        
        # Determine which pairs to plot
        if ion_pairs is None:
            pairs_to_plot = list(rdf_data.keys())
        else:
            pairs_to_plot = [pair for pair in ion_pairs if pair in rdf_data]
        
        if not pairs_to_plot:
            print(f"No valid ion pairs found. Available: {list(rdf_data.keys())}")
            return
        
        print(f"\n🎨 Plotting RDF vs z for {len(pairs_to_plot)} ion pairs")
        
        # Set default radii dictionaries if not provided
        if ion_solvation_radii is None:
            ion_solvation_radii = {
                'MG': [2.88, 5.03, 7.25],
                'NA': [3.18, 5.53, 7.58],
                'CL': [3.78, 6.18, 8.43],
            }
        
        if ion_pair_radii is None:
            ion_pair_radii = {
                'NA-CL': [3.50, 5.90, 8.20],
                'MG-CL': [3.50, 5.40, 7.70],
                'CA-CL': [3.65, 6.35, 8.55],
                'K-CL': [3.90, 6.35, 8.55],
            }
        
        # Helper function to detect ion type and get boundary radii
        def get_boundary_radii(pair_name):
            """
            Automatically detect ion type and return appropriate boundary radii
            
            Returns
            -------
            radii : list or None
                List of boundary radii positions, or None if not found
            boundary_type : str
                'solvation' or 'pairing' or None
            """
            pair_upper = pair_name.upper()
            
            # Check for ion-water pairs (solvation)
            if show_solvation_boundary:
                water_keywords = ['OW', 'WATER', 'H2O', 'WAT']
                for water_key in water_keywords:
                    if water_key in pair_upper:
                        # Extract ion name (first part before dash or water keyword)
                        ion_name = pair_upper.split('-')[0] if '-' in pair_upper else pair_upper.replace(water_key, '')
                        ion_name = ion_name.strip()
                        
                        # Look up in solvation radii dictionary
                        if ion_name in ion_solvation_radii:
                            return ion_solvation_radii[ion_name], 'solvation'
            
            # Check for ion-ion pairs (pairing)
            if show_pairing_boundary:
                # Try exact match first
                if pair_upper in ion_pair_radii:
                    return ion_pair_radii[pair_upper], 'pairing'
                
                # Try reversed pair (e.g., CL-NA instead of NA-CL)
                if '-' in pair_upper:
                    reversed_pair = '-'.join(reversed(pair_upper.split('-')))
                    if reversed_pair in ion_pair_radii:
                        return ion_pair_radii[reversed_pair], 'pairing'
            
            return None, None
        
        # Helper function to draw boundaries on an axis
        def draw_boundaries(ax, pair_name, swap_axes_flag):
            """Draw shell boundary lines on the given axis"""
            radii, boundary_type = get_boundary_radii(pair_name)
            
            if radii is None:
                return
            
            for radius in radii:
                if swap_axes_flag:
                    # r on y-axis: draw horizontal lines
                    ax.axhline(y=radius, color=boundary_line_color, 
                              linestyle=boundary_line_style, 
                              linewidth=boundary_line_width,
                              alpha=boundary_line_alpha, zorder=10)
                else:
                    # r on x-axis: draw vertical lines
                    ax.axvline(x=radius, color=boundary_line_color,
                              linestyle=boundary_line_style,
                              linewidth=boundary_line_width,
                              alpha=boundary_line_alpha, zorder=10)
        
        # Track saved and displayed figures
        figures_saved = []
        figures_displayed = []
        individual_results = []
        
        # Process individual figures if requested
        if save_individual_figures or show_individual_figures:
            print(f"  📄 Processing individual figures...")
            
            for idx, pair in enumerate(pairs_to_plot):
                print(f"    Individual figure {idx + 1}/{len(pairs_to_plot)}: {pair}")
                
                # Create individual figure
                fig_ind = plt.figure(figsize=individual_figsize, dpi=dpi)
                ax_ind = fig_ind.add_subplot(111)
                
                pair_data = rdf_data[pair]
                r_centers = pair_data['r_centers']
                z_centers = pair_data['z_centers']
                rdf_matrix = pair_data['rdf_matrix']
                
                # Plot based on axis orientation
                if swap_axes:
                    # z on x-axis (horizontal), r on y-axis (vertical)
                    im = ax_ind.contourf(z_centers, r_centers, rdf_matrix.T,
                                        levels=20, cmap=cmap)
                    ax_ind.set_xlabel('z (Å)', fontsize=label_fontsize)
                    ax_ind.set_ylabel('r (Å)', fontsize=label_fontsize)
                    ax_ind.axvline(0, color='white', linestyle='--', alpha=0.7, linewidth=1)
                else:
                    # r on x-axis (horizontal), z on y-axis (vertical)
                    im = ax_ind.contourf(r_centers, z_centers, rdf_matrix,
                                        levels=20, cmap=cmap)
                    ax_ind.set_xlabel('r (Å)', fontsize=label_fontsize)
                    ax_ind.set_ylabel('z (Å)', fontsize=label_fontsize)
                    ax_ind.axhline(0, color='white', linestyle='--', alpha=0.7, linewidth=1)
                
                # Add title if requested
                if show_title:
                    ax_ind.set_title(f'RDF: {pair}', fontweight='bold', fontsize=title_fontsize)
                
                # Set tick label font sizes
                ax_ind.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                
                # Add colorbar
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax_ind)
                cax = divider.append_axes("right", size=colorbar_width, pad=colorbar_pad)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label('g(r)', fontsize=label_fontsize)
                cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
                
                # Draw shell boundary lines if requested
                if show_solvation_boundary or show_pairing_boundary:
                    draw_boundaries(ax_ind, pair, swap_axes)
                
                plt.tight_layout()
                
                # Save individual figure
                individual_filename = None
                if save_individual_figures:
                    # Simple filename that overwrites each time
                    individual_filename = f'rdf_vs_z_{pair}_individual.png'
                    try:
                        fig_ind.savefig(individual_filename, dpi=dpi, bbox_inches='tight', 
                                       pad_inches=0.2, facecolor='white')
                        figures_saved.append(individual_filename)
                        print(f"      💾 Saved: {individual_filename}")
                    except Exception as e:
                        print(f"      ⚠️ Error saving {individual_filename}: {e}")
                
                # Store individual figure info
                individual_results.append({
                    'pair': pair,
                    'figure': fig_ind,
                    'axis': ax_ind,
                    'filename': individual_filename
                })
                
                # Show individual figure if requested
                if show_individual_figures:
                    plt.show()
                    figures_displayed.append(f"{pair}_individual")
                    print(f"      🖼️ Displayed individual figure for {pair}")
                else:
                    plt.close(fig_ind)
        
        # Create combined figure if requested
        if show_combined_figure or save_combined_figure:
            print(f"  📊 Creating combined figure...")
            
            # Create subplots
            n_pairs = len(pairs_to_plot)
            n_cols = min(2, n_pairs)
            n_rows = (n_pairs + n_cols - 1) // n_cols
            
            # Use provided figsize or calculate based on number of subplots
            if figsize is None:
                figsize = (8 * n_cols, 6 * n_rows)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)
            
            for idx, pair in enumerate(pairs_to_plot):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]
                
                pair_data = rdf_data[pair]
                r_centers = pair_data['r_centers']
                z_centers = pair_data['z_centers']
                rdf_matrix = pair_data['rdf_matrix']
                
                if swap_axes:
                    # Plot with z on x-axis (horizontal) and r on y-axis (vertical)
                    im = ax.contourf(z_centers, r_centers, rdf_matrix.T,  # Transpose the matrix
                                    levels=20, cmap=cmap)
                    
                    ax.set_xlabel('z (Å)', fontsize=label_fontsize)
                    ax.set_ylabel('r (Å)', fontsize=label_fontsize)
                    ax.axvline(0, color='white', linestyle='--', alpha=0.7, linewidth=1)
                else:
                    # Standard: r on x-axis (horizontal) and z on y-axis (vertical)
                    im = ax.contourf(r_centers, z_centers, rdf_matrix,
                                    levels=20, cmap=cmap)
                    
                    ax.set_xlabel('r (Å)', fontsize=label_fontsize)
                    ax.set_ylabel('z (Å)', fontsize=label_fontsize)
                    ax.axhline(0, color='white', linestyle='--', alpha=0.7, linewidth=1)
                
                if show_title:
                    ax.set_title(f'RDF: {pair}', fontweight='bold', fontsize=title_fontsize)
                
                # Set tick label font sizes
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                
                # Add colorbar
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size=colorbar_width, pad=colorbar_pad)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label('g(r)', fontsize=label_fontsize)
                cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
                
                # Draw shell boundary lines if requested
                if show_solvation_boundary or show_pairing_boundary:
                    draw_boundaries(ax, pair, swap_axes)
            
            # Hide unused subplots
            for idx in range(n_pairs, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # Save combined figure
            if save_combined_figure:
                if filename is None:
                    filename = 'rdf_vs_z_combined.png'
                
                try:
                    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
                    figures_saved.append(filename)
                    print(f"  💾 Saved combined: {filename}")
                except Exception as e:
                    print(f"  ⚠️ Error saving combined figure: {e}")
            
            # Show combined figure
            if show_combined_figure:
                plt.show()
                figures_displayed.append("combined")
                print(f"  🖼️ Displayed combined figure")
            else:
                plt.close(fig)
        
        # Legacy save_plots behavior (deprecated but maintained for backward compatibility)
        elif save_plots and not save_individual_figures and not save_combined_figure:
            print("  ⚠️ save_plots is deprecated. Use save_individual_figures and save_combined_figure instead.")
            if filename is None:
                filename = 'rdf_vs_z.png'
            
            # Create a basic combined figure using old logic
            n_pairs = len(pairs_to_plot)
            n_cols = min(2, n_pairs)
            n_rows = (n_pairs + n_cols - 1) // n_cols
            
            if figsize is None:
                figsize = (8 * n_cols, 6 * n_rows)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
            
            for idx, pair in enumerate(pairs_to_plot):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]
                
                pair_data = rdf_data[pair]
                r_centers = pair_data['r_centers']
                z_centers = pair_data['z_centers']
                rdf_matrix = pair_data['rdf_matrix']
                
                if swap_axes:
                    im = ax.contourf(z_centers, r_centers, rdf_matrix.T, levels=20, cmap=cmap)
                    ax.set_xlabel('z (Å)', fontsize=label_fontsize)
                    ax.set_ylabel('r (Å)', fontsize=label_fontsize)
                    ax.axvline(0, color='white', linestyle='--', alpha=0.7, linewidth=1)
                else:
                    im = ax.contourf(r_centers, z_centers, rdf_matrix, levels=20, cmap=cmap)
                    ax.set_xlabel('r (Å)', fontsize=label_fontsize)
                    ax.set_ylabel('z (Å)', fontsize=label_fontsize)
                    ax.axhline(0, color='white', linestyle='--', alpha=0.7, linewidth=1)
                
                if show_title:
                    ax.set_title(f'RDF: {pair}', fontweight='bold', fontsize=title_fontsize)
                
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('g(r)', fontsize=label_fontsize)
                cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
                
                # Draw shell boundary lines if requested
                if show_solvation_boundary or show_pairing_boundary:
                    draw_boundaries(ax, pair, swap_axes)
            
            for idx in range(n_pairs, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            figures_saved.append(filename)
            print(f"  RDF vs z plot saved as: {filename}")
            plt.show()
        
        # Print summary
        print(f"\n✅ Plot generation complete!")
        print(f"  Plot settings:")
        print(f"    Colormap: {cmap}")
        print(f"    Orientation: {'z (horizontal) vs r (vertical)' if swap_axes else 'r (horizontal) vs z (vertical)'}")
        print(f"    DPI: {dpi}")
        if show_solvation_boundary:
            print(f"    Solvation shell boundaries: Enabled")
        if show_pairing_boundary:
            print(f"    Ion pairing boundaries: Enabled")
        if save_individual_figures:
            print(f"    Individual figures saved: {len([f for f in figures_saved if 'individual' in f])}")
        if save_combined_figure:
            print(f"    Combined figure saved: Yes")
        if show_individual_figures:
            print(f"    Individual figures displayed: {len([f for f in figures_displayed if 'individual' in f])}")
        if show_combined_figure:
            print(f"    Combined figure displayed: Yes")
        
        # Return results dictionary
        return {
            'individual_figures': individual_results,
            'figures_saved': figures_saved,
            'figures_displayed': figures_displayed,
            'n_pairs': len(pairs_to_plot),
            'pairs_plotted': pairs_to_plot
        }



    def calculate_electrostatic_potential_vs_z(self, charge_dict=None, 
                                            method='poisson_1d',
                                            dielectric_constant=78.0,
                                            boundary_condition='periodic',
                                            units='kT_per_e',
                                            calculate_electric_field=True):
        """
        Calculate electrostatic potential profile along z-axis using proper physics.
        
        Parameters
        ----------
        charge_dict : dict, optional
            Ion charges in elementary charge units
        method : str, default='poisson_1d'
            Calculation method:
            - 'poisson_1d': Solve 1D Poisson equation properly
            - 'green_function': Use Green's function approach
        dielectric_constant : float, default=78.0
            Relative permittivity of water
        boundary_condition : str, default='periodic'
            Boundary conditions: 'periodic', 'zero', 'neutral'
        units : str, default='kT_per_e'
            Output units: 'kT_per_e' (kT/e), 'volts', 'eV'
        calculate_electric_field : bool, default=True
            If True, also calculate electric field E = -dV/dz
        
        Returns
        -------
        potential_profile : np.ndarray
            Electrostatic potential at each z-position
        """
        
        if charge_dict is None:
            charge_dict = {'NA': 1.0, 'CL': -1.0, 'MG': 2.0, 'CA': 2.0, 
                        'K': 1.0, 'BR': -1.0, 'F': -1.0, 'I': -1.0}
        
        print(f"Calculating electrostatic potential profile using {method} method...")
        print(f"Dielectric constant: ε_r = {dielectric_constant}")
        print(f"Boundary condition: {boundary_condition}")
        print(f"Output units: {units}")
        if calculate_electric_field:
            print(f"Electric field calculation: ENABLED")
        
        # Physical constants
        k_B = 1.381e-23  # J/K
        T = 300.0        # K (room temperature)
        e = 1.602e-19    # C (elementary charge)
        epsilon_0 = 8.854e-12  # F/m (vacuum permittivity)
        
        # Calculate charge density profile
        charge_density_profile = np.zeros(self.n_bins)
        
        print("Building charge density profile...")
        ion_contributions = {}
        
        for ion_type, density in self.results.ion_densities_by_type.items():
            # Find matching charge (more robust matching)
            charge = 0
            for charge_key, charge_val in charge_dict.items():
                if (charge_key.upper() in ion_type.upper() or 
                    ion_type.upper() in charge_key.upper()):
                    charge = charge_val
                    break
            
            if charge != 0:
                ion_contribution = density * charge
                charge_density_profile += ion_contribution
                ion_contributions[ion_type] = {
                    'charge': charge,
                    'density': density,
                    'contribution': ion_contribution
                }
                
                print(f"  {ion_type}: charge = {charge:+.1f}e, "
                    f"max density = {np.max(density):.4f}, "
                    f"max contribution = {np.max(ion_contribution):.4f}")
        
        # Check charge neutrality
        total_charge = np.trapz(charge_density_profile, dx=self.bin_width)
        print(f"Total integrated charge: {total_charge:.6f} e·Å")
        
        if abs(total_charge) > 1e-3:
            print(f"⚠️  Warning: System is not charge neutral (total = {total_charge:.6f})")
            print("   Consider adjusting charge_dict or checking ion assignments")
        
        # Convert to proper units for calculation
        # Charge density in e/Å³ → C/m³
        charge_density_SI = charge_density_profile * e / (1e-10)**3  # C/m³
        dz = self.bin_width * 1e-10  # Convert Å to m
        
        if method == 'poisson_1d':
            # Solve 1D Poisson equation: d²φ/dz² = -ρ(z)/(ε₀ε_r)
            potential_SI = self._solve_poisson_1d(charge_density_SI, dz, 
                                                dielectric_constant, boundary_condition)
            
        elif method == 'green_function':
            # Use Green's function approach for 1D
            potential_SI = self._green_function_1d(charge_density_SI, dz, 
                                                dielectric_constant)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to requested units
        if units == 'kT_per_e':
            # Convert V to kT/e units
            potential_profile = potential_SI * e / (k_B * T)
            unit_label = "kT/e"
            
        elif units == 'volts':
            potential_profile = potential_SI
            unit_label = "V"
            
        elif units == 'eV':
            potential_profile = potential_SI * e / e  # V to eV
            unit_label = "eV"
            
        else:
            raise ValueError(f"Unknown units: {units}")
        
        # Calculate electric field if requested
        electric_field = None
        if calculate_electric_field:
            # E = -dV/dz (negative gradient of potential)
            electric_field = -np.gradient(potential_profile, self.z_centers)
            
            print(f"Electric field calculated:")
            print(f"  Field range: {np.min(electric_field):.3f} to {np.max(electric_field):.3f} {unit_label}/Å")
            print(f"  Max field magnitude: {np.max(np.abs(electric_field)):.3f} {unit_label}/Å")
        
        # Store results
        self.results.electrostatic_potential = potential_profile
        self.results.charge_density = charge_density_profile
        
        # Store electric field if calculated
        if calculate_electric_field:
            self.results.electric_field = electric_field
        
        self.results.electrostatic_potential_metadata = {
            'method': method,
            'dielectric_constant': dielectric_constant,
            'boundary_condition': boundary_condition,
            'units': units,
            'total_charge': total_charge,
            'ion_contributions': ion_contributions,
            'temperature': T,
            'charge_neutrality_check': abs(total_charge) < 1e-3,
            'electric_field_calculated': calculate_electric_field
        }
        
        print(f"\nElectrostatic potential calculated:")
        print(f"  Method: {method}")
        print(f"  Potential range: {np.min(potential_profile):.3f} to {np.max(potential_profile):.3f} {unit_label}")
        print(f"  Charge density range: {np.min(charge_density_profile):.6f} to {np.max(charge_density_profile):.6f} e/Å³")
        print(f"  System charge neutrality: {'✓' if abs(total_charge) < 1e-3 else '✗'}")
        
        return potential_profile




    def plot_electrostatic_potential(self, save_plots=False, show_electric_field=True, 
                                electric_field_units='same_as_potential'):
        """
        Plot electrostatic potential profile with optional electric field
        
        Parameters
        ----------
        save_plots : bool
            Save plots to files
        show_electric_field : bool, default=True
            If True, adds electric field subplot
        electric_field_units : str, default='same_as_potential'
            Units for electric field:
            - 'same_as_potential': Use same units as potential (kT/e/Å, V/Å, eV/Å)
            - 'V_per_angstrom': Always show as V/Å
            - 'kT_per_e_per_angstrom': Always show as kT/e/Å
        """
       
        self._validate_analysis()
        
        if not hasattr(self.analysis.results, 'electrostatic_potential'):
            print("No electrostatic potential data available. Run calculate_electrostatic_potential_vs_z() first.")
            return
        
        # Determine number of subplots
        n_subplots = 3 if show_electric_field else 2
        
        fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 4*n_subplots), sharex=True)
        if n_subplots == 1:
            axes = [axes]
        
        z_centers = self.analysis.z_centers
        
        # Plot 1: Charge density
        ax1 = axes[0]
        charge_density = self.analysis.results.charge_density
        
        ax1.plot(z_centers, charge_density, 'r-', linewidth=2, label='Total charge density')
        ax1.set_ylabel('Charge Density (e/Å³)')
        ax1.set_title('Charge Density Profile', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax1.legend()
        
        # Add individual ion contributions if available
        if hasattr(self.analysis.results, 'electrostatic_potential_metadata'):
            metadata = self.analysis.results.electrostatic_potential_metadata
            ion_contributions = metadata.get('ion_contributions', {})
            
            colors = ['blue', 'green', 'orange', 'purple', 'brown']
            for i, (ion_type, contrib_data) in enumerate(ion_contributions.items()):
                if 'contribution' in contrib_data:
                    color = colors[i % len(colors)]
                    ax1.plot(z_centers, contrib_data['contribution'], 
                            color=color, linestyle='--', alpha=0.7, 
                            label=f'{ion_type} ({contrib_data["charge"]:+.1f}e)')
            
            ax1.legend()
        
        # Plot 2: Electrostatic potential
        ax2 = axes[1]
        potential = self.analysis.results.electrostatic_potential
        
        # Get units from metadata
        units = 'kT/e'  # default
        if hasattr(self.analysis.results, 'electrostatic_potential_metadata'):
            units = self.analysis.results.electrostatic_potential_metadata.get('units', 'kT/e')
        
        ax2.plot(z_centers, potential, 'b-', linewidth=2, label=f'Electrostatic potential')
        ax2.set_ylabel(f'Electrostatic Potential ({units})')
        ax2.set_title('Electrostatic Potential Profile', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax2.legend()
        
        # Plot 3: Electric field (if requested)
        if show_electric_field:
            ax3 = axes[2]
            
            # Get electric field from results or calculate it
            if hasattr(self.analysis.results, 'electric_field'):
                electric_field = self.analysis.results.electric_field
            else:
                # Calculate electric field as negative gradient of potential
                electric_field = -np.gradient(potential, z_centers)
            
            # Determine electric field units
            if electric_field_units == 'same_as_potential':
                if units == 'kT/e':
                    field_units = 'kT/e/Å'
                elif units == 'V':
                    field_units = 'V/Å'
                elif units == 'eV':
                    field_units = 'eV/Å'
                else:
                    field_units = f'{units}/Å'
            elif electric_field_units == 'V_per_angstrom':
                field_units = 'V/Å'
                # Convert if necessary (would need conversion factors)
            elif electric_field_units == 'kT_per_e_per_angstrom':
                field_units = 'kT/e/Å'
                # Convert if necessary (would need conversion factors)
            else:
                field_units = f'{units}/Å'
            
            ax3.plot(z_centers, electric_field, 'g-', linewidth=2, label='Electric field')
            ax3.set_xlabel('z (Å)')
            ax3.set_ylabel(f'Electric Field ({field_units})')
            ax3.set_title('Electric Field Profile (E = -dV/dz)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.axvline(0, color='k', linestyle='--', alpha=0.5)
            ax3.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax3.legend()
            
            # Add colored regions for field direction
            positive_field = electric_field > 0
            negative_field = electric_field < 0
            
            if np.any(positive_field):
                ax3.fill_between(z_centers, 0, electric_field, 
                            where=positive_field, color='red', alpha=0.2, 
                            label='E > 0 (field points +z)')
            
            if np.any(negative_field):
                ax3.fill_between(z_centers, 0, electric_field, 
                            where=negative_field, color='blue', alpha=0.2, 
                            label='E < 0 (field points -z)')
            
            ax3.legend()
        
        else:
            # If not showing electric field, add xlabel to potential plot
            ax2.set_xlabel('z (Å)')
        
        # Add clay boundaries if available
        if hasattr(self.analysis.results, 'clay_interface_boundaries'):
            clay_boundaries = self.analysis.results.clay_interface_boundaries
            
            for ax in axes:
                # Add main clay boundaries
                for boundary_key in ['clay_average_z_positive', 'clay_average_z_negative']:
                    if boundary_key in clay_boundaries and clay_boundaries[boundary_key] is not None:
                        z_pos = clay_boundaries[boundary_key]
                        ax.axvline(z_pos, color='brown', linestyle='-', alpha=0.8, linewidth=2)
                
                # Add Si layer boundaries
                for boundary_key in ['si_average_z_positive', 'si_average_z_negative']:
                    if boundary_key in clay_boundaries and clay_boundaries[boundary_key] is not None:
                        z_pos = clay_boundaries[boundary_key]
                        ax.axvline(z_pos, color='orange', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        if save_plots:
            suffix = '_with_efield' if show_electric_field else ''
            filename = f'electrostatic_potential{suffix}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Electrostatic potential plot saved as: {filename}")
        
        plt.show()
        
        # Print summary statistics
        print(f"\nElectrostatic Analysis Summary:")
        print(f"  Potential range: {np.min(potential):.3f} to {np.max(potential):.3f} {units}")
        
        if show_electric_field:
            if hasattr(self.analysis.results, 'electric_field'):
                electric_field = self.analysis.results.electric_field
            else:
                electric_field = -np.gradient(potential, z_centers)
            
            print(f"  Electric field range: {np.min(electric_field):.3f} to {np.max(electric_field):.3f} {field_units}")
            print(f"  Max field magnitude: {np.max(np.abs(electric_field)):.3f} {field_units}")
            
            # Find field peaks
            field_peaks = find_peaks(np.abs(electric_field), height=np.max(np.abs(electric_field))*0.5)
            if len(field_peaks[0]) > 0:
                print(f"  Strong field regions at z = {z_centers[field_peaks[0]]}")
   

   
    def plot_spatial_binding_interactive(self, spatial_results, structure_file,
                                        universe=None, density_threshold=0.02,
                                        sphere_size=0.4, sphere_opacity=0.3,
                                        stick_radius=0.15, ball_scale=0.3,
                                        width=800, height=600,
                                        show_output=True, max_spheres=500):
        """
        Create interactive 3D visualization showing molecule + ion binding positions in space.
        
        This shows:
        1. Target molecule as ball-and-stick structure (the actual molecule)
        2. Spheres in 3D space at actual ion binding locations (not at molecule atoms)
        
        **Requires py3Dmol**: Install with `pip install py3Dmol`
        
        Parameters
        ----------
        spatial_results : dict
            Results from spatial_binding_analysis() with return_positions=True
            Must contain 'ion_positions_relative' key with stored ion coordinates
        structure_file : str
            Path to PDB structure file for the target molecule
        universe : MDAnalysis.Universe, optional
            Universe object (used for reference only)
        density_threshold : float
            Minimum density to show a binding position sphere (0-1). Default 0.02
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
        
        # Check for ion positions
        if 'ion_positions_relative' not in spatial_results:
            raise ValueError(
                "spatial_results must contain 'ion_positions_relative'.\n"
                "Re-run spatial_binding_analysis() with return_positions=True"
            )
        
        # Concatenate ion positions from all frames
        ion_positions_list = spatial_results['ion_positions_relative']
        if len(ion_positions_list) == 0:
            raise ValueError("No ion positions found in spatial_results")
        
        # Flatten list of arrays into single array
        ion_positions = np.vstack(ion_positions_list)
        
        if show_output:
            print(f"\n{'='*60}")
            print(f"Spatial Binding Visualization")
            print(f"{'='*60}")
            print(f"Total ion positions recorded: {len(ion_positions)}")
        
        # Read PDB structure
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
        
        if show_output:
            print(f"✓ Molecule COM in PDB: [{molecule_com[0]:.2f}, {molecule_com[1]:.2f}, {molecule_com[2]:.2f}]")
        
        # Shift ion positions from origin to molecule COM
        # Ion positions are relative to COM (centered at origin), so we add the COM offset
        ion_positions_shifted = ion_positions + molecule_com
        
        if show_output:
            print(f"✓ Shifted {len(ion_positions)} ion positions to molecule COM")
        
        # Create viewer
        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_string, 'pdb')
        
        # Style molecule as ball-and-stick (normal molecular structure)
        view.setStyle({}, {'stick': {'radius': stick_radius, 'color': 'lightgray'},
                          'sphere': {'radius': ball_scale, 'color': 'spectrum'}})
        
        if show_output:
            print(f"✓ Molecule rendered as ball-and-stick")
        
        # Calculate spatial density of ion positions using neighbor counting
        from scipy.spatial import cKDTree
        
        # Build KDTree for efficient neighbor searches (use shifted positions)
        tree = cKDTree(ion_positions_shifted)
        
        # Calculate local density for each position (neighbors within 2 Å)
        search_radius = 2.0
        densities = np.array([len(tree.query_ball_point(pos, search_radius)) 
                             for pos in ion_positions_shifted])
        
        # Normalize densities
        max_density = densities.max()
        densities_norm = densities / max_density
        
        # Filter by threshold
        mask = densities_norm >= density_threshold
        filtered_positions = ion_positions_shifted[mask]
        filtered_densities = densities_norm[mask]
        
        if show_output:
            print(f"Density range: {densities.min():.1f} - {densities.max():.1f} neighbors")
            print(f"Filtered to {len(filtered_positions)} positions (threshold: {density_threshold*100:.1f}%)")
        
        # Subsample if too many points (for performance)
        if len(filtered_positions) > max_spheres:
            # Keep highest density points
            top_indices = np.argsort(filtered_densities)[-max_spheres:]
            filtered_positions = filtered_positions[top_indices]
            filtered_densities = filtered_densities[top_indices]
            if show_output:
                print(f"Subsampled to {max_spheres} highest-density positions")
        
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
        
        if show_output:
            print(f"✓ Added {len(filtered_positions)} ion binding spheres in space")
            print(f"\n{'='*60}")
            print(f"✓ Interactive 3D viewer ready!")
            print(f"{'='*60}")
            print(f"Controls:")
            print(f"  • Click and drag to rotate")
            print(f"  • Scroll to zoom")
            print(f"  • Right-click and drag to pan")
            print(f"\nColor scheme:")
            print(f"  🔵 Blue spheres = Low-density binding regions")
            print(f"  🟣 Purple/White spheres = Medium-density regions")
            print(f"  🔴 Red spheres = High-density binding regions")
            print(f"\nMolecule: Ball-and-stick structure (colored by atom type)")
        
        view.zoomTo()
        return view



    def molecular_rdf(self, group1_sel, group2_sel, bin_width=0.05, range=(0, 15), 
                     step=1, njobs=1, center_method=None, normalize=True,
                     save_cache=True, cache_file=None, force_rerun=False):
        '''
        Calculate RDF between arbitrary molecular groups with flexible centering options.
        Now includes automatic ion type handling and caching support.
        
        Parameters
        ----------
        group1_sel : str
            Selection string for first group
        group2_sel : str
            Selection string for second group
        bin_width : float
            Width of RDF bins in Angstroms
        range : tuple
            (min, max) distance range for RDF
        step : int
            Trajectory frame step
        njobs : int
            Number of parallel jobs
        center_method : str or None
            'COM', 'COG', 'atom', or None (uses self.center_method)
        normalize : bool
            Whether to normalize RDF
        save_cache : bool
            Whether to save results to cache file
        cache_file : str or None
            Custom cache filename. If None, auto-generates from parameters
        force_rerun : bool
            Force recalculation even if cache exists
        
        Returns
        -------
        results : object
            RDF results with .bins, .rdf, .count, .edges attributes
        '''
        
        import hashlib
        import os
        
        if center_method is None:
            center_method = self.center_method
        
        # Generate cache filename if not provided
        if cache_file is None:
            # Create hash from parameters for unique cache filename
            param_str = f"{group1_sel}_{group2_sel}_bw{bin_width}_r{range[0]}-{range[1]}_s{step}_cm{center_method}_n{normalize}"
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            cache_file = f"rdf_cache_{param_hash}.npz"
        
        # Check for existing cache
        if save_cache and not force_rerun and os.path.exists(cache_file):
            print(f"📂 Found existing RDF cache: {cache_file}")
            try:
                print("   Loading cached RDF results...")
                cached_data = np.load(cache_file, allow_pickle=True)
                
                # Reconstruct results object
                class RDFResults:
                    def __init__(self, bins, rdf, count, edges):
                        self.bins = bins
                        self.rdf = rdf
                        self.count = count
                        self.edges = edges
                
                results = RDFResults(
                    bins=cached_data['bins'],
                    rdf=cached_data['rdf'],
                    count=cached_data['count'],
                    edges=cached_data['edges']
                )
                
                print(f"   ✅ Loaded cached RDF successfully!")
                print(f"   RDF range: {results.bins[0]:.2f} - {results.bins[-1]:.2f} Å")
                return results
                
            except Exception as e:
                print(f"   ⚠️ Failed to load cache: {e}")
                print(f"   Recalculating RDF...")
        
        # Perform RDF calculation
        group1 = self.universe.select_atoms(group1_sel)
        group2 = self.universe.select_atoms(group2_sel)
        
        if len(group1) == 0 or len(group2) == 0:
            raise ValueError("One or both atom groups are empty")
        
        nbins = int((range[1] - range[0]) / bin_width)
        
        if center_method in ['COM', 'COG']:
            results = self._rdf_with_centers(group1, group2, nbins, range, step, 
                                        njobs, center_method, normalize)
        else:
            rdf = InterRDF(group1, group2, nbins=nbins, range=range, 
                          norm='rdf' if normalize else 'none', verbose=True)
            rdf.run(step=step, njobs=njobs)
            results = rdf.results
        
        # Save to cache if requested
        if save_cache:
            print(f"\n💾 Saving RDF to cache: {cache_file}")
            try:
                np.savez(cache_file,
                        bins=results.bins,
                        rdf=results.rdf,
                        count=results.count,
                        edges=results.edges,
                        # Metadata
                        group1_sel=group1_sel,
                        group2_sel=group2_sel,
                        bin_width=bin_width,
                        range=range,
                        step=step,
                        center_method=center_method,
                        normalize=normalize)
                print(f"   ✅ Cache saved successfully!")
                print(f"   📁 File: {cache_file}")
            except Exception as e:
                print(f"   ⚠️ Failed to save cache: {e}")
                print(f"   Continuing without cache...")
        
        return results



    def plot_ion_binding_comparison(self, binding_results_dict, 
                                   # Overall plot control
                                   title='Ion Binding Comparison',
                                   subplot_layout='horizontal',  # 'horizontal', 'vertical', or 'single'
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
        Plot comparison of ion binding across multiple targets in a single figure with grouped bars
        
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
            