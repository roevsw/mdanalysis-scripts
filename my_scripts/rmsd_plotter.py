#!/usr/bin/env python3
"""
Visualization tools for RMSD density-based clustering.

This module provides specialized plotting classes for visualizing RMSD clustering
results with various techniques including density scatter plots, heatmaps, contours,
and cluster assignments.

Classes:
    RMSDPlotter: Main class for creating publication-quality visualizations

Example:
    >>> from density_scatter import RMSDClusterAnalyzer
    >>> analyzer = RMSDClusterAnalyzer()
    >>> analyzer.load_rmsd_data("flat.xvg", "cross.xvg")
    >>> analyzer.compute_density()
    >>> 
    >>> plotter = RMSDPlotter(analyzer)
    >>> plotter.plot_density_scatter_with_contours()
    >>> plotter.plot_combined_view()
"""
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from scipy.integrate import trapz

try:
    from density_scatter import RMSDClusterAnalyzer
except ImportError:
    # Handle relative import if needed
    pass


class RMSDPlotter:
    """
    Publication-quality visualization tools for RMSD clustering analysis.
    
    This class provides various plotting methods for visualizing RMSD density maps,
    clusters, and related analyses. Supports multiple plot types including scatter
    plots, heatmaps, contour plots, and combined views.
    
    Attributes:
        analyzer (RMSDClusterAnalyzer): The analyzer object containing data and results
        default_figsize (tuple): Default figure size for plots
        default_dpi (int): Default DPI for saved figures
        default_cmap (str): Default colormap
        
    Example:
        >>> analyzer = RMSDClusterAnalyzer()
        >>> analyzer.load_rmsd_data("flat.xvg", "cross.xvg")
        >>> analyzer.compute_density()
        >>> 
        >>> plotter = RMSDPlotter(analyzer)
        >>> plotter.plot_density_scatter()
        >>> plotter.plot_contour_map()
    """
    
    def __init__(self, analyzer: 'RMSDClusterAnalyzer',
                 figsize: Tuple[float, float] = (9, 7),
                 dpi: int = 300,
                 cmap: str = 'viridis',
                 font_family: str = 'Times New Roman',
                 font_size: int = 12):
        """
        Initialize the RMSD plotter.
        
        Args:
            analyzer: RMSDClusterAnalyzer instance with loaded data
            figsize: Default figure size (width, height)
            dpi: Default DPI for saved figures
            cmap: Default colormap name
            font_family: Default font family for all text in plots (default='Times New Roman')
            font_size: Default base font size for all plots (default=12)
            
        Example:
            >>> plotter = RMSDPlotter(analyzer, figsize=(8, 6), dpi=300, 
            ...                       cmap='jet', font_family='Times New Roman')
        """
        self.analyzer = analyzer
        self.default_figsize = figsize
        self.default_dpi = dpi
        self.default_cmap = cmap
        self.font_family = font_family
        self.font_size = font_size
        
        # Configure matplotlib font defaults for ALL plots created by this plotter
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.labelsize'] = font_size
        plt.rcParams['axes.titlesize'] = font_size + 2
        plt.rcParams['xtick.labelsize'] = font_size - 1
        plt.rcParams['ytick.labelsize'] = font_size - 1
        plt.rcParams['legend.fontsize'] = font_size - 1
        plt.rcParams['figure.titlesize'] = font_size + 3
    
    def _validate_data(self, require_density: bool = False, 
                      require_clusters: bool = False) -> None:
        """
        Validate that required data is available.
        
        Args:
            require_density: Check if density map is computed
            require_clusters: Check if clusters are identified
            
        Raises:
            ValueError: If required data is not available
        """
        if self.analyzer.rmsd_flat is None or self.analyzer.rmsd_cross is None:
            raise ValueError("RMSD data not loaded in analyzer")
        
        if require_density and self.analyzer.density_map is None:
            raise ValueError("Density not computed. Call analyzer.compute_density() first")
        
        if require_clusters and (self.analyzer.cluster_labels is None or 
                                self.analyzer.cluster_centers is None):
            raise ValueError("Clusters not found. Call analyzer.find_clusters() first")
    
    def plot_density_scatter(self, figsize: Optional[Tuple[float, float]] = None,
                            marker: str = 's', msize: float = 10,
                            cmap: Optional[str] = None, alpha: float = 0.8,
                            show_colorbar: bool = True,
                            
                            # Font styling
                            title_fontsize: int = 14,
                            title_fontweight: str = 'bold',
                            label_fontsize: int = 12,
                            label_fontweight: str = 'bold',
                            tick_fontsize: int = 9,
                            tick_fontweight: str = 'normal',
                            colorbar_label_fontsize: int = 11,
                            colorbar_label_fontweight: str = 'normal',
                            
                            save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create density scatter plot with points colored by local density.
        
        Args:
            figsize: Figure size (uses default if None)
            marker: Marker style ('s', 'o', '.', etc.)
            msize: Marker size
            cmap: Colormap name (uses default if None)
            alpha: Point transparency (0-1)
            show_colorbar: Whether to show colorbar
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        self._validate_data(require_density=True)
        
        figsize = figsize or self.default_figsize
        cmap = cmap or self.default_cmap
        
        colors, valid = self.analyzer.get_point_densities()
        
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(
            self.analyzer.rmsd_flat[valid], 
            self.analyzer.rmsd_cross[valid],
            s=msize, c=colors[valid], marker=marker,
            cmap=cmap, vmin=0, vmax=self.analyzer._target_max,
            alpha=alpha, edgecolors='none'
        )
        
        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Density (counts per bin)', fontsize=colorbar_label_fontsize, fontweight=colorbar_label_fontweight)
        
        ax.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_title('RMSD Density Scatter', fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontweight(tick_fontweight)
        ax.grid(alpha=0.3, ls='--', lw=0.5)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_heatmap(self, figsize: Optional[Tuple[float, float]] = None,
                    cmap: Optional[str] = None,
                    interpolation: str = 'bilinear',
                    show_colorbar: bool = True,
                    
                    # Font styling
                    title_fontsize: int = 14,
                    title_fontweight: str = 'bold',
                    label_fontsize: int = 12,
                    label_fontweight: str = 'bold',
                    tick_fontsize: int = 9,
                    tick_fontweight: str = 'normal',
                    colorbar_label_fontsize: int = 11,
                    colorbar_label_fontweight: str = 'normal',
                    
                    save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create 2D density heatmap.
        
        Args:
            figsize: Figure size (uses default if None)
            cmap: Colormap name (uses default if None)
            interpolation: Interpolation method ('bilinear', 'nearest', 'gaussian')
            show_colorbar: Whether to show colorbar
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        self._validate_data(require_density=True)
        
        figsize = figsize or self.default_figsize
        cmap = cmap or self.default_cmap
        
        fig, ax = plt.subplots(figsize=figsize)
        extent = [
            self.analyzer.rmsd_flat.min(), self.analyzer.rmsd_flat.max(),
            self.analyzer.rmsd_cross.min(), self.analyzer.rmsd_cross.max()
        ]
        
        im = ax.imshow(
            self.analyzer.density_map,
            extent=extent, origin='lower', aspect='auto',
            cmap=cmap, vmin=0, vmax=self.analyzer._target_max,
            interpolation=interpolation
        )
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Density (counts per bin)', fontsize=colorbar_label_fontsize, fontweight=colorbar_label_fontweight)
        
        ax.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_title('2D Density Heatmap', fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontweight(tick_fontweight)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_contour_map(self, figsize: Optional[Tuple[float, float]] = None,
                        cmap: Optional[str] = None,
                        n_contours: int = 10,
                        filled: bool = True,
                        show_labels: bool = False,
                        
                        # Font styling
                        title_fontsize: int = 14,
                        title_fontweight: str = 'bold',
                        label_fontsize: int = 12,
                        label_fontweight: str = 'bold',
                        tick_fontsize: int = 9,
                        tick_fontweight: str = 'normal',
                        colorbar_label_fontsize: int = 11,
                        colorbar_label_fontweight: str = 'normal',
                        
                        save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create contour plot of density map.
        
        Args:
            figsize: Figure size (uses default if None)
            cmap: Colormap name (uses default if None)
            n_contours: Number of contour levels
            filled: Whether to fill contours
            show_labels: Whether to show contour labels
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        self._validate_data(require_density=True)
        
        figsize = figsize or self.default_figsize
        cmap = cmap or self.default_cmap
        
        fig, ax = plt.subplots(figsize=figsize)
        
        X, Y = np.meshgrid(self.analyzer.bin_centers_x, self.analyzer.bin_centers_y)
        
        if filled:
            contour = ax.contourf(X, Y, self.analyzer.density_map,
                                 levels=n_contours, cmap=cmap)
        else:
            contour = ax.contour(X, Y, self.analyzer.density_map,
                                levels=n_contours, cmap=cmap)
        
        if show_labels and not filled:
            ax.clabel(contour, inline=True, fontsize=8)
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Density (counts per bin)', fontsize=colorbar_label_fontsize, fontweight=colorbar_label_fontweight)
        
        ax.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_title('Density Contour Map', fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontweight(tick_fontweight)
        ax.grid(alpha=0.3, ls='--', lw=0.5)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_density_scatter_with_contours(self, 
                                          figsize: Optional[Tuple[float, float]] = None,
                                          marker: str = 's', msize: float = 8,
                                          cmap: Optional[str] = None,
                                          n_contours: int = 8,
                                          contour_alpha: float = 0.3,
                                          
                                          # Font styling
                                          title_fontsize: int = 14,
                                          title_fontweight: str = 'bold',
                                          label_fontsize: int = 12,
                                          label_fontweight: str = 'bold',
                                          tick_fontsize: int = 9,
                                          tick_fontweight: str = 'normal',
                                          colorbar_label_fontsize: int = 11,
                                          colorbar_label_fontweight: str = 'normal',
                                          
                                          save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create scatter plot overlaid with density contours.
        
        Args:
            figsize: Figure size (uses default if None)
            marker: Marker style
            msize: Marker size
            cmap: Colormap name (uses default if None)
            n_contours: Number of contour levels
            contour_alpha: Transparency of contour lines
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        self._validate_data(require_density=True)
        
        figsize = figsize or self.default_figsize
        cmap = cmap or self.default_cmap
        
        colors, valid = self.analyzer.get_point_densities()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        scatter = ax.scatter(
            self.analyzer.rmsd_flat[valid],
            self.analyzer.rmsd_cross[valid],
            s=msize, c=colors[valid], marker=marker,
            cmap=cmap, vmin=0, vmax=self.analyzer._target_max,
            alpha=0.6, edgecolors='none'
        )
        
        # Contours
        X, Y = np.meshgrid(self.analyzer.bin_centers_x, self.analyzer.bin_centers_y)
        contour = ax.contour(X, Y, self.analyzer.density_map,
                           levels=n_contours, colors='black',
                           alpha=contour_alpha, linewidths=1)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Density (counts per bin)', fontsize=colorbar_label_fontsize, fontweight=colorbar_label_fontweight)
        
        ax.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_title('Density Scatter with Contours', fontsize=title_fontsize, fontweight=title_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontweight(tick_fontweight)
        ax.grid(alpha=0.3, ls='--', lw=0.5)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_clusters(self, figsize: Optional[Tuple[float, float]] = None,
                     msize: float = 10,
                     show_centers: bool = True,
                     center_size: float = 300,
                     legends: bool = True,
                     legend_frame_alpha: float = 0.2,
                     
                     # Cluster labeling options
                     label_clusters: bool = False,
                     cluster_label_fontsize: int = 18,
                     cluster_label_fontweight: str = 'bold',
                     cluster_label_color: str = 'black',
                     
                     # Density overlay options
                     overlay_density: bool = False,
                     overlay_type: str = 'contour',
                     overlay_cmap: Optional[str] = None,
                     overlay_alpha: float = 0.5,
                     contour_levels: int = 10,
                     contour_linewidth: float = 1.5,
                     contour_alpha: float = 0.8,
                     contour_color: Optional[str] = 'black',
                     overlay_smooth: bool = True,
                     smooth_sigma: float = 1.0,
                     
                     show_title: bool = True,
                     title_fontsize: int = 22,
                     title_fontweight: str = 'bold',
                     label_fontsize: int = 22,
                     label_fontweight: str = 'bold',
                     tick_fontsize: int = 18,
                     tick_fontweight: str = 'normal',
                     legend_fontsize: int = 18,
                     legend_fontweight: str = 'bold',
                     show_grid: bool = True,
                     grid_alpha: float = 0.2,
                     save_fig: bool = True,
                     save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot data with cluster assignments, optionally with density overlay.
        
        Args:
            figsize: Figure size (uses default if None)
            msize: Marker size for data points
            show_centers: Whether to show cluster centers
            center_size: Size of cluster center markers
            legends: Whether to show legend (default=True)
            legend_frame_alpha: Transparency of legend frame (default=0.2)
            
            # Cluster labeling options
            label_clusters: Whether to add cluster ID labels at cluster centers (default=False)
            cluster_label_fontsize: Font size for cluster labels (default=18)
            cluster_label_fontweight: Font weight for cluster labels (default='bold')
            cluster_label_color: Color for cluster labels (default='black')
            
            # Density overlay options
            overlay_density: Whether to overlay density map under clusters (default=False)
            overlay_type: Type of density overlay - 'contour', 'heatmap', or 'both' (default='contour')
                         'heatmap': continuous color map showing density
                         'contour': contour lines showing density levels
                         'both': heatmap background with contour lines on top
            overlay_cmap: Colormap for density overlay (default=None uses default_cmap)
            overlay_alpha: Transparency of density overlay (default=0.5)
            contour_levels: Number of contour levels if overlay_type='contour' or 'both' (default=10)
            contour_linewidth: Width of contour lines (default=1.5)
            contour_alpha: Transparency of contour lines (default=0.8)
            contour_color: Color for contour lines (default='black'), None uses overlay_cmap
            overlay_smooth: Apply edge smoothing for softer cluster boundaries (default=True)
                           For heatmap: creates soft alpha fade at edges (density stays sharp inside)
                           For contour: expands boundaries slightly for smoother look
            smooth_sigma: Edge smoothing strength (default=1.0, higher=softer edges)
            
            show_title: Whether to show title (default=True)
            title_fontsize: Font size for title (default=22)
            title_fontweight: Font weight for title (default='bold')
            label_fontsize: Font size for axis labels (default=22)
            label_fontweight: Font weight for axis labels (default='bold')
            tick_fontsize: Font size for tick labels (default=18)
            tick_fontweight: Font weight for tick labels (default='normal')
            legend_fontsize: Font size for legend (default=18)
            legend_fontweight: Font weight for legend (default='bold')
            show_grid: Whether to show grid (default=True)
            grid_alpha: Transparency of grid lines (default=0.2)
            save_fig: Whether to save figure if save_path provided (default=True)
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
            
        Example:
            >>> # Clusters with smooth-edged heatmap (sharp density inside, soft boundaries)
            >>> fig = plotter.plot_clusters(
            ...     overlay_density=True,
            ...     overlay_type='heatmap',
            ...     overlay_cmap='jet',
            ...     overlay_alpha=1.0,
            ...     overlay_smooth=True,  # Smooths edges only, not the density data
            ...     smooth_sigma=1.5      # Controls edge softness
            ... )
            >>> 
            >>> # Clusters with smooth-edged contours
            >>> fig = plotter.plot_clusters(
            ...     overlay_density=True,
            ...     overlay_type='contour',
            ...     overlay_cmap='viridis',
            ...     overlay_alpha=0.5,
            ...     contour_levels=10,
            ...     overlay_smooth=True  # Expands boundaries slightly for smoother look
            ... )
            >>> 
            >>> # Clusters with both heatmap and contour lines
            >>> fig = plotter.plot_clusters(
            ...     overlay_density=True,
            ...     overlay_type='both',  # Shows heatmap + contour lines
            ...     overlay_cmap='jet',
            ...     overlay_alpha=0.8,
            ...     contour_levels=8,
            ...     overlay_smooth=True
            ... )
        """
        self._validate_data(require_clusters=True)
        
        figsize = figsize or self.default_figsize
        overlay_cmap = overlay_cmap or self.default_cmap
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # === DENSITY OVERLAY (background layer - only in cluster regions) ===
        if overlay_density:
            self._validate_data(require_density=True)
            
            if overlay_type == 'heatmap':
                # Create a mask for cluster regions to clip the heatmap
                from scipy.spatial import ConvexHull
                from scipy.ndimage import gaussian_filter
                import matplotlib.path as mpath
                
                # Create grid for checking cluster membership
                extent = [self.analyzer.rmsd_flat.min(), self.analyzer.rmsd_flat.max(),
                         self.analyzer.rmsd_cross.min(), self.analyzer.rmsd_cross.max()]
                
                # Get grid dimensions from density map
                ny, nx = self.analyzer.density_map.shape
                x_grid = np.linspace(extent[0], extent[1], nx)
                y_grid = np.linspace(extent[2], extent[3], ny)
                X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                
                # Create mask for cluster regions
                cluster_mask = np.zeros_like(self.analyzer.density_map, dtype=bool)
                
                for i in range(len(self.analyzer.cluster_centers)):
                    mask = self.analyzer.cluster_labels == i
                    if np.sum(mask) < 3:  # Need at least 3 points
                        continue
                    
                    cluster_points = np.column_stack([
                        self.analyzer.rmsd_flat[mask],
                        self.analyzer.rmsd_cross[mask]
                    ])
                    
                    try:
                        hull = ConvexHull(cluster_points)
                        hull_path = mpath.Path(cluster_points[hull.vertices])
                        
                        # Check which grid points are inside this cluster hull
                        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                        inside = hull_path.contains_points(grid_points)
                        inside_grid = inside.reshape(X_grid.shape)
                        cluster_mask |= inside_grid
                    except:
                        continue
                
                # Mask the density map to only show cluster regions (keep density sharp)
                masked_density = np.ma.array(self.analyzer.density_map, mask=~cluster_mask)
                
                # Apply edge smoothing by creating alpha gradient only at boundaries
                if overlay_smooth:
                    from scipy.ndimage import binary_erosion, distance_transform_edt
                    
                    # Erode mask to get interior region (full alpha)
                    erosion_size = int(smooth_sigma * 2)  # Pixels to erode
                    interior_mask = binary_erosion(cluster_mask, iterations=erosion_size)
                    
                    # Create distance transform from edges inward
                    # This gives 0 at edges, increasing toward interior
                    dist_from_edge = distance_transform_edt(cluster_mask)
                    
                    # Normalize distance to create alpha: 0 at boundary, 1 at interior
                    max_dist = erosion_size if erosion_size > 0 else 1
                    edge_alpha = np.clip(dist_from_edge / max_dist, 0, 1)
                    
                    # Apply smoothing to the edge alpha for even softer transition
                    edge_alpha = gaussian_filter(edge_alpha, sigma=smooth_sigma * 0.5)
                    
                    # Final mask: 1.0 inside, gradient at edges, 0.0 outside
                    alpha_mask = edge_alpha
                    alpha_mask[~cluster_mask] = 0  # Zero outside clusters
                else:
                    alpha_mask = cluster_mask.astype(float)
                
                # Use actual min/max of cluster region densities for better color range
                cluster_region_values = masked_density[~masked_density.mask]
                vmin_local = np.min(cluster_region_values) if len(cluster_region_values) > 0 else 0
                vmax_local = np.max(cluster_region_values) if len(cluster_region_values) > 0 else self.analyzer._target_max
                
                # Plot the heatmap with smoothed edges (sharp density inside, soft boundaries)
                im = ax.imshow(self.analyzer.density_map, extent=extent, origin='lower',
                         aspect='auto', cmap=overlay_cmap,
                         vmin=vmin_local, vmax=vmax_local,
                         interpolation='nearest', zorder=0)
                # Apply the alpha mask: full inside, gradient at edges
                im.set_alpha(alpha_mask * overlay_alpha)
            
            elif overlay_type == 'contour':
                # Draw contours only within cluster regions
                from scipy.spatial import ConvexHull
                from scipy.ndimage import gaussian_filter
                import matplotlib.path as mpath
                
                X, Y = np.meshgrid(self.analyzer.bin_centers_x, self.analyzer.bin_centers_y)
                
                # Create a mask for cluster regions
                # Create combined mask for all cluster regions
                cluster_mask = np.zeros_like(self.analyzer.density_map, dtype=bool)
                
                for i in range(len(self.analyzer.cluster_centers)):
                    mask = self.analyzer.cluster_labels == i
                    if np.sum(mask) < 3:  # Need at least 3 points
                        continue
                    
                    cluster_points = np.column_stack([
                        self.analyzer.rmsd_flat[mask],
                        self.analyzer.rmsd_cross[mask]
                    ])
                    
                    try:
                        hull = ConvexHull(cluster_points)
                        hull_path = mpath.Path(cluster_points[hull.vertices])
                        
                        # Check which grid points are inside this cluster hull
                        grid_points = np.column_stack([X.ravel(), Y.ravel()])
                        inside = hull_path.contains_points(grid_points)
                        inside_grid = inside.reshape(X.shape)
                        cluster_mask |= inside_grid
                    except:
                        continue
                
                # Optionally expand/smooth the mask for softer contour boundaries
                if overlay_smooth:
                    # Smooth the mask to extend contours slightly beyond strict boundaries
                    mask_float = cluster_mask.astype(float)
                    smoothed_mask = gaussian_filter(mask_float, sigma=smooth_sigma)
                    # Use threshold to expand boundaries slightly
                    cluster_mask = smoothed_mask > 0.3  # Expand beyond original boundary
                
                # Mask the density map (keep density values sharp)
                masked_density = np.ma.array(self.analyzer.density_map, mask=~cluster_mask)
                
                # Use actual min/max of cluster region densities for better color range
                cluster_region_values = masked_density[~masked_density.mask]
                vmin_local = np.min(cluster_region_values) if len(cluster_region_values) > 0 else 0
                vmax_local = np.max(cluster_region_values) if len(cluster_region_values) > 0 else self.analyzer._target_max
                
                # Draw filled contours in cluster regions
                ax.contourf(X, Y, masked_density, levels=contour_levels,
                           cmap=overlay_cmap, vmin=vmin_local, vmax=vmax_local,
                           alpha=overlay_alpha, zorder=0)
                # Add contour lines for better definition
                if contour_color is not None:
                    ax.contour(X, Y, masked_density, levels=contour_levels,
                              colors=contour_color, linewidths=contour_linewidth,
                              alpha=contour_alpha, zorder=1)
                else:
                    ax.contour(X, Y, masked_density, levels=contour_levels,
                              cmap=overlay_cmap, vmin=vmin_local, vmax=vmax_local,
                              linewidths=contour_linewidth, alpha=contour_alpha, zorder=1)
            
            elif overlay_type == 'both':
                # Plot both heatmap and contours together
                from scipy.spatial import ConvexHull
                from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
                import matplotlib.path as mpath
                
                # Reuse the grid and mask creation from above
                extent = [self.analyzer.rmsd_flat.min(), self.analyzer.rmsd_flat.max(),
                         self.analyzer.rmsd_cross.min(), self.analyzer.rmsd_cross.max()]
                ny, nx = self.analyzer.density_map.shape
                x_grid = np.linspace(extent[0], extent[1], nx)
                y_grid = np.linspace(extent[2], extent[3], ny)
                X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                X, Y = np.meshgrid(self.analyzer.bin_centers_x, self.analyzer.bin_centers_y)
                
                cluster_mask = np.zeros_like(self.analyzer.density_map, dtype=bool)
                for i in range(len(self.analyzer.cluster_centers)):
                    mask = self.analyzer.cluster_labels == i
                    if np.sum(mask) < 3:
                        continue
                    cluster_points = np.column_stack([
                        self.analyzer.rmsd_flat[mask],
                        self.analyzer.rmsd_cross[mask]
                    ])
                    try:
                        hull = ConvexHull(cluster_points)
                        hull_path = mpath.Path(cluster_points[hull.vertices])
                        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                        inside = hull_path.contains_points(grid_points)
                        inside_grid = inside.reshape(X_grid.shape)
                        cluster_mask |= inside_grid
                    except:
                        continue
                
                masked_density = np.ma.array(self.analyzer.density_map, mask=~cluster_mask)
                
                if overlay_smooth:
                    erosion_size = int(smooth_sigma * 2)
                    dist_from_edge = distance_transform_edt(cluster_mask)
                    max_dist = erosion_size if erosion_size > 0 else 1
                    edge_alpha = np.clip(dist_from_edge / max_dist, 0, 1)
                    edge_alpha = gaussian_filter(edge_alpha, sigma=smooth_sigma * 0.5)
                    alpha_mask = edge_alpha
                    alpha_mask[~cluster_mask] = 0
                else:
                    alpha_mask = cluster_mask.astype(float)
                
                cluster_region_values = masked_density[~masked_density.mask]
                vmin_local = np.min(cluster_region_values) if len(cluster_region_values) > 0 else 0
                vmax_local = np.max(cluster_region_values) if len(cluster_region_values) > 0 else self.analyzer._target_max
                
                # LAYER 1: Plot heatmap background
                im = ax.imshow(self.analyzer.density_map, extent=extent, origin='lower',
                         aspect='auto', cmap=overlay_cmap,
                         vmin=vmin_local, vmax=vmax_local,
                         interpolation='nearest', zorder=0)
                im.set_alpha(alpha_mask * overlay_alpha * 0.7)
                
                # LAYER 2: Plot contour lines on top
                if contour_color is not None:
                    ax.contour(X, Y, masked_density, levels=contour_levels,
                              colors=contour_color, linewidths=contour_linewidth,
                              alpha=contour_alpha, zorder=1)
                else:
                    ax.contour(X, Y, masked_density, levels=contour_levels,
                              cmap=overlay_cmap, vmin=vmin_local, vmax=vmax_local,
                              linewidths=contour_linewidth, alpha=contour_alpha, zorder=1)
            else:
                raise ValueError(f"overlay_type must be 'contour', 'heatmap', or 'both', got '{overlay_type}'")
        
        # === CLUSTER SCATTER POINTS (foreground layer) ===
        # Skip scatter points if density overlay is shown (density shows the data)
        if not (overlay_density and overlay_type in ['heatmap', 'contour', 'both']):
            colors_cluster = plt.cm.tab10(
                np.linspace(0, 1, len(self.analyzer.cluster_centers))
            )
            
            for i in range(len(self.analyzer.cluster_centers)):
                mask = self.analyzer.cluster_labels == i
                count = np.sum(mask)
                
                # Normal filled markers
                ax.scatter(
                    self.analyzer.rmsd_flat[mask],
                    self.analyzer.rmsd_cross[mask],
                    s=msize, alpha=0.6, color=colors_cluster[i],
                    label=f'Cluster {i} (n={count})',
                    zorder=2
                )
        else:
            # For density overlay, just add legend entries without scatter
            colors_cluster = plt.cm.tab10(
                np.linspace(0, 1, len(self.analyzer.cluster_centers))
            )
            for i in range(len(self.analyzer.cluster_centers)):
                mask = self.analyzer.cluster_labels == i
                count = np.sum(mask)
                # Create invisible scatter for legend only
                # Use a single point at NaN instead of empty array to ensure legend entry appears
                ax.scatter([np.nan], [np.nan], s=msize, color=colors_cluster[i],
                          label=f'Cluster {i} (n={count})', zorder=2)
        
        # === CLUSTER CENTERS (top layer) ===
        if show_centers:
            ax.scatter(
                self.analyzer.cluster_centers[:, 0],
                self.analyzer.cluster_centers[:, 1],
                s=center_size, c='black', marker='X',
                edgecolors='white', linewidths=2, zorder=10,
                label='Centers'
            )
        
        # === CLUSTER LABELS ===
        if label_clusters:
            for i, center in enumerate(self.analyzer.cluster_centers):
                ax.text(
                    center[0], center[1], str(i),
                    fontsize=cluster_label_fontsize,
                    fontweight=cluster_label_fontweight,
                    color=cluster_label_color,
                    ha='center', va='center',
                    zorder=11,
                    bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', 
                             edgecolor=cluster_label_color, linewidth=2, alpha=0.9)
                )
        
        ax.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        
        if show_title:
            ax.set_title('Cluster Assignments', fontsize=title_fontsize, fontweight=title_fontweight)
        
        # Set tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontweight(tick_fontweight)
        
        if legends:
            legend = ax.legend(loc='best', fontsize=legend_fontsize, framealpha=legend_frame_alpha)
            # Set legend text to bold
            for text in legend.get_texts():
                text.set_fontweight(legend_fontweight)
        
        if show_grid:
            ax.grid(alpha=grid_alpha, ls='--', lw=0.5)
        
        plt.tight_layout()
        
        if save_fig and save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_cluster_overlay(self, figsize: Optional[Tuple[float, float]] = None,
                            msize: float = 15,
                            cmap: Optional[str] = None,
                            alpha_all: float = 0.3,
                            alpha_cluster: float = 0.8,
                            show_centers: bool = True,
                            center_size: float = 400,
                            legends: bool = True,
                            legend_frame_alpha: float = 0.9,
                            
                            # Cluster labeling options
                            label_clusters: bool = False,
                            cluster_label_fontsize: int = 18,
                            cluster_label_fontweight: str = 'bold',
                            cluster_label_color: str = 'black',
                            show_title: bool = True,
                            title_fontsize: int = 22,
                            title_fontweight: str = 'bold',
                            label_fontsize: int = 22,
                            label_fontweight: str = 'bold',
                            tick_fontsize: int = 18,
                            tick_fontweight: str = 'normal',
                            legend_fontsize: int = 18,
                            legend_fontweight: str = 'bold',
                            show_grid: bool = True,
                            grid_alpha: float = 0.2,
                            save_fig: bool = True,
                            save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Overlay cluster regions on full density scatter plot.
        
        Shows all points colored by density, with cluster members highlighted
        using colored borders. This clearly shows which regions were selected
        as clusters versus scattered noise points.
        
        Args:
            figsize: Figure size (uses default if None)
            msize: Marker size for cluster points
            cmap: Colormap for density (uses default if None)
            alpha_all: Transparency for all points (background)
            alpha_cluster: Transparency for cluster points (highlighted)
            show_centers: Whether to show cluster centers
            center_size: Size of cluster center markers
            legends: Whether to show legend (default=True)
            legend_frame_alpha: Transparency of legend frame (default=0.9)
            
            # Cluster labeling options
            label_clusters: Whether to add cluster ID labels at cluster centers (default=False)
            cluster_label_fontsize: Font size for cluster labels (default=18)
            cluster_label_fontweight: Font weight for cluster labels (default='bold')
            cluster_label_color: Color for cluster labels (default='black')
            
            show_title: Whether to show title (default=True)
            title_fontsize: Font size for title (default=22)
            title_fontweight: Font weight for title (default='bold')
            label_fontsize: Font size for axis labels (default=22)
            label_fontweight: Font weight for axis labels (default='bold')
            tick_fontsize: Font size for tick labels (default=18)
            tick_fontweight: Font weight for tick labels (default='normal')
            legend_fontsize: Font size for legend (default=18)
            legend_fontweight: Font weight for legend (default='bold')
            show_grid: Whether to show grid (default=True)
            grid_alpha: Transparency of grid lines (default=0.2)
            save_fig: Whether to save figure if save_path provided (default=True)
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        self._validate_data(require_density=True, require_clusters=True)
        
        figsize = figsize or self.default_figsize
        cmap = cmap or self.default_cmap
        
        # Get density colors for all points
        colors, valid = self.analyzer.get_point_densities()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 1. Plot ALL points with density coloring (background layer)
        scatter_all = ax.scatter(
            self.analyzer.rmsd_flat[valid],
            self.analyzer.rmsd_cross[valid],
            s=msize*0.7, c=colors[valid], marker='o',
            cmap=cmap, vmin=0, vmax=self.analyzer._target_max,
            alpha=alpha_all, edgecolors='none', zorder=1,
            label='All points (density)'
        )
        
        # 2. Highlight cluster members with colored borders
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(self.analyzer.cluster_centers)))
        
        for i in range(len(self.analyzer.cluster_centers)):
            mask = self.analyzer.cluster_labels == i
            count = np.sum(mask)
            if count > 0:
                ax.scatter(
                    self.analyzer.rmsd_flat[mask],
                    self.analyzer.rmsd_cross[mask],
                    s=msize, marker='o',
                    facecolors='none',
                    edgecolors=cluster_colors[i],
                    linewidths=1.5, alpha=alpha_cluster,
                    zorder=2,
                    label=f'Cluster {i} ({count} frames)'
                )
        
        # 3. Plot cluster centers
        if show_centers:
            ax.scatter(
                self.analyzer.cluster_centers[:, 0],
                self.analyzer.cluster_centers[:, 1],
                s=center_size, c='red', marker='X',
                edgecolors='black', linewidths=2, zorder=10,
                label='Cluster centers'
            )
        
        # 4. Add cluster labels
        if label_clusters:
            for i, center in enumerate(self.analyzer.cluster_centers):
                ax.text(
                    center[0], center[1], str(i),
                    fontsize=cluster_label_fontsize,
                    fontweight=cluster_label_fontweight,
                    color=cluster_label_color,
                    ha='center', va='center',
                    zorder=11,
                    bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', 
                             edgecolor=cluster_label_color, linewidth=2, alpha=0.9)
                )
        
        # Add colorbar for density
        cbar = plt.colorbar(scatter_all, ax=ax)
        cbar.set_label('Density (counts per bin)', fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        
        ax.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        
        if show_title:
            ax.set_title('Cluster Regions Overlaid on Density Scatter', fontsize=title_fontsize, fontweight=title_fontweight)
        
        # Set tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        if legends:
            legend = ax.legend(loc='best', fontsize=legend_fontsize, framealpha=legend_frame_alpha)
            # Set legend text to bold
            for text in legend.get_texts():
                text.set_fontweight(legend_fontweight)
        
        if show_grid:
            ax.grid(alpha=grid_alpha, ls='--', lw=0.5)
        
        plt.tight_layout()
        
        if save_fig and save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_clusters_with_density(self, figsize: Optional[Tuple[float, float]] = None,
                                   msize: float = 15,
                                   cmap: Optional[str] = None,
                                   cluster_colors: Optional[Dict[str, str]] = None,
                                   show_boundaries: bool = True,
                                   boundary_linewidth: float = 3.0,
                                   boundary_alpha: float = 0.8,
                                   show_contours: bool = False,
                                   contour_levels: int = 5,
                                   contour_alpha: float = 0.3,
                                   contour_linewidth: float = 1.5,
                                   show_centers: bool = True,
                                   center_size: float = 400,
                                   legends: bool = True,
                                   legend_frame_alpha: float = 0.2,
                                   show_legend_title: bool = True,
                                   show_all_points_label: bool = True,
                                   
                                   # Cluster labeling options
                                   label_clusters: bool = False,
                                   cluster_label_fontsize: int = 18,
                                   cluster_label_fontweight: str = 'bold',
                                   cluster_label_color: str = 'black',
                                   show_title: bool = True,
                                   title_fontsize: int = 22,
                                   title_fontweight: str = 'bold',
                                   axis_label_fontsize: int = 22,
                                   axis_label_fontweight: str = 'bold',
                                   tick_fontsize: int = 18,
                                   tick_fontweight: str = 'normal',
                                   legend_fontsize: int = 18,
                                   legend_fontweight: str = 'bold',
                                   show_grid: bool = True,
                                   grid_alpha: float = 0.2,
                                   save_fig: bool = True,
                                   save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot clusters with density coloring and colored boundaries.
        
        Enhanced cluster visualization showing:
        - Points colored by DENSITY (using cmap) - shows data concentration
        - Cluster boundaries/contours colored by CLUSTER (using cluster_colors) - shows cluster identity
        
        Args:
            figsize: Figure size (uses default if None)
            msize: Marker size for data points
            cmap: Colormap for density-based point coloring (e.g., 'viridis', 'plasma')
            cluster_colors: Dict mapping cluster IDs to colors for boundaries/contours,
                          e.g., {'cluster_0': 'red', 'cluster_1': 'blue'}
                          If None, uses default tab10 colormap
            show_boundaries: Draw cluster boundary outlines (default=True)
            boundary_linewidth: Width of boundary lines (default=3.0)
            boundary_alpha: Transparency of boundary lines (default=0.8)
            show_contours: Draw density contour lines within clusters (default=False)
            contour_levels: Number of contour levels (default=5)
            contour_alpha: Transparency of contour lines (default=0.3)
            contour_linewidth: Width of contour lines (default=1.5)
            show_centers: Show cluster centers as markers (default=True)
            center_size: Size of cluster center markers (default=400)
            legends: Whether to show legend (default=True)
            legend_frame_alpha: Transparency of legend frame (default=0.2)
            show_legend_title: Whether to show legend title above entries (default=False)
            show_all_points_label: Whether to show 'All points (density)' in legend (default=True)
            
            # Cluster labeling options
            label_clusters: Whether to add cluster ID labels at cluster centers (default=False)
            cluster_label_fontsize: Font size for cluster labels (default=18)
            cluster_label_fontweight: Font weight for cluster labels (default='bold')
            cluster_label_color: Color for cluster labels (default='black')
            
            show_title: Whether to show title (default=True)
            title_fontsize: Font size for title (default=22)
            title_fontweight: Font weight for title (default='bold')
            axis_label_fontsize: Font size for axis labels (default=22)
            axis_label_fontweight: Font weight for axis labels (default='bold')
            tick_fontsize: Font size for tick labels (default=18)
            tick_fontweight: Font weight for tick labels (default='normal')
            legend_fontsize: Font size for legend (default=18)
            legend_fontweight: Font weight for legend (default='bold')
            show_grid: Whether to show grid (default=True)
            grid_alpha: Transparency of grid lines (default=0.2)
            save_fig: Whether to save figure if save_path provided (default=True)
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
            
        Example:
            >>> fig = plotter.plot_clusters_with_density(
            ...     cmap='viridis',  # Density coloring for points
            ...     cluster_colors={'cluster_0': 'red', 'cluster_1': 'blue'},  # Boundary colors
            ...     show_boundaries=True,
            ...     boundary_linewidth=3.0,
            ...     show_contours=True,
            ...     contour_levels=5,
            ...     contour_linewidth=2.0,
            ...     save_path='clusters_density.png'
            ... )
        """
        self._validate_data(require_density=True, require_clusters=True)
        
        figsize = figsize or self.default_figsize
        cmap = cmap or self.default_cmap
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define default colors for each cluster if not provided
        n_clusters = len(self.analyzer.cluster_centers)
        if cluster_colors is None:
            default_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            cluster_colors = {f'cluster_{i}': default_colors[i] for i in range(n_clusters)}
        
        # Use boundary colors from cluster_colors
        boundary_colors = [cluster_colors.get(f'cluster_{i}', plt.cm.tab10(i/n_clusters)) 
                          for i in range(n_clusters)]
        
        # Get density colors for ALL points
        density_colors, valid = self.analyzer.get_point_densities()
        
        # 1. FIRST: Plot ALL points with density coloring (background layer)
        all_points_label = 'All points (density)' if show_all_points_label else None
        scatter_all = ax.scatter(
            self.analyzer.rmsd_flat[valid],
            self.analyzer.rmsd_cross[valid],
            s=msize, c=density_colors[valid], marker='o',
            cmap=cmap, vmin=0, vmax=self.analyzer._target_max,
            alpha=0.6, edgecolors='none', zorder=1,
            label=all_points_label
        )
        
        # 2. THEN: Overlay cluster boundaries and contours
        for i in range(n_clusters):
            mask = self.analyzer.cluster_labels == i
            count = np.sum(mask)
            
            if count < 1:
                continue
            
            cluster_x = self.analyzer.rmsd_flat[mask]
            cluster_y = self.analyzer.rmsd_cross[mask]
            cluster_points = np.column_stack([cluster_x, cluster_y])
            
            if count < 3:  # Need at least 3 points for boundary
                continue
            
            # Draw cluster boundary using convex hull
            if show_boundaries:
                try:
                    hull = ConvexHull(cluster_points)
                    # Close the hull by appending first point
                    hull_points = cluster_points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    
                    ax.plot(
                        hull_points[:, 0], hull_points[:, 1],
                        color=boundary_colors[i], linewidth=boundary_linewidth,
                        alpha=boundary_alpha, zorder=3,
                        label=f'Cluster {i} boundary (n={count})'
                    )
                except Exception as e:
                    print(f"Warning: Could not compute boundary for cluster {i}: {e}")
            
            # Draw density contours within cluster
            if show_contours:
                try:
                    # Create grid for this cluster region
                    x_min, x_max = cluster_x.min(), cluster_x.max()
                    y_min, y_max = cluster_y.min(), cluster_y.max()
                    
                    # Add padding
                    x_padding = (x_max - x_min) * 0.1
                    y_padding = (y_max - y_min) * 0.1
                    
                    xi = np.linspace(x_min - x_padding, x_max + x_padding, 50)
                    yi = np.linspace(y_min - y_padding, y_max + y_padding, 50)
                    Xi, Yi = np.meshgrid(xi, yi)
                    
                    # Get density values for cluster points
                    # Map cluster points to grid indices
                    from scipy.stats import gaussian_kde
                    
                    if len(cluster_x) > 3:
                        kde = gaussian_kde(np.vstack([cluster_x, cluster_y]))
                        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                        
                        # Draw contours
                        contours = ax.contour(
                            Xi, Yi, Zi, levels=contour_levels,
                            colors=boundary_colors[i], alpha=contour_alpha,
                            linewidths=contour_linewidth, zorder=2
                        )
                except Exception as e:
                    print(f"Warning: Could not compute contours for cluster {i}: {e}")
        
        # Add colorbar for density
        cbar = plt.colorbar(scatter_all, ax=ax)
        cbar.set_label('Density (counts per bin)', fontsize=axis_label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        
        # Plot cluster centers
        if show_centers:
            ax.scatter(
                self.analyzer.cluster_centers[:, 0],
                self.analyzer.cluster_centers[:, 1],
                s=center_size, c='red', marker='X',
                edgecolors='black', linewidths=2, zorder=10,
                label='Cluster centers'
            )
        
        # Print frame numbers for cluster centers and edges (for VMD visualization)
        print("\n" + "="*60)
        print("CLUSTER CENTER FRAME NUMBERS (for VMD visualization)")
        print("="*60)
        for i, center in enumerate(self.analyzer.cluster_centers):
            # Calculate distance from all frames to this cluster center
            distances = np.sqrt(
                (self.analyzer.rmsd_flat - center[0])**2 + 
                (self.analyzer.rmsd_cross - center[1])**2
            )
            # Find the frame with minimum distance (closest to center)
            center_frame = np.argmin(distances)
            print(f"Cluster {i} center: Frame {center_frame} "
                  f"(RMSD_flat={self.analyzer.rmsd_flat[center_frame]:.4f} nm, "
                  f"RMSD_cross={self.analyzer.rmsd_cross[center_frame]:.4f} nm)")
        print("="*60 + "\n")
        
        print("="*60)
        print("CLUSTER MIDDLE FRAME NUMBERS (for VMD visualization)")
        print("="*60)
        for i, center in enumerate(self.analyzer.cluster_centers):
            # Get frames belonging to this cluster
            mask = self.analyzer.cluster_labels == i
            if np.sum(mask) == 0:
                continue
            
            # Calculate distances for frames in this cluster only
            cluster_distances = np.sqrt(
                (self.analyzer.rmsd_flat[mask] - center[0])**2 + 
                (self.analyzer.rmsd_cross[mask] - center[1])**2
            )
            
            # Find the frame closest to the median distance (middle configuration)
            median_distance = np.median(cluster_distances)
            middle_idx_within_cluster = np.argmin(np.abs(cluster_distances - median_distance))
            # Map back to global frame index
            cluster_frame_indices = np.where(mask)[0]
            middle_frame = cluster_frame_indices[middle_idx_within_cluster]
            
            print(f"Cluster {i} middle: Frame {middle_frame} "
                  f"(RMSD_flat={self.analyzer.rmsd_flat[middle_frame]:.4f} nm, "
                  f"RMSD_cross={self.analyzer.rmsd_cross[middle_frame]:.4f} nm)")
        print("="*60 + "\n")
        
        print("="*60)
        print("CLUSTER EDGE FRAME NUMBERS (for VMD visualization)")
        print("="*60)
        for i, center in enumerate(self.analyzer.cluster_centers):
            # Get frames belonging to this cluster
            mask = self.analyzer.cluster_labels == i
            if np.sum(mask) == 0:
                continue
            
            # Calculate distances for frames in this cluster only
            cluster_distances = np.sqrt(
                (self.analyzer.rmsd_flat[mask] - center[0])**2 + 
                (self.analyzer.rmsd_cross[mask] - center[1])**2
            )
            
            # Find the frame index within the cluster that is furthest from center
            edge_idx_within_cluster = np.argmax(cluster_distances)
            # Map back to global frame index
            cluster_frame_indices = np.where(mask)[0]
            edge_frame = cluster_frame_indices[edge_idx_within_cluster]
            
            print(f"Cluster {i} edge: Frame {edge_frame} "
                  f"(RMSD_flat={self.analyzer.rmsd_flat[edge_frame]:.4f} nm, "
                  f"RMSD_cross={self.analyzer.rmsd_cross[edge_frame]:.4f} nm)")
        print("="*60 + "\n")
        
        # Add cluster labels
        if label_clusters:
            for i, center in enumerate(self.analyzer.cluster_centers):
                ax.text(
                    center[0], center[1], str(i),
                    fontsize=cluster_label_fontsize,
                    fontweight=cluster_label_fontweight,
                    color=cluster_label_color,
                    ha='center', va='center',
                    zorder=11,
                    bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', 
                             edgecolor=cluster_label_color, linewidth=2, alpha=0.9)
                )
        
        # Labels and formatting
        ax.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=axis_label_fontsize, fontweight=axis_label_fontweight)
        ax.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=axis_label_fontsize, fontweight=axis_label_fontweight)
        
        if show_title:
            ax.set_title('Clusters with Density Distribution', fontsize=title_fontsize, fontweight=title_fontweight)
        
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontweight(tick_fontweight)
        
        if legends:
            legend_title = 'Clusters & Density' if show_legend_title else None
            legend = ax.legend(loc='best', fontsize=legend_fontsize, framealpha=legend_frame_alpha, title=legend_title)
            for text in legend.get_texts():
                text.set_fontweight(legend_fontweight)
            if legend.get_title():
                legend.get_title().set_fontweight(legend_fontweight)
        
        if show_grid:
            ax.grid(alpha=grid_alpha, ls='--', lw=0.5)
        
        plt.tight_layout()
        
        if save_fig and save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_combined_view(self, figsize: Optional[Tuple[float, float]] = None,
                          cmap: Optional[str] = None,
                          msize: float = 8,
                          
                          # Publication formatting
                          show_title: bool = True,
                          title_fontsize: int = 14,
                          title_fontweight: str = 'bold',
                          subplot_title_fontsize: int = 11,
                          subplot_title_fontweight: str = 'bold',
                          label_fontsize: int = 10,
                          label_fontweight: str = 'bold',
                          tick_fontsize: int = 9,
                          tick_fontweight: str = 'normal',
                          
                          # Colorbar control
                          colorbar_label_fontsize: int = 10,
                          colorbar_label_fontweight: str = 'normal',
                          colorbar_tick_fontsize: int = 9,
                          colorbar_pad: float = 0.05,
                          colorbar_width: float = 0.03,
                          
                          # Figure quality
                          dpi: Optional[int] = None,
                          
                          # Legend control
                          show_legend: bool = True,
                          legend_loc: str = 'best',
                          legend_ncol: int = 3,
                          legend_fontsize: int = 8,
                          legend_fontweight: str = 'normal',
                          legend_frame_alpha: float = 0.7,
                          
                          # Grid control
                          show_grid: bool = True,
                          grid_alpha: float = 0.3,
                          grid_linestyle: str = '--',
                          grid_linewidth: float = 0.5,
                          
                          # Individual plots control
                          show_individual_figures: bool = False,
                          individual_figsize: Tuple[float, float] = (8, 6),
                          save_individual_figures: bool = False,
                          individual_save_dir: Optional[str] = None,
                          
                          # Combined figure control
                          save_combined_figure: bool = False,
                          show_combined_figure: bool = True,
                          save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create publication-ready combined view with multiple subplots.
        
        Creates a 2x2 grid showing:
        - Top left: Density scatter
        - Top right: Heatmap
        - Bottom left: Contour map
        - Bottom right: Cluster assignments (if available)
        
        Args:
            figsize: Figure size for combined view (uses default multiplied by 1.8 if None)
            cmap: Colormap name (uses default if None)
            msize: Marker size for scatter points
            
            # Publication formatting
            show_title: Show main figure title (default=True)
            title_fontsize: Font size for main title (default=14)
            title_fontweight: Font weight for main title (default='bold')
            subplot_title_fontsize: Font size for subplot titles (default=11)
            subplot_title_fontweight: Font weight for subplot titles (default='bold')
            label_fontsize: Font size for axis labels (default=10)
            label_fontweight: Font weight for axis labels (default='bold')
            tick_fontsize: Font size for tick labels (default=9)
            tick_fontweight: Font weight for tick labels (default='normal')
            
            # Colorbar control
            colorbar_label_fontsize: Font size for colorbar labels (default=10)
            colorbar_label_fontweight: Font weight for colorbar labels (default='normal')
            colorbar_tick_fontsize: Font size for colorbar tick labels (default=9)
            colorbar_pad: Padding between plot and colorbar (default=0.05)
            colorbar_width: Aspect ratio for colorbar width (default=0.03, controls width via aspect=1/colorbar_width)
            
            # Figure quality
            dpi: DPI for saved figures (default=None, uses self.default_dpi)
            
            # Legend control
            show_legend: Show legend in cluster subplot (default=True)
            legend_loc: Legend location (default='best')
            legend_ncol: Number of legend columns (default=3)
            legend_fontsize: Font size for legend (default=8)
            legend_fontweight: Font weight for legend (default='normal')
            legend_frame_alpha: Transparency of legend frame (default=0.7)
            
            # Grid control
            show_grid: Show grid lines (default=True)
            grid_alpha: Transparency of grid lines (default=0.3)
            grid_linestyle: Style of grid lines (default='--')
            grid_linewidth: Width of grid lines (default=0.5)
            
            # Individual plots control
            show_individual_figures: Show each subplot as separate figure (default=False)
            individual_figsize: Figure size for individual plots (default=(8, 6))
            save_individual_figures: Save individual plots to files (default=False)
            individual_save_dir: Directory to save individual plots (default=None, uses current dir)
            
            # Combined figure control
            save_combined_figure: Save the combined figure (default=False)
            show_combined_figure: Show the combined figure (default=True)
            save_path: Path to save combined figure
            
        Returns:
            Matplotlib figure object (combined view)
            
        Example:
            >>> fig = plotter.plot_combined_view(
            ...     cmap='jet',
            ...     show_title=False,
            ...     show_legend=True,
            ...     legend_ncol=2,
            ...     show_grid=True,
            ...     grid_alpha=0.2,
            ...     save_individual_figures=True,
            ...     individual_save_dir='analysis_plots',
            ...     save_combined_figure=True,
            ...     save_path='combined_view.png'
            ... )
        """
        self._validate_data(require_density=True)
        
        # Adjusted figsize for 3 horizontal plots
        figsize = figsize or (self.default_figsize[0] * 3, self.default_figsize[1] * 1.2)
        cmap = cmap or self.default_cmap
        dpi = dpi or self.default_dpi
        
        # Create directory for individual plots if needed
        if save_individual_figures and individual_save_dir:
            from pathlib import Path
            Path(individual_save_dir).mkdir(parents=True, exist_ok=True)
            save_dir = individual_save_dir
        else:
            save_dir = "."
        
        # Helper function to save individual plots
        def save_individual(fig_obj, name):
            if save_individual_figures:
                path = f"{save_dir}/{name}.png"
                fig_obj.savefig(path, dpi=dpi, bbox_inches='tight')
                print(f"Saved individual plot: {path}")
        
        # Get data once
        colors, valid = self.analyzer.get_point_densities()
        extent = [self.analyzer.rmsd_flat.min(), self.analyzer.rmsd_flat.max(),
                 self.analyzer.rmsd_cross.min(), self.analyzer.rmsd_cross.max()]
        X, Y = np.meshgrid(self.analyzer.bin_centers_x, self.analyzer.bin_centers_y)
        
        # === PLOT 1: Density Scatter ===
        if show_individual_figures or save_individual_figures:
            fig1 = plt.figure(figsize=individual_figsize)
            ax1_ind = fig1.add_subplot(111)
            scatter1_ind = ax1_ind.scatter(
                self.analyzer.rmsd_flat[valid],
                self.analyzer.rmsd_cross[valid],
                s=msize, c=colors[valid], marker='o', cmap=cmap,
                vmin=0, vmax=self.analyzer._target_max,
                alpha=0.7, edgecolors='none'
            )
            ax1_ind.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
            ax1_ind.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title:
                ax1_ind.set_title('Density Scatter', fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
            ax1_ind.tick_params(axis='both', labelsize=tick_fontsize)
            for label in (ax1_ind.get_xticklabels() + ax1_ind.get_yticklabels()):
                label.set_fontweight(tick_fontweight)
            if show_grid:
                ax1_ind.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
            cbar1 = plt.colorbar(scatter1_ind, ax=ax1_ind, pad=colorbar_pad, 
                                aspect=int(1/colorbar_width), shrink=1.0)
            cbar1.set_label('Density', fontsize=colorbar_label_fontsize, fontweight=colorbar_label_fontweight)
            cbar1.ax.tick_params(labelsize=colorbar_tick_fontsize)
            plt.tight_layout()
            save_individual(fig1, 'density_scatter')
            if show_individual_figures:
                plt.show()
            else:
                plt.close(fig1)
        
        # === PLOT 2: Heatmap ===
        if show_individual_figures or save_individual_figures:
            fig2 = plt.figure(figsize=individual_figsize)
            ax2_ind = fig2.add_subplot(111)
            im_ind = ax2_ind.imshow(self.analyzer.density_map, extent=extent, origin='lower',
                           aspect='auto', cmap=cmap, vmin=0,
                           vmax=self.analyzer._target_max, interpolation='bilinear')
            ax2_ind.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
            ax2_ind.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title:
                ax2_ind.set_title('Heatmap', fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
            ax2_ind.tick_params(axis='both', labelsize=tick_fontsize)
            for label in (ax2_ind.get_xticklabels() + ax2_ind.get_yticklabels()):
                label.set_fontweight(tick_fontweight)
            if show_grid:
                ax2_ind.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
            cbar2 = plt.colorbar(im_ind, ax=ax2_ind, pad=colorbar_pad,
                                aspect=int(1/colorbar_width), shrink=1.0)
            cbar2.set_label('Density', fontsize=colorbar_label_fontsize, fontweight=colorbar_label_fontweight)
            cbar2.ax.tick_params(labelsize=colorbar_tick_fontsize)
            plt.tight_layout()
            save_individual(fig2, 'heatmap')
            if show_individual_figures:
                plt.show()
            else:
                plt.close(fig2)
        
        # === PLOT 3: Contour Map ===
        if show_individual_figures or save_individual_figures:
            fig3 = plt.figure(figsize=individual_figsize)
            ax3_ind = fig3.add_subplot(111)
            contour_ind = ax3_ind.contourf(X, Y, self.analyzer.density_map, levels=10, cmap=cmap)
            ax3_ind.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
            ax3_ind.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title:
                ax3_ind.set_title('Contour Map', fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
            ax3_ind.tick_params(axis='both', labelsize=tick_fontsize)
            for label in (ax3_ind.get_xticklabels() + ax3_ind.get_yticklabels()):
                label.set_fontweight(tick_fontweight)
            if show_grid:
                ax3_ind.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
            cbar3 = plt.colorbar(contour_ind, ax=ax3_ind, pad=colorbar_pad,
                                aspect=int(1/colorbar_width), shrink=1.0)
            cbar3.set_label('Density', fontsize=colorbar_label_fontsize, fontweight=colorbar_label_fontweight)
            cbar3.ax.tick_params(labelsize=colorbar_tick_fontsize)
            plt.tight_layout()
            save_individual(fig3, 'contour_map')
            if show_individual_figures:
                plt.show()
            else:
                plt.close(fig3)
        
        # === PLOT 4: Clusters or Raw Scatter ===
        if show_individual_figures or save_individual_figures:
            fig4 = plt.figure(figsize=individual_figsize)
            ax4_ind = fig4.add_subplot(111)
            if self.analyzer.cluster_labels is not None and self.analyzer.cluster_centers is not None:
                colors_cluster = plt.cm.tab10(np.linspace(0, 1, len(self.analyzer.cluster_centers)))
                for i in range(len(self.analyzer.cluster_centers)):
                    mask = self.analyzer.cluster_labels == i
                    count = np.sum(mask)
                    ax4_ind.scatter(self.analyzer.rmsd_flat[mask], self.analyzer.rmsd_cross[mask],
                              s=msize, alpha=0.6, color=colors_cluster[i], label=f'Cluster {i} (n={count})')
                ax4_ind.scatter(self.analyzer.cluster_centers[:, 0], self.analyzer.cluster_centers[:, 1],
                           s=200, c='black', marker='X', edgecolors='white', linewidths=2, zorder=10,
                           label='Centers')
                ax4_ind.set_title('Clusters', fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
                if show_legend:
                    legend = ax4_ind.legend(loc=legend_loc, fontsize=legend_fontsize, 
                                           ncol=legend_ncol, framealpha=legend_frame_alpha)
                    for text in legend.get_texts():
                        text.set_fontweight(legend_fontweight)
            else:
                ax4_ind.scatter(self.analyzer.rmsd_flat, self.analyzer.rmsd_cross,
                           s=msize, alpha=0.3, c='blue')
                ax4_ind.set_title('Raw Scatter', fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
            ax4_ind.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
            ax4_ind.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
            ax4_ind.tick_params(axis='both', labelsize=tick_fontsize)
            for label in (ax4_ind.get_xticklabels() + ax4_ind.get_yticklabels()):
                label.set_fontweight(tick_fontweight)
            if show_grid:
                ax4_ind.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
            plt.tight_layout()
            save_individual(fig4, 'clusters')
            if show_individual_figures:
                plt.show()
            else:
                plt.close(fig4)
        
        # === COMBINED FIGURE ===
        if not show_combined_figure and not save_combined_figure:
            return None
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 3, hspace=0.3, wspace=0.3)
        
        # Density scatter
        ax1 = fig.add_subplot(gs[0, 0])
        scatter1 = ax1.scatter(
            self.analyzer.rmsd_flat[valid],
            self.analyzer.rmsd_cross[valid],
            s=msize, c=colors[valid], marker='o', cmap=cmap,
            vmin=0, vmax=self.analyzer._target_max,
            alpha=0.7, edgecolors='none'
        )
        ax1.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax1.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax1.set_title('Density Scatter', fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontweight(tick_fontweight)
        if show_grid:
            ax1.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
        cbar1 = plt.colorbar(scatter1, ax=ax1, pad=colorbar_pad,
                            aspect=int(1/colorbar_width), shrink=1.0)
        cbar1.set_label('Density', fontsize=colorbar_label_fontsize, fontweight=colorbar_label_fontweight)
        cbar1.ax.tick_params(labelsize=colorbar_tick_fontsize)
        
        # Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(self.analyzer.density_map, extent=extent, origin='lower',
                       aspect='auto', cmap=cmap, vmin=0,
                       vmax=self.analyzer._target_max, interpolation='bilinear')
        ax2.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax2.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax2.set_title('Heatmap', fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            label.set_fontweight(tick_fontweight)
        if show_grid:
            ax2.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
        cbar2 = plt.colorbar(im, ax=ax2, pad=colorbar_pad,
                            aspect=int(1/colorbar_width), shrink=1.0)
        cbar2.set_label('Density', fontsize=colorbar_label_fontsize, fontweight=colorbar_label_fontweight)
        cbar2.ax.tick_params(labelsize=colorbar_tick_fontsize)
        
        # Contour map
        ax3 = fig.add_subplot(gs[0, 2])
        contour = ax3.contourf(X, Y, self.analyzer.density_map, levels=10, cmap=cmap)
        ax3.set_xlabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax3.set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax3.set_title('Contour Map', fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
        ax3.tick_params(axis='both', labelsize=tick_fontsize)
        for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
            label.set_fontweight(tick_fontweight)
        if show_grid:
            ax3.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
        cbar3 = plt.colorbar(contour, ax=ax3, pad=colorbar_pad,
                            aspect=int(1/colorbar_width), shrink=1.0)
        cbar3.set_label('Density', fontsize=colorbar_label_fontsize, fontweight=colorbar_label_fontweight)
        cbar3.ax.tick_params(labelsize=colorbar_tick_fontsize)
        
        if show_title:
            fig.suptitle('RMSD Density Analysis - Combined View', 
                        fontsize=title_fontsize, fontweight=title_fontweight, y=0.98)
        
        if save_combined_figure and save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved combined figure: {save_path}")
        
        if show_combined_figure:
            return fig
        else:
            plt.close(fig)
            return None
    
    def plot_time_evolution(self, figsize: Optional[Tuple[float, float]] = None,
                           single_plot: bool = False,
                           
                           # Line/marker styling
                           msize: float = 3,
                           lw: float = 0.8,
                           alpha: float = 0.7,
                           markers: List[str] = ['o', '^'],
                           filled_markers: List[bool] = [True, True],
                           cluster_colors: Optional[List[str]] = None,
                           
                           # Publication formatting
                           show_title: bool = True,
                           title_fontsize: int = 12,
                           title_fontweight: str = 'bold',
                           label_fontsize: int = 10,
                           label_fontweight: str = 'bold',
                           tick_fontsize: int = 9,
                           tick_fontweight: str = 'normal',
                           
                           # Legend control
                           show_legend: bool = True,
                           legend_fontsize: int = 9,
                           legend_fontweight: str = 'normal',
                           legend_frame_alpha: float = 0.7,
                           legend_loc: str = 'best',
                           legend_ncol: int = 2,
                           
                           # Grid control
                           show_grid: bool = True,
                           grid_alpha: float = 0.3,
                           grid_linestyle: str = '--',
                           grid_linewidth: float = 0.5,
                           
                           # Figure quality
                           dpi: Optional[int] = None,
                           
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot time evolution of RMSD values and cluster assignments.
        
        Args:
            figsize: Figure size (width, height*1.5 for vertical layout if single_plot=False)
            single_plot: If True, plot both RMSD values on same axes (default=False)
            
            # Line/marker styling
            msize: Marker size for scatter plots (default=3)
            lw: Line width for line plots (default=0.8)
            alpha: Transparency for markers/lines (default=0.7)
            markers: List of marker types for [RMSD flat, RMSD cross] 
                    (default=['o', '^'], e.g., ['s', 'D'] for square and diamond)
            filled_markers: List of bools for whether each marker is filled [flat, cross]
                           (default=[True, True], use [False, False] for hollow markers)
            cluster_colors: List of colors for clusters (default=None uses 'tab10' colormap)
                           e.g., ['red', 'blue', 'green'] or ['#FF5733', '#33FF57', '#3357FF']
            
            # Publication formatting
            show_title: Show plot title (default=True)
            title_fontsize: Font size for title (default=12)
            title_fontweight: Font weight for title (default='bold')
            label_fontsize: Font size for axis labels (default=10)
            label_fontweight: Font weight for axis labels (default='bold')
            tick_fontsize: Font size for tick labels (default=9)
            tick_fontweight: Font weight for tick labels (default='normal')
            
            # Legend control
            show_legend: Show legend (default=True)
            legend_fontsize: Font size for legend (default=9)
            legend_fontweight: Font weight for legend (default='normal')
            legend_frame_alpha: Transparency of legend frame (default=0.7)
            legend_loc: Legend location (default='best')
            legend_ncol: Number of legend columns (default=2)
            
            # Grid control
            show_grid: Show grid lines (default=True)
            grid_alpha: Transparency of grid lines (default=0.3)
            grid_linestyle: Style of grid lines (default='--')
            grid_linewidth: Width of grid lines (default=0.5)
            
            # Figure quality
            dpi: DPI for saved figure (default=None, uses self.default_dpi)
            
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
            
        Example:
            >>> # Single plot mode with custom formatting
            >>> fig = plotter.plot_time_evolution(
            ...     single_plot=True,
            ...     msize=5,
            ...     markers=['s', 'D'],  # Square and diamond
            ...     filled_markers=[False, True],  # Hollow square, filled diamond
            ...     title_fontsize=14,
            ...     label_fontsize=12,
            ...     legend_fontsize=10,
            ...     save_path='time_evolution.png'
            ... )
        """
        self._validate_data()
        
        dpi = dpi or self.default_dpi
        
        has_clusters = (self.analyzer.cluster_labels is not None and 
                       self.analyzer.cluster_centers is not None)
        
        if single_plot:
            # Single plot mode: both RMSDs on same xy plane
            figsize = figsize or self.default_figsize
            fig, ax = plt.subplots(figsize=figsize)
            
            if has_clusters:
                # Color points by cluster assignment
                if cluster_colors is not None:
                    # Use custom colors
                    colors = [cluster_colors[label] for label in self.analyzer.cluster_labels]
                else:
                    # Use tab10 colormap
                    colors = plt.cm.tab10(self.analyzer.cluster_labels / len(self.analyzer.cluster_centers))
                
                # Plot RMSD flat
                if filled_markers[0]:
                    ax.scatter(self.analyzer.time, self.analyzer.rmsd_flat,
                              s=msize, alpha=alpha, c=colors, label=f'RMSD {self.analyzer.label_x}', 
                              marker=markers[0])
                else:
                    ax.scatter(self.analyzer.time, self.analyzer.rmsd_flat,
                              s=msize, alpha=alpha, facecolors='none', 
                              edgecolors=colors, label=f'RMSD {self.analyzer.label_x}', 
                              marker=markers[0], linewidths=1.0)
                
                # Plot RMSD cross
                if filled_markers[1]:
                    ax.scatter(self.analyzer.time, self.analyzer.rmsd_cross,
                              s=msize, alpha=alpha, c=colors, label=f'RMSD {self.analyzer.label_y}', 
                              marker=markers[1])
                else:
                    ax.scatter(self.analyzer.time, self.analyzer.rmsd_cross,
                              s=msize, alpha=alpha, facecolors='none',
                              edgecolors=colors, label=f'RMSD {self.analyzer.label_y}', 
                              marker=markers[1], linewidths=1.0)
                
                if show_legend:
                    # Add cluster legend
                    from matplotlib.patches import Patch
                    if cluster_colors is not None:
                        cluster_handles = [Patch(facecolor=cluster_colors[i], 
                                                label=f'Cluster {i}')
                                         for i in range(len(self.analyzer.cluster_centers))]
                    else:
                        cluster_handles = [Patch(facecolor=plt.cm.tab10(i/len(self.analyzer.cluster_centers)), 
                                                label=f'Cluster {i}')
                                         for i in range(len(self.analyzer.cluster_centers))]
                    
                    # Add RMSD type legend
                    from matplotlib.lines import Line2D
                    # Adjust marker style based on filled_markers
                    if filled_markers[0]:
                        flat_handle = Line2D([0], [0], marker=markers[0], color='w', 
                                            markerfacecolor='gray', markersize=6, 
                                            label=f'RMSD {self.analyzer.label_x}', linestyle='None')
                    else:
                        flat_handle = Line2D([0], [0], marker=markers[0], color='w',
                                            markerfacecolor='none', markeredgecolor='gray',
                                            markersize=6, markeredgewidth=1.5,
                                            label=f'RMSD {self.analyzer.label_x}', linestyle='None')
                    
                    if filled_markers[1]:
                        cross_handle = Line2D([0], [0], marker=markers[1], color='w',
                                             markerfacecolor='gray', markersize=6,
                                             label=f'RMSD {self.analyzer.label_y}', linestyle='None')
                    else:
                        cross_handle = Line2D([0], [0], marker=markers[1], color='w',
                                             markerfacecolor='none', markeredgecolor='gray',
                                             markersize=6, markeredgewidth=1.5,
                                             label=f'RMSD {self.analyzer.label_y}', linestyle='None')
                    
                    rmsd_handles = [flat_handle, cross_handle]
                    
                    # Combine all handles into one legend (adjacent layout)
                    all_handles = rmsd_handles + cluster_handles
                    legend = ax.legend(handles=all_handles, loc=legend_loc, 
                                      fontsize=legend_fontsize, framealpha=legend_frame_alpha,
                                      ncol=legend_ncol)
                    # Make legend text bold if requested
                    for text in legend.get_texts():
                        text.set_fontweight(legend_fontweight)
            else:
                # No clusters: simple colored lines
                ax.plot(self.analyzer.time, self.analyzer.rmsd_flat,
                       lw=lw, alpha=alpha, c='blue', label=f'RMSD {self.analyzer.label_x}')
                ax.plot(self.analyzer.time, self.analyzer.rmsd_cross,
                       lw=lw, alpha=alpha, c='red', label=f'RMSD {self.analyzer.label_y}')
                if show_legend:
                    legend = ax.legend(loc=legend_loc, fontsize=legend_fontsize, 
                                      framealpha=legend_frame_alpha)
                    for text in legend.get_texts():
                        text.set_fontweight(legend_fontweight)
            
            ax.set_xlabel('Time (ns)', fontsize=label_fontsize, fontweight=label_fontweight)
            ax.set_ylabel('RMSD (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title:
                ax.set_title('RMSD Time Evolution', fontsize=title_fontsize, fontweight=title_fontweight)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontweight(tick_fontweight)
            if show_grid:
                ax.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
            
        else:
            # Multi-plot mode: separate subplots (original behavior)
            n_plots = 3 if has_clusters else 2
            figsize = figsize or (self.default_figsize[0], self.default_figsize[1] * 1.2)
            
            fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
            
            # RMSD flat vs time
            axes[0].plot(self.analyzer.time, self.analyzer.rmsd_flat, 
                        lw=lw, alpha=alpha, c='blue')
            axes[0].set_ylabel(f'RMSD {self.analyzer.label_x} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title:
                axes[0].set_title('Time Evolution', fontsize=title_fontsize, fontweight=title_fontweight)
            axes[0].tick_params(axis='both', labelsize=tick_fontsize)
            for label in (axes[0].get_xticklabels() + axes[0].get_yticklabels()):
                label.set_fontweight(tick_fontweight)
            if show_grid:
                axes[0].grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
            
            # RMSD cross vs time
            axes[1].plot(self.analyzer.time, self.analyzer.rmsd_cross,
                        lw=lw, alpha=alpha, c='red')
            axes[1].set_ylabel(f'RMSD {self.analyzer.label_y} (nm)', fontsize=label_fontsize, fontweight=label_fontweight)
            axes[1].tick_params(axis='both', labelsize=tick_fontsize)
            for label in (axes[1].get_xticklabels() + axes[1].get_yticklabels()):
                label.set_fontweight(tick_fontweight)
            if show_grid:
                axes[1].grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
            
            if has_clusters:
                # Cluster assignment vs time
                if cluster_colors is not None:
                    # Use custom colors
                    point_colors = [cluster_colors[label] for label in self.analyzer.cluster_labels]
                    axes[2].scatter(self.analyzer.time, self.analyzer.cluster_labels,
                                  s=msize*0.5, alpha=alpha, c=point_colors)
                else:
                    # Use tab10 colormap
                    axes[2].scatter(self.analyzer.time, self.analyzer.cluster_labels,
                                  s=msize*0.5, alpha=alpha, c=self.analyzer.cluster_labels,
                                  cmap='tab10')
                axes[2].set_ylabel('Cluster', fontsize=label_fontsize, fontweight=label_fontweight)
                axes[2].set_xlabel('Time (ns)', fontsize=label_fontsize, fontweight=label_fontweight)
                axes[2].set_yticks(range(len(self.analyzer.cluster_centers)))
                axes[2].tick_params(axis='both', labelsize=tick_fontsize)
                for label in (axes[2].get_xticklabels() + axes[2].get_yticklabels()):
                    label.set_fontweight(tick_fontweight)
                if show_grid:
                    axes[2].grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
            else:
                axes[1].set_xlabel('Time (ns)', fontsize=label_fontsize, fontweight=label_fontweight)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig    
    # -------------------------------------------------------------------------
    # Trajectory Analysis Visualization Methods
    # -------------------------------------------------------------------------
    
    def plot_rdf_comparison(self, rdf_key: str, 
                           cluster_ids: Optional[List[int]] = None,
                           figsize: Tuple[float, float] = (10, 6),
                           colors: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot RDF comparison between clusters.
        
        Parameters
        ----------
        rdf_key : str
            Key for RDF data (format: 'selection1__selection2')
        cluster_ids : list of int, optional
            Clusters to plot (default: all)
        figsize : tuple
            Figure size (default: (10, 6))
        colors : list of str, optional
            Colors for each cluster (default: automatic)
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        
        Example
        -------
        >>> plotter.plot_rdf_comparison('resname CIP__resname SOL and name OW')
        """
        if not hasattr(self.analyzer, 'rdf_data'):
            raise ValueError("No RDF data found. Run compute_rdf() first.")
        
        if rdf_key not in self.analyzer.rdf_data:
            raise ValueError(f"RDF key '{rdf_key}' not found. Available: {list(self.analyzer.rdf_data.keys())}")
        
        rdf_results = self.analyzer.rdf_data[rdf_key]
        
        if cluster_ids is None:
            cluster_ids = list(rdf_results.keys())
        
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot g(r)
        for i, cluster_id in enumerate(cluster_ids):
            data = rdf_results[cluster_id]
            ax1.plot(data['r'], data['rdf'], 
                    label=f"Cluster {cluster_id} ({data['n_frames']} frames)",
                    color=colors[i], lw=2, alpha=0.8)
        
        ax1.axhline(y=1.0, color='gray', ls='--', lw=1, alpha=0.5, label='Bulk density')
        ax1.set_ylabel('g(r)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Radial Distribution Function', fontsize=13, fontweight='bold')
        ax1.legend(frameon=True, fontsize=10)
        ax1.grid(alpha=0.3, ls='--')
        
        # Plot coordination number
        for i, cluster_id in enumerate(cluster_ids):
            data = rdf_results[cluster_id]
            ax2.plot(data['r'], data['count'], 
                    label=f"Cluster {cluster_id}",
                    color=colors[i], lw=2, alpha=0.8)
        
        ax2.set_xlabel('Distance r (Å)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Coordination Number', fontsize=12, fontweight='bold')
        ax2.set_title('Running Coordination Number', fontsize=13, fontweight='bold')
        ax2.legend(frameon=True, fontsize=10)
        ax2.grid(alpha=0.3, ls='--')
        
        # Add selection info
        sel1 = rdf_results[cluster_ids[0]]['selection1']
        sel2 = rdf_results[cluster_ids[0]]['selection2']
        fig.suptitle(f'{sel1} ↔ {sel2}', fontsize=11, y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_distance_distributions(self, 
                                   dist_key,
                                   cluster_ids: Optional[Union[str, List[int]]] = None,
                                   figsize: Tuple[float, float] = (12, 5),
                                   show_statistic_plot: bool = True,
                                   colors: Optional[Union[Dict[int, str], List[str]]] = None,
                                   xlim: Optional[Tuple[float, float]] = None,
                                   ylim_hist: Optional[Tuple[float, float]] = None,
                                   ylim_box: Optional[Tuple[float, float]] = None,
                                   show_title: bool = True,
                                   title: Optional[str] = None,
                                   title_fontsize: int = 11,
                                   title_fontweight: str = 'bold',
                                   xlabel: str = 'Distance (Å)',
                                   ylabel_hist: str = 'Counts',
                                   ylabel_box: str = 'Distance (Å)',
                                   label_fontsize: int = 12,
                                   label_fontweight: str = 'bold',
                                   tick_fontsize: int = 10,
                                   legend_fontsize: int = 10,
                                   legend_fontweight: str = 'normal',
                                   legend_loc: str = 'best',
                                   legend_ncol: int = 1,
                                   legend_frameon: bool = True,
                                   legend_framealpha: float = 0.9,
                                   linewidth: float = 2,
                                   linestyle: str = '-',
                                   alpha: float = 0.8,
                                   grid: bool = True,
                                   grid_alpha: float = 0.3,
                                   grid_linestyle: str = '--',
                                   show_values: bool = True,
                                   value_fontsize: int = 9,
                                   value_fontweight: str = 'bold',
                                   box_alpha: float = 0.6,
                                   median_color: str = 'red',
                                   median_linewidth: float = 2,
                                   save_fig: bool = False,
                                   dpi: int = 300,
                                   bbox_inches: str = 'tight') -> plt.Figure:
        """
        Plot distance distribution comparison between clusters with full customization.
        
        Automatically uses normalized histograms (counts per frame) if available from
        compute_distance_distribution(normalize=True), enabling fair comparison between
        clusters with different numbers of frames.
        
        Parameters
        ----------
        dist_key : str or list of str
            Key(s) for distance data. Can be:
            - Single string: plot one selection across clusters
            - List of strings: plot multiple selections on same figure (e.g., ['carboxylic_acid', 'quinolone'])
        cluster_ids : list of int, 'all', or None, optional
            Clusters to plot (default: all)
            Note: When plotting multiple dist_keys, uses first cluster by default for clarity
        
        Figure dimensions:
        figsize : tuple, default=(12, 5)
            Figure size (width, height) in inches
        show_statistic_plot : bool, default=True
            Show the box plot statistics panel (right side)
            Set to False to show only the histogram (simpler plot)
        xlim : tuple, optional
            X-axis limits (min, max) for histogram
        ylim_hist : tuple, optional
            Y-axis limits for histogram panel
        ylim_box : tuple, optional
            Y-axis limits for box plot panel
        
        Title styling:
        show_title : bool, default=True
            Show main figure title (e.g., 'CIP - surface_O')
            When False, hides title for a cleaner plot
        title : str, optional
            Overall figure title (auto-generated from selections if None)
        title_fontsize : int, default=11
            Figure title font size
        title_fontweight : str, default='bold'
            Figure title font weight
        
        Axis labels:
        xlabel : str, default='Distance (Å)'
            X-axis label for histogram
        ylabel_hist : str, default='Counts'
            Y-axis label for histogram
        ylabel_box : str, default='Distance (Å)'
            Y-axis label for box plot
        label_fontsize : int, default=12
            Axis label font size
        label_fontweight : str, default='bold'
            Axis label font weight
        tick_fontsize : int, default=10
            Tick label font size
        
        Legend styling:
        legend_fontsize : int, default=10
            Legend font size
        legend_fontweight : str, default='normal'
            Legend font weight
        legend_loc : str, default='best'
            Legend location
        legend_ncol : int, default=1
            Number of legend columns
        legend_frameon : bool, default=True
            Show legend frame
        legend_framealpha : float, default=0.9
            Legend frame transparency
        
        Line/curve styling:
        colors : dict or list, optional
            Colors for each cluster. Can be:
            - Dict: {cluster_id: color} e.g., {0: 'red', 1: 'blue'}
            - List: colors in cluster order
            - None: auto-generate colors
        linewidth : float, default=2
            Line width for histogram curves
        linestyle : str, default='-'
            Line style for histogram curves
        alpha : float, default=0.8
            Line transparency
        
        Grid styling:
        grid : bool, default=True
            Show grid
        grid_alpha : float, default=0.3
            Grid transparency
        grid_linestyle : str, default='--'
            Grid line style
        
        Box plot styling:
        box_alpha : float, default=0.6
            Box transparency
        median_color : str, default='red'
            Median line color
        median_linewidth : float, default=2
            Median line width
        show_values : bool, default=True
            Show mean±std text on boxes
        value_fontsize : int, default=9
            Value label font size
        value_fontweight : str, default='bold'
            Value label font weight
        
        Save options:
        save_fig : bool, default=False
            If True, auto-saves figure as 'distance_distribution_<dist_key>.png'
        dpi : int, default=300
            Resolution for saved figure
        bbox_inches : str, default='tight'
            Bounding box for saved figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        
        Examples
        --------
        >>> # Basic usage
        >>> fig = plotter.plot_distance_distributions(
        ...     dist_key='resname api__resname MMT and name Ob'
        ... )
        
        >>> # With custom colors and no statistics panel
        >>> fig = plotter.plot_distance_distributions(
        ...     dist_key='resname api__resname MMT and name Ob',
        ...     colors={0: 'red', 1: 'black', 2: 'blue', 3: 'green', 4: 'purple'},
        ...     show_statistic_plot=False,  # Only histogram
        ...     linewidth=3,
        ...     save_fig=True
        ... )
        
        >>> # Compare multiple functional groups
        >>> fig = plotter.plot_distance_distributions(
        ...     dist_key=['carboxylic_acid', 'quinolone', 'piperazine'],
        ...     cluster_ids=[0],  # Just one cluster for clarity
        ...     show_statistic_plot=False,
        ...     colors=['red', 'blue', 'green']
        ... )
        """
        if not hasattr(self.analyzer, 'distance_data'):
            raise ValueError("No distance data found. Run compute_distance_distribution() first.")
        
        # Handle multiple dist_keys
        if isinstance(dist_key, (list, tuple)):
            return self._plot_multiple_distance_distributions(
                dist_key, cluster_ids, figsize, show_statistic_plot, colors,
                xlim, ylim_hist, ylim_box, show_title, title, title_fontsize,
                title_fontweight, xlabel, ylabel_hist, ylabel_box, label_fontsize,
                label_fontweight, tick_fontsize, legend_fontsize, legend_fontweight,
                legend_loc, legend_ncol, legend_frameon, legend_framealpha,
                linewidth, linestyle, alpha, grid, grid_alpha, grid_linestyle,
                show_values, value_fontsize, value_fontweight, box_alpha,
                median_color, median_linewidth, save_fig, dpi, bbox_inches
            )
        
        if dist_key not in self.analyzer.distance_data:
            available_keys = list(self.analyzer.distance_data.keys())
            raise ValueError(
                f"Distance key '{dist_key}' not found.\n"
                f"Available keys: {available_keys}\n"
                f"Note: Keys use double underscore '__' as separator."
            )
        
        dist_results = self.analyzer.distance_data[dist_key]
        
        # Handle cluster_ids='all' or None
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = sorted(dist_results.keys())
        
        # Handle colors - can be dict or list
        if colors is None:
            colors_list = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
        elif isinstance(colors, dict):
            # Convert dict to list in cluster order
            colors_list = [colors.get(cid, plt.cm.tab10(i/len(cluster_ids))) 
                          for i, cid in enumerate(cluster_ids)]
        else:
            colors_list = colors
        
        # ============ Create figure layout ============
        fig = plt.figure(figsize=figsize)
        
        if show_statistic_plot:
            # Two-panel layout: histogram + box plot
            gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
        else:
            # Single panel: histogram only
            ax1 = fig.add_subplot(111)
            ax2 = None
        
        # ============ Histogram plot ============
        for i, cluster_id in enumerate(cluster_ids):
            data = dist_results[cluster_id]
            # Use normalized histogram if available (for cross-cluster comparison)
            hist_data = data.get('hist_normalized', data['hist'])
            if hist_data is None:
                hist_data = data['hist']
            ax1.plot(data['bin_centers'], hist_data, 
                    label=f"Cluster {cluster_id}",
                    color=colors_list[i], lw=linewidth, 
                    linestyle=linestyle, alpha=alpha)
        
        # Update ylabel if using normalized data
        if any(dist_results[cid].get('hist_normalized') is not None for cid in cluster_ids):
            ylabel_hist_actual = ylabel_hist if ylabel_hist != 'Counts' else 'Counts per frame'
        else:
            ylabel_hist_actual = ylabel_hist
        
        ax1.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax1.set_ylabel(ylabel_hist_actual, fontsize=label_fontsize, fontweight=label_fontweight)
        
        ax1.legend(loc=legend_loc, frameon=legend_frameon, fontsize=legend_fontsize,
                  framealpha=legend_framealpha, ncol=legend_ncol, 
                  prop={'weight': legend_fontweight})
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        
        if xlim:
            ax1.set_xlim(xlim)
        if ylim_hist:
            ax1.set_ylim(ylim_hist)
        if grid:
            ax1.grid(alpha=grid_alpha, ls=grid_linestyle)
        
        # ============ Box plot for statistics ============
        if show_statistic_plot and ax2 is not None:
            box_data = []
            labels = []
            for cluster_id in cluster_ids:
                box_data.append(dist_results[cluster_id]['distances'])
                labels.append(f"Cluster {cluster_id}")
            
            bp = ax2.boxplot(box_data, labels=labels, patch_artist=True,
                            medianprops=dict(color=median_color, linewidth=median_linewidth),
                            boxprops=dict(facecolor='lightblue', alpha=box_alpha))
            
            # Color boxes
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(box_alpha)
            
            ax2.set_ylabel(ylabel_box, fontsize=label_fontsize, fontweight=label_fontweight)
            ax2.tick_params(axis='both', labelsize=tick_fontsize)
            
            if ylim_box:
                ax2.set_ylim(ylim_box)
            if grid:
                ax2.grid(alpha=grid_alpha, ls=grid_linestyle, axis='y')
            
            # Add mean values as text
            if show_values:
                for i, cluster_id in enumerate(cluster_ids):
                    data = dist_results[cluster_id]
                    ax2.text(i+1, data['mean'], f"{data['mean']:.2f}±{data['std']:.2f}",
                            ha='center', va='bottom', fontsize=value_fontsize, 
                            fontweight=value_fontweight)
        
        # ============ Overall title ============
        if show_title:
            if title is None:
                # Auto-generate from selections with friendly names
                sel1 = dist_results[cluster_ids[0]]['selection1']
                sel2 = dist_results[cluster_ids[0]]['selection2']
                sel1_name = self.analyzer.get_selection_name(sel1) or sel1
                sel2_name = self.analyzer.get_selection_name(sel2) or sel2
                title = f'{sel1_name} - {sel2_name}'
            
            fig.suptitle(title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        plt.tight_layout()
        
        if save_fig:
            # Auto-generate filename from dist_key
            safe_key = dist_key.replace('__', '_to_').replace(' ', '_').replace('and', '')
            filename = f'distance_distribution_{safe_key}.png'
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            print(f"Saved: {filename}")
        
        return fig
    
    def _plot_multiple_distance_distributions(self, dist_keys, cluster_ids,
                                              figsize, show_statistic_plot, colors,
                                              xlim, ylim_hist, ylim_box, show_title, title,
                                              title_fontsize, title_fontweight, xlabel, ylabel_hist,
                                              ylabel_box, label_fontsize, label_fontweight,
                                              tick_fontsize, legend_fontsize, legend_fontweight,
                                              legend_loc, legend_ncol, legend_frameon,
                                              legend_framealpha, linewidth, linestyle, alpha,
                                              grid, grid_alpha, grid_linestyle, show_values,
                                              value_fontsize, value_fontweight, box_alpha,
                                              median_color, median_linewidth, save_fig, dpi, bbox_inches):
        """Internal method to plot multiple distance distributions (different selections) on same figure."""
        
        # Validate all keys exist
        for key in dist_keys:
            if key not in self.analyzer.distance_data:
                available_keys = list(self.analyzer.distance_data.keys())
                raise ValueError(f"Distance key '{key}' not found. Available: {available_keys}")
        
        # For multiple selections, default to first cluster only for clarity
        if cluster_ids is None or cluster_ids == 'all':
            first_dist = self.analyzer.distance_data[dist_keys[0]]
            cluster_ids = [sorted(first_dist.keys())[0]]
            print(f"Multiple selections: using cluster {cluster_ids[0]} for comparison")
        
        # Handle colors - can be dict or list
        if colors is None:
            colors_list = plt.cm.tab10(np.linspace(0, 1, len(dist_keys)))
        elif isinstance(colors, dict):
            colors_list = [colors.get(i, plt.cm.tab10(i/len(dist_keys))) 
                          for i in range(len(dist_keys))]
        else:
            colors_list = colors
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        if show_statistic_plot:
            gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
        else:
            ax1 = fig.add_subplot(111)
            ax2 = None
        
        # Plot each selection
        all_box_data = []
        all_labels = []
        
        for idx, key in enumerate(dist_keys):
            dist_results = self.analyzer.distance_data[key]
            
            for cluster_id in cluster_ids:
                if cluster_id not in dist_results:
                    continue
                    
                data = dist_results[cluster_id]
                hist_data = data.get('hist_normalized', data['hist'])
                if hist_data is None:
                    hist_data = data['hist']
                
                # Label with key name (includes cluster if multiple)
                if len(cluster_ids) > 1:
                    label = f"{key} (C{cluster_id})"
                else:
                    label = key
                
                ax1.plot(data['bin_centers'], hist_data,
                        label=label, color=colors_list[idx],
                        lw=linewidth, linestyle=linestyle, alpha=alpha)
                
                # Collect box data
                if show_statistic_plot and ax2 is not None:
                    all_box_data.append(data['distances'])
                    all_labels.append(label)
        
        # Check if normalized
        first_data = self.analyzer.distance_data[dist_keys[0]]
        first_cluster = list(first_data.keys())[0]
        if first_data[first_cluster].get('hist_normalized') is not None:
            ylabel_hist_actual = ylabel_hist if ylabel_hist != 'Counts' else 'Counts per frame'
        else:
            ylabel_hist_actual = ylabel_hist
        
        ax1.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax1.set_ylabel(ylabel_hist_actual, fontsize=label_fontsize, fontweight=label_fontweight)
        ax1.legend(loc=legend_loc, frameon=legend_frameon, fontsize=legend_fontsize,
                  framealpha=legend_framealpha, ncol=legend_ncol,
                  prop={'weight': legend_fontweight})
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        
        if xlim:
            ax1.set_xlim(xlim)
        if ylim_hist:
            ax1.set_ylim(ylim_hist)
        if grid:
            ax1.grid(alpha=grid_alpha, ls=grid_linestyle)
        
        # Box plot
        if show_statistic_plot and ax2 is not None and all_box_data:
            bp = ax2.boxplot(all_box_data, labels=all_labels, patch_artist=True,
                            medianprops=dict(color=median_color, linewidth=median_linewidth),
                            boxprops=dict(facecolor='lightblue', alpha=box_alpha))
            
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(box_alpha)
            
            ax2.set_ylabel(ylabel_box, fontsize=label_fontsize, fontweight=label_fontweight)
            ax2.tick_params(axis='both', labelsize=tick_fontsize)
            
            if ylim_box:
                ax2.set_ylim(ylim_box)
            if grid:
                ax2.grid(alpha=grid_alpha, ls=grid_linestyle, axis='y')
            
            if show_values:
                for i, (key, cluster_id) in enumerate([(k, cluster_ids[0]) for k in dist_keys]):
                    dist_results = self.analyzer.distance_data[key]
                    if cluster_id in dist_results:
                        data = dist_results[cluster_id]
                        ax2.text(i+1, data['mean'], f"{data['mean']:.2f}±{data['std']:.2f}",
                                ha='center', va='bottom', fontsize=value_fontsize,
                                fontweight=value_fontweight)
        
        # Title
        if show_title:
            if title is None:
                title = f"Distance Distributions Comparison"
            fig.suptitle(title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        plt.tight_layout()
        
        if save_fig:
            filename = f"distance_distribution_{'_'.join(dist_keys)}.png"
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            print(f"Saved: {filename}")
        
        return fig
    
    def plot_distance_distributions_3d(self,
                                      dist_key,
                                      cluster_ids: Optional[Union[str, List[int]]] = None,
                                      x_axis_mode: str = 'cluster',
                                      figsize: Tuple[float, float] = (16, 12),
                                      colors: Optional[Union[Dict[int, str], List[str]]] = None,
                                      xlim: Optional[Tuple[float, float]] = None,
                                      show_title: bool = True,
                                      title: Optional[str] = None,
                                      title_fontsize: int = 14,
                                      title_fontweight: str = 'bold',
                                      label_fontsize: int = 12,
                                      label_fontweight: str = 'bold',
                                      xlabel: Optional[str] = None,
                                      ylabel: str = 'Distance (Å)',
                                      zlabel: str = 'Counts per frame',
                                      show_xlabel: bool = True,
                                      show_ylabel: bool = True,
                                      show_zlabel: bool = True,
                                      tick_fontsize: int = 10,
                                      show_legend: bool = True,
                                      legend_loc: str = 'upper left',
                                      legend_ncol: int = 1,
                                      legend_bbox: Optional[Tuple[float, float]] = None,
                                      legend_fontsize: int = 9,
                                      legend_fontweight: str = 'normal',
                                      legend_framealpha: float = 0.9,
                                      linewidth: float = 2.0,
                                      linestyles: Optional[List] = None,
                                      linewidths: Optional[List[float]] = None,
                                      linecolors: Optional[List[str]] = None,
                                      alpha: float = 0.8,
                                      elevation: float = 20,
                                      azimuth: float = 45,
                                      fill_under_curve: bool = True,
                                      fill_alpha: float = 0.3,
                                      fill_clusters: Optional[Union[str, List[int]]] = 'all',
                                      fill_selections: Optional[Union[str, List[str]]] = 'all',
                                      show_grid: bool = True,
                                      grid_alpha: float = 0.2,
                                      # Cluster spacing control
                                      cluster_spacing: float = 1.0,
                                      # 3D plot spacing adjustments
                                      xlabel_pad: float = 10,
                                      ylabel_pad: float = 10,
                                      zlabel_pad: float = 10,
                                      xtick_pad: float = 5,
                                      ytick_pad: float = 5,
                                      ztick_pad: float = 5,
                                      xtick_rotation: float = 0,
                                      subplot_left: float = 0.02,
                                      subplot_right: float = 0.98,
                                      subplot_bottom: float = 0.05,
                                      subplot_top: float = 0.98,
                                      tight_layout_pad: float = 2.0,
                                      # Export
                                      save_fig: bool = False,
                                      dpi: int = 300,
                                      bbox_inches: Optional[str] = 'tight',
                                      pad_inches: float = 0.1) -> plt.Figure:
        """
        Create 3D visualization of distance distributions.
        
        Two plotting modes controlled by x_axis_mode:
        
        Mode 'cluster' (default):
        - X-axis: Cluster ID (C0, C1, C2, ...)
        - Y-axis: Distance (Å)
        - Z-axis: Counts per frame
        - Multiple selections differentiated by line styles/widths
        
        Mode 'selection':
        - X-axis: Selection groups (carboxylic_acid, quinolone, ...)
        - Y-axis: Distance (Å)
        - Z-axis: Counts per frame
        - Multiple clusters differentiated by line styles/widths
        
        Parameters
        ----------
        dist_key : str or list of str
            Key(s) for distance data. Use list to compare multiple selections.
        cluster_ids : list of int, 'all', or None, optional
            Clusters to plot (default: all)
        x_axis_mode : str, default='cluster'
            Determines x-axis grouping: 'cluster' or 'selection'
        figsize : tuple, default=(16, 12)
            Figure size (width, height) in inches
        colors : dict or list, optional
            Colors for each cluster. Dict: {cluster_id: color}, List: colors in order
        xlim : tuple, optional
            Distance range (min, max) to display
        show_title : bool, default=True
            Show figure title
        title : str, optional
            Figure title (auto-generated if None)
        title_fontsize : int, default=14
            Title font size
        title_fontweight : str, default='bold'
            Title font weight
        label_fontsize : int, default=12
            Axis label font size
        label_fontweight : str, default='bold'
            Axis label font weight
        xlabel : str, optional
            X-axis label. Auto-generated based on x_axis_mode if None.
        ylabel : str, default='Distance (Å)'
            Y-axis label
        zlabel : str, default='Counts per frame'
            Z-axis label
        show_xlabel : bool, default=True
            Whether to display the x-axis label
        show_ylabel : bool, default=True
            Whether to display the y-axis label
        show_zlabel : bool, default=True
            Whether to display the z-axis label
        tick_fontsize : int, default=10
            Tick label font size
        show_legend : bool, default=True
            Show legend
        legend_loc : str, default='upper left'
            Legend location. Options: 'best', 'upper right', 'upper left', 'lower left',
            'lower right', 'center', 'center left', 'center right', 'lower center', 'upper center'
        legend_ncol : int, default=1
            Number of columns in legend
        legend_bbox : tuple of float, optional
            Custom bbox_to_anchor for legend positioning.
            If None, uses standard loc positioning (legend inside plot).
            Example: (1.05, 1.0) places legend outside to the right
        legend_fontsize : int, default=9
            Legend font size
        legend_fontweight : str, default='normal'
            Legend font weight
        legend_framealpha : float, default=0.9
            Legend frame transparency (0=transparent, 1=opaque)
        linewidth : float, default=2.0
            Default line width for curves
        linestyles : list, optional
            Line styles for multiple selections. Can be dash patterns or thickness variations.
            Examples: ['-', '--', '-.'] or use linewidths instead for thickness variations
        linewidths : list of float, optional
            Line widths for multiple selections (e.g., [1.5, 2.5, 3.5])
            If provided, overrides linewidth parameter for each selection
        linecolors : list of str, optional
            Colors for the lines themselves (independent of mode-based color assignment).
            In 'cluster' mode: overrides cluster colors for the lines (e.g., ['red', 'blue', 'green'] for selections)
            In 'selection' mode: overrides selection colors for the lines (e.g., ['red', 'blue'] for clusters)
            If None, uses the mode-based colors from 'colors' parameter
        alpha : float, default=0.8
            Line transparency
        elevation : float, default=20
            Viewing elevation angle (degrees)
        azimuth : float, default=45
            Viewing azimuth angle (degrees)
        fill_under_curve : bool, default=True
            Fill area under curves
        fill_alpha : float, default=0.3
            Transparency for curve fill
        fill_clusters : list of int, 'all', or None, default='all'
            Which clusters to fill. 'all' fills all, None fills none, or provide list like [0, 2]
        fill_selections : list of str, 'all', or None, default='all'
            Which selections to fill. 'all' fills all, None fills none, or provide list like ['carboxylic_acid']
        show_grid : bool, default=True
            Show grid lines
        grid_alpha : float, default=0.2
            Grid transparency
        cluster_spacing : float, default=1.0
            Controls separation between clusters/selections on x-axis.
            - 1.0: Default spacing (clusters at integer positions)
            - 0.5: Tighter spacing (clusters closer together)
            - 2.0: Wider spacing (clusters further apart)
            - Does not affect distance range (Y-axis), only x-axis positioning
            - Centers clusters around x=0 instead of spreading to edges
        
        3D Plot Spacing Adjustments
        ---------------------------
        xlabel_pad : float, default=10
            Distance between x-axis and its label (in points)
        ylabel_pad : float, default=10
            Distance between y-axis and its label (in points)
        zlabel_pad : float, default=10
            Distance between z-axis and its label (in points)
        xtick_pad : float, default=5
            Distance between x-axis tick marks and their labels (in points).
            Increase to prevent overlap with y-axis tick labels at corners.
        ytick_pad : float, default=5
            Distance between y-axis tick marks and their labels (in points).
            Increase to prevent overlap with x-axis tick labels at corners.
        ztick_pad : float, default=5
            Distance between z-axis tick marks and their labels (in points)
        xtick_rotation : float, default=0
            Rotation angle in degrees for x-axis tick labels.
            Use 15-45 degrees to reduce horizontal space and prevent overlap.
        subplot_left : float, default=0.02
            Left margin for subplot (0-1 scale)
        subplot_right : float, default=0.98
            Right margin for subplot (0-1 scale)
        subplot_bottom : float, default=0.05
            Bottom margin for subplot (0-1 scale)
        subplot_top : float, default=0.98
            Top margin for subplot (0-1 scale)
        tight_layout_pad : float, default=2.0
            Padding for tight_layout (not effective for 3D plots, kept for compatibility)
        
        Export
        ------
        save_fig : bool, default=False
            Auto-save figure
        dpi : int, default=300
            Resolution for saved figure
        bbox_inches : str, default='tight'
            Bounding box for saved figure. Use 'tight' to minimize white space,
            or None to use manual subplot positioning.
        pad_inches : float, default=0.1
            Padding in inches around the figure when bbox_inches='tight'.
            Reduce (e.g., 0.05 or 0.02) to minimize margins, especially when
            show_title=False and no title space is needed.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        
        Examples
        --------
        >>> # Single selection across all clusters
        >>> fig = plotter.plot_distance_distributions_3d(
        ...     dist_key='carboxylic_acid',
        ...     cluster_ids='all'
        ... )
        
        >>> # Multiple selections with custom line widths, fill only specific ones
        >>> fig = plotter.plot_distance_distributions_3d(
        ...     dist_key=['carboxylic_acid', 'quinolone', 'piperazine'],
        ...     cluster_ids=[0, 1, 2],
        ...     linewidths=[1.5, 2.5, 3.5],  # Vary thickness instead of dash style
        ...     fill_clusters=[0, 1],  # Only fill clusters 0 and 1
        ...     fill_selections=['carboxylic_acid'],  # Only fill carboxylic_acid
        ...     legend_loc='best',  # Automatic best position
        ...     legend_ncol=2,  # Two columns
        ...     legend_framealpha=0,  # Transparent background
        ...     elevation=25,
        ...     azimuth=60
        ... )
        """
        if not hasattr(self.analyzer, 'distance_data'):
            raise ValueError("No distance data found. Run compute_distance_distribution() first.")
        
        # Validate x_axis_mode
        if x_axis_mode not in ['cluster', 'selection']:
            raise ValueError(f"x_axis_mode must be 'cluster' or 'selection', got '{x_axis_mode}'")
        
        # Handle single or multiple dist_keys
        if isinstance(dist_key, (list, tuple)):
            dist_keys = list(dist_key)
        else:
            dist_keys = [dist_key]
        
        # Validate all keys exist
        for key in dist_keys:
            if key not in self.analyzer.distance_data:
                available_keys = list(self.analyzer.distance_data.keys())
                raise ValueError(f"Distance key '{key}' not found. Available: {available_keys}")
        
        # Get cluster_ids from first dist_key
        first_dist = self.analyzer.distance_data[dist_keys[0]]
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = sorted(first_dist.keys())
        
        # Auto-generate xlabel based on mode if not provided
        if xlabel is None:
            xlabel = 'Cluster' if x_axis_mode == 'cluster' else 'Selection'
        
        # Handle colors based on x_axis_mode
        if x_axis_mode == 'cluster':
            # Colors represent clusters
            n_items = len(cluster_ids)
            if colors is None:
                colors_list = plt.cm.tab10(np.linspace(0, 1, n_items))
            elif isinstance(colors, dict):
                colors_list = [colors.get(cid, plt.cm.tab10(i/n_items)) 
                              for i, cid in enumerate(cluster_ids)]
            else:
                colors_list = colors
        else:  # x_axis_mode == 'selection'
            # Colors represent selections
            n_items = len(dist_keys)
            if colors is None:
                colors_list = plt.cm.tab10(np.linspace(0, 1, n_items))
            elif isinstance(colors, list):
                colors_list = colors
            else:
                # If dict provided, try to map to selection keys
                colors_list = [colors.get(key, plt.cm.tab10(i/n_items)) 
                              for i, key in enumerate(dist_keys)]
        
        # Define line styles for multiple selections
        if linestyles is None:
            line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5))]
        else:
            line_styles = linestyles
        
        # Define line widths for multiple selections
        if linewidths is None:
            line_widths = [linewidth] * len(dist_keys)
        else:
            line_widths = linewidths
        
        # Define line colors - independent of mode-based color assignment
        # In cluster mode: colors for different selections
        # In selection mode: colors for different clusters
        if linecolors is not None:
            line_colors = linecolors
        else:
            line_colors = None  # Will use colors_list based on mode
        
        # Determine which clusters/selections to fill
        if fill_clusters == 'all':
            clusters_to_fill = set(cluster_ids)
        elif fill_clusters is None:
            clusters_to_fill = set()
        else:
            clusters_to_fill = set(fill_clusters)
        
        if fill_selections == 'all':
            selections_to_fill = set(dist_keys)
        elif fill_selections is None:
            selections_to_fill = set()
        else:
            selections_to_fill = set(fill_selections)
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize, facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        
        print(f"Creating 3D plot with {len(cluster_ids)} clusters and {len(dist_keys)} selection(s)...")
        print(f"X-axis mode: {x_axis_mode}")
        
        if x_axis_mode == 'cluster':
            # X-axis shows clusters, selections differentiated by line style
            # Center clusters around x=0 with controlled spacing
            cluster_positions = np.arange(len(cluster_ids)) - (len(cluster_ids) - 1) / 2
            x_positions = cluster_spacing * cluster_positions
            
            for sel_idx, key in enumerate(dist_keys):
                dist_results = self.analyzer.distance_data[key]
                line_style = line_styles[sel_idx % len(line_styles)]
                line_width = line_widths[sel_idx % len(line_widths)]
                # Use linecolors if provided, else use cluster-based colors
                line_color = line_colors[sel_idx % len(line_colors)] if line_colors else None
                
                for cluster_idx, cluster_id in enumerate(cluster_ids):
                    if cluster_id not in dist_results:
                        continue
                    
                    data = dist_results[cluster_id]
                    hist_data = data.get('hist_normalized', data['hist'])
                    if hist_data is None:
                        hist_data = data['hist']
                    distances = data['bin_centers']
                    
                    if xlim is not None:
                        mask = (distances >= xlim[0]) & (distances <= xlim[1])
                        distances = distances[mask]
                        hist_data = hist_data[mask]
                    
                    x = np.full_like(distances, x_positions[cluster_idx])
                    
                    label = f"{key} (Cluster {cluster_id})" if len(dist_keys) > 1 else f"Cluster {cluster_id}"
                    
                    # Determine color: linecolor overrides cluster color
                    plot_color = line_color if line_color else colors_list[cluster_idx]
                    
                    ax.plot(x, distances, hist_data,
                           label=label, linewidth=line_width,
                           color=plot_color, linestyle=line_style, alpha=alpha)
                    
                    should_fill = (fill_under_curve and 
                                  cluster_id in clusters_to_fill and 
                                  key in selections_to_fill)
                    
                    if should_fill:
                        x_fill = np.vstack([x, x])
                        y_fill = np.vstack([distances, distances])
                        z_fill = np.vstack([np.zeros_like(hist_data), hist_data])
                        ax.plot_surface(x_fill, y_fill, z_fill,
                                       color=plot_color, alpha=fill_alpha,
                                       linewidth=0, shade=True)
        
        else:  # x_axis_mode == 'selection'
            # X-axis shows selections, clusters differentiated by line style
            # Center selections around x=0 with controlled spacing
            selection_positions = np.arange(len(dist_keys)) - (len(dist_keys) - 1) / 2
            x_positions = cluster_spacing * selection_positions
            
            for cluster_idx, cluster_id in enumerate(cluster_ids):
                line_style = line_styles[cluster_idx % len(line_styles)]
                line_width = line_widths[cluster_idx % len(line_widths)]
                # Use linecolors if provided, else use selection-based colors
                line_color = line_colors[cluster_idx % len(line_colors)] if line_colors else None
                
                for sel_idx, key in enumerate(dist_keys):
                    dist_results = self.analyzer.distance_data[key]
                    
                    if cluster_id not in dist_results:
                        continue
                    
                    data = dist_results[cluster_id]
                    hist_data = data.get('hist_normalized', data['hist'])
                    if hist_data is None:
                        hist_data = data['hist']
                    distances = data['bin_centers']
                    
                    if xlim is not None:
                        mask = (distances >= xlim[0]) & (distances <= xlim[1])
                        distances = distances[mask]
                        hist_data = hist_data[mask]
                    
                    x = np.full_like(distances, x_positions[sel_idx])
                    
                    label = f"Cluster {cluster_id} ({key})" if len(cluster_ids) > 1 else f"{key}"
                    
                    # Determine color: linecolor overrides selection color
                    plot_color = line_color if line_color else colors_list[sel_idx]
                    
                    ax.plot(x, distances, hist_data,
                           label=label, linewidth=line_width,
                           color=plot_color, linestyle=line_style, alpha=alpha)
                    
                    should_fill = (fill_under_curve and 
                                  cluster_id in clusters_to_fill and 
                                  key in selections_to_fill)
                    
                    if should_fill:
                        x_fill = np.vstack([x, x])
                        y_fill = np.vstack([distances, distances])
                        z_fill = np.vstack([np.zeros_like(hist_data), hist_data])
                        ax.plot_surface(x_fill, y_fill, z_fill,
                                       color=plot_color, alpha=fill_alpha,
                                       linewidth=0, shade=True)
        
        # Set Y-axis limits based on xlim (distance range)
        if xlim is not None:
            ax.set_ylim(xlim[0], xlim[1])
            
        # Fix X-axis limits to show spacing effect (prevent auto-scaling)
        # Use wider range than actual positions to make spacing visible
        max_spacing = max(1.0, abs(max(x_positions)), abs(min(x_positions)))
        ax.set_xlim(-max_spacing * 1.5, max_spacing * 1.5)
        
        # Customize 3D plot
        # Use configurable labelpad values to prevent cropping in 3D plots
        if show_xlabel:
            ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight, labelpad=xlabel_pad)
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight, labelpad=ylabel_pad)
        if show_zlabel:
            ax.set_zlabel(zlabel, fontsize=label_fontsize, fontweight=label_fontweight, labelpad=zlabel_pad)
        
        # Disable clipping on z-axis label to prevent it being cut off by figure boundaries
        # This is necessary because 3D plot labels can extend outside the axes box
        ax.zaxis.label.set_clip_on(False)
        
        # Set X-axis labels based on mode
        ax.set_xticks(x_positions)
        
        # Determine horizontal alignment based on rotation angle
        if xtick_rotation > 0:
            ha = 'right'  # Right-align for positive (clockwise) rotation
        elif xtick_rotation < 0:
            ha = 'left'   # Left-align for negative (counter-clockwise) rotation
        else:
            ha = 'center' # Center-align for no rotation
        
        if x_axis_mode == 'cluster':
            ax.set_xticklabels([f"C{cid}" for cid in cluster_ids], 
                              rotation=xtick_rotation, ha=ha)
        else:  # x_axis_mode == 'selection'
            ax.set_xticklabels(dist_keys, rotation=xtick_rotation, ha=ha)
        
        # Configure tick label appearance and padding
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.tick_params(axis='x', pad=xtick_pad)
        ax.tick_params(axis='y', pad=ytick_pad)
        ax.tick_params(axis='z', pad=ztick_pad)
        
        # Title
        if show_title:
            if title is None:
                if len(dist_keys) == 1:
                    title = f"3D Distance Distribution: {dist_keys[0]}"
                else:
                    title = f"3D Distance Distributions Comparison"
                if xlim is not None:
                    title += f"\\n(Distance: {xlim[0]:.1f}-{xlim[1]:.1f} Å)"
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight, pad=20)
        
        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Set z-axis label rotation AFTER view_init to make text read bottom-to-top
        # For 3D axes, we need to disable automatic rotation and set it manually
        if show_zlabel:
            ax.zaxis.set_rotate_label(False)  # Disable automatic label rotation
            ax.zaxis.label.set_rotation(90)   # Set to 90 to read bottom-to-top
        
        # Add legend
        if show_legend and len(cluster_ids) * len(dist_keys) <= 20:
            from matplotlib.lines import Line2D
            legend_elements = []
            
            if x_axis_mode == 'cluster' and len(dist_keys) > 1:
                # Cluster mode: show selections in legend
                for i, key in enumerate(dist_keys):
                    line_style = line_styles[i % len(line_styles)]
                    line_width_legend = line_widths[i % len(line_widths)]
                    # Use linecolor if provided, else gray
                    legend_color = line_colors[i % len(line_colors)] if line_colors else 'gray'
                    legend_elements.append(Line2D([0], [0], color=legend_color, linestyle=line_style,
                                                 linewidth=line_width_legend, label=key))
            elif x_axis_mode == 'selection' and len(cluster_ids) > 1:
                # Selection mode: show clusters in legend
                for i, cluster_id in enumerate(cluster_ids):
                    line_style = line_styles[i % len(line_styles)]
                    line_width_legend = line_widths[i % len(line_widths)]
                    # Use linecolor if provided, else gray
                    legend_color = line_colors[i % len(line_colors)] if line_colors else 'gray'
                    legend_elements.append(Line2D([0], [0], color=legend_color, linestyle=line_style,
                                                 linewidth=line_width_legend, label=f"Cluster {cluster_id}"))
            
            if legend_elements:
                
                if legend_bbox is not None:
                    ax.legend(handles=legend_elements, loc=legend_loc,
                             bbox_to_anchor=legend_bbox, ncol=legend_ncol,
                             framealpha=legend_framealpha,
                             prop={'size': legend_fontsize, 'weight': legend_fontweight})
                else:
                    ax.legend(handles=legend_elements, loc=legend_loc,
                             ncol=legend_ncol, framealpha=legend_framealpha,
                             prop={'size': legend_fontsize, 'weight': legend_fontweight})
        elif not show_legend:
            pass  # No legend
        else:
            print(f"Legend suppressed (too many combinations: {len(cluster_ids)} × {len(dist_keys)})")
        
        # Remove grey tint and configure grid
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        if show_grid:
            ax.xaxis.pane.set_edgecolor('black')
            ax.yaxis.pane.set_edgecolor('black')
            ax.zaxis.pane.set_edgecolor('black')
            ax.xaxis.pane.set_alpha(grid_alpha)
            ax.yaxis.pane.set_alpha(grid_alpha)
            ax.zaxis.pane.set_alpha(grid_alpha)
            ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5)
        else:
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            ax.xaxis.pane.set_alpha(0)
            ax.yaxis.pane.set_alpha(0)
            ax.zaxis.pane.set_alpha(0)
            ax.grid(False)
        
        fig.patch.set_facecolor('white')
        
        # For 3D plots, tight_layout() doesn't work properly
        # Instead, we use bbox_inches='tight' when saving to minimize white space
        # The set_clip_on(False) on z-label ensures it remains visible despite tight cropping
        
        if save_fig:
            mode_suffix = f"_{x_axis_mode}mode"
            if len(dist_keys) == 1:
                filename = f"distance_distribution_3d_{dist_keys[0]}{mode_suffix}.png"
            else:
                filename = f"distance_distribution_3d_{'_'.join(dist_keys[:3])}{mode_suffix}.png"
            
            # When using bbox_inches='tight', explicitly include z-label in bounding box calculation
            # pad_inches controls the amount of padding around the tight bounding box
            if bbox_inches == 'tight':
                fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                           bbox_extra_artists=[ax.zaxis.label], pad_inches=pad_inches)
            else:
                fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            print(f"Saved: {filename}")
        
        return fig
    
    def plot_multi_system_distance_distributions_3d(self,
                                                    systems_data: Dict,
                                                    dist_key: Union[str, List[str]],
                                                    cluster_ids: Optional[Union[str, List[int]]] = None,
                                                    figsize: Tuple[float, float] = (18, 12),
                                                    colors: Optional[Union[Dict[int, str], List[str]]] = None,
                                                    colors_per_system: Optional[Dict[str, Union[str, List[str]]]] = None,
                                                    xlim: Optional[Tuple[float, float]] = None,
                                                    show_title: bool = True,
                                                    title: Optional[str] = None,
                                                    title_fontsize: int = 16,
                                                    title_fontweight: str = 'bold',
                                                    label_fontsize: int = 12,
                                                    label_fontweight: str = 'bold',
                                                    ylabel: str = 'Distance (Å)',
                                                    zlabel: str = 'Counts / frame',
                                                    tick_fontsize: int = 10,
                                                    show_legend: bool = True,
                                                    show_moiety_legend: bool = True,
                                                    legend_labels: Optional[List[str]] = None,
                                                    moiety_labels: Optional[Dict[str, str]] = None,
                                                    legend_title: str = 'Cluster',
                                                    moiety_legend_title: str = 'Moiety',
                                                    legend_loc: str = 'upper left',
                                                    legend_ncol_cluster: int = 1,
                                                    legend_ncol_moiety: int = 1,
                                                    legend1_bbox: Optional[Tuple[float, float]] = None,
                                                    legend2_bbox: Optional[Tuple[float, float]] = None,
                                                    legend_fontsize: int = 10,
                                                    legend_fontweight: str = 'normal',
                                                    legend_title_fontsize: int = 11,
                                                    legend_title_fontweight: str = 'bold',
                                                    legend_framealpha: float = 0.9,
                                                    legend_edgecolor: str = 'black',
                                                    legend_handletextpad: float = 0.8,
                                                    linewidth: float = 2.5,
                                                    linewidths: Optional[List[float]] = None,
                                                    linestyles: Optional[List] = None,
                                                    alpha: float = 0.8,
                                                    elevation: float = 20,
                                                    azimuth: float = 45,
                                                    fill_under_curve: bool = True,
                                                    fill_alpha: float = 0.3,
                                                    fill_selections: Optional[Union[str, List[str]]] = 'all',
                                                    show_grid: bool = True,
                                                    grid_alpha: float = 0.2,
                                                    grid_linewidth: float = 0.5,
                                                    system_spacing: float = 2.0,
                                                    cluster_spacing: float = 1.0,
                                                    xlabel: str = 'System',
                                                    xlabel_pad: float = 15,
                                                    ylabel_pad: float = 15,
                                                    zlabel_pad: float = 15,
                                                    xtick_pad: float = 8,
                                                    ytick_pad: float = 8,
                                                    ztick_pad: float = 8,
                                                    xtick_rotation: float = 0,
                                                    system_label_rotation: Optional[float] = None,
                                                    system_label_y_offset: float = 0.10,
                                                    system_display_names: Optional[Dict[str, str]] = None,
                                                    system_label_fontfamily: str = 'Times New Roman',
                                                    system_label_fontsize: Optional[int] = None,
                                                    system_label_fontweight: str = 'normal',
                                                    save_fig: bool = False,
                                                    save_path: Optional[str] = None,
                                                    dpi: int = 300,
                                                    bbox_inches: str = 'tight',
                                                    pad_inches: float = 0.1,
                                                    save_right_pad: float = 0.3) -> plt.Figure:
        """
        Create 3D visualization comparing distance distributions across multiple systems.
        
        Similar to plot_multi_system_free_energy_barplot, but for 3D distance distributions.
        Systems are separated along the x-axis, with clusters within each system grouped together.
        
        Parameters
        ----------
        systems_data : dict
            Dictionary with structure:
            {system_name: {
                'system_name': str,
                'distance_data': dict of distance distributions (from compute_distance_distribution),
                'cluster_ids': list of cluster IDs
            }}
            
        dist_key : str or list of str
            Key(s) for the distance data to plot.
            - Single: 'carboxylic_acid'
            - Multiple: ['carboxylic_acid', 'quinolone', 'piperazine']
            When multiple keys provided, they are overlaid at each cluster position
            and differentiated by linewidth/linestyle.
            
        cluster_ids : list of int, 'all', or None, optional
            Clusters to plot. If 'all' or None, plots all available clusters.
            Will use clusters common across all systems or all available if specified per-system.
            
        colors : dict or list, optional
            Cluster color mapping (overrides colors_per_system for cluster colors).
            Format: {cluster_id: color_string}
            Example: {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple'}
            All systems use the same color for each cluster ID.
            If None, uses colors_per_system or default palette.
            
        colors_per_system : dict, optional
            Custom colors for each system. Supports two formats:
            1. Single color per system: {system_name: color_string}
               Example: {'CIP+': '#FFB6C1', 'CIP+/-': '#ADD8E6'}
               All clusters in system use same color.
            2. List of colors per system: {system_name: [color1, color2, ...]}
               Example: {'CIP+': ['#FFB6C1', '#ADD8E6', '#90EE90'], ...}
               Each cluster gets its own color from the list.
            If colors parameter is provided, colors_per_system is ignored for cluster coloring.
            If None, uses default color palette.
            
        figsize : tuple, default=(18, 12)
            Figure size (width, height) in inches
            
        xlim : tuple, optional
            Distance range (min, max) to display on Y-axis
            
        show_title : bool, default=True
            Show figure title
            
        title : str, optional
            Figure title (auto-generated if None)
            
        title_fontsize : int, default=16
            Title font size
            
        title_fontweight : str, default='bold'
            Title font weight
            
        label_fontsize : int, default=12
            Axis label font size
            
        label_fontweight : str, default='bold'
            Axis label font weight
            
        ylabel : str, default='Distance (Å)'
            Y-axis label
            
        zlabel : str, default='Counts per frame'
            Z-axis label
            
        tick_fontsize : int, default=10
            Tick label font size
            
        show_legend : bool, default=True
            Show legend
            
        legend_loc : str, default='upper left'
            Legend location
            
        legend_ncol : int, default=1
            Number of columns in legend
            
        legend_bbox : tuple, optional
            Custom bbox_to_anchor for legend positioning
            
        legend_fontsize : int, default=10
            Legend font size
            
        legend_fontweight : str, default='normal'
            Legend font weight
            
        legend_framealpha : float, default=0.9
            Legend frame transparency
            
        linewidth : float, default=2.5
            Default line width for curves (used when linewidths not provided)
            
        linewidths : list of float, optional
            Line widths for multiple selections (when dist_key is a list).
            Example: [3, 2, 1] for [carboxylic_acid, quinolone, piperazine]
            If None, uses linewidth for all selections.
            
        linestyles : list, optional
            Line styles for multiple selections (when dist_key is a list).
            Example: ['-', '--', '-.'] or ['-', '-', '-']
            If None, uses solid lines for all.
            
        alpha : float, default=0.8
            Line transparency
            
        elevation : float, default=20
            Viewing elevation angle (degrees)
            
        azimuth : float, default=45
            Viewing azimuth angle (degrees)
            
        fill_under_curve : bool, default=True
            Fill area under curves
            
        fill_alpha : float, default=0.3
            Transparency for curve fill
            
        fill_selections : list of str, 'all', or None, default='all'
            Which selections to fill when dist_key is a list.
            Example: ['carboxylic_acid'] fills only carboxylic_acid
            'all' fills all selections, None fills none.
            
        show_grid : bool, default=True
            Show grid lines
            
        grid_alpha : float, default=0.2
            Grid transparency
            
        system_spacing : float, default=2.0
            Spacing between different systems on x-axis
            
        cluster_spacing : float, default=1.0
            Spacing between clusters within the same system
            
        xlabel_pad : float, default=15
            Distance between x-axis and its label
            
        ylabel_pad : float, default=15
            Distance between y-axis and its label
            
        zlabel_pad : float, default=15
            Distance between z-axis and its label
            
        xtick_pad : float, default=8
            Distance between x-tick marks and labels
            
        ytick_pad : float, default=8
            Distance between y-tick marks and labels
            
        ztick_pad : float, default=8
            Distance between z-tick marks and labels
            
        xtick_rotation : float, default=30
            Rotation angle for x-tick labels
            
        save_fig : bool, default=False
            Auto-save figure
            
        save_path : str, optional
            Path to save figure (auto-generated if None)
            
        dpi : int, default=300
            Resolution for saved figure
            
        bbox_inches : str, default='tight'
            Bounding box for saved figure
            
        pad_inches : float, default=0.1
            Padding around figure when bbox_inches='tight'
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            
        Examples
        --------
        >>> # Prepare systems data
        >>> systems_data = {
        ...     'CIP+': {
        ...         'system_name': 'CIP+',
        ...         'distance_data': analyzer1.distance_data,
        ...         'cluster_ids': [0, 1, 2]
        ...     },
        ...     'CIP+/-': {
        ...         'system_name': 'CIP+/-',
        ...         'distance_data': analyzer2.distance_data,
        ...         'cluster_ids': [0, 1, 2, 3]
        ...     }
        ... }
        >>> 
        >>> # Plot with single color per system
        >>> fig = plotter.plot_multi_system_distance_distributions_3d(
        ...     systems_data,
        ...     dist_key='carboxylic_acid',
        ...     colors_per_system={'CIP+': '#FF6B6B', 'CIP+/-': '#4ECDC4'},
        ...     system_spacing=3.0,
        ...     cluster_spacing=1.0,
        ...     elevation=25,
        ...     azimuth=60
        ... )
        >>> 
        >>> # Or with different colors per cluster (like free energy barplot)
        >>> fig = plotter.plot_multi_system_distance_distributions_3d(
        ...     systems_data,
        ...     dist_key='carboxylic_acid',
        ...     colors_per_system={
        ...         'CIP+': ['#FFB6C1', '#ADD8E6', '#90EE90'],
        ...         'CIP+/-': ['#FFB6C1', '#ADD8E6', '#90EE90', '#FFFFE0']
        ...     },
        ...     system_spacing=3.0,
        ...     elevation=25,
        ...     save_fig=True
        ... )
        """
        # Validate systems_data
        if not systems_data:
            raise ValueError("systems_data cannot be empty")
        
        system_names = list(systems_data.keys())
        n_systems = len(system_names)
        
        # Handle single or multiple dist_keys
        if isinstance(dist_key, (list, tuple)):
            dist_keys = list(dist_key)
        else:
            dist_keys = [dist_key]
        
        print(f"\\nCreating multi-system 3D distance distribution plot")
        print(f"Systems: {system_names}")
        print(f"Distance keys: {dist_keys}")
        
        # Set up linewidths and linestyles for multiple selections
        if linewidths is None:
            line_widths = [linewidth] * len(dist_keys)
        else:
            line_widths = linewidths
        
        if linestyles is None:
            line_styles = ['-'] * len(dist_keys)
        else:
            line_styles = linestyles
        
        # Determine which selections to fill
        if fill_selections == 'all':
            selections_to_fill = set(dist_keys)
        elif fill_selections is None:
            selections_to_fill = set()
        else:
            selections_to_fill = set(fill_selections)
        
        # Set up default colors for systems
        if colors_per_system is None:
            colors_per_system = {}
            default_colors = plt.cm.tab10(np.linspace(0, 1, n_systems))
            for i, sys_name in enumerate(system_names):
                colors_per_system[sys_name] = default_colors[i]
        
        # Collect cluster IDs for each system and validate dist_keys
        system_cluster_map = {}
        for sys_name in system_names:
            sys_data = systems_data[sys_name]
            
            # Get distance data
            if 'distance_data' not in sys_data:
                raise ValueError(f"System '{sys_name}' missing 'distance_data'")
            
            dist_data = sys_data['distance_data']
            
            # Check if all dist_keys exist
            for key in dist_keys:
                if key not in dist_data:
                    raise ValueError(f"Distance key '{key}' not found in system '{sys_name}'. "
                               f"Available: {list(dist_data.keys())}")
            
            # Get cluster IDs from first dist_key
            if cluster_ids is None or cluster_ids == 'all':
                sys_cluster_ids = sorted(dist_data[dist_keys[0]].keys())
            else:
                sys_cluster_ids = cluster_ids
            
            system_cluster_map[sys_name] = sys_cluster_ids
            print(f"  {sys_name}: {len(sys_cluster_ids)} clusters")
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize, facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        
        # Calculate x-positions for each system and cluster
        x_positions = []
        x_labels = []  # Just cluster IDs: C0, C1, C2, etc.
        system_label_positions = []  # Positions to place system names
        system_label_names = []  # System names to display
        current_x = 0
        
        for sys_idx, sys_name in enumerate(system_names):
            sys_cluster_ids = system_cluster_map[sys_name]
            n_clusters = len(sys_cluster_ids)
            
            # Center clusters within each system
            cluster_positions = np.arange(n_clusters) - (n_clusters - 1) / 2
            cluster_x = current_x + cluster_spacing * cluster_positions
            
            # Store positions and labels
            for cidx, cid in enumerate(sys_cluster_ids):
                x_positions.append(cluster_x[cidx])
                x_labels.append(f"C{cid}")  # Only cluster ID
            
            # Store system name position (center of this system's clusters)
            system_center_x = current_x
            system_label_positions.append(system_center_x)
            system_label_names.append(sys_name)
            
            # Move to next system position
            current_x += (n_clusters * cluster_spacing + system_spacing)
        
        x_positions = np.array(x_positions)
        
        # Plot data for each system
        pos_idx = 0
        for sys_idx, sys_name in enumerate(system_names):
            sys_data = systems_data[sys_name]
            sys_cluster_ids = system_cluster_map[sys_name]
            sys_color_spec = colors_per_system[sys_name]
            
            for cluster_idx, cluster_id in enumerate(sys_cluster_ids):
                # Resolve color for this cluster
                # Priority: colors param > colors_per_system
                if colors is not None:
                    # Use global cluster color mapping (same color for cluster X across all systems)
                    if isinstance(colors, dict):
                        cluster_color = colors.get(cluster_id, plt.cm.tab10(cluster_id / 10))
                    else:
                        # List format
                        cluster_color = colors[cluster_id % len(colors)]
                else:
                    # Use system-specific coloring from colors_per_system
                    if isinstance(sys_color_spec, list):
                        # List format: [color1, color2, ...] indexed by cluster_idx
                        cluster_color = sys_color_spec[cluster_idx % len(sys_color_spec)]
                    else:
                        # Single color string
                        cluster_color = sys_color_spec
                
                # Plot each selection overlaid at this cluster position
                for sel_idx, key in enumerate(dist_keys):
                    dist_data = sys_data['distance_data'][key]
                    
                    if cluster_id not in dist_data:
                        if sel_idx == 0:  # Only warn once per cluster
                            print(f"  Warning: Cluster {cluster_id} not found in {sys_name}/{key}, skipping")
                        continue
                    
                    data = dist_data[cluster_id]
                    hist_data = data.get('hist_normalized', data['hist'])
                    if hist_data is None:
                        hist_data = data['hist']
                    distances = data['bin_centers']
                    
                    # Apply xlim filter
                    if xlim is not None:
                        mask = (distances >= xlim[0]) & (distances <= xlim[1])
                        distances = distances[mask]
                        hist_data = hist_data[mask]
                    
                    x = np.full_like(distances, x_positions[pos_idx])
                    
                    # Get linewidth and linestyle for this selection
                    sel_linewidth = line_widths[sel_idx % len(line_widths)]
                    sel_linestyle = line_styles[sel_idx % len(line_styles)]
                    
                    # Label only for legend - show selections when multiple
                    if len(dist_keys) > 1:
                        # Multiple selections: label each selection (shown once)
                        label = f"{key}" if (sys_idx == 0 and cluster_idx == 0) else None
                    else:
                        # Single selection: no label needed (or use system labels)
                        label = None
                    
                    # Plot line
                    ax.plot(x, distances, hist_data,
                           label=label, linewidth=sel_linewidth, linestyle=sel_linestyle,
                           color=cluster_color, alpha=alpha)
                    
                    # Fill under curve only if this selection is in selections_to_fill
                    if fill_under_curve and key in selections_to_fill:
                        x_fill = np.vstack([x, x])
                        y_fill = np.vstack([distances, distances])
                        z_fill = np.vstack([np.zeros_like(hist_data), hist_data])
                        ax.plot_surface(x_fill, y_fill, z_fill,
                                       color=cluster_color, alpha=fill_alpha,
                                       linewidth=0, shade=True)
                
                pos_idx += 1
        
        # Set Y-axis limits based on xlim (distance range)
        if xlim is not None:
            ax.set_ylim(xlim[0], xlim[1])
        
        # Set X-axis properties
        ax.set_xlim(x_positions.min() - 1, x_positions.max() + 1)
        ax.set_xticks(x_positions)
        
        # Hide matplotlib's auto tick labels - we'll place them manually in 3D space
        # so they stay exactly aligned with ticks at any viewing angle
        ax.set_xticklabels(['' for _ in x_labels])
        
        # Labels
        ax.set_xlabel(xlabel, fontsize=label_fontsize, 
                     fontweight=label_fontweight, labelpad=xlabel_pad)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, 
                     fontweight=label_fontweight, labelpad=ylabel_pad)
        ax.set_zlabel(zlabel, fontsize=label_fontsize, 
                     fontweight=label_fontweight, labelpad=zlabel_pad)
        
        # Disable clipping on z-axis label
        ax.zaxis.label.set_clip_on(False)
        
        # Configure ticks
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.tick_params(axis='x', pad=xtick_pad)
        ax.tick_params(axis='y', pad=ytick_pad)
        ax.tick_params(axis='z', pad=ztick_pad)
        
        # Force y-axis (distance) to use integer ticks so large font sizes
        # don't cause matplotlib to pick half-integer ticks with decimals
        from matplotlib.ticker import MaxNLocator, FormatStrFormatter, MultipleLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Pick a z-axis tick step that is always ≥ 0.1 so that the %.1f
        # formatter never produces duplicate labels.  We target ~5-8 ticks
        # and round up to the nearest "nice" multiple.
        _zlim = ax.get_zlim()
        _z_range = _zlim[1] - _zlim[0]
        _raw_step = _z_range / 6.0
        _nice = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        _z_step = next((s for s in _nice if s >= _raw_step), _nice[-1])
        ax.zaxis.set_major_locator(MultipleLocator(_z_step))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Set viewing angle FIRST so the projection matrix is correct when we
        # compute screen-space angles below
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Get axis limits for text positioning
        ylim_range = ax.get_ylim()
        y_range = ylim_range[1] - ylim_range[0]
        
        # Place cluster ID labels (C0, C1, ...) at each tick position in 3D space
        # Slightly offset beyond y_max so they appear just outside the axis
        tick_y_offset = ylim_range[1] + (0.05 * y_range)
        for pos, label in zip(x_positions, x_labels):
            ax.text(pos, tick_y_offset, 0,
                   label,
                   ha='center', va='top',
                   fontsize=tick_fontsize,
                   fontweight='normal')
        
        # Place system name labels as 2D text so rotation actually works.
        # ax.text() creates Text3D which overrides rotation during draw.
        # ax.text2D() creates regular Text where rotation is respected.
        # We project the 3D positions to 2D via the projection matrix.
        sys_y_offset = ylim_range[1] + (system_label_y_offset * y_range)
        
        # Force a draw so the projection matrix (ax.get_proj()) is up to date
        fig.canvas.draw()
        from mpl_toolkits.mplot3d import proj3d as _proj3d
        _M = ax.get_proj()
        
        # Compute x-axis screen angle from projection
        _xlim = ax.get_xlim()
        _p0 = _proj3d.proj_transform(_xlim[0], ylim_range[1], 0, _M)
        _p1 = _proj3d.proj_transform(_xlim[1], ylim_range[1], 0, _M)
        _xaxis_angle = float(np.degrees(np.arctan2(_p1[1] - _p0[1], _p1[0] - _p0[0])))
        # Normalize so text always reads left-to-right (never upside down)
        if _xaxis_angle > 90:
            _xaxis_angle -= 180
        elif _xaxis_angle < -90:
            _xaxis_angle += 180
        
        # Use caller's override if provided, otherwise auto-computed angle
        _sys_rot = system_label_rotation if system_label_rotation is not None else _xaxis_angle
        
        # Configure mathtext to use a Times-like font (STIX) so that superscripts
        # in $CIP^{+}$ match the Times New Roman labels
        _orig_mathtext = plt.rcParams.get('mathtext.fontset', 'dejavusans')
        if system_label_fontfamily and 'times' in system_label_fontfamily.lower():
            plt.rcParams['mathtext.fontset'] = 'stix'
        
        _sys_text_objects = []
        # Build a FontProperties object for reliable bold / font-family
        # resolution.  Individual kwargs (fontweight, fontfamily) can fail
        # to locate the bold variant on macOS; FontProperties uses the full
        # font-matching pipeline which is more robust.
        from matplotlib.font_manager import FontProperties as _FP
        _sys_fp = _FP(
            family=system_label_fontfamily,
            weight=system_label_fontweight,
            size=system_label_fontsize if system_label_fontsize is not None else tick_fontsize,
        )
        for sys_x, sys_name in zip(system_label_positions, system_label_names):
            _display = system_display_names.get(sys_name, sys_name) if system_display_names else sys_name
            # Project 3D label position to 2D
            _px, _py, _ = _proj3d.proj_transform(sys_x, sys_y_offset, 0, _M)
            _t = ax.text2D(_px, _py, _display,
                   transform=ax.transData,
                   ha='center', va='top',
                   fontproperties=_sys_fp,
                   rotation=_sys_rot)
            _sys_text_objects.append(_t)
        
        # Restore original mathtext setting
        plt.rcParams['mathtext.fontset'] = _orig_mathtext
        
        # Title
        if show_title:
            if title is None:
                if len(dist_keys) == 1:
                    title = f"Multi-System 3D Distance Distribution: {dist_keys[0]}"
                else:
                    title = f"Multi-System 3D Distance Distributions: {', '.join(dist_keys)}"
                if xlim is not None:
                    title += f"\\n(Distance: {xlim[0]:.1f}-{xlim[1]:.1f} Å)"
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight, pad=20)
        
        # Set z-axis label rotation
        ax.zaxis.set_rotate_label(False)
        ax.zaxis.label.set_rotation(90)
        
        # (view_init already called above before projection computation)
        
        # ── Legends ───────────────────────────────────────────────────────
        from matplotlib.lines import Line2D

        def _style_legend(leg):
            if leg is None:
                return
            for txt in leg.get_texts():
                txt.set_fontweight(legend_fontweight)
            ttl = leg.get_title()
            if ttl is not None:
                ttl.set_fontsize(legend_title_fontsize)
                ttl.set_fontweight(legend_title_fontweight)

        # --- Legend 1: Cluster colours ---
        cluster_handles = []
        if show_legend:
            all_cids_ordered = []
            for sn in system_names:
                for cid in system_cluster_map[sn]:
                    if cid not in all_cids_ordered:
                        all_cids_ordered.append(cid)
            for cidx, cid in enumerate(all_cids_ordered):
                if colors is not None:
                    if isinstance(colors, dict):
                        c = colors.get(cid, plt.cm.tab10(cid / 10))
                    else:
                        c = colors[cid % len(colors)]
                else:
                    first_sys = system_names[0]
                    sys_color_spec = colors_per_system[first_sys]
                    if isinstance(sys_color_spec, list):
                        c = sys_color_spec[cidx % len(sys_color_spec)]
                    else:
                        c = sys_color_spec
                lbl = (legend_labels[cidx]
                       if legend_labels and cidx < len(legend_labels)
                       else f'C{cid}')
                cluster_handles.append(
                    Line2D([0], [0], color=c, linewidth=linewidth, label=lbl)
                )

        # --- Legend 2: Moiety (linestyle) ---
        moiety_handles = []
        if show_moiety_legend and len(dist_keys) > 1:
            for sel_idx, key in enumerate(dist_keys):
                sel_lw = line_widths[sel_idx % len(line_widths)]
                sel_ls = line_styles[sel_idx % len(line_styles)]
                lbl = (moiety_labels.get(key, key) if moiety_labels else key)
                moiety_handles.append(
                    Line2D([0], [0], color='gray', linewidth=sel_lw,
                           linestyle=sel_ls, label=lbl)
                )

        # Draw both legends
        if cluster_handles:
            _kw1 = dict(handles=cluster_handles,
                        title=legend_title,
                        loc=legend_loc,
                        fontsize=legend_fontsize,
                        framealpha=legend_framealpha,
                        edgecolor=legend_edgecolor,
                        ncol=legend_ncol_cluster,
                        handletextpad=legend_handletextpad)
            if legend1_bbox is not None:
                _kw1['bbox_to_anchor'] = legend1_bbox
            _leg1 = ax.legend(**_kw1)
            _style_legend(_leg1)
            if moiety_handles:
                ax.add_artist(_leg1)   # keep leg1 alive when leg2 is added

        if moiety_handles:
            _kw2 = dict(handles=moiety_handles,
                        title=moiety_legend_title,
                        loc=legend_loc,
                        fontsize=legend_fontsize,
                        framealpha=legend_framealpha,
                        edgecolor=legend_edgecolor,
                        ncol=legend_ncol_moiety,
                        handletextpad=legend_handletextpad)
            if legend2_bbox is not None:
                _kw2['bbox_to_anchor'] = legend2_bbox
            _leg2 = ax.legend(**_kw2)
            _style_legend(_leg2)
        
        # Configure grid and panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        if show_grid:
            ax.xaxis.pane.set_edgecolor('black')
            ax.yaxis.pane.set_edgecolor('black')
            ax.zaxis.pane.set_edgecolor('black')
            ax.xaxis.pane.set_alpha(grid_alpha)
            ax.yaxis.pane.set_alpha(grid_alpha)
            ax.zaxis.pane.set_alpha(grid_alpha)
            ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=grid_linewidth)
        else:
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            ax.xaxis.pane.set_alpha(0)
            ax.yaxis.pane.set_alpha(0)
            ax.zaxis.pane.set_alpha(0)
            ax.grid(False)
        
        fig.patch.set_facecolor('white')
        
        # Save figure
        if save_fig:
            if save_path is None:
                if len(dist_keys) == 1:
                    save_path = f"multi_system_3d_{dist_keys[0]}.png"
                else:
                    save_path = f"multi_system_3d_{'_'.join(dist_keys)}.png"
            
            if bbox_inches == 'tight':
                # Include text2D system labels in bbox computation so they
                # aren't cropped.  save_right_pad adds extra breathing room
                # (applied uniformly via pad_inches; the right side benefits
                # most because that's where the last system label sits).
                _save_pad = pad_inches + save_right_pad
                fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches,
                           bbox_extra_artists=[ax.zaxis.label] + _sys_text_objects,
                           pad_inches=_save_pad)
            else:
                fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
            print(f"\\nSaved: {save_path}")
        
        print(f"\\nPlot created successfully!")
        return fig

    def plot_multi_system_rdfs_3d(self,
                                   systems_data: Dict,
                                   sel_names: Optional[Union[str, List[str]]] = None,
                                   cluster_ids: Optional[Union[str, List[int]]] = None,
                                   figsize: Tuple[float, float] = (18, 12),
                                   colors: Optional[Union[Dict[int, str], List[str]]] = None,
                                   colors_per_system: Optional[Dict[str, Union[str, List[str]]]] = None,
                                   xlim: Optional[Tuple[float, float]] = None,
                                   show_title: bool = True,
                                   title: Optional[str] = None,
                                   title_fontsize: int = 16,
                                   title_fontweight: str = 'bold',
                                   label_fontsize: int = 12,
                                   label_fontweight: str = 'bold',
                                   ylabel: str = 'r (Å)',
                                   zlabel: str = 'g(r)',
                                   tick_fontsize: int = 10,
                                   show_legend: bool = True,
                                   legend_loc: str = 'upper left',
                                   legend_ncol: int = 1,
                                   legend_bbox: Optional[Tuple[float, float]] = None,
                                   legend_fontsize: int = 10,
                                   legend_fontweight: str = 'normal',
                                   legend_framealpha: float = 0.9,
                                   linewidth: float = 2.5,
                                   linewidths: Optional[List[float]] = None,
                                   linestyles: Optional[List] = None,
                                   alpha: float = 0.8,
                                   elevation: float = 20,
                                   azimuth: float = 45,
                                   fill_under_curve: bool = True,
                                   fill_alpha: float = 0.3,
                                   fill_selections: Optional[Union[str, List[str]]] = 'all',
                                   show_bulk_line: bool = False,
                                   bulk_line_color: str = 'gray',
                                   bulk_line_style: str = '--',
                                   bulk_line_width: float = 1.0,
                                   bulk_line_alpha: float = 0.4,
                                   show_grid: bool = True,
                                   grid_alpha: float = 0.2,
                                   grid_linewidth: float = 0.5,
                                   system_spacing: float = 2.0,
                                   cluster_spacing: float = 1.0,
                                   xlabel: str = 'System',
                                   xlabel_pad: float = 15,
                                   ylabel_pad: float = 15,
                                   zlabel_pad: float = 15,
                                   xtick_pad: float = 8,
                                   ytick_pad: float = 8,
                                   ztick_pad: float = 8,
                                   xtick_rotation: float = 0,
                                   system_label_rotation: Optional[float] = None,
                                   system_label_y_offset: float = 0.10,
                                   system_display_names: Optional[Dict[str, str]] = None,
                                   system_label_fontfamily: str = 'Times New Roman',
                                   system_label_fontsize: Optional[int] = None,
                                   system_label_fontweight: str = 'normal',
                                   group_by_system: bool = False,
                                   save_fig: bool = False,
                                   save_path: Optional[str] = None,
                                   dpi: int = 300,
                                   bbox_inches: str = 'tight',
                                   pad_inches: float = 0.1,
                                   save_right_pad: float = 0.3) -> plt.Figure:
        """
        Create a 3D waterfall plot comparing RDF curves across multiple systems.

        Direct RDF equivalent of ``plot_multi_system_distance_distributions_3d``.
        Input comes from ``export_rdf_data_for_multi_system()`` /
        ``RMSDClusterAnalyzer.export_rdf_data_for_multi_system()``.

        Parameters
        ----------
        systems_data : dict
            Built with ``export_rdf_data_for_multi_system()``::

                {system_name: {'system_name', 'cluster_ids',
                               'selection_names', 'rdf_by_cluster'}}

        sel_names : str or list of str, optional
            Which moiety/selection name(s) to plot.  When a list is supplied,
            all selections are **overlaid** at each cluster position and
            differentiated by ``linestyles`` / ``linewidths``.
            If ``None``, uses all selection names from the first system.

        cluster_ids : list, 'all', or None
            Clusters to include.  Default: all clusters in each system.

        colors : dict, optional
            ``{cluster_id: color}`` — same color per cluster across all systems.
            Takes priority over ``colors_per_system``.

        colors_per_system : dict, optional
            ``{system_name: color}`` or ``{system_name: [c0, c1, ...]}``.

        xlim : tuple, optional
            ``(r_min, r_max)`` range shown on the r-axis.

        fill_selections : list, 'all', or None
            Which sel_names to fill under.  Default ``'all'``.
            Example: ``['carboxylic_acid']``.

        show_bulk_line : bool
            Draw g(r) = 1 reference line for each curve.

        Returns
        -------
        fig : matplotlib.figure.Figure

        Examples
        --------
        >>> fig = plotter.plot_multi_system_rdfs_3d(
        ...     systems_rdf_ow,
        ...     sel_names='carboxylic_acid',
        ...     system_display_names={
        ...         'CIP+':   r'$CIP^+$',
        ...         'CIP+/-': r'$CIP^{+/-}$',
        ...         'CIP-':   r'$CIP^-$',
        ...     },
        ...     colors_per_system={
        ...         'CIP+':   ['#FFB6C1'],
        ...         'CIP+/-': ['#FFB6C1', '#ADD8E6'],
        ...         'CIP-':   ['#FFB6C1', '#ADD8E6', '#90EE90', '#FFFFE0', '#E6E6FA'],
        ...     },
        ...     fill_under_curve=True,
        ...     fill_alpha=0.3,
        ...     fill_selections=['carboxylic_acid'],
        ...     system_spacing=3.0,
        ...     cluster_spacing=1.5,
        ...     elevation=20,
        ...     azimuth=45,
        ...     save_fig=True,
        ...     save_path='multi_system_rdf_3d_carboxylic_acid.png',
        ...     dpi=600,
        ... )
        """
        if not systems_data:
            raise ValueError("systems_data cannot be empty")

        system_names = list(systems_data.keys())
        n_systems = len(system_names)

        # ── resolve sel_names ──────────────────────────────────────────────
        if sel_names is None:
            sel_names_list = systems_data[system_names[0]].get('selection_names', [])
        elif isinstance(sel_names, str):
            sel_names_list = [sel_names]
        else:
            sel_names_list = list(sel_names)

        print(f"\\nCreating multi-system 3D RDF plot")
        print(f"Systems: {system_names}")
        print(f"Selections: {sel_names_list}")

        # ── linewidths / linestyles per selection ──────────────────────────
        if linewidths is None:
            line_widths = [linewidth] * len(sel_names_list)
        else:
            line_widths = linewidths

        if linestyles is None:
            line_styles = ['-'] * len(sel_names_list)
        else:
            line_styles = linestyles

        # ── fill set ───────────────────────────────────────────────────────
        if fill_selections == 'all':
            selections_to_fill = set(sel_names_list)
        elif fill_selections is None:
            selections_to_fill = set()
        else:
            selections_to_fill = set(fill_selections)

        # ── default colors ─────────────────────────────────────────────────
        if colors_per_system is None:
            colors_per_system = {}
            _def_colors = plt.cm.tab10(np.linspace(0, 1, n_systems))
            for i, sn in enumerate(system_names):
                colors_per_system[sn] = _def_colors[i]

        # ── gather cluster IDs per system ──────────────────────────────────
        system_cluster_map = {}
        for sn in system_names:
            sd = systems_data[sn]
            if cluster_ids is None or cluster_ids == 'all':
                sys_cids = sorted(sd['cluster_ids'])
            else:
                sys_cids = list(cluster_ids)
            system_cluster_map[sn] = sys_cids
            print(f"  {sn}: {len(sys_cids)} clusters")

        # ── build figure ───────────────────────────────────────────────────
        fig = plt.figure(figsize=figsize, facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')

        # ── x-positions for each (system, cluster) pair ───────────────────
        x_positions = []      # one entry per (system, cluster) pair
        x_labels = []
        system_label_positions = []
        system_label_names_list = []
        # system_x_map: system_name → x position used for its curves
        system_x_map: Dict[str, float] = {}
        current_x = 0

        for sn in system_names:
            sys_cids = system_cluster_map[sn]
            n_c = len(sys_cids)

            if group_by_system:
                # All clusters share the same x position
                sys_x = float(current_x)
                system_x_map[sn] = sys_x
                for cid in sys_cids:
                    x_positions.append(sys_x)
                    x_labels.append(f"C{cid}")
                system_label_positions.append(sys_x)
                system_label_names_list.append(sn)
                current_x += system_spacing
            else:
                offsets = np.arange(n_c) - (n_c - 1) / 2
                cluster_xs = current_x + cluster_spacing * offsets
                for cidx, cid in enumerate(sys_cids):
                    x_positions.append(cluster_xs[cidx])
                    x_labels.append(f"C{cid}")
                system_label_positions.append(current_x)
                system_label_names_list.append(sn)
                current_x += n_c * cluster_spacing + system_spacing

        x_positions = np.array(x_positions)

        # ── plot curves ───────────────────────────────────────────────────
        pos_idx = 0
        for sn in system_names:
            sd = systems_data[sn]
            rdf_by_cluster = sd['rdf_by_cluster']
            sys_cids = system_cluster_map[sn]
            sys_color_spec = colors_per_system[sn]

            for cidx, cid in enumerate(sys_cids):
                # resolve color
                if colors is not None:
                    if isinstance(colors, dict):
                        cluster_color = colors.get(cid, plt.cm.tab10(cid / 10))
                    else:
                        cluster_color = colors[cid % len(colors)]
                else:
                    if isinstance(sys_color_spec, list):
                        cluster_color = sys_color_spec[cidx % len(sys_color_spec)]
                    else:
                        cluster_color = sys_color_spec

                # x coordinate: shared system position or per-cluster position
                curve_x = system_x_map[sn] if group_by_system else x_positions[pos_idx]

                for sel_idx, sel in enumerate(sel_names_list):
                    cid_data = rdf_by_cluster.get(cid, {})
                    if sel not in cid_data:
                        continue
                    rdf_entry = cid_data[sel]
                    r_vals = np.asarray(rdf_entry['r'])
                    g_vals = np.asarray(rdf_entry['rdf'])

                    if xlim is not None:
                        mask = (r_vals >= xlim[0]) & (r_vals <= xlim[1])
                        r_vals = r_vals[mask]
                        g_vals = g_vals[mask]

                    x_curve = np.full_like(r_vals, curve_x)
                    sel_lw = line_widths[sel_idx % len(line_widths)]
                    sel_ls = line_styles[sel_idx % len(line_styles)]

                    ax.plot(x_curve, r_vals, g_vals,
                            linewidth=sel_lw, linestyle=sel_ls,
                            color=cluster_color, alpha=alpha)

                    if fill_under_curve and sel in selections_to_fill:
                        xf = np.vstack([x_curve, x_curve])
                        yf = np.vstack([r_vals, r_vals])
                        zf = np.vstack([np.zeros_like(g_vals), g_vals])
                        ax.plot_surface(xf, yf, zf,
                                        color=cluster_color, alpha=fill_alpha,
                                        linewidth=0, shade=True)

                    if show_bulk_line:
                        ax.plot(x_curve, r_vals, np.ones_like(g_vals),
                                color=bulk_line_color, linestyle=bulk_line_style,
                                linewidth=bulk_line_width, alpha=bulk_line_alpha)

                pos_idx += 1

        # ── axes ──────────────────────────────────────────────────────────
        if xlim is not None:
            ax.set_ylim(xlim[0], xlim[1])
        ax.set_xlim(x_positions.min() - 1, x_positions.max() + 1)
        _xtick_positions = np.array(system_label_positions) if group_by_system else x_positions
        ax.set_xticks(_xtick_positions)
        ax.set_xticklabels(['' for _ in _xtick_positions])

        ax.set_xlabel(xlabel, fontsize=label_fontsize,
                      fontweight=label_fontweight, labelpad=xlabel_pad)
        ax.set_ylabel(ylabel, fontsize=label_fontsize,
                      fontweight=label_fontweight, labelpad=ylabel_pad)
        ax.set_zlabel(zlabel, fontsize=label_fontsize,
                      fontweight=label_fontweight, labelpad=zlabel_pad)
        ax.zaxis.label.set_clip_on(False)

        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.tick_params(axis='x', pad=xtick_pad)
        ax.tick_params(axis='y', pad=ytick_pad)
        ax.tick_params(axis='z', pad=ztick_pad)

        from matplotlib.ticker import MaxNLocator, FormatStrFormatter, MultipleLocator
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        _zlim = ax.get_zlim()
        _z_range = _zlim[1] - _zlim[0]
        _raw_step = _z_range / 6.0
        _nice = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        _z_step = next((s for s in _nice if s >= _raw_step), _nice[-1])
        ax.zaxis.set_major_locator(MultipleLocator(_z_step))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.view_init(elev=elevation, azim=azimuth)

        # ── cluster tick labels (skipped in group_by_system mode) ──────────
        ylim_range = ax.get_ylim()
        y_range = ylim_range[1] - ylim_range[0]
        if not group_by_system:
            tick_y_offset = ylim_range[1] + 0.05 * y_range
            for pos, lbl in zip(x_positions, x_labels):
                ax.text(pos, tick_y_offset, 0, lbl,
                        ha='center', va='top', fontsize=tick_fontsize)

        # ── system labels (2D, rotation-safe) ─────────────────────────────
        sys_y_offset = ylim_range[1] + system_label_y_offset * y_range
        fig.canvas.draw()
        from mpl_toolkits.mplot3d import proj3d as _proj3d
        _M = ax.get_proj()
        _xlim2 = ax.get_xlim()
        _p0 = _proj3d.proj_transform(_xlim2[0], ylim_range[1], 0, _M)
        _p1 = _proj3d.proj_transform(_xlim2[1], ylim_range[1], 0, _M)
        _xaxis_angle = float(np.degrees(np.arctan2(_p1[1] - _p0[1], _p1[0] - _p0[0])))
        if _xaxis_angle > 90:
            _xaxis_angle -= 180
        elif _xaxis_angle < -90:
            _xaxis_angle += 180
        _sys_rot = system_label_rotation if system_label_rotation is not None else _xaxis_angle

        _orig_mathtext = plt.rcParams.get('mathtext.fontset', 'dejavusans')
        if system_label_fontfamily and 'times' in system_label_fontfamily.lower():
            plt.rcParams['mathtext.fontset'] = 'stix'

        from matplotlib.font_manager import FontProperties as _FP
        _sys_fp = _FP(family=system_label_fontfamily,
                      weight=system_label_fontweight,
                      size=system_label_fontsize if system_label_fontsize is not None else tick_fontsize)

        _sys_text_objects = []
        for sys_x, sn in zip(system_label_positions, system_label_names_list):
            _display = system_display_names.get(sn, sn) if system_display_names else sn
            _px, _py, _ = _proj3d.proj_transform(sys_x, sys_y_offset, 0, _M)
            _t = ax.text2D(_px, _py, _display,
                           transform=ax.transData,
                           ha='center', va='top',
                           fontproperties=_sys_fp,
                           rotation=_sys_rot)
            _sys_text_objects.append(_t)

        plt.rcParams['mathtext.fontset'] = _orig_mathtext

        # ── title ─────────────────────────────────────────────────────────
        if show_title:
            if title is None:
                title = f"Multi-System 3D RDF: {', '.join(sel_names_list)}"
            ax.set_title(title, fontsize=title_fontsize,
                         fontweight=title_fontweight, pad=20)

        ax.zaxis.set_rotate_label(False)
        ax.zaxis.label.set_rotation(90)

        # ── legend ────────────────────────────────────────────────────────
        if show_legend:
            from matplotlib.lines import Line2D
            legend_elements = []
            if len(sel_names_list) > 1:
                for sel_idx, sel in enumerate(sel_names_list):
                    legend_elements.append(
                        Line2D([0], [0], color='gray',
                               linestyle=line_styles[sel_idx % len(line_styles)],
                               linewidth=line_widths[sel_idx % len(line_widths)],
                               label=sel))
            else:
                for sn in system_names:
                    sys_color_spec = colors_per_system[sn]
                    sys_cids = system_cluster_map[sn]
                    if isinstance(sys_color_spec, list):
                        for cidx, cid in enumerate(sys_cids):
                            cc = sys_color_spec[cidx % len(sys_color_spec)]
                            legend_elements.append(
                                Line2D([0], [0], color=cc, linewidth=linewidth,
                                       label=f"{sn} C{cid}"))
                    else:
                        legend_elements.append(
                            Line2D([0], [0], color=sys_color_spec,
                                   linewidth=linewidth, label=sn))
            if legend_elements:
                kw = dict(ncol=legend_ncol, framealpha=legend_framealpha,
                          prop={'size': legend_fontsize, 'weight': legend_fontweight})
                if legend_bbox is not None:
                    ax.legend(handles=legend_elements, loc=legend_loc,
                              bbox_to_anchor=legend_bbox, **kw)
                else:
                    ax.legend(handles=legend_elements, loc=legend_loc, **kw)

        # ── grid & panes ──────────────────────────────────────────────────
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        if show_grid:
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.set_edgecolor('black')
                pane.set_alpha(grid_alpha)
            ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=grid_linewidth)
        else:
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.set_edgecolor('none')
                pane.set_alpha(0)
            ax.grid(False)

        fig.patch.set_facecolor('white')

        # ── save ──────────────────────────────────────────────────────────
        if save_fig:
            if save_path is None:
                save_path = f"multi_system_rdf_3d_{'_'.join(sel_names_list)}.png"
            if bbox_inches == 'tight':
                _save_pad = pad_inches + save_right_pad
                fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches,
                            bbox_extra_artists=[ax.zaxis.label] + _sys_text_objects,
                            pad_inches=_save_pad)
            else:
                fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
            print(f"\\nSaved: {save_path}")

        print(f"\\nPlot created successfully!")
        return fig

    def plot_cluster_free_energies(self,
                                   fe_data: Dict,
                                   figsize: Tuple[float, float] = (10, 6),
                                   colors: Optional[Union[Dict[int, str], List[str]]] = None,
                                   show_title: bool = True,
                                   title: Optional[str] = None,
                                   title_fontsize: int = 14,
                                   title_fontweight: str = 'bold',
                                   xlabel: str = 'Cluster ID',
                                   ylabel: Optional[str] = None,
                                   label_fontsize: int = 12,
                                   label_fontweight: str = 'bold',
                                   tick_fontsize: int = 10,
                                   bar_alpha: float = 0.75,
                                   edgecolor: str = 'black',
                                   edgewidth: float = 1.5,
                                   show_errorbar: bool = True,
                                   errorbar_capsize: float = 5,
                                   errorbar_capthick: float = 1.5,
                                   show_values: bool = True,
                                   value_fontsize: int = 10,
                                   value_fontweight: str = 'bold',
                                   value_format: str = '{:.2f}',
                                   show_population: bool = True,
                                   pop_fontsize: int = 9,
                                   grid: bool = True,
                                   grid_alpha: float = 0.3,
                                   grid_linestyle: str = '--',
                                   save_path: Optional[str] = None,
                                   dpi: int = 300,
                                   bbox_inches: str = 'tight') -> plt.Figure:
        """
        Plot relative free energies between clusters as bar chart.
        
        Parameters
        ----------
        fe_data : dict
            Free energy data from compute_cluster_free_energies()
        figsize : tuple, default=(10, 6)
            Figure size
        colors : dict or list, optional
            Colors for each cluster (auto-generated if None)
        
        Title and labels:
        show_title : bool, default=True
            Show title
        title : str, optional
            Custom title (auto-generated if None)
        xlabel, ylabel : str
            Axis labels (ylabel auto-generated from units if None)
        
        Bar styling:
        bar_alpha : float, default=0.75
            Bar transparency
        edgecolor : str, default='black'
            Bar edge color
        edgewidth : float, default=1.5
            Bar edge width
        
        Error bars:
        show_errorbar : bool, default=True
            Show error bars
        errorbar_capsize : float, default=5
            Error bar cap size
        errorbar_capthick : float, default=1.5
            Error bar cap thickness
        
        Value labels:
        show_values : bool, default=True
            Show ΔG values on bars
        value_fontsize : int, default=10
            Value label font size
        value_format : str, default='{:.2f}'
            Format string for values
        show_population : bool, default=True
            Show population fractions below bars
        pop_fontsize : int, default=9
            Population label font size
        
        save_path : str, optional
            Path to save figure
        dpi : int, default=300
            Resolution for saved figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        
        Example
        -------
        >>> fe_data = analyzer.compute_cluster_free_energies(temperature=300)
        >>> fig = plotter.plot_cluster_free_energies(
        ...     fe_data,
        ...     colors={0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple'},
        ...     save_path='free_energies.png'
        ... )
        """
        cluster_ids = sorted(fe_data.keys())
        n_clusters = len(cluster_ids)
        
        # Extract data
        delta_Gs = [fe_data[cid]['delta_G'] for cid in cluster_ids]
        errors = [fe_data[cid]['std_error'] for cid in cluster_ids]
        populations = [fe_data[cid]['population'] for cid in cluster_ids]
        is_reference = [fe_data[cid]['is_reference'] for cid in cluster_ids]
        
        # Determine units from first entry
        sample_data = fe_data[cluster_ids[0]]
        # Try to infer units from stored data if available
        units = None
        if hasattr(self.analyzer, 'free_energy_data'):
            for key, stored in self.analyzer.free_energy_data.items():
                if stored['results'] == fe_data:
                    units = stored['units']
                    break
        if units is None:
            units = 'kJ/mol'  # Default
        
        # Handle colors
        if colors is None:
            colors_list = plt.cm.viridis(np.linspace(0.2, 0.9, n_clusters))
        elif isinstance(colors, dict):
            colors_list = [colors.get(cid, plt.cm.viridis(i/n_clusters)) 
                          for i, cid in enumerate(cluster_ids)]
        else:
            colors_list = colors
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        x_pos = np.arange(n_clusters)
        
        # Plot bars
        bars = ax.bar(x_pos, delta_Gs, color=colors_list, alpha=bar_alpha,
                     edgecolor=edgecolor, linewidth=edgewidth)
        
        # Highlight reference cluster
        for i, is_ref in enumerate(is_reference):
            if is_ref:
                bars[i].set_edgecolor('red')
                bars[i].set_linewidth(edgewidth * 1.5)
        
        # Add error bars
        if show_errorbar:
            ax.errorbar(x_pos, delta_Gs, yerr=errors, fmt='none',
                       ecolor='black', capsize=errorbar_capsize,
                       capthick=errorbar_capthick, elinewidth=1.5,
                       alpha=0.8, zorder=10)
        
        # Add value labels on bars
        if show_values:
            for i, (dg, err) in enumerate(zip(delta_Gs, errors)):
                y_pos = dg + (err if dg >= 0 else -err)
                offset = 0.5 if dg >= 0 else -0.5
                ax.text(i, y_pos + offset, value_format.format(dg),
                       ha='center', va='bottom' if dg >= 0 else 'top',
                       fontsize=value_fontsize, fontweight=value_fontweight)
        
        # Add population labels below x-axis
        if show_population:
            y_min = min(delta_Gs) - max(errors) - 1
            for i, (pop, n_frames) in enumerate(zip(populations, 
                                                    [fe_data[cid]['n_frames'] for cid in cluster_ids])):
                ax.text(i, y_min, f'p={pop:.3f}\n(N={n_frames})',
                       ha='center', va='top', fontsize=pop_fontsize,
                       color='gray')
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        if ylabel is None:
            ylabel = f'ΔG ({units})'
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Cluster {cid}' for cid in cluster_ids])
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        # Add reference line at ΔG = 0
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
        
        if grid:
            ax.grid(alpha=grid_alpha, linestyle=grid_linestyle, axis='y', zorder=0)
        
        # Title
        if show_title:
            if title is None:
                ref_id = [cid for cid, is_ref in zip(cluster_ids, is_reference) if is_ref][0]
                title = f'Relative Free Energies (ref: Cluster {ref_id})'
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_free_energy_landscape(self,
                                   fe_data: Dict,
                                   figsize: Tuple[float, float] = (10, 8),
                                   colors: Optional[Union[Dict[int, str], List[str]]] = None,
                                   show_title: bool = True,
                                   title: str = 'Free Energy Landscape',
                                   title_fontsize: int = 14,
                                   title_fontweight: str = 'bold',
                                   xlabel: str = 'PC1',
                                   ylabel: str = 'PC2',
                                   label_fontsize: int = 12,
                                   label_fontweight: str = 'bold',
                                   tick_fontsize: int = 10,
                                   marker_size: float = 300,
                                   marker_alpha: float = 0.7,
                                   edgecolor: str = 'black',
                                   edgewidth: float = 2,
                                   show_labels: bool = True,
                                   label_fontsize_markers: int = 11,
                                   label_fontweight_markers: str = 'bold',
                                   show_colorbar: bool = True,
                                   cbar_label: Optional[str] = None,
                                   cbar_fontsize: int = 11,
                                   cmap: str = 'RdYlGn_r',
                                   grid: bool = True,
                                   grid_alpha: float = 0.3,
                                   grid_linestyle: str = '--',
                                   save_path: Optional[str] = None,
                                   dpi: int = 300,
                                   bbox_inches: str = 'tight') -> plt.Figure:
        """
        Plot 2D free energy landscape on principal component space.
        
        Projects cluster centers onto PC1-PC2 plane and colors by free energy.
        
        Parameters
        ----------
        fe_data : dict
            Free energy data from compute_cluster_free_energies()
        figsize : tuple, default=(10, 8)
            Figure size
        colors : dict or list, optional
            Marker colors (overrides colormap if provided)
        
        Title and labels:
        show_title : bool, default=True
            Show title
        title : str, default='Free Energy Landscape'
            Figure title
        xlabel, ylabel : str
            Axis labels (default: PC1, PC2)
        
        Marker styling:
        marker_size : float, default=300
            Size of cluster markers (scaled by population)
        marker_alpha : float, default=0.7
            Marker transparency
        edgecolor : str, default='black'
            Marker edge color
        edgewidth : float, default=2
            Marker edge width
        
        Labels:
        show_labels : bool, default=True
            Show cluster ID labels on markers
        label_fontsize_markers : int, default=11
            Marker label font size
        
        Colorbar:
        show_colorbar : bool, default=True
            Show free energy colorbar
        cbar_label : str, optional
            Colorbar label (auto-generated if None)
        cmap : str, default='RdYlGn_r'
            Colormap (reversed: green=stable, red=unstable)
        
        save_path : str, optional
            Path to save figure
        dpi : int, default=300
            Resolution for saved figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        
        Example
        -------
        >>> fe_data = analyzer.compute_cluster_free_energies(temperature=300)
        >>> fig = plotter.plot_free_energy_landscape(
        ...     fe_data,
        ...     cmap='coolwarm',
        ...     save_path='pmf_landscape.png'
        ... )
        """
        if not hasattr(self.analyzer, 'pca_components'):
            raise ValueError("PCA not computed. Run perform_pca() first.")
        
        cluster_ids = sorted(fe_data.keys())
        n_clusters = len(cluster_ids)
        
        # Extract free energies
        delta_Gs = np.array([fe_data[cid]['delta_G'] for cid in cluster_ids])
        populations = np.array([fe_data[cid]['population'] for cid in cluster_ids])
        
        # Get cluster centers in PC space (first 2 components)
        pc_centers = self.analyzer.pca_components[[self.analyzer.cluster_centers[cid] 
                                                   for cid in cluster_ids], :2]
        
        # Determine units
        units = 'kJ/mol'
        if hasattr(self.analyzer, 'free_energy_data'):
            for key, stored in self.analyzer.free_energy_data.items():
                if stored['results'] == fe_data:
                    units = stored['units']
                    break
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Handle colors
        if colors is None:
            # Use colormap based on free energy
            scatter = ax.scatter(pc_centers[:, 0], pc_centers[:, 1],
                               c=delta_Gs, s=marker_size * populations / populations.max(),
                               alpha=marker_alpha, edgecolors=edgecolor,
                               linewidths=edgewidth, cmap=cmap, zorder=5)
            
            if show_colorbar:
                cbar = plt.colorbar(scatter, ax=ax)
                if cbar_label is None:
                    cbar_label = f'ΔG ({units})'
                cbar.set_label(cbar_label, fontsize=cbar_fontsize, fontweight='bold')
                cbar.ax.tick_params(labelsize=tick_fontsize)
        else:
            # Use provided colors
            if isinstance(colors, dict):
                colors_list = [colors.get(cid, 'gray') for cid in cluster_ids]
            else:
                colors_list = colors
            
            for i, cid in enumerate(cluster_ids):
                ax.scatter(pc_centers[i, 0], pc_centers[i, 1],
                          s=marker_size * populations[i] / populations.max(),
                          color=colors_list[i], alpha=marker_alpha,
                          edgecolors=edgecolor, linewidths=edgewidth,
                          zorder=5)
        
        # Add cluster labels
        if show_labels:
            for i, cid in enumerate(cluster_ids):
                ax.text(pc_centers[i, 0], pc_centers[i, 1], f'{cid}',
                       ha='center', va='center', fontsize=label_fontsize_markers,
                       fontweight=label_fontweight_markers, color='white',
                       zorder=10)
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        if grid:
            ax.grid(alpha=grid_alpha, linestyle=grid_linestyle, zorder=0)
        
        if show_title:
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_free_energy_barplot(self,
                                 fe_data: Dict,
                                 cluster_ids: Optional[List[int]] = None,
                                 # Colors and styling
                                 colors: Optional[List[str]] = None,
                                 colormap: str = 'RdYlGn_r',
                                 # Bar styling
                                 bar_width: float = 0.6,
                                 bar_spacing: float = 1.0,
                                 bar_alpha: float = 0.7,
                                 edgecolor: str = 'black',
                                 edgewidth: float = 1.5,
                                 # Figure layout
                                 figsize: Tuple[float, float] = (10, 6),
                                 # Font controls
                                 title_fontsize: int = 14,
                                 title_fontweight: str = 'bold',
                                 label_fontsize: int = 12,
                                 label_fontweight: str = 'bold',
                                 tick_fontsize: int = 11,
                                 # Grid
                                 show_grid: bool = True,
                                 grid_alpha: float = 0.3,
                                 grid_style: str = '--',
                                 grid_width: float = 0.7,
                                 # Reference line
                                 show_reference_line: bool = True,
                                 reference_line_color: str = 'black',
                                 reference_line_style: str = '--',
                                 reference_line_width: float = 1.0,
                                 reference_line_alpha: float = 0.5,
                                 # Axis labels and title
                                 xlabel: Optional[str] = None,
                                 ylabel: Optional[str] = None,
                                 show_title: bool = True,
                                 title: Optional[str] = None,
                                 # Error bars
                                 show_error_bars: bool = True,
                                 error_bar_capsize: float = 5,
                                 error_bar_capthick: float = 1.5,
                                 error_bar_linewidth: float = 1.5,
                                 # X-axis labels
                                 show_population: bool = True,
                                 population_format: str = '.1%',
                                 xlabel_separator: str = '\n',
                                 # Legend
                                 show_legend: bool = False,
                                 legend_labels: Optional[List[str]] = None,
                                 legend_exclude_noise: bool = True,
                                 legend_title: Optional[str] = None,
                                 legend_loc: str = 'best',
                                 legend_bbox: Optional[Tuple[float, float]] = None,
                                 legend1_bbox: Optional[Tuple[float, float]] = None,  # (x, y) for primary legend
                                 legend2_bbox: Optional[Tuple[float, float]] = None,  # (x, y) Reserved for future use
                                 legend_ncol: int = 1,
                                 legend_fontsize: int = 10,
                                 legend_fontweight: str = 'normal',
                                 legend_title_fontsize: int = 11,
                                 legend_title_fontweight: str = 'bold',
                                 legend_frame_alpha: float = 0.9,
                                 legend_edgecolor: str = 'black',
                                 legend_fancybox: bool = True,
                                 legend_shadow: bool = False,
                                 # Export
                                 save_path: Optional[str] = None,
                                 save_fig: bool = True,
                                 dpi: int = 300,
                                 bbox_inches: str = 'tight',
                                 transparent_bg: bool = False) -> plt.Figure:
        """
        Create bar plot of cluster free energies with full publication controls.
        
        Plots ΔG values for each cluster with error bars, colored by custom colors
        or colormap. Bar width is fixed to maintain consistency across different
        numbers of clusters.
        
        Parameters
        ----------
        fe_data : dict
            Output from compute_cluster_free_energies()
            Format: {cluster_id: {'delta_G': float, 'std_error': float, 'population': float}}
        cluster_ids : list of int, optional
            Specific clusters to plot. If None, plots all clusters in fe_data.
        
        Colors and styling
        ------------------
        colors : list of str, optional
            Custom colors for bars. If None, uses colormap.
            Example: ['#F08080', '#ADD8E6', '#90EE90', '#FFFFE0', '#E6E6FA', '#FFB6C1']
        colormap : str, default='RdYlGn_r'
            Matplotlib colormap if colors not provided (reversed: green=low, red=high)
        
        Bar styling
        -----------
        bar_width : float, default=0.6
            Fixed width of bars (prevents auto-expansion with fewer clusters)
        bar_spacing : float, default=1.0
            Spacing multiplier between bar centers. Controls gap between bars while
            maintaining constant visual bar width. Default 1.0 places bars at integer
            positions (0, 1, 2, ...). Values > 1.0 increase spacing (e.g., 1.5 gives
            wider gaps), values < 1.0 decrease spacing (tighter gaps). X-axis limits
            auto-adjust to keep bars centered and maintain constant visual appearance
            regardless of spacing value. Figure size remains constant.
        bar_alpha : float, default=0.7
            Transparency of bars (0=transparent, 1=opaque)
        edgecolor : str, default='black'
            Color of bar edges
        edgewidth : float, default=1.5
            Width of bar edges
        
        Figure layout
        -------------
        figsize : tuple, default=(10, 6)
            Figure size (width, height) in inches
        
        Font controls
        -------------
        title_fontsize : int, default=14
            Title font size
        title_fontweight : str, default='bold'
            Title font weight ('normal', 'bold', 'heavy', 'light', etc.)
        label_fontsize : int, default=12
            Axis label font size
        label_fontweight : str, default='bold'
            Axis label font weight
        tick_fontsize : int, default=11
            Tick label font size
        
        Grid
        ----
        show_grid : bool, default=True
            Show grid lines
        grid_alpha : float, default=0.3
            Grid transparency
        grid_style : str, default='--'
            Grid line style ('-', '--', '-.', ':')
        grid_width : float, default=0.7
            Grid line width
        
        Reference line
        --------------
        show_reference_line : bool, default=True
            Show horizontal line at ΔG=0
        reference_line_color : str, default='black'
            Reference line color
        reference_line_style : str, default='--'
            Reference line style
        reference_line_width : float, default=1.0
            Reference line width
        reference_line_alpha : float, default=0.5
            Reference line transparency
        
        Labels and titles
        -----------------
        xlabel : str, optional
            X-axis label. Default: 'Cluster (population)'
        ylabel : str, optional
            Y-axis label. Auto-detected from units if None.
        show_title : bool, default=True
            Show plot title
        title : str, optional
            Plot title. Default: 'Free Energy Landscape'
        
        Error bars
        ----------
        show_error_bars : bool, default=True
            Show error bars (bootstrap standard errors)
        error_bar_capsize : float, default=5
            Width of error bar caps
        error_bar_capthick : float, default=1.5
            Thickness of error bar caps
        error_bar_linewidth : float, default=1.5
            Width of error bar lines
        
        X-axis labels
        -------------
        show_population : bool, default=True
            Show population percentage below cluster ID
        population_format : str, default='.1%'
            Format string for population (e.g., '.1%' gives '25.3%')
        xlabel_separator : str, default='\\n'
            Separator between cluster ID and population (e.g., '\\n' or ' ')
        
        Legend
        ------
        show_legend : bool, default=False
            Show legend with cluster labels and colors
        legend_labels : list of str, optional
            Custom labels for legend entries. If None, uses cluster IDs.
        legend_exclude_noise : bool, default=True
            Exclude cluster -1 (noise) from legend even if present in data.
            Only applies when legend_labels is None (auto-generated labels).
        legend_title : str, optional
            Title for the legend
        legend_loc : str, default='best'
            Legend location ('best', 'upper right', 'upper left', 'lower left',
            'lower right', 'right', 'center left', 'center right', 'lower center',
            'upper center', 'center')
        legend_bbox : tuple of (float, float), optional
            DEPRECATED: Use legend1_bbox instead. Kept for backward compatibility.
            Legend position in bbox_to_anchor format (x, y). Overrides legend_loc.
            Example: (1.05, 1) places legend outside plot area
        legend1_bbox : tuple of (float, float), optional
            Primary legend position in bbox_to_anchor format (x, y). Overrides legend_bbox.
            Example: (0.02, 0.98) places legend at top-left
        legend2_bbox : tuple of (float, float), optional
            Reserved for future use (API consistency with other methods).
            Currently not used in this single-legend plot.
        legend_ncol : int, default=1
            Number of columns in legend
        legend_fontsize : int, default=10
            Legend text font size
        legend_fontweight : str, default='normal'
            Legend text font weight
        legend_title_fontsize : int, default=11
            Legend title font size
        legend_title_fontweight : str, default='bold'
            Legend title font weight
        legend_frame_alpha : float, default=0.9
            Legend box transparency (0=transparent, 1=opaque)
        legend_edgecolor : str, default='black'
            Legend box edge color
        legend_fancybox : bool, default=True
            Use rounded corners for legend box
        legend_shadow : bool, default=False
            Add shadow to legend box
        
        Export
        ------
        save_path : str, optional
            Path to save figure
        save_fig : bool, default=False
            Whether to save figure (requires save_path)
        dpi : int, default=300
            Resolution for saved figure
        bbox_inches : str, default='tight'
            Bounding box setting for saved figure
        transparent_bg : bool, default=False
            Use transparent background
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object
        
        Examples
        --------
        Basic usage with custom colors:
        >>> fe_data = analyzer.compute_cluster_free_energies(temperature=300)
        >>> fig = plotter.plot_free_energy_barplot(
        ...     fe_data,
        ...     colors=['#F08080', '#ADD8E6', '#90EE90', '#FFFFE0', '#E6E6FA', '#FFB6C1']
        ... )
        
        Publication-quality with fine control:
        >>> fig = plotter.plot_free_energy_barplot(
        ...     fe_data,
        ...     colors=['#F08080', '#ADD8E6', '#90EE90', '#FFFFE0', '#E6E6FA', '#FFB6C1'],
        ...     bar_width=0.5,
        ...     edgewidth=2.0,
        ...     title='API Conformer Free Energies',
        ...     ylabel='Relative Free Energy (kJ/mol)',
        ...     label_fontsize=14,
        ...     save_path='free_energy_publication.png',
        ...     dpi=600
        ... )
        
        Filter specific clusters:
        >>> fig = plotter.plot_free_energy_barplot(
        ...     fe_data,
        ...     cluster_ids=[0, 1, 2, 3],  # Exclude noise cluster
        ...     colors=['#F08080', '#ADD8E6', '#90EE90', '#FFFFE0']
        ... )
        """
        # Handle cluster IDs
        if cluster_ids is None:
            cluster_ids = sorted(fe_data.keys())
        else:
            cluster_ids = sorted(cluster_ids)
        
        # Extract data
        delta_Gs = [fe_data[c]['delta_G'] for c in cluster_ids]
        errors = [fe_data[c]['std_error'] for c in cluster_ids]
        populations = [fe_data[c]['population'] for c in cluster_ids]
        
        # Determine units
        units = 'kJ/mol'
        if hasattr(self.analyzer, 'free_energy_data'):
            for key, stored in self.analyzer.free_energy_data.items():
                if stored['results'] == fe_data:
                    units = stored['units']
                    break
        
        # Handle colors
        if colors is None:
            # Use colormap
            cmap = plt.cm.get_cmap(colormap)
            # Normalize by energy values
            if len(delta_Gs) > 1:
                norm_values = np.array(delta_Gs) / max(delta_Gs)
                bar_colors = [cmap(v) for v in norm_values]
            else:
                bar_colors = [cmap(0.5)]
        else:
            # Use custom colors (cycle if more clusters than colors)
            bar_colors = [colors[i % len(colors)] for i in range(len(cluster_ids))]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bars with custom spacing
        x_positions = np.arange(len(cluster_ids)) * bar_spacing
        bars = ax.bar(x_positions, delta_Gs,
                      width=bar_width,
                      yerr=errors if show_error_bars else None,
                      capsize=error_bar_capsize if show_error_bars else 0,
                      alpha=bar_alpha,
                      edgecolor=edgecolor,
                      linewidth=edgewidth,
                      color=bar_colors,
                      error_kw={'capthick': error_bar_capthick, 
                               'elinewidth': error_bar_linewidth} if show_error_bars else None)
        
        # X-axis ticks and labels
        ax.set_xticks(x_positions)
        if show_population:
            xlabels = [f'{c}{xlabel_separator}({p:{population_format}})' 
                      for c, p in zip(cluster_ids, populations)]
        else:
            xlabels = [str(c) for c in cluster_ids]
        ax.set_xticklabels(xlabels, fontsize=tick_fontsize)
        
        # Dynamically adjust x-axis limits to maintain constant visual bar width
        # regardless of bar_spacing value (keeps figsize constant, adjusts margins)
        if len(cluster_ids) > 1:
            # Calculate total span of bars
            total_span = (len(cluster_ids) - 1) * bar_spacing
            
            # Add proportional margins to keep bars centered
            # Smaller spacing → tighter span → more margin
            # Larger spacing → wider span → less margin
            margin_factor = 0.1  # Base margin as fraction of total span
            base_margin = 0.5  # Minimum margin in data units
            margin = max(total_span * margin_factor, base_margin) + bar_width
            
            # Set limits to center bars with dynamic margins
            ax.set_xlim(-margin, total_span + margin)
        else:
            # Single bar - use symmetric margins
            ax.set_xlim(-1, 1)
        
        # Axis labels
        if xlabel is None:
            xlabel = 'Cluster (population)' if show_population else 'Cluster'
        if ylabel is None:
            ylabel = f'ΔG ({units})'
        
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        
        # Title
        if show_title:
            if title is None:
                title = 'Free Energy Landscape'
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        # Reference line at ΔG=0
        if show_reference_line:
            ax.axhline(0, color=reference_line_color, linestyle=reference_line_style,
                      linewidth=reference_line_width, alpha=reference_line_alpha)
        
        # Grid
        if show_grid:
            ax.grid(alpha=grid_alpha, linestyle=grid_style, linewidth=grid_width)
        
        # Y-axis ticks
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        
        # Legend
        if show_legend:
            # Create legend labels and filter bars/labels for noise cluster
            if legend_labels is None:
                # Auto-generate labels, optionally excluding noise cluster
                if legend_exclude_noise:
                    # Filter out cluster -1
                    legend_indices = [i for i, c in enumerate(cluster_ids) if c >= 0]
                    legend_cluster_ids = [cluster_ids[i] for i in legend_indices]
                    legend_bars = [bars[i] for i in legend_indices]
                    legend_labels = [f'Cluster {c}' for c in legend_cluster_ids]
                else:
                    # Include all clusters
                    legend_bars = bars
                    legend_labels = [f'Cluster {c}' for c in cluster_ids]
            else:
                # Custom labels provided
                if len(legend_labels) != len(cluster_ids):
                    print(f"Warning: legend_labels length ({len(legend_labels)}) doesn't match "
                          f"number of clusters ({len(cluster_ids)}). Using cluster IDs.")
                    legend_labels = [f'Cluster {c}' for c in cluster_ids]
                legend_bars = bars
            
            # Create legend
            legend_kwargs = {
                'fontsize': legend_fontsize,
                'framealpha': legend_frame_alpha,
                'edgecolor': legend_edgecolor,
                'fancybox': legend_fancybox,
                'shadow': legend_shadow,
                'ncol': legend_ncol
            }
            
            # Use legend1_bbox if provided, otherwise fall back to legend_bbox for backward compatibility
            if legend1_bbox is not None:
                legend_kwargs['bbox_to_anchor'] = legend1_bbox
                legend_kwargs['loc'] = 'upper left'  # Default for bbox_to_anchor
            elif legend_bbox is not None:
                legend_kwargs['bbox_to_anchor'] = legend_bbox
                legend_kwargs['loc'] = 'upper left'  # Default for bbox_to_anchor
            else:
                legend_kwargs['loc'] = legend_loc
            
            legend = ax.legend(legend_bars, legend_labels, **legend_kwargs)
            
            # Set legend title if provided
            if legend_title is not None:
                legend.set_title(legend_title, prop={'size': legend_title_fontsize, 
                                                     'weight': legend_title_fontweight})
            
            # Set legend text properties
            for text in legend.get_texts():
                text.set_fontweight(legend_fontweight)
        
        plt.tight_layout()
        
        # Save if requested
        if save_fig and save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, 
                       transparent=transparent_bg)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_multi_system_free_energy_barplot(self,
                                             systems_data: Dict,
                                             colors_per_system: Optional[Dict[str, List[str]]] = None,
                                             # Figure layout
                                             figsize: Tuple[float, float] = (12, 6),
                                             separate_y_axes: bool = False,
                                             subplot_width_ratios: Optional[List[float]] = None,
                                             equal_bar_width: bool = True,
                                             # Bar styling
                                             bar_width: float = 0.6,
                                             bar_spacing: float = 0.8,
                                             section_spacing: float = 1.5,
                                             bar_alpha: float = 0.85,
                                             edgecolor: str = 'black',
                                             edgewidth: float = 1.0,
                                             # Font controls
                                             title_fontsize: int = 14,
                                             title_fontweight: str = 'bold',
                                             label_fontsize: int = 12,
                                             label_fontweight: str = 'bold',
                                             tick_fontsize: int = 10,
                                             show_section_label: bool = True,
                                             section_label_fontsize: int = 14,
                                             # Y-axis limits
                                             ymin: Optional[Union[float, List[float]]] = None,
                                             ymax: Optional[Union[float, List[float]]] = None,
                                             # Population labels
                                             show_population: bool = True,
                                             population_fontsize: Optional[int] = None,
                                             population_fontweight: str = 'bold',
                                             population_y_offset: float = 0.5,
                                             # Grid
                                             show_grid: bool = True,
                                             grid_alpha: float = 0.3,
                                             # Legend
                                             show_legend: bool = False,
                                             legend_labels: Optional[List[str]] = None,
                                             legend_title: Optional[str] = None,
                                             legend_loc: str = 'best',
                                             legend_bbox: Optional[Tuple[float, float]] = None,
                                             legend_ncol: int = 1,
                                             legend_fontsize: int = 10,
                                             legend_fontweight: str = 'normal',
                                             legend_title_fontsize: int = 11,
                                             legend_title_fontweight: str = 'bold',
                                             legend_frame_alpha: float = 0.9,
                                             legend_edgecolor: str = 'black',
                                             # Axis labels and title
                                             xlabel: Optional[str] = None,
                                             ylabel: str = 'ΔΔG (kJ/mol)',
                                             show_title: bool = True,
                                             title: str = 'Multi-System Free Energy Comparison',
                                             # Save options
                                             save_fig: bool = False,
                                             save_path: str = 'combined_free_energy_barplot.png',
                                             dpi: int = 300,
                                             bbox_inches: str = 'tight',
                                             transparent_bg: bool = False) -> plt.Figure:
        """
        Plot grouped free energy comparison across multiple systems.
        
        Designed for comparing independent systems (e.g., CIP+, CIP+/-, CIP-) 
        where each system has different numbers of clusters.
        
        Parameters
        ----------
        systems_data : dict
            Dictionary with structure:
            {system_name: {
                'system_name': str,
                'fe_data': dict of {cluster_id: energy_value},
                'n_clusters': int,
                'clusters': list
            }}
            
        colors_per_system : dict, optional
            Custom colors for each system's clusters.
            Format: {system_name: [color1, color2, ...]}
            If None, uses default color palette.
        
        figsize : tuple
            Figure size (width, height) in inches
        
        separate_y_axes : bool, default=False
            If True, creates separate subplots with independent y-axes for each system.
            Recommended when systems have different reference states, as it allows
            optimal y-scale for each system independently. If False, uses single
            shared y-axis (original behavior).
        
        subplot_width_ratios : list of float, optional
            Width ratios for subplots when separate_y_axes=True.
            If None, automatically calculated. Overrides equal_bar_width when provided.
            Example: [1, 2, 6] gives CIP+ half the width of CIP+/- and 1/6 of CIP-.
        
        equal_bar_width : bool, default=True
            When separate_y_axes=True and subplot_width_ratios=None, this controls
            whether bars appear visually uniform across subplots. If True, subplot
            widths are proportional to (n_bars × bar_spacing) so bars look the same
            size. If False, widths are simply proportional to number of bars, which
            makes bars in sections with fewer clusters appear wider.
            
        bar_width : float
            Width of individual bars
            
        bar_spacing : float  
            Spacing between bars within the same system
            
        section_spacing : float
            Extra spacing between different system sections (only used when separate_y_axes=False)
            
        section_label_fontsize : int
            Font size for system section labels
        
        ymin : float or list of float, optional
            Minimum y-axis value(s). If a single float, applies to all subplots.
            If a list, each value applies to the corresponding subplot (must match
            number of systems). If None, uses automatic limits.
        
        ymax : float or list of float, optional
            Maximum y-axis value(s). If a single float, applies to all subplots.
            If a list, each value applies to the corresponding subplot (must match
            number of systems). If None, uses automatic limits.
            
        show_population : bool
            Whether to show population percentage labels on bars
            
        population_fontsize : int, optional
            Font size for population labels. If None, uses tick_fontsize - 1
            
        population_fontweight : str
            Font weight for population labels
            
        population_y_offset : float
            Vertical offset for population labels above bars
        
        Legend
        ------
        show_legend : bool, default=False
            Show legend with cluster color mapping
        
        legend_labels : list of str, optional
            Custom labels for legend entries. If None, uses 'Cluster 0', 'Cluster 1', etc.
            Must match the total number of unique clusters across all systems.
            Example: ['State A', 'State B', 'State C', 'State D', 'State E', 'State F']
        
        legend_title : str, optional
            Title for the legend
        
        legend_loc : str, default='best'
            Legend location ('best', 'upper right', 'upper left', 'lower left',
            'lower right', 'right', 'center left', 'center right', 'lower center',
            'upper center', 'center'). Only used if legend_bbox is None.
        
        legend_bbox : tuple of (float, float), optional
            Legend position in bbox_to_anchor format (x, y). Overrides legend_loc.
            Examples:
            - (0.05, 1.01): Above plot, left-aligned
            - (1.05, 1.0): Right of plot, top-aligned
            - (0.5, -0.15): Below plot, centered
        
        legend_ncol : int, default=1
            Number of columns in legend. Use 6 for horizontal layout with 6 clusters.
        
        legend_fontsize : int, default=10
            Legend text font size
        
        legend_fontweight : str, default='normal'
            Legend text font weight
        
        legend_title_fontsize : int, default=11
            Legend title font size
        
        legend_title_fontweight : str, default='bold'
            Legend title font weight
        
        legend_frame_alpha : float, default=0.9
            Legend box transparency (0=transparent, 1=opaque)
        
        legend_edgecolor : str, default='black'
            Legend box edge color
            
        Other Parameters
        ----------------
        Standard matplotlib styling parameters for fonts, grid, saving, etc.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure object
            
        Examples
        --------
        >>> # Load exported system data
        >>> import pickle
        >>> systems_data = {}
        >>> for name in ['CIP+', 'CIP+/-', 'CIP-']:
        ...     with open(f'fe_data_{name}.pkl', 'rb') as f:
        ...         systems_data[name] = pickle.load(f)
        >>> 
        >>> # Plot with single shared y-axis (default)
        >>> fig = plotter.plot_multi_system_free_energy_barplot(
        ...     systems_data,
        ...     colors_per_system={
        ...         'CIP+': ['#FF6B6B'],
        ...         'CIP+/-': ['#4ECDC4', '#45B7D1'],
        ...         'CIP-': ['#96CEB4', '#FECA57', '#FF9FF3', '#A8E6CF', '#FFD93D', '#6C5CE7']
        ...     },
        ...     section_spacing=2.0,
        ...     save_fig=True
        ... )
        >>> 
        >>> # Plot with separate y-axes (recommended for independent systems)
        >>> fig = plotter.plot_multi_system_free_energy_barplot(
        ...     systems_data,
        ...     colors_per_system={
        ...         'CIP+': ['#FF6B6B'],
        ...         'CIP+/-': ['#4ECDC4', '#45B7D1'],
        ...         'CIP-': ['#96CEB4', '#FECA57', '#FF9FF3', '#A8E6CF', '#FFD93D', '#6C5CE7']
        ...     },
        ...     separate_y_axes=True,
        ...     save_fig=True
        ... )
        >>> 
        >>> # Add legend showing cluster colors (positioned above plot)
        >>> fig = plotter.plot_multi_system_free_energy_barplot(
        ...     systems_data,
        ...     colors_per_system={'CIP-': ['#FFB6C1', '#F08080', '#ADD8E6', '#90EE90', '#FFFFE0', '#E6E6FA']},
        ...     separate_y_axes=True,
        ...     show_legend=True,
        ...     legend_labels=['State A', 'State B', 'State C', 'State D', 'State E', 'State F'],
        ...     legend_bbox=(0.05, 1.01),
        ...     legend_ncol=6,
        ...     legend_fontsize=10
        ... )
        """
        
        import numpy as np
        
        # Default colors if not provided
        if colors_per_system is None:
            default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
                            '#FF9FF3', '#A8E6CF', '#FFD93D', '#6C5CE7', '#DDA15E']
            colors_per_system = {
                system: default_colors[:len(data['fe_data'])] 
                for system, data in systems_data.items()
            }
        
        # Pre-process all systems data
        systems_plot_data = {}
        for system_name, system_data in systems_data.items():
            fe_data = system_data['fe_data']
            
            # Filter out reference cluster (-1) and extract energy values
            filtered_clusters = {}
            for cluster_id, cluster_data in fe_data.items():
                # Skip reference cluster (usually -1)
                if isinstance(cluster_data, dict) and cluster_data.get('is_reference', False):
                    continue
                
                # Extract delta_G if data is dict, otherwise use as-is
                if isinstance(cluster_data, dict):
                    energy_value = cluster_data.get('delta_G', cluster_data.get('energy', 0))
                    error_value = cluster_data.get('std_error', None)
                    population_value = cluster_data.get('population', None)
                else:
                    energy_value = cluster_data
                    error_value = None
                    population_value = None
                
                filtered_clusters[cluster_id] = {
                    'energy': energy_value,
                    'error': error_value,
                    'population': population_value
                }
            
            systems_plot_data[system_name] = {
                'filtered_clusters': filtered_clusters,
                'colors': colors_per_system.get(system_name, ['gray'] * len(filtered_clusters))
            }
        
        # Branch: Separate Y-axes (subplots) or single shared axis
        if separate_y_axes:
            # Create subplots with separate y-axes
            n_systems = len(systems_data)
            
            # Determine width ratios
            if subplot_width_ratios is None:
                if equal_bar_width:
                    # Calculate width ratios based on actual x-axis data range (including margins)
                    # This ensures bars appear uniform width across all subplots
                    subplot_width_ratios = []
                    for name in systems_data.keys():
                        n_bars = len(systems_plot_data[name]['filtered_clusters'])
                        if n_bars > 1:
                            # Multiple bars: span + margins
                            x_span = (n_bars - 1) * bar_spacing
                            margin = bar_width * 1.2
                            x_range = x_span + 2 * margin
                        else:
                            # Single bar: symmetric margins
                            x_range = 2 * (bar_width * 1.5)
                        subplot_width_ratios.append(x_range)
                else:
                    # Auto-calculate: proportional to number of clusters
                    subplot_width_ratios = [len(systems_plot_data[name]['filtered_clusters']) 
                                           for name in systems_data.keys()]
            
            fig, axes = plt.subplots(1, n_systems, figsize=figsize, 
                                    gridspec_kw={'width_ratios': subplot_width_ratios})
            
            # Handle single system case (axes won't be array)
            if n_systems == 1:
                axes = [axes]
            
            # Plot each system in its own subplot
            for idx, (system_name, plot_data) in enumerate(systems_plot_data.items()):
                ax = axes[idx]
                filtered_clusters = plot_data['filtered_clusters']
                colors = plot_data['colors']
                n_clusters = len(filtered_clusters)
                
                # X positions for this subplot (always start from 0)
                x_positions = np.arange(n_clusters) * bar_spacing
                
                # Plot bars first to establish y-range, then add labels
                for i, (cluster_id, cluster_info) in enumerate(filtered_clusters.items()):
                    x_pos = x_positions[i]
                    color = colors[i % len(colors)]
                    energy = cluster_info['energy']
                    error = cluster_info['error']
                    
                    ax.bar(x_pos, energy, width=bar_width, 
                          color=color, alpha=bar_alpha,
                          edgecolor=edgecolor,
                          linewidth=edgewidth)
                    
                    # Add error bars if available
                    if error is not None and error > 0:
                        ax.errorbar(x_pos, energy, yerr=error, 
                                  fmt='none', ecolor='black', 
                                  capsize=3, capthick=1.5, alpha=0.7)
                
                # Now get the y-range to properly estimate text height in data coordinates
                y_min_temp, y_max_temp = ax.get_ylim()
                y_range_temp = y_max_temp - y_min_temp
                
                # Estimate text height as percentage of y-range
                # For typical font sizes (8-12pt), text occupies about 4-6% of plot height
                text_height_estimate = 0.05 * y_range_temp
                
                # Track actual label positions for proper y-axis adjustment
                max_label_y = None
                min_label_y = None
                
                # Add population labels with proper tracking
                for i, (cluster_id, cluster_info) in enumerate(filtered_clusters.items()):
                    x_pos = x_positions[i]
                    energy = cluster_info['energy']
                    error = cluster_info['error']
                    population = cluster_info['population']
                    
                    if show_population and energy is not None:
                        pop_fs = population_fontsize if population_fontsize is not None else max(8, tick_fontsize - 1)
                        
                        # Position text based on bar direction
                        if energy >= 0:
                            text_y = energy + (error if error is not None else 0) + population_y_offset
                            valign = 'bottom'
                            # Track max for positive bars (text extends upward from anchor)
                            label_extent = text_y + text_height_estimate
                            if max_label_y is None or label_extent > max_label_y:
                                max_label_y = label_extent
                        else:
                            text_y = energy - (error if error is not None else 0) - population_y_offset
                            valign = 'top'
                            # Track min for negative bars (text extends downward from anchor)
                            label_extent = text_y - text_height_estimate
                            if min_label_y is None or label_extent < min_label_y:
                                min_label_y = label_extent
                        
                        # Show energy value on bar
                        ax.text(x_pos, text_y, f'{energy:.2f}', 
                               ha='center', va=valign, 
                               fontsize=pop_fs,
                               fontweight=population_fontweight,
                               color='black')
                
                # Format this subplot
                ax.set_xticks(x_positions)
                # X-axis labels show population percentages
                x_labels = []
                for cluster_id, cluster_info in filtered_clusters.items():
                    pop = cluster_info['population']
                    if pop is not None:
                        x_labels.append(f'{pop * 100:.1f}%')
                    else:
                        x_labels.append(f'C{cluster_id}')
                ax.set_xticklabels(x_labels, fontsize=tick_fontsize)
                
                # Adjust x-axis limits
                if n_clusters > 1:
                    margin = bar_width * 1.2
                    ax.set_xlim(x_positions[0] - margin, x_positions[-1] + margin)
                else:
                    ax.set_xlim(-bar_width * 1.5, bar_width * 1.5)
                
                # Add system section label (as subplot title or xlabel)
                if show_section_label:
                    ax.set_xlabel(system_name, fontsize=section_label_fontsize, fontweight='bold')
                
                # Y-label only on first subplot
                if idx == 0:
                    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                
                # Grid
                if show_grid:
                    ax.grid(True, alpha=grid_alpha, axis='y')
                
                # Adjust y-axis for population labels based on actual label positions
                y_min_orig, y_max_orig = ax.get_ylim()
                new_y_min = y_min_orig
                new_y_max = y_max_orig
                
                if show_population:
                    # Add small margin beyond the actual label positions
                    y_range = y_max_orig - y_min_orig
                    margin = 0.05 * y_range  # Small margin for visual comfort
                    
                    if max_label_y is not None and max_label_y > y_max_orig:
                        new_y_max = max_label_y + margin
                    
                    if min_label_y is not None and min_label_y < y_min_orig:
                        new_y_min = min_label_y - margin
                
                # Apply user-specified y-limits if provided
                if ymin is not None:
                    if isinstance(ymin, list):
                        if idx < len(ymin):
                            new_y_min = ymin[idx]
                    else:
                        new_y_min = ymin
                
                if ymax is not None:
                    if isinstance(ymax, list):
                        if idx < len(ymax):
                            new_y_max = ymax[idx]
                    else:
                        new_y_max = ymax
                
                ax.set_ylim(new_y_min, new_y_max)
                
                # Tick formatting
                ax.tick_params(axis='both', labelsize=tick_fontsize)
            
            # Overall title
            if show_title and title:
                fig.suptitle(title, fontsize=title_fontsize, fontweight=title_fontweight)
            
            # Add legend if requested (separate y-axes mode)
            if show_legend:
                # Collect unique cluster IDs and their colors across all systems
                # Use a dict to store first occurrence of each cluster ID
                unique_clusters = {}  # {cluster_id: color}
                
                for system_name in systems_data.keys():
                    plot_data = systems_plot_data[system_name]
                    filtered_clusters = plot_data['filtered_clusters']
                    colors = plot_data['colors']
                    
                    for i, cluster_id in enumerate(filtered_clusters.keys()):
                        if cluster_id not in unique_clusters:
                            color = colors[i % len(colors)]
                            unique_clusters[cluster_id] = color
                
                # Create legend handles from unique clusters
                from matplotlib.patches import Patch
                legend_handles = []
                for i, (cluster_id, color) in enumerate(sorted(unique_clusters.items())):
                    if legend_labels is not None and i < len(legend_labels):
                        label = legend_labels[i]
                    else:
                        label = f'Cluster {cluster_id}'
                    legend_handles.append(Patch(facecolor=color, edgecolor=edgecolor, 
                                               linewidth=edgewidth, label=label, alpha=bar_alpha))
                
                # Create figure-level legend
                legend_kwargs = {
                    'handles': legend_handles,
                    'fontsize': legend_fontsize,
                    'ncol': legend_ncol,
                    'framealpha': legend_frame_alpha,
                    'edgecolor': legend_edgecolor
                }
                
                if legend_title:
                    legend_kwargs['title'] = legend_title
                    legend_kwargs['title_fontsize'] = legend_title_fontsize
                
                if legend_bbox is not None:
                    legend_kwargs['bbox_to_anchor'] = legend_bbox
                    legend_kwargs['loc'] = 'upper left'  # Reference point for bbox_to_anchor
                else:
                    legend_kwargs['loc'] = legend_loc
                
                # Add font weight via prop
                from matplotlib.font_manager import FontProperties
                font_props = FontProperties(weight=legend_fontweight, size=legend_fontsize)
                legend_kwargs['prop'] = font_props
                
                if legend_title:
                    title_props = FontProperties(weight=legend_title_fontweight, size=legend_title_fontsize)
                    legend_kwargs['title_fontproperties'] = title_props
                
                fig.legend(**legend_kwargs)
            
            plt.tight_layout()
            
        else:
            # Original single-axis mode
            fig, ax = plt.subplots(figsize=figsize)
            
            # Calculate positions for grouped bars
            current_x = 0
            all_bars = []
            x_positions = []
            x_labels = []
            system_boundaries = []
            system_centers = []
            
            for system_name, plot_data in systems_plot_data.items():
                filtered_clusters = plot_data['filtered_clusters']
                colors = plot_data['colors']
                n_clusters = len(filtered_clusters)
                
                # Calculate x positions for this system's clusters
                if n_clusters == 1:
                    # Single cluster - center it
                    cluster_positions = [current_x]
                else:
                    # Multiple clusters - distribute them
                    cluster_start = current_x - (n_clusters - 1) * bar_spacing / 2
                    cluster_positions = [cluster_start + i * bar_spacing for i in range(n_clusters)]
                
                # Plot bars for this system
                for i, (cluster_id, cluster_info) in enumerate(filtered_clusters.items()):
                    x_pos = cluster_positions[i]
                    color = colors[i % len(colors)]
                    energy = cluster_info['energy']
                    error = cluster_info['error']
                    population = cluster_info['population']
                    
                    bar = ax.bar(x_pos, energy, width=bar_width, 
                               color=color, alpha=bar_alpha,
                               edgecolor=edgecolor,
                               linewidth=edgewidth)
                    
                    # Add error bars if available
                    if error is not None and error > 0:
                        ax.errorbar(x_pos, energy, yerr=error, 
                                  fmt='none', ecolor='black', 
                                  capsize=3, capthick=1.5, alpha=0.7)
                    
                    # Add energy value label on top/bottom of bar
                    if show_population and energy is not None:
                        pop_fs = population_fontsize if population_fontsize is not None else max(8, tick_fontsize - 1)
                        
                        # Position text based on bar direction
                        if energy >= 0:
                            # Positive values: label above bar
                            text_y = energy + (error if error is not None else 0) + population_y_offset
                            valign = 'bottom'
                        else:
                            # Negative values: label below bar
                            text_y = energy - (error if error is not None else 0) - population_y_offset
                            valign = 'top'
                        
                        # Show energy value on bar
                        ax.text(x_pos, text_y, f'{energy:.2f}', 
                               ha='center', va=valign, 
                               fontsize=pop_fs,
                               fontweight=population_fontweight,
                               color='black')
                    
                    all_bars.extend(bar)
                    
                    # Store positions and labels
                    x_positions.append(x_pos)
                    # X-axis labels show population percentages
                    if population is not None:
                        x_labels.append(f'{population * 100:.1f}%')
                    else:
                        x_labels.append(f'C{cluster_id}')
                
                # Store system center for section label
                system_center = np.mean(cluster_positions)
                system_centers.append((system_center, system_name))
                
                # Update position for next system
                if current_x == 0:
                    # First system - move to its max position
                    current_x = max(cluster_positions) + section_spacing
                else:
                    current_x = max(cluster_positions) + section_spacing
                    
                # Add system boundary for vertical line
                if len(system_boundaries) < len(systems_data) - 1:
                    system_boundaries.append(max(cluster_positions) + section_spacing / 2)
            
            # Add vertical lines between systems
            for boundary in system_boundaries:
                ax.axvline(boundary, color='lightgray', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Format x-axis
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=tick_fontsize)
            
            # Get current y-axis limits
            y_min, y_max = ax.get_ylim()
            
            # Add system section labels below x-axis
            if show_section_label:
                label_y = y_min - 0.12 * (y_max - y_min)
                
                for center_x, sys_name in system_centers:
                    ax.text(center_x, label_y, sys_name, 
                           ha='center', va='top', 
                           fontweight='bold',
                           fontsize=section_label_fontsize)
            
            # Labels and title
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title and title:
                ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
            
            # Grid
            if show_grid:
                ax.grid(True, alpha=grid_alpha, axis='y')
            
            # Adjust y-axis to make room for population labels and/or section labels
            y_range = y_max - y_min
            new_y_min = y_min
            new_y_max = y_max
            
            if show_section_label:
                # Full padding for section labels at bottom
                new_y_min = y_min - 0.15 * y_range
            elif show_population:
                # Less padding when only population labels are shown at bottom
                new_y_min = y_min - 0.08 * y_range
            
            if show_population:
                # Add padding at top for population labels above positive bars
                new_y_max = y_max + 0.08 * y_range
            
            ax.set_ylim(new_y_min, new_y_max)
            
            # Tick formatting
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            
            # Add legend if requested (single-axis mode)
            if show_legend:
                # Collect unique cluster IDs and their colors across all systems
                # Use a dict to store first occurrence of each cluster ID
                unique_clusters = {}  # {cluster_id: color}
                
                for system_name in systems_data.keys():
                    plot_data = systems_plot_data[system_name]
                    filtered_clusters = plot_data['filtered_clusters']
                    colors = plot_data['colors']
                    
                    for i, cluster_id in enumerate(filtered_clusters.keys()):
                        if cluster_id not in unique_clusters:
                            color = colors[i % len(colors)]
                            unique_clusters[cluster_id] = color
                
                # Create legend handles from unique clusters
                from matplotlib.patches import Patch
                legend_handles = []
                for i, (cluster_id, color) in enumerate(sorted(unique_clusters.items())):
                    if legend_labels is not None and i < len(legend_labels):
                        label = legend_labels[i]
                    else:
                        label = f'Cluster {cluster_id}'
                    legend_handles.append(Patch(facecolor=color, edgecolor=edgecolor, 
                                               linewidth=edgewidth, label=label, alpha=bar_alpha))
                
                # Create axis legend
                legend_kwargs = {
                    'handles': legend_handles,
                    'fontsize': legend_fontsize,
                    'ncol': legend_ncol,
                    'framealpha': legend_frame_alpha,
                    'edgecolor': legend_edgecolor
                }
                
                if legend_title:
                    legend_kwargs['title'] = legend_title
                    legend_kwargs['title_fontsize'] = legend_title_fontsize
                
                if legend_bbox is not None:
                    legend_kwargs['bbox_to_anchor'] = legend_bbox
                    legend_kwargs['loc'] = 'upper left'  # Reference point for bbox_to_anchor
                else:
                    legend_kwargs['loc'] = legend_loc
                
                # Add font weight via prop
                from matplotlib.font_manager import FontProperties
                font_props = FontProperties(weight=legend_fontweight, size=legend_fontsize)
                legend_kwargs['prop'] = font_props
                
                if legend_title:
                    title_props = FontProperties(weight=legend_title_fontweight, size=legend_title_fontsize)
                    legend_kwargs['title_fontproperties'] = title_props
                
                ax.legend(**legend_kwargs)
            
            plt.tight_layout()
        
        # Save if requested
        if save_fig and save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, 
                       transparent=transparent_bg)
            print(f"✓ Saved combined plot: {save_path}")
        
        return fig

    def plot_multi_system_coordination_barplot(self,
                                               systems_data: Dict,
                                               coord_key: Union[str, List[str]],
                                               coord_key_labels: Optional[Dict[str, str]] = None,
                                               system_display_names: Optional[Dict[str, str]] = None,
                                               colors_per_system: Optional[Dict[str, List[str]]] = None,
                                               cn_hatches: Optional[Dict[int, str]] = None,
                                               # Figure layout
                                               figsize: Tuple[float, float] = (10, 5),
                                               separate_y_axes: bool = False,
                                               subplot_width_ratios: Optional[List[float]] = None,
                                               equal_bar_width: bool = True,
                                               panel_spacing: float = 0.35,
                                               shared_y_across_panels: bool = False,
                                               # Bar styling
                                               bar_width: float = 0.6,
                                               bar_spacing: float = 0.0,
                                               section_spacing: float = 1.5,
                                               bar_alpha: float = 0.85,
                                               edgecolor: str = 'black',
                                               edgewidth: float = 1.0,
                                               # Error bars
                                               show_error_bars: bool = True,
                                               error_bar_capsize: float = 4.0,
                                               error_bar_linewidth: float = 1.5,
                                               error_bar_color: str = 'black',
                                               # Font controls
                                               title_fontsize: int = 14,
                                               title_fontweight: str = 'bold',
                                               label_fontsize: int = 12,
                                               label_fontweight: str = 'bold',
                                               tick_fontsize: int = 10,
                                               show_section_label: bool = True,
                                               section_label_fontsize: int = 14,
                                               section_label_fontweight: str = 'bold',
                                               section_label_offset: float = -0.06,
                                               section_label_rotation: float = 0.0,
                                               section_label_style: str = 'text',
                                               section_bracket_capheight: float = 0.015,
                                               section_bracket_linewidth: float = 1.2,
                                               # Y-axis limits
                                               ymin: Optional[Union[float, List[float]]] = None,
                                               ymax: Optional[Union[float, List[float]]] = None,
                                               # Grid
                                               show_grid: bool = True,
                                               grid_alpha: float = 0.3,
                                               grid_linestyle: str = '--',
                                               # Spines / minor ticks
                                               hide_top_right_spines: bool = False,
                                               show_minor_ticks: bool = False,
                                               # Population labels above bars
                                               show_population: bool = False,
                                               population_fontsize: int = 9,
                                               population_fontweight: str = 'bold',
                                               population_y_offset: float = 0.01,
                                               # Legend
                                               show_legend: bool = False,
                                               show_cn_legend: bool = True,
                                               legend_layout: str = 'horizontal',
                                               legend_labels: Optional[List[str]] = None,
                                               legend_title: Optional[str] = None,
                                               legend_loc: str = 'upper left',
                                               legend_ncol_cluster: int = 1,
                                               legend_ncol_cn: int = 1,
                                               legend1_bbox: Optional[Tuple[float, float]] = None,
                                               legend2_bbox: Optional[Tuple[float, float]] = None,
                                               legend_fontsize: int = 10,
                                               legend_fontweight: str = 'bold',
                                               legend_title_fontsize: int = 10,
                                               legend_title_fontweight: str = 'bold',
                                               legend_framealpha: float = 0.9,
                                               legend_edgecolor: str = 'black',
                                               legend_handletextpad: float = 0.8,
                                               # Axis labels and title
                                               xlabel: Optional[str] = None,
                                               ylabel: str = 'Mean CN',
                                               show_panel_titles: bool = True,
                                               show_suptitle: bool = False,
                                               title: str = 'Multi-System Coordination Comparison',
                                               # Filtering
                                               exclude_keys: Optional[List[str]] = None,
                                               # Save options
                                               save_fig: bool = False,
                                               save_path: str = 'combined_coordination_barplot.png',
                                               dpi: int = 300,
                                               bbox_inches: str = 'tight',
                                               transparent_bg: bool = False) -> plt.Figure:
        """
        Plot grouped mean coordination number comparison across multiple systems.

        Supports plotting a single coordination key or a **list of keys** as
        side-by-side moiety panels (one panel per key), similar to how
        ``plot_multi_system_distance_distributions_3d`` handles a ``dist_key`` list.

        Parameters
        ----------
        systems_data : dict
            Loaded from ``export_coordination_data_for_multi_system()``:
            ``{system_name: {'system_name', 'coordination_data', 'cluster_ids', 'available_keys'}}``

        coord_key : str or list of str
            The coordination key(s) to plot.  Each key is the string produced by
            ``compute_coordination_numbers()``:
            ``"<center_selection>__<neighbor_selection>__<cutoff>"``.
            If a **list** is supplied, one panel is created per key (side-by-side).

        coord_key_labels : dict, optional
            Human-readable panel titles keyed by coord_key string.
            Example::

                {'sel_A__sel_B__2.35': 'Carboxylic Acid',
                 'sel_A__sel_B__3.35': 'Quinolone'}

        system_display_names : dict, optional
            Display names for each system (supports LaTeX).
            Example::

                {'CIP+': r'$CIP^+$', 'CIP+/-': r'$CIP^{+/-}$', 'CIP-': r'$CIP^-$'}

        colors_per_system : dict, optional
            ``{system_name: [color1, color2, ...]}`` — one color per cluster.

        figsize : tuple
            Figure width × height in inches.

        panel_spacing : float, default=0.35
            Horizontal spacing (wspace) between moiety panels when coord_key is a list.

        shared_y_across_panels : bool, default=False
            When True and coord_key is a list, all panels share the same y-axis range
            (maximum across all panels). Useful for direct comparisons.

        separate_y_axes : bool, default=False
            Within each moiety panel: split each *system* into its own sub-column
            with an independent y-axis.  Ignored when coord_key is a list (each
            panel already has one shared y-axis per moiety).

        bar_width, bar_spacing, section_spacing : float
            Bar geometry controls.

        show_error_bars : bool, default=True
            Draw standard-deviation error bars.

        show_section_label : bool, default=True
            Draw system-name labels under the x-axis ticks.

        ymin, ymax : float or list of float, optional
            Y-axis limits.  When coord_key is a list, passing a list of floats
            assigns per-panel limits.

        show_legend : bool, default=False
            Draw a cluster-colour legend on the last panel.

        ylabel : str
            Shared y-axis label (left panel only when multiple panels).

        title : str
            Suptitle when coord_key is a list; axes title otherwise.

        save_fig, save_path, dpi : standard save parameters.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        from matplotlib.patches import Patch

        # ------------------------------------------------------------------ #
        # Normalise coord_key to a list and apply exclusions
        # ------------------------------------------------------------------ #
        if isinstance(coord_key, str):
            coord_keys = [coord_key]
        else:
            coord_keys = list(coord_key)

        if exclude_keys:
            coord_keys = [k for k in coord_keys if k not in exclude_keys]

        n_panels = len(coord_keys)

        if coord_key_labels is None:
            coord_key_labels = {}
        if system_display_names is None:
            system_display_names = {}

        system_names = list(systems_data.keys())
        n_systems = len(system_names)

        # Default cluster colours
        _default_colors = ['#E07B7B', '#7BB3E0', '#7BE09A', '#E0D07B', '#C07BE0',
                           '#E0A87B', '#7BE0D8', '#B07BE0', '#E07BAE', '#A0A0A0']

        # CN hatch defaults: assigned by ascending CN integer value (same as plot_multi_moiety_coordination_stacked)
        _default_cn_hatch_list = ['', '///', '\\\\\\', 'xxx', '...', '|||', '+++', 'ooo', '**', 'oo']

        def _resolve_cn_hatches(cn_vals_sorted):
            """Build {cn_val: hatch} for all CN integers observed in this panel."""
            resolved = {}
            for i, cn_val in enumerate(cn_vals_sorted):
                if cn_hatches is not None and cn_val in cn_hatches:
                    resolved[cn_val] = cn_hatches[cn_val]
                else:
                    resolved[cn_val] = _default_cn_hatch_list[i % len(_default_cn_hatch_list)]
            return resolved


        # ------------------------------------------------------------------ #
        # Resolve coord_keys: each entry can be either a raw key string that
        # exists identically in all systems, OR a moiety alias name stored
        # in system_data['moiety_aliases'] (allowing different cutoffs per
        # system for the same moiety).
        #
        # resolved_keys[sys_name][canonical_key] = actual_key_in_that_system
        # ------------------------------------------------------------------ #
        resolved_keys = {sys_name: {} for sys_name in system_names}

        for key in coord_keys:
            for sys_name in system_names:
                sys_data = systems_data[sys_name]
                coord_data = sys_data['coordination_data']
                aliases = sys_data.get('moiety_aliases', {})

                if key in coord_data:
                    # Direct match — same key across all systems
                    resolved_keys[sys_name][key] = key
                elif key in aliases and aliases[key] in coord_data:
                    # Alias match — key is a human-readable name
                    resolved_keys[sys_name][key] = aliases[key]
                else:
                    available = sys_data.get('available_keys', [])
                    alias_names = list(aliases.keys())
                    raise ValueError(
                        f"coord_key '{key}' not found in system '{sys_name}'.\n"
                        f"Available raw keys:\n" +
                        "\n".join(f"  - {k}" for k in available) +
                        (f"\n\nAvailable alias names: {alias_names}" if alias_names else
                         "\n\nNo moiety_aliases stored — export with moiety_aliases= to enable alias lookup.")
                    )

        # ------------------------------------------------------------------ #
        # Helper: compute per-cluster stats + CN integer breakdown for one key.
        # Returns (info_dict, cn_vals_sorted) where cn_vals_sorted is the
        # sorted list of all integer CN values observed across every
        # system/cluster — used for consistent hatch + stacking order.
        # ------------------------------------------------------------------ #
        def _compute_systems_info(key):
            info = {}
            all_cn_vals = set()
            for sys_name in system_names:
                sys_data = systems_data[sys_name]
                actual_key = resolved_keys[sys_name][key]
                cluster_ids = sys_data['cluster_ids']
                coord_results = sys_data['coordination_data'][actual_key]

                total_frames = sum(
                    len(coord_results[cid]['coordination'])
                    for cid in cluster_ids if cid in coord_results
                )
                mean_cns, std_cns, cn_contributions = {}, {}, {}
                for cid in cluster_ids:
                    if cid not in coord_results:
                        continue
                    cn_arr = coord_results[cid]['coordination']
                    mean_cns[cid] = float(np.mean(cn_arr))
                    std_cns[cid] = float(np.std(cn_arr))
                    # Stacked-bar breakdown: each segment = cn_val × P(cn_val)
                    unique_cns, counts = np.unique(cn_arr, return_counts=True)
                    probs = counts / len(cn_arr)
                    cn_contributions[cid] = {
                        int(cn): int(cn) * float(p)
                        for cn, p in zip(unique_cns, probs)
                    }
                    all_cn_vals.update(int(cn) for cn in unique_cns)
                populations = {}
                if total_frames > 0:
                    for cid in cluster_ids:
                        if cid in coord_results:
                            populations[cid] = len(coord_results[cid]['coordination']) / total_frames
                info[sys_name] = {
                    'cluster_ids': cluster_ids,
                    'mean_cns': mean_cns,
                    'std_cns': std_cns,
                    'cn_contributions': cn_contributions,
                    'populations': populations,
                }
            return info, sorted(all_cn_vals)

        # ------------------------------------------------------------------ #
        # Helper: build colours per system
        # ------------------------------------------------------------------ #
        def _resolved_colors(systems_info):
            rc = {}
            for sys_name in system_names:
                if colors_per_system and sys_name in colors_per_system:
                    rc[sys_name] = list(colors_per_system[sys_name])
                else:
                    n_clust = len(systems_info[sys_name]['cluster_ids'])
                    rc[sys_name] = _default_colors[:n_clust]
            return rc

        # ------------------------------------------------------------------ #
        # Helper: draw stacked-CN bars for one (key, ax) pair — shared y model
        # Colors = clusters, hatches = CN integer values (same encoding as
        # plot_multi_moiety_coordination_stacked mean panel).
        # ------------------------------------------------------------------ #
        def _draw_shared_panel(ax, systems_info, resolved_colors_map,
                               cn_vals_sorted, resolved_cn_hatches,
                               ymin_val=None, ymax_val=None,
                               show_ylabel_flag=True, panel_title=None):
            current_x = 0.0
            section_centers = {}
            section_edges = {}  # sys_name -> (left_x, right_x) in data coords

            for sys_name in system_names:
                info = systems_info[sys_name]
                cluster_ids = info['cluster_ids']
                colors_list = resolved_colors_map[sys_name]
                _pitch = bar_width + bar_spacing  # center-to-center distance (0 gap = bars touch)
                xs = [current_x + i * _pitch for i in range(len(cluster_ids))]
                section_centers[sys_name] = float(np.mean(xs))
                section_edges[sys_name] = (xs[0] - bar_width / 2, xs[-1] + bar_width / 2)

                for bar_idx, cid in enumerate(cluster_ids):
                    if cid not in info['mean_cns']:
                        current_x += _pitch
                        continue
                    mean_val = info['mean_cns'][cid]
                    std_val = info['std_cns'][cid]
                    color = colors_list[bar_idx % len(colors_list)]
                    cn_contribs = info['cn_contributions'].get(cid, {})

                    # Stack bars: highest CN at bottom, lowest at top
                    # (same order as plot_multi_moiety_coordination_stacked mean panel)
                    bottom = 0.0
                    for cn_val in reversed(cn_vals_sorted):
                        seg_height = cn_contribs.get(cn_val, 0.0)
                        if seg_height <= 0.0:
                            continue
                        hatch = resolved_cn_hatches.get(cn_val, '')
                        ax.bar(xs[bar_idx], seg_height, bottom=bottom,
                               width=bar_width, color=color, alpha=bar_alpha,
                               hatch=hatch, edgecolor=edgecolor,
                               linewidth=edgewidth, zorder=2)
                        bottom += seg_height

                    # Error bar at the mean (top of stack) to show ±std
                    if show_error_bars:
                        ax.errorbar(xs[bar_idx], mean_val, yerr=std_val,
                                    fmt='none', capsize=error_bar_capsize,
                                    linewidth=error_bar_linewidth,
                                    ecolor=error_bar_color,
                                    capthick=error_bar_linewidth, zorder=3)

                    # Population label above bar
                    if show_population:
                        pop = info['populations'].get(cid, 0.0)
                        bar_top = mean_val + std_val if show_error_bars else mean_val
                        ax.annotate(
                            f'{pop:.0%}',
                            xy=(xs[bar_idx], bar_top + population_y_offset),
                            xycoords=('data', 'data'),
                            ha='center', va='bottom',
                            fontsize=population_fontsize,
                            fontweight=population_fontweight,
                            zorder=4,
                        )

                current_x = xs[-1] + bar_width + section_spacing

            # X ticks: one per cluster across all systems
            all_xs, all_labels = [], []
            cur_x2 = 0.0
            for sys_name in system_names:
                cluster_ids = systems_info[sys_name]['cluster_ids']
                _pitch2 = bar_width + bar_spacing
                for bar_idx, cid in enumerate(cluster_ids):
                    all_xs.append(cur_x2 + bar_idx * _pitch2)
                    all_labels.append(f'C{cid}')
                cur_x2 += len(cluster_ids) * _pitch2 + section_spacing

            ax.set_xticks(all_xs)
            ax.set_xticklabels([''] * len(all_xs))

            # System section labels — placed just below the x-axis using the
            # xaxis transform (x in data units, y in axes fraction) so the
            # position is consistent regardless of the data y-range.
            if show_section_label:
                from matplotlib.font_manager import FontProperties as _FP

                def _apply_bold_mathtext(s, weight):
                    """Inject \\boldsymbol{} for pure mathtext strings since
                    FontProperties.weight is ignored by the mathtext renderer."""
                    if weight == 'bold' and s.startswith('$') and s.endswith('$'):
                        inner = s[1:-1]
                        return f'$\\boldsymbol{{{inner}}}$'
                    return s

                _orig_mathtext = plt.rcParams.get('mathtext.fontset', 'dejavusans')
                if 'times' in self.font_family.lower():
                    plt.rcParams['mathtext.fontset'] = 'stix'
                _sec_fp = _FP(family=self.font_family, weight=section_label_fontweight,
                              size=section_label_fontsize)
                xform = ax.get_xaxis_transform()
                for sys_name, cx in section_centers.items():
                    display = system_display_names.get(sys_name, sys_name)
                    display = _apply_bold_mathtext(display, section_label_fontweight)
                    if section_label_style == 'bracket':
                        lx, rx = section_edges[sys_name]
                        by = section_label_offset
                        _cap = section_bracket_capheight
                        _lw = section_bracket_linewidth
                        # Horizontal bar spanning the full section width
                        ax.plot([lx, rx], [by, by], transform=xform,
                                color=edgecolor, linewidth=_lw,
                                clip_on=False, zorder=5, solid_capstyle='butt')
                        # Left endcap (upward)
                        ax.plot([lx, lx], [by, by + _cap], transform=xform,
                                color=edgecolor, linewidth=_lw, clip_on=False, zorder=5)
                        # Right endcap (upward)
                        ax.plot([rx, rx], [by, by + _cap], transform=xform,
                                color=edgecolor, linewidth=_lw, clip_on=False, zorder=5)
                        # Center notch (downward) — mimics curly-brace pinch
                        ax.plot([cx, cx], [by, by - _cap], transform=xform,
                                color=edgecolor, linewidth=_lw, clip_on=False, zorder=5)
                        # Label centred below the center notch
                        ax.annotate(display, xy=(cx, by - _cap - 0.01),
                                    xycoords=xform, ha='center', va='top',
                                    rotation=section_label_rotation,
                                    fontproperties=_sec_fp)
                    else:
                        ax.annotate(display, xy=(cx, section_label_offset),
                                    xycoords=xform, ha='center', va='top',
                                    rotation=section_label_rotation,
                                    fontproperties=_sec_fp)
                plt.rcParams['mathtext.fontset'] = _orig_mathtext

            ax.tick_params(axis='both', labelsize=tick_fontsize)

            if show_ylabel_flag:
                ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
            if xlabel is not None:
                ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
            if panel_title:
                ax.set_title(panel_title, fontsize=title_fontsize, fontweight=title_fontweight)

            cur_ylim = ax.get_ylim()
            ax.set_ylim(
                bottom=ymin_val if ymin_val is not None else max(0.0, cur_ylim[0]),
                top=ymax_val if ymax_val is not None else None,
            )

            if show_grid:
                ax.grid(True, axis='y', alpha=grid_alpha, linestyle=grid_linestyle, zorder=0)
            ax.set_axisbelow(True)

            if hide_top_right_spines:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            if show_minor_ticks:
                import matplotlib.ticker as _ticker
                ax.yaxis.set_minor_locator(_ticker.AutoMinorLocator())

        # ------------------------------------------------------------------ #
        # Helper: draw separate-y-axis panel with stacked CN bars
        # ------------------------------------------------------------------ #
        def _draw_separate_panel(axes_row, systems_info, resolved_colors_map,
                                 cn_vals_sorted, resolved_cn_hatches,
                                 ymin_list, ymax_list, panel_title=None):
            for ax_idx, (sys_name, ax) in enumerate(zip(system_names, axes_row)):
                info = systems_info[sys_name]
                cluster_ids = info['cluster_ids']
                colors_list = resolved_colors_map[sys_name]
                _pitch = bar_width + bar_spacing  # center-to-center (0 gap = bars touch)
                xs = [i * _pitch for i in range(len(cluster_ids))]

                for bar_idx, cid in enumerate(cluster_ids):
                    if cid not in info['mean_cns']:
                        continue
                    mean_val = info['mean_cns'][cid]
                    std_val = info['std_cns'][cid]
                    color = colors_list[bar_idx % len(colors_list)]
                    cn_contribs = info['cn_contributions'].get(cid, {})

                    bottom = 0.0
                    for cn_val in cn_vals_sorted:
                        seg_height = cn_contribs.get(cn_val, 0.0)
                        if seg_height <= 0.0:
                            continue
                        hatch = resolved_cn_hatches.get(cn_val, '')
                        ax.bar(xs[bar_idx], seg_height, bottom=bottom,
                               width=bar_width, color=color, alpha=bar_alpha,
                               hatch=hatch, edgecolor=edgecolor,
                               linewidth=edgewidth, zorder=2)
                        bottom += seg_height

                    if show_error_bars:
                        ax.errorbar(xs[bar_idx], mean_val, yerr=std_val,
                                    fmt='none', capsize=error_bar_capsize,
                                    linewidth=error_bar_linewidth,
                                    ecolor=error_bar_color,
                                    capthick=error_bar_linewidth, zorder=3)

                ax.set_xticks(xs)
                ax.set_xticklabels([''] * len(cluster_ids), fontsize=tick_fontsize)
                ax.tick_params(axis='y', labelsize=tick_fontsize)

                if show_section_label:
                    from matplotlib.font_manager import FontProperties as _FP

                    def _apply_bold_mathtext(s, weight):
                        """Inject \\boldsymbol{} for pure mathtext strings since
                        FontProperties.weight is ignored by the mathtext renderer."""
                        if weight == 'bold' and s.startswith('$') and s.endswith('$'):
                            inner = s[1:-1]
                            return f'$\\boldsymbol{{{inner}}}$'
                        return s

                    _orig_mathtext = plt.rcParams.get('mathtext.fontset', 'dejavusans')
                    if 'times' in self.font_family.lower():
                        plt.rcParams['mathtext.fontset'] = 'stix'
                    _sec_fp = _FP(family=self.font_family, weight=section_label_fontweight,
                                  size=section_label_fontsize)
                    display = system_display_names.get(sys_name, sys_name)
                    display = _apply_bold_mathtext(display, section_label_fontweight)
                    ax.set_xlabel(display, fontproperties=_sec_fp,
                                  rotation=section_label_rotation)
                    plt.rcParams['mathtext.fontset'] = _orig_mathtext

                if show_grid:
                    ax.grid(True, axis='y', alpha=grid_alpha, linestyle='--', zorder=0)
                ax.set_axisbelow(True)

                if ax_idx == 0:
                    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                    if panel_title:
                        ax.set_title(panel_title, fontsize=title_fontsize,
                                     fontweight=title_fontweight)
                else:
                    ax.tick_params(labelleft=False)
                    if ax_idx > 0:
                        ax.spines['left'].set_visible(False)
                if ax_idx < n_systems - 1:
                    ax.spines['right'].set_visible(False)

                cur_ylim = ax.get_ylim()
                ax.set_ylim(
                    bottom=ymin_list[ax_idx] if ymin_list[ax_idx] is not None else max(0.0, cur_ylim[0]),
                    top=ymax_list[ax_idx] if ymax_list[ax_idx] is not None else None,
                )

        # ================================================================== #
        #  Shared split-legend helper (used by both single- and multi-key paths)
        # ================================================================== #
        def _add_legends(ref_ax, cluster_handles, cn_handles, fig=None):
            """Draw Cluster + CN legends.
            When fig is provided (multi-panel): uses fig.legend() at figure level —
            completely decoupled from panel geometry.  bbox_to_anchor coords are
            in figure-fraction space: 0=left edge, 1=right edge, 0.5=centre.
            When fig is None (single panel): uses ref_ax.legend() as normal."""
            def _style(leg):
                for _t in leg.get_texts():
                    _t.set_fontweight(legend_fontweight)
                leg.get_title().set_fontweight(legend_title_fontweight)
                leg.get_title().set_fontsize(legend_title_fontsize)

            _use_fig = fig is not None and legend_layout == 'horizontal'
            _legend_fn = (lambda **kw: fig.legend(**kw)) if _use_fig else ref_ax.legend
            _transform_kw = ({'bbox_transform': fig.transFigure} if _use_fig else {})

            if legend_layout == 'horizontal':
                _bbox1 = legend1_bbox if legend1_bbox is not None else (0.70, 0.98)
                _bbox2 = legend2_bbox if legend2_bbox is not None else (0.85, 0.98)
                if show_legend and cluster_handles:
                    leg1 = _legend_fn(handles=cluster_handles,
                                      title=legend_title or 'Cluster',
                                      bbox_to_anchor=_bbox1, loc='upper left',
                                      fontsize=legend_fontsize,
                                      framealpha=legend_framealpha,
                                      edgecolor=legend_edgecolor,
                                      ncol=legend_ncol_cluster,
                                      handletextpad=legend_handletextpad,
                                      **_transform_kw)
                    _style(leg1)
                    if not _use_fig and show_cn_legend and cn_handles:
                        ref_ax.add_artist(leg1)
                if show_cn_legend and cn_handles:
                    leg2 = _legend_fn(handles=cn_handles, title='CN',
                                      bbox_to_anchor=_bbox2, loc='upper left',
                                      fontsize=legend_fontsize,
                                      framealpha=legend_framealpha,
                                      edgecolor=legend_edgecolor,
                                      ncol=legend_ncol_cn,
                                      handletextpad=legend_handletextpad,
                                      **_transform_kw)
                    _style(leg2)
            else:
                if show_legend and cluster_handles:
                    leg1 = ref_ax.legend(handles=cluster_handles,
                                         title=legend_title or 'Cluster',
                                         loc=legend_loc,
                                         fontsize=legend_fontsize,
                                         framealpha=legend_framealpha,
                                         edgecolor=legend_edgecolor,
                                         ncol=legend_ncol_cluster,
                                         handletextpad=legend_handletextpad)
                    _style(leg1)
                    if show_cn_legend and cn_handles:
                        ref_ax.add_artist(leg1)
                if show_cn_legend and cn_handles:
                    leg2 = ref_ax.legend(handles=cn_handles, title='CN',
                                         loc=legend_loc,
                                         fontsize=legend_fontsize,
                                         framealpha=legend_framealpha,
                                         edgecolor=legend_edgecolor,
                                         ncol=legend_ncol_cn,
                                         handletextpad=legend_handletextpad)
                    _style(leg2)

        # ================================================================== #
        #                       SINGLE KEY – ORIGINAL LOGIC
        # ================================================================== #
        if n_panels == 1:
            key = coord_keys[0]
            systems_info, cn_vals_sorted = _compute_systems_info(key)
            resolved_cn_hatches = _resolve_cn_hatches(cn_vals_sorted)
            resolved_colors_map = _resolved_colors(systems_info)
            panel_title = coord_key_labels.get(key, key) if show_panel_titles else None

            if separate_y_axes:
                # Build width ratios
                if subplot_width_ratios is not None:
                    width_ratios = subplot_width_ratios
                elif equal_bar_width:
                    width_ratios = [max(1, len(systems_info[s]['cluster_ids'])) * bar_spacing
                                    for s in system_names]
                else:
                    width_ratios = [max(1, len(systems_info[s]['cluster_ids']))
                                    for s in system_names]

                fig, axes = plt.subplots(1, n_systems, figsize=figsize,
                                         gridspec_kw={'width_ratios': width_ratios,
                                                      'wspace': 0.08},
                                         sharey=False)
                if n_systems == 1:
                    axes = [axes]

                ymin_list = ([ymin] * n_systems if ymin is None or isinstance(ymin, (int, float))
                             else list(ymin))
                ymax_list = ([ymax] * n_systems if ymax is None or isinstance(ymax, (int, float))
                             else list(ymax))

                _draw_separate_panel(axes, systems_info, resolved_colors_map,
                                     cn_vals_sorted, resolved_cn_hatches,
                                     ymin_list, ymax_list, panel_title=panel_title)

                if show_suptitle and panel_title:
                    fig.suptitle(panel_title if title == 'Multi-System Coordination Comparison'
                                 else title,
                                 fontsize=title_fontsize, fontweight=title_fontweight, y=1.02)
                if xlabel is not None:
                    fig.text(0.5, -0.04, xlabel, ha='center', va='top',
                             fontsize=label_fontsize, fontweight=label_fontweight)

                # Build legend handles before tight_layout
                _cluster_handles, _cn_handles = [], []
                if show_legend or show_cn_legend:
                    all_cids_flat = []
                    for sn in system_names:
                        for cid in systems_info[sn]['cluster_ids']:
                            if cid not in all_cids_flat:
                                all_cids_flat.append(cid)
                    _cluster_handles = [
                        Patch(facecolor=_default_colors[i % len(_default_colors)],
                              edgecolor=edgecolor, alpha=bar_alpha,
                              label=(legend_labels[i] if legend_labels and i < len(legend_labels)
                                     else f'C{cid}'))
                        for i, cid in enumerate(all_cids_flat)
                    ]
                    _cn_handles = [
                        Patch(facecolor='white', edgecolor=edgecolor,
                              hatch=resolved_cn_hatches.get(cv, ''), label=f'{cv}')
                        for cv in cn_vals_sorted
                    ]

                plt.tight_layout()
                _add_legends(axes[0], _cluster_handles, _cn_handles, fig=fig)

            else:
                # Shared single y-axis
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                _ymin_val = (ymin if isinstance(ymin, (int, float))
                             else (ymin[0] if ymin else None))
                _ymax_val = (ymax if isinstance(ymax, (int, float))
                             else (ymax[0] if ymax else None))
                _title = panel_title if show_panel_titles else None
                if show_panel_titles and title != 'Multi-System Coordination Comparison':
                    _title = title
                _draw_shared_panel(ax, systems_info, resolved_colors_map,
                                   cn_vals_sorted, resolved_cn_hatches,
                                   ymin_val=_ymin_val, ymax_val=_ymax_val,
                                   show_ylabel_flag=True, panel_title=_title)

                # Build legend handles before tight_layout
                _cluster_handles, _cn_handles = [], []
                if show_legend or show_cn_legend:
                    all_cids_flat = []
                    for sn in system_names:
                        for cid in systems_info[sn]['cluster_ids']:
                            if cid not in all_cids_flat:
                                all_cids_flat.append(cid)
                    _cluster_handles = [
                        Patch(facecolor=_default_colors[i % len(_default_colors)],
                              edgecolor=edgecolor, alpha=bar_alpha,
                              label=(legend_labels[i] if legend_labels and i < len(legend_labels)
                                     else f'C{cid}'))
                        for i, cid in enumerate(all_cids_flat)
                    ]
                    _cn_handles = [
                        Patch(facecolor='white', edgecolor=edgecolor,
                              hatch=resolved_cn_hatches.get(cv, ''), label=f'{cv}')
                        for cv in cn_vals_sorted
                    ]

                plt.tight_layout()
                _add_legends(ax, _cluster_handles, _cn_handles, fig=fig)

        # ================================================================== #
        #               MULTI-KEY – ONE FIGURE, n_panels COLUMNS
        # ================================================================== #
        else:
            # Normalise per-panel ymin / ymax
            if ymin is None or isinstance(ymin, (int, float)):
                ymin_per_panel = [ymin] * n_panels
            else:
                ymin_per_panel = list(ymin) + [None] * max(0, n_panels - len(ymin))
            if ymax is None or isinstance(ymax, (int, float)):
                ymax_per_panel = [ymax] * n_panels
            else:
                ymax_per_panel = list(ymax) + [None] * max(0, n_panels - len(ymax))

            _layout = 'constrained' if shared_y_across_panels else None
            fig, axes = plt.subplots(1, n_panels, figsize=figsize,
                                     sharey=shared_y_across_panels,
                                     layout=_layout,
                                     gridspec_kw={'wspace': panel_spacing})
            if n_panels == 1:
                axes = [axes]

            # Pre-compute all systems_info to optionally share y across panels
            _all_computed = [_compute_systems_info(k) for k in coord_keys]
            all_systems_info = [d[0] for d in _all_computed]
            all_cn_vals_per_panel = [d[1] for d in _all_computed]
            all_resolved_cn_hatches = [_resolve_cn_hatches(cvs) for cvs in all_cn_vals_per_panel]
            all_resolved = [_resolved_colors(si) for si in all_systems_info]

            # Optionally compute a global y-max
            global_ymax = None
            if shared_y_across_panels:
                # User-supplied ymax takes precedence; fall back to data-driven max
                _user_ymax = (ymax if isinstance(ymax, (int, float))
                              else (ymax[0] if ymax else None))
                if _user_ymax is not None:
                    global_ymax = _user_ymax
                else:
                    all_maxes = []
                    for si in all_systems_info:
                        for sn in system_names:
                            for cid in si[sn]['mean_cns']:
                                all_maxes.append(si[sn]['mean_cns'][cid]
                                                 + si[sn]['std_cns'].get(cid, 0))
                    global_ymax = max(all_maxes) * 1.15 if all_maxes else None

            for pidx, (key, ax) in enumerate(zip(coord_keys, axes)):
                systems_info = all_systems_info[pidx]
                resolved_colors_map = all_resolved[pidx]
                cn_vals_sorted = all_cn_vals_per_panel[pidx]
                resolved_cn_hatches = all_resolved_cn_hatches[pidx]
                panel_label = coord_key_labels.get(key, f'Panel {pidx + 1}')
                panel_title = panel_label if show_panel_titles else None

                _ymax_val = global_ymax if shared_y_across_panels else ymax_per_panel[pidx]
                _ymin_val = ymin_per_panel[pidx]

                _draw_shared_panel(ax, systems_info, resolved_colors_map,
                                   cn_vals_sorted, resolved_cn_hatches,
                                   ymin_val=_ymin_val, ymax_val=_ymax_val,
                                   show_ylabel_flag=(pidx == 0),
                                   panel_title=panel_title)

                # Remove left spine for non-leftmost panels
                if pidx > 0:
                    ax.tick_params(labelleft=False)
                    ax.set_ylabel('')

                # Shared y across panels: synchronise after drawing
                if shared_y_across_panels and global_ymax is not None:
                    ax.set_ylim(bottom=0.0, top=global_ymax)

            if show_suptitle:
                fig.suptitle(title, fontsize=title_fontsize + 1,
                             fontweight=title_fontweight, y=1.02)

            # Build legend handles before tight_layout
            _cluster_handles_multi, _cn_handles_multi = [], []
            if show_legend or show_cn_legend:
                all_cids_flat = []
                for si in all_systems_info:
                    for sn in system_names:
                        for cid in si[sn]['cluster_ids']:
                            if cid not in all_cids_flat:
                                all_cids_flat.append(cid)
                for i, cid in enumerate(all_cids_flat):
                    color = _default_colors[i % len(_default_colors)]
                    if colors_per_system:
                        for sn in system_names:
                            sc = colors_per_system.get(sn, [])
                            if sc and i < len(sc):
                                color = sc[i]
                                break
                    label = (legend_labels[i] if legend_labels and i < len(legend_labels)
                             else f'C{cid}')
                    _cluster_handles_multi.append(Patch(facecolor=color, edgecolor=edgecolor,
                                                        alpha=bar_alpha, label=label))
                global_cn_vals = sorted(set(
                    cn for cvs in all_cn_vals_per_panel for cn in cvs
                ))
                global_cn_hatches = _resolve_cn_hatches(global_cn_vals)
                _cn_handles_multi = [
                    Patch(facecolor='white', edgecolor=edgecolor,
                          hatch=global_cn_hatches.get(cv, ''), label=f'{cv}')
                    for cv in global_cn_vals
                ]

            # No tight_layout here — it would override the user-set wspace/panel_spacing.
            # The gridspec wspace is already correct from fig creation.
            # constrained_layout handles the shared_y case.
            _add_legends(axes[0], _cluster_handles_multi, _cn_handles_multi, fig=fig)

        # ------------------------------------------------------------------ #
        # Save
        # ------------------------------------------------------------------ #
        if save_fig and save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
            print(f"✓ Saved coordination comparison plot: {save_path}")

        return fig

    def plot_energy_decomposition(self,
                                 energy_data: Dict,
                                 cluster_stats: Optional[Dict] = None,
                                 groups_to_plot: Optional[List[Tuple[str, str]]] = None,
                                 figsize: Tuple[float, float] = (12, 6),
                                 colors: Optional[Dict[str, str]] = None,
                                 show_title: bool = True,
                                 title: Optional[str] = None,
                                 title_fontsize: int = 14,
                                 title_fontweight: str = 'bold',
                                 xlabel: str = 'Cluster ID',
                                 ylabel: str = 'Energy (kJ/mol)',
                                 label_fontsize: int = 12,
                                 label_fontweight: str = 'bold',
                                 tick_fontsize: int = 10,
                                 show_components: bool = True,
                                 show_total: bool = True,
                                 stacked: bool = False,
                                 bar_width: float = 0.25,
                                 bar_alpha: float = 0.8,
                                 edgecolor: str = 'black',
                                 edgewidth: float = 1.2,
                                 show_errorbar: bool = True,
                                 errorbar_capsize: float = 4,
                                 show_legend: bool = True,
                                 legend_loc: str = 'best',
                                 legend_fontsize: int = 10,
                                 legend_ncol: int = 1,
                                 grid: bool = True,
                                 grid_alpha: float = 0.3,
                                 grid_linestyle: str = '--',
                                 save_path: Optional[str] = None,
                                 dpi: int = 300,
                                 bbox_inches: str = 'tight') -> plt.Figure:
        """
        Plot energy decomposition as bar chart.
        
        Shows vdW (LJ-SR) and Coulomb (Coul-SR) components per cluster.
        Can show side-by-side or stacked bars.
        
        Parameters
        ----------
        energy_data : dict
            From compute_energy_decomposition() or compute_per_cluster_energies()
        cluster_stats : dict, optional
            Pre-computed statistics with stderr (if None, computed from energy_data)
        groups_to_plot : list of tuple, optional
            Subset of groups to plot (plots all if None)
        figsize : tuple, default=(12, 6)
            Figure size
        colors : dict, optional
            Component colors: {'Coul-SR': 'red', 'LJ-SR': 'blue', 'Total': 'purple'}
        
        Bar options:
        show_components : bool, default=True
            Show Coul-SR and LJ-SR bars
        show_total : bool, default=True
            Show Total energy bar
        stacked : bool, default=False
            Stack components (only if show_components=True)
        bar_width : float, default=0.25
            Width of bars
        show_errorbar : bool, default=True
            Show error bars (requires stderr in cluster_stats)
        
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        
        Example
        -------
        >>> cluster_stats = analyzer.compute_per_cluster_energies(energy_data)
        >>> fig = plotter.plot_energy_decomposition(
        ...     energy_data,
        ...     cluster_stats=cluster_stats,
        ...     stacked=True,
        ...     save_path='energy_decomposition.png'
        ... )
        """
        # Compute statistics if not provided
        if cluster_stats is None:
            cluster_stats = self.analyzer.compute_per_cluster_energies(energy_data)
        
        cluster_ids = sorted(energy_data.keys())
        
        # Determine which groups to plot
        if groups_to_plot is None:
            # Get all unique group pairs
            groups_to_plot = []
            for cid in cluster_ids:
                for pair in energy_data[cid]:
                    if pair not in groups_to_plot:
                        groups_to_plot.append(pair)
        
        n_groups = len(groups_to_plot)
        n_clusters = len(cluster_ids)
        
        # Default colors
        if colors is None:
            colors = {
                'Coul-SR': '#d62728',  # Red
                'LJ-SR': '#1f77b4',    # Blue
                'Total': '#9467bd'     # Purple
            }
        
        # Create subplots for each group pair
        fig, axes = plt.subplots(1, n_groups, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        x = np.arange(n_clusters)
        
        for ax_idx, pair in enumerate(groups_to_plot):
            ax = axes[ax_idx]
            g1, g2 = pair
            
            # Collect data
            coul_vals = []
            coul_errs = []
            lj_vals = []
            lj_errs = []
            total_vals = []
            total_errs = []
            
            for cid in cluster_ids:
                if pair not in cluster_stats[cid]:
                    coul_vals.append(0)
                    coul_errs.append(0)
                    lj_vals.append(0)
                    lj_errs.append(0)
                    total_vals.append(0)
                    total_errs.append(0)
                    continue
                
                stats = cluster_stats[cid][pair]
                coul_vals.append(stats.get('Coul-SR', 0))
                lj_vals.append(stats.get('LJ-SR', 0))
                total_vals.append(stats.get('Total', 0))
                
                if show_errorbar and 'stderr' in stats:
                    stderr = stats['stderr']
                    coul_errs.append(stderr.get('Coul-SR', 0))
                    lj_errs.append(stderr.get('LJ-SR', 0))
                    total_errs.append(stderr.get('Total', 0))
                else:
                    coul_errs.append(0)
                    lj_errs.append(0)
                    total_errs.append(0)
            
            coul_vals = np.array(coul_vals)
            lj_vals = np.array(lj_vals)
            total_vals = np.array(total_vals)
            coul_errs = np.array(coul_errs)
            lj_errs = np.array(lj_errs)
            total_errs = np.array(total_errs)
            
            # Plot bars
            if stacked and show_components:
                # Stacked bars
                ax.bar(x, lj_vals, bar_width, label='LJ-SR',
                      color=colors['LJ-SR'], alpha=bar_alpha,
                      edgecolor=edgecolor, linewidth=edgewidth)
                ax.bar(x, coul_vals, bar_width, bottom=lj_vals, label='Coul-SR',
                      color=colors['Coul-SR'], alpha=bar_alpha,
                      edgecolor=edgecolor, linewidth=edgewidth)
                
                if show_errorbar:
                    ax.errorbar(x, total_vals, yerr=total_errs, fmt='none',
                              ecolor='black', capsize=errorbar_capsize,
                              elinewidth=1.5, alpha=0.7)
            
            else:
                # Side-by-side bars
                offset = 0
                
                if show_components:
                    # LJ-SR
                    pos_lj = x + offset - bar_width
                    ax.bar(pos_lj, lj_vals, bar_width, label='LJ-SR',
                          color=colors['LJ-SR'], alpha=bar_alpha,
                          edgecolor=edgecolor, linewidth=edgewidth)
                    if show_errorbar:
                        ax.errorbar(pos_lj, lj_vals, yerr=lj_errs, fmt='none',
                                  ecolor='black', capsize=errorbar_capsize,
                                  elinewidth=1.2, alpha=0.7)
                    
                    # Coul-SR
                    pos_coul = x + offset
                    ax.bar(pos_coul, coul_vals, bar_width, label='Coul-SR',
                          color=colors['Coul-SR'], alpha=bar_alpha,
                          edgecolor=edgecolor, linewidth=edgewidth)
                    if show_errorbar:
                        ax.errorbar(pos_coul, coul_vals, yerr=coul_errs, fmt='none',
                                  ecolor='black', capsize=errorbar_capsize,
                                  elinewidth=1.2, alpha=0.7)
                    
                    offset += bar_width
                
                if show_total:
                    pos_total = x + offset + bar_width
                    ax.bar(pos_total, total_vals, bar_width, label='Total',
                          color=colors['Total'], alpha=bar_alpha,
                          edgecolor=edgecolor, linewidth=edgewidth)
                    if show_errorbar:
                        ax.errorbar(pos_total, total_vals, yerr=total_errs, fmt='none',
                                  ecolor='black', capsize=errorbar_capsize,
                                  elinewidth=1.2, alpha=0.7)
            
            # Formatting
            ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
            if ax_idx == 0:
                ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
            
            ax.set_title(f"{g1} - {g2}", fontsize=title_fontsize - 2, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f"{cid}" for cid in cluster_ids])
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            
            if grid:
                ax.grid(alpha=grid_alpha, linestyle=grid_linestyle, axis='y', zorder=0)
            
            if show_legend and ax_idx == n_groups - 1:
                ax.legend(loc=legend_loc, fontsize=legend_fontsize, ncol=legend_ncol)
        
        if show_title and title:
            fig.suptitle(title, fontsize=title_fontsize, fontweight=title_fontweight, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_energy_time_series(self,
                               energy_data: Dict,
                               cluster_id: int,
                               group_pair: Tuple[str, str],
                               figsize: Tuple[float, float] = (12, 5),
                               colors: Optional[Dict[str, str]] = None,
                               show_title: bool = True,
                               title: Optional[str] = None,
                               xlabel: str = 'Time (ps)',
                               ylabel: str = 'Energy (kJ/mol)',
                               label_fontsize: int = 12,
                               label_fontweight: str = 'bold',
                               tick_fontsize: int = 10,
                               linewidth: float = 1.5,
                               alpha: float = 0.7,
                               show_mean: bool = True,
                               mean_linestyle: str = '--',
                               mean_linewidth: float = 2,
                               show_legend: bool = True,
                               legend_loc: str = 'best',
                               legend_fontsize: int = 10,
                               grid: bool = True,
                               grid_alpha: float = 0.3,
                               save_path: Optional[str] = None,
                               dpi: int = 300) -> plt.Figure:
        """
        Plot energy time series for specific cluster.
        
        Shows fluctuations of vdW and Coulomb over trajectory.
        Useful for convergence checking.
        
        Parameters
        ----------
        energy_data : dict
            From compute_energy_decomposition()
        cluster_id : int
            Cluster to plot
        group_pair : tuple of str
            Which group pair: ('CIP', 'MMT')
        show_mean : bool, default=True
            Show mean as horizontal line
        
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if cluster_id not in energy_data:
            raise ValueError(f"Cluster {cluster_id} not in energy_data")
        
        if group_pair not in energy_data[cluster_id]:
            raise ValueError(f"Group pair {group_pair} not found for cluster {cluster_id}")
        
        pair_data = energy_data[cluster_id][group_pair]
        
        # Default colors
        if colors is None:
            colors = {
                'Coul-SR': '#d62728',
                'LJ-SR': '#1f77b4',
                'Total': '#9467bd'
            }
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get time
        time = pair_data.get('Time', np.arange(len(pair_data.get('Total', []))))
        
        # Plot components
        for comp in ['LJ-SR', 'Coul-SR', 'Total']:
            if comp not in pair_data:
                continue
            
            values = pair_data[comp]
            ax.plot(time, values, label=comp, color=colors.get(comp, None),
                   linewidth=linewidth, alpha=alpha)
            
            if show_mean:
                mean_val = np.mean(values)
                ax.axhline(y=mean_val, color=colors.get(comp, None),
                          linestyle=mean_linestyle, linewidth=mean_linewidth,
                          alpha=0.6, label=f"{comp} mean")
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        if grid:
            ax.grid(alpha=grid_alpha, zorder=0)
        
        if show_legend:
            ax.legend(loc=legend_loc, fontsize=legend_fontsize)
        
        if show_title:
            if title is None:
                g1, g2 = group_pair
                title = f"Energy Time Series: {g1}-{g2} (Cluster {cluster_id})"
            ax.set_title(title, fontsize=label_fontsize + 2, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_energy_heatmap(self,
                           cluster_stats: Dict,
                           component: str = 'Total',
                           figsize: Tuple[float, float] = (10, 6),
                           cmap: str = 'RdYlBu_r',
                           show_values: bool = True,
                           value_format: str = '{:.1f}',
                           value_fontsize: int = 9,
                           show_title: bool = True,
                           title: Optional[str] = None,
                           xlabel: str = 'Energy Group Pair',
                           ylabel: str = 'Cluster ID',
                           label_fontsize: int = 12,
                           label_fontweight: str = 'bold',
                           tick_fontsize: int = 10,
                           show_colorbar: bool = True,
                           cbar_label: str = 'Energy (kJ/mol)',
                           cbar_fontsize: int = 11,
                           save_path: Optional[str] = None,
                           dpi: int = 300) -> plt.Figure:
        """
        Plot energy heatmap: clusters vs group pairs.
        
        Color intensity shows interaction strength.
        Identifies which interactions dominate in each cluster.
        
        Parameters
        ----------
        cluster_stats : dict
            From compute_per_cluster_energies()
        component : str, default='Total'
            Which component: 'Coul-SR', 'LJ-SR', 'Total'
        cmap : str, default='RdYlBu_r'
            Colormap (reversed: blue=favorable, red=unfavorable)
        show_values : bool, default=True
            Annotate cells with values
        
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        cluster_ids = sorted(cluster_stats.keys())
        
        # Get all group pairs
        group_pairs = []
        for cid in cluster_ids:
            for pair in cluster_stats[cid]:
                if pair not in group_pairs:
                    group_pairs.append(pair)
        
        n_clusters = len(cluster_ids)
        n_pairs = len(group_pairs)
        
        # Build matrix
        matrix = np.zeros((n_clusters, n_pairs))
        
        for i, cid in enumerate(cluster_ids):
            for j, pair in enumerate(group_pairs):
                if pair in cluster_stats[cid] and component in cluster_stats[cid][pair]:
                    matrix[i, j] = cluster_stats[cid][pair][component]
                else:
                    matrix[i, j] = np.nan
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # Colorbar
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(cbar_label, fontsize=cbar_fontsize, fontweight='bold')
            cbar.ax.tick_params(labelsize=tick_fontsize)
        
        # Annotate with values
        if show_values:
            for i in range(n_clusters):
                for j in range(n_pairs):
                    if not np.isnan(matrix[i, j]):
                        text = ax.text(j, i, value_format.format(matrix[i, j]),
                                     ha='center', va='center',
                                     fontsize=value_fontsize,
                                     color='white' if np.abs(matrix[i, j]) > np.nanmax(np.abs(matrix)) / 2 else 'black')
        
        # Formatting
        ax.set_xticks(np.arange(n_pairs))
        ax.set_yticks(np.arange(n_clusters))
        ax.set_xticklabels([f"{g1}-{g2}" for g1, g2 in group_pairs], rotation=45, ha='right')
        ax.set_yticklabels([f"Cluster {cid}" for cid in cluster_ids])
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        
        if show_title:
            if title is None:
                title = f"Energy Heatmap: {component}"
            ax.set_title(title, fontsize=label_fontsize + 2, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_energy_correlation(self,
                               correlation_data: Dict,
                               figsize: Tuple[float, float] = (8, 6),
                               colors: Optional[Union[Dict[int, str], List[str]]] = None,
                               marker_size: float = 150,
                               marker_alpha: float = 0.7,
                               edgecolor: str = 'black',
                               edgewidth: float = 1.5,
                               show_labels: bool = True,
                               label_fontsize: int = 10,
                               show_regression: bool = True,
                               regression_color: str = 'red',
                               regression_linestyle: str = '--',
                               regression_linewidth: float = 2,
                               show_title: bool = True,
                               title: Optional[str] = None,
                               xlabel: str = 'Interaction Energy (kJ/mol)',
                               ylabel: str = 'ΔG (kJ/mol)',
                               label_fontsize_axes: int = 12,
                               label_fontweight: str = 'bold',
                               tick_fontsize: int = 10,
                               show_stats: bool = True,
                               stats_loc: str = 'upper left',
                               stats_fontsize: int = 10,
                               grid: bool = True,
                               grid_alpha: float = 0.3,
                               save_path: Optional[str] = None,
                               dpi: int = 300) -> plt.Figure:
        """
        Plot correlation between RMSD/energy and free energy.
        
        Scatter plot colored by cluster showing relationship between
        structural/energetic properties and thermodynamic stability.
        
        Parameters
        ----------
        correlation_data : dict
            From correlate_energy_free_energy()
        colors : dict or list, optional
            Cluster colors
        show_regression : bool, default=True
            Show linear regression line
        show_stats : bool, default=True
            Show correlation statistics (r, p-value)
        
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        cluster_data = correlation_data['cluster_data']
        cluster_ids = sorted(cluster_data.keys())
        
        energies = [cluster_data[cid]['energy'] for cid in cluster_ids]
        delta_Gs = [cluster_data[cid]['delta_G'] for cid in cluster_ids]
        
        # Handle colors
        if colors is None:
            colors_list = plt.cm.viridis(np.linspace(0.2, 0.9, len(cluster_ids)))
        elif isinstance(colors, dict):
            colors_list = [colors.get(cid, 'gray') for cid in cluster_ids]
        else:
            colors_list = colors
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        for i, cid in enumerate(cluster_ids):
            ax.scatter(energies[i], delta_Gs[i], s=marker_size,
                      color=colors_list[i], alpha=marker_alpha,
                      edgecolors=edgecolor, linewidths=edgewidth,
                      label=f"Cluster {cid}", zorder=5)
        
        # Cluster labels
        if show_labels:
            for i, cid in enumerate(cluster_ids):
                ax.text(energies[i], delta_Gs[i], f" {cid}",
                       fontsize=label_fontsize, ha='left', va='center')
        
        # Regression line
        if show_regression:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(energies, delta_Gs)
            x_fit = np.array([min(energies), max(energies)])
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color=regression_color,
                   linestyle=regression_linestyle, linewidth=regression_linewidth,
                   alpha=0.8, zorder=3)
        
        # Statistics annotation
        if show_stats:
            pearson_r = correlation_data['pearson_r']
            pearson_p = correlation_data['pearson_p']
            stats_text = f"Pearson r = {pearson_r:.3f}\np = {pearson_p:.3e}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=stats_fontsize, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=label_fontsize_axes, fontweight=label_fontweight)
        ax.set_ylabel(ylabel, fontsize=label_fontsize_axes, fontweight=label_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        if grid:
            ax.grid(alpha=grid_alpha, zorder=0)
        
        if show_title:
            if title is None:
                title = "Energy-Free Energy Correlation"
            ax.set_title(title, fontsize=label_fontsize_axes + 2, fontweight='bold')
        
        # Legend
        ax.legend(loc='best', fontsize=9, ncol=2)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_coordination_comparison(self, coord_key: str,
                                    cluster_ids: Optional[List[int]] = None,
                                    figsize: Tuple[float, float] = (12, 5),
                                    colors: Optional[List[str]] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot coordination number comparison between clusters.
        
        Parameters
        ----------
        coord_key : str
            Key for coordination data (format: 'center__neighbor__cutoff')
        cluster_ids : list of int, optional
            Clusters to plot
        figsize : tuple
            Figure size
        colors : list of str, optional
            Colors for each cluster
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        
        Example
        -------
        >>> plotter.plot_coordination_comparison('resname CIP and name O1 O3__resname NA__3.5')
        """
        if not hasattr(self.analyzer, 'coordination_data'):
            raise ValueError("No coordination data found. Run compute_coordination_numbers() first.")
        
        if coord_key not in self.analyzer.coordination_data:
            raise ValueError(f"Coordination key '{coord_key}' not found.")
        
        coord_results = self.analyzer.coordination_data[coord_key]
        
        if cluster_ids is None:
            cluster_ids = list(coord_results.keys())
        
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Time series / histogram
        all_bin_centers = []
        for i, cluster_id in enumerate(cluster_ids):
            data = coord_results[cluster_id]
            cn = data['coordination']
            
            # Histogram
            hist, bins = np.histogram(cn, bins=range(int(cn.min()), int(cn.max())+2))
            bin_centers = bins[:-1]  # Use bin left edges for integer CN values
            all_bin_centers.extend(bin_centers)
            ax1.plot(bin_centers, hist, 'o-',
                    label=f"Cluster {cluster_id}",
                    color=colors[i], lw=2, markersize=6, alpha=0.8)
        
        ax1.set_xlabel('Coordination Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('CN Distribution', fontsize=13, fontweight='bold')
        ax1.legend(frameon=True, fontsize=10)
        ax1.grid(alpha=0.3, ls='--')
        
        # Set x-axis limits with proper margins to show all data points
        if all_bin_centers:
            min_cn = min(all_bin_centers)
            max_cn = max(all_bin_centers)
            margin = 0.5  # Half a bin width margin
            ax1.set_xlim(left=max(0, min_cn - margin), right=max_cn + margin)
        
        # Bar plot for mean CN
        means = [coord_results[cid]['mean_cn'] for cid in cluster_ids]
        stds = [coord_results[cid]['std_cn'] for cid in cluster_ids]
        x_pos = np.arange(len(cluster_ids))
        
        # Create asymmetric error bars (prevent negative coordination numbers)
        # Lower error clipped so mean - lower_error >= 0
        lower_errors = [min(m, s) for m, s in zip(means, stds)]
        upper_errors = stds
        asymmetric_error = [lower_errors, upper_errors]
        
        bars = ax2.bar(x_pos, means, yerr=asymmetric_error, 
                      color=colors, alpha=0.7, capsize=5,
                      edgecolor='black', linewidth=1.5)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([])  # No labels - legend identifies clusters by color
        ax2.set_ylabel('Mean Coordination Number', fontsize=12, fontweight='bold')
        ax2.set_title('Average CN', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3, ls='--', axis='y')
        ax2.set_ylim(bottom=0)  # Coordination numbers cannot be negative
        
        # Add values on bars
        for i, (m, s) in enumerate(zip(means, stds)):
            ax2.text(i, m + s, f'{m:.2f}±{s:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add selection info with friendly names
        data_sample = coord_results[cluster_ids[0]]
        cutoff = data_sample['cutoff']
        center_sel_raw = data_sample['center_selection']
        neighbor_sel_raw = data_sample['neighbor_selection']
        
        # Try to get friendly names
        center_name = self.analyzer.get_selection_name(center_sel_raw) or center_sel_raw
        neighbor_name = self.analyzer.get_selection_name(neighbor_sel_raw) or neighbor_sel_raw
        
        fig.suptitle(f'{center_name} - {neighbor_name} (cutoff={cutoff:.1f}Å)', 
                    fontsize=10, y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_coordination_comparison_styled(self, coord_key: str,
                                           cluster_ids: Optional[List[int]] = None,
                                           # Plot style
                                           distribution_style: str = 'bar',  # 'line' or 'bar'
                                           normalize_distribution: bool = True,
                                           # Figure layout
                                           figsize: Tuple[float, float] = (14, 6),
                                           layout: str = 'side-by-side',  # 'side-by-side' or 'stacked'
                                           # Colors and styling
                                           colors: Optional[Union[List[str], dict]] = None,
                                           colormap: str = 'Set2',
                                           bar_alpha: float = 0.8,
                                           edgecolor: str = 'black',
                                           edgewidth: float = 1.5,
                                           hatches: Optional[List[str]] = None,
                                           # Distribution plot specifics
                                           bar_width: float = 0.8,
                                           line_width: float = 2.5,
                                           marker_size: int = 8,
                                           marker_style: str = 'o',
                                           # Error bars
                                           show_error_bars: bool = True,
                                           errorbar_capsize: int = 5,
                                           errorbar_capthick: float = 1.5,
                                           # Value labels
                                           show_values: bool = True,
                                           value_fontsize: int = 10,
                                           value_format: str = '{:.2f}',
                                           value_fontweight: str = 'bold',
                                           # Font controls
                                           title_fontsize: int = 14,
                                           title_fontweight: str = 'bold',
                                           label_fontsize: int = 12,
                                           label_fontweight: str = 'bold',
                                           tick_fontsize: int = 11,
                                           legend_fontsize: int = 10,
                                           legend_fontweight: str = 'bold',
                                           # Legend
                                           show_legend: bool = True,
                                           legend_loc: str = 'best',
                                           legend_framealpha: float = 0.9,
                                           # Grid
                                           show_grid: bool = True,
                                           grid_alpha: float = 0.3,
                                           grid_style: str = '--',
                                           grid_width: float = 0.7,
                                           # Axis labels
                                           xlabel_dist: Optional[str] = None,
                                           ylabel_dist: Optional[str] = None,
                                           xlabel_mean: Optional[str] = None,
                                           ylabel_mean: Optional[str] = None,
                                           # Titles
                                           title_dist: Optional[str] = None,
                                           title_mean: Optional[str] = None,
                                           suptitle: Optional[str] = None,
                                           # Axis limits
                                           ylim_mean: Optional[Tuple[float, float]] = None,
                                           # Export
                                           save_path: Optional[str] = None,
                                           save_fig: bool = False,
                                           save_individual_figures: bool = False,
                                           individual_save_dir: str = './coordination_figs',
                                           individual_figsize: Tuple[float, float] = (8, 6),
                                           dpi: int = 300,
                                           bbox_inches: str = 'tight',
                                           transparent_bg: bool = False) -> plt.Figure:
        """
        Enhanced coordination number comparison plot with professional styling.
        
        Combines distribution visualization (line or bar) with mean comparison,
        using styling inspired by ClayOrganicIonWaterAnalysisPlotter.
        
        Parameters
        ----------
        coord_key : str
            Key for coordination data (format: 'center__neighbor__cutoff')
        cluster_ids : list of int, optional
            Clusters to plot. If None, plots all clusters.
        distribution_style : str, default='bar'
            Style for distribution plot: 'line' (line plot) or 'bar' (histogram bars)
        normalize_distribution : bool, default=True
            If True, normalize each cluster's distribution to probability (sum=1).
            If False, show raw frequency counts. Normalization is essential for
            comparing clusters with different numbers of frames.
        layout : str, default='side-by-side'
            Figure layout: 'side-by-side' (2 panels) or 'stacked' (2 rows)
        colors : list or dict, optional
            Colors for each cluster. Can be list or dict mapping cluster_id to color.
            If None, uses colormap.
        colormap : str, default='Set2'
            Matplotlib colormap name if colors not provided
        bar_alpha : float, default=0.8
            Transparency for bar plots
        edgecolor : str, default='black'
            Edge color for bars
        edgewidth : float, default=1.5
            Edge width for bars
        hatches : list of str, optional
            Hatch patterns for each cluster (e.g., ['///', '\\\\\\', 'xxx'])
        bar_width : float, default=0.8
            Width of bars in histogram
        line_width : float, default=2.5
            Width of lines in line plot
        marker_size : int, default=8
            Size of markers in line plot
        marker_style : str, default='o'
            Marker style for line plot
        show_error_bars : bool, default=True
            Show error bars on mean plot
        errorbar_capsize : int, default=5
            Size of error bar caps
        errorbar_capthick : float, default=1.5
            Thickness of error bar caps
        show_values : bool, default=True
            Show values on top of mean bars
        value_fontsize : int, default=10
            Font size for value labels
        value_format : str, default='{:.2f}'
            Format string for value labels
        value_fontweight : str, default='bold'
            Font weight for value labels
        title_fontsize : int, default=14
            Font size for subplot titles
        title_fontweight : str, default='bold'
            Font weight for subplot titles
        label_fontsize : int, default=12
            Font size for axis labels
        label_fontweight : str, default='bold'
            Font weight for axis labels
        tick_fontsize : int, default=11
            Font size for tick labels
        legend_fontsize : int, default=10
            Font size for legend
        legend_fontweight : str, default='bold'
            Font weight for legend
        show_legend : bool, default=True
            Show legend
        legend_loc : str, default='best'
            Legend location
        legend_framealpha : float, default=0.9
            Legend frame transparency
        show_grid : bool, default=True
            Show grid
        grid_alpha : float, default=0.3
            Grid transparency
        grid_style : str, default='--'
            Grid line style
        grid_width : float, default=0.7
            Grid line width
        xlabel_dist, ylabel_dist : str, optional
            Custom axis labels for distribution plot
        xlabel_mean, ylabel_mean : str, optional
            Custom axis labels for mean plot
        title_dist, title_mean : str, optional
            Custom titles for subplots
        suptitle : str, optional
            Overall figure title. If None, auto-generates from data.
        ylim_mean : tuple of float, optional
            Y-axis limits for mean CN plot (min, max). Use this to prevent label
            overlap with plot edges. If None, auto-scales with bottom=0.
            Example: ylim_mean=(0, 8) to set max y-value at 8.
        save_path : str, optional
            Path to save combined figure (used with save_fig=True)
        save_fig : bool, default=False
            Whether to save the combined figure to save_path
        save_individual_figures : bool, default=False
            Whether to save distribution and mean plots as separate PNG files
        individual_save_dir : str, default='./coordination_figs'
            Directory to save individual figures. If not explicitly set and save_path
            is provided, uses the same directory as save_path. Created if doesn't exist.
        individual_figsize : tuple of float, default=(8, 6)
            Figure size for individual plots (width, height) in inches
        dpi : int, default=300
            Resolution for saved figures
        bbox_inches : str, default='tight'
            Bounding box for saved figures
        transparent_bg : bool, default=False
            Transparent background for saved figures
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        
        Examples
        --------
        >>> # Bar histogram style with normalized probability
        >>> fig = plotter.plot_coordination_comparison_styled(
        ...     coord_key=key,
        ...     distribution_style='bar',
        ...     normalize_distribution=True,  # Show probability instead of raw counts
        ...     colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
        ...     hatches=['///', '\\\\\\\\\\\\', 'xxx', '...', '|||'],
        ...     show_values=True
        ... )
        >>> 
        >>> # Line plot style with stacked layout
        >>> fig = plotter.plot_coordination_comparison_styled(
        ...     coord_key=key,
        ...     distribution_style='line',
        ...     layout='stacked',
        ...     marker_style='s',
        ...     line_width=3
        ... )
        >>> 
        >>> # Save individual figures with custom y-limit to prevent label overlap
        >>> fig = plotter.plot_coordination_comparison_styled(
        ...     coord_key=key,
        ...     ylim_mean=(0, 8),  # Set max to 8 to prevent label overlap
        ...     save_individual_figures=True,
        ...     individual_save_dir='./rdf_oct_mg',
        ...     individual_figsize=(8, 6),
        ...     save_fig=True,
        ...     save_path='./combined_coordination.png',
        ...     dpi=600
        ... )
        """
        # Validate data
        if not hasattr(self.analyzer, 'coordination_data'):
            raise ValueError("No coordination data found. Run compute_coordination_numbers() first.")
        
        if coord_key not in self.analyzer.coordination_data:
            available_keys = list(self.analyzer.coordination_data.keys())
            raise ValueError(f"Coordination key '{coord_key}' not found. Available keys:\n" + 
                           "\n".join(f"  - {k}" for k in available_keys))
        
        coord_results = self.analyzer.coordination_data[coord_key]
        
        if cluster_ids is None:
            cluster_ids = sorted(coord_results.keys())
        
        n_clusters = len(cluster_ids)
        
        # Set up colors
        if colors is None:
            cmap = plt.cm.get_cmap(colormap)
            colors_list = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]
        elif isinstance(colors, dict):
            colors_list = [colors.get(cid, 'gray') for cid in cluster_ids]
        else:
            colors_list = list(colors)
        
        # Set up hatches
        if hatches is None:
            hatch_patterns = ['', '///', '\\\\\\\\\\\\', 'xxx', '...', '|||', '+++', 'ooo']
            hatches_list = [hatch_patterns[i % len(hatch_patterns)] for i in range(n_clusters)]
        else:
            hatches_list = list(hatches)
        
        # Create figure
        if layout == 'stacked':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        else:  # side-by-side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                          gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.3})
        
        # ============ PANEL 1: Distribution ============
        if distribution_style == 'bar':
            # Histogram as bars
            # Collect all CN data to determine bins
            all_cn = np.concatenate([coord_results[cid]['coordination'] for cid in cluster_ids])
            min_cn, max_cn = int(all_cn.min()), int(all_cn.max())
            bins = range(min_cn, max_cn + 2)
            bin_centers = np.array([(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)])
            
            # Calculate bar positions with grouping
            n_bins = len(bin_centers)
            group_width = bar_width * n_clusters * 1.1
            x_base = np.arange(n_bins) * group_width
            bar_offset = bar_width * (np.arange(n_clusters) - (n_clusters - 1) / 2)
            
            for i, cluster_id in enumerate(cluster_ids):
                cn = coord_results[cluster_id]['coordination']
                hist, _ = np.histogram(cn, bins=bins)
                
                # Normalize if requested
                if normalize_distribution:
                    hist = hist / len(cn)
                
                x_pos = x_base + bar_offset[i]
                ax1.bar(x_pos, hist, width=bar_width,
                       label=f'Cluster {cluster_id}',
                       color=colors_list[i], alpha=bar_alpha,
                       edgecolor=edgecolor, linewidth=edgewidth,
                       hatch=hatches_list[i])
            
            ax1.set_xticks(x_base)
            ax1.set_xticklabels([f'{int(bc)}' for bc in bin_centers], fontsize=tick_fontsize)
            
            # Set limits to show all bars with small margin
            first_bar_left = x_base[0] + bar_offset[0] - bar_width/2
            last_bar_right = x_base[-1] + bar_offset[-1] + bar_width/2
            margin = bar_width * 0.5
            ax1.set_xlim(left=first_bar_left - margin, right=last_bar_right + margin)
            
        else:  # line plot
            all_bin_centers = []
            for i, cluster_id in enumerate(cluster_ids):
                cn = coord_results[cluster_id]['coordination']
                hist, bins = np.histogram(cn, bins=range(int(cn.min()), int(cn.max()) + 2))
                bin_centers = bins[:-1]  # Use bin left edges for integer CN values
                all_bin_centers.extend(bin_centers)
                
                # Normalize if requested
                if normalize_distribution:
                    hist = hist / len(cn)
                
                ax1.plot(bin_centers, hist, marker=marker_style, 
                        label=f'Cluster {cluster_id}',
                        color=colors_list[i], linewidth=line_width,
                        markersize=marker_size, alpha=0.9)
            
            ax1.tick_params(axis='both', labelsize=tick_fontsize)
            
            # Set x-axis limits with proper margins to show all data points
            if all_bin_centers:
                min_cn = min(all_bin_centers)
                max_cn = max(all_bin_centers)
                margin = 0.5  # Half a bin width margin
                ax1.set_xlim(left=max(0, min_cn - margin), right=max_cn + margin)
        
        # Distribution plot labels
        ax1.set_xlabel(xlabel_dist or 'Coordination Number', 
                      fontsize=label_fontsize, fontweight=label_fontweight)
        default_ylabel = 'Probability' if normalize_distribution else 'Frequency'
        ax1.set_ylabel(ylabel_dist or default_ylabel, 
                      fontsize=label_fontsize, fontweight=label_fontweight)
        ax1.set_title(title_dist or 'CN Distribution', 
                     fontsize=title_fontsize, fontweight=title_fontweight)
        
        if show_grid:
            ax1.grid(True, alpha=grid_alpha, linestyle=grid_style, 
                    linewidth=grid_width, zorder=0)
        
        if show_legend:
            leg = ax1.legend(loc=legend_loc, framealpha=legend_framealpha, 
                           fontsize=legend_fontsize)
            for text in leg.get_texts():
                text.set_fontweight(legend_fontweight)
        
        # ============ PANEL 2: Mean Comparison ============
        means = [coord_results[cid]['mean_cn'] for cid in cluster_ids]
        stds = [coord_results[cid]['std_cn'] for cid in cluster_ids]
        x_pos = np.arange(n_clusters)
        
        # Create bars
        bars = ax2.bar(x_pos, means, 
                      color=colors_list, alpha=bar_alpha,
                      edgecolor=edgecolor, linewidth=edgewidth)
        
        # Add hatching
        for bar, hatch in zip(bars, hatches_list):
            bar.set_hatch(hatch)
        
        # Add error bars separately (for better control and asymmetry)
        if show_error_bars:
            lower_errors = [min(m, s) for m, s in zip(means, stds)]
            upper_errors = stds
            
            ax2.errorbar(x_pos, means, yerr=[lower_errors, upper_errors],
                        fmt='none', ecolor='black', elinewidth=1.5,
                        capsize=errorbar_capsize, capthick=errorbar_capthick, 
                        alpha=0.8, zorder=10)
        
        # Add value labels on bars
        if show_values:
            y_max = max(means) + max(stds) if means else 1
            value_offset = y_max * 0.02
            
            for i, (m, s) in enumerate(zip(means, stds)):
                label_y = m + s + value_offset if show_error_bars else m + value_offset
                ax2.text(i, label_y, value_format.format(m),
                        ha='center', va='bottom', 
                        fontsize=value_fontsize, fontweight=value_fontweight)
        
        # Mean plot labels
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([])  # No labels - legend identifies clusters by color
        ax2.set_ylabel(ylabel_mean or 'Mean CN', 
                      fontsize=label_fontsize, fontweight=label_fontweight)
        ax2.set_title(title_mean or 'Average CN', 
                     fontsize=title_fontsize, fontweight=title_fontweight)
        
        if show_grid:
            ax2.grid(True, alpha=grid_alpha, linestyle=grid_style, 
                    linewidth=grid_width, axis='y', zorder=0)
        
        # Set y-axis limits
        if ylim_mean:
            ax2.set_ylim(ylim_mean)
        else:
            ax2.set_ylim(bottom=0)  # CN cannot be negative
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        
        # ============ Overall Title ============
        if suptitle is None:
            # Auto-generate from data with friendly names
            data_sample = coord_results[cluster_ids[0]]
            cutoff = data_sample['cutoff']
            center_sel_raw = data_sample.get('center_selection', 'Center')
            neighbor_sel_raw = data_sample.get('neighbor_selection', 'Neighbor')
            
            # Try to get friendly names from reverse lookup
            center_name = self.analyzer.get_selection_name(center_sel_raw)
            neighbor_name = self.analyzer.get_selection_name(neighbor_sel_raw)
            
            # Use friendly name if available, otherwise shorten raw string
            if center_name:
                center_label = center_name
            elif len(center_sel_raw) > 40:
                center_label = center_sel_raw[:37] + '...'
            else:
                center_label = center_sel_raw
            
            if neighbor_name:
                neighbor_label = neighbor_name
            elif len(neighbor_sel_raw) > 40:
                neighbor_label = neighbor_sel_raw[:37] + '...'
            else:
                neighbor_label = neighbor_sel_raw
            
            suptitle = f'{center_label} - {neighbor_label} (cutoff={cutoff:.1f}Å)'
        
        if suptitle:
            fig.suptitle(suptitle, fontsize=title_fontsize, fontweight=title_fontweight, y=0.98)
        
        plt.tight_layout()
        
        # Save combined figure
        if save_fig and save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, 
                       transparent=transparent_bg)
            print(f"✓ Saved combined figure: {save_path}")
        elif save_path:  # Backward compatibility
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, 
                       transparent=transparent_bg)
            print(f"✓ Saved: {save_path}")
        
        # Save individual figures
        if save_individual_figures:
            import os
            
            # Use save_path directory if individual_save_dir not explicitly changed
            if individual_save_dir == './coordination_figs' and save_path:
                # Extract directory from save_path
                individual_save_dir = os.path.dirname(save_path) or '.'
            
            os.makedirs(individual_save_dir, exist_ok=True)
            
            # Generate base filename from coord_key
            base_name = coord_key.replace('__', '_')
            
            # Save distribution plot
            fig_dist, ax_dist = plt.subplots(figsize=individual_figsize)
            
            if distribution_style == 'bar':
                # Recreate bar histogram
                all_cn = np.concatenate([coord_results[cid]['coordination'] for cid in cluster_ids])
                min_cn, max_cn = int(all_cn.min()), int(all_cn.max())
                bins = range(min_cn, max_cn + 2)
                bin_centers = np.array([(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)])
                n_bins = len(bin_centers)
                group_width = bar_width * n_clusters * 1.1
                x_base = np.arange(n_bins) * group_width
                bar_offset = bar_width * (np.arange(n_clusters) - (n_clusters - 1) / 2)
                
                for i, cluster_id in enumerate(cluster_ids):
                    cn = coord_results[cluster_id]['coordination']
                    hist, _ = np.histogram(cn, bins=bins)
                    
                    # Normalize if requested
                    if normalize_distribution:
                        hist = hist / len(cn)
                    
                    x_positions = x_base + bar_offset[i]  # Don't overwrite x_pos
                    ax_dist.bar(x_positions, hist, width=bar_width,
                               label=f'Cluster {cluster_id}',
                               color=colors_list[i], alpha=bar_alpha,
                               edgecolor=edgecolor, linewidth=edgewidth,
                               hatch=hatches_list[i])
                
                ax_dist.set_xticks(x_base)
                ax_dist.set_xticklabels([f'{int(bc)}' for bc in bin_centers], fontsize=tick_fontsize)
                first_bar_left = x_base[0] + bar_offset[0] - bar_width/2
                last_bar_right = x_base[-1] + bar_offset[-1] + bar_width/2
                margin = bar_width * 0.5
                ax_dist.set_xlim(left=first_bar_left - margin, right=last_bar_right + margin)
            else:  # line plot
                for i, cluster_id in enumerate(cluster_ids):
                    cn = coord_results[cluster_id]['coordination']
                    hist, bins = np.histogram(cn, bins=range(int(cn.min()), int(cn.max()) + 2))
                    bin_centers = bins[:-1]
                    
                    # Normalize if requested
                    if normalize_distribution:
                        hist = hist / len(cn)
                    
                    ax_dist.plot(bin_centers, hist, marker=marker_style, 
                                label=f'Cluster {cluster_id}',
                                color=colors_list[i], linewidth=line_width,
                                markersize=marker_size, alpha=0.9)
                ax_dist.tick_params(axis='both', labelsize=tick_fontsize)
            
            ax_dist.set_xlabel(xlabel_dist or 'Coordination Number', 
                              fontsize=label_fontsize, fontweight=label_fontweight)
            default_ylabel = 'Probability' if normalize_distribution else 'Frequency'
            ax_dist.set_ylabel(ylabel_dist or default_ylabel, 
                              fontsize=label_fontsize, fontweight=label_fontweight)
            ax_dist.set_title(title_dist or 'CN Distribution', 
                             fontsize=title_fontsize, fontweight=title_fontweight)
            
            if show_grid:
                ax_dist.grid(True, alpha=grid_alpha, linestyle=grid_style, 
                            linewidth=grid_width, zorder=0)
            if show_legend:
                leg = ax_dist.legend(loc=legend_loc, framealpha=legend_framealpha, 
                                   fontsize=legend_fontsize)
                for text in leg.get_texts():
                    text.set_fontweight(legend_fontweight)
            
            plt.tight_layout()
            dist_path = os.path.join(individual_save_dir, f'{base_name}_distribution.png')
            fig_dist.savefig(dist_path, dpi=dpi, bbox_inches=bbox_inches, 
                           transparent=transparent_bg)
            print(f"✓ Saved distribution plot: {dist_path}")
            plt.close(fig_dist)
            
            # Save mean comparison plot
            fig_mean, ax_mean = plt.subplots(figsize=individual_figsize)
            
            # Recreate x_pos for mean plot (not corrupted from distribution loop)
            x_pos = np.arange(n_clusters)
            
            bars = ax_mean.bar(x_pos, means, 
                              color=colors_list, alpha=bar_alpha,
                              edgecolor=edgecolor, linewidth=edgewidth)
            for bar, hatch in zip(bars, hatches_list):
                bar.set_hatch(hatch)
            
            if show_error_bars:
                lower_errors = [min(m, s) for m, s in zip(means, stds)]
                upper_errors = stds
                ax_mean.errorbar(x_pos, means, yerr=[lower_errors, upper_errors],
                                fmt='none', ecolor='black', elinewidth=1.5,
                                capsize=errorbar_capsize, capthick=errorbar_capthick, 
                                alpha=0.8, zorder=10)
            
            if show_values:
                y_max = max(means) + max(stds) if means else 1
                value_offset = y_max * 0.02
                for i, (m, s) in enumerate(zip(means, stds)):
                    label_y = m + s + value_offset if show_error_bars else m + value_offset
                    ax_mean.text(i, label_y, value_format.format(m),
                                ha='center', va='bottom', 
                                fontsize=value_fontsize, fontweight=value_fontweight)
            
            ax_mean.set_xticks(x_pos)
            ax_mean.set_xticklabels([f'C{cid}' for cid in cluster_ids], fontsize=tick_fontsize)
            ax_mean.set_ylabel(ylabel_mean or 'Mean CN', 
                              fontsize=label_fontsize, fontweight=label_fontweight)
            ax_mean.set_title(title_mean or 'Average CN', 
                             fontsize=title_fontsize, fontweight=title_fontweight)
            
            if show_grid:
                ax_mean.grid(True, alpha=grid_alpha, linestyle=grid_style, 
                            linewidth=grid_width, axis='y', zorder=0)
            
            if ylim_mean:
                ax_mean.set_ylim(ylim_mean)
            else:
                ax_mean.set_ylim(bottom=0)
            ax_mean.tick_params(axis='both', labelsize=tick_fontsize)
            
            plt.tight_layout()
            mean_path = os.path.join(individual_save_dir, f'{base_name}_mean.png')
            fig_mean.savefig(mean_path, dpi=dpi, bbox_inches=bbox_inches, 
                           transparent=transparent_bg)
            print(f"✓ Saved mean plot: {mean_path}")
            plt.close(fig_mean)
        
        return fig
    
    def plot_multi_moiety_coordination_stacked(self, 
                                              coord_keys_dict: dict,
                                               cluster_ids: Union[str, List[int], None] = None,
                                              # Colors and styling
                                              colors: Optional[List[str]] = None,
                                              colormap: str = 'Set2',
                                              cn_hatches: Optional[dict] = None,
                                              # Bar styling
                                              bar_width: float = 0.7,
                                              bar_alpha: float = 0.85,
                                              edgecolor: str = 'black',
                                              edgewidth: float = 1.5,
                                              # Figure layout
                                              figsize: Tuple[float, float] = (20, 6),
                                              layout: str = 'side-by-side',  # 'side-by-side' or 'stacked'
                                              # Font controls
                                              title_fontsize: int = 16,
                                              title_fontweight: str = 'bold',
                                              label_fontsize: int = 13,
                                              label_fontweight: str = 'bold',
                                              tick_fontsize: int = 11,
                                              legend_fontsize: int = 10,
                                              legend_fontweight: str = 'bold',
                                              # Legend
                                              show_legend: bool = True,
                                              legend_loc: str = 'best',
                                              legend_framealpha: float = 0.9,
                                              legend_layout: str = 'horizontal',
                                              legend_ncol_cluster: int = 1,
                                              legend_ncol_cn: int = 1,
                                              legend1_bbox: Optional[Tuple[float, float]] = None,  # (x, y) for cluster legend
                                              legend2_bbox: Optional[Tuple[float, float]] = None,  # (x, y) for CN legend
                                              # Grid
                                              show_grid: bool = True,
                                              grid_alpha: float = 0.3,
                                              grid_style: str = '--',
                                              grid_width: float = 0.7,
                                              # Axis labels
                                              xlabel: Optional[str] = None,
                                              ylabel_dist: Optional[str] = None,
                                              ylabel_mean: Optional[str] = None,
                                              show_xlabel: bool = True,
                                              show_ylabel: bool = True,
                                              # Titles
                                              show_title_dist: bool = True,
                                              show_title_mean: bool = True,
                                              title_dist: Optional[str] = None,
                                              title_mean: Optional[str] = None,
                                              suptitle: Optional[str] = None,
                                              # Enhanced broken axis parameters
                                              enable_broken_axis: bool = True,
                                              broken_axis_threshold: float = 15.0,
                                              broken_axis_min_gap: float = 20.0,
                                              break_ratio: float = 0.3,
                                              max_y_axis: Optional[float] = None,
                                              manual_break_point: Optional[float] = None,
                                              manual_top_start: Optional[float] = None,
                                              manual_break_ratio: Optional[float] = None,
                                              # Export
                                              save_path: Optional[str] = None,
                                              save_fig: bool = False,
                                              save_individual_figures: bool = False,
                                              individual_save_dir: Optional[str] = None,
                                              individual_figsize: Tuple[float, float] = (8, 6),
                                              dpi: int = 300,
                                              bbox_inches: str = 'tight',
                                              transparent_bg: bool = False) -> plt.Figure:
        """
        Create stacked bar plot comparing coordination across multiple moieties.
        
        Each moiety shows 5 bars (one per cluster), with bars stacked by CN values.
        Distribution panel shows probability (sum=1.0), mean panel shows contributions
        to mean CN (sum=mean CN). Uses split legend with colors for clusters and
        hatches for CN values.
        
        Parameters
        ----------
        coord_keys_dict : dict
            Dictionary mapping moiety names to coord_keys.
            Example: {'quinolone': 'key1', 'piperazine': 'key2', ...}
        cluster_ids : 'all', list of int, or None, optional
            Cluster IDs to include. If None or 'all', uses all clusters found in data.
            Use list to specify specific clusters (e.g., [0, 1, 2, 3, 4]).
        colors : list of str, optional
            Colors for each cluster. If None, uses colormap.
        colormap : str, default='Set2'
            Matplotlib colormap if colors not provided
        cn_hatches : dict, optional
            Hatch patterns for each CN value. Example: {0: '///', 1: '\\\\\\', 2: 'xxx'}
            If None, uses default patterns.
        bar_width : float, default=0.7
            Width of each bar
        bar_alpha : float, default=0.85
            Transparency for bars
        edgecolor : str, default='black'
            Edge color for bar segments
        edgewidth : float, default=1.5
            Edge width for bar segments
        figsize : tuple, default=(16, 6)
            Figure size (width, height)
        layout : str, default='side-by-side'
            Layout: 'side-by-side' (horizontal) or 'stacked' (vertical)
        
        Font styling parameters similar to plot_coordination_comparison_styled.
        
        xlabel : str, optional
            X-axis label for both panels
        ylabel_dist : str, optional
            Y-axis label for distribution panel
        ylabel_mean : str, optional
            Y-axis label for mean panel
        show_xlabel : bool, default=True
            Whether to display the x-axis label
        show_ylabel : bool, default=True
            Whether to display the y-axis labels (applies to both panels)
        show_title_dist : bool, default=True
            Whether to show title on distribution panel
        show_title_mean : bool, default=True
            Whether to show title on mean panel
        title_dist : str, optional
            Custom title for distribution panel
        title_mean : str, optional
            Custom title for mean panel
        suptitle : str, optional
            Overall figure title (suptitle)
        
        show_legend : bool, default=True
            Whether to show split legend
        legend_layout : str, default='horizontal'
            Layout for split legends: 'horizontal', 'vertical', 'auto'
        legend_ncol_cluster : int, default=1
            Number of columns for Cluster legend
        legend_ncol_cn : int, default=1
            Number of columns for CN legend
        legend1_bbox : tuple, optional
            Custom position for Cluster legend as (x, y) in axes coordinates (0-1).
            For 'horizontal' layout, default is (0.02, 0.98) (far left, top).
            Example: (0.70, 0.98) for upper right
        legend2_bbox : tuple, optional
            Custom position for CN legend as (x, y) in axes coordinates (0-1).
            For 'horizontal' layout, default is (0.16, 0.98) (next to legend1, top).
            Example: (0.85, 0.98) for far right
        
        enable_broken_axis : bool, default=True
            Whether to enable automatic broken y-axis when large gaps detected
        broken_axis_threshold : float, default=15.0
            Percentage gap threshold to trigger broken axis
        broken_axis_min_gap : float, default=20.0
            Minimum absolute gap (percentage points) required to apply broken axis
        break_ratio : float, default=0.3
            Default height ratio for bottom section (used only when auto-calculation fails)
        max_y_axis : float, optional
            Manual maximum y-axis limit **in probability scale (0-1.0)**.
            For example, 1.0 means 100% probability. If provided, overrides automatic scaling.
            For broken axis: applies to top section maximum.
        manual_break_point : float, optional
            Manual break point position **in probability scale (0-1.0)**.
            For example, 0.14 means break at 14% probability.
            If provided, overrides automatic detection.
            Only used when enable_broken_axis=True.
        manual_top_start : float, optional
            Manual top section start position **in probability scale (0-1.0)**.
            For example, 0.85 means top section starts at 85% probability.
            If provided, overrides automatic calculation.
            Only used when enable_broken_axis=True and manual_break_point is set.
        manual_break_ratio : float, optional
            Manual break ratio override (0.0-1.0). If provided, overrides automatic calculation.
            0.3 means bottom section gets 30% of figure height, top gets 70%.
            0.8 means bottom section gets 80% of figure height, top gets 20%.
            Only used when enable_broken_axis=True.
        
        save_path, save_fig, dpi : standard save parameters
        save_individual_figures : bool, default=False
            Whether to save distribution and mean panels as separate figures
        individual_save_dir : str, optional
            Directory for individual figures. If None, uses save_path directory
        individual_figsize : tuple, default=(8, 6)
            Figure size for individual figures
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
            
        Examples
        --------
        >>> coord_keys = {
        ...     'quinolone': 'resname api and (...)__resname NA__3.25',
        ...     'piperazine': 'resname api and (...)__resname NA__2.45',
        ...     'carboxylic_acid': 'resname api and (...)__resname NA__2.25',
        ...     'cyclopropyl': 'resname api and (...)__resname NA__3.95'
        ... }
        >>> 
        >>> fig = plotter.plot_multi_moiety_coordination_stacked(
        ...     coord_keys_dict=coord_keys,
        ...     cluster_ids=[0, 1, 2, 3, 4],
        ...     colors=['#F08080', '#ADD8E6', '#90EE90', '#FFFFE0', '#E6E6FA'],
        ...     manual_break_ratio=0.8,  # Larger bottom section (80% of figure height)
        ...     manual_break_point=0.14,  # Break at 14% probability (0.14 on 0-1 scale)
        ...     manual_top_start=0.85,    # Top section starts at 85% probability
        ...     save_path='coordination_multi_moiety_stacked.png',
        ...     save_fig=True,
        ...     dpi=300
        ... )
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Validate data
        if not hasattr(self.analyzer, 'coordination_data'):
            raise ValueError("No coordination data found. Run compute_coordination_numbers() first.")
        
        # Validate all coord_keys exist
        for moiety_name, coord_key in coord_keys_dict.items():
            if coord_key not in self.analyzer.coordination_data:
                available_keys = list(self.analyzer.coordination_data.keys())
                raise ValueError(f"Coordination key '{coord_key}' for moiety '{moiety_name}' not found. "
                               f"Available keys:\n" + "\n".join(f"  - {k}" for k in available_keys))
        
        # ============ HELPER FUNCTIONS FOR BROKEN AXIS ============
        
        def analyze_bar_heights(bar_heights, debug=True):
            '''
            Analyze bar heights to determine if broken axis should be applied.
            Handles manual override parameters.
            '''
            # PRIORITY 1: Manual override takes precedence - FORCE broken axis
            if manual_break_point is not None:
                break_point = manual_break_point
                
                if manual_top_start is not None:
                    top_start = manual_top_start
                else:
                    # Auto-calculate top_start: use highest bar value or default
                    if len(bar_heights) > 0:
                        max_height = max(bar_heights)
                        top_start = max(break_point + 5, max_height * 0.90)
                    else:
                        top_start = break_point + 10  # Default gap
                    
                return True, break_point, top_start
            
            # PRIORITY 2: No manual override - check if automatic detection should apply
            if not enable_broken_axis or len(bar_heights) < 2:
                return False, None, None
            
            # Remove zeros and sort heights to find top values
            non_zero_heights = [h for h in bar_heights if h > 0.1]  # Ignore very small values
            
            if len(non_zero_heights) < 2:
                return False, None, None
            
            sorted_heights = sorted(non_zero_heights, reverse=True)
            
            # Find the largest gap between consecutive values
            gaps = []
            for i in range(len(sorted_heights) - 1):
                gap = sorted_heights[i] - sorted_heights[i + 1]
                gaps.append((gap, i, sorted_heights[i], sorted_heights[i + 1]))
            
            # Sort gaps by size (largest first)
            gaps.sort(reverse=True, key=lambda x: x[0])
            
            if gaps:
                largest_gap, gap_index, higher_val, lower_val = gaps[0]
                
                # Check if this gap meets our criteria
                relative_gap = (largest_gap / higher_val) * 100 if higher_val > 0 else 0
                should_break = (relative_gap >= broken_axis_threshold and largest_gap >= broken_axis_min_gap)
                
                if should_break:
                    # Automatic calculation
                    break_point = lower_val * 1.05  # 5% above the lower value
                    top_start = higher_val * 0.90   # Start at 90% of highest value
                    return True, break_point, top_start
                else:
                    return False, None, None
            else:
                return False, None, None
        
        def create_broken_axis_subplots(fig, ax_to_break, break_point, top_start):
            '''Create broken axis subplot configuration with automatic or manual height ratios'''
            # Calculate height ratios based on actual data ranges or use manual override
            if manual_break_ratio is not None:
                # Manual override
                bottom_ratio = manual_break_ratio
                top_ratio = 1 - manual_break_ratio
            else:
                # Automatic calculation based on data ranges
                bottom_range = break_point - 0  # 0 to break_point
                top_range = (max_y_axis if max_y_axis is not None else 105) - top_start  # top_start to max
                
                total_range = bottom_range + top_range
                
                # Calculate proportional ratios
                bottom_ratio = bottom_range / total_range if total_range > 0 else break_ratio
                top_ratio = top_range / total_range if total_range > 0 else (1 - break_ratio)
                
                # Add small padding to ensure visibility and constrain ratios
                bottom_ratio = max(0.15, min(0.85, bottom_ratio))  # Constrain between 15% and 85%
                top_ratio = 1 - bottom_ratio
            
            return bottom_ratio, top_ratio, break_point, top_start
        
        # ============ END HELPER FUNCTIONS ============
        
        # Get data for first moiety to determine cluster_ids if not provided
        first_key = list(coord_keys_dict.values())[0]
        coord_results_sample = self.analyzer.coordination_data[first_key]
        
        # Handle cluster_ids parameter
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = sorted(coord_results_sample.keys())
        elif isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]
        else:
            cluster_ids = sorted(cluster_ids)
        
        n_clusters = len(cluster_ids)
        n_moieties = len(coord_keys_dict)
        moiety_names = list(coord_keys_dict.keys())
        
        # Determine all possible CN values across all moieties and clusters
        all_cn_values = set()
        for coord_key in coord_keys_dict.values():
            coord_results = self.analyzer.coordination_data[coord_key]
            for cluster_id in cluster_ids:
                if cluster_id in coord_results:
                    cn_data = coord_results[cluster_id]['coordination']
                    # Convert to Python int to avoid numpy dtype issues in dictionary lookups
                    all_cn_values.update(int(x) for x in np.unique(cn_data))
        cn_values_sorted = sorted(all_cn_values)  # Normal order for hatch assignment
        n_cn_values = len(cn_values_sorted)
        
        # Set up colors for clusters
        if colors is None:
            cmap = plt.cm.get_cmap(colormap)
            colors_list = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]
        else:
            colors_list = list(colors)[:n_clusters]
        
        # Set up hatches for CN values
        if cn_hatches is None:
            default_hatches = ['', '///', '\\\\\\', 'xxx', '...', '|||', '+++', 'ooo', '**', 'oo']
            # Explicitly ensure keys are Python int for compatibility
            cn_hatches = {int(cn_val): default_hatches[i % len(default_hatches)] 
                         for i, cn_val in enumerate(cn_values_sorted)}
        else:
            # If user provided cn_hatches, ensure it has all necessary CN values
            missing_cn = [cn for cn in cn_values_sorted if cn not in cn_hatches and int(cn) not in cn_hatches]
            if missing_cn:
                default_hatches = ['', '///', '\\\\\\', 'xxx', '...', '|||', '+++', 'ooo', '**', 'oo']
                for cn in missing_cn:
                    cn_hatches[int(cn)] = default_hatches[len(cn_hatches) % len(default_hatches)]
        
        # Create figure
        if layout == 'side-by-side':
            # Horizontal: 1 row, 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                          gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})
        else:  # 'stacked' or default
            # Vertical: 2 rows, 1 column
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # ============ PANEL 1: Distribution (Stacked Probability) ============
        
        # Calculate bar positions
        bar_positions = []  # Will store x-position for each cluster within each moiety
        x_labels = []
        x_ticks = []
        
        current_x = 0
        for moiety_idx, moiety_name in enumerate(moiety_names):
            # Calculate positions for this moiety's clusters (no spacing between bars in same moiety)
            cluster_positions = [current_x + i * bar_width for i in range(n_clusters)]
            bar_positions.append(cluster_positions)
            
            # X-tick at center of this moiety's cluster group
            center_x = np.mean(cluster_positions)
            x_ticks.append(center_x)
            x_labels.append(moiety_name.replace('_', ' ').title())
            
            # Move to next moiety group (with extra spacing)
            current_x = cluster_positions[-1] + bar_width + 0.8
        
        # First, collect all bar heights and data for broken axis analysis
        all_dist_heights = []
        dist_data_collection = {}  # Store data for plotting
        
        for moiety_idx, (moiety_name, coord_key) in enumerate(coord_keys_dict.items()):
            coord_results = self.analyzer.coordination_data[coord_key]
            dist_data_collection[moiety_idx] = {}
            
            for cluster_idx, cluster_id in enumerate(cluster_ids):
                if cluster_id not in coord_results:
                    continue
                
                cn_data = coord_results[cluster_id]['coordination']
                n_frames = len(cn_data)
                
                # Calculate probability for each CN value
                cn_probs = {}
                for cn_val in cn_values_sorted:
                    # Count frames with this CN (cn_val is already Python int from above)
                    count = np.sum(cn_data == cn_val)
                    cn_probs[cn_val] = count / n_frames
                
                # Calculate total height for this bar (should be ~1.0 for probabilities)
                total_height = sum(cn_probs.values())
                all_dist_heights.append(total_height)
                
                # Store for plotting
                dist_data_collection[moiety_idx][cluster_idx] = cn_probs
        
        # Analyze bar heights for broken axis
        # Convert manual parameters from probability scale (0-1) to percentage (0-100) for internal use
        manual_break_point_pct = manual_break_point * 100 if manual_break_point is not None else None
        manual_top_start_pct = manual_top_start * 100 if manual_top_start is not None else None
        max_y_axis_pct = max_y_axis * 100 if max_y_axis is not None else None
        
        # Temporarily override manual parameters for analysis
        original_manual_break = manual_break_point
        original_manual_top = manual_top_start
        original_max_y = max_y_axis
        
        manual_break_point = manual_break_point_pct
        manual_top_start = manual_top_start_pct
        max_y_axis = max_y_axis_pct
        
        # Convert bar heights to percentage scale for analysis
        all_dist_heights_pct = [h * 100 for h in all_dist_heights]
        apply_broken_axis, break_point, top_start = analyze_bar_heights(all_dist_heights_pct, debug=True)
        
        # Restore original values
        manual_break_point = original_manual_break
        manual_top_start = original_manual_top
        max_y_axis = original_max_y
        
        # Initialize broken axis variables (will be set properly if broken axis is applied)
        break_point_prob = None
        top_start_prob = None
        bottom_ratio = 0.5
        top_ratio = 0.5
        
        # Convert break points back to probability scale (0-1.0)
        if apply_broken_axis:
            break_point_prob = break_point / 100
            top_start_prob = top_start / 100
        
        # Create broken axis subplots if needed
        if apply_broken_axis:
            # Get height ratios (these are already calculated correctly for percentage scale)
            bottom_ratio, top_ratio, break_point, top_start = create_broken_axis_subplots(
                fig, ax1, break_point, top_start
            )
            
            # Recreate the figure with broken axis for ax1
            if layout == 'side-by-side':
                # Need to recreate with gridspec for broken axis on left panel
                fig.clear()
                import matplotlib.gridspec as gridspec
                
                # Create main grid: 1 row, 2 columns
                gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.3)
                
                # Left panel (ax1) - broken axis
                gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0],
                                                          height_ratios=[top_ratio, bottom_ratio],
                                                          hspace=0.08)
                ax1_top = fig.add_subplot(gs_left[0])
                ax1_bottom = fig.add_subplot(gs_left[1])
                
                # Right panel (ax2) - normal
                ax2 = fig.add_subplot(gs_main[1])
                
                # Plot distribution bars on broken axis
                for moiety_idx in dist_data_collection:
                    for cluster_idx in dist_data_collection[moiety_idx]:
                        cn_probs = dist_data_collection[moiety_idx][cluster_idx]
                        x_pos = bar_positions[moiety_idx][cluster_idx]
                        
                        # Plot on both top and bottom axes
                        for ax_target in [ax1_bottom, ax1_top]:
                            bottom = 0
                            for cn_val in reversed(cn_values_sorted):
                                prob = cn_probs[cn_val]
                                if prob > 0:
                                    # Ensure cn_val is Python int for dictionary lookup
                                    cn_key = int(cn_val) if not isinstance(cn_val, int) else cn_val
                                    ax_target.bar(x_pos, prob, width=bar_width,
                                                 bottom=bottom,
                                                 color=colors_list[cluster_idx],
                                                 alpha=bar_alpha,
                                                 edgecolor=edgecolor,
                                                 linewidth=edgewidth,
                                                 hatch=cn_hatches[cn_key])
                                    bottom += prob
                
                # Set y-limits for broken axis (in probability scale 0-1)
                if max_y_axis is not None:
                    ax1_top.set_ylim(top_start_prob, max_y_axis / 100)
                else:
                    ax1_top.set_ylim(top_start_prob, 1.05)
                ax1_bottom.set_ylim(0, break_point_prob)
                
                # Hide spines between subplots
                ax1_top.spines['bottom'].set_visible(False)
                ax1_bottom.spines['top'].set_visible(False)
                ax1_top.xaxis.tick_top()
                ax1_top.tick_params(labeltop=False, top=False)
                
                # Add break lines
                d = 0.015
                kwargs = dict(transform=ax1_top.transAxes, color='k', clip_on=False, linewidth=2)
                ax1_top.plot((-d, +d), (-d, +d), **kwargs)
                ax1_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
                
                kwargs.update(transform=ax1_bottom.transAxes)
                ax1_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                ax1_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
                
                # Format bottom axis (has x-labels)
                ax1_bottom.set_xticks(x_ticks)
                ax1_bottom.set_xticklabels(x_labels, fontsize=tick_fontsize, rotation=0)
                if show_xlabel:
                    ax1_bottom.set_xlabel(xlabel or 'Moiety', fontsize=label_fontsize, fontweight=label_fontweight)
                if show_ylabel:
                    ax1_bottom.set_ylabel(ylabel_dist or 'Probability', fontsize=label_fontsize, fontweight=label_fontweight)
                ax1_bottom.tick_params(axis='both', labelsize=tick_fontsize)
                
                # Format top axis (no x-labels, no y-label)
                ax1_top.set_xticks(x_ticks)
                ax1_top.set_xticklabels([])
                if show_title_dist:
                    ax1_top.set_title(title_dist or 'CN Distribution', fontsize=title_fontsize, fontweight=title_fontweight)
                ax1_top.tick_params(axis='both', labelsize=tick_fontsize)
                
                # Manually set y-ticks for top panel to avoid crowding (use only major values)
                top_ylim = ax1_top.get_ylim()
                # Generate clean tick values: start, midpoint (if range is large), end
                top_range = top_ylim[1] - top_ylim[0]
                if top_range > 0.15:  # If range is large enough
                    # Use start and end only
                    top_yticks = [top_ylim[0], top_ylim[1]]
                else:
                    # For very small ranges, just use endpoints
                    top_yticks = [top_ylim[0], top_ylim[1]]
                
                # Round to clean values
                top_yticks_clean = []
                for val in top_yticks:
                    # Round to 1 decimal place for clean display
                    rounded = round(val, 1)
                    top_yticks_clean.append(rounded)
                
                ax1_top.set_yticks(top_yticks_clean)
                
                # Format y-axis tick labels to remove trailing zeros
                from matplotlib.ticker import FuncFormatter
                def clean_tick_formatter(x, pos):
                    """Remove trailing zeros from tick labels"""
                    return f'{x:g}'
                
                ax1_bottom.yaxis.set_major_formatter(FuncFormatter(clean_tick_formatter))
                ax1_top.yaxis.set_major_formatter(FuncFormatter(clean_tick_formatter))
                
                if show_grid:
                    ax1_bottom.grid(True, alpha=grid_alpha, linestyle=grid_style,
                                   linewidth=grid_width, axis='y', zorder=0)
                    ax1_top.grid(True, alpha=grid_alpha, linestyle=grid_style,
                                linewidth=grid_width, axis='y', zorder=0)
                
            else:  # stacked layout
                # Similar approach for stacked layout
                print("WARNING: Broken axis for stacked layout not yet implemented. Using normal axis.")
                apply_broken_axis = False
        
        # Normal plotting (no broken axis)
        if not apply_broken_axis:
            # Plot distribution bars (stacked by CN)
            for moiety_idx in dist_data_collection:
                for cluster_idx in dist_data_collection[moiety_idx]:
                    cn_probs = dist_data_collection[moiety_idx][cluster_idx]
                    x_pos = bar_positions[moiety_idx][cluster_idx]
                    
                    bottom = 0
                    for cn_val in reversed(cn_values_sorted):
                        prob = cn_probs[cn_val]
                        if prob > 0:
                            # Ensure cn_val is Python int for dictionary lookup
                            cn_key = int(cn_val) if not isinstance(cn_val, int) else cn_val
                            ax1.bar(x_pos, prob, width=bar_width,
                                   bottom=bottom,
                                   color=colors_list[cluster_idx],
                                   alpha=bar_alpha,
                                   edgecolor=edgecolor,
                                   linewidth=edgewidth,
                                   hatch=cn_hatches[cn_key])
                            bottom += prob
            
            # Distribution plot formatting
            ax1.set_xticks(x_ticks)
            ax1.set_xticklabels(x_labels, fontsize=tick_fontsize, rotation=0)
            if show_xlabel:
                ax1.set_xlabel(xlabel or 'Moiety', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_ylabel:
                ax1.set_ylabel(ylabel_dist or 'Probability', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title_dist:
                ax1.set_title(title_dist or 'CN Distribution', fontsize=title_fontsize, fontweight=title_fontweight)
            
            if max_y_axis is not None:
                ax1.set_ylim(0, max_y_axis / 100)
            else:
                ax1.set_ylim(0, 1.05)
            
            ax1.tick_params(axis='both', labelsize=tick_fontsize)
            
            if show_grid:
                ax1.grid(True, alpha=grid_alpha, linestyle=grid_style,
                        linewidth=grid_width, axis='y', zorder=0)
        
        # ============ PANEL 2: Mean CN (Stacked Contributions) - ALWAYS NORMAL (NO BROKEN AXIS) ============
        
        # Plot mean CN bars (stacked by CN contributions)
        for moiety_idx, (moiety_name, coord_key) in enumerate(coord_keys_dict.items()):
            coord_results = self.analyzer.coordination_data[coord_key]
            
            for cluster_idx, cluster_id in enumerate(cluster_ids):
                if cluster_id not in coord_results:
                    continue
                
                cn_data = coord_results[cluster_id]['coordination']
                n_frames = len(cn_data)
                
                # Calculate contribution of each CN value to mean
                # Contribution of CN=k is: k × P(CN=k)
                cn_contributions = {}
                for cn_val in cn_values_sorted:
                    count = np.sum(cn_data == cn_val)
                    prob = count / n_frames
                    cn_contributions[cn_val] = cn_val * prob
                
                # Stack bars (reverse order: max CN at bottom, zero at top)
                bottom = 0
                x_pos = bar_positions[moiety_idx][cluster_idx]
                
                for cn_val in reversed(cn_values_sorted):
                    contribution = cn_contributions[cn_val]
                    if contribution > 0:  # Only plot if non-zero
                        # Ensure cn_val is Python int for dictionary lookup
                        cn_key = int(cn_val) if not isinstance(cn_val, int) else cn_val
                        ax2.bar(x_pos, contribution, width=bar_width,
                               bottom=bottom,
                               color=colors_list[cluster_idx],
                               alpha=bar_alpha,
                               edgecolor=edgecolor,
                               linewidth=edgewidth,
                               hatch=cn_hatches[cn_key])
                        bottom += contribution
        
        # Mean CN plot formatting
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels, fontsize=tick_fontsize, rotation=0)
        if show_xlabel:
            ax2.set_xlabel(xlabel or 'Moiety', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_ylabel:
            ax2.set_ylabel(ylabel_mean or 'Mean CN', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title_mean:
            ax2.set_title(title_mean or 'Average Coordination Number', 
                         fontsize=title_fontsize, fontweight=title_fontweight)
        ax2.set_ylim(bottom=0)
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        
        if show_grid:
            ax2.grid(True, alpha=grid_alpha, linestyle=grid_style,
                    linewidth=grid_width, axis='y', zorder=0)
        
        # ============ Split Legend ============
        
        if show_legend:
            # Create cluster color legend handles
            cluster_handles = [Rectangle((0, 0), 1, 1, facecolor=colors_list[i],
                                        edgecolor=edgecolor, linewidth=edgewidth,
                                        alpha=bar_alpha, label=f'C{cid}')
                             for i, cid in enumerate(cluster_ids)]
            
            # Create CN hatch legend handles (ascending order 0, 1, 2, ...)
            cn_handles = [Rectangle((0, 0), 1, 1, facecolor='white',
                                   edgecolor=edgecolor, linewidth=edgewidth,
                                   hatch=cn_hatches[int(cn_val) if not isinstance(cn_val, int) else cn_val], 
                                   label=f'{cn_val}')
                         for cn_val in sorted(cn_values_sorted)]
            
            # Create two adjacent legends (Cluster and CN)
            if legend_layout == 'horizontal':
                # Side by side - use bbox_to_anchor for precise positioning
                bbox1 = legend1_bbox if legend1_bbox is not None else (0.02, 0.98)
                bbox2 = legend2_bbox if legend2_bbox is not None else (0.16, 0.98)
                
                legend1 = ax2.legend(handles=cluster_handles, title='Cluster',
                                   bbox_to_anchor=bbox1, loc='upper left',
                                   fontsize=legend_fontsize,
                                   framealpha=legend_framealpha, title_fontsize=legend_fontsize,
                                   ncol=legend_ncol_cluster)
                for text in legend1.get_texts():
                    text.set_fontweight(legend_fontweight)
                legend1.get_title().set_fontweight(legend_fontweight)
                ax2.add_artist(legend1)
                
                legend2 = ax2.legend(handles=cn_handles, title='CN',
                                   bbox_to_anchor=bbox2, loc='upper left',
                                   fontsize=legend_fontsize,
                                   framealpha=legend_framealpha, title_fontsize=legend_fontsize,
                                   ncol=legend_ncol_cn)
                for text in legend2.get_texts():
                    text.set_fontweight(legend_fontweight)
                legend2.get_title().set_fontweight(legend_fontweight)
            else:
                # Use legend_loc for primary position, place CN legend adjacent
                legend1 = ax2.legend(handles=cluster_handles, title='Cluster',
                                   loc=legend_loc, fontsize=legend_fontsize,
                                   framealpha=legend_framealpha, title_fontsize=legend_fontsize,
                                   ncol=legend_ncol_cluster)
                for text in legend1.get_texts():
                    text.set_fontweight(legend_fontweight)
                legend1.get_title().set_fontweight(legend_fontweight)
                ax2.add_artist(legend1)
                
                # Get bbox of first legend to position second adjacent
                bbox1 = legend1.get_window_extent(ax2.figure.canvas.get_renderer())
                # Position CN legend right below or beside the Cluster legend
                legend2 = ax2.legend(handles=cn_handles, title='CN',
                                   loc=legend_loc, fontsize=legend_fontsize,
                                   framealpha=legend_framealpha, title_fontsize=legend_fontsize,
                                   ncol=legend_ncol_cn)
                for text in legend2.get_texts():
                    text.set_fontweight(legend_fontweight)
                legend2.get_title().set_fontweight(legend_fontweight)
        
        # Suptitle
        if suptitle:
            fig.suptitle(suptitle, fontsize=title_fontsize + 2, fontweight=title_fontweight)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            plt.tight_layout()
        
        # Save figure
        if save_fig and save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches,
                       transparent=transparent_bg)
            print(f"✓ Saved multi-moiety coordination plot: {save_path}")
        
        # Save individual figures if requested
        if save_individual_figures:
            import os
            
            # Determine save directory
            if individual_save_dir is None:
                if save_path:
                    individual_save_dir = os.path.dirname(save_path) or '.'
                else:
                    individual_save_dir = '.'
            
            # Create directory if needed
            os.makedirs(individual_save_dir, exist_ok=True)
            
            # ===== DISTRIBUTION PANEL =====
            fig_dist = plt.figure(figsize=individual_figsize)
            
            # Check if broken axis should be applied
            if apply_broken_axis and layout == 'side-by-side':
                # Apply broken axis to individual distribution figure
                import matplotlib.gridspec as gridspec
                
                gs_dist = gridspec.GridSpec(2, 1, figure=fig_dist,
                                           height_ratios=[top_ratio, bottom_ratio],
                                           hspace=0.12)
                ax_dist_top = fig_dist.add_subplot(gs_dist[0])
                ax_dist_bottom = fig_dist.add_subplot(gs_dist[1])
                
                # Plot distribution bars on BOTH axes (stacked by CN)
                for moiety_idx, (moiety_name, coord_key) in enumerate(coord_keys_dict.items()):
                    coord_results = self.analyzer.coordination_data[coord_key]
                    
                    for cluster_idx, cluster_id in enumerate(cluster_ids):
                        if cluster_id not in coord_results:
                            continue
                        
                        cn_data = coord_results[cluster_id]['coordination']
                        n_frames = len(cn_data)
                        
                        # Calculate probability for each CN value
                        cn_probs = {}
                        for cn_val in cn_values_sorted:
                            count = np.sum(cn_data == cn_val)
                            cn_probs[cn_val] = count / n_frames
                        
                        x_pos = bar_positions[moiety_idx][cluster_idx]
                        
                        # Plot on BOTH top and bottom axes
                        for ax_target in [ax_dist_bottom, ax_dist_top]:
                            bottom = 0
                            for cn_val in reversed(cn_values_sorted):
                                prob = cn_probs[cn_val]
                                if prob > 0:
                                    # Ensure cn_val is Python int for dictionary lookup
                                    cn_key = int(cn_val) if not isinstance(cn_val, int) else cn_val
                                    ax_target.bar(x_pos, prob, width=bar_width,
                                               bottom=bottom,
                                               color=colors_list[cluster_idx],
                                               alpha=bar_alpha,
                                               edgecolor=edgecolor,
                                               linewidth=edgewidth,
                                               hatch=cn_hatches[cn_key])
                                    bottom += prob
                
                # Set y-limits for broken axis
                if max_y_axis is not None:
                    ax_dist_top.set_ylim(top_start_prob, max_y_axis / 100)
                else:
                    ax_dist_top.set_ylim(top_start_prob, 1.05)
                ax_dist_bottom.set_ylim(0, break_point_prob)
                
                # Hide spines between subplots
                ax_dist_top.spines['bottom'].set_visible(False)
                ax_dist_bottom.spines['top'].set_visible(False)
                ax_dist_top.xaxis.tick_top()
                ax_dist_top.tick_params(labeltop=False, top=False)
                
                # Add break lines
                d = 0.025
                kwargs = dict(transform=ax_dist_top.transAxes, color='k', clip_on=False, linewidth=2)
                ax_dist_top.plot((-d, +d), (-d, +d), **kwargs)
                ax_dist_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
                
                kwargs.update(transform=ax_dist_bottom.transAxes)
                ax_dist_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                ax_dist_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
                
                # Format bottom axis (has x-labels)
                ax_dist_bottom.set_xticks(x_ticks)
                ax_dist_bottom.set_xticklabels(x_labels, fontsize=tick_fontsize, rotation=0)
                if show_xlabel:
                    ax_dist_bottom.set_xlabel(xlabel or 'Moiety', fontsize=label_fontsize, fontweight=label_fontweight)
                if show_ylabel:
                    ax_dist_bottom.set_ylabel(ylabel_dist or 'Probability', fontsize=label_fontsize, fontweight=label_fontweight)
                ax_dist_bottom.tick_params(axis='both', labelsize=tick_fontsize)
                
                # Format top axis (no x-labels, no y-label)
                ax_dist_top.set_xticks(x_ticks)
                ax_dist_top.set_xticklabels([])
                if show_title_dist:
                    ax_dist_top.set_title(title_dist or 'CN Distribution', fontsize=title_fontsize, fontweight=title_fontweight)
                ax_dist_top.tick_params(axis='both', labelsize=tick_fontsize)
                
                # Manually set y-ticks for top panel to avoid crowding (use only major values)
                top_ylim = ax_dist_top.get_ylim()
                # Generate clean tick values: start, midpoint (if range is large), end
                top_range = top_ylim[1] - top_ylim[0]
                if top_range > 0.15:  # If range is large enough
                    # Use start and end only
                    top_yticks = [top_ylim[0], top_ylim[1]]
                else:
                    # For very small ranges, just use endpoints
                    top_yticks = [top_ylim[0], top_ylim[1]]
                
                # Round to clean values
                top_yticks_clean = []
                for val in top_yticks:
                    # Round to 1 decimal place for clean display
                    rounded = round(val, 1)
                    top_yticks_clean.append(rounded)
                
                ax_dist_top.set_yticks(top_yticks_clean)
                
                # Format y-axis tick labels to remove trailing zeros
                from matplotlib.ticker import FuncFormatter
                def clean_tick_formatter(x, pos):
                    """Remove trailing zeros from tick labels"""
                    return f'{x:g}'
                
                ax_dist_bottom.yaxis.set_major_formatter(FuncFormatter(clean_tick_formatter))
                ax_dist_top.yaxis.set_major_formatter(FuncFormatter(clean_tick_formatter))
                
                if show_grid:
                    ax_dist_bottom.grid(True, alpha=grid_alpha, linestyle=grid_style,
                                       linewidth=grid_width, axis='y', zorder=0)
                    ax_dist_top.grid(True, alpha=grid_alpha, linestyle=grid_style,
                                    linewidth=grid_width, axis='y', zorder=0)
                
            else:
                # Normal axis (no break)
                ax_dist = fig_dist.add_subplot(111)
                
                # Plot distribution bars (stacked by CN)
                for moiety_idx, (moiety_name, coord_key) in enumerate(coord_keys_dict.items()):
                    coord_results = self.analyzer.coordination_data[coord_key]
                    
                    for cluster_idx, cluster_id in enumerate(cluster_ids):
                        if cluster_id not in coord_results:
                            continue
                        
                        cn_data = coord_results[cluster_id]['coordination']
                        n_frames = len(cn_data)
                        
                        # Calculate probability for each CN value
                        cn_probs = {}
                        for cn_val in cn_values_sorted:
                            count = np.sum(cn_data == cn_val)
                            cn_probs[cn_val] = count / n_frames
                        
                        # Stack bars (reverse order: max CN at bottom, zero at top)
                        bottom = 0
                        x_pos = bar_positions[moiety_idx][cluster_idx]
                        
                        for cn_val in reversed(cn_values_sorted):
                            prob = cn_probs[cn_val]
                            if prob > 0:
                                # Ensure cn_val is Python int for dictionary lookup
                                cn_key = int(cn_val) if not isinstance(cn_val, int) else cn_val
                                ax_dist.bar(x_pos, prob, width=bar_width,
                                           bottom=bottom,
                                           color=colors_list[cluster_idx],
                                           alpha=bar_alpha,
                                           edgecolor=edgecolor,
                                           linewidth=edgewidth,
                                           hatch=cn_hatches[cn_key])
                                bottom += prob
                
                # Distribution plot formatting
                ax_dist.set_xticks(x_ticks)
                ax_dist.set_xticklabels(x_labels, fontsize=tick_fontsize, rotation=0)
                if show_xlabel:
                    ax_dist.set_xlabel(xlabel or 'Moiety', fontsize=label_fontsize, fontweight=label_fontweight)
                if show_ylabel:
                    ax_dist.set_ylabel(ylabel_dist or 'Probability', fontsize=label_fontsize, fontweight=label_fontweight)
                if show_title_dist:
                    ax_dist.set_title(title_dist or 'CN Distribution', fontsize=title_fontsize, fontweight=title_fontweight)
                
                if max_y_axis is not None:
                    ax_dist.set_ylim(0, max_y_axis / 100)
                else:
                    ax_dist.set_ylim(0, 1.05)
                    
                ax_dist.tick_params(axis='both', labelsize=tick_fontsize)
                
                if show_grid:
                    ax_dist.grid(True, alpha=grid_alpha, linestyle=grid_style,
                                linewidth=grid_width, axis='y', zorder=0)
            
            # No legends on distribution plot (not enough space)
            
            plt.tight_layout()
            dist_path = os.path.join(individual_save_dir, 'multi_moiety_distribution.png')
            fig_dist.savefig(dist_path, dpi=dpi, bbox_inches=bbox_inches,
                           transparent=transparent_bg)
            print(f"✓ Saved distribution panel: {dist_path}")
            plt.close(fig_dist)
            
            # ===== MEAN PANEL =====
            fig_mean = plt.figure(figsize=individual_figsize)
            ax_mean = fig_mean.add_subplot(111)
            
            # Plot mean CN bars (stacked by CN contributions)
            for moiety_idx, (moiety_name, coord_key) in enumerate(coord_keys_dict.items()):
                coord_results = self.analyzer.coordination_data[coord_key]
                
                for cluster_idx, cluster_id in enumerate(cluster_ids):
                    if cluster_id not in coord_results:
                        continue
                    
                    cn_data = coord_results[cluster_id]['coordination']
                    n_frames = len(cn_data)
                    
                    # Calculate contribution of each CN value to mean
                    cn_contributions = {}
                    for cn_val in cn_values_sorted:
                        count = np.sum(cn_data == cn_val)
                        prob = count / n_frames
                        cn_contributions[cn_val] = cn_val * prob
                    
                    # Stack bars (reverse order: max CN at bottom, zero at top)
                    bottom = 0
                    x_pos = bar_positions[moiety_idx][cluster_idx]
                    
                    for cn_val in reversed(cn_values_sorted):
                        contribution = cn_contributions[cn_val]
                        if contribution > 0:
                            # Ensure cn_val is Python int for dictionary lookup
                            cn_key = int(cn_val) if not isinstance(cn_val, int) else cn_val
                            ax_mean.bar(x_pos, contribution, width=bar_width,
                                       bottom=bottom,
                                       color=colors_list[cluster_idx],
                                       alpha=bar_alpha,
                                       edgecolor=edgecolor,
                                       linewidth=edgewidth,
                                       hatch=cn_hatches[cn_key])
                            bottom += contribution
            
            # Mean CN plot formatting
            ax_mean.set_xticks(x_ticks)
            ax_mean.set_xticklabels(x_labels, fontsize=tick_fontsize, rotation=0)
            if show_xlabel:
                ax_mean.set_xlabel(xlabel or 'Moiety', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_ylabel:
                ax_mean.set_ylabel(ylabel_mean or 'Mean CN', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title_mean:
                ax_mean.set_title(title_mean or 'Average Coordination Number', fontsize=title_fontsize, fontweight=title_fontweight)
            ax_mean.set_ylim(bottom=0)
            ax_mean.tick_params(axis='both', labelsize=tick_fontsize)
            
            if show_grid:
                ax_mean.grid(True, alpha=grid_alpha, linestyle=grid_style,
                            linewidth=grid_width, axis='y', zorder=0)
            
            # Add two adjacent legends (Cluster and CN)
            if show_legend:
                if legend_layout == 'horizontal':
                    # Side by side - use bbox_to_anchor for precise positioning
                    bbox1 = legend1_bbox if legend1_bbox is not None else (0.02, 0.98)
                    bbox2 = legend2_bbox if legend2_bbox is not None else (0.16, 0.98)
                    
                    legend1 = ax_mean.legend(handles=cluster_handles, title='Cluster',
                                           bbox_to_anchor=bbox1, loc='upper left',
                                           fontsize=legend_fontsize,
                                           framealpha=legend_framealpha, title_fontsize=legend_fontsize,
                                           ncol=legend_ncol_cluster)
                    for text in legend1.get_texts():
                        text.set_fontweight(legend_fontweight)
                    legend1.get_title().set_fontweight(legend_fontweight)
                    ax_mean.add_artist(legend1)
                    
                    legend2 = ax_mean.legend(handles=cn_handles, title='CN',
                                           bbox_to_anchor=bbox2, loc='upper left',
                                           fontsize=legend_fontsize,
                                           framealpha=legend_framealpha, title_fontsize=legend_fontsize,
                                           ncol=legend_ncol_cn)
                    for text in legend2.get_texts():
                        text.set_fontweight(legend_fontweight)
                    legend2.get_title().set_fontweight(legend_fontweight)
                else:
                    # Use legend_loc for primary position, place CN legend adjacent
                    legend1 = ax_mean.legend(handles=cluster_handles, title='Cluster',
                                           loc=legend_loc, fontsize=legend_fontsize,
                                           framealpha=legend_framealpha, title_fontsize=legend_fontsize,
                                           ncol=legend_ncol_cluster)
                    for text in legend1.get_texts():
                        text.set_fontweight(legend_fontweight)
                    legend1.get_title().set_fontweight(legend_fontweight)
                    ax_mean.add_artist(legend1)
                    
                    # Position CN legend adjacent to Cluster legend
                    legend2 = ax_mean.legend(handles=cn_handles, title='CN',
                                           loc=legend_loc, fontsize=legend_fontsize,
                                           framealpha=legend_framealpha, title_fontsize=legend_fontsize,
                                           ncol=legend_ncol_cn)
                    for text in legend2.get_texts():
                        text.set_fontweight(legend_fontweight)
                    legend2.get_title().set_fontweight(legend_fontweight)
            
            plt.tight_layout()
            mean_path = os.path.join(individual_save_dir, 'multi_moiety_mean.png')
            fig_mean.savefig(mean_path, dpi=dpi, bbox_inches=bbox_inches,
                           transparent=transparent_bg)
            print(f"✓ Saved mean panel: {mean_path}")
            plt.close(fig_mean)
        
        return fig
    
    def plot_interaction_summary(self, cluster_ids: Optional[List[int]] = None,
                                figsize: Tuple[float, float] = (14, 10),
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive interaction summary comparing all analyses.
        
        Parameters
        ----------
        cluster_ids : list of int, optional
            Clusters to include
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        
        Example
        -------
        >>> # After running all interaction analyses
        >>> plotter.plot_interaction_summary()
        """
        has_rdf = hasattr(self.analyzer, 'rdf_data') and len(self.analyzer.rdf_data) > 0
        has_dist = hasattr(self.analyzer, 'distance_data') and len(self.analyzer.distance_data) > 0
        has_coord = hasattr(self.analyzer, 'coordination_data') and len(self.analyzer.coordination_data) > 0
        
        if not (has_rdf or has_dist or has_coord):
            raise ValueError("No interaction data found. Run analysis methods first.")
        
        if cluster_ids is None:
            if has_rdf:
                cluster_ids = list(list(self.analyzer.rdf_data.values())[0].keys())
            elif has_dist:
                cluster_ids = list(list(self.analyzer.distance_data.values())[0].keys())
            elif has_coord:
                cluster_ids = list(list(self.analyzer.coordination_data.values())[0].keys())
        
        # Determine grid layout
        n_plots = sum([has_rdf, has_dist, has_coord])
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_plots, 1, hspace=0.4)
        
        plot_idx = 0
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
        
        # RDF plots
        if has_rdf:
            ax = fig.add_subplot(gs[plot_idx])
            for rdf_key in self.analyzer.rdf_data:
                rdf_results = self.analyzer.rdf_data[rdf_key]
                for i, cluster_id in enumerate(cluster_ids):
                    if cluster_id in rdf_results:
                        data = rdf_results[cluster_id]
                        ax.plot(data['r'], data['rdf'], 
                               label=f"C{cluster_id}: {rdf_key.split('__')[1][:20]}...",
                               color=colors[i], lw=1.5, alpha=0.7)
            ax.axhline(y=1.0, color='gray', ls='--', lw=1, alpha=0.5)
            ax.set_xlabel('Distance (Å)', fontsize=11)
            ax.set_ylabel('g(r)', fontsize=11, fontweight='bold')
            ax.set_title('Radial Distribution Functions', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(alpha=0.3)
            plot_idx += 1
        
        # Distance plots
        if has_dist:
            ax = fig.add_subplot(gs[plot_idx])
            for dist_key in self.analyzer.distance_data:
                dist_results = self.analyzer.distance_data[dist_key]
                box_data = []
                labels = []
                for cluster_id in cluster_ids:
                    if cluster_id in dist_results:
                        box_data.append(dist_results[cluster_id]['distances'])
                        labels.append(f"C{cluster_id}")
                
                bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
            
            ax.set_ylabel('Distance (Å)', fontsize=11, fontweight='bold')
            ax.set_title('Distance Distributions', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')
            plot_idx += 1
        
        # Coordination plots
        if has_coord:
            ax = fig.add_subplot(gs[plot_idx])
            width = 0.8 / len(self.analyzer.coordination_data)
            offset = 0
            
            for coord_key in self.analyzer.coordination_data:
                coord_results = self.analyzer.coordination_data[coord_key]
                means = [coord_results[cid]['mean_cn'] for cid in cluster_ids if cid in coord_results]
                x_pos = np.arange(len(means)) + offset
                
                ax.bar(x_pos, means, width=width, 
                      label=coord_key.split('__')[1][:20],
                      alpha=0.7)
                offset += width
            
            ax.set_xticks(np.arange(len(cluster_ids)) + width * (len(self.analyzer.coordination_data) - 1) / 2)
            ax.set_xticklabels([f"Cluster {cid}" for cid in cluster_ids])
            ax.set_ylabel('Coordination Number', fontsize=11, fontweight='bold')
            ax.set_title('Mean Coordination Numbers', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis='y')
        
        fig.suptitle('Interaction Analysis Summary', fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_pi_cation_analysis(self, pi_cation_key: str,
                                cluster_ids: Optional[List[int]] = None,
                                figsize: Tuple[float, float] = (14, 10),
                                colors: Optional[List] = None,
                                save_path: Optional[str] = None):
        """
        Plot comprehensive π-cation interaction analysis.
        
        Creates 4-panel figure showing:
        1. RDF (distance to ring center)
        2. Distance distributions
        3. Angle distributions (stacking geometry)
        4. Contact frequency comparison
        
        Parameters
        ----------
        pi_cation_key : str
            Key in analyzer.pi_cation_data dictionary
        cluster_ids : list of int, optional
            Clusters to plot
        figsize : tuple
            Figure size
        colors : list, optional
            Colors for each cluster
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib Figure
        
        Example
        -------
        >>> plotter.plot_pi_cation_analysis(
        ...     'resname api and (...ring atoms...)__resname NA',
        ...     cluster_ids=[0, 1],
        ...     save_path='pi_cation.png'
        ... )
        """
        if not hasattr(self.analyzer, 'pi_cation_data') or pi_cation_key not in self.analyzer.pi_cation_data:
            raise ValueError(f"π-cation data not found for key: {pi_cation_key}")
        
        pi_results = self.analyzer.pi_cation_data[pi_cation_key]
        
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = list(pi_results.keys())
        
        if colors is None:
            colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_ids)))
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
        
        # Panel 1: RDF
        ax1 = fig.add_subplot(gs[0, 0])
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id in pi_results:
                data = pi_results[cluster_id]
                dist = data['distances_to_center']
                
                # Compute RDF-like histogram
                hist, bins = np.histogram(dist, bins=50, range=(0, data['distance_cutoff']))
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Normalize
                rdf = hist / hist.max() if hist.max() > 0 else hist
                
                ax1.plot(bin_centers, rdf, label=f"Cluster {cluster_id}",
                        color=colors[i], lw=2)
                
                # Mark preferred distance
                if not np.isnan(data['preferred_distance']):
                    ax1.axvline(data['preferred_distance'], color=colors[i],
                              ls='--', lw=1, alpha=0.5)
        
        ax1.set_xlabel('Distance to Ring Center (Å)', fontsize=11)
        ax1.set_ylabel('Normalized Density', fontsize=11, fontweight='bold')
        ax1.set_title('π-Cation Distance Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Panel 2: Distance box plots
        ax2 = fig.add_subplot(gs[0, 1])
        box_data = []
        labels = []
        for cluster_id in cluster_ids:
            if cluster_id in pi_results:
                box_data.append(pi_results[cluster_id]['distances_to_center'])
                labels.append(f"C{cluster_id}")
        
        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Annotate means
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id in pi_results:
                mean_val = pi_results[cluster_id]['mean_distance']
                ax2.text(i+1, mean_val, f'{mean_val:.2f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_ylabel('Distance (Å)', fontsize=11, fontweight='bold')
        ax2.set_title('Distance Statistics', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        # Panel 3: Angle distributions
        ax3 = fig.add_subplot(gs[1, 0])
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id in pi_results:
                angles = pi_results[cluster_id]['angles']
                ax3.hist(angles, bins=30, alpha=0.6, color=colors[i],
                        label=f"C{cluster_id} (μ={angles.mean():.1f}°)",
                        edgecolor='black', lw=0.5)
        
        ax3.axvline(0, color='red', ls='--', lw=1, label='Ideal (0°)')
        ax3.set_xlabel('Angle Deviation from Perpendicular (°)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('Stacking Geometry', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        
        # Panel 4: Contact frequency
        ax4 = fig.add_subplot(gs[1, 1])
        contact_freqs = [pi_results[cid]['contact_frequency'] for cid in cluster_ids if cid in pi_results]
        x_pos = np.arange(len(contact_freqs))
        
        bars = ax4.bar(x_pos, contact_freqs, color=colors[:len(contact_freqs)], alpha=0.7, edgecolor='black')
        
        # Annotate bars
        for i, (bar, freq) in enumerate(zip(bars, contact_freqs)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height,
                    f'{freq:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"Cluster {cid}" for cid in cluster_ids if cid in pi_results])
        ax4.set_ylabel('Contact Frequency (%)', fontsize=11, fontweight='bold')
        ax4.set_title('π-Cation Contact Occurrence', fontsize=12, fontweight='bold')
        ax4.set_ylim(0, 100)
        ax4.grid(alpha=0.3, axis='y')
        
        # Selections info
        sel_info = pi_results[cluster_ids[0]]
        ring_sel = sel_info['ring_selection']
        cat_sel = sel_info['cation_selection']
        cutoff = sel_info['distance_cutoff']
        angle_cut = sel_info['angle_cutoff']
        
        fig.suptitle(f'π-Cation Interaction Analysis\n' +
                    f'Ring: {ring_sel[:40]}... | Cations: {cat_sel} | ' +
                    f'Cutoff: {cutoff:.1f}Å, ±{angle_cut:.0f}°',
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def _sanitize_key_for_filename(self, key: str, max_length: int = 50) -> str:
        """
        Sanitize orientation key string to create a clean filename.
        
        Parameters
        ----------
        key : str
            The orientation key string to sanitize
        max_length : int
            Maximum length for the sanitized string (default: 50)
        
        Returns
        -------
        str
            Sanitized string suitable for use in filenames
        """
        import re
        
        # Remove problematic characters: parentheses, commas, quotes
        clean = key.replace('(', '').replace(')', '').replace(',', '')
        clean = clean.replace("'", '').replace('"', '')
        
        # Replace spaces and special characters with underscores
        clean = re.sub(r'[\s:;]+', '_', clean)
        
        # Remove any remaining non-alphanumeric characters except underscores and hyphens
        clean = re.sub(r'[^a-zA-Z0-9_-]+', '_', clean)
        
        # Replace consecutive underscores with single underscore
        clean = re.sub(r'_+', '_', clean)
        
        # Remove leading/trailing underscores
        clean = clean.strip('_')
        
        # Truncate to max_length if needed
        if len(clean) > max_length:
            clean = clean[:max_length].rstrip('_')
        
        return clean
    
    def _is_selection_string(self, key: str) -> bool:
        """
        Check if a key appears to be an MDAnalysis selection string rather than a clean variable name.
        
        Parameters
        ----------
        key : str
            The key to check
        
        Returns
        -------
        bool
            True if key looks like a selection string, False if it's a clean variable name
        """
        # Keywords that indicate a selection string
        selection_keywords = ['resname', 'name ', 'resid', 'segid', 'and ', 'or ', 'not ', 
                             'protein', 'nucleic', 'backbone', 'type ', 'around']
        
        key_lower = key.lower()
        return any(keyword in key_lower for keyword in selection_keywords)
    
    def plot_orientation_analysis(self, orientation_key: str,
                                  cluster_ids: Optional[List[int]] = None,
                                  figsize: Tuple[float, float] = (12, 5),
                                  colors: Optional[List] = None,
                                  custom_colors_dist: Optional[List[str]] = None,
                                  custom_colors_mean: Optional[List[str]] = None,
                                  
                                  # Axis labels
                                  xlabel_hist: str = 'Angle (°)',
                                  ylabel_hist: str = 'Probability Density',
                                  ylabel_bar: str = 'Mean Angle (°)',
                                  
                                  # Title control
                                  show_title_dist: bool = True,
                                  show_title_mean: bool = True,
                                  title_fontsize: int = 12,
                                  title_fontweight: str = 'bold',
                                  show_main_title: bool = True,
                                  main_title_fontsize: int = 12,
                                  
                                  # Label and tick formatting
                                  label_fontsize: int = 11,
                                  label_fontweight: str = 'bold',
                                  tick_fontsize: int = 9,
                                  tick_fontweight: str = 'normal',
                                  
                                  # Annotation text on bars
                                  show_bar_annotations: bool = True,
                                  annotation_fontsize: int = 9,
                                  annotation_fontweight: str = 'bold',
                                  annotation_multiline: bool = True,
                                  
                                  # Grid control
                                  show_grid: bool = True,
                                  grid_alpha: float = 0.3,
                                  grid_linestyle: str = '-',
                                  grid_linewidth: float = 0.5,
                                  
                                  # Legend control
                                  show_legend: bool = True,
                                  legend_fontsize: int = 12,
                                  legend_fontweight: str = 'bold',
                                  legend_loc: str = 'best',
                                  legend_frame_alpha: float = 0.8,
                                  
                                  # Histogram control
                                  hist_bins: int = 30,
                                  hist_alpha: float = 0.6,
                                  hist_edgecolor: str = 'black',
                                  hist_linewidth: float = 0.5,
                                  
                                  # Bar chart control
                                  bar_width: float = 0.4,
                                  bar_spacing: float = 0.7,
                                  bar_alpha: float = 0.85,
                                  bar_edgecolor: str = 'black',
                                  bar_edgewidth: float = 1.0,
                                  bar_capsize: float = 5,
                                  bar_error_linewidth: float = 2,
                                  ymin_mean: Optional[Union[float, dict]] = None,
                                  ymax_mean: Optional[Union[float, dict]] = None,
                                  
                                  # Figure quality
                                  dpi: Optional[int] = None,
                                  
                                  # Save control
                                  save_figure: bool = False,
                                  skip_selection_strings: bool = True,
                                  save_dir: Optional[str] = None,
                                  save_path: Optional[str] = None,
                                  
                                  # Individual figures control
                                  save_individual_figures: bool = True,
                                  individual_save_dir: Optional[str] = None,
                                  individual_figsize: Tuple[float, float] = (8, 6)):
        """
        Plot molecular group orientation analysis.
        
        Creates 2-panel figure:
        1. Angle distributions (histograms)
        2. Mean angle comparison (bar chart with error bars)
        
        Parameters
        ----------
        orientation_key : str
            Key in analyzer.orientation_data
        cluster_ids : list of int, optional
            Clusters to plot. If None, plots all available.
        figsize : tuple
            Figure size (width, height) in inches (default: (12, 5))
        colors : list, optional
            Colors for clusters. If None, uses Set2 colormap. Overridden by custom_colors_dist/custom_colors_mean if provided.
        custom_colors_dist : list of str, optional
            Custom color list for distribution histograms (e.g., ['#8B0000', '#4682B4', '#228B22']).
            Use darker colors for overlapping histograms with alpha transparency.
            If None, falls back to colors or default colormap.
        custom_colors_mean : list of str, optional
            Custom color list for mean angle bar chart (e.g., ['#F08080', '#ADD8E6', '#90EE90']).
            Can use lighter colors since bars don't overlap.
            If None, falls back to colors or default colormap.
        
        # Axis labels
        xlabel_hist : str
            X-axis label for histogram panel (default: 'Angle (°)')
        ylabel_hist : str
            Y-axis label for histogram panel (default: 'Probability Density')
        ylabel_bar : str
            Y-axis label for bar chart panel (default: 'Mean Angle (°)')
        
        # Title control
        show_title_dist : bool
            Show distribution panel title (default: True)
        show_title_mean : bool
            Show mean angle panel title (default: True)
        title_fontsize : int
            Font size for panel titles (default: 12)
        title_fontweight : str
            Font weight for panel titles (default: 'bold')
        show_main_title : bool
            Show main figure title (default: True)
        main_title_fontsize : int
            Font size for main figure title (default: 12)
        
        # Label and tick formatting
        label_fontsize : int
            Font size for axis labels (default: 11)
        label_fontweight : str
            Font weight for axis labels (default: 'bold')
        tick_fontsize : int
            Font size for tick labels (default: 9)
        tick_fontweight : str
            Font weight for tick labels (default: 'normal')
        
        # Annotation text on bars
        show_bar_annotations : bool
            Show mean±std annotations on bars (default: True)
        annotation_fontsize : int
            Font size for bar annotations (default: 9)
        annotation_fontweight : str
            Font weight for bar annotations (default: 'bold')
        annotation_multiline : bool
            If True, annotation displays as 'mean°\n±std' (two lines).
            If False, displays as 'mean° ± std' (single line). Default: True
        
        # Grid control
        show_grid : bool
            Show grid lines (default: True)
        grid_alpha : float
            Transparency of grid lines (default: 0.3)
        grid_linestyle : str
            Style of grid lines (default: '-')
        grid_linewidth : float
            Width of grid lines (default: 0.5)
        
        # Legend control
        show_legend : bool
            Show legend on histogram panel (default: True)
        legend_fontsize : int
            Font size for legend (default: 18)
        legend_fontweight : str
            Font weight for legend text (default: 'bold')
        legend_loc : str
            Legend location (default: 'best')
        legend_frame_alpha : float
            Transparency of legend frame (default: 0.8)
        
        # Histogram control
        hist_bins : int
            Number of bins for histogram (default: 30)
        hist_alpha : float
            Transparency of histogram bars (default: 0.6)
        hist_edgecolor : str
            Edge color for histogram bars (default: 'black')
        hist_linewidth : float
            Edge line width for histogram bars (default: 0.5)
        
        # Bar chart control
        bar_width : float
            Width of each bar (default: 0.4). Fixed width maintains visual consistency.
        bar_spacing : float
            Spacing multiplier between bars (default: 0.7). Larger values increase spacing
            without changing bar width. E.g., 1.0 = standard spacing, 0.5 = tight, 1.5 = wide.
        bar_alpha : float
            Transparency of bars (default: 0.85)
        bar_edgecolor : str
            Edge color for bars (default: 'black')
        bar_edgewidth : float
            Edge line width for bars (default: 1.0)
        bar_capsize : float
            Cap size for error bars (default: 5)
        bar_error_linewidth : float
            Line width for error bars (default: 2)
        ymin_mean : float, dict, or None, optional
            Minimum y-axis value for mean angle panel. Can be:
            - float: same limit for all orientation keys
            - dict: per-key limits (e.g., {'quinolone': 60, 'carboxylic_acid': 30})
            - None: auto-scale (default)
        ymax_mean : float, dict, or None, optional
            Maximum y-axis value for mean angle panel. Can be:
            - float: same limit for all orientation keys
            - dict: per-key limits (e.g., {'quinolone': 95, 'carboxylic_acid': 90})
            - None: auto-scale (default)
        
        # Figure quality
        dpi : int, optional
            DPI for saved figure (default: None, uses self.default_dpi)
        
        # Save control
        save_figure : bool
            Auto-save figure with filename based on orientation_key (default: False)
        skip_selection_strings : bool
            When save_figure=True, skip saving if orientation_key looks like a selection string
            rather than a clean variable name (default: True). Set to False to save all keys.
        save_dir : str, optional
            Directory to save figure (default: None, uses current directory)
        save_path : str, optional
            Custom path to save figure. If provided, overrides save_figure and uses this exact path.
            If None and save_figure=True, auto-generates filename as '{orientation_key}_orientation.png'
        
        # Individual figures control
        save_individual_figures : bool
            Save separate figures for distribution and mean panels (default: True)
            Creates two individual files: one for the distribution histogram (all clusters)
            and one for the mean angle bar chart (all clusters).
        individual_save_dir : str, optional
            Directory to save individual panel figures (default: None, uses current directory)
        individual_figsize : tuple
            Figure size for individual panel plots (default: (8, 6))
            
        Returns
        -------
        fig : matplotlib Figure
            The created figure
        """
        if not hasattr(self.analyzer, 'orientation_data') or orientation_key not in self.analyzer.orientation_data:
            raise ValueError(f"Orientation data not found: {orientation_key}")
        
        orient_results = self.analyzer.orientation_data[orientation_key]
        
        if cluster_ids is None:
            cluster_ids = list(orient_results.keys())
        
        # Set colors for distribution panel (histograms)
        if custom_colors_dist is not None:
            dist_colors = custom_colors_dist[:len(cluster_ids)]
        elif colors is not None:
            dist_colors = colors
        else:
            dist_colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_ids)))
        
        # Set colors for mean panel (bar chart)
        if custom_colors_mean is not None:
            mean_colors = custom_colors_mean[:len(cluster_ids)]
        elif colors is not None:
            mean_colors = colors
        else:
            mean_colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_ids)))
        
        # Set default DPI
        dpi = dpi or self.default_dpi
        
        # Extract y-axis limits for this orientation_key if dict provided
        if isinstance(ymin_mean, dict):
            ymin_mean_val = ymin_mean.get(orientation_key, None)
        else:
            ymin_mean_val = ymin_mean
        
        if isinstance(ymax_mean, dict):
            ymax_mean_val = ymax_mean.get(orientation_key, None)
        else:
            ymax_mean_val = ymax_mean
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Panel 1: Distributions (normalized by density)
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id in orient_results:
                data = orient_results[cluster_id]
                ax1.hist(data['angles'], bins=hist_bins, alpha=hist_alpha, color=dist_colors[i],
                        label=f"Cluster {cluster_id}",
                        edgecolor=hist_edgecolor, lw=hist_linewidth,
                        density=True)  # Normalize to probability density
        
        ax1.set_xlabel(xlabel_hist, fontsize=label_fontsize, fontweight=label_fontweight)
        ax1.set_ylabel(ylabel_hist, fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title_dist:
            ax1.set_title('Orientation Angle Distribution', fontsize=title_fontsize, fontweight=title_fontweight)
        
        # Tick formatting for ax1
        ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontweight(tick_fontweight)
        
        if show_legend:
            legend = ax1.legend(fontsize=legend_fontsize, loc=legend_loc, framealpha=legend_frame_alpha)
            for text in legend.get_texts():
                text.set_fontweight(legend_fontweight)
        
        if show_grid:
            ax1.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
        
        # Panel 2: Mean comparison
        means = [orient_results[cid]['mean_angle'] for cid in cluster_ids if cid in orient_results]
        stds = [orient_results[cid]['std_angle'] for cid in cluster_ids if cid in orient_results]
        x_pos = np.arange(len(means)) * bar_spacing
        
        bars = ax2.bar(x_pos, means, yerr=stds, width=bar_width,
                      color=mean_colors[:len(means)],
                      alpha=bar_alpha, edgecolor=bar_edgecolor, 
                      linewidth=bar_edgewidth, capsize=bar_capsize, 
                      error_kw={'lw': bar_error_linewidth})
        
        # Annotate (place text on top of error bar)
        if show_bar_annotations:
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                # Position text at top of error bar (mean + std)
                y_position = mean + std
                # Format annotation based on multiline setting
                if annotation_multiline:
                    annotation_text = f'{mean:.1f}°\n±{std:.1f}'
                else:
                    annotation_text = f'{mean:.1f}° ± {std:.1f}'
                ax2.text(bar.get_x() + bar.get_width()/2, y_position,
                        annotation_text,
                        ha='center', va='bottom', 
                        fontsize=annotation_fontsize, fontweight=annotation_fontweight)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"C{cid}" for cid in cluster_ids if cid in orient_results])
        ax2.set_ylabel(ylabel_bar, fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title_mean:
            ax2.set_title('Mean Orientation Angle', fontsize=title_fontsize, fontweight=title_fontweight)
        
        # Adjust x-axis limits to maintain constant visual bar width
        if len(means) > 1:
            # Calculate total span of bars
            total_span = (len(means) - 1) * bar_spacing
            # Add proportional margins to keep bars centered
            margin_factor = 0.1  # Base margin as fraction of total span
            base_margin = 0.5  # Minimum margin in data units
            margin = max(total_span * margin_factor, base_margin) + bar_width
            # Set limits to center bars with dynamic margins
            ax2.set_xlim(-margin, total_span + margin)
        else:
            # Single bar - use symmetric margins
            ax2.set_xlim(-1, 1)
        
        # Tick formatting for ax2
        ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            label.set_fontweight(tick_fontweight)
        
        if show_grid:
            ax2.grid(alpha=grid_alpha, axis='y', ls=grid_linestyle, lw=grid_linewidth)
        
        # Set y-axis limits for mean panel if specified
        if ymin_mean_val is not None or ymax_mean_val is not None:
            current_ylim = ax2.get_ylim()
            new_ymin = ymin_mean_val if ymin_mean_val is not None else current_ylim[0]
            new_ymax = ymax_mean_val if ymax_mean_val is not None else current_ylim[1]
            ax2.set_ylim(new_ymin, new_ymax)
        
        # Main title
        if show_main_title:
            # Use clean orientation_key variable name instead of selection string
            title_key = orientation_key.replace('_', ' ').title()
            fig.suptitle(f'Orientation Analysis: {title_key}',
                        fontsize=main_title_fontsize, fontweight='bold')
        
        plt.tight_layout()
        
        # Save individual figures (separate distribution and mean panels)
        if save_individual_figures and not (skip_selection_strings and self._is_selection_string(orientation_key)):
            import os
            
            # Determine save directory
            if individual_save_dir is None:
                if save_path:
                    individual_save_dir = os.path.dirname(save_path) or '.'
                else:
                    individual_save_dir = '.'
            
            # Create directory if needed
            os.makedirs(individual_save_dir, exist_ok=True)
            
            # Generate base filename
            clean_key = self._sanitize_key_for_filename(orientation_key)
            
            # ===== DISTRIBUTION PANEL =====
            fig_dist = plt.figure(figsize=individual_figsize)
            ax_dist = fig_dist.add_subplot(111)
            
            # Plot distribution for all clusters
            for i, cluster_id in enumerate(cluster_ids):
                if cluster_id in orient_results:
                    data = orient_results[cluster_id]
                    ax_dist.hist(data['angles'], bins=hist_bins, alpha=hist_alpha, 
                                color=dist_colors[i],
                                label=f"Cluster {cluster_id}",
                                edgecolor=hist_edgecolor, lw=hist_linewidth,
                                density=True)
            
            ax_dist.set_xlabel(xlabel_hist, fontsize=label_fontsize, fontweight=label_fontweight)
            ax_dist.set_ylabel(ylabel_hist, fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title_dist:
                ax_dist.set_title('Orientation Angle Distribution', 
                                 fontsize=title_fontsize, fontweight=title_fontweight)
            
            ax_dist.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            for label in (ax_dist.get_xticklabels() + ax_dist.get_yticklabels()):
                label.set_fontweight(tick_fontweight)
            
            if show_legend:
                legend_dist = ax_dist.legend(fontsize=legend_fontsize, loc=legend_loc, 
                                            framealpha=legend_frame_alpha)
                for text in legend_dist.get_texts():
                    text.set_fontweight(legend_fontweight)
            
            if show_grid:
                ax_dist.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
            
            plt.tight_layout()
            dist_path = os.path.join(individual_save_dir, f'{clean_key}_orientation_distribution.png')
            fig_dist.savefig(dist_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved distribution panel: {dist_path}")
            plt.close(fig_dist)
            
            # ===== MEAN ANGLE PANEL =====
            fig_mean = plt.figure(figsize=individual_figsize)
            ax_mean = fig_mean.add_subplot(111)
            
            # Plot mean angles for all clusters
            means = [orient_results[cid]['mean_angle'] for cid in cluster_ids if cid in orient_results]
            stds = [orient_results[cid]['std_angle'] for cid in cluster_ids if cid in orient_results]
            x_pos = np.arange(len(means)) * bar_spacing
            
            bars_mean = ax_mean.bar(x_pos, means, yerr=stds, width=bar_width,
                                   color=mean_colors[:len(means)],
                                   alpha=bar_alpha, edgecolor=bar_edgecolor,
                                   linewidth=bar_edgewidth, capsize=bar_capsize,
                                   error_kw={'lw': bar_error_linewidth})
            
            # Annotate
            if show_bar_annotations:
                for i, (bar, mean, std) in enumerate(zip(bars_mean, means, stds)):
                    y_position = mean + std
                    # Format annotation based on multiline setting
                    if annotation_multiline:
                        annotation_text = f'{mean:.1f}°\n±{std:.1f}'
                    else:
                        annotation_text = f'{mean:.1f}° ± {std:.1f}'
                    ax_mean.text(bar.get_x() + bar.get_width()/2, y_position,
                                annotation_text,
                                ha='center', va='bottom',
                                fontsize=annotation_fontsize, fontweight=annotation_fontweight)
            
            ax_mean.set_xticks(x_pos)
            ax_mean.set_xticklabels([f"C{cid}" for cid in cluster_ids if cid in orient_results])
            ax_mean.set_ylabel(ylabel_bar, fontsize=label_fontsize, fontweight=label_fontweight)
            if show_title_mean:
                ax_mean.set_title('Mean Orientation Angle', 
                                 fontsize=title_fontsize, fontweight=title_fontweight)
            
            # Adjust x-axis limits to maintain constant visual bar width
            if len(means) > 1:
                # Calculate total span of bars
                total_span = (len(means) - 1) * bar_spacing
                # Add proportional margins to keep bars centered
                margin_factor = 0.1  # Base margin as fraction of total span
                base_margin = 0.5  # Minimum margin in data units
                margin = max(total_span * margin_factor, base_margin) + bar_width
                # Set limits to center bars with dynamic margins
                ax_mean.set_xlim(-margin, total_span + margin)
            else:
                # Single bar - use symmetric margins
                ax_mean.set_xlim(-1, 1)
            
            ax_mean.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            for label in (ax_mean.get_xticklabels() + ax_mean.get_yticklabels()):
                label.set_fontweight(tick_fontweight)
            
            if show_grid:
                ax_mean.grid(alpha=grid_alpha, axis='y', ls=grid_linestyle, lw=grid_linewidth)
            
            # Set y-axis limits for mean panel if specified
            if ymin_mean_val is not None or ymax_mean_val is not None:
                current_ylim = ax_mean.get_ylim()
                new_ymin = ymin_mean_val if ymin_mean_val is not None else current_ylim[0]
                new_ymax = ymax_mean_val if ymax_mean_val is not None else current_ylim[1]
                ax_mean.set_ylim(new_ymin, new_ymax)
            
            plt.tight_layout()
            mean_path = os.path.join(individual_save_dir, f'{clean_key}_orientation_mean.png')
            fig_mean.savefig(mean_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved mean angle panel: {mean_path}")
            plt.close(fig_mean)
        
        # Save combined figure
        if save_path or (save_figure and not (skip_selection_strings and self._is_selection_string(orientation_key))):
            # Set default DPI
            save_dpi = dpi or self.default_dpi
            
            # Determine filename
            if save_path:
                # Use custom save path
                filename = save_path
            else:
                # Auto-generate clean filename from orientation_key
                # Sanitize the key to remove special characters
                clean_key = self._sanitize_key_for_filename(orientation_key)
                base_dir = save_dir if save_dir else "."
                filename = f"{base_dir}/{clean_key}_orientation.png"
            
            fig.savefig(filename, dpi=save_dpi, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        return fig
    
    def plot_orientation_spatial_heatmap(self, orientation_keys=None,
                                        cluster_ids: Union[str, List[int], None] = None,
                                        z_bin: float = 1.0,
                                        angle_bin: float = 3.0,
                                        angle_range: int = 180,
                                        figsize: Optional[Tuple[float, float]] = None,
                                        cmap: str = 'viridis',
                                        normalize: bool = True,
                                        clay_boundaries: Optional[dict] = None,
                                        
                                        # Axis labels
                                        xlabel: Optional[str] = None,
                                        ylabel: str = 'Tilt Angle (°)',
                                        
                                        # Publication formatting
                                        show_title: bool = True,
                                        title_fontsize: int = 11,
                                        title_fontweight: str = 'bold',
                                        label_fontsize: int = 10,
                                        label_fontweight: str = 'bold',
                                        tick_fontsize: int = 9,
                                        tick_fontweight: str = 'normal',
                                        
                                        # Statistics text box
                                        show_stats_box: bool = False,
                                        stats_fontsize: int = 9,
                                        stats_box_alpha: float = 0.7,
                                        
                                        # Contour overlay
                                        show_contour: bool = False,
                                        contour_levels: int = 5,
                                        contour_colors: str = 'white',
                                        contour_linewidths: float = 0.5,
                                        contour_alpha: float = 0.5,
                                        
                                        # Reference lines
                                        show_reference_lines: bool = True,
                                        reference_line_alpha: float = 0.5,
                                        reference_line_width: float = 1.0,
                                        
                                        # Clay boundary lines
                                        show_clay_lines: bool = True,
                                        clay_line_alpha: float = 0.7,
                                        clay_line_width: float = 2.0,
                                        
                                        # Colorbar control
                                        colorbar_label_fontsize: int = 10,
                                        colorbar_label_fontweight: str = 'normal',
                                        colorbar_tick_fontsize: int = 9,
                                        colorbar_pad: float = 0.01,
                                        colorbar_width: float = 0.03,
                                        
                                        # Legend control
                                        show_legend: bool = True,
                                        legend_fontsize: int = 8,
                                        legend_loc: str = 'upper right',
                                        legend_ncol: int = 1,
                                        legend_frame_alpha: float = 0.8,
                                        
                                        # Grid control
                                        show_grid: bool = False,
                                        grid_alpha: float = 0.3,
                                        grid_linestyle: str = '--',
                                        grid_linewidth: float = 0.5,
                                        
                                        # Figure quality
                                        dpi: Optional[int] = None,
                                        
                                        # Individual plots control
                                        save_individual_figures: bool = False,
                                        skip_selection_strings: bool = True,
                                        individual_save_dir: Optional[str] = None,
                                        individual_figsize: Tuple[float, float] = (8, 6),
                                        
                                        # Combined figure control
                                        save_combined_figure: bool = False,
                                        save_path: Optional[str] = None):
        """
        Create 2D spatial heatmaps showing orientation angles vs Z-position.
        Similar to MATLAB run_9_angles.m visualization.
        
        Shows how molecular group orientations vary with distance from clay surface.
        **Z-range is automatically set to full simulation box dimensions.**
        Creates subplots for each group x cluster combination.
        
        Parameters
        ----------
        orientation_keys : str, list, or None
            Orientation data keys to plot. If None, plots all available.
            Can be single key string or list of keys.
        cluster_ids : 'all', list of int, or None, optional
            Clusters to analyze. If None or 'all', uses all available clusters.
        z_bin : float
            Z-position bin size in Angstroms (default: 1.0 Å)
        angle_bin : float
            Angle bin size in degrees (default: 3.0°)
        angle_range : int
            Maximum angle for y-axis: 90 for tilt angles (0-90°), 
            180 for full range (0-180°). Default: 90
        figsize : tuple, optional
            Figure size for combined plot. If None, auto-calculated based on subplots.
        cmap : str
            Colormap for heatmap (default: 'viridis')
        normalize : bool
            Normalize each heatmap to [0,1] (default: True)
        clay_boundaries : dict, optional
            Clay interface boundary data from ZDirectionalAnalysis.
            If provided, adds vertical reference lines at clay surface positions.
            **Note:** Z-range is determined automatically from box dimensions,
            this parameter only adds visual reference lines.
            Expected keys: 'clay_average_z_positive', 'clay_average_z_negative'
        
        # Axis labels
        xlabel : str, optional
            Custom X-axis label. If None (default), automatically set to
            'Z from box center (Å)' for centered coordinates or
            'Z-position (Å)' for absolute coordinates.
        ylabel : str
            Y-axis label (default: 'Tilt Angle (°)')
        
        # Publication formatting
        show_title : bool
            Show subplot titles (default: True). Note: There is no main figure title - 
            this parameter controls individual subplot titles only.
        title_fontsize : int
            Font size for titles (default: 11)
        title_fontweight : str
            Font weight for titles (default: 'bold')
        label_fontsize : int
            Font size for axis labels (default: 10)
        label_fontweight : str
            Font weight for axis labels (default: 'bold')
        tick_fontsize : int
            Font size for tick labels (default: 9)
        tick_fontweight : str
            Font weight for tick labels (default: 'normal')
        
        # Statistics text box
        show_stats_box : bool
            Show statistics text box with mean and std (default: False)
        stats_fontsize : int
            Font size for statistics text (default: 9)
        stats_box_alpha : float
            Transparency of statistics box background (default: 0.7)
        
        # Contour overlay
        show_contour : bool
            Overlay contour lines on heatmap (default: False)
        contour_levels : int
            Number of contour levels (default: 5)
        contour_colors : str
            Color for contour lines (default: 'white')
        contour_linewidths : float
            Width of contour lines (default: 0.5)
        contour_alpha : float
            Transparency of contour lines (default: 0.5)
        
        # Reference lines
        show_reference_lines : bool
            Show horizontal reference lines at 0°, 90°, 180° (default: True)
        reference_line_alpha : float
            Transparency of reference lines (default: 0.5)
        reference_line_width : float
            Width of reference lines (default: 1.0)
        
        # Clay boundary lines
        show_clay_lines : bool
            Show clay boundary vertical lines (default: True, requires clay_boundaries)
        clay_line_alpha : float
            Transparency of clay boundary lines (default: 0.7)
        clay_line_width : float
            Width of clay boundary lines (default: 2.0)
        
        # Colorbar control
        colorbar_label_fontsize : int
            Font size for colorbar label (default: 10)
        colorbar_label_fontweight : str
            Font weight for colorbar label (default: 'normal')
        colorbar_tick_fontsize : int
            Font size for colorbar ticks (default: 9)
        colorbar_pad : float
            Padding between plot and colorbar (default: 0.01)
        colorbar_width : float
            Colorbar width as fraction of axes (default: 0.03 = 3%)
        
        # Legend control
        show_legend : bool
            Show legend in first subplot (default: True)
        legend_fontsize : int
            Font size for legend (default: 8)
        legend_loc : str
            Legend location (default: 'upper right')
        legend_ncol : int
            Number of columns in legend (default: 1)
        legend_frame_alpha : float
            Transparency of legend frame (default: 0.8)
        
        # Grid control
        show_grid : bool
            Show grid lines on plots (default: False)
        grid_alpha : float
            Transparency of grid lines (default: 0.3)
        grid_linestyle : str
            Style of grid lines (default: '--')
        grid_linewidth : float
            Width of grid lines (default: 0.5)
        
        # Figure quality
        dpi : int, optional
            DPI for saved figures (default: None, uses self.default_dpi)
        
        # Individual plots control
        save_individual_figures : bool
            Save each group/cluster combination as individual figure (default: False)
        skip_selection_strings : bool
            When save_individual_figures=True, skip saving if orientation_key looks like a selection string
            rather than a clean variable name (default: True). Set to False to save all keys.
        individual_save_dir : str, optional
            Directory to save individual plots (default: None, uses current dir)
        individual_figsize : tuple
            Figure size for individual plots (default: (8, 6))
        
        # Combined figure control
        save_combined_figure : bool
            Save the combined multi-panel figure (default: False)
        save_path : str, optional
            Path to save combined figure (overrides save_combined_figure if provided)
            
        Returns
        -------
        fig : matplotlib Figure
            The combined figure with all subplots
        
        Example
        -------
        >>> # Basic usage with auto box range (no subplot titles)
        >>> fig = plotter.plot_orientation_spatial_heatmap(
        ...     orientation_keys=['quinolone', 'carboxylic_acid', 'piperazine'],
        ...     cluster_ids=[0, 1],
        ...     angle_range=180,
        ...     show_title=False  # Hide all subplot titles for clean look
        ... )
        
        >>> # Publication-ready with contours and custom formatting
        >>> fig = plotter.plot_orientation_spatial_heatmap(
        ...     orientation_keys=['quinolone', 'carboxylic_acid'],
        ...     cluster_ids=[0, 1],
        ...     angle_range=180,
        ...     cmap='jet',
        ...     show_contour=True,  # Overlay contour lines
        ...     contour_levels=8,
        ...     show_title=True,  # Show subplot titles
        ...     show_legend=False,
        ...     show_grid=True,
        ...     grid_alpha=0.2,
        ...     dpi=600,
        ...     save_individual_figures=True,
        ...     individual_save_dir='orientation_heatmaps',
        ...     save_path='combined_orientation.png'
        ... )
        """
        if not hasattr(self.analyzer, 'orientation_data'):
            raise ValueError("No orientation data found. Run compute_orientation_angles first.")
        
        # Set default DPI
        dpi = dpi or self.default_dpi
        
        # Create directory for individual plots if needed
        if save_individual_figures and individual_save_dir:
            from pathlib import Path
            Path(individual_save_dir).mkdir(parents=True, exist_ok=True)
            save_dir = individual_save_dir
        else:
            save_dir = "."
        
        # Handle orientation_keys
        if orientation_keys is None:
            orientation_keys = list(self.analyzer.orientation_data.keys())
        elif isinstance(orientation_keys, str):
            orientation_keys = [orientation_keys]
        
        # Validate keys
        for key in orientation_keys:
            if key not in self.analyzer.orientation_data:
                raise ValueError(f"Orientation key not found: {key}")
        
        # Get all available cluster IDs if not specified or if 'all'
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = list(self.analyzer.orientation_data[orientation_keys[0]].keys())
        
        # Calculate subplot layout
        n_groups = len(orientation_keys)
        n_clusters = len(cluster_ids)
        
        if figsize is None:
            figsize = (5 * n_clusters, 4 * n_groups)
        
        fig, axes = plt.subplots(n_groups, n_clusters, figsize=figsize,
                                squeeze=False, constrained_layout=True)
        
        # Determine Z-range from simulation box dimensions
        # Get box dimensions from first available cluster
        use_centered = False
        box_z_length = None
        
        for orient_key in orientation_keys:
            orient_data = self.analyzer.orientation_data[orient_key]
            for cluster_id in cluster_ids:
                if cluster_id in orient_data:
                    cluster_data = orient_data[cluster_id]
                    if 'z_positions_centered' in cluster_data:
                        use_centered = True
                    if 'box_z_length' in cluster_data:
                        box_z_length = cluster_data['box_z_length']
                        break
            if box_z_length is not None:
                break
        
        # Fallback: get box dimensions from trajectory if not in orientation data
        if box_z_length is None:
            first_key = orientation_keys[0]
            first_cluster = cluster_ids[0]
            if hasattr(self.analyzer, 'trajectory_data'):
                if first_cluster in self.analyzer.trajectory_data:
                    u = self.analyzer.trajectory_data[first_cluster]['universe']
                    box_z_length = u.dimensions[2]
                    print(f"Note: Using box Z-length from trajectory: {box_z_length:.2f} Å")
        
        if box_z_length is None:
            raise ValueError("Could not determine box dimensions. Recompute orientation angles.")
        
        # Set Z-range to full simulation box
        if use_centered:
            # Centered coordinates: -Lz/2 to +Lz/2
            global_z_min = -box_z_length / 2.0
            global_z_max = box_z_length / 2.0
            default_xlabel = 'Z from box center (Å)'
        else:
            # Absolute coordinates: 0 to Lz
            global_z_min = 0.0
            global_z_max = box_z_length
            default_xlabel = 'Z-position (Å)'
        
        # Use custom labels or defaults
        x_label = xlabel if xlabel is not None else default_xlabel
        y_label = ylabel
        
        # Process each group x cluster combination
        for i, orient_key in enumerate(orientation_keys):
            orient_data = self.analyzer.orientation_data[orient_key]
            
            for j, cluster_id in enumerate(cluster_ids):
                ax = axes[i, j]
                
                if cluster_id not in orient_data:
                    ax.text(0.5, 0.5, f'No data for\nCluster {cluster_id}',
                           ha='center', va='center', fontsize=title_fontsize)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                cluster_data = orient_data[cluster_id]
                angles = cluster_data['angles']
                
                # Use centered Z if available, otherwise use original
                if 'z_positions_centered' in cluster_data:
                    z_positions = cluster_data['z_positions_centered']
                else:
                    z_positions = cluster_data['z_positions']
                
                # Create 2D histogram bins using GLOBAL Z-range
                z_min, z_max = global_z_min, global_z_max
                angle_min, angle_max = 0, angle_range
                
                z_edges = np.arange(z_min, z_max + z_bin, z_bin)
                angle_edges = np.arange(angle_min, angle_max + angle_bin, angle_bin)
                
                # Compute 2D histogram
                hist_2d, z_edges_out, angle_edges_out = np.histogram2d(
                    z_positions, angles,
                    bins=[z_edges, angle_edges]
                )
                
                # Normalize if requested
                if normalize:
                    hist_2d_norm = hist_2d / (hist_2d.max() + 1e-10)
                else:
                    hist_2d_norm = hist_2d
                
                # ===== INDIVIDUAL FIGURE =====
                if save_individual_figures and not (skip_selection_strings and self._is_selection_string(orient_key)):
                    fig_ind = plt.figure(figsize=individual_figsize)
                    ax_ind = fig_ind.add_subplot(111)
                    
                    im_ind = ax_ind.imshow(hist_2d_norm.T, origin='lower', aspect='auto',
                                          cmap=cmap, interpolation='bilinear',
                                          extent=[z_min, z_max, angle_min, angle_max])
                    
                    # Contour overlay
                    if show_contour:
                        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
                        angle_centers = (angle_edges[:-1] + angle_edges[1:]) / 2
                        Z_mesh, A_mesh = np.meshgrid(z_centers, angle_centers)
                        ax_ind.contour(Z_mesh, A_mesh, hist_2d_norm.T, 
                                      levels=contour_levels, colors=contour_colors,
                                      linewidths=contour_linewidths, alpha=contour_alpha)
                    
                    # Reference lines
                    if show_reference_lines:
                        if angle_range >= 90:
                            ax_ind.axhline(90, color='white', linestyle='--', 
                                          linewidth=reference_line_width,
                                          alpha=reference_line_alpha, label='Parallel (90°)')
                        ax_ind.axhline(0, color='cyan', linestyle='--', 
                                      linewidth=reference_line_width,
                                      alpha=reference_line_alpha, label='Perpendicular (0°)')
                        if angle_range >= 180:
                            ax_ind.axhline(180, color='yellow', linestyle='--', 
                                          linewidth=reference_line_width,
                                          alpha=reference_line_alpha, label='180°')
                    
                    # Clay boundary lines
                    if show_clay_lines and clay_boundaries is not None:
                        upper_clay = clay_boundaries.get('clay_average_z_positive')
                        lower_clay = clay_boundaries.get('clay_average_z_negative')
                        
                        if upper_clay is not None:
                            ax_ind.axvline(upper_clay, color='red', linestyle='-', 
                                          linewidth=clay_line_width, alpha=clay_line_alpha, 
                                          label='Clay surface (upper)')
                        if lower_clay is not None:
                            ax_ind.axvline(lower_clay, color='blue', linestyle='-',
                                          linewidth=clay_line_width, alpha=clay_line_alpha, 
                                          label='Clay surface (lower)')
                    
                    # Labels and title
                    ax_ind.set_xlabel(x_label, fontsize=label_fontsize, fontweight=label_fontweight)
                    ax_ind.set_ylabel(y_label, fontsize=label_fontsize, fontweight=label_fontweight)
                    if show_title:
                        ax_ind.set_title(f'{orient_key} - Cluster {cluster_id}',
                                        fontsize=title_fontsize, fontweight=title_fontweight)
                    
                    # Tick formatting
                    ax_ind.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    for label in (ax_ind.get_xticklabels() + ax_ind.get_yticklabels()):
                        label.set_fontweight(tick_fontweight)
                    
                    # Colorbar
                    cbar_ind = plt.colorbar(im_ind, ax=ax_ind, 
                                           pad=colorbar_pad,
                                           aspect=int(1/colorbar_width),
                                           shrink=1.0)
                    cbar_ind.set_label('Normalized Frequency' if normalize else 'Count',
                                      fontsize=colorbar_label_fontsize,
                                      fontweight=colorbar_label_fontweight)
                    cbar_ind.ax.tick_params(labelsize=colorbar_tick_fontsize)
                    
                    # Statistics text box
                    if show_stats_box:
                        mean_angle = cluster_data['mean_angle']
                        std_angle = cluster_data['std_angle']
                        ax_ind.text(0.02, 0.98, f"μ={mean_angle:.1f}° ± {std_angle:.1f}°",
                                   transform=ax_ind.transAxes, fontsize=stats_fontsize,
                                   verticalalignment='top', bbox=dict(boxstyle='round',
                                   facecolor='white', alpha=stats_box_alpha))
                    
                    # Legend
                    if show_legend:
                        legend_ind = ax_ind.legend(fontsize=legend_fontsize, loc=legend_loc,
                                                  ncol=legend_ncol,
                                                  framealpha=legend_frame_alpha)
                    
                    # Grid
                    if show_grid:
                        ax_ind.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
                    
                    plt.tight_layout()
                    
                    # Save individual figure
                    clean_key = self._sanitize_key_for_filename(orient_key)
                    ind_filename = f"{save_dir}/orientation_heatmap_{clean_key}_C{cluster_id}.png"
                    fig_ind.savefig(ind_filename, dpi=dpi, bbox_inches='tight')
                    print(f"Saved individual plot: {ind_filename}")
                    plt.close(fig_ind)
                
                # ===== COMBINED FIGURE SUBPLOT =====
                # Plot heatmap
                im = ax.imshow(hist_2d_norm.T, origin='lower', aspect='auto',
                              cmap=cmap, interpolation='bilinear',
                              extent=[z_min, z_max, angle_min, angle_max])
                
                # Contour overlay
                if show_contour:
                    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
                    angle_centers = (angle_edges[:-1] + angle_edges[1:]) / 2
                    Z_mesh, A_mesh = np.meshgrid(z_centers, angle_centers)
                    ax.contour(Z_mesh, A_mesh, hist_2d_norm.T, 
                              levels=contour_levels, colors=contour_colors,
                              linewidths=contour_linewidths, alpha=contour_alpha)
                
                # Reference lines
                if show_reference_lines:
                    if angle_range >= 90:
                        ax.axhline(90, color='white', linestyle='--', 
                                  linewidth=reference_line_width,
                                  alpha=reference_line_alpha, label='Parallel (90°)')
                    ax.axhline(0, color='cyan', linestyle='--', 
                              linewidth=reference_line_width,
                              alpha=reference_line_alpha, label='Perpendicular (0°)')
                    if angle_range >= 180:
                        ax.axhline(180, color='yellow', linestyle='--', 
                                  linewidth=reference_line_width,
                                  alpha=reference_line_alpha, label='180°')
                
                # Clay boundary lines
                if show_clay_lines and clay_boundaries is not None:
                    upper_clay = clay_boundaries.get('clay_average_z_positive')
                    lower_clay = clay_boundaries.get('clay_average_z_negative')
                    
                    if upper_clay is not None:
                        ax.axvline(upper_clay, color='red', linestyle='-', 
                                  linewidth=clay_line_width, alpha=clay_line_alpha, 
                                  label='Clay surface (upper)')
                    if lower_clay is not None:
                        ax.axvline(lower_clay, color='blue', linestyle='-',
                                  linewidth=clay_line_width, alpha=clay_line_alpha, 
                                  label='Clay surface (lower)')
                
                # Labels and title
                ax.set_xlabel(x_label, fontsize=label_fontsize, fontweight=label_fontweight)
                if j == 0:
                    ax.set_ylabel(y_label, fontsize=label_fontsize, fontweight=label_fontweight)
                
                if show_title:
                    ax.set_title(f'{orient_key}\nCluster {cluster_id}',
                               fontsize=title_fontsize, fontweight=title_fontweight)
                
                # Tick formatting
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontweight(tick_fontweight)
                
                # Colorbar
                cbar = plt.colorbar(im, ax=ax, 
                                   pad=colorbar_pad,
                                   aspect=int(1/colorbar_width),
                                   shrink=1.0)
                cbar.set_label('Normalized Frequency' if normalize else 'Count',
                             fontsize=colorbar_label_fontsize,
                             fontweight=colorbar_label_fontweight)
                cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
                
                # Legend (only for first subplot)
                if show_legend and i == 0 and j == 0:
                    legend = ax.legend(fontsize=legend_fontsize, loc=legend_loc,
                                     ncol=legend_ncol,
                                     framealpha=legend_frame_alpha)
                
                # Statistics text box
                if show_stats_box:
                    mean_angle = cluster_data['mean_angle']
                    std_angle = cluster_data['std_angle']
                    ax.text(0.02, 0.98, f"μ={mean_angle:.1f}° ± {std_angle:.1f}°",
                           transform=ax.transAxes, fontsize=stats_fontsize,
                           verticalalignment='top', bbox=dict(boxstyle='round',
                           facecolor='white', alpha=stats_box_alpha))
                
                # Grid
                if show_grid:
                    ax.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
        
        # Save combined figure (no main title - show_title controls subplot titles only)
        if save_combined_figure:
            save_file = save_path if save_path else f"{save_dir}/orientation_heatmap_combined.png"
            fig.savefig(save_file, dpi=dpi, bbox_inches='tight')
            print(f"Saved combined figure: {save_file}")
        
        return fig
    
    def plot_mean_angle_vs_z(self, orientation_keys=None,
                            cluster_ids: Optional[List[int]] = None,
                            z_bin: float = 1.0,
                            angle_range: int = 90,
                            figsize: Tuple[float, float] = (12, 6),
                            colors: Optional[List] = None,
                            clay_boundaries: Optional[dict] = None,
                            save_path: Optional[str] = None):
        """
        Plot mean orientation angle profiles vs Z-position.
        Shows how average molecular orientation changes with distance from surface.
        **Z-range is automatically set to full simulation box dimensions.**
        
        Parameters
        ----------
        orientation_keys : str, list, or None
            Orientation data keys. If None, plots all.
        cluster_ids : list of int, optional
            Clusters to plot
        z_bin : float
            Z-position bin size in Angstroms
        angle_range : int
            Maximum angle for y-axis: 90 for tilt angles (0-90°),
            180 for full range (0-180°). Default: 90
        figsize : tuple
            Figure size
        colors : list, optional
            Colors for different groups/clusters
        clay_boundaries : dict, optional
            Clay interface boundary data from ZDirectionalAnalysis.
            If provided, adds vertical reference lines at clay surface positions.
            **Note:** Z-range is determined automatically from box dimensions,
            this parameter only adds visual reference lines.
        save_path : str, optional
            Save path
            
        Returns
        -------
        fig : matplotlib Figure
        
        Example
        -------
        >>> fig = plotter.plot_mean_angle_vs_z(
        ...     orientation_keys=['quinolone', 'carboxylic_acid', 'piperazine'],
        ...     cluster_ids=[0, 1],
        ...     z_bin=1.0
        ... )
        """
        if not hasattr(self.analyzer, 'orientation_data'):
            raise ValueError("No orientation data. Run compute_orientation_angles first.")
        
        # Handle orientation_keys
        if orientation_keys is None:
            orientation_keys = list(self.analyzer.orientation_data.keys())
        elif isinstance(orientation_keys, str):
            orientation_keys = [orientation_keys]
        
        # Get cluster IDs
        if cluster_ids is None:
            cluster_ids = list(self.analyzer.orientation_data[orientation_keys[0]].keys())
        
        # Setup colors
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(orientation_keys) * len(cluster_ids)))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine Z-range from simulation box dimensions
        first_key = orientation_keys[0]
        first_cluster = cluster_ids[0]
        cluster_data = self.analyzer.orientation_data[first_key][first_cluster]
        
        use_centered = 'z_positions_centered' in cluster_data
        box_z_length = cluster_data.get('box_z_length', None)
        
        # Fallback: get box dimensions from trajectory if not in orientation data
        if box_z_length is None:
            if hasattr(self.analyzer, 'trajectory_data'):
                if first_cluster in self.analyzer.trajectory_data:
                    u = self.analyzer.trajectory_data[first_cluster]['universe']
                    box_z_length = u.dimensions[2]
                    print(f"Note: Using box Z-length from trajectory: {box_z_length:.2f} Å")
        
        if box_z_length is None:
            raise ValueError("Could not determine box dimensions. Recompute orientation angles.")
        
        # Set Z-range to full simulation box
        if use_centered:
            # Centered coordinates: -Lz/2 to +Lz/2
            global_z_min = -box_z_length / 2.0
            global_z_max = box_z_length / 2.0
            z_label = 'Z from box center (Å)'
        else:
            # Absolute coordinates: 0 to Lz
            global_z_min = 0.0
            global_z_max = box_z_length
            z_label = 'Z-position (Å)'
        
        color_idx = 0
        for orient_key in orientation_keys:
            orient_data = self.analyzer.orientation_data[orient_key]
            
            for cluster_id in cluster_ids:
                if cluster_id not in orient_data:
                    continue
                
                cluster_data = orient_data[cluster_id]
                angles = cluster_data['angles']
                
                # Use centered Z if available
                if use_centered and 'z_positions_centered' in cluster_data:
                    z_positions = cluster_data['z_positions_centered']
                else:
                    z_positions = cluster_data['z_positions']
                
                # Bin by Z-position using global range
                z_edges = np.arange(global_z_min, global_z_max + z_bin, z_bin)
                z_centers = (z_edges[:-1] + z_edges[1:]) / 2
                
                mean_angles = []
                std_angles = []
                for i in range(len(z_edges) - 1):
                    mask = (z_positions >= z_edges[i]) & (z_positions < z_edges[i+1])
                    if mask.sum() > 0:
                        mean_angles.append(angles[mask].mean())
                        std_angles.append(angles[mask].std())
                    else:
                        mean_angles.append(np.nan)
                        std_angles.append(np.nan)
                
                mean_angles = np.array(mean_angles)
                std_angles = np.array(std_angles)
                
                # Plot with error bands
                label = f'{orient_key} - C{cluster_id}'
                ax.plot(z_centers, mean_angles, '-o', color=colors[color_idx],
                       label=label, linewidth=2, markersize=4)
                ax.fill_between(z_centers, mean_angles - std_angles,
                               mean_angles + std_angles,
                               alpha=0.2, color=colors[color_idx])
                
                color_idx += 1
        
        # Reference lines
        if angle_range >= 90:
            ax.axhline(90, color='red', linestyle='--', linewidth=1,
                      alpha=0.5, label='Parallel (90°)')
            ax.axhline(45, color='gray', linestyle=':', linewidth=1,
                      alpha=0.3, label='45°')
        ax.axhline(0, color='blue', linestyle='--', linewidth=1,
                  alpha=0.5, label='Perpendicular (0°)')
        if angle_range >= 180:
            ax.axhline(180, color='orange', linestyle='--', linewidth=1,
                      alpha=0.5, label='180°')
            ax.axhline(135, color='gray', linestyle=':', linewidth=1,
                      alpha=0.3, label='135°')
        
        # Add clay boundary lines if provided
        if clay_boundaries is not None:
            upper_clay = clay_boundaries.get('clay_average_z_positive')
            lower_clay = clay_boundaries.get('clay_average_z_negative')
            
            if upper_clay is not None:
                ax.axvline(upper_clay, color='darkred', linestyle='-', 
                          linewidth=2.5, alpha=0.8, label='Clay surface (upper)')
            if lower_clay is not None:
                ax.axvline(lower_clay, color='darkblue', linestyle='-',
                          linewidth=2.5, alpha=0.8, label='Clay surface (lower)')
        
        ax.set_xlabel(z_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Tilt Angle (°)', fontsize=12, fontweight='bold')
        ax.set_title('Molecular Orientation vs Distance from Clay Surface',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(alpha=0.3)
        ax.set_ylim(-5, angle_range + 5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_contact_analysis(self, contact_key: str,
                             cluster_ids: Optional[List[int]] = None,
                             figsize: Tuple[float, float] = (14, 5),
                             colors: Optional[List] = None,
                             save_path: Optional[str] = None):
        """
        Plot contact analysis (frequency and lifetimes).
        
        Creates 3-panel figure:
        1. Contact frequency (% time)
        2. Lifetime distributions
        3. Lifetime statistics (box plot)
        
        Parameters
        ----------
        contact_key : str
            Key in analyzer.contact_data
        cluster_ids : list of int, optional
            Clusters to plot
        figsize : tuple
            Figure size
        colors : list, optional
            Colors for clusters
        save_path : str, optional
            Save path
            
        Returns
        -------
        fig : matplotlib Figure
        """
        if not hasattr(self.analyzer, 'contact_data') or contact_key not in self.analyzer.contact_data:
            raise ValueError(f"Contact data not found: {contact_key}")
        
        contact_results = self.analyzer.contact_data[contact_key]
        
        if cluster_ids is None:
            cluster_ids = list(contact_results.keys())
        
        if colors is None:
            colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_ids)))
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 3, wspace=0.3)
        
        # Panel 1: Contact frequency
        ax1 = fig.add_subplot(gs[0, 0])
        freqs = [contact_results[cid]['contact_frequency'] for cid in cluster_ids if cid in contact_results]
        x_pos = np.arange(len(freqs))
        
        bars = ax1.bar(x_pos, freqs, color=colors[:len(freqs)], alpha=0.7, edgecolor='black')
        
        for bar, freq in zip(bars, freqs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{freq:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"C{cid}" for cid in cluster_ids if cid in contact_results])
        ax1.set_ylabel('Contact Frequency (%)', fontsize=11, fontweight='bold')
        ax1.set_title('% Time in Contact', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(alpha=0.3, axis='y')
        
        # Panel 2: Lifetime distributions
        ax2 = fig.add_subplot(gs[0, 1])
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id in contact_results:
                lifetimes = contact_results[cluster_id]['contact_lifetimes']
                if len(lifetimes) > 0:
                    ax2.hist(lifetimes, bins=30, alpha=0.6, color=colors[i],
                            label=f"C{cluster_id} (n={len(lifetimes)})",
                            edgecolor='black', lw=0.5)
        
        ax2.set_xlabel('Contact Lifetime (frames)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Contact Duration Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        
        # Panel 3: Lifetime statistics
        ax3 = fig.add_subplot(gs[0, 2])
        box_data = []
        labels = []
        for cluster_id in cluster_ids:
            if cluster_id in contact_results:
                lifetimes = contact_results[cluster_id]['contact_lifetimes']
                if len(lifetimes) > 0:
                    box_data.append(lifetimes)
                    labels.append(f"C{cluster_id}")
        
        if len(box_data) > 0:
            bp = ax3.boxplot(box_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            # Annotate medians
            for i, cluster_id in enumerate([cid for cid in cluster_ids if cid in contact_results]):
                median_val = contact_results[cluster_id]['median_lifetime']
                ax3.text(i+1, median_val, f'{median_val:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.set_ylabel('Lifetime (frames)', fontsize=11, fontweight='bold')
        ax3.set_title('Lifetime Statistics', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
        
        # Info
        data0 = contact_results[cluster_ids[0]]
        sel1 = data0['selection1']
        sel2 = data0['selection2']
        cutoff = data0['cutoff']
        
        fig.suptitle(f'Contact Analysis\n' +
                    f'{sel1[:30]}... ↔ {sel2[:30]}... | Cutoff: {cutoff:.1f}Å',
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_hydrogen_bond_analysis(self, hbond_key: str,
                                    cluster_ids: Union[str, List[int], None] = None,
                                    figsize: Tuple[float, float] = (16, 10),
                                    colors: Optional[List] = None,
                                    save_path: Optional[str] = None):
        """
        Plot comprehensive hydrogen bond analysis.
        
        Creates 5-panel figure showing:
        1. H-bond count timeseries (both clusters)
        2. Mean H-bond count comparison (bar chart)
        3. Lifetime distributions
        4. Top H-bond pairs by occupancy
        5. Lifetime statistics (box plot)
        
        Parameters
        ----------
        hbond_key : str
            Key in analyzer.hbond_data dictionary
        cluster_ids : str or list of int, optional
            Clusters to plot. Use 'all' or None for all clusters, or provide specific cluster IDs
        figsize : tuple
            Figure size
        colors : list, optional
            Colors for clusters
        save_path : str, optional
            Save path
            
        Returns
        -------
        fig : matplotlib Figure
        
        Example
        -------
        >>> plotter.plot_hydrogen_bond_analysis(
        ...     'resname api and (...)__name OW',
        ...     cluster_ids=[0, 1],
        ...     save_path='hbond_analysis.png'
        ... )
        """
        if not hasattr(self.analyzer, 'hbond_data') or hbond_key not in self.analyzer.hbond_data:
            raise ValueError(f"H-bond data not found: {hbond_key}")
        
        hbond_results = self.analyzer.hbond_data[hbond_key]
        
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = list(hbond_results.keys())
        
        if colors is None:
            colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_ids)))
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)
        
        # Panel 1: Timeseries (full width)
        ax1 = fig.add_subplot(gs[0, :])
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id in hbond_results:
                data = hbond_results[cluster_id]
                timeseries = data['timeseries']
                frames = np.arange(len(timeseries))
                ax1.plot(frames, timeseries, label=f"Cluster {cluster_id}",
                        color=colors[i], lw=1.5, alpha=0.7)
        
        ax1.set_xlabel('Frame Index', fontsize=11)
        ax1.set_ylabel('Number of H-bonds', fontsize=11, fontweight='bold')
        ax1.set_title('H-Bond Count Over Time', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Panel 2: Mean H-bond count
        ax2 = fig.add_subplot(gs[1, 0])
        mean_counts = [hbond_results[cid]['mean_count'] for cid in cluster_ids if cid in hbond_results]
        x_pos = np.arange(len(mean_counts))
        
        bars = ax2.bar(x_pos, mean_counts, color=colors[:len(mean_counts)],
                      alpha=0.7, edgecolor='black')
        
        for bar, count in zip(bars, mean_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"C{cid}" for cid in cluster_ids if cid in hbond_results])
        ax2.set_ylabel('Mean H-bonds/frame', fontsize=11, fontweight='bold')
        ax2.set_title('Average H-Bond Count', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        # Panel 3: Number of unique H-bonds
        ax3 = fig.add_subplot(gs[1, 1])
        n_hbonds = [hbond_results[cid]['n_hbonds'] for cid in cluster_ids if cid in hbond_results]
        x_pos = np.arange(len(n_hbonds))
        
        bars = ax3.bar(x_pos, n_hbonds, color=colors[:len(n_hbonds)],
                      alpha=0.7, edgecolor='black')
        
        for bar, count in zip(bars, n_hbonds):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f"C{cid}" for cid in cluster_ids if cid in hbond_results])
        ax3.set_ylabel('Number of Unique H-bonds', fontsize=11, fontweight='bold')
        ax3.set_title('H-Bond Diversity', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
        
        # Panel 4: Lifetime distributions
        ax4 = fig.add_subplot(gs[2, 0])
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id in hbond_results:
                lifetimes_dict = hbond_results[cluster_id]['lifetimes']
                all_lifetimes = []
                for lifetimes in lifetimes_dict.values():
                    all_lifetimes.extend(lifetimes)
                
                if len(all_lifetimes) > 0:
                    ax4.hist(all_lifetimes, bins=30, alpha=0.6, color=colors[i],
                            label=f"C{cluster_id} (μ={np.mean(all_lifetimes):.1f})",
                            edgecolor='black', lw=0.5)
        
        ax4.set_xlabel('H-Bond Lifetime (frames)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title('Lifetime Distribution', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        # Panel 5: Mean lifetime comparison
        ax5 = fig.add_subplot(gs[2, 1])
        mean_lifetimes = [hbond_results[cid]['mean_lifetime'] for cid in cluster_ids if cid in hbond_results]
        x_pos = np.arange(len(mean_lifetimes))
        
        bars = ax5.bar(x_pos, mean_lifetimes, color=colors[:len(mean_lifetimes)],
                      alpha=0.7, edgecolor='black')
        
        for bar, lifetime in zip(bars, mean_lifetimes):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{lifetime:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([f"C{cid}" for cid in cluster_ids if cid in hbond_results])
        ax5.set_ylabel('Mean Lifetime (frames)', fontsize=11, fontweight='bold')
        ax5.set_title('Average H-Bond Persistence', fontsize=12, fontweight='bold')
        ax5.grid(alpha=0.3, axis='y')
        
        # Info from first cluster
        data0 = hbond_results[cluster_ids[0]]
        donors = data0['donors']
        acceptors = data0['acceptors']
        dist_cut = data0['distance_cutoff']
        angle_cut = data0['angle_cutoff']
        
        fig.suptitle(f'Hydrogen Bond Analysis\n' +
                    f'Donors: {donors[:40]}... | Acceptors: {acceptors[:40]}...\n' +
                    f'Cutoffs: {dist_cut:.1f}Å, {angle_cut:.0f}°',
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_bridging_analysis(self, bridging_key: str,
                               cluster_ids: Optional[List[int]] = None,
                               
                               # Figure layout
                               figsize: Tuple[float, float] = (18, 12),
                               layout: str = '2x3',
                               
                               # Colors and styling
                               colors: Optional[List] = None,
                               cmap: str = 'viridis',
                               alpha_bars: float = 0.7,
                               alpha_hist: float = 0.6,
                               alpha_scatter: float = 0.5,
                               
                               # Title styling
                               show_main_title: bool = True,
                               main_title_fontsize: int = 14,
                               main_title_fontweight: str = 'bold',
                               show_subplot_titles: bool = True,
                               subplot_title_fontsize: int = 12,
                               subplot_title_fontweight: str = 'bold',
                               
                               # Axis label styling
                               label_fontsize: int = 11,
                               label_fontweight: str = 'bold',
                               tick_fontsize: int = 10,
                               tick_fontweight: str = 'normal',
                               
                               # Legend styling
                               show_legend: bool = True,
                               legend_fontsize: int = 10,
                               legend_fontweight: str = 'normal',
                               legend_framealpha: float = 0.8,
                               legend_loc: str = 'best',
                               
                               # Grid styling
                               show_grid: bool = True,
                               grid_alpha: float = 0.3,
                               grid_linestyle: str = '--',
                               grid_linewidth: float = 0.5,
                               
                               # Bar annotation control
                               show_bar_values: bool = True,
                               bar_value_fontsize: int = 9,
                               bar_value_fontweight: str = 'bold',
                               bar_value_format: str = '.2f',
                               
                               # Histogram bins
                               hist_bins: int = 30,
                               hist_range: Optional[Tuple[float, float]] = None,
                               
                               # Save options
                               save_path: Optional[str] = None,
                               dpi: Optional[int] = None):
        """
        Create comprehensive publication-ready bridging analysis visualization.
        
        Generates a multi-panel figure displaying all aspects of ion bridging:
        1. Bridge frequency comparison (bar chart)
        2. Bridge lifetime distributions (histograms)
        3. Mean bridge lifetime comparison (bar chart)
        4. Bridge geometry: Clay-Ion-Molecule angles (violin/box plots)
        5. Distance analysis during bridging (scatter/box plots)
        6. Coordination numbers during bridging (bar chart)
        
        Parameters
        ----------
        bridging_key : str
            Key in analyzer.bridging_data (format: 'clay__ion__molecule')
        cluster_ids : list of int, optional
            Clusters to plot (default: all available)
        
        Figure layout:
        figsize : tuple, default=(18, 12)
            Figure size (width, height) in inches
        layout : str, default='2x3'
            Panel layout: '2x3' (6 panels) or '2x2' (4 main panels)
        
        Colors and styling:
        colors : list, optional
            Custom colors for each cluster (default: uses Set2 colormap)
        cmap : str, default='viridis'
            Colormap for gradient plots
        alpha_bars : float, default=0.7
            Transparency for bar charts
        alpha_hist : float, default=0.6
            Transparency for histograms
        alpha_scatter : float, default=0.5
            Transparency for scatter plots
        
        Title styling:
        show_main_title : bool, default=True
            Show overall figure title
        main_title_fontsize : int, default=14
            Font size for main title
        main_title_fontweight : str, default='bold'
            Font weight for main title
        show_subplot_titles : bool, default=True
            Show individual panel titles
        subplot_title_fontsize : int, default=12
            Font size for subplot titles
        subplot_title_fontweight : str, default='bold'
            Font weight for subplot titles
        
        Axis labels:
        label_fontsize : int, default=11
            Font size for axis labels
        label_fontweight : str, default='bold'
            Font weight for axis labels
        tick_fontsize : int, default=10
            Font size for tick labels
        tick_fontweight : str, default='normal'
            Font weight for tick labels
        
        Legend:
        show_legend : bool, default=True
            Show legend
        legend_fontsize : int, default=10
            Legend font size
        legend_fontweight : str, default='normal'
            Legend font weight
        legend_framealpha : float, default=0.8
            Legend frame transparency
        legend_loc : str, default='best'
            Legend location
        
        Grid:
        show_grid : bool, default=True
            Show grid lines
        grid_alpha : float, default=0.3
            Grid transparency
        grid_linestyle : str, default='--'
            Grid line style
        grid_linewidth : float, default=0.5
            Grid line width
        
        Bar annotations:
        show_bar_values : bool, default=True
            Show values on top of bars
        bar_value_fontsize : int, default=9
            Font size for bar values
        bar_value_fontweight : str, default='bold'
            Font weight for bar values
        bar_value_format : str, default='.2f'
            Format string for bar values (e.g., '.1f', '.2f', '.0f')
        
        Histogram:
        hist_bins : int, default=30
            Number of histogram bins
        hist_range : tuple, optional
            Range for histogram bins (min, max)
        
        Save:
        save_path : str, optional
            Path to save figure  
        dpi : int, optional
            Figure DPI (default: self.default_dpi)
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        
        Example
        -------
        >>> # Create comprehensive bridging analysis plot
        >>> fig = plotter.plot_bridging_analysis(
        ...     'name Ob__resname NA__resname CIP and (name O1 or name O3)',
        ...     cluster_ids=[0, 1, 2],
        ...     figsize=(20, 14),
        ...     colors=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        ...     show_bar_values=True,
        ...     subplot_title_fontsize=14,
        ...     save_path='bridging_analysis.png',
        ...     dpi=300
        ... )
        """
        if not hasattr(self.analyzer, 'bridging_data') or bridging_key not in self.analyzer.bridging_data:
            raise ValueError(f"Bridging data not found for key: {bridging_key}")
        
        bridging_results = self.analyzer.bridging_data[bridging_key]
        
        if cluster_ids is None:
            cluster_ids = sorted(bridging_results.keys())
        
        if colors is None:
            colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_ids)))
        
        dpi = dpi or self.default_dpi
        
        # Create figure with layout
        if layout == '2x3':
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
        else:  # 2x2
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
        
        # ════════════════════════════════════════════════════════════════
        # Panel 1: Bridge Frequency Comparison
        # ════════════════════════════════════════════════════════════════
        ax1 = fig.add_subplot(gs[0, 0])
        bridge_freqs = [bridging_results[cid]['bridge_frequency'] 
                       for cid in cluster_ids if cid in bridging_results]
        x_pos = np.arange(len(bridge_freqs))
        
        bars1 = ax1.bar(x_pos, bridge_freqs, color=colors[:len(bridge_freqs)],
                       alpha=alpha_bars, edgecolor='black', linewidth=1.5)
        
        if show_bar_values:
            for bar, freq in zip(bars1, bridge_freqs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height,
                        f'{freq:{bar_value_format}}%',
                        ha='center', va='bottom', 
                        fontsize=bar_value_fontsize,
                        fontweight=bar_value_fontweight)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"Cluster {cid}" for cid in cluster_ids if cid in bridging_results],
                           fontsize=tick_fontsize, fontweight=tick_fontweight)
        ax1.set_ylabel('Bridge Frequency (%)', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_subplot_titles:
            ax1.set_title('Bridging Configuration Frequency', 
                         fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        if show_grid:
            ax1.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth, axis='y')
        
        # ════════════════════════════════════════════════════════════════
        # Panel 2: Bridge Lifetime Distributions
        # ════════════════════════════════════════════════════════════════
        ax2 = fig.add_subplot(gs[0, 1])
        
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id in bridging_results:
                data = bridging_results[cluster_id]
                lifetimes = data['bridge_lifetimes']
                if len(lifetimes) > 0:
                    ax2.hist(lifetimes, bins=hist_bins, range=hist_range,
                            alpha=alpha_hist, color=colors[i],
                            label=f'Cluster {cluster_id}', edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Bridge Lifetime (frames)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax2.set_ylabel('Frequency', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_subplot_titles:
            ax2.set_title('Bridge Persistence Distribution', 
                         fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        if show_legend:
            legend = ax2.legend(fontsize=legend_fontsize, framealpha=legend_framealpha, loc=legend_loc)
            for text in legend.get_texts():
                text.set_fontweight(legend_fontweight)
        if show_grid:
            ax2.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
        
        # ════════════════════════════════════════════════════════════════
        # Panel 3: Mean Bridge Lifetime Comparison
        # ════════════════════════════════════════════════════════════════
        ax3 = fig.add_subplot(gs[0, 2])
        mean_lifetimes = [bridging_results[cid]['mean_bridge_lifetime'] 
                         for cid in cluster_ids if cid in bridging_results]
        x_pos = np.arange(len(mean_lifetimes))
        
        bars3 = ax3.bar(x_pos, mean_lifetimes, color=colors[:len(mean_lifetimes)],
                       alpha=alpha_bars, edgecolor='black', linewidth=1.5)
        
        if show_bar_values:
            for bar, lifetime in zip(bars3, mean_lifetimes):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height,
                        f'{lifetime:{bar_value_format}}',
                        ha='center', va='bottom',
                        fontsize=bar_value_fontsize,
                        fontweight=bar_value_fontweight)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f"Cluster {cid}" for cid in cluster_ids if cid in bridging_results],
                           fontsize=tick_fontsize, fontweight=tick_fontweight)
        ax3.set_ylabel('Mean Lifetime (frames)', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_subplot_titles:
            ax3.set_title('Average Bridge Persistence', 
                         fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
        ax3.tick_params(axis='both', labelsize=tick_fontsize)
        if show_grid:
            ax3.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth, axis='y')
        
        # ════════════════════════════════════════════════════════════════
        # Panel 4: Bridge Geometry (Angles)
        # ════════════════════════════════════════════════════════════════
        ax4 = fig.add_subplot(gs[1, 0])
        
        angle_data = []
        angle_labels = []
        for cluster_id in cluster_ids:
            if cluster_id in bridging_results:
                angles = bridging_results[cluster_id]['bridge_angles']
                if len(angles) > 0:
                    angle_data.append(angles)
                    angle_labels.append(f"C{cluster_id}")
        
        if len(angle_data) > 0:
            bp = ax4.boxplot(angle_data, labels=angle_labels,
                            patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
            
            for patch, color in zip(bp['boxes'], colors[:len(angle_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(alpha_bars)
        
        ax4.set_xlabel('Cluster', fontsize=label_fontsize, fontweight=label_fontweight)
        ax4.set_ylabel('Clay-Ion-Molecule Angle (°)', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_subplot_titles:
            ax4.set_title('Bridge Geometry (Linearity)', 
                         fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
        ax4.tick_params(axis='both', labelsize=tick_fontsize)
        if show_grid:
            ax4.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth, axis='y')
        
        # ════════════════════════════════════════════════════════════════
        # Panel 5: Distance Analysis
        # ════════════════════════════════════════════════════════════════
        ax5 = fig.add_subplot(gs[1, 1])
        
        x_pos = np.arange(len(cluster_ids))
        width = 0.35
        
        dist_clay = [bridging_results[cid]['mean_distance_clay'] 
                    for cid in cluster_ids if cid in bridging_results]
        dist_mol = [bridging_results[cid]['mean_distance_mol'] 
                   for cid in cluster_ids if cid in bridging_results]
        
        bars5a = ax5.bar(x_pos - width/2, dist_clay, width,
                        label='Clay-Ion', alpha=alpha_bars, 
                        color='skyblue', edgecolor='black', linewidth=1)
        bars5b = ax5.bar(x_pos + width/2, dist_mol, width,
                        label='Ion-Molecule', alpha=alpha_bars,
                        color='salmon', edgecolor='black', linewidth=1)
        
        if show_bar_values:
            for bar in bars5a:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2, height,
                        f'{height:.2f}',
                        ha='center', va='bottom',
                        fontsize=bar_value_fontsize-1)
            for bar in bars5b:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2, height,
                        f'{height:.2f}',
                        ha='center', va='bottom',
                        fontsize=bar_value_fontsize-1)
        
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([f"C{cid}" for cid in cluster_ids if cid in bridging_results],
                           fontsize=tick_fontsize, fontweight=tick_fontweight)
        ax5.set_ylabel('Distance (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_subplot_titles:
            ax5.set_title('Bridge Distances', 
                         fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
        ax5.tick_params(axis='both', labelsize=tick_fontsize)
        if show_legend:
            legend = ax5.legend(fontsize=legend_fontsize, framealpha=legend_framealpha)
            for text in legend.get_texts():
                text.set_fontweight(legend_fontweight)
        if show_grid:
            ax5.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth, axis='y')
        
        # ════════════════════════════════════════════════════════════════
        # Panel 6: Coordination Numbers
        # ════════════════════════════════════════════════════════════════
        if layout == '2x3':
            ax6 = fig.add_subplot(gs[1, 2])
            
            x_pos = np.arange(len(cluster_ids))
            width = 0.35
            
            coord_clay = [bridging_results[cid]['mean_coordination_clay'] 
                         for cid in cluster_ids if cid in bridging_results]
            coord_mol = [bridging_results[cid]['mean_coordination_mol'] 
                        for cid in cluster_ids if cid in bridging_results]
            
            bars6a = ax6.bar(x_pos - width/2, coord_clay, width,
                            label='Clay', alpha=alpha_bars,
                            color='lightgreen', edgecolor='black', linewidth=1)
            bars6b = ax6.bar(x_pos + width/2, coord_mol, width,
                            label='Molecule', alpha=alpha_bars,
                            color='plum', edgecolor='black', linewidth=1)
            
            if show_bar_values:
                for bar in bars6a:
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2, height,
                            f'{height:.1f}',
                            ha='center', va='bottom',
                            fontsize=bar_value_fontsize-1)
                for bar in bars6b:
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2, height,
                            f'{height:.1f}',
                            ha='center', va='bottom',
                            fontsize=bar_value_fontsize-1)
            
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels([f"C{cid}" for cid in cluster_ids if cid in bridging_results],
                               fontsize=tick_fontsize, fontweight=tick_fontweight)
            ax6.set_ylabel('Coordination Number', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_subplot_titles:
                ax6.set_title('Ion Coordination During Bridging', 
                             fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight)
            ax6.tick_params(axis='both', labelsize=tick_fontsize)
            if show_legend:
                legend = ax6.legend(fontsize=legend_fontsize, framealpha=legend_framealpha)
                for text in legend.get_texts():
                    text.set_fontweight(legend_fontweight)
            if show_grid:
                ax6.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth, axis='y')
        
        # ════════════════════════════════════════════════════════════════
        # Main title with configuration info
        # ════════════════════════════════════════════════════════════════
        if show_main_title:
            data0 = bridging_results[cluster_ids[0]]
            clay_sel = data0['clay_sel'][:30]
            ion_sel = data0['ion_sel'][:20]
            mol_sel = data0['molecule_sel'][:30]
            cutoff_clay = data0['cutoff_clay_ion']
            cutoff_mol = data0['cutoff_ion_molecule']
            angle_thresh = data0['angle_threshold']
            
            fig.suptitle(f'Ion Bridging Analysis\n' +
                        f'Clay: {clay_sel}... | Ion: {ion_sel} | Molecule: {mol_sel}...\n' +
                        f'Cutoffs: {cutoff_clay:.1f}Å (clay), {cutoff_mol:.1f}Å (mol) | Angle > {angle_thresh:.0f}°',
                        fontsize=main_title_fontsize, fontweight=main_title_fontweight)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def _get_ion_label(self, ion_sel: str) -> str:
        """Extract ion name from selection string and format with charge superscript.
        
        Parameters
        ----------
        ion_sel : str
            Ion selection string (e.g., 'resname NA', 'name Na')
            
        Returns
        -------
        str
            Formatted ion name with charge in LaTeX math mode (e.g., '$Na^{+}$', '$Cl^{-}$', '$Mg^{2+}$')
        """
        # Common ion charge mapping (using LaTeX-style superscripts)
        ion_charges = {
            'NA': '^{+}',      # Sodium
            'Na': '^{+}',
            'K': '^{+}',       # Potassium
            'LI': '^{+}',      # Lithium
            'Li': '^{+}',
            'CL': '^{-}',      # Chloride
            'Cl': '^{-}',
            'BR': '^{-}',      # Bromide
            'Br': '^{-}',
            'I': '^{-}',       # Iodide
            'F': '^{-}',       # Fluoride
            'MG': '^{2+}',     # Magnesium
            'Mg': '^{2+}',
            'CA': '^{2+}',     # Calcium
            'Ca': '^{2+}',
            'ZN': '^{2+}',     # Zinc
            'Zn': '^{2+}',
            'FE': '^{2+}',     # Iron (assuming Fe2+)
            'Fe': '^{2+}',
            'AL': '^{3+}',     # Aluminum
            'Al': '^{3+}',
        }
        
        # Try to extract ion name from selection string
        import re
        # Look for patterns like 'resname NA', 'name Na', 'resname NA and...'
        patterns = [
            r'resname\s+(\w+)',
            r'name\s+(\w+)',
            r'type\s+(\w+)'
        ]
        
        ion_name = None
        for pattern in patterns:
            match = re.search(pattern, ion_sel, re.IGNORECASE)
            if match:
                ion_name = match.group(1)
                break
        
        if not ion_name:
            # Fallback: return 'Ion' if can't parse
            return 'Ion'
        
        # Get charge symbol
        # Try both uppercase and capitalized versions
        charge = ion_charges.get(ion_name.upper(), ion_charges.get(ion_name.capitalize(), ''))
        
        # Format with proper capitalization
        formatted_name = ion_name.capitalize() if len(ion_name) > 1 else ion_name.upper()
        
        # Return in LaTeX math mode for proper superscript rendering
        return f"${formatted_name}{charge}$"
    
    def plot_angle_comparison(self, 
                             angle_comparison_df,
                             figsize: Tuple[float, float] = (14, 10),
                             colors: Optional[List[str]] = None,
                             # Title and labels
                             show_title: bool = False,
                             show_subplot_titles: bool = True,
                             main_title: str = 'Angle Threshold Comparison',
                             main_title_fontsize: int = 24,
                             main_title_fontweight: str = 'bold',
                             title_fontweight: str = 'bold',
                             label_fontsize: int = 22,
                             label_fontweight: str = 'bold',
                             tick_fontsize: int = 18,
                             tick_fontweight: str = 'normal',
                             # Legend
                             show_legend: bool = True,
                             legend_fontsize: int = 10,
                             legend_fontweight: str = 'normal',
                             legend_loc: str = 'best',
                             legend_framealpha: float = 0.9,
                             # Grid
                             show_grid: bool = True,
                             grid_alpha: float = 0.3,
                             grid_linestyle: str = '--',
                             grid_linewidth: float = 0.5,
                             # Styling
                             marker_size: float = 8,
                             line_width: float = 2,
                             alpha: float = 0.7,
                             # Save control
                             save_path: Optional[str] = None,
                             dpi: int = 300,
                             save_individual_figures: bool = True,
                             individual_save_dir: Optional[str] = None,
                             individual_figsize: Tuple[float, float] = (8, 6)) -> plt.Figure:
        """
        Visualize angle threshold comparison showing effect on bridging detection.
        
        Parameters
        ----------
        angle_comparison_df : pandas.DataFrame
            DataFrame from generate_bridging_report with angle comparison
        figsize : tuple, default=(14, 10)
            Figure size (width, height)
        colors : list of str, optional
            Colors for different clusters
        show_title : bool, default=False
            Show main figure title
        show_subplot_titles : bool, default=True
            Show individual panel titles
        ... (styling parameters)
        save_path : str, optional
            Path to save combined figure
        dpi : int, default=300
            Resolution for saved figures (dots per inch)
        save_individual_figures : bool, default=True
            If True, save each of the 4 panels as separate individual figure files
        individual_save_dir : str, optional
            Directory to save individual panel figures. If None, uses current directory
        individual_figsize : tuple, default=(8, 6)
            Figure size (width, height) in inches for individual saved plots
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        import pandas as pd
        
        if not isinstance(angle_comparison_df, pd.DataFrame):
            raise ValueError("angle_comparison_df must be a pandas DataFrame")
        
        if colors is None:
            colors = plt.cm.Set2(np.linspace(0, 1, len (angle_comparison_df['Cluster'].unique())))
        
        # Try to extract ion name from bridging data for better labels
        ion_label = 'Ion'  # Default
        try:
            # Get the bridging data key to extract ion_sel
            if hasattr(self.analyzer, 'bridging_data') and len(self.analyzer.bridging_data) > 0:
                # Get first bridging data entry
                first_key = list(self.analyzer.bridging_data.keys())[0]
                bridging_results = self.analyzer.bridging_data[first_key]
                # Get ion_sel from first cluster
                first_cluster_id = list(bridging_results.keys())[0]
                ion_sel = bridging_results[first_cluster_id].get('ion_sel', '')
                ion_label = self._get_ion_label(ion_sel)
        except Exception as e:
            # If extraction fails, use default 'Ion'
            pass
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # ═══════════════════════════════════════════════════════════
        # Panel 1: Frequency vs Angle Threshold
        # ═══════════════════════════════════════════════════════════
        ax1 = fig.add_subplot(gs[0, 0])
        
        for i, cluster_id in enumerate(sorted(angle_comparison_df['Cluster'].unique())):
            cluster_data = angle_comparison_df[angle_comparison_df['Cluster'] == cluster_id]
            ax1.plot(cluster_data['Angle_Threshold'], cluster_data['Frequency_Percent'],
                    marker='o', markersize=marker_size, linewidth=line_width,
                    color=colors[i], alpha=alpha, label=f'Cluster {cluster_id}')
            # Note: No error bars for frequency (it's a percentage)
        
        ax1.set_xlabel('Angle Threshold (°)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax1.set_ylabel('Bridging Frequency (%)', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_subplot_titles:
            ax1.set_title('Effect of Angle Threshold', fontsize=label_fontsize, fontweight=title_fontweight)
        ax1.tick_params(axis='both', labelsize=tick_fontsize, labelcolor='black')
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight(tick_fontweight)
        if show_legend:
            legend = ax1.legend(fontsize=legend_fontsize, framealpha=legend_framealpha, loc=legend_loc)
            for text in legend.get_texts():
                text.set_fontweight(legend_fontweight)
        if show_grid:
            ax1.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
        
        # ═══════════════════════════════════════════════════════════
        # Panel 2: Number of Events vs Angle
        # ═══════════════════════════════════════════════════════════
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Aggregate by angle threshold
        agg_data = angle_comparison_df.groupby('Angle_Threshold').agg({
            'Events': 'sum',
            'Cone_Width': 'first'
        }).reset_index()
        
        ax2.bar(agg_data['Angle_Threshold'], agg_data['Events'],
               width=8, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
        
        ax2.set_xlabel('Angle Threshold (°)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax2.set_ylabel('Total Bridge Events', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_subplot_titles:
            ax2.set_title('Total Events Detected', fontsize=label_fontsize, fontweight=title_fontweight)
        ax2.tick_params(axis='both', labelsize=tick_fontsize, labelcolor='black')
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontweight(tick_fontweight)
        if show_grid:
            ax2.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth, axis='y')
        
        # ═══════════════════════════════════════════════════════════
        # Panel 3: Mean Detected Angle vs Threshold
        # ═══════════════════════════════════════════════════════════
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Calculate mean and std across clusters for each threshold
        angle_stats = angle_comparison_df.groupby('Angle_Threshold').agg({
            'Mean_Angle': 'mean',
            'Std_Angle': lambda x: np.sqrt(np.mean(x**2))  # RMS of individual stds
        })
        
        # Plot with error bars
        ax3.errorbar(angle_stats.index, angle_stats['Mean_Angle'].values, 
                    yerr=angle_stats['Std_Angle'].values,
                    marker='o', markersize=marker_size+2,
                    linewidth=line_width, color='crimson', alpha=alpha,
                    capsize=5, capthick=line_width, elinewidth=line_width*0.7)
        ax3.plot([90, 160], [90, 160], 'k--', alpha=0.3, linewidth=1, label='y=x reference')
        
        ax3.set_xlabel('Angle Threshold (°)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax3.set_ylabel('Mean Detected Angle (°)', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_subplot_titles:
            ax3.set_title('Angle Selectivity', fontsize=label_fontsize, fontweight=title_fontweight)
        ax3.tick_params(axis='both', labelsize=tick_fontsize, labelcolor='black')
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_fontweight(tick_fontweight)
        ax3.legend(fontsize=legend_fontsize-2, framealpha=legend_framealpha)
        if show_grid:
            ax3.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
        
        # ═══════════════════════════════════════════════════════════
        # Panel 4: Distance Statistics vs Angle
        # ═══════════════════════════════════════════════════════════
        ax4 = fig.add_subplot(gs[1, 1])
        
        dist_data = angle_comparison_df.groupby('Angle_Threshold').agg({
            'Mean_Clay_Dist': 'mean',
            'Std_Clay_Dist': lambda x: np.sqrt(np.mean(x**2)),
            'Mean_Mol_Dist': 'mean',
            'Std_Mol_Dist': lambda x: np.sqrt(np.mean(x**2))
        })
        
        ax4.errorbar(dist_data.index, dist_data['Mean_Clay_Dist'], 
                    yerr=dist_data['Std_Clay_Dist'],
                    marker='o', markersize=marker_size, linewidth=line_width, 
                    color='brown', alpha=alpha, label=f'Clay-{ion_label}',
                    capsize=5, capthick=line_width, elinewidth=line_width*0.7)
        ax4.errorbar(dist_data.index, dist_data['Mean_Mol_Dist'], 
                    yerr=dist_data['Std_Mol_Dist'],
                    marker='o', markersize=marker_size, linewidth=line_width, 
                    color='green', alpha=alpha, label=f'{ion_label}-CIP',
                    capsize=5, capthick=line_width, elinewidth=line_width*0.7)
        
        ax4.set_xlabel('Angle Threshold (°)', fontsize=label_fontsize, fontweight=label_fontweight)
        ax4.set_ylabel('Mean Distance (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
        if show_subplot_titles:
            ax4.set_title('Bridge Distances', fontsize=label_fontsize, fontweight=title_fontweight)
        ax4.tick_params(axis='both', labelsize=tick_fontsize, labelcolor='black')
        for label in ax4.get_xticklabels() + ax4.get_yticklabels():
            label.set_fontweight(tick_fontweight)
        ax4.legend(fontsize=legend_fontsize, framealpha=legend_framealpha)
        if show_grid:
            ax4.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
        
        if show_title:
            fig.suptitle(main_title, fontsize=main_title_fontsize, fontweight=main_title_fontweight)
        
        plt.tight_layout()
        
        # Save individual panel figures if requested
        if save_individual_figures:
            import os
            
            # Determine save directory
            if individual_save_dir is None:
                individual_save_dir = os.getcwd()
            
            # Create directory if needed
            os.makedirs(individual_save_dir, exist_ok=True)
            
            # ===== PANEL 1: Frequency vs Angle =====
            fig1 = plt.figure(figsize=individual_figsize)
            ax1_ind = fig1.add_subplot(111)
            
            for i, cluster_id in enumerate(sorted(angle_comparison_df['Cluster'].unique())):
                cluster_data = angle_comparison_df[angle_comparison_df['Cluster'] == cluster_id]
                ax1_ind.plot(cluster_data['Angle_Threshold'], cluster_data['Frequency_Percent'],
                           marker='o', markersize=marker_size, linewidth=line_width,
                           color=colors[i], alpha=alpha, label=f'Cluster {cluster_id}')
                # Note: No error bars for frequency (it's a percentage)
            
            ax1_ind.set_xlabel('Angle Threshold (°)', fontsize=label_fontsize, fontweight=label_fontweight)
            ax1_ind.set_ylabel('Bridging Frequency (%)', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_subplot_titles:
                ax1_ind.set_title('Effect of Angle Threshold', fontsize=label_fontsize, fontweight=title_fontweight)
            ax1_ind.tick_params(axis='both', labelsize=tick_fontsize, labelcolor='black')
            for label in ax1_ind.get_xticklabels() + ax1_ind.get_yticklabels():
                label.set_fontweight(tick_fontweight)
            if show_legend:
                legend = ax1_ind.legend(fontsize=legend_fontsize, framealpha=legend_framealpha, loc=legend_loc)
                for text in legend.get_texts():
                    text.set_fontweight(legend_fontweight)
            if show_grid:
                ax1_ind.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
            
            plt.tight_layout()
            path1 = os.path.join(individual_save_dir, 'angle_comparison_frequency.png')
            fig1.savefig(path1, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved Panel 1: {path1}")
            plt.close(fig1)
            
            # ===== PANEL 2: Total Events =====
            fig2 = plt.figure(figsize=individual_figsize)
            ax2_ind = fig2.add_subplot(111)
            
            ax2_ind.bar(agg_data['Angle_Threshold'], agg_data['Events'],
                       width=8, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
            
            ax2_ind.set_xlabel('Angle Threshold (°)', fontsize=label_fontsize, fontweight=label_fontweight)
            ax2_ind.set_ylabel('Total Bridge Events', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_subplot_titles:
                ax2_ind.set_title('Total Events Detected', fontsize=label_fontsize, fontweight=title_fontweight)
            ax2_ind.tick_params(axis='both', labelsize=tick_fontsize, labelcolor='black')
            for label in ax2_ind.get_xticklabels() + ax2_ind.get_yticklabels():
                label.set_fontweight(tick_fontweight)
            if show_grid:
                ax2_ind.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth, axis='y')
            
            plt.tight_layout()
            path2 = os.path.join(individual_save_dir, 'angle_comparison_events.png')
            fig2.savefig(path2, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved Panel 2: {path2}")
            plt.close(fig2)
            
            # ===== PANEL 3: Angle Selectivity =====
            fig3 = plt.figure(figsize=individual_figsize)
            ax3_ind = fig3.add_subplot(111)
            
            # Recalculate angle_stats for individual plot (in case it's not in scope)
            angle_stats_ind = angle_comparison_df.groupby('Angle_Threshold').agg({
                'Mean_Angle': 'mean',
                'Std_Angle': lambda x: np.sqrt(np.mean(x**2))
            })
            
            ax3_ind.errorbar(angle_stats_ind.index, angle_stats_ind['Mean_Angle'].values,
                           yerr=angle_stats_ind['Std_Angle'].values,
                           marker='s', markersize=marker_size+2,
                           linewidth=line_width, color='crimson', alpha=alpha,
                           capsize=5, capthick=line_width, elinewidth=line_width*0.7)
            ax3_ind.plot([90, 160], [90, 160], 'k--', alpha=0.3, linewidth=1, label='y=x reference')
            
            ax3_ind.set_xlabel('Angle Threshold (°)', fontsize=label_fontsize, fontweight=label_fontweight)
            ax3_ind.set_ylabel('Mean Detected Angle (°)', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_subplot_titles:
                ax3_ind.set_title('Angle Selectivity', fontsize=label_fontsize, fontweight=title_fontweight)
            ax3_ind.tick_params(axis='both', labelsize=tick_fontsize, labelcolor='black')
            for label in ax3_ind.get_xticklabels() + ax3_ind.get_yticklabels():
                label.set_fontweight(tick_fontweight)
            ax3_ind.legend(fontsize=legend_fontsize-2, framealpha=legend_framealpha)
            if show_grid:
                ax3_ind.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
            
            plt.tight_layout()
            path3 = os.path.join(individual_save_dir, 'angle_comparison_selectivity.png')
            fig3.savefig(path3, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved Panel 3: {path3}")
            plt.close(fig3)
            
            # ===== PANEL 4: Bridge Distances =====
            fig4 = plt.figure(figsize=individual_figsize)
            ax4_ind = fig4.add_subplot(111)
            
            # Recalculate dist_data for individual plot (in case it's not in scope)
            dist_data_ind = angle_comparison_df.groupby('Angle_Threshold').agg({
                'Mean_Clay_Dist': 'mean',
                'Std_Clay_Dist': lambda x: np.sqrt(np.mean(x**2)),
                'Mean_Mol_Dist': 'mean',
                'Std_Mol_Dist': lambda x: np.sqrt(np.mean(x**2))
            })
            
            ax4_ind.errorbar(dist_data_ind.index, dist_data_ind['Mean_Clay_Dist'],
                           yerr=dist_data_ind['Std_Clay_Dist'],
                           marker='o', markersize=marker_size, linewidth=line_width,
                           color='brown', alpha=alpha, label=f'Clay-{ion_label}',
                           capsize=5, capthick=line_width, elinewidth=line_width*0.7)
            ax4_ind.errorbar(dist_data_ind.index, dist_data_ind['Mean_Mol_Dist'],
                           yerr=dist_data_ind['Std_Mol_Dist'],
                           marker='o', markersize=marker_size, linewidth=line_width,
                           color='green', alpha=alpha, label=f'{ion_label}-CIP',
                           capsize=5, capthick=line_width, elinewidth=line_width*0.7)
            
            ax4_ind.set_xlabel('Angle Threshold (°)', fontsize=label_fontsize, fontweight=label_fontweight)
            ax4_ind.set_ylabel('Mean Distance (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
            if show_subplot_titles:
                ax4_ind.set_title('Bridge Distances', fontsize=label_fontsize, fontweight=title_fontweight)
            ax4_ind.tick_params(axis='both', labelsize=tick_fontsize, labelcolor='black')
            for label in ax4_ind.get_xticklabels() + ax4_ind.get_yticklabels():
                label.set_fontweight(tick_fontweight)
            ax4_ind.legend(fontsize=legend_fontsize, framealpha=legend_framealpha)
            if show_grid:
                ax4_ind.grid(alpha=grid_alpha, ls=grid_linestyle, lw=grid_linewidth)
            
            plt.tight_layout()
            path4 = os.path.join(individual_save_dir, 'angle_comparison_distances.png')
            fig4.savefig(path4, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved Panel 4: {path4}")
            plt.close(fig4)
        
        # Save combined figure
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_bridge_snapshots_3d(self,
                                bridge_frames_info: List[Dict],
                                max_frames: int = 9,
                                figsize: Tuple[float, float] = (20, 12),
                                sample_clay_atoms: int = 50,
                                # Styling
                                clay_color: str = 'brown',
                                clay_alpha: float = 0.3,
                                clay_size: int = 20,
                                ion_color: str = 'blue',
                                ion_size: int = 200,
                                nearest_clay_color: str = 'red',
                                nearest_clay_size: int = 150,
                                mol_color: str = 'green',
                                mol_size: int = 150,
                                bridge_line_color: str = 'black',
                                bridge_line_width: float = 2,
                                bridge_line_alpha: float = 0.7,
                                # Labels and titles
                                show_title: bool = True,
                                title_fontsize: int = 10,
                                label_fontsize: int = 9,
                                show_legend: bool = True,
                                legend_fontsize: int = 8,
                                # Output
                                dpi: int = 150,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 3D snapshots of bridge configurations.
        
        Parameters
        ----------
        bridge_frames_info : list of dict
            List of bridge frame information from generate_bridging_report
        max_frames : int, default=9
            Maximum number of frames to visualize
        figsize : tuple, default=(20, 12)
            Figure size
        sample_clay_atoms : int, default=50
            Number of clay atoms to sample for visualization
        ... (styling parameters)
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        if not bridge_frames_info:
            print("No bridge frames to visualize")
            return None
        
        # Limit to max_frames
        frames_to_plot = bridge_frames_info[:max_frames]
        n_frames = len(frames_to_plot)
        
        # Calculate grid dimensions
        ncols = 3
        nrows = int(np.ceil(n_frames / ncols))
        
        fig = plt.figure(figsize=figsize)
        
        for plot_num, frame_info in enumerate(frames_to_plot, 1):
            cluster_id = frame_info['Cluster']
            actual_frame = frame_info['Actual_Frame']
            angle = frame_info['Angle']
            clay_dist = frame_info['Clay_Dist']
            mol_dist = frame_info['Mol_Dist']
            
            # Get universe and load frame
            u = self.analyzer.trajectory_data[cluster_id]['universe']
            u.trajectory[actual_frame]
            
            # Get bridge frame data
            bridging_results = self.analyzer.bridging_data
            key = list(bridging_results.keys())[0]
            data = bridging_results[key][cluster_id]
            
            clay_sel = data['clay_sel']
            ion_sel = data['ion_sel']
            mol_sel = data['molecule_sel']
            
            clay_atoms = u.select_atoms(clay_sel)
            na_ions = u.select_atoms(ion_sel)
            mol_atoms = u.select_atoms(mol_sel)
            
            # Find the bridging Na+
            min_sum_dist = float('inf')
            bridge_na_pos = None
            
            for na in na_ions:
                d_clay = np.min(np.linalg.norm(clay_atoms.positions - na.position, axis=1))
                d_mol = np.min(np.linalg.norm(mol_atoms.positions - na.position, axis=1))
                if d_clay < 4.0 and d_mol < 4.0:
                    if d_clay + d_mol < min_sum_dist:
                        min_sum_dist = d_clay + d_mol
                        bridge_na_pos = na.position
            
            if bridge_na_pos is None:
                continue
            
            # Get nearest clay and molecule positions
            clay_dists = np.linalg.norm(clay_atoms.positions - bridge_na_pos, axis=1)
            mol_dists = np.linalg.norm(mol_atoms.positions - bridge_na_pos, axis=1)
            
            nearest_clay_pos = clay_atoms.positions[np.argmin(clay_dists)]
            nearest_mol_pos = mol_atoms.positions[np.argmin(mol_dists)]
            
            # Create 3D subplot
            ax = fig.add_subplot(nrows, ncols, plot_num, projection='3d')
            
            # Sample clay surface atoms
            clay_sample_indices = np.linspace(0, len(clay_atoms)-1, sample_clay_atoms, dtype=int)
            clay_sample = clay_atoms.positions[clay_sample_indices]
            
            ax.scatter(clay_sample[:, 0], clay_sample[:, 1], clay_sample[:, 2],
                      c=clay_color, s=clay_size, alpha=clay_alpha, label='Clay Ob')
            
            # Plot bridging Na+
            ax.scatter([bridge_na_pos[0]], [bridge_na_pos[1]], [bridge_na_pos[2]],
                      c=ion_color, s=ion_size, marker='o', edgecolors='black',
                      linewidths=2, label='Bridge Na+')
            
            # Plot nearest clay and molecule oxygens
            ax.scatter([nearest_clay_pos[0]], [nearest_clay_pos[1]], [nearest_clay_pos[2]],
                      c=nearest_clay_color, s=nearest_clay_size, marker='s',
                      edgecolors='black', linewidths=2, label='Nearest Ob')
            ax.scatter([nearest_mol_pos[0]], [nearest_mol_pos[1]], [nearest_mol_pos[2]],
                      c=mol_color, s=mol_size, marker='^',
                      edgecolors='black', linewidths=2, label='Mol O')
            
            # Draw bridge lines
            ax.plot([nearest_clay_pos[0], bridge_na_pos[0], nearest_mol_pos[0]],
                   [nearest_clay_pos[1], bridge_na_pos[1], nearest_mol_pos[1]],
                   [nearest_clay_pos[2], bridge_na_pos[2], nearest_mol_pos[2]],
                   color=bridge_line_color, ls='--', linewidth=bridge_line_width,
                   alpha=bridge_line_alpha)
            
            ax.set_xlabel('X (Å)', fontsize=label_fontsize)
            ax.set_ylabel('Y (Å)', fontsize=label_fontsize)
            ax.set_zlabel('Z (Å)', fontsize=label_fontsize)
            
            if show_title:
                ax.set_title(f'Cluster {cluster_id}, Frame {actual_frame}\n'
                           f'Angle: {angle:.1f}°, d(clay)={clay_dist:.2f}Å, d(mol)={mol_dist:.2f}Å',
                           fontsize=title_fontsize)
            
            if plot_num == 1 and show_legend:
                ax.legend(fontsize=legend_fontsize, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_bridging_geometry_3d(self, 
                                 bridging_results: dict,
                                 cluster_id: int,
                                 max_bridges: int = 4,
                                 min_bridges: int = 1,
                                 sort_by: str = 'frequency',
                                 angle_min: Optional[float] = None,
                                 angle_max: Optional[float] = None,
                                 distance_clay_max:Optional[float] = None,
                                 distance_mol_max: Optional[float] = None,
                                 show_surface_atoms: bool = True,
                                 surface_radius: float = 10.0,
                                 surface_z_thickness: float = 10.0,
                                 surface_floor_value: float = 1.0,
                                 surface_ceiling_value: float = 0.0,
                                 surface_sel: str = 'name Si or name Ob or name Op or name Ohs or name Mgo',
                                 molecule_sel: Optional[str] = None,
                                 molecule_radius: float = 10.0,
                                 show_waters: bool = False,
                                 water_radius: float = 5.0,
                                 bridge_line_color: str = 'black',
                                 bridge_linewidth: float = 4.0,
                                 atom_scale_factor: float = 100,
                                 molecule_alpha: float = 0.7,
                                 surface_alpha: float = 0.6,
                                 water_alpha: float = 0.4,
                                 carbon_color: str = "#000000",
                                 si_color: str = "#F0C8A0",
                                 show_surface_bonds: bool = False,
                                 surface_bond_color: str = 'orange',
                                 surface_bond_linewidth: float = 1.0,
                                 boundary_linewidth: float = 0.5,
                                 view_elevation: float = 25,
                                 view_azimuth: float = 45,
                                 figsize_per_panel: float = 5,
                                 dpi: int = 300,
                                 save_path: Optional[str] = None,
                                 save_combined_figure: bool = True,
                                 save_individual_figures: bool = True,
                                 individual_figsize: tuple = (8, 6),
                                 title_fontsize: int = 14,
                                 title_fontweight: str = 'bold',
                                 label_fontsize: int = 16,
                                 label_fontweight: str = 'bold',
                                 tick_fontsize: int = 14,
                                 bridge_linestyle: str = 'dashed',
                                 bond_style: str = 'solid',
                                 atom_edge_style: str = 'solid',
                                 surface_atom_style: str = 'scatter',
                                 show_angle_arc: bool = True,
                                 angle_arc_radius: float = 3.0,
                                 angle_arc_color: str = 'cyan',
                                 angle_arc_linewidth: float = 2.0,
                                 show_title: bool = True,
                                 axis_info: str = 'detailed',
                                 use_saved_data: bool = False,
                                 start_search_frame: int = 1,
                                 skip_when_searching: int = 1):
        """
        Plot 3D bridging geometry showing Clay-Ion-Molecule arrangements.
        
        Creates publication-quality 3D visualization of ion bridging configurations with:
        - Van der Waals-sized atomic spheres
        - Clean white background
        - Side-by-side panels for multiple bridging events
        - Clay surface atoms around bridging site
        - Full molecule structure for context
        - Bridging angle and distance annotations
        - Professional styling matching plot_hbond_geometry_3d()
        
        This method works similar to plot_hbond_geometry_3d() but for visualizing
        Clay-Ion-Molecule bridging configurations from generate_bridging_report().
        
        Parameters
        ----------
        bridging_results : dict
            Results from analyzer.bridging_data or generate_bridging_report()
            Expected structure: {key: {cluster_id: {...bridging_data...}}}
        cluster_id : int
            Cluster to visualize
        max_bridges : int, default=4
            Maximum number of bridging events to show (side-by-side panels)
        min_bridges : int, default=1
            Minimum number of valid bridging events required to create visualization.
            If fewer than min_bridges are found with frames meeting criteria,
            the visualization is skipped.
        sort_by : str, default='frequency'
            How to select bridging events: 'frequency', 'lifetime', 'angle', 'random'
        angle_min : float, optional
            Minimum Clay-Ion-Molecule angle to display (degrees)
            This is the ACTUAL angle at the ion vertex (not deviation).
            Example: angle_min=120.0 means show bridges with angle ≥ 120° (deviation ≤ 60°)
            If None, no minimum filter applied
        angle_max : float, optional
            Maximum Clay-Ion-Molecule angle to display (degrees)
            This is the ACTUAL angle at the ion vertex (not deviation).
            Example: angle_max=160.0 means show bridges with angle ≤ 160° (deviation ≥ 20°)
            If None, no maximum filter applied
        distance_clay_max : float, optional
            Maximum Clay-Ion distance to display (Å)
            If None, no filter applied
        distance_mol_max : float, optional
            Maximum Ion-Molecule distance to display (Å)
            If None, no filter applied
        show_surface_atoms : bool, default=True
            Show clay surface atoms around ion site
            If no atoms appear, try: increasing surface_radius (e.g., 15.0),
            increasing surface_z_thickness (e.g., 15.0), or setting
            surface_floor_value=0.0 and surface_ceiling_value=0.0
        surface_radius : float, default=10.0
            XY-plane radius around ion to show surface atoms (Å)
            Increase to 12-15Å if not enough surface atoms are visible
        surface_z_thickness : float, default=10.0
            Z-direction thickness to limit surface selection (Å)
            Increase to 15-20Å if surface atoms are being cut off
        surface_floor_value : float, default=1.0
            Floor filtering below Si plane (Å)
            Set to 0.0 to disable floor filtering if it's too restrictive
        surface_ceiling_value : float, default=0.0
            Ceiling filtering above Si plane (Å)
            Set to 0.0 to disable ceiling filtering (default already disabled)
        surface_sel : str
            MDAnalysis selection for surface atoms
        molecule_sel : str, optional
            Selection for full molecule context (e.g., 'resname api')
            If provided, shows ALL atoms in the same residue as molecule
        molecule_radius : float, default=10.0
            [DEPRECATED] Molecule selection now shows entire residue
        show_waters : bool, default=False
            Show water molecules near bridging site
        water_radius : float, default=5.0
            Radius for water display (Å)
        bridge_line_color : str, default='gold'
            Color for bridging line (Clay-Ion-Molecule)
        bridge_linewidth : float, default=4.0
            Width of bridging line
        atom_scale_factor : float, default=100
            Scaling factor for atom sizes based on VdW radii
        molecule_alpha : float, default=0.7
            Transparency for molecule atoms (0.0=invisible, 1.0=opaque)
        surface_alpha : float, default=0.6
            Transparency for clay surface atoms (0.0=invisible, 1.0=opaque)
        water_alpha : float, default=0.4
            Transparency for water molecules (0.0=invisible, 1.0=opaque)
        carbon_color : str, default='#202020'
            Color for carbon atoms. Options: '#202020' (very dark gray), 'black',
            '#505050' (medium gray), 'gray', or any matplotlib color
        si_color : str, default='#F0C8A0'
            Color for silicon atoms. Default is light tan (#F0C8A0, standard CPK color).
            Options: '#F0C8A0' (tan), 'orange', 'gray', or any matplotlib color
        show_surface_bonds : bool, default=False
            Draw bonds between surface atoms to reveal Si rings and ditrigonal cavities.
            Uses distance-based Si-Si connectivity (4.5 Å threshold)
        surface_bond_color : str, default='orange'
            Color for surface Si-Si bonds
        surface_bond_linewidth : float, default=1.0
            Line width for surface Si-Si bonds
        boundary_linewidth : float, default=0.5
            Width for atom edge lines
        view_elevation : float, default=25
            Viewing angle elevation (degrees)
        view_azimuth : float, default=45
            Viewing angle azimuth (degrees)
        figsize_per_panel : float, default=5
            Width of each panel in inches
        dpi : int, default=300
            Resolution for saved figure
        save_path : str, optional
            Path to save combined figure
        save_combined_figure : bool, default=True
            Whether to save combined figure with all panels
        save_individual_figures : bool, default=True
            Whether to save each panel as individual figure
        individual_figsize : tuple, default=(8, 6)
            Figure size for individual panels (width, height)
        title_fontsize : int, default=14
            Font size for panel titles
        title_fontweight : str, default='bold'
            Font weight for titles
        label_fontsize : int, default=16
            Font size for axis labels
        label_fontweight : str, default='bold'
            Font weight for axis labels
        tick_fontsize : int, default=14
            Font size for tick labels
        bridge_linestyle : str, default='dashed'
            Line style for bridge connection
        bond_style : str, default='solid'
            Line style for molecular bonds
        atom_edge_style : str, default='solid'
            Line style for atom edges
        surface_atom_style : str, default='scatter'
            Style for surface atoms: 'scatter' or 'surface'
        show_angle_arc : bool, default=True
            Draw an arc at the ion vertex showing the bridge angle visually
        angle_arc_radius : float, default=3.0
            Radius of the angle arc in Angstroms
        angle_arc_color : str, default='cyan'
            Color of the angle arc
        angle_arc_linewidth : float, default=2.0
            Line width of the angle arc
        show_title : bool, default=True
            Show panel titles
        axis_info : str, default='detailed'
            Axis display mode: 'detailed', 'simple', 'minimal', 'off', 'None'
            - 'None' or 'off': Completely hide all axis elements (cleanest)
            - 'minimal': Hide tick labels but keep axes
            - 'simple': Show axes with simple labels (X, Y, Z)
            - 'detailed': Show axes with units (X (Å), Y (Å), Z (Å))
        use_saved_data : bool, default=False
            If True, use pre-calculated bridging events from generate_bridging_report()
            results instead of re-searching the trajectory. This is much faster and
            ensures angles match the saved analysis exactly. The angle_min/max and
            distance filters are still applied to the saved data.
            If False (default), re-calculate all bridging events by searching trajectory.
        start_search_frame : int, default=1
            Starting frame for searching bridging configurations (only used when use_saved_data=False)
        skip_when_searching : int, default=1
            Frame skip when searching for configurations (only used when use_saved_data=False)
            
        Returns
        -------
        tuple
            (fig, axes) - Figure and array of Axes3D objects, or (None, None) if insufficient data
        
        Notes
        -----
        **Angle Interpretation:**
        
        The "Bridge Angle" is measured AT THE ION vertex between two vectors:
        - Vector 1: From Ion → Clay atom
        - Vector 2: From Ion → Molecule atom
        
        Example:
            Clay────Ion────Molecule   (Bridge Angle = 150°, deviation = 30°)
                     ^^^
                  Angle measured here
        
        - **180°** = Perfectly linear bridge (ideal)
        - **150°** = Nearly linear (30° deviation from perfect)
        - **120°** = Moderately bent (60° deviation)
        - **90°** = Right angle (highly bent)
        
        The "deviation angle" (180° - Bridge Angle) shown in titles represents 
        how far the configuration is from perfect linearity. Smaller deviations 
        indicate more linear, stable bridging configurations.
        
        **Data Source Options:**
        
        - **use_saved_data=False (default)**: Re-searches trajectory and recalculates 
          all angles/distances fresh. Slower but allows exploring different frame 
          ranges or applying stricter filters than original analysis.
          
        - **use_saved_data=True**: Uses pre-calculated bridging events from 
          generate_bridging_report(). Much faster and guarantees angles exactly 
          match the saved analysis. Filters (angle_min/max, distance_max) are 
          still applied to the saved data. Requires 'bridging_events' key in 
          bridging_results.
            
        Examples
        --------
        >>> # Basic usage after running generate_bridging_report()
        >>> report = analyzer.generate_bridging_report(
        ...     clay_sel='name Ob',
        ...     ion_sel='resname NA',
        ...     molecule_sel='resname api and (name O1 or name O3)',
        ...     cutoff_clay_ion=3.2,
        ...     cutoff_ion_molecule=3.2,
        ...     angle_threshold=130
        ... )
        >>> 
        >>> fig, axes = plotter.plot_bridging_geometry_3d(
        ...     bridging_results=analyzer.bridging_data,
        ...     cluster_id=0,
        ...     max_bridges=4,
        ...     save_path='bridging_3d.png'
        ... )
        
        >>> # Filter by angle and distance criteria
        >>> fig, axes = plotter.plot_bridging_geometry_3d(
        ...     bridging_results=analyzer.bridging_data,
        ...     cluster_id=0,
        ...     max_bridges=4,
        ...     angle_min=120.0,      # Only show angles >= 120°
        ...     angle_max=160.0,       # Only show angles <= 160°
        ...     distance_clay_max=3.5, # Max Clay-Ion distance
        ...     distance_mol_max=3.5,  # Max Ion-Mol distance
        ...     sort_by='angle',       # Sort by bridging angle
        ...     save_path='bridging_filtered.png',
        ...     dpi=600
        ... )
        
        >>> # Show full molecule and wide surface view
        >>> fig, axes = plotter.plot_bridging_geometry_3d(
        ...     bridging_results=analyzer.bridging_data,
        ...     cluster_id=0,
        ...     molecule_sel='resname api',  # Show full CIP molecule
        ...     surface_radius=12.0,          # Wide surface view
        ...     view_elevation=30,            # Adjust viewing angle
        ...     view_azimuth=-60,
        ...     bridge_line_color='black',
        ...     bridge_linewidth=4.0,
        ...     atom_scale_factor=100,
        ...     save_individual_figures=True,
        ...     save_path='bridging_context.png'
        ... )
        
        >>> # Troubleshooting: If surface atoms not showing consistently
        >>> fig, axes = plotter.plot_bridging_geometry_3d(
        ...     bridging_results=analyzer.bridging_data,
        ...     cluster_id=0,
        ...     surface_radius=15.0,           # Wider radius
        ...     surface_z_thickness=15.0,      # More lenient Z-filter
        ...     surface_floor_value=0.0,       # Disable floor filter
        ...     surface_ceiling_value=0.0,     # Disable ceiling filter
        ...     save_path='bridging_more_surface.png'
        ... )
        
        >>> # Control transparency and show surface connectivity
        >>> fig, axes = plotter.plot_bridging_geometry_3d(
        ...     bridging_results=analyzer.bridging_data,
        ...     cluster_id=0,
        ...     carbon_color='#202020',        # Very dark gray carbon (almost black)
        ...     molecule_alpha=0.8,            # More opaque CIP atoms
        ...     surface_alpha=0.5,             # More transparent surface
        ...     water_alpha=0.3,               # Very transparent waters
        ...     show_surface_bonds=True,       # Show Si rings/ditrigonal cavities
        ...     show_surface_atoms=True,
        ...     surface_radius=12.0,
        ...     save_path='bridging_with_connectivity.png'
        ... )
        
        >>> # Customize angle arc visualization
        >>> fig, axes = plotter.plot_bridging_geometry_3d(
        ...     bridging_results=analyzer.bridging_data,
        ...     cluster_id=0,
        ...     show_angle_arc=True,           # Show arc at ion vertex
        ...     angle_arc_radius=4.0,          # Larger arc radius
        ...     angle_arc_color='magenta',     # Different arc color
        ...     angle_arc_linewidth=3.0,       # Thicker arc line
        ...     bridge_line_color='gold',
        ...     save_path='bridging_with_arc.png'
        ... )
        
        >>> # Customize surface bonds (Si-Si connectivity)
        >>> fig, axes = plotter.plot_bridging_geometry_3d(
        ...     bridging_results=analyzer.bridging_data,
        ...     cluster_id=0,
        ...     show_surface_bonds=True,
        ...     surface_bond_color='darkblue',  # Custom bond color
        ...     surface_bond_linewidth=1.5,     # Thicker bonds
        ...     show_surface_atoms=True,
        ...     surface_radius=12.0,
        ...     save_path='bridging_custom_bonds.png'
        ... )
        
        >>> # Use pre-calculated data for faster visualization
        >>> fig, axes = plotter.plot_bridging_geometry_3d(
        ...     bridging_results=analyzer.bridging_data,
        ...     cluster_id=0,
        ...     use_saved_data=True,            # Use saved angles (faster, exact match)
        ...     angle_min=140.0,                # Still filter by angle
        ...     max_bridges=6,
        ...     save_path='bridging_from_saved.png'
        ... )
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        # Get bridging data for this cluster
        if not hasattr(self.analyzer, 'bridging_data'):
            raise ValueError("No bridging data found. Run generate_bridging_report() first.")
        
        # Get first key from bridging_results
        bridge_key = list(bridging_results.keys())[0]
        cluster_data = bridging_results[bridge_key].get(cluster_id)
        
        if cluster_data is None:
            print(f"No bridging data for cluster {cluster_id}")
            return None, None
        
        # Extract selections and parameters
        clay_sel = cluster_data['clay_sel']
        ion_sel = cluster_data['ion_sel']
        mol_sel = cluster_data['molecule_sel']
        cutoff_clay = cluster_data.get('cutoff_clay_ion', 3.5)
        cutoff_mol = cluster_data.get('cutoff_ion_molecule', 3.5)
        angle_threshold = cluster_data.get('angle_threshold', 130.0)
        
        # Get trajectory
        u = self.analyzer.trajectory_data[cluster_id]['universe']
        frames = self.analyzer.trajectory_data[cluster_id]['frames']
        n_frames = len(frames)
        
        # DEBUG: Check if universes are actually different between clusters
        print(f"\n{'='*70}")
        print(f"DEBUG INFO FOR CLUSTER {cluster_id}")
        print(f"{'='*70}")
        print(f"Universe object ID: {id(u)}")
        print(f"Universe: {u}")
        print(f"Total trajectory frames: {len(u.trajectory)}")
        print(f"Cluster-specific frames: {n_frames}")
        print(f"Frame indices (first 10): {frames[:10]}")
        print(f"{'='*70}")
        
        print(f"\n{'='*70}")
        print(f"Searching for bridging configurations in Cluster {cluster_id}")
        print(f"{'='*70}")
        print(f"Universe: {u}")
        print(f"Trajectory frames: {n_frames}")
        print(f"Clay selection: {clay_sel}")
        print(f"Ion selection: {ion_sel}")
        print(f"Molecule selection: {mol_sel}")
        print(f"Angle threshold: {angle_threshold}° (deviation from linear: {180-angle_threshold}°)")
        print(f"Distance cutoffs: Clay-Ion={cutoff_clay}Å, Ion-Mol={cutoff_mol}Å")
        if angle_min is not None:
            dev_min = 180 - angle_max if angle_max else 0
            dev_max = 180 - angle_min if angle_min else 180
            print(f"Angle filter: {angle_min}° - {angle_max or 180}° (deviation: {dev_min}° - {dev_max}°)")
        print(f"{'='*70}\n")
        
        # Search for valid bridging frames
        valid_frames = []
        
        if use_saved_data:
            # Try to use pre-calculated bridging data from generate_bridging_report()
            print("Using pre-calculated bridging data from saved results...")
            
            # Check if bridging events data exists in cluster_data
            if 'bridging_events' in cluster_data:
                saved_events = cluster_data['bridging_events']
                
                for event in saved_events:
                    # Extract event data (structure may vary)
                    frame_idx = event.get('frame', None)
                    ion_idx = event.get('ion_idx', None)
                    clay_idx = event.get('clay_idx', None)
                    mol_idx = event.get('mol_idx', None)
                    angle = event.get('angle', None)
                    clay_dist = event.get('clay_dist', None)
                    mol_dist = event.get('mol_dist', None)
                    
                    if None in [frame_idx, ion_idx, clay_idx, mol_idx, angle]:
                        continue
                    
                    # Apply filters
                    if angle_min is not None and angle < angle_min:
                        continue
                    if angle_max is not None and angle > angle_max:
                        continue
                    if distance_clay_max is not None and clay_dist is not None and clay_dist > distance_clay_max:
                        continue
                    if distance_mol_max is not None and mol_dist is not None and mol_dist > distance_mol_max:
                        continue
                    
                    # Load frame to get positions
                    u.trajectory[frame_idx]
                    ion_atom = u.atoms[ion_idx]
                    clay_atom = u.atoms[clay_idx]
                    mol_atom = u.atoms[mol_idx]
                    
                    valid_frames.append({
                        'frame': frame_idx,
                        'ion_idx': ion_idx,
                        'clay_idx': clay_idx,
                        'mol_idx': mol_idx,
                        'angle': angle,
                        'clay_dist': clay_dist if clay_dist is not None else np.linalg.norm(clay_atom.position - ion_atom.position),
                        'mol_dist': mol_dist if mol_dist is not None else np.linalg.norm(mol_atom.position - ion_atom.position),
                        'ion_pos': ion_atom.position.copy(),
                        'clay_pos': clay_atom.position.copy(),
                        'mol_pos': mol_atom.position.copy()
                    })
                
                if len(valid_frames) > 0:
                    print(f"✓ Loaded {len(valid_frames)} pre-calculated bridging events")
                else:
                    print("WARNING: No valid events in saved data after filtering")
            else:
                print("WARNING: 'bridging_events' not found in saved data, falling back to trajectory search")
                use_saved_data = False  # Fall back to trajectory search
        
        if not use_saved_data:
            # Re-calculate by searching trajectory (current approach)
            print("Searching trajectory for bridging configurations...")
            
            # Convert to 0-based indexing and create search frame list
            start_idx = max(0, start_search_frame - 1)  # Convert 1-based to 0-based
            search_frames = frames[start_idx::skip_when_searching]  # Use cluster-specific frames!
            print(f"Searching {len(search_frames)} frames from cluster (out of {n_frames} total cluster frames)")
            
            for frame_idx in search_frames:
                u.trajectory[frame_idx]
                
                # Get atoms
                clay_atoms = u.select_atoms(clay_sel)
                ion_atoms = u.select_atoms(ion_sel)
                mol_atoms = u.select_atoms(mol_sel)
                
                if len(clay_atoms) == 0 or len(ion_atoms) == 0 or len(mol_atoms) == 0:
                    continue
                
                # Find bridging ions
                for ion in ion_atoms:
                    ion_pos = ion.position
                    
                    # Calculate distances
                    clay_dists = np.linalg.norm(clay_atoms.positions - ion_pos, axis=1)
                    mol_dists = np.linalg.norm(mol_atoms.positions - ion_pos, axis=1)
                    
                    min_clay_dist = np.min(clay_dists)
                    min_mol_dist = np.min(mol_dists)
                    
                    # Check distance criteria
                    if min_clay_dist > cutoff_clay or min_mol_dist > cutoff_mol:
                        continue
                    
                    # Get nearest atoms
                    nearest_clay_idx = np.argmin(clay_dists)
                    nearest_mol_idx = np.argmin(mol_dists)
                    nearest_clay_pos = clay_atoms[nearest_clay_idx].position
                    nearest_mol_pos = mol_atoms[nearest_mol_idx].position
                    
                    # Calculate bridging angle (Clay-Ion-Molecule)
                    vec_clay = nearest_clay_pos - ion_pos
                    vec_mol = nearest_mol_pos - ion_pos
                    
                    cos_angle = np.dot(vec_clay, vec_mol) / (np.linalg.norm(vec_clay) * np.linalg.norm(vec_mol))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))
                    
                    # Apply angle filter
                    if angle_min is not None and angle < angle_min:
                        continue
                    if angle_max is not None and angle > angle_max:
                        continue
                    
                    # Apply distance filters
                    if distance_clay_max is not None and min_clay_dist > distance_clay_max:
                        continue
                    if distance_mol_max is not None and min_mol_dist > distance_mol_max:
                        continue
                    
                    # Valid bridging configuration found
                    valid_frames.append({
                        'frame': frame_idx,
                        'ion_idx': ion.index,
                        'clay_idx': clay_atoms[nearest_clay_idx].index,
                        'mol_idx': mol_atoms[nearest_mol_idx].index,
                        'angle': angle,
                        'clay_dist': min_clay_dist,
                        'mol_dist': min_mol_dist,
                        'ion_pos': ion_pos.copy(),
                        'clay_pos': nearest_clay_pos.copy(),
                        'mol_pos': nearest_mol_pos.copy()
                    })
                    
                    # Debug: Print found frame info
                    if len(valid_frames) <= 10:  # Only print first 10 to avoid spam
                        print(f"  Found bridging frame {frame_idx}: angle={angle:.1f}°, clay_dist={min_clay_dist:.2f}Å, mol_dist={min_mol_dist:.2f}Å")
        
        data_source = "pre-calculated data" if use_saved_data else "trajectory search"
        print(f"Found {len(valid_frames)} valid bridging configurations from {data_source}")
        
        if len(valid_frames) > 0:
            # Show angle statistics
            angles = [f['angle'] for f in valid_frames]
            deviations = [180.0 - a for a in angles]
            print(f"  Bridge angle range: {min(angles):.1f}° - {max(angles):.1f}°")
            print(f"  Deviation from linear: {min(deviations):.1f}° - {max(deviations):.1f}°")
            print(f"  Mean bridge angle: {np.mean(angles):.1f}° (deviation: {np.mean(deviations):.1f}°)")
        
        if len(valid_frames) < min_bridges:
            print(f"ERROR: Found {len(valid_frames)} valid bridging configs, but need at least {min_bridges}")
            return None, None
        
        # Sort frames based on sort_by parameter
        if sort_by == 'angle':
            valid_frames.sort(key=lambda x: x['angle'], reverse=True)
        elif sort_by == 'frequency':
            # Keep original order (more frequent configurations appear first)
            pass
        elif sort_by == 'random':
            np.random.shuffle(valid_frames)
        
        # Select frames to visualize
        frames_to_plot = valid_frames[:max_bridges]
        n_panels = len(frames_to_plot)
        
        print(f"Visualizing {n_panels} bridging configurations")
        print("Selected frames for visualization:")
        for i, frame_info in enumerate(frames_to_plot):
            print(f"  Panel {i+1}: Frame {frame_info['frame']}, Angle={frame_info['angle']:.1f}°")
        print()
        
        # VdW radii (Å)
        vdw_radii = {
            'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
            'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Na': 2.27, 'Mg': 1.73,
            'Si': 2.10, 'Ca': 2.31, 'K': 2.75
        }
        
        # Atom colors (CPK coloring)
        atom_colors = {
            'H': 'white', 'C': carbon_color, 'N': '#3050F8', 'O': '#FF0D0D',
            'F': '#90E050', 'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F',
            'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Si': si_color, 'Ca': '#3DFF00',
            'K': '#8F40D4'
        }
        
        # Create figure
        fig = plt.figure(figsize=(figsize_per_panel * n_panels, figsize_per_panel), facecolor='white')
        axes = []
        
        for panel_idx, frame_info in enumerate(frames_to_plot):
            # Load frame
            u.trajectory[frame_info['frame']]
            
            # Create subplot
            ax = fig.add_subplot(1, n_panels, panel_idx + 1, projection='3d', facecolor='white')
            axes.append(ax)
            
            # Get bridging atoms
            ion_pos = frame_info['ion_pos']
            clay_pos = frame_info['clay_pos']
            mol_pos = frame_info['mol_pos']
            
            # Ion atom (center of visualization)
            ion_atom = u.atoms[frame_info['ion_idx']]
            # Extract element properly (handle multi-character like Na, Mg, Ca)
            if hasattr(ion_atom, 'element') and ion_atom.element:
                ion_element = ion_atom.element
            elif ion_atom.name[:2] in ['Mg', 'Si', 'Ca', 'Na', 'Cl']:
                ion_element = ion_atom.name[:2]
            else:
                ion_element = ion_atom.name[0].upper()
            ion_radius = vdw_radii.get(ion_element, 1.7) * atom_scale_factor
            ion_color = atom_colors.get(ion_element, 'purple')
            
            ax.scatter([ion_pos[0]], [ion_pos[1]], [ion_pos[2]],
                      s=ion_radius, c=ion_color, edgecolors='black',
                      linewidths=boundary_linewidth, alpha=1.0, zorder=100)
            
            # Clay atom
            clay_atom = u.atoms[frame_info['clay_idx']]
            # Extract element properly (handle multi-character like Mg, Si, Ca)
            if hasattr(clay_atom, 'element') and clay_atom.element:
                clay_element = clay_atom.element
            elif clay_atom.name[:2] in ['Mg', 'Si', 'Ca', 'Na', 'Cl']:
                clay_element = clay_atom.name[:2]
            else:
                clay_element = clay_atom.name[0].upper()
            clay_radius = vdw_radii.get(clay_element, 1.7) * atom_scale_factor
            clay_color = atom_colors.get(clay_element, 'red')
            
            ax.scatter([clay_pos[0]], [clay_pos[1]], [clay_pos[2]],
                      s=clay_radius, c=clay_color, edgecolors='black',
                      linewidths=boundary_linewidth, alpha=1.0, zorder=90)
            
            # Molecule atom
            mol_atom = u.atoms[frame_info['mol_idx']]
            # Extract element properly (handle multi-character like Mg, Si, Ca)
            if hasattr(mol_atom, 'element') and mol_atom.element:
                mol_element = mol_atom.element
            elif mol_atom.name[:2] in ['Mg', 'Si', 'Ca', 'Na', 'Cl']:
                mol_element = mol_atom.name[:2]
            else:
                mol_element = mol_atom.name[0].upper()
            mol_radius = vdw_radii.get(mol_element, 1.7) * atom_scale_factor
            mol_color = atom_colors.get(mol_element, 'red')
            
            ax.scatter([mol_pos[0]], [mol_pos[1]], [mol_pos[2]],
                      s=mol_radius, c=mol_color, edgecolors='black',
                      linewidths=boundary_linewidth, alpha=1.0, zorder=90)
            
            # Draw bridging lines
            if bridge_linestyle == 'dashed':
                ls = '--'
            elif bridge_linestyle == 'dotted':
                ls = ':'
            elif bridge_linestyle == 'dashdot':
                ls = '-.'
            else:
                ls = '-'
            
            # Clay-Ion line
            ax.plot([clay_pos[0], ion_pos[0]], 
                   [clay_pos[1], ion_pos[1]],
                   [clay_pos[2], ion_pos[2]],
                   color=bridge_line_color, linewidth=bridge_linewidth,
                   linestyle=ls, alpha=0.8, zorder=80)
            
            # Ion-Molecule line
            ax.plot([ion_pos[0], mol_pos[0]], 
                   [ion_pos[1], mol_pos[1]],
                   [ion_pos[2], mol_pos[2]],
                   color=bridge_line_color, linewidth=bridge_linewidth,
                   linestyle=ls, alpha=0.8, zorder=80)
            
            # Draw angle arc at ion vertex if requested
            if show_angle_arc:
                # Calculate vectors from ion to clay and ion to molecule
                vec_to_clay = clay_pos - ion_pos
                vec_to_mol = mol_pos - ion_pos
                
                # Normalize the vectors
                vec_to_clay_norm = vec_to_clay / np.linalg.norm(vec_to_clay)
                vec_to_mol_norm = vec_to_mol / np.linalg.norm(vec_to_mol)
                
                # Calculate the angle (same as bridge angle)
                angle_rad = np.arccos(np.clip(np.dot(vec_to_clay_norm, vec_to_mol_norm), -1.0, 1.0))
                
                # Create arc points in the plane defined by the two vectors
                n_arc_points = 30
                arc_angles = np.linspace(0, angle_rad, n_arc_points)
                
                # Use Rodrigues' rotation formula to rotate vec_to_clay towards vec_to_mol
                # Rotation axis is perpendicular to both vectors
                rotation_axis = np.cross(vec_to_clay_norm, vec_to_mol_norm)
                rotation_axis_norm = np.linalg.norm(rotation_axis)
                
                if rotation_axis_norm > 1e-6:  # Avoid degenerate case (parallel vectors)
                    rotation_axis = rotation_axis / rotation_axis_norm
                    
                    # Generate arc points
                    arc_points = []
                    for theta in arc_angles:
                        # Rodrigues' rotation formula
                        cos_theta = np.cos(theta)
                        sin_theta = np.sin(theta)
                        rotated_vec = (vec_to_clay_norm * cos_theta +
                                      np.cross(rotation_axis, vec_to_clay_norm) * sin_theta +
                                      rotation_axis * np.dot(rotation_axis, vec_to_clay_norm) * (1 - cos_theta))
                        
                        arc_point = ion_pos + rotated_vec * angle_arc_radius
                        arc_points.append(arc_point)
                    
                    arc_points = np.array(arc_points)
                    
                    # Plot the arc
                    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
                           color=angle_arc_color, linewidth=angle_arc_linewidth,
                           alpha=0.8, zorder=85)
            
            # Show surface atoms
            filtered_surface = None  # Initialize to avoid scope issues in individual figure saving
            if show_surface_atoms:
                surface_atoms = u.select_atoms(surface_sel)
                
                if len(surface_atoms) > 0:
                    # XY distance from ion
                    xy_dists = np.sqrt((surface_atoms.positions[:, 0] - ion_pos[0])**2 +
                                      (surface_atoms.positions[:, 1] - ion_pos[1])**2)
                    
                    # Z distance from ion
                    z_dists = np.abs(surface_atoms.positions[:, 2] - ion_pos[2])
                    
                    # Start with XY radius filter
                    mask = xy_dists <= surface_radius
                    
                    # Apply Z-thickness filter (but be more lenient)
                    mask = mask & (z_dists <= surface_z_thickness)
                    
                    # Apply floor/ceiling filters if Si atoms available (optional refinement)
                    si_atoms = u.select_atoms('name Si')
                    if len(si_atoms) > 0 and (surface_floor_value > 0 or surface_ceiling_value > 0):
                        avg_si_z = np.mean(si_atoms.positions[:, 2])
                        floor_z = avg_si_z - surface_floor_value
                        ceiling_z = avg_si_z + surface_ceiling_value if surface_ceiling_value > 0 else np.inf
                        
                        # Only apply if it doesn't eliminate ALL atoms
                        z_filter = (surface_atoms.positions[:, 2] >= floor_z) & (surface_atoms.positions[:, 2] <= ceiling_z)
                        potential_mask = mask & z_filter
                        
                        # If floor/ceiling filter leaves some atoms, use it; otherwise keep radius-only filter
                        if np.sum(potential_mask) > 0:
                            mask = potential_mask
                    
                    filtered_surface = surface_atoms[mask]
                    
                    # If still no atoms, fall back to just XY radius (no Z filtering)
                    if len(filtered_surface) == 0:
                        mask = xy_dists <= surface_radius
                        filtered_surface = surface_atoms[mask]
                        print(f"    Panel {panel_idx+1} Warning: Z-filters too restrictive, using XY-only filter → {len(filtered_surface)} atoms")
                    
                    # Plot surface atoms
                    for atom in filtered_surface:
                        # Extract element properly (handle multi-character like Mg, Si, Ca)
                        if hasattr(atom, 'element') and atom.element:
                            element = atom.element
                        elif atom.name[:2] in ['Mg', 'Si', 'Ca', 'Na', 'Cl']:  # Common 2-char elements
                            element = atom.name[:2]
                        else:
                            element = atom.name[0].upper()
                        radius = vdw_radii.get(element, 1.7) * atom_scale_factor * 0.8
                        color = atom_colors.get(element, 'gray')
                        
                        ax.scatter([atom.position[0]], [atom.position[1]], [atom.position[2]],
                                  s=radius, c=color, edgecolors='black',
                                  linewidths=boundary_linewidth*0.5, alpha=surface_alpha, zorder=50)
                    
                    # Draw bonds between surface atoms if requested
                    if show_surface_bonds:
                        # Use distance-based approach (like H-bond method) instead of topology bonds
                        # Collect Si atom positions for connectivity
                        si_atoms_in_surface = []
                        for atom in filtered_surface:
                            if 'Si' in atom.name:
                                si_atoms_in_surface.append((atom.index, atom.position))
                        
                        # Draw Si-Si connections within ~4.5 Å (typical Si-Si distance in clay)
                        si_connections = 0
                        for i in range(len(si_atoms_in_surface)):
                            for j in range(i+1, len(si_atoms_in_surface)):
                                idx1, pos1 = si_atoms_in_surface[i]
                                idx2, pos2 = si_atoms_in_surface[j]
                                dist = np.linalg.norm(pos1 - pos2)
                                
                                if dist < 4.5:  # Si-Si connectivity threshold
                                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                                           color=surface_bond_color, linewidth=surface_bond_linewidth, alpha=0.4, linestyle='-', zorder=45)
                                    si_connections += 1
                        
                        if panel_idx == 0 and si_connections > 0:  # Only report once
                            print(f"    Drew {si_connections} Si-Si connections (ditrigonal cavity structure)")
            
            # Show full molecule if requested
            if molecule_sel is not None:
                try:
                    # ✅ ADVANCED MOLECULE RENDERING: Select ENTIRE molecule by residue
                    # Get the residue of the bridging molecule atom
                    mol_residue = mol_atom.residue
                    mol_resid = mol_atom.resid
                    mol_resname = mol_atom.resname
                    
                    # Select all atoms in the same residue
                    mol_atoms = u.select_atoms(f'{molecule_sel} and resid {mol_resid}')
                    
                    # Exclude the bridging molecule atom itself to avoid duplication
                    mol_atoms = mol_atoms - mol_atom
                    print(f"    Molecule atoms: {len(mol_atoms)} (residue {mol_resname} {mol_resid})")
                    
                    # ✅ GET BOX DIMENSIONS for PBC unwrapping
                    box = u.dimensions  # [a, b, c, alpha, beta, gamma]
                    box_lengths = box[:3]  # a, b, c
                    
                    # ✅ UNWRAP POSITIONS to prevent broken molecules across PBC
                    def unwrap_position(pos, ref_pos, box_lengths):
                        """Unwrap position relative to reference using minimum image convention"""
                        delta = pos - ref_pos
                        delta = delta - box_lengths * np.round(delta / box_lengths)
                        return ref_pos + delta
                    
                    # Use bridging molecule atom as reference (stays in original position)
                    mol_atom_pos_unwrapped = mol_atom.position.copy()
                    
                    # Store unwrapped positions for bond drawing
                    mol_atoms_pos = {}
                    
                    for atom in mol_atoms:
                        # Unwrap molecule atom relative to bridging molecule atom
                        mol_pos = unwrap_position(atom.position, mol_atom_pos_unwrapped, box_lengths)
                        mol_atoms_pos[atom.index] = mol_pos
                        
                        # Extract element properly (handle multi-character like Mg, Si, Ca)
                        if hasattr(atom, 'element') and atom.element:
                            element = atom.element
                        elif atom.name[:2] in ['Mg', 'Si', 'Ca', 'Na', 'Cl']:
                            element = atom.name[:2]
                        else:
                            element = atom.name[0].upper()
                        radius = vdw_radii.get(element, 1.7) * atom_scale_factor
                        color = atom_colors.get(element, 'gray')
                        
                        ax.scatter([mol_pos[0]], [mol_pos[1]], [mol_pos[2]],
                                  s=radius, c=color, edgecolors='black',
                                  linewidths=boundary_linewidth, alpha=molecule_alpha, zorder=70)
                    
                    # ✅ DRAW MOLECULAR BONDS (using MDAnalysis topology + distance-based fallback)
                    # Add the bridging molecule atom to position dict (use unwrapped position)
                    mol_atoms_pos[mol_atom.index] = mol_atom_pos_unwrapped
                    
                    # Get all atoms including bridging atom for bond drawing
                    all_mol_atoms = mol_atoms + mol_atom
                    
                    # STEP 1: Draw bonds from MDAnalysis topology
                    drawn_bonds = set()
                    topology_bonds = 0
                    
                    for atom in all_mol_atoms:
                        if hasattr(atom, 'bonds') and atom.bonds:
                            for bond in atom.bonds:
                                # Get bonded atom indices
                                idx1, idx2 = bond.atoms[0].index, bond.atoms[1].index
                                bond_key = tuple(sorted([idx1, idx2]))
                                
                                # Skip if already drawn
                                if bond_key in drawn_bonds:
                                    continue
                                
                                # Check if both atoms are in our molecule selection
                                if idx1 in mol_atoms_pos and idx2 in mol_atoms_pos:
                                    pos1 = mol_atoms_pos[idx1]
                                    pos2 = mol_atoms_pos[idx2]
                                    
                                    # Draw bond as line with configurable style
                                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                                           color='black', linewidth=1.5, linestyle=bond_style, alpha=molecule_alpha, zorder=60)
                                    drawn_bonds.add(bond_key)
                                    topology_bonds += 1
                    
                    # STEP 2: Distance-based bond detection as fallback
                    # This catches missing bonds like cyclopropyl groups, aromatic rings, etc.
                    distance_bonds = self.detect_bonds_by_distance(all_mol_atoms, mol_atoms_pos)
                    fallback_bonds = 0
                    
                    # Draw bonds that were detected by distance but missing from topology
                    for bond_key in distance_bonds:
                        if bond_key not in drawn_bonds:
                            idx1, idx2 = bond_key
                            if idx1 in mol_atoms_pos and idx2 in mol_atoms_pos:
                                pos1 = mol_atoms_pos[idx1]
                                pos2 = mol_atoms_pos[idx2]
                                
                                # Draw fallback bond with slightly different style to distinguish
                                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                                       color='darkgray', linewidth=1.2, linestyle=bond_style, alpha=molecule_alpha, zorder=60)
                                drawn_bonds.add(bond_key)
                                fallback_bonds += 1
                    
                    total_bonds = len(drawn_bonds)
                    if fallback_bonds > 0:
                        print(f"    Molecular bonds: {total_bonds} total ({topology_bonds} topology + {fallback_bonds} distance-detected)")
                        print(f"    ✓ Distance-based fallback caught {fallback_bonds} missing bonds (likely cyclopropyl/aromatic)")
                    else:
                        print(f"    Molecular bonds: {total_bonds} (all from topology)")
                        
                except Exception as e:
                    print(f"    ⚠️  Could not load molecule: {e}")
            
            # Show waters if requested
            if show_waters:
                water_atoms = u.select_atoms('resname SOL or resname WAT or resname TIP3')
                if len(water_atoms) > 0:
                    water_dists = np.linalg.norm(water_atoms.positions - ion_pos, axis=1)
                    nearby_waters = water_atoms[water_dists <= water_radius]
                    
                    for atom in nearby_waters:
                        # Extract element properly (handle multi-character like Mg, Si, Ca)
                        if hasattr(atom, 'element') and atom.element:
                            element = atom.element
                        elif atom.name[:2] in ['Mg', 'Si', 'Ca', 'Na', 'Cl']:
                            element = atom.name[:2]
                        else:
                            element = atom.name[0].upper()
                        radius = vdw_radii.get(element, 1.7) * atom_scale_factor * 0.6
                        color = atom_colors.get(element, 'cyan')
                        
                        ax.scatter([atom.position[0]], [atom.position[1]], [atom.position[2]],
                                  s=radius, c=color, edgecolors='black',
                                  linewidths=boundary_linewidth*0.3, alpha=water_alpha, zorder=40)
            
            # Set viewing angle
            ax.view_init(elev=view_elevation, azim=view_azimuth)
            
            # Axis formatting
            if axis_info == 'None' or axis_info == 'off':
                # Completely remove all axis elements (cleanest presentation)
                ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.grid(False)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.zaxis.set_visible(False)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('none')
                ax.yaxis.pane.set_edgecolor('none')
                ax.zaxis.pane.set_edgecolor('none')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_zlabel('')
            elif axis_info == 'minimal':
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
            elif axis_info == 'simple':
                ax.set_xlabel('X', fontsize=label_fontsize, fontweight=label_fontweight)
                ax.set_ylabel('Y', fontsize=label_fontsize, fontweight=label_fontweight)
                ax.set_zlabel('Z', fontsize=label_fontsize, fontweight=label_fontweight)
                ax.tick_params(labelsize=tick_fontsize)
            else:  # 'detailed'
                ax.set_xlabel('X (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
                ax.set_ylabel('Y (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
                ax.set_zlabel('Z (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
                ax.tick_params(labelsize=tick_fontsize)
            
            # Title
            if show_title:
                deviation_angle = 180.0 - frame_info['angle']  # Cone angle (deviation from linear)
                title_str = f"Frame {frame_info['frame']}\n"
                title_str += f"Bridge Angle: {frame_info['angle']:.1f}° (deviation: {deviation_angle:.1f}°)\n"
                title_str += f"d(Clay): {frame_info['clay_dist']:.2f}Å, "
                title_str += f"d(Mol): {frame_info['mol_dist']:.2f}Å"
                ax.set_title(title_str, fontsize=title_fontsize, fontweight=title_fontweight)
            
            # Clean background - no grid, no pane colors
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            
            # Save individual figure if requested
            if save_individual_figures and save_path:
                fig_ind = plt.figure(figsize=individual_figsize, facecolor='white')
                ax_ind = fig_ind.add_subplot(111, projection='3d', facecolor='white')
                
                # Copy FULL visualization to individual figure (all elements from main panel)
                
                # 1. Plot surface atoms if requested
                if show_surface_atoms and filtered_surface is not None and len(filtered_surface) > 0:
                    for atom in filtered_surface:
                        # Extract element properly (handle multi-character like Mg, Si, Ca)
                        if hasattr(atom, 'element') and atom.element:
                            element = atom.element
                        elif atom.name[:2] in ['Mg', 'Si', 'Ca', 'Na', 'Cl']:  # Common 2-char elements
                            element = atom.name[:2]
                        else:
                            element = atom.name[0].upper()
                        radius = vdw_radii.get(element, 1.7) * atom_scale_factor * 0.8
                        color = atom_colors.get(element, 'gray')
                        
                        ax_ind.scatter([atom.position[0]], [atom.position[1]], [atom.position[2]],
                                      s=radius, c=color, edgecolors='black',
                                      linewidths=boundary_linewidth*0.5, alpha=surface_alpha, zorder=50)
                    
                    # Draw surface bonds if requested
                    if show_surface_bonds:
                        # Use distance-based approach (like H-bond method) instead of topology bonds
                        # Collect Si atom positions for connectivity
                        si_atoms_in_surface = []
                        for atom in filtered_surface:
                            if 'Si' in atom.name:
                                si_atoms_in_surface.append((atom.index, atom.position))
                        
                        # Draw Si-Si connections within ~4.5 Å (typical Si-Si distance in clay)
                        for i in range(len(si_atoms_in_surface)):
                            for j in range(i+1, len(si_atoms_in_surface)):
                                idx1, pos1 = si_atoms_in_surface[i]
                                idx2, pos2 = si_atoms_in_surface[j]
                                dist = np.linalg.norm(pos1 - pos2)
                                
                                if dist < 4.5:  # Si-Si connectivity threshold
                                    ax_ind.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                                               color=surface_bond_color, linewidth=surface_bond_linewidth, alpha=0.4, linestyle='-', zorder=45)
                
                # 2. Plot full molecule if requested
                if molecule_sel is not None:
                    try:
                        # ✅ ADVANCED MOLECULE RENDERING: Select ENTIRE molecule by residue (same as main panel)
                        mol_residue = mol_atom.residue
                        mol_resid = mol_atom.resid
                        mol_resname = mol_atom.resname
                        
                        # Select all atoms in the same residue
                        mol_atoms = u.select_atoms(f'{molecule_sel} and resid {mol_resid}')
                        mol_atoms = mol_atoms - mol_atom  # Exclude bridging atom to avoid duplication
                        
                        # ✅ GET BOX DIMENSIONS for PBC unwrapping (individual figure)
                        box = u.dimensions  # [a, b, c, alpha, beta, gamma]
                        box_lengths = box[:3]  # a, b, c
                        
                        # ✅ UNWRAP POSITIONS to prevent broken molecules across PBC
                        def unwrap_position(pos, ref_pos, box_lengths):
                            """Unwrap position relative to reference using minimum image convention"""
                            delta = pos - ref_pos
                            delta = delta - box_lengths * np.round(delta / box_lengths)
                            return ref_pos + delta
                        
                        # Use bridging molecule atom as reference (stays in original position)
                        mol_atom_pos_unwrapped = mol_atom.position.copy()
                        
                        # Store unwrapped positions for bond drawing
                        mol_atoms_pos = {}
                        
                        for atom in mol_atoms:
                            # Unwrap molecule atom relative to bridging molecule atom
                            mol_pos = unwrap_position(atom.position, mol_atom_pos_unwrapped, box_lengths)
                            mol_atoms_pos[atom.index] = mol_pos
                            
                            # Extract element properly (handle multi-character like Mg, Si, Ca)
                            if hasattr(atom, 'element') and atom.element:
                                element = atom.element
                            elif atom.name[:2] in ['Mg', 'Si', 'Ca', 'Na', 'Cl']:
                                element = atom.name[:2]
                            else:
                                element = atom.name[0].upper()
                            radius = vdw_radii.get(element, 1.7) * atom_scale_factor
                            color = atom_colors.get(element, 'gray')
                            
                            ax_ind.scatter([mol_pos[0]], [mol_pos[1]], [mol_pos[2]],
                                          s=radius, c=color, edgecolors='black',
                                          linewidths=boundary_linewidth, alpha=molecule_alpha, zorder=70)
                        
                        # ✅ DRAW MOLECULAR BONDS (using MDAnalysis topology + distance-based fallback)
                        # Add the bridging molecule atom to position dict (use unwrapped position)
                        mol_atoms_pos[mol_atom.index] = mol_atom_pos_unwrapped
                        
                        # Get all atoms including bridging atom for bond drawing
                        all_mol_atoms = mol_atoms + mol_atom
                        
                        # STEP 1: Draw bonds from MDAnalysis topology
                        drawn_bonds = set()
                        topology_bonds = 0
                        
                        for atom in all_mol_atoms:
                            if hasattr(atom, 'bonds') and atom.bonds:
                                for bond in atom.bonds:
                                    # Get bonded atom indices
                                    idx1, idx2 = bond.atoms[0].index, bond.atoms[1].index
                                    bond_key = tuple(sorted([idx1, idx2]))
                                    
                                    # Skip if already drawn
                                    if bond_key in drawn_bonds:
                                        continue
                                    
                                    # Check if both atoms are in our molecule selection
                                    if idx1 in mol_atoms_pos and idx2 in mol_atoms_pos:
                                        pos1 = mol_atoms_pos[idx1]
                                        pos2 = mol_atoms_pos[idx2]
                                        
                                        # Draw bond as line with configurable style
                                        ax_ind.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                                                   color='black', linewidth=1.5, linestyle=bond_style, alpha=molecule_alpha, zorder=60)
                                        drawn_bonds.add(bond_key)
                                        topology_bonds += 1
                        
                        # STEP 2: Distance-based bond detection as fallback
                        # This catches missing bonds like cyclopropyl groups, aromatic rings, etc.
                        distance_bonds = self.detect_bonds_by_distance(all_mol_atoms, mol_atoms_pos)
                        fallback_bonds = 0
                        
                        # Draw bonds that were detected by distance but missing from topology
                        for bond_key in distance_bonds:
                            if bond_key not in drawn_bonds:
                                idx1, idx2 = bond_key
                                if idx1 in mol_atoms_pos and idx2 in mol_atoms_pos:
                                    pos1 = mol_atoms_pos[idx1]
                                    pos2 = mol_atoms_pos[idx2]
                                    
                                    # Draw fallback bond with slightly different style to distinguish
                                    ax_ind.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                                              color='darkgray', linewidth=1.2, linestyle=bond_style, alpha=molecule_alpha, zorder=60)
                                    drawn_bonds.add(bond_key)
                                    fallback_bonds += 1
                                    
                    except Exception as e:
                        print(f"    ⚠️  Could not load molecule in individual figure: {e}")
                
                # 3. Plot water molecules if requested
                if show_waters:
                    water_atoms = u.select_atoms('resname SOL or resname WAT or resname TIP3')
                    if len(water_atoms) > 0:
                        water_dists = np.linalg.norm(water_atoms.positions - ion_pos, axis=1)
                        nearby_waters = water_atoms[water_dists <= water_radius]
                        
                        for atom in nearby_waters:
                            # Extract element properly (handle multi-character like Mg, Si, Ca)
                            if hasattr(atom, 'element') and atom.element:
                                element = atom.element
                            elif atom.name[:2] in ['Mg', 'Si', 'Ca', 'Na', 'Cl']:
                                element = atom.name[:2]
                            else:
                                element = atom.name[0].upper()
                            radius = vdw_radii.get(element, 1.7) * atom_scale_factor * 0.6
                            color = atom_colors.get(element, 'cyan')
                            
                            ax_ind.scatter([atom.position[0]], [atom.position[1]], [atom.position[2]],
                                          s=radius, c=color, edgecolors='black',
                                          linewidths=boundary_linewidth*0.3, alpha=water_alpha, zorder=40)
                
                # 4. Plot key bridging atoms (Clay, Ion, Molecule points)
                ax_ind.scatter([ion_pos[0]], [ion_pos[1]], [ion_pos[2]],
                              s=ion_radius, c=ion_color, edgecolors='black',
                              linewidths=boundary_linewidth, alpha=1.0, zorder=100)
                ax_ind.scatter([clay_pos[0]], [clay_pos[1]], [clay_pos[2]],
                              s=clay_radius, c=clay_color, edgecolors='black',
                              linewidths=boundary_linewidth, alpha=1.0, zorder=90)
                ax_ind.scatter([mol_pos[0]], [mol_pos[1]], [mol_pos[2]],
                              s=mol_radius, c=mol_color, edgecolors='black',
                              linewidths=boundary_linewidth, alpha=1.0, zorder=90)
                
                # 5. Draw bridging lines
                ax_ind.plot([clay_pos[0], ion_pos[0]], [clay_pos[1], ion_pos[1]], [clay_pos[2], ion_pos[2]],
                           color=bridge_line_color, linewidth=bridge_linewidth, linestyle=ls, alpha=0.8, zorder=80)
                ax_ind.plot([ion_pos[0], mol_pos[0]], [ion_pos[1], mol_pos[1]], [ion_pos[2], mol_pos[2]],
                           color=bridge_line_color, linewidth=bridge_linewidth, linestyle=ls, alpha=0.8, zorder=80)
                
                # 5b. Draw angle arc at ion vertex if requested
                if show_angle_arc:
                    # Calculate vectors from ion to clay and ion to molecule
                    vec_to_clay = clay_pos - ion_pos
                    vec_to_mol = mol_pos - ion_pos
                    
                    # Normalize the vectors
                    vec_to_clay_norm = vec_to_clay / np.linalg.norm(vec_to_clay)
                    vec_to_mol_norm = vec_to_mol / np.linalg.norm(vec_to_mol)
                    
                    # Calculate the angle
                    angle_rad = np.arccos(np.clip(np.dot(vec_to_clay_norm, vec_to_mol_norm), -1.0, 1.0))
                    
                    # Create arc points
                    n_arc_points = 30
                    arc_angles = np.linspace(0, angle_rad, n_arc_points)
                    
                    # Rotation axis perpendicular to both vectors
                    rotation_axis = np.cross(vec_to_clay_norm, vec_to_mol_norm)
                    rotation_axis_norm = np.linalg.norm(rotation_axis)
                    
                    if rotation_axis_norm > 1e-6:
                        rotation_axis = rotation_axis / rotation_axis_norm
                        
                        # Generate arc points using Rodrigues' rotation
                        arc_points = []
                        for theta in arc_angles:
                            cos_theta = np.cos(theta)
                            sin_theta = np.sin(theta)
                            rotated_vec = (vec_to_clay_norm * cos_theta +
                                          np.cross(rotation_axis, vec_to_clay_norm) * sin_theta +
                                          rotation_axis * np.dot(rotation_axis, vec_to_clay_norm) * (1 - cos_theta))
                            
                            arc_point = ion_pos + rotated_vec * angle_arc_radius
                            arc_points.append(arc_point)
                        
                        arc_points = np.array(arc_points)
                        
                        # Plot the arc
                        ax_ind.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
                                   color=angle_arc_color, linewidth=angle_arc_linewidth,
                                   alpha=0.8, zorder=85)
                
                # 6. Set viewing angle
                ax_ind.view_init(elev=view_elevation, azim=view_azimuth)
                
                # 7. Apply axis formatting
                if axis_info == 'None' or axis_info == 'off':
                    # Completely remove all axis elements (cleanest presentation)
                    ax_ind.axis('off')
                    ax_ind.set_xticks([])
                    ax_ind.set_yticks([])
                    ax_ind.set_zticks([])
                    ax_ind.grid(False)
                    ax_ind.xaxis.set_visible(False)
                    ax_ind.yaxis.set_visible(False)
                    ax_ind.zaxis.set_visible(False)
                    ax_ind.xaxis.pane.fill = False
                    ax_ind.yaxis.pane.fill = False
                    ax_ind.zaxis.pane.fill = False
                    ax_ind.xaxis.pane.set_edgecolor('none')
                    ax_ind.yaxis.pane.set_edgecolor('none')
                    ax_ind.zaxis.pane.set_edgecolor('none')
                    ax_ind.set_xlabel('')
                    ax_ind.set_ylabel('')
                    ax_ind.set_zlabel('')
                elif axis_info == 'minimal':
                    ax_ind.set_xticks([])
                    ax_ind.set_yticks([])
                    ax_ind.set_zticks([])
                elif axis_info == 'simple':
                    ax_ind.set_xlabel('X', fontsize=label_fontsize, fontweight=label_fontweight)
                    ax_ind.set_ylabel('Y', fontsize=label_fontsize, fontweight=label_fontweight)
                    ax_ind.set_zlabel('Z', fontsize=label_fontsize, fontweight=label_fontweight)
                    ax_ind.tick_params(labelsize=tick_fontsize)
                else:  # 'detailed'
                    ax_ind.set_xlabel('X (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
                    ax_ind.set_ylabel('Y (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
                    ax_ind.set_zlabel('Z (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
                    ax_ind.tick_params(labelsize=tick_fontsize)
                
                # 8. Add title
                if show_title:
                    deviation_angle = 180.0 - frame_info['angle']
                    title_str_ind = f"Frame {frame_info['frame']}\n"
                    title_str_ind += f"Bridge Angle: {frame_info['angle']:.1f}° (deviation: {deviation_angle:.1f}°)\n"
                    title_str_ind += f"d(Clay): {frame_info['clay_dist']:.2f}Å, "
                    title_str_ind += f"d(Mol): {frame_info['mol_dist']:.2f}Å"
                    ax_ind.set_title(title_str_ind, fontsize=title_fontsize, fontweight=title_fontweight)
                
                # 9. Clean background styling
                ax_ind.grid(False)
                ax_ind.xaxis.pane.fill = False
                ax_ind.yaxis.pane.fill = False
                ax_ind.zaxis.pane.fill = False
                ax_ind.xaxis.pane.set_edgecolor('none')
                ax_ind.yaxis.pane.set_edgecolor('none')
                ax_ind.zaxis.pane.set_edgecolor('none')
                
                # 10. Save and close
                ind_path = save_path.replace('.png', f'_panel{panel_idx+1}.png')
                fig_ind.savefig(ind_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                print(f"✓ Saved individual panel: {ind_path}")
                plt.close(fig_ind)
        
        plt.tight_layout()
        
        # Save combined figure
        if save_combined_figure and save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved combined figure: {save_path}")
        
        print(f"\n{'='*70}")
        print(f"Bridging geometry visualization complete!")
        print(f"Displayed {n_panels} bridging configurations")
        print(f"{'='*70}\n")
        
        return fig, np.array(axes)
    
    def plot_multiple_rdfs(self, 
                          rdf_data: Union[Dict, List[Dict]],
                          figsize: Tuple[float, float] = (8, 6),
                          xlim: Optional[Tuple[float, float]] = None,
                          ylim: Optional[Tuple[float, float]] = None,
                          cluster_treatment: str = 'separate',
                          show_title: bool = True,
                          title: str = 'Radial Distribution Function',
                          title_fontsize: int = 14,
                          title_fontweight: str = 'bold',
                          xlabel: str = 'r (Å)',
                          ylabel: str = 'g(r)',
                          label_fontsize: int = 12,
                          label_fontweight: str = 'bold',
                          tick_fontsize: int = 10,
                          show_legend: bool = True,
                          legend_fontsize: int = 10,
                          legend_fontweight: str = 'normal',
                          legend_loc: str = 'best',
                          legend_bbox: Optional[Union[Tuple[float, float], Dict[str, Tuple[float, float]]]] = None,
                          legend_ncol: int = 1,
                          legend_frame_alpha: float = 0.9,
                          legend_frameon: bool = True,
                          legend_columnspacing: float = 2.0,
                          colors: Optional[Dict[str, str]] = None,
                          linewidth: float = 2,
                          linestyle: str = '-',
                          alpha: float = 1.0,
                          show_bulk_line: bool = True,
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
                          show_shell_label: bool = True,
                          ion_pairing_gradient: Optional[Union[bool, Dict[str, Union[bool, Dict]]]] = None,
                          ion_pairing_gradient_alpha: float = 0.25,
                          add_inset: bool = False,
                          inset_xlim: Optional[Union[Tuple[float, float], Dict[str, Tuple[float, float]]]] = None,
                          inset_ylim: Optional[Union[Tuple[float, float], Dict[str, Tuple[float, float]]]] = None,
                          inset_bbox: Optional[Union[List[float], Dict[str, List[float]]]] = None,
                          add_inset_colorbar: bool = False,
                          first_peak_min_r: Optional[Union[float, Dict[str, float]]] = None,
                          first_peak_max_r: Optional[Union[float, Dict[str, float]]] = None,
                          first_peak_min_height: Optional[Union[float, Dict[str, float]]] = None,
                          first_peak_prominence: Union[float, Dict[str, float]] = 0.1,
                          first_peak_min_distance: Optional[Union[int, Dict[str, int]]] = None,
                          save_fig: bool = False,
                          filename: str = 'rdf_plot.png',
                          dpi: int = 300,
                          bbox_inches: str = 'tight',
                          save_individual_figures: bool = False,
                          individual_save_dir: Optional[str] = None,
                          individual_figsize: Tuple[float, float] = (8, 6),
                          show_RCN: bool = False,
                          RCN_ylabel: str = 'RCN',
                          RCN_curve_style: str = '--',
                          RCN_curve_weight: float = 2,
                          RCN_label_fontsize: Optional[int] = None,
                          RCN_label_fontweight: Optional[str] = None,
                          RCN_tick_fontsize: Optional[int] = None,
                          RCN_notation: Optional[str] = None,
                          rcn_scale_factor: float = 8.0) -> plt.Figure:
        """
        Create publication-ready RDF plots with comprehensive customization.
        
        This method provides full control over all visual aspects for creating
        publication-quality radial distribution function plots. Handles both 
        single and batch RDF data structures.
        
        Parameters
        ----------
        rdf_data : dict or list of dict
            RDF data in one of these formats:
            1. Batch RDF: {sel1: {sel2: {cluster_id: {'r': ..., 'rdf': ...}}}}
            2. Single RDF: {cluster_id: {'r': ..., 'rdf': ...}}
            3. List of single RDFs: [{'r': ..., 'rdf': ..., 'label': ...}, ...]
        
        Figure dimensions:
        figsize : tuple, default=(8, 6)
            Figure size (width, height) in inches
        xlim, ylim : tuple, optional
            Axis limits (min, max)
        cluster_treatment : str, default='separate'
            How to organize panels:
            - 'separate': One panel per cluster, showing all moieties
                         (compare moieties within same cluster)
            - 'group': One panel per moiety, showing all clusters
                      (compare clusters for same moiety - see peak shifts)
        
        Title styling:
        show_title : bool, default=True
            Whether to display title
        title : str
            Plot title text
        title_fontsize : int, default=14
            Title font size
        title_fontweight : str, default='bold'
            Title font weight ('normal', 'bold', 'light', 'heavy')
        
        Axis labels:
        xlabel, ylabel : str
            Axis label text
        label_fontsize : int, default=12
            Axis label font size
        label_fontweight : str, default='bold'
            Axis label font weight
        
        Tick styling:
        tick_fontsize : int, default=10
            Tick label font size
        
        Legend styling:
        show_legend : bool, default=True
            Whether to display legend
        legend_fontsize : int, default=10
            Legend font size
        legend_fontweight : str, default='normal'
            Legend font weight
        legend_loc : str, default='best'
            Legend location
        legend_ncol : int, default=1
            Number of legend columns
        legend_frame_alpha : float, default=0.9
            Legend frame transparency (0=transparent, 1=opaque)
        legend_frameon : bool, default=True
            Whether to draw legend frame
        
        Line styling:
        colors : dict, optional
            Custom colors for selections or clusters.
            - For cluster_treatment='separate': keys are moiety names (str)
              Example: {'quinolone': 'red', 'piperazine': 'blue'}
            - For cluster_treatment='group': keys are cluster IDs (int)
              Example: {0: 'red', 1: 'blue', 2: 'green'}
            If None, auto-generates colors using tab10 colormap.
        linewidth : float, default=2
            Line width
        linestyle : str, default='-'
            Line style ('-', '--', '-.', ':')
        alpha : float, default=1.0
            Line transparency
        
        Bulk density reference:
        show_bulk_line : bool, default=True
            Show g(r)=1 reference line
        bulk_line_color : str, default='gray'
            Bulk line color
        bulk_line_style : str, default='--'
            Bulk line style
        bulk_line_width : float, default=1
            Bulk line width
        bulk_line_alpha : float, default=0.5
            Bulk line transparency
        
        Grid styling:
        grid : bool, default=True
            Show grid
        grid_alpha : float, default=0.3
            Grid transparency
        grid_linestyle : str, default='--'
            Grid line style
        
        Shell region shading (for functional group RDFs):
        shell_boundaries : list or dict, optional
            Shell boundary radii. Can be:
            - List: [2.88, 5.03, 7.25] - Same boundaries for all moieties
            - Dict: {'quinolone': [3.18, 5.53], 'piperazine': [2.88, 5.03]} - Per-moiety boundaries
            Creates shaded regions: Shell 1: 0 to first radius, Shell 2: first to second radius, etc.
            Uses same blue saturation gradient as plot_rdf_curves_per_layer()
        shell_alpha : float, default=0.15
            Transparency of shell region shading (0=invisible, 1=opaque)
        shell_label_fontsize : int, default=10
            Font size for shell region labels (e.g., "Shell 1", "Shell 2", "Bulk")
        shell_label_style : str, default='complete'
            Label style for shell regions:
            - 'complete': "Shell 1", "Shell 2", "Shell 3", "Bulk"
            - 'short': "S1", "S2", "S3", "Bulk"
        shell_label_ha : str, default='center'
            Horizontal alignment of shell region labels:
            - 'center': Label at midpoint of each shell region (default)
            - 'left': Label at left edge of each shell region
            - 'right': Label at right edge of each shell region
        
        Ion pairing gradient shading (for non-aqueous RDFs):
        ion_pairing_gradient : bool or dict, optional
            Apply smooth ion-pairing gradient background (lightcoral → lightyellow → lightgreen → lightblue).
            Represents CIP → SIP → DSIP → FI progression without discrete boundaries.
            Can be:
            - None: No gradient (default)
            - Bool: True applies to all RDF pairs, False disables
            - Dict with pair keys '{moiety}-{reference}': Enables per-pair control
              Format: {'quinolone-Na': True, 'carboxylic_acid-surface_o': True}
            - Dict with config: {'quinolone-Na': {'apply': True, 'transitions': [2.5, 5.0, 8.0]}}
              'transitions' specifies custom boundary points between color regions
            Gradient is mutually exclusive with shell_boundaries (shells take priority).
        ion_pairing_gradient_alpha : float, default=0.25
            Transparency of gradient shading (0=invisible, 1=opaque).
        
        Inset zoom:
        add_inset : bool, default=False
            Whether to add an inset zoom plot
        inset_xlim : tuple or dict, optional
            X-axis limits for inset zoom. Can be:
            - Tuple: (2.0, 4.0) - Same limits for all moieties
            - Dict: {'quinolone': (2.35, 2.40), 'piperazine': (2.0, 2.1)} - Per-moiety limits
        inset_ylim : tuple or dict, optional
            Y-axis limits for inset zoom. Can be:
            - Tuple: (0, 3.0) - Same limits for all moieties
            - Dict: {'quinolone': (9.7, 10), 'piperazine': (8, 9)} - Per-moiety limits
        inset_bbox : list or dict, optional
            Inset position in data coordinates. Can be:
            - List: [6.0, 10.0, 0.5, 2.0] - Same position for all (x_min, x_max, y_min, y_max)
            - Dict: {'quinolone': [4.0, 8, 3, 8], 'piperazine': [3.5, 7, 2.5, 7]} - Per-moiety positions
        add_inset_colorbar : bool, default=False
            If True, add a colorbar to the right of the inset showing the cluster/condition
            to color mapping. The colorbar shows the full range from the data.
        
        Peak detection (for first peak reporting in GROUP mode):
        first_peak_min_r : float or dict, optional
            Minimum r value to search for first peak (e.g., 2.7 to skip artifacts).
            Can be per-moiety dict: {'quinolone': 2.5, 'piperazine': 2.7}
            If None, searches from r=0. Use to exclude noise/artifacts at low r.
        first_peak_max_r : float or dict, optional
            Maximum r value to search for first peak (e.g., 6.0 to limit to first shell).
            Can be per-moiety dict: {'quinolone': 5.5, 'piperazine': 6.0}
            If None, searches entire r range.
        first_peak_min_height : float or dict, optional
            Minimum g(r) height for peak detection (e.g., 1.5 for peaks 1.5× bulk density).
            Can be per-moiety dict. If None, uses 1.0 (bulk baseline).
        first_peak_prominence : float or dict, default=0.1
            Minimum prominence required for peak detection (filters noise vs real peaks).
            Can be per-moiety dict: {'quinolone': 0.2, 'piperazine': 0.15}
            Higher values = stricter peak detection.
        first_peak_min_distance : int or dict, optional
            Minimum number of data points between peaks. If None, no distance constraint.
            Can be per-moiety dict. Useful for closely-spaced peak separation.
        
        Saving:
        save_fig : bool, default=False
            Whether to save the combined multi-panel figure
        filename : str, default='rdf_plot.png'
            Output filename for combined figure
        dpi : int, default=300
            Resolution for saved figures
        bbox_inches : str, default='tight'
            Bounding box setting for saved figure
        save_individual_figures : bool, default=False
            Whether to save each cluster panel as a separate figure
        individual_save_dir : str, optional
            Directory to save individual cluster figures. 
            If None, uses same directory as filename.
        individual_figsize : tuple, default=(8, 6)
            Figure size (width, height) for individual cluster plots
        
        Running Coordination Number (for 'group' mode only):
        show_RCN : bool, default=False
            Plot running coordination number on secondary y-axis (cluster_treatment='group' only)
        RCN_ylabel : str, default='RCN'
            Label for secondary y-axis
        RCN_curve_style : str, default='--'
            Line style for RCN curves (dashed by default)
        RCN_curve_weight : float, default=2
            Line width for RCN curves
        RCN_label_fontsize : int, optional
            Font size for RCN y-axis label. If None, inherits from label_fontsize
        RCN_label_fontweight : str, optional
            Font weight for RCN y-axis label. If None, inherits from label_fontweight
        RCN_tick_fontsize : int, optional
            Font size for RCN y-axis tick labels. If None, inherits from tick_fontsize
        RCN_notation : str, optional
            Number formatting for RCN axis. Options:
            - None (default): Standard notation (10000, 20000, ...)
            - 'offset': Offset notation (10, 20, 30... with ×10³ at top)
            - 'scaled': Manual scaling (shows 10, 20, 30... and changes label to 'RCN (×10³)')
            Useful when RCN values are large (e.g., 10000-140000)
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        
        Examples
        --------
        Simple batch RDF plot:
        >>> rdf_batch = analyzer.compute_rdf(['quinolone', 'piperazine'], 'water_o')
        >>> plotter.plot_multiple_rdfs(
        ...     rdf_batch,
        ...     colors={'quinolone': 'red', 'piperazine': 'blue'},
        ...     save_fig=True,
        ...     filename='cip_water_rdf.png'
        ... )
        
        Publication-ready plot:
        >>> plotter.plot_multiple_rdfs(
        ...     rdf_water,
        ...     xlim=(0, 10),
        ...     show_title=False,
        ...     title_fontsize=22,
        ...     label_fontsize=22,
        ...     label_fontweight='bold',
        ...     tick_fontsize=18,
        ...     legend_fontsize=18,
        ...     colors={'quinolone': 'red', 'carboxylic_acid': 'black'},
        ...     legend_ncol=2,
        ...     legend_frame_alpha=0.0,
        ...     linewidth=3,
        ...     save_fig=True,
        ...     dpi=600,
        ...     filename='rdf_publication.png'
        ... )
        
        Save each cluster as individual figure:
        >>> plotter.plot_multiple_rdfs(
        ...     rdf_batch,
        ...     xlim=(0, 8),
        ...     colors={'quinolone': '#E74C3C', 'piperazine': '#3498DB'},
        ...     save_individual_figures=True,
        ...     individual_save_dir='./rdf_individual',
        ...     individual_figsize=(8, 6),
        ...     dpi=600,
        ...     filename='rdf_combined.png'
        ... )
        # Saves: rdf_individual/rdf_combined_cluster_0.png
        #        rdf_individual/rdf_combined_cluster_1.png, etc.
        
        Group by moiety to compare clusters:
        >>> plotter.plot_multiple_rdfs(
        ...     rdf_batch,
        ...     cluster_treatment='group',  # One panel per moiety
        ...     xlim=(0, 8),
        ...     colors={0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}  # Cluster colors
        ... )
        # Each panel shows one moiety with all cluster RDFs overlaid
        
        Group mode with running coordination numbers on secondary axis:
        >>> plotter.plot_multiple_rdfs(
        ...     rdf_batch,
        ...     cluster_treatment='group',
        ...     xlim=(0, 8),
        ...     label_fontsize=22,  # Primary axis font size
        ...     colors={0: 'red', 1: 'blue', 2: 'green'},
        ...     show_RCN=True,  # Add RCN on secondary y-axis (starts at 0)
        ...     RCN_ylabel='RCN',  # Default label
        ...     RCN_curve_style='--',  # Dashed lines for RCN
        ...     RCN_curve_weight=2,
        ...     RCN_label_fontsize=18,  # Optional: different size for RCN label
        ...     RCN_tick_fontsize=16  # Optional: different size for RCN ticks
        ...     # If not set, RCN inherits from label_fontsize and tick_fontsize
        ... )
        # Each panel shows g(r) on left y-axis and N(r) on right y-axis  
        # Both axes start at 0. RCN can have custom or inherited font styling
        
        RCN axis with large numbers (offset notation):
        >>> plotter.plot_multiple_rdfs(
        ...     rdf_batch,
        ...     cluster_treatment='group',
        ...     show_RCN=True,
        ...     RCN_notation='offset'  # Shows: 10, 20, 30... with ×10³ at top
        ... )
        # For RCN values like 10000-140000, displays as 10, 20...140 with scientific offset
        
        RCN axis with large numbers (scaled notation):
        >>> plotter.plot_multiple_rdfs(
        ...     rdf_batch,
        ...     cluster_treatment='group',
        ...     show_RCN=True,
        ...     RCN_notation='scaled',  # Shows: 10, 20, 30... with updated label
        ...     RCN_ylabel='RCN'  # Label becomes 'RCN (×10³)' automatically
        ... )
        # Cleaner for publications: divides values by 1000 and updates label
        
        Compare moieties within clusters (default):
        >>> plotter.plot_multiple_rdfs(
        ...     rdf_batch,
        ...     cluster_treatment='separate',  # One panel per cluster
        ...     xlim=(0, 8),
        ...     colors={'quinolone': 'red', 'piperazine': 'blue'}  # Moiety colors
        ... )
        # Each panel shows one cluster with all moieties overlaid
        """
        # Helper function to resolve per-moiety parameters (dict or single value)
        def resolve_param(param, moiety_name):
            """
            Resolve parameter value for a specific moiety.
            If param is dict, look up by moiety name. Otherwise return param as-is.
            """
            if isinstance(param, dict):
                return param.get(moiety_name, None)
            else:
                return param
        
        # Define color generation function for shell shading (same as plot_rdf_curves_per_layer)
        def get_blue_saturation_colors_from_00c5ff(n_shells):
            """Generate blue saturation gradient colors for shell shading"""
            import matplotlib.colors as mcolors
            
            base_rgb = mcolors.hex2color('#00c5ff')
            base_hsv = mcolors.rgb_to_hsv(base_rgb)
            base_hue, base_saturation, base_value = base_hsv[0], base_hsv[1], base_hsv[2]
            
            if n_shells == 1:
                saturations = [base_saturation, 0.2]
            elif n_shells == 2:
                saturations = [base_saturation, 0.6, 0.2]
            elif n_shells == 3:
                saturations = [base_saturation, 0.7, 0.4, 0.2]
            else:
                step = (base_saturation - 0.2) / n_shells
                saturations = [base_saturation - (i * step) for i in range(n_shells)]
                saturations.append(0.2)
            
            colors = []
            for sat in saturations:
                hsv = (base_hue, sat, base_value)
                rgb = mcolors.hsv_to_rgb(hsv)
                colors.append(mcolors.to_hex(rgb))
            
            return colors
        
        # Helper function to create ion pairing gradient colormap
        def create_ion_pairing_gradient_cmap():
            """Create smooth gradient colormap: lightcoral → lightyellow → lightgreen → lightblue"""
            from matplotlib.colors import LinearSegmentedColormap
            colors_list = ['lightcoral', 'lightyellow', 'lightgreen', 'lightblue']
            return LinearSegmentedColormap.from_list('ion_pairing', colors_list, N=256)
        
        # Parse RDF data structure and organize by cluster
        cluster_data = {}  # {cluster_id: {sel_name: {'r': ..., 'rdf': ...}}}
        selection_names = []
        reference_names = []  # Track reference names for pair key construction
        
        # Detect data structure and extract data organized by cluster
        if isinstance(rdf_data, dict):
            first_key = next(iter(rdf_data.keys()))
            first_val = rdf_data[first_key]
            
            # Check if first_val directly contains RDF data
            if isinstance(first_val, dict) and 'r' in first_val:
                # Format: {name: {'r': ..., 'rdf': ...}} - single cluster, assume cluster 0
                cluster_data[0] = {}
                for name, data in rdf_data.items():
                    cluster_data[0][name] = {'r': data['r'], 'rdf': data['rdf'], 
                                            'n_frames': data.get('n_frames', 'N/A'),
                                            'count': data.get('count', None)}
                    selection_names.append(name)
            
            elif isinstance(first_val, dict):
                inner_first_key = next(iter(first_val.keys()))
                inner_first_val = first_val[inner_first_key]
                
                # Check if this is batch RDF format (3 levels: sel1 -> sel2 -> cluster_id)
                if isinstance(inner_first_val, dict):
                    innermost_key = next(iter(inner_first_val.keys()))
                    innermost_val = inner_first_val[innermost_key]
                    
                    if isinstance(innermost_val, dict) and 'r' in innermost_val:
                        # Format: {sel1: {sel2: {cluster_id: {'r': ..., 'rdf': ...}}}}
                        # Reorganize by cluster
                        for sel1_name, sel1_data in rdf_data.items():
                            selection_names.append(sel1_name)
                            for sel2_name, sel2_data in sel1_data.items():
                                if sel2_name not in reference_names:
                                    reference_names.append(sel2_name)
                                for cluster_id, cluster_rdf in sel2_data.items():
                                    if cluster_id not in cluster_data:
                                        cluster_data[cluster_id] = {}
                                    cluster_data[cluster_id][sel1_name] = {
                                        'r': cluster_rdf['r'],
                                        'rdf': cluster_rdf['rdf'],
                                        'n_frames': cluster_rdf.get('n_frames', 'N/A'),
                                        'count': cluster_rdf.get('count', None),
                                        'reference': sel2_name  # Store reference name
                                    }
                    
                    elif 'r' in inner_first_val:
                        # Format: {name: {cluster_id: {'r': ..., 'rdf': ...}}}
                        for name, cluster_dict in rdf_data.items():
                            selection_names.append(name)
                            for cluster_id, cluster_rdf in cluster_dict.items():
                                if cluster_id not in cluster_data:
                                    cluster_data[cluster_id] = {}
                                cluster_data[cluster_id][name] = {
                                    'r': cluster_rdf['r'],
                                    'rdf': cluster_rdf['rdf'],
                                    'n_frames': cluster_rdf.get('n_frames', 'N/A'),
                                    'count': cluster_rdf.get('count', None)
                                }
                    else:
                        raise ValueError("Unrecognized RDF data structure")
                
                elif 'r' in inner_first_val:
                    # Format: {cluster_id: {'r': ..., 'rdf': ...}}
                    for cluster_id, cluster_rdf in rdf_data.items():
                        cluster_data[cluster_id] = {
                            'data': {
                                'r': cluster_rdf['r'],
                                'rdf': cluster_rdf['rdf'],
                                'n_frames': cluster_rdf.get('n_frames', 'N/A'),
                                'count': cluster_rdf.get('count', None)
                            }
                        }
                        selection_names = ['data']
                else:
                    raise ValueError("Unrecognized RDF data structure")
            else:
                raise ValueError("Unrecognized RDF data structure")
        
        elif isinstance(rdf_data, list):
            # List format: [{'r': ..., 'rdf': ..., 'label': ...}, ...]
            cluster_data[0] = {}
            for i, data in enumerate(rdf_data):
                name = data.get('name', data.get('label', f'RDF {i+1}'))
                cluster_data[0][name] = {'r': data['r'], 'rdf': data['rdf'], 
                                        'n_frames': 'N/A', 'count': data.get('count', None)}
                selection_names.append(name)
        else:
            raise ValueError("rdf_data must be dict or list")
        
        # Remove duplicates from selection_names while preserving order
        selection_names = list(dict.fromkeys(selection_names))
        
        # Determine plotting mode and setup accordingly
        cluster_ids = sorted(cluster_data.keys())
        
        if cluster_treatment == 'group':
            # Group mode: one panel per moiety, showing all clusters
            panel_items = selection_names  # Each panel is a moiety
            n_panels = len(selection_names)
            
            # Get colors for clusters
            if colors is None:
                cmap = plt.cm.get_cmap('tab10')
                colors = {}
                for i, cluster_id in enumerate(cluster_ids):
                    colors[cluster_id] = cmap(i % 10)
        else:
            # Separate mode: one panel per cluster, showing all moieties (current behavior)
            panel_items = cluster_ids  # Each panel is a cluster
            n_panels = len(cluster_ids)
            
            # Get colors for moieties
            if colors is None:
                cmap = plt.cm.get_cmap('tab10')
                colors = {}
                for i, name in enumerate(selection_names):
                    colors[name] = cmap(i % 10)
        
        # Create subplots
        n_clusters = len(cluster_data)
        
        # Calculate subplot layout
        if n_panels <= 3:
            nrows, ncols = 1, n_panels
            subplot_figsize = (figsize[0] * ncols, figsize[1])
        else:
            ncols = 3
            nrows = int(np.ceil(n_panels / ncols))
            subplot_figsize = (figsize[0] * ncols, figsize[1] * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=subplot_figsize, squeeze=False)
        axes = axes.flatten()
        
        # Plot based on cluster_treatment mode
        if cluster_treatment == 'group':
            # GROUP MODE: One panel per moiety, showing all clusters
            print(f"DEBUG: In GROUP mode, show_RCN={show_RCN}")
            for idx, sel_name in enumerate(selection_names):
                ax = axes[idx]
                
                print(f"\n{sel_name} RDF statistics:")
                print(f"DEBUG: Processing moiety {sel_name}, show_RCN={show_RCN}")
                
                # Resolve per-moiety parameters for this moiety
                current_shell_boundaries = resolve_param(shell_boundaries, sel_name)
                current_inset_xlim = resolve_param(inset_xlim, sel_name)
                current_inset_ylim = resolve_param(inset_ylim, sel_name)
                current_inset_bbox = resolve_param(inset_bbox, sel_name)
                
                # Resolve legend bbox for this moiety
                current_legend_bbox = resolve_param(legend_bbox, sel_name)
                
                # Resolve peak detection parameters for this moiety
                current_peak_min_r = resolve_param(first_peak_min_r, sel_name)
                current_peak_max_r = resolve_param(first_peak_max_r, sel_name)
                current_peak_min_height = resolve_param(first_peak_min_height, sel_name)
                current_peak_prominence = resolve_param(first_peak_prominence, sel_name)
                if current_peak_prominence is None:
                    current_peak_prominence = 0.1
                current_peak_min_distance = resolve_param(first_peak_min_distance, sel_name)
                
                # Add shell shading if boundaries provided
                shell_colors_map = {}
                bulk_color = None
                bulk_start = None
                
                if current_shell_boundaries is not None and len(current_shell_boundaries) > 0:
                    n_shells = len(current_shell_boundaries)
                    all_colors = get_blue_saturation_colors_from_00c5ff(n_shells)
                    shell_colors = all_colors[:-1]
                    bulk_color = all_colors[-1]
                    
                    # Add shading for each shell
                    prev_r = 0.0
                    for i, r_max in enumerate(current_shell_boundaries):
                        color = shell_colors[i]
                        ax.axvspan(prev_r, r_max, alpha=shell_alpha, color=color, zorder=0)
                        shell_colors_map[(prev_r, r_max)] = color
                        prev_r = r_max
                    
                    # Add bulk region (extends to figure limit, not data range)
                    bulk_start = current_shell_boundaries[-1]
                    bulk_end = xlim[1] if xlim is not None else 20  # Will be adjusted by actual axis limits
                    ax.axvspan(bulk_start, bulk_end, alpha=shell_alpha, color=bulk_color, zorder=0)
                
                # Add ion pairing gradient if configured (only if no shell boundaries)
                # Construct pair key: "{moiety}-{reference}"
                reference_name = None
                for cluster_id in cluster_ids:
                    if cluster_id in cluster_data and sel_name in cluster_data[cluster_id]:
                        reference_name = cluster_data[cluster_id][sel_name].get('reference', None)
                        if reference_name:
                            break
               
                if reference_name and ion_pairing_gradient is not None:
                    pair_key = f"{sel_name}-{reference_name}"
                    gradient_config = resolve_param(ion_pairing_gradient, pair_key)
                    
                    # Apply gradient if configured AND no shell boundaries
                    should_apply_gradient = False
                    custom_transitions = None
                    
                    if gradient_config is True:
                        should_apply_gradient = True
                    elif isinstance(gradient_config, dict):
                        should_apply_gradient = gradient_config.get('apply', False)
                        custom_transitions = gradient_config.get('transitions', None)
                    
                    # Shells take priority over gradient
                    if should_apply_gradient and (current_shell_boundaries is None or len(current_shell_boundaries) == 0):
                        # Get x-axis limits
                        x_min = xlim[0] if xlim is not None else 0
                        x_max = xlim[1] if xlim is not None else 20
                        
                        # Get y-axis limits for gradient extent
                        y_min = ylim[0] if ylim is not None else 0
                        if ylim is not None:
                            y_max = ylim[1]
                        else:
                            # Auto-detect y_max from data for this selection
                            max_rdf = 0
                            for cid in cluster_ids:
                                if cid in cluster_data and sel_name in cluster_data[cid]:
                                    max_rdf = max(max_rdf, np.max(cluster_data[cid][sel_name]['rdf']))
                            y_max = max_rdf * 1.1  # Add 10% padding
                        
                        # Create gradient array (horizontal gradient from left to right)
                        gradient = np.linspace(0, 1, 256).reshape(1, -1)
                        
                        # Create colormap
                        cmap = create_ion_pairing_gradient_cmap()
                        
                        # Apply gradient with imshow
                        extent = [x_min, x_max, y_min, y_max]
                        ax.imshow(gradient, aspect='auto', extent=extent, origin='lower',
                                 cmap=cmap, alpha=ion_pairing_gradient_alpha, zorder=0,
                                 interpolation='bilinear')
                
                # Plot this moiety for all clusters
                for cluster_id in cluster_ids:
                    if cluster_id in cluster_data and sel_name in cluster_data[cluster_id]:
                        sel_data = cluster_data[cluster_id][sel_name]
                        n_frames = sel_data.get('n_frames', 'N/A')
                        color = colors.get(cluster_id, f"C{cluster_id}")
                        
                        # Find first peak for reporting (by position, not height)
                        r = sel_data['r']
                        g_r = sel_data['rdf']
                        try:
                            from scipy.signal import find_peaks
                            
                            # Apply r-range filtering
                            search_mask = np.ones(len(r), dtype=bool)
                            if current_peak_min_r is not None:
                                search_mask &= (r >= current_peak_min_r)
                            if current_peak_max_r is not None:
                                search_mask &= (r <= current_peak_max_r)
                            
                            # Extract search region
                            r_search = r[search_mask]
                            g_r_search = g_r[search_mask]
                            
                            if len(r_search) > 0:
                                # Prepare peak detection parameters
                                peak_kwargs = {}
                                if current_peak_min_height is not None:
                                    peak_kwargs['height'] = current_peak_min_height
                                else:
                                    peak_kwargs['height'] = 1.0  # Default: above bulk density
                                peak_kwargs['prominence'] = current_peak_prominence
                                if current_peak_min_distance is not None:
                                    peak_kwargs['distance'] = current_peak_min_distance
                                
                                # Find all peaks in search region
                                peaks, properties = find_peaks(g_r_search, **peak_kwargs)
                                
                                if len(peaks) > 0:
                                    first_peak_idx = peaks[0]  # First peak by position
                                    first_peak_r = r_search[first_peak_idx]
                                else:
                                    # Fallback: find maximum in search region
                                    first_peak_idx = np.argmax(g_r_search)
                                    first_peak_r = r_search[first_peak_idx] if len(r_search) > 0 else 0
                            else:
                                # No valid search region, use full first half
                                first_peak_idx = np.argmax(g_r[:len(g_r)//2]) if len(g_r) > 0 else 0
                                first_peak_r = r[first_peak_idx] if len(r) > 0 else 0
                        except Exception as e:
                            # Fallback if scipy not available or error
                            first_peak_idx = np.argmax(g_r[:len(g_r)//2]) if len(g_r) > 0 else 0
                            first_peak_r = r[first_peak_idx] if len(r) > 0 else 0
                        
                        # Print statistics (not in legend)
                        print(f"  Cluster {cluster_id}: {n_frames} frames, 1st peak @ {first_peak_r:.2f} Å")
                        
                        ax.plot(sel_data['r'], sel_data['rdf'],
                               label=f'C{cluster_id}',
                               color=color,
                               linewidth=linewidth,
                               linestyle=linestyle,
                               alpha=alpha,
                               zorder=2)
                
                # Add bulk density reference line
                if show_bulk_line:
                    ax.axhline(y=1.0, 
                              color=bulk_line_color,
                              linestyle=bulk_line_style,
                              linewidth=bulk_line_width,
                              alpha=bulk_line_alpha,
                              zorder=0)
                
                # Set labels and title
                ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
                ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                
                if show_title:
                    ax.set_title(f'{sel_name}', 
                               fontsize=title_fontsize, fontweight=title_fontweight)
                
                # Set tick label sizes
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                
                # Set axis limits
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                elif show_RCN:
                    # When showing RCN, ensure primary axis starts at 0 for alignment
                    ax.set_ylim(0, None)
                
                # Add grid
                if grid:
                    ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
                
                # Add shell labels if boundaries provided
                if show_shell_label and current_shell_boundaries is not None and len(current_shell_boundaries) > 0:
                    y_min, y_max = ax.get_ylim()
                    label_y_pos = y_max * 0.98
                    
                    prev_r = 0.0
                    for i, r_max in enumerate(current_shell_boundaries):
                        # Calculate label position based on alignment
                        if shell_label_ha == 'left':
                            label_x_pos = prev_r
                        elif shell_label_ha == 'right':
                            label_x_pos = r_max
                        else:  # 'center'
                            label_x_pos = (prev_r + r_max) / 2
                        
                        label_text = f'S{i+1}' if shell_label_style == 'short' else f'Shell {i+1}'
                        ax.text(label_x_pos, label_y_pos, label_text,
                            ha=shell_label_ha, va='top', fontsize=shell_label_fontsize, 
                            fontweight='bold', color='black')
                        prev_r = r_max
                    
                    # Bulk region extends to figure limit
                    bulk_start = current_shell_boundaries[-1]
                    x_max = ax.get_xlim()[1]
                    if shell_label_ha == 'left':
                        bulk_label_x_pos = bulk_start
                    elif shell_label_ha == 'right':
                        bulk_label_x_pos = x_max
                    else:  # 'center'
                        bulk_label_x_pos = (bulk_start + x_max) / 2
                    
                    ax.text(bulk_label_x_pos, label_y_pos, 'Bulk',
                        ha=shell_label_ha, va='top', fontsize=shell_label_fontsize, 
                        fontweight='bold', color='black')
                
                # Add inset zoom if requested AND if xlim/ylim are provided for this moiety
                if add_inset and current_inset_xlim is not None and current_inset_ylim is not None:
                    if current_inset_bbox is None:
                        ax_inset = ax.inset_axes([0.55, 0.55, 0.35, 0.35])
                    else:
                        xmin, xmax, ymin, ymax = current_inset_bbox
                        xlim_curr, ylim_curr = ax.get_xlim(), ax.get_ylim()
                        
                        left = (xmin - xlim_curr[0]) / (xlim_curr[1] - xlim_curr[0])
                        width = (xmax - xmin) / (xlim_curr[1] - xlim_curr[0])
                        bottom = (ymin - ylim_curr[0]) / (ylim_curr[1] - ylim_curr[0])
                        height = (ymax - ymin) / (ylim_curr[1] - ylim_curr[0])
                        
                        ax_inset = ax.inset_axes([left, bottom, width, height])
                    
                    # Add shell shading to inset if boundaries provided
                    if current_shell_boundaries is not None:
                        inset_xmin, inset_xmax = current_inset_xlim
                        
                        # Add shading for shells that overlap with inset x-range
                        for (r_min, r_max), color in shell_colors_map.items():
                            if r_max > inset_xmin and r_min < inset_xmax:
                                shade_min = max(r_min, inset_xmin)
                                shade_max = min(r_max, inset_xmax)
                                ax_inset.axvspan(shade_min, shade_max, alpha=shell_alpha, 
                                                color=color, zorder=0)
                        
                        # Add bulk shading if it overlaps with inset
                        if bulk_start is not None and bulk_start < inset_xmax:
                            shade_min = max(bulk_start, inset_xmin)
                            ax_inset.axvspan(shade_min, inset_xmax, alpha=shell_alpha, 
                                            color=bulk_color, zorder=0)
                    
                    # Add ion pairing gradient to inset if configured (only if no shell boundaries)
                    if reference_name and ion_pairing_gradient is not None:
                        pair_key = f"{sel_name}-{reference_name}"
                        gradient_config = resolve_param(ion_pairing_gradient, pair_key)
                        
                        should_apply_gradient = False
                        if gradient_config is True:
                            should_apply_gradient = True
                        elif isinstance(gradient_config, dict):
                            should_apply_gradient = gradient_config.get('apply', False)
                        
                        # Shells take priority over gradient
                        if should_apply_gradient and (current_shell_boundaries is None or len(current_shell_boundaries) == 0):
                            inset_xmin, inset_xmax = current_inset_xlim
                            inset_ymin, inset_ymax = current_inset_ylim
                            
                            # Create gradient for inset
                            gradient = np.linspace(0, 1, 256).reshape(1, -1)
                            cmap = create_ion_pairing_gradient_cmap()
                            
                            # Apply gradient to inset
                            extent = [inset_xmin, inset_xmax, inset_ymin, inset_ymax]
                            ax_inset.imshow(gradient, aspect='auto', extent=extent, origin='lower',
                                           cmap=cmap, alpha=ion_pairing_gradient_alpha, zorder=0,
                                           interpolation='bilinear')
                    
                    # Set inset limits BEFORE plotting to ensure proper clipping
                    ax_inset.set_xlim(current_inset_xlim)
                    ax_inset.set_ylim(current_inset_ylim)
                    
                    # Plot RDF curves in inset
                    for cluster_id in cluster_ids:
                        if cluster_id in cluster_data and sel_name in cluster_data[cluster_id]:
                            sel_data = cluster_data[cluster_id][sel_name]
                            color = colors.get(cluster_id, f"C{cluster_id}")
                            ax_inset.plot(sel_data['r'], sel_data['rdf'], color=color, 
                                        linewidth=1.5, alpha=alpha, zorder=2, clip_on=True)
                    
                    # Inset styling
                    ax_inset.tick_params(labelsize=max(6, tick_fontsize - 2))
                    ax_inset.grid(False)
                    
                    # Format tick labels to 1 decimal place
                    from matplotlib.ticker import FormatStrFormatter
                    ax_inset.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax_inset.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    
                    # Add colorbar if requested
                    if add_inset_colorbar:
                        from matplotlib.colors import Normalize
                        import matplotlib.cm as cm
                        
                        # Get inset position in figure coordinates
                        inset_bbox_fig = ax_inset.get_position()
                        
                        # Create colorbar axes to the right of inset, same height
                        cbar_width_val = 0.03
                        cbar_spacing = 0.1
                        cbar_ax = fig.add_axes([
                            inset_bbox_fig.x1 + cbar_spacing,
                            inset_bbox_fig.y0,
                            cbar_width_val,
                            inset_bbox_fig.height
                        ])
                        
                        # Create colorbar mapping cluster IDs
                        norm = Normalize(vmin=min(cluster_ids), vmax=max(cluster_ids))
                        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='tab10'),
                                           cax=cbar_ax, orientation='vertical')
                        cbar.set_label('Cluster ID', fontsize=label_fontsize*0.8, rotation=90, labelpad=8)
                        cbar.ax.tick_params(labelsize=label_fontsize*0.7)
                
                print(f"DEBUG: Before RCN section - show_RCN={show_RCN} for {sel_name}")
                # Add secondary axis for coordination number if requested
                if show_RCN:
                    print(f"DEBUG: show_RCN=True, entering RCN calculation for {sel_name}")
                    ax2 = ax.twinx()
                    
                    # Determine scaling factor for RCN values
                    rcn_scale = 1000 if RCN_notation == 'scaled' else 1
                    print(f"DEBUG: RCN_notation={RCN_notation}, rcn_scale={rcn_scale}")
                    
                    # Plot coordination numbers
                    for cluster_id in cluster_ids:
                        print(f"DEBUG: Processing cluster {cluster_id}")
                        if cluster_id in cluster_data and sel_name in cluster_data[cluster_id]:
                            sel_data = cluster_data[cluster_id][sel_name]
                            print(f"DEBUG: Found data for {sel_name}, cluster {cluster_id}")
                            
                            # Check what's currently in the sel_data
                            current_count = sel_data.get('count', 'NOT_FOUND')
                            if hasattr(current_count, '__len__') and len(current_count) > 0:
                                print(f"DEBUG: Current 'count' max value: {max(current_count):.2f}")
                            else:
                                print(f"DEBUG: Current 'count': {current_count}")
                            
                            # Always recalculate RCN when show_RCN=True to ensure correct formula
                            # This overrides any pre-calculated 'count' values that may use incorrect formula
                            r_vals = sel_data['r']
                            rdf_vals = sel_data['rdf']
                            print(f"DEBUG: r_vals range: {r_vals[0]:.2f} to {r_vals[-1]:.2f}")
                            print(f"DEBUG: rdf_vals range: {min(rdf_vals):.4f} to {max(rdf_vals):.4f}")
                            
                            # Calculate number density (key missing factor!)
                            # Assuming standard water density for now - could be made more sophisticated
                            # This is a simplified estimate - for exact calculation we'd need actual particle counts and box dimensions
                            rho = 0.0334  # particles/Å³ (approximate water density)
                            
                            print(f"DEBUG: Using density rho = {rho:.6f} particles/Å³ for RCN calculation")

                            # Calculate running coordination number: N(r) = 4π * ρ * ∫₀ʳ g(r') * r'² dr'
                            rcn_values = np.zeros_like(r_vals)
                            for i in range(1, len(r_vals)):
                                # Calculate g(r') * r'² for proper volume weighting
                                integrand = rdf_vals[:i+1] * (r_vals[:i+1] ** 2)
                                # Integrate with 4π * ρ factor (ρ is the crucial missing piece!)
                                rcn_values[i] = 4.0 * np.pi * rho * trapz(integrand, r_vals[:i+1])
                            
                            # Store corrected values back
                            sel_data['count'] = rcn_values
                            print(f"   *** NEW CALCULATED RCN for {sel_name}, cluster {cluster_id}: max N(r) = {rcn_values[-1]:.2f} ***")
                            
                            color = colors.get(cluster_id, f"C{cluster_id}")
                            ax2.plot(sel_data['r'], rcn_values / rcn_scale,
                                   linestyle=RCN_curve_style,
                                   linewidth=RCN_curve_weight,
                                   color=color,
                                   alpha=alpha)
                        else:
                            print(f"DEBUG: No data found for {sel_name}, cluster {cluster_id}")
                    
                    # Set secondary y-axis label and styling (inherit from primary if not specified)
                    rcn_label_fs = RCN_label_fontsize if RCN_label_fontsize is not None else label_fontsize
                    rcn_label_fw = RCN_label_fontweight if RCN_label_fontweight is not None else label_fontweight
                    rcn_tick_fs = RCN_tick_fontsize if RCN_tick_fontsize is not None else tick_fontsize
                    
                    # Apply RCN notation formatting if requested
                    if RCN_notation == 'offset':
                        # Option 1: Offset notation (10, 20, 30... with ×10³ at top)
                        from matplotlib.ticker import ScalarFormatter
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_powerlimits((0, 0))  # Always use scientific notation
                        ax2.yaxis.set_major_formatter(formatter)
                        ax2.set_ylabel(RCN_ylabel, fontsize=rcn_label_fs, fontweight=rcn_label_fw)
                        # Increase offset text size to match tick labels
                        ax2.yaxis.offsetText.set_fontsize(rcn_tick_fs)
                    elif RCN_notation == 'scaled':
                        # Option 2: Manual scaling (divide by 1000, update label)
                        ax2.set_ylabel(f'{RCN_ylabel} (×10³)', fontsize=rcn_label_fs, fontweight=rcn_label_fw)
                    else:
                        # Default: no special formatting
                        ax2.set_ylabel(RCN_ylabel, fontsize=rcn_label_fs, fontweight=rcn_label_fw)
                    
                    ax2.tick_params(axis='y', labelsize=rcn_tick_fs)
                    
                    # Smart Y-axis scaling for RCN (properly accounting for RCN_notation scaling)
                    # Try to get coordination radius for intelligent scaling
                    r0 = None
                    
                    # For now, use a typical coordination radius for water around ions (~2.5-3.0 Å)
                    # In a more sophisticated implementation, this could be extracted from ion parameters
                    r0 = 2.8  # Approximate first shell radius for ion-water coordination
                    
                    if r0 is not None:
                        # Find CN at coordination radius from any available cluster data  
                        cn_at_r0_raw = None
                        for cluster_id in cluster_ids:
                            if cluster_id in cluster_data and sel_name in cluster_data[cluster_id]:
                                sel_data = cluster_data[cluster_id][sel_name]
                                if 'count' in sel_data and 'r' in sel_data:
                                    r_vals = sel_data['r']
                                    rcn_vals = sel_data['count']
                                    if len(r_vals) > 0 and len(rcn_vals) > 0:
                                        idx_r0 = np.argmin(np.abs(r_vals - r0))
                                        cn_at_r0_raw = rcn_vals[idx_r0]  # Keep raw value for scaling calculation
                                        break
                        
                        if cn_at_r0_raw is not None and cn_at_r0_raw > 0:
                            # Calculate displayed value (accounting for rcn_scale)
                            cn_at_r0_displayed = cn_at_r0_raw / rcn_scale
                            
                            # Scale so CN at r₀ appears at 1/rcn_scale_factor of figure height (in displayed units)
                            y_max_rcn_displayed = cn_at_r0_displayed * rcn_scale_factor
                            ax2.set_ylim(0, y_max_rcn_displayed)
                            
                            print(f"DEBUG: Smart RCN scaling - CN at r₀={r0:.1f}Å: {cn_at_r0_displayed:.1f} (displayed), y_max: {y_max_rcn_displayed:.1f}, scale_factor: {rcn_scale_factor}")
                        else:
                            # Fallback to simple scaling
                            ax2.set_ylim(0, None)
                            print("DEBUG: Using fallback RCN scaling")
                    else:
                        ax2.set_ylim(0, None)  # Original fallback
                
                print(f"DEBUG: COMPLETED RCN setup for {sel_name}")
                
                # Add legend
                if show_legend:
                    if current_legend_bbox is not None:
                        legend = ax.legend(loc=legend_loc, bbox_to_anchor=current_legend_bbox, ncol=legend_ncol, 
                                 fontsize=legend_fontsize, framealpha=legend_frame_alpha, frameon=legend_frameon,
                                 columnspacing=legend_columnspacing)
                    else:
                        legend = ax.legend(loc=legend_loc, ncol=legend_ncol, fontsize=legend_fontsize,
                                 framealpha=legend_frame_alpha, frameon=legend_frameon,
                                 columnspacing=legend_columnspacing)
                    for text in legend.get_texts():
                        text.set_fontweight(legend_fontweight)
                
                # Save individual moiety figure if requested
                if save_individual_figures:
                    fig_individual, ax_individual = plt.subplots(figsize=individual_figsize)
                    
                    # Add shell shading to individual figure
                    if current_shell_boundaries is not None and len(current_shell_boundaries) > 0:
                        # Use same shell colors
                        prev_r = 0.0
                        for i, r_max in enumerate(current_shell_boundaries):
                            color = shell_colors[i]
                            ax_individual.axvspan(prev_r, r_max, alpha=shell_alpha, color=color, zorder=0)
                            prev_r = r_max
                        
                        # Add bulk region (extends to figure limit)
                        bulk_start = current_shell_boundaries[-1]
                        bulk_end = xlim[1] if xlim is not None else 20
                        ax_individual.axvspan(bulk_start, bulk_end, alpha=shell_alpha, color=bulk_color, zorder=0)
                    
                    # Add ion pairing gradient to individual figure if configured
                    if reference_name and ion_pairing_gradient is not None:
                        pair_key = f"{sel_name}-{reference_name}"
                        gradient_config = resolve_param(ion_pairing_gradient, pair_key)
                        
                        should_apply_gradient = False
                        if gradient_config is True:
                            should_apply_gradient = True
                        elif isinstance(gradient_config, dict):
                            should_apply_gradient = gradient_config.get('apply', False)
                        
                        # Shells take priority over gradient
                        if should_apply_gradient and (current_shell_boundaries is None or len(current_shell_boundaries) == 0):
                            x_min = xlim[0] if xlim is not None else 0
                            x_max = xlim[1] if xlim is not None else 20
                            y_min = ylim[0] if ylim is not None else 0
                            if ylim is not None:
                                y_max = ylim[1]
                            else:
                                # Auto-detect y_max from data
                                max_rdf = 0
                                for cid in cluster_ids:
                                    if cid in cluster_data and sel_name in cluster_data[cid]:
                                        max_rdf = max(max_rdf, np.max(cluster_data[cid][sel_name]['rdf']))
                                y_max = max_rdf * 1.1  # Add 10% padding
                            
                            gradient = np.linspace(0, 1, 256).reshape(1, -1)
                            cmap = create_ion_pairing_gradient_cmap()
                            extent = [x_min, x_max, y_min, y_max]
                            ax_individual.imshow(gradient, aspect='auto', extent=extent, origin='lower',
                                               cmap=cmap, alpha=ion_pairing_gradient_alpha, zorder=0,
                                               interpolation='bilinear')
                    
                    # Plot this moiety for all clusters (with peak detection)
                    for cluster_id in cluster_ids:
                        if cluster_id in cluster_data and sel_name in cluster_data[cluster_id]:
                            sel_data = cluster_data[cluster_id][sel_name]
                            n_frames = sel_data.get('n_frames', 'N/A')
                            color = colors.get(cluster_id, f"C{cluster_id}")
                            r = sel_data['r']
                            g_r = sel_data['rdf']
                            # Find first peak by position (not height) with same parameters
                            try:
                                from scipy.signal import find_peaks
                                
                                # Apply r-range filtering
                                search_mask = np.ones(len(r), dtype=bool)
                                if current_peak_min_r is not None:
                                    search_mask &= (r >= current_peak_min_r)
                                if current_peak_max_r is not None:
                                    search_mask &= (r <= current_peak_max_r)
                                
                                r_search = r[search_mask]
                                g_r_search = g_r[search_mask]
                                
                                if len(r_search) > 0:
                                    peak_kwargs = {}
                                    if current_peak_min_height is not None:
                                        peak_kwargs['height'] = current_peak_min_height
                                    else:
                                        peak_kwargs['height'] = 1.0
                                    peak_kwargs['prominence'] = current_peak_prominence
                                    if current_peak_min_distance is not None:
                                        peak_kwargs['distance'] = current_peak_min_distance
                                    
                                    peaks, properties = find_peaks(g_r_search, **peak_kwargs)
                                    
                                    if len(peaks) > 0:
                                        first_peak_idx = peaks[0]
                                        first_peak_r = r_search[first_peak_idx]
                                    else:
                                        first_peak_idx = np.argmax(g_r_search)
                                        first_peak_r = r_search[first_peak_idx] if len(r_search) > 0 else 0
                                else:
                                    first_peak_idx = np.argmax(g_r[:len(g_r)//2]) if len(g_r) > 0 else 0
                                    first_peak_r = r[first_peak_idx] if len(r) > 0 else 0
                            except:
                                first_peak_idx = np.argmax(g_r[:len(g_r)//2]) if len(g_r) > 0 else 0
                                first_peak_r = r[first_peak_idx] if len(r) > 0 else 0
                            
                            ax_individual.plot(sel_data['r'], sel_data['rdf'],
                                   label=f'C{cluster_id}',
                                   color=color,
                                   linewidth=linewidth,
                                   linestyle=linestyle,
                                   alpha=alpha,
                                   zorder=2)
                    
                    if show_bulk_line:
                        ax_individual.axhline(y=1.0, color=bulk_line_color, linestyle=bulk_line_style,
                                  linewidth=bulk_line_width, alpha=bulk_line_alpha, zorder=0)
                    
                    ax_individual.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
                    ax_individual.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                    if show_title:
                        ax_individual.set_title(f'{sel_name}', fontsize=title_fontsize, fontweight=title_fontweight)
                    ax_individual.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    if xlim is not None:
                        ax_individual.set_xlim(xlim)
                    if ylim is not None:
                        ax_individual.set_ylim(ylim)
                    elif show_RCN:
                        # When showing RCN, ensure primary axis starts at 0 for alignment
                        ax_individual.set_ylim(0, None)
                    if grid:
                        ax_individual.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
                    
                    # Add shell labels to individual figure
                    if show_shell_label and current_shell_boundaries is not None and len(current_shell_boundaries) > 0:
                        y_min, y_max = ax_individual.get_ylim()
                        label_y_pos = y_max * 0.98
                        
                        prev_r = 0.0
                        for i, r_max in enumerate(current_shell_boundaries):
                            # Calculate label position based on alignment
                            if shell_label_ha == 'left':
                                label_x_pos = prev_r
                            elif shell_label_ha == 'right':
                                label_x_pos = r_max
                            else:  # 'center'
                                label_x_pos = (prev_r + r_max) / 2
                            
                            label_text = f'S{i+1}' if shell_label_style == 'short' else f'Shell {i+1}'
                            ax_individual.text(label_x_pos, label_y_pos, label_text,
                                ha=shell_label_ha, va='top', fontsize=shell_label_fontsize, 
                                fontweight='bold', color='black')
                            prev_r = r_max
                        
                        # Bulk region extends to figure limit
                        bulk_start = current_shell_boundaries[-1]
                        x_max = ax_individual.get_xlim()[1]
                        if shell_label_ha == 'left':
                            bulk_label_x_pos = bulk_start
                        elif shell_label_ha == 'right':
                            bulk_label_x_pos = x_max
                        else:  # 'center'
                            bulk_label_x_pos = (bulk_start + x_max) / 2
                        
                        ax_individual.text(bulk_label_x_pos, label_y_pos, 'Bulk',
                            ha=shell_label_ha, va='top', fontsize=shell_label_fontsize, 
                            fontweight='bold', color='black')
                    
                    # Add inset zoom to individual figure if requested
                    if add_inset and current_inset_xlim is not None and current_inset_ylim is not None:
                        if current_inset_bbox is None:
                            ax_inset_ind = ax_individual.inset_axes([0.55, 0.55, 0.35, 0.35])
                        else:
                            xmin, xmax, ymin, ymax = current_inset_bbox
                            xlim_curr, ylim_curr = ax_individual.get_xlim(), ax_individual.get_ylim()
                            
                            left = (xmin - xlim_curr[0]) / (xlim_curr[1] - xlim_curr[0])
                            width = (xmax - xmin) / (xlim_curr[1] - xlim_curr[0])
                            bottom = (ymin - ylim_curr[0]) / (ylim_curr[1] - ylim_curr[0])
                            height = (ymax - ymin) / (ylim_curr[1] - ylim_curr[0])
                            
                            ax_inset_ind = ax_individual.inset_axes([left, bottom, width, height])
                        
                        # Add shell shading to inset if boundaries provided
                        if current_shell_boundaries is not None and len(current_shell_boundaries) > 0:
                            inset_xmin, inset_xmax = current_inset_xlim
                            n_shells = len(current_shell_boundaries)
                            all_colors = get_blue_saturation_colors_from_00c5ff(n_shells)
                            shell_colors = all_colors[:-1]
                            bulk_color = all_colors[-1]
                            
                            # Add shading for shells that overlap with inset x-range
                            prev_r = 0.0
                            for i, r_max in enumerate(current_shell_boundaries):
                                if r_max > inset_xmin and prev_r < inset_xmax:
                                    shade_min = max(prev_r, inset_xmin)
                                    shade_max = min(r_max, inset_xmax)
                                    ax_inset_ind.axvspan(shade_min, shade_max, alpha=shell_alpha, 
                                                    color=shell_colors[i], zorder=0)
                                prev_r = r_max
                            
                            # Add bulk shading if it overlaps with inset
                            bulk_start = current_shell_boundaries[-1]
                            if bulk_start < inset_xmax:
                                shade_min = max(bulk_start, inset_xmin)
                                ax_inset_ind.axvspan(shade_min, inset_xmax, alpha=shell_alpha, 
                                                color=bulk_color, zorder=0)
                        
                        # Add ion pairing gradient to inset if configured
                        if reference_name and ion_pairing_gradient is not None:
                            pair_key = f"{sel_name}-{reference_name}"
                            gradient_config = resolve_param(ion_pairing_gradient, pair_key)
                            
                            should_apply_gradient = False
                            if gradient_config is True:
                                should_apply_gradient = True
                            elif isinstance(gradient_config, dict):
                                should_apply_gradient = gradient_config.get('apply', False)
                            
                           # Shells take priority over gradient
                            if should_apply_gradient and (current_shell_boundaries is None or len(current_shell_boundaries) == 0):
                                inset_xmin, inset_xmax = current_inset_xlim
                                inset_ymin, inset_ymax = current_inset_ylim
                                
                                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                                cmap = create_ion_pairing_gradient_cmap()
                                extent = [inset_xmin, inset_xmax, inset_ymin, inset_ymax]
                                ax_inset_ind.imshow(gradient, aspect='auto', extent=extent, origin='lower',
                                                   cmap=cmap, alpha=ion_pairing_gradient_alpha, zorder=0,
                                                   interpolation='bilinear')
                        
                        # Set inset limits BEFORE plotting
                        ax_inset_ind.set_xlim(current_inset_xlim)
                        ax_inset_ind.set_ylim(current_inset_ylim)
                        
                        # Plot RDF curves in inset (all clusters for this moiety)
                        for cluster_id in cluster_ids:
                            if cluster_id in cluster_data and sel_name in cluster_data[cluster_id]:
                                sel_data = cluster_data[cluster_id][sel_name]
                                color = colors.get(cluster_id, f"C{cluster_id}")
                                ax_inset_ind.plot(sel_data['r'], sel_data['rdf'], color=color, 
                                            linewidth=1.5, alpha=alpha, zorder=2, clip_on=True)
                        
                        # Inset styling
                        ax_inset_ind.tick_params(labelsize=max(6, tick_fontsize - 2))
                        ax_inset_ind.grid(False)
                        
                        # Format tick labels to 1 decimal place
                        from matplotlib.ticker import FormatStrFormatter
                        ax_inset_ind.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        ax_inset_ind.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        
                        # Add colorbar if requested
                        if add_inset_colorbar:
                            from matplotlib.colors import Normalize
                            import matplotlib.cm as cm
                            
                            # Get inset position in figure coordinates
                            inset_bbox_fig = ax_inset_ind.get_position()
                            
                            # Create colorbar axes to the right of inset, same height
                            cbar_width_val = 0.03
                            cbar_spacing = 0.1
                            cbar_ax = fig_individual.add_axes([
                                inset_bbox_fig.x1 + cbar_spacing,
                                inset_bbox_fig.y0,
                                cbar_width_val,
                                inset_bbox_fig.height
                            ])
                            
                            # Create colorbar mapping cluster IDs
                            norm = Normalize(vmin=min(cluster_ids), vmax=max(cluster_ids))
                            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='tab10'),
                                               cax=cbar_ax, orientation='vertical')
                            cbar.set_label('Cluster ID', fontsize=label_fontsize*0.8, rotation=90, labelpad=8)
                            cbar.ax.tick_params(labelsize=label_fontsize*0.7)
                    
                    # Add secondary axis for coordination number if requested
                    if show_RCN:
                        ax2_ind = ax_individual.twinx()
                        
                        # Determine scaling factor for RCN values
                        rcn_scale = 1000 if RCN_notation == 'scaled' else 1
                        
                        # Plot coordination numbers
                        for cluster_id in cluster_ids:
                            if cluster_id in cluster_data and sel_name in cluster_data[cluster_id]:
                                sel_data = cluster_data[cluster_id][sel_name]
                                
                                # Always recalculate RCN when show_RCN=True to ensure correct formula
                                # This overrides potentially incorrect pre-calculated 'count' values
                                r_vals = sel_data['r']
                                rdf_vals = sel_data['rdf']
                                
                                # Calculate number density (key missing factor!)
                                rho = 0.0334  # particles/Å³ (approximate water density)
                                
                                # Calculate running coordination number: N(r) = 4π * ρ * ∫₀ʳ g(r') * r'² dr'
                                rcn_values = np.zeros_like(r_vals)
                                for i in range(1, len(r_vals)):
                                    # Calculate g(r') * r'² for proper volume weighting
                                    integrand = rdf_vals[:i+1] * (r_vals[:i+1] ** 2)
                                    # Integrate with 4π * ρ factor (ρ is the crucial missing piece!)
                                    rcn_values[i] = 4.0 * np.pi * rho * trapz(integrand, r_vals[:i+1])
                                
                                # Store corrected values back
                                sel_data['count'] = rcn_values
                                print(f"   Recalculated RCN for individual plot {sel_name}, cluster {cluster_id}: max N(r) = {rcn_values[-1]:.2f}")
                                
                                if rcn_values is not None:
                                    color = colors.get(cluster_id, f"C{cluster_id}")
                                    ax2_ind.plot(sel_data['r'], rcn_values / rcn_scale,
                                           linestyle=RCN_curve_style,
                                           linewidth=RCN_curve_weight,
                                           color=color,
                                           alpha=alpha)
                        
                        # Set secondary y-axis label and styling (inherit from primary if not specified)
                        rcn_label_fs = RCN_label_fontsize if RCN_label_fontsize is not None else label_fontsize
                        rcn_label_fw = RCN_label_fontweight if RCN_label_fontweight is not None else label_fontweight
                        rcn_tick_fs = RCN_tick_fontsize if RCN_tick_fontsize is not None else tick_fontsize
                        
                        # Apply RCN notation formatting if requested
                        if RCN_notation == 'offset':
                            # Option 1: Offset notation (10, 20, 30... with ×10³ at top)
                            from matplotlib.ticker import ScalarFormatter
                            formatter = ScalarFormatter(useMathText=True)
                            formatter.set_powerlimits((0, 0))  # Always use scientific notation
                            ax2_ind.yaxis.set_major_formatter(formatter)
                            ax2_ind.set_ylabel(RCN_ylabel, fontsize=rcn_label_fs, fontweight=rcn_label_fw)
                            # Increase offset text size to match tick labels
                            ax2_ind.yaxis.offsetText.set_fontsize(rcn_tick_fs)
                        elif RCN_notation == 'scaled':
                            # Option 2: Manual scaling (divide by 1000, update label)
                            ax2_ind.set_ylabel(f'{RCN_ylabel} (×10³)', fontsize=rcn_label_fs, fontweight=rcn_label_fw)
                        else:
                            # Default: no special formatting
                            ax2_ind.set_ylabel(RCN_ylabel, fontsize=rcn_label_fs, fontweight=rcn_label_fw)
                        
                        ax2_ind.tick_params(axis='y', labelsize=rcn_tick_fs)
                        
                        # Smart Y-axis scaling for individual plots (same as main plot)
                        r0 = 2.8  # Approximate first shell radius for ion-water coordination
                        
                        if r0 is not None:
                            # Find CN at coordination radius from any available cluster data  
                            cn_at_r0_raw = None
                            for cluster_id in cluster_ids:
                                if cluster_id in cluster_data and sel_name in cluster_data[cluster_id]:
                                    sel_data = cluster_data[cluster_id][sel_name]
                                    if 'count' in sel_data and 'r' in sel_data:
                                        r_vals = sel_data['r']
                                        rcn_vals = sel_data['count']
                                        if len(r_vals) > 0 and len(rcn_vals) > 0:
                                            idx_r0 = np.argmin(np.abs(r_vals - r0))
                                            cn_at_r0_raw = rcn_vals[idx_r0]  # Keep raw value for scaling calculation
                                            break
                            
                            if cn_at_r0_raw is not None and cn_at_r0_raw > 0:
                                # Calculate displayed value (accounting for rcn_scale)
                                cn_at_r0_displayed = cn_at_r0_raw / rcn_scale
                                
                                # Scale so CN at r₀ appears at 1/rcn_scale_factor of figure height (in displayed units)
                                y_max_rcn_displayed = cn_at_r0_displayed * rcn_scale_factor
                                ax2_ind.set_ylim(0, y_max_rcn_displayed)
                            else:
                                # Fallback to simple scaling
                                ax2_ind.set_ylim(0, None)
                        else:
                            ax2_ind.set_ylim(0, None)  # Original fallback
                    
                    if show_legend:
                        if current_legend_bbox is not None:
                            legend_ind = ax_individual.legend(loc=legend_loc, bbox_to_anchor=current_legend_bbox, 
                                     ncol=legend_ncol, fontsize=legend_fontsize,
                                     framealpha=legend_frame_alpha, frameon=legend_frameon,
                                     columnspacing=legend_columnspacing)
                        else:
                            legend_ind = ax_individual.legend(loc=legend_loc, ncol=legend_ncol, fontsize=legend_fontsize,
                                     framealpha=legend_frame_alpha, frameon=legend_frameon,
                                     columnspacing=legend_columnspacing)
                        for text in legend_ind.get_texts():
                            text.set_fontweight(legend_fontweight)
                    
                    plt.tight_layout()
                    
                    import os
                    save_dir = individual_save_dir if individual_save_dir is not None else (os.path.dirname(filename) or '.')
                    os.makedirs(save_dir, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(filename))[0]
                    individual_filename = os.path.join(save_dir, f"{base_name}_{sel_name}.png")
                    fig_individual.savefig(individual_filename, dpi=dpi, bbox_inches=bbox_inches)
                    print(f"  ✓ Saved: {individual_filename}")
                    plt.close(fig_individual)
        
        else:
            # SEPARATE MODE: One panel per cluster, showing all moieties (original behavior)
            for idx, cluster_id in enumerate(cluster_ids):
                ax = axes[idx]
                cluster_selections = cluster_data[cluster_id]
                
                # Get frame count for title
                first_sel = next(iter(cluster_selections.values()))
                n_frames = first_sel.get('n_frames', 'N/A')
                
                # Add shell shading for each moiety (will overlay multiple if multiple moieties have boundaries)
                for sel_name in selection_names:
                    if sel_name in cluster_selections:
                        current_shell_boundaries = resolve_param(shell_boundaries, sel_name)
                        
                        if current_shell_boundaries is not None and len(current_shell_boundaries) > 0:
                            n_shells = len(current_shell_boundaries)
                            all_colors = get_blue_saturation_colors_from_00c5ff(n_shells)
                            shell_colors = all_colors[:-1]
                            bulk_color = all_colors[-1]
                            
                            # Add shading for each shell
                            prev_r = 0.0
                            for i, r_max in enumerate(current_shell_boundaries):
                                color = shell_colors[i]
                                ax.axvspan(prev_r, r_max, alpha=shell_alpha, color=color, zorder=0)
                                prev_r = r_max
                            
                            # Add bulk region (extends to figure limit)
                            bulk_start = current_shell_boundaries[-1]
                            bulk_end = xlim[1] if xlim is not None else 20
                            ax.axvspan(bulk_start, bulk_end, alpha=shell_alpha, color=bulk_color, zorder=0)
                            
                            # Add shell labels (will be positioned correctly after plotting sets ylim)
                            # Store for later positioning
                            shell_boundaries_to_label = current_shell_boundaries
                            
                            # Only show shell for first moiety with boundaries to avoid clutter
                            break
                
                # Add ion pairing gradient if configured (only for first moiety without shells)
                # Check each moiety for gradient configuration
                gradient_applied = False
                for sel_name in selection_names:
                    if sel_name in cluster_selections and not gradient_applied:
                        sel_data = cluster_selections[sel_name]
                        reference_name = sel_data.get('reference', None)
                        current_shell_boundaries = resolve_param(shell_boundaries, sel_name)
                        
                        if reference_name and ion_pairing_gradient is not None:
                            pair_key = f"{sel_name}-{reference_name}"
                            gradient_config = resolve_param(ion_pairing_gradient, pair_key)
                            
                            should_apply_gradient = False
                            if gradient_config is True:
                                should_apply_gradient = True
                            elif isinstance(gradient_config, dict):
                                should_apply_gradient = gradient_config.get('apply', False)
                            
                            # Shells take priority, apply only first matching gradient
                            if should_apply_gradient and (current_shell_boundaries is None or len(current_shell_boundaries) == 0):
                                x_min = xlim[0] if xlim is not None else 0
                                x_max = xlim[1] if xlim is not None else 20
                                y_min = ylim[0] if ylim is not None else 0
                                if ylim is not None:
                                    y_max = ylim[1]
                                else:
                                    # Auto-detect y_max from data for this cluster
                                    max_rdf = max(np.max(cluster_selections[sn]['rdf']) for sn in selection_names if sn in cluster_selections)
                                    y_max = max_rdf * 1.1  # Add 10% padding
                                
                                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                                cmap = create_ion_pairing_gradient_cmap()
                                extent = [x_min, x_max, y_min, y_max]
                                ax.imshow(gradient, aspect='auto', extent=extent, origin='lower',
                                         cmap=cmap, alpha=ion_pairing_gradient_alpha, zorder=0,
                                         interpolation='bilinear')
                                gradient_applied = True
                                break
                
                # Plot each selection
                for sel_name in selection_names:
                    if sel_name in cluster_selections:
                        sel_data = cluster_selections[sel_name]
                        color = colors.get(sel_name, f"C{len(colors)}")
                        ax.plot(sel_data['r'], sel_data['rdf'],
                               label=sel_name,
                               color=color,
                               linewidth=linewidth,
                               linestyle=linestyle,
                               alpha=alpha,
                               zorder=2)
                
                # Add bulk density reference line
                if show_bulk_line:
                    ax.axhline(y=1.0, 
                              color=bulk_line_color,
                              linestyle=bulk_line_style,
                              linewidth=bulk_line_width,
                              alpha=bulk_line_alpha,
                              zorder=0)
                
                # Set labels and title for each subplot
                ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
                ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                
                if show_title:
                    ax.set_title(f'Cluster {cluster_id} ({n_frames} frames)', 
                               fontsize=title_fontsize, fontweight=title_fontweight)
                
                # Set tick label sizes
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                
                # Set axis limits
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                
                # Add grid
                if grid:
                    ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
                
                # Add shell labels after plotting (so ylim is set correctly)
                if show_shell_label:
                    for sel_name in selection_names:
                        current_shell_boundaries = resolve_param(shell_boundaries, sel_name)
                        if current_shell_boundaries is not None and len(current_shell_boundaries) > 0:
                            y_min, y_max = ax.get_ylim()
                            label_y_pos = y_max * 0.98
                            
                            prev_r = 0.0
                            for i, r_max in enumerate(current_shell_boundaries):
                                # Calculate label position based on alignment
                                if shell_label_ha == 'left':
                                    label_x_pos = prev_r
                                elif shell_label_ha == 'right':
                                    label_x_pos = r_max
                                else:  # 'center'
                                    label_x_pos = (prev_r + r_max) / 2
                                
                                label_text = f'S{i+1}' if shell_label_style == 'short' else f'Shell {i+1}'
                                ax.text(label_x_pos, label_y_pos, label_text,
                                    ha=shell_label_ha, va='top', fontsize=shell_label_fontsize, 
                                    fontweight='bold', color='black')
                                prev_r = r_max
                            
                            # Bulk region extends to figure limit
                            bulk_start = current_shell_boundaries[-1]
                            x_max = ax.get_xlim()[1]
                            if shell_label_ha == 'left':
                                bulk_label_x_pos = bulk_start
                            elif shell_label_ha == 'right':
                                bulk_label_x_pos = x_max
                            else:  # 'center'
                                bulk_label_x_pos = (bulk_start + x_max) / 2
                            
                            ax.text(bulk_label_x_pos, label_y_pos, 'Bulk',
                                ha=shell_label_ha, va='top', fontsize=shell_label_fontsize, 
                                fontweight='bold', color='black')
                            
                            # Only add labels for first moiety with boundaries
                            break
                
                # Add inset zoom if requested (check each moiety for inset parameters)
                for sel_name in selection_names:
                    current_inset_xlim = resolve_param(inset_xlim, sel_name)
                    current_inset_ylim = resolve_param(inset_ylim, sel_name)
                    current_inset_bbox = resolve_param(inset_bbox, sel_name)
                    current_shell_boundaries = resolve_param(shell_boundaries, sel_name)
                    
                    if add_inset and current_inset_xlim is not None and current_inset_ylim is not None:
                        if current_inset_bbox is None:
                            ax_inset = ax.inset_axes([0.55, 0.55, 0.35, 0.35])
                        else:
                            xmin, xmax, ymin, ymax = current_inset_bbox
                            xlim_curr, ylim_curr = ax.get_xlim(), ax.get_ylim()
                            
                            left = (xmin - xlim_curr[0]) / (xlim_curr[1] - xlim_curr[0])
                            width = (xmax - xmin) / (xlim_curr[1] - xlim_curr[0])
                            bottom = (ymin - ylim_curr[0]) / (ylim_curr[1] - ylim_curr[0])
                            height = (ymax - ymin) / (ylim_curr[1] - ylim_curr[0])
                            
                            ax_inset = ax.inset_axes([left, bottom, width, height])
                        
                        # Add shell shading to inset if boundaries provided for this moiety
                        if current_shell_boundaries is not None and len(current_shell_boundaries) > 0:
                            inset_xmin, inset_xmax = current_inset_xlim
                            n_shells = len(current_shell_boundaries)
                            all_colors = get_blue_saturation_colors_from_00c5ff(n_shells)
                            shell_colors = all_colors[:-1]
                            bulk_color = all_colors[-1]
                            
                            # Create shell colors map for inset
                            prev_r = 0.0
                            for i, r_max in enumerate(current_shell_boundaries):
                                if r_max > inset_xmin and prev_r < inset_xmax:
                                    shade_min = max(prev_r, inset_xmin)
                                    shade_max = min(r_max, inset_xmax)
                                    ax_inset.axvspan(shade_min, shade_max, alpha=shell_alpha, 
                                                    color=shell_colors[i], zorder=0)
                                prev_r = r_max
                            
                            # Add bulk shading if it overlaps with inset
                            bulk_start = current_shell_boundaries[-1]
                            if bulk_start < inset_xmax:
                                shade_min = max(bulk_start, inset_xmin)
                                ax_inset.axvspan(shade_min, inset_xmax, alpha=shell_alpha, 
                                                color=bulk_color, zorder=0)
                        
                        # Add ion pairing gradient to inset if configured and no shell boundaries
                        if sel_name in cluster_selections:
                            sel_data = cluster_selections[sel_name]
                            reference_name = sel_data.get('reference', None)
                            
                            if reference_name and ion_pairing_gradient is not None:
                                pair_key = f"{sel_name}-{reference_name}"
                                gradient_config = resolve_param(ion_pairing_gradient, pair_key)
                                
                                should_apply_gradient = False
                                if gradient_config is True:
                                    should_apply_gradient = True
                                elif isinstance(gradient_config, dict):
                                    should_apply_gradient = gradient_config.get('apply', False)
                                
                                # Only apply gradient if no shell boundaries (shells take priority)
                                if should_apply_gradient and (current_shell_boundaries is None or len(current_shell_boundaries) == 0):
                                    inset_xmin, inset_xmax = current_inset_xlim
                                    inset_ymin, inset_ymax = current_inset_ylim
                                    
                                    gradient = np.linspace(0, 1, 256).reshape(1, -1)
                                    cmap = create_ion_pairing_gradient_cmap()
                                    extent = [inset_xmin, inset_xmax, inset_ymin, inset_ymax]
                                    ax_inset.imshow(gradient, aspect='auto', extent=extent, origin='lower',
                                                   cmap=cmap, alpha=ion_pairing_gradient_alpha, zorder=0,
                                                   interpolation='bilinear')
                        
                        # Set inset limits BEFORE plotting
                        ax_inset.set_xlim(current_inset_xlim)
                        ax_inset.set_ylim(current_inset_ylim)
                        
                        # Plot RDF curves in inset (all moieties in this cluster)
                        for moiety in selection_names:
                            if moiety in cluster_selections:
                                sel_data = cluster_selections[moiety]
                                color = colors.get(moiety, f"C{len(colors)}")
                                ax_inset.plot(sel_data['r'], sel_data['rdf'], color=color, 
                                            linewidth=1.5, alpha=alpha, zorder=2, clip_on=True)
                        
                        # Inset styling
                        ax_inset.tick_params(labelsize=max(6, tick_fontsize - 2))
                        ax_inset.grid(False)
                        
                        # Format tick labels
                        from matplotlib.ticker import FormatStrFormatter
                        ax_inset.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        ax_inset.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        
                        # Add colorbar if requested
                        if add_inset_colorbar:
                            from matplotlib.colors import Normalize
                            import matplotlib.cm as cm
                            
                            # Get inset position in figure coordinates
                            inset_bbox_fig = ax_inset.get_position()
                            
                            # Create colorbar axes to the right of inset
                            cbar_width_val = 0.03
                            cbar_spacing = 0.1
                            cbar_ax = fig.add_axes([
                                inset_bbox_fig.x1 + cbar_spacing,
                                inset_bbox_fig.y0,
                                cbar_width_val,
                                inset_bbox_fig.height
                            ])
                            
                            # Create colorbar mapping moieties
                            norm = Normalize(vmin=0, vmax=len(selection_names)-1)
                            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='tab10'),
                                               cax=cbar_ax, orientation='vertical')
                            cbar.set_label('Moiety', fontsize=label_fontsize*0.8, rotation=90, labelpad=8)
                            cbar.ax.tick_params(labelsize=label_fontsize*0.7)
                        
                        # Only add inset for first moiety with parameters to avoid clutter
                        break
                
                # Add legend
                if show_legend:
                    if legend_bbox is not None:
                        legend = ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox, ncol=legend_ncol, 
                                 fontsize=legend_fontsize, framealpha=legend_frame_alpha, frameon=legend_frameon,
                                 columnspacing=legend_columnspacing)
                    else:
                        legend = ax.legend(loc=legend_loc, ncol=legend_ncol, fontsize=legend_fontsize,
                                 framealpha=legend_frame_alpha, frameon=legend_frameon,
                                 columnspacing=legend_columnspacing)
                    # Set legend text weight
                    for text in legend.get_texts():
                        text.set_fontweight(legend_fontweight)
                
                # Save individual cluster figure if requested
                if save_individual_figures:
                    # Create separate figure for this cluster
                    fig_individual, ax_individual = plt.subplots(figsize=individual_figsize)
                    
                    # Add shell shading to individual figure
                    for sel_name in selection_names:
                        if sel_name in cluster_selections:
                            current_shell_boundaries = resolve_param(shell_boundaries, sel_name)
                            
                            if current_shell_boundaries is not None and len(current_shell_boundaries) > 0:
                                n_shells = len(current_shell_boundaries)
                                all_colors = get_blue_saturation_colors_from_00c5ff(n_shells)
                                shell_colors = all_colors[:-1]
                                bulk_color = all_colors[-1]
                                
                                # Add shading for each shell
                                prev_r = 0.0
                                for i, r_max in enumerate(current_shell_boundaries):
                                    color = shell_colors[i]
                                    ax_individual.axvspan(prev_r, r_max, alpha=shell_alpha, color=color, zorder=0)
                                    prev_r = r_max
                                
                                # Add bulk region (extends to figure limit)
                                bulk_start = current_shell_boundaries[-1]
                                bulk_end = xlim[1] if xlim is not None else 20
                                ax_individual.axvspan(bulk_start, bulk_end, alpha=shell_alpha, color=bulk_color, zorder=0)
                                
                                # Only show shell for first moiety with boundaries
                                break
                    
                    # Add ion pairing gradient to individual figure if configured and no shell boundaries
                    gradient_applied = False
                    for sel_name in selection_names:
                        if sel_name in cluster_selections and not gradient_applied:
                            sel_data = cluster_selections[sel_name]
                            reference_name = sel_data.get('reference', None)
                            
                            if reference_name and ion_pairing_gradient is not None:
                                pair_key = f"{sel_name}-{reference_name}"
                                gradient_config = resolve_param(ion_pairing_gradient, pair_key)
                                
                                should_apply_gradient = False
                                if gradient_config is True:
                                    should_apply_gradient = True
                                elif isinstance(gradient_config, dict):
                                    should_apply_gradient = gradient_config.get('apply', False)
                                
                                # Only apply gradient if no shell boundaries (shells take priority)
                                current_shell_boundaries = resolve_param(shell_boundaries, sel_name)
                                if should_apply_gradient and (current_shell_boundaries is None or len(current_shell_boundaries) == 0):
                                    x_min = xlim[0] if xlim is not None else 0
                                    x_max = xlim[1] if xlim is not None else 20
                                    y_min = ylim[0] if ylim is not None else 0
                                    if ylim is not None:
                                        y_max = ylim[1]
                                    else:
                                        # Auto-detect y_max from data
                                        max_rdf = max(np.max(cluster_selections[sn]['rdf']) for sn in selection_names if sn in cluster_selections)
                                        y_max = max_rdf * 1.1  # Add 10% padding
                                    
                                    gradient = np.linspace(0, 1, 256).reshape(1, -1)
                                    cmap = create_ion_pairing_gradient_cmap()
                                    extent = [x_min, x_max, y_min, y_max]
                                    ax_individual.imshow(gradient, aspect='auto', extent=extent, origin='lower',
                                                        cmap=cmap, alpha=ion_pairing_gradient_alpha, zorder=0,
                                                        interpolation='bilinear')
                                    gradient_applied = True
                    
                    # Plot each selection
                    for sel_name in selection_names:
                        if sel_name in cluster_selections:
                            sel_data = cluster_selections[sel_name]
                            color = colors.get(sel_name, f"C{len(colors)}")
                            ax_individual.plot(sel_data['r'], sel_data['rdf'],
                                   label=sel_name,
                                   color=color,
                                   linewidth=linewidth,
                                   linestyle=linestyle,
                                   alpha=alpha,
                                   zorder=2)
                    
                    # Add bulk density reference line
                    if show_bulk_line:
                        ax_individual.axhline(y=1.0, 
                                  color=bulk_line_color,
                                  linestyle=bulk_line_style,
                                  linewidth=bulk_line_width,
                                  alpha=bulk_line_alpha,
                                  zorder=0)
                    
                    # Set labels and title
                    ax_individual.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
                    ax_individual.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
                    
                    if show_title:
                        ax_individual.set_title(f'Cluster {cluster_id} ({n_frames} frames)', 
                                   fontsize=title_fontsize, fontweight=title_fontweight)
                    
                    # Set tick label sizes
                    ax_individual.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    
                    # Set axis limits
                    if xlim is not None:
                        ax_individual.set_xlim(xlim)
                    if ylim is not None:
                        ax_individual.set_ylim(ylim)
                    
                    # Add grid
                    if grid:
                        ax_individual.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
                    
                    # Add shell labels after plotting (so ylim is set correctly)
                    if show_shell_label:
                        for sel_name in selection_names:
                            current_shell_boundaries = resolve_param(shell_boundaries, sel_name)
                            if current_shell_boundaries is not None and len(current_shell_boundaries) > 0:
                                y_min, y_max = ax_individual.get_ylim()
                                label_y_pos = y_max * 0.98
                                
                                prev_r = 0.0
                                for i, r_max in enumerate(current_shell_boundaries):
                                    # Calculate label position based on alignment
                                    if shell_label_ha == 'left':
                                        label_x_pos = prev_r
                                    elif shell_label_ha == 'right':
                                        label_x_pos = r_max
                                    else:  # 'center'
                                        label_x_pos = (prev_r + r_max) / 2
                                    
                                    label_text = f'S{i+1}' if shell_label_style == 'short' else f'Shell {i+1}'
                                    ax_individual.text(label_x_pos, label_y_pos, label_text,
                                        ha=shell_label_ha, va='top', fontsize=shell_label_fontsize, 
                                        fontweight='bold', color='black')
                                    prev_r = r_max
                                
                                # Bulk region extends to figure limit
                                bulk_start = current_shell_boundaries[-1]
                                x_max = ax_individual.get_xlim()[1]
                                if shell_label_ha == 'left':
                                    bulk_label_x_pos = bulk_start
                                elif shell_label_ha == 'right':
                                    bulk_label_x_pos = x_max
                                else:  # 'center'
                                    bulk_label_x_pos = (bulk_start + x_max) / 2
                                
                                ax_individual.text(bulk_label_x_pos, label_y_pos, 'Bulk',
                                    ha=shell_label_ha, va='top', fontsize=shell_label_fontsize, 
                                    fontweight='bold', color='black')
                                
                                # Only add labels for first moiety with boundaries
                                break
                    
                    # Add inset zoom to individual figure
                    for sel_name in selection_names:
                        current_inset_xlim = resolve_param(inset_xlim, sel_name)
                        current_inset_ylim = resolve_param(inset_ylim, sel_name)
                        current_inset_bbox = resolve_param(inset_bbox, sel_name)
                        current_shell_boundaries = resolve_param(shell_boundaries, sel_name)
                        
                        if add_inset and current_inset_xlim is not None and current_inset_ylim is not None:
                            if current_inset_bbox is None:
                                ax_inset_ind = ax_individual.inset_axes([0.55, 0.55, 0.35, 0.35])
                            else:
                                xmin, xmax, ymin, ymax = current_inset_bbox
                                xlim_curr, ylim_curr = ax_individual.get_xlim(), ax_individual.get_ylim()
                                
                                left = (xmin - xlim_curr[0]) / (xlim_curr[1] - xlim_curr[0])
                                width = (xmax - xmin) / (xlim_curr[1] - xlim_curr[0])
                                bottom = (ymin - ylim_curr[0]) / (ylim_curr[1] - ylim_curr[0])
                                height = (ymax - ymin) / (ylim_curr[1] - ylim_curr[0])
                                
                                ax_inset_ind = ax_individual.inset_axes([left, bottom, width, height])
                            
                            # Add shell shading to inset
                            if current_shell_boundaries is not None and len(current_shell_boundaries) > 0:
                                inset_xmin, inset_xmax = current_inset_xlim
                                n_shells = len(current_shell_boundaries)
                                all_colors = get_blue_saturation_colors_from_00c5ff(n_shells)
                                shell_colors = all_colors[:-1]
                                bulk_color = all_colors[-1]
                                
                                # Add shading for shells that overlap with inset x-range
                                prev_r = 0.0
                                for i, r_max in enumerate(current_shell_boundaries):
                                    if r_max > inset_xmin and prev_r < inset_xmax:
                                        shade_min = max(prev_r, inset_xmin)
                                        shade_max = min(r_max, inset_xmax)
                                        ax_inset_ind.axvspan(shade_min, shade_max, alpha=shell_alpha, 
                                                        color=shell_colors[i], zorder=0)
                                    prev_r = r_max
                                
                                # Add bulk shading if it overlaps with inset
                                bulk_start = current_shell_boundaries[-1]
                                if bulk_start < inset_xmax:
                                    shade_min = max(bulk_start, inset_xmin)
                                    ax_inset_ind.axvspan(shade_min, inset_xmax, alpha=shell_alpha, 
                                                    color=bulk_color, zorder=0)
                            
                            # Add ion pairing gradient to individual figure inset if configured and no shell boundaries
                            if sel_name in cluster_selections:
                                sel_data = cluster_selections[sel_name]
                                reference_name = sel_data.get('reference', None)
                                
                                if reference_name and ion_pairing_gradient is not None:
                                    pair_key = f"{sel_name}-{reference_name}"
                                    gradient_config = resolve_param(ion_pairing_gradient, pair_key)
                                    
                                    should_apply_gradient = False
                                    if gradient_config is True:
                                        should_apply_gradient = True
                                    elif isinstance(gradient_config, dict):
                                        should_apply_gradient = gradient_config.get('apply', False)
                                    
                                    # Only apply gradient if no shell boundaries (shells take priority)
                                    if should_apply_gradient and (current_shell_boundaries is None or len(current_shell_boundaries) == 0):
                                        inset_xmin, inset_xmax = current_inset_xlim
                                        inset_ymin, inset_ymax = current_inset_ylim
                                        
                                        gradient = np.linspace(0, 1, 256).reshape(1, -1)
                                        cmap = create_ion_pairing_gradient_cmap()
                                        extent = [inset_xmin, inset_xmax, inset_ymin, inset_ymax]
                                        ax_inset_ind.imshow(gradient, aspect='auto', extent=extent, origin='lower',
                                                           cmap=cmap, alpha=ion_pairing_gradient_alpha, zorder=0,
                                                           interpolation='bilinear')
                            
                            # Set inset limits BEFORE plotting
                            ax_inset_ind.set_xlim(current_inset_xlim)
                            ax_inset_ind.set_ylim(current_inset_ylim)
                            
                            # Plot RDF curves in inset
                            for moiety in selection_names:
                                if moiety in cluster_selections:
                                    sel_data = cluster_selections[moiety]
                                    color = colors.get(moiety, f"C{len(colors)}")
                                    ax_inset_ind.plot(sel_data['r'], sel_data['rdf'], color=color, 
                                                linewidth=1.5, alpha=alpha, zorder=2, clip_on=True)
                            
                            # Inset styling
                            ax_inset_ind.tick_params(labelsize=max(6, tick_fontsize - 2))
                            ax_inset_ind.grid(False)
                            
                            # Format tick labels
                            from matplotlib.ticker import FormatStrFormatter
                            ax_inset_ind.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                            ax_inset_ind.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                            
                            # Add colorbar if requested
                            if add_inset_colorbar:
                                from matplotlib.colors import Normalize
                                import matplotlib.cm as cm
                                
                                # Get inset position in figure coordinates
                                inset_bbox_fig = ax_inset_ind.get_position()
                                
                                # Create colorbar axes to the right of inset
                                cbar_width_val = 0.03
                                cbar_spacing = 0.1
                                cbar_ax = fig_individual.add_axes([
                                    inset_bbox_fig.x1 + cbar_spacing,
                                    inset_bbox_fig.y0,
                                    cbar_width_val,
                                    inset_bbox_fig.height
                                ])
                                
                                # Create colorbar mapping moieties
                                norm = Normalize(vmin=0, vmax=len(selection_names)-1)
                                cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='tab10'),
                                                   cax=cbar_ax, orientation='vertical')
                                cbar.set_label('Moiety', fontsize=label_fontsize*0.8, rotation=90, labelpad=8)
                                cbar.ax.tick_params(labelsize=label_fontsize*0.7)
                            
                            # Only add inset for first moiety with parameters
                            break
                    
                    # Add secondary axis for coordination number if requested
                    if show_RCN:
                        ax2_ind = ax_individual.twinx()
                        
                        # Determine scaling factor for RCN values
                        rcn_scale = 1000 if RCN_notation == 'scaled' else 1
                        
                        # Plot coordination numbers
                        for cluster_id in cluster_ids:
                            if cluster_id in cluster_data and sel_name in cluster_data[cluster_id]:
                                sel_data = cluster_data[cluster_id][sel_name]
                                
                                # Always recalculate RCN when show_RCN=True to ensure correct formula
                                # This overrides potentially incorrect pre-calculated 'count' values
                                r_vals = sel_data['r']
                                rdf_vals = sel_data['rdf']
                                
                                # Calculate number density (key missing factor!)
                                rho = 0.0334  # particles/Å³ (approximate water density)
                                
                                # Calculate running coordination number: N(r) = 4π * ρ * ∫₀ʳ g(r') * r'² dr'
                                rcn_values = np.zeros_like(r_vals)
                                for i in range(1, len(r_vals)):
                                    # Calculate g(r') * r'² for proper volume weighting
                                    integrand = rdf_vals[:i+1] * (r_vals[:i+1] ** 2)
                                    # Integrate with 4π * ρ factor (ρ is the crucial missing piece!)
                                    rcn_values[i] = 4.0 * np.pi * rho * trapz(integrand, r_vals[:i+1])
                                
                                # Store corrected values back
                                sel_data['count'] = rcn_values
                                print(f"   Recalculated RCN for individual plot (moiety) {sel_name}, cluster {cluster_id}: max N(r) = {rcn_values[-1]:.2f}")
                                
                                if rcn_values is not None:
                                    color = colors.get(cluster_id, f"C{cluster_id}")
                                    ax2_ind.plot(sel_data['r'], rcn_values / rcn_scale,
                                           linestyle=RCN_curve_style,
                                           linewidth=RCN_curve_weight,
                                           color=color,
                                           alpha=alpha)
                        
                        # Set secondary y-axis label and styling (inherit from primary if not specified)
                        rcn_label_fs = RCN_label_fontsize if RCN_label_fontsize is not None else label_fontsize
                        rcn_label_fw = RCN_label_fontweight if RCN_label_fontweight is not None else label_fontweight
                        rcn_tick_fs = RCN_tick_fontsize if RCN_tick_fontsize is not None else tick_fontsize
                        
                        # Apply RCN notation formatting if requested
                        if RCN_notation == 'offset':
                            # Option 1: Offset notation (10, 20, 30... with ×10³ at top)
                            from matplotlib.ticker import ScalarFormatter
                            formatter = ScalarFormatter(useMathText=True)
                            formatter.set_powerlimits((0, 0))  # Always use scientific notation
                            ax2_ind.yaxis.set_major_formatter(formatter)
                            ax2_ind.set_ylabel(RCN_ylabel, fontsize=rcn_label_fs, fontweight=rcn_label_fw)
                            # Increase offset text size to match tick labels
                            ax2_ind.yaxis.offsetText.set_fontsize(rcn_tick_fs)
                        elif RCN_notation == 'scaled':
                            # Option 2: Manual scaling (divide by 1000, update label)
                            ax2_ind.set_ylabel(f'{RCN_ylabel} (×10³)', fontsize=rcn_label_fs, fontweight=rcn_label_fw)
                        else:
                            # Default: no special formatting
                            ax2_ind.set_ylabel(RCN_ylabel, fontsize=rcn_label_fs, fontweight=rcn_label_fw)
                        
                        ax2_ind.tick_params(axis='y', labelsize=rcn_tick_fs)
                        
                        # Smart Y-axis scaling for individual plots (same as main plot)
                        r0 = 2.8  # Approximate first shell radius for ion-water coordination
                        
                        if r0 is not None:
                            # Find CN at coordination radius from any available cluster data  
                            cn_at_r0_raw = None
                            for cluster_id in cluster_ids:
                                if cluster_id in cluster_data and sel_name in cluster_data[cluster_id]:
                                    sel_data = cluster_data[cluster_id][sel_name]
                                    if 'count' in sel_data and 'r' in sel_data:
                                        r_vals = sel_data['r']
                                        rcn_vals = sel_data['count']
                                        if len(r_vals) > 0 and len(rcn_vals) > 0:
                                            idx_r0 = np.argmin(np.abs(r_vals - r0))
                                            cn_at_r0_raw = rcn_vals[idx_r0]  # Keep raw value for scaling calculation
                                            break
                            
                            if cn_at_r0_raw is not None and cn_at_r0_raw > 0:
                                # Calculate displayed value (accounting for rcn_scale)
                                cn_at_r0_displayed = cn_at_r0_raw / rcn_scale
                                
                                # Scale so CN at r₀ appears at 1/rcn_scale_factor of figure height (in displayed units)
                                y_max_rcn_displayed = cn_at_r0_displayed * rcn_scale_factor
                                ax2_ind.set_ylim(0, y_max_rcn_displayed)
                            else:
                                # Fallback to simple scaling
                                ax2_ind.set_ylim(0, None)
                        else:
                            ax2_ind.set_ylim(0, None)  # Original fallback
                    
                    # Add legend
                    if show_legend:
                        if legend_bbox is not None:
                            legend_ind = ax_individual.legend(loc=legend_loc, bbox_to_anchor=legend_bbox, 
                                     ncol=legend_ncol, fontsize=legend_fontsize,
                                     framealpha=legend_frame_alpha, frameon=legend_frameon,
                                     columnspacing=legend_columnspacing)
                        else:
                            legend_ind = ax_individual.legend(loc=legend_loc, ncol=legend_ncol, fontsize=legend_fontsize,
                                     framealpha=legend_frame_alpha, frameon=legend_frameon,
                                     columnspacing=legend_columnspacing)
                        for text in legend_ind.get_texts():
                            text.set_fontweight(legend_fontweight)
                    
                    plt.tight_layout()
                    
                    # Determine save directory
                    if individual_save_dir is not None:
                        save_dir = individual_save_dir
                    else:
                        # Use same directory as the combined figure
                        import os
                        save_dir = os.path.dirname(filename) or '.'
                    
                    # Create directory if it doesn't exist
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Generate individual filename
                    base_name = os.path.splitext(os.path.basename(filename))[0]
                    individual_filename = os.path.join(save_dir, f"{base_name}_cluster_{cluster_id}.png")
                    
                    # Save individual figure
                    fig_individual.savefig(individual_filename, dpi=dpi, bbox_inches=bbox_inches)
                    print(f"  ✓ Saved: {individual_filename}")
                    plt.close(fig_individual)
        
        # Hide extra subplots if any
        for idx in range(n_panels, len(axes)):
            axes[idx].set_visible(False)
        
        # Add overall title if requested
        if show_title and title != 'Radial Distribution Function':
            fig.suptitle(title, fontsize=title_fontsize + 2, fontweight=title_fontweight)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            print(f"✓ Saved combined figure: {filename} (DPI={dpi})")
        
        # Print summary if individual figures were saved
        if save_individual_figures:
            if cluster_treatment == 'group':
                print(f"✓ Saved {len(selection_names)} individual moiety figures (DPI={dpi})")
            else:
                print(f"✓ Saved {n_clusters} individual cluster figures (DPI={dpi})")
        
        return fig

    def export_rdf_data_for_multi_system(self,
                                          rdf_data: Union[Dict, List[Dict]],
                                          system_name: str) -> Dict:
        """
        Export RDF data in the format required by ``plot_multi_system_rdfs()``.

        Call this on each system's plotter instance to package its RDF data into
        a portable dict; then combine the results into a single ``systems_data``
        dict and pass it to ``plot_multi_system_rdfs()``.

        Parameters
        ----------
        rdf_data : dict
            RDF data produced by ``compute_rdf()``.  Accepts all formats that
            ``plot_multiple_rdfs()`` understands:

            1. Batch 3-level: ``{sel1: {sel2: {cluster_id: {'r', 'rdf', ...}}}}``
            2. Per-selection: ``{sel_name: {cluster_id: {'r', 'rdf', ...}}}``
            3. Single-cluster: ``{name: {'r', 'rdf', ...}}`` (treated as cluster 0)
            4. Cluster dict:   ``{cluster_id: {'r', 'rdf', ...}}`` (single selection)

        system_name : str
            Label for this system (e.g. ``'CIP+'``, ``'CIP-'``).

        Returns
        -------
        dict
            ``{'system_name': str,
               'cluster_ids': list,
               'selection_names': list,
               'rdf_by_cluster': {cluster_id: {sel_name: {'r', 'rdf', 'n_frames', 'count', 'reference'}}}}``

        Examples
        --------
        >>> data_plus  = plotter_plus.export_rdf_data_for_multi_system(rdf_plus,  'CIP+')
        >>> data_minus = plotter_minus.export_rdf_data_for_multi_system(rdf_minus, 'CIP-')
        >>> systems_data = {'CIP+': data_plus, 'CIP-': data_minus}
        >>> fig = any_plotter.plot_multi_system_rdfs(systems_data, sel_names=['quinolone', 'piperazine'])
        """
        cluster_data: Dict = {}
        selection_names: List[str] = []

        if isinstance(rdf_data, list):
            # List: [{'r', 'rdf', 'label'}, ...]  → single cluster 0
            cluster_data[0] = {}
            for i, item in enumerate(rdf_data):
                name = item.get('name', item.get('label', f'RDF {i+1}'))
                cluster_data[0][name] = {
                    'r': item['r'], 'rdf': item['rdf'],
                    'n_frames': item.get('n_frames', 'N/A'),
                    'count': item.get('count', None),
                }
                if name not in selection_names:
                    selection_names.append(name)

        elif isinstance(rdf_data, dict):
            first_key = next(iter(rdf_data))
            first_val = rdf_data[first_key]

            # ── Case 3 (single-cluster): {name: {'r', 'rdf'}} ──
            if isinstance(first_val, dict) and 'r' in first_val:
                cluster_data[0] = {}
                for name, data in rdf_data.items():
                    cluster_data[0][name] = {
                        'r': data['r'], 'rdf': data['rdf'],
                        'n_frames': data.get('n_frames', 'N/A'),
                        'count': data.get('count', None),
                    }
                    if name not in selection_names:
                        selection_names.append(name)

            elif isinstance(first_val, dict):
                inner_key = next(iter(first_val))
                inner_val = first_val[inner_key]

                # ── Case 4 (cluster dict): {cluster_id: {'r', 'rdf'}} ──
                if isinstance(inner_val, dict) and 'r' in inner_val:
                    sel = 'data'
                    selection_names = [sel]
                    for cid, crdf in rdf_data.items():
                        cluster_data[cid] = {sel: {
                            'r': crdf['r'], 'rdf': crdf['rdf'],
                            'n_frames': crdf.get('n_frames', 'N/A'),
                            'count': crdf.get('count', None),
                        }}

                elif isinstance(inner_val, dict):
                    innermost_key = next(iter(inner_val))
                    innermost_val = inner_val[innermost_key]

                    # ── Case 1 (batch 3-level): {sel1: {sel2: {cid: data}}} ──
                    if isinstance(innermost_val, dict) and 'r' in innermost_val:
                        for sel1, sel1_data in rdf_data.items():
                            if sel1 not in selection_names:
                                selection_names.append(sel1)
                            for sel2, sel2_data in sel1_data.items():
                                for cid, crdf in sel2_data.items():
                                    if cid not in cluster_data:
                                        cluster_data[cid] = {}
                                    cluster_data[cid][sel1] = {
                                        'r': crdf['r'], 'rdf': crdf['rdf'],
                                        'n_frames': crdf.get('n_frames', 'N/A'),
                                        'count': crdf.get('count', None),
                                        'reference': sel2,
                                    }
                    # ── Case 2 (per-selection): {sel_name: {cid: data}} ──
                    else:
                        for sel, sel_data in rdf_data.items():
                            if sel not in selection_names:
                                selection_names.append(sel)
                            for cid, crdf in sel_data.items():
                                if cid not in cluster_data:
                                    cluster_data[cid] = {}
                                cluster_data[cid][sel] = {
                                    'r': crdf['r'], 'rdf': crdf['rdf'],
                                    'n_frames': crdf.get('n_frames', 'N/A'),
                                    'count': crdf.get('count', None),
                                }
        else:
            raise ValueError("rdf_data must be a dict or list.")

        selection_names = list(dict.fromkeys(selection_names))
        cluster_ids = sorted(cluster_data.keys())
        return {
            'system_name': system_name,
            'cluster_ids': cluster_ids,
            'selection_names': selection_names,
            'rdf_by_cluster': cluster_data,
        }

    def plot_multi_system_rdfs(self,
                                systems_data: Dict,
                                sel_names: Optional[Union[str, List[str]]] = None,
                                sel_name_labels: Optional[Dict[str, str]] = None,
                                system_display_names: Optional[Dict[str, str]] = None,
                                colors_per_system: Optional[Dict[str, List[str]]] = None,
                                linestyles_per_cluster: Optional[List[str]] = None,
                                # Figure
                                figsize: Tuple[float, float] = (8, 6),
                                panel_spacing: float = 0.05,
                                # Axes limits
                                xlim: Optional[Tuple[float, float]] = None,
                                ylim: Optional[Tuple[float, float]] = None,
                                shared_y: bool = True,
                                # Panel titles
                                show_panel_titles: bool = True,
                                title_fontsize: int = 14,
                                title_fontweight: str = 'bold',
                                # Axis labels
                                xlabel: str = 'r (Å)',
                                ylabel: str = 'g(r)',
                                label_fontsize: int = 12,
                                label_fontweight: str = 'bold',
                                tick_fontsize: int = 10,
                                # Legend
                                show_legend: bool = True,
                                legend_fontsize: int = 10,
                                legend_fontweight: str = 'normal',
                                legend_title: Optional[str] = None,
                                legend_title_fontsize: int = 11,
                                legend_title_fontweight: str = 'bold',
                                legend_loc: str = 'best',
                                legend_bbox: Optional[Tuple[float, float]] = None,
                                legend_ncol: int = 1,
                                legend_columnspacing: float = 2.0,
                                legend_handletextpad: float = 0.8,
                                legend_frame_alpha: float = 0.9,
                                legend_frameon: bool = True,
                                # Line styling
                                linewidth: float = 2,
                                alpha: float = 1.0,
                                # Bulk density reference line
                                show_bulk_line: bool = True,
                                bulk_line_color: str = 'gray',
                                bulk_line_style: str = '--',
                                bulk_line_width: float = 1,
                                bulk_line_alpha: float = 0.5,
                                # Grid
                                grid: bool = True,
                                grid_alpha: float = 0.3,
                                grid_linestyle: str = '--',
                                # Shell shading
                                shell_boundaries: Optional[Union[List[float], Dict[str, List[float]]]] = None,
                                shell_alpha: float = 0.15,
                                shell_label_fontsize: int = 10,
                                shell_label_style: str = 'complete',
                                shell_label_ha: str = 'center',
                                show_shell_label: bool = True,
                                # Save
                                save_fig: bool = False,
                                filename: str = 'multi_system_rdf.png',
                                dpi: int = 300,
                                bbox_inches: str = 'tight') -> plt.Figure:
        """
        Compare RDF curves across multiple systems in a publication-ready figure.

        Creates **one panel per moiety** (selection), each panel overlaying RDF
        curves from every system / cluster.  Designed as a multi-system companion
        to ``plot_multiple_rdfs()``.

        Parameters
        ----------
        systems_data : dict
            Built with ``export_rdf_data_for_multi_system()``::

                {system_name: {'system_name', 'cluster_ids',
                               'selection_names', 'rdf_by_cluster'}}

        sel_names : str or list of str, optional
            Which moiety/selection names to show (one panel each).
            If ``None``, uses all selection names found across all systems
            (intersection to avoid missing data).

        sel_name_labels : dict, optional
            Human-readable panel titles keyed by sel_name.
            Example: ``{'quinolone': 'Quinolone ring', 'piperazine': 'Piperazine'}``

        system_display_names : dict, optional
            Display names (supports LaTeX) for legend & titles.
            Example: ``{'CIP+': r'$CIP^+$', 'CIP-': r'$CIP^-$'}``

        colors_per_system : dict, optional
            ``{system_name: [color_cluster0, color_cluster1, ...]}`` —
            one color per cluster within that system.
            If ``None``, auto-assigns colors cycling the ``tab10`` palette
            across (system, cluster) pairs.

        linestyles_per_cluster : list of str, optional
            Cycle of line styles applied by cluster *index within each system*.
            Default: ``['-', '--', '-.', ':']``.
            Example: ``['-', '--']`` → first cluster solid, second dashed.

        figsize : tuple
            Figure size (width × height) in inches.  Width is scaled by the
            number of panels if a single value pair is given.

        shared_y : bool, default=True
            If ``True``, all panels share the same y-axis limits (set to the
            global data maximum).

        show_panel_titles : bool, default=True
            Whether to add a title above each panel (the moiety name).

        legend_bbox : tuple (x, y), optional
            Position the legend manually via ``bbox_to_anchor``.

        shell_boundaries : list or dict, optional
            Shell boundary radii for shaded regions.  Same semantics as
            ``plot_multiple_rdfs()``: a list applies to all panels; a dict
            ``{sel_name: [r1, r2, ...]}`` applies boundaries per moiety.

        Returns
        -------
        fig : matplotlib.figure.Figure

        Examples
        --------
        >>> data_plus  = plotter_plus.export_rdf_data_for_multi_system(rdf_plus,  'CIP+')
        >>> data_minus = plotter_minus.export_rdf_data_for_multi_system(rdf_minus, 'CIP-')
        >>> systems_data = {'CIP+': data_plus, 'CIP-': data_minus}
        >>>
        >>> fig = any_plotter.plot_multi_system_rdfs(
        ...     systems_data,
        ...     sel_names=['quinolone', 'piperazine'],
        ...     system_display_names={'CIP+': r'$CIP^+$', 'CIP-': r'$CIP^-$'},
        ...     colors_per_system={'CIP+': ['#E74C3C', '#C0392B'],   # cluster 0, 1
        ...                        'CIP-': ['#3498DB', '#2980B9']},
        ...     xlim=(0, 8),
        ...     label_fontsize=14,
        ...     show_legend=True,
        ...     legend_ncol=2,
        ...     save_fig=True,
        ...     filename='multi_system_rdf.png'
        ... )
        """
        import matplotlib.colors as mcolors

        if not systems_data:
            raise ValueError("systems_data cannot be empty.")

        system_names = list(systems_data.keys())

        # ── resolve sel_names ──────────────────────────────────────────────
        if sel_names is None:
            # intersection of selection_names across all systems
            all_sels = [set(systems_data[sn]['selection_names']) for sn in system_names]
            common = all_sels[0]
            for s in all_sels[1:]:
                common = common & s
            # preserve order from first system
            sel_names_list = [s for s in systems_data[system_names[0]]['selection_names']
                              if s in common]
        elif isinstance(sel_names, str):
            sel_names_list = [sel_names]
        else:
            sel_names_list = list(sel_names)

        n_panels = len(sel_names_list)
        if n_panels == 0:
            raise ValueError("No common selection names found across systems.")

        # ── default linestyles ─────────────────────────────────────────────
        if linestyles_per_cluster is None:
            linestyles_per_cluster = ['-', '--', '-.', ':']

        # ── build (system, cluster_id) pairs & auto-assign colors ──────────
        pairs: List[tuple] = []  # [(sys_name, cid), ...]
        for sn in system_names:
            for cid in systems_data[sn]['cluster_ids']:
                pairs.append((sn, cid))

        if colors_per_system is None:
            cmap10 = plt.cm.get_cmap('tab10')
            auto_colors: Dict = {}
            for i, sn in enumerate(system_names):
                n_c = len(systems_data[sn]['cluster_ids'])
                base_color = cmap10(i % 10)
                # If only 1 cluster, use the base color; if multiple, shade variants
                auto_colors[sn] = []
                for j in range(n_c):
                    if n_c == 1:
                        auto_colors[sn].append(base_color)
                    else:
                        # lighten progressively
                        r, g, b, _a = base_color
                        fac = 0.3 * j / max(1, n_c - 1)
                        auto_colors[sn].append((min(1, r + fac), min(1, g + fac), min(1, b + fac), _a))
            colors_per_system = auto_colors

        # ── shell color helper (reuse from plot_multiple_rdfs) ─────────────
        def _shell_colors(n_shells):
            base_rgb = mcolors.hex2color('#00c5ff')
            base_hsv = mcolors.rgb_to_hsv(base_rgb)
            h, s, v = base_hsv
            if n_shells == 1:
                saturations = [s, 0.2]
            elif n_shells == 2:
                saturations = [s, 0.6, 0.2]
            elif n_shells == 3:
                saturations = [s, 0.7, 0.4, 0.2]
            else:
                step = (s - 0.2) / n_shells
                saturations = [s - i * step for i in range(n_shells)] + [0.2]
            return [mcolors.to_hex(mcolors.hsv_to_rgb((h, sat, v))) for sat in saturations]

        def _resolve_shell(param, sel_name):
            if isinstance(param, dict):
                return param.get(sel_name, None)
            return param

        # ── figure ─────────────────────────────────────────────────────────
        fig_w = figsize[0] * n_panels if n_panels > 1 else figsize[0]
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(fig_w, figsize[1]),
                                 squeeze=False,
                                 gridspec_kw={'wspace': panel_spacing})
        axes = axes.flatten()

        all_y_maxes: List[float] = []

        for pidx, sel_name in enumerate(sel_names_list):
            ax = axes[pidx]

            # Shell shading
            current_shell_bnd = _resolve_shell(shell_boundaries, sel_name)
            shell_color_list: List = []
            bulk_color_val = None
            if current_shell_bnd is not None and len(current_shell_bnd) > 0:
                n_sh = len(current_shell_bnd)
                all_sh_colors = _shell_colors(n_sh)
                shell_color_list = all_sh_colors[:-1]
                bulk_color_val = all_sh_colors[-1]
                prev_r = 0.0
                for i, r_max in enumerate(current_shell_bnd):
                    ax.axvspan(prev_r, r_max, alpha=shell_alpha,
                               color=shell_color_list[i], zorder=0)
                    prev_r = r_max
                x_end = xlim[1] if xlim is not None else 20
                ax.axvspan(current_shell_bnd[-1], x_end,
                           alpha=shell_alpha, color=bulk_color_val, zorder=0)

            # Plot RDF lines for every (system, cluster) pair
            for sn in system_names:
                sys_entry = systems_data[sn]
                rdf_by_cluster = sys_entry['rdf_by_cluster']
                cluster_ids_sys = sys_entry['cluster_ids']
                disp_name = (system_display_names or {}).get(sn, sn)
                sys_colors = colors_per_system.get(sn, [])

                for cidx, cid in enumerate(cluster_ids_sys):
                    if sel_name not in rdf_by_cluster.get(cid, {}):
                        continue
                    sel_entry = rdf_by_cluster[cid][sel_name]
                    color = sys_colors[cidx] if cidx < len(sys_colors) else f'C{cidx}'
                    ls = linestyles_per_cluster[cidx % len(linestyles_per_cluster)]

                    if len(cluster_ids_sys) == 1:
                        label = disp_name
                    else:
                        label = f'{disp_name} C{cid}'

                    ax.plot(sel_entry['r'], sel_entry['rdf'],
                            label=label, color=color,
                            linewidth=linewidth, linestyle=ls,
                            alpha=alpha, zorder=2)

                    y_max_data = float(np.max(sel_entry['rdf']))
                    all_y_maxes.append(y_max_data)

            # Bulk reference line
            if show_bulk_line:
                ax.axhline(y=1.0, color=bulk_line_color,
                           linestyle=bulk_line_style,
                           linewidth=bulk_line_width,
                           alpha=bulk_line_alpha, zorder=0)

            # Axis limits
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            # Labels
            ax.set_xlabel(xlabel, fontsize=label_fontsize,
                          fontweight=label_fontweight)
            if pidx == 0:
                ax.set_ylabel(ylabel, fontsize=label_fontsize,
                              fontweight=label_fontweight)
            else:
                ax.set_ylabel('')
                if shared_y:
                    ax.tick_params(labelleft=False)

            if show_panel_titles:
                panel_title = (sel_name_labels or {}).get(sel_name, sel_name)
                ax.set_title(panel_title, fontsize=title_fontsize,
                             fontweight=title_fontweight)

            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

            if grid:
                ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)

            # Shell labels
            if show_shell_label and current_shell_bnd is not None and len(current_shell_bnd) > 0:
                y_lo, y_hi = ax.get_ylim()
                label_y = y_hi * 0.98
                prev_r = 0.0
                for i, r_max in enumerate(current_shell_bnd):
                    if shell_label_ha == 'left':
                        lx = prev_r
                    elif shell_label_ha == 'right':
                        lx = r_max
                    else:
                        lx = (prev_r + r_max) / 2
                    ltxt = f'S{i+1}' if shell_label_style == 'short' else f'Shell {i+1}'
                    ax.text(lx, label_y, ltxt, ha=shell_label_ha, va='top',
                            fontsize=shell_label_fontsize, fontweight='bold', color='black')
                    prev_r = r_max
                bk_start = current_shell_bnd[-1]
                x_hi = ax.get_xlim()[1]
                if shell_label_ha == 'left':
                    bx = bk_start
                elif shell_label_ha == 'right':
                    bx = x_hi
                else:
                    bx = (bk_start + x_hi) / 2
                ax.text(bx, label_y, 'Bulk', ha=shell_label_ha, va='top',
                        fontsize=shell_label_fontsize, fontweight='bold', color='black')

            # Legend (drawn on all panels, or only first — on first by default)
            if show_legend and pidx == 0:
                leg_kw = dict(
                    fontsize=legend_fontsize, framealpha=legend_frame_alpha,
                    frameon=legend_frameon, ncol=legend_ncol,
                    columnspacing=legend_columnspacing,
                    handletextpad=legend_handletextpad,
                )
                if legend_bbox is not None:
                    leg_kw['bbox_to_anchor'] = legend_bbox
                    leg_kw['loc'] = 'upper left'
                else:
                    leg_kw['loc'] = legend_loc
                if legend_title:
                    leg_kw['title'] = legend_title
                leg = ax.legend(**leg_kw)
                for _t in leg.get_texts():
                    _t.set_fontweight(legend_fontweight)
                if legend_title:
                    leg.get_title().set_fontsize(legend_title_fontsize)
                    leg.get_title().set_fontweight(legend_title_fontweight)

        # Shared y limit
        if shared_y and ylim is None and all_y_maxes:
            global_ymax = max(all_y_maxes) * 1.05
            for ax in axes[:n_panels]:
                ax.set_ylim(bottom=0, top=global_ymax)

        plt.tight_layout()

        if save_fig:
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            print(f'✓ Saved: {filename}')

        return fig

    def plot_stacked_rdfs(self,
                         rdf_data: Union[Dict, List[Dict]],
                         vertical_offset: float = 2.0,
                         stack_by: str = 'auto',
                         figsize: Tuple[float, float] = (8, 10),
                         xlim: Optional[Tuple[float, float]] = None,
                         ylim: Optional[Tuple[float, float]] = None,
                         show_title: bool = True,
                         title: str = 'Stacked RDF Comparison',
                         title_fontsize: int = 14,
                         title_fontweight: str = 'bold',
                         xlabel: str = 'r (Å)',
                         ylabel: str = 'g(r)',
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
                         colors: Optional[Dict] = None,
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
                         ion_pairing_gradient: Optional[Union[bool, Dict[str, Union[bool, Dict]]]] = None,
                         ion_pairing_gradient_alpha: float = 0.25,
                         show_stack_labels: bool = True,
                         stack_label_fontsize: int = 11,
                         stack_label_fontweight: str = 'bold',
                         stack_label_ha: str = 'right',
                         stack_label_x_position: float = 0.98,
                         stack_label_y_position: str = 'center',
                         stack_label_y_offset: float = 0.0,
                         save_fig: bool = False,
                         filename: str = 'rdf_stacked.png',
                         dpi: int = 300,
                         bbox_inches: str = 'tight') -> plt.Figure:
        """
        Create XRD-style stacked RDF plots with vertical offset.
        
        Supports two stacking modes:
        1. Stack by moiety: Each moiety (quinolone, piperazine, etc.) gets a vertical band
           with all clusters shown together
        2. Stack by cluster: Each cluster gets a vertical band
        
        Parameters
        ----------
        rdf_data : dict or list of dict
            RDF data from analyzer.
            - For stacking by moiety: {moiety: {ref: {cluster_id: {r:..., rdf:...}}}}
            - For stacking by cluster: {cluster_id: {r:..., rdf:...}}
        
        stack_by : str, default='auto'
            Stacking mode: 'auto' (detect), 'moiety', or 'cluster'
        
        vertical_offset : float, default=2.0
            Vertical spacing between stacked groups
        
        colors : dict, optional
            Color mapping by cluster ID: {cluster_id: color}.
            Example: {0: 'red', 1: 'blue', 2: 'green'}
            When stacking by moiety, all C0 will be red, all C1 blue, etc.
        
        ion_pairing_gradient : bool or dict, optional
            Apply gradient background shading (lightcoral → lightyellow → lightgreen → lightblue).
            For moiety stacking, provide dict: {'moiety-reference': True}
        
        ion_pairing_gradient_alpha : float, default=0.25
            Transparency of gradient shading (0=transparent, 1=opaque)
        
        show_stack_labels : bool, default=True
            Show labels on right side (moiety names or cluster IDs)
        
        stack_label_y_position : str, default='center'
            Vertical position of stack labels: 'bottom', 'center', or 'top'
        
        stack_label_y_offset : float, default=0.0
            Additional vertical offset (in g(r) units) to adjust label position.
            Positive values move labels up, negative values move them down.
        
        grid : bool, default=True
            Show grid lines
        
        Other styling parameters: Same as plot_multiple_rdfs()
        
        Returns
        -------
        matplotlib.figure.Figure
        
        Example
        -------
        >>> # Stack by moiety (all moieties in one plot)
        >>> # Colors map to cluster IDs, not moieties
        >>> fig = plotter.plot_stacked_rdfs(
        ...     rdf_surface_o,  # Full nested structure
        ...     stack_by='moiety',
        ...     vertical_offset=3.0,
        ...     colors={0: 'red', 1: 'blue', 2: 'green'},
        ...     ion_pairing_gradient={'quinolone-surface_o': True},
        ...     ion_pairing_gradient_alpha=0.2
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
        
        # Parse RDF data structure and determine stacking mode
        moiety_data = {}  # {moiety_name: {cluster_id: {r, rdf, ...}}}
        reference_name = None
        actual_stack_by = stack_by
        
        if isinstance(rdf_data, dict) and rdf_data:
            # Check for nested format: {moiety: {reference: {cluster_id: data}}}
            moiety_keys = list(rdf_data.keys())
            first_moiety = rdf_data[moiety_keys[0]]
            
            if isinstance(first_moiety, dict):
                # Check if this looks like {reference: {cluster_id: data}}
                ref_keys = list(first_moiety.keys())
                if ref_keys:
                    first_ref_data = first_moiety[ref_keys[0]]
                    if isinstance(first_ref_data, dict):
                        # Check if looks like {cluster_id: {r, rdf}}
                        test_keys = list(first_ref_data.keys())
                        if test_keys:
                            test_cluster_data = first_ref_data[test_keys[0]]
                            if isinstance(test_cluster_data, dict) and 'r' in test_cluster_data and 'rdf' in test_cluster_data:
                                # This is nested format!
                                # Extract reference name (should be same for all moieties)
                                reference_name = ref_keys[0]
                                
                                # Extract all moieties
                                for moiety_name in moiety_keys:
                                    moiety_dict = rdf_data[moiety_name]
                                    if reference_name in moiety_dict:
                                        moiety_data[moiety_name] = moiety_dict[reference_name]
                                
                                # Auto-detect or validate stack_by
                                if actual_stack_by == 'auto':
                                    actual_stack_by = 'moiety' if len(moiety_keys) > 1 else 'cluster'
                                elif actual_stack_by == 'cluster' and len(moiety_keys) > 1:
                                    raise ValueError(
                                        f"stack_by='cluster' but received multiple moieties: {moiety_keys}.\n"
                                        f"Use stack_by='moiety' or select one moiety: rdf_data['quinolone']['surface_o']"
                                    )
            
            # If moiety_data is still empty, try flat cluster format
            if not moiety_data:
                # Check if this is {cluster_id: {r, rdf, ...}}
                first_val = first_moiety
                if isinstance(first_val, dict) and 'r' in first_val and 'rdf' in first_val:
                    # Flat cluster format
                    moiety_data = {'default': rdf_data}
                    actual_stack_by = 'cluster'
        
        if not moiety_data:
            raise ValueError(
                "No valid RDF data found. Expected format:\n"
                "  - Nested (moiety stacking): {moiety: {ref: {cluster_id: {r:..., rdf:...}}}}\n"
                "  - Flat (cluster stacking): {cluster_id: {r:..., rdf:...}}\n"
                f"Received type: {type(rdf_data)}"
            )
        
        # Organize data based on stacking mode
        if actual_stack_by == 'moiety':
            # Stack moieties: Each moiety is one vertical band
            stack_keys = sorted(moiety_data.keys())
            stack_label_prefix = ""
        else:
            # Stack clusters: Extract clusters from single moiety
            single_moiety = list(moiety_data.keys())[0]
            cluster_dict = moiety_data[single_moiety]
            stack_keys = sorted(cluster_dict.keys())
            moiety_data = {single_moiety: cluster_dict}
            stack_label_prefix = "C"
        
        # Set default colors (always by cluster ID, not by stack key)
        if colors is None:
            # Get all unique cluster IDs across all moieties
            all_cluster_ids = set()
            for moiety_name, cluster_dict in moiety_data.items():
                all_cluster_ids.update(cluster_dict.keys())
            colors = {cid: f"C{cid}" for cid in sorted(all_cluster_ids)}
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate y-extent for gradient/shells
        max_rdf_value = 0
        for i, stack_key in enumerate(stack_keys):
            offset = i * vertical_offset
            if actual_stack_by == 'moiety':
                # Get max across all clusters for this moiety
                cluster_dict = moiety_data[stack_key]
                for cluster_id, data in cluster_dict.items():
                    max_rdf_value = max(max_rdf_value, np.max(data['rdf']) + offset)
            else:
                # Single moiety, multiple clusters
                moiety_name = list(moiety_data.keys())[0]
                data = moiety_data[moiety_name][stack_key]
                max_rdf_value = max(max_rdf_value, np.max(data['rdf']) + offset)
        
        # Calculate axis limits
        x_min = xlim[0] if xlim is not None else 0
        x_max = xlim[1] if xlim is not None else 20
        if ylim is not None:
            y_min, y_max = ylim
        else:
            y_min = 0
            # No padding for stacked plots - end exactly where data ends
            y_max = max_rdf_value
        
        # Apply gradient or shell shading
        # For moiety stacking, apply single gradient across all stacks
        if actual_stack_by == 'moiety':
            # Check if ANY moiety has gradient enabled
            should_apply_gradient = False
            if ion_pairing_gradient is not None:
                for moiety_name in stack_keys:
                    pair_key = f"{moiety_name}-{reference_name}" if reference_name else None
                    if pair_key:
                        gradient_config = resolve_param(ion_pairing_gradient, pair_key)
                        if gradient_config is True:
                            should_apply_gradient = True
                            break
            
            # Apply single gradient covering all stacked moieties
            if should_apply_gradient:
                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                cmap = create_ion_pairing_gradient_cmap()
                extent = [x_min, x_max, y_min, y_max]
                ax.imshow(gradient, aspect='auto', extent=extent, origin='lower',
                         cmap=cmap, alpha=ion_pairing_gradient_alpha, zorder=0,
                         interpolation='bilinear')
        else:
            # Cluster stacking: single gradient for all
            moiety_name = list(moiety_data.keys())[0]
            pair_key = f"{moiety_name}-{reference_name}" if reference_name else None
            
            should_apply_gradient = False
            if ion_pairing_gradient is not None and pair_key:
                gradient_config = resolve_param(ion_pairing_gradient, pair_key)
                if gradient_config is True:
                    should_apply_gradient = True
            
            if should_apply_gradient:
                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                cmap = create_ion_pairing_gradient_cmap()
                extent = [x_min, x_max, y_min, y_max]
                ax.imshow(gradient, aspect='auto', extent=extent, origin='lower',
                         cmap=cmap, alpha=ion_pairing_gradient_alpha, zorder=0,
                         interpolation='bilinear')
        
        # Plot stacked RDFs
        if actual_stack_by == 'moiety':
            # Stack by moiety: Each moiety gets one vertical band
            for i, moiety_name in enumerate(stack_keys):
                offset = i * vertical_offset
                cluster_dict = moiety_data[moiety_name]
                
                # Plot all clusters for this moiety at the same offset
                for cluster_id in sorted(cluster_dict.keys()):
                    data = cluster_dict[cluster_id]
                    r = data['r']
                    g_r = data['rdf']
                    n_frames = data.get('n_frames', 'N/A')
                    
                    # Use cluster color, not moiety color
                    color = colors.get(cluster_id, f"C{cluster_id}")
                    label = f"{moiety_name} - C{cluster_id} ({n_frames} frames)"
                    
                    ax.plot(r, g_r + offset, color=color, linewidth=linewidth,
                           linestyle=linestyle, alpha=alpha, label=label, zorder=2)
                
                # Add bulk reference line if requested
                if show_bulk_line:
                    ax.hlines(1.0 + offset, x_min, x_max,
                             colors=bulk_line_color, linestyles=bulk_line_style,
                             linewidth=bulk_line_width, alpha=bulk_line_alpha, zorder=1)
                
                # Add moiety label on right side
                if show_stack_labels:
                    # Calculate y position based on stack_label_y_position
                    cluster_dict = moiety_data[moiety_name]
                    if stack_label_y_position == 'bottom':
                        y_label = offset + 0.1  # Small padding from baseline
                    elif stack_label_y_position == 'top':
                        max_rdf = max(np.max(cluster_dict[cid]['rdf']) for cid in cluster_dict.keys())
                        y_label = offset + max_rdf - 0.1  # Small padding from top
                    else:  # 'center'
                        avg_rdf = np.mean([np.mean(cluster_dict[cid]['rdf']) 
                                          for cid in cluster_dict.keys()])
                        y_label = offset + avg_rdf
                    
                    # Apply user-defined offset
                    y_label += stack_label_y_offset
                    
                    ax.text(stack_label_x_position, y_label, moiety_name,
                           transform=ax.get_yaxis_transform(),
                           fontsize=stack_label_fontsize,
                           fontweight=stack_label_fontweight,
                           ha=stack_label_ha, va='center')
        else:
            # Stack by cluster: Each cluster gets one vertical band
            moiety_name = list(moiety_data.keys())[0]
            cluster_dict = moiety_data[moiety_name]
            
            for i, cluster_id in enumerate(stack_keys):
                offset = i * vertical_offset
                data = cluster_dict[cluster_id]
                r = data['r']
                g_r = data['rdf']
                n_frames = data.get('n_frames', 'N/A')
                
                color = colors.get(cluster_id, f"C{cluster_id}")
                label = f"Cluster {cluster_id} ({n_frames} frames)"
                
                ax.plot(r, g_r + offset, color=color, linewidth=linewidth,
                       linestyle=linestyle, alpha=alpha, label=label, zorder=2)
                
                # Add bulk reference line if requested
                if show_bulk_line:
                    ax.hlines(1.0 + offset, x_min, x_max,
                             colors=bulk_line_color, linestyles=bulk_line_style,
                             linewidth=bulk_line_width, alpha=bulk_line_alpha, zorder=1)
                
                # Add cluster label on right side
                if show_stack_labels:
                    # Calculate y position based on stack_label_y_position
                    if stack_label_y_position == 'bottom':
                        y_label = offset + 0.1
                    elif stack_label_y_position == 'top':
                        y_label = offset + np.max(g_r) - 0.1
                    else:  # 'center'
                        y_label = offset + np.mean(g_r)
                    
                    # Apply user-defined offset
                    y_label += stack_label_y_offset
                    
                    ax.text(stack_label_x_position, y_label, f"C{cluster_id}",
                           transform=ax.get_yaxis_transform(),
                           fontsize=stack_label_fontsize,
                           fontweight=stack_label_fontweight,
                           ha=stack_label_ha, va='center')
        
        # Set labels and title
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        if show_title:
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        # Set tick sizes
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Custom y-ticks for stacked mode: show actual g(r) values for each band
        if actual_stack_by == 'moiety':
            # Create custom y-ticks showing actual g(r) for each moiety band
            yticks = []
            ytick_labels = []
            
            for i, stack_key in enumerate(stack_keys):
                offset = i * vertical_offset
                
                # Get max g(r) for this band to determine tick spacing
                if actual_stack_by == 'moiety':
                    cluster_dict = moiety_data[stack_key]
                    band_max = max(np.max(cluster_dict[cid]['rdf']) for cid in cluster_dict.keys())
                
                # Create ticks at 0, 1, 2, 3... for this band
                # Adjust number of ticks based on band height
                n_ticks = min(4, int(band_max) + 1)  # Show 0, 1, 2, 3 or fewer if band is small
                
                for tick_val in range(n_ticks):
                    if offset + tick_val <= y_max:
                        yticks.append(offset + tick_val)
                        ytick_labels.append(str(tick_val))
            
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels)
        
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
    
    @staticmethod
    def batch_visualize_basic(analyzer_pairs: Dict[str, 'RMSDClusterAnalyzer'],
                              base_output_dir: str = 'plots',
                              figsize: Tuple[float, float] = (8, 6),
                              dpi: int = 300,
                              cmap: str = 'jet',
                              msize: float = 10,
                              alpha: float = 0.8,
                              n_contours: int = 8) -> None:
        """
        Batch visualization for multiple analyzer pairs - basic plots.
        
        Creates density scatter and contour plots for each analyzer pair.
        Each pair gets its own output directory with plots named by pair.
        
        Args:
            analyzer_pairs: Dictionary of {'pair_name': analyzer_instance}
                          e.g., {'flat_vs_cross': analyzer1, 'flat_vs_side': analyzer2}
            base_output_dir: Base directory name prefix for outputs (default: 'plots')
            figsize: Figure size for plots (default: (8, 6))
            dpi: Resolution for saved figures (default: 300)
            cmap: Colormap name (default: 'jet')
            msize: Marker size for scatter plots (default: 10)
            alpha: Transparency for scatter points (default: 0.8)
            n_contours: Number of contour levels (default: 8)
            
        Example:
            >>> analyzer_pairs = {
            ...     'flat_vs_cross': flat_cross_analyzer,
            ...     'flat_vs_side': flat_side_analyzer,
            ...     'side_vs_cross': side_cross_analyzer
            ... }
            >>> RMSDPlotter.batch_visualize_basic(analyzer_pairs)
            
        Output Structure:
            plots_flat_vs_cross/
                density_scatter_flat_vs_cross.png
                scatter_contours_flat_vs_cross.png
            plots_flat_vs_side/
                density_scatter_flat_vs_side.png
                scatter_contours_flat_vs_side.png
            plots_side_vs_cross/
                density_scatter_side_vs_cross.png
                scatter_contours_side_vs_cross.png
        """
        print("="*70)
        print("BATCH VISUALIZATION: All Reference Pairs")
        print("="*70)
        
        # Loop through each pair and create visualizations
        for pair_name, analyzer in analyzer_pairs.items():
            print(f"\n{'='*70}")
            print(f"Visualizing: {analyzer.label_x.upper()} vs {analyzer.label_y.upper()}")
            print(f"{'='*70}")
            
            # Create output directory for this pair
            output_dir = f"{base_output_dir}_{pair_name}"
            Path(output_dir).mkdir(exist_ok=True)
            
            # Initialize plotter
            plotter = RMSDPlotter(analyzer, figsize=figsize, dpi=dpi, cmap=cmap)
            
            # Create density scatter plot
            print(f"  Creating density scatter plot...")
            fig1 = plotter.plot_density_scatter(
                msize=msize, 
                alpha=alpha,
                save_path=f"{output_dir}/density_scatter_{pair_name}.png"
            )
            plt.close(fig1)
            
            # Create scatter with contours overlay
            print(f"  Creating contours overlay...")
            fig2 = plotter.plot_density_scatter_with_contours(
                msize=msize * 0.8,
                n_contours=n_contours,
                save_path=f"{output_dir}/scatter_contours_{pair_name}.png"
            )
            plt.close(fig2)
            
            print(f"  ✓ Plots saved to {output_dir}/")
        
        print(f"\n{'='*70}")
        print("✓ BATCH VISUALIZATION COMPLETE")
        print(f"{'='*70}")
        print("\nAll plots generated and saved to respective directories:")
        for pair_name in analyzer_pairs.keys():
            print(f"  - {base_output_dir}_{pair_name}/")
    
    @staticmethod
    def batch_visualize_comprehensive(analyzer_pairs: Dict[str, 'RMSDClusterAnalyzer'],
                                       base_output_dir: str = 'comprehensive_plots',
                                       figsize: Tuple[float, float] = (8, 6),
                                       dpi: int = 300,
                                       cmap: str = 'jet',
                                       high_dpi: int = 600,
                                       msize: float = 10,
                                       alpha: float = 0.8,
                                       n_contours: int = 8,
                                       show_title: bool = False) -> None:
        """
        Comprehensive batch visualization for multiple analyzer pairs - all plot types.
        
        Creates all available plot types for each analyzer pair:
        1. Density scatter
        2. Scatter with contours
        3. Combined view (2x2 grid)
        4. Time evolution
        5. Clusters with overlay
        6. Cluster overlay
        7. Clusters with density
        
        Args:
            analyzer_pairs: Dictionary of {'pair_name': analyzer_instance}
            base_output_dir: Base directory name prefix (default: 'comprehensive_plots')
            figsize: Figure size for most plots (default: (8, 6))
            dpi: Standard resolution (default: 300)
            cmap: Colormap name (default: 'jet')
            high_dpi: High resolution for publication plots (default: 600)
            msize: Marker size for scatter plots (default: 10)
            alpha: Transparency for scatter points (default: 0.8)
            n_contours: Number of contour levels (default: 8)
            show_title: Whether to show titles on plots (default: False)
            
        Example:
            >>> analyzer_pairs = {
            ...     'flat_vs_cross': flat_cross_analyzer,
            ...     'flat_vs_side': flat_side_analyzer
            ... }
            >>> RMSDPlotter.batch_visualize_comprehensive(
            ...     analyzer_pairs,
            ...     high_dpi=1200,
            ...     show_title=True
            ... )
            
        Output Structure:
            comprehensive_plots_flat_vs_cross/
                density_scatter_flat_vs_cross.png
                scatter_contours_flat_vs_cross.png
                combined_view_flat_vs_cross.png
                time_evolution_flat_vs_cross.png
                clusters_overlay_flat_vs_cross.png
                cluster_overlay_flat_vs_cross.png
                clusters_with_density_flat_vs_cross.png
            comprehensive_plots_flat_vs_side/
                [same structure]
        """
        print("="*70)
        print("COMPREHENSIVE BATCH VISUALIZATION")
        print("="*70)
        
        # Loop through each pair
        for pair_name, analyzer in analyzer_pairs.items():
            print(f"\n{'='*70}")
            print(f"Processing: {analyzer.label_x.upper()} vs {analyzer.label_y.upper()}")
            print(f"{'='*70}")
            
            # Create output directory
            output_dir = f"{base_output_dir}_{pair_name}"
            Path(output_dir).mkdir(exist_ok=True)
            
            # Initialize plotter
            plotter = RMSDPlotter(analyzer, figsize=figsize, dpi=dpi, cmap=cmap)
            
            # 1. Density scatter
            print("  [1/7] Density scatter...")
            fig1 = plotter.plot_density_scatter(
                msize=msize, 
                alpha=alpha, 
                save_path=f"{output_dir}/density_scatter_{pair_name}.png"
            )
            plt.close(fig1)
            
            # 2. Scatter with contours
            print("  [2/7] Scatter with contours...")
            fig2 = plotter.plot_density_scatter_with_contours(
                msize=msize * 0.8,
                n_contours=n_contours,
                save_path=f"{output_dir}/scatter_contours_{pair_name}.png"
            )
            plt.close(fig2)
            
            # 3. Combined view
            print("  [3/7] Combined view...")
            fig3 = plotter.plot_combined_view(
                cmap=cmap,
                msize=msize,
                show_title=show_title,
                dpi=high_dpi,
                save_individual_figures=True,
                individual_save_dir=output_dir,
                save_combined_figure=True,
                save_path=f"{output_dir}/combined_view_{pair_name}.png"
            )
            plt.close(fig3)
            
            # 4. Time evolution
            print("  [4/7] Time evolution...")
            fig4 = plotter.plot_time_evolution(
                single_plot=True,
                show_title=show_title,
                dpi=high_dpi,
                save_path=f"{output_dir}/time_evolution_{pair_name}.png"
            )
            plt.close(fig4)
            
            # 5. Clusters with density overlay
            print("  [5/7] Clusters with overlay...")
            fig5 = plotter.plot_clusters(
                figsize=figsize,
                msize=msize,
                overlay_density=True,
                overlay_type='both',
                show_title=show_title,
                save_fig=True,
                save_path=f"{output_dir}/clusters_overlay_{pair_name}.png"
            )
            plt.close(fig5)
            
            # 6. Cluster overlay
            print("  [6/7] Cluster overlay...")
            fig6 = plotter.plot_cluster_overlay(
                figsize=(10, 8),
                msize=msize,
                show_title=show_title,
                save_fig=True,
                save_path=f"{output_dir}/cluster_overlay_{pair_name}.png"
            )
            plt.close(fig6)
            
            # 7. Clusters with density
            print("  [7/7] Clusters with density...")
            fig7 = plotter.plot_clusters_with_density(
                figsize=(10, 8),
                msize=msize * 1.5,
                cmap=cmap,
                show_title=show_title,
                save_fig=True,
                save_path=f"{output_dir}/clusters_with_density_{pair_name}.png"
            )
            plt.close(fig7)
            
            print(f"  ✓ All plots saved to {output_dir}/")
        
        print(f"\n{'='*70}")
        print("✓ COMPREHENSIVE BATCH VISUALIZATION COMPLETE")
        print(f"{'='*70}")
        print("\nGenerated comprehensive plots for all reference pairs:")
        for pair_name in analyzer_pairs.keys():
            print(f"  - {base_output_dir}_{pair_name}/")
    
    def generate_cluster_characterization_report(self,
                                                cluster_ids: Optional[Union[str, List[int]]] = None,
                                                include_distances: bool = True,
                                                include_orientations: bool = True,
                                                include_bridging: bool = True,
                                                include_hbonds: bool = True,
                                                include_free_energy: bool = True,
                                                include_energetics: bool = True,
                                                save_report: bool = False,
                                                report_path: Optional[str] = None,
                                                verbose: bool = True) -> Dict:
        """
        Generate comprehensive characterization report for clusters combining RMSD-based
        conformational analysis with all available molecular analyses.
        
        This method provides a complete structural, thermodynamic, and interaction
        profile for each cluster, integrating data from multiple analysis methods.
        
        Parameters
        ----------
        cluster_ids : 'all', list of int, or None, optional
            Cluster IDs to characterize. If None or 'all', analyzes all clusters.
        include_distances : bool, default=True
            Include distance distribution analysis (requires compute_distance_distribution)
        include_orientations : bool, default=True
            Include orientation analysis (requires compute_orientation_angles)
        include_bridging : bool, default=True
            Include bridging analysis (requires generate_bridging_report)
        include_hbonds : bool, default=True
            Include hydrogen bond analysis (requires compute_hydrogen_bonds)
        include_free_energy : bool, default=True
            Include free energy analysis (requires compute_cluster_free_energies)
        include_energetics : bool, default=True
            Include energy decomposition (requires compute_energy_decomposition)
        save_report : bool, default=False
            Save report to file
        report_path : str, optional
            Path for saved report. If None, auto-generates name.
        verbose : bool, default=True
            Print formatted tables and summary
        
        Returns
        -------
        report : dict
            Structured dictionary containing:
            
            'rmsd_analysis': dict
                Centroid positions, spreads, inter-cluster distances
            'cluster_characterization': dict
                Per-cluster conformational interpretation
            'similarity_matrix': np.ndarray
                Conformational similarity between clusters
            'distance_analysis': dict (if available)
                Binding distance statistics per cluster
            'orientation_analysis': dict (if available)
                Spatial orientation preferences per cluster
            'bridging_analysis': dict (if available)
                Inter-molecular bridging patterns per cluster
            'hbond_analysis': dict (if available)
                Hydrogen bond network characteristics per cluster
            'thermodynamics': dict (if available)
                Free energy landscape and populations
            'energetics': dict (if available)
                Interaction energy decomposition per cluster
            'ranking_tables': dict
                Clusters ranked by various metrics
            'interpretation': str
                Overall conformational landscape interpretation
        
        Examples
        --------
        >>> # Basic RMSD-only characterization
        >>> report = plotter.generate_cluster_characterization_report(
        ...     cluster_ids=[0, 1, 2, 3, 4]
        ... )
        
        >>> # Full characterization with all analyses
        >>> report = plotter.generate_cluster_characterization_report(
        ...     cluster_ids='all',
        ...     save_report=True,
        ...     report_path='cluster_analysis_report.txt'
        ... )
        
        >>> # Access specific metrics
        >>> centroids = report['rmsd_analysis']['centroids']
        >>> free_energies = report['thermodynamics']['delta_G']
        >>> similarity = report['similarity_matrix']
        
        >>> # Get ranking by stability
        >>> stable_clusters = report['ranking_tables']['by_compactness']
        
        Notes
        -----
        - Requires DBSCAN clustering to be performed
        - Optional analyses only included if corresponding compute methods were run
        - Generates publication-ready tables and interpretations
        - Similarity matrix based on Euclidean distance in RMSD space
        """
        import pandas as pd
        from scipy.spatial.distance import cdist, euclidean
        
        # Validate clustering was performed
        # Check for both DBSCAN (labels) and k-means (cluster_labels) clustering
        if hasattr(self.analyzer, 'labels') and self.analyzer.labels is not None:
            cluster_label_attr = 'labels'
        elif hasattr(self.analyzer, 'cluster_labels') and self.analyzer.cluster_labels is not None:
            cluster_label_attr = 'cluster_labels'
        else:
            raise ValueError("No clustering found. Run clustering first (DBSCAN or k-means).")
        
        # Get cluster IDs to analyze
        labels_array = getattr(self.analyzer, cluster_label_attr)
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = sorted([c for c in np.unique(labels_array) if c >= 0])
        elif isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]
        
        if len(cluster_ids) == 0:
            raise ValueError("No valid cluster IDs found.")
        
        # Initialize report structure
        report = {
            'rmsd_analysis': {},
            'cluster_characterization': {},
            'similarity_matrix': None,
            'ranking_tables': {},
            'interpretation': ""
        }
        
        # ============ RMSD-BASED CONFORMATIONAL ANALYSIS ============
        
        if verbose:
            print("\n" + "="*80)
            print(" CLUSTER CONFORMATIONAL CHARACTERIZATION REPORT")
            print("="*80)
            print(f"\nAnalyzing {len(cluster_ids)} clusters")
            print(f"RMSD space dimensions: {self.analyzer.label_x} vs {self.analyzer.label_y}")
        
        # Calculate cluster centroids and spreads
        centroids = {}
        spreads = {}
        sizes = {}
        rmsd_from_origin = {}
        
        for cluster_id in cluster_ids:
            mask = labels_array == cluster_id
            cluster_points = np.column_stack([
                self.analyzer.rmsd_x[mask],
                self.analyzer.rmsd_y[mask]
            ])
            
            centroid = cluster_points.mean(axis=0)
            spread = cluster_points.std(axis=0).mean()  # Average of x and y std
            size = mask.sum()
            dist_origin = np.linalg.norm(centroid)
            
            centroids[cluster_id] = centroid
            spreads[cluster_id] = spread
            sizes[cluster_id] = size
            rmsd_from_origin[cluster_id] = dist_origin
        
        report['rmsd_analysis'] = {
            'centroids': centroids,
            'spreads': spreads,
            'sizes': sizes,
            'rmsd_from_origin': rmsd_from_origin
        }
        
        # Calculate inter-cluster distances (similarity matrix)
        n_clusters = len(cluster_ids)
        similarity_matrix = np.zeros((n_clusters, n_clusters))
        centroid_array = np.array([centroids[cid] for cid in cluster_ids])
        similarity_matrix = cdist(centroid_array, centroid_array, metric='euclidean')
        report['similarity_matrix'] = similarity_matrix
        
        # Print RMSD-based characterization table
        if verbose:
            print("\n" + "-"*80)
            print(" RMSD-BASED CONFORMATIONAL METRICS")
            print("-"*80)
            
            rmsd_df = pd.DataFrame({
                'Cluster': cluster_ids,
                'Centroid_X': [centroids[c][0] for c in cluster_ids],
                'Centroid_Y': [centroids[c][1] for c in cluster_ids],
                'Spread (STD)': [spreads[c] for c in cluster_ids],
                'Size (frames)': [sizes[c] for c in cluster_ids],
                'Dist_Origin': [rmsd_from_origin[c] for c in cluster_ids]
            })
            
            print(rmsd_df.to_string(index=False, float_format='%.3f'))
            
            # Interpretation
            print("\nInterpretation:")
            print("  • Dist_Origin: Conformational deviation from reference structure")
            print("  • Spread: Conformational flexibility (low = rigid, high = flexible)")
            print("  • Size: Statistical weight (more frames = more prevalent)")
        
        # Print similarity matrix
        if verbose:
            print("\n" + "-"*80)
            print(" CONFORMATIONAL SIMILARITY MATRIX (RMSD distance)")
            print("-"*80)
            
            sim_df = pd.DataFrame(
                similarity_matrix,
                index=[f"C{c}" for c in cluster_ids],
                columns=[f"C{c}" for c in cluster_ids]
            )
            print(sim_df.to_string(float_format='%.3f'))
            
            # Find most similar and most different pairs
            triu_indices = np.triu_indices(n_clusters, k=1)
            triu_distances = similarity_matrix[triu_indices]
            if len(triu_distances) > 0:
                min_dist_idx = np.argmin(triu_distances)
                max_dist_idx = np.argmax(triu_distances)
                
                i_min, j_min = triu_indices[0][min_dist_idx], triu_indices[1][min_dist_idx]
                i_max, j_max = triu_indices[0][max_dist_idx], triu_indices[1][max_dist_idx]
                
                print(f"\n  Most similar: C{cluster_ids[i_min]} ↔ C{cluster_ids[j_min]} "
                      f"(dist={triu_distances[min_dist_idx]:.3f})")
                print(f"  Most different: C{cluster_ids[i_max]} ↔ C{cluster_ids[j_max]} "
                      f"(dist={triu_distances[max_dist_idx]:.3f})")
        
        # ============ CHARACTERIZATION PER CLUSTER ============
        
        for cluster_id in cluster_ids:
            char = {
                'rmsd_position': f"({centroids[cluster_id][0]:.2f}, {centroids[cluster_id][1]:.2f})",
                'conformational_state': self._interpret_rmsd_position(centroid[0], rmsd_from_origin[cluster_id]),
                'flexibility': 'Rigid' if spreads[cluster_id] < 0.1 else 'Flexible' if spreads[cluster_id] > 0.2 else 'Moderate',
                'population_fraction': sizes[cluster_id] / labels_array.size,
                'deviation_from_reference': rmsd_from_origin[cluster_id]
            }
            report['cluster_characterization'][cluster_id] = char
        
        # ============ INTEGRATE OPTIONAL ANALYSES ============
        
        # Distance distributions
        if include_distances and hasattr(self.analyzer, 'distance_data'):
            report['distance_analysis'] = self._extract_distance_stats(cluster_ids)
            if verbose:
                print("\n" + "-"*80)
                print(" DISTANCE DISTRIBUTION ANALYSIS")
                print("-"*80)
                self._print_distance_table(report['distance_analysis'])
        
        # Orientations
        if include_orientations and hasattr(self.analyzer, 'orientation_data'):
            report['orientation_analysis'] = self._extract_orientation_stats(cluster_ids)
            if verbose:
                print("\n" + "-"*80)
                print(" ORIENTATION ANALYSIS")
                print("-"*80)
                self._print_orientation_table(report['orientation_analysis'])
        
        # Bridging
        if include_bridging and hasattr(self.analyzer, 'bridging_data'):
            report['bridging_analysis'] = self._extract_bridging_stats(cluster_ids)
            if verbose:
                print("\n" + "-"*80)
                print(" BRIDGING ANALYSIS")
                print("-"*80)
                self._print_bridging_table(report['bridging_analysis'])
        
        # Hydrogen bonds
        if include_hbonds and hasattr(self.analyzer, 'hbond_data'):
            report['hbond_analysis'] = self._extract_hbond_stats(cluster_ids)
            if verbose:
                print("\n" + "-"*80)
                print(" HYDROGEN BOND ANALYSIS")
                print("-"*80)
                self._print_hbond_table(report['hbond_analysis'])
        
        # Free energy
        if include_free_energy and hasattr(self.analyzer, 'fe_data'):
            report['thermodynamics'] = self._extract_thermodynamic_stats(cluster_ids)
            if verbose:
                print("\n" + "-"*80)
                print(" THERMODYNAMIC ANALYSIS")
                print("-"*80)
                self._print_thermodynamics_table(report['thermodynamics'])
        
        # Energy decomposition
        if include_energetics and hasattr(self.analyzer, 'energy_data'):
            report['energetics'] = self._extract_energy_stats(cluster_ids)
            if verbose:
                print("\n" + "-"*80)
                print(" INTERACTION ENERGY DECOMPOSITION")
                print("-"*80)
                self._print_energetics_table(report['energetics'])
        
        # ============ RANKING TABLES ============
        
        # Rank by various metrics
        rankings = {}
        
        # By size (population)
        rankings['by_population'] = sorted(cluster_ids, key=lambda c: sizes[c], reverse=True)
        
        # By compactness (low spread = more stable)
        rankings['by_compactness'] = sorted(cluster_ids, key=lambda c: spreads[c])
        
        # By deviation from reference
        rankings['by_similarity_to_ref'] = sorted(cluster_ids, key=lambda c: rmsd_from_origin[c])
        
        if 'thermodynamics' in report and report['thermodynamics']:
            fe_dict = report['thermodynamics'].get('delta_G', {})
            rankings['by_stability'] = sorted(
                [c for c in cluster_ids if c in fe_dict],
                key=lambda c: fe_dict[c]
            )
        
        report['ranking_tables'] = rankings
        
        if verbose:
            print("\n" + "-"*80)
            print(" CLUSTER RANKINGS")
            print("-"*80)
            
            print("\n  By Population (most prevalent):")
            print("    " + " > ".join([f"C{c}" for c in rankings['by_population'][:5]]))
            
            print("\n  By Compactness (most rigid/stable):")
            print("    " + " > ".join([f"C{c}" for c in rankings['by_compactness'][:5]]))
            
            print("\n  By Similarity to Reference:")
            print("    " + " > ".join([f"C{c}" for c in rankings['by_similarity_to_ref'][:5]]))
            
            if 'by_stability' in rankings:
                print("\n  By Thermodynamic Stability (lowest ΔG):")
                print("    " + " > ".join([f"C{c}" for c in rankings['by_stability'][:5]]))
        
        # ============ OVERALL INTERPRETATION ============
        
        interpretation = self._generate_interpretation(report, cluster_ids)
        report['interpretation'] = interpretation
        
        if verbose:
            print("\n" + "="*80)
            print(" OVERALL CONFORMATIONAL LANDSCAPE INTERPRETATION")
            print("="*80)
            print(interpretation)
            print("="*80 + "\n")
        
        # ============ SAVE REPORT ============
        
        if save_report:
            if report_path is None:
                report_path = "cluster_characterization_report.txt"
            
            self._save_report_to_file(report, report_path, cluster_ids)
            if verbose:
                print(f"\n✓ Report saved to: {report_path}")
        
        return report
    
    # ============ HELPER METHODS FOR CHARACTERIZATION ============
    
    def _interpret_rmsd_position(self, rmsd_x: float, rmsd_norm: float) -> str:
        """Interpret RMSD position as conformational state."""
        if rmsd_norm < 0.2:
            return "Native-like"
        elif rmsd_norm < 0.5:
            return "Slightly perturbed"
        elif rmsd_norm < 1.0:
            return "Moderately altered"
        else:
            return "Highly divergent"
    
    def _extract_distance_stats(self, cluster_ids: List[int]) -> Dict:
        """Extract statistics from distance distribution data."""
        stats = {}
        
        for key, dist_data in self.analyzer.distance_data.items():
            stats[key] = {}
            for cluster_id in cluster_ids:
                if cluster_id not in dist_data:
                    continue
                
                distances = dist_data[cluster_id]['distances']
                hist_data = dist_data[cluster_id]['hist_data']
                
                # Calculate mean distance (weighted by histogram)
                mean_dist = np.average(distances, weights=hist_data)
                # Most probable distance (mode)
                mode_idx = np.argmax(hist_data)
                mode_dist = distances[mode_idx]
                
                stats[key][cluster_id] = {
                    'mean_distance': mean_dist,
                    'mode_distance': mode_dist,
                    'range': (distances.min(), distances.max())
                }
        
        return stats
    
    def _extract_orientation_stats(self, cluster_ids: List[int]) -> Dict:
        """Extract statistics from orientation analysis data."""
        stats = {}
        
        for pair_key, orient_data in self.analyzer.orientation_data.items():
            stats[pair_key] = {}
            for cluster_id in cluster_ids:
                if cluster_id not in orient_data:
                    continue
                
                angles = orient_data[cluster_id]['angles']
                mean_angle = angles.mean()
                std_angle = angles.std()
                
                # Classify orientation preference
                if mean_angle < 30:
                    preference = "Parallel"
                elif mean_angle > 60:
                    preference = "Perpendicular"
                else:
                    preference = "Intermediate"
                
                stats[pair_key][cluster_id] = {
                    'mean_angle': mean_angle,
                    'std_angle': std_angle,
                    'preference': preference
                }
        
        return stats
    
    def _extract_bridging_stats(self, cluster_ids: List[int]) -> Dict:
        """Extract bridging statistics."""
        stats = {}
        
        if hasattr(self.analyzer, 'bridging_data'):
            for cluster_id in cluster_ids:
                if cluster_id in self.analyzer.bridging_data:
                    bridge_info = self.analyzer.bridging_data[cluster_id]
                    stats[cluster_id] = {
                        'n_bridging_frames': bridge_info.get('n_bridging', 0),
                        'bridging_fraction': bridge_info.get('frac_bridging', 0.0),
                        'avg_bridges': bridge_info.get('mean_n_bridges', 0.0)
                    }
        
        return stats
    
    def _extract_hbond_stats(self, cluster_ids: List[int]) -> Dict:
        """Extract hydrogen bond statistics."""
        stats = {}
        
        if hasattr(self.analyzer, 'hbond_data'):
            for pair_key, hb_data in self.analyzer.hbond_data.items():
                stats[pair_key] = {}
                for cluster_id in cluster_ids:
                    if cluster_id in hb_data:
                        hb_counts = hb_data[cluster_id]['hbond_counts']
                        stats[pair_key][cluster_id] = {
                            'mean_hbonds': hb_counts.mean(),
                            'max_hbonds': hb_counts.max(),
                            'occupancy': (hb_counts > 0).sum() / len(hb_counts)
                        }
        
        return stats
    
    def _extract_thermodynamic_stats(self, cluster_ids: List[int]) -> Dict:
        """Extract thermodynamic statistics from free energy data."""
        stats = {}
        
        if hasattr(self.analyzer, 'fe_data'):
            for cluster_id in cluster_ids:
                if cluster_id in self.analyzer.fe_data:
                    fe_info = self.analyzer.fe_data[cluster_id]
                    stats[cluster_id] = {
                        'delta_G': fe_info['delta_G'],
                        'std_error': fe_info['std_error'],
                        'population': fe_info['population']
                    }
            
            # Add delta_G dict for easy sorting
            stats['delta_G'] = {c: stats[c]['delta_G'] for c in cluster_ids if c in stats}
        
        return stats
    
    def _extract_energy_stats(self, cluster_ids: List[int]) -> Dict:
        """Extract interaction energy statistics."""
        stats = {}
        
        if hasattr(self.analyzer, 'energy_data'):
            for cluster_id in cluster_ids:
                if cluster_id in self.analyzer.energy_data:
                    cluster_energies = self.analyzer.energy_data[cluster_id]
                    stats[cluster_id] = {}
                    
                    for group_pair, components in cluster_energies.items():
                        stats[cluster_id][group_pair] = {
                            'Coul-SR_mean': components['Coul-SR'].mean(),
                            'LJ-SR_mean': components['LJ-SR'].mean(),
                            'Total_mean': components['Total'].mean(),
                            'Coul-SR_std': components['Coul-SR'].std(),
                            'LJ-SR_std': components['LJ-SR'].std()
                        }
        
        return stats
    
    def _print_distance_table(self, distance_analysis: Dict) -> None:
        """Print formatted distance analysis table."""
        import pandas as pd
        
        for key, cluster_stats in distance_analysis.items():
            print(f"\n  {key}:")
            df_data = []
            for cluster_id, stats in cluster_stats.items():
                df_data.append({
                    'Cluster': cluster_id,
                    'Mean (Å)': stats['mean_distance'],
                    'Mode (Å)': stats['mode_distance'],
                    'Range (Å)': f"{stats['range'][0]:.1f}-{stats['range'][1]:.1f}"
                })
            df = pd.DataFrame(df_data)
            print(df.to_string(index=False, float_format='%.2f'))
    
    def _print_orientation_table(self, orientation_analysis: Dict) -> None:
        """Print formatted orientation analysis table."""
        import pandas as pd
        
        for pair_key, cluster_stats in orientation_analysis.items():
            print(f"\n  {pair_key}:")
            df_data = []
            for cluster_id, stats in cluster_stats.items():
                df_data.append({
                    'Cluster': cluster_id,
                    'Mean Angle (°)': stats['mean_angle'],
                    'Std (°)': stats['std_angle'],
                    'Preference': stats['preference']
                })
            df = pd.DataFrame(df_data)
            print(df.to_string(index=False, float_format='%.1f'))
    
    def _print_bridging_table(self, bridging_analysis: Dict) -> None:
        """Print formatted bridging analysis table."""
        import pandas as pd
        
        df_data = []
        for cluster_id, stats in bridging_analysis.items():
            df_data.append({
                'Cluster': cluster_id,
                'Bridging Frames': stats['n_bridging_frames'],
                'Fraction': stats['bridging_fraction'],
                'Avg Bridges': stats['avg_bridges']
            })
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False, float_format='%.3f'))
    
    def _print_hbond_table(self, hbond_analysis: Dict) -> None:
        """Print formatted hydrogen bond analysis table."""
        import pandas as pd
        
        for pair_key, cluster_stats in hbond_analysis.items():
            print(f"\n  {pair_key}:")
            df_data = []
            for cluster_id, stats in cluster_stats.items():
                df_data.append({
                    'Cluster': cluster_id,
                    'Mean H-bonds': stats['mean_hbonds'],
                    'Max H-bonds': stats['max_hbonds'],
                    'Occupancy': stats['occupancy']
                })
            df = pd.DataFrame(df_data)
            print(df.to_string(index=False, float_format='%.2f'))
    
    def _print_thermodynamics_table(self, thermo_analysis: Dict) -> None:
        """Print formatted thermodynamics table."""
        import pandas as pd
        
        df_data = []
        for cluster_id, stats in thermo_analysis.items():
            if cluster_id == 'delta_G':
                continue
            df_data.append({
                'Cluster': cluster_id,
                'ΔG (kJ/mol)': stats['delta_G'],
                'Std Error': stats['std_error'],
                'Population': stats['population']
            })
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False, float_format='%.3f'))
    
    def _print_energetics_table(self, energetics_analysis: Dict) -> None:
        """Print formatted energetics table."""
        import pandas as pd
        
        for cluster_id, group_energies in energetics_analysis.items():
            print(f"\n  Cluster {cluster_id}:")
            df_data = []
            for group_pair, stats in group_energies.items():
                df_data.append({
                    'Group Pair': f"{group_pair[0]}-{group_pair[1]}",
                    'Coulomb (kJ/mol)': stats['Coul-SR_mean'],
                    'LJ (kJ/mol)': stats['LJ-SR_mean'],
                    'Total (kJ/mol)': stats['Total_mean']
                })
            df = pd.DataFrame(df_data)
            print(df.to_string(index=False, float_format='%.1f'))
    
    def _generate_interpretation(self, report: Dict, cluster_ids: List[int]) -> str:
        """Generate overall interpretation text."""
        n_clusters = len(cluster_ids)
        
        # Get dominant cluster
        sizes = report['rmsd_analysis']['sizes']
        dominant_cluster = max(cluster_ids, key=lambda c: sizes[c])
        dominant_frac = sizes[dominant_cluster] / sum(sizes.values())
        
        # Get most compact cluster
        spreads = report['rmsd_analysis']['spreads']
        compact_cluster = min(cluster_ids, key=lambda c: spreads[c])
        
        # Get most deviant cluster
        deviations = report['rmsd_analysis']['rmsd_from_origin']
        deviant_cluster = max(cluster_ids, key=lambda c: deviations[c])
        
        text = f"""
The molecular system exhibits {n_clusters} distinct conformational clusters in RMSD space.

CONFORMATIONAL LANDSCAPE:
  • Cluster {dominant_cluster} is the dominant conformation, representing {dominant_frac*100:.1f}% 
    of all sampled structures, indicating it is the most statistically prevalent state.
  
  • Cluster {compact_cluster} shows the highest conformational rigidity (lowest spread = 
    {spreads[compact_cluster]:.3f}), suggesting a well-defined, stable structural ensemble.
  
  • Cluster {deviant_cluster} exhibits the largest deviation from the reference structure 
    (RMSD = {deviations[deviant_cluster]:.2f}), representing the most structurally distinct 
    conformational state.

STRUCTURAL RELATIONSHIPS:
  • Conformational similarity analysis reveals """
        
        # Add similarity insights
        sim_matrix = report['similarity_matrix']
        if sim_matrix is not None and len(sim_matrix) > 1:
            triu_indices = np.triu_indices(len(cluster_ids), k=1)
            triu_distances = sim_matrix[triu_indices]
            mean_intercluster_dist = triu_distances.mean()
            text += f"an average inter-cluster RMSD distance of {mean_intercluster_dist:.2f}, "
            
            if mean_intercluster_dist < 0.5:
                text += "indicating closely related conformations with subtle structural variations."
            elif mean_intercluster_dist < 1.0:
                text += "indicating moderately distinct conformational states."
            else:
                text += "indicating highly diverse conformational ensemble with distinct structural states."
        
        # Add thermodynamics if available
        if 'thermodynamics' in report and report['thermodynamics']:
            fe_dict = report['thermodynamics'].get('delta_G', {})
            if fe_dict:
                stable_cluster = min(fe_dict.keys(), key=lambda c: fe_dict[c])
                min_fe = fe_dict[stable_cluster]
                text += f"""

THERMODYNAMIC PROFILE:
  • Cluster {stable_cluster} is the most thermodynamically stable state (ΔG = {min_fe:.2f} kJ/mol),
    representing the energetically most favorable conformation.
"""
                
                # Compare population vs stability
                if stable_cluster != dominant_cluster:
                    text += f"""  • Note: The most populous cluster (C{dominant_cluster}) differs from the most stable 
    cluster (C{stable_cluster}), suggesting kinetic trapping or barrier-separated states.
"""
        
        # Add interaction insights if available
        if 'energetics' in report and report['energetics']:
            text += """

INTERACTION ENERGETICS:
  • Interaction energy decomposition reveals cluster-specific binding modes and 
    stabilization patterns through electrostatic (Coulomb) and van der Waals (LJ) components.
"""
        
        text += """

RECOMMENDATION:
  Focus detailed structural analysis on the dominant and most compact clusters to 
  understand the predominant binding modes. Examine transition pathways between 
  clusters to understand conformational dynamics and interconversion mechanisms.
"""
        
        return text
    
    def _save_report_to_file(self, report: Dict, filepath: str, cluster_ids: List[int]) -> None:
        """Save report to text file."""
        import pandas as pd
        from datetime import datetime
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" CLUSTER CONFORMATIONAL CHARACTERIZATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Clusters analyzed: {', '.join([f'C{c}' for c in cluster_ids])}\n")
            f.write("="*80 + "\n\n")
            
            # RMSD analysis
            f.write("RMSD-BASED CONFORMATIONAL METRICS\n")
            f.write("-"*80 + "\n")
            rmsd_df = pd.DataFrame({
                'Cluster': cluster_ids,
                'Centroid_X': [report['rmsd_analysis']['centroids'][c][0] for c in cluster_ids],
                'Centroid_Y': [report['rmsd_analysis']['centroids'][c][1] for c in cluster_ids],
                'Spread': [report['rmsd_analysis']['spreads'][c] for c in cluster_ids],
                'Size': [report['rmsd_analysis']['sizes'][c] for c in cluster_ids],
                'Dist_Origin': [report['rmsd_analysis']['rmsd_from_origin'][c] for c in cluster_ids]
            })
            f.write(rmsd_df.to_string(index=False, float_format='%.3f') + "\n\n")
            
            # Similarity matrix
            f.write("CONFORMATIONAL SIMILARITY MATRIX\n")
            f.write("-"*80 + "\n")
            sim_df = pd.DataFrame(
                report['similarity_matrix'],
                index=[f"C{c}" for c in cluster_ids],
                columns=[f"C{c}" for c in cluster_ids]
            )
            f.write(sim_df.to_string(float_format='%.3f') + "\n\n")
            
            # Other analyses if present
            for key in ['distance_analysis', 'orientation_analysis', 'bridging_analysis',
                       'hbond_analysis', 'thermodynamics', 'energetics']:
                if key in report and report[key]:
                    f.write(f"\n{key.upper().replace('_', ' ')}\n")
                    f.write("-"*80 + "\n")
                    f.write(str(report[key]) + "\n")
            
            # Rankings
            f.write("\nCLUSTER RANKINGS\n")
            f.write("-"*80 + "\n")
            for rank_type, ranked_ids in report['ranking_tables'].items():
                f.write(f"{rank_type}: " + " > ".join([f"C{c}" for c in ranked_ids[:5]]) + "\n")
            
            # Interpretation
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL INTERPRETATION\n")
            f.write("="*80 + "\n")
            f.write(report['interpretation'])
            f.write("\n" + "="*80 + "\n")

    def plot_cluster_spatial_binding_interactive(self,
                                                 spatial_results: Dict,
                                                 structure_file: Optional[str] = None,
                                                 universe=None,
                                                 density_threshold: float = 0.02,
                                                 distance_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
                                                 distance_method: str = 'nearest_atom',
                                                 sphere_size: float = 0.4,
                                                 sphere_opacity: float = 0.3,
                                                 stick_radius: float = 0.15,
                                                 ball_scale: float = 0.3,
                                                 width: int = 800,
                                                 height: int = 600,
                                                 show_output: bool = True,
                                                 max_spheres: int = 500,
                                                 # Enhanced parameters
                                                 plot_regions=None,
                                                 shell_info_display: bool = True,
                                                 show_boundary_spheres: bool = False,
                                                 boundary_sphere_alpha: float = 0.3,
                                                 color_shade_style: str = 'modified',
                                                 boundary_sphere_data_extent: bool = False,
                                                 show_aromatic_rings: bool = True,
                                                 aromatic_ring_color: str = 'gold',
                                                 aromatic_ring_alpha: float = 0.7,
                                                 aromatic_ring_scale: float = 0.8,
                                                 aromatic_ring_thickness: float = 0.15,
                                                 use_triangulation: bool = True,
                                                 reconstruction_method: str = 'atom',
                                                 density_scale='auto'):
        """
        Visualize spatial binding for a specific cluster in interactive 3D.
        
        This is a wrapper around MolecularAnalysisPlotter.plot_spatial_binding_interactive()
        that adds cluster-specific labeling and context. Shows the target molecule
        structure with colored spheres indicating where ions bind for a specific cluster.
        
        Parameters
        ----------
        spatial_results : dict
            Results from compute_cluster_spatial_binding()
            Must contain cluster_metadata and ion_positions_relative or triangulation_data
        structure_file : str, optional
            Path to PDB structure file for the target molecule
            If None, structure will be generated from universe
        universe : MDAnalysis.Universe, optional
            Universe object (required if structure_file is None)
        density_threshold : float, default=0.02
            Minimum density to show a binding position sphere (0-1)
        distance_cutoff : float or tuple, optional
            Distance filtering: None, float (max distance), or (min, max) tuple
        distance_method : str, default='nearest_atom'
            'nearest_atom' or 'com' for distance calculation
        sphere_size : float, default=0.4
            Radius of ion position spheres in Angstroms
        sphere_opacity : float, default=0.3
            Opacity of binding site spheres (0-1)
        stick_radius : float, default=0.15
            Radius of molecular structure sticks
        ball_scale : float, default=0.3
            Radius scale for molecule atoms
        width : int, default=800
            Viewer width in pixels
        height : int, default=600
            Viewer height in pixels
        show_output : bool, default=True
            Whether to print detailed visualization information
        max_spheres : int, default=500
            Maximum number of binding position spheres to display
        plot_regions : list, optional
            Specific shell regions to visualize (e.g., ['P1', 'P3'])
        shell_info_display : bool, default=True
            Whether to print shell information
        show_boundary_spheres : bool, default=False
            Display transparent spheres marking shell boundaries
        boundary_sphere_alpha : float, default=0.3
            Transparency for boundary spheres (0-1)
        color_shade_style : str, default='modified'
            Color scheme: 'modified', 'original', or 'vibrant'
        boundary_sphere_data_extent : bool, default=False
            Show only angular regions with actual binding data
        show_aromatic_rings : bool, default=True
            Visualize aromatic rings
        aromatic_ring_color : str, default='gold'
            Color for aromatic rings
        aromatic_ring_alpha : float, default=0.7
            Transparency for aromatic rings
        aromatic_ring_scale : float, default=0.8
            Size scale for aromatic rings
        aromatic_ring_thickness : float, default=0.15
            Thickness of aromatic ring visualization
        use_triangulation : bool, default=True
            Use triangulation data for precise geometric mapping
        reconstruction_method : str, default='atom'
            Method for ion position reconstruction:
            'atom', 'com', 'spherical', 'molecular', 'molecular_atom', 'molecular_spherical'
        density_scale : str or float, default='auto'
            Color scale for cluster comparison. Options:
            - 'auto': Use max density from this cluster (default)
            - float: Fixed scale for comparing across clusters (e.g., 0.5)
            Use same value for all clusters to make colors comparable.
            
        Returns
        -------
        view : py3Dmol.view
            Interactive 3D viewer object (displays automatically in Jupyter)
            
        Raises
        ------
        ValueError
            If spatial_results doesn't contain cluster_metadata
        ImportError
            If MolecularAnalysisPlotter is not available
            
        Examples
        --------
        >>> # Analyze cluster 0
        >>> spatial_data = analyzer.compute_cluster_spatial_binding(
        ...     cluster_id=0,
        ...     target_sel='resname api',
        ...     ion_type='K',
        ...     cutoff=3.5
        ... )
        >>> 
        >>> # Visualize it
        >>> view = plotter.plot_cluster_spatial_binding_interactive(
        ...     spatial_data,
        ...     structure_file='api.pdb'
        ... )
        >>> 
        >>> # With shell-specific visualization
        >>> view = plotter.plot_cluster_spatial_binding_interactive(
        ...     spatial_data,
        ...     structure_file='api.pdb',
        ...     plot_regions=['P1', 'P2'],
        ...     show_boundary_spheres=True
        ... )
        
        Notes
        -----
        - Molecule shown as ball-and-stick structure
        - Spheres positioned where ions bind for this specific cluster
        - Blue spheres = low-density binding, Red spheres = high-density binding
        - Click and drag to rotate, scroll to zoom, right-click to pan
        - Right-click → "Save Image As..." for publication-quality output
        """
        # Validate cluster metadata
        if 'cluster_metadata' not in spatial_results:
            raise ValueError(
                "spatial_results must contain 'cluster_metadata'. "
                "Did you use compute_cluster_spatial_binding()?"
            )
        
        # Import MolecularAnalysisPlotter
        try:
            from MolecularAnalysisPlotter import MolecularAnalysisPlotter
        except ImportError:
            raise ImportError(
                "MolecularAnalysisPlotter module not found. "
                "Ensure it's in your Python path."
            )
        
        # Extract cluster info
        cluster_metadata = spatial_results['cluster_metadata']
        cluster_id = cluster_metadata['cluster_id']
        n_frames = cluster_metadata['n_cluster_frames']
        representative_frame = cluster_metadata.get('representative_frame', 0)
        
        # Position universe to cluster's representative frame
        # This matches temp trajectory frame 0 used during analysis
        if universe is not None and representative_frame is not None:
            universe.trajectory[representative_frame]
            if show_output:
                print(f"✓ Using cluster representative structure from frame {representative_frame}")
                print(f"  (Matches temp trajectory frame 0 used in analysis)")
        
        # Print cluster-specific header
        if show_output:
            print(f"\n{'='*70}")
            print(f"3D Spatial Binding Visualization - Cluster {cluster_id}")
            print(f"{'='*70}")
            print(f"Cluster information:")
            print(f"  Cluster ID: {cluster_id}")
            print(f"  Frames in cluster: {n_frames}")
            print(f"  Representative frame: {representative_frame}")
            print(f"  RMSD center: flat={cluster_metadata['cluster_center'][0]:.4f}, "
                  f"cross={cluster_metadata['cluster_center'][1]:.4f} nm")
            print()
        
        # Create plotter instance (needs analyzer reference)
        plotter = MolecularAnalysisPlotter(self.analyzer)
        
        # Call the existing spatial binding interactive plot
        view = plotter.plot_spatial_binding_interactive(
            spatial_results=spatial_results,
            structure_file=structure_file,
            universe=universe,
            density_threshold=density_threshold,
            distance_cutoff=distance_cutoff,
            distance_method=distance_method,
            sphere_size=sphere_size,
            sphere_opacity=sphere_opacity,
            stick_radius=stick_radius,
            ball_scale=ball_scale,
            width=width,
            height=height,
            show_output=show_output,
            max_spheres=max_spheres,
            plot_regions=plot_regions,
            shell_info_display=shell_info_display,
            show_boundary_spheres=show_boundary_spheres,
            boundary_sphere_alpha=boundary_sphere_alpha,
            color_shade_style=color_shade_style,
            boundary_sphere_data_extent=boundary_sphere_data_extent,
            show_aromatic_rings=show_aromatic_rings,
            aromatic_ring_color=aromatic_ring_color,
            aromatic_ring_alpha=aromatic_ring_alpha,
            aromatic_ring_scale=aromatic_ring_scale,
            aromatic_ring_thickness=aromatic_ring_thickness,
            use_triangulation=use_triangulation,
            reconstruction_method=reconstruction_method,
            density_scale=density_scale
        )
        
        if show_output:
            print(f"\n✓ 3D visualization ready for Cluster {cluster_id}")
            print(f"  Interact with the view above:")
            print(f"    - Click and drag to rotate")
            print(f"    - Scroll to zoom")
            print(f"    - Right-click to pan")
            print(f"    - Right-click → 'Save Image As...' to export")
        
        return view
    
   
    def detect_bonds_by_distance(self, atoms, positions_dict, tolerance=0.4):
        """
        Detect bonds between atoms based on interatomic distances and covalent radii.
        
        This is used as a fallback when MDAnalysis topology is missing bonds
        (e.g., cyclopropyl rings, some aromatic systems).
        
        Parameters
        ----------
        atoms : MDAnalysis.AtomGroup
            Atoms to check for bonding
        positions_dict : dict
            Dictionary mapping atom indices to 3D positions (numpy arrays)
        tolerance : float, optional
            Distance tolerance beyond sum of covalent radii (Angstroms). Default: 0.4
            
        Returns
        -------
        set of tuple
            Set of bond tuples (idx1, idx2) where idx1 < idx2
        """
        # Covalent radii in Angstroms (from Cordero et al. 2008)
        covalent_radii = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
            'S': 1.05, 'P': 1.07, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
            'Si': 1.11, 'Al': 1.21, 'Mg': 1.41, 'Ca': 1.76, 'Na': 1.66,
            'K': 2.03
        }
        
        detected_bonds = set()
        atom_indices = list(positions_dict.keys())
        
        # Check all pairs of atoms
        for i, idx1 in enumerate(atom_indices):
            for idx2 in atom_indices[i+1:]:
                # Get atom objects
                atom1 = atoms[atoms.indices == idx1][0]
                atom2 = atoms[atoms.indices == idx2][0]
                
                # Get element symbols
                elem1 = atom1.element if (hasattr(atom1, 'element') and atom1.element) else atom1.name[0]
                elem2 = atom2.element if (hasattr(atom2, 'element') and atom2.element) else atom2.name[0]
                
                # Get covalent radii (use default if element not found)
                r1 = covalent_radii.get(elem1, 0.77)
                r2 = covalent_radii.get(elem2, 0.77)
                
                # Calculate distance threshold
                max_bond_distance = r1 + r2 + tolerance
                
                # Calculate actual distance
                pos1 = positions_dict[idx1]
                pos2 = positions_dict[idx2]
                distance = np.linalg.norm(pos1 - pos2)
                
                # If within bonding distance, add to set
                if distance < max_bond_distance:
                    # Store as sorted tuple to avoid duplicates
                    bond_key = (min(idx1, idx2), max(idx1, idx2))
                    detected_bonds.add(bond_key)
        
        return detected_bonds
    
    
    def plot_hbond_geometry_3d(self, hbond_results: dict,
                               cluster_id: int,
                               max_hbonds: int = 4,
                               min_hbonds: int = 1,
                               sort_by: str = 'occupancy',
                               prioritise_O: bool = False,
                               distance_cutoff: Optional[float] = None,
                               angle_cutoff: Optional[float] = None,
                               search_distance_cutoff: Optional[Union[float, List[float]]] = None,
                               search_angle_cutoff: Optional[Union[float, List[float]]] = None,
                               show_surface_atoms: bool = True,
                               surface_radius: float = 8.0,
                               surface_z_thickness: float = 10.0,
                               surface_floor_value: float = 1.0,
                               surface_ceiling_value: float = 0.0,
                               surface_sel: str = 'name Si or name Ob or name Op or name Ohs or name Mgo',
                               show_surface_bonds: bool = True,
                               surface_bond_color: str = 'orange',
                               surface_bond_linewidth: float = 3.0,
                               molecule_sel: Optional[str] = None,
                               molecule_radius: float = 10.0,
                               show_waters: bool = True,
                               water_radius: float = 5.0,
                               hbond_line_color: str = 'black',
                               hbond_linewidth: float = 4.0,
                               atom_scale_factor: float = 100,
                               molecule_alpha: float = 0.7,
                               boundary_linewidth: float = 0.5,
                               view_elevation: float = 25,
                               view_azimuth: float = 45,
                               figsize_per_panel: float = 5,
                               dpi: int = 300,
                               save_path: Optional[str] = None,
                               save_combined_figure: bool = True,
                               save_individual_figures: bool = True,
                               individual_figsize: tuple = (8, 6),
                               title_fontsize: int = 14,
                               title_fontweight: str = 'bold',
                               label_fontsize: int = 16,
                               label_fontweight: str = 'bold',
                               tick_fontsize: int = 14,
                               hbond_linestyle: str = 'dashed',
                               bond_style: str = 'solid',
                               atom_edge_style: str = 'solid',
                               surface_atom_style: str = 'scatter',
                               si_color: str = "#F0C8A0",
                               carbon_color: str = "#000000",
                               show_title: bool = True,
                               axis_info: str = 'detailed',
                               start_search_frame: int = 1,
                               skip_when_searching: int = 1):
        """
        Plot 3D H-bond geometry in the style of plot_clean_cross_sections().
        
        Creates publication-quality 3D visualization of hydrogen bonds with:
        - Van der Waals-sized atomic spheres
        - Clean white background  
        - Side-by-side panels for multiple H-bonds
        - **CLEAN VIEW: Shows ONLY api (CIP) and MMT atoms - no waters, no other molecules**
        - Molecular bonds for complete structure connectivity
        - Si-Si connectivity showing clay tetrahedral/ditrigonal structure
        - Gold H-bond lines
        - Professional styling
        
        This method matches the aesthetic of the plot_clean_cross_sections() 
        visualization with atoms shown as spheres sized by their van der Waals radii,
        proper colors, and clean presentation.
        
        Parameters
        ----------
        hbond_results : dict
            Results from density_scatter.compute_hydrogen_bonds()
        cluster_id : int
            Cluster to visualize
        max_hbonds : int, default=4
            Maximum number of H-bonds to show (side-by-side panels)
        min_hbonds : int, default=1
            Minimum number of valid H-bonds required to create visualization.
            If fewer than min_hbonds are found with frames meeting cutoff criteria,
            the visualization is skipped. Set to max_hbonds (e.g., 4) to require
            exactly that many valid H-bonds, or set to 1 to allow partial visualizations.
        sort_by : str, default='occupancy'
            How to select H-bonds: 'occupancy', 'lifetime', 'random'
        prioritise_O : bool, default=False
            **Prioritize carboxylic acid O→Ob H-bonds**
            When True, actively searches for frames where O1/O2 (carboxylic acid) 
            donates H-bonds to Ob (basal oxygen). Ensures at least one O→Ob H-bond
            is visualized if it exists in the data, even if occupancy is very low.
            If O1/O2→Ob not found at specified cutoff, searches other cutoffs.
            Useful for rare but chemically important H-bond interactions.
        distance_cutoff : float, optional
            For multi-cutoff data: which distance cutoff to use (e.g., 2.5)
            If None, uses first available cutoff or searches for O1/O2 if prioritise_O=True
        angle_cutoff : float, optional
            For multi-cutoff data: which angle cutoff to use (e.g., 150.0)
            If None, uses first available cutoff or searches for O1/O2 if prioritise_O=True
        search_distance_cutoff : float or list of float, optional
            Distance threshold(s) for frame search validation (Å)
            Controls what H-A distance is considered valid when searching frames
            
            - Single value: 2.5 (uses this distance)
            - List: [2.0, 2.5, 3.0] (tries in order: strictest to most lenient)
            
            If None, uses the detection distance_cutoff or defaults to 2.5Å
        search_angle_cutoff : float or list of float, optional
            Angle threshold(s) for frame search validation (degrees)
            Controls what D-H-A angle is considered valid when searching frames
            
            - Single value: 150.0 (uses this angle)
            - List: [160.0, 150.0, 140.0] (tries in order: strictest to most lenient)
            
            If None, uses the detection angle_cutoff or defaults to 120°
        show_surface_atoms : bool, default=True
            Show clay surface atoms around H-bond site (MMT atoms only)
        surface_radius : float, default=8.0
            **Adjustable XY-plane radius** around H-bond acceptor to show surface atoms (Å)
            Controls how much of the clay surface is visible around the H-bond site
            Try values: 6.0 (tight), 8.0 (default), 10.0 (wide), 12.0 (very wide)
        surface_z_thickness : float, default=10.0
            **Z-direction thickness** to limit surface selection (Å)
            Prevents showing the opposite clay sheet - only shows atoms within ±z_thickness of acceptor
            Clay sheets are ~6-7Å thick, so 10Å ensures we only get the relevant sheet
        surface_floor_value : float, default=2.0
            **Floor filtering** below Si plane (Å)
            Creates floor at (average_Si_z - floor_value), hides atoms below this level
            Prevents showing underlying layer atoms even if within Z-thickness
            Use 2.0 for standard filtering, 0.0 to disable, 3.0 for tighter filtering
        surface_ceiling_value : float, default=0.0
            **Ceiling filtering** above Si plane (Å)
            Creates ceiling at (average_Si_z + ceiling_value), hides atoms above this level
            Prevents showing overlying layer atoms even if within Z-thickness
            Use 0.0 to disable (default), 2.0 for standard filtering, 3.0 for tighter filtering
        surface_sel : str
            MDAnalysis selection for surface atoms (automatically filtered to MMT residues)
        show_surface_bonds : bool, default=True
            Show bonds between surface atoms in clay structure
        surface_bond_color : str, default='orange'
            Color for bonds between surface atoms
        surface_bond_linewidth : float, default=3.0
            Line width for bonds between surface atoms
        molecule_sel : str, optional
            Selection for full molecule context (e.g., 'resname api')
            If provided, shows ALL atoms in the same residue as donor
            (not limited by distance - shows complete molecular structure)
        molecule_radius : float, default=10.0
            [DEPRECATED/UNUSED] Kept for backward compatibility
            Molecule selection now shows entire residue, not distance-limited
        show_waters : bool, default=True
            [IGNORED] Waters are not displayed - visualization limited to api+MMT only
        water_radius : float, default=5.0
            [IGNORED] Waters are not displayed - visualization limited to api+MMT only
        hbond_line_color : str, default='gold'
            Color for H-bond line
        hbond_linewidth : float, default=4.0
            Width of H-bond line
        atom_scale_factor : float, default=100
            Scaling factor for atom sizes based on VdW radii
        boundary_linewidth : float, default=0.5
            Width for atom edge lines
        view_elevation : float, default=25
            Viewing angle elevation (degrees) - matches plot_clean_cross_sections()
        view_azimuth : float, default=45
            Viewing angle azimuth (degrees) - matches plot_clean_cross_sections()
        figsize_per_panel : float, default=5
            Width of each panel in inches
        dpi : int, default=300
            Resolution for saved figure
        save_path : str, optional
            Path to save figure
        save_combined_figure : bool, default=False
            Whether to save a combined figure with all panels
        save_individual_figures : bool, default=True
            Whether to save individual figures for each H-bond panel
        individual_figsize : tuple, default=(8, 6)
            Figure size for individual panel figures (width, height) in inches
        title_fontsize : int, default=14
            Font size for panel titles
        title_fontweight : str, default='bold'
            Font weight for panel titles ('normal', 'bold', 'heavy', etc.)
        label_fontsize : int, default=16
            Font size for axis labels
        label_fontweight : str, default='bold'
            Font weight for axis labels ('normal', 'bold', 'heavy', etc.)
        tick_fontsize : int, default=14
            Font size for axis tick labels
        hbond_linestyle : str, default='dashed'
            Line style for H-bond visualization: 'solid', 'dashed', 'dotted', 'dashdot'
        bond_style : str, default='solid'
            Line style for molecular bonds: 'solid', 'dashed', 'dotted'
        atom_edge_style : str, default='solid'
            Edge line style for atoms: 'solid', 'dashed', 'dotted'
        surface_atom_style : str, default='scatter'
            Style for surface atoms: 'scatter', 'wireframe'
        si_color : str, default='#F0C8A0'
            Color for silicon atoms. Default is light tan (#F0C8A0, standard CPK color).
            Options: '#F0C8A0' (tan), 'orange', 'gray', or any matplotlib color
        show_title : bool, default=True
            Whether to display panel titles
        axis_info : str, default='detailed'
            Axis display mode:
            - 'None': Hide all axis elements completely (cleanest presentation)
            - 'simple': Hide all axes completely (like plot_clean_cross_sections)
            - 'detailed': Show axes with Z-axis positioned on the left (min X corner)
        start_search_frame : int, default=1
            Starting frame number for H-bond search (1-based indexing)
            Controls where the frame search begins in the trajectory
        skip_when_searching : int, default=1
            Number of frames to skip when searching (stride/step size)
            - 1: Check every frame (no skipping)
            - 2: Check every other frame
            - 5: Check every 5th frame, etc.
        
        Returns
        -------
        fig : matplotlib Figure
        axes : array of matplotlib Axes3D
        
        Examples
        --------
        >>> # Custom search with single cutoffs
        >>> fig, axes = plotter.plot_hbond_geometry_3d(
        ...     hbond_results=hbond_cip_mmt,
        ...     cluster_id=0,
        ...     search_distance_cutoff=2.5,
        ...     search_angle_cutoff=140.0
        ... )
        
        >>> # Custom search with multiple cutoffs (uses most lenient)
        >>> fig, axes = plotter.plot_hbond_geometry_3d(
        ...     hbond_results=hbond_cip_mmt,
        ...     cluster_id=0,
        ...     search_distance_cutoff=[2.0, 2.5, 3.0],  # Uses 3.0Å (most lenient)
        ...     search_angle_cutoff=[160.0, 150.0, 140.0]  # Uses 140.0° (most lenient)
        ... )
        
        >>> # Ultra-clean presentation with no axes
        >>> fig, axes = plotter.plot_hbond_geometry_3d(
        ...     hbond_results=hbond_cip_mmt,
        ...     cluster_id=0,
        ...     axis_info='None',  # No axes at all
        ...     show_title=False,
        ...     save_path='hbond_minimal.png'
        ... )
        
        >>> # Publication-ready H-bonds with custom styles
        >>> fig, axes = plotter.plot_hbond_geometry_3d(
        ...     hbond_results=hbond_cip_mmt,
        ...     cluster_id=0,
        ...     max_hbonds=4,
        ...     axis_info='simple',
        ...     hbond_linestyle='dotted',
        ...     bond_style='dashed',
        ...     atom_edge_style='solid',
        ...     save_combined_figure=True,
        ...     show_title=False,  # Clean presentation
        ...     save_path='hbond_3d_styled.png'
        ... )
        
        >>> # With detailed axes and publication formatting
        >>> fig, axes = plotter.plot_hbond_geometry_3d(
        ...     hbond_results=hbond_cip_mmt,
        ...     cluster_id=0,
        ...     axis_info='detailed', 
        ...     molecule_sel='resname api',
        ...     label_fontsize=16,
        ...     label_fontweight='bold',
        ...     tick_fontsize=14,
        ...     save_individual_figures=True,
        ...     individual_figsize=(10, 8)
        ... )
        """
        from mpl_toolkits.mplot3d import Axes3D
        import MDAnalysis as mda
        
        print(f"\n{'='*70}")
        print(f"3D H-BOND GEOMETRY VISUALIZATION (Cross-Section Style)")
        print(f"{'='*70}")
        
        # Detect if multi-cutoff data
        first_key = list(hbond_results.keys())[0]
        is_multi_cutoff = isinstance(first_key, tuple)
        
        if is_multi_cutoff:
            # User specified cutoffs
            if distance_cutoff is not None and angle_cutoff is not None:
                cutoff_key = (distance_cutoff, angle_cutoff)
                if cutoff_key not in hbond_results:
                    available = list(hbond_results.keys())
                    raise ValueError(f"Cutoff ({distance_cutoff}, {angle_cutoff}) not found. Available: {available}")
                cluster_data = hbond_results[cutoff_key][cluster_id]
                print(f"Using specified cutoff: distance={cutoff_key[0]:.1f}Å, angle={cutoff_key[1]:.1f}°")
            
            # No cutoff specified: automatically select best one
            else:
                # Find cutoff with highest occupancy H-bonds
                best_cutoff = None
                best_score = -1
                
                for cutoff_key in hbond_results.keys():
                    test_data = hbond_results[cutoff_key][cluster_id]
                    n_hbonds = len(test_data['hbond_pairs'])
                    if n_hbonds == 0:
                        continue
                    
                    # Get max occupancy
                    max_occ = max(test_data['occupancy'].values()) if test_data['occupancy'] else 0.0
                    
                    # Score = n_hbonds * max_occupancy
                    score = n_hbonds * max_occ
                    
                    if score > best_score:
                        best_score = score
                        best_cutoff = cutoff_key
                
                cutoff_key = best_cutoff if best_cutoff else first_key
                cluster_data = hbond_results[cutoff_key][cluster_id]
                print(f"Auto-selected cutoff: distance={cutoff_key[0]:.1f}Å, angle={cutoff_key[1]:.1f}° ({len(cluster_data['hbond_pairs'])} H-bonds)")
        else:
            cluster_data = hbond_results[cluster_id]
            cutoff_key = None
        
        # Extract H-bond information
        hbond_pairs = cluster_data['hbond_pairs']
        occupancy = cluster_data['occupancy']
        lifetimes = cluster_data['lifetimes']
        
        n_hbonds = len(hbond_pairs)
        print(f"Cluster {cluster_id}: {n_hbonds} unique H-bonds")
        
        if n_hbonds == 0:
            print("⚠️  No H-bonds to visualize!")
            return None, None
        
        # Select and sort H-bonds
        if sort_by == 'occupancy':
            sorted_pairs = []
            for pair_info in hbond_pairs:
                donor_res, donor_name, donor_idx, h_name, h_idx, acc_res, acc_name, acc_idx = pair_info
                occ = occupancy.get((donor_idx, h_idx, acc_idx), 0.0)
                sorted_pairs.append((pair_info, occ))
            sorted_pairs.sort(key=lambda x: x[1], reverse=True)
            selected_pairs = [p[0] for p in sorted_pairs[:max_hbonds]]
        elif sort_by == 'lifetime':
            sorted_pairs = []
            for pair_info in hbond_pairs:
                donor_res, donor_name, donor_idx, h_name, h_idx, acc_res, acc_name, acc_idx = pair_info
                lts = lifetimes.get((donor_idx, h_idx, acc_idx), [0])
                mean_lt = np.mean(lts)
                sorted_pairs.append((pair_info, mean_lt))
            sorted_pairs.sort(key=lambda x: x[1], reverse=True)
            selected_pairs = [p[0] for p in sorted_pairs[:max_hbonds]]
        else:
            import random
            selected_pairs = random.sample(hbond_pairs, min(max_hbonds, n_hbonds))
        
        n_selected = len(selected_pairs)
        print(f"Displaying {n_selected} H-bonds")
        
        # ⭐ PRIORITIZE O1/O2 DONORS if requested
        if prioritise_O:
            print(f"\n{'='*70}")
            print(f"PRIORITIZING CARBOXYLIC ACID O→Ob H-BONDS")
            print(f"{'='*70}")
            
            # Find O1/O2 donors in the H-bond data
            o_donors = [pair for pair in hbond_pairs 
                       if pair[1] in ['O1', 'O2'] and pair[6] == 'Ob']
            
            if len(o_donors) > 0:
                print(f"✓ Found {len(o_donors)} O1/O2→Ob H-bonds in data")
                
                # Check if any O1/O2 donors are already in selected_pairs
                already_has_o = any(pair[1] in ['O1', 'O2'] for pair in selected_pairs)
                
                if already_has_o:
                    print(f"✓ O1/O2→Ob H-bond already in selection")
                else:
                    print(f"⚠️  No O1/O2→Ob in current selection - adding prioritized O→Ob H-bond")
                    
                    # Sort O donors by occupancy to get the most common one
                    o_sorted = []
                    for pair_info in o_donors:
                        donor_res, donor_name, donor_idx, h_name, h_idx, acc_res, acc_name, acc_idx = pair_info
                        occ = occupancy.get((donor_idx, h_idx, acc_idx), 0.0)
                        o_sorted.append((pair_info, occ))
                    o_sorted.sort(key=lambda x: x[1], reverse=True)
                    
                    # Replace last selected pair with highest occupancy O→Ob
                    best_o_pair = o_sorted[0][0]
                    best_o_occ = o_sorted[0][1]
                    
                    if n_selected >= max_hbonds:
                        # Replace the lowest priority H-bond with O→Ob
                        selected_pairs[-1] = best_o_pair
                        print(f"   Replaced last H-bond with O→Ob (occupancy: {best_o_occ:.2%})")
                    else:
                        # Add O→Ob to selection
                        selected_pairs.append(best_o_pair)
                        n_selected += 1
                        print(f"   Added O→Ob H-bond (occupancy: {best_o_occ:.2%})")
                    
                    donor_name = best_o_pair[1]
                    acc_name = best_o_pair[6]
                    print(f"   {donor_name} → {acc_name}")
            else:
                print(f"⚠️  No O1/O2→Ob H-bonds found in this cluster's data")
                print(f"   (O1/O2 may not be donating to Ob at these cutoff criteria)")
            
            print(f"{'='*70}\n")
        
        # Get universe
        if not hasattr(self.analyzer, 'trajectory_data') or cluster_id not in self.analyzer.trajectory_data:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        u = self.analyzer.trajectory_data[cluster_id]['universe']
        frames = self.analyzer.trajectory_data[cluster_id]['frames']
        
        print(f"Searching frames to validate H-bond geometry (distance AND angle)...")
        print(f"Frame search: starting from frame {start_search_frame}, step size {skip_when_searching}")
        
        # Convert to 0-based indexing and create search frame list
        start_idx = max(0, start_search_frame - 1)  # Convert 1-based to 0-based
        search_frames = frames[start_idx::skip_when_searching]  # Apply start and skip
        print(f"Effective search frames: {len(search_frames)} frames (from {len(frames)} total)")
        
        # VdW radii
        vdw_radii = {
            'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
            'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80,
            'S': 1.80, 'Cl': 1.75, 'K': 2.75, 'Ca': 2.31
        }
        
        # ✅ FRAME SEARCH: Manual scanning required
        # compute_hydrogen_bonds() doesn't return per-pair frame lists (only occupancy/lifetimes)
        # So we need to search frames and verify BOTH distance AND angle criteria match the cutoff
        
        # Import distance_array for PBC-aware distance calculations
        from MDAnalysis.lib.distances import distance_array
        
        # ✅ PRE-VALIDATION: Find which H-bonds actually have valid frames
        print(f"\n🔍 Pre-validating H-bonds to find frames meeting cutoff criteria...")
        
        valid_hbonds = []  # List of (pair_info, found_frame)
        
        # Store cutoff values for frame search validation
        # Handle lists: convert to single values using most lenient criteria
        if search_distance_cutoff is not None:
            if isinstance(search_distance_cutoff, list):
                final_search_distance_cutoff = max(search_distance_cutoff)  # Most lenient
                print(f"Using custom search distance cutoffs: {search_distance_cutoff} → {final_search_distance_cutoff:.1f}Å (most lenient)")
            else:
                final_search_distance_cutoff = search_distance_cutoff
                print(f"Using custom search distance cutoff: {final_search_distance_cutoff:.1f}Å")
        elif is_multi_cutoff and cutoff_key is not None:
            final_search_distance_cutoff = cutoff_key[0]
            print(f"Using H-bond detection distance cutoff: {final_search_distance_cutoff:.1f}Å")
        else:
            # For single-cutoff data, use from cluster_data or default
            final_search_distance_cutoff = cluster_data.get('distance_cutoff', 2.5)
            print(f"Using stored/default distance cutoff: {final_search_distance_cutoff:.1f}Å")
        
        if search_angle_cutoff is not None:
            if isinstance(search_angle_cutoff, list):
                final_search_angle_cutoff = min(search_angle_cutoff)  # Most lenient (lower angle threshold)
                print(f"Using custom search angle cutoffs: {search_angle_cutoff} → {final_search_angle_cutoff:.1f}° (most lenient)")
            else:
                final_search_angle_cutoff = search_angle_cutoff
                print(f"Using custom search angle cutoff: {final_search_angle_cutoff:.1f}°")
        elif is_multi_cutoff and cutoff_key is not None:
            final_search_angle_cutoff = cutoff_key[1]
            print(f"Using H-bond detection angle cutoff: {final_search_angle_cutoff:.1f}°")
        else:
            # For single-cutoff data, use from cluster_data or default
            final_search_angle_cutoff = cluster_data.get('angle_cutoff', 120.0)
            print(f"Using stored/default angle cutoff: {final_search_angle_cutoff:.1f}°")
        
        print(f"Frame search criteria: distance ≤ {final_search_distance_cutoff:.1f}Å AND angle ≥ {final_search_angle_cutoff:.1f}°")
        
        for pair_info in selected_pairs:
            donor_res, donor_name, donor_idx, h_name, h_idx, acc_res, acc_name, acc_idx = pair_info
            
            # Special handling for O1/O2: search ALL frames
            if prioritise_O and donor_name in ['O1', 'O2']:
                search_frame_list = frames  # Search ALL frames for O1/O2
            else:
                search_frame_list = search_frames  # Use sampled frames for others
            
            found_frame = None
            for frame_idx in search_frame_list:
                u.trajectory[frame_idx]
                
                donor_atom = u.atoms[donor_idx]
                h_atom = u.atoms[h_idx]
                acc_atom = u.atoms[acc_idx]
                
                h_pos = h_atom.position
                acc_pos = acc_atom.position
                donor_pos = donor_atom.position
                box = u.dimensions
                
                # Check distance
                dist_ha = distance_array(h_pos.reshape(1, 3), acc_pos.reshape(1, 3), box=box)[0][0]
                if dist_ha > final_search_distance_cutoff:  # Use final_ prefix
                    continue
                
                # Check angle
                vec_hd = donor_pos - h_pos
                vec_ha = acc_pos - h_pos
                norm_hd = np.linalg.norm(vec_hd)
                norm_ha = np.linalg.norm(vec_ha)
                
                if norm_hd > 0 and norm_ha > 0:
                    cos_angle = np.dot(vec_hd, vec_ha) / (norm_hd * norm_ha)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_dha = np.degrees(np.arccos(cos_angle))
                    
                    if angle_dha >= final_search_angle_cutoff:  # Use final_ prefix
                        found_frame = frame_idx
                        break
            
            if found_frame is not None:
                valid_hbonds.append((pair_info, found_frame))
        
        print(f"   Found {len(valid_hbonds)} H-bonds with valid frames (from {len(selected_pairs)} selected)")
        
        # If no valid O-Ob H-bonds found, try to add N-Ob H-bonds
        if len(valid_hbonds) == 0 or (prioritise_O and not any(p[0][1] in ['O1', 'O2'] for p in valid_hbonds)):
            print(f"\n⚠️  No O→Ob H-bonds with valid frames found")
            print(f"   Searching for N→Ob H-bonds instead...")
            
            # Get N-Ob H-bonds from full list
            n_hbonds = [pair for pair in hbond_pairs 
                       if pair[1].startswith('N') and pair[6] == 'Ob']
            
            if len(n_hbonds) > 0:
                print(f"   Found {len(n_hbonds)} N→Ob H-bonds in data")
                
                # Sort by occupancy
                n_sorted = []
                for pair_info in n_hbonds:
                    donor_res, donor_name, donor_idx, h_name, h_idx, acc_res, acc_name, acc_idx = pair_info
                    occ = occupancy.get((donor_idx, h_idx, acc_idx), 0.0)
                    n_sorted.append((pair_info, occ))
                n_sorted.sort(key=lambda x: x[1], reverse=True)
                
                # Validate top N-Ob H-bonds
                for pair_info, occ in n_sorted[:max_hbonds]:
                    donor_res, donor_name, donor_idx, h_name, h_idx, acc_res, acc_name, acc_idx = pair_info
                    
                    # Check if already in valid list
                    if any(p[0] == pair_info for p in valid_hbonds):
                        continue
                    
                    # Search for valid frame
                    found_frame = None
                    for frame_idx in search_frames:
                        u.trajectory[frame_idx]
                        
                        donor_atom = u.atoms[donor_idx]
                        h_atom = u.atoms[h_idx]
                        acc_atom = u.atoms[acc_idx]
                        
                        h_pos = h_atom.position
                        acc_pos = acc_atom.position
                        donor_pos = donor_atom.position
                        box = u.dimensions
                        
                        dist_ha = distance_array(h_pos.reshape(1, 3), acc_pos.reshape(1, 3), box=box)[0][0]
                        if dist_ha > final_search_distance_cutoff:  # Use final_ prefix
                            continue
                        
                        vec_hd = donor_pos - h_pos
                        vec_ha = acc_pos - h_pos
                        norm_hd = np.linalg.norm(vec_hd)
                        norm_ha = np.linalg.norm(vec_ha)
                        
                        if norm_hd > 0 and norm_ha > 0:
                            cos_angle = np.dot(vec_hd, vec_ha) / (norm_hd * norm_ha)
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle_dha = np.degrees(np.arccos(cos_angle))
                            
                            if angle_dha >= final_search_angle_cutoff:  # Use final_ prefix
                                found_frame = frame_idx
                                break
                    
                    if found_frame is not None:
                        valid_hbonds.append((pair_info, found_frame))
                        print(f"   ✓ Added N→Ob: {donor_name}→Ob (occupancy: {occ:.2%})")
                        
                        if len(valid_hbonds) >= max_hbonds:
                            break
                
                print(f"   Total valid H-bonds after adding N→Ob: {len(valid_hbonds)}")
        
        # Check if we have enough valid H-bonds
        if len(valid_hbonds) == 0:
            print(f"\n⚠️  No H-bonds with frames meeting cutoff criteria found!")
            print(f"   Occupancies may be too low, or cutoff too strict")
            print(f"   Try: looser cutoffs, increase skip_when_searching=1, or check trajectory")
            return None, None
        
        if len(valid_hbonds) < min_hbonds:
            print(f"\n⚠️  Found only {len(valid_hbonds)} valid H-bonds (minimum required: {min_hbonds})")
            print(f"   Not enough valid H-bonds to create meaningful visualization")
            print(f"   Try: looser cutoffs, decrease min_hbonds, or check trajectory")
            return None, None
        
        n_valid = len(valid_hbonds)
        print(f"\n✅ Proceeding with {n_valid} valid H-bonds (min: {min_hbonds}, max: {max_hbonds})\n")
        
        # Create figure with correct number of panels
        fig = plt.figure(figsize=(figsize_per_panel * n_valid, 6))
        fig.patch.set_facecolor('white')
        
        axes = []
        
        # Track found donor types for diversity across panels
        found_donor_types = set()
        
        # Convert style names to matplotlib linestyle codes
        style_map = {
            'solid': '-',
            'dashed': '--', 
            'dotted': ':',
            'dashdot': '-.',
            'none': 'None'
        }
        
        hbond_ls = style_map.get(hbond_linestyle, '--')  # Default to dashed
        bond_ls = style_map.get(bond_style, '-')         # Default to solid
        edge_ls = style_map.get(atom_edge_style, '-')    # Default to solid
        
        # Main rendering loop - use pre-validated H-bonds
        for i, (pair_info, found_frame) in enumerate(valid_hbonds):
            ax = fig.add_subplot(1, n_valid, i+1, projection='3d')
            axes.append(ax)
            
            # Extract H-bond atoms
            donor_res, donor_name, donor_idx, h_name, h_idx, acc_res, acc_name, acc_idx = pair_info
            
            # Calculate stats for this H-bond
            occ = occupancy.get((donor_idx, h_idx, acc_idx), 0.0)
            lts = lifetimes.get((donor_idx, h_idx, acc_idx), [0])
            mean_lt = np.mean(lts)
            
            # Track donor type
            donor_atom_temp = u.atoms[donor_idx]
            donor_element = donor_atom_temp.element if (hasattr(donor_atom_temp, 'element') and donor_atom_temp.element) else donor_name[0]
            found_donor_types.add(donor_element.upper())
            
            # Set trajectory to the pre-found valid frame
            u.trajectory[found_frame]
            
            donor_atom = u.atoms[donor_idx]
            h_atom = u.atoms[h_idx]
            acc_atom = u.atoms[acc_idx]
            
            donor_pos = donor_atom.position
            h_pos = h_atom.position
            acc_pos = acc_atom.position
            
            # ✅ Calculate H-bond geometry with PBC corrections at the found frame
            # Get box dimensions for periodic boundary conditions
            box = u.dimensions  # [a, b, c, alpha, beta, gamma]
            box_lengths = box[:3]  # a, b, c
            
            # Calculate H···A distance with PBC (same as H-bond computation method)
            dist_ha = distance_array(h_pos.reshape(1, 3), acc_pos.reshape(1, 3), box=box)[0][0]
            
            # ✅ SIMPLIFIED UNWRAPPING - Use acceptor as central reference
            # Keep the acceptor in its original position, unwrap everything else relative to it
            def unwrap_position(pos, ref_pos, box_lengths):
                """Unwrap position relative to reference using minimum image convention"""
                delta = pos - ref_pos
                delta = delta - box_lengths * np.round(delta / box_lengths)
                return ref_pos + delta
            
            # Strategy: Use acceptor as reference - it stays in its original clay sheet position
            # This prevents the acceptor from being displaced to wrong locations (like Si ring centers)
            
            # Acceptor stays in original position (no unwrapping)
            acc_pos_unwrapped = acc_pos.copy()
            
            # Unwrap hydrogen and donor relative to acceptor
            h_pos_unwrapped = unwrap_position(h_pos, acc_pos_unwrapped, box_lengths)
            donor_pos_unwrapped = unwrap_position(donor_pos, acc_pos_unwrapped, box_lengths)
            
            print(f"    Using acceptor as reference: ({acc_pos_unwrapped[0]:.1f}, {acc_pos_unwrapped[1]:.1f}, {acc_pos_unwrapped[2]:.1f})")
            print(f"    H-bond geometry: H=({h_pos_unwrapped[0]:.1f}, {h_pos_unwrapped[1]:.1f}, {h_pos_unwrapped[2]:.1f}) → A=({acc_pos_unwrapped[0]:.1f}, {acc_pos_unwrapped[1]:.1f}, {acc_pos_unwrapped[2]:.1f})")
            
            # ✅ Calculate H-bond geometry with PBC corrections at the found frame
            # Use unwrapped positions for accurate geometry calculations
            dist_ha = distance_array(h_pos_unwrapped.reshape(1, 3), acc_pos_unwrapped.reshape(1, 3), box=box)[0][0]
            
            # Calculate D-H-A angle with PBC-corrected vectors
            vec_hd = donor_pos_unwrapped - h_pos_unwrapped  # H → D (pointing back to donor)
            vec_ha = acc_pos_unwrapped - h_pos_unwrapped    # H → A (pointing to acceptor)
            
            # Calculate D-H-A angle (180° = linear)
            norm_hd = np.linalg.norm(vec_hd)
            norm_ha = np.linalg.norm(vec_ha)
            if norm_hd > 0 and norm_ha > 0:
                cos_angle = np.dot(vec_hd, vec_ha) / (norm_hd * norm_ha)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_dha = np.degrees(np.arccos(cos_angle))
            else:
                angle_dha = 0.0
            
            print(f"\n  H-bond {i+1}: {donor_res}:{donor_name}-{h_name}···{acc_res}:{acc_name}")
            print(f"    Frame: {found_frame} | Distance: {dist_ha:.2f} Å, Angle: {angle_dha:.1f}°")
            print(f"    d={dist_ha:.2f}Å, ang={angle_dha:.1f}°, frm={found_frame}")
            print(f"    Occupancy: {occ*100:.1f}%, Lifetime: {mean_lt:.1f} frames")
            
            # SHOW SURFACE ATOMS (MMT only)
            if show_surface_atoms:
                try:
                    # ✅ Only show MMT clay atoms from the SAME SURFACE as acceptor
                    # Use Z-filtering to avoid showing the opposite clay sheet
                    
                    # Get acceptor Z position (stays in original position)
                    acc_z = acc_pos_unwrapped[2]
                    
                    # Initial selection with XY radius and Z-thickness
                    z_min = acc_z - surface_z_thickness
                    z_max = acc_z + surface_z_thickness
                    
                    initial_surface_atoms = u.select_atoms(
                        f'resname MMT and ({surface_sel}) and '
                        f'(sphzone {surface_radius} index {acc_idx}) and '
                        f'(prop z > {z_min} and prop z < {z_max})'
                    )
                    
                    # ✅ FLOOR FILTERING: Use Si atoms plane as reference
                    if surface_floor_value > 0 and len(initial_surface_atoms) > 0:
                        # Find Si atoms to determine the silicate sheet plane
                        si_atoms_in_selection = initial_surface_atoms.select_atoms('name Si')
                        
                        if len(si_atoms_in_selection) > 0:
                            # Calculate average Z of Si atoms (tetrahedral sheet level)
                            # Unwrap Si positions relative to acceptor for consistency
                            si_positions = np.array([unwrap_position(si_atom.position, acc_pos_unwrapped, box_lengths) 
                                                   for si_atom in si_atoms_in_selection])
                            avg_si_z = np.mean(si_positions[:, 2])
                            
                            # Set floor: avg_si_z - floor_value
                            z_floor = avg_si_z - surface_floor_value
                            
                            # Filter atoms above floor
                            filtered_indices = []
                            for atom in initial_surface_atoms:
                                atom_pos_unwrapped = unwrap_position(atom.position, acc_pos_unwrapped, box_lengths)
                                if atom_pos_unwrapped[2] >= z_floor:
                                    filtered_indices.append(atom.index)
                            
                            surface_atoms = u.atoms[filtered_indices]
                            print(f"    Surface atoms (MMT, z>{z_floor:.1f}Å, Si_avg={avg_si_z:.1f}Å): {len(surface_atoms)}")
                        else:
                            # No Si atoms found, use initial selection
                            surface_atoms = initial_surface_atoms
                            print(f"    Surface atoms (MMT, no Si reference): {len(surface_atoms)}")
                    else:
                        # Floor filtering disabled
                        surface_atoms = initial_surface_atoms
                        print(f"    Surface atoms (MMT, z={acc_z:.1f}±{surface_z_thickness:.1f}Å): {len(surface_atoms)}")
                    
                    # ✅ CEILING FILTERING: Use Si atoms plane as reference
                    if surface_ceiling_value > 0 and len(surface_atoms) > 0:
                        # Find Si atoms to determine the silicate sheet plane
                        si_atoms_in_selection = surface_atoms.select_atoms('name Si')
                        
                        if len(si_atoms_in_selection) > 0:
                            # Calculate average Z of Si atoms (tetrahedral sheet level)
                            # Unwrap Si positions relative to acceptor for consistency
                            si_positions = np.array([unwrap_position(si_atom.position, acc_pos_unwrapped, box_lengths) 
                                                   for si_atom in si_atoms_in_selection])
                            avg_si_z = np.mean(si_positions[:, 2])
                            
                            # Set ceiling: avg_si_z + ceiling_value
                            z_ceiling = avg_si_z + surface_ceiling_value
                            
                            # Filter atoms below ceiling
                            filtered_indices = []
                            for atom in surface_atoms:
                                atom_pos_unwrapped = unwrap_position(atom.position, acc_pos_unwrapped, box_lengths)
                                if atom_pos_unwrapped[2] <= z_ceiling:
                                    filtered_indices.append(atom.index)
                            
                            surface_atoms = u.atoms[filtered_indices]
                            print(f"    Surface ceiling filtering: keeping atoms below z={z_ceiling:.1f}Å (Si_avg + {surface_ceiling_value:.1f})")
                            print(f"    Surface atoms after ceiling filtering: {len(surface_atoms)}")
                        else:
                            # No Si atoms found, skip ceiling filtering
                            print(f"    Surface atoms (no Si for ceiling filtering): {len(surface_atoms)}")
                    
                    # Store surface atom positions for bond drawing
                    surface_atoms_pos = {}
                    si_atoms_pos = []
                    
                    for surf_atom in surface_atoms:
                        # Skip the H-bond acceptor itself (it's rendered separately as prominent H-bond atom)
                        if surf_atom.index == acc_idx:
                            continue
                        
                        # Unwrap surface atom position relative to acceptor
                        surf_pos = unwrap_position(surf_atom.position, acc_pos_unwrapped, box_lengths)
                        surface_atoms_pos[surf_atom.index] = surf_pos
                        
                        # Extract element - check if element attribute exists AND is not empty
                        element = surf_atom.element if (hasattr(surf_atom, 'element') and surf_atom.element) else surf_atom.name[0]
                        radius = vdw_radii.get(element, 1.5)
                        size = (radius ** 2) * atom_scale_factor * 0.3
                        
                        # Color by atom type
                        if 'Si' in surf_atom.name:
                            color = si_color
                            alpha = 0.5
                            si_atoms_pos.append((surf_atom.index, surf_pos))
                        elif 'Mg' in surf_atom.name:
                            color = 'darkgreen'
                            alpha = 0.4
                        elif 'O' in surf_atom.name:
                            color = 'red'  # ✅ Using water oxygen color (red) for Ob atoms
                            alpha = 0.5
                        else:
                            color = 'gray'
                            alpha = 0.3
                        
                        ax.scatter(surf_pos[0], surf_pos[1], surf_pos[2],
                                 c=color, s=size, alpha=alpha, edgecolors='black',
                                 linewidth=boundary_linewidth, zorder=1)
                    
                    # ✅ DRAW Si-Si CONNECTIVITY (ditrigonal/tetrahedral structure)
                    # Connect Si atoms within ~4.5 Å (typical Si-Si distance in clay)
                    if show_surface_bonds:
                        si_connections = 0
                        for i in range(len(si_atoms_pos)):
                            for j in range(i+1, len(si_atoms_pos)):
                                idx1, pos1 = si_atoms_pos[i]
                                idx2, pos2 = si_atoms_pos[j]
                                dist = np.linalg.norm(pos1 - pos2)
                                
                                if dist < 4.5:  # Si-Si connectivity threshold
                                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                                           color=surface_bond_color, linewidth=surface_bond_linewidth, 
                                           alpha=0.4, linestyle='-', zorder=0)
                                    si_connections += 1
                        
                        if si_connections > 0:
                            print(f"    Si connections: {si_connections}")
                    
                except Exception as e:
                    print(f"    ⚠️  Could not load surface atoms: {e}")
            
            # ✅ WATERS REMOVED - Visualization limited to api (CIP) + MMT only
            
            # SHOW MOLECULE CONTEXT
            if molecule_sel:
                try:
                    # ✅ FIXED: Select ENTIRE molecule by residue, not just nearby atoms
                    # Get the residue of the donor atom
                    donor_resid = donor_atom.resid
                    donor_resname = donor_atom.resname
                    
                    # Select all atoms in the same residue
                    mol_atoms = u.select_atoms(f'{molecule_sel} and resid {donor_resid}')
                    
                    # Exclude the H-bond atoms themselves
                    mol_atoms = mol_atoms - donor_atom - h_atom
                    print(f"    Molecule atoms: {len(mol_atoms)} (residue {donor_resname} {donor_resid})")
                    
                    # Store unwrapped positions for bond drawing
                    mol_atoms_pos_unwrapped = {}
                    
                    for mol_atom in mol_atoms:
                        # Unwrap molecule atom relative to donor
                        mol_pos = unwrap_position(mol_atom.position, donor_pos_unwrapped, box_lengths)
                        mol_atoms_pos_unwrapped[mol_atom.index] = mol_pos
                        
                        # Extract element - check if element attribute exists AND is not empty
                        element = mol_atom.element if (hasattr(mol_atom, 'element') and mol_atom.element) else mol_atom.name[0]
                        radius = vdw_radii.get(element, 1.5)
                        size = (radius ** 2) * atom_scale_factor * 0.4
                        
                        # Standard CPK colors
                        color_map = {
                            'C': carbon_color,
                            'N': 'blue',
                            'O': 'red',
                            'H': 'lightcyan',  # ✅ Light cyan for hydrogens on CIP
                            'F': 'lightgreen',
                            'S': 'yellow',
                            'P': 'orange'
                        }
                        color = color_map.get(element, 'gray')
                        
                        ax.scatter(mol_pos[0], mol_pos[1], mol_pos[2],
                                 c=color, s=size, alpha=molecule_alpha, edgecolors='black',
                                 linewidth=boundary_linewidth, zorder=3)
                    
                    # ✅ DRAW MOLECULAR BONDS (using MDAnalysis topology + distance-based fallback)
                    # Add donor atom to position dict
                    mol_atoms_pos_unwrapped[donor_idx] = donor_pos_unwrapped
                    mol_atoms_pos_unwrapped[h_idx] = h_pos_unwrapped
                    
                    # Get all atoms including H-bond atoms for bond drawing
                    all_mol_atoms = mol_atoms + donor_atom + h_atom
                    
                    # STEP 1: Draw bonds from MDAnalysis topology
                    drawn_bonds = set()
                    topology_bonds = 0
                    
                    for atom in all_mol_atoms:
                        if hasattr(atom, 'bonds') and atom.bonds:
                            for bond in atom.bonds:
                                # Get bonded atom indices
                                idx1, idx2 = bond.atoms[0].index, bond.atoms[1].index
                                bond_key = tuple(sorted([idx1, idx2]))
                                
                                # Skip if already drawn
                                if bond_key in drawn_bonds:
                                    continue
                                
                                # Check if both atoms are in our molecule selection
                                if idx1 in mol_atoms_pos_unwrapped and idx2 in mol_atoms_pos_unwrapped:
                                    pos1 = mol_atoms_pos_unwrapped[idx1]
                                    pos2 = mol_atoms_pos_unwrapped[idx2]
                                    
                                    # Draw bond as line with configurable style
                                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                                           color='black', linewidth=1.5, linestyle=bond_ls, alpha=molecule_alpha, zorder=2)
                                    drawn_bonds.add(bond_key)
                                    topology_bonds += 1
                    
                    # STEP 2: Distance-based bond detection as fallback
                    # This catches missing bonds like cyclopropyl groups, aromatic rings, etc.
                    distance_bonds = self.detect_bonds_by_distance(all_mol_atoms, mol_atoms_pos_unwrapped)
                    fallback_bonds = 0
                    
                    # Draw bonds that were detected by distance but missing from topology
                    for bond_key in distance_bonds:
                        if bond_key not in drawn_bonds:
                            idx1, idx2 = bond_key
                            if idx1 in mol_atoms_pos_unwrapped and idx2 in mol_atoms_pos_unwrapped:
                                pos1 = mol_atoms_pos_unwrapped[idx1]
                                pos2 = mol_atoms_pos_unwrapped[idx2]
                                
                                # Draw fallback bond with slightly different style to distinguish
                                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                                       color='darkgray', linewidth=1.2, linestyle=bond_ls, alpha=molecule_alpha, zorder=2)
                                drawn_bonds.add(bond_key)
                                fallback_bonds += 1
                    
                    total_bonds = len(drawn_bonds)
                    if fallback_bonds > 0:
                        print(f"    Molecular bonds: {total_bonds} total ({topology_bonds} topology + {fallback_bonds} distance-detected)")
                        print(f"    ✓ Distance-based fallback caught {fallback_bonds} missing bonds (likely cyclopropyl/aromatic)")
                    else:
                        print(f"    Molecular bonds: {total_bonds} (all from topology)")
                    
                except Exception as e:
                    print(f"    ⚠️  Could not load molecule: {e}")
            
            # PLOT H-BOND ATOMS (PROMINENTLY) - Use unwrapped positions
            # Donor
            donor_element = donor_atom.element if (hasattr(donor_atom, 'element') and donor_atom.element) else donor_name[0]
            donor_radius = vdw_radii.get(donor_element, 1.5)
            donor_size = max(200, (donor_radius ** 2) * atom_scale_factor)
            donor_color = 'blue' if donor_element == 'N' else 'red'
            
            ax.scatter(donor_pos_unwrapped[0], donor_pos_unwrapped[1], donor_pos_unwrapped[2],
                      c=donor_color, s=donor_size, marker='o', alpha=0.9,
                      edgecolors='black', linewidth=1.5, zorder=10)
            
            # Hydrogen (use unwrapped position for consistency)
            h_radius = vdw_radii['H']
            h_size = (h_radius ** 2) * atom_scale_factor
            
            ax.scatter(h_pos_unwrapped[0], h_pos_unwrapped[1], h_pos_unwrapped[2],
                      c='white', s=h_size, marker='o', alpha=0.9,
                      edgecolors='gray', linewidth=1.5, zorder=10)
            
            # Acceptor (surface oxygen) - Use unwrapped position
            acc_radius = vdw_radii['O']
            acc_size = max(200, (acc_radius ** 2) * atom_scale_factor)
            
            ax.scatter(acc_pos_unwrapped[0], acc_pos_unwrapped[1], acc_pos_unwrapped[2],
                      c='red', s=acc_size, marker='o', alpha=0.9,
                      edgecolors='black', linewidth=1.5, zorder=10)
            
            # DRAW BONDS - Use unwrapped positions
            # D-H covalent bond (black solid)
            ax.plot([donor_pos_unwrapped[0], h_pos_unwrapped[0]], [donor_pos_unwrapped[1], h_pos_unwrapped[1]], 
                   [donor_pos_unwrapped[2], h_pos_unwrapped[2]],
                   color='black', linewidth=3, alpha=0.8, zorder=9)
            
            # H···A hydrogen bond with configurable style
            ax.plot([h_pos_unwrapped[0], acc_pos_unwrapped[0]], [h_pos_unwrapped[1], acc_pos_unwrapped[1]], 
                   [h_pos_unwrapped[2], acc_pos_unwrapped[2]],
                   color=hbond_line_color, linewidth=hbond_linewidth, linestyle=hbond_ls,
                   alpha=0.9, zorder=9)
            
            # FORMATTING - Handle different axis display modes
            if axis_info == 'None':
                # Completely remove all axis elements (cleanest presentation)
                ax.axis('off')  # Turn off all axis elements
                # Additional cleanup to ensure everything is hidden
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.grid(False)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.zaxis.set_visible(False)
                # Completely hide panes
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('none')
                ax.yaxis.pane.set_edgecolor('none')
                ax.zaxis.pane.set_edgecolor('none')
                ax.xaxis.pane.set_alpha(0)
                ax.yaxis.pane.set_alpha(0)
                ax.zaxis.pane.set_alpha(0)
                # Hide all lines
                ax.xaxis.line.set_color('none')
                ax.yaxis.line.set_color('none')
                ax.zaxis.line.set_color('none')
                # Remove labels
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_zlabel('')
            elif axis_info == 'simple':
                # Hide axes completely like plot_clean_cross_sections()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.grid(False)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.zaxis.set_visible(False)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('white')
                ax.yaxis.pane.set_edgecolor('white')
                ax.zaxis.pane.set_edgecolor('white')
                ax.xaxis.pane.set_alpha(0)
                ax.yaxis.pane.set_alpha(0)
                ax.zaxis.pane.set_alpha(0)
            else:
                # Detailed axis mode - show normal axes with publication-ready formatting
                ax.set_xlabel('X (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
                ax.set_ylabel('Y (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
                ax.set_zlabel('Z (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
                
                # Set tick label font sizes
                ax.tick_params(axis='x', labelsize=tick_fontsize)
                ax.tick_params(axis='y', labelsize=tick_fontsize)
                ax.tick_params(axis='z', labelsize=tick_fontsize)
                
                # ✅ Adjust Y-axis tick spacing (less tight than 2.5Å intervals)
                # Get current Y-limits and set ticks every 5Å instead of 2.5Å
                y_lims = ax.get_ylim()
                y_min, y_max = y_lims
                y_tick_spacing = 5.0  # 5Å intervals instead of 2.5Å
                y_ticks = np.arange(np.floor(y_min/y_tick_spacing)*y_tick_spacing,
                                   np.ceil(y_max/y_tick_spacing)*y_tick_spacing + y_tick_spacing,
                                   y_tick_spacing)
                ax.set_yticks(y_ticks)
                
                # ✅ CLEAN WHITE BACKGROUND (remove gray panes)
                ax.grid(False)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('white')
                ax.yaxis.pane.set_edgecolor('white')
                ax.zaxis.pane.set_edgecolor('white')
                ax.xaxis.pane.set_alpha(0)
                ax.yaxis.pane.set_alpha(0)
                ax.zaxis.pane.set_alpha(0)
            
            # Set viewing angle
            ax.view_init(elev=view_elevation, azim=view_azimuth)
            
            # Title with H-bond info (if enabled)
            if show_title:
                title = f'{donor_res}:{donor_name}-{h_name}···{acc_res}:{acc_name}\n'
                title += f'd={dist_ha:.2f}Å, ang={angle_dha:.1f}°, frm={found_frame}'
                # Replace 'api' with 'CIP' in title
                title = title.replace('api:', 'CIP:')
                ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight, pad=10)
            
            # Equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
        
        plt.tight_layout()
        
        # Save figures based on user preferences
        saved_files = []
        
        # Handle save_path logic: if individual figures requested but no save_path, generate default
        if save_individual_figures and save_path is None:
            save_path = 'hbond_3d_geometry'
        
        if save_path:
            if save_combined_figure:
                # Save combined figure with all panels
                combined_path = save_path
                if not combined_path.endswith(('.png', '.jpg', '.pdf', '.svg')):
                    combined_path += '_combined.png'
                fig.savefig(combined_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                saved_files.append(combined_path)
                print(f"\n✅ Combined figure saved: {combined_path}")
            
            if save_individual_figures:
                # Create and save individual figures for each H-bond panel
                print(f"\n📊 Creating individual figures...")
                base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
                
                for i, (pair_info, found_frame) in enumerate(valid_hbonds):
                    donor_res, donor_name, donor_idx, h_name, h_idx, acc_res, acc_name, acc_idx = pair_info
                    
                    # Create new figure for this individual H-bond
                    individual_fig = plt.figure(figsize=individual_figsize)
                    individual_fig.patch.set_facecolor('white')
                    individual_ax = individual_fig.add_subplot(111, projection='3d')
                    
                    print(f"  🔄 Re-rendering H-bond {i+1}: {donor_res}:{donor_name}-{h_name}···{acc_res}:{acc_name}")
                    print(f"    Using frame: {found_frame}")
                    
                    # Set trajectory to the pre-found valid frame
                    u.trajectory[found_frame]
                    
                    # Get atom positions
                    donor_atom = u.atoms[donor_idx]
                    h_atom = u.atoms[h_idx]
                    acc_atom = u.atoms[acc_idx]
                    
                    donor_pos = donor_atom.position
                    h_pos = h_atom.position
                    acc_pos = acc_atom.position
                    
                    # Calculate unwrapped positions (using acceptor as reference)
                    box_lengths = u.dimensions[:3]
                    acc_pos_unwrapped = acc_pos.copy()
                    h_pos_unwrapped = unwrap_position(h_pos, acc_pos_unwrapped, box_lengths)
                    donor_pos_unwrapped = unwrap_position(donor_pos, acc_pos_unwrapped, box_lengths)
                    
                    # Calculate H-bond geometry
                    dist_ha = distance_array(h_pos_unwrapped.reshape(1, 3), acc_pos_unwrapped.reshape(1, 3), box=u.dimensions)[0][0]
                    vec_hd = donor_pos_unwrapped - h_pos_unwrapped
                    vec_ha = acc_pos_unwrapped - h_pos_unwrapped
                    norm_hd = np.linalg.norm(vec_hd)
                    norm_ha = np.linalg.norm(vec_ha)
                    if norm_hd > 0 and norm_ha > 0:
                        cos_angle = np.dot(vec_hd, vec_ha) / (norm_hd * norm_ha)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle_dha = np.degrees(np.arccos(cos_angle))
                    else:
                        angle_dha = 0.0
                    
                    # Re-plot surface atoms
                    if show_surface_atoms:
                        try:
                            acc_z = acc_pos_unwrapped[2]
                            z_min = acc_z - surface_z_thickness
                            z_max = acc_z + surface_z_thickness
                            
                            initial_surface_atoms = u.select_atoms(
                                f'resname MMT and ({surface_sel}) and '
                                f'(sphzone {surface_radius} index {acc_idx}) and '
                                f'(prop z > {z_min} and prop z < {z_max})'
                            )
                            
                            # Apply floor filtering
                            if surface_floor_value > 0 and len(initial_surface_atoms) > 0:
                                si_atoms_in_selection = initial_surface_atoms.select_atoms('name Si')
                                if len(si_atoms_in_selection) > 0:
                                    si_positions = np.array([unwrap_position(si_atom.position, acc_pos_unwrapped, box_lengths) 
                                                           for si_atom in si_atoms_in_selection])
                                    avg_si_z = np.mean(si_positions[:, 2])
                                    z_floor = avg_si_z - surface_floor_value
                                    
                                    filtered_indices = []
                                    for atom in initial_surface_atoms:
                                        atom_pos_unwrapped = unwrap_position(atom.position, acc_pos_unwrapped, box_lengths)
                                        if atom_pos_unwrapped[2] >= z_floor:
                                            filtered_indices.append(atom.index)
                                    surface_atoms = u.atoms[filtered_indices]
                                else:
                                    surface_atoms = initial_surface_atoms
                            else:
                                surface_atoms = initial_surface_atoms
                            
                            # Apply ceiling filtering
                            if surface_ceiling_value > 0 and len(surface_atoms) > 0:
                                si_atoms_in_selection = surface_atoms.select_atoms('name Si')
                                if len(si_atoms_in_selection) > 0:
                                    si_positions = np.array([unwrap_position(si_atom.position, acc_pos_unwrapped, box_lengths) 
                                                           for si_atom in si_atoms_in_selection])
                                    avg_si_z = np.mean(si_positions[:, 2])
                                    z_ceiling = avg_si_z + surface_ceiling_value
                                    
                                    filtered_indices = []
                                    for atom in surface_atoms:
                                        atom_pos_unwrapped = unwrap_position(atom.position, acc_pos_unwrapped, box_lengths)
                                        if atom_pos_unwrapped[2] <= z_ceiling:
                                            filtered_indices.append(atom.index)
                                    surface_atoms = u.atoms[filtered_indices]
                            
                            # Plot surface atoms
                            si_atoms_pos = []
                            for surf_atom in surface_atoms:
                                if surf_atom.index == acc_idx:
                                    continue
                                
                                surf_pos = unwrap_position(surf_atom.position, acc_pos_unwrapped, box_lengths)
                                element = surf_atom.element if (hasattr(surf_atom, 'element') and surf_atom.element) else surf_atom.name[0]
                                radius = vdw_radii.get(element, 1.5)
                                size = (radius ** 2) * atom_scale_factor * 0.3
                                
                                if 'Si' in surf_atom.name:
                                    color = 'orange'
                                    alpha = 0.5
                                    si_atoms_pos.append((surf_atom.index, surf_pos))
                                elif 'Mg' in surf_atom.name:
                                    color = 'darkgreen'
                                    alpha = 0.4
                                elif 'O' in surf_atom.name:
                                    color = 'red'
                                    alpha = 0.5
                                else:
                                    color = 'gray'
                                    alpha = 0.3
                                
                                individual_ax.scatter(surf_pos[0], surf_pos[1], surf_pos[2],
                                                     c=color, s=size, alpha=alpha, edgecolors='black',
                                                     linewidth=boundary_linewidth, zorder=1)
                            
                            # Draw Si-Si connectivity (surface bonds)
                            if show_surface_bonds:
                                for j in range(len(si_atoms_pos)):
                                    for k in range(j+1, len(si_atoms_pos)):
                                        _, pos1 = si_atoms_pos[j]
                                        _, pos2 = si_atoms_pos[k]
                                        dist = np.linalg.norm(pos1 - pos2)
                                        if dist < 4.5:
                                            individual_ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                                                              color=surface_bond_color, linewidth=surface_bond_linewidth, 
                                                              alpha=0.4, linestyle='-', zorder=0)
                        except Exception as e:
                            print(f"    ⚠️  Could not load surface atoms: {e}")
                    
                    # Re-plot molecule context
                    if molecule_sel:
                        try:
                            donor_resid = donor_atom.resid
                            mol_atoms = u.select_atoms(f'{molecule_sel} and resid {donor_resid}')
                            mol_atoms = mol_atoms - donor_atom - h_atom
                            
                            mol_atoms_pos_unwrapped = {}
                            for mol_atom in mol_atoms:
                                mol_pos = unwrap_position(mol_atom.position, donor_pos_unwrapped, box_lengths)
                                mol_atoms_pos_unwrapped[mol_atom.index] = mol_pos
                                
                                element = mol_atom.element if (hasattr(mol_atom, 'element') and mol_atom.element) else mol_atom.name[0]
                                radius = vdw_radii.get(element, 1.5)
                                size = (radius ** 2) * atom_scale_factor * 0.4
                                
                                color_map = {'C': carbon_color, 'N': 'blue', 'O': 'red', 'H': 'lightcyan', 'F': 'lightgreen', 'S': 'yellow', 'P': 'orange'}
                                color = color_map.get(element, 'gray')
                                
                                individual_ax.scatter(mol_pos[0], mol_pos[1], mol_pos[2],
                                                     c=color, s=size, alpha=molecule_alpha, edgecolors='black',
                                                     linewidth=boundary_linewidth, zorder=3)
                            
                            # Draw molecular bonds (topology + distance-based fallback)
                            mol_atoms_pos_unwrapped[donor_idx] = donor_pos_unwrapped
                            mol_atoms_pos_unwrapped[h_idx] = h_pos_unwrapped
                            all_mol_atoms = mol_atoms + donor_atom + h_atom
                            
                            # STEP 1: Draw bonds from MDAnalysis topology
                            drawn_bonds = set()
                            
                            for atom in all_mol_atoms:
                                if hasattr(atom, 'bonds') and atom.bonds:
                                    for bond in atom.bonds:
                                        idx1, idx2 = bond.atoms[0].index, bond.atoms[1].index
                                        bond_key = tuple(sorted([idx1, idx2]))
                                        
                                        if bond_key in drawn_bonds:
                                            continue
                                        
                                        if idx1 in mol_atoms_pos_unwrapped and idx2 in mol_atoms_pos_unwrapped:
                                            pos1 = mol_atoms_pos_unwrapped[idx1]
                                            pos2 = mol_atoms_pos_unwrapped[idx2]
                                            
                                            individual_ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                                                              color='black', linewidth=1.5, linestyle=bond_ls, alpha=molecule_alpha, zorder=2)
                                            drawn_bonds.add(bond_key)
                            
                            # STEP 2: Distance-based bond detection as fallback
                            distance_bonds = self.detect_bonds_by_distance(all_mol_atoms, mol_atoms_pos_unwrapped)
                            
                            # Draw bonds that were detected by distance but missing from topology
                            for bond_key in distance_bonds:
                                if bond_key not in drawn_bonds:
                                    idx1, idx2 = bond_key
                                    if idx1 in mol_atoms_pos_unwrapped and idx2 in mol_atoms_pos_unwrapped:
                                        pos1 = mol_atoms_pos_unwrapped[idx1]
                                        pos2 = mol_atoms_pos_unwrapped[idx2]
                                        
                                        # Draw fallback bond with slightly different style
                                        individual_ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                                                          color='darkgray', linewidth=1.2, linestyle=bond_ls, alpha=molecule_alpha, zorder=2)
                                        drawn_bonds.add(bond_key)
                        except Exception as e:
                            print(f"    ⚠️  Could not load molecule: {e}")
                    
                    # Plot H-bond atoms
                    donor_element = donor_atom.element if (hasattr(donor_atom, 'element') and donor_atom.element) else donor_name[0]
                    donor_radius = vdw_radii.get(donor_element, 1.5)
                    donor_size = max(200, (donor_radius ** 2) * atom_scale_factor)
                    donor_color = 'blue' if donor_element == 'N' else 'red'
                    
                    individual_ax.scatter(donor_pos_unwrapped[0], donor_pos_unwrapped[1], donor_pos_unwrapped[2],
                                         c=donor_color, s=donor_size, marker='o', alpha=0.9,
                                         edgecolors='black', linewidth=1.5, zorder=10)
                    
                    h_radius = vdw_radii['H']
                    h_size = (h_radius ** 2) * atom_scale_factor
                    individual_ax.scatter(h_pos_unwrapped[0], h_pos_unwrapped[1], h_pos_unwrapped[2],
                                         c='white', s=h_size, marker='o', alpha=0.9,
                                         edgecolors='gray', linewidth=1.5, zorder=10)
                    
                    acc_radius = vdw_radii['O']
                    acc_size = max(200, (acc_radius ** 2) * atom_scale_factor)
                    individual_ax.scatter(acc_pos_unwrapped[0], acc_pos_unwrapped[1], acc_pos_unwrapped[2],
                                         c='red', s=acc_size, marker='o', alpha=0.9,
                                         edgecolors='black', linewidth=1.5, zorder=10)
                    
                    # Draw bonds
                    individual_ax.plot([donor_pos_unwrapped[0], h_pos_unwrapped[0]], [donor_pos_unwrapped[1], h_pos_unwrapped[1]], 
                                      [donor_pos_unwrapped[2], h_pos_unwrapped[2]],
                                      color='black', linewidth=3, alpha=0.8, zorder=9)
                    
                    individual_ax.plot([h_pos_unwrapped[0], acc_pos_unwrapped[0]], [h_pos_unwrapped[1], acc_pos_unwrapped[1]], 
                                      [h_pos_unwrapped[2], acc_pos_unwrapped[2]],
                                      color=hbond_line_color, linewidth=hbond_linewidth, linestyle=hbond_ls,
                                      alpha=0.9, zorder=9)
                    
                    # Apply axis formatting
                    if axis_info == 'None':
                        # Completely remove all axis elements (cleanest presentation)
                        individual_ax.axis('off')  # Turn off all axis elements
                        # Additional cleanup to ensure everything is hidden
                        individual_ax.set_xticks([])
                        individual_ax.set_yticks([])
                        individual_ax.set_zticks([])
                        individual_ax.grid(False)
                        individual_ax.xaxis.set_visible(False)
                        individual_ax.yaxis.set_visible(False)
                        individual_ax.zaxis.set_visible(False)
                        # Completely hide panes
                        individual_ax.xaxis.pane.fill = False
                        individual_ax.yaxis.pane.fill = False
                        individual_ax.zaxis.pane.fill = False
                        individual_ax.xaxis.pane.set_edgecolor('none')
                        individual_ax.yaxis.pane.set_edgecolor('none')
                        individual_ax.zaxis.pane.set_edgecolor('none')
                        individual_ax.xaxis.pane.set_alpha(0)
                        individual_ax.yaxis.pane.set_alpha(0)
                        individual_ax.zaxis.pane.set_alpha(0)
                        # Hide all lines
                        individual_ax.xaxis.line.set_color('none')
                        individual_ax.yaxis.line.set_color('none')
                        individual_ax.zaxis.line.set_color('none')
                        # Remove labels
                        individual_ax.set_xlabel('')
                        individual_ax.set_ylabel('')
                        individual_ax.set_zlabel('')
                    elif axis_info == 'simple':
                        individual_ax.set_xticks([])
                        individual_ax.set_yticks([])
                        individual_ax.set_zticks([])
                        individual_ax.grid(False)
                        individual_ax.xaxis.set_visible(False)
                        individual_ax.yaxis.set_visible(False)
                        individual_ax.zaxis.set_visible(False)
                        individual_ax.xaxis.pane.fill = False
                        individual_ax.yaxis.pane.fill = False
                        individual_ax.zaxis.pane.fill = False
                        individual_ax.xaxis.pane.set_edgecolor('white')
                        individual_ax.yaxis.pane.set_edgecolor('white')
                        individual_ax.zaxis.pane.set_edgecolor('white')
                        individual_ax.xaxis.pane.set_alpha(0)
                        individual_ax.yaxis.pane.set_alpha(0)
                        individual_ax.zaxis.pane.set_alpha(0)
                    else:
                        individual_ax.set_xlabel('X (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
                        individual_ax.set_ylabel('Y (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
                        individual_ax.set_zlabel('Z (Å)', fontsize=label_fontsize, fontweight=label_fontweight)
                        individual_ax.tick_params(axis='x', labelsize=tick_fontsize)
                        individual_ax.tick_params(axis='y', labelsize=tick_fontsize)
                        individual_ax.tick_params(axis='z', labelsize=tick_fontsize)
                        # Apply Y-axis tick spacing
                        y_lims = individual_ax.get_ylim()
                        y_min, y_max = y_lims
                        y_tick_spacing = 5.0
                        y_ticks = np.arange(np.floor(y_min/y_tick_spacing)*y_tick_spacing,
                                           np.ceil(y_max/y_tick_spacing)*y_tick_spacing + y_tick_spacing,
                                           y_tick_spacing)
                        individual_ax.set_yticks(y_ticks)
                        individual_ax.grid(False)
                        individual_ax.xaxis.pane.fill = False
                        individual_ax.yaxis.pane.fill = False
                        individual_ax.zaxis.pane.fill = False
                        individual_ax.xaxis.pane.set_edgecolor('white')
                        individual_ax.yaxis.pane.set_edgecolor('white')
                        individual_ax.zaxis.pane.set_edgecolor('white')
                        individual_ax.xaxis.pane.set_alpha(0)
                        individual_ax.yaxis.pane.set_alpha(0)
                        individual_ax.zaxis.pane.set_alpha(0)
                    
                    # Set viewing angle and aspect ratio
                    individual_ax.view_init(elev=view_elevation, azim=view_azimuth)
                    individual_ax.set_box_aspect([1, 1, 1])
                    
                    # Add title if enabled
                    if show_title:
                        occ = occupancy.get((donor_idx, h_idx, acc_idx), 0.0)
                        lts = lifetimes.get((donor_idx, h_idx, acc_idx), [0])
                        mean_lt = np.mean(lts)
                        
                        title = f'{donor_res}:{donor_name}-{h_name}···{acc_res}:{acc_name}\n'
                        title += f'd={dist_ha:.2f}Å, ang={angle_dha:.1f}°, frm={found_frame}'
                        # Replace 'api' with 'CIP' in title
                        title = title.replace('api:', 'CIP:')
                        individual_ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight, pad=10)
                    
                    # Build descriptive filename
                    individual_path = f"{base_path}_panel{i+1}_{donor_res}_{donor_name}_{acc_res}_{acc_name}.png"
                    
                    # Save individual figure
                    individual_fig.savefig(individual_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                    saved_files.append(individual_path)
                    print(f"  ✅ Panel {i+1} saved: {individual_path}")
                    
                    # Close individual figure to free memory
                    plt.close(individual_fig)
            
            # Default: save combined figure if neither option explicitly chosen
            if not save_combined_figure and not save_individual_figures:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                saved_files.append(save_path)
                print(f"\n✅ Figure saved: {save_path}")
        
        print(f"\n{'='*70}")
        print(f"✓ 3D H-bond geometry visualization complete")
        print(f"{'='*70}")
        
        return fig, np.array(axes)