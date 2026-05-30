#!/usr/bin/env python3
"""
Density-based RMSD Clustering Analysis

This module provides tools for analyzing molecular dynamics RMSD data using
density scatter plots to reveal natural conformational clusters.

Classes:
    RMSDClusterAnalyzer: Main class for density-based clustering analysis

Example:
    >>> analyzer = RMSDClusterAnalyzer()
    >>> analyzer.load_rmsd_data("rmsd_flat.xvg", "rmsd_cross.xvg")
    >>> analyzer.compute_density(bins=(200, 200), smoothing=20)
    >>> analyzer.plot_density_scatter()
"""
import argparse
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import hashlib
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist


class RMSDClusterAnalyzer:
    """
    Density-based clustering analyzer for RMSD conformational data.
    
    This class provides methods for loading RMSD data, computing density maps,
    identifying natural clusters, and extracting cluster information.
    
    Attributes:
        rmsd_flat (np.ndarray): RMSD values for flat reference structure
        rmsd_cross (np.ndarray): RMSD values for cross reference structure
        time (np.ndarray): Time points corresponding to RMSD values
        density_map (np.ndarray): 2D density histogram (smoothed)
        raw_counts (np.ndarray): 2D histogram of raw counts
        cluster_labels (np.ndarray): Cluster assignment for each frame
        cluster_centers (np.ndarray): Coordinates of cluster centers
        
    Example:
        >>> analyzer = RMSDClusterAnalyzer()
        >>> analyzer.load_rmsd_data("flat.xvg", "cross.xvg")
        >>> analyzer.compute_density(bins=(200, 200), smoothing=20)
        >>> n_clusters = analyzer.find_clusters(method='peaks', threshold=0.3)
        >>> analyzer.save_cluster_frames("output_dir")
    """
    
    def __init__(self):
        """Initialize the RMSD cluster analyzer."""
        self.rmsd_flat: Optional[np.ndarray] = None
        self.rmsd_cross: Optional[np.ndarray] = None
        self.time: Optional[np.ndarray] = None
        self.density_map: Optional[np.ndarray] = None
        self.raw_counts: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.bin_centers_x: Optional[np.ndarray] = None
        self.bin_centers_y: Optional[np.ndarray] = None
        # Reference structure labels for flexible analysis
        self.label_x: str = "flat"
        self.label_y: str = "cross"
        self._nx: int = 200
        self._ny: int = 200
        self._smoothing: float = 20
        self._target_max: float = 50
        # Selection aliases for molecular analysis
        self._selections: Dict[str, str] = {}
    
    def define_selections(self, selections: Dict[str, Union[str, Dict[str, str]]]) -> None:
        """
        Register named selections for molecular analysis.
        
        Args:
            selections: Dictionary of selection definitions. Can be nested:
                        - Flat: {'name': 'selection string'}
                        - Nested: {'category': {'name': 'selection string'}}
        
        Example:
            >>> analyzer.define_selections({
            ...     'CIP_parts': {
            ...         'quinolone': 'resname api and (name N6 or name C10 or name C11 or name C12 or name C19 or name C21 or name C22 or name C23 or name C4 or name C5)',
            ...         'carboxylic_acid': 'resname api and (name O1 or name O3 or name C2)',
            ...         'piperazine': 'resname api and (name N13 or name N16 or name C14 or name C15 or name C17 or name C18)'
            ...     },
            ...     'solvent': {
            ...         'water_O': 'resname SOL and name OW',
            ...         'water_H': 'resname SOL and (name HW1 or name HW2)'
            ...     },
            ...     'all_api': 'resname api'
            ... })
        """
        for key, value in selections.items():
            if isinstance(value, dict):
                # Nested structure: flatten with 'category_name' format
                for sub_key, sel_string in value.items():
                    self._selections[sub_key] = sel_string
            else:
                # Direct string selection
                self._selections[key] = value
    
    def sel(self, name: str) -> str:
        """
        Retrieve a selection string by name.
        
        Args:
            name: Name of the selection (as defined in define_selections)
        
        Returns:
            MDAnalysis selection string
        
        Raises:
            KeyError: If selection name not found
        
        Example:
            >>> rdf_data = analyzer.compute_rdf(
            ...     'system.tpr', 'traj.xtc',
            ...     analyzer.sel('quinolone'),
            ...     analyzer.sel('water_O')
            ... )
        """
        if name not in self._selections:
            available = ', '.join(sorted(self._selections.keys()))
            raise KeyError(f"Selection '{name}' not found. Available: {available}")
        return self._selections[name]
    
    def get_selection_name(self, selection_string: str) -> Optional[str]:
        """
        Reverse lookup: find the name for a selection string.
        
        Args:
            selection_string: MDAnalysis selection string
        
        Returns:
            Name of the selection, or None if not found
        
        Example:
            >>> name = analyzer.get_selection_name('resname api and (name O1 or name O3)')
            >>> print(name)  # 'carboxylic_acid'
        """
        # Find all matching names
        matches = [name for name, sel_str in self._selections.items() 
                   if sel_str == selection_string]
        
        if not matches:
            return None
        
        # Prefer names that don't start with 'all_'
        non_all_matches = [m for m in matches if not m.startswith('all_')]
        
        if non_all_matches:
            # Return shortest non-'all_*' name
            return min(non_all_matches, key=len)
        else:
            # Return shortest name overall
            return min(matches, key=len)
    
    @staticmethod
    def read_xvg(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read GROMACS XVG file.
        
        Args:
            path: Path to XVG file
            
        Returns:
            Tuple of (times, values) as numpy arrays
            
        Raises:
            FileNotFoundError: If XVG file does not exist
            ValueError: If file format is invalid
        """
        times = []
        values = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("@"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                times.append(float(parts[0]))
                values.append(float(parts[1]))
        
        if len(times) == 0:
            raise ValueError(f"No data found in {path}")
        
        return np.array(times), np.array(values)
    
    def load_rmsd_data(self, ref1_path: Union[str, Path], 
                      ref2_path: Union[str, Path],
                      label_x: str = "flat",
                      label_y: str = "cross") -> None:
        """
        Load RMSD data from XVG files for any two reference structures.
        
        Args:
            ref1_path: Path to first RMSD reference XVG file (x-axis)
            ref2_path: Path to second RMSD reference XVG file (y-axis)
            label_x: Label for first reference structure (default="flat")
            label_y: Label for second reference structure (default="cross")
            
        Raises:
            ValueError: If data lengths don't match or time points differ
            
        Example:
            >>> analyzer.load_rmsd_data("rmsd_flat_4q.xvg", "rmsd_cross_4q.xvg", 
            ...                        label_x="flat", label_y="cross")
            >>> analyzer.load_rmsd_data("rmsd_flat_4q.xvg", "rmsd_side_4q.xvg",
            ...                        label_x="flat", label_y="side")
            >>> analyzer.load_rmsd_data("rmsd_side_4q.xvg", "rmsd_cross_4q.xvg",
            ...                        label_x="side", label_y="cross")
        """
        time1, rmsd1 = self.read_xvg(ref1_path)
        time2, rmsd2 = self.read_xvg(ref2_path)
        
        if len(rmsd1) != len(rmsd2):
            raise ValueError(
                f"RMSD data length mismatch: {label_x}={len(rmsd1)}, "
                f"{label_y}={len(rmsd2)}"
            )
        
        if not np.allclose(time1, time2):
            raise ValueError("Time points do not match between reference files")
        
        # Store data with generic names (keeps backward compatibility)
        self.rmsd_flat = rmsd1
        self.rmsd_cross = rmsd2
        self.time = time1
        
        # Store labels for output and plotting
        self.label_x = label_x
        self.label_y = label_y
        
        print(f"Loaded {len(self.rmsd_flat)} frames")
        print(f"  RMSD {label_x} range: {rmsd1.min():.4f} - {rmsd1.max():.4f} nm")
        print(f"  RMSD {label_y} range: {rmsd2.min():.4f} - {rmsd2.max():.4f} nm")

    
    @staticmethod
    def _smooth_1d(Y: np.ndarray, lambda_val: float) -> np.ndarray:
        """
        Apply 1D penalized least squares smoothing.
        
        Args:
            Y: 1D or 2D array to smooth
            lambda_val: Smoothing parameter (higher = smoother)
            
        Returns:
            Smoothed array with same shape as input
        """
        m = Y.shape[0]
        E = sparse.eye(m, format='csc')
        D1 = sparse.diags([np.ones(m-1), -np.ones(m-1)], [0, 1], 
                         shape=(m-1, m), format='csc')
        D2 = D1[1:] - D1[:-1]
        
        P = lambda_val**2 * (D2.T @ D2) + 2 * lambda_val * (D1.T @ D1)
        Z = spsolve(E + P, Y)
        
        if Y.ndim == 2:
            Z = Z.reshape(Y.shape)
        return Z
    
    def _smooth_2d_counts(self, H: np.ndarray, lambda_val: float) -> np.ndarray:
        """
        Apply 2D smoothing by sequential 1D smoothing in both directions.
        
        Args:
            H: 2D histogram array
            lambda_val: Smoothing parameter
            
        Returns:
            Smoothed 2D array
        """
        m, n = H.shape
        # Smooth rows
        G = np.zeros_like(H)
        for i in range(m):
            G[i, :] = self._smooth_1d(H[i, :], n / lambda_val)
        # Smooth columns
        F = np.zeros_like(G)
        for j in range(n):
            F[:, j] = self._smooth_1d(G[:, j], m / lambda_val)
        return F
    
    def compute_density(self, bins: Tuple[int, int] = (200, 200),
                       smoothing: float = 20, target_max: float = 50) -> np.ndarray:
        """
        Compute 2D density map from RMSD data.
        
        Args:
            bins: Tuple of (nx, ny) bin counts for x and y axes
            smoothing: Smoothing parameter (higher = smoother)
            target_max: Maximum density value for scaling
            
        Returns:
            2D density map array
            
        Raises:
            ValueError: If RMSD data not loaded
        """
        if self.rmsd_flat is None or self.rmsd_cross is None:
            raise ValueError("RMSD data not loaded. Call load_rmsd_data() first.")
        
        self._nx, self._ny = bins
        self._smoothing = smoothing
        self._target_max = target_max
        
        # Create bin edges
        edges_x = np.linspace(self.rmsd_flat.min(), self.rmsd_flat.max(), self._nx + 1)
        self.bin_centers_x = edges_x[:-1] + np.diff(edges_x)[0] / 2
        
        edges_y = np.linspace(self.rmsd_cross.min(), self.rmsd_cross.max(), self._ny + 1)
        self.bin_centers_y = edges_y[:-1] + np.diff(edges_y)[0] / 2
        
        # Bin the data
        binx = np.digitize(self.rmsd_flat, edges_x) - 1
        biny = np.digitize(self.rmsd_cross, edges_y) - 1
        
        # Keep only valid bins
        valid = (binx >= 0) & (binx < self._nx) & (biny >= 0) & (biny < self._ny)
        
        # Count points in each bin
        H = np.zeros((self._ny, self._nx))
        for i in range(len(self.rmsd_flat)):
            if valid[i]:
                H[biny[i], binx[i]] += 1
        
        self.raw_counts = H.copy()
        
        # Smooth counts
        self.density_map = self._smooth_2d_counts(H, smoothing)
        
        # Scale to target max
        current_max = self.density_map.max()
        if current_max > 0:
            self.density_map = self.density_map * (target_max / current_max)
        
        print(f"Density map computed:")
        print(f"  Grid: {self._nx}×{self._ny}")
        print(f"  Max density: {self.density_map.max():.2f} counts/bin")
        print(f"  Mean density (non-zero): {self.density_map[self.density_map > 0].mean():.2f}")
        
        return self.density_map
    
    def get_point_densities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get density value for each data point.
        
        Returns:
            Tuple of (densities, valid_mask) where densities are the density
            values for each point and valid_mask indicates valid points
            
        Raises:
            ValueError: If density not computed
        """
        if self.density_map is None:
            raise ValueError("Density not computed. Call compute_density() first.")
        
        edges_x = np.linspace(self.rmsd_flat.min(), self.rmsd_flat.max(), self._nx + 1)
        edges_y = np.linspace(self.rmsd_cross.min(), self.rmsd_cross.max(), self._ny + 1)
        
        binx = np.digitize(self.rmsd_flat, edges_x) - 1
        biny = np.digitize(self.rmsd_cross, edges_y) - 1
        
        valid = (binx >= 0) & (binx < self._nx) & (biny >= 0) & (biny < self._ny)
        
        colors = np.zeros(len(self.rmsd_flat))
        for i in range(len(self.rmsd_flat)):
            if valid[i]:
                colors[i] = self.density_map[biny[i], binx[i]]
        
        return colors, valid
    
    def plot_density_scatter(self, figsize: Tuple[float, float] = (9, 7),
                            marker: str = 's', msize: float = 10,
                            cmap: str = 'viridis', save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create density scatter plot with points colored by local density.
        
        Args:
            figsize: Figure size as (width, height)
            marker: Marker style
            msize: Marker size
            cmap: Colormap name
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
            
        Raises:
            ValueError: If density not computed
        """
        if self.density_map is None:
            raise ValueError("Density not computed. Call compute_density() first.")
        
        colors, valid = self.get_point_densities()
        
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(self.rmsd_flat[valid], self.rmsd_cross[valid],
                           s=msize, c=colors[valid], marker=marker,
                           cmap=cmap, vmin=0, vmax=self._target_max,
                           alpha=0.8, edgecolors='none')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f'Density (counts per bin)', fontsize=11)
        ax.set_xlabel('RMSD flat (nm)', fontsize=12)
        ax.set_ylabel('RMSD cross (nm)', fontsize=12)
        ax.set_title('RMSD Density Scatter (flat vs cross)', fontsize=13)
        ax.grid(alpha=0.3, ls='--')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_density_heatmap(self, figsize: Tuple[float, float] = (9, 7),
                            cmap: str = 'viridis', save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create 2D density heatmap.
        
        Args:
            figsize: Figure size as (width, height)
            cmap: Colormap name
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
            
        Raises:
            ValueError: If density not computed
        """
        if self.density_map is None:
            raise ValueError("Density not computed. Call compute_density() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        extent = [self.rmsd_flat.min(), self.rmsd_flat.max(),
                 self.rmsd_cross.min(), self.rmsd_cross.max()]
        im = ax.imshow(self.density_map, extent=extent, origin='lower',
                      aspect='auto', cmap=cmap, vmin=0, vmax=self._target_max,
                      interpolation='bilinear')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density (counts per bin)', fontsize=11)
        ax.set_xlabel('RMSD flat (nm)', fontsize=12)
        ax.set_ylabel('RMSD cross (nm)', fontsize=12)
        ax.set_title('2D Density Heatmap', fontsize=13)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def find_clusters(self, method: str = 'peaks', threshold: float = 0.3,
                     peak_size: int = 20, density_threshold: float = 0.15,
                     density_ratio: float = 0.5) -> int:
        """
        Identify cluster centers from density map and assign frames.
        
        Args:
            method: Clustering method ('peaks' or 'threshold')
            threshold: Minimum density threshold for peak detection (fraction of max density)
            peak_size: Size of local maximum filter window
            density_threshold: Minimum density for cluster membership (fraction of max)
                             Points below this are marked as noise
            density_ratio: Point must have density >= this fraction of cluster peak density
                         (prevents merging clusters with different densities)
            
        Returns:
            Number of clusters found
            
        Raises:
            ValueError: If density not computed or invalid method
        """
        if self.density_map is None:
            raise ValueError("Density not computed. Call compute_density() first.")
        
        if method == 'peaks':
            # Find local maxima
            local_max = maximum_filter(self.density_map, size=peak_size) == self.density_map
            peaks = np.where(local_max & (self.density_map > self.density_map.max() * threshold))
            
            # Convert to RMSD coordinates
            peak_flat = self.bin_centers_x[peaks[1]]
            peak_cross = self.bin_centers_y[peaks[0]]
            
            self.cluster_centers = np.column_stack([peak_flat, peak_cross])
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Found {len(self.cluster_centers)} clusters")
        for i, (pf, pc) in enumerate(self.cluster_centers):
            print(f"  Cluster {i}: flat={pf:.4f} nm, cross={pc:.4f} nm")
        
        # Assign points to clusters with density filtering
        self._assign_to_clusters(density_threshold=density_threshold, 
                                density_ratio=density_ratio)
        
        return len(self.cluster_centers)
    
    def _assign_to_clusters(self, density_threshold: float = 0.15, 
                           density_ratio: float = 0.5) -> None:
        """
        Assign frames to clusters based on BOTH distance and density compatibility.
        
        Only frames in high-density regions are assigned to clusters.
        Additionally, frames must have density within a reasonable range of the cluster peak.
        This prevents merging distinct clusters with very different densities.
        
        Args:
            density_threshold: Minimum density (fraction of max) for cluster membership
            density_ratio: Point must have density >= this fraction of cluster peak density
                         (prevents low-density points from joining high-density clusters)
        """
        if self.cluster_centers is None or len(self.cluster_centers) == 0:
            raise ValueError("No cluster centers found")
        
        # Get density for each point
        colors, valid = self.get_point_densities()
        
        # Get peak density at each cluster center
        cluster_peak_densities = np.zeros(len(self.cluster_centers))
        for i, center in enumerate(self.cluster_centers):
            # Find bin indices for this cluster center
            x_idx = np.argmin(np.abs(self.bin_centers_x - center[0]))
            y_idx = np.argmin(np.abs(self.bin_centers_y - center[1]))
            cluster_peak_densities[i] = self.density_map[y_idx, x_idx]
        
        print(f"\nCluster peak densities:")
        for i, peak in enumerate(cluster_peak_densities):
            print(f"  Cluster {i}: {peak:.1f} counts/bin")
        
        # Initialize all as noise (-1)
        self.cluster_labels = np.full(len(self.rmsd_flat), -1, dtype=int)
        
        # Only assign points with sufficient density
        density_cutoff = self._target_max * density_threshold
        high_density_mask = colors >= density_cutoff
        
        if np.sum(high_density_mask) > 0:
            # For high-density points, assign considering BOTH distance and density
            points = np.column_stack([self.rmsd_flat, self.rmsd_cross])
            high_density_points = points[high_density_mask]
            high_density_colors = colors[high_density_mask]
            
            # Calculate distances to all clusters
            distances = cdist(high_density_points, self.cluster_centers)
            
            # For each point, find best cluster considering BOTH distance and density
            assignments = np.full(len(high_density_points), -1, dtype=int)
            
            for i, (point_density, point_distances) in enumerate(zip(high_density_colors, distances)):
                # Check which clusters this point is dense enough for
                # Point must have density >= density_ratio × cluster_peak_density
                valid_clusters = point_density >= (cluster_peak_densities * density_ratio)
                
                if np.any(valid_clusters):
                    # Among valid clusters, assign to nearest
                    valid_distances = np.where(valid_clusters, point_distances, np.inf)
                    assignments[i] = np.argmin(valid_distances)
                # else: remains -1 (noise) - not dense enough for any cluster
            
            self.cluster_labels[high_density_mask] = assignments
        
        print(f"\nCluster assignments (density threshold = {density_threshold:.2f} × max):")
        for i in range(len(self.cluster_centers)):
            count = np.sum(self.cluster_labels == i)
            percent = 100 * count / len(self.cluster_labels)
            print(f"  Cluster {i}: {count} frames ({percent:.1f}%)")
        
        noise_count = np.sum(self.cluster_labels == -1)
        noise_percent = 100 * noise_count / len(self.cluster_labels)
        print(f"  Noise: {noise_count} frames ({noise_percent:.1f}%)")
        
        # === REMOVE PHANTOM CLUSTERS (clusters with 0 assigned points) ===
        # Find clusters that have no points assigned
        valid_clusters = []
        for i in range(len(self.cluster_centers)):
            if np.sum(self.cluster_labels == i) > 0:
                valid_clusters.append(i)
        
        if len(valid_clusters) < len(self.cluster_centers):
            phantom_count = len(self.cluster_centers) - len(valid_clusters)
            phantom_ids = [i for i in range(len(self.cluster_centers)) if i not in valid_clusters]
            
            print(f"\n⚠ WARNING: Found {phantom_count} phantom cluster(s) with 0 points: {phantom_ids}")
            print("  These are local density peaks with no nearby high-density data points.")
            print("  Removing phantom clusters and re-indexing...")
            
            # Keep only valid cluster centers
            self.cluster_centers = self.cluster_centers[valid_clusters]
            
            # Re-map cluster labels to be consecutive (0, 1, 2, ...)
            new_labels = np.full(len(self.cluster_labels), -1, dtype=int)
            for new_idx, old_idx in enumerate(valid_clusters):
                mask = self.cluster_labels == old_idx
                new_labels[mask] = new_idx
            self.cluster_labels = new_labels
            
            print(f"\n✓ Fixed: Now have {len(self.cluster_centers)} valid clusters (0-{len(self.cluster_centers)-1})")
            print("   All clusters now have assigned points.")
    
    def plot_clusters(self, figsize: Tuple[float, float] = (9, 7),
                     msize: float = 10, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot data with cluster assignments.
        
        Args:
            figsize: Figure size as (width, height)
            msize: Marker size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
            
        Raises:
            ValueError: If clusters not found
        """
        if self.cluster_labels is None or self.cluster_centers is None:
            raise ValueError("Clusters not found. Call find_clusters() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        colors_cluster = plt.cm.tab10(np.linspace(0, 1, len(self.cluster_centers)))
        
        # Plot noise points first (if any)
        noise_mask = self.cluster_labels == -1
        if np.sum(noise_mask) > 0:
            ax.scatter(self.rmsd_flat[noise_mask], self.rmsd_cross[noise_mask],
                      s=msize*0.5, alpha=0.2, color='gray',
                      label=f'Noise (n={np.sum(noise_mask)})', zorder=1)
        
        # Plot clusters
        for i in range(len(self.cluster_centers)):
            mask = self.cluster_labels == i
            count = np.sum(mask)
            if count > 0:
                ax.scatter(self.rmsd_flat[mask], self.rmsd_cross[mask],
                          s=msize, alpha=0.6, color=colors_cluster[i],
                          label=f'Cluster {i} (n={count})', zorder=2)
        
        # Plot cluster centers
        ax.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1],
                  s=300, c='black', marker='X', edgecolors='white',
                  linewidths=2, zorder=10, label='Centers')
        
        ax.set_xlabel('RMSD flat (nm)', fontsize=12)
        ax.set_ylabel('RMSD cross (nm)', fontsize=12)
        ax.set_title('Frame Assignment to Clusters', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3, ls='--')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def get_cluster_frames(self, cluster_id: Optional[int] = None, 
                          include_noise: bool = False) -> Union[Dict[int, np.ndarray], np.ndarray]:
        """
        Get sorted frame numbers for each cluster.
        
        Each frame corresponds to one line in the XVG file (starting from frame 0).
        Frame numbers are sorted in ascending order. Noise points (label=-1) are
        excluded by default.
        
        Args:
            cluster_id: Specific cluster ID to retrieve frames for.
                       If None, returns all clusters. Use -1 for noise points.
            include_noise: If True and cluster_id is None, includes noise as cluster -1
        
        Returns:
            If cluster_id is None: Dictionary mapping cluster_id -> sorted frame numbers
            If cluster_id specified: 1D array of sorted frame numbers for that cluster
        
        Raises:
            ValueError: If clusters not found or invalid cluster_id
            
        Example:
            >>> analyzer.find_clusters()
            >>> # Get all clusters (excluding noise)
            >>> all_frames = analyzer.get_cluster_frames()
            >>> print(all_frames[0])  # Frames for cluster 0: [10, 25, 40, ...]
            >>> 
            >>> # Get specific cluster
            >>> cluster_0_frames = analyzer.get_cluster_frames(cluster_id=0)
            >>> 
            >>> # Get noise points
            >>> noise_frames = analyzer.get_cluster_frames(cluster_id=-1)
        """
        if self.cluster_labels is None or self.cluster_centers is None:
            raise ValueError("Clusters not found. Call find_clusters() first.")
        
        n_clusters = len(self.cluster_centers)
        
        # If specific cluster requested
        if cluster_id is not None:
            if cluster_id == -1:
                # Return noise points
                mask = self.cluster_labels == -1
                frames = np.where(mask)[0]
                return np.sort(frames)
            elif cluster_id < 0 or cluster_id >= n_clusters:
                raise ValueError(f"Invalid cluster_id={cluster_id}. Must be in range [0, {n_clusters-1}] or -1 for noise")
            
            mask = self.cluster_labels == cluster_id
            frames = np.where(mask)[0]  # Get frame indices (0-based)
            return np.sort(frames)  # Return sorted frame numbers
        
        # Return all clusters
        cluster_frames = {}
        for i in range(n_clusters):
            mask = self.cluster_labels == i
            frames = np.where(mask)[0]  # Get frame indices (0-based)
            if len(frames) > 0:  # Only include non-empty clusters
                cluster_frames[i] = np.sort(frames)  # Sort frame numbers
        
        # Optionally include noise
        if include_noise:
            noise_mask = self.cluster_labels == -1
            noise_frames = np.where(noise_mask)[0]
            if len(noise_frames) > 0:
                cluster_frames[-1] = np.sort(noise_frames)
        
        return cluster_frames
    
    def save_cluster_frames(self, output_dir: Union[str, Path]) -> None:
        """
        Save frame numbers and times for each cluster.
        
        Args:
            output_dir: Directory to save cluster data
            
        Raises:
            ValueError: If clusters not found
        """
        if self.cluster_labels is None or self.cluster_centers is None:
            raise ValueError("Clusters not found. Call find_clusters() first.")
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(len(self.cluster_centers)):
            mask = self.cluster_labels == i
            frames = np.where(mask)[0]
            times = self.time[mask]
            
            # Save frame numbers
            with open(out_path / f"cluster_{i}_frames.txt", "w") as f:
                for frame in frames:
                    f.write(f"{frame}\n")
            
            # Save times
            with open(out_path / f"cluster_{i}_times_ns.txt", "w") as f:
                for t in times:
                    f.write(f"{t:.7f}\n")
            
            print(f"Saved cluster {i}: {len(frames)} frames")
        
        # Save summary
        with open(out_path / "cluster_summary.txt", "w") as f:
            f.write("Density-based Clustering Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Reference structures: {self.label_x} vs {self.label_y}\n")
            f.write(f"Method: Density scatter with peak detection\n")
            f.write(f"Grid size: {self._nx}×{self._ny}\n")
            f.write(f"Smoothing: {self._smoothing}\n")
            f.write(f"Total frames: {len(self.rmsd_flat)}\n\n")
            
            for i in range(len(self.cluster_centers)):
                count = np.sum(self.cluster_labels == i)
                percent = 100 * count / len(self.cluster_labels)
                f.write(f"Cluster {i}: {count} frames ({percent:.2f}%)\n")
                f.write(f"  Center: {self.label_x}={self.cluster_centers[i,0]:.6f} nm, ")
                f.write(f"{self.label_y}={self.cluster_centers[i,1]:.6f} nm\n")
            
            # Add noise statistics
            noise_count = np.sum(self.cluster_labels == -1)
            noise_percent = 100 * noise_count / len(self.cluster_labels)
            f.write(f"\nNoise (scattered): {noise_count} frames ({noise_percent:.2f}%)\n")
        
        print(f"\n✓ All cluster data saved to {out_path}/")
    
    def export_gromacs_index(self, output_file: Union[str, Path],
                            include_noise: bool = False) -> None:
        """
        Create GROMACS .ndx index file for cluster trajectory extraction.
        
        The index file contains frame numbers for each cluster in GROMACS format,
        using 1-based indexing as required by GROMACS tools. Each cluster is
        written as a separate index group [ cluster_X ] with frame numbers
        formatted 15 per line.
        
        Args:
            output_file: Path to output .ndx file
            include_noise: If True, includes noise points as [ cluster_-1 ]
            
        Raises:
            ValueError: If clusters not found
            
        Example:
            >>> analyzer.find_clusters()
            >>> analyzer.export_gromacs_index("clusters.ndx")
            >>> # Use with: gmx extract-cluster -f traj.xtc -s topol.tpr -clusters clusters.ndx
            
        Note:
            GROMACS uses 1-based frame indexing, so this method automatically
            adds 1 to Python's 0-based frame numbers.
        """
        if self.cluster_labels is None or self.cluster_centers is None:
            raise ValueError("Clusters not found. Call find_clusters() first.")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for cluster_id in range(len(self.cluster_centers)):
                # Get frame numbers for this cluster (0-based from Python)
                frames = self.get_cluster_frames(cluster_id=cluster_id)
                
                # Write cluster header
                f.write(f"[ cluster_{cluster_id} ]\n")
                
                # Write frame numbers (add 1 for GROMACS 1-based indexing, 15 per line)
                for i in range(0, len(frames), 15):
                    line_frames = frames[i:i+15]
                    # Convert to 1-based indexing
                    f.write(" " + " ".join(f"{frame+1:6d}" for frame in line_frames) + "\n")
                
                f.write("\n")
            
            # Optionally include noise points
            if include_noise:
                noise_frames = self.get_cluster_frames(cluster_id=-1)
                if len(noise_frames) > 0:
                    f.write("[ cluster_-1 ]\n")
                    for i in range(0, len(noise_frames), 15):
                        line_frames = noise_frames[i:i+15]
                        f.write(" " + " ".join(f"{frame+1:6d}" for frame in line_frames) + "\n")
                    f.write("\n")
        
        print(f"✓ Created GROMACS index file: {output_path}")
        print(f"\nFile contains:")
        for cluster_id in range(len(self.cluster_centers)):
            frames = self.get_cluster_frames(cluster_id=cluster_id)
            print(f"  [ cluster_{cluster_id} ] - {len(frames)} frames (1-based indexing)")
        if include_noise:
            noise_frames = self.get_cluster_frames(cluster_id=-1)
            if len(noise_frames) > 0:
                print(f"  [ cluster_-1 ] - {len(noise_frames)} noise frames (1-based indexing)")
        print(f"\nReady for: gmx extract-cluster -f trajectory.xtc -s topology.tpr -clusters {output_path}")
    
    def extract_cluster_trajectories(self, 
                                     topology_file: Union[str, Path],
                                     trajectory_file: Union[str, Path],
                                     output_dir: Union[str, Path],
                                     cluster_ids: Optional[List[int]] = None,
                                     include_noise: bool = False) -> None:
        """
        Extract separate trajectory files for each cluster using MDAnalysis.
        
        This method extracts the frames belonging to each cluster into separate
        .xtc trajectory files. It uses MDAnalysis to avoid GROMACS version
        compatibility issues and provides progress feedback.
        
        Args:
            topology_file: Path to topology file (.tpr, .gro, .pdb)
            trajectory_file: Path to trajectory file (.xtc, .trr)
            output_dir: Directory where cluster trajectories will be saved
            cluster_ids: Specific cluster IDs to extract. If None, extracts all clusters.
            include_noise: If True, also extracts noise points as cluster_-1.xtc
            
        Raises:
            ValueError: If clusters not found
            ImportError: If MDAnalysis is not installed
            
        Example:
            >>> analyzer.find_clusters()
            >>> analyzer.extract_cluster_trajectories(
            ...     topology_file="system.tpr",
            ...     trajectory_file="traj.xtc",
            ...     output_dir="cluster_trajectories"
            ... )
            >>> # Extract specific clusters only
            >>> analyzer.extract_cluster_trajectories(
            ...     "system.tpr", "traj.xtc", "output", cluster_ids=[0, 2]
            ... )
            
        Note:
            Output files are named cluster_0.xtc, cluster_1.xtc, etc.
            Frame ordering is preserved (sorted by frame number).
        """
        if self.cluster_labels is None or self.cluster_centers is None:
            raise ValueError("Clusters not found. Call find_clusters() first.")
        
        try:
            import MDAnalysis as mda
        except ImportError:
            raise ImportError(
                "MDAnalysis is required for trajectory extraction. "
                "Install with: pip install MDAnalysis"
            )
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load universe
        print("Loading trajectory...")
        u = mda.Universe(str(topology_file), str(trajectory_file))
        print(f"✓ Loaded {len(u.trajectory)} frames")
        
        # Determine which clusters to extract
        if cluster_ids is None:
            cluster_ids = list(range(len(self.cluster_centers)))
        
        # Extract each cluster
        for cluster_id in cluster_ids:
            # Get frame numbers for this cluster (0-based indexing)
            frames = self.get_cluster_frames(cluster_id=cluster_id)
            
            output_file = output_path / f"cluster_{cluster_id}.xtc"
            
            print(f"\nExtracting Cluster {cluster_id}:")
            print(f"  Frames: {len(frames)}")
            print(f"  Output: {output_file}")
            
            # Write trajectory with only the selected frames
            with mda.Writer(str(output_file), u.atoms.n_atoms) as writer:
                for frame_idx in frames:
                    u.trajectory[frame_idx]  # Go to specific frame
                    writer.write(u.atoms)
            
            print(f"  ✓ Written {len(frames)} frames")
        
        # Optionally extract noise
        if include_noise:
            noise_frames = self.get_cluster_frames(cluster_id=-1)
            if len(noise_frames) > 0:
                output_file = output_path / "cluster_-1.xtc"
                print(f"\nExtracting Noise points:")
                print(f"  Frames: {len(noise_frames)}")
                print(f"  Output: {output_file}")
                
                with mda.Writer(str(output_file), u.atoms.n_atoms) as writer:
                    for frame_idx in noise_frames:
                        u.trajectory[frame_idx]
                        writer.write(u.atoms)
                
                print(f"  ✓ Written {len(noise_frames)} frames")
        
        print("\n" + "="*60)
        print("✓ All cluster trajectories extracted successfully!")
        print(f"\nOutput files in {output_path}:")
        for cluster_id in cluster_ids:
            output_file = output_path / f"cluster_{cluster_id}.xtc"
            frames = self.get_cluster_frames(cluster_id=cluster_id)
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024**2)
                print(f"  - {output_file.name}: {len(frames)} frames, {size_mb:.2f} MB")
        if include_noise:
            noise_file = output_path / "cluster_-1.xtc"
            if noise_file.exists():
                noise_frames = self.get_cluster_frames(cluster_id=-1)
                size_mb = noise_file.stat().st_size / (1024**2)
                print(f"  - {noise_file.name}: {len(noise_frames)} frames, {size_mb:.2f} MB")
    
    # -------------------------------------------------------------------------
    # Trajectory Analysis Methods for Cluster Interactions
    # -------------------------------------------------------------------------
    
    def load_cluster_trajectories(self, topology_file: str, trajectory_file: str,
                                  cluster_ids: Union[List[int], str, None] = None):
        """
        Load MD trajectories for cluster analysis.
        
        Parameters
        ----------
        topology_file : str
            Path to topology file (.tpr, .gro, .pdb)
        trajectory_file : str
            Path to trajectory file (.xtc, .trr)
        cluster_ids : list of int, 'all', or None, optional
            Specific cluster IDs to load. If None or 'all', loads all clusters.
            
        Sets
        ----
        self.trajectory_data : dict
            Dictionary storing MDAnalysis Universe objects for each cluster:
            {cluster_id: {'universe': Universe, 'frames': frame_list}}
        
        Example
        -------
        >>> analyzer.load_cluster_trajectories('system.tpr', 'traj.xtc')
        >>> # Analyze all clusters explicitly
        >>> analyzer.load_cluster_trajectories('system.tpr', 'traj.xtc', 'all')
        >>> # Analyze specific clusters only
        >>> analyzer.load_cluster_trajectories('system.tpr', 'traj.xtc', [0, 1])
        """
        try:
            import MDAnalysis as mda
        except ImportError:
            raise ImportError(
                "MDAnalysis is required for trajectory analysis. "
                "Install with: pip install MDAnalysis"
            )
        
        if self.cluster_labels is None:
            raise ValueError(
                "No cluster assignments found. Run find_clusters() first."
            )
        
        print(f"\nLoading trajectory data...")
        print(f"  Topology: {topology_file}")
        print(f"  Trajectory: {trajectory_file}")
        
        # Load full trajectory
        u = mda.Universe(topology_file, trajectory_file)
        print(f"  ✓ Loaded {len(u.trajectory)} frames")
        
        # Determine which clusters to process
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = list(range(len(self.cluster_centers)))
        
        # Store trajectory data for each cluster
        self.trajectory_data = {}
        self.topology_file = topology_file
        self.trajectory_file = trajectory_file
        
        for cluster_id in cluster_ids:
            frames = self.get_cluster_frames(cluster_id=cluster_id)
            self.trajectory_data[cluster_id] = {
                'universe': u,  # Share the same universe
                'frames': frames,
                'n_frames': len(frames)
            }
            print(f"  Cluster {cluster_id}: {len(frames)} frames")
        
        print(f"✓ Trajectory data loaded for {len(cluster_ids)} clusters")
    
    def perform_pca(self, n_components: int = 2, exclude_noise: bool = True) -> Dict:
        """
        Perform Principal Component Analysis on RMSD data for dimensionality reduction.
        
        Projects high-dimensional RMSD space onto principal components for visualization.
        Computes cluster centers in PC space for free energy landscape plotting.
        
        Parameters
        ----------
        n_components : int, default=2
            Number of principal components to compute (typically 2 for visualization)
        exclude_noise : bool, default=True
            If True, exclude noise cluster (label=-1) from PCA fitting (but still 
            transform and compute center for visualization). Recommended for better
            PCA quality since noise points are outliers.
            
        Returns
        -------
        pca_results : dict
            {
                'pca_components': array (n_frames, n_components) - all frames in PC space,
                'explained_variance': array - variance explained by each PC,
                'explained_variance_ratio': array - fraction of total variance,
                'cluster_centers_pc': dict - {cluster_id: center} for all clusters,
                'cluster_centers_indices': dict - {cluster_id: center_index},
                'n_components': int
            }
            
        Notes
        -----
        - PCA is fitted on non-noise data (if exclude_noise=True) for better quality
        - ALL points (including noise) are transformed to PC space for plotting
        - Cluster centers computed for ALL clusters, including noise cluster -1
        - Data is standardized (mean=0, std=1) before PCA
        - Required for plot_free_energy_landscape() visualization
        
        Examples
        --------
        >>> # Compute 2D PCA for visualization
        >>> pca_results = analyzer.perform_pca(n_components=2)
        >>> print(f"PC1 explains {pca_results['explained_variance_ratio'][0]:.1%}")
        >>> 
        >>> # Plot free energy landscape
        >>> fig = plotter.plot_free_energy_landscape(fe_data)
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        if self.cluster_labels is None:
            raise ValueError("No cluster assignments. Run find_clusters() first.")
        
        if self.rmsd_flat is None or self.rmsd_cross is None:
            raise ValueError("No RMSD data loaded. Run load_rmsd_data() first.")
        
        print(f"\n{'='*70}")
        print("PRINCIPAL COMPONENT ANALYSIS")
        print(f"{'='*70}")
        
        # Prepare data: combine rmsd_flat and rmsd_cross into 2D array
        X = np.column_stack([self.rmsd_flat, self.rmsd_cross])
        labels = self.cluster_labels.copy()
        
        # Store original data for later transformation
        X_full = X.copy()
        labels_full = labels.copy()
        
        # Optionally exclude noise points from PCA fitting
        if exclude_noise and -1 in labels:
            mask = labels != -1
            X = X[mask]
            labels = labels[mask]
            print(f"  Excluding {np.sum(~mask)} noise points from PCA fitting (cluster -1)")
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"  Input data: {X.shape[0]} frames × {X.shape[1]} dimensions")
        print(f"  Computing {n_components} principal components...")
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Transform ALL data (including noise) to PC space for plotting
        X_full_scaled = scaler.transform(X_full)
        X_full_pca = pca.transform(X_full_scaled)
        
        # Compute cluster centers in PC space for ALL clusters (including noise)
        unique_clusters = np.unique(labels_full)
        cluster_centers_pc = {}
        cluster_centers_indices = {}
        
        # For plotting, we need indices of cluster centers
        # We'll compute mean position in PC space for each cluster
        for cid in unique_clusters:
            cluster_mask = labels_full == cid
            cluster_pc = X_full_pca[cluster_mask]
            cluster_center = cluster_pc.mean(axis=0)
            
            # Find closest point to center (for plotting)
            distances = np.linalg.norm(cluster_pc - cluster_center, axis=1)
            center_idx = np.where(cluster_mask)[0][np.argmin(distances)]
            
            cluster_centers_pc[int(cid)] = cluster_center
            cluster_centers_indices[int(cid)] = center_idx
        
        # Store results as instance attributes (required for plotting)
        self.pca_components = X_full_pca  # Store ALL transformed data
        self.pca_model = pca
        self.pca_scaler = scaler
        self.cluster_centers = cluster_centers_indices  # Includes all clusters
        self.cluster_centers_pc = cluster_centers_pc  # Includes all clusters
        
        # Print results
        print(f"\n  ✓ PCA complete")
        print(f"  Explained variance by component:")
        for i, (var, ratio) in enumerate(zip(pca.explained_variance_, 
                                             pca.explained_variance_ratio_), 1):
            print(f"    PC{i}: {ratio*100:.1f}% (variance={var:.2f})")
        print(f"  Total variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        print(f"\n  Cluster centers in PC space (computed for all {len(cluster_centers_pc)} clusters):")
        for cid in sorted(cluster_centers_pc.keys()):
            center = cluster_centers_pc[cid]
            label = " (noise)" if cid == -1 else ""
            print(f"    Cluster {cid}{label}: PC1={center[0]:.2f}, PC2={center[1]:.2f}")
        print(f"{'='*70}")
        
        # Return results dict
        pca_results = {
            'pca_components': X_pca,
            'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cluster_centers_pc': cluster_centers_pc,
            'cluster_centers_indices': cluster_centers_indices,
            'n_components': n_components
        }
        
        return pca_results
    
    def compute_cluster_free_energies(self, 
                                     temperature: float = 300.0,
                                     reference_cluster: Union[int, str] = 'auto',
                                     units: str = 'kJ/mol',
                                     bootstrap_samples: int = 1000,
                                     random_seed: Optional[int] = None) -> Dict:
        """
        Compute relative free energies between clusters using Boltzmann distribution.
        
        Uses the fundamental thermodynamic relation:
            ΔG_ij = -RT ln(N_i/N_j)
        
        where N_i and N_j are the populations (frame counts) of clusters i and j.
        
        Parameters
        ----------
        temperature : float, default=300.0
            Simulation temperature in Kelvin
        reference_cluster : int or 'auto', default='auto'
            Reference cluster for ΔG = 0:
            - 'auto': Use most populated cluster
            - int: Use specific cluster ID
        units : str, default='kJ/mol'
            Energy units: 'kJ/mol' or 'kcal/mol'
        bootstrap_samples : int, default=1000
            Number of bootstrap samples for error estimation
        random_seed : int, optional
            Random seed for reproducible bootstrap sampling
            
        Returns
        -------
        free_energy_data : dict
            {cluster_id: {
                'delta_G': float,        # Free energy relative to reference (kJ/mol or kcal/mol)
                'std_error': float,      # Bootstrap standard error
                'population': float,     # Population fraction
                'n_frames': int,         # Frame count
                'is_reference': bool     # True if this is reference cluster
            }}
            
        Notes
        -----
        - Assumes system is at equilibrium
        - Only computes relative free energies (not absolute values)
        - Error increases for low-population clusters
        - Valid when clusters are well-separated conformational states
        
        References
        ----------
        - Boltzmann, L. (1877). "Über die Beziehung zwischen dem zweiten Hauptsatze"
        - Hub, J. S., de Groot, B. L., & van der Spoel, D. (2010). J. Chem. Theory Comput.
        
        Examples
        --------
        >>> # Compute free energies with most populated as reference
        >>> fe_data = analyzer.compute_cluster_free_energies(temperature=300)
        >>> 
        >>> # Use specific cluster as reference
        >>> fe_data = analyzer.compute_cluster_free_energies(
        ...     temperature=300, 
        ...     reference_cluster=0,
        ...     units='kcal/mol'
        ... )
        """
        if self.cluster_labels is None:
            raise ValueError("No cluster assignments. Run find_clusters() first.")
        
        # Constants
        if units == 'kJ/mol':
            R = 8.314e-3  # kJ/(mol·K)
        elif units == 'kcal/mol':
            R = 1.987e-3  # kcal/(mol·K)
        else:
            raise ValueError(f"Unknown units '{units}'. Use 'kJ/mol' or 'kcal/mol'.")
        
        RT = R * temperature
        
        # Count frames per cluster
        unique_clusters = np.unique(self.cluster_labels)
        cluster_counts = {int(cid): int(np.sum(self.cluster_labels == cid)) 
                         for cid in unique_clusters}
        total_frames = len(self.cluster_labels)
        
        # Determine reference cluster
        if reference_cluster == 'auto':
            ref_id = max(cluster_counts, key=cluster_counts.get)
            print(f"\nAuto-selected reference cluster: {ref_id} (most populated, N={cluster_counts[ref_id]})")
        else:
            ref_id = int(reference_cluster)
            if ref_id not in cluster_counts:
                raise ValueError(f"Reference cluster {ref_id} not found in data.")
            print(f"\nUsing cluster {ref_id} as reference (N={cluster_counts[ref_id]})")
        
        ref_population = cluster_counts[ref_id] / total_frames
        
        print(f"Temperature: {temperature} K")
        print(f"RT: {RT:.4f} {units}")
        print(f"Total frames: {total_frames}")
        print(f"\nComputing free energies:")
        
        # Compute free energies
        fe_data = {}
        
        for cluster_id in sorted(cluster_counts.keys()):
            n_frames = cluster_counts[cluster_id]
            population = n_frames / total_frames
            
            if cluster_id == ref_id:
                delta_G = 0.0
                is_ref = True
            else:
                # ΔG = -RT ln(N_i/N_ref)
                delta_G = -RT * np.log(population / ref_population)
                is_ref = False
            
            # Bootstrap error estimation
            if random_seed is not None:
                np.random.seed(random_seed)
            
            bootstrap_dGs = []
            for _ in range(bootstrap_samples):
                # Resample with replacement
                resampled_labels = np.random.choice(self.cluster_labels, 
                                                   size=total_frames, 
                                                   replace=True)
                boot_counts = {int(cid): int(np.sum(resampled_labels == cid)) 
                              for cid in unique_clusters}
                
                boot_pop = boot_counts[cluster_id] / total_frames
                boot_ref_pop = boot_counts[ref_id] / total_frames
                
                if boot_pop > 0 and boot_ref_pop > 0:
                    boot_dG = -RT * np.log(boot_pop / boot_ref_pop)
                    bootstrap_dGs.append(boot_dG)
            
            std_error = np.std(bootstrap_dGs) if bootstrap_dGs else 0.0
            
            fe_data[cluster_id] = {
                'delta_G': delta_G,
                'std_error': std_error,
                'population': population,
                'n_frames': n_frames,
                'is_reference': is_ref
            }
            
            ref_marker = " (reference)" if is_ref else ""
            print(f"  Cluster {cluster_id}: ΔG = {delta_G:6.2f} ± {std_error:5.2f} {units}, "
                  f"p = {population:.3f} (N={n_frames}){ref_marker}")
        
        # Store for later use
        if not hasattr(self, 'free_energy_data'):
            self.free_energy_data = {}
        
        self.free_energy_data[f"T{temperature}_{units}_{ref_id}"] = {
            'temperature': temperature,
            'units': units,
            'reference_cluster': ref_id,
            'results': fe_data
        }
        
        print(f"\n✓ Free energy calculation complete")
        print(f"  Range: {min(d['delta_G'] for d in fe_data.values()):.2f} to "
              f"{max(d['delta_G'] for d in fe_data.values()):.2f} {units}")
        
        return fe_data
    
    def validate_energy_groups(self, tpr_file: str, gmx_command: str = 'gmx') -> List[str]:
        """
        Validate available energy groups from TPR file.
        
        Uses gmx dump to extract energygrps from TPR file.
        Useful for checking which groups are available before extraction.
        
        Parameters
        ----------
        tpr_file : str
            Path to .tpr file
        gmx_command : str, default='gmx'
            GROMACS executable path
        
        Returns
        -------
        energy_groups : list of str
            Available energy group names
        
        Example
        -------
        >>> groups = analyzer.validate_energy_groups('topol.tpr')
        >>> print(f"Available groups: {groups}")
        ['CIP', 'MMT', 'Water', 'Ion']
        """
        import subprocess
        import re
        
        if not os.path.exists(tpr_file):
            raise FileNotFoundError(f"TPR file not found: {tpr_file}")
        
        print(f"\n{'='*60}")
        print(f"Validating Energy Groups from TPR")
        print(f"{'='*60}")
        print(f"TPR file: {tpr_file}")
        
        try:
            # Run gmx dump to get TPR contents
            cmd = [gmx_command, 'dump', '-s', tpr_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"gmx dump failed: {result.stderr}")
            
            # Parse output for energy groups
            energy_groups = []
            in_grps_section = False
            
            for line in result.stdout.split('\n'):
                if 'energygrps' in line.lower() or 'energy-groups' in line.lower():
                    in_grps_section = True
                    # Try to extract groups from same line
                    match = re.search(r'=\s*(.+)', line)
                    if match:
                        grps = match.group(1).strip()
                        energy_groups.extend([g.strip() for g in grps.split() if g.strip()])
                elif in_grps_section and line.strip() and not line.startswith('#'):
                    # Continue reading group names
                    parts = line.strip().split()
                    if parts and not parts[0].startswith('('):
                        energy_groups.extend([g.strip() for g in parts if g.strip()])
                    else:
                        break
            
            if not energy_groups:
                print("⚠ Warning: No energy groups found in TPR file")
                print("  Check if energygrps was set in mdp file")
            else:
                print(f"\n✓ Found {len(energy_groups)} energy groups:")
                for grp in energy_groups:
                    print(f"  - {grp}")
            
            return energy_groups
            
        except Exception as e:
            print(f"✗ Error validating energy groups: {e}")
            return []
    
    def compute_energy_decomposition(self,
                                    edr_file: str,
                                    energy_groups: List[Tuple[str, str]],
                                    cluster_ids: Union[str, List[int]] = 'all',
                                    components: Optional[List[str]] = None,
                                    include_noise: bool = False,
                                    gmx_command: str = 'gmx',
                                    temperature: float = 300.0,
                                    normalize: str = 'none',
                                    save_xvg: bool = False,
                                    xvg_dir: str = './energy_xvg') -> Dict:
        """
        Extract pairwise interaction energies using gmx energy.
        
        Decomposes interactions into vdW (LJ) and Coulomb contributions.
        Maps energy values to cluster assignments for per-cluster analysis.
        
        Parameters
        ----------
        edr_file : str
            Path to .edr energy file from simulation
        energy_groups : list of tuple
            Pairs to analyze: [('CIP', 'MMT'), ('CIP', 'Water'), ...]
        cluster_ids : 'all' or list of int, default='all'
            Clusters to analyze
        components : list of str, optional
            Energy terms to extract. Default: ['Coul-SR', 'LJ-SR']
            Options: 'Coul-SR', 'LJ-SR', 'Coul-14', 'LJ-14'
        include_noise : bool, default=False
            Include cluster -1 (noise/unclassified) in reporting
        gmx_command : str, default='gmx'
            GROMACS executable
        temperature : float, default=300.0
            Temperature (K) for kT normalization if normalize='kT'
        normalize : str, default='none'
            Normalization: 'none', 'kT', 'per_molecule', 'per_atom'
        save_xvg : bool, default=False
            Save extracted XVG files
        xvg_dir : str, default='./energy_xvg'
            Directory for XVG files
        
        Returns
        -------
        energy_data : dict
            {cluster_id: {(group1, group2): {'Coul-SR': array, 'LJ-SR': array, 'Total': array, 'Time': array}}}
            Arrays contain energy values (kJ/mol) per frame in that cluster.
            Special key 'all': whole-system energies (all frames aggregated)
        
        Example
        -------
        >>> energy_data = analyzer.compute_energy_decomposition(
        ...     edr_file='../production.edr',
        ...     energy_groups=[('CIP', 'MMT'), ('CIP', 'Water')],
        ...     components=['Coul-SR', 'LJ-SR'],
        ...     cluster_ids='all'
        ... )
        >>> # Access specific cluster and group pair
        >>> cip_mmt_coul = energy_data[0][('CIP', 'MMT')]['Coul-SR']
        >>> print(f"Mean Coulomb: {cip_mmt_coul.mean():.2f} kJ/mol")
        """
        import subprocess
        import tempfile
        
        if components is None:
            components = ['Coul-SR', 'LJ-SR']
        
        if not os.path.exists(edr_file):
            raise FileNotFoundError(f"EDR file not found: {edr_file}")
        
        # Handle cluster_ids
        if cluster_ids == 'all':
            if not hasattr(self, 'cluster_labels'):
                raise ValueError("No clustering performed. Run perform_clustering() first.")
            cluster_ids = sorted(set(self.cluster_labels))
        elif isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]
        
        print(f"\n{'='*60}")
        print(f"Energy Decomposition Analysis")
        print(f"{'='*60}")
        print(f"Analyzer object ID: {id(self)}")  # DEBUG: Show which analyzer is being used
        print(f"EDR file: {edr_file}")
        print(f"Energy groups: {len(energy_groups)} pairs")
        for g1, g2 in energy_groups:
            print(f"  - {g1} <-> {g2}")
        print(f"Components: {', '.join(components)}")
        print(f"Clusters: {cluster_ids}")
        print(f"Temperature: {temperature} K")
        print(f"Normalization: {normalize}")
        print(f"Include noise cluster (-1): {include_noise}")
        
        # Show cluster distribution
        if hasattr(self, 'cluster_labels'):
            cluster_counts = {}
            for label in set(self.cluster_labels):
                cluster_counts[label] = sum(np.array(self.cluster_labels) == label)
            print(f"\nCluster frame distribution:")
            for cid in sorted(cluster_counts.keys()):
                pct = 100 * cluster_counts[cid] / len(self.cluster_labels)
                if cid == -1:
                    print(f"  Cluster {cid} (NOISE): {cluster_counts[cid]} frames ({pct:.1f}%)")
                else:
                    print(f"  Cluster {cid}: {cluster_counts[cid]} frames ({pct:.1f}%)")
        
        if save_xvg:
            os.makedirs(xvg_dir, exist_ok=True)
            print(f"XVG output: {xvg_dir}")
        
        energy_data = {cid: {} for cid in cluster_ids}
        energy_data['all'] = {}  # Whole-system energies
        
        # Get frame-to-cluster mapping
        if not hasattr(self, 'cluster_labels'):
            raise ValueError("No clustering performed. Run perform_clustering() first.")
        
        cluster_labels = np.array(self.cluster_labels)
        
        # DEBUG: Show cluster label distribution
        print(f"\n{'='*60}")
        print(f"DEBUG: Cluster Labels Information")
        print(f"{'='*60}")
        print(f"Total frames in cluster_labels: {len(cluster_labels)}")
        print(f"Unique clusters: {np.unique(cluster_labels)}")
        for cid in sorted(np.unique(cluster_labels)):
            n_frames = np.sum(cluster_labels == cid)
            print(f"  Cluster {cid}: {n_frames} frames ({100*n_frames/len(cluster_labels):.1f}%)")
        print(f"{'='*60}\n")
        
        # Extract energies for each group pair
        for g1, g2 in energy_groups:
            print(f"\nProcessing {g1} <-> {g2}...")
            
            pair_energies = {}
            
            for comp in components:
                # Construct energy term name (GROMACS format)
                energy_term = f"{comp}:{g1}-{g2}"
                
                print(f"  Extracting {comp}...", end=' ')
                
                try:
                    # Create input for gmx energy (select energy term)
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                        # Try to find the term by name
                        f.write(f"{energy_term}\n")
                        input_file = f.name
                    
                    # Run gmx energy
                    cmd = [gmx_command, 'energy', '-f', edr_file]
                    
                    if save_xvg:
                        xvg_file = os.path.join(xvg_dir, f"{g1}_{g2}_{comp}.xvg")
                        cmd.extend(['-o', xvg_file])
                    else:
                        # Use temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xvg') as temp_xvg:
                            xvg_file = temp_xvg.name
                        cmd.extend(['-o', xvg_file])
                    
                    # Run command
                    with open(input_file, 'r') as stdin:
                        result = subprocess.run(cmd, stdin=stdin, capture_output=True, text=True)
                    
                    os.unlink(input_file)
                    
                    if result.returncode != 0:
                        print(f"✗ Failed: {result.stderr}")
                        continue
                    
                    # Parse XVG file
                    times = []
                    energies = []
                    
                    with open(xvg_file, 'r') as f:
                        for line in f:
                            if line.startswith('#') or line.startswith('@'):
                                continue
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    times.append(float(parts[0]))
                                    energies.append(float(parts[1]))
                                except ValueError:
                                    continue
                    
                    if not save_xvg:
                        os.unlink(xvg_file)
                    
                    pair_energies[comp] = np.array(energies)
                    
                    if 'Time' not in pair_energies:
                        pair_energies['Time'] = np.array(times)
                    
                    print(f"✓ ({len(energies)} frames, mean={np.mean(energies):.2f} kJ/mol)")
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue
            
            # Compute total energy
            if pair_energies:
                total = sum(pair_energies[comp] for comp in components if comp in pair_energies)
                pair_energies['Total'] = total
            
            # Split by clusters
            for cid in cluster_ids:
                cluster_mask = cluster_labels == cid
                cluster_energies = {}
                
                for key, values in pair_energies.items():
                    cluster_energies[key] = values[cluster_mask]
                
                energy_data[cid][(g1, g2)] = cluster_energies
            
            # Store whole-system data (all frames)
            energy_data['all'][(g1, g2)] = pair_energies.copy()
        
        # Apply normalization if requested
        if normalize != 'none':
            kB = 8.314e-3  # kJ/(mol·K)
            kT = kB * temperature
            
            print(f"\nApplying normalization: {normalize}")
            
            for cid in energy_data:
                for pair in energy_data[cid]:
                    for comp in energy_data[cid][pair]:
                        if comp == 'Time':
                            continue
                        
                        if normalize == 'kT':
                            energy_data[cid][pair][comp] /= kT
        
        # Store for later use
        if not hasattr(self, 'energy_decomposition_data'):
            self.energy_decomposition_data = {}
        
        self.energy_decomposition_data[edr_file] = {
            'energy_groups': energy_groups,
            'components': components,
            'cluster_ids': cluster_ids,
            'temperature': temperature,
            'normalize': normalize,
            'data': energy_data
        }
        
        print(f"\n✓ Energy decomposition complete")
        print(f"  Extracted energies for {len(energy_groups)} pairs across {len(cluster_ids)} clusters + whole system")
        
        # Print cluster-wise summary
        print(f"\n{'='*80}")
        print(f"CLUSTER-WISE ENERGY SUMMARY")
        print(f"{'='*80}")
        
        # Filter clusters based on include_noise parameter
        if include_noise:
            valid_clusters = sorted(cluster_ids)
            print(f"\nDisplaying ALL clusters (including noise cluster -1)")
        else:
            valid_clusters = [c for c in cluster_ids if c >= 0]
            noise_clusters = [c for c in cluster_ids if c < 0]
            if noise_clusters:
                print(f"\n⚠️  Hiding noise cluster(s) {noise_clusters} from display (set include_noise=True to show)")
                print(f"   Note: Noise frames ARE still included in 'ALL' row statistics")
        
        for cid in sorted(valid_clusters):
            n_frames = len(energy_data[cid][energy_groups[0]]['Time']) if energy_groups else 0
            print(f"\n{'─'*80}")
            print(f"Cluster {cid} ({n_frames} frames)")
            print(f"{'─'*80}")
            
            for g1, g2 in energy_groups:
                if (g1, g2) in energy_data[cid]:
                    print(f"\n  {g1} <-> {g2}:")
                    
                    for comp in components:
                        if comp in energy_data[cid][(g1, g2)]:
                            values = energy_data[cid][(g1, g2)][comp]
                            mean_val = np.mean(values)
                            std_val = np.std(values)
                            min_val = np.min(values)
                            max_val = np.max(values)
                            
                            print(f"    {comp:12s}: {mean_val:8.2f} ± {std_val:6.2f} kJ/mol  "
                                  f"[{min_val:7.2f}, {max_val:7.2f}]")
                    
                    # Print total
                    if 'Total' in energy_data[cid][(g1, g2)]:
                        total_values = energy_data[cid][(g1, g2)]['Total']
                        mean_total = np.mean(total_values)
                        std_total = np.std(total_values)
                        min_total = np.min(total_values)
                        max_total = np.max(total_values)
                        
                        print(f"    {'Total':12s}: {mean_total:8.2f} ± {std_total:6.2f} kJ/mol  "
                              f"[{min_total:7.2f}, {max_total:7.2f}]")
        
        # Print whole-system summary
        n_frames_all = len(energy_data['all'][energy_groups[0]]['Time']) if energy_groups else 0
        print(f"\n{'─'*80}")
        print(f"WHOLE SYSTEM (All {n_frames_all} frames)")
        print(f"{'─'*80}")
        
        for g1, g2 in energy_groups:
            if (g1, g2) in energy_data['all']:
                print(f"\n  {g1} <-> {g2}:")
                
                for comp in components:
                    if comp in energy_data['all'][(g1, g2)]:
                        values = energy_data['all'][(g1, g2)][comp]
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        min_val = np.min(values)
                        max_val = np.max(values)
                        
                        print(f"    {comp:12s}: {mean_val:8.2f} ± {std_val:6.2f} kJ/mol  "
                              f"[{min_val:7.2f}, {max_val:7.2f}]")
                
                # Print total
                if 'Total' in energy_data['all'][(g1, g2)]:
                    total_values = energy_data['all'][(g1, g2)]['Total']
                    mean_total = np.mean(total_values)
                    std_total = np.std(total_values)
                    min_total = np.min(total_values)
                    max_total = np.max(total_values)
                    
                    print(f"    {'Total':12s}: {mean_total:8.2f} ± {std_total:6.2f} kJ/mol  "
                          f"[{min_total:7.2f}, {max_total:7.2f}]")
        
        # Comparison table across clusters
        print(f"\n{'='*80}")
        print(f"COMPARISON: Mean Total Energies Across Clusters (kJ/mol)")
        print(f"{'='*80}")
        
        # Get total frame counts for percentages
        total_frames = len(cluster_labels) if hasattr(self, 'cluster_labels') else 0
        
        for g1, g2 in energy_groups:
            print(f"\n{g1} <-> {g2}:")
            header = f"  {'Cluster':>10s} {'N frames':>15s}"
            for comp in components + ['Total']:
                header += f" {comp:>12s}"
            print(header)
            print("  " + "─" * (25 + 13 * (len(components) + 1)))
            
            for cid in sorted(valid_clusters):
                if (g1, g2) in energy_data[cid]:
                    n_frames = len(energy_data[cid][(g1, g2)]['Time'])
                    pct = 100 * n_frames / total_frames if total_frames > 0 else 0
                    row = f"  {cid:>10d} {n_frames:>7d} ({pct:>4.1f}%)"
                    for comp in components + ['Total']:
                        if comp in energy_data[cid][(g1, g2)]:
                            mean_val = np.mean(energy_data[cid][(g1, g2)][comp])
                            row += f" {mean_val:>12.2f}"
                        else:
                            row += f" {'-':>12s}"
                    print(row)
            
            # Add whole-system row
            print("  " + "─" * (25 + 13 * (len(components) + 1)))
            if (g1, g2) in energy_data['all']:
                n_frames_all = len(energy_data['all'][(g1, g2)]['Time'])
                row = f"  {'ALL':>10s} {n_frames_all:>7d} (100.0%)"
                for comp in components + ['Total']:
                    if comp in energy_data['all'][(g1, g2)]:
                        mean_val = np.mean(energy_data['all'][(g1, g2)][comp])
                        row += f" {mean_val:>12.2f}"
                    else:
                        row += f" {'-':>12s}"
                print(row)
        
        print(f"\n{'='*80}\n")
        
        return energy_data
    
    def compute_per_cluster_energies(self,
                                    energy_data: Dict,
                                    statistic: str = 'mean',
                                    compute_stderr: bool = True) -> Dict:
        """
        Compute statistical summary of energies per cluster.
        
        Aggregates energy time series into mean/median/std per cluster.
        Useful for comparing interaction strengths between clusters.
        
        Parameters
        ----------
        energy_data : dict
            Output from compute_energy_decomposition()
        statistic : str, default='mean'
            Statistic to compute: 'mean', 'median', 'std', 'min', 'max'
        compute_stderr : bool, default=True
            Compute standard error of the mean
        
        Returns
        -------
        cluster_stats : dict
            {cluster_id: {(group1, group2): {'Coul-SR': stat, 'LJ-SR': stat, 'Total': stat, 'stderr': {...}}}}
        
        Example
        -------
        >>> cluster_stats = analyzer.compute_per_cluster_energies(
        ...     energy_data,
        ...     statistic='mean'
        ... )
        >>> # Get mean CIP-MMT Coulomb energy for cluster 0
        >>> mean_coul = cluster_stats[0][('CIP', 'MMT')]['Coul-SR']
        >>> stderr = cluster_stats[0][('CIP', 'MMT')]['stderr']['Coul-SR']
        >>> print(f"Coulomb: {mean_coul:.2f} ± {stderr:.2f} kJ/mol")
        """
        stat_func_map = {
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
            'min': np.min,
            'max': np.max
        }
        
        if statistic not in stat_func_map:
            raise ValueError(f"Unknown statistic: {statistic}. Choose from {list(stat_func_map.keys())}")
        
        stat_func = stat_func_map[statistic]
        
        print(f"\n{'='*60}")
        print(f"Per-Cluster Energy Statistics")
        print(f"{'='*60}")
        print(f"Statistic: {statistic}")
        print(f"Standard error: {'Yes' if compute_stderr else 'No'}")
        
        cluster_stats = {}
        
        for cid in sorted(energy_data.keys()):
            cluster_stats[cid] = {}
            print(f"\nCluster {cid}:")
            
            for pair in energy_data[cid]:
                pair_stats = {}
                stderr_stats = {}
                
                for comp in energy_data[cid][pair]:
                    if comp == 'Time':
                        continue
                    
                    values = energy_data[cid][pair][comp]
                    
                    if len(values) == 0:
                        pair_stats[comp] = np.nan
                        stderr_stats[comp] = np.nan
                        continue
                    
                    pair_stats[comp] = stat_func(values)
                    
                    if compute_stderr:
                        stderr_stats[comp] = np.std(values) / np.sqrt(len(values))
                
                cluster_stats[cid][pair] = pair_stats
                if compute_stderr:
                    cluster_stats[cid][pair]['stderr'] = stderr_stats
                
                # Print summary
                g1, g2 = pair
                print(f"  {g1}-{g2}:")
                for comp in ['Coul-SR', 'LJ-SR', 'Total']:
                    if comp in pair_stats:
                        val = pair_stats[comp]
                        if compute_stderr and comp in stderr_stats:
                            err = stderr_stats[comp]
                            print(f"    {comp:10s}: {val:8.2f} ± {err:6.2f} kJ/mol")
                        else:
                            print(f"    {comp:10s}: {val:8.2f} kJ/mol")
        
        print(f"\n✓ Per-cluster statistics computed")
        
        return cluster_stats
    
    def compute_energy_contributions(self, energy_data: Dict) -> Dict:
        """
        Compute percentage contributions of vdW vs Coulomb.
        
        Analyzes relative importance of electrostatic vs van der Waals
        interactions for each cluster and group pair.
        
        Parameters
        ----------
        energy_data : dict
            Output from compute_energy_decomposition()
        
        Returns
        -------
        contributions : dict
            {cluster_id: {(group1, group2): {'vdw_percent': float, 'coulomb_percent': float,
                                             'vdw_mean': float, 'coulomb_mean': float}}}
        
        Example
        -------
        >>> contrib = analyzer.compute_energy_contributions(energy_data)
        >>> vdw_pct = contrib[0][('CIP', 'MMT')]['vdw_percent']
        >>> coul_pct = contrib[0][('CIP', 'MMT')]['coulomb_percent']
        >>> print(f"vdW: {vdw_pct:.1f}%, Coulomb: {coul_pct:.1f}%")
        """
        print(f"\n{'='*60}")
        print(f"Energy Contribution Analysis")
        print(f"{'='*60}")
        
        contributions = {}
        
        for cid in sorted(energy_data.keys()):
            contributions[cid] = {}
            print(f"\nCluster {cid}:")
            
            for pair in energy_data[cid]:
                pair_data = energy_data[cid][pair]
                
                # Get component energies
                vdw = pair_data.get('LJ-SR', np.array([]))
                coulomb = pair_data.get('Coul-SR', np.array([]))
                
                if len(vdw) == 0 or len(coulomb) == 0:
                    contributions[cid][pair] = {
                        'vdw_percent': np.nan,
                        'coulomb_percent': np.nan,
                        'vdw_mean': np.nan,
                        'coulomb_mean': np.nan
                    }
                    continue
                
                vdw_mean = np.mean(vdw)
                coulomb_mean = np.mean(coulomb)
                total_abs = np.abs(vdw_mean) + np.abs(coulomb_mean)
                
                if total_abs == 0:
                    vdw_pct = 50.0
                    coulomb_pct = 50.0
                else:
                    vdw_pct = 100.0 * np.abs(vdw_mean) / total_abs
                    coulomb_pct = 100.0 * np.abs(coulomb_mean) / total_abs
                
                contributions[cid][pair] = {
                    'vdw_percent': vdw_pct,
                    'coulomb_percent': coulomb_pct,
                    'vdw_mean': vdw_mean,
                    'coulomb_mean': coulomb_mean
                }
                
                g1, g2 = pair
                print(f"  {g1}-{g2}:")
                print(f"    vdW:     {vdw_pct:5.1f}% ({vdw_mean:8.2f} kJ/mol)")
                print(f"    Coulomb: {coulomb_pct:5.1f}% ({coulomb_mean:8.2f} kJ/mol)")
        
        print(f"\n✓ Energy contributions computed")
        
        return contributions
    
    def correlate_energy_free_energy(self,
                                    fe_data: Dict,
                                    energy_data: Dict,
                                    energy_group_pair: Tuple[str, str],
                                    component: str = 'Total') -> Dict:
        """
        Correlate cluster free energies with interaction energies.
        
        Tests hypothesis: Lower interaction energy → lower free energy (more stable).
        Computes correlation coefficients between ΔG and mean interaction energies.
        
        Parameters
        ----------
        fe_data : dict
            Output from compute_cluster_free_energies()
        energy_data : dict
            Output from compute_energy_decomposition()
        energy_group_pair : tuple of str
            Which group pair to correlate: ('CIP', 'MMT')
        component : str, default='Total'
            Energy component: 'Coul-SR', 'LJ-SR', 'Total'
        
        Returns
        -------
        correlation : dict
            {'pearson_r': float, 'pearson_p': float, 'spearman_r': float, 'spearman_p': float,
             'cluster_data': {cluster_id: {'delta_G': float, 'energy': float}}}
        
        Example
        -------
        >>> corr = analyzer.correlate_energy_free_energy(
        ...     fe_data,
        ...     energy_data,
        ...     energy_group_pair=('CIP', 'MMT'),
        ...     component='Total'
        ... )
        >>> print(f"Pearson r = {corr['pearson_r']:.3f}, p = {corr['pearson_p']:.3e}")
        """
        from scipy import stats
        
        print(f"\n{'='*60}")
        print(f"Energy-Free Energy Correlation Analysis")
        print(f"{'='*60}")
        print(f"Group pair: {energy_group_pair[0]} <-> {energy_group_pair[1]}")
        print(f"Component: {component}")
        
        # Collect data for each cluster
        cluster_ids = sorted(set(fe_data.keys()) & set(energy_data.keys()))
        
        delta_Gs = []
        energies = []
        cluster_data = {}
        
        for cid in cluster_ids:
            if energy_group_pair not in energy_data[cid]:
                print(f"⚠ Warning: Energy pair {energy_group_pair} not found for cluster {cid}")
                continue
            
            if component not in energy_data[cid][energy_group_pair]:
                print(f"⚠ Warning: Component {component} not found for cluster {cid}")
                continue
            
            dG = fe_data[cid]['delta_G']
            E = np.mean(energy_data[cid][energy_group_pair][component])
            
            delta_Gs.append(dG)
            energies.append(E)
            cluster_data[cid] = {'delta_G': dG, 'energy': E}
        
        if len(delta_Gs) < 3:
            print("✗ Error: Need at least 3 clusters for correlation")
            return {'pearson_r': np.nan, 'pearson_p': np.nan,
                   'spearman_r': np.nan, 'spearman_p': np.nan,
                   'cluster_data': cluster_data}
        
        delta_Gs = np.array(delta_Gs)
        energies = np.array(energies)
        
        # Compute correlations
        pearson_r, pearson_p = stats.pearsonr(energies, delta_Gs)
        spearman_r, spearman_p = stats.spearmanr(energies, delta_Gs)
        
        print(f"\nCorrelation Results:")
        print(f"  Pearson  r = {pearson_r:6.3f}, p = {pearson_p:.3e}")
        print(f"  Spearman r = {spearman_r:6.3f}, p = {spearman_p:.3e}")
        
        if np.abs(pearson_r) > 0.7 and pearson_p < 0.05:
            print(f"  ✓ Strong significant correlation")
        elif np.abs(pearson_r) > 0.5 and pearson_p < 0.05:
            print(f"  ✓ Moderate significant correlation")
        elif pearson_p < 0.05:
            print(f"  ✓ Weak but significant correlation")
        else:
            print(f"  ○ No significant correlation")
        
        print(f"\nCluster Data:")
        for cid in sorted(cluster_data.keys()):
            dG = cluster_data[cid]['delta_G']
            E = cluster_data[cid]['energy']
            print(f"  Cluster {cid}: ΔG = {dG:6.2f} kJ/mol, E = {E:8.2f} kJ/mol")
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'cluster_data': cluster_data
        }
    
    def compute_rdf(self, selection1: Union[str, List[str]], 
                   selection2: Union[str, List[str]], 
                   cluster_ids: Optional[Union[List[int], str]] = None,
                   rmin: float = 0.0, rmax: float = 10.0, 
                   nbins: Optional[int] = None,
                   dr: Optional[float] = None,
                   n_jobs: int = 1,
                   save_cache: bool = False,
                   force_rerun: bool = False,
                   cache_dir: str = './rdf_cache'):
        """
        Compute radial distribution function (RDF) for clusters.
        
        Parameters
        ----------
        selection1 : str or list of str
            MDAnalysis selection string(s) for reference group.
            - Single string: One RDF calculation
            - List: Batch compute RDF for all selection1/selection2 pairs
        selection2 : str or list of str
            MDAnalysis selection string(s) for target group.
            - Single string: One RDF calculation
            - List: Batch compute RDF for all selection1/selection2 pairs
        cluster_ids : list of int, 'all', or None, optional
            Cluster IDs to analyze. 
            - None or 'all': Analyzes all loaded clusters
            - List of int: Analyzes specified clusters only
        rmin : float
            Minimum distance for RDF (Angstroms, default=0.0)
        rmax : float
            Maximum distance for RDF (Angstroms, default=10.0)
        nbins : int, optional
            Number of bins for RDF histogram (default=None).
            If both nbins and dr are None, defaults to 200 bins.
        dr : float, optional
            Bin width in Angstroms (default=None).
            If specified, nbins is calculated as: nbins = int((rmax - rmin) / dr).
            Takes precedence over nbins parameter.
            Recommended: dr=0.1 for smoother RDF curves with less noise.
        n_jobs : int
            DEPRECATED: Parallel execution is disabled due to thread-safety issues.
            This parameter is ignored. All RDF computations run serially.
        save_cache : bool
            If True, save computed RDF results to cache file (default=False).
            Cache enables fast reloading without recomputation.
        force_rerun : bool
            If True, recompute RDF even if valid cache exists (default=False).
            Use to regenerate cache with updated trajectories.
        cache_dir : str
            Directory for cache files (default='./rdf_cache').
            Created automatically if it doesn't exist.
            
        Returns
        -------
        rdf_results : dict
            - Single pair: {cluster_id: {'r': distances, 'rdf': g(r) values, ...}}
            - Batch: {sel1: {sel2: {cluster_id: {...}}}}
        
        Examples
        --------
        Single pair:
        >>> rdfs = analyzer.compute_rdf('resname CIP', 'resname SOL and name OW')
        >>> r = rdfs[0]['r']  # distances for cluster 0
        
        Batch analysis (serial):
        >>> analyzer.define_selections({
        ...     'CIP_parts': {
        ...         'quinolone': 'resname api and (name N6 or name C10 or name C11 or name C12 or name C19 or name C21 or name C22 or name C23 or name C4 or name C5)',
        ...         'piperazine': 'resname api and (name N13 or name N16 or name C14 or name C15 or name C17 or name C18)'
        ...     },
        ...     'solvent': {'water_O': 'resname SOL and name OW'}
        ... })
        >>> rdfs = analyzer.compute_rdf(
        ...     ['quinolone', 'piperazine'],
        ...     'water_O'
        ... )
        >>> # Access: rdfs['quinolone']['water_O'][0]['rdf']
        
        Batch analysis:
        >>> rdfs = analyzer.compute_rdf(
        ...     ['quinolone', 'piperazine', 'carboxylic_acid', 'fluoride'],
        ...     ['water_o', 'surface_o']
        ... )
        
        With caching (save and reuse results):
        >>> rdfs = analyzer.compute_rdf(
        ...     ['quinolone', 'piperazine'],
        ...     'water_o',
        ...     save_cache=True  # Save results to cache
        ... )
        >>> # Later, reload from cache (instant):
        >>> rdfs = analyzer.compute_rdf(
        ...     ['quinolone', 'piperazine'],
        ...     'water_o',
        ...     save_cache=True  # Loads from cache automatically
        ... )
        
        Using dr for smoother RDF (less noise):
        >>> rdfs = analyzer.compute_rdf(
        ...     ['quinolone', 'piperazine'],
        ...     'water_o',
        ...     rmax=10.0,
        ...     dr=0.1  # 0.1 Å bin width -> 100 bins
        ... )
        """
        # Calculate nbins from dr if specified
        if dr is not None:
            nbins = int((rmax - rmin) / dr)
            print(f"Using dr={dr:.3f} Å → nbins={nbins}")
        elif nbins is None:
            nbins = 200  # Default value
        
        # Generate cache key and check for existing cache
        cache_file = None
        if save_cache:
            cache_file = self._get_cache_filename(
                selection1, selection2, cluster_ids, rmin, rmax, nbins, cache_dir
            )
            
            if not force_rerun and os.path.exists(cache_file):
                print(f"\n{'='*60}")
                print(f"LOADING RDF FROM CACHE")
                print(f"{'='*60}")
                print(f"Cache file: {os.path.basename(cache_file)}")
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    print(f"✓ Successfully loaded cached RDF data")
                    print(f"{'='*60}\n")
                    return cached_data
                except Exception as e:
                    print(f"⚠ Failed to load cache: {e}")
                    print(f"Recomputing RDF...\n")
        
        # Handle batch processing
        if isinstance(selection1, list) or isinstance(selection2, list):
            results = self._compute_rdf_batch(
                selection1, selection2, cluster_ids, rmin, rmax, nbins, n_jobs
            )
        else:
            # Single pair processing (original implementation)
            results = self._compute_rdf_single(
                selection1, selection2, cluster_ids, rmin, rmax, nbins
            )
        
        # Save to cache if requested
        if save_cache and cache_file:
            self._save_rdf_cache(results, cache_file)
        
        return results
    
    def _compute_rdf_single(self, selection1: str, selection2: str,
                           cluster_ids: Optional[Union[List[int], str]],
                           rmin: float, rmax: float, nbins: int):
        """Internal method for single RDF computation."""
        try:
            from MDAnalysis.analysis import rdf as mda_rdf
        except ImportError:
            raise ImportError("MDAnalysis.analysis.rdf module required")
        
        if not hasattr(self, 'trajectory_data'):
            raise ValueError(
                "No trajectory data loaded. Run load_cluster_trajectories() first."
            )
        
        # Handle cluster_ids parameter
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = list(self.trajectory_data.keys())
        elif not isinstance(cluster_ids, list):
            raise ValueError(
                f"cluster_ids must be None, 'all', or a list of integers, got {type(cluster_ids)}"
            )
        
        # Resolve selection names if they reference defined selections
        sel1 = self._selections.get(selection1, selection1)
        sel2 = self._selections.get(selection2, selection2)
        
        print(f"\nComputing RDF: {selection1} <-> {selection2}")
        if sel1 != selection1 or sel2 != selection2:
            print(f"  Resolved to: '{sel1}' <-> '{sel2}'")
        print(f"  Distance range: {rmin:.1f}-{rmax:.1f} Å")
        print(f"  Bins: {nbins}")
        
        rdf_results = {}
        
        for cluster_id in cluster_ids:
            print(f"\n  Cluster {cluster_id}:")
            u = self.trajectory_data[cluster_id]['universe']
            frames = self.trajectory_data[cluster_id]['frames']
            
            # Create selections
            try:
                group1 = u.select_atoms(sel1)
                group2 = u.select_atoms(sel2)
            except Exception as e:
                print(f"    ⚠ Selection error: {e}")
                continue
            
            print(f"    Group 1: {len(group1)} atoms")
            print(f"    Group 2: {len(group2)} atoms")
            print(f"    Analyzing {len(frames)} frames...")
            
            # Compute RDF using only cluster frames
            rdf_analysis = mda_rdf.InterRDF(
                group1, group2,
                nbins=nbins,
                range=(rmin, rmax)
            )
            
            # Run over specific frames (use frames parameter only, not start/stop/step)
            rdf_analysis.run(frames=frames)
            
            # Store results
            rdf_results[cluster_id] = {
                'r': rdf_analysis.results.bins,  # Distance bins (midpoints)
                'rdf': rdf_analysis.results.rdf,  # g(r) values
                'count': rdf_analysis.results.count,  # Running coordination number
                'selection1': selection1,
                'selection2': selection2,
                'n_frames': len(frames)
            }
            
            # Report first peak info
            r = rdf_analysis.results.bins
            gr = rdf_analysis.results.rdf
            first_peak_idx = np.argmax(gr[:len(gr)//2])
            first_peak_r = r[first_peak_idx]
            first_peak_gr = gr[first_peak_idx]
            print(f"    ✓ First peak at r={first_peak_r:.2f} Å, g(r)={first_peak_gr:.2f}")
        
        # Store for later access
        if not hasattr(self, 'rdf_data'):
            self.rdf_data = {}
        self.rdf_data[f"{selection1}__{selection2}"] = rdf_results
        
        print(f"\n✓ RDF computed for {len(rdf_results)} clusters")
        return rdf_results
    
    def _get_cache_filename(self, selection1, selection2, cluster_ids, 
                           rmin, rmax, nbins, cache_dir):
        """
        Generate unique cache filename based on RDF parameters.
        
        Uses MD5 hash of parameters to create reproducible cache keys.
        """
        # Normalize cluster_ids for consistent hashing
        if cluster_ids is None or cluster_ids == 'all':
            cluster_key = 'all'
        else:
            cluster_key = ','.join(map(str, sorted(cluster_ids)))
        
        # Normalize selections (handle both single strings and lists)
        if isinstance(selection1, list):
            sel1_key = '|'.join(sorted(selection1))
        else:
            sel1_key = selection1
            
        if isinstance(selection2, list):
            sel2_key = '|'.join(sorted(selection2))
        else:
            sel2_key = selection2
        
        # Calculate dr for cache key (more intuitive than nbins)
        dr_value = (rmax - rmin) / nbins
        
        # Create cache key string
        cache_key = f"{sel1_key}__{sel2_key}__clusters_{cluster_key}__r{rmin}-{rmax}__dr{dr_value:.4f}"
        
        # Generate MD5 hash for shorter filename
        hash_obj = hashlib.md5(cache_key.encode())
        hash_str = hash_obj.hexdigest()[:16]  # Use first 16 chars
        
        # Create descriptive filename with hash
        if isinstance(selection1, list) and isinstance(selection2, list):
            desc = f"batch_{len(selection1)}x{len(selection2)}_pairs"
        elif isinstance(selection1, list):
            desc = f"batch_{len(selection1)}_vs_{selection2}"
        elif isinstance(selection2, list):
            desc = f"{selection1}_vs_batch_{len(selection2)}"
        else:
            desc = f"{selection1}_vs_{selection2}"
        
        # Clean description for filename safety
        desc = desc.replace(' ', '_').replace('/', '_').replace('\\', '_')
        desc = ''.join(c for c in desc if c.isalnum() or c in ('_', '-'))[:50]
        
        filename = f"rdf_{desc}_{hash_str}.pkl"
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        return os.path.join(cache_dir, filename)
    
    def _save_rdf_cache(self, rdf_data, cache_file):
        """
        Save RDF results to cache file using pickle.
        
        The data structure is preserved exactly as computed, allowing
        seamless reloading without any modifications needed.
        """
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(rdf_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Get file size for user feedback
            size_bytes = os.path.getsize(cache_file)
            size_kb = size_bytes / 1024
            size_mb = size_kb / 1024
            
            if size_mb >= 1:
                size_str = f"{size_mb:.2f} MB"
            else:
                size_str = f"{size_kb:.2f} KB"
            
            print(f"\n{'='*60}")
            print(f"✓ RDF CACHE SAVED")
            print(f"{'='*60}")
            print(f"File: {os.path.basename(cache_file)}")
            print(f"Size: {size_str}")
            print(f"Path: {cache_file}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"\n⚠ Failed to save cache: {e}\n")
    
    def _compute_rdf_batch(self, selection1: Union[str, List[str]], 
                          selection2: Union[str, List[str]],
                          cluster_ids: Optional[Union[List[int], str]],
                          rmin: float, rmax: float, nbins: int, n_jobs: int = 1):
        """Internal method for batch RDF computation with optional parallelization."""
        import os
        
        # CRITICAL: MDAnalysis Universe objects are NOT thread-safe!
        # Parallel execution causes data corruption and garbage RDF values.
        # Force serial execution until proper thread-safe implementation.
        if n_jobs > 1:
            print(f"\n{'='*60}")
            print(f"⚠ WARNING: Parallel RDF computation disabled")
            print(f"{'='*60}")
            print(f"  MDAnalysis trajectory access is not thread-safe.")
            print(f"  Parallel execution causes data corruption.")
            print(f"  Forcing n_jobs=1 (serial execution).")
            print(f"{'='*60}\n")
            n_jobs = 1
        
        # Ensure both are lists for uniform processing
        sel1_list = [selection1] if isinstance(selection1, str) else selection1
        sel2_list = [selection2] if isinstance(selection2, str) else selection2
        
        total_pairs = len(sel1_list) * len(sel2_list)
        
        print(f"\n{'='*60}")
        print(f"BATCH RDF COMPUTATION")
        print(f"{'='*60}")
        print(f"  Selection 1: {len(sel1_list)} items")
        print(f"  Selection 2: {len(sel2_list)} items")
        print(f"  Total pairs: {total_pairs}")
        print(f"{'='*60}\n")
        
        # Build list of all pairs to process
        pairs_to_compute = []
        for sel1_name in sel1_list:
            for sel2_name in sel2_list:
                pairs_to_compute.append((sel1_name, sel2_name))
        
        # Serial execution (always, due to thread-safety issues)
        batch_results = {}
        pair_count = 0
        
        for sel1_name in sel1_list:
            batch_results[sel1_name] = {}
            
            for sel2_name in sel2_list:
                pair_count += 1
                print(f"\n[Pair {pair_count}/{total_pairs}] {sel1_name} <-> {sel2_name}")
                print("-" * 60)
                
                rdf_result = self._compute_rdf_single(
                    sel1_name, sel2_name, cluster_ids, rmin, rmax, nbins
                )
                batch_results[sel1_name][sel2_name] = rdf_result
        
        print(f"\n{'='*60}")
        print(f"✓ BATCH COMPLETE: {total_pairs} RDF pairs computed")
        print(f"{'='*60}\n")
        
        return batch_results
    
    def compute_distance_distribution(self, selection1, selection2: str,
                                     cluster_ids: Optional[List[int]] = None,
                                     bins: int = 100, range_dist: Tuple[float, float] = (0, 15),
                                     normalize: bool = True):
        """
        Compute minimum distance distributions between two selections.
        
        Parameters
        ----------
        selection1 : str or dict
            MDAnalysis selection for first group, or dict of {name: selection_string}
            e.g., {'carboxylic_acid': 'resname API and ...', 'quinolone': '...'}
        selection2 : str
            MDAnalysis selection for second group
        cluster_ids : list of int, optional
            Clusters to analyze (default: all)
        bins : int
            Number of histogram bins (default=100)
        range_dist : tuple of float
            Distance range in Angstroms (default=(0, 15))
        normalize : bool
            If True, normalize histogram by number of frames for cross-cluster comparison (default=True)
            Allows comparing distributions from clusters with different numbers of frames
            
        Returns
        -------
        dist_results : dict
            {cluster_id: {'distances': array of min distances per frame,
                          'hist': histogram counts (raw),
                          'hist_normalized': histogram normalized by n_frames (if normalize=True),
                          'bin_edges': bin edges,
                          'bin_centers': bin centers,
                          'n_frames': number of frames,
                          'mean': mean distance, 'std': standard deviation}}
        
        Example
        -------
        >>> # Single selection
        >>> dists = analyzer.compute_distance_distribution(
        ...     'resname CIP',
        ...     'resname MMT and name Ob',
        ...     normalize=True
        ... )
        
        >>> # Multiple selections at once
        >>> dists = analyzer.compute_distance_distribution(
        ...     {'carboxylic_acid': analyzer.sel('carboxylic_acid'),
        ...      'quinolone': analyzer.sel('quinolone'),
        ...      'piperazine': analyzer.sel('piperazine')},
        ...     analyzer.sel('surface_o'),
        ...     normalize=True
        ... )
        """
        if not hasattr(self, 'trajectory_data'):
            raise ValueError("Load trajectories first with load_cluster_trajectories()")
        
        # Handle cluster_ids='all' or None
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = sorted(self.trajectory_data.keys())
        
        # Handle dict of selections
        if isinstance(selection1, dict):
            all_results = {}
            for name, sel in selection1.items():
                print(f"\n{'='*60}")
                print(f"Processing: {name}")
                result = self._compute_single_distance_distribution(
                    sel, selection2, cluster_ids, bins, range_dist, normalize, name
                )
                all_results[name] = result
            return all_results
        else:
            # Single selection (original behavior)
            return self._compute_single_distance_distribution(
                selection1, selection2, cluster_ids, bins, range_dist, normalize
            )
    
    def _compute_single_distance_distribution(self, selection1: str, selection2: str,
                                             cluster_ids: List[int],
                                             bins: int, range_dist: Tuple[float, float],
                                             normalize: bool, name: str = None):
        """Internal method to compute distance distribution for a single selection pair."""
        from MDAnalysis.analysis import distances
        print(f"\nComputing distance distributions:")
        print(f"  Group 1: {selection1}")
        print(f"  Group 2: {selection2}")
        print(f"  Range: {range_dist[0]:.1f}-{range_dist[1]:.1f} Å")
        
        dist_results = {}
        
        for cluster_id in cluster_ids:
            print(f"\n  Cluster {cluster_id}:")
            u = self.trajectory_data[cluster_id]['universe']
            frames = self.trajectory_data[cluster_id]['frames']
            
            # Create selections
            group1 = u.select_atoms(selection1)
            group2 = u.select_atoms(selection2)
            
            print(f"    Group 1: {len(group1)} atoms")
            print(f"    Group 2: {len(group2)} atoms")
            
            # Compute minimum distances for each frame
            min_distances = []
            for frame_idx in frames:
                u.trajectory[frame_idx]
                
                # Compute all pairwise distances with PBC
                dists = distances.distance_array(group1.positions, group2.positions, box=u.dimensions)
                min_dist = dists.min()
                min_distances.append(min_dist)
            
            min_distances = np.array(min_distances)
            n_frames = len(min_distances)
            
            # Create histogram
            hist, bin_edges = np.histogram(min_distances, bins=bins, range=range_dist)
            
            # Normalize if requested (for cross-cluster comparison)
            if normalize:
                hist_normalized = hist / n_frames
            else:
                hist_normalized = None
            
            # Statistics
            mean_dist = np.mean(min_distances)
            std_dist = np.std(min_distances)
            median_dist = np.median(min_distances)
            
            dist_results[cluster_id] = {
                'distances': min_distances,
                'hist': hist,
                'hist_normalized': hist_normalized,
                'bin_edges': bin_edges,
                'bin_centers': (bin_edges[:-1] + bin_edges[1:]) / 2,
                'n_frames': n_frames,
                'mean': mean_dist,
                'std': std_dist,
                'median': median_dist,
                'selection1': selection1,
                'selection2': selection2
            }
            
            print(f"    ✓ Frames: {n_frames}")
            print(f"      Mean: {mean_dist:.2f} ± {std_dist:.2f} Å")
            print(f"      Median: {median_dist:.2f} Å")
            if normalize:
                print(f"      Normalized: counts/frame (max={hist_normalized.max():.4f})")
        
        # Store for later access
        if not hasattr(self, 'distance_data'):
            self.distance_data = {}
        
        # Use simple name if provided, otherwise full selection string
        key = name if name else f"{selection1}__{selection2}"
        self.distance_data[key] = dist_results
        
        print(f"\n✓ Distance distributions computed (key: '{key}')")
        return dist_results
    
    def export_distance_data_for_multi_system(self, system_name: str, 
                                              output_file: Optional[Union[str, Path]] = None,
                                              cluster_ids: Optional[Union[str, List[int]]] = None) -> Dict:
        """
        Export distance distribution data for multi-system comparison plots.
        
        This method prepares and exports distance distribution data in the format
        required by RMSDPlotter.plot_multi_system_distance_distributions_3d().
        
        Parameters
        ----------
        system_name : str
            Name of the system (e.g., 'CIP+', 'CIP+/-', 'CIP-')
        output_file : str or Path, optional
            Path to save the exported data as a pickle file.
            If None, only returns the data without saving.
            Example: 'distance_data_CIPplus.pkl'
        cluster_ids : list of int, 'all', or None, optional
            Clusters to include in export. If 'all' or None, exports all available clusters.
        
        Returns
        -------
        export_data : dict
            Dictionary with structure:
            {
                'system_name': str,
                'distance_data': dict of distance distributions,
                'cluster_ids': list of cluster IDs,
                'available_keys': list of distance data keys
            }
        
        Raises
        ------
        ValueError
            If distance_data not available (run compute_distance_distribution first)
        
        Example
        -------
        >>> # After computing distance distributions
        >>> analyzer.compute_distance_distribution(
        ...     {'carboxylic_acid': sel1, 'quinolone': sel2},
        ...     'resname MMT and name Ob'
        ... )
        >>> 
        >>> # Export for multi-system plotting
        >>> data = analyzer.export_distance_data_for_multi_system(
        ...     system_name='CIP+',
        ...     output_file='distance_data_CIPplus.pkl'
        ... )
        >>> 
        >>> # Later, load and plot multiple systems
        >>> systems_data = {}
        >>> for name, file in [('CIP+', 'distance_data_CIPplus.pkl'), ...]:
        ...     with open(file, 'rb') as f:
        ...         systems_data[name] = pickle.load(f)
        >>> 
        >>> fig = plotter.plot_multi_system_distance_distributions_3d(
        ...     systems_data, dist_key='carboxylic_acid'
        ... )
        """
        # Check if distance data exists
        if not hasattr(self, 'distance_data') or not self.distance_data:
            raise ValueError(
                "No distance data available. Run compute_distance_distribution() first."
            )
        
        # Get all available distance data keys
        available_keys = list(self.distance_data.keys())
        
        # Determine cluster IDs from the first distance key
        first_key = available_keys[0]
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = sorted(self.distance_data[first_key].keys())
        
        # Prepare export data
        export_data = {
            'system_name': system_name,
            'distance_data': self.distance_data,
            'cluster_ids': cluster_ids,
            'available_keys': available_keys
        }
        
        # Save to file if requested
        if output_file is not None:
            output_path = Path(output_file)
            with open(output_path, 'wb') as f:
                pickle.dump(export_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\n✓ Distance data exported to: {output_path}")
            print(f"  System: {system_name}")
            print(f"  Clusters: {cluster_ids}")
            print(f"  Distance keys: {available_keys}")
        else:
            print(f"\n✓ Distance data prepared for system: {system_name}")
            print(f"  Clusters: {cluster_ids}")
            print(f"  Distance keys: {available_keys}")
        
        return export_data

    def export_coordination_data_for_multi_system(self, system_name: str,
                                                  output_file: Optional[Union[str, Path]] = None,
                                                  cluster_ids: Optional[Union[str, List[int]]] = None,
                                                  coord_keys: Optional[List[str]] = None,
                                                  moiety_aliases: Optional[Dict[str, str]] = None) -> Dict:
        """
        Export coordination number data for multi-system comparison plots.

        This method prepares and exports coordination data in the format required by
        RMSDPlotter.plot_multi_system_coordination_barplot().

        Parameters
        ----------
        system_name : str
            Name of the system (e.g., 'CIP+', 'CIP+/-', 'CIP-')
        output_file : str or Path, optional
            Path to save the exported data as a pickle file.
            If None, only returns the data without saving.
            Example: 'coordination_data_CIPplus.pkl'
        cluster_ids : list of int, 'all', or None, optional
            Clusters to include in export. If 'all' or None, exports all available clusters.
        coord_keys : list of str, optional
            Coordination keys to include. If None, exports all available keys.
        moiety_aliases : dict, optional
            Human-readable name → coordination key mapping for this system.
            This is the **recommended** way to handle systems that may use slightly
            different cutoffs so the multi-system plotting notebook can always
            look up keys by name instead of by cutoff value.
            Example::

                moiety_aliases={
                    'Carboxylic Acid': f"{analyzer.sel('carboxylic_acid')}__{analyzer.sel('Na')}__2.35",
                    'Quinolone':       f"{analyzer.sel('quinolone')}__{analyzer.sel('Na')}__3.35",
                    'Piperazine':      f"{analyzer.sel('piperazine')}__{analyzer.sel('Na')}__3.95",
                    'Cyclopropyl':     f"{analyzer.sel('cyclopropyl')}__{analyzer.sel('Na')}__3.85",
                }

        Returns
        -------
        export_data : dict
            Dictionary with structure:
            {
                'system_name': str,
                'coordination_data': dict of coordination results,
                'cluster_ids': list of cluster IDs,
                'available_keys': list of coordination data keys,
                'moiety_aliases': dict of {moiety_name: coord_key} (empty if not provided)
            }

        Raises
        ------
        ValueError
            If coordination_data not available (run compute_coordination_numbers first)

        Example
        -------
        >>> analyzer.compute_coordination_numbers(
        ...     center_selection='resname CIP and name O1 O3',
        ...     neighbor_selection='resname NA',
        ...     cutoff=3.5
        ... )
        >>>
        >>> data = analyzer.export_coordination_data_for_multi_system(
        ...     system_name='CIP+',
        ...     output_file='coordination_data_CIPplus.pkl'
        ... )
        >>>
        >>> # Later, load and plot multiple systems
        >>> systems_data = {}
        >>> for name, file in [('CIP+', 'coordination_data_CIPplus.pkl'), ...]:
        ...     with open(file, 'rb') as f:
        ...         systems_data[name] = pickle.load(f)
        >>>
        >>> fig = multi_plotter.plot_multi_system_coordination_barplot(
        ...     systems_data=systems_data,
        ...     coord_key='resname CIP and name O1 O3__resname NA__3.5'
        ... )
        """
        if not hasattr(self, 'coordination_data') or not self.coordination_data:
            raise ValueError(
                "No coordination data available. Run compute_coordination_numbers() first."
            )

        # Determine keys to export
        all_available_keys = list(self.coordination_data.keys())
        if coord_keys is not None:
            available_keys = [k for k in coord_keys if k in self.coordination_data]
            missing = [k for k in coord_keys if k not in self.coordination_data]
            if missing:
                print(f"WARNING: The following coord_keys were not found and will be skipped:\n"
                      + "\n".join(f"  - {k}" for k in missing))
        else:
            available_keys = all_available_keys

        if not available_keys:
            raise ValueError("No valid coordination keys found to export.")

        # Filter coordination data to requested keys
        filtered_data = {k: self.coordination_data[k] for k in available_keys}

        # Determine cluster IDs from the first available key
        first_key = available_keys[0]
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = sorted(filtered_data[first_key].keys())
        elif isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]
        else:
            cluster_ids = sorted(cluster_ids)

        export_data = {
            'system_name': system_name,
            'coordination_data': filtered_data,
            'cluster_ids': cluster_ids,
            'available_keys': available_keys,
            'moiety_aliases': {},
        }

        # Validate and store moiety_aliases if provided
        if moiety_aliases is not None:
            validated_aliases = {}
            for alias_name, alias_key in moiety_aliases.items():
                if alias_key not in filtered_data:
                    print(f"WARNING: moiety_aliases['{alias_name}'] = '{alias_key}' "
                          f"not found in exported keys — skipping.")
                else:
                    validated_aliases[alias_name] = alias_key
            export_data['moiety_aliases'] = validated_aliases

        if output_file is not None:
            output_path = Path(output_file)
            with open(output_path, 'wb') as f:
                pickle.dump(export_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\n✓ Coordination data exported to: {output_path}")
            print(f"  System: {system_name}")
            print(f"  Clusters: {cluster_ids}")
            print(f"  Coordination keys: {available_keys}")
            if export_data['moiety_aliases']:
                print(f"  Moiety aliases:")
                for alias_name, alias_key in export_data['moiety_aliases'].items():
                    print(f"    '{alias_name}' → '{alias_key}'")
        else:
            print(f"\n✓ Coordination data prepared for system: {system_name}")
            print(f"  Clusters: {cluster_ids}")
            print(f"  Coordination keys: {available_keys}")
            if export_data['moiety_aliases']:
                print(f"  Moiety aliases:")
                for alias_name, alias_key in export_data['moiety_aliases'].items():
                    print(f"    '{alias_name}' → '{alias_key}'")

        return export_data

    def export_rdf_data_for_multi_system(self,
                                          rdf_data,
                                          system_name: str) -> Dict:
        """
        Export RDF data in the format required by
        ``RMSDPlotter.plot_multi_system_rdfs()``.

        Call this after ``compute_rdf()`` to package the result into a portable
        dict; then combine dicts from multiple systems and pass to
        ``plot_multi_system_rdfs()``.

        Parameters
        ----------
        rdf_data : dict
            Output of ``compute_rdf()``.  Handles all formats:

            1. Batch 3-level : ``{sel1: {sel2: {cluster_id: {'r','rdf',...}}}}``
            2. Per-selection : ``{sel_name: {cluster_id: {'r','rdf',...}}}``
            3. Single-cluster: ``{name: {'r','rdf',...}}`` → cluster 0
            4. Cluster dict  : ``{cluster_id: {'r','rdf',...}}`` → single sel

        system_name : str
            Label for this system, e.g. ``'CIP+'``.

        Returns
        -------
        dict
            ``{'system_name': str,
               'cluster_ids': list,
               'selection_names': list,
               'rdf_by_cluster': {cluster_id: {sel_name: {'r','rdf','n_frames','count',...}}}}``

        Examples
        --------
        >>> rdf_ow = analyzer.compute_rdf(['quinolone','piperazine'], 'water_o',
        ...                               cluster_ids='all', rmax=8.0)
        >>> export_rdf_ow = analyzer.export_rdf_data_for_multi_system(rdf_ow, 'CIP+')
        >>> with open('rdf_ow_CIPplus.pkl', 'wb') as f:
        ...     pickle.dump(export_rdf_ow, f)
        """
        cluster_data: Dict = {}
        selection_names: List = []

        if isinstance(rdf_data, list):
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

            # Case: {name: {'r', 'rdf'}} — single cluster
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

                # Case: {cluster_id: {'r', 'rdf'}} — cluster dict, single sel
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

                    # Case: {sel1: {sel2: {cid: data}}} — batch 3-level
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
                    # Case: {sel_name: {cid: data}} — per-selection
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

    def compute_coordination_numbers(self, center_selection: str,
                                     neighbor_selection: str,
                                     cutoff: float = 3.5,
                                     cluster_ids: Optional[Union[str, List[int]]] = None):
        """
        Compute coordination numbers (atoms within cutoff distance).
        
        Parameters
        ----------
        center_selection : str
            Selection for center atoms (e.g., 'resname CIP and name O1 O3')
        neighbor_selection : str
            Selection for neighbors (e.g., 'resname NA')
        cutoff : float
            Distance cutoff in Angstroms (default=3.5)
        cluster_ids : list of int, 'all', or None, optional
            Clusters to analyze. If 'all', analyzes all clusters.
            If None, analyzes all clusters.
            
        Returns
        -------
        coord_results : dict
            {cluster_id: {'coordination': array of CN per frame,
                          'mean_cn': mean coordination number,
                          'std_cn': standard deviation,
                          'median_cn': median coordination number,
                          'cutoff': cutoff distance used,
                          'center_selection': center selection string,
                          'neighbor_selection': neighbor selection string}}
        
        Example
        -------
        >>> # How many Na+ near CIP carboxylate?
        >>> cn = analyzer.compute_coordination_numbers(
        ...     'resname CIP and (name O1 or name O3)',
        ...     'resname NA',
        ...     cutoff=3.5,
        ...     cluster_ids='all'
        ... )
        """
        if not hasattr(self, 'trajectory_data'):
            raise ValueError("Load trajectories first")
        
        # Handle cluster_ids='all' or None
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = sorted(self.trajectory_data.keys())
        
        print(f"\nComputing coordination numbers:")
        print(f"  Center: {center_selection}")
        print(f"  Neighbors: {neighbor_selection}")
        print(f"  Cutoff: {cutoff:.2f} Å")
        
        coord_results = {}
        
        for cluster_id in cluster_ids:
            print(f"\n  Cluster {cluster_id}:")
            u = self.trajectory_data[cluster_id]['universe']
            frames = self.trajectory_data[cluster_id]['frames']
            
            center = u.select_atoms(center_selection)
            neighbors = u.select_atoms(neighbor_selection)
            
            print(f"    Center atoms: {len(center)}")
            print(f"    Neighbor atoms: {len(neighbors)}")
            
            coordination = []
            for frame_idx in frames:
                u.trajectory[frame_idx]
                pos_center = center.positions
                pos_neighbors = neighbors.positions
                
                # Count neighbors within cutoff
                dists = cdist(pos_center, pos_neighbors)
                cn = np.sum(dists <= cutoff, axis=1).sum()  # Total CN
                coordination.append(cn)
            
            coordination = np.array(coordination)
            
            coord_results[cluster_id] = {
                'coordination': coordination,
                'mean_cn': np.mean(coordination),
                'std_cn': np.std(coordination),
                'median_cn': np.median(coordination),
                'cutoff': cutoff,
                'center_selection': center_selection,
                'neighbor_selection': neighbor_selection
            }
            
            print(f"    ✓ Mean CN: {coord_results[cluster_id]['mean_cn']:.2f} ± {coord_results[cluster_id]['std_cn']:.2f}")
        
        # Store
        if not hasattr(self, 'coordination_data'):
            self.coordination_data = {}
        key = f"{center_selection}__{neighbor_selection}__{cutoff}"
        self.coordination_data[key] = coord_results
        
        print(f"\n✓ Coordination numbers computed")
        return coord_results
    
    def _detect_separate_rings(self, atom_group, min_ring_size: int = 5):
        """
        Detect separate rings in a molecular fragment using RDKit.
        
        For fused ring systems (e.g., quinolone = 2 fused 6-membered rings),
        this identifies each ring separately and returns atom indices.
        
        Parameters
        ----------
        atom_group : MDAnalysis.AtomGroup
            Atoms to analyze for ring detection
        min_ring_size : int
            Minimum ring size to consider (default=5)
            
        Returns
        -------
        rings : list of list
            Each sublist contains atom indices (in atom_group) for one ring.
            Empty list if RDKit not available or no rings detected.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            print("    ⚠ RDKit not available - using all atoms as single ring")
            return [list(range(len(atom_group)))]
        
        # Build RDKit molecule from atom connectivity
        mol = Chem.RWMol()
        atom_idx_map = {}  # MDAnalysis idx -> RDKit idx
        
        # Add atoms
        for i, atom in enumerate(atom_group):
            # Get element symbol - try atom.element first, then guess from name
            try:
                element = atom.element if hasattr(atom, 'element') and atom.element else None
            except:
                element = None
            
            # If element not available, guess from atom name (e.g., 'C10' -> 'C')
            if not element or element == '':
                import re
                # Extract letters from atom name (e.g., 'C10' -> 'C', 'N6' -> 'N')
                match = re.match(r'([A-Z][a-z]?)', atom.name)
                if match:
                    element = match.group(1)
                else:
                    print(f"    ⚠ Cannot determine element for atom {atom.name} - using Carbon")
                    element = 'C'
            
            try:
                rd_atom = Chem.Atom(element)
                rd_idx = mol.AddAtom(rd_atom)
                atom_idx_map[atom.index] = rd_idx
            except Exception as e:
                print(f"    ⚠ Error adding atom {atom.name} (element: {element}): {e}")
                print(f"    Falling back to single ring (all atoms)")
                return [list(range(len(atom_group)))]
        
        # Add bonds (use MDAnalysis topology)
        for bond in atom_group.bonds:
            atom1_md_idx = bond.atoms[0].index
            atom2_md_idx = bond.atoms[1].index
            
            # Only add bond if both atoms are in our selection
            if atom1_md_idx in atom_idx_map and atom2_md_idx in atom_idx_map:
                rd_idx1 = atom_idx_map[atom1_md_idx]
                rd_idx2 = atom_idx_map[atom2_md_idx]
                mol.AddBond(rd_idx1, rd_idx2, Chem.BondType.SINGLE)
        
        # Convert to Mol object
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        
        # Detect rings
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        # Filter by size and convert back to atom_group indices
        detected_rings = []
        rd_to_ag_map = {v: k for k, v in atom_idx_map.items()}  # RDKit idx -> MDAnalysis idx
        
        for ring in atom_rings:
            if len(ring) >= min_ring_size:
                # Convert RDKit indices to atom_group array indices
                ag_indices = []
                for rd_idx in ring:
                    md_idx = rd_to_ag_map[rd_idx]
                    # Find position in atom_group
                    ag_idx = np.where(atom_group.indices == md_idx)[0][0]
                    ag_indices.append(ag_idx)
                detected_rings.append(ag_indices)
        
        if len(detected_rings) == 0:
            print("    ⚠ No rings detected - using all atoms")
            return [list(range(len(atom_group)))]
        
        return detected_rings
    
    def compute_pi_cation_interactions(self, ring_selection: str,
                                      cation_selection: str,
                                      clay_sel: Optional[str] = None,
                                      cluster_ids: Optional[List[int]] = None,
                                      require_z_ordering: bool = False,
                                      distance_cutoff: float = 5.0,
                                      max_clay_distance: Optional[float] = None,
                                      angle_cutoff: float = 30.0,
                                      max_ring_clay_angle: Optional[float] = None,
                                      detect_rings: bool = True,
                                      extract_trajectory_frames: bool = False,
                                      topology_file: str = None,
                                      trajectory_file: str = None,
                                      output_dir: str = None):
        """
        Compute π-cation interactions between aromatic rings and cations.
        
        Detects and analyzes π-cation interactions by measuring:
        - Distance from cation to ring center (centroid)
        - Distance from cation to nearest ring atom  
        - Angle between cation-centroid vector and ring plane normal
        - Contact frequency (% frames with interaction)
        - Optional: Z-ordering to ensure cation is between clay and ring
        
        Parameters
        ----------
        ring_selection : str
            Selection for aromatic ring atoms. Can be either:
            - A selection name defined in define_selections() (e.g., 'quinolone')
            - Full MDAnalysis selection string (e.g., 'resname CIP and (name C4 C5...)')
            IMPORTANT: Centroid is computed as mean of ALL selected atoms. For quinolone
            (2 fused rings), you may want only the 6-membered aromatic ring atoms.
            Example: 'resname api and (name C4 C5 C10 C11 C12 C19)' for aromatic ring only.
        cation_selection : str
            Selection for cations. Can be either:
            - A selection name defined in define_selections() (e.g., 'Na')
            - Full MDAnalysis selection string (e.g., 'resname NA')
        clay_sel : str, optional
            Selection for clay surface atoms (e.g., 'name Ob'). Required if require_z_ordering=True
            or if max_clay_distance is set. Used to check if cation is positioned between clay
            and aromatic ring, and/or within distance of clay surface.
        cluster_ids : list of int, optional
            Clusters to analyze
        require_z_ordering : bool, default=False
            If True, only count π-cation interactions when the cation is positioned
            between the clay surface and the aromatic ring in the Z-direction.
            This ensures the cation mediates clay-molecule interaction.
            Requires clay_sel to be provided.
        distance_cutoff : float
            Maximum distance from cation to ring centroid (Angstroms, default=5.0).
            Uses PBC-aware distance calculation.
        max_clay_distance : float, optional
            If provided, also require that cation is within this distance from clay surface.
            This ensures π-cation interactions occur near the clay interface.
            Requires clay_sel to be provided. Uses PBC-aware distance.
        angle_cutoff : float
            Maximum angle deviation from perpendicular (degrees, default=30)
            Measures deviation from perpendicular: 0° = cation directly above/below ring,
            90° = cation in ring plane. Typical values: 20-30° for strict π-cation,
            40-45° for looser geometries.
        max_ring_clay_angle : float, optional
            Maximum angle between ring normal and clay Z-axis (degrees).
            For π-cation-clay interactions, ring should be parallel to clay surface.
            0° = ring perfectly parallel to clay, 90° = ring perpendicular to clay.
            Typical values: 30° for strict parallel orientation, 45° for looser.
            If None, no ring-clay orientation check is performed.
        detect_rings : bool, default=True
            If True, automatically detect separate rings using RDKit (e.g., both rings
            in quinolone bicyclic structure). Computes π-cation interaction with the
            closest ring. If False or RDKit unavailable, uses centroid of all atoms.
        extract_trajectory_frames : bool, default=False
            If True, extract frames with π-cation contacts into separate .xtc files per cluster
        topology_file : str, optional
            Path to topology file (e.g., 'nvt.tpr'). Required if extract_trajectory_frames=True
        trajectory_file : str, optional
            Path to trajectory file (e.g., 'nvt.xtc'). Required if extract_trajectory_frames=True
        output_dir : str, optional
            Directory to save extracted trajectories. Default: 'pi_cation_trajectories'
            
        Returns
        -------
        pi_cation_results : dict
            {cluster_id: {
                'distances_to_center': array,  # Distance to ring centroid per frame
                'distances_to_atoms': array,    # Min distance to any ring atom
                'angles': array,                # Angle from perpendicular (degrees)
                'contact_frames': array,        # Boolean - has contact per frame
                'contact_frequency': float,     # % frames with contact
                'mean_distance': float,         # Mean distance when in contact
                'preferred_distance': float,    # Most common distance (RDF peak)
                'extracted_trajectory': str,    # Path to extracted .xtc file (if extraction enabled)
            }}
        
        Example
        -------
        >>> # Smart ring detection for quinolone (2 fused 6-membered rings)
        >>> # Automatically detects both rings and finds which cation prefers
        >>> pi_cat = analyzer.compute_pi_cation_interactions(
        ...     ring_selection='resname api and (name N6 C10 C11 C12 C19 C21 C22 C23 C4 C5)',
        ...     cation_selection='Na',
        ...     clay_sel='name Ob',
        ...     require_z_ordering=True,
        ...     detect_rings=True,  # Automatic ring detection (default)
        ...     distance_cutoff=5.0,
        ...     max_clay_distance=6.0,  # Cation must be within 6Å of clay
        ...     angle_cutoff=30.0
        ... )
        
        >>> # Basic usage without ring detection
        >>> pi_cat = analyzer.compute_pi_cation_interactions(
        ...     ring_selection='quinolone',
        ...     cation_selection='Na',
        ...     detect_rings=False,  # Use centroid of all atoms
        ...     cluster_ids=[0, 1],
        ...     distance_cutoff=5.0,
        ...     angle_cutoff=30.0
        ... )
        
        >>> # With Z-ordering to ensure Na+ is between clay and ring
        >>> pi_cat = analyzer.compute_pi_cation_interactions(
        ...     ring_selection='quinolone',
        ...     cation_selection='Na',
        ...     clay_sel='name Ob',
        ...     require_z_ordering=True,
        ...     cluster_ids=[0, 1],
        ...     distance_cutoff=5.0
        ... )
        
        >>> # Or using full MDAnalysis selection strings
        >>> pi_cat = analyzer.compute_pi_cation_interactions(
        ...     'resname api and (name N6 C10 C11 C12 C19 C21 C22 C23 C4 C5)',
        ...     'resname NA',
        ...     distance_cutoff=5.0
        ... )
        
        >>> # With trajectory extraction and all features
        >>> pi_cat = analyzer.compute_pi_cation_interactions(
        ...     ring_selection='resname api and (name N6 C10 C11 C12 C19 C21 C22 C23 C4 C5)',
        ...     cation_selection='Na',
        ...     clay_sel='name Ob',
        ...     require_z_ordering=True,        # Ion between clay and ring
        ...     detect_rings=True,              # Detect separate rings in quinolone
        ...     distance_cutoff=5.0,            # Max cation-ring distance
        ...     max_clay_distance=6.0,          # Max cation-clay distance
        ...     angle_cutoff=30.0,              # Angle from perpendicular
        ...     extract_trajectory_frames=True,
        ...     topology_file='nvt.tpr',
        ...     trajectory_file='nvt.xtc',
        ...     output_dir='pi_cation_trajectories'
        ... )
        """
        if not hasattr(self, 'trajectory_data'):
            raise ValueError("Load trajectories first")
        
        # Validate parameters
        if require_z_ordering and clay_sel is None:
            raise ValueError("clay_sel must be provided when require_z_ordering=True")
        if max_clay_distance is not None and clay_sel is None:
            raise ValueError("clay_sel must be provided when max_clay_distance is set")
        
        # Resolve selection names to selection strings if they exist in defined selections
        if ring_selection in self._selections:
            ring_selection_resolved = self._selections[ring_selection]
            print(f"  Resolved ring selection '{ring_selection}' to: {ring_selection_resolved}")
            ring_selection = ring_selection_resolved
        
        if cation_selection in self._selections:
            cation_selection_resolved = self._selections[cation_selection]
            print(f"  Resolved cation selection '{cation_selection}' to: {cation_selection_resolved}")
            cation_selection = cation_selection_resolved
        
        # Handle cluster_ids='all' or None
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = sorted(self.trajectory_data.keys())
        
        print(f"\nComputing π-cation interactions:")
        print(f"  Ring: {ring_selection}")
        print(f"  Cations: {cation_selection}")
        if clay_sel:
            print(f"  Clay: {clay_sel}")
        if require_z_ordering:
            print(f"  Z-ordering: Enabled (cation must be between clay and ring)")
        else:
            print(f"  Z-ordering: Disabled")
        print(f"  Distance cutoff (cation↔ring): {distance_cutoff:.1f} Å (PBC-aware)")
        if max_clay_distance is not None:
            print(f"  Max clay distance (cation↔clay): {max_clay_distance:.1f} Å (PBC-aware)")
        if max_ring_clay_angle is not None:
            print(f"  Ring-clay angle cutoff: ≤{max_ring_clay_angle:.1f}° (ring parallel to clay)")
        print(f"  Angle cutoff: ±{angle_cutoff:.1f}°")
        print(f"")
        print(f"  Angle interpretation:")
        print(f"    0° = cation directly above/below ring (perfect π-cation)")
        print(f"    {angle_cutoff:.1f}° = current cutoff (cation within {angle_cutoff:.1f}° of perpendicular)")
        print(f"    45° = intermediate angle")
        print(f"    90° = cation in ring plane (no π-interaction)")
        print(f"")
        if angle_cutoff < 15:
            print(f"  NOTE: {angle_cutoff:.1f}° is quite restrictive. Consider trying 30° if no contacts found.")
        
        pi_cation_results = {}
        
        for cluster_id in cluster_ids:
            print(f"\n  Cluster {cluster_id}:")
            u = self.trajectory_data[cluster_id]['universe']
            frames = self.trajectory_data[cluster_id]['frames']
            
            ring = u.select_atoms(ring_selection)
            cations = u.select_atoms(cation_selection)
            
            # Select clay atoms if needed
            clay_atoms = None
            if clay_sel:
                clay_atoms = u.select_atoms(clay_sel)
                print(f"    Clay atoms: {len(clay_atoms)}")
            
            print(f"    Ring atoms: {len(ring)}")
            print(f"    Cations: {len(cations)}")
            
            # Detect separate rings if requested
            separate_rings = None
            if detect_rings:
                print("    Detecting separate rings...")
                separate_rings = self._detect_separate_rings(ring)
                print(f"    Detected {len(separate_rings)} ring(s):")
                for i, ring_indices in enumerate(separate_rings, 1):
                    ring_atoms = [ring[idx].name for idx in ring_indices]
                    print(f"      Ring {i}: {len(ring_indices)} atoms - {', '.join(ring_atoms)}")
            
            # Get box dimensions for PBC
            box_dimensions = u.dimensions
            
            distances_to_center = []
            distances_to_atoms = []
            angles = []
            contact_frames = []
            ring_centroids_list = []
            closest_ring_indices = []
            # Track additional criteria for diagnostics
            ring_clay_angles = []
            clay_distances = []
            
            for frame_idx in frames:
                u.trajectory[frame_idx]
                
                # Update box dimensions for this frame
                box = u.dimensions
                
                # Get ring positions
                ring_pos = ring.positions
                
                # Compute centroid and normal for each ring (or single centroid if no detection)
                if detect_rings and separate_rings and len(separate_rings) > 1:
                    # Multiple rings detected - compute for each
                    ring_centroids_list_frame = []
                    ring_normals_list_frame = []
                    
                    for ring_indices in separate_rings:
                        ring_subset_pos = ring_pos[ring_indices]
                        ring_centroid = ring_subset_pos.mean(axis=0)
                        ring_centroids_list_frame.append(ring_centroid)
                        
                        # Compute normal for this ring
                        if len(ring_indices) >= 3:
                            v1 = ring_subset_pos[1] - ring_subset_pos[0]
                            v2 = ring_subset_pos[2] - ring_subset_pos[0]
                            ring_normal = np.cross(v1, v2)
                            ring_normal = ring_normal / np.linalg.norm(ring_normal)
                        else:
                            ring_normal = np.array([0, 0, 1])
                        ring_normals_list_frame.append(ring_normal)
                    
                    # Convert lists to numpy arrays using vstack for explicit 2D shape
                    ring_centroids = np.vstack(ring_centroids_list_frame)  # Shape: (n_rings, 3)
                    ring_normals = np.vstack(ring_normals_list_frame)      # Shape: (n_rings, 3)
                else:
                    # Single ring or detection disabled - use all atoms
                    centroid = ring_pos.mean(axis=0)
                    ring_centroids = centroid.reshape(1, 3)  # Shape: (1, 3)
                    
                    # Compute normal
                    if len(ring) >= 3:
                        v1 = ring_pos[1] - ring_pos[0]
                        v2 = ring_pos[2] - ring_pos[0]
                        normal = np.cross(v1, v2)
                        normal = normal / np.linalg.norm(normal)
                    else:
                        normal = np.array([0, 0, 1])
                    ring_normals = normal.reshape(1, 3)  # Shape: (1, 3)
                
                # Cation positions
                cat_pos = cations.positions
                
                # PBC-aware distance to ALL ring centroids (find closest ring)
                from MDAnalysis.lib.distances import distance_array
                dists_to_all_centroids = distance_array(cat_pos, ring_centroids, box=box)  # [n_cations, n_rings]
                
                # For each cation, find closest ring
                closest_ring_per_cation = np.argmin(dists_to_all_centroids, axis=1)
                min_dists_per_cation = np.min(dists_to_all_centroids, axis=1)
                
                # Find cation closest to any ring
                closest_cation_idx = np.argmin(min_dists_per_cation)
                closest_ring_idx = closest_ring_per_cation[closest_cation_idx]
                min_dist_center = min_dists_per_cation[closest_cation_idx]
                
                # Use the closest ring's centroid and normal
                centroid = ring_centroids[closest_ring_idx]
                normal = ring_normals[closest_ring_idx]
                closest_cat_pos = cat_pos[closest_cation_idx]
                
                # Store ring info
                ring_centroids_list.append(ring_centroids.copy())
                closest_ring_indices.append(closest_ring_idx)
                
                # PBC-aware distance to nearest ring atom
                dists_to_atoms = distance_array(cat_pos, ring_pos, box=box)
                min_dist_atoms = dists_to_atoms.min(axis=1).min()
                
                # PBC-aware vector from centroid to cation
                from MDAnalysis.lib.distances import calc_bonds
                vec_distance = calc_bonds(centroid.reshape(1, 3), closest_cat_pos.reshape(1, 3), box=box)[0]
                if vec_distance > 0.01:  # Avoid division by zero
                    # Get actual vector (accounting for PBC)
                    delta = closest_cat_pos - centroid
                    # Minimum image convention
                    if box is not None:
                        delta = delta - box[:3] * np.round(delta / box[:3])
                    vec_to_cation = delta / np.linalg.norm(delta)
                else:
                    vec_to_cation = np.array([0, 0, 1])
                
                # Angle from perpendicular (0° = perpendicular to ring, 90° = in ring plane)
                dot_product = np.abs(np.dot(vec_to_cation, normal))
                angle_from_perp = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
                

                
                # Contact criteria: distance < cutoff AND angle < threshold
                has_contact = (min_dist_center < distance_cutoff) and (angle_from_perp < angle_cutoff)
                
                # Optional: Ring-clay orientation check (ring must be parallel to clay surface)
                ring_clay_angle_deg = None
                if max_ring_clay_angle is not None and has_contact:
                    # Clay surface normal is Z-axis [0, 0, 1]
                    clay_normal = np.array([0, 0, 1])
                    
                    # Angle between ring normal and clay normal
                    # 0° = ring parallel to clay, 90° = ring perpendicular to clay
                    ring_clay_dot = np.abs(np.dot(normal, clay_normal))
                    ring_clay_angle_deg = np.degrees(np.arccos(np.clip(ring_clay_dot, -1, 1)))
                    
                    if ring_clay_angle_deg > max_ring_clay_angle:
                        has_contact = False  # Reject if ring not parallel enough to clay
                
                # Optional: Check cation-clay distance
                clay_dist = None
                if max_clay_distance is not None and has_contact and clay_atoms is not None:
                    clay_pos = clay_atoms.positions
                    # PBC-aware distance to nearest clay atom
                    dists_to_clay = distance_array(closest_cat_pos.reshape(1, 3), clay_pos, box=box)[0]
                    clay_dist = dists_to_clay.min()
                    
                    if clay_dist > max_clay_distance:
                        has_contact = False  # Reject if cation too far from clay
                
                # Optional: Z-ordering check (cation must be between clay and ring)
                if require_z_ordering and has_contact and clay_atoms is not None:
                    clay_pos = clay_atoms.positions
                    
                    # Use nearest clay atom for Z-position
                    if clay_dist is not None:
                        # Reuse already calculated distances
                        dists_to_clay_3d = distance_array(closest_cat_pos.reshape(1, 3), clay_pos, box=box)[0]
                    else:
                        dists_to_clay_3d = distance_array(closest_cat_pos.reshape(1, 3), clay_pos, box=box)[0]
                        clay_dist = dists_to_clay_3d.min()
                    
                    nearest_clay_idx = np.argmin(dists_to_clay_3d)
                    z_clay = clay_pos[nearest_clay_idx, 2]
                    
                    z_ion = closest_cat_pos[2]
                    z_ring = centroid[2]
                    
                    # Ion must be between clay and ring in Z
                    is_between = (z_clay < z_ion < z_ring) or (z_ring < z_ion < z_clay)
                    
                    if not is_between:
                        has_contact = False  # Reject contact if not between surfaces
                
                distances_to_center.append(min_dist_center)
                distances_to_atoms.append(min_dist_atoms)
                angles.append(angle_from_perp)
                contact_frames.append(has_contact)
                
                # Track additional criteria for diagnostics
                ring_clay_angles.append(ring_clay_angle_deg if ring_clay_angle_deg is not None else np.nan)
                clay_distances.append(clay_dist if clay_dist is not None else np.nan)
            
            distances_to_center = np.array(distances_to_center)
            distances_to_atoms = np.array(distances_to_atoms)
            angles = np.array(angles)
            contact_frames = np.array(contact_frames)
            ring_clay_angles = np.array(ring_clay_angles)
            clay_distances = np.array(clay_distances)
            
            # Contact statistics
            contact_freq = 100 * np.sum(contact_frames) / len(contact_frames)
            
            # Diagnostic: Show how many frames passed each criterion
            print(f"\n    Diagnostic breakdown:")
            print(f"      Total frames analyzed: {len(contact_frames)}")
            distance_passed = np.sum(distances_to_center < distance_cutoff)
            angle_passed = np.sum(angles < angle_cutoff)
            both_passed = np.sum((distances_to_center < distance_cutoff) & (angles < angle_cutoff))
            
            print(f"      Frames with distance < {distance_cutoff:.1f} Å: {distance_passed} ({100*distance_passed/len(contact_frames):.1f}%)")
            print(f"      Frames with angle < {angle_cutoff:.1f}°: {angle_passed} ({100*angle_passed/len(contact_frames):.1f}%)")
            print(f"      Frames passing distance AND angle: {both_passed} ({100*both_passed/len(contact_frames):.1f}%)")
            
            # Additional criteria diagnostics
            if max_ring_clay_angle is not None:
                valid_ring_angles = ~np.isnan(ring_clay_angles)
                if np.any(valid_ring_angles):
                    ring_angle_passed = np.sum(ring_clay_angles[valid_ring_angles] <= max_ring_clay_angle)
                    print(f"      Frames with ring-clay angle ≤ {max_ring_clay_angle:.1f}°: {ring_angle_passed}/{np.sum(valid_ring_angles)} ({100*ring_angle_passed/np.sum(valid_ring_angles):.1f}%)")
                    print(f"      Average ring-clay angle: {np.nanmean(ring_clay_angles):.1f}°")
            
            if max_clay_distance is not None:
                valid_clay_dists = ~np.isnan(clay_distances)
                if np.any(valid_clay_dists):
                    clay_dist_passed = np.sum(clay_distances[valid_clay_dists] <= max_clay_distance)
                    print(f"      Frames with cation-clay distance ≤ {max_clay_distance:.1f} Å: {clay_dist_passed}/{np.sum(valid_clay_dists)} ({100*clay_dist_passed/np.sum(valid_clay_dists):.1f}%)")
                    print(f"      Average cation-clay distance: {np.nanmean(clay_distances):.2f} Å")
            
            print(f"      Final contacts (after all checks): {np.sum(contact_frames)} ({contact_freq:.1f}%)")
            print(f"      Average cation-ring distance: {np.mean(distances_to_center):.2f} Å")
            print(f"      Average angle deviation: {np.mean(angles):.1f}°")
            
            # Angle distribution analysis
            print(f"\n    Angle Distribution Analysis:")
            angle_percentiles = np.percentile(angles, [10, 25, 50, 75, 90])
            print(f"      Angle percentiles: 10%={angle_percentiles[0]:.1f}°, 25%={angle_percentiles[1]:.1f}°, " +
                  f"50%={angle_percentiles[2]:.1f}°, 75%={angle_percentiles[3]:.1f}°, 90%={angle_percentiles[4]:.1f}°")
            
            # Show what percentage would pass with different cutoffs
            for test_cutoff in [10, 15, 20, 30, 45]:
                test_pass = np.sum(angles < test_cutoff)
                print(f"      Would pass with {test_cutoff:2d}° cutoff: {test_pass:4d} frames ({100*test_pass/len(angles):5.1f}%)")
            
            if np.sum(contact_frames) == 0:
                print(f"\n    NO CONTACTS FOUND! Consider:")
                print(f"       • Increasing angle_cutoff (try 30° instead of {angle_cutoff:.1f}°)")
                print(f"       • Checking if ring selection contains correct atoms")
                print(f"       • Verifying ring normal calculation is correct")
            
            # Mean distance when in contact
            if np.any(contact_frames):
                mean_contact_dist = np.mean(distances_to_center[contact_frames])
                
                # Preferred distance (RDF peak) - histogram mode
                hist, bins = np.histogram(distances_to_center[contact_frames], bins=50)
                peak_idx = np.argmax(hist)
                preferred_dist = (bins[peak_idx] + bins[peak_idx + 1]) / 2
            else:
                mean_contact_dist = np.nan
                preferred_dist = np.nan
            
            pi_cation_results[cluster_id] = {
                'distances_to_center': distances_to_center,
                'distances_to_atoms': distances_to_atoms,
                'angles': angles,
                'contact_frames': contact_frames,
                'contact_frequency': contact_freq,
                'mean_distance': np.mean(distances_to_center),
                'mean_contact_distance': mean_contact_dist,
                'preferred_distance': preferred_dist,
                'mean_angle': np.mean(angles),
                'ring_selection': ring_selection,
                'cation_selection': cation_selection,
                'distance_cutoff': distance_cutoff,
                'max_clay_distance': max_clay_distance,
                'angle_cutoff': angle_cutoff,
                'clay_selection': clay_sel,
                'require_z_ordering': require_z_ordering,
                'detect_rings': detect_rings,
                'ring_centroids_per_frame': ring_centroids_list if detect_rings else None,
                'closest_ring_per_frame': closest_ring_indices if detect_rings else None
            }
            
            print(f"    Contact frequency: {contact_freq:.1f}%")
            print(f"      Mean distance: {np.mean(distances_to_center):.2f} Å")
            if not np.isnan(preferred_dist):
                print(f"      Preferred distance: {preferred_dist:.2f} Å")
            print(f"      Mean angle deviation: {np.mean(angles):.1f}°")
            
            # If multiple rings detected, show preference
            if detect_rings and separate_rings and len(separate_rings) > 1:
                closest_ring_array = np.array(closest_ring_indices)
                print(f"      Ring preference:")
                for ring_idx in range(len(separate_rings)):
                    count = np.sum(closest_ring_array == ring_idx)
                    percentage = 100 * count / len(closest_ring_array)
                    print(f"        Ring {ring_idx + 1}: {percentage:.1f}% of frames")
                
                # Show preference during contacts only
                if len(contact_frames) > 0 and np.any(contact_frames):
                    contact_ring_indices = closest_ring_array[contact_frames]
                    print(f"      Ring preference (during π-cation contacts):")
                    for ring_idx in range(len(separate_rings)):
                        count = np.sum(contact_ring_indices == ring_idx)
                        percentage = 100 * count / len(contact_ring_indices) if len(contact_ring_indices) > 0 else 0
                        print(f"        Ring {ring_idx + 1}: {percentage:.1f}% of contact frames")
        
        # Store
        if not hasattr(self, 'pi_cation_data'):
            self.pi_cation_data = {}
        key = f"{ring_selection}__{cation_selection}"
        self.pi_cation_data[key] = pi_cation_results
        
        # ═══════════════════════════════════════════════════════════
        # Optional: Extract trajectory frames with π-cation contacts
        # ═══════════════════════════════════════════════════════════
        if extract_trajectory_frames:
            if not topology_file or not trajectory_file:
                print("\n⚠ WARNING: Trajectory extraction skipped!")
                print("  Missing required files: topology_file and/or trajectory_file")
                print("  Example: topology_file='nvt.tpr', trajectory_file='nvt.xtc'")
            else:
                # Set default output directory
                if output_dir is None:
                    output_dir = 'pi_cation_trajectories'
                
                # Create output directory
                import os
                os.makedirs(output_dir, exist_ok=True)
                
                print(f"\n{'='*80}")
                print(f"EXTRACTING π-CATION CONTACT TRAJECTORIES")
                print(f"{'='*80}")
                print(f"\n  Topology: {topology_file}")
                print(f"  Trajectory: {trajectory_file}")
                print(f"  Output directory: {output_dir}")
                
                # Load trajectory
                print(f"\n  Loading trajectory...")
                try:
                    import MDAnalysis as mda
                    u_extract = mda.Universe(topology_file, trajectory_file)
                    print(f"  Loaded trajectory with {len(u_extract.trajectory)} frames")
                    print(f"  System contains {len(u_extract.atoms)} atoms")
                    
                    extracted_files = []
                    
                    # Extract frames for each cluster
                    print(f"\n  Extracting frames by cluster...")
                    for cluster_id, data in pi_cation_results.items():
                        # Get frames with π-cation contacts
                        contact_frames_bool = data['contact_frames']
                        contact_indices = np.where(contact_frames_bool)[0]
                        
                        if len(contact_indices) == 0:
                            print(f"\n    Cluster {cluster_id}: No π-cation contacts - skipping")
                            data['extracted_trajectory'] = None
                            continue
                        
                        # Map to actual frame numbers in original trajectory
                        cluster_frames = self.trajectory_data[cluster_id]['frames']
                        actual_frames = cluster_frames[contact_indices]
                        n_frames = len(actual_frames)
                        
                        output_file = os.path.join(output_dir, f"pi_cation_cluster_{int(cluster_id)}.xtc")
                        
                        print(f"\n    Cluster {int(cluster_id)}: {n_frames} contact frames")
                        print(f"      Contact frequency: {data['contact_frequency']:.1f}%")
                        print(f"      Frame range: {actual_frames.min()}-{actual_frames.max()}")
                        print(f"      Output: {output_file}")
                        
                        # Write frames to new trajectory
                        with mda.Writer(output_file, n_atoms=len(u_extract.atoms)) as writer:
                            for frame_idx, frame_num in enumerate(actual_frames, 1):
                                # Go to this frame
                                u_extract.trajectory[int(frame_num)]
                                
                                # Write this frame
                                writer.write(u_extract.atoms)
                                
                                # Progress indicator
                                if frame_idx % 10 == 0 or frame_idx == n_frames:
                                    print(f"      Progress: {frame_idx}/{n_frames} frames", end="\r")
                        
                        print(f"\n      Saved to {output_file}")
                        extracted_files.append(output_file)
                        data['extracted_trajectory'] = output_file
                    
                    # Summary
                    print(f"\n{'='*80}")
                    print(f"TRAJECTORY EXTRACTION COMPLETE")
                    print(f"{'='*80}")
                    if extracted_files:
                        print(f"\n  Generated {len(extracted_files)} trajectory file(s):")
                        for filepath in extracted_files:
                            if os.path.exists(filepath):
                                file_size_mb = os.path.getsize(filepath) / (1024**2)
                                cluster_id = int(filepath.split('cluster_')[1].split('.')[0])
                                n_frames = np.sum(pi_cation_results[cluster_id]['contact_frames'])
                                print(f"    {os.path.basename(filepath)}: {n_frames} frames, {file_size_mb:.2f} MB")
                        print(f"\n  Use these files to visualize π-cation interactions in VMD/PyMOL/OVITO")
                    else:
                        print(f"\n  No trajectories extracted (no π-cation contacts found)")
                    print(f"{'='*80}")
                    
                except Exception as e:
                    print(f"\nERROR during trajectory extraction: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\nπ-cation interactions computed")
        return pi_cation_results
    
    def compute_orientation_angles(self, group_selection,
                                  reference_vector: Tuple[float, float, float],
                                  cluster_ids: Optional[List[int]] = None,
                                  center_z: bool = True):
        """
        Compute orientation angles of molecular groups relative to reference.
        
        Calculates the angle between the molecular principal axis (from PCA)
        and the reference vector. Returns angles in the range [0-180°] where:
        - 0° = principal axis parallel to reference (e.g., pointing up)
        - 90° = perpendicular to reference (e.g., lying flat)
        - 180° = antiparallel to reference (e.g., pointing down)
        
        Parameters
        ----------
        group_selection : str, list, or dict
            Selection defining molecular group(s). Can be:
            - Single string: 'piperazine' or 'resname api and (name N13...)'
            - List of names: ['quinolone', 'carboxylic_acid', 'piperazine']
            - Dict: {'quinolone': 'resname api and...', 'piperazine': '...'}
        reference_vector : tuple of float
            Reference vector (e.g., (0,0,1) for surface normal)
        cluster_ids : list of int, optional
            Clusters to analyze
        center_z : bool
            If True (default), recenters Z-positions at box center and uses
            internal molecular reference (topmost atom) to resolve PCA direction
            ambiguity. This ensures consistent orientation assignment while
            allowing full 0-180° angle range at any Z-position.
            If False, uses original Z-coordinates and raw PCA axis.
            
        Returns
        -------
        angle_results : dict
            If single selection:
                {cluster_id: {'angles': array, 'mean_angle': float, ...}}
            If multiple selections:
                {selection_name: {cluster_id: {'angles': array, ...}}}
        
        Example
        -------
        >>> # Single selection
        >>> angles = analyzer.compute_orientation_angles(
        ...     group_selection='piperazine',
        ...     reference_vector=(0, 0, 1),
        ...     cluster_ids=[0, 1]
        ... )
        
        >>> # Multiple selections at once (recommended)
        >>> angles = analyzer.compute_orientation_angles(
        ...     group_selection=['quinolone', 'carboxylic_acid', 'piperazine'],
        ...     reference_vector=(0, 0, 1),
        ...     cluster_ids=[0, 1]
        ... )
        """
        if not hasattr(self, 'trajectory_data'):
            raise ValueError("Load trajectories first")
        
        # Handle cluster_ids='all' or None
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = sorted(self.trajectory_data.keys())
        
        # Handle list or dict of selections
        if isinstance(group_selection, (list, tuple)):
            # Convert list to dict using selection names
            group_dict = {}
            for sel_name in group_selection:
                if sel_name in self._selections:
                    group_dict[sel_name] = self._selections[sel_name]
                else:
                    group_dict[sel_name] = sel_name  # Use as-is if not in defined selections
            all_results = {}
            for name, sel in group_dict.items():
                print(f"\n{'='*60}")
                print(f"Processing: {name}")
                result = self._compute_single_orientation_angles(
                    sel, reference_vector, cluster_ids, name, center_z
                )
                all_results[name] = result
            return all_results
        
        elif isinstance(group_selection, dict):
            # Dict of selections
            all_results = {}
            for name, sel in group_selection.items():
                print(f"\n{'='*60}")
                print(f"Processing: {name}")
                result = self._compute_single_orientation_angles(
                    sel, reference_vector, cluster_ids, name, center_z
                )
                all_results[name] = result
            return all_results
        
        else:
            # Single selection (original behavior)
            # Resolve selection name if it exists
            if group_selection in self._selections:
                group_selection_resolved = self._selections[group_selection]
                print(f"  Resolved group selection '{group_selection}' to: {group_selection_resolved}")
                sel_name = group_selection
                group_selection = group_selection_resolved
            else:
                sel_name = None
            
            return self._compute_single_orientation_angles(
                group_selection, reference_vector, cluster_ids, sel_name, center_z
            )
    
    def _compute_single_orientation_angles(self, group_selection: str,
                                          reference_vector: Tuple[float, float, float],
                                          cluster_ids: List[int],
                                          name: str = None,
                                          center_z: bool = True):
        """Internal method to compute orientation angles for a single selection."""
        
        ref_vec = np.array(reference_vector, dtype=float)
        ref_vec = ref_vec / np.linalg.norm(ref_vec)
        
        print(f"\nComputing orientation angles:")
        print(f"  Group: {group_selection}")
        print(f"  Reference: {reference_vector}")
        print(f"  Angle range: 0-180°")
        print(f"  Z-centering: {center_z}")
        print(f"  Direction reference: {'topmost atom in group' if center_z else 'raw PCA axis'}")
        
        angle_results = {}
        
        for cluster_id in cluster_ids:
            print(f"\n  Cluster {cluster_id}:")
            u = self.trajectory_data[cluster_id]['universe']
            frames = self.trajectory_data[cluster_id]['frames']
            
            # Get box center Z for this cluster
            box_center_z = u.dimensions[2] / 2.0
            print(f"    Box Z-center: {box_center_z:.2f} Å")
            
            group = u.select_atoms(group_selection)
            print(f"    Group atoms: {len(group)}")
            
            angles = []
            z_positions = []  # Track Z-position of group centroid
            
            for frame_idx in frames:
                u.trajectory[frame_idx]
                # Ensure molecule is whole across periodic boundaries for accurate PCA
                group.unwrap()
                pos = group.positions
                
                # Store Z-position of group centroid
                centroid_z = pos[:, 2].mean()  # Z is 3rd dimension
                z_positions.append(centroid_z)
                
                # Compute group vector using PCA (principal axis)
                if len(pos) >= 2:
                    centered = pos - pos.mean(axis=0)
                    cov = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eig(cov)
                    # Principal axis is eigenvector with largest eigenvalue
                    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
                    
                    # Enforce consistent direction using internal molecular reference
                    # Make principal axis point toward the atom with highest Z-coordinate
                    # This removes PCA sign ambiguity while allowing full 0-180° range 
                    # at any position in the box
                    if center_z:
                        # Find the atom with maximum Z within this molecular group
                        max_z_idx = np.argmax(pos[:, 2])
                        direction_to_top_atom = pos[max_z_idx] - pos.mean(axis=0)
                        
                        # Flip principal axis if it points away from the top atom
                        if np.dot(principal_axis, direction_to_top_atom) < 0:
                            principal_axis = -principal_axis
                else:
                    principal_axis = np.array([0, 0, 1])
                
                # Angle between principal axis and reference (0-180°)
                cos_angle = np.dot(principal_axis, ref_vec)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                angles.append(angle)
            
            angles = np.array(angles)
            z_positions = np.array(z_positions)
            
            # Recenter Z if requested
            if center_z:
                z_positions_centered = z_positions - box_center_z
            else:
                z_positions_centered = z_positions
            
            # Get box dimensions
            box_z_length = u.dimensions[2]
            
            # Histogram
            hist, bins = np.histogram(angles, bins=36, range=(0, 180))
            
            angle_results[cluster_id] = {
                'angles': angles,
                'z_positions': z_positions,  # Original Z-coordinates
                'z_positions_centered': z_positions_centered,  # Z relative to box center
                'box_center_z': box_center_z,
                'box_z_length': box_z_length,
                'mean_angle': np.mean(angles),
                'std_angle': np.std(angles),
                'median_angle': np.median(angles),
                'hist': hist,
                'bin_edges': bins,
                'bin_centers': (bins[:-1] + bins[1:]) / 2,
                'group_selection': group_selection,
                'reference_vector': reference_vector
            }
            
            print(f"    ✓ Mean angle: {np.mean(angles):.1f} ± {np.std(angles):.1f}°")
            print(f"      Median angle: {np.median(angles):.1f}°")
        
        # Store
        if not hasattr(self, 'orientation_data'):
            self.orientation_data = {}
        
        # Use name if provided, otherwise create key from selection string
        if name:
            storage_key = name
        else:
            storage_key = f"{group_selection}__ref_{reference_vector}"
        
        self.orientation_data[storage_key] = angle_results
        
        print(f"\n✓ Orientation angles computed")
        return angle_results

    
    def compute_hydrogen_bonds(self, donors: str, acceptors: str,
                               cluster_ids: Optional[List[int]] = None,
                               distance_cutoff: Union[float, List[float]] = 2.5,
                               angle_cutoff: Union[float, List[float]] = 150.0,
                               update_selections: bool = True,
                               show_report: bool = False):
        """
        Analyze hydrogen bonds between donors and acceptors.
        
        Uses CORRECT H-bond geometry with hydrogen-acceptor distance:
        - Distance: H...A distance (bonded hydrogen to acceptor)
        - Angle: D-H...A angle (180° = linear)
        - PBC corrections: Applies minimum image convention for accurate distances
        
        IMPORTANT: 
        - distance_cutoff is applied to the HYDROGEN-ACCEPTOR distance,
          not the donor-acceptor distance. This is the correct H-bond definition.
        - All distance and angle calculations respect periodic boundary conditions
          to handle molecules crossing box boundaries correctly.
        
        Parameters
        ----------
        donors : str
            Selection for hydrogen bond donors (e.g., 'resname api and (name N or name O1)')
            These are the HEAVY ATOMS (N, O, F). The method will automatically
            find their bonded hydrogens.
        acceptors : str
            Selection for hydrogen bond acceptors (e.g., 'name Ob or name OW')
        cluster_ids : list of int, optional
            Clusters to analyze
        distance_cutoff : float or list of float
            Maximum HYDROGEN-ACCEPTOR distance for H-bond (Angstroms, default=2.5)
            Can be a single value or list for multi-cutoff analysis
            Typical values: 2.0 Å (strong), 2.5 Å (standard), 3.0 Å (moderate)
        angle_cutoff : float or list of float
            Minimum donor-H-acceptor angle (degrees, default=150°)
            Can be a single value or list for multi-cutoff analysis
            Typical values: 120° (bent), 150° (linear), 180° (perfect)
        update_selections : bool
            Update atom selections each frame (default=True)
        show_report : bool
            If True, display detailed table of results for all cutoff combinations
            If False, print compact summary (default=False)
            
        Returns
        -------
        hbond_results : dict
            For single cutoffs:
                {cluster_id: {...data...}}
            For multi-cutoff analysis:
                {(distance, angle): {cluster_id: {...data...}}}
            
            Data structure for each cluster:
                'timeseries': array - H-bond count per frame,
                'n_hbonds': int - unique H-bonds detected,
                'occupancy': dict - {(donor_idx, hydrogen_idx, acceptor_idx): occupancy_fraction},
                'lifetimes': dict - {(donor_idx, hydrogen_idx, acceptor_idx): [lifetimes]},
                'mean_lifetime': float - average H-bond lifetime (frames),
                'mean_count': float - average H-bonds per frame,
                'hbond_pairs': list - [(donor_resname, donor_name, donor_idx, 
                                        hydrogen_name, hydrogen_idx,
                                        acceptor_resname, acceptor_name, acceptor_idx)]
        
        Examples
        --------
        >>> # Single cutoff analysis
        >>> hbonds = analyzer.compute_hydrogen_bonds(
        ...     donors='resname api and name N',
        ...     acceptors='name Ob or name Op',
        ...     distance_cutoff=2.5,
        ...     angle_cutoff=150.0
        ... )
        
        >>> # Multi-cutoff comparative analysis with report
        >>> hbonds = analyzer.compute_hydrogen_bonds(
        ...     donors='resname api and (name N* or name O*)',
        ...     acceptors='name Ob or name Op',
        ...     distance_cutoff=[2.0, 2.5, 3.0],
        ...     angle_cutoff=[120.0, 150.0],
        ...     cluster_ids='all',
        ...     show_report=True
        ... )
        """
        if not hasattr(self, 'trajectory_data'):
            raise ValueError("Load trajectories first")
        
        # Handle cluster_ids='all' or None
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = sorted(self.trajectory_data.keys())
        
        from MDAnalysis.lib.distances import distance_array
        
        # Convert single values to lists for uniform handling
        if isinstance(distance_cutoff, (int, float)):
            distance_cutoffs = [float(distance_cutoff)]
        else:
            distance_cutoffs = list(distance_cutoff)
        
        if isinstance(angle_cutoff, (int, float)):
            angle_cutoffs = [float(angle_cutoff)]
        else:
            angle_cutoffs = list(angle_cutoff)
        
        # Determine if multi-cutoff analysis
        is_multi_cutoff = len(distance_cutoffs) > 1 or len(angle_cutoffs) > 1
        
        print(f"\nComputing hydrogen bonds (HYDROGEN-ACCEPTOR distance method):")
        print(f"  Donors: {donors}")
        print(f"  Acceptors: {acceptors}")
        if is_multi_cutoff:
            print(f"  Distance cutoffs: {distance_cutoffs} Å")
            print(f"  Angle cutoffs: {angle_cutoffs}°")
            print(f"  Total combinations: {len(distance_cutoffs) * len(angle_cutoffs)}")
        else:
            print(f"  H...A distance cutoff: {distance_cutoffs[0]:.1f} Å")
            print(f"  D-H-A angle cutoff: {angle_cutoffs[0]:.1f}°")
        print(f"  PBC corrections: Enabled (minimum image convention)")
        
        # Store results: {(dist, angle): {cluster_id: {...}}}
        all_results = {}
        
        # Loop over all cutoff combinations
        for dist_cut in distance_cutoffs:
            for ang_cut in angle_cutoffs:
                cutoff_key = (dist_cut, ang_cut)
                
                if is_multi_cutoff:
                    print(f"\n{'='*60}")
                    print(f"Analyzing: distance={dist_cut:.1f} Å, angle={ang_cut:.1f}°")
                    print(f"{'='*60}")
                
                hbond_results = {}
        
                for cluster_id in cluster_ids:
                    if is_multi_cutoff:
                        print(f"\n  Cluster {cluster_id}:")
                    else:
                        print(f"\n  Cluster {cluster_id}:")
                    u = self.trajectory_data[cluster_id]['universe']
                    frames = self.trajectory_data[cluster_id]['frames']
                    
                    # Get donor heavy atoms and acceptors
                    donor_atoms = u.select_atoms(donors)
                    acceptor_atoms = u.select_atoms(acceptors)
                    
                    if not is_multi_cutoff:
                        print(f"    Donor heavy atoms: {len(donor_atoms)}")
                        print(f"    Acceptor atoms: {len(acceptor_atoms)}")
                    
                    # Find bonded hydrogens for each donor
                    donor_hydrogen_pairs = []
                    for donor in donor_atoms:
                        try:
                            bonded = donor.bonded_atoms
                            hydrogens = [h for h in bonded if h.element == 'H' or 'H' in h.name]
                            for h in hydrogens:
                                donor_hydrogen_pairs.append((donor, h))
                        except:
                            pass  # Skip if no bonds or no hydrogens
                    
                    if not is_multi_cutoff:
                        print(f"    Donor-H pairs found: {len(donor_hydrogen_pairs)}")
                    
                    if len(donor_hydrogen_pairs) == 0:
                        if not is_multi_cutoff:
                            print(f"    ⚠️  No bonded hydrogens found! Check topology.")
                        hbond_results[cluster_id] = {
                            'timeseries': np.zeros(len(frames)),
                            'n_hbonds': 0,
                            'occupancy': {},
                            'lifetimes': {},
                            'mean_lifetime': 0.0,
                            'mean_count': 0.0,
                            'hbond_pairs': [],
                            'donors': donors,
                            'acceptors': acceptors,
                            'distance_cutoff': dist_cut,
                            'angle_cutoff': ang_cut
                        }
                        continue
                    
                    # Analyze each frame
                    if not is_multi_cutoff:
                        print(f"    Analyzing {len(frames)} frames...")
                    
                    hbonds_per_frame = []  # List of sets of (donor_idx, h_idx, acc_idx) per frame
                    
                    for frame_idx in frames:
                        u.trajectory[frame_idx]
                        
                        # Get box dimensions for PBC-aware distance calculations
                        box = u.dimensions  # [a, b, c, alpha, beta, gamma]
                        
                        frame_hbonds = set()
                        
                        # Check each donor-hydrogen pair
                        for donor, h in donor_hydrogen_pairs:
                            # Calculate H...A distances with PBC correction
                            h_pos = h.position.reshape(1, 3)
                            acc_pos = acceptor_atoms.positions
                            
                            # Apply PBC correction for accurate distances
                            dists = distance_array(h_pos, acc_pos, box=box)[0]
                            
                            # Find acceptors within distance cutoff (using current dist_cut)
                            close_mask = dists < dist_cut
                            
                            if not np.any(close_mask):
                                continue
                            
                            close_acceptors = acceptor_atoms[close_mask]
                            
                            # Check angle for each close acceptor
                            for acc in close_acceptors:
                                # Calculate D-H-A angle with correct vector directions
                                # For proper H-bond angle at H: use H→D and H→A vectors
                                # Linear H-bonds (180°) have these vectors pointing opposite directions
                                vec_hd = donor.position - h.position  # H → D (pointing back to donor)
                                vec_ha = acc.position - h.position    # H → A (pointing to acceptor)
                                
                                # Apply minimum image convention for vectors
                                if box is not None:
                                    # For orthorhombic boxes (most MD simulations)
                                    box_lengths = box[:3]  # a, b, c
                                    
                                    # Wrap vec_hd
                                    vec_hd = vec_hd - box_lengths * np.round(vec_hd / box_lengths)
                                    
                                    # Wrap vec_ha
                                    vec_ha = vec_ha - box_lengths * np.round(vec_ha / box_lengths)
                                
                                # Calculate D-H-A angle with PBC-corrected vectors
                                norm_hd = np.linalg.norm(vec_hd)
                                norm_ha = np.linalg.norm(vec_ha)
                                
                                if norm_hd > 0 and norm_ha > 0:
                                    cos_angle = np.dot(vec_hd, vec_ha) / (norm_hd * norm_ha)
                                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                                    angle = np.degrees(np.arccos(cos_angle))
                                    
                                    # Check if meets angle criterion (using current ang_cut)
                                    # For linear H-bonds, angle ≈ 180°
                                    if angle >= ang_cut:
                                        frame_hbonds.add((donor.index, h.index, acc.index))
                        
                        hbonds_per_frame.append(frame_hbonds)
                    
                    # Collect all unique H-bonds
                    all_hbonds = set()
                    for frame_hbonds in hbonds_per_frame:
                        all_hbonds.update(frame_hbonds)
                    
                    # Calculate occupancy and lifetimes
                    occupancy_dict = {}
                    lifetimes_dict = {}
                    
                    for hbond in all_hbonds:
                        # Track which frames this H-bond appears in
                        hbond_frames = []
                        for i, frame_idx in enumerate(frames):
                            if hbond in hbonds_per_frame[i]:
                                hbond_frames.append(frame_idx)
                        
                        # Occupancy
                        occupancy_dict[hbond] = len(hbond_frames) / len(frames)
                        
                        # Lifetimes (consecutive frames)
                        if len(hbond_frames) > 0:
                            hbond_frames = sorted(hbond_frames)
                            lifetimes = []
                            current_lifetime = 1
                            
                            for i in range(1, len(hbond_frames)):
                                if hbond_frames[i] == hbond_frames[i-1] + 1:
                                    current_lifetime += 1
                                else:
                                    lifetimes.append(current_lifetime)
                                    current_lifetime = 1
                            
                            lifetimes.append(current_lifetime)
                            lifetimes_dict[hbond] = lifetimes
                    
                    # Timeseries
                    timeseries = np.array([len(frame_hbonds) for frame_hbonds in hbonds_per_frame])
                    
                    # Mean statistics
                    all_lifetimes = []
                    for lifetimes in lifetimes_dict.values():
                        all_lifetimes.extend(lifetimes)
                    
                    mean_lifetime = np.mean(all_lifetimes) if len(all_lifetimes) > 0 else 0.0
                    mean_count = np.mean(timeseries) if len(timeseries) > 0 else 0.0
                    
                    # Get H-bond pair details
                    hbond_pairs = []
                    for (donor_idx, h_idx, acc_idx) in all_hbonds:
                        try:
                            donor_atom = u.atoms[donor_idx]
                            h_atom = u.atoms[h_idx]
                            acc_atom = u.atoms[acc_idx]
                            hbond_pairs.append((
                                donor_atom.resname,
                                donor_atom.name,
                                donor_idx,
                                h_atom.name,
                                h_idx,
                                acc_atom.resname,
                                acc_atom.name,
                                acc_idx
                            ))
                        except:
                            pass
                    
                    hbond_results[cluster_id] = {
                        'timeseries': timeseries,
                        'n_hbonds': len(all_hbonds),
                        'occupancy': occupancy_dict,
                        'lifetimes': lifetimes_dict,
                        'mean_lifetime': mean_lifetime,
                        'mean_count': mean_count,
                        'hbond_pairs': hbond_pairs,
                        'donors': donors,
                        'acceptors': acceptors,
                        'distance_cutoff': dist_cut,
                        'angle_cutoff': ang_cut
                    }
                    
                    if not is_multi_cutoff:
                        print(f"    ✓ Unique H-bonds detected: {len(all_hbonds)}")
                        print(f"      Mean H-bonds per frame: {mean_count:.2f}")
                        print(f"      Mean H-bond lifetime: {mean_lifetime:.1f} frames")
                        if len(all_hbonds) > 0:
                            top_pairs = sorted(occupancy_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                            print(f"      Top 3 H-bonds by occupancy:")
                            for (donor_idx, h_idx, acc_idx), occ in top_pairs:
                                try:
                                    donor_atom = u.atoms[donor_idx]
                                    h_atom = u.atoms[h_idx]
                                    acc_atom = u.atoms[acc_idx]
                                    print(f"        {donor_atom.resname}:{donor_atom.name}[{donor_idx}]-{h_atom.name}[{h_idx}]...{acc_atom.resname}:{acc_atom.name}[{acc_idx}]: {occ*100:.1f}%")
                                except:
                                    print(f"        Donor[{donor_idx}]-H[{h_idx}]...Acceptor[{acc_idx}]: {occ*100:.1f}%")
                
                # Store this cutoff combination
                all_results[cutoff_key] = hbond_results
        
        # Generate report if requested
        if show_report and is_multi_cutoff:
            print(f"\n{'='*80}")
            print("COMPREHENSIVE H-BOND ANALYSIS REPORT")
            print(f"{'='*80}")
            
            import pandas as pd
            
            report_data = []
            for (dist, ang), cluster_results in all_results.items():
                for cluster_id, data in cluster_results.items():
                    report_data.append({
                        'Distance_Cutoff': f"{dist:.1f}",
                        'Angle_Cutoff': f"{ang:.1f}",
                        'Cluster': cluster_id,
                        'N_HBonds': data['n_hbonds'],
                        'Mean_per_Frame': f"{data['mean_count']:.2f}",
                        'Mean_Lifetime': f"{data['mean_lifetime']:.1f}",
                        'Max_Occupancy': f"{max(data['occupancy'].values())*100:.1f}%" if data['occupancy'] else "0.0%"
                    })
            
            df = pd.DataFrame(report_data)
            print("\n" + df.to_string(index=False))
            print(f"\n{'='*80}")
        
        elif not show_report and is_multi_cutoff:
            # Compact summary for multi-cutoff
            print(f"\n{'='*60}")
            print("H-BOND ANALYSIS SUMMARY")
            print(f"{'='*60}")
            for (dist, ang), cluster_results in all_results.items():
                total_hbonds = sum(data['n_hbonds'] for data in cluster_results.values())
                avg_per_frame = np.mean([data['mean_count'] for data in cluster_results.values()])
                print(f"  Distance={dist:.1f}Å, Angle={ang:.1f}°: {total_hbonds} unique H-bonds (avg {avg_per_frame:.2f}/frame)")
        
        # Store in self.hbond_data
        if not hasattr(self, 'hbond_data'):
            self.hbond_data = {}
        key = f"{donors}__{acceptors}"
        self.hbond_data[key] = all_results
        
        # Return appropriate structure
        if is_multi_cutoff:
            print(f"\n✓ Multi-cutoff hydrogen bond analysis completed")
            print(f"  Access results: hbond_results[(distance, angle)][cluster_id]")
            return all_results
        else:
            # For single cutoff, return the inner dict directly (backward compatible)
            print(f"\n✓ Hydrogen bond analysis completed")
            return all_results[(distance_cutoffs[0], angle_cutoffs[0])]
    
    def format_hbond_pairs(self, hbond_key: str, cluster_id: int, 
                          max_pairs: int = None,
                          show_indices: bool = True) -> str:
        """
        Format hydrogen bond pairs for readable display.
        
        Note: The same atom names (e.g., "api:N16 → MMT:Ob") can appear multiple
        times because they represent bonds to DIFFERENT atoms with the same name.
        For example, multiple "Ob" atoms exist in the clay surface. The atom indices
        distinguish between these different atoms.
        
        Parameters
        ----------
        hbond_key : str
            Key in hbond_data dictionary
        cluster_id : int
            Cluster to display
        max_pairs : int, optional
            Maximum number of pairs to show (None = all)
        show_indices : bool, default=True
            Show atom indices to distinguish atoms with same names
            
        Returns
        -------
        str
            Formatted string of H-bond pairs
            
        Example
        -------
        >>> print(analyzer.format_hbond_pairs(
        ...     'resname api and (...)__name OW',
        ...     cluster_id=0,
        ...     max_pairs=10,
        ...     show_indices=True
        ... ))
        """
        if not hasattr(self, 'hbond_data') or hbond_key not in self.hbond_data:
            raise ValueError(f"H-bond data not found: {hbond_key}")
        
        if cluster_id not in self.hbond_data[hbond_key]:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        results = self.hbond_data[hbond_key][cluster_id]
        hbond_pairs = results['hbond_pairs']
        occupancy = results['occupancy']
        
        # Sort by occupancy
        pairs_with_occ = []
        for pair_info in hbond_pairs:
            donor_res, donor_name, donor_idx, acc_res, acc_name, acc_idx = pair_info
            occ = occupancy.get((donor_idx, acc_idx), 0.0)
            pairs_with_occ.append((pair_info, occ))
        
        pairs_with_occ.sort(key=lambda x: x[1], reverse=True)
        
        if max_pairs is not None:
            pairs_with_occ = pairs_with_occ[:max_pairs]
        
        lines = []
        lines.append(f"Cluster {cluster_id}: {len(hbond_pairs)} unique H-bond pairs")
        lines.append(f"Mean H-bonds per frame: {results['mean_count']:.2f}")
        lines.append(f"Mean lifetime: {results['mean_lifetime']:.1f} frames")
        lines.append("\nH-bond pairs (sorted by occupancy):")
        
        for (donor_res, donor_name, donor_idx, acc_res, acc_name, acc_idx), occ in pairs_with_occ:
            if show_indices:
                lines.append(
                    f"  {donor_res}:{donor_name}[{donor_idx}] → "
                    f"{acc_res}:{acc_name}[{acc_idx}]  ({occ*100:.1f}%)"
                )
            else:
                lines.append(
                    f"  {donor_res}:{donor_name} → {acc_res}:{acc_name}  ({occ*100:.1f}%)"
                )
        
        return "\n".join(lines)
    
    def plot_hydrogen_bond_analysis(self, hbond_results: dict,
                                    plot_type: str = 'timeseries',
                                    cluster_ids: Optional[List[int]] = None,
                                    distance_cutoff: Optional[Union[float, List[float]]] = None,
                                    angle_cutoff: Optional[Union[float, List[float]]] = None,
                                    figsize: Tuple[float, float] = (12, 8),
                                    colors: Optional[List[str]] = None,
                                    show_legend: bool = True,
                                    save_path: Optional[str] = None,
                                    dpi: int = 300):
        """
        Plot hydrogen bond analysis results with support for multi-cutoff comparisons.
        
        This method automatically detects single vs multi-cutoff data and provides
        various visualization options.
        
        Parameters
        ----------
        hbond_results : dict
            Results from compute_hydrogen_bonds()
            Single cutoff: {cluster_id: {...}}
            Multi-cutoff: {(distance, angle): {cluster_id: {...}}}
        plot_type : str
            Type of plot to generate:
            - 'timeseries': H-bond count over trajectory
            - 'occupancy': H-bond occupancy distribution
            - 'lifetime': H-bond lifetime distribution
            - 'comparison': Compare cutoffs (multi-cutoff only)
            - 'summary_bar': Bar chart summary across clusters/cutoffs
        cluster_ids : list of int, optional
            Which clusters to plot (default: all)
        distance_cutoff : float or list of float, optional
            For multi-cutoff data, select specific distance(s) to plot
            If None, plots all distances
        angle_cutoff : float or list of float, optional
            For multi-cutoff data, select specific angle(s) to plot
            If None, plots all angles
        figsize : tuple
            Figure size (width, height) in inches
        colors : list of str, optional
            Colors for different series
        show_legend : bool
            Display legend (default=True)
        save_path : str, optional
            Path to save figure
        dpi : int
            Resolution for saved figure (default=300)
        
        Returns
        -------
        fig : matplotlib Figure
        ax : matplotlib Axes or array of Axes
        
        Examples
        --------
        >>> # Single cutoff timeseries
        >>> fig, ax = analyzer.plot_hydrogen_bond_analysis(
        ...     hbond_results,
        ...     plot_type='timeseries',
        ...     cluster_ids=[0, 1]
        ... )
        
        >>> # Multi-cutoff comparison
        >>> fig, ax = analyzer.plot_hydrogen_bond_analysis(
        ...     hbond_results,
        ...     plot_type='comparison',
        ...     cluster_ids=[0]
        ... )
        
        >>> # Summary bar chart for all cutoffs
        >>> fig, ax = analyzer.plot_hydrogen_bond_analysis(
        ...     hbond_results,
        ...     plot_type='summary_bar'
        ... )
        """
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        
        # Set Times New Roman font
        rcParams['font.family'] = 'Times New Roman'
        rcParams['mathtext.fontset'] = 'custom'
        rcParams['mathtext.rm'] = 'Times New Roman'
        
        # Detect if multi-cutoff data
        first_key = list(hbond_results.keys())[0]
        is_multi_cutoff = isinstance(first_key, tuple)
        
        if is_multi_cutoff:
            # Extract available cutoffs
            available_cutoffs = list(hbond_results.keys())
            distance_cutoffs = sorted(list(set(k[0] for k in available_cutoffs)))
            angle_cutoffs = sorted(list(set(k[1] for k in available_cutoffs)))
            
            # Filter by user selection
            if distance_cutoff is not None:
                if isinstance(distance_cutoff, (int, float)):
                    distance_cutoffs = [float(distance_cutoff)]
                else:
                    distance_cutoffs = [d for d in distance_cutoffs if d in distance_cutoff]
            
            if angle_cutoff is not None:
                if isinstance(angle_cutoff, (int, float)):
                    angle_cutoffs = [float(angle_cutoff)]
                else:
                    angle_cutoffs = [a for a in angle_cutoffs if a in angle_cutoff]
            
            # Get cluster IDs from first available cutoff combination
            first_cutoff = (distance_cutoffs[0], angle_cutoffs[0])
            available_clusters = list(hbond_results[first_cutoff].keys())
        else:
            # Single cutoff data
            available_clusters = list(hbond_results.keys())
            distance_cutoffs = [hbond_results[available_clusters[0]]['distance_cutoff']]
            angle_cutoffs = [hbond_results[available_clusters[0]]['angle_cutoff']]
        
        if cluster_ids is None:
            cluster_ids = available_clusters
        
        # Default colors
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(cluster_ids))))
        
        # Create figure based on plot type
        fig, ax = plt.subplots(figsize=figsize)
        
        if plot_type == 'timeseries':
            # Plot H-bond count over time
            for i, cluster_id in enumerate(cluster_ids):
                if is_multi_cutoff:
                    for (d, a) in [(d, a) for d in distance_cutoffs for a in angle_cutoffs]:
                        if (d, a) in hbond_results and cluster_id in hbond_results[(d, a)]:
                            data = hbond_results[(d, a)][cluster_id]
                            label = f"C{cluster_id} d={d:.1f}Å a={a:.0f}°"
                            ax.plot(data['timeseries'], label=label, alpha=0.7)
                else:
                    if cluster_id in hbond_results:
                        data = hbond_results[cluster_id]
                        ax.plot(data['timeseries'], label=f"Cluster {cluster_id}", 
                               color=colors[i], linewidth=1.5)
            
            ax.set_xlabel('Frame', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of H-bonds', fontsize=14, fontweight='bold')
            ax.set_title('Hydrogen Bond Timeseries', fontsize=16, fontweight='bold')
            
        elif plot_type == 'occupancy':
            # Histogram of H-bond occupancies
            for i, cluster_id in enumerate(cluster_ids):
                if is_multi_cutoff:
                    cutoff_key = (distance_cutoffs[0], angle_cutoffs[0])
                    if cluster_id in hbond_results[cutoff_key]:
                        data = hbond_results[cutoff_key][cluster_id]
                        occupancies = list(data['occupancy'].values())
                        ax.hist(occupancies, bins=20, alpha=0.6, label=f"Cluster {cluster_id}",
                               color=colors[i])
                else:
                    if cluster_id in hbond_results:
                        occupancies = list(hbond_results[cluster_id]['occupancy'].values())
                        ax.hist(occupancies, bins=20, alpha=0.6, label=f"Cluster {cluster_id}",
                               color=colors[i])
            
            ax.set_xlabel('Occupancy', fontsize=14, fontweight='bold')
            ax.set_ylabel('Count', fontsize=14, fontweight='bold')
            ax.set_title('H-bond Occupancy Distribution', fontsize=16, fontweight='bold')
            
        elif plot_type == 'lifetime':
            # Histogram of H-bond lifetimes
            for i, cluster_id in enumerate(cluster_ids):
                if is_multi_cutoff:
                    cutoff_key = (distance_cutoffs[0], angle_cutoffs[0])
                    if cluster_id in hbond_results[cutoff_key]:
                        data = hbond_results[cutoff_key][cluster_id]
                        all_lifetimes = []
                        for lts in data['lifetimes'].values():
                            all_lifetimes.extend(lts)
                        if all_lifetimes:
                            ax.hist(all_lifetimes, bins=30, alpha=0.6, label=f"Cluster {cluster_id}",
                                   color=colors[i])
                else:
                    if cluster_id in hbond_results:
                        all_lifetimes = []
                        for lts in hbond_results[cluster_id]['lifetimes'].values():
                            all_lifetimes.extend(lts)
                        if all_lifetimes:
                            ax.hist(all_lifetimes, bins=30, alpha=0.6, label=f"Cluster {cluster_id}",
                                   color=colors[i])
            
            ax.set_xlabel('Lifetime (frames)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Count', fontsize=14, fontweight='bold')
            ax.set_title('H-bond Lifetime Distribution', fontsize=16, fontweight='bold')
            
        elif plot_type == 'comparison' and is_multi_cutoff:
            # Compare different cutoff combinations
            x_labels = []
            y_values = []
            
            for (d, a) in [(d, a) for d in distance_cutoffs for a in angle_cutoffs]:
                if (d, a) in hbond_results:
                    for cluster_id in cluster_ids:
                        if cluster_id in hbond_results[(d, a)]:
                            data = hbond_results[(d, a)][cluster_id]
                            x_labels.append(f"C{cluster_id}\nd={d:.1f}\na={a:.0f}")
                            y_values.append(data['mean_count'])
            
            x_pos = np.arange(len(x_labels))
            ax.bar(x_pos, y_values, color=colors[:len(y_values)])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Mean H-bonds per Frame', fontsize=14, fontweight='bold')
            ax.set_title('H-bond Comparison: Cutoff Effects', fontsize=16, fontweight='bold')
            
        elif plot_type == 'summary_bar':
            # Summary bar chart
            if is_multi_cutoff:
                x_labels = []
                n_hbonds = []
                mean_counts = []
                
                for (d, a) in [(d, a) for d in distance_cutoffs for a in angle_cutoffs]:
                    if (d, a) in hbond_results:
                        for cluster_id in cluster_ids:
                            if cluster_id in hbond_results[(d, a)]:
                                data = hbond_results[(d, a)][cluster_id]
                                x_labels.append(f"C{cluster_id}\n{d:.1f}Å\n{a:.0f}°")
                                n_hbonds.append(data['n_hbonds'])
                                mean_counts.append(data['mean_count'])
                
                x_pos = np.arange(len(x_labels))
                width = 0.35
                
                ax.bar(x_pos - width/2, n_hbonds, width, label='Unique H-bonds', alpha=0.8)
                ax.bar(x_pos + width/2, mean_counts, width, label='Mean per frame', alpha=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
                ax.set_ylabel('Count', fontsize=14, fontweight='bold')
                ax.set_title('H-bond Summary Statistics', fontsize=16, fontweight='bold')
            else:
                x_labels = [f"Cluster {cid}" for cid in cluster_ids]
                n_hbonds = [hbond_results[cid]['n_hbonds'] for cid in cluster_ids]
                mean_counts = [hbond_results[cid]['mean_count'] for cid in cluster_ids]
                
                x_pos = np.arange(len(x_labels))
                width = 0.35
                
                ax.bar(x_pos - width/2, n_hbonds, width, label='Unique H-bonds', 
                      color=colors[0], alpha=0.8)
                ax.bar(x_pos + width/2, mean_counts, width, label='Mean per frame', 
                      color=colors[1], alpha=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, fontsize=12)
                ax.set_ylabel('Count', fontsize=14, fontweight='bold')
                ax.set_title('H-bond Summary Statistics', fontsize=16, fontweight='bold')
        
        # Apply common styling
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(alpha=0.3, linestyle='--')
        
        if show_legend and plot_type != 'comparison':
            ax.legend(fontsize=10, framealpha=0.9)
        elif show_legend and plot_type == 'summary_bar':
            ax.legend(fontsize=12, framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        
        return fig, ax
    
    def compute_bridging_analysis(self, clay_sel: str, ion_sel: str, molecule_sel: str,
                                  cutoff_clay_ion: float = 3.5,
                                  cutoff_ion_molecule: float = 3.5,
                                  angle_threshold: float = 120.0,
                                  require_z_ordering: bool = True,
                                  min_coordination_clay: int = 1,
                                  min_coordination_molecule: int = 1,
                                  cluster_ids: Optional[List[int]] = None):
        """
        Analyze ion bridging effects between clay surface and organic molecules.
        
        A true bridge requires the ion to be geometrically positioned BETWEEN
        the clay and molecule, not just touching both. This method implements
        comprehensive geometric criteria to detect authentic bridging configurations.
        
        **Bridging Criteria (ALL must be satisfied):**
        
        1. **Distance criterion**: 
           - Clay-Ion distance < cutoff_clay_ion
           - Ion-Molecule distance < cutoff_ion_molecule
        
        2. **Position criterion** (if require_z_ordering=True):
           - Z_clay < Z_ion < Z_molecule
           - Ion must spatially between surfaces
        
        3. **Angle criterion**:
           - Clay_O---Ion---Mol_O angle > angle_threshold
           - Tests for linear arrangement (bridge-like geometry)
           - Rejects bent/folded configurations
        
        4. **Coordination criterion**:
           - Ion coordinates ≥ min_coordination_clay atoms from clay
           - Ion coordinates ≥ min_coordination_molecule atoms from molecule
           - Ensures meaningful multi-atomic interactions
        
        Parameters
        ----------
        clay_sel : str
            Selection for clay surface atoms (typically oxygens).
            Example: 'name Ob' or 'resname MMT and name Ob'
        ion_sel : str
            Selection for bridging ions.
            Example: 'resname NA' or 'resname K'
        molecule_sel : str
            Selection for organic molecule coordination sites.
            Example: 'resname CIP and (name O1 or name O3)'
        cutoff_clay_ion : float, default=3.5
            Maximum distance (Å) for clay-ion contact
        cutoff_ion_molecule : float, default=3.5
            Maximum distance (Å) for ion-molecule contact
        angle_threshold : float, default=120.0
            Minimum Clay-Ion-Molecule angle (degrees) for linear arrangement.
            Values: 120° (relaxed), 140° (moderate), 150° (strict)
        require_z_ordering : bool, default=True
            Enforce Z-position criterion (ion between surfaces).
            Set False if surfaces are not parallel or oriented differently.
        min_coordination_clay : int, default=1
            Minimum number of clay atoms that must coordinate the ion
        min_coordination_molecule : int, default=1
            Minimum number of molecule atoms that must coordinate the ion
        cluster_ids : list of int, optional
            Clusters to analyze (default: all)
        
        Returns
        -------
        bridging_results : dict
            {cluster_id: {
                'bridge_frames': ndarray(bool) - has bridge per frame,
                'bridge_frequency': float - % time in bridging configuration,
                'bridge_lifetimes': ndarray(int) - duration of each bridge event (frames),
                'mean_bridge_lifetime': float - average bridge persistence (frames),
                'median_bridge_lifetime': float - median persistence,
                'n_bridge_events': int - total number of bridging events,
                
                # Geometric properties during bridging
                'bridge_angles': ndarray - Clay-Ion-Mol angles when bridging,
                'mean_bridge_angle': float - average angle during bridges,
                'bridge_distances_clay': ndarray - Clay-Ion distances when bridging,
                'bridge_distances_mol': ndarray - Ion-Mol distances when bridging,
                'mean_distance_clay': float,
                'mean_distance_mol': float,
                
                # Coordination numbers during bridging
                'coordination_clay': ndarray - # clay atoms coordinating ion,
                'coordination_mol': ndarray - # molecule atoms coordinating ion,
                'mean_coordination_clay': float,
                'mean_coordination_mol': float,
                
                # Configuration details
                'clay_sel': str,
                'ion_sel': str,
                'molecule_sel': str,
                'cutoff_clay_ion': float,
                'cutoff_ion_molecule': float,
                'angle_threshold': float
            }}
        
        Example
        -------
        >>> # Analyze Na+ bridging between MMT clay and CIP molecule
        >>> bridging = analyzer.compute_bridging_analysis(
        ...     clay_sel='name Ob',
        ...     ion_sel='resname NA',
        ...     molecule_sel='resname CIP and (name O1 or name O3)',
        ...     cutoff_clay_ion=3.2,
        ...     cutoff_ion_molecule=3.2,
        ...     angle_threshold=130.0,
        ...     require_z_ordering=True
        ... )
        >>> 
        >>> # Access results
        >>> for cluster in bridging:
        ...     freq = bridging[cluster]['bridge_frequency']
        ...     lifetime = bridging[cluster]['mean_bridge_lifetime']
        ...     angle = bridging[cluster]['mean_bridge_angle']
        ...     print(f"Cluster {cluster}: {freq:.1f}% bridging, "
        ...           f"lifetime={lifetime:.1f} frames, angle={angle:.1f}°")
        """
        if not hasattr(self, 'trajectory_data'):
            raise ValueError("Load trajectories first with load_cluster_trajectories()")
        
        # Handle cluster_ids
        if cluster_ids is None or cluster_ids == 'all':
            cluster_ids = sorted(self.trajectory_data.keys())
        
        print(f"\n{'='*70}")
        print(f"Computing Ion Bridging Analysis")
        print(f"{'='*70}")
        print(f"  Clay surface: {clay_sel}")
        print(f"  Bridging ion: {ion_sel}")
        print(f"  Molecule: {molecule_sel}")
        print(f"\n  Criteria:")
        print(f"    • Distance: Clay-Ion < {cutoff_clay_ion}Å, Ion-Mol < {cutoff_ion_molecule}Å")
        print(f"    • Angle: Clay-Ion-Mol > {angle_threshold}°")
        print(f"    • Z-ordering: {'Enabled' if require_z_ordering else 'Disabled'}")
        print(f"    • Coordination: Clay≥{min_coordination_clay}, Mol≥{min_coordination_molecule}")
        
        bridging_results = {}
        
        for cluster_id in cluster_ids:
            print(f"\n  {'─'*66}")
            print(f"  Cluster {cluster_id}:")
            print(f"  {'─'*66}")
            
            u = self.trajectory_data[cluster_id]['universe']
            frames = self.trajectory_data[cluster_id]['frames']
            
            # Select atoms
            clay_atoms = u.select_atoms(clay_sel)
            ion_atoms = u.select_atoms(ion_sel)
            mol_atoms = u.select_atoms(molecule_sel)
            
            print(f"    Selected atoms:")
            print(f"      • Clay: {len(clay_atoms)} atoms")
            print(f"      • Ions: {len(ion_atoms)} atoms")
            print(f"      • Molecule: {len(mol_atoms)} atoms")
            
            if len(ion_atoms) == 0:
                print(f"    ⚠ Warning: No ions found, skipping cluster {cluster_id}")
                continue
            
            # Storage arrays
            bridge_frames = []
            bridge_angles = []
            bridge_dist_clay = []
            bridge_dist_mol = []
            coord_clay_list = []
            coord_mol_list = []
            
            # Analyze each frame
            for fidx, frame_idx in enumerate(frames):
                u.trajectory[frame_idx]
                
                # Get positions
                clay_pos = clay_atoms.positions
                ion_pos = ion_atoms.positions
                mol_pos = mol_atoms.positions
                
                # CRITICAL FIX: Check EACH ion individually, not their average!
                # A frame has bridging if ANY ion forms a bridge
                frame_has_bridge = False
                frame_angles = []
                frame_dist_clay = []
                frame_dist_mol = []
                frame_coord_clay = []
                frame_coord_mol = []
                
                # Loop through each Na+ ion
                for ion_center in ion_pos:
                    # ═══════════════════════════════════════════════════════════
                    # CRITERION 1: Distance check
                    # ═══════════════════════════════════════════════════════════
                    # Compute distances from THIS ion to all clay/mol atoms
                    dist_to_all_clay = cdist([ion_center], clay_pos)[0]
                    dist_to_all_mol = cdist([ion_center], mol_pos)[0]
                    
                    # Get minimum distances
                    dist_clay_ion = np.min(dist_to_all_clay)
                    dist_ion_mol = np.min(dist_to_all_mol)
                    
                    if dist_clay_ion >= cutoff_clay_ion or dist_ion_mol >= cutoff_ion_molecule:
                        continue  # This ion doesn't bridge, try next ion
                    
                    # Get the nearest atoms for geometry calculations
                    nearest_clay_idx = np.argmin(dist_to_all_clay)
                    nearest_mol_idx = np.argmin(dist_to_all_mol)
                    nearest_clay_pos = clay_pos[nearest_clay_idx]
                    nearest_mol_pos = mol_pos[nearest_mol_idx]
                    
                    # ═══════════════════════════════════════════════════════════
                    # CRITERION 2: Z-position check (ion between surfaces)
                    # ═══════════════════════════════════════════════════════════
                    if require_z_ordering:
                        z_clay = nearest_clay_pos[2]
                        z_ion = ion_center[2]
                        z_mol = nearest_mol_pos[2]
                        
                        # Ion must be between clay and molecule
                        is_between = (z_clay < z_ion < z_mol) or (z_mol < z_ion < z_clay)
                        
                        if not is_between:
                            continue  # This ion doesn't satisfy Z-ordering, try next ion
                    
                    # ═══════════════════════════════════════════════════════════
                    # CRITERION 3: Angle check (linear arrangement)
                    # ═══════════════════════════════════════════════════════════
                    # Vectors from ion to nearest clay and molecule atoms
                    vec_clay = nearest_clay_pos - ion_center
                    vec_mol = nearest_mol_pos - ion_center
                    
                    # Normalize
                    vec_clay_norm = vec_clay / np.linalg.norm(vec_clay)
                    vec_mol_norm = vec_mol / np.linalg.norm(vec_mol)
                    
                    # Angle between vectors
                    cos_angle = np.dot(vec_clay_norm, vec_mol_norm)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    
                    # For Clay-Ion-Mol angle, we want the supplementary angle if needed
                    # The angle should be large (close to 180°) for linear bridge
                    if angle_deg < angle_threshold:
                        continue  # This ion doesn't satisfy angle requirement, try next ion
                    
                    # ═══════════════════════════════════════════════════════════
                    # CRITERION 4: Coordination number check
                    # ═══════════════════════════════════════════════════════════
                    # Count atoms within cutoff of ion (reuse already computed distances)
                    n_coord_clay = np.sum(dist_to_all_clay < cutoff_clay_ion)
                    n_coord_mol = np.sum(dist_to_all_mol < cutoff_ion_molecule)
                    
                    if n_coord_clay < min_coordination_clay or n_coord_mol < min_coordination_molecule:
                        continue  # This ion doesn't satisfy coordination requirement, try next ion
                    
                    # ═══════════════════════════════════════════════════════════
                    # ALL CRITERIA SATISFIED → THIS ION FORMS A BRIDGE
                    # ═══════════════════════════════════════════════════════════
                    frame_has_bridge = True
                    frame_angles.append(angle_deg)
                    frame_dist_clay.append(dist_clay_ion)
                    frame_dist_mol.append(dist_ion_mol)
                    frame_coord_clay.append(n_coord_clay)
                    frame_coord_mol.append(n_coord_mol)
                    # Note: We only record the first bridging ion per frame
                    # If you want all bridging ions, remove this break
                    break
                
                # After checking all ions in this frame, record frame result
                bridge_frames.append(frame_has_bridge)
                if frame_has_bridge:
                    # Use the first bridging ion's properties
                    bridge_angles.append(frame_angles[0])
                    bridge_dist_clay.append(frame_dist_clay[0])
                    bridge_dist_mol.append(frame_dist_mol[0])
                    coord_clay_list.append(frame_coord_clay[0])
                    coord_mol_list.append(frame_coord_mol[0])
                
                # Progress indicator every 1000 frames
                if (fidx + 1) % 1000 == 0:
                    n_bridges = sum(bridge_frames)
                    print(f"    Progress: {fidx+1}/{len(frames)} frames, "
                          f"{n_bridges} bridges detected ({100*n_bridges/(fidx+1):.1f}%)")
            
            # Convert to arrays
            bridge_frames = np.array(bridge_frames, dtype=bool)
            bridge_angles = np.array(bridge_angles)
            bridge_dist_clay = np.array(bridge_dist_clay)
            bridge_dist_mol = np.array(bridge_dist_mol)
            coord_clay_arr = np.array(coord_clay_list)
            coord_mol_arr = np.array(coord_mol_list)
            
            # Calculate bridge frequency
            bridge_freq = 100.0 * np.sum(bridge_frames) / len(bridge_frames)
            
            # ═══════════════════════════════════════════════════════════════
            # Compute bridge lifetimes (consecutive True values)
            # ═══════════════════════════════════════════════════════════════
            lifetimes = []
            current_lifetime = 0
            
            for is_bridge in bridge_frames:
                if is_bridge:
                    current_lifetime += 1
                else:
                    if current_lifetime > 0:
                        lifetimes.append(current_lifetime)
                        current_lifetime = 0
            
            # Don't forget final bridge
            if current_lifetime > 0:
                lifetimes.append(current_lifetime)
            
            lifetimes = np.array(lifetimes) if len(lifetimes) > 0 else np.array([0])
            
            # ═══════════════════════════════════════════════════════════════
            # Store results
            # ═══════════════════════════════════════════════════════════════
            bridging_results[cluster_id] = {
                # Bridge occurrence
                'bridge_frames': bridge_frames,
                'bridge_frequency': bridge_freq,
                'bridge_lifetimes': lifetimes,
                'mean_bridge_lifetime': np.mean(lifetimes),
                'median_bridge_lifetime': np.median(lifetimes),
                'n_bridge_events': len(lifetimes),
                
                # Geometric properties during bridging
                'bridge_angles': bridge_angles,
                'mean_bridge_angle': np.mean(bridge_angles) if len(bridge_angles) > 0 else 0.0,
                'std_bridge_angle': np.std(bridge_angles) if len(bridge_angles) > 0 else 0.0,
                
                'bridge_distances_clay': bridge_dist_clay,
                'bridge_distances_mol': bridge_dist_mol,
                'mean_distance_clay': np.mean(bridge_dist_clay) if len(bridge_dist_clay) > 0 else 0.0,
                'std_distance_clay': np.std(bridge_dist_clay) if len(bridge_dist_clay) > 0 else 0.0,
                'mean_distance_mol': np.mean(bridge_dist_mol) if len(bridge_dist_mol) > 0 else 0.0,
                'std_distance_mol': np.std(bridge_dist_mol) if len(bridge_dist_mol) > 0 else 0.0,
                
                # Coordination during bridging
                'coordination_clay': coord_clay_arr,
                'coordination_mol': coord_mol_arr,
                'mean_coordination_clay': np.mean(coord_clay_arr) if len(coord_clay_arr) > 0 else 0.0,
                'mean_coordination_mol': np.mean(coord_mol_arr) if len(coord_mol_arr) > 0 else 0.0,
                
                # Configuration
                'clay_sel': clay_sel,
                'ion_sel': ion_sel,
                'molecule_sel': molecule_sel,
                'cutoff_clay_ion': cutoff_clay_ion,
                'cutoff_ion_molecule': cutoff_ion_molecule,
                'angle_threshold': angle_threshold
            }
            
            # Print summary
            print(f"\n    ═══════════════════════════════════════════════════════════")
            print(f"    ✓ Bridging Analysis Complete")
            print(f"    ═══════════════════════════════════════════════════════════")
            print(f"    Bridge frequency: {bridge_freq:.2f}%")
            print(f"    Number of bridging events: {len(lifetimes)}")
            print(f"    Mean bridge lifetime: {np.mean(lifetimes):.1f} frames")
            print(f"    Median bridge lifetime: {np.median(lifetimes):.1f} frames")
            if len(bridge_angles) > 0:
                print(f"\n    Geometry during bridging:")
                print(f"      • Mean Clay-Ion-Mol angle: {np.mean(bridge_angles):.1f}° ± {np.std(bridge_angles):.1f}°")
                print(f"      • Mean Clay-Ion distance: {np.mean(bridge_dist_clay):.2f} ± {np.std(bridge_dist_clay):.2f} Å")
                print(f"      • Mean Ion-Mol distance: {np.mean(bridge_dist_mol):.2f} ± {np.std(bridge_dist_mol):.2f} Å")
                print(f"      • Mean coordination (Clay): {np.mean(coord_clay_arr):.1f}")
                print(f"      • Mean coordination (Mol): {np.mean(coord_mol_arr):.1f}")
        
        # Store in analyzer
        if not hasattr(self, 'bridging_data'):
            self.bridging_data = {}
        key = f"{clay_sel}__{ion_sel}__{molecule_sel}"
        self.bridging_data[key] = bridging_results
        
        print(f"\n{'='*70}")
        print(f"✓ Ion Bridging Analysis Completed")
        print(f"{'='*70}\n")
        
        return bridging_results
    
    def generate_bridging_report(self,
                                clay_sel: str,
                                ion_sel: str,
                                molecule_sel: str,
                                cutoff_clay_ion: float = 3.2,
                                cutoff_ion_molecule: float = 3.2,
                                angle_threshold: float = 130.0,
                                require_z_ordering: bool = True,
                                cluster_ids: Union[str, List[int]] = 'all',
                                                # Optional diagnostics
                                compare_angles: bool = False,
                                angle_range: List[float] = None,
                                analyze_distances: bool = False,
                                extract_frames: bool = False,
                                save_frames_to_file: str = None,
                                save_to_file: str = None,
                                return_dataframe: bool = True,
                                display_tables: bool = True,
                                # Direct trajectory extraction
                                extract_trajectory_frames: bool = False,
                                topology_file: str = None,
                                trajectory_file: str = None,
                                output_dir: str = None,
                                **kwargs) -> Dict:
        """
        Comprehensive bridging analysis with optional diagnostics and reporting.
        
        This is a convenience wrapper around compute_bridging_analysis() that adds:
        - Multi-angle threshold comparison
        - Distance distribution analysis  
        - Frame extraction for visualization
        - Formatted report generation
        
        Parameters
        ----------
        clay_sel : str
            MDAnalysis selection string for clay surface atoms
        ion_sel : str
            MDAnalysis selection string for bridging ions
        molecule_sel : str
            MDAnalysis selection string for molecule atoms
        cutoff_clay_ion : float, default=3.2
            Distance cutoff (Å) for clay-ion interaction
        cutoff_ion_molecule : float, default=3.2
            Distance cutoff (Å) for ion-molecule interaction
        angle_threshold : float, default=130.0
            Minimum Clay-Ion-Mol angle (degrees) for bridging
        require_z_ordering : bool, default=True
            If True, ion must be between clay and molecule in Z
        cluster_ids : 'all' or list of int
            Which clusters to analyze
        compare_angles : bool, default=False
            If True, test multiple angle thresholds and compare
        angle_range : list of float, optional
            Angles to test if compare_angles=True. Default: [90,110,120,130,140,150,160]
        analyze_distances : bool, default=True
            If True, compute average distance distributions between ion_sel (Na) and clay_sel (Ob).
            Calculates minimum distance from each Na to nearest clay Ob atom, averaged over frames.
        extract_frames : bool, default=False
            If True, extract frame numbers where bridges occur
        save_frames_to_file : str, optional
            If provided, save frame info to this file (VMD/OVITO compatible format)
        save_to_file : str, optional
            If provided, save comprehensive text report to this file path
        return_dataframe : bool, default=True
            If True and compare_angles=True, return pandas DataFrame
        display_tables : bool, default=True
            If True, print formatted tables to console (like notebook output)
        extract_trajectory_frames : bool, default=False
            If True, extract bridge frames into separate .xtc trajectory files per cluster
        topology_file : str, optional
            Path to topology file (e.g., 'nvt.tpr'). Required if extract_trajectory_frames=True
        trajectory_file : str, optional
            Path to trajectory file (e.g., 'nvt.xtc'). Required if extract_trajectory_frames=True
        output_dir : str, optional
            Directory to save extracted trajectories. Default: 'bridge_trajectories'
        **kwargs
            Additional arguments passed to compute_bridging_analysis
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'bridging_results': Main results from compute_bridging_analysis()
            - 'angle_comparison': DataFrame of angle threshold comparison (if compare_angles=True)
            - 'distance_analysis': Distance distribution stats including average Na-Ob distances (computed by default)
            - 'bridge_frames': List of frame info dicts (if extract_frames=True)
            - 'report_text': Formatted text report (if save_to_file used)
            - 'extracted_trajectories': List of extracted trajectory file paths (if extract_trajectory_frames=True)
        """
        from scipy.spatial.distance import cdist
        from MDAnalysis.analysis import distances
        
        print("="*80)
        print("COMPREHENSIVE BRIDGING ANALYSIS REPORT")
        print("="*80)
        
        report = {}
        report_text = []
        
        # ═══════════════════════════════════════════════════════════
        # 1. Core bridging analysis
        # ═══════════════════════════════════════════════════════════
        print("\n[1/4] Running core bridging analysis...")
        bridging_results = self.compute_bridging_analysis(
            clay_sel=clay_sel,
            ion_sel=ion_sel,
            molecule_sel=molecule_sel,
            cutoff_clay_ion=cutoff_clay_ion,
            cutoff_ion_molecule=cutoff_ion_molecule,
            angle_threshold=angle_threshold,
            require_z_ordering=require_z_ordering,
            cluster_ids=cluster_ids,
            **kwargs
        )
        report['bridging_results'] = bridging_results
        
        # ═══════════════════════════════════════════════════════════
        # 2. Optional: Angle threshold comparison
        # ═══════════════════════════════════════════════════════════
        if compare_angles:
            print("\n[2/4] Comparing angle thresholds...")
            if angle_range is None:
                angle_range = [90, 110, 120, 130, 140, 150, 160]
            
            angle_comparison_data = []
            
            for angle_thresh in angle_range:
                print(f"  Testing {angle_thresh}° (cone: {180-angle_thresh}°)...")
                temp_results = self.compute_bridging_analysis(
                    clay_sel=clay_sel,
                    ion_sel=ion_sel,
                    molecule_sel=molecule_sel,
                    cutoff_clay_ion=cutoff_clay_ion,
                    cutoff_ion_molecule=cutoff_ion_molecule,
                    angle_threshold=angle_thresh,
                    require_z_ordering=require_z_ordering,
                    cluster_ids=cluster_ids,
                    **kwargs
                )
                
                for cid, data in temp_results.items():
                    angle_comparison_data.append({
                        'Angle_Threshold': angle_thresh,
                        'Cone_Width': 180 - angle_thresh,
                        'Cluster': cid,
                        'Frequency_Percent': data['bridge_frequency'],
                        'Events': data['n_bridge_events'],
                        'Mean_Lifetime': data['mean_bridge_lifetime'],
                        'Mean_Angle': data['mean_bridge_angle'] if data['bridge_frequency'] > 0 else np.nan,
                        'Std_Angle': data['std_bridge_angle'] if data['bridge_frequency'] > 0 else np.nan,
                        'Mean_Clay_Dist': data['mean_distance_clay'] if data['bridge_frequency'] > 0 else np.nan,
                        'Std_Clay_Dist': data['std_distance_clay'] if data['bridge_frequency'] > 0 else np.nan,
                        'Mean_Mol_Dist': data['mean_distance_mol'] if data['bridge_frequency'] > 0 else np.nan,
                        'Std_Mol_Dist': data['std_distance_mol'] if data['bridge_frequency'] > 0 else np.nan
                    })
            
            if return_dataframe:
                import pandas as pd
                df_angles = pd.DataFrame(angle_comparison_data)
                report['angle_comparison'] = df_angles
                print(f"\n  ✓ Angle comparison complete ({len(angle_range)} thresholds tested)")
                
                # Display formatted tables
                if display_tables:
                    print("\n" + "="*80)
                    print("RESULTS SUMMARY - ALL CLUSTERS")
                    print("="*80)
                    
                    # Format for display with cleaner column names
                    df_display = df_angles.copy()
                    df_display.columns = ['Angle Threshold', 'Cone Width', 'Cluster', 
                                          'Frequency (%)', 'Events', 'Mean Lifetime', 
                                          'Mean Angle', 'Std Angle', 
                                          'Clay Dist (Å)', 'Std Clay', 'Mol Dist (Å)', 'Std Mol']
                    df_display['Angle Threshold'] = df_display['Angle Threshold'].astype(str) + '°'
                    df_display['Cone Width'] = df_display['Cone Width'].astype(str) + '°'
                    
                    print(df_display.to_string(index=False))
                    
                    # Aggregate by angle threshold
                    print("\n" + "="*80)
                    print("AGGREGATED BY ANGLE THRESHOLD (Mean across clusters)")
                    print("="*80)
                    
                    df_agg = df_angles.groupby(['Angle_Threshold', 'Cone_Width']).agg({
                        'Frequency_Percent': 'mean',
                        'Events': 'sum',
                        'Mean_Lifetime': 'mean',
                        'Mean_Angle': 'mean',
                        'Std_Angle': lambda x: np.sqrt(np.mean(x**2)),  # RMS of stds
                        'Mean_Clay_Dist': 'mean',
                        'Std_Clay_Dist': lambda x: np.sqrt(np.mean(x**2)),
                        'Mean_Mol_Dist': 'mean',
                        'Std_Mol_Dist': lambda x: np.sqrt(np.mean(x**2))
                    }).reset_index()
                    
                    df_agg_display = df_agg.copy()
                    df_agg_display.columns = ['Angle Threshold', 'Cone Width', 'Frequency (%)', 
                                              'Events', 'Mean Lifetime', 'Mean Angle', 'Std Angle',
                                              'Clay Dist (Å)', 'Std Clay', 'Mol Dist (Å)', 'Std Mol']
                    df_agg_display['Angle Threshold'] = df_agg_display['Angle Threshold'].astype(str) + '°'
                    df_agg_display['Cone Width'] = df_agg_display['Cone Width'].astype(str) + '°'
                    
                    print(df_agg_display.to_string(index=False))
                    
                    print("\n" + "="*80)
                    print("INTERPRETATION:")
                    print("="*80)
                    print("• 90° threshold (90° cone): Most permissive - includes bent bridges")
                    print("• 130° threshold (50° cone): Balanced - near-linear bridges only")
                    print("• 160° threshold (20° cone): Strictest - almost perfectly linear")
                    print("="*80)
            else:
                report['angle_comparison'] = angle_comparison_data
        else:
            report['angle_comparison'] = None
            print("\n[2/4] Angle comparison skipped")
        
        # ═══════════════════════════════════════════════════════════
        # 3. Optional: Distance distribution analysis
        # ═══════════════════════════════════════════════════════════
        if analyze_distances:
            print("\n[3/4] Analyzing distance distributions...")
            
            # Use first cluster as representative
            cluster_id = list(self.trajectory_data.keys())[0]
            u = self.trajectory_data[cluster_id]['universe']
            frames = self.trajectory_data[cluster_id]['frames'][:1000]  # Sample 1000 frames
            
            clay_atoms = u.select_atoms(clay_sel)
            na_ions = u.select_atoms(ion_sel)
            mol_atoms = u.select_atoms(molecule_sel)
            
            distances_clay_na = []
            distances_na_mol = []
            
            for ts in u.trajectory[frames]:
                # Use MDAnalysis distance functions to handle PBC properly
                for na in na_ions:
                    # Calculate minimum distance to clay with PBC
                    dist_to_clay = distances.distance_array(na.position.reshape(1,3), clay_atoms.positions, box=u.dimensions)[0]
                    distances_clay_na.append(np.min(dist_to_clay))
                    
                    # Calculate minimum distance to molecule with PBC  
                    dist_to_mol = distances.distance_array(na.position.reshape(1,3), mol_atoms.positions, box=u.dimensions)[0]
                    distances_na_mol.append(np.min(dist_to_mol))
            
            distances_clay_na = np.array(distances_clay_na)
            distances_na_mol = np.array(distances_na_mol)
            
            distance_stats = {
                'clay_to_ion': {
                    'mean': np.mean(distances_clay_na),
                    'median': np.median(distances_clay_na),
                    'min': np.min(distances_clay_na),
                    'max': np.max(distances_clay_na),
                    'pct_within_3.2A': 100 * np.sum(distances_clay_na < 3.2) / len(distances_clay_na),
                    'pct_within_4.0A': 100 * np.sum(distances_clay_na < 4.0) / len(distances_clay_na)
                },
                'ion_to_molecule': {
                    'mean': np.mean(distances_na_mol),
                    'median': np.median(distances_na_mol),
                    'min': np.min(distances_na_mol),
                    'max': np.max(distances_na_mol),
                    'pct_within_3.2A': 100 * np.sum(distances_na_mol < 3.2) / len(distances_na_mol),
                    'pct_within_4.0A': 100 * np.sum(distances_na_mol < 4.0) / len(distances_na_mol)
                }
            }
            report['distance_analysis'] = distance_stats
            print(f"  ✓ Distance analysis complete (sampled {len(frames)} frames)")
            
            # Display formatted distance statistics
            if display_tables:
                print("\n" + "="*80)
                print("MINIMUM DISTANCE TO ANY ATOM (Å)")
                print("="*80)
                print(f"  Na+ → Nearest Clay Ob:")
                print(f"    Mean: {distance_stats['clay_to_ion']['mean']:.2f} Å")
                print(f"    Median: {distance_stats['clay_to_ion']['median']:.2f} Å")
                print(f"    Min: {distance_stats['clay_to_ion']['min']:.2f} Å")
                print(f"    Max: {distance_stats['clay_to_ion']['max']:.2f} Å")
                print(f"    % within 3.2Å: {distance_stats['clay_to_ion']['pct_within_3.2A']:.1f}%")
                print(f"    % within 4.0Å: {distance_stats['clay_to_ion']['pct_within_4.0A']:.1f}%")
                
                print(f"\n  Na+ → Nearest Molecule O:")
                print(f"    Mean: {distance_stats['ion_to_molecule']['mean']:.2f} Å")
                print(f"    Median: {distance_stats['ion_to_molecule']['median']:.2f} Å")
                print(f"    Min: {distance_stats['ion_to_molecule']['min']:.2f} Å")
                print(f"    Max: {distance_stats['ion_to_molecule']['max']:.2f} Å")
                print(f"    % within 3.2Å: {distance_stats['ion_to_molecule']['pct_within_3.2A']:.1f}%")
                print(f"    % within 4.0Å: {distance_stats['ion_to_molecule']['pct_within_4.0A']:.1f}%")
                
                # Calculate contact patterns
                na_near_clay = distance_stats['clay_to_ion']['pct_within_4.0A']
                na_near_mol = distance_stats['ion_to_molecule']['pct_within_4.0A']
                
                print(f"\n" + "─"*80)
                print(f"CONTACT PATTERNS (4.0Å cutoff to ANY atom):")
                print(f"─"*80)
                print(f"  Na+ near clay: {na_near_clay:.1f}%")
                print(f"  Na+ near molecule: {na_near_mol:.1f}%")
                print("="*80)
        else:
            report['distance_analysis'] = None
            print("\n[3/4] Distance analysis skipped (set analyze_distances=True to enable)")
        
        # ═══════════════════════════════════════════════════════════
        # 4. Optional: Extract bridge frames
        # ═══════════════════════════════════════════════════════════
        if extract_frames:
            print("\n[4/4] Extracting bridge frame information...")
            
            bridge_frames_info = []
            
            for cluster_id, data in bridging_results.items():
                bridge_frames_array = data['bridge_frames']
                bridge_indices = np.where(bridge_frames_array)[0]
                
                if len(bridge_indices) > 0:
                    frames = self.trajectory_data[cluster_id]['frames']
                    
                    # Get indices in the recorded arrays (only bridge frames have entries)
                    bridge_count = 0
                    for idx in bridge_indices:
                        actual_frame = frames[idx]
                        
                        bridge_frames_info.append({
                            'Cluster': cluster_id,
                            'Frame_Index': int(idx),
                            'Actual_Frame': int(actual_frame),
                            'Angle': float(data['bridge_angles'][bridge_count]) if bridge_count < len(data['bridge_angles']) else None,
                            'Clay_Dist': float(data['bridge_distances_clay'][bridge_count]) if bridge_count < len(data['bridge_distances_clay']) else None,
                            'Mol_Dist': float(data['bridge_distances_mol'][bridge_count]) if bridge_count < len(data['bridge_distances_mol']) else None
                        })
                        bridge_count += 1
            
            report['bridge_frames'] = bridge_frames_info
            print(f"  ✓ Extracted {len(bridge_frames_info)} bridge frames")
            
            # Display frame information table
            if display_tables and len(bridge_frames_info) > 0:
                import pandas as pd
                df_frames = pd.DataFrame(bridge_frames_info)
                
                print("\n" + "="*80)
                print("BRIDGE FRAME DETAILS")
                print("="*80)
                print(df_frames.to_string(index=False))
                
                # Show frame numbers grouped by cluster
                print("\n" + "="*80)
                print("FRAME NUMBERS FOR VISUALIZATION:")
                print("="*80)
                for cluster_id in sorted(df_frames['Cluster'].unique()):
                    cluster_frames = df_frames[df_frames['Cluster'] == cluster_id]['Actual_Frame'].values
                    print(f"\nCluster {cluster_id}: {len(cluster_frames)} bridge frames")
                    print(f"  Frame numbers: {', '.join(map(str, cluster_frames[:10]))}" + 
                          (f" ... ({len(cluster_frames)-10} more)" if len(cluster_frames) > 10 else ""))
                print("="*80)
            
            # Save frame information to file for VMD/OVITO
            if save_frames_to_file and len(bridge_frames_info) > 0:
                import pandas as pd
                df_frames = pd.DataFrame(bridge_frames_info)
                
                with open(save_frames_to_file, 'w') as f:
                    f.write(f"# Bridge Frames ({angle_threshold}° angle threshold)\n")
                    f.write("# Format: Cluster_ID Frame_Number Angle Clay_Dist Mol_Dist\n")
                    for _, row in df_frames.iterrows():
                        angle_val = row['Angle'] if row['Angle'] is not None else 0.0
                        clay_val = row['Clay_Dist'] if row['Clay_Dist'] is not None else 0.0
                        mol_val = row['Mol_Dist'] if row['Mol_Dist'] is not None else 0.0
                        f.write(f"{row['Cluster']} {row['Actual_Frame']} {angle_val:.2f} {clay_val:.2f} {mol_val:.2f}\n")
                
                print(f"\n  ✓ Frame information saved to: {save_frames_to_file}")
        else:
            report['bridge_frames'] = None
            print("\n[4/4] Frame extraction skipped")
        
        # ═══════════════════════════════════════════════════════════
        # 5. Generate text report if requested
        # ═══════════════════════════════════════════════════════════
        if save_to_file:
            report_text.append("="*80)
            report_text.append("ION BRIDGING ANALYSIS REPORT")
            report_text.append("="*80)
            report_text.append(f"\nAnalysis Parameters:")
            report_text.append(f"  Clay selection: {clay_sel}")
            report_text.append(f"  Ion selection: {ion_sel}")
            report_text.append(f"  Molecule selection: {molecule_sel}")
            report_text.append(f"  Distance cutoffs: {cutoff_clay_ion:.2f}Å (clay-ion), {cutoff_ion_molecule:.2f}Å (ion-mol)")
            report_text.append(f"  Angle threshold: {angle_threshold}°")
            report_text.append(f"  Z-ordering: {'Enabled' if require_z_ordering else 'Disabled'}")
            
            report_text.append(f"\n{'='*80}")
            report_text.append("BRIDGING FREQUENCY BY CLUSTER")
            report_text.append("="*80)
            for cid, data in bridging_results.items():
                report_text.append(f"\nCluster {cid}:")
                report_text.append(f"  Frequency: {data['bridge_frequency']:.2f}%")
                report_text.append(f"  Events: {data['n_bridge_events']}")
                report_text.append(f"  Mean lifetime: {data['mean_bridge_lifetime']:.1f} frames")
                if data['bridge_frequency'] > 0:
                    report_text.append(f"  Mean angle: {data['mean_bridge_angle']:.1f}°")
                    report_text.append(f"  Mean distances: {data['mean_distance_clay']:.2f}Å (clay), {data['mean_distance_mol']:.2f}Å (mol)")
            
            report_text_str = "\n".join(report_text)
            with open(save_to_file, 'w') as f:
                f.write(report_text_str)
            report['report_text'] = report_text_str
            print(f"\n✓ Report saved to: {save_to_file}")
        
        # ═══════════════════════════════════════════════════════════
        # 6. Optional: Extract trajectory frames directly
        # ═══════════════════════════════════════════════════════════
        if extract_trajectory_frames:
            print("\n" + "="*80)
            print("EXTRACTING BRIDGE FRAMES TO TRAJECTORY FILES")
            print("="*80)
            
            if topology_file is None or trajectory_file is None:
                print("⚠ ERROR: extract_trajectory_frames requires topology_file and trajectory_file!")
                print("         Example: topology_file='nvt.tpr', trajectory_file='nvt.xtc'")
                report['extracted_trajectories'] = None
            elif not extract_frames:
                print("⚠ ERROR: extract_trajectory_frames requires extract_frames=True!")
                print("         Enable extract_frames=True to collect bridge frame information.")
                report['extracted_trajectories'] = None
            elif not bridge_frames_info:
                print("⚠ WARNING: No bridge frames found in the analysis!")
                print("           Check your selection criteria and cutoff parameters.")
                print(f"           Current settings: angle_threshold={angle_threshold}°, ")
                print(f"           cutoff_clay_ion={cutoff_clay_ion}Å, cutoff_ion_molecule={cutoff_ion_molecule}Å")
                report['extracted_trajectories'] = None
            else:
                # Set default output directory
                if output_dir is None:
                    output_dir = 'bridge_trajectories'
                
                # Create output directory if needed
                import os
                os.makedirs(output_dir, exist_ok=True)
                
                print(f"\n  Topology: {topology_file}")
                print(f"  Trajectory: {trajectory_file}")
                print(f"  Output directory: {output_dir}")
                
                # Load trajectory
                print(f"\n  Loading trajectory...")
                try:
                    import MDAnalysis as mda
                    u = mda.Universe(topology_file, trajectory_file)
                    print(f"  ✓ Loaded trajectory with {len(u.trajectory)} frames")
                    print(f"  ✓ System contains {len(u.atoms)} atoms")
                    
                    # Convert bridge_frames_info to DataFrame for easier manipulation
                    df_frames = pd.DataFrame(bridge_frames_info)
                    
                    extracted_files = []
                    
                    # Extract frames for each cluster
                    print(f"\n  Extracting frames by cluster...")
                    for cluster_id in sorted(df_frames['Cluster'].unique()):
                        cluster_frames = df_frames[df_frames['Cluster'] == cluster_id]['Actual_Frame'].astype(int).values
                        n_frames = len(cluster_frames)
                        
                        output_file = os.path.join(output_dir, f"bridge_frames_cluster_{int(cluster_id)}.xtc")
                        
                        print(f"\n    Cluster {int(cluster_id)}: {n_frames} frames")
                        print(f"      Frames: {cluster_frames.tolist()}")
                        print(f"      Output: {output_file}")
                        
                        # Write frames to new trajectory
                        with mda.Writer(output_file, n_atoms=len(u.atoms)) as writer:
                            for frame_idx, frame_num in enumerate(cluster_frames, 1):
                                # Go to this frame
                                u.trajectory[int(frame_num)]
                                
                                # Write this frame
                                writer.write(u.atoms)
                                
                                # Progress indicator
                                if frame_idx % 5 == 0 or frame_idx == n_frames:
                                    print(f"      Progress: {frame_idx}/{n_frames} frames", end="\r")
                        
                        print(f"\n      ✓ Saved to {output_file}")
                        extracted_files.append(output_file)
                    
                    report['extracted_trajectories'] = extracted_files
                    
                    # Summary
                    print(f"\n" + "="*80)
                    print(f"EXTRACTION COMPLETE")
                    print(f"="*80)
                    print(f"\n  Generated {len(extracted_files)} trajectory files:")
                    for filepath in extracted_files:
                        if os.path.exists(filepath):
                            file_size_mb = os.path.getsize(filepath) / (1024**2)
                            cluster_id = int(filepath.split('cluster_')[1].split('.')[0])
                            n_frames = len(df_frames[df_frames['Cluster'] == cluster_id])
                            print(f"    {os.path.basename(filepath)}: {n_frames} frames ({file_size_mb:.2f} MB)")
                    
                    print(f"\n  To visualize in VMD:")
                    print(f"    vmd {topology_file} {extracted_files[0]}")
                    print(f"\n  To visualize in OVITO:")
                    print(f"    File → Import → {extracted_files[0]}")
                    print("="*80)
                    
                except Exception as e:
                    print(f"\n⚠ ERROR during trajectory extraction: {e}")
                    import traceback
                    traceback.print_exc()
                    report['extracted_trajectories'] = None
        else:
            report['extracted_trajectories'] = None
        
        print("\n" + "="*80)
        print("COMPREHENSIVE BRIDGING REPORT COMPLETE")
        print("="*80)
        
        return report
    
    def define_spatial_binding_regions(self,
                                       rdf_data: Dict,
                                       cluster_id: Optional[int] = None,
                                       initial_boundaries: Optional[Dict] = None) -> Dict:
        """
        Interactively define spatial binding regions (P1, P2, P3...) from existing RDF data.
        
        This wraps the interactive_rdf_boundary_editor() from MolecularAnalysis and 
        stores the results for automatic use with compute_cluster_spatial_binding() 
        and plot_cluster_spatial_binding_interactive().
        
        Handles nested RDF data from batch compute_rdf() calls automatically.
        
        Parameters
        ----------
        rdf_data : dict
            RDF results from compute_rdf(). Can be:
            - Batch format: {'selection1': {'selection2': {cluster_data}}}
            - Single format: {'selection-pair': {cluster_data}}
            - Flat format: {'label': {'r': [...], 'gr': [...]}}
        cluster_id : int, optional
            If RDF data contains multiple clusters, which cluster to use for defining regions.
            If None, will use cluster 0. You can also specify 'average' to average across clusters.
        initial_boundaries : dict, optional
            Initial peak boundaries to start with or load from file
            Format: {ion_label: {'P1': (start, end), 'P2': (start, end), ...}}
            
        Returns
        -------
        boundaries : dict
            Dictionary with spatial binding regions: {ion_label: {'P1': (start, end), ...}}
            Also stored as self.spatial_binding_boundaries for automatic use
            
        Examples
        --------
        >>> # You already computed batch RDF
        >>> rdf_Na = analyzer.compute_rdf(
        ...     ['quinolone', 'piperazine', 'carboxylic_acid'],
        ...     'Na',
        ...     cluster_ids='all',
        ...     rmax=8.0
        ... )
        >>> 
        >>> # Define regions from cluster 0 data
        >>> boundaries = analyzer.define_spatial_binding_regions(
        ...     rdf_Na,
        ...     cluster_id=0
        ... )
        >>> 
        >>> # In the interactive editor:
        >>> # Available labels: ['quinolone', 'piperazine', 'carboxylic_acid']
        >>> # select quinolone
        >>> # plot
        >>> # add P1 2.0 3.5
        >>> # add P2 3.5 5.5
        >>> # add P3 5.5 8.0
        >>> # save boundaries_Na.json  # Optional
        >>> # quit
        >>> 
        >>> # Now spatial binding automatically uses these regions
        >>> spatial_data = analyzer.compute_cluster_spatial_binding(
        ...     cluster_id=0,
        ...     ion_type='Na',
        ...     target_sel='resname api',
        ...     cutoff=3.5
        ... )
        >>> 
        >>> # Visualize with regions (automatically detected)
        >>> plotter.plot_cluster_spatial_binding_interactive(
        ...     spatial_data,
        ...     plot_regions=['P1', 'P2', 'P3']
        ... )
        
        Notes
        -----
        - Use 'P1', 'P2', 'P3' naming for consistency (not 'shell_1', 'shell_2')
        - Boundaries are stored in self.spatial_binding_boundaries
        - Boundaries are automatically included in compute_cluster_spatial_binding() results
        - Plotter automatically detects boundaries from spatial_results
        - Use 'save' command in editor to save boundaries for future sessions
        - Use 'load' command in editor to load previously saved boundaries
        """
        from MolecularAnalysis import MolecularAnalysis
        
        # Get or create MolecularAnalysis instance for the editor
        if hasattr(self, 'analyzer') and self.analyzer is not None:
            temp_analysis = self.analyzer
        else:
            # Need an instance just for the editor methods
            if not hasattr(self, 'trajectory_data') or len(self.trajectory_data) == 0:
                raise ValueError("No trajectory data loaded. Run clustering analysis first.")
            temp_analysis = MolecularAnalysis(
                top=self.topology_file,
                traj=self.trajectory_file
            )
        
        # Flatten and extract RDF data for the editor
        # Handle nested structure from batch compute_rdf
        rdf_dict_flat = {}
        
        for key1, value1 in rdf_data.items():
            # Check if this is batch format (nested dict)
            if isinstance(value1, dict):
                # Check if it's cluster data or another nest level
                if 'cluster_0' in value1 or 0 in value1:
                    # Direct cluster data: {'cluster_0': {...}, 'cluster_1': {...}}
                    # Extract specified cluster
                    cluster_key = f'cluster_{cluster_id}' if cluster_id is not None else 'cluster_0'
                    if cluster_key in value1:
                        rdf_dict_flat[key1] = value1[cluster_key]
                    elif cluster_id in value1:  # Try integer key
                        rdf_dict_flat[key1] = value1[cluster_id]
                    else:
                        print(f"⚠️  Cluster {cluster_id} not found for {key1}, using cluster 0")
                        rdf_dict_flat[key1] = value1.get('cluster_0', value1.get(0, value1))
                else:
                    # Another level (batch format): {'Na': {cluster_data}}
                    for key2, value2 in value1.items():
                        if isinstance(value2, dict) and ('cluster_0' in value2 or 0 in value2):
                            # Extract cluster data
                            cluster_key = f'cluster_{cluster_id}' if cluster_id is not None else 'cluster_0'
                            if cluster_key in value2:
                                label = key1  # Use first level key as label
                                rdf_dict_flat[label] = value2[cluster_key]
                            elif cluster_id in value2:
                                label = key1
                                rdf_dict_flat[label] = value2[cluster_id]
                            else:
                                label = key1
                                rdf_dict_flat[label] = value2.get('cluster_0', value2.get(0, value2))
                        else:
                            # Already flat data
                            rdf_dict_flat[f"{key1}"] = value2
            else:
                # Already flat, keep as is
                rdf_dict_flat[key1] = value1
        
        if not rdf_dict_flat:
            raise ValueError(
                "Could not extract RDF data from input. "
                "Expected structure: batch RDF output from compute_rdf()"
            )
        
        # Normalize field names for the interactive editor
        # The editor expects 'bins' and 'rdf', but data might have 'r' and 'gr'
        for label, rdf_data_item in rdf_dict_flat.items():
            if isinstance(rdf_data_item, dict):
                # Check and convert field names
                if 'r' in rdf_data_item and 'bins' not in rdf_data_item:
                    rdf_data_item['bins'] = rdf_data_item['r']
                if 'gr' in rdf_data_item and 'rdf' not in rdf_data_item:
                    rdf_data_item['rdf'] = rdf_data_item['gr']
                
                # Ensure required fields exist
                if 'bins' not in rdf_data_item or 'rdf' not in rdf_data_item:
                    print(f"⚠️  Warning: {label} missing required fields. Available: {list(rdf_data_item.keys())}")
        
        print(f"\n{'='*70}")
        print(f"SPATIAL BINDING REGION EDITOR")
        print(f"{'='*70}")
        print(f"Extracted RDF curves: {list(rdf_dict_flat.keys())}")
        if cluster_id is not None:
            print(f"Using data from: Cluster {cluster_id}")
        else:
            print(f"Using data from: Cluster 0 (default)")
        print()
        print(f"Quick start commands:")
        first_label = list(rdf_dict_flat.keys())[0]
        print(f"  select {first_label}")
        print(f"  plot                    # View the RDF curve")
        print(f"  add P1 2.0 3.5          # Define first peak region")
        print(f"  add P2 3.5 5.5          # Define second peak region")
        print(f"  add P3 5.5 8.0          # Define third peak region")
        print(f"  replot                  # Check boundaries")
        print(f"  save boundaries_{first_label}.json  # Optional: save for later")
        print(f"  quit                    # Exit editor")
        print()
        
        # Open interactive boundary editor
        boundaries = temp_analysis.interactive_rdf_boundary_editor(
            rdf_dict_flat,
            initial_boundaries=initial_boundaries or {}
        )
        
        # Store boundaries for automatic use
        self.spatial_binding_boundaries = boundaries
        
        # Print summary
        defined_regions = {k: v for k, v in boundaries.items() if len(v) > 0}
        
        if defined_regions:
            print(f"\n{'='*70}")
            print(f"✓ SPATIAL BINDING REGIONS DEFINED")
            print(f"{'='*70}")
            for ion_label, regions in defined_regions.items():
                print(f"\n{ion_label}:")
                for region_name, (start, end) in sorted(regions.items()):
                    print(f"  {region_name}: {start:.2f} - {end:.2f} Å")
            print()
            print(f"✓ Boundaries stored in analyzer.spatial_binding_boundaries")
            print(f"✓ Will be automatically used in spatial binding analysis & visualization")
            print(f"{'='*70}\n")
        else:
            print(f"\n⚠️  No regions defined. Run this method again to define regions.")
        
        return boundaries
    
    def compute_cluster_spatial_binding(self, 
                                       cluster_id: int,
                                       target_sel: Union[str, List[str]],
                                       ion_type: Optional[str] = None,
                                       solvation: Optional[str] = None,
                                       cutoff: float = 3.5,
                                       step: int = 1,
                                       method: str = 'both',
                                       angular_bins: Tuple[int, int] = (18, 36),
                                       return_positions: bool = True,
                                       molecular_frame_tracking: bool = True,
                                       reference_atoms: str = 'auto',
                                       reference_target: Optional[str] = None,
                                       **kwargs) -> Dict:
        """
        Compute spatial binding analysis for specific cluster frames only.
        
        This method filters the trajectory to only the frames belonging to the 
        specified cluster, then runs spatial_binding_analysis() from MolecularAnalysis.
        Results can be visualized with plot_cluster_spatial_binding_interactive().
        
        Parameters
        ----------
        cluster_id : int
            Which cluster to analyze (0, 1, 2, ...)
        target_sel : str, list of str, or dict
            Target molecule selection
            - str: Single target (e.g., 'resname api')
            - list: Multiple targets with auto-generated labels (e.g., [sel1, sel2, sel3])
            - dict: Multiple targets with custom labels (e.g., {'quinolone': sel1, 'piperazine': sel2})
        ion_type : str, optional
            Ion to analyze (e.g., 'K', 'NA', 'CL')
            Exactly one of ion_type or solvation must be provided
        solvation : str, optional  
            Solvent to analyze (e.g., 'Ow' for water oxygen)
            Exactly one of ion_type or solvation must be provided
        cutoff : float, default=3.5
            Distance cutoff in Angstroms
        step : int, default=1
            Frame step for analysis within cluster frames
        method : str, default='both'
            'per-atom', 'spherical', or 'both'
        angular_bins : tuple, default=(18, 36)
            (n_theta, n_phi) bins for spherical mapping
        return_positions : bool, default=True
            Must be True for 3D visualization
        molecular_frame_tracking : bool, default=True
            Enable rotation-aware triangulation for accurate reconstruction
        reference_atoms : str or list, default='auto'
            Method for selecting reference atoms: 'auto', 'heavy', or list of indices
        reference_target : str, optional
            Specific atom selection for reference frame (e.g., 'resname QUI')
        **kwargs
            Additional arguments for spatial_binding_analysis():
            - rdf_boundaries : dict
                Shell boundaries from define_spatial_binding_regions()
                Auto-injected if available from analyzer.spatial_binding_boundaries
            - peaks : dict
                Which peaks to analyze per selection, e.g.:
                {'quinolone': ['P1', 'P2'], 'carboxylic_acid': ['P1', 'P2', 'P3']}
                Auto-generated if boundaries are available
            - Other spatial_binding_analysis() parameters
        
        Returns
        -------
        spatial_results : dict
            Results from spatial_binding_analysis():
            - If target_sel is str: Single results dict with ion binding data
            - If target_sel is list: {auto_label: results} dict  
            - If target_sel is dict: {custom_label: results} dict
            
            All results include:
            - Ion binding positions and densities
            - Triangulation data for 3D reconstruction
            - cluster_metadata (cluster_id, frame info, representative_frame)
            - spatial_binding_boundaries (if defined)
        
        Examples
        --------
        >>> # Single target
        >>> spatial = analyzer.compute_cluster_spatial_binding(
        ...     cluster_id=0,
        ...     target_sel='resname QUI',
        ...     ion_type='Na'
        ... )
        >>> 
        >>> # Multiple targets with custom labels (RECOMMENDED)
        >>> quinolone_sel = 'resname QUI'
        >>> piperazine_sel = 'resname PIP'
        >>> carboxylic_sel = 'resname CAR'
        >>> 
        >>> spatial_multi = analyzer.compute_cluster_spatial_binding(
        ...     cluster_id=0,
        ...     target_sel={
        ...         'quinolone': quinolone_sel,
        ...         'piperazine': piperazine_sel, 
        ...         'carboxylic_acid': carboxylic_sel
        ...     },
        ...     ion_type='Na',
        ...     cutoff=3.5,
        ...     peaks={
        ...         'quinolone': ['P1', 'P2'],
        ...         'carboxylic_acid': ['P1', 'P2', 'P3'],
        ...         'piperazine': ['P1', 'P2']
        ...     }
        ... )
        >>> 
        >>> # Access individual results
        >>> quinolone_results = spatial_multi['quinolone']
        >>> 
        >>> # Multiple targets with auto-generated labels (list)
        >>> spatial_auto = analyzer.compute_cluster_spatial_binding(
        ...     cluster_id=0,
        ...     target_sel=['resname QUI', 'resname PIP', 'resname CAR'],
        ...     ion_type='Na',
        ...     cutoff=3.5
        ... )
        >>> # Labels will be auto-extracted: ['QUI', 'PIP', 'CAR']
        
        Notes
        -----
        - If spatial_binding_boundaries are stored (from define_spatial_binding_regions()),
          they are automatically used unless explicitly overridden
        - For multi-target analysis, peaks dict is auto-generated for all matching boundary keys
        - Dict input (with custom labels) is recommended over list input (auto-generated labels)
        - Representative frame is the first frame of the cluster
        - Cluster-specific temporary trajectory is created and cleaned up automatically
        - Requires trajectory to be loaded via load_cluster_trajectories()
        - Only analyzes frames assigned to the specified cluster
        - Results include metadata about which cluster was analyzed
        - Use return_positions=True to enable 3D visualization
        """
        # Validate prerequisites
        if not hasattr(self, 'trajectory_data'):
            raise ValueError(
                "Trajectory data not loaded. Call load_cluster_trajectories() first."
            )
        
        if self.cluster_labels is None or self.cluster_centers is None:
            raise ValueError(
                "No cluster assignments found. Run find_clusters() first."
            )
        
        if cluster_id not in self.trajectory_data:
            available = sorted(self.trajectory_data.keys())
            raise ValueError(
                f"Cluster {cluster_id} not in loaded trajectory data. "
                f"Available clusters: {available}"
            )
        
        # Import MolecularAnalysis
        try:
            from MolecularAnalysis import MolecularAnalysis
        except ImportError:
            raise ImportError(
                "MolecularAnalysis module not found. Ensure it's in your Python path."
            )
        
        print(f"\n{'='*60}")
        print(f"Spatial Binding Analysis for Cluster {cluster_id}")
        print(f"{'='*60}")
        
        # Get cluster frame information
        cluster_frames = self.get_cluster_frames(cluster_id=cluster_id)
        n_frames = len(cluster_frames)
        
        print(f"Cluster info:")
        print(f"  Cluster ID: {cluster_id}")
        print(f"  Number of frames: {n_frames}")
        print(f"  Frame step: {step}")
        print(f"  Effective frames analyzed: {len(cluster_frames[::step])}")
        
        # Get the universe and frame information
        print(f"\nPreparing cluster-specific trajectory...")
        u = self.trajectory_data[cluster_id]['universe']
        
        # Create temporary trajectory file with cluster frames only
        import tempfile
        import MDAnalysis as mda
        
        with tempfile.NamedTemporaryFile(suffix='.xtc', delete=False) as tmp_traj:
            tmp_traj_path = tmp_traj.name
        
        try:
            # Write cluster frames to temporary trajectory
            print(f"  Writing {len(cluster_frames[::step])} frames to temporary file...")
            print(f"  Frame indices being written: {cluster_frames[::step][:5]}... (first 5)")
            
            # Track some atom positions to verify frames are different
            first_atom_positions = []
            
            with mda.Writer(tmp_traj_path, u.atoms.n_atoms) as writer:
                for i, frame_idx in enumerate(cluster_frames[::step]):
                    u.trajectory[frame_idx]
                    writer.write(u.atoms)
                    # Store first atom position for first few frames
                    if i < 3:
                        first_atom_positions.append(u.atoms[0].position.copy())
            
            print(f"  ✓ Temporary cluster trajectory created")
            print(f"  Verification - First atom positions in written frames:")
            for i, pos in enumerate(first_atom_positions):
                print(f"    Frame {i}: {pos[:3]}")  # First 3 coordinates
            
            # Verify by reading back the temporary trajectory
            print(f"  Verifying temporary trajectory...")
            u_temp = mda.Universe(self.topology_file, tmp_traj_path)
            print(f"    Temporary trajectory has {len(u_temp.trajectory)} frames")
            print(f"    First atom position in temp traj frame 0: {u_temp.atoms[0].position[:3]}")
            u_temp.trajectory[0]  # Reset to first frame
            
            # Create MolecularAnalysis instance with cluster-only trajectory
            print(f"  Initializing MolecularAnalysis...")
            analysis = MolecularAnalysis(top=self.topology_file, traj=tmp_traj_path)
            
            # Prepare kwargs with boundaries if available
            analysis_kwargs = kwargs.copy()
            
            # Auto-inject spatial binding boundaries if defined and not explicitly provided
            if hasattr(self, 'spatial_binding_boundaries') and self.spatial_binding_boundaries:
                if 'rdf_boundaries' not in analysis_kwargs:
                    analysis_kwargs['rdf_boundaries'] = self.spatial_binding_boundaries
                    print(f"  ✓ Using stored spatial binding boundaries")
                    
                    # Auto-generate peaks dict if not provided
                    if 'peaks' not in analysis_kwargs:
                        # Match boundary keys to target selections
                        target_list = [target_sel] if isinstance(target_sel, str) else target_sel
                        auto_peaks = {}
                        
                        for boundary_key in self.spatial_binding_boundaries.keys():
                            # Check if this boundary key matches any target
                            for target in target_list:
                                # Match if boundary key is in target name or vice versa
                                if (boundary_key.lower() in target.lower() or 
                                    target.lower() in boundary_key.lower() or
                                    boundary_key == target):
                                    regions = list(self.spatial_binding_boundaries[boundary_key].keys())
                                    auto_peaks[boundary_key] = regions
                                    break
                        
                        if auto_peaks:
                            analysis_kwargs['peaks'] = auto_peaks
                            print(f"  ✓ Auto-generated peaks dict:")
                            for key, regions in auto_peaks.items():
                                print(f"    {key}: {regions}")
            
            # Run spatial binding analysis on cluster-only trajectory
            print(f"\nRunning spatial binding analysis...")
            spatial_results = analysis.spatial_binding_analysis(
                target_sel=target_sel,
                ion_type=ion_type,
                solvation=solvation,
                cutoff=cutoff,
                step=1,  # Analyze all frames in cluster trajectory
                method=method,
                angular_bins=angular_bins,
                return_positions=return_positions,
                molecular_frame_tracking=molecular_frame_tracking,
                reference_atoms=reference_atoms,
                reference_target=reference_target,
                force_rerun=True, 
                **analysis_kwargs
            )
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(tmp_traj_path):
                os.remove(tmp_traj_path)
                print(f"  ✓ Temporary file cleaned up")
        
        # Prepare cluster metadata
        cluster_metadata = {
            'cluster_id': cluster_id,
            'cluster_frames': cluster_frames,
            'n_cluster_frames': n_frames,
            'step_used': step,
            'cluster_center': self.cluster_centers[cluster_id].tolist(),
            'representative_frame': cluster_frames[0]  # First frame of cluster for structure
        }
        
        # Check if multi-target results (dict) or single-target (single dict)
        # Multi-target: spatial_results is {'target1': {...}, 'target2': {...}}
        # Single-target: spatial_results is {'contact_frequency': [...], ...}
        is_multi_target = isinstance(target_sel, (list, dict))
        
        # Use the boundaries that were actually passed to the analysis (with ion suffix)
        boundaries_to_store = analysis_kwargs.get('rdf_boundaries', None)
        if boundaries_to_store is None and hasattr(self, 'spatial_binding_boundaries'):
            boundaries_to_store = self.spatial_binding_boundaries
        
        if is_multi_target:
            # Add cluster metadata to each individual target result
            for target_key in spatial_results.keys():
                if isinstance(spatial_results[target_key], dict):
                    spatial_results[target_key]['cluster_metadata'] = cluster_metadata
                    
                    # Add spatial binding boundaries if defined
                    if boundaries_to_store:
                        spatial_results[target_key]['spatial_binding_boundaries'] = boundaries_to_store
            
            # Also add to top level for backward compatibility
            spatial_results['cluster_metadata'] = cluster_metadata
            if boundaries_to_store:
                spatial_results['spatial_binding_boundaries'] = boundaries_to_store
        else:
            # Single target - add directly
            spatial_results['cluster_metadata'] = cluster_metadata
            
            # Add spatial binding boundaries if defined
            if boundaries_to_store:
                spatial_results['spatial_binding_boundaries'] = boundaries_to_store
        
        # Print summary
        print(f"\n✓ Spatial binding analysis complete for Cluster {cluster_id}")
        if is_multi_target:
            print(f"  Targets analyzed: {[k for k in spatial_results.keys() if k not in ['cluster_metadata', 'spatial_binding_boundaries']]}")
            total_contacts_list = [spatial_results[k].get('total_contacts', 0) for k in spatial_results.keys() 
                                  if k not in ['cluster_metadata', 'spatial_binding_boundaries'] and isinstance(spatial_results[k], dict)]
            if total_contacts_list:
                print(f"  Total contacts: {sum(total_contacts_list)} (sum across targets)")
        else:
            print(f"  Total contacts: {spatial_results.get('total_contacts', 'N/A')}")
        print(f"  Representative frame: {cluster_frames[0]} (first frame of cluster)")
        
        # Print available regions if boundaries defined
        if hasattr(self, 'spatial_binding_boundaries') and self.spatial_binding_boundaries:
            if ion_type and ion_type in self.spatial_binding_boundaries:
                region_names = list(self.spatial_binding_boundaries[ion_type].keys())
                if region_names:
                    print(f"  Available spatial regions: {', '.join(region_names)}")
        
        return spatial_results


# Legacy function for backward compatibility
def density_scatter(X, Y, nx=200, ny=200, smoothing=20, target_max=50, 
                   marker='s', msize=10, cmap='viridis', out_path=None):
    """
    Create density scatter plot where points are colored by local density.
    
    Parameters:
    -----------
    X, Y : array-like
        Data coordinates
    nx, ny : int
        Number of bins in x and y directions
    smoothing : float
        Smoothing parameter (higher = smoother)
    target_max : float
        Maximum count for colorbar scaling
    marker : str
        Marker style
    msize : float
        Marker size
    cmap : str
        Colormap name
    out_path : str or Path
        Output file path for saving plot
    
    Returns:
    --------
    density_map : ndarray
        2D density histogram
    bin_centers_x, bin_centers_y : ndarray
        Bin center coordinates
    """
    # Create bin edges
    edges_x = np.linspace(X.min(), X.max(), nx + 1)
    ctrs_x = edges_x[:-1] + np.diff(edges_x)[0] / 2
    
    edges_y = np.linspace(Y.min(), Y.max(), ny + 1)
    ctrs_y = edges_y[:-1] + np.diff(edges_y)[0] / 2
    
    # Bin the data
    binx = np.digitize(X, edges_x) - 1
    biny = np.digitize(Y, edges_y) - 1
    
    # Keep only valid bins
    valid = (binx >= 0) & (binx < nx) & (biny >= 0) & (biny < ny)
    
    # Count points in each bin
    H = np.zeros((ny, nx))
    for i in range(len(X)):
        if valid[i]:
            H[biny[i], binx[i]] += 1
    
    # Smooth counts - need to use the class methods
    analyzer = RMSDClusterAnalyzer()
    F = analyzer._smooth_2d_counts(H, smoothing)
    
    # Scale to target max
    current_max = F.max()
    if current_max > 0:
        F = F * (target_max / current_max)
    
    # Get density for each point
    colors = np.zeros(len(X))
    for i in range(len(X)):
        if valid[i]:
            colors[i] = F[biny[i], binx[i]]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(X[valid], Y[valid], s=msize, c=colors[valid], 
                        marker=marker, cmap=cmap, vmin=0, vmax=target_max,
                        alpha=0.8, edgecolors='none')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'Counts per bin ({nx}×{ny} grid)', fontsize=11)
    
    # Set integer ticks on colorbar
    n_ticks = min(6, int(target_max) + 1)
    ticks = np.linspace(0, target_max, n_ticks)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{int(t)}' for t in ticks])
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Density Scatter Plot', fontsize=13)
    ax.grid(alpha=0.3, ls='--')
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")
    
    return F, ctrs_x, ctrs_y, fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Density scatter plot for RMSD clustering visualization"
    )
    parser.add_argument("--flat", required=True, help="Path to rmsd_flat_4q.xvg")
    parser.add_argument("--cross", required=True, help="Path to rmsd_cross_4q.xvg")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--bins", type=int, nargs=2, default=[200, 200],
                       help="Number of bins [nx ny] (default: 200 200)")
    parser.add_argument("--smoothing", type=float, default=20,
                       help="Smoothing parameter (default: 20)")
    parser.add_argument("--target-max", type=float, default=50,
                       help="Maximum count for colorbar (default: 50)")
    parser.add_argument("--marker", default='s', help="Marker style (default: square)")
    parser.add_argument("--msize", type=float, default=10, help="Marker size (default: 10)")
    parser.add_argument("--cmap", default='viridis', 
                       help="Colormap (default: viridis; try: plasma, hot, turbo)")
    
def main():
    """Command-line interface for density-based RMSD clustering."""
    parser = argparse.ArgumentParser(
        description="Density scatter plot for RMSD clustering visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s --flat rmsd_flat.xvg --cross rmsd_cross.xvg --out results/
  
  # High-resolution with more smoothing
  %(prog)s --flat flat.xvg --cross cross.xvg --out out/ --bins 300 300 --smoothing 30
  
  # Find clusters automatically
  %(prog)s --flat flat.xvg --cross cross.xvg --out out/ --find-clusters
        """
    )
    parser.add_argument("--flat", required=True, help="Path to rmsd_flat_4q.xvg")
    parser.add_argument("--cross", required=True, help="Path to rmsd_cross_4q.xvg")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--bins", type=int, nargs=2, default=[200, 200],
                       help="Number of bins [nx ny] (default: 200 200)")
    parser.add_argument("--smoothing", type=float, default=20,
                       help="Smoothing parameter (default: 20)")
    parser.add_argument("--target-max", type=float, default=50,
                       help="Maximum count for colorbar (default: 50)")
    parser.add_argument("--marker", default='s', help="Marker style (default: square)")
    parser.add_argument("--msize", type=float, default=10, help="Marker size (default: 10)")
    parser.add_argument("--cmap", default='viridis', 
                       help="Colormap (default: viridis; try: plasma, hot, turbo)")
    parser.add_argument("--find-clusters", action='store_true',
                       help="Automatically find and assign clusters")
    parser.add_argument("--cluster-threshold", type=float, default=0.3,
                       help="Cluster detection threshold (default: 0.3)")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    print("=" * 60)
    print("RMSD Density-Based Clustering Analysis")
    print("=" * 60)
    
    analyzer = RMSDClusterAnalyzer()
    
    # Load data
    print("\n1. Loading RMSD data...")
    analyzer.load_rmsd_data(args.flat, args.cross)
    
    # Compute density
    print(f"\n2. Computing density map...")
    analyzer.compute_density(
        bins=tuple(args.bins),
        smoothing=args.smoothing,
        target_max=args.target_max
    )
    
    # Create plots
    print(f"\n3. Creating visualizations...")
    analyzer.plot_density_scatter(
        msize=args.msize,
        marker=args.marker,
        cmap=args.cmap,
        save_path=out_dir / "density_scatter.png"
    )
    
    analyzer.plot_density_heatmap(
        cmap=args.cmap,
        save_path=out_dir / "density_heatmap.png"
    )
    
    # Find clusters if requested
    if args.find_clusters:
        print(f"\n4. Finding clusters...")
        n_clusters = analyzer.find_clusters(
            method='peaks',
            threshold=args.cluster_threshold
        )
        
        # Plot clusters
        analyzer.plot_clusters(save_path=out_dir / "clusters_assigned.png")
        
        # Save cluster data
        analyzer.save_cluster_frames(out_dir)
        
        print(f"\n✓ Found and saved {n_clusters} clusters")
    else:
        # Save basic summary
        with open(out_dir / "density_summary.txt", "w") as f:
            f.write("Density Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Grid size: {args.bins[0]}×{args.bins[1]}\n")
            f.write(f"Smoothing: {args.smoothing}\n")
            f.write(f"Total frames: {len(analyzer.rmsd_flat)}\n\n")
            f.write(f"Density statistics:\n")
            f.write(f"  Max density: {analyzer.density_map.max():.2f} counts/bin\n")
            f.write(f"  Mean density (non-zero): ")
            f.write(f"{analyzer.density_map[analyzer.density_map > 0].mean():.2f} counts/bin\n")
            f.write(f"  Non-empty bins: {np.sum(analyzer.density_map > 0)}\n")
            f.write(f"\nTo find clusters, re-run with --find-clusters\n")
    
    print(f"\n" + "=" * 60)
    print(f"✓ All outputs saved to {out_dir}/")
    print(f"  - density_scatter.png")
    print(f"  - density_heatmap.png")
    if args.find_clusters:
        print(f"  - clusters_assigned.png")
        print(f"  - cluster_*_frames.txt")
        print(f"  - cluster_*_times_ns.txt")
        print(f"  - cluster_summary.txt")
    else:
        print(f"  - density_summary.txt")
    print("=" * 60)


def analyze_multiple_reference_pairs(rmsd_files: Dict[str, str],
                                     pairs: List[Tuple[str, str]],
                                     base_output_dir: Union[str, Path],
                                     **kwargs) -> Dict[str, RMSDClusterAnalyzer]:
    """
    Run density-based clustering analysis for multiple RMSD reference pairs.
    
    This convenience function allows batch analysis of multiple reference structure
    combinations (e.g., flat vs cross, flat vs side, side vs cross) without
    duplicating code. Each pair gets its own analyzer and output directory.
    
    Parameters
    ----------
    rmsd_files : dict
        Dictionary mapping reference labels to XVG file paths.
        Example: {'flat': 'rmsd_flat_4q.xvg', 'side': 'rmsd_side_4q.xvg', 
                  'cross': 'rmsd_cross_4q.xvg'}
    pairs : list of tuples
        List of (label1, label2) tuples defining which pairs to analyze.
        Example: [('flat', 'cross'), ('flat', 'side'), ('side', 'cross')]
    base_output_dir : str or Path
        Base directory for outputs. Each pair gets a subdirectory.
    **kwargs : dict
        Optional parameters passed to compute_density() and find_clusters().
        Common options: bins=(200, 200), smoothing=20, threshold=0.3, 
                       peak_size=20, density_threshold=0.15, density_ratio=0.5, 
                       plot=True
    
    Returns
    -------
    analyzers : dict
        Dictionary mapping pair labels to RMSDClusterAnalyzer instances.
        Keys are formatted as "label1_vs_label2".
    
    Example
    -------
    >>> # Define available RMSD files
    >>> rmsd_files = {
    ...     'flat': 'rmsd_flat_4q.xvg',
    ...     'side': 'rmsd_side_4q.xvg',
    ...     'cross': 'rmsd_cross_4q.xvg'
    ... }
    >>> 
    >>> # Analyze all three combinations
    >>> pairs = [('flat', 'cross'), ('flat', 'side'), ('side', 'cross')]
    >>> analyzers = analyze_multiple_reference_pairs(
    ...     rmsd_files=rmsd_files,
    ...     pairs=pairs,
    ...     base_output_dir='analysis_results',
    ...     bins=(200, 200),
    ...     smoothing=20,
    ...     threshold=0.3,
    ...     peak_size=20
    ... )
    >>> 
    >>> # Access individual analyzers
    >>> flat_cross = analyzers['flat_vs_cross']
    >>> flat_side = analyzers['flat_vs_side']
    >>> side_cross = analyzers['side_vs_cross']
    """
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    analyzers = {}
    
    # Extract parameters for density computation and clustering
    bins = kwargs.get('bins', (200, 200))
    smoothing = kwargs.get('smoothing', 20)
    threshold = kwargs.get('threshold', 0.3)
    peak_size = kwargs.get('peak_size', 20)
    density_threshold = kwargs.get('density_threshold', 0.15)
    density_ratio = kwargs.get('density_ratio', 0.5)
    plot = kwargs.get('plot', True)
    
    print("="*70)
    print("BATCH ANALYSIS: Multiple RMSD Reference Pairs")
    print("="*70)
    
    for label1, label2 in pairs:
        pair_name = f"{label1}_vs_{label2}"
        print(f"\n{'='*70}")
        print(f"Analyzing: {label1.upper()} vs {label2.upper()}")
        print(f"{'='*70}")
        
        # Check if files exist
        if label1 not in rmsd_files:
            print(f"⚠ Warning: No file provided for '{label1}', skipping...")
            continue
        if label2 not in rmsd_files:
            print(f"⚠ Warning: No file provided for '{label2}', skipping...")
            continue
        
        # Create output directory for this pair
        pair_dir = base_path / pair_name
        pair_dir.mkdir(exist_ok=True)
        
        # Initialize analyzer
        analyzer = RMSDClusterAnalyzer()
        
        # Load data
        print(f"\n1. Loading RMSD data...")
        analyzer.load_rmsd_data(
            rmsd_files[label1], 
            rmsd_files[label2],
            label_x=label1,
            label_y=label2
        )
        
        # Compute density
        print(f"\n2. Computing density map...")
        analyzer.compute_density(bins=bins, smoothing=smoothing)
        
        # Find clusters
        print(f"\n3. Finding clusters...")
        n_clusters = analyzer.find_clusters(
            method='peaks',
            threshold=threshold,
            peak_size=peak_size,
            density_threshold=density_threshold,
            density_ratio=density_ratio
        )
        print(f"   Found {n_clusters} clusters")
        
        # Save cluster data
        print(f"\n4. Saving cluster data...")
        analyzer.save_cluster_frames(pair_dir)
        
        # Generate plots if requested
        if plot:
            print(f"\n5. Generating plots...")
            try:
                fig_scatter = analyzer.plot_density_scatter(figsize=(10, 8))
                fig_scatter.savefig(pair_dir / f"{pair_name}_density_scatter.png", 
                                   dpi=300, bbox_inches='tight')
                plt.close(fig_scatter)
                
                fig_heatmap = analyzer.plot_density_heatmap(figsize=(10, 8))
                fig_heatmap.savefig(pair_dir / f"{pair_name}_density_heatmap.png",
                                   dpi=300, bbox_inches='tight')
                plt.close(fig_heatmap)
                
                print(f"   ✓ Plots saved to {pair_dir}/")
            except Exception as e:
                print(f"   ⚠ Plot generation warning: {e}")
        
        # Store analyzer
        analyzers[pair_name] = analyzer
        
        print(f"\n✓ Completed analysis for {pair_name}")
        print(f"  Output directory: {pair_dir}/")
    
    print(f"\n{'='*70}")
    print(f"✓ BATCH ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Analyzed {len(analyzers)} reference pairs:")
    for pair_name in analyzers.keys():
        print(f"  - {pair_name}")
    print(f"\nAll results saved to: {base_path}/")
    
    return analyzers


if __name__ == "__main__":
    main()

