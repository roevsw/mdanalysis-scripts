"""
Clay-Organic-Ion-Water (CIP) Interaction Analysis

This module provides comprehensive analysis tools for studying interactions between
clay surfaces, organic molecules, ions, and water in molecular dynamics simulations.

Author: Swai
Date: December 2025
"""

import hashlib
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx

import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.density import DensityAnalysis

from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')


# Module-level class for RDF results (must be picklable for multiprocessing)
class _RDFResult:
    """Simple container for RDF results that can be pickled."""
    def __init__(self, bins, rdf, count, edges):
        self.bins = bins
        self.rdf = rdf
        self.count = count
        self.edges = edges


def _calculate_rdf_worker(top_file, traj_file, g1_sel_str, g2_sel_str, 
                         bin_width, range_val, step, center_method, normalize,
                         save_cache, cache_dir, force_rerun, label):
    """
    Worker function for parallel RDF calculation.
    
    This function is executed in a separate process and must be at module level
    to be picklable by multiprocessing.
    
    Parameters
    ----------
    top_file : str
        Path to topology file
    traj_file : str
        Path to trajectory file
    g1_sel_str : str
        Selection string for group 1
    g2_sel_str : str
        Selection string for group 2
    bin_width : float
        Width of each bin in Å
    range_val : tuple
        (min, max) distance range in Å
    step : int
        Frame step size
    center_method : str
        'COM' or 'COG' for center calculation
    normalize : bool
        Whether to normalize RDF
    save_cache : bool
        Whether to save cache file
    cache_dir : str
        Directory for cache files
    force_rerun : bool
        Whether to force recalculation
    label : str
        Label for this RDF (e.g., 'Na-O_water')
        
    Returns
    -------
    tuple
        (label, rdf_result) where rdf_result contains bins, rdf, count, edges
    """
    # Recreate universe in this worker process
    u = mda.Universe(top_file, traj_file)
    
    # Create atom groups from selection strings
    g1 = u.select_atoms(g1_sel_str)
    g2 = u.select_atoms(g2_sel_str)
    
    # Generate cache filename
    cache_params = f"{g1_sel_str}_{g2_sel_str}_{bin_width}_{range_val}_{step}_{center_method}"
    cache_hash = hashlib.md5(cache_params.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"rdf_cache_{cache_hash}.npz")
    
    # Check cache first (unless force_rerun)
    if not force_rerun and os.path.exists(cache_file):
        try:
            cached_data = np.load(cache_file)
            # Create result object mimicking InterRDF output
            result = _RDFResult(
                bins=cached_data['bins'],
                rdf=cached_data['rdf'],
                count=cached_data['count'],
                edges=cached_data['edges']
            )
            return (label, result)
        except Exception:
            pass  # Cache load failed, recalculate
    
    # Convert boolean normalize to InterRDF norm parameter
    norm_value = 'rdf' if normalize else 'none'
    
    # Calculate RDF using InterRDF
    # Note: InterRDF always uses atomistic distances, center_method is handled in main method
    rdf_analysis = InterRDF(g1, g2, nbins=int((range_val[1] - range_val[0]) / bin_width),
                           range=range_val, norm=norm_value)
    rdf_analysis.run(step=step)
    
    # Save cache if requested
    if save_cache:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache_file,
                    bins=rdf_analysis.results.bins,
                    rdf=rdf_analysis.results.rdf,
                    count=rdf_analysis.results.count,
                    edges=rdf_analysis.results.edges)
        except Exception:
            pass  # Cache save failed, not critical
    
    return (label, rdf_analysis.results)


def _calculate_edl_density_worker(top_file, traj_file, frame_range,
                                  ion_sel_dict, charge_dict, z_bins, box_x, box_y, z_bin_width, z_offset, analysis_z_range=None):
    """
    Worker function for parallel EDL density calculation.
    
    Processes a chunk of frames and calculates ion densities and charge density.
    Must be at module level to be picklable by multiprocessing.
    
    Parameters
    ----------
    top_file : str
        Path to topology file
    traj_file : str
        Path to trajectory file
    frame_range : tuple
        (start, stop, step) for this worker's frame range
    ion_sel_dict : dict
        {ion_name: selection_string}
    charge_dict : dict
        {ion_name: charge_value}
    z_bins : np.ndarray
        Bin edges for z-direction
    box_x, box_y : float
        Box dimensions in x and y
    z_bin_width : float
        Width of each z-bin
    z_offset : float
        Offset to apply for box centering (box_z/2 if centering, else 0)
    analysis_z_range : tuple of (float, float), optional
        (z_min, z_max) to restrict ion analysis
        
    Returns
    -------
    dict
        Results with ion_densities and charge_density arrays
    """
    # Recreate universe in this worker process
    u = mda.Universe(top_file, traj_file)
    
    # Recreate ion AtomGroups
    ions = {name: u.select_atoms(sel_str) for name, sel_str in ion_sel_dict.items()}
    
    # Initialize storage arrays
    n_bins = len(z_bins) - 1
    ion_densities = {ion_name: np.zeros(n_bins) for ion_name in ions.keys()}
    charge_density = np.zeros(n_bins)
    
    # Helper function to get ion charge
    def get_ion_charge(ion_name):
        ion_normalized = ''.join(c for c in ion_name.upper() if c.isalpha())
        if ion_normalized in charge_dict:
            return charge_dict[ion_normalized]
        # Substring matching
        for key, value in charge_dict.items():
            if ion_normalized.startswith(key) or key.startswith(ion_normalized):
                return value
        return 0.0
    
    # Process frames
    start, stop, step = frame_range
    n_frames = 0
    
    for ts in u.trajectory[start:stop:step]:
        for ion_name, ion_atoms in ions.items():
            if len(ion_atoms) == 0:
                continue
            
            # Get ion z-positions and apply centering offset
            ion_z = ion_atoms.positions[:, 2] - z_offset
            
            # Apply z-range filter if specified
            if analysis_z_range is not None:
                z_min_analysis, z_max_analysis = analysis_z_range
                mask = (ion_z >= z_min_analysis) & (ion_z <= z_max_analysis)
                ion_z = ion_z[mask]
            
            # Bin ions
            hist, _ = np.histogram(ion_z, bins=z_bins)
            
            # Convert to number density (ions/Å³)
            bin_volume = box_x * box_y * z_bin_width
            density = hist / bin_volume
            
            ion_densities[ion_name] += density
            
            # Add to charge density
            ion_charge = get_ion_charge(ion_name)
            charge_density += density * ion_charge
        
        n_frames += 1
    
    return {
        'ion_densities': ion_densities,
        'charge_density': charge_density,
        'n_frames': n_frames
    }


def _calculate_competitive_adsorption_worker(top_file, traj_file, frame_range, 
                                             target_sel_dict, ion_sel_dict, organic_sel_dict,
                                             distance_ranges, store_per_atom_organics):
    """
    Worker function for parallel competitive adsorption calculation.
    
    Processes a chunk of frames and returns time series data.
    Must be at module level to be picklable by multiprocessing.
    
    Parameters
    ----------
    top_file : str
        Path to topology file
    traj_file : str
        Path to trajectory file
    frame_range : tuple
        (start, stop, step) for this worker's frame range
    target_sel_dict : dict
        {target_name: selection_string}
    ion_sel_dict : dict
        {ion_name: selection_string}
    organic_sel_dict : dict
        {organic_name: selection_string}
    distance_ranges : dict
        {label: (min_dist, max_dist)}
    store_per_atom_organics : bool
        Whether to store per-atom data for organics
        
    Returns
    -------
    dict
        Results for this frame chunk with time series lists
    """
    # Recreate universe in this worker process
    u = mda.Universe(top_file, traj_file)
    
    # Recreate target AtomGroups
    targets = {name: u.select_atoms(sel_str) for name, sel_str in target_sel_dict.items()}
    
    # Recreate ion AtomGroups
    ions = {name: u.select_atoms(sel_str) for name, sel_str in ion_sel_dict.items()}
    
    # Recreate organic AtomGroups
    organics = {name: u.select_atoms(sel_str) for name, sel_str in organic_sel_dict.items()}
    
    # Initialize results structure
    results = {
        'ions': {},
        'organics': {}
    }
    
    # Initialize ion storage
    for ion_name in ions.keys():
        results['ions'][ion_name] = {}
        for target_name in targets.keys():
            results['ions'][ion_name][target_name] = {label: [] for label in distance_ranges.keys()}
    
    # Initialize organic storage
    for org_name in organics.keys():
        results['organics'][org_name] = {}
        for target_name in targets.keys():
            if store_per_atom_organics:
                results['organics'][org_name][target_name] = {label: {} for label in distance_ranges.keys()}
            else:
                results['organics'][org_name][target_name] = {label: [] for label in distance_ranges.keys()}
    
    # Process frames
    start, stop, step = frame_range
    for ts in u.trajectory[start:stop:step]:
        # Loop over each target
        for target_name, target_atoms in targets.items():
            target_positions = target_atoms.positions
            
            # Analyze ions
            for ion_name, ion_atoms in ions.items():
                if len(ion_atoms) == 0:
                    continue
                
                ion_positions = ion_atoms.positions
                distances_matrix = cdist(ion_positions, target_positions)
                min_distances = np.min(distances_matrix, axis=1)
                
                for label, (min_dist, max_dist) in distance_ranges.items():
                    count = np.sum((min_distances >= min_dist) & (min_distances < max_dist))
                    results['ions'][ion_name][target_name][label].append(count)
            
            # Analyze organics
            for org_name, org_atoms in organics.items():
                if len(org_atoms) == 0:
                    continue
                
                org_positions = org_atoms.positions
                distances_matrix = cdist(org_positions, target_positions)
                min_distances = np.min(distances_matrix, axis=1)
                
                if store_per_atom_organics:
                    # Store per-atom data
                    for atom, min_dist in zip(org_atoms, min_distances):
                        atom_name = atom.name
                        
                        for label, (range_min, range_max) in distance_ranges.items():
                            if atom_name not in results['organics'][org_name][target_name][label]:
                                results['organics'][org_name][target_name][label][atom_name] = []
                            
                            in_range = 1 if (min_dist >= range_min and min_dist < range_max) else 0
                            results['organics'][org_name][target_name][label][atom_name].append(in_range)
                else:
                    # Aggregate all atoms
                    for label, (min_dist, max_dist) in distance_ranges.items():
                        count = np.sum((min_distances >= min_dist) & (min_distances < max_dist))
                        results['organics'][org_name][target_name][label].append(count)
    
    return results


class ClayOrganicIonWaterAnalysis:
    """
    Comprehensive analysis class for clay-organic-ion-water systems.
    
    This class provides methods to analyze:
    - Competitive adsorption between ions and organics
    - Multi-component radial distribution functions
    - Organic molecule conformations and orientations
    - Network topology and bridging effects
    - Water-mediated interactions
    - Spatial organization and layering
    - Dynamic exchange processes
    - Thermodynamic properties
    """
    
    def __init__(self, top, traj, solute_sel, solvent_sel, 
                 cation_sel=None, anion_sel=None, clay_sel=None, center_method='COM'):
        """
        Initialize the analysis object.
        
        Parameters:
        -----------
        top : str
            Path to topology file (e.g., 'nvt.tpr')
        traj : str
            Path to trajectory file (e.g., 'nvt.trr')
        solute_sel : str
            MDAnalysis selection string for organic/solute molecules (e.g., 'resname api')
        solvent_sel : str
            MDAnalysis selection string for water molecules (e.g., 'resname SOL or resname WAT')
        cation_sel : dict, optional
            Dictionary of cation selections {'ion_name': 'selection_string'}
            Example: {'Na': 'name NA', 'K': 'name K', 'Mg': 'name MG', 'Ca': 'name CA'}
        anion_sel : str or dict, optional
            Anion selection string or dictionary {'ion_name': 'selection_string'}
            Example: 'name CL' or {'Cl': 'name CL'}
        clay_sel : str, optional
            MDAnalysis selection string for clay atoms (e.g., 'resname MMT')
            If None, can be defined later using define_selections()
        center_method : str, default='COM'
            Method for centering calculations ('COM', 'COG', or 'centroid')
        """
        # Store file paths for parallel processing
        self.top = top
        self.traj = traj
        
        # Load trajectory
        self.u = mda.Universe(top, traj)
        
        # Store center method
        self.center_method = center_method
        
        # Initialize main selections
        organic_name = self._parse_organic_name(solute_sel, default='solute')
        self.organics = {organic_name: self.u.select_atoms(solute_sel)}
        self.water = self.u.select_atoms(solvent_sel)
        
        # Handle clay selection
        if clay_sel is not None:
            self.clay = self.u.select_atoms(clay_sel)
        else:
            self.clay = self.u.select_atoms('resid 0')  # Empty selection, will be defined later
        
        # Handle ion selections
        self.ions = {}
        if cation_sel is not None:
            if isinstance(cation_sel, dict):
                for name, sel in cation_sel.items():
                    self.ions[name] = self.u.select_atoms(sel)
            else:
                self.ions['cation'] = self.u.select_atoms(cation_sel)
        
        if anion_sel is not None:
            if isinstance(anion_sel, dict):
                for name, sel in anion_sel.items():
                    self.ions[name] = self.u.select_atoms(sel)
            else:
                # Extract element name from selection string (e.g., 'name CL' -> 'Cl')
                ion_name = self._parse_ion_name(anion_sel, default='anion')
                self.ions[ion_name] = self.u.select_atoms(anion_sel)
        
        # Storage for custom selections
        self.custom_selections = {}
        
        # Map AtomGroup IDs to custom names for reliable labeling
        self._atomgroup_to_name = {}
        
        # Storage for analysis results
        self.results = {}
        self.rdf_results = {}
        self.density_results = {}
        self.network_results = {}
        self.dynamics_results = {}
        
        # Analysis parameters
        self.rdf_range = (0.0, 15.0)
        self.rdf_nbins = 150
        self.density_bin_size = 0.5
    
    def _parse_ion_name(self, selection_string, default='ion'):
        """
        Extract ion name from selection string.
        
        Parameters:
        -----------
        selection_string : str
            MDAnalysis selection string (e.g., 'name CL' or 'name NA')
        default : str
            Default name if parsing fails
            
        Returns:
        --------
        str : Capitalized ion name (e.g., 'Cl', 'Na')
        """
        import re
        # Try to extract from "name XXX" pattern
        match = re.search(r'name\s+([A-Za-z]+)', selection_string, re.IGNORECASE)
        if match:
            ion_name = match.group(1)
            # Capitalize properly (e.g., CL -> Cl, NA -> Na)
            return ion_name.capitalize()
        return default
    
    def _parse_organic_name(self, selection_string, default='solute'):
        """
        Extract organic/residue name from selection string.
        
        Parameters:
        -----------
        selection_string : str
            MDAnalysis selection string (e.g., 'resname api' or 'resname CIP')
        default : str
            Default name if parsing fails
            
        Returns:
        --------
        str : Lowercase organic name (e.g., 'api', 'cip')
        """
        import re
        # Try to extract from "resname XXX" pattern
        match = re.search(r'resname\s+([A-Za-z0-9_-]+)', selection_string, re.IGNORECASE)
        if match:
            organic_name = match.group(1)
            # Keep lowercase for consistency
            return organic_name.lower()
        return default
    
    def define_selections(self, selections_dict):
        """
        Define custom atom selections for different molecular components.
        
        Parameters:
        -----------
        selections_dict : dict
            Nested dictionary of selections. Structure:
            {
                'group_name': {
                    'subgroup_name': 'selection_string',
                    ...
                },
                ...
            }
            
        Example:
        --------
        analysis.define_selections({
            'CIP_parts': {
                'quinolone': 'resname api and (name N1 or name C)',
                'piperazine': 'resname api and (name N or name N2)'
            },
            'MMT_surface': {
                'surface_oxygen': 'resname MMT and name Ob',
                'surface_silicon': 'resname MMT and name Si'
            },
            'solvent': {
                'water_oxygen': 'resname SOL WAT and (name OW or name Ow)',
                'water_hydrogen': 'resname SOL WAT and (name HW1 or name HW2)'
            }
        })
        
        Returns:
        --------
        None : Updates self.custom_selections with AtomGroup objects
        """
        for group_name, subgroups in selections_dict.items():
            if group_name not in self.custom_selections:
                self.custom_selections[group_name] = {}
            
            for subgroup_name, selection_string in subgroups.items():
                try:
                    atoms = self.u.select_atoms(selection_string)
                    # Store the selection string as attribute
                    atoms._selection_string = selection_string
                    atoms._custom_name = subgroup_name  # Store as attribute (may not persist)
                    
                    # ALSO store in persistent mapping using id(atoms)
                    self._atomgroup_to_name[id(atoms)] = subgroup_name
                    
                    self.custom_selections[group_name][subgroup_name] = atoms
                    print(f"✓ Defined {group_name}.{subgroup_name}: {len(atoms)} atoms")
                    
                    # Special handling for clay/MMT surface
                    if group_name == 'MMT_surface' and len(self.clay) == 0:
                        # Combine all MMT selections to create clay selection
                        all_mmt_atoms = self.u.select_atoms(' or '.join(
                            [f'({sel})' for sel in subgroups.values()]
                        ))
                        self.clay = all_mmt_atoms
                        print(f"  → Updated clay selection: {len(self.clay)} atoms")
                        
                except Exception as e:
                    print(f"✗ Warning: Could not define {group_name}.{subgroup_name}: {e}")
        
        print(f"\n✓ Selection definitions complete")
        print(f"  Groups defined: {list(self.custom_selections.keys())}")
    
    def get_selection(self, path):
        """
        Retrieve a custom selection by path.
        
        Parameters:
        -----------
        path : str
            Selection path in 'group.subgroup' format
            Example: 'CIP_parts.quinolone' or 'solvent.water_oxygen'
        
        Returns:
        --------
        AtomGroup : The selected atoms
        
        Example:
        --------
        >>> quinolone = analysis.get_selection('CIP_parts.quinolone')
        >>> water_oxygen = analysis.get_selection('solvent.water_oxygen')
        """
        if '.' not in path:
            raise ValueError(f"Path must be in 'group.subgroup' format, got: {path}")
        
        group, subgroup = path.split('.', 1)
        if group not in self.custom_selections:
            raise KeyError(f"Group '{group}' not found. Available: {list(self.custom_selections.keys())}")
        if subgroup not in self.custom_selections[group]:
            raise KeyError(f"Subgroup '{subgroup}' not found in '{group}'. Available: {list(self.custom_selections[group].keys())}")
        
        return self.custom_selections[group][subgroup]
    
    def get_selections(self, *paths):
        """
        Retrieve multiple custom selections at once.
        
        Parameters:
        -----------
        *paths : str
            Variable number of selection paths
        
        Returns:
        --------
        tuple : AtomGroups for each path
        
        Example:
        --------
        >>> quinolone, piperazine, water = analysis.get_selections(
        ...     'CIP_parts.quinolone',
        ...     'CIP_parts.piperazine',
        ...     'solvent.water_oxygen'
        ... )
        """
        return tuple(self.get_selection(path) for path in paths)
    
    def _extract_label_from_atomgroup(self, atomgroup_or_sel, return_selection_string=False):
        """
        Extract a label from an AtomGroup or selection string.
        
        Parameters:
        -----------
        atomgroup_or_sel : AtomGroup or str
            AtomGroup object or selection string
        return_selection_string : bool
            If True, also return the selection string for reporting
        
        Returns:
        --------
        str or tuple : Label for the group, optionally with selection string
        """
        import re
        
        selection_string = None
        label = None
        
        # If it's a string (selection path), extract from it
        if isinstance(atomgroup_or_sel, str):
            if '.' in atomgroup_or_sel:
                # Custom selection path like 'CIP_parts.quinolone'
                label = atomgroup_or_sel.split('.')[1]
                # Try to get the actual selection string
                try:
                    group, subgroup = atomgroup_or_sel.split('.')
                    if group in self.custom_selections and subgroup in self.custom_selections[group]:
                        ag = self.custom_selections[group][subgroup]
                        selection_string = getattr(ag, '_selection_string', None)
                except:
                    pass
            else:
                # Try to extract from selection string - prioritize atom name over resname
                selection_string = atomgroup_or_sel
                
                # First try to get atom name (most specific)
                name_match = re.search(r'\bname\s+(\S+)', atomgroup_or_sel, re.IGNORECASE)
                if name_match:
                    label = name_match.group(1)
                else:
                    # Fall back to resname
                    resname_match = re.search(r'\bresname\s+(\S+)', atomgroup_or_sel, re.IGNORECASE)
                    if resname_match:
                        label = resname_match.group(1)
                    else:
                        label = atomgroup_or_sel[:20]
        else:
            # If it's an AtomGroup, try to get meaningful name
            try:
                # SMART LABELING: Use custom name only for multi-atom groups (3+ atom types)
                # For single-atom-type groups, use the atom name for brevity
                
                # First, check if this is a multi-atom group
                unique_atom_names = None
                if hasattr(atomgroup_or_sel, 'names') and len(atomgroup_or_sel) > 0:
                    # Only check first 100 atoms to avoid performance hit
                    sample_size = min(len(atomgroup_or_sel), 100)
                    unique_atom_names = set(atomgroup_or_sel[:sample_size].names)
                
                # PRIORITY 1: For multi-atom groups (3+), use custom name if available
                if unique_atom_names and len(unique_atom_names) >= 3:
                    atomgroup_id = id(atomgroup_or_sel)
                    if hasattr(self, '_atomgroup_to_name') and atomgroup_id in self._atomgroup_to_name:
                        label = self._atomgroup_to_name[atomgroup_id]
                    elif hasattr(atomgroup_or_sel, '_custom_name'):
                        label = atomgroup_or_sel._custom_name
                
                # PRIORITY 2: For single/dual atom groups, extract atom name directly
                if not label and unique_atom_names and len(unique_atom_names) <= 2:
                    # Use atom name(s) directly for single/dual atom groups
                    label = '_'.join(sorted(unique_atom_names))
                
                # PRIORITY 3: Try selection string if no atom names found
                if not label and hasattr(atomgroup_or_sel, '_selection_string'):
                    selection_string = atomgroup_or_sel._selection_string
                    
                    # Try to extract from selection string - prioritize atom name over resname
                    name_match = re.search(r'\bname\s+(\S+)', selection_string, re.IGNORECASE)
                    if name_match:
                        label = name_match.group(1)
                    else:
                        # Fall back to resname
                        resname_match = re.search(r'\bresname\s+(\S+)', selection_string, re.IGNORECASE)
                        if resname_match:
                            label = resname_match.group(1)
                
                # PRIORITY 4: Check residue name (fallback)
                if not label and hasattr(atomgroup_or_sel, 'resnames'):
                    # Sample a few residues instead of all (faster for large groups)
                    if len(atomgroup_or_sel) <= 100:
                        unique_resnames = set(atomgroup_or_sel.resnames)
                    else:
                        # For large groups, sample first 100 atoms only
                        unique_resnames = set(atomgroup_or_sel[:100].resnames)
                    
                    if len(unique_resnames) == 1:
                        label = list(unique_resnames)[0]
                        if not selection_string:
                            selection_string = f"resname {label}"
                
                # PRIORITY 5: Generic fallback for large groups
                if not label:
                    if unique_atom_names:
                        sorted_names = sorted(unique_atom_names)
                        label = '_'.join(sorted_names[:3]) + f'_etc{len(unique_atom_names)}atoms'
                    elif len(atomgroup_or_sel) > 100:
                        label = f"group_{len(atomgroup_or_sel)}"
                    else:
                        label = f"group_{len(atomgroup_or_sel)}"
                    
            except Exception:
                label = f"group_{len(atomgroup_or_sel)}"
        
        # Ensure we have a selection string for reporting
        if not selection_string:
            try:
                # Try to reconstruct selection string from AtomGroup properties
                if not isinstance(atomgroup_or_sel, str):
                    # Check if all atoms have the same name (common for simple ions)
                    if hasattr(atomgroup_or_sel, 'names') and len(atomgroup_or_sel) > 0:
                        unique_names = set(atomgroup_or_sel.names)
                        if len(unique_names) == 1:
                            # Single atom type - use "name XXX"
                            atom_name = list(unique_names)[0]
                            selection_string = f"name {atom_name}"
                        elif len(unique_names) <= 3:
                            # Few atom types - use "name XXX or name YYY"
                            selection_string = " or ".join([f"name {n}" for n in sorted(unique_names)])
                        else:
                            # Many atom types - use index-based selection
                            indices = ' '.join(map(str, atomgroup_or_sel.indices))
                            selection_string = f"index {indices}"
                    else:
                        # Last resort: use indices
                        indices = ' '.join(map(str, atomgroup_or_sel.indices))
                        selection_string = f"index {indices}"
                else:
                    selection_string = atomgroup_or_sel
            except:
                selection_string = "unknown selection"
        
        # Always return consistently based on return_selection_string
        if return_selection_string:
            return label, selection_string
        return label
        
    def calculate_multi_component_rdfs(self, components=None, atom_types=None):
        """
        Calculate radial distribution functions between all component pairs.
        
        Parameters:
        -----------
        components : list, optional
            List of component names to include. If None, includes all components.
        atom_types : dict, optional
            Dictionary specifying atom types for each component
        
        Returns:
        --------
        dict : RDF results for each component pair
        """
        print("Calculating multi-component RDFs...")
        
        if components is None:
            all_components = {'clay': self.clay}
            all_components.update(self.ions)
            all_components.update(self.organics)
            all_components['water'] = self.water
        else:
            all_components = {comp: getattr(self, comp) if hasattr(self, comp) 
                            else self.ions.get(comp, self.organics.get(comp)) 
                            for comp in components}
        
        rdf_results = {}
        
        for i, (comp1, atoms1) in enumerate(all_components.items()):
            for j, (comp2, atoms2) in enumerate(all_components.items()):
                if i <= j:  # Avoid duplicates
                    pair_name = f"{comp1}-{comp2}"
                    print(f"  Computing RDF for {pair_name}")
                    
                    try:
                        rdf_analysis = rdf.InterRDF(atoms1, atoms2,
                                                  nbins=self.rdf_nbins,
                                                  range=self.rdf_range)
                        rdf_analysis.run()
                        
                        rdf_results[pair_name] = {
                            'r': rdf_analysis.bins,
                            'gr': rdf_analysis.rdf,
                            'coordination_number': np.cumsum(rdf_analysis.rdf * 
                                                           rdf_analysis.bins**2 * 
                                                           4 * np.pi * 
                                                           (rdf_analysis.bins[1] - rdf_analysis.bins[0]))
                        }
                    except Exception as e:
                        print(f"    Warning: Could not calculate RDF for {pair_name}: {e}")
        
        self.rdf_results.update(rdf_results)
        return rdf_results
    
    def molecular_rdf(self, group1_sel, group2_sel, bin_width=0.05, range=(0, 15), 
                     step=1, njobs=1, center_method=None, normalize=True,
                     save_cache=True, cache_file=None, force_rerun=False,
                     store_per_atom=False):
        '''
        Calculate RDF between arbitrary molecular groups with flexible centering options.
        Supports lists for batch RDF calculations. Now includes automatic ion type handling 
        and caching support.
        
        Parameters
        ----------
        group1_sel : str or list of str
            Selection string(s) for first group. If list, calculates RDF for each selection.
        group2_sel : str or list of str
            Selection string(s) for second group. If list, calculates RDF for each selection.
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
            Custom cache filename. If None, auto-generates from parameters.
            Ignored when using lists (each RDF gets its own cache).
        force_rerun : bool
            Force recalculation even if cache exists
        store_per_atom : bool, default=False
            If True, for multi-atom selections (3+ unique atom types), calculate and 
            store RDF for each individual atom in addition to the grouped RDF.
            Results stored as: 'Ob-piperazine' (grouped) + 'Ob-N13', 'Ob-N16', etc. (individual)
        
        Returns
        -------
        results : object or dict
            If both selections are strings: Returns RDF results object with 
            .bins, .rdf, .count, .edges attributes (backward compatible).
            
            If one or both selections are lists: Returns dictionary with format
            {label: rdf_results} where labels are extracted from selection strings.
        '''
        
        import hashlib
        import os
        
        # Check if either parameter is a list
        is_group1_list = isinstance(group1_sel, list)
        is_group2_list = isinstance(group2_sel, list)
        
        # If lists provided, calculate multiple RDFs and return dictionary
        if is_group1_list or is_group2_list:
            print("🔄 Batch RDF calculation mode activated")
            
            # Convert single strings to lists for uniform handling
            group1_list = group1_sel if is_group1_list else [group1_sel]
            group2_list = group2_sel if is_group2_list else [group2_sel]
            
            results_dict = {}
            total_rdfs = len(group1_list) * len(group2_list)
            
            # Auto-adjust njobs to avoid wasting workers
            actual_njobs = min(njobs, total_rdfs) if njobs > 1 else 1
            
            # Parallel execution if njobs > 1
            if actual_njobs > 1:
                # Prepare cache directory
                cache_dir = os.path.join(os.path.dirname(self.traj), '.rdf_cache')
                
                # Collect all tasks with their parameters (grouped RDFs)
                grouped_tasks = []
                per_atom_tasks = []
                
                for g1_sel in group1_list:
                    g1_label, g1_sel_str = self._extract_label_from_atomgroup(g1_sel, return_selection_string=True)
                    
                    for g2_sel in group2_list:
                        g2_label, g2_sel_str = self._extract_label_from_atomgroup(g2_sel, return_selection_string=True)
                        label = f"{g1_label}-{g2_label}"
                        
                        # Always add grouped RDF task
                        grouped_tasks.append({
                            'label': label,
                            'g1_sel_str': g1_sel_str,
                            'g2_sel_str': g2_sel_str,
                            'g1_label': g1_label,
                            'g2_label': g2_label,
                            'g1_sel': g1_sel,
                            'g2_sel': g2_sel,
                            'is_per_atom': False
                        })
                        
                        # If store_per_atom, also add per-atom tasks
                        if store_per_atom:
                            # Get the actual AtomGroup
                            if not isinstance(g2_sel, str):
                                g2_group = g2_sel
                            else:
                                g2_group = self.u.select_atoms(g2_sel_str)
                            
                            # Add per-atom task for each unique atom name
                            unique_atom_names = set(g2_group.names)
                            for atom_name in sorted(unique_atom_names):
                                # Get selection string for this atom
                                atom_sel_str = f"({g2_sel_str}) and name {atom_name}"
                                atom_label = f"{g1_label}-{atom_name}"
                                
                                per_atom_tasks.append({
                                    'label': atom_label,
                                    'g1_sel_str': g1_sel_str,
                                    'g2_sel_str': atom_sel_str,
                                    'g1_label': g1_label,
                                    'g2_label': g2_label,
                                    'parent_group': g2_label,
                                    'is_per_atom': True
                                })
                
                # Combine all tasks
                all_tasks = grouped_tasks + per_atom_tasks
                total_all_rdfs = len(all_tasks)
                
                # Re-adjust njobs for total task count and available CPUs
                available_cpus = os.cpu_count() or 1  # Fallback to 1 if None
                actual_njobs = min(njobs, total_all_rdfs, available_cpus) if njobs > 1 else 1
                
                if len(per_atom_tasks) > 0:
                    print(f"⚡ Parallel mode: Using {actual_njobs} workers for {total_all_rdfs} RDFs")
                    print(f"   ({len(grouped_tasks)} grouped + {len(per_atom_tasks)} per-atom)")
                    if actual_njobs < njobs:
                        if actual_njobs == available_cpus:
                            print(f"   ℹ️  Auto-scaled from njobs={njobs} to {actual_njobs} (CPU limit)")
                        else:
                            print(f"   ℹ️  Auto-scaled from njobs={njobs} to {actual_njobs} (task count limit)")
                else:
                    print(f"⚡ Parallel mode: Using {actual_njobs} workers for {total_all_rdfs} RDFs")
                    if actual_njobs < njobs:
                        if actual_njobs == available_cpus:
                            print(f"   ℹ️  Auto-scaled from njobs={njobs} to {actual_njobs} (CPU limit)")
                        else:
                            print(f"   ℹ️  Auto-scaled from njobs={njobs} to {actual_njobs} (task count limit)")
                
                # Submit tasks to ProcessPoolExecutor
                try:
                    with ProcessPoolExecutor(max_workers=actual_njobs) as executor:
                        # Submit all RDF calculations
                        future_to_task = {}
                        for task in all_tasks:
                            future = executor.submit(
                                _calculate_rdf_worker,
                                self.top,
                                self.traj,
                                task['g1_sel_str'],
                                task['g2_sel_str'],
                                bin_width,
                                range,
                                step,
                                center_method if center_method is not None else self.center_method,
                                normalize,
                                save_cache,
                                cache_dir,
                                force_rerun,
                                task['label']
                            )
                            future_to_task[future] = task
                        
                        # Collect results as they complete
                        completed = 0
                        for future in as_completed(future_to_task):
                            task = future_to_task[future]
                            label, rdf_result = future.result()
                            
                            # Add parent group metadata for per-atom RDFs
                            if task['is_per_atom']:
                                if not hasattr(rdf_result, '_parent_group'):
                                    rdf_result._parent_group = task['parent_group']
                            
                            results_dict[label] = rdf_result
                            
                            completed += 1
                            task_type = "per-atom" if task['is_per_atom'] else "grouped"
                            print(f"✓ [{completed}/{total_all_rdfs}] {label} ({task_type})")
                
                except Exception as e:
                    print(f"⚠️ Parallel execution failed: {e}")
                    print("  Falling back to sequential execution...")
                    actual_njobs = 1  # Fall back to sequential
            
            # Sequential execution (fallback or njobs=1)
            if actual_njobs == 1:
                print(f"📋 Sequential mode: Processing {total_rdfs} RDFs")
                current_rdf = 0
                
                for g1_sel in group1_list:
                    g1_label, g1_sel_str = self._extract_label_from_atomgroup(g1_sel, return_selection_string=True)
                    
                    for g2_sel in group2_list:
                        g2_label, g2_sel_str = self._extract_label_from_atomgroup(g2_sel, return_selection_string=True)
                        label = f"{g1_label}-{g2_label}"
                        
                        current_rdf += 1
                        print(f"\n[{current_rdf}/{total_rdfs}] Calculating RDF: {label}")
                        print(f"  Group 1: {g1_sel_str}")
                        print(f"  Group 2: {g2_sel_str}")
                        
                        rdf_result = self.molecular_rdf(
                            g1_sel, g2_sel,
                            bin_width=bin_width,
                            range=range,
                            step=step,
                            njobs=1,
                            center_method=center_method,
                            normalize=normalize,
                            save_cache=save_cache,
                            cache_file=None,
                            force_rerun=force_rerun,
                            store_per_atom=store_per_atom
                        )
                        
                        results_dict[label] = rdf_result
                    
                    # If store_per_atom is True, check if group2 has multiple atoms
                    # and calculate individual RDFs for each atom
                    if store_per_atom:
                        # Get the actual AtomGroup
                        if not isinstance(g2_sel, str):
                            g2_group = g2_sel
                        else:
                            g2_group = self.u.select_atoms(g2_sel)
                        
                        # Check if it has 3+ unique atom types (multi-atom functional group)
                        unique_atom_names = set(g2_group.names)
                        if len(unique_atom_names) >= 3:
                            print(f"  📊 Calculating per-atom RDFs for {g2_label} ({len(unique_atom_names)} atom types)")
                            
                            # Calculate RDF for each unique atom type
                            for atom_name in sorted(unique_atom_names):
                                # Create selection for this specific atom
                                atom_indices = [i for i, name in enumerate(g2_group.names) if name == atom_name]
                                atom_group = g2_group[atom_indices]
                                
                                # Create label for individual atom
                                atom_label = f"{g1_label}-{atom_name}"
                                
                                print(f"    → {atom_label} ({len(atom_group)} atoms)")
                                
                                # Calculate RDF for this atom
                                atom_rdf = self.molecular_rdf(
                                    g1_sel, atom_group,
                                    bin_width=bin_width,
                                    range=range,
                                    step=step,
                                    njobs=njobs,
                                    center_method=center_method,
                                    normalize=normalize,
                                    save_cache=save_cache,
                                    cache_file=None,
                                    force_rerun=force_rerun,
                                    store_per_atom=False  # Don't recurse
                                )
                                
                                # Store with metadata about parent group
                                if not hasattr(atom_rdf, '_parent_group'):
                                    atom_rdf._parent_group = g2_label
                                results_dict[atom_label] = atom_rdf
            
            print(f"\n✅ Batch RDF calculation complete! Generated {len(results_dict)} RDFs")
            print(f"   Labels: {list(results_dict.keys())}")
            return results_dict
        
        # Original single RDF calculation (backward compatible)
        if center_method is None:
            center_method = self.center_method
        
        # Resolve AtomGroups if needed
        if not isinstance(group1_sel, str):
            # It's an AtomGroup - use it directly
            group1 = group1_sel
        else:
            group1 = self.u.select_atoms(group1_sel)
            
        if not isinstance(group2_sel, str):
            # It's an AtomGroup - use it directly  
            group2 = group2_sel
        else:
            group2 = self.u.select_atoms(group2_sel)
        
        # Generate cache filename if not provided
        if cache_file is None:
            # Create hash from parameters for unique cache filename
            param_str = f"{len(group1)}_{len(group2)}_bw{bin_width}_r{range[0]}-{range[1]}_s{step}_cm{center_method}_n{normalize}"
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
        if len(group1) == 0 or len(group2) == 0:
            raise ValueError("One or both atom groups are empty")
        
        print(f"  🔍 Starting RDF calculation: {len(group1)} × {len(group2)} atoms")
        nbins = int((range[1] - range[0]) / bin_width)
        print(f"  📊 Using {nbins} bins, method: {center_method}")
        
        # Use different calculation methods based on center_method
        if center_method in ['COM', 'COG']:
            # Use center-based RDF calculation
            print(f"  ⚙️ Using custom center-based RDF calculation")
            results = self._rdf_with_centers(group1, group2, nbins, range, step, 
                                            njobs, center_method, normalize)
        else:
            # Use standard InterRDF calculation
            print(f"  ⚙️ Initializing MDAnalysis InterRDF...")
            rdf_analysis = InterRDF(group1, group2, nbins=nbins, range=range, 
                          norm='rdf' if normalize else 'none', verbose=True)
            print(f"  ▶️ Running InterRDF.run(step={step}, njobs={njobs})...")
            
            # Run with multiprocessing support and fallback
            # Note: InterRDF does not support parallel processing in any MDAnalysis version
            # The njobs parameter is accepted but ignored, and the fallback ensures compatibility
            if njobs == 1:
                rdf_analysis.run(step=step)
            else:
                try:
                    rdf_analysis.run(step=step, njobs=njobs)
                except Exception as e:
                    print(f"  ⚠️ Multiprocessing not supported by InterRDF - using single thread")
                    rdf_analysis.run(step=step)
            
            print(f"  ✅ InterRDF calculation complete!")
            results = rdf_analysis.results
        
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
    
    def _rdf_with_centers(self, group1, group2, nbins, range_rdf, step, njobs, center_method, normalize):
        '''
        Calculate RDF using molecular centers (COM or COG).
        
        Treats group1 as ONE molecular entity (single center point).
        Calculates distances from that center to all group2 atoms.
        '''
        
        from scipy.spatial import distance as scipy_distance
        from tqdm import tqdm
        
        print(f"\nCalculating {center_method}-based RDF:")
        print(f"  Group1: {len(group1)} atoms -> 1 center point")
        print(f"  Group2: {len(group2)} atoms")
        
        # Initialize RDF calculation
        dr = (range_rdf[1] - range_rdf[0]) / nbins
        bins = np.linspace(range_rdf[0], range_rdf[1], nbins + 1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        rdf_hist = np.zeros(nbins)
        n_frames = 0
        
        # Get box dimensions for volume (assuming orthorhombic box)
        box_vol = np.prod(self.u.dimensions[:3])
        
        # Calculate number density of group2
        n_group2 = len(group2)
        number_density = n_group2 / box_vol
        
        print(f"  Processing trajectory (step={step})...")
        
        # Calculate RDF over trajectory
        for ts in tqdm(self.u.trajectory[::step], desc="Computing RDF"):
            # Calculate center for group1 (single point)
            if center_method == 'COM':
                center1 = group1.center_of_mass().reshape(1, 3)
            else:  # COG
                center1 = group1.center_of_geometry().reshape(1, 3)
            
            # Get positions for group2 (all atoms)
            positions2 = group2.positions
            
            # Calculate distances from group1 center to all group2 atoms
            distances_array = scipy_distance.cdist(center1, positions2, metric='euclidean').flatten()
            
            # Bin the distances
            hist, _ = np.histogram(distances_array, bins=bins)
            rdf_hist += hist
            n_frames += 1
        
        # Normalize the RDF
        if normalize:
            # Calculate shell volumes
            shell_volumes = (4.0/3.0) * np.pi * (bins[1:]**3 - bins[:-1]**3)
            
            # Normalize: g(r) = (histogram) / (n_frames * number_density * shell_volume)
            rdf = rdf_hist / (n_frames * number_density * shell_volumes)
        else:
            rdf = rdf_hist / n_frames
        
        # Create results object matching InterRDF format
        class RDFResults:
            def __init__(self, bins, rdf, hist, edges):
                self.bins = bins
                self.rdf = rdf
                self.count = hist
                self.edges = edges
        
        results = RDFResults(bin_centers, rdf, rdf_hist, bins)
        
        print(f"✓ Complete! Processed {n_frames} frames")
        print(f"  First peak: {bin_centers[np.argmax(rdf[:50])]:.2f} Å (g(r)={np.max(rdf[:50]):.3f})")
        
        return results
    
    def analyze_competitive_adsorption(self, target_sel, ion_types=None, organic_parts=None,
                                       distance_cutoff=None, save_cache=True, cache_file=None, 
                                       force_rerun=False, step=1, store_per_atom_organics=False,
                                       njobs=1):
        """
        Analyze competitive adsorption with distance-range categorization.
        
        Counts how many ions/atoms fall into each distance range (CIP, SIP, DSIP, etc.)
        relative to target surface atoms. Supports multiple separate targets for comparison.
        
        Parameters:
        -----------
        target_sel : AtomGroup, list of AtomGroups, or dict of AtomGroups
            Clay surface atoms to measure distances to. Can be:
            - Single AtomGroup: e.g., octahedral_hydroxyl
            - List: [octahedral_mg, octahedral_hydroxyl] - analyzed separately
            - Dict: {'Mgo': octahedral_mg, 'OH': octahedral_hydroxyl} - with custom names
        ion_types : list of str, optional
            Ion names to analyze (e.g., ['Na', 'Mg']). If None, analyzes all ions in self.ions.
        organic_parts : dict, optional
            Dictionary of organic parts to analyze: {'part_name': AtomGroup}.
            Example: {'quinolone': quinolone_atoms, 'piperazine': piperazine_atoms}
            If None, SKIPS organic analysis (changed from previous behavior).
        distance_cutoff : dict
            Distance ranges with labels. Format:
            {'CIP': '3', 'SIP': '3-6', 'DSIP': '6-9'}
            where '3' means 0-3 Å, '3-6' means 3-6 Å, etc.
        save_cache : bool, default=True
            Whether to save results to cache file
        cache_file : str, optional
            Custom cache filename. If None, auto-generates.
        force_rerun : bool, default=False
            Force recalculation even if cache exists
        step : int, default=1
            Trajectory frame step
        store_per_atom_organics : bool, default=False
            If True, stores per-atom data for organic molecules (e.g., O1, O3, C2 separately).
            If False, aggregates all atoms in each organic moiety.
            Useful for understanding which specific atoms drive binding interactions.
        njobs : int, default=1
            Number of parallel workers for trajectory processing.
            If njobs > 1, splits frames across multiple processes.
            Use njobs=-1 to use all available CPU cores.
            Recommended for large trajectories (>1000 frames).
        
        Returns:
        --------
        dict : Competitive adsorption results with structure:
            {
                'ions': {
                    'Na': {
                        'Ohmg': {'CIP': {...}, 'SIP': {...}},
                        'Mgo': {'CIP': {...}, 'SIP': {...}}
                    },
                    'Mg': {...}
                },
                'organics': {
                    'quinolone': {
                        'Ohmg': {'CIP': {...}, 'SIP': {...}},
                        'Mgo': {'CIP': {...}, 'SIP': {...}}
                    }
                }
            }
            Each range contains: {'mean': float, 'std': float, 'time_series': array}
            
            If store_per_atom_organics=True, organic structure changes to:
            'organics': {
                'quinolone': {
                    'Ohmg': {
                        'CIP': {
                            'O1': {'mean': ..., 'std': ..., 'time_series': ...},
                            'C2': {'mean': ..., 'std': ..., 'time_series': ...}
                        }
                    }
                }
            }
        
        Example:
        --------
        >>> # Multiple targets analyzed separately
        >>> results = analysis.analyze_competitive_adsorption(
        ...     target_sel=[octahedral_mg, octahedral_hydroxyl],
        ...     ion_types=['Na', 'Mg'],
        ...     organic_parts={'quinolone': quinolone, 'carboxylic_acid': carboxylic_acid},
        ...     distance_cutoff={'CIP': '3', 'SIP': '3-6', 'DSIP': '6-9'}
        ... )
        >>> # Access: results['ions']['Na']['Ohmg']['CIP']['mean']
        >>> 
        >>> # Skip organics (new default behavior)
        >>> results = analysis.analyze_competitive_adsorption(
        ...     target_sel=surface_oxygen,
        ...     ion_types=['Na', 'Mg'],
        ...     distance_cutoff={'CIP': '3', 'SIP': '3-6'}
        ... )  # organic_parts=None skips organics
        """
        from tqdm import tqdm
        import hashlib
        
        print("="*60)
        print("Analyzing Competitive Adsorption with Distance Ranges")
        print("="*60)
        
        # Validate distance_cutoff
        if distance_cutoff is None:
            raise ValueError("distance_cutoff dictionary is required. Example: {'CIP': '3', 'SIP': '3-6'}")
        
        # Process target_sel into target_dict {name: AtomGroup}
        if isinstance(target_sel, dict):
            # Already a dict with custom names
            target_dict = target_sel
            print(f"  Analyzing {len(target_dict)} separate targets (dict provided)")
        elif isinstance(target_sel, list):
            # List of AtomGroups - generate names from atom types
            target_dict = {}
            for i, target_atoms in enumerate(target_sel):
                # Try to extract meaningful name from atom names
                target_name = self._extract_label_from_atomgroup(target_atoms)
                target_dict[target_name] = target_atoms
            print(f"  Analyzing {len(target_dict)} separate targets (list provided)")
        else:
            # Single AtomGroup - extract name
            target_name = self._extract_label_from_atomgroup(target_sel)
            target_dict = {target_name: target_sel}
            print(f"  Analyzing 1 target: {target_name}")
        
        # Print target details
        for target_name, target_atoms in target_dict.items():
            print(f"    {target_name}: {len(target_atoms)} atoms")
        
        # Parse distance ranges
        distance_ranges = {}
        for label, range_str in distance_cutoff.items():
            if '-' in range_str:
                # Range like '3-6'
                parts = range_str.split('-')
                min_dist = float(parts[0])
                max_dist = float(parts[1])
            else:
                # Single value like '3' means 0-3
                min_dist = 0.0
                max_dist = float(range_str)
            distance_ranges[label] = (min_dist, max_dist)
            print(f"  {label}: {min_dist:.1f} - {max_dist:.1f} Å")
        
        # Determine which ions to analyze
        if ion_types is None:
            # Skip ions (only analyze what's explicitly requested)
            ion_types = []
            print(f"  Skipping ions (ion_types=None)")
        else:
            print(f"  Analyzing ions: {ion_types}")
        
        # Determine which organics to analyze
        if organic_parts is None:
            # Skip organics (only analyze what's explicitly requested)
            organic_parts = {}
            print(f"  Skipping organics (organic_parts=None)")
        else:
            print(f"  Analyzing organic parts: {list(organic_parts.keys())}")
        
        # Generate cache filenames for ions and organics separately
        if cache_file is None:
            total_target_atoms = sum(len(atoms) for atoms in target_dict.values())
            target_str = '-'.join(target_dict.keys())
            
            # Separate cache for ions
            if ion_types:
                ion_param_str = f"comp_ads_ions_{total_target_atoms}_{target_str}_{'-'.join(ion_types)}_s{step}"
                ion_hash = hashlib.md5(ion_param_str.encode()).hexdigest()[:8]
                ion_cache_file = f"competitive_adsorption_ions_{ion_hash}.npz"
            else:
                ion_cache_file = None
            
            # Separate cache for organics (include per-atom flag)
            if organic_parts:
                org_str = '-'.join(organic_parts.keys())
                per_atom_str = "per_atom" if store_per_atom_organics else "aggregate"
                org_param_str = f"comp_ads_org_{total_target_atoms}_{target_str}_{org_str}_{per_atom_str}_s{step}"
                org_hash = hashlib.md5(org_param_str.encode()).hexdigest()[:8]
                org_cache_file = f"competitive_adsorption_org_{org_hash}.npz"
            else:
                org_cache_file = None
        else:
            # User provided custom cache file - use it for both
            ion_cache_file = cache_file if ion_types else None
            org_cache_file = cache_file if organic_parts else None
        
        # Initialize results structure
        results = {
            'ions': {},
            'organics': {},
            'metadata': {
                'distance_ranges': distance_ranges,
                'targets': {name: len(atoms) for name, atoms in target_dict.items()},
                'n_frames': 0
            }
        }
        
        # Load existing ion cache if available
        ion_cached = False
        if save_cache and not force_rerun and ion_cache_file and os.path.exists(ion_cache_file):
            print(f"\n📂 Found existing ion cache: {ion_cache_file}")
            try:
                print("   Loading cached ion results...")
                cached_data = np.load(ion_cache_file, allow_pickle=True)
                cached_results = cached_data['results'].item()
                results['ions'] = cached_results.get('ions', {})
                if 'metadata' in cached_results:
                    results['metadata']['n_frames'] = cached_results['metadata'].get('n_frames', 0)
                print(f"   ✅ Loaded {len(results['ions'])} ion types from cache")
                ion_cached = True
            except Exception as e:
                print(f"   ⚠️ Failed to load ion cache: {e}")
                results['ions'] = {}
        
        # Load existing organic cache if available
        org_cached = False
        if save_cache and not force_rerun and org_cache_file and os.path.exists(org_cache_file):
            print(f"\n📂 Found existing organic cache: {org_cache_file}")
            try:
                print("   Loading cached organic results...")
                cached_data = np.load(org_cache_file, allow_pickle=True)
                cached_results = cached_data['results'].item()
                results['organics'] = cached_results.get('organics', {})
                if 'metadata' in cached_results and results['metadata']['n_frames'] == 0:
                    results['metadata']['n_frames'] = cached_results['metadata'].get('n_frames', 0)
                print(f"   ✅ Loaded {len(results['organics'])} organic parts from cache")
                org_cached = True
            except Exception as e:
                print(f"   ⚠️ Failed to load organic cache: {e}")
                results['organics'] = {}
        
        # If both are cached, return merged results
        if ion_cached and org_cached:
            print(f"\n✅ Both ions and organics loaded from cache - returning merged results")
            self.results['competitive_adsorption'] = results
            return results
        
        # If ions are cached and no organics requested, return ion results
        if ion_cached and not organic_parts:
            print(f"\n✅ Ion data loaded from cache, no organics requested - returning results")
            self.results['competitive_adsorption'] = results
            return results
        
        # If organics are cached and no ions requested, return organic results
        if org_cached and not ion_types:
            print(f"\n✅ Organic data loaded from cache, no ions requested - returning results")
            self.results['competitive_adsorption'] = results
            return results
        
        # If only analyzing ions and they're cached, just need organics
        if ion_types and ion_cached and organic_parts:
            print(f"\n🔄 Ion data loaded from cache, will calculate organics only")
            ion_types = []  # Skip ion calculation
        
        # If only analyzing organics and they're cached, just need ions
        if organic_parts and org_cached and ion_types:
            print(f"\n🔄 Organic data loaded from cache, will calculate ions only")
            organic_parts = {}  # Skip organic calculation
        
        # Initialize storage for each ion type (ion -> target -> range)
        # Only initialize as lists if we need to calculate (not if already cached as dicts)
        for ion_name in ion_types:
            if ion_name not in self.ions:
                print(f"  ⚠️ Warning: Ion '{ion_name}' not found in self.ions")
                continue
            if ion_name not in results['ions']:
                # New ion - create structure
                results['ions'][ion_name] = {}
                for target_name in target_dict.keys():
                    results['ions'][ion_name][target_name] = {label: [] for label in distance_ranges.keys()}
            else:
                # Ion exists from cache - check if we need to reinitialize for calculation
                for target_name in target_dict.keys():
                    if target_name not in results['ions'][ion_name]:
                        results['ions'][ion_name][target_name] = {label: [] for label in distance_ranges.keys()}
                    else:
                        # Check if data is in dict format (cached) - convert to list for recalculation
                        for label in distance_ranges.keys():
                            if label in results['ions'][ion_name][target_name]:
                                data = results['ions'][ion_name][target_name][label]
                                if isinstance(data, dict):
                                    # Cached data - reinitialize as list for new calculation
                                    results['ions'][ion_name][target_name][label] = []
        
        # Initialize storage for each organic part (organic -> target -> range -> [atom])
        for org_name in organic_parts.keys():
            if org_name not in results['organics']:
                # New organic - create structure
                results['organics'][org_name] = {}
                for target_name in target_dict.keys():
                    if store_per_atom_organics:
                        # Per-atom: organic -> target -> range -> atom -> list
                        results['organics'][org_name][target_name] = {label: {} for label in distance_ranges.keys()}
                    else:
                        # Aggregated: organic -> target -> range -> list
                        results['organics'][org_name][target_name] = {label: [] for label in distance_ranges.keys()}
            else:
                # Organic exists from cache - check if we need to reinitialize
                for target_name in target_dict.keys():
                    if target_name not in results['organics'][org_name]:
                        if store_per_atom_organics:
                            results['organics'][org_name][target_name] = {label: {} for label in distance_ranges.keys()}
                        else:
                            results['organics'][org_name][target_name] = {label: [] for label in distance_ranges.keys()}
                    else:
                        # Check if data is in dict format (cached) - convert to list/dict for recalculation
                        for label in distance_ranges.keys():
                            if label in results['organics'][org_name][target_name]:
                                data = results['organics'][org_name][target_name][label]
                                if isinstance(data, dict) and not store_per_atom_organics:
                                    # Cached aggregated data - reinitialize as list for new calculation
                                    results['organics'][org_name][target_name][label] = []
                                elif isinstance(data, dict) and store_per_atom_organics:
                                    # Might be per-atom cache - check if has 'mean' key (cached stats) or atom names
                                    if data and len(data) > 0:
                                        first_value = list(data.values())[0]
                                        if isinstance(first_value, dict) and 'mean' in first_value:
                                            # Cached per-atom stats - reinitialize as dict of lists
                                            results['organics'][org_name][target_name][label] = {}
                                elif isinstance(data, list) and store_per_atom_organics:
                                    # Cached aggregated but now want per-atom - reinitialize
                                    results['organics'][org_name][target_name][label] = {}
        
        # Check if we need to process trajectory at all
        need_calculation = bool(ion_types) or bool(organic_parts)
        
        if not need_calculation:
            print(f"\n✅ All requested data loaded from cache - no calculation needed")
            self.results['competitive_adsorption'] = results
            return results
        
        # Determine actual number of workers
        total_frames = len(self.u.trajectory[::step])
        if njobs == -1:
            import multiprocessing
            actual_njobs = multiprocessing.cpu_count()
        else:
            actual_njobs = min(njobs, total_frames) if njobs > 1 else 1
        
        # Process trajectory
        if actual_njobs > 1:
            print(f"\n⚡ Parallel mode: Using {actual_njobs} workers for {total_frames} frames")
            
            # Prepare selection strings for workers
            target_sel_dict = {}
            for target_name, target_atoms in target_dict.items():
                # Create selection string from atom indices
                indices_str = ' '.join(map(str, target_atoms.indices))
                target_sel_dict[target_name] = f"index {indices_str}"
            
            ion_sel_dict = {}
            for ion_name in ion_types:
                if ion_name in self.ions:
                    indices_str = ' '.join(map(str, self.ions[ion_name].indices))
                    ion_sel_dict[ion_name] = f"index {indices_str}"
            
            organic_sel_dict = {}
            for org_name, org_atoms in organic_parts.items():
                indices_str = ' '.join(map(str, org_atoms.indices))
                organic_sel_dict[org_name] = f"index {indices_str}"
            
            # Split frames into chunks for each worker
            frames = list(range(0, len(self.u.trajectory), step))
            chunk_size = len(frames) // actual_njobs
            frame_chunks = []
            
            for i in range(actual_njobs):
                start_idx = i * chunk_size
                if i == actual_njobs - 1:
                    # Last worker gets remaining frames
                    end_idx = len(frames)
                else:
                    end_idx = (i + 1) * chunk_size
                
                start_frame = frames[start_idx]
                end_frame = frames[end_idx - 1] + step if end_idx < len(frames) else len(self.u.trajectory)
                frame_chunks.append((start_frame, end_frame, step))
            
            # Submit tasks to ProcessPoolExecutor
            try:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                
                with ProcessPoolExecutor(max_workers=actual_njobs) as executor:
                    futures = []
                    for chunk_idx, frame_range in enumerate(frame_chunks):
                        future = executor.submit(
                            _calculate_competitive_adsorption_worker,
                            self.top,
                            self.traj,
                            frame_range,
                            target_sel_dict,
                            ion_sel_dict,
                            organic_sel_dict,
                            distance_ranges,
                            store_per_atom_organics
                        )
                        futures.append(future)
                    
                    # Collect results from workers
                    worker_results = []
                    for future in as_completed(futures):
                        worker_result = future.result()
                        worker_results.append(worker_result)
                        print(f"  ✓ Worker completed ({len(worker_results)}/{actual_njobs})")
                
                # Merge results from all workers
                print("\n  Merging results from workers...")
                for worker_result in worker_results:
                    # Merge ion results
                    for ion_name in worker_result['ions']:
                        for target_name in worker_result['ions'][ion_name]:
                            for label in worker_result['ions'][ion_name][target_name]:
                                results['ions'][ion_name][target_name][label].extend(
                                    worker_result['ions'][ion_name][target_name][label]
                                )
                    
                    # Merge organic results
                    for org_name in worker_result['organics']:
                        for target_name in worker_result['organics'][org_name]:
                            for label in worker_result['organics'][org_name][target_name]:
                                if store_per_atom_organics:
                                    # Merge per-atom data
                                    for atom_name in worker_result['organics'][org_name][target_name][label]:
                                        if atom_name not in results['organics'][org_name][target_name][label]:
                                            results['organics'][org_name][target_name][label][atom_name] = []
                                        results['organics'][org_name][target_name][label][atom_name].extend(
                                            worker_result['organics'][org_name][target_name][label][atom_name]
                                        )
                                else:
                                    # Merge aggregated data
                                    results['organics'][org_name][target_name][label].extend(
                                        worker_result['organics'][org_name][target_name][label]
                                    )
                
                n_frames = total_frames
                
            except Exception as e:
                print(f"\n⚠️  Parallel execution failed: {e}")
                print("  Falling back to sequential mode...")
                actual_njobs = 1
        
        # Sequential execution (fallback or njobs=1)
        if actual_njobs == 1:
            print(f"\n📋 Sequential mode: Processing {total_frames} frames (step={step})...")
            n_frames = 0
            
            for ts in tqdm(self.u.trajectory[::step], desc="Computing distances"):
                # Loop over each target separately
                for target_name, target_atoms in target_dict.items():
                    target_positions = target_atoms.positions
                    
                    # Analyze ions for this target
                    for ion_name in ion_types:
                        if ion_name not in self.ions:
                            continue
                            
                        ion_atoms = self.ions[ion_name]
                        if len(ion_atoms) == 0:
                            continue
                        
                        ion_positions = ion_atoms.positions
                        
                        # Calculate minimum distance from each ion to any atom in THIS target
                        distances_matrix = cdist(ion_positions, target_positions)
                        min_distances = np.min(distances_matrix, axis=1)  # Min distance per ion
                        
                        # Count ions in each distance range for this target
                        for label, (min_dist, max_dist) in distance_ranges.items():
                            count = np.sum((min_distances >= min_dist) & (min_distances < max_dist))
                            results['ions'][ion_name][target_name][label].append(count)
                    
                    # Analyze organics for this target
                    for org_name, org_atoms in organic_parts.items():
                        if len(org_atoms) == 0:
                            continue
                        
                        org_positions = org_atoms.positions
                        
                        # Calculate minimum distance from each organic atom to any atom in THIS target
                        distances_matrix = cdist(org_positions, target_positions)
                        min_distances = np.min(distances_matrix, axis=1)  # Min distance per atom
                        
                        if store_per_atom_organics:
                            # Store per-atom data
                            for atom_idx, (atom, min_dist) in enumerate(zip(org_atoms, min_distances)):
                                atom_name = atom.name
                                
                                # Count this atom in appropriate distance range
                                for label, (range_min, range_max) in distance_ranges.items():
                                    # Initialize atom entry if first time seeing it
                                    if atom_name not in results['organics'][org_name][target_name][label]:
                                        results['organics'][org_name][target_name][label][atom_name] = []
                                    
                                    # Check if this atom is in range (1 if yes, 0 if no)
                                    in_range = 1 if (min_dist >= range_min and min_dist < range_max) else 0
                                    results['organics'][org_name][target_name][label][atom_name].append(in_range)
                        else:
                            # Aggregate all atoms (original behavior)
                            for label, (min_dist, max_dist) in distance_ranges.items():
                                count = np.sum((min_distances >= min_dist) & (min_distances < max_dist))
                                results['organics'][org_name][target_name][label].append(count)
                
                n_frames += 1
        
        results['metadata']['n_frames'] = max(n_frames, results['metadata']['n_frames'])
        
        # Calculate statistics (mean, std, time_series)
        print("\n  Calculating statistics...")
        
        # Ion statistics: ion -> target -> range (only for newly calculated data)
        for ion_name in results['ions'].keys():
            for target_name in results['ions'][ion_name].keys():
                for label in distance_ranges.keys():
                    # Check if this is raw list data (newly calculated) or dict (cached)
                    data = results['ions'][ion_name][target_name][label]
                    if isinstance(data, list):
                        time_series = np.array(data)
                        results['ions'][ion_name][target_name][label] = {
                            'mean': np.mean(time_series),
                            'std': np.std(time_series),
                            'time_series': time_series
                        }
                    # If it's already a dict, it was loaded from cache - skip
        
        # Organic statistics: organic -> target -> range (only for newly calculated data)
        for org_name in results['organics'].keys():
            for target_name in results['organics'][org_name].keys():
                for label in distance_ranges.keys():
                    data = results['organics'][org_name][target_name][label]
                    
                    if store_per_atom_organics and isinstance(data, dict):
                        # Per-atom mode: process each atom
                        for atom_name in list(data.keys()):
                            atom_data = data[atom_name]
                            if isinstance(atom_data, list) and len(atom_data) > 0:
                                time_series = np.array(atom_data)
                                results['organics'][org_name][target_name][label][atom_name] = {
                                    'mean': np.mean(time_series),
                                    'std': np.std(time_series),
                                    'time_series': time_series
                                }
                    elif isinstance(data, list) and len(data) > 0:
                        # Aggregated mode: original behavior
                        time_series = np.array(data)
                        results['organics'][org_name][target_name][label] = {
                            'mean': np.mean(time_series),
                            'std': np.std(time_series),
                            'time_series': time_series
                        }
                    # If it's already a dict with 'mean' key, it was loaded from cache - skip
        
        # Save to cache if requested (save ions and organics separately)
        if save_cache:
            # Save ion cache if we have ion data
            if ion_cache_file and results['ions']:
                print(f"\n💾 Saving ion results to cache: {ion_cache_file}")
                try:
                    ion_results = {
                        'ions': results['ions'],
                        'metadata': results['metadata']
                    }
                    np.savez(ion_cache_file, results=ion_results)
                    print(f"   ✅ Ion cache saved successfully!")
                except Exception as e:
                    print(f"   ⚠️ Failed to save ion cache: {e}")
            
            # Save organic cache if we have organic data
            if org_cache_file and results['organics']:
                print(f"\n💾 Saving organic results to cache: {org_cache_file}")
                try:
                    org_results = {
                        'organics': results['organics'],
                        'metadata': results['metadata']
                    }
                    np.savez(org_cache_file, results=org_results)
                    print(f"   ✅ Organic cache saved successfully!")
                except Exception as e:
                    print(f"   ⚠️ Failed to save organic cache: {e}")
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY: Competitive Adsorption Results")
        print("="*80)
        
        # Print IONS table
        if results['ions']:
            print("\n📊 IONS:")
            # Collect all data for table
            ion_data = []
            for ion_name in sorted(results['ions'].keys()):
                for target_name in sorted(results['ions'][ion_name].keys()):
                    row = [ion_name, target_name]
                    for label in distance_ranges.keys():
                        mean_val = results['ions'][ion_name][target_name][label]['mean']
                        std_val = results['ions'][ion_name][target_name][label]['std']
                        row.append(f"{mean_val:.2f} ± {std_val:.2f}")
                    ion_data.append(row)
            
            # Print table
            if ion_data:
                # Header
                header = ["Ion", "Target"] + list(distance_ranges.keys())
                col_widths = [max(8, max(len(str(row[i])) for row in [header] + ion_data)) 
                             for i in range(len(header))]
                
                # Print header
                header_str = "  " + " | ".join(f"{header[i]:<{col_widths[i]}}" for i in range(len(header)))
                print(header_str)
                print("  " + "-" * (sum(col_widths) + 3 * (len(header) - 1)))
                
                # Print rows
                for row in ion_data:
                    row_str = "  " + " | ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row)))
                    print(row_str)
        
        # Print ORGANICS table
        if results['organics']:
            print("\n📊 ORGANICS:")
            
            # Check if data is per-atom or aggregated by examining first entry
            first_org = list(results['organics'].keys())[0]
            first_target = list(results['organics'][first_org].keys())[0]
            first_label = list(results['organics'][first_org][first_target].keys())[0]
            first_data = results['organics'][first_org][first_target][first_label]
            
            # Check if per-atom: dict without 'mean' key directly, or dict with atom names as keys
            is_per_atom = isinstance(first_data, dict) and 'mean' not in first_data
            
            if is_per_atom:
                print("  (Per-Atom Analysis)")
                # For per-atom, show each atom separately
                for org_name in sorted(results['organics'].keys()):
                    print(f"\n  {org_name.upper()}:")
                    for target_name in sorted(results['organics'][org_name].keys()):
                        print(f"    Target: {target_name}")
                        
                        # Get all unique atoms across all distance ranges for this org-target pair
                        all_atoms = set()
                        for label in distance_ranges.keys():
                            all_atoms.update(results['organics'][org_name][target_name][label].keys())
                        all_atoms = sorted(all_atoms)
                        
                        # Build table data
                        atom_data = []
                        for atom_name in all_atoms:
                            row = [atom_name]
                            for label in distance_ranges.keys():
                                if atom_name in results['organics'][org_name][target_name][label]:
                                    stats = results['organics'][org_name][target_name][label][atom_name]
                                    row.append(f"{stats['mean']:.2f} ± {stats['std']:.2f}")
                                else:
                                    row.append("N/A")
                            atom_data.append(row)
                        
                        # Print table
                        if atom_data:
                            header = ["Atom"] + list(distance_ranges.keys())
                            col_widths = [max(8, max(len(str(row[i])) for row in [header] + atom_data)) 
                                         for i in range(len(header))]
                            
                            # Print header
                            header_str = "      " + " | ".join(f"{header[i]:<{col_widths[i]}}" for i in range(len(header)))
                            print(header_str)
                            print("      " + "-" * (sum(col_widths) + 3 * (len(header) - 1)))
                            
                            # Print rows
                            for row in atom_data:
                                row_str = "      " + " | ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row)))
                                print(row_str)
                        print()
            else:
                # Aggregated data: original table format
                org_data = []
                for org_name in sorted(results['organics'].keys()):
                    for target_name in sorted(results['organics'][org_name].keys()):
                        row = [org_name, target_name]
                        for label in distance_ranges.keys():
                            data = results['organics'][org_name][target_name][label]
                            # Safety check: make sure this is aggregated data with 'mean' key
                            if isinstance(data, dict) and 'mean' in data:
                                mean_val = data['mean']
                                std_val = data['std']
                                row.append(f"{mean_val:.2f} ± {std_val:.2f}")
                            else:
                                # Shouldn't happen, but handle gracefully
                                row.append("N/A")
                        org_data.append(row)
                
                # Print table
                if org_data:
                    # Header
                    header = ["Organic", "Target"] + list(distance_ranges.keys())
                    col_widths = [max(15, max(len(str(row[i])) for row in [header] + org_data)) 
                                 for i in range(len(header))]
                    
                    # Print header
                    header_str = "  " + " | ".join(f"{header[i]:<{col_widths[i]}}" for i in range(len(header)))
                    print(header_str)
                    print("  " + "-" * (sum(col_widths) + 3 * (len(header) - 1)))
                    
                    # Print rows
                    for row in org_data:
                        row_str = "  " + " | ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row)))
                        print(row_str)
        
        print("\n" + "="*80)
        
        # Merge results with existing data instead of replacing
        if 'competitive_adsorption' in self.results:
            existing = self.results['competitive_adsorption']
            
            # Merge ions: update only the ions that were analyzed this run
            for ion_name in results['ions'].keys():
                if ion_name not in existing['ions']:
                    existing['ions'][ion_name] = {}
                existing['ions'][ion_name].update(results['ions'][ion_name])
            
            # Merge organics: update only the organics that were analyzed this run
            for org_name in results['organics'].keys():
                if org_name not in existing['organics']:
                    existing['organics'][org_name] = {}
                existing['organics'][org_name].update(results['organics'][org_name])
            
            # Update metadata (keep max frame count)
            existing['metadata']['n_frames'] = max(
                existing['metadata'].get('n_frames', 0),
                results['metadata']['n_frames']
            )
            existing['metadata']['distance_ranges'] = results['metadata']['distance_ranges']
            # Update timestamp if present (may not exist in older cached data)
            if 'last_updated' in results['metadata']:
                existing['metadata']['last_updated'] = results['metadata']['last_updated']
        else:
            # First time running - just store results directly
            self.results['competitive_adsorption'] = results
        
        return self.results['competitive_adsorption']
    
    def analyze_organic_conformations(self):
        """
        Analyze organic molecule conformations and orientations.
        
        Returns:
        --------
        dict : Organic conformation analysis results
        """
        print("Analyzing organic conformations...")
        
        results = {}
        
        for org_name, org_atoms in self.organics.items():
            if len(org_atoms) == 0:
                continue
                
            print(f"  Analyzing {org_name} conformations")
            
            # Get bonds/angles if topology information is available
            try:
                # Calculate radius of gyration
                rg_values = []
                # Calculate aspect ratios
                aspect_ratios = []
                # Calculate principal axes
                principal_axes = []
                
                for ts in self.u.trajectory:
                    positions = org_atoms.positions
                    
                    # Radius of gyration
                    center_of_mass = np.mean(positions, axis=0)
                    rg = np.sqrt(np.mean(np.sum((positions - center_of_mass)**2, axis=1)))
                    rg_values.append(rg)
                    
                    # Principal component analysis for shape
                    centered_positions = positions - center_of_mass
                    cov_matrix = np.cov(centered_positions.T)
                    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
                    eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
                    
                    # Aspect ratio (largest/smallest eigenvalue)
                    if eigenvals[-1] > 0:
                        aspect_ratio = eigenvals[0] / eigenvals[-1]
                        aspect_ratios.append(aspect_ratio)
                    
                    principal_axes.append(eigenvecs)
                
                results[org_name] = {
                    'radius_of_gyration': {
                        'mean': np.mean(rg_values),
                        'std': np.std(rg_values),
                        'time_series': np.array(rg_values)
                    },
                    'aspect_ratio': {
                        'mean': np.mean(aspect_ratios),
                        'std': np.std(aspect_ratios),
                        'time_series': np.array(aspect_ratios)
                    },
                    'principal_axes': np.array(principal_axes)
                }
                
            except Exception as e:
                print(f"    Warning: Could not analyze {org_name} conformations: {e}")
        
        self.results['organic_conformations'] = results
        return results
    
    def analyze_three_component_bridges(self, bridge_cutoff=4.0):
        """
        Analyze three-component bridges (clay-ion-organic, clay-water-ion, etc.).
        
        Parameters:
        -----------
        bridge_cutoff : float
            Maximum distance for bridge formation (Angstrom)
        
        Returns:
        --------
        dict : Bridge analysis results
        """
        print("Analyzing three-component bridges...")
        
        bridge_types = {
            'clay-ion-organic': [],
            'clay-water-ion': [],
            'clay-water-organic': [],
            'ion-water-organic': []
        }
        
        for ts in self.u.trajectory:
            frame_bridges = {key: 0 for key in bridge_types.keys()}
            
            # Clay-ion-organic bridges
            for ion_name, ion_atoms in self.ions.items():
                for org_name, org_atoms in self.organics.items():
                    if len(ion_atoms) > 0 and len(org_atoms) > 0:
                        # Find ions near clay
                        ion_clay_dist = cdist(ion_atoms.positions, self.clay.positions)
                        ions_near_clay = np.any(ion_clay_dist < bridge_cutoff, axis=1)
                        
                        # Find organics near the same clay atoms that ions are near
                        for i, ion_near in enumerate(ions_near_clay):
                            if ion_near:
                                clay_indices = np.where(ion_clay_dist[i] < bridge_cutoff)[0]
                                clay_positions = self.clay.positions[clay_indices]
                                
                                org_clay_dist = cdist(org_atoms.positions, clay_positions)
                                if np.any(org_clay_dist < bridge_cutoff):
                                    # Check if ion and organic are also close to each other
                                    ion_org_dist = cdist([ion_atoms.positions[i]], org_atoms.positions)
                                    if np.any(ion_org_dist < bridge_cutoff * 1.5):
                                        frame_bridges['clay-ion-organic'] += 1
            
            # Clay-water-ion bridges
            for ion_name, ion_atoms in self.ions.items():
                if len(ion_atoms) > 0:
                    water_positions = self.water.positions[::3]  # Assuming water oxygens every 3rd atom
                    
                    # Find water molecules that bridge clay and ions
                    water_clay_dist = cdist(water_positions, self.clay.positions)
                    water_ion_dist = cdist(water_positions, ion_atoms.positions)
                    
                    water_near_clay = np.any(water_clay_dist < bridge_cutoff, axis=1)
                    water_near_ion = np.any(water_ion_dist < bridge_cutoff, axis=1)
                    
                    bridges = water_near_clay & water_near_ion
                    frame_bridges['clay-water-ion'] += np.sum(bridges)
            
            # Clay-water-organic bridges
            for org_name, org_atoms in self.organics.items():
                if len(org_atoms) > 0:
                    water_positions = self.water.positions[::3]
                    
                    water_clay_dist = cdist(water_positions, self.clay.positions)
                    water_org_dist = cdist(water_positions, org_atoms.positions)
                    
                    water_near_clay = np.any(water_clay_dist < bridge_cutoff, axis=1)
                    water_near_org = np.any(water_org_dist < bridge_cutoff, axis=1)
                    
                    bridges = water_near_clay & water_near_org
                    frame_bridges['clay-water-organic'] += np.sum(bridges)
            
            # Ion-water-organic bridges
            for ion_name, ion_atoms in self.ions.items():
                for org_name, org_atoms in self.organics.items():
                    if len(ion_atoms) > 0 and len(org_atoms) > 0:
                        water_positions = self.water.positions[::3]
                        
                        water_ion_dist = cdist(water_positions, ion_atoms.positions)
                        water_org_dist = cdist(water_positions, org_atoms.positions)
                        
                        water_near_ion = np.any(water_ion_dist < bridge_cutoff, axis=1)
                        water_near_org = np.any(water_org_dist < bridge_cutoff, axis=1)
                        
                        bridges = water_near_ion & water_near_org
                        frame_bridges['ion-water-organic'] += np.sum(bridges)
            
            for bridge_type in bridge_types:
                bridge_types[bridge_type].append(frame_bridges[bridge_type])
        
        # Calculate statistics
        results = {}
        for bridge_type, counts in bridge_types.items():
            counts = np.array(counts)
            results[bridge_type] = {
                'mean': np.mean(counts),
                'std': np.std(counts),
                'max': np.max(counts),
                'time_series': counts
            }
        
        self.results['three_component_bridges'] = results
        return results
    
    def analyze_hydration_shell_competition(self, shell_cutoffs=[3.5, 5.0]):
        """
        Analyze how organics and ions compete for water coordination.
        
        Parameters:
        -----------
        shell_cutoffs : list
            Distance cutoffs for first and second hydration shells
        
        Returns:
        --------
        dict : Hydration shell competition results
        """
        print("Analyzing hydration shell competition...")
        
        results = {
            'ion_hydration': {},
            'organic_hydration': {},
            'water_coordination_competition': {}
        }
        
        water_oxygens = self.water.select_atoms('name O* or name OW')
        
        for cutoff in shell_cutoffs:
            shell_name = f'shell_{cutoff}A'
            results['ion_hydration'][shell_name] = {}
            results['organic_hydration'][shell_name] = {}
            
            for ion_name, ion_atoms in self.ions.items():
                if len(ion_atoms) == 0:
                    continue
                    
                hydration_numbers = []
                
                for ts in self.u.trajectory:
                    distances = cdist(ion_atoms.positions, water_oxygens.positions)
                    coordinated_waters = np.sum(distances < cutoff, axis=1)
                    hydration_numbers.append(np.mean(coordinated_waters))
                
                results['ion_hydration'][shell_name][ion_name] = {
                    'mean': np.mean(hydration_numbers),
                    'std': np.std(hydration_numbers),
                    'time_series': np.array(hydration_numbers)
                }
            
            for org_name, org_atoms in self.organics.items():
                if len(org_atoms) == 0:
                    continue
                    
                hydration_numbers = []
                
                for ts in self.u.trajectory:
                    distances = cdist(org_atoms.positions, water_oxygens.positions)
                    coordinated_waters = np.sum(distances < cutoff, axis=1)
                    hydration_numbers.append(np.mean(coordinated_waters))
                
                results['organic_hydration'][shell_name][org_name] = {
                    'mean': np.mean(hydration_numbers),
                    'std': np.std(hydration_numbers),
                    'time_series': np.array(hydration_numbers)
                }
        
        self.results['hydration_shell_competition'] = results
        return results
    
    def analyze_stratified_adsorption(self, z_direction='z', bin_size=0.5):
        """
        Analyze stratified (layered) adsorption of components.
        
        Parameters:
        -----------
        z_direction : str
            Direction perpendicular to clay surface ('x', 'y', or 'z')
        bin_size : float
            Bin size for density profiles (Angstrom)
        
        Returns:
        --------
        dict : Stratified adsorption analysis results
        """
        print("Analyzing stratified adsorption...")
        
        # Determine clay surface position
        clay_positions = []
        for ts in self.u.trajectory:
            clay_positions.append(self.clay.positions)
        
        clay_positions = np.concatenate(clay_positions, axis=0)
        
        if z_direction == 'x':
            coord_index = 0
        elif z_direction == 'y':
            coord_index = 1
        else:
            coord_index = 2
        
        clay_surface_z = np.mean(clay_positions[:, coord_index])
        
        # Define bins
        all_positions = []
        for component in [self.clay] + list(self.ions.values()) + list(self.organics.values()) + [self.water]:
            for ts in self.u.trajectory:
                if len(component) > 0:
                    all_positions.extend(component.positions[:, coord_index])
        
        z_min = min(all_positions)
        z_max = max(all_positions)
        bins = np.arange(z_min, z_max + bin_size, bin_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate density profiles
        density_profiles = {}
        
        # Clay density
        clay_densities = []
        for ts in self.u.trajectory:
            hist, _ = np.histogram(self.clay.positions[:, coord_index], bins=bins)
            clay_densities.append(hist)
        density_profiles['clay'] = np.mean(clay_densities, axis=0)
        
        # Ion densities
        for ion_name, ion_atoms in self.ions.items():
            if len(ion_atoms) == 0:
                continue
            ion_densities = []
            for ts in self.u.trajectory:
                hist, _ = np.histogram(ion_atoms.positions[:, coord_index], bins=bins)
                ion_densities.append(hist)
            density_profiles[ion_name] = np.mean(ion_densities, axis=0)
        
        # Organic densities
        for org_name, org_atoms in self.organics.items():
            if len(org_atoms) == 0:
                continue
            org_densities = []
            for ts in self.u.trajectory:
                hist, _ = np.histogram(org_atoms.positions[:, coord_index], bins=bins)
                org_densities.append(hist)
            density_profiles[org_name] = np.mean(org_densities, axis=0)
        
        # Water density
        water_densities = []
        water_oxygens = self.water.select_atoms('name O* or name OW')
        for ts in self.u.trajectory:
            hist, _ = np.histogram(water_oxygens.positions[:, coord_index], bins=bins)
            water_densities.append(hist)
        density_profiles['water'] = np.mean(water_densities, axis=0)
        
        results = {
            'bin_centers': bin_centers,
            'clay_surface_position': clay_surface_z,
            'density_profiles': density_profiles,
            'relative_positions': bin_centers - clay_surface_z
        }
        
        self.results['stratified_adsorption'] = results
        return results
    
    def analyze_exchange_kinetics(self, distance_cutoff=4.0, time_step=None):
        """
        Analyze adsorption-desorption kinetics for each component.
        
        Parameters:
        -----------
        distance_cutoff : float
            Distance cutoff for surface binding
        time_step : float, optional
            Time step between frames (ps)
        
        Returns:
        --------
        dict : Exchange kinetics analysis results
        """
        print("Analyzing exchange kinetics...")
        
        if time_step is None:
            time_step = self.u.trajectory.dt
        
        results = {
            'residence_times': {},
            'exchange_rates': {},
            'binding_events': {}
        }
        
        # Analyze ion exchange kinetics
        for ion_name, ion_atoms in self.ions.items():
            if len(ion_atoms) == 0:
                continue
                
            print(f"  Analyzing {ion_name} exchange kinetics")
            
            bound_states = []  # Track which ions are bound at each frame
            
            for ts in self.u.trajectory:
                distances = cdist(ion_atoms.positions, self.clay.positions)
                is_bound = np.any(distances < distance_cutoff, axis=1)
                bound_states.append(is_bound)
            
            bound_states = np.array(bound_states)
            
            # Calculate residence times
            residence_times = []
            for ion_idx in range(len(ion_atoms)):
                ion_trajectory = bound_states[:, ion_idx]
                
                # Find bound periods
                bound_periods = []
                in_bound_period = False
                start_time = 0
                
                for frame, is_bound in enumerate(ion_trajectory):
                    if is_bound and not in_bound_period:
                        start_time = frame
                        in_bound_period = True
                    elif not is_bound and in_bound_period:
                        bound_periods.append(frame - start_time)
                        in_bound_period = False
                
                # Handle case where simulation ends while ion is still bound
                if in_bound_period:
                    bound_periods.append(len(ion_trajectory) - start_time)
                
                residence_times.extend(bound_periods)
            
            if residence_times:
                residence_times = np.array(residence_times) * time_step
                results['residence_times'][ion_name] = {
                    'mean': np.mean(residence_times),
                    'std': np.std(residence_times),
                    'distribution': residence_times
                }
                
                # Calculate exchange rate (events per unit time)
                total_time = len(bound_states) * time_step
                n_events = len(residence_times)
                results['exchange_rates'][ion_name] = n_events / total_time
        
        # Analyze organic exchange kinetics
        for org_name, org_atoms in self.organics.items():
            if len(org_atoms) == 0:
                continue
                
            print(f"  Analyzing {org_name} exchange kinetics")
            
            bound_states = []
            
            for ts in self.u.trajectory:
                distances = cdist(org_atoms.positions, self.clay.positions)
                is_bound = np.any(distances < distance_cutoff, axis=1)
                bound_states.append(is_bound)
            
            bound_states = np.array(bound_states)
            
            # Similar analysis as for ions
            residence_times = []
            for org_idx in range(len(org_atoms)):
                org_trajectory = bound_states[:, org_idx]
                
                bound_periods = []
                in_bound_period = False
                start_time = 0
                
                for frame, is_bound in enumerate(org_trajectory):
                    if is_bound and not in_bound_period:
                        start_time = frame
                        in_bound_period = True
                    elif not is_bound and in_bound_period:
                        bound_periods.append(frame - start_time)
                        in_bound_period = False
                
                if in_bound_period:
                    bound_periods.append(len(org_trajectory) - start_time)
                
                residence_times.extend(bound_periods)
            
            if residence_times:
                residence_times = np.array(residence_times) * time_step
                results['residence_times'][org_name] = {
                    'mean': np.mean(residence_times),
                    'std': np.std(residence_times),
                    'distribution': residence_times
                }
                
                total_time = len(bound_states) * time_step
                n_events = len(residence_times)
                results['exchange_rates'][org_name] = n_events / total_time
        
        self.results['exchange_kinetics'] = results
        return results
    
    def calculate_selectivity_coefficients(self):
        """
        Calculate selectivity coefficients for competitive binding.
        
        Returns:
        --------
        dict : Selectivity coefficient results
        """
        print("Calculating selectivity coefficients...")
        
        if 'competitive_adsorption' not in self.results:
            print("  Running competitive adsorption analysis first...")
            self.analyze_competitive_adsorption()
        
        adsorption_data = self.results['competitive_adsorption']
        
        selectivity_coefficients = {}
        
        # Calculate ion vs ion selectivity
        ion_names = list(adsorption_data['ion_surface_contacts'].keys())
        for i, ion1 in enumerate(ion_names):
            for j, ion2 in enumerate(ion_names):
                if i < j:
                    contact1 = adsorption_data['ion_surface_contacts'][ion1]['mean']
                    contact2 = adsorption_data['ion_surface_contacts'][ion2]['mean']
                    
                    if contact2 > 0:
                        selectivity = contact1 / contact2
                        selectivity_coefficients[f'{ion1}_vs_{ion2}'] = selectivity
        
        # Calculate organic vs organic selectivity
        org_names = list(adsorption_data['organic_surface_contacts'].keys())
        for i, org1 in enumerate(org_names):
            for j, org2 in enumerate(org_names):
                if i < j:
                    contact1 = adsorption_data['organic_surface_contacts'][org1]['mean']
                    contact2 = adsorption_data['organic_surface_contacts'][org2]['mean']
                    
                    if contact2 > 0:
                        selectivity = contact1 / contact2
                        selectivity_coefficients[f'{org1}_vs_{org2}'] = selectivity
        
        # Calculate ion vs organic selectivity
        for ion_name in ion_names:
            for org_name in org_names:
                contact_ion = adsorption_data['ion_surface_contacts'][ion_name]['mean']
                contact_org = adsorption_data['organic_surface_contacts'][org_name]['mean']
                
                if contact_org > 0:
                    selectivity = contact_ion / contact_org
                    selectivity_coefficients[f'{ion_name}_vs_{org_name}'] = selectivity
        
        self.results['selectivity_coefficients'] = selectivity_coefficients
        return selectivity_coefficients
    
    def analyze_electrical_double_layer_complete(self, clay_surface_sel, 
                                                 z_bin_width=0.5,
                                                 charge_dict=None,
                                                 dielectric_constant=78.0,
                                                 temperature=300.0,
                                                 bulk_reference_distance=None,
                                                 bulk_span=2.0,
                                                 identify_stern_layer=True,
                                                 calculate_debye_length=True,
                                                 compare_gouy_chapman=True,
                                                 classify_adsorption_modes=True,
                                                 use_manual_peaks=False,
                                                 step=1,
                                                 njobs=1,
                                                 save_cache=True,
                                                 cache_file=None,
                                                 force_rerun=False,
                                                 center_box=True,
                                                 clay_surface_charge=None,
                                                 save_charge_profiles_csv=False,
                                                 include_clay_charge_density=False,
                                                 include_organic_charge_density=False,
                                                 summary_report=False,
                                                 analysis_z_range=None):
        """
        Comprehensive electrical double layer (EDL) analysis near clay surface.
        
        Calculates all key EDL properties including:
        - Charge density profile ρ(z)
        - Electrostatic potential ψ(z) via Poisson equation
        - Electric field E(z) = -dψ/dz
        - Debye screening length λ_D
        - Stern layer thickness (inner vs outer Helmholtz planes)
        - Surface charge density σ_surface
        - Ion adsorption modes (inner-sphere, outer-sphere, diffuse layer)
        - Comparison with Gouy-Chapman theory
        
        Parameters
        ----------
        clay_surface_sel : AtomGroup
            Selection of clay surface atoms for reference positioning
        z_bin_width : float, default=0.5
            Bin width for z-direction discretization (Å)
        charge_dict : dict, optional
            Dictionary of ion charges in elementary charge units
            Example: {'NA': 1.0, 'MG': 2.0, 'CL': -1.0}
            If None, uses common defaults
        dielectric_constant : float, default=78.0
            Relative permittivity of water (ε_r)
        temperature : float, default=300.0
            System temperature in Kelvin
        bulk_reference_distance : float or None, default=None
            If None (default): Uses automatic detection based on system geometry.
              - Two-surface centered systems: entire interlayer with edge buffers
              - Single-surface systems: region away from surface
            If float: Explicitly sets the z-coordinate center of the bulk region (Å).
              Bulk region will span from (bulk_reference_distance - bulk_span) to
              (bulk_reference_distance + bulk_span). Example: bulk_reference_distance=0,
              bulk_span=5 → bulk from -5 to +5 Å.
        bulk_span : float, default=2.0
            For automatic mode: Edge buffer distance (Å) to exclude boundary artifacts.
            For manual mode: Half-width of bulk region (Å) around bulk_reference_distance.
            Example: bulk_span=5 with bulk_reference_distance=0 → bulk from -5 to +5 Å.
        identify_stern_layer : bool, default=True
            Calculate Stern layer thickness from ion density minima
        calculate_debye_length : bool, default=True
            Calculate Debye screening length from exponential fit
        compare_gouy_chapman : bool, default=True
            Compare MD results with Gouy-Chapman theory predictions
        classify_adsorption_modes : bool, default=True
            Classify ions into inner-sphere, outer-sphere, and diffuse layer
        use_manual_peaks : bool, default=False
            If True, use manual peak positions from analyze_ion_peaks_manual() for IHP/OHP.
            If False, use automatic peak finding from ion density profiles.
            Only takes effect if manual peaks have been previously set via analyze_ion_peaks_manual().
        step : int, default=1
            Analyze every nth frame
        njobs : int, default=1
            Number of parallel jobs for frame processing. Use -1 for all CPUs.
        save_cache : bool, default=True
            Save results to cache file for faster subsequent analysis
        cache_file : str, optional
            Custom cache filename. If None, auto-generates from parameters.
        force_rerun : bool, default=False
            Force recalculation even if cache exists
        center_box : bool, default=True
            If True, center the simulation box so z=0 is at box center.
            This allows proper use of 'prop z > 0' and 'prop z < 0' selections
            for top and bottom surfaces.
        clay_surface_charge : float, optional
            Total charge of the clay surface in elementary charge units (e).
            If None, calculated from clay_surface_sel atom charges.
            For charged clays (e.g., montmorillonite), this should be negative.
            Example: For 0.01 e/Å² and 3000 Å² area, use -30.0 e
        save_charge_profiles_csv : bool, default=False
            If True, saves charge density profiles to 'charge_density_profiles.csv'
            in the current working directory. CSV contains z-position, charge density
            and number density for each ion type, plus total charge density.
        include_clay_charge_density : bool, default=False
            If True, calculates and reports clay charge density alongside mobile ion
            charge density. Shows hierarchical breakdown: mobile ions, clay (static),
            and total system charge. Useful for comparing static vs mobile screening.
            Note: EDL analysis (Poisson, Stern, Debye) still uses mobile charge only.
        include_organic_charge_density : bool, default=False
            If True, calculates per-atom partial-charge density profile of all organic
            molecules (from self.organics) and includes it in the Poisson equation for
            the electrostatic potential. This corrects the charge neutrality check when
            charged organics (e.g., CIP at net -1e) are present alongside the mobile
            salt ions. Uses the same trajectory step as ions (full sampling).
            Requires partial charges available in the topology (e.g., from a .tpr file).
            Note: Stern layer, Debye length and adsorption-mode analyses still use
            ion densities only; only the Poisson potential is affected.
        summary_report : bool, default=False
            If True, prints a formatted summary table of all EDL analysis results
            at the end, including surface properties, Stern layer, Debye length,
            bulk concentrations, and ion adsorption modes in easy-to-read tables.
        analysis_z_range : tuple of (float, float), optional
            (z_min, z_max) range in Å to restrict ion analysis. Only ions within this
            z-range will be included in density profiles.
            If None (default), automatically determines range based on clay_surface_sel position:
            - For centered box (center_box=True):
              * Top surface (z > 5): analyzes ions from 0 to z_max
              * Bottom surface (z < -5): analyzes ions from z_min to 0
            - For non-centered box: analyzes ±30 Å from surface
            Manual override useful for custom analysis regions.
        
        Returns
        -------
        dict
            Comprehensive EDL analysis results with keys:
            - 'z_centers': z-position array (Å)
            - 'charge_density': ρ(z) in e/Å³ (mobile ions only)
            - 'electrostatic_potential': ψ(z) in kT/e
            - 'electric_field': E(z) in kT/e/Å
            - 'ion_densities': {ion_type: density(z)} in ions/Å³
            - 'surface_position': z-coordinate of clay surface (Å)
            - 'surface_charge_density': σ_surface in e/Å²
            - 'stern_layer': dict with IHP, OHP positions and thicknesses
            - 'debye_length': dict with theoretical and fitted values (Å)
            - 'adsorption_modes': ion counts in each region (if requested)
            - 'gouy_chapman_comparison': theory vs MD (if requested)
            - 'clay_charge_density': ρ_clay(z) in e/Å³ (if include_clay_charge_density=True)
            - 'clay_number_density': clay atom density (if include_clay_charge_density=True)
            - 'organic_charge_density': ρ_org(z) in e/Å³ (if include_organic_charge_density=True)
            - 'organic_number_density': organic atom density (if include_organic_charge_density=True)
            - 'total_charge_density': ρ_mobile [+ ρ_clay] [+ ρ_organic] (all enabled components)
            - 'metadata': calculation parameters and diagnostics
        """
        
        print("\n" + "="*80)
        print("ELECTRICAL DOUBLE LAYER ANALYSIS")
        print("="*80)
        
        # Default charge dictionary
        if charge_dict is None:
            charge_dict = {
                'NA': 1.0, 'K': 1.0, 'LI': 1.0, 'RB': 1.0, 'CS': 1.0,
                'MG': 2.0, 'CA': 2.0, 'SR': 2.0, 'BA': 2.0,
                'CL': -1.0, 'BR': -1.0, 'I': -1.0, 'F': -1.0
            }
        
        # Get box dimensions (needed for z_offset calculation)
        box_x = self.u.dimensions[0]
        box_y = self.u.dimensions[1]
        box_z = self.u.dimensions[2]
        
        # Determine z-offset for centering
        if center_box:
            z_offset = box_z / 2
        else:
            z_offset = 0.0
        
        # Quickly determine surface position for cache key (need this before cache check)
        # Sample first frame only for cache differentiation
        self.u.trajectory[0]
        clay_z_raw = clay_surface_sel.positions[:, 2]
        clay_z_centered = clay_z_raw - z_offset
        surface_position_approx = np.mean(clay_z_centered)
        
        # === CACHE SETUP ===
        # Generate cache filename based on parameters
        if cache_file is None:
            # Create hash from key parameters including surface position to differentiate top/bottom
            z_range_str = f"zr{analysis_z_range[0]:.0f}to{analysis_z_range[1]:.0f}" if analysis_z_range else f"surf{surface_position_approx:.0f}"
            cache_params = (
                f"zbw{z_bin_width}_step{step}_T{temperature}_"
                f"eps{dielectric_constant}_center{center_box}_{z_range_str}_"
                f"ions{'_'.join(sorted(self.ions.keys()))}_"
                f"orgCharge{include_organic_charge_density}"
            )
            cache_hash = hashlib.md5(cache_params.encode()).hexdigest()[:12]
            cache_dir = os.path.join(os.path.dirname(self.traj), '.edl_cache')
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"edl_analysis_{cache_hash}.npz")
        
        # Check cache
        if not force_rerun and os.path.exists(cache_file):
            try:
                print(f"\n📂 Loading cached EDL analysis from: {cache_file}")
                cached_data = np.load(cache_file, allow_pickle=True)
                
                # Reconstruct results dictionary
                results = {
                    'z_centers': cached_data['z_centers'],
                    'charge_density': cached_data['charge_density'],
                    'electrostatic_potential': cached_data['electrostatic_potential'],
                    'electric_field': cached_data['electric_field'],
                    'ion_densities': cached_data['ion_densities'].item(),
                    'surface_position': float(cached_data['surface_position']),
                    'surface_position_std': float(cached_data['surface_position_std']),
                    'surface_charge_density': float(cached_data['surface_charge_density']),
                    'metadata': cached_data['metadata'].item()
                }
                
                # Load optional analyses if present
                if 'stern_layer' in cached_data:
                    results['stern_layer'] = cached_data['stern_layer'].item()
                if 'debye_length' in cached_data:
                    results['debye_length'] = cached_data['debye_length'].item()
                if 'adsorption_modes' in cached_data:
                    results['adsorption_modes'] = cached_data['adsorption_modes'].item()
                if 'gouy_chapman_comparison' in cached_data:
                    results['gouy_chapman_comparison'] = cached_data['gouy_chapman_comparison'].item()
                if 'organic_charge_density' in cached_data:
                    results['organic_charge_density'] = cached_data['organic_charge_density']
                if 'organic_number_density' in cached_data:
                    results['organic_number_density'] = cached_data['organic_number_density']
                if 'total_charge_density' in cached_data:
                    results['total_charge_density'] = cached_data['total_charge_density']
                
                self.results['edl_analysis'] = results
                print("✅ Cached data loaded successfully!")
                return results
                
            except Exception as e:
                print(f"⚠️  Cache load failed ({e}), recalculating...")
        
        # Physical constants
        k_B = 1.381e-23  # J/K
        e = 1.602e-19    # C
        epsilon_0 = 8.854e-12  # F/m
        N_A = 6.022e23   # Avogadro's number
        
        # box_x, box_y, box_z, z_offset already calculated before cache check
        # (moved earlier to enable surface-specific cache keys)
        box_volume = box_x * box_y * box_z  # Å³
        
        if center_box:
            print(f"  Centering box: z_offset = {z_offset:.2f} Å")
        
        # Determine z-range and create bins
        if center_box:
            # Center box around z=0
            half_box = box_z / 2
            z_min = -half_box
            z_max = half_box
        else:
            # Use original coordinate system
            z_min = 0
            z_max = box_z
        
        z_bins = np.arange(z_min, z_max + z_bin_width, z_bin_width)
        z_centers = (z_bins[:-1] + z_bins[1:]) / 2
        n_bins = len(z_centers)
        
        print(f"\nSystem configuration:")
        print(f"  Box dimensions: {box_x:.1f} × {box_y:.1f} × {box_z:.1f} Å")
        print(f"  Box centering: {'Enabled (z=0 at center)' if center_box else 'Disabled (original coordinates)'}")
        print(f"  Z-range: {z_min:.2f} to {z_max:.2f} Å")
        print(f"  Z-bin width: {z_bin_width} Å")
        print(f"  Number of z-bins: {n_bins}")
        print(f"  Temperature: {temperature} K")
        print(f"  Dielectric constant: {dielectric_constant}")
        
        # Print clay surface selection info
        n_clay_atoms = len(clay_surface_sel)
        print(f"\nClay surface selection:")
        print(f"  Number of atoms: {n_clay_atoms}")
        if n_clay_atoms == 0:
            raise ValueError("❌ No atoms found in clay_surface_sel! Check your selection string.")
        
        # Determine clay surface position (average z-position)
        clay_surface_positions = []
        for ts in self.u.trajectory[::step]:
            clay_z_raw = clay_surface_sel.positions[:, 2]
            # Apply centering offset
            clay_z_centered = clay_z_raw - z_offset
            clay_surface_positions.append(np.mean(clay_z_centered))
        
        surface_position = np.mean(clay_surface_positions)
        surface_std = np.std(clay_surface_positions)
        
        print(f"\nClay surface analysis:")
        print(f"  Average z-position: {surface_position:.2f} ± {surface_std:.2f} Å")
        
        # Print z-range of clay atoms for verification
        self.u.trajectory[0]  # Go to first frame
        clay_z_positions = clay_surface_sel.positions[:, 2] - z_offset
        z_min_clay = np.min(clay_z_positions)
        z_max_clay = np.max(clay_z_positions)
        print(f"  Z-range: {z_min_clay:.2f} to {z_max_clay:.2f} Å (spread: {z_max_clay - z_min_clay:.2f} Å)")
        
        # Use manual z-range if provided, otherwise analyze entire box
        if analysis_z_range is not None:
            print(f"  Manual z-range override: analyzing ions from {analysis_z_range[0]:.2f} to {analysis_z_range[1]:.2f} Å")
        else:
            # Analyze entire box (both surface EDLs included)
            print(f"  Analyzing full z-range: {z_min:.1f} to {z_max:.1f} Å (includes both surface interaction regions)")
        
        # Warn if selection might span both surfaces
        spread = z_max_clay - z_min_clay
        if spread > (box_z / 3):
            print(f"  ⚠️  WARNING: Clay selection spans {spread:.1f} Å")
            print(f"     This might include BOTH top and bottom surfaces!")
            if center_box:
                print(f"     With centering enabled, use 'prop z > 0' (top) or 'prop z < 0' (bottom).")
            else:
                print(f"     Enable center_box=True to use 'prop z > 0' / 'prop z < 0' selections.")
        
        # Initialize storage arrays
        ion_densities = {ion_name: np.zeros(n_bins) for ion_name in self.ions.keys()}
        charge_density = np.zeros(n_bins)
        
        # Calculate ion density and charge density profiles
        print(f"\nCalculating ion density profiles...")
        
        # === PARALLEL PROCESSING ===
        total_frames = len(self.u.trajectory[::step])
        
        # Determine number of jobs
        if njobs == -1:
            import multiprocessing
            njobs = multiprocessing.cpu_count()
        
        # Use parallel processing if njobs > 1
        if njobs > 1 and total_frames > njobs:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            print(f"  Using {njobs} parallel workers for {total_frames} frames")
            
            # Create ion selection dictionary for workers
            ion_sel_dict = {}
            for ion_name, ion_atoms in self.ions.items():
                # Skip empty ion groups
                if len(ion_atoms) == 0:
                    continue
                # Reconstruct selection using atom indices (most reliable)
                indices_str = ' '.join(map(str, ion_atoms.indices))
                ion_sel_dict[ion_name] = f"index {indices_str}"
            
            # Split frames into chunks
            frame_indices = list(range(0, len(self.u.trajectory), step))
            chunk_size = max(1, len(frame_indices) // njobs)
            frame_chunks = []
            
            for i in range(0, len(frame_indices), chunk_size):
                chunk_indices = frame_indices[i:i+chunk_size]
                if len(chunk_indices) > 0:
                    start_frame = chunk_indices[0]
                    end_frame = chunk_indices[-1] + step
                    frame_chunks.append((start_frame, end_frame, step))
            
            # Execute workers in parallel
            with ProcessPoolExecutor(max_workers=njobs) as executor:
                # Submit all tasks
                futures = []
                for chunk in frame_chunks:
                    future = executor.submit(
                        _calculate_edl_density_worker,
                        self.top, self.traj, chunk,
                        ion_sel_dict, charge_dict, z_bins,
                        box_x, box_y, z_bin_width, z_offset,
                        analysis_z_range
                    )
                    futures.append(future)
                
                # Collect results
                completed = 0
                n_frames = 0
                for future in as_completed(futures):
                    result = future.result()
                    
                    # Accumulate densities
                    for ion_name in result['ion_densities'].keys():
                        ion_densities[ion_name] += result['ion_densities'][ion_name]
                    charge_density += result['charge_density']
                    n_frames += result['n_frames']
                    
                    completed += 1
                    print(f"    Worker {completed}/{len(futures)} completed ({result['n_frames']} frames)")
            
            # Average over frames
            for ion_name in ion_densities.keys():
                ion_densities[ion_name] /= n_frames
            charge_density /= n_frames
            
        else:
            # Serial processing
            if njobs > 1:
                print(f"  Using serial processing (insufficient frames for parallelization)")
            
            n_frames = 0
            for ts in self.u.trajectory[::step]:
                for ion_name, ion_atoms in self.ions.items():
                    if len(ion_atoms) == 0:
                        continue
                    
                    # Get ion z-positions and apply centering offset
                    ion_z = ion_atoms.positions[:, 2] - z_offset
                    
                    # Apply z-range filter if specified
                    if analysis_z_range is not None:
                        z_min_analysis, z_max_analysis = analysis_z_range
                        mask = (ion_z >= z_min_analysis) & (ion_z <= z_max_analysis)
                        ion_z = ion_z[mask]
                    
                    # Bin ions
                    hist, _ = np.histogram(ion_z, bins=z_bins)
                    
                    # Convert to number density (ions/Å³)
                    bin_volume = box_x * box_y * z_bin_width
                    density = hist / bin_volume
                    
                    ion_densities[ion_name] += density
                    
                    # Add to charge density
                    ion_charge = self._get_ion_charge(ion_name, charge_dict)
                    charge_density += density * ion_charge
                
                n_frames += 1
            
            # Average over frames
            for ion_name in ion_densities.keys():
                ion_densities[ion_name] /= n_frames
            charge_density /= n_frames
        
        print(f"  Processed {n_frames} frames")
        
        # === CLAY CHARGE DENSITY (OPTIONAL) ===
        clay_charge_density = None
        clay_number_density = None
        total_clay_charge = None
        
        if include_clay_charge_density:
            print(f"\nCalculating clay charge density profile...")
            
            # Method 1: Use ALL clay atoms from initialization (preferred)
            all_clay_atoms = None
            if hasattr(self, 'clay_sel') and self.clay_sel is not None:
                all_clay_atoms = self.clay_sel
                print(f"  Using clay selection from initialization: all clay atoms")
                print(f"  Total clay atoms: {len(all_clay_atoms)}")
            else:
                # Method 2: Fallback - extract resname from clay_surface_sel
                print("  Clay selection not found in initialization, attempting fallback...")
                clay_resnames = np.unique(clay_surface_sel.resnames)
                if len(clay_resnames) > 0:
                    clay_resname = clay_resnames[0]
                    print(f"  Extracted clay resname: {clay_resname}")
                    try:
                        all_clay_atoms = self.u.select_atoms(f"resname {clay_resname}")
                        print(f"  Total clay atoms: {len(all_clay_atoms)}")
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not select clay atoms by resname: {e}")
                else:
                    print("  ⚠️  Warning: Could not determine clay resname from surface selection")
            
            # Calculate clay charge density if we have atoms
            if all_clay_atoms is not None and len(all_clay_atoms) > 0:
                try:
                    # Get total clay charge (use first frame - clay doesn't move)
                    self.u.trajectory[0]
                    total_clay_charge_sum = np.sum(all_clay_atoms.charges)
                    total_clay_charge = total_clay_charge_sum  # Store for later use
                    print(f"  Total clay charge: {total_clay_charge:+.4f} e")
                    
                    # Calculate clay density and charge density profiles (average over a few frames)
                    clay_number_density = np.zeros(n_bins)
                    clay_charge_density = np.zeros(n_bins)
                    
                    # Sample only a few frames for clay (it doesn't move much)
                    clay_sample_frames = min(100, n_frames)  # Sample 100 frames max
                    clay_step = max(1, n_frames // clay_sample_frames)
                    
                    n_clay_frames = 0
                    for ts in self.u.trajectory[::clay_step]:
                        # Get clay z-positions and apply centering
                        clay_z = all_clay_atoms.positions[:, 2] - z_offset
                        clay_charges = all_clay_atoms.charges
                        
                        # Bin clay atoms for number density
                        hist, _ = np.histogram(clay_z, bins=z_bins)
                        bin_volume = box_x * box_y * z_bin_width
                        clay_number_density += hist / bin_volume
                        
                        # Bin clay charge density
                        charge_hist, _ = np.histogram(clay_z, bins=z_bins, weights=clay_charges)
                        clay_charge_density += charge_hist / bin_volume
                        
                        n_clay_frames += 1
                    
                    # Average over sampled frames
                    clay_number_density /= n_clay_frames
                    clay_charge_density /= n_clay_frames
                    
                    print(f"  Sampled {n_clay_frames} frames for density profile")
                    print(f"  Clay charge density range: {np.min(clay_charge_density):.6f} to {np.max(clay_charge_density):.6f} e/Å³")
                    
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not calculate clay charge density: {e}")
                    clay_charge_density = None
                    clay_number_density = None
            else:
                print("  ⚠️  Skipping clay charge density calculation (no clay atoms found)")
        
        # === ORGANIC CHARGE DENSITY (OPTIONAL) ===
        organic_charge_density = None
        organic_number_density = None
        total_organic_charge = None
        
        if include_organic_charge_density:
            print(f"\nCalculating organic charge density profile...")
            
            all_organic_atoms = None
            if hasattr(self, 'organics') and self.organics:
                org_list = list(self.organics.values())
                all_organic_atoms = org_list[0]
                for ag in org_list[1:]:
                    all_organic_atoms = all_organic_atoms + ag
                print(f"  Organic molecule types: {list(self.organics.keys())}")
                print(f"  Total organic atoms: {len(all_organic_atoms)}")
            
            if all_organic_atoms is not None and len(all_organic_atoms) > 0:
                try:
                    self.u.trajectory[0]
                    total_organic_charge = float(np.sum(all_organic_atoms.charges))
                    print(f"  Total organic charge: {total_organic_charge:+.4f} e")
                    
                    organic_number_density = np.zeros(n_bins)
                    organic_charge_density = np.zeros(n_bins)
                    
                    # Organics move — sample all frames using same step as ions
                    n_org_frames = 0
                    bin_volume_org = box_x * box_y * z_bin_width
                    for ts in self.u.trajectory[::step]:
                        org_z = all_organic_atoms.positions[:, 2] - z_offset
                        org_charges = all_organic_atoms.charges
                        
                        hist, _ = np.histogram(org_z, bins=z_bins)
                        organic_number_density += hist / bin_volume_org
                        
                        charge_hist, _ = np.histogram(org_z, bins=z_bins, weights=org_charges)
                        organic_charge_density += charge_hist / bin_volume_org
                        
                        n_org_frames += 1
                    
                    organic_number_density /= n_org_frames
                    organic_charge_density /= n_org_frames
                    
                    print(f"  Sampled {n_org_frames} frames for organic density profile")
                    print(f"  Organic charge density range: {np.min(organic_charge_density):.6f} to {np.max(organic_charge_density):.6f} e/Å³")
                    
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not calculate organic charge density: {e}")
                    organic_charge_density = None
                    organic_number_density = None
            else:
                print("  ⚠️  Skipping organic charge density calculation (no organic atoms found)")
        
        # Determine bulk region for concentration calculation
        z_min = np.min(z_centers)
        z_max = np.max(z_centers)
        
        if bulk_reference_distance is not None:
            # MANUAL MODE: User explicitly defines bulk region center and span
            bulk_mask = (z_centers > (bulk_reference_distance - bulk_span)) & \
                       (z_centers < (bulk_reference_distance + bulk_span))
            print(f"  Manual bulk region: {bulk_reference_distance - bulk_span:.1f} to {bulk_reference_distance + bulk_span:.1f} Å")
        else:
            # AUTOMATIC MODE: Detect bulk region based on system geometry
            if abs(surface_position) > 10.0:  # Centered system with two surfaces
                # Bulk spans entire interlayer between both surfaces, excluding edges
                # This gives same bulk concentration for both top and bottom surface analyses
                bulk_mask = (z_centers > (z_min + bulk_span)) & (z_centers < (z_max - bulk_span))
                print(f"  Automatic bulk region (full interlayer): {z_min + bulk_span:.1f} to {z_max - bulk_span:.1f} Å")
            else:  # Asymmetric single-surface system
                # Bulk is region away from the single surface
                if surface_position < (z_min + z_max) / 2:  # Surface on bottom
                    bulk_mask = (z_centers > (surface_position + 10.0)) & (z_centers < (z_max - bulk_span))
                else:  # Surface on top
                    bulk_mask = (z_centers < (surface_position - 10.0)) & (z_centers > (z_min + bulk_span))
                print(f"  Automatic bulk region (away from surface): {z_centers[bulk_mask].min():.1f} to {z_centers[bulk_mask].max():.1f} Å")
        
        if not np.any(bulk_mask):
            # Fallback
            if surface_position > 0:
                bulk_mask = z_centers < (surface_position - 5.0)
            else:
                bulk_mask = z_centers > (surface_position + 5.0)
            if not np.any(bulk_mask):
                bulk_mask = np.abs(z_centers - surface_position) > 5.0
        
        # Avogadro's number for concentration conversion
        N_A = 6.02214076e23
        
        # Print ion density statistics
        print(f"\nIon density statistics:")
        for ion_name, density in ion_densities.items():
            if np.max(density) > 0:
                charge = self._get_ion_charge(ion_name, charge_dict)
                
                # Max peak density
                max_density = np.max(density)
                max_molar = max_density * (1e10)**3 / N_A / 1000
                
                # Bulk average density
                bulk_density = np.mean(density[bulk_mask])
                bulk_molar = bulk_density * (1e10)**3 / N_A / 1000
                
                print(f"  {ion_name} (q={charge:+.0f}e):")
                print(f"    Peak:  density = {max_density:.6f} ions/Å³,  concentration = {max_molar:.4f} M")
                print(f"    Bulk:  density = {bulk_density:.6f} ions/Å³,  concentration = {bulk_molar:.4f} M")
        
        # Create charge density breakdown table
        print(f"\nCharge density breakdown:")
        print("="*80)
        
        # Calculate charge density contribution for each ion
        ion_charge_densities = {}
        mobile_breakdown_table = []
        
        for ion_name, density in ion_densities.items():
            charge = self._get_ion_charge(ion_name, charge_dict)
            ion_charge_density = density * charge  # e/Å³
            ion_charge_densities[ion_name] = ion_charge_density
            
            # Calculate statistics
            integrated_charge = np.trapz(ion_charge_density, dx=z_bin_width)
            max_charge_density = np.max(np.abs(ion_charge_density))
            
            mobile_breakdown_table.append({
                'Ion': ion_name,
                'Charge': f'{charge:+.0f}e',
                'Integrated_Charge_eA': integrated_charge,
                'Max_Charge_Density_e_per_A3': max_charge_density,
                'Contribution_%': 0.0  # Will calculate after total
            })
        
        # Calculate mobile ion percentages (relative to mobile total)
        mobile_total_abs_charge = sum(abs(row['Integrated_Charge_eA']) for row in mobile_breakdown_table)
        for row in mobile_breakdown_table:
            if mobile_total_abs_charge > 0:
                row['Contribution_%'] = 100 * abs(row['Integrated_Charge_eA']) / mobile_total_abs_charge
        
        # Calculate mobile subtotal
        mobile_total_charge = sum(row['Integrated_Charge_eA'] for row in mobile_breakdown_table)
        
        # Print hierarchical breakdown
        import pandas as pd
        
        # Calculate total charge in electrons (multiply by XY area)
        xy_area = box_x * box_y
        mobile_total_charge_electrons = mobile_total_charge * xy_area
        
        print("\n" + "="*80)
        print("MOBILE IONS (standard EDL analysis):")
        print("-"*80)
        df_mobile = pd.DataFrame(mobile_breakdown_table)
        print(df_mobile.to_string(index=False))
        print("-"*80)
        print(f"Mobile subtotal (per area):  {mobile_total_charge:+.6f} e/Å²")
        print(f"Mobile subtotal (total):     {mobile_total_charge_electrons:+.2f} e  (over {xy_area:.1f} Å² area)")
        print("="*80)
        
        # Add clay charge density if requested
        if include_clay_charge_density and clay_charge_density is not None:
            # Use actual total clay charge (not integrated from profile)
            clay_total_charge_electrons = total_clay_charge  # Direct sum, already in electrons
            clay_integrated_charge = total_clay_charge / xy_area  # Convert to charge per area
            clay_max_charge_density = np.max(clay_charge_density)  # Most positive
            clay_min_charge_density = np.min(clay_charge_density)  # Most negative
            
            # Find z-positions of max and min charge densities
            max_idx = np.argmax(clay_charge_density)
            min_idx = np.argmin(clay_charge_density)
            z_at_max = z_centers[max_idx]
            z_at_min = z_centers[min_idx]
            distance_max_from_surface = abs(z_at_max - surface_position)
            distance_min_from_surface = abs(z_at_min - surface_position)
            
            print("\nSTATIC CHARGES (clay atoms):")
            print("-"*80)
            print(f"{'Type':<6} {'Charge':<8} {'Integrated_Charge_eA':>22} {'Max_Density (e/Å³)':>22} {'Min_Density (e/Å³)':>22}")
            print(f"{'Clay':<6} {'varies':<8} {clay_integrated_charge:>22.6f} {clay_max_charge_density:>22.6f} {clay_min_charge_density:>22.6f}")
            print("-"*80)
            print(f"  Max density location: z = {z_at_max:+.2f} Å  ({distance_max_from_surface:.2f} Å from surface at {surface_position:+.2f} Å)")
            print(f"  Min density location: z = {z_at_min:+.2f} Å  ({distance_min_from_surface:.2f} Å from surface at {surface_position:+.2f} Å)")
            print("-"*80)
            print(f"Clay subtotal (per area):    {clay_integrated_charge:+.6f} e/Å²")
            print(f"Clay subtotal (total):       {clay_total_charge_electrons:+.6f} e")
            print("="*80)
        
        # Add organic charge density if requested
        if include_organic_charge_density and organic_charge_density is not None:
            org_integrated_charge = total_organic_charge / xy_area
            org_max_charge_density = np.max(organic_charge_density)
            org_min_charge_density = np.min(organic_charge_density)
            max_idx_org = np.argmax(organic_charge_density)
            min_idx_org = np.argmin(organic_charge_density)
            z_at_org_max = z_centers[max_idx_org]
            z_at_org_min = z_centers[min_idx_org]
            
            print("\nMOBILE ORGANIC CHARGES (per-atom partial charges):")
            print("-"*80)
            print(f"{'Type':<12} {'Net_Charge':>12} {'Integrated_eÅ':>16} {'Max_ρ (e/Å³)':>16} {'Min_ρ (e/Å³)':>16}")
            org_names_str = '+'.join(self.organics.keys())
            print(f"{org_names_str:<12} {total_organic_charge:>+12.4f} {org_integrated_charge:>16.6f} {org_max_charge_density:>16.6f} {org_min_charge_density:>16.6f}")
            print("-"*80)
            print(f"  Max density location: z = {z_at_org_max:+.2f} Å  ({abs(z_at_org_max - surface_position):.2f} Å from surface)")
            print(f"  Min density location: z = {z_at_org_min:+.2f} Å  ({abs(z_at_org_min - surface_position):.2f} Å from surface)")
            print("-"*80)
            print(f"Organic subtotal (per area): {org_integrated_charge:+.6f} e/Å²")
            print(f"Organic subtotal (total):    {total_organic_charge:+.6f} e")
            print("="*80)
        
        # Grand total — always print, using whatever components are enabled
        _have_clay = include_clay_charge_density and clay_charge_density is not None
        _have_org  = include_organic_charge_density and organic_charge_density is not None
        
        if _have_clay or _have_org:
            _total_per_area = mobile_total_charge
            _total_electrons = mobile_total_charge_electrons
            _abs_total = abs(mobile_total_charge)
            _label_parts = ['Mobile']
            
            if _have_clay:
                _clay_int = total_clay_charge / xy_area
                _total_per_area += _clay_int
                _total_electrons += total_clay_charge
                _abs_total += abs(_clay_int)
                _label_parts.append('Clay')
            
            if _have_org:
                _org_int = total_organic_charge / xy_area
                _total_per_area += _org_int
                _total_electrons += total_organic_charge
                _abs_total += abs(_org_int)
                _label_parts.append('Organic')
            
            _mobile_pct = (abs(mobile_total_charge) / _abs_total * 100) if _abs_total > 0 else 0.0
            
            print(f"\nTOTAL SYSTEM CHARGE ({' + '.join(_label_parts)}):")
            print("-"*80)
            print(f"Integrated (per area):   {_total_per_area:+.6f} e/Å²")
            if _have_clay:
                print(f"Total charge:            {_total_electrons:+.2f} e  (should be ≈0 for neutral system)")
            else:
                print(f"Total charge:            {_total_electrons:+.2f} e  (Clay not included — should ≈ −clay charge)")
            print(f"Mobile ions contribute:  {_mobile_pct:5.1f}% of total absolute charge")
            if _have_clay:
                _clay_pct = (abs(total_clay_charge / xy_area) / _abs_total * 100) if _abs_total > 0 else 0.0
                print(f"Clay contributes:        {_clay_pct:5.1f}% of total absolute charge")
            if _have_org:
                _org_pct = (abs(total_organic_charge / xy_area) / _abs_total * 100) if _abs_total > 0 else 0.0
                print(f"Organic contributes:     {_org_pct:5.1f}% of total absolute charge")
            print("="*80)
        else:
            # Show only mobile total (current behavior when neither clay nor organic included)
            mobile_total_charge_electrons = mobile_total_charge * xy_area
            print(f"\nTotal mobile charge (per area): {mobile_total_charge:+.6f} e/Å²")
            print(f"Total mobile charge (total):    {mobile_total_charge_electrons:+.2f} e")
            print("="*80)
        
        # Save detailed charge density profiles to CSV
        csv_filename = 'charge_density_profiles.csv'
        print(f"\nSaving detailed charge density profiles to: {csv_filename}")
        
        # Create dataframe with z-positions and all charge densities
        profile_data = {'z_position_A': z_centers}
        
        # Add each ion's charge density contribution
        for ion_name, ion_charge_dens in ion_charge_densities.items():
            charge = self._get_ion_charge(ion_name, charge_dict)
            profile_data[f'{ion_name}_charge_density_e_per_A3'] = ion_charge_dens
            profile_data[f'{ion_name}_number_density_per_A3'] = ion_densities[ion_name]
        
        # Add mobile total charge density
        profile_data['Mobile_charge_density_e_per_A3'] = charge_density
        
        # Add clay charge density if calculated
        if include_clay_charge_density and clay_charge_density is not None:
            profile_data['Clay_charge_density_e_per_A3'] = clay_charge_density
            profile_data['Clay_number_density_per_A3'] = clay_number_density
        
        # Add organic charge density if calculated
        if include_organic_charge_density and organic_charge_density is not None:
            profile_data['Organic_charge_density_e_per_A3'] = organic_charge_density
            profile_data['Organic_number_density_per_A3'] = organic_number_density
        
        # Total charge density: sum all available components
        _total_cd = charge_density.copy()
        if include_clay_charge_density and clay_charge_density is not None:
            _total_cd = _total_cd + clay_charge_density
        if include_organic_charge_density and organic_charge_density is not None:
            _total_cd = _total_cd + organic_charge_density
        profile_data['Total_charge_density_e_per_A3'] = _total_cd
        
        # Save CSV if requested
        if save_charge_profiles_csv:
            df_profiles = pd.DataFrame(profile_data)
            df_profiles.to_csv(csv_filename, index=False, float_format='%.8e')
            print(f"  Saved charge density profiles to: {csv_filename}")
            print(f"  ({len(z_centers)} z-positions, columns: z_position, {', '.join([f'{ion}_charge_density' for ion in ion_charge_densities.keys()])}, Total)")
        
        # Determine charge density for Poisson equation:
        # mobile ions + organic (both are mobile charges).
        # Clay is a fixed background handled via the surface charge boundary condition.
        if include_organic_charge_density and organic_charge_density is not None:
            poisson_charge_density = charge_density + organic_charge_density
        else:
            poisson_charge_density = charge_density
        
        # Check charge neutrality — include clay to assess full-system balance
        total_charge = np.trapz(poisson_charge_density, dx=z_bin_width)
        xy_area = box_x * box_y

        # Get clay charge for the neutrality check (use already-computed value if available,
        # otherwise read it directly from atom charges).
        _clay_q_for_check = total_clay_charge  # already a float if include_clay_charge_density=True
        if _clay_q_for_check is None and self.clay is not None and len(self.clay) > 0:
            try:
                _clay_q_for_check = float(np.sum(self.clay.charges))
            except Exception:
                _clay_q_for_check = None

        # Full-system integrated charge (per area): mobile+organic + clay
        if _clay_q_for_check is not None:
            total_charge_full = total_charge + _clay_q_for_check / xy_area
            _clay_note = f" + clay ({_clay_q_for_check:+.2f} e)"
        else:
            total_charge_full = total_charge
            _clay_note = ""

        print(f"\nCharge neutrality check:")
        print(f"  Mobile+Organic integrated charge (per area): {total_charge:+.6f} e/Å²")
        if _clay_note:
            print(f"  Full system charge (mobile+organic{_clay_note}): {total_charge_full:+.6f} e/Å²")
        if include_organic_charge_density and organic_charge_density is not None:
            print(f"  (Mobile+organic includes salt ions + organic partial charges)")

        if abs(total_charge_full) * xy_area > 0.5:
            print(f"  ⚠️  Warning: System not charge neutral!")
        else:
            print(f"  ✓ System is approximately charge neutral")
        
        # Calculate electrostatic potential via Poisson equation
        print(f"\nSolving Poisson equation for electrostatic potential...")
        potential, electric_field = self._solve_poisson_1d_edl(
            poisson_charge_density, z_bin_width, dielectric_constant, temperature
        )
        
        print(f"  Potential range: {np.min(potential):.3f} to {np.max(potential):.3f} kT/e")
        print(f"  Electric field range: {np.min(electric_field):.3f} to {np.max(electric_field):.3f} kT/e/Å")
        
        # Calculate surface charge density
        surface_bin_idx = np.argmin(np.abs(z_centers - surface_position))
        
        # Calculate surface charge density
        print(f"\nClay surface charge analysis:")
        print(f"  Surface area: {xy_area:.2f} Å² ({box_x:.2f} × {box_y:.2f})")
        
        # Method 1: From clay atom charges (preferred)
        if clay_surface_charge is None:
            try:
                # Get ALL clay atoms (entire layer, including wrapped portions)
                self.u.trajectory[0]
                clay_resname = clay_surface_sel.resnames[0]
                
                clay_full_sel = self.u.select_atoms(f"resname {clay_resname}")
                
                print(f"  Surface selection: {len(clay_surface_sel)} atoms ({clay_surface_sel.names[0]} type)")
                print(f"  Total clay atoms: {len(clay_full_sel)} atoms (all types)")
                print(f"  Atom types: {np.unique(clay_full_sel.names)}")
                
                # Sum charges from all clay atoms
                total_clay_charge = np.sum(clay_full_sel.charges)
                print(f"  Source: All clay atoms (resname {clay_resname})")
                print(f"  Total charge: {total_clay_charge:.4f} e")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not calculate clay surface charge ({e})")
                print(f"     Assuming neutral clay (σ=0). Specify clay_surface_charge parameter.")
                total_clay_charge = 0.0
        else:
            total_clay_charge = clay_surface_charge
            print(f"  Source: User-specified")
            print(f"  Total charge: {total_clay_charge:.4f} e")
        
        # Surface charge density
        surface_charge_density = total_clay_charge / xy_area
        
        print(f"  Surface charge density:")
        print(f"    σ_surface = {surface_charge_density:.6f} e/Å²")
        print(f"    σ_surface = {surface_charge_density * 1.602e-19 / (1e-10)**2:.6f} C/m²")
        
        # Initialize results dictionary
        results = {
            'z_centers': z_centers,
            'charge_density': charge_density,  # Mobile ions only (backward compat)
            'electrostatic_potential': potential,
            'electric_field': electric_field,
            'ion_densities': ion_densities,
            'surface_position': surface_position,
            'surface_position_std': surface_std,
            'surface_charge_density': surface_charge_density,
            'metadata': {
                'n_frames': n_frames,
                'z_bin_width': z_bin_width,
                'temperature': temperature,
                'dielectric_constant': dielectric_constant,
                'box_dimensions': (box_x, box_y, box_z),
                'charge_dict': charge_dict,
                'charge_neutrality': abs(total_charge) < 1e-3,
                'include_clay_charge_density': include_clay_charge_density,
                'include_organic_charge_density': include_organic_charge_density
            }
        }
        
        # Add clay charge density if calculated
        if include_clay_charge_density and clay_charge_density is not None:
            results['clay_charge_density'] = clay_charge_density
            results['clay_number_density'] = clay_number_density
        
        # Add organic charge density if calculated
        if include_organic_charge_density and organic_charge_density is not None:
            results['organic_charge_density'] = organic_charge_density
            results['organic_number_density'] = organic_number_density
        
        # Build total_charge_density from all enabled components
        _total_cd = charge_density.copy()
        if include_clay_charge_density and clay_charge_density is not None:
            _total_cd = _total_cd + clay_charge_density
        if include_organic_charge_density and organic_charge_density is not None:
            _total_cd = _total_cd + organic_charge_density
        if include_clay_charge_density or include_organic_charge_density:
            results['total_charge_density'] = _total_cd
        
        # 1. STERN LAYER IDENTIFICATION
        if identify_stern_layer:
            print(f"\n" + "-"*80)
            print("STERN LAYER ANALYSIS")
            print("-"*80)
            stern_layer_results = self._identify_stern_layer(
                z_centers, ion_densities, surface_position, charge_dict, use_manual_peaks
            )
            results['stern_layer'] = stern_layer_results
        
        # 2. DEBYE LENGTH CALCULATION
        if calculate_debye_length:
            print(f"\n" + "-"*80)
            print("DEBYE LENGTH CALCULATION")
            print("-"*80)
            
            # Use OHP position if Stern layer was identified, otherwise use default
            ohp_distance = 5.0  # Default
            if identify_stern_layer and 'stern_layer' in results and 'ohp_position' in results['stern_layer']:
                ohp_distance = abs(results['stern_layer']['ohp_position'] - surface_position)
                print(f"  Using OHP position ({ohp_distance:.1f} Å from surface) to define diffuse layer start")
            else:
                print(f"  Using default diffuse layer start ({ohp_distance:.1f} Å from surface)")
            
            debye_results = self._calculate_debye_length(
                z_centers, potential, ion_densities, surface_position,
                temperature, dielectric_constant, charge_dict, bulk_reference_distance,
                ohp_distance
            )
            results['debye_length'] = debye_results
        
        # 3. ADSORPTION MODE CLASSIFICATION
        if classify_adsorption_modes and identify_stern_layer:
            print(f"\n" + "-"*80)
            print("ION ADSORPTION MODE CLASSIFICATION")
            print("-"*80)
            adsorption_results = self._classify_adsorption_modes(
                ion_densities, z_centers, surface_position,
                stern_layer_results, n_frames, box_x, box_y, z_bin_width
            )
            results['adsorption_modes'] = adsorption_results
        
        # 4. GOUY-CHAPMAN COMPARISON
        if compare_gouy_chapman:
            print(f"\n" + "-"*80)
            print("GOUY-CHAPMAN THEORY COMPARISON")
            print("-"*80)
            gc_results = self._compare_gouy_chapman_theory(
                z_centers, potential, ion_densities, surface_position,
                surface_charge_density, temperature, dielectric_constant,
                charge_dict, bulk_reference_distance
            )
            results['gouy_chapman_comparison'] = gc_results
        
        # Store results
        self.results['edl_analysis'] = results
        
        # === SAVE CACHE ===
        if save_cache:
            try:
                print(f"\n💾 Saving results to cache: {cache_file}")
                
                # Prepare data for saving
                save_dict = {
                    'z_centers': results['z_centers'],
                    'charge_density': results['charge_density'],
                    'electrostatic_potential': results['electrostatic_potential'],
                    'electric_field': results['electric_field'],
                    'ion_densities': results['ion_densities'],
                    'surface_position': results['surface_position'],
                    'surface_position_std': results['surface_position_std'],
                    'surface_charge_density': results['surface_charge_density'],
                    'metadata': results['metadata']
                }
                
                # Add optional analyses if computed
                if 'stern_layer' in results:
                    save_dict['stern_layer'] = results['stern_layer']
                if 'debye_length' in results:
                    save_dict['debye_length'] = results['debye_length']
                if 'adsorption_modes' in results:
                    save_dict['adsorption_modes'] = results['adsorption_modes']
                if 'gouy_chapman_comparison' in results:
                    save_dict['gouy_chapman_comparison'] = results['gouy_chapman_comparison']
                if 'organic_charge_density' in results:
                    save_dict['organic_charge_density'] = results['organic_charge_density']
                if 'organic_number_density' in results:
                    save_dict['organic_number_density'] = results['organic_number_density']
                if 'total_charge_density' in results:
                    save_dict['total_charge_density'] = results['total_charge_density']
                
                # Save to npz file
                np.savez_compressed(cache_file, **save_dict)
                print(f"✅ Cache saved successfully!")
                
            except Exception as e:
                print(f"⚠️  Cache save failed: {e}")
        
        # === PRINT SUMMARY REPORT ===
        if summary_report:
            self._print_edl_summary_report(results)
        
        print(f"\n" + "="*80)
        print("ELECTRICAL DOUBLE LAYER ANALYSIS COMPLETE")
        print("="*80)
        
        return results

    def analyze_electrical_double_layer_complete_OLD(self, clay_surface_sel, 
                                                    z_bin_width=0.5,
                                                    charge_dict=None,
                                                    dielectric_constant=78.0,
                                                    temperature=300.0,
                                                    bulk_reference_distance=0.0,
                                                    identify_stern_layer=True,
                                                    calculate_debye_length=True,
                                                    compare_gouy_chapman=True,
                                                    classify_adsorption_modes=True,
                                                    use_manual_peaks=False,
                                                    step=1,
                                                    njobs=1,
                                                    save_cache=True,
                                                    cache_file=None,
                                                    force_rerun=False,
                                                    center_box=True,
                                                    clay_surface_charge=None,
                                                    save_charge_profiles_csv=False,
                                                    include_clay_charge_density=False,
                                                    summary_report=False):
            """
            Comprehensive electrical double layer (EDL) analysis near clay surface.
            
            Calculates all key EDL properties including:
            - Charge density profile ρ(z)
            - Electrostatic potential ψ(z) via Poisson equation
            - Electric field E(z) = -dψ/dz
            - Debye screening length λ_D
            - Stern layer thickness (inner vs outer Helmholtz planes)
            - Surface charge density σ_surface
            - Ion adsorption modes (inner-sphere, outer-sphere, diffuse layer)
            - Comparison with Gouy-Chapman theory
            
            Parameters
            ----------
            clay_surface_sel : AtomGroup
                Selection of clay surface atoms for reference positioning
            z_bin_width : float, default=0.5
                Bin width for z-direction discretization (Å)
            charge_dict : dict, optional
                Dictionary of ion charges in elementary charge units
                Example: {'NA': 1.0, 'MG': 2.0, 'CL': -1.0}
                If None, uses common defaults
            dielectric_constant : float, default=78.0
                Relative permittivity of water (ε_r)
            temperature : float, default=300.0
                System temperature in Kelvin
            bulk_reference_distance : float, default=0.0
                Distance from surface considered as "bulk" reference (Å)
            identify_stern_layer : bool, default=True
                Calculate Stern layer thickness from ion density minima
            calculate_debye_length : bool, default=True
                Calculate Debye screening length from exponential fit
            compare_gouy_chapman : bool, default=True
                Compare MD results with Gouy-Chapman theory predictions
            classify_adsorption_modes : bool, default=True
                Classify ions into inner-sphere, outer-sphere, and diffuse layer
            use_manual_peaks : bool, default=False
                If True, use manual peak positions from analyze_ion_peaks_manual() for IHP/OHP.
                If False, use automatic peak finding from ion density profiles.
                Only takes effect if manual peaks have been previously set via analyze_ion_peaks_manual().
            step : int, default=1
                Analyze every nth frame
            njobs : int, default=1
                Number of parallel jobs for frame processing. Use -1 for all CPUs.
            save_cache : bool, default=True
                Save results to cache file for faster subsequent analysis
            cache_file : str, optional
                Custom cache filename. If None, auto-generates from parameters.
            force_rerun : bool, default=False
                Force recalculation even if cache exists
            center_box : bool, default=True
                If True, center the simulation box so z=0 is at box center.
                This allows proper use of 'prop z > 0' and 'prop z < 0' selections
                for top and bottom surfaces.
            clay_surface_charge : float, optional
                Total charge of the clay surface in elementary charge units (e).
                If None, calculated from clay_surface_sel atom charges.
                For charged clays (e.g., montmorillonite), this should be negative.
                Example: For 0.01 e/Å² and 3000 Å² area, use -30.0 e
            save_charge_profiles_csv : bool, default=False
                If True, saves charge density profiles to 'charge_density_profiles.csv'
                in the current working directory. CSV contains z-position, charge density
                and number density for each ion type, plus total charge density.
            include_clay_charge_density : bool, default=False
                If True, calculates and reports clay charge density alongside mobile ion
                charge density. Shows hierarchical breakdown: mobile ions, clay (static),
                and total system charge. Useful for comparing static vs mobile screening.
                Note: EDL analysis (Poisson, Stern, Debye) still uses mobile charge only.
            summary_report : bool, default=False
                If True, prints a formatted summary table of all EDL analysis results
                at the end, including surface properties, Stern layer, Debye length,
                bulk concentrations, and ion adsorption modes in easy-to-read tables.
            
            Returns
            -------
            dict
                Comprehensive EDL analysis results with keys:
                - 'z_centers': z-position array (Å)
                - 'charge_density': ρ(z) in e/Å³ (mobile ions only)
                - 'electrostatic_potential': ψ(z) in kT/e
                - 'electric_field': E(z) in kT/e/Å
                - 'ion_densities': {ion_type: density(z)} in ions/Å³
                - 'surface_position': z-coordinate of clay surface (Å)
                - 'surface_charge_density': σ_surface in e/Å²
                - 'stern_layer': dict with IHP, OHP positions and thicknesses
                - 'debye_length': dict with theoretical and fitted values (Å)
                - 'adsorption_modes': ion counts in each region (if requested)
                - 'gouy_chapman_comparison': theory vs MD (if requested)
                - 'clay_charge_density': ρ_clay(z) in e/Å³ (if include_clay_charge_density=True)
                - 'clay_number_density': clay atom density (if include_clay_charge_density=True)
                - 'total_charge_density': ρ_mobile + ρ_clay (if include_clay_charge_density=True)
                - 'metadata': calculation parameters and diagnostics
                - 'metadata': calculation parameters and diagnostics
            """
            
            print("\n" + "="*80)
            print("ELECTRICAL DOUBLE LAYER ANALYSIS")
            print("="*80)
            
            # Default charge dictionary
            if charge_dict is None:
                charge_dict = {
                    'NA': 1.0, 'K': 1.0, 'LI': 1.0, 'RB': 1.0, 'CS': 1.0,
                    'MG': 2.0, 'CA': 2.0, 'SR': 2.0, 'BA': 2.0,
                    'CL': -1.0, 'BR': -1.0, 'I': -1.0, 'F': -1.0
                }
            
            # === CACHE SETUP ===
            # Generate cache filename based on parameters
            if cache_file is None:
                # Create hash from key parameters
                cache_params = (
                    f"zbw{z_bin_width}_step{step}_T{temperature}_"
                    f"eps{dielectric_constant}_center{center_box}_"
                    f"ions{'_'.join(sorted(self.ions.keys()))}"
                )
                cache_hash = hashlib.md5(cache_params.encode()).hexdigest()[:12]
                cache_dir = os.path.join(os.path.dirname(self.traj), '.edl_cache')
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, f"edl_analysis_{cache_hash}.npz")
            
            # Check cache
            if not force_rerun and os.path.exists(cache_file):
                try:
                    print(f"\n📂 Loading cached EDL analysis from: {cache_file}")
                    cached_data = np.load(cache_file, allow_pickle=True)
                    
                    # Reconstruct results dictionary
                    results = {
                        'z_centers': cached_data['z_centers'],
                        'charge_density': cached_data['charge_density'],
                        'electrostatic_potential': cached_data['electrostatic_potential'],
                        'electric_field': cached_data['electric_field'],
                        'ion_densities': cached_data['ion_densities'].item(),
                        'surface_position': float(cached_data['surface_position']),
                        'surface_position_std': float(cached_data['surface_position_std']),
                        'surface_charge_density': float(cached_data['surface_charge_density']),
                        'metadata': cached_data['metadata'].item()
                    }
                    
                    # Load optional analyses if present
                    if 'stern_layer' in cached_data:
                        results['stern_layer'] = cached_data['stern_layer'].item()
                    if 'debye_length' in cached_data:
                        results['debye_length'] = cached_data['debye_length'].item()
                    if 'adsorption_modes' in cached_data:
                        results['adsorption_modes'] = cached_data['adsorption_modes'].item()
                    if 'gouy_chapman_comparison' in cached_data:
                        results['gouy_chapman_comparison'] = cached_data['gouy_chapman_comparison'].item()
                    
                    self.results['edl_analysis'] = results
                    print("✅ Cached data loaded successfully!")
                    return results
                    
                except Exception as e:
                    print(f"⚠️  Cache load failed ({e}), recalculating...")
            
            # Physical constants
            k_B = 1.381e-23  # J/K
            e = 1.602e-19    # C
            epsilon_0 = 8.854e-12  # F/m
            N_A = 6.022e23   # Avogadro's number
            
            # Get box dimensions
            box_x = self.u.dimensions[0]
            box_y = self.u.dimensions[1]
            box_z = self.u.dimensions[2]
            box_volume = box_x * box_y * box_z  # Å³
            
            # Calculate z-offset for centering
            if center_box:
                z_offset = box_z / 2
                print(f"  Centering box: z_offset = {z_offset:.2f} Å")
            else:
                z_offset = 0.0
            
            # Determine z-range and create bins
            if center_box:
                # Center box around z=0
                half_box = box_z / 2
                z_min = -half_box
                z_max = half_box
            else:
                # Use original coordinate system
                z_min = 0
                z_max = box_z
            
            z_bins = np.arange(z_min, z_max + z_bin_width, z_bin_width)
            z_centers = (z_bins[:-1] + z_bins[1:]) / 2
            n_bins = len(z_centers)
            
            print(f"\nSystem configuration:")
            print(f"  Box dimensions: {box_x:.1f} × {box_y:.1f} × {box_z:.1f} Å")
            print(f"  Box centering: {'Enabled (z=0 at center)' if center_box else 'Disabled (original coordinates)'}")
            print(f"  Z-range: {z_min:.2f} to {z_max:.2f} Å")
            print(f"  Z-bin width: {z_bin_width} Å")
            print(f"  Number of z-bins: {n_bins}")
            print(f"  Temperature: {temperature} K")
            print(f"  Dielectric constant: {dielectric_constant}")
            
            # Print clay surface selection info
            n_clay_atoms = len(clay_surface_sel)
            print(f"\nClay surface selection:")
            print(f"  Number of atoms: {n_clay_atoms}")
            if n_clay_atoms == 0:
                raise ValueError("❌ No atoms found in clay_surface_sel! Check your selection string.")
            
            # Determine clay surface position (average z-position)
            clay_surface_positions = []
            for ts in self.u.trajectory[::step]:
                clay_z_raw = clay_surface_sel.positions[:, 2]
                # Apply centering offset
                clay_z_centered = clay_z_raw - z_offset
                clay_surface_positions.append(np.mean(clay_z_centered))
            
            surface_position = np.mean(clay_surface_positions)
            surface_std = np.std(clay_surface_positions)
            
            print(f"\nClay surface analysis:")
            print(f"  Average z-position: {surface_position:.2f} ± {surface_std:.2f} Å")
            
            # Print z-range of clay atoms for verification
            self.u.trajectory[0]  # Go to first frame
            clay_z_positions = clay_surface_sel.positions[:, 2] - z_offset
            z_min_clay = np.min(clay_z_positions)
            z_max_clay = np.max(clay_z_positions)
            print(f"  Z-range: {z_min_clay:.2f} to {z_max_clay:.2f} Å (spread: {z_max_clay - z_min_clay:.2f} Å)")
            
            # Warn if selection might span both surfaces
            spread = z_max_clay - z_min_clay
            if spread > (box_z / 3):
                print(f"  ⚠️  WARNING: Clay selection spans {spread:.1f} Å")
                print(f"     This might include BOTH top and bottom surfaces!")
                if center_box:
                    print(f"     With centering enabled, use 'prop z > 0' (top) or 'prop z < 0' (bottom).")
                else:
                    print(f"     Enable center_box=True to use 'prop z > 0' / 'prop z < 0' selections.")
            
            # Initialize storage arrays
            ion_densities = {ion_name: np.zeros(n_bins) for ion_name in self.ions.keys()}
            charge_density = np.zeros(n_bins)
            
            # Calculate ion density and charge density profiles
            print(f"\nCalculating ion density profiles...")
            
            # === PARALLEL PROCESSING ===
            total_frames = len(self.u.trajectory[::step])
            
            # Determine number of jobs
            if njobs == -1:
                import multiprocessing
                njobs = multiprocessing.cpu_count()
            
            # Use parallel processing if njobs > 1
            if njobs > 1 and total_frames > njobs:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                
                print(f"  Using {njobs} parallel workers for {total_frames} frames")
                
                # Create ion selection dictionary for workers
                ion_sel_dict = {}
                for ion_name, ion_atoms in self.ions.items():
                    # Skip empty ion groups
                    if len(ion_atoms) == 0:
                        continue
                    # Reconstruct selection using atom indices (most reliable)
                    indices_str = ' '.join(map(str, ion_atoms.indices))
                    ion_sel_dict[ion_name] = f"index {indices_str}"
                
                # Split frames into chunks
                frame_indices = list(range(0, len(self.u.trajectory), step))
                chunk_size = max(1, len(frame_indices) // njobs)
                frame_chunks = []
                
                for i in range(0, len(frame_indices), chunk_size):
                    chunk_indices = frame_indices[i:i+chunk_size]
                    if len(chunk_indices) > 0:
                        start_frame = chunk_indices[0]
                        end_frame = chunk_indices[-1] + step
                        frame_chunks.append((start_frame, end_frame, step))
                
                # Execute workers in parallel
                with ProcessPoolExecutor(max_workers=njobs) as executor:
                    # Submit all tasks
                    futures = []
                    for chunk in frame_chunks:
                        future = executor.submit(
                            _calculate_edl_density_worker,
                            self.top, self.traj, chunk,
                            ion_sel_dict, charge_dict, z_bins,
                            box_x, box_y, z_bin_width, z_offset
                        )
                        futures.append(future)
                    
                    # Collect results
                    completed = 0
                    n_frames = 0
                    for future in as_completed(futures):
                        result = future.result()
                        
                        # Accumulate densities
                        for ion_name in result['ion_densities'].keys():
                            ion_densities[ion_name] += result['ion_densities'][ion_name]
                        charge_density += result['charge_density']
                        n_frames += result['n_frames']
                        
                        completed += 1
                        print(f"    Worker {completed}/{len(futures)} completed ({result['n_frames']} frames)")
                
                # Average over frames
                for ion_name in ion_densities.keys():
                    ion_densities[ion_name] /= n_frames
                charge_density /= n_frames
                
            else:
                # Serial processing
                if njobs > 1:
                    print(f"  Using serial processing (insufficient frames for parallelization)")
                
                n_frames = 0
                for ts in self.u.trajectory[::step]:
                    for ion_name, ion_atoms in self.ions.items():
                        if len(ion_atoms) == 0:
                            continue
                        
                        # Get ion z-positions and apply centering offset
                        ion_z = ion_atoms.positions[:, 2] - z_offset
                        
                        # Bin ions
                        hist, _ = np.histogram(ion_z, bins=z_bins)
                        
                        # Convert to number density (ions/Å³)
                        bin_volume = box_x * box_y * z_bin_width
                        density = hist / bin_volume
                        
                        ion_densities[ion_name] += density
                        
                        # Add to charge density
                        ion_charge = self._get_ion_charge(ion_name, charge_dict)
                        charge_density += density * ion_charge
                    
                    n_frames += 1
                
                # Average over frames
                for ion_name in ion_densities.keys():
                    ion_densities[ion_name] /= n_frames
                charge_density /= n_frames
            
            print(f"  Processed {n_frames} frames")
            
            # === CLAY CHARGE DENSITY (OPTIONAL) ===
            clay_charge_density = None
            clay_number_density = None
            total_clay_charge = None
            
            if include_clay_charge_density:
                print(f"\nCalculating clay charge density profile...")
                
                # Method 1: Use ALL clay atoms from initialization (preferred)
                all_clay_atoms = None
                if hasattr(self, 'clay_sel') and self.clay_sel is not None:
                    all_clay_atoms = self.clay_sel
                    print(f"  Using clay selection from initialization: all clay atoms")
                    print(f"  Total clay atoms: {len(all_clay_atoms)}")
                else:
                    # Method 2: Fallback - extract resname from clay_surface_sel
                    print("  Clay selection not found in initialization, attempting fallback...")
                    clay_resnames = np.unique(clay_surface_sel.resnames)
                    if len(clay_resnames) > 0:
                        clay_resname = clay_resnames[0]
                        print(f"  Extracted clay resname: {clay_resname}")
                        try:
                            all_clay_atoms = self.u.select_atoms(f"resname {clay_resname}")
                            print(f"  Total clay atoms: {len(all_clay_atoms)}")
                        except Exception as e:
                            print(f"  ⚠️  Warning: Could not select clay atoms by resname: {e}")
                    else:
                        print("  ⚠️  Warning: Could not determine clay resname from surface selection")
                
                # Calculate clay charge density if we have atoms
                if all_clay_atoms is not None and len(all_clay_atoms) > 0:
                    try:
                        # Get total clay charge (use first frame - clay doesn't move)
                        self.u.trajectory[0]
                        total_clay_charge_sum = np.sum(all_clay_atoms.charges)
                        total_clay_charge = total_clay_charge_sum  # Store for later use
                        print(f"  Total clay charge: {total_clay_charge:+.4f} e")
                        
                        # Calculate clay density and charge density profiles (average over a few frames)
                        clay_number_density = np.zeros(n_bins)
                        clay_charge_density = np.zeros(n_bins)
                        
                        # Sample only a few frames for clay (it doesn't move much)
                        clay_sample_frames = min(100, n_frames)  # Sample 100 frames max
                        clay_step = max(1, n_frames // clay_sample_frames)
                        
                        n_clay_frames = 0
                        for ts in self.u.trajectory[::clay_step]:
                            # Get clay z-positions and apply centering
                            clay_z = all_clay_atoms.positions[:, 2] - z_offset
                            clay_charges = all_clay_atoms.charges
                            
                            # Bin clay atoms for number density
                            hist, _ = np.histogram(clay_z, bins=z_bins)
                            bin_volume = box_x * box_y * z_bin_width
                            clay_number_density += hist / bin_volume
                            
                            # Bin clay charge density
                            charge_hist, _ = np.histogram(clay_z, bins=z_bins, weights=clay_charges)
                            clay_charge_density += charge_hist / bin_volume
                            
                            n_clay_frames += 1
                        
                        # Average over sampled frames
                        clay_number_density /= n_clay_frames
                        clay_charge_density /= n_clay_frames
                        
                        print(f"  Sampled {n_clay_frames} frames for density profile")
                        print(f"  Clay charge density range: {np.min(clay_charge_density):.6f} to {np.max(clay_charge_density):.6f} e/Å³")
                        
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not calculate clay charge density: {e}")
                        clay_charge_density = None
                        clay_number_density = None
                else:
                    print("  ⚠️  Skipping clay charge density calculation (no clay atoms found)")
            
            # Determine bulk region for concentration calculation
            z_min = np.min(z_centers)
            z_max = np.max(z_centers)
            if abs(surface_position) > 10.0:  # Centered system
                if surface_position > 0:  # Top surface
                    bulk_mask = (z_centers < (surface_position - bulk_reference_distance)) & \
                            (z_centers > z_min + 2.0)
                else:  # Bottom surface
                    bulk_mask = (z_centers > (surface_position + bulk_reference_distance)) & \
                            (z_centers < z_max - 2.0)
            else:  # Asymmetric system
                bulk_mask = (z_centers > (surface_position + bulk_reference_distance)) & \
                        (z_centers < z_max - 2.0)
            
            if not np.any(bulk_mask):
                # Fallback
                if surface_position > 0:
                    bulk_mask = z_centers < (surface_position - 5.0)
                else:
                    bulk_mask = z_centers > (surface_position + 5.0)
                if not np.any(bulk_mask):
                    bulk_mask = np.abs(z_centers - surface_position) > 5.0
            
            # Avogadro's number for concentration conversion
            N_A = 6.02214076e23
            
            # Print ion density statistics
            print(f"\nIon density statistics:")
            for ion_name, density in ion_densities.items():
                if np.max(density) > 0:
                    charge = self._get_ion_charge(ion_name, charge_dict)
                    
                    # Max peak density
                    max_density = np.max(density)
                    max_molar = max_density * (1e10)**3 / N_A / 1000
                    
                    # Bulk average density
                    bulk_density = np.mean(density[bulk_mask])
                    bulk_molar = bulk_density * (1e10)**3 / N_A / 1000
                    
                    print(f"  {ion_name} (q={charge:+.0f}e):")
                    print(f"    Peak:  density = {max_density:.6f} ions/Å³,  concentration = {max_molar:.4f} M")
                    print(f"    Bulk:  density = {bulk_density:.6f} ions/Å³,  concentration = {bulk_molar:.4f} M")
            
            # Create charge density breakdown table
            print(f"\nCharge density breakdown:")
            print("="*80)
            
            # Calculate charge density contribution for each ion
            ion_charge_densities = {}
            mobile_breakdown_table = []
            
            for ion_name, density in ion_densities.items():
                charge = self._get_ion_charge(ion_name, charge_dict)
                ion_charge_density = density * charge  # e/Å³
                ion_charge_densities[ion_name] = ion_charge_density
                
                # Calculate statistics
                integrated_charge = np.trapz(ion_charge_density, dx=z_bin_width)
                max_charge_density = np.max(np.abs(ion_charge_density))
                
                mobile_breakdown_table.append({
                    'Ion': ion_name,
                    'Charge': f'{charge:+.0f}e',
                    'Integrated_Charge_eA': integrated_charge,
                    'Max_Charge_Density_e_per_A3': max_charge_density,
                    'Contribution_%': 0.0  # Will calculate after total
                })
            
            # Calculate mobile ion percentages (relative to mobile total)
            mobile_total_abs_charge = sum(abs(row['Integrated_Charge_eA']) for row in mobile_breakdown_table)
            for row in mobile_breakdown_table:
                if mobile_total_abs_charge > 0:
                    row['Contribution_%'] = 100 * abs(row['Integrated_Charge_eA']) / mobile_total_abs_charge
            
            # Calculate mobile subtotal
            mobile_total_charge = sum(row['Integrated_Charge_eA'] for row in mobile_breakdown_table)
            
            # Print hierarchical breakdown
            import pandas as pd
            
            # Calculate total charge in electrons (multiply by XY area)
            xy_area = box_x * box_y
            mobile_total_charge_electrons = mobile_total_charge * xy_area
            
            print("\n" + "="*80)
            print("MOBILE IONS (standard EDL analysis):")
            print("-"*80)
            df_mobile = pd.DataFrame(mobile_breakdown_table)
            print(df_mobile.to_string(index=False))
            print("-"*80)
            print(f"Mobile subtotal (per area):  {mobile_total_charge:+.6f} e/Å²")
            print(f"Mobile subtotal (total):     {mobile_total_charge_electrons:+.2f} e  (over {xy_area:.1f} Å² area)")
            print("="*80)
            
            # Add clay charge density if requested
            if include_clay_charge_density and clay_charge_density is not None:
                # Use actual total clay charge (not integrated from profile)
                clay_total_charge_electrons = total_clay_charge  # Direct sum, already in electrons
                clay_integrated_charge = total_clay_charge / xy_area  # Convert to charge per area
                clay_max_charge_density = np.max(clay_charge_density)  # Most positive
                clay_min_charge_density = np.min(clay_charge_density)  # Most negative
                
                # Find z-positions of max and min charge densities
                max_idx = np.argmax(clay_charge_density)
                min_idx = np.argmin(clay_charge_density)
                z_at_max = z_centers[max_idx]
                z_at_min = z_centers[min_idx]
                distance_max_from_surface = abs(z_at_max - surface_position)
                distance_min_from_surface = abs(z_at_min - surface_position)
                
                print("\nSTATIC CHARGES (clay atoms):")
                print("-"*80)
                print(f"{'Type':<6} {'Charge':<8} {'Integrated_Charge_eA':>22} {'Max_Density (e/Å³)':>22} {'Min_Density (e/Å³)':>22}")
                print(f"{'Clay':<6} {'varies':<8} {clay_integrated_charge:>22.6f} {clay_max_charge_density:>22.6f} {clay_min_charge_density:>22.6f}")
                print("-"*80)
                print(f"  Max density location: z = {z_at_max:+.2f} Å  ({distance_max_from_surface:.2f} Å from surface at {surface_position:+.2f} Å)")
                print(f"  Min density location: z = {z_at_min:+.2f} Å  ({distance_min_from_surface:.2f} Å from surface at {surface_position:+.2f} Å)")
                print("-"*80)
                print(f"Clay subtotal (per area):    {clay_integrated_charge:+.6f} e/Å²")
                print(f"Clay subtotal (total):       {clay_total_charge_electrons:+.6f} e")
                print("="*80)
                
                # Grand total
                total_system_charge_per_area = mobile_total_charge + clay_integrated_charge
                total_system_charge_electrons = mobile_total_charge_electrons + clay_total_charge_electrons
                total_abs_charge = abs(mobile_total_charge) + abs(clay_integrated_charge)
                mobile_contribution = (abs(mobile_total_charge) / total_abs_charge * 100) if total_abs_charge > 0 else 0.0
                clay_contribution = (abs(clay_integrated_charge) / total_abs_charge * 100) if total_abs_charge > 0 else 0.0
                
                print("\nTOTAL SYSTEM CHARGE (neutrality check):")
                print("-"*80)
                print(f"Integrated (per area):   {total_system_charge_per_area:+.6f} e/Å²")
                print(f"Total charge:            {total_system_charge_electrons:+.2f} e  (should be ≈0)")
                print(f"Mobile contributes:      {mobile_contribution:5.1f}% of total absolute charge")
                print(f"Clay contributes:        {clay_contribution:5.1f}% of total absolute charge")
                print("="*80)
            else:
                # Show only mobile total (current behavior when clay not included)
                mobile_total_charge_electrons = mobile_total_charge * xy_area
                print(f"\nTotal mobile charge (per area): {mobile_total_charge:+.6f} e/Å²")
                print(f"Total mobile charge (total):    {mobile_total_charge_electrons:+.2f} e")
                print("="*80)
            
            # Save detailed charge density profiles to CSV
            csv_filename = 'charge_density_profiles.csv'
            print(f"\nSaving detailed charge density profiles to: {csv_filename}")
            
            # Create dataframe with z-positions and all charge densities
            profile_data = {'z_position_A': z_centers}
            
            # Add each ion's charge density contribution
            for ion_name, ion_charge_dens in ion_charge_densities.items():
                charge = self._get_ion_charge(ion_name, charge_dict)
                profile_data[f'{ion_name}_charge_density_e_per_A3'] = ion_charge_dens
                profile_data[f'{ion_name}_number_density_per_A3'] = ion_densities[ion_name]
            
            # Add mobile total charge density
            profile_data['Mobile_charge_density_e_per_A3'] = charge_density
            
            # Add clay charge density if calculated
            if include_clay_charge_density and clay_charge_density is not None:
                profile_data['Clay_charge_density_e_per_A3'] = clay_charge_density
                profile_data['Clay_number_density_per_A3'] = clay_number_density
                profile_data['Total_charge_density_e_per_A3'] = charge_density + clay_charge_density
            else:
                profile_data['Total_charge_density_e_per_A3'] = charge_density
            
            # Save CSV if requested
            if save_charge_profiles_csv:
                df_profiles = pd.DataFrame(profile_data)
                df_profiles.to_csv(csv_filename, index=False, float_format='%.8e')
                print(f"  Saved charge density profiles to: {csv_filename}")
                print(f"  ({len(z_centers)} z-positions, columns: z_position, {', '.join([f'{ion}_charge_density' for ion in ion_charge_densities.keys()])}, Total)")
            
            # Check charge neutrality
            total_charge = np.trapz(charge_density, dx=z_bin_width)
            xy_area = box_x * box_y
            print(f"\nCharge neutrality check:")
            print(f"  Total integrated charge: {total_charge:.6f} e·Å")
            print(f"  Charge per area: {total_charge/xy_area:.6f} e/Å²")
            
            if abs(total_charge) > 1e-3:
                print(f"  ⚠️  Warning: System not charge neutral!")
            else:
                print(f"  ✓ System is approximately charge neutral")
            
            # Calculate electrostatic potential via Poisson equation
            print(f"\nSolving Poisson equation for electrostatic potential...")
            potential, electric_field = self._solve_poisson_1d_edl(
                charge_density, z_bin_width, dielectric_constant, temperature
            )
            
            print(f"  Potential range: {np.min(potential):.3f} to {np.max(potential):.3f} kT/e")
            print(f"  Electric field range: {np.min(electric_field):.3f} to {np.max(electric_field):.3f} kT/e/Å")
            
            # Calculate surface charge density
            surface_bin_idx = np.argmin(np.abs(z_centers - surface_position))
            
            # Calculate surface charge density
            print(f"\nClay surface charge analysis:")
            print(f"  Surface area: {xy_area:.2f} Å² ({box_x:.2f} × {box_y:.2f})")
            
            # Method 1: From clay atom charges (preferred)
            if clay_surface_charge is None:
                try:
                    # Get ALL clay atoms (entire layer, including wrapped portions)
                    self.u.trajectory[0]
                    clay_resname = clay_surface_sel.resnames[0]
                    
                    clay_full_sel = self.u.select_atoms(f"resname {clay_resname}")
                    
                    print(f"  Surface selection: {len(clay_surface_sel)} atoms ({clay_surface_sel.names[0]} type)")
                    print(f"  Total clay atoms: {len(clay_full_sel)} atoms (all types)")
                    print(f"  Atom types: {np.unique(clay_full_sel.names)}")
                    
                    # Sum charges from all clay atoms
                    total_clay_charge = np.sum(clay_full_sel.charges)
                    print(f"  Source: All clay atoms (resname {clay_resname})")
                    print(f"  Total charge: {total_clay_charge:.4f} e")
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not calculate clay surface charge ({e})")
                    print(f"     Assuming neutral clay (σ=0). Specify clay_surface_charge parameter.")
                    total_clay_charge = 0.0
            else:
                total_clay_charge = clay_surface_charge
                print(f"  Source: User-specified")
                print(f"  Total charge: {total_clay_charge:.4f} e")
            
            # Surface charge density
            surface_charge_density = total_clay_charge / xy_area
            
            print(f"  Surface charge density:")
            print(f"    σ_surface = {surface_charge_density:.6f} e/Å²")
            print(f"    σ_surface = {surface_charge_density * 1.602e-19 / (1e-10)**2:.6f} C/m²")
            
            # Initialize results dictionary
            results = {
                'z_centers': z_centers,
                'charge_density': charge_density,  # Mobile ions only
                'electrostatic_potential': potential,
                'electric_field': electric_field,
                'ion_densities': ion_densities,
                'surface_position': surface_position,
                'surface_position_std': surface_std,
                'surface_charge_density': surface_charge_density,
                'metadata': {
                    'n_frames': n_frames,
                    'z_bin_width': z_bin_width,
                    'temperature': temperature,
                    'dielectric_constant': dielectric_constant,
                    'box_dimensions': (box_x, box_y, box_z),
                    'charge_dict': charge_dict,
                    'charge_neutrality': abs(total_charge) < 1e-3,
                    'include_clay_charge_density': include_clay_charge_density
                }
            }
            
            # Add clay charge density if calculated
            if include_clay_charge_density and clay_charge_density is not None:
                results['clay_charge_density'] = clay_charge_density
                results['clay_number_density'] = clay_number_density
                results['total_charge_density'] = charge_density + clay_charge_density  # Mobile + clay
            
            # 1. STERN LAYER IDENTIFICATION
            if identify_stern_layer:
                print(f"\n" + "-"*80)
                print("STERN LAYER ANALYSIS")
                print("-"*80)
                stern_layer_results = self._identify_stern_layer(
                    z_centers, ion_densities, surface_position, charge_dict, use_manual_peaks
                )
                results['stern_layer'] = stern_layer_results
            
            # 2. DEBYE LENGTH CALCULATION
            if calculate_debye_length:
                print(f"\n" + "-"*80)
                print("DEBYE LENGTH CALCULATION")
                print("-"*80)
                
                # Use OHP position if Stern layer was identified, otherwise use default
                ohp_distance = 5.0  # Default
                if identify_stern_layer and 'stern_layer' in results and 'ohp_position' in results['stern_layer']:
                    ohp_distance = abs(results['stern_layer']['ohp_position'] - surface_position)
                    print(f"  Using OHP position ({ohp_distance:.1f} Å from surface) to define diffuse layer start")
                else:
                    print(f"  Using default diffuse layer start ({ohp_distance:.1f} Å from surface)")
                
                debye_results = self._calculate_debye_length(
                    z_centers, potential, ion_densities, surface_position,
                    temperature, dielectric_constant, charge_dict, bulk_reference_distance,
                    ohp_distance
                )
                results['debye_length'] = debye_results
            
            # 3. ADSORPTION MODE CLASSIFICATION
            if classify_adsorption_modes and identify_stern_layer:
                print(f"\n" + "-"*80)
                print("ION ADSORPTION MODE CLASSIFICATION")
                print("-"*80)
                adsorption_results = self._classify_adsorption_modes(
                    ion_densities, z_centers, surface_position,
                    stern_layer_results, n_frames, box_x, box_y, z_bin_width
                )
                results['adsorption_modes'] = adsorption_results
            
            # 4. GOUY-CHAPMAN COMPARISON
            if compare_gouy_chapman:
                print(f"\n" + "-"*80)
                print("GOUY-CHAPMAN THEORY COMPARISON")
                print("-"*80)
                gc_results = self._compare_gouy_chapman_theory(
                    z_centers, potential, ion_densities, surface_position,
                    surface_charge_density, temperature, dielectric_constant,
                    charge_dict, bulk_reference_distance
                )
                results['gouy_chapman_comparison'] = gc_results
            
            # Store results
            self.results['edl_analysis'] = results
            
            # === SAVE CACHE ===
            if save_cache:
                try:
                    print(f"\n💾 Saving results to cache: {cache_file}")
                    
                    # Prepare data for saving
                    save_dict = {
                        'z_centers': results['z_centers'],
                        'charge_density': results['charge_density'],
                        'electrostatic_potential': results['electrostatic_potential'],
                        'electric_field': results['electric_field'],
                        'ion_densities': results['ion_densities'],
                        'surface_position': results['surface_position'],
                        'surface_position_std': results['surface_position_std'],
                        'surface_charge_density': results['surface_charge_density'],
                        'metadata': results['metadata']
                    }
                    
                    # Add optional analyses if computed
                    if 'stern_layer' in results:
                        save_dict['stern_layer'] = results['stern_layer']
                    if 'debye_length' in results:
                        save_dict['debye_length'] = results['debye_length']
                    if 'adsorption_modes' in results:
                        save_dict['adsorption_modes'] = results['adsorption_modes']
                    if 'gouy_chapman_comparison' in results:
                        save_dict['gouy_chapman_comparison'] = results['gouy_chapman_comparison']
                    
                    # Save to npz file
                    np.savez_compressed(cache_file, **save_dict)
                    print(f"✅ Cache saved successfully!")
                    
                except Exception as e:
                    print(f"⚠️  Cache save failed: {e}")
            
            # === PRINT SUMMARY REPORT ===
            if summary_report:
                self._print_edl_summary_report(results)
            
            print(f"\n" + "="*80)
            print("ELECTRICAL DOUBLE LAYER ANALYSIS COMPLETE")
            print("="*80)
            




    def analyze_edl_both_surfaces(self, clay_surface_top, clay_surface_bottom,
                                  top_peak_positions=None, bottom_peak_positions=None,
                                  print_comparison=True, **edl_params):
        """
        Analyze EDL for both top and bottom clay surfaces and compare results.
        
        Convenience method that runs analyze_electrical_double_layer_complete() for
        both surfaces with automatic peak detection setup and returns organized results.
        
        Parameters
        ----------
        clay_surface_top : AtomGroup
            Selection of top clay surface atoms
        clay_surface_bottom : AtomGroup
            Selection of bottom clay surface atoms
        top_peak_positions : dict, optional
            Manual peak positions for top surface, e.g. {'Na': [21.9, 19.7], 'Cl': [15.1]}
            If None, uses automatic peak detection
        bottom_peak_positions : dict, optional
            Manual peak positions for bottom surface, e.g. {'Na': [-23.7, -22.1], 'Cl': [-15.3]}
            If None, uses automatic peak detection
        print_comparison : bool, default=True
            If True, prints side-by-side comparison of key metrics
        **edl_params : dict
            Additional parameters passed to analyze_electrical_double_layer_complete()
            Common parameters: z_bin_width, temperature, dielectric_constant, step, njobs, etc.
            
        Returns
        -------
        dict
            Results dictionary with keys 'top' and 'bottom', each containing full EDL analysis
            
        Examples
        --------
        >>> results = occ_analysis.analyze_edl_both_surfaces(
        ...     clay_surface_top=clay_surface_top,
        ...     clay_surface_bottom=clay_surface_bottom,
        ...     top_peak_positions={'Na': [21.9, 19.7], 'Cl': [15.1]},
        ...     bottom_peak_positions={'Na': [-23.7, -22.1], 'Cl': [-15.3]},
        ...     center_box=True,
        ...     z_bin_width=0.2,
        ...     njobs=5
        ... )
        >>> print(results['top']['surface_charge_density'])
        >>> print(results['bottom']['surface_charge_density'])
        """
        
        print("\n" + "="*80)
        print("DUAL SURFACE EDL ANALYSIS")
        print("="*80)
        
        # Set manual peaks for bottom surface if provided
        if bottom_peak_positions is not None:
            print("\n🔻 Setting manual peaks for BOTTOM surface...")
            self.analyze_ion_peaks_manual(
                peak_positions_dict=bottom_peak_positions,
                show_plot=False
            )
        
        # Analyze bottom surface
        print("\n" + "="*80)
        print("ANALYZING BOTTOM SURFACE")
        print("="*80)
        edl_bottom = self.analyze_electrical_double_layer_complete(
            clay_surface_sel=clay_surface_bottom,
            use_manual_peaks=(bottom_peak_positions is not None),
            **edl_params
        )
        
        # Set manual peaks for top surface if provided
        if top_peak_positions is not None:
            print("\n🔺 Setting manual peaks for TOP surface...")
            self.analyze_ion_peaks_manual(
                peak_positions_dict=top_peak_positions,
                show_plot=False
            )
        
        # Analyze top surface
        print("\n" + "="*80)
        print("ANALYZING TOP SURFACE")
        print("="*80)
        edl_top = self.analyze_electrical_double_layer_complete(
            clay_surface_sel=clay_surface_top,
            use_manual_peaks=(top_peak_positions is not None),
            **edl_params
        )
        
        # Organize results
        results = {
            'top': edl_top,
            'bottom': edl_bottom
        }
        
        # Print comparison if requested
        if print_comparison:
            self._print_edl_comparison(edl_top, edl_bottom)
        
        return results
    
    def _print_edl_comparison(self, edl_top, edl_bottom):
        """Print side-by-side comparison of top and bottom EDL analyses"""
        
        print("\n" + "="*80)
        print("EDL COMPARISON: Top vs Bottom Surfaces".center(80))
        print("="*80)
        
        # Surface properties
        print(f"\n{'Property':<35} {'Bottom':>20} {'Top':>20}")
        print("-"*80)
        print(f"{'Surface position (Å)':<35} {edl_bottom['surface_position']:>20.2f} {edl_top['surface_position']:>20.2f}")
        print(f"{'Surface charge density (e/Å²)':<35} {edl_bottom['surface_charge_density']:>20.6f} {edl_top['surface_charge_density']:>20.6f}")
        
        # Stern layer
        if 'stern_layer' in edl_bottom and 'stern_layer' in edl_top:
            print(f"\n{'Stern Layer':<35} {'Bottom':>20} {'Top':>20}")
            print("-"*80)
            
            ihp_dist_bottom = abs(edl_bottom['stern_layer']['ihp_position'] - edl_bottom['surface_position'])
            ihp_dist_top = abs(edl_top['stern_layer']['ihp_position'] - edl_top['surface_position'])
            print(f"{'IHP distance from surface (Å)':<35} {ihp_dist_bottom:>20.2f} {ihp_dist_top:>20.2f}")
            
            print(f"{'Stern thickness (Å)':<35} {edl_bottom['stern_layer']['stern_thickness']:>20.2f} {edl_top['stern_layer']['stern_thickness']:>20.2f}")
        
        # Debye length
        if 'debye_length' in edl_bottom and 'debye_length' in edl_top:
            print(f"\n{'Debye Length':<35} {'Bottom':>20} {'Top':>20}")
            print("-"*80)
            print(f"{'Theoretical (Å)':<35} {edl_bottom['debye_length']['lambda_D_theoretical']:>20.2f} {edl_top['debye_length']['lambda_D_theoretical']:>20.2f}")
            if edl_bottom['debye_length'].get('lambda_D_fitted') and edl_top['debye_length'].get('lambda_D_fitted'):
                print(f"{'Fitted (Å)':<35} {edl_bottom['debye_length']['lambda_D_fitted']:>20.2f} {edl_top['debye_length']['lambda_D_fitted']:>20.2f}")
        
        # Symmetry assessment
        print(f"\n{'Symmetry Assessment':<35}")
        print("-"*80)
        charge_diff = abs(edl_top['surface_charge_density'] - edl_bottom['surface_charge_density'])
        charge_avg = (abs(edl_top['surface_charge_density']) + abs(edl_bottom['surface_charge_density'])) / 2
        if charge_avg > 0:
            symmetry = (1 - charge_diff / charge_avg) * 100
            print(f"{'Charge density symmetry (%)':<35} {symmetry:>20.1f}")
        
        print("="*80)
    
    def _print_edl_summary_report(self, results):
        """Print formatted summary table of EDL analysis results"""
        
        print("\n" + "="*80)
        print("EDL ANALYSIS SUMMARY REPORT".center(80))
        print("="*80)
        
        # Surface Properties Table
        print("\n" + "─"*80)
        print("SURFACE PROPERTIES".center(80))
        print("─"*80)
        print(f"{'Property':<40} {'Value':>20} {'Unit':>15}")
        print("─"*80)
        print(f"{'Surface charge density':<40} {results['surface_charge_density']:>20.6f} {'e/Å²':>15}")
        print("─"*80)
        
        # Stern Layer Table
        if 'stern_layer' in results:
            stern = results['stern_layer']
            print("\n" + "─"*80)
            print("STERN LAYER".center(80))
            print("─"*80)
            print(f"{'Parameter':<40} {'Value':>20} {'Unit':>15}")
            print("─"*80)
            print(f"{'Inner Helmholtz Plane (IHP)':<40} {stern['ihp_position']:>20.2f} {'Å':>15}")
            print(f"{'Outer Helmholtz Plane (OHP)':<40} {stern['ohp_position']:>20.2f} {'Å':>15}")
            print(f"{'Stern layer thickness':<40} {stern['stern_thickness']:>20.2f} {'Å':>15}")
            if 'ihp_density' in stern:
                print(f"{'IHP peak density':<40} {stern['ihp_density']:>20.4f} {'1/Å³':>15}")
            if 'ohp_density' in stern:
                print(f"{'OHP peak density':<40} {stern['ohp_density']:>20.4f} {'1/Å³':>15}")
            print("─"*80)
        
        # Debye Length Table
        if 'debye_length' in results:
            debye = results['debye_length']
            print("\n" + "─"*80)
            print("DEBYE SCREENING LENGTH".center(80))
            print("─"*80)
            print(f"{'Parameter':<40} {'Value':>20} {'Unit':>15}")
            print("─"*80)
            print(f"{'Theoretical (from bulk conc.)':<40} {debye['lambda_D_theoretical']:>20.2f} {'Å':>15}")
            if debye.get('lambda_D_fitted'):
                print(f"{'Fitted (from potential decay)':<40} {debye['lambda_D_fitted']:>20.2f} {'Å':>15}")
            else:
                print(f"{'Fitted (from potential decay)':<40} {'N/A':>20} {'':>15}")
            print("─"*80)
            
            # Bulk Concentrations Table
            print("\n" + "─"*80)
            print("BULK ION CONCENTRATIONS".center(80))
            print("─"*80)
            print(f"{'Ion':<15} {'Concentration (M)':>25} {'Concentration (mM)':>25}")
            print("─"*80)
            for ion, data in sorted(debye['bulk_concentrations'].items()):
                conc_M = data['concentration_M']
                conc_mM = conc_M * 1000
                print(f"{ion:<15} {conc_M:>25.4f} {conc_mM:>25.2f}")
            print("─"*80)
        
        # Ion Adsorption Modes Table
        if 'adsorption_modes' in results:
            print("\n" + "─"*80)
            print("ION ADSORPTION MODES".center(80))
            print("─"*80)
            print(f"{'Ion':<10} {'Inner-sphere (%)':>20} {'Outer-sphere (%)':>20} {'Diffuse layer (%)':>20}")
            print("─"*80)
            for ion, modes in sorted(results['adsorption_modes'].items()):
                inner = modes['fraction_inner_sphere'] * 100
                outer = modes['fraction_outer_sphere'] * 100
                diffuse = modes['fraction_diffuse_layer'] * 100
                print(f"{ion:<10} {inner:>20.1f} {outer:>20.1f} {diffuse:>20.1f}")
            print("─"*80)
            
            # Detailed counts
            print("\n" + "─"*80)
            print("ION COUNTS BY REGION".center(80))
            print("─"*80)
            print(f"{'Ion':<10} {'Inner-sphere':>20} {'Outer-sphere':>20} {'Diffuse layer':>20} {'Total':>15}")
            print("─"*80)
            for ion, modes in sorted(results['adsorption_modes'].items()):
                inner_count = modes['inner_sphere_count']
                outer_count = modes['outer_sphere_count']
                diffuse_count = modes['diffuse_layer_count']
                total = modes['total_count']
                print(f"{ion:<10} {inner_count:>20.2f} {outer_count:>20.2f} {diffuse_count:>20.2f} {total:>15.2f}")
            print("─"*80)
    
    def _get_ion_charge(self, ion_name, charge_dict):
        """Helper to get ion charge with flexible matching"""
        ion_normalized = ''.join(c for c in ion_name.upper() if c.isalpha())
        
        if ion_normalized in charge_dict:
            return charge_dict[ion_normalized]
        
        # Substring matching
        for key, value in charge_dict.items():
            if key.upper() in ion_name.upper() or ion_name.upper() in key.upper():
                return value
        
        return 0.0
    
    def _solve_poisson_1d_edl(self, charge_density_per_angstrom3, dz, epsilon_r, T):
        """Solve 1D Poisson equation for EDL"""
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve
        
        n = len(charge_density_per_angstrom3)
        
        # Convert units: e/Å³ → C/m³
        e = 1.602e-19  # C
        charge_density_SI = charge_density_per_angstrom3 * e / (1e-10)**3
        
        # Convert dz from Å to m
        dz_m = dz * 1e-10
        
        # Physical constants
        epsilon_0 = 8.854e-12  # F/m
        k_B = 1.381e-23  # J/K
        
        # RHS: -ρ(z)/(ε₀ε_r)
        rhs = -charge_density_SI / (epsilon_0 * epsilon_r)
        
        # Finite difference matrix: d²φ/dz² ≈ (φ[i+1] - 2φ[i] + φ[i-1])/dz²
        main_diag = -2.0 * np.ones(n) / (dz_m**2)
        off_diag = 1.0 * np.ones(n-1) / (dz_m**2)
        
        A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n, n), format='csr')
        
        # Periodic boundary conditions
        A[0, -1] = 1.0 / (dz_m**2)
        A[-1, 0] = 1.0 / (dz_m**2)
        
        # Enforce charge neutrality for periodic BC
        if abs(np.sum(rhs) * dz_m) > 1e-10:
            rhs -= np.mean(rhs)
        
        # Solve
        potential_SI = spsolve(A, rhs)
        potential_SI = potential_SI - np.mean(potential_SI)
        
        # Convert to kT/e units
        potential_kTe = potential_SI * e / (k_B * T)
        
        # Calculate electric field: E = -dψ/dz
        electric_field_kTe = -np.gradient(potential_kTe, dz)
        
        return potential_kTe, electric_field_kTe
    
    def _identify_stern_layer(self, z_centers, ion_densities, surface_position, charge_dict, use_manual_peaks=False):
        """Identify Stern layer from ion density minima"""
        
        # Check if manual peak data exists and user wants to use it
        manual_peaks = None
        if use_manual_peaks and 'ion_peak_analysis' in self.results:
            manual_peaks = self.results['ion_peak_analysis']
            print("  ℹ️  Using MANUAL peak data for IHP/OHP (use_manual_peaks=True)")
        elif use_manual_peaks and 'ion_peak_analysis' not in self.results:
            print("  ⚠️  use_manual_peaks=True but no manual peaks found - using automatic detection")
        
        # Find first minimum in total cation density after surface
        cation_density = np.zeros_like(z_centers)
        anion_density = np.zeros_like(z_centers)
        
        for ion_name, density in ion_densities.items():
            charge = self._get_ion_charge(ion_name, charge_dict)
            if charge > 0:
                cation_density += density
            elif charge < 0:
                anion_density += density
        
        # Detect surface direction: ions accumulate INWARD toward box center
        # Top surface (z > 0): ions are BELOW surface (z < surface)
        # Bottom surface (z < 0): ions are ABOVE surface (z > surface)
        is_top_surface = surface_position > 0
        
        # If manual peaks exist, use them for IHP/OHP
        if manual_peaks is not None:
            # Find dominant cation (highest charge * peak density across all peaks)
            dominant_cation = None
            best_peaks = None
            max_total_weight = 0
            
            for ion_name, peak_data in manual_peaks.items():
                charge = self._get_ion_charge(ion_name, charge_dict)
                if charge > 0 and peak_data.get('n_peaks', 0) >= 2:  # Cation with at least 2 peaks
                    peak_positions = peak_data['peak_positions']
                    peak_densities = peak_data['peak_densities']
                    
                    # Calculate total weight (charge × sum of top 2 peak densities)
                    if len(peak_positions) >= 2:
                        total_weight = charge * np.sum(sorted(peak_densities, reverse=True)[:2])
                        if total_weight > max_total_weight:
                            max_total_weight = total_weight
                            dominant_cation = ion_name
                            best_peaks = (peak_positions, peak_densities)
            
            if best_peaks is not None:
                peak_positions, peak_densities = best_peaks
                
                # Sort peaks by DENSITY (highest first), not by distance
                sorted_indices = np.argsort(peak_densities)[::-1]  # Descending order
                sorted_positions = np.array(peak_positions)[sorted_indices]
                sorted_densities = np.array(peak_densities)[sorted_indices]
                
                # IHP = highest peak, OHP = 2nd highest peak
                ihp_position = sorted_positions[0]
                ihp_density = sorted_densities[0]
                ohp_position = sorted_positions[1]
                ohp_density = sorted_densities[1]
                
                stern_thickness = abs(ihp_position - ohp_position)
                
                print(f"  Stern layer from MANUAL peaks (selected by peak height):")
                print(f"    Dominant cation: {dominant_cation}")
                print(f"    Inner Helmholtz Plane (IHP): {ihp_position:.2f} Å (highest peak, density={ihp_density:.4f})")
                print(f"    Outer Helmholtz Plane (OHP): {ohp_position:.2f} Å (2nd highest peak, density={ohp_density:.4f})")
                print(f"    Stern layer thickness: {stern_thickness:.2f} Å")
                
                return {
                    'ihp_position': ihp_position,
                    'ohp_position': ohp_position,
                    'stern_thickness': stern_thickness,
                    'method': 'manual_peaks',
                    'dominant_cation': dominant_cation,
                    'ihp_density': ihp_density,
                    'ohp_density': ohp_density
                }
        
        # Fall back to automatic peak finding if no manual peaks
        if is_top_surface:
            # Top surface: look for ions BELOW (toward center)
            beyond_surface_idx = z_centers < surface_position
        else:
            # Bottom surface: look for ions ABOVE (toward center)
            beyond_surface_idx = z_centers > surface_position
        
        if not np.any(beyond_surface_idx):
            print("  ⚠️  Warning: No data points toward solution from surface")
            return {'ihp_position': None, 'ohp_position': None}
        
        # Limit search to reasonable distance from surface (max 15 Å)
        # This prevents finding features from the opposite surface
        max_search_distance = 15.0  # Å
        
        if is_top_surface:
            # Top surface: look 15 Å below surface
            search_region_idx = (z_centers < surface_position) & (z_centers > surface_position - max_search_distance)
        else:
            # Bottom surface: look 15 Å above surface
            search_region_idx = (z_centers > surface_position) & (z_centers < surface_position + max_search_distance)
        
        if not np.any(search_region_idx):
            print(f"  ⚠️  Warning: No data in search region ({max_search_distance} Å from surface)")
            # Fall back to estimates
            if is_top_surface:
                ihp_position = surface_position - 2.5
                ohp_position = surface_position - 5.0
            else:
                ihp_position = surface_position + 2.5
                ohp_position = surface_position + 5.0
            return {
                'ihp_position': ihp_position,
                'ohp_position': ohp_position,
                'stern_thickness': abs(ohp_position - ihp_position),
                'note': 'Estimated values (insufficient search region)'
            }
        
        # Find first minimum for cations (typically closer to negatively charged surface)
        z_search = z_centers[search_region_idx]
        cation_search = cation_density[search_region_idx]
        
        # CRITICAL: Sort data from surface toward bulk for consistent peak detection
        # Top surface: sort descending (high z to low z, surface to center)
        # Bottom surface: sort ascending (low z to high z, surface to center)
        if is_top_surface:
            sort_idx = np.argsort(z_search)[::-1]  # Descending
        else:
            sort_idx = np.argsort(z_search)  # Ascending
        
        z_search = z_search[sort_idx]
        cation_search = cation_search[sort_idx]
        
        # Look for first minimum
        from scipy.signal import argrelextrema
        minima_idx = argrelextrema(cation_search, np.less)[0]
        
        if len(minima_idx) > 0 and minima_idx[0] > 0:
            # Inner Helmholtz Plane (IHP) - first peak position before first minimum
            first_peak_idx = argrelextrema(cation_search[:minima_idx[0]], np.greater)[0]
            if len(first_peak_idx) > 0:
                ihp_idx = first_peak_idx[-1]
                ihp_position = z_search[ihp_idx]
            else:
                # Use surface ± typical ion radius if no peak found
                if is_top_surface:
                    ihp_position = surface_position - 2.5
                else:
                    ihp_position = surface_position + 2.5
            
            # Outer Helmholtz Plane (OHP) - first minimum
            ohp_idx = minima_idx[0]
            ohp_position = z_search[ohp_idx]
            
            stern_thickness = abs(ohp_position - ihp_position)
            
            print(f"  Stern layer identified:")
            print(f"    Inner Helmholtz Plane (IHP): {ihp_position:.2f} Å")
            print(f"    Outer Helmholtz Plane (OHP): {ohp_position:.2f} Å")
            print(f"    Stern layer thickness: {stern_thickness:.2f} Å")
            
            return {
                'ihp_position': ihp_position,
                'ohp_position': ohp_position,
                'stern_thickness': stern_thickness,
                'cation_density_at_ihp': cation_density[np.argmin(np.abs(z_centers - ihp_position))],
                'cation_density_at_ohp': cation_density[np.argmin(np.abs(z_centers - ohp_position))]
            }
        else:
            print("  ⚠️  Warning: Could not identify clear Stern layer minimum")
            # Estimate based on typical values
            # Top surface: subtract (ions below), Bottom surface: add (ions above)
            is_top_surface = surface_position > 0
            if is_top_surface:
                ihp_position = surface_position - 2.5
                ohp_position = surface_position - 5.0
            else:
                ihp_position = surface_position + 2.5
                ohp_position = surface_position + 5.0
            
            return {
                'ihp_position': ihp_position,
                'ohp_position': ohp_position,
                'stern_thickness': abs(ohp_position - ihp_position),
                'note': 'Estimated values (no clear minimum found)'
            }
    
    def _calculate_debye_length(self, z_centers, potential, ion_densities,
                                surface_position, T, epsilon_r, charge_dict,
                                bulk_reference_distance, diffuse_layer_start=5.0):
        """Calculate Debye length from theory and fit
        
        Parameters
        ----------
        diffuse_layer_start : float
            Distance from surface where diffuse layer begins (typically OHP position).
            Default is 5.0 Å if Stern layer not identified.
        """
        
        # Physical constants
        k_B = 1.381e-23  # J/K
        e = 1.602e-19    # C
        epsilon_0 = 8.854e-12  # F/m
        N_A = 6.022e23
        
        # 1. Theoretical Debye length from bulk concentrations
        # Find bulk region (far from surface, but within box)
        z_min = np.min(z_centers)
        z_max = np.max(z_centers)
        
        # Determine bulk region based on system geometry
        # For centered systems, bulk is near z=0 (center)
        # For asymmetric systems, bulk is far from surface
        if abs(surface_position) > 10.0:  # Likely a centered system with surface away from z=0
            # Bulk is toward the center, away from surface
            if surface_position > 0:  # Top surface
                # Bulk is below surface, toward z=0
                bulk_mask = (z_centers < (surface_position - bulk_reference_distance)) & \
                           (z_centers > z_min + 2.0)
            else:  # Bottom surface
                # Bulk is above surface, toward z=0
                bulk_mask = (z_centers > (surface_position + bulk_reference_distance)) & \
                           (z_centers < z_max - 2.0)
        else:
            # Asymmetric system or surface near origin
            # Bulk is far from surface in positive direction
            bulk_mask = (z_centers > (surface_position + bulk_reference_distance)) & \
                       (z_centers < z_max - 2.0)
        
        if not np.any(bulk_mask):
            print(f"  ⚠️  Warning: No bulk region found with bulk_reference_distance={bulk_reference_distance:.1f} Å")
            print(f"     Surface at {surface_position:.1f} Å, box range: [{z_min:.1f}, {z_max:.1f}] Å")
            # Fallback: use region far from surface (either direction)
            if surface_position > 0:
                bulk_mask = z_centers < (surface_position - 5.0)
            else:
                bulk_mask = z_centers > (surface_position + 5.0)
            
        if not np.any(bulk_mask):
            print("  ⚠️  Warning: Still no bulk region, using all regions away from immediate surface")
            bulk_mask = np.abs(z_centers - surface_position) > 5.0
        
        # Calculate ionic strength: I = 0.5 * Σ c_i z_i²
        ionic_strength_SI = 0.0
        bulk_concentrations = {}
        
        for ion_name, density in ion_densities.items():
            charge = self._get_ion_charge(ion_name, charge_dict)
            if charge != 0:
                # Bulk concentration in ions/Å³
                c_bulk = np.mean(density[bulk_mask])
                
                # Convert to mol/L (M)
                # ions/Å³ → ions/m³ → mol/m³ → mol/L
                c_bulk_M = c_bulk * (1e10)**3 / N_A / 1000
                
                # Convert to ions/m³ for SI calculation
                c_bulk_SI = c_bulk * (1e10)**3
                
                ionic_strength_SI += 0.5 * c_bulk_SI * charge**2
                
                bulk_concentrations[ion_name] = {
                    'density': c_bulk,
                    'concentration_M': c_bulk_M,
                    'charge': charge
                }
        
        # Debye length: λ_D = sqrt(ε₀ ε_r k_B T / (e² Σ c_i z_i²))
        if ionic_strength_SI > 0:
            lambda_D_theory = np.sqrt(epsilon_0 * epsilon_r * k_B * T / (e**2 * 2 * ionic_strength_SI))
            lambda_D_theory_angstrom = lambda_D_theory * 1e10
        else:
            lambda_D_theory_angstrom = np.inf
        
        print(f"  Theoretical Debye length:")
        print(f"    λ_D = {lambda_D_theory_angstrom:.2f} Å")
        print(f"  Bulk ion concentrations:")
        for ion_name, data in bulk_concentrations.items():
            # Only print ions with measurable concentration (> 0.0001 M)
            if data['concentration_M'] > 0.0001:
                print(f"    {ion_name}: {data['concentration_M']:.4f} M (z={data['charge']:+.0f})")
        
        # 2. Fitted Debye length from potential decay
        # Find diffuse layer region where exponential decay is strongest
        # Start from OHP (end of Stern layer) and extend to where decay completes
        # For λ_D ~ 3 Å, decay is complete within ~10-12 Å (3-4 × λ_D)
        # Direction depends on surface position
        is_top_surface = surface_position > 0
        
        diffuse_end = diffuse_layer_start + 7.0  # 7 Å range for fitting
        
        if is_top_surface:
            # Top surface: look inward (below surface)
            fit_mask = (z_centers < surface_position - diffuse_layer_start) & \
                      (z_centers > surface_position - diffuse_end)
        else:
            # Bottom surface: look upward (above surface)
            fit_mask = (z_centers > surface_position + diffuse_layer_start) & \
                      (z_centers < surface_position + diffuse_end)
        
        if np.sum(fit_mask) > 5:
            # Use absolute distance from surface for fitting
            z_fit = np.abs(z_centers[fit_mask] - surface_position)
            psi_fit = np.abs(potential[fit_mask])
            
            # Check if there's meaningful potential variation
            psi_range = np.max(psi_fit) - np.min(psi_fit)
            
            if psi_range < 0.01:  # Very flat potential, skip fitting
                print(f"  ⚠️  Potential too flat in diffuse layer (range: {psi_range:.4f} kT/e)")
                print(f"      Cannot reliably fit Debye length")
                lambda_D_fitted = None
                fit_successful = False
            else:
                # Fit exponential decay: ψ(z) = ψ_0 * exp(-z/λ_D)
                try:
                    from scipy.optimize import curve_fit
                    
                    def exp_decay(z, psi0, lambda_d):
                        return psi0 * np.exp(-z / lambda_d)
                    
                    # Initial guess
                    p0 = [np.max(psi_fit), lambda_D_theory_angstrom if lambda_D_theory_angstrom < np.inf else 10.0]
                    
                    # Set bounds to prevent unreasonable values
                    # Debye length should be between 0.5 Å and 100 Å for typical systems
                    bounds = ([0, 0.5], [np.inf, 100.0])
                    
                    popt, pcov = curve_fit(exp_decay, z_fit, psi_fit, p0=p0, bounds=bounds, maxfev=10000)
                    lambda_D_fitted = popt[1]
                    
                    # Check if fit is reasonable
                    if lambda_D_fitted > 50.0:
                        print(f"  ⚠️  Fitted Debye length unusually large ({lambda_D_fitted:.2f} Å)")
                        print(f"      Potential may be too flat for reliable fitting")
                    
                    print(f"  Fitted Debye length from potential decay:")
                    print(f"    λ_D,fit = {lambda_D_fitted:.2f} Å")
                    print(f"    Ratio (theory/fitted): {lambda_D_theory_angstrom/lambda_D_fitted:.2f}")
                    fit_successful = True
                    
                    return {
                        'lambda_D_theoretical': lambda_D_theory_angstrom,
                        'lambda_D_fitted': lambda_D_fitted,
                        'ionic_strength': 2 * ionic_strength_SI,
                        'bulk_concentrations': bulk_concentrations,
                        'fit_successful': fit_successful
                    }
                except Exception as e:
                    print(f"  ⚠️  Exponential fit failed: {e}")
                    lambda_D_fitted = None
                    fit_successful = False
                    
            return {
                'lambda_D_theoretical': lambda_D_theory_angstrom,
                'lambda_D_fitted': lambda_D_fitted,
                'ionic_strength': 2 * ionic_strength_SI,
                'bulk_concentrations': bulk_concentrations,
                'fit_successful': fit_successful
            }
        else:
            print("  ⚠️  Insufficient data for Debye length fitting")
            return {
                'lambda_D_theoretical': lambda_D_theory_angstrom,
                'lambda_D_fitted': None,
                'bulk_concentrations': bulk_concentrations,
                'fit_successful': False
            }
    
    def _classify_adsorption_modes(self, ion_densities, z_centers, surface_position,
                                   stern_layer, n_frames, box_x, box_y, z_bin_width):
        """Classify ions into inner-sphere, outer-sphere, and diffuse layer"""
        
        ihp = stern_layer.get('ihp_position', surface_position + 2.5)
        ohp = stern_layer.get('ohp_position', surface_position + 5.0)
        
        # Get box limits
        z_max = np.max(z_centers)
        z_min = np.min(z_centers)
        
        print(f"  Surface position: {surface_position:.2f} Å")
        print(f"  Z-data range: {z_min:.2f} to {z_max:.2f} Å")
        print(f"  IHP: {ihp:.2f} Å, OHP: {ohp:.2f} Å")
        
        # Detect surface direction from IHP position relative to surface
        # Top surface: IHP < surface (ions below), Bottom surface: IHP > surface (ions above)
        is_top_surface = ihp < surface_position
        
        # Clip IHP/OHP to box boundaries based on surface direction
        if is_top_surface:
            # Top surface: clip upper bounds
            ihp_clipped = min(ihp, z_max - 0.1)
            ohp_clipped = min(ohp, z_max - 0.1)
        else:
            # Bottom surface: clip lower bounds
            ihp_clipped = max(ihp, z_min + 0.1)
            ohp_clipped = max(ohp, z_min + 0.1)
        
        if ihp != ihp_clipped or ohp != ohp_clipped:
            print(f"  ⚠️  Warning: Stern layer positions exceed box boundary!")
            if ihp != ihp_clipped:
                print(f"     IHP clipped: {ihp:.2f} → {ihp_clipped:.2f} Å")
            if ohp != ohp_clipped:
                print(f"     OHP clipped: {ohp:.2f} → {ohp_clipped:.2f} Å")
        
        # Limit diffuse layer to box boundaries
        # Top surface: ions below surface, diffuse extends downward from OHP
        # Bottom surface: ions above surface, diffuse extends upward from OHP
        if is_top_surface:
            # Diffuse layer extends from OHP toward bulk (lower z)
            # Limit to avoid opposite surface (at least 15 Å from OHP or box edge)
            diffuse_layer_end = max(ohp_clipped - 15.0, z_min + 0.5)
        else:
            # Diffuse layer extends from OHP toward bulk (higher z)
            diffuse_layer_end = min(ohp_clipped + 15.0, z_max - 0.5)
        
        results = {}
        
        for ion_name, density in ion_densities.items():
            # Define regions based on surface direction
            if is_top_surface:
                # Top surface: ions BELOW surface (lower z), regions extend DOWNWARD
                # Surface (highest z) → IHP → OHP → diffuse (lowest z)
                inner_sphere_mask = (z_centers <= surface_position) & (z_centers > ihp_clipped)
                
                if ohp_clipped < ihp_clipped - 0.1:
                    outer_sphere_mask = (z_centers <= ihp_clipped) & (z_centers > ohp_clipped)
                else:
                    outer_sphere_mask = np.zeros_like(z_centers, dtype=bool)
                
                if diffuse_layer_end < ohp_clipped - 0.1:
                    diffuse_layer_mask = (z_centers <= ohp_clipped) & (z_centers > diffuse_layer_end)
                else:
                    diffuse_layer_mask = np.zeros_like(z_centers, dtype=bool)
            else:
                # Bottom surface: ions ABOVE surface (higher z), regions extend UPWARD
                # Surface (lowest z) → IHP → OHP → diffuse (highest z)
                inner_sphere_mask = (z_centers >= surface_position) & (z_centers < ihp_clipped)
                
                if ohp_clipped > ihp_clipped + 0.1:
                    outer_sphere_mask = (z_centers >= ihp_clipped) & (z_centers < ohp_clipped)
                else:
                    outer_sphere_mask = np.zeros_like(z_centers, dtype=bool)
                
                if diffuse_layer_end > ohp_clipped + 0.1:
                    diffuse_layer_mask = (z_centers >= ohp_clipped) & (z_centers < diffuse_layer_end)
                else:
                    diffuse_layer_mask = np.zeros_like(z_centers, dtype=bool)
            
            # Integrate to get average number of ions in each region
            bin_volume = box_x * box_y * z_bin_width
            
            n_inner_sphere = np.sum(density[inner_sphere_mask]) * bin_volume
            n_outer_sphere = np.sum(density[outer_sphere_mask]) * bin_volume
            n_diffuse_layer = np.sum(density[diffuse_layer_mask]) * bin_volume
            
            total = n_inner_sphere + n_outer_sphere + n_diffuse_layer
            
            # Skip ions with negligible or zero counts
            if total < 0.01:  # Less than 0.01 ions means essentially none
                print(f"  {ion_name}: total = {total:.4f} ions (skipped, < 0.01)")
                continue
            
            if total > 0:
                frac_inner = n_inner_sphere / total
                frac_outer = n_outer_sphere / total
                frac_diffuse = n_diffuse_layer / total
            else:
                frac_inner = frac_outer = frac_diffuse = 0.0
            
            results[ion_name] = {
                'inner_sphere_count': n_inner_sphere,
                'outer_sphere_count': n_outer_sphere,
                'diffuse_layer_count': n_diffuse_layer,
                'total_count': total,
                'fraction_inner_sphere': frac_inner,
                'fraction_outer_sphere': frac_outer,
                'fraction_diffuse_layer': frac_diffuse
            }
            
            print(f"  {ion_name}:")
            if is_top_surface:
                print(f"    Inner-sphere ({ihp_clipped:.1f} < z < {surface_position:.1f} Å): {n_inner_sphere:.2f} ions ({frac_inner*100:.1f}%)")
                print(f"    Outer-sphere ({ohp_clipped:.1f} < z < {ihp_clipped:.1f} Å): {n_outer_sphere:.2f} ions ({frac_outer*100:.1f}%)")
                print(f"    Diffuse layer ({diffuse_layer_end:.1f} < z < {ohp_clipped:.1f} Å): {n_diffuse_layer:.2f} ions ({frac_diffuse*100:.1f}%)")
            else:
                print(f"    Inner-sphere ({surface_position:.1f} < z < {ihp_clipped:.1f} Å): {n_inner_sphere:.2f} ions ({frac_inner*100:.1f}%)")
                print(f"    Outer-sphere ({ihp_clipped:.1f} < z < {ohp_clipped:.1f} Å): {n_outer_sphere:.2f} ions ({frac_outer*100:.1f}%)")
                print(f"    Diffuse layer ({ohp_clipped:.1f} < z < {diffuse_layer_end:.1f} Å): {n_diffuse_layer:.2f} ions ({frac_diffuse*100:.1f}%)")
        
        return results
    
    def _calculate_capacitance_profile(self, z_centers, charge_density, potential,
                                       dz, surface_position, epsilon_r):
        """Calculate differential capacitance C(z) = dσ/dψ"""
        
        # Integrate charge to get surface charge as function of position
        surface_idx = np.argmin(np.abs(z_centers - surface_position))
        
        # Calculate cumulative charge from surface
        sigma_cumulative = np.zeros_like(charge_density)
        for i in range(surface_idx, len(z_centers)):
            sigma_cumulative[i] = np.trapz(charge_density[surface_idx:i+1], dx=dz)
        
        # Differential capacitance: C = dσ/dψ
        # Use numerical differentiation
        dpsi = np.gradient(potential, dz)
        dsigma = np.gradient(sigma_cumulative, dz)
        
        # Avoid division by zero
        capacitance = np.zeros_like(z_centers)
        mask = np.abs(dpsi) > 1e-10
        capacitance[mask] = dsigma[mask] / dpsi[mask]
        
        # Convert to SI units (F/m²)
        # Current: e/Å per (kT/e) = e² Å/kT
        # Need: F/m²
        k_B = 1.381e-23  # J/K
        e = 1.602e-19    # C
        T = 300.0  # K (from metadata if needed)
        
        # e²/(kT·Å) → C²/(J·m) = F/m
        # Then divide by Å to get F/m²
        capacitance_SI = capacitance * (e**2) / (k_B * T * 1e-10**2)
        
        # Calculate integral capacitance at specific distance (e.g., 10 Å from surface)
        distance_10A_idx = np.argmin(np.abs(z_centers - (surface_position + 10.0)))
        
        if distance_10A_idx > surface_idx:
            sigma_at_10A = sigma_cumulative[distance_10A_idx]
            psi_at_10A = potential[distance_10A_idx]
            
            if abs(psi_at_10A) > 1e-6:
                C_integral_10A = sigma_at_10A / psi_at_10A
                C_integral_10A_SI = C_integral_10A * (e**2) / (k_B * T * 1e-10**2)
            else:
                C_integral_10A_SI = 0.0
        else:
            C_integral_10A_SI = 0.0
        
        print(f"  Capacitance calculated:")
        print(f"    Peak differential capacitance: {np.max(np.abs(capacitance_SI)):.3e} F/m²")
        print(f"    Integral capacitance at 10 Å: {C_integral_10A_SI:.3e} F/m²")
        
        return {
            'capacitance_profile': capacitance_SI,
            'integral_capacitance_10A': C_integral_10A_SI,
            'surface_charge_cumulative': sigma_cumulative
        }
    
    def _compare_gouy_chapman_theory(self, z_centers, potential_MD, ion_densities,
                                     surface_position, sigma_surface, T, epsilon_r,
                                     charge_dict, bulk_distance):
        """Compare MD results with Gouy-Chapman theory"""
        
        # Get bulk concentrations
        bulk_mask = z_centers > (surface_position + bulk_distance)
        
        if not np.any(bulk_mask):
            bulk_mask = z_centers > (surface_position + 20.0)
        
        k_B = 1.381e-23  # J/K
        e = 1.602e-19    # C
        
        # Gouy-Chapman predictions for ion concentrations
        gc_ion_densities = {}
        
        print(f"  Gouy-Chapman theory comparison:")
        
        for ion_name, density_MD in ion_densities.items():
            charge = self._get_ion_charge(ion_name, charge_dict)
            
            if charge == 0:
                continue
            
            # Bulk concentration
            c_bulk = np.mean(density_MD[bulk_mask])
            
            # Gouy-Chapman: c(z) = c_bulk * exp(-z * e * ψ(z) / (k_B * T))
            # ψ in kT/e, so: c(z) = c_bulk * exp(-z * ψ(z))
            c_GC = c_bulk * np.exp(-charge * potential_MD)
            
            gc_ion_densities[ion_name] = c_GC
            
            # Calculate deviation
            deviation_mask = z_centers > surface_position
            if np.any(deviation_mask):
                rmsd = np.sqrt(np.mean((density_MD[deviation_mask] - c_GC[deviation_mask])**2))
                max_deviation = np.max(np.abs(density_MD[deviation_mask] - c_GC[deviation_mask]))
                
                # Only print if there's measurable density (skip if rmsd is nan or negligible)
                if not np.isnan(rmsd) and np.max(density_MD[deviation_mask]) > 1e-6:
                    print(f"    {ion_name}:")
                    print(f"      RMSD (MD vs GC): {rmsd:.6f} ions/Å³")
                    print(f"      Max deviation: {max_deviation:.6f} ions/Å³")
        
        return {
            'gouy_chapman_ion_densities': gc_ion_densities,
            'md_ion_densities': ion_densities
        }
    
    def create_comprehensive_summary(self):
        """
        Create a comprehensive summary of all analysis results.
        
        Returns:
        --------
        dict : Summary of all analysis results
        """
        summary = {
            'system_composition': {
                'n_clay_atoms': len(self.clay),
                'n_ions': {name: len(atoms) for name, atoms in self.ions.items()},
                'n_organics': {name: len(atoms) for name, atoms in self.organics.items()},
                'n_water_molecules': len(self.water) // 3,  # Assuming 3 atoms per water
                'n_frames': len(self.u.trajectory)
            },
            'analysis_completed': list(self.results.keys()),
            'key_findings': {}
        }
        
        # Extract key findings from each analysis
        if 'competitive_adsorption' in self.results:
            ca_data = self.results['competitive_adsorption']
            summary['key_findings']['most_adsorbed_ion'] = max(
                ca_data['ion_surface_contacts'].items(),
                key=lambda x: x[1]['mean']
            )[0] if ca_data['ion_surface_contacts'] else None
            
            summary['key_findings']['most_adsorbed_organic'] = max(
                ca_data['organic_surface_contacts'].items(),
                key=lambda x: x[1]['mean']
            )[0] if ca_data['organic_surface_contacts'] else None
        
        if 'three_component_bridges' in self.results:
            bridge_data = self.results['three_component_bridges']
            summary['key_findings']['dominant_bridge_type'] = max(
                bridge_data.items(),
                key=lambda x: x[1]['mean']
            )[0]
        
        if 'selectivity_coefficients' in self.results:
            sc_data = self.results['selectivity_coefficients']
            summary['key_findings']['highest_selectivity'] = max(
                sc_data.items(),
                key=lambda x: x[1]
            ) if sc_data else None
        
        return summary
    
    def run_full_analysis(self, save_results=True, output_file=None):
        """
        Run all analysis methods in sequence.
        
        Parameters:
        -----------
        save_results : bool
            Whether to save results to file
        output_file : str, optional
            Output file path for results
        
        Returns:
        --------
        dict : Complete analysis results
        """
        print("="*60)
        print("Running comprehensive clay-organic-ion-water analysis")
        print("="*60)
        
        # Run all analyses
        self.calculate_multi_component_rdfs()
        self.analyze_competitive_adsorption()
        self.analyze_organic_conformations()
        self.analyze_three_component_bridges()
        self.analyze_hydration_shell_competition()
        self.analyze_stratified_adsorption()
        self.analyze_exchange_kinetics()
        self.calculate_selectivity_coefficients()
        
        # Create summary
        summary = self.create_comprehensive_summary()
        
        print("\n" + "="*60)
        print("Analysis completed successfully!")
        print(f"Analyzed {summary['system_composition']['n_frames']} frames")
        print(f"Completed analyses: {', '.join(summary['analysis_completed'])}")
        print("="*60)
        
        if save_results:
            if output_file is None:
                output_file = 'clay_organic_ion_water_analysis_results.npz'
            
            # Save results
            np.savez_compressed(output_file, 
                              results=self.results,
                              rdf_results=self.rdf_results,
                              summary=summary)
            print(f"Results saved to: {output_file}")
        
        return {
            'results': self.results,
            'rdf_results': self.rdf_results,
            'summary': summary
        }
    
    def _detect_si_hexagonal_rings(self, si_positions, si_si_threshold=4.5, 
                                   min_ring_size=6, max_ring_size=6):
        """
        Detect Si hexagonal rings (cavities) in clay structure using graph-based approach.
        
        Parameters
        ----------
        si_positions : np.ndarray, shape (N, 3)
            XYZ coordinates of Si atoms
        si_si_threshold : float, default=4.5
            Maximum Si-Si distance (Å) to consider atoms as neighbors
        min_ring_size : int, default=6
            Minimum ring size to detect
        max_ring_size : int, default=6
            Maximum ring size to detect
        
        Returns
        -------
        dict
            - 'ring_centers': np.ndarray, shape (M, 3) - XYZ coordinates of ring centers
            - 'ring_indices': list of lists - Si atom indices forming each ring
            - 'ring_radii': np.ndarray, shape (M,) - Average radius of each ring
        """
        # Build neighbor graph based on distance
        n_si = len(si_positions)
        distances = cdist(si_positions, si_positions)
        
        # Create graph with Si atoms as nodes
        G = nx.Graph()
        for i in range(n_si):
            G.add_node(i, pos=si_positions[i])
        
        # Add edges for Si atoms within threshold distance
        for i in range(n_si):
            for j in range(i+1, n_si):
                if distances[i, j] < si_si_threshold:
                    G.add_edge(i, j)
        
        # Find all simple cycles (rings) of specified size
        all_rings = []
        try:
            # Find cycles between min and max size
            for cycle in nx.simple_cycles(G.to_directed(), length_bound=max_ring_size):
                if min_ring_size <= len(cycle) <= max_ring_size:
                    # Store canonical form (smallest index first, clockwise)
                    cycle_sorted = tuple(sorted(cycle))
                    if cycle_sorted not in [tuple(sorted(r)) for r in all_rings]:
                        all_rings.append(list(cycle))
        except:
            # Fallback: find cycles using cycle_basis (may miss some rings)
            cycles = nx.cycle_basis(G)
            for cycle in cycles:
                if min_ring_size <= len(cycle) <= max_ring_size:
                    all_rings.append(cycle)
        
        if len(all_rings) == 0:
            print("WARNING: No Si hexagonal rings detected. Check si_si_threshold parameter.")
            return {
                'ring_centers': np.array([]),
                'ring_indices': [],
                'ring_radii': np.array([])
            }
        
        # Calculate ring centers and radii
        ring_centers = []
        ring_radii = []
        
        for ring_indices in all_rings:
            # Get positions of Si atoms in this ring
            ring_positions = si_positions[ring_indices]
            
            # Calculate center as mean position
            center = np.mean(ring_positions, axis=0)
            ring_centers.append(center)
            
            # Calculate average radius (distance from center to ring atoms)
            radii = np.linalg.norm(ring_positions - center, axis=1)
            avg_radius = np.mean(radii)
            ring_radii.append(avg_radius)
        
        return {
            'ring_centers': np.array(ring_centers),
            'ring_indices': all_rings,
            'ring_radii': np.array(ring_radii)
        }
    
    def _print_cavity_report(self, cavity_data, z_slice_centers, 
                            si_si_threshold, cavity_radius, cavity_height,
                            center_box, z_offset):
        """Print detailed cavity analysis report"""
        print("\n" + "="*80)
        print("CAVITY ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\nAnalysis Parameters:")
        print(f"  Si-Si threshold: {si_si_threshold} Å")
        print(f"  Cavity radius (for ion counting): {cavity_radius} Å")
        print(f"  Cavity height: {cavity_height} Å")
        print(f"  Center box: {center_box}")
        if center_box:
            print(f"  Z-offset: {z_offset:.2f} Å")
        
        print(f"\n{'='*80}")
        print("DETECTED CAVITIES BY Z-SLICE")
        print(f"{'='*80}")
        
        for z_center in z_slice_centers:
            if z_center not in cavity_data:
                print(f"\n❌ Z = {z_center:.2f} Å: No Si atoms found in this slice")
                continue
            
            cavity_info = cavity_data[z_center]
            ring_centers = cavity_info['ring_centers']
            ring_radii = cavity_info['ring_radii']
            ring_indices = cavity_info['ring_indices']
            
            n_cavities = len(ring_centers)
            
            print(f"\n{'─'*80}")
            print(f"Z-slice: {z_center:.2f} Å")
            print(f"Number of cavities: {n_cavities}")
            print(f"{'─'*80}")
            
            if n_cavities == 0:
                print("  No hexagonal cavities detected")
                continue
            
            # Print statistics
            print(f"\nCavity radius statistics:")
            print(f"  Mean: {np.mean(ring_radii):.3f} Å")
            print(f"  Std:  {np.std(ring_radii):.3f} Å")
            print(f"  Min:  {np.min(ring_radii):.3f} Å")
            print(f"  Max:  {np.max(ring_radii):.3f} Å")
            
            # Print individual cavities
            print(f"\nIndividual cavity details:")
            print(f"  {'Cavity':<8} {'Center (x, y, z)':<30} {'Radius':<10} {'Si atoms'}")
            print(f"  {'-'*70}")
            
            for i, (center, radius, indices) in enumerate(zip(ring_centers, ring_radii, ring_indices)):
                center_str = f"({center[0]:6.2f}, {center[1]:6.2f}, {center[2]:6.2f})"
                indices_str = f"{len(indices)} atoms"
                print(f"  {i+1:<8} {center_str:<30} {radius:6.3f} Å   {indices_str}")
            
            # Show first cavity atom indices as example
            if n_cavities > 0:
                print(f"\n  Example (Cavity 1 Si atom indices): {ring_indices[0]}")
        
        print(f"\n{'='*80}")
    
    def analyze_cavity_ion_binding(self, ion_types=None, z_slice_centers=None,
                                   z_slice_width=2.0, cavity_radius=3.0, 
                                   cavity_height=6.0, si_si_threshold=4.5,
                                   compute_per_cavity_timeseries=True,
                                   compute_avg_occupancy=True,
                                   compute_preferential_sites=True,
                                   compute_spatial_correlation=True,
                                   compute_xy_spatial=False,
                                   xy_grid_size=0.5,
                                   use_weighted_cavity_interpolation=False,
                                   step=1, save_cache=True, force_rerun=False,
                                   cache_file=None, verbose=True,
                                   center_box=False, show_cavity_report=False):
        """
        Analyze ion binding to Si hexagonal ring cavities in clay surface.
        
        This method:
        1. Detects Si hexagonal rings (cavities) in each z-slice
        2. Counts ions within cylindrical regions above each cavity
        3. Calculates occupancy statistics over time
        4. Identifies preferential binding sites
        
        Parameters
        ----------
        ion_types : list of str, optional
            Ion names to analyze (e.g., ['NA', 'MG']). If None, uses all ions.
        z_slice_centers : list of float, optional
            Z-positions to analyze (Å). If None, auto-detects clay layers.
            Note: If center_box=True, use centered coordinates (e.g., [-20, 0, 20])
        z_slice_width : float, default=2.0
            Width of z-slice for Si atom selection (Å)
        cavity_radius : float, default=3.0
            Radius of cylindrical region extending from cavity center (Å)
        cavity_height : float, default=6.0
            Height of cylindrical region extending toward box center (Å)
            For upper surface (z>0): extends downward; for lower surface (z<0): extends upward
        si_si_threshold : float, default=4.5
            Maximum Si-Si distance for ring detection (Å)
        compute_per_cavity_timeseries : bool, default=True
            Calculate ion count timeseries for each cavity
        compute_avg_occupancy : bool, default=True
            Calculate average occupancy per cavity
        compute_preferential_sites : bool, default=True
            Identify cavities with highest occupancy
        compute_spatial_correlation : bool, default=True
            Analyze correlation between cavity position and ion density
        compute_xy_spatial : bool, default=False
            Compute 2D spatial distribution in XY plane showing:
            - Ion density grid (ions per Å²)
            - Cavity occupancy grid (mapped to XY space)
            Creates heatmaps for visualizing spatial binding patterns
        xy_grid_size : float, default=0.5
            Grid spacing for XY spatial analysis (Å)
            Only used if compute_xy_spatial=True
        use_weighted_cavity_interpolation : bool, default=True
            Method for creating cavity occupancy grid:
            - True: Distance-weighted interpolation from all cavities (smooth gradients,
              occupancy decreases with distance from cavity centers)
            - False: Nearest-neighbor assignment (each grid point gets occupancy of
              nearest cavity, creates discrete regions)
        step : int, default=1
            Frame step size
        save_cache : bool, default=True
            Save results to cache file for faster subsequent analysis
        force_rerun : bool, default=False
            Force recalculation even if cache exists
        cache_file : str, optional
            Custom cache filename. If None, auto-generates from parameters.
        verbose : bool, default=True
            Print progress messages
        center_box : bool, default=False
            If True, center the simulation box so z=0 is at box center.
            This matches the centering used in EDL analysis and allows
            consistent coordinate systems across different methods.
        show_cavity_report : bool, default=False
            If True, print detailed cavity report with individual cavity
            positions and radii. Always shows summary statistics.
        
        Returns
        -------
        dict
            Results stored in self.results['cavity_ion_binding']:
            - 'z_slice_centers': np.ndarray - Z-positions analyzed
            - 'cavity_data': dict - Per z-slice cavity information
                - z_center: dict with 'ring_centers', 'ring_indices', 'ring_radii'
            - 'ion_data': dict - Per ion type binding data
                - ion_type: dict with:
                    - z_center: dict with:
                        - 'per_cavity_timeseries': np.ndarray (n_cavities, n_frames)
                        - 'avg_occupancy': np.ndarray (n_cavities,)
                        - 'std_occupancy': np.ndarray (n_cavities,)
                        - 'max_occupancy': np.ndarray (n_cavities,)
                        - 'occupancy_fraction': np.ndarray (n_cavities,)
            - 'preferential_sites': dict - Most occupied cavities per ion per z-slice
            - 'metadata': dict - Analysis parameters
        """
        if verbose:
            print("\n" + "="*60)
            print("Analyzing Cavity-Specific Ion Binding")
            print("="*60)
        
        # Validate ion types
        if ion_types is None:
            ion_types = list(self.ions.keys())
        elif isinstance(ion_types, str):
            ion_types = [ion_types]
        
        # Calculate z-offset for centering
        box_z = self.u.dimensions[2]
        if center_box:
            z_offset = box_z / 2
            if verbose:
                print(f"  Box centering: Enabled (z=0 at center, z_offset = {z_offset:.2f} Å)")
        else:
            z_offset = 0.0
            if verbose:
                print(f"  Box centering: Disabled (using original coordinates)")
        
        # Auto-detect z-slices if not provided
        if z_slice_centers is None:
            # Get Si z-positions from first frame
            self.u.trajectory[0]
            si_atoms = self.clay.select_atoms('name Si or name SI')
            si_z_raw = si_atoms.positions[:, 2]
            # Apply centering offset
            si_z = si_z_raw - z_offset
            
            # Find peaks in Si z-distribution (clay layers)
            hist, edges = np.histogram(si_z, bins=50)
            bin_centers = (edges[:-1] + edges[1:]) / 2
            
            # Simple peak detection: find local maxima
            peaks = []
            for i in range(1, len(hist)-1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist)*0.2:
                    peaks.append(bin_centers[i])
            
            z_slice_centers = sorted(peaks)
            if verbose:
                print(f"Auto-detected {len(z_slice_centers)} clay layers at z = {z_slice_centers}")
        
        # Cache management
        if save_cache or cache_file:
            import hashlib
            import os
            
            # Generate cache filename if not provided
            if cache_file is None:
                # Create cache directory
                cache_dir = '.cavity_cache'
                os.makedirs(cache_dir, exist_ok=True)
                
                # Create a hash of key parameters
                param_str = f"{ion_types}_{z_slice_centers}_{z_slice_width}_{cavity_radius}_{cavity_height}_{si_si_threshold}_{center_box}_{step}_{compute_xy_spatial}_{xy_grid_size}"
                cache_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
                cache_file = os.path.join(cache_dir, f'cavity_analysis_{cache_hash}.npz')
            
            # Try to load from cache
            if not force_rerun and os.path.exists(cache_file):
                if verbose:
                    print(f"\n  Loading cached results from: {cache_file}")
                
                try:
                    cached = np.load(cache_file, allow_pickle=True)
                    
                    # Reconstruct cavity_data (use float keys to match the analysis code)
                    cavity_data = {}
                    for z_slice in cached['z_slice_centers']:
                        z_key = f"z_{z_slice:.1f}"
                        cavity_data[z_slice] = {  # Use float key, not string
                            'ring_centers': cached[f'{z_key}_ring_centers'],
                            'ring_indices': cached[f'{z_key}_ring_indices'],
                            'ring_radii': cached[f'{z_key}_ring_radii']
                        }
                    
                    # Reconstruct ion_data (use float keys to match the analysis code)
                    ion_data = {}
                    for ion_type in ion_types:
                        ion_data[ion_type] = {}
                        for z_slice in cached['z_slice_centers']:
                            z_key = f"z_{z_slice:.1f}"
                            ion_data[ion_type][z_slice] = {}  # Use float key, not string
                            
                            # Load all available keys for this ion/z-slice combination
                            if f'{ion_type}_{z_key}_per_cavity_timeseries' in cached:
                                ion_data[ion_type][z_slice]['per_cavity_timeseries'] = cached[f'{ion_type}_{z_key}_per_cavity_timeseries']
                            if f'{ion_type}_{z_key}_avg_occupancy' in cached:
                                ion_data[ion_type][z_slice]['avg_occupancy'] = cached[f'{ion_type}_{z_key}_avg_occupancy']
                            if f'{ion_type}_{z_key}_std_occupancy' in cached:
                                ion_data[ion_type][z_slice]['std_occupancy'] = cached[f'{ion_type}_{z_key}_std_occupancy']
                            if f'{ion_type}_{z_key}_max_occupancy' in cached:
                                ion_data[ion_type][z_slice]['max_occupancy'] = cached[f'{ion_type}_{z_key}_max_occupancy']
                            if f'{ion_type}_{z_key}_occupancy_fraction' in cached:
                                ion_data[ion_type][z_slice]['occupancy_fraction'] = cached[f'{ion_type}_{z_key}_occupancy_fraction']
                            
                            # Load xy_spatial data if present
                            if f'{ion_type}_{z_key}_xy_ion_density' in cached:
                                ion_data[ion_type][z_slice]['xy_spatial'] = {
                                    'ion_density_grid': cached[f'{ion_type}_{z_key}_xy_ion_density'],
                                    'cavity_occupancy_grid': cached[f'{ion_type}_{z_key}_xy_cavity_occupancy'],
                                    'x_centers': cached[f'{ion_type}_{z_key}_xy_x_centers'],
                                    'y_centers': cached[f'{ion_type}_{z_key}_xy_y_centers'],
                                    'x_edges': cached[f'{ion_type}_{z_key}_xy_x_edges'],
                                    'y_edges': cached[f'{ion_type}_{z_key}_xy_y_edges'],
                                    'grid_dimensions': tuple(cached[f'{ion_type}_{z_key}_xy_grid_dims']),
                                    'box_dimensions': tuple(cached[f'{ion_type}_{z_key}_xy_box_dims']),
                                    'surface_type': str(cached[f'{ion_type}_{z_key}_xy_surface_type']),
                                    'cavity_centers_xy': cached[f'{ion_type}_{z_key}_xy_cavity_centers'],
                                    'n_frames': int(cached[f'{ion_type}_{z_key}_xy_n_frames'])
                                }
                    
                    # Reconstruct preferential_sites (safe handling for object arrays)
                    if 'preferential_sites' in cached:
                        pref_sites = cached['preferential_sites']
                        if isinstance(pref_sites, np.ndarray) and pref_sites.dtype == object:
                            preferential_sites = pref_sites.item() if pref_sites.size == 1 else pref_sites
                        else:
                            preferential_sites = pref_sites
                    else:
                        preferential_sites = {}
                    
                    # Construct result
                    result = {
                        'z_slice_centers': z_slice_centers,
                        'cavity_data': cavity_data,
                        'ion_data': ion_data,
                        'preferential_sites': preferential_sites,
                        'metadata': {
                            'ion_types': ion_types,
                            'z_slice_centers': z_slice_centers,
                            'z_slice_width': z_slice_width,
                            'cavity_radius': cavity_radius,
                            'cavity_height': cavity_height,
                            'si_si_threshold': si_si_threshold,
                            'center_box': center_box,
                            'n_frames': len(self.u.trajectory)
                        }
                    }
                    
                    # Store in self.results before returning
                    self.results['cavity_ion_binding'] = result
                    
                    if verbose:
                        print("  ✓ Successfully loaded from cache")
                    
                    return result
                    
                except Exception as e:
                    if verbose:
                        print(f"  ⚠ Failed to load cache: {e}")
                        print("  Proceeding with full analysis...")
        
        # Initialize results
        cavity_data = {}
        ion_data = {ion_type: {} for ion_type in ion_types}
        preferential_sites = {ion_type: {} for ion_type in ion_types}
        
        # Detect cavities in each z-slice
        if verbose:
            print(f"\n📍 Detecting Si hexagonal ring cavities...")
            print(f"   Si-Si threshold: {si_si_threshold} Å")
            print(f"   Z-slices: {len(z_slice_centers)}")
        
        self.u.trajectory[0]
        si_atoms = self.clay.select_atoms('name Si or name SI')
        
        for z_center in z_slice_centers:
            # Select Si atoms in this z-slice (apply z-offset to raw positions)
            si_z_centered = si_atoms.positions[:, 2] - z_offset
            z_mask = np.abs(si_z_centered - z_center) <= z_slice_width / 2
            si_positions_raw = si_atoms.positions[z_mask]
            
            # Apply z-offset to positions for centered coordinates
            si_positions = si_positions_raw.copy()
            si_positions[:, 2] -= z_offset
            
            if len(si_positions) < 6:
                if verbose:
                    print(f"   WARNING: Only {len(si_positions)} Si atoms in slice z={z_center:.1f}, skipping.")
                continue
            
            # Detect hexagonal rings
            ring_data = self._detect_si_hexagonal_rings(
                si_positions, 
                si_si_threshold=si_si_threshold,
                min_ring_size=6,
                max_ring_size=6
            )
            
            cavity_data[z_center] = ring_data
            
            if verbose:
                n_rings = len(ring_data['ring_centers'])
                print(f"   z = {z_center:6.1f} Å: Found {n_rings} hexagonal ring cavities")
                
                # Print cavity radius statistics
                if n_rings > 0:
                    ring_radii = ring_data['ring_radii']
                    print(f"      Cavity radius: Mean={np.mean(ring_radii):.3f} Å, "
                          f"Std={np.std(ring_radii):.3f} Å, "
                          f"Min={np.min(ring_radii):.3f} Å, "
                          f"Max={np.max(ring_radii):.3f} Å")
        
        # Print detailed cavity report if requested
        if show_cavity_report and len(cavity_data) > 0:
            self._print_cavity_report(cavity_data, z_slice_centers, 
                                     si_si_threshold, cavity_radius, 
                                     cavity_height, center_box, z_offset)
        
        # Analyze ion binding to each cavity over trajectory
        if verbose:
            print(f"\n🔬 Analyzing ion binding to cavities...")
            print(f"   Cavity radius: {cavity_radius} Å")
            print(f"   Cavity height: {cavity_height} Å")
            print(f"   Ion types: {', '.join(ion_types)}")
        
        n_frames = len(self.u.trajectory[::step])
        
        for ion_type in ion_types:
            if verbose:
                print(f"\n   Processing {ion_type}...")
            
            ion_atoms = self.ions[ion_type]
            
            for z_center in z_slice_centers:
                if z_center not in cavity_data:
                    continue
                
                ring_centers = cavity_data[z_center]['ring_centers']
                n_cavities = len(ring_centers)
                
                if n_cavities == 0:
                    continue
                
                # Initialize storage arrays
                if compute_per_cavity_timeseries:
                    per_cavity_ts = np.zeros((n_cavities, n_frames))
                
                # Analyze each frame
                for frame_idx, ts in enumerate(self.u.trajectory[::step]):
                    ion_positions_raw = ion_atoms.positions
                    # Apply z-offset for centered coordinates
                    ion_positions = ion_positions_raw.copy()
                    ion_positions[:, 2] -= z_offset
                    
                    # Count ions in each cavity
                    for cavity_idx, cavity_center in enumerate(ring_centers):
                        # Define cylindrical region extending toward box center (z=0)
                        # Check XY distance (radius)
                        xy_distance = np.linalg.norm(
                            ion_positions[:, :2] - cavity_center[:2], axis=1
                        )
                        
                        # Check Z distance - cylinder extends toward z=0 (box center)
                        # For upper surface (z>0): ions should be below the cavity (negative z_distance)
                        # For lower surface (z<0): ions should be above the cavity (positive z_distance)
                        z_distance = ion_positions[:, 2] - cavity_center[2]
                        
                        # Determine cylinder direction based on surface position
                        if cavity_center[2] > 0:  # Upper surface
                            # Cylinder extends downward (toward z=0)
                            in_cavity_mask = (xy_distance <= cavity_radius) & \
                                            (z_distance <= 0) & \
                                            (z_distance >= -cavity_height)
                        else:  # Lower surface
                            # Cylinder extends upward (toward z=0)
                            in_cavity_mask = (xy_distance <= cavity_radius) & \
                                            (z_distance >= 0) & \
                                            (z_distance <= cavity_height)
                        
                        n_ions_in_cavity = np.sum(in_cavity_mask)
                        
                        if compute_per_cavity_timeseries:
                            per_cavity_ts[cavity_idx, frame_idx] = n_ions_in_cavity
                
                # Calculate statistics
                ion_data[ion_type][z_center] = {}
                
                if compute_per_cavity_timeseries:
                    ion_data[ion_type][z_center]['per_cavity_timeseries'] = per_cavity_ts
                
                if compute_avg_occupancy:
                    ion_data[ion_type][z_center]['avg_occupancy'] = np.mean(per_cavity_ts, axis=1)
                    ion_data[ion_type][z_center]['std_occupancy'] = np.std(per_cavity_ts, axis=1)
                    ion_data[ion_type][z_center]['max_occupancy'] = np.max(per_cavity_ts, axis=1)
                    
                    # Occupancy fraction: fraction of frames where cavity is occupied
                    ion_data[ion_type][z_center]['occupancy_fraction'] = \
                        np.sum(per_cavity_ts > 0, axis=1) / n_frames
                
                if compute_preferential_sites:
                    # Find cavities with highest average occupancy
                    avg_occ = ion_data[ion_type][z_center]['avg_occupancy']
                    top_indices = np.argsort(avg_occ)[::-1][:5]  # Top 5 cavities
                    
                    preferential_sites[ion_type][z_center] = {
                        'cavity_indices': top_indices,
                        'cavity_positions': ring_centers[top_indices],
                        'avg_occupancy': avg_occ[top_indices]
                    }
                
                if verbose:
                    avg_total = np.mean(np.sum(per_cavity_ts, axis=0))
                    print(f"      z = {z_center:6.1f} Å: {n_cavities} cavities, "
                          f"avg {avg_total:.2f} ions bound to cavities")
        
        # Compute XY spatial distribution if requested
        if compute_xy_spatial:
            if verbose:
                print(f"\n📍 Computing XY spatial distributions...")
                print(f"   Grid size: {xy_grid_size} Å")
            
            # Get box dimensions
            box_x = self.u.dimensions[0]
            box_y = self.u.dimensions[1]
            
            # Create XY grid
            n_x = int(np.ceil(box_x / xy_grid_size))
            n_y = int(np.ceil(box_y / xy_grid_size))
            x_edges = np.linspace(0, box_x, n_x + 1)
            y_edges = np.linspace(0, box_y, n_y + 1)
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            X_grid, Y_grid = np.meshgrid(x_centers, y_centers)
            
            if verbose:
                print(f"   Grid dimensions: {n_x} × {n_y} bins")
                print(f"   Box size: {box_x:.1f} × {box_y:.1f} Å")
            
            # Process each z-slice
            for z_center in z_slice_centers:
                if z_center not in cavity_data:
                    continue
                
                ring_centers = cavity_data[z_center]['ring_centers']
                n_cavities = len(ring_centers)
                
                if n_cavities == 0:
                    continue
                
                # Determine surface type based on z position
                surface_type = 'top' if z_center > 0 else 'bottom'
                
                # Process each ion type
                for ion_type in ion_types:
                    if ion_type not in self.ions or z_center not in ion_data[ion_type]:
                        continue
                    
                    # Initialize grids
                    ion_density_grid = np.zeros((n_y, n_x))
                    cavity_occupancy_grid = np.zeros((n_y, n_x))
                    
                    # Get ion group
                    ion_group = self.ions[ion_type]
                    
                    # Calculate ion density grid from trajectory
                    # Use the same z-range as cavity ion binding: cylindrical region toward box center
                    # Upper surface (z>0): extend downward; Lower surface (z<0): extend upward
                    if z_center > 0:  # Upper surface
                        z_min_slice = z_center - cavity_height
                        z_max_slice = z_center
                    else:  # Lower surface
                        z_min_slice = z_center
                        z_max_slice = z_center + cavity_height
                    
                    for frame_idx, ts in enumerate(self.u.trajectory[::step]):
                        ion_positions_raw = ion_group.positions
                        ion_positions = ion_positions_raw.copy()
                        ion_positions[:, 2] -= z_offset
                        
                        # Select ions in z-slice
                        z_positions = ion_positions[:, 2]
                        in_slice = (z_positions >= z_min_slice) & (z_positions < z_max_slice)
                        
                        if np.any(in_slice):
                            ions_in_slice = ion_positions[in_slice]
                            
                            # Bin ion XY positions
                            for ion_pos in ions_in_slice:
                                x, y = ion_pos[0] % box_x, ion_pos[1] % box_y
                                x_idx = int(x / xy_grid_size)
                                y_idx = int(y / xy_grid_size)
                                
                                if 0 <= x_idx < n_x and 0 <= y_idx < n_y:
                                    ion_density_grid[y_idx, x_idx] += 1
                    
                    # Get frame count
                    frame_count = len(self.u.trajectory[::step])
                    
                    # Normalize ion density by number of frames and grid area
                    grid_area = xy_grid_size ** 2
                    ion_density_grid = ion_density_grid / (frame_count * grid_area)
                    
                    # Create BOTH cavity occupancy grids for comparison
                    # Always compute both methods so user can choose which to plot
                    cavity_occupancy_grid_weighted = np.zeros((n_y, n_x))
                    cavity_occupancy_grid_nearest = np.zeros((n_y, n_x))
                    
                    if 'avg_occupancy' in ion_data[ion_type][z_center]:
                        avg_occupancy = ion_data[ion_type][z_center]['avg_occupancy']
                        
                        # COMPUTE WEIGHTED INTERPOLATION
                        for i in range(n_y):
                            for j in range(n_x):
                                grid_x = x_centers[j]
                                grid_y = y_centers[i]
                                
                                # Calculate distance to all cavities (with PBC)
                                dx = ring_centers[:, 0] - grid_x
                                dy = ring_centers[:, 1] - grid_y
                                
                                # Apply periodic boundary conditions
                                dx = dx - box_x * np.round(dx / box_x)
                                dy = dy - box_y * np.round(dy / box_y)
                                
                                distances = np.sqrt(dx**2 + dy**2)
                                
                                # Weight by inverse distance squared (with small constant to avoid division by zero)
                                epsilon = 0.1  # Small constant (Å)
                                weights = 1.0 / (distances**2 + epsilon**2)
                                
                                # Weighted average of cavity occupancies
                                cavity_occupancy_grid_weighted[i, j] = np.sum(weights * avg_occupancy) / np.sum(weights)
                                
                                # Also compute nearest-neighbor for comparison
                                nearest_cavity = np.argmin(distances)
                                cavity_occupancy_grid_nearest[i, j] = avg_occupancy[nearest_cavity]
                    
                    # Store XY spatial results with BOTH cavity occupancy methods
                    ion_data[ion_type][z_center]['xy_spatial'] = {
                        'ion_density_grid': ion_density_grid,
                        'cavity_occupancy_grid_weighted': cavity_occupancy_grid_weighted,
                        'cavity_occupancy_grid_nearest': cavity_occupancy_grid_nearest,
                        # Backward compatibility: default to weighted
                        'cavity_occupancy_grid': cavity_occupancy_grid_weighted,
                        'x_centers': x_centers,
                        'y_centers': y_centers,
                        'x_edges': x_edges,
                        'y_edges': y_edges,
                        'grid_dimensions': (n_x, n_y),
                        'box_dimensions': (box_x, box_y),
                        'surface_type': surface_type,
                        'cavity_centers_xy': ring_centers[:, :2],
                        'n_frames': frame_count
                    }
                
                if verbose:
                    print(f"      z = {z_center:6.1f} Å ({surface_type} surface): XY grids computed")
        
        # Store results
        self.results['cavity_ion_binding'] = {
            'z_slice_centers': np.array(z_slice_centers),
            'cavity_data': cavity_data,
            'ion_data': ion_data,
            'preferential_sites': preferential_sites,
            'metadata': {
                'ion_types': ion_types,
                'z_slice_width': z_slice_width,
                'cavity_radius': cavity_radius,
                'cavity_height': cavity_height,
                'si_si_threshold': si_si_threshold,
                'n_frames': n_frames,
                'step': step,
                'center_box': center_box,
                'z_offset': z_offset
            }
        }
        
        if verbose:
            print("\n" + "="*60)
            print("✅ Cavity ion binding analysis completed!")
            print("="*60)
        
        # Save to cache if requested
        if save_cache and cache_file:
            try:
                if verbose:
                    print(f"\n  Saving results to cache: {cache_file}")
                
                # Prepare data for saving
                save_dict = {
                    'z_slice_centers': z_slice_centers,
                    'ion_types': ion_types
                }
                
                # Save cavity_data (use float keys from cavity_data dict)
                for z_slice in z_slice_centers:
                    z_key = f"z_{z_slice:.1f}"
                    save_dict[f'{z_key}_ring_centers'] = cavity_data[z_slice]['ring_centers']
                    save_dict[f'{z_key}_ring_indices'] = cavity_data[z_slice]['ring_indices']
                    save_dict[f'{z_key}_ring_radii'] = cavity_data[z_slice]['ring_radii']
                
                # Save ion_data (use float keys from ion_data dict)
                for ion_type in ion_types:
                    for z_slice in z_slice_centers:
                        z_key = f"z_{z_slice:.1f}"
                        ion_slice_data = ion_data[ion_type][z_slice]
                        
                        # Save all available keys for this ion/z-slice combination
                        if 'per_cavity_timeseries' in ion_slice_data:
                            save_dict[f'{ion_type}_{z_key}_per_cavity_timeseries'] = ion_slice_data['per_cavity_timeseries']
                        if 'avg_occupancy' in ion_slice_data:
                            save_dict[f'{ion_type}_{z_key}_avg_occupancy'] = ion_slice_data['avg_occupancy']
                        if 'std_occupancy' in ion_slice_data:
                            save_dict[f'{ion_type}_{z_key}_std_occupancy'] = ion_slice_data['std_occupancy']
                        if 'max_occupancy' in ion_slice_data:
                            save_dict[f'{ion_type}_{z_key}_max_occupancy'] = ion_slice_data['max_occupancy']
                        if 'occupancy_fraction' in ion_slice_data:
                            save_dict[f'{ion_type}_{z_key}_occupancy_fraction'] = ion_slice_data['occupancy_fraction']
                        
                        # Save xy_spatial data if present
                        if 'xy_spatial' in ion_slice_data:
                            xy_data = ion_slice_data['xy_spatial']
                            save_dict[f'{ion_type}_{z_key}_xy_ion_density'] = xy_data['ion_density_grid']
                            save_dict[f'{ion_type}_{z_key}_xy_cavity_occupancy'] = xy_data['cavity_occupancy_grid']
                            save_dict[f'{ion_type}_{z_key}_xy_x_centers'] = xy_data['x_centers']
                            save_dict[f'{ion_type}_{z_key}_xy_y_centers'] = xy_data['y_centers']
                            save_dict[f'{ion_type}_{z_key}_xy_x_edges'] = xy_data['x_edges']
                            save_dict[f'{ion_type}_{z_key}_xy_y_edges'] = xy_data['y_edges']
                            save_dict[f'{ion_type}_{z_key}_xy_grid_dims'] = xy_data['grid_dimensions']
                            save_dict[f'{ion_type}_{z_key}_xy_box_dims'] = xy_data['box_dimensions']
                            save_dict[f'{ion_type}_{z_key}_xy_surface_type'] = xy_data['surface_type']
                            save_dict[f'{ion_type}_{z_key}_xy_cavity_centers'] = xy_data['cavity_centers_xy']
                            save_dict[f'{ion_type}_{z_key}_xy_n_frames'] = xy_data['n_frames']
                
                # Save preferential_sites
                save_dict['preferential_sites'] = preferential_sites
                
                # Save with compression
                np.savez_compressed(cache_file, **save_dict)
                
                if verbose:
                    print("  ✓ Cache saved successfully")
                    
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Failed to save cache: {e}")
        
        return self.results['cavity_ion_binding']
    
    def analyze_cavity_organic_binding(self, functional_groups=None, z_slice_centers=None,
                                       z_slice_width=2.0, cavity_radius=3.0, 
                                       cavity_height=6.0, si_si_threshold=4.5,
                                       compute_occupancy_lifetime=True,
                                       compute_avg_occupancy=True,
                                       compute_preferential_sites=True,
                                       compute_spatial_correlation=True,
                                       compute_xy_spatial=False,
                                       compute_orientation=True,
                                       xy_grid_size=0.5,
                                       step=1, save_cache=True, force_rerun=False,
                                       cache_file=None, verbose=True,
                                       center_box=False, show_cavity_report=False):
        """
        Analyze organic functional group binding to Si hexagonal ring cavities in clay surface.
        
        Similar to analyze_cavity_ion_binding() but treats each functional group independently,
        allowing analysis of which parts of CIP (or other organic molecules) preferentially
        bind to clay surface cavities.
        
        This method:
        1. Detects Si hexagonal rings (cavities) in each z-slice
        2. Counts atoms from each functional group within cylindrical regions above each cavity
        3. Calculates occupancy statistics over time
        4. Identifies preferential binding sites for each functional group
        5. Optionally analyzes orientation (which functional group faces the surface)
        
        Parameters
        ----------
        functional_groups : list of str or None, optional
            Names of functional groups to analyze (must match keys in self.custom_selections['CIP_parts']).
            Examples: ['quinolone', 'piperazine', 'carboxylic_acid', 'cyclopropyl', 'O_ketone', 'fluoride']
            If None, analyzes all available functional groups from custom_selections['CIP_parts'].
        z_slice_centers : list of float, optional
            Z-positions to analyze (Å). If None, auto-detects clay layers.
            Note: If center_box=True, use centered coordinates (e.g., [-20, 0, 20])
        z_slice_width : float, default=2.0
            Width of z-slice for Si atom selection (Å)
        cavity_radius : float, default=3.0
            Radius of cylindrical region extending from cavity center (Å)
        cavity_height : float, default=6.0
            Height of cylindrical region extending toward box center (Å)
            For upper surface (z>0): extends downward; for lower surface (z<0): extends upward
        si_si_threshold : float, default=4.5
            Maximum Si-Si distance for ring detection (Å)
        compute_occupancy_lifetime : bool, default=True
            Calculate functional group count timeseries for each cavity
        compute_avg_occupancy : bool, default=True
            Calculate average occupancy per cavity
        compute_preferential_sites : bool, default=True
            Identify cavities with highest occupancy for each functional group
        compute_spatial_correlation : bool, default=True
            Analyze correlation between cavity position and functional group density
        compute_xy_spatial : bool, default=False
            Compute 2D spatial distribution in XY plane showing:
            - Functional group density grid (atoms per Å²)
            - Cavity occupancy grid (mapped to XY space)
            Creates heatmaps for visualizing spatial binding patterns
        compute_orientation : bool, default=True
            Analyze which functional group is closest to cavities (orientation analysis)
            Helps determine molecular orientation at the surface
        xy_grid_size : float, default=0.5
            Grid spacing for XY spatial analysis (Å)
            Only used if compute_xy_spatial=True
        step : int, default=1
            Frame step size
        save_cache : bool, default=True
            Save results to cache file for faster subsequent analysis
        force_rerun : bool, default=False
            Force recalculation even if cache exists
        cache_file : str, optional
            Custom cache filename. If None, auto-generates from parameters.
        verbose : bool, default=True
            Print progress messages
        center_box : bool, default=False
            If True, center the simulation box so z=0 is at box center.
            This matches the centering used in EDL analysis.
        show_cavity_report : bool, default=False
            If True, print detailed cavity report with individual cavity
            positions and radii.
        
        Returns
        -------
        dict
            Results stored in self.results['cavity_organic_binding']:
            - 'z_slice_centers': np.ndarray - Z-positions analyzed
            - 'cavity_data': dict - Per z-slice cavity information
                - z_center: dict with 'ring_centers', 'ring_indices', 'ring_radii'
            - 'functional_group_data': dict - Per functional group binding data
                - group_name: dict with:
                    - z_center: dict with:
                        - 'per_cavity_lifetime': np.ndarray (n_cavities, n_frames)
                        - 'avg_occupancy': np.ndarray (n_cavities,)
                        - 'std_occupancy': np.ndarray (n_cavities,)
                        - 'max_occupancy': np.ndarray (n_cavities,)
                        - 'occupancy_fraction': np.ndarray (n_cavities,)
                        - 'xy_spatial': dict (if compute_xy_spatial=True)
            - 'preferential_sites': dict - Most occupied cavities per group per z-slice
            - 'orientation_data': dict - Which functional group is closest to each cavity (if compute_orientation=True)
            - 'metadata': dict - Analysis parameters
        
        Examples
        --------
        >>> # Basic analysis of all CIP functional groups
        >>> occ_analysis.analyze_cavity_organic_binding()
        
        >>> # Analyze specific functional groups with spatial distribution
        >>> occ_analysis.analyze_cavity_organic_binding(
        ...     functional_groups=['quinolone', 'carboxylic_acid', 'piperazine'],
        ...     compute_xy_spatial=True,
        ...     compute_orientation=True,
        ...     center_box=True
        ... )
        
        >>> # High-resolution spatial analysis
        >>> occ_analysis.analyze_cavity_organic_binding(
        ...     xy_grid_size=0.3,
        ...     cavity_radius=4.0,
        ...     cavity_height=8.0,
        ...     compute_xy_spatial=True
        ... )
        """
        if verbose:
            print("\n" + "="*70)
            print("Analyzing Cavity-Specific Organic Functional Group Binding")
            print("="*70)
        
        # Validate functional groups
        if not hasattr(self, 'custom_selections') or 'CIP_parts' not in self.custom_selections:
            raise ValueError("No 'CIP_parts' defined in custom_selections. Please call define_selections() first.")
        
        if functional_groups is None:
            functional_groups = list(self.custom_selections['CIP_parts'].keys())
        elif isinstance(functional_groups, str):
            functional_groups = [functional_groups]
        
        # Validate that all requested groups exist
        missing_groups = [g for g in functional_groups if g not in self.custom_selections['CIP_parts']]
        if missing_groups:
            raise ValueError(f"Functional groups not found in CIP_parts: {missing_groups}")
        
        if verbose:
            print(f"  Functional groups to analyze: {', '.join(functional_groups)}")
        
        # Calculate z-offset for centering
        box_z = self.u.dimensions[2]
        if center_box:
            z_offset = box_z / 2
            if verbose:
                print(f"  Box centering: Enabled (z=0 at center, z_offset = {z_offset:.2f} Å)")
        else:
            z_offset = 0.0
            if verbose:
                print(f"  Box centering: Disabled (using original coordinates)")
        
        # Auto-detect z-slices if not provided (same logic as ion binding)
        if z_slice_centers is None:
            self.u.trajectory[0]
            si_atoms = self.clay.select_atoms('name Si or name SI')
            si_z_raw = si_atoms.positions[:, 2]
            si_z = si_z_raw - z_offset
            
            hist, edges = np.histogram(si_z, bins=50)
            bin_centers = (edges[:-1] + edges[1:]) / 2
            
            peaks = []
            for i in range(1, len(hist)-1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist)*0.2:
                    peaks.append(bin_centers[i])
            
            z_slice_centers = sorted(peaks)
            if verbose:
                print(f"  Auto-detected {len(z_slice_centers)} clay layers at z = {z_slice_centers}")
        
        # Cache management (similar to ion binding)
        if save_cache or cache_file:
            import hashlib
            import os
            
            if cache_file is None:
                cache_dir = '.cavity_cache'
                os.makedirs(cache_dir, exist_ok=True)
                
                param_str = f"organic_{functional_groups}_{z_slice_centers}_{z_slice_width}_{cavity_radius}_{cavity_height}_{si_si_threshold}_{center_box}_{step}_{compute_xy_spatial}_{xy_grid_size}"
                cache_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
                cache_file = os.path.join(cache_dir, f'cavity_organic_{cache_hash}.npz')
            
            # Try to load from cache
            if not force_rerun and os.path.exists(cache_file):
                if verbose:
                    print(f"\n  Loading cached results from: {cache_file}")
                
                try:
                    cached = np.load(cache_file, allow_pickle=True)
                    
                    # Reconstruct data structures (similar to ion binding cache loading)
                    cavity_data = {}
                    for z_slice in cached['z_slice_centers']:
                        z_key = f"z_{z_slice:.1f}"
                        cavity_data[z_slice] = {
                            'ring_centers': cached[f'{z_key}_ring_centers'],
                            'ring_indices': cached[f'{z_key}_ring_indices'],
                            'ring_radii': cached[f'{z_key}_ring_radii']
                        }
                    
                    functional_group_data = {}
                    for group_name in functional_groups:
                        functional_group_data[group_name] = {}
                        for z_slice in cached['z_slice_centers']:
                            z_key = f"z_{z_slice:.1f}"
                            functional_group_data[group_name][z_slice] = {}
                            
                            if f'{group_name}_{z_key}_per_cavity_lifetime' in cached:
                                functional_group_data[group_name][z_slice]['per_cavity_lifetime'] = cached[f'{group_name}_{z_key}_per_cavity_lifetime']
                            if f'{group_name}_{z_key}_avg_occupancy' in cached:
                                functional_group_data[group_name][z_slice]['avg_occupancy'] = cached[f'{group_name}_{z_key}_avg_occupancy']
                            if f'{group_name}_{z_key}_std_occupancy' in cached:
                                functional_group_data[group_name][z_slice]['std_occupancy'] = cached[f'{group_name}_{z_key}_std_occupancy']
                            if f'{group_name}_{z_key}_max_occupancy' in cached:
                                functional_group_data[group_name][z_slice]['max_occupancy'] = cached[f'{group_name}_{z_key}_max_occupancy']
                            if f'{group_name}_{z_key}_occupancy_fraction' in cached:
                                functional_group_data[group_name][z_slice]['occupancy_fraction'] = cached[f'{group_name}_{z_key}_occupancy_fraction']
                            
                            # Load xy_spatial data if present
                            if f'{group_name}_{z_key}_xy_density' in cached:
                                functional_group_data[group_name][z_slice]['xy_spatial'] = {
                                    'functional_group_density_grid': cached[f'{group_name}_{z_key}_xy_density'],
                                    'cavity_occupancy_grid': cached[f'{group_name}_{z_key}_xy_cavity_occupancy'],
                                    'x_centers': cached[f'{group_name}_{z_key}_xy_x_centers'],
                                    'y_centers': cached[f'{group_name}_{z_key}_xy_y_centers'],
                                    'x_edges': cached[f'{group_name}_{z_key}_xy_x_edges'],
                                    'y_edges': cached[f'{group_name}_{z_key}_xy_y_edges'],
                                    'grid_dimensions': tuple(cached[f'{group_name}_{z_key}_xy_grid_dims']),
                                    'box_dimensions': tuple(cached[f'{group_name}_{z_key}_xy_box_dims']),
                                    'surface_type': str(cached[f'{group_name}_{z_key}_xy_surface_type']),
                                    'cavity_centers_xy': cached[f'{group_name}_{z_key}_xy_cavity_centers'],
                                    'n_frames': int(cached[f'{group_name}_{z_key}_xy_n_frames'])
                                }
                    
                    # Load preferential_sites - handle both dict and object array storage
                    if 'preferential_sites' in cached:
                        pref_sites = cached['preferential_sites']
                        if isinstance(pref_sites, np.ndarray) and pref_sites.dtype == object:
                            preferential_sites = pref_sites.item() if pref_sites.size == 1 else pref_sites
                        else:
                            preferential_sites = pref_sites
                    else:
                        preferential_sites = {}
                    
                    # Load orientation_data - handle both dict and object array storage
                    if 'orientation_data' in cached:
                        orient_data = cached['orientation_data']
                        if isinstance(orient_data, np.ndarray) and orient_data.dtype == object:
                            orientation_data = orient_data.item() if orient_data.size == 1 else orient_data
                        else:
                            orientation_data = orient_data
                    else:
                        orientation_data = {}
                    
                    result = {
                        'cavity_data': cavity_data,
                        'functional_group_data': functional_group_data,
                        'preferential_sites': preferential_sites,
                        'orientation_data': orientation_data,
                        'metadata': {
                            'functional_groups': functional_groups,
                            'z_slice_centers': z_slice_centers,
                            'z_slice_width': z_slice_width,
                            'cavity_radius': cavity_radius,
                            'cavity_height': cavity_height,
                            'si_si_threshold': si_si_threshold,
                            'center_box': center_box,
                            'n_frames': self.u.trajectory.n_frames
                        }
                    }
                    
                    self.results['cavity_organic_binding'] = result
                    
                    if verbose:
                        print("  ✓ Successfully loaded from cache")
                    
                    return result
                    
                except Exception as e:
                    if verbose:
                        print(f"  ⚠ Failed to load cache: {e}")
                        print("  Proceeding with full analysis...")
        
        # Initialize results
        cavity_data = {}
        functional_group_data = {group: {} for group in functional_groups}
        preferential_sites = {group: {} for group in functional_groups}
        orientation_data = {} if compute_orientation else None
        
        # Detect cavities in each z-slice (reuse same logic)
        if verbose:
            print(f"\n📍 Detecting Si hexagonal ring cavities...")
            print(f"   Si-Si threshold: {si_si_threshold} Å")
            print(f"   Z-slices: {len(z_slice_centers)}")
        
        self.u.trajectory[0]
        si_atoms = self.clay.select_atoms('name Si or name SI')
        
        for z_center in z_slice_centers:
            si_z_centered = si_atoms.positions[:, 2] - z_offset
            z_mask = np.abs(si_z_centered - z_center) <= z_slice_width / 2
            si_positions_raw = si_atoms.positions[z_mask]
            
            si_positions = si_positions_raw.copy()
            si_positions[:, 2] -= z_offset
            
            if len(si_positions) < 6:
                if verbose:
                    print(f"   WARNING: Only {len(si_positions)} Si atoms in slice z={z_center:.1f}, skipping.")
                continue
            
            ring_data = self._detect_si_hexagonal_rings(
                si_positions, 
                si_si_threshold=si_si_threshold,
                min_ring_size=6,
                max_ring_size=6
            )
            
            cavity_data[z_center] = ring_data
            
            if verbose:
                n_rings = len(ring_data['ring_centers'])
                print(f"   z = {z_center:6.1f} Å: Found {n_rings} hexagonal ring cavities")
                
                if n_rings > 0:
                    ring_radii = ring_data['ring_radii']
                    print(f"      Cavity radius: Mean={np.mean(ring_radii):.3f} Å, "
                          f"Std={np.std(ring_radii):.3f} Å")
        
        # Print detailed cavity report if requested
        if show_cavity_report and len(cavity_data) > 0:
            self._print_cavity_report(cavity_data, z_slice_centers, 
                                     si_si_threshold, cavity_radius, 
                                     cavity_height, center_box, z_offset)
        
        # Analyze functional group binding to cavities
        if verbose:
            print(f"\n🔬 Analyzing functional group binding to cavities...")
            print(f"   Cavity radius: {cavity_radius} Å")
            print(f"   Cavity height: {cavity_height} Å")
            print(f"   Functional groups: {', '.join(functional_groups)}")
        
        n_frames = len(self.u.trajectory[::step])
        
        # Create atom selections for each functional group
        functional_group_atoms = {}
        for group_name in functional_groups:
            selection_or_group = self.custom_selections['CIP_parts'][group_name]
            
            # Check if it's a string (selection) or already an AtomGroup
            if isinstance(selection_or_group, str):
                functional_group_atoms[group_name] = self.u.select_atoms(selection_or_group)
            else:
                # It's already an AtomGroup from define_selections()
                functional_group_atoms[group_name] = selection_or_group
            
            if verbose:
                print(f"\n   {group_name}: {len(functional_group_atoms[group_name])} atoms")
        
        # Process each functional group
        for group_name in functional_groups:
            if verbose:
                print(f"\n   Processing {group_name}...")
            
            group_atoms = functional_group_atoms[group_name]
            
            for z_center in z_slice_centers:
                if z_center not in cavity_data:
                    continue
                
                ring_centers = cavity_data[z_center]['ring_centers']
                n_cavities = len(ring_centers)
                
                if n_cavities == 0:
                    continue
                
                # Initialize storage
                if compute_occupancy_lifetime:
                    per_cavity_lifetime = np.zeros((n_cavities, n_frames))
                
                if compute_orientation:
                    # Store closest functional group for each cavity at each frame
                    if z_center not in orientation_data:
                        orientation_data[z_center] = {
                            'closest_group_per_cavity': np.zeros((n_cavities, n_frames), dtype=object),
                            'closest_distance_per_cavity': np.full((n_cavities, n_frames), np.inf)
                        }
                
                # Analyze each frame
                for frame_idx, ts in enumerate(self.u.trajectory[::step]):
                    group_positions_raw = group_atoms.positions
                    group_positions = group_positions_raw.copy()
                    group_positions[:, 2] -= z_offset
                    
                    # Count atoms in each cavity
                    for cavity_idx, cavity_center in enumerate(ring_centers):
                        # Calculate center of mass of functional group atoms
                        if len(group_positions) > 0:
                            group_com = np.mean(group_positions, axis=0)
                            
                            # XY distance from cavity center
                            xy_distance = np.linalg.norm(
                                group_positions[:, :2] - cavity_center[:2], axis=1
                            )
                            
                            # Z distance - cylinder extends toward z=0
                            z_distance = group_positions[:, 2] - cavity_center[2]
                            
                            # Apply cylindrical mask (same logic as ion binding)
                            if cavity_center[2] > 0:  # Upper surface
                                in_cavity_mask = (xy_distance <= cavity_radius) & \
                                                (z_distance <= 0) & \
                                                (z_distance >= -cavity_height)
                            else:  # Lower surface
                                in_cavity_mask = (xy_distance <= cavity_radius) & \
                                                (z_distance >= 0) & \
                                                (z_distance <= cavity_height)
                            
                            n_atoms_in_cavity = np.sum(in_cavity_mask)
                            
                            if compute_occupancy_lifetime:
                                per_cavity_lifetime[cavity_idx, frame_idx] = n_atoms_in_cavity
                            
                            # Orientation analysis: track closest functional group to each cavity
                            if compute_orientation and n_atoms_in_cavity > 0:
                                # Find minimum distance from any atom in this group to cavity center
                                atoms_in_cavity = group_positions[in_cavity_mask]
                                distances_to_cavity = np.linalg.norm(
                                    atoms_in_cavity - cavity_center, axis=1
                                )
                                min_distance = np.min(distances_to_cavity)
                                
                                # Update if this is closer than previous groups
                                if min_distance < orientation_data[z_center]['closest_distance_per_cavity'][cavity_idx, frame_idx]:
                                    orientation_data[z_center]['closest_distance_per_cavity'][cavity_idx, frame_idx] = min_distance
                                    orientation_data[z_center]['closest_group_per_cavity'][cavity_idx, frame_idx] = group_name
                
                # Calculate statistics
                functional_group_data[group_name][z_center] = {}
                
                if compute_occupancy_lifetime:
                    functional_group_data[group_name][z_center]['per_cavity_lifetime'] = per_cavity_lifetime
                
                if compute_avg_occupancy:
                    functional_group_data[group_name][z_center]['avg_occupancy'] = np.mean(per_cavity_lifetime, axis=1)
                    functional_group_data[group_name][z_center]['std_occupancy'] = np.std(per_cavity_lifetime, axis=1)
                    functional_group_data[group_name][z_center]['max_occupancy'] = np.max(per_cavity_lifetime, axis=1)
                    functional_group_data[group_name][z_center]['occupancy_fraction'] = \
                        np.sum(per_cavity_lifetime > 0, axis=1) / n_frames
                
                if compute_preferential_sites:
                    avg_occ = functional_group_data[group_name][z_center]['avg_occupancy']
                    top_indices = np.argsort(avg_occ)[::-1][:5]
                    
                    preferential_sites[group_name][z_center] = {
                        'cavity_indices': top_indices,
                        'cavity_positions': ring_centers[top_indices],
                        'avg_occupancy': avg_occ[top_indices]
                    }
                
                if verbose:
                    avg_total = np.mean(np.sum(per_cavity_lifetime, axis=0))
                    print(f"      z = {z_center:6.1f} Å: {n_cavities} cavities, "
                          f"avg {avg_total:.2f} {group_name} atoms in cavities")
        
        # Compute orientation statistics
        if compute_orientation and orientation_data:
            if verbose:
                print(f"\n🧭 Computing orientation statistics...")
            
            for z_center in orientation_data.keys():
                closest_groups = orientation_data[z_center]['closest_group_per_cavity']
                
                # Count frequency of each functional group being closest
                orientation_data[z_center]['group_frequency'] = {}
                for group_name in functional_groups:
                    # Count how many times this group was closest across all cavities and frames
                    count = np.sum(closest_groups == group_name)
                    total = closest_groups.size
                    orientation_data[z_center]['group_frequency'][group_name] = count / total if total > 0 else 0.0
                
                if verbose:
                    print(f"   z = {z_center:6.1f} Å:")
                    for group_name, freq in orientation_data[z_center]['group_frequency'].items():
                        print(f"      {group_name}: {freq*100:.1f}% of cavity-frames")
        
        # Compute XY spatial distribution if requested
        if compute_xy_spatial:
            if verbose:
                print(f"\n📍 Computing XY spatial distributions for functional groups...")
                print(f"   Grid size: {xy_grid_size} Å")
            
            box_x = self.u.dimensions[0]
            box_y = self.u.dimensions[1]
            
            n_x = int(np.ceil(box_x / xy_grid_size))
            n_y = int(np.ceil(box_y / xy_grid_size))
            x_edges = np.linspace(0, box_x, n_x + 1)
            y_edges = np.linspace(0, box_y, n_y + 1)
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            
            if verbose:
                print(f"   Grid dimensions: {n_x} × {n_y} bins")
                print(f"   Box size: {box_x:.1f} × {box_y:.1f} Å")
            
            for z_center in z_slice_centers:
                if z_center not in cavity_data:
                    continue
                
                ring_centers = cavity_data[z_center]['ring_centers']
                n_cavities = len(ring_centers)
                
                if n_cavities == 0:
                    continue
                
                surface_type = 'top' if z_center > 0 else 'bottom'
                
                # Determine z-range for spatial analysis
                if z_center > 0:
                    z_min_slice = z_center - cavity_height
                    z_max_slice = z_center
                else:
                    z_min_slice = z_center
                    z_max_slice = z_center + cavity_height
                
                for group_name in functional_groups:
                    if z_center not in functional_group_data[group_name]:
                        continue
                    
                    group_atoms = functional_group_atoms[group_name]
                    
                    # Initialize grids
                    functional_group_density_grid = np.zeros((n_y, n_x))
                    cavity_occupancy_grid = np.zeros((n_y, n_x))
                    
                    # Calculate density grid from trajectory
                    for frame_idx, ts in enumerate(self.u.trajectory[::step]):
                        group_positions_raw = group_atoms.positions
                        group_positions = group_positions_raw.copy()
                        group_positions[:, 2] -= z_offset
                        
                        z_positions = group_positions[:, 2]
                        in_slice = (z_positions >= z_min_slice) & (z_positions < z_max_slice)
                        
                        if np.any(in_slice):
                            atoms_in_slice = group_positions[in_slice]
                            
                            for atom_pos in atoms_in_slice:
                                x, y = atom_pos[0] % box_x, atom_pos[1] % box_y
                                x_idx = int(x / xy_grid_size)
                                y_idx = int(y / xy_grid_size)
                                
                                if 0 <= x_idx < n_x and 0 <= y_idx < n_y:
                                    functional_group_density_grid[y_idx, x_idx] += 1
                    
                    # Normalize density
                    grid_area = xy_grid_size ** 2
                    functional_group_density_grid = functional_group_density_grid / (n_frames * grid_area)
                    
                    # Create cavity occupancy grid
                    if 'avg_occupancy' in functional_group_data[group_name][z_center]:
                        avg_occupancy = functional_group_data[group_name][z_center]['avg_occupancy']
                        
                        for i in range(n_y):
                            for j in range(n_x):
                                grid_x = x_centers[j]
                                grid_y = y_centers[i]
                                
                                dx = ring_centers[:, 0] - grid_x
                                dy = ring_centers[:, 1] - grid_y
                                
                                dx = dx - box_x * np.round(dx / box_x)
                                dy = dy - box_y * np.round(dy / box_y)
                                
                                distances = np.sqrt(dx**2 + dy**2)
                                nearest_cavity_idx = np.argmin(distances)
                                cavity_occupancy_grid[i, j] = avg_occupancy[nearest_cavity_idx]
                    
                    # Store spatial data
                    functional_group_data[group_name][z_center]['xy_spatial'] = {
                        'functional_group_density_grid': functional_group_density_grid,
                        'cavity_occupancy_grid': cavity_occupancy_grid,
                        'x_centers': x_centers,
                        'y_centers': y_centers,
                        'x_edges': x_edges,
                        'y_edges': y_edges,
                        'grid_dimensions': (n_x, n_y),
                        'box_dimensions': (box_x, box_y),
                        'surface_type': surface_type,
                        'cavity_centers_xy': ring_centers[:, :2],
                        'n_frames': n_frames
                    }
                    
                    if verbose:
                        print(f"   {group_name} @ z={z_center:.1f} Å: "
                              f"density range [{np.min(functional_group_density_grid):.4f}, "
                              f"{np.max(functional_group_density_grid):.4f}]")
        
        # Store results
        self.results['cavity_organic_binding'] = {
            'cavity_data': cavity_data,
            'functional_group_data': functional_group_data,
            'preferential_sites': preferential_sites,
            'orientation_data': orientation_data,
            'metadata': {
                'functional_groups': functional_groups,
                'z_slice_centers': z_slice_centers,
                'z_slice_width': z_slice_width,
                'cavity_radius': cavity_radius,
                'cavity_height': cavity_height,
                'si_si_threshold': si_si_threshold,
                'center_box': center_box,
                'n_frames': self.u.trajectory.n_frames,
                'step': step,
                'xy_grid_size': xy_grid_size if compute_xy_spatial else None
            }
        }
        
        # Save cache if requested
        if save_cache and cache_file:
            if verbose:
                print(f"\n💾 Saving cache to: {cache_file}")
            
            try:
                save_dict = {
                    'z_slice_centers': z_slice_centers,
                    'preferential_sites': preferential_sites,
                    'orientation_data': orientation_data if orientation_data else {}
                }
                
                # Save cavity data
                for z_slice in z_slice_centers:
                    if z_slice in cavity_data:
                        z_key = f"z_{z_slice:.1f}"
                        save_dict[f'{z_key}_ring_centers'] = cavity_data[z_slice]['ring_centers']
                        save_dict[f'{z_key}_ring_indices'] = cavity_data[z_slice]['ring_indices']
                        save_dict[f'{z_key}_ring_radii'] = cavity_data[z_slice]['ring_radii']
                
                # Save functional group data
                for group_name in functional_groups:
                    for z_slice in z_slice_centers:
                        if z_slice in functional_group_data[group_name]:
                            z_key = f"z_{z_slice:.1f}"
                            group_data = functional_group_data[group_name][z_slice]
                            
                            if 'per_cavity_lifetime' in group_data:
                                save_dict[f'{group_name}_{z_key}_per_cavity_lifetime'] = group_data['per_cavity_lifetime']
                            if 'avg_occupancy' in group_data:
                                save_dict[f'{group_name}_{z_key}_avg_occupancy'] = group_data['avg_occupancy']
                            if 'std_occupancy' in group_data:
                                save_dict[f'{group_name}_{z_key}_std_occupancy'] = group_data['std_occupancy']
                            if 'max_occupancy' in group_data:
                                save_dict[f'{group_name}_{z_key}_max_occupancy'] = group_data['max_occupancy']
                            if 'occupancy_fraction' in group_data:
                                save_dict[f'{group_name}_{z_key}_occupancy_fraction'] = group_data['occupancy_fraction']
                            
                            # Save xy_spatial data
                            if 'xy_spatial' in group_data:
                                xy = group_data['xy_spatial']
                                save_dict[f'{group_name}_{z_key}_xy_density'] = xy['functional_group_density_grid']
                                save_dict[f'{group_name}_{z_key}_xy_cavity_occupancy'] = xy['cavity_occupancy_grid']
                                save_dict[f'{group_name}_{z_key}_xy_x_centers'] = xy['x_centers']
                                save_dict[f'{group_name}_{z_key}_xy_y_centers'] = xy['y_centers']
                                save_dict[f'{group_name}_{z_key}_xy_x_edges'] = xy['x_edges']
                                save_dict[f'{group_name}_{z_key}_xy_y_edges'] = xy['y_edges']
                                save_dict[f'{group_name}_{z_key}_xy_grid_dims'] = xy['grid_dimensions']
                                save_dict[f'{group_name}_{z_key}_xy_box_dims'] = xy['box_dimensions']
                                save_dict[f'{group_name}_{z_key}_xy_surface_type'] = xy['surface_type']
                                save_dict[f'{group_name}_{z_key}_xy_cavity_centers'] = xy['cavity_centers_xy']
                                save_dict[f'{group_name}_{z_key}_xy_n_frames'] = xy['n_frames']
                
                np.savez_compressed(cache_file, **save_dict)
                
                if verbose:
                    print("  ✓ Cache saved successfully")
                    
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Failed to save cache: {e}")
        
        if verbose:
            print("\n" + "="*70)
            print("✅ Analysis Complete!")
            print("="*70)
            print(f"Results stored in self.results['cavity_organic_binding']")
            print(f"Functional groups analyzed: {', '.join(functional_groups)}")
            print(f"Z-slices: {len(z_slice_centers)}")
            if compute_orientation:
                print("Orientation analysis: ✓ Enabled")
            if compute_xy_spatial:
                print(f"XY spatial grids: ✓ Generated ({xy_grid_size} Å resolution)")
        
        return self.results['cavity_organic_binding']
    
    def analyze_ion_peaks_manual(self, ion_densities_dict=None, z_centers=None,
                                 peak_positions_dict=None,
                                 peak_height_threshold=0.1,
                                 peak_distance=5.0,
                                 save_plots=True,
                                 show_plot=True,
                                 figsize=(15, 10),
                                 filename='ion_peak_analysis_manual.png',
                                 show_clay_boundaries=True,
                                 save_table=False,
                                 table_filename='ion_peaks_table.csv',
                                 print_table=True):
        """
        Flexible ion peak analysis with manual control over input data and peak positions.
        
        This method allows you to:
        1. Pass custom ion density profiles (instead of auto-calculating from trajectory)
        2. Override automatic peak finding with manual peak positions
        3. Analyze peaks with full control over the input data
        
        Parameters
        ----------
        ion_densities_dict : dict, optional
            Manual ion density data. Format: {'NA': density_array, 'CL': density_array}
            If None, uses self.results['edl_analysis']['ion_densities']
        z_centers : array_like, optional
            Z-position array corresponding to density data
            If None, uses self.results['edl_analysis']['z_centers']
        peak_positions_dict : dict, optional
            Manual peak positions for each ion. Format: {'NA': [24.5, 12.3], 'CL': [22.1]}
            If provided, overrides automatic peak finding for that ion
            If None or ion not in dict, uses automatic peak finding
        peak_height_threshold : float, default=0.1
            Minimum peak height for automatic peak finding (only used if peak_positions_dict not provided)
        peak_distance : float, default=5.0
            Minimum distance between peaks in Angstroms (for automatic finding)
        save_plots : bool, default=True
            Save plots to file
        show_plot : bool, default=True
            Display the plot interactively. Set to False to suppress plot display.
        figsize : tuple, default=(15, 10)
            Figure size for plots
        filename : str, default='ion_peak_analysis_manual.png'
            Output filename for plots
        show_clay_boundaries : bool, default=True
            Show clay surface boundaries on plots
        save_table : bool, default=False
            Save peak table to CSV file
        table_filename : str, default='ion_peaks_table.csv'
            Output filename for CSV table
        print_table : bool, default=True
            Print formatted peak table to console
        
        Returns
        -------
        dict
            Dictionary containing peak data for each ion type with keys:
            - 'peak_positions': array of z-positions of peaks
            - 'peak_densities': array of density values at peaks
            - 'peak_heights': array of peak heights
            - 'source': 'manual' or 'automatic' indicating how peaks were determined
        
        Examples
        --------
        # Example 1: Use EDL analysis results with automatic peak finding
        peaks = occ.analyze_ion_peaks_manual()
        
        # Example 2: Use EDL results but manually specify peak positions for Na
        peaks = occ.analyze_ion_peaks_manual(
            peak_positions_dict={'NA': [24.5, 18.2, 12.3]}
        )
        
        # Example 3: Provide completely custom density data
        custom_densities = {'NA': my_na_density, 'CL': my_cl_density}
        custom_z = np.linspace(-30, 30, 100)
        peaks = occ.analyze_ion_peaks_manual(
            ion_densities_dict=custom_densities,
            z_centers=custom_z
        )
        
        # Example 4: Full manual control - custom data AND custom peaks
        peaks = occ.analyze_ion_peaks_manual(
            ion_densities_dict=custom_densities,
            z_centers=custom_z,
            peak_positions_dict={'NA': [24.5], 'CL': [22.1]}
        )
        """
        
        from scipy.signal import find_peaks
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Get ion density data
        if ion_densities_dict is None or z_centers is None:
            # Calculate ion densities independently
            print("Calculating ion density profiles from trajectory...")
            
            # Get box dimensions
            self.u.trajectory[0]
            box_dims = self.u.dimensions
            if box_dims is None:
                raise ValueError("Universe has no box dimensions. Cannot calculate densities.")
            
            box_x, box_y, box_z = box_dims[:3]
            
            # Determine centering offset (same logic as EDL analysis)
            center_box = True  # Default to centering for consistency
            if center_box:
                z_offset = box_z / 2
                z_min = -box_z / 2
                z_max = box_z / 2
            else:
                z_offset = 0
                z_min = 0
                z_max = box_z
            
            # Create bins
            z_bin_width = 0.2  # Default bin width
            z_bins = np.arange(z_min, z_max + z_bin_width, z_bin_width)
            z_centers_calc = (z_bins[:-1] + z_bins[1:]) / 2
            n_bins = len(z_centers_calc)
            
            print(f"  Box: {box_x:.1f} × {box_y:.1f} × {box_z:.1f} Å")
            print(f"  Z-range: {z_min:.2f} to {z_max:.2f} Å (centered at z=0)")
            print(f"  Bin width: {z_bin_width} Å, n_bins: {n_bins}")
            
            # Initialize storage
            ion_densities_calc = {}
            for ion_name in self.ions.keys():
                ion_densities_calc[ion_name] = np.zeros(n_bins)
            
            # Calculate densities over trajectory
            n_frames = 0
            step = 1  # Analyze every frame
            
            for ts in self.u.trajectory[::step]:
                for ion_name, ion_atoms in self.ions.items():
                    if len(ion_atoms) == 0:
                        continue
                    
                    # Get ion z-positions and apply centering
                    ion_z = ion_atoms.positions[:, 2] - z_offset
                    
                    # Bin the ions
                    hist, _ = np.histogram(ion_z, bins=z_bins)
                    
                    # Convert counts to density (ions/Å³)
                    bin_volume = box_x * box_y * z_bin_width
                    density = hist / bin_volume
                    
                    # Accumulate
                    ion_densities_calc[ion_name] += density
                
                n_frames += 1
            
            # Average over frames
            for ion_name in ion_densities_calc.keys():
                ion_densities_calc[ion_name] /= n_frames
            
            print(f"  Processed {n_frames} frames")
            
            # Use calculated values
            if ion_densities_dict is None:
                ion_densities_dict = ion_densities_calc
            if z_centers is None:
                z_centers = z_centers_calc
        
        elif ion_densities_dict is None:
            # Try to get from EDL analysis results
            if 'edl_analysis' in self.results and 'ion_densities' in self.results['edl_analysis']:
                ion_densities_dict = self.results['edl_analysis']['ion_densities']
                print("Using ion densities from EDL analysis")
            else:
                raise ValueError("No ion density data available. Provide ion_densities_dict or run EDL analysis first.")
        
        # Get z_centers if still None
        if z_centers is None:
            if 'edl_analysis' in self.results and 'z_centers' in self.results['edl_analysis']:
                z_centers = self.results['edl_analysis']['z_centers']
            else:
                raise ValueError("No z_centers data available. Provide z_centers parameter.")
        
        z_centers = np.array(z_centers)
        ion_types = list(ion_densities_dict.keys())
        
        # Create subplots for each ion
        n_ions = len(ion_types)
        ncols = min(2, n_ions)
        nrows = (n_ions + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_ions == 1:
            axes = [axes]
        elif nrows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()
        
        # Initialize peak data storage
        peak_summary = {}
        all_peaks_data = []  # For CSV table
        
        # Get surface position for boundaries (if available)
        surface_position = None
        if 'edl_analysis' in self.results and 'surface_position' in self.results['edl_analysis']:
            surface_position = self.results['edl_analysis']['surface_position']
        
        for idx, ion_type in enumerate(ion_types):
            ax = axes[idx]
            density = np.array(ion_densities_dict[ion_type])
            
            # Plot density profile
            ax.plot(z_centers, density, 'b-', linewidth=2, label=f'{ion_type} density')
            
            # Determine peaks: manual or automatic
            if peak_positions_dict is not None and ion_type in peak_positions_dict:
                # Use manual peak positions
                manual_peak_positions = np.array(peak_positions_dict[ion_type])
                peak_positions = manual_peak_positions
                
                # Find closest z_centers indices for density values
                peak_indices = []
                for pos in manual_peak_positions:
                    idx_closest = np.argmin(np.abs(z_centers - pos))
                    peak_indices.append(idx_closest)
                peak_indices = np.array(peak_indices)
                
                peak_densities = density[peak_indices]
                peak_heights = peak_densities  # For manual, height = density value
                peak_source = 'manual'
                
                print(f"\n{ion_type}: Using {len(peak_positions)} manual peak positions")
            else:
                # Automatic peak finding
                # Debug: print density stats
                print(f"\n{ion_type} density statistics:")
                print(f"  Min: {np.min(density):.6e}, Max: {np.max(density):.6e}")
                print(f"  Mean: {np.mean(density):.6e}, Std: {np.std(density):.6e}")
                print(f"  Threshold: {peak_height_threshold:.6e}")
                print(f"  Distance: {peak_distance} Å ({int(peak_distance / np.mean(np.diff(z_centers)))} bins)")
                
                peaks_idx, properties = find_peaks(
                    density,
                    height=peak_height_threshold,
                    distance=int(peak_distance / np.mean(np.diff(z_centers)))
                )
                
                if len(peaks_idx) > 0:
                    peak_positions = z_centers[peaks_idx]
                    peak_densities = density[peaks_idx]
                    peak_heights = properties['peak_heights']
                    peak_source = 'automatic'
                    
                    print(f"  ✓ Found {len(peak_positions)} peaks automatically")
                else:
                    peak_positions = np.array([])
                    peak_densities = np.array([])
                    peak_heights = np.array([])
                    peak_source = 'automatic'
                    
                    print(f"  ✗ No peaks found (max density {np.max(density):.6e} < threshold {peak_height_threshold:.6e}?)")
            
            # Plot peaks if any exist
            if len(peak_positions) > 0:
                ax.plot(peak_positions, peak_densities, 'ro', markersize=8,
                       markeredgecolor='black', markeredgewidth=1, label='Peaks')
                
                # Annotate peaks
                for i, (pos, dens) in enumerate(zip(peak_positions, peak_densities)):
                    ax.annotate(f'P{i+1}\n{pos:.1f}Å\n{dens:.4f}',
                               xy=(pos, dens),
                               xytext=(10, 10),
                               textcoords='offset points',
                               fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                    
                    # Store for table
                    all_peaks_data.append({
                        'Ion': ion_type,
                        'Peak_Number': i+1,
                        'Z_Position_A': pos,
                        'Density': dens,
                        'Height': peak_heights[i] if i < len(peak_heights) else dens,
                        'Source': peak_source
                    })
                
                # Store summary
                peak_summary[ion_type] = {
                    'peak_positions': peak_positions,
                    'peak_densities': peak_densities,
                    'peak_heights': peak_heights,
                    'source': peak_source,
                    'n_peaks': len(peak_positions)
                }
            else:
                peak_summary[ion_type] = {
                    'peak_positions': np.array([]),
                    'peak_densities': np.array([]),
                    'peak_heights': np.array([]),
                    'source': peak_source,
                    'n_peaks': 0
                }
            
            # Show clay boundaries if available
            if show_clay_boundaries and surface_position is not None:
                ax.axvline(surface_position, color='brown', linestyle='--',
                          linewidth=2, alpha=0.7, label=f'Surface ({surface_position:.1f}Å)')
            
            # Formatting
            ax.set_xlabel('Z-position (Å)', fontsize=12)
            ax.set_ylabel('Ion Density', fontsize=12)
            ax.set_title(f'{ion_type} Ion Density Profile', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_ions, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n✅ Peak analysis plot saved: {filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Create and display table
        if len(all_peaks_data) > 0:
            df = pd.DataFrame(all_peaks_data)
            
            if print_table:
                print("\n" + "="*80)
                print("ION PEAK ANALYSIS SUMMARY")
                print("="*80)
                print(df.to_string(index=False))
                print("="*80)
            
            if save_table:
                df.to_csv(table_filename, index=False)
                print(f"\n✅ Peak table saved: {table_filename}")
        else:
            print("\n⚠️  No peaks found or specified for any ion type")
        
        # Store results for use by other methods (e.g., EDL analysis)
        self.results['ion_peak_analysis'] = peak_summary
        
        return peak_summary

    def calculate_clay_spatial_distribution_xy(self, 
                                            z_slice_centers=None,
                                            z_slice_width=2.0,
                                            xy_grid_size=0.5,
                                            mg_vdw_radius=0.72,
                                            si_vdw_radius=1.11,
                                            mg_vdw_scaling=2.0,
                                            si_vdw_scaling=1.5,
                                            step=1,
                                            max_frames=None,
                                            combine_mg_layers=True,
                                            center_box=False,
                                            save_cache=True,
                                            force_rerun=False,
                                            cache_file=None):
        """
        Calculate clay spatial distribution for contour overlay plots.
        
        This method calculates 2D density grids for Mg and Si atoms from clay layers,
        which can be used to overlay clay structure on ion distribution plots.
        
        Parameters
        ----------
        z_slice_centers : array-like, optional
            Z-positions for slice centers. If None, uses default range (-30 to 30 Å).
        z_slice_width : float, default=2.0
            Width of each z-slice (Å)
        xy_grid_size : float, default=0.5
            Grid spacing for XY plane (Å)
        mg_vdw_radius : float, default=0.72
            Base Van der Waals radius for Mg atoms (Å)
        si_vdw_radius : float, default=1.11
            Base Van der Waals radius for Si atoms (Å)
        mg_vdw_scaling : float, default=2.0
            Scaling factor for Mg VdW radius
        si_vdw_scaling : float, default=1.5
            Scaling factor for Si VdW radius
        step : int, default=1
            Frame sampling interval
        max_frames : int, optional
            Maximum number of frames to analyze
        combine_mg_layers : bool, default=True
            If True, combines Mg atoms from both clay layers in all z-slices.
        center_box : bool, default=False
            If True, centers coordinates around z=0 by subtracting box_center_z
        save_cache : bool, default=True
            Save results to cache file
        force_rerun : bool, default=False
            Force reanalysis even if cache exists
        cache_file : str, optional
            Custom cache filename. If None, auto-generates from parameters
        
        Returns
        -------
        dict
            Clay spatial distribution results with Mg and Si grids for each z-slice
            
        Example
        -------
        >>> # Calculate clay distribution for contour overlays
        >>> clay_results = analysis.calculate_clay_spatial_distribution_xy(
        ...     z_slice_centers=[-27, 27],
        ...     xy_grid_size=0.5,
        ...     mg_vdw_scaling=2.0
        ... )
        >>> 
        >>> # Use in plotting
        >>> plotter.plot_cavity_occupancy(
        ...     overlay_clay_contours=True,
        ...     clay_contour_results=clay_results
        ... )
        """
        
        # Generate cache filename if not provided
        if cache_file is None:
            z_str = 'all' if z_slice_centers is None else f"{len(z_slice_centers)}slices"
            cache_file = f".clay_spatial_xy_{z_str}_grid{xy_grid_size}_mg{mg_vdw_scaling}_si{si_vdw_scaling}_combined{combine_mg_layers}.npz"
        
        # Try to load from cache
        if not force_rerun and os.path.exists(cache_file):
            print(f"📦 Loading clay spatial distribution from cache: {cache_file}")
            try:
                cached = np.load(cache_file, allow_pickle=True)
                
                # Reconstruct results dict
                results = {
                    'z_slice_width': float(cached['z_slice_width']),
                    'xy_grid_size': float(cached['xy_grid_size']),
                    'mg_vdw_scaling': float(cached['mg_vdw_scaling']),
                    'si_vdw_scaling': float(cached['si_vdw_scaling']),
                    'mg_vdw_radius': float(cached['mg_vdw_radius']),
                    'si_vdw_radius': float(cached['si_vdw_radius']),
                    'combine_mg_layers': bool(cached['combine_mg_layers']),
                    'z_slices': []
                }
                
                # Reconstruct z_slices
                n_slices = int(cached['n_slices'])
                for i in range(n_slices):
                    z_slice_data = {
                        'z_center': float(cached[f'z{i}_center']),
                        'x_centers': cached[f'z{i}_x_centers'],
                        'y_centers': cached[f'z{i}_y_centers'],
                        'x_edges': cached[f'z{i}_x_edges'],
                        'y_edges': cached[f'z{i}_y_edges'],
                        'mg_grid': cached[f'z{i}_mg_grid'],
                        'si_grid': cached[f'z{i}_si_grid'],
                        'combined_grid': cached[f'z{i}_combined_grid']
                    }
                    results['z_slices'].append(z_slice_data)
                
                print(f"✅ Loaded {n_slices} z-slices from cache")
                return results
                
            except Exception as e:
                print(f"⚠️ Cache loading failed: {e}")
                print(f"   Proceeding with fresh analysis...")
        
        print(f"\n🧱 Clay Spatial Distribution Analysis")
        print(f"{'='*50}")
        print(f"  Grid approach: Fast vectorized operations")
        print(f"  Mg VdW scaling: {mg_vdw_scaling}x")
        print(f"  Si VdW scaling: {si_vdw_scaling}x")
        
        if combine_mg_layers:
            print(f"  🔗 Mg layer combination: ENABLED (both clay layers)")
        else:
            print(f"  🔗 Mg layer combination: DISABLED (separate by layer)")
        
        # Get clay atoms
        try:
            # Mg atoms
            mg_atoms = self.u.select_atoms("resname MMT and (name Mgo or name MGO or name Mg or name MG)")
            print(f"Found {len(mg_atoms)} Mg atoms")
            
            # Si atoms  
            si_atoms = self.u.select_atoms("resname MMT and (name Si or name SI or name Sio or name SIO)")
            print(f"Found {len(si_atoms)} Si atoms")
            
            if len(mg_atoms) == 0 and len(si_atoms) == 0:
                print("❌ No clay atoms found!")
                return None
                
        except Exception as e:
            print(f"❌ Error selecting clay atoms: {e}")
            return None
        
        # Set up z-slices
        if z_slice_centers is None:
            z_slice_centers = np.arange(-30, 31, z_slice_width)
        
        print(f"Processing {len(z_slice_centers)} z-slices...")
        
        # Mg and Si parameters
        scaled_mg_radius = mg_vdw_radius * mg_vdw_scaling
        scaled_si_radius = si_vdw_radius * si_vdw_scaling
        
        # Determine frame range
        total_frames = len(self.u.trajectory)
        if max_frames is not None:
            frames_to_process = min(max_frames, total_frames)
        else:
            frames_to_process = total_frames
        
        frame_indices = range(0, frames_to_process, step)
        print(f"Processing {len(frame_indices)} frames (step={step})")
        
        # Get box dimensions from first frame
        self.u.trajectory[0]
        box_dims = self.u.dimensions
        box_x, box_y = box_dims[0], box_dims[1]
        box_center_z = box_dims[2] / 2 if center_box else 0
        
        # Create XY grid
        x_edges = np.arange(0, box_x + xy_grid_size, xy_grid_size)
        y_edges = np.arange(0, box_y + xy_grid_size, xy_grid_size)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        print(f"XY grid: {len(x_centers)} × {len(y_centers)} points")
        
        # Initialize results
        results = {
            'z_slices': [],
            'z_slice_width': z_slice_width,
            'xy_grid_size': xy_grid_size,
            'mg_vdw_scaling': mg_vdw_scaling,
            'si_vdw_scaling': si_vdw_scaling,
            'mg_vdw_radius': mg_vdw_radius,
            'si_vdw_radius': si_vdw_radius,
            'combine_mg_layers': combine_mg_layers
        }
        
        # Process each z-slice
        for z_idx, z_center in enumerate(z_slice_centers):
            print(f"  Processing z-slice {z_idx+1}/{len(z_slice_centers)}: z = {z_center:.1f} Å")
            
            z_slice_data = {
                'z_center': z_center,
                'x_centers': x_centers,
                'y_centers': y_centers,
                'x_edges': x_edges,
                'y_edges': y_edges,
                'mg_grid': np.zeros((len(y_centers), len(x_centers))),
                'si_grid': np.zeros((len(y_centers), len(x_centers))),
                'combined_grid': np.zeros((len(y_centers), len(x_centers)))
            }
            
            # Process frames for this z-slice
            mg_positions_all = []
            si_positions_all = []
            
            for frame_idx in frame_indices:
                self.u.trajectory[frame_idx]
                
                # Get Mg positions in this z-slice
                if len(mg_atoms) > 0:
                    mg_positions = mg_atoms.positions.copy()
                    if center_box:
                        mg_positions[:, 2] -= box_center_z
                    
                    if combine_mg_layers:
                        # Include ALL Mg atoms regardless of z-position
                        mg_in_slice = mg_positions
                    else:
                        # Filter by z-slice (layer separation)
                        z_mask = np.abs(mg_positions[:, 2] - z_center) <= (z_slice_width / 2)
                        mg_in_slice = mg_positions[z_mask]
                    
                    if len(mg_in_slice) > 0:
                        # Apply PBC
                        mg_in_slice[:, 0] = mg_in_slice[:, 0] % box_x
                        mg_in_slice[:, 1] = mg_in_slice[:, 1] % box_y
                        mg_positions_all.extend(mg_in_slice[:, :2])  # Only XY
                
                # Get Si positions in this z-slice
                if len(si_atoms) > 0:
                    si_positions = si_atoms.positions.copy()
                    if center_box:
                        si_positions[:, 2] -= box_center_z
                    
                    # Filter by z-slice
                    z_mask = np.abs(si_positions[:, 2] - z_center) <= (z_slice_width / 2)
                    si_in_slice = si_positions[z_mask]
                    
                    if len(si_in_slice) > 0:
                        # Apply PBC
                        si_in_slice[:, 0] = si_in_slice[:, 0] % box_x
                        si_in_slice[:, 1] = si_in_slice[:, 1] % box_y
                        si_positions_all.extend(si_in_slice[:, :2])  # Only XY
            
            # Create 2D histograms
            if mg_positions_all:
                mg_array = np.array(mg_positions_all)
                mg_hist, _, _ = np.histogram2d(mg_array[:, 1], mg_array[:, 0], 
                                            bins=[y_edges, x_edges])
                z_slice_data['mg_grid'] = mg_hist
                
                # Apply Gaussian smoothing based on VdW radius
                if scaled_mg_radius > 0:
                    from scipy.ndimage import gaussian_filter
                    sigma = scaled_mg_radius / xy_grid_size  # Convert to grid units
                    z_slice_data['mg_grid'] = gaussian_filter(z_slice_data['mg_grid'], sigma=sigma)
            
            if si_positions_all:
                si_array = np.array(si_positions_all)
                si_hist, _, _ = np.histogram2d(si_array[:, 1], si_array[:, 0],
                                            bins=[y_edges, x_edges])
                z_slice_data['si_grid'] = si_hist
                
                # Apply Gaussian smoothing based on VdW radius
                if scaled_si_radius > 0:
                    from scipy.ndimage import gaussian_filter
                    sigma = scaled_si_radius / xy_grid_size  # Convert to grid units
                    z_slice_data['si_grid'] = gaussian_filter(z_slice_data['si_grid'], sigma=sigma)
            
            # Create combined grid
            z_slice_data['combined_grid'] = z_slice_data['mg_grid'] * 2.0 + z_slice_data['si_grid']
            
            results['z_slices'].append(z_slice_data)
            
            print(f"    ✓ Mg grid max: {np.max(z_slice_data['mg_grid']):.2f}")
            print(f"    ✓ Si grid max: {np.max(z_slice_data['si_grid']):.2f}")
        
        print(f"✅ Clay spatial distribution complete")
        
        # Save cache if requested
        if save_cache and cache_file:
            print(f"💾 Saving cache to {cache_file}")
            try:
                cache_dict = {
                    'z_slice_width': z_slice_width,
                    'xy_grid_size': xy_grid_size,
                    'mg_vdw_scaling': mg_vdw_scaling,
                    'si_vdw_scaling': si_vdw_scaling,
                    'mg_vdw_radius': mg_vdw_radius,
                    'si_vdw_radius': si_vdw_radius,
                    'combine_mg_layers': combine_mg_layers,
                    'n_slices': len(results['z_slices'])
                }
                
                # Save each z-slice data
                for i, z_slice_data in enumerate(results['z_slices']):
                    cache_dict[f'z{i}_center'] = z_slice_data['z_center']
                    cache_dict[f'z{i}_x_centers'] = z_slice_data['x_centers']
                    cache_dict[f'z{i}_y_centers'] = z_slice_data['y_centers']
                    cache_dict[f'z{i}_x_edges'] = z_slice_data['x_edges']
                    cache_dict[f'z{i}_y_edges'] = z_slice_data['y_edges']
                    cache_dict[f'z{i}_mg_grid'] = z_slice_data['mg_grid']
                    cache_dict[f'z{i}_si_grid'] = z_slice_data['si_grid']
                    cache_dict[f'z{i}_combined_grid'] = z_slice_data['combined_grid']
                
                np.savez(cache_file, **cache_dict)
                print(f"   ✓ Cache saved successfully")
                
            except Exception as e:
                print(f"   ⚠️ Cache saving failed: {e}")
        
        return results
    
    def calculate_clay_interface_boundaries(self, n_frames_average=50):
        """
        Calculate clay interface boundaries with configurable frame averaging for cleaved clay systems.
        
        Analyzes Si and Mg atom positions across multiple frames to determine the average z-positions
        of the upper and lower clay surfaces. This is useful for understanding the clay structure
        and identifying where ions should be located relative to the clay surfaces.
        
        Parameters
        ----------
        n_frames_average : int, default=50
            Number of frames to average around frame 100 for boundary calculation.
            More frames provide more stable statistics but increase computation time.
        
        Returns
        -------
        dict or None
            Dictionary containing clay interface boundaries with keys:
            - 'system_type': Type of clay system (e.g., 'cleaved_clay')
            - 'clay_average_z_positive': Average z-position of upper clay surface (Å)
            - 'clay_average_z_negative': Average z-position of lower clay surface (Å)
            - 'si_average_z_positive': Average z-position of upper Si layer (Å)
            - 'si_average_z_negative': Average z-position of lower Si layer (Å)
            - 'mgo_average_z_positive': Average z-position of upper Mg layer (Å)
            - 'mgo_average_z_negative': Average z-position of lower Mg layer (Å)
            - 'mg_distribution': Mg distribution pattern ('split', 'upper_only', 'lower_only', 'none_found')
            - 'n_si_upper', 'n_si_lower': Number of Si atoms in each region
            - 'n_mg_upper', 'n_mg_lower': Number of Mg atoms in each region
            - 'frames_averaged': Actual number of frames used
            Returns None if clay detection fails.
        
        Examples
        --------
        >>> boundaries = analysis.calculate_clay_interface_boundaries(n_frames_average=100)
        >>> print(f"Upper clay surface at z = {boundaries['clay_average_z_positive']:.2f} Å")
        >>> print(f"Lower clay surface at z = {boundaries['clay_average_z_negative']:.2f} Å")
        """
        
        print("\n🧱 CALCULATING CLAY INTERFACE BOUNDARIES...")
        print("="*50)
        
        # Determine centering setting from cavity results if available
        center_box = False
        if 'cavity_ion_binding' in self.results:
            cavity_results = self.results['cavity_ion_binding']
            # Check metadata first (standard location)
            if 'metadata' in cavity_results:
                center_box = cavity_results['metadata'].get('center_box', False)
                print(f"📊 Detected center_box={center_box} from cavity analysis metadata")
            # Fallback to analysis_parameters
            elif 'analysis_parameters' in cavity_results:
                center_box = cavity_results['analysis_parameters'].get('center_box', False)
                print(f"📊 Detected center_box={center_box} from cavity analysis parameters")
            # Last resort: top-level
            elif 'center_box' in cavity_results:
                center_box = cavity_results.get('center_box', False)
                print(f"📊 Detected center_box={center_box} from cavity analysis results")
        
        # Fallback: check if it's an instance attribute
        if not center_box and hasattr(self, 'center_box'):
            center_box = getattr(self, 'center_box', False)
            print(f"📊 Using center_box={center_box} from instance attribute")
        
        if not center_box:
            print(f"⚠️  WARNING: center_box not detected from previous analysis")
            print(f"   If your cavity analysis used center_box=True, clay boundaries may be incorrect")
        
        print(f"   Box centering: {'ENABLED (z=0 at box center)' if center_box else 'DISABLED (original coordinates)'}")
        
        original_frame = self.u.trajectory.ts.frame
        
        try:
            # ✅ COLLECT BOTH SI AND MG POSITIONS FROM MULTIPLE FRAMES
            all_si_positions = []
            all_mg_positions = []
            
            # Use frames around frame 100 (centered on frame 100)
            center_frame = min(100, len(self.u.trajectory) - 1)
            start_frame = max(0, center_frame - n_frames_average//2)
            end_frame = min(len(self.u.trajectory), center_frame + n_frames_average//2)
            
            # Ensure we have the requested number of frames (or close to it)
            actual_frames = end_frame - start_frame
            
            print(f"📊 Frame averaging configuration:")
            print(f"   Requested frames: {n_frames_average}")
            print(f"   Center frame: {center_frame}")
            print(f"   Frame range: {start_frame} to {end_frame-1}")
            print(f"   Actual frames: {actual_frames}")
            
            for frame_idx in range(start_frame, end_frame):
                self.u.trajectory[frame_idx]
                
                # Get Si atoms (always split between upper/lower in cleaved systems)
                si_atoms = self.u.select_atoms("resname MMT and (name Si or name SI or name Sio or name SIO)")
                
                # Get Mg atoms (variable distribution between frames)
                mg_atoms = self.u.select_atoms("resname MMT and (name Mgo or name MGO or name Mg or name MG)")
                
                if len(si_atoms) > 0:
                    si_z = si_atoms.positions[:, 2].copy()
                    
                    # Apply centering if enabled
                    if center_box:
                        si_z -= self.u.dimensions[2] / 2.0
                    
                    all_si_positions.extend(si_z)
                
                if len(mg_atoms) > 0:
                    mg_z = mg_atoms.positions[:, 2].copy()
                    
                    # Apply centering if enabled
                    if center_box:
                        mg_z -= self.u.dimensions[2] / 2.0
                    
                    all_mg_positions.extend(mg_z)
            
            # Convert to arrays
            si_positions = np.array(all_si_positions) if all_si_positions else np.array([])
            mg_positions = np.array(all_mg_positions) if all_mg_positions else np.array([])
            
            # ✅ CHECK FOR CLAY DETECTION FAILURE
            if len(si_positions) == 0 and len(mg_positions) == 0:
                print("❌ CLAY DETECTION FAILED: No Si or Mg atoms found")
                print("   Checked atom selections:")
                print("   - Si: 'resname MMT and (name Si or name SI or name Sio or name SIO)'")
                print("   - Mg: 'resname MMT and (name Mgo or name MGO or name Mg or name MG)'")
                return None
            
            print(f"✅ Found clay atoms:")
            print(f"   Si positions analyzed: {len(si_positions)} (from {actual_frames} frames)")
            print(f"   Mg positions analyzed: {len(mg_positions)} (from {actual_frames} frames)")
            
            # ✅ ANALYZE SI ATOMS (PRIMARY REFERENCE - ALWAYS SPLIT)
            si_upper = si_positions[si_positions >= 0] if len(si_positions) > 0 else np.array([])
            si_lower = si_positions[si_positions < 0] if len(si_positions) > 0 else np.array([])
            
            # ✅ ANALYZE MG ATOMS (VARIABLE DISTRIBUTION)
            mg_upper = mg_positions[mg_positions >= 0] if len(mg_positions) > 0 else np.array([])
            mg_lower = mg_positions[mg_positions < 0] if len(mg_positions) > 0 else np.array([])
            
            # Print distribution analysis
            print(f"\n📊 Cleaved clay distribution analysis:")
            print(f"   Si atoms (always split):")
            print(f"     Upper region: {len(si_upper)} atoms")
            print(f"     Lower region: {len(si_lower)} atoms")
            print(f"   Mg atoms (variable):")
            print(f"     Upper region: {len(mg_upper)} atoms")
            print(f"     Lower region: {len(mg_lower)} atoms")
            
            # ✅ ADDITIONAL VALIDATION: Check if we have reasonable clay distribution
            if len(si_positions) > 0:
                # For cleaved clay, we should have Si in both regions
                if len(si_upper) == 0 or len(si_lower) == 0:
                    print("⚠️  WARNING: Si atoms found only in one region")
                    print("   This might indicate:")
                    print("   1. Non-cleaved clay system")
                    print("   2. Unusual clay orientation")
                    print("   3. Insufficient sampling frames")
            
            # ✅ CALCULATE BOUNDARIES USING KEYS EXPECTED BY PLOTTING METHOD
            boundaries = {
                'system_type': 'cleaved_clay',
                'si_distribution': 'always_split',
                'mg_distribution': 'variable'
            }
            
            # Overall clay boundaries (using Si as primary reference)
            boundaries['clay_average_z_positive'] = np.mean(si_upper) if len(si_upper) > 0 else None
            boundaries['clay_average_z_negative'] = np.mean(si_lower) if len(si_lower) > 0 else None
            
            # Si layer boundaries (same as clay for cleaved systems)
            boundaries['si_average_z_positive'] = boundaries['clay_average_z_positive']
            boundaries['si_average_z_negative'] = boundaries['clay_average_z_negative']
            
            # Mg layer boundaries (supplementary information)
            boundaries['mgo_average_z_positive'] = np.mean(mg_upper) if len(mg_upper) > 0 else None
            boundaries['mgo_average_z_negative'] = np.mean(mg_lower) if len(mg_lower) > 0 else None
            
            # Additional boundaries for compatibility with plotting method
            boundaries['upper_clay_avg'] = boundaries['clay_average_z_positive']  # Alias
            boundaries['lower_clay_avg'] = boundaries['clay_average_z_negative']  # Alias
            boundaries['upper_si_above'] = boundaries['si_average_z_positive']    # Alias
            boundaries['lower_si_below'] = boundaries['si_average_z_negative']    # Alias
            boundaries['upper_mgo'] = boundaries['mgo_average_z_positive']        # Alias
            boundaries['lower_mgo'] = boundaries['mgo_average_z_negative']        # Alias
            
            # Min/max boundaries for each region
            boundaries['clay_min_z_positive'] = np.min(si_upper) if len(si_upper) > 0 else None
            boundaries['clay_max_z_positive'] = np.max(si_upper) if len(si_upper) > 0 else None
            boundaries['clay_min_z_negative'] = np.min(si_lower) if len(si_lower) > 0 else None
            boundaries['clay_max_z_negative'] = np.max(si_lower) if len(si_lower) > 0 else None
            
            # Mg distribution characterization
            if len(mg_upper) > 0 and len(mg_lower) > 0:
                boundaries['mg_distribution'] = 'split'
            elif len(mg_upper) > 0:
                boundaries['mg_distribution'] = 'upper_only'
            elif len(mg_lower) > 0:
                boundaries['mg_distribution'] = 'lower_only'
            else:
                boundaries['mg_distribution'] = 'none_found'
            
            # Store additional statistics
            boundaries['n_si_upper'] = len(si_upper)
            boundaries['n_si_lower'] = len(si_lower)
            boundaries['n_mg_upper'] = len(mg_upper)
            boundaries['n_mg_lower'] = len(mg_lower)
            boundaries['n_si_atoms'] = len(si_positions)
            boundaries['n_mgo_atoms'] = len(mg_positions)
            boundaries['n_total_clay_atoms'] = len(si_positions) + len(mg_positions)
            boundaries['frames_averaged'] = actual_frames
            boundaries['n_frames_requested'] = n_frames_average
            boundaries['center_frame'] = center_frame
            
            # Store in results
            self.results['clay_interface_boundaries'] = boundaries
            
            # Print summary with proper handling of None values
            print(f"\n📊 CLAY INTERFACE BOUNDARIES SUMMARY:")
            print(f"   System type: Cleaved clay (Si always split)")
            print(f"   Upper region (z > 0):")
            print(f"     Si layer average:     {boundaries['si_average_z_positive']:.2f} Å" if boundaries['si_average_z_positive'] is not None else "     Si layer average:     No Si atoms found")
            print(f"     Mg layer average:     {boundaries['mgo_average_z_positive']:.2f} Å" if boundaries['mgo_average_z_positive'] is not None else "     Mg layer average:     No Mg atoms found")
            print(f"     Overall clay average: {boundaries['clay_average_z_positive']:.2f} Å" if boundaries['clay_average_z_positive'] is not None else "     Overall clay average: No atoms found")
            print(f"   Lower region (z < 0):")
            print(f"     Si layer average:     {boundaries['si_average_z_negative']:.2f} Å" if boundaries['si_average_z_negative'] is not None else "     Si layer average:     No Si atoms found")
            print(f"     Mg layer average:     {boundaries['mgo_average_z_negative']:.2f} Å" if boundaries['mgo_average_z_negative'] is not None else "     Mg layer average:     No Mg atoms found")
            print(f"     Overall clay average: {boundaries['clay_average_z_negative']:.2f} Å" if boundaries['clay_average_z_negative'] is not None else "     Overall clay average: No atoms found")
            print(f"   Mg distribution: {boundaries['mg_distribution']}")
            print(f"   Frame averaging: {actual_frames} frames (requested: {n_frames_average})")
            print(f"✅ Clay interface boundaries calculation completed!")
            
            return boundaries
            
        except Exception as e:
            print(f"❌ CLAY DETECTION FAILED: {e}")
            print("   Error occurred during clay boundary calculation")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Restore original frame
            self.u.trajectory[original_frame]
