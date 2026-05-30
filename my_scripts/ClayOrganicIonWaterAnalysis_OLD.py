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
                                       force_rerun=False, step=1, store_per_atom_organics=False):
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
            
            # Separate cache for organics
            if organic_parts:
                org_str = '-'.join(organic_parts.keys())
                org_param_str = f"comp_ads_org_{total_target_atoms}_{target_str}_{org_str}_s{step}"
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
                                    if data and 'mean' in list(data.values())[0] if len(data) > 0 else False:
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
        
        # Process trajectory
        print(f"\n  Processing trajectory (step={step})...")
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
                    # Check if this is raw list data (newly calculated) or dict (cached)
                    data = results['organics'][org_name][target_name][label]
                    if isinstance(data, list):
                        time_series = np.array(data)
                        results['organics'][org_name][target_name][label] = {
                            'mean': np.mean(time_series),
                            'std': np.std(time_series),
                            'time_series': time_series
                        }
                    # If it's already a dict, it was loaded from cache - skip
        
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
            # Collect all data for table
            org_data = []
            for org_name in sorted(results['organics'].keys()):
                for target_name in sorted(results['organics'][org_name].keys()):
                    row = [org_name, target_name]
                    for label in distance_ranges.keys():
                        mean_val = results['organics'][org_name][target_name][label]['mean']
                        std_val = results['organics'][org_name][target_name][label]['std']
                        row.append(f"{mean_val:.2f} ± {std_val:.2f}")
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
        
        self.results['competitive_adsorption'] = results
        return results
    
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