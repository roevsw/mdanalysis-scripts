
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib
import os
import math
import re

import MDAnalysis as mda
from MDAnalysis.analysis import distances, contacts
from MDAnalysis.analysis.base import Results, AnalysisBase
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.analysis.leaflet import LeafletFinder

import multiprocessing
from multiprocessing import Pool
from functools import partial

from scipy.spatial import ConvexHull, distance_matrix, KDTree
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

import warnings
warnings.filterwarnings('ignore')

from utils.linear_algebra import *
from utils.file_rw import vdW_radii
from utils.ParallelMDAnalysis import ParallelInterRDF as InterRDF

class MolecularAnalysis:
    '''
    Comprehensive molecular dynamics analysis toolkit for organic molecules and complex systems with multiple ion types.
    
    This class handles:
    - Organic molecules (drugs, ligands, small molecules)
    - Protein-ligand interactions
    - Membrane systems
    - Polymer systems
    - Multi-component solutions
    - Multiple cation and anion types
    - Ion competition and selectivity
    
    Parameters
    ----------
    top : str
        Path to topology file (TPR, PSF, etc.)
    traj : str or list of str
        Path(s) to trajectory file(s) (XTC, TRR, DCD, etc.)
    solute_sel : str
        MDAnalysis selection for solute molecules, default='not resname SOL WAT'
    solvent_sel : str
        MDAnalysis selection for solvent, default='resname SOL WAT'
    cation_sel : str, optional
        MDAnalysis selection for cations, default='resname NA K MG CA'
    anion_sel : str, optional
        MDAnalysis selection for anions, default='resname CL SO4'
    center_method : str
        Method for defining molecular center: 'COM', 'COG', 'atom', default='COM'
    
    Examples
    --------
    Basic usage with multiple ions:
    
    >>> analysis = MolecularAnalysis('system.tpr', 'traj.xtc', 
    ...                              solute_sel='resname LIG', 
    ...                              solvent_sel='resname SOL',
    ...                              cation_sel='resname NA K MG',
    ...                              anion_sel='resname CL SO4')
    >>> ion_binding = analysis.ion_binding_analysis('resname LIG')
    >>> competition = analysis.ion_competition_analysis('resname LIG')
    '''

    def __init__(self, top, traj, solute_sel='not resname SOL WAT', solvent_sel='resname SOL WAT', 
                 cation_sel='resname NA K MG CA', anion_sel='resname CL SO4 PO4', center_method='COM'):
        
        self.universe = mda.Universe(top, traj)
        self.n_frames = len(self.universe.trajectory)
        
        # Core molecular selections
        self.solutes = self.universe.select_atoms(solute_sel)
        self.solvents = self.universe.select_atoms(solvent_sel)
        self.center_method = center_method
        
        # Ion selections - handle both single and multiple ion types
        try:
            self.cations = self.universe.select_atoms(cation_sel)
            self.cation_sel = cation_sel
        except:
            print(f"Warning: No cations found with selection: {cation_sel}")
            self.cations = mda.AtomGroup([], self.universe)
            self.cation_sel = ''
        
        try:
            self.anions = self.universe.select_atoms(anion_sel)
            self.anion_sel = anion_sel
        except:
            print(f"Warning: No anions found with selection: {anion_sel}")
            self.anions = mda.AtomGroup([], self.universe)
            self.anion_sel = ''
        
        # Store selections for reference
        self.solute_sel = solute_sel
        self.solvent_sel = solvent_sel
        
        # Get individual ion types
        self._identify_ion_types()
        
        # Validation
        if len(self.solutes) == 0:
            raise ValueError(f'No solutes found with selection: {solute_sel}')
        if len(self.solvents) == 0:
            print(f'Warning: No solvents found with selection: {solvent_sel}')
        
        # Initialize data containers
        self.rdfs = {}
        self.contacts_data = {}
        self.solvation_shells = {}
        self.ion_binding_data = {}
        
        # Custom atom selections storage
        self.custom_selections = {}
        
        # Get vdW radii
        self.vdW_radii = vdW_radii().get_dict()
        
        print(f"MolecularAnalysis initialized:")
        print(f"  Solutes: {len(self.solutes)} atoms")
        print(f"  Solvents: {len(self.solvents)} atoms")
        print(f"  Cations: {len(self.cations)} atoms ({len(self.cation_types)} types)")
        print(f"  Anions: {len(self.anions)} atoms ({len(self.anion_types)} types)")
        print(f"  Trajectory: {self.n_frames} frames")

    def _identify_ion_types(self):
        '''Identify individual ion types present in the system'''
        
        # Get unique cation types
        if len(self.cations) > 0:
            cation_resnames = np.unique(self.cations.resnames)
            self.cation_types = {}
            for resname in cation_resnames:
                atoms = self.cations.select_atoms(f'resname {resname}')
                self.cation_types[resname] = atoms
                print(f"    Cation {resname}: {len(atoms)} atoms")
        else:
            self.cation_types = {}
        
        # Get unique anion types
        if len(self.anions) > 0:
            anion_resnames = np.unique(self.anions.resnames)
            self.anion_types = {}
            for resname in anion_resnames:
                atoms = self.anions.select_atoms(f'resname {resname}')
                self.anion_types[resname] = atoms
                print(f"    Anion {resname}: {len(atoms)} atoms")
        else:
            self.anion_types = {}

    def _extract_label_from_selection(self, selection_string):
        '''
        Extract a concise label from a selection string for RDF labeling.
        Checks custom_selections first to use user-defined names, but prefers
        short atom/residue names for water selections.
        
        Parameters
        ----------
        selection_string : str
            MDAnalysis selection string (e.g., 'name NA', 'resname SOL')
        
        Returns
        -------
        label : str
            Extracted label (e.g., 'NA', 'SOL', 'quinolone', 'Ow')
        '''
        # Try regex patterns first for concise names
        # This gives priority to short names like 'Ow', 'NA', etc.
        # Use word boundaries \b to avoid matching 'name' in 'resname'
        
        import re
        
        # Special case for water oxygen: always use lowercase 'Ow' for consistency
        # This prevents key mismatches between 'Ow' and 'OW' in boundaries
        if 'name Ow' in selection_string or 'name OW' in selection_string:
            return 'Ow'
        
        patterns = [
            (r'\bname\s+(\S+)', 1),         # 'name NA' -> 'NA', 'name Ow' -> 'Ow'
            (r'\btype\s+(\S+)', 1),         # 'type OW' -> 'OW'
            (r'\bresname\s+(\S+)', 1),      # 'resname SOL' -> 'SOL'
            (r'\bresid\s+(\d+)', 1),        # 'resid 42' -> '42'
        ]
        
        for pattern, group_idx in patterns:
            match = re.search(pattern, selection_string, re.IGNORECASE)
            if match:
                extracted = match.group(group_idx)
                # Return short name if it's simple (not a complex moiety selection)
                if not ' and ' in selection_string or any(x in selection_string.lower() for x in ['name ow', 'name na', 'name k', 'name cl']):
                    return extracted
        
        # Check custom_selections for moiety names (quinolone, piperazine, etc.)
        # This allows user-defined names for complex selections
        if hasattr(self, 'custom_selections') and self.custom_selections:
            for category, selections in self.custom_selections.items():
                for name, sel_string in selections.items():
                    if sel_string == selection_string:
                        return name
        
        # If no pattern matches, use a shortened version
        # Remove common words and return first meaningful part
        cleaned = selection_string.replace('and', '').replace('or', '').strip()
        parts = cleaned.split()
        if len(parts) > 0:
            return parts[-1][:20]  # Last part, max 20 chars
        
        return selection_string[:20]  # Fallback

    def molecular_rdf(self, group1_sel, group2_sel, bin_width=0.05, range=(0, 15), 
                     step=1, njobs=1, center_method=None, normalize=True,
                     save_cache=True, cache_file=None, force_rerun=False):
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
        
        Returns
        -------
        results : object or dict
            If both selections are strings: Returns RDF results object with 
            .bins, .rdf, .count, .edges attributes (backward compatible).
            
            If one or both selections are lists: Returns dictionary with format
            {label: rdf_results} where labels are extracted from selection strings.
            
        Examples
        --------
        >>> # Single RDF (backward compatible)
        >>> rdf = analysis.molecular_rdf('name NA', 'resname SOL')
        >>> print(rdf.bins, rdf.rdf)
        
        >>> # Multiple group2 selections
        >>> rdfs = analysis.molecular_rdf('resname LIG', ['name NA', 'name K', 'name CL'])
        >>> # Returns: {'NA': rdf1, 'K': rdf2, 'CL': rdf3}
        
        >>> # Multiple group1 selections
        >>> rdfs = analysis.molecular_rdf(['resname LIG', 'resname PROT'], 'resname SOL')
        >>> # Returns: {'LIG': rdf1, 'PROT': rdf2}
        
        >>> # Both as lists (all combinations)
        >>> rdfs = analysis.molecular_rdf(['resname LIG1', 'resname LIG2'], 
        ...                                ['name NA', 'name K'])
        >>> # Returns: {'LIG1-NA': rdf1, 'LIG1-K': rdf2, 'LIG2-NA': rdf3, 'LIG2-K': rdf4}
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
            current_rdf = 0
            
            # Calculate all combinations
            for g1_sel in group1_list:
                g1_label = self._extract_label_from_selection(g1_sel)
                
                for g2_sel in group2_list:
                    g2_label = self._extract_label_from_selection(g2_sel)
                    
                    # Create combined label - always use both parts for proper coordination detection
                    label = f"{g1_label}-{g2_label}"
                    
                    current_rdf += 1
                    print(f"\n[{current_rdf}/{total_rdfs}] Calculating RDF: {label}")
                    print(f"  Group 1: {g1_sel}")
                    print(f"  Group 2: {g2_sel}")
                    
                    # Recursively call this function with single strings
                    # This ensures caching still works for each individual RDF
                    rdf_result = self.molecular_rdf(
                        g1_sel, g2_sel,
                        bin_width=bin_width,
                        range=range,
                        step=step,
                        njobs=njobs,
                        center_method=center_method,
                        normalize=normalize,
                        save_cache=save_cache,
                        cache_file=None,  # Auto-generate cache for each
                        force_rerun=force_rerun
                    )
                    
                    results_dict[label] = rdf_result
            
            print(f"\n✅ Batch RDF calculation complete! Generated {len(results_dict)} RDFs")
            print(f"   Labels: {list(results_dict.keys())}")
            return results_dict
        
        # Original single RDF calculation (backward compatible)
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

    def ion_binding_analysis(self, target_sel, cutoff=3.5, step=1, ion_types=None,
                            save_cache=True, cache_file=None, force_rerun=False,
                            rdf_boundaries=None, peaks=None, fallback_cutoff=3.5,
                            # Volume normalization parameters
                            calculate_volumes=True, volume_units='angstrom3', 
                            geometry_type='auto'):
        '''
        Comprehensive analysis of ion binding to target molecules.
        Supports lists for batch binding analysis with caching.
        Enhanced with RDF boundary support for peak-specific analysis.
        
        Parameters
        ----------
        target_sel : str or list of str
            Selection(s) for target molecules (e.g., 'resname LIG', 'protein')
            If list, calculates binding for each target separately
        cutoff : float
            Distance cutoff for ion binding, default=3.5 Å
            Used as fallback when rdf_boundaries not available
        step : int
            Trajectory step, default=1
        ion_types : list or None, optional
            List of specific ion types to analyze (e.g., ['K', 'NA', 'CL']).
            If None, analyzes all available ion types. Ion names should match
            residue names (case-sensitive).
        save_cache : bool
            Whether to save results to cache file (default: True)
        cache_file : str or None
            Custom cache filename. If None, auto-generates from parameters.
            Ignored when using lists (each analysis gets its own cache).
        force_rerun : bool
            Force recalculation even if cache exists (default: False)
        rdf_boundaries : dict or None, optional
            RDF boundaries from interactive_rdf_boundary_editor for peak-specific analysis.
            Format: {rdf_label: {shell_name: (start, end), ...}}
            If provided, will use refined boundaries instead of fixed cutoff.
        peaks : dict or None, optional
            Peak selection for each ion-moiety pair.
            Format: {'piperazine-NA': ['P1', 'P2'], 'quinolone-K': ['P1'], ...}
            If None and rdf_boundaries provided, defaults to ['P1'] for all pairs.
        fallback_cutoff : float
            Cutoff to use when rdf_boundaries not available for a specific pair.
            Default=3.5 Å
        calculate_volumes : bool
            Whether to calculate region volumes and binding densities (default: True).
            Enables volume-normalized analysis for peak regions.
        volume_units : str
            Units for volume calculations (default: 'angstrom3'):
            - 'angstrom3': Ångström³ (Å³)
            - 'nm3': nanometer³ (nm³)
        geometry_type : str
            Geometry type for volume calculations (default: 'auto'):
            - 'auto': Auto-detect based on boundaries (spherical for radial)
            - 'spherical': Spherical shells and spheres
            - 'spherical_shell': Spherical shell (annular region)
            - 'sphere': Complete sphere from center
            
        Returns
        -------
        binding_results : dict
            If target_sel is string: Returns binding analysis results dict
            If target_sel is list: Returns {label: binding_results} dict
            
            Structure maintained for compatibility with plot_ion_binding_comparison():
            {
                'cation_binding': {ion_name: {
                    'binding_events': array,
                    'average_binding': float,
                    'peak_analysis': {peak_name: {...}} if using RDF boundaries
                }},
                'anion_binding': {similar structure},
                'total_binding': {'cations': [], 'anions': []},
                'binding_sites': {},
                'selectivity': {}
            }
        
        Examples
        --------
        >>> # Traditional analysis (backward compatible)
        >>> results = analysis.ion_binding_analysis('resname api', cutoff=3.5)
        
        >>> # RDF boundary-based analysis
        >>> results = analysis.ion_binding_analysis(
        ...     target_sel=[quinolone, piperazine],
        ...     rdf_boundaries=boundaries_ions_refined,
        ...     peaks={
        ...         'piperazine-NA': ['P1', 'P2'],
        ...         'piperazine-K': ['P1'],
        ...         'quinolone-NA': ['P1', 'P2']
        ...     }
        ... )
        
        >>> # Mixed approach with fallback
        >>> results = analysis.ion_binding_analysis(
        ...     target_sel='resname api',
        ...     rdf_boundaries=boundaries_ions_refined,
        ...     fallback_cutoff=4.0  # Used when no RDF boundary found
        ... )
        '''
        
        import hashlib
        import os
        
        # Check if target_sel is a list
        if isinstance(target_sel, list):
            print("🔄 Batch ion binding analysis mode activated")
            
            results_dict = {}
            targets_to_calculate = []
            
            # Check each target for existing individual caches
            if save_cache and not force_rerun:
                print("📂 Checking for existing individual target caches...")
                
                for sel in target_sel:
                    label = self._extract_label_from_selection(sel)
                    
                    # Generate cache filename for this specific target
                    ion_types_str = '_'.join(sorted(ion_types)) if ion_types else 'all'
                    param_str = f"{sel}_c{cutoff}_s{step}_ions{ion_types_str}"
                    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
                    target_cache_file = f"binding_cache_{param_hash}.npz"
                    
                    # Check if this target has a valid cache
                    if os.path.exists(target_cache_file):
                        try:
                            print(f"   ✅ Found cache for {label}: {target_cache_file}")
                            cached_data = np.load(target_cache_file, allow_pickle=True)
                            
                            # Reconstruct results dictionary for this target
                            target_result = {
                                'cation_binding': cached_data['cation_binding'].item(),
                                'anion_binding': cached_data['anion_binding'].item(),
                                'total_binding': cached_data['total_binding'].item(),
                                'binding_sites': cached_data['binding_sites'].item(),
                                'selectivity': cached_data['selectivity'].item()
                            }
                            
                            results_dict[label] = target_result
                            print(f"   📋 Loaded cached results for {label}")
                            
                        except Exception as e:
                            print(f"   ⚠️ Failed to load cache for {label}: {e}")
                            targets_to_calculate.append(sel)
                    else:
                        print(f"   📋 No cache found for {label}")
                        targets_to_calculate.append(sel)
                
                # Summary of cache status
                cached_count = len(results_dict)
                missing_count = len(targets_to_calculate)
                print(f"\n📊 Cache Status:")
                print(f"   ✅ Cached targets: {cached_count}")
                print(f"   🔄 Missing targets: {missing_count}")
                if cached_count > 0:
                    print(f"   📋 Cached: {sorted(results_dict.keys())}")
                if missing_count > 0:
                    missing_labels = [self._extract_label_from_selection(sel) for sel in targets_to_calculate]
                    print(f"   📋 Missing: {sorted(missing_labels)}")
                    
            else:
                # No cache check or force rerun - calculate all targets  
                if force_rerun:
                    print("🔄 Force rerun enabled - calculating all targets")
                else:
                    print("💾 Caching disabled - calculating all targets")
                targets_to_calculate = target_sel
            
            # Calculate missing targets only
            if targets_to_calculate:
                total_to_calc = len(targets_to_calculate)
                print(f"\n🔬 Calculating {total_to_calc} target(s)...")
                
                for idx, sel in enumerate(targets_to_calculate, 1):
                    label = self._extract_label_from_selection(sel)
                    
                    print(f"\n[{idx}/{total_to_calc}] Analyzing ion binding: {label}")
                    print(f"  Target: {sel}")
                    
                    # Recursively call this function with single string
                    binding_result = self.ion_binding_analysis(
                        sel,
                        cutoff=cutoff,
                        step=step,
                        ion_types=ion_types,
                        save_cache=save_cache,  # Save individual cache for this target
                        cache_file=None,        # Auto-generate cache filename
                        force_rerun=force_rerun,
                        rdf_boundaries=rdf_boundaries,
                        peaks=peaks,
                        fallback_cutoff=fallback_cutoff
                    )
                    
                    results_dict[label] = binding_result
            
            print(f"\n✅ Batch binding analysis complete!")
            print(f"   📊 Total results: {len(results_dict)}")
            print(f"   📋 Labels: {sorted(results_dict.keys())}")
            return results_dict
        
        # Single target analysis (original logic)
        target = self.universe.select_atoms(target_sel)
        if len(target) == 0:
            raise ValueError(f"No target atoms found with selection: {target_sel}")
        
        # Generate cache filename if not provided
        if cache_file is None:
            # Create hash from parameters for unique cache filename
            ion_types_str = '_'.join(sorted(ion_types)) if ion_types else 'all'
            param_str = f"{target_sel}_c{cutoff}_s{step}_ions{ion_types_str}"
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            cache_file = f"binding_cache_{param_hash}.npz"
        
        # Check for existing cache
        if save_cache and not force_rerun and os.path.exists(cache_file):
            print(f"📂 Found existing binding cache: {cache_file}")
            try:
                print("   Loading cached binding results...")
                cached_data = np.load(cache_file, allow_pickle=True)
                
                # Reconstruct results dictionary
                results = {
                    'cation_binding': cached_data['cation_binding'].item(),
                    'anion_binding': cached_data['anion_binding'].item(),
                    'total_binding': cached_data['total_binding'].item(),
                    'binding_sites': cached_data['binding_sites'].item(),
                    'selectivity': cached_data['selectivity'].item()
                }
                
                print(f"   ✅ Loaded cached binding results successfully!")
                return results
                
            except Exception as e:
                print(f"   ⚠️ Failed to load cache: {e}")
                print(f"   Recalculating binding analysis...")
        
        # Filter ion types if specified
        if ion_types is not None:
            # Convert to uppercase for case-insensitive matching
            ion_types_upper = [ion.upper() for ion in ion_types]
            print(f"Analyzing ion binding to {len(target)} target atoms...")
            print(f"  Selected ions: {', '.join(ion_types)}")
        else:
            ion_types_upper = None
            print(f"Analyzing ion binding to {len(target)} target atoms...")
            print(f"  Analyzing all available ions")
        
        results = {
            'cation_binding': {},
            'anion_binding': {},
            'total_binding': {'cations': [], 'anions': []},
            'binding_sites': {},
            'selectivity': {}
        }
        
        # Analyze each cation type
        target_label = self._extract_label_from_selection(target_sel) if isinstance(target_sel, str) else 'target'
        
        for cation_name, cation_atoms in self.cation_types.items():
            # Skip if ion_types specified and this ion not in list
            if ion_types_upper is not None and cation_name.upper() not in ion_types_upper:
                continue
            
            print(f"  Analyzing {cation_name} binding...")
            
            # Check if RDF boundaries are available for this pair
            if rdf_boundaries is not None:
                rdf_label = self._find_rdf_boundary_label(target_label, cation_name, rdf_boundaries)
                if rdf_label is not None:
                    # Get peaks to analyze for this pair
                    pair_key = f"{target_label}-{cation_name}"
                    selected_peaks = peaks.get(pair_key, ['P1']) if peaks else ['P1']
                    
                    print(f"    Using RDF boundaries from '{rdf_label}' for peaks: {selected_peaks}")
                    binding_data = self._analyze_ion_binding_with_peaks(
                        target, cation_atoms, rdf_boundaries[rdf_label], selected_peaks, step,
                        calculate_volumes, volume_units, geometry_type
                    )
                else:
                    print(f"    No RDF boundaries found, using fallback cutoff: {fallback_cutoff} Å")
                    binding_data = self._analyze_ion_binding(target, cation_atoms, fallback_cutoff, step)
            else:
                # Traditional analysis
                binding_data = self._analyze_ion_binding(target, cation_atoms, cutoff, step)
            
            results['cation_binding'][cation_name] = binding_data
        
        # Analyze each anion type
        for anion_name, anion_atoms in self.anion_types.items():
            # Skip if ion_types specified and this ion not in list
            if ion_types_upper is not None and anion_name.upper() not in ion_types_upper:
                continue
            
            print(f"  Analyzing {anion_name} binding...")
            
            # Check if RDF boundaries are available for this pair
            if rdf_boundaries is not None:
                rdf_label = self._find_rdf_boundary_label(target_label, anion_name, rdf_boundaries)
                if rdf_label is not None:
                    # Get peaks to analyze for this pair
                    pair_key = f"{target_label}-{anion_name}"
                    selected_peaks = peaks.get(pair_key, ['P1']) if peaks else ['P1']
                    
                    print(f"    Using RDF boundaries from '{rdf_label}' for peaks: {selected_peaks}")
                    binding_data = self._analyze_ion_binding_with_peaks(
                        target, anion_atoms, rdf_boundaries[rdf_label], selected_peaks, step,
                        calculate_volumes, volume_units, geometry_type
                    )
                else:
                    print(f"    No RDF boundaries found, using fallback cutoff: {fallback_cutoff} Å")
                    binding_data = self._analyze_ion_binding(target, anion_atoms, fallback_cutoff, step)
            else:
                # Traditional analysis
                binding_data = self._analyze_ion_binding(target, anion_atoms, cutoff, step)
            
            results['anion_binding'][anion_name] = binding_data
        
        # Calculate total ion binding per frame
        # Use appropriate cutoff based on whether RDF boundaries are used
        effective_cutoff = fallback_cutoff if rdf_boundaries is not None else cutoff
        
        for ts in tqdm(self.universe.trajectory[::step], desc="Calculating total binding"):
            total_cations = 0
            total_anions = 0
            
            for cation_atoms in self.cation_types.values():
                if len(cation_atoms) > 0:
                    dist_matrix = distances.distance_array(target.positions, 
                                                         cation_atoms.positions,
                                                         box=ts.dimensions)
                    # Count unique ions that have at least one contact within cutoff
                    # dist_matrix shape: (n_target_atoms, n_ions)
                    # any(axis=0): for each ion, is it within cutoff of ANY target atom?
                    ions_bound = (dist_matrix <= effective_cutoff).any(axis=0)
                    total_cations += ions_bound.sum()  # Count bound ions, not contacts
            
            for anion_atoms in self.anion_types.values():
                if len(anion_atoms) > 0:
                    dist_matrix = distances.distance_array(target.positions, 
                                                         anion_atoms.positions,
                                                         box=ts.dimensions)
                    # Count unique ions that have at least one contact within cutoff
                    # dist_matrix shape: (n_target_atoms, n_ions)
                    # any(axis=0): for each ion, is it within cutoff of ANY target atom?
                    ions_bound = (dist_matrix <= effective_cutoff).any(axis=0)
                    total_anions += ions_bound.sum()  # Count bound ions, not contacts
            
            results['total_binding']['cations'].append(total_cations)
            results['total_binding']['anions'].append(total_anions)
        
        # Calculate selectivity indices
        results['selectivity'] = self._calculate_ion_selectivity(results)
        
        # Save to cache if requested
        if save_cache:
            print(f"\n💾 Saving binding analysis to cache: {cache_file}")
            try:
                np.savez(cache_file,
                        cation_binding=results['cation_binding'],
                        anion_binding=results['anion_binding'],
                        total_binding=results['total_binding'],
                        binding_sites=results['binding_sites'],
                        selectivity=results['selectivity'],
                        # Metadata
                        target_sel=target_sel,
                        cutoff=cutoff,
                        step=step,
                        ion_types=ion_types)
                print(f"   ✅ Cache saved successfully!")
                print(f"   📁 File: {cache_file}")
            except Exception as e:
                print(f"   ⚠️ Failed to save cache: {e}")
                print(f"   Continuing without cache...")
        
        self.ion_binding_data[target_sel] = results
        return results
    
    def spatial_binding_analysis(self, target_sel, ion_type=None, solvation=None, cutoff=3.5, step=1,
                                 method='per-atom', angular_bins=(18, 36),
                                 return_positions=False,
                                 # Shell-aware parameters
                                 rdf_data=None, rdf_boundaries=None, solvation_shells=None,
                                 peaks=None,
                                 # Molecular frame tracking parameters
                                 molecular_frame_tracking=True, reference_atoms='auto',
                                 molecular_frame_method='gram_schmidt', n_reference_atoms=None,
                                 reference_target=None,
                                 # Caching parameters
                                 save_cache=True, cache_file=None, force_rerun=False,
                                 # Label override for dict input
                                 target_label=None):
        """
        Analyze spatial distribution of ion binding around target molecule
        
        Calculates WHERE ions bind on the molecular surface using two approaches:
        1. per-atom: Contact frequency for each atom in target molecule
        2. spherical: Angular distribution in spherical coordinates (theta, phi)
        
        Can use either simple distance cutoff or shell-boundary based analysis.
        
        Parameters
        ----------
        target_sel : str, list of str, or dict
            Selection string(s) for target molecule (e.g., 'resname api')
            - str: Single target analysis
            - list: Batch analysis with auto-generated labels from selections
            - dict: Batch analysis with custom labels, format: {'label': 'selection', ...}
                    Example: {'quinolone': 'resname QUI', 'piperazine': 'resname PIP'}
        ion_type : str, optional
            Ion type to analyze (e.g., 'K', 'NA', 'CL')
            Case-insensitive. Use for ion coordination analysis.
        solvation : str, optional
            Solvent type to analyze (e.g., 'Ow' for water oxygen)
            Case-insensitive. Use for solvation analysis.
            Note: Exactly one of ion_type or solvation must be provided.
        cutoff : float
            Distance cutoff for contact definition (Å). Used only if shell data not provided.
        step : int
            Frame step for analysis
        method : str
            Analysis method: 'per-atom', 'spherical', or 'both'
        angular_bins : tuple
            (n_theta, n_phi) bins for spherical mapping
            theta: polar angle (0 to π), phi: azimuthal angle (0 to 2π)
        return_positions : bool
            If True, returns all ion positions relative to COM (for 3D visualization)
        rdf_data : dict, optional
            RDF calculation results from water_solvation_analysis or similar
        rdf_boundaries : dict, optional
            Shell boundary definitions (from peak analysis)
        solvation_shells : dict, optional
            Shell selection for analysis, format: {'target-solvent': ['shell_2', 'shell_3']}
            e.g., {'api-Ow': ['shell_2', 'shell_3']} to analyze water in shells 2&3
        peaks : dict, optional
            Peak selection for ion analysis, format: {'target-ion': ['P1', 'P2', ...]}
            e.g., {'quinolone-NA': ['P2', 'P3'], 'carboxylic_acid-K': ['P1', 'P2']}
            Only used for ion_type analysis with rdf_boundaries.
        angular_bins : tuple
            (n_theta, n_phi) bins for spherical mapping
            theta: polar angle (0 to π), phi: azimuthal angle (0 to 2π)
        return_positions : bool
            If True, returns all ion positions relative to COM (for 3D visualization)
        molecular_frame_tracking : bool
            If True (default), enables rotation-aware triangulation using molecular reference frames.
            Tracks molecular orientation changes to prevent ion-atom overlaps in reconstruction.
            When False, uses legacy COM-relative positioning (may have rotation artifacts).
        reference_atoms : str or list
            Method for selecting molecular reference atoms to define coordinate frame:
            - 'auto': Automatically select 3 non-collinear atoms with largest distances
            - 'heavy': Use heaviest atoms (non-hydrogen)
            - list of 3 atom indices: Manual specification of reference atoms
            Only used when molecular_frame_tracking=True.
        reference_target : str or None
            Specific atom selection string for reference frame establishment.
            If provided, reference atoms will be selected from this subset instead of
            the full target_sel. Useful when target_sel contains multiple molecular
            regions but you want the reference frame from a specific region.
            e.g., reference_target='resname QUI' to use only quinolone atoms
        save_cache : bool
            Whether to save results to cache file (default: True)
        cache_file : str or None
            Custom cache filename. If None, auto-generates from parameters.
        force_rerun : bool
            Force recalculation even if cache exists (default: False)
        
        Returns
        -------
        results : dict
            If target_sel is string: Returns spatial binding analysis results dict
            If target_sel is list: Returns {auto_label: spatial_results} dict
            If target_sel is dict: Returns {custom_label: spatial_results} dict
            
            Dictionary with keys depending on method:
            
            If method='per-atom':
                'contact_frequency': np.array of contact counts per atom
                'atom_indices': np.array of atom indices
                'atom_names': list of atom names
                'atom_positions': np.array of average atom positions
                
            If method='spherical':
                'angular_histogram': 2D array [n_theta, n_phi]
                'theta_bins': np.array of theta bin edges
                'phi_bins': np.array of phi bin edges
                'theta_centers': np.array of theta bin centers
                'phi_centers': np.array of phi bin centers
                
            If method='both':
                All keys from both methods
                
            Common keys:
                'total_contacts': int, total number of ion contacts
                'frames_analyzed': int
                
            If return_positions=True:
                'ion_positions_relative': list of arrays (ion positions relative to COM)
        
        Examples
        --------
        >>> # Ion spatial binding analysis
        >>> results = analysis.spatial_binding_analysis(
        ...     target_sel=[carboxylic_acid, piperazine, quinolone],
        ...     ion_type='Na',
        ...     rdf_boundaries=boundaries_sodium_refined,
        ...     peaks={
        ...         'quinolone-NA': ['P2', 'P3', 'P4'],
        ...         'carboxylic_acid-NA': ['P1', 'P2', 'P3'],
        ...         'piperazine-NA': ['P2', 'P3', 'P4']
        ...     },
        ...     method='both'
        ... )
        
        >>> # Water solvation analysis
        >>> results = analysis.spatial_binding_analysis(
        ...     target_sel=[carboxylic_acid, piperazine],
        ...     solvation='Ow',
        ...     rdf_boundaries=boundaries_refined,
        ...     solvation_shells={
        ...         'carboxylic_acid-Ow': ['shell_2', 'shell_3'],
        ...         'piperazine-Ow': ['shell_2', 'shell_3']
        ...     },
        ...     method='both'
        ... )
        
        >>> # Simple distance cutoff analysis
        >>> results = analysis.spatial_binding_analysis(
        ...     target_sel='resname api', ion_type='K', 
        ...     cutoff=3.5, method='per-atom')
        >>> print(results['contact_frequency'])  # Contacts per atom
        """
        
        # Validate inputs
        if method not in ['per-atom', 'spherical', 'both']:
            raise ValueError("method must be 'per-atom', 'spherical', or 'both'")
        
        # Validate analysis mode
        if ion_type is not None and solvation is not None:
            raise ValueError("Cannot specify both ion_type and solvation. Choose one analysis mode.")
        if ion_type is None and solvation is None:
            raise ValueError("Must specify either ion_type (for ion analysis) or solvation (for solvent analysis).")
        
        # Determine analysis mode
        analysis_mode = 'ion' if ion_type is not None else 'solvation'
        analysis_species = ion_type if ion_type is not None else solvation
        
        print(f"🔬 {analysis_mode.title()} spatial binding analysis mode")
        print(f"   Analyzing: {analysis_species}")
        
        # Check if target_sel is a dict - handle batch processing with custom labels
        if isinstance(target_sel, dict):
            print("🔄 Batch spatial binding analysis mode activated (dict with custom labels)")
            
            results_dict = {}
            targets_to_calculate = {}
            
            # Check each target for existing individual caches
            if save_cache and not force_rerun:
                print("📂 Checking for existing individual target caches...")
                
                for label, sel in target_sel.items():
                    # Generate cache filename for this specific target
                    shells_str = str(solvation_shells) if solvation_shells else 'None'
                    peaks_str = str(peaks) if peaks else 'None'
                    boundaries_str = str(bool(rdf_boundaries))
                    molecular_frame_str = f"mft{molecular_frame_tracking}_ra{reference_atoms}_rt{reference_target}"
                    param_str = f"{sel}_{analysis_mode}_{analysis_species}_c{cutoff}_s{step}_m{method}_ab{angular_bins}_rp{return_positions}_sh{shells_str}_pk{peaks_str}_b{boundaries_str}_{molecular_frame_str}"
                    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
                    target_cache_file = f"spatial_cache_{param_hash}.npz"
                    
                    # Check if this target has a valid cache
                    if os.path.exists(target_cache_file):
                        try:
                            print(f"   ✅ Loading cache for {label}")
                            cached_data = np.load(target_cache_file, allow_pickle=True)
                            
                            # Reconstruct results dictionary
                            target_results = {}
                            for key in cached_data.files:
                                if key.startswith('metadata_'):
                                    continue
                                data = cached_data[key]
                                try:
                                    if data.dtype == object:
                                        target_results[key] = data.item()
                                    elif data.ndim == 0:  # Scalar array
                                        target_results[key] = data.item()
                                    else:
                                        target_results[key] = data
                                except (ValueError, AttributeError):
                                    # Fallback for problematic data
                                    target_results[key] = data
                            
                            # Load position data if present and requested
                            pos_cache_file = target_cache_file.replace('.npz', '_positions.pkl')
                            if return_positions and os.path.exists(pos_cache_file):
                                try:
                                    import pickle
                                    with open(pos_cache_file, 'rb') as f:
                                        position_data = pickle.load(f)
                                    target_results['ion_positions_relative'] = position_data
                                    print(f"       Position data loaded: {len(position_data)} frames")
                                except Exception as e:
                                    print(f"       ⚠️ Failed to load position data: {e}")
                                    # Position data missing - force recalculation
                                    targets_to_calculate[label] = sel
                                    continue
                            elif return_positions:
                                print(f"       ⚠️ Position cache missing: {pos_cache_file}")
                                # Position data missing - force recalculation
                                targets_to_calculate[label] = sel
                                continue
                            
                            results_dict[label] = target_results
                            
                        except Exception as e:
                            print(f"   ⚠️ Cache loading failed for {label}: {e}")
                            targets_to_calculate[label] = sel
                    else:
                        print(f"   📋 No cache found for {label}")
                        targets_to_calculate[label] = sel
                
                # Summary of cache status
                cached_count = len(results_dict)
                missing_count = len(targets_to_calculate)
                print(f"\n📊 Cache Status:")
                print(f"   ✅ Cached targets: {cached_count}")
                print(f"   🔄 Missing targets: {missing_count}")
                if cached_count > 0:
                    print(f"   📋 Cached: {sorted(results_dict.keys())}")
                if missing_count > 0:
                    print(f"   📋 Missing: {sorted(targets_to_calculate.keys())}")
                    
            else:
                # No cache check or force rerun - calculate all targets  
                if force_rerun:
                    print("🔄 Force rerun enabled - calculating all targets")
                else:
                    print("💾 Caching disabled - calculating all targets")
                targets_to_calculate = target_sel
            
            # Calculate missing targets only
            if targets_to_calculate:
                total_to_calc = len(targets_to_calculate)
                print(f"\n🔬 Calculating {total_to_calc} target(s)...")
                
                for idx, (label, sel) in enumerate(targets_to_calculate.items(), 1):
                    print(f"\n[{idx}/{total_to_calc}] Analyzing spatial binding: {label}")
                    print(f"  Target: {sel}")
                    
                    # Recursively call this function with single string
                    # Use the current target as its own reference frame source
                    spatial_result = self.spatial_binding_analysis(
                        sel,
                        ion_type=ion_type,
                        solvation=solvation,
                        cutoff=cutoff,
                        step=step,
                        method=method,
                        angular_bins=angular_bins,
                        return_positions=return_positions,
                        rdf_data=rdf_data,
                        rdf_boundaries=rdf_boundaries,
                        solvation_shells=solvation_shells,
                        peaks=peaks,
                        molecular_frame_tracking=molecular_frame_tracking,
                        reference_atoms=reference_atoms,
                        reference_target=sel,  # Use current target as reference source
                        save_cache=save_cache,  # Save individual cache for this target
                        cache_file=None,        # Auto-generate cache filename
                        force_rerun=force_rerun,
                        target_label=label      # Pass the custom label for peak matching
                    )
                    
                    results_dict[label] = spatial_result
            
            print(f"\n✅ Batch spatial binding analysis complete!")
            print(f"   📊 Total results: {len(results_dict)}")
            print(f"   📋 Labels: {sorted(results_dict.keys())}")
            return results_dict
        
        # Check if target_sel is a list - handle batch processing
        elif isinstance(target_sel, list):
            print("🔄 Batch spatial binding analysis mode activated")
            
            results_dict = {}
            targets_to_calculate = []
            
            # Check each target for existing individual caches
            if save_cache and not force_rerun:
                print("📂 Checking for existing individual target caches...")
                
                for sel in target_sel:
                    label = self._extract_label_from_selection(sel)
                    
                    # Generate cache filename for this specific target
                    shells_str = str(solvation_shells) if solvation_shells else 'None'
                    peaks_str = str(peaks) if peaks else 'None'
                    boundaries_str = str(bool(rdf_boundaries))
                    molecular_frame_str = f"mft{molecular_frame_tracking}_ra{reference_atoms}_rt{reference_target}"
                    param_str = f"{sel}_{analysis_mode}_{analysis_species}_c{cutoff}_s{step}_m{method}_ab{angular_bins}_rp{return_positions}_sh{shells_str}_pk{peaks_str}_b{boundaries_str}_{molecular_frame_str}"
                    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
                    target_cache_file = f"spatial_cache_{param_hash}.npz"
                    
                    # Check if this target has a valid cache
                    if os.path.exists(target_cache_file):
                        try:
                            print(f"   ✅ Loading cache for {label}")
                            cached_data = np.load(target_cache_file, allow_pickle=True)
                            
                            # Reconstruct results dictionary
                            target_results = {}
                            for key in cached_data.files:
                                if key.startswith('metadata_'):
                                    continue
                                data = cached_data[key]
                                try:
                                    if data.dtype == object:
                                        target_results[key] = data.item()
                                    elif data.ndim == 0:  # Scalar array
                                        target_results[key] = data.item()
                                    else:
                                        target_results[key] = data
                                except (ValueError, AttributeError):
                                    # Fallback for problematic data
                                    target_results[key] = data
                            
                            # Load position data if present and requested
                            pos_cache_file = target_cache_file.replace('.npz', '_positions.pkl')
                            if return_positions and os.path.exists(pos_cache_file):
                                try:
                                    import pickle
                                    with open(pos_cache_file, 'rb') as f:
                                        position_data = pickle.load(f)
                                    target_results['ion_positions_relative'] = position_data
                                    print(f"       Position data loaded: {len(position_data)} frames")
                                except Exception as e:
                                    print(f"       ⚠️ Failed to load position data: {e}")
                                    # Position data missing - force recalculation
                                    targets_to_calculate.append(sel)
                                    continue
                            elif return_positions:
                                print(f"       ⚠️ Position cache missing: {pos_cache_file}")
                                # Position data missing - force recalculation
                                targets_to_calculate.append(sel)
                                continue
                            
                            results_dict[label] = target_results
                            
                        except Exception as e:
                            print(f"   ⚠️ Cache loading failed for {label}: {e}")
                            targets_to_calculate.append(sel)
                    else:
                        print(f"   📋 No cache found for {label}")
                        targets_to_calculate.append(sel)
                
                # Summary of cache status
                cached_count = len(results_dict)
                missing_count = len(targets_to_calculate)
                print(f"\n📊 Cache Status:")
                print(f"   ✅ Cached targets: {cached_count}")
                print(f"   🔄 Missing targets: {missing_count}")
                if cached_count > 0:
                    print(f"   📋 Cached: {sorted(results_dict.keys())}")
                if missing_count > 0:
                    missing_labels = [self._extract_label_from_selection(sel) for sel in targets_to_calculate]
                    print(f"   📋 Missing: {sorted(missing_labels)}")
                    
            else:
                # No cache check or force rerun - calculate all targets  
                if force_rerun:
                    print("🔄 Force rerun enabled - calculating all targets")
                else:
                    print("💾 Caching disabled - calculating all targets")
                targets_to_calculate = target_sel
            
            # Calculate missing targets only
            if targets_to_calculate:
                total_to_calc = len(targets_to_calculate)
                print(f"\n🔬 Calculating {total_to_calc} target(s)...")
                
                for idx, sel in enumerate(targets_to_calculate, 1):
                    label = self._extract_label_from_selection(sel)
                    
                    print(f"\n[{idx}/{total_to_calc}] Analyzing spatial binding: {label}")
                    print(f"  Target: {sel}")
                    
                    # Recursively call this function with single string
                    # Use the current target as its own reference frame source
                    spatial_result = self.spatial_binding_analysis(
                        sel,
                        ion_type=ion_type,
                        solvation=solvation,
                        cutoff=cutoff,
                        step=step,
                        method=method,
                        angular_bins=angular_bins,
                        return_positions=return_positions,
                        rdf_data=rdf_data,
                        rdf_boundaries=rdf_boundaries,
                        solvation_shells=solvation_shells,
                        peaks=peaks,
                        molecular_frame_tracking=molecular_frame_tracking,
                        reference_atoms=reference_atoms,
                        reference_target=sel,  # Use current target as reference source
                        save_cache=save_cache,  # Save individual cache for this target
                        cache_file=None,        # Auto-generate cache filename
                        force_rerun=force_rerun,
                        target_label=label      # Pass the extracted label for peak matching
                    )
                    
                    results_dict[label] = spatial_result
            
            print(f"\n✅ Batch spatial binding analysis complete!")
            print(f"   📊 Total results: {len(results_dict)}")
            print(f"   📋 Labels: {sorted(results_dict.keys())}")
            return results_dict
        
        # Single target analysis (original logic)
        # Generate cache filename if not provided
        if cache_file is None:
            # Create hash from parameters for unique cache filename
            shells_str = str(solvation_shells) if solvation_shells else 'None'
            peaks_str = str(peaks) if peaks else 'None'
            boundaries_str = str(bool(rdf_boundaries))  # Just track if boundaries are used
            molecular_frame_str = f"mft{molecular_frame_tracking}_ra{reference_atoms}_rt{reference_target}"
            param_str = f"{target_sel}_{analysis_mode}_{analysis_species}_c{cutoff}_s{step}_m{method}_ab{angular_bins}_rp{return_positions}_sh{shells_str}_pk{peaks_str}_b{boundaries_str}_{molecular_frame_str}"
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            cache_file = f"spatial_cache_{param_hash}.npz"
        
        # Check for existing cache
        if save_cache and not force_rerun and os.path.exists(cache_file):
            print(f"📂 Found existing spatial binding cache: {cache_file}")
            try:
                print("   Loading cached spatial binding results...")
                cached_data = np.load(cache_file, allow_pickle=True)
                
                # Reconstruct results dictionary
                results = {}
                for key in cached_data.files:
                    if key.startswith('metadata_'):
                        continue
                    # Handle different data types safely
                    data = cached_data[key]
                    try:
                        if data.dtype == object:
                            results[key] = data.item()
                        elif data.ndim == 0:  # Scalar array
                            results[key] = data.item()
                        else:
                            results[key] = data
                    except (ValueError, AttributeError):
                        # Fallback for problematic data
                        results[key] = data
                
                # Load position data if present
                pos_cache_file = cache_file.replace('.npz', '_positions.pkl')
                if return_positions and os.path.exists(pos_cache_file):
                    try:
                        import pickle
                        with open(pos_cache_file, 'rb') as f:
                            position_data = pickle.load(f)
                        results['ion_positions_relative'] = position_data
                        print(f"   ✅ Position data loaded: {len(position_data)} frames")
                    except Exception as e:
                        print(f"   ⚠️ Failed to load position data: {e}")
                elif return_positions:
                    print(f"   ⚠️ Position cache file not found: {pos_cache_file}")
                    print(f"   💡 Position data will be missing - recommend force_rerun=True")
                
                # Load triangulation data if present
                tri_cache_file = cache_file.replace('.npz', '_triangulation.pkl')
                if return_positions and os.path.exists(tri_cache_file):
                    try:
                        import pickle
                        with open(tri_cache_file, 'rb') as f:
                            triangulation_data = pickle.load(f)
                        results['triangulation_data'] = triangulation_data
                        print(f"   ✅ Triangulation data loaded: {len(triangulation_data['frame_indices'])} ion-atom pairs")
                    except Exception as e:
                        print(f"   ⚠️ Failed to load triangulation data: {e}")
                elif return_positions:
                    print(f"   ⚠️ Triangulation cache file not found: {tri_cache_file}")
                    print(f"   💡 Consider re-running with new triangulation features")
                
                # Verify position data if return_positions was requested
                if return_positions and 'ion_positions_relative' not in results:
                    print(f"   ⚠️ Missing position data! Forcing recalculation...")
                    print(f"   📂 Cache loading failed - falling through to recalculation")
                    # Don't return here - fall through to recalculation
                else:
                    print(f"   ✅ Loaded cached spatial binding results successfully!")
                    print(f"   Method: {results.get('method', 'unknown')}")
                    print(f"   Total contacts: {results.get('total_contacts', 'unknown')}")
                    if return_positions:
                        pos_count = len(results.get('ion_positions_relative', []))
                        total_positions = sum(len(pos) for pos in results.get('ion_positions_relative', []))
                        print(f"   Position frames: {pos_count}, Total positions: {total_positions}")
                    return results
                
            except Exception as e:
                print(f"   ⚠️ Failed to load cache: {e}")
                print(f"   Recalculating spatial binding analysis...")
        
        # Get target and ion atoms
        target = self.universe.select_atoms(target_sel)
        if len(target) == 0:
            raise ValueError(f"No atoms found for target selection: {target_sel}")
        
        # Set up molecular frame reference tracking
        molecular_reference_atoms = None
        if molecular_frame_tracking and return_positions:
            print(f"🔧 Setting up molecular frame reference tracking...")
            print(f"   Target selection: {target_sel}")
            print(f"   Target atoms: {len(target)} atoms")
            print(f"   Target atom types: {set(target.names)}")
            
            # Use reference_target if specified, otherwise use full target
            reference_target_atoms = target
            if reference_target is not None:
                reference_target_atoms = self.universe.select_atoms(reference_target)
                print(f"   Reference target selection: {reference_target}")
                print(f"   Reference target atoms: {len(reference_target_atoms)} atoms")
                print(f"   Reference atom types: {set(reference_target_atoms.names)}")
                
                if len(reference_target_atoms) == 0:
                    print(f"   ⚠️ Reference target selection returned no atoms, falling back to full target")
                    reference_target_atoms = target
            
            # Determine number of reference atoms based on method
            if n_reference_atoms is None:
                # Auto-determine based on molecular_frame_method
                if molecular_frame_method == 'pca':
                    n_ref = min(10, len(reference_target_atoms))  # Use up to 10 atoms for PCA
                    print(f"   Using PCA mode with {n_ref} reference atoms")
                else:
                    n_ref = 3  # Default for Gram-Schmidt
            else:
                n_ref = n_reference_atoms
            
            molecular_reference_atoms = self._select_reference_atoms(reference_target_atoms, reference_atoms, n_atoms=n_ref)
            if molecular_reference_atoms is not None:
                ref_names = [reference_target_atoms[idx].name for idx in molecular_reference_atoms]
                ref_indices = [reference_target_atoms[idx].index for idx in molecular_reference_atoms]
                ref_positions = reference_target_atoms[molecular_reference_atoms].positions
                
                print(f"   Reference atoms: {list(zip(ref_names, ref_indices))}")
                print(f"   Reference positions: {ref_positions}")
                print(f"   Reference elements: {[reference_target_atoms[idx].element for idx in molecular_reference_atoms]}")
                
                # Check which molecular region these atoms belong to (if possible)
                try:
                    ref_resnames = [reference_target_atoms[idx].resname for idx in molecular_reference_atoms]
                    ref_segids = [reference_target_atoms[idx].segid for idx in molecular_reference_atoms] if hasattr(reference_target_atoms[0], 'segid') else ['N/A'] * 3
                    print(f"   Reference residues: {ref_resnames}")
                    print(f"   Reference segments: {ref_segids}")
                except AttributeError:
                    print(f"   Residue/segment info not available")
                    
                # Convert local indices to global indices if using reference_target
                if reference_target is not None:
                    global_ref_indices = []
                    for idx in molecular_reference_atoms:
                        global_idx = reference_target_atoms[idx].index
                        # Find the corresponding index in the full target selection
                        try:
                            target_local_idx = np.where(target.indices == global_idx)[0][0]
                            global_ref_indices.append(target_local_idx)
                        except IndexError:
                            print(f"   ⚠️ Reference atom {global_idx} not found in target selection")
                            global_ref_indices = None
                            break
                    molecular_reference_atoms = global_ref_indices
                    
                print(f"   This will enable rotation-aware ion position reconstruction")
            else:
                print(f"   ⚠️ Failed to select reference atoms, falling back to COM-relative positioning")
                molecular_frame_tracking = False
        
        # Find analysis atoms based on mode
        if analysis_mode == 'ion':
            # Find ion atoms (existing logic)
            analysis_species_upper = analysis_species.upper()
            analysis_atoms = None
            
            # Check cations
            for ion_name, atoms in self.cation_types.items():
                if ion_name.upper() == analysis_species_upper:
                    analysis_atoms = atoms
                    break
            
            # Check anions if not found
            if analysis_atoms is None:
                for ion_name, atoms in self.anion_types.items():
                    if ion_name.upper() == analysis_species_upper:
                        analysis_atoms = atoms
                        break
            
            if analysis_atoms is None or len(analysis_atoms) == 0:
                raise ValueError(f"Ion type '{analysis_species}' not found in system")
                
        else:  # solvation mode
            # Find solvent atoms
            try:
                # Try direct atom name selection
                analysis_atoms = self.universe.select_atoms(f'name {analysis_species}')
                if len(analysis_atoms) == 0:
                    # Try as part of solvent selection
                    analysis_atoms = self.solvents.select_atoms(f'name {analysis_species}')
                if len(analysis_atoms) == 0:
                    raise ValueError(f"No atoms found")
            except Exception as e:
                raise ValueError(f"Solvation species '{analysis_species}' not found in system. Error: {e}")
        
        # Determine analysis mode for shell boundaries
        # Shell boundaries can be used if rdf_boundaries is provided AND appropriate analysis regions are specified
        use_shell_boundaries = False
        analysis_type = None
        
        if rdf_boundaries is not None:
            if analysis_mode == 'ion' and peaks is not None:
                use_shell_boundaries = True
                analysis_type = 'peaks'
                print(f"Using shell-boundary based analysis with peaks")
            elif analysis_mode == 'solvation' and solvation_shells is not None:
                use_shell_boundaries = True
                analysis_type = 'shells'
                print(f"Using shell-boundary based analysis with solvation shells")
            else:
                analysis_type = None
                print(f"Shell boundaries available but no peaks/shells specified for {analysis_mode} mode")
        
        if use_shell_boundaries:
            # Find the target-species key
            target_species_key = None
            # Use provided target_label (from dict key) if available, otherwise extract from selection
            if target_label is None:
                target_label = self._extract_label_from_selection(target_sel) if isinstance(target_sel, str) else 'target'
            
            if analysis_type == 'peaks':
                # Look for target-ion combinations in peaks
                for key in peaks.keys():
                    if analysis_species.upper() in key.upper() and target_label.lower() in key.lower():
                        target_species_key = key
                        break
                if target_species_key:
                    selected_analysis_regions = peaks[target_species_key]
                    print(f"Found peak definition for '{target_species_key}': {selected_analysis_regions}")
                else:
                    print(f"Warning: No peak definition found for {target_label}-{analysis_species}, falling back to cutoff method")
                    use_shell_boundaries = False
                    
            else:  # solvation_shells
                # Look for target-solvation combinations in solvation_shells
                for key in solvation_shells.keys():
                    if analysis_species.upper() in key.upper() and target_label.lower() in key.lower():
                        target_species_key = key
                        break
                if target_species_key:
                    selected_analysis_regions = solvation_shells[target_species_key]
                    print(f"Found shell definition for '{target_species_key}': {selected_analysis_regions}")
                else:
                    print(f"Warning: No shell definition found for {target_label}-{analysis_species}, falling back to cutoff method")
                    use_shell_boundaries = False
                    
            # Validate that the key exists in rdf_boundaries
            if use_shell_boundaries and target_species_key not in rdf_boundaries:
                print(f"Warning: No RDF boundaries found for '{target_species_key}', falling back to cutoff method")
                use_shell_boundaries = False
        
        if not use_shell_boundaries:
            print(f"Using distance cutoff based analysis (cutoff: {cutoff} Å)")
        
        print(f"Analyzing spatial binding of {len(analysis_atoms)} {analysis_species} atoms to {len(target)} target atoms")
        print(f"Method: {method}, Step: {step}")
        
        # Initialize results
        results = {
            'total_contacts': 0,
            'frames_analyzed': 0,
            'method': method,
            'analysis_mode': analysis_mode,
            'analysis_species': analysis_species,
            'cutoff': cutoff,
            'use_shell_boundaries': use_shell_boundaries,
            'target_species_key': target_species_key if use_shell_boundaries else None,
            'selected_analysis_regions': selected_analysis_regions if use_shell_boundaries else None
        }
        
        # Always store target atom information for consistent plotting
        results['atom_indices'] = target.indices
        results['atom_names'] = target.names
        
        # Per-atom analysis
        if method in ['per-atom', 'both']:
            contact_frequency = np.zeros(len(target))
        
        # Spherical analysis
        if method in ['spherical', 'both']:
            n_theta, n_phi = angular_bins
            angular_histogram = np.zeros((n_theta, n_phi))
            theta_edges = np.linspace(0, np.pi, n_theta + 1)
            phi_edges = np.linspace(0, 2 * np.pi, n_phi + 1)
            results['theta_bins'] = theta_edges
            results['phi_bins'] = phi_edges
            results['theta_centers'] = (theta_edges[:-1] + theta_edges[1:]) / 2
            results['phi_centers'] = (phi_edges[:-1] + phi_edges[1:]) / 2
        
        # Triangulation storage for precise geometric mapping
        if return_positions:
            triangulation_data = {
                'frame_indices': [],
                'ion_indices': [],
                'ion_types': [],
                'target_atom_indices': [],
                'COM_to_atom_vectors': [],     # Target COM -> specific target atom
                'atom_to_ion_vectors': [],     # Specific target atom -> ion
                'COM_to_ion_vectors': [],      # Target COM -> ion (validation)
                'distances_to_target': [],     # Distance from ion to target atom
                'box_dimensions': [],          # Store box dimensions for PBC reconstruction
                'target_com_positions': [],    # Store target COM for each contact
                # Spherical coordinate system (more robust for molecular rotation)
                'spherical_r': [],             # Distance from target COM to ion
                'spherical_theta': [],         # Polar angle (0 to π)
                'spherical_phi': [],           # Azimuthal angle (0 to 2π)
                # Peak region labels (for shell-boundary analysis)
                'region_labels': [],           # Which peak region (P1, P2, P3, etc.) each contact belongs to
                # Molecular frame tracking data (rotation-aware triangulation)
                'molecular_frame_tracking': molecular_frame_tracking,
                'reference_atom_indices': molecular_reference_atoms,  # Global atom indices of reference atoms
                'reference_atom_positions': [],     # Reference atom positions per frame
                'molecular_frame_vectors': [],      # Ion position in molecular coordinate system
                'frame_rotation_matrices': [],      # Rotation matrices to original frame (if needed)
                'reference_frame_established': None   # Frame 0 reference configuration
            }
            
            # Store reference frame information if molecular tracking is enabled
            if molecular_frame_tracking and molecular_reference_atoms is not None:
                self.universe.trajectory[0]  # Go to first frame
                reference_positions = target[molecular_reference_atoms].positions.copy()
                reference_target_com = target.center_of_mass()  # Store COM from reference frame
                triangulation_data['reference_frame_established'] = reference_positions
                triangulation_data['reference_target_com'] = reference_target_com
                print(f"   Reference frame established from frame 0")
            
            # Keep backward compatibility
            ion_positions_relative = []
            
        # Analyze each frame
        for ts in tqdm(self.universe.trajectory[::step], desc=f"Analyzing {analysis_species} binding"):
            results['frames_analyzed'] += 1
            
            # Get COM of target
            target_com = target.center_of_mass()
            
            # Initialize frame data collection
            frame_contact_positions = []
            frame_ion_positions_relative = []
            
            # Calculate distances
            dist_matrix = distances.distance_array(target.positions,
                                                  analysis_atoms.positions,
                                                  box=ts.dimensions)
            
            # Find contacts using appropriate method
            if use_shell_boundaries:
                # Shell-boundary based contact detection
                contacts = np.zeros_like(dist_matrix, dtype=bool)
                shell_boundaries = rdf_boundaries[target_species_key]
                
                # Also track which region each cell belongs to (for per-region filtering later)
                contact_region_map = {}  # (target_idx, ion_idx) -> region_name
                
                for region_name in selected_analysis_regions:
                    if region_name in shell_boundaries:
                        r_min, r_max = shell_boundaries[region_name]
                        # Atoms in this region for any target atom
                        region_contacts = (dist_matrix >= r_min) & (dist_matrix <= r_max)
                        contacts |= region_contacts
                        
                        # Map each contact to its region
                        region_pairs = np.where(region_contacts)
                        for target_idx, ion_idx in zip(region_pairs[0], region_pairs[1]):
                            # If already mapped (overlapping regions), keep first region
                            if (target_idx, ion_idx) not in contact_region_map:
                                contact_region_map[(target_idx, ion_idx)] = region_name
            else:
                # Distance cutoff based contact detection
                contacts = dist_matrix <= cutoff
                contact_region_map = {}  # Empty for non-boundary mode
            
            # Track contacts and positions properly per target atom
            total_frame_contacts = 0
            
            # Collect all contacted ion positions for this frame (for ion_positions_relative backward compatibility)
            frame_ion_positions_relative = []
            if return_positions:
                contacted_ion_indices = np.where(contacts.any(axis=0))[0]
                for ion_idx in contacted_ion_indices:
                    ion_pos = analysis_atoms[ion_idx].position
                    # Apply PBC correction for COM-relative position
                    ion_relative_to_com = ion_pos - target_com
                    if ts.dimensions is not None:
                        box = ts.dimensions[:3]
                        for i in range(3):
                            if ion_relative_to_com[i] > box[i] * 0.5:
                                ion_relative_to_com[i] -= box[i]
                            elif ion_relative_to_com[i] < -box[i] * 0.5:
                                ion_relative_to_com[i] += box[i]
                    frame_ion_positions_relative.append(ion_relative_to_com.copy())
            
            # Store frame positions in backward compatibility format
            if return_positions and frame_ion_positions_relative:
                ion_positions_relative.append(np.array(frame_ion_positions_relative))
            
            # Get target center of mass for this frame
            target_com = target.center_of_mass()
            
            # Per-atom analysis
            if method in ['per-atom', 'both']:
                # Count contacts per target atom
                contact_frequency += contacts.sum(axis=1)
                
                # Store triangulation vectors for precise geometric mapping
                if return_positions:
                    # For each contacted ion, find closest target atom and store triangulation
                    contacted_pairs = np.where(contacts)
                    target_atom_indices = contacted_pairs[0]
                    ion_indices = contacted_pairs[1]
                    
                    for target_idx, ion_idx in zip(target_atom_indices, ion_indices):
                        # Get positions - use MDAnalysis for proper PBC handling
                        target_atom = target[target_idx]
                        ion_atom = analysis_atoms[ion_idx]
                        
                        # Use MDAnalysis distance calculations with automatic PBC handling
                        from MDAnalysis.lib.distances import distance_array, calc_bonds
                        
                        # Calculate distance using MDAnalysis (handles PBC correctly)
                        distance_pbc = calc_bonds(target_atom.position.reshape(1, 3),
                                                ion_atom.position.reshape(1, 3),
                                                box=ts.dimensions)[0]
                        
                        # Calculate vectors using proper PBC-aware distance calculation
                        # Get the actual shortest vector between atoms (accounting for PBC)
                        target_pos = target_atom.position
                        ion_pos = ion_atom.position
                        
                        # Calculate minimum image vectors using proper algorithm
                        atom_to_ion_vector = ion_pos - target_pos
                        if ts.dimensions is not None:
                            box = ts.dimensions[:3]
                            for i in range(3):
                                if atom_to_ion_vector[i] > box[i] * 0.5:
                                    atom_to_ion_vector[i] -= box[i]
                                elif atom_to_ion_vector[i] < -box[i] * 0.5:
                                    atom_to_ion_vector[i] += box[i]
                        
                        # Calculate COM vectors using same corrected approach
                        COM_to_atom_vector = target_pos - target_com
                        COM_to_ion_vector = ion_pos - target_com
                        if ts.dimensions is not None:
                            box = ts.dimensions[:3]
                            for vec in [COM_to_atom_vector, COM_to_ion_vector]:
                                for i in range(3):
                                    if vec[i] > box[i] * 0.5:
                                        vec[i] -= box[i]
                                    elif vec[i] < -box[i] * 0.5:
                                        vec[i] += box[i]
                        
                        # Store triangulation data with validated PBC vectors
                        triangulation_data['frame_indices'].append(results['frames_analyzed'] - 1)
                        triangulation_data['ion_indices'].append(ion_idx)
                        triangulation_data['ion_types'].append(analysis_species)
                        triangulation_data['target_atom_indices'].append(target_idx)
                        triangulation_data['COM_to_atom_vectors'].append(COM_to_atom_vector.copy())
                        triangulation_data['atom_to_ion_vectors'].append(atom_to_ion_vector.copy())
                        triangulation_data['COM_to_ion_vectors'].append(COM_to_ion_vector.copy())
                        triangulation_data['distances_to_target'].append(distance_pbc)
                        triangulation_data['box_dimensions'].append(ts.dimensions.copy() if ts.dimensions is not None else None)
                        triangulation_data['target_com_positions'].append(target_com.copy())
                        
                        # Calculate spherical coordinates directly from ion position relative to COM (truly independent)
                        ion_relative_to_com = ion_pos - target_com
                        # Apply proper PBC correction for spherical coordinates
                        if ts.dimensions is not None:
                            box = ts.dimensions[:3]
                            for i in range(3):
                                if ion_relative_to_com[i] > box[i] * 0.5:
                                    ion_relative_to_com[i] -= box[i]
                                elif ion_relative_to_com[i] < -box[i] * 0.5:
                                    ion_relative_to_com[i] += box[i]
                        
                        r = np.linalg.norm(ion_relative_to_com)
                        if r > 0:  # Avoid division by zero
                            x, y, z = ion_relative_to_com
                            theta = np.arccos(np.clip(z / r, -1, 1))  # Polar angle [0, π]
                            phi = np.arctan2(y, x)  # Azimuthal [-π, π]
                            if phi < 0:
                                phi += 2 * np.pi  # Convert to [0, 2π]
                        else:
                            theta = 0.0
                            phi = 0.0
                        
                        triangulation_data['spherical_r'].append(r)
                        triangulation_data['spherical_theta'].append(theta)
                        triangulation_data['spherical_phi'].append(phi)
                        
                        # Store region label if using shell boundaries
                        if use_shell_boundaries and (target_idx, ion_idx) in contact_region_map:
                            triangulation_data['region_labels'].append(contact_region_map[(target_idx, ion_idx)])
                        else:
                            triangulation_data['region_labels'].append(None)  # No region for cutoff mode
                        
                        # Molecular frame tracking (rotation-aware triangulation)
                        if molecular_frame_tracking and molecular_reference_atoms is not None:
                            # Get current reference atom positions
                            current_ref_positions = target[molecular_reference_atoms].positions.copy()
                            triangulation_data['reference_atom_positions'].append(current_ref_positions)
                            
                            # Calculate ion position in molecular coordinate system
                            molecular_frame_vector = self._compute_molecular_frame_vector(
                                ion_pos, target_com, current_ref_positions,
                                triangulation_data['reference_frame_established'], ts.dimensions
                            )
                            triangulation_data['molecular_frame_vectors'].append(molecular_frame_vector)
                        else:
                            # Maintain data structure consistency for backward compatibility
                            triangulation_data['reference_atom_positions'].append(None)
                            triangulation_data['molecular_frame_vectors'].append(None)
                        
                        # ENHANCED VALIDATION: Always check PBC consistency properly
                        reconstructed_check = target_pos + atom_to_ion_vector
                        
                        # Calculate minimum image distance for validation
                        raw_diff = reconstructed_check - ion_pos
                        pbc_corrected_diff = raw_diff.copy()
                        if ts.dimensions is not None:
                            box = ts.dimensions[:3]
                            for i in range(3):
                                if pbc_corrected_diff[i] > box[i] * 0.5:
                                    pbc_corrected_diff[i] -= box[i]
                                elif pbc_corrected_diff[i] < -box[i] * 0.5:
                                    pbc_corrected_diff[i] += box[i]
                        
                        distance_check = np.linalg.norm(pbc_corrected_diff)
                        
                        # Track validation statistics
                        if not hasattr(results, 'validation_stats'):
                            results['validation_stats'] = {'max_error': 0, 'total_checks': 0, 'pbc_crossings': 0}
                        
                        results['validation_stats']['total_checks'] += 1
                        results['validation_stats']['max_error'] = max(results['validation_stats']['max_error'], distance_check)
                        
                        # Check for PBC crossings (when ion and target are > 1/4 box apart)
                        if np.any(np.abs(raw_diff) > ts.dimensions[:3] * 0.25):
                            results['validation_stats']['pbc_crossings'] += 1
                        
                        if distance_check > 0.01:  # This should never happen with correct PBC
                            print(f"🚨 REAL PBC ERROR: {distance_check:.6f} Å after PBC correction!")
                            print(f"   Frame {results['frames_analyzed']}, target atom {target_idx}, ion {ion_idx}")
                            print(f"   Box: {ts.dimensions[:3]}")
                            print(f"   Target: {target_pos}, Ion: {ion_pos}")
                            print(f"   Raw diff: {raw_diff}")
                            print(f"   PBC corrected diff: {pbc_corrected_diff}")
                        elif len(triangulation_data['frame_indices']) % 50000 == 1:  # Periodic status
                            stats = results['validation_stats']
                            pbc_rate = stats['pbc_crossings'] / stats['total_checks'] * 100
                            print(f"📊 PBC Stats: {stats['total_checks']} checks, {stats['pbc_crossings']} crossings ({pbc_rate:.1f}%), max_error={stats['max_error']:.6f}Å")
                        
                        total_frame_contacts += 1
            
            # Spherical analysis
            if method in ['spherical', 'both']:
                # Get all contact indices for spherical analysis
                contact_indices = np.where(contacts.any(axis=0))[0]
                if len(contact_indices) > 0:
                    # Count contacts for spherical analysis (avoid double counting)
                    if method == 'spherical':
                        # For spherical-only method, count all contacts and store triangulation data
                        total_frame_contacts += len(contact_indices)
                        if return_positions:
                            # Use same PBC-corrected triangulation logic as per-atom method
                            contacted_pairs = np.where(contacts)
                            target_atom_indices = contacted_pairs[0]
                            ion_indices = contacted_pairs[1]
                            
                            for target_idx, ion_idx in zip(target_atom_indices, ion_indices):
                                # Get positions - use MDAnalysis for proper PBC handling
                                target_atom = target[target_idx]
                                ion_atom = analysis_atoms[ion_idx]
                                
                        # Use MDAnalysis for proper PBC distance calculation
                        from MDAnalysis.lib.distances import calc_bonds
                        
                        # Calculate distance using MDAnalysis (handles PBC correctly)
                        distance_pbc = calc_bonds(target_pos.reshape(1, 3),
                                                ion_pos.reshape(1, 3),
                                                box=ts.dimensions)[0]
                        
                        # Calculate minimum image vectors using MDAnalysis
                        if ts.dimensions is not None:
                            # Use MDAnalysis distance calculation to get proper minimum image vectors
                            from MDAnalysis.lib.distances import calc_distance_array
                            
                            # Get minimum image vector correctly
                            target_pos_reshaped = target_pos.reshape(1, 3)
                            ion_pos_reshaped = ion_pos.reshape(1, 3)
                            
                            # Calculate the minimum image vector manually using proper algorithm
                            box = ts.dimensions[:3]
                            atom_to_ion_vector = ion_pos - target_pos
                            
                            # Apply proper minimum image convention
                            for i in range(3):
                                if atom_to_ion_vector[i] > box[i] * 0.5:
                                    atom_to_ion_vector[i] -= box[i]
                                elif atom_to_ion_vector[i] < -box[i] * 0.5:
                                    atom_to_ion_vector[i] += box[i]
                        else:
                            atom_to_ion_vector = ion_pos - target_pos
                            distance_pbc = np.linalg.norm(atom_to_ion_vector)
                        
                        # Calculate COM vectors using the same corrected approach
                        COM_to_atom_vector = target_pos - target_com
                        COM_to_ion_vector = ion_pos - target_com
                        
                        if ts.dimensions is not None:
                            box = ts.dimensions[:3]
                            # Apply minimum image convention to COM vectors
                            for vec in [COM_to_atom_vector, COM_to_ion_vector]:
                                for i in range(3):
                                    if vec[i] > box[i] * 0.5:
                                        vec[i] -= box[i]
                                    elif vec[i] < -box[i] * 0.5:
                                        vec[i] += box[i]
                                
                                # Store triangulation data with validated PBC vectors
                                triangulation_data['frame_indices'].append(results['frames_analyzed'] - 1)
                                triangulation_data['ion_indices'].append(ion_idx)
                                triangulation_data['ion_types'].append(analysis_species)
                                triangulation_data['target_atom_indices'].append(target_idx)
                                triangulation_data['COM_to_atom_vectors'].append(COM_to_atom_vector.copy())
                                triangulation_data['atom_to_ion_vectors'].append(atom_to_ion_vector.copy())
                                triangulation_data['COM_to_ion_vectors'].append(COM_to_ion_vector.copy())
                                triangulation_data['distances_to_target'].append(distance_pbc)
                                triangulation_data['box_dimensions'].append(ts.dimensions.copy() if ts.dimensions is not None else None)
                                triangulation_data['target_com_positions'].append(target_com.copy())
                                
                                # Calculate spherical coordinates directly from ion position relative to COM (truly independent)
                                ion_relative_to_com = ion_pos - target_com
                                # Apply PBC correction for spherical coordinates
                                if ts.dimensions is not None:
                                    box = ts.dimensions[:3]
                                    for i in range(3):
                                        while ion_relative_to_com[i] > box[i]/2:
                                            ion_relative_to_com[i] -= box[i]
                                        while ion_relative_to_com[i] < -box[i]/2:
                                            ion_relative_to_com[i] += box[i]
                                
                                r = np.linalg.norm(ion_relative_to_com)
                                if r > 0:  # Avoid division by zero
                                    x, y, z = ion_relative_to_com
                                    theta = np.arccos(np.clip(z / r, -1, 1))  # Polar angle [0, π]
                                    phi = np.arctan2(y, x)  # Azimuthal [-π, π]
                                    if phi < 0:
                                        phi += 2 * np.pi  # Convert to [0, 2π]
                                else:
                                    theta = 0.0
                                    phi = 0.0
                                
                                triangulation_data['spherical_r'].append(r)
                                triangulation_data['spherical_theta'].append(theta)
                                triangulation_data['spherical_phi'].append(phi)
                    
                    # Calculate angular coordinates for spherical histogram
                    contact_positions = analysis_atoms[contact_indices].positions
                    
                    # Apply proper PBC for spherical coordinate calculation
                    spherical_vectors = []
                    for ion_pos in contact_positions:
                        diff = ion_pos - target_com
                        if ts.dimensions is not None:
                            # Apply minimum image convention
                            box = ts.dimensions[:3]
                            for i in range(3):
                                while diff[i] > box[i]/2:
                                    diff[i] -= box[i]
                                while diff[i] < -box[i]/2:
                                    diff[i] += box[i]
                        spherical_vectors.append(diff)
                    
                    # Convert to spherical coordinates and update histogram
                    if spherical_vectors:
                        spherical_positions = np.array(spherical_vectors)
                        x, y, z = spherical_positions.T
                        r = np.sqrt(x**2 + y**2 + z**2)
                        theta = np.arccos(np.clip(z / r, -1, 1))  # Polar angle [0, π]
                        phi = np.arctan2(y, x)  # Azimuthal angle [-π, π]
                        phi[phi < 0] += 2 * np.pi  # Convert to [0, 2π]
                        
                        # Bin into histogram
                        for t, p in zip(theta, phi):
                            theta_idx = np.digitize(t, theta_edges) - 1
                            phi_idx = np.digitize(p, phi_edges) - 1
                            
                            # Handle edge cases
                            theta_idx = np.clip(theta_idx, 0, n_theta - 1)
                            phi_idx = np.clip(phi_idx, 0, n_phi - 1)
                            
                            angular_histogram[theta_idx, phi_idx] += 1
            
            # Update total contacts
            results['total_contacts'] += total_frame_contacts
        
        # Store results
        if method in ['per-atom', 'both']:
            results['contact_frequency'] = contact_frequency
            # Store atom positions from first frame (already visited)
            self.universe.trajectory[0]
            results['atom_positions'] = target.positions.copy()

        if method in ['spherical', 'both']:
            results['angular_histogram'] = angular_histogram

        # Always include triangulation data and ion positions when requested
        if return_positions:
            # Store triangulation data for precise geometric mapping
            if triangulation_data['frame_indices']:
                # Convert lists to numpy arrays for efficient storage and processing
                for key in triangulation_data:
                    if key.endswith('_vectors') or key == 'distances_to_target':
                        triangulation_data[key] = np.array(triangulation_data[key])
                    else:
                        triangulation_data[key] = np.array(triangulation_data[key])
                
                results['triangulation_data'] = triangulation_data
                print(f"  Triangulation data stored: {len(triangulation_data['frame_indices'])} ion-atom pairs")
            
            # Keep backward compatibility with ion_positions_relative
            results['ion_positions_relative'] = ion_positions_relative
            print(f"  Ion positions stored: {len(ion_positions_relative)} frames with {sum(len(pos) for pos in ion_positions_relative) if ion_positions_relative else 0} total positions")

        # Save to cache if requested
        if save_cache:
            print(f"\n💾 Saving spatial binding analysis to cache: {cache_file}")
            try:
                # Separate position data and triangulation data from other results
                save_dict = {}
                position_data = None
                triangulation_cache_data = None
                
                for key, value in results.items():
                    if key == 'ion_positions_relative':
                        position_data = value
                    elif key == 'triangulation_data':
                        triangulation_cache_data = value
                    else:
                        save_dict[key] = value
                
                # Add metadata
                save_dict['metadata_target_sel'] = target_sel
                save_dict['metadata_analysis_mode'] = analysis_mode
                save_dict['metadata_analysis_species'] = analysis_species
                save_dict['metadata_cutoff'] = cutoff
                save_dict['metadata_step'] = step
                save_dict['metadata_method'] = method
                save_dict['metadata_angular_bins'] = angular_bins
                save_dict['metadata_return_positions'] = return_positions
                save_dict['metadata_molecular_frame_tracking'] = molecular_frame_tracking
                save_dict['metadata_reference_atoms'] = str(reference_atoms)
                save_dict['metadata_molecular_frame_tracking'] = molecular_frame_tracking
                save_dict['metadata_reference_atoms'] = reference_atoms
                
                # Save main data as npz
                np.savez(cache_file, **save_dict)
                
                # Save position data separately using pickle if present
                if position_data is not None:
                    import pickle
                    pos_cache_file = cache_file.replace('.npz', '_positions.pkl')
                    with open(pos_cache_file, 'wb') as f:
                        pickle.dump(position_data, f)
                
                # Save triangulation data separately using pickle if present
                if triangulation_cache_data is not None:
                    import pickle
                    tri_cache_file = cache_file.replace('.npz', '_triangulation.pkl')
                    with open(tri_cache_file, 'wb') as f:
                        pickle.dump(triangulation_cache_data, f)
                
                print(f"   ✅ Cache saved successfully!")
                print(f"   📁 File: {cache_file}")
            except Exception as e:
                print(f"   ⚠️ Failed to save cache: {e}")
                print(f"   Continuing without cache...")
        
        print(f"✓ Analysis complete!")
        print(f"  Total contacts: {results['total_contacts']}")
        print(f"  Frames analyzed: {results['frames_analyzed']}")
        if method in ['per-atom', 'both']:
            max_contacts = contact_frequency.max()
            max_atom_idx = contact_frequency.argmax()
            print(f"  Max contacts: {max_contacts:.0f} at atom {target.names[max_atom_idx]} (index {target.indices[max_atom_idx]})")
        
        return results

    def _select_reference_atoms(self, target, reference_atoms, n_atoms=3):
        """
        Select molecular reference atoms to define coordinate frame for rotation-aware triangulation.
        
        Parameters
        ----------
        target : MDAnalysis.AtomGroup
            Target molecule atoms
        reference_atoms : str or list
            Reference atom selection method or explicit indices
        n_atoms : int
            Number of reference atoms to select (default: 3 for Gram-Schmidt, use 5-10 for PCA)
            
        Returns
        -------
        ref_indices : list or None
            List of reference atom indices within target, or None if selection failed
        """
        if isinstance(reference_atoms, list):
            # Manual specification - validate indices
            if len(reference_atoms) < 3:
                print(f"⚠️ Manual reference_atoms must contain at least 3 indices, got {len(reference_atoms)}")
                return None
            
            # Validate indices are within target
            if any(idx >= len(target) or idx < 0 for idx in reference_atoms):
                print(f"⚠️ Reference atom indices {reference_atoms} outside target range [0, {len(target)-1}]")
                return None
            
            # Return requested number of atoms (or all if fewer specified)
            return reference_atoms[:n_atoms] if len(reference_atoms) > n_atoms else reference_atoms
            
        elif reference_atoms == 'auto':
            # Enhanced automatic selection with stability checking
            if len(target) < 3:
                print(f"⚠️ Target has only {len(target)} atoms, need at least 3 for reference frame")
                return None
                
            # Get first frame positions for distance calculation
            self.universe.trajectory[0]
            positions = target.positions.copy()
            
            # Strategy: Select atoms that form stable, well-separated geometry
            # 1. Prioritize backbone/ring atoms over flexible side chains
            # 2. Ensure good geometric separation
            # 3. Check for stability across a few frames
            
            stable_candidates = self._find_stable_reference_candidates(target, positions, n_atoms=n_atoms)
            if stable_candidates is None:
                print(f"⚠️ Could not find stable reference atom candidates")
                return None
                
            return stable_candidates
            
        elif reference_atoms == 'heavy':
            # Select heaviest atoms (non-hydrogen)
            heavy_atoms = []
            for i, atom in enumerate(target):
                if atom.element != 'H':  # Skip hydrogens
                    # Approximate atomic masses for common elements
                    mass_map = {'C': 12, 'N': 14, 'O': 16, 'S': 32, 'P': 31, 'F': 19, 'CL': 35, 'BR': 80}
                    mass = mass_map.get(atom.element, 10)  # Default weight for unknown elements
                    heavy_atoms.append((i, mass))
                    
            if len(heavy_atoms) < 3:
                print(f"⚠️ Found only {len(heavy_atoms)} heavy atoms, need at least 3 for reference frame")
                return None
                
            # Sort by mass and take top n_atoms
            heavy_atoms.sort(key=lambda x: x[1], reverse=True)
            return [atom[0] for atom in heavy_atoms[:min(n_atoms, len(heavy_atoms))]]
            
        else:
            print(f"⚠️ Unknown reference_atoms method: {reference_atoms}")
            return None

    def _compute_molecular_frame_vector(self, ion_pos, target_com, current_ref_positions, 
                                      reference_frame, box_dimensions):
        """
        Compute ion position in rotation-invariant molecular coordinate system.
        
        This transforms ion positions to molecular coordinates that remain consistent
        even when the molecule rotates, enabling accurate spatial reconstruction.
        
        Parameters
        ----------
        ion_pos : array
            Current ion position
        target_com : array  
            Target center of mass
        current_ref_positions : array
            Current reference atom positions (3x3)
        reference_frame : array
            Reference frame atom positions from frame 0 (3x3)
        box_dimensions : array
            Box dimensions for PBC handling
            
        Returns
        -------
        molecular_vector : array
            Ion position in rotation-invariant molecular coordinate system
        """
        try:
            # Apply PBC correction to ion position relative to COM
            ion_relative_to_com = ion_pos - target_com
            if box_dimensions is not None:
                box = box_dimensions[:3]
                for i in range(3):
                    if ion_relative_to_com[i] > box[i] * 0.5:
                        ion_relative_to_com[i] -= box[i]
                    elif ion_relative_to_com[i] < -box[i] * 0.5:
                        ion_relative_to_com[i] += box[i]
            
            # Determine method based on number of reference atoms
            method = 'pca' if len(reference_frame) > 3 else 'gram_schmidt'
            
            # Build molecular coordinate systems
            reference_result = self._build_coordinate_system(reference_frame, method=method)
            current_result = self._build_coordinate_system(current_ref_positions, method=method)
            
            if method == 'pca':
                reference_coords, ref_mean = reference_result
                current_coords, curr_mean = current_result
            else:
                reference_coords, _ = reference_result
                current_coords, _ = current_result
            
            if reference_coords is None or current_coords is None:
                # Fallback to COM-relative positioning if coordinate system fails
                return ion_relative_to_com.copy()
            
            # Express ion position in current molecular coordinate system
            # The coordinate system is built from reference atoms, but we need consistency
            # with the target COM as origin
            ion_in_molecular_coords = current_coords.T @ ion_relative_to_com
            
            return ion_in_molecular_coords
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # Fallback to COM-relative positioning if transformation fails
            ion_relative_to_com = ion_pos - target_com
            if box_dimensions is not None:
                box = box_dimensions[:3]
                for i in range(3):
                    if ion_relative_to_com[i] > box[i] * 0.5:
                        ion_relative_to_com[i] -= box[i]
                    elif ion_relative_to_com[i] < -box[i] * 0.5:
                        ion_relative_to_com[i] += box[i]
            return ion_relative_to_com.copy()
    
    def _find_stable_reference_candidates(self, target, initial_positions, n_atoms=3):
        """
        Find N reference atoms that form a stable, well-conditioned coordinate system.
        
        Strategy:
        1. Prioritize non-hydrogen atoms (more stable)
        2. Look for atoms in rings or backbone (less flexible)
        3. Ensure good geometric separation and non-collinearity (for n_atoms=3)
        4. For n_atoms>3 (PCA mode): ensure good spread across molecular geometry
        5. Check stability across multiple frames
        
        Parameters
        ----------
        target : MDAnalysis.AtomGroup
            Target molecule
        initial_positions : array
            Initial atom positions
        n_atoms : int
            Number of reference atoms to select (default: 3)
        
        Returns
        -------
        ref_indices : list or None
            List of n_atoms atom indices
        """
        # Get non-hydrogen atoms first (more stable than hydrogens)
        heavy_atom_indices = []
        for i, atom in enumerate(target):
            if atom.element != 'H':
                heavy_atom_indices.append(i)
                
        if len(heavy_atom_indices) < n_atoms:
            print(f"⚠️ Only {len(heavy_atom_indices)} non-hydrogen atoms available, need {n_atoms}")
            # Fall back to all atoms if necessary
            heavy_atom_indices = list(range(len(target)))
            
        if len(heavy_atom_indices) < n_atoms:
            return None
            
        # Calculate distances between all heavy atoms
        positions = initial_positions[heavy_atom_indices]
        from scipy.spatial.distance import pdist, squareform
        dist_matrix = squareform(pdist(positions))
        
        if n_atoms == 3:
            # Original triplet selection logic
            best_triplet = None
            best_score = -1
            
            # Evaluate all possible triplets of heavy atoms
            from itertools import combinations
            for triplet_indices in combinations(range(len(heavy_atom_indices)), 3):
                i, j, k = triplet_indices
                
                # Map back to original target indices
                orig_i, orig_j, orig_k = [heavy_atom_indices[idx] for idx in [i, j, k]]
                
                # Calculate geometric quality score
                score = self._evaluate_reference_triplet_quality(
                    positions[i], positions[j], positions[k]
                )
                
                if score > best_score:
                    best_score = score
                    best_triplet = [orig_i, orig_j, orig_k]
                    
            if best_triplet is None or best_score < 0.1:
                print(f"⚠️ No suitable reference atom triplet found (best score: {best_score:.3f})")
                return None
                
            print(f"✓ Selected {n_atoms} reference atoms {best_triplet} with quality score {best_score:.3f}")
            return best_triplet
        
        else:
            # For PCA mode (n_atoms > 3): Select atoms with maximum spread
            # Strategy: Pick atoms that are well-distributed across the molecule
            
            # Start with the atom pair with maximum distance
            max_pair_dist = 0
            best_pair = (0, 1)
            for i in range(len(heavy_atom_indices)):
                for j in range(i+1, len(heavy_atom_indices)):
                    if dist_matrix[i, j] > max_pair_dist:
                        max_pair_dist = dist_matrix[i, j]
                        best_pair = (i, j)
            
            # Initialize selected atoms with this pair
            selected_local_indices = list(best_pair)
            
            # Iteratively add atoms that maximize minimum distance to selected set
            while len(selected_local_indices) < min(n_atoms, len(heavy_atom_indices)):
                best_candidate = None
                best_min_dist = -1
                
                for candidate in range(len(heavy_atom_indices)):
                    if candidate in selected_local_indices:
                        continue
                    
                    # Calculate minimum distance to already selected atoms
                    min_dist_to_selected = min([dist_matrix[candidate, sel] for sel in selected_local_indices])
                    
                    if min_dist_to_selected > best_min_dist:
                        best_min_dist = min_dist_to_selected
                        best_candidate = candidate
                
                if best_candidate is not None:
                    selected_local_indices.append(best_candidate)
                else:
                    break  # No more suitable candidates
            
            # Map back to original target indices
            selected_atoms = [heavy_atom_indices[idx] for idx in selected_local_indices]
            
            # Calculate spread metric for quality reporting
            selected_positions = initial_positions[selected_atoms]
            centroid = np.mean(selected_positions, axis=0)
            avg_distance_from_center = np.mean([np.linalg.norm(pos - centroid) for pos in selected_positions])
            
            print(f"✓ Selected {len(selected_atoms)} reference atoms {selected_atoms} for PCA")
            print(f"   Average distance from centroid: {avg_distance_from_center:.2f} Å")
            
            return selected_atoms
        
    def _evaluate_reference_triplet_quality(self, pos1, pos2, pos3):
        """
        Evaluate the geometric quality of a reference atom triplet.
        
        Returns a score (0-1) where higher is better:
        - Good separation between atoms
        - Non-collinear arrangement  
        - Well-conditioned triangle
        """
        # Calculate side lengths
        d12 = np.linalg.norm(pos2 - pos1)
        d13 = np.linalg.norm(pos3 - pos1)  
        d23 = np.linalg.norm(pos3 - pos2)
        
        # Minimum separation requirement
        min_distance = 1.0  # Angstroms
        if min(d12, d13, d23) < min_distance:
            return 0.0  # Too close together
            
        # Calculate triangle area using cross product
        vec12 = pos2 - pos1
        vec13 = pos3 - pos1
        cross_product = np.cross(vec12, vec13)
        area = 0.5 * np.linalg.norm(cross_product)
        
        # Calculate perimeter for normalization
        perimeter = d12 + d13 + d23
        
        # Isoperimetric quotient (measure of how close to equilateral)
        # For equilateral triangle: 4π * area / perimeter² = π/√3 ≈ 1.814
        if perimeter > 0:
            shape_factor = (4 * np.pi * area) / (perimeter * perimeter)
            shape_score = min(1.0, shape_factor / 1.814)
        else:
            shape_score = 0.0
            
        # Size score (prefer larger triangles for stability)
        size_score = min(1.0, area / 10.0)  # Normalize to 10 Ų
        
        # Combined score  
        total_score = 0.7 * shape_score + 0.3 * size_score
        
        return total_score

    def _build_coordinate_system(self, ref_positions, method='gram_schmidt'):
        """
        Build orthonormal coordinate system from reference atom positions.
        
        Parameters
        ----------
        ref_positions : array
            Reference atom positions (Nx3, where N>=3)
        method : str
            'gram_schmidt': Use 3 atoms with Gram-Schmidt (default, backward compatible)
            'pca': Use PCA on all N atoms (better for planar molecules)
            
        Returns
        -------
        coord_system : array or None
            3x3 matrix where columns are orthonormal basis vectors, or None if failed
        mean_position : array or None
            Mean position of reference atoms (only for PCA method)
        """
        if method == 'pca':
            return self._build_coordinate_system_pca(ref_positions)
        else:
            # Legacy Gram-Schmidt method (uses only first 3 atoms)
            result = self._build_coordinate_system_gram_schmidt(ref_positions[:3])
            if result is not None:
                return result, None  # No mean for Gram-Schmidt
            return None, None
    
    def _build_coordinate_system_gram_schmidt(self, ref_positions):
        """
        Build orthonormal coordinate system from 3 reference atom positions using Gram-Schmidt.
        
        Parameters
        ----------
        ref_positions : array
            Reference atom positions (3x3)
            
        Returns
        -------
        coord_system : array or None
            3x3 matrix where columns are orthonormal basis vectors, or None if failed
        """
        try:
            # Center positions at first atom
            centered_positions = ref_positions - ref_positions[0]
            
            # First basis vector: from atom 1 to atom 2
            v1 = centered_positions[1]
            if np.linalg.norm(v1) < 1e-6:
                return None
            v1_norm = v1 / np.linalg.norm(v1)
            
            # Second basis vector: orthogonalize atom 3 direction to v1
            v2 = centered_positions[2]
            v2_perp = v2 - np.dot(v2, v1_norm) * v1_norm
            
            # Strengthen collinearity detection
            v2_perp_norm = np.linalg.norm(v2_perp)
            if v2_perp_norm < 1e-4:  # Stricter threshold for numerical stability
                return None  # Atoms are collinear
            v2_norm = v2_perp / v2_perp_norm
            
            # Third basis vector: cross product for right-handed system
            v3_norm = np.cross(v1_norm, v2_norm)
            
            # Additional validation: ensure cross product is well-defined
            v3_norm_magnitude = np.linalg.norm(v3_norm)
            if v3_norm_magnitude < 1e-4:
                return None  # Degenerate coordinate system
            v3_norm = v3_norm / v3_norm_magnitude
            
            # Build coordinate system matrix
            coord_system = np.column_stack([v1_norm, v2_norm, v3_norm])
            
            # Comprehensive validation of coordinate system quality
            # 1. Check orthonormality
            orthogonality_matrix = coord_system.T @ coord_system
            if not np.allclose(orthogonality_matrix, np.eye(3), atol=1e-4):
                return None  # Not sufficiently orthonormal
                
            # 2. Check determinant (should be +1 for right-handed system)
            det = np.linalg.det(coord_system)
            if abs(det - 1.0) > 1e-4:
                return None  # Improper coordinate system
                
            # 3. Check condition number (measure of numerical stability)
            cond_number = np.linalg.cond(coord_system)
            if cond_number > 1e6:  # Too ill-conditioned
                return None  # Numerically unstable
                
            return coord_system
            
        except (np.linalg.LinAlgError, ValueError):
            return None
    
    def _build_coordinate_system_pca(self, ref_positions):
        """
        Build orthonormal coordinate system using PCA on multiple reference atoms.
        
        Better for planar molecules (like quinolone) where simple 3-atom frames
        can have inconsistent orientations. PCA finds principal axes of the
        molecular geometry, giving more standardized coordinate frames.
        
        Parameters
        ----------
        ref_positions : array
            Reference atom positions (Nx3, N>=3)
            
        Returns
        -------
        coord_system : array or None
            3x3 matrix where columns are principal component vectors (PC1, PC2, PC3)
        mean_position : array or None
            Mean position of all reference atoms (PCA center)
        """
        try:
            from sklearn.decomposition import PCA
            
            if len(ref_positions) < 3:
                return None, None
            
            # Center the positions
            mean_pos = np.mean(ref_positions, axis=0)
            centered = ref_positions - mean_pos
            
            # Perform PCA
            pca = PCA(n_components=3)
            pca.fit(centered)
            
            # Get principal components (already orthonormal)
            pc1 = pca.components_[0]  # Direction of maximum variance
            pc2 = pca.components_[1]  # Second direction
            pc3 = pca.components_[2]  # Third direction
            
            # Build coordinate system matrix
            coord_system = np.column_stack([pc1, pc2, pc3])
            
            # Validate determinant (ensure right-handed system)
            det = np.linalg.det(coord_system)
            if det < 0:
                # Flip third axis to make right-handed
                coord_system[:, 2] *= -1
            
            # Validate the coordinate system
            if not np.allclose(coord_system.T @ coord_system, np.eye(3), atol=1e-4):
                return None, None
            
            if abs(np.linalg.det(coord_system) - 1.0) > 1e-4:
                return None, None
            
            # Check explained variance ratio (quality metric)
            # For well-defined geometry, first PC should explain substantial variance
            if pca.explained_variance_ratio_[0] < 0.1:  # Less than 10% variance
                return None, None  # Geometry is too spherical for stable frame
            
            return coord_system, mean_pos
            
        except (ImportError, np.linalg.LinAlgError, ValueError) as e:
            # sklearn not available or PCA failed
            return None, None

    def _analyze_ion_binding(self, target, ion_atoms, cutoff, step):
        '''Analyze binding of specific ion type to target'''
        
        if len(ion_atoms) == 0:
            return {'binding_events': [], 'average_binding': 0, 'occupancy': 0}
        
        binding_events = []
        
        for ts in self.universe.trajectory[::step]:
            dist_matrix = distances.distance_array(target.positions, 
                                                 ion_atoms.positions,
                                                 box=ts.dimensions)
            
            # Count bound ions
            bound_ions = (dist_matrix <= cutoff).sum()
            binding_events.append(bound_ions)
        
        return {
            'binding_events': np.array(binding_events),
            'average_binding': np.mean(binding_events),
            'max_binding': np.max(binding_events),
            'occupancy': np.mean(np.array(binding_events) > 0),
            'std_binding': np.std(binding_events)
        }

    def _calculate_peak_volume(self, boundary, geometry_type='auto', volume_units='angstrom3'):
        """
        Calculate volume for peak regions with different geometries
        
        Parameters
        ----------
        boundary : tuple
            (start, end) distance boundaries in Ångströms
        geometry_type : str
            Type of geometry: 'auto', 'spherical', 'spherical_shell', 'sphere'
        volume_units : str
            Output units: 'angstrom3' or 'nm3'
            
        Returns
        -------
        volume_data : dict
            Dictionary containing volume and geometry metadata
        """
        import math
        
        start, end = boundary
        
        # Auto-detect geometry type
        if geometry_type == 'auto':
            if start > 0:
                geometry_type = 'spherical_shell'
            else:
                geometry_type = 'sphere'
        
        # Calculate volume in Ångström³
        if geometry_type in ['spherical_shell', 'spherical']:
            if start == 0:
                # Complete sphere from center
                volume_a3 = (4/3) * math.pi * (end**3)
                actual_geometry = 'sphere'
            else:
                # Spherical shell (annular region)
                volume_a3 = (4/3) * math.pi * (end**3 - start**3)
                actual_geometry = 'spherical_shell'
        elif geometry_type == 'sphere':
            # Force complete sphere calculation
            volume_a3 = (4/3) * math.pi * (end**3)
            actual_geometry = 'sphere'
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")
        
        # Convert units if requested
        if volume_units == 'nm3':
            # 1 nm = 10 Å, so 1 nm³ = 1000 Å³
            volume = volume_a3 / 1000.0
            unit_symbol = 'nm³'
        elif volume_units == 'angstrom3':
            volume = volume_a3
            unit_symbol = 'Å³'
        else:
            raise ValueError(f"Unsupported volume units: {volume_units}")
        
        return {
            'volume': volume,
            'volume_units': volume_units,
            'unit_symbol': unit_symbol,
            'volume_angstrom3': volume_a3,  # Always keep original Å³ value
            'geometry': {
                'type': actual_geometry,
                'r_inner': start,
                'r_outer': end,
                'boundary': boundary
            }
        }

    def _find_rdf_boundary_label(self, target_label, ion_name, rdf_boundaries):
        '''
        Find matching RDF boundary label for target-ion pair.
        Tries multiple naming conventions.
        '''
        # Possible label formats to try
        possible_labels = [
            f"{target_label}-{ion_name}",
            f"{target_label}-{ion_name.lower()}",
            f"{target_label}-{ion_name.upper()}",
            f"{target_label.lower()}-{ion_name}",
            f"{target_label.lower()}-{ion_name.lower()}",
            f"{target_label.upper()}-{ion_name.upper()}",
        ]
        
        for label in possible_labels:
            if label in rdf_boundaries:
                return label
        
        return None

    def _analyze_ion_binding_with_peaks(self, target, ion_atoms, boundaries, peaks_to_analyze, step,
                                       calculate_volumes=True, volume_units='angstrom3', geometry_type='auto'):
        '''
        Analyze ion binding using RDF-derived peak boundaries.
        
        Parameters
        ----------
        target : AtomGroup
            Target atoms
        ion_atoms : AtomGroup 
            Ion atoms to analyze
        boundaries : dict
            Shell boundaries {shell_name: (start, end)}
        peaks_to_analyze : list
            List of peak names to analyze (e.g., ['P1', 'P2'])
        step : int
            Trajectory step
            
        Returns
        -------
        binding_data : dict
            Enhanced binding data with peak-specific analysis
        '''
        
        if len(ion_atoms) == 0:
            return {'binding_events': [], 'average_binding': 0, 'occupancy': 0}
        
        # Initialize results structure
        binding_data = {
            'binding_events': [],
            'average_binding': 0,
            'max_binding': 0,
            'occupancy': 0,
            'std_binding': 0,
            'peak_analysis': {}
        }
        
        # Validate peaks exist in boundaries
        valid_peaks = []
        for peak in peaks_to_analyze:
            if peak in boundaries:
                valid_peaks.append(peak)
                
                # Calculate volume for this peak if requested
                volume_data = None
                if calculate_volumes:
                    start, end = boundaries[peak]
                    end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                    try:
                        volume_data = self._calculate_peak_volume(
                            boundaries[peak], geometry_type, volume_units
                        )
                        print(f"      {peak} ({start:.2f}-{end_str} Å): "
                              f"{volume_data['volume']:.1f} {volume_data['unit_symbol']}")
                    except Exception as e:
                        print(f"      Warning: Could not calculate volume for {peak}: {e}")
                        volume_data = None
                
                binding_data['peak_analysis'][peak] = {
                    'binding_events': [],
                    'average_binding': 0,
                    'cutoff_range': boundaries[peak],
                    'volume_data': volume_data
                }
            else:
                print(f"    Warning: Peak '{peak}' not found in boundaries, skipping")
        
        if not valid_peaks:
            print(f"    No valid peaks found, falling back to traditional analysis")
            return self._analyze_ion_binding(target, ion_atoms, 3.5, step)
        
        print(f"    Analyzing peaks: {valid_peaks}")
        
        # Analyze trajectory
        total_binding_per_frame = []
        
        for ts in tqdm(self.universe.trajectory[::step], 
                      desc=f"    Peak analysis", leave=False):
            
            dist_matrix = distances.distance_array(target.positions, 
                                                 ion_atoms.positions,
                                                 box=ts.dimensions)
            
            frame_total = 0
            
            # Analyze each peak
            for peak in valid_peaks:
                start, end = boundaries[peak]
                
                # Count ions in this peak's range
                if np.isinf(end):
                    # Handle infinite boundaries (like Bulk)
                    peak_bound = (dist_matrix >= start).sum()
                else:
                    peak_bound = ((dist_matrix >= start) & (dist_matrix <= end)).sum()
                
                binding_data['peak_analysis'][peak]['binding_events'].append(peak_bound)
                frame_total += peak_bound
            
            total_binding_per_frame.append(frame_total)
        
        # Calculate overall statistics
        binding_data['binding_events'] = np.array(total_binding_per_frame)
        binding_data['average_binding'] = np.mean(total_binding_per_frame)
        binding_data['max_binding'] = np.max(total_binding_per_frame)
        binding_data['occupancy'] = np.mean(np.array(total_binding_per_frame) > 0)
        binding_data['std_binding'] = np.std(total_binding_per_frame)
        
        # Calculate peak-specific statistics
        for peak in valid_peaks:
            peak_events = np.array(binding_data['peak_analysis'][peak]['binding_events'])
            
            # Basic statistics
            avg_binding = np.mean(peak_events)
            max_binding = np.max(peak_events)
            occupancy = np.mean(peak_events > 0)
            std_binding = np.std(peak_events)
            
            # Volume-normalized density calculations
            volume_density = None
            volume_density_std = None
            if (calculate_volumes and 
                binding_data['peak_analysis'][peak]['volume_data'] is not None):
                
                volume_data = binding_data['peak_analysis'][peak]['volume_data']
                volume = volume_data['volume']
                
                if volume > 0:
                    # Calculate density (ions/frame/volume)
                    density_per_frame = peak_events / volume
                    volume_density = np.mean(density_per_frame)
                    volume_density_std = np.std(density_per_frame)
            
            # Update peak analysis with all metrics
            binding_data['peak_analysis'][peak].update({
                'binding_events': peak_events,
                'average_binding': avg_binding,
                'max_binding': max_binding,
                'occupancy': occupancy,
                'std_binding': std_binding,
                'volume_density': volume_density,
                'volume_density_std': volume_density_std
            })
            
            # Print peak statistics
            start, end = boundaries[peak]
            end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
            print(f"      {peak} ({start:.2f}-{end_str} Å): {avg_binding:.2f} ions/frame", end='')
            
            if volume_density is not None:
                volume_units_str = binding_data['peak_analysis'][peak]['volume_data']['unit_symbol']
                print(f", {volume_density:.4f} ions/frame/{volume_units_str}")
            else:
                print()
        
        return binding_data

    def ion_competition_analysis(self, target_sel, cutoff=5.0, step=1):
        '''
        Analyze competition between different ion types for binding sites.
        
        Parameters
        ----------
        target_sel : str
            Selection for target molecules
        cutoff : float
            Distance cutoff for competition analysis, default=5.0 Å
        step : int
            Trajectory step, default=1
            
        Returns
        -------
        competition_results : dict
            Ion competition analysis results
        '''
        
        target = self.universe.select_atoms(target_sel)
        print(f"Analyzing ion competition around {len(target)} target atoms...")
        
        results = {
            'cation_competition': {},
            'anion_competition': {},
            'mixed_competition': [],
            'dominance_analysis': {}
        }
        
        # Analyze cation competition
        cation_names = list(self.cation_types.keys())
        if len(cation_names) > 1:
            for i, cat1 in enumerate(cation_names):
                for cat2 in cation_names[i+1:]:
                    competition = self._analyze_pairwise_competition(
                        target, self.cation_types[cat1], self.cation_types[cat2], 
                        cutoff, step, f"{cat1}_vs_{cat2}"
                    )
                    results['cation_competition'][f"{cat1}_vs_{cat2}"] = competition
        
        # Analyze anion competition
        anion_names = list(self.anion_types.keys())
        if len(anion_names) > 1:
            for i, an1 in enumerate(anion_names):
                for an2 in anion_names[i+1:]:
                    competition = self._analyze_pairwise_competition(
                        target, self.anion_types[an1], self.anion_types[an2], 
                        cutoff, step, f"{an1}_vs_{an2}"
                    )
                    results['anion_competition'][f"{an1}_vs_{an2}"] = competition
        
        # Calculate dominance analysis
        results['dominance_analysis'] = self._calculate_ion_dominance(target, cutoff, step)
        
        return results

    def _analyze_pairwise_competition(self, target, ions1, ions2, cutoff, step, pair_name):
        '''Analyze competition between two specific ion types'''
        
        if len(ions1) == 0 or len(ions2) == 0:
            return {'competition_index': 0, 'exclusion_events': 0}
        
        competition_events = []
        exclusion_events = 0
        
        for ts in self.universe.trajectory[::step]:
            # Find ions within cutoff of target
            dist1 = distances.distance_array(target.positions, ions1.positions, box=ts.dimensions)
            dist2 = distances.distance_array(target.positions, ions2.positions, box=ts.dimensions)
            
            close1 = (dist1 <= cutoff).sum()
            close2 = (dist2 <= cutoff).sum()
            
            # Competition index: how often both ion types are present
            if close1 > 0 and close2 > 0:
                competition_events.append(1)
            else:
                competition_events.append(0)
            
            # Exclusion: when one type excludes the other
            if (close1 > 0 and close2 == 0) or (close1 == 0 and close2 > 0):
                exclusion_events += 1
        
        return {
            'competition_index': np.mean(competition_events),
            'exclusion_events': exclusion_events,
            'total_frames': len(competition_events)
        }

    def _calculate_ion_selectivity(self, binding_results):
        '''Calculate selectivity indices between different ion types
        
        Returns both metrics:
        - fraction: ion1 / (ion1 + ion2) - ranges 0 to 1, where 0.5 = equal
        - ratio: ion1 / ion2 - ranges 0 to inf, where 1.0 = equal, >1 = ion1 preferred
        '''
        
        selectivity = {}
        
        # Cation selectivity
        cation_names = list(binding_results['cation_binding'].keys())
        if len(cation_names) > 1:
            for i, cat1 in enumerate(cation_names):
                for cat2 in cation_names[i+1:]:
                    binding1 = binding_results['cation_binding'][cat1]['average_binding']
                    binding2 = binding_results['cation_binding'][cat2]['average_binding']
                    
                    pair_key = f"{cat1}_over_{cat2}"
                    selectivity[pair_key] = {}
                    
                    if binding1 + binding2 > 0:
                        # Fraction: ranges 0-1, where 0.5 = equal preference
                        selectivity[pair_key]['fraction'] = binding1 / (binding1 + binding2)
                    else:
                        selectivity[pair_key]['fraction'] = 0.5
                    
                    if binding2 > 0:
                        # Ratio: >1 means ion1 preferred, <1 means ion2 preferred
                        selectivity[pair_key]['ratio'] = binding1 / binding2
                    else:
                        selectivity[pair_key]['ratio'] = float('inf') if binding1 > 0 else 1.0
        
        # Anion selectivity
        anion_names = list(binding_results['anion_binding'].keys())
        if len(anion_names) > 1:
            for i, an1 in enumerate(anion_names):
                for an2 in anion_names[i+1:]:
                    binding1 = binding_results['anion_binding'][an1]['average_binding']
                    binding2 = binding_results['anion_binding'][an2]['average_binding']
                    
                    pair_key = f"{an1}_over_{an2}"
                    selectivity[pair_key] = {}
                    
                    if binding1 + binding2 > 0:
                        # Fraction: ranges 0-1, where 0.5 = equal preference
                        selectivity[pair_key]['fraction'] = binding1 / (binding1 + binding2)
                    else:
                        selectivity[pair_key]['fraction'] = 0.5
                    
                    if binding2 > 0:
                        # Ratio: >1 means ion1 preferred, <1 means ion2 preferred
                        selectivity[pair_key]['ratio'] = binding1 / binding2
                    else:
                        selectivity[pair_key]['ratio'] = float('inf') if binding1 > 0 else 1.0
        
        return selectivity

    def _calculate_ion_dominance(self, target, cutoff, step):
        '''Calculate which ion types dominate in different regions'''
        
        dominance = {}
        
        for ts in self.universe.trajectory[::step]:
            frame_dominance = {}
            
            # Calculate local concentrations for each ion type
            for ion_name, ion_atoms in {**self.cation_types, **self.anion_types}.items():
                if len(ion_atoms) > 0:
                    dist_matrix = distances.distance_array(target.positions, 
                                                         ion_atoms.positions,
                                                         box=ts.dimensions)
                    local_concentration = (dist_matrix <= cutoff).sum()
                    frame_dominance[ion_name] = local_concentration
            
            # Find dominant ion type
            if frame_dominance:
                dominant_ion = max(frame_dominance, key=frame_dominance.get)
                if dominant_ion not in dominance:
                    dominance[dominant_ion] = 0
                dominance[dominant_ion] += 1
        
        # Convert to percentages
        total_frames = len(self.universe.trajectory[::step])
        for ion_name in dominance:
            dominance[ion_name] = (dominance[ion_name] / total_frames) * 100
        
        return dominance

    def specific_ion_pair_analysis(self, cation_sel, anion_sel, cutoff=4.0, step=1):
        '''
        Analyze specific ion pair formation (e.g., Na-Cl, Mg-SO4).
        
        Parameters
        ----------
        cation_sel : str
            Selection for specific cation type
        anion_sel : str
            Selection for specific anion type
        cutoff : float
            Distance cutoff for ion pair formation, default=4.0 Å
        step : int
            Trajectory step, default=1
            
        Returns
        -------
        pair_results : dict
            Ion pair analysis results
        '''
        
        cations = self.universe.select_atoms(cation_sel)
        anions = self.universe.select_atoms(anion_sel)
        
        if len(cations) == 0 or len(anions) == 0:
            print(f"Warning: No ions found for pair analysis")
            return None
        
        print(f"Analyzing {len(cations)} cations and {len(anions)} anions for pairing...")
        
        pair_formation = []
        contact_pairs = []
        
        for ts in tqdm(self.universe.trajectory[::step], desc="Analyzing ion pairs"):
            dist_matrix = distances.distance_array(cations.positions, 
                                                 anions.positions,
                                                 box=ts.dimensions)
            
            # Find ion pairs within cutoff
            pairs = np.where(dist_matrix <= cutoff)
            n_pairs = len(pairs[0])
            pair_formation.append(n_pairs)
            
            # Store individual pair information
            frame_pairs = []
            for i in range(n_pairs):
                cation_idx = pairs[0][i]
                anion_idx = pairs[1][i]
                distance = dist_matrix[cation_idx, anion_idx]
                frame_pairs.append({
                    'cation_id': cation_idx,
                    'anion_id': anion_idx,
                    'distance': distance
                })
            contact_pairs.append(frame_pairs)
        
        results = {
            'pair_formation_timeline': np.array(pair_formation),
            'average_pairs': np.mean(pair_formation),
            'max_pairs': np.max(pair_formation),
            'pair_probability': np.mean(np.array(pair_formation) > 0),
            'contact_pairs_detailed': contact_pairs,
            'total_cations': len(cations),
            'total_anions': len(anions)
        }
        
        return results

    def counterion_condensation_analysis(self, polymer_sel, charged_sites_sel, cutoff=3.5, step=1):
        '''
        Analyze counterion condensation around charged polymer chains.
        
        Parameters
        ----------
        polymer_sel : str
            Selection for polymer chains
        charged_sites_sel : str
            Selection for charged sites on polymer (e.g., 'name COO')
        cutoff : float
            Distance cutoff for condensation, default=3.5 Å
        step : int
            Trajectory step, default=1
            
        Returns
        -------
        condensation_results : dict
            Counterion condensation analysis results
        '''
        
        polymer = self.universe.select_atoms(polymer_sel)
        charged_sites = self.universe.select_atoms(f'{polymer_sel} and {charged_sites_sel}')
        
        if len(charged_sites) == 0:
            print("Warning: No charged sites found on polymer")
            return None
        
        print(f"Analyzing counterion condensation around {len(charged_sites)} charged sites...")
        
        results = {
            'condensation_per_site': {},
            'total_condensation': [],
            'condensation_efficiency': {}
        }
        
        # Analyze condensation for each ion type
        for ion_name, ion_atoms in {**self.cation_types, **self.anion_types}.items():
            if len(ion_atoms) == 0:
                continue
                
            condensation_data = []
            
            for ts in tqdm(self.universe.trajectory[::step], desc=f"Analyzing {ion_name}"):
                dist_matrix = distances.distance_array(charged_sites.positions, 
                                                     ion_atoms.positions,
                                                     box=ts.dimensions)
                
                # Count condensed ions per charged site
                condensed_per_site = (dist_matrix <= cutoff).sum(axis=1)
                total_condensed = condensed_per_site.sum()
                
                condensation_data.append({
                    'total_condensed': total_condensed,
                    'per_site': condensed_per_site,
                    'efficiency': total_condensed / len(ion_atoms) if len(ion_atoms) > 0 else 0
                })
            
            results['condensation_per_site'][ion_name] = condensation_data
            
            # Calculate averages
            avg_total = np.mean([data['total_condensed'] for data in condensation_data])
            avg_efficiency = np.mean([data['efficiency'] for data in condensation_data])
            
            results['condensation_efficiency'][ion_name] = {
                'average_total': avg_total,
                'average_efficiency': avg_efficiency,
                'per_charged_site': avg_total / len(charged_sites)
            }
        
        return results

    def ion_selectivity_analysis(self, target_sel, reference_distance=5.0, step=1):
        '''
        Analyze ion selectivity preferences of target molecules.
        
        Parameters
        ----------
        target_sel : str
            Selection for target molecules
        reference_distance : float
            Reference distance for calculating local concentrations, default=5.0 Å
        step : int
            Trajectory step, default=1
            
        Returns
        -------
        selectivity_results : dict
            Ion selectivity analysis results
        '''
        
        target = self.universe.select_atoms(target_sel)
        
        print(f"Analyzing ion selectivity around {len(target)} target atoms...")
        
        results = {
            'local_concentrations': {},
            'bulk_concentrations': {},
            'enrichment_factors': {},
            'selectivity_coefficients': {}
        }
        
        # Calculate bulk concentrations (reference)
        box_volume = np.mean([ts.volume for ts in self.universe.trajectory[::step]])
        
        for ion_name, ion_atoms in {**self.cation_types, **self.anion_types}.items():
            if len(ion_atoms) > 0:
                bulk_conc = len(ion_atoms) / box_volume
                results['bulk_concentrations'][ion_name] = bulk_conc
        
        # Calculate local concentrations around target
        for ion_name, ion_atoms in {**self.cation_types, **self.anion_types}.items():
            if len(ion_atoms) == 0:
                continue
                
            local_counts = []
            
            for ts in self.universe.trajectory[::step]:
                dist_matrix = distances.distance_array(target.positions, 
                                                     ion_atoms.positions,
                                                     box=ts.dimensions)
                
                # Count ions within reference distance
                local_count = (dist_matrix <= reference_distance).sum()
                local_counts.append(local_count)
            
            # Calculate local concentration
            shell_volume = (4/3) * np.pi * reference_distance**3 * len(target)
            avg_local_count = np.mean(local_counts)
            local_conc = avg_local_count / shell_volume
            
            results['local_concentrations'][ion_name] = local_conc
            
            # Calculate enrichment factor
            if results['bulk_concentrations'][ion_name] > 0:
                enrichment = local_conc / results['bulk_concentrations'][ion_name]
                results['enrichment_factors'][ion_name] = enrichment
        
        # Calculate selectivity coefficients between ion pairs
        ion_names = list(results['enrichment_factors'].keys())
        for i, ion1 in enumerate(ion_names):
            for ion2 in ion_names[i+1:]:
                if ion1 in results['enrichment_factors'] and ion2 in results['enrichment_factors']:
                    selectivity = (results['enrichment_factors'][ion1] / 
                                 results['enrichment_factors'][ion2])
                    results['selectivity_coefficients'][f"{ion1}_over_{ion2}"] = selectivity
        
        return results

    def membrane_ion_distribution(self, membrane_center_z=0.0, leaflet_thickness=20.0, step=10):
        '''
        Analyze ion distribution across membrane leaflets.
        
        Parameters
        ----------
        membrane_center_z : float
            Z-coordinate of membrane center, default=0.0
        leaflet_thickness : float
            Thickness of each leaflet for analysis, default=20.0 Å
        step : int
            Trajectory step, default=10
            
        Returns
        -------
        distribution_results : dict
            Membrane ion distribution results
        '''
        
        print("Analyzing ion distribution across membrane leaflets...")
        
        results = {
            'upper_leaflet': {},
            'lower_leaflet': {},
            'asymmetry_index': {},
            'z_profiles': {}
        }
        
        # Define leaflet regions
        upper_bounds = (membrane_center_z, membrane_center_z + leaflet_thickness/2)
        lower_bounds = (membrane_center_z - leaflet_thickness/2, membrane_center_z)
        
        for ion_name, ion_atoms in {**self.cation_types, **self.anion_types}.items():
            if len(ion_atoms) == 0:
                continue
                
            upper_counts = []
            lower_counts = []
            z_positions = []
            
            for ts in tqdm(self.universe.trajectory[::step], desc=f"Analyzing {ion_name}"):
                z_coords = ion_atoms.positions[:, 2]  # Z-coordinates
                z_positions.extend(z_coords)
                
                # Count ions in each leaflet
                upper_count = np.sum((z_coords >= upper_bounds[0]) & (z_coords <= upper_bounds[1]))
                lower_count = np.sum((z_coords >= lower_bounds[0]) & (z_coords <= lower_bounds[1]))
                
                upper_counts.append(upper_count)
                lower_counts.append(lower_count)
            
            results['upper_leaflet'][ion_name] = {
                'counts': upper_counts,
                'average': np.mean(upper_counts),
                'std': np.std(upper_counts)
            }
            
            results['lower_leaflet'][ion_name] = {
                'counts': lower_counts,
                'average': np.mean(lower_counts),
                'std': np.std(lower_counts)
            }
            
            # Calculate asymmetry index
            total_upper = np.mean(upper_counts)
            total_lower = np.mean(lower_counts)
            if total_upper + total_lower > 0:
                asymmetry = (total_upper - total_lower) / (total_upper + total_lower)
                results['asymmetry_index'][ion_name] = asymmetry
            
            # Store z-profiles for plotting
            results['z_profiles'][ion_name] = np.array(z_positions)
        
        return results

    def ion_protein_analysis(self, protein_sel='protein', cutoff=3.5, step=1):
        '''
        Comprehensive analysis of ion-protein interactions.
        
        Parameters
        ----------
        protein_sel : str
            Selection for protein atoms, default='protein'
        cutoff : float
            Distance cutoff for interactions, default=3.5 Å
        step : int
            Trajectory step, default=1
            
        Returns
        -------
        protein_ion_results : dict
            Comprehensive ion-protein interaction results
        '''
        
        protein = self.universe.select_atoms(protein_sel)
        
        print(f"Analyzing ion-protein interactions with {len(protein)} protein atoms...")
        
        results = {
            'cation_sites': {},
            'anion_sites': {},
            'residue_preferences': {},
            'binding_hotspots': {}
        }
        
        # Analyze each ion type separately
        for ion_name, ion_atoms in {**self.cation_types, **self.anion_types}.items():
            if len(ion_atoms) == 0:
                continue
                
            print(f"  Analyzing {ion_name}-protein interactions...")
            
            binding_events = []
            residue_contacts = {}
            
            for ts in tqdm(self.universe.trajectory[::step], desc=f"{ion_name} binding"):
                dist_matrix = distances.distance_array(protein.positions, 
                                                     ion_atoms.positions,
                                                     box=ts.dimensions)
                
                # Find protein atoms in contact with ions
                contacts = np.where(dist_matrix <= cutoff)
                
                # Analyze which residues are involved
                frame_residue_contacts = {}
                for protein_idx in contacts[0]:
                    residue = protein[protein_idx].residue
                    resname = residue.resname
                    resid = residue.resid
                    
                    key = f"{resname}_{resid}"
                    if key not in frame_residue_contacts:
                        frame_residue_contacts[key] = 0
                    frame_residue_contacts[key] += 1
                
                # Store contacts per residue
                for res_key, count in frame_residue_contacts.items():
                    if res_key not in residue_contacts:
                        residue_contacts[res_key] = []
                    residue_contacts[res_key].append(count)
                
                binding_events.append(len(contacts[0]))
            
            # Store results for this ion type
            ion_type = 'cation_sites' if ion_name in self.cation_types else 'anion_sites'
            results[ion_type][ion_name] = {
                'binding_timeline': binding_events,
                'average_binding': np.mean(binding_events),
                'residue_contacts': residue_contacts
            }
            
            # Calculate residue preferences
            results['residue_preferences'][ion_name] = self._calculate_residue_preferences(residue_contacts)
        
        return results

    def _calculate_residue_preferences(self, residue_contacts):
        '''Calculate which residue types are preferred by each ion'''
        
        preferences = {}
        
        for res_key, contacts in residue_contacts.items():
            resname = res_key.split('_')[0]
            
            if resname not in preferences:
                preferences[resname] = []
            preferences[resname].extend(contacts)
        
        # Calculate statistics for each residue type
        residue_stats = {}
        for resname, all_contacts in preferences.items():
            residue_stats[resname] = {
                'average_contacts': np.mean(all_contacts),
                'total_contacts': np.sum(all_contacts),
                'occupancy': np.mean(np.array(all_contacts) > 0),
                'max_contacts': np.max(all_contacts)
            }
        
        return residue_stats

    def solvation_analysis_organic(self, site_selections, cutoff=3.5, step=1, include_ions=False,
                                   save_cache=True, cache_file=None, force_rerun=False):
        '''
        Enhanced multi-site solvation analysis including ion effects.
        
        Parameters
        ----------
        site_selections : dict
            Dictionary of site names and their atom selections
        cutoff : float
            Distance cutoff for solvation shell, default=3.5 Å
        step : int
            Trajectory step, default=1
        include_ions : bool
            Whether to include ion solvation analysis, default=False
        save_cache : bool
            Whether to save results to cache file, default=True
        cache_file : str or None
            Custom cache filename. If None, auto-generates from parameters
        force_rerun : bool
            Force recalculation even if cache exists, default=False
            
        Returns
        -------
        solvation_results : dict
            Enhanced solvation analysis results including ions
        '''
        
        import hashlib
        import pickle
        import os
        
        # Generate cache filename if not provided
        if cache_file is None:
            # Create hash from parameters for unique cache filename
            site_names = '_'.join(sorted(site_selections.keys()))
            param_str = f"{site_names}_cut{cutoff}_s{step}_ions{include_ions}"
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            cache_file = f"solvation_organic_cache_{param_hash}.pkl"
        
        # Check for existing cache
        if save_cache and not force_rerun and os.path.exists(cache_file):
            print(f"📂 Found existing cache: {cache_file}")
            try:
                print("   Loading cached solvation analysis...")
                with open(cache_file, 'rb') as f:
                    results = pickle.load(f)
                print(f"✓ Loaded cached results for {len(results)} sites")
                self.site_solvation = results
                return results
            except Exception as e:
                print(f"   Warning: Failed to load cache ({e}), recalculating...")
        
        print(f"🔄 Running solvation analysis...")
        results = {}
        
        for site_name, selection in site_selections.items():
            print(f"Analyzing solvation around {site_name} sites...")
            
            site_atoms = self.solutes.select_atoms(selection)
            if len(site_atoms) == 0:
                print(f"Warning: No atoms found for site '{site_name}' with selection '{selection}'")
                continue
            
            # Standard solvent solvation
            site_results = {
                'solvent_solvation': self._analyze_site_solvation(site_atoms, cutoff, step)
            }
            
            # Ion solvation if requested
            if include_ions:
                site_results['ion_solvation'] = {}
                
                # Analyze each ion type
                for ion_name, ion_atoms in {**self.cation_types, **self.anion_types}.items():
                    if len(ion_atoms) > 0:
                        ion_solvation = self._analyze_ion_solvation_at_site(
                            site_atoms, ion_atoms, cutoff, step
                        )
                        site_results['ion_solvation'][ion_name] = ion_solvation
            
            # Calculate RDF
            rdf = self.molecular_rdf(selection, self.solvent_sel + ' and name O*', 
                                   range=(0, 10), step=step)
            site_results['rdf'] = rdf
            
            results[site_name] = site_results
        
        # Save cache if requested
        if save_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
                print(f"💾 Saved solvation analysis cache to {cache_file}")
            except Exception as e:
                print(f"Warning: Failed to save cache ({e})")
        
        self.site_solvation = results
        return results

    def _analyze_ion_solvation_at_site(self, site_atoms, ion_atoms, cutoff, step):
        '''Analyze ion solvation at specific sites'''
        
        coordination_numbers = []
        
        for ts in self.universe.trajectory[::step]:
            distances_matrix = distances.distance_array(site_atoms.positions, 
                                                      ion_atoms.positions,
                                                      box=ts.dimensions)
            
            coordinated = (distances_matrix <= cutoff).sum(axis=1)
            coordination_numbers.extend(coordinated)
        
        return {
            'coordination_numbers': np.array(coordination_numbers),
            'average_coordination': np.mean(coordination_numbers),
            'coordination_std': np.std(coordination_numbers),
            'max_coordination': np.max(coordination_numbers) if len(coordination_numbers) > 0 else 0
        }

    def water_solvation_analysis(self, target_sel, rdf_data=None, rdf_boundaries=None, 
                                 solvation_shells=None, step=1, save_cache=True, 
                                 cache_file=None, force_rerun=False,
                                 # Volume normalization parameters
                                 calculate_volumes=True, volume_units='angstrom3', 
                                 geometry_type='auto', fallback_cutoff=3.5,
                                 # Ion inclusion parameters
                                 include_ions=False):
        '''
        Advanced water solvation analysis using RDF boundaries and shell definitions.
        Compatible with existing volume normalization and plotting framework.
        
        Parameters
        ----------
        target_sel : str or list of str
            Selection(s) for target molecules (e.g., 'name O13 O14' for carboxylic acids)
            If list, calculates solvation for each target separately
        rdf_data : dict or None
            Pre-calculated RDF data from molecular_rdf(). If None, uses existing RDF cache.
            Format: {rdf_label: (distances, rdf_values, bins)}
        rdf_boundaries : dict or None
            RDF boundaries from interactive_rdf_boundary_editor.
            Format: {rdf_label: {'S1': (start, end), 'S2': (start, end), ...}}
            If None, uses fallback_cutoff for simple distance analysis.
        solvation_shells : dict or None
            Shell selection for each target-water pair.
            Format: {'carboxylic_acid-OW': ['S1', 'S2'], 'piperazine-OW': ['S1', 'S2', 'S3'], ...}
            If None and rdf_boundaries provided, defaults to all available shells.
        step : int
            Trajectory step for analysis, default=1
        save_cache : bool
            Whether to save results to cache file, default=True
        cache_file : str or None
            Custom cache filename. If None, auto-generates from parameters
        force_rerun : bool
            Force recalculation even if cache exists, default=False
        calculate_volumes : bool
            Whether to calculate shell volumes and solvation densities (default: True)
        volume_units : str
            Units for volume calculations: 'angstrom3' or 'nm3'
        geometry_type : str
            Geometry type for volume calculations: 'auto', 'spherical', 'spherical_shell'
        fallback_cutoff : float
            Distance cutoff when boundaries not available, default=3.5 Å
        include_ions : bool
            Whether to include ion coordination analysis alongside water (default: False)
            If True, analyzes both water and ion coordination in the same shell regions
            
        Returns
        -------
        solvation_results : dict
            If target_sel is string: Returns solvation analysis results dict
            If target_sel is list: Returns {label: solvation_results} dict
            
            Structure compatible with plotting methods:
            {
                'water_solvation': {
                    'OW': {  # Water oxygen
                        'binding_events': array,  # Per-frame coordination
                        'average_binding': float,
                        'peak_analysis': {shell_name: {
                            'average_binding': float,
                            'binding_events': array,
                            'volume_data': {...} if calculate_volumes,
                            'volume_density': array if calculate_volumes
                        }}
                    }
                },
                'ion_solvation': {  # Added when include_ions=True
                    'NA': {...}, 'K': {...}, 'CL': {...}  # Same structure as water
                },
                'total_solvation': {'water': array, 'ions': array, 'combined': array},
                'solvation_sites': {},  # Compatibility placeholder
                'shell_analysis': {}    # Shell-specific data
            }
            
        Examples
        --------
        >>> # Using existing RDF data and boundaries
        >>> solvation_results = analysis.water_solvation_analysis(
        ...     target_sel=[quinolone, carboxylic_acid, piperazine, cyclopropyl],
        ...     rdf_data=rdf_water,  # From molecular_rdf()
        ...     rdf_boundaries=boundaries_refined,  # From interactive_rdf_boundary_editor()
        ...     solvation_shells={
        ...         'carboxylic_acid-OW': ['S1', 'S2'],
        ...         'piperazine-OW': ['S1', 'S2', 'S3'],
        ...         'cyclopropyl-OW': ['S1']
        ...     }
        ... )
        
        >>> # Simple fallback analysis
        >>> solvation_results = analysis.water_solvation_analysis(
        ...     target_sel='resname API',
        ...     fallback_cutoff=4.0
        ... )
        '''
        
        import hashlib
        import os
        
        # Check if target_sel is a list
        if isinstance(target_sel, list):
            print("🔄 Batch water solvation analysis mode activated")
            
            results_dict = {}
            
            for sel in target_sel:
                label = self._extract_label_from_selection(sel)
                print(f"   Analyzing water solvation for: {label}")
                
                # Recursive call for single target
                result = self.water_solvation_analysis(
                    target_sel=sel,
                    rdf_data=rdf_data,
                    rdf_boundaries=rdf_boundaries,
                    solvation_shells=solvation_shells,
                    step=step,
                    save_cache=save_cache,
                    cache_file=None,  # Auto-generate for each target
                    force_rerun=force_rerun,
                    calculate_volumes=calculate_volumes,
                    volume_units=volume_units,
                    geometry_type=geometry_type,
                    fallback_cutoff=fallback_cutoff,
                    include_ions=include_ions
                )
                
                results_dict[label] = result
                
            print("✅ Batch water solvation analysis completed")
            return results_dict
        
        # Single target analysis
        print(f"🔄 Water solvation analysis for: {target_sel}")
        
        # Generate cache filename if not provided
        if cache_file is None:
            target_label = self._extract_label_from_selection(target_sel)
            use_boundaries = rdf_boundaries is not None
            param_str = f"{target_label}_s{step}_bounds{use_boundaries}_vol{calculate_volumes}_ions{include_ions}"
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            cache_file = f"water_solvation_cache_{param_hash}.pkl"
        
        # Check for existing cache
        if save_cache and not force_rerun and os.path.exists(cache_file):
            print(f"📂 Found existing cache: {cache_file}")
            try:
                print("   Loading cached water solvation analysis...")
                import pickle
                with open(cache_file, 'rb') as f:
                    results = pickle.load(f)
                print(f"✓ Loaded cached solvation results")
                return results
            except Exception as e:
                print(f"   Warning: Failed to load cache ({e}), recalculating...")
        
        # Get target atoms
        target_atoms = self.solutes.select_atoms(target_sel)
        if len(target_atoms) == 0:
            print(f"Warning: No atoms found for selection '{target_sel}'")
            return {}
        
        target_label = self._extract_label_from_selection(target_sel)
        print(f"   Target: {target_label} ({len(target_atoms)} atoms)")
        
        # Get water atoms (use oxygen for solvation analysis)
        if hasattr(self, 'water_oxygens') and len(self.water_oxygens) > 0:
            water_atoms = self.water_oxygens
        else:
            # Try to find water oxygens from solvents
            try:
                water_atoms = self.solvents.select_atoms('name OW or name O or (resname SOL and name O*)')
                if len(water_atoms) == 0:
                    water_atoms = self.solvents.select_atoms('name O*')
            except:
                print("Warning: Could not identify water atoms")
                return {}
        
        print(f"   Water atoms: {len(water_atoms)} oxygens")
        
        # Determine RDF label for boundary lookup - try both formats
        rdf_label_OW = f"{target_label}-OW"    # Uppercase
        rdf_label_Ow = f"{target_label}-Ow"    # Mixed case
        
        # Check which format exists in boundaries
        if rdf_boundaries is not None:
            if rdf_label_OW in rdf_boundaries:
                rdf_label = rdf_label_OW
            elif rdf_label_Ow in rdf_boundaries:
                rdf_label = rdf_label_Ow
            else:
                rdf_label = rdf_label_OW  # Default fallback
        else:
            rdf_label = rdf_label_OW
        
        # Initialize results structure
        results = {
            'water_solvation': {
                'OW': {
                    'binding_events': [],
                    'average_binding': 0.0,
                    'peak_analysis': {}
                }
            },
            'total_solvation': {'water': []},
            'solvation_sites': {},
            'shell_analysis': {}
        }
        
        # Add ion solvation structure if requested
        if include_ions:
            results['ion_solvation'] = {}
            results['total_solvation']['ions'] = []
            results['total_solvation']['combined'] = []
        
        # Check for boundary-based analysis
        use_boundaries = (rdf_boundaries is not None and rdf_label in rdf_boundaries)
        
        if use_boundaries:
            print(f"   Using RDF boundaries for shell analysis: {rdf_label}")
            boundaries = rdf_boundaries[rdf_label]
            
            # Determine shells to analyze
            if solvation_shells and rdf_label in solvation_shells:
                shells_to_analyze = solvation_shells[rdf_label]
            else:
                shells_to_analyze = list(boundaries.keys())
            
            print(f"   Shells: {shells_to_analyze}")
            
            # Analyze each shell
            shell_data = {}
            total_binding_events = []
            
            for shell in shells_to_analyze:
                if shell not in boundaries:
                    print(f"   Warning: Shell '{shell}' not found in boundaries, skipping")
                    continue
                
                start, end = boundaries[shell]
                print(f"   Analyzing shell {shell}: {start:.2f} - {end:.2f} Å")
                
                # Calculate per-frame coordination for this shell
                shell_coordination = []
                
                for ts in tqdm(self.universe.trajectory[::step], desc=f"Shell {shell}"):
                    distances_matrix = distances.distance_array(
                        target_atoms.positions, 
                        water_atoms.positions,
                        box=ts.dimensions
                    )
                    
                    # Count waters in this shell
                    in_shell = ((distances_matrix >= start) & (distances_matrix <= end))
                    coordinated = in_shell.sum(axis=1)
                    shell_coordination.extend(coordinated)
                
                shell_coordination = np.array(shell_coordination)
                
                # Store shell data
                shell_results = {
                    'binding_events': shell_coordination,
                    'average_binding': np.mean(shell_coordination),
                    'boundary': (start, end)
                }
                
                # Calculate volume if requested
                if calculate_volumes:
                    try:
                        volume_data = self._calculate_peak_volume(
                            (start, end), geometry_type, volume_units
                        )
                        shell_results['volume_data'] = volume_data
                        
                        # Calculate volume-normalized density
                        volume = volume_data['volume']
                        volume_density = shell_coordination / volume
                        shell_results['volume_density'] = volume_density
                        
                        print(f"      Volume: {volume:.3f} {volume_units}")
                        print(f"      Density: {np.mean(volume_density):.6f} waters/frame/{volume_units}")
                        
                    except Exception as e:
                        print(f"      Warning: Could not calculate volume for {shell}: {e}")
                        shell_results['volume_data'] = None
                        shell_results['volume_density'] = None
                
                shell_data[shell] = shell_results
                
                # Add to total binding events
                if len(total_binding_events) == 0:
                    total_binding_events = shell_coordination.copy()
                else:
                    total_binding_events += shell_coordination
            
            # Store shell analysis results
            results['water_solvation']['OW']['peak_analysis'] = shell_data
            results['water_solvation']['OW']['binding_events'] = total_binding_events
            results['water_solvation']['OW']['average_binding'] = np.mean(total_binding_events)
            results['total_solvation']['water'] = total_binding_events
            results['shell_analysis'] = shell_data
            
        else:
            # Fallback: simple cutoff-based analysis
            print(f"   Using fallback cutoff analysis: {fallback_cutoff} Å")
            
            coordination_numbers = []
            
            for ts in tqdm(self.universe.trajectory[::step], desc="Water solvation"):
                distances_matrix = distances.distance_array(
                    target_atoms.positions, 
                    water_atoms.positions,
                    box=ts.dimensions
                )
                
                coordinated = (distances_matrix <= fallback_cutoff).sum(axis=1)
                coordination_numbers.extend(coordinated)
            
            coordination_numbers = np.array(coordination_numbers)
            
            # Store results in compatible format
            results['water_solvation']['OW']['binding_events'] = coordination_numbers
            results['water_solvation']['OW']['average_binding'] = np.mean(coordination_numbers)
            results['total_solvation']['water'] = coordination_numbers
        
        # Ion analysis (if requested)
        if include_ions:
            print("🔄 Analyzing ion coordination in same shell regions...")
            
            # Analyze each ion type using the same shell definitions
            for ion_type_name, ion_atoms in {**self.cation_types, **self.anion_types}.items():
                if len(ion_atoms) == 0:
                    continue
                    
                print(f"   Analyzing {ion_type_name} coordination...")
                
                if use_boundaries:
                    # Use the same shells and boundaries as water
                    ion_shell_data = {}
                    total_ion_coordination = []
                    
                    for shell in shells_to_analyze:
                        if shell not in boundaries:
                            continue
                            
                        start, end = boundaries[shell]
                        shell_coordination = []
                        
                        for ts in tqdm(self.universe.trajectory[::step], desc=f"{ion_type_name} {shell}", leave=False):
                            distances_matrix = distances.distance_array(
                                target_atoms.positions, 
                                ion_atoms.positions,
                                box=ts.dimensions
                            )
                            
                            # Count ions in this shell
                            in_shell = ((distances_matrix >= start) & (distances_matrix <= end))
                            coordinated = in_shell.sum(axis=1)
                            shell_coordination.extend(coordinated)
                        
                        shell_coordination = np.array(shell_coordination)
                        
                        # Store shell data (same structure as water)
                        shell_results = {
                            'binding_events': shell_coordination,
                            'average_binding': np.mean(shell_coordination),
                            'boundary': (start, end)
                        }
                        
                        # Calculate volume if requested (use same volumes as water)
                        if calculate_volumes and shell in results['water_solvation']['OW']['peak_analysis']:
                            water_shell_data = results['water_solvation']['OW']['peak_analysis'][shell]
                            if 'volume_data' in water_shell_data:
                                # Use same volume data as water
                                shell_results['volume_data'] = water_shell_data['volume_data']
                                
                                # Calculate volume-normalized density for ions
                                volume = shell_results['volume_data']['volume']
                                volume_density = shell_coordination / volume
                                shell_results['volume_density'] = volume_density
                        
                        ion_shell_data[shell] = shell_results
                        
                        # Add to total ion coordination
                        if len(total_ion_coordination) == 0:
                            total_ion_coordination = shell_coordination.copy()
                        else:
                            total_ion_coordination += shell_coordination
                    
                    # Store ion results
                    if ion_type_name not in results['ion_solvation']:
                        results['ion_solvation'][ion_type_name] = {
                            'binding_events': total_ion_coordination,
                            'average_binding': np.mean(total_ion_coordination) if len(total_ion_coordination) > 0 else 0.0,
                            'peak_analysis': ion_shell_data
                        }
                
                else:
                    # Fallback analysis for ions
                    ion_coordination = []
                    
                    for ts in tqdm(self.universe.trajectory[::step], desc=f"{ion_type_name} solvation", leave=False):
                        distances_matrix = distances.distance_array(
                            target_atoms.positions, 
                            ion_atoms.positions,
                            box=ts.dimensions
                        )
                        
                        coordinated = (distances_matrix <= fallback_cutoff).sum(axis=1)
                        ion_coordination.extend(coordinated)
                    
                    ion_coordination = np.array(ion_coordination)
                    
                    results['ion_solvation'][ion_type_name] = {
                        'binding_events': ion_coordination,
                        'average_binding': np.mean(ion_coordination) if len(ion_coordination) > 0 else 0.0,
                        'peak_analysis': {}
                    }
            
            # Calculate combined totals
            if results['ion_solvation']:
                # Sum all ion coordination
                total_ion_events = None
                for ion_data in results['ion_solvation'].values():
                    if ion_data['binding_events'] is not None and len(ion_data['binding_events']) > 0:
                        if total_ion_events is None:
                            total_ion_events = ion_data['binding_events'].copy()
                        else:
                            total_ion_events += ion_data['binding_events']
                
                if total_ion_events is not None:
                    results['total_solvation']['ions'] = total_ion_events
                    # Combined water + ions
                    water_events = results['total_solvation']['water']
                    if len(water_events) == len(total_ion_events):
                        results['total_solvation']['combined'] = water_events + total_ion_events
                    else:
                        print("   Warning: Mismatched trajectory lengths for water and ions")
                        results['total_solvation']['combined'] = water_events
                else:
                    results['total_solvation']['ions'] = []
                    results['total_solvation']['combined'] = results['total_solvation']['water']
        
        # Save cache if requested
        if save_cache:
            try:
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
                print(f"💾 Saved water solvation cache to {cache_file}")
            except Exception as e:
                print(f"Warning: Failed to save cache ({e})")
        
        print("✅ Water solvation analysis completed")
        return results

    def plot_ion_binding_comparison(self, binding_results, save_fig=True):
        '''Plot comparison of ion binding for different ion types'''
        
        ion_names = []
        avg_binding = []
        
        # Collect data from both cations and anions
        for ion_type in ['cation_binding', 'anion_binding']:
            if ion_type in binding_results:
                for ion_name, data in binding_results[ion_type].items():
                    ion_names.append(ion_name)
                    avg_binding.append(data['average_binding'])
        
        if not ion_names:
            print("No binding data to plot")
            return
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(ion_names, avg_binding, color=['lightblue' if name in self.cation_types else 'lightcoral' for name in ion_names])
        
        ax.set_xlabel('Ion Type', fontsize=12)
        ax.set_ylabel('Average Number of Bound Ions', fontsize=12)
        ax.set_title('Ion Binding Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_binding):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('ion_binding_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()

    def summary_report(self):
        '''Generate enhanced summary report including ion analysis'''
        
        print("\n" + "="*70)
        print("MOLECULAR ANALYSIS SUMMARY REPORT")
        print("="*70)
        
        print(f"\nSystem Information:")
        print(f"  Total atoms: {len(self.universe.atoms)}")
        print(f"  Solute atoms: {len(self.solutes)}")
        print(f"  Solvent atoms: {len(self.solvents)}")
        print(f"  Total cations: {len(self.cations)}")
        print(f"  Total anions: {len(self.anions)}")
        print(f"  Trajectory frames: {self.n_frames}")
        
        print(f"\nIon Types Present:")
        print(f"  Cation types: {list(self.cation_types.keys())}")
        print(f"  Anion types: {list(self.anion_types.keys())}")
        
        for ion_name, ion_atoms in self.cation_types.items():
            print(f"    {ion_name}: {len(ion_atoms)} atoms")
        for ion_name, ion_atoms in self.anion_types.items():
            print(f"    {ion_name}: {len(ion_atoms)} atoms")
        
        print(f"\nAnalyses Performed:")
        print(f"  RDFs calculated: {len(self.rdfs)}")
        print(f"  Contact analyses: {len(self.contacts_data)}")
        print(f"  Ion binding analyses: {len(self.ion_binding_data)}")
        
        if hasattr(self, 'site_solvation'):
            print(f"  Site solvation analyses: {len(self.site_solvation)}")
            
        if self.rdfs:
            print(f"\n  Available RDFs:")
            for key in self.rdfs.keys():
                print(f"    - {key}")
        
        if self.ion_binding_data:
            print(f"\n  Ion binding targets analyzed:")
            for target in self.ion_binding_data.keys():
                print(f"    - {target}")
        
        print("\n" + "="*70)

    # Keep all the existing methods from the original class
    def _analyze_site_solvation(self, site_atoms, cutoff, step):
        '''Analyze solvation around specific atomic sites'''
        
        coordination_numbers = []
        
        for ts in tqdm(self.universe.trajectory[::step], desc="Analyzing solvation"):
            distances_matrix = distances.distance_array(site_atoms.positions, 
                                                      self.solvents.positions,
                                                      box=ts.dimensions)
            
            coordinated = (distances_matrix <= cutoff).sum(axis=1)
            coordination_numbers.extend(coordinated)
        
        return {
            'coordination_numbers': np.array(coordination_numbers),
            'average_coordination': np.mean(coordination_numbers),
            'coordination_std': np.std(coordination_numbers)
        }

    def _rdf_with_centers(self, group1, group2, nbins, range_rdf, step, njobs, center_method, normalize):
        '''
        Calculate RDF using molecular centers (COM or COG).
        
        Treats group1 as ONE molecular entity (single center point).
        Calculates distances from that center to all group2 atoms.
        '''
        
        from scipy.spatial import distance as scipy_distance
        
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
        box_vol = np.prod(self.universe.dimensions[:3])
        
        # Calculate number density of group2
        n_group2 = len(group2)
        number_density = n_group2 / box_vol
        
        print(f"  Processing trajectory (step={step})...")
        
        # Calculate RDF over trajectory
        for ts in tqdm(self.universe.trajectory[::step], desc="Computing RDF"):
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
    
    # =========================================================================
    # CUSTOM ATOM SELECTIONS MANAGEMENT
    # =========================================================================
    
    def define_selections(self, selections_dict, verify=True):
        """
        Batch definition of custom atom selections with automatic verification.
        
        Allows organizing selections by category (e.g., molecule parts, solvent, substrate).
        Automatically verifies that selections are valid and contain atoms.
        
        Parameters
        ----------
        selections_dict : dict
            Nested dictionary: {category: {name: selection_string}}
            or flat dictionary: {name: selection_string}
        verify : bool
            If True, verify all selections and print report. Default True.
        
        Examples
        --------
        >>> # Nested (with categories)
        >>> analysis.define_selections({
        ...     'CIP_parts': {
        ...         'quinolone': 'resname api and (name N6 or name C10 ...)',
        ...         'piperazine': 'resname api and (name N13 or name N16 ...)',
        ...         'carboxylic_acid': 'resname api and (name O1 or name O3 or name C2)',
        ...     },
        ...     'solvent': {
        ...         'water_oxygen': 'resname SOL WAT and (name OW or name Ow)',
        ...         'water_hydrogen': 'resname SOL WAT and (name HW1 or name HW2 ...)',
        ...     }
        ... })
        
        >>> # Flat (no categories)
        >>> analysis.define_selections({
        ...     'quinolone': 'resname api and (name N6 or name C10 ...)',
        ...     'water_oxygen': 'resname SOL WAT and (name OW or name Ow)',
        ... })
        
        >>> # Access selections
        >>> rdf = analysis.molecular_rdf(
        ...     group1_sel=analysis.sel('quinolone'),
        ...     group2_sel=analysis.sel('water_oxygen')
        ... )
        """
        # Detect if nested (with categories) or flat
        first_value = next(iter(selections_dict.values()))
        is_nested = isinstance(first_value, dict)
        
        if is_nested:
            # Nested structure: {category: {name: selection}}
            for category, selections in selections_dict.items():
                if category not in self.custom_selections:
                    self.custom_selections[category] = {}
                for name, selection_string in selections.items():
                    self.custom_selections[category][name] = selection_string
        else:
            # Flat structure: {name: selection}
            if 'default' not in self.custom_selections:
                self.custom_selections['default'] = {}
            for name, selection_string in selections_dict.items():
                self.custom_selections['default'][name] = selection_string
        
        if verify:
            self.verify_selections()
    
    def add_selection(self, name, selection_string, category='default', verify=True):
        """
        Add a single custom atom selection with automatic verification.
        
        Parameters
        ----------
        name : str
            Name for this selection (e.g., 'quinolone', 'water_oxygen')
        selection_string : str
            MDAnalysis selection string
        category : str
            Category to organize selections (e.g., 'molecule_parts', 'solvent'). Default 'default'
        verify : bool
            If True, verify the selection and print info. Default True.
        
        Examples
        --------
        >>> analysis.add_selection('quinolone', 
        ...                        'resname api and (name N6 or name C10 ...)',
        ...                        category='CIP_parts')
        
        >>> analysis.add_selection('water_oxygen', 
        ...                        'resname SOL WAT and (name OW or name Ow)',
        ...                        category='solvent')
        """
        if category not in self.custom_selections:
            self.custom_selections[category] = {}
        
        self.custom_selections[category][name] = selection_string
        
        if verify:
            # Verify just this selection
            try:
                atoms = self.universe.select_atoms(selection_string)
                n_atoms = len(atoms)
                status = "✓" if n_atoms > 0 else "⚠️ EMPTY"
                print(f"{status} Added selection '{name}' in category '{category}': {n_atoms} atoms")
            except Exception as e:
                print(f"❌ Error adding selection '{name}': {str(e)}")
    
    def verify_selections(self, categories=None):
        """
        Verify all custom selections and print detailed report.
        
        Parameters
        ----------
        categories : list of str, optional
            List of specific categories to verify. If None, verify all.
        
        Examples
        --------
        >>> analysis.verify_selections()  # Verify all
        >>> analysis.verify_selections(['CIP_parts', 'solvent'])  # Verify specific categories
        """
        if len(self.custom_selections) == 0:
            print("No custom selections defined yet.")
            print("Use define_selections() or add_selection() to add selections.")
            return
        
        print("="*70)
        print("CUSTOM SELECTIONS VERIFICATION")
        print("="*70)
        
        categories_to_check = categories if categories else self.custom_selections.keys()
        
        for category in categories_to_check:
            if category not in self.custom_selections:
                print(f"\n⚠️  Category '{category}' not found")
                continue
            
            print(f"\n{category}:")
            selections = self.custom_selections[category]
            
            for name, selection_string in selections.items():
                try:
                    atoms = self.universe.select_atoms(selection_string)
                    n_atoms = len(atoms)
                    status = "✓" if n_atoms > 0 else "⚠️ EMPTY"
                    print(f"  {status} {name:25s}: {n_atoms:6d} atoms")
                except Exception as e:
                    print(f"  ❌ {name:25s}: ERROR - {str(e)}")
        
        print("\n" + "="*70)
    
    def sel(self, name):
        """
        Retrieve a custom selection string by name.
        
        Searches all categories for the selection name.
        
        Parameters
        ----------
        name : str
            Name of the selection to retrieve
        
        Returns
        -------
        str
            MDAnalysis selection string
        
        Raises
        ------
        KeyError
            If selection name not found
        
        Examples
        --------
        >>> rdf = analysis.molecular_rdf(
        ...     group1_sel=analysis.sel('quinolone'),
        ...     group2_sel=analysis.sel('water_oxygen')
        ... )
        """
        # Search all categories for this name
        for category, selections in self.custom_selections.items():
            if name in selections:
                return selections[name]
        
        # Not found
        available = []
        for cat, sels in self.custom_selections.items():
            available.extend(sels.keys())
        
        raise KeyError(f"Selection '{name}' not found. Available selections: {', '.join(available)}")
    
    def get_selection(self, name, category=None):
        """
        Retrieve a custom selection string by name and optional category.
        
        Parameters
        ----------
        name : str
            Name of the selection
        category : str, optional
            Specific category to search in. If None, searches all categories.
        
        Returns
        -------
        str
            MDAnalysis selection string
        
        Examples
        --------
        >>> sel = analysis.get_selection('quinolone', category='CIP_parts')
        >>> sel = analysis.get_selection('quinolone')  # Search all categories
        """
        if category:
            if category not in self.custom_selections:
                raise KeyError(f"Category '{category}' not found")
            if name not in self.custom_selections[category]:
                raise KeyError(f"Selection '{name}' not found in category '{category}'")
            return self.custom_selections[category][name]
        else:
            return self.sel(name)
    
    def list_selections(self, category=None):
        """
        List all available custom selections.
        
        Parameters
        ----------
        category : str, optional
            Show selections only from this category. If None, show all.
        
        Examples
        --------
        >>> analysis.list_selections()  # Show all
        >>> analysis.list_selections('CIP_parts')  # Show specific category
        """
        if len(self.custom_selections) == 0:
            print("No custom selections defined yet.")
            return
        
        print("="*70)
        print("AVAILABLE CUSTOM SELECTIONS")
        print("="*70)
        
        categories_to_show = [category] if category else self.custom_selections.keys()
        
        for cat in categories_to_show:
            if cat not in self.custom_selections:
                print(f"\n⚠️  Category '{cat}' not found")
                continue
            
            print(f"\n{cat}:")
            for name in self.custom_selections[cat].keys():
                print(f"  • {name}")
        
        print("\n" + "="*70)
    
    def save_selections(self, filename='selections.json'):
        """
        Save custom selections to JSON file for reuse.
        
        Parameters
        ----------
        filename : str
            Path to save JSON file. Default 'selections.json'
        
        Examples
        --------
        >>> analysis.save_selections('cip_selections.json')
        """
        import json
        with open(filename, 'w') as f:
            json.dump(self.custom_selections, f, indent=2)
        print(f"✓ Selections saved to {filename}")
    
    def load_selections(self, filename='selections.json', verify=True):
        """
        Load custom selections from JSON file.
        
        Parameters
        ----------
        filename : str
            Path to JSON file. Default 'selections.json'
        verify : bool
            If True, verify loaded selections. Default True.
        
        Examples
        --------
        >>> analysis.load_selections('cip_selections.json')
        """
        import json
        with open(filename, 'r') as f:
            loaded = json.load(f)
        
        self.custom_selections.update(loaded)
        print(f"✓ Selections loaded from {filename}")
        
        if verify:
            self.verify_selections()

    def calculate_running_coordination_number(self, rdf_dict, density_dict=None):
        '''
        Calculate running coordination numbers (RCN) for RDF data.
        
        RCN(r) = 4π * ρ * ∫[0 to r] g(r') * r'² dr'
        
        where ρ is the number density of coordinating particles.
        
        Parameters
        ----------
        rdf_dict : dict
            Dictionary of RDF results: {label: rdf_results}
        density_dict : dict, optional
            Dictionary of number densities: {label: density_in_particles_per_A3}
            If None, attempts to estimate from typical water density (0.0334 particles/Å³)
        
        Returns
        -------
        rcn_dict : dict
            Dictionary of RCN data: {label: {'r': r_array, 'rcn': rcn_array, 'rdf': g_r}}
        
        Examples
        --------
        >>> # Calculate RCN for all RDFs with default water density
        >>> rcn_data = analysis.calculate_running_coordination_number(rdf_water)
        
        >>> # With custom densities
        >>> densities = {'quinolone-OW': 0.0334, 'quinolone-NA': 0.0001}
        >>> rcn_data = analysis.calculate_running_coordination_number(rdf_dict, densities)
        
        >>> # Use with plot_rdf_with_shells
        >>> plotter.plot_rdf_with_shells(rdf_dict, boundaries, show_rcn=True, rcn_data=rcn_data)
        '''
        
        rcn_dict = {}
        
        # Default water density: ~33.4 molecules/nm³ = 0.0334 molecules/Å³
        default_density = 0.0334
        
        for label, rdf_results in rdf_dict.items():
            # Extract RDF data
            if hasattr(rdf_results, 'bins'):
                r = rdf_results.bins
                g_r = rdf_results.rdf
            else:
                r = rdf_results['bins']
                g_r = rdf_results['rdf']
            
            # Get density for this RDF
            if density_dict and label in density_dict:
                rho = density_dict[label]
            else:
                # Use default water density
                rho = default_density
                print(f"Using default density {rho:.6f} particles/Å³ for '{label}'")
            
            # Calculate RCN using trapezoidal integration
            # RCN(r) = 4π * ρ * ∫[g(r') * r'² dr'] from 0 to r
            integrand = g_r * r**2
            
            # Cumulative integration using scipy's cumulative_trapezoid (more accurate than cumsum)
            from scipy.integrate import cumulative_trapezoid
            
            # cumulative_trapezoid returns n-1 points, prepend 0 for first point
            rcn_integral = cumulative_trapezoid(integrand, r, initial=0)
            rcn = 4 * np.pi * rho * rcn_integral
            
            # Store results
            rcn_dict[label] = {
                'r': r.copy(),
                'rcn': rcn.copy(),
                'rdf': g_r.copy(),
                'density': rho
            }
            
            print(f"✓ RCN calculated for '{label}': CN at r={r[-1]:.2f} Å = {rcn[-1]:.2f}")
        
        return rcn_dict

    def determine_oc_coordination_shells(self, rdf_dict, find_peaks_kwargs=None, plot=True, 
                                         save_plots=True, plot_range=12, output_dir='.', 
                                         shell_naming='auto', max_peaks=4):
        '''
        Automatically determine solvation/coordination shell boundaries for organic compound analysis.
        
        Detects shell boundaries using RDF minima detection and applies appropriate labeling
        and color schemes based on coordination type:
        - Water coordination (e.g., quinolone-OW): Labels as shell_1, shell_2, shell_3 with blue gradient
        - Ion coordination (e.g., quinolone-NA): Labels as CIP, SIP, DSIP, FI with ion-pairing colors
        
        Parameters
        ----------
        rdf_dict : dict
            Dictionary of RDF results from molecular_rdf(): {label: rdf_results}
        find_peaks_kwargs : dict, optional
            Parameters for scipy.signal.find_peaks to detect minima.
            If None (default), uses adaptive parameters based on coordination type:
              - Water: {'distance': 10, 'height': -3, 'prominence': 0.01}
              - Ions: {'distance': 5, 'height': -1.1, 'prominence': 0.1}
            You can override the adaptive defaults by providing custom parameters:
              find_peaks_kwargs={'distance': 8, 'height': -2.0, 'prominence': 0.05}
        plot : bool, default=True
            Whether to generate plots
        save_plots : bool, default=True
            Whether to save plots to disk
        plot_range : float, default=12
            Maximum r value for plotting (Angstroms)
        output_dir : str, default='.'
            Directory to save plots
        shell_naming : str, default='auto'
            Shell naming convention:
            - 'auto': Automatic naming based on coordination type
              * Water: shell_1, shell_2, shell_3, bulk
              * Ions: CIP, SIP, DSIP, FI
            - 'shell': Generic shell naming (Shell_1, Shell_2, Shell_3, ...)
            - 'peak': Peak-based naming (P1, P2, P3, ..., up to max_peaks, then Bulk)
        max_peaks : int, default=4
            Maximum number of numbered peaks before switching to 'Bulk' label.
            Only applies to 'peak' and 'shell' naming modes.
        
        Returns
        -------
        boundaries : dict
            Dictionary of shell boundaries compatible with plot_rdf_with_shells():
            {label: {'shell_1': (start, end), ...}} for water
            {label: {'CIP': (start, end), 'SIP': (start, end), ...}} for ions
        
        Examples
        --------
        >>> # Calculate RDFs
        >>> rdf_water = oc_analysis.molecular_rdf(
        ...     group1_sel=[quinolone, carboxylic_acid],
        ...     group2_sel=water_oxygen,
        ...     bin_width=0.1, range=(0, 20))
        >>> 
        >>> # Automatically detect boundaries with adaptive defaults
        >>> boundaries_water = oc_analysis.determine_oc_coordination_shells(
        ...     rdf_water, plot=True, save_plots=True)
        >>> 
        >>> # Override peak detection parameters if needed
        >>> boundaries_water = oc_analysis.determine_oc_coordination_shells(
        ...     rdf_water, 
        ...     find_peaks_kwargs={'distance': 8, 'height': -2.5, 'prominence': 0.02},
        ...     plot=True, save_plots=True)
        >>> 
        >>> # Use boundaries for visualization
        >>> plotter.plot_rdf_with_shells(rdf_water, boundaries_water, ncols=2)
        >>> 
        >>> # Use flexible shell naming
        >>> boundaries_peak = oc_analysis.determine_oc_coordination_shells(
        ...     rdf_ions, shell_naming='peak')  # Creates P1, P2, P3, P4, Bulk
        >>> 
        >>> boundaries_shell = oc_analysis.determine_oc_coordination_shells(
        ...     rdf_water, shell_naming='shell', max_peaks=3)  # Creates Shell_1, Shell_2, Shell_3, Bulk
        '''
        
        from scipy.signal import find_peaks
        import os
        
        if output_dir != '.' and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("\n" + "="*80)
        print("AUTOMATIC COORDINATION SHELL DETERMINATION")
        print("="*80)
        
        boundaries = {}
        
        # Common ion names for detection
        import re
        
        water_indicators = ['OW', 'HW', 'O_WATER', 'H_WATER', 'WAT', 'SOL', 'WATER', 'H2O', 'OXYGEN', 'HYDROGEN']
        cation_names = ['NA', 'K', 'MG', 'CA', 'LI', 'NH4', 'RB', 'CS', 'SR', 'BA']
        anion_names = ['CL', 'BR', 'F', 'I']
        
        for label, rdf_data in rdf_dict.items():
            print(f"\nProcessing: {label}")
            
            # Determine coordination type from label
            label_upper = label.upper()
            print(f"  Label (uppercase): {label_upper}")
            
            # Check for water indicators (substring match is fine)
            is_water = any(indicator in label_upper for indicator in water_indicators)
            
            # Check for ions using word boundaries to avoid false matches
            # (e.g., 'I' in 'QUINOLONE' or 'CA' in 'CARBOXYLIC_ACID')
            is_cation = any(re.search(r'\b' + ion + r'\b', label_upper) for ion in cation_names)
            is_anion = any(re.search(r'\b' + ion + r'\b', label_upper) for ion in anion_names)
            is_ion = is_cation or is_anion
            
            print(f"  Detection results: water={is_water}, cation={is_cation}, anion={is_anion}")
            
            if is_water:
                coord_type = 'water'
                print(f"  → Detected: Water coordination")
            elif is_ion:
                coord_type = 'ion'
                ion_type = 'cation' if is_cation else 'anion'
                print(f"  → Detected: {ion_type.title()} coordination")
            else:
                coord_type = 'unknown'
                print(f"  → Warning: Could not determine coordination type, defaulting to water")
                coord_type = 'water'  # Fixed: actually set coord_type
                is_water = True
            
            # Set default find_peaks parameters based on coordination type
            if find_peaks_kwargs is None:
                if coord_type == 'ion':
                    peaks_kwargs = {'distance': 5, 'height': -1.1, 'prominence': 0.1}
                else:  # water or unknown
                    peaks_kwargs = {'distance': 10, 'height': -3, 'prominence': 0.01}
            else:
                peaks_kwargs = find_peaks_kwargs.copy()
            
            print(f"  Using find_peaks parameters: {peaks_kwargs}")
            
            # Get RDF data
            r = rdf_data.bins
            g_r = rdf_data.rdf
            
            # Find peaks (maxima) and minima
            peaks, _ = find_peaks(g_r, height=1.0, distance=5)
            minima, _ = find_peaks(-g_r, **peaks_kwargs)
            
            # Sort by position
            peaks = peaks[np.argsort(r[peaks])]
            minima = minima[np.argsort(r[minima])]
            
            print(f"  Found {len(peaks)} peaks and {len(minima)} minima")
            
            if len(minima) == 0:
                print(f"  ✗ No minima found - skipping {label}")
                boundaries[label] = {}
                continue
            
            # Filter minima that are too far
            valid_minima = minima[r[minima] <= plot_range]
            if len(valid_minima) < len(minima):
                print(f"  Filtered to {len(valid_minima)} minima within {plot_range} Å")
                minima = valid_minima
            
            # Determine boundaries based on coordination type
            label_boundaries = {}
            
            if coord_type == 'water':
                # Water coordination: flexible naming based on shell_naming parameter
                max_shells = min(3, len(minima)) if shell_naming == 'auto' else len(minima)
                
                # Generate shell names based on naming convention
                if shell_naming == 'auto':
                    shell_names = [f'shell_{i+1}' for i in range(max_shells)]
                    bulk_name = 'bulk'
                elif shell_naming == 'shell':
                    shell_names = [f'Shell_{i+1}' for i in range(max_shells)]
                    bulk_name = 'Bulk'
                elif shell_naming == 'peak':
                    # Limit to max_peaks
                    max_shells = min(max_peaks, len(minima))
                    shell_names = [f'P{i+1}' for i in range(max_shells)]
                    bulk_name = 'Bulk'
                else:
                    raise ValueError(f"Unknown shell_naming: {shell_naming}. Use 'auto', 'shell', or 'peak'.")
                
                # First shell: 0 to first minimum
                if max_shells > 0:
                    label_boundaries[shell_names[0]] = (0.0, float(r[minima[0]]))
                
                # Additional shells
                for i in range(1, max_shells):
                    if i < len(minima):
                        shell_start = label_boundaries[shell_names[i-1]][1]
                        shell_end = float(r[minima[i]])
                        
                        if shell_end > shell_start + 0.5:  # Minimum 0.5 Å width
                            label_boundaries[shell_names[i]] = (shell_start, shell_end)
                
                # Add bulk/beyond region from end of last shell to infinity
                last_shell_end = 0.0
                for shell_name, (start, end) in label_boundaries.items():
                    if not shell_name.lower().startswith(('bulk', 'beyond')):
                        last_shell_end = max(last_shell_end, end)
                
                if last_shell_end > 0:
                    label_boundaries[bulk_name] = (last_shell_end, np.inf)
                
                shell_count = len([k for k in label_boundaries.keys() if not k.lower().startswith(('bulk', 'beyond'))])
                print(f"  ✓ Determined {shell_count} water solvation shells + {bulk_name.lower()}:")
                for shell_name, (start, end) in label_boundaries.items():
                    if shell_name.lower().startswith(('bulk', 'beyond')):
                        print(f"    {shell_name}: {start:.2f} - ∞ Å")
                    else:
                        print(f"    {shell_name}: {start:.2f} - {end:.2f} Å")
            
            elif coord_type == 'ion':
                # Ion coordination: flexible naming based on shell_naming parameter
                if shell_naming == 'auto':
                    # Traditional ion pairing nomenclature
                    if len(minima) >= 3:
                        label_boundaries['CIP'] = (0.0, float(r[minima[0]]))
                        label_boundaries['SIP'] = (float(r[minima[0]]), float(r[minima[1]]))
                        label_boundaries['DSIP'] = (float(r[minima[1]]), float(r[minima[2]]))
                        label_boundaries['FI'] = (float(r[minima[2]]), np.inf)
                        print(f"  ✓ Determined ion pairing regions:")
                        print(f"    CIP: {label_boundaries['CIP'][0]:.2f} - {label_boundaries['CIP'][1]:.2f} Å")
                        print(f"    SIP: {label_boundaries['SIP'][0]:.2f} - {label_boundaries['SIP'][1]:.2f} Å")
                        print(f"    DSIP: {label_boundaries['DSIP'][0]:.2f} - {label_boundaries['DSIP'][1]:.2f} Å")
                        print(f"    FI: {label_boundaries['FI'][0]:.2f} - ∞ Å")
                    
                    elif len(minima) == 2:
                        label_boundaries['CIP'] = (0.0, float(r[minima[0]]))
                        label_boundaries['SIP'] = (float(r[minima[0]]), float(r[minima[1]]))
                        label_boundaries['FI'] = (float(r[minima[1]]), np.inf)
                        print(f"  ✓ Determined ion pairing regions (2 minima):")
                        print(f"    CIP: {label_boundaries['CIP'][0]:.2f} - {label_boundaries['CIP'][1]:.2f} Å")
                        print(f"    SIP: {label_boundaries['SIP'][0]:.2f} - {label_boundaries['SIP'][1]:.2f} Å")
                        print(f"    FI: {label_boundaries['FI'][0]:.2f} - ∞ Å")
                    
                    elif len(minima) == 1:
                        label_boundaries['CIP'] = (0.0, float(r[minima[0]]))
                        label_boundaries['FI'] = (float(r[minima[0]]), np.inf)
                        print(f"  ✓ Determined ion pairing regions (1 minimum):")
                        print(f"    CIP: {label_boundaries['CIP'][0]:.2f} - {label_boundaries['CIP'][1]:.2f} Å")
                        print(f"    FI: {label_boundaries['FI'][0]:.2f} - ∞ Å")
                
                else:
                    # Generic peak/shell naming for ions
                    if shell_naming == 'shell':
                        # Limit to max_peaks
                        max_shell_count = min(max_peaks, len(minima))
                        shell_names = [f'S{i+1}' for i in range(max_shell_count)]
                        beyond_name = 'Bulk'
                    elif shell_naming == 'peak':
                        # Limit to max_peaks
                        max_shell_count = min(max_peaks, len(minima))
                        shell_names = [f'P{i+1}' for i in range(max_shell_count)]
                        beyond_name = 'Bulk'
                    else:
                        raise ValueError(f"Unknown shell_naming: {shell_naming}. Use 'auto', 'shell', or 'peak'.")
                    
                    # Create boundaries for numbered peaks (up to max_peaks)
                    if len(minima) > 0:
                        # First shell: 0 to first minimum
                        label_boundaries[shell_names[0]] = (0.0, float(r[minima[0]]))
                        
                        # Additional shells up to max_peaks-1
                        for i in range(1, min(max_shell_count, len(minima))):
                            shell_start = float(r[minima[i-1]])
                            shell_end = float(r[minima[i]])
                            label_boundaries[shell_names[i]] = (shell_start, shell_end)
                        
                        # Bulk region: from last numbered shell to infinity
                        if len(minima) > max_shell_count:
                            # More minima than max_peaks, bulk starts from max_peaks-th minimum
                            bulk_start = float(r[minima[max_shell_count-1]])
                        elif len(minima) == max_shell_count:
                            # Exactly max_peaks minima, bulk starts from last minimum
                            bulk_start = float(r[minima[-1]])
                        else:
                            # Fewer minima than max_peaks, bulk starts from last minimum
                            bulk_start = float(r[minima[-1]])
                        
                        label_boundaries[beyond_name] = (bulk_start, np.inf)
                        
                        print(f"  ✓ Determined {len(shell_names)} ion coordination shells + {beyond_name.lower()}:")
                        for shell_name, (start, end) in label_boundaries.items():
                            if shell_name == beyond_name:
                                print(f"    {shell_name}: {start:.2f} - ∞ Å")
                            else:
                                print(f"    {shell_name}: {start:.2f} - {end:.2f} Å")
            
            boundaries[label] = label_boundaries
            
            # Generate plot if requested
            if plot and len(label_boundaries) > 0:
                self._plot_determined_shells(
                    label, r, g_r, label_boundaries, coord_type,
                    minima, peaks_kwargs, save_plots, plot_range, output_dir
                )
        
        print("\n" + "="*80)
        print(f"✓ Shell determination complete for {len(boundaries)} RDF curves")
        print("="*80 + "\n")
        
        return boundaries

    def _plot_determined_shells(self, label, r, g_r, boundaries, coord_type, 
                                minima, peaks_kwargs, save_plots, plot_range, output_dir):
        '''Helper method to plot determined shells with appropriate colors'''
        import matplotlib.colors as mcolors
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot RDF
        ax.plot(r, g_r, color='k', linewidth=2, label='g(r)')
        
        # Get colors based on coordination type
        if coord_type == 'water':
            # Blue saturation gradient starting from #00c5ff
            base_rgb = mcolors.hex2color('#00c5ff')
            base_hsv = mcolors.rgb_to_hsv(base_rgb)
            
            # Count only shell regions (not bulk)
            n_shells = len([k for k in boundaries.keys() if k.startswith('shell_')])
            has_bulk = 'bulk' in boundaries
            
            # Generate colors including bulk (lightest shade)
            if n_shells == 1:
                saturations = [base_hsv[1], 0.2]  # Shell 1 + Bulk
            elif n_shells == 2:
                saturations = [base_hsv[1], 0.6, 0.2]  # Shell 1 + Shell 2 + Bulk
            elif n_shells == 3:
                saturations = [base_hsv[1], 0.7, 0.4, 0.2]  # Shell 1 + Shell 2 + Shell 3 + Bulk
            else:
                step = (base_hsv[1] - 0.2) / n_shells
                saturations = [base_hsv[1] - (i * step) for i in range(n_shells)]
                saturations.append(0.2)  # Always very light for bulk
            
            colors = []
            for sat in saturations:
                hsv = (base_hsv[0], sat, base_hsv[2])
                rgb = mcolors.hsv_to_rgb(hsv)
                colors.append(mcolors.to_hex(rgb))
        
        else:  # ion coordination
            # Ion pairing colors: CIP, SIP, DSIP, FI
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
        
        # Calculate label position
        rdf_max = max(g_r[r <= plot_range])
        label_y_position = rdf_max * 1.15
        
        # Plot shell regions
        if coord_type == 'water':
            # For water: separate shells from bulk for proper coloring
            shell_items = [(k, v) for k, v in boundaries.items() if k.startswith('shell_')]
            bulk_item = boundaries.get('bulk', None)
            
            # Plot shells
            for i, (shell_name, (start, end)) in enumerate(shell_items):
                color = colors[i % len(colors)]
                end_plot = min(end, plot_range) if np.isfinite(end) else plot_range
                
                # Fill region
                ax.axvspan(start, end_plot, alpha=0.4, color=color)
                
                # Add label
                mid_point = (start + end_plot) / 2
                display_name = shell_name.replace('_', ' ').title()
                ax.text(mid_point, label_y_position, display_name,
                       horizontalalignment='center', fontweight='bold', fontsize=12,
                       color='black')
            
            # Plot bulk region separately with lightest color
            if bulk_item is not None:
                bulk_start, bulk_end = bulk_item
                bulk_plot_end = min(bulk_end, plot_range) if np.isfinite(bulk_end) else plot_range
                bulk_color = colors[-1]  # Lightest color
                
                ax.axvspan(bulk_start, bulk_plot_end, alpha=0.4, color=bulk_color)
                
                bulk_mid_point = (bulk_start + bulk_plot_end) / 2
                ax.text(bulk_mid_point, label_y_position, 'Bulk',
                       horizontalalignment='center', fontweight='bold', fontsize=12,
                       color='black')
        
        else:  # ion coordination
            # For ions: plot all shells (including combined ones) with hierarchical colors
            # Support flexible naming: P1/P2/P3/P4, Shell_1/Shell_2/Shell_3/Shell_4, and traditional CIP/SIP/DSIP/FI
            ion_colors_map = {
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
                'S1': 'lightcoral',    # Abbreviated shell names
                'S2': 'lightblue',     # Same as SIP
                'S3': 'lightgreen',    # Same as DSIP
                'S4': 'lightyellow',   # Same as FI
                'Bulk': 'lightgoldenrodyellow'  # Very light yellow for bulk
            }
            
            # Define order for hierarchical logic
            region_order_traditional = ['CIP', 'SIP', 'DSIP', 'FI']
            region_order_peak = ['P1', 'P2', 'P3', 'P4']
            region_order_shell = ['Shell_1', 'Shell_2', 'Shell_3', 'Shell_4']
            region_order_abbrev = ['S1', 'S2', 'S3', 'S4']
            
            # Sort shells by start position
            sorted_boundaries = sorted(boundaries.items(), key=lambda x: x[1][0])
            
            for i, (shell_name, (start, end)) in enumerate(sorted_boundaries):
                # Determine color for this shell
                if '+' in shell_name:
                    # Combined region - use hierarchical color logic
                    parts = shell_name.split('+')
                    if 'FI' in parts or 'P4' in parts or 'Shell_4' in parts or 'S4' in parts:
                        color = ion_colors_map.get('P4', ion_colors_map.get('FI', 'lightyellow'))
                    elif any(p in region_order_traditional for p in parts):
                        highest = max([p for p in parts if p in region_order_traditional], 
                                    key=lambda x: region_order_traditional.index(x))
                        color = ion_colors_map.get(highest, 'lightgray')
                    elif any(p in region_order_peak for p in parts):
                        highest = max([p for p in parts if p in region_order_peak], 
                                    key=lambda x: region_order_peak.index(x))
                        color = ion_colors_map.get(highest, 'lightgray')
                    elif any(p in region_order_shell for p in parts):
                        highest = max([p for p in parts if p in region_order_shell], 
                                    key=lambda x: region_order_shell.index(x))
                        color = ion_colors_map.get(highest, 'lightgray')
                    elif any(p in region_order_abbrev for p in parts):
                        highest = max([p for p in parts if p in region_order_abbrev], 
                                    key=lambda x: region_order_abbrev.index(x))
                        color = ion_colors_map.get(highest, 'lightgray')
                    else:
                        color = 'lightgray'
                else:
                    # Direct name lookup
                    color = ion_colors_map.get(shell_name, 'lightgray')
                
                # Plot this shell region
                end_plot = min(end, plot_range) if np.isfinite(end) else plot_range
                
                # Fill region
                ax.axvspan(start, end_plot, alpha=0.4, color=color)
                
                # Add label
                mid_point = (start + end_plot) / 2
                ax.text(mid_point, label_y_position, shell_name,
                       horizontalalignment='center', fontweight='bold', fontsize=12,
                       color='black')
        
        # Mark minima
        if len(minima) > 0:
            ax.scatter(r[minima], g_r[minima], color='red', s=100, zorder=5,
                      marker='v', edgecolor='black', linewidth=2)
        
        # Formatting
        ax.set_xlabel('r (Å)', fontsize=14)
        ax.set_ylabel('g(r)', fontsize=14)
        
        title_type = 'Water Solvation' if coord_type == 'water' else 'Ion Pairing'
        param_str = ', '.join([f"{k}={v}" for k, v in peaks_kwargs.items()])
        ax.set_title(f'{label}\n{title_type} Shells (find_peaks: {param_str})',
                    fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, plot_range)
        ax.set_ylim(bottom=0, top=rdf_max * 1.3)
        
        plt.tight_layout()
        
        if save_plots:
            import os
            filename = f'{label}_{coord_type}_shells_auto.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  Plot saved: {filepath}")
        
        plt.show()

    def interactive_rdf_boundary_editor(self, rdf_dict, initial_boundaries=None):
        '''
        Interactive editor for defining solvation shell boundaries for RDF curves.
        
        Parameters
        ----------
        rdf_dict : dict
            Dictionary of RDF results: {label: rdf_results}
        initial_boundaries : dict, optional
            Initial boundaries: {label: {'shell_1': (start, end), 'shell_2': (start, end), ...}}
            If None, starts with empty boundaries
        
        Returns
        -------
        boundaries : dict
            Dictionary of shell boundaries: {label: {'shell_1': (start, end), 'shell_2': (start, end), ...}}
        
        Commands
        --------
        - list [label]              : Show current boundaries (all or specific label)
        - select <label>            : Select RDF curve to edit
        - add <shell_name> <start> <end> : Add shell boundary (e.g., 'add shell_1 2.0 3.5')
        - modify <shell_name> <start|end> <value> : Modify boundary
        - remove <shell_name>       : Remove shell boundary
        - plot [label]              : Plot RDF with current boundaries
        - save <filename>           : Save boundaries to file
        - load <filename>           : Load boundaries from file
        - quit                      : Exit editor
        
        Examples
        --------
        >>> boundaries = analysis.interactive_rdf_boundary_editor(rdf)
        >>> # Then use with plotter:
        >>> plotter.plot_rdf_with_shells(rdf, boundaries)
        '''
        
        import json
        import copy
        
        # Initialize boundaries dict with proper deep copying
        if initial_boundaries is None:
            boundaries = {label: {} for label in rdf_dict.keys()}
        else:
            # Use deep copy to ensure nested dictionaries are independent
            boundaries = copy.deepcopy(initial_boundaries)
            # Ensure all labels from rdf_dict exist in boundaries
            for label in rdf_dict.keys():
                if label not in boundaries:
                    boundaries[label] = {}
        
        current_label = None
        available_labels = list(rdf_dict.keys())
        
        # Store original boundaries for reset functionality
        original_boundaries = copy.deepcopy(boundaries)
        
        print("\n" + "="*80)
        print("INTERACTIVE RDF BOUNDARY EDITOR")
        print("="*80)
        print(f"Available RDF curves: {available_labels}")
        print("\nCommands:")
        print("  list [label]                   - Show boundaries")
        print("  select <label>                 - Select RDF to edit")
        print("  add <shell> <start> <end>      - Add shell")
        print("     Water: 'add shell_1 2.0 3.5'  |  Ions: 'add CIP 0.0 1.2'")
        print("  modify <shell> <start|end> <val> - Modify boundary")
        print("     Water: 'modify shell_2 end 4.5'  |  Ions: 'modify SIP start 1.3'")
        print("  expand <shell> <direction> <amount> - Expand shell")
        print("     Water: 'expand shell_2 outward 0.5'  |  Ions: 'expand DSIP both 0.3'")
        print("  contract <shell> <direction> <amount> - Contract shell")
        print("     Water: 'contract shell_1 both 0.2'  |  Ions: 'contract CIP inward 0.1'")
        print("  merge <shell1> <shell2> [...]  - Combine adjacent shells")
        print("     Ions: 'merge DSIP FI' → creates 'DSIP+FI'")
        print("  remove <shell> [merge_option]  - Remove shell with gap-filling")
        print("     merge_option: expand_previous, expand_next (default), expand_both")
        print("  replot                         - Replot current selection with boundaries")
        print("  reset [label]                  - Reset to original boundaries")
        print("  save <filename>                - Save boundaries to JSON")
        print("  load <filename>                - Load boundaries from JSON")
        print("  quit (or Ctrl+C)               - Exit editor")
        print("="*80 + "\n")
        print("NOTE: Type label names WITHOUT brackets (e.g., 'select quinolone-NA')")
        print("="*80 + "\n")
        
        try:
            while True:
                # Show current context
                if current_label:
                    prompt = f"[{current_label}] > "
                else:
                    prompt = "[No selection] > "
                
                try:
                    command = input(prompt).strip()
                except KeyboardInterrupt:
                    print("\n\nExiting boundary editor (Ctrl+C)")
                    break
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    print("Exiting boundary editor")
                    break
                
                elif cmd == 'list':
                    if len(parts) > 1:
                        label = parts[1].strip('[]')  # Strip brackets if present
                        if label in boundaries:
                            self._print_boundaries(label, boundaries[label])
                        else:
                            print(f"Label '{label}' not found")
                            print(f"Available labels: {available_labels}")
                    else:
                        print("\nAll boundaries:")
                        for label in available_labels:
                            try:
                                self._print_boundaries(label, boundaries[label])
                            except KeyError:
                                print(f"  {label}: No boundaries defined (missing from boundaries dict)")
                            except Exception as e:
                                print(f"  {label}: Error displaying boundaries: {e}")
                
                elif cmd == 'select':
                    if len(parts) < 2:
                        print("Usage: select <label>")
                        print(f"Available labels (without brackets):")
                        for lbl in available_labels:
                            print(f"  - {lbl}")
                        continue
                    # Strip brackets if user copied from list output
                    label = parts[1].strip('[]').strip("'\"")
                    if label in available_labels:
                        current_label = label
                        print(f"✓ Selected: {label}")
                        self._print_boundaries(label, boundaries[label])
                    else:
                        print(f"✗ Label '{label}' not found.")
                        print(f"Available labels (type without brackets):")
                        for lbl in available_labels:
                            print(f"  - {lbl}")
                
                elif cmd == 'add':
                    if current_label is None:
                        print("No RDF selected. Use 'select <label>' first")
                        continue
                    if len(parts) < 4:
                        print("Usage: add <shell_name> <start> <end>")
                        continue
                    shell_name = parts[1]
                    try:
                        start = float(parts[2])
                        end = float(parts[3])
                        boundaries[current_label][shell_name] = (start, end)
                        print(f"Added {shell_name}: {start:.2f} - {end:.2f} Å")
                    except ValueError:
                        print("Invalid values. Use numbers for start and end")
                
                elif cmd == 'modify':
                    if current_label is None:
                        print("No RDF selected. Use 'select <label>' first")
                        continue
                    if len(parts) < 4:
                        print("Usage: modify <shell_name> <start|end> <value>")
                        continue
                    shell_name = parts[1]
                    boundary_type = parts[2].lower()
                    try:
                        value = float(parts[3])
                        if shell_name not in boundaries[current_label]:
                            print(f"Shell '{shell_name}' not found")
                            continue
                        
                        # Apply modification with cascading adjustments
                        success = self._modify_boundary_with_cascade(
                            boundaries[current_label], shell_name, boundary_type, value
                        )
                        
                        if success:
                            print(f"\nUpdated boundaries:")
                            self._print_boundaries(current_label, boundaries[current_label])
                        else:
                            print(f"✗ Modification failed - see error above")
                    except ValueError:
                        print("Invalid value. Use a number")
                
                elif cmd == 'remove':
                    if current_label is None:
                        print("No RDF selected. Use 'select <label>' first")
                        continue
                    if len(parts) < 2:
                        print("Usage: remove <shell_name> [merge_option]")
                        print("Merge options: expand_previous, expand_next (default), expand_both")
                        continue
                    
                    shell_name = parts[1]
                    merge_option = parts[2] if len(parts) > 2 else 'expand_next'
                    
                    if merge_option not in ['expand_previous', 'expand_next', 'expand_both']:
                        print("Invalid merge option. Use: expand_previous, expand_next, or expand_both")
                        continue
                    
                    if shell_name in boundaries[current_label]:
                        success = self._remove_shell_with_gap_filling(
                            boundaries[current_label], shell_name, merge_option)
                        if success:
                            print(f"\nUpdated boundaries:")
                            self._print_boundaries(current_label, boundaries[current_label])
                    else:
                        print(f"Shell '{shell_name}' not found")
                
                elif cmd == 'merge':
                    if current_label is None:
                        print("No RDF selected. Use 'select <label>' first")
                        continue
                    if len(parts) < 3:
                        print("Usage: merge <shell1> <shell2> [shell3 ...]")
                        print("Example: merge DSIP FI")
                        print("Example: merge SIP DSIP FI")
                        continue
                    
                    shell_names = parts[1:]
                    # Verify all shells exist
                    missing = [s for s in shell_names if s not in boundaries[current_label]]
                    if missing:
                        print(f"Shell(s) not found: {', '.join(missing)}")
                        continue
                    
                    # Check if shells are adjacent
                    shells = boundaries[current_label]
                    sorted_shells = sorted([(name, shells[name]) for name in shell_names], 
                                         key=lambda x: x[1][0])
                    
                    # Verify adjacency
                    for i in range(len(sorted_shells) - 1):
                        curr_name, (curr_start, curr_end) = sorted_shells[i]
                        next_name, (next_start, next_end) = sorted_shells[i + 1]
                        if abs(curr_end - next_start) > 0.01:  # Small tolerance for float comparison
                            print(f"Warning: {curr_name} and {next_name} are not adjacent")
                            print(f"  {curr_name} ends at {curr_end:.2f}, {next_name} starts at {next_start:.2f}")
                    
                    # Create merged shell
                    merged_name = '+'.join([name for name, _ in sorted_shells])
                    merged_start = sorted_shells[0][1][0]
                    merged_end = sorted_shells[-1][1][1]
                    
                    # Remove original shells
                    for shell_name in shell_names:
                        del boundaries[current_label][shell_name]
                    
                    # Add merged shell
                    boundaries[current_label][merged_name] = (merged_start, merged_end)
                    
                    print(f"\n✓ Merged {' + '.join(shell_names)} → {merged_name}")
                    print(f"  Range: {merged_start:.2f} - {merged_end:.2f} Å")
                    print(f"\nUpdated boundaries:")
                    self._print_boundaries(current_label, boundaries[current_label])
                
                elif cmd == 'plot':
                    plot_label = parts[1] if len(parts) > 1 else current_label
                    if plot_label is None:
                        print("No RDF selected. Use 'plot <label>' or 'select <label>' first")
                        continue
                    if plot_label not in rdf_dict:
                        print(f"Label '{plot_label}' not found")
                        continue
                    
                    # Quick plot
                    self._quick_plot_rdf_with_boundaries(rdf_dict[plot_label], boundaries[plot_label], plot_label)
                
                elif cmd == 'replot':
                    if current_label is None:
                        print("No RDF selected. Use 'select <label>' first")
                        continue
                    # Replot current selection
                    self._quick_plot_rdf_with_boundaries(rdf_dict[current_label], boundaries[current_label], current_label)
                
                elif cmd == 'reset':
                    if len(parts) > 1:
                        reset_label = parts[1].strip('[]').strip("'\"")
                        if reset_label in boundaries:
                            boundaries[reset_label] = original_boundaries[reset_label].copy()
                            print(f"✓ Reset {reset_label} to original boundaries")
                            self._print_boundaries(reset_label, boundaries[reset_label])
                        else:
                            print(f"Label '{reset_label}' not found")
                    elif current_label is not None:
                        boundaries[current_label] = original_boundaries[current_label].copy()
                        print(f"✓ Reset {current_label} to original boundaries")
                        self._print_boundaries(current_label, boundaries[current_label])
                    else:
                        print("No RDF selected. Use 'reset <label>' or 'select <label>' first")
                
                elif cmd == 'expand':
                    if current_label is None:
                        print("No RDF selected. Use 'select <label>' first")
                        continue
                    if len(parts) < 4:
                        print("Usage: expand <shell> <inward|outward|both> <amount>")
                        continue
                    shell_name = parts[1]
                    direction = parts[2].lower()
                    try:
                        amount = float(parts[3])
                        self._expand_shell(boundaries, current_label, shell_name, direction, amount)
                    except ValueError:
                        print("Invalid amount. Use a number.")
                
                elif cmd == 'contract':
                    if current_label is None:
                        print("No RDF selected. Use 'select <label>' first")
                        continue
                    if len(parts) < 4:
                        print("Usage: contract <shell> <inward|outward|both> <amount>")
                        continue
                    shell_name = parts[1]
                    direction = parts[2].lower()
                    try:
                        amount = float(parts[3])
                        self._contract_shell(boundaries, current_label, shell_name, direction, amount)
                    except ValueError:
                        print("Invalid amount. Use a number.")
            
                elif cmd == 'save':
                    if len(parts) < 2:
                        print("Usage: save <filename>")
                        continue
                    filename = parts[1]
                    # Convert tuples to lists for JSON serialization
                    boundaries_json = {}
                    for label, shells in boundaries.items():
                        boundaries_json[label] = {k: list(v) for k, v in shells.items()}
                    with open(filename, 'w') as f:
                        json.dump(boundaries_json, f, indent=2)
                    print(f"✓ Boundaries saved to {filename}")
                
                elif cmd == 'load':
                    if len(parts) < 2:
                        print("Usage: load <filename>")
                        continue
                    filename = parts[1]
                    try:
                        with open(filename, 'r') as f:
                            boundaries_json = json.load(f)
                        
                        # Smart key matching: handle case variations (Ow vs OW)
                        loaded_count = 0
                        skipped_labels = []
                        
                        for json_label, shells in boundaries_json.items():
                            # Try exact match first
                            if json_label in available_labels:
                                boundaries[json_label] = {k: tuple(v) for k, v in shells.items()}
                                loaded_count += 1
                            else:
                                # Try case-insensitive matching for water oxygen labels
                                matched = False
                                for avail_label in available_labels:
                                    # Create normalized versions (Ow/OW -> ow for comparison)
                                    json_norm = json_label.replace('-Ow', '-ow').replace('-OW', '-ow')
                                    avail_norm = avail_label.replace('-Ow', '-ow').replace('-OW', '-ow')
                                    
                                    if json_norm == avail_norm:
                                        # Found a match! Use the current RDF dict's label key
                                        boundaries[avail_label] = {k: tuple(v) for k, v in shells.items()}
                                        loaded_count += 1
                                        matched = True
                                        if json_label != avail_label:
                                            print(f"  • Mapped '{json_label}' → '{avail_label}' (case variant)")
                                        break
                                
                                if not matched:
                                    # No match found in current RDF dict - keep it anyway
                                    boundaries[json_label] = {k: tuple(v) for k, v in shells.items()}
                                    skipped_labels.append(json_label)
                        
                        print(f"✓ Boundaries loaded from {filename}")
                        print(f"  {loaded_count} label(s) matched with current RDF data")
                        
                        if skipped_labels:
                            print(f"  ⚠ {len(skipped_labels)} label(s) not found in current RDF dict:")
                            for lbl in skipped_labels:
                                print(f"    - {lbl}")
                            print("  (These boundaries are loaded but won't be used for plotting)")
                        
                    except FileNotFoundError:
                        print(f"File '{filename}' not found")
                    except json.JSONDecodeError:
                        print(f"Invalid JSON in '{filename}'")
            
                elif cmd == 'help':
                    print("\nCommands:")
                    print("  list, select, add, modify, expand, contract, remove")
                    print("  replot, reset, save, load, quit")
                
                else:
                    print(f"Unknown command: '{cmd}'. Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\n\nExiting boundary editor (Ctrl+C)")
        except Exception as e:
            print(f"\nError in editor: {e}")
            print("Exiting boundary editor")
        
        return boundaries
    
    def _expand_shell(self, boundaries, label, shell_name, direction, amount):
        '''Expand a shell region by specified amount in given direction with cascading adjustments'''
        if shell_name not in boundaries[label]:
            print(f"Shell '{shell_name}' not found")
            available_shells = list(boundaries[label].keys())
            print(f"Available shells: {available_shells}")
            return False
        
        current_start, current_end = boundaries[label][shell_name]
        
        if direction == 'inward':
            # Expand inward (decrease start boundary) - cascades to previous shell
            new_start = max(0, current_start - amount)
            return self._modify_boundary_with_cascade(boundaries[label], shell_name, 'start', new_start)
            
        elif direction == 'outward':
            # Expand outward (increase end boundary) - cascades to next shell
            if np.isinf(current_end):
                print(f"Cannot expand {shell_name}: end boundary is already infinite")
                return False
            new_end = current_end + amount
            return self._modify_boundary_with_cascade(boundaries[label], shell_name, 'end', new_end)
            
        elif direction == 'both':
            # Expand in both directions with cascading
            success_start = True
            success_end = True
            
            # Expand inward first
            new_start = max(0, current_start - amount/2)
            success_start = self._modify_boundary_with_cascade(boundaries[label], shell_name, 'start', new_start)
            
            # Expand outward
            if not np.isinf(current_end):
                # Need to get updated end value after first cascade
                updated_start, updated_end = boundaries[label][shell_name]
                new_end = updated_end + amount/2
                success_end = self._modify_boundary_with_cascade(boundaries[label], shell_name, 'end', new_end)
            else:
                print(f"Cannot expand {shell_name} outward: end boundary is infinite")
                success_end = False
            
            return success_start and success_end
        else:
            print("Direction must be 'inward', 'outward', or 'both'")
            return False
    
    def _contract_shell(self, boundaries, label, shell_name, direction, amount):
        '''Contract a shell region by specified amount in given direction with cascading adjustments'''
        if shell_name not in boundaries[label]:
            print(f"Shell '{shell_name}' not found")
            available_shells = list(boundaries[label].keys())
            print(f"Available shells: {available_shells}")
            return False
        
        current_start, current_end = boundaries[label][shell_name]
        
        if direction == 'inward':
            # Contract inward (increase start boundary) - cascades to previous shell
            new_start = current_start + amount
            if new_start >= current_end:
                print(f"Cannot contract {shell_name}: would make start >= end")
                return False
            return self._modify_boundary_with_cascade(boundaries[label], shell_name, 'start', new_start)
            
        elif direction == 'outward':
            # Contract outward (decrease end boundary) - cascades to next shell
            if np.isinf(current_end):
                print(f"Cannot contract {shell_name}: end boundary is infinite")
                return False
            new_end = current_end - amount
            if new_end <= current_start:
                print(f"Cannot contract {shell_name}: would make end <= start")
                return False
            return self._modify_boundary_with_cascade(boundaries[label], shell_name, 'end', new_end)
            
        elif direction == 'both':
            # Contract in both directions with cascading
            current_width = current_end - current_start if not np.isinf(current_end) else float('inf')
            
            if np.isinf(current_width):
                print(f"Cannot contract {shell_name}: end boundary is infinite")
                return False
                
            if current_width <= amount:
                print(f"Cannot contract {shell_name}: contraction amount ({amount:.2f}) >= current width ({current_width:.2f})")
                return False
            
            success_start = True
            success_end = True
            
            # Contract inward first
            new_start = current_start + amount/2
            success_start = self._modify_boundary_with_cascade(boundaries[label], shell_name, 'start', new_start)
            
            # Contract outward
            # Need to get updated end value after first cascade
            updated_start, updated_end = boundaries[label][shell_name]
            new_end = updated_end - amount/2
            if new_end > updated_start:
                success_end = self._modify_boundary_with_cascade(boundaries[label], shell_name, 'end', new_end)
            else:
                print(f"Cannot contract {shell_name} outward: would make end <= start")
                success_end = False
            
            return success_start and success_end
        else:
            print("Direction must be 'inward', 'outward', or 'both'")
            return False
    
    def _modify_boundary_with_cascade(self, shells, shell_name, boundary_type, new_value):
        '''
        Modify a shell boundary with automatic cascading adjustments to adjacent shells.
        Maintains shell continuity: end of shell_N = start of shell_N+1
        '''
        
        if shell_name not in shells:
            print(f"Shell '{shell_name}' not found")
            return False
        
        current_start, current_end = shells[shell_name]
        
        # Validate the new value
        if boundary_type == 'start':
            if new_value >= current_end:
                print(f"Invalid: new start ({new_value:.2f}) must be < current end ({current_end:.2f})")
                return False
            if new_value < 0:
                print(f"Invalid: new start ({new_value:.2f}) must be >= 0")
                return False
        elif boundary_type == 'end':
            if new_value <= current_start:
                print(f"Invalid: new end ({new_value:.2f}) must be > current start ({current_start:.2f})")
                return False
        else:
            print("Boundary type must be 'start' or 'end'")
            return False
        
        # Determine shell ordering based on shell names
        # Check if these are ion shells (CIP/SIP/DSIP/FI or P1/P2/P3/P4) or water shells (shell_1/shell_2/etc.)
        traditional_ion_order = ['CIP', 'SIP', 'DSIP', 'FI']
        flexible_ion_order = ['P1', 'P2', 'P3', 'P4', 'Bulk']
        
        # Get list of shell names present in shells dict
        present_shells = list(shells.keys())
        
        # Determine shell type and ordering
        is_traditional_ion = any(name in traditional_ion_order for name in present_shells)
        is_flexible_ion = any(name in flexible_ion_order for name in present_shells)
        
        if is_traditional_ion:
            # Use traditional ion order (CIP, SIP, DSIP, FI)
            shell_names = [name for name in traditional_ion_order if name in present_shells]
        elif is_flexible_ion:
            # Use flexible ion order (P1, P2, P3, P4, Bulk)
            shell_names = [name for name in flexible_ion_order if name in present_shells]
        else:
            # Sort water shells by start position (shell_1, shell_2, shell_3, bulk)
            sorted_shells = sorted(shells.items(), key=lambda x: x[1][0])
            shell_names = [name for name, _ in sorted_shells]
        
        # Find current shell index
        try:
            shell_idx = shell_names.index(shell_name)
        except ValueError:
            print(f"Error: Could not find {shell_name} in ordered shells")
            return False
        
        # Apply modification with recursive cascading
        if boundary_type == 'start':
            # Modify start boundary
            shells[shell_name] = (new_value, current_end)
            print(f"Modified {shell_name} start: {current_start:.2f} -> {new_value:.2f} Å")
            
            # Cascade to previous shell (adjust its end to match our new start)
            if shell_idx > 0:
                prev_shell_name = shell_names[shell_idx - 1]
                prev_start, prev_end = shells[prev_shell_name]
                
                # Don't modify if previous shell is bulk with infinity
                if not np.isinf(prev_end):
                    shells[prev_shell_name] = (prev_start, new_value)
                    print(f"  Auto-adjusted {prev_shell_name} end: {prev_end:.2f} -> {new_value:.2f} Å")
                    
                    # Check if previous shell is now invalid (start >= end)
                    if prev_start >= new_value:
                        # Need to push previous shell start earlier
                        new_prev_start = max(0, new_value - 0.5)  # Give it 0.5 Å width
                        print(f"  → {prev_shell_name} became invalid, adjusting start: {prev_start:.2f} -> {new_prev_start:.2f} Å")
                        # Recursively cascade upstream
                        self._modify_boundary_with_cascade(shells, prev_shell_name, 'start', new_prev_start)
        
        elif boundary_type == 'end':
            # Modify end boundary
            shells[shell_name] = (current_start, new_value)
            print(f"Modified {shell_name} end: {current_end:.2f} -> {new_value:.2f} Å")
            
            # Cascade to next shell (adjust its start to match our new end)
            if shell_idx < len(shell_names) - 1:
                next_shell_name = shell_names[shell_idx + 1]
                next_start, next_end = shells[next_shell_name]
                shells[next_shell_name] = (new_value, next_end)
                print(f"  Auto-adjusted {next_shell_name} start: {next_start:.2f} -> {new_value:.2f} Å")
                
                # Check if next shell is now invalid (start >= end) and not infinite
                if not np.isinf(next_end) and new_value >= next_end:
                    # Need to push next shell end further out
                    new_next_end = new_value + 0.5  # Give it 0.5 Å width
                    print(f"  → {next_shell_name} became invalid, adjusting end: {next_end:.2f} -> {new_next_end:.2f} Å")
                    # Recursively cascade downstream
                    self._modify_boundary_with_cascade(shells, next_shell_name, 'end', new_next_end)
        
        return True
    
    def _remove_shell_with_gap_filling(self, shells, shell_to_remove, merge_option='expand_next'):
        '''
        Remove a shell with intelligent gap-filling to maintain boundary continuity.
        
        Parameters
        ----------
        shells : dict
            Dictionary of shell boundaries {shell_name: (start, end)}
        shell_to_remove : str
            Name of shell to remove
        merge_option : str
            How to fill the gap:
            - 'expand_previous': Expand previous shell to include removed shell's range
            - 'expand_next': Expand next shell to include removed shell's range
            - 'expand_both': Split removed shell's range between adjacent shells
        
        Returns
        -------
        success : bool
            Whether removal was successful
        '''
        
        if shell_to_remove not in shells:
            print(f"Shell '{shell_to_remove}' not found")
            return False
        
        # Get shell boundaries
        remove_start, remove_end = shells[shell_to_remove]
        print(f"Removing {shell_to_remove}: {remove_start:.2f} - {remove_end:.2f} Å")
        print(f"Using merge option: {merge_option}")
        
        # Determine shell ordering based on shell names and types
        traditional_ion_order = ['CIP', 'SIP', 'DSIP', 'FI']
        flexible_ion_order = ['P1', 'P2', 'P3', 'P4', 'Bulk']
        
        present_shells = list(shells.keys())
        
        # Determine shell type and get ordering
        is_traditional_ion = any(name in traditional_ion_order for name in present_shells)
        is_flexible_ion = any(name in flexible_ion_order for name in present_shells)
        
        if is_traditional_ion:
            # For traditional ion shells, maintain order but include any custom shells sorted by position
            shell_order = [name for name in traditional_ion_order if name in present_shells]
            # Add any custom shells not in traditional order, sorted by position
            custom_shells = [name for name in present_shells if name not in traditional_ion_order]
            if custom_shells:
                custom_sorted = sorted([(name, shells[name]) for name in custom_shells], key=lambda x: x[1][0])
                # Insert custom shells in appropriate positions based on their boundaries
                for custom_name, (custom_start, custom_end) in custom_sorted:
                    inserted = False
                    for i, existing_name in enumerate(shell_order):
                        existing_start, existing_end = shells[existing_name]
                        if custom_start < existing_start:
                            shell_order.insert(i, custom_name)
                            inserted = True
                            break
                    if not inserted:
                        shell_order.append(custom_name)
        elif is_flexible_ion:
            # For flexible ion shells, maintain order but include any custom shells
            shell_order = [name for name in flexible_ion_order if name in present_shells]
            # Add any custom shells not in flexible order, sorted by position
            custom_shells = [name for name in present_shells if name not in flexible_ion_order]
            if custom_shells:
                custom_sorted = sorted([(name, shells[name]) for name in custom_shells], key=lambda x: x[1][0])
                # Insert custom shells in appropriate positions based on their boundaries
                for custom_name, (custom_start, custom_end) in custom_sorted:
                    inserted = False
                    for i, existing_name in enumerate(shell_order):
                        existing_start, existing_end = shells[existing_name]
                        if custom_start < existing_start:
                            shell_order.insert(i, custom_name)
                            inserted = True
                            break
                    if not inserted:
                        shell_order.append(custom_name)
        else:
            # Sort all shells by start position (including custom names)
            sorted_shells = sorted(shells.items(), key=lambda x: x[1][0])
            shell_order = [name for name, _ in sorted_shells]
        
        # Find shell index
        if shell_to_remove not in shell_order:
            print(f"Error: Could not find {shell_to_remove} in ordered shells")
            return False
        
        shell_idx = shell_order.index(shell_to_remove)
        is_first = (shell_idx == 0)
        is_last = (shell_idx == len(shell_order) - 1)
        
        # Validate merge options for edge cases
        if is_first and merge_option == 'expand_previous':
            print("Warning: Cannot expand previous shell for first shell (no previous shell exists)")
            print("Switching to 'expand_next' option")
            merge_option = 'expand_next'
        
        if is_last and merge_option == 'expand_next':
            print(f"Note: {shell_to_remove} is the last shell.")
            if shell_to_remove in ['FI', 'Bulk']:
                print("Warning: Removing final region - this will eliminate the bulk category")
        
        if (is_first or is_last) and merge_option == 'expand_both':
            print("Warning: Cannot expand both directions for first or last shell")
            if is_first:
                print("Switching to 'expand_next' option")
                merge_option = 'expand_next'
            else:
                print("Switching to 'expand_previous' option") 
                merge_option = 'expand_previous'
        
        # Apply gap-filling logic
        if merge_option == 'expand_previous':
            if not is_first:
                # Expand previous shell to include removed shell's end
                prev_shell = shell_order[shell_idx - 1]
                prev_start, prev_end = shells[prev_shell]
                shells[prev_shell] = (prev_start, remove_end)
                print(f"  Expanded {prev_shell}: {prev_start:.2f} - {remove_end:.2f} Å")
        
        elif merge_option == 'expand_next':
            if is_last:
                # Special case: removing last shell
                if len(shell_order) > 1:
                    # Extend second-to-last shell to infinity or remove_end
                    second_last_shell = shell_order[-2]
                    second_last_start, second_last_end = shells[second_last_shell]
                    new_end = np.inf if remove_end == np.inf else remove_end
                    shells[second_last_shell] = (second_last_start, new_end)
                    end_str = "∞" if np.isinf(new_end) else f"{new_end:.2f}"
                    print(f"  Extended {second_last_shell} to fill gap: {second_last_start:.2f} - {end_str} Å")
            else:
                # Normal case: expand next shell to include removed shell's start
                next_shell = shell_order[shell_idx + 1]
                next_start, next_end = shells[next_shell]
                shells[next_shell] = (remove_start, next_end)
                end_str = "∞" if np.isinf(next_end) else f"{next_end:.2f}"
                print(f"  Expanded {next_shell}: {remove_start:.2f} - {end_str} Å")
        
        elif merge_option == 'expand_both':
            # Split removed shell's range between adjacent shells
            remove_width = remove_end - remove_start
            midpoint = remove_start + remove_width / 2
            
            if not is_first:
                # Expand previous shell to midpoint
                prev_shell = shell_order[shell_idx - 1]
                prev_start, prev_end = shells[prev_shell]
                shells[prev_shell] = (prev_start, midpoint)
                print(f"  Expanded {prev_shell}: {prev_start:.2f} - {midpoint:.2f} Å")
            
            if not is_last:
                # Expand next shell from midpoint
                next_shell = shell_order[shell_idx + 1]
                next_start, next_end = shells[next_shell]
                shells[next_shell] = (midpoint, next_end)
                end_str = "∞" if np.isinf(next_end) else f"{next_end:.2f}"
                print(f"  Expanded {next_shell}: {midpoint:.2f} - {end_str} Å")
        
        # Remove the shell
        del shells[shell_to_remove]
        print(f"Successfully removed {shell_to_remove}")
        
        return True
    
    def _print_boundaries(self, label, shells):
        '''Print boundaries for a specific RDF'''
        if not shells:
            print(f"  {label}: No boundaries defined")
        else:
            print(f"  {label}:")
            for shell_name in sorted(shells.keys()):
                start, end = shells[shell_name]
                print(f"    {shell_name}: {start:.2f} - {end:.2f} Å")
    
    def _quick_plot_rdf_with_boundaries(self, rdf_results, shells, label):
        '''Quick plot of RDF with boundaries for preview'''
        # Extract data
        if hasattr(rdf_results, 'bins'):
            r = rdf_results.bins
            g_r = rdf_results.rdf
        else:
            r = rdf_results['bins']
            g_r = rdf_results['rdf']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot RDF
        ax.plot(r, g_r, 'k-', linewidth=2, label='g(r)')
        
        # Plot shell regions if any
        if shells:
            colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(shells)))
            for i, (shell_name, (start, end)) in enumerate(sorted(shells.items())):
                ax.axvspan(start, end, alpha=0.3, color=colors[i], label=shell_name)
                ax.axvline(start, color=colors[i], linestyle='--', alpha=0.7)
                ax.axvline(end, color=colors[i], linestyle='--', alpha=0.7)
        
        ax.set_xlabel('r (Å)', fontsize=12)
        ax.set_ylabel('g(r)', fontsize=12)
        ax.set_title(f'RDF: {label}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # ... include all other methods from the original implementation ...
    # (protein_ligand_contacts, hydrogen_bond_analysis, molecular_clustering, etc.)