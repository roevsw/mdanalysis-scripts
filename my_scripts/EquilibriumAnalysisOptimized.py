# Optimized EquilibriumAnalysis class with performance improvements
import signal
from datetime import datetime  

# Add signal handling protection at the top
def safe_signal_handler(signum, frame):
    """Safe signal handler that doesn't cause crashes"""
    pass

# Protect against signal handling issues
try:
    signal.signal(signal.SIGTERM, safe_signal_handler)
    signal.signal(signal.SIGINT, safe_signal_handler)
except (OSError, ValueError):
    # Signal handling may not work in all environments
    pass

import pickle  
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

from tqdm import tqdm

import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.base import Results

import multiprocessing
from multiprocessing import Pool
from functools import partial

# Fix the scipy imports
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist  # Correct import location
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

import psutil
import warnings
import gc

# Fix the import path
import sys
import os
sys.path.append('/Users/roev0007/Documents/solvation_shells/solvation_shells')
sys.path.append('/Users/roev0007/Documents/solvation_shells')

from solvation_analysis.solute import Solute
from utils.linear_algebra import *
from utils.file_rw import vdW_radii
from utils.ParallelMDAnalysis import ParallelInterRDF as InterRDF

# Change this import to avoid the relative import issue
# from .equilibrium_analysis import EquilibriumAnalysis
from equilibrium_analysis import EquilibriumAnalysis

def _test_worker(x):
    """Simple worker function for testing multiprocessing"""
    return x

def _worker_init():
    """Worker pool initializer: restore default SIGTERM in each spawned worker.
    gsd registers signal.signal(SIGTERM, lambda n, f: sys.exit(1)) on import,
    overwriting our safe_signal_handler. This runs after all imports in each
    worker process so the default (clean exit) handler wins instead."""
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

class MockSolute:
    '''Pickleable mock solute class for quick initialization'''
    
    def __init__(self, name, radius):
        self.name = name
        self.radii = {'water': radius}
    
    def __repr__(self):
        return f"MockSolute(name='{self.name}', radius={self.radii['water']:.2f})"


class EquilibriumAnalysisOptimized(EquilibriumAnalysis):
    '''
    Optimized version of EquilibriumAnalysis with performance improvements for large systems.
    
    Key optimizations:
    - Memory management and caching
    - Vectorized calculations
    - Parallel processing improvements
    - KDTree-based neighbor searches
    - Batch processing for large trajectories
    - Smart parameter tuning based on system size
    '''


    def __init__(self, top, traj, water='type OW', cation='resname NA', anion='resname CL', 
                start_frame=None, end_frame=None, step_frame=None, max_frames=None, debug=False):
        '''
        Initialize optimized equilibrium analysis with caching and memory management.
        
        Parameters
        ----------
        top : str
            Topology file
        traj : str or list
            Trajectory file(s)
        water : str
            MDAnalysis selection for water
        cation : str
            MDAnalysis selection for cations
        anion : str
            MDAnalysis selection for anions
        start_frame : int, optional
            Starting frame for analysis (for debugging)
        end_frame : int, optional
            Ending frame for analysis (for debugging)
        step_frame : int, optional
            Step between frames (for debugging)
        max_frames : int, optional
            Maximum number of frames to load (for debugging)
        debug : bool, optional
            Enable debug mode (limits frames). Default is False.
        '''
        
        # Initialize universe
        self.universe = mda.Universe(top, traj)
        
        # Store original trajectory info for debugging
        original_length = len(self.universe.trajectory)
        
        # FIXED: Apply trajectory slicing ONLY if debug=True (not if frame params are provided)
        # Debug mode is active ONLY if debug=True explicitly
        debug_mode_active = debug
        
        if debug_mode_active:
            # Apply default debug limits if not specified
            if max_frames is not None:
                end_frame = min(max_frames, original_length)
                start_frame = 0 if start_frame is None else start_frame
            
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = min(10001, original_length)  # Default debug limit
            if step_frame is None:
                step_frame = 1
            
            print(f"DEBUG MODE: Loading frames {start_frame}:{end_frame}:{step_frame}")
            print(f"  Original trajectory: {original_length} frames")
            
            # Create a new trajectory reader with the slice instead of modifying the existing one
            try:
                # Method 1: Try using start/stop/step parameters if available
                if hasattr(self.universe.trajectory, 'start'):
                    self.universe.trajectory.start = start_frame
                    self.universe.trajectory.stop = end_frame  
                    self.universe.trajectory.step = step_frame
                else:
                    # Method 2: Create index array and iterate manually
                    self._debug_frame_indices = list(range(start_frame, end_frame, step_frame))
                    print(f"  Using manual frame indexing: {len(self._debug_frame_indices)} frames")
            
            except Exception as e:
                print(f"  Warning: Could not apply trajectory slicing: {e}")
                print(f"  Continuing with full trajectory...")
                self._debug_frame_indices = None
        else:
            # NOT in debug mode - use all frames or user-specified range
            self._debug_frame_indices = None
            
            # Apply user-specified frame range without printing "DEBUG MODE"
            if start_frame is not None or end_frame is not None or step_frame is not None:
                # User wants to limit frames but NOT in debug mode
                if start_frame is None:
                    start_frame = 0
                if end_frame is None:
                    end_frame = original_length
                if step_frame is None:
                    step_frame = 1
                
                # Only apply if different from full trajectory
                if start_frame != 0 or end_frame != original_length or step_frame != 1:
                    print(f"Using custom frame range: {start_frame}:{end_frame}:{step_frame}")
                    print(f"  Original trajectory: {original_length} frames")
                    
                    try:
                        if hasattr(self.universe.trajectory, 'start'):
                            self.universe.trajectory.start = start_frame
                            self.universe.trajectory.stop = end_frame
                            self.universe.trajectory.step = step_frame
                        else:
                            self._debug_frame_indices = list(range(start_frame, end_frame, step_frame))
                            print(f"  Using manual frame indexing: {len(self._debug_frame_indices)} frames")
                    except Exception as e:
                        print(f"  Warning: Could not apply frame range: {e}")
            
            print(f"Loading full trajectory: {original_length} frames")
        
        # Initialize atom groups
        self.waters = self.universe.select_atoms(water)
        self.cations = self.universe.select_atoms(cation)
        self.anions = self.universe.select_atoms(anion)
        
        # Store trajectory info
        if self._debug_frame_indices is not None:
            self.n_frames = len(self._debug_frame_indices)
            print(f"  Debug trajectory: {self.n_frames} frames")
        else:
            self.n_frames = len(self.universe.trajectory)
        
        print(f'Loaded system with:')
        print(f'  {len(self.waters)} waters')
        print(f'  {len(self.cations)} cations')
        print(f'  {len(self.anions)} anions')
        print(f'  {self.n_frames} frames')
        
        # Add caching and memory management
        self._cache = {}
        self._cache_enabled = True
        self.max_cache_size = 200  # MB
        
        # Auto-configure based on system size
        self._auto_configure_for_system_size()
        
        # Pre-compute frequently used data
        self._precompute_static_data()
        
        print(f"EquilibriumAnalysisOptimized initialized with auto-tuned parameters")


    def _auto_configure_for_system_size(self):
        '''Automatically configure parameters based on system size'''
        
        n_atoms = len(self.universe.atoms)
        n_frames = len(self.universe.trajectory)
        
        if n_atoms > 100000:  # Large system
            self.default_step = 10
            self.default_njobs = -1
            self.water_step = max(100, n_frames // 50)
            self.batch_size = 50
            print(f"Large system detected ({n_atoms} atoms): Using optimized parameters")
            
        elif n_atoms > 50000:  # Medium-large system
            self.default_step = 5
            self.default_njobs = multiprocessing.cpu_count()
            self.water_step = max(50, n_frames // 75)
            self.batch_size = 100
            print(f"Medium-large system detected ({n_atoms} atoms): Using balanced parameters")
            
        elif n_atoms > 10000:  # Medium system  
            self.default_step = 2
            self.default_njobs = multiprocessing.cpu_count() // 2
            self.water_step = max(20, n_frames // 100)
            self.batch_size = 200
            print(f"Medium system detected ({n_atoms} atoms): Using standard parameters")
            
        else:  # Small system
            self.default_step = 1
            self.default_njobs = 1
            self.water_step = 10
            self.batch_size = 500
            print(f"Small system detected ({n_atoms} atoms): Using detailed parameters")

    def _precompute_static_data(self):
        '''Pre-compute static data that won\'t change across trajectory'''
        # Cache atom indices for faster selection
        self._water_indices = self.waters.indices
        self._cation_indices = self.cations.indices 
        self._anion_indices = self.anions.indices
        
        # Pre-allocate distance matrix cache
        self._distance_matrix_cache = {}
        
        print(f"Pre-computed static data for {len(self.waters)} waters, {len(self.cations)} cations, {len(self.anions)} anions")


    def calculate_salt_concentration(self, verbose=True):
        '''
        Calculate the salt concentration in the simulation box based on the number of ions and water molecules.
        
        Strategy:
        1. Identify all cation types present (Na, Mg, K, Ca, etc.)
        2. Pair them with appropriate anions based on stoichiometry
        3. Calculate molar concentration based on box volume
        
        Parameters
        ----------
        verbose : bool
            Whether to print detailed breakdown, default=True
        
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'total_concentration_M': Total salt concentration in Molarity (M)
            - 'salt_breakdown': Dictionary of individual salt concentrations
            - 'n_waters': Number of water molecules
            - 'box_volume_L': Box volume in liters
            - 'composition': Human-readable composition string
        '''
        
        # Get box volume from universe
        # dimensions = [a, b, c, alpha, beta, gamma]
        dimensions = self.universe.dimensions
        
        # Calculate volume in Å³
        box_volume_A3 = dimensions[0] * dimensions[1] * dimensions[2]
        
        # Convert to liters (1 Å³ = 1e-27 L)
        box_volume_L = box_volume_A3 * 1e-27
        
        # Get unique ion types
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        # Count ions by type
        cation_counts = {ion_type: len(group) for ion_type, group in cation_types.items()}
        anion_counts = {ion_type: len(group) for ion_type, group in anion_types.items()}
        
        # Number of water molecules
        n_waters = len(self.waters)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"SALT CONCENTRATION CALCULATION")
            print(f"{'='*70}")
            print(f"Box dimensions: {dimensions[0]:.2f} × {dimensions[1]:.2f} × {dimensions[2]:.2f} Å")
            print(f"Box volume: {box_volume_A3:.2f} Å³ = {box_volume_L:.6e} L")
            print(f"Water molecules: {n_waters}")
            print(f"\nIon inventory:")
            print(f"  Cations: {cation_counts}")
            print(f"  Anions: {anion_counts}")
        
        # Define charge states for common ions
        ion_charges = {
            'Na': +1, 'K': +1, 'Li': +1, 'Rb': +1, 'Cs': +1,
            'Mg': +2, 'Ca': +2, 'Sr': +2, 'Ba': +2,
            'Al': +3,
            'Cl': -1, 'Br': -1, 'F': -1, 'I': -1,
            'SO4': -2, 'CO3': -2
        }
        
        # Calculate salt concentrations
        salt_breakdown = {}
        total_ion_pairs = 0
        
        # Strategy: Pair cations with anions based on charge balance
        # Priority: Non-Na cations first, then Na with remaining Cl
        
        remaining_anions = anion_counts.copy()
        
        # First, pair all non-Na cations with anions
        for cation_type, n_cations in cation_counts.items():
            if cation_type == 'Na':
                continue  # Handle Na last
            
            cation_charge = abs(ion_charges.get(cation_type, 1))
            
            # Find available anions
            for anion_type, n_anions in remaining_anions.items():
                if n_anions == 0:
                    continue
                
                anion_charge = abs(ion_charges.get(anion_type, 1))
                
                # Calculate stoichiometry
                # For example: Mg²⁺ + 2Cl⁻ → MgCl₂
                cation_coeff = anion_charge  # How many cations per formula unit
                anion_coeff = cation_charge   # How many anions per formula unit
                
                # How many formula units can we make?
                max_formula_units = min(n_cations // cation_coeff, 
                                    n_anions // anion_coeff)
                
                if max_formula_units > 0:
                    # Create salt name
                    if cation_coeff == 1 and anion_coeff == 1:
                        salt_name = f"{cation_type}{anion_type}"
                    elif cation_coeff == 1:
                        salt_name = f"{cation_type}{anion_type}{anion_coeff}"
                    elif anion_coeff == 1:
                        salt_name = f"{cation_type}{cation_coeff}{anion_type}"
                    else:
                        salt_name = f"{cation_type}{cation_coeff}{anion_type}{anion_coeff}"
                    
                    # Calculate moles and concentration
                    n_formula_units = max_formula_units
                    moles = n_formula_units / 6.022e23  # Avogadro's number
                    concentration_M = moles / box_volume_L
                    
                    salt_breakdown[salt_name] = {
                        'n_formula_units': n_formula_units,
                        'concentration_M': concentration_M,
                        'cation': cation_type,
                        'anion': anion_type,
                        'cation_coeff': cation_coeff,
                        'anion_coeff': anion_coeff
                    }
                    
                    # Update remaining ions
                    remaining_anions[anion_type] -= n_formula_units * anion_coeff
                    total_ion_pairs += n_formula_units
        
        # Now handle Na with remaining Cl (or other anions)
        if 'Na' in cation_counts:
            n_na = cation_counts['Na']
            
            for anion_type, n_anions in remaining_anions.items():
                if n_anions == 0:
                    continue
                
                anion_charge = abs(ion_charges.get(anion_type, 1))
                
                # Stoichiometry for Na salt
                na_coeff = anion_charge
                anion_coeff = 1  # Na is always +1
                
                max_formula_units = min(n_na // na_coeff, n_anions // anion_coeff)
                
                if max_formula_units > 0:
                    # Create salt name
                    if na_coeff == 1:
                        salt_name = f"Na{anion_type}"
                    else:
                        salt_name = f"Na{na_coeff}{anion_type}"
                    
                    n_formula_units = max_formula_units
                    moles = n_formula_units / 6.022e23
                    concentration_M = moles / box_volume_L
                    
                    salt_breakdown[salt_name] = {
                        'n_formula_units': n_formula_units,
                        'concentration_M': concentration_M,
                        'cation': 'Na',
                        'anion': anion_type,
                        'cation_coeff': na_coeff,
                        'anion_coeff': anion_coeff
                    }
                    
                    remaining_anions[anion_type] -= n_formula_units * anion_coeff
                    total_ion_pairs += n_formula_units
        
        # Calculate total concentration
        total_concentration_M = sum(salt['concentration_M'] for salt in salt_breakdown.values())
        
        # Create human-readable composition
        composition_parts = []
        for salt_name, salt_data in salt_breakdown.items():
            composition_parts.append(f"{salt_data['concentration_M']:.3f}M {salt_name}")
        composition_str = " + ".join(composition_parts) if composition_parts else "No salts detected"
        
        # Print results
        if verbose:
            print(f"\n{'='*70}")
            print(f"SALT COMPOSITION:")
            print(f"{'='*70}")
            
            for salt_name, salt_data in salt_breakdown.items():
                print(f"\n{salt_name}:")
                print(f"  Formula units: {salt_data['n_formula_units']}")
                print(f"  Stoichiometry: {salt_data['cation']}{salt_data['cation_coeff'] if salt_data['cation_coeff'] > 1 else ''}{salt_data['anion']}{salt_data['anion_coeff'] if salt_data['anion_coeff'] > 1 else ''}")
                print(f"  Concentration: {salt_data['concentration_M']:.4f} M")
            
            print(f"\n{'='*70}")
            print(f"TOTAL CONCENTRATION: {total_concentration_M:.4f} M")
            print(f"{'='*70}")
            print(f"Composition: {composition_str}")
            print(f"{'='*70}")
        
        # Prepare results dictionary
        results = {
            'total_concentration_M': total_concentration_M,
            'salt_breakdown': salt_breakdown,
            'n_waters': n_waters,
            'box_volume_L': box_volume_L,
            'box_volume_A3': box_volume_A3,
            'composition': composition_str,
            'cation_counts': cation_counts,
            'anion_counts': anion_counts
        }
        
        return results


    def save_solutes_to_file(self, filename='solutes_cache.pkl'):
        '''Save initialized solutes to file for persistence across sessions'''
        
        if hasattr(self, 'solutes_ci') and hasattr(self, 'solutes_ai'):
            # Convert solutes to a pickleable format
            solute_data = {
                'cations': {},
                'anions': {},
                'combined_ci': None,
                'combined_ai': None
            }
            
            # Process cation solutes
            for ion_type, solute in self.solutes_ci.items():
                if solute is not None:
                    if hasattr(solute, 'radii') and 'water' in solute.radii:
                        # Store just the essential data
                        solute_data['cations'][ion_type] = {
                            'name': ion_type,
                            'radius': solute.radii['water'],
                            'type': 'mock' if hasattr(solute, 'name') else 'full'
                        }
                    else:
                        solute_data['cations'][ion_type] = None
                else:
                    solute_data['cations'][ion_type] = None
            
            # Process anion solutes
            for ion_type, solute in self.solutes_ai.items():
                if solute is not None:
                    if hasattr(solute, 'radii') and 'water' in solute.radii:
                        # Store just the essential data
                        solute_data['anions'][ion_type] = {
                            'name': ion_type,
                            'radius': solute.radii['water'],
                            'type': 'mock' if hasattr(solute, 'name') else 'full'
                        }
                    else:
                        solute_data['anions'][ion_type] = None
                else:
                    solute_data['anions'][ion_type] = None
            
            # Store combined solute info
            if hasattr(self, 'solute_ci') and self.solute_ci is not None:
                if hasattr(self.solute_ci, 'radii') and 'water' in self.solute_ci.radii:
                    solute_data['combined_ci'] = {
                        'radius': self.solute_ci.radii['water'],
                        'source': getattr(self.solute_ci, 'name', 'unknown')
                    }
            
            if hasattr(self, 'solute_ai') and self.solute_ai is not None:
                if hasattr(self.solute_ai, 'radii') and 'water' in self.solute_ai.radii:
                    solute_data['combined_ai'] = {
                        'radius': self.solute_ai.radii['water'],
                        'source': getattr(self.solute_ai, 'name', 'unknown')
                    }
            
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(solute_data, f)
                
                print(f"Solutes saved to {filename}")
                print(f"  Saved {len([v for v in solute_data['cations'].values() if v is not None])} cation types")
                print(f"  Saved {len([v for v in solute_data['anions'].values() if v is not None])} anion types")
                
            except Exception as e:
                print(f"Error saving solutes: {e}")
                return False
                
            return True
        else:
            print("No solutes to save")
            return False

    def load_solutes_from_file(self, filename='solutes_cache.pkl'):
        '''Load initialized solutes from file with better error handling'''
        
        if os.path.exists(filename):
            try:
                # Check file size first
                file_size = os.path.getsize(filename)
                if file_size == 0:
                    print(f"Cache file {filename} is empty, skipping...")
                    return False
                
                with open(filename, 'rb') as f:
                    solute_data = pickle.load(f)
                
                # Validate the loaded data structure
                if not isinstance(solute_data, dict):
                    print(f"Invalid cache file format in {filename}")
                    return False
                
                if 'cations' not in solute_data or 'anions' not in solute_data:
                    print(f"Incomplete cache file format in {filename}")
                    return False
                
                # Reconstruct solute objects
                self.solutes_ci = {}
                self.solutes_ai = {}
                
                # Reconstruct cation solutes
                for ion_type, data in solute_data.get('cations', {}).items():
                    if data is not None and isinstance(data, dict) and 'radius' in data:
                        self.solutes_ci[ion_type] = MockSolute(data['name'], data['radius'])
                    else:
                        self.solutes_ci[ion_type] = None
                
                # Reconstruct anion solutes
                for ion_type, data in solute_data.get('anions', {}).items():
                    if data is not None and isinstance(data, dict) and 'radius' in data:
                        self.solutes_ai[ion_type] = MockSolute(data['name'], data['radius'])
                    else:
                        self.solutes_ai[ion_type] = None
                
                # Reconstruct combined solutes
                combined_ci_data = solute_data.get('combined_ci')
                if combined_ci_data is not None and isinstance(combined_ci_data, dict):
                    self.solute_ci = MockSolute(combined_ci_data.get('source', 'combined_cation'), 
                                            combined_ci_data['radius'])
                else:
                    self.solute_ci = None
                
                combined_ai_data = solute_data.get('combined_ai')
                if combined_ai_data is not None and isinstance(combined_ai_data, dict):
                    self.solute_ai = MockSolute(combined_ai_data.get('source', 'combined_anion'), 
                                            combined_ai_data['radius'])
                else:
                    self.solute_ai = None
                
                print(f"Solutes loaded from {filename}")
                print(f"  Loaded {len([v for v in self.solutes_ci.values() if v is not None])} cation types")
                print(f"  Loaded {len([v for v in self.solutes_ai.values() if v is not None])} anion types")
                
                if hasattr(self, 'print_coordination_radii_summary'):
                    self.print_coordination_radii_summary()
                return True
                
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Cache file {filename} is corrupted: {e}")
                print("Removing corrupted cache file...")
                try:
                    os.remove(filename)
                    print(f"Corrupted file {filename} removed")
                except:
                    pass
                return False
            except Exception as e:
                print(f"Error loading solutes from {filename}: {e}")
                return False
        else:
            print(f"File {filename} not found")
            return False

    def safe_initialization_workflow(self, use_cache=True, cache_filename='solutes_cache.pkl'):
        '''
        Safe workflow for solute initialization with automatic fallback and cleanup.
        '''
        
        print("=== SAFE SOLUTE INITIALIZATION ===")
        
        # Try to load from cache first
        if use_cache:
            print("1. Attempting to load from cache...")
            if self.load_solutes_from_file(cache_filename):
                print("✓ Successfully loaded from cache")
                return self.solutes_ci, self.solutes_ai
            else:
                print("✗ Cache loading failed or not available")
        
        # Fallback to quick initialization
        print("2. Using quick initialization with defaults...")
        try:
            self.quick_initialize_solutes_with_defaults()
            print("✓ Quick initialization successful")
            
            # Try to save to cache
            print("3. Saving to cache for future use...")
            if self.save_solutes_to_file(cache_filename):
                print("✓ Cache saved successfully")
            else:
                print("✗ Cache saving failed, but continuing...")
                
            return self.solutes_ci, self.solutes_ai
            
        except Exception as e:
            print(f"✗ Quick initialization failed: {e}")
            print("4. Attempting full initialization as last resort...")
            
            try:
                # Last resort - full initialization with minimal parameters
                self.initialize_Solutes_by_type(step=5, force_recalc=True)
                print("✓ Full initialization successful")
                
                # Save to cache
                if self.save_solutes_to_file(cache_filename):
                    print("✓ Cache saved successfully")
                    
                return self.solutes_ci, self.solutes_ai
                
            except Exception as e2:
                print(f"✗ All initialization methods failed: {e2}")
                return None, None

    def clear_cache_files(self, cache_dir='.'):
        '''Clear all cache files in the specified directory'''
        
        cache_patterns = ['solutes_cache.*', '*_solvation_shells*.png', '*_rdfs*.png']
        
        removed_files = []
        
        for pattern in cache_patterns:
            files = glob.glob(os.path.join(cache_dir, pattern))
            for file in files:
                try:
                    os.remove(file)
                    removed_files.append(file)
                except Exception as e:
                    print(f"Could not remove {file}: {e}")
        
        if removed_files:
            print(f"Removed cache files: {removed_files}")
        else:
            print("No cache files found to remove")
        
        return removed_files

    def validate_system_integrity(self):
        '''Validate that all system components are properly initialized'''
        
        print("\n=== SYSTEM INTEGRITY CHECK ===")
        
        checks = {
            'Universe': hasattr(self, 'universe') and self.universe is not None,
            'Waters': hasattr(self, 'waters') and len(self.waters) > 0,
            'Cations': hasattr(self, 'cations') and len(self.cations) > 0,
            'Anions': hasattr(self, 'anions') and len(self.anions) > 0,
            'Trajectory': hasattr(self, 'n_frames') and self.n_frames > 0,
            'Cation Solutes': hasattr(self, 'solutes_ci') and len(self.solutes_ci) > 0,
            'Anion Solutes': hasattr(self, 'solutes_ai') and len(self.solutes_ai) > 0,
            'RDFs': hasattr(self, 'rdfs') and len(self.rdfs) > 0,
        }
        
        all_good = True
        for component, status in checks.items():
            status_symbol = "✓" if status else "✗"
            print(f"{status_symbol} {component}")
            if not status:
                all_good = False
        
        if all_good:
            print("✓ All systems operational!")
        else:
            print("✗ Some components need attention")
        
        return all_good


    def save_full_solutes_to_file(self, filename='full_solutes_cache.pkl'):
        '''
        Save full Solute objects (with speciation data) to file for persistence.
        Excludes very large arrays to keep file size manageable.
        
        Parameters
        ----------
        filename : str
            Output filename, default='full_solutes_cache.pkl'
        
        Returns
        -------
        success : bool
            True if save was successful
        '''
        
        if not hasattr(self, 'solutes_ci') or not hasattr(self, 'solutes_ai'):
            print("No solutes to save")
            return False
        
        try:
            # Prepare solute data - save the actual Solute objects
            solute_data = {
                'cations': {},
                'anions': {},
                'combined_ci': None,
                'combined_ai': None,
                'metadata': {
                    'saved_date': datetime.now().isoformat(),
                    'n_frames': self.n_frames,
                    'trajectory_length': len(self.universe.trajectory)
                }
            }
            
            # Save cation solutes (full objects)
            for ion_type, solute in self.solutes_ci.items():
                if solute is not None:
                    # Store the full solute object
                    # Note: This will include speciation data
                    solute_data['cations'][ion_type] = solute
                else:
                    solute_data['cations'][ion_type] = None
            
            # Save anion solutes (full objects)
            for ion_type, solute in self.solutes_ai.items():
                if solute is not None:
                    solute_data['anions'][ion_type] = solute
                else:
                    solute_data['anions'][ion_type] = None
            
            # Save combined solutes if they exist
            if hasattr(self, 'solute_ci') and self.solute_ci is not None:
                solute_data['combined_ci'] = self.solute_ci
            
            if hasattr(self, 'solute_ai') and self.solute_ai is not None:
                solute_data['combined_ai'] = self.solute_ai
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(solute_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Check file size
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            
            print(f"Full Solute objects saved to {filename}")
            print(f"  File size: {file_size_mb:.1f} MB")
            print(f"  Saved {len([v for v in solute_data['cations'].values() if v is not None])} cation types")
            print(f"  Saved {len([v for v in solute_data['anions'].values() if v is not None])} anion types")
            print(f"  ⚠️  Note: Full solutes include speciation data and may be large")
            
            return True
            
        except Exception as e:
            print(f"Error saving full solutes: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_full_solutes_from_file(self, filename='full_solutes_cache.pkl'):
        '''
        Load full Solute objects (with speciation data) from file.
        
        Parameters
        ----------
        filename : str
            Input filename, default='full_solutes_cache.pkl'
        
        Returns
        -------
        success : bool
            True if load was successful
        '''
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        try:
            # Check file size
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"Loading full solutes from {filename} ({file_size_mb:.1f} MB)...")
            
            if file_size_mb == 0:
                print(f"Cache file {filename} is empty")
                return False
            
            # Load data
            with open(filename, 'rb') as f:
                solute_data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(solute_data, dict):
                print(f"Invalid full solute cache format")
                return False
            
            # Load cation solutes
            self.solutes_ci = {}
            for ion_type, solute in solute_data.get('cations', {}).items():
                self.solutes_ci[ion_type] = solute
            
            # Load anion solutes
            self.solutes_ai = {}
            for ion_type, solute in solute_data.get('anions', {}).items():
                self.solutes_ai[ion_type] = solute
            
            # Load combined solutes
            if 'combined_ci' in solute_data and solute_data['combined_ci'] is not None:
                self.solute_ci = solute_data['combined_ci']
            
            if 'combined_ai' in solute_data and solute_data['combined_ai'] is not None:
                self.solute_ai = solute_data['combined_ai']
            
            # Print summary
            successful_cations = [k for k, v in self.solutes_ci.items() if v is not None]
            successful_anions = [k for k, v in self.solutes_ai.items() if v is not None]
            
            print(f"Full Solute objects loaded from {filename}")
            print(f"  Loaded {len(successful_cations)} cation types: {successful_cations}")
            print(f"  Loaded {len(successful_anions)} anion types: {successful_anions}")
            
            # Check if solutes have speciation data
            print(f"\n  Checking speciation data availability:")
            for ion_type in successful_cations + successful_anions:
                solute = self.solutes_ci.get(ion_type) or self.solutes_ai.get(ion_type)
                has_speciation = hasattr(solute, 'speciation') and solute.speciation is not None
                has_fraction = has_speciation and hasattr(solute.speciation, 'speciation_fraction')
                
                status = "✓" if has_fraction else "✗"
                print(f"    {status} {ion_type}: speciation={'Yes' if has_speciation else 'No'}, fraction={'Yes' if has_fraction else 'No'}")
            
            return True
            
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            print("Removing corrupted cache file...")
            try:
                os.remove(filename)
                print(f"Corrupted file {filename} removed")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading full solutes from {filename}: {e}")
            import traceback
            traceback.print_exc()
            return False


    def clear_cache(self):
        '''Clear cached data to free memory'''
        self._cache.clear()
        self._distance_matrix_cache.clear()
        gc.collect()
        print("Cache cleared and garbage collection performed")

    def get_memory_usage(self):
        '''Monitor memory usage for optimization'''
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Current memory usage: {memory_mb:.1f} MB")
        return memory_mb

    def _test_multiprocessing_compatibility(self, njobs):
        """
        Test if multiprocessing works properly in the current environment.
        
        Returns
        -------
        njobs : int
            Number of jobs to use (1 if multiprocessing fails)
        """
        
        if njobs <= 1:
            return 1
        
        try:
            # Test with the global function (not self._test_multiprocessing_worker)
            with Pool(min(2, njobs), initializer=_worker_init) as test_pool:
                test_result = test_pool.map(_test_worker, [1, 2])  # Use global function
                
            # Test with a slightly more complex operation
            with Pool(min(2, njobs), initializer=_worker_init) as test_pool:
                test_result = test_pool.map(np.sqrt, [4, 9])
                
            return njobs  # Multiprocessing works
            
        except BaseException as e:
            # Catch BaseException (not just Exception) so that SystemExit raised
            # by gsd's SIGTERM handler inside workers doesn't escape to the caller.
            print(f"Multiprocessing test failed: {e}")
            print("Falling back to single-threaded mode")
            return 1

    def generate_rdfs(self, bin_width=0.05, range=(0,20), step=None, filename=None, njobs=None, water_step=None, separate_ion_types=True):
        '''
        Optimized RDF calculation with better memory management and parallel processing.
        Now supports separate RDFs for different ion types.
        
        Parameters
        ----------
        separate_ion_types : bool
            If True, calculate separate RDFs for different cation/anion types, default=True
        '''
        
        # Use auto-tuned defaults if not specified
        if step is None:
            step = self.default_step
        if njobs is None:
            njobs = self.default_njobs
        if water_step is None:
            water_step = self.water_step
        
        # Use all available CPUs if -1, but cap to avoid signal issues
        if njobs == -1:
            njobs = min(multiprocessing.cpu_count(), 8)

        # Test if multiprocessing works properly
        njobs = self._test_multiprocessing_compatibility(njobs)    

        # Force single-threaded ONLY if debug mode is active AND we have debug frames
        # Check both the debug flag AND the debug_frame_indices
        debug_mode_active = getattr(self, 'debug', False) or (
            hasattr(self, '_debug_frame_indices') and 
            self._debug_frame_indices is not None and
            len(self._debug_frame_indices) < len(self.universe.trajectory)
        )
        
        if debug_mode_active:
            njobs = 1
            print("Debug mode detected: Using single-threaded calculation")
        
        nbins = int((range[1] - range[0]) / bin_width)
        self.rdfs = {}
        
        print(f'\nCalculating RDFs with optimized parameters:')
        print(f'  Step: {step}, Water step: {water_step}, CPUs: {njobs}')
        print(f'  Separate ion types: {separate_ion_types}')
        print('  Memory usage before:', end=' ')
        self.get_memory_usage()
        
        if separate_ion_types:
            # Get unique ion types
            cation_types = self._get_unique_ion_types(self.cations)
            anion_types = self._get_unique_ion_types(self.anions)
            
            print(f"\nFound ion types:")
            print(f"  Cations: {list(cation_types.keys())}")
            print(f"  Anions: {list(anion_types.keys())}")
            
            # Calculate RDFs for each cation type with water
            for cation_name, cation_group in cation_types.items():
                rdf_name = f'{cation_name}-w'
                print(f'Calculating {rdf_name} RDF ({len(cation_group)} ions, step={step})...')
                
                try:
                    rdf = InterRDF(cation_group, self.waters, nbins=nbins, range=range, 
                                norm='rdf', verbose=False)
                    
                    # Handle debug trajectory
                    if self._debug_frame_indices is not None:
                        # Manual frame iteration for debug mode
                        rdf_results = self._calculate_rdf_manual(rdf, self._debug_frame_indices)
                        rdf.results = rdf_results
                    else:
                        # Normal calculation
                        if njobs == 1:
                            rdf.run(step=step)
                        else:
                            try:
                                rdf.run(step=step, njobs=njobs)
                            except Exception as e:
                                print(f"Multiprocessing failed for {rdf_name}: {e}")
                                print("Retrying with single thread...")
                                rdf.run(step=step, njobs=1)
                    
                    self.rdfs[rdf_name] = rdf.results
                    del rdf
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error calculating {rdf_name} RDF: {e}")
                    self.rdfs[rdf_name] = None
            
            # Calculate RDFs for each anion type with water
            for anion_name, anion_group in anion_types.items():
                rdf_name = f'{anion_name}-w'
                print(f'Calculating {anion_name} RDF ({len(anion_group)} ions, step={step})...')
                
                try:
                    rdf = InterRDF(anion_group, self.waters, nbins=nbins, range=range, 
                                norm='rdf', verbose=False)
                    
                    # Handle debug trajectory
                    if self._debug_frame_indices is not None:
                        # Manual frame iteration for debug mode
                        rdf_results = self._calculate_rdf_manual(rdf, self._debug_frame_indices)
                        rdf.results = rdf_results
                    else:
                        # Normal calculation
                        if njobs == 1:
                            rdf.run(step=step)
                        else:
                            try:
                                rdf.run(step=step, njobs=njobs)
                            except Exception as e:
                                print(f"Multiprocessing failed for {rdf_name}: {e}")
                                print("Retrying with single thread...")
                                rdf.run(step=step, njobs=1)
                    
                    self.rdfs[rdf_name] = rdf.results
                    del rdf
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error calculating {rdf_name} RDF: {e}")
                    self.rdfs[rdf_name] = None
            
            # Calculate cross-cation-anion RDFs
            for cation_name, cation_group in cation_types.items():
                for anion_name, anion_group in anion_types.items():
                    rdf_name = f'{cation_name}-{anion_name}'
                    print(f'Calculating {rdf_name} RDF (step={step})...')
                    
                    try:
                        rdf = InterRDF(cation_group, anion_group, nbins=nbins, range=range, 
                                    norm='rdf', verbose=False)
                        
                        # Handle debug trajectory
                        if self._debug_frame_indices is not None:
                            rdf_results = self._calculate_rdf_manual(rdf, self._debug_frame_indices)
                            rdf.results = rdf_results
                        else:
                            if njobs == 1:
                                rdf.run(step=step)
                            else:
                                try:
                                    rdf.run(step=step, njobs=njobs)
                                except Exception as e:
                                    print(f"Multiprocessing failed for {rdf_name}: {e}")
                                    print("Retrying with single thread...")
                                    rdf.run(step=step, njobs=1)
                        
                        self.rdfs[rdf_name] = rdf.results
                        del rdf
                        gc.collect()
                        
                    except Exception as e:
                        print(f"Error calculating {rdf_name} RDF: {e}")
                        self.rdfs[rdf_name] = None
            
            # Also calculate combined RDFs for backward compatibility
            combined_rdfs = [
                ('ci-w', self.cations, self.waters, step),
                ('ai-w', self.anions, self.waters, step),
                ('ci-ai', self.cations, self.anions, step)
            ]
            
            for name, group1, group2, current_step in combined_rdfs:
                print(f'Calculating combined {name} RDF (step={current_step})...')
                try:
                    rdf = InterRDF(group1, group2, nbins=nbins, range=range, 
                                norm='rdf', verbose=False)
                    
                    if self._debug_frame_indices is not None:
                        rdf_results = self._calculate_rdf_manual(rdf, self._debug_frame_indices)
                        rdf.results = rdf_results
                    else:
                        if njobs == 1:
                            rdf.run(step=current_step)
                        else:
                            try:
                                rdf.run(step=current_step, njobs=njobs)
                            except Exception as e:
                                print(f"Multiprocessing failed for {name}: {e}")
                                rdf.run(step=current_step, njobs=1)
                    
                    self.rdfs[name] = rdf.results
                    del rdf
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error calculating {name} RDF: {e}")
                    self.rdfs[name] = None
        
        else:
            # Original behavior - combined RDFs only
            rdfs_to_calc = [
                ('ci-w', self.cations, self.waters, step),
                ('ai-w', self.anions, self.waters, step),
                ('ci-ai', self.cations, self.anions, step)
            ]
            
            for name, group1, group2, current_step in rdfs_to_calc:
                print(f'Calculating {name} RDF (step={current_step})...')
                
                try:
                    rdf = InterRDF(group1, group2, nbins=nbins, range=range, 
                                norm='rdf', verbose=False)
                    
                    if self._debug_frame_indices is not None:
                        rdf_results = self._calculate_rdf_manual(rdf, self._debug_frame_indices)
                        rdf.results = rdf_results
                    else:
                        if njobs == 1:
                            rdf.run(step=current_step)
                        else:
                            try:
                                rdf.run(step=current_step, njobs=njobs)
                            except Exception as e:
                                print(f"Multiprocessing failed for {name}: {e}")
                                print("Retrying with single thread...")
                                rdf.run(step=current_step, njobs=1)
                    
                    self.rdfs[name] = rdf.results
                    del rdf
                    if name != 'ci-ai':
                        gc.collect()
                        
                except Exception as e:
                    print(f"Error calculating {name} RDF: {e}")
                    self.rdfs[name] = None
        
        # Water-water RDF (always calculated the same way)
        print(f'\nCalculating water-water RDF (step={water_step})...')
        try:
            w_w = InterRDF(self.waters, self.waters, nbins=nbins, range=range, norm='rdf', verbose=False)
            
            if self._debug_frame_indices is not None:
                # Use a subset of frames for water-water in debug mode
                debug_water_indices = self._debug_frame_indices[::max(1, len(self._debug_frame_indices)//50)]
                rdf_results = self._calculate_rdf_manual(w_w, debug_water_indices)
                w_w.results = rdf_results
            else:
                if njobs == 1:
                    w_w.run(step=water_step)
                else:
                    try:
                        w_w.run(step=water_step, njobs=njobs)
                    except Exception as e:
                        print(f"Multiprocessing failed for water-water: {e}")
                        print("Retrying with single thread...")
                        w_w.run(step=water_step, njobs=1)
            
            self.rdfs['w-w'] = w_w.results
            del w_w
        except Exception as e:
            print(f"Error calculating water-water RDF: {e}")
            self.rdfs['w-w'] = None
        
        gc.collect()
        
        print('  Memory usage after:', end=' ')
        self.get_memory_usage()
        
        # Print summary of calculated RDFs
        successful_rdfs = [key for key, value in self.rdfs.items() if value is not None]
        failed_rdfs = [key for key, value in self.rdfs.items() if value is None]
        
        print(f"\nRDF Calculation Summary:")
        print(f"  Successful RDFs ({len(successful_rdfs)}): {', '.join(successful_rdfs)}")
        if failed_rdfs:
            print(f"  Failed RDFs ({len(failed_rdfs)}): {', '.join(failed_rdfs)}")
        
        # Save to file if requested
        if filename is not None:
            self._save_rdfs_optimized(filename)
        
        return self.rdfs

    def save_rdfs_to_file(self, filename='rdfs_cache.pkl'):
        '''
        Save calculated RDFs to file for persistence across sessions.
        
        Parameters
        ----------
        filename : str
            Output filename, default='rdfs_cache.pkl'
        
        Returns
        -------
        success : bool
            True if save was successful
        '''
        
        if not hasattr(self, 'rdfs') or not self.rdfs:
            print("No RDFs to save")
            return False
        
        try:
            # Prepare RDF data for serialization
            rdf_data = {}
            
            for rdf_name, rdf_results in self.rdfs.items():
                if rdf_results is not None:
                    # Store the essential data from Results object
                    rdf_data[rdf_name] = {
                        'bins': rdf_results.bins.copy(),
                        'rdf': rdf_results.rdf.copy(),
                        'count': rdf_results.count.copy() if hasattr(rdf_results, 'count') else None,
                        'metadata': {
                            'n_frames': len(rdf_results.rdf) if hasattr(rdf_results, 'rdf') else 0
                        }
                    }
                else:
                    rdf_data[rdf_name] = None
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(rdf_data, f)
            
            print(f"RDFs saved to {filename}")
            print(f"  Saved {len([v for v in rdf_data.values() if v is not None])} RDFs")
            print(f"  RDF types: {list(rdf_data.keys())}")
            
            return True
            
        except Exception as e:
            print(f"Error saving RDFs: {e}")
            return False

    def load_rdfs_from_file(self, filename='rdfs_cache.pkl'):
        '''
        Load RDFs from file with error handling.
        
        Parameters
        ----------
        filename : str
            Input filename, default='rdfs_cache.pkl'
        
        Returns
        -------
        success : bool
            True if load was successful
        '''
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        try:
            # Check file size first
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print(f"Cache file {filename} is empty")
                return False
            
            # Load data
            with open(filename, 'rb') as f:
                rdf_data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(rdf_data, dict):
                print(f"Invalid RDF cache format")
                return False
            
            # Reconstruct RDF Results objects
            from MDAnalysis.analysis.base import Results
            
            self.rdfs = {}
            
            for rdf_name, data in rdf_data.items():
                if data is not None and isinstance(data, dict):
                    # Create Results object
                    results = Results()
                    results.bins = data['bins']
                    results.rdf = data['rdf']
                    if data.get('count') is not None:
                        results.count = data['count']
                    
                    self.rdfs[rdf_name] = results
                else:
                    self.rdfs[rdf_name] = None
            
            # Print summary
            successful_rdfs = [k for k, v in self.rdfs.items() if v is not None]
            failed_rdfs = [k for k, v in self.rdfs.items() if v is None]
            
            print(f"RDFs loaded from {filename}")
            print(f"  Loaded {len(successful_rdfs)} RDFs successfully")
            if successful_rdfs:
                print(f"  Available RDFs: {', '.join(successful_rdfs)}")
            if failed_rdfs:
                print(f"  Failed to load: {', '.join(failed_rdfs)}")
            
            return True
            
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            print("Removing corrupted cache file...")
            try:
                os.remove(filename)
                print(f"Corrupted file {filename} removed")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading RDFs from {filename}: {e}")
            return False

    def generate_rdfs_with_cache(self, cache_filename='rdfs_cache.pkl', 
                                force_recalc=False, **kwargs):
        '''
        Generate RDFs with automatic caching.
        
        Parameters
        ----------
        cache_filename : str
            Cache file name, default='rdfs_cache.pkl'
        force_recalc : bool
            Force recalculation even if cache exists, default=False
        **kwargs : dict
            Arguments to pass to generate_rdfs()
        
        Returns
        -------
        rdfs : dict
            Dictionary of RDF results
        '''
        
        # Try to load from cache first
        if not force_recalc and os.path.exists(cache_filename):
            print("Attempting to load RDFs from cache...")
            if self.load_rdfs_from_file(cache_filename):
                print("✓ Successfully loaded RDFs from cache")
                return self.rdfs
            else:
                print("✗ Cache loading failed, will recalculate")
        
        # Calculate RDFs
        print("Calculating RDFs...")
        rdfs = self.generate_rdfs(**kwargs)
        
        # Save to cache
        if rdfs:
            print("Saving RDFs to cache...")
            if self.save_rdfs_to_file(cache_filename):
                print("✓ RDFs cached successfully")
            else:
                print("✗ Cache saving failed, but RDFs are available in memory")
        
        return rdfs


    def calculate_running_coordination_number(self, rdf_name='Na-w', save_data=True):
        '''
        Calculate the running coordination number (RCN) from an RDF.
        RCN(r) = 4π * ρ * integral[0 to r] of g(r') * r'^2 dr'
        
        Parameters
        ----------
        rdf_name : str
            Name of the RDF to use (e.g., 'Na-w', 'Cl-w', 'ci-w')
        save_data : bool
            Whether to save the RCN data, default=True
        
        Returns
        -------
        r : np.array
            Radial distances
        rcn : np.array
            Running coordination numbers at each r
        '''
        
        # Check if we have the RDF
        if not hasattr(self, 'rdfs') or rdf_name not in self.rdfs:
            print(f"RDF '{rdf_name}' not found. Available RDFs: {list(self.rdfs.keys()) if hasattr(self, 'rdfs') else 'None'}")
            return None, None
        
        rdf_data = self.rdfs[rdf_name]
        if rdf_data is None:
            print(f"RDF '{rdf_name}' is None")
            return None, None
        
        # Get RDF data
        r = rdf_data.bins
        g_r = rdf_data.rdf
        
        # Calculate number density
        # Density = number of particles / box volume
        box_volume = np.prod(self.universe.dimensions[:3])  # Å³
        
        # Determine which particles to count for density
        if rdf_name.endswith('-w'):
            # Ion-water RDF
            n_particles = len(self.waters)
        elif rdf_name == 'ci-ai':
            # Cation-anion RDF
            ion_type = rdf_name.split('-')[0]
            if ion_type == 'ci':
                n_particles = len(self.anions)
            else:
                n_particles = len(self.cations)
        else:
            # Try to determine from RDF name
            parts = rdf_name.split('-')
            if len(parts) == 2:
                if parts[1] == 'w':
                    n_particles = len(self.waters)
                else:
                    # Cross ion-ion RDF - need to determine which is the second ion
                    # This is tricky - for now use a reasonable guess
                    n_particles = max(len(self.cations), len(self.anions))
            else:
                print(f"Cannot determine particle count for RDF '{rdf_name}'")
                return None, None
        
        # Number density in particles/Å³
        rho = n_particles / box_volume
        
        print(f"Calculating running coordination number for {rdf_name}")
        print(f"  Box volume: {box_volume:.1f} Å³")
        print(f"  Number of particles: {n_particles}")
        print(f"  Number density: {rho:.6f} particles/Å³")
        
        # Calculate RCN using trapezoidal integration
        # RCN(r) = 4π * ρ * integral[g(r') * r'^2 dr'] from 0 to r
        
        integrand = g_r * r**2
        
        # Cumulative integration using numpy's cumulative trapezoid
        # This gives us the running integral at each point
        rcn = 4 * np.pi * rho * np.cumsum(integrand) * (r[1] - r[0])  # Assuming uniform spacing
        
        # Store results
        if not hasattr(self, 'running_coordination_numbers'):
            self.running_coordination_numbers = {}
        
        self.running_coordination_numbers[rdf_name] = {
            'r': r.copy(),
            'rcn': rcn.copy(),
            'rdf': g_r.copy(),
            'density': rho
        }
        
        # Print key values
        print(f"\nRunning Coordination Number Results:")
        # Find first minimum in RDF (typical first shell boundary)
        if hasattr(self, 'solutes_ci') or hasattr(self, 'solutes_ai'):
            # Try to get coordination radius
            if rdf_name.endswith('-w'):
                ion_type_str = rdf_name.split('-')[0]
                r0 = None
                
                if hasattr(self, 'solutes_ci') and ion_type_str in self.solutes_ci:
                    r0 = self.solutes_ci[ion_type_str].radii.get('water')
                elif hasattr(self, 'solutes_ai') and ion_type_str in self.solutes_ai:
                    r0 = self.solutes_ai[ion_type_str].radii.get('water')
                elif ion_type_str == 'ci' and hasattr(self, 'solute_ci'):
                    r0 = self.solute_ci.radii.get('water')
                elif ion_type_str == 'ai' and hasattr(self, 'solute_ai'):
                    r0 = self.solute_ai.radii.get('water')
                
                if r0 is not None:
                    idx = np.argmin(np.abs(r - r0))
                    print(f"  RCN at r₀={r0:.2f} Å: {rcn[idx]:.2f}")
        
        # Print RCN at some key distances
        for distance in [2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
            if distance <= r.max():
                idx = np.argmin(np.abs(r - distance))
                print(f"  RCN at r={distance:.1f} Å: {rcn[idx]:.2f}")
        
        if save_data:
            filename = f'rcn_{rdf_name}.txt'
            data_array = np.column_stack([r, rcn, g_r])
            np.savetxt(filename, data_array, 
                    header=f'r (Å)  RCN(r)  g(r) for {rdf_name}\nDensity: {rho:.6f} particles/Å³',
                    fmt='%.6f')
            print(f"Data saved to: {filename}")
        
        return r, rcn

 
    def plot_running_coordination_number(self, rdf_name='Na-w', save_plot=True, plot_range=12):
        '''
        Plot RDF and RCN on the same plot with dual y-axes.
        RCN y-axis is auto-scaled so the curve around r₀ takes up ~25% of figure height.
        FIXED: Proper scaling to ensure CN at r₀ appears at 25% of figure height.
        
        Parameters
        ----------
        rdf_name : str
            Name of the RDF to plot
        save_plot : bool
            Whether to save the plot
        plot_range : float
            Maximum r value for plotting
        '''
        
        # Calculate RCN if not already done
        if not hasattr(self, 'running_coordination_numbers') or rdf_name not in self.running_coordination_numbers:
            r, rcn = self.calculate_running_coordination_number(rdf_name)
            if r is None:
                return
        else:
            rcn_data = self.running_coordination_numbers[rdf_name]
            r = rcn_data['r']
            rcn = rcn_data['rcn']
        
        # Get RDF data
        if rdf_name in self.rdfs and self.rdfs[rdf_name] is not None:
            rdf_data = self.rdfs[rdf_name]
            g_r = rdf_data.rdf
        else:
            g_r = self.running_coordination_numbers[rdf_name]['rdf']
        
        # Get coordination radius
        r0 = None
        if rdf_name.endswith('-w'):
            ion_type_str = rdf_name.split('-')[0]
            
            if hasattr(self, 'solutes_ci') and ion_type_str in self.solutes_ci:
                r0 = self.solutes_ci[ion_type_str].radii.get('water')
            elif hasattr(self, 'solutes_ai') and ion_type_str in self.solutes_ai:
                r0 = self.solutes_ai[ion_type_str].radii.get('water')
            elif ion_type_str == 'ci' and hasattr(self, 'solute_ci'):
                r0 = self.solute_ci.radii.get('water')
            elif ion_type_str == 'ai' and hasattr(self, 'solute_ai'):
                r0 = self.solute_ai.radii.get('water')
        
        # Create figure with single plot but dual y-axes
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot RDF on primary y-axis (left)
        color_rdf = 'blue'
        ax1.set_xlabel('r (Å)', fontsize=12)
        ax1.set_ylabel('g(r)', color=color_rdf, fontsize=12)
        line1 = ax1.plot(r, g_r, color=color_rdf, linewidth=2, label='g(r)', alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color_rdf)
        ax1.set_xlim(0, plot_range)
        
        # FIXED: Set RDF y-axis limits to start from 0
        rdf_max = g_r[r <= plot_range].max()
        ax1.set_ylim(0, rdf_max * 1.1)  # Start from 0, add 10% headroom
        
        # Mark coordination radius on RDF
        if r0 is not None:
            ax1.axvline(r0, color='red', linestyle='--', linewidth=2, alpha=0.5)
            idx = np.argmin(np.abs(r - r0))
            ax1.text(r0, ax1.get_ylim()[1]*0.9, f'r₀={r0:.2f} Å', 
                    ha='center', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Create secondary y-axis for RCN (right)
        ax2 = ax1.twinx()
        color_rcn = 'red'
        ax2.set_ylabel('Running Coordination Number', color=color_rcn, fontsize=12)
        line2 = ax2.plot(r, rcn, color=color_rcn, linewidth=2, label='RCN(r)', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color_rcn)
        
        # FIXED: SMART Y-AXIS SCALING - CN at r₀ should be at 25% of figure height
        if r0 is not None:
            idx_r0 = np.argmin(np.abs(r - r0))
            cn_at_r0 = rcn[idx_r0]
            
            # CRITICAL FIX: Set y_max so CN at r₀ appears at exactly 25% of figure height
            # If CN at r₀ should be at 25%, then y_max = CN at r₀ * 4
            y_max_rcn = cn_at_r0 * 4.0
            
            # CRITICAL FIX: Set both axes to start from 0 for alignment
            ax2.set_ylim(0, y_max_rcn)
            
            print(f"Auto-scaled RCN y-axis:")
            print(f"  CN at r₀={cn_at_r0:.2f}")
            print(f"  y_max chosen: {y_max_rcn:.2f}")
            print(f"  CN at r₀ appears at {(cn_at_r0/y_max_rcn)*100:.0f}% of figure height")
            
            # Mark CN value
            ax2.axhline(cn_at_r0, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
            ax2.text(plot_range*0.95, cn_at_r0, f'CN={cn_at_r0:.2f}', 
                    ha='right', va='center', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        else:
            # Fallback: just use data range with buffer if no r₀ available
            max_rcn_in_range = rcn[r <= plot_range].max()
            ax2.set_ylim(0, max_rcn_in_range * 1.1)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=11)
        
        # Title
        ax1.set_title(f'{rdf_name.upper()} RDF and Running Coordination Number', 
                    fontweight='bold', fontsize=14)
        
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'rcn_{rdf_name}_dual_axis_scaled.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {filename}")
        
        plt.show()



    def plot_all_running_coordination_numbers(self, ion_types=None, save_plot=True, plot_range=12):
        '''
        Plot running coordination numbers for multiple ion types in a comparison plot.
        
        Parameters
        ----------
        ion_types : list or None
            List of ion types to plot. If None, plots all available ion-water RDFs
        save_plot : bool
            Whether to save the plot
        plot_range : float
            Maximum r value for plotting
        '''
        
        # Determine which RDFs to plot
        if ion_types is None:
            # Find all ion-water RDFs
            rdf_names = [name for name in self.rdfs.keys() 
                        if name.endswith('-w') and name not in ['ci-w', 'ai-w', 'w-w']]
        else:
            rdf_names = [f'{ion_type}-w' for ion_type in ion_types]
        
        if not rdf_names:
            print("No ion-water RDFs found to plot")
            return
        
        # Separate cations and anions
        cation_types_in_system = set(self._get_unique_ion_types(self.cations).keys())
        
        cation_rdfs = []
        anion_rdfs = []
        
        for rdf_name in rdf_names:
            ion_type = rdf_name.split('-')[0]
            if ion_type in cation_types_in_system:
                cation_rdfs.append(rdf_name)
            else:
                anion_rdfs.append(rdf_name)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot cations
        if cation_rdfs:
            colors_c = plt.cm.Blues(np.linspace(0.4, 0.9, len(cation_rdfs)))
            
            for i, rdf_name in enumerate(cation_rdfs):
                # Calculate RCN if needed
                if not hasattr(self, 'running_coordination_numbers') or rdf_name not in self.running_coordination_numbers:
                    r, rcn = self.calculate_running_coordination_number(rdf_name, save_data=False)
                else:
                    rcn_data = self.running_coordination_numbers[rdf_name]
                    r = rcn_data['r']
                    rcn = rcn_data['rcn']
                
                if r is not None:
                    ion_type = rdf_name.split('-')[0]
                    ax1.plot(r, rcn, linewidth=2, color=colors_c[i], label=ion_type.upper())
            
            ax1.set_xlabel('r (Å)', fontsize=12)
            ax1.set_ylabel('Running Coordination Number', fontsize=12)
            ax1.set_title('Cation Running Coordination Numbers', fontweight='bold', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, plot_range)
        
        # Plot anions
        if anion_rdfs:
            colors_a = plt.cm.Reds(np.linspace(0.4, 0.9, len(anion_rdfs)))
            
            for i, rdf_name in enumerate(anion_rdfs):
                # Calculate RCN if needed
                if not hasattr(self, 'running_coordination_numbers') or rdf_name not in self.running_coordination_numbers:
                    r, rcn = self.calculate_running_coordination_number(rdf_name, save_data=False)
                else:
                    rcn_data = self.running_coordination_numbers[rdf_name]
                    r = rcn_data['r']
                    rcn = rcn_data['rcn']
                
                if r is not None:
                    ion_type = rdf_name.split('-')[0]
                    ax2.plot(r, rcn, linewidth=2, color=colors_a[i], label=ion_type.upper())
            
            ax2.set_xlabel('r (Å)', fontsize=12)
            ax2.set_ylabel('Running Coordination Number', fontsize=12)
            ax2.set_title('Anion Running Coordination Numbers', fontweight='bold', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, plot_range)
        
        plt.tight_layout()
        
        if save_plot:
            filename = 'all_running_coordination_numbers.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved: {filename}")
        
        plt.show()



    def save_solutes_to_file(self, filename='solutes_cache.pkl'):
        '''Save initialized solutes to file for persistence across sessions'''
        
        import pickle
        
        if hasattr(self, 'solutes_ci') and hasattr(self, 'solutes_ai'):
            # Convert solutes to a pickleable format
            solute_data = {
                'cations': {},
                'anions': {},
                'combined_ci': None,
                'combined_ai': None
            }
            
            # Process cation solutes
            for ion_type, solute in self.solutes_ci.items():
                if solute is not None:
                    if hasattr(solute, 'radii') and 'water' in solute.radii:
                        # Store just the essential data
                        solute_data['cations'][ion_type] = {
                            'name': ion_type,
                            'radius': solute.radii['water'],
                            'type': 'mock' if hasattr(solute, 'name') else 'full'
                        }
                    else:
                        solute_data['cations'][ion_type] = None
                else:
                    solute_data['cations'][ion_type] = None
            
            # Process anion solutes
            for ion_type, solute in self.solutes_ai.items():
                if solute is not None:
                    if hasattr(solute, 'radii') and 'water' in solute.radii:
                        # Store just the essential data
                        solute_data['anions'][ion_type] = {
                            'name': ion_type,
                            'radius': solute.radii['water'],
                            'type': 'mock' if hasattr(solute, 'name') else 'full'
                        }
                    else:
                        solute_data['anions'][ion_type] = None
                else:
                    solute_data['anions'][ion_type] = None
            
            # Store combined solute info
            if hasattr(self, 'solute_ci') and self.solute_ci is not None:
                if hasattr(self.solute_ci, 'radii') and 'water' in self.solute_ci.radii:
                    solute_data['combined_ci'] = {
                        'radius': self.solute_ci.radii['water'],
                        'source': getattr(self.solute_ci, 'name', 'unknown')
                    }
            
            if hasattr(self, 'solute_ai') and self.solute_ai is not None:
                if hasattr(self.solute_ai, 'radii') and 'water' in self.solute_ai.radii:
                    solute_data['combined_ai'] = {
                        'radius': self.solute_ai.radii['water'],
                        'source': getattr(self.solute_ai, 'name', 'unknown')
                    }
            
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(solute_data, f)
                
                print(f"Solutes saved to {filename}")
                print(f"  Saved {len([v for v in solute_data['cations'].values() if v is not None])} cation types")
                print(f"  Saved {len([v for v in solute_data['anions'].values() if v is not None])} anion types")
                
            except Exception as e:
                print(f"Error saving solutes: {e}")
                return False
                
            return True
        else:
            print("No solutes to save")
            return False

    def load_solutes_from_file(self, filename='solutes_cache.pkl'):
        '''Load initialized solutes from file with better error handling'''
        
        import pickle
        import os
        
        if os.path.exists(filename):
            try:
                # Check file size first
                file_size = os.path.getsize(filename)
                if file_size == 0:
                    print(f"Cache file {filename} is empty, skipping...")
                    return False
                
                with open(filename, 'rb') as f:
                    solute_data = pickle.load(f)
                
                # Validate the loaded data structure
                if not isinstance(solute_data, dict):
                    print(f"Invalid cache file format in {filename}")
                    return False
                
                if 'cations' not in solute_data or 'anions' not in solute_data:
                    print(f"Incomplete cache file format in {filename}")
                    return False
                
                # Reconstruct solute objects
                self.solutes_ci = {}
                self.solutes_ai = {}
                
                # Reconstruct cation solutes
                for ion_type, data in solute_data.get('cations', {}).items():
                    if data is not None and isinstance(data, dict) and 'radius' in data:
                        self.solutes_ci[ion_type] = MockSolute(data['name'], data['radius'])
                    else:
                        self.solutes_ci[ion_type] = None
                
                # Reconstruct anion solutes
                for ion_type, data in solute_data.get('anions', {}).items():
                    if data is not None and isinstance(data, dict) and 'radius' in data:
                        self.solutes_ai[ion_type] = MockSolute(data['name'], data['radius'])
                    else:
                        self.solutes_ai[ion_type] = None
                
                # Reconstruct combined solutes
                combined_ci_data = solute_data.get('combined_ci')
                if combined_ci_data is not None and isinstance(combined_ci_data, dict):
                    self.solute_ci = MockSolute(combined_ci_data.get('source', 'combined_cation'), 
                                            combined_ci_data['radius'])
                else:
                    self.solute_ci = None
                
                combined_ai_data = solute_data.get('combined_ai')
                if combined_ai_data is not None and isinstance(combined_ai_data, dict):
                    self.solute_ai = MockSolute(combined_ai_data.get('source', 'combined_anion'), 
                                            combined_ai_data['radius'])
                else:
                    self.solute_ai = None
                
                print(f"Solutes loaded from {filename}")
                print(f"  Loaded {len([v for v in self.solutes_ci.values() if v is not None])} cation types")
                print(f"  Loaded {len([v for v in self.solutes_ai.values() if v is not None])} anion types")
                
                if hasattr(self, 'print_coordination_radii_summary'):
                    self.print_coordination_radii_summary()
                return True
                
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Cache file {filename} is corrupted: {e}")
                print("Removing corrupted cache file...")
                try:
                    os.remove(filename)
                    print(f"Corrupted file {filename} removed")
                except:
                    pass
                return False
            except Exception as e:
                print(f"Error loading solutes from {filename}: {e}")
                return False
        else:
            print(f"File {filename} not found")
            return False

    def safe_initialization_workflow(self, use_cache=True, cache_filename='solutes_cache.pkl'):
        '''
        Safe workflow for solute initialization with automatic fallback and cleanup.
        '''
        
        print("=== SAFE SOLUTE INITIALIZATION ===")
        
        # Try to load from cache first
        if use_cache:
            print("1. Attempting to load from cache...")
            if self.load_solutes_from_file(cache_filename):
                print("✓ Successfully loaded from cache")
                return self.solutes_ci, self.solutes_ai
            else:
                print("✗ Cache loading failed or not available")
        
        # Fallback to quick initialization
        print("2. Using quick initialization with defaults...")
        try:
            self.quick_initialize_solutes_with_defaults()
            print("✓ Quick initialization successful")
            
            # Try to save to cache
            print("3. Saving to cache for future use...")
            if self.save_solutes_to_file(cache_filename):
                print("✓ Cache saved successfully")
            else:
                print("✗ Cache saving failed, but continuing...")
                
            return self.solutes_ci, self.solutes_ai
            
        except Exception as e:
            print(f"✗ Quick initialization failed: {e}")
            print("4. Attempting full initialization as last resort...")
            
            try:
                # Last resort - full initialization with minimal parameters
                self.initialize_Solutes_by_type(step=5, force_recalc=True)
                print("✓ Full initialization successful")
                
                # Save to cache
                if self.save_solutes_to_file(cache_filename):
                    print("✓ Cache saved successfully")
                    
                return self.solutes_ci, self.solutes_ai
                
            except Exception as e2:
                print(f"✗ All initialization methods failed: {e2}")
                return None, None

    def clear_cache_files(self, cache_dir='.'):
        '''Clear all cache files in the specified directory'''
        
        import glob
        import os
        
        cache_patterns = ['solutes_cache.*', '*_solvation_shells*.png', '*_rdfs*.png']
        
        removed_files = []
        
        for pattern in cache_patterns:
            files = glob.glob(os.path.join(cache_dir, pattern))
            for file in files:
                try:
                    os.remove(file)
                    removed_files.append(file)
                except Exception as e:
                    print(f"Could not remove {file}: {e}")
        
        if removed_files:
            print(f"Removed cache files: {removed_files}")
        else:
            print("No cache files found to remove")
        
        return removed_files

    def validate_system_integrity(self):
        '''Validate that all system components are properly initialized'''
        
        print("\n=== SYSTEM INTEGRITY CHECK ===")
        
        checks = {
            'Universe': hasattr(self, 'universe') and self.universe is not None,
            'Waters': hasattr(self, 'waters') and len(self.waters) > 0,
            'Cations': hasattr(self, 'cations') and len(self.cations) > 0,
            'Anions': hasattr(self, 'anions') and len(self.anions) > 0,
            'Trajectory': hasattr(self, 'n_frames') and self.n_frames > 0,
            'Cation Solutes': hasattr(self, 'solutes_ci') and len(self.solutes_ci) > 0,
            'Anion Solutes': hasattr(self, 'solutes_ai') and len(self.solutes_ai) > 0,
            'RDFs': hasattr(self, 'rdfs') and len(self.rdfs) > 0,
        }
        
        all_good = True
        for component, status in checks.items():
            status_symbol = "✓" if status else "✗"
            print(f"{status_symbol} {component}")
            if not status:
                all_good = False
        
        if all_good:
            print("✓ All systems operational!")
        else:
            print("✗ Some components need attention")
        
        return all_good

    def _calculate_rdf_manual(self, rdf_analyzer, frame_indices):
        '''
        Manually calculate RDF for debug trajectories.
        '''
        
        print(f"    Using manual RDF calculation for {len(frame_indices)} frames...")
        
        # Initialize the RDF analyzer without running
        rdf_analyzer._prepare()
        
        # Initialize result storage for accumulation
        if not hasattr(rdf_analyzer, '_result'):
            rdf_analyzer._result = []
        
        # Manually iterate through selected frames
        for i, frame_idx in enumerate(tqdm(frame_indices, desc="Manual RDF", leave=False)):
            # Set the trajectory to the correct frame
            ts = self.universe.trajectory[frame_idx]
            
            # Set frame information for the analyzer
            rdf_analyzer._frame_index = i
            rdf_analyzer._ts = ts
            
            try:
                # Try the newer version signature first (with frame_idx parameter)
                result = rdf_analyzer._single_frame(i)
                if hasattr(rdf_analyzer, '_result'):
                    rdf_analyzer._result.append(result)
                    
            except TypeError:
                try:
                    # Try the older version signature (no parameters)
                    result = rdf_analyzer._single_frame()
                    if hasattr(rdf_analyzer, '_result'):
                        rdf_analyzer._result.append(result)
                        
                except Exception as e:
                    print(f"    Error in frame {frame_idx}: {e}")
                    continue
        
        # Set required attributes for _conclude()
        rdf_analyzer.n_frames = len(frame_indices)
        
        # Finalize the calculation
        rdf_analyzer._conclude()
        
        return rdf_analyzer.results


    def generate_rdfs(self, bin_width=0.05, range=(0,20), step=None, filename=None, njobs=None, water_step=None, separate_ion_types=True, single_ion=None, partner_type='water', custom_params=None):
        '''
        Optimized RDF calculation with better memory management and parallel processing.
        Now supports separate RDFs for different ion types AND single ion RDF calculation.
        
        Parameters
        ----------
        separate_ion_types : bool
            If True, calculate separate RDFs for different cation/anion types, default=True
        single_ion : str, optional
            Calculate RDF for only this specific ion type (e.g., 'Na', 'Mg', 'Cl').
            If specified, only this ion's RDF will be calculated.
        partner_type : str
            Partner species for single ion RDF calculation:
            - 'water' or 'w' for ion-water RDF (default)
            - Specific ion type (e.g., 'Cl') for ion-ion RDF
            - 'anions' for cation-anion RDF
            - 'cations' for anion-cation RDF
        custom_params : dict, optional
            Custom parameters for single ion RDF calculation:
            - 'bin_width': float, bin width override
            - 'range': tuple, range override
            - 'step': int, step override
            - 'njobs': int, njobs override
        '''
        
        # Handle custom parameters for single ion calculation
        if single_ion is not None and custom_params is not None:
            bin_width = custom_params.get('bin_width', bin_width)
            range = custom_params.get('range', range)
            step = custom_params.get('step', step)
            njobs = custom_params.get('njobs', njobs)
            print(f"Using custom parameters for {single_ion}: bin_width={bin_width}, range={range}, step={step}, njobs={njobs}")
        
        # Use auto-tuned defaults if not specified
        if step is None:
            step = self.default_step
        if njobs is None:
            njobs = self.default_njobs
        if water_step is None:
            water_step = self.water_step
        
        # Use all available CPUs if -1, but cap to avoid signal issues
        if njobs == -1:
            njobs = min(multiprocessing.cpu_count(), 8)

        # Test if multiprocessing works properly
        njobs = self._test_multiprocessing_compatibility(njobs)    

        # Force single-threaded for debug mode to avoid complications
        if hasattr(self, '_debug_frame_indices') and self._debug_frame_indices is not None:
            njobs = 1
            print("Debug mode detected: Using single-threaded calculation")
        
        nbins = int((range[1] - range[0]) / bin_width)
        self.rdfs = {} if not hasattr(self, 'rdfs') else self.rdfs
        
        # SINGLE ION RDF CALCULATION
        if single_ion is not None:
            print(f'\nCalculating single ion RDF for {single_ion}:')
            print(f'  Partner: {partner_type}')
            print(f'  Parameters: step={step}, njobs={njobs}, range={range}, bin_width={bin_width}')
            
            # Get unique ion types
            cation_types = self._get_unique_ion_types(self.cations)
            anion_types = self._get_unique_ion_types(self.anions)
            
            # Find the requested ion type
            if single_ion in cation_types:
                ion_group = cation_types[single_ion]
                ion_category = 'cation'
            elif single_ion in anion_types:
                ion_group = anion_types[single_ion]
                ion_category = 'anion'
            else:
                available_types = list(cation_types.keys()) + list(anion_types.keys())
                print(f"Error: Ion type '{single_ion}' not found. Available types: {available_types}")
                return self.rdfs
            
            # Determine partner species
            if partner_type in ['water', 'w']:
                partner_group = self.waters
                rdf_name = f'{single_ion}-w'
            elif partner_type in ['anions', 'anion']:
                partner_group = self.anions
                rdf_name = f'{single_ion}-anions'
            elif partner_type in ['cations', 'cation']:
                partner_group = self.cations
                rdf_name = f'{single_ion}-cations'
            elif partner_type in cation_types:
                partner_group = cation_types[partner_type]
                rdf_name = f'{single_ion}-{partner_type}'
            elif partner_type in anion_types:
                partner_group = anion_types[partner_type]
                rdf_name = f'{single_ion}-{partner_type}'
            else:
                available_partners = ['water', 'anions', 'cations'] + list(cation_types.keys()) + list(anion_types.keys())
                print(f"Error: Partner type '{partner_type}' not found. Available: {available_partners}")
                return self.rdfs
            
            print(f'Calculating {rdf_name} RDF ({len(ion_group)} ions, {len(partner_group)} partners)...')
            
            try:
                rdf = InterRDF(ion_group, partner_group, nbins=nbins, range=range, 
                            norm='rdf', verbose=False)
                
                # Handle debug trajectory
                if self._debug_frame_indices is not None:
                    # Manual frame iteration for debug mode
                    rdf_results = self._calculate_rdf_manual(rdf, self._debug_frame_indices)
                    rdf.results = rdf_results
                else:
                    # Normal calculation
                    if njobs == 1:
                        rdf.run(step=step)
                    else:
                        try:
                            rdf.run(step=step, njobs=njobs)
                        except Exception as e:
                            print(f"Multiprocessing failed for {rdf_name}: {e}")
                            print("Retrying with single thread...")
                            rdf.run(step=step, njobs=1)
                
                self.rdfs[rdf_name] = rdf.results
                del rdf
                gc.collect()
                
                print(f'✓ Single ion RDF calculation complete: {rdf_name}')
                print(f'  r range: {self.rdfs[rdf_name].bins.min():.2f} - {self.rdfs[rdf_name].bins.max():.2f} Å')
                print(f'  Max g(r): {self.rdfs[rdf_name].rdf.max():.3f}')
                
                # Save to file if requested
                if filename is not None:
                    data = np.column_stack([self.rdfs[rdf_name].bins, self.rdfs[rdf_name].rdf])
                    np.savetxt(filename, data, header=f'r (Angstroms), {rdf_name} g(r)')
                    print(f"Single ion RDF saved to: {filename}")
                
                return self.rdfs
                
            except Exception as e:
                print(f"Error calculating single ion RDF {rdf_name}: {e}")
                return self.rdfs
        
        # STANDARD MULTI-ION RDF CALCULATION (existing code)
        print(f'\nCalculating RDFs with optimized parameters:')
        print(f'  Step: {step}, CPUs: {njobs}')
        print(f'  Separate ion types: {separate_ion_types}')
        print('  Memory usage before:', end=' ')
        self.get_memory_usage()
        
        if separate_ion_types:
            # Get unique ion types
            cation_types = self._get_unique_ion_types(self.cations)
            anion_types = self._get_unique_ion_types(self.anions)
            
            print(f"\nFound ion types:")
            print(f"  Cations: {list(cation_types.keys())}")
            print(f"  Anions: {list(anion_types.keys())}")
            
            # Calculate RDFs for each cation type with water
            for cation_name, cation_group in cation_types.items():
                rdf_name = f'{cation_name}-w'
                print(f'Calculating {rdf_name} RDF ({len(cation_group)} ions, step={step})...')
                
                try:
                    rdf = InterRDF(cation_group, self.waters, nbins=nbins, range=range, 
                                norm='rdf', verbose=False)
                    
                    # Handle debug trajectory
                    if self._debug_frame_indices is not None:
                        # Manual frame iteration for debug mode
                        rdf_results = self._calculate_rdf_manual(rdf, self._debug_frame_indices)
                        rdf.results = rdf_results
                    else:
                        # Normal calculation
                        if njobs == 1:
                            rdf.run(step=step)
                        else:
                            try:
                                rdf.run(step=step, njobs=njobs)
                            except Exception as e:
                                print(f"Multiprocessing failed for {rdf_name}: {e}")
                                print("Retrying with single thread...")
                                rdf.run(step=step, njobs=1)
                    
                    self.rdfs[rdf_name] = rdf.results
                    del rdf
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error calculating {rdf_name} RDF: {e}")
                    self.rdfs[rdf_name] = None
            
            # Calculate RDFs for each anion type with water
            for anion_name, anion_group in anion_types.items():
                rdf_name = f'{anion_name}-w'
                print(f'Calculating {anion_name} RDF ({len(anion_group)} ions, step={step})...')
                
                try:
                    rdf = InterRDF(anion_group, self.waters, nbins=nbins, range=range, 
                                norm='rdf', verbose=False)
                    
                    # Handle debug trajectory
                    if self._debug_frame_indices is not None:
                        # Manual frame iteration for debug mode
                        rdf_results = self._calculate_rdf_manual(rdf, self._debug_frame_indices)
                        rdf.results = rdf_results
                    else:
                        # Normal calculation
                        if njobs == 1:
                            rdf.run(step=step)
                        else:
                            try:
                                rdf.run(step=step, njobs=njobs)
                            except Exception as e:
                                print(f"Multiprocessing failed for {rdf_name}: {e}")
                                print("Retrying with single thread...")
                                rdf.run(step=step, njobs=1)
                    
                    self.rdfs[rdf_name] = rdf.results
                    del rdf
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error calculating {rdf_name} RDF: {e}")
                    self.rdfs[rdf_name] = None
            
            # Calculate cross-cation-anion RDFs
            for cation_name, cation_group in cation_types.items():
                for anion_name, anion_group in anion_types.items():
                    rdf_name = f'{cation_name}-{anion_name}'
                    print(f'Calculating {rdf_name} RDF (step={step})...')
                    
                    try:
                        rdf = InterRDF(cation_group, anion_group, nbins=nbins, range=range, 
                                    norm='rdf', verbose=False)
                        
                        # Handle debug trajectory
                        if self._debug_frame_indices is not None:
                            rdf_results = self._calculate_rdf_manual(rdf, self._debug_frame_indices)
                            rdf.results = rdf_results
                        else:
                            if njobs == 1:
                                rdf.run(step=step)
                            else:
                                try:
                                    rdf.run(step=step, njobs=njobs)
                                except Exception as e:
                                    print(f"Multiprocessing failed for {rdf_name}: {e}")
                                    print("Retrying with single thread...")
                                    rdf.run(step=step, njobs=1)
                        
                        self.rdfs[rdf_name] = rdf.results
                        del rdf
                        gc.collect()
                        
                    except Exception as e:
                        print(f"Error calculating {rdf_name} RDF: {e}")
                        self.rdfs[rdf_name] = None
        
        else:
            # If separate_ion_types=False, skip all calculations
            print("separate_ion_types=False: No RDFs will be calculated")
            print("Use separate_ion_types=True to calculate ion-type-specific RDFs")
        
        gc.collect()
        
        print('  Memory usage after:', end=' ')
        self.get_memory_usage()
        
        # Print summary of calculated RDFs
        successful_rdfs = [key for key, value in self.rdfs.items() if value is not None]
        failed_rdfs = [key for key, value in self.rdfs.items() if value is None]
        
        print(f"\nRDF Calculation Summary:")
        print(f"  Successful RDFs ({len(successful_rdfs)}): {', '.join(successful_rdfs)}")
        if failed_rdfs:
            print(f"  Failed RDFs ({len(failed_rdfs)}): {', '.join(failed_rdfs)}")
        
        # Save to file if requested
        if filename is not None and single_ion is None:
            self._save_rdfs_optimized(filename)
        
        return self.rdfs

 

    def _get_unique_ion_types(self, ion_group):
        '''
        Get unique ion types from an ion group and return separate AtomGroups.
        FIXED: Now handles empty ion groups gracefully
        
        Parameters
        ----------
        ion_group : AtomGroup
            Group of ions (cations or anions)
        
        Returns
        -------
        ion_types : dict  
            Dictionary with ion type names as keys and AtomGroups as values
        '''
        
        # FIXED: Handle empty ion groups
        if len(ion_group) == 0:
            print(f"Warning: Empty ion group provided to _get_unique_ion_types")
            return {}
        
        # Get unique ion types based on element, name, or resname
        unique_types = {}
        
        # Try to group by element first (most reliable)
        if hasattr(ion_group[0], 'element'):
            for element in set(ion_group.elements):
                ions_of_type = ion_group.select_atoms(f'element {element}')
                unique_types[element] = ions_of_type
        
        # If no element info, try by name
        elif hasattr(ion_group[0], 'name'):
            for name in set(ion_group.names):
                ions_of_type = ion_group.select_atoms(f'name {name}')
                unique_types[name] = ions_of_type
        
        # If no name info, try by resname
        elif hasattr(ion_group[0], 'resname'):
            for resname in set(ion_group.resnames):
                ions_of_type = ion_group.select_atoms(f'resname {resname}')
                unique_types[resname] = ions_of_type
        
        # Fallback - treat all as same type
        else:
            unique_types['ion'] = ion_group
        
        return unique_types




    def plot_rdfs_by_type(self, ion_types=None, save_plots=True, plot_range=12):
        '''
        Plot RDFs separated by ion type.
        
        Parameters
        ----------
        ion_types : list
            List of ion types to plot. If None, plots all available types
        save_plots : bool
            Whether to save individual plots, default=True
        plot_range : float
            Maximum r value for plotting, default=12
        '''
        
        if not hasattr(self, 'rdfs') or not self.rdfs:
            print("No RDFs available. Run generate_rdfs() first.")
            return
        
        # Get the actual ion types from the system for proper classification
        cation_types_in_system = set(self._get_unique_ion_types(self.cations).keys())
        anion_types_in_system = set(self._get_unique_ion_types(self.anions).keys())
        
        print(f"DEBUG: Cation types in system: {cation_types_in_system}")
        print(f"DEBUG: Anion types in system: {anion_types_in_system}")
        print(f"DEBUG: Available RDFs: {list(self.rdfs.keys())}")
        
        # Properly separate cation-water and anion-water RDFs based on actual system composition
        cation_water_rdfs = {}
        anion_water_rdfs = {}
        
        for rdf_name, rdf_data in self.rdfs.items():
            if rdf_name.endswith('-w') and rdf_name not in ['ci-w', 'ai-w', 'w-w']:
                ion_type = rdf_name.split('-')[0]
                
                # Check if this ion type belongs to cations or anions in the actual system
                if ion_type in cation_types_in_system:
                    cation_water_rdfs[rdf_name] = rdf_data
                elif ion_type in anion_types_in_system:
                    anion_water_rdfs[rdf_name] = rdf_data
                else:
                    # If we can't determine, print a warning
                    print(f"WARNING: Cannot classify ion type '{ion_type}' as cation or anion")
        
        print(f"DEBUG: Classified cation-water RDFs: {list(cation_water_rdfs.keys())}")
        print(f"DEBUG: Classified anion-water RDFs: {list(anion_water_rdfs.keys())}")
        
        # Plot cation-water RDFs
        if cation_water_rdfs:
            n_cation_types = len(cation_water_rdfs)
            fig, axes = plt.subplots(1, n_cation_types, figsize=(5*n_cation_types, 4))
            if n_cation_types == 1:
                axes = [axes]
            
            for i, (rdf_name, rdf_data) in enumerate(cation_water_rdfs.items()):
                if rdf_data is not None:
                    ion_name = rdf_name.split('-')[0]
                    axes[i].plot(rdf_data.bins, rdf_data.rdf, 'b-', linewidth=2)
                    axes[i].set_title(f'{ion_name.upper()}-Water RDF', fontweight='bold')
                    axes[i].set_xlabel('r (Å)')
                    axes[i].set_ylabel('g(r)')
                    axes[i].set_xlim(0, plot_range)
                    # axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('cation_water_rdfs_by_type.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot anion-water RDFs
        if anion_water_rdfs:
            n_anion_types = len(anion_water_rdfs)
            fig, axes = plt.subplots(1, n_anion_types, figsize=(5*n_anion_types, 4))
            if n_anion_types == 1:
                axes = [axes]
            
            for i, (rdf_name, rdf_data) in enumerate(anion_water_rdfs.items()):
                if rdf_data is not None:
                    ion_name = rdf_name.split('-')[0]
                    axes[i].plot(rdf_data.bins, rdf_data.rdf, 'r-', linewidth=2)
                    axes[i].set_title(f'{ion_name.upper()}-Water RDF', fontweight='bold')
                    axes[i].set_xlabel('r (Å)')
                    axes[i].set_ylabel('g(r)')
                    axes[i].set_xlim(0, plot_range)
                    # axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('anion_water_rdfs_by_type.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot cation-anion cross RDFs
        cross_rdfs = {}
        for rdf_name, rdf_data in self.rdfs.items():
            if '-' in rdf_name and not rdf_name.endswith('-w') and rdf_name not in ['ci-w', 'ai-w', 'w-w', 'ci-ai']:
                cross_rdfs[rdf_name] = rdf_data
        
        if cross_rdfs:
            n_cross = len(cross_rdfs)
            cols = min(3, n_cross)
            rows = (n_cross + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if n_cross == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, (rdf_name, rdf_data) in enumerate(cross_rdfs.items()):
                if rdf_data is not None and i < len(axes):
                    cation_name, anion_name = rdf_name.split('-')
                    axes[i].plot(rdf_data.bins, rdf_data.rdf, 'g-', linewidth=2)
                    axes[i].set_title(f'{cation_name.upper()}-{anion_name.upper()} RDF', fontweight='bold')
                    axes[i].set_xlabel('r (Å)')
                    axes[i].set_ylabel('g(r)')
                    axes[i].set_xlim(0, plot_range)
                    # axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(cross_rdfs), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('cation_anion_rdfs_by_type.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Print summary
        print(f"\nRDF Summary by Ion Type:")
        print(f"  Cation-water RDFs: {len(cation_water_rdfs)}")
        print(f"  Anion-water RDFs: {len(anion_water_rdfs)}")  
        print(f"  Cation-anion cross RDFs: {len(cross_rdfs)}")


    def get_rdf_by_type(self, ion1_type, ion2_type=None):
        '''
        Get specific RDF by ion type names.
        
        Parameters
        ----------
        ion1_type : str
            First ion type (e.g., 'Na', 'K', 'Cl')
        ion2_type : str
            Second ion type. Use 'w' for water, or another ion type
        
        Returns
        -------
        rdf_data : Results or None
            RDF data if found, None otherwise
        '''
        
        if ion2_type is None:
            ion2_type = 'w'  # Default to water
        
        rdf_key = f'{ion1_type}-{ion2_type}'
        
        if rdf_key in self.rdfs:
            return self.rdfs[rdf_key]
        
        # Try reverse order for cation-anion
        reverse_key = f'{ion2_type}-{ion1_type}'
        if reverse_key in self.rdfs:
            return self.rdfs[reverse_key]
        
        print(f"RDF {rdf_key} not found. Available RDFs: {list(self.rdfs.keys())}")
        return None

    def _save_rdfs_optimized(self, filename):
        '''Optimized RDF saving using efficient formats'''
        
        if filename.endswith('.npz'):
            # Use numpy's compressed format
            data_dict = {}
            for key, rdf in self.rdfs.items():
                data_dict[f'{key}_bins'] = rdf.bins
                data_dict[f'{key}_rdf'] = rdf.rdf
            np.savez_compressed(filename, **data_dict)
            print(f"RDFs saved to compressed format: {filename}")
            
        elif filename.endswith('.h5') or filename.endswith('.hdf5'):
            # Use HDF5 for better performance with large datasets
            import h5py
            with h5py.File(filename, 'w') as f:
                for key, rdf in self.rdfs.items():
                    grp = f.create_group(key)
                    grp.create_dataset('bins', data=rdf.bins, compression='gzip')
                    grp.create_dataset('rdf', data=rdf.rdf, compression='gzip')
            print(f"RDFs saved to HDF5 format: {filename}")
            
        else:
            # Fallback to original format
            data = np.column_stack([
                self.rdfs['ci-w'].bins,
                self.rdfs['ci-w'].rdf,
                self.rdfs['ai-w'].rdf,
                self.rdfs['w-w'].rdf,
                self.rdfs['ci-ai'].rdf
            ])
            np.savetxt(filename, data, 
                      header='r (Angstroms), cation-water g(r), anion-water g(r), water-water g(r), cation-anion g(r)')
            print(f"RDFs saved to text format: {filename}")


    def extract_coordination_radii_from_rdfs(self, save_to_solutes=True):
        '''
        Extract coordination radii from ion-water RDFs by finding first minimum.
        
        Parameters
        ----------
        save_to_solutes : bool
            Whether to save results to solute objects for compatibility
        
        Returns
        -------
        radii_dict : dict
            Dictionary with coordination radii by ion type
        '''
        
        from scipy.signal import find_peaks
        
        if not hasattr(self, 'rdfs') or not self.rdfs:
            print("No RDFs available. Run generate_rdfs(separate_ion_types=True) first.")
            return None
        
        # Get unique ion types
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        coordination_radii = {'cations': {}, 'anions': {}}
        
        print("Extracting coordination radii from ion-water RDFs...")
        
        # Process cation RDFs
        for cation_name in cation_types.keys():
            rdf_key = f'{cation_name}-w'
            
            if rdf_key in self.rdfs and self.rdfs[rdf_key] is not None:
                rdf_data = self.rdfs[rdf_key]
                r = rdf_data.bins
                g_r = rdf_data.rdf
                
                # Find first minimum
                try:
                    # Find minima (peaks in negative RDF)
                    minima, _ = find_peaks(-g_r, height=-10, distance=10, prominence=0.01)
                    
                    if len(minima) > 0:
                        first_minimum = r[minima[0]]
                        coordination_radii['cations'][cation_name] = first_minimum
                        print(f"  {cation_name}: {first_minimum:.2f} Å (from RDF minimum)")
                    else:
                        # Fallback: use distance where g(r) first crosses 1.0
                        crossing = np.where(g_r > 1.0)[0]
                        if len(crossing) > 0:
                            first_crossing = r[crossing[0]]
                            coordination_radii['cations'][cation_name] = first_crossing
                            print(f"  {cation_name}: {first_crossing:.2f} Å (from g(r)=1 crossing)")
                        else:
                            # Default fallback
                            default_radius = 2.8
                            coordination_radii['cations'][cation_name] = default_radius
                            print(f"  {cation_name}: {default_radius:.2f} Å (default - no clear minimum)")
                            
                except Exception as e:
                    print(f"  Error processing {cation_name}: {e}")
                    coordination_radii['cations'][cation_name] = 2.8  # Default
            else:
                print(f"  Warning: No RDF data for {cation_name}")
        
        # Process anion RDFs
        for anion_name in anion_types.keys():
            rdf_key = f'{anion_name}-w'
            
            if rdf_key in self.rdfs and self.rdfs[rdf_key] is not None:
                rdf_data = self.rdfs[rdf_key]
                r = rdf_data.bins
                g_r = rdf_data.rdf
                
                # Find first minimum
                try:
                    # Find minima (peaks in negative RDF)
                    minima, _ = find_peaks(-g_r, height=-10, distance=10, prominence=0.01)
                    
                    if len(minima) > 0:
                        first_minimum = r[minima[0]]
                        coordination_radii['anions'][anion_name] = first_minimum
                        print(f"  {anion_name}: {first_minimum:.2f} Å (from RDF minimum)")
                    else:
                        # Fallback: use distance where g(r) first crosses 1.0
                        crossing = np.where(g_r > 1.0)[0]
                        if len(crossing) > 0:
                            first_crossing = r[crossing[0]]
                            coordination_radii['anions'][anion_name] = first_crossing
                            print(f"  {anion_name}: {first_crossing:.2f} Å (from g(r)=1 crossing)")
                        else:
                            # Default fallback
                            default_radius = 3.5
                            coordination_radii['anions'][anion_name] = default_radius
                            print(f"  {anion_name}: {default_radius:.2f} Å (default - no clear minimum)")
                            
                except Exception as e:
                    print(f"  Error processing {anion_name}: {e}")
                    coordination_radii['anions'][anion_name] = 3.5  # Default
            else:
                print(f"  Warning: No RDF data for {anion_name}")
        
        # Optionally save to solute objects for compatibility
        if save_to_solutes:
            self._create_solutes_from_radii(coordination_radii)
        
        return coordination_radii

    def _create_solutes_from_radii(self, radii_dict):
        '''Create mock solute objects from extracted radii for compatibility'''
        
        self.solutes_ci = {}
        self.solutes_ai = {}
        
        # Create cation solutes
        for ion_type, radius in radii_dict['cations'].items():
            mock_solute = MockSolute(ion_type, radius)
            self.solutes_ci[ion_type] = mock_solute
        
        # Create anion solutes  
        for ion_type, radius in radii_dict['anions'].items():
            mock_solute = MockSolute(ion_type, radius)
            self.solutes_ai[ion_type] = mock_solute
        
        print(f"Created mock solute objects for compatibility")

    def plot_rdf_minima_detection(self, ion_types=None, save_plots=True):
        '''
        Plot RDFs with detected minima for verification.
        '''
        
        from scipy.signal import find_peaks
        import matplotlib.pyplot as plt
        
        if ion_types is None:
            # Get all ion types with water RDFs
            cation_types = list(self._get_unique_ion_types(self.cations).keys())
            anion_types = list(self._get_unique_ion_types(self.anions).keys())
            ion_types = cation_types + anion_types
        
        # Calculate grid layout
        n_ions = len(ion_types)
        n_cols = min(3, n_ions)
        n_rows = (n_ions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_ions == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, ion_type in enumerate(ion_types):
            if i >= len(axes):
                break
                
            ax = axes[i]
            rdf_key = f'{ion_type}-w'
            
            if rdf_key in self.rdfs and self.rdfs[rdf_key] is not None:
                rdf_data = self.rdfs[rdf_key]
                r = rdf_data.bins
                g_r = rdf_data.rdf
                
                # Plot RDF
                ax.plot(r, g_r, 'b-', linewidth=2, label='g(r)')
                
                # Find and mark minima
                minima, _ = find_peaks(-g_r, height=-10, distance=10, prominence=0.01)
                
                if len(minima) > 0:
                    ax.scatter(r[minima], g_r[minima], color='red', s=100, 
                            marker='v', zorder=5, label='Minima')
                    
                    # Mark first minimum (coordination radius)
                    first_min = r[minima[0]]
                    ax.axvline(first_min, color='red', linestyle='--', alpha=0.7,
                            label=f'r₀ = {first_min:.2f} Å')
                
                ax.set_title(f'{ion_type}-Water RDF', fontweight='bold')
                ax.set_xlabel('r (Å)')
                ax.set_ylabel('g(r)')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 10)
            else:
                ax.text(0.5, 0.5, f'No RDF\n{rdf_key}', ha='center', va='center',
                    transform=ax.transAxes)
                ax.set_title(f'{ion_type}-Water RDF', fontweight='bold')
        
        # Hide unused subplots
        for i in range(len(ion_types), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('RDF Minima Detection for Coordination Radii', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('rdf_minima_detection.png', dpi=300, bbox_inches='tight')
            print("Plot saved: rdf_minima_detection.png")
        
        plt.show()


    def get_coordination_numbers_by_type(self, step=None):
        '''
        Calculate coordination numbers separately for each ion type.
        
        Parameters
        ----------
        step : int
            Step size for trajectory analysis, default uses auto-tuned value
        
        Returns
        -------
        coordination_numbers : dict
            Dictionary with ion types as keys and coordination data as values
        '''
        
        if step is None:
            step = self.default_step
        
        # Check if solutes are initialized by type
        if not (hasattr(self, 'solutes_ci') and hasattr(self, 'solutes_ai')):
            print('Ion-type-specific solutes not initialized. Run initialize_Solutes_by_type() first')
            return None
        
        # Get unique ion types and their coordination radii
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        n_frames = len(self.universe.trajectory[::step])
        
        # Initialize results dictionary
        coordination_numbers = {}
        
        print(f'Calculating coordination numbers by ion type (step={step})...')
        print('  Using vectorized distance calculations for better performance')
        
        # Process each cation type
        for cation_name, cation_group in cation_types.items():
            if cation_name in self.solutes_ci and self.solutes_ci[cation_name] is not None:
                cutoff = self.solutes_ci[cation_name].radii['water']
                
                # Pre-allocate array for this ion type
                cn_data = np.zeros(n_frames)
                
                print(f'  Processing {cation_name} ({len(cation_group)} ions, cutoff={cutoff:.2f} Å)...')
                
                for i, ts in enumerate(tqdm(self.universe.trajectory[::step], 
                                        desc=f"Calculating {cation_name} CNs", leave=False)):
                    
                    # Vectorized distance calculation for this cation type
                    d_cat = cdist(cation_group.positions, self.waters.positions)
                    n_coordinating = (d_cat <= cutoff).sum()
                    cn_data[i] = n_coordinating / len(cation_group)
                
                coordination_numbers[cation_name] = {
                    'type': 'cation',
                    'cutoff': cutoff,
                    'coordination_numbers': cn_data,
                    'mean': cn_data.mean(),
                    'std': cn_data.std(),
                    'std_error': cn_data.std() / np.sqrt(len(cn_data)),
                    'n_ions': len(cation_group)
                }
            else:
                print(f'  Warning: {cation_name} solute not properly initialized')
        
        # Process each anion type
        for anion_name, anion_group in anion_types.items():
            if anion_name in self.solutes_ai and self.solutes_ai[anion_name] is not None:
                cutoff = self.solutes_ai[anion_name].radii['water']
                
                # Pre-allocate array for this ion type
                cn_data = np.zeros(n_frames)
                
                print(f'  Processing {anion_name} ({len(anion_group)} ions, cutoff={cutoff:.2f} Å)...')
                
                for i, ts in enumerate(tqdm(self.universe.trajectory[::step], 
                                        desc=f"Calculating {anion_name} CNs", leave=False)):
                    
                    # Vectorized distance calculation for this anion type
                    d_an = cdist(anion_group.positions, self.waters.positions)
                    n_coordinating = (d_an <= cutoff).sum()
                    cn_data[i] = n_coordinating / len(anion_group)
                
                coordination_numbers[anion_name] = {
                    'type': 'anion',
                    'cutoff': cutoff,
                    'coordination_numbers': cn_data,
                    'mean': cn_data.mean(),
                    'std': cn_data.std(),
                    'std_error': cn_data.std() / np.sqrt(len(cn_data)),
                    'n_ions': len(anion_group)
                }
            else:
                print(f'  Warning: {anion_name} solute not properly initialized')
        
        # Store results
        self.coordination_numbers_by_type = coordination_numbers
        
        # Print summary
        self._print_coordination_summary_by_type(coordination_numbers)
        
        return coordination_numbers

    def _print_coordination_summary_by_type(self, coordination_numbers):
        '''Print summary of coordination numbers by ion type'''
        
        print("\n" + "="*70)
        print("COORDINATION NUMBERS BY ION TYPE")
        print("="*70)
        
        # Separate cations and anions
        cations_data = {k: v for k, v in coordination_numbers.items() if v['type'] == 'cation'}
        anions_data = {k: v for k, v in coordination_numbers.items() if v['type'] == 'anion'}
        
        if cations_data:
            print("\nCATIONS:")
            print("-" * 50)
            for ion_type, data in cations_data.items():
                print(f"{ion_type:>8s} ({data['n_ions']:>2d} ions): "
                    f"CN = {data['mean']:>5.2f} ± {data['std_error']:>4.2f}  "
                    f"(cutoff = {data['cutoff']:>4.2f} Å)")
        
        if anions_data:
            print("\nANIONS:")
            print("-" * 50)
            for ion_type, data in anions_data.items():
                print(f"{ion_type:>8s} ({data['n_ions']:>2d} ions): "
                    f"CN = {data['mean']:>5.2f} ± {data['std_error']:>4.2f}  "
                    f"(cutoff = {data['cutoff']:>4.2f} Å)")
        
        print("="*70)

    def plot_coordination_numbers_by_type(self, save_plot=True, plot_range=None):
        '''
        Plot coordination number time series for each ion type.
        
        Parameters
        ----------
        save_plot : bool
            Whether to save the plot, default=True
        plot_range : tuple
            Range of frames to plot (start, end), default=None (all frames)
        '''
        
        if not hasattr(self, 'coordination_numbers_by_type'):
            print("Coordination numbers by type not calculated. Run get_coordination_numbers_by_type() first.")
            return
        
        coordination_data = self.coordination_numbers_by_type
        
        # Separate cations and anions
        cations_data = {k: v for k, v in coordination_data.items() if v['type'] == 'cation'}
        anions_data = {k: v for k, v in coordination_data.items() if v['type'] == 'anion'}
        
        n_cations = len(cations_data)
        n_anions = len(anions_data)
        
        if n_cations == 0 and n_anions == 0:
            print("No coordination data found")
            return
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Get frame indices
        if plot_range is not None:
            start, end = plot_range
            frame_indices = np.arange(start, end)
        else:
            # Use the length from any ion type
            sample_data = list(coordination_data.values())[0]
            frame_indices = np.arange(len(sample_data['coordination_numbers']))
        
        # Plot cations
        if n_cations > 0:
            colors = plt.cm.Set1(np.linspace(0, 1, n_cations))
            
            for i, (ion_type, data) in enumerate(cations_data.items()):
                cn_data = data['coordination_numbers']
                if plot_range is not None:
                    cn_data = cn_data[start:end]
                
                axes[0].plot(frame_indices, cn_data, 
                            color=colors[i], linewidth=1.5, 
                            label=f"{ion_type} (mean={data['mean']:.2f})")
                
                # Add horizontal line for mean
                axes[0].axhline(data['mean'], color=colors[i], 
                            linestyle='--', alpha=0.7, linewidth=1)
            
            axes[0].set_title('Cation Coordination Numbers', fontweight='bold')
            axes[0].set_xlabel('Frame')
            axes[0].set_ylabel('Coordination Number')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'No cation data', 
                        transform=axes[0].transAxes, ha='center', va='center')
            axes[0].set_title('Cation Coordination Numbers', fontweight='bold')
        
        # Plot anions
        if n_anions > 0:
            colors = plt.cm.Set2(np.linspace(0, 1, n_anions))
            
            for i, (ion_type, data) in enumerate(anions_data.items()):
                cn_data = data['coordination_numbers']
                if plot_range is not None:
                    cn_data = cn_data[start:end]
                
                axes[1].plot(frame_indices, cn_data, 
                            color=colors[i], linewidth=1.5,
                            label=f"{ion_type} (mean={data['mean']:.2f})")
                
                # Add horizontal line for mean
                axes[1].axhline(data['mean'], color=colors[i], 
                            linestyle='--', alpha=0.7, linewidth=1)
            
            axes[1].set_title('Anion Coordination Numbers', fontweight='bold')
            axes[1].set_xlabel('Frame')
            axes[1].set_ylabel('Coordination Number')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No anion data', 
                        transform=axes[1].transAxes, ha='center', va='center')
            axes[1].set_title('Anion Coordination Numbers', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('coordination_numbers_by_type.png', dpi=300, bbox_inches='tight')
            print("Plot saved as: coordination_numbers_by_type.png")
        
        plt.show()

    def get_coordination_number_for_type(self, ion_type):
        '''
        Get coordination number data for a specific ion type.
        
        Parameters
        ----------
        ion_type : str
            Ion type name (e.g., 'Na', 'Mg', 'Cl')
        
        Returns
        -------
        data : dict or None
            Coordination number data for the specified ion type
        '''
        
        if not hasattr(self, 'coordination_numbers_by_type'):
            print("Coordination numbers by type not calculated. Run get_coordination_numbers_by_type() first.")
            return None
        
        if ion_type in self.coordination_numbers_by_type:
            return self.coordination_numbers_by_type[ion_type]
        else:
            available_types = list(self.coordination_numbers_by_type.keys())
            print(f"Ion type '{ion_type}' not found. Available types: {available_types}")
            return None


    def get_coordination_numbers_by_shell_and_type(self, step=None, use_kdtree=True, njobs=None):
        '''
        Calculate coordination numbers by shell for each ion type separately.
        OPTIMIZED: Process different ion types in parallel (much better than frame parallelization).
        '''
        
        if step is None:
            step = self.default_step
        
        if njobs is None:
            njobs = self.default_njobs
        
        if njobs == -1:
            njobs = min(multiprocessing.cpu_count(), 8)
        
        njobs = self._test_multiprocessing_compatibility(njobs)
        
        if not (hasattr(self, 'cation_shells_by_type') and hasattr(self, 'anion_shells_by_type')):
            print('Ion-type-specific shells not determined. Run determine_ion_solvation_shells_by_type() first')
            return None
        
        # Get unique ion types
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        # Prepare tasks - one task per ion type (BETTER parallelization unit)
        tasks = []
        
        for cation_name, cation_group in cation_types.items():
            if (cation_name in self.cation_shells_by_type and 
                self.cation_shells_by_type[cation_name] is not None):
                
                shells = self.cation_shells_by_type[cation_name]
                tasks.append({
                    'ion_name': cation_name,
                    'ion_category': 'cation',
                    'ion_group': cation_group,
                    'shells': shells,
                    'step': step,
                    'use_kdtree': use_kdtree
                })
        
        for anion_name, anion_group in anion_types.items():
            if (anion_name in self.anion_shells_by_type and 
                self.anion_shells_by_type[anion_name] is not None):
                
                shells = self.anion_shells_by_type[anion_name]
                tasks.append({
                    'ion_name': anion_name,
                    'ion_category': 'anion',
                    'ion_group': anion_group,
                    'shells': shells,
                    'step': step,
                    'use_kdtree': use_kdtree
                })
        
        if not tasks:
            print("No valid ion types with shell data found.")
            return None
        
        n_tasks = len(tasks)
        available_cpus = multiprocessing.cpu_count()
        actual_cpus = min(njobs, n_tasks)  # Don't use more CPUs than ion types
        
        print(f'Calculating shell coordination numbers by ion type')
        print(f'  Ion types to process: {n_tasks}')
        print(f'  Available CPUs: {available_cpus}')
        print(f'  CPUs to use: {actual_cpus}')
        print(f'  Method: {"KDTree" if use_kdtree else "vectorized distance matrices"}')
        print(f'  Step size: {step}')
        print(f'  Strategy: Parallel ion types (much faster than frame parallelization)')
        
        combined_results = {}
        
        # SIMPLE SEQUENTIAL - fastest for typical case
        if actual_cpus == 1 or n_tasks == 1:
            print("Using sequential processing (single CPU or single ion type)")
            for task in tqdm(tasks, desc="Processing ion types"):
                result = self._process_ion_shell_coordination_simple(task)
                if result['success']:
                    combined_results[result['ion_name']] = {
                        'type': result['ion_category'],
                        'shells': result['shell_data']
                    }
        else:
            # ONLY parallelize if we have multiple ion types and multiple CPUs
            print(f"Using parallel processing with {actual_cpus} workers")
            print("  (This is MUCH faster than parallelizing frames)")
            
            try:
                with Pool(actual_cpus, initializer=_worker_init) as pool:
                    results = list(tqdm(
                        pool.imap(self._process_ion_shell_coordination_simple, tasks),
                        total=len(tasks),
                        desc="Processing ion types (parallel)"
                    ))
                
                for result in results:
                    if result['success']:
                        combined_results[result['ion_name']] = {
                            'type': result['ion_category'],
                            'shells': result['shell_data']
                        }
                        
            except Exception as e:
                print(f"Parallel processing failed: {e}, falling back to sequential")
                for task in tqdm(tasks, desc="Processing ion types (fallback)"):
                    result = self._process_ion_shell_coordination_simple(task)
                    if result['success']:
                        combined_results[result['ion_name']] = {
                            'type': result['ion_category'],
                            'shells': result['shell_data']
                        }
        
        self.coordination_by_shell_and_type = combined_results
        self._print_shell_coordination_summary_by_type(combined_results)
        
        return combined_results


    def _process_ion_shell_coordination_simple(self, task):
        '''
        SIMPLIFIED: Process one ion type sequentially through all frames.
        This is MUCH faster than trying to parallelize frames.
        '''
        
        ion_name = task['ion_name']
        ion_category = task['ion_category']
        ion_group = task['ion_group']
        shells = task['shells']
        step = task['step']
        use_kdtree = task['use_kdtree']
        
        try:
            # Get frame indices
            if hasattr(self, '_debug_frame_indices') and self._debug_frame_indices is not None:
                frame_indices = self._debug_frame_indices[::step]
            else:
                frame_indices = list(range(0, len(self.universe.trajectory), step))
            
            n_frames = len(frame_indices)
            
            # Initialize shell data (EXCLUDE bulk)
            shell_data = {}
            for shell_name, (start, end) in shells.data.items():
                if shell_name != 'bulk':
                    shell_data[shell_name] = {
                        'bounds': (start, end),
                        'coordination_numbers': np.zeros(n_frames),
                        'name': shell_name.replace('_', ' ').title()
                    }
            
            if not shell_data:
                return {'success': False, 'ion_name': ion_name, 'ion_category': ion_category, 
                    'shell_data': None, 'error': 'No finite shells found'}
            
            # SEQUENTIAL frame processing (fastest for typical workload)
            for i, frame_idx in enumerate(frame_indices):
                self.universe.trajectory[frame_idx]
                
                if use_kdtree:
                    water_tree = cKDTree(self.waters.positions)
                    
                    for shell_name, shell_info in shell_data.items():
                        start_r, end_r = shell_info['bounds']
                        n_waters_in_shell = 0
                        
                        for ion_atom in ion_group:
                            indices_in_range = water_tree.query_ball_point(ion_atom.position, end_r)
                            for idx in indices_in_range:
                                dist = np.linalg.norm(self.waters.positions[idx] - ion_atom.position)
                                if start_r <= dist < end_r:
                                    n_waters_in_shell += 1
                        
                        shell_info['coordination_numbers'][i] = n_waters_in_shell / len(ion_group)
                else:
                    d = cdist(ion_group.positions, self.waters.positions)
                    
                    for shell_name, shell_info in shell_data.items():
                        start_r, end_r = shell_info['bounds']
                        in_shell = (d >= start_r) & (d < end_r)
                        n_waters_in_shell = in_shell.sum()
                        shell_info['coordination_numbers'][i] = n_waters_in_shell / len(ion_group)
            
            # Calculate statistics
            for shell_name, shell_info in shell_data.items():
                cn_data = shell_info['coordination_numbers']
                shell_info['mean'] = cn_data.mean()
                shell_info['std'] = cn_data.std()
                shell_info['std_error'] = cn_data.std() / np.sqrt(len(cn_data))
            
            return {
                'success': True,
                'ion_name': ion_name,
                'ion_category': ion_category,
                'shell_data': shell_data,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'ion_name': ion_name,
                'ion_category': ion_category,
                'shell_data': None,
                'error': str(e)
            }
 

    def _process_frame_chunk_for_shell_coordination(self, frame_chunk, ion_group, 
                                                    shell_data_template, use_kdtree):
        '''
        Process a chunk of frames for shell coordination in parallel.
        Returns coordination numbers for this chunk.
        '''
        
        # Initialize results for this chunk
        chunk_results = {shell_name: [] for shell_name in shell_data_template.keys()}
        
        for frame_idx in frame_chunk:
            self.universe.trajectory[frame_idx]
            
            if use_kdtree:
                water_tree = cKDTree(self.waters.positions)
                
                for shell_name, shell_info in shell_data_template.items():
                    start_r, end_r = shell_info['bounds']
                    n_waters_in_shell = 0
                    
                    for ion_atom in ion_group:
                        indices_in_range = water_tree.query_ball_point(ion_atom.position, end_r)
                        for idx in indices_in_range:
                            dist = np.linalg.norm(self.waters.positions[idx] - ion_atom.position)
                            if start_r <= dist < end_r:
                                n_waters_in_shell += 1
                    
                    chunk_results[shell_name].append(n_waters_in_shell / len(ion_group))
            else:
                d = cdist(ion_group.positions, self.waters.positions)
                
                for shell_name, shell_info in shell_data_template.items():
                    start_r, end_r = shell_info['bounds']
                    in_shell = (d >= start_r) & (d < end_r)
                    n_waters_in_shell = in_shell.sum()
                    chunk_results[shell_name].append(n_waters_in_shell / len(ion_group))
        
        # Convert lists to arrays
        for shell_name in chunk_results:
            chunk_results[shell_name] = np.array(chunk_results[shell_name])
        
        return chunk_results


    def _process_shell_coordination_task(self, task):
        '''
        Worker function for parallel shell coordination calculation.
        Processes one ion type.
        
        Parameters
        ----------
        task : dict
            Task dictionary with ion_name, ion_category, ion_group, shells, step, use_kdtree
        
        Returns
        -------
        result : dict
            Result dictionary with success status and shell coordination data
        '''
        
        ion_name = task['ion_name']
        ion_category = task['ion_category']
        ion_group = task['ion_group']
        shells = task['shells']
        step = task['step']
        use_kdtree = task['use_kdtree']
        
        try:
            print(f'  Processing {ion_name} shells ({len(ion_group)} ions)...')
            
            # Get frame indices
            if hasattr(self, '_debug_frame_indices') and self._debug_frame_indices is not None:
                frame_indices = self._debug_frame_indices[::step]
            else:
                frame_indices = range(0, len(self.universe.trajectory), step)
            
            n_frames = len(list(frame_indices))
            
            # Initialize coordination arrays for each shell (EXCLUDE bulk)
            shell_data = {}
            shell_count = 0
            
            for shell_name, (start, end) in shells.data.items():
                if shell_name != 'bulk':  # SKIP bulk region
                    shell_data[shell_name] = {
                        'bounds': (start, end),
                        'coordination_numbers': np.zeros(n_frames),
                        'name': shell_name.replace('_', ' ').title()
                    }
                    shell_count += 1
                else:
                    print(f'    Skipping {shell_name} region (extends to infinity)')
            
            if shell_count == 0:
                print(f'    Warning: No finite shells found for {ion_name}')
                return {
                    'success': False,
                    'ion_name': ion_name,
                    'ion_category': ion_category,
                    'shell_data': None,
                    'error': 'No finite shells found'
                }
            
            # Calculate coordination numbers for this ion type
            for i, frame_idx in enumerate(frame_indices):
                # Set trajectory to frame
                ts = self.universe.trajectory[frame_idx]
                
                if use_kdtree:
                    # Use KDTree for efficient neighbor searches
                    water_tree = cKDTree(self.waters.positions)
                    
                    for shell_name, shell_info in shell_data.items():
                        start_r, end_r = shell_info['bounds']
                        
                        # For each ion, count waters in shell
                        n_waters_in_shell = 0
                        for ion_atom in ion_group:
                            # Query waters in shell range
                            indices_in_range = water_tree.query_ball_point(ion_atom.position, end_r)
                            
                            # Filter to only those >= start_r
                            for idx in indices_in_range:
                                dist = np.linalg.norm(self.waters.positions[idx] - ion_atom.position)
                                if start_r <= dist < end_r:
                                    n_waters_in_shell += 1
                        
                        # Average over all ions
                        shell_info['coordination_numbers'][i] = n_waters_in_shell / len(ion_group)
                
                else:
                    # Use vectorized distance matrix
                    d = cdist(ion_group.positions, self.waters.positions)
                    
                    for shell_name, shell_info in shell_data.items():
                        start_r, end_r = shell_info['bounds']
                        
                        # Count waters in shell for all ions
                        in_shell = (d >= start_r) & (d < end_r)
                        n_waters_in_shell = in_shell.sum()
                        
                        # Average over all ions
                        shell_info['coordination_numbers'][i] = n_waters_in_shell / len(ion_group)
            
            # Calculate statistics for all shells
            for shell_name, shell_info in shell_data.items():
                cn_data = shell_info['coordination_numbers']
                shell_info['mean'] = cn_data.mean()
                shell_info['std'] = cn_data.std()
                shell_info['std_error'] = cn_data.std() / np.sqrt(len(cn_data))
            
            return {
                'success': True,
                'ion_name': ion_name,
                'ion_category': ion_category,
                'shell_data': shell_data,
                'error': None
            }
        
        except Exception as e:
            return {
                'success': False,
                'ion_name': ion_name,
                'ion_category': ion_category,
                'shell_data': None,
                'error': str(e)
            }


    def _print_shell_coordination_summary_by_type(self, results):
        '''Print summary of shell coordination numbers by ion type'''
        
        print("\n" + "="*70)
        print("SHELL COORDINATION NUMBERS BY ION TYPE")
        print("="*70)
        
        for ion_type, ion_data in results.items():
            ion_class = ion_data['type'].upper()
            print(f"\n{ion_class}: {ion_type}")
            print("-" * 50)
            
            for shell_name, shell_data in ion_data['shells'].items():
                start, end = shell_data['bounds']
                mean_cn = shell_data['mean']
                std_err = shell_data['std_error']
                
                end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                print(f"{shell_data['name']:12s}: {start:.2f} - {end_str:>6s} Å  |  "
                    f"CN = {mean_cn:.2f} ± {std_err:.2f}")
        
        print("="*70)

    def plot_shell_coordination_by_type(self, save_plot=True, plot_range=None):
        '''
        Plot shell coordination numbers for each ion type.
        
        Parameters
        ----------
        save_plot : bool
            Whether to save the plot, default=True
        plot_range : tuple
            Range of frames to plot (start, end), default=None (all frames)
        '''
        
        if not hasattr(self, 'coordination_by_shell_and_type'):
            print("Shell coordination numbers by type not calculated. Run get_coordination_numbers_by_shell_and_type() first.")
            return
        
        coordination_data = self.coordination_by_shell_and_type
        
        # Separate cations and anions
        cations_data = {k: v for k, v in coordination_data.items() if v['type'] == 'cation'}
        anions_data = {k: v for k, v in coordination_data.items() if v['type'] == 'anion'}
        
        n_cations = len(cations_data)
        n_anions = len(anions_data)
        
        if n_cations == 0 and n_anions == 0:
            print("No shell coordination data found")
            return
        
        # Create subplots - one for cations, one for anions
        fig = plt.figure(figsize=(16, 8))
        
        # Get frame indices
        if plot_range is not None:
            start, end = plot_range
            frame_indices = np.arange(start, end)
        else:
            # Use the length from any ion type's first shell
            sample_data = list(coordination_data.values())[0]
            sample_shell = list(sample_data['shells'].values())[0]
            frame_indices = np.arange(len(sample_shell['coordination_numbers']))
        
        plot_idx = 1
        
        # Plot cations
        if n_cations > 0:
            for cation_type, cation_data in cations_data.items():
                ax = plt.subplot(2, max(n_cations, n_anions), plot_idx)
                
                colors = plt.cm.Set1(np.linspace(0, 1, len(cation_data['shells'])))
                
                for i, (shell_name, shell_data) in enumerate(cation_data['shells'].items()):
                    cn_data = shell_data['coordination_numbers']
                    if plot_range is not None:
                        cn_data = cn_data[start:end]
                    
                    ax.plot(frame_indices, cn_data, 
                        color=colors[i], linewidth=1.5,
                        label=f"{shell_data['name']} (mean={shell_data['mean']:.2f})")
                    
                    # Add horizontal line for mean
                    ax.axhline(shell_data['mean'], color=colors[i], 
                            linestyle='--', alpha=0.7, linewidth=1)
                
                ax.set_title(f'{cation_type} Shell Coordination', fontweight='bold')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Coordination Number')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        # Fill remaining cation slots
        while plot_idx <= max(n_cations, n_anions):
            ax = plt.subplot(2, max(n_cations, n_anions), plot_idx)
            ax.set_visible(False)
            plot_idx += 1
        
        # Plot anions
        if n_anions > 0:
            for anion_type, anion_data in anions_data.items():
                ax = plt.subplot(2, max(n_cations, n_anions), plot_idx)
                
                colors = plt.cm.Set2(np.linspace(0, 1, len(anion_data['shells'])))
                
                for i, (shell_name, shell_data) in enumerate(anion_data['shells'].items()):
                    cn_data = shell_data['coordination_numbers']
                    if plot_range is not None:
                        cn_data = cn_data[start:end]
                    
                    ax.plot(frame_indices, cn_data, 
                        color=colors[i], linewidth=1.5,
                        label=f"{shell_data['name']} (mean={shell_data['mean']:.2f})")
                    
                    # Add horizontal line for mean
                    ax.axhline(shell_data['mean'], color=colors[i], 
                            linestyle='--', alpha=0.7, linewidth=1)
                
                ax.set_title(f'{anion_type} Shell Coordination', fontweight='bold')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Coordination Number')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        # Fill remaining anion slots
        while plot_idx <= 2 * max(n_cations, n_anions):
            ax = plt.subplot(2, max(n_cations, n_anions), plot_idx)
            ax.set_visible(False)
            plot_idx += 1
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('shell_coordination_by_type.png', dpi=300, bbox_inches='tight')
            print("Plot saved as: shell_coordination_by_type.png")
        
        plt.show()

    def get_shell_coordination_for_type(self, ion_type):
        '''
        Get shell coordination data for a specific ion type.
        
        Parameters
        ----------
        ion_type : str
            Ion type name (e.g., 'Na', 'Mg', 'Cl')
        
        Returns
        -------
        data : dict or None
            Shell coordination data for the specified ion type
        '''
        
        if not hasattr(self, 'coordination_by_shell_and_type'):
            print("Shell coordination numbers by type not calculated. Run get_coordination_numbers_by_shell_and_type() first.")
            return None
        
        if ion_type in self.coordination_by_shell_and_type:
            return self.coordination_by_shell_and_type[ion_type]
        else:
            available_types = list(self.coordination_by_shell_and_type.keys())
            print(f"Ion type '{ion_type}' not found. Available types: {available_types}")
            return None


    def save_shell_coordination_to_file(self, filename='shell_coordination_cache.pkl'):
        '''
        Save shell coordination numbers to file for persistence across sessions.
        
        Parameters
        ----------
        filename : str
            Output filename, default='shell_coordination_cache.pkl'
        
        Returns
        -------
        success : bool
            True if save was successful
        '''
        
        if not hasattr(self, 'coordination_by_shell_and_type') or not self.coordination_by_shell_and_type:
            print("No shell coordination data to save")
            return False
        
        try:
            # Prepare shell coordination data for serialization
            shell_coord_data = {}
            
            for ion_type, ion_data in self.coordination_by_shell_and_type.items():
                shell_coord_data[ion_type] = {
                    'type': ion_data['type'],
                    'shells': {}
                }
                
                for shell_name, shell_data in ion_data['shells'].items():
                    shell_coord_data[ion_type]['shells'][shell_name] = {
                        'bounds': shell_data['bounds'],
                        'coordination_numbers': shell_data['coordination_numbers'].copy(),
                        'name': shell_data['name'],
                        'mean': shell_data['mean'],
                        'std': shell_data['std'],
                        'std_error': shell_data['std_error']
                    }
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(shell_coord_data, f)
            
            print(f"Shell coordination data saved to {filename}")
            print(f"  Saved {len(shell_coord_data)} ion types")
            print(f"  Ion types: {list(shell_coord_data.keys())}")
            
            return True
            
        except Exception as e:
            print(f"Error saving shell coordination data: {e}")
            return False

    def load_shell_coordination_from_file(self, filename='shell_coordination_cache.pkl'):
        '''
        Load shell coordination data from file with error handling.
        
        Parameters
        ----------
        filename : str
            Input filename, default='shell_coordination_cache.pkl'
        
        Returns
        -------
        success : bool
            True if load was successful
        '''
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        try:
            # Check file size first
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print(f"Cache file {filename} is empty")
                return False
            
            # Load data
            with open(filename, 'rb') as f:
                shell_coord_data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(shell_coord_data, dict):
                print(f"Invalid shell coordination cache format")
                return False
            
            # Reconstruct the coordination data structure
            self.coordination_by_shell_and_type = {}
            
            for ion_type, ion_data in shell_coord_data.items():
                self.coordination_by_shell_and_type[ion_type] = {
                    'type': ion_data['type'],
                    'shells': {}
                }
                
                for shell_name, shell_data in ion_data['shells'].items():
                    self.coordination_by_shell_and_type[ion_type]['shells'][shell_name] = {
                        'bounds': shell_data['bounds'],
                        'coordination_numbers': shell_data['coordination_numbers'],
                        'name': shell_data['name'],
                        'mean': shell_data['mean'],
                        'std': shell_data['std'],
                        'std_error': shell_data['std_error']
                    }
            
            # Print summary
            successful_types = list(self.coordination_by_shell_and_type.keys())
            
            print(f"Shell coordination data loaded from {filename}")
            print(f"  Loaded {len(successful_types)} ion types successfully")
            if successful_types:
                print(f"  Available types: {', '.join(successful_types)}")
            
            # Print detailed summary
            print(f"\n  Shell coordination summary:")
            for ion_type, ion_data in self.coordination_by_shell_and_type.items():
                shell_count = len(ion_data['shells'])
                print(f"    {ion_type} ({ion_data['type']}): {shell_count} shells")
                for shell_name, shell_data in ion_data['shells'].items():
                    start, end = shell_data['bounds']
                    end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                    print(f"      {shell_data['name']}: {start:.2f}-{end_str} Å, CN={shell_data['mean']:.2f}±{shell_data['std_error']:.2f}")
            
            return True
            
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            print("Removing corrupted cache file...")
            try:
                os.remove(filename)
                print(f"Corrupted file {filename} removed")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading shell coordination data from {filename}: {e}")
            return False

    def get_coordination_numbers_by_shell_and_type_with_cache(self, cache_filename='shell_coordination_cache.pkl', 
                                                            force_recalc=False, **kwargs):
        '''
        Calculate shell coordination numbers with automatic caching.
        
        Parameters
        ----------
        cache_filename : str
            Cache file name, default='shell_coordination_cache.pkl'
        force_recalc : bool
            Force recalculation even if cache exists, default=False
        **kwargs : dict
            Arguments to pass to get_coordination_numbers_by_shell_and_type()
        
        Returns
        -------
        results : dict
            Dictionary of shell coordination results
        '''
        
        # Try to load from cache first
        if not force_recalc and os.path.exists(cache_filename):
            print("Attempting to load shell coordination data from cache...")
            if self.load_shell_coordination_from_file(cache_filename):
                print("✓ Successfully loaded shell coordination data from cache")
                return self.coordination_by_shell_and_type
            else:
                print("✗ Cache loading failed, will recalculate")
        
        # Calculate shell coordination numbers
        print("Calculating shell coordination numbers...")
        results = self.get_coordination_numbers_by_shell_and_type(**kwargs)
        
        # Save to cache
        if results:
            print("Saving shell coordination data to cache...")
            if self.save_shell_coordination_to_file(cache_filename):
                print("✓ Shell coordination data cached successfully")
            else:
                print("✗ Cache saving failed, but results are available in memory")
        
        return results


    def determine_ion_solvation_shells_by_type(self, ion_specific_params=None, plot=True, save_plots=True, plot_range=20, use_extended_rdf=False, create_combined=False, plot_only=None):
        '''
        Determine solvation shells with ion-type-specific parameters for maximum flexibility.
        
        Parameters
        ----------
        ion_specific_params : dict
            Ion-specific parameters for shell determination
        plot : bool
            Whether to generate plots
        save_plots : bool
            Whether to save plots
        plot_range : float
            Range for plotting
        use_extended_rdf : bool
            Whether to use extended RDF range
        create_combined : bool
            Whether to create combined shells for backward compatibility, default=False
        plot_only : str or list, optional
            Specific ion type(s) to plot. If None, plots all ions (if plot=True).
            Can be a single ion name (e.g., 'Na') or list of ion names (e.g., ['Na', 'Cl'])
        '''
        
        if not hasattr(self, 'rdfs') or not self.rdfs:
            print("No RDFs available. Run generate_rdfs() first.")
            return
        
        # Set default parameters if not provided
        if ion_specific_params is None:
            ion_specific_params = {}
        
        # Handle plot_only parameter
        if plot_only is not None:
            if isinstance(plot_only, str):
                plot_only = [plot_only]  # Convert single string to list
            plot_only = set(plot_only)  # Convert to set for faster lookup
            print(f"Plotting enabled only for: {list(plot_only)}")
        
        # Get unique ion types
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        print(f"Determining solvation shells with ion-specific parameters:")
        print(f"  Cation types: {list(cation_types.keys())}")  
        print(f"  Anion types: {list(anion_types.keys())}")
        
        # Initialize shell dictionaries
        self.cation_shells_by_type = {}
        self.anion_shells_by_type = {}
        
        # Process each cation type with its specific parameters
        for cation_name in cation_types.keys():
            rdf_key = f'{cation_name}-w'
            
            if rdf_key in self.rdfs and self.rdfs[rdf_key] is not None:
                print(f"  Processing {cation_name} solvation shells...")
                
                # Get ion-specific parameters or use defaults
                ion_params = ion_specific_params.get(cation_name, {})
                
                # Default parameters for this ion type
                default_find_peaks_kwargs = {'distance': 10, 'height': -3, 'prominence': 0.01}
                find_peaks_kwargs = ion_params.get('find_peaks_kwargs', default_find_peaks_kwargs)
                ion_plot_range = ion_params.get('plot_range', plot_range)
                ion_use_extended = ion_params.get('use_extended_rdf', use_extended_rdf)
                
                print(f"    Using parameters: {find_peaks_kwargs}")
                print(f"    Plot range: {ion_plot_range} Å, Extended RDF: {ion_use_extended}")
                
                # SEPARATE shell determination from plotting
                try:
                    shells = self._determine_shells_for_rdf_enhanced(
                        self.rdfs[rdf_key], 
                        ion_name=cation_name,
                        ion_type='cation',
                        use_extended_rdf=ion_use_extended,
                        find_peaks_kwargs=find_peaks_kwargs
                    )
                    
                    self.cation_shells_by_type[cation_name] = shells
                    print(f"    ✓ Shell determination successful for {cation_name}")
                    
                except Exception as e:
                    print(f"    ✗ Error determining {cation_name} shells: {e}")
                    self.cation_shells_by_type[cation_name] = None
                
                # CONDITIONAL plotting - only plot if this ion is in plot_only list (or plot_only is None)
                should_plot = plot and self.cation_shells_by_type[cation_name] is not None
                if plot_only is not None:
                    should_plot = should_plot and cation_name in plot_only
                
                if should_plot:
                    try:
                        self._plot_shells_for_ion_type_enhanced(
                            self.rdfs[rdf_key], shells, cation_name, 'cation', 
                            save_plots=save_plots, plot_range=ion_plot_range,
                            find_peaks_kwargs=find_peaks_kwargs
                        )
                        print(f"    ✓ Plot generated for {cation_name}")
                    except Exception as e:
                        print(f"    ⚠ Warning: Plotting failed for {cation_name}: {e}")
                        print(f"    (Shell data preserved)")
                elif plot and plot_only is not None and cation_name not in plot_only:
                    print(f"    - Skipping plot for {cation_name} (not in plot_only list)")
                            
            else:
                print(f"    Warning: {rdf_key} RDF not available")
                self.cation_shells_by_type[cation_name] = None
        
        # Process each anion type with its specific parameters
        for anion_name in anion_types.keys():
            rdf_key = f'{anion_name}-w'
            
            if rdf_key in self.rdfs and self.rdfs[rdf_key] is not None:
                print(f"  Processing {anion_name} solvation shells...")
                
                # Get ion-specific parameters or use defaults
                ion_params = ion_specific_params.get(anion_name, {})
                
                # Default parameters for this ion type
                default_find_peaks_kwargs = {'distance': 10, 'height': -3, 'prominence': 0.01}
                find_peaks_kwargs = ion_params.get('find_peaks_kwargs', default_find_peaks_kwargs)
                ion_plot_range = ion_params.get('plot_range', plot_range)
                ion_use_extended = ion_params.get('use_extended_rdf', use_extended_rdf)
                
                print(f"    Using parameters: {find_peaks_kwargs}")
                print(f"    Plot range: {ion_plot_range} Å, Extended RDF: {ion_use_extended}")
                
                # SEPARATE shell determination from plotting
                try:
                    shells = self._determine_shells_for_rdf_enhanced(
                        self.rdfs[rdf_key], 
                        ion_name=anion_name,
                        ion_type='anion',
                        use_extended_rdf=ion_use_extended,
                        find_peaks_kwargs=find_peaks_kwargs
                    )
                    
                    self.anion_shells_by_type[anion_name] = shells
                    print(f"    ✓ Shell determination successful for {anion_name}")
                    
                except Exception as e:
                    print(f"    ✗ Error determining {anion_name} shells: {e}")
                    self.anion_shells_by_type[anion_name] = None
                
                # CONDITIONAL plotting - only plot if this ion is in plot_only list (or plot_only is None)
                should_plot = plot and self.anion_shells_by_type[anion_name] is not None
                if plot_only is not None:
                    should_plot = should_plot and anion_name in plot_only
                
                if should_plot:
                    try:
                        self._plot_shells_for_ion_type_enhanced(
                            self.rdfs[rdf_key], shells, anion_name, 'anion',
                            save_plots=save_plots, plot_range=ion_plot_range,
                            find_peaks_kwargs=find_peaks_kwargs
                        )
                        print(f"    ✓ Plot generated for {anion_name}")
                    except Exception as e:
                        print(f"    ⚠ Warning: Plotting failed for {anion_name}: {e}")
                        print(f"    (Shell data preserved)")
                elif plot and plot_only is not None and anion_name not in plot_only:
                    print(f"    - Skipping plot for {anion_name} (not in plot_only list)")
                            
            else:
                print(f"    Warning: {rdf_key} RDF not available")
                self.anion_shells_by_type[anion_name] = None
        
        # Only create combined shells if explicitly requested
        if create_combined:
            print(f"  Creating combined shells for backward compatibility...")
            self._create_combined_shells_from_types(use_extended_rdf)
        else:
            print(f"  Skipping combined shell creation (create_combined=False)")
        
        # Print summary
        self._print_shells_summary_by_type()
        
        print("Ion-type-specific solvation shell determination complete!")
        
        return self.cation_shells_by_type, self.anion_shells_by_type


    def adjust_single_ion_shells(self, ion_type, find_peaks_kwargs=None, plot_range=None, use_extended_rdf=None, replot=True):
        '''
        Re-analyze solvation shells for a single ion type with new parameters.
        
        Parameters
        ----------
        ion_type : str
            Ion type to re-analyze (e.g., 'Na', 'Mg', 'Cl')
        find_peaks_kwargs : dict
            New peak finding parameters for this ion type
        plot_range : float
            New plot range for this ion type
        use_extended_rdf : bool
            Whether to use extended RDF for this ion type
        replot : bool
            Whether to generate new plot, default=True
        '''
        
        if not hasattr(self, 'rdfs') or not self.rdfs:
            print("No RDFs available. Run generate_rdfs() first.")
            return
        
        # Find which category this ion belongs to
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        if ion_type in cation_types:
            ion_category = 'cation'
            shells_dict = self.cation_shells_by_type
        elif ion_type in anion_types:
            ion_category = 'anion'
            shells_dict = self.anion_shells_by_type
        else:
            print(f"Ion type '{ion_type}' not found in system.")
            return
        
        rdf_key = f'{ion_type}-w'
        
        if rdf_key not in self.rdfs or self.rdfs[rdf_key] is None:
            print(f"RDF for {ion_type} not available.")
            return
        
        # Use current parameters if not specified
        if find_peaks_kwargs is None:
            find_peaks_kwargs = {'distance': 10, 'height': -3, 'prominence': 0.01}
        if plot_range is None:
            plot_range = 20
        if use_extended_rdf is None:
            use_extended_rdf = False
        
        print(f"Re-analyzing {ion_type} ({ion_category}) solvation shells...")
        print(f"  Parameters: {find_peaks_kwargs}")
        print(f"  Plot range: {plot_range} Å, Extended RDF: {use_extended_rdf}")
        
        try:
            shells = self._determine_shells_for_rdf_enhanced(
                self.rdfs[rdf_key], 
                ion_name=ion_type,
                ion_type=ion_category,
                use_extended_rdf=use_extended_rdf,
                find_peaks_kwargs=find_peaks_kwargs
            )
            
            shells_dict[ion_type] = shells
            
            if replot:
                self._plot_shells_for_ion_type_enhanced(
                    self.rdfs[rdf_key], shells, ion_type, ion_category, 
                    save_plots=True, plot_range=plot_range,
                    find_peaks_kwargs=find_peaks_kwargs
                )
            
            # Print updated shell info
            if shells is not None:
                print(f"\nUpdated {ion_type} shells:")
                for shell_name, (start, end) in shells.data.items():
                    width = end - start
                    print(f"  {shell_name.replace('_', ' ').title()}: "
                        f"{start:.2f} - {end:.2f} Å (width: {width:.2f} Å)")
            
            return shells
            
        except Exception as e:
            print(f"Error re-analyzing {ion_type} shells: {e}")
            return None

    def optimize_parameters_for_ion(self, ion_type, distance_range=(5, 20), height_range=(-5, -1), prominence_range=(0.005, 0.1), plot_best=True):
        '''
        Automatically test different parameter combinations for a specific ion type to find optimal shell detection.
        
        Parameters
        ----------
        ion_type : str
            Ion type to optimize (e.g., 'Na', 'Mg', 'Cl')
        distance_range : tuple
            Range of distance values to test, default=(5, 20)
        height_range : tuple
            Range of height values to test, default=(-5, -1)
        prominence_range : tuple
            Range of prominence values to test, default=(0.005, 0.1)
        plot_best : bool
            Whether to plot the best result, default=True
        
        Returns
        -------
        best_params : dict
            Best parameters found
        results : list
            List of all tested parameter combinations and their results
        '''
        
        if not hasattr(self, 'rdfs') or not self.rdfs:
            print("No RDFs available. Run generate_rdfs() first.")
            return None, None
        
        rdf_key = f'{ion_type}-w'
        if rdf_key not in self.rdfs or self.rdfs[rdf_key] is None:
            print(f"RDF for {ion_type} not available.")
            return None, None
        
        # Parameter grid
        distance_values = [5, 8, 10, 12, 15, 20]
        height_values = [-5, -4, -3, -2, -1.5, -1]
        prominence_values = [0.005, 0.01, 0.02, 0.05, 0.1]
        
        print(f"Optimizing parameters for {ion_type}...")
        print(f"Testing {len(distance_values)} × {len(height_values)} × {len(prominence_values)} = {len(distance_values)*len(height_values)*len(prominence_values)} combinations")
        
        results = []
        
        for distance in tqdm(distance_values, desc="Distance"):
            for height in height_values:
                for prominence in prominence_values:
                    try:
                        find_peaks_kwargs = {
                            'distance': distance,
                            'height': height,
                            'prominence': prominence
                        }
                        
                        # Test these parameters
                        shells = self._determine_shells_for_rdf_enhanced(
                            self.rdfs[rdf_key], 
                            ion_name=ion_type,
                            ion_type='cation' if ion_type in self._get_unique_ion_types(self.cations) else 'anion',
                            use_extended_rdf=True,
                            find_peaks_kwargs=find_peaks_kwargs
                        )
                        
                        if shells is not None:
                            n_shells = len(shells.data)
                            n_minima = len(shells.minima)
                            
                            # Score based on number of shells and minima found
                            # Prefer 2-3 shells with clear minima
                            if n_shells == 2:
                                score = 10 + n_minima
                            elif n_shells == 3:
                                score = 15 + n_minima
                            elif n_shells == 1:
                                score = 5 + n_minima
                            else:
                                score = n_minima
                            
                            results.append({
                                'params': find_peaks_kwargs.copy(),
                                'n_shells': n_shells,
                                'n_minima': n_minima,
                                'score': score,
                                'shells': shells
                            })
                        
                    except Exception:
                        # Skip parameter combinations that fail
                        continue
        
        if not results:
            print("No valid parameter combinations found.")
            return None, None
        
        # Find best result
        best_result = max(results, key=lambda x: x['score'])
        best_params = best_result['params']
        
        print(f"\nOptimization complete for {ion_type}:")
        print(f"  Best parameters: {best_params}")
        print(f"  Results: {best_result['n_shells']} shells, {best_result['n_minima']} minima")
        print(f"  Score: {best_result['score']}")
        
        # Show top 3 results
        top_results = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
        print(f"\nTop 3 parameter combinations:")
        for i, result in enumerate(top_results):
            print(f"  {i+1}. {result['params']} -> {result['n_shells']} shells, {result['n_minima']} minima (score: {result['score']})")
        
        # Apply best parameters and plot
        if plot_best:
            print(f"\nApplying best parameters and plotting...")
            self.adjust_single_ion_shells(
                ion_type, 
                find_peaks_kwargs=best_params,
                plot_range=20,
                use_extended_rdf=True,
                replot=True
            )
        
        return best_params, results

    def compare_ion_parameters(self, ion_types=None, parameter_sets=None):
        '''
        Compare different parameter sets across multiple ion types.
        
        Parameters
        ----------
        ion_types : list
            List of ion types to compare. If None, uses all available types
        parameter_sets : dict
            Dictionary of parameter sets to test. Keys are set names, values are find_peaks_kwargs
        '''
        
        if ion_types is None:
            cation_types = list(self._get_unique_ion_types(self.cations).keys())
            anion_types = list(self._get_unique_ion_types(self.anions).keys())
            ion_types = cation_types + anion_types
        
        if parameter_sets is None:
            parameter_sets = {
                'Conservative': {'distance': 15, 'height': -4, 'prominence': 0.05},
                'Standard': {'distance': 10, 'height': -3, 'prominence': 0.01},
                'Aggressive': {'distance': 5, 'height': -2, 'prominence': 0.005}
            }
        
        print(f"Comparing parameter sets across {len(ion_types)} ion types:")
        print(f"Ion types: {ion_types}")
        print(f"Parameter sets: {list(parameter_sets.keys())}")
        
        comparison_results = {}
        
        for ion_type in ion_types:
            comparison_results[ion_type] = {}
            
            for set_name, params in parameter_sets.items():
                try:
                    shells = self.adjust_single_ion_shells(
                        ion_type, 
                        find_peaks_kwargs=params,
                        replot=False
                    )
                    
                    if shells is not None:
                        comparison_results[ion_type][set_name] = {
                            'n_shells': len(shells.data),
                            'n_minima': len(shells.minima),
                            'shells_data': shells.data
                        }
                    else:
                        comparison_results[ion_type][set_name] = None
                        
                except Exception as e:
                    comparison_results[ion_type][set_name] = None
        
        # Print comparison table
        print(f"\n{'='*80}")
        print(f"PARAMETER COMPARISON RESULTS")
        print(f"{'='*80}")
        
        for ion_type in ion_types:
            print(f"\n{ion_type.upper()}:")
            print("-" * 40)
            
            for set_name, result in comparison_results[ion_type].items():
                if result is not None:
                    print(f"  {set_name:12s}: {result['n_shells']} shells, {result['n_minima']} minima")
                else:
                    print(f"  {set_name:12s}: FAILED")
        
        return comparison_results

    def _create_combined_shells_from_types(self, use_extended_rdf=False):
        '''
        Create combined cation and anion shells from ion-type-specific shells.
        This maintains backward compatibility with existing code.
        '''
        
        # Find the most abundant ion types for combined shells
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        # Use most abundant cation type for combined cation shells
        if cation_types:
            most_abundant_cation = max(cation_types.keys(), 
                                    key=lambda x: len(cation_types[x]))
            
            if (hasattr(self, 'cation_shells_by_type') and 
                most_abundant_cation in self.cation_shells_by_type and 
                self.cation_shells_by_type[most_abundant_cation] is not None):
                self.cation_solvation_shells = self.cation_shells_by_type[most_abundant_cation]
                print(f"    Combined cation shells use: {most_abundant_cation}")
            else:
                print(f"    Warning: Could not create combined cation shells")
                self.cation_solvation_shells = None
        
        # Use most abundant anion type for combined anion shells
        if anion_types:
            most_abundant_anion = max(anion_types.keys(),
                                    key=lambda x: len(anion_types[x]))
            
            if (hasattr(self, 'anion_shells_by_type') and 
                most_abundant_anion in self.anion_shells_by_type and 
                self.anion_shells_by_type[most_abundant_anion] is not None):
                self.anion_solvation_shells = self.anion_shells_by_type[most_abundant_anion]
                print(f"    Combined anion shells use: {most_abundant_anion}")
            else:
                print(f"    Warning: Could not create combined anion shells")
                self.anion_solvation_shells = None

    def interactive_shell_tuning(self, ion_type):
        '''
        Interactive parameter tuning for a specific ion type.
        '''
        
        print(f"Interactive shell parameter tuning for {ion_type}")
        print("Enter 'quit' to exit, 'help' for commands")
        
        current_params = {'distance': 10, 'height': -3, 'prominence': 0.01}
        current_plot_range = 20
        
        while True:
            command = input(f"\n[{ion_type}] Current params {current_params}, range={current_plot_range} > ").strip()
            
            if command.lower() == 'quit':
                break
            elif command.lower() == 'help':
                print("Commands:")
                print("  distance X     - Set distance parameter")
                print("  height X       - Set height parameter") 
                print("  prominence X   - Set prominence parameter")
                print("  range X        - Set plot range")
                print("  test           - Test current parameters")
                print("  reset          - Reset to defaults")
                print("  quit           - Exit")
            elif command.startswith('distance '):
                try:
                    current_params['distance'] = int(float(command.split()[1]))
                    print(f"Distance set to {current_params['distance']}")
                except:
                    print("Invalid distance value")
            elif command.startswith('height '):
                try:
                    current_params['height'] = float(command.split()[1])
                    print(f"Height set to {current_params['height']}")
                except:
                    print("Invalid height value")
            elif command.startswith('prominence '):
                try:
                    current_params['prominence'] = float(command.split()[1])
                    print(f"Prominence set to {current_params['prominence']}")
                except:
                    print("Invalid prominence value")
            elif command.startswith('range '):
                try:
                    current_plot_range = float(command.split()[1])
                    print(f"Plot range set to {current_plot_range}")
                except:
                    print("Invalid range value")
            elif command == 'test':
                self.adjust_single_ion_shells(
                    ion_type,
                    find_peaks_kwargs=current_params,
                    plot_range=current_plot_range,
                    use_extended_rdf=True
                )
            elif command == 'reset':
                current_params = {'distance': 10, 'height': -3, 'prominence': 0.01}
                current_plot_range = 20
                print("Parameters reset to defaults")
            else:
                print("Unknown command. Type 'help' for available commands.")
        
        return current_params, current_plot_range


    def _determine_shells_for_rdf_enhanced(self, rdf_data, ion_name, ion_type, use_extended_rdf=False, find_peaks_kwargs=None):
        '''
        Enhanced shell determination with proper peak/minima finding and guaranteed attribute setting.
        FIXED: Uses non-protected attribute names for MDAnalysis Results compatibility.
        '''
        
        from MDAnalysis.analysis.base import Results
        
        # Get RDF data
        r = rdf_data.bins
        g_r = rdf_data.rdf
        
        # Set default find_peaks_kwargs if not provided
        if find_peaks_kwargs is None:
            find_peaks_kwargs = {'distance': 10, 'height': -3, 'prominence': 0.01}
        
        # Find peaks in the RDF (maxima)
        peaks, peak_properties = find_peaks(g_r, height=1.0, distance=5)
        
        # Find minima using the provided parameters
        minima, minima_properties = find_peaks(-g_r, **find_peaks_kwargs)
        
        if len(peaks) == 0:
            print(f"    Warning: No clear peaks found in {ion_name} RDF")
            return None
        
        # Sort by position
        peaks = peaks[np.argsort(r[peaks])]
        minima = minima[np.argsort(r[minima])]
        
        print(f"    Found {len(peaks)} peaks and {len(minima)} minima for {ion_name}")
        
        # Create Results object and use NON-PROTECTED attribute names
        shells = Results()
        
        # FIXED: Use different attribute names to avoid protected dictionary attributes
        shells.data = {}                    # Shell boundaries - this is fine
        shells.peak_indices = peaks.copy()  # Changed from 'peaks'
        shells.minima_indices = minima.copy()  # Changed from 'minima'  
        shells.rdf_r = r.copy()            # Changed from 'r'
        shells.rdf_g_r = g_r.copy()        # Changed from 'g_r'
        
        # Also use setattr as backup with the new names
        setattr(shells, 'data', {})
        setattr(shells, 'peak_indices', peaks.copy())
        setattr(shells, 'minima_indices', minima.copy())
        setattr(shells, 'rdf_r', r.copy())
        setattr(shells, 'rdf_g_r', g_r.copy())
        
        # Verify attributes were set
        r_exists = hasattr(shells, 'rdf_r') and shells.rdf_r is not None
        g_r_exists = hasattr(shells, 'rdf_g_r') and shells.rdf_g_r is not None
        minima_exists = hasattr(shells, 'minima_indices') and shells.minima_indices is not None
        peaks_exists = hasattr(shells, 'peak_indices') and shells.peak_indices is not None
        
        if not all([r_exists, g_r_exists, minima_exists, peaks_exists]):
            print(f"    ERROR: Failed to set shell attributes!")
            print(f"    rdf_r={r_exists}, rdf_g_r={g_r_exists}, minima_indices={minima_exists}, peak_indices={peaks_exists}")
            return None
        
        # Determine the maximum range to use
        max_r = r[-1] if use_extended_rdf else min(12.0, r[-1])
        
        # Filter minima if too many found
        if len(minima) > 10:
            print(f"    Warning: Found {len(minima)} minima for {ion_name}, using only first 5 most prominent ones")
            minima_heights = -g_r[minima]
            most_prominent_indices = np.argsort(minima_heights)[:5]
            minima = minima[most_prominent_indices]
            minima = minima[np.argsort(r[minima])]
            print(f"    Using {len(minima)} most prominent minima")
            
            # Update stored minima with new name
            shells.minima_indices = minima.copy()
            setattr(shells, 'minima_indices', minima.copy())
        
        # Check if we have minima to work with
        if len(minima) == 0:
            print(f"    Warning: No minima found for {ion_name}")
            return None
        
        # Determine shell boundaries
        shells.data = {}  # Reset to ensure clean state
        
        # First shell - from 0 to first minimum
        first_shell_end = min(r[minima[0]], max_r)
        shells.data['shell_1'] = (0.0, first_shell_end)
        
        # Additional shells based on remaining minima (limit to 3 shells max)
        for i in range(1, min(len(minima), 3)):
            shell_start = shells.data[f'shell_{i}'][1]  # End of previous shell
            shell_end = min(r[minima[i]], max_r)
            
            # Only create shell if it's meaningful (minimum width of 0.5 Å)
            if shell_end > shell_start + 0.5:
                shells.data[f'shell_{i+1}'] = (shell_start, shell_end)
            else:
                break
        
        # ADD BULK REGION - this was missing!
        # Find the end of the last shell
        last_shell_end = 0
        for shell_name, (start, end) in shells.data.items():
            if shell_name.startswith('shell_'):
                last_shell_end = max(last_shell_end, end)
        
        # Add bulk region from end of last shell to max_r (or infinity)
        if last_shell_end < max_r:
            shells.data['bulk'] = (last_shell_end, np.inf)
            setattr(shells, 'bulk', (last_shell_end, np.inf))
            print(f"    Added bulk region: {last_shell_end:.2f} - ∞ Å")
        
        # Validate shell data
        validated_shells = {}
        for shell_name, shell_bounds in shells.data.items():
            if ((shell_name.startswith('shell_') or shell_name == 'bulk') and 
                isinstance(shell_bounds, (tuple, list)) and 
                len(shell_bounds) == 2):
                
                try:
                    start, end = float(shell_bounds[0]), float(shell_bounds[1])
                    
                    if shell_name == 'bulk':
                        # For bulk, end can be infinity
                        if (np.isfinite(start) and start >= 0 and 
                            (np.isfinite(end) or np.isinf(end)) and end > start):
                            validated_shells[shell_name] = (start, end)
                        else:
                            print(f"    Warning: Invalid bulk bounds: ({start:.2f}, {end:.2f})")
                    else:
                        # For shells, both must be finite
                        if (np.isfinite(start) and np.isfinite(end) and 
                            end > start and start >= 0):
                            validated_shells[shell_name] = (start, end)
                        else:
                            print(f"    Warning: Invalid shell bounds for {shell_name}: ({start:.2f}, {end:.2f})")
                            
                except (TypeError, ValueError) as e:
                    print(f"    Warning: Error converting shell bounds for {shell_name}: {e}")
                    continue
            else:
                print(f"    Warning: Invalid shell format for {shell_name}: {shell_bounds}")
                continue
        
        shells.data = validated_shells
        
        if len([k for k in shells.data.keys() if k.startswith('shell_')]) == 0:
            print(f"    Warning: No valid shells determined for {ion_name}")
            return None
        
        # Success message
        shell_count = len([k for k in shells.data.keys() if k.startswith('shell_')])
        has_bulk = 'bulk' in shells.data
        print(f"    Successfully determined {shell_count} shells{'+ bulk' if has_bulk else ''} for {ion_name}")
        for shell_name, (start, end) in shells.data.items():
            end_str = f"{end:.2f}" if np.isfinite(end) else "∞"
            print(f"      {shell_name}: {start:.2f} - {end_str} Å")
        
        return shells

    def _plot_shells_for_ion_type_enhanced(self, rdf_data, shells, ion_name, ion_type, save_plots=True, plot_range=20, find_peaks_kwargs=None):
        '''
        Enhanced plotting method for ion-type-specific solvation shells.
        MODIFIED: Starting with #00c5ff and decreasing saturation, includes bulk region.
        '''
        
        if shells is None:
            print(f"No shells to plot for {ion_name}")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Get RDF data using the new attribute names
        if hasattr(shells, 'rdf_r') and hasattr(shells, 'rdf_g_r'):
            r = shells.rdf_r
            g_r = shells.rdf_g_r
        else:
            # Fallback to rdf_data
            r = rdf_data.bins
            g_r = rdf_data.rdf
        
        # Get minima using the new attribute names (we still need this for shell boundaries)
        if hasattr(shells, 'minima_indices'):
            minima = shells.minima_indices
        else:
            if find_peaks_kwargs is None:
                find_peaks_kwargs = {'distance': 10, 'height': -3, 'prominence': 0.01}
            minima, _ = find_peaks(-g_r, **find_peaks_kwargs)
        
        # Plot RDF (no label to avoid legend)
        plt.plot(r, g_r, color='k', linewidth=2)
        
        # FIXED: Count only shell regions, not bulk
        shell_regions = {k: v for k, v in shells.data.items() if k.startswith('shell_')}
        n_shells = len(shell_regions)
        has_bulk = 'bulk' in shells.data
        
        # Define blue saturation gradient starting with #00c5ff INCLUDING bulk region
        def get_blue_saturation_colors_from_00c5ff(n_shells):
            """Generate blue colors starting from #00c5ff with decreasing saturation, including bulk"""
            import matplotlib.colors as mcolors
            
            # Convert #00c5ff to HSV to get the base hue and saturation
            base_rgb = mcolors.hex2color('#00c5ff')
            base_hsv = mcolors.rgb_to_hsv(base_rgb)
            base_hue = base_hsv[0]          # Extract the hue
            base_saturation = base_hsv[1]   # Extract the saturation (~1.0)
            base_value = base_hsv[2]        # Extract the value/brightness
            
            # Create colors from base saturation down to very light
            if n_shells == 1:
                saturations = [base_saturation, 0.2]  # Shell 1 (#00c5ff) + Bulk (very light)
            elif n_shells == 2:
                saturations = [base_saturation, 0.6, 0.2]  # Shell 1 + Shell 2 + Bulk
            elif n_shells == 3:
                saturations = [base_saturation, 0.7, 0.4, 0.2]  # Shell 1 + Shell 2 + Shell 3 + Bulk
            else:
                # For more shells, create even gradient from base_saturation down
                step = (base_saturation - 0.2) / n_shells
                saturations = [base_saturation - (i * step) for i in range(n_shells)]
                saturations.append(0.2)  # Always very light for bulk
            
            # Convert HSV back to RGB hex colors
            colors = []
            for sat in saturations:
                # HSV: (hue, saturation, value/brightness)
                hsv = (base_hue, sat, base_value)  # Keep original hue and brightness
                rgb = mcolors.hsv_to_rgb(hsv)
                colors.append(mcolors.to_hex(rgb))
            
            return colors
        
        # Generate colors for shells + bulk
        all_colors = get_blue_saturation_colors_from_00c5ff(n_shells)  # n_shells excludes bulk
        shell_colors = all_colors[:-1]  # Colors for shells only
        bulk_color = all_colors[-1]     # Lightest color for bulk
        
        # Calculate label position well above the RDF curve
        rdf_max = max(g_r)
        label_y_position = rdf_max * 1.15  # 15% above the highest point
        
        # Plot defined shells ONLY
        last_shell_end = 0
        shell_items = [(k, v) for k, v in shells.data.items() if k.startswith('shell_')]
        shell_items.sort(key=lambda x: x[1][0])  # Sort by start position
        
        for i, (shell_name, (start, end)) in enumerate(shell_items):
            if i < len(shell_colors):
                shell_color = shell_colors[i]
            else:
                shell_color = shell_colors[-1]  # Fallback color
            
            # Fill shell region (no label to avoid legend)
            plt.axvspan(start, end, alpha=0.4, color=shell_color)
            
            # Add simple shell labels ABOVE the RDF curve
            mid_point = (start + end) / 2
            plt.text(mid_point, label_y_position, f'Shell {i+1}', 
                    horizontalalignment='center', fontweight='bold', fontsize=12,
                    color='black')
            
            # Track the end of the last shell
            last_shell_end = max(last_shell_end, end)
        
        # Add bulk region if it exists in the data OR if we have space beyond last shell
        if has_bulk:
            bulk_start, bulk_end = shells.data['bulk']
            # Use the actual bulk boundaries from the data
            if np.isinf(bulk_end):
                bulk_plot_end = plot_range
            else:
                bulk_plot_end = min(bulk_end, plot_range)
            
            # Fill bulk region with the lightest blue color
            plt.axvspan(bulk_start, bulk_plot_end, alpha=0.4, color=bulk_color)
            
            # Add bulk label
            bulk_mid_point = (bulk_start + bulk_plot_end) / 2
            plt.text(bulk_mid_point, label_y_position, 'Bulk', 
                    horizontalalignment='center', fontweight='bold', fontsize=12,
                    color='black')
            
            print(f"    Plotted bulk region: {bulk_start:.2f} - {bulk_plot_end:.2f} Å")
        
        elif last_shell_end < plot_range:
            # Fallback: create bulk region if none exists but there's space
            plt.axvspan(last_shell_end, plot_range, alpha=0.4, color=bulk_color)
            
            # Add bulk label
            bulk_mid_point = (last_shell_end + plot_range) / 2
            plt.text(bulk_mid_point, label_y_position, 'Bulk', 
                    horizontalalignment='center', fontweight='bold', fontsize=12,
                    color='black')
            
            print(f"    Created fallback bulk region: {last_shell_end:.2f} - {plot_range:.2f} Å")
        
        # Formatting
        plt.xlabel('r (Å)', fontsize=12)
        plt.ylabel('g(r)', fontsize=12)
        plt.title(f'{ion_name.upper()} Solvation Shells Analysis', fontsize=14, fontweight='bold')
        
        # Add parameter info to title
        if find_peaks_kwargs:
            param_str = ', '.join([f"{k}={v}" for k, v in find_peaks_kwargs.items()])
            plt.title(f'{ion_name.upper()} Solvation Shells\n(find_peaks: {param_str})', 
                    fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.xlim(0, plot_range)
        
        # Adjust y-axis limits to accommodate the labels above
        plt.ylim(bottom=0, top=rdf_max * 1.3)  # 30% extra space at top for labels
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'{ion_name}_{ion_type}_solvation_shells_enhanced.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"    Enhanced shell plot saved: {filename}")
        
        plt.show()


    def _print_shells_summary_by_type(self):
        '''Enhanced summary with more details'''
        
        print("\n" + "="*70)
        print("SOLVATION SHELLS BY ION TYPE")
        print("="*70)
        
        if hasattr(self, 'cation_shells_by_type'):
            print("\nCATIONS:")
            print("-" * 50)
            for ion_type, shells in self.cation_shells_by_type.items():
                if shells is not None:
                    print(f"\n{ion_type}:")
                    for shell_name, (start, end) in shells.data.items():
                        width = end - start
                        print(f"  {shell_name.replace('_', ' ').title()}: "
                            f"{start:.2f} - {end:.2f} Å (width: {width:.2f} Å)")
                    
                    # Print peak/minima info
                    if hasattr(shells, 'peaks') and hasattr(shells, 'minima'):
                        print(f"  Found {len(shells.peaks)} peaks, {len(shells.minima)} minima")
                else:
                    print(f"\n{ion_type}: Shell determination failed")
        
        if hasattr(self, 'anion_shells_by_type'):
            print("\nANIONS:")
            print("-" * 50)
            for ion_type, shells in self.anion_shells_by_type.items():
                if shells is not None:
                    print(f"\n{ion_type}:")
                    for shell_name, (start, end) in shells.data.items():
                        width = end - start
                        print(f"  {shell_name.replace('_', ' ').title()}: "
                            f"{start:.2f} - {end:.2f} Å (width: {width:.2f} Å)")
                    
                    # Print peak/minima info
                    if hasattr(shells, 'peaks') and hasattr(shells, 'minima'):
                        print(f"  Found {len(shells.peaks)} peaks, {len(shells.minima)} minima")
                else:
                    print(f"\n{ion_type}: Shell determination failed")
        
        print("="*70)

    def get_solvation_shells_for_type(self, ion_type):
        '''
        Get solvation shells for a specific ion type.
        
        Parameters
        ----------
        ion_type : str
            Ion type name (e.g., 'Na', 'Mg', 'Cl')
        
        Returns
        -------
        shells : Results or None
            Solvation shell data for the specified ion type
        '''
        
        if hasattr(self, 'cation_shells_by_type') and ion_type in self.cation_shells_by_type:
            return self.cation_shells_by_type[ion_type]
        elif hasattr(self, 'anion_shells_by_type') and ion_type in self.anion_shells_by_type:
            return self.anion_shells_by_type[ion_type]
        else:
            available_types = []
            if hasattr(self, 'cation_shells_by_type'):
                available_types.extend(list(self.cation_shells_by_type.keys()))
            if hasattr(self, 'anion_shells_by_type'):
                available_types.extend(list(self.anion_shells_by_type.keys()))
            
            print(f"Ion type '{ion_type}' not found. Available types: {available_types}")
            return None 


    def remove_ion_shell(self, ion_type='cation', shell_to_remove='shell_2', merge_option='expand_previous'):
        '''
        Remove a specific solvation shell from the determined shells and update the results.
        This allows manual curation of shells if automated peak detection finds spurious minima.
        Can handle both broad categories ('cation', 'anion') and specific ion types ('Na', 'Mg', 'Cl').
        
        Parameters
        ----------
        ion_type : str
            Type of ion to modify. Options are:
            - Broad categories: 'cation' or 'anion' (works with existing shell data)
            - Specific ion types: 'Na', 'Mg', 'Cl', etc. (requires ion-type-specific shells)
            default='cation'
        shell_to_remove : str
            Shell attribute to remove (e.g., 'shell_2', 'shell_3'), default='shell_2'
        merge_option : str
            How to handle the removal:
            - 'expand_previous': Expand the previous shell to include the removed shell's range
            - 'expand_next': Expand the next shell to include the removed shell's range
            - 'expand_both': Expand both previous and next shells to meet in the middle
            default='expand_previous'
        
        Returns
        -------
        success : bool
            Whether the shell was successfully removed
        '''
        
        # First, try to determine if this is a specific ion type or broad category
        if ion_type in ['cation', 'anion']:
            # Handle broad categories (existing functionality)
            return self._remove_shell_broad_category(ion_type, shell_to_remove, merge_option)
        else:
            # Handle specific ion types (new functionality)
            return self._remove_shell_specific_ion(ion_type, shell_to_remove, merge_option)

    def _remove_shell_broad_category(self, ion_type, shell_to_remove, merge_option):
        '''Handle removal for broad categories (cation/anion) - original functionality'''
        
        try:
            if ion_type == 'cation':
                shells = self.cation_solvation_shells
            elif ion_type == 'anion':
                shells = self.anion_solvation_shells
            else:
                print("Error: ion_type must be 'cation' or 'anion' for broad categories")
                return False
        except AttributeError:
            print(f"Error: {ion_type} solvation shells not found. Run determine_ion_solvation_shells() first")
            return False
        
        # Get all shell attributes
        available_shells = list(shells.data.keys()) if hasattr(shells, 'data') else [attr for attr in dir(shells) if attr.startswith('shell_') or attr == 'bulk']
        
        # Check if the shell exists
        if shell_to_remove not in available_shells:
            print(f"Error: Shell '{shell_to_remove}' not found in {ion_type} shells")
            print(f"Available shells: {available_shells}")
            return False
        
        # Get shell numbers only (exclude bulk)
        shell_attrs = [attr for attr in available_shells if attr.startswith('shell_')]
        shell_attrs.sort()
        
        # Find the shell index
        shell_idx = shell_attrs.index(shell_to_remove)
        
        # Get shell boundaries
        if hasattr(shells, 'data'):
            shell_start, shell_end = shells.data[shell_to_remove]
        else:
            shell_start, shell_end = getattr(shells, shell_to_remove)
        
        print(f"Removing {ion_type} {shell_to_remove}: {shell_start:.2f} - {shell_end:.2f} Å")
        print(f"Using merge option: {merge_option}")
        
        # Handle edge cases
        is_first_shell = (shell_to_remove == 'shell_1')
        is_last_shell = (shell_idx == len(shell_attrs) - 1)
        
        # Validate merge options for edge cases
        if is_first_shell and merge_option == 'expand_previous':
            print("Warning: Cannot expand previous shell for shell_1 (no previous shell exists)")
            print("Switching to 'expand_next' option")
            merge_option = 'expand_next'
        
        if is_last_shell and merge_option == 'expand_next':
            print(f"Note: {shell_to_remove} is the last shell. Will expand bulk region.")
        
        if (is_first_shell or is_last_shell) and merge_option == 'expand_both':
            print("Warning: Cannot expand both directions for first or last shell")
            if is_first_shell:
                print("Switching to 'expand_next' option")
                merge_option = 'expand_next'
            else:
                print("Switching to 'expand_previous' option") 
                merge_option = 'expand_previous'
        
        # Create new shell structure
        new_shells_data = {}
        
        if merge_option == 'expand_previous':
            # Copy all shells except the one to remove
            for attr in shell_attrs:
                if attr != shell_to_remove:
                    if shell_attrs.index(attr) == shell_idx - 1:  # Previous shell
                        # Expand previous shell to include removed shell's end
                        if hasattr(shells, 'data'):
                            prev_start, prev_end = shells.data[attr]
                        else:
                            prev_start, prev_end = getattr(shells, attr)
                        new_bounds = (prev_start, shell_end)
                        new_shells_data[attr] = new_bounds
                        print(f"  Expanded {attr}: {prev_start:.2f} - {shell_end:.2f} Å")
                    else:
                        # Keep other shells unchanged
                        if hasattr(shells, 'data'):
                            new_shells_data[attr] = shells.data[attr]
                        else:
                            new_shells_data[attr] = getattr(shells, attr)
            
            # Keep bulk unchanged for expand_previous
            if hasattr(shells, 'data') and 'bulk' in shells.data:
                new_shells_data['bulk'] = shells.data['bulk']
            elif hasattr(shells, 'bulk'):
                new_shells_data['bulk'] = shells.bulk
        
        elif merge_option == 'expand_next':
            if is_last_shell:
                # Special case: removing last shell, expand bulk
                for attr in shell_attrs:
                    if attr != shell_to_remove:
                        # Keep all other shells unchanged
                        if hasattr(shells, 'data'):
                            new_shells_data[attr] = shells.data[attr]
                        else:
                            new_shells_data[attr] = getattr(shells, attr)
                
                # Expand bulk to start where removed shell started
                if hasattr(shells, 'data') and 'bulk' in shells.data:
                    bulk_start, bulk_end = shells.data['bulk']
                    new_shells_data['bulk'] = (shell_start, bulk_end)
                    print(f"  Expanded bulk: {shell_start:.2f} - ∞ Å")
                elif hasattr(shells, 'bulk'):
                    bulk_start, bulk_end = shells.bulk
                    new_shells_data['bulk'] = (shell_start, bulk_end)
                    print(f"  Expanded bulk: {shell_start:.2f} - ∞ Å")
            else:
                # Normal case: expand the next shell
                for attr in shell_attrs:
                    if attr != shell_to_remove:
                        if shell_attrs.index(attr) == shell_idx + 1:  # Next shell
                            # Expand next shell to include removed shell's start
                            if hasattr(shells, 'data'):
                                next_start, next_end = shells.data[attr]
                            else:
                                next_start, next_end = getattr(shells, attr)
                            new_bounds = (shell_start, next_end)
                            new_shells_data[attr] = new_bounds
                            print(f"  Expanded {attr}: {shell_start:.2f} - {next_end:.2f} Å")
                        else:
                            # Keep other shells unchanged
                            if hasattr(shells, 'data'):
                                new_shells_data[attr] = shells.data[attr]
                            else:
                                new_shells_data[attr] = getattr(shells, attr)
                
                # Keep bulk unchanged for normal expand_next
                if hasattr(shells, 'data') and 'bulk' in shells.data:
                    new_shells_data['bulk'] = shells.data['bulk']
                elif hasattr(shells, 'bulk'):
                    new_shells_data['bulk'] = shells.bulk
        
        elif merge_option == 'expand_both':
            # NEW OPTION: Expand both previous and next shells to meet in the middle
            shell_width = shell_end - shell_start
            midpoint = shell_start + shell_width / 2
            
            for attr in shell_attrs:
                if attr != shell_to_remove:
                    if shell_attrs.index(attr) == shell_idx - 1:  # Previous shell
                        # Expand previous shell to midpoint of removed shell
                        if hasattr(shells, 'data'):
                            prev_start, prev_end = shells.data[attr]
                        else:
                            prev_start, prev_end = getattr(shells, attr)
                        new_bounds = (prev_start, midpoint)
                        new_shells_data[attr] = new_bounds
                        print(f"  Expanded {attr}: {prev_start:.2f} - {midpoint:.2f} Å")
                    elif shell_attrs.index(attr) == shell_idx + 1:  # Next shell
                        # Expand next shell to midpoint of removed shell
                        if hasattr(shells, 'data'):
                            next_start, next_end = shells.data[attr]
                        else:
                            next_start, next_end = getattr(shells, attr)
                        new_bounds = (midpoint, next_end)
                        new_shells_data[attr] = new_bounds
                        print(f"  Expanded {attr}: {midpoint:.2f} - {next_end:.2f} Å")
                    else:
                        # Keep other shells unchanged
                        if hasattr(shells, 'data'):
                            new_shells_data[attr] = shells.data[attr]
                        else:
                            new_shells_data[attr] = getattr(shells, attr)
            
            # Keep bulk unchanged for expand_both
            if hasattr(shells, 'data') and 'bulk' in shells.data:
                new_shells_data['bulk'] = shells.data['bulk']
            elif hasattr(shells, 'bulk'):
                new_shells_data['bulk'] = shells.bulk
        
        # Renumber shells sequentially
        final_shells = Results()
        final_shells.data = {}
        
        # Get remaining shell attributes (excluding bulk)
        remaining_shell_attrs = [attr for attr in new_shells_data.keys() if attr.startswith('shell_')]
        remaining_shell_attrs.sort()
        
        # Renumber sequentially
        for i, old_attr in enumerate(remaining_shell_attrs, 1):
            new_attr = f'shell_{i}'
            bounds = new_shells_data[old_attr]
            final_shells.data[new_attr] = bounds
            setattr(final_shells, new_attr, bounds)
        
        # Add bulk back
        if 'bulk' in new_shells_data:
            final_shells.data['bulk'] = new_shells_data['bulk']
            setattr(final_shells, 'bulk', new_shells_data['bulk'])
        
        # Update the class attribute
        if ion_type == 'cation':
            self.cation_solvation_shells = final_shells
        else:
            self.anion_solvation_shells = final_shells
        
        # Print updated shell structure
        final_shell_attrs = [attr for attr in final_shells.data.keys() if attr.startswith('shell_')]
        final_shell_attrs.sort()
        if 'bulk' in final_shells.data:
            final_shell_attrs.append('bulk')
        
        print(f"\nUpdated {ion_type} solvation structure:")
        for attr in final_shell_attrs:
            start, end = final_shells.data[attr]
            end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
            name = attr.replace('_', ' ').title()
            print(f"  {name}: {start:.2f} - {end_str} Å")
        
        return True



    def plot_all_modified_shells(
        self,
        save_plots=True,
        plot_range=12,
        figsize_per_ion=(6, 4),
        max_cols=3,
        show_rcn=True,
        rcn_scale_factor=4.0,
        # ── Typography ────────────────────────────────────────────────────────
        font_size=10,
        font_weight='normal',
        title_font_size=12,
        title_font_weight='bold',
        show_title=True,               # Set False to hide the subplot titles
        label_font_size=None,      # axes labels; falls back to font_size
        tick_font_size=None,       # tick labels; falls back to font_size
        legend_font_size=None,     # legend text; falls back to font_size - 2
        shell_label_font_size=None,# S1/S2/Bulk labels; falls back to font_size
        shell_label_font_weight='bold',
        rcn_label_font_size=None,  # RCN y-axis label; falls back to font_size - 1
        # ── Lines ─────────────────────────────────────────────────────────────
        rdf_linewidth=2.0,
        rdf_linestyle='-',
        rdf_color='k',
        rcn_linewidth=1.5,
        rcn_linestyle='--',
        rcn_alpha=0.7,
        rcn_color=None,            # None → 'darkred' cation / 'darkblue' anion
        # ── Shaded regions ────────────────────────────────────────────────────
        shell_alpha=0.4,
        # ── Legend ────────────────────────────────────────────────────────────
        legend_loc='upper left',
        legend_bbox_to_anchor=(0.02, 0.85),  # (x, y) in axes fraction; None → ignored
        legend_frameon=False,
        legend_framealpha=0.9,
        legend_ncol=1,
        # ── Layout / output ───────────────────────────────────────────────────
        dpi=300,
        output_filename=None,      # None → auto-generated name
        save_combined=True,        # Save the combined grid figure
        save_individual=False,     # Save each ion as its own figure
    ):
        '''
        Plot solvation shells for all ion types in a grid layout after modifications.
        NOW WITH OPTIONAL RCN CURVES OVERLAID ON RDF PLOTS.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save the combined plot, default=True
        plot_range : float
            Maximum r value for plotting, default=12
        figsize_per_ion : tuple
            Figure size for each ion subplot, default=(6, 4)
        max_cols : int
            Maximum number of columns in the grid, default=3
        show_rcn : bool
            Whether to show RCN curves overlaid on RDF, default=True
        rcn_scale_factor : float
            Scale factor for RCN y-axis, default=4.0
        font_size : float
            Base font size used as fallback for all text, default=10
        font_weight : str
            Base font weight for axis labels ('normal', 'bold'), default='normal'
        title_font_size : float
            Subplot title font size, default=12
        title_font_weight : str
            Subplot title font weight, default='bold'
        show_title : bool
            Whether to display the subplot title, default=True
        label_font_size : float or None
            Axis label font size; None → font_size
        tick_font_size : float or None
            Tick label font size; None → font_size
        legend_font_size : float or None
            Legend text font size; None → font_size - 2
        shell_label_font_size : float or None
            Font size for S1/S2/Bulk region labels; None → font_size
        shell_label_font_weight : str
            Font weight for region labels, default='bold'
        rcn_label_font_size : float or None
            RCN right-axis label font size; None → font_size - 1
        rdf_linewidth : float
            g(r) curve line width, default=2.0
        rdf_linestyle : str
            g(r) curve line style, default='-'
        rdf_color : str
            g(r) curve color, default='k'
        rcn_linewidth : float
            RCN curve line width, default=1.5
        rcn_linestyle : str
            RCN curve line style, default='--'
        rcn_alpha : float
            RCN curve alpha, default=0.7
        rcn_color : str or None
            RCN curve color; None → 'darkred' (cation) / 'darkblue' (anion)
        shell_alpha : float
            Opacity of shaded shell/bulk regions, default=0.4
        legend_loc : str
            Legend anchor string, e.g. 'upper left', 'upper right', 'lower right',
            'best', etc., default='upper left'
        legend_bbox_to_anchor : tuple or None
            Fine-tune legend position as (x, y) in axes-fraction coordinates.
            Set to None to rely on legend_loc alone, default=(0.02, 0.85)
        legend_frameon : bool
            Whether to draw the legend box, default=False
        legend_framealpha : float
            Legend box opacity (only relevant when legend_frameon=True), default=0.9
        legend_ncol : int
            Number of columns in the legend, default=1
        dpi : int
            Resolution for saved figure, default=300
        output_filename : str or None
            Output filename for the combined figure; None → auto-generated.
            Individual figures are named <base>_<ion_type>.png automatically.
        save_combined : bool
            Save the combined multi-ion grid figure, default=True
        save_individual : bool
            Save each ion as a separate figure, default=False
        
        Returns
        -------
        success : bool
            True if plotting was successful
        '''
        # Resolve font-size fallbacks
        _label_fs  = label_font_size  if label_font_size  is not None else font_size
        _tick_fs   = tick_font_size   if tick_font_size   is not None else font_size
        _leg_fs    = legend_font_size if legend_font_size is not None else max(font_size - 2, 6)
        _slabel_fs = shell_label_font_size if shell_label_font_size is not None else font_size
        _rcn_lfs   = rcn_label_font_size   if rcn_label_font_size   is not None else max(font_size - 1, 6)
        
        # Check if we have ion-type-specific shells
        if not (hasattr(self, 'cation_shells_by_type') and hasattr(self, 'anion_shells_by_type')):
            print("Ion-type-specific shells not found. Run determine_ion_solvation_shells_by_type() first.")
            return False
        
        # Check if we have RDFs
        if not hasattr(self, 'rdfs') or not self.rdfs:
            print("No RDFs available. Run generate_rdfs() first.")
            return False
        
        # Collect all ions with valid shells and RDFs
        ions_to_plot = []
        
        # Check cations
        for ion_type, shells in self.cation_shells_by_type.items():
            rdf_key = f'{ion_type}-w'
            if shells is not None and rdf_key in self.rdfs and self.rdfs[rdf_key] is not None:
                ions_to_plot.append((ion_type, 'cation', shells, self.rdfs[rdf_key]))
            elif shells is None:
                print(f"Warning: No shells for {ion_type} (cation)")
            elif rdf_key not in self.rdfs or self.rdfs[rdf_key] is None:
                print(f"Warning: No RDF data for {ion_type} (cation)")
        
        # Check anions
        for ion_type, shells in self.anion_shells_by_type.items():
            rdf_key = f'{ion_type}-w'
            if shells is not None and rdf_key in self.rdfs and self.rdfs[rdf_key] is not None:
                ions_to_plot.append((ion_type, 'anion', shells, self.rdfs[rdf_key]))
            elif shells is None:
                print(f"Warning: No shells for {ion_type} (anion)")
            elif rdf_key not in self.rdfs or self.rdfs[rdf_key] is None:
                print(f"Warning: No RDF data for {ion_type} (anion)")
        
        if not ions_to_plot:
            print("No ions available for plotting.")
            return False
        
        print(f"Plotting shells for {len(ions_to_plot)} ion types:")
        for ion_type, ion_category, _, _ in ions_to_plot:
            print(f"  {ion_type} ({ion_category})")
        
        # Calculate grid dimensions
        n_ions = len(ions_to_plot)
        n_cols = min(max_cols, n_ions)
        n_rows = (n_ions + n_cols - 1) // n_cols
        
        # Calculate figure size
        fig_width = n_cols * figsize_per_ion[0]
        fig_height = n_rows * figsize_per_ion[1]
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # Handle single subplot case
        if n_ions == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each ion
        for i, (ion_type, ion_category, shells, rdf_data) in enumerate(ions_to_plot):
            ax = axes[i]
            
            # Get RDF data
            if hasattr(shells, 'rdf_r') and hasattr(shells, 'rdf_g_r'):
                r = shells.rdf_r
                g_r = shells.rdf_g_r
            else:
                r = rdf_data.bins
                g_r = rdf_data.rdf
            
            # Plot RDF on primary axis
            line_rdf = ax.plot(r, g_r, color=rdf_color, linewidth=rdf_linewidth,
                               linestyle=rdf_linestyle, label='g(r)', zorder=1)
            
            # Get shell regions and bulk
            shell_regions = {k: v for k, v in shells.data.items() if k.startswith('shell_')}
            n_shells = len(shell_regions)
            has_bulk = 'bulk' in shells.data
            
            # Generate colors (same as your existing method)
            def get_blue_saturation_colors_from_00c5ff(n_shells):
                base_rgb = mcolors.hex2color('#00c5ff')
                base_hsv = mcolors.rgb_to_hsv(base_rgb)
                base_hue = base_hsv[0]
                base_saturation = base_hsv[1]
                base_value = base_hsv[2]
                
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
            
            # Generate colors
            all_colors = get_blue_saturation_colors_from_00c5ff(n_shells)
            shell_colors = all_colors[:-1] if len(all_colors) > 1 else all_colors
            bulk_color = all_colors[-1] if len(all_colors) > 1 else all_colors[0]
            
            # Calculate label position
            rdf_max = max(g_r)
            label_y_position = rdf_max * 1.15
            
            # Plot shells
            last_shell_end = 0
            shell_items = [(k, v) for k, v in shells.data.items() if k.startswith('shell_')]
            shell_items.sort(key=lambda x: x[1][0])
            
            for j, (shell_name, (start, end)) in enumerate(shell_items):
                if j < len(shell_colors):
                    shell_color = shell_colors[j]
                else:
                    shell_color = shell_colors[-1]
                
                # Fill shell region
                ax.axvspan(start, end, alpha=shell_alpha, color=shell_color, zorder=0)
                
                # Add shell label
                mid_point = (start + end) / 2
                ax.text(mid_point, label_y_position, f'S{j+1}',
                    horizontalalignment='center',
                    fontweight=shell_label_font_weight,
                    fontsize=_slabel_fs,
                    color='black')
                
                last_shell_end = max(last_shell_end, end)
            
            # Plot bulk region
            if has_bulk:
                bulk_start, bulk_end = shells.data['bulk']
                bulk_plot_end = plot_range if np.isinf(bulk_end) else min(bulk_end, plot_range)
                
                ax.axvspan(bulk_start, bulk_plot_end, alpha=shell_alpha, color=bulk_color, zorder=0)
                
                bulk_mid_point = (bulk_start + bulk_plot_end) / 2
                ax.text(bulk_mid_point, label_y_position, 'Bulk',
                    horizontalalignment='center',
                    fontweight=shell_label_font_weight,
                    fontsize=_slabel_fs,
                    color='black')
            elif last_shell_end < plot_range:
                # Fallback bulk
                ax.axvspan(last_shell_end, plot_range, alpha=shell_alpha, color=bulk_color, zorder=0)
                bulk_mid_point = (last_shell_end + plot_range) / 2
                ax.text(bulk_mid_point, label_y_position, 'Bulk',
                    horizontalalignment='center',
                    fontweight=shell_label_font_weight,
                    fontsize=_slabel_fs,
                    color='black')
            
            # ADD RCN CURVE OVERLAY
            if show_rcn:
                # Calculate or get RCN data
                rdf_key = f'{ion_type}-w'
                
                if not hasattr(self, 'running_coordination_numbers') or rdf_key not in self.running_coordination_numbers:
                    # Calculate RCN
                    r_rcn, rcn = self.calculate_running_coordination_number(rdf_key, save_data=False)
                else:
                    rcn_data = self.running_coordination_numbers[rdf_key]
                    r_rcn = rcn_data['r']
                    rcn = rcn_data['rcn']
                
                if r_rcn is not None and rcn is not None:
                    # Create secondary y-axis for RCN
                    ax2 = ax.twinx()
                    
                    # Get coordination radius for scaling
                    r0 = None
                    if hasattr(self, 'solutes_ci') and ion_type in self.solutes_ci:
                        r0 = self.solutes_ci[ion_type].radii.get('water')
                    elif hasattr(self, 'solutes_ai') and ion_type in self.solutes_ai:
                        r0 = self.solutes_ai[ion_type].radii.get('water')
                    
                    # Smart y-axis scaling for RCN
                    if r0 is not None:
                        idx_r0 = np.argmin(np.abs(r_rcn - r0))
                        cn_at_r0 = rcn[idx_r0]
                        
                        # Scale so CN at r₀ appears at 1/rcn_scale_factor of figure height
                        y_max_rcn = cn_at_r0 * rcn_scale_factor
                        
                        ax2.set_ylim(0, y_max_rcn)
                        
                        # Mark CN value
                        ax2.axhline(cn_at_r0, color='orange', linestyle=':', linewidth=1, alpha=0.5, zorder=2)
                    else:
                        # Fallback: just use data range
                        max_rcn_in_range = rcn[r_rcn <= plot_range].max()
                        ax2.set_ylim(0, max_rcn_in_range * 1.1)
                    
                    # Plot RCN curve
                    _color_rcn = rcn_color if rcn_color is not None else (
                        'darkred' if ion_category == 'cation' else 'darkblue'
                    )
                    line_rcn = ax2.plot(r_rcn, rcn, color=_color_rcn,
                                    linewidth=rcn_linewidth,
                                    label='RCN(r)', alpha=rcn_alpha,
                                    linestyle=rcn_linestyle, zorder=3)
                    
                    # Label RCN axis
                    ax2.set_ylabel('RCN(r)', color=_color_rcn, fontsize=_rcn_lfs,
                                   fontweight=font_weight)
                    ax2.tick_params(axis='y', labelcolor=_color_rcn, labelsize=_tick_fs)
                    
                    # Combined legend
                    lines = line_rdf + line_rcn
                    labels = [l.get_label() for l in lines]
                    _leg_kw = dict(
                        loc=legend_loc,
                        fontsize=_leg_fs,
                        frameon=legend_frameon,
                        framealpha=legend_framealpha,
                        ncol=legend_ncol,
                    )
                    if legend_bbox_to_anchor is not None:
                        _leg_kw['bbox_to_anchor'] = legend_bbox_to_anchor
                    legend = ax.legend(lines, labels, **_leg_kw)
            
            # Format subplot
            ax.set_xlabel('r (Å)', fontsize=_label_fs, fontweight=font_weight)
            ax.set_ylabel('g(r)',   fontsize=_label_fs, fontweight=font_weight)
            ax.tick_params(axis='both', labelsize=_tick_fs)
            
            # Color-code the title by ion type
            title_color = 'blue' if ion_category == 'cation' else 'red'
            title_text = f'{ion_type.upper()} ({ion_category.title()})'
            if show_rcn:
                title_text += ' + RCN'
            
            if show_title:
                ax.set_title(title_text, fontweight=title_font_weight,
                             fontsize=title_font_size, color=title_color)
            
            ax.set_xlim(0, plot_range)
            ax.set_ylim(bottom=0, top=rdf_max * 1.3)
            
            # Print shell info for this ion
            print(f"\n{ion_type} ({ion_category}) shells:")
            for shell_name, (start, end) in shell_items:
                width = end - start
                print(f"  {shell_name}: {start:.2f} - {end:.2f} Å (width: {width:.2f} Å)")
            if has_bulk:
                bulk_start, bulk_end = shells.data['bulk']
                end_str = "∞" if np.isinf(bulk_end) else f"{bulk_end:.2f}"
                print(f"  bulk: {bulk_start:.2f} - {end_str} Å")
            
            if show_rcn and r0 is not None:
                print(f"  RCN at r₀={r0:.2f} Å: {cn_at_r0:.2f}")
        
        # Hide unused subplots
        for i in range(len(ions_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save individual figures (one per ion)
        if save_plots and save_individual:
            import os
            fig.canvas.draw()  # ensure renderer is available
            suffix = '_with_rcn' if show_rcn else ''
            for idx, (ion_type, ion_category, shells_item, rdf_data) in enumerate(ions_to_plot):
                ax_ind = axes[idx]
                # Build filename
                if output_filename is not None:
                    base, ext = os.path.splitext(output_filename)
                    ext = ext if ext else '.png'
                    ind_fname = f'{base}_{ion_type}{ion_category[0].upper()}{ext}'
                else:
                    ind_fname = f'{ion_type}_{ion_category}_solvation_shell{suffix}.png'
                # Crop to this axes slot, including any twin axes (e.g. RCN right axis)
                renderer = fig.canvas.get_renderer()
                ax_pos_bounds = ax_ind.get_position().bounds
                all_axes_in_slot = [a for a in fig.axes
                                    if a.get_position().bounds == ax_pos_bounds]
                from matplotlib.transforms import Bbox
                bboxes = [a.get_tightbbox(renderer) for a in all_axes_in_slot
                          if a.get_tightbbox(renderer) is not None]
                if not bboxes:
                    bboxes = [ax_ind.get_tightbbox(renderer)]
                bbox = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(ind_fname, dpi=dpi, bbox_inches=bbox)
                print(f"  Individual figure saved: {ind_fname}")
        
        # Save combined figure
        if save_plots and save_combined:
            suffix = '_with_rcn' if show_rcn else ''
            if output_filename is not None:
                filename = output_filename
            else:
                filename = f'all_modified_solvation_shells{suffix}.png'
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"\nCombined plot saved as: {filename}")
        
        plt.show()
        
        return True



    def plot_shell_comparison_summary(self, save_plots=True, plot_range=20, figsize=(12, 8)):
        '''
        Create a summary comparison of all ion types with their shell boundaries marked.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save the comparison plot, default=True
        plot_range : float
            Maximum r value for plotting, default=20
        figsize : tuple
            Figure size as (width, height) in inches, default=(12, 8)
        '''
    
        
        # Check prerequisites
        if not (hasattr(self, 'cation_shells_by_type') and hasattr(self, 'anion_shells_by_type')):
            print("Ion-type-specific shells not found.")
            return False
        
        if not hasattr(self, 'rdfs') or not self.rdfs:
            print("No RDFs available.")
            return False
        
        # Collect all valid ions
        all_ions = []
        
        for ion_type, shells in self.cation_shells_by_type.items():
            rdf_key = f'{ion_type}-w'
            if shells is not None and rdf_key in self.rdfs and self.rdfs[rdf_key] is not None:
                all_ions.append((ion_type, 'cation', shells, self.rdfs[rdf_key]))
        
        for ion_type, shells in self.anion_shells_by_type.items():
            rdf_key = f'{ion_type}-w'
            if shells is not None and rdf_key in self.rdfs and self.rdfs[rdf_key] is not None:
                all_ions.append((ion_type, 'anion', shells, self.rdfs[rdf_key]))
        
        if not all_ions:
            print("No valid ions for comparison.")
            return False
        
        # Create comparison table
        print("\n" + "="*80)
        print("SHELL BOUNDARY COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Ion':<6} {'Type':<8} {'Shell 1':<12} {'Shell 2':<12} {'Shell 3':<12} {'Bulk Start':<12}")
        print("-" * 80)
        
        for ion_type, ion_category, shells, rdf_data in all_ions:
            shell_info = {"Shell 1": "—", "Shell 2": "—", "Shell 3": "—", "Bulk": "—"}
            
            # Get shell boundaries
            shell_items = [(k, v) for k, v in shells.data.items() if k.startswith('shell_')]
            shell_items.sort(key=lambda x: x[1][0])
            
            for i, (shell_name, (start, end)) in enumerate(shell_items):
                shell_key = f"Shell {i+1}"
                if shell_key in shell_info:
                    shell_info[shell_key] = f"{start:.1f}-{end:.1f}"
            
            # Get bulk info
            if 'bulk' in shells.data:
                bulk_start, bulk_end = shells.data['bulk']
                shell_info['Bulk'] = f"{bulk_start:.1f}"
            
            print(f"{ion_type:<6} {ion_category:<8} {shell_info['Shell 1']:<12} {shell_info['Shell 2']:<12} {shell_info['Shell 3']:<12} {shell_info['Bulk']:<12}")
        
        print("="*80)
        
        # Create visual plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)  # Use the figsize parameter
        
        colors_cation = plt.cm.Blues(np.linspace(0.4, 0.9, sum(1 for ion in all_ions if ion[1] == 'cation')))
        colors_anion = plt.cm.Reds(np.linspace(0.4, 0.9, sum(1 for ion in all_ions if ion[1] == 'anion')))
        
        cation_idx = 0
        anion_idx = 0
        
        for i, (ion_type, ion_category, shells, rdf_data) in enumerate(all_ions):
            # Get RDF data
            if hasattr(shells, 'rdf_r') and hasattr(shells, 'rdf_g_r'):
                r = shells.rdf_r
                g_r = shells.rdf_g_r
            else:
                r = rdf_data.bins
                g_r = rdf_data.rdf
            
            # Choose color based on category
            if ion_category == 'cation':
                color = colors_cation[cation_idx] if len(colors_cation) > 0 else 'blue'
                cation_idx += 1
            else:
                color = colors_anion[anion_idx] if len(colors_anion) > 0 else 'red'
                anion_idx += 1
            
            # Plot RDF (offset for visibility)
            offset = i * 0.5
            ax.plot(r, g_r + offset, color=color, linewidth=1.5, 
                label=f'{ion_type} ({ion_category})', alpha=0.8)
            
            # Mark shell boundaries
            shell_items = [(k, v) for k, v in shells.data.items() if k.startswith('shell_')]
            for shell_name, (start, end) in shell_items:
                ax.axvline(start, color=color, linestyle='--', alpha=0.6, linewidth=1)
                ax.axvline(end, color=color, linestyle=':', alpha=0.6, linewidth=1)
        
        ax.set_xlabel('r (Å)', fontsize=12)
        ax.set_ylabel('g(r) (offset for clarity)', fontsize=12)
        # ax.set_title('Shell Boundary Comparison for All Ion Types', fontweight='bold', fontsize=14)
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, plot_range)
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'shell_comparison_summary.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved as: {filename}")
        
        plt.show()
        return True


    def modify_shell_boundary(self, ion_type, shell_name, boundary_type, new_value, auto_adjust=True):
        '''
        Modify a specific boundary of a shell with automatic adjustment of adjacent shells.
        
        Parameters
        ----------
        ion_type : str
            Ion type name (e.g., 'Na', 'Mg', 'Cl')
        shell_name : str
            Shell name to modify (e.g., 'shell_1', 'shell_2', 'shell_3', 'bulk')
        boundary_type : str
            'start' or 'end' boundary to modify
        new_value : float
            New boundary value in Angstroms
        auto_adjust : bool
            Whether to automatically adjust adjacent shells, default=True
        
        Returns
        -------
        success : bool
            True if modification was successful, False otherwise
        '''
        
        # Find which category this ion belongs to
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        if ion_type in cation_types:
            ion_category = 'cation'
            shells_dict = self.cation_shells_by_type
        elif ion_type in anion_types:
            ion_category = 'anion'
            shells_dict = self.anion_shells_by_type
        else:
            print(f"Ion type '{ion_type}' not found in system.")
            return False
        
        # Check if ion has shells
        if ion_type not in shells_dict or shells_dict[ion_type] is None:
            print(f"No shells found for {ion_type}")
            return False
        
        shells = shells_dict[ion_type]
        
        # Check if shell exists
        if shell_name not in shells.data:
            available_shells = list(shells.data.keys())
            print(f"Shell '{shell_name}' not found for {ion_type}.")
            print(f"Available shells: {available_shells}")
            return False
        
        # Get current boundaries
        current_start, current_end = shells.data[shell_name]
        
        # Validate new value
        if boundary_type == 'start':
            if new_value >= current_end:
                print(f"Invalid start boundary: {new_value:.2f} must be less than end boundary {current_end:.2f}")
                return False
            if new_value < 0:
                print(f"Invalid start boundary: {new_value:.2f} must be >= 0")
                return False
            increment = new_value - current_start
            print(f"Modifying {shell_name} start boundary: {current_start:.2f} -> {new_value:.2f} Å (increment: {increment:+.2f})")
            
        elif boundary_type == 'end':
            if new_value <= current_start:
                print(f"Invalid end boundary: {new_value:.2f} must be greater than start boundary {current_start:.2f}")
                return False
            increment = new_value - current_end
            print(f"Modifying {shell_name} end boundary: {current_end:.2f} -> {new_value:.2f} Å (increment: {increment:+.2f})")
            
        else:
            print("boundary_type must be 'start' or 'end'")
            return False
        
        if not auto_adjust:
            # Simple modification without adjusting adjacent shells
            if boundary_type == 'start':
                shells.data[shell_name] = (new_value, current_end)
            else:
                shells.data[shell_name] = (current_start, new_value)
            
            print(f"Modified {shell_name} without adjusting adjacent shells")
            self._print_shell_boundaries(ion_type, shells)
            return True
        
        # Get all shells sorted by position
        all_shells = [(name, bounds) for name, bounds in shells.data.items()]
        all_shells.sort(key=lambda x: x[1][0])  # Sort by start position
        
        # Find the position of the shell to modify
        shell_index = next(i for i, (name, bounds) in enumerate(all_shells) if name == shell_name)
        
        # Apply the modification with cascading adjustments
        success = self._apply_boundary_modification_with_cascade(
            shells, all_shells, shell_index, shell_name, boundary_type, new_value, increment
        )
        
        if success:
            print(f"Successfully modified {shell_name} {boundary_type} boundary with cascade adjustments")
            self._print_shell_boundaries(ion_type, shells)
        else:
            print(f"Failed to modify {shell_name} {boundary_type} boundary")
        
        return success

    def _apply_boundary_modification_with_cascade(self, shells, all_shells, shell_index, shell_name, boundary_type, new_value, increment):
        '''Apply boundary modification with cascading adjustments to adjacent shells'''
        
        # Create a working copy of shell data
        new_shell_data = shells.data.copy()
        
        if boundary_type == 'start':
            # Modifying start boundary
            current_start, current_end = shells.data[shell_name]
            new_shell_data[shell_name] = (new_value, current_end)
            
            # If we have a previous shell, adjust its end boundary
            if shell_index > 0:
                prev_shell_name, (prev_start, prev_end) = all_shells[shell_index - 1]
                new_prev_end = prev_end + increment
                
                # Validate the adjustment
                if new_prev_end <= prev_start:
                    print(f"Cannot adjust {prev_shell_name}: end boundary would be <= start boundary")
                    return False
                
                new_shell_data[prev_shell_name] = (prev_start, new_prev_end)
                print(f"  Adjusted {prev_shell_name} end boundary: {prev_end:.2f} -> {new_prev_end:.2f} Å")
                
                # Cascade further backwards if needed
                if not self._cascade_boundary_changes(new_shell_data, all_shells, shell_index - 1, 'end', increment):
                    return False
        
        elif boundary_type == 'end':
            # Modifying end boundary
            current_start, current_end = shells.data[shell_name]
            
            # Handle infinity case for bulk
            if np.isinf(current_end) and shell_name == 'bulk':
                # For bulk region, we can't change infinity, so just update the start
                new_shell_data[shell_name] = (current_start, np.inf)
                print(f"  Bulk region end boundary remains at infinity")
            else:
                new_shell_data[shell_name] = (current_start, new_value)
            
            # If we have a next shell, adjust its start boundary
            if shell_index < len(all_shells) - 1:
                next_shell_name, (next_start, next_end) = all_shells[shell_index + 1]
                new_next_start = next_start + increment
                
                # Validate the adjustment
                if new_next_start >= next_end and not np.isinf(next_end):
                    print(f"Cannot adjust {next_shell_name}: start boundary would be >= end boundary")
                    return False
                if new_next_start < 0:
                    print(f"Cannot adjust {next_shell_name}: start boundary would be negative")
                    return False
                
                new_shell_data[next_shell_name] = (new_next_start, next_end)
                print(f"  Adjusted {next_shell_name} start boundary: {next_start:.2f} -> {new_next_start:.2f} Å")
                
                # Cascade further forwards if needed
                if not self._cascade_boundary_changes(new_shell_data, all_shells, shell_index + 1, 'start', increment):
                    return False
        
        # Apply all changes
        shells.data = new_shell_data
        
        return True

    def _cascade_boundary_changes(self, new_shell_data, all_shells, start_index, boundary_type, increment):
        '''Recursively cascade boundary changes to maintain shell consistency'''
        
        if boundary_type == 'end' and start_index > 0:
            # Cascading backwards (adjusting end boundaries)
            shell_name, (shell_start, shell_end) = all_shells[start_index]
            prev_shell_name, (prev_start, prev_end) = all_shells[start_index - 1]
            
            # Check if the current shell's start was already adjusted
            current_shell_start = new_shell_data[shell_name][0]
            if current_shell_start != shell_start:
                # This shell was already adjusted, need to cascade to previous
                prev_adjustment = current_shell_start - shell_start
                new_prev_end = prev_end + prev_adjustment
                
                if new_prev_end <= prev_start:
                    print(f"Cascade failed: {prev_shell_name} end boundary would be <= start boundary")
                    return False
                
                new_shell_data[prev_shell_name] = (prev_start, new_prev_end)
                print(f"  Cascaded {prev_shell_name} end boundary: {prev_end:.2f} -> {new_prev_end:.2f} Å")
                
                # Continue cascading
                return self._cascade_boundary_changes(new_shell_data, all_shells, start_index - 1, 'end', prev_adjustment)
        
        elif boundary_type == 'start' and start_index < len(all_shells) - 1:
            # Cascading forwards (adjusting start boundaries)
            shell_name, (shell_start, shell_end) = all_shells[start_index]
            next_shell_name, (next_start, next_end) = all_shells[start_index + 1]
            
            # Check if the current shell's end was already adjusted
            current_shell_end = new_shell_data[shell_name][1]
            if current_shell_end != shell_end and not np.isinf(shell_end):
                # This shell was already adjusted, need to cascade to next
                next_adjustment = current_shell_end - shell_end
                new_next_start = next_start + next_adjustment
                
                if new_next_start >= next_end and not np.isinf(next_end):
                    print(f"Cascade failed: {next_shell_name} start boundary would be >= end boundary")
                    return False
                if new_next_start < 0:
                    print(f"Cascade failed: {next_shell_name} start boundary would be negative")
                    return False
                
                new_shell_data[next_shell_name] = (new_next_start, next_end)
                print(f"  Cascaded {next_shell_name} start boundary: {next_start:.2f} -> {new_next_start:.2f} Å")
                
                # Continue cascading
                return self._cascade_boundary_changes(new_shell_data, all_shells, start_index + 1, 'start', next_adjustment)
        
        return True

    def _print_shell_boundaries(self, ion_type, shells):
        '''Print current shell boundaries in a nice format'''
        
        print(f"\nCurrent {ion_type} shell boundaries:")
        print("-" * 40)
        
        # Sort shells by start position
        shell_items = [(name, bounds) for name, bounds in shells.data.items()]
        shell_items.sort(key=lambda x: x[1][0])
        
        for shell_name, (start, end) in shell_items:
            if np.isinf(end):
                end_str = "∞"
            else:
                end_str = f"{end:.2f}"
            
            width = end - start if not np.isinf(end) else "∞"
            width_str = f"{width:.2f}" if width != "∞" else "∞"
            
            display_name = shell_name.replace('_', ' ').title()
            print(f"  {display_name:12s}: {start:6.2f} - {end_str:>6s} Å  (width: {width_str:>6s} Å)")

    def expand_shell_region(self, ion_type, shell_name, direction='both', amount=1.0):
        '''
        Expand a shell region by a specific amount in one or both directions.
        Automatically adjusts adjacent shells to maintain consistency.
        
        Parameters
        ----------
        ion_type : str
            Ion type name (e.g., 'Na', 'Mg', 'Cl')
        shell_name : str
            Shell name to expand (e.g., 'shell_1', 'shell_2', 'shell_3')
        direction : str
            Direction to expand: 'inward' (decrease start), 'outward' (increase end), 'both'
        amount : float
            Amount to expand in Angstroms
        
        Returns
        -------
        success : bool
            True if expansion was successful, False otherwise
        '''
        
        if direction == 'inward':
            # Expand inward by decreasing start boundary
            shells = self._get_shells_for_ion(ion_type)
            if shells is None:
                return False
            
            current_start, current_end = shells.data[shell_name]
            new_start = max(0, current_start - amount)
            
            return self.modify_shell_boundary(ion_type, shell_name, 'start', new_start, auto_adjust=True)
        
        elif direction == 'outward':
            # Expand outward by increasing end boundary
            shells = self._get_shells_for_ion(ion_type)
            if shells is None:
                return False
            
            current_start, current_end = shells.data[shell_name]
            if np.isinf(current_end):
                print(f"Cannot expand {shell_name}: end boundary is already infinite")
                return False
            
            new_end = current_end + amount
            
            return self.modify_shell_boundary(ion_type, shell_name, 'end', new_end, auto_adjust=True)
        
        elif direction == 'both':
            # Expand in both directions
            shells = self._get_shells_for_ion(ion_type)
            if shells is None:
                return False
            
            current_start, current_end = shells.data[shell_name]
            new_start = max(0, current_start - amount/2)
            
            # First expand inward
            success1 = self.modify_shell_boundary(ion_type, shell_name, 'start', new_start, auto_adjust=True)
            
            if success1 and not np.isinf(current_end):
                # Then expand outward
                shells = self._get_shells_for_ion(ion_type)  # Refresh shells object
                current_start, current_end = shells.data[shell_name]
                new_end = current_end + amount/2
                success2 = self.modify_shell_boundary(ion_type, shell_name, 'end', new_end, auto_adjust=True)
                return success2
            
            return success1
        
        else:
            print("direction must be 'inward', 'outward', or 'both'")
            return False

    def contract_shell_region(self, ion_type, shell_name, direction='both', amount=1.0):
        '''
        Contract a shell region by a specific amount in one or both directions.
        Automatically adjusts adjacent shells to maintain consistency.
        
        Parameters
        ----------
        ion_type : str
            Ion type name (e.g., 'Na', 'Mg', 'Cl')
        shell_name : str
            Shell name to contract (e.g., 'shell_1', 'shell_2', 'shell_3')
        direction : str
            Direction to contract: 'inward' (increase start), 'outward' (decrease end), 'both'
        amount : float
            Amount to contract in Angstroms
        
        Returns
        -------
        success : bool
            True if contraction was successful, False otherwise
        '''
        
        if direction == 'inward':
            # Contract inward by increasing start boundary
            shells = self._get_shells_for_ion(ion_type)
            if shells is None:
                return False
            
            current_start, current_end = shells.data[shell_name]
            new_start = current_start + amount
            
            if new_start >= current_end:
                print(f"Cannot contract {shell_name}: would make start >= end")
                return False
            
            return self.modify_shell_boundary(ion_type, shell_name, 'start', new_start, auto_adjust=True)
        
        elif direction == 'outward':
            # Contract outward by decreasing end boundary
            shells = self._get_shells_for_ion(ion_type)
            if shells is None:
                return False
            
            current_start, current_end = shells.data[shell_name]
            if np.isinf(current_end):
                print(f"Cannot contract {shell_name}: end boundary is infinite")
                return False
            
            new_end = current_end - amount
            
            if new_end <= current_start:
                print(f"Cannot contract {shell_name}: would make end <= start")
                return False
            
            return self.modify_shell_boundary(ion_type, shell_name, 'end', new_end, auto_adjust=True)
        
        elif direction == 'both':
            # Contract in both directions
            shells = self._get_shells_for_ion(ion_type)
            if shells is None:
                return False
            
            current_start, current_end = shells.data[shell_name]
            current_width = current_end - current_start if not np.isinf(current_end) else float('inf')
            
            if current_width <= amount:
                print(f"Cannot contract {shell_name}: contraction amount ({amount:.2f}) >= current width ({current_width:.2f})")
                return False
            
            new_start = current_start + amount/2
            
            # First contract inward
            success1 = self.modify_shell_boundary(ion_type, shell_name, 'start', new_start, auto_adjust=True)
            
            if success1 and not np.isinf(current_end):
                # Then contract outward
                shells = self._get_shells_for_ion(ion_type)  # Refresh shells object
                current_start, current_end = shells.data[shell_name]
                new_end = current_end - amount/2
                success2 = self.modify_shell_boundary(ion_type, shell_name, 'end', new_end, auto_adjust=True)
                return success2
            
            return success1
        
        else:
            print("direction must be 'inward', 'outward', or 'both'")
            return False

    def _get_shells_for_ion(self, ion_type):
        '''Helper method to get shells object for an ion type'''
        
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        if ion_type in cation_types:
            shells_dict = self.cation_shells_by_type
        elif ion_type in anion_types:
            shells_dict = self.anion_shells_by_type
        else:
            print(f"Ion type '{ion_type}' not found in system.")
            return None
        
        if ion_type not in shells_dict or shells_dict[ion_type] is None:
            print(f"No shells found for {ion_type}")
            return None
        
        return shells_dict[ion_type]

    def set_shell_boundaries_manually(self, ion_type, shell_boundaries):
        '''
        Manually set all shell boundaries for an ion type.
        
        Parameters
        ----------
        ion_type : str
            Ion type name (e.g., 'Na', 'Mg', 'Cl')
        shell_boundaries : dict or list
            Shell boundaries as dict {'shell_1': (start, end), 'shell_2': (start, end), ...}
            or as list of tuples [(start1, end1), (start2, end2), ...]
        
        Returns
        -------
        success : bool
            True if boundaries were set successfully
        '''
        
        shells = self._get_shells_for_ion(ion_type)
        if shells is None:
            return False
        
        # Convert list to dict if needed
        if isinstance(shell_boundaries, list):
            new_boundaries = {}
            for i, (start, end) in enumerate(shell_boundaries):
                new_boundaries[f'shell_{i+1}'] = (start, end)
            shell_boundaries = new_boundaries
        
        # Validate boundaries
        boundary_list = []
        for shell_name, (start, end) in shell_boundaries.items():
            if start >= end and not np.isinf(end):
                print(f"Invalid boundaries for {shell_name}: start ({start:.2f}) >= end ({end:.2f})")
                return False
            if start < 0:
                print(f"Invalid start boundary for {shell_name}: {start:.2f} < 0")
                return False
            boundary_list.append((start, end, shell_name))
        
        # Sort by start position to check for overlaps
        boundary_list.sort()
        
        for i in range(len(boundary_list) - 1):
            current_start, current_end, current_name = boundary_list[i]
            next_start, next_end, next_name = boundary_list[i + 1]
            
            if current_end > next_start and not np.isinf(current_end):
                print(f"Overlapping boundaries: {current_name} ends at {current_end:.2f} but {next_name} starts at {next_start:.2f}")
                return False
        
        # Apply new boundaries
        new_shell_data = shells.data.copy()
        
        # Remove existing shell boundaries
        keys_to_remove = [k for k in new_shell_data.keys() if k.startswith('shell_')]
        for key in keys_to_remove:
            del new_shell_data[key]
        
        # Add new boundaries
        for shell_name, (start, end) in shell_boundaries.items():
            new_shell_data[shell_name] = (start, end)
        
        # Add bulk region if the last shell doesn't go to infinity
        boundary_list_sorted = sorted(boundary_list)
        if boundary_list_sorted:
            last_start, last_end, last_name = boundary_list_sorted[-1]
            if not np.isinf(last_end):
                new_shell_data['bulk'] = (last_end, np.inf)
                print(f"Added bulk region: {last_end:.2f} - ∞ Å")
        
        shells.data = new_shell_data
        
        print(f"Successfully set manual boundaries for {ion_type}")
        self._print_shell_boundaries(ion_type, shells)
        
        return True

    def interactive_ion_pairing_editor(self, ion_type):
        '''
        Interactive editor for fine-tuning ion pairing cutoffs (CIP, SIP, DSIP, FI).
        
        Parameters
        ----------
        ion_type : str
            Ion type to edit (e.g., 'Na', 'Mg', 'Cl', 'cation', 'anion')
        '''
        
        # Check if ion pairing cutoffs exist
        if not hasattr(self, 'ion_pairs_by_type') or ion_type not in self.ion_pairs_by_type:
            print(f"No ion pairing cutoffs found for {ion_type}.")
            print("Run determine_ion_pairing_cutoffs() first.")
            return
        
        pairing_data = self.ion_pairs_by_type[ion_type]
        ion_pairs = pairing_data['ion_pairs']
        rdf_key = pairing_data['rdf_key']
        
        print(f"\n=== INTERACTIVE ION PAIRING EDITOR FOR {ion_type.upper()} ===")
        print("Commands:")
        print("  list                          - Show current ion pairing regions")
        print("  modify <region> <start|end> <value> - Modify boundary (e.g., 'modify CIP end 3.5')")
        print("  expand <region> <amount>      - Expand region (e.g., 'expand SIP 0.5')")
        print("  contract <region> <amount>    - Contract region (e.g., 'contract CIP 0.2')")
        print("  replot                        - Replot with current boundaries")
        print("  reset                         - Reset to original boundaries")
        print("  quit                          - Exit editor")
        print()
        
        # Store original boundaries for reset
        original_boundaries = {}
        for region, bounds in ion_pairs.items():
            original_boundaries[region] = bounds
        
        while True:
            # Show current boundaries
            self._print_ion_pairing_boundaries(ion_type, ion_pairs)
            
            command = input(f"\n[{ion_type} pairing] > ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'list':
                continue  # Already printed above
            elif command == 'replot':
                self.replot_ion_pairing_after_modification(ion_type)
            elif command == 'reset':
                # Reset to original boundaries
                for region, bounds in original_boundaries.items():
                    ion_pairs[region] = bounds
                print("Reset to original ion pairing boundaries")
            elif command.startswith('modify '):
                parts = command.split()
                if len(parts) == 4:
                    _, region_name, boundary_type, value = parts
                    region_name = region_name.upper()
                    try:
                        new_value = float(value)
                        success = self.modify_ion_pairing_boundary(ion_type, region_name, boundary_type, new_value)
                        if not success:
                            print("Modification failed. Check your input.")
                    except ValueError:
                        print("Invalid value. Use a number.")
                else:
                    print("Usage: modify <region> <start|end> <value>")
                    print("Regions: CIP, SIP, DSIP, FI")
            elif command.startswith('expand '):
                parts = command.split()
                if len(parts) == 3:
                    _, region_name, amount = parts
                    region_name = region_name.upper()
                    try:
                        amount_val = float(amount)
                        success = self.expand_ion_pairing_region(ion_type, region_name, amount_val)
                        if not success:
                            print("Expansion failed. Check your input.")
                    except ValueError:
                        print("Invalid amount. Use a number.")
                else:
                    print("Usage: expand <region> <amount>")
                    print("Regions: CIP, SIP, DSIP, FI")
            elif command.startswith('contract '):
                parts = command.split()
                if len(parts) == 3:
                    _, region_name, amount = parts
                    region_name = region_name.upper()
                    try:
                        amount_val = float(amount)
                        success = self.contract_ion_pairing_region(ion_type, region_name, amount_val)
                        if not success:
                            print("Contraction failed. Check your input.")
                    except ValueError:
                        print("Invalid amount. Use a number.")
                else:
                    print("Usage: contract <region> <amount>")
                    print("Regions: CIP, SIP, DSIP, FI")
            elif command == 'help':
                print("Commands: list, modify, expand, contract, replot, reset, quit")
            else:
                print("Unknown command. Type 'help' for available commands.")
        
        print("Ion pairing editor closed")

    def _print_ion_pairing_boundaries(self, ion_type, ion_pairs):
        '''Print current ion pairing boundaries in a nice format'''
        
        print(f"\nCurrent {ion_type} ion pairing regions:")
        print("-" * 50)
        
        # Order regions logically
        region_order = ['CIP', 'SIP', 'DSIP', 'FI']
        
        for region in region_order:
            if region in ion_pairs:
                start, end = ion_pairs[region]
                if np.isinf(end):
                    end_str = "∞"
                    width_str = "∞"
                else:
                    end_str = f"{end:.2f}"
                    width_str = f"{end - start:.2f}"
                
                # Full names for clarity
                full_names = {
                    'CIP': 'Contact Ion Pair',
                    'SIP': 'Solvent-separated',
                    'DSIP': 'Double Solvent-separated',
                    'FI': 'Free Ions'
                }
                
                full_name = full_names.get(region, region)
                print(f"  {region:4s} ({full_name:20s}): {start:6.2f} - {end_str:>6s} Å  (width: {width_str:>6s} Å)")

    def modify_ion_pairing_boundary(self, ion_type, region_name, boundary_type, new_value):
        '''
        Modify a specific boundary of an ion pairing region.
        
        Parameters
        ----------
        ion_type : str
            Ion type name (e.g., 'Na', 'Mg', 'Cl', 'cation', 'anion')
        region_name : str
            Region name to modify ('CIP', 'SIP', 'DSIP', 'FI')
        boundary_type : str
            'start' or 'end' boundary to modify
        new_value : float
            New boundary value in Angstroms
        
        Returns
        -------
        success : bool
            True if modification was successful
        '''
        
        if not hasattr(self, 'ion_pairs_by_type') or ion_type not in self.ion_pairs_by_type:
            print(f"No ion pairing data for {ion_type}")
            return False
        
        ion_pairs = self.ion_pairs_by_type[ion_type]['ion_pairs']
        
        if region_name not in ion_pairs:
            available_regions = list(ion_pairs.keys())
            print(f"Region '{region_name}' not found. Available regions: {available_regions}")
            return False
        
        current_start, current_end = ion_pairs[region_name]
        
        # Validate new value
        if boundary_type == 'start':
            if new_value >= current_end and not np.isinf(current_end):
                print(f"Invalid start boundary: {new_value:.2f} must be less than end boundary {current_end:.2f}")
                return False
            if new_value < 0:
                print(f"Invalid start boundary: {new_value:.2f} must be >= 0")
                return False
            
            print(f"Modifying {region_name} start boundary: {current_start:.2f} -> {new_value:.2f} Å")
            
            # Update the boundary
            ion_pairs[region_name] = (new_value, current_end)
            
            # Adjust adjacent regions to maintain consistency
            self._adjust_adjacent_pairing_regions(ion_pairs, region_name, 'start', new_value - current_start)
            
        elif boundary_type == 'end':
            if not np.isinf(new_value) and new_value <= current_start:
                print(f"Invalid end boundary: {new_value:.2f} must be greater than start boundary {current_start:.2f}")
                return False
            
            if region_name == 'FI' and not np.isinf(new_value):
                print("Warning: FI (Free Ions) region typically extends to infinity")
            
            print(f"Modifying {region_name} end boundary: {current_end:.2f} -> {new_value:.2f} Å")
            
            # Update the boundary
            ion_pairs[region_name] = (current_start, new_value)
            
            # Adjust adjacent regions to maintain consistency
            if not np.isinf(new_value):
                self._adjust_adjacent_pairing_regions(ion_pairs, region_name, 'end', new_value - current_end)
            
        else:
            print("boundary_type must be 'start' or 'end'")
            return False
        
        return True

    def _adjust_adjacent_pairing_regions(self, ion_pairs, modified_region, boundary_type, change):
        '''Adjust adjacent ion pairing regions to maintain consistency'''
        
        region_order = ['CIP', 'SIP', 'DSIP', 'FI']
        
        if modified_region not in region_order:
            return
        
        modified_idx = region_order.index(modified_region)
        
        if boundary_type == 'start' and modified_idx > 0:
            # Adjust the previous region's end boundary
            prev_region = region_order[modified_idx - 1]
            if prev_region in ion_pairs:
                prev_start, prev_end = ion_pairs[prev_region]
                new_prev_end = prev_end + change
                
                if new_prev_end > prev_start and new_prev_end >= 0:
                    ion_pairs[prev_region] = (prev_start, new_prev_end)
                    print(f"  Adjusted {prev_region} end boundary: {prev_end:.2f} -> {new_prev_end:.2f} Å")
        
        elif boundary_type == 'end' and modified_idx < len(region_order) - 1:
            # Adjust the next region's start boundary
            next_region = region_order[modified_idx + 1]
            if next_region in ion_pairs:
                next_start, next_end = ion_pairs[next_region]
                new_next_start = next_start + change
                
                if (new_next_start < next_end or np.isinf(next_end)) and new_next_start >= 0:
                    ion_pairs[next_region] = (new_next_start, next_end)
                    print(f"  Adjusted {next_region} start boundary: {next_start:.2f} -> {new_next_start:.2f} Å")

    def expand_ion_pairing_region(self, ion_type, region_name, amount):
        '''
        Expand an ion pairing region by a specific amount.
        
        Parameters
        ----------
        ion_type : str
            Ion type name
        region_name : str
            Region name ('CIP', 'SIP', 'DSIP', 'FI')
        amount : float
            Amount to expand in Angstroms (expands both directions)
        
        Returns
        -------
        success : bool
            True if expansion was successful
        '''
        
        if not hasattr(self, 'ion_pairs_by_type') or ion_type not in self.ion_pairs_by_type:
            print(f"No ion pairing data for {ion_type}")
            return False
        
        ion_pairs = self.ion_pairs_by_type[ion_type]['ion_pairs']
        
        if region_name not in ion_pairs:
            available_regions = list(ion_pairs.keys())
            print(f"Region '{region_name}' not found. Available regions: {available_regions}")
            return False
        
        current_start, current_end = ion_pairs[region_name]
        
        # Expand both directions by half the amount
        half_amount = amount / 2
        new_start = max(0, current_start - half_amount)
        
        if np.isinf(current_end):
            new_end = np.inf
            print(f"Expanding {region_name}: {current_start:.2f} - ∞ -> {new_start:.2f} - ∞ Å")
        else:
            new_end = current_end + half_amount
            print(f"Expanding {region_name}: {current_start:.2f} - {current_end:.2f} -> {new_start:.2f} - {new_end:.2f} Å")
        
        # Apply changes
        ion_pairs[region_name] = (new_start, new_end)
        
        # Adjust adjacent regions
        if new_start != current_start:
            self._adjust_adjacent_pairing_regions(ion_pairs, region_name, 'start', new_start - current_start)
        if not np.isinf(new_end) and new_end != current_end:
            self._adjust_adjacent_pairing_regions(ion_pairs, region_name, 'end', new_end - current_end)
        
        return True

    def contract_ion_pairing_region(self, ion_type, region_name, amount):
        '''
        Contract an ion pairing region by a specific amount.
        
        Parameters
        ----------
        ion_type : str
            Ion type name
        region_name : str
            Region name ('CIP', 'SIP', 'DSIP', 'FI')
        amount : float
            Amount to contract in Angstroms (contracts both directions)
        
        Returns
        -------
        success : bool
            True if contraction was successful
        '''
        
        if not hasattr(self, 'ion_pairs_by_type') or ion_type not in self.ion_pairs_by_type:
            print(f"No ion pairing data for {ion_type}")
            return False
        
        ion_pairs = self.ion_pairs_by_type[ion_type]['ion_pairs']
        
        if region_name not in ion_pairs:
            available_regions = list(ion_pairs.keys())
            print(f"Region '{region_name}' not found. Available regions: {available_regions}")
            return False
        
        current_start, current_end = ion_pairs[region_name]
        
        # Contract both directions by half the amount
        half_amount = amount / 2
        new_start = current_start + half_amount
        
        if np.isinf(current_end):
            new_end = np.inf
            if region_name == 'FI':
                print("Note: Cannot contract the end of FI region (extends to infinity)")
            print(f"Contracting {region_name}: {current_start:.2f} - ∞ -> {new_start:.2f} - ∞ Å")
        else:
            new_end = current_end - half_amount
            
            if new_end <= new_start:
                print(f"Cannot contract {region_name}: would make region too small or negative")
                return False
            
            print(f"Contracting {region_name}: {current_start:.2f} - {current_end:.2f} -> {new_start:.2f} - {new_end:.2f} Å")
        
        # Apply changes
        ion_pairs[region_name] = (new_start, new_end)
        
        # Adjust adjacent regions
        self._adjust_adjacent_pairing_regions(ion_pairs, region_name, 'start', new_start - current_start)
        if not np.isinf(new_end) and new_end != current_end:
            self._adjust_adjacent_pairing_regions(ion_pairs, region_name, 'end', new_end - current_end)
        
        return True


    def replot_shell_region_coordination_probabilities(self, save_plots=True):
        '''
        Replot shell region coordination probabilities from cached data without recalculating.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save the plots, default=True
        '''
        
        if not hasattr(self, 'shell_region_coordination_probabilities') or not self.shell_region_coordination_probabilities:
            print("No shell region coordination probabilities in memory.")
            print("Try loading from cache first:")
            print("  eq_opt.load_shell_region_coordination_probabilities_from_file()")
            return False
        
        print("Replotting shell region coordination probabilities from cached data...")
        self._plot_shell_region_coordination_probabilities(
            self.shell_region_coordination_probabilities, 
            save_plots
        )
        
        return True

    def replot_ion_pairing_after_modification(
        self,
        ion_type,
        plot_range=None,
        save_plots=True,
        use_extended_rdf=True,
        # --- layout ---
        figsize=(10, 6),
        # --- typography (base) ---
        font_size=12,
        font_weight='normal',
        # --- title ---
        show_title=True,
        title_font_size=14,
        title_font_weight='bold',
        title=None,                 # None → auto-generated title
        # --- axis labels ---
        xlabel=None,                # None → r'r ($\mathrm{\AA}$)'
        ylabel=None,                # None → 'g(r)'
        label_font_size=None,       # None → falls back to font_size
        label_font_weight=None,     # None → falls back to font_weight
        # --- ticks ---
        tick_font_size=None,        # None → falls back to font_size
        tick_font_weight=None,      # None → falls back to font_weight
        # --- RDF line ---
        rdf_color='k',
        rdf_linewidth=2,
        # --- region fills ---
        region_colors=None,         # list | dict {region: color} | None → defaults
        region_alpha=0.4,
        region_label_font_size=None,  # None → falls back to font_size
        region_label_font_weight='bold',
        show_region_labels=True,        # show CIP/SIP/DSIP/FI text labels on fills
        # --- grid ---
        show_grid=False,
        grid_alpha=0.3,
        # --- legend ---
        show_legend=True,
        legend_show_rdf=True,       # include g(r) line entry in legend
        legend_patch_edgecolor='black',  # border color on region patches; None → no border
        legend_font_size=None,      # None → falls back to font_size
        legend_font_weight=None,    # None → falls back to font_weight
        legend_bbox_to_anchor=None,
        legend_frame_alpha=None,
        # --- output ---
        dpi=300,
        output_filename=None,       # None → 'ion_pairing_cutoffs_{ion_type}_modified.png'
        transparent=False,
    ):
        '''
        Replot ion pairing analysis after modifications.

        Parameters
        ----------
        ion_type : str
            Ion type name.
        plot_range : float or None
            Plot range in Ångströms. None → 12.
        save_plots : bool
            Whether to save the plot, default=True.
        use_extended_rdf : bool
            Whether to use extended RDF data, default=True.
        figsize : tuple
            Figure size (width, height), default=(10, 6).
        font_size : float
            Base font size used as fallback when specific sizes are None, default=12.
        font_weight : str
            Base font weight used as fallback, default='normal'.
        show_title : bool
            Whether to show the figure title, default=True.
        title_font_size : float
            Title font size, default=14.
        title_font_weight : str
            Title font weight, default='bold'.
        title : str or None
            Custom title string. None → auto-generated from ion_type.
        xlabel : str or None
            X-axis label. None → r'r ($\\mathrm{\\AA}$)'.
        ylabel : str or None
            Y-axis label. None → 'g(r)'.
        label_font_size : float or None
            Axis label font size; None falls back to font_size.
        label_font_weight : str or None
            Axis label font weight; None falls back to font_weight.
        tick_font_size : float or None
            Tick label font size; None falls back to font_size.
        tick_font_weight : str or None
            Tick label font weight; None falls back to font_weight.
        rdf_color : str
            Color of the g(r) line, default='k'.
        rdf_linewidth : float
            Line width of the g(r) line, default=2.
        region_colors : list | dict | None
            Fill colors for CIP/SIP/DSIP/FI regions.
            - list: indexed by region order.
            - dict: ``{'CIP': 'lightcoral', ...}``.
            - None → ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'].
        region_alpha : float
            Transparency of region fills, default=0.4.
        region_label_font_size : float or None
            Font size of CIP/SIP/DSIP/FI text labels inside fills;
            None falls back to font_size.
        region_label_font_weight : str
            Font weight of region text labels, default='bold'.
        show_region_labels : bool
            Whether to draw the CIP/SIP/DSIP/FI text labels on top of the
            fill regions, default=True.
        show_grid : bool
            Whether to show a horizontal grid, default=False.
        grid_alpha : float
            Grid alpha, default=0.3.
        show_legend : bool
            Whether to show the legend, default=True.
        legend_show_rdf : bool
            Whether to include the g(r) line entry in the legend, default=True.
            Set to False to show only the region color patches (CIP, SIP, etc.).
        legend_patch_edgecolor : str or None
            Border color drawn around each region patch handle in the legend.
            Default='black' — makes light-colored patches (e.g. FI/lightyellow)
            visible even when the legend lands on a similar background.
            Set to None to disable borders.
        legend_font_size : float or None
            Legend font size; None falls back to font_size.
        legend_font_weight : str or None
            Legend font weight; None falls back to font_weight.
        legend_bbox_to_anchor : tuple or None
            Legend anchor, e.g. (1.0, 1.0); None → matplotlib default.
        legend_frame_alpha : float or None
            Legend frame alpha; None → matplotlib default.
        dpi : int
            Resolution for saved figure, default=300.
        output_filename : str or None
            Override output filename; None → auto-name.
        transparent : bool
            Transparent background when saving, default=False.
        '''

        if plot_range is None:
            plot_range = 12

        if not hasattr(self, 'ion_pairs_by_type') or ion_type not in self.ion_pairs_by_type:
            print(f"No ion pairing data for {ion_type}")
            return

        pairing_data = self.ion_pairs_by_type[ion_type]
        ion_pairs = pairing_data['ion_pairs']
        rdf_key = pairing_data['rdf_key']
        ion_category = pairing_data['ion_category']

        if not hasattr(self, 'rdfs') or rdf_key not in self.rdfs:
            print(f"No RDF data available for {rdf_key}")
            return

        rdf_data = self.rdfs[rdf_key]
        r = rdf_data.bins
        rdf = rdf_data.rdf

        # Resolve font fallbacks
        _lbl    = label_font_size        if label_font_size        is not None else font_size
        _lbl_w  = label_font_weight      if label_font_weight      is not None else font_weight
        _tick   = tick_font_size         if tick_font_size         is not None else font_size
        _tick_w = tick_font_weight       if tick_font_weight       is not None else font_weight
        _rlbl   = region_label_font_size if region_label_font_size is not None else font_size
        _leg    = legend_font_size       if legend_font_size       is not None else font_size
        _leg_w  = legend_font_weight     if legend_font_weight     is not None else font_weight

        # Default region colors
        _default_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
        pair_types = ['CIP', 'SIP', 'DSIP', 'FI']

        def _region_color(region, idx):
            if region_colors is None:
                return _default_colors[idx % len(_default_colors)]
            if isinstance(region_colors, dict):
                return region_colors.get(region, _default_colors[idx % len(_default_colors)])
            return region_colors[idx % len(region_colors)]

        print(f"Replotting {ion_type} ion pairing with modified boundaries...")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(r, rdf, color=rdf_color, linewidth=rdf_linewidth,
                label='g(r)' if legend_show_rdf else '_nolegend_')

        y_min = 0
        y_max = np.max(rdf) * 1.1
        text_y_pos = y_max * 0.95

        le = max(2, r.min())
        for i, region in enumerate(pair_types):
            if region in ion_pairs:
                start, end = ion_pairs[region]
                end_plot = min(end, plot_range) if not np.isinf(end) else plot_range
                ax.fill_betweenx(
                    np.linspace(y_min, y_max), max(le, start), end_plot,
                    alpha=region_alpha, color=_region_color(region, i), label=region,
                )
                if show_region_labels:
                    ax.text(
                        (max(le, start) + end_plot) / 2, text_y_pos, region,
                        ha='center', fontweight=region_label_font_weight, fontsize=_rlbl,
                    )
                le = end_plot

        ax.set_xlabel(xlabel if xlabel is not None else r'r ($\mathrm{\AA}$)',
                      fontsize=_lbl, fontweight=_lbl_w)
        ax.set_ylabel(ylabel if ylabel is not None else 'g(r)',
                      fontsize=_lbl, fontweight=_lbl_w)
        ax.set_xlim(2, plot_range)
        ax.set_ylim(y_min, y_max)

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(_tick)
            lbl.set_fontweight(_tick_w)

        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        if show_title:
            if title is not None:
                _title = title
            elif ion_type in ['cation', 'anion']:
                _title = f'{ion_type.title()}-{("Anion" if ion_type == "cation" else "Cation")} Ion Pairing (Modified)'
            else:
                _title = f'{ion_type} Ion Pairing Analysis (Modified)'
            ax.set_title(_title, fontsize=title_font_size, fontweight=title_font_weight)

        if show_legend:
            _leg_kwargs = dict(fontsize=_leg)
            if legend_bbox_to_anchor is not None:
                _leg_kwargs['bbox_to_anchor'] = legend_bbox_to_anchor
            if legend_frame_alpha is not None:
                _leg_kwargs['framealpha'] = legend_frame_alpha
            _legend = ax.legend(**_leg_kwargs)
            if _legend is not None:
                for txt in _legend.get_texts():
                    txt.set_fontweight(_leg_w)
                if legend_patch_edgecolor is not None:
                    _handles = getattr(_legend, 'legend_handles',
                                       getattr(_legend, 'legendHandles', []))
                    for _h in _handles:
                        # Line2D has get_data(); patch/collection handles don't
                        if not hasattr(_h, 'get_data'):
                            try:
                                _h.set_edgecolor(legend_patch_edgecolor)
                                _h.set_linewidth(1.0)
                            except Exception:
                                pass

        plt.tight_layout()

        if save_plots:
            filename = output_filename if output_filename is not None \
                else f'ion_pairing_cutoffs_{ion_type}_modified.png'
            fig.savefig(filename, dpi=dpi, bbox_inches='tight', transparent=transparent)
            print(f"Modified plot saved as: {filename}")

        plt.show()

    def set_ion_pairing_boundaries_manually(self, ion_type, pairing_boundaries):
        '''
        Manually set all ion pairing boundaries for an ion type.
        
        Parameters
        ----------
        ion_type : str
            Ion type name
        pairing_boundaries : dict
            Ion pairing boundaries as dict {'CIP': (start, end), 'SIP': (start, end), ...}
        
        Returns
        -------
        success : bool
            True if boundaries were set successfully
        '''
        
        if not hasattr(self, 'ion_pairs_by_type') or ion_type not in self.ion_pairs_by_type:
            print(f"No ion pairing data for {ion_type}")
            return False
        
        # Validate boundaries
        region_order = ['CIP', 'SIP', 'DSIP', 'FI']
        
        for region, (start, end) in pairing_boundaries.items():
            if start >= end and not np.isinf(end):
                print(f"Invalid boundaries for {region}: start ({start:.2f}) >= end ({end:.2f})")
                return False
            if start < 0:
                print(f"Invalid start boundary for {region}: {start:.2f} < 0")
                return False
        
        # Check for overlaps and gaps
        sorted_regions = []
        for region in region_order:
            if region in pairing_boundaries:
                sorted_regions.append((region, pairing_boundaries[region]))
        
        for i in range(len(sorted_regions) - 1):
            current_region, (current_start, current_end) = sorted_regions[i]
            next_region, (next_start, next_end) = sorted_regions[i + 1]
            
            if not np.isinf(current_end) and current_end != next_start:
                print(f"Gap or overlap between {current_region} and {next_region}")
                print(f"  {current_region} ends at {current_end:.2f}, {next_region} starts at {next_start:.2f}")
                return False
        
        # Apply new boundaries
        ion_pairs = self.ion_pairs_by_type[ion_type]['ion_pairs']
        
        # Clear existing boundaries
        ion_pairs.clear()
        
        # Add new boundaries
        for region, bounds in pairing_boundaries.items():
            ion_pairs[region] = bounds
        
        print(f"Successfully set manual ion pairing boundaries for {ion_type}")
        self._print_ion_pairing_boundaries(ion_type, ion_pairs)
        
        return True

    def remove_ion_pairing_region(self, ion_type, region_to_remove='SIP', merge_option='expand_previous', 
                                rename_regions=None):
        '''
        Remove a specific ion pairing region and merge with adjacent regions.
        Optionally rename remaining regions after removal.
        
        Parameters
        ----------
        ion_type : str
            Ion type to modify (e.g., 'Na', 'Mg', 'Cl', 'cation', 'anion')
        region_to_remove : str
            Pairing region to remove ('CIP', 'SIP', 'DSIP', 'FI'), default='SIP'
        merge_option : str
            How to handle the removal:
            - 'expand_previous': Expand the previous region to include the removed region's range
            - 'expand_next': Expand the next region to include the removed region's range  
            - 'expand_both': Expand both previous and next regions to meet in the middle
            default='expand_previous'
        rename_regions : dict, optional
            Dictionary to rename remaining regions after removal.
            Format: {'old_name': 'new_name', 'old_name2': 'new_name2'}
            Example: {'CIP': 'CONTACT', 'DSIP': 'SIP', 'FI': 'FREE'}
        
        Returns
        -------
        success : bool
            Whether the region was successfully removed
        '''
        
        # Check if ion pairing cutoffs exist
        if not hasattr(self, 'ion_pairs_by_type') or ion_type not in self.ion_pairs_by_type:
            print(f"No ion pairing cutoffs found for {ion_type}.")
            print("Run determine_ion_pairing_cutoffs() first.")
            return False
        
        pairing_data = self.ion_pairs_by_type[ion_type]
        ion_pairs = pairing_data['ion_pairs']
        
        # Get available regions
        available_regions = list(ion_pairs.keys())
        
        # Check if the region exists
        if region_to_remove not in available_regions:
            print(f"Error: Region '{region_to_remove}' not found in {ion_type} ion pairs")
            print(f"Available regions: {available_regions}")
            return False
        
        # Define region order for logical processing
        region_order = ['CIP', 'SIP', 'DSIP', 'FI']
        existing_regions = [r for r in region_order if r in available_regions]
        
        # Find the region index
        if region_to_remove not in existing_regions:
            print(f"Error: Region '{region_to_remove}' not found in existing regions")
            return False
        
        region_idx = existing_regions.index(region_to_remove)
        
        # Get region boundaries
        region_start, region_end = ion_pairs[region_to_remove]
        
        print(f"Removing {ion_type} {region_to_remove}: {region_start:.2f} - {region_end:.2f} Å")
        print(f"Using merge option: {merge_option}")
        
        # Handle edge cases
        is_first_region = (region_idx == 0)
        is_last_region = (region_idx == len(existing_regions) - 1)
        
        # Validate merge options for edge cases
        if is_first_region and merge_option == 'expand_previous':
            print("Warning: Cannot expand previous region for first region (no previous region exists)")
            print("Switching to 'expand_next' option")
            merge_option = 'expand_next'
        
        if is_last_region and merge_option == 'expand_next':
            print(f"Note: {region_to_remove} is the last region.")
            if region_to_remove == 'FI':
                print("Warning: Removing FI region - this will eliminate the 'free ions' category")
        
        if (is_first_region or is_last_region) and merge_option == 'expand_both':
            print("Warning: Cannot expand both directions for first or last region")
            if is_first_region:
                print("Switching to 'expand_next' option")
                merge_option = 'expand_next'
            else:
                print("Switching to 'expand_previous' option") 
                merge_option = 'expand_previous'
        
        # Create new pairing structure
        new_ion_pairs = {}
        
        if merge_option == 'expand_previous':
            # Copy all regions except the one to remove
            for region in existing_regions:
                if region != region_to_remove:
                    if region_idx > 0 and existing_regions.index(region) == region_idx - 1:  # Previous region
                        # Expand previous region to include removed region's end
                        prev_start, prev_end = ion_pairs[region]
                        new_bounds = (prev_start, region_end)
                        new_ion_pairs[region] = new_bounds
                        print(f"  Expanded {region}: {prev_start:.2f} - {region_end:.2f} Å")
                    else:
                        # Keep other regions unchanged
                        new_ion_pairs[region] = ion_pairs[region]
        
        elif merge_option == 'expand_next':
            if is_last_region:
                # Special case: removing last region
                for region in existing_regions:
                    if region != region_to_remove:
                        # Keep all other regions unchanged
                        new_ion_pairs[region] = ion_pairs[region]
                
                # If we removed FI, the second-to-last region becomes the new "free" region
                if region_to_remove == 'FI' and len(existing_regions) > 1:
                    second_last_region = existing_regions[-2]
                    second_last_start, second_last_end = new_ion_pairs[second_last_region]
                    new_ion_pairs[second_last_region] = (second_last_start, np.inf)
                    print(f"  Extended {second_last_region} to infinity: {second_last_start:.2f} - ∞ Å")
            else:
                # Normal case: expand the next region
                for region in existing_regions:
                    if region != region_to_remove:
                        if existing_regions.index(region) == region_idx + 1:  # Next region
                            # Expand next region to include removed region's start
                            next_start, next_end = ion_pairs[region]
                            new_bounds = (region_start, next_end)
                            new_ion_pairs[region] = new_bounds
                            print(f"  Expanded {region}: {region_start:.2f} - {next_end:.2f} Å")
                        else:
                            # Keep other regions unchanged
                            new_ion_pairs[region] = ion_pairs[region]
        
        elif merge_option == 'expand_both':
            # Expand both previous and next regions to meet in the middle
            region_width = region_end - region_start
            midpoint = region_start + region_width / 2
            
            for region in existing_regions:
                if region != region_to_remove:
                    if existing_regions.index(region) == region_idx - 1:  # Previous region
                        # Expand previous region to midpoint of removed region
                        prev_start, prev_end = ion_pairs[region]
                        new_bounds = (prev_start, midpoint)
                        new_ion_pairs[region] = new_bounds
                        print(f"  Expanded {region}: {prev_start:.2f} - {midpoint:.2f} Å")
                    elif existing_regions.index(region) == region_idx + 1:  # Next region
                        # Expand next region to midpoint of removed region
                        next_start, next_end = ion_pairs[region]
                        new_bounds = (midpoint, next_end)
                        new_ion_pairs[region] = new_bounds
                        print(f"  Expanded {region}: {midpoint:.2f} - {next_end:.2f} Å")
                    else:
                        # Keep other regions unchanged
                        new_ion_pairs[region] = ion_pairs[region]
        
        # APPLY RENAMING if requested
        if rename_regions is not None:
            print(f"\nApplying region renaming...")
            final_ion_pairs = {}
            
            for old_name, bounds in new_ion_pairs.items():
                if old_name in rename_regions:
                    new_name = rename_regions[old_name]
                    final_ion_pairs[new_name] = bounds
                    print(f"  Renamed: {old_name} -> {new_name}")
                else:
                    final_ion_pairs[old_name] = bounds
                    print(f"  Kept: {old_name}")
            
            # Validate that no new names conflict with existing names
            if len(final_ion_pairs) != len(set(final_ion_pairs.keys())):
                print("Error: Renaming would create duplicate region names!")
                return False
            
            new_ion_pairs = final_ion_pairs
        
        # Clear the original ion_pairs and add new regions
        ion_pairs.clear()
        for region, bounds in new_ion_pairs.items():
            ion_pairs[region] = bounds
        
        # Print updated ion pairing structure
        remaining_regions = list(new_ion_pairs.keys())
        remaining_regions.sort(key=lambda x: new_ion_pairs[x][0])  # Sort by start position
        
        print(f"\nUpdated {ion_type} ion pairing structure:")
        for region in remaining_regions:
            start, end = ion_pairs[region]
            end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
            
            # Try to get full name, but handle custom names gracefully
            full_names = {
                'CIP': 'Contact Ion Pair',
                'SIP': 'Solvent-separated',
                'DSIP': 'Double Solvent-separated',
                'FI': 'Free Ions'
            }
            
            if region in full_names:
                display_name = f"{region} ({full_names[region]})"
            else:
                display_name = f"{region} (Custom)"
            
            print(f"  {display_name}: {start:.2f} - {end_str} Å")
        
        print(f"\nSuccessfully removed {region_to_remove} region from {ion_type} ion pairs")
        
        return True

    def reset_ion_pairing_cutoffs(self, ion_type):
        '''
        Reset ion pairing cutoffs to original automatically-determined values.
        
        Parameters
        ----------
        ion_type : str
            Ion type to reset (e.g., 'Na', 'Mg', 'Cl', 'cation', 'anion')
        
        Returns
        -------
        success : bool
            Whether the reset was successful
        '''
        
        if not hasattr(self, 'ion_pairs_by_type') or ion_type not in self.ion_pairs_by_type:
            print(f"No ion pairing cutoffs found for {ion_type}.")
            return False
        
        print(f"Resetting {ion_type} ion pairing cutoffs to original values...")
        print("This requires re-running determine_ion_pairing_cutoffs()...")
        
        # Get the RDF info for re-analysis
        pairing_data = self.ion_pairs_by_type[ion_type]
        rdf_key = pairing_data['rdf_key']
        ion_category = pairing_data['ion_category']
        
        # Re-run the analysis with original parameters
        self.determine_ion_pairing_cutoffs(ion_type=ion_type, use_extended_rdf=True, save_plots=False)
        
        print(f"✓ {ion_type} ion pairing cutoffs reset to original values")
        
        return True

    def list_ion_pairing_regions(self, ion_type):
        '''
        List all available ion pairing regions for a specific ion type.
        
        Parameters
        ----------
        ion_type : str
            Ion type to list regions for
        
        Returns
        -------
        regions : list
            List of available pairing regions
        '''
        
        if not hasattr(self, 'ion_pairs_by_type') or ion_type not in self.ion_pairs_by_type:
            print(f"No ion pairing cutoffs found for {ion_type}.")
            return []
        
        ion_pairs = self.ion_pairs_by_type[ion_type]['ion_pairs']
        regions = list(ion_pairs.keys())
        
        print(f"\nAvailable ion pairing regions for {ion_type}:")
        for region in regions:
            start, end = ion_pairs[region]
            end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
            
            # Full names for clarity
            full_names = {
                'CIP': 'Contact Ion Pair',
                'SIP': 'Solvent-separated',
                'DSIP': 'Double Solvent-separated',
                'FI': 'Free Ions'
            }
            full_name = full_names.get(region, region)
            print(f"  {region} ({full_name}): {start:.2f} - {end_str} Å")
        
        return regions


    def interactive_boundary_editor(self, ion_type):
        '''
        Interactive boundary editor for fine-tuning shell boundaries.
        '''
        
        shells = self._get_shells_for_ion(ion_type)
        if shells is None:
            return
        
        print(f"\n=== INTERACTIVE BOUNDARY EDITOR FOR {ion_type.upper()} ===")
        print("Commands:")
        print("  list                          - Show current boundaries")
        print("  modify <shell> <start|end> <value> - Modify boundary (e.g., 'modify shell_1 end 3.5')")
        print("  expand <shell> <direction> <amount> - Expand shell (e.g., 'expand shell_2 outward 0.5')")
        print("  contract <shell> <direction> <amount> - Contract shell (e.g., 'contract shell_1 both 0.2')")
        print("  add <shell> <start> <end>     - Add a new region (e.g., 'add shell_4 6.12 8.32' or end=inf)")
        print("  remove <shell>                - Remove a region (e.g., 'remove shell_3')")
        print("  replot                        - Replot with current boundaries")
        print("  reset                         - Reset to original boundaries (if available)")
        print("  save [filename]               - Save all boundaries to JSON (default: shell_boundaries.json)")
        print("  load [filename]               - Load boundaries from JSON (default: shell_boundaries.json)")
        print("  quit                          - Exit editor")
        print()
        
        # Store original boundaries for reset
        original_boundaries = shells.data.copy()
        
        while True:
            # Show current boundaries
            self._print_shell_boundaries(ion_type, shells)
            
            command = input(f"\n[{ion_type}] > ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'list':
                continue  # Already printed above
            elif command == 'replot':
                self.replot_ion_after_modification(ion_type)
            elif command == 'reset':
                shells.data = original_boundaries.copy()
                print("Reset to original boundaries")
            elif command.startswith('modify '):
                parts = command.split()
                if len(parts) == 4:
                    _, shell_name, boundary_type, value = parts
                    try:
                        new_value = float(value)
                        self.modify_shell_boundary(ion_type, shell_name, boundary_type, new_value)
                    except ValueError:
                        print("Invalid value. Use a number.")
                else:
                    print("Usage: modify <shell> <start|end> <value>")
            elif command.startswith('expand '):
                parts = command.split()
                if len(parts) == 4:
                    _, shell_name, direction, amount = parts
                    try:
                        amount_val = float(amount)
                        self.expand_shell_region(ion_type, shell_name, direction, amount_val)
                    except ValueError:
                        print("Invalid amount. Use a number.")
                else:
                    print("Usage: expand <shell> <inward|outward|both> <amount>")
            elif command.startswith('contract '):
                parts = command.split()
                if len(parts) == 4:
                    _, shell_name, direction, amount = parts
                    try:
                        amount_val = float(amount)
                        self.contract_shell_region(ion_type, shell_name, direction, amount_val)
                    except ValueError:
                        print("Invalid amount. Use a number.")
                else:
                    print("Usage: contract <shell> <inward|outward|both> <amount>")
            elif command.startswith('add '):
                parts = command.split()
                if len(parts) == 4:
                    _, shell_name, start_str, end_str = parts
                    try:
                        start_val = float(start_str)
                        end_val = np.inf if end_str in ('inf', 'infinity', '∞') else float(end_str)
                        if start_val < 0:
                            print("Invalid start value: must be >= 0")
                        elif end_val <= start_val:
                            print(f"Invalid bounds: end ({end_str}) must be greater than start ({start_val})")
                        else:
                            shells.data[shell_name] = (start_val, end_val)
                            end_display = '∞' if np.isinf(end_val) else f'{end_val:.2f}'
                            print(f"Added '{shell_name}': {start_val:.2f} - {end_display} Å")
                            # Auto-update bulk start to follow the last non-bulk shell end
                            if shell_name != 'bulk' and 'bulk' in shells.data:
                                last_end = max(
                                    v[1] for k, v in shells.data.items()
                                    if k != 'bulk' and not np.isinf(v[1])
                                )
                                old_bulk = shells.data['bulk']
                                shells.data['bulk'] = (last_end, old_bulk[1])
                                print(f"Updated 'bulk': {last_end:.2f} - ∞ Å")
                            print("Use 'replot' to see changes")
                    except ValueError:
                        print("Invalid start/end values. Use numbers (or 'inf' for infinity).")
                else:
                    print("Usage: add <shell_name> <start> <end>")
                    print("Example: add shell_4 6.12 8.32")
                    print("         add bulk 8.32 inf")
            elif command.startswith('remove '):
                parts = command.split()
                if len(parts) == 2:
                    _, shell_name = parts
                    if shell_name in shells.data:
                        start_val, end_val = shells.data.pop(shell_name)
                        end_display = '∞' if np.isinf(end_val) else f'{end_val:.2f}'
                        print(f"Removed '{shell_name}' ({start_val:.2f} - {end_display} Å)")
                        print("Use 'replot' to see changes")
                    else:
                        available = list(shells.data.keys())
                        print(f"Region '{shell_name}' not found. Available: {available}")
                else:
                    print("Usage: remove <shell_name>")
            elif command == 'help':
                print("Commands: list, modify, expand, contract, add, remove, replot, reset, save, load, quit")
            elif command.startswith('save'):
                parts = command.split(maxsplit=1)
                fname = parts[1].strip() if len(parts) > 1 else 'shell_boundaries.json'
                self.save_shell_boundaries_to_json(fname)
            elif command.startswith('load'):
                parts = command.split(maxsplit=1)
                fname = parts[1].strip() if len(parts) > 1 else 'shell_boundaries.json'
                self.load_shell_boundaries_from_json(fname)
            else:
                print("Unknown command. Type 'help' for available commands.")
        
        print("Boundary editor closed")

    # ------------------------------------------------------------------
    def save_shell_boundaries_to_json(self, filename='shell_boundaries.json'):
        '''
        Save current shell boundaries for all ion types to a JSON file.

        Parameters
        ----------
        filename : str
            Output JSON filename, default='shell_boundaries.json'

        Returns
        -------
        success : bool
        '''
        data = {}
        for attr in ('cation_shells_by_type', 'anion_shells_by_type'):
            shells_dict = getattr(self, attr, {})
            for ion_name, shells in shells_dict.items():
                if shells is None or not hasattr(shells, 'data'):
                    continue
                ion_bounds = {}
                for shell_name, bounds in shells.data.items():
                    start, end = float(bounds[0]), bounds[1]
                    ion_bounds[shell_name] = [start, None if np.isinf(end) else float(end)]
                data[ion_name] = ion_bounds
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Shell boundaries saved to '{filename}'")
            return True
        except Exception as e:
            print(f"Error saving shell boundaries: {e}")
            return False

    # ------------------------------------------------------------------
    def load_shell_boundaries_from_json(self, filename='shell_boundaries.json'):
        '''
        Load shell boundaries from a JSON file previously saved by
        ``save_shell_boundaries_to_json`` and apply them to the current
        shell objects.

        Parameters
        ----------
        filename : str
            Input JSON filename, default='shell_boundaries.json'

        Returns
        -------
        success : bool
        '''
        import os
        if not os.path.exists(filename):
            print(f"File '{filename}' not found")
            return False
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading '{filename}': {e}")
            return False

        loaded = []
        skipped = []
        for ion_name, ion_bounds in data.items():
            shells = self._get_shells_for_ion(ion_name)
            if shells is None:
                skipped.append(ion_name)
                continue
            new_data = {}
            for shell_name, (start, end) in ion_bounds.items():
                new_data[shell_name] = (float(start), np.inf if end is None else float(end))
            shells.data = new_data
            loaded.append(ion_name)

        if loaded:
            print(f"Loaded boundaries for: {loaded}")
        if skipped:
            print(f"Skipped (ion not in system): {skipped}")
        return bool(loaded)

        '''Handle removal for specific ion types (Na, Mg, Cl, etc.) - new functionality'''
        
        # Check if we have ion-type-specific shells available
        if not (hasattr(self, 'cation_shells_by_type') or hasattr(self, 'anion_shells_by_type')):
            print(f"Error: Ion-type-specific shells not found.")
            print(f"This method requires shells determined by specific ion types.")
            print(f"Try running determine_ion_solvation_shells_by_type() first, or use broad categories ('cation'/'anion').")
            return False
        
        # Determine if this is a cation or anion and get the appropriate shells
        shells = None
        ion_category = None
        
        # Check in cation shells
        if hasattr(self, 'cation_shells_by_type') and ion_type in self.cation_shells_by_type:
            shells = self.cation_shells_by_type[ion_type]
            ion_category = 'cation'
        # Check in anion shells  
        elif hasattr(self, 'anion_shells_by_type') and ion_type in self.anion_shells_by_type:
            shells = self.anion_shells_by_type[ion_type]
            ion_category = 'anion'
        else:
            print(f"Error: Ion type '{ion_type}' not found in ion-type-specific shells.")
            available_cations = list(self.cation_shells_by_type.keys()) if hasattr(self, 'cation_shells_by_type') else []
            available_anions = list(self.anion_shells_by_type.keys()) if hasattr(self, 'anion_shells_by_type') else []
            print(f"Available cation types: {available_cations}")
            print(f"Available anion types: {available_anions}")
            return False
        
        if shells is None:
            print(f"Error: No shells found for {ion_type}")
            return False
        
        # Get available shells
        available_shells = list(shells.data.keys()) if hasattr(shells, 'data') else []
        
        # Check if the shell exists
        if shell_to_remove not in available_shells:
            print(f"Error: Shell '{shell_to_remove}' not found in {ion_type} shells")
            print(f"Available shells: {available_shells}")
            return False
        
        # Get shell numbers only (exclude bulk)
        shell_attrs = [attr for attr in available_shells if attr.startswith('shell_')]
        shell_attrs.sort()
        
        # Find the shell index
        shell_idx = shell_attrs.index(shell_to_remove)
        shell_start, shell_end = shells.data[shell_to_remove]
        
        print(f"Removing {ion_type} ({ion_category}) {shell_to_remove}: {shell_start:.2f} - {shell_end:.2f} Å")
        print(f"Using merge option: {merge_option}")
        
        # Handle edge cases
        is_first_shell = (shell_to_remove == 'shell_1')
        is_last_shell = (shell_idx == len(shell_attrs) - 1)
        
        # Validate merge options for edge cases
        if is_first_shell and merge_option == 'expand_previous':
            print("Warning: Cannot expand previous shell for shell_1 (no previous shell exists)")
            print("Switching to 'expand_next' option")
            merge_option = 'expand_next'
        
        if is_last_shell and merge_option == 'expand_next':
            print(f"Note: {shell_to_remove} is the last shell. Will expand bulk region.")
        
        # Create new shell structure
        new_shells_data = {}
        
        if merge_option == 'expand_previous':
            # Copy all shells except the one to remove
            for attr in shell_attrs:
                if attr != shell_to_remove:
                    if shell_attrs.index(attr) == shell_idx - 1:  # Previous shell
                        # Expand previous shell to include removed shell's end
                        prev_start, prev_end = shells.data[attr]
                        new_bounds = (prev_start, shell_end)
                        new_shells_data[attr] = new_bounds
                        print(f"  Expanded {attr}: {prev_start:.2f} - {shell_end:.2f} Å")
                    else:
                        # Keep other shells unchanged
                        new_shells_data[attr] = shells.data[attr]
            
            # Keep or create bulk region
            if 'bulk' in shells.data:
                new_shells_data['bulk'] = shells.data['bulk']
            else:
                # Create bulk region starting from end of last remaining shell
                last_shell_end = max([end for start, end in new_shells_data.values()])
                new_shells_data['bulk'] = (last_shell_end, np.inf)
                print(f"  Created bulk region: {last_shell_end:.2f} - ∞ Å")
        
        elif merge_option == 'expand_next':
            if is_last_shell:
                # Special case: removing last shell, expand or create bulk
                for attr in shell_attrs:
                    if attr != shell_to_remove:
                        # Keep all other shells unchanged
                        new_shells_data[attr] = shells.data[attr]
                
                # Always create/expand bulk region when removing last shell
                if 'bulk' in shells.data:
                    bulk_start, bulk_end = shells.data['bulk']
                    new_shells_data['bulk'] = (shell_start, bulk_end)
                    print(f"  Expanded bulk: {shell_start:.2f} - ∞ Å")
                else:
                    # Create new bulk region starting from removed shell start
                    new_shells_data['bulk'] = (shell_start, np.inf)
                    print(f"  Created bulk region: {shell_start:.2f} - ∞ Å")
            else:
                # Normal case: expand the next shell
                for attr in shell_attrs:
                    if attr != shell_to_remove:
                        if shell_attrs.index(attr) == shell_idx + 1:  # Next shell
                            # Expand next shell to include removed shell's start
                            next_start, next_end = shells.data[attr]
                            new_bounds = (shell_start, next_end)
                            new_shells_data[attr] = new_bounds
                            print(f"  Expanded {attr}: {shell_start:.2f} - {next_end:.2f} Å")
                        else:
                            # Keep other shells unchanged
                            new_shells_data[attr] = shells.data[attr]
                
                # Keep or create bulk region
                if 'bulk' in shells.data:
                    new_shells_data['bulk'] = shells.data['bulk']
                else:
                    # Create bulk region starting from end of last shell
                    last_shell_end = max([end for start, end in new_shells_data.values()])
                    new_shells_data['bulk'] = (last_shell_end, np.inf)
                    print(f"  Created bulk region: {last_shell_end:.2f} - ∞ Å")
        
        # Renumber shells sequentially
        final_shells = Results()
        final_shells.data = {}
        
        # Get remaining shell attributes (excluding bulk)
        remaining_shell_attrs = [attr for attr in new_shells_data.keys() if attr.startswith('shell_')]
        remaining_shell_attrs.sort()
        
        # Renumber sequentially
        for i, old_attr in enumerate(remaining_shell_attrs, 1):
            new_attr = f'shell_{i}'
            bounds = new_shells_data[old_attr]
            final_shells.data[new_attr] = bounds
            setattr(final_shells, new_attr, bounds)
        
        # ALWAYS add bulk region
        if 'bulk' in new_shells_data:
            final_shells.data['bulk'] = new_shells_data['bulk']
            setattr(final_shells, 'bulk', new_shells_data['bulk'])
        else:
            # Failsafe: create bulk if it doesn't exist
            if final_shells.data:
                last_shell_end = max([end for start, end in final_shells.data.values()])
                final_shells.data['bulk'] = (last_shell_end, np.inf)
                setattr(final_shells, 'bulk', (last_shell_end, np.inf))
                print(f"  Failsafe: Created bulk region: {last_shell_end:.2f} - ∞ Å")
        
        # Copy RDF data for plotting
        if hasattr(shells, 'rdf_r'):
            final_shells.rdf_r = shells.rdf_r.copy()
            setattr(final_shells, 'rdf_r', shells.rdf_r.copy())
        if hasattr(shells, 'rdf_g_r'):
            final_shells.rdf_g_r = shells.rdf_g_r.copy()
            setattr(final_shells, 'rdf_g_r', shells.rdf_g_r.copy())
        if hasattr(shells, 'minima_indices'):
            final_shells.minima_indices = shells.minima_indices.copy()
            setattr(final_shells, 'minima_indices', shells.minima_indices.copy())
        if hasattr(shells, 'peak_indices'):
            final_shells.peak_indices = shells.peak_indices.copy()
            setattr(final_shells, 'peak_indices', shells.peak_indices.copy())
        
        # Update the shells in the appropriate dictionary
        if ion_category == 'cation':
            self.cation_shells_by_type[ion_type] = final_shells
        else:
            self.anion_shells_by_type[ion_type] = final_shells
        
        # Print updated shell structure
        final_shell_attrs = [attr for attr in final_shells.data.keys() if attr.startswith('shell_')]
        final_shell_attrs.sort()
        if 'bulk' in final_shells.data:
            final_shell_attrs.append('bulk')
        
        print(f"\nUpdated {ion_type} solvation structure:")
        for attr in final_shell_attrs:
            start, end = final_shells.data[attr]
            end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
            name = attr.replace('_', ' ').title()
            print(f"  {name}: {start:.2f} - {end_str} Å")
        
        return True 


    def replot_ion_after_modification(self, ion_type, plot_range=None, save_plots=True, use_extended_rdf=True):
        '''
        Replot an ion's solvation shells after modification - works with specific ion types.
        
        Parameters
        ----------
        ion_type : str
            Specific ion type name (e.g., 'Na', 'Mg', 'Cl')
        plot_range : float, optional
            Plot range in Angstroms. If None, uses 20
        save_plots : bool
            Whether to save the plot
        use_extended_rdf : bool
            Whether to use extended RDF data
        '''
        
        if plot_range is None:
            plot_range = 20
        
        # Find which category this ion belongs to
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        if ion_type in cation_types:
            ion_category = 'cation'
            shells_dict = self.cation_shells_by_type
        elif ion_type in anion_types:
            ion_category = 'anion'
            shells_dict = self.anion_shells_by_type
        else:
            print(f"Ion type '{ion_type}' not found in system.")
            return
        
        # Check if we have RDF data
        rdf_key = f'{ion_type}-w'
        if not hasattr(self, 'rdfs') or rdf_key not in self.rdfs:
            print(f"No RDF data available for {ion_type}")
            return
        
        # Check if ion has shells
        if ion_type not in shells_dict or shells_dict[ion_type] is None:
            print(f"No shells found for {ion_type}")
            return
        
        shells = shells_dict[ion_type]
        
        print(f"Replotting {ion_type} with current shells...")
        
        # Plot with current shells using your enhanced plotting method
        self._plot_shells_for_ion_type_enhanced(
            self.rdfs[rdf_key], 
            shells, 
            ion_type, 
            ion_category,
            save_plots=save_plots,
            plot_range=plot_range
        )



    def calculate_water_residence_times_in_shells(self, ion_type=None, shell=None, step=None, min_residence=1):
        '''
        Calculate average residence times for water molecules in different coordination shells.
        Analyzes how long a water molecule stays continuously in each shell region.
        
        Parameters
        ----------
        ion_type : str, list, or None
            Specific ion type(s) to analyze:
            - None: analyzes all ion types
            - Single string (e.g., 'Na'): analyzes only this ion
            - List (e.g., ['Na', 'Mg']): analyzes only these ions
        shell : int or None
            Specific shell to analyze (1, 2, or 3). If None, analyzes all shells.
        step : int, optional
            Step size for trajectory analysis
        min_residence : int
            Minimum number of consecutive frames to count as a residence event, default=1
        
        Returns
        -------
        results : dict
            Dictionary with residence time statistics for each ion type and shell region
        '''
        
        if step is None:
            step = self.default_step
        
        # Check if shells are available
        if not (hasattr(self, 'cation_shells_by_type') and hasattr(self, 'anion_shells_by_type')):
            print("Ion-type-specific shells not found. Run determine_ion_solvation_shells_by_type() first.")
            return None
        
        # Get unique ion types
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        # FIXED: Handle ion_type parameter to support lists
        if ion_type is not None:
            # Convert single string to list
            if isinstance(ion_type, str):
                ion_type = [ion_type]
            
            # Filter to requested ion types
            ions_to_process = []
            for ion in ion_type:
                if ion in cation_types:
                    ions_to_process.append((ion, 'cation', cation_types[ion]))
                elif ion in anion_types:
                    ions_to_process.append((ion, 'anion', anion_types[ion]))
                else:
                    print(f"Warning: Ion type '{ion}' not found in system")
            
            if not ions_to_process:
                available_types = list(cation_types.keys()) + list(anion_types.keys())
                print(f"No valid ion types specified. Available: {available_types}")
                return None
        else:
            # Use all available ion types
            ions_to_process = [(name, 'cation', group) for name, group in cation_types.items()]
            ions_to_process += [(name, 'anion', group) for name, group in anion_types.items()]
        
        # Determine which shell(s) to analyze
        if shell is not None:
            if shell not in [1, 2, 3]:
                print(f"Invalid shell number: {shell}. Must be 1, 2, or 3.")
                return None
            shells_to_analyze = [f'shell_{shell}']
            print(f"Analyzing shell_{shell} only")
        else:
            shells_to_analyze = None  # Will analyze all available shells
            print(f"Analyzing all shells")
        
        print(f"Calculating water residence times in coordination shells")
        print(f"  Ion types: {[ion[0] for ion in ions_to_process]}")
        print(f"  Step: {step}, Min residence: {min_residence} frames")
        
        # Handle debug trajectory
        if hasattr(self, '_debug_frame_indices') and self._debug_frame_indices is not None:
            frame_indices = self._debug_frame_indices[::step]
        else:
            frame_indices = range(0, len(self.universe.trajectory), step)
        
        results = {}
        
        for ion_name, ion_category, ion_group in ions_to_process:
            print(f"\nAnalyzing {ion_name} ({ion_category})...")
            
            # Get shells for this ion type
            if ion_category == 'cation':
                shells = self.cation_shells_by_type.get(ion_name)
            else:
                shells = self.anion_shells_by_type.get(ion_name)
            
            if shells is None:
                print(f"  No shells found for {ion_name}, skipping")
                continue
            
            # Get shell regions (exclude bulk)
            all_shell_regions = {k: v for k, v in shells.data.items() if k.startswith('shell_')}
            
            # Filter by specified shell(s)
            if shells_to_analyze is not None:
                shell_regions = {k: v for k, v in all_shell_regions.items() if k in shells_to_analyze}
            else:
                shell_regions = all_shell_regions
            
            if not shell_regions:
                print(f"  No valid shells to analyze for {ion_name}")
                continue
            
            # Track water molecule occupancy in each shell over time
            water_shell_history = {}  # {water_id: {frame_idx: shell_name}}
            
            print(f"  Tracking {len(self.waters)} water molecules across {len(frame_indices)} frames...")
            
            # Build occupancy history
            for frame_idx in tqdm(frame_indices, desc=f"Tracking waters for {ion_name}", leave=False):
                self.universe.trajectory[frame_idx]
                
                for water in self.waters:
                    water_id = water.index
                    
                    # Calculate distances to all ions of this type
                    distances = cdist([water.position], ion_group.positions)[0]
                    min_dist = distances.min()
                    
                    # Determine which shell (if any) the water is in
                    current_shell = None
                    for shell_name, (shell_start, shell_end) in shell_regions.items():
                        if shell_start <= min_dist < shell_end:
                            current_shell = shell_name
                            break
                    
                    # Store shell assignment
                    if water_id not in water_shell_history:
                        water_shell_history[water_id] = {}
                    water_shell_history[water_id][frame_idx] = current_shell
            
            # Analyze residence times from occupancy history
            shell_residence_times = {shell_name: [] for shell_name in shell_regions.keys()}
            
            print(f"  Analyzing residence times for {len(water_shell_history)} waters...")
            
            for water_id, frame_history in water_shell_history.items():
                sorted_frames = sorted(frame_history.keys())
                
                if not sorted_frames:
                    continue
                
                # Track consecutive residence events
                current_shell = frame_history[sorted_frames[0]]
                residence_start = sorted_frames[0]
                
                for i in range(1, len(sorted_frames)):
                    frame = sorted_frames[i]
                    new_shell = frame_history[frame]
                    
                    if new_shell != current_shell:
                        # Ended a residence event
                        if current_shell is not None:
                            residence_duration = frame - residence_start
                            if residence_duration >= min_residence:
                                shell_residence_times[current_shell].append(residence_duration)
                        
                        # Start new residence event
                        current_shell = new_shell
                        residence_start = frame
                
                # Handle final residence event
                if current_shell is not None:
                    residence_duration = sorted_frames[-1] - residence_start + 1
                    if residence_duration >= min_residence:
                        shell_residence_times[current_shell].append(residence_duration)
            
            # Calculate statistics
            ion_results = {
                'ion_type': ion_name,
                'ion_category': ion_category,
                'n_ions': len(ion_group),
                'n_frames_analyzed': len(frame_indices),
                'step': step,
                'shells': {}
            }
            
            for shell_name, (shell_start, shell_end) in shell_regions.items():
                residence_array = np.array(shell_residence_times[shell_name])
                
                ion_results['shells'][shell_name] = {
                    'bounds': (shell_start, shell_end),
                    'name': shell_name,
                    'residence_times': residence_array,
                    'n_residence_events': len(residence_array),
                    'mean_residence': residence_array.mean() if len(residence_array) > 0 else 0,
                    'median_residence': np.median(residence_array) if len(residence_array) > 0 else 0,
                    'std_residence': residence_array.std() if len(residence_array) > 0 else 0
                }
            
            results[ion_name] = ion_results
        
        # Store results
        self.water_residence_times = results
        
        # Print summary
        self._print_water_residence_summary(results)
        
        return results



    def _print_water_residence_summary(self, results):
        '''Print summary of water residence times'''
        
        print(f"\n{'='*80}")
        print("WATER RESIDENCE TIMES IN COORDINATION SHELLS")
        print(f"{'='*80}")
        
        for ion_type, ion_data in results.items():
            print(f"\n{ion_type.upper()} ({ion_data['ion_category']}):")
            print(f"  Analyzed {ion_data['n_ions']} ions over {ion_data['n_frames_analyzed']} frames (step={ion_data['step']})")
            print(f"  Shell residence times:")
            
            for shell_name in sorted(ion_data['shells'].keys()):
                shell_data = ion_data['shells'][shell_name]
                bounds = shell_data['bounds']
                
                print(f"\n    {shell_name} ({bounds[0]:.2f}-{bounds[1]:.2f} Å):")
                print(f"      Residence events: {shell_data['n_residence_events']}")
                
                if shell_data['n_residence_events'] > 0:
                    # Use correct key names that match what's stored in the calculation
                    print(f"      Mean residence: {shell_data['mean_residence']:.2f} ± {shell_data['std_residence']:.2f} frames")
                    print(f"      Median residence: {shell_data['median_residence']:.2f} frames")
                    
                    # Calculate min/max from residence_times array if available
                    if 'residence_times' in shell_data and len(shell_data['residence_times']) > 0:
                        min_res = shell_data['residence_times'].min()
                        max_res = shell_data['residence_times'].max()
                        print(f"      Range: {min_res:.2f} - {max_res:.2f} frames")
                else:
                    print(f"      No residence events recorded")
        
        print(f"{'='*80}")



    def plot_water_residence_distributions(self, ion_types=None, save_plots=True, combined_layout=True):
        '''
        Plot distributions of water residence times in different shells.
        MODIFIED: Can now create single figure with all ion types in separate panels.
        FIXED: Calculate mean on-the-fly from residence_times array since 'mean' key doesn't exist
        NOW STORES PLOT DATA for interactive inset editor support.
        UPDATED: Removed legend, mean line, and vertical grid lines for cleaner plots
        
        Parameters
        ----------
        ion_types : str, list, or None
            Specific ion type(s) to plot (e.g., 'Na' or ['Na', 'Cl']).
            If None, plots all available ion types.
        save_plots : bool
            Whether to save plots
        combined_layout : bool
            If True, creates single figure with all ions in separate panels.
            If False, separates by cation/anion categories (original behavior).
        '''
        
        if not hasattr(self, 'water_residence_times'):
            print("No water residence time data. Run calculate_water_residence_times_in_shells() first.")
            return
        
        results = self.water_residence_times
        
        # Filter by ion_types if specified
        if ion_types is not None:
            if isinstance(ion_types, str):
                ion_types = [ion_types]
            results = {k: v for k, v in results.items() if k in ion_types}
            
            if not results:
                available_types = list(self.water_residence_times.keys())
                print(f"No data found for specified ion types: {ion_types}")
                print(f"Available types: {available_types}")
                return
        
        if combined_layout:
            # NEW: Create single figure with all ion types in separate panels
            n_ions = len(results)
            
            # Get maximum number of shells across all ions
            max_shells = max(len(ion_data['shells']) for ion_data in results.values())
            
            # Create grid layout
            fig, axes = plt.subplots(max_shells, n_ions, figsize=(5*n_ions, 4*max_shells))
            
            # Handle single subplot cases
            if n_ions == 1 and max_shells == 1:
                axes = np.array([[axes]])
            elif n_ions == 1:
                axes = axes.reshape(-1, 1)
            elif max_shells == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle('Water Residence Time Distributions', fontsize=16, fontweight='bold')
            
            # ADDED: Store axes and data for interactive editing
            self._residence_plot_data = {
                'fig': fig,
                'axes': axes,
                'data': results,
                'category': 'combined',
                'n_ions': n_ions,
                'max_shells': max_shells,
                'insets': {},
                'layout': 'combined'
            }
            
            # Determine colors based on ion category
            cation_types_in_system = set()
            if hasattr(self, '_get_unique_ion_types'):
                cation_types_in_system = set(self._get_unique_ion_types(self.cations).keys())
            
            # Plot each ion type in a column
            for col, (ion_name, ion_data) in enumerate(results.items()):
                shell_names = sorted(ion_data['shells'].keys())
                
                # Determine color scheme for this ion
                if ion_name in cation_types_in_system:
                    color_scheme = plt.cm.Blues
                    ion_category = 'cation'
                else:
                    color_scheme = plt.cm.Reds
                    ion_category = 'anion'
                
                for row, shell_name in enumerate(shell_names):
                    ax = axes[row, col]
                    shell_data = ion_data['shells'][shell_name]
                    
                    residence_times = shell_data['residence_times']
                    
                    if len(residence_times) > 0:
                        # Plot histogram
                        color = color_scheme(0.6)
                        ax.hist(residence_times, bins=30, alpha=0.7, color=color, 
                            edgecolor='black', linewidth=0.5, density=True)
                        
                        # REMOVED: Mean line calculation and plotting
                        # mean_val = np.mean(residence_times)
                        # ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                        #           label=f'Mean: {mean_val:.1f}')
                        
                        # Shell info in title
                        start, end = shell_data['bounds']
                        end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                        
                        ax.set_title(f'{ion_name} {shell_data["name"]}\n'
                                f'({start:.2f}-{end_str} Å)', 
                                fontsize=10, fontweight='bold')
                        ax.set_xlabel('Residence Time (frames)', fontsize=9)
                        ax.set_ylabel('Probability Density', fontsize=9)
                        # REMOVED: ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines
                    else:
                        # No data for this shell
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
                        ax.set_title(f'{ion_name} {shell_data["name"]}', 
                                fontsize=10, fontweight='bold')
                
                # Hide unused subplots in this column
                for row in range(len(shell_names), max_shells):
                    axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            if save_plots:
                # Create filename with all ion types
                ion_types_str = '_'.join(results.keys())
                filename = f'water_residence_distributions_{ion_types_str}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Combined plot saved as: {filename}")
            
            plt.show()
            
            # ADDED: Offer interactive zoom editor
            print("\n" + "="*80)
            print("INTERACTIVE ZOOM INSET EDITOR")
            print("="*80)
            print("Add zoomed-in insets to highlight low-frequency, long residence time regions.")
            print("Use: eq_opt.interactive_residence_inset_editor(plot_type='water_residence')")
            print(f"Available plots: rows 0-{max_shells-1}, cols 0-{n_ions-1}")
            print("="*80)
        
        else:
            # Original behavior: separate by category
            # Separate cations and anions
            cations_data = {k: v for k, v in results.items() if v['ion_category'] == 'cation'}
            anions_data = {k: v for k, v in results.items() if v['ion_category'] == 'anion'}
            
            # Create plots for cations
            if cations_data:
                self._plot_residence_distributions_by_category(cations_data, 'Cation', save_plots)
            
            # Create plots for anions
            if anions_data:
                self._plot_residence_distributions_by_category(anions_data, 'Anion', save_plots)



    def plot_ion_pairing_residence_distributions(self, ion_types=None, save_plots=True):
        '''
        Plot distributions of ion pairing residence times in different pairing regions.
        
        Parameters
        ----------
        ion_types : str, list, or None
            Specific ion type(s) to plot (e.g., 'Na' or ['Na', 'Cl']).
            If None, plots all available ion types.
        save_plots : bool
            Whether to save plots
        '''
        
        if not hasattr(self, 'ion_pairing_residence_times'):
            print("No ion pairing residence time data. Run calculate_ion_residence_times_in_pairing_regions() first.")
            return
        
        results = self.ion_pairing_residence_times
        
        # Filter by ion_types if specified
        if ion_types is not None:
            if isinstance(ion_types, str):
                ion_types = [ion_types]
            results = {k: v for k, v in results.items() if k in ion_types}
            
            if not results:
                print(f"No data found for specified ion types: {ion_types}")
                print(f"Available ion types: {list(self.ion_pairing_residence_times.keys())}")
                return
        
        # Separate cations and anions
        cations_data = {k: v for k, v in results.items() if v['ion_category'] == 'cation'}
        anions_data = {k: v for k, v in results.items() if v['ion_category'] == 'anion'}
        
        # Create plots for cations
        if cations_data:
            self._plot_pairing_residence_by_category(cations_data, 'Cation', save_plots)
        
        # Create plots for anions
        if anions_data:
            self._plot_pairing_residence_by_category(anions_data, 'Anion', save_plots)


    def calculate_ion_residence_times_in_pairing_regions(self, ion_type=None, region=None, step=None, min_residence=1):
        '''
        Calculate average residence times for ions in different ion pairing regions (CIP, SIP, DSIP).
        Analyzes how long an ion stays continuously in each pairing region.
        
        Parameters
        ----------
        ion_type : str, list, or None
            Specific ion type(s) to analyze:
            - None: analyzes all ion types
            - Single string (e.g., 'Na'): analyzes only this ion
            - List (e.g., ['Na', 'Mg']): analyzes only these ions
        region : str or None
            Specific pairing region to analyze ('CIP', 'SIP', or 'DSIP'). If None, analyzes all regions.
        step : int, optional
            Step size for trajectory analysis
        min_residence : int
            Minimum number of consecutive frames to count as a residence event, default=1
        
        Returns
        -------
        results : dict
            Dictionary with residence time statistics for each ion type and pairing region
        '''
        
        if step is None:
            step = self.default_step
        
        # Check if ion pairing cutoffs are available
        if not hasattr(self, 'ion_pairs_by_type'):
            print("Ion pairing cutoffs not found. Run determine_ion_pairing_cutoffs() first.")
            return None
        
        # Validate region parameter
        if region is not None:
            if region not in ['CIP', 'SIP', 'DSIP']:
                print(f"Invalid region: {region}. Must be 'CIP', 'SIP', or 'DSIP'")
                return None
            regions_to_analyze = [region]
            print(f"Analyzing {region} only")
        else:
            regions_to_analyze = ['CIP', 'SIP', 'DSIP']
            print(f"Analyzing all pairing regions")
        
        # FIXED: Handle ion_type parameter to support lists
        if ion_type is not None:
            # Convert single string to list
            if isinstance(ion_type, str):
                ion_type = [ion_type]
            
            # Filter to requested ion types
            ions_to_process = []
            for ion in ion_type:
                if ion in self.ion_pairs_by_type:
                    ions_to_process.append(ion)
                else:
                    print(f"Warning: Ion type '{ion}' not found in ion_pairs_by_type")
            
            if not ions_to_process:
                available_types = list(self.ion_pairs_by_type.keys())
                print(f"No valid ion types specified. Available: {available_types}")
                return None
        else:
            # Use all available ion types
            ions_to_process = list(self.ion_pairs_by_type.keys())
        
        print(f"Calculating ion residence times in pairing regions")
        print(f"  Ion types: {ions_to_process}")
        print(f"  Step: {step}, Min residence: {min_residence} frames")
        
        # Handle debug trajectory
        if hasattr(self, '_debug_frame_indices') and self._debug_frame_indices is not None:
            frame_indices = self._debug_frame_indices[::step]
        else:
            frame_indices = range(0, len(self.universe.trajectory), step)
        
        results = {}
        
        for ion_name in ions_to_process:
            print(f"\nAnalyzing {ion_name}...")
            
            pairing_data = self.ion_pairs_by_type[ion_name]
            ion_pairs = pairing_data['ion_pairs']
            ion_category = pairing_data['ion_category']
            
            # Get pairing regions (CIP, SIP, DSIP only - no FI)
            pairing_regions = {}
            for region_name in regions_to_analyze:
                if region_name in ion_pairs:
                    pairing_regions[region_name] = ion_pairs[region_name]
            
            if not pairing_regions:
                print(f"  No valid pairing regions found for {ion_name}")
                continue
            
            # Get ions and counterions
            if ion_category == 'cation':
                cation_types = self._get_unique_ion_types(self.cations)
                if ion_name in cation_types:
                    ions = cation_types[ion_name]
                else:
                    ions = self.cations
                counterions = self.anions
            else:
                anion_types = self._get_unique_ion_types(self.anions)
                if ion_name in anion_types:
                    ions = anion_types[ion_name]
                else:
                    ions = self.anions
                counterions = self.cations
            
            # Track ion occupancy in each pairing region over time
            ion_region_history = {}  # {ion_id: {frame_idx: region_name}}
            
            print(f"  Tracking {len(ions)} {ion_name} ions across {len(frame_indices)} frames...")
            
            # Build occupancy history
            for frame_idx in tqdm(frame_indices, desc=f"Tracking {ion_name} pairing", leave=False):
                self.universe.trajectory[frame_idx]
                
                for ion in ions:
                    ion_id = ion.index
                    
                    # Find closest counterion
                    distances = cdist([ion.position], counterions.positions)[0]
                    min_dist = distances.min()
                    
                    # Determine which pairing region
                    current_region = None
                    for region_name, (start, end) in pairing_regions.items():
                        if start <= min_dist < end:
                            current_region = region_name
                            break
                    
                    # Store region
                    if ion_id not in ion_region_history:
                        ion_region_history[ion_id] = {}
                    ion_region_history[ion_id][frame_idx] = current_region
            
            # Analyze residence times from occupancy history
            region_residence_times = {region_name: [] for region_name in pairing_regions.keys()}
            
            print(f"  Analyzing residence times for {len(ion_region_history)} ions...")
            
            for ion_id, frame_history in ion_region_history.items():
                sorted_frames = sorted(frame_history.keys())
                
                if not sorted_frames:
                    continue
                
                # Track consecutive residence events
                current_region = frame_history[sorted_frames[0]]
                residence_start = sorted_frames[0]
                
                for i in range(1, len(sorted_frames)):
                    frame = sorted_frames[i]
                    new_region = frame_history[frame]
                    
                    if new_region != current_region:
                        # Ended a residence event
                        if current_region is not None:
                            residence_duration = frame - residence_start
                            if residence_duration >= min_residence:
                                region_residence_times[current_region].append(residence_duration)
                        
                        # Start new residence event
                        current_region = new_region
                        residence_start = frame
                
                # Handle final residence event
                if current_region is not None:
                    residence_duration = sorted_frames[-1] - residence_start + 1
                    if residence_duration >= min_residence:
                        region_residence_times[current_region].append(residence_duration)
            
            # Calculate statistics
            ion_results = {
                'ion_type': ion_name,
                'ion_category': ion_category,
                'n_ions': len(ions),
                'n_frames_analyzed': len(frame_indices),
                'step': step,
                'regions': {}
            }
            
            for region_name, (region_start, region_end) in pairing_regions.items():
                residence_array = np.array(region_residence_times[region_name])
                
                ion_results['regions'][region_name] = {
                    'bounds': (region_start, region_end),
                    'name': region_name,
                    'residence_times': residence_array,
                    'n_residence_events': len(residence_array),
                    'mean_residence': residence_array.mean() if len(residence_array) > 0 else 0,
                    'median_residence': np.median(residence_array) if len(residence_array) > 0 else 0,
                    'std_residence': residence_array.std() if len(residence_array) > 0 else 0
                }
            
            results[ion_name] = ion_results
        
        # Store results
        self.ion_pairing_residence_times = results
        
        # Print summary
        self._print_ion_pairing_residence_summary(results)
        
        return results



    def _print_ion_pairing_residence_summary(self, results):
        '''Print summary of ion pairing residence times'''
        
        print(f"\n{'='*80}")
        print("ION RESIDENCE TIMES IN PAIRING REGIONS")
        print(f"{'='*80}")
        
        for ion_type, ion_data in results.items():
            print(f"\n{ion_type.upper()} ({ion_data['ion_category']}):")
            print(f"  Analyzed {ion_data['n_ions']} ions over {ion_data['n_frames_analyzed']} frames (step={ion_data['step']})")
            print(f"  Pairing region residence times:")
            
            for region_name in ['CIP', 'SIP', 'DSIP']:
                if region_name in ion_data['regions']:
                    region_data = ion_data['regions'][region_name]
                    bounds = region_data['bounds']
                    
                    print(f"\n    {region_name} ({bounds[0]:.2f}-{bounds[1]:.2f} Å):")
                    print(f"      Residence events: {region_data['n_residence_events']}")
                    
                    if region_data['n_residence_events'] > 0:
                        # FIXED: Use correct key names that match the calculation method
                        print(f"      Mean residence: {region_data['mean_residence']:.2f} ± {region_data['std_residence']:.2f} frames")
                        print(f"      Median residence: {region_data['median_residence']:.2f} frames")
                        
                        # Calculate min/max from residence_times array if available
                        if 'residence_times' in region_data and len(region_data['residence_times']) > 0:
                            min_res = region_data['residence_times'].min()
                            max_res = region_data['residence_times'].max()
                            print(f"      Range: {min_res:.2f} - {max_res:.2f} frames")
                    else:
                        print(f"      No residence events recorded")
        
        print(f"{'='*80}")



    def _plot_residence_distributions_by_category(self, data, category, save_plots):
        '''Plot residence time distributions for one category with interactive zoomed inset'''
        
        n_ions = len(data)
        
        # Get maximum number of shells across all ions
        max_shells = max(len(ion_data['shells']) for ion_data in data.values())
        
        fig, axes = plt.subplots(max_shells, n_ions, figsize=(5*n_ions, 4*max_shells))
        
        if n_ions == 1 and max_shells == 1:
            axes = np.array([[axes]])
        elif n_ions == 1:
            axes = axes.reshape(-1, 1)
        elif max_shells == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{category} Water Residence Time Distributions', fontsize=16, fontweight='bold')
        
        # Store axes and data for interactive editing
        self._residence_plot_data = {
            'fig': fig,
            'axes': axes,
            'data': data,
            'category': category,
            'n_ions': n_ions,
            'max_shells': max_shells,
            'insets': {}  # Will store inset axes
        }
        
        for col, (ion_name, ion_data) in enumerate(data.items()):
            shell_names = sorted(ion_data['shells'].keys())
            
            for row, shell_name in enumerate(shell_names):
                ax = axes[row][col]
                
                shell_data = ion_data['shells'][shell_name]
                
                if shell_data['n_residence_events'] > 0:
                    residence_times = shell_data['residence_times']
                    
                    # Plot histogram
                    n, bins, patches = ax.hist(residence_times, bins=50, alpha=0.7, 
                                            color='steelblue' if category == 'Cation' else 'crimson')
                    
                    ax.set_xlabel('Residence Time (frames)', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    ax.set_title(f'{ion_name} {shell_name}\n({shell_data["bounds"][0]:.2f}-{shell_data["bounds"][1]:.2f} Å)',
                            fontweight='bold', fontsize=11)
                    ax.grid(True, alpha=0.3, axis='y')  # Only horizontal grid
                    
                    # Store histogram data for later use
                    self._residence_plot_data['insets'][(row, col)] = {
                        'ax': ax,
                        'residence_times': residence_times,
                        'ion_name': ion_name,
                        'shell_name': shell_name,
                        'color': 'steelblue' if category == 'Cation' else 'crimson',
                        'n': n,
                        'bins': bins,
                        'inset_ax': None,
                        'rect_patch': None
                    }
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{ion_name} {shell_name}', fontweight='bold')
            
            # Hide unused subplots
            for row in range(len(shell_names), max_shells):
                ax = axes[row][col]
                ax.set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'{category.lower()}_water_residence_distributions.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {filename}")
        
        plt.show()
        
        # Offer interactive zoom editor
        print("\n" + "="*80)
        print("INTERACTIVE ZOOM INSET EDITOR")
        print("="*80)
        print("Add zoomed-in insets to highlight low-frequency, long residence time regions.")
        print("Use: eq_opt.add_residence_inset(row, col, x_min, x_max, y_min, y_max)")
        print(f"Available plots: rows 0-{max_shells-1}, cols 0-{n_ions-1}")
        print("="*80)




    def _plot_pairing_residence_by_category(self, data, category, save_plots):
        '''Plot pairing residence time distributions for one category with interactive zoomed inset'''
        
        n_ions = len(data)
        
        # Get number of regions being analyzed
        n_regions = max(len(ion_data['regions']) for ion_data in data.values())
        
        fig, axes = plt.subplots(n_regions, n_ions, figsize=(5*n_ions, 4*n_regions))
        
        if n_ions == 1 and n_regions == 1:
            axes = np.array([[axes]])
        elif n_ions == 1:
            axes = axes.reshape(-1, 1)
        elif n_regions == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{category} Ion Pairing Residence Time Distributions', fontsize=16, fontweight='bold')
        
        # Store axes and data for interactive editing
        self._pairing_residence_plot_data = {
            'fig': fig,
            'axes': axes,
            'data': data,
            'category': category,
            'n_ions': n_ions,
            'n_regions': n_regions,
            'insets': {}
        }
        
        # Get actual region names from data
        all_region_names = sorted(set(
            region_name 
            for ion_data in data.values() 
            for region_name in ion_data['regions'].keys()
        ))
        
        colors = {'CIP': 'lightcoral', 'SIP': 'lightblue', 'DSIP': 'lightgreen'}
        
        for col, (ion_name, ion_data) in enumerate(data.items()):
            for row, region_name in enumerate(all_region_names):
                if region_name in ion_data['regions']:
                    ax = axes[row, col]
                    region_data = ion_data['regions'][region_name]
                    
                    residence_times = region_data['residence_times']
                    
                    if len(residence_times) > 0:
                        # FIXED: Changed from default to density=True for probability density
                        color = colors.get(region_name, 'lightgray')
                        ax.hist(residence_times, bins=30, alpha=0.7, color=color, 
                            edgecolor='black', linewidth=0.5, density=True)
                        
                        # Shell info in title
                        start, end = region_data['bounds']
                        end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                        
                        ax.set_title(f'{ion_name} {region_data["name"]}\n({start:.2f}-{end_str} Å)', 
                                fontsize=10, fontweight='bold')
                        ax.set_xlabel('Residence Time (frames)', fontsize=9)
                        ax.set_ylabel('Probability Density', fontsize=9)  # FIXED: Updated label
                        ax.grid(True, alpha=0.3, axis='y')
                    else:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
                        ax.set_title(f'{ion_name} {region_name}', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'{category.lower()}_ion_pairing_residence_distributions.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {filename}")
        
        plt.show()
        
        # Offer interactive zoom editor
        print("\n" + "="*80)
        print("INTERACTIVE ZOOM INSET EDITOR (ION PAIRING)")
        print("="*80)
        print("Add zoomed-in insets to highlight low-frequency, long residence time regions.")
        print("Use: eq_opt.add_pairing_residence_inset(row, col, x_min, x_max, y_min, y_max)")
        print(f"Available plots: rows 0-{n_regions-1}, cols 0-{n_ions-1}")
        print("="*80)


    def interactive_residence_inset_editor(self, plot_type='water_residence', inset_size='35%'):
        '''
        Interactive editor for adding zoomed insets to residence time plots.
        FIXED: replot now regenerates the entire figure from scratch so it displays properly
        '''
        
        # Check if we have the plot data
        if plot_type == 'water_residence':
            if not hasattr(self, '_residence_plot_data'):
                print("No water residence plot data available.")
                print("Run plot_water_residence_distributions() first with combined_layout=True")
                return
            plot_data = self._residence_plot_data
        else:  # ion_pairing_residence
            if not hasattr(self, '_pairing_residence_plot_data'):
                print("No ion pairing residence plot data available.")
                print("Run plot_ion_pairing_residence_distributions() first")
                return
            plot_data = self._pairing_residence_plot_data
        
        print(f"\n{'='*70}")
        print(f"INTERACTIVE RESIDENCE INSET EDITOR - {plot_type.upper()}")
        print(f"{'='*70}")
        print("Commands:")
        print("  list                                         - Show available subplots")
        print("  add <row> <col> <x_min> <x_max> [y_min] [y_max] - Add/update inset")
        print("     Example: add 0 0 500 1000 0 10")
        print("  remove <row> <col>                           - Remove inset")
        print("  size <percentage>                            - Change inset size (e.g., 'size 50%')")
        print("  replot                                       - Regenerate plot with current insets")
        print("  save                                         - Save current figure")
        print("  quit                                         - Exit editor")
        print(f"Current inset size: {inset_size}")
        print()
        
        # Store inset configurations
        inset_configs = {}
        current_size = inset_size
        
        while True:
            command = input(f"\n[{plot_type} insets] > ").strip().lower()
            
            if command == 'quit':
                print(f"\nInset editor closed. {len(inset_configs)} insets configured.")
                break
                
            elif command == 'list':
                # Display available subplots
                fig = plot_data['fig']
                axes = plot_data['axes']
                data = plot_data['data']
                
                if plot_data.get('layout') == 'combined':
                    n_ions = plot_data['n_ions']
                    max_shells = plot_data['max_shells']
                    
                    print(f"\nAvailable subplots (combined layout):")
                    print(f"  Grid: {max_shells} rows × {n_ions} columns")
                    print(f"  Ion types: {list(data.keys())}")
                    print()
                    
                    ion_names = list(data.keys())
                    
                    for row in range(max_shells):
                        for col, ion_name in enumerate(ion_names):
                            ion_data = data[ion_name]
                            shell_names = sorted(ion_data['shells'].keys())
                            
                            if row < len(shell_names):
                                shell_name = shell_names[row]
                                shell_data = ion_data['shells'][shell_name]
                                status = "✓ with inset" if (row, col) in inset_configs else "○ available"
                                print(f"  [{row},{col}]: {ion_name} {shell_data['name']} {status}")
                            else:
                                print(f"  [{row},{col}]: (empty)")
            
            elif command.startswith('size '):
                try:
                    new_size = command.split()[1]
                    if '%' not in new_size:
                        new_size = f"{new_size}%"
                    current_size = new_size
                    print(f"Inset size changed to: {current_size}")
                except:
                    print("Invalid size format. Use: size 50% or size 50")
            
            elif command == 'replot':
                # CRITICAL FIX: Regenerate the entire plot from scratch
                print("Regenerating plot with current insets...")
                self._regenerate_residence_plot_with_insets(plot_type, inset_configs, current_size)
                
            elif command == 'save':
                fig = plot_data['fig']
                filename = f'{plot_type}_with_insets.png'
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved figure to: {filename}")
                    
            elif command.startswith('add '):
                try:
                    parts = command.split()
                    row = int(parts[1])
                    col = int(parts[2])
                    x_min = float(parts[3])
                    x_max = float(parts[4])
                    
                    # Optional y limits
                    if len(parts) >= 7:
                        y_min = float(parts[5])
                        y_max = float(parts[6])
                    else:
                        y_min = None
                        y_max = None
                    
                    # Validate row, col
                    axes = plot_data['axes']
                    
                    if axes.ndim == 2:
                        if row >= axes.shape[0] or col >= axes.shape[1]:
                            print(f"Error: Invalid subplot position [{row},{col}]")
                            print(f"Grid size: {axes.shape[0]} rows × {axes.shape[1]} columns")
                            continue
                        
                        target_ax = axes[row, col]
                    else:
                        print("Error: Unexpected axes structure")
                        continue
                    
                    # Check if subplot is visible
                    if not target_ax.get_visible():
                        print(f"Error: Subplot at [{row},{col}] is not visible (no data)")
                        continue
                    
                    # Store config
                    inset_configs[(row, col)] = {
                        'x_min': x_min,
                        'x_max': x_max,
                        'y_min': y_min,
                        'y_max': y_max
                    }
                    
                    print(f"Inset configured for [{row},{col}]: x=[{x_min},{x_max}], y=[{y_min},{y_max}]")
                    print("Use 'replot' to see changes")
                    
                except Exception as e:
                    print(f"Error adding inset: {e}")
                    print("Usage: add <row> <col> <x_min> <x_max> [y_min] [y_max]")
                    
            elif command.startswith('remove '):
                try:
                    parts = command.split()
                    row = int(parts[1])
                    col = int(parts[2])
                    
                    if (row, col) in inset_configs:
                        del inset_configs[(row, col)]
                        print(f"Removed inset at [{row},{col}]")
                        print("Use 'replot' to see changes")
                    else:
                        print(f"No inset at [{row},{col}]")
                        
                except:
                    print("Usage: remove <row> <col>")
                    
            elif command == 'help':
                print("Commands: list, add, remove, size, replot, save, quit")
                
            else:
                print("Unknown command. Type 'help' for available commands.")


    def _regenerate_residence_plot_with_insets(self, plot_type, inset_configs, inset_size):
        '''
        Regenerate the entire residence time plot from scratch with insets.
        This ensures the plot displays properly after modifications.
        '''
        
        if plot_type == 'water_residence':
            plot_data = self._residence_plot_data
        else:
            plot_data = self._pairing_residence_plot_data
        
        data = plot_data['data']
        n_ions = plot_data['n_ions']
        max_shells = plot_data['max_shells']
        
        print(f"Regenerating plot with {len(inset_configs)} insets...")
        
        # Close old figure
        plt.close(plot_data['fig'])
        
        # Create NEW figure from scratch
        fig, axes = plt.subplots(max_shells, n_ions, figsize=(5*n_ions, 4*max_shells))
        
        # Handle single subplot cases
        if n_ions == 1 and max_shells == 1:
            axes = np.array([[axes]])
        elif n_ions == 1:
            axes = axes.reshape(-1, 1)
        elif max_shells == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Water Residence Time Distributions', fontsize=16, fontweight='bold')
        
        # Determine colors based on ion category
        cation_types_in_system = set()
        if hasattr(self, '_get_unique_ion_types'):
            cation_types_in_system = set(self._get_unique_ion_types(self.cations).keys())
        
        # Plot each ion type
        for col, (ion_name, ion_data) in enumerate(data.items()):
            shell_names = sorted(ion_data['shells'].keys())
            
            # Determine color scheme
            if ion_name in cation_types_in_system:
                color_scheme = plt.cm.Blues
            else:
                color_scheme = plt.cm.Reds
            
            for row, shell_name in enumerate(shell_names):
                ax = axes[row, col]
                shell_data = ion_data['shells'][shell_name]
                
                residence_times = shell_data['residence_times']
                
                if len(residence_times) > 0:
                    # Plot histogram
                    color = color_scheme(0.6)
                    ax.hist(residence_times, bins=30, alpha=0.7, color=color, 
                        edgecolor='black', linewidth=0.5, density=True)
                    
                    # Shell info in title
                    start, end = shell_data['bounds']
                    end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                    
                    ax.set_title(f'{ion_name} {shell_data["name"]}\n'
                            f'({start:.2f}-{end_str} Å)', 
                            fontsize=10, fontweight='bold')
                    ax.set_xlabel('Residence Time (frames)', fontsize=9)
                    ax.set_ylabel('Probability Density', fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # ADD INSET if configured for this subplot
                    if (row, col) in inset_configs:
                        config = inset_configs[(row, col)]
                        
                        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                        axins = inset_axes(ax, width=inset_size, height=inset_size, loc='upper right')
                        
                        # Re-plot histogram in inset
                        axins.hist(residence_times, bins=30, alpha=0.7, color=color, 
                                edgecolor='black', linewidth=0.5, density=True)
                        
                        # Set limits
                        axins.set_xlim(config['x_min'], config['x_max'])
                        
                        if config['y_min'] is not None and config['y_max'] is not None:
                            axins.set_ylim(config['y_min'], config['y_max'])
                        
                        axins.grid(True, alpha=0.3, axis='y')
                        axins.tick_params(labelsize=8)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{ion_name} {shell_data["name"]}', 
                            fontsize=10, fontweight='bold')
            
            # Hide unused subplots
            for row in range(len(shell_names), max_shells):
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Update stored plot data
        plot_data['fig'] = fig
        plot_data['axes'] = axes
        
        # NOW show the plot - this will work because it's a fresh figure
        plt.show()
        
        print(f"✓ Plot regenerated with {len(inset_configs)} insets") 




    def save_all_residence_times_to_file(self, filename='all_residence_times_cache.pkl', residence_types=['water', 'ion_pairing'], ion_types=None):
        '''
        Save ALL residence times data (both water and ion pairing) for specified or all ion types in a single file.
        FIXED: Now safely handles missing statistical keys
        
        Parameters
        ----------
        filename : str
            Output filename, default='all_residence_times_cache.pkl'
        residence_types : list
            Types to save: ['water', 'ion_pairing'], default=both
        ion_types : str, list, or None
            Specific ion type(s) to save:
            - None: Save ALL ion types (default)
            - 'all': Same as None, saves all ion types
            - Single string: e.g., 'Na' - saves only this ion
            - List: e.g., ['Na', 'Cl'] - saves only these ions
        
        Returns
        -------
        success : bool
            True if save was successful
        '''
        
        # Handle ion_types parameter
        if ion_types == 'all':
            ion_types = None  # Treat 'all' same as None
        elif isinstance(ion_types, str):
            ion_types = [ion_types]  # Convert single string to list
        
        data_to_save = {
            'metadata': {
                'saved_date': datetime.now().isoformat(),
                'n_frames': self.n_frames,
                'trajectory_length': len(self.universe.trajectory),
                'ion_types_saved': ion_types if ion_types else 'all'
            }
        }
        
        saved_count = 0
        
        # Save water residence times if requested
        if 'water' in residence_types and hasattr(self, 'water_residence_times') and self.water_residence_times:
            water_residence_data = {}
            
            for ion_type, ion_data in self.water_residence_times.items():
                # Filter by ion_types if specified
                if ion_types is not None and ion_type not in ion_types:
                    continue
                
                water_residence_data[ion_type] = {
                    'ion_type': ion_data['ion_type'],
                    'ion_category': ion_data['ion_category'],
                    'n_ions': ion_data['n_ions'],
                    'n_frames_analyzed': ion_data['n_frames_analyzed'],
                    'step': ion_data['step'],
                    'shells': {}
                }
                
                for shell_name, shell_data in ion_data['shells'].items():
                    water_residence_data[ion_type]['shells'][shell_name] = {
                        'bounds': shell_data['bounds'],
                        # FIXED: Use shell_name as name if 'name' key doesn't exist
                        'name': shell_data.get('name', shell_name),
                        'residence_times': shell_data['residence_times'].copy(),
                        'n_residence_events': shell_data['n_residence_events'],
                        # FIXED: Use .get() with defaults for statistical keys
                        'mean_residence': shell_data.get('mean_residence', np.mean(shell_data['residence_times']) if len(shell_data['residence_times']) > 0 else 0.0),
                        'median_residence': shell_data.get('median_residence', np.median(shell_data['residence_times']) if len(shell_data['residence_times']) > 0 else 0.0),
                        'std_residence': shell_data.get('std_residence', np.std(shell_data['residence_times']) if len(shell_data['residence_times']) > 0 else 0.0)
                    }
            
            if water_residence_data:
                data_to_save['water_residence'] = water_residence_data
                saved_count += len(water_residence_data)
                print(f"  Prepared water residence data for {len(water_residence_data)} ion types")
        
        # Save ion pairing residence times if requested
        if 'ion_pairing' in residence_types and hasattr(self, 'ion_pairing_residence_times') and self.ion_pairing_residence_times:
            pairing_residence_data = {}
            
            for ion_type, ion_data in self.ion_pairing_residence_times.items():
                # Filter by ion_types if specified
                if ion_types is not None and ion_type not in ion_types:
                    continue
                
                pairing_residence_data[ion_type] = {
                    'ion_type': ion_data['ion_type'],
                    'ion_category': ion_data['ion_category'],
                    'n_ions': ion_data['n_ions'],
                    'n_frames_analyzed': ion_data['n_frames_analyzed'],
                    'step': ion_data['step'],
                    'regions': {}
                }
                
                for region_name, region_data in ion_data['regions'].items():
                    pairing_residence_data[ion_type]['regions'][region_name] = {
                        'bounds': region_data['bounds'],
                        # FIXED: Use region_name as name if 'name' key doesn't exist
                        'name': region_data.get('name', region_name),
                        'residence_times': region_data['residence_times'].copy(),
                        'n_residence_events': region_data['n_residence_events'],
                        # FIXED: Use .get() with defaults for statistical keys
                        'mean_residence': region_data.get('mean_residence', np.mean(region_data['residence_times']) if len(region_data['residence_times']) > 0 else 0.0),
                        'median_residence': region_data.get('median_residence', np.median(region_data['residence_times']) if len(region_data['residence_times']) > 0 else 0.0),
                        'std_residence': region_data.get('std_residence', np.std(region_data['residence_times']) if len(region_data['residence_times']) > 0 else 0.0)
                    }
            
            if pairing_residence_data:
                data_to_save['ion_pairing_residence'] = pairing_residence_data
                saved_count += len(pairing_residence_data)
                print(f"  Prepared ion pairing residence data for {len(pairing_residence_data)} ion types")
        
        if saved_count == 0:
            print("No residence time data to save")
            return False
        
        try:
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            
            print(f"\n✓ Residence times saved to {filename}")
            print(f"  File size: {file_size_mb:.1f} MB")
            
            if ion_types is None:
                print(f"  Saved ALL ion types: {saved_count} total")
            else:
                print(f"  Saved specific ion types: {ion_types}")
            
            if 'water_residence' in data_to_save:
                print(f"  Water residence: {list(data_to_save['water_residence'].keys())}")
            if 'ion_pairing_residence' in data_to_save:
                print(f"  Ion pairing residence: {list(data_to_save['ion_pairing_residence'].keys())}")
            
            return True
            
        except Exception as e:
            print(f"Error saving residence times: {e}")
            traceback.print_exc()
            return False



    def load_all_residence_times_from_file(self, filename='all_residence_times_cache.pkl', ion_types=None):
        '''
        Load residence times data from a single file with optional ion type filtering.
        
        Parameters
        ----------
        filename : str
            Input filename, default='all_residence_times_cache.pkl'
        ion_types : str, list, or None
            Specific ion type(s) to load:
            - None: Load ALL ion types from file (default)
            - 'all': Same as None
            - Single string: e.g., 'Na' - loads only this ion
            - List: e.g., ['Na', 'Cl'] - loads only these ions
        
        Returns
        -------
        success : bool
            True if load was successful
            
        Examples
        --------
        # Load all ions:
        eq_opt.load_all_residence_times_from_file()
        
        # Load only Na:
        eq_opt.load_all_residence_times_from_file(ion_types='Na')
        
        # Load Na and Cl only:
        eq_opt.load_all_residence_times_from_file(ion_types=['Na', 'Cl'])
        '''
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        # Handle ion_types parameter
        if ion_types == 'all':
            ion_types = None
        elif isinstance(ion_types, str):
            ion_types = [ion_types]
        
        try:
            # Check file size first
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print(f"Cache file {filename} is empty")
                return False
            
            file_size_mb = file_size / (1024 * 1024)
            print(f"Loading residence times from {filename} ({file_size_mb:.1f} MB)...")
            
            # Load data
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(data, dict):
                print(f"Invalid cache format")
                return False
            
            loaded_count = 0
            
            # Load water residence times
            if 'water_residence' in data:
                # Filter by ion_types if specified
                if ion_types is not None:
                    filtered_data = {k: v for k, v in data['water_residence'].items() if k in ion_types}
                    if not filtered_data:
                        print(f"  No water residence data found for specified ion types: {ion_types}")
                        print(f"  Available in file: {list(data['water_residence'].keys())}")
                else:
                    filtered_data = data['water_residence']
                
                self.water_residence_times = filtered_data
                loaded_count += len(self.water_residence_times)
                print(f"  ✓ Loaded water residence for {len(self.water_residence_times)} ion types")
                print(f"    Ion types: {list(self.water_residence_times.keys())}")
            
            # Load ion pairing residence times
            if 'ion_pairing_residence' in data:
                # Filter by ion_types if specified
                if ion_types is not None:
                    filtered_data = {k: v for k, v in data['ion_pairing_residence'].items() if k in ion_types}
                    if not filtered_data:
                        print(f"  No ion pairing residence data found for specified ion types: {ion_types}")
                        print(f"  Available in file: {list(data['ion_pairing_residence'].keys())}")
                else:
                    filtered_data = data['ion_pairing_residence']
                
                self.ion_pairing_residence_times = filtered_data
                loaded_count += len(self.ion_pairing_residence_times)
                print(f"  ✓ Loaded ion pairing residence for {len(self.ion_pairing_residence_times)} ion types")
                print(f"    Ion types: {list(self.ion_pairing_residence_times.keys())}")
            
            if loaded_count == 0:
                print("No residence time data found in file")
                return False
            
            if ion_types is None:
                print(f"\n✓ Successfully loaded residence times for ALL ion types ({loaded_count} total)")
            else:
                print(f"\n✓ Successfully loaded residence times for specified ion types: {ion_types}")
            
            return True
            
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            print("Removing corrupted cache file...")
            try:
                os.remove(filename)
                print(f"Corrupted file {filename} removed")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading residence times from {filename}: {e}")
            traceback.print_exc()
            return False




    def calculate_residence_times_with_cache(self, cache_filename='residence_times_cache.pkl', 
                                            residence_type='water', force_recalc=False, **kwargs):
        '''
        Calculate residence times with automatic caching.
        
        Parameters
        ----------
        cache_filename : str
            Cache file name, default='residence_times_cache.pkl'
        residence_type : str
            Type of residence data: 'water' or 'ion_pairing'
        force_recalc : bool
            Force recalculation even if cache exists, default=False
        **kwargs : dict
            Arguments to pass to calculation method
        
        Returns
        -------
        results : dict
            Dictionary of residence time results
        '''
        
        # Try to load from cache first
        if not force_recalc and os.path.exists(cache_filename):
            print(f"Attempting to load {residence_type} residence times from cache...")
            if self.load_residence_times_from_file(cache_filename, residence_type):
                print(f"✓ Successfully loaded {residence_type} residence times from cache")
                if residence_type == 'water':
                    return self.water_residence_times
                else:
                    return self.ion_pairing_residence_times
            else:
                print("✗ Cache loading failed, will recalculate")
        
        # Calculate residence times
        print(f"Calculating {residence_type} residence times...")
        
        if residence_type == 'water':
            results = self.calculate_water_residence_times_in_shells(**kwargs)
        elif residence_type == 'ion_pairing':
            results = self.calculate_ion_residence_times_in_pairing_regions(**kwargs)
        else:
            print(f"Invalid residence_type: {residence_type}")
            return None
        
        # Save to cache
        if results:
            print(f"Saving {residence_type} residence times to cache...")
            if self.save_residence_times_to_file(cache_filename, residence_type):
                print(f"✓ {residence_type.title()} residence times cached successfully")
            else:
                print(f"✗ Cache saving failed, but results are available in memory")
        
        return results


    def save_residence_plot_with_insets(self, plot_type='water_residence', filename=None, dpi=300, save_data=True):
        '''
        Save the residence time distribution plot with any applied insets.
        Call this after using interactive_residence_inset_editor to save final version.
        
        Parameters
        ----------
        plot_type : str
            Type of residence plot: 'water_residence' or 'ion_pairing_residence'
        filename : str, optional
            Output filename for plot. If None, auto-generates based on plot type
        dpi : int
            Resolution for saved figure (default: 300)
        save_data : bool
            If True, also saves the underlying data to a pickle file (default: True)
            
        Returns
        -------
        bool
            True if successful, False otherwise
        '''
        
        # Check if we have the plot data
        if plot_type == 'water_residence':
            if not hasattr(self, '_residence_plot_data'):
                print("No water residence plot data available.")
                print("Run plot_water_residence_distributions() first")
                return False
            plot_data = self._residence_plot_data
            default_filename = 'water_residence_distributions_final.png'
        elif plot_type == 'ion_pairing_residence':
            if not hasattr(self, '_pairing_residence_plot_data'):
                print("No ion pairing residence plot data available.")
                print("Run plot_ion_pairing_residence_distributions() first")
                return False
            plot_data = self._pairing_residence_plot_data
            default_filename = 'ion_pairing_residence_distributions_final.png'
        else:
            print(f"Unknown plot type: {plot_type}")
            print("Use 'water_residence' or 'ion_pairing_residence'")
            return False
        
        # Use provided filename or default
        if filename is None:
            filename = default_filename
        
        # Get the figure
        fig = plot_data['fig']
        
        # Get ion types for filename
        data = plot_data['data']
        ion_types_str = '_'.join(data.keys())
        
        # Update filename with ion types if using default
        if filename == default_filename:
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{ion_types_str}{ext}"
        
        try:
            # Save the figure
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            
            print(f"\n{'='*70}")
            print(f"RESIDENCE PLOT SAVED")
            print(f"{'='*70}")
            print(f"  Plot filename: {filename}")
            print(f"  Plot type: {plot_type}")
            print(f"  Ion types: {list(data.keys())}")
            print(f"  Resolution: {dpi} DPI")
            
            # Check if insets were applied
            axes = plot_data['axes']
            n_insets = 0
            for row in range(axes.shape[0]):
                for col in range(axes.shape[1]):
                    ax = axes[row, col]
                    if hasattr(ax, '_inset_ax') and ax._inset_ax is not None:
                        n_insets += 1
            
            if n_insets > 0:
                print(f"  Insets applied: {n_insets}")
            else:
                print(f"  Insets applied: None")
            
            # Save underlying data if requested using your EXISTING method
            if save_data:
                # Generate data filename from plot filename
                base, ext = os.path.splitext(filename)
                data_filename = f"{base}_data.pkl"
                
                print(f"\n  Saving underlying data...")
                
                # FIXED: Use your existing save_all_residence_times_to_file method
                # Determine which residence types to save
                if plot_type == 'water_residence':
                    residence_types = ['water']
                else:
                    residence_types = ['ion_pairing']
                
                # Get ion types from plot data
                ion_types_list = list(data.keys())
                
                success = self.save_all_residence_times_to_file(
                    filename=data_filename,
                    residence_types=residence_types,
                    ion_types=ion_types_list
                )
                
                if success:
                    print(f"  Data filename: {data_filename}")
            
            print(f"{'='*70}")
            
            return True
            
        except Exception as e:
            print(f"Error saving figure: {e}")
            return False



    def get_shell_occupancy_probabilities_by_type(self, step=None, use_kdtree=True):
        '''
        Calculate shell occupancy probabilities for each ion type separately.
        
        Parameters
        ----------
        step : int
            Step size for trajectory analysis, default uses auto-tuned value
        use_kdtree : bool
            Whether to use KDTree for faster calculations, default=True
        
        Returns
        -------
        probabilities : dict
            Nested dictionary with ion types and their shell probability data
        '''
        
        if step is None:
            step = self.default_step
        
        # Check if ion-type-specific shells are available
        if not (hasattr(self, 'cation_shells_by_type') and hasattr(self, 'anion_shells_by_type')):
            print('Ion-type-specific shells not determined. Run determine_ion_solvation_shells_by_type() first')
            return None
        
        # Get unique ion types
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        n_frames = len(self.universe.trajectory[::step])
        results = {}
        
        print(f'Calculating shell occupancy probabilities by ion type (step={step})...')
        method_str = 'KDTree' if use_kdtree else 'vectorized distance matrices'
        print(f'  Using {method_str} for calculations')
        
        # Process each cation type
        for cation_name, cation_group in cation_types.items():
            if (cation_name in self.cation_shells_by_type and 
                self.cation_shells_by_type[cation_name] is not None):
                
                shells = self.cation_shells_by_type[cation_name]
                results[cation_name] = {'type': 'cation', 'shells': {}}
                
                print(f'  Processing {cation_name} shell probabilities ({len(cation_group)} ions)...')
                
                # Initialize occupancy arrays for each shell (binary: occupied=1, not occupied=0)
                for shell_name, (start, end) in shells.data.items():
                    if shell_name != 'bulk':  # Skip bulk for probability calculations
                        results[cation_name]['shells'][shell_name] = {
                            'bounds': (start, end),
                            'occupancy': np.zeros((len(cation_group), n_frames)),  # Per ion, per frame
                            'name': shell_name.replace('_', ' ').title()
                        }
                
                # Calculate occupancy for this cation type
                for i, ts in enumerate(tqdm(self.universe.trajectory[::step], 
                                        desc=f"Calculating {cation_name} shell occupancy", leave=False)):
                    
                    if use_kdtree:
                        # Use KDTree for efficient neighbor searches
                        water_tree = cKDTree(self.waters.positions)
                        
                        for ion_idx, cation_pos in enumerate(cation_group.positions):
                            for shell_name, shell_data in results[cation_name]['shells'].items():
                                start, end = shell_data['bounds']
                                
                                if start == 0:
                                    waters_in_shell = len(water_tree.query_ball_point(cation_pos, end))
                                else:
                                    waters_in_end = len(water_tree.query_ball_point(cation_pos, end))
                                    waters_in_start = len(water_tree.query_ball_point(cation_pos, start))
                                    waters_in_shell = waters_in_end - waters_in_start
                                
                                # Binary occupancy: 1 if any water in shell, 0 if none
                                shell_data['occupancy'][ion_idx, i] = 1 if waters_in_shell > 0 else 0
                    
                    else:
                        # Use vectorized distance matrix
                        d_cat = cdist(cation_group.positions, self.waters.positions)
                        
                        for ion_idx in range(len(cation_group)):
                            ion_distances = d_cat[ion_idx]
                            
                            for shell_name, shell_data in results[cation_name]['shells'].items():
                                start, end = shell_data['bounds']
                                
                                waters_in_shell = np.sum((ion_distances >= start) & (ion_distances < end))
                                shell_data['occupancy'][ion_idx, i] = 1 if waters_in_shell > 0 else 0
        
        # Process each anion type  
        for anion_name, anion_group in anion_types.items():
            if (anion_name in self.anion_shells_by_type and 
                self.anion_shells_by_type[anion_name] is not None):
                
                shells = self.anion_shells_by_type[anion_name]
                results[anion_name] = {'type': 'anion', 'shells': {}}
                
                print(f'  Processing {anion_name} shell probabilities ({len(anion_group)} ions)...')
                
                # Initialize occupancy arrays for each shell
                for shell_name, (start, end) in shells.data.items():
                    if shell_name != 'bulk':  # Skip bulk for probability calculations
                        results[anion_name]['shells'][shell_name] = {
                            'bounds': (start, end),
                            'occupancy': np.zeros((len(anion_group), n_frames)),  # Per ion, per frame
                            'name': shell_name.replace('_', ' ').title()
                        }
                
                # Calculate occupancy for this anion type
                for i, ts in enumerate(tqdm(self.universe.trajectory[::step], 
                                        desc=f"Calculating {anion_name} shell occupancy", leave=False)):
                    
                    if use_kdtree:
                        # Use KDTree for efficient neighbor searches
                        water_tree = cKDTree(self.waters.positions)
                        
                        for ion_idx, anion_pos in enumerate(anion_group.positions):
                            for shell_name, shell_data in results[anion_name]['shells'].items():
                                start, end = shell_data['bounds']
                                
                                if start == 0:
                                    waters_in_shell = len(water_tree.query_ball_point(anion_pos, end))
                                else:
                                    waters_in_end = len(water_tree.query_ball_point(anion_pos, end))
                                    waters_in_start = len(water_tree.query_ball_point(anion_pos, start))
                                    waters_in_shell = waters_in_end - waters_in_start
                                
                                shell_data['occupancy'][ion_idx, i] = 1 if waters_in_shell > 0 else 0
                    
                    else:
                        # Use vectorized distance matrix
                        d_an = cdist(anion_group.positions, self.waters.positions)
                        
                        for ion_idx in range(len(anion_group)):
                            ion_distances = d_an[ion_idx]
                            
                            for shell_name, shell_data in results[anion_name]['shells'].items():
                                start, end = shell_data['bounds']
                                
                                waters_in_shell = np.sum((ion_distances >= start) & (ion_distances < end))
                                shell_data['occupancy'][ion_idx, i] = 1 if waters_in_shell > 0 else 0
        
        # Calculate probability statistics for all shells
        for ion_type in results.keys():
            for shell_name, shell_data in results[ion_type]['shells'].items():
                occupancy_data = shell_data['occupancy']
                
                # Calculate probabilities
                shell_data['probability_per_ion'] = occupancy_data.mean(axis=1)  # Probability for each ion
                shell_data['mean_probability'] = occupancy_data.mean()  # Overall mean probability
                shell_data['std_probability'] = occupancy_data.std()   # Standard deviation
                shell_data['probability_distribution'] = occupancy_data.mean(axis=0)  # Probability vs time
                
                # Calculate additional statistics
                shell_data['always_occupied_ions'] = np.sum(shell_data['probability_per_ion'] == 1.0)
                shell_data['never_occupied_ions'] = np.sum(shell_data['probability_per_ion'] == 0.0)
                shell_data['partially_occupied_ions'] = np.sum((shell_data['probability_per_ion'] > 0.0) & 
                                                            (shell_data['probability_per_ion'] < 1.0))
        
        # Store results
        self.shell_probabilities_by_type = results
        
        # Print summary
        self._print_shell_probability_summary_by_type(results)
        
        return results


    def save_shell_occupancy_probabilities_to_file(self, filename='shell_occupancy_probs_cache.pkl'):
        '''
        Save shell occupancy probabilities to file for persistence across sessions.
        
        Parameters
        ----------
        filename : str
            Output filename, default='shell_occupancy_probs_cache.pkl'
        
        Returns
        -------
        success : bool
            True if save was successful
        '''
        
        if not hasattr(self, 'shell_probabilities_by_type') or not self.shell_probabilities_by_type:
            print("No shell occupancy probabilities to save")
            return False
        
        try:
            # Prepare shell occupancy data for serialization
            occupancy_data = {}
            
            for ion_type, ion_data in self.shell_probabilities_by_type.items():
                occupancy_data[ion_type] = {
                    'type': ion_data['type'],
                    'shells': {}
                }
                
                for shell_name, shell_data in ion_data['shells'].items():
                    occupancy_data[ion_type]['shells'][shell_name] = {
                        'bounds': shell_data['bounds'],
                        'occupancy': shell_data['occupancy'].copy(),
                        'name': shell_data['name'],
                        'probability_per_ion': shell_data['probability_per_ion'].copy(),
                        'mean_probability': shell_data['mean_probability'],
                        'std_probability': shell_data['std_probability'],
                        'probability_distribution': shell_data['probability_distribution'].copy(),
                        'always_occupied_ions': shell_data['always_occupied_ions'],
                        'never_occupied_ions': shell_data['never_occupied_ions'],
                        'partially_occupied_ions': shell_data['partially_occupied_ions']
                    }
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(occupancy_data, f)
            
            print(f"Shell occupancy probabilities saved to {filename}")
            print(f"  Saved {len(occupancy_data)} ion types")
            print(f"  Ion types: {list(occupancy_data.keys())}")
            
            # Print summary
            for ion_type, data in occupancy_data.items():
                n_shells = len(data['shells'])
                print(f"    {ion_type}: {n_shells} shells")
            
            return True
            
        except Exception as e:
            print(f"Error saving shell occupancy probabilities: {e}")
            return False

    def load_shell_occupancy_probabilities_from_file(self, filename='shell_occupancy_probs_cache.pkl'):
        '''
        Load shell occupancy probabilities from file with error handling.
        
        Parameters
        ----------
        filename : str
            Input filename, default='shell_occupancy_probs_cache.pkl'
        
        Returns
        -------
        success : bool
            True if load was successful
        '''
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        try:
            # Check file size first
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print(f"Cache file {filename} is empty")
                return False
            
            # Load data
            with open(filename, 'rb') as f:
                occupancy_data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(occupancy_data, dict):
                print(f"Invalid shell occupancy cache format")
                return False
            
            # Reconstruct the shell occupancy data structure
            self.shell_probabilities_by_type = {}
            
            for ion_type, ion_data in occupancy_data.items():
                self.shell_probabilities_by_type[ion_type] = {
                    'type': ion_data['type'],
                    'shells': {}
                }
                
                for shell_name, shell_data in ion_data['shells'].items():
                    self.shell_probabilities_by_type[ion_type]['shells'][shell_name] = {
                        'bounds': shell_data['bounds'],
                        'occupancy': shell_data['occupancy'],
                        'name': shell_data['name'],
                        'probability_per_ion': shell_data['probability_per_ion'],
                        'mean_probability': shell_data['mean_probability'],
                        'std_probability': shell_data['std_probability'],
                        'probability_distribution': shell_data['probability_distribution'],
                        'always_occupied_ions': shell_data['always_occupied_ions'],
                        'never_occupied_ions': shell_data['never_occupied_ions'],
                        'partially_occupied_ions': shell_data['partially_occupied_ions']
                    }
            
            # Print summary
            successful_types = list(self.shell_probabilities_by_type.keys())
            
            print(f"Shell occupancy probabilities loaded from {filename}")
            print(f"  Loaded {len(successful_types)} ion types successfully")
            if successful_types:
                print(f"  Available types: {', '.join(successful_types)}")
            
            # Print detailed summary
            print(f"\n  Shell occupancy summary:")
            for ion_type, ion_data in self.shell_probabilities_by_type.items():
                shell_count = len(ion_data['shells'])
                print(f"    {ion_type} ({ion_data['type']}): {shell_count} shells")
                for shell_name, shell_data in ion_data['shells'].items():
                    start, end = shell_data['bounds']
                    end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                    mean_prob = shell_data['mean_probability']
                    print(f"      {shell_data['name']}: {start:.2f}-{end_str} Å, P={mean_prob:.3f}")
            
            return True
            
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            print("Removing corrupted cache file...")
            try:
                os.remove(filename)
                print(f"Corrupted file {filename} removed")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading shell occupancy probabilities from {filename}: {e}")
            return False

    def get_shell_occupancy_probabilities_by_type_with_cache(self, cache_filename='shell_occupancy_probs_cache.pkl', 
                                                            force_recalc=False, **kwargs):
        '''
        Calculate shell occupancy probabilities with automatic caching.
        
        Parameters
        ----------
        cache_filename : str
            Cache file name, default='shell_occupancy_probs_cache.pkl'
        force_recalc : bool
            Force recalculation even if cache exists, default=False
        **kwargs : dict
            Arguments to pass to get_shell_occupancy_probabilities_by_type()
        
        Returns
        -------
        results : dict
            Dictionary of shell occupancy probability results
        '''
        
        # Try to load from cache first
        if not force_recalc and os.path.exists(cache_filename):
            print("Attempting to load shell occupancy probabilities from cache...")
            if self.load_shell_occupancy_probabilities_from_file(cache_filename):
                print("✓ Successfully loaded shell occupancy probabilities from cache")
                return self.shell_probabilities_by_type
            else:
                print("✗ Cache loading failed, will recalculate")
        
        # Calculate shell occupancy probabilities
        print("Calculating shell occupancy probabilities...")
        results = self.get_shell_occupancy_probabilities_by_type(**kwargs)
        
        # Save to cache
        if results:
            print("Saving shell occupancy probabilities to cache...")
            if self.save_shell_occupancy_probabilities_to_file(cache_filename):
                print("✓ Shell occupancy probabilities cached successfully")
            else:
                print("✗ Cache saving failed, but results are available in memory")
        
        return results


    def _print_shell_probability_summary_by_type(self, results):
        '''Print summary of shell occupancy probabilities by ion type'''
        
        print("\n" + "="*80)
        print("SHELL OCCUPANCY PROBABILITIES BY ION TYPE")
        print("="*80)
        
        for ion_type, ion_data in results.items():
            ion_class = ion_data['type'].upper()
            print(f"\n{ion_class}: {ion_type}")
            print("-" * 60)
            
            for shell_name, shell_data in ion_data['shells'].items():
                start, end = shell_data['bounds']
                mean_prob = shell_data['mean_probability']
                std_prob = shell_data['std_probability']
                always_occ = shell_data['always_occupied_ions']
                never_occ = shell_data['never_occupied_ions']
                partial_occ = shell_data['partially_occupied_ions']
                
                end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                print(f"{shell_data['name']:12s}: {start:.2f} - {end_str:>6s} Å")
                print(f"    Mean Probability: {mean_prob:.3f} ± {std_prob:.3f}")
                print(f"    Always occupied: {always_occ:2d} ions")
                print(f"    Never occupied:  {never_occ:2d} ions") 
                print(f"    Partial occupy:  {partial_occ:2d} ions")
                print()
        
        print("="*80)

    def plot_shell_probabilities_by_type(self, save_plots=True, plot_range=None, show_individual_ions=False):
        '''
        Plot shell occupancy probabilities for each ion type.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save plots, default=True
        plot_range : tuple
            Range of frames to plot (start, end), default=None (all frames)
        show_individual_ions : bool
            Whether to show individual ion traces, default=False
        '''
        
        if not hasattr(self, 'shell_probabilities_by_type'):
            print("Shell probabilities by type not calculated. Run get_shell_occupancy_probabilities_by_type() first.")
            return
        
        probability_data = self.shell_probabilities_by_type
        
        # Separate cations and anions
        cations_data = {k: v for k, v in probability_data.items() if v['type'] == 'cation'}
        anions_data = {k: v for k, v in probability_data.items() if v['type'] == 'anion'}
        
        n_cations = len(cations_data)
        n_anions = len(anions_data)
        
        if n_cations == 0 and n_anions == 0:
            print("No shell probability data found")
            return
        
        # Get frame indices
        if plot_range is not None:
            start, end = plot_range
            frame_indices = np.arange(start, end)
        else:
            # Use the length from any ion type's first shell
            sample_data = list(probability_data.values())[0]
            sample_shell = list(sample_data['shells'].values())[0]
            frame_indices = np.arange(sample_shell['occupancy'].shape[1])
        
        # Create figure for probability distributions over time
        fig = plt.figure(figsize=(16, 10))
        
        plot_idx = 1
        max_subplots = max(n_cations, n_anions)
        
        # Plot cations
        if n_cations > 0:
            for cation_type, cation_data in cations_data.items():
                ax = plt.subplot(2, max_subplots, plot_idx)
                
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cation_data['shells'])))
                
                for i, (shell_name, shell_data) in enumerate(cation_data['shells'].items()):
                    prob_vs_time = shell_data['probability_distribution']
                    if plot_range is not None:
                        prob_vs_time = prob_vs_time[start:end]
                    
                    # Plot mean probability over time
                    ax.plot(frame_indices, prob_vs_time, 
                        color=colors[i], linewidth=2, alpha=0.8,
                        label=f"{shell_data['name']} (μ={shell_data['mean_probability']:.3f})")
                    
                    # Add horizontal line for mean
                    ax.axhline(shell_data['mean_probability'], color=colors[i], 
                            linestyle='--', alpha=0.5, linewidth=1)
                    
                    # Optionally show individual ion traces
                    if show_individual_ions:
                        for ion_idx in range(shell_data['occupancy'].shape[0]):
                            ion_trace = shell_data['occupancy'][ion_idx, :]
                            if plot_range is not None:
                                ion_trace = ion_trace[start:end]
                            ax.plot(frame_indices, ion_trace, 
                                color=colors[i], alpha=0.1, linewidth=0.5)
                
                ax.set_title(f'{cation_type} Shell Probabilities', fontweight='bold')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Occupancy Probability')
                ax.set_ylim(0, 1.1)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        # Fill remaining cation slots
        while plot_idx <= max_subplots:
            ax = plt.subplot(2, max_subplots, plot_idx)
            ax.set_visible(False)
            plot_idx += 1
        
        # Plot anions
        if n_anions > 0:
            for anion_type, anion_data in anions_data.items():
                ax = plt.subplot(2, max_subplots, plot_idx)
                
                colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(anion_data['shells'])))
                
                for i, (shell_name, shell_data) in enumerate(anion_data['shells'].items()):
                    prob_vs_time = shell_data['probability_distribution']
                    if plot_range is not None:
                        prob_vs_time = prob_vs_time[start:end]
                    
                    # Plot mean probability over time
                    ax.plot(frame_indices, prob_vs_time, 
                        color=colors[i], linewidth=2, alpha=0.8,
                        label=f"{shell_data['name']} (μ={shell_data['mean_probability']:.3f})")
                    
                    # Add horizontal line for mean
                    ax.axhline(shell_data['mean_probability'], color=colors[i], 
                            linestyle='--', alpha=0.5, linewidth=1)
                    
                    # Optionally show individual ion traces
                    if show_individual_ions:
                        for ion_idx in range(shell_data['occupancy'].shape[0]):
                            ion_trace = shell_data['occupancy'][ion_idx, :]
                            if plot_range is not None:
                                ion_trace = ion_trace[start:end]
                            ax.plot(frame_indices, ion_trace, 
                                color=colors[i], alpha=0.1, linewidth=0.5)
                
                ax.set_title(f'{anion_type} Shell Probabilities', fontweight='bold')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Occupancy Probability')
                ax.set_ylim(0, 1.1)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        # Fill remaining anion slots
        while plot_idx <= 2 * max_subplots:
            ax = plt.subplot(2, max_subplots, plot_idx)
            ax.set_visible(False)
            plot_idx += 1
        
        plt.suptitle('Shell Occupancy Probabilities by Ion Type', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('shell_probabilities_by_type.png', dpi=300, bbox_inches='tight')
            print("Plot saved as: shell_probabilities_by_type.png")
        
        plt.show()

    def plot_shell_probabilities_single_ion(self, ion_type, save_plots=True, plot_range=None, show_individual_ions=False):
        '''
        Plot shell occupancy probabilities for a single ion type.
        
        Parameters
        ----------
        ion_type : str
            Specific ion type to plot (e.g., 'Na', 'Mg', 'Cl')
        save_plots : bool
            Whether to save plots, default=True
        plot_range : tuple
            Range of frames to plot (start, end), default=None (all frames)
        show_individual_ions : bool
            Whether to show individual ion traces, default=False
        '''
        
        if not hasattr(self, 'shell_probabilities_by_type'):
            print("Shell probabilities by type not calculated. Run get_shell_occupancy_probabilities_by_type() first.")
            return
        
        if ion_type not in self.shell_probabilities_by_type:
            available_types = list(self.shell_probabilities_by_type.keys())
            print(f"Ion type '{ion_type}' not found. Available types: {available_types}")
            return
        
        ion_data = self.shell_probabilities_by_type[ion_type]
        ion_category = ion_data['type']
        
        # Get frame indices
        if plot_range is not None:
            start, end = plot_range
            frame_indices = np.arange(start, end)
        else:
            # Use the length from first shell
            sample_shell = list(ion_data['shells'].values())[0]
            frame_indices = np.arange(sample_shell['occupancy'].shape[1])
        
        # Create figure with 2 subplots: time series + histograms
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Probability time series
        ax = axes[0]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(ion_data['shells']))) if ion_category == 'cation' else plt.cm.Reds(np.linspace(0.4, 0.9, len(ion_data['shells'])))
        
        for i, (shell_name, shell_data) in enumerate(ion_data['shells'].items()):
            prob_vs_time = shell_data['probability_distribution']
            if plot_range is not None:
                prob_vs_time = prob_vs_time[start:end]
            
            # Plot mean probability over time
            ax.plot(frame_indices, prob_vs_time, 
                color=colors[i], linewidth=2, alpha=0.8,
                label=f"{shell_data['name']} (μ={shell_data['mean_probability']:.3f})")
            
            # Add horizontal line for mean
            ax.axhline(shell_data['mean_probability'], color=colors[i], 
                    linestyle='--', alpha=0.5, linewidth=1)
            
            # Optionally show individual ion traces
            if show_individual_ions:
                for ion_idx in range(shell_data['occupancy'].shape[0]):
                    ion_trace = shell_data['occupancy'][ion_idx, :]
                    if plot_range is not None:
                        ion_trace = ion_trace[start:end]
                    ax.plot(frame_indices, ion_trace, 
                        color=colors[i], alpha=0.1, linewidth=0.5)
        
        ax.set_title(f'{ion_type} Shell Occupancy Probabilities Over Time', fontweight='bold', fontsize=14)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Occupancy Probability')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Histograms side by side
        ax = axes[1]
        
        n_shells = len(ion_data['shells'])
        
        if n_shells == 1:
            # Single histogram
            shell_name, shell_data = list(ion_data['shells'].items())[0]
            per_ion_probs = shell_data['probability_per_ion']
            
            bins = np.linspace(0, 1, 21)
            ax.hist(per_ion_probs, bins=bins, alpha=0.7, 
                color=colors[0], edgecolor='black', linewidth=0.5)
            
            ax.axvline(shell_data['mean_probability'], color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {shell_data["mean_probability"]:.3f}')
            
            start, end = shell_data['bounds']
            end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
            ax.set_title(f'{ion_type} {shell_data["name"]} Probability Distribution\n({start:.2f} - {end_str} Å)', 
                    fontweight='bold', fontsize=12)
            ax.set_xlabel('Occupancy Probability')
            ax.set_ylabel('Number of Ions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            
        else:
            # Multiple histograms side by side
            x_positions = np.arange(n_shells)
            width = 0.6
            
            # Create subplots for histograms
            fig2, hist_axes = plt.subplots(1, n_shells, figsize=(4*n_shells, 4))
            if n_shells == 1:
                hist_axes = [hist_axes]
            
            for i, (shell_name, shell_data) in enumerate(ion_data['shells'].items()):
                hist_ax = hist_axes[i]
                
                # Get per-ion probabilities
                per_ion_probs = shell_data['probability_per_ion']
                
                # Create histogram
                bins = np.linspace(0, 1, 21)
                hist_ax.hist(per_ion_probs, bins=bins, alpha=0.7, 
                            color=colors[i], edgecolor='black', linewidth=0.5)
                
                # Add statistics
                mean_prob = shell_data['mean_probability']
                hist_ax.axvline(mean_prob, color='red', linestyle='--', linewidth=2, 
                            label=f'Mean: {mean_prob:.3f}')
                
                # Format plot
                start, end = shell_data['bounds']
                end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                
                hist_ax.set_title(f'{shell_data["name"]}\n({start:.2f} - {end_str} Å)', 
                                fontweight='bold', fontsize=10)
                hist_ax.set_xlabel('Occupancy Probability')
                hist_ax.set_ylabel('Number of Ions')
                hist_ax.legend(fontsize=8)
                hist_ax.grid(True, alpha=0.3)
                hist_ax.set_xlim(0, 1)
            
            plt.suptitle(f'{ion_type} Shell Probability Histograms', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_plots:
                filename2 = f'{ion_type}_shell_probability_histograms.png'
                plt.savefig(filename2, dpi=300, bbox_inches='tight')
                print(f"Histograms saved as: {filename2}")
            
            plt.show()
            
            # For the bar chart in the main plot
            means = []
            stds = []
            shell_names = []
            
            for shell_name, shell_data in ion_data['shells'].items():
                means.append(shell_data['mean_probability'])
                stds.append(shell_data['std_probability'])
                shell_names.append(shell_data['name'])
            
            bars = ax.bar(x_positions, means, width, yerr=stds, 
                        color=colors[:len(means)], alpha=0.7, capsize=5)
            
            # Add value labels on bars
            for i, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{ion_type} Mean Shell Occupancy Probabilities', fontweight='bold', fontsize=14)
            ax.set_xlabel('Shell')
            ax.set_ylabel('Mean Occupancy Probability')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(shell_names)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'{ion_type}_shell_probabilities.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {filename}")
        
        plt.show()
    

    def plot_probability_histograms_by_type(self, save_plots=True):
        '''
        Plot histograms of individual ion probabilities for each shell and ion type.
        '''
        
        if not hasattr(self, 'shell_probabilities_by_type'):
            print("Shell probabilities by type not calculated. Run get_shell_occupancy_probabilities_by_type() first.")
            return
        
        probability_data = self.shell_probabilities_by_type
        
        # Calculate total number of subplots needed
        total_plots = 0
        for ion_data in probability_data.values():
            total_plots += len(ion_data['shells'])
        
        if total_plots == 0:
            print("No shell data available for histograms")
            return
        
        # Create figure
        cols = min(4, total_plots)
        rows = (total_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        
        if total_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        for ion_type, ion_data in probability_data.items():
            ion_category = ion_data['type']
            color_map = plt.cm.Blues if ion_category == 'cation' else plt.cm.Reds
            
            for shell_name, shell_data in ion_data['shells'].items():
                if plot_idx < len(axes):
                    ax = axes[plot_idx]
                    
                    # Get per-ion probabilities
                    per_ion_probs = shell_data['probability_per_ion']
                    
                    # Create histogram
                    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
                    ax.hist(per_ion_probs, bins=bins, alpha=0.7, 
                        color=color_map(0.7), edgecolor='black', linewidth=0.5)
                    
                    # Add statistics
                    mean_prob = shell_data['mean_probability']
                    std_prob = shell_data['std_probability']
                    
                    ax.axvline(mean_prob, color='red', linestyle='--', linewidth=2, 
                            label=f'Mean: {mean_prob:.3f}')
                    
                    # Format plot
                    start, end = shell_data['bounds']
                    end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                    
                    ax.set_title(f'{ion_type} {shell_data["name"]}\n({start:.2f} - {end_str} Å)', 
                            fontweight='bold', fontsize=10)
                    ax.set_xlabel('Occupancy Probability')
                    ax.set_ylabel('Number of Ions')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 1)
                    
                    plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Distribution of Individual Ion Shell Occupancy Probabilities', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('shell_probability_histograms_by_type.png', dpi=300, bbox_inches='tight')
            print("Histogram plot saved as: shell_probability_histograms_by_type.png")
        
        plt.show()

    def get_shell_probability_for_type(self, ion_type):
        '''
        Get shell probability data for a specific ion type.
        
        Parameters
        ----------
        ion_type : str
            Ion type name (e.g., 'Na', 'Mg', 'Cl')
        
        Returns
        -------
        data : dict or None
            Shell probability data for the specified ion type
        '''
        
        if not hasattr(self, 'shell_probabilities_by_type'):
            print("Shell probabilities by type not calculated. Run get_shell_occupancy_probabilities_by_type() first.")
            return None
        
        if ion_type in self.shell_probabilities_by_type:
            return self.shell_probabilities_by_type[ion_type]
        else:
            available_types = list(self.shell_probabilities_by_type.keys())
            print(f"Ion type '{ion_type}' not found. Available types: {available_types}")
            return None

    def analyze_shell_dynamics_by_type(self, ion_type, shell_name='shell_1', save_plots=True):
        '''
        Analyze the dynamics of shell occupancy for a specific ion type and shell.
        
        Parameters
        ----------
        ion_type : str
            Ion type to analyze (e.g., 'Na', 'Mg', 'Cl')
        shell_name : str
            Shell to analyze, default='shell_1'
        save_plots : bool
            Whether to save plots, default=True
        '''
        
        if not hasattr(self, 'shell_probabilities_by_type'):
            print("Shell probabilities by type not calculated. Run get_shell_occupancy_probabilities_by_type() first.")
            return
        
        if ion_type not in self.shell_probabilities_by_type:
            available_types = list(self.shell_probabilities_by_type.keys())
            print(f"Ion type '{ion_type}' not found. Available types: {available_types}")
            return
        
        ion_data = self.shell_probabilities_by_type[ion_type]
        
        if shell_name not in ion_data['shells']:
            available_shells = list(ion_data['shells'].keys())
            print(f"Shell '{shell_name}' not found for {ion_type}. Available shells: {available_shells}")
            return
        
        shell_data = ion_data['shells'][shell_name]
        occupancy = shell_data['occupancy']
        
        print(f"\n=== SHELL DYNAMICS ANALYSIS: {ion_type} {shell_data['name']} ===")
        print(f"Shell bounds: {shell_data['bounds'][0]:.2f} - {shell_data['bounds'][1]:.2f} Å")
        print(f"Number of ions: {occupancy.shape[0]}")
        print(f"Number of frames: {occupancy.shape[1]}")
        
        # Calculate transition statistics
        transitions_in = []   # Empty -> Occupied transitions
        transitions_out = []  # Occupied -> Empty transitions
        residence_times = []  # How long shells stay occupied
        vacancy_times = []    # How long shells stay empty
        
        for ion_idx in range(occupancy.shape[0]):
            ion_occupancy = occupancy[ion_idx, :]
            
            # Find transitions and residence times
            current_state = ion_occupancy[0]
            state_start = 0
            
            for frame_idx in range(1, len(ion_occupancy)):
                new_state = ion_occupancy[frame_idx]
                
                if new_state != current_state:
                    # Transition occurred
                    duration = frame_idx - state_start
                    
                    if current_state == 1:
                        # Was occupied, now empty
                        transitions_out.append((ion_idx, state_start, frame_idx))
                        residence_times.append(duration)
                    else:
                        # Was empty, now occupied
                        transitions_in.append((ion_idx, state_start, frame_idx))
                        vacancy_times.append(duration)
                    
                    current_state = new_state
                    state_start = frame_idx
            
            # Handle final state
            final_duration = len(ion_occupancy) - state_start
            if current_state == 1:
                residence_times.append(final_duration)
            else:
                vacancy_times.append(final_duration)
        
        # Print statistics
        print(f"\nDYNAMICS STATISTICS:")
        print(f"  Total transitions in:  {len(transitions_in)}")
        print(f"  Total transitions out: {len(transitions_out)}")
        
        if residence_times:
            print(f"  Mean residence time:   {np.mean(residence_times):.1f} ± {np.std(residence_times):.1f} frames")
            print(f"  Max residence time:    {np.max(residence_times)} frames")
        
        if vacancy_times:
            print(f"  Mean vacancy time:     {np.mean(vacancy_times):.1f} ± {np.std(vacancy_times):.1f} frames")
            print(f"  Max vacancy time:      {np.max(vacancy_times)} frames")
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Occupancy time series for first few ions
        ax = axes[0, 0]
        n_ions_to_show = min(5, occupancy.shape[0])
        colors = plt.cm.Set1(np.linspace(0, 1, n_ions_to_show))
        
        for i in range(n_ions_to_show):
            ax.plot(occupancy[i, :] + i*0.1, color=colors[i], 
                linewidth=1, alpha=0.7, label=f'Ion {i+1}')
        
        ax.set_title(f'{ion_type} {shell_data["name"]} Occupancy Time Series')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Occupancy + Offset')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residence time distribution
        ax = axes[0, 1]
        if residence_times:
            ax.hist(residence_times, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(np.mean(residence_times), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(residence_times):.1f}')
        ax.set_title('Residence Time Distribution')
        ax.set_xlabel('Residence Time (frames)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Vacancy time distribution
        ax = axes[1, 0]
        if vacancy_times:
            ax.hist(vacancy_times, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax.axvline(np.mean(vacancy_times), color='red', linestyle='--',
                    label=f'Mean: {np.mean(vacancy_times):.1f}')
        ax.set_title('Vacancy Time Distribution')
        ax.set_xlabel('Vacancy Time (frames)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Overall probability distribution
        ax = axes[1, 1]
        prob_vs_time = shell_data['probability_distribution']
        ax.plot(prob_vs_time, color='green', linewidth=2)
        ax.axhline(shell_data['mean_probability'], color='red', linestyle='--',
                label=f'Mean: {shell_data["mean_probability"]:.3f}')
        ax.set_title('Shell Occupancy Probability vs Time')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{ion_type} {shell_data["name"]} Dynamics Analysis', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            filename = f'{ion_type}_{shell_name}_dynamics_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Dynamics analysis plot saved as: {filename}")
        
        plt.show()
        
        # Return analysis results
        return {
            'transitions_in': transitions_in,
            'transitions_out': transitions_out,
            'residence_times': residence_times,
            'vacancy_times': vacancy_times,
            'mean_residence_time': np.mean(residence_times) if residence_times else 0,
            'mean_vacancy_time': np.mean(vacancy_times) if vacancy_times else 0
        }


    def shell_coordination_probabilities_by_type(
        self,
        ion_type=None,
        plot=True,
        save_plots=True,
        # --- layout ---
        figsize_per_ion=(5, 4),
        max_cols=3,
        # --- typography (base) ---
        font_size=10,
        font_weight='normal',
        # --- title ---
        show_title=True,
        title_font_size=12,
        title_font_weight='bold',
        # --- axis labels ---
        label_font_size=None,
        label_font_weight=None,     # None → falls back to font_weight
        tick_font_size=None,
        tick_font_weight=None,      # None → falls back to font_weight
        # --- filtering ---
        min_plot_pct=0.0,           # skip entire ion panel if max bar < this %
        # --- bar value labels ---
        bar_label_font_size=None,
        bar_label_font_weight=None,  # None → falls back to font_weight
        min_label_pct=5.0,          # only label bars above this %
        # --- bars ---
        bar_width=0.5,              # fixed bar width in data units (matplotlib default 0.8)
        bar_alpha=0.7,
        bar_linewidth=1.0,
        bar_color_cation='steelblue',
        bar_color_anion='crimson',
        edgecolor='black',
        # --- grid ---
        grid_alpha=0.3,
        # --- output ---
        dpi=300,
        output_filename=None,       # None → auto 'shell_probabilities_by_ion_type.png'
        save_combined=True,
        save_individual=False,
        transparent=False,
        # --- per-ion colour / hatch overrides ---
        ion_colors=None,            # dict {ion_name: color} to override cation/anion defaults
        bar_hatches=None,           # str | dict {ion_name: hatch} | list — hatch per ion panel
        xlabel=None,                # x-axis label; None → no label
        # --- legend ---
        show_legend=True,
        legend_font_size=None,
        legend_font_weight=None,
        legend_bbox_to_anchor=None,
        legend_frame_alpha=None,
        # --- bar value label control ---
        show_bar_label=True,        # show percentage labels on bars
        show_pct_mark=True,         # include '%' in bar labels; False → numbers only
    ):
        '''
        Calculate shell coordination probabilities (coordination environments) for
        specific ion types using their individual Solute objects.
        
        Parameters
        ----------
        ion_type : str or None
            Specific ion type to analyze (e.g., 'Na', 'Cl'). If None, analyzes all ion types.
        plot : bool
            Whether to plot the distributions, default=True
        save_plots : bool
            Whether to save plots, default=True
        figsize_per_ion : tuple
            (width, height) per subplot panel, default=(5, 4)
        max_cols : int
            Maximum columns in the grid, default=3
        font_size : float
            Base font size used as fallback when specific sizes are None, default=10
        font_weight : str
            Axis label font weight, default='normal'
        show_title : bool
            Whether to show subplot titles, default=True
        title_font_size : float
            Title font size, default=12
        title_font_weight : str
            Title font weight, default='bold'
        label_font_size : float or None
            Axis label font size; None falls back to font_size
        label_font_weight : str or None
            Axis label font weight; None falls back to font_weight
        tick_font_size : float or None
            Tick label font size; None falls back to font_size
        tick_font_weight : str or None
            Tick label font weight; None falls back to font_weight
        min_plot_pct : float
            Skip entire ion panel when its highest bar is below this percentage,
            default=0.0 (show everything)
        bar_label_font_size : float or None
            Bar value label font size; None falls back to font_size - 2
        bar_label_font_weight : str or None
            Bar value label font weight; None falls back to font_weight
        min_label_pct : float
            Only draw bar value labels on bars >= this percentage, default=5.0
        bar_width : float
            Bar width in data units, default=0.5 (matplotlib default is 0.8).
            Narrower values (0.3–0.5) look better when there are few shell types.
        bar_alpha : float
            Bar fill alpha, default=0.7
        bar_linewidth : float
            Bar edge linewidth, default=1.0
        bar_color_cation : str
            Bar colour for cation panels, default='steelblue'
        bar_color_anion : str
            Bar colour for anion panels, default='crimson'
        edgecolor : str
            Bar edge colour, default='black'
        grid_alpha : float
            Horizontal grid alpha, default=0.3
        dpi : int
            Resolution for saved figures, default=300
        output_filename : str or None
            Override output filename for combined figure; None → auto-name
        save_combined : bool
            Save the multi-panel figure, default=True
        save_individual : bool
            Save each ion panel as a separate file, default=False
        transparent : bool
            Transparent background when saving, default=False
        
        Returns
        -------
        results : dict
            Dictionary with ion types as keys and their shell probability data as values
        '''
        
        # Check if ion-type-specific solutes are available
        if not (hasattr(self, 'solutes_ci') and hasattr(self, 'solutes_ai')):
            print('Ion-type-specific solutes not initialized. Try initialize_Solutes_by_type() first')
            return None
        
        results = {}
        
        # Get unique ion types
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        # Determine which ions to process
        if ion_type is not None:
            # Process specific ion type
            if ion_type in cation_types:
                ions_to_process = [(ion_type, 'cation')]
            elif ion_type in anion_types:
                ions_to_process = [(ion_type, 'anion')]
            else:
                print(f"Ion type '{ion_type}' not found.")
                available_types = list(cation_types.keys()) + list(anion_types.keys())
                print(f"Available types: {available_types}")
                return None
        else:
            # Process all ion types
            ions_to_process = [(name, 'cation') for name in cation_types.keys()]
            ions_to_process += [(name, 'anion') for name in anion_types.keys()]
        
        print(f"Calculating shell coordination probabilities for: {[ion[0] for ion in ions_to_process]}")
        
        # Process each ion type
        for ion_name, ion_category in ions_to_process:
            print(f"\nProcessing {ion_name} ({ion_category})...")
            
            # Get the appropriate solute
            if ion_category == 'cation':
                if ion_name in self.solutes_ci and self.solutes_ci[ion_name] is not None:
                    solute = self.solutes_ci[ion_name]
                else:
                    print(f"  No solute available for {ion_name}")
                    continue
            else:  # anion
                if ion_name in self.solutes_ai and self.solutes_ai[ion_name] is not None:
                    solute = self.solutes_ai[ion_name]
                else:
                    print(f"  No solute available for {ion_name}")
                    continue
            
            # Check if solute has speciation data
            if not hasattr(solute, 'speciation') or solute.speciation is None:
                print(f"  No speciation data for {ion_name}")
                continue
            
            if not hasattr(solute.speciation, 'speciation_fraction'):
                print(f"  No speciation_fraction data for {ion_name}")
                continue
            
            # Get shell data
            try:
                df = solute.speciation.speciation_fraction.copy()
                
                # Create shell labels
                shell_labels = []
                for i in range(df.shape[0]):
                    row = df.iloc[i]
                    # Handle different possible column names
                    if 'coion' in df.columns and 'water' in df.columns:
                        shell_labels.append(f'{row.coion:.0f}-{row.water:.0f}')
                    elif 'n_coion' in df.columns and 'n_water' in df.columns:
                        shell_labels.append(f'{row.n_coion:.0f}-{row.n_water:.0f}')
                    else:
                        # Fallback: use row index
                        shell_labels.append(f'Shell_{i+1}')
                
                df['shell'] = shell_labels
                
                # Ensure fraction column exists
                if 'fraction' not in df.columns:
                    if 'count' in df.columns:
                        df['fraction'] = df['count'] / df['count'].sum()
                    elif 'probability' in df.columns:
                        df['fraction'] = df['probability']
                    else:
                        # Create equal probabilities as fallback
                        df['fraction'] = 1.0 / len(df)
                        print(f"  Warning: No fraction/count data for {ion_name}, using equal probabilities")
                
                # Store results
                results[ion_name] = {
                    'data': df,
                    'category': ion_category,
                    'n_shells': len(df),
                    'most_probable_shell': df.loc[df['fraction'].idxmax(), 'shell'],
                    'max_probability': df['fraction'].max()
                }
                
                # Print summary
                print(f"  Found {len(df)} different shell types for {ion_name}")
                print(f"  Most probable shell: {results[ion_name]['most_probable_shell']} ({results[ion_name]['max_probability']*100:.1f}%)")
                
                # Print top 3 shells
                top_shells = df.nlargest(min(3, len(df)), 'fraction')
                for idx, (_, row) in enumerate(top_shells.iterrows()):
                    print(f"    {idx+1}. {row['shell']}: {row['fraction']*100:.1f}%")
                    
            except Exception as e:
                print(f"  Error processing {ion_name}: {e}")
                continue
        
        # Plot results if requested
        if plot and results:
            self._plot_shell_coordination_probabilities_by_type(
                results, save_plots,
                figsize_per_ion=figsize_per_ion,
                max_cols=max_cols,
                font_size=font_size,
                font_weight=font_weight,
                show_title=show_title,
                title_font_size=title_font_size,
                title_font_weight=title_font_weight,
                label_font_size=label_font_size,
                label_font_weight=label_font_weight,
                tick_font_size=tick_font_size,
                tick_font_weight=tick_font_weight,
                bar_label_font_size=bar_label_font_size,
                bar_label_font_weight=bar_label_font_weight,
                min_plot_pct=min_plot_pct,
                min_label_pct=min_label_pct,
                bar_width=bar_width,
                bar_alpha=bar_alpha,
                bar_linewidth=bar_linewidth,
                bar_color_cation=bar_color_cation,
                bar_color_anion=bar_color_anion,
                edgecolor=edgecolor,
                grid_alpha=grid_alpha,
                dpi=dpi,
                output_filename=output_filename,
                save_combined=save_combined,
                save_individual=save_individual,
                transparent=transparent,
                ion_colors=ion_colors,
                bar_hatches=bar_hatches,
                xlabel=xlabel,
                show_legend=show_legend,
                legend_font_size=legend_font_size,
                legend_font_weight=legend_font_weight,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                legend_frame_alpha=legend_frame_alpha,
                show_bar_label=show_bar_label,
                show_pct_mark=show_pct_mark,
            )
        
        # Store results for future reference
        self.shell_probabilities_by_ion_type = results
        
        return results

    def _plot_shell_coordination_probabilities_by_type(
            self, results, save_plots,
            figsize_per_ion=(5, 4), max_cols=3,
            font_size=10, font_weight='normal',
            show_title=True, title_font_size=12, title_font_weight='bold',
            label_font_size=None, label_font_weight=None,
            tick_font_size=None,  tick_font_weight=None,
            bar_label_font_size=None, bar_label_font_weight=None,
            min_plot_pct=0.0, min_label_pct=5.0,
            bar_width=0.5,
            bar_alpha=0.7, bar_linewidth=1.0,
            bar_color_cation='steelblue', bar_color_anion='crimson', edgecolor='black',
            grid_alpha=0.3,
            dpi=300, output_filename=None,
            save_combined=True, save_individual=False, transparent=False,
            ion_colors=None, bar_hatches=None, xlabel=None,
            show_legend=True,
            legend_font_size=None, legend_font_weight=None,
            legend_bbox_to_anchor=None, legend_frame_alpha=None,
            show_bar_label=True, show_pct_mark=True,
        ):
            '''Plot shell coordination probabilities for each ion type'''

            # Resolve font size fallbacks
            _lbl   = label_font_size     if label_font_size     is not None else font_size
            _lbl_w = label_font_weight   if label_font_weight   is not None else font_weight

            def _resolve_ion_hatch(ion_name, ion_idx, bar_hatches):
                '''Return a hatch string for this ion panel, or None.'''
                if bar_hatches is None:
                    return None
                if isinstance(bar_hatches, str):
                    return bar_hatches
                if isinstance(bar_hatches, dict):
                    return bar_hatches.get(ion_name, None)
                if isinstance(bar_hatches, (list, tuple)):
                    return bar_hatches[ion_idx % len(bar_hatches)] if bar_hatches else None
                return None
            _tick  = tick_font_size      if tick_font_size       is not None else font_size
            _tick_w= tick_font_weight    if tick_font_weight     is not None else font_weight
            _blbl  = bar_label_font_size   if bar_label_font_size   is not None else max(font_size - 2, 6)
            _blbl_w= bar_label_font_weight if bar_label_font_weight is not None else font_weight

            # --- threshold filter: drop panels whose peak bar is below min_plot_pct ---
            if min_plot_pct > 0.0:
                results = {
                    k: v for k, v in results.items()
                    if v['max_probability'] * 100 >= min_plot_pct
                }
                if not results:
                    print(f"No ion types pass min_plot_pct={min_plot_pct}% threshold — nothing to plot.")
                    return

            n_ions = len(results)
            if n_ions == 0:
                return

            # Pre-filter each ion's df and find global max n_bars so every
            # panel gets the same xlim → same physical bar width everywhere.
            filtered_dfs = {}
            for _ion, _id in results.items():
                _df = _id['data'].copy()
                if min_plot_pct > 0.0:
                    _df = _df[_df['fraction'] * 100 >= min_plot_pct].reset_index(drop=True)
                filtered_dfs[_ion] = _df

            _global_max_nb = max((len(df) for df in filtered_dfs.values()), default=1)
            _pad  = bar_width * 1.5
            _xlim = (-_pad, _global_max_nb - 1 + _pad)

            # Calculate grid dimensions
            cols = min(max_cols, n_ions)
            rows = (n_ions + cols - 1) // cols

            fw, fh = figsize_per_ion
            fig, axes = plt.subplots(rows, cols, figsize=(fw * cols, fh * rows))

            # Normalise axes to a flat list
            if n_ions == 1:
                axes = [axes]
            elif rows == 1:
                axes = list(np.atleast_1d(axes))
            else:
                axes = list(axes.flatten())

            plot_idx = 0

            for ion_name, ion_data in results.items():
                if plot_idx >= len(axes):
                    break

                ax  = axes[plot_idx]
                df  = filtered_dfs[ion_name]
                ion_category = ion_data['category']

                if df.empty:
                    print(f"  {ion_name}: all bars below {min_plot_pct}% — panel skipped.")
                    axes[plot_idx].set_visible(False)
                    plot_idx += 1
                    continue

                color  = bar_color_cation if ion_category == 'cation' else bar_color_anion
                _color = ion_colors.get(ion_name, color) if ion_colors else color
                _hatch = _resolve_ion_hatch(ion_name, plot_idx, bar_hatches)
                n_bars = len(df)
                # Center this panel's bars within the global xlim
                offset = (_global_max_nb - n_bars) / 2.0
                x_pos  = np.arange(n_bars) + offset

                ax.set_xlim(*_xlim)

                # Bar plot
                bars = ax.bar(
                    x_pos, df['fraction'] * 100,
                    width=bar_width,
                    color=_color, alpha=bar_alpha,
                    edgecolor=edgecolor, linewidth=bar_linewidth,
                    hatch=_hatch or '',
                )

                # Axis labels
                ax.set_xlabel(xlabel if xlabel is not None else '', fontsize=_lbl, fontweight=_lbl_w)
                ax.set_ylabel('Probability (%)', fontsize=_lbl, fontweight=_lbl_w)

                # Title
                if show_title:
                    ax.set_title(
                        f'{ion_name} Shell Distribution\n({ion_category.title()})',
                        fontweight=title_font_weight, fontsize=title_font_size,
                    )

                # Ticks
                ax.set_xticks(x_pos)
                ax.set_xticklabels(df['shell'], rotation=45, ha='right', fontsize=_tick, fontweight=_tick_w)
                ax.tick_params(axis='y', labelsize=_tick)
                plt.setp(ax.get_yticklabels(), fontweight=_tick_w)

                ax.grid(True, alpha=grid_alpha, axis='y')

                # Bar value labels
                _any_lbl = False
                if show_bar_label:
                    for bar, percentage in zip(bars, df['fraction'] * 100):
                        if percentage >= min_label_pct:
                            _any_lbl = True
                            _lbl_txt = f'{percentage:.1f}%' if show_pct_mark else f'{percentage:.1f}'
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.,
                                bar.get_height() + 0.5,
                                _lbl_txt,
                                ha='center', va='bottom', fontsize=_blbl, fontweight=_blbl_w,
                            )

                # y-limit: expand so bar labels don't clip at top
                _max_val   = df['fraction'].max() * 100
                _smart_top = _max_val * 1.15
                if _any_lbl and _max_val > 0:
                    _ax_h_in = ax.get_figure().get_figheight() * ax.get_position().height
                    if _ax_h_in > 0:
                        _dpui     = _smart_top / _ax_h_in
                        _lbl_h_du = (_blbl / 72.0) * _dpui * 2.0
                        _needed   = _max_val + 0.5 + _lbl_h_du
                        _smart_top = max(_smart_top, _needed * 1.05)
                ax.set_ylim(0, _smart_top)

                # --- save individual panel ---
                if save_individual and save_plots:
                    single_fig, single_ax = plt.subplots(figsize=figsize_per_ion)
                    single_ax.set_xlim(*_xlim)
                    single_bars = single_ax.bar(
                        x_pos, df['fraction'] * 100,
                        width=bar_width,
                        color=_color, alpha=bar_alpha,
                        edgecolor=edgecolor, linewidth=bar_linewidth,
                        hatch=_hatch or '',
                    )
                    single_ax.set_xlabel(xlabel if xlabel is not None else '', fontsize=_lbl, fontweight=_lbl_w)
                    single_ax.set_ylabel('Probability (%)', fontsize=_lbl, fontweight=_lbl_w)
                    if show_title:
                        single_ax.set_title(
                            f'{ion_name} Shell Distribution\n({ion_category.title()})',
                            fontweight=title_font_weight, fontsize=title_font_size,
                        )
                    single_ax.set_xticks(x_pos)
                    single_ax.set_xticklabels(df['shell'], rotation=45, ha='right', fontsize=_tick, fontweight=_tick_w)
                    single_ax.tick_params(axis='y', labelsize=_tick)
                    plt.setp(single_ax.get_yticklabels(), fontweight=_tick_w)
                    single_ax.grid(True, alpha=grid_alpha, axis='y')
                    _any_lbl_si = False
                    if show_bar_label:
                        for b, pct in zip(single_bars, df['fraction'] * 100):
                            if pct >= min_label_pct:
                                _any_lbl_si = True
                                _lbl_txt = f'{pct:.1f}%' if show_pct_mark else f'{pct:.1f}'
                                single_ax.text(
                                    b.get_x() + b.get_width() / 2., b.get_height() + 0.5,
                                    _lbl_txt, ha='center', va='bottom', fontsize=_blbl, fontweight=_blbl_w,
                                )
                    _max_val_si   = df['fraction'].max() * 100
                    _smart_top_si = _max_val_si * 1.15
                    if _any_lbl_si and _max_val_si > 0:
                        _ax_h_in_si = single_fig.get_figheight() * single_ax.get_position().height
                        if _ax_h_in_si > 0:
                            _dpui_si     = _smart_top_si / _ax_h_in_si
                            _lbl_h_du_si = (_blbl / 72.0) * _dpui_si * 2.0
                            _needed_si   = _max_val_si + 0.5 + _lbl_h_du_si
                            _smart_top_si = max(_smart_top_si, _needed_si * 1.05)
                    single_ax.set_ylim(0, _smart_top_si)
                    single_fig.tight_layout()
                    ind_fname = f'shell_probabilities_{ion_name}.png'
                    single_fig.savefig(ind_fname, dpi=dpi, bbox_inches='tight', transparent=transparent)
                    print(f"  Individual plot saved as: {ind_fname}")
                    plt.close(single_fig)

                plot_idx += 1

            # Hide unused subplots
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()

            if save_combined and save_plots:
                combined_fname = output_filename if output_filename else 'shell_probabilities_by_ion_type.png'
                fig.savefig(combined_fname, dpi=dpi, bbox_inches='tight', transparent=transparent)
                print(f"Combined plot saved as: {combined_fname}")

            plt.show()


    def compare_shell_probabilities_by_type(
        self,
        ion_types=None,
        save_plots=True,
        figsize=(16, 6),
        font_size=10,
        font_weight='normal',
        title_font_size=12,
        title_font_weight='bold',
        show_title=True,
        label_font_size=None,
        label_font_weight=None,
        tick_font_size=None,
        tick_font_weight=None,
        bar_label_font_size=None,
        bar_label_font_weight=None,
        min_label_pct=5.0,
        bar_width=0.8,
        bar_alpha=0.8,
        grid_alpha=0.3,
        legend_font_size=None,
        legend_font_weight=None,
        legend_bbox_to_anchor=None,
        legend_frame_alpha=None,
        dpi=300,
        output_filename=None,
        save_combined=True,
        save_individual=False,
        transparent=False,
        # --- per-ion colour / hatch overrides ---
        ion_colors=None,            # dict {ion_name: color} to override colormap defaults
        bar_hatches=None,           # str | dict {ion_name: hatch} | list — hatch per ion
        xlabel=None,                # x-axis label; None → no label
        # --- ordering ---
        bar_order='label',          # 'label' (alphabetical) | 'probability' (highest first)
        # --- bar value label control ---
        show_bar_label=True,        # show percentage labels on bars
        show_pct_mark=True,         # include '%' in bar labels; False → numbers only
    ):
        '''
        Create a comparison plot showing ALL shell coordination probabilities for multiple ion types.
        FIXED: Now compares ALL shell types found in shell_1, not just the most probable one.

        Parameters
        ----------
        ion_types : list or None
            List of ion types to compare. If None, compares all available types.
        save_plots : bool
            Whether to save the plot, default=True
        figsize : tuple
            Figure size (width, height) in inches, default=(16, 6)
        font_size : float
            Base font size for all text elements, default=10
        font_weight : str
            Base font weight for all text elements, default='normal'
        title_font_size : float
            Subplot title font size, default=12
        title_font_weight : str
            Subplot title font weight, default='bold'
        show_title : bool
            Whether to show subplot titles, default=True
        label_font_size : float or None
            Axis label font size; None falls back to font_size
        label_font_weight : str or None
            Axis label font weight; None falls back to font_weight
        tick_font_size : float or None
            Tick label font size; None falls back to font_size
        tick_font_weight : str or None
            Tick label font weight; None falls back to font_weight
        bar_label_font_size : float or None
            Bar value label font size; None falls back to font_size - 2
        bar_label_font_weight : str or None
            Bar value label font weight; None falls back to font_weight
        min_label_pct : float
            Exclude coordination environments where all ions are below this
            percentage (bars hidden from chart); also suppresses labels on
            individual bars below the threshold, default=5.0
        bar_alpha : float
            Bar transparency, default=0.8
        grid_alpha : float
            Grid line transparency, default=0.3
        legend_font_size : float or None
            Legend font size; None falls back to font_size
        legend_font_weight : str or None
            Legend font weight; None falls back to font_weight
        legend_bbox_to_anchor : tuple or None
            Legend anchor position, e.g. (0.5, 0.95); None uses matplotlib default
        legend_frame_alpha : float or None
            Legend frame alpha (0.0 = transparent); None uses matplotlib default
        dpi : int
            Resolution for saved figures, default=300
        output_filename : str or None
            Output filename for combined figure; None uses default name
        save_combined : bool
            Save the combined (cation + anion side-by-side) figure, default=True
        save_individual : bool
            Save cation and anion panels as separate figures, default=False
        transparent : bool
            Save figure with transparent background, default=False
        '''

        if not hasattr(self, 'shell_region_coordination_probabilities'):
            print("Shell region coordination probabilities not calculated.")
            print("Run shell_coordination_probabilities_by_shell_region_by_type() first.")
            return

        results = self.shell_region_coordination_probabilities

        if ion_types is not None:
            # Filter to requested ion types
            results = {k: v for k, v in results.items() if k in ion_types}

        if not results:
            print("No shell region coordination probability data available for comparison.")
            return

        # Resolve font size/weight fallbacks
        _lbl    = label_font_size       if label_font_size       is not None else font_size
        _lbl_w  = label_font_weight     if label_font_weight     is not None else font_weight
        _tick   = tick_font_size        if tick_font_size        is not None else font_size
        _tick_w = tick_font_weight      if tick_font_weight      is not None else font_weight
        _blbl   = bar_label_font_size   if bar_label_font_size   is not None else max(font_size - 2, 6)
        _blbl_w = bar_label_font_weight if bar_label_font_weight is not None else font_weight
        _leg    = legend_font_size   if legend_font_size   is not None else font_size
        _leg_w  = legend_font_weight  if legend_font_weight is not None else font_weight

        # Separate cations and anions
        cations_data = {k: v for k, v in results.items() if v['ion_category'] == 'cation'}
        anions_data = {k: v for k, v in results.items() if v['ion_category'] == 'anion'}

        # Helper: True if any ion in data_dict has prob >= min_label_pct for this env
        def _peak_pct_for_env(env, data_dict):
            for _d in data_dict.values():
                if 'shell_1' in _d['shell_regions']:
                    _sd = _d['shell_regions']['shell_1']
                    _c = _sd['coordination_environments'].get(env, 0)
                    _t = _sd['total_observations']
                    if _t > 0 and (_c / _t) * 100 >= min_label_pct:
                        return True
            return False

        # ------------------------------------------------------------------ #
        # Pre-compute env lists + probability matrices for both panels so we
        # can determine the global max n_envs before touching any axes.
        # Same xlim on both panels → identical physical bar widths everywhere.
        # ------------------------------------------------------------------ #
        def _build_panel_data(data_dict):
            all_envs = set()
            for _d in data_dict.values():
                if 'shell_1' in _d['shell_regions']:
                    for _e in _d['shell_regions']['shell_1']['coordination_environments']:
                        all_envs.add(_e)
            all_envs = sorted(all_envs)
            if bar_order == 'probability':
                # sort envs by the max probability any ion achieves for that env
                def _max_prob_for_env(e):
                    best = 0.0
                    for _d in data_dict.values():
                        if 'shell_1' in _d['shell_regions']:
                            _sd = _d['shell_regions']['shell_1']
                            _c  = _sd['coordination_environments'].get(e, 0)
                            _t  = _sd['total_observations']
                            best = max(best, (_c / _t) if _t > 0 else 0)
                    return best
                all_envs = sorted(all_envs, key=_max_prob_for_env, reverse=True)
            if min_label_pct > 0.0:
                all_envs = [e for e in all_envs if _peak_pct_for_env(e, data_dict)]
            probs_matrix = {}
            for _ion, _d in data_dict.items():
                _probs = []
                for _e in all_envs:
                    if 'shell_1' in _d['shell_regions']:
                        _sd = _d['shell_regions']['shell_1']
                        _c  = _sd['coordination_environments'].get(_e, 0)
                        _t  = _sd['total_observations']
                        _probs.append((_c / _t) if _t > 0 else 0)
                    else:
                        _probs.append(0)
                probs_matrix[_ion] = _probs
            return all_envs, probs_matrix

        cation_envs, cation_probs = _build_panel_data(cations_data) if cations_data else ([], {})
        anion_envs,  anion_probs  = _build_panel_data(anions_data)  if anions_data  else ([], {})

        # Global span → same axis scale in both panels
        _gmax = max(len(cation_envs), len(anion_envs), 1)
        _pad  = bar_width * 1.5
        _xlim = (-_pad, _gmax - 1 + _pad)

        def _resolve_ion_hatch(ion_name, ion_idx):
            '''Resolve hatch string for a given ion in the comparison panel.'''
            if bar_hatches is None:
                return None
            if isinstance(bar_hatches, str):
                return bar_hatches
            if isinstance(bar_hatches, dict):
                return bar_hatches.get(ion_name, None)
            if isinstance(bar_hatches, (list, tuple)):
                return bar_hatches[ion_idx % len(bar_hatches)] if bar_hatches else None
            return None

        def _draw_comparison_panel(ax, envs, probs_matrix, colormap, label_suffix):
            ion_names    = list(probs_matrix.keys())
            n_ions       = len(ion_names)
            n_envs       = len(envs)
            colors       = colormap(np.linspace(0.4, 0.9, n_ions))
            bar_w        = bar_width / n_ions if n_ions > 1 else bar_width
            # Center this panel's groups within the global span
            x_offset     = (_gmax - n_envs) / 2.0
            x            = np.arange(n_envs) + x_offset
            _max_bar_val = 0.0
            _any_lbl     = False

            for i, ion_name in enumerate(ion_names):
                probs  = probs_matrix[ion_name]
                offset = bar_w * (i - (n_ions - 1) / 2.0)
                _color = ion_colors.get(ion_name, colors[i]) if ion_colors else colors[i]
                _hatch = _resolve_ion_hatch(ion_name, i)
                bars   = ax.bar(x + offset, np.array(probs) * 100, bar_w,
                                label=ion_name, color=_color, alpha=bar_alpha,
                                hatch=_hatch or '')
                if probs:
                    _max_bar_val = max(_max_bar_val, float(np.max(np.array(probs) * 100)))
                if show_bar_label:
                    for bar, prob in zip(bars, probs):
                        if prob * 100 >= min_label_pct:
                            _any_lbl = True
                            _lbl_txt = f'{prob*100:.0f}%' if show_pct_mark else f'{prob*100:.0f}'
                            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                                    _lbl_txt, ha='center', va='bottom',
                                    fontsize=_blbl, fontweight=_blbl_w)

            ax.set_xlim(*_xlim)
            ax.set_xlabel(xlabel if xlabel is not None else '', fontsize=_lbl, fontweight=_lbl_w)
            ax.set_ylabel('Probability (%)', fontsize=_lbl, fontweight=_lbl_w)
            if show_title:
                ax.set_title(f'{label_suffix} Shell_1 ALL Coordination Environments',
                             fontweight=title_font_weight, fontsize=title_font_size)
            ax.set_xticks(x)
            ax.set_xticklabels(envs, rotation=45, ha='right', fontsize=_tick, fontweight=_tick_w)
            ax.tick_params(axis='y', labelsize=_tick)
            plt.setp(ax.get_yticklabels(), fontweight=_tick_w)
            _leg_kw = dict(prop={'size': _leg, 'weight': _leg_w})
            if legend_bbox_to_anchor is not None:
                _leg_kw['bbox_to_anchor'] = legend_bbox_to_anchor
                _leg_kw['loc'] = 'upper center'
            if legend_frame_alpha is not None:
                _leg_kw['framealpha'] = legend_frame_alpha
            ax.legend(**_leg_kw)
            ax.grid(True, alpha=grid_alpha, axis='y')
            # Smart ylim: expand so bar labels don't clip at top
            _smart_top_c = _max_bar_val * 1.15 if _max_bar_val > 0 else 1.0
            if _any_lbl and _max_bar_val > 0:
                _ax_h_in_c = ax.get_figure().get_figheight() * ax.get_position().height
                if _ax_h_in_c > 0:
                    _dpui_c     = _smart_top_c / _ax_h_in_c
                    _lbl_h_du_c = (_blbl / 72.0) * _dpui_c * 2.0
                    _needed_c   = _max_bar_val + _lbl_h_du_c
                    _smart_top_c = max(_smart_top_c, _needed_c * 1.05)
            if _max_bar_val > 0:
                ax.set_ylim(0, _smart_top_c)

        # ------------------------------------------------------------------ #
        # Create comparison plot
        # ------------------------------------------------------------------ #
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        if cations_data:
            print(f"Cations - Shell_1 ALL coordination environments: {cation_envs}")
            _draw_comparison_panel(axes[0], cation_envs, cation_probs, plt.cm.Blues, 'Cation')
        else:
            axes[0].text(0.5, 0.5, 'No Cation Data', ha='center', va='center',
                         transform=axes[0].transAxes, fontsize=title_font_size)
            if show_title:
                axes[0].set_title('Cation Shell_1 Coordination Comparison',
                                  fontweight=title_font_weight, fontsize=title_font_size)

        if anions_data:
            print(f"Anions - Shell_1 ALL coordination environments: {anion_envs}")
            _draw_comparison_panel(axes[1], anion_envs, anion_probs, plt.cm.Reds, 'Anion')
        else:
            axes[1].text(0.5, 0.5, 'No Anion Data', ha='center', va='center',
                         transform=axes[1].transAxes, fontsize=title_font_size)
            if show_title:
                axes[1].set_title('Anion Shell_1 Coordination Comparison',
                                  fontweight=title_font_weight, fontsize=title_font_size)

        plt.tight_layout()

        if save_plots and save_combined:
            filename = output_filename if output_filename is not None \
                else 'shell_1_ALL_coordination_probabilities_comparison.png'
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', transparent=transparent)
            print(f"ALL shell types comparison plot saved as: {filename}")

        plt.show()

        if save_plots and save_individual:
            if cations_data:
                fig_ind, ax_ind = plt.subplots(figsize=(figsize[0] / 2, figsize[1]))
                _draw_comparison_panel(ax_ind, cation_envs, cation_probs, plt.cm.Blues, 'Cation')
                fig_ind.tight_layout()
                cation_fname = (output_filename.replace('.png', '_cations.png')
                                if output_filename else 'shell_1_comparison_cations.png')
                fig_ind.savefig(cation_fname, dpi=dpi, bbox_inches='tight', transparent=transparent)
                print(f"Cation comparison plot saved as: {cation_fname}")
                plt.close(fig_ind)
            if anions_data:
                fig_ind, ax_ind = plt.subplots(figsize=(figsize[0] / 2, figsize[1]))
                _draw_comparison_panel(ax_ind, anion_envs, anion_probs, plt.cm.Reds, 'Anion')
                fig_ind.tight_layout()
                anion_fname = (output_filename.replace('.png', '_anions.png')
                               if output_filename else 'shell_1_comparison_anions.png')
                fig_ind.savefig(anion_fname, dpi=dpi, bbox_inches='tight', transparent=transparent)
                print(f"Anion comparison plot saved as: {anion_fname}")
                plt.close(fig_ind)




    def get_shell_probabilities_summary_by_type(self):
        '''
        Print a comprehensive summary of shell probabilities for all ion types.
        UPDATED: Now includes results from both shell_probabilities_by_ion_type AND shell_region_coordination_probabilities.
        FIXED: Handle float vs int type in count column
        '''
        
        print("\n" + "="*80)
        print("COMPREHENSIVE SHELL PROBABILITY SUMMARY BY ION TYPE")
        print("="*80)
        
        # ===== PART 1: Shell Coordination Probabilities (from Solute speciation) =====
        if hasattr(self, 'shell_probabilities_by_ion_type') and self.shell_probabilities_by_ion_type:
            print("\n" + "-"*80)
            print("PART 1: SHELL COORDINATION PROBABILITIES (from Solute speciation)")
            print("-"*80)
            
            # Separate cations and anions
            cations_data = {k: v for k, v in self.shell_probabilities_by_ion_type.items() if v['category'] == 'cation'}
            anions_data = {k: v for k, v in self.shell_probabilities_by_ion_type.items() if v['category'] == 'anion'}
            
            if cations_data:
                print("\nCATIONS:")
                print("-" * 60)
                for ion_name, ion_data in cations_data.items():
                    df = ion_data['data']
                    print(f"\n{ion_name}:")
                    print(f"  Total coordination environments: {len(df)}")
                    print(f"  Top 5 most probable:")
                    top_5 = df.nlargest(5, 'fraction')
                    for idx, row in top_5.iterrows():
                        # FIXED: Convert count to int before formatting, handle potential float
                        count_val = int(row['count']) if isinstance(row['count'], (int, float)) else row['count']
                        print(f"    {row['shell']:>6s}: {row['fraction']*100:>6.2f}% (count: {count_val:>6d})")
            
            if anions_data:
                print("\nANIONS:")
                print("-" * 60)
                for ion_name, ion_data in anions_data.items():
                    df = ion_data['data']
                    print(f"\n{ion_name}:")
                    print(f"  Total coordination environments: {len(df)}")
                    print(f"  Top 5 most probable:")
                    top_5 = df.nlargest(5, 'fraction')
                    for idx, row in top_5.iterrows():
                        # FIXED: Convert count to int before formatting, handle potential float
                        count_val = int(row['count']) if isinstance(row['count'], (int, float)) else row['count']
                        print(f"    {row['shell']:>6s}: {row['fraction']*100:>6.2f}% (count: {count_val:>6d})")
        else:
            print("\n⚠️  Shell coordination probabilities not calculated.")
            print("   Run shell_coordination_probabilities_by_type() first.")
        
        # ===== PART 2: Shell Region Coordination Probabilities =====
        if hasattr(self, 'shell_region_coordination_probabilities') and self.shell_region_coordination_probabilities:
            print("\n" + "-"*80)
            print("PART 2: SHELL REGION COORDINATION PROBABILITIES (by shell region)")
            print("-"*80)
            
            # Separate cations and anions
            cations_region = {k: v for k, v in self.shell_region_coordination_probabilities.items() if v['ion_category'] == 'cation'}
            anions_region = {k: v for k, v in self.shell_region_coordination_probabilities.items() if v['ion_category'] == 'anion'}
            
            if cations_region:
                print("\nCATIONS (by shell region):")
                print("-" * 60)
                for ion_name, ion_data in cations_region.items():
                    print(f"\n{ion_name} (r₀={ion_data['coordination_radius']:.2f} Å):")
                    
                    for shell_name, shell_data in ion_data['shell_regions'].items():
                        bounds = shell_data['bounds']
                        total_obs = shell_data['total_observations']
                        
                        print(f"\n  {shell_name} ({bounds[0]:.2f}-{bounds[1]:.2f} Å, n={total_obs}):")
                        
                        if shell_data['probabilities']:
                            # Show top 5 coordination environments for this shell region
                            sorted_probs = sorted(shell_data['probabilities'].items(), 
                                                key=lambda x: x[1], reverse=True)[:5]
                            
                            for coord_env, prob in sorted_probs:
                                count = shell_data['coordination_environments'].get(coord_env, 0)
                                # FIXED: Ensure count is integer
                                count_val = int(count) if isinstance(count, (int, float)) else count
                                print(f"    {coord_env:>6s}: {prob*100:>6.2f}% (count: {count_val:>6d})")
                            
                            # Show most probable
                            most_prob = shell_data['most_probable_environment']
                            max_prob = shell_data['max_probability']
                            print(f"    → Most probable: {most_prob} ({max_prob*100:.1f}%)")
                        else:
                            print(f"    No coordination data for this shell")
            
            if anions_region:
                print("\nANIONS (by shell region):")
                print("-" * 60)
                for ion_name, ion_data in anions_region.items():
                    print(f"\n{ion_name} (r₀={ion_data['coordination_radius']:.2f} Å):")
                    
                    for shell_name, shell_data in ion_data['shell_regions'].items():
                        bounds = shell_data['bounds']
                        total_obs = shell_data['total_observations']
                        
                        print(f"\n  {shell_name} ({bounds[0]:.2f}-{bounds[1]:.2f} Å, n={total_obs}):")
                        
                        if shell_data['probabilities']:
                            # Show top 5 coordination environments for this shell region
                            sorted_probs = sorted(shell_data['probabilities'].items(), 
                                                key=lambda x: x[1], reverse=True)[:5]
                            
                            for coord_env, prob in sorted_probs:
                                count = shell_data['coordination_environments'].get(coord_env, 0)
                                # FIXED: Ensure count is integer
                                count_val = int(count) if isinstance(count, (int, float)) else count
                                print(f"    {coord_env:>6s}: {prob*100:>6.2f}% (count: {count_val:>6d})")
                            
                            # Show most probable
                            most_prob = shell_data['most_probable_environment']
                            max_prob = shell_data['max_probability']
                            print(f"    → Most probable: {most_prob} ({max_prob*100:.1f}%)")
                        else:
                            print(f"    No coordination data for this shell")
        else:
            print("\n⚠️  Shell region coordination probabilities not calculated.")
            print("   Run shell_coordination_probabilities_by_shell_region_by_type() first.")
        
        print("\n" + "="*80)
        
        # Print availability summary
        has_shell_probs = hasattr(self, 'shell_probabilities_by_ion_type') and self.shell_probabilities_by_ion_type
        has_region_probs = hasattr(self, 'shell_region_coordination_probabilities') and self.shell_region_coordination_probabilities
        
        print("\nDATA AVAILABILITY:")
        print(f"  Shell coordination probabilities: {'✓ Available' if has_shell_probs else '✗ Not calculated'}")
        print(f"  Shell region probabilities: {'✓ Available' if has_region_probs else '✗ Not calculated'}")
        print("="*80)




    # ------------------------------------------------------------------
    def compute_shell_region_coordination_by_type(
        self, ion_type=None, step=None, use_kdtree=True,
        save_cache=False, cache_filename=None, force_rerun=False,
    ):
        '''
        Compute shell region coordination probabilities — analysis only, no plotting.

        Results are stored on ``self.shell_region_coordination_probabilities`` and
        returned so they can be passed directly to
        ``plot_shell_region_coordination_by_type``.

        Parameters
        ----------
        ion_type : str or None
            Single ion type to analyse (e.g. ``'Na'``), or None for all types.
        step : int or None
            Frame stride. Default: ``self.default_step``.
        use_kdtree : bool
            Reserved (kept for API compatibility).
        save_cache : bool
            If True, persist results to a pickle file after calculation
            (default=False).
        cache_filename : str or None
            Cache file path. Default:
            ``'shell_region_coordination_probs_cache.pkl'``.
        force_rerun : bool
            If True, ignore an existing cache file and recompute (default=False).

        Returns
        -------
        results : dict or None
        '''
        _cache_file = cache_filename or 'shell_region_coordination_probs_cache.pkl'

        # Try loading from cache
        if not force_rerun and os.path.exists(_cache_file):
            print(f"Loading shell region coordination cache from {_cache_file} ...")
            if self.load_shell_region_coordination_probabilities_from_file(_cache_file):
                return self.shell_region_coordination_probabilities
            print("Cache load failed — recomputing.")

        if step is None:
            step = self.default_step

        # Check prerequisites
        if not (hasattr(self, 'cation_shells_by_type') and hasattr(self, 'anion_shells_by_type')):
            print('Ion-type-specific shells not determined. Run determine_ion_solvation_shells_by_type() first')
            return None

        # Get unique ion types
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        all_ion_types = {**cation_types, **anion_types}

        # Determine which ions to process
        if ion_type is not None:
            if ion_type in cation_types:
                ions_to_process = [(ion_type, 'cation', cation_types[ion_type])]
            elif ion_type in anion_types:
                ions_to_process = [(ion_type, 'anion', anion_types[ion_type])]
            else:
                available_types = list(cation_types.keys()) + list(anion_types.keys())
                print(f"Ion type '{ion_type}' not found. Available types: {available_types}")
                return None
        else:
            ions_to_process = [(name, 'cation', group) for name, group in cation_types.items()]
            ions_to_process += [(name, 'anion', group) for name, group in anion_types.items()]

        print(f"Calculating shell region coordination probabilities for: {[ion[0] for ion in ions_to_process]}")
        print("MODIFIED ANALYSIS:")
        print("  Shell_1: Full coordination environment (coions + waters)")
        print("  Shell_2: Specific ion type interactions (e.g., Na finds Mg, Cl separately)")
        print("  Shell_3: Specific ion type interactions (e.g., Na finds Mg, Cl separately)")

        n_frames = len(self.universe.trajectory[::step])
        results = {}
        
        # Process each ion type
        for ion_name, ion_category, ion_group in ions_to_process:
            print(f"\nProcessing {ion_name} ({ion_category})...")
            
            # Get shell data for this ion type
            if ion_category == 'cation':
                if (ion_name not in self.cation_shells_by_type or 
                    self.cation_shells_by_type[ion_name] is None):
                    print(f"  No shell data for {ion_name}")
                    continue
                shells = self.cation_shells_by_type[ion_name]
            else:  # anion
                if (ion_name not in self.anion_shells_by_type or 
                    self.anion_shells_by_type[ion_name] is None):
                    print(f"  No shell data for {ion_name}")
                    continue
                shells = self.anion_shells_by_type[ion_name]
            
            # Get shell regions (exclude bulk)
            shell_regions = {k: v for k, v in shells.data.items() if k.startswith('shell_')}
            if not shell_regions:
                print(f"  No shell regions found for {ion_name}")
                continue
            
            print(f"  Analyzing shell regions: {list(shell_regions.keys())}")
            for shell_name, (start, end) in shell_regions.items():
                if shell_name == 'shell_1':
                    analysis_type = "Full (coions+waters)"
                else:
                    # For shell_2 and shell_3, show which ion types we'll analyze
                    other_ion_types = [itype for itype in all_ion_types.keys() if itype != ion_name]
                    analysis_type = f"Specific ions ({', '.join(other_ion_types)})"
                print(f"    {shell_name}: {start:.2f} - {end:.2f} Å ({analysis_type})")
            
            # Get coordination radius for this ion type
            coordination_radius = None
            if ion_category == 'cation':
                if hasattr(self, 'solutes_ci') and ion_name in self.solutes_ci:
                    if self.solutes_ci[ion_name] is not None:
                        coordination_radius = self.solutes_ci[ion_name].radii['water']
            else:  # anion
                if hasattr(self, 'solutes_ai') and ion_name in self.solutes_ai:
                    if self.solutes_ai[ion_name] is not None:
                        coordination_radius = self.solutes_ai[ion_name].radii['water']
            
            if coordination_radius is None:
                print(f"  Warning: No coordination radius found for {ion_name}")
                continue
            
            # Initialize results for this ion type
            results[ion_name] = {
                'ion_category': ion_category,
                'coordination_radius': coordination_radius,
                'shell_regions': {}
            }
            
            # Initialize counters for each shell region
            for shell_name, (start_r, end_r) in shell_regions.items():
                results[ion_name]['shell_regions'][shell_name] = {
                    'bounds': (start_r, end_r),
                    'coordination_environments': {},
                    'total_observations': 0,
                    'analysis_type': 'full' if shell_name == 'shell_1' else 'specific_ions'
                }
            
            # Analyze each frame
            print(f"  Analyzing {n_frames} frames...")
            
            # Handle debug trajectory
            if hasattr(self, '_debug_frame_indices') and self._debug_frame_indices is not None:
                frame_indices = self._debug_frame_indices[::step]
            else:
                frame_indices = range(0, len(self.universe.trajectory), step)
            
            for frame_idx in tqdm(frame_indices, desc=f"Analyzing {ion_name} shell environments", leave=False):
                ts = self.universe.trajectory[frame_idx]
                
                for ion_atom in ion_group:
                    # For each shell region, search directly in that region
                    for shell_name, (start_r, end_r) in shell_regions.items():
                        
                        # MODIFIED: Different analysis based on shell region
                        if shell_name == 'shell_1':
                            # Shell_1: Full coordination environment analysis (coions + waters)
                            
                            # Count waters in shell_1
                            water_distances = cdist([ion_atom.position], self.waters.positions)[0]
                            waters_in_shell = np.sum((water_distances >= start_r) & (water_distances < end_r))
                            
                            # Count coions in shell_1 (all other ions combined)
                            coions_in_shell = 0
                            for other_ion_name, other_ion_group in all_ion_types.items():
                                if other_ion_name != ion_name:
                                    other_distances = cdist([ion_atom.position], other_ion_group.positions)[0]
                                    coions_in_shell += np.sum((other_distances >= start_r) & (other_distances < end_r))
                            
                            # Create coordination environment string: "coions-waters"
                            coord_env = f"{coions_in_shell}-{waters_in_shell}"
                            
                        else:
                            # Shell_2 and Shell_3: Specific ion type analysis
                            
                            # Count each specific ion type separately
                            ion_counts = []
                            for other_ion_name, other_ion_group in all_ion_types.items():
                                if other_ion_name != ion_name:
                                    other_distances = cdist([ion_atom.position], other_ion_group.positions)[0]
                                    count = np.sum((other_distances >= start_r) & (other_distances < end_r))
                                    ion_counts.append(f"{other_ion_name}:{count}")
                            
                            # Create coordination environment string: "Na:1_Cl:2" format
                            coord_env = "_".join(ion_counts)
                        
                        # Update counters
                        shell_data = results[ion_name]['shell_regions'][shell_name]
                        if coord_env not in shell_data['coordination_environments']:
                            shell_data['coordination_environments'][coord_env] = 0
                        
                        shell_data['coordination_environments'][coord_env] += 1
                        shell_data['total_observations'] += 1
            
            # Calculate probabilities for each shell region
            for shell_name, shell_data in results[ion_name]['shell_regions'].items():
                total_obs = shell_data['total_observations']
                
                if total_obs > 0:
                    # Convert counts to probabilities
                    probabilities = {}
                    for coord_env, count in shell_data['coordination_environments'].items():
                        probabilities[coord_env] = count / total_obs
                    
                    shell_data['probabilities'] = probabilities
                    
                    # Find most probable coordination environment
                    most_prob_env = max(probabilities.keys(), key=probabilities.get)
                    shell_data['most_probable_environment'] = most_prob_env
                    shell_data['max_probability'] = probabilities[most_prob_env]
                    
                    # Print results with appropriate interpretation
                    analysis_type = shell_data['analysis_type']
                    print(f"    {shell_name} ({shell_data['bounds'][0]:.2f}-{shell_data['bounds'][1]:.2f} Å) - {analysis_type}:")
                    
                    if analysis_type == 'full':
                        print(f"      Most probable: {most_prob_env} ({probabilities[most_prob_env]:.1%}) [coions-waters]")
                    else:
                        print(f"      Most probable: {most_prob_env} ({probabilities[most_prob_env]:.1%}) [specific ions]")
                    
                    print(f"      Total observations: {total_obs}")
                    
                    # Show top 3 environments
                    top_envs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
                    for j, (env, prob) in enumerate(top_envs):
                        print(f"        {j+1}. {env}: {prob:.1%}")
                else:
                    print(f"    {shell_name}: No observations")
        
        # Store results
        self.shell_region_coordination_probabilities = results

        # Optionally persist to disk
        if save_cache and results:
            self.save_shell_region_coordination_probabilities_to_file(_cache_file)

        return results

    # ------------------------------------------------------------------
    def plot_shell_region_coordination_by_type(
        self, results=None, save_plots=True,
        figsize_per_ion=(5, 4),
        font_size=10, font_weight='normal',
        show_title=True, title_font_size=12, title_font_weight='bold',
        label_font_size=None, label_font_weight=None,
        tick_font_size=None, tick_font_weight=None,
        bar_label_font_size=None, bar_label_font_weight=None,
        min_plot_pct=0.0, min_label_pct=5.0,
        bar_width=0.6, bar_alpha=0.7, bar_linewidth=1.0,
        bar_color_cation='steelblue', bar_color_anion='crimson',
        ion_colors=None,
        bar_hatches=None,
        edgecolor='black', grid_alpha=0.3,
        dpi=300, output_filename=None,
        save_combined=True, save_individual=False, transparent=False,
        xlabel='Coordination Environment',
        # --- legend ---
        show_legend=True,
        legend_font_size=None,
        legend_font_weight=None,
        legend_bbox_to_anchor=None,
        legend_frame_alpha=None,
        # --- ordering ---
        bar_order='probability',    # 'probability' (highest first) | 'label' (alphabetical)
        # --- bar value label control ---
        show_bar_label=True,        # show percentage labels on bars
        show_pct_mark=True,         # include '%' in bar labels; False → numbers only
    ):
        '''
        Plot shell region coordination probabilities — plotting only, no analysis.

        Parameters
        ----------
        results : dict or None
            As returned by ``compute_shell_region_coordination_by_type``.
            If None, reads from ``self.shell_region_coordination_probabilities``.
        save_plots : bool
            Whether to save figures to disk (default=True).
        (All other style parameters are identical to those in
        ``shell_coordination_probabilities_by_shell_region_by_type``.)
        '''
        if results is None:
            if not hasattr(self, 'shell_region_coordination_probabilities') or \
                    not self.shell_region_coordination_probabilities:
                raise ValueError(
                    "No cached results found. Run "
                    "compute_shell_region_coordination_by_type() first "
                    "or pass results explicitly.")
            results = self.shell_region_coordination_probabilities

        _lbl    = label_font_size       if label_font_size       is not None else font_size
        _lbl_w  = label_font_weight     if label_font_weight     is not None else font_weight
        _tick   = tick_font_size        if tick_font_size        is not None else font_size
        _tick_w = tick_font_weight      if tick_font_weight      is not None else font_weight
        _blbl   = bar_label_font_size   if bar_label_font_size   is not None else max(font_size - 2, 6)
        _blbl_w = bar_label_font_weight if bar_label_font_weight is not None else font_weight
        self._plot_shell_region_coordination_probabilities(
            results, save_plots,
            figsize_per_ion=figsize_per_ion,
            font_size=font_size, font_weight=font_weight,
            show_title=show_title,
            title_font_size=title_font_size, title_font_weight=title_font_weight,
            label_font_size=_lbl, label_font_weight=_lbl_w,
            tick_font_size=_tick, tick_font_weight=_tick_w,
            bar_label_font_size=_blbl, bar_label_font_weight=_blbl_w,
            min_plot_pct=min_plot_pct, min_label_pct=min_label_pct,
            bar_width=bar_width, bar_alpha=bar_alpha, bar_linewidth=bar_linewidth,
            bar_color_cation=bar_color_cation, bar_color_anion=bar_color_anion,
            edgecolor=edgecolor, grid_alpha=grid_alpha,
            dpi=dpi, output_filename=output_filename,
            save_combined=save_combined, save_individual=save_individual,
            transparent=transparent,
            xlabel=xlabel,
            ion_colors=ion_colors,
            bar_hatches=bar_hatches,
            show_legend=show_legend,
            legend_font_size=legend_font_size,
            legend_font_weight=legend_font_weight,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
            legend_frame_alpha=legend_frame_alpha,
            bar_order=bar_order,
            show_bar_label=show_bar_label,
            show_pct_mark=show_pct_mark,
        )

    # ------------------------------------------------------------------
    def shell_coordination_probabilities_by_shell_region_by_type(
        self, ion_type=None, plot=False, save_plots=True, step=None, use_kdtree=True,
        save_cache=False, cache_filename=None, force_rerun=False,
        figsize_per_ion=(5, 4),
        font_size=10, font_weight='normal',
        show_title=True, title_font_size=12, title_font_weight='bold',
        label_font_size=None, label_font_weight=None,
        tick_font_size=None, tick_font_weight=None,
        bar_label_font_size=None, bar_label_font_weight=None,
        min_plot_pct=0.0, min_label_pct=5.0,
        bar_width=0.6, bar_alpha=0.7, bar_linewidth=1.0,
        bar_color_cation='steelblue', bar_color_anion='crimson',
        ion_colors=None,
        bar_hatches=None,
        edgecolor='black', grid_alpha=0.3,
        dpi=300, output_filename=None,
        save_combined=True, save_individual=False, transparent=False,
        xlabel='Coordination Environment',
        # --- legend ---
        show_legend=True,
        legend_font_size=None,
        legend_font_weight=None,
        legend_bbox_to_anchor=None,
        legend_frame_alpha=None,
        # --- ordering ---
        bar_order='probability',    # 'probability' (highest first) | 'label' (alphabetical)
        # --- bar value label control ---
        show_bar_label=True,        # show percentage labels on bars
        show_pct_mark=True,         # include '%' in bar labels; False → numbers only
    ):
        '''
        Convenience wrapper: compute + plot in one call.

        Calls ``compute_shell_region_coordination_by_type`` then optionally
        ``plot_shell_region_coordination_by_type``.  Use the two separate methods
        when you want to adjust plot aesthetics without re-running the analysis.
        '''
        results = self.compute_shell_region_coordination_by_type(
            ion_type=ion_type, step=step, use_kdtree=use_kdtree,
            save_cache=save_cache, cache_filename=cache_filename, force_rerun=force_rerun,
        )
        if plot and results:
            self.plot_shell_region_coordination_by_type(
                results=results, save_plots=save_plots,
                figsize_per_ion=figsize_per_ion,
                font_size=font_size, font_weight=font_weight,
                show_title=show_title,
                title_font_size=title_font_size, title_font_weight=title_font_weight,
                label_font_size=label_font_size, label_font_weight=label_font_weight,
                tick_font_size=tick_font_size, tick_font_weight=tick_font_weight,
                bar_label_font_size=bar_label_font_size, bar_label_font_weight=bar_label_font_weight,
                min_plot_pct=min_plot_pct, min_label_pct=min_label_pct,
                bar_width=bar_width, bar_alpha=bar_alpha, bar_linewidth=bar_linewidth,
                bar_color_cation=bar_color_cation, bar_color_anion=bar_color_anion,
                edgecolor=edgecolor, grid_alpha=grid_alpha,
                dpi=dpi, output_filename=output_filename,
                save_combined=save_combined, save_individual=save_individual,
                transparent=transparent,
                xlabel=xlabel,
                ion_colors=ion_colors,
                bar_hatches=bar_hatches,
                show_legend=show_legend,
                legend_font_size=legend_font_size,
                legend_font_weight=legend_font_weight,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                legend_frame_alpha=legend_frame_alpha,
                bar_order=bar_order,
                show_bar_label=show_bar_label,
                show_pct_mark=show_pct_mark,
            )
        return results


    def load_shell_region_coordination_probabilities_from_file(self, filename='shell_region_coordination_probs_cache.pkl'):
        '''
        Load shell region coordination probabilities from file with error handling.
        
        Parameters
        ----------
        filename : str
            Input filename, default='shell_region_coordination_probs_cache.pkl'
        
        Returns
        -------
        success : bool
            True if load was successful
        '''
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        try:
            # Check file size first
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print(f"Cache file {filename} is empty")
                return False
            
            # Load data
            with open(filename, 'rb') as f:
                region_data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(region_data, dict):
                print(f"Invalid shell region coordination cache format")
                return False
            
            # Reconstruct the shell region coordination data structure
            self.shell_region_coordination_probabilities = {}
            
            for ion_type, ion_data in region_data.items():
                self.shell_region_coordination_probabilities[ion_type] = {
                    'ion_category': ion_data['ion_category'],
                    'coordination_radius': ion_data['coordination_radius'],
                    'shell_regions': {}
                }
                
                for shell_name, shell_data in ion_data['shell_regions'].items():
                    self.shell_region_coordination_probabilities[ion_type]['shell_regions'][shell_name] = {
                        'bounds': shell_data['bounds'],
                        'coordination_environments': shell_data['coordination_environments'],
                        'total_observations': shell_data['total_observations'],
                        'analysis_type': shell_data['analysis_type'],
                        'probabilities': shell_data.get('probabilities', {}),
                        'most_probable_environment': shell_data.get('most_probable_environment', None),
                        'max_probability': shell_data.get('max_probability', 0.0)
                    }
            
            # Print summary
            successful_types = list(self.shell_region_coordination_probabilities.keys())
            
            print(f"Shell region coordination probabilities loaded from {filename}")
            print(f"  Loaded {len(successful_types)} ion types successfully")
            if successful_types:
                print(f"  Available types: {', '.join(successful_types)}")
            
            # Print detailed summary
            print(f"\n  Shell region coordination summary:")
            for ion_type, ion_data in self.shell_region_coordination_probabilities.items():
                region_count = len(ion_data['shell_regions'])
                print(f"    {ion_type} ({ion_data['ion_category']}): {region_count} shell regions")
                for shell_name, shell_data in ion_data['shell_regions'].items():
                    start, end = shell_data['bounds']
                    end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                    total_obs = shell_data['total_observations']
                    analysis_type = shell_data['analysis_type']
                    most_prob = shell_data.get('most_probable_environment', 'N/A')
                    max_prob = shell_data.get('max_probability', 0.0)
                    
                    print(f"      {shell_name} ({start:.2f}-{end_str} Å, {analysis_type}):")
                    print(f"        Total observations: {total_obs}")
                    print(f"        Most probable: {most_prob} ({max_prob:.1%})")
            
            return True
            
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            print("Removing corrupted cache file...")
            try:
                os.remove(filename)
                print(f"Corrupted file {filename} removed")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading shell region coordination probabilities from {filename}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def shell_coordination_probabilities_by_shell_region_by_type_with_cache(self, cache_filename='shell_region_coordination_probs_cache.pkl', 
                                                                            force_recalc=False, **kwargs):
        '''
        Calculate shell region coordination probabilities with automatic caching.
        
        Parameters
        ----------
        cache_filename : str
            Cache file name, default='shell_region_coordination_probs_cache.pkl'
        force_recalc : bool
            Force recalculation even if cache exists, default=False
        **kwargs : dict
            Arguments to pass to shell_coordination_probabilities_by_shell_region_by_type()
        
        Returns
        -------
        results : dict
            Dictionary of shell region coordination probability results
        '''
        
        # Try to load from cache first
        if not force_recalc and os.path.exists(cache_filename):
            print("Attempting to load shell region coordination probabilities from cache...")
            if self.load_shell_region_coordination_probabilities_from_file(cache_filename):
                print("✓ Successfully loaded shell region coordination probabilities from cache")
                return self.shell_region_coordination_probabilities
            else:
                print("✗ Cache loading failed, will recalculate")
        
        # Calculate shell region coordination probabilities
        print("Calculating shell region coordination probabilities...")
        results = self.shell_coordination_probabilities_by_shell_region_by_type(**kwargs)
        
        # Save to cache
        if results:
            print("Saving shell region coordination probabilities to cache...")
            if self.save_shell_region_coordination_probabilities_to_file(cache_filename):
                print("✓ Shell region coordination probabilities cached successfully")
            else:
                print("✗ Cache saving failed, but results are available in memory")
        
        return results



    def _plot_shell_region_coordination_probabilities(
        self, results, save_plots,
        figsize_per_ion=(5, 4),
        font_size=10, font_weight='normal',
        show_title=True, title_font_size=12, title_font_weight='bold',
        label_font_size=9, label_font_weight='normal',
        tick_font_size=8, tick_font_weight='normal',
        bar_label_font_size=8, bar_label_font_weight='bold',
        min_plot_pct=0.0, min_label_pct=5.0,
        bar_width=0.6, bar_alpha=0.7, bar_linewidth=1.0,
        bar_color_cation='steelblue', bar_color_anion='crimson',
        ion_colors=None,
        bar_hatches=None,
        edgecolor='black', grid_alpha=0.3,
        dpi=300, output_filename=None,
        save_combined=True, save_individual=False, transparent=False,
        xlabel='Coordination Environment',
        # --- legend ---
        show_legend=True,
        legend_font_size=None,
        legend_font_weight=None,
        legend_bbox_to_anchor=None,
        legend_frame_alpha=None,
        # --- ordering ---
        bar_order='probability',    # 'probability' (highest first) | 'label' (alphabetical)
        # --- bar value label control ---
        show_bar_label=True,        # show percentage labels on bars
        show_pct_mark=True,         # include '%' in bar labels; False → numbers only
    ):
        '''Plot shell region coordination probabilities - FIXED to show detailed labels for ALL shells'''

        n_ions = len(results)
        if n_ions == 0:
            return

        max_shells = max(len(data['shell_regions']) for data in results.values())

        cations_data = {k: v for k, v in results.items() if v['ion_category'] == 'cation'}
        anions_data  = {k: v for k, v in results.items() if v['ion_category'] == 'anion'}

        # ------------------------------------------------------------------ #
        # Pre-compute filtered data for every (ion, shell) slot so we can
        # find the global max_n_bars and give ALL panels the same xlim.
        # Same xlim + same axes physical width = same physical bar width.
        # ------------------------------------------------------------------ #
        def _filter_slot(ion_data, shell_name):
            '''Return (env_labels, env_probs) after sorting/capping/filtering.'''
            shell_data    = ion_data['shell_regions'][shell_name]
            probabilities = shell_data.get('probabilities', {})
            coord_envs    = shell_data.get('coordination_environments', {})
            if not coord_envs:
                return [], []
            env_labels = list(coord_envs.keys())
            env_probs  = [probabilities.get(e, 0.0) for e in env_labels]
            if bar_order == 'label':
                idx = np.argsort(env_labels)          # alphabetical
            else:
                idx = np.argsort(env_probs)[::-1]     # probability, highest first
            env_labels = [env_labels[i] for i in idx]
            env_probs  = [env_probs[i]  for i in idx]
            if len(env_labels) > 10:
                env_labels = env_labels[:10]
                env_probs  = env_probs[:10]
            if min_plot_pct > 0.0:
                keep       = [p * 100 >= min_plot_pct for p in env_probs]
                env_labels = [l for l, k in zip(env_labels, keep) if k]
                env_probs  = [p for p, k in zip(env_probs,  keep) if k]
            return env_labels, env_probs

        # Build cache and find global max bars across ALL ions & shells
        _slot_cache = {}   # (ion_name, shell_name) -> (env_labels, env_probs)
        all_ion_data = {**cations_data, **anions_data}
        for _ion_name, _ion_data in all_ion_data.items():
            for _shell_name in _ion_data['shell_regions']:
                _labels, _probs = _filter_slot(_ion_data, _shell_name)
                _slot_cache[(_ion_name, _shell_name)] = (_labels, _probs)

        _all_n_bars    = [len(v[0]) for v in _slot_cache.values() if v[0]]
        _global_max_nb = max(_all_n_bars) if _all_n_bars else 1
        _pad           = bar_width * 1.5   # symmetric padding; same for every panel
        _xlim          = (-_pad, _global_max_nb - 1 + _pad)

        def _resolve_hatch(shell_name, shell_idx, bar_hatches):
            '''Return a single hatch string for this shell panel, or None.'''
            if bar_hatches is None:
                return None
            if isinstance(bar_hatches, str):
                return bar_hatches  # same hatch for every shell
            if isinstance(bar_hatches, dict):
                return bar_hatches.get(shell_name, None)
            # list / sequence — indexed by shell order (0-based)
            if isinstance(bar_hatches, (list, tuple)):
                return bar_hatches[shell_idx % len(bar_hatches)] if bar_hatches else None
            return None

        def _draw_slot(ax, ion_name, ion_data, shell_name, bar_color, hatch_spec=None):
            '''Draw one (ion, shell) panel into ax using pre-computed data.'''
            shell_data    = ion_data['shell_regions'][shell_name]
            bounds        = shell_data['bounds']
            analysis_type = shell_data['analysis_type']

            env_labels, env_probs = _slot_cache[(ion_name, shell_name)]

            if not env_labels:
                msg = 'No data' if not shell_data.get('coordination_environments') else 'No data above threshold'
                ax.text(0.5, 0.5, msg, ha='center', va='center',
                        transform=ax.transAxes, fontsize=font_size)
                if show_title:
                    ax.set_title(f'{ion_name} {shell_name}',
                                 fontweight=title_font_weight, fontsize=title_font_size)
                return

            n_bars = len(env_labels)
            # Center this panel's bars within the global xlim so every panel
            # uses the same axis span → identical physical bar width everywhere.
            offset = (_global_max_nb - n_bars) / 2.0
            x_pos  = np.arange(n_bars) + offset

            ax.set_xlim(*_xlim)
            bars = ax.bar(x_pos, [p * 100 for p in env_probs],
                          width=bar_width, color=bar_color, alpha=bar_alpha,
                          edgecolor=edgecolor, linewidth=bar_linewidth,
                          hatch=hatch_spec or '', label=ion_name)

            max_prob = max(env_probs) * 100 if env_probs else 0

            _any_lbl_s = False
            if show_bar_label:
                for bar, prob in zip(bars, env_probs):
                    prob_percent = prob * 100
                    if prob_percent < min_label_pct:
                        continue
                    _any_lbl_s = True
                    height     = bar.get_height()
                    y_pos      = height + max_prob * 0.02
                    if show_pct_mark:
                        label_text = f'{prob_percent:.1f}%' if prob_percent >= 1.0 else f'{prob_percent:.2f}%'
                    else:
                        label_text = f'{prob_percent:.1f}' if prob_percent >= 1.0 else f'{prob_percent:.2f}'
                    ax.text(bar.get_x() + bar.get_width() / 2., y_pos, label_text,
                            ha='center', va='bottom',
                            fontweight=bar_label_font_weight, fontsize=bar_label_font_size,
                            color='black')

            ax.set_xticks(x_pos)
            ax.set_xticklabels(env_labels, rotation=45, ha='right',
                               fontsize=tick_font_size, fontweight=tick_font_weight)
            if xlabel is not None:
                ax.set_xlabel(xlabel,
                              fontsize=label_font_size, fontweight=label_font_weight)
            ax.set_ylabel('Probability (%)',
                          fontsize=label_font_size, fontweight=label_font_weight)
            plt.setp(ax.get_yticklabels(), fontsize=tick_font_size, fontweight=tick_font_weight)
            # Smart ylim: expand so bar labels don't clip at top
            _smart_top_s = max_prob * 1.15
            if _any_lbl_s and max_prob > 0:
                _ax_h_in_s = ax.get_figure().get_figheight() * ax.get_position().height
                if _ax_h_in_s > 0:
                    _dpui_s     = _smart_top_s / _ax_h_in_s
                    _lbl_h_du_s = (bar_label_font_size / 72.0) * _dpui_s * 2.0
                    _needed_s   = max_prob * 1.02 + _lbl_h_du_s
                    _smart_top_s = max(_smart_top_s, _needed_s * 1.05)
            ax.set_ylim(0, _smart_top_s)
            ax.grid(True, alpha=grid_alpha, axis='y')

            if show_legend:
                _leg_kw = dict(prop={
                    'size':   legend_font_size   if legend_font_size   is not None else font_size,
                    'weight': legend_font_weight if legend_font_weight is not None else font_weight,
                })
                if legend_bbox_to_anchor is not None:
                    _leg_kw['bbox_to_anchor'] = legend_bbox_to_anchor
                    _leg_kw['loc'] = 'upper right'
                if legend_frame_alpha is not None:
                    _leg_kw['framealpha'] = legend_frame_alpha
                ax.legend(**_leg_kw)

            end_str = f"{bounds[1]:.2f}" if not np.isinf(bounds[1]) else "∞"
            if analysis_type == 'full':
                title = f'{ion_name} {shell_name}\n({bounds[0]:.2f}-{end_str} Å, full coord.)'
            else:
                title = f'{ion_name} {shell_name}\n({bounds[0]:.2f}-{end_str} Å, specific ions)'
            if show_title:
                ax.set_title(title, fontweight=title_font_weight, fontsize=title_font_size)

        # ------------------------------------------------------------------ #
        # Cations combined figure
        # ------------------------------------------------------------------ #
        if cations_data:
            n_cations = len(cations_data)
            fig, axes = plt.subplots(
                max_shells, n_cations,
                figsize=(figsize_per_ion[0] * n_cations, figsize_per_ion[1] * max_shells),
            )
            if n_cations == 1:
                axes = axes.reshape(-1, 1) if max_shells > 1 else [[axes]]
            elif max_shells == 1:
                axes = axes.reshape(1, -1)

            if show_title:
                fig.suptitle('Cation Shell Region Coordination Environments',
                             fontsize=title_font_size, fontweight=title_font_weight)

            for col, (ion_name, ion_data) in enumerate(cations_data.items()):
                _color  = ion_colors.get(ion_name, bar_color_cation) if ion_colors else bar_color_cation
                shell_names = sorted(ion_data['shell_regions'].keys())
                for row, shell_name in enumerate(shell_names):
                    _hatch = _resolve_hatch(shell_name, row, bar_hatches)
                    ax = axes[row][col] if n_cations > 1 or max_shells > 1 else axes[0][0]
                    _draw_slot(ax, ion_name, ion_data, shell_name, _color, _hatch)
                for row in range(len(shell_names), max_shells):
                    ax = axes[row][col] if n_cations > 1 or max_shells > 1 else axes[0][0]
                    ax.set_visible(False)

            plt.tight_layout()

            if save_combined:
                fname = (output_filename if output_filename
                         else 'cation_shell_region_coordination_probabilities.png')
                plt.savefig(fname, dpi=dpi, bbox_inches='tight', transparent=transparent)
                print(f"Cation plot saved as: {fname}")

            plt.show()

            if save_individual:
                for ion_name, ion_data in cations_data.items():
                    _color = ion_colors.get(ion_name, bar_color_cation) if ion_colors else bar_color_cation
                    for row, shell_name in enumerate(sorted(ion_data['shell_regions'].keys())):
                        _hatch = _resolve_hatch(shell_name, row, bar_hatches)
                        fig_ind, ax_ind = plt.subplots(1, 1, figsize=figsize_per_ion)
                        _draw_slot(ax_ind, ion_name, ion_data, shell_name, _color, _hatch)
                        plt.tight_layout()
                        ind_fname = f'{ion_name}_{shell_name}_region_coordination.png'
                        fig_ind.savefig(ind_fname, dpi=dpi, bbox_inches='tight', transparent=transparent)
                        plt.close(fig_ind)

        # ------------------------------------------------------------------ #
        # Anions combined figure
        # ------------------------------------------------------------------ #
        if anions_data:
            n_anions = len(anions_data)
            fig, axes = plt.subplots(
                max_shells, n_anions,
                figsize=(figsize_per_ion[0] * n_anions, figsize_per_ion[1] * max_shells),
            )
            if n_anions == 1:
                axes = axes.reshape(-1, 1) if max_shells > 1 else [[axes]]
            elif max_shells == 1:
                axes = axes.reshape(1, -1)

            if show_title:
                fig.suptitle('Anion Shell Region Coordination Environments',
                             fontsize=title_font_size, fontweight=title_font_weight)

            for col, (ion_name, ion_data) in enumerate(anions_data.items()):
                _color  = ion_colors.get(ion_name, bar_color_anion) if ion_colors else bar_color_anion
                shell_names = sorted(ion_data['shell_regions'].keys())
                for row, shell_name in enumerate(shell_names):
                    _hatch = _resolve_hatch(shell_name, row, bar_hatches)
                    ax = axes[row][col] if n_anions > 1 or max_shells > 1 else axes[0][0]
                    _draw_slot(ax, ion_name, ion_data, shell_name, _color, _hatch)
                for row in range(len(shell_names), max_shells):
                    ax = axes[row][col] if n_anions > 1 or max_shells > 1 else axes[0][0]
                    ax.set_visible(False)

            plt.tight_layout()

            if save_combined:
                fname = (output_filename if output_filename
                         else 'anion_shell_region_coordination_probabilities.png')
                plt.savefig(fname, dpi=dpi, bbox_inches='tight', transparent=transparent)
                print(f"Anion plot saved as: {fname}")

            plt.show()

            if save_individual:
                for ion_name, ion_data in anions_data.items():
                    _color = ion_colors.get(ion_name, bar_color_anion) if ion_colors else bar_color_anion
                    for row, shell_name in enumerate(sorted(ion_data['shell_regions'].keys())):
                        _hatch = _resolve_hatch(shell_name, row, bar_hatches)
                        fig_ind, ax_ind = plt.subplots(1, 1, figsize=figsize_per_ion)
                        _draw_slot(ax_ind, ion_name, ion_data, shell_name, _color, _hatch)
                        plt.tight_layout()
                        ind_fname = f'{ion_name}_{shell_name}_region_coordination.png'
                        fig_ind.savefig(ind_fname, dpi=dpi, bbox_inches='tight', transparent=transparent)
                        plt.close(fig_ind)


    def get_shell_region_coordination_summary(self):
        '''Print summary of shell region coordination probabilities'''
        
        if not hasattr(self, 'shell_region_coordination_probabilities'):
            print("Shell region coordination probabilities not calculated.")
            print("Run shell_coordination_probabilities_by_shell_region_by_type() first.")
            return
        
        results = self.shell_region_coordination_probabilities
        
        print("\n" + "="*80)
        print("SHELL REGION COORDINATION ENVIRONMENT SUMMARY")
        print("="*80)
        
        for ion_type, ion_data in results.items():
            ion_category = ion_data['ion_category']
            coordination_radius = ion_data['coordination_radius']
            
            print(f"\n{ion_type.upper()} ({ion_category.title()}) - Coordination radius: {coordination_radius:.2f} Å")
            print("-" * 60)
            
            for shell_name, shell_data in ion_data['shell_regions'].items():
                bounds = shell_data['bounds']
                total_obs = shell_data['total_observations']
                most_prob = shell_data.get('most_probable_environment', 'None')
                max_prob = shell_data.get('max_probability', 0)
                
                print(f"{shell_name} ({bounds[0]:.2f}-{bounds[1]:.2f} Å):")
                print(f"  Most probable: {most_prob} ({max_prob:.1%})")
                print(f"  Total observations: {total_obs}")
                
                # Show top coordination environments
                probabilities = shell_data.get('probabilities', {})
                if probabilities:
                    top_envs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
                    for i, (env, prob) in enumerate(top_envs):
                        print(f"    {i+1}. {env}: {prob:.1%}")
                print()
        
        print("="*80)


    def save_shell_region_coordination_probabilities_to_file(self, filename='shell_region_coordination_probs_cache.pkl'):
        '''
        Save shell region coordination probabilities to file for persistence across sessions.
        
        Parameters
        ----------
        filename : str
            Output filename, default='shell_region_coordination_probs_cache.pkl'
        
        Returns
        -------
        success : bool
            True if save was successful
        '''
        
        if not hasattr(self, 'shell_region_coordination_probabilities') or not self.shell_region_coordination_probabilities:
            print("No shell region coordination probabilities to save")
            return False
        
        try:
            # Prepare shell region coordination data for serialization
            region_data = {}
            
            for ion_type, ion_data in self.shell_region_coordination_probabilities.items():
                region_data[ion_type] = {
                    'ion_category': ion_data['ion_category'],
                    'coordination_radius': ion_data['coordination_radius'],
                    'shell_regions': {}
                }
                
                for shell_name, shell_data in ion_data['shell_regions'].items():
                    region_data[ion_type]['shell_regions'][shell_name] = {
                        'bounds': shell_data['bounds'],
                        'coordination_environments': dict(shell_data['coordination_environments']),
                        'total_observations': shell_data['total_observations'],
                        'analysis_type': shell_data['analysis_type'],
                        'probabilities': shell_data.get('probabilities', {}),
                        'most_probable_environment': shell_data.get('most_probable_environment', None),
                        'max_probability': shell_data.get('max_probability', 0.0)
                    }
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(region_data, f)
            
            print(f"Shell region coordination probabilities saved to {filename}")
            print(f"  Saved {len(region_data)} ion types")
            print(f"  Ion types: {list(region_data.keys())}")
            
            # Print summary
            for ion_type, data in region_data.items():
                n_regions = len(data['shell_regions'])
                print(f"    {ion_type}: {n_regions} shell regions")
            
            return True
            
        except Exception as e:
            print(f"Error saving shell region coordination probabilities: {e}")
            import traceback
            traceback.print_exc()
            return False


    def compute_ion_pairing_probabilities_by_coordination(self, ion_type, coordination_states=None, step=None, save_plots=True, include_zero_states=True):
        '''
        Compute ion pairing probabilities for ions in specific coordination states.
        Analyzes how coordination number affects ion pairing behavior.
        
        Parameters
        ----------
        ion_type : str
            Ion type to analyze (e.g., 'Na', 'Mg', 'Cl', 'cation', 'anion')
        coordination_states : list or None
            List of coordination numbers to analyze (e.g., [4, 5, 6, 7]).
            If None, analyzes all observed coordination states.
        step : int
            Step size for trajectory analysis, default uses auto-tuned value
        save_plots : bool
            Whether to save plots, default=True
        include_zero_states : bool
            Whether to include requested coordination states with zero observations, default=True
        
        Returns
        -------
        results : dict
            Dictionary with coordination states as keys and pairing probabilities as values
        '''
        
        if step is None:
            step = self.default_step
        
        # Check if we have ion pairing cutoffs for this ion type
        if not hasattr(self, 'ion_pairs_by_type') or ion_type not in self.ion_pairs_by_type:
            print(f"No ion pairing cutoffs found for {ion_type}.")
            print("Run determine_ion_pairing_cutoffs() first.")
            return None
        
        # Get ion pairing regions
        pairing_data = self.ion_pairs_by_type[ion_type]
        ion_pairs = pairing_data['ion_pairs']
        ion_category = pairing_data['ion_category']
        
        # Get ions and coordination radius
        if ion_type in ['cation', 'anion']:
            # Handle broad categories
            ions = self.cations if ion_type == 'cation' else self.anions
            partner_ions = self.anions if ion_type == 'cation' else self.cations
            
            # Get coordination radius
            try:
                if ion_type == 'cation':
                    r0 = self.solute_ci.radii['water']
                else:
                    r0 = self.solute_ai.radii['water']
            except (AttributeError, NameError):
                print(f"Solutes not initialized for {ion_type}")
                return None
        else:
            # Handle specific ion types
            cation_types = self._get_unique_ion_types(self.cations)
            anion_types = self._get_unique_ion_types(self.anions)
            
            if ion_type in cation_types:
                ions = cation_types[ion_type]
                partner_ions = self.anions
                ion_category = 'cation'
                
                # Get coordination radius
                if (hasattr(self, 'solutes_ci') and ion_type in self.solutes_ci and 
                    self.solutes_ci[ion_type] is not None):
                    r0 = self.solutes_ci[ion_type].radii['water']
                else:
                    print(f"No solute data for {ion_type}")
                    return None
                    
            elif ion_type in anion_types:
                ions = anion_types[ion_type]
                partner_ions = self.cations
                ion_category = 'anion'
                
                # Get coordination radius
                if (hasattr(self, 'solutes_ai') and ion_type in self.solutes_ai and 
                    self.solutes_ai[ion_type] is not None):
                    r0 = self.solutes_ai[ion_type].radii['water']
                else:
                    print(f"No solute data for {ion_type}")
                    return None
            else:
                print(f"Ion type '{ion_type}' not found")
                return None
        
        print(f"Computing ion pairing probabilities by coordination for {ion_type} ({ion_category})")
        print(f"Using coordination radius: {r0:.2f} Å")
        print(f"Analyzing {len(ions)} ions over trajectory with step={step}")
        
        # Handle debug trajectory
        if hasattr(self, '_debug_frame_indices') and self._debug_frame_indices is not None:
            frame_indices = self._debug_frame_indices[::step]
            print(f"Debug mode: analyzing {len(frame_indices)} frames")
        else:
            frame_indices = range(0, len(self.universe.trajectory), step)
        
        # Storage for data collection
        coordination_pairing_data = {}  # {coordination_number: {'CIP': count, 'SIP': count, ...}}
        
        print("Collecting coordination and pairing data...")
        
        # Analyze each frame
        for frame_idx in tqdm(frame_indices, desc=f"Analyzing {ion_type} coordination-pairing"):
            # Set trajectory to frame
            ts = self.universe.trajectory[frame_idx]
            
            for ion in ions:
                # Get coordinating waters
                coordinating_waters = self.universe.select_atoms(f'sphzone {r0} index {ion.index}') & self.waters
                coordination_number = len(coordinating_waters)
                
                # Find closest partner ion and classify pairing
                ion_distances = cdist([ion.position], partner_ions.positions)[0]
                closest_distance = ion_distances.min()
                
                # Classify ion pairing based on closest partner distance
                pairing_type = None
                for pair_type, (r_min, r_max) in ion_pairs.items():
                    if r_min <= closest_distance < r_max:
                        pairing_type = pair_type
                        break
                
                if pairing_type is None:
                    continue  # Skip if distance doesn't fit any category
                
                # Record data
                if coordination_number not in coordination_pairing_data:
                    coordination_pairing_data[coordination_number] = {pair_type: 0 for pair_type in ion_pairs.keys()}
                
                coordination_pairing_data[coordination_number][pairing_type] += 1
        
        if not coordination_pairing_data:
            print("No data collected. Check that ion pairing cutoffs and coordination radius are reasonable.")
            return None
        
        # MODIFIED: Handle coordination states with better zero-state inclusion
        if coordination_states is None:
            # Use all observed coordination states
            coordination_states = sorted(coordination_pairing_data.keys())
            print(f"Using all observed coordination states: {coordination_states}")
        else:
            # Check which requested states have observations
            observed_states = [cn for cn in coordination_states if cn in coordination_pairing_data]
            zero_states = [cn for cn in coordination_states if cn not in coordination_pairing_data]
            
            print(f"Requested coordination states: {coordination_states}")
            print(f"States with observations: {observed_states}")
            if zero_states:
                print(f"States with zero observations: {zero_states}")
            
            if include_zero_states and zero_states:
                # Add zero entries for unobserved states
                for cn in zero_states:
                    coordination_pairing_data[cn] = {pair_type: 0 for pair_type in ion_pairs.keys()}
                    print(f"  Added zero data for coordination state {cn}")
        
        # Compute probabilities for all requested states (including zeros)
        results = {}
        
        for cn in coordination_states:
            if cn in coordination_pairing_data:
                counts = coordination_pairing_data[cn]
                total_counts = sum(counts.values())
                
                if total_counts > 0:
                    probabilities = {pair_type: count/total_counts for pair_type, count in counts.items()}
                else:
                    # Handle zero observation case
                    probabilities = {pair_type: 0.0 for pair_type in ion_pairs.keys()}
                
                results[cn] = {
                    'probabilities': probabilities,
                    'counts': counts,
                    'total_observations': total_counts,
                    'coordination_number': cn
                }
        
        # Store results
        if not hasattr(self, 'coordination_pairing_analysis'):
            self.coordination_pairing_analysis = {}
        
        self.coordination_pairing_analysis[ion_type] = results
        
        # Print summary
        self._print_coordination_pairing_summary(ion_type, results)
        
        # Create plots
        if save_plots and results:
            self._plot_coordination_pairing_probabilities(ion_type, results, ion_category)
        
        return results


    def save_dipole_distributions_to_file(self, filename='dipole_distributions_cache.pkl'):
        '''
        Save water dipole distributions to file for persistence across sessions.
        
        Parameters
        ----------
        filename : str
            Output filename, default='dipole_distributions_cache.pkl'
        
        Returns
        -------
        success : bool
            True if save was successful
        '''
        
        if not hasattr(self, 'dipole_distributions_by_type') or not self.dipole_distributions_by_type:
            print("No dipole distributions to save")
            return False
        
        try:
            # Prepare dipole distribution data for serialization
            dipole_data = {}
            
            for ion_type, angles in self.dipole_distributions_by_type.items():
                dipole_data[ion_type] = {
                    'angles': angles.copy(),
                    'n_samples': len(angles),
                    'mean_angle': angles.mean(),
                    'std_angle': angles.std(),
                    'median_angle': np.median(angles),
                    'min_angle': angles.min(),
                    'max_angle': angles.max()
                }
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(dipole_data, f)
            
            print(f"Dipole distributions saved to {filename}")
            print(f"  Saved {len(dipole_data)} ion types")
            print(f"  Ion types: {list(dipole_data.keys())}")
            
            # Print summary
            for ion_type, data in dipole_data.items():
                print(f"    {ion_type}: {data['n_samples']} samples, mean={data['mean_angle']:.1f}°±{data['std_angle']:.1f}°")
            
            return True
            
        except Exception as e:
            print(f"Error saving dipole distributions: {e}")
            traceback.print_exc()
            return False

    def load_dipole_distributions_from_file(self, filename='dipole_distributions_cache.pkl'):
        '''
        Load water dipole distributions from file with error handling.
        
        Parameters
        ----------
        filename : str
            Input filename, default='dipole_distributions_cache.pkl'
        
        Returns
        -------
        success : bool
            True if load was successful
        '''
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        try:
            # Check file size first
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print(f"Cache file {filename} is empty")
                return False
            
            # Load data
            with open(filename, 'rb') as f:
                dipole_data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(dipole_data, dict):
                print(f"Invalid dipole distribution cache format")
                return False
            
            # Reconstruct the dipole distributions structure
            self.dipole_distributions_by_type = {}
            
            for ion_type, data in dipole_data.items():
                self.dipole_distributions_by_type[ion_type] = data['angles']
            
            # Print summary
            successful_types = list(self.dipole_distributions_by_type.keys())
            
            print(f"Dipole distributions loaded from {filename}")
            print(f"  Loaded {len(successful_types)} ion types successfully")
            if successful_types:
                print(f"  Available types: {', '.join(successful_types)}")
            
            # Print detailed summary
            print(f"\n  Dipole distribution summary:")
            for ion_type, data in dipole_data.items():
                print(f"    {ion_type}: {data['n_samples']} samples")
                print(f"      Mean angle: {data['mean_angle']:.1f}° ± {data['std_angle']:.1f}°")
                print(f"      Median: {data['median_angle']:.1f}°")
                print(f"      Range: {data['min_angle']:.1f}° - {data['max_angle']:.1f}°")
            
            return True
            
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            print("Removing corrupted cache file...")
            try:
                os.remove(filename)
                print(f"Corrupted file {filename} removed")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading dipole distributions from {filename}: {e}")
            traceback.print_exc()
            return False

    def water_dipole_distribution_with_cache(self, cache_filename='dipole_distributions_cache.pkl', 
                                            force_recalc=False, **kwargs):
        '''
        Calculate water dipole distributions with automatic caching.
        
        Parameters
        ----------
        cache_filename : str
            Cache file name, default='dipole_distributions_cache.pkl'
        force_recalc : bool
            Force recalculation even if cache exists, default=False
        **kwargs : dict
            Arguments to pass to water_dipole_distribution()
        
        Returns
        -------
        results : dict
            Dictionary of dipole distribution results
        '''
        
        # Get ion_type from kwargs if specified
        ion_type = kwargs.get('ion_type', None)
        
        # Try to load from cache first
        if not force_recalc and os.path.exists(cache_filename):
            print("Attempting to load dipole distributions from cache...")
            if self.load_dipole_distributions_from_file(cache_filename):
                # Check if we have the requested ion type
                if ion_type is not None:
                    if ion_type in self.dipole_distributions_by_type:
                        print(f"✓ Successfully loaded dipole distribution for {ion_type} from cache")
                        return self.dipole_distributions_by_type[ion_type]
                    else:
                        print(f"✗ {ion_type} not found in cache, will calculate")
                else:
                    print("✓ Successfully loaded all dipole distributions from cache")
                    return self.dipole_distributions_by_type
            else:
                print("✗ Cache loading failed, will recalculate")
        
        # Calculate dipole distributions
        print("Calculating water dipole distributions...")
        
        if ion_type is not None:
            # Single ion type
            angles = self.water_dipole_distribution(**kwargs)
            
            # Ensure the dipole_distributions_by_type dict exists
            if not hasattr(self, 'dipole_distributions_by_type'):
                self.dipole_distributions_by_type = {}
            
            # Store result
            self.dipole_distributions_by_type[ion_type] = angles
            
            # Save to cache
            print("Saving dipole distributions to cache...")
            if self.save_dipole_distributions_to_file(cache_filename):
                print("✓ Dipole distributions cached successfully")
            else:
                print("✗ Cache saving failed, but results are available in memory")
            
            return angles
        else:
            # Multiple ion types - calculate all
            if not hasattr(self, 'dipole_distributions_by_type'):
                self.dipole_distributions_by_type = {}
            
            # Get all ion types
            cation_types = list(self._get_unique_ion_types(self.cations).keys())
            anion_types = list(self._get_unique_ion_types(self.anions).keys())
            all_ion_types = cation_types + anion_types
            
            for ion in all_ion_types:
                print(f"Calculating dipoles for {ion}...")
                angles = self.water_dipole_distribution(ion_type=ion, **{k: v for k, v in kwargs.items() if k != 'ion_type'})
                self.dipole_distributions_by_type[ion] = angles
            
            # Save to cache
            print("Saving all dipole distributions to cache...")
            if self.save_dipole_distributions_to_file(cache_filename):
                print("✓ All dipole distributions cached successfully")
            else:
                print("✗ Cache saving failed, but results are available in memory")
            
            return self.dipole_distributions_by_type



    def compute_water_dipole_by_coordination(self, ion_type='cation', radius=None, step=None,
                                             coordination_states=None,
                                             save_cache=False,
                                             cache_filename=None,
                                             force_rerun=False):
        '''
        Compute water dipole angle distributions grouped by coordination number.

        This is the **analysis-only** step — no plotting is done here.
        Results are stored on ``self.dipole_by_coordination[ion_type]`` and returned
        so they can be passed directly to ``plot_water_dipole_by_coordination``.

        Parameters
        ----------
        ion_type : str or list
            Ion type(s) to analyse: single name (e.g. ``'Na'``), a list
            (e.g. ``['Na', 'Mg', 'Cl']``), or broad category ``'cation'``/``'anion'``.
        radius : float, optional
            Coordination shell radius in Å. If None, uses the ion-type-specific radius.
        step : int, optional
            Frame stride for trajectory iteration (default: ``self.default_step``).
        coordination_states : list or None
            Restrict results to these coordination numbers (e.g. ``[5, 6, 7]``).
            If None, all observed coordination numbers are returned.
        save_cache : bool
            If True, save results to a pickle file after calculation (default=False).
        cache_filename : str or None
            Path for the cache file.  If None, auto-generated as
            ``'dipole_by_coordination_<ion_type>.pkl'``.
        force_rerun : bool
            If True, ignore an existing cache file and recompute (default=False).

        Returns
        -------
        results : dict
            ``{cn: np.ndarray of angles}`` for a single ion type, or
            ``{ion_type: {cn: np.ndarray}}`` when a list is supplied.
        '''

        # Handle list of ion types
        if isinstance(ion_type, list):
            print(f"Computing dipole-by-coordination for {len(ion_type)} ion types...")
            all_results = {}
            for single_ion in ion_type:
                print(f"\n{'='*60}\nProcessing {single_ion}\n{'='*60}")
                single_result = self.compute_water_dipole_by_coordination(
                    ion_type=single_ion,
                    radius=radius,
                    step=step,
                    coordination_states=coordination_states,
                    save_cache=save_cache,
                    cache_filename=cache_filename,
                    force_rerun=force_rerun,
                )
                if single_result is not None:
                    all_results[single_ion] = single_result
            return all_results

        # --- resolve cache filename for single ion ---
        if cache_filename is None:
            _cache_file = f'dipole_by_coordination_{ion_type}.pkl'
        else:
            _cache_file = cache_filename

        _cache_abs = os.path.abspath(_cache_file)

        # --- try in-memory cache first (same session, no rerun needed) ---
        if (not force_rerun
                and hasattr(self, 'dipole_by_coordination')
                and ion_type in self.dipole_by_coordination):
            _mem = self.dipole_by_coordination[ion_type]
            if _mem:
                print(f"[cache] Using in-memory results for {ion_type} "
                      f"(pass force_rerun=True to recompute).")
                if coordination_states is not None:
                    return {cn: arr for cn, arr in _mem.items()
                            if cn in coordination_states}
                return _mem

        # --- try loading from disk ---
        if not force_rerun:
            print(f"[cache] Looking for: {_cache_abs}")
            if os.path.exists(_cache_abs):
                loaded = self.load_dipole_cache(_cache_abs)
                if loaded is not None:
                    if coordination_states is not None:
                        loaded = {cn: arr for cn, arr in loaded.items()
                                  if cn in coordination_states}
                    return loaded
                print("[cache] File found but failed to load — recomputing.")
            else:
                print(f"[cache] File not found — will compute and save to {_cache_abs}")

        # --- single ion type ---
        if step is None:
            step = self.default_step

        # Resolve ions, category, and radius
        if ion_type in ['cation', 'anion']:
            ions = self.cations if ion_type == 'cation' else self.anions
            ion_category = ion_type
            try:
                r0 = (self.solute_ci.radii['water'] if ion_type == 'cation'
                      else self.solute_ai.radii['water']) if radius is None else radius
            except (AttributeError, NameError):
                print(f"Coordination radius not available for {ion_type}")
                return None
        else:
            cation_types = self._get_unique_ion_types(self.cations)
            anion_types  = self._get_unique_ion_types(self.anions)
            if ion_type in cation_types:
                ions, ion_category = cation_types[ion_type], 'cation'
                if hasattr(self, 'solutes_ci') and ion_type in self.solutes_ci:
                    r0 = self.solutes_ci[ion_type].radii['water'] if radius is None else radius
                else:
                    print(f"No coordination radius available for {ion_type}")
                    return None
            elif ion_type in anion_types:
                ions, ion_category = anion_types[ion_type], 'anion'
                if hasattr(self, 'solutes_ai') and ion_type in self.solutes_ai:
                    r0 = self.solutes_ai[ion_type].radii['water'] if radius is None else radius
                else:
                    print(f"No coordination radius available for {ion_type}")
                    return None
            else:
                print(f"Ion type '{ion_type}' not found")
                return None

        print(f"Computing dipole distribution by coordination for {ion_type} ({ion_category})")
        print(f"Coordination radius: {r0:.2f} Å  |  step: {step}")

        frame_indices = (self._debug_frame_indices[::step]
                         if (hasattr(self, '_debug_frame_indices') and
                             self._debug_frame_indices is not None)
                         else range(0, len(self.universe.trajectory), step))

        angles_by_cn = {}

        for frame_idx in tqdm(frame_indices, desc=f"Analyzing {ion_type} dipoles by CN"):
            if (hasattr(self, '_debug_frame_indices') and
                    self._debug_frame_indices is not None):
                self.universe.trajectory[self._debug_frame_indices[frame_idx]]
            else:
                self.universe.trajectory[frame_idx]

            for ion in ions:
                shell  = self.universe.select_atoms(f'(sphzone {r0} index {ion.index})')
                waters = shell & self.waters
                cn     = len(waters)
                if cn == 0:
                    continue
                if cn not in angles_by_cn:
                    angles_by_cn[cn] = []

                box = self.universe.trajectory.ts.dimensions
                for water_O in waters:
                    hydrogens = water_O.residue.atoms.select_atoms(
                        'name HW* or name H or name H1 or name H2')
                    if len(hydrogens) < 2:
                        continue
                    H1_pos, H2_pos, O_pos = (hydrogens[0].position,
                                              hydrogens[1].position,
                                              water_O.position)
                    dipole    = (H1_pos + H2_pos) / 2.0 - O_pos
                    dipole   /= np.linalg.norm(dipole)
                    # Apply minimum-image convention to avoid spurious ~180° angles
                    # for waters that cross a periodic boundary relative to the ion.
                    raw_vec   = O_pos - ion.position
                    ion_O_vec = mda.lib.distances.minimize_vectors(
                        raw_vec[np.newaxis, :], box)[0]
                    ion_O_vec /= np.linalg.norm(ion_O_vec)
                    cos_angle = np.clip(np.dot(dipole, ion_O_vec), -1.0, 1.0)
                    angles_by_cn[cn].append(np.arccos(cos_angle) * 180.0 / np.pi)

        results = {cn: np.array(lst)
                   for cn, lst in angles_by_cn.items()
                   if coordination_states is None or cn in coordination_states}

        if not results:
            print("No dipole data collected for the specified coordination states")
            return None

        # Print summary table
        print(f"\n{'='*70}")
        print(f"WATER DIPOLE BY COORDINATION — {ion_type.upper()}   (category: {ion_category})")
        print(f"{'='*70}")
        print(f"{'CN':<4} {'N samples':<12} {'Mean°':<9} {'Std°':<9} {'Median°'}")
        print("-" * 70)
        for cn in sorted(results):
            a = results[cn]
            print(f"{cn:<4} {len(a):<12} {a.mean():.1f}     {a.std():.1f}     {np.median(a):.1f}")
        print(f"{'='*70}\n")

        # Cache on instance
        if not hasattr(self, 'dipole_by_coordination'):
            self.dipole_by_coordination = {}
        self.dipole_by_coordination[ion_type] = results

        # --- optionally persist to disk ---
        if save_cache:
            self.save_dipole_cache(ion_type=ion_type, filename=_cache_abs)

        return results

    # ------------------------------------------------------------------
    def save_dipole_cache(self, ion_type=None, filename=None):
        '''
        Save dipole-by-coordination results to a pickle file.

        Parameters
        ----------
        ion_type : str or None
            Save only this ion type.  If None, saves all ions stored in
            ``self.dipole_by_coordination``.
        filename : str or None
            Output path.  If None, auto-generated as
            ``'dipole_by_coordination_<ion_type>.pkl'`` (per-ion) or
            ``'dipole_by_coordination_all.pkl'`` (all ions).

        Returns
        -------
        success : bool
        '''
        if not hasattr(self, 'dipole_by_coordination') or not self.dipole_by_coordination:
            print("No dipole-by-coordination data to save. Run compute_water_dipole_by_coordination first.")
            return False

        if ion_type is not None:
            if ion_type not in self.dipole_by_coordination:
                print(f"No cached results for '{ion_type}'.")
                return False
            data_to_save = {ion_type: self.dipole_by_coordination[ion_type]}
            if filename is None:
                filename = f'dipole_by_coordination_{ion_type}.pkl'
        else:
            data_to_save = dict(self.dipole_by_coordination)
            if filename is None:
                filename = 'dipole_by_coordination_all.pkl'

        try:
            cache_payload = {
                'dipole_by_coordination': data_to_save,
                'metadata': {
                    'ion_types': list(data_to_save.keys()),
                    'cn_per_ion': {it: sorted(v.keys()) for it, v in data_to_save.items()},
                    'n_samples': {it: {cn: len(v) for cn, v in cns.items()}
                                  for it, cns in data_to_save.items()},
                }
            }
            with open(filename, 'wb') as f:
                pickle.dump(cache_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            ions_saved = list(data_to_save.keys())
            print(f"Dipole cache saved to {filename}")
            print(f"  Ion types: {ions_saved}")
            for it in ions_saved:
                cns = sorted(data_to_save[it].keys())
                print(f"  {it}: CNs {cns}")
            return True

        except Exception as e:
            print(f"Error saving dipole cache: {e}")
            return False

    # ------------------------------------------------------------------
    def load_dipole_cache(self, filename):
        '''
        Load dipole-by-coordination results from a pickle file.

        The loaded data is merged into ``self.dipole_by_coordination`` and the
        results dict for the first (or only) ion type is returned, so you can
        assign it directly::

            results_na = eq_opt.load_dipole_cache('dipole_by_coordination_Na.pkl')

        If the file contains multiple ion types, the full dict is returned and
        is also stored on ``self.dipole_by_coordination``.

        Parameters
        ----------
        filename : str
            Path to the cache file.

        Returns
        -------
        results : dict or None
            ``{cn: np.ndarray}`` for a single-ion file, or
            ``{ion_type: {cn: np.ndarray}}`` for a multi-ion file.
            Returns None on failure.
        '''
        if not os.path.exists(filename):
            print(f"Cache file not found: {filename}")
            return None

        if os.path.getsize(filename) == 0:
            print(f"Cache file is empty: {filename}")
            return None

        try:
            with open(filename, 'rb') as f:
                payload = pickle.load(f)

            if not isinstance(payload, dict) or 'dipole_by_coordination' not in payload:
                print(f"Unrecognised cache format in {filename}")
                return None

            data = payload['dipole_by_coordination']
            meta = payload.get('metadata', {})

            # Validate: every value must be {int: np.ndarray}
            for it, cns in data.items():
                if not isinstance(cns, dict):
                    print(f"Invalid data structure for ion type '{it}' in {filename}")
                    return None
                for cn, arr in cns.items():
                    if not isinstance(arr, np.ndarray):
                        print(f"Expected np.ndarray for {it} CN={cn}, got {type(arr)}")
                        return None

            # Merge into instance cache
            if not hasattr(self, 'dipole_by_coordination'):
                self.dipole_by_coordination = {}
            self.dipole_by_coordination.update(data)

            ions_loaded = list(data.keys())
            print(f"Dipole cache loaded from {filename}")
            print(f"  Ion types: {ions_loaded}")
            for it in ions_loaded:
                cns = sorted(data[it].keys())
                n   = {cn: len(data[it][cn]) for cn in cns}
                print(f"  {it}: CNs {cns}  |  samples {n}")

            # Return convenient value
            if len(ions_loaded) == 1:
                return data[ions_loaded[0]]
            return data

        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            try:
                os.remove(filename)
                print(f"Removed corrupted file {filename}")
            except Exception:
                pass
            return None
        except Exception as e:
            print(f"Error loading dipole cache from {filename}: {e}")
            return None

    # ------------------------------------------------------------------
    def plot_water_dipole_by_coordination(self, ion_type, results=None,
                                          coordination_states=None,
                                          bins=50, save_plots=True,
                                          figsize_per_panel=(5, 4),
                                          figsize_overlay=(10, 6),
                                          figsize_combined=(8, 6),
                                          plot_combined=False,
                                          dpi=300, show_title=True,
                                          title_font_size=11,
                                          title_font_weight='bold',
                                          suptitle_font_size=14,
                                          output_filename=None,
                                          label_font_size=11,
                                          label_font_weight='normal',
                                          tick_font_size=9,
                                          tick_font_weight='normal',
                                          legend_loc='best',
                                          legend_font_size=9,
                                          legend_bbox_to_anchor=None,
                                          show_combined_in_overlay=False,
                                          combined_color=None,
                                          combined_alpha=0.85,
                                          combined_smooth=True,
                                          combined_linewidth=2.5):
        '''
        Plot water dipole angle distributions by coordination number.

        This is the **plotting-only** step — no trajectory analysis is done here.
        Results can be supplied explicitly or read from the cache stored by
        ``compute_water_dipole_by_coordination``.

        Parameters
        ----------
        ion_type : str
            Ion type label (used for titles and filenames).
        results : dict or None
            ``{cn: np.ndarray}`` as returned by ``compute_water_dipole_by_coordination``.
            If None, the cached value from ``self.dipole_by_coordination[ion_type]``
            is used automatically.
        coordination_states : list of int or None
            Subset of coordination numbers to plot (e.g. ``[4, 6, 7, 8, 9]`` to
            exclude CN=5).  If None *all* keys in ``results`` are plotted.
        bins : int
            Number of histogram bins (default=50).
        save_plots : bool
            Whether to save figures to disk (default=True).
        figsize_per_panel : tuple
            (width, height) per subplot panel in inches (default=(5,4)).
        figsize_overlay : tuple
            (width, height) for the overlay figure in inches (default=(10,6)).
        dpi : int
            Figure resolution (default=300).
        show_title : bool
            Whether to display panel and suptitles (default=True).
        title_font_size : int
            Font size of individual panel titles (default=11).
        title_font_weight : str
            Font weight of individual panel titles (default=``'bold'``).
        suptitle_font_size : int
            Font size of the overall suptitle (default=14).
        output_filename : str or None
            Base filename for saved figures.  If None, auto-generated from
            ``ion_type`` and ``bins``.
        label_font_size : int
            Font size for axis labels (default=11).
        label_font_weight : str
            Font weight for axis labels (default=``'normal'``).
        tick_font_size : int
            Font size for tick labels (default=9).
        tick_font_weight : str
            Font weight for tick labels (default=``'normal'``).
        legend_loc : str or tuple
            Matplotlib legend location (default=``'best'``).
        legend_font_size : int
            Font size for legend text (default=9).
        figsize_combined : tuple
            (width, height) for the combined figure in inches (default=(8,6)).
        plot_combined : bool
            If True, also produce a single histogram of *all* angles pooled
            across every coordination number (no CN grouping), with mean and
            median lines.  Saved as ``<base>_combined.png`` when
            ``save_plots=True`` (default=False).
        show_combined_in_overlay : bool
            If True, add a thick dashed line showing the pooled
            (all-CN) distribution on top of the existing overlay plot,
            labelled "All CNs".  Default=False.
        combined_color : str or None
            Colour of the "All CNs" pooled line in the overlay plot.
            Accepts any Matplotlib colour string (e.g. ``'#8FD6ED'``,
            ``'steelblue'``).  If None, defaults to ``'black'``.
        combined_alpha : float
            Transparency of the "All CNs" KDE line (0=invisible, 1=opaque).
            Default=0.85.
        combined_smooth : bool
            If True (default), the pooled "All CNs" trace is drawn as a KDE
            smooth curve.  If False, a stepped histogram outline is drawn
            instead.
        combined_linewidth : float
            Line width of the "All CNs" trace (default=2.5).
        legend_bbox_to_anchor : tuple or None
            ``bbox_to_anchor`` passed to ``ax.legend()`` (e.g. ``(0.45, 0.85)``),
            default=None.
        '''
        # Resolve results from cache if not provided
        if results is None:
            if (hasattr(self, 'dipole_by_coordination') and
                    ion_type in self.dipole_by_coordination):
                results = self.dipole_by_coordination[ion_type]
            else:
                raise ValueError(
                    f"No cached results for '{ion_type}'.  "
                    f"Run compute_water_dipole_by_coordination('{ion_type}') first, "
                    f"or pass results explicitly.")

        # Optionally restrict which coordination numbers are plotted
        if coordination_states is not None:
            results = {cn: v for cn, v in results.items() if cn in coordination_states}
            if not results:
                raise ValueError(
                    f"None of the requested coordination_states {coordination_states} "
                    f"are present in results (available: {list(results.keys())}).")

        # Determine ion category for titles
        cation_types = self._get_unique_ion_types(self.cations)
        ion_category = 'cation' if ion_type in cation_types else 'anion'

        self._plot_dipole_by_coordination(
            ion_type, ion_category, results, bins,
            figsize_per_panel=figsize_per_panel,
            figsize_overlay=figsize_overlay,
            dpi=dpi, show_title=show_title,
            title_font_size=title_font_size,
            title_font_weight=title_font_weight,
            suptitle_font_size=suptitle_font_size,
            output_filename=output_filename,
            label_font_size=label_font_size,
            label_font_weight=label_font_weight,
            tick_font_size=tick_font_size,
            tick_font_weight=tick_font_weight,
            legend_loc=legend_loc,
            legend_font_size=legend_font_size,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
            show_combined_in_overlay=show_combined_in_overlay,
            combined_color=combined_color,
            combined_alpha=combined_alpha,
            combined_smooth=combined_smooth,
            combined_linewidth=combined_linewidth,
        )

        # --- combined (all-CN pooled) plot ---
        if plot_combined:
            all_angles = np.concatenate([results[cn] for cn in sorted(results.keys())])
            n_total = len(all_angles)
            mean_angle   = all_angles.mean()
            median_angle = np.median(all_angles)

            fig, ax = plt.subplots(1, 1, figsize=figsize_combined)
            ax.hist(all_angles, bins=bins, alpha=0.85, density=True,
                    edgecolor='black', linewidth=0.5,
                    color=plt.cm.viridis(0.5))
            ax.axvline(mean_angle,   color='red',  linestyle='--', linewidth=2,
                       label=f'Mean: {mean_angle:.1f}°')
            ax.axvline(median_angle, color='blue', linestyle=':',  linewidth=2,
                       label=f'Median: {median_angle:.1f}°')

            ax.set_xlabel('Dipole Angle (degrees)',
                          fontsize=label_font_size, fontweight=label_font_weight)
            ax.set_ylabel('Probability Density',
                          fontsize=label_font_size, fontweight=label_font_weight)
            ax.tick_params(axis='both', labelsize=tick_font_size)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight(tick_font_weight)
            ax.set_xlim(0, 180)
            ax.grid(True, alpha=0.3)

            if show_title:
                ax.set_title(f'{ion_type.upper()} Water Dipole — All CNs Pooled '
                             f'(n={n_total})',
                             fontsize=suptitle_font_size, fontweight=title_font_weight)

            _legend_kw = dict(fontsize=legend_font_size, frameon=False, loc=legend_loc)
            if legend_bbox_to_anchor is not None:
                _legend_kw['bbox_to_anchor'] = legend_bbox_to_anchor
            ax.legend(**_legend_kw)

            plt.tight_layout()
            if save_plots:
                _base = output_filename if output_filename is not None \
                    else f'dipole_by_coordination_{ion_type}_bins{bins}'
                _fname = f'{_base}_combined.png'
                plt.savefig(_fname, dpi=dpi, bbox_inches='tight')
                print(f'Combined plot saved as: {_fname}')
            plt.show()

    # ------------------------------------------------------------------
    def water_dipole_distribution_by_coordination(self, ion_type='cation', radius=None, step=None,
                                                  coordination_states=None, bins=50, save_plots=True,
                                                  figsize_per_panel=(5, 4), figsize_overlay=(10, 6),
                                                  dpi=300, show_title=True,
                                                  title_font_size=11, title_font_weight='bold',
                                                  suptitle_font_size=14, output_filename=None,
                                                  label_font_size=11, label_font_weight='normal',
                                                  tick_font_size=9, tick_font_weight='normal',
                                                  legend_loc='best', legend_font_size=9,
                                                  legend_bbox_to_anchor=None):
        '''
        Convenience wrapper: compute + plot in one call.

        Calls ``compute_water_dipole_by_coordination`` followed by
        ``plot_water_dipole_by_coordination``.  Use the two separate methods
        when you want to tweak plot aesthetics without re-running the analysis.
        '''
        if isinstance(ion_type, list):
            all_results = {}
            for single_ion in ion_type:
                res = self.water_dipole_distribution_by_coordination(
                    ion_type=single_ion, radius=radius, step=step,
                    coordination_states=coordination_states, bins=bins,
                    save_plots=save_plots, figsize_per_panel=figsize_per_panel,
                    figsize_overlay=figsize_overlay, dpi=dpi, show_title=show_title,
                    title_font_size=title_font_size, title_font_weight=title_font_weight,
                    suptitle_font_size=suptitle_font_size, output_filename=output_filename,
                    label_font_size=label_font_size, label_font_weight=label_font_weight,
                    tick_font_size=tick_font_size, tick_font_weight=tick_font_weight,
                    legend_loc=legend_loc, legend_font_size=legend_font_size,
                    legend_bbox_to_anchor=legend_bbox_to_anchor,
                )
                if res is not None:
                    all_results[single_ion] = res
            return all_results

        results = self.compute_water_dipole_by_coordination(
            ion_type=ion_type, radius=radius, step=step,
            coordination_states=coordination_states,
        )
        if results is None:
            return None

        if save_plots:
            self.plot_water_dipole_by_coordination(
                ion_type=ion_type, results=results, bins=bins,
                save_plots=save_plots, figsize_per_panel=figsize_per_panel,
                figsize_overlay=figsize_overlay, dpi=dpi, show_title=show_title,
                title_font_size=title_font_size, title_font_weight=title_font_weight,
                suptitle_font_size=suptitle_font_size, output_filename=output_filename,
                label_font_size=label_font_size, label_font_weight=label_font_weight,
                tick_font_size=tick_font_size, tick_font_weight=tick_font_weight,
                legend_loc=legend_loc, legend_font_size=legend_font_size,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
            )

        return results



    def _plot_dipole_by_coordination(self, ion_type, ion_category, results, bins,
                                     figsize_per_panel=(5, 4), figsize_overlay=(10, 6),
                                     dpi=300, show_title=True,
                                     title_font_size=11, title_font_weight='bold',
                                     suptitle_font_size=14, output_filename=None,
                                     label_font_size=11, label_font_weight='normal',
                                     tick_font_size=9, tick_font_weight='normal',
                                     legend_loc='best', legend_font_size=9,
                                     legend_bbox_to_anchor=None,
                                     show_combined_in_overlay=False,
                                     combined_color=None,
                                     combined_alpha=0.85,
                                     combined_smooth=True,
                                     combined_linewidth=2.5):
        '''
        Plot water dipole distributions by coordination number
        UPDATED: Now uses filled histograms with transparency instead of outlines
        UPDATED: Darker fill colors and frameless legend
        '''
        
        coordination_numbers = sorted(results.keys())
        n_coords = len(coordination_numbers)
        
        # Create figure with subplots
        n_cols = min(3, n_coords)
        n_rows = (n_coords + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_panel[0]*n_cols, figsize_per_panel[1]*n_rows))
        
        if n_coords == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        # UPDATED: Use darker colors from viridis colormap
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_coords))  # Changed from (0, 1) to (0.2, 0.8) for darker colors
        
        # Plot each coordination number
        for i, cn in enumerate(coordination_numbers):
            ax = axes[i]
            angles = results[cn]
            
            # UPDATED: Filled histogram with higher alpha for darker appearance
            ax.hist(angles, bins=bins, alpha=0.85, color=colors[i],  # Changed alpha from 0.7 to 0.85
                    edgecolor='black', linewidth=0.5, density=True)
            
            # Add statistics lines
            mean_angle = angles.mean()
            median_angle = np.median(angles)
            
            ax.axvline(mean_angle, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean_angle:.1f}°')
            ax.axvline(median_angle, color='blue', linestyle=':', linewidth=2,
                    label=f'Median: {median_angle:.1f}°')
            
            ax.set_xlabel('Dipole Angle (degrees)', fontsize=label_font_size, fontweight=label_font_weight)
            ax.set_ylabel('Probability Density', fontsize=label_font_size, fontweight=label_font_weight)
            ax.tick_params(axis='both', labelsize=tick_font_size)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight(tick_font_weight)
            if show_title:
                ax.set_title(f'CN = {cn} (n={len(angles)})',
                             fontweight=title_font_weight, fontsize=title_font_size)
            
            # UPDATED: Frameless legend
            _legend_kw = dict(fontsize=legend_font_size, frameon=False, loc=legend_loc)
            if legend_bbox_to_anchor is not None:
                _legend_kw['bbox_to_anchor'] = legend_bbox_to_anchor
            ax.legend(**_legend_kw)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 180)
        
        # Hide unused subplots
        for i in range(n_coords, len(axes)):
            axes[i].set_visible(False)
        
        if show_title:
            plt.suptitle(f'{ion_type.upper()} Water Dipole Distribution by Coordination Number', 
                        fontsize=suptitle_font_size, fontweight='bold')
        plt.tight_layout()
        
        base = output_filename if output_filename is not None else f'dipole_by_coordination_{ion_type}_bins{bins}'
        filename = f'{base}.png'
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
        
        plt.show()
        
        # Also create overlay comparison plot with filled histograms
        fig, ax = plt.subplots(1, 1, figsize=figsize_overlay)
        
        for i, cn in enumerate(coordination_numbers):
            angles = results[cn]
            # UPDATED: Filled histogram with higher alpha for darker colors
            ax.hist(angles, bins=bins, alpha=0.65, color=colors[i],  # Changed alpha from 0.5 to 0.65
                    label=f'CN={cn} (n={len(angles)})', density=True,
                    edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Dipole Angle (degrees)', fontsize=label_font_size, fontweight=label_font_weight)
        ax.set_ylabel('Probability Density', fontsize=label_font_size, fontweight=label_font_weight)
        ax.tick_params(axis='both', labelsize=tick_font_size)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight(tick_font_weight)
        if show_title:
            ax.set_title(f'{ion_type.upper()} Water Dipole Distribution: All Coordination Numbers', 
                        fontweight='bold', fontsize=suptitle_font_size)
        
        # UPDATED: Frameless legend
        _legend_kw_ov = dict(frameon=False, loc=legend_loc, fontsize=legend_font_size)
        if legend_bbox_to_anchor is not None:
            _legend_kw_ov['bbox_to_anchor'] = legend_bbox_to_anchor
        ax.legend(**_legend_kw_ov)
        
        # Optionally add pooled (all-CN) KDE smooth curve on top of overlay
        if show_combined_in_overlay:
            all_angles = np.concatenate([results[cn] for cn in coordination_numbers])
            _combined_color = combined_color if combined_color is not None else 'black'
            if combined_smooth:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(all_angles)
                x_grid = np.linspace(0, 180, 500)
                ax.plot(x_grid, kde(x_grid), color=_combined_color,
                        linewidth=combined_linewidth, linestyle='--', alpha=combined_alpha,
                        label='All CNs')
            else:
                ax.hist(all_angles, bins=bins, density=True,
                        histtype='step', linewidth=combined_linewidth,
                        color=_combined_color, linestyle='--',
                        alpha=combined_alpha, label='All CNs')
            ax.legend(**{k: v for k, v in _legend_kw_ov.items() if k != 'handles'})

        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)
        
        plt.tight_layout()
        
        overlay_base = (output_filename + '_overlay') if output_filename is not None else f'dipole_by_coordination_{ion_type}_overlay_bins{bins}'
        filename = f'{overlay_base}.png'
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Overlay plot saved as: {filename}")
        
        plt.show()



    def compare_dipole_by_coordination_across_ions(self, ion_types=None, coordination_number=None, 
                                                save_plots=True, bins=50):
        '''
        Compare water dipole distributions across different ion types for the same coordination number.
        UPDATED: Added bins parameter and uses filled histograms
        
        Parameters
        ----------
        ion_types : list or None
            Ion types to compare. If None, uses all available.
        coordination_number : int or None
            Specific CN to compare. If None, compares the most common CN for each ion.
        save_plots : bool
            Whether to save plots
        bins : int
            Number of histogram bins (default=50)
        '''
        
        if not hasattr(self, 'dipole_by_coordination') or not self.dipole_by_coordination:
            print("No dipole-by-coordination data available.")
            print("Run water_dipole_distribution_by_coordination() first.")
            return None
        
        if ion_types is None:
            ion_types = list(self.dipole_by_coordination.keys())
        
        print(f"Comparing dipole distributions across: {ion_types}")
        print(f"Using {bins} bins")
        
        # Determine colors
        cation_types_in_system = set(self._get_unique_ion_types(self.cations).keys())
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        for ion_type in ion_types:
            if ion_type not in self.dipole_by_coordination:
                continue
            
            results = self.dipole_by_coordination[ion_type]
            
            if coordination_number is None:
                # Use most common CN for this ion
                cn_to_use = max(results.keys(), key=lambda cn: len(results[cn]))
            else:
                if coordination_number not in results:
                    print(f"CN={coordination_number} not found for {ion_type}")
                    continue
                cn_to_use = coordination_number
            
            angles = results[cn_to_use]
            
            # Determine color
            if ion_type in cation_types_in_system or ion_type == 'cation':
                color = 'blue'
            else:
                color = 'red'
            
            # UPDATED: Filled histogram with transparency
            ax.hist(angles, bins=bins, alpha=0.5, color=color,
                    label=f'{ion_type} CN={cn_to_use} (n={len(angles)})',
                    density=True, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Dipole Angle (degrees)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        
        if coordination_number:
            title = f'Water Dipole Comparison: CN={coordination_number}'
        else:
            title = 'Water Dipole Comparison: Most Common CN per Ion'
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'dipole_comparison_CN{coordination_number if coordination_number else "auto"}_bins{bins}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved as: {filename}")
        
        plt.show()


    # def _plot_dipole_by_coordination_enhanced(self, ion_type, ion_category, results, bins, 
    #                                         overlay_style='step'):
    #     '''
    #     Enhanced plotting with selectable overlay styles for better visibility
        
    #     Parameters
    #     ----------
    #     overlay_style : str
    #         Style for overlay plot: 'step' (outlines), 'filled' (low alpha), 'kde' (smooth curves)
    #     '''
        
    #     coordination_numbers = sorted(results.keys())
    #     n_coords = len(coordination_numbers)
        
    #     # Create figure with subplots (same as before)
    #     n_cols = min(3, n_coords)
    #     n_rows = (n_coords + n_cols - 1) // n_cols
        
    #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
    #     if n_coords == 1:
    #         axes = [axes]
    #     elif n_rows == 1:
    #         axes = axes if n_cols == 1 else list(axes)
    #     else:
    #         axes = axes.flatten()
        
    #     # Use qualitative color map for better distinction
    #     colors = plt.cm.Set1(np.linspace(0, 1, n_coords))
        
    #     # Individual plots (same as before)
    #     for i, cn in enumerate(coordination_numbers):
    #         ax = axes[i]
    #         angles = results[cn]
            
    #         ax.hist(angles, bins=bins, alpha=0.7, color=colors[i], 
    #                 edgecolor='black', linewidth=0.5, density=True)
            
    #         mean_angle = angles.mean()
    #         median_angle = np.median(angles)
            
    #         ax.axvline(mean_angle, color='red', linestyle='--', linewidth=2, 
    #                 label=f'Mean: {mean_angle:.1f}°')
    #         ax.axvline(median_angle, color='blue', linestyle=':', linewidth=2,
    #                 label=f'Median: {median_angle:.1f}°')
            
    #         ax.set_xlabel('Dipole Angle (degrees)', fontsize=10)
    #         ax.set_ylabel('Probability Density', fontsize=10)
    #         ax.set_title(f'CN = {cn} (n={len(angles)})', fontweight='bold', fontsize=11)
    #         ax.legend(fontsize=9)
    #         ax.grid(True, alpha=0.3)
    #         ax.set_xlim(0, 180)
        
    #     for i in range(n_coords, len(axes)):
    #         axes[i].set_visible(False)
        
    #     plt.suptitle(f'{ion_type.upper()} Water Dipole Distribution by Coordination Number', 
    #                 fontsize=14, fontweight='bold')
    #     plt.tight_layout()
        
    #     filename = f'dipole_by_coordination_{ion_type}_bins{bins}.png'
    #     plt.savefig(filename, dpi=300, bbox_inches='tight')
    #     print(f"Plot saved as: {filename}")
    #     plt.show()
        
    #     # OVERLAY PLOT with selectable style
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
    #     if overlay_style == 'step':
    #         # Outline histograms only - best for many overlapping distributions
    #         for i, cn in enumerate(coordination_numbers):
    #             angles = results[cn]
    #             ax.hist(angles, bins=bins, alpha=0, color=colors[i],
    #                     label=f'CN={cn} (n={len(angles)})', density=True,
    #                     histtype='step', linewidth=2.5)
        
    #     elif overlay_style == 'filled':
    #         # Filled histograms with low alpha - good for 2-3 distributions
    #         for i, cn in enumerate(coordination_numbers):
    #             angles = results[cn]
    #             ax.hist(angles, bins=bins, alpha=0.25, color=colors[i],
    #                     label=f'CN={cn} (n={len(angles)})', density=True,
    #                     edgecolor=colors[i], linewidth=1.5)
        
    #     elif overlay_style == 'kde':
    #         # Smooth KDE curves - best for visual appeal and clarity
    #         from scipy import stats
            
    #         x_range = np.linspace(0, 180, 500)
    #         for i, cn in enumerate(coordination_numbers):
    #             angles = results[cn]
    #             kde = stats.gaussian_kde(angles)
    #             density = kde(x_range)
    #             ax.plot(x_range, density, color=colors[i], linewidth=2.5,
    #                     label=f'CN={cn} (n={len(angles)})')
    #             ax.fill_between(x_range, density, alpha=0.15, color=colors[i])
        
    #     ax.set_xlabel('Dipole Angle (degrees)', fontsize=12)
    #     ax.set_ylabel('Probability Density', fontsize=12)
    #     ax.set_title(f'{ion_type.upper()} Water Dipole Distribution: All Coordination Numbers', 
    #                 fontweight='bold', fontsize=14)
    #     ax.legend(fontsize=10, framealpha=0.9)
    #     ax.grid(True, alpha=0.3)
    #     ax.set_xlim(0, 180)
        
    #     plt.tight_layout()
        
    #     filename = f'dipole_by_coordination_{ion_type}_overlay_{overlay_style}_bins{bins}.png'
    #     plt.savefig(filename, dpi=300, bbox_inches='tight')
    #     print(f"Overlay plot saved as: {filename}")
    #     plt.show()




    def save_dipole_by_coordination_to_file(self, filename='dipole_by_coordination_cache.pkl'):
        '''
        Save water dipole distribution by coordination data to file for persistence across sessions.
        
        Parameters
        ----------
        filename : str
            Output filename, default='dipole_by_coordination_cache.pkl'
        
        Returns
        -------
        success : bool
            True if save was successful
        '''
        
        if not hasattr(self, 'dipole_by_coordination') or not self.dipole_by_coordination:
            print("No dipole-by-coordination data to save")
            return False
        
        try:
            # Prepare dipole-by-coordination data for serialization
            dipole_coord_data = {}
            
            for ion_type, cn_data in self.dipole_by_coordination.items():
                dipole_coord_data[ion_type] = {}
                
                for cn, angles in cn_data.items():
                    # Store as list for JSON compatibility if needed
                    dipole_coord_data[ion_type][cn] = {
                        'angles': angles.copy(),
                        'mean': float(angles.mean()),
                        'std': float(angles.std()),
                        'median': float(np.median(angles)),
                        'n_samples': len(angles)
                    }
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(dipole_coord_data, f)
            
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            
            print(f"Dipole-by-coordination data saved to {filename}")
            print(f"  File size: {file_size_mb:.1f} MB")
            print(f"  Saved {len(dipole_coord_data)} ion types")
            print(f"  Ion types: {list(dipole_coord_data.keys())}")
            
            # Print summary
            print(f"\n  Dipole-by-coordination summary:")
            for ion_type, cn_data in dipole_coord_data.items():
                coordination_numbers = sorted(cn_data.keys())
                total_samples = sum(data['n_samples'] for data in cn_data.values())
                print(f"    {ion_type}: {len(coordination_numbers)} CNs, {total_samples} total angles")
            
            return True
            
        except Exception as e:
            print(f"Error saving dipole-by-coordination data: {e}")
            traceback.print_exc()
            return False


    def load_dipole_by_coordination_from_file(self, filename='dipole_by_coordination_cache.pkl'):
        '''
        Load water dipole distribution by coordination data from file with error handling.
        
        Parameters
        ----------
        filename : str
            Input filename, default='dipole_by_coordination_cache.pkl'
        
        Returns
        -------
        success : bool
            True if load was successful
        '''
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        try:
            # Check file size first
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print(f"Cache file {filename} is empty")
                return False
            
            file_size_mb = file_size / (1024 * 1024)
            print(f"Loading dipole-by-coordination data from {filename} ({file_size_mb:.1f} MB)...")
            
            # Load data
            with open(filename, 'rb') as f:
                dipole_coord_data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(dipole_coord_data, dict):
                print(f"Invalid dipole-by-coordination cache format")
                return False
            
            # Reconstruct the dipole_by_coordination structure
            self.dipole_by_coordination = {}
            
            for ion_type, cn_data in dipole_coord_data.items():
                self.dipole_by_coordination[ion_type] = {}
                
                for cn, data in cn_data.items():
                    # Reconstruct numpy array from stored data
                    self.dipole_by_coordination[ion_type][cn] = data['angles']
            
            # Print summary
            successful_types = list(self.dipole_by_coordination.keys())
            
            print(f"Dipole-by-coordination data loaded from {filename}")
            print(f"  Loaded {len(successful_types)} ion types successfully")
            if successful_types:
                print(f"  Ion types: {', '.join(successful_types)}")
            
            # Print detailed summary
            print(f"\n  Dipole-by-coordination summary:")
            for ion_type, cn_data in self.dipole_by_coordination.items():
                coordination_numbers = sorted(cn_data.keys())
                total_samples = sum(len(angles) for angles in cn_data.values())
                print(f"    {ion_type}: {len(coordination_numbers)} CNs, {total_samples} total angles")
                
                # Show CN range
                if coordination_numbers:
                    print(f"      CN range: {min(coordination_numbers)} - {max(coordination_numbers)}")
            
            return True
            
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            print("Removing corrupted cache file...")
            try:
                os.remove(filename)
                print(f"Corrupted file {filename} removed")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading dipole-by-coordination data from {filename}: {e}")
            traceback.print_exc()
            return False


    def water_dipole_distribution_by_coordination_with_cache(self, cache_filename='dipole_by_coordination_cache.pkl', 
                                                            force_recalc=False, **kwargs):
        '''
        Calculate water dipole distribution by coordination with automatic caching.
        
        Parameters
        ----------
        cache_filename : str
            Cache file name, default='dipole_by_coordination_cache.pkl'
        force_recalc : bool
            Force recalculation even if cache exists, default=False
        **kwargs : dict
            Arguments to pass to water_dipole_distribution_by_coordination()
        
        Returns
        -------
        results : dict
            Dictionary of dipole-by-coordination results
        '''
        
        # Get ion_type from kwargs if specified
        ion_type = kwargs.get('ion_type', None)
        
        # Try to load from cache first
        if not force_recalc and os.path.exists(cache_filename):
            print("Attempting to load dipole-by-coordination data from cache...")
            if self.load_dipole_by_coordination_from_file(cache_filename):
                # Check if we have the requested ion type(s)
                if ion_type is not None:
                    if isinstance(ion_type, list):
                        missing_types = [ion for ion in ion_type if ion not in self.dipole_by_coordination]
                        if not missing_types:
                            print(f"✓ All requested ion types found in cache")
                            return self.dipole_by_coordination
                        else:
                            print(f"✗ Missing ion types in cache: {missing_types}")
                            print("  Will recalculate missing types...")
                    elif ion_type in self.dipole_by_coordination:
                        print(f"✓ Ion type '{ion_type}' found in cache")
                        return self.dipole_by_coordination[ion_type]
                    else:
                        print(f"✗ Ion type '{ion_type}' not in cache, will recalculate...")
                else:
                    print("✓ Successfully loaded dipole-by-coordination data from cache")
                    return self.dipole_by_coordination
            else:
                print("✗ Cache loading failed, will recalculate")
        
        # Calculate dipole distribution by coordination
        print("Calculating dipole distribution by coordination...")
        results = self.water_dipole_distribution_by_coordination(**kwargs)
        
        # Save to cache
        if results:
            print("Saving dipole-by-coordination data to cache...")
            if self.save_dipole_by_coordination_to_file(cache_filename):
                print("✓ Dipole-by-coordination data cached successfully")
            else:
                print("✗ Cache saving failed, but results are available in memory")
        
        return results




    def analyze_coordination_pairing_trends(self, ion_types=None):
        '''
        Analyze trends in ion pairing probabilities as a function of coordination number.
        
        Parameters
        ----------
        ion_types : list or None
            Ion types to analyze. If None, analyzes all available types.
        
        Returns
        -------
        trends : dict
            Dictionary with trend analysis results for each ion type
        '''
        
        if not hasattr(self, 'coordination_pairing_analysis'):
            print("No coordination-pairing analysis data available.")
            print("Run compute_ion_pairing_probabilities_by_coordination() first.")
            return None
        
        if ion_types is None:
            ion_types = list(self.coordination_pairing_analysis.keys())
        
        trends = {}
        
        print(f"\n{'='*80}")
        print("COORDINATION-PAIRING TREND ANALYSIS")
        print(f"{'='*80}")
        
        for ion_type in ion_types:
            if ion_type not in self.coordination_pairing_analysis:
                print(f"No data for {ion_type}")
                continue
            
            results = self.coordination_pairing_analysis[ion_type]
            
            # Extract coordination numbers and pairing probabilities
            coord_numbers = sorted(results.keys())
            
            # Initialize arrays for each pairing type
            cip_probs = []
            sip_probs = []
            dsip_probs = []
            fi_probs = []
            
            for cn in coord_numbers:
                probs = results[cn]['probabilities']
                cip_probs.append(probs.get('CIP', 0))
                sip_probs.append(probs.get('SIP', 0))
                dsip_probs.append(probs.get('DSIP', 0))
                fi_probs.append(probs.get('FI', 0))
            
            # Calculate trends using linear regression
            from scipy import stats
            
            # CIP trend
            if len(coord_numbers) > 1:
                cip_slope, cip_intercept, cip_r, _, _ = stats.linregress(coord_numbers, cip_probs)
                sip_slope, sip_intercept, sip_r, _, _ = stats.linregress(coord_numbers, sip_probs)
                fi_slope, fi_intercept, fi_r, _, _ = stats.linregress(coord_numbers, fi_probs)
            else:
                cip_slope = sip_slope = fi_slope = 0
                cip_r = sip_r = fi_r = 0
            
            # Find coordination number with maximum/minimum pairing for each type
            max_cip_cn = coord_numbers[np.argmax(cip_probs)]
            max_sip_cn = coord_numbers[np.argmax(sip_probs)]
            max_fi_cn = coord_numbers[np.argmax(fi_probs)]
            
            # Calculate coordination number with most observations
            total_obs = [results[cn]['total_observations'] for cn in coord_numbers]
            most_common_cn = coord_numbers[np.argmax(total_obs)]
            
            trends[ion_type] = {
                'coordination_numbers': coord_numbers,
                'cip_probabilities': cip_probs,
                'sip_probabilities': sip_probs,
                'dsip_probabilities': dsip_probs,
                'fi_probabilities': fi_probs,
                'cip_trend': {'slope': cip_slope, 'r_value': cip_r},
                'sip_trend': {'slope': sip_slope, 'r_value': sip_r},
                'fi_trend': {'slope': fi_slope, 'r_value': fi_r},
                'max_cip_cn': max_cip_cn,
                'max_sip_cn': max_sip_cn,
                'max_fi_cn': max_fi_cn,
                'most_common_cn': most_common_cn,
                'total_observations': total_obs
            }
            
            # Print analysis
            print(f"\n{ion_type.upper()}:")
            print("-" * 60)
            print(f"Coordination number range: {min(coord_numbers)} - {max(coord_numbers)}")
            print(f"Most common coordination: CN={most_common_cn} ({max(total_obs)} observations)")
            print(f"\nPairing trends with coordination:")
            
            # CIP trend
            if abs(cip_slope) > 0.01:
                direction = "increases" if cip_slope > 0 else "decreases"
                print(f"  CIP {direction} with CN (slope={cip_slope:.3f}, R²={cip_r**2:.3f})")
                print(f"    Maximum at CN={max_cip_cn} ({max(cip_probs):.1%})")
            else:
                print(f"  CIP shows no clear trend with CN")
            
            # SIP trend
            if abs(sip_slope) > 0.01:
                direction = "increases" if sip_slope > 0 else "decreases"
                print(f"  SIP {direction} with CN (slope={sip_slope:.3f}, R²={sip_r**2:.3f})")
                print(f"    Maximum at CN={max_sip_cn} ({max(sip_probs):.1%})")
            else:
                print(f"  SIP shows no clear trend with CN")
            
            # FI trend
            if abs(fi_slope) > 0.01:
                direction = "increases" if fi_slope > 0 else "decreases"
                print(f"  FI {direction} with CN (slope={fi_slope:.3f}, R²={fi_r**2:.3f})")
                print(f"    Maximum at CN={max_fi_cn} ({max(fi_probs):.1%})")
            else:
                print(f"  FI shows no clear trend with CN")
            
            # Physical interpretation
            print(f"\nPhysical interpretation:")
            if cip_slope < -0.05:
                print("  ⚠ Higher coordination → LESS contact pairing")
                print("    → Water coordination competes with direct ion contact")
            elif cip_slope > 0.05:
                print("  ⚠ Higher coordination → MORE contact pairing")
                print("    → Unusual behavior - check system")
            
            if fi_slope > 0.05:
                print("  ✓ Higher coordination → MORE free ions")
                print("    → Well-solvated ions are less likely to pair")
            elif fi_slope < -0.05:
                print("  ⚠ Higher coordination → FEWER free ions")
                print("    → Highly coordinated ions still pair significantly")
        
        print(f"\n{'='*80}")
        
        return trends

    def compare_coordination_pairing_across_types(self, save_plots=True, figsize=(16, 10)):
        '''
        Create comprehensive comparison of coordination-pairing relationships across ion types.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save plots
        figsize : tuple
            Figure size
        '''
        
        if not hasattr(self, 'coordination_pairing_analysis'):
            print("No coordination-pairing analysis data available.")
            return None
        
        # Get trend analysis
        trends = self.analyze_coordination_pairing_trends()
        
        if not trends:
            return None
        
        # Create comprehensive comparison figure
        fig = plt.figure(figsize=figsize)
        
        # Determine ion categories
        cation_types_in_system = set()
        anion_types_in_system = set()
        
        if hasattr(self, '_get_unique_ion_types'):
            cation_types_in_system = set(self._get_unique_ion_types(self.cations).keys())
            anion_types_in_system = set(self._get_unique_ion_types(self.anions).keys())
        
        # Separate data by category
        cation_trends = {k: v for k, v in trends.items() if k in cation_types_in_system}
        anion_trends = {k: v for k, v in trends.items() if k in anion_types_in_system}
        
        # Create 2x3 subplot layout
        # Row 1: CIP and FI trends
        # Row 2: Dominant pairing by CN and Population distribution
        
        # Plot 1: CIP probability trends
        ax1 = plt.subplot(2, 3, 1)
        for ion_type, data in cation_trends.items():
            ax1.plot(data['coordination_numbers'], data['cip_probabilities'], 
                    'o-', label=f"{ion_type} (cat)", linewidth=2, markersize=6)
        for ion_type, data in anion_trends.items():
            ax1.plot(data['coordination_numbers'], data['cip_probabilities'], 
                    's--', label=f"{ion_type} (an)", linewidth=2, markersize=6)
        
        ax1.set_xlabel('Coordination Number')
        ax1.set_ylabel('CIP Probability')
        ax1.set_title('Contact Ion Pair Probability vs CN', fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: FI probability trends
        ax2 = plt.subplot(2, 3, 2)
        for ion_type, data in cation_trends.items():
            ax2.plot(data['coordination_numbers'], data['fi_probabilities'], 
                    'o-', label=f"{ion_type} (cat)", linewidth=2, markersize=6)
        for ion_type, data in anion_trends.items():
            ax2.plot(data['coordination_numbers'], data['fi_probabilities'], 
                    's--', label=f"{ion_type} (an)", linewidth=2, markersize=6)
        
        ax2.set_xlabel('Coordination Number')
        ax2.set_ylabel('FI Probability')
        ax2.set_title('Free Ion Probability vs CN', fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: SIP probability trends
        ax3 = plt.subplot(2, 3, 3)
        for ion_type, data in cation_trends.items():
            ax3.plot(data['coordination_numbers'], data['sip_probabilities'], 
                    'o-', label=f"{ion_type} (cat)", linewidth=2, markersize=6)
        for ion_type, data in anion_trends.items():
            ax3.plot(data['coordination_numbers'], data['sip_probabilities'], 
                    's--', label=f"{ion_type} (an)", linewidth=2, markersize=6)
        
        ax3.set_xlabel('Coordination Number')
        ax3.set_ylabel('SIP Probability')
        ax3.set_title('Solvent-Separated Pair Probability vs CN', fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Dominant pairing type by coordination number (heatmap-style)
        ax4 = plt.subplot(2, 3, 4)
        
        # Create matrix showing dominant pairing type for each ion at each CN
        all_ion_types = list(trends.keys())
        all_cns = sorted(set([cn for data in trends.values() for cn in data['coordination_numbers']]))
        
        # Encode pairing types as numbers for visualization
        pairing_encoding = {'CIP': 1, 'SIP': 2, 'DSIP': 3, 'FI': 4}
        
        heatmap_data = np.zeros((len(all_ion_types), len(all_cns)))
        
        for i, ion_type in enumerate(all_ion_types):
            data = trends[ion_type]
            for j, cn in enumerate(all_cns):
                if cn in data['coordination_numbers']:
                    cn_idx = data['coordination_numbers'].index(cn)
                    # Find dominant pairing type
                    probs = {
                        'CIP': data['cip_probabilities'][cn_idx],
                        'SIP': data['sip_probabilities'][cn_idx],
                        'DSIP': data['dsip_probabilities'][cn_idx],
                        'FI': data['fi_probabilities'][cn_idx]
                    }
                    dominant = max(probs, key=probs.get)
                    heatmap_data[i, j] = pairing_encoding[dominant]
                else:
                    heatmap_data[i, j] = 0  # No data
        
        im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=4)
        ax4.set_xticks(range(len(all_cns)))
        ax4.set_xticklabels(all_cns)
        ax4.set_yticks(range(len(all_ion_types)))
        ax4.set_yticklabels(all_ion_types)
        ax4.set_xlabel('Coordination Number')
        ax4.set_title('Dominant Pairing Type by CN', fontweight='bold')
        
        # Add colorbar with labels
        cbar = plt.colorbar(im, ax=ax4, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(['None', 'CIP', 'SIP', 'DSIP', 'FI'])
        
        # Plot 5: Coordination number population distribution
        ax5 = plt.subplot(2, 3, 5)
        
        x_positions = np.arange(len(all_ion_types))
        width = 0.15
        
        # Get most common CN for each ion type
        most_common_cns = [trends[ion_type]['most_common_cn'] for ion_type in all_ion_types]
        colors_by_category = ['steelblue' if ion_type in cation_types_in_system else 'crimson' 
                            for ion_type in all_ion_types]
        
        bars = ax5.bar(x_positions, most_common_cns, color=colors_by_category, alpha=0.7)
        
        # Add value labels
        for bar, cn in zip(bars, most_common_cns):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{cn}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax5.set_xticks(x_positions)
        ax5.set_xticklabels(all_ion_types, rotation=45, ha='right')
        ax5.set_ylabel('Most Common CN')
        ax5.set_title('Preferred Coordination Numbers', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Trend slopes comparison (bar chart)
        ax6 = plt.subplot(2, 3, 6)
        
        cip_slopes = [trends[ion_type]['cip_trend']['slope'] for ion_type in all_ion_types]
        fi_slopes = [trends[ion_type]['fi_trend']['slope'] for ion_type in all_ion_types]
        
        x = np.arange(len(all_ion_types))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, cip_slopes, width, label='CIP', color='lightcoral', alpha=0.8)
        bars2 = ax6.bar(x + width/2, fi_slopes, width, label='FI', color='lightgreen', alpha=0.8)
        
        ax6.axhline(0, color='black', linestyle='--', linewidth=1)
        ax6.set_xticks(x)
        ax6.set_xticklabels(all_ion_types, rotation=45, ha='right')
        ax6.set_ylabel('Trend Slope (prob/CN)')
        ax6.set_title('Pairing-CN Correlation Strength', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Coordination-Pairing Analysis: Multi-Ion Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            filename = 'coordination_pairing_comprehensive_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Comprehensive analysis plot saved as: {filename}")
        
        plt.show()
        
        return trends

    def export_coordination_pairing_to_csv(self, filename='coordination_pairing_data.csv'):
        '''
        Export coordination-pairing analysis results to CSV for external analysis.
        
        Parameters
        ----------
        filename : str
            Output CSV filename
        
        Returns
        -------
        success : bool
            True if export was successful
        '''
        
        if not hasattr(self, 'coordination_pairing_analysis'):
            print("No coordination-pairing analysis data available.")
            return False
        
        import csv
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['ion_type', 'coordination_number', 'total_observations', 
                            'CIP_prob', 'SIP_prob', 'DSIP_prob', 'FI_prob',
                            'CIP_count', 'SIP_count', 'DSIP_count', 'FI_count']
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for ion_type, results in self.coordination_pairing_analysis.items():
                    for cn, data in sorted(results.items()):
                        row = {
                            'ion_type': ion_type,
                            'coordination_number': cn,
                            'total_observations': data['total_observations'],
                            'CIP_prob': data['probabilities'].get('CIP', 0),
                            'SIP_prob': data['probabilities'].get('SIP', 0),
                            'DSIP_prob': data['probabilities'].get('DSIP', 0),
                            'FI_prob': data['probabilities'].get('FI', 0),
                            'CIP_count': data['counts'].get('CIP', 0),
                            'SIP_count': data['counts'].get('SIP', 0),
                            'DSIP_count': data['counts'].get('DSIP', 0),
                            'FI_count': data['counts'].get('FI', 0)
                        }
                        writer.writerow(row)
            
            print(f"Coordination-pairing data exported to {filename}")
            print(f"  Columns: {', '.join(fieldnames)}")
            
            return True
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False



    def _print_coordination_pairing_summary(self, ion_type, results):
        '''Print summary of coordination-pairing analysis'''
        
        print(f"\n{'='*70}")
        print(f"ION PAIRING PROBABILITIES BY COORDINATION STATE: {ion_type.upper()}")
        print(f"{'='*70}")
        
        # Create header
        pair_types = []
        if results:
            pair_types = list(next(iter(results.values()))['probabilities'].keys())
        
        header = f"{'CN':<4} {'Obs':<6}"
        for pair_type in pair_types:
            header += f" {pair_type:<8}"
        print(header)
        print("-" * len(header))
        
        # Print data for each coordination number
        for cn in sorted(results.keys()):
            data = results[cn]
            probs = data['probabilities']
            total_obs = data['total_observations']
            
            row = f"{cn:<4} {total_obs:<6}"
            for pair_type in pair_types:
                prob = probs.get(pair_type, 0.0)
                row += f" {prob:.3f}   "
            
            print(row)
        
        print(f"{'='*70}")
        print("CN = Coordination Number, Obs = Total Observations")
        print("Values show probability of each pairing type for that coordination number")

    def _plot_coordination_pairing_probabilities(self, ion_type, results, ion_category):
        '''Plot coordination-pairing probability analysis'''
        
        if not results:
            return
        
        # Extract data for plotting
        coordination_numbers = sorted(results.keys())
        pair_types = list(next(iter(results.values()))['probabilities'].keys())
        
        # Create probability matrix
        prob_matrix = np.zeros((len(coordination_numbers), len(pair_types)))
        observation_counts = []
        
        for i, cn in enumerate(coordination_numbers):
            observation_counts.append(results[cn]['total_observations'])
            for j, pair_type in enumerate(pair_types):
                prob_matrix[i, j] = results[cn]['probabilities'].get(pair_type, 0.0)
        
        # MODIFIED: Changed from 2x2 to 2x3 layout to add histogram
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot 1: Stacked bar chart of probabilities
        ax = axes[0, 0]
        
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'][:len(pair_types)]
        bottom = np.zeros(len(coordination_numbers))
        
        for j, pair_type in enumerate(pair_types):
            ax.bar(coordination_numbers, prob_matrix[:, j], bottom=bottom, 
                color=colors[j], label=pair_type, alpha=0.8)
            bottom += prob_matrix[:, j]
        
        ax.set_xlabel('Coordination Number')
        ax.set_ylabel('Probability')
        ax.set_title(f'{ion_type} Ion Pairing by Coordination State', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Individual probability curves
        ax = axes[0, 1]
        
        for j, pair_type in enumerate(pair_types):
            ax.plot(coordination_numbers, prob_matrix[:, j], 
                    marker='o', linewidth=2, label=pair_type, color=colors[j])
        
        ax.set_xlabel('Coordination Number')
        ax.set_ylabel('Probability')
        ax.set_title(f'{ion_type} Pairing Probability Trends', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 3: Observation counts (absolute numbers)
        ax = axes[1, 0]
        
        color = 'steelblue' if ion_category == 'cation' else 'crimson'
        bars = ax.bar(coordination_numbers, observation_counts, color=color, alpha=0.7)
        
        # Add count labels on bars
        for bar, count in zip(bars, observation_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(observation_counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Coordination Number')
        ax.set_ylabel('Number of Observations')
        ax.set_title(f'{ion_type} Coordination State Populations', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Heatmap
        ax = axes[1, 1]
        
        im = ax.imshow(prob_matrix.T, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(coordination_numbers)))
        ax.set_xticklabels(coordination_numbers)
        ax.set_yticks(range(len(pair_types)))
        ax.set_yticklabels(pair_types)
        
        # Add text annotations
        for i in range(len(coordination_numbers)):
            for j in range(len(pair_types)):
                text = ax.text(i, j, f'{prob_matrix[i, j]:.2f}',
                            ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_xlabel('Coordination Number')
        ax.set_ylabel('Ion Pairing Type')
        ax.set_title(f'{ion_type} Pairing-Coordination Heatmap', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability')
        
        # NEW Plot 5: Coordination state populations as PERCENTAGE histogram
        ax = axes[0, 2]
        
        # Calculate percentages
        total_observations = sum(observation_counts)
        observation_percentages = [(count / total_observations) * 100 for count in observation_counts]
        
        bars = ax.bar(coordination_numbers, observation_percentages, color=color, alpha=0.7, 
                    edgecolor='black', linewidth=1)
        
        # Add percentage labels on bars
        for bar, percentage in zip(bars, observation_percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(observation_percentages)*0.01,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Coordination Number', fontsize=11)
        ax.set_ylabel('Population (%)', fontsize=11)
        ax.set_title(f'{ion_type} Coordination State Distribution', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(observation_percentages) * 1.15)  # Add headroom for labels
        
        # NEW Plot 6: Combined population + pairing info
        ax = axes[1, 2]
        
        # Create a text summary of most probable pairing for each coordination state
        summary_text = f"{ion_type} Coordination-Pairing Summary\n"
        summary_text += "="*40 + "\n\n"
        
        for cn in coordination_numbers:
            data = results[cn]
            probs = data['probabilities']
            total_obs = data['total_observations']
            percentage = (total_obs / total_observations) * 100
            
            # Find most probable pairing type
            most_probable = max(probs.items(), key=lambda x: x[1])
            
            summary_text += f"CN {cn} ({percentage:.1f}%):\n"
            summary_text += f"  Most likely: {most_probable[0]} ({most_probable[1]:.1%})\n"
            summary_text += f"  Observations: {total_obs}\n\n"
        
        # Display text summary
        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.suptitle(f'{ion_type} Ion Pairing Analysis by Coordination State', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f'{ion_type}_coordination_pairing_probabilities.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
        
        plt.show()


    def save_coordination_pairing_analysis_to_file(self, filename='coordination_pairing_analysis_cache.pkl'):
        '''
        Save coordination-pairing analysis results to file for persistence across sessions.
        
        Parameters
        ----------
        filename : str
            Output filename, default='coordination_pairing_analysis_cache.pkl'
        
        Returns
        -------
        success : bool
            True if save was successful
        '''
        
        if not hasattr(self, 'coordination_pairing_analysis') or not self.coordination_pairing_analysis:
            print("No coordination-pairing analysis results to save")
            return False
        
        try:
            # Prepare coordination-pairing data for serialization
            pairing_data = {}
            
            for ion_type, coordination_data in self.coordination_pairing_analysis.items():
                pairing_data[ion_type] = {}
                
                for coordination_number, cn_data in coordination_data.items():
                    pairing_data[ion_type][coordination_number] = {
                        'probabilities': cn_data['probabilities'],
                        'counts': cn_data['counts'],
                        'total_observations': cn_data['total_observations'],
                        'coordination_number': cn_data['coordination_number']
                    }
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(pairing_data, f)
            
            print(f"Coordination-pairing analysis saved to {filename}")
            print(f"  Saved {len(pairing_data)} ion types")
            print(f"  Ion types: {list(pairing_data.keys())}")
            
            # Print summary
            for ion_type, data in pairing_data.items():
                n_coord_states = len(data)
                print(f"    {ion_type}: {n_coord_states} coordination states")
            
            return True
            
        except Exception as e:
            print(f"Error saving coordination-pairing analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_coordination_pairing_analysis_from_file(self, filename='coordination_pairing_analysis_cache.pkl'):
        '''
        Load coordination-pairing analysis results from file with error handling.
        
        Parameters
        ----------
        filename : str
            Input filename, default='coordination_pairing_analysis_cache.pkl'
        
        Returns
        -------
        success : bool
            True if load was successful
        '''
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        try:
            # Check file size first
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print(f"Cache file {filename} is empty")
                return False
            
            # Load data
            with open(filename, 'rb') as f:
                pairing_data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(pairing_data, dict):
                print(f"Invalid coordination-pairing analysis cache format")
                return False
            
            # Reconstruct the coordination-pairing analysis structure
            self.coordination_pairing_analysis = {}
            
            for ion_type, coordination_data in pairing_data.items():
                self.coordination_pairing_analysis[ion_type] = {}
                
                for coordination_number, cn_data in coordination_data.items():
                    self.coordination_pairing_analysis[ion_type][coordination_number] = {
                        'probabilities': cn_data['probabilities'],
                        'counts': cn_data['counts'],
                        'total_observations': cn_data['total_observations'],
                        'coordination_number': cn_data['coordination_number']
                    }
            
            # Print summary
            successful_types = list(self.coordination_pairing_analysis.keys())
            
            print(f"Coordination-pairing analysis loaded from {filename}")
            print(f"  Loaded {len(successful_types)} ion types successfully")
            if successful_types:
                print(f"  Available types: {', '.join(successful_types)}")
            
            # Print detailed summary
            print(f"\n  Coordination-pairing analysis summary:")
            for ion_type, coordination_data in self.coordination_pairing_analysis.items():
                n_coord_states = len(coordination_data)
                coord_numbers = sorted(coordination_data.keys())
                print(f"    {ion_type}: {n_coord_states} coordination states")
                print(f"      Coordination numbers: {coord_numbers}")
                
                # Show pairing breakdown for each coordination state
                for cn in coord_numbers:
                    cn_data = coordination_data[cn]
                    total_obs = cn_data['total_observations']
                    probs = cn_data['probabilities']
                    
                    # Find most probable pairing type
                    if probs:
                        most_probable = max(probs.items(), key=lambda x: x[1])
                        print(f"        CN={cn}: {total_obs} obs, most likely {most_probable[0]} ({most_probable[1]:.1%})")
            
            return True
            
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            print("Removing corrupted cache file...")
            try:
                os.remove(filename)
                print(f"Corrupted file {filename} removed")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading coordination-pairing analysis from {filename}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def compute_ion_pairing_probabilities_by_coordination_with_cache(self, cache_filename='coordination_pairing_analysis_cache.pkl', 
                                                                    force_recalc=False, **kwargs):
        '''
        Compute ion pairing probabilities by coordination with automatic caching.
        
        Parameters
        ----------
        cache_filename : str
            Cache file name, default='coordination_pairing_analysis_cache.pkl'
        force_recalc : bool
            Force recalculation even if cache exists, default=False
        **kwargs : dict
            Arguments to pass to compute_ion_pairing_probabilities_by_coordination()
        
        Returns
        -------
        results : dict
            Dictionary of coordination-pairing analysis results
        '''
        
        # Try to load from cache first
        if not force_recalc and os.path.exists(cache_filename):
            print("Attempting to load coordination-pairing analysis from cache...")
            if self.load_coordination_pairing_analysis_from_file(cache_filename):
                print("✓ Successfully loaded coordination-pairing analysis from cache")
                return self.coordination_pairing_analysis
            else:
                print("✗ Cache loading failed, will recalculate")
        
        # Calculate coordination-pairing analysis
        print("Calculating coordination-pairing analysis...")
        results = self.compute_ion_pairing_probabilities_by_coordination(**kwargs)
        
        # Save to cache
        if results:
            print("Saving coordination-pairing analysis to cache...")
            if self.save_coordination_pairing_analysis_to_file(cache_filename):
                print("✓ Coordination-pairing analysis cached successfully")
            else:
                print("✗ Cache saving failed, but results are available in memory")
        
        return results



    def analyze_coordination_pairing_for_all_types(self, coordination_states=None, step=None, save_plots=True):
        '''
        Analyze coordination-pairing relationships for all ion types.
        
        Parameters
        ----------
        coordination_states : list or None
            Coordination numbers to analyze
        step : int
            Step size for trajectory analysis
        save_plots : bool
            Whether to save plots
        
        Returns
        -------
        all_results : dict
            Results for all ion types
        '''
        
        # Check if we have ion pairing data
        if not hasattr(self, 'ion_pairs_by_type'):
            print("No ion pairing data available. Run determine_ion_pairing_cutoffs() for different ion types first.")
            return None
        
        available_ion_types = list(self.ion_pairs_by_type.keys())
        print(f"Analyzing coordination-pairing relationships for: {available_ion_types}")
        
        all_results = {}
        
        for ion_type in available_ion_types:
            print(f"\n{'='*50}")
            print(f"Processing {ion_type}")
            print(f"{'='*50}")
            
            try:
                results = self.compute_ion_pairing_probabilities_by_coordination(
                    ion_type=ion_type,
                    coordination_states=coordination_states,
                    step=step,
                    save_plots=save_plots
                )
                
                if results:
                    all_results[ion_type] = results
                    
            except Exception as e:
                print(f"Error analyzing {ion_type}: {e}")
                continue
        
        # Create comparison plot if we have multiple ion types
        if len(all_results) > 1 and save_plots:
            self._plot_coordination_pairing_comparison(all_results)
        
        return all_results

    def _plot_coordination_pairing_comparison(self, all_results):
        '''Create comparison plot across different ion types'''
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Separate by ion category
        cation_results = {}
        anion_results = {}
        
        for ion_type, results in all_results.items():
            if hasattr(self, 'ion_pairs_by_type') and ion_type in self.ion_pairs_by_type:
                ion_category = self.ion_pairs_by_type[ion_type]['ion_category']
                if ion_category == 'cation':
                    cation_results[ion_type] = results
                else:
                    anion_results[ion_type] = results
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: CIP probabilities by coordination number
        ax = axes[0, 0]
        
        for ion_type, results in cation_results.items():
            cns = sorted(results.keys())
            cip_probs = [results[cn]['probabilities'].get('CIP', 0) for cn in cns]
            ax.plot(cns, cip_probs, 'o-', label=f'{ion_type} (cation)', linewidth=2)
        
        for ion_type, results in anion_results.items():
            cns = sorted(results.keys())
            cip_probs = [results[cn]['probabilities'].get('CIP', 0) for cn in cns]
            ax.plot(cns, cip_probs, 's--', label=f'{ion_type} (anion)', linewidth=2)
        
        ax.set_xlabel('Coordination Number')
        ax.set_ylabel('CIP Probability')
        ax.set_title('Contact Ion Pair Probability vs Coordination', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: SIP probabilities by coordination number
        ax = axes[0, 1]
        
        for ion_type, results in cation_results.items():
            cns = sorted(results.keys())
            sip_probs = [results[cn]['probabilities'].get('SIP', 0) for cn in cns]
            ax.plot(cns, sip_probs, 'o-', label=f'{ion_type} (cation)', linewidth=2)
        
        for ion_type, results in anion_results.items():
            cns = sorted(results.keys())
            sip_probs = [results[cn]['probabilities'].get('SIP', 0) for cn in cns]
            ax.plot(cns, sip_probs, 's--', label=f'{ion_type} (anion)', linewidth=2)
        
        ax.set_xlabel('Coordination Number')
        ax.set_ylabel('SIP Probability')
        ax.set_title('Solvent-Separated Pair Probability vs Coordination', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Free ion probabilities by coordination number
        ax = axes[1, 0]
        
        for ion_type, results in cation_results.items():
            cns = sorted(results.keys())
            fi_probs = [results[cn]['probabilities'].get('FI', 0) for cn in cns]
            ax.plot(cns, fi_probs, 'o-', label=f'{ion_type} (cation)', linewidth=2)
        
        for ion_type, results in anion_results.items():
            cns = sorted(results.keys())
            fi_probs = [results[cn]['probabilities'].get('FI', 0) for cn in cns]
            ax.plot(cns, fi_probs, 's--', label=f'{ion_type} (anion)', linewidth=2)
        
        ax.set_xlabel('Coordination Number')
        ax.set_ylabel('FI Probability')
        ax.set_title('Free Ion Probability vs Coordination', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Most probable coordination numbers
        ax = axes[1, 1]
        
        ion_names = []
        most_probable_cns = []
        colors = []
        
        for ion_type, results in all_results.items():
            ion_names.append(ion_type)
            
            # Find most probable coordination number (highest observation count)
            max_obs = 0
            most_prob_cn = 0
            for cn, data in results.items():
                if data['total_observations'] > max_obs:
                    max_obs = data['total_observations']
                    most_prob_cn = cn
            
            most_probable_cns.append(most_prob_cn)
            
            # Color by category
            if hasattr(self, 'ion_pairs_by_type') and ion_type in self.ion_pairs_by_type:
                ion_category = self.ion_pairs_by_type[ion_type]['ion_category']
                colors.append('steelblue' if ion_category == 'cation' else 'crimson')
            else:
                colors.append('gray')
        
        bars = ax.bar(ion_names, most_probable_cns, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, cn in zip(bars, most_probable_cns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(cn), ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Ion Type')
        ax.set_ylabel('Most Probable Coordination Number')
        ax.set_title('Preferred Coordination Numbers', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Ion Pairing vs Coordination: Multi-Ion Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = 'coordination_pairing_comparison_all_types.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved as: {filename}")
        
        plt.show()

    def get_coordination_pairing_data(self, ion_type, coordination_number):
        '''
        Get detailed pairing data for a specific ion type and coordination number.
        
        Parameters
        ----------
        ion_type : str
            Ion type name
        coordination_number : int
            Coordination number of interest
        
        Returns
        -------
        data : dict or None
            Detailed pairing data for the specified state
        '''
        
        if not hasattr(self, 'coordination_pairing_analysis'):
            print("Coordination-pairing analysis not performed. Run compute_ion_pairing_probabilities_by_coordination() first.")
            return None
        
        if ion_type not in self.coordination_pairing_analysis:
            available_types = list(self.coordination_pairing_analysis.keys())
            print(f"Ion type '{ion_type}' not found. Available types: {available_types}")
            return None
        
        if coordination_number not in self.coordination_pairing_analysis[ion_type]:
            available_cns = list(self.coordination_pairing_analysis[ion_type].keys())
            print(f"Coordination number {coordination_number} not found for {ion_type}. Available: {available_cns}")
            return None
        
        return self.coordination_pairing_analysis[ion_type][coordination_number]



    def polyhedron_size(self, ion='cation', r0=None, njobs=None, step=None, n_points=100):
        '''
        Optimized polyhedron analysis with batching and reduced point sampling.
        '''
        
        # Use auto-tuned defaults
        if step is None:
            step = self.default_step
        if njobs is None:
            njobs = self.default_njobs
        
        if ion == 'cation':
            ions = self.cations
            if r0 is None:
                try:
                    r0 = self.solute_ci.radii['water']
                except NameError:
                    print('Solutes not initialized. Try `initialize_Solutes()` first')
                    return None
        elif ion == 'anion':
            ions = self.anions
            if r0 is None:
                try:
                    r0 = self.solute_ai.radii['water']
                except NameError:
                    print('Solutes not initialized. Try `initialize_Solutes()` first')
                    return None
        else:
            raise NameError("Options for kwarg ion are 'cation' or 'anion'")
        
        # Use all CPUs if -1
        if njobs == -1:
            njobs = multiprocessing.cpu_count()
        
        # Process in batches to reduce memory usage
        n_frames = len(self.universe.trajectory[::step])
        frame_batches = [range(i, min(i + self.batch_size, n_frames)) 
                         for i in range(0, n_frames, self.batch_size)]
        
        results = Results()
        results.areas = np.zeros((len(ions), n_frames))
        results.volumes = np.zeros((len(ions), n_frames))
        
        print(f"Processing {n_frames} frames in {len(frame_batches)} batches...")
        print(f"Using {njobs} CPUs, reduced point sampling ({n_points} points)")
        
        for batch_idx, frame_range in enumerate(tqdm(frame_batches, desc="Processing batches")):
            
            if njobs == 1:
                # Single-threaded batch processing
                for i in frame_range:
                    a, v = self._polyhedron_size_per_frame(i, ions, r0, n_points)
                    results.areas[:, i] = a
                    results.volumes[:, i] = v
            else:
                # Multi-threaded batch processing
                run_per_frame = partial(self._polyhedron_size_per_frame,
                                       ions=ions, r0=r0, n_points=n_points)
                
                with Pool(njobs, initializer=_worker_init) as worker_pool:
                    batch_results = worker_pool.map(run_per_frame, frame_range)
                
                for i, (a, v) in enumerate(batch_results):
                    frame_idx = batch_idx * self.batch_size + i
                    if frame_idx < n_frames:
                        results.areas[:, frame_idx] = a
                        results.volumes[:, frame_idx] = v
            
            # Clear memory periodically
            if batch_idx % 5 == 0 and batch_idx > 0:
                gc.collect()
        
        print("Polyhedron analysis completed")
        return results

    def _polyhedron_size_per_frame(self, frame_idx, ions, r0, n_points=100):
        '''
        Optimized per-frame polyhedron calculation with reduced point sampling.
        '''
        
        self.universe.trajectory[frame_idx]
        
        volumes = np.zeros(len(ions))
        areas = np.zeros(len(ions))
        
        # Pre-compute water positions for efficiency
        water_positions = self.waters.positions
        
        for j, ion in enumerate(ions):
            try:
                # More efficient shell selection using vectorized distance
                ion_pos = ion.position
                distances_to_ion = np.linalg.norm(water_positions - ion_pos, axis=1)
                shell_mask = distances_to_ion <= r0
                
                if shell_mask.sum() < 4:  # Need at least 4 points for ConvexHull
                    volumes[j] = 0
                    areas[j] = 0
                    continue
                
                shell_positions = water_positions[shell_mask]
                
                # Unwrap shell more efficiently
                pos = self._unwrap_shell_optimized(ion_pos, shell_positions)
                
                # Reduced point sampling for speed
                pos = self._points_on_atomic_radius_optimized(pos, n_points)
                
                # Create ConvexHull
                hull = ConvexHull(pos)
                volumes[j] = hull.volume
                
                # Simplified area calculation
                areas[j] = self._calculate_max_area_optimized(pos, hull)
                
            except Exception as e:
                # Handle degenerate cases gracefully
                volumes[j] = 0
                areas[j] = 0
        
        return areas, volumes


    def save_polyhedron_results_to_file(self, filename='polyhedron_cache.pkl'):
        '''
        Save polyhedron analysis results to file for persistence across sessions.
        
        Parameters
        ----------
        filename : str
            Output filename, default='polyhedron_cache.pkl'
        
        Returns
        -------
        success : bool
            True if save was successful
        '''
        
        if not hasattr(self, 'polyhedron_results_by_type') or not self.polyhedron_results_by_type:
            print("No polyhedron results to save")
            return False
        
        try:
            # Prepare polyhedron data for serialization
            polyhedron_data = {}
            
            for ion_type, poly_results in self.polyhedron_results_by_type.items():
                if poly_results is not None:
                    # Store the essential data from Results object
                    polyhedron_data[ion_type] = {
                        'areas': poly_results.areas.copy(),
                        'volumes': poly_results.volumes.copy(),
                        'ion_type': poly_results.ion_type,
                        'ion_category': poly_results.ion_category,
                        'coordination_radius': poly_results.coordination_radius,
                        'n_ions': poly_results.n_ions,
                        'mean_area': poly_results.mean_area.copy(),
                        'mean_volume': poly_results.mean_volume.copy(),
                        'std_area': poly_results.std_area.copy(),
                        'std_volume': poly_results.std_volume.copy(),
                        'overall_mean_area': poly_results.overall_mean_area,
                        'overall_mean_volume': poly_results.overall_mean_volume,
                        'overall_std_area': poly_results.overall_std_area,
                        'overall_std_volume': poly_results.overall_std_volume
                    }
                else:
                    polyhedron_data[ion_type] = None
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(polyhedron_data, f)
            
            print(f"Polyhedron results saved to {filename}")
            print(f"  Saved {len([v for v in polyhedron_data.values() if v is not None])} ion types")
            print(f"  Ion types: {list(polyhedron_data.keys())}")
            
            return True
            
        except Exception as e:
            print(f"Error saving polyhedron results: {e}")
            return False

    def load_polyhedron_results_from_file(self, filename='polyhedron_cache.pkl'):
        '''
        Load polyhedron results from file with error handling.
        
        Parameters
        ----------
        filename : str
            Input filename, default='polyhedron_cache.pkl'
        
        Returns
        -------
        success : bool
            True if load was successful
        '''
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        try:
            # Check file size first
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print(f"Cache file {filename} is empty")
                return False
            
            # Load data
            with open(filename, 'rb') as f:
                polyhedron_data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(polyhedron_data, dict):
                print(f"Invalid polyhedron cache format")
                return False
            
            # Reconstruct Results objects
            from MDAnalysis.analysis.base import Results
            
            self.polyhedron_results_by_type = {}
            
            for ion_type, data in polyhedron_data.items():
                if data is not None and isinstance(data, dict):
                    # Create Results object
                    results = Results()
                    results.areas = data['areas']
                    results.volumes = data['volumes']
                    results.ion_type = data['ion_type']
                    results.ion_category = data['ion_category']
                    results.coordination_radius = data['coordination_radius']
                    results.n_ions = data['n_ions']
                    results.mean_area = data['mean_area']
                    results.mean_volume = data['mean_volume']
                    results.std_area = data['std_area']
                    results.std_volume = data['std_volume']
                    results.overall_mean_area = data['overall_mean_area']
                    results.overall_mean_volume = data['overall_mean_volume']
                    results.overall_std_area = data['overall_std_area']
                    results.overall_std_volume = data['overall_std_volume']
                    
                    self.polyhedron_results_by_type[ion_type] = results
                else:
                    self.polyhedron_results_by_type[ion_type] = None
            
            # Print summary
            successful_types = [k for k, v in self.polyhedron_results_by_type.items() if v is not None]
            failed_types = [k for k, v in self.polyhedron_results_by_type.items() if v is None]
            
            print(f"Polyhedron results loaded from {filename}")
            print(f"  Loaded {len(successful_types)} ion types successfully")
            if successful_types:
                print(f"  Available types: {', '.join(successful_types)}")
            if failed_types:
                print(f"  Failed to load: {', '.join(failed_types)}")
            
            # Print summary of loaded data
            print(f"\n  Polyhedron summary:")
            for ion_type in successful_types:
                poly_data = self.polyhedron_results_by_type[ion_type]
                print(f"    {ion_type}: Vol={poly_data.overall_mean_volume:.1f}±{poly_data.overall_std_volume:.1f} Å³, "
                    f"Area={poly_data.overall_mean_area:.1f}±{poly_data.overall_std_area:.1f} Å²")
            
            return True
            
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            print("Removing corrupted cache file...")
            try:
                os.remove(filename)
                print(f"Corrupted file {filename} removed")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading polyhedron results from {filename}: {e}")
            return False

    def polyhedron_size_by_type_with_cache(self, cache_filename='polyhedron_cache.pkl', 
                                        force_recalc=False, **kwargs):
        '''
        Calculate polyhedron sizes with automatic caching.
        
        Parameters
        ----------
        cache_filename : str
            Cache file name, default='polyhedron_cache.pkl'
        force_recalc : bool
            Force recalculation even if cache exists, default=False
        **kwargs : dict
            Arguments to pass to polyhedron_size_by_type()
        
        Returns
        -------
        results : dict
            Dictionary of polyhedron results
        '''
        
        # Try to load from cache first
        if not force_recalc and os.path.exists(cache_filename):
            print("Attempting to load polyhedron results from cache...")
            if self.load_polyhedron_results_from_file(cache_filename):
                print("✓ Successfully loaded polyhedron results from cache")
                return self.polyhedron_results_by_type
            else:
                print("✗ Cache loading failed, will recalculate")
        
        # Calculate polyhedron sizes
        print("Calculating polyhedron sizes...")
        results = self.polyhedron_size_by_type(**kwargs)
        
        # Save to cache
        if results:
            print("Saving polyhedron results to cache...")
            if self.save_polyhedron_results_to_file(cache_filename):
                print("✓ Polyhedron results cached successfully")
            else:
                print("✗ Cache saving failed, but results are available in memory")
        
        return results


    def _unwrap_shell_optimized(self, ion_pos, shell_positions):
        '''Optimized shell unwrapping using vectorized operations'''
        dims = self.universe.dimensions[:3]
        dist = ion_pos - shell_positions
        
        # Vectorized unwrapping
        correction = np.where(np.abs(dist) > dims/2,
                             np.sign(dist) * dims,
                             0)
        return shell_positions + correction

    def _points_on_atomic_radius_optimized(self, positions, n_points=100):
        '''Optimized point generation with reduced sampling'''
        
        n_atoms = len(positions)
        
        # Pre-allocate arrays
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        # Generate points more efficiently
        theta = np.arccos(rng.uniform(-1, 1, (n_atoms, n_points)))
        phi = rng.uniform(0, 2*np.pi, (n_atoms, n_points))
        
        # Default radius for simplification - can be made more sophisticated
        radius = 1.5  # Simplified single radius
        
        # Vectorized coordinate calculation
        x = radius * np.sin(theta) * np.cos(phi) + positions[:, 0, None]
        y = radius * np.sin(theta) * np.sin(phi) + positions[:, 1, None]
        z = radius * np.cos(theta) + positions[:, 2, None]
        
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    def _calculate_max_area_optimized(self, pos, hull):
        '''Simplified area calculation for better performance'''
        
        # Use a simpler approximation for max cross-sectional area
        # This is much faster than the full PCA + plane intersection method
        
        # Get the bounding box dimensions
        mins = pos.min(axis=0)
        maxs = pos.max(axis=0)
        dimensions = maxs - mins
        
        # Approximate maximum area as the largest rectangular cross-section
        # This is a simplification but much faster
        areas = [
            dimensions[1] * dimensions[2],  # YZ plane
            dimensions[0] * dimensions[2],  # XZ plane  
            dimensions[0] * dimensions[1]   # XY plane
        ]
        
        return max(areas)


    def polyhedron_size_by_type(self, ion_type=None, r0=None, njobs=None, step=None, n_points=100):
        '''
        Optimized polyhedron analysis with TRUE parallel processing for specific ion types.
        FIXED: Now properly parallelizes frame batches for significant speedup.
        
        Parameters
        ----------
        ion_type : str or None
            Specific ion type to analyze (e.g., 'Na', 'Mg', 'Cl'). 
            If None, analyzes all ion types separately.
        r0 : float, optional
            Coordination radius. If None, uses the radius from the ion-type-specific solute.
        njobs : int, optional
            Number of parallel jobs
        step : int, optional
            Step size for trajectory analysis
        n_points : int
            Number of points for surface sampling, default=100
        
        Returns
        -------
        results : dict
            Dictionary with ion types as keys and their polyhedron data as values
        '''
        
        # Use auto-tuned defaults
        if step is None:
            step = self.default_step
        if njobs is None:
            njobs = self.default_njobs
        
        # Check if ion-type-specific solutes are available
        if not (hasattr(self, 'solutes_ci') and hasattr(self, 'solutes_ai')):
            print('Ion-type-specific solutes not initialized. Try initialize_Solutes_by_type() first')
            return None
        
        # Get unique ion types
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        # Determine which ions to process
        if ion_type is not None:
            # Process specific ion type
            if ion_type in cation_types:
                ions_to_process = [(ion_type, 'cation', cation_types[ion_type])]
            elif ion_type in anion_types:
                ions_to_process = [(ion_type, 'anion', anion_types[ion_type])]
            else:
                print(f"Ion type '{ion_type}' not found.")
                available_types = list(cation_types.keys()) + list(anion_types.keys())
                print(f"Available types: {available_types}")
                return None
        else:
            # Process all ion types
            ions_to_process = [(name, 'cation', group) for name, group in cation_types.items()]
            ions_to_process += [(name, 'anion', group) for name, group in anion_types.items()]
        
        print(f"Calculating polyhedron sizes for ion types: {[ion[0] for ion in ions_to_process]}")
        
        # Use all CPUs if -1
        if njobs == -1:
            njobs = min(multiprocessing.cpu_count(), 8)
        
        # Test multiprocessing compatibility
        njobs = self._test_multiprocessing_compatibility(njobs)
        
        n_frames = len(self.universe.trajectory[::step])
        results = {}
        
        # Process each ion type
        for ion_name, ion_category, ion_group in ions_to_process:
            print(f"\nProcessing {ion_name} ({ion_category}) polyhedra...")
            print(f"  {len(ion_group)} ions, {n_frames} frames")
            
            # Get coordination radius for this ion type
            if r0 is None:
                if ion_category == 'cation':
                    if ion_name in self.solutes_ci and self.solutes_ci[ion_name] is not None:
                        coordination_radius = self.solutes_ci[ion_name].radii['water']
                    else:
                        print(f"  Warning: No radius for {ion_name}, using default 2.8")
                        coordination_radius = 2.8
                else:
                    if ion_name in self.solutes_ai and self.solutes_ai[ion_name] is not None:
                        coordination_radius = self.solutes_ai[ion_name].radii['water']
                    else:
                        print(f"  Warning: No radius for {ion_name}, using default 3.5")
                        coordination_radius = 3.5
            else:
                coordination_radius = r0
                print(f"  Using provided coordination radius: {coordination_radius:.2f} Å")
            
            # FIXED: Create frame batches for parallel processing
            frame_batches = []
            batch_size = max(1, n_frames // (njobs * 2))  # More batches than CPUs for better load balancing
            
            for i in range(0, n_frames, batch_size):
                batch_end = min(i + batch_size, n_frames)
                frame_batches.append(range(i, batch_end))
            
            # Initialize results for this ion type
            ion_results = Results()
            ion_results.areas = np.zeros((len(ion_group), n_frames))
            ion_results.volumes = np.zeros((len(ion_group), n_frames))
            ion_results.ion_type = ion_name
            ion_results.ion_category = ion_category
            ion_results.coordination_radius = coordination_radius
            ion_results.n_ions = len(ion_group)
            
            print(f"  Processing {n_frames} frames in {len(frame_batches)} batches...")
            print(f"  Using {njobs} CPUs, {n_points} surface points")
            
            if njobs == 1:
                # Sequential processing
                for batch_idx, frame_range in enumerate(tqdm(frame_batches, 
                                                        desc=f"Processing {ion_name} batches", 
                                                        leave=False)):
                    for frame_idx in frame_range:
                        areas, volumes = self._polyhedron_size_per_frame_by_type(
                            frame_idx, ion_group, coordination_radius, n_points
                        )
                        ion_results.areas[:, frame_idx] = areas
                        ion_results.volumes[:, frame_idx] = volumes
            else:
                # FIXED: TRUE parallel processing
                print(f"  Using parallel processing with {njobs} workers")
                
                # Create partial function with fixed parameters
                process_frame_func = partial(
                    self._polyhedron_size_per_frame_by_type,
                    ions=ion_group,
                    r0=coordination_radius,
                    n_points=n_points
                )
                
                try:
                    with Pool(njobs, initializer=_worker_init) as worker_pool:
                        # Process ALL frames in parallel
                        if hasattr(self, '_debug_frame_indices') and self._debug_frame_indices is not None:
                            frame_indices = self._debug_frame_indices[::step]
                        else:
                            frame_indices = list(range(0, len(self.universe.trajectory), step))
                        
                        # Map frames to workers with progress bar
                        batch_results = list(tqdm(
                            worker_pool.imap(process_frame_func, frame_indices),
                            total=len(frame_indices),
                            desc=f"Processing {ion_name} frames (parallel)"
                        ))
                        
                        # Store results
                        for i, (areas, volumes) in enumerate(batch_results):
                            ion_results.areas[:, i] = areas
                            ion_results.volumes[:, i] = volumes
                    
                    print(f"  ✓ Parallel processing completed successfully")
                    
                except Exception as e:
                    print(f"  ✗ Parallel processing failed: {e}")
                    print(f"  Falling back to sequential processing...")
                    
                    # Fallback to sequential
                    for batch_idx, frame_range in enumerate(tqdm(frame_batches, 
                                                            desc=f"Processing {ion_name} batches (fallback)", 
                                                            leave=False)):
                        for frame_idx in frame_range:
                            areas, volumes = self._polyhedron_size_per_frame_by_type(
                                frame_idx, ion_group, coordination_radius, n_points
                            )
                            ion_results.areas[:, frame_idx] = areas
                            ion_results.volumes[:, frame_idx] = volumes
            
            # Calculate statistics for this ion type
            ion_results.mean_area = ion_results.areas.mean(axis=1)  # Per ion
            ion_results.mean_volume = ion_results.volumes.mean(axis=1)  # Per ion
            ion_results.std_area = ion_results.areas.std(axis=1)
            ion_results.std_volume = ion_results.volumes.std(axis=1)
            
            # Overall statistics
            ion_results.overall_mean_area = ion_results.areas.mean()
            ion_results.overall_mean_volume = ion_results.volumes.mean()
            ion_results.overall_std_area = ion_results.areas.std()
            ion_results.overall_std_volume = ion_results.volumes.std()
            
            results[ion_name] = ion_results
            
            print(f"  ✓ {ion_name} polyhedron analysis complete")
            print(f"    Mean volume: {ion_results.overall_mean_volume:.2f} ± {ion_results.overall_std_volume:.2f} Å³")
            print(f"    Mean area: {ion_results.overall_mean_area:.2f} ± {ion_results.overall_std_area:.2f} Å²")
            
            # Clear memory periodically
            gc.collect()

        # Store results
        self.polyhedron_results_by_type = results
        
        # Print summary
        self._print_polyhedron_summary_by_type(results)
        
        print("Ion-type-specific polyhedron analysis completed")
        return results



    def _polyhedron_size_per_frame_by_type(self, frame_idx, ions, r0, n_points=100):
        '''
        Optimized per-frame polyhedron calculation for specific ion type.
        '''
        
        self.universe.trajectory[frame_idx]
        
        volumes = np.zeros(len(ions))
        areas = np.zeros(len(ions))
        
        # Pre-compute water positions for efficiency
        water_positions = self.waters.positions
        
        for j, ion in enumerate(ions):
            try:
                # More efficient shell selection using vectorized distance
                ion_pos = ion.position
                distances_to_ion = np.linalg.norm(water_positions - ion_pos, axis=1)
                shell_mask = distances_to_ion <= r0
                
                if shell_mask.sum() < 4:  # Need at least 4 points for ConvexHull
                    volumes[j] = 0
                    areas[j] = 0
                    continue
                
                shell_positions = water_positions[shell_mask]
                
                # Unwrap shell more efficiently
                pos = self._unwrap_shell_optimized(ion_pos, shell_positions)
                
                # Reduced point sampling for speed
                pos = self._points_on_atomic_radius_optimized(pos, n_points)
                
                # Create ConvexHull
                hull = ConvexHull(pos)
                volumes[j] = hull.volume
                
                # Simplified area calculation
                areas[j] = self._calculate_max_area_optimized(pos, hull)
                
            except Exception as e:
                # Handle degenerate cases gracefully
                volumes[j] = 0
                areas[j] = 0
        
        return areas, volumes

    def _print_polyhedron_summary_by_type(self, results):
        '''Print summary of polyhedron analysis by ion type'''
        
        print("\n" + "="*70)
        print("POLYHEDRON ANALYSIS SUMMARY BY ION TYPE")
        print("="*70)
        
        for ion_type, ion_results in results.items():
            ion_category = ion_results.ion_category.upper()
            print(f"\n{ion_category}: {ion_type}")
            print("-" * 50)
            print(f"Number of ions: {ion_results.n_ions}")
            print(f"Coordination radius: {ion_results.coordination_radius:.2f} Å")
            print(f"Mean volume: {ion_results.overall_mean_volume:.2f} ± {ion_results.overall_std_volume:.2f} Å³")
            print(f"Mean area: {ion_results.overall_mean_area:.2f} ± {ion_results.overall_std_area:.2f} Å²")

            # Show range
            min_vol = ion_results.volumes.min()
            max_vol = ion_results.volumes.max()
            min_area = ion_results.areas.min()
            max_area = ion_results.areas.max()

            print(f"Volume range: {min_vol:.2f} - {max_vol:.2f} Å³")
            print(f"Area range: {min_area:.2f} - {max_area:.2f} Å²")

        print("="*70)

    def plot_polyhedron_results_by_type(self, save_plots=True, plot_range=None):
        '''
        Plot polyhedron analysis results for each ion type.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save plots, default=True
        plot_range : tuple
            Range of frames to plot (start, end), default=None (all frames)
        '''
        
        if not hasattr(self, 'polyhedron_results_by_type'):
            print("Polyhedron results by type not calculated. Run polyhedron_size_by_type() first.")
            return
        
        results = self.polyhedron_results_by_type
        
        # Separate cations and anions
        cations_data = {k: v for k, v in results.items() if v.ion_category == 'cation'}
        anions_data = {k: v for k, v in results.items() if v.ion_category == 'anion'}
        
        n_cations = len(cations_data)
        n_anions = len(anions_data)
        
        if n_cations == 0 and n_anions == 0:
            print("No polyhedron data found")
            return
        
        # Get frame indices
        if plot_range is not None:
            start, end = plot_range
            frame_indices = np.arange(start, end)
        else:
            # Use the length from any ion type
            sample_data = list(results.values())[0]
            frame_indices = np.arange(sample_data.volumes.shape[1])
        
        # Create figure - separate plots for volumes and areas
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot cation volumes
        if n_cations > 0:
            ax = axes[0, 0]
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_cations))
            
            for i, (ion_type, ion_data) in enumerate(cations_data.items()):
                # Plot mean volume over time
                mean_volumes = ion_data.volumes.mean(axis=0)
                if plot_range is not None:
                    mean_volumes = mean_volumes[start:end]
                
                ax.plot(frame_indices, mean_volumes, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f"{ion_type} (μ={ion_data.overall_mean_volume:.1f})")
                
                # Add horizontal line for overall mean
                ax.axhline(ion_data.overall_mean_volume, color=colors[i], 
                        linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_title('Cation Polyhedron Volumes', fontweight='bold')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Volume (Å³)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Cation Data', ha='center', va='center', 
                        transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Cation Polyhedron Volumes', fontweight='bold')
        
        # Plot cation areas
        if n_cations > 0:
            ax = axes[0, 1]
            
            for i, (ion_type, ion_data) in enumerate(cations_data.items()):
                # Plot mean area over time
                mean_areas = ion_data.areas.mean(axis=0)
                if plot_range is not None:
                    mean_areas = mean_areas[start:end]
                
                ax.plot(frame_indices, mean_areas, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f"{ion_type} (μ={ion_data.overall_mean_area:.1f})")
                
                # Add horizontal line for overall mean
                ax.axhline(ion_data.overall_mean_area, color=colors[i], 
                        linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_title('Cation Polyhedron Areas', fontweight='bold')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Area (Å²)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Cation Data', ha='center', va='center', 
                        transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Cation Polyhedron Areas', fontweight='bold')
        
        # Plot anion volumes
        if n_anions > 0:
            ax = axes[1, 0]
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, n_anions))
            
            for i, (ion_type, ion_data) in enumerate(anions_data.items()):
                # Plot mean volume over time
                mean_volumes = ion_data.volumes.mean(axis=0)
                if plot_range is not None:
                    mean_volumes = mean_volumes[start:end]
                
                ax.plot(frame_indices, mean_volumes, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f"{ion_type} (μ={ion_data.overall_mean_volume:.1f})")
                
                # Add horizontal line for overall mean
                ax.axhline(ion_data.overall_mean_volume, color=colors[i], 
                        linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_title('Anion Polyhedron Volumes', fontweight='bold')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Volume (Å³)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Anion Data', ha='center', va='center', 
                        transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Anion Polyhedron Volumes', fontweight='bold')
        
        # Plot anion areas
        if n_anions > 0:
            ax = axes[1, 1]
            
            for i, (ion_type, ion_data) in enumerate(anions_data.items()):
                # Plot mean area over time
                mean_areas = ion_data.areas.mean(axis=0)
                if plot_range is not None:
                    mean_areas = mean_areas[start:end]
                
                ax.plot(frame_indices, mean_areas, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f"{ion_type} (μ={ion_data.overall_mean_area:.1f})")
                
                # Add horizontal line for overall mean
                ax.axhline(ion_data.overall_mean_area, color=colors[i], 
                        linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_title('Anion Polyhedron Areas', fontweight='bold')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Area (Å²)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Anion Data', ha='center', va='center', 
                        transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Anion Polyhedron Areas', fontweight='bold')
        
        plt.suptitle('Polyhedron Analysis by Ion Type', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('polyhedron_analysis_by_type.png', dpi=300, bbox_inches='tight')
            print("Plot saved as: polyhedron_analysis_by_type.png")
        
        plt.show()

    def get_polyhedron_results_for_type(self, ion_type):
        '''
        Get polyhedron analysis results for a specific ion type.
        
        Parameters
        ----------
        ion_type : str
            Ion type name (e.g., 'Na', 'Mg', 'Cl')
        
        Returns
        -------
        results : Results or None
            Polyhedron analysis results for the specified ion type
        '''
        
        if not hasattr(self, 'polyhedron_results_by_type'):
            print("Polyhedron results by type not calculated. Run polyhedron_size_by_type() first.")
            return None
        
        if ion_type in self.polyhedron_results_by_type:
            return self.polyhedron_results_by_type[ion_type]
        else:
            available_types = list(self.polyhedron_results_by_type.keys())
            print(f"Ion type '{ion_type}' not found. Available types: {available_types}")
            return None



    def compare_polyhedron_sizes_by_type(
        self,
        save_plots=True,
        figsize=(14, 8),
        # --- typography (base) ---
        font_size=12,
        font_weight='normal',
        # --- title ---
        show_title=True,
        title_font_size=None,           # None → font_size + 2
        title_font_weight='bold',
        # --- axis labels ---
        label_font_size=None,           # None → font_size
        label_font_weight=None,         # None → font_weight
        # --- tick labels ---
        tick_font_size=None,            # None → font_size - 1
        tick_font_weight=None,          # None → font_weight
        # --- bar value labels ---
        show_bar_label=True,
        bar_label_font_size=None,       # None → max(font_size - 3, 6)
        bar_label_font_weight=None,     # None → font_weight
        # --- bars ---
        bar_width=0.35,
        bar_alpha=0.8,
        bar_linewidth=1.0,
        edgecolor='black',
        capsize=5,
        # --- per-ion colours ---
        ion_colors=None,                # dict {ion_name: color}; None → auto palette
        # --- hatch to distinguish volume vs area ---
        volume_hatch='',                # hatch for volume bars, e.g. '///'
        area_hatch='xxx',               # hatch for area bars, e.g. '\\\\'
        # --- grid ---
        grid_alpha=0.3,
        # --- legend (ion colours) ---
        show_ion_legend=True,
        ion_legend_title='Ion',
        ion_legend_bbox_to_anchor=None, # e.g. (1.0, 1.0)
        ion_legend_loc='upper left',
        # --- legend (volume/area hatches) ---
        show_hatch_legend=True,
        hatch_legend_title='Quantity',
        hatch_legend_bbox_to_anchor=None,
        hatch_legend_loc='upper right',
        # --- shared legend style ---
        legend_font_size=None,          # None → font_size
        legend_font_weight=None,        # None → font_weight
        legend_frame_alpha=None,        # e.g. 0.0 for transparent
        # --- axis label text ---
        xlabel='Ion Type',
        ylabel='\u00c5\u00b3 for Volume, \u00c5\u00b2 for Area',  # None → suppress
        # --- output ---
        dpi=300,
        output_filename=None,
        transparent=False,
    ):
        '''
        Comparison bar chart of polyhedron Volume and Area for every ion type.

        Encoding:
          - **Color**  → ion identity  (one colour per ion, same for both bars)
          - **Hatch**  → quantity type (volume_hatch vs area_hatch)
          - **Legend 1** (ion colours): maps colour patch → ion name
          - **Legend 2** (hatch types): maps hatch patch → "Volume (Å³)" / "Area (Å²)"

        Parameters
        ----------
        ion_colors : dict or None
            ``{ion_name: color}`` overrides. None → auto-assigned from a default
            palette (cations use blue tones, anions use red tones).
        volume_hatch : str
            Matplotlib hatch string for volume bars, default='' (solid).
        area_hatch : str
            Matplotlib hatch string for area bars, default='xxx'.
        show_ion_legend : bool
            Show colour → ion-name legend, default=True.
        ion_legend_title : str
            Title for the ion legend, default='Ion'.
        ion_legend_bbox_to_anchor : tuple or None
            Anchor for ion legend; None → matplotlib default.
        ion_legend_loc : str
            Location string for ion legend, default='upper left'.
        show_hatch_legend : bool
            Show hatch → quantity legend, default=True.
        hatch_legend_title : str
            Title for the hatch legend, default='Quantity'.
        hatch_legend_bbox_to_anchor : tuple or None
            Anchor for hatch legend; None → matplotlib default.
        hatch_legend_loc : str
            Location string for hatch legend, default='upper right'.
        legend_font_size : float or None
            Font size for both legends; None → font_size.
        legend_font_weight : str or None
            Font weight for both legends; None → font_weight.
        legend_frame_alpha : float or None
            Frame alpha for both legends (0.0 = transparent); None → default.
        '''

        if not hasattr(self, 'polyhedron_results_by_type'):
            print("Polyhedron results by type not calculated. Run polyhedron_size_by_type() first.")
            return

        # --- resolve font fallbacks ---
        _title_fs = title_font_size       if title_font_size       is not None else font_size + 2
        _lbl      = label_font_size       if label_font_size       is not None else font_size
        _lbl_w    = label_font_weight     if label_font_weight     is not None else font_weight
        _tick     = tick_font_size        if tick_font_size        is not None else font_size - 1
        _tick_w   = tick_font_weight      if tick_font_weight      is not None else font_weight
        _blbl     = bar_label_font_size   if bar_label_font_size   is not None else max(font_size - 3, 6)
        _blbl_w   = bar_label_font_weight if bar_label_font_weight is not None else font_weight
        _leg      = legend_font_size      if legend_font_size      is not None else font_size
        _leg_w    = legend_font_weight    if legend_font_weight    is not None else font_weight

        results = self.polyhedron_results_by_type

        cations_data = {k: v for k, v in results.items() if v.ion_category == 'cation'}
        anions_data  = {k: v for k, v in results.items() if v.ion_category == 'anion'}
        all_ion_names = list(cations_data.keys()) + list(anions_data.keys())

        if not all_ion_names:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=_lbl)
            plt.tight_layout()
            plt.show()
            return

        # --- build per-ion colour map ---
        _default_cation_palette = ['#ADD8E6', '#47C2EC', '#1E90FF', '#00BFFF', '#87CEEB']
        _default_anion_palette  = ['#F08080', '#F44646', '#DC143C', '#FF6347', '#FA8072']
        _color_map = {}
        if ion_colors:
            _color_map.update(ion_colors)
        # fill any missing ions with auto colours
        _ci, _ai = 0, 0
        for name in all_ion_names:
            if name not in _color_map:
                if name in cations_data:
                    _color_map[name] = _default_cation_palette[_ci % len(_default_cation_palette)]
                    _ci += 1
                else:
                    _color_map[name] = _default_anion_palette[_ai % len(_default_anion_palette)]
                    _ai += 1

        all_volumes       = ([d.overall_mean_volume for d in cations_data.values()] +
                             [d.overall_mean_volume for d in anions_data.values()])
        all_volume_errors = ([d.overall_std_volume  for d in cations_data.values()] +
                             [d.overall_std_volume  for d in anions_data.values()])
        all_areas         = ([d.overall_mean_area   for d in cations_data.values()] +
                             [d.overall_mean_area   for d in anions_data.values()])
        all_area_errors   = ([d.overall_std_area    for d in cations_data.values()] +
                             [d.overall_std_area    for d in anions_data.values()])

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        x = np.arange(len(all_ion_names))

        for i, ion_name in enumerate(all_ion_names):
            clr = _color_map[ion_name]
            # volume bar
            ax.bar(x[i] - bar_width / 2, all_volumes[i], bar_width,
                   yerr=all_volume_errors[i],
                   color=clr, alpha=bar_alpha, capsize=capsize,
                   edgecolor=edgecolor, linewidth=bar_linewidth,
                   hatch=volume_hatch)
            # area bar
            ax.bar(x[i] + bar_width / 2, all_areas[i], bar_width,
                   yerr=all_area_errors[i],
                   color=clr, alpha=bar_alpha, capsize=capsize,
                   edgecolor=edgecolor, linewidth=bar_linewidth,
                   hatch=area_hatch)

        # value labels
        if show_bar_label:
            _offset = max(all_volumes + all_areas) * 0.01
            for i, (vol, err) in enumerate(zip(all_volumes, all_volume_errors)):
                ax.text(x[i] - bar_width / 2, vol + err + _offset,
                        f'{vol:.1f}', ha='center', va='bottom',
                        fontsize=_blbl, fontweight=_blbl_w)
            for i, (area, err) in enumerate(zip(all_areas, all_area_errors)):
                ax.text(x[i] + bar_width / 2, area + err + _offset,
                        f'{area:.1f}', ha='center', va='bottom',
                        fontsize=_blbl, fontweight=_blbl_w)

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=_lbl, fontweight=_lbl_w)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=_lbl, fontweight=_lbl_w)
        ax.set_xticks(x)
        ax.set_xticklabels(all_ion_names, fontsize=_tick, fontweight=_tick_w)
        ax.tick_params(axis='y', labelsize=_tick)

        # color-code x-tick labels
        for i, lbl in enumerate(ax.get_xticklabels()):
            lbl.set_color('royalblue' if i < len(cations_data) else 'crimson')

        ax.grid(True, alpha=grid_alpha, axis='y')

        max_val = max(max(all_volumes), max(all_areas))
        max_err = max(max(all_volume_errors), max(all_area_errors))
        ax.set_ylim(0, max_val + max_err + max_val * 0.20)

        if show_title:
            ax.set_title('Polyhedron Size Comparison by Ion Type',
                         fontsize=_title_fs, fontweight=title_font_weight)

        _leg_prop = {'size': _leg, 'weight': _leg_w}

        # --- Legend 1: ion colours ---
        if show_ion_legend:
            import matplotlib.patches as mpatches
            _ion_handles = [
                mpatches.Patch(facecolor=_color_map[n], edgecolor=edgecolor,
                               linewidth=bar_linewidth, label=n)
                for n in all_ion_names
            ]
            _kw1 = dict(handles=_ion_handles, prop=_leg_prop,
                        title=ion_legend_title, loc=ion_legend_loc)
            if ion_legend_bbox_to_anchor is not None:
                _kw1['bbox_to_anchor'] = ion_legend_bbox_to_anchor
            leg1 = ax.legend(**_kw1)
            if legend_frame_alpha is not None:
                leg1.get_frame().set_alpha(legend_frame_alpha)
            ax.add_artist(leg1)

        # --- Legend 2: hatch → quantity ---
        if show_hatch_legend:
            import matplotlib.patches as mpatches
            _hatch_handles = [
                mpatches.Patch(facecolor='white', edgecolor=edgecolor,
                               hatch=volume_hatch,
                               label=f'Volume (\u00c5\u00b3)'),
                mpatches.Patch(facecolor='white', edgecolor=edgecolor,
                               hatch=area_hatch,
                               label=f'Area (\u00c5\u00b2)'),
            ]
            _kw2 = dict(handles=_hatch_handles, prop=_leg_prop,
                        title=hatch_legend_title, loc=hatch_legend_loc)
            if hatch_legend_bbox_to_anchor is not None:
                _kw2['bbox_to_anchor'] = hatch_legend_bbox_to_anchor
            leg2 = ax.legend(**_kw2)
            if legend_frame_alpha is not None:
                leg2.get_frame().set_alpha(legend_frame_alpha)

        plt.tight_layout()

        if save_plots:
            _fname = output_filename or 'polyhedron_size_comparison_by_type.png'
            plt.savefig(_fname, dpi=dpi, bbox_inches='tight', transparent=transparent)
            print(f"Comparison plot saved as: {_fname}")

        plt.show()



    def find_representative_frames_by_shell_by_type(self, ion_type, max_shells=5, max_frames_to_scan=100):
        '''
        Find representative frame numbers for different shell types (coordination environments) for a specific ion type.
        FIXED: Now properly handles both 'Shell_1' format and 'coion-water' format
        
        Parameters
        ----------
        ion_type : str
            Specific ion type to analyze (e.g., 'Na', 'Mg', 'Cl')
        max_shells : int
            Maximum number of shell types to find, default=5
        max_frames_to_scan : int
            Maximum number of frames to scan when looking for representative ions, default=100
        
        Returns
        -------
        representative_frames : dict
            Dictionary with shell types as keys and frame numbers as values
            Format: {'shell_type': frame_number}
        '''
        
        # FIXED: Check which shell probability data we have
        has_region_probs = hasattr(self, 'shell_region_coordination_probabilities') and ion_type in self.shell_region_coordination_probabilities
        has_ion_probs = hasattr(self, 'shell_probabilities_by_ion_type') and ion_type in self.shell_probabilities_by_ion_type
        
        if not has_region_probs and not has_ion_probs:
            print('Ion-type-specific shell probabilities not calculated.')
            print('Run shell_coordination_probabilities_by_shell_region_by_type() or shell_coordination_probabilities_by_type() first')
            return None
        
        # FIXED: Determine which data source to use and get shell data
        if has_region_probs:
            # Use shell_region_coordination_probabilities (preferred - has shell_1, shell_2, etc.)
            region_data = self.shell_region_coordination_probabilities[ion_type]
            shell_regions = region_data['shell_regions']
            
            # Get most probable coordination environments from shell_1 (first coordination shell)
            if 'shell_1' in shell_regions:
                shell_1_data = shell_regions['shell_1']
                coordination_envs = shell_1_data['coordination_environments']
                
                # Get top coordination environments by probability
                top_envs = sorted(coordination_envs.items(), key=lambda x: x[1], reverse=True)[:max_shells]
                top_shells = [env[0] for env in top_envs]
                
                print(f"Searching for representative frames for top {len(top_shells)} {ion_type} coordination environments:")
                for i, (shell_type, prob) in enumerate(top_envs):
                    print(f"  {i+1}. Shell {shell_type}: {prob:.1%} probability")
            else:
                print(f"No shell_1 data found for {ion_type}")
                return None
            
            ion_category = region_data['ion_category']
            use_region_search = True
            
        elif has_ion_probs:
            # Use shell_probabilities_by_ion_type (from Solute speciation)
            shell_data = self.shell_probabilities_by_ion_type[ion_type]['data']
            ion_category = self.shell_probabilities_by_ion_type[ion_type]['category']
            
            # Get the most common shell types
            top_shells = shell_data.nlargest(max_shells, 'fraction')['shell'].values
            print(f"Searching for representative frames for top {len(top_shells)} {ion_type} shell types:")
            for i, shell in enumerate(top_shells):
                fraction = shell_data[shell_data['shell'] == shell]['fraction'].iloc[0]
                print(f"  {i+1}. Shell {shell}: {fraction:.1%} probability")
            
            use_region_search = False
        
        # Get ion group and coordination radius
        if ion_category == 'cation':
            cation_types = self._get_unique_ion_types(self.cations)
            if ion_type in cation_types:
                ions = cation_types[ion_type]
            else:
                print(f"Ion type '{ion_type}' not found in cation types")
                return None
            
            # Get coordination radius
            if hasattr(self, 'solutes_ci') and ion_type in self.solutes_ci:
                r0 = self.solutes_ci[ion_type].radii['water']
            else:
                print(f"No coordination radius found for {ion_type}")
                return None
        else:  # anion
            anion_types = self._get_unique_ion_types(self.anions)
            if ion_type in anion_types:
                ions = anion_types[ion_type]
            else:
                print(f"Ion type '{ion_type}' not found in anion types")
                return None
            
            # Get coordination radius
            if hasattr(self, 'solutes_ai') and ion_type in self.solutes_ai:
                r0 = self.solutes_ai[ion_type].radii['water']
            else:
                print(f"No coordination radius found for {ion_type}")
                return None
        
        # Dictionary to store shell type -> frame number mapping
        representative_frames = {}
        
        # Determine how many frames to scan (don't exceed trajectory length)
        frames_to_scan = min(max_frames_to_scan, len(self.universe.trajectory))
        frame_indices = np.linspace(0, len(self.universe.trajectory)-1, frames_to_scan, dtype=int)
        
        print(f"\nScanning {frames_to_scan} frames to find representative ions...")
        
        # Scan frames to find representative ions for each shell type
        for frame_idx in tqdm(frame_indices, desc="Scanning frames"):
            self.universe.trajectory[frame_idx]
            
            for shell_type in top_shells:
                # Skip if we already found this shell type
                if shell_type in representative_frames:
                    continue
                
                # Parse shell type (e.g., '0-6' means 0 counter-ions, 6 waters)
                try:
                    coion_count, water_count = map(int, shell_type.split('-'))
                except ValueError:
                    # FIXED: Skip shell types that can't be parsed (like 'Shell_1')
                    # This is expected when using shell_probabilities_by_ion_type which doesn't
                    # provide the detailed coion-water breakdown
                    continue
                
                # Look for an ion with this shell configuration
                for ion_atom in ions:
                    # Get coordinated species
                    shell_atoms = self.universe.select_atoms(f'sphzone {r0} index {ion_atom.index}')
                    
                    if ion_category == 'cation':
                        coordinated_waters = (shell_atoms & self.waters)
                        coordinated_coions = (shell_atoms & self.anions)
                    else:
                        coordinated_waters = (shell_atoms & self.waters)
                        coordinated_coions = (shell_atoms & self.cations)
                    
                    if len(coordinated_waters) == water_count and len(coordinated_coions) == coion_count:
                        representative_frames[shell_type] = frame_idx
                        print(f"  Found shell type {shell_type} in frame {frame_idx}")
                        break
            
            # Break early if we found all shell types
            if len(representative_frames) == len(top_shells):
                print(f"Found all {len(top_shells)} shell types!")
                break
        
        # Report results
        found_shells = list(representative_frames.keys())
        missing_shells = [shell for shell in top_shells if shell not in found_shells]
        
        # Print results
        print(f"\n{'='*50}")
        print(f"REPRESENTATIVE FRAMES FOR {ion_type.upper()} SHELL TYPES")
        print(f"{'='*50}")
        
        if use_region_search and has_region_probs:
            # Use shell_region_coordination_probabilities (preferred - has shell_1, shell_2, etc.)
            region_data = self.shell_region_coordination_probabilities[ion_type]
            shell_regions = region_data['shell_regions']
            
            if 'shell_1' in shell_regions:
                coordination_envs = shell_regions['shell_1']['coordination_environments']
                
                for shell_type in found_shells:
                    frame_num = representative_frames[shell_type]
                    prob = coordination_envs[shell_type]
                    # FIXED: prob is already 0-1, multiply by 100 manually and use .1f
                    print(f"Shell {shell_type:>4s}: Frame {frame_num:>5d} ({prob*100:.1f}% probability)")
        else:
            # Use shell_probabilities_by_ion_type
            shell_data = self.shell_probabilities_by_ion_type[ion_type]['data']
            
            for shell_type in found_shells:
                frame_num = representative_frames[shell_type]
                fraction = shell_data[shell_data['shell'] == shell_type]['fraction'].iloc[0]
                # FIXED: fraction is already 0-1, multiply by 100 manually
                print(f"Shell {shell_type:>4s}: Frame {frame_num:>5d} ({fraction*100:.1f}% probability)")
        
        if missing_shells:
            print(f"\nWarning: Could not find representative ions for shell types: {missing_shells}")
            print("These shell types may be rare or require scanning more frames.")
            print("Note: Shell types without detailed coion-water breakdown cannot be searched.")
        
        print(f"\nSummary: Found {len(found_shells)}/{len(top_shells)} shell types")
        
        return representative_frames



    def find_representative_frames_for_all_ion_types(self, max_shells=5, max_frames_to_scan=100):
        '''
        Find representative frames for all ion types.
        FIXED: Corrected probability display (was showing as percentage of percentage)
        
        Parameters
        ----------
        max_shells : int
            Maximum number of shell types to find per ion type, default=5
        max_frames_to_scan : int
            Maximum number of frames to scan, default=100
        
        Returns
        -------
        all_representatives : dict
            Nested dictionary: {ion_type: {shell_type: frame_number}}
        '''
        
        if not hasattr(self, 'shell_probabilities_by_ion_type'):
            print('Ion-type-specific shell probabilities not calculated.')
            print('Run shell_coordination_probabilities_by_type() first')
            return None
        
        all_representatives = {}
        available_types = list(self.shell_probabilities_by_ion_type.keys())
        
        print(f"Finding representative frames for {len(available_types)} ion types...")
        
        for ion_type in available_types:
            print(f"\n{'='*60}")
            print(f"Processing {ion_type}")
            print(f"{'='*60}")
            
            representatives = self.find_representative_frames_by_shell_by_type(
                ion_type, max_shells, max_frames_to_scan
            )
            
            if representatives:
                all_representatives[ion_type] = representatives
        
        # Print overall summary
        print(f"\n{'='*70}")
        print(f"OVERALL SUMMARY - REPRESENTATIVE FRAMES BY ION TYPE")
        print(f"{'='*70}")
        
        for ion_type, representatives in all_representatives.items():
            ion_category = self.shell_probabilities_by_ion_type[ion_type]['category']
            print(f"\n{ion_type.upper()} ({ion_category}):")
            
            # FIXED: Determine which data source to use for probabilities
            has_region_probs = (hasattr(self, 'shell_region_coordination_probabilities') and 
                            ion_type in self.shell_region_coordination_probabilities)
            has_ion_probs = (hasattr(self, 'shell_probabilities_by_ion_type') and 
                            ion_type in self.shell_probabilities_by_ion_type)
            
            if has_region_probs:
                # Use shell_region_coordination_probabilities - has detailed coion-water format
                region_data = self.shell_region_coordination_probabilities[ion_type]
                shell_regions = region_data['shell_regions']
                
                if 'shell_1' in shell_regions:
                    coordination_envs = shell_regions['shell_1']['coordination_environments']
                    
                    for shell_type, frame_num in representatives.items():
                        if shell_type in coordination_envs:
                            prob = coordination_envs[shell_type]
                            # CRITICAL FIX: prob is already a fraction (0-1), so just multiply by 100
                            # Don't use :.1% format which would multiply by 100 again
                            print(f"  {shell_type}: Frame {frame_num} ({prob*100:.1f}% probability)")
                        else:
                            print(f"  {shell_type}: Frame {frame_num} (probability data not found in shell_1)")
                else:
                    print(f"  No shell_1 data available - cannot show probabilities")
                    for shell_type, frame_num in representatives.items():
                        print(f"  {shell_type}: Frame {frame_num}")
            
            elif has_ion_probs:
                # Use shell_probabilities_by_ion_type - has Shell_1, Shell_2 format
                shell_data = self.shell_probabilities_by_ion_type[ion_type]['data']
                
                for shell_type, frame_num in representatives.items():
                    matching_rows = shell_data[shell_data['shell'] == shell_type]
                    
                    if len(matching_rows) > 0:
                        fraction = matching_rows['fraction'].iloc[0]
                        # FIXED: fraction is already 0-1, so multiply by 100 manually
                        print(f"  {shell_type}: Frame {frame_num} ({fraction*100:.1f}% probability)")
                    else:
                        print(f"  {shell_type}: Frame {frame_num} (probability data not available)")
            else:
                # No probability data available
                print(f"  No probability data source available")
                for shell_type, frame_num in representatives.items():
                    print(f"  {shell_type}: Frame {frame_num}")
        
        print(f"{'='*70}")
        
        return all_representatives


    def get_ion_configuration_at_frame(self, ion_type, frame_idx, ion_index=0):
        '''
        Get detailed coordination information for a specific ion at a specific frame.
        
        Parameters
        ----------
        ion_type : str
            Ion type (e.g., 'Na', 'Mg', 'Cl')
        frame_idx : int
            Frame number to analyze
        ion_index : int
            Index of the ion within the ion type group, default=0 (first ion)
        
        Returns
        -------
        config_info : dict
            Detailed coordination configuration information
        '''
        
        # Get ion group and coordination radius
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        if ion_type in cation_types:
            ions = cation_types[ion_type]
            ion_category = 'cation'
            if hasattr(self, 'solutes_ci') and ion_type in self.solutes_ci:
                r0 = self.solutes_ci[ion_type].radii['water'] if self.solutes_ci[ion_type] else 2.8
            else:
                r0 = 2.8
        elif ion_type in anion_types:
            ions = anion_types[ion_type]
            ion_category = 'anion'
            if hasattr(self, 'solutes_ai') and ion_type in self.solutes_ai:
                r0 = self.solutes_ai[ion_type].radii['water'] if self.solutes_ai[ion_type] else 3.5
            else:
                r0 = 3.5
        else:
            print(f"Ion type '{ion_type}' not found")
            return None
        
        if ion_index >= len(ions):
            print(f"Ion index {ion_index} out of range for {ion_type} (has {len(ions)} ions)")
            return None
        
        # Set trajectory to the specified frame
        self.universe.trajectory[frame_idx]
        
        # Get the specific ion
        target_ion = ions[ion_index]
        
        # Get coordinated species
        shell_atoms = self.universe.select_atoms(f'sphzone {r0} index {target_ion.index}')
        
        coordinated_waters = (shell_atoms & self.waters)
        if ion_category == 'cation':
            coordinated_coions = (shell_atoms & self.anions)
            coion_name = 'anions'
        else:
            coordinated_coions = (shell_atoms & self.cations)
            coion_name = 'cations'
        
        # Calculate distances
        water_distances = []
        for water in coordinated_waters:
            dist = np.linalg.norm(target_ion.position - water.position)
            water_distances.append(dist)
        
        coion_distances = []
        for coion in coordinated_coions:
            dist = np.linalg.norm(target_ion.position - coion.position)
            coion_distances.append(dist)
        
        # Create shell type string
        shell_type = f"{len(coordinated_coions)}-{len(coordinated_waters)}"
        
        config_info = {
            'ion_type': ion_type,
            'ion_category': ion_category,
            'ion_index': ion_index,
            'frame': frame_idx,
            'position': target_ion.position.copy(),
            'coordination_radius': r0,
            'shell_type': shell_type,
            'n_waters': len(coordinated_waters),
            'n_coions': len(coordinated_coions),
            'coion_type': coion_name,
            'water_distances': np.array(water_distances),
            'coion_distances': np.array(coion_distances),
            'mean_water_distance': np.mean(water_distances) if water_distances else 0,
            'mean_coion_distance': np.mean(coion_distances) if coion_distances else 0,
            'coordinated_water_indices': [w.index for w in coordinated_waters],
            'coordinated_coion_indices': [c.index for c in coordinated_coions]
        }
        
        return config_info

    def print_ion_configuration(self, config_info):
        '''Print detailed information about an ion's coordination configuration'''
        
        if config_info is None:
            return
        
        print(f"\n{'='*60}")
        print(f"ION COORDINATION CONFIGURATION")
        print(f"{'='*60}")
        print(f"Ion Type: {config_info['ion_type']} ({config_info['ion_category']})")
        print(f"Ion Index: {config_info['ion_index']}")
        print(f"Frame: {config_info['frame']}")
        print(f"Position: ({config_info['position'][0]:.2f}, {config_info['position'][1]:.2f}, {config_info['position'][2]:.2f})")
        print(f"Coordination Radius: {config_info['coordination_radius']:.2f} Å")
        print(f"Shell Type: {config_info['shell_type']}")
        print(f"\nCoordination Details:")
        print(f"  Waters: {config_info['n_waters']} (mean distance: {config_info['mean_water_distance']:.2f} Å)")
        print(f"  {config_info['coion_type'].title()}: {config_info['n_coions']} (mean distance: {config_info['mean_coion_distance']:.2f} Å)")
        
        if len(config_info['water_distances']) > 0:
            print(f"\nWater Distances: {', '.join([f'{d:.2f}' for d in config_info['water_distances']])} Å")
        
        if len(config_info['coion_distances']) > 0:
            print(f"{config_info['coion_type'].title()} Distances: {', '.join([f'{d:.2f}' for d in config_info['coion_distances']])} Å")
        
        print(f"{'='*60}")

    def plot_all_ion_pairing_after_modifications(self, save_plots=True, plot_range=12, figsize=(16, 10)):
        '''
        Plot ion pairing analysis results for all ion types after modifications in a grid layout.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save the combined plot, default=True
        plot_range : float
            Maximum r value for plotting, default=12
        figsize : tuple
            Overall figure size, default=(16, 10)
        
        Returns
        -------
        success : bool
            True if plotting was successful
        '''
        
        # Check if we have ion pairing data
        if not hasattr(self, 'ion_pairs_by_type') or not self.ion_pairs_by_type:
            print("No ion pairing data available.")
            print("Run determine_ion_pairing_cutoffs() for different ion types first.")
            return False
        
        # Check if we have RDFs
        if not hasattr(self, 'rdfs') or not self.rdfs:
            print("No RDFs available. Run generate_rdfs() first.")
            return False
        
        # Collect all ions with valid pairing data and RDFs
        ions_to_plot = []
        
        for ion_type, pairing_data in self.ion_pairs_by_type.items():
            ion_pairs = pairing_data['ion_pairs']
            rdf_key = pairing_data['rdf_key']
            ion_category = pairing_data['ion_category']
            
            # Check if we have the corresponding RDF
            if rdf_key in self.rdfs and self.rdfs[rdf_key] is not None:
                ions_to_plot.append((ion_type, ion_pairs, rdf_key, ion_category, self.rdfs[rdf_key]))
            else:
                print(f"Warning: No RDF data for {ion_type} ({rdf_key})")
        
        if not ions_to_plot:
            print("No ions available for plotting.")
            return False
        
        print(f"Plotting ion pairing results for {len(ions_to_plot)} ion types:")
        for ion_type, _, rdf_key, ion_category, _ in ions_to_plot:
            print(f"  {ion_type} ({ion_category}) - {rdf_key}")
        
        # Calculate grid dimensions
        n_ions = len(ions_to_plot)
        n_cols = min(3, n_ions)  # Max 3 columns
        n_rows = (n_ions + n_cols - 1) // n_cols
        
        # Calculate figure size
        subplot_width = figsize[0] / n_cols
        subplot_height = figsize[1] / n_rows
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle single subplot case
        if n_ions == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each ion
        for i, (ion_type, ion_pairs, rdf_key, ion_category, rdf_data) in enumerate(ions_to_plot):
            ax = axes[i]
            
            # Get RDF data
            r = rdf_data.bins
            rdf = rdf_data.rdf
            
            # Plot RDF
            ax.plot(r, rdf, color='k', linewidth=2, label='g(r)')
            
            # Calculate dynamic y-limits based on data
            y_min = 0
            y_max = np.max(rdf) * 1.1
            text_y_pos = y_max * 0.95
            
            # Plot regions with current boundaries
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
            pair_types = ['CIP', 'SIP', 'DSIP', 'FI']
            
            le = max(2, r.min())
            for j, region in enumerate(pair_types):
                if region in ion_pairs:
                    start, end = ion_pairs[region]
                    end_plot = min(end, plot_range) if not np.isinf(end) else plot_range
                    
                    ax.fill_betweenx(np.linspace(y_min, y_max), max(le, start), end_plot, 
                                alpha=0.4, color=colors[j % len(colors)], label=region)
                    ax.text((max(le, start) + end_plot) / 2, text_y_pos, region, ha='center', 
                        fontweight='bold', fontsize=10)
                    le = end_plot
            
            # Format subplot
            ax.set_xlabel(r'r ($\mathrm{\AA}$)', fontsize=10)
            ax.set_ylabel('g(r)', fontsize=10)
            
            # Create title based on ion type and category
            if ion_type in ['cation', 'anion']:
                title_text = f'{ion_type.title()}-{("Anion" if ion_type == "cation" else "Cation")} Pairing'
            else:
                title_text = f'{ion_type} Ion Pairing'
            
            # Color-code the title by ion category
            title_color = 'blue' if ion_category == 'cation' else 'red'
            ax.set_title(title_text, fontweight='bold', fontsize=11, color=title_color)
            
            ax.set_xlim(2, plot_range)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3)
            
            # Print pairing info for this ion
            print(f"\n{ion_type} ({ion_category}) ion pairing regions:")
            for region in pair_types:
                if region in ion_pairs:
                    start, end = ion_pairs[region]
                    end_str = "∞" if np.isinf(end) else f"{end:.2f}"
                    width = end - start if not np.isinf(end) else "∞"
                    width_str = f"{width:.2f}" if width != "∞" else "∞"
                    
                    # Full names for clarity
                    full_names = {
                        'CIP': 'Contact Ion Pair',
                        'SIP': 'Solvent-separated',
                        'DSIP': 'Double Solvent-separated',
                        'FI': 'Free Ions'
                    }
                    full_name = full_names.get(region, region)
                    print(f"  {region} ({full_name}): {start:.2f} - {end_str} Å (width: {width_str} Å)")
        
        # Hide unused subplots
        for i in range(len(ions_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        plt.suptitle('Ion Pairing Analysis - All Ion Types (After Modifications)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle
        
        # Save plot
        if save_plots:
            filename = 'all_ion_pairing_after_modifications.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\nCombined plot saved as: {filename}")
        
        plt.show()
        
        return True

    def plot_ion_pairing_comparison_by_category(self, save_plots=True, plot_range=12, figsize=(14, 6)):
        '''
        Plot ion pairing results separated by cation/anion categories after modifications.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save plots, default=True
        plot_range : float
            Maximum r value for plotting, default=12
        figsize : tuple
            Figure size, default=(14, 6)
        '''
        
        if not hasattr(self, 'ion_pairs_by_type') or not self.ion_pairs_by_type:
            print("No ion pairing data available.")
            return False
        
        # Separate ions by category
        cations_data = []
        anions_data = []
        
        for ion_type, pairing_data in self.ion_pairs_by_type.items():
            ion_pairs = pairing_data['ion_pairs']
            rdf_key = pairing_data['rdf_key']
            ion_category = pairing_data['ion_category']
            
            if rdf_key in self.rdfs and self.rdfs[rdf_key] is not None:
                rdf_data = self.rdfs[rdf_key]
                
                if ion_category == 'cation':
                    cations_data.append((ion_type, ion_pairs, rdf_data))
                else:
                    anions_data.append((ion_type, ion_pairs, rdf_data))
        
        if not cations_data and not anions_data:
            print("No valid ion pairing data found.")
            return False
        
        # Create figure with subplots
        n_plots = (1 if cations_data else 0) + (1 if anions_data else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot cations
        if cations_data:
            ax = axes[plot_idx]
            
            print(f"Plotting {len(cations_data)} cation types:")
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cations_data)))
            
            for i, (ion_type, ion_pairs, rdf_data) in enumerate(cations_data):
                print(f"  {ion_type}")
                
                r = rdf_data.bins
                rdf = rdf_data.rdf
                
                # Plot RDF with different colors for each ion
                ax.plot(r, rdf, color=colors[i], linewidth=2, label=f'{ion_type.upper()}')
            
            ax.set_xlabel(r'r ($\mathrm{\AA}$)', fontsize=12)
            ax.set_ylabel('g(r)', fontsize=12)
            ax.set_title('Cation Ion Pairing RDFs', fontweight='bold', fontsize=14, color='blue')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(2, plot_range)
            
            plot_idx += 1
        
        # Plot anions
        if anions_data:
            ax = axes[plot_idx]
            
            print(f"Plotting {len(anions_data)} anion types:")
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(anions_data)))
            
            for i, (ion_type, ion_pairs, rdf_data) in enumerate(anions_data):
                print(f"  {ion_type}")
                
                r = rdf_data.bins
                rdf = rdf_data.rdf
                
                # Plot RDF with different colors for each ion
                ax.plot(r, rdf, color=colors[i], linewidth=2, label=f'{ion_type.upper()}')
            
            ax.set_xlabel(r'r ($\mathrm{\AA}$)', fontsize=12)
            ax.set_ylabel('g(r)', fontsize=12)
            ax.set_title('Anion Ion Pairing RDFs', fontweight='bold', fontsize=14, color='red')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(2, plot_range)
        
        plt.suptitle('Ion Pairing RDFs by Category (After Modifications)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            filename = 'ion_pairing_by_category_after_modifications.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Category plot saved as: {filename}")
        
        plt.show()
        return True

    def print_all_ion_pairing_summary(self):
        '''
        Print a comprehensive summary of all ion pairing cutoffs after modifications.
        '''
        
        if not hasattr(self, 'ion_pairs_by_type') or not self.ion_pairs_by_type:
            print("No ion pairing data available.")
            return
        
        print("\n" + "="*80)
        print("ION PAIRING CUTOFFS SUMMARY (AFTER MODIFICATIONS)")
        print("="*80)
        
        # Separate by category
        cations_data = {}
        anions_data = {}
        
        for ion_type, pairing_data in self.ion_pairs_by_type.items():
            ion_category = pairing_data['ion_category']
            if ion_category == 'cation':
                cations_data[ion_type] = pairing_data
            else:
                anions_data[ion_type] = pairing_data
        
        # Print cations
        if cations_data:
            print("\nCATIONS:")
            print("-" * 60)
            for ion_type, pairing_data in cations_data.items():
                ion_pairs = pairing_data['ion_pairs']
                rdf_key = pairing_data['rdf_key']
                
                print(f"\n{ion_type.upper()} (RDF: {rdf_key}):")
                
                # Order regions logically
                region_order = ['CIP', 'SIP', 'DSIP', 'FI']
                
                for region in region_order:
                    if region in ion_pairs:
                        start, end = ion_pairs[region]
                        end_str = "∞" if np.isinf(end) else f"{end:.2f}"
                        width = end - start if not np.isinf(end) else "∞"
                        width_str = f"{width:.2f}" if width != "∞" else "∞"
                        
                        # Full names for clarity
                        full_names = {
                            'CIP': 'Contact Ion Pair',
                            'SIP': 'Solvent-separated',
                            'DSIP': 'Double Solvent-separated',
                            'FI': 'Free Ions'
                        }
                        full_name = full_names.get(region, region)
                        print(f"  {region:4s} ({full_name:20s}): {start:6.2f} - {end_str:>6s} Å  (width: {width_str:>6s} Å)")
        
        # Print anions
        if anions_data:
            print("\nANIONS:")
            print("-" * 60)
            for ion_type, pairing_data in anions_data.items():
                ion_pairs = pairing_data['ion_pairs']
                rdf_key = pairing_data['rdf_key']
                
                print(f"\n{ion_type.upper()} (RDF: {rdf_key}):")
                
                # Order regions logically
                region_order = ['CIP', 'SIP', 'DSIP', 'FI']
                
                for region in region_order:
                    if region in ion_pairs:
                        start, end = ion_pairs[region]
                        end_str = "∞" if np.isinf(end) else f"{end:.2f}"
                        width = end - start if not np.isinf(end) else "∞"
                        width_str = f"{width:.2f}" if width != "∞" else "∞"
                        
                        # Full names for clarity
                        full_names = {
                            'CIP': 'Contact Ion Pair',
                            'SIP': 'Solvent-separated',
                            'DSIP': 'Double Solvent-separated',
                            'FI': 'Free Ions'
                        }
                        full_name = full_names.get(region, region)
                        print(f"  {region:4s} ({full_name:20s}): {start:6.2f} - {end_str:>6s} Å  (width: {width_str:>6s} Å)")
        
        print("="*80)
        print(f"Total ion types analyzed: {len(self.ion_pairs_by_type)}")
        print("="*80)


    def water_dipole_distribution(self, ion_type='cation', radius=None, step=None):
        '''
        Optimized water dipole distribution calculation with ion-specific support.
        Now supports both broad categories ('cation', 'anion') and specific ion types ('Na', 'Mg', 'Cl').
        
        Parameters
        ----------
        ion_type : str
            Ion type to analyze. Can be:
            - Broad categories: 'cation', 'anion' 
            - Specific ion types: 'Na', 'Mg', 'K', 'Cl', 'Br', etc.
        radius : float, optional
            Coordination radius. If None, uses the radius from the ion-type-specific solute.
        step : int, optional
            Step size for trajectory analysis
        
        Returns
        -------
        angles : np.array
            Array of dipole angles in degrees
        '''
        
        if step is None:
            step = self.default_step
        
        # Handle both broad categories and specific ion types
        def get_ions_and_radius(eq_analysis, ion_type, radius):
            """Get ions and coordination radius based on ion_type"""
            
            # Handle broad categories (original behavior)
            if ion_type in ['cation', 'anion']:
                if ion_type == 'cation':
                    ions = eq_analysis.cations
                    if radius is None:
                        try:
                            radius = eq_analysis.solute_ci.radii['water']
                        except (AttributeError, NameError):
                            print('Cation solutes not initialized. Using default radius 2.8 Å')
                            radius = 2.8
                else:  # anion
                    ions = eq_analysis.anions
                    if radius is None:
                        try:
                            radius = eq_analysis.solute_ai.radii['water']
                        except (AttributeError, NameError):
                            print('Anion solutes not initialized. Using default radius 3.5 Å')
                            radius = 3.5
                
                return ions, radius, ion_type
            
            # Handle specific ion types (new behavior)
            else:
                # Check if optimized class with ion-type-specific methods
                if hasattr(eq_analysis, '_get_unique_ion_types'):
                    cation_types = eq_analysis._get_unique_ion_types(eq_analysis.cations)
                    anion_types = eq_analysis._get_unique_ion_types(eq_analysis.anions)
                    
                    if ion_type in cation_types:
                        ions = cation_types[ion_type]
                        ion_category = 'cation'
                        
                        # Get ion-specific radius
                        if radius is None:
                            if (hasattr(eq_analysis, 'solutes_ci') and 
                                ion_type in eq_analysis.solutes_ci and 
                                eq_analysis.solutes_ci[ion_type] is not None):
                                try:
                                    radius = eq_analysis.solutes_ci[ion_type].radii['water']
                                    print(f"Using {ion_type}-specific coordination radius: {radius:.2f} Å")
                                except (AttributeError, KeyError):
                                    print(f'No radius found for {ion_type}, using default 2.8 Å')
                                    radius = 2.8
                            else:
                                print(f'No solute found for {ion_type}, using default 2.8 Å')
                                radius = 2.8
                        
                        return ions, radius, ion_category
                    
                    elif ion_type in anion_types:
                        ions = anion_types[ion_type]
                        ion_category = 'anion'
                        
                        # Get ion-specific radius
                        if radius is None:
                            if (hasattr(eq_analysis, 'solutes_ai') and 
                                ion_type in eq_analysis.solutes_ai and 
                                eq_analysis.solutes_ai[ion_type] is not None):
                                try:
                                    radius = eq_analysis.solutes_ai[ion_type].radii['water']
                                    print(f"Using {ion_type}-specific coordination radius: {radius:.2f} Å")
                                except (AttributeError, KeyError):
                                    print(f'No radius found for {ion_type}, using default 3.5 Å')
                                    radius = 3.5
                            else:
                                print(f'No solute found for {ion_type}, using default 3.5 Å')
                                radius = 3.5
                        
                        return ions, radius, ion_category
                    
                    else:
                        available_types = list(cation_types.keys()) + list(anion_types.keys())
                        raise ValueError(f"Ion type '{ion_type}' not found. Available types: {available_types}")
                else:
                    # Fallback for original class - filter by element/name
                    all_cations = eq_analysis.cations
                    all_anions = eq_analysis.anions
                    
                    # Try to filter by element first, then by name
                    cation_matches = []
                    anion_matches = []
                    
                    for atom in all_cations:
                        if (hasattr(atom, 'element') and atom.element == ion_type) or \
                        (hasattr(atom, 'name') and atom.name.startswith(ion_type)):
                            cation_matches.append(atom)
                    
                    for atom in all_anions:
                        if (hasattr(atom, 'element') and atom.element == ion_type) or \
                        (hasattr(atom, 'name') and atom.name.startswith(ion_type)):
                            anion_matches.append(atom)
                    
                    if cation_matches:
                        # Create AtomGroup for cation matches
                        ions = eq_analysis.universe.atoms[np.array([atom.index for atom in cation_matches])]
                        ion_category = 'cation'
                        if radius is None:
                            radius = 2.8  # Default
                        return ions, radius, ion_category
                    elif anion_matches:
                        # Create AtomGroup for anion matches
                        ions = eq_analysis.universe.atoms[np.array([atom.index for atom in anion_matches])]
                        ion_category = 'anion'
                        if radius is None:
                            radius = 3.5  # Default
                        return ions, radius, ion_category
                    else:
                        raise ValueError(f"No atoms found matching ion type '{ion_type}'")
        
        # Get ions, radius, and category
        ions, coordination_radius, ion_category = get_ions_and_radius(self, ion_type, radius)
        
        print(f"Calculating water dipole distribution for {len(ions)} {ion_type} ions ({ion_category})")
        print(f"Using coordination radius: {coordination_radius:.2f} Å, step: {step}")
        
        # Pre-allocate list for angles
        angles = []
        
        # Handle debug trajectory
        if hasattr(self, '_debug_frame_indices') and self._debug_frame_indices is not None:
            frame_indices = self._debug_frame_indices[::step]
            print(f"Debug mode: analyzing {len(frame_indices)} frames")
        else:
            frame_indices = range(0, len(self.universe.trajectory), step)
        
        # Loop through frames with optimized approach
        for frame_idx in tqdm(frame_indices, desc=f"Dipole calculation for {ion_type}"):
            # Set trajectory to frame
            if hasattr(self, '_debug_frame_indices') and self._debug_frame_indices is not None:
                ts = self.universe.trajectory[frame_idx]
            else:
                ts = self.universe.trajectory[frame_idx]
            
            for ion in ions:
                # Use more efficient atom selection
                shell_atoms = self.universe.select_atoms(f'sphzone {coordination_radius} index {ion.index}') - ion
                shell_waters = shell_atoms & self.waters
                
                if len(shell_waters) == 0:
                    continue
                
                # Vectorized calculations where possible
                for ow in shell_waters:
                    
                    dist = ion.position - ow.position
                    
                    # Optimized PBC handling
                    for d in range(3):
                        if dist[d] >= ts.dimensions[d]/2:
                            ow.residue.atoms.positions[:, d] += ts.dimensions[d]
                        elif dist[d] <= -ts.dimensions[d]/2:
                            ow.residue.atoms.positions[:, d] -= ts.dimensions[d]
                    
                    # Calculate angles more efficiently
                    pos = ow.position
                    bonded_Hs = ow.bonded_atoms
                    
                    if len(bonded_Hs) >= 2:  # Ensure we have hydrogen atoms
                        tmp_pt = bonded_Hs.positions.mean(axis=0)
                        
                        v1 = ion.position - pos
                        v2 = pos - tmp_pt
                        ang = get_angle(v1, v2) * 180 / np.pi
                        angles.append(ang)
        
        angles_array = np.array(angles)
        
        # Print summary statistics
        if len(angles_array) > 0:
            print(f"Analysis complete for {ion_type}:")
            print(f"  Total angles calculated: {len(angles_array)}")
            print(f"  Mean angle: {angles_array.mean():.1f}°")
            print(f"  Std deviation: {angles_array.std():.1f}°")
            print(f"  Range: {angles_array.min():.1f}° - {angles_array.max():.1f}°")
        else:
            print(f"Warning: No dipole angles calculated for {ion_type}")
        
        return angles_array


    def plot_water_dipole_distribution_by_type(self, ion_types=None, save_plots=True, bins=50, plot_range=(0, 180), use_cache=True):
        '''
        Plot water dipole distributions for different ion types with caching.
        '''
        
        # Initialize cache if it doesn't exist
        if not hasattr(self, 'dipole_distributions_by_type'):
            self.dipole_distributions_by_type = {}
        
        # Determine which ion types to analyze
        if ion_types is None:
            cation_types = list(self._get_unique_ion_types(self.cations).keys())
            anion_types = list(self._get_unique_ion_types(self.anions).keys())
            ion_types = cation_types + anion_types
        
        if not ion_types:
            print("No ion types available for dipole analysis")
            return
        
        print(f"Calculating dipole distributions for: {ion_types}")
        
        # Calculate dipole distributions for each ion type (with caching)
        dipole_data = {}
        for ion_type in ion_types:
            # Check if we already have cached data
            if use_cache and ion_type in self.dipole_distributions_by_type:
                print(f"Using cached dipole data for {ion_type}")
                angles = self.dipole_distributions_by_type[ion_type]
            else:
                print(f"Calculating dipoles for {ion_type}...")
                angles = self.water_dipole_distribution(ion_type=ion_type)
                # Store in cache
                self.dipole_distributions_by_type[ion_type] = angles
            
            if len(angles) > 0:
                dipole_data[ion_type] = angles
            else:
                print(f"  Warning: No dipole data for {ion_type}")
        
        if not dipole_data:
            print("No dipole data available for plotting")
            return
        
        # Separate cations and anions for plotting
        cation_types_in_system = set(self._get_unique_ion_types(self.cations).keys())
        anion_types_in_system = set(self._get_unique_ion_types(self.anions).keys())
        
        cation_data = {k: v for k, v in dipole_data.items() if k in cation_types_in_system}
        anion_data = {k: v for k, v in dipole_data.items() if k in anion_types_in_system}
        
        # Create plots
        n_plots = (1 if cation_data else 0) + (1 if anion_data else 0)
        if n_plots == 0:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 6))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot cation dipole distributions
        if cation_data:
            ax = axes[plot_idx]
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cation_data)))
            
            for i, (ion_type, angles) in enumerate(cation_data.items()):
                ax.hist(angles, bins=bins, alpha=0.7, color=colors[i], 
                    label=f'{ion_type} (n={len(angles)})', density=True)
            
            ax.set_xlabel('Dipole Angle (degrees)')
            ax.set_ylabel('Probability Density')
            ax.set_title('Cation Water Dipole Distributions', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(plot_range)
            
            plot_idx += 1
        
        # Plot anion dipole distributions
        if anion_data:
            ax = axes[plot_idx]
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(anion_data)))
            
            for i, (ion_type, angles) in enumerate(anion_data.items()):
                ax.hist(angles, bins=bins, alpha=0.7, color=colors[i], 
                    label=f'{ion_type} (n={len(angles)})', density=True)
            
            ax.set_xlabel('Dipole Angle (degrees)')
            ax.set_ylabel('Probability Density')
            ax.set_title('Anion Water Dipole Distributions', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(plot_range)
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'water_dipole_distributions_by_type.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {filename}")
        
        plt.show()
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("DIPOLE DISTRIBUTION SUMMARY")
        print(f"{'='*60}")
        
        for ion_type, angles in dipole_data.items():
            ion_category = 'cation' if ion_type in cation_types_in_system else 'anion'
            print(f"{ion_type} ({ion_category}):")
            print(f"  Sample size: {len(angles)}")
            print(f"  Mean angle: {angles.mean():.1f}° ± {angles.std():.1f}°")
            print(f"  Median angle: {np.median(angles):.1f}°")
            print(f"  Range: {angles.min():.1f}° - {angles.max():.1f}°")
            print()
        
        return dipole_data

    def compare_dipole_distributions_by_type(self, ion_types=None, save_plots=True, bins=50, use_cache=True):
        '''
        Create a comparison plot of dipole distributions overlaid for different ion types.
        FIXED: Now uses cached data if available to avoid recalculation.
        
        Parameters
        ----------
        ion_types : list or None
            List of ion types to compare
        save_plots : bool
            Whether to save the plot
        bins : int
            Number of histogram bins
        use_cache : bool
            Whether to use cached dipole data, default=True
        '''
        
        # Initialize cache if it doesn't exist
        if not hasattr(self, 'dipole_distributions_by_type'):
            self.dipole_distributions_by_type = {}
        
        # Determine which ion types to analyze
        if ion_types is None:
            cation_types = list(self._get_unique_ion_types(self.cations).keys())
            anion_types = list(self._get_unique_ion_types(self.anions).keys())
            ion_types = cation_types + anion_types
        
        # FIXED: Collect dipole data using cache first
        dipole_data = {}
        for ion_type in ion_types:
            # Check if we already have cached data
            if use_cache and ion_type in self.dipole_distributions_by_type:
                print(f"Using cached dipole data for {ion_type}")
                angles = self.dipole_distributions_by_type[ion_type]
            else:
                print(f"Calculating dipoles for {ion_type}...")
                angles = self.water_dipole_distribution(ion_type=ion_type)
                # Store in cache
                self.dipole_distributions_by_type[ion_type] = angles
            
            if len(angles) > 0:
                dipole_data[ion_type] = angles
            else:
                print(f"  Warning: No dipole data for {ion_type}")
        
        if not dipole_data:
            print("No dipole data available")
            return None
        
        # Create overlay plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Determine colors based on ion category
        cation_types_in_system = set(self._get_unique_ion_types(self.cations).keys())
        
        colors = []
        for ion_type in dipole_data.keys():
            if ion_type in cation_types_in_system:
                colors.append('blue')
            else:
                colors.append('red')
        
        # Plot all distributions
        for i, (ion_type, angles) in enumerate(dipole_data.items()):
            ion_category = 'cation' if ion_type in cation_types_in_system else 'anion'
            
            ax.hist(angles, bins=bins, alpha=0.6, color=colors[i], 
                label=f'{ion_type} ({ion_category}, n={len(angles)})', 
                density=True, histtype='step', linewidth=2)
        
        ax.set_xlabel('Dipole Angle (degrees)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Water Dipole Distribution Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'dipole_distribution_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved as: {filename}")
        
        plt.show()
        
        return dipole_data


    def interpret_dipole_orientation_by_type(self, ion_type):
        """
        Analyze and interpret water dipole orientations for a specific ion type.
        FIXED: Now uses cached dipole data if available to avoid recalculation.
        
        Parameters
        ----------
        ion_type : str
            Specific ion type (e.g., 'Na', 'Mg', 'Cl') or broad category ('cation', 'anion')
        """
        
        # FIXED: Check for cached dipole data first - use self instead of eq_opt
        angles = None
        
        # Try to get cached dipole data
        if hasattr(self, 'dipole_distributions_by_type') and ion_type in self.dipole_distributions_by_type:
            print(f"Using cached dipole data for {ion_type}")
            angles = self.dipole_distributions_by_type[ion_type]
        else:
            print(f"No cached data found for {ion_type}, calculating dipole angles...")
            # Calculate dipole angles and cache them
            angles = self.water_dipole_distribution(ion_type=ion_type)
            
            # Cache the result for future use
            if not hasattr(self, 'dipole_distributions_by_type'):
                self.dipole_distributions_by_type = {}
            self.dipole_distributions_by_type[ion_type] = angles
            print(f"Cached dipole data for {ion_type}")
        
        if len(angles) == 0:
            print(f"No dipole data available for {ion_type}")
            return None
        
        # Determine ion category for interpretation
        if hasattr(self, '_get_unique_ion_types'):
            cation_types = self._get_unique_ion_types(self.cations)
            anion_types = self._get_unique_ion_types(self.anions)
            
            if ion_type in cation_types:
                ion_category = 'cation'
                charge_sign = '+'
            elif ion_type in anion_types:
                ion_category = 'anion'
                charge_sign = '-'
            elif ion_type in ['cation', 'anion']:
                ion_category = ion_type
                charge_sign = '+' if ion_type == 'cation' else '-'
            else:
                ion_category = 'unknown'
                charge_sign = '?'
        else:
            # Fallback for cases where we can't determine
            ion_category = 'unknown'
            charge_sign = '?'
        
        # Calculate statistics
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        median_angle = np.median(angles)
        
        print(f"\n{'='*60}")
        print(f"{ion_type.upper()} HYDRATION ANALYSIS ({ion_category} {charge_sign})")
        print(f"{'='*60}")
        print(f"Sample size: {len(angles)} dipole measurements")
        print(f"Mean dipole angle: {mean_angle:.1f}° ± {std_angle:.1f}°")
        print(f"Median angle: {median_angle:.1f}°")
        print(f"Range: {np.min(angles):.1f}° - {np.max(angles):.1f}°")
        
        # Interpret the orientation
        print(f"\nORIENTATION INTERPRETATION:")
        if mean_angle < 30:
            print("→ STRONG ELECTROSTATIC ATTRACTION")
            print("  Water dipoles strongly align toward the ion")
            if ion_category == 'cation':
                print("  Oxygen atoms point toward cation (δ- toward +)")
            elif ion_category == 'anion':
                print("  Hydrogen atoms point toward anion (δ+ toward -)")
                
        elif mean_angle < 60:
            print("→ MODERATE ATTRACTION")
            print("  Clear preferential orientation with some geometric constraints")
            print("  Ion-water interaction dominates over thermal motion")
            
        elif mean_angle < 120:
            print("→ WEAK PREFERENTIAL ORIENTATION")
            print("  Mostly random orientation with slight bias")
            print("  Thermal motion competes with electrostatic forces")
            
        elif mean_angle < 150:
            print("→ SLIGHT REPULSION OR STERIC EFFECTS")
            print("  Dipoles show tendency to avoid direct alignment")
            print("  May indicate crowded coordination environment")
            
        else:
            print("→ STRONG REPULSION")
            print("  Dipoles actively point away from ion")
            print("  Unusual - check system or ion charge assignment")
        
        # Calculate and interpret order parameter
        cos_angles = np.cos(np.deg2rad(angles))
        order_param = 0.5 * (3 * np.mean(cos_angles) - 1)
        
        print(f"\nORDER PARAMETER ANALYSIS:")
        print(f"P₂ = {order_param:.3f}")
        
        if order_param > 0.3:
            print("→ HIGHLY ORDERED water structure")
            print("  Strong ion-induced water organization")
        elif order_param > 0.1:
            print("→ MODERATELY ORDERED water structure") 
            print("  Clear but not extreme water organization")
        elif order_param > -0.1:
            print("→ RANDOM ORIENTATION")
            print("  No significant preferential orientation")
        elif order_param > -0.3:
            print("→ MODERATELY INVERTED ordering")
            print("  Slight preference for perpendicular orientation")
        else:
            print("→ HIGHLY INVERTED ordering")
            print("  Strong preference for perpendicular orientation")
        
        # Ion-specific insights
        print(f"\nION-SPECIFIC INSIGHTS:")
        
        # Charge density effects
        if ion_type in ['Li', 'Mg', 'Al']:
            print("→ High charge density ion - expect strong ordering")
        elif ion_type in ['Na', 'Ca', 'F', 'Cl']:
            print("→ Moderate charge density - typical electrostatic behavior")
        elif ion_type in ['K', 'Rb', 'Br', 'I']:
            print("→ Low charge density - weaker water organization")
        
        # Compare to expected behavior
        if ion_category == 'cation' and mean_angle < 90:
            print("→ Behavior consistent with cationic hydration")
        elif ion_category == 'anion' and mean_angle < 90:
            print("→ Behavior consistent with anionic hydration")  
        elif mean_angle > 90:
            print("→ Unexpected orientation - investigate further")
        
        print(f"{'='*60}")
        
        # Return results for further analysis
        results = {
            'ion_type': ion_type,
            'ion_category': ion_category,
            'n_samples': len(angles),
            'mean_angle': mean_angle,
            'std_angle': std_angle,
            'median_angle': median_angle,
            'order_parameter': order_param,
            'angles': angles
        }
        
        return results 



    # Usage examples:
    def analyze_all_ion_dipole_orientations(eq_opt, save_summary=True):
        """
        Analyze dipole orientations for all available ion types.
        """
        
        # Get all available ion types
        if hasattr(eq_opt, '_get_unique_ion_types'):
            cation_types = list(eq_opt._get_unique_ion_types(eq_opt.cations).keys())
            anion_types = list(eq_opt._get_unique_ion_types(eq_opt.anions).keys())
            all_ion_types = cation_types + anion_types
        else:
            # Fallback to broad categories
            all_ion_types = ['cation', 'anion']
        
        print(f"Analyzing dipole orientations for: {all_ion_types}")
        
        all_results = {}
        
        for ion_type in all_ion_types:
            try:
                results = interpret_dipole_orientation_by_type(eq_opt, ion_type)
                if results:
                    all_results[ion_type] = results
            except Exception as e:
                print(f"Error analyzing {ion_type}: {e}")
                continue
        
        # Create summary comparison
        if len(all_results) > 1:
            print(f"\n{'='*80}")
            print("DIPOLE ORIENTATION SUMMARY - ALL ION TYPES")
            print(f"{'='*80}")
            print(f"{'Ion':<6} {'Category':<8} {'Mean°':<8} {'Std°':<8} {'P₂':<8} {'Samples':<8} {'Interpretation'}")
            print("-" * 80)
            
            for ion_type, data in all_results.items():
                interpretation = "Strong" if abs(data['order_parameter']) > 0.3 else \
                            "Moderate" if abs(data['order_parameter']) > 0.1 else "Weak"
                
                print(f"{ion_type:<6} {data['ion_category']:<8} {data['mean_angle']:<8.1f} "
                    f"{data['std_angle']:<8.1f} {data['order_parameter']:<8.3f} "
                    f"{data['n_samples']:<8} {interpretation}")
            
            print(f"{'='*80}")
        
        if save_summary:
            # Save detailed results
            import pickle
            with open('dipole_orientation_analysis.pkl', 'wb') as f:
                pickle.dump(all_results, f)
            print("Detailed results saved to: dipole_orientation_analysis.pkl")
        
        return all_results

    # Quick analysis function
    def quick_dipole_analysis(eq_opt, ion_types=None):
        """
        Quick dipole analysis for specified ion types.
        
        Parameters
        ----------
        eq_opt : EquilibriumAnalysisOptimized
            Analysis object
        ion_types : list or None
            List of ion types to analyze. If None, analyzes all available.
        """
        
        if ion_types is None:
            return analyze_all_ion_dipole_orientations(eq_opt)
        
        results = {}
        for ion_type in ion_types:
            results[ion_type] = interpret_dipole_orientation_by_type(eq_opt, ion_type)
        
        return results



    def replot_dipole_distributions_by_coordination_histogram(self, ion_type, coordination_states=None, 
                                                            bins=100, save_plots=True):
        '''
        Replot water dipole distributions as HISTOGRAMS from already calculated data.
        
        Parameters
        ----------
        ion_type : str
            Ion type to plot (e.g., 'Ca', 'Na', 'Mg')
        coordination_states : list, optional
            Which coordination numbers to include. If None, plots all available.
        bins : int
            Number of histogram bins for display
        save_plots : bool
            Whether to save the plot
        '''
        
        if not hasattr(self, 'dipole_by_coordination') or ion_type not in self.dipole_by_coordination:
            print(f"No dipole data found for {ion_type}. Run water_dipole_distribution_by_coordination() first.")
            return None
        
        dipole_data = self.dipole_by_coordination[ion_type]
        
        # Use all available CNs if not specified
        if coordination_states is None:
            coordination_states = sorted(dipole_data.keys())
        else:
            # Filter to available CNs
            coordination_states = [cn for cn in coordination_states if cn in dipole_data]
        
        if not coordination_states:
            print(f"No data available for requested coordination states")
            print(f"Available CNs: {sorted(dipole_data.keys())}")
            return None
        
        # Determine ion category for coloring
        cation_types_in_system = set()
        if hasattr(self, '_get_unique_ion_types'):
            cation_types_in_system = set(self._get_unique_ion_types(self.cations).keys())
        
        ion_category = 'cation' if ion_type in cation_types_in_system else 'anion'
        
        # Use the same plotting method as the original
        self._plot_dipole_by_coordination(ion_type, ion_category, 
                                        {cn: dipole_data[cn] for cn in coordination_states}, 
                                        bins)
        
        return True


    def replot_dipole_distributions_by_coordination(self, ion_type, coordination_states=None, 
                                                    bins=100, save_plots=True):
        '''
        Replot water dipole distributions from already calculated data without recalculating.
        
        Parameters
        ----------
        ion_type : str
            Ion type to plot (e.g., 'Ca', 'Na', 'Mg')
        coordination_states : list, optional
            Which coordination numbers to include. If None, plots all available.
        bins : int
            Number of histogram bins for display
        save_plots : bool
            Whether to save the plot
        '''
        
        if not hasattr(self, 'dipole_by_coordination') or ion_type not in self.dipole_by_coordination:
            print(f"No dipole data found for {ion_type}. Run water_dipole_distribution_by_coordination() first.")
            return None
        
        dipole_data = self.dipole_by_coordination[ion_type]
        
        # Use all available CNs if not specified
        if coordination_states is None:
            coordination_states = sorted(dipole_data.keys())
        else:
            # Filter to available CNs
            coordination_states = [cn for cn in coordination_states if cn in dipole_data]
        
        if not coordination_states:
            print(f"No data available for requested coordination states")
            print(f"Available CNs: {sorted(dipole_data.keys())}")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(coordination_states)))
        
        for i, cn in enumerate(sorted(coordination_states)):
            angles = dipole_data[cn]
            
            # Calculate histogram
            counts, bin_edges = np.histogram(angles, bins=bins, range=(0, 180), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot
            ax.plot(bin_centers, counts, label=f'CN={cn} (n={len(angles)})', 
                    color=colors[i], linewidth=2)
        
        ax.set_xlabel('Dipole Angle (degrees)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Water Dipole Orientation Distribution - {ion_type}', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'dipole_distribution_{ion_type}_CN_{"_".join(map(str, coordination_states))}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {filename}")
        
        plt.show()
        
        return fig, ax






    def determine_ion_pairing_cutoffs(self, ion_type='cation', find_peaks_kwargs={'distance': 5, 'height': -1.1, 'prominence': 0.1}, 
                                    save_plots=True, use_extended_rdf=False, plot_range=None):
        '''
        Calculate the cation-anion radial distributions and identify cutoffs for ion pairing events.
        Now supports both broad categories ('cation', 'anion') and specific ion types ('Na', 'Mg', 'Cl').
        Should plot to ensure the cutoff regimes visually look correct, since these are 
        sensitive to the peak detection algorithm. 

        Parameters
        ----------
        ion_type : str
            Ion type to analyze. Can be:
            - Broad categories: 'cation', 'anion' (original behavior)
            - Specific ion types: 'Na', 'Mg', 'K', 'Cl', 'Br', etc. (new functionality)
        find_peak_kwargs : dict
            Keyword arguments for `scipy.find_peaks` used to find the first 3 minima in the cation-anion RDF,
            default={'distance' : 5, 'height' : -1.1} worked well for NaCl at 0.6 M with OPC3 water
        save_plots : bool
            Whether to save and show the plot with regions shaded, default=True
        use_extended_rdf : bool
            Whether to use extended RDF data from generate_rdfs() instead of SolvationAnalysis data, default=False
        plot_range : float
            Maximum r value for plotting. If None, uses 10 for standard RDF or max range for extended RDF

        Returns
        -------
        ion_pairs : Results
            Results object with ion pairing cutoffs (CIP, SIP, DSIP, FI)
        '''

        # Handle both broad categories and specific ion types
        def get_rdf_data_and_category(eq_analysis, ion_type, use_extended_rdf):
            """Get appropriate RDF data and determine ion category"""
            
            # Handle broad categories (original behavior)
            if ion_type in ['cation', 'anion']:
                if use_extended_rdf:
                    try:
                        # Use the extended RDF data from generate_rdfs
                        r = eq_analysis.rdfs['ci-ai'].bins
                        rdf = eq_analysis.rdfs['ci-ai'].rdf
                        rdf_key = 'ci-ai'
                        print(f"Using extended RDF data: {r.min():.2f} - {r.max():.2f} Å")
                        return r, rdf, rdf_key, ion_type
                    except (AttributeError, KeyError):
                        print('Extended RDFs not generated. Run `generate_rdfs()` first or set use_extended_rdf=False')
                        return None, None, None, None
                else:
                    try:
                        # Use the standard SolvationAnalysis data
                        r = eq_analysis.solute_ci.rdf_data['Cation']['coion'][0]
                        rdf = eq_analysis.solute_ci.rdf_data['Cation']['coion'][1]
                        rdf_key = 'ci-ai'
                        print(f"Using SolvationAnalysis RDF data: {r.min():.2f} - {r.max():.2f} Å")
                        return r, rdf, rdf_key, ion_type
                    except AttributeError:
                        print('Solutes not initialized. Try `initialize_Solutes()` first')
                        return None, None, None, None
            
            # Handle specific ion types (new behavior)
            else:
                if not use_extended_rdf:
                    print("Warning: Ion-specific analysis requires extended RDFs.")
                    print("Setting use_extended_rdf=True automatically.")
                    use_extended_rdf = True
                
                try:
                    # Check available RDFs for specific ion types
                    if not hasattr(eq_analysis, 'rdfs') or not eq_analysis.rdfs:
                        print('Extended RDFs not generated. Run `generate_rdfs()` with ion-specific RDFs first')
                        return None, None, None, None
                    
                    # Try to find ion-specific RDF
                    # Look for patterns like 'Na-Cl', 'Mg-Cl', etc.
                    possible_rdfs = []
                    
                    # Determine if ion_type is cation or anion by checking element properties
                    # Common cations: Na, K, Mg, Ca, Li
                    # Common anions: Cl, Br, F, I
                    cation_elements = ['Na', 'K', 'Mg', 'Ca', 'Li', 'NH4']
                    anion_elements = ['Cl', 'Br', 'F', 'I']
                    
                    if ion_type in cation_elements:
                        ion_category = 'cation'
                        # Look for cation-anion RDFs
                        for anion in anion_elements:
                            rdf_key = f'{ion_type}-{anion}'
                            if rdf_key in eq_analysis.rdfs and eq_analysis.rdfs[rdf_key] is not None:
                                possible_rdfs.append(rdf_key)
                    elif ion_type in anion_elements:
                        ion_category = 'anion'
                        # Look for cation-anion RDFs
                        for cation in cation_elements:
                            rdf_key = f'{cation}-{ion_type}'
                            if rdf_key in eq_analysis.rdfs and eq_analysis.rdfs[rdf_key] is not None:
                                possible_rdfs.append(rdf_key)
                    else:
                        # Unknown ion type - try to guess from available RDFs
                        available_rdfs = [k for k, v in eq_analysis.rdfs.items() if v is not None]
                        possible_rdfs = [k for k in available_rdfs if ion_type in k and '-' in k]
                        ion_category = 'unknown'
                    
                    if possible_rdfs:
                        # Use the first available ion-specific RDF
                        rdf_key = possible_rdfs[0]
                        rdf_data = eq_analysis.rdfs[rdf_key]
                        r = rdf_data.bins
                        rdf = rdf_data.rdf
                        print(f"Using ion-specific RDF '{rdf_key}': {r.min():.2f} - {r.max():.2f} Å")
                        return r, rdf, rdf_key, ion_category
                    else:
                        # Fallback to combined cation-anion RDF
                        print(f"No specific RDF found for '{ion_type}'. Using combined cation-anion RDF.")
                        if 'ci-ai' in eq_analysis.rdfs and eq_analysis.rdfs['ci-ai'] is not None:
                            rdf_data = eq_analysis.rdfs['ci-ai']
                            r = rdf_data.bins
                            rdf = rdf_data.rdf
                            rdf_key = 'ci-ai'
                            print(f"Using combined RDF '{rdf_key}': {r.min():.2f} - {r.max():.2f} Å")
                            return r, rdf, rdf_key, 'combined'
                        else:
                            available_rdfs = [k for k, v in eq_analysis.rdfs.items() if v is not None]
                            print(f"No suitable RDFs found for ion pairing analysis.")
                            print(f"Available RDFs: {available_rdfs}")
                            return None, None, None, None
                            
                except (AttributeError, KeyError) as e:
                    print(f'Error accessing RDF data: {e}')
                    return None, None, None, None
        
        # Get appropriate RDF data and ion category
        r, rdf, rdf_key, ion_category = get_rdf_data_and_category(self, ion_type, use_extended_rdf)
        
        if r is None or rdf is None:
            return None
        
        # Set plot range
        if plot_range is None:
            if use_extended_rdf:
                plot_range = min(20, r.max())  # Default to 20 Å or max range if smaller
            else:
                plot_range = 10  # Default for standard range
        
        print(f"Determining ion pairing cutoffs for {ion_type} ({ion_category})")
        print(f"Using RDF: {rdf_key}")
        
        # Find peaks (minima in RDF)
        mins, min_props = find_peaks(-rdf, **find_peaks_kwargs)

        # DEBUG: Print information about found peaks
        print(f"Found {len(mins)} minima at positions: {r[mins]}")
        if len(mins) > 0:
            print(f"RDF values at minima: {rdf[mins]}")

        # Create ion pairs results - store by ion type for specific analysis
        if not hasattr(self, 'ion_pairs_by_type'):
            self.ion_pairs_by_type = {}
        
        ion_pairs = Results()
        
        # Handle cases with different numbers of minima
        if len(mins) >= 3:
            # Standard case with 3+ minima
            ion_pairs['CIP'] = (0, r[mins[0]])
            ion_pairs['SIP'] = (r[mins[0]], r[mins[1]])
            ion_pairs['DSIP'] = (r[mins[1]], r[mins[2]])
            ion_pairs['FI'] = (r[mins[2]], np.inf)
            plot_mins = mins[:3]  # Use first 3 minima for plotting
            print(f"CIP (Contact Ion Pair): 0.00 - {r[mins[0]]:.2f} Å")
            print(f"SIP (Solvent-separated): {r[mins[0]]:.2f} - {r[mins[1]]:.2f} Å")
            print(f"DSIP (Double Solvent-separated): {r[mins[1]]:.2f} - {r[mins[2]]:.2f} Å")
            print(f"FI (Free Ions): {r[mins[2]]:.2f} - ∞ Å")
        elif len(mins) == 2:
            # Only 2 minima found - merge DSIP into SIP
            print("Warning: Only 2 minima found. Combining SIP and DSIP regions.")
            ion_pairs['CIP'] = (0, r[mins[0]])
            ion_pairs['SIP'] = (r[mins[0]], r[mins[1]])
            ion_pairs['FI'] = (r[mins[1]], np.inf)
            plot_mins = mins  # Use both minima for plotting
            print(f"CIP (Contact Ion Pair): 0.00 - {r[mins[0]]:.2f} Å")
            print(f"SIP (Solvent-separated): {r[mins[0]]:.2f} - {r[mins[1]]:.2f} Å")
            print(f"FI (Free Ions): {r[mins[1]]:.2f} - ∞ Å")
        elif len(mins) == 1:
            # Only 1 minimum found - simple CIP/FI division
            print("Warning: Only 1 minimum found. Using simple CIP/FI division.")
            ion_pairs['CIP'] = (0, r[mins[0]])
            ion_pairs['FI'] = (r[mins[0]], np.inf)
            plot_mins = mins  # Use the single minimum for plotting
            print(f"CIP (Contact Ion Pair): 0.00 - {r[mins[0]]:.2f} Å")
            print(f"FI (Free Ions): {r[mins[0]]:.2f} - ∞ Å")
        else:
            # No minima found - use default cutoffs
            print("Warning: No minima found. Using default cutoffs.")
            ion_pairs['CIP'] = (0, 4.0)
            ion_pairs['SIP'] = (4.0, 7.0)
            ion_pairs['FI'] = (7.0, np.inf)
            plot_mins = []  # No minima to plot
            print("CIP (Contact Ion Pair): 0.00 - 4.00 Å")
            print("SIP (Solvent-separated): 4.00 - 7.00 Å")
            print("FI (Free Ions): 7.00 - ∞ Å")

        # Store ion-specific results
        self.ion_pairs_by_type[ion_type] = {
            'ion_pairs': ion_pairs,
            'rdf_key': rdf_key,
            'ion_category': ion_category,
            'minima_positions': r[mins] if len(mins) > 0 else [],
            'minima_values': rdf[mins] if len(mins) > 0 else []
        }
        
        # For backward compatibility, also store in the original attribute for broad categories
        if ion_type in ['cation', 'anion']:
            self.ion_pairs = ion_pairs

        if save_plots:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(r, rdf, color='k', linewidth=2, label='g(r)')

            # Calculate dynamic y-limits based on data
            y_min = 0
            y_max = np.max(rdf) * 1.1
            text_y_pos = y_max * 0.95

            # Plot regions based on what was actually found
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
            pair_types = list(ion_pairs.keys())
            
            le = 2
            for i, (key, (start, end)) in enumerate(ion_pairs.items()):
                end_plot = min(end, plot_range) if not np.isinf(end) else plot_range
                ax.fill_betweenx(np.linspace(y_min, y_max), max(le, start), end_plot, 
                            alpha=0.4, color=colors[i % len(colors)], label=key)
                ax.text((max(le, start) + end_plot) / 2, text_y_pos, key, ha='center', 
                    fontweight='bold', fontsize=12)
                le = end_plot

            # Mark the found minima with red triangles
            if len(plot_mins) > 0:
                ax.scatter(r[plot_mins], rdf[plot_mins], color='red', s=100, zorder=5, 
                        marker='v', edgecolor='black', linewidth=2, label='Detected minima')
            
            ax.set_xlabel(r'r ($\mathrm{\AA}$)', fontsize=12)
            ax.set_ylabel('g(r)', fontsize=12)
            ax.set_xlim(2, plot_range)
            ax.set_ylim(y_min, y_max)
            
            # Create title based on ion type and RDF used
            if ion_type in ['cation', 'anion']:
                title = f'{ion_type.title()}-{("Anion" if ion_type == "cation" else "Cation")} Ion Pairing Analysis'
            else:
                title = f'{ion_type} Ion Pairing Analysis'
            
            rdf_suffix = " (Extended RDF)" if use_extended_rdf else " (Standard RDF)"
            ax.set_title(title + rdf_suffix, fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Create filename based on ion type and RDF type
            rdf_type = "extended" if use_extended_rdf else "standard"
            filename = f'ion_pairing_cutoffs_{ion_type}_{rdf_type}.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {filename}")
            plt.show()

        return ion_pairs

    def get_ion_pairing_cutoffs_for_type(self, ion_type):
        '''
        Get ion pairing cutoffs for a specific ion type.
        
        Parameters
        ----------
        ion_type : str
            Ion type name (e.g., 'Na', 'Mg', 'Cl', 'cation', 'anion')
        
        Returns
        -------
        cutoffs : dict or None
            Ion pairing cutoffs for the specified ion type
        '''
        
        if not hasattr(self, 'ion_pairs_by_type'):
            print("Ion pairing cutoffs not calculated. Run determine_ion_pairing_cutoffs() first.")
            return None
        
        if ion_type in self.ion_pairs_by_type:
            return self.ion_pairs_by_type[ion_type]['ion_pairs']
        else:
            available_types = list(self.ion_pairs_by_type.keys())
            print(f"Ion type '{ion_type}' not found. Available types: {available_types}")
            return None

    def compare_ion_pairing_cutoffs_by_type(self, ion_types=None, save_plots=True):
        '''
        Compare ion pairing cutoffs for different ion types.
        
        Parameters
        ----------
        ion_types : list or None
            List of ion types to compare. If None, compares all available types
        save_plots : bool
            Whether to save the comparison plot
        '''
        
        if not hasattr(self, 'ion_pairs_by_type'):
            print("Ion pairing cutoffs not calculated. Run determine_ion_pairing_cutoffs() for different ion types first.")
            return
        
        if ion_types is None:
            ion_types = list(self.ion_pairs_by_type.keys())
        
        if not ion_types:
            print("No ion types available for comparison.")
            return
        
        print(f"Comparing ion pairing cutoffs for: {ion_types}")
        
        # Create comparison plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(ion_types)))
        
        for i, ion_type in enumerate(ion_types):
            if ion_type not in self.ion_pairs_by_type:
                print(f"Warning: No data for {ion_type}")
                continue
                
            data = self.ion_pairs_by_type[ion_type]
            rdf_key = data['rdf_key']
            
            # Get RDF data
            try:
                if 'ci-ai' in rdf_key:
                    rdf_data = self.rdfs['ci-ai']
                else:
                    rdf_data = self.rdfs[rdf_key]
                
                r = rdf_data.bins
                rdf = rdf_data.rdf
                
                # Plot RDF
                ax.plot(r, rdf, color=colors[i], linewidth=2, 
                    label=f'{ion_type} ({rdf_key})', alpha=0.8)
                
                # Mark minima
                if len(data['minima_positions']) > 0:
                    ax.scatter(data['minima_positions'], data['minima_values'], 
                            color=colors[i], s=60, marker='v', 
                            edgecolor='black', linewidth=1, zorder=5)
            
            except (KeyError, AttributeError):
                print(f"Warning: Could not plot RDF for {ion_type}")
                continue
        
        ax.set_xlabel(r'r ($\mathrm{\AA}$)', fontsize=12)
        ax.set_ylabel('g(r)', fontsize=12)
        ax.set_title('Ion Pairing RDF Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_xlim(2, 12)
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'ion_pairing_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved as: {filename}")
        
        plt.show()
        
        # Print summary table
        print(f"\n{'='*80}")
        print(f"ION PAIRING CUTOFFS COMPARISON")
        print(f"{'='*80}")
        print(f"{'Ion Type':<10} {'RDF Used':<15} {'CIP (Å)':<12} {'SIP (Å)':<15} {'DSIP (Å)':<15} {'FI (Å)':<10}")
        print(f"{'-'*80}")
        
        for ion_type in ion_types:
            if ion_type not in self.ion_pairs_by_type:
                continue
                
            data = self.ion_pairs_by_type[ion_type]
            pairs = data['ion_pairs']
            rdf_key = data['rdf_key']
            
            cip_str = f"0.00-{pairs['CIP'][1]:.2f}" if 'CIP' in pairs else "N/A"
            sip_str = f"{pairs['SIP'][0]:.2f}-{pairs['SIP'][1]:.2f}" if 'SIP' in pairs else "N/A"
            dsip_str = f"{pairs['DSIP'][0]:.2f}-{pairs['DSIP'][1]:.2f}" if 'DSIP' in pairs else "N/A"
            fi_str = f"{pairs['FI'][0]:.2f}-∞" if 'FI' in pairs else "N/A"
            
            print(f"{ion_type:<10} {rdf_key:<15} {cip_str:<12} {sip_str:<15} {dsip_str:<15} {fi_str:<10}")


    def save_ion_pairing_cutoffs_to_file(self, filename='ion_pairing_cutoffs_cache.pkl'):
        '''
        Save ion pairing cutoffs to file for persistence across sessions.
        Includes all modifications made through the interactive editor.
        
        Parameters
        ----------
        filename : str
            Output filename, default='ion_pairing_cutoffs_cache.pkl'
        
        Returns
        -------
        success : bool
            True if save was successful
        '''
        
        if not hasattr(self, 'ion_pairs_by_type') or not self.ion_pairs_by_type:
            print("No ion pairing cutoffs to save")
            return False
        
        try:
            # Prepare ion pairing data for serialization
            pairing_data = {}
            
            for ion_type, pairing_info in self.ion_pairs_by_type.items():
                pairing_data[ion_type] = {
                    'ion_pairs': {},
                    'rdf_key': pairing_info['rdf_key'],
                    'ion_category': pairing_info['ion_category'],
                    'minima_positions': pairing_info.get('minima_positions', []).copy() if isinstance(pairing_info.get('minima_positions'), np.ndarray) else pairing_info.get('minima_positions', []),
                    'minima_values': pairing_info.get('minima_values', []).copy() if isinstance(pairing_info.get('minima_values'), np.ndarray) else pairing_info.get('minima_values', [])
                }
                
                # Save ion pairing regions (including any modifications)
                for region_name, bounds in pairing_info['ion_pairs'].items():
                    pairing_data[ion_type]['ion_pairs'][region_name] = bounds
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(pairing_data, f)
            
            print(f"Ion pairing cutoffs saved to {filename}")
            print(f"  Saved {len(pairing_data)} ion types")
            print(f"  Ion types: {list(pairing_data.keys())}")
            
            # Print summary of saved cutoffs
            for ion_type, data in pairing_data.items():
                print(f"\n  {ion_type} ({data['ion_category']}):")
                for region_name, (start, end) in data['ion_pairs'].items():
                    end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                    print(f"    {region_name}: {start:.2f} - {end_str} Å")
            
            return True
            
        except Exception as e:
            print(f"Error saving ion pairing cutoffs: {e}")
            traceback.print_exc()
            return False

    def load_ion_pairing_cutoffs_from_file(self, filename='ion_pairing_cutoffs_cache.pkl'):
        '''
        Load ion pairing cutoffs from file with error handling.
        
        Parameters
        ----------
        filename : str
            Input filename, default='ion_pairing_cutoffs_cache.pkl'
        
        Returns
        -------
        success : bool
            True if load was successful
        '''
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        try:
            # Check file size first
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print(f"Cache file {filename} is empty")
                return False
            
            # Load data
            with open(filename, 'rb') as f:
                pairing_data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(pairing_data, dict):
                print(f"Invalid ion pairing cutoffs cache format")
                return False
            
            # Reconstruct the ion pairing cutoffs structure
            self.ion_pairs_by_type = {}
            
            for ion_type, data in pairing_data.items():
                # Create Results object for ion_pairs
                from MDAnalysis.analysis.base import Results
                ion_pairs = Results()
                
                for region_name, bounds in data['ion_pairs'].items():
                    ion_pairs[region_name] = bounds
                
                self.ion_pairs_by_type[ion_type] = {
                    'ion_pairs': ion_pairs,
                    'rdf_key': data['rdf_key'],
                    'ion_category': data['ion_category'],
                    'minima_positions': data.get('minima_positions', []),
                    'minima_values': data.get('minima_values', [])
                }
            
            # Print summary
            successful_types = list(self.ion_pairs_by_type.keys())
            
            print(f"Ion pairing cutoffs loaded from {filename}")
            print(f"  Loaded {len(successful_types)} ion types successfully")
            if successful_types:
                print(f"  Available types: {', '.join(successful_types)}")
            
            # Print detailed summary
            print(f"\n  Ion pairing cutoffs summary:")
            for ion_type, pairing_info in self.ion_pairs_by_type.items():
                print(f"    {ion_type} ({pairing_info['ion_category']}):")
                for region_name, (start, end) in pairing_info['ion_pairs'].items():
                    end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                    print(f"      {region_name}: {start:.2f} - {end_str} Å")
            
            return True
            
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            print("Removing corrupted cache file...")
            try:
                os.remove(filename)
                print(f"Corrupted file {filename} removed")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading ion pairing cutoffs from {filename}: {e}")
            traceback.print_exc()
            return False

    def determine_ion_pairing_cutoffs_with_cache(self, cache_filename='ion_pairing_cutoffs_cache.pkl', 
                                                force_recalc=False, **kwargs):
        '''
        Determine ion pairing cutoffs with automatic caching.
        
        Parameters
        ----------
        cache_filename : str
            Cache file name, default='ion_pairing_cutoffs_cache.pkl'
        force_recalc : bool
            Force recalculation even if cache exists, default=False
        **kwargs : dict
            Arguments to pass to determine_ion_pairing_cutoffs()
        
        Returns
        -------
        cutoffs : dict
            Dictionary of ion pairing cutoffs
        '''
        
        # Try to load from cache first
        if not force_recalc and os.path.exists(cache_filename):
            print("Attempting to load ion pairing cutoffs from cache...")
            if self.load_ion_pairing_cutoffs_from_file(cache_filename):
                print("✓ Successfully loaded ion pairing cutoffs from cache")
                return self.ion_pairs_by_type
            else:
                print("✗ Cache loading failed, will recalculate")
        
        # Calculate ion pairing cutoffs
        print("Calculating ion pairing cutoffs...")
        cutoffs = self.determine_ion_pairing_cutoffs(**kwargs)
        
        # Save to cache
        if cutoffs:
            print("Saving ion pairing cutoffs to cache...")
            if self.save_ion_pairing_cutoffs_to_file(cache_filename):
                print("✓ Ion pairing cutoffs cached successfully")
            else:
                print("✗ Cache saving failed, but cutoffs are available in memory")
        
        return cutoffs


    def benchmark_performance(self, methods=['rdfs', 'coordination', 'shells'], step=None, njobs=None):
        '''
        Benchmark the performance of different methods.
        
        Parameters
        ----------
        methods : list
            List of methods to benchmark: 'rdfs', 'coordination', 'shells'
        step : int
            Step size for benchmarking, default uses auto-tuned value
        njobs : int
            Number of CPUs for benchmarking, default uses auto-tuned value
        '''
        
        import time
        
        if step is None:
            step = self.default_step
        if njobs is None:
            njobs = self.default_njobs
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE BENCHMARK")
        print(f"{'='*60}")
        print(f"System: {len(self.universe.atoms)} atoms, {len(self.universe.trajectory)} frames")
        print(f"Parameters: step={step}, njobs={njobs}")
        
        results = {}
        
        if 'rdfs' in methods:
            print(f"\nBenchmarking RDF calculation...")
            start_time = time.time()
            self.generate_rdfs(step=step, njobs=njobs, range=(0, 10))  # Shorter range for speed
            rdf_time = time.time() - start_time
            results['rdfs'] = rdf_time
            print(f"RDF calculation: {rdf_time:.2f} seconds")
        
        if 'coordination' in methods and hasattr(self, 'solute_ci'):
            print(f"\nBenchmarking coordination number calculation...")
            start_time = time.time()
            self.get_coordination_numbers(step=step)
            coord_time = time.time() - start_time
            results['coordination'] = coord_time
            print(f"Coordination numbers: {coord_time:.2f} seconds")
        
        if 'shells' in methods:
            if hasattr(self, 'cation_solvation_shells'):
                print(f"\nBenchmarking shell coordination (matrix vs KDTree)...")
                
                # Test matrix method
                start_time = time.time()
                self.get_coordination_numbers_by_shell(step=step, use_kdtree=False)
                matrix_time = time.time() - start_time
                results['shells_matrix'] = matrix_time
                print(f"Shell coordination (matrix): {matrix_time:.2f} seconds")
                
                # Test KDTree method
                start_time = time.time()
                self.get_coordination_numbers_by_shell(step=step, use_kdtree=True)
                kdtree_time = time.time() - start_time
                results['shells_kdtree'] = kdtree_time
                print(f"Shell coordination (KDTree): {kdtree_time:.2f} seconds")
                
                speedup = matrix_time / kdtree_time if kdtree_time > 0 else 0
                print(f"KDTree speedup: {speedup:.2f}x")
            else:
                print("Shells not available - run determine_ion_solvation_shells() first")
        
        # Memory usage
        memory_mb = self.get_memory_usage()
        results['memory_mb'] = memory_mb
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY:")
        for method, time_taken in results.items():
            if method != 'memory_mb':
                print(f"  {method}: {time_taken:.2f} seconds")
        print(f"  Peak memory: {memory_mb:.1f} MB")
        print(f"{'='*60}")
        
        return results


    def plot_cross_sections_enhanced_edges_no_lines(self, frame=0, ion_type='cation', max_ions=4, 
                                                show_edge_markers=False, boundary_linewidth=1.0, 
                                                marker_size=50, ion_size=250, marker_type='o',
                                                atom_scale_factor=100):
        """
        Enhanced version with filled cross-sections, atoms sized by van der Waals radii, and optional hull vertex markers
        Now supports ion-specific analysis (e.g., 'Na', 'Mg', 'Cl') in addition to broad categories
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from scipy.spatial import ConvexHull
        import numpy as np
        
        # Import utilities - handle different import paths
        try:
            from utils.linear_algebra import create_plane_from_point_and_normal, line_plane_intersection, project_to_plane
        except ImportError:
            # Try alternative import path for optimized class
            import sys
            sys.path.append('/Users/roev0007/Documents/solvation_shells')
            from utils.linear_algebra import create_plane_from_point_and_normal, line_plane_intersection, project_to_plane
        
        # Handle both broad categories and specific ion types
        def get_ions_and_category(eq_analysis, ion_type):
            """Get ions and determine category based on ion_type"""
            
            # Handle broad categories (original behavior)
            if ion_type in ['cation', 'anion']:
                ions = eq_analysis.cations if ion_type == 'cation' else eq_analysis.anions
                return ions, ion_type
            
            # Handle specific ion types (new behavior)
            else:
                # Check if optimized class with ion-type-specific methods
                if hasattr(eq_analysis, '_get_unique_ion_types'):
                    cation_types = eq_analysis._get_unique_ion_types(eq_analysis.cations)
                    anion_types = eq_analysis._get_unique_ion_types(eq_analysis.anions)
                    
                    if ion_type in cation_types:
                        return cation_types[ion_type], 'cation'
                    elif ion_type in anion_types:
                        return anion_types[ion_type], 'anion'
                    else:
                        available_types = list(cation_types.keys()) + list(anion_types.keys())
                        raise ValueError(f"Ion type '{ion_type}' not found. Available types: {available_types}")
                else:
                    # Fallback for original class - filter by element/name
                    all_cations = eq_analysis.cations
                    all_anions = eq_analysis.anions
                    
                    # Try to filter by element first, then by name
                    cation_matches = []
                    anion_matches = []
                    
                    for atom in all_cations:
                        if (hasattr(atom, 'element') and atom.element == ion_type) or \
                        (hasattr(atom, 'name') and atom.name.startswith(ion_type)):
                            cation_matches.append(atom)
                    
                    for atom in all_anions:
                        if (hasattr(atom, 'element') and atom.element == ion_type) or \
                        (hasattr(atom, 'name') and atom.name.startswith(ion_type)):
                            anion_matches.append(atom)
                    
                    if cation_matches:
                        # Create AtomGroup for cation matches
                        return eq_analysis.universe.atoms[np.array([atom.index for atom in cation_matches])], 'cation'
                    elif anion_matches:
                        # Create AtomGroup for anion matches
                        return eq_analysis.universe.atoms[np.array([atom.index for atom in anion_matches])], 'anion'
                    else:
                        raise ValueError(f"No atoms found matching ion type '{ion_type}'")
        
        # Get ions and determine actual category
        ions, actual_category = get_ions_and_category(self, ion_type)
        
        # Get radius based on specific ion type or category
        def get_coordination_radius(eq_analysis, ion_type, actual_category):
            """Get coordination radius for the specified ion type"""
            
            # Try ion-type-specific radius first (optimized class)
            if hasattr(eq_analysis, 'get_coordination_radius_by_type'):
                radius = eq_analysis.get_coordination_radius_by_type(ion_type)
                if radius is not None:
                    print(f"Using {ion_type}-specific coordination radius: {radius:.2f} Å")
                    return radius
            
            # Try ion-type-specific solutes
            if hasattr(eq_analysis, 'solutes_ci') and hasattr(eq_analysis, 'solutes_ai'):
                if actual_category == 'cation' and ion_type in eq_analysis.solutes_ci:
                    solute = eq_analysis.solutes_ci[ion_type]
                    if solute is not None and hasattr(solute, 'radii') and 'water' in solute.radii:
                        radius = solute.radii['water']
                        print(f"Using {ion_type} solute radius: {radius:.2f} Å")
                        return radius
                elif actual_category == 'anion' and ion_type in eq_analysis.solutes_ai:
                    solute = eq_analysis.solutes_ai[ion_type]
                    if solute is not None and hasattr(solute, 'radii') and 'water' in solute.radii:
                        radius = solute.radii['water']
                        print(f"Using {ion_type} solute radius: {radius:.2f} Å")
                        return radius
            
            # Fallback to broad category solutes
            try:
                if actual_category == 'cation':
                    radius = eq_analysis.solute_ci.radii['water']
                    print(f"Using general cation radius: {radius:.2f} Å")
                    return radius
                else:
                    radius = eq_analysis.solute_ai.radii['water']
                    print(f"Using general anion radius: {radius:.2f} Å")
                    return radius
            except (AttributeError, NameError):
                print("Solutes not initialized. Trying to initialize...")
                if hasattr(eq_analysis, 'initialize_Solutes'):
                    eq_analysis.initialize_Solutes()
                    if actual_category == 'cation':
                        return eq_analysis.solute_ci.radii['water']
                    else:
                        return eq_analysis.solute_ai.radii['water']
                else:
                    # Default fallback values
                    default_radii = {'cation': 2.8, 'anion': 3.5}
                    radius = default_radii[actual_category]
                    print(f"Using default {actual_category} radius: {radius:.2f} Å")
                    return radius
        
        r0 = get_coordination_radius(self, ion_type, actual_category)
        
        # FIXED: Get vdW radii - handle both class types and different import formats
        def get_vdw_radii(eq_analysis):
            """Get van der Waals radii dictionary"""
            
            # Try to get from the class first
            if hasattr(eq_analysis, 'vdW_radii'):
                vdw_obj = eq_analysis.vdW_radii
                # Check if it's already a dictionary
                if isinstance(vdw_obj, dict):
                    return vdw_obj
                # Check if it has a dictionary attribute
                elif hasattr(vdw_obj, '__dict__'):
                    # Look for common dictionary attribute names
                    for attr_name in ['radii', 'vdw_radii', 'data', '__dict__']:
                        if hasattr(vdw_obj, attr_name):
                            attr_val = getattr(vdw_obj, attr_name)
                            if isinstance(attr_val, dict):
                                return attr_val
            
            # Try to import vdW radii directly
            try:
                from utils.file_rw import vdW_radii as imported_vdw
                # Check if imported object is a dictionary
                if isinstance(imported_vdw, dict):
                    return imported_vdw
                # Check if it has dictionary attributes
                elif hasattr(imported_vdw, '__dict__'):
                    for attr_name in ['radii', 'vdw_radii', 'data']:
                        if hasattr(imported_vdw, attr_name):
                            attr_val = getattr(imported_vdw, attr_name)
                            if isinstance(attr_val, dict):
                                return attr_val
            except ImportError:
                pass
            
            # Fallback to default van der Waals radii
            print("Warning: Could not import vdW_radii, using defaults")
            return {
                'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
                'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80,
                'S': 1.80, 'Cl': 1.75, 'K': 2.75, 'Ca': 2.31, 'Br': 1.85,
                'I': 1.98
            }
        
        vdw_radii = get_vdw_radii(self)
        
        n_ions = min(len(ions), max_ions)
        fig = plt.figure(figsize=(5*n_ions, 6))
        fig.patch.set_facecolor('white')
        
        print(f"Plotting {n_ions} {ion_type} ions ({actual_category}) with coordination radius {r0:.2f} Å")
        
        for i in range(n_ions):
            ax = fig.add_subplot(1, n_ions, i+1, projection='3d')
            
            # Get polyhedron for this ion
            self.universe.trajectory[frame]
            ion = ions[i]
            
            shell = self.universe.select_atoms(f'(sphzone {r0} index {ion.index})')
            
            # Handle unwrapping - use method from either class
            if hasattr(self, '_unwrap_shell_optimized'):
                # Optimized class method
                pos = self._unwrap_shell_optimized(ion.position, shell.positions)
            elif hasattr(self, '_unwrap_shell'):
                # Original class method
                pos = self._unwrap_shell(ion, r0)
            else:
                # Fallback - simple unwrapping
                dims = self.universe.dimensions[:3]
                dist = ion.position - shell.positions
                correction = np.where(np.abs(dist) > dims/2, np.sign(dist) * dims, 0)
                pos = shell.positions + correction
            
            shell.positions = pos
            
            # Handle point generation - use method from either class
            if hasattr(self, '_points_on_atomic_radius_optimized'):
                # Optimized class method
                surface_points = self._points_on_atomic_radius_optimized(shell.positions, n_points=150)
            elif hasattr(self, '_points_on_atomic_radius'):
                # Original class method
                surface_points = self._points_on_atomic_radius(shell, n_points=150)
            else:
                # Fallback - simple sphere point generation
                n_atoms = len(shell.positions)
                n_points = 150
                rng = np.random.default_rng(42)
                
                theta = np.arccos(rng.uniform(-1, 1, (n_atoms, n_points)))
                phi = rng.uniform(0, 2*np.pi, (n_atoms, n_points))
                radius = 1.5
                
                x = radius * np.sin(theta) * np.cos(phi) + shell.positions[:, 0, None]
                y = radius * np.sin(theta) * np.sin(phi) + shell.positions[:, 1, None]
                z = radius * np.cos(theta) + shell.positions[:, 2, None]
                
                surface_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            
            hull = ConvexHull(surface_points)
            
            # Plot polyhedron with more transparency to see cross-sections and atoms better
            faces = []
            for simplex in hull.simplices:
                face = surface_points[simplex]
                faces.append(face)
            
            poly3d = Poly3DCollection(faces, alpha=0.1, facecolor='lightcyan', 
                                    edgecolor='steelblue', linewidth=0.5)
            ax.add_collection3d(poly3d)
            
            # Plot water molecules and other atoms within the shell with van der Waals radii
            waters_in_shell = shell.select_atoms('name OW or name Ow')  # water oxygens
            hydrogens_in_shell = shell.select_atoms('name HW1 or name HW2 or name HW or name Hw or name H or name H1 or name H2')  # hydrogens
            
            # FIXED: Plot water oxygens with safe element access
            if len(waters_in_shell) > 0:
                # Get van der Waals radii for oxygen atoms and scale for plotting
                o_radii = []
                for atom in waters_in_shell:
                    # Try different ways to get element
                    if hasattr(atom, 'element') and atom.element:
                        element = atom.element
                    elif hasattr(atom, 'name') and atom.name:
                        # Try to extract element from name (e.g., 'OW' -> 'O')
                        element = atom.name[0] if atom.name else 'O'
                    else:
                        element = 'O'  # Default to oxygen
                    
                    radius = vdw_radii.get(element, 1.5)  # Default 1.5 for oxygen
                    o_radii.append(radius)
                
                o_radii = np.array(o_radii)
                o_sizes = (o_radii ** 2) * atom_scale_factor  # Use squared radii for better visual scaling
                
                ax.scatter(waters_in_shell.positions[:,0], waters_in_shell.positions[:,1], 
                        waters_in_shell.positions[:,2], c='red', s=o_sizes, marker='o', 
                        alpha=0.8, edgecolors='darkred', linewidth=1, zorder=10)
            
            # FIXED: Plot hydrogens with safe element access
            if len(hydrogens_in_shell) > 0:
                # Get van der Waals radii for hydrogen atoms and scale for plotting
                h_radii = []
                for atom in hydrogens_in_shell:
                    # Try different ways to get element
                    if hasattr(atom, 'element') and atom.element:
                        element = atom.element
                    elif hasattr(atom, 'name') and atom.name:
                        # Try to extract element from name (e.g., 'HW1' -> 'H')
                        element = atom.name[0] if atom.name else 'H'
                    else:
                        element = 'H'  # Default to hydrogen
                    
                    radius = vdw_radii.get(element, 1.2)  # Default 1.2 for hydrogen
                    h_radii.append(radius)
                
                h_radii = np.array(h_radii)
                h_sizes = (h_radii ** 2) * atom_scale_factor  # Use squared radii for better visual scaling
                
                ax.scatter(hydrogens_in_shell.positions[:,0], hydrogens_in_shell.positions[:,1], 
                        hydrogens_in_shell.positions[:,2], c='white', s=h_sizes, marker='o', 
                        alpha=0.9, edgecolors='gray', linewidth=1, zorder=10)
            
            # FIXED: Plot ion with proper color and van der Waals size based on specific ion type
            # Try different ways to get ion element
            if hasattr(ion, 'element') and ion.element:
                ion_element_for_radius = ion.element
            elif hasattr(ion, 'name') and ion.name:
                # Try to extract element from name
                ion_element_for_radius = ion.name.strip('0123456789')  # Remove numbers
            else:
                ion_element_for_radius = ion_type  # Use ion_type as fallback
            
            ion_radius = vdw_radii.get(ion_element_for_radius, 2.0)
            ion_plot_size = max(ion_size, (ion_radius ** 2) * atom_scale_factor * 2)  # Ensure ion is visible
            
            # Color mapping for different ion types
            ion_color_map = {
                'Na': 'blue',
                'K': 'violet', 
                'Mg': 'darkgreen',
                'Ca': 'darkorange',
                'Li': 'magenta',
                'Cl': 'coral',
                'Br': 'darkorange',
                'F': 'red',
                'I': 'darkred'
            }
            
            # Determine ion color
            if ion_type in ion_color_map:
                ion_color = ion_color_map[ion_type]
            elif actual_category == 'cation':
                ion_color = 'purple'
            else:
                ion_color = 'orange'
            
            ax.scatter(ion.position[0], ion.position[1], ion.position[2], 
                    c=ion_color, s=ion_plot_size, marker='o', alpha=0.9, 
                    edgecolors='black', linewidth=1, zorder=12)
            
            # Calculate cross-sections
            def calculate_cross_section(center, normal, hull_points):
                edges = []
                for simplex in hull.simplices:
                    for s in range(len(simplex)):
                        edge = tuple(sorted((simplex[s], simplex[(s + 1) % len(simplex)])))
                        edges.append(edge)
                edges = list(set(edges))
                
                A, B, C, D = create_plane_from_point_and_normal(center, normal)
                
                intersection_points = []
                for edge in edges:
                    p1 = hull_points[edge[0]]
                    p2 = hull_points[edge[1]]
                    intersection_point = line_plane_intersection(p1, p2, A, B, C, D)
                    if intersection_point is not None:
                        intersection_points.append(intersection_point)
                
                if len(intersection_points) > 2:
                    intersection_points = np.array(intersection_points)
                    try:
                        projected_points, rot_mat, mean_point = project_to_plane(intersection_points)
                        intersection_hull = ConvexHull(projected_points)
                        return intersection_points, intersection_hull.volume, intersection_hull
                    except:
                        return None, 0, None
                else:
                    return None, 0, None
            
            # Process each cross-section
            colors = ['red', 'green', 'blue']
            normals = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
            names = ['YZ', 'XZ', 'XY']
            cross_section_areas = []
            
            for j, (normal, color, name) in enumerate(zip(normals, colors, names)):
                intersection_points, area, intersection_hull = calculate_cross_section(ion.position, normal, surface_points)
                cross_section_areas.append(area)
                
                if intersection_points is not None and intersection_hull is not None:
                    
                    # Create filled cross-section polygon (transparent) - NO INTERNAL LINES
                    boundary_points = intersection_points[intersection_hull.vertices]
                    
                    # Create a filled polygon by triangulating from a center point
                    center_point = np.mean(boundary_points, axis=0)
                    cross_section_faces = []
                    
                    for k in range(len(boundary_points)):
                        # Create triangles from center to each edge of the boundary
                        p1 = center_point
                        p2 = boundary_points[k]
                        p3 = boundary_points[(k + 1) % len(boundary_points)]
                        cross_section_faces.append([p1, p2, p3])
                    
                    # Add the filled cross-section with transparent color - NO EDGE LINES
                    cross_section_collection = Poly3DCollection(cross_section_faces, alpha=0.3, 
                                                            facecolor=color, edgecolor='none', 
                                                            linewidth=0, zorder=5)
                    ax.add_collection3d(cross_section_collection)
                    
                    # Plot boundary lines only if boundary_linewidth > 0 and not False
                    if boundary_linewidth and boundary_linewidth > 0:
                        boundary_closed = np.vstack([boundary_points, boundary_points[0]])
                        
                        # Black outline (slightly thicker)
                        ax.plot(boundary_closed[:,0], boundary_closed[:,1], boundary_closed[:,2], 
                            color='black', linewidth=boundary_linewidth * 1.5, alpha=0.9, 
                            solid_capstyle='round', zorder=8)
                        # Colored line on top
                        ax.plot(boundary_closed[:,0], boundary_closed[:,1], boundary_closed[:,2], 
                            color=color, linewidth=boundary_linewidth, alpha=0.9, 
                            solid_capstyle='round', zorder=9)
                    
                    # OPTIONAL: Add only hull vertex markers if requested
                    if show_edge_markers:
                        # Mark only hull vertices with chosen marker type
                        hull_vertices_3d = intersection_points[intersection_hull.vertices]
                        ax.scatter(hull_vertices_3d[:,0], hull_vertices_3d[:,1], hull_vertices_3d[:,2], 
                                c=color, s=marker_size, marker=marker_type, alpha=0.9, 
                                edgecolors='black', linewidth=2, zorder=11)
            
            # Hide axes completely
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
            
            # Clean title with ion-specific information
            n_waters = len(waters_in_shell)
            n_hydrogens = len(hydrogens_in_shell)
            areas_text = f"R:{cross_section_areas[0]:.1f} G:{cross_section_areas[1]:.1f} B:{cross_section_areas[2]:.1f}"
            
            # Show ion element if available
            ion_element_for_display = getattr(ion, 'element', ion_type) if hasattr(ion, 'element') else ion_type
            ax.set_title(f'{ion_element_for_display} {i+1} (r₀={r0:.1f}Å)\nVol: {hull.volume:.1f} Å³ | {n_waters}W {n_hydrogens}H\n' +
                        f'Areas (Å²): {areas_text}', 
                        fontsize=9, pad=20)
            
            # Set viewing parameters
            all_points = surface_points
            max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 1.5
            center = ion.position
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
            
            ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        
        # Determine class type for filename
        class_name = self.__class__.__name__
        suffix = "_with_markers" if show_edge_markers else "_clean"
        plt.savefig(f'polyhedra_cross_sections{suffix}_{ion_type}_frame_{frame}_{class_name}.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()





    def plot_cross_sections_with_hull_markers(self, frame=0, ion_type='cation', max_ions=4,
                                            marker_size=40, ion_size=250, marker_type='o', 
                                            atom_scale_factor=100):
        """Plot cross-sections with hull vertex markers"""
        self.plot_cross_sections_enhanced_edges_no_lines(
            frame=frame, ion_type=ion_type, max_ions=max_ions, show_edge_markers=True,
            boundary_linewidth=0, marker_size=marker_size, ion_size=ion_size,
            marker_type=marker_type, atom_scale_factor=atom_scale_factor
        )

        def __repr__(self):
            return f'EquilibriumAnalysisOptimized object with {len(self.waters)} waters, {len(self.cations)} cations, and {len(self.anions)} anions over {self.n_frames} frames'



     
    def plot_clean_cross_sections(self, frame=0, ion_type='cation', max_ions=4,
                                boundary_linewidth=0.5, ion_size=200, atom_scale_factor=100,
                                figsize_per_ion=(5, 6),
                                show_title=True,
                                title_font_size=9, title_font_weight='normal',
                                title_pad=1.0,
                                elev=25, azim=45,
                                dpi=300, output_filename=None,
                                save_combined=True, save_individual=False,
                                individual_pad=0.05,
                                transparent_bg=True):
        """
        Enhanced version with filled cross-sections, atoms sized by van der Waals radii, and optional hull vertex markers
        Now supports ion-specific analysis (e.g., 'Na', 'Mg', 'Cl') in addition to broad categories
        FIXED: Now shows coions within coordination shell using circles with proper colors and VdW sizing
        """
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from scipy.spatial import ConvexHull
        
        # Import utilities - handle different import paths
        try:
            from utils.linear_algebra import create_plane_from_point_and_normal, line_plane_intersection, project_to_plane
        except ImportError:
            # Try alternative import path for optimized class
            sys.path.append('/Users/roev0007/Documents/solvation_shells')
            from utils.linear_algebra import create_plane_from_point_and_normal, line_plane_intersection, project_to_plane
        
        # CRITICAL FIX: Set trajectory frame FIRST and ensure it's properly set
        print(f"Setting trajectory to frame {frame}")
        self.universe.trajectory[frame]
        # Force a refresh of the trajectory state
        current_frame = self.universe.trajectory.frame
        if current_frame != frame:
            print(f"Warning: Trajectory frame mismatch. Requested {frame}, got {current_frame}")
            self.universe.trajectory[frame]  # Try again
        
        print(f"Confirmed trajectory frame: {self.universe.trajectory.frame}")
        
        # Handle both broad categories and specific ion types
        def get_ions_and_category(eq_analysis, ion_type):
            """Get ions and determine category based on ion_type"""
            
            # Handle broad categories (original behavior)
            if ion_type in ['cation', 'anion']:
                ions = eq_analysis.cations if ion_type == 'cation' else eq_analysis.anions
                return ions, ion_type
            
            # Handle specific ion types (new behavior)
            else:
                # Check if optimized class with ion-type-specific methods
                if hasattr(eq_analysis, '_get_unique_ion_types'):
                    cation_types = eq_analysis._get_unique_ion_types(eq_analysis.cations)
                    anion_types = eq_analysis._get_unique_ion_types(eq_analysis.anions)
                    
                    if ion_type in cation_types:
                        return cation_types[ion_type], 'cation'
                    elif ion_type in anion_types:
                        return anion_types[ion_type], 'anion'
                    else:
                        available_types = list(cation_types.keys()) + list(anion_types.keys())
                        raise ValueError(f"Ion type '{ion_type}' not found. Available types: {available_types}")
                else:
                    # Fallback for original class - filter by element/name
                    all_cations = eq_analysis.cations
                    all_anions = eq_analysis.anions
                    
                    # Try to filter by element first, then by name
                    cation_matches = []
                    anion_matches = []
                    
                    for atom in all_cations:
                        if (hasattr(atom, 'element') and atom.element == ion_type) or \
                        (hasattr(atom, 'name') and atom.name.startswith(ion_type)):
                            cation_matches.append(atom)
                    
                    for atom in all_anions:
                        if (hasattr(atom, 'element') and atom.element == ion_type) or \
                        (hasattr(atom, 'name') and atom.name.startswith(ion_type)):
                            anion_matches.append(atom)
                    
                    if cation_matches:
                        # Create AtomGroup for cation matches
                        ions = eq_analysis.universe.atoms[np.array([atom.index for atom in cation_matches])]
                        return ions, 'cation'
                    elif anion_matches:
                        # Create AtomGroup for anion matches
                        ions = eq_analysis.universe.atoms[np.array([atom.index for atom in anion_matches])]
                        return ions, 'anion'
                    else:
                        raise ValueError(f"No atoms found matching ion type '{ion_type}'")
        
        # Get ions and determine actual category
        ions, actual_category = get_ions_and_category(self, ion_type)
        
        # Get radius based on specific ion type or category
        def get_coordination_radius(eq_analysis, ion_type, actual_category):
            """Get coordination radius for the specified ion type"""
            
            # Try ion-type-specific radius first (optimized class)
            if hasattr(eq_analysis, 'get_coordination_radius_by_type'):
                radius = eq_analysis.get_coordination_radius_by_type(ion_type)
                if radius is not None:
                    print(f"Using {ion_type}-specific coordination radius: {radius:.2f} Å")
                    return radius
            
            # Try ion-type-specific solutes
            if hasattr(eq_analysis, 'solutes_ci') and hasattr(eq_analysis, 'solutes_ai'):
                if actual_category == 'cation' and ion_type in eq_analysis.solutes_ci:
                    solute = eq_analysis.solutes_ci[ion_type]
                    if solute is not None and hasattr(solute, 'radii') and 'water' in solute.radii:
                        radius = solute.radii['water']
                        print(f"Using {ion_type} solute radius: {radius:.2f} Å")
                        return radius
                elif actual_category == 'anion' and ion_type in eq_analysis.solutes_ai:
                    solute = eq_analysis.solutes_ai[ion_type]
                    if solute is not None and hasattr(solute, 'radii') and 'water' in solute.radii:
                        radius = solute.radii['water']
                        print(f"Using {ion_type} solute radius: {radius:.2f} Å")
                        return radius
            
            # Fallback to broad category solutes
            try:
                if actual_category == 'cation':
                    radius = eq_analysis.solute_ci.radii['water']
                    print(f"Using general cation radius: {radius:.2f} Å")
                    return radius
                else:
                    radius = eq_analysis.solute_ai.radii['water']
                    print(f"Using general anion radius: {radius:.2f} Å")
                    return radius
            except (AttributeError, NameError):
                print("Solutes not initialized. Trying to initialize...")
                if hasattr(eq_analysis, 'initialize_Solutes'):
                    eq_analysis.initialize_Solutes()
                    if actual_category == 'cation':
                        return eq_analysis.solute_ci.radii['water']
                    else:
                        return eq_analysis.solute_ai.radii['water']
                else:
                    # Default fallback values
                    default_radii = {'cation': 2.8, 'anion': 3.5}
                    radius = default_radii[actual_category]
                    print(f"Using default {actual_category} radius: {radius:.2f} Å")
                    return radius
        
        r0 = get_coordination_radius(self, ion_type, actual_category)
        
        # Get vdW radii
        def get_vdw_radii(eq_analysis):
            """Get van der Waals radii dictionary"""
            
            # Try to get from the class first
            if hasattr(eq_analysis, 'vdW_radii'):
                vdw_obj = eq_analysis.vdW_radii
                # Check if it's already a dictionary
                if isinstance(vdw_obj, dict):
                    return vdw_obj
                # Check if it has a dictionary attribute
                elif hasattr(vdw_obj, '__dict__'):
                    # Look for common dictionary attribute names
                    for attr_name in ['radii', 'vdw_radii', 'data', '__dict__']:
                        if hasattr(vdw_obj, attr_name):
                            attr_val = getattr(vdw_obj, attr_name)
                            if isinstance(attr_val, dict):
                                return attr_val
            
            # Try to import vdW radii directly
            try:
                from utils.file_rw import vdW_radii as imported_vdw
                # Check if imported object is a dictionary
                if isinstance(imported_vdw, dict):
                    return imported_vdw
                # Check if it has dictionary attributes
                elif hasattr(imported_vdw, '__dict__'):
                    for attr_name in ['radii', 'vdw_radii', 'data']:
                        if hasattr(imported_vdw, attr_name):
                            attr_val = getattr(imported_vdw, attr_name)
                            if isinstance(attr_val, dict):
                                return attr_val
            except ImportError:
                pass
            
            # Fallback to default van der Waals radii
            print("Warning: Could not import vdW_radii, using defaults")
            return {
                'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
                'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80,
                'S': 1.80, 'Cl': 1.75, 'K': 2.75, 'Ca': 2.31, 'Br': 1.85,
                'I': 1.98
            }
        
        vdw_radii = get_vdw_radii(self)
        
        n_ions = min(len(ions), max_ions)
        # When there is no title, shrink figure height so the 3D content fills it
        fig_h = figsize_per_ion[1] if show_title else figsize_per_ion[1] * 0.80
        fig = plt.figure(figsize=(figsize_per_ion[0] * n_ions, fig_h))
        
        print(f"Plotting {n_ions} {ion_type} ions ({actual_category}) with coordination radius {r0:.2f} Å")
        
        for i in range(n_ions):
            ax = fig.add_subplot(1, n_ions, i+1, projection='3d')
            
            # CRITICAL: Re-confirm frame before each ion analysis
            if self.universe.trajectory.frame != frame:
                print(f"  Frame drift detected for ion {i+1}, resetting to frame {frame}")
                self.universe.trajectory[frame]
            
            ion = ions[i]
            
            print(f"  Processing ion {i+1} at frame {self.universe.trajectory.frame}")
            
            # Use the original coordination radius - don't change the search radius
            shell = self.universe.select_atoms(f'(sphzone {r0} index {ion.index})')
            print(f"    Shell has {len(shell)} atoms total")
            
            # Handle unwrapping - use method from either class
            if hasattr(self, '_unwrap_shell_optimized'):
                # Optimized class method
                pos = self._unwrap_shell_optimized(ion.position, shell.positions)
            elif hasattr(self, '_unwrap_shell'):
                # Original class method
                pos = self._unwrap_shell(ion, r0)
            else:
                # Fallback - simple unwrapping
                dims = self.universe.dimensions[:3]
                dist = ion.position - shell.positions
                correction = np.where(np.abs(dist) > dims/2, np.sign(dist) * dims, 0)
                pos = shell.positions + correction
            
            shell.positions = pos
            
            # Handle point generation - use method from either class
            if hasattr(self, '_points_on_atomic_radius_optimized'):
                # Optimized class method
                surface_points = self._points_on_atomic_radius_optimized(shell.positions, n_points=150)
            elif hasattr(self, '_points_on_atomic_radius'):
                # Original class method
                surface_points = self._points_on_atomic_radius(shell, n_points=150)
            else:
                # Fallback - simple sphere point generation
                n_atoms = len(shell.positions)
                n_points = 150
                rng = np.random.default_rng(42)
                
                theta = np.arccos(rng.uniform(-1, 1, (n_atoms, n_points)))
                phi = rng.uniform(0, 2*np.pi, (n_atoms, n_points))
                radius = 1.5
                
                x = radius * np.sin(theta) * np.cos(phi) + shell.positions[:, 0, None]
                y = radius * np.sin(theta) * np.sin(phi) + shell.positions[:, 1, None]
                z = radius * np.cos(theta) + shell.positions[:, 2, None]
                
                surface_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            
            hull = ConvexHull(surface_points)
            
            # Plot polyhedron with more transparency to see cross-sections and atoms better
            faces = []
            for simplex in hull.simplices:
                face = surface_points[simplex]
                faces.append(face)
            
            poly3d = Poly3DCollection(faces, alpha=0.1, facecolor='lightcyan', 
                                    edgecolor='steelblue', linewidth=0.5)
            ax.add_collection3d(poly3d)
            
            # Plot ALL atoms within the shell
            waters_in_shell = shell.select_atoms('name OW or name Ow')  # water oxygens
            hydrogens_in_shell = shell.select_atoms('name HW1 or name HW2 or name HW or name Hw or name H or name H1 or name H2')  # hydrogens
            
            print(f"    Waters in shell: {len(waters_in_shell)}")
            print(f"    Hydrogens in shell: {len(hydrogens_in_shell)}")
            
            # FIXED: Get coions without using selection_string
            print(f"    Looking for coions in shell for {ion_type} (category: {actual_category})")
            print(f"    Ion position: {ion.position}")
            print(f"    Using coordination radius: {r0:.2f} Å")
            
            if actual_category == 'cation':
                # For cations, coions are anions - intersect shell with anions
                coions_in_shell = shell & self.anions
                print(f"    Checking cation {ion_type}, looking for anion coions")
            else:
                # For anions, coions are cations - intersect shell with cations
                coions_in_shell = shell & self.cations
                print(f"    Checking anion {ion_type}, looking for cation coions")
            
            print(f"    Coions found in shell: {len(coions_in_shell)}")
            
            # Enhanced debugging if no coions found
            if len(coions_in_shell) == 0:
                print(f"    DEBUG: No coions found. Enhanced debugging:")
                print(f"      Total shell atoms: {len(shell)}")
                print(f"      Shell atom types: {set(shell.types) if hasattr(shell, 'types') else 'N/A'}")
                print(f"      Shell atom names: {set(shell.names)}")
                
                # Check distances manually
                all_system_ions = self.cations + self.anions
                distances_to_ion = np.linalg.norm(all_system_ions.positions - ion.position, axis=1)
                nearby_ions = all_system_ions[distances_to_ion <= r0]
                
                print(f"      Manual distance check: {len(nearby_ions)} ions within {r0:.2f} Å")
                if len(nearby_ions) > 0:
                    print(f"      Nearby ion names: {set(nearby_ions.names)}")
                    print(f"      Nearby ion distances: {distances_to_ion[distances_to_ion <= r0]}")
            
            # Plot water oxygens with safe element access
            if len(waters_in_shell) > 0:
                # Get van der Waals radii for oxygen atoms and scale for plotting
                o_radii = []
                for atom in waters_in_shell:
                    # Try different ways to get element
                    if hasattr(atom, 'element') and atom.element:
                        element = atom.element
                    elif hasattr(atom, 'name') and atom.name:
                        # Try to extract element from name (e.g., 'OW' -> 'O')
                        element = atom.name[0] if atom.name else 'O'
                    else:
                        element = 'O'  # Default to oxygen
                    
                    radius = vdw_radii.get(element, 1.5)  # Default 1.5 for oxygen
                    o_radii.append(radius)
                
                o_radii = np.array(o_radii)
                o_sizes = (o_radii ** 2) * atom_scale_factor  # Use squared radii for better visual scaling
                
                ax.scatter(waters_in_shell.positions[:,0], waters_in_shell.positions[:,1], 
                        waters_in_shell.positions[:,2], c='red', s=o_sizes, marker='o', 
                        alpha=0.8, edgecolors='darkred', linewidth=1, zorder=10)
            
            # Plot hydrogens with safe element access
            if len(hydrogens_in_shell) > 0:
                # Get van der Waals radii for hydrogen atoms and scale for plotting
                h_radii = []
                for atom in hydrogens_in_shell:
                    # Try different ways to get element
                    if hasattr(atom, 'element') and atom.element:
                        element = atom.element
                    elif hasattr(atom, 'name') and atom.name:
                        # Try to extract element from name (e.g., 'HW1' -> 'H')
                        element = atom.name[0] if atom.name else 'H'
                    else:
                        element = 'H'  # Default to hydrogen
                    
                    radius = vdw_radii.get(element, 1.2)  # Default 1.2 for hydrogen
                    h_radii.append(radius)
                
                h_radii = np.array(h_radii)
                h_sizes = (h_radii ** 2) * atom_scale_factor  # Use squared radii for better visual scaling
                
                ax.scatter(hydrogens_in_shell.positions[:,0], hydrogens_in_shell.positions[:,1], 
                        hydrogens_in_shell.positions[:,2], c='white', s=h_sizes, marker='o', 
                        alpha=0.9, edgecolors='gray', linewidth=1, zorder=10)
            
            # Plot coions within the shell
            if len(coions_in_shell) > 0:
                print(f"    Found {len(coions_in_shell)} coions within shell for ion {i+1}")
                
                # Get van der Waals radii for coions and scale for plotting
                coion_radii = []
                coion_colors = []
                coion_elements = []
                
                for atom in coions_in_shell:
                    # Try different ways to get element
                    if hasattr(atom, 'element') and atom.element:
                        element = atom.element
                    elif hasattr(atom, 'name') and atom.name:
                        # Try to extract element from name
                        element = atom.name.strip('0123456789')  # Remove numbers
                    else:
                        element = 'X'  # Default unknown
                    
                    coion_elements.append(element)
                    radius = vdw_radii.get(element, 2.0)
                    coion_radii.append(radius)
                    
                    # Use the same color mapping as central ions
                    ion_color_map = {
                        'Na': 'blue',
                        'K': 'violet', 
                        'Mg': 'darkgreen',
                        'Ca': 'darkorange',
                        'Li': 'magenta',
                        'Cl': 'coral',
                        'Br': 'navy',
                        'F': 'red',
                        'I': 'darkred'
                    }
                    
                    # Determine coion color using the same logic as central ions
                    if element in ion_color_map:
                        color = ion_color_map[element]
                    else:
                        # Fallback based on typical ion categories
                        if element in ['Na', 'K', 'Mg', 'Ca', 'Li']:
                            color = 'purple'  # Default cation color
                        elif element in ['Cl', 'Br', 'F', 'I']:
                            color = 'orange'  # Default anion color
                        else:
                            # If we can't determine, use the opposite of the central ion
                            if actual_category == 'cation':
                                color = 'orange'  # Coions are likely anions
                            else:
                                color = 'purple'  # Coions are likely cations
                    
                    coion_colors.append(color)
                
                coion_radii = np.array(coion_radii)
                # Use full van der Waals radii without scaling down
                coion_sizes = []
                for radius in coion_radii:
                    # Use the same calculation as for central ions - no scaling
                    coion_size = max(ion_size, (radius ** 2) * atom_scale_factor * 2)
                    coion_sizes.append(coion_size)
                
                # Plot coions with circles using full van der Waals sizes
                ax.scatter(coions_in_shell.positions[:,0], coions_in_shell.positions[:,1], 
                        coions_in_shell.positions[:,2], c=coion_colors, s=coion_sizes, 
                        marker='o', alpha=0.9, edgecolors='black', linewidth=1.5, zorder=11)
                
                # Print what coions we found
                unique_coions = list(set(coion_elements))
                coion_counts = {elem: coion_elements.count(elem) for elem in unique_coions}
                print(f"      Coion breakdown: {coion_counts}")
                
                # Print color assignments for debugging
                for element, color in zip(set(coion_elements), set(coion_colors)):
                    print(f"      {element} coions plotted in {color}")
            else:
                print(f"    No coions found in shell for ion {i+1}")
            
            # Plot central ion with proper color and van der Waals size based on specific ion type
            # Try different ways to get ion element
            if hasattr(ion, 'element') and ion.element:
                ion_element_for_radius = ion.element
            elif hasattr(ion, 'name') and ion.name:
                # Try to extract element from name
                ion_element_for_radius = ion.name.strip('0123456789')  # Remove numbers
            else:
                ion_element_for_radius = ion_type  # Use ion_type as fallback
            
            ion_radius = vdw_radii.get(ion_element_for_radius, 2.0)
            ion_plot_size = max(ion_size, (ion_radius ** 2) * atom_scale_factor * 2)  # Ensure ion is visible
            
            # Color mapping for different ion types
            ion_color_map = {
                'Na': 'blue',
                'K': 'violet', 
                'Mg': 'darkgreen',
                'Ca': 'darkorange',
                'Li': 'magenta',
                'Cl': 'coral',
                'Br': 'darkorange',
                'F': 'red',
                'I': 'darkred'
            }
            
            # Determine ion color
            if ion_type in ion_color_map:
                ion_color = ion_color_map[ion_type]
            elif actual_category == 'cation':
                ion_color = 'purple'
            else:
                ion_color = 'orange'
            
            ax.scatter(ion.position[0], ion.position[1], ion.position[2], 
                    c=ion_color, s=ion_plot_size, marker='o', alpha=0.9, 
                    edgecolors='black', linewidth=1, zorder=12)
            
            # Calculate cross-sections
            def calculate_cross_section(center, normal, hull_points):
                edges = []
                for simplex in hull.simplices:
                    for s in range(len(simplex)):
                        edge = tuple(sorted((simplex[s], simplex[(s + 1) % len(simplex)])))
                        edges.append(edge)
                edges = list(set(edges))
                
                A, B, C, D = create_plane_from_point_and_normal(center, normal)
                
                intersection_points = []
                for edge in edges:
                    p1 = hull_points[edge[0]]
                    p2 = hull_points[edge[1]]
                    intersection_point = line_plane_intersection(p1, p2, A, B, C, D)
                    if intersection_point is not None:
                        intersection_points.append(intersection_point)
                
                if len(intersection_points) > 2:
                    intersection_points = np.array(intersection_points)
                    try:
                        projected_points, rot_mat, mean_point = project_to_plane(intersection_points)
                        intersection_hull = ConvexHull(projected_points)
                        return intersection_points, projected_points[intersection_hull.vertices].shape[0] * 0.5, intersection_hull
                    except:
                        return None, 0, None
                else:
                    return None, 0, None
            
            # Process each cross-section
            colors = ['red', 'green', 'blue']
            normals = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
            names = ['YZ', 'XZ', 'XY']
            cross_section_areas = []
            
            for j, (normal, color, name) in enumerate(zip(normals, colors, names)):
                intersection_points, area, intersection_hull = calculate_cross_section(ion.position, normal, surface_points)
                cross_section_areas.append(area)
                
                if intersection_points is not None and intersection_hull is not None:
                    
                    # Create filled cross-section polygon (transparent) - NO INTERNAL LINES
                    boundary_points = intersection_points[intersection_hull.vertices]
                    
                    # Create a filled polygon by triangulating from a center point
                    center_point = np.mean(boundary_points, axis=0)
                    cross_section_faces = []
                    
                    for k in range(len(boundary_points)):
                        # Create triangles from center to each edge of the boundary
                        p1 = center_point
                        p2 = boundary_points[k]
                        p3 = boundary_points[(k + 1) % len(boundary_points)]
                        cross_section_faces.append([p1, p2, p3])
                    
                    # Add the filled cross-section with transparent color - NO EDGE LINES
                    cross_section_collection = Poly3DCollection(cross_section_faces, alpha=0.3, 
                                                            facecolor=color, edgecolor='none', 
                                                            linewidth=0, zorder=5)
                    ax.add_collection3d(cross_section_collection)
                    
                    # Plot boundary lines only if boundary_linewidth > 0 and not False
                    if boundary_linewidth and boundary_linewidth > 0:
                        boundary_closed = np.vstack([boundary_points, boundary_points[0]])
                        
                        # Black outline (slightly thicker)
                        ax.plot(boundary_closed[:,0], boundary_closed[:,1], boundary_closed[:,2], 
                            color='black', linewidth=boundary_linewidth * 1.5, alpha=0.9, 
                            solid_capstyle='round', zorder=8)
                        # Colored line on top
                        ax.plot(boundary_closed[:,0], boundary_closed[:,1], boundary_closed[:,2], 
                            color=color, linewidth=boundary_linewidth, alpha=0.9, 
                            solid_capstyle='round', zorder=9)
            
            # Hide axes completely
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
            
            # Clean title with ion-specific information including coions
            n_waters = len(waters_in_shell)
            n_hydrogens = len(hydrogens_in_shell)
            n_coions = len(coions_in_shell)
            areas_text = f"R:{cross_section_areas[0]:.1f} G:{cross_section_areas[1]:.1f} B:{cross_section_areas[2]:.1f}"
            
            # Show ion element if available
            ion_element_for_display = getattr(ion, 'element', ion_type) if hasattr(ion, 'element') else ion_type
            
            # Create composition string
            composition_parts = [f"{n_waters}W"]
            if n_coions > 0:
                composition_parts.append(f"{n_coions}I")  # I for ions (coions)
            if n_hydrogens > 0:
                composition_parts.append(f"{n_hydrogens}H")
            
            composition_str = " ".join(composition_parts)
            
            if show_title:
                ax.set_title(f'{ion_element_for_display} {i+1} (r₀={r0:.1f}Å)\nVol: {hull.volume:.1f} Å³ | {composition_str}\n' +
                            f'Areas (Å²): {areas_text}',
                            fontsize=title_font_size, fontweight=title_font_weight,
                            y=title_pad)
            
            # Set viewing parameters
            all_points = surface_points
            max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 1.5
            center = ion.position
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
            
            ax.view_init(elev=elev, azim=azim)
        
        fig.subplots_adjust(top=0.88 if show_title else 0.99,
                            bottom=0.01, left=0.01, right=0.99, wspace=0.02)

        import os
        class_name = self.__class__.__name__
        auto_name = f'polyhedra_cross_sections_clean_{ion_type}_frame_{frame}_{class_name}.png'

        if save_combined:
            fname = output_filename if output_filename is not None else auto_name
            fig.savefig(fname, dpi=dpi, bbox_inches='tight', transparent=transparent_bg)
            print(f"Combined figure saved: {fname}")

        if save_individual:
            fig.canvas.draw()
            from matplotlib.transforms import Bbox
            renderer = fig.canvas.get_renderer()
            fig_w, fig_h = fig.get_size_inches()
            # axes positions in normalised figure coords (0-1)
            positions = [fig.axes[i].get_position() for i in range(n_ions)]
            for idx in range(n_ions):
                ax_s = fig.axes[idx]
                tight = ax_s.get_tightbbox(renderer)
                if tight is None:
                    tight = ax_s.get_window_extent(renderer)
                # convert from display pixels → inches
                tight_inch = tight.transformed(fig.dpi_scale_trans.inverted())
                pos = positions[idx]
                # column clip: midpoints between adjacent axes prevent neighbour bleed
                col_left  = ((positions[idx - 1].x1 + pos.x0) / 2.0 * fig_w
                             if idx > 0 else 0.0)
                col_right = ((pos.x1 + positions[idx + 1].x0) / 2.0 * fig_w
                             if idx < n_ions - 1 else fig_w)
                x0 = max(tight_inch.x0, col_left)  - individual_pad
                x1 = min(tight_inch.x1, col_right) + individual_pad
                y0 = tight_inch.y0 - individual_pad
                y1 = tight_inch.y1 + individual_pad
                # clamp left/right to column bounds; do NOT clamp top so the title is never cut
                x0, x1 = max(x0, 0.0), min(x1, fig_w)
                y0 = max(y0, 0.0)
                bbox_ind = Bbox([[x0, y0], [x1, y1]])
                if output_filename is not None:
                    base, ext = os.path.splitext(output_filename)
                    ext = ext if ext else '.png'
                    ind_fname = f'{base}_{ion_type}_{idx+1}{ext}'
                else:
                    ind_fname = f'polyhedra_cross_sections_clean_{ion_type}_frame_{frame}_{class_name}_ion{idx+1}.png'
                fig.savefig(ind_fname, dpi=dpi, bbox_inches=bbox_ind, transparent=transparent_bg)
                print(f"  Individual figure saved: {ind_fname}")

        plt.show()



    def _find_peaks_wrapper(self, bins, data, **kwargs):
        '''Wrapper for scipy.signal.find_peaks to use with SolvationAnalysis to find cutoff'''
        
        peaks, _  = find_peaks(-data, **kwargs)
        radii = bins[peaks[0]]
        return radii


    def initialize_Solutes_by_type(self, step=1, plot_cutoffs=False, use_cache=True, force_recalc=False, njobs=None):
        '''
        Initialize separate Solute objects for each ion type with caching and parallel processing support.
        FIXED: Now handles systems with no anions or no cations gracefully
        FIXED: Don't pass None for coions - exclude the key entirely
        
        Parameters
        ----------
        step : int
            Step size for analysis, default=1
        plot_cutoffs : bool
            Whether to plot cutoff determination, default=False
        use_cache : bool
            Whether to use cached results, default=True
        force_recalc : bool
            Force recalculation even if cache exists, default=False
        njobs : int, optional
            Number of parallel jobs. If None, uses auto-tuned value. Use 1 for sequential.
        '''
        
        # Use auto-tuned default if not specified
        if njobs is None:
            njobs = self.default_njobs
        
        # Use all CPUs if -1
        if njobs == -1:
            njobs = min(multiprocessing.cpu_count(), 8)  # Cap to avoid issues
        
        # Check if already initialized and not forcing recalculation
        # Done BEFORE _test_multiprocessing_compatibility to avoid spawning test
        # Pools on every call when the result is already cached.
        if (hasattr(self, 'solutes_ci') and hasattr(self, 'solutes_ai') and 
            use_cache and not force_recalc):
            print("Solutes already initialized. Use force_recalc=True to recalculate.")
            return self.solutes_ci, self.solutes_ai
        
        # Ensure dicts exist from this point on so a future force_recalc=False
        # call won't fall through if this run is interrupted later.
        if not hasattr(self, 'solutes_ci'):
            self.solutes_ci = {}
        if not hasattr(self, 'solutes_ai'):
            self.solutes_ai = {}
        
        # Test multiprocessing compatibility
        njobs = self._test_multiprocessing_compatibility(njobs)
        
        # Get unique ion types
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        # FIXED: Check if system has any ions at all
        if not cation_types and not anion_types:
            print("Error: No ions found in system!")
            print(f"  Cations: {len(self.cations)} atoms")
            print(f"  Anions: {len(self.anions)} atoms")
            print("Check your ion selections in the Universe initialization.")
            return None, None
        
        print(f"Initializing Solutes for multiple ion types:")
        print(f"  Cation types: {list(cation_types.keys()) if cation_types else 'None'}")
        print(f"  Anion types: {list(anion_types.keys()) if anion_types else 'None'}")
        print(f"  Using {njobs} CPU(s) for parallel processing")
        
        # Prepare tasks for parallel processing
        tasks = []
        
        # Add cation tasks (only if we have cations)
        if cation_types:
            for cation_name, cation_group in cation_types.items():
                # Skip if already cached and not forcing recalculation
                if (cation_name in self.solutes_ci and 
                    self.solutes_ci[cation_name] is not None and 
                    use_cache and not force_recalc):
                    print(f"  Skipping {cation_name} (already cached)")
                    continue
                
                # FIXED: Build solvents dict properly - exclude coion if no anions
                solvents = {'water': self.waters}
                if len(self.anions) > 0:
                    solvents['coion'] = self.anions
                
                tasks.append({
                    'ion_type': cation_name,
                    'ion_category': 'cation',
                    'ion_group': cation_group,
                    'solvents': solvents,
                    'step': step
                })
        else:
            print("  Warning: No cations found in system - skipping cation solute initialization")
        
        # Add anion tasks (only if we have anions)
        if anion_types:
            for anion_name, anion_group in anion_types.items():
                # Skip if already cached and not forcing recalculation
                if (anion_name in self.solutes_ai and 
                    self.solutes_ai[anion_name] is not None and 
                    use_cache and not force_recalc):
                    print(f"  Skipping {anion_name} (already cached)")
                    continue
                
                # FIXED: Build solvents dict properly - exclude coion if no cations
                solvents = {'water': self.waters}
                if len(self.cations) > 0:
                    solvents['coion'] = self.cations
                
                tasks.append({
                    'ion_type': anion_name,
                    'ion_category': 'anion',
                    'ion_group': anion_group,
                    'solvents': solvents,
                    'step': step
                })
        else:
            print("  Warning: No anions found in system - skipping anion solute initialization")
        
        if not tasks:
            print("All solutes already cached, nothing to calculate.")
            self._create_combined_solutes()
            return self.solutes_ci, self.solutes_ai
        
        print(f"  Processing {len(tasks)} ion types...")
        
        # Process tasks (parallel or sequential)
        if njobs == 1 or len(tasks) == 1:  # FIXED: Also check task count
            # Sequential processing
            print("Using sequential processing (single CPU or single ion type)")
            results = []
            for task in tqdm(tasks, desc="Initializing solutes"):
                results.append(self._create_solute_for_task(task))
        else:
            # Parallel processing
            print(f"  Using parallel processing with {njobs} workers")
            
            try:
                with Pool(njobs, initializer=_worker_init) as pool:
                    results = list(tqdm(
                        pool.imap(self._create_solute_for_task, tasks),
                        total=len(tasks),
                        desc="Initializing solutes"
                    ))
            except BaseException as e:
                print(f"Parallel processing failed: {e}")
                print("Falling back to sequential processing...")
                results = []
                for task in tqdm(tasks, desc="Initializing solutes"):
                    results.append(self._create_solute_for_task(task))
        
        # Store results
        for result in results:
            if result['success']:
                if result['ion_category'] == 'cation':
                    self.solutes_ci[result['ion_type']] = result['solute']
                    print(f"  ✓ {result['ion_type']} (cation) initialized")
                else:
                    self.solutes_ai[result['ion_type']] = result['solute']
                    print(f"  ✓ {result['ion_type']} (anion) initialized")
            else:
                print(f"  ✗ {result['ion_type']} ({result['ion_category']}) failed: {result['error']}")
        
        # Create combined solutes for backward compatibility
        self._create_combined_solutes()
        
        print("Multi-type Solute initialization complete!")
        return self.solutes_ci, self.solutes_ai


    def _create_solute_for_task(self, task):
        '''
        Worker function for parallel solute creation.
        
        Parameters
        ----------
        task : dict
            Task dictionary with ion_type, ion_category, ion_group, solvents, step
        
        Returns
        -------
        result : dict
            Result dictionary with success status, solute, and error info
        '''
        
        ion_type = task['ion_type']
        ion_category = task['ion_category']
        ion_group = task['ion_group']
        solvents = task['solvents']
        step = task['step']
        
        try:
            print(f"  Processing {ion_type} ({ion_category}, {len(ion_group)} ions)...")
            
            # Create solute
            solute = self._create_solute_optimized(ion_group, solvents, step)
            
            return {
                'success': True,
                'ion_type': ion_type,
                'ion_category': ion_category,
                'solute': solute,
                'error': None
            }
        
        except Exception as e:
            return {
                'success': False,
                'ion_type': ion_type,
                'ion_category': ion_category,
                'solute': None,
                'error': str(e)
            }


    def _create_solute_optimized(self, ion_group, solvents, step=1):
        '''
        Create a solute with user-specified step size - respects all step values.
        '''
        
        print(f"    Using step={step} for initialization")
        
        # Create solute
        solute = Solute.from_atoms(
            ion_group,
            solvents,
            rdf_kernel=self._find_peaks_wrapper,
            kernel_kwargs={'distance': 5}
        )
        
        # Run with user-specified step - no minimum enforcement
        solute.run(step=step)
        
        return solute

    def _create_combined_solutes(self):
        '''Create combined solutes for backward compatibility'''
        
        print(f"  Creating combined solutes for backward compatibility...")
        
        # Use the most abundant ion type that was successfully initialized
        successful_cations = {k: v for k, v in self.solutes_ci.items() if v is not None}
        successful_anions = {k: v for k, v in self.solutes_ai.items() if v is not None}
        
        if successful_cations:
            cation_types = self._get_unique_ion_types(self.cations)
            most_abundant_cation = max(successful_cations.keys(), 
                                    key=lambda x: len(cation_types[x]))
            self.solute_ci = successful_cations[most_abundant_cation]
            print(f"  Combined cation solute uses: {most_abundant_cation}")
        else:
            print("  Warning: No cation solutes successfully initialized")
            self.solute_ci = None
        
        if successful_anions:
            anion_types = self._get_unique_ion_types(self.anions)
            most_abundant_anion = max(successful_anions.keys(), 
                                    key=lambda x: len(anion_types[x]))
            self.solute_ai = successful_anions[most_abundant_anion]
            print(f"  Combined anion solute uses: {most_abundant_anion}")
        else:
            print("  Warning: No anion solutes successfully initialized")
            self.solute_ai = None

    def reinitialize_single_ion_solute(self, ion_type, step=1):
        '''
        Reinitialize solute for a single ion type without affecting others.
        
        Parameters
        ----------
        ion_type : str
            Ion type to reinitialize (e.g., 'Na', 'Mg', 'Cl')
        step : int
            Step size for analysis
        '''
        
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        if ion_type in cation_types:
            print(f"Reinitializing {ion_type} cation solute...")
            ion_group = cation_types[ion_type]
            solvents = {'water': self.waters, 'coion': self.anions}
            
            try:
                solute = self._create_solute_optimized(ion_group, solvents, step)
                self.solutes_ci[ion_type] = solute
                
                if hasattr(solute, 'radii') and 'water' in solute.radii:
                    print(f"  {ion_type}-water coordination radius: {solute.radii['water']:.2f} Å")
                    return solute
            except Exception as e:
                print(f"  Error: {e}")
                return None
                
        elif ion_type in anion_types:
            print(f"Reinitializing {ion_type} anion solute...")
            ion_group = anion_types[ion_type]
            solvents = {'water': self.waters, 'coion': self.cations}
            
            try:
                solute = self._create_solute_optimized(ion_group, solvents, step)
                self.solutes_ai[ion_type] = solute
                
                if hasattr(solute, 'radii') and 'water' in solute.radii:
                    print(f"  {ion_type}-water coordination radius: {solute.radii['water']:.2f} Å")
                    return solute
            except Exception as e:
                print(f"  Error: {e}")
                return None
        else:
            print(f"Ion type '{ion_type}' not found in system")
            return None

    def quick_initialize_solutes_with_defaults(self):
        '''
        Quick initialization using default coordination radii - much faster!
        '''
        
        print("Quick solute initialization using default radii...")
        
        # Default coordination radii (typical values)
        default_radii = {
            'Na': 2.8,   # Na+ typical first shell
            'K': 3.2,    # K+ typical first shell  
            'Mg': 2.4,   # Mg2+ typical first shell
            'Ca': 2.8,   # Ca2+ typical first shell
            'Li': 2.2,   # Li+ typical first shell
            'Cl': 3.5,   # Cl- typical first shell
            'Br': 3.7,   # Br- typical first shell
            'F': 2.8,    # F- typical first shell
            'I': 4.0     # I- typical first shell
        }
        
        # Get unique ion types
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        # Initialize dictionaries
        self.solutes_ci = {}
        self.solutes_ai = {}
        
        # Create mock solute objects with default radii
        for cation_name in cation_types.keys():
            radius = default_radii.get(cation_name, 2.8)  # Default fallback
            
            # Create a simple mock solute object
            mock_solute = type('MockSolute', (), {
                'radii': {'water': radius},
                'name': cation_name
            })()
            
            self.solutes_ci[cation_name] = mock_solute
            print(f"  {cation_name} (cation): {radius:.2f} Å (default)")
        
        for anion_name in anion_types.keys():
            radius = default_radii.get(anion_name, 3.5)  # Default fallback
            
            # Create a simple mock solute object
            mock_solute = type('MockSolute', (), {
                'radii': {'water': radius},
                'name': anion_name
            })()
            
            self.solutes_ai[anion_name] = mock_solute
            print(f"  {anion_name} (anion): {radius:.2f} Å (default)")
        
        # Create combined solutes
        self._create_combined_solutes()
        
        print("Quick initialization complete! Use regular initialization later if needed.")
        return self.solutes_ci, self.solutes_ai
    

    def get_coordination_radius_by_type(self, ion_type):
        '''
        Get coordination radius for a specific ion type.
        
        Parameters
        ----------
        ion_type : str
            Ion type name (e.g., 'Na', 'K', 'Cl')
        
        Returns
        -------
        radius : float
            Coordination radius in Angstroms
        '''
        
        if hasattr(self, 'solutes_ci') and ion_type in self.solutes_ci:
            solute = self.solutes_ci[ion_type]
            if solute is not None and hasattr(solute, 'radii') and 'water' in solute.radii:
                return solute.radii['water']
            else:
                print(f"Cation {ion_type} solute has no water radius")
                return None
                
        elif hasattr(self, 'solutes_ai') and ion_type in self.solutes_ai:
            solute = self.solutes_ai[ion_type]
            if solute is not None and hasattr(solute, 'radii') and 'water' in solute.radii:
                return solute.radii['water']
            else:
                print(f"Anion {ion_type} solute has no water radius")
                return None
        else:
            print(f"Ion type '{ion_type}' not found. Available types:")
            if hasattr(self, 'solutes_ci'):
                print(f"  Cations: {list(self.solutes_ci.keys())}")
            if hasattr(self, 'solutes_ai'):
                print(f"  Anions: {list(self.solutes_ai.keys())}")
            return None


    def print_coordination_radii_summary(self):
        '''Print summary of all coordination radii by ion type - works with cached mock solutes'''
        
        print("\n" + "="*50)
        print("COORDINATION RADII BY ION TYPE")
        print("="*50)
        
        found_any = False
        
        if hasattr(self, 'solutes_ci') and self.solutes_ci:
            print("CATIONS:")
            for ion_type, solute in self.solutes_ci.items():
                if solute is not None:
                    try:
                        # Handle both mock solutes and real solutes
                        if hasattr(solute, 'radii') and isinstance(solute.radii, dict) and 'water' in solute.radii:
                            radius = solute.radii['water']
                            print(f"  {ion_type}: {radius:.2f} Å")
                            found_any = True
                        else:
                            print(f"  {ion_type}: No radius data available")
                    except Exception as e:
                        print(f"  {ion_type}: Error accessing radius - {e}")
                else:
                    print(f"  {ion_type}: None")
        else:
            print("CATIONS: No cation solutes found")
        
        if hasattr(self, 'solutes_ai') and self.solutes_ai:
            print("\nANIONS:")
            for ion_type, solute in self.solutes_ai.items():
                if solute is not None:
                    try:
                        # Handle both mock solutes and real solutes
                        if hasattr(solute, 'radii') and isinstance(solute.radii, dict) and 'water' in solute.radii:
                            radius = solute.radii['water']
                            print(f"  {ion_type}: {radius:.2f} Å")
                            found_any = True
                        else:
                            print(f"  {ion_type}: No radius data available")
                    except Exception as e:
                        print(f"  {ion_type}: Error accessing radius - {e}")
                else:
                    print(f"  {ion_type}: None")
        else:
            print("\nANIONS: No anion solutes found")
        
        if not found_any:
            print("\n⚠️  No coordination radii found!")
            print("   The cached solutes might be corrupted.")
            print("   Try: eq_opt.quick_initialize_solutes_with_defaults()")
        
        print("="*50)

    def debug_rdf_classification(self):
        '''Debug method to check RDF classification'''
        
        if not hasattr(self, 'rdfs') or not self.rdfs:
            print("No RDFs available.")
            return
        
        print("="*60)
        print("RDF CLASSIFICATION DEBUG")
        print("="*60)
        
        # Get actual ion types from system
        cation_types = self._get_unique_ion_types(self.cations)
        anion_types = self._get_unique_ion_types(self.anions)
        
        print(f"Cation types in system: {list(cation_types.keys())}")
        print(f"Anion types in system: {list(anion_types.keys())}")
        
        print(f"\nAll available RDFs:")
        for rdf_name, rdf_data in self.rdfs.items():
            status = "✓" if rdf_data is not None else "✗"
            print(f"  {status} {rdf_name}")
        
        print(f"\nIon-water RDFs:")
        for rdf_name, rdf_data in self.rdfs.items():
            if rdf_name.endswith('-w') and rdf_name not in ['ci-w', 'ai-w', 'w-w']:
                ion_type = rdf_name.split('-')[0]
                if ion_type in cation_types:
                    classification = "CATION"
                elif ion_type in anion_types:
                    classification = "ANION"
                else:
                    classification = "UNKNOWN"
                
                status = "✓" if rdf_data is not None else "✗"
                print(f"  {status} {rdf_name} -> {classification}")
        
        print(f"\nCation-anion cross RDFs:")
        for rdf_name, rdf_data in self.rdfs.items():
            if ('-' in rdf_name and not rdf_name.endswith('-w') and 
                rdf_name not in ['ci-w', 'ai-w', 'w-w', 'ci-ai']):
                status = "✓" if rdf_data is not None else "✗"
                print(f"  {status} {rdf_name}")  



    def export_analysis_results(self, concentration, filename=None):
        '''
        Export all analysis results to a file for later comparison across concentrations.
        UPDATED: Now prioritizes coordination_by_shell_and_type from get_coordination_numbers_by_shell_and_type()
        FIXED: Now properly checks for and exports all coordination number data.
        UPDATED: Now includes residence time data (water and ion pairing)
        FIXED: Removed 'name' key that doesn't exist in shell_region_coordination_probabilities
        UPDATED: Now includes water dipole distribution by coordination data
        FIXED: Corrected key name from 'coordination_envs' to 'coordination_environments'
        FIXED: Safely handles missing statistical keys in residence time data
        
        Parameters
        ----------
        concentration : str or float
            Salt concentration identifier (e.g., '0.1M', '0.5M', '1.0M', '2.0M')
        filename : str, optional
            Output filename. If None, uses concentration-based naming
        '''
        
        # Add missing import
        from datetime import datetime
        
        if filename is None:
            filename = f'analysis_results_{concentration}.pkl'
        
        # Collect all important results
        results = {
            'concentration': concentration,
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'n_waters': len(self.waters),
                'n_cations': len(self.cations),
                'n_anions': len(self.anions),
                'n_frames': self.n_frames
            }
        }
        
        # RDFs
        if hasattr(self, 'rdfs') and self.rdfs:
            results['rdfs'] = {}
            for rdf_name, rdf_data in self.rdfs.items():
                if rdf_data is not None:
                    results['rdfs'][rdf_name] = {
                        'bins': rdf_data.bins.copy(),
                        'rdf': rdf_data.rdf.copy()
                    }
        
        # PRIORITIZED: Shell coordination numbers by type from get_coordination_numbers_by_shell_and_type()
        # This is the main coordination data we want to export
        coord_data_found = False
        
        if hasattr(self, 'coordination_by_shell_and_type') and self.coordination_by_shell_and_type:
            results['shell_coordination_numbers'] = {}
            for ion_type, ion_data in self.coordination_by_shell_and_type.items():
                results['shell_coordination_numbers'][ion_type] = {
                    'type': ion_data['type'],
                    'shells': {}
                }
                
                for shell_name, shell_data in ion_data['shells'].items():
                    results['shell_coordination_numbers'][ion_type]['shells'][shell_name] = {
                        'coordination_numbers': shell_data['coordination_numbers'].copy(),
                        'mean': shell_data['mean'],
                        'std': shell_data['std'],
                        'bounds': shell_data['bounds']
                    }
            coord_data_found = True
            print(f"  ✓ Saved shell coordination numbers: {list(self.coordination_by_shell_and_type.keys())}")
        else:
            print("  ⚠️  Shell coordination numbers not found.")
            print("     Run get_coordination_numbers_by_shell_and_type() before exporting")
        
        # OPTIONAL: Basic coordination numbers (overall, not by shell)
        # Only saved if available - not critical
        if hasattr(self, 'coordination_numbers_by_type') and self.coordination_numbers_by_type:
            results['coordination_numbers_overall'] = {}
            for ion_type, data in self.coordination_numbers_by_type.items():
                results['coordination_numbers_overall'][ion_type] = {
                    'type': data['type'],
                    'coordination_numbers': data['coordination_numbers'].copy(),
                    'mean': data['mean'],
                    'std': data['std']
                }
            print(f"  ✓ Saved overall coordination numbers: {list(self.coordination_numbers_by_type.keys())}")
        else:
            print("  Note: Overall coordination numbers not calculated (optional)")
        
        # Shell coordination probabilities
        if hasattr(self, 'shell_probabilities_by_ion_type'):
            results['shell_coordination_probabilities'] = {}
            for ion_type, ion_data in self.shell_probabilities_by_ion_type.items():
                results['shell_coordination_probabilities'][ion_type] = {
                    'category': ion_data['category'],
                    'data': ion_data['data'].copy()
                }
        
        # Shell Region Coordination Probabilities
        # FIXED: Handle both old and new key names for backward compatibility
        if hasattr(self, 'shell_region_coordination_probabilities'):
            results['shell_region_coordination_probabilities'] = {}
            for ion_type, ion_data in self.shell_region_coordination_probabilities.items():
                results['shell_region_coordination_probabilities'][ion_type] = {
                    'ion_category': ion_data['ion_category'],
                    'coordination_radius': ion_data['coordination_radius'],
                    'shell_regions': {}
                }
                
                for shell_name, shell_data in ion_data['shell_regions'].items():
                    # FIXED: Try both key names for backward compatibility
                    if 'coordination_environments' in shell_data:
                        coord_envs = shell_data['coordination_environments'].copy()
                    elif 'coordination_envs' in shell_data:
                        coord_envs = shell_data['coordination_envs'].copy()
                    else:
                        print(f"    Warning: No coordination environments found for {ion_type} {shell_name}")
                        coord_envs = []
                    
                    results['shell_region_coordination_probabilities'][ion_type]['shell_regions'][shell_name] = {
                        'bounds': shell_data['bounds'],
                        'coordination_environments': coord_envs,  # Use consistent key name
                        'probabilities': shell_data['probabilities'].copy()
                    }
        
        # Water Residence Times
        # FIXED: Safely handle missing statistical keys
        if hasattr(self, 'water_residence_times') and self.water_residence_times:
            results['water_residence_times'] = {}
            for ion_type, ion_data in self.water_residence_times.items():
                results['water_residence_times'][ion_type] = {
                    'ion_type': ion_data['ion_type'],
                    'ion_category': ion_data['ion_category'],
                    'n_ions': ion_data['n_ions'],
                    'n_frames_analyzed': ion_data['n_frames_analyzed'],
                    'step': ion_data['step'],
                    'shells': {}
                }
                
                for shell_name, shell_data in ion_data['shells'].items():
                    # FIXED: Safely get statistical values with fallback to calculated values
                    residence_times = shell_data['residence_times']
                    
                    # Calculate statistics on-the-fly if not present
                    mean_residence_time = shell_data.get('mean_residence_time', 
                                                        np.mean(residence_times) if len(residence_times) > 0 else 0)
                    std_residence_time = shell_data.get('std_residence_time',
                                                        np.std(residence_times) if len(residence_times) > 0 else 0)
                    n_events = shell_data.get('n_events', len(residence_times))
                    
                    results['water_residence_times'][ion_type]['shells'][shell_name] = {
                        'bounds': shell_data['bounds'],
                        'residence_times': residence_times.copy(),
                        'mean_residence_time': float(mean_residence_time),
                        'std_residence_time': float(std_residence_time),
                        'n_events': int(n_events)
                    }
            print(f"  ✓ Saved water residence times: {len(results['water_residence_times'])} ion types")
        
        # Ion Pairing Residence Times
        # FIXED: Safely handle missing statistical keys
        if hasattr(self, 'ion_pairing_residence_times') and self.ion_pairing_residence_times:
            results['ion_pairing_residence_times'] = {}
            for ion_type, ion_data in self.ion_pairing_residence_times.items():
                results['ion_pairing_residence_times'][ion_type] = {
                    'ion_type': ion_data['ion_type'],
                    'ion_category': ion_data['ion_category'],
                    'n_ions': ion_data['n_ions'],
                    'n_frames_analyzed': ion_data['n_frames_analyzed'],
                    'step': ion_data['step'],
                    'regions': {}
                }
                
                for region_name, region_data in ion_data['regions'].items():
                    # FIXED: Safely get statistical values with fallback to calculated values
                    residence_times = region_data['residence_times']
                    
                    # Calculate statistics on-the-fly if not present
                    mean_residence_time = region_data.get('mean_residence_time',
                                                        np.mean(residence_times) if len(residence_times) > 0 else 0)
                    std_residence_time = region_data.get('std_residence_time',
                                                        np.std(residence_times) if len(residence_times) > 0 else 0)
                    n_events = region_data.get('n_events', len(residence_times))
                    
                    results['ion_pairing_residence_times'][ion_type]['regions'][region_name] = {
                        'bounds': region_data['bounds'],
                        'residence_times': residence_times.copy(),
                        'mean_residence_time': float(mean_residence_time),
                        'std_residence_time': float(std_residence_time),
                        'n_events': int(n_events)
                    }
            print(f"  ✓ Saved ion pairing residence times: {len(results['ion_pairing_residence_times'])} ion types")
        
        # NEW: Water Dipole Distribution by Coordination
        if hasattr(self, 'dipole_by_coordination') and self.dipole_by_coordination:
            results['dipole_by_coordination'] = {}
            for ion_type, cn_data in self.dipole_by_coordination.items():
                results['dipole_by_coordination'][ion_type] = {}
                
                for cn, angles in cn_data.items():
                    results['dipole_by_coordination'][ion_type][cn] = {
                        'angles': angles.copy(),
                        'mean': float(angles.mean()),
                        'std': float(angles.std()),
                        'median': float(np.median(angles)),
                        'n_samples': len(angles)
                    }
            print(f"  ✓ Saved dipole by coordination: {len(results['dipole_by_coordination'])} ion types")
        
        # Polyhedron sizes by type
        if hasattr(self, 'polyhedron_results_by_type'):
            results['polyhedron_sizes'] = {}
            for ion_type, poly_data in self.polyhedron_results_by_type.items():
                results['polyhedron_sizes'][ion_type] = {
                    'ion_category': poly_data.ion_category,
                    'coordination_radius': poly_data.coordination_radius,
                    'n_ions': poly_data.n_ions,
                    'volumes': poly_data.volumes.copy(),
                    'areas': poly_data.areas.copy(),
                    'mean_volume': poly_data.overall_mean_volume,
                    'std_volume': poly_data.overall_std_volume,
                    'mean_area': poly_data.overall_mean_area,
                    'std_area': poly_data.overall_std_area
                }
        
        # Ion pairing cutoffs
        if hasattr(self, 'ion_pairs_by_type'):
            results['ion_pairing_cutoffs'] = {}
            for ion_type, pairing_data in self.ion_pairs_by_type.items():
                results['ion_pairing_cutoffs'][ion_type] = {
                    'ion_pairs': dict(pairing_data['ion_pairs']),
                    'rdf_key': pairing_data['rdf_key'],
                    'ion_category': pairing_data['ion_category']
                }
        
        # Ion pairing probabilities by coordination state
        if hasattr(self, 'coordination_pairing_analysis'):
            results['ion_pairing_probabilities_by_coordination'] = {}
            for ion_type, coord_data in self.coordination_pairing_analysis.items():
                results['ion_pairing_probabilities_by_coordination'][ion_type] = {}
                
                for cn, cn_data in coord_data.items():
                    results['ion_pairing_probabilities_by_coordination'][ion_type][cn] = {
                        'probabilities': cn_data['probabilities'].copy(),
                        'total_observations': cn_data['total_observations']
                    }
        
        # Water dipole distributions
        if hasattr(self, 'dipole_distributions_by_type'):
            results['water_dipole_distributions'] = {}
            for ion_type, dipole_data in self.dipole_distributions_by_type.items():
                results['water_dipole_distributions'][ion_type] = {
                    'angles': dipole_data.copy(),
                    'mean': float(dipole_data.mean()),
                    'std': float(dipole_data.std()),
                    'median': float(np.median(dipole_data)),
                    'n_samples': len(dipole_data)
                }
        
        # Solvation shells
        if hasattr(self, 'cation_shells_by_type'):
            results['solvation_shells'] = {'cations': {}, 'anions': {}}
            for ion_type, shells in self.cation_shells_by_type.items():
                if shells is not None:
                    results['solvation_shells']['cations'][ion_type] = dict(shells.data)
            
            if hasattr(self, 'anion_shells_by_type'):
                for ion_type, shells in self.anion_shells_by_type.items():
                    if shells is not None:
                        results['solvation_shells']['anions'][ion_type] = dict(shells.data)
        
        # Coordination radii
        if hasattr(self, 'solutes_ci') and hasattr(self, 'solutes_ai'):
            results['coordination_radii'] = {'cations': {}, 'anions': {}}
            
            for ion_type, solute in self.solutes_ci.items():
                if solute is not None and hasattr(solute, 'radii'):
                    results['coordination_radii']['cations'][ion_type] = solute.radii.get('water')
            
            for ion_type, solute in self.solutes_ai.items():
                if solute is not None and hasattr(solute, 'radii'):
                    results['coordination_radii']['anions'][ion_type] = solute.radii.get('water')
        
        # Shell occupancy probabilities
        if hasattr(self, 'shell_probabilities_by_type'):
            results['shell_occupancy_probabilities'] = {}
            for ion_type, prob_data in self.shell_probabilities_by_type.items():
                results['shell_occupancy_probabilities'][ion_type] = {
                    'type': prob_data['type'],
                    'shells': {}
                }
                
                for shell_name, shell_data in prob_data['shells'].items():
                    results['shell_occupancy_probabilities'][ion_type]['shells'][shell_name] = {
                        'probability_distribution': shell_data['probability_distribution'].copy(),
                        'mean_probability': shell_data['mean_probability'],
                        'std_probability': shell_data['std_probability'],
                        'bounds': shell_data['bounds']
                    }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        # Print comprehensive summary
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE ANALYSIS RESULTS EXPORTED")
        print(f"{'='*70}")
        print(f"Filename: {filename}")
        print(f"Concentration: {concentration}")
        print(f"\nData Summary:")
        print(f"  RDFs: {len(results.get('rdfs', {}))}")
        print(f"  Shell coordination (BY SHELL): {len(results.get('shell_coordination_numbers', {}))} ion types")
        if results.get('coordination_numbers_overall'):
            print(f"  Overall coordination: {len(results.get('coordination_numbers_overall', {}))} ion types")
        print(f"  Shell probabilities: {len(results.get('shell_coordination_probabilities', {}))}")
        print(f"  Shell region probs: {len(results.get('shell_region_coordination_probabilities', {}))}")
        print(f"  Water residence times: {len(results.get('water_residence_times', {}))}")
        print(f"  Ion pairing residence times: {len(results.get('ion_pairing_residence_times', {}))}")
        print(f"  Dipole by coordination: {len(results.get('dipole_by_coordination', {}))}")
        print(f"  Polyhedron data: {len(results.get('polyhedron_sizes', {}))}")
        print(f"  Ion pairing cutoffs: {len(results.get('ion_pairing_cutoffs', {}))}")
        print(f"  Pairing by coordination: {len(results.get('ion_pairing_probabilities_by_coordination', {}))}")
        print(f"  Dipole distributions: {len(results.get('water_dipole_distributions', {}))}")
        print(f"  Shell occupancy probs: {len(results.get('shell_occupancy_probabilities', {}))}")
        print(f"{'='*70}")
        
        if not coord_data_found:
            print(f"\n⚠️  WARNING: Shell coordination numbers were NOT exported!")
            print(f"   Run: eq_opt.get_coordination_numbers_by_shell_and_type()")
            print(f"   before exporting to include this critical data.")
        
        return filename
 
 

    def save_ion_solvation_shells_by_type_to_file(self, filename='ion_solvation_shells_by_type_cache.pkl'):
        '''
        Save ion-type-specific solvation shells to file for persistence across sessions.
        Includes all modifications made through boundary editors and manual adjustments.
        
        Parameters
        ----------
        filename : str
            Output filename, default='ion_solvation_shells_by_type_cache.pkl'
        
        Returns
        -------
        success : bool
            True if save was successful
        '''
        
        if not (hasattr(self, 'cation_shells_by_type') and hasattr(self, 'anion_shells_by_type')):
            print("No ion-type-specific shells to save")
            return False
        
        try:
            # Prepare shell data for serialization
            shell_data = {
                'metadata': {
                    'saved_date': datetime.now().isoformat(),
                    'n_frames': self.n_frames,
                    'trajectory_length': len(self.universe.trajectory)
                },
                'cation_shells': {},
                'anion_shells': {}
            }
            
            # Save cation shells
            for ion_type, shells in self.cation_shells_by_type.items():
                if shells is not None:
                    _d = dict(shells.data)
                    if 'bulk' in _d:
                        _ends = [v[1] for k, v in _d.items() if k != 'bulk' and not np.isinf(v[1])]
                        if _ends:
                            _d['bulk'] = (max(_ends), _d['bulk'][1])
                    shell_data['cation_shells'][ion_type] = {
                        'data': _d,  # Shell boundaries (bulk normalised)
                        'rdf_r': shells.rdf_r.copy() if hasattr(shells, 'rdf_r') else None,
                        'rdf_g_r': shells.rdf_g_r.copy() if hasattr(shells, 'rdf_g_r') else None,
                        'minima_indices': shells.minima_indices.copy() if hasattr(shells, 'minima_indices') else None,
                        'peak_indices': shells.peak_indices.copy() if hasattr(shells, 'peak_indices') else None
                    }
                else:
                    shell_data['cation_shells'][ion_type] = None
            
            # Save anion shells
            for ion_type, shells in self.anion_shells_by_type.items():
                if shells is not None:
                    _d = dict(shells.data)
                    if 'bulk' in _d:
                        _ends = [v[1] for k, v in _d.items() if k != 'bulk' and not np.isinf(v[1])]
                        if _ends:
                            _d['bulk'] = (max(_ends), _d['bulk'][1])
                    shell_data['anion_shells'][ion_type] = {
                        'data': _d,  # Shell boundaries (bulk normalised)
                        'rdf_r': shells.rdf_r.copy() if hasattr(shells, 'rdf_r') else None,
                        'rdf_g_r': shells.rdf_g_r.copy() if hasattr(shells, 'rdf_g_r') else None,
                        'minima_indices': shells.minima_indices.copy() if hasattr(shells, 'minima_indices') else None,
                        'peak_indices': shells.peak_indices.copy() if hasattr(shells, 'peak_indices') else None
                    }
                else:
                    shell_data['anion_shells'][ion_type] = None
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(shell_data, f)
            
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            
            print(f"Ion-type-specific solvation shells saved to {filename}")
            print(f"  File size: {file_size_mb:.1f} MB")
            print(f"  Saved {len([v for v in shell_data['cation_shells'].values() if v is not None])} cation types")
            print(f"  Saved {len([v for v in shell_data['anion_shells'].values() if v is not None])} anion types")
            
            # Print summary
            print(f"\n  Shell boundaries summary:")
            for ion_type, shells_info in shell_data['cation_shells'].items():
                if shells_info is not None:
                    n_shells = len([k for k in shells_info['data'].keys() if k.startswith('shell_')])
                    has_bulk = 'bulk' in shells_info['data']
                    print(f"    {ion_type} (cation): {n_shells} shells {'+ bulk' if has_bulk else ''}")
            
            for ion_type, shells_info in shell_data['anion_shells'].items():
                if shells_info is not None:
                    n_shells = len([k for k in shells_info['data'].keys() if k.startswith('shell_')])
                    has_bulk = 'bulk' in shells_info['data']
                    print(f"    {ion_type} (anion): {n_shells} shells {'+ bulk' if has_bulk else ''}")
            
            return True
            
        except Exception as e:
            print(f"Error saving ion-type-specific shells: {e}")
            traceback.print_exc()
            return False


    def load_ion_solvation_shells_by_type_from_file(self, filename='ion_solvation_shells_by_type_cache.pkl'):
        '''
        Load ion-type-specific solvation shells from file with error handling.
        
        Parameters
        ----------
        filename : str
            Input filename, default='ion_solvation_shells_by_type_cache.pkl'
        
        Returns
        -------
        success : bool
            True if load was successful
        '''
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        try:
            # Check file size first
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print(f"Cache file {filename} is empty")
                return False
            
            file_size_mb = file_size / (1024 * 1024)
            print(f"Loading ion-type-specific shells from {filename} ({file_size_mb:.1f} MB)...")
            
            # Load data
            with open(filename, 'rb') as f:
                shell_data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(shell_data, dict):
                print(f"Invalid shell data cache format")
                return False
            
            from MDAnalysis.analysis.base import Results
            
            # Reconstruct cation shells
            self.cation_shells_by_type = {}
            
            for ion_type, shells_info in shell_data.get('cation_shells', {}).items():
                if shells_info is not None:
                    shells = Results()
                    shells.data = dict(shells_info['data'])
                    # Normalise bulk start to follow last non-bulk shell end
                    if 'bulk' in shells.data:
                        _ends = [v[1] for k, v in shells.data.items() if k != 'bulk' and not np.isinf(v[1])]
                        if _ends:
                            shells.data['bulk'] = (max(_ends), shells.data['bulk'][1])
                    
                    # Restore RDF data if available
                    if shells_info['rdf_r'] is not None:
                        shells.rdf_r = shells_info['rdf_r']
                        setattr(shells, 'rdf_r', shells_info['rdf_r'])
                    
                    if shells_info['rdf_g_r'] is not None:
                        shells.rdf_g_r = shells_info['rdf_g_r']
                        setattr(shells, 'rdf_g_r', shells_info['rdf_g_r'])
                    
                    if shells_info['minima_indices'] is not None:
                        shells.minima_indices = shells_info['minima_indices']
                        setattr(shells, 'minima_indices', shells_info['minima_indices'])
                    
                    if shells_info['peak_indices'] is not None:
                        shells.peak_indices = shells_info['peak_indices']
                        setattr(shells, 'peak_indices', shells_info['peak_indices'])
                    
                    self.cation_shells_by_type[ion_type] = shells
                else:
                    self.cation_shells_by_type[ion_type] = None
            
            # Reconstruct anion shells
            self.anion_shells_by_type = {}
            
            for ion_type, shells_info in shell_data.get('anion_shells', {}).items():
                if shells_info is not None:
                    shells = Results()
                    shells.data = dict(shells_info['data'])
                    # Normalise bulk start to follow last non-bulk shell end
                    if 'bulk' in shells.data:
                        _ends = [v[1] for k, v in shells.data.items() if k != 'bulk' and not np.isinf(v[1])]
                        if _ends:
                            shells.data['bulk'] = (max(_ends), shells.data['bulk'][1])
                    
                    # Restore RDF data if available
                    if shells_info['rdf_r'] is not None:
                        shells.rdf_r = shells_info['rdf_r']
                        setattr(shells, 'rdf_r', shells_info['rdf_r'])
                    
                    if shells_info['rdf_g_r'] is not None:
                        shells.rdf_g_r = shells_info['rdf_g_r']
                        setattr(shells, 'rdf_g_r', shells_info['rdf_g_r'])
                    
                    if shells_info['minima_indices'] is not None:
                        shells.minima_indices = shells_info['minima_indices']
                        setattr(shells, 'minima_indices', shells_info['minima_indices'])
                    
                    if shells_info['peak_indices'] is not None:
                        shells.peak_indices = shells_info['peak_indices']
                        setattr(shells, 'peak_indices', shells_info['peak_indices'])
                    
                    self.anion_shells_by_type[ion_type] = shells
                else:
                    self.anion_shells_by_type[ion_type] = None
            
            # Print summary
            successful_cations = [k for k, v in self.cation_shells_by_type.items() if v is not None]
            successful_anions = [k for k, v in self.anion_shells_by_type.items() if v is not None]
            
            print(f"Ion-type-specific shells loaded from {filename}")
            print(f"  Loaded {len(successful_cations)} cation types successfully")
            print(f"  Loaded {len(successful_anions)} anion types successfully")
            
            if successful_cations:
                print(f"  Available cation types: {', '.join(successful_cations)}")
            if successful_anions:
                print(f"  Available anion types: {', '.join(successful_anions)}")
            
            # Print detailed summary
            print(f"\n  Shell boundaries summary:")
            for ion_type in successful_cations:
                shells = self.cation_shells_by_type[ion_type]
                n_shells = len([k for k in shells.data.keys() if k.startswith('shell_')])
                has_bulk = 'bulk' in shells.data
                print(f"    {ion_type} (cation): {n_shells} shells {'+ bulk' if has_bulk else ''}")
                for shell_name, (start, end) in shells.data.items():
                    end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                    print(f"      {shell_name}: {start:.2f} - {end_str} Å")
            
            for ion_type in successful_anions:
                shells = self.anion_shells_by_type[ion_type]
                n_shells = len([k for k in shells.data.keys() if k.startswith('shell_')])
                has_bulk = 'bulk' in shells.data
                print(f"    {ion_type} (anion): {n_shells} shells {'+ bulk' if has_bulk else ''}")
                for shell_name, (start, end) in shells.data.items():
                    end_str = f"{end:.2f}" if not np.isinf(end) else "∞"
                    print(f"      {shell_name}: {start:.2f} - {end_str} Å")
            
            return True
            
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file {filename} is corrupted: {e}")
            print("Removing corrupted cache file...")
            try:
                os.remove(filename)
                print(f"Corrupted file {filename} removed")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading ion-type-specific shells from {filename}: {e}")
            traceback.print_exc()
            return False

    def determine_ion_solvation_shells_by_type_with_cache(self, cache_filename='ion_solvation_shells_by_type_cache.pkl', 
                                                        force_recalc=False, **kwargs):
        '''
        Determine ion solvation shells with automatic caching.
        
        Parameters
        ----------
        cache_filename : str
            Cache file name, default='ion_solvation_shells_by_type_cache.pkl'
        force_recalc : bool
            Force recalculation even if cache exists, default=False
        **kwargs : dict
            Arguments to pass to determine_ion_solvation_shells_by_type()
        
        Returns
        -------
        cation_shells, anion_shells : tuple
            Dictionaries of shell data by ion type
        '''
        
        # Try to load from cache first
        if not force_recalc and os.path.exists(cache_filename):
            print("Attempting to load ion-type-specific shells from cache...")
            if self.load_ion_solvation_shells_by_type_from_file(cache_filename):
                print("✓ Successfully loaded ion-type-specific shells from cache")
                return self.cation_shells_by_type, self.anion_shells_by_type
            else:
                print("✗ Cache loading failed, will recalculate")
        
        # Calculate shells
        print("Calculating ion-type-specific shells...")
        cation_shells, anion_shells = self.determine_ion_solvation_shells_by_type(**kwargs)
        
        # Save to cache
        if cation_shells or anion_shells:
            print("Saving ion-type-specific shells to cache...")
            if self.save_ion_solvation_shells_by_type_to_file(cache_filename):
                print("✓ Shells cached successfully")
            else:
                print("✗ Cache saving failed, but shells are available in memory")
        
        return cation_shells, anion_shells    
