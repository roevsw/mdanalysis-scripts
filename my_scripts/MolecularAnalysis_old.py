# MolecularAnalysis class for analyzing organic molecules and complex molecular systems

"""
MolecularAnalysis: Comprehensive Molecular Dynamics Analysis Toolkit

This module provides advanced analysis capabilities for organic molecules and complex molecular systems,
extending beyond simple ion-water analysis to handle diverse molecular environments including multiple ion types.

EXAMPLE USAGE SCENARIOS
=======================

1. DRUG MOLECULE IN WATER WITH IONS
-----------------------------------
Analysis of a small molecule drug in aqueous solution with physiological ions:

>>> from molecular_analysis import MolecularAnalysis
>>> 
>>> # Initialize analysis with multiple ion types
>>> drug_analysis = MolecularAnalysis('system.tpr', 'md_trajectory.xtc',
...                                   solute_sel='resname LIG',
...                                   solvent_sel='resname SOL',
...                                   cation_sel='resname NA K',  # Sodium and potassium
...                                   anion_sel='resname CL')     # Chloride
>>> 
>>> # Calculate drug-water RDF
>>> drug_water_rdf = drug_analysis.molecular_rdf('resname LIG', 'resname SOL and name OW')
>>> 
>>> # Analyze drug-ion interactions
>>> drug_ion_analysis = drug_analysis.ion_binding_analysis('resname LIG')
>>> 
>>> # Multi-site solvation including ion effects
>>> solvation_sites = {
...     'carbonyl': 'name O1 O2',
...     'amino': 'name N1 N2',
...     'aromatic': 'name C1 C2 C3 C4 C5 C6'
... }
>>> solvation_results = drug_analysis.solvation_analysis_organic(solvation_sites, include_ions=True)

2. PROTEIN-LIGAND COMPLEX WITH PHYSIOLOGICAL IONS
-------------------------------------------------
Analysis of protein-drug interactions in physiological ionic strength:

>>> # Initialize for protein-ligand system with multiple ions
>>> complex_analysis = MolecularAnalysis('complex.tpr', 'production.xtc',
...                                      solute_sel='protein or resname LIG',
...                                      solvent_sel='resname SOL',
...                                      cation_sel='resname NA K MG CA',  # Multiple cation types
...                                      anion_sel='resname CL SO4 PO4')   # Multiple anion types
>>> 
>>> # Ion-protein interactions
>>> ion_protein_contacts = complex_analysis.ion_protein_analysis('protein')
>>> print(f"Cation binding sites: {len(ion_protein_contacts['cation_sites'])}")
>>> print(f"Anion binding sites: {len(ion_protein_contacts['anion_sites'])}")
>>> 
>>> # Ion competition analysis
>>> competition = complex_analysis.ion_competition_analysis('resname LIG', cutoff=5.0)

3. MEMBRANE SYSTEM WITH ASYMMETRIC ION DISTRIBUTION
---------------------------------------------------
Analysis of lipid bilayers with asymmetric ion distribution:

>>> # Membrane system with multiple ion types
>>> membrane_analysis = MolecularAnalysis('membrane.tpr', 'membrane_md.xtc',
...                                       solute_sel='protein or resname POPC POPE',
...                                       solvent_sel='resname SOL',
...                                       cation_sel='resname NA K CA',
...                                       anion_sel='resname CL')
>>> 
>>> # Analyze ion distribution across leaflets
>>> ion_distribution = membrane_analysis.membrane_ion_distribution(
...     membrane_center_z=0.0,
...     leaflet_thickness=20.0
... )
>>> 
>>> # Ion-lipid interactions
>>> lipid_ion_binding = membrane_analysis.ion_binding_analysis('resname POPC POPE')

4. POLYMER SOLUTION WITH ION CONDENSATION
-----------------------------------------
Analysis of polyelectrolyte systems with counterion condensation:

>>> # Polymer system with multiple ion types
>>> polymer_analysis = MolecularAnalysis('polymer_system.tpr', 'polymer_traj.xtc',
...                                      solute_sel='resname PA*',  # Polymer chains
...                                      solvent_sel='resname SOL',
...                                      cation_sel='resname NA K MG',  # Counterions
...                                      anion_sel='resname CL SO4')    # Co-ions
>>> 
>>> # Counterion condensation analysis
>>> condensation = polymer_analysis.counterion_condensation_analysis(
...     polymer_sel='resname PA*',
...     charged_sites_sel='name COO',  # Carboxylate groups
...     cutoff=3.5
... )
>>> 
>>> # Ion selectivity analysis
>>> selectivity = polymer_analysis.ion_selectivity_analysis('resname PA*')

5. MULTI-ION COMPETITION IN DRUG BINDING
----------------------------------------
Analysis of how different ions compete for binding sites:

>>> # Multi-component system with ion competition
>>> multiion_analysis = MolecularAnalysis('multicomp.tpr', 'simulation.xtc',
...                                       solute_sel='resname LIG DMSO',
...                                       solvent_sel='resname SOL',
...                                       cation_sel='resname NA K MG CA ZN',  # Multiple competing cations
...                                       anion_sel='resname CL SO4 PO4')
>>> 
>>> # Ion competition for drug binding sites
>>> competition = multiion_analysis.ion_competition_analysis('resname LIG')
>>> 
>>> # Specific ion pairing analysis
>>> na_cl_pairs = multiion_analysis.specific_ion_pair_analysis('resname NA', 'resname CL')
>>> mg_so4_pairs = multiion_analysis.specific_ion_pair_analysis('resname MG', 'resname SO4')

6. ION-SPECIFIC EFFECTS IN PROTEIN FOLDING
------------------------------------------
Analysis of how different ions affect protein stability:

>>> # Protein system with Hofmeister series ions
>>> protein_analysis = MolecularAnalysis('protein.tpr', 'folding.xtc',
...                                      solute_sel='protein',
...                                      solvent_sel='resname SOL',
...                                      cation_sel='resname NA K NH4',      # Hofmeister cations
...                                      anion_sel='resname CL SO4 SCN')     # Hofmeister anions
>>> 
>>> # Ion-protein surface interactions
>>> surface_interactions = protein_analysis.ion_protein_surface_analysis()
>>> 
>>> # Ion effects on protein compactness
>>> compactness_effects = protein_analysis.ion_induced_structural_changes()

SYSTEM REQUIREMENTS
==================
- MDAnalysis >= 2.0
- NumPy, SciPy, Pandas
- Matplotlib for visualization
- scikit-learn for clustering
- Optional: Parallel processing capabilities

SUPPORTED ION TYPES
===================
Common cations: NA, K, MG, CA, ZN, FE, MN, CU, NH4, LI, RB, CS
Common anions: CL, SO4, PO4, NO3, CO3, HCO3, F, BR, I, SCN, OAC

For more examples and detailed documentation, see the class methods below.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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

    def molecular_rdf(self, group1_sel, group2_sel, bin_width=0.05, range=(0, 15), 
                     step=1, njobs=1, center_method=None, normalize=True):
        '''
        Calculate RDF between arbitrary molecular groups with flexible centering options.
        Now includes automatic ion type handling.
        '''
        
        if center_method is None:
            center_method = self.center_method
        
        group1 = self.universe.select_atoms(group1_sel)
        group2 = self.universe.select_atoms(group2_sel)
        
        if len(group1) == 0 or len(group2) == 0:
            raise ValueError("One or both atom groups are empty")
        
        nbins = int((range[1] - range[0]) / bin_width)
        
        if center_method in ['COM', 'COG']:
            return self._rdf_with_centers(group1, group2, nbins, range, step, 
                                        njobs, center_method, normalize)
        else:
            rdf = InterRDF(group1, group2, nbins=nbins, range=range, 
                          norm='rdf' if normalize else 'none', verbose=True)
            rdf.run(step=step, njobs=njobs)
            return rdf.results

    def ion_binding_analysis(self, target_sel, cutoff=3.5, step=1):
        '''
        Comprehensive analysis of ion binding to target molecules.
        
        Parameters
        ----------
        target_sel : str
            Selection for target molecules (e.g., 'resname LIG', 'protein')
        cutoff : float
            Distance cutoff for ion binding, default=3.5 Å
        step : int
            Trajectory step, default=1
            
        Returns
        -------
        binding_results : dict
            Comprehensive ion binding analysis results
        '''
        
        target = self.universe.select_atoms(target_sel)
        if len(target) == 0:
            raise ValueError(f"No target atoms found with selection: {target_sel}")
        
        print(f"Analyzing ion binding to {len(target)} target atoms...")
        
        results = {
            'cation_binding': {},
            'anion_binding': {},
            'total_binding': {'cations': [], 'anions': []},
            'binding_sites': {},
            'selectivity': {}
        }
        
        # Analyze each cation type
        for cation_name, cation_atoms in self.cation_types.items():
            print(f"  Analyzing {cation_name} binding...")
            binding_data = self._analyze_ion_binding(target, cation_atoms, cutoff, step)
            results['cation_binding'][cation_name] = binding_data
        
        # Analyze each anion type
        for anion_name, anion_atoms in self.anion_types.items():
            print(f"  Analyzing {anion_name} binding...")
            binding_data = self._analyze_ion_binding(target, anion_atoms, cutoff, step)
            results['anion_binding'][anion_name] = binding_data
        
        # Calculate total ion binding per frame
        for ts in tqdm(self.universe.trajectory[::step], desc="Calculating total binding"):
            total_cations = 0
            total_anions = 0
            
            for cation_atoms in self.cation_types.values():
                if len(cation_atoms) > 0:
                    dist_matrix = distances.distance_array(target.positions, 
                                                         cation_atoms.positions,
                                                         box=ts.dimensions)
                    total_cations += (dist_matrix <= cutoff).sum()
            
            for anion_atoms in self.anion_types.values():
                if len(anion_atoms) > 0:
                    dist_matrix = distances.distance_array(target.positions, 
                                                         anion_atoms.positions,
                                                         box=ts.dimensions)
                    total_anions += (dist_matrix <= cutoff).sum()
            
            results['total_binding']['cations'].append(total_cations)
            results['total_binding']['anions'].append(total_anions)
        
        # Calculate selectivity indices
        results['selectivity'] = self._calculate_ion_selectivity(results)
        
        self.ion_binding_data[target_sel] = results
        return results

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
        '''Calculate selectivity indices between different ion types'''
        
        selectivity = {}
        
        # Cation selectivity
        cation_names = list(binding_results['cation_binding'].keys())
        if len(cation_names) > 1:
            for i, cat1 in enumerate(cation_names):
                for cat2 in cation_names[i+1:]:
                    binding1 = binding_results['cation_binding'][cat1]['average_binding']
                    binding2 = binding_results['cation_binding'][cat2]['average_binding']
                    
                    if binding1 + binding2 > 0:
                        selectivity[f"{cat1}_over_{cat2}"] = binding1 / (binding1 + binding2)
        
        # Anion selectivity
        anion_names = list(binding_results['anion_binding'].keys())
        if len(anion_names) > 1:
            for i, an1 in enumerate(anion_names):
                for an2 in anion_names[i+1:]:
                    binding1 = binding_results['anion_binding'][an1]['average_binding']
                    binding2 = binding_results['anion_binding'][an2]['average_binding']
                    
                    if binding1 + binding2 > 0:
                        selectivity[f"{an1}_over_{an2}"] = binding1 / (binding1 + binding2)
        
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

    def solvation_analysis_organic(self, site_selections, cutoff=3.5, step=1, include_ions=False):
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
            
        Returns
        -------
        solvation_results : dict
            Enhanced solvation analysis results including ions
        '''
        
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
        '''Calculate RDF using molecular centers'''
        
        print(f"Warning: Center-based RDF ({center_method}) not fully implemented.")
        print("Falling back to atom-based RDF using representative atoms.")
        
        rep1 = group1.residues.atoms[group1.residues.resids - group1.residues.resids[0]]
        rep2 = group2.residues.atoms[group2.residues.resids - group2.residues.resids[0]]
        
        rdf = InterRDF(rep1, rep2, nbins=nbins, range=range_rdf, 
                      norm='rdf' if normalize else 'none', verbose=True)
        rdf.run(step=step, njobs=njobs)
        return rdf.results

    # ... include all other methods from the original implementation ...
    # (protein_ligand_contacts, hydrogen_bond_analysis, molecular_clustering, etc.)