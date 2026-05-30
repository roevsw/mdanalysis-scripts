"""
ConcentrationComparison.py

Separate class for comparing analysis results across different concentrations.
Handles all concentration comparison functionality.

Author: R.Swai
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict
from datetime import datetime


class ConcentrationComparison:
    def __init__(self, concentration_data=None):
        '''
        Initialize ConcentrationComparison with optional concentration data.
        
        Parameters
        ----------
        concentration_data : dict, optional
            Dictionary with concentration keys and folder/label/color info
            Format: {
                'conc_label': {
                    'folder': '/path/to/folder',
                    'label': 'display label',
                    'color': 'plot color'
                }
            }
        '''
        self.concentration_data = concentration_data if concentration_data is not None else {}
        self.concentrations = {}
        self.loaded_data = {}

    def load_all_concentrations(self, filename_pattern='results_*.pkl', save_log=True):
        '''
        Load all exported analysis results from the specified folders.
        
        Parameters
        ----------
        filename_pattern : str
            Pattern for result files (default: 'results_*.pkl')
        save_log : bool
            Whether to save loading log to file, default=True
        
        Returns
        -------
        bool
            True if all files loaded successfully, False otherwise
        '''
        
        log_lines = []
        log_lines.append("="*80)
        log_lines.append("CONCENTRATION DATA LOADING LOG")
        log_lines.append("="*80)
        log_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append("")
        
        if not self.concentration_data:
            msg = "No concentration data defined. Set concentration_data first."
            log_lines.append(msg)
            if save_log:
                with open('concentration_loading_ERROR.log', 'w') as f:
                    f.write('\n'.join(log_lines))
            return False
        
        all_loaded = True
        
        log_lines.append(f"Attempting to load {len(self.concentration_data)} concentrations")
        log_lines.append("-"*80)
        
        for conc_key, conc_info in self.concentration_data.items():
            folder = conc_info['folder']
            
            log_lines.append(f"\nConcentration: {conc_key}")
            log_lines.append(f"  Folder: {folder}")
            
            # Look for pkl files in the folder
            if not os.path.exists(folder):
                msg = f"  ✗ Folder not found"
                log_lines.append(msg)
                all_loaded = False
                continue
            
            # Find result files (try multiple patterns)
            possible_files = [
                os.path.join(folder, f'results_{conc_key}.pkl'),
                os.path.join(folder, f'analysis_results_{conc_key}.pkl'),
                os.path.join(folder, 'analysis_results.pkl'),
                os.path.join(folder, 'results.pkl')
            ]
            
            # Find the first file that exists
            result_file = None
            for filepath in possible_files:
                if os.path.exists(filepath):
                    result_file = filepath
                    break
            
            if result_file is None:
                log_lines.append(f"  ✗ No result file found")
                log_lines.append(f"  Tried files:")
                for f in possible_files:
                    log_lines.append(f"    - {os.path.basename(f)}")
                all_loaded = False
                continue
            
            # Load the data
            try:
                with open(result_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.loaded_data[conc_key] = data
                log_lines.append(f"  ✓ Successfully loaded: {os.path.basename(result_file)}")
                log_lines.append(f"  Data keys: {', '.join(list(data.keys())[:8])}...")
                
            except Exception as e:
                log_lines.append(f"  ✗ Error loading file: {e}")
                all_loaded = False
        
        log_lines.append("")
        log_lines.append("="*80)
        log_lines.append("SUMMARY")
        log_lines.append("="*80)
        
        if self.loaded_data:
            log_lines.append(f"Successfully loaded: {len(self.loaded_data)}/{len(self.concentration_data)} concentrations")
            log_lines.append(f"Loaded concentrations: {', '.join(sorted(self.loaded_data.keys()))}")
            
            # Save log
            if save_log:
                log_filename = 'concentration_loading.log'
                with open(log_filename, 'w') as f:
                    f.write('\n'.join(log_lines))
                print(f"Loading log saved: {log_filename}")
                print(f"Successfully loaded {len(self.loaded_data)}/{len(self.concentration_data)} concentrations")
            
            return all_loaded
        else:
            log_lines.append("No data was loaded successfully")
            
            if save_log:
                log_filename = 'concentration_loading_ERROR.log'
                with open(log_filename, 'w') as f:
                    f.write('\n'.join(log_lines))
                print(f"Error log saved: {log_filename}")
            
            return False
    
    def get_available_ion_types(self):
        '''Get list of all ion types available across all loaded concentrations'''
        
        ion_types = set()
        
        for conc_data in self.loaded_data.values():
            # Check different possible data structures
            if 'shell_coordination_numbers' in conc_data:
                ion_types.update(conc_data['shell_coordination_numbers'].keys())
            if 'coordination_numbers_overall' in conc_data:
                ion_types.update(conc_data['coordination_numbers_overall'].keys())
            if 'water_residence_times' in conc_data:
                ion_types.update(conc_data['water_residence_times'].keys())
        
        return sorted(ion_types)
    


    def compare_shells_across_concentrations(self, ion_type, save_plots=True, save_log=True,
                                            bar_width=0.15, xlabel_fontsize=12, ylabel_fontsize=12,
                                            title_fontsize=14, legend_fontsize=10, tick_fontsize=10):
        '''
        Compare shell coordination numbers for a specific ion across all concentrations.
        
        Parameters
        ----------
        ion_type : str
            Ion type to compare (e.g., 'Na', 'Mg', 'Cl')
        save_plots : bool
            Whether to save the plot
        save_log : bool
            Whether to save analysis log to file
        bar_width : float
            Width of bars in grouped bar chart, default=0.15
        xlabel_fontsize : int
            Font size for x-axis label, default=12
        ylabel_fontsize : int
            Font size for y-axis label, default=12
        title_fontsize : int
            Font size for title, default=14
        legend_fontsize : int
            Font size for legend, default=10
        tick_fontsize : int
            Font size for axis tick labels, default=10
        
        Returns
        -------
        comparison_data : dict
            Dictionary with shell coordination data for each concentration
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"SHELL COORDINATION COMPARISON: {ion_type}")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append("")
        
        # Collect shell coordination data
        comparison_data = {}
        
        log_lines.append("LOADING SHELL COORDINATION DATA")
        log_lines.append("-"*80)
        
        for conc_key in sorted(self.loaded_data.keys()):
            conc_data = self.loaded_data[conc_key]
            
            if 'shell_coordination_numbers' in conc_data:
                if ion_type in conc_data['shell_coordination_numbers']:
                    ion_data = conc_data['shell_coordination_numbers'][ion_type]
                    
                    # FIXED: Check if data has 'shells' key (nested structure)
                    if isinstance(ion_data, dict) and 'shells' in ion_data:
                        # Structure: {'type': 'cation', 'shells': {'shell_1': {...}, ...}}
                        comparison_data[conc_key] = ion_data['shells']
                        log_lines.append(f"✓ {conc_key}: Loaded shell coordination data (nested structure)")
                    elif isinstance(ion_data, dict):
                        # Direct structure: {'shell_1': {...}, 'shell_2': {...}}
                        comparison_data[conc_key] = ion_data
                        log_lines.append(f"✓ {conc_key}: Loaded shell coordination data (direct structure)")
                    else:
                        log_lines.append(f"⚠ {conc_key}: Unexpected data structure: {type(ion_data)}")
                else:
                    log_lines.append(f"⚠ {conc_key}: Ion type '{ion_type}' not found")
            else:
                log_lines.append(f"⚠ {conc_key}: No shell_coordination_numbers data")
        
        if not comparison_data:
            log_lines.append("")
            log_lines.append(f"ERROR: No shell coordination data found for {ion_type}")
            
            if save_log:
                log_filename = f'shell_comparison_{ion_type}_ERROR.log'
                with open(log_filename, 'w') as f:
                    f.write('\n'.join(log_lines))
                print(f"Error log saved: {log_filename}")
            
            return None
        
        log_lines.append("")
        log_lines.append(f"Total concentrations: {len(comparison_data)}")
        log_lines.append("")
        
        # Extract shell data
        log_lines.append("SHELL COORDINATION NUMBERS")
        log_lines.append("-"*80)
        
        # Determine all shell names
        all_shells = set()
        for conc_shells in comparison_data.values():
            all_shells.update(conc_shells.keys())
        
        # Remove 'overall' if present and sort shells
        if 'overall' in all_shells:
            all_shells.remove('overall')
        shell_names = sorted(all_shells, key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        
        log_lines.append(f"Shells detected: {', '.join(shell_names)}")
        log_lines.append("")
        
        # Prepare data for plotting
        concentrations = sorted(comparison_data.keys(), key=lambda x: float(x.replace('M', '')))
        
        # Create table header
        log_lines.append(f"{'Concentration':<15} " + "  ".join([f"{shell:<15}" for shell in shell_names]))
        log_lines.append("-"*80)
        
        shell_data = {shell: [] for shell in shell_names}
        shell_errors = {shell: [] for shell in shell_names}
        
        # Extract data - handle nested dict structure
        for conc_key in concentrations:
            conc_shells = comparison_data[conc_key]
            row_data = [f"{conc_key:<15}"]
            
            for shell in shell_names:
                if shell in conc_shells:
                    shell_info = conc_shells[shell]
                    
                    # Extract mean and std from nested dict
                    if isinstance(shell_info, dict) and 'mean' in shell_info:
                        mean = shell_info['mean']
                        std = shell_info['std']
                        shell_data[shell].append(mean)
                        shell_errors[shell].append(std)
                        row_data.append(f"{mean:6.3f}±{std:5.3f}")
                    elif isinstance(shell_info, (int, float)):
                        # Direct value
                        shell_data[shell].append(shell_info)
                        shell_errors[shell].append(0)
                        row_data.append(f"{shell_info:6.3f}")
                    else:
                        log_lines.append(f"WARNING: Unexpected structure for {shell}: {type(shell_info)}")
                        shell_data[shell].append(0)
                        shell_errors[shell].append(0)
                        row_data.append(f"{'N/A':<15}")
                else:
                    shell_data[shell].append(0)
                    shell_errors[shell].append(0)
                    row_data.append(f"{'N/A':<15}")
            
            log_lines.append("  ".join(row_data))
        
        log_lines.append("")
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n_conc = len(concentrations)
        n_shells = len(shell_names)
        
        # X positions for each concentration group
        x_pos = np.arange(n_conc)
        
        # Colors for each shell (blue gradient)
        shell_colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_shells))
        
        # Plot bars for each shell
        for i, shell in enumerate(shell_names):
            offset = (i - n_shells/2 + 0.5) * bar_width
            
            # Only add error bars if we have non-zero errors
            has_errors = any(err > 0 for err in shell_errors[shell])
            
            if has_errors:
                ax.bar(x_pos + offset, shell_data[shell], bar_width, 
                    yerr=shell_errors[shell], 
                    label=shell.replace('_', ' ').title(),
                    color=shell_colors[i],
                    capsize=3,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5)
            else:
                ax.bar(x_pos + offset, shell_data[shell], bar_width,
                    label=shell.replace('_', ' ').title(),
                    color=shell_colors[i],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('Concentration', fontsize=xlabel_fontsize)
        ax.set_ylabel('Coordination Number', fontsize=ylabel_fontsize)
        ax.set_title(f'Shell Coordination Numbers: {ion_type}', fontsize=title_fontsize, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([self.concentration_data[c].get('label', c) for c in concentrations], 
                        rotation=0)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.legend(fontsize=legend_fontsize, frameon=False, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # Save files
        if save_plots:
            filename = f'shell_comparison_{ion_type}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            log_lines.append(f"PLOT SAVED: {filename}")
            print(f"Plot saved: {filename}")
        
        if save_log:
            log_filename = f'shell_comparison_{ion_type}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Analysis log saved: {log_filename}")
        
        plt.show()
        
        return comparison_data




    def compare_coordination_numbers(self, ion_type, save_plots=True, save_log=True,
                                    xlabel_fontsize=12, ylabel_fontsize=12,
                                    title_fontsize=14, legend_fontsize=10, tick_fontsize=10):
        '''
        Compare overall coordination numbers for a specific ion across concentrations.
        
        Parameters
        ----------
        ion_type : str
            Ion type to compare (e.g., 'Na', 'Mg', 'Cl')
        save_plots : bool
            Whether to save the plot
        save_log : bool
            Whether to save analysis log to file
        xlabel_fontsize : int
            Font size for x-axis label, default=12
        ylabel_fontsize : int
            Font size for y-axis label, default=12
        title_fontsize : int
            Font size for title, default=14
        legend_fontsize : int
            Font size for legend, default=10
        tick_fontsize : int
            Font size for axis tick labels, default=10
        
        Returns
        -------
        comparison_data : dict
            Dictionary with coordination data for each concentration
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"COORDINATION NUMBER COMPARISON: {ion_type}")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append("")
        
        comparison_data = {}
        
        log_lines.append("LOADING COORDINATION DATA")
        log_lines.append("-"*80)
        
        for conc_key in sorted(self.loaded_data.keys()):
            conc_data = self.loaded_data[conc_key]
            
            if 'shell_coordination_numbers' in conc_data:
                if ion_type in conc_data['shell_coordination_numbers']:
                    if 'overall' in conc_data['shell_coordination_numbers'][ion_type]:
                        comparison_data[conc_key] = conc_data['shell_coordination_numbers'][ion_type]['overall']
                        log_lines.append(f"✓ {conc_key}: Loaded overall coordination data")
                    else:
                        log_lines.append(f"⚠ {conc_key}: No 'overall' coordination data")
                else:
                    log_lines.append(f"⚠ {conc_key}: Ion type '{ion_type}' not found")
            elif 'coordination_numbers_overall' in conc_data:
                if ion_type in conc_data['coordination_numbers_overall']:
                    comparison_data[conc_key] = conc_data['coordination_numbers_overall'][ion_type]
                    log_lines.append(f"✓ {conc_key}: Loaded coordination data")
            else:
                log_lines.append(f"⚠ {conc_key}: No coordination data")
        
        if not comparison_data:
            log_lines.append("")
            log_lines.append(f"ERROR: No coordination data found for {ion_type}")
            
            if save_log:
                log_filename = f'coordination_comparison_{ion_type}_ERROR.log'
                with open(log_filename, 'w') as f:
                    f.write('\n'.join(log_lines))
                print(f"Error log saved: {log_filename}")
            
            return None
        
        log_lines.append("")
        log_lines.append(f"Total concentrations: {len(comparison_data)}")
        log_lines.append("")
        
        # Extract data
        log_lines.append("COORDINATION NUMBER DATA")
        log_lines.append("-"*80)
        log_lines.append(f"{'Concentration':<15} {'Mean':<10} {'Std Dev':<10}")
        log_lines.append("-"*80)
        
        concentrations, means, stds = [], [], []
        
        sorted_concs = sorted(comparison_data.keys(), key=lambda x: float(x.replace('M', '')))
        
        for conc_key in sorted_concs:
            cn_data = comparison_data[conc_key]
            conc_val = float(conc_key.replace('M', ''))
            concentrations.append(conc_val)
            means.append(cn_data['mean'])
            stds.append(cn_data['std'])
            
            log_lines.append(f"{conc_key:<15} {cn_data['mean']:<10.3f} {cn_data['std']:<10.3f}")
        
        log_lines.append("")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(concentrations, means, yerr=stds, 
                marker='o', markersize=8, capsize=5, linewidth=2,
                color='#00c5ff', markerfacecolor='#00c5ff', 
                markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel('Concentration (M)', fontsize=xlabel_fontsize)
        ax.set_ylabel('Mean Coordination Number', fontsize=ylabel_fontsize)
        ax.set_title(f'Coordination Number vs Concentration: {ion_type}', 
                    fontsize=title_fontsize, fontweight='bold')
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'coordination_vs_concentration_{ion_type}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            log_lines.append(f"PLOT SAVED: {filename}")
            print(f"Plot saved: {filename}")
        
        if save_log:
            log_filename = f'coordination_comparison_{ion_type}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Analysis log saved: {log_filename}")
        
        plt.show()
        
        return comparison_data



    def compare_rdfs_across_concentrations(self, ion_type, partner_type='w', save_plots=True, 
                                        plot_range=12, shell_boundaries=None, 
                                        add_inset=False, inset_xlim=None, inset_ylim=None,
                                        inset_bbox=None,
                                        shell_alpha=0.35, xlabel_fontsize=12, ylabel_fontsize=12,
                                        title_fontsize=14, legend_fontsize=10, tick_fontsize=10,
                                        shell_label_fontsize=10, save_log=True):
        '''
        Compare RDFs for a specific ion type across all concentrations.
        
        Parameters
        ----------
        ion_type : str
            Ion type to compare (e.g., 'Na', 'Mg', 'Cl')
        partner_type : str
            Partner type for RDF: 'w' for water, 'ci' for cation-ion, 'ai' for anion-ion
        save_plots : bool
            Whether to save the plot
        plot_range : float
            Maximum distance to plot (Angstroms)
        shell_boundaries : dict, optional
            Manual shell boundaries to add to plot. Format:
            {'shell_1': {'r_min': 0.0, 'r_max': 3.2}, ...}
        add_inset : bool
            Whether to add an inset zoom plot
        inset_xlim : tuple, optional
            X-axis limits for inset (e.g., (2.0, 4.0))
        inset_ylim : tuple, optional
            Y-axis limits for inset (e.g., (0, 3.0))
        inset_bbox : list or tuple, optional
            Inset bounding box in data coordinates [xmin, xmax, ymin, ymax]
            Example: [6.0, 10.0, 0.5, 2.0] places inset from x=6-10, y=0.5-2.0
        shell_alpha : float
            Transparency of shell region shading (0-1), default=0.35
        xlabel_fontsize : int
            Font size for x-axis label, default=12
        ylabel_fontsize : int
            Font size for y-axis label, default=12
        title_fontsize : int
            Font size for title, default=14
        legend_fontsize : int
            Font size for legend, default=10
        tick_fontsize : int
            Font size for axis tick labels, default=10
        shell_label_fontsize : int
            Font size for shell region labels, default=10
        save_log : bool
            Whether to save analysis log to file, default=True
        
        Returns
        -------
        comparison_data : dict
            Dictionary with RDF data for each concentration
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        
        # Initialize log file
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"RDF COMPARISON ANALYSIS: {ion_type}-{partner_type}")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Partner Type: {partner_type}")
        log_lines.append(f"Plot Range: {plot_range} Å")
        log_lines.append("")
        
        partner = partner_type
        
        # Collect RDF data
        comparison_data = {}
        loaded_shell_boundaries = {}
        
        log_lines.append("LOADING RDF DATA")
        log_lines.append("-"*80)
        
        for conc_key in sorted(self.loaded_data.keys()):
            conc_data = self.loaded_data[conc_key]
            
            if 'rdfs' not in conc_data:
                log_lines.append(f"⚠ {conc_key}: No RDF data found")
                continue
            
            rdfs = conc_data['rdfs']
            
            possible_keys = [
                f"{ion_type}-{partner}",
                f"{ion_type}-{partner_type}",
            ]
            
            rdf_key = None
            for key in possible_keys:
                if key in rdfs:
                    rdf_key = key
                    break
            
            if rdf_key is None:
                log_lines.append(f"⚠ {conc_key}: RDF '{ion_type}-{partner}' not found")
                log_lines.append(f"  Available: {', '.join(list(rdfs.keys()))}")
                continue
            
            comparison_data[conc_key] = rdfs[rdf_key]
            
            if shell_boundaries is None:
                if 'modified_shell_boundaries' in conc_data and ion_type in conc_data['modified_shell_boundaries']:
                    loaded_shell_boundaries[conc_key] = conc_data['modified_shell_boundaries'][ion_type]
                    log_lines.append(f"✓ {conc_key}: Loaded RDF + MODIFIED shell boundaries")
                elif 'shell_boundaries' in conc_data and ion_type in conc_data['shell_boundaries']:
                    loaded_shell_boundaries[conc_key] = conc_data['shell_boundaries'][ion_type]
                    log_lines.append(f"✓ {conc_key}: Loaded RDF + original shell boundaries")
                else:
                    log_lines.append(f"✓ {conc_key}: Loaded RDF (no shell boundaries)")
            else:
                log_lines.append(f"✓ {conc_key}: Loaded RDF")
        
        if not comparison_data:
            log_lines.append("")
            log_lines.append("ERROR: No RDF data found to compare")
            
            if save_log:
                log_filename = f'rdf_comparison_{ion_type}_{partner}_ERROR.log'
                with open(log_filename, 'w') as f:
                    f.write('\n'.join(log_lines))
                print(f"Error log saved: {log_filename}")
            
            return None
        
        log_lines.append("")
        log_lines.append(f"Total concentrations: {len(comparison_data)}")
        log_lines.append("")
        
        # Shell boundaries
        shells_to_plot = None
        if shell_boundaries is not None:
            shells_to_plot = shell_boundaries
            log_lines.append("SHELL BOUNDARIES")
            log_lines.append("-"*80)
            log_lines.append("Source: Manually provided")
        elif loaded_shell_boundaries:
            first_conc = sorted(comparison_data.keys(), 
                            key=lambda x: float(x.replace('M', '')))[0]
            if first_conc in loaded_shell_boundaries:
                shells_to_plot = loaded_shell_boundaries[first_conc]
                log_lines.append("SHELL BOUNDARIES")
                log_lines.append("-"*80)
                log_lines.append(f"Source: Loaded from {first_conc}")
        else:
            log_lines.append("SHELL BOUNDARIES: None (plotting without regions)")
        
        if shells_to_plot:
            for shell_name, bounds in shells_to_plot.items():
                log_lines.append(f"  {shell_name}: {bounds['r_min']:.2f} - {bounds['r_max']:.2f} Å")
        
        log_lines.append("")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define color generation function (needed for both main and inset)
        def get_blue_saturation_colors_from_00c5ff(n_shells):
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
        
        # Store shell colors for use in inset
        shell_colors_map = {}
        bulk_color = None
        bulk_start = None
        
        # Add shell shading to main plot
        if shells_to_plot:
            shell_items = [(k, v) for k, v in shells_to_plot.items() if k.startswith('shell_')]
            n_shells = len(shell_items)
            
            all_colors = get_blue_saturation_colors_from_00c5ff(n_shells)
            shell_colors = all_colors[:-1]
            bulk_color = all_colors[-1]
            
            shell_items.sort(key=lambda x: x[1]['r_min'])
            
            # Add shading and store color mapping
            for i, (shell_name, bounds) in enumerate(shell_items):
                color = shell_colors[i]
                ax.axvspan(bounds['r_min'], bounds['r_max'], alpha=shell_alpha, color=color, zorder=0)
                # Store color for each distance range
                shell_colors_map[(bounds['r_min'], bounds['r_max'])] = color
            
            last_shell = list(shells_to_plot.values())[-1]
            bulk_start = last_shell['r_max'] if 'r_max' in last_shell else last_shell[1]
            if bulk_start < plot_range:
                ax.axvspan(bulk_start, plot_range, alpha=shell_alpha, color=bulk_color, zorder=0)
        
        # Plot RDFs
        log_lines.append("PLOT DATA")
        log_lines.append("-"*80)
        
        sorted_conc_keys = sorted(comparison_data.keys(), key=lambda x: float(x.replace('M', '')))
        
        for conc_key in sorted_conc_keys:
            rdf_data = comparison_data[conc_key]
            label = self.concentration_data[conc_key].get('label', conc_key)
            color = self.concentration_data[conc_key].get('color', None)
            
            ax.plot(rdf_data['bins'], rdf_data['rdf'], 
                label=label, color=color, linewidth=2, zorder=2)
            
            log_lines.append(f"  {label}: color={color}")
        
        log_lines.append("")
        
        # Axis properties
        ax.set_xlabel('Distance (Å)', fontsize=xlabel_fontsize)
        ax.set_ylabel('g(r)', fontsize=ylabel_fontsize)
        ax.set_title(f'RDF Comparison: {ion_type}-{partner}', fontsize=title_fontsize, fontweight='bold', pad=20)
        ax.set_xlim(0, plot_range)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        # Add inset
        if add_inset:
            if inset_bbox is None:
                ax_inset = ax.inset_axes([0.55, 0.55, 0.35, 0.35])
                log_lines.append("INSET: Position = axes fraction [0.55, 0.55, 0.35, 0.35]")
            else:
                xmin, xmax, ymin, ymax = inset_bbox
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                
                left = (xmin - xlim[0]) / (xlim[1] - xlim[0])
                width = (xmax - xmin) / (xlim[1] - xlim[0])
                bottom = (ymin - ylim[0]) / (ylim[1] - ylim[0])
                height = (ymax - ymin) / (ylim[1] - ylim[0])
                
                ax_inset = ax.inset_axes([left, bottom, width, height])
                log_lines.append(f"INSET: Position = data coords x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]")
            
            # FIXED: Add shell shading to inset (inherit from main plot)
            if shells_to_plot and inset_xlim:
                inset_xmin, inset_xmax = inset_xlim
                
                # Add shading for shells that overlap with inset x-range
                for (r_min, r_max), color in shell_colors_map.items():
                    # Check if shell overlaps with inset range
                    if r_max > inset_xmin and r_min < inset_xmax:
                        # Calculate overlapping region
                        shade_min = max(r_min, inset_xmin)
                        shade_max = min(r_max, inset_xmax)
                        ax_inset.axvspan(shade_min, shade_max, alpha=shell_alpha, color=color, zorder=0)
                
                # Add bulk shading if it overlaps with inset
                if bulk_start is not None and bulk_start < inset_xmax:
                    shade_min = max(bulk_start, inset_xmin)
                    ax_inset.axvspan(shade_min, inset_xmax, alpha=shell_alpha, color=bulk_color, zorder=0)
            
            # Plot RDF data in inset
            for conc_key in sorted_conc_keys:
                rdf_data = comparison_data[conc_key]
                color = self.concentration_data[conc_key].get('color', None)
                ax_inset.plot(rdf_data['bins'], rdf_data['rdf'], color=color, linewidth=1.5, zorder=2)
            
            # Set inset limits
            if inset_xlim:
                ax_inset.set_xlim(inset_xlim)
                log_lines.append(f"  X-limits: {inset_xlim}")
            if inset_ylim:
                ax_inset.set_ylim(inset_ylim)
                log_lines.append(f"  Y-limits: {inset_ylim}")
            
            # Inset styling
            ax_inset.tick_params(labelsize=max(6, tick_fontsize - 2))
            ax_inset.grid(False)
            
            # REMOVED: ax.indicate_inset_zoom() - No zoom indicator lines
            
            log_lines.append("")
        
        # Shell labels
        if shells_to_plot:
            y_min, y_max = ax.get_ylim()
            label_y_pos = y_max * 0.98
            
            for i, (shell_name, bounds) in enumerate(shell_items):
                mid_point = (bounds['r_min'] + bounds['r_max']) / 2
                label_text = shell_name.replace('_', ' ').title()
                ax.text(mid_point, label_y_pos, label_text,
                    ha='center', va='top', fontsize=shell_label_fontsize, fontweight='bold', color='black')
            
            if bulk_start < plot_range:
                bulk_mid = (bulk_start + plot_range) / 2
                ax.text(bulk_mid, label_y_pos, 'Bulk',
                    ha='center', va='top', fontsize=shell_label_fontsize, fontweight='bold', color='black')
        
        ax.legend(loc='upper right', fontsize=legend_fontsize, frameon=False)
        
        # Settings summary
        log_lines.append("PLOT SETTINGS")
        log_lines.append("-"*80)
        log_lines.append(f"Shell alpha: {shell_alpha}")
        log_lines.append(f"Font sizes: xlabel={xlabel_fontsize}, ylabel={ylabel_fontsize}, title={title_fontsize}")
        log_lines.append(f"            legend={legend_fontsize}, ticks={tick_fontsize}, shell_labels={shell_label_fontsize}")
        log_lines.append("")
        
        plt.tight_layout()
        
        # Save files
        inset_suffix = '_inset' if add_inset else ''
        
        if save_plots:
            filename = f'rdf_comparison_{ion_type}_{partner}{inset_suffix}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            log_lines.append(f"PLOT SAVED: {filename}")
            print(f"Plot saved: {filename}")
        
        if save_log:
            log_filename = f'rdf_comparison_{ion_type}_{partner}{inset_suffix}_analysis.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Analysis log saved: {log_filename}")
        
        plt.show()
        
        return comparison_data 


    def compare_ion_pairing_rdfs_across_concentrations(self, ion_type, counter_ion='Cl', 
                                                       save_plots=True, save_log=True,
                                                       plot_range=12, show_cutoffs=True,
                                                       xlabel_fontsize=12, ylabel_fontsize=12,
                                                       title_fontsize=14, legend_fontsize=10, 
                                                       tick_fontsize=10):
        '''
        Compare ion pairing RDFs (cation-anion) across all concentrations.
        Shows the cutoff boundaries for CIP, SIP, DSIP, and FI regions with labels at top.
        
        Parameters
        ----------
        ion_type : str
            Ion type (e.g., 'Na', 'Mg')
        counter_ion : str
            Counter ion type (e.g., 'Cl'), default='Cl'
        save_plots : bool
            Whether to save the plot
        save_log : bool
            Whether to save analysis log
        plot_range : float
            Maximum r value for plotting (Å), default=12
        show_cutoffs : bool
            Whether to show ion pairing cutoff boundaries, default=True
        xlabel_fontsize : int
            Font size for x-axis label, default=12
        ylabel_fontsize : int
            Font size for y-axis label, default=12
        title_fontsize : int
            Font size for title, default=14
        legend_fontsize : int
            Font size for legend, default=10
        tick_fontsize : int
            Font size for axis tick labels, default=10
        
        Returns
        -------
        comparison_data : dict
            Dictionary with RDF and cutoff data for each concentration
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"ION PAIRING RDF COMPARISON: {ion_type}-{counter_ion}")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append("")
        
        # Collect data
        comparison_data = {}
        skipped_concentrations = []
        
        log_lines.append("LOADING RDF AND ION PAIRING DATA")
        log_lines.append("-"*80)
        
        for conc_key in sorted(self.loaded_data.keys()):
            conc_data = self.loaded_data[conc_key]
            
            # Look for RDF data
            rdf_data = None
            if 'rdfs' in conc_data:
                # Try different RDF key formats
                possible_keys = [f'{ion_type}-{counter_ion}', 'ci-ai']
                
                for key in possible_keys:
                    if key in conc_data['rdfs']:
                        rdf_data = conc_data['rdfs'][key]
                        break
            
            # Look for ion pairing cutoffs
            cutoffs = None
            if 'ion_pairing_cutoffs' in conc_data and ion_type in conc_data['ion_pairing_cutoffs']:
                cutoff_info = conc_data['ion_pairing_cutoffs'][ion_type]
                if 'ion_pairs' in cutoff_info:
                    cutoffs = cutoff_info['ion_pairs']
            
            if rdf_data is not None:
                comparison_data[conc_key] = {
                    'rdf': rdf_data,
                    'cutoffs': cutoffs
                }
                log_lines.append(f"✓ {conc_key}: Loaded RDF" + (" + cutoffs" if cutoffs else ""))
            else:
                skipped_concentrations.append(conc_key)
                log_lines.append(f"⚠ {conc_key}: No RDF data found")
        
        if skipped_concentrations:
            log_lines.append("")
            log_lines.append(f"Skipped: {', '.join(skipped_concentrations)}")
        
        if not comparison_data:
            log_lines.append("")
            log_lines.append("ERROR: No RDF data found")
            
            if save_log:
                log_filename = f'ion_pairing_rdf_comparison_{ion_type}_ERROR.log'
                with open(log_filename, 'w') as f:
                    f.write('\n'.join(log_lines))
                print(f"Error log saved: {log_filename}")
            
            return None
        
        log_lines.append("")
        log_lines.append(f"Total concentrations: {len(comparison_data)}")
        log_lines.append("")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get first concentration's cutoffs for shading (if available)
        first_conc = sorted(comparison_data.keys(), key=lambda x: float(x.replace('M', '')))[0]
        cutoffs_for_shading = comparison_data[first_conc].get('cutoffs')
        
        # Use the EXACT same colors as in determine_ion_pairing_cutoffs
        color_map = {
            'CIP': 'lightcoral',
            'SIP': 'lightblue',
            'DSIP': 'lightgreen',
            'FI': 'lightyellow'
        }
        
        # Store region labels for adding later (FIXED: use local variable)
        region_labels = []
        
        # Add ion pairing region shading WITHOUT adding to legend
        if show_cutoffs and cutoffs_for_shading:
            log_lines.append("ION PAIRING REGIONS (from first concentration)")
            log_lines.append("-"*80)
            
            for pair_type in ['CIP', 'SIP', 'DSIP', 'FI']:
                if pair_type in cutoffs_for_shading:
                    r_min, r_max = cutoffs_for_shading[pair_type]
                    # Ensure FI region is properly shaded (handle infinity)
                    r_max_plot = min(r_max, plot_range) if not np.isinf(r_max) else plot_range
                    
                    # Shade the region (NO label in legend)
                    ax.axvspan(r_min, r_max_plot, alpha=0.2, color=color_map[pair_type], 
                            zorder=0)
                    
                    # Calculate midpoint for label
                    mid_r = (r_min + r_max_plot) / 2
                    region_labels.append((mid_r, pair_type))
                    
                    # Format r_max separately to avoid f-string format specifier issue
                    r_max_str = f"{r_max:.2f}" if not np.isinf(r_max) else "inf"
                    log_lines.append(f"  {pair_type}: {r_min:.2f} - {r_max_str} Å")
            
            log_lines.append("")
        
        # Plot RDFs
        log_lines.append("RDF PLOT DATA")
        log_lines.append("-"*80)
        
        concentrations = sorted(comparison_data.keys(), key=lambda x: float(x.replace('M', '')))
        
        # Track max y value for region labels
        y_max = 0
        
        for conc_key in concentrations:
            rdf_info = comparison_data[conc_key]['rdf']
            label = self.concentration_data[conc_key].get('label', conc_key)
            color = self.concentration_data[conc_key].get('color', 'black')
            
            ax.plot(rdf_info['bins'], rdf_info['rdf'], 
                   color=color, linewidth=2.5, label=label, 
                   alpha=0.8, zorder=2)
            
            # Track max y value
            y_max = max(y_max, np.max(rdf_info['rdf']))
            
            log_lines.append(f"  {label}: color={color}")
        
        log_lines.append("")
        
        # FIXED: Add region labels at the top (use the region_labels list we created)
        if show_cutoffs and cutoffs_for_shading and region_labels:
            label_y = y_max * 1.05  # Position labels 5% above max RDF value
            
            for mid_r, pair_type in region_labels:
                ax.text(mid_r, label_y, pair_type, 
                       ha='center', va='bottom', fontweight='bold', 
                       fontsize=12, color='black',
                    #    bbox=dict(boxstyle='round,pad=0.3', 
                    #             facecolor=color_map[pair_type], 
                    #             alpha=0.3, edgecolor='none')
                    )
        
        # Customize plot
        ax.set_xlabel(r'Distance r (Å)', fontsize=xlabel_fontsize)
        ax.set_ylabel('g(r)', fontsize=ylabel_fontsize)
        ax.set_title(f'Ion Pairing RDF Comparison: {ion_type}-{counter_ion}', 
                    fontsize=title_fontsize, fontweight='bold')
        ax.set_xlim(0, plot_range)
        ax.set_ylim(bottom=0, top=y_max * 1.15)  # Extra space for labels
        ax.legend(fontsize=legend_fontsize, frameon=False, loc='best')
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        plt.tight_layout()
        
        # Save files
        if save_plots:
            filename = f'ion_pairing_rdf_comparison_{ion_type}_{counter_ion}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            log_lines.append(f"PLOT SAVED: {filename}")
            print(f"Plot saved: {filename}")
        
        if save_log:
            log_filename = f'ion_pairing_rdf_comparison_{ion_type}_{counter_ion}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Analysis log saved: {log_filename}")
        
        plt.show()
        
        return comparison_data


    def compare_dipole_distributions_all_CNs(self, ion_type, save_plots=True, save_log=True,
                                            xlabel_fontsize=12, ylabel_fontsize=12,
                                            title_fontsize=14, legend_fontsize=10, 
                                            tick_fontsize=10, bins=50, ncols=3, alpha=0.85):
        '''
        Compare water dipole angle distributions for ALL coordination numbers across concentrations.
        Creates a multi-panel figure with one subplot per coordination number.
        Uses viridis colormap with darker colors matching EquilibriumAnalysisOptimized.
        
        Parameters
        ----------
        ion_type : str
            Ion type to compare (e.g., 'Na', 'Mg', 'Cl')
        save_plots : bool
            Whether to save the plot
        save_log : bool
            Whether to save analysis log
        xlabel_fontsize : int
            Font size for x-axis label, default=12
        ylabel_fontsize : int
            Font size for y-axis label, default=12
        title_fontsize : int
            Font size for title, default=14
        legend_fontsize : int
            Font size for legend, default=10
        tick_fontsize : int
            Font size for axis tick labels, default=10
        bins : int
            Number of bins for histogram, default=50
        ncols : int
            Number of columns in multi-panel figure, default=3
        alpha : float
            Transparency level for histograms (0-1), default=0.85
            Lower values (e.g., 0.5) make histograms more transparent
        
        Returns
        -------
        comparison_data : dict
            Dictionary with dipole data for each concentration and CN
            Format: {conc_key: {CN: dipole_data}}
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"WATER DIPOLE DISTRIBUTION COMPARISON: {ion_type} (ALL CNs)")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Histogram Bins: {bins}")
        log_lines.append(f"Histogram Alpha: {alpha}")
        log_lines.append("")
        
        # Collect dipole data for all CNs
        comparison_data = {}
        all_cns = set()
        skipped_concentrations = []
        
        log_lines.append("LOADING WATER DIPOLE DATA (ALL CNs)")
        log_lines.append("-"*80)
        
        for conc_key in sorted(self.loaded_data.keys()):
            conc_data = self.loaded_data[conc_key]
            
            # Look for CN-specific dipole data
            if 'dipole_by_coordination' in conc_data:
                if ion_type in conc_data['dipole_by_coordination']:
                    cn_dipoles = conc_data['dipole_by_coordination'][ion_type]
                    
                    # Store all CNs for this concentration
                    comparison_data[conc_key] = {}
                    
                    for cn, cn_data in cn_dipoles.items():
                        if isinstance(cn_data, dict) and 'angles' in cn_data:
                            comparison_data[conc_key][cn] = cn_data
                            all_cns.add(cn)
                    
                    log_lines.append(f"✓ {conc_key}: Loaded {len(comparison_data[conc_key])} CNs")
                    log_lines.append(f"    CNs: {sorted(comparison_data[conc_key].keys())}")
                else:
                    log_lines.append(f"⚠ {conc_key}: Ion '{ion_type}' not found")
                    skipped_concentrations.append(conc_key)
            else:
                log_lines.append(f"⚠ {conc_key}: No 'dipole_by_coordination' data")
                skipped_concentrations.append(conc_key)
        
        if skipped_concentrations:
            log_lines.append("")
            log_lines.append(f"Skipped: {', '.join(skipped_concentrations)}")
        
        if not comparison_data:
            log_lines.append("")
            log_lines.append("ERROR: No dipole data found")
            
            if save_log:
                log_filename = f'dipole_comparison_{ion_type}_allCNs_ERROR.log'
                with open(log_filename, 'w') as f:
                    f.write('\n'.join(log_lines))
                print(f"Error log saved: {log_filename}")
            
            return None
        
        # Sort CNs
        sorted_cns = sorted(all_cns)
        n_cns = len(sorted_cns)
        
        log_lines.append("")
        log_lines.append(f"Total concentrations: {len(comparison_data)}")
        log_lines.append(f"Total CNs found: {n_cns}")
        log_lines.append(f"CNs: {sorted_cns}")
        log_lines.append("")
        
        # Calculate grid dimensions
        nrows = int(np.ceil(n_cns / ncols))
        
        # Create multi-panel figure
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        
        # Flatten axes array for easy indexing
        if n_cns == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        # Get sorted concentrations
        concentrations = sorted(comparison_data.keys(), key=lambda x: float(x.replace('M', '')))
        
        log_lines.append("DIPOLE ANGLE STATISTICS BY CN")
        log_lines.append("-"*80)
        log_lines.append("Color scheme: Viridis colormap (darker colors, matching EquilibriumAnalysisOptimized)")
        log_lines.append("")
        
        # Use darker colors from viridis colormap
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(concentrations)))
        
        # Plot each CN in its own subplot
        for i, cn in enumerate(sorted_cns):
            ax = axes[i]
            
            log_lines.append(f"\nCoordination Number: {cn}")
            log_lines.append("-"*40)
            log_lines.append(f"{'Concentration':<15} {'Mean (°)':<12} {'Std (°)':<12} {'N samples':<12}")
            
            # Plot histogram for each concentration
            for j, conc_key in enumerate(concentrations):
                if cn not in comparison_data[conc_key]:
                    continue
                
                dipole_info = comparison_data[conc_key][cn]
                label = self.concentration_data[conc_key].get('label', conc_key)
                
                if 'angles' in dipole_info:
                    angles = dipole_info['angles']
                    # Flatten if needed
                    if isinstance(angles, np.ndarray) and angles.ndim > 1:
                        angles = angles.flatten()
                    # Remove NaN values
                    angles = angles[~np.isnan(angles)]
                    
                    # Use user-specified alpha parameter
                    ax.hist(angles, bins=bins, alpha=alpha, color=colors[j],
                        edgecolor='black', linewidth=0.5, density=True, label=label)
                    
                    # Log statistics
                    mean = dipole_info.get('mean', np.mean(angles))
                    std = dipole_info.get('std', np.std(angles))
                    n_samples = dipole_info.get('n_samples', len(angles))
                    
                    log_lines.append(f"{label:<15} {mean:<12.2f} {std:<12.2f} {n_samples:<12}")
            
            # Customize subplot
            ax.set_xlabel('Angle (°)', fontsize=xlabel_fontsize-2)
            ax.set_ylabel('Probability Density', fontsize=ylabel_fontsize-2)
            ax.set_title(f'CN = {cn}', fontsize=title_fontsize-2, fontweight='bold')
            ax.set_xlim(0, 180)
            ax.tick_params(axis='both', labelsize=tick_fontsize-2)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Add legend only to first subplot
            if i == 0:
                ax.legend(fontsize=legend_fontsize-2, frameon=False, loc='best')
        
        # Hide unused subplots
        for i in range(n_cns, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        fig.suptitle(f'Water Dipole Angle Distributions: {ion_type} (All CNs)', 
                    fontsize=title_fontsize+2, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Save files
        if save_plots:
            filename = f'dipole_comparison_{ion_type}_allCNs_alpha{alpha:.2f}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            log_lines.append("")
            log_lines.append(f"PLOT SAVED: {filename}")
            print(f"Plot saved: {filename}")
        
        if save_log:
            log_filename = f'dipole_comparison_{ion_type}_allCNs_alpha{alpha:.2f}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Analysis log saved: {log_filename}")
        
        plt.show()
        
        return comparison_data


    def compare_dipole_distributions(self, ion_type, coordination_number=None, 
                                    save_plots=True, save_log=True,
                                    xlabel_fontsize=12, ylabel_fontsize=12,
                                    title_fontsize=14, legend_fontsize=10, 
                                    tick_fontsize=10, bins=50, alpha=0.85):
        '''
        Compare water dipole angle distributions for a specific ion across concentrations.
        UPDATED: Now uses viridis colormap and alpha parameter for transparency control.
        
        Parameters
        ----------
        ion_type : str
            Ion type to compare (e.g., 'Na', 'Mg', 'Cl')
        coordination_number : int, optional
            Specific coordination number to compare (e.g., 6)
            If None, uses overall dipole distribution
        save_plots : bool
            Whether to save the plot
        save_log : bool
            Whether to save analysis log
        xlabel_fontsize : int
            Font size for x-axis label, default=12
        ylabel_fontsize : int
            Font size for y-axis label, default=12
        title_fontsize : int
            Font size for title, default=14
        legend_fontsize : int
            Font size for legend, default=10
        tick_fontsize : int
            Font size for axis tick labels, default=10
        bins : int
            Number of bins for histogram, default=50
        alpha : float
            Transparency level for histograms (0-1), default=0.85
            Lower values (e.g., 0.5) make histograms more transparent
        
        Returns
        -------
        comparison_data : dict
            Dictionary with dipole data for each concentration
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        cn_text = f" (CN={coordination_number})" if coordination_number is not None else " (Overall)"
        log_lines.append(f"WATER DIPOLE DISTRIBUTION COMPARISON: {ion_type}{cn_text}")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Coordination Number: {coordination_number if coordination_number is not None else 'Overall (all CNs combined)'}")
        log_lines.append(f"Histogram Bins: {bins}")
        log_lines.append(f"Histogram Alpha: {alpha}")
        log_lines.append("")
        
        # Collect dipole data
        comparison_data = {}
        skipped_concentrations = []
        
        log_lines.append("LOADING WATER DIPOLE DATA")
        log_lines.append("-"*80)
        
        for conc_key in sorted(self.loaded_data.keys()):
            conc_data = self.loaded_data[conc_key]
            
            dipole_data = None
            
            if coordination_number is not None:
                # Look for CN-specific dipole data
                if 'dipole_by_coordination' in conc_data:
                    if ion_type in conc_data['dipole_by_coordination']:
                        cn_dipoles = conc_data['dipole_by_coordination'][ion_type]
                        if coordination_number in cn_dipoles:
                            dipole_data = cn_dipoles[coordination_number]
                            log_lines.append(f"✓ {conc_key}: Loaded CN-specific dipole data (CN={coordination_number})")
                            log_lines.append(f"    Mean: {dipole_data.get('mean', 'N/A'):.2f}°, N samples: {dipole_data.get('n_samples', 'N/A')}")
                        else:
                            log_lines.append(f"⚠ {conc_key}: CN={coordination_number} not found")
                            log_lines.append(f"    Available CNs: {list(cn_dipoles.keys())}")
                    else:
                        log_lines.append(f"⚠ {conc_key}: Ion '{ion_type}' not found in dipole_by_coordination")
                        log_lines.append(f"    Available ions: {list(conc_data['dipole_by_coordination'].keys())}")
                else:
                    log_lines.append(f"⚠ {conc_key}: No 'dipole_by_coordination' data")
            else:
                # Look for overall dipole distribution (all CNs combined)
                if 'water_dipole_distributions' in conc_data:
                    if ion_type in conc_data['water_dipole_distributions']:
                        dipole_data = conc_data['water_dipole_distributions'][ion_type]
                        log_lines.append(f"✓ {conc_key}: Loaded OVERALL dipole data (all CNs)")
                        log_lines.append(f"    Mean: {dipole_data.get('mean', 'N/A'):.2f}°, N samples: {dipole_data.get('n_samples', 'N/A')}")
                    else:
                        log_lines.append(f"⚠ {conc_key}: Ion '{ion_type}' not found in water_dipole_distributions")
                        log_lines.append(f"    Available ions: {list(conc_data['water_dipole_distributions'].keys())}")
                else:
                    # Fallback: Try to aggregate from dipole_by_coordination if overall not available
                    if 'dipole_by_coordination' in conc_data and ion_type in conc_data['dipole_by_coordination']:
                        cn_dipoles = conc_data['dipole_by_coordination'][ion_type]
                        
                        # Aggregate all CNs
                        all_angles = []
                        for cn, cn_data in cn_dipoles.items():
                            if 'angles' in cn_data:
                                angles = cn_data['angles']
                                if isinstance(angles, np.ndarray):
                                    angles = angles.flatten()
                                    angles = angles[~np.isnan(angles)]
                                    all_angles.extend(angles)
                        
                        if all_angles:
                            all_angles = np.array(all_angles)
                            dipole_data = {
                                'angles': all_angles,
                                'mean': np.mean(all_angles),
                                'std': np.std(all_angles),
                                'median': np.median(all_angles),
                                'n_samples': len(all_angles)
                            }
                            log_lines.append(f"✓ {conc_key}: Aggregated overall dipole data from all CNs")
                            log_lines.append(f"    Mean: {dipole_data['mean']:.2f}°, N samples: {dipole_data['n_samples']}")
                        else:
                            log_lines.append(f"⚠ {conc_key}: No dipole data available")
                    else:
                        log_lines.append(f"⚠ {conc_key}: No 'water_dipole_distributions' or 'dipole_by_coordination' data")
            
            if dipole_data is not None:
                comparison_data[conc_key] = dipole_data
            else:
                skipped_concentrations.append(conc_key)
        
        if skipped_concentrations:
            log_lines.append("")
            log_lines.append(f"Skipped: {', '.join(skipped_concentrations)}")
        
        if not comparison_data:
            log_lines.append("")
            log_lines.append("ERROR: No dipole data found")
            
            if save_log:
                cn_suffix = f"_CN{coordination_number}" if coordination_number is not None else "_overall"
                log_filename = f'dipole_comparison_{ion_type}{cn_suffix}_ERROR.log'
                with open(log_filename, 'w') as f:
                    f.write('\n'.join(log_lines))
                print(f"Error log saved: {log_filename}")
            
            return None
        
        log_lines.append("")
        log_lines.append(f"Total concentrations: {len(comparison_data)}")
        log_lines.append("")
        
        # Extract statistics
        log_lines.append("DIPOLE ANGLE STATISTICS")
        log_lines.append("-"*80)
        log_lines.append(f"{'Concentration':<15} {'Mean (°)':<12} {'Std Dev (°)':<12} {'Median (°)':<12} {'N samples':<12}")
        log_lines.append("-"*80)
        
        concentrations = sorted(comparison_data.keys(), key=lambda x: float(x.replace('M', '')))
        
        for conc_key in concentrations:
            dipole_info = comparison_data[conc_key]
            
            mean = dipole_info.get('mean', np.nan)
            std = dipole_info.get('std', np.nan)
            median = dipole_info.get('median', np.nan)
            n_samples = dipole_info.get('n_samples', 0)
            
            log_lines.append(f"{conc_key:<15} {mean:<12.2f} {std:<12.2f} {median:<12.2f} {n_samples:<12}")
        
        log_lines.append("")
        
        # UPDATED: Use viridis colormap for concentrations
        log_lines.append("Color scheme: Viridis colormap (darker colors, matching EquilibriumAnalysisOptimized)")
        log_lines.append("")
        
        # Generate colors from viridis
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(concentrations)))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Histograms with viridis colors
        for i, conc_key in enumerate(concentrations):
            dipole_info = comparison_data[conc_key]
            label = self.concentration_data[conc_key].get('label', conc_key)
            color = colors[i]  # Use viridis color
            
            if 'angles' in dipole_info:
                angles = dipole_info['angles']
                # Flatten if needed
                if isinstance(angles, np.ndarray) and angles.ndim > 1:
                    angles = angles.flatten()
                # Remove NaN values
                angles = angles[~np.isnan(angles)]
                
                # UPDATED: Use alpha parameter for transparency
                ax1.hist(angles, bins=bins, alpha=alpha, label=label, 
                        color=color, edgecolor='black', linewidth=0.5, density=True)
        
        ax1.set_xlabel('Angle (°)', fontsize=xlabel_fontsize)
        ax1.set_ylabel('Probability Density', fontsize=ylabel_fontsize)
        
        # Update title to reflect what's being plotted
        if coordination_number is not None:
            title_text = f'Water Dipole Angles (CN={coordination_number})'
        else:
            title_text = 'Water Dipole Angles (Overall)'
        ax1.set_title(title_text, fontsize=title_fontsize, fontweight='bold')
        
        ax1.legend(fontsize=legend_fontsize, frameon=False)
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        ax1.set_xlim(0, 180)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Plot 2: Mean angles with error bars
        conc_values = [float(c.replace('M', '')) for c in concentrations]
        means = [comparison_data[c].get('mean', np.nan) for c in concentrations]
        stds = [comparison_data[c].get('std', np.nan) for c in concentrations]
        
        ax2.errorbar(conc_values, means, yerr=stds, 
                    marker='o', markersize=10, capsize=5, linewidth=2.5,
                    color='black', markerfacecolor='#00c5ff',
                    markeredgecolor='black', markeredgewidth=1.5)
        
        ax2.set_xlabel('Concentration (M)', fontsize=xlabel_fontsize)
        ax2.set_ylabel('Mean Dipole Angle (°)', fontsize=ylabel_fontsize)
        ax2.set_title('Mean Angle vs Concentration', fontsize=title_fontsize, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.set_ylim(0, 180)
        
        plt.tight_layout()
        
        # Save files
        if save_plots:
            cn_suffix = f"_CN{coordination_number}" if coordination_number is not None else "_overall"
            filename = f'dipole_comparison_{ion_type}{cn_suffix}_alpha{alpha:.2f}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            log_lines.append(f"PLOT SAVED: {filename}")
            print(f"Plot saved: {filename}")
        
        if save_log:
            cn_suffix = f"_CN{coordination_number}" if coordination_number is not None else "_overall"
            log_filename = f'dipole_comparison_{ion_type}{cn_suffix}_alpha{alpha:.2f}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Analysis log saved: {log_filename}")
        
        plt.show()
        
        return comparison_data



    def compare_polyhedron_sizes(self, ion_type, save_plots=True, save_log=True,
                                xlabel_fontsize=12, ylabel_fontsize=12,
                                title_fontsize=14, legend_fontsize=10, 
                                tick_fontsize=10, alpha=0.6, bins=50):
        '''
        Compare polyhedron volumes for a specific ion across concentrations.
        
        Parameters
        ----------
        ion_type : str
            Ion type to compare (e.g., 'Na', 'Mg', 'Cl')
        save_plots : bool
            Whether to save the plot
        save_log : bool
            Whether to save analysis log
        xlabel_fontsize : int
            Font size for x-axis label, default=12
        ylabel_fontsize : int
            Font size for y-axis label, default=12
        title_fontsize : int
            Font size for title, default=14
        legend_fontsize : int
            Font size for legend, default=10
        tick_fontsize : int
            Font size for axis tick labels, default=10
        alpha : float
            Transparency for histogram, default=0.6
        bins : int
            Number of histogram bins, default=50
        
        Returns
        -------
        comparison_data : dict
            Dictionary with polyhedron data for each concentration
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"POLYHEDRON VOLUME COMPARISON: {ion_type}")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append("")
        
        # Collect polyhedron data
        comparison_data = {}
        skipped_concentrations = []
        
        log_lines.append("LOADING POLYHEDRON DATA")
        log_lines.append("-"*80)
        
        for conc_key in sorted(self.loaded_data.keys()):
            conc_data = self.loaded_data[conc_key]
            
            polyhedron_data = None
            
            # Try multiple possible locations for polyhedron data
            possible_locations = [
                ('polyhedron_sizes', ion_type),
                ('polyhedron_volumes', ion_type),
                ('coordination_polyhedra', ion_type),
                ('polyhedra', ion_type),
            ]
            
            for data_key, ion_key in possible_locations:
                if data_key in conc_data:
                    if isinstance(conc_data[data_key], dict) and ion_key in conc_data[data_key]:
                        polyhedron_data = conc_data[data_key][ion_key]
                        log_lines.append(f"✓ {conc_key}: Found polyhedron data in '{data_key}'")
                        break
            
            # If still not found, check if data is stored directly (not nested)
            if polyhedron_data is None:
                for key in conc_data.keys():
                    if 'polyhedron' in key.lower() and ion_type in str(key):
                        polyhedron_data = conc_data[key]
                        log_lines.append(f"✓ {conc_key}: Found polyhedron data in '{key}'")
                        break
            
            if polyhedron_data is not None:
                # Validate that data has required fields
                if isinstance(polyhedron_data, dict):
                    if 'volumes' in polyhedron_data or 'mean' in polyhedron_data:
                        comparison_data[conc_key] = polyhedron_data
                        
                        # Log what we found
                        if 'volumes' in polyhedron_data:
                            vols = polyhedron_data['volumes']
                            if isinstance(vols, np.ndarray):
                                vols = vols.flatten()
                                vols = vols[~np.isnan(vols)]
                                n_samples = len(vols)
                                log_lines.append(f"    Contains {n_samples} volume samples")
                        elif 'mean' in polyhedron_data:
                            log_lines.append(f"    Mean volume: {polyhedron_data['mean']:.3f} Ų")
                    else:
                        log_lines.append(f"⚠ {conc_key}: Polyhedron data missing 'volumes' or 'mean'")
                        log_lines.append(f"    Available keys: {list(polyhedron_data.keys())}")
                        skipped_concentrations.append(conc_key)
                else:
                    log_lines.append(f"⚠ {conc_key}: Polyhedron data is not a dict: {type(polyhedron_data)}")
                    skipped_concentrations.append(conc_key)
            else:
                log_lines.append(f"⚠ {conc_key}: No polyhedron data found")
                skipped_concentrations.append(conc_key)
        
        if skipped_concentrations:
            log_lines.append("")
            log_lines.append(f"Skipped: {', '.join(skipped_concentrations)}")
        
        if not comparison_data:
            log_lines.append("")
            log_lines.append(f"ERROR: No polyhedron data found for {ion_type}")
            log_lines.append("")
            log_lines.append("SUGGESTION: Run the following to debug:")
            log_lines.append(f"  comp.debug_data_structure(ion_type='{ion_type}')")
            
            if save_log:
                log_filename = f'polyhedron_comparison_{ion_type}_ERROR.log'
                with open(log_filename, 'w') as f:
                    f.write('\n'.join(log_lines))
                print(f"Error log saved: {log_filename}")
            
            print("")
            print(f"ERROR: No polyhedron data found for {ion_type}")
            print(f"Run: comp.debug_data_structure(ion_type='{ion_type}') to inspect data structure")
            
            return None
        
        log_lines.append("")
        log_lines.append(f"Total concentrations with data: {len(comparison_data)}")
        log_lines.append("")
        
        # Extract statistics
        log_lines.append("POLYHEDRON VOLUME STATISTICS")
        log_lines.append("-"*80)
        log_lines.append(f"{'Concentration':<15} {'Mean (Å³)':<15} {'Std Dev (Å³)':<15} {'Median (Å³)':<15} {'N samples':<12}")
        log_lines.append("-"*80)
        
        concentrations = sorted(comparison_data.keys(), key=lambda x: float(x.replace('M', '')))
        
        for conc_key in concentrations:
            poly_info = comparison_data[conc_key]
            
            # Calculate statistics if not already present
            if 'volumes' in poly_info and 'mean' not in poly_info:
                volumes = poly_info['volumes']
                if isinstance(volumes, np.ndarray):
                    volumes = volumes.flatten()
                    volumes = volumes[~np.isnan(volumes)]
                    poly_info['mean'] = np.mean(volumes)
                    poly_info['std'] = np.std(volumes)
                    poly_info['median'] = np.median(volumes)
                    poly_info['n_samples'] = len(volumes)
            
            mean = poly_info.get('mean', np.nan)
            std = poly_info.get('std', np.nan)
            median = poly_info.get('median', np.nan)
            n_samples = poly_info.get('n_samples', 0)
            
            log_lines.append(f"{conc_key:<15} {mean:<15.3f} {std:<15.3f} {median:<15.3f} {n_samples:<12}")
        
        log_lines.append("")
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Use viridis colormap for concentrations
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(concentrations)))
        
        # Plot 1: Distribution of polyhedron volumes (histogram)
        for i, conc_key in enumerate(concentrations):
            poly_info = comparison_data[conc_key]
            label = self.concentration_data[conc_key].get('label', conc_key)
            color = colors[i]
            
            if 'volumes' in poly_info:
                volumes = poly_info['volumes']
                # Flatten if needed
                if isinstance(volumes, np.ndarray) and volumes.ndim > 1:
                    volumes = volumes.flatten()
                # Remove NaN values
                volumes = volumes[~np.isnan(volumes)]
                
                if len(volumes) > 0:
                    ax1.hist(volumes, bins=bins, alpha=alpha, label=label, 
                            color=color, edgecolor='black', linewidth=0.5, density=True)
                    log_lines.append(f"Plotted histogram for {label}: {len(volumes)} samples")
        
        ax1.set_xlabel('Polyhedron Volume (Å³)', fontsize=xlabel_fontsize)
        ax1.set_ylabel('Probability Density', fontsize=ylabel_fontsize)
        ax1.set_title(f'Polyhedron Volume Distributions: {ion_type}', fontsize=title_fontsize, fontweight='bold')
        ax1.legend(fontsize=legend_fontsize, frameon=False)
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Plot 2: Mean volume vs concentration with error bars
        conc_values = [float(c.replace('M', '')) for c in concentrations]
        means = [comparison_data[c].get('mean', np.nan) for c in concentrations]
        stds = [comparison_data[c].get('std', np.nan) for c in concentrations]
        
        # Filter out NaN values
        valid_data = [(c, m, s) for c, m, s in zip(conc_values, means, stds) if not np.isnan(m)]
        
        if valid_data:
            conc_values, means, stds = zip(*valid_data)
            
            ax2.errorbar(conc_values, means, yerr=stds, 
                        marker='o', markersize=10, capsize=5, linewidth=2.5,
                        color='black', markerfacecolor='#00c5ff',
                        markeredgecolor='black', markeredgewidth=1.5)
            
            log_lines.append(f"Plotted mean vs concentration: {len(conc_values)} points")
        
        ax2.set_xlabel('Concentration (M)', fontsize=xlabel_fontsize)
        ax2.set_ylabel('Mean Polyhedron Volume (Å³)', fontsize=ylabel_fontsize)
        ax2.set_title('Mean Volume vs Concentration', fontsize=title_fontsize, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # Save files
        if save_plots:
            filename = f'polyhedron_comparison_{ion_type}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            log_lines.append("")
            log_lines.append(f"PLOT SAVED: {filename}")
            print(f"Plot saved: {filename}")
        
        if save_log:
            log_filename = f'polyhedron_comparison_{ion_type}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Analysis log saved: {log_filename}")
        
        plt.show()
        
        return comparison_data


    def compare_residence_times(self, ion_type, residence_type='water', shell='shell_1',
                                save_plots=True, save_log=True,
                                xlabel_fontsize=12, ylabel_fontsize=12,
                                title_fontsize=14, legend_fontsize=10, 
                                tick_fontsize=10, alpha=0.6, bins=50,
                                add_inset=False, inset_xlim=None, inset_ylim=None,
                                inset_bbox=None):
        '''
        Compare residence times (water or ion pairing) for a specific ion across concentrations.
        UPDATED: Now includes interactive inset support.
        
        Parameters
        ----------
        ion_type : str
            Ion type to compare (e.g., 'Na', 'Mg', 'Cl')
        residence_type : str
            Type of residence time to compare:
            - 'water': Water residence times in solvation shell
            - 'ion_pairing': Ion pairing lifetimes (for cations)
        shell : str or list of str
            Which shell(s) to analyze (for water residence times):
            - Single shell: 'shell_1', 'shell_2', 'shell_3'
            - Multiple shells: ['shell_1', 'shell_2', 'shell_3']
            When multiple shells provided, creates multi-panel comparison
            NOTE: Inset only works with single shell
        save_plots : bool
            Whether to save the plot
        save_log : bool
            Whether to save analysis log
        xlabel_fontsize : int
            Font size for x-axis label, default=12
        ylabel_fontsize : int
            Font size for y-axis label, default=12
        title_fontsize : int
            Font size for title, default=14
        legend_fontsize : int
            Font size for legend, default=10
        tick_fontsize : int
            Font size for axis tick labels, default=10
        alpha : float
            Transparency for histogram, default=0.6
        bins : int
            Number of histogram bins, default=50
        add_inset : bool
            Whether to add an inset zoom plot (only works with single shell)
        inset_xlim : tuple, optional
            X-axis limits for inset (e.g., (10, 50))
        inset_ylim : tuple, optional
            Y-axis limits for inset (e.g., (0, 0.05))
        inset_bbox : list or tuple, optional
            Inset bounding box in data coordinates [xmin, xmax, ymin, ymax]
            Example: [150, 350, 0.002, 0.015] places inset from x=150-350, y=0.002-0.015
        
        Returns
        -------
        comparison_data : dict
            Dictionary with residence time data for each concentration and shell
            Format: {conc_key: {shell: residence_data}} or {conc_key: residence_data}
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        
        # Convert shell to list if single string provided
        if isinstance(shell, str):
            shells_to_analyze = [shell]
            multi_shell = False
        else:
            shells_to_analyze = shell
            multi_shell = True
        
        # Check if inset is requested with multiple shells
        if add_inset and multi_shell:
            print("WARNING: Inset only supported for single shell. Disabling inset.")
            add_inset = False
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"RESIDENCE TIME COMPARISON: {ion_type} ({residence_type})")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Residence Type: {residence_type}")
        if residence_type == 'water':
            log_lines.append(f"Shells: {shells_to_analyze}")
        if add_inset:
            log_lines.append(f"Inset: Enabled")
        log_lines.append("")
        
        # Collect residence time data for all requested shells
        comparison_data = {}
        skipped_concentrations = []
        
        log_lines.append("LOADING RESIDENCE TIME DATA")
        log_lines.append("-"*80)
        
        for conc_key in sorted(self.loaded_data.keys()):
            conc_data = self.loaded_data[conc_key]
            
            if residence_type == 'water':
                # Handle nested structure in water_residence_times
                if 'water_residence_times' in conc_data:
                    if ion_type in conc_data['water_residence_times']:
                        residence_info = conc_data['water_residence_times'][ion_type]
                        
                        # Check if 'shells' key exists
                        if 'shells' in residence_info:
                            shells_data = residence_info['shells']
                            
                            # Initialize storage for this concentration
                            if multi_shell:
                                comparison_data[conc_key] = {}
                            
                            # Extract data for each requested shell
                            found_any = False
                            for shell_name in shells_to_analyze:
                                if shell_name in shells_data:
                                    shell_data = shells_data[shell_name]
                                    
                                    # Extract residence times
                                    times = None
                                    possible_time_keys = ['residence_times', 'times', 'lifetimes', 'tau']
                                    
                                    for time_key in possible_time_keys:
                                        if time_key in shell_data:
                                            times = shell_data[time_key]
                                            break
                                    
                                    if times is not None:
                                        if not isinstance(times, np.ndarray):
                                            times = np.array(times)
                                        times = times[~np.isnan(times)]
                                        
                                        residence_data = {
                                            'times': times,
                                            'mean': shell_data.get('mean_residence_time', 
                                                shell_data.get('mean', np.mean(times) if len(times) > 0 else np.nan)),
                                            'std': shell_data.get('std_residence_time',
                                                shell_data.get('std', np.std(times) if len(times) > 0 else np.nan)),
                                            'median': np.median(times) if len(times) > 0 else np.nan,
                                            'n_samples': len(times)
                                        }
                                        
                                        if multi_shell:
                                            comparison_data[conc_key][shell_name] = residence_data
                                        else:
                                            comparison_data[conc_key] = residence_data
                                        
                                        found_any = True
                            
                            if found_any:
                                if multi_shell:
                                    log_lines.append(f"✓ {conc_key}: Loaded {len(comparison_data[conc_key])} shells")
                                else:
                                    log_lines.append(f"✓ {conc_key}: Loaded {shells_to_analyze[0]}")
                            else:
                                skipped_concentrations.append(conc_key)
                                log_lines.append(f"⚠ {conc_key}: No data for requested shells")
                        else:
                            skipped_concentrations.append(conc_key)
                            log_lines.append(f"⚠ {conc_key}: No 'shells' key")
                    else:
                        skipped_concentrations.append(conc_key)
                        log_lines.append(f"⚠ {conc_key}: Ion '{ion_type}' not found")
                else:
                    skipped_concentrations.append(conc_key)
                    log_lines.append(f"⚠ {conc_key}: No 'water_residence_times'")
            
            elif residence_type == 'ion_pairing':
                # Ion pairing logic (unchanged)
                residence_data = None
                possible_locations = [
                    ('ion_pairing_lifetimes', ion_type),
                    ('ion_pair_dynamics', ion_type),
                    ('pairing_lifetimes', ion_type),
                ]
                
                for data_key, ion_key in possible_locations:
                    if data_key in conc_data:
                        if isinstance(conc_data[data_key], dict) and ion_key in conc_data[data_key]:
                            residence_data = conc_data[data_key][ion_key]
                            comparison_data[conc_key] = residence_data
                            log_lines.append(f"✓ {conc_key}: Found ion pairing data in '{data_key}'")
                            break
                
                if residence_data is None:
                    skipped_concentrations.append(conc_key)
                    log_lines.append(f"⚠ {conc_key}: No ion pairing data found")
        
        if skipped_concentrations:
            log_lines.append("")
            log_lines.append(f"Skipped: {', '.join(skipped_concentrations)}")
        
        if not comparison_data:
            log_lines.append("")
            log_lines.append(f"ERROR: No {residence_type} residence data found for {ion_type}")
            
            if save_log:
                shell_suffix = f"_{'_'.join(shells_to_analyze)}" if residence_type == 'water' else ""
                log_filename = f'residence_comparison_{ion_type}_{residence_type}{shell_suffix}_ERROR.log'
                with open(log_filename, 'w') as f:
                    f.write('\n'.join(log_lines))
                print(f"Error log saved: {log_filename}")
            
            print("\n".join(log_lines[-10:]))
            return None
        
        log_lines.append("")
        log_lines.append(f"Total concentrations with data: {len(comparison_data)}")
        log_lines.append("")
        
        # Get sorted concentrations
        concentrations = sorted(comparison_data.keys(), key=lambda x: float(x.replace('M', '')))
        
        # Use viridis colormap for concentrations
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(concentrations)))
        
        # PLOTTING: Different logic for single vs multiple shells
        if multi_shell:
            # Multi-panel plot: one subplot per shell
            n_shells = len(shells_to_analyze)
            ncols = min(3, n_shells)
            nrows = int(np.ceil(n_shells / ncols))
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows))
            if n_shells == 1:
                axes = np.array([axes])
            else:
                axes = axes.flatten()
            
            log_lines.append("RESIDENCE TIME STATISTICS (BY SHELL)")
            log_lines.append("-"*80)
            
            for i, shell_name in enumerate(shells_to_analyze):
                ax = axes[i]
                
                log_lines.append(f"\n{shell_name.upper()}:")
                log_lines.append(f"{'Concentration':<15} {'Mean (ps)':<15} {'Std (ps)':<15} {'N samples':<12}")
                log_lines.append("-"*60)
                
                # Plot histogram for each concentration
                for j, conc_key in enumerate(concentrations):
                    if shell_name in comparison_data[conc_key]:
                        res_info = comparison_data[conc_key][shell_name]
                        label = self.concentration_data[conc_key].get('label', conc_key)
                        color = colors[j]
                        
                        if 'times' in res_info:
                            times = res_info['times']
                            if len(times) > 0:
                                ax.hist(times, bins=bins, alpha=alpha, label=label, 
                                    color=color, edgecolor='black', linewidth=0.5, density=True)
                        
                        # Log statistics
                        mean = res_info.get('mean', np.nan)
                        std = res_info.get('std', np.nan)
                        n_samples = res_info.get('n_samples', 0)
                        log_lines.append(f"{label:<15} {mean:<15.3f} {std:<15.3f} {n_samples:<12}")
                
                # Customize subplot
                ax.set_xlabel(f'Residence Time (ps)', fontsize=xlabel_fontsize-2)
                ax.set_ylabel('Probability Density', fontsize=ylabel_fontsize-2)
                ax.set_title(f'{shell_name.replace("_", " ").title()}', 
                            fontsize=title_fontsize, fontweight='bold')
                ax.tick_params(axis='both', labelsize=tick_fontsize-2)
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                
                # Add legend only to first subplot
                if i == 0:
                    ax.legend(fontsize=legend_fontsize-2, frameon=False, loc='best')
            
            # Hide unused subplots
            for i in range(n_shells, len(axes)):
                axes[i].set_visible(False)
            
            # Overall title
            fig.suptitle(f'Water Residence Times: {ion_type} (Multiple Shells)', 
                        fontsize=title_fontsize+2, fontweight='bold', y=0.995)
            
            plt.tight_layout()
            
        else:
            # Single shell: two-panel plot (histogram + mean vs concentration)
            shell_name = shells_to_analyze[0]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            log_lines.append("RESIDENCE TIME STATISTICS")
            log_lines.append("-"*80)
            log_lines.append(f"{'Concentration':<15} {'Mean (ps)':<15} {'Std Dev (ps)':<15} {'Median (ps)':<15} {'N samples':<12}")
            log_lines.append("-"*80)
            
            # Track max y value for inset positioning
            y_max = 0


            # Plot 1: Histogram distributions
            for i, conc_key in enumerate(concentrations):
                res_info = comparison_data[conc_key]
                label = self.concentration_data[conc_key].get('label', conc_key)
                color = colors[i]
                
                if 'times' in res_info:
                    times = res_info['times']
                    if len(times) > 0:
                        counts, edges, patches = ax1.hist(times, bins=bins, alpha=alpha, label=label, 
                                color=color, edgecolor='black', linewidth=0.5, density=True)
                        y_max = max(y_max, np.max(counts))
            
            ax1.set_xlabel('Residence Time (ps)', fontsize=xlabel_fontsize)
            ax1.set_ylabel('Probability Density', fontsize=ylabel_fontsize)
            shell_text = f" ({shell_name.replace('_', ' ').title()})"
            ax1.set_title(f'Water Residence Time Distributions: {ion_type}{shell_text}', 
                        fontsize=title_fontsize, fontweight='bold')
            ax1.legend(fontsize=legend_fontsize, frameon=False)
            ax1.tick_params(axis='both', labelsize=tick_fontsize)
            ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # CRITICAL: Store original axis limits BEFORE creating inset
            original_xlim = ax1.get_xlim()
            original_ylim = ax1.get_ylim()
            
            log_lines.append(f"DEBUG: Original limits before inset: xlim={original_xlim}, ylim={original_ylim}")
            
            # Add inset to histogram plot
            if add_inset:
                # Create inset axes
                if inset_bbox is None:
                    # Default position in axes fraction coordinates
                    ax_inset = ax1.inset_axes([0.55, 0.55, 0.35, 0.35])
                    log_lines.append("INSET: Position = axes fraction [0.55, 0.55, 0.35, 0.35]")
                else:
                    # Convert data coordinates to axes fraction
                    xmin, xmax, ymin, ymax = inset_bbox
                    xlim, ylim = original_xlim, original_ylim  # Use stored limits
                    
                    left = (xmin - xlim[0]) / (xlim[1] - xlim[0])
                    width = (xmax - xmin) / (xlim[1] - xlim[0])
                    bottom = (ymin - ylim[0]) / (ylim[1] - ylim[0])
                    height = (ymax - ymin) / (ylim[1] - ylim[0])
                    
                    # Clamp to valid range
                    left = np.clip(left, 0, 0.95)
                    bottom = np.clip(bottom, 0, 0.95)
                    width = np.clip(width, 0.05, 1.0 - left)
                    height = np.clip(height, 0.05, 1.0 - bottom)
                    
                    ax_inset = ax1.inset_axes([left, bottom, width, height])
                    log_lines.append(f"INSET: Position = data coords x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]")
                    log_lines.append(f"       Axes fraction = [{left:.3f}, {bottom:.3f}, {width:.3f}, {height:.3f}]")
                
                # Plot data in inset (re-plot histograms)
                for i, conc_key in enumerate(concentrations):
                    res_info = comparison_data[conc_key]
                    color = colors[i]
                    
                    if 'times' in res_info:
                        times = res_info['times']
                        if len(times) > 0:
                            ax_inset.hist(times, bins=bins, alpha=alpha, 
                                        color=color, edgecolor='black', linewidth=0.5, density=True)
                
                # Set inset limits to zoom into specified region
                if inset_xlim:
                    ax_inset.set_xlim(inset_xlim)
                    log_lines.append(f"  Inset X-limits: {inset_xlim}")
                if inset_ylim:
                    ax_inset.set_ylim(inset_ylim)
                    log_lines.append(f"  Inset Y-limits: {inset_ylim}")
                
                # Customize inset appearance
                ax_inset.tick_params(labelsize=max(6, tick_fontsize - 2))
                ax_inset.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                
                # Add border to make inset stand out
                for spine in ax_inset.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.5)
                
                log_lines.append(f"DEBUG: Limits after inset creation: xlim={ax1.get_xlim()}, ylim={ax1.get_ylim()}")
                
                # CRITICAL FIX: Force restore original limits AFTER all inset setup
                ax1.set_xlim(original_xlim)
                ax1.set_ylim(original_ylim)
                
                # FORCE update - sometimes matplotlib needs this
                ax1.figure.canvas.draw_idle()
                
                log_lines.append(f"DEBUG: Limits after restoration: xlim={ax1.get_xlim()}, ylim={ax1.get_ylim()}")
                log_lines.append(f"  Parent axes RESTORED: xlim={original_xlim}, ylim={original_ylim}")
                log_lines.append("")
            
            # ADDITIONAL SAFETY: Ensure limits are correct one more time before continuing
            if add_inset:
                ax1.set_xlim(original_xlim)
                ax1.set_ylim(original_ylim)
            
            # Plot 2: Mean vs concentration
            conc_values = [float(c.replace('M', '')) for c in concentrations]
            means = [comparison_data[c].get('mean', np.nan) for c in concentrations]
            stds = [comparison_data[c].get('std', np.nan) for c in concentrations]
            
            valid_data = [(c, m, s) for c, m, s in zip(conc_values, means, stds) if not np.isnan(m)]
            
            if valid_data:
                conc_values, means, stds = zip(*valid_data)
                ax2.errorbar(conc_values, means, yerr=stds, 
                            marker='o', markersize=10, capsize=5, linewidth=2.5,
                            color='black', markerfacecolor='#00c5ff',
                            markeredgecolor='black', markeredgewidth=1.5)
            
            ax2.set_xlabel('Concentration (M)', fontsize=xlabel_fontsize)
            ax2.set_ylabel('Mean Residence Time (ps)', fontsize=ylabel_fontsize)
            ax2.set_title('Mean Residence Time vs Concentration', 
                        fontsize=title_fontsize, fontweight='bold')
            ax2.tick_params(axis='both', labelsize=tick_fontsize)
            ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax2.set_ylim(bottom=0)
            
            plt.tight_layout()
        
        log_lines.append("")
        
        # Save files
        if save_plots:
            if multi_shell:
                shell_suffix = f"_{'_'.join(shells_to_analyze)}"
            else:
                shell_suffix = f"_{shells_to_analyze[0]}"
            
            inset_suffix = '_inset' if add_inset else ''
            
            filename = f'residence_comparison_{ion_type}_{residence_type}{shell_suffix}{inset_suffix}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            log_lines.append(f"PLOT SAVED: {filename}")
            print(f"Plot saved: {filename}")
        
        if save_log:
            if multi_shell:
                shell_suffix = f"_{'_'.join(shells_to_analyze)}"
            else:
                shell_suffix = f"_{shells_to_analyze[0]}"
            
            inset_suffix = '_inset' if add_inset else ''
            
            log_filename = f'residence_comparison_{ion_type}_{residence_type}{shell_suffix}{inset_suffix}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Analysis log saved: {log_filename}")
        
        plt.show()
        
        return comparison_data


    def interactive_residence_inset_editor(self, ion_type, residence_type='water', 
                                        shell='shell_1', bins=50, alpha=0.6):
        '''
        Interactive editor for residence time insets - SIMPLIFIED IMAGE-BASED VERSION.
        Saves plot to file and displays as image, then asks for text input.
        
        Parameters
        ----------
        ion_type : str
            Ion type to analyze
        residence_type : str
            'water' or 'ion_pairing'
        shell : str
            Shell to analyze (for water residence times)
        bins : int
            Number of histogram bins
        alpha : float
            Histogram transparency
        
        Returns
        -------
        inset_params : dict
            Dictionary with inset parameters to use in compare_residence_times()
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded.")
            return None
        
        # Collect data
        comparison_data = {}
        for conc_key in sorted(self.loaded_data.keys()):
            conc_data = self.loaded_data[conc_key]
            
            if residence_type == 'water':
                if 'water_residence_times' in conc_data:
                    if ion_type in conc_data['water_residence_times']:
                        res_info = conc_data['water_residence_times'][ion_type]
                        if 'shells' in res_info and shell in res_info['shells']:
                            shell_data = res_info['shells'][shell]
                            
                            times = None
                            for time_key in ['residence_times', 'times', 'lifetimes']:
                                if time_key in shell_data:
                                    times = shell_data[time_key]
                                    break
                            
                            if times is not None:
                                if not isinstance(times, np.ndarray):
                                    times = np.array(times)
                                times = times[~np.isnan(times)]
                                comparison_data[conc_key] = {'times': times}
        
        if not comparison_data:
            print(f"ERROR: No residence time data found for {ion_type}")
            return None
        
        # Get concentrations and colors
        concentrations = sorted(comparison_data.keys(), key=lambda x: float(x.replace('M', '')))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(concentrations)))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        print("Creating residence time plot...")
        
        # Plot histograms
        for i, conc_key in enumerate(concentrations):
            res_info = comparison_data[conc_key]
            label = self.concentration_data[conc_key].get('label', conc_key)
            color = colors[i]
            
            times = res_info['times']
            ax.hist(times, bins=bins, alpha=alpha, label=label,
                color=color, edgecolor='black', linewidth=0.5, density=True)
            print(f"  {label}: {len(times)} samples")
        
        # Customize plot
        ax.set_xlabel('Residence Time (ps)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Residence Time Distribution: {ion_type} ({shell})\nUse this plot to choose inset parameters', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, frameon=False, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Get axis limits BEFORE saving
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Save to temporary file
        temp_filename = f'_temp_residence_inset_{ion_type}_{shell}.png'
        plt.savefig(temp_filename, dpi=150, bbox_inches='tight')
        
        print(f"\n✓ Plot saved: {temp_filename}")
        
        # Display the image using IPython BEFORE closing figure
        try:
            from IPython.display import Image, display
            print("\n" + "="*80)
            print("PLOT:")
            print("="*80)
            display(Image(filename=temp_filename))
            print("\n✓ Plot displayed above")
        except Exception as e:
            print(f"\n⚠ Could not display inline (error: {e})")
            print(f"   Please open the file manually: {temp_filename}")
        
        # NOW close the figure (AFTER displaying)
        plt.close(fig)
        
        # Provide parameter input instructions
        print("\n" + "="*80)
        print("INSET PARAMETER SELECTION")
        print("="*80)
        print("Look at the plot above to determine inset parameters.")
        print("")
        print("You need TWO things:")
        print("1. DATA REGION: What X and Y range to zoom into (inside the inset)")
        print("2. INSET POSITION: Where to place the inset box on the plot")
        print("="*80)
        print("")
        print(f"Current plot ranges:")
        print(f"  X-axis: {xlim[0]:.1f} to {xlim[1]:.1f} ps")
        print(f"  Y-axis: {ylim[0]:.5f} to {ylim[1]:.5f}")
        print("")
        
        # Get zoom region
        print("STEP 1: DATA REGION to zoom into")
        print("-"*80)
        print("Example: To zoom into 20-100 ps, enter: 20, 100")
        print("")
        
        while True:
            try:
                x_input = input("Enter inset X-axis range (xmin, xmax) in ps: ").strip()
                if x_input.lower() in ['q', 'quit', 'exit']:
                    print("Cancelled")
                    return None
                
                xmin, xmax = map(float, x_input.split(','))
                
                if xmin >= xmax:
                    print("ERROR: xmin must be less than xmax")
                    continue
                
                break
            except ValueError:
                print("ERROR: Invalid input. Use format: xmin, xmax")
            except Exception as e:
                print(f"ERROR: {e}")
        
        while True:
            try:
                y_input = input("Enter inset Y-axis range (ymin, ymax): ").strip()
                if y_input.lower() in ['q', 'quit', 'exit']:
                    print("Cancelled")
                    return None
                
                ymin, ymax = map(float, y_input.split(','))
                
                if ymin >= ymax:
                    print("ERROR: ymin must be less than ymax")
                    continue
                
                break
            except ValueError:
                print("ERROR: Invalid input. Use format: ymin, ymax")
            except Exception as e:
                print(f"ERROR: {e}")
        
        print(f"\n✓ Zoom region: X=[{xmin:.1f}, {xmax:.1f}], Y=[{ymin:.5f}, {ymax:.5f}]")
        print("")
        
        # Get inset position
        print("STEP 2: INSET POSITION on plot")
        print("-"*80)
        print("Specify where the inset box appears (in data coordinates)")
        print(f"Example: 150, 300, 0.005, 0.02")
        print(f"This would place the inset from x=150-300 ps, y=0.005-0.02")
        print("")
        
        while True:
            try:
                pos_input = input("Enter inset position (xmin, xmax, ymin, ymax): ").strip()
                if pos_input.lower() in ['q', 'quit', 'exit']:
                    print("Cancelled")
                    return None
                
                bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax = map(float, pos_input.split(','))
                
                if bbox_xmin >= bbox_xmax or bbox_ymin >= bbox_ymax:
                    print("ERROR: min values must be less than max values")
                    continue
                
                break
            except ValueError:
                print("ERROR: Invalid input. Use format: xmin, xmax, ymin, ymax")
            except Exception as e:
                print(f"ERROR: {e}")
        
        # Create parameters
        inset_params = {
            'add_inset': True,
            'inset_xlim': (xmin, xmax),
            'inset_ylim': (ymin, ymax),
            'inset_bbox': [bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax]
        }
        
        print("")
        print("="*80)
        print("✓ INSET PARAMETERS COMPLETE")
        print("="*80)
        print("Copy and paste this code:")
        print("")
        print(f"comp.compare_residence_times(")
        print(f"    ion_type='{ion_type}',")
        print(f"    residence_type='{residence_type}',")
        print(f"    shell='{shell}',")
        print(f"    add_inset=True,")
        print(f"    inset_xlim=({xmin}, {xmax}),")
        print(f"    inset_ylim=({ymin}, {ymax}),")
        print(f"    inset_bbox=[{bbox_xmin}, {bbox_xmax}, {bbox_ymin}, {bbox_ymax}]")
        print(f")")
        print("="*80)
        
        # Clean up temp file
        try:
            import os
            os.remove(temp_filename)
            print(f"\n(Cleaned up: {temp_filename})")
        except:
            pass
        
        return inset_params



    def create_multipanel_comparison(self, ion_types=['Na', 'Cl'], 
                                    properties=['rdf', 'coordination', 'dipole', 'residence'],
                                    save_plot=True, output_file='multipanel_comparison.png',
                                    dpi=300, figsize=None):
        '''
        Create a comprehensive multi-panel figure comparing multiple properties across concentrations.
        
        Parameters
        ----------
        ion_types : list of str
            Ion types to include in comparison (e.g., ['Na', 'Cl'])
        properties : list of str
            Properties to plot. Options:
            - 'rdf': Radial distribution functions
            - 'coordination': Coordination numbers vs concentration
            - 'dipole': Mean dipole angles vs concentration
            - 'residence': Mean residence times vs concentration
            - 'pairing': Ion pairing RDF (for cations only)
            - 'polyhedron': Polyhedron volumes vs concentration
        save_plot : bool
            Whether to save the figure
        output_file : str
            Output filename for saved figure
        dpi : int
            Resolution for saved figure
        figsize : tuple, optional
            Figure size (width, height) in inches
            If None, auto-calculated based on number of panels
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        
        # Calculate grid dimensions
        n_ions = len(ion_types)
        n_props = len(properties)
        n_panels = n_ions * n_props
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            width = 6 * n_props
            height = 5 * n_ions
            figsize = (width, height)
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_ions, n_props, hspace=0.3, wspace=0.3)
        
        # Get sorted concentrations and colors
        concentrations = sorted(self.loaded_data.keys(), key=lambda x: float(x.replace('M', '')))
        conc_values = [float(c.replace('M', '')) for c in concentrations]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(concentrations)))
        
        panel_idx = 0
        
        for i, ion_type in enumerate(ion_types):
            for j, prop in enumerate(properties):
                ax = fig.add_subplot(gs[i, j])
                
                if prop == 'rdf':
                    # Plot RDFs for this ion
                    for k, conc_key in enumerate(concentrations):
                        conc_data = self.loaded_data[conc_key]
                        
                        if 'rdfs' in conc_data:
                            rdf_key = f"{ion_type}-w"
                            if rdf_key in conc_data['rdfs']:
                                rdf_data = conc_data['rdfs'][rdf_key]
                                label = self.concentration_data[conc_key].get('label', conc_key)
                                ax.plot(rdf_data['bins'], rdf_data['rdf'], 
                                    color=colors[k], linewidth=2, label=label, alpha=0.8)
                    
                    ax.set_xlabel('Distance (Å)', fontsize=10)
                    ax.set_ylabel('g(r)', fontsize=10)
                    ax.set_title(f'{ion_type}-Water RDF', fontsize=11, fontweight='bold')
                    ax.set_xlim(0, 10)
                    if i == 0 and j == 0:
                        ax.legend(fontsize=8, frameon=False, loc='best')
                
                elif prop == 'coordination':
                    # Plot coordination number vs concentration
                    means, stds = [], []
                    
                    for conc_key in concentrations:
                        conc_data = self.loaded_data[conc_key]
                        
                        # Try to extract coordination number
                        cn_mean = None
                        if 'shell_coordination_numbers' in conc_data:
                            if ion_type in conc_data['shell_coordination_numbers']:
                                ion_data = conc_data['shell_coordination_numbers'][ion_type]
                                if isinstance(ion_data, dict) and 'overall' in ion_data:
                                    cn_mean = ion_data['overall'].get('mean')
                                    cn_std = ion_data['overall'].get('std')
                                elif isinstance(ion_data, dict) and 'shells' in ion_data:
                                    # Aggregate from shells
                                    all_cns = []
                                    for shell_data in ion_data['shells'].values():
                                        if 'coordination_numbers' in shell_data:
                                            all_cns.extend(shell_data['coordination_numbers'])
                                    if all_cns:
                                        cn_mean = np.mean(all_cns)
                                        cn_std = np.std(all_cns)
                        
                        means.append(cn_mean if cn_mean is not None else np.nan)
                        stds.append(cn_std if cn_mean is not None else np.nan)
                    
                    if any(~np.isnan(means)):
                        ax.errorbar(conc_values, means, yerr=stds,
                                marker='o', markersize=8, capsize=5, linewidth=2,
                                color='black', markerfacecolor='#00c5ff',
                                markeredgecolor='black', markeredgewidth=1.5)
                    
                    ax.set_xlabel('Concentration (M)', fontsize=10)
                    ax.set_ylabel('Coordination Number', fontsize=10)
                    ax.set_title(f'{ion_type} CN vs Conc.', fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                
                elif prop == 'dipole':
                    # Plot mean dipole angle vs concentration
                    means, stds = [], []
                    
                    for conc_key in concentrations:
                        conc_data = self.loaded_data[conc_key]
                        
                        dipole_mean = None
                        if 'water_dipole_distributions' in conc_data:
                            if ion_type in conc_data['water_dipole_distributions']:
                                dipole_data = conc_data['water_dipole_distributions'][ion_type]
                                dipole_mean = dipole_data.get('mean')
                                dipole_std = dipole_data.get('std')
                        
                        means.append(dipole_mean if dipole_mean is not None else np.nan)
                        stds.append(dipole_std if dipole_mean is not None else np.nan)
                    
                    if any(~np.isnan(means)):
                        ax.errorbar(conc_values, means, yerr=stds,
                                marker='s', markersize=8, capsize=5, linewidth=2,
                                color='black', markerfacecolor='#ff6b6b',
                                markeredgecolor='black', markeredgewidth=1.5)
                    
                    ax.set_xlabel('Concentration (M)', fontsize=10)
                    ax.set_ylabel('Mean Dipole Angle (°)', fontsize=10)
                    ax.set_title(f'{ion_type} Dipole Angle', fontsize=11, fontweight='bold')
                    ax.set_ylim(0, 180)
                    ax.grid(True, alpha=0.3)
                
                elif prop == 'residence':
                    # Plot mean residence time vs concentration
                    means, stds = [], []
                    
                    for conc_key in concentrations:
                        conc_data = self.loaded_data[conc_key]
                        
                        res_mean = None
                        if 'water_residence_times' in conc_data:
                            if ion_type in conc_data['water_residence_times']:
                                res_info = conc_data['water_residence_times'][ion_type]
                                if 'shells' in res_info and 'shell_1' in res_info['shells']:
                                    shell_data = res_info['shells']['shell_1']
                                    res_mean = shell_data.get('mean_residence_time', 
                                                            shell_data.get('mean'))
                                    res_std = shell_data.get('std_residence_time',
                                                            shell_data.get('std'))
                        
                        means.append(res_mean if res_mean is not None else np.nan)
                        stds.append(res_std if res_mean is not None else np.nan)
                    
                    if any(~np.isnan(means)):
                        ax.errorbar(conc_values, means, yerr=stds,
                                marker='^', markersize=8, capsize=5, linewidth=2,
                                color='black', markerfacecolor='#4ecdc4',
                                markeredgecolor='black', markeredgewidth=1.5)
                    
                    ax.set_xlabel('Concentration (M)', fontsize=10)
                    ax.set_ylabel('Residence Time (ps)', fontsize=10)
                    ax.set_title(f'{ion_type} Residence Time', fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                
                elif prop == 'pairing':
                    # Plot ion pairing RDF (cations only)
                    if ion_type in ['Na', 'Mg', 'Ca', 'K']:  # Add other cations as needed
                        for k, conc_key in enumerate(concentrations):
                            conc_data = self.loaded_data[conc_key]
                            
                            if 'rdfs' in conc_data:
                                # Try common pairing RDF keys
                                for rdf_key in [f'{ion_type}-Cl', 'ci-ai']:
                                    if rdf_key in conc_data['rdfs']:
                                        rdf_data = conc_data['rdfs'][rdf_key]
                                        label = self.concentration_data[conc_key].get('label', conc_key)
                                        ax.plot(rdf_data['bins'], rdf_data['rdf'],
                                            color=colors[k], linewidth=2, label=label, alpha=0.8)
                                        break
                        
                        ax.set_xlabel('Distance (Å)', fontsize=10)
                        ax.set_ylabel('g(r)', fontsize=10)
                        ax.set_title(f'{ion_type}-Anion RDF', fontsize=11, fontweight='bold')
                        ax.set_xlim(0, 12)
                    else:
                        ax.text(0.5, 0.5, 'N/A\n(anion)', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
                        ax.set_xticks([])
                        ax.set_yticks([])
                
                elif prop == 'polyhedron':
                    # Plot polyhedron volume vs concentration
                    means, stds = [], []
                    
                    for conc_key in concentrations:
                        conc_data = self.loaded_data[conc_key]
                        
                        vol_mean = None
                        if 'polyhedron_sizes' in conc_data:
                            if ion_type in conc_data['polyhedron_sizes']:
                                poly_data = conc_data['polyhedron_sizes'][ion_type]
                                vol_mean = poly_data.get('mean')
                                vol_std = poly_data.get('std')
                        
                        means.append(vol_mean if vol_mean is not None else np.nan)
                        stds.append(vol_std if vol_mean is not None else np.nan)
                    
                    if any(~np.isnan(means)):
                        ax.errorbar(conc_values, means, yerr=stds,
                                marker='D', markersize=8, capsize=5, linewidth=2,
                                color='black', markerfacecolor='#95e1d3',
                                markeredgecolor='black', markeredgewidth=1.5)
                    
                    ax.set_xlabel('Concentration (M)', fontsize=10)
                    ax.set_ylabel('Polyhedron Volume (Å³)', fontsize=10)
                    ax.set_title(f'{ion_type} Polyhedron Vol.', fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                
                ax.tick_params(labelsize=9)
                panel_idx += 1
        
        # Add overall title
        fig.suptitle('Concentration Comparison: Multi-Property Analysis', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        if save_plot:
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            print(f"Multi-panel comparison saved: {output_file}")
        
        plt.show()
        
        return fig


    def plot_concentration_trends(self, properties=['coordination_number', 'dipole_angle', 'residence_time'],
                                ion_types=None, save_plots=True, figsize=(14, 5)):
        '''
        Create concentration-dependent trend plots for multiple properties.
        Each property gets its own figure with all ions shown.
        
        Parameters
        ----------
        properties : list of str
            Properties to plot. Options:
            - 'coordination_number': Mean CN vs concentration
            - 'dipole_angle': Mean dipole angle vs concentration
            - 'residence_time': Mean residence time vs concentration
            - 'polyhedron_volume': Mean polyhedron volume vs concentration
            - 'first_shell_radius': First shell outer radius vs concentration
            - 'CIP_probability': Contact ion pair probability vs concentration (cations only)
        ion_types : list of str, optional
            Ion types to include. If None, uses all available ions
        save_plots : bool
            Whether to save plots
        figsize : tuple
            Figure size for each property plot
        
        Returns
        -------
        figures : dict
            Dictionary of matplotlib figures, keyed by property name
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        
        # Determine ion types
        if ion_types is None:
            ion_types = self.get_available_ion_types()
        
        # Get concentrations
        concentrations = sorted(self.loaded_data.keys(), key=lambda x: float(x.replace('M', '')))
        conc_values = [float(c.replace('M', '')) for c in concentrations]
        
        # Define colors for ions
        ion_colors = {
            'Na': '#00c5ff',
            'Mg': '#ff6b6b',
            'Ca': '#4ecdc4',
            'K': '#95e1d3',
            'Cl': '#ffd93d',
            'Br': '#ff9ff3',
            'I': '#a8e6cf'
        }
        
        figures = {}
        
        for prop in properties:
            fig, ax = plt.subplots(figsize=figsize)
            
            for ion_type in ion_types:
                values, errors = [], []
                
                for conc_key in concentrations:
                    conc_data = self.loaded_data[conc_key]
                    val, err = None, None
                    
                    if prop == 'coordination_number':
                        # Extract coordination number
                        if 'shell_coordination_numbers' in conc_data:
                            if ion_type in conc_data['shell_coordination_numbers']:
                                ion_data = conc_data['shell_coordination_numbers'][ion_type]
                                if isinstance(ion_data, dict) and 'overall' in ion_data:
                                    val = ion_data['overall'].get('mean')
                                    err = ion_data['overall'].get('std')
                                elif isinstance(ion_data, dict) and 'shells' in ion_data:
                                    all_cns = []
                                    for shell_data in ion_data['shells'].values():
                                        if 'coordination_numbers' in shell_data:
                                            all_cns.extend(shell_data['coordination_numbers'])
                                    if all_cns:
                                        val = np.mean(all_cns)
                                        err = np.std(all_cns)
                    
                    elif prop == 'dipole_angle':
                        if 'water_dipole_distributions' in conc_data:
                            if ion_type in conc_data['water_dipole_distributions']:
                                dipole_data = conc_data['water_dipole_distributions'][ion_type]
                                val = dipole_data.get('mean')
                                err = dipole_data.get('std')
                    
                    elif prop == 'residence_time':
                        if 'water_residence_times' in conc_data:
                            if ion_type in conc_data['water_residence_times']:
                                res_info = conc_data['water_residence_times'][ion_type]
                                if 'shells' in res_info and 'shell_1' in res_info['shells']:
                                    shell_data = res_info['shells']['shell_1']
                                    val = shell_data.get('mean_residence_time', shell_data.get('mean'))
                                    err = shell_data.get('std_residence_time', shell_data.get('std'))
                    
                    elif prop == 'polyhedron_volume':
                        if 'polyhedron_sizes' in conc_data:
                            if ion_type in conc_data['polyhedron_sizes']:
                                poly_data = conc_data['polyhedron_sizes'][ion_type]
                                val = poly_data.get('mean')
                                err = poly_data.get('std')
                    
                    elif prop == 'first_shell_radius':
                        if 'shell_boundaries' in conc_data:
                            if ion_type in conc_data['shell_boundaries']:
                                shell_bounds = conc_data['shell_boundaries'][ion_type]
                                if 'shell_1' in shell_bounds:
                                    val = shell_bounds['shell_1'].get('r_max')
                                    err = 0  # No error for boundary
                    
                    elif prop == 'CIP_probability':
                        # Only for cations
                        if ion_type in ['Na', 'Mg', 'Ca', 'K']:
                            if 'ion_pairing_lifetimes' in conc_data:
                                if ion_type in conc_data['ion_pairing_lifetimes']:
                                    pairing_data = conc_data['ion_pairing_lifetimes'][ion_type]
                                    if 'CIP' in pairing_data:
                                        val = pairing_data['CIP'].get('probability', 
                                                                    pairing_data['CIP'].get('fraction'))
                                        err = 0
                    
                    values.append(val if val is not None else np.nan)
                    errors.append(err if err is not None else np.nan)
                
                # Plot if we have data
                if any(~np.isnan(values)):
                    color = ion_colors.get(ion_type, 'black')
                    
                    # Only add error bars if errors are non-zero
                    if any(np.array(errors) > 0):
                        ax.errorbar(conc_values, values, yerr=errors,
                                marker='o', markersize=10, capsize=5, linewidth=2.5,
                                color=color, label=ion_type,
                                markeredgecolor='black', markeredgewidth=1.5, alpha=0.8)
                    else:
                        ax.plot(conc_values, values, marker='o', markersize=10, 
                            linewidth=2.5, color=color, label=ion_type,
                            markeredgecolor='black', markeredgewidth=1.5, alpha=0.8)
            
            # Customize plot
            ax.set_xlabel('Concentration (M)', fontsize=12, fontweight='bold')
            
            # Set y-label based on property
            ylabel_map = {
                'coordination_number': 'Coordination Number',
                'dipole_angle': 'Mean Dipole Angle (°)',
                'residence_time': 'Mean Residence Time (ps)',
                'polyhedron_volume': 'Mean Polyhedron Volume (Å³)',
                'first_shell_radius': 'First Shell Radius (Å)',
                'CIP_probability': 'CIP Probability'
            }
            ax.set_ylabel(ylabel_map.get(prop, prop), fontsize=12, fontweight='bold')
            
            # Title
            title_map = {
                'coordination_number': 'Coordination Number vs Concentration',
                'dipole_angle': 'Water Dipole Angle vs Concentration',
                'residence_time': 'Water Residence Time vs Concentration',
                'polyhedron_volume': 'Coordination Polyhedron Volume vs Concentration',
                'first_shell_radius': 'First Solvation Shell Radius vs Concentration',
                'CIP_probability': 'Contact Ion Pair Probability vs Concentration'
            }
            ax.set_title(title_map.get(prop, prop), fontsize=14, fontweight='bold')
            
            ax.legend(fontsize=10, frameon=False, loc='best')
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            plt.tight_layout()
            
            if save_plots:
                filename = f'concentration_trend_{prop}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Trend plot saved: {filename}")
            
            figures[prop] = fig
            plt.show()
        
        return figures


    def debug_data_structure(self, ion_type=None, concentration_key=None):
        '''
        Debug method to show the structure of loaded data.
        Helps identify where different data types are stored.
        
        Parameters
        ----------
        ion_type : str, optional
            Specific ion type to search for (e.g., 'Na', 'Mg')
        concentration_key : str, optional
            Specific concentration to debug (e.g., '0.5M')
            If None, shows all concentrations
        '''
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return
        
        print("="*80)
        print("DATA STRUCTURE DEBUG")
        print("="*80)
        
        # Determine which concentrations to debug
        if concentration_key:
            if concentration_key not in self.loaded_data:
                print(f"ERROR: Concentration '{concentration_key}' not found")
                print(f"Available: {', '.join(sorted(self.loaded_data.keys()))}")
                return
            conc_keys = [concentration_key]
        else:
            conc_keys = sorted(self.loaded_data.keys())
        
        for conc_key in conc_keys:
            print(f"\n{conc_key}:")
            print("-"*40)
            
            conc_data = self.loaded_data[conc_key]
            print(f"Top-level keys: {list(conc_data.keys())}")
            
            # Check for ion-specific data
            if ion_type:
                print(f"\nSearching for '{ion_type}' data:")
                found_any = False
                
                for key, value in conc_data.items():
                    if isinstance(value, dict):
                        if ion_type in value:
                            found_any = True
                            print(f"  ✓ Found in '{key}':")
                            ion_data = value[ion_type]
                            
                            if isinstance(ion_data, dict):
                                print(f"    Type: dict")
                                print(f"    Keys: {list(ion_data.keys())}")
                                
                                # Show first level of nested structure
                                for sub_key, sub_value in list(ion_data.items())[:3]:
                                    print(f"      '{sub_key}': {type(sub_value).__name__}", end='')
                                    
                                    if isinstance(sub_value, dict):
                                        print(f" with keys: {list(sub_value.keys())[:5]}")
                                    elif isinstance(sub_value, np.ndarray):
                                        print(f" shape: {sub_value.shape}")
                                    elif isinstance(sub_value, (list, tuple)):
                                        print(f" length: {len(sub_value)}")
                                    else:
                                        print(f" = {sub_value}")
                                
                                if len(ion_data) > 3:
                                    print(f"      ... and {len(ion_data) - 3} more keys")
                            else:
                                print(f"    Type: {type(ion_data).__name__}")
                                if isinstance(ion_data, np.ndarray):
                                    print(f"    Shape: {ion_data.shape}")
                
                if not found_any:
                    print(f"  ✗ '{ion_type}' not found in any top-level dict keys")
            
            # Look for polyhedron-related keys
            print(f"\nPolyhedron-related keys:")
            polyhedron_keys = [k for k in conc_data.keys() if 'polyhedron' in k.lower() or 'polyhedra' in k.lower()]
            if polyhedron_keys:
                for key in polyhedron_keys:
                    print(f"  ✓ '{key}': {type(conc_data[key]).__name__}")
                    if isinstance(conc_data[key], dict):
                        print(f"    Keys: {list(conc_data[key].keys())}")
            else:
                print(f"  ✗ No polyhedron-related keys found")
            
            # Look for dipole-related keys
            print(f"\nDipole-related keys:")
            dipole_keys = [k for k in conc_data.keys() if 'dipole' in k.lower()]
            if dipole_keys:
                for key in dipole_keys:
                    print(f"  ✓ '{key}': {type(conc_data[key]).__name__}")
                    if isinstance(conc_data[key], dict):
                        print(f"    Keys: {list(conc_data[key].keys())}")
            else:
                print(f"  ✗ No dipole-related keys found")
            
            # Look for coordination-related keys
            print(f"\nCoordination-related keys:")
            coord_keys = [k for k in conc_data.keys() if 'coordination' in k.lower() or 'shell' in k.lower()]
            if coord_keys:
                for key in coord_keys:
                    print(f"  ✓ '{key}': {type(conc_data[key]).__name__}")
                    if isinstance(conc_data[key], dict):
                        print(f"    Keys: {list(conc_data[key].keys())[:5]}")
            else:
                print(f"  ✗ No coordination-related keys found")
            
            # Look for ion pairing related keys
            print(f"\nIon pairing-related keys:")
            pairing_keys = [k for k in conc_data.keys() if 'pair' in k.lower() or 'ion' in k.lower()]
            if pairing_keys:
                for key in pairing_keys:
                    print(f"  ✓ '{key}': {type(conc_data[key]).__name__}")
                    if isinstance(conc_data[key], dict):
                        print(f"    Keys: {list(conc_data[key].keys())[:5]}")
            else:
                print(f"  ✗ No ion pairing-related keys found")
        
        print("\n" + "="*80)
        print("DEBUG COMPLETE")
        print("="*80)


    def perform_statistical_comparison(self, property_type, ion_type=None, 
                                    test_type='anova', save_log=True,
                                    concentrations_to_compare=None):
        '''
        Perform statistical tests to compare properties across concentrations.
        
        Parameters
        ----------
        property_type : str
            Type of property to compare:
            - 'coordination_number': Overall coordination numbers
            - 'shell_coordination': Shell-specific coordination numbers
            - 'dipole_angle': Water dipole angles
            - 'residence_time': Water residence times
            - 'polyhedron_volume': Coordination polyhedron volumes
        ion_type : str, optional
            Ion type to analyze (required for most property types)
        test_type : str
            Statistical test to perform:
            - 'anova': One-way ANOVA (parametric, >2 groups)
            - 't-test': Independent t-test (parametric, 2 groups)
            - 'kruskal': Kruskal-Wallis test (non-parametric, >2 groups)
            - 'mannwhitney': Mann-Whitney U test (non-parametric, 2 groups)
        save_log : bool
            Whether to save results to log file
        concentrations_to_compare : list, optional
            List of specific concentration keys to compare (e.g., ['0.0M', '0.42M'])
            If None, compares all loaded concentrations
            Required for t-test and mannwhitney (must be exactly 2)
        
        Returns
        -------
        results : dict
            Dictionary with test results including statistic, p-value, and effect size
        '''
        
        from scipy import stats as scipy_stats
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"STATISTICAL COMPARISON: {property_type.upper()}")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Property: {property_type}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Test: {test_type}")
        if concentrations_to_compare:
            log_lines.append(f"Concentrations: {concentrations_to_compare}")
        log_lines.append("")
        
        # Determine which concentrations to use
        if concentrations_to_compare is not None:
            concs_to_use = concentrations_to_compare
            # Validate that they exist
            for conc in concs_to_use:
                if conc not in self.loaded_data:
                    print(f"ERROR: Concentration '{conc}' not found in loaded data")
                    print(f"Available: {sorted(self.loaded_data.keys())}")
                    return None
        else:
            concs_to_use = sorted(self.loaded_data.keys())
        
        # Check requirements for specific tests
        if test_type in ['t-test', 'mannwhitney']:
            if len(concs_to_use) != 2:
                error_msg = f"ERROR: {test_type} requires exactly 2 concentrations. "
                error_msg += f"Found {len(concs_to_use)}: {concs_to_use}\n"
                error_msg += "Please specify concentrations_to_compare=['conc1', 'conc2']"
                print(error_msg)
                log_lines.append(error_msg)
                
                if save_log:
                    log_filename = f'statistical_comparison_{property_type}_{ion_type}_{test_type}_ERROR.log'
                    with open(log_filename, 'w') as f:
                        f.write('\n'.join(log_lines))
                    print(f"Error log saved: {log_filename}")
                
                return None
        
        # Collect data based on property type
        data_by_concentration = {}
        
        log_lines.append("COLLECTING DATA")
        log_lines.append("-"*80)

        if property_type == 'coordination_number':
            for conc_key in concs_to_use:
                conc_data = self.loaded_data[conc_key]
                
                found_data = False
                
                # Method 1: Extract from shell_coordination_probabilities (DataFrame) - MOST RELIABLE
                if 'shell_coordination_probabilities' in conc_data:
                    if ion_type in conc_data['shell_coordination_probabilities']:
                        prob_info = conc_data['shell_coordination_probabilities'][ion_type]
                        
                        if 'data' in prob_info:
                            import pandas as pd
                            prob_df = prob_info['data']
                            
                            if isinstance(prob_df, pd.DataFrame):
                                # Extract CN values and counts/fractions
                                if 'water' in prob_df.columns and 'count' in prob_df.columns:
                                    cn_values = prob_df['water'].values
                                    counts = prob_df['count'].values
                                    
                                    # Create samples: repeat each CN by its count * scale factor
                                    samples = []
                                    for cn, count in zip(cn_values, counts):
                                        # Scale counts to get reasonable sample size
                                        n_samples = int(count * 10000)  # Scale factor
                                        samples.extend([cn] * n_samples)
                                    
                                    if samples:
                                        data_by_concentration[conc_key] = np.array(samples)
                                        log_lines.append(f"✓ {conc_key}: Created {len(samples)} samples from probability distribution")
                                        log_lines.append(f"    CN range: {min(samples)} - {max(samples)}, Mean: {np.mean(samples):.2f}")
                                        found_data = True
                
                # Method 2: Try shell_coordination_numbers with 'coordination_numbers' key (FIXED)
                if not found_data and 'shell_coordination_numbers' in conc_data:
                    if ion_type in conc_data['shell_coordination_numbers']:
                        ion_data = conc_data['shell_coordination_numbers'][ion_type]
                        
                        if isinstance(ion_data, dict) and 'shells' in ion_data:
                            shells_data = ion_data['shells']
                            
                            # Aggregate coordination_numbers from all shells
                            all_values = []
                            for shell_name, shell_info in shells_data.items():
                                if isinstance(shell_info, dict):
                                    # FIXED: Key is 'coordination_numbers', not 'values'
                                    if 'coordination_numbers' in shell_info:
                                        cn_array = shell_info['coordination_numbers']
                                        if isinstance(cn_array, np.ndarray):
                                            cn_array = cn_array.flatten()
                                            cn_array = cn_array[~np.isnan(cn_array)]
                                            all_values.extend(cn_array)
                                            log_lines.append(f"  Added {len(cn_array)} from '{shell_name}['coordination_numbers']'")
                            
                            if all_values:
                                data_by_concentration[conc_key] = np.array(all_values)
                                log_lines.append(f"✓ {conc_key}: {len(all_values)} samples from shell coordination_numbers")
                                found_data = True
                
                # Method 3: Try coordination_numbers_overall
                if not found_data and 'coordination_numbers_overall' in conc_data:
                    if ion_type in conc_data['coordination_numbers_overall']:
                        cn_data = conc_data['coordination_numbers_overall'][ion_type]
                        if isinstance(cn_data, dict):
                            # Try both 'values' and 'coordination_numbers'
                            for key in ['values', 'coordination_numbers']:
                                if key in cn_data:
                                    values = cn_data[key]
                                    if isinstance(values, (list, np.ndarray)):
                                        values = np.array(values).flatten()
                                        values = values[~np.isnan(values)]
                                        data_by_concentration[conc_key] = values
                                        log_lines.append(f"✓ {conc_key}: {len(values)} samples from coordination_numbers_overall['{key}']")
                                        found_data = True
                                        break
                
                if not found_data:
                    log_lines.append(f"⚠ {conc_key}: No coordination number data found")
        
        elif property_type == 'dipole_angle':
            for conc_key in concs_to_use:
                conc_data = self.loaded_data[conc_key]
                
                if 'water_dipole_distributions' in conc_data:
                    if ion_type in conc_data['water_dipole_distributions']:
                        dipole_data = conc_data['water_dipole_distributions'][ion_type]
                        if 'angles' in dipole_data:
                            angles = dipole_data['angles']
                            if isinstance(angles, np.ndarray):
                                angles = angles.flatten()
                                angles = angles[~np.isnan(angles)]
                                data_by_concentration[conc_key] = angles
                                log_lines.append(f"✓ {conc_key}: {len(angles)} samples")
        
        elif property_type == 'residence_time':
            for conc_key in concs_to_use:
                conc_data = self.loaded_data[conc_key]
                
                if 'water_residence_times' in conc_data:
                    if ion_type in conc_data['water_residence_times']:
                        res_info = conc_data['water_residence_times'][ion_type]
                        if 'shells' in res_info and 'shell_1' in res_info['shells']:
                            shell_data = res_info['shells']['shell_1']
                            if 'residence_times' in shell_data:
                                times = np.array(shell_data['residence_times'])
                                times = times[~np.isnan(times)]
                                data_by_concentration[conc_key] = times
                                log_lines.append(f"✓ {conc_key}: {len(times)} samples")
        
        elif property_type == 'polyhedron_volume':
            for conc_key in concs_to_use:
                conc_data = self.loaded_data[conc_key]
                
                if 'polyhedron_sizes' in conc_data:
                    if ion_type in conc_data['polyhedron_sizes']:
                        poly_data = conc_data['polyhedron_sizes'][ion_type]
                        if 'volumes' in poly_data:
                            volumes = np.array(poly_data['volumes'])
                            volumes = volumes.flatten() if volumes.ndim > 1 else volumes
                            volumes = volumes[~np.isnan(volumes)]
                            data_by_concentration[conc_key] = volumes
                            log_lines.append(f"✓ {conc_key}: {len(volumes)} samples")
        
        if not data_by_concentration:
            log_lines.append("")
            log_lines.append("ERROR: No data collected for statistical comparison")
            log_lines.append("")
            log_lines.append("DEBUGGING SUGGESTIONS:")
            log_lines.append(f"1. Run: comp.debug_data_structure(ion_type='{ion_type}')")
            log_lines.append("2. Check if ion_type is correct")
            log_lines.append("3. Verify that the property exists in your data")
            
            if save_log:
                log_filename = f'statistical_comparison_{property_type}_{ion_type}_{test_type}_ERROR.log'
                with open(log_filename, 'w') as f:
                    f.write('\n'.join(log_lines))
                print(f"Error log saved: {log_filename}")
            
            return None
        
        log_lines.append("")
        log_lines.append(f"Total concentrations with data: {len(data_by_concentration)}")
        log_lines.append("")
        
        # Prepare data arrays
        concentrations = sorted(data_by_concentration.keys(), key=lambda x: float(x.replace('M', '')))
        data_arrays = [data_by_concentration[c] for c in concentrations]
        
        # Descriptive statistics
        log_lines.append("DESCRIPTIVE STATISTICS")
        log_lines.append("-"*80)
        log_lines.append(f"{'Concentration':<15} {'N':<10} {'Mean':<12} {'Std':<12} {'Median':<12}")
        log_lines.append("-"*80)
        
        for conc_key in concentrations:
            data = data_by_concentration[conc_key]
            log_lines.append(f"{conc_key:<15} {len(data):<10} {np.mean(data):<12.3f} "
                            f"{np.std(data):<12.3f} {np.median(data):<12.3f}")
        
        log_lines.append("")
        
        # Perform statistical test
        results = {
            'property_type': property_type,
            'ion_type': ion_type,
            'test_type': test_type,
            'concentrations': concentrations,
            'n_groups': len(concentrations)
        }
        
        log_lines.append("STATISTICAL TEST RESULTS")
        log_lines.append("-"*80)
        
        if test_type == 'anova':
            if len(data_arrays) < 2:
                log_lines.append("ERROR: ANOVA requires at least 2 groups")
                results['error'] = "Insufficient groups"
            else:
                statistic, p_value = scipy_stats.f_oneway(*data_arrays)
                results['statistic'] = statistic
                results['p_value'] = p_value
                results['test_name'] = 'One-way ANOVA'
                
                log_lines.append(f"Test: One-way ANOVA")
                log_lines.append(f"F-statistic: {statistic:.4f}")
                log_lines.append(f"p-value: {p_value:.4e}")
                
                # Effect size (eta-squared)
                all_data = np.concatenate(data_arrays)
                grand_mean = np.mean(all_data)
                
                ss_between = sum([len(arr) * (np.mean(arr) - grand_mean)**2 for arr in data_arrays])
                ss_total = sum([(x - grand_mean)**2 for x in all_data])
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                results['effect_size'] = eta_squared
                results['effect_size_type'] = 'eta_squared'
                log_lines.append(f"Effect size (η²): {eta_squared:.4f}")
        
        elif test_type == 't-test':
            if len(data_arrays) != 2:
                log_lines.append("ERROR: t-test requires exactly 2 groups")
                results['error'] = "t-test requires 2 groups"
            else:
                statistic, p_value = scipy_stats.ttest_ind(data_arrays[0], data_arrays[1])
                results['statistic'] = statistic
                results['p_value'] = p_value
                results['test_name'] = 'Independent t-test'
                
                log_lines.append(f"Test: Independent t-test")
                log_lines.append(f"Comparing: {concentrations[0]} vs {concentrations[1]}")
                log_lines.append(f"t-statistic: {statistic:.4f}")
                log_lines.append(f"p-value: {p_value:.4e}")
                
                # Effect size (Cohen's d)
                mean_diff = np.mean(data_arrays[0]) - np.mean(data_arrays[1])
                pooled_std = np.sqrt((np.var(data_arrays[0]) + np.var(data_arrays[1])) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                results['effect_size'] = cohens_d
                results['effect_size_type'] = 'cohens_d'
                log_lines.append(f"Effect size (Cohen's d): {cohens_d:.4f}")
        
        elif test_type == 'kruskal':
            if len(data_arrays) < 2:
                log_lines.append("ERROR: Kruskal-Wallis test requires at least 2 groups")
                results['error'] = "Insufficient groups"
            else:
                statistic, p_value = scipy_stats.kruskal(*data_arrays)
                results['statistic'] = statistic
                results['p_value'] = p_value
                results['test_name'] = 'Kruskal-Wallis H-test'
                
                log_lines.append(f"Test: Kruskal-Wallis H-test (non-parametric)")
                log_lines.append(f"H-statistic: {statistic:.4f}")
                log_lines.append(f"p-value: {p_value:.4e}")
                
                # Effect size (epsilon-squared)
                n_total = sum([len(arr) for arr in data_arrays])
                epsilon_squared = (statistic - len(data_arrays) + 1) / (n_total - len(data_arrays))
                
                results['effect_size'] = epsilon_squared
                results['effect_size_type'] = 'epsilon_squared'
                log_lines.append(f"Effect size (ε²): {epsilon_squared:.4f}")
        
        elif test_type == 'mannwhitney':
            if len(data_arrays) != 2:
                log_lines.append("ERROR: Mann-Whitney U test requires exactly 2 groups")
                results['error'] = "Mann-Whitney requires 2 groups"
            else:
                statistic, p_value = scipy_stats.mannwhitneyu(data_arrays[0], data_arrays[1])
                results['statistic'] = statistic
                results['p_value'] = p_value
                results['test_name'] = 'Mann-Whitney U test'
                
                log_lines.append(f"Test: Mann-Whitney U test (non-parametric)")
                log_lines.append(f"Comparing: {concentrations[0]} vs {concentrations[1]}")
                log_lines.append(f"U-statistic: {statistic:.4f}")
                log_lines.append(f"p-value: {p_value:.4e}")
                
                # Effect size (rank-biserial correlation)
                n1, n2 = len(data_arrays[0]), len(data_arrays[1])
                r = 1 - (2*statistic) / (n1 * n2)
                
                results['effect_size'] = r
                results['effect_size_type'] = 'rank_biserial'
                log_lines.append(f"Effect size (rank-biserial r): {r:.4f}")
        
        # Interpretation
        log_lines.append("")
        log_lines.append("INTERPRETATION")
        log_lines.append("-"*80)
        
        if 'p_value' in results:
            alpha = 0.05
            if results['p_value'] < alpha:
                log_lines.append(f"Result: SIGNIFICANT (p < {alpha})")
                if len(concentrations) == 2:
                    log_lines.append(f"There IS a statistically significant difference between {concentrations[0]} and {concentrations[1]}")
                else:
                    log_lines.append(f"There IS a statistically significant difference among concentrations")
            else:
                log_lines.append(f"Result: NOT SIGNIFICANT (p >= {alpha})")
                if len(concentrations) == 2:
                    log_lines.append(f"There is NO statistically significant difference between {concentrations[0]} and {concentrations[1]}")
                else:
                    log_lines.append(f"There is NO statistically significant difference among concentrations")
            
            # Effect size interpretation
            if 'effect_size' in results:
                es = abs(results['effect_size'])
                es_type = results['effect_size_type']
                
                if es_type == 'eta_squared':
                    if es < 0.01:
                        effect_label = "negligible"
                    elif es < 0.06:
                        effect_label = "small"
                    elif es < 0.14:
                        effect_label = "medium"
                    else:
                        effect_label = "large"
                elif es_type == 'cohens_d':
                    if es < 0.2:
                        effect_label = "negligible"
                    elif es < 0.5:
                        effect_label = "small"
                    elif es < 0.8:
                        effect_label = "medium"
                    else:
                        effect_label = "large"
                else:
                    effect_label = "see literature"
                
                log_lines.append(f"Effect size: {effect_label}")
        
        log_lines.append("")
        
        # Save log
        if save_log:
            log_filename = f'statistical_comparison_{property_type}_{ion_type}_{test_type}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Statistical analysis log saved: {log_filename}")
        
        # Store raw data in results
        results['data'] = data_by_concentration
        results['log'] = log_lines
        
        return results 


    def print_statistical_summary(self, results):
        '''
        Print a formatted summary of statistical test results.
        
        Parameters
        ----------
        results : dict
            Results dictionary from perform_statistical_comparison()
        '''
        
        if results is None:
            print("No results to display")
            return
        
        print("\n" + "="*80)
        print("STATISTICAL COMPARISON SUMMARY")
        print("="*80)
        print(f"Property: {results.get('property_type', 'N/A')}")
        print(f"Ion Type: {results.get('ion_type', 'N/A')}")
        print(f"Test: {results.get('test_name', results.get('test_type', 'N/A'))}")
        print(f"Number of groups: {results.get('n_groups', 'N/A')}")
        
        if 'concentrations' in results:
            print(f"Concentrations: {', '.join(results['concentrations'])}")
        
        print("")
        
        if 'error' in results:
            print(f"ERROR: {results['error']}")
            return
        
        print("Results:")
        print("-"*80)
        
        if 'statistic' in results:
            stat_name = 'Statistic'
            test_name = results.get('test_name', '')
            if 'F-statistic' in test_name or 'ANOVA' in test_name:
                stat_name = 'F-statistic'
            elif 't-statistic' in test_name or 't-test' in test_name:
                stat_name = 't-statistic'
            elif 'H-statistic' in test_name or 'Kruskal' in test_name:
                stat_name = 'H-statistic'
            elif 'U-statistic' in test_name or 'Mann-Whitney' in test_name:
                stat_name = 'U-statistic'
            
            print(f"{stat_name}: {results['statistic']:.4f}")
        
        if 'p_value' in results:
            print(f"p-value: {results['p_value']:.4e}")
            
            alpha = 0.05
            if results['p_value'] < alpha:
                print(f"✓ SIGNIFICANT (p < {alpha})")
                print("   → There IS a statistically significant difference")
            else:
                print(f"✗ NOT SIGNIFICANT (p >= {alpha})")
                print("   → There is NO statistically significant difference")
        
        if 'effect_size' in results:
            es_type = results.get('effect_size_type', 'effect size')
            es_value = results['effect_size']
            
            # Format effect size name
            es_name_map = {
                'eta_squared': 'η² (eta-squared)',
                'cohens_d': "Cohen's d",
                'epsilon_squared': 'ε² (epsilon-squared)',
                'rank_biserial': 'rank-biserial r'
            }
            es_name = es_name_map.get(es_type, es_type)
            
            print(f"\nEffect size ({es_name}): {es_value:.4f}")
            
            # Interpret effect size
            es = abs(es_value)
            if es_type == 'eta_squared' or es_type == 'epsilon_squared':
                if es < 0.01:
                    effect_label = "negligible"
                elif es < 0.06:
                    effect_label = "small"
                elif es < 0.14:
                    effect_label = "medium"
                else:
                    effect_label = "large"
            elif es_type == 'cohens_d':
                if es < 0.2:
                    effect_label = "negligible"
                elif es < 0.5:
                    effect_label = "small"
                elif es < 0.8:
                    effect_label = "medium"
                else:
                    effect_label = "large"
            elif es_type == 'rank_biserial':
                if es < 0.1:
                    effect_label = "negligible"
                elif es < 0.3:
                    effect_label = "small"
                elif es < 0.5:
                    effect_label = "medium"
                else:
                    effect_label = "large"
            else:
                effect_label = "(see literature for interpretation)"
            
            print(f"   → Effect size magnitude: {effect_label}")
        
        print("="*80 + "\n")

    def perform_pairwise_comparisons(self, property_type, ion_type, test_type='t-test',
                                    correction='bonferroni', save_log=True):
        '''
        Perform pairwise comparisons between all concentration pairs.
        
        Parameters
        ----------
        property_type : str
            Type of property to compare (same as perform_statistical_comparison)
        ion_type : str
            Ion type to analyze
        test_type : str
            Test for pairwise comparisons: 't-test' or 'mannwhitney'
        correction : str
            Multiple comparison correction method:
            - 'bonferroni': Bonferroni correction (conservative)
            - 'holm': Holm-Bonferroni (less conservative)
            - 'none': No correction
        save_log : bool
            Whether to save results to log file
        
        Returns
        -------
        pairwise_results : dict
            Dictionary with pairwise comparison results
        '''
        
        from scipy import stats as scipy_stats
        from itertools import combinations
        
        if not self.loaded_data:
            print("ERROR: No data loaded.")
            return None
        
        # Reuse data collection logic - FIXED: properly collect data
        data_by_concentration = {}

        if property_type == 'coordination_number':
            for conc_key in sorted(self.loaded_data.keys()):  # FIXED: use loaded_data.keys()
                conc_data = self.loaded_data[conc_key]
                
                found_data = False
                
                # Method 1: Extract from shell_coordination_probabilities (DataFrame) - MOST RELIABLE
                if 'shell_coordination_probabilities' in conc_data:
                    if ion_type in conc_data['shell_coordination_probabilities']:
                        prob_info = conc_data['shell_coordination_probabilities'][ion_type]
                        
                        if 'data' in prob_info:
                            import pandas as pd
                            prob_df = prob_info['data']
                            
                            if isinstance(prob_df, pd.DataFrame):
                                if 'water' in prob_df.columns and 'count' in prob_df.columns:
                                    cn_values = prob_df['water'].values
                                    counts = prob_df['count'].values
                                    
                                    samples = []
                                    for cn, count in zip(cn_values, counts):
                                        n_samples = int(count * 10000)
                                        samples.extend([cn] * n_samples)
                                    
                                    if samples:
                                        data_by_concentration[conc_key] = np.array(samples)
                                        found_data = True
                
                # Method 2: Try shell_coordination_numbers with 'coordination_numbers' key
                if not found_data and 'shell_coordination_numbers' in conc_data:
                    if ion_type in conc_data['shell_coordination_numbers']:
                        ion_data = conc_data['shell_coordination_numbers'][ion_type]
                        
                        if isinstance(ion_data, dict) and 'shells' in ion_data:
                            shells_data = ion_data['shells']
                            
                            all_values = []
                            for shell_name, shell_info in shells_data.items():
                                if isinstance(shell_info, dict):
                                    if 'coordination_numbers' in shell_info:
                                        cn_array = shell_info['coordination_numbers']
                                        if isinstance(cn_array, np.ndarray):
                                            cn_array = cn_array.flatten()
                                            cn_array = cn_array[~np.isnan(cn_array)]
                                            all_values.extend(cn_array)
                            
                            if all_values:
                                data_by_concentration[conc_key] = np.array(all_values)
                                found_data = True
        
        elif property_type == 'dipole_angle':
            for conc_key in sorted(self.loaded_data.keys()):
                conc_data = self.loaded_data[conc_key]
                if 'water_dipole_distributions' in conc_data:
                    if ion_type in conc_data['water_dipole_distributions']:
                        dipole_data = conc_data['water_dipole_distributions'][ion_type]
                        if 'angles' in dipole_data:
                            angles = dipole_data['angles']
                            if isinstance(angles, np.ndarray):
                                angles = angles.flatten()[~np.isnan(angles.flatten())]
                                data_by_concentration[conc_key] = angles
        
        elif property_type == 'polyhedron_volume':
            for conc_key in sorted(self.loaded_data.keys()):
                conc_data = self.loaded_data[conc_key]
                if 'polyhedron_sizes' in conc_data:
                    if ion_type in conc_data['polyhedron_sizes']:
                        poly_data = conc_data['polyhedron_sizes'][ion_type]
                        if 'volumes' in poly_data:
                            volumes = np.array(poly_data['volumes']).flatten()
                            volumes = volumes[~np.isnan(volumes)]
                            data_by_concentration[conc_key] = volumes
        
        if len(data_by_concentration) < 2:
            print("ERROR: Need at least 2 concentrations for pairwise comparisons")
            return None
        
        # Generate all pairs
        concentrations = sorted(data_by_concentration.keys(), key=lambda x: float(x.replace('M', '')))
        pairs = list(combinations(concentrations, 2))
        
        # Perform pairwise tests
        pairwise_results = {
            'property_type': property_type,
            'ion_type': ion_type,
            'test_type': test_type,
            'correction': correction,
            'n_comparisons': len(pairs),
            'comparisons': []
        }
        
        print(f"\nPerforming {len(pairs)} pairwise comparisons...")
        
        for conc1, conc2 in pairs:
            data1 = data_by_concentration[conc1]
            data2 = data_by_concentration[conc2]
            
            if test_type == 't-test':
                statistic, p_value = scipy_stats.ttest_ind(data1, data2)
            elif test_type == 'mannwhitney':
                statistic, p_value = scipy_stats.mannwhitneyu(data1, data2)
            else:
                print(f"ERROR: Unknown test type '{test_type}'")
                return None
            
            pairwise_results['comparisons'].append({
                'pair': (conc1, conc2),
                'statistic': statistic,
                'p_value': p_value,
                'mean1': np.mean(data1),
                'mean2': np.mean(data2),
                'n1': len(data1),
                'n2': len(data2)
            })
        
        # Apply multiple comparison correction
        p_values = [comp['p_value'] for comp in pairwise_results['comparisons']]
        
        if correction == 'bonferroni':
            alpha_corrected = 0.05 / len(pairs)
            pairwise_results['alpha_corrected'] = alpha_corrected
            for comp in pairwise_results['comparisons']:
                comp['significant'] = comp['p_value'] < alpha_corrected
        
        elif correction == 'holm':
            # Holm-Bonferroni step-down procedure
            sorted_indices = np.argsort(p_values)
            for rank, idx in enumerate(sorted_indices):
                alpha_i = 0.05 / (len(pairs) - rank)
                pairwise_results['comparisons'][idx]['significant'] = p_values[idx] < alpha_i
            pairwise_results['alpha_corrected'] = 'Holm-Bonferroni (step-down)'
        
        else:  # no correction
            alpha_corrected = 0.05
            pairwise_results['alpha_corrected'] = alpha_corrected
            for comp in pairwise_results['comparisons']:
                comp['significant'] = comp['p_value'] < alpha_corrected
        
        # Print summary
        print("\nPairwise Comparison Results:")
        print("="*80)
        print(f"{'Pair':<30} {'Mean1':<12} {'Mean2':<12} {'p-value':<12} {'Sig?':<8}")
        print("-"*80)
        
        for comp in pairwise_results['comparisons']:
            pair_str = f"{comp['pair'][0]} vs {comp['pair'][1]}"
            sig_str = "✓" if comp['significant'] else "✗"
            print(f"{pair_str:<30} {comp['mean1']:<12.3f} {comp['mean2']:<12.3f} "
                f"{comp['p_value']:<12.4e} {sig_str:<8}")
        
        print("="*80)
        print(f"Correction method: {correction}")
        print(f"Corrected α: {pairwise_results.get('alpha_corrected', 'N/A')}")
        
        # Save log if requested
        if save_log:
            log_filename = f'pairwise_comparisons_{property_type}_{ion_type}_{test_type}_{correction}.log'
            with open(log_filename, 'w') as f:
                f.write(f"Pairwise Comparisons: {property_type} - {ion_type}\n")
                f.write(f"Test: {test_type}\n")
                f.write(f"Correction: {correction}\n")
                f.write("="*80 + "\n\n")
                
                for comp in pairwise_results['comparisons']:
                    f.write(f"{comp['pair'][0]} vs {comp['pair'][1]}:\n")
                    f.write(f"  Mean 1: {comp['mean1']:.3f}\n")
                    f.write(f"  Mean 2: {comp['mean2']:.3f}\n")
                    f.write(f"  p-value: {comp['p_value']:.4e}\n")
                    f.write(f"  Significant: {comp['significant']}\n\n")
            
            print(f"\nPairwise comparison log saved: {log_filename}")
        
        return pairwise_results




    
    def compare_shell_boundaries(self, ion_type, save_plots=True):
        '''Compare solvation shell boundaries across concentrations'''
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        print("compare_shell_boundaries() - To be implemented")
        
    def compare_ion_pairing_cutoffs(self, plot=True, save_plots=True):
        '''Compare ion pairing cutoffs across concentrations'''
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        print("compare_ion_pairing_cutoffs() - To be implemented")
        
    def compare_coordination_probabilities(self, ion_type=None, plot=True, save_plots=True):
        '''Compare coordination number probabilities across concentrations'''
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_concentrations() first.")
            return None
        print("compare_coordination_probabilities() - To be implemented")
        
    def save_comparison_data(self, filename='concentration_comparison.pkl'):
        '''Save all comparison data to file'''
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.concentration_data, f)
            print(f"Comparison data saved: {filename}")
            return True
        except Exception as e:
            print(f"Error saving: {e}")
            return False
            
    def load_comparison_data(self, filename='concentration_comparison.pkl'):
        '''Load comparison data from file'''
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return False
        
        try:
            with open(filename, 'rb') as f:
                self.concentration_data = pickle.load(f)
            print(f"Comparison data loaded: {filename}")
            return True
        except Exception as e:
            print(f"Error loading: {e}")
            return False


if __name__ == "__main__":
    print("ConcentrationComparison - Module for comparing analysis across concentrations")
    print("Import and use:")
    print("from ConcentrationComparison import ConcentrationComparison")
    print("comp = ConcentrationComparison(concentration_data)")
    print("comp.load_all_concentrations()")