import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pickle

class SaltComparison:
    '''
    Compare solvation properties across different salt types at various concentrations.
    
    This class loads and compares data from multiple salt systems (e.g., NaCl, KCl, MgCl2, CaCl2)
    across different concentrations.
    '''
    
    def __init__(self, salt_data):
        '''
        Initialize SaltComparison with salt system information.
        
        Parameters
        ----------
        salt_data : dict
            Dictionary with salt information. Format:
            {
                'NaCl': {
                    'label': 'NaCl',
                    'color': 'black',
                    'concentrations': {
                        '0.0M': {'folder': '/path/to/NaCl/0.0M', 'label': '0.0 M'},
                        '0.5M': {'folder': '/path/to/NaCl/0.5M', 'label': '0.5 M'},
                        ...
                    }
                },
                'KCl': {
                    'label': 'KCl',
                    'color': 'red',
                    'concentrations': {
                        '0.0M': {'folder': '/path/to/KCl/0.0M', 'label': '0.0 M'},
                        ...
                    }
                },
                ...
            }
        '''
        self.salt_data = salt_data
        self.loaded_data = {}  # Will store: {salt_key: {conc_key: data}}
        # Instance-level cache: survives %autoreload 2 reloads because instance
        # __dict__ is preserved when the class is redefined by autoreload.
        if not hasattr(self, '_loaded_caches'):
            self._loaded_caches = {}

    def debug_data_structure(self, salt_key='NaCl', conc_key='0.11M'):
        '''Debug helper to see what's actually in the loaded data'''
        print("="*80)
        print(f"DEBUG: Data Structure for {salt_key} {conc_key}")
        print("="*80)
        
        if salt_key not in self.loaded_data:
            print(f"❌ Salt '{salt_key}' not found in loaded_data")
            print(f"Available salts: {list(self.loaded_data.keys())}")
            return
        
        if conc_key not in self.loaded_data[salt_key]:
            print(f"❌ Concentration '{conc_key}' not found for {salt_key}")
            print(f"Available concentrations: {list(self.loaded_data[salt_key].keys())}")
            return
        
        data = self.loaded_data[salt_key][conc_key]
        
        print(f"\n✓ Found data for {salt_key} {conc_key}")
        print(f"\nTop-level keys in data:")
        for key in data.keys():
            print(f"  - {key}")
        
        # Check RDFs structure
        if 'rdfs' in data:
            print(f"\n'rdfs' structure:")
            rdfs = data['rdfs']
            for key in rdfs.keys():
                print(f"  - {key}")
        else:
            print("\n❌ No 'rdfs' key found in data!")
        
        # Check coordination numbers
        if 'shell_coordination_numbers' in data:
            print(f"\n'shell_coordination_numbers' structure:")
            cn = data['shell_coordination_numbers']
            for ion_key in cn.keys():
                print(f"  - {ion_key}")
        else:
            print("\n❌ No 'shell_coordination_numbers' key found!")
        
        # Check residence times
        if 'water_residence_times' in data:
            print(f"\n'water_residence_times' structure:")
            res = data['water_residence_times']
            for ion_key in res.keys():
                print(f"  - {ion_key}")
        else:
            print("\n❌ No 'water_residence_times' key found!")
        
        print("="*80)

    def load_all_salts(self, filename_pattern='results_{conc}.pkl'):
        '''
        Load analysis data for all salts and concentrations.
        
        Parameters
        ----------
        filename_pattern : str
            Filename pattern with {conc} placeholder for concentration key.
            Default: 'results_{conc}.pkl'
            Examples:
            - 'results_{conc}.pkl' will look for results_0.0M.pkl, results_0.11M.pkl, etc.
            - 'solvation_analysis.pkl' for same filename everywhere
            - 'analysis_{conc}_data.pkl' for custom patterns
        
        Returns
        -------
        loaded_count : dict
            Dictionary with count of loaded concentrations per salt
        '''
        print("="*80)
        print("LOADING SALT DATA")
        print("="*80)
        
        loaded_count = {}
        
        for salt_key, salt_info in self.salt_data.items():
            print(f"\n{salt_info['label']}:")
            print("-"*40)
            
            self.loaded_data[salt_key] = {}
            loaded = 0
            
            for conc_key, conc_info in salt_info['concentrations'].items():
                folder = Path(conc_info['folder'])
                
                # Generate filename from pattern
                if '{conc}' in filename_pattern:
                    filename = filename_pattern.replace('{conc}', conc_key)
                else:
                    filename = filename_pattern
                
                filepath = folder / filename
                
                if filepath.exists():
                    try:
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                        self.loaded_data[salt_key][conc_key] = data
                        loaded += 1
                        print(f"  ✓ {conc_info['label']}: {filepath.name}")
                    except Exception as e:
                        print(f"  ✗ {conc_info['label']}: Error loading - {e}")
                else:
                    print(f"  ✗ {conc_info['label']}: File not found - {filepath}")
            
            loaded_count[salt_key] = loaded
            print(f"  Total loaded: {loaded}/{len(salt_info['concentrations'])}")
        
        print("\n" + "="*80)
        print(f"TOTAL LOADED: {sum(loaded_count.values())} concentrations across {len(loaded_count)} salts")
        print("="*80)
        
        return loaded_count


    def compare_rdfs(self, ion_type, pair_type='ion_water', concentration='0.5M',
                    salts=None,
                    save_plot=True, save_log=True,
                    xlabel_fontsize=12, ylabel_fontsize=12,
                    title_fontsize=14, legend_fontsize=10, tick_fontsize=10,
                    shell_boundaries=None,
                    ion_pairing_boundaries=None,
                    alpha=0.35):
        '''
        Compare RDFs for a specific ion across different salts at a given concentration.
        
        Parameters
        ----------
        ion_type : str
            Ion to compare (e.g., 'Na', 'K', 'Mg', 'Ca', 'Cl')
        pair_type : str
            Type of RDF: 'ion_water', 'ion_ion', or 'water_water'
        concentration : str
            Concentration to compare (e.g., '0.5M')
        salts : list of str, optional
            List of salt keys to compare (e.g., ['NaCl', 'KCl']). If None, uses all available.
        save_plot : bool
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
        shell_boundaries : dict, optional
            For ion_water: Dictionary with shell boundaries
            Format: {'shell_1': (r_min, r_max), 'shell_2': (r_min, r_max), ...}
        ion_pairing_boundaries : dict, optional
            For ion_ion: Dictionary with ion pairing region boundaries
            Format: {'CIP': (r_min, r_max), 'SIP': (r_min, r_max), ...}
        alpha : float
            Transparency for shaded regions, default=0.35
        
        Returns
        -------
        comparison_data : dict
            Dictionary with RDF data for each salt
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_salts() first.")
            return None
        
        # Determine which salts to analyze
        if salts is None:
            salts_to_compare = list(self.loaded_data.keys())
        else:
            # Validate salt keys
            salts_to_compare = []
            for salt in salts:
                if salt in self.loaded_data:
                    salts_to_compare.append(salt)
                else:
                    print(f"WARNING: Salt '{salt}' not found in loaded data. Skipping.")
        
        if not salts_to_compare:
            print("ERROR: No valid salts to compare.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"RDF COMPARISON ACROSS SALTS: {ion_type} ({pair_type})")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Pair Type: {pair_type}")
        log_lines.append(f"Concentration: {concentration}")
        log_lines.append(f"Salts: {', '.join([self.salt_data[s]['label'] for s in salts_to_compare])}")
        log_lines.append("")
        
        # Collect RDF data
        comparison_data = {}
        
        log_lines.append("LOADING RDF DATA")
        log_lines.append("-"*80)
        
        for salt_key in salts_to_compare:
            if concentration not in self.loaded_data[salt_key]:
                log_lines.append(f"⚠ {self.salt_data[salt_key]['label']}: Concentration {concentration} not found")
                continue
            
            conc_data = self.loaded_data[salt_key][concentration]
            
            # Navigate actual data structure (Na-w, Cl-w, Na-Cl format)
            rdf_data = None
            
            if 'rdfs' in conc_data:
                rdfs_dict = conc_data['rdfs']
                
                # Build possible RDF keys based on pair_type
                possible_keys = []
                
                if pair_type == 'ion_water':
                    possible_keys = [
                        f'{ion_type}-w',
                        f'{ion_type}+-w',
                        f'{ion_type}--w',
                        f'{ion_type.upper()}-w'
                    ]
                elif pair_type == 'ion_ion':
                    possible_keys = [
                        f'{ion_type}-Cl',
                        f'{ion_type}-{ion_type}',
                        f'Cl-{ion_type}',
                        f'{ion_type}+-Cl-'
                    ]
                elif pair_type == 'water_water':
                    possible_keys = ['w-w', 'water-water', 'O-O']
                
                # Try to find RDF data
                for key in possible_keys:
                    if key in rdfs_dict:
                        rdf_raw = rdfs_dict[key]
                        
                        # Handle the actual data format {'bins': array, 'rdf': array}
                        if isinstance(rdf_raw, dict):
                            if 'bins' in rdf_raw and 'rdf' in rdf_raw:
                                # This is the format! bins = r values, rdf = g(r) values
                                rdf_data = {'r': rdf_raw['bins'], 'g_r': rdf_raw['rdf']}
                                log_lines.append(f"✓ {self.salt_data[salt_key]['label']}: Found RDF: {key}")
                                break
                            elif 'r' in rdf_raw and 'g_r' in rdf_raw:
                                # Alternative direct format
                                rdf_data = rdf_raw
                                log_lines.append(f"✓ {self.salt_data[salt_key]['label']}: Found RDF: {key}")
                                break
                            else:
                                log_lines.append(f"⚠ {self.salt_data[salt_key]['label']}: Found {key} but unknown dict format. Keys: {list(rdf_raw.keys())}")
                        elif isinstance(rdf_raw, np.ndarray):
                            # Fallback: plain array format
                            g_r = rdf_raw
                            bin_size = 0.1  # Default
                            if 'system_info' in conc_data:
                                sys_info = conc_data['system_info']
                                bin_size = sys_info.get('rdf_bin_size', sys_info.get('bin_size', 0.1))
                            n_bins = len(g_r)
                            r = np.arange(n_bins) * bin_size + bin_size / 2
                            rdf_data = {'r': r, 'g_r': g_r}
                            log_lines.append(f"✓ {self.salt_data[salt_key]['label']}: Found RDF: {key} (array format)")
                            break
            
            if rdf_data is not None and isinstance(rdf_data, dict) and 'r' in rdf_data and 'g_r' in rdf_data:
                comparison_data[salt_key] = rdf_data
            else:
                log_lines.append(f"⚠ {self.salt_data[salt_key]['label']}: No RDF data found for {ion_type}")
        
        if not comparison_data:
            log_lines.append("")
            log_lines.append(f"ERROR: No RDF data found for {ion_type}")
            print("\n".join(log_lines))
            return None
        
        log_lines.append("")
        log_lines.append(f"Total salts with data: {len(comparison_data)}")
        log_lines.append("")
        
        # Plot RDFs
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # FIRST: Plot all RDF lines and determine x-axis range
        log_lines.append("RDF STATISTICS")
        log_lines.append("-"*80)
        log_lines.append(f"{'Salt':<20} {'First Peak (Å)':<20} {'g(r) max':<15}")
        log_lines.append("-"*80)
        
        max_r = 0
        for salt_key in sorted(comparison_data.keys()):
            rdf_info = comparison_data[salt_key]
            salt_label = self.salt_data[salt_key]['label']
            color = self.salt_data[salt_key].get('color', None)
            
            r = rdf_info['r']
            g_r = rdf_info['g_r']
            
            # Track maximum r value
            max_r = max(max_r, r[-1])
            
            ax.plot(r, g_r, label=salt_label, linewidth=2, color=color, zorder=10)
            
            # Find first peak
            first_peak_idx = np.argmax(g_r)
            first_peak_r = r[first_peak_idx]
            first_peak_g = g_r[first_peak_idx]
            
            log_lines.append(f"{salt_label:<20} {first_peak_r:<20.3f} {first_peak_g:<15.3f}")
        
        log_lines.append("")
        
        # Set x-axis limits: start at 0, end at max r value
        ax.set_xlim(0, max_r)
        
        # Get current y-limits AFTER plotting
        y_min, y_max = ax.get_ylim()
        
        # Calculate where we want to place labels (90% of current y_max)
        label_y_position = y_max * 0.90
        
        # Check if we need to extend y_max to make room for labels
        # Find the maximum g(r) value across all curves
        max_g_across_all = 0
        for salt_key in comparison_data.keys():
            rdf_info = comparison_data[salt_key]
            max_g_across_all = max(max_g_across_all, np.max(rdf_info['g_r']))
        
        # If labels would be too close to the curves, extend y_max
        if label_y_position < max_g_across_all * 1.05:
            new_y_max = max_g_across_all * 1.2  # 20% above highest peak
            ax.set_ylim(y_min, new_y_max)
            y_min, y_max = ax.get_ylim()  # Update y_max
            label_y_position = y_max * 0.90
        
        # NOW: Add shading AFTER plotting and y-axis adjustment
        if pair_type == 'ion_water' and shell_boundaries is not None:
            # Define blue saturation color gradient (matching ConcentrationComparison)
            import matplotlib.colors as mcolors
            
            def get_blue_colors(n_shells):
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
            
            # Get shell items and colors
            shell_items = [(k, v) for k, v in shell_boundaries.items() if k.startswith('shell_')]
            shell_items.sort(key=lambda x: x[1] if isinstance(x[1], tuple) else (x[1]['r_min'] if isinstance(x[1], dict) and 'r_min' in x[1] else 0))
            
            n_shells = len(shell_items)
            all_colors = get_blue_colors(n_shells)
            shell_colors = all_colors[:-1]
            bulk_color = all_colors[-1]
            
            log_lines.append("SOLVATION SHELL BOUNDARIES")
            log_lines.append("-"*80)
            
            # Add shell shading - ALL LABELS AT SAME Y LEVEL
            for i, (shell_name, bounds) in enumerate(shell_items):
                if isinstance(bounds, dict):
                    r_min, r_max = bounds['r_min'], bounds['r_max']
                else:
                    r_min, r_max = bounds
                
                color = shell_colors[i]
                ax.axvspan(r_min, r_max, alpha=alpha, color=color, zorder=0)
                
                mid_point = (r_min + r_max) / 2
                label_text = shell_name.replace('_', ' ').title()
                
                # ALL LABELS AT THE SAME Y LEVEL
                ax.text(mid_point, label_y_position, label_text,
                    ha='center', va='center', fontsize=10, fontweight='bold', 
                    color='black', zorder=15)
                
                log_lines.append(f"  {shell_name}: {r_min:.2f} - {r_max:.2f} Å (color: {color})")
            
            # Add bulk region - everything after last shell, but only up to max_r
            if shell_items:
                last_bounds = shell_items[-1][1]
                bulk_start = last_bounds['r_max'] if isinstance(last_bounds, dict) else last_bounds[1]
                
                if bulk_start < max_r:
                    ax.axvspan(bulk_start, max_r, alpha=alpha, color=bulk_color, zorder=0)
                    
                    bulk_mid = (bulk_start + max_r) / 2
                    
                    # BULK LABEL AT SAME Y LEVEL
                    ax.text(bulk_mid, label_y_position, 'Bulk',
                        ha='center', va='center', fontsize=10, fontweight='bold', 
                        color='black', zorder=15)
                    
                    log_lines.append(f"  Bulk: {bulk_start:.2f} - {max_r:.2f} Å (color: {bulk_color})")
            
            log_lines.append("")
        
        elif pair_type == 'ion_ion' and ion_pairing_boundaries is not None:
            # DARKER colors with better contrast
            region_colors = {
                'CIP': '#FFB3B3',   # Darker coral red
                'SIP': '#B3D9FF',   # Darker sky blue
                'DSIP': '#B3FFB3',  # Darker mint green
                'FI': '#E6E6E6'     # Darker light gray
            }
            
            log_lines.append("ION PAIRING REGION BOUNDARIES")
            log_lines.append("-"*80)
            
            # Find the last defined region to determine FI start
            defined_regions = []
            for region_name in ['CIP', 'SIP', 'DSIP']:
                if region_name in ion_pairing_boundaries:
                    bounds = ion_pairing_boundaries[region_name]
                    if isinstance(bounds, dict):
                        r_min = bounds.get('r_min', bounds.get('min', 0))
                        r_max = bounds.get('r_max', bounds.get('max', 10))
                    else:
                        r_min, r_max = bounds
                    defined_regions.append((region_name, r_min, r_max))
            
            # Sort by r_min
            defined_regions.sort(key=lambda x: x[1])
            
            # Plot defined regions - ALL LABELS AT SAME Y LEVEL
            for region_name, r_min, r_max in defined_regions:
                color = region_colors.get(region_name, '#F0F0F0')
                ax.axvspan(r_min, r_max, alpha=alpha, color=color, zorder=0)
                
                r_mid = (r_min + r_max) / 2
                
                # ALL LABELS AT THE SAME Y LEVEL
                ax.text(r_mid, label_y_position, region_name,
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='black', zorder=15)
                
                log_lines.append(f"{region_name}: {r_min:.2f} - {r_max:.2f} Å")
            
            # Add FI region - everything after last defined region, but only up to max_r
            if defined_regions:
                fi_start = defined_regions[-1][2]  # r_max of last region
                
                if fi_start < max_r:
                    color = region_colors['FI']
                    ax.axvspan(fi_start, max_r, alpha=alpha, color=color, zorder=0)
                    
                    fi_mid = (fi_start + max_r) / 2
                    
                    # FI LABEL AT SAME Y LEVEL
                    ax.text(fi_mid, label_y_position, 'FI',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='black', zorder=15)
                    
                    log_lines.append(f"FI: {fi_start:.2f} - {max_r:.2f} Å")
            
            log_lines.append("")
        
        # Customize plot
        ax.set_xlabel('r (Å)', fontsize=xlabel_fontsize)
        ax.set_ylabel('g(r)', fontsize=ylabel_fontsize)
        
        if salts is not None:
            salts_str = ', '.join([self.salt_data[s]['label'] for s in salts_to_compare])
            ax.set_title(f'RDF Comparison: {ion_type} ({pair_type}) at {concentration}\n{salts_str}',
                        fontsize=title_fontsize, fontweight='bold')
        else:
            ax.set_title(f'RDF Comparison: {ion_type} ({pair_type}) at {concentration}',
                        fontsize=title_fontsize, fontweight='bold')
        
        ax.legend(fontsize=legend_fontsize, frameon=False)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        # REMOVED GRID
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        
        plt.tight_layout()
            
        # Save files
        if save_plot:
            salt_str = '_'.join(salts_to_compare) if salts else 'all'
            filename = f'salt_comparison_rdf_{ion_type}_{pair_type}_{concentration}_{salt_str}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            log_lines.append("")
            log_lines.append(f"PLOT SAVED: {filename}")
            print(f"Plot saved: {filename}")
        
        if save_log:
            salt_str = '_'.join(salts_to_compare) if salts else 'all'
            log_filename = f'salt_comparison_rdf_{ion_type}_{pair_type}_{concentration}_{salt_str}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Log saved: {log_filename}")
        
        plt.show()
        
        return comparison_data


    def compare_rdfs_ridge(self, ion_type, pair_type='ion_water', concentration='0.5M',
                        salts=None, save_plot=True, save_log=True,
                        xlabel_fontsize=12, ylabel_fontsize=12, title_fontsize=14,
                        legend_fontsize=10, tick_fontsize=10,
                        vertical_spacing=2.0, shell_boundaries=None,
                        ion_pairing_boundaries=None, alpha=0.35):
        '''
        Create ridge plot (stacked curves) of RDFs across different salts.
        
        Parameters
        ----------
        ion_type : str
            Ion to compare (e.g., 'Na', 'K', 'Mg', 'Ca', 'Cl')
        pair_type : str
            Type of RDF: 'ion_water', 'ion_ion', or 'water_water'
        concentration : str
            Concentration to compare (e.g., '0.5M')
        salts : list of str, optional
            List of salt keys to compare. If None, uses all available.
        save_plot : bool
            Whether to save the plot
        save_log : bool
            Whether to save analysis log
        xlabel_fontsize : int
            Font size for x-axis label
        ylabel_fontsize : int
            Font size for y-axis label
        title_fontsize : int
            Font size for title
        legend_fontsize : int
            Font size for legend
        tick_fontsize : int
            Font size for tick labels
        vertical_spacing : float
            Vertical spacing between curves
        shell_boundaries : dict, optional
            Shell boundaries for shading
        ion_pairing_boundaries : dict, optional
            Ion pairing boundaries for shading
        alpha : float
            Transparency for shaded regions
        
        Returns
        -------
        comparison_data : dict
            Dictionary with RDF data for each salt
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_salts() first.")
            return None
        
        # Get comparison data using existing method
        comparison_data = self.compare_rdfs(
            ion_type=ion_type, pair_type=pair_type, concentration=concentration,
            salts=salts, save_plot=False, save_log=save_log,
            shell_boundaries=shell_boundaries, ion_pairing_boundaries=ion_pairing_boundaries
        )
        
        if not comparison_data:
            return None
        
        # Determine which salts we have
        salts_to_plot = list(comparison_data.keys())
        
        # Create ridge plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        max_r = 0
        y_positions = []
        
        for i, salt_key in enumerate(salts_to_plot):
            rdf_info = comparison_data[salt_key]
            salt_label = self.salt_data[salt_key]['label']
            color = self.salt_data[salt_key].get('color', None)
            
            r = rdf_info['r']
            g_r = rdf_info['g_r']
            
            # Track maximum r value
            max_r = max(max_r, r[-1])
            
            # Calculate vertical offset
            y_offset = i * vertical_spacing
            y_positions.append(y_offset)
            
            # Plot the offset curve
            ax.plot(r, g_r + y_offset, label=salt_label, linewidth=2, color=color, zorder=10)
            
            # Fill under the curve for better visual effect
            ax.fill_between(r, y_offset, g_r + y_offset, alpha=0.3, color=color, zorder=5)
            
            # Add salt label on the left
            ax.text(-0.5, y_offset + 1, salt_label, fontsize=12, fontweight='bold',
                ha='right', va='center', color=color)
        
        # Add shaded regions for shells/regions
        max_y = max(y_positions) + vertical_spacing
        
        if pair_type == 'ion_water' and shell_boundaries is not None:
            import matplotlib.colors as mcolors
            
            def get_blue_colors(n_shells):
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
            
            shell_items = [(k, v) for k, v in shell_boundaries.items() if k.startswith('shell_')]
            shell_items.sort(key=lambda x: x[1] if isinstance(x[1], tuple) else 
                            (x[1]['r_min'] if isinstance(x[1], dict) and 'r_min' in x[1] else 0))
            
            n_shells = len(shell_items)
            all_colors = get_blue_colors(n_shells)
            shell_colors = all_colors[:-1]
            bulk_color = all_colors[-1]
            
            for i, (shell_name, bounds) in enumerate(shell_items):
                if isinstance(bounds, dict):
                    r_min, r_max = bounds['r_min'], bounds['r_max']
                else:
                    r_min, r_max = bounds
                
                color = shell_colors[i]
                ax.axvspan(r_min, r_max, alpha=alpha, color=color, zorder=0)
                
                # Add label at top
                mid_point = (r_min + r_max) / 2
                label_text = shell_name.replace('_', ' ').title()
                ax.text(mid_point, max_y * 0.95, label_text,
                    ha='center', va='center', fontsize=10, fontweight='bold', 
                    color='black', zorder=15)
            
            # Add bulk region
            if shell_items:
                last_bounds = shell_items[-1][1]
                bulk_start = last_bounds['r_max'] if isinstance(last_bounds, dict) else last_bounds[1]
                
                if bulk_start < max_r:
                    ax.axvspan(bulk_start, max_r, alpha=alpha, color=bulk_color, zorder=0)
                    bulk_mid = (bulk_start + max_r) / 2
                    ax.text(bulk_mid, max_y * 0.95, 'Bulk',
                        ha='center', va='center', fontsize=10, fontweight='bold', 
                        color='black', zorder=15)
        
        elif pair_type == 'ion_ion' and ion_pairing_boundaries is not None:
            region_colors = {
                'CIP': '#FFB3B3', 'SIP': '#B3D9FF', 'DSIP': '#B3FFB3', 'FI': '#E6E6E6'
            }
            
            defined_regions = []
            for region_name in ['CIP', 'SIP', 'DSIP']:
                if region_name in ion_pairing_boundaries:
                    bounds = ion_pairing_boundaries[region_name]
                    if isinstance(bounds, dict):
                        r_min = bounds.get('r_min', bounds.get('min', 0))
                        r_max = bounds.get('r_max', bounds.get('max', 10))
                    else:
                        r_min, r_max = bounds
                    defined_regions.append((region_name, r_min, r_max))
            
            defined_regions.sort(key=lambda x: x[1])
            
            for region_name, r_min, r_max in defined_regions:
                color = region_colors.get(region_name, '#F0F0F0')
                ax.axvspan(r_min, r_max, alpha=alpha, color=color, zorder=0)
                
                r_mid = (r_min + r_max) / 2
                ax.text(r_mid, max_y * 0.95, region_name,
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='black', zorder=15)
            
            # Add FI region
            if defined_regions:
                fi_start = defined_regions[-1][2]
                if fi_start < max_r:
                    color = region_colors['FI']
                    ax.axvspan(fi_start, max_r, alpha=alpha, color=color, zorder=0)
                    fi_mid = (fi_start + max_r) / 2
                    ax.text(fi_mid, max_y * 0.95, 'FI',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='black', zorder=15)
        
        # Customize plot
        ax.set_xlabel('r (Å)', fontsize=xlabel_fontsize)
        ax.set_ylabel('g(r) + offset', fontsize=ylabel_fontsize)
        ax.set_title(f'Ridge Plot: {ion_type} ({pair_type}) at {concentration}',
                    fontsize=title_fontsize, fontweight='bold')
        
        # Remove y-axis ticks (not meaningful with offsets)
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        
        # Set x-axis limits
        ax.set_xlim(0, max_r)
        ax.set_ylim(-0.5, max_y)
        
        plt.tight_layout()
        
        if save_plot:
            salt_str = '_'.join(salts_to_plot)
            filename = f'salt_comparison_rdf_ridge_{ion_type}_{pair_type}_{concentration}_{salt_str}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Ridge plot saved: {filename}")
        
        plt.show()
        
        return comparison_data


    def compare_rdfs_multi_concentration(self, ion_type, pair_type='ion_water',
                                        concentrations=None, salts=None,
                                        save_plot=True, save_log=True,
                                        xlabel_fontsize=12, ylabel_fontsize=12,
                                        title_fontsize=14, legend_fontsize=10,
                                        tick_fontsize=10,
                                        shell_boundaries=None,
                                        ion_pairing_boundaries=None,
                                        alpha=0.35):
        '''
        Compare RDFs across salts AND concentrations on the same plot.
        
        Parameters
        ----------
        ion_type : str
            Ion to compare (e.g., 'Na', 'K', 'Cl')
        pair_type : str
            Type of RDF: 'ion_water', 'ion_ion', or 'water_water'
        concentrations : list of str, optional
            List of concentrations to compare. If None, uses all available.
        salts : list of str, optional
            List of salt keys to compare (e.g., ['NaCl', 'KCl']). If None, uses all available.
        save_plot : bool
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
            Font size for tick labels, default=10
        shell_boundaries : dict, optional
            For ion_water: Dictionary with shell boundaries
            Format: {'shell_1': (r_min, r_max), 'shell_2': (r_min, r_max), ...}
        ion_pairing_boundaries : dict, optional
            For ion_ion: Dictionary with ion pairing region boundaries
            Format: {'CIP': (r_min, r_max), 'SIP': (r_min, r_max), 'DSIP': (r_min, r_max)}
        alpha : float
            Transparency for shaded regions, default=0.35
        
        Returns
        -------
        comparison_data : dict
            Dictionary with RDF data: {salt_key: {conc_key: rdf_data}}
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_salts() first.")
            return None
        
        # Determine which salts to analyze
        if salts is None:
            salts_to_compare = list(self.loaded_data.keys())
        else:
            salts_to_compare = []
            for salt in salts:
                if salt in self.loaded_data:
                    salts_to_compare.append(salt)
                else:
                    print(f"WARNING: Salt '{salt}' not found in loaded data. Skipping.")
        
        if not salts_to_compare:
            print("ERROR: No valid salts to compare.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"RDF MULTI-CONCENTRATION COMPARISON: {ion_type} ({pair_type})")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Pair Type: {pair_type}")
        log_lines.append(f"Salts: {', '.join([self.salt_data[s]['label'] for s in salts_to_compare])}")
        log_lines.append("")
        
        # Determine concentrations
        if concentrations is None:
            all_concs = set()
            for salt_key in salts_to_compare:
                all_concs.update(self.loaded_data[salt_key].keys())
            concentrations = sorted(list(all_concs), key=lambda x: float(x.replace('M', '')))
        
        log_lines.append(f"Concentrations: {', '.join(concentrations)}")
        log_lines.append("")
        
        # Collect RDF data
        comparison_data = {}
        
        log_lines.append("LOADING RDF DATA")
        log_lines.append("-"*80)
        
        for salt_key in salts_to_compare:
            comparison_data[salt_key] = {}
            
            for conc_key in concentrations:
                if conc_key not in self.loaded_data[salt_key]:
                    continue
                
                conc_data = self.loaded_data[salt_key][conc_key]
                
                rdf_data = None
                
                if 'rdfs' in conc_data:
                    rdfs_dict = conc_data['rdfs']
                    
                    possible_keys = []
                    
                    if pair_type == 'ion_water':
                        possible_keys = [
                            f'{ion_type}-w',
                            f'{ion_type}+-w',
                            f'{ion_type}--w',
                            f'{ion_type.upper()}-w'
                        ]
                    elif pair_type == 'ion_ion':
                        possible_keys = [
                            f'{ion_type}-Cl',
                            f'{ion_type}-{ion_type}',
                            f'Cl-{ion_type}',
                            f'{ion_type}+-Cl-',
                            f'{ion_type.upper()}-CL'
                        ]
                    elif pair_type == 'water_water':
                        possible_keys = ['w-w', 'water-water', 'O-O']
                    
                    for key in possible_keys:
                        if key in rdfs_dict:
                            rdf_raw = rdfs_dict[key]
                            
                            if isinstance(rdf_raw, dict):
                                if 'bins' in rdf_raw and 'rdf' in rdf_raw:
                                    rdf_data = {'r': rdf_raw['bins'], 'g_r': rdf_raw['rdf']}
                                    log_lines.append(f"✓ {self.salt_data[salt_key]['label']:<10} {conc_key:<10} Found RDF: {key}")
                                    break
                                elif 'r' in rdf_raw and 'g_r' in rdf_raw:
                                    rdf_data = rdf_raw
                                    log_lines.append(f"✓ {self.salt_data[salt_key]['label']:<10} {conc_key:<10} Found RDF: {key}")
                                    break
                                else:
                                    log_lines.append(f"⚠ {self.salt_data[salt_key]['label']:<10} {conc_key:<10} Found {key} but unknown dict format. Keys: {list(rdf_raw.keys())}")
                            elif isinstance(rdf_raw, np.ndarray):
                                g_r = rdf_raw
                                bin_size = 0.1
                                if 'system_info' in conc_data:
                                    sys_info = conc_data['system_info']
                                    bin_size = sys_info.get('rdf_bin_size', sys_info.get('bin_size', 0.1))
                                n_bins = len(g_r)
                                r = np.arange(n_bins) * bin_size + bin_size / 2
                                rdf_data = {'r': r, 'g_r': g_r}
                                log_lines.append(f"✓ {self.salt_data[salt_key]['label']:<10} {conc_key:<10} Found RDF: {key} (array format)")
                                break
                
                if rdf_data is not None and isinstance(rdf_data, dict) and 'r' in rdf_data and 'g_r' in rdf_data:
                    comparison_data[salt_key][conc_key] = rdf_data
                else:
                    log_lines.append(f"⚠ {self.salt_data[salt_key]['label']:<10} {conc_key:<10} No valid RDF data found (tried: {', '.join(possible_keys[:3])})")
        
        if not comparison_data or all(len(v) == 0 for v in comparison_data.values()):
            log_lines.append("")
            log_lines.append(f"ERROR: No RDF data found for {ion_type}")
            print("\n".join(log_lines))
            return None
        
        log_lines.append("")
        
        # Plot RDFs
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # FIRST: Plot all RDF lines and determine x-axis range
        log_lines.append("RDF STATISTICS")
        log_lines.append("-"*80)
        log_lines.append(f"{'Salt':<15} {'Conc':<10} {'First Peak (Å)':<20} {'g(r) max':<15}")
        log_lines.append("-"*80)
        
        line_styles = ['-', '--', '-.', ':']
        max_r = 0
        
        for salt_key in sorted(comparison_data.keys()):
            if not comparison_data[salt_key]:
                continue
            
            salt_label = self.salt_data[salt_key]['label']
            salt_color = self.salt_data[salt_key].get('color', None)
            
            for i, conc_key in enumerate(sorted(comparison_data[salt_key].keys(),
                                            key=lambda x: float(x.replace('M', '')))):
                rdf_info = comparison_data[salt_key][conc_key]
                
                r = rdf_info['r']
                g_r = rdf_info['g_r']
                
                # Track maximum r value
                max_r = max(max_r, r[-1])
                
                label = f"{salt_label} {conc_key}"
                linestyle = line_styles[i % len(line_styles)]
                
                ax.plot(r, g_r, label=label, linewidth=2, 
                    color=salt_color, linestyle=linestyle, zorder=10)
                
                first_peak_idx = np.argmax(g_r)
                first_peak_r = r[first_peak_idx]
                first_peak_g = g_r[first_peak_idx]
                
                log_lines.append(f"{salt_label:<15} {conc_key:<10} {first_peak_r:<20.3f} {first_peak_g:<15.3f}")
        
        log_lines.append("")
        
        # Set x-axis limits: start at 0, end at max r value
        ax.set_xlim(0, max_r)
        
        # Get current y-limits AFTER plotting
        y_min, y_max = ax.get_ylim()
        
        # Calculate where we want to place labels (90% of current y_max)
        label_y_position = y_max * 0.90
        
        # Check if we need to extend y_max to make room for labels
        # Find the maximum g(r) value across all curves
        max_g_across_all = 0
        for salt_key in comparison_data.keys():
            for conc_key in comparison_data[salt_key].keys():
                rdf_info = comparison_data[salt_key][conc_key]
                max_g_across_all = max(max_g_across_all, np.max(rdf_info['g_r']))
        
        # If labels would be too close to the curves, extend y_max
        if label_y_position < max_g_across_all * 1.05:
            new_y_max = max_g_across_all * 1.2  # 20% above highest peak
            ax.set_ylim(y_min, new_y_max)
            y_min, y_max = ax.get_ylim()  # Update y_max
            label_y_position = y_max * 0.90
        
        # NOW: Add shading AFTER plotting and y-axis adjustment
        if pair_type == 'ion_water' and shell_boundaries is not None:
            import matplotlib.colors as mcolors
            
            def get_blue_colors(n_shells):
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
            
            shell_items = [(k, v) for k, v in shell_boundaries.items() if k.startswith('shell_')]
            shell_items.sort(key=lambda x: x[1] if isinstance(x[1], tuple) else (x[1]['r_min'] if isinstance(x[1], dict) and 'r_min' in x[1] else 0))
            
            n_shells = len(shell_items)
            all_colors = get_blue_colors(n_shells)
            shell_colors = all_colors[:-1]
            bulk_color = all_colors[-1]
            
            log_lines.append("SOLVATION SHELL BOUNDARIES")
            log_lines.append("-"*80)
            
            # Add shell shading - ALL LABELS AT SAME Y LEVEL
            for i, (shell_name, bounds) in enumerate(shell_items):
                if isinstance(bounds, dict):
                    r_min, r_max = bounds['r_min'], bounds['r_max']
                else:
                    r_min, r_max = bounds
                
                color = shell_colors[i]
                ax.axvspan(r_min, r_max, alpha=alpha, color=color, zorder=0)
                
                mid_point = (r_min + r_max) / 2
                label_text = shell_name.replace('_', ' ').title()
                
                # ALL LABELS AT THE SAME Y LEVEL
                ax.text(mid_point, label_y_position, label_text,
                    ha='center', va='center', fontsize=10, fontweight='bold', 
                    color='black', zorder=15)
                
                log_lines.append(f"  {shell_name}: {r_min:.2f} - {r_max:.2f} Å (color: {color})")
            
            # Add bulk region - everything after last shell, but only up to max_r
            if shell_items:
                last_bounds = shell_items[-1][1]
                bulk_start = last_bounds['r_max'] if isinstance(last_bounds, dict) else last_bounds[1]
                
                if bulk_start < max_r:
                    ax.axvspan(bulk_start, max_r, alpha=alpha, color=bulk_color, zorder=0)
                    
                    bulk_mid = (bulk_start + max_r) / 2
                    
                    # BULK LABEL AT SAME Y LEVEL
                    ax.text(bulk_mid, label_y_position, 'Bulk',
                        ha='center', va='center', fontsize=10, fontweight='bold', 
                        color='black', zorder=15)
                    
                    log_lines.append(f"  Bulk: {bulk_start:.2f} - {max_r:.2f} Å (color: {bulk_color})")
            
            log_lines.append("")
        
        elif pair_type == 'ion_ion' and ion_pairing_boundaries is not None:
            # DARKER colors with better contrast
            region_colors = {
                'CIP': '#FFB3B3',   # Darker coral red
                'SIP': '#B3D9FF',   # Darker sky blue
                'DSIP': '#B3FFB3',  # Darker mint green
                'FI': '#E6E6E6'     # Darker light gray
            }
            
            log_lines.append("ION PAIRING REGION BOUNDARIES")
            log_lines.append("-"*80)
            
            # Find the last defined region to determine FI start
            defined_regions = []
            for region_name in ['CIP', 'SIP', 'DSIP']:
                if region_name in ion_pairing_boundaries:
                    bounds = ion_pairing_boundaries[region_name]
                    if isinstance(bounds, dict):
                        r_min = bounds.get('r_min', bounds.get('min', 0))
                        r_max = bounds.get('r_max', bounds.get('max', 10))
                    else:
                        r_min, r_max = bounds
                    defined_regions.append((region_name, r_min, r_max))
            
            # Sort by r_min
            defined_regions.sort(key=lambda x: x[1])
            
            # Plot defined regions - ALL LABELS AT SAME Y LEVEL
            for region_name, r_min, r_max in defined_regions:
                color = region_colors.get(region_name, '#F0F0F0')
                ax.axvspan(r_min, r_max, alpha=alpha, color=color, zorder=0)
                
                r_mid = (r_min + r_max) / 2
                
                # ALL LABELS AT THE SAME Y LEVEL
                ax.text(r_mid, label_y_position, region_name,
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='black', zorder=15)
                
                log_lines.append(f"{region_name}: {r_min:.2f} - {r_max:.2f} Å")
            
            # Add FI region - everything after last defined region, but only up to max_r
            if defined_regions:
                fi_start = defined_regions[-1][2]  # r_max of last region
                
                if fi_start < max_r:
                    color = region_colors['FI']
                    ax.axvspan(fi_start, max_r, alpha=alpha, color=color, zorder=0)
                    
                    fi_mid = (fi_start + max_r) / 2
                    
                    # FI LABEL AT SAME Y LEVEL
                    ax.text(fi_mid, label_y_position, 'FI',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='black', zorder=15)
                    
                    log_lines.append(f"FI: {fi_start:.2f} - {max_r:.2f} Å")
            
            log_lines.append("")
        
        # Customize plot
        ax.set_xlabel('r (Å)', fontsize=xlabel_fontsize)
        ax.set_ylabel('g(r)', fontsize=ylabel_fontsize)
        
        if salts is not None:
            salts_str = ', '.join([self.salt_data[s]['label'] for s in salts_to_compare])
            ax.set_title(f'RDF Comparison: {ion_type} ({pair_type}) - Multiple Concentrations\n{salts_str}',
                        fontsize=title_fontsize, fontweight='bold')
        else:
            ax.set_title(f'RDF Comparison: {ion_type} ({pair_type}) - Multiple Concentrations',
                        fontsize=title_fontsize, fontweight='bold')
        
        ax.legend(fontsize=legend_fontsize, frameon=False, loc='best')
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        # REMOVED GRID
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        
        plt.tight_layout()
        
        if save_plot:
            conc_str = '_'.join(concentrations)
            salt_str = '_'.join(salts_to_compare) if salts else 'all'
            filename = f'salt_comparison_rdf_{ion_type}_{pair_type}_multi_{conc_str}_{salt_str}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            log_lines.append("")
            log_lines.append(f"PLOT SAVED: {filename}")
            print(f"Plot saved: {filename}")
        
        if save_log:
            conc_str = '_'.join(concentrations)
            salt_str = '_'.join(salts_to_compare) if salts else 'all'
            log_filename = f'salt_comparison_rdf_{ion_type}_{pair_type}_multi_{conc_str}_{salt_str}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Log saved: {log_filename}")
        
        plt.show()
        
        return comparison_data


    def compare_rdfs_3d(self, ion_type, pair_type='ion_water', concentration='0.5M',
                        salts=None, save_plot=True, save_log=True,
                        title_fontsize=14, label_fontsize=12, tick_fontsize=10,
                        shell_boundaries=None, ion_pairing_boundaries=None,
                        alpha=0.35, elevation=20, azimuth=45,
                        fill_under_curve=True, fill_alpha=0.3, 
                        line_width=2.0, show_grid=True, grid_alpha=0.2,
                        min_r=None, max_r=None):
        '''
        Create 3D visualization of RDFs across different salts.
        
        Parameters
        ----------
        ion_type : str
            Ion to compare (e.g., 'Na', 'K', 'Mg', 'Ca', 'Cl')
        pair_type : str
            Type of RDF: 'ion_water', 'ion_ion', or 'water_water'
        concentration : str
            Concentration to compare (e.g., '0.5M')
        salts : list of str, optional
            List of salt keys to compare. If None, uses all available.
        save_plot : bool
            Whether to save the plot
        save_log : bool
            Whether to save analysis log
        title_fontsize : int
            Font size for title
        label_fontsize : int
            Font size for axis labels
        tick_fontsize : int
            Font size for tick labels
        shell_boundaries : dict, optional
            Shell boundaries for shading
        ion_pairing_boundaries : dict, optional
            Ion pairing boundaries for shading
        alpha : float
            Transparency for shaded regions
        elevation : float
            Viewing elevation angle (degrees)
        azimuth : float
            Viewing azimuth angle (degrees)
        fill_under_curve : bool, default=True
            Whether to fill area under RDF curves
        fill_alpha : float, default=0.3
            Transparency for curve fill
        line_width : float, default=2.0
            Width of RDF curve lines
        show_grid : bool, default=True
            Whether to show grid lines
        grid_alpha : float, default=0.2
            Transparency of grid lines (0.0 = invisible, 1.0 = opaque)
        min_r : float, optional
            Minimum r(Å) value to plot. If None, starts from 0.
        max_r : float, optional
            Maximum r(Å) value to plot. If None, uses full range from data.
        
        Returns
        -------
        comparison_data : dict
            Dictionary with RDF data for each salt
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_salts() first.")
            return None
        
        # Get comparison data using existing method
        comparison_data = self.compare_rdfs(
            ion_type=ion_type, pair_type=pair_type, concentration=concentration,
            salts=salts, save_plot=False, save_log=save_log,
            shell_boundaries=shell_boundaries, ion_pairing_boundaries=ion_pairing_boundaries
        )
        
        if not comparison_data:
            return None
        
        # Determine which salts we have
        salts_to_plot = list(comparison_data.keys())
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        
        # Prepare data for 3D plotting - SWAPPED AXES
        x_positions = np.arange(len(salts_to_plot))  # Salt positions along X-axis
        
        for i, salt_key in enumerate(salts_to_plot):
            rdf_info = comparison_data[salt_key]
            salt_label = self.salt_data[salt_key]['label']
            color = self.salt_data[salt_key].get('color', None)
            
            r = rdf_info['r']
            g_r = rdf_info['g_r']
            
            # Apply r range filters if specified
            mask = np.ones(len(r), dtype=bool)
            if min_r is not None:
                mask &= (r >= min_r)
            if max_r is not None:
                mask &= (r <= max_r)
            
            r = r[mask]
            g_r = g_r[mask]
            
            # Create X values (all same for this salt)
            x = np.full_like(r, x_positions[i])
            
            # Plot the 3D line - SWAPPED: (x=salt, y=r, z=g_r)
            ax.plot(x, r, g_r, label=salt_label, linewidth=line_width, color=color, alpha=0.8)
            
            # Fill under curve if requested
            if fill_under_curve:
                # Create surface from baseline (z=0) to curve
                n_points = len(r)
                
                # Create mesh for fill surface
                x_fill = np.vstack([x, x])  # 2 x n_points
                y_fill = np.vstack([r, r])  # 2 x n_points (r values)
                z_fill = np.vstack([np.zeros_like(g_r), g_r])  # From 0 to g_r values
                
                ax.plot_surface(x_fill, y_fill, z_fill, 
                            color=color, alpha=fill_alpha, linewidth=0, shade=True)
        
        # Set Y-axis limits based on min_r and max_r
        y_min = min_r if min_r is not None else 0
        if max_r is not None:
            ax.set_ylim(y_min, max_r)
        else:
            ax.set_ylim(y_min, ax.get_ylim()[1])
        
        # Get z-axis limits for boundary planes
        z_min, z_max = ax.get_zlim()
        
        # Add shaded regions if provided - UPDATED FOR SWAPPED AXES
        if pair_type == 'ion_water' and shell_boundaries is not None:
            shell_items = [(k, v) for k, v in shell_boundaries.items() if k.startswith('shell_')]
            shell_items.sort(key=lambda x: x[1] if isinstance(x[1], tuple) else 
                            (x[1]['r_min'] if isinstance(x[1], dict) and 'r_min' in x[1] else 0))
            
            # Get blue colors
            import matplotlib.colors as mcolors
            base_rgb = mcolors.hex2color('#00c5ff')
            
            for i, (shell_name, bounds) in enumerate(shell_items):
                if isinstance(bounds, dict):
                    r_min_shell, r_max_shell = bounds['r_min'], bounds['r_max']
                else:
                    r_min_shell, r_max_shell = bounds
                
                # Only plot boundaries within the specified r range
                plot_r_min = min_r if min_r is not None else 0
                plot_r_max = max_r if max_r is not None else float('inf')
                
                if (r_min_shell <= plot_r_max and r_max_shell >= plot_r_min):
                    # Create vertical planes at shell boundaries - SWAPPED AXES
                    x_plane = np.linspace(-0.5, len(salts_to_plot) - 0.5, 10)
                    z_plane = np.linspace(z_min, z_max, 10)
                    X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
                    
                    # Add vertical planes at r_min and r_max - NOW Y-axis planes
                    Y_min = np.full_like(X_plane, r_min_shell)
                    Y_max = np.full_like(X_plane, r_max_shell)
                    
                    # Uncomment to show boundary planes
                    # ax.plot_surface(X_plane, Y_min, Z_plane, alpha=0.1, color=base_rgb)
                    # ax.plot_surface(X_plane, Y_max, Z_plane, alpha=0.1, color=base_rgb)
        
        elif pair_type == 'ion_ion' and ion_pairing_boundaries is not None:
            region_colors = {'CIP': '#FFB3B3', 'SIP': '#B3D9FF', 'DSIP': '#B3FFB3', 'FI': '#E6E6E6'}
            
            for region_name in ['CIP', 'SIP', 'DSIP']:
                if region_name in ion_pairing_boundaries:
                    bounds = ion_pairing_boundaries[region_name]
                    if isinstance(bounds, dict):
                        r_min_region = bounds.get('r_min', bounds.get('min', 0))
                        r_max_region = bounds.get('r_max', bounds.get('max', 10))
                    else:
                        r_min_region, r_max_region = bounds
                    
                    # Only plot boundaries within the specified r range
                    plot_r_min = min_r if min_r is not None else 0
                    plot_r_max = max_r if max_r is not None else float('inf')
                    
                    if (r_min_region <= plot_r_max and r_max_region >= plot_r_min):
                        # Add vertical planes - SWAPPED AXES
                        x_plane = np.linspace(-0.5, len(salts_to_plot) - 0.5, 10)
                        z_plane = np.linspace(z_min, z_max, 10)
                        X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
                        
                        color = region_colors[region_name]
                        Y_min = np.full_like(X_plane, r_min_region)
                        Y_max = np.full_like(X_plane, r_max_region)
                        
                        # Uncomment to show boundary planes
                        # ax.plot_surface(X_plane, Y_min, Z_plane, alpha=0.1, color=color)
                        # ax.plot_surface(X_plane, Y_max, Z_plane, alpha=0.1, color=color)
        
        # Customize 3D plot - SWAPPED AXIS LABELS
        ax.set_xlabel('Salt Type', fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel('r (Å)', fontsize=label_fontsize, labelpad=10)
        ax.set_zlabel('g(r)', fontsize=label_fontsize, labelpad=10)
        
        # Set X-axis labels to salt names - UPDATED FOR X-AXIS
        ax.set_xticks(x_positions)
        ax.set_xticklabels([self.salt_data[salt]['label'] for salt in salts_to_plot])
        
        # Update title to show r range if specified
        title_base = f'3D RDF Comparison: {ion_type} ({pair_type}) at {concentration}'
        if min_r is not None or max_r is not None:
            r_min_str = f"{min_r}" if min_r is not None else "0"
            r_max_str = f"{max_r}" if max_r is not None else "max"
            title_base += f' (r: {r_min_str}-{r_max_str} Å)'
        ax.set_title(title_base, fontsize=title_fontsize, fontweight='bold', pad=20)
        
        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)

        # Remove grey tint and make background clean
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Configure grid lines - FAINT GRID OPTION
        if show_grid:
            # Make pane edges with custom transparency
            ax.xaxis.pane.set_edgecolor('black')
            ax.yaxis.pane.set_edgecolor('black')
            ax.zaxis.pane.set_edgecolor('black')
            ax.xaxis.pane.set_alpha(grid_alpha)
            ax.yaxis.pane.set_alpha(grid_alpha)
            ax.zaxis.pane.set_alpha(grid_alpha)
            
            # Add faint grid lines
            ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5)
        else:
            # No grid at all
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            ax.xaxis.pane.set_alpha(0)
            ax.yaxis.pane.set_alpha(0)
            ax.zaxis.pane.set_alpha(0)
            ax.grid(False)

        # Ensure white background
        fig.patch.set_facecolor('white')

        plt.tight_layout()
        
        if save_plot:
            salt_str = '_'.join(salts_to_plot)
            r_range_str = ''
            if min_r is not None or max_r is not None:
                r_min_str = f"{min_r}" if min_r is not None else "0"
                r_max_str = f"{max_r}" if max_r is not None else "max"
                r_range_str = f'_r{r_min_str}to{r_max_str}'
            filename = f'salt_comparison_rdf_3d_{ion_type}_{pair_type}_{concentration}_{salt_str}{r_range_str}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"3D plot saved: {filename}")
        
        plt.show()
        
        return comparison_data


    def compare_rdfs_3d_multi_concentration(self, ion_type, pair_type='ion_water',
                                        concentrations=None, salts=None,
                                        save_plot=True, save_log=True,
                                        title_fontsize=14, label_fontsize=12, tick_fontsize=10,
                                        shell_boundaries=None, ion_pairing_boundaries=None,
                                        alpha=0.35, elevation=20, azimuth=45,
                                        fill_under_curve=True, fill_alpha=0.15, 
                                        line_width=2.0, show_grid=True, grid_alpha=0.2,
                                        min_r=None, max_r=None):
        '''
        Create 3D visualization of RDFs across different salts AND concentrations.
        Uses line styles to differentiate concentrations.
        
        Parameters
        ----------
        ion_type : str
            Ion to compare (e.g., 'Na', 'K', 'Mg', 'Ca', 'Cl')
        pair_type : str
            Type of RDF: 'ion_water', 'ion_ion', or 'water_water'
        concentrations : list of str, optional
            List of concentrations to compare. If None, uses all available.
        salts : list of str, optional
            List of salt keys to compare. If None, uses all available.
        save_plot : bool
            Whether to save the plot
        save_log : bool
            Whether to save analysis log
        title_fontsize : int
            Font size for title
        label_fontsize : int
            Font size for axis labels
        tick_fontsize : int
            Font size for tick labels
        shell_boundaries : dict, optional
            Shell boundaries for shading
        ion_pairing_boundaries : dict, optional
            Ion pairing boundaries for shading
        alpha : float
            Transparency for shaded regions
        elevation : float
            Viewing elevation angle (degrees)
        azimuth : float
            Viewing azimuth angle (degrees)
        fill_under_curve : bool, default=True
            Whether to fill area under RDF curves
        fill_alpha : float, default=0.15
            Transparency for curve fill (lower for multi-conc)
        line_width : float, default=2.0
            Width of RDF curve lines
        show_grid : bool, default=True
            Whether to show grid lines
        grid_alpha : float, default=0.2
            Transparency of grid lines
        min_r : float, optional
            Minimum r(Å) value to plot. If None, starts from 0.
        max_r : float, optional
            Maximum r(Å) value to plot. If None, uses full range from data.
        
        Returns
        -------
        comparison_data : dict
            Dictionary with RDF data: {salt_key: {conc_key: rdf_data}}
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_salts() first.")
            return None
        
        # Determine which salts to analyze
        if salts is None:
            salts_to_compare = list(self.loaded_data.keys())
        else:
            salts_to_compare = []
            for salt in salts:
                if salt in self.loaded_data:
                    salts_to_compare.append(salt)
                else:
                    print(f"WARNING: Salt '{salt}' not found in loaded data. Skipping.")
        
        if not salts_to_compare:
            print("ERROR: No valid salts to compare.")
            return None
        
        # Determine concentrations
        if concentrations is None:
            all_concs = set()
            for salt_key in salts_to_compare:
                all_concs.update(self.loaded_data[salt_key].keys())
            concentrations = sorted(list(all_concs), key=lambda x: float(x.replace('M', '')))
        
        # Get comparison data using existing multi-concentration method
        comparison_data = self.compare_rdfs_multi_concentration(
            ion_type=ion_type, pair_type=pair_type, concentrations=concentrations,
            salts=salts_to_compare, save_plot=False, save_log=save_log,
            shell_boundaries=shell_boundaries, ion_pairing_boundaries=ion_pairing_boundaries
        )
        
        if not comparison_data:
            return None
        
        # Create 3D plot
        fig = plt.figure(figsize=(16, 12), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        
        # Define line styles for different concentrations
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5))]
        
        # Map salts to X positions (same as single concentration method)
        x_positions = np.arange(len(salts_to_compare))
        
        print(f"Creating 3D plot with {len(salts_to_compare)} salts and {len(concentrations)} concentrations...")
        if min_r is not None or max_r is not None:
            r_min_str = f"{min_r}" if min_r is not None else "0"
            r_max_str = f"{max_r}" if max_r is not None else "max"
            print(f"Limiting r range to {r_min_str}-{r_max_str} Å")
        
        # Plot each salt-concentration combination
        for salt_key in salts_to_compare:
            if salt_key not in comparison_data:
                continue
            
            salt_label = self.salt_data[salt_key]['label']
            salt_color = self.salt_data[salt_key].get('color', None)
            salt_x_pos = x_positions[list(salts_to_compare).index(salt_key)]
            
            for i, conc_key in enumerate(sorted(comparison_data[salt_key].keys(),
                                            key=lambda x: float(x.replace('M', '')))):
                if conc_key not in comparison_data[salt_key]:
                    continue
                
                rdf_info = comparison_data[salt_key][conc_key]
                r = rdf_info['r']
                g_r = rdf_info['g_r']
                
                # Apply r range filters if specified
                mask = np.ones(len(r), dtype=bool)
                if min_r is not None:
                    mask &= (r >= min_r)
                if max_r is not None:
                    mask &= (r <= max_r)
                
                r = r[mask]
                g_r = g_r[mask]
                
                # Create X values (all same for this salt)
                x = np.full_like(r, salt_x_pos)
                
                # Get line style for this concentration
                line_style = line_styles[i % len(line_styles)]
                
                # Create label showing both salt and concentration
                label = f"{salt_label} {conc_key}"
                
                # Plot the 3D line - SAME AXES AS SINGLE CONCENTRATION: (x=salt, y=r, z=g_r)
                ax.plot(x, r, g_r, label=label, linewidth=line_width, 
                    color=salt_color, linestyle=line_style, alpha=0.8)
                
                # Fill under curve if requested
                if fill_under_curve:
                    # Create surface from baseline (z=0) to curve
                    x_fill = np.vstack([x, x])  # 2 x n_points
                    y_fill = np.vstack([r, r])  # 2 x n_points (r values)
                    z_fill = np.vstack([np.zeros_like(g_r), g_r])  # From 0 to g_r values
                    
                    ax.plot_surface(x_fill, y_fill, z_fill, 
                                color=salt_color, alpha=fill_alpha, linewidth=0, shade=True)
        
        # Set Y-axis limits based on min_r and max_r
        y_min = min_r if min_r is not None else 0
        if max_r is not None:
            ax.set_ylim(y_min, max_r)
        else:
            ax.set_ylim(y_min, ax.get_ylim()[1])
        
        # Get z-axis limits for boundary planes
        z_min, z_max = ax.get_zlim()
        
        # Add boundary planes if provided (same as single concentration method)
        if pair_type == 'ion_water' and shell_boundaries is not None:
            shell_items = [(k, v) for k, v in shell_boundaries.items() if k.startswith('shell_')]
            shell_items.sort(key=lambda x: x[1] if isinstance(x[1], tuple) else 
                            (x[1]['r_min'] if isinstance(x[1], dict) and 'r_min' in x[1] else 0))
            
            # Get blue colors
            import matplotlib.colors as mcolors
            base_rgb = mcolors.hex2color('#00c5ff')
            
            for i, (shell_name, bounds) in enumerate(shell_items):
                if isinstance(bounds, dict):
                    r_min_shell, r_max_shell = bounds['r_min'], bounds['r_max']
                else:
                    r_min_shell, r_max_shell = bounds
                
                # Only plot boundaries within the specified r range
                plot_r_min = min_r if min_r is not None else 0
                plot_r_max = max_r if max_r is not None else float('inf')
                
                if (r_min_shell <= plot_r_max and r_max_shell >= plot_r_min):
                    # Create vertical planes at shell boundaries
                    x_plane = np.linspace(-0.5, len(salts_to_compare) - 0.5, 10)
                    z_plane = np.linspace(z_min, z_max, 10)
                    X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
                    
                    # Add vertical planes at r_min and r_max
                    Y_min = np.full_like(X_plane, r_min_shell)
                    Y_max = np.full_like(X_plane, r_max_shell)
                    
                    # Uncomment to show boundary planes
                    # ax.plot_surface(X_plane, Y_min, Z_plane, alpha=0.1, color=base_rgb)
                    # ax.plot_surface(X_plane, Y_max, Z_plane, alpha=0.1, color=base_rgb)
        
        elif pair_type == 'ion_ion' and ion_pairing_boundaries is not None:
            region_colors = {'CIP': '#FFB3B3', 'SIP': '#B3D9FF', 'DSIP': '#B3FFB3', 'FI': '#E6E6E6'}
            
            for region_name in ['CIP', 'SIP', 'DSIP']:
                if region_name in ion_pairing_boundaries:
                    bounds = ion_pairing_boundaries[region_name]
                    if isinstance(bounds, dict):
                        r_min_region = bounds.get('r_min', bounds.get('min', 0))
                        r_max_region = bounds.get('r_max', bounds.get('max', 10))
                    else:
                        r_min_region, r_max_region = bounds
                    
                    # Only plot boundaries within the specified r range
                    plot_r_min = min_r if min_r is not None else 0
                    plot_r_max = max_r if max_r is not None else float('inf')
                    
                    if (r_min_region <= plot_r_max and r_max_region >= plot_r_min):
                        # Add vertical planes
                        x_plane = np.linspace(-0.5, len(salts_to_compare) - 0.5, 10)
                        z_plane = np.linspace(z_min, z_max, 10)
                        X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
                        
                        color = region_colors[region_name]
                        Y_min = np.full_like(X_plane, r_min_region)
                        Y_max = np.full_like(X_plane, r_max_region)
                        
                        # Uncomment to show boundary planes
                        # ax.plot_surface(X_plane, Y_min, Z_plane, alpha=0.1, color=color)
                        # ax.plot_surface(X_plane, Y_max, Z_plane, alpha=0.1, color=color)
        
        # Customize 3D plot - SAME AXES AS SINGLE CONCENTRATION
        ax.set_xlabel('Salt Type', fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel('r (Å)', fontsize=label_fontsize, labelpad=10)
        ax.set_zlabel('g(r)', fontsize=label_fontsize, labelpad=10)
        
        # Set X-axis labels to salt names
        ax.set_xticks(x_positions)
        ax.set_xticklabels([self.salt_data[salt]['label'] for salt in salts_to_compare])
        
        # Title with r range info
        conc_str = f"{len(concentrations)} concentrations" if len(concentrations) > 3 else f"{', '.join(concentrations)}"
        title_base = f'3D RDF Comparison: {ion_type} ({pair_type})\nMultiple Concentrations: {conc_str}'
        if min_r is not None or max_r is not None:
            r_min_str = f"{min_r}" if min_r is not None else "0"
            r_max_str = f"{max_r}" if max_r is not None else "max"
            title_base += f' (r: {r_min_str}-{r_max_str} Å)'
        ax.set_title(title_base, fontsize=title_fontsize, fontweight='bold', pad=20)
        
        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Add legend (group by concentration for clarity)
        if len(salts_to_compare) * len(concentrations) <= 16:
            # Create custom legend with concentration info
            from matplotlib.lines import Line2D
            legend_elements = []
            
            # Add concentration legend
            for i, conc in enumerate(sorted(concentrations, key=lambda x: float(x.replace('M', '')))):
                line_style = line_styles[i % len(line_styles)]
                legend_elements.append(Line2D([0], [0], color='gray', linestyle=line_style, 
                                            linewidth=2, label=f'Concentration: {conc}'))
            
            # Add separator
            legend_elements.append(Line2D([0], [0], color='white', linewidth=0, label=''))
            
            # Add salt color legend
            for salt_key in salts_to_compare:
                salt_label = self.salt_data[salt_key]['label']
                salt_color = self.salt_data[salt_key].get('color', 'black')
                legend_elements.append(Line2D([0], [0], color=salt_color, linewidth=2, 
                                            label=f'Salt: {salt_label}'))
            
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)
        else:
            print(f"Legend suppressed (too many combinations: {len(salts_to_compare)} × {len(concentrations)})")
            print("Line styles represent concentrations (in order):")
            for i, conc in enumerate(sorted(concentrations, key=lambda x: float(x.replace('M', '')))):
                style_name = ['-', '--', '-.', ':'][i % 4]
                print(f"  {conc}: {style_name}")
        
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
        
        # Ensure white background
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        if save_plot:
            conc_str = '_'.join(concentrations) if len(concentrations) <= 3 else f"multi_{len(concentrations)}concs"
            salt_str = '_'.join(salts_to_compare) if len(salts_to_compare) <= 3 else f"multi_{len(salts_to_compare)}salts"
            r_range_str = ''
            if min_r is not None or max_r is not None:
                r_min_str = f"{min_r}" if min_r is not None else "0"
                r_max_str = f"{max_r}" if max_r is not None else "max"
                r_range_str = f'_r{r_min_str}to{r_max_str}'
            filename = f'salt_comparison_rdf_3d_multi_{ion_type}_{pair_type}_{conc_str}_{salt_str}{r_range_str}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"3D multi-concentration plot saved: {filename}")
        
        plt.show()
        
        return comparison_data

    def compare_shell_coordination_histogram(self, ion_type, shell='shell_1', 
                                            salt_pairs=None, concentrations=None,
                                            save_plots=True, save_log=True,
                                            figsize=(12, 8), alpha=0.7,
                                            title_fontsize=14, label_fontsize=12, 
                                            legend_fontsize=10, tick_fontsize=10,
                                            x_axis_type='concentration',  # 'concentration' or 'chloride_count'
                                            chloride_count_groups=None,
                                            bar_width=0.35, capsize=5,
                                            y_min=None, y_max=None,
                                            shells=None):  # NEW PARAMETER for multiple shells
        '''
        Create histogram (bar chart) plots comparing coordination numbers across concentrations between salt pairs.
        Shows discrete concentration groups on x-axis with side-by-side bars for each salt.
        When multiple shells are specified, plots all in single plot using color variations.
        
        Parameters
        ----------
        ion_type : str
            Ion to compare (e.g., 'Na', 'K', 'Mg', 'Ca', 'Cl')
        shell : str, default='shell_1'
            Shell to analyze ('shell_1', 'shell_2', 'shell_3'). Used when shells=None.
        salt_pairs : list of tuples, optional
            Pairs of salts to compare, e.g., [('NaCl', 'KCl'), ('CaCl2', 'MgCl2')]
            If None, will compare monovalent salts and divalent salts separately
        concentrations : list of str, optional
            Concentrations to include. If None, uses all available.
        save_plots : bool
            Whether to save the plots
        save_log : bool
            Whether to save analysis log
        figsize : tuple
            Figure size (width, height)
        alpha : float
            Transparency for histogram bars
        title_fontsize : int
            Font size for titles
        label_fontsize : int
            Font size for axis labels
        legend_fontsize : int
            Font size for legend
        tick_fontsize : int
            Font size for tick labels
        x_axis_type : str, default='concentration'
            X-axis type: 'concentration' (M) or 'chloride_count' (number of Cl- ions)
        chloride_count_groups : dict, optional
            Custom mapping of chloride counts to salt concentrations.
        bar_width : float, default=0.35
            Width of histogram bars
        capsize : float, default=5
            Size of error bar caps
        y_min : float, optional
            Minimum y-axis value. If None, uses automatic scaling.
        y_max : float, optional
            Maximum y-axis value. If None, uses automatic scaling with padding.
        shells : list of str, optional
            List of shells to compare (e.g., ['shell_1', 'shell_2', 'shell_3']).
            If provided, plots all shells in single plot using color variations for each salt.
            If None, uses single shell specified by 'shell' parameter.
        
        Returns
        -------
        comparison_data : dict
            Dictionary with coordination number data including means and standard deviations
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_salts() first.")
            return None
        
        # Determine which shells to analyze
        if shells is not None:
            shells_to_analyze = shells
            multi_shell = True
        else:
            shells_to_analyze = [shell]
            multi_shell = False
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        if multi_shell:
            log_lines.append(f"COORDINATION NUMBER HISTOGRAM COMPARISON: {ion_type} (Multiple Shells - Single Plot)")
            log_lines.append(f"Shells: {', '.join(shells_to_analyze)}")
        else:
            log_lines.append(f"COORDINATION NUMBER HISTOGRAM COMPARISON: {ion_type} ({shell})")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"X-axis Type: {x_axis_type}")
        
        if y_min is not None:
            log_lines.append(f"Custom Y-axis minimum: {y_min}")
        else:
            log_lines.append("Y-axis minimum: Automatic scaling")
        
        if y_max is not None:
            log_lines.append(f"Custom Y-axis maximum: {y_max}")
        else:
            log_lines.append("Y-axis maximum: Automatic scaling with padding")
        log_lines.append("")
        
        # Determine salt pairs if not provided
        if salt_pairs is None:
            available_salts = list(self.loaded_data.keys())
            
            # Function to normalize salt labels for comparison
            def normalize_salt_label(label):
                '''Normalize salt label to handle different naming conventions'''
                normalized = label.upper()
                # Replace subscript/superscript characters with regular numbers
                normalized = normalized.replace('₂', '2').replace('²', '2')
                normalized = normalized.replace('₃', '3').replace('³', '3')
                normalized = normalized.replace('₄', '4').replace('⁴', '4')
                # Remove spaces and common separators
                normalized = normalized.replace(' ', '').replace('-', '').replace('_', '')
                return normalized
            
            # Separate monovalent and divalent based on labels
            monovalent = []
            divalent = []
            
            for salt_key in available_salts:
                label_normalized = normalize_salt_label(self.salt_data[salt_key]['label'])
                if ('CL2' in label_normalized or 'CACL2' in label_normalized or 
                    'MGCL2' in label_normalized or 'BACL2' in label_normalized or
                    'SRCL2' in label_normalized):
                    divalent.append(salt_key)
                else:
                    monovalent.append(salt_key)
            
            salt_pairs = []
            if len(monovalent) >= 2:
                salt_pairs.append((monovalent[0], monovalent[1]))  # e.g., NaCl vs KCl
            if len(divalent) >= 2:
                salt_pairs.append((divalent[0], divalent[1]))      # e.g., CaCl2 vs MgCl2
        
        log_lines.append(f"Salt pairs to compare:")
        for pair in salt_pairs:
            salt1_label = self.salt_data[pair[0]]['label']
            salt2_label = self.salt_data[pair[1]]['label']
            log_lines.append(f"  {salt1_label} vs {salt2_label}")
        log_lines.append("")
        
        # Determine concentrations
        if concentrations is None:
            if chloride_count_groups is not None:
                # Get all unique concentrations from the custom mapping
                all_concs = set()
                for cl_group, salt_concs in chloride_count_groups.items():
                    all_concs.update(salt_concs.values())
                concentrations = sorted(list(all_concs), key=lambda x: float(x.replace('M', '')))
            else:
                # Get all unique concentrations from loaded data
                all_concs = set()
                for salt_key in [salt for pair in salt_pairs for salt in pair]:
                    if salt_key in self.loaded_data:
                        all_concs.update(self.loaded_data[salt_key].keys())
                concentrations = sorted(list(all_concs), key=lambda x: float(x.replace('M', '')))
        
        log_lines.append(f"Concentrations: {', '.join(concentrations)}")
        log_lines.append("")
        
        # Function to normalize salt labels for comparison
        def normalize_salt_label(label):
            '''Normalize salt label to handle different naming conventions'''
            normalized = label.upper()
            normalized = normalized.replace('₂', '2').replace('²', '2')
            normalized = normalized.replace('₃', '3').replace('³', '3')
            normalized = normalized.replace('₄', '4').replace('⁴', '4')
            normalized = normalized.replace(' ', '').replace('-', '').replace('_', '')
            return normalized
        
        # Function to convert concentration to chloride count
        def get_chloride_count(salt_key, conc_str):
            '''Convert salt concentration to chloride ion count'''
            
            if chloride_count_groups is not None:
                # Use custom mapping - find which chloride group this salt-concentration belongs to
                salt_label = self.salt_data[salt_key]['label']
                salt_label_normalized = normalize_salt_label(salt_label)
                
                for cl_group, salt_concs in chloride_count_groups.items():
                    for mapping_salt_name, mapping_conc in salt_concs.items():
                        mapping_salt_normalized = normalize_salt_label(mapping_salt_name)
                        
                        # Check if normalized labels match and concentrations match
                        if (salt_label_normalized == mapping_salt_normalized and 
                            mapping_conc == conc_str):
                            # Extract numeric value from chloride group name (e.g., '11_Cl' -> 11)
                            chloride_count = float(cl_group.replace('_Cl', ''))
                            return chloride_count
                
                # Fallback to automatic calculation if not found in custom mapping
                print(f"WARNING: {salt_label} ({salt_label_normalized}) {conc_str} not found in chloride_count_groups, using automatic calculation")
            
            # Automatic calculation
            conc_value = float(conc_str.replace('M', ''))
            salt_label_normalized = normalize_salt_label(self.salt_data[salt_key]['label'])
            
            # Check for divalent salts using normalized labels
            if ('CL2' in salt_label_normalized or 'CACL2' in salt_label_normalized or 
                'MGCL2' in salt_label_normalized or 'BACL2' in salt_label_normalized or
                'SRCL2' in salt_label_normalized):
                chloride_multiplier = 2
            else:
                chloride_multiplier = 1
            
            chloride_count = conc_value * chloride_multiplier * 100  # Scale to reasonable numbers
            return chloride_count
        
        # Function to generate shell-specific colors
        def get_shell_colors(base_color, n_shells):
            '''
            Generate different shades of base_color for each shell.
            Shell 1 = darker, Shell 2 = medium, Shell 3 = lighter
            '''
            import matplotlib.colors as mcolors
            
            # Convert base color to RGB if it's a named color
            if isinstance(base_color, str):
                if base_color.startswith('#'):
                    base_rgb = mcolors.hex2color(base_color)
                else:
                    base_rgb = mcolors.to_rgb(base_color)
            else:
                base_rgb = base_color
            
            # Convert to HSV for easier manipulation
            base_hsv = mcolors.rgb_to_hsv(base_rgb)
            base_hue, base_saturation, base_value = base_hsv[0], base_hsv[1], base_hsv[2]
            
            shell_colors = []
            
            if n_shells == 1:
                # Just return the base color
                shell_colors = [mcolors.to_hex(base_rgb)]
            elif n_shells == 2:
                # Darker and lighter versions
                darker_value = max(0.3, base_value * 0.7)  # 70% of brightness, min 0.3
                lighter_value = min(1.0, base_value * 1.3)  # 130% of brightness, max 1.0
                
                shell_colors = [
                    mcolors.to_hex(mcolors.hsv_to_rgb((base_hue, base_saturation, darker_value))),   # Shell 1: darker
                    mcolors.to_hex(mcolors.hsv_to_rgb((base_hue, base_saturation, lighter_value)))   # Shell 2: lighter
                ]
            elif n_shells == 3:
                # Darker, medium (original), lighter
                darker_value = max(0.2, base_value * 0.6)   # 60% of brightness, min 0.2
                medium_value = base_value                     # Original brightness
                lighter_value = min(1.0, base_value * 1.4)  # 140% of brightness, max 1.0
                
                shell_colors = [
                    mcolors.to_hex(mcolors.hsv_to_rgb((base_hue, base_saturation, darker_value))),   # Shell 1: darker
                    mcolors.to_hex(mcolors.hsv_to_rgb((base_hue, base_saturation, medium_value))),   # Shell 2: medium
                    mcolors.to_hex(mcolors.hsv_to_rgb((base_hue, base_saturation, lighter_value)))   # Shell 3: lighter
                ]
            else:
                # For more than 3 shells, create a gradient
                for i in range(n_shells):
                    # Create gradient from darker (shell 1) to lighter (shell n)
                    value_factor = 0.4 + (0.8 * i / (n_shells - 1))  # Range from 0.4 to 1.2
                    new_value = max(0.1, min(1.0, base_value * value_factor))
                    shell_colors.append(mcolors.to_hex(mcolors.hsv_to_rgb((base_hue, base_saturation, new_value))))
            
            return shell_colors
        
        # Collect coordination number data for all relevant salts and shells
        comparison_data = {}
        
        log_lines.append("COORDINATION NUMBER DATA")
        log_lines.append("-"*80)
        if x_axis_type == 'chloride_count':
            log_lines.append(f"{'Salt':<15} {'Shell':<10} {'Conc (M)':<10} {'Cl- Count':<12} {'Mean CN':<10} {'Std CN':<10}")
        else:
            log_lines.append(f"{'Salt':<15} {'Shell':<10} {'Conc (M)':<10} {'Mean CN':<10} {'Std CN':<10}")
        log_lines.append("-"*80)
        
        for salt_key in [salt for pair in salt_pairs for salt in pair]:
            if salt_key not in self.loaded_data:
                continue
                
            comparison_data[salt_key] = {}
            
            for shell_name in shells_to_analyze:
                comparison_data[salt_key][shell_name] = {}
                
                for conc_key in concentrations:
                    if conc_key not in self.loaded_data[salt_key]:
                        continue
                    
                    conc_data = self.loaded_data[salt_key][conc_key]
                    
                    # Get coordination number data with statistics
                    cn_mean = None
                    cn_std = None
                    
                    if 'shell_coordination_numbers' in conc_data:
                        cn_dict = conc_data['shell_coordination_numbers']
                        
                        # Try exact ion name and variations
                        for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
                            if ion_variant in cn_dict:
                                cn_data = cn_dict[ion_variant]
                                if 'shells' in cn_data and shell_name in cn_data['shells']:
                                    shell_data = cn_data['shells'][shell_name]
                                    cn_mean = shell_data.get('mean_cn', shell_data.get('mean', None))
                                    cn_std = shell_data.get('std_cn', shell_data.get('std', 0.0))
                                    break
                    
                    if cn_mean is not None:
                        # Store concentration, chloride count, mean, and std
                        chloride_count = get_chloride_count(salt_key, conc_key)
                        
                        comparison_data[salt_key][shell_name][conc_key] = {
                            'cn_mean': cn_mean,
                            'cn_std': cn_std if cn_std is not None else 0.0,
                            'concentration': float(conc_key.replace('M', '')),
                            'chloride_count': chloride_count
                        }
                        
                        # Log output
                        if x_axis_type == 'chloride_count':
                            log_lines.append(f"{self.salt_data[salt_key]['label']:<15} {shell_name:<10} {conc_key:<10} {chloride_count:<12.0f} {cn_mean:<10.2f} {cn_std:<10.2f}")
                        else:
                            log_lines.append(f"{self.salt_data[salt_key]['label']:<15} {shell_name:<10} {conc_key:<10} {cn_mean:<10.2f} {cn_std:<10.2f}")
        
        log_lines.append("")
        
        # Create histogram plots for each salt pair
        for i, (salt1_key, salt2_key) in enumerate(salt_pairs):
            
            if salt1_key not in comparison_data or salt2_key not in comparison_data:
                log_lines.append(f"WARNING: Skipping {salt1_key} vs {salt2_key} - missing data")
                continue
            
            salt1_label = self.salt_data[salt1_key]['label']
            salt2_label = self.salt_data[salt2_key]['label']
            salt1_base_color = self.salt_data[salt1_key].get('color', 'blue')
            salt2_base_color = self.salt_data[salt2_key].get('color', 'red')
            
            # Generate shell-specific colors for each salt
            n_shells = len(shells_to_analyze)
            salt1_shell_colors = get_shell_colors(salt1_base_color, n_shells)
            salt2_shell_colors = get_shell_colors(salt2_base_color, n_shells)
            
            log_lines.append("SHELL COLOR MAPPING")
            log_lines.append("-"*40)
            log_lines.append(f"{salt1_label} colors:")
            for j, shell_name in enumerate(shells_to_analyze):
                log_lines.append(f"  {shell_name}: {salt1_shell_colors[j]}")
            log_lines.append(f"{salt2_label} colors:")
            for j, shell_name in enumerate(shells_to_analyze):
                log_lines.append(f"  {shell_name}: {salt2_shell_colors[j]}")
            log_lines.append("")
            
            # Create single figure for all shells
            fig, ax = plt.subplots(figsize=figsize, facecolor='white')
            
            # Find common concentrations across all shells for both salts
            common_concentrations = []
            for conc_key in concentrations:
                # Check if concentration exists for both salts across all shells
                exists_in_all = True
                for shell_name in shells_to_analyze:
                    if (conc_key not in comparison_data[salt1_key].get(shell_name, {}) or 
                        conc_key not in comparison_data[salt2_key].get(shell_name, {})):
                        exists_in_all = False
                        break
                if exists_in_all:
                    common_concentrations.append(conc_key)
            
            if not common_concentrations:
                print(f"No common concentrations found for {salt1_label} vs {salt2_label} across all shells")
                continue
            
            # Sort concentrations properly
            common_concentrations.sort(key=lambda x: float(x.replace('M', '')))
            
            # Prepare data for grouped bars
            n_groups = len(common_concentrations)
            x_pos = np.arange(n_groups)
            
            # FIXED BAR POSITIONING: Group by shell, then by salt within each shell
            # Total bars per concentration group: n_shells × 2_salts
            total_bars_per_group = n_shells * 2  
            
            # Adjust bar width to fit all bars
            effective_bar_width = bar_width / n_shells
            
            # Create x-axis labels
            x_labels = []
            for conc_key in common_concentrations:
                # Use first salt's first shell data for creating labels (should be same for all)
                first_shell = shells_to_analyze[0]
                data_point = comparison_data[salt1_key][first_shell][conc_key]
                
                if x_axis_type == 'chloride_count':
                    if chloride_count_groups is not None:
                        cl_count = data_point['chloride_count']
                        x_labels.append(f"{int(cl_count)} Cl⁻")
                    else:
                        cl_count = data_point['chloride_count']
                        x_labels.append(f"{int(cl_count)}")
                else:
                    x_labels.append(conc_key)

            # CORRECTED BAR POSITIONING: Group by shell first
            for shell_idx, shell_name in enumerate(shells_to_analyze):
                
                # Collect data for this shell
                salt1_cn_means = []
                salt1_cn_stds = []
                salt2_cn_means = []
                salt2_cn_stds = []
                
                for conc_key in common_concentrations:
                    # Salt 1 data for this shell
                    salt1_data = comparison_data[salt1_key][shell_name][conc_key]
                    salt1_cn_means.append(salt1_data['cn_mean'])
                    salt1_cn_stds.append(salt1_data['cn_std'])
                    
                    # Salt 2 data for this shell
                    salt2_data = comparison_data[salt2_key][shell_name][conc_key]
                    salt2_cn_means.append(salt2_data['cn_mean'])
                    salt2_cn_stds.append(salt2_data['cn_std'])
                
                # Convert to numpy arrays
                salt1_cn_means = np.array(salt1_cn_means)
                salt1_cn_stds = np.array(salt1_cn_stds)
                salt2_cn_means = np.array(salt2_cn_means)
                salt2_cn_stds = np.array(salt2_cn_stds)
                
                # FIXED: Calculate x positions for this shell's bars
                # Total width available for all shells
                total_shell_width = bar_width  # Use 100% - NO GAPS between shell groups

                # Width allocated to each shell
                single_shell_width = total_shell_width / n_shells

                # Width for each salt within a shell - MAKE THEM TOUCH
                salt_bar_width = single_shell_width / 2  # Each salt gets exactly half of shell width

                # Calculate the center position for this shell group
                shell_start = -total_shell_width / 2  # Start from left edge
                shell_center = shell_start + (shell_idx + 0.5) * single_shell_width

                # Position salts within the shell region - NO GAPS BETWEEN SALTS
                salt1_offset = shell_center - salt_bar_width  # Left half
                salt2_offset = shell_center                   # Right half (touching salt1)

                x1 = x_pos + salt1_offset
                x2 = x_pos + salt2_offset
                
                # Get colors for this shell
                salt1_color = salt1_shell_colors[shell_idx]
                salt2_color = salt2_shell_colors[shell_idx]
                
                # Create labels for legend
                salt1_label_text = f"{salt1_label} {shell_name.replace('_', ' ').title()}"
                salt2_label_text = f"{salt2_label} {shell_name.replace('_', ' ').title()}"
                
                # Plot bars with the corrected width
                bars1 = ax.bar(x1, salt1_cn_means, salt_bar_width, alpha=alpha, 
                            color=salt1_color, label=salt1_label_text)
                bars2 = ax.bar(x2, salt2_cn_means, salt_bar_width, alpha=alpha, 
                            color=salt2_color, label=salt2_label_text)
                
                # Add error bars
                ax.errorbar(x1, salt1_cn_means, yerr=salt1_cn_stds,
                        fmt='none', color='black', capsize=capsize, capthick=1.5, 
                        elinewidth=1.5, alpha=0.8)
                
                ax.errorbar(x2, salt2_cn_means, yerr=salt2_cn_stds,
                        fmt='none', color='black', capsize=capsize, capthick=1.5, 
                        elinewidth=1.5, alpha=0.8)

            # Log the positioning for debugging
            log_lines.append("BAR POSITIONING DEBUG")
            log_lines.append("-"*40)
            log_lines.append(f"Number of shells: {n_shells}")
            log_lines.append(f"Total shell width: {total_shell_width:.3f}")
            log_lines.append(f"Single shell width: {single_shell_width:.3f}")
            log_lines.append(f"Salt bar width: {salt_bar_width:.3f}")
            for shell_idx, shell_name in enumerate(shells_to_analyze):
                shell_start = -total_shell_width / 2
                shell_center = shell_start + (shell_idx + 0.5) * single_shell_width
                salt1_offset = shell_center - salt_bar_width
                salt2_offset = shell_center
                log_lines.append(f"  {shell_name}: Center={shell_center:.3f}, Salt1={salt1_offset:.3f}, Salt2={salt2_offset:.3f}")
                log_lines.append(f"    Salt1 range: [{salt1_offset:.3f}, {salt1_offset + salt_bar_width:.3f}]")
                log_lines.append(f"    Salt2 range: [{salt2_offset:.3f}, {salt2_offset + salt_bar_width:.3f}]")
            log_lines.append("")
            
            # Customize plot
            if x_axis_type == 'chloride_count':
                ax.set_xlabel('Chloride Ion Count (Cl⁻)', fontsize=label_fontsize)
                title_suffix = '(vs Cl⁻ count)'
            else:
                ax.set_xlabel('Concentration (M)', fontsize=label_fontsize)
                title_suffix = '(vs concentration)'
            
            ax.set_ylabel('Coordination Number', fontsize=label_fontsize)
            
            if multi_shell:
                shells_str = ', '.join([s.replace('_', ' ').title() for s in shells_to_analyze])
                ax.set_title(f'Coordination Number Histogram: {ion_type} {title_suffix}\n'
                            f'{salt1_label} vs {salt2_label} - {shells_str}',
                            fontsize=title_fontsize, fontweight='bold')
            else:
                ax.set_title(f'Coordination Number Histogram: {ion_type} ({shell}) {title_suffix}\n'
                            f'{salt1_label} vs {salt2_label}',
                            fontsize=title_fontsize, fontweight='bold')
            
            # Set x-axis ticks and labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            
            # Add legend
            ax.legend(fontsize=legend_fontsize, frameon=False, loc='best')
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            
            # Apply y-axis limits
            if y_min is not None or y_max is not None:
                current_y_min, current_y_max = ax.get_ylim()
                
                y_min_actual = y_min if y_min is not None else max(0, current_y_min - 0.1)
                y_max_actual = y_max if y_max is not None else current_y_max + 0.1
                
                ax.set_ylim(y_min_actual, y_max_actual)
            
            # Add padding around bars
            ax.set_xlim(-0.5, n_groups - 0.5)
            
            plt.tight_layout()
            
            # Save plot
            if save_plots:
                custom_suffix = '_custom' if chloride_count_groups is not None else ''
                y_range_suffix = ''
                if y_min is not None or y_max is not None:
                    y_min_str = f"{y_min}" if y_min is not None else "auto"
                    y_max_str = f"{y_max}" if y_max is not None else "auto"
                    y_range_suffix = f'_y{y_min_str}to{y_max_str}'
                
                if multi_shell:
                    shells_str = '_'.join(shells_to_analyze)
                    filename = f'salt_histogram_comparison_{ion_type}_{shells_str}_{salt1_key}_vs_{salt2_key}_{x_axis_type}_single{custom_suffix}{y_range_suffix}.png'
                else:
                    filename = f'salt_histogram_comparison_{ion_type}_{shell}_{salt1_key}_vs_{salt2_key}_{x_axis_type}{custom_suffix}{y_range_suffix}.png'
                
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                log_lines.append(f"HISTOGRAM PLOT SAVED: {filename}")
                print(f"Histogram comparison plot saved: {filename}")
            
            plt.show()
        
        # Save log
        if save_log:
            custom_suffix = '_custom' if chloride_count_groups is not None else ''
            y_range_suffix = ''
            if y_min is not None or y_max is not None:
                y_min_str = f"{y_min}" if y_min is not None else "auto"
                y_max_str = f"{y_max}" if y_max is not None else "auto"
                y_range_suffix = f'_y{y_min_str}to{y_max_str}'
            
            if multi_shell:
                shells_str = '_'.join(shells_to_analyze)
                log_filename = f'salt_histogram_comparison_{ion_type}_{shells_str}_{x_axis_type}_single{custom_suffix}{y_range_suffix}.log'
            else:
                log_filename = f'salt_histogram_comparison_{ion_type}_{shell}_{x_axis_type}{custom_suffix}{y_range_suffix}.log'
            
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Log saved: {log_filename}")
        
        return comparison_data


    def compare_coordination_environment_probabilities(self, ion_type, shell='shell_1',
                                                    concentration='0.11M', salt_pairs=None,
                                                    save_plots=True, save_log=True,
                                                    figsize=(14, 8), alpha=0.7,
                                                    title_fontsize=14, label_fontsize=12,
                                                    legend_fontsize=10, tick_fontsize=10,
                                                    bar_width=0.35, group_spacing=0.8):
        '''
        Compare coordination environment probabilities for a single concentration across different salts.
        Shows discrete coordination environments (0-6, 0-5, 0-4, etc.) on x-axis with probability percentages on y-axis.
        FIXED: Now uses proper shell coordination probabilities from Solute speciation data.
        
        Parameters
        ----------
        ion_type : str
            Ion to analyze (e.g., 'Na', 'K', 'Mg', 'Ca', 'Cl')
        shell : str, default='shell_1'
            Shell to analyze ('shell_1', 'shell_2', 'shell_3')
        concentration : str, default='0.11M'
            Concentration to analyze (e.g., '0.11M', '0.2M')
        salt_pairs : list of tuples, optional
            Pairs of salts to compare, e.g., [('NaCl', 'KCl')]
            If None, will compare first available monovalent pair
        save_plots : bool
            Whether to save the plots
        save_log : bool
            Whether to save analysis log
        figsize : tuple
            Figure size (width, height)
        alpha : float
            Transparency for bars
        title_fontsize : int
            Font size for titles
        label_fontsize : int
            Font size for axis labels
        legend_fontsize : int
            Font size for legend
        tick_fontsize : int
            Font size for tick labels
        bar_width : float, default=0.35
            Width of individual bars
        group_spacing : float, default=0.8
            Spacing between coordination environment groups
        
        Returns
        -------
        comparison_data : dict
            Dictionary with coordination environment probability data
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_salts() first.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"COORDINATION ENVIRONMENT PROBABILITY COMPARISON: {ion_type} ({shell})")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Shell: {shell}")
        log_lines.append(f"Concentration: {concentration}")
        log_lines.append("")
        
        # Determine salt pairs if not provided
        if salt_pairs is None:
            available_salts = list(self.loaded_data.keys())
            excluded_salts = {'No_Salt', 'Pure_Water', 'Water', 'no_salt', 'pure_water'}
            available_salts = [s for s in available_salts if s not in excluded_salts]
            
            # Function to normalize salt labels
            def normalize_salt_label(label):
                normalized = label.upper()
                normalized = normalized.replace('₂', '2').replace('²', '2')
                normalized = normalized.replace('₃', '3').replace('³', '3')
                normalized = normalized.replace('₄', '4').replace('⁴', '4')
                normalized = normalized.replace(' ', '').replace('-', '').replace('_', '')
                return normalized
            
            # Separate monovalent and divalent
            monovalent = []
            divalent = []
            
            for salt_key in available_salts:
                label_normalized = normalize_salt_label(self.salt_data[salt_key]['label'])
                if ('CL2' in label_normalized or 'CACL2' in label_normalized or 
                    'MGCL2' in label_normalized):
                    divalent.append(salt_key)
                else:
                    monovalent.append(salt_key)
            
            # Use first available monovalent pair
            if len(monovalent) >= 2:
                salt_pairs = [(monovalent[0], monovalent[1])]
            elif len(divalent) >= 2:
                salt_pairs = [(divalent[0], divalent[1])]
            else:
                print("ERROR: Need at least 2 salts for comparison")
                return None
        
        log_lines.append(f"Salt pairs to compare:")
        for pair in salt_pairs:
            salt1_label = self.salt_data[pair[0]]['label']
            salt2_label = self.salt_data[pair[1]]['label']
            log_lines.append(f"  {salt1_label} vs {salt2_label}")
        log_lines.append("")
        
        # Collect coordination environment probability data
        comparison_data = {}
        
        log_lines.append("COORDINATION ENVIRONMENT PROBABILITY DATA")
        log_lines.append("-"*80)
        log_lines.append(f"{'Salt':<15} {'Environment':<15} {'Probability (%)':<15}")
        log_lines.append("-"*80)
        
        for salt_key in [salt for pair in salt_pairs for salt in pair]:
            if salt_key not in self.loaded_data:
                continue
            
            if concentration not in self.loaded_data[salt_key]:
                print(f"WARNING: {concentration} not found for {self.salt_data[salt_key]['label']}")
                continue
            
            conc_data = self.loaded_data[salt_key][concentration]
            comparison_data[salt_key] = {}
            
            # FIXED: Look for shell coordination probabilities data (from Solute speciation)
            env_probs = None
            
            # Method 1: Direct shell probabilities by ion type
            if 'shell_probabilities_by_ion_type' in conc_data:
                shell_prob_data = conc_data['shell_probabilities_by_ion_type']
                
                # Try to find the specific ion type
                for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
                    if ion_variant in shell_prob_data:
                        ion_data = shell_prob_data[ion_variant]
                        
                        if 'data' in ion_data and hasattr(ion_data['data'], 'iterrows'):
                            # This is a DataFrame with shell types and fractions
                            df = ion_data['data']
                            env_probs = {}
                            
                            for _, row in df.iterrows():
                                shell_type = row['shell']
                                fraction = row['fraction']
                                probability = fraction * 100  # Convert to percentage
                                
                                # Extract coordination environment (e.g., "0-6" from shell type)
                                env_probs[shell_type] = probability
                            
                            break
            
            # Method 2: Look for shell_coordination_probabilities
            if env_probs is None and 'shell_coordination_probabilities' in conc_data:
                shell_prob_data = conc_data['shell_coordination_probabilities']
                
                for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
                    if ion_variant in shell_prob_data:
                        ion_prob_data = shell_prob_data[ion_variant]
                        
                        if shell in ion_prob_data and isinstance(ion_prob_data[shell], dict):
                            shell_data = ion_prob_data[shell]
                            env_probs = {}
                            
                            # Extract environment probabilities
                            for env_name, prob in shell_data.items():
                                if isinstance(prob, (int, float)):
                                    env_probs[env_name] = prob * 100  # Convert to percentage
                            break
            
                        # Method 3: Fallback - try to extract from shell_region_coordination_probabilities
                        if env_probs is None and 'shell_region_coordination_probabilities' in conc_data:
                            region_prob_data = conc_data['shell_region_coordination_probabilities']
                            
                            for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
                                if ion_variant in region_prob_data:
                                    ion_data = region_prob_data[ion_variant]
                                    
                                    if 'shell_regions' in ion_data and shell in ion_data['shell_regions']:
                                        shell_data = ion_data['shell_regions'][shell]
                                        
                                        if 'coordination_environments' in shell_data:
                                            coord_envs = shell_data['coordination_environments']
                                            env_probs = {}
                                            
                                            for env_name, env_data in coord_envs.items():
                                                # FIXED: Handle different data types
                                                if isinstance(env_data, dict):
                                                    # env_data is a dictionary
                                                    if 'probability' in env_data:
                                                        probability = env_data['probability'] * 100  # Convert to percentage
                                                        env_probs[env_name] = probability
                                                    elif 'prob' in env_data:
                                                        probability = env_data['prob'] * 100
                                                        env_probs[env_name] = probability
                                                elif isinstance(env_data, (int, float)):
                                                    # env_data is directly the probability value
                                                    probability = float(env_data) * 100  # Convert to percentage
                                                    env_probs[env_name] = probability
                                                else:
                                                    print(f"WARNING: Unknown env_data type for {env_name}: {type(env_data)}")
                                                    continue
                                    break
                        # Method 4: Try to extract directly from shell_region_coordination_probabilities structure
                        if env_probs is None and 'shell_region_coordination_probabilities' in conc_data:
                            region_prob_data = conc_data['shell_region_coordination_probabilities']
                            
                            for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
                                if ion_variant in region_prob_data:
                                    ion_data = region_prob_data[ion_variant]
                                    env_probs = {}
                                    
                                    # Try different possible structures
                                    if isinstance(ion_data, dict):
                                        # Look for direct environment probabilities at ion level
                                        for key, value in ion_data.items():
                                            if isinstance(value, (int, float)) and 0 <= value <= 1:
                                                # This looks like a probability value
                                                env_probs[key] = value * 100  # Convert to percentage
                                            elif isinstance(value, dict) and shell in value:
                                                # This might be shell-specific data
                                                shell_data = value[shell]
                                                if isinstance(shell_data, dict):
                                                    for env_name, env_prob in shell_data.items():
                                                        if isinstance(env_prob, (int, float)) and 0 <= env_prob <= 1:
                                                            env_probs[env_name] = env_prob * 100
                                                elif isinstance(shell_data, (int, float)) and 0 <= shell_data <= 1:
                                                    env_probs[key] = shell_data * 100
                                    
                                    if env_probs:  # If we found some probabilities
                                        break
            
            # Store the data if found
            if env_probs:
                comparison_data[salt_key] = env_probs
                
                # Log the data
                salt_label = self.salt_data[salt_key]['label']
                for env_name, prob in sorted(env_probs.items()):
                    log_lines.append(f"{salt_label:<15} {env_name:<15} {prob:<15.2f}")
            else:
                print(f"WARNING: No coordination environment data found for {self.salt_data[salt_key]['label']} {ion_type}")
                comparison_data[salt_key] = {}
        
        log_lines.append("")
        
        # Check if we have valid data to plot
        valid_salts = [salt_key for salt_key in comparison_data.keys() if comparison_data[salt_key]]
        
        if len(valid_salts) < 2:
            print("ERROR: Need at least 2 salts with valid coordination environment data")
            print(f"Available data: {[self.salt_data[k]['label'] for k in valid_salts]}")
            return comparison_data
        
        # Create comparison plot for each salt pair
        for i, (salt1_key, salt2_key) in enumerate(salt_pairs):
            
            if salt1_key not in valid_salts or salt2_key not in valid_salts:
                log_lines.append(f"WARNING: Skipping {salt1_key} vs {salt2_key} - missing data")
                continue
            
            salt1_label = self.salt_data[salt1_key]['label']
            salt2_label = self.salt_data[salt2_key]['label']
            salt1_color = self.salt_data[salt1_key].get('color', 'blue')
            salt2_color = self.salt_data[salt2_key].get('color', 'red')
            
            # Get all unique coordination environments from both salts
            all_environments = set()
            all_environments.update(comparison_data[salt1_key].keys())
            all_environments.update(comparison_data[salt2_key].keys())
            
            # Sort environments naturally (try to extract numeric parts for sorting)
            def sort_env_key(env_name):
                # Handle formats like "0-6", "1-5", "Shell_1", etc.
                import re
                
                # Look for patterns like "number-number"
                match = re.search(r'(\d+)-(\d+)', env_name)
                if match:
                    coion_count = int(match.group(1))
                    water_count = int(match.group(2))
                    return (coion_count, water_count)
                
                # Look for patterns like "Shell_number"
                match = re.search(r'Shell_(\d+)', env_name)
                if match:
                    return (0, int(match.group(1)))
                
                # Fallback to string sorting
                return (999, 999, env_name)
            
            sorted_environments = sorted(all_environments, key=sort_env_key)
            
            if not sorted_environments:
                print(f"No coordination environments found for {salt1_label} vs {salt2_label}")
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize, facecolor='white')
            
            # Prepare data for plotting
            n_envs = len(sorted_environments)
            x_positions = np.arange(n_envs) * group_spacing  # Spacing between environment groups
            
            salt1_probs = []
            salt2_probs = []
            x_labels = []
            
            for env_name in sorted_environments:
                # Get probabilities for each salt (0 if environment not present)
                salt1_prob = comparison_data[salt1_key].get(env_name, 0.0)
                salt2_prob = comparison_data[salt2_key].get(env_name, 0.0)
                
                salt1_probs.append(salt1_prob)
                salt2_probs.append(salt2_prob)
                
                # Clean up environment name for x-axis label
                x_labels.append(env_name)
            
            # Convert to numpy arrays
            salt1_probs = np.array(salt1_probs)
            salt2_probs = np.array(salt2_probs)
            
            # Calculate bar positions (side by side within each environment group)
            width = bar_width
            x1 = x_positions - width/2  # Salt 1 positions (left)
            x2 = x_positions + width/2  # Salt 2 positions (right)
            
            # Plot bars
            bars1 = ax.bar(x1, salt1_probs, width, alpha=alpha, color=salt1_color, 
                        label=salt1_label)
            bars2 = ax.bar(x2, salt2_probs, width, alpha=alpha, color=salt2_color, 
                        label=salt2_label)
            
            # Customize plot
            ax.set_xlabel('Coordination Environment', fontsize=label_fontsize)
            ax.set_ylabel('Probability (%)', fontsize=label_fontsize)
            ax.set_title(f'Coordination Environment Probabilities: {ion_type} ({shell})\n'
                        f'{salt1_label} vs {salt2_label} at {concentration}',
                        fontsize=title_fontsize, fontweight='bold')
            
            # Set x-axis ticks and labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            # Add legend and styling
            ax.legend(fontsize=legend_fontsize, frameon=False, loc='best')
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            
            # Set y-axis to show percentages from 0 to 100 (or max value + padding)
            max_prob = max(np.max(salt1_probs), np.max(salt2_probs))
            if max_prob > 0:
                ax.set_ylim(0, min(100, max_prob * 1.1))
            else:
                ax.set_ylim(0, 100)
            
            # Add value labels on top of bars (optional)
            def add_value_labels(bars, values):
                for bar, value in zip(bars, values):
                    if value > 0.1:  # Only show labels for non-zero values
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max_prob*0.01,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=tick_fontsize-2)
            
            add_value_labels(bars1, salt1_probs)
            add_value_labels(bars2, salt2_probs)
            
            plt.tight_layout()
            
            # Save plot
            if save_plots:
                filename = f'coord_env_comparison_{ion_type}_{shell}_{salt1_key}_vs_{salt2_key}_{concentration.replace(".", "p")}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                log_lines.append(f"PLOT SAVED: {filename}")
                print(f"Coordination environment comparison plot saved: {filename}")
            
            plt.show()
            
            # Print verification - check if probabilities sum to ~100%
            print(f"\nProbability verification for {ion_type} at {concentration}:")
            print(f"  {salt1_label}: Total = {sum(salt1_probs):.1f}% (should be ~100%)")
            print(f"  {salt2_label}: Total = {sum(salt2_probs):.1f}% (should be ~100%)")
            
            if abs(sum(salt1_probs) - 100) > 5:
                print(f"  ⚠️  WARNING: {salt1_label} probabilities don't sum to 100%")
            if abs(sum(salt2_probs) - 100) > 5:
                print(f"  ⚠️  WARNING: {salt2_label} probabilities don't sum to 100%")
        
        # Save log
        if save_log:
            log_filename = f'coord_env_comparison_{ion_type}_{shell}_{concentration.replace(".", "p")}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Log saved: {log_filename}")
        
        return comparison_data

    def compare_coordination_numbers(self, ion_type, shell='shell_1',
                                    concentrations=None,
                                    plot_type='line',  # 'line' or 'histogram'
                                    x_axis_type='concentration',  # 'concentration' or 'chloride_count'
                                    chloride_count_groups=None,  # NEW: Custom chloride mapping
                                    save_plot=True, save_log=True,
                                    xlabel_fontsize=12, ylabel_fontsize=12,
                                    title_fontsize=14, legend_fontsize=10,
                                    tick_fontsize=10,
                                    # Histogram-specific parameters
                                    bins=20, alpha_hist=0.7, 
                                    hist_type='bar',  # 'bar', 'step', 'stepfilled'
                                    density=False, cumulative=False):
        '''
        Compare coordination numbers across salts and concentrations.
        
        Parameters
        ----------
        ion_type : str
            Ion to compare (e.g., 'Na', 'K', 'Mg', 'Ca')
        shell : str
            Shell to analyze ('shell_1', 'shell_2', 'shell_3')
        concentrations : list of str, optional
            List of concentrations to compare. If None, uses all available.
        plot_type : str, default='line'
            Type of plot: 'line' for line plot vs concentration, 'histogram' for distribution
        x_axis_type : str, default='concentration'
            X-axis type for line plots: 'concentration' (M) or 'chloride_count' (number of Cl- ions)
        chloride_count_groups : dict, optional
            Custom mapping of chloride counts to salt concentrations.
            Format: {
                '11_Cl': {'NaCl': '0.11M', 'KCl': '0.11M', 'CaCl2': '0.06M', 'MgCl2': '0.06M'},
                '22_Cl': {'NaCl': '0.2M', 'KCl': '0.2M', 'CaCl2': '0.11M', 'MgCl2': '0.11M'},
                ...
            }
            If provided, overrides automatic chloride count calculation.
        save_plot : bool
            Whether to save the plot
        save_log : bool
            Whether to save analysis log
        xlabel_fontsize : int
            Font size for x-axis label
        ylabel_fontsize : int
            Font size for y-axis label
        title_fontsize : int
            Font size for title
        legend_fontsize : int
            Font size for legend
        tick_fontsize : int
            Font size for tick labels
        bins : int or sequence, default=20
            Number of histogram bins (for histogram plot)
        alpha_hist : float, default=0.7
            Transparency for histogram bars
        hist_type : str, default='bar'
            Histogram type: 'bar', 'step', or 'stepfilled'
        density : bool, default=False
            If True, normalize histogram to form probability density
        cumulative : bool, default=False
            If True, create cumulative histogram
        
        Returns
        -------
        comparison_data : dict
            Dictionary with coordination number data
        '''
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_salts() first.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"COORDINATION NUMBER COMPARISON: {ion_type} ({shell})")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Shell: {shell}")
        log_lines.append(f"Plot Type: {plot_type}")
        log_lines.append(f"X-axis Type: {x_axis_type}")
        
        if chloride_count_groups is not None:
            log_lines.append("Using CUSTOM chloride count mapping:")
            for cl_group, salt_concs in chloride_count_groups.items():
                log_lines.append(f"  {cl_group}: {salt_concs}")
        else:
            log_lines.append("Using AUTOMATIC chloride count calculation")
        log_lines.append("")
        
        # Determine concentrations to analyze
        if concentrations is None:
            if chloride_count_groups is not None:
                # Get all unique concentrations from the custom mapping
                all_concs = set()
                for cl_group, salt_concs in chloride_count_groups.items():
                    all_concs.update(salt_concs.values())
                concentrations = sorted(list(all_concs), key=lambda x: float(x.replace('M', '')))
            else:
                # Get all unique concentrations from loaded data
                all_concs = set()
                for salt_key in self.loaded_data.keys():
                    all_concs.update(self.loaded_data[salt_key].keys())
                concentrations = sorted(list(all_concs), key=lambda x: float(x.replace('M', '')))
        
        log_lines.append(f"Concentrations: {', '.join(concentrations)}")
        log_lines.append("")
        
        # Function to normalize salt labels for comparison
        def normalize_salt_label(label):
            '''Normalize salt label to handle different naming conventions'''
            normalized = label.upper()
            # Replace subscript/superscript characters with regular numbers
            normalized = normalized.replace('₂', '2').replace('²', '2')
            normalized = normalized.replace('₃', '3').replace('³', '3')
            normalized = normalized.replace('₄', '4').replace('⁴', '4')
            # Remove spaces and common separators
            normalized = normalized.replace(' ', '').replace('-', '').replace('_', '')
            return normalized
        
        # Function to convert concentration to chloride count
        def get_chloride_count(salt_key, conc_str):
            '''Convert salt concentration to chloride ion count'''
            
            if chloride_count_groups is not None:
                # Use custom mapping - find which chloride group this salt-concentration belongs to
                salt_label = self.salt_data[salt_key]['label']
                salt_label_normalized = normalize_salt_label(salt_label)
                
                for cl_group, salt_concs in chloride_count_groups.items():
                    for mapping_salt_name, mapping_conc in salt_concs.items():
                        mapping_salt_normalized = normalize_salt_label(mapping_salt_name)
                        
                        # Check if normalized labels match and concentrations match
                        if (salt_label_normalized == mapping_salt_normalized and 
                            mapping_conc == conc_str):
                            # Extract numeric value from chloride group name (e.g., '11_Cl' -> 11)
                            chloride_count = float(cl_group.replace('_Cl', ''))
                            log_lines.append(f"  Custom mapping: {salt_label} ({salt_label_normalized}) → {mapping_salt_name} ({mapping_salt_normalized}) → {chloride_count} Cl⁻")
                            return chloride_count
                
                # Fallback to automatic calculation if not found in custom mapping
                print(f"WARNING: {salt_label} ({salt_label_normalized}) {conc_str} not found in chloride_count_groups, using automatic calculation")
            
            # Automatic calculation (original method)
            conc_value = float(conc_str.replace('M', ''))
            salt_label_normalized = normalize_salt_label(self.salt_data[salt_key]['label'])
            
            # Check for divalent salts using normalized labels
            if ('CL2' in salt_label_normalized or 
                'CACL2' in salt_label_normalized or 
                'MGCL2' in salt_label_normalized or
                'BACL2' in salt_label_normalized or
                'SRCL2' in salt_label_normalized or
                'FECL2' in salt_label_normalized or
                'COCL2' in salt_label_normalized or
                'NICL2' in salt_label_normalized or
                'ZNCL2' in salt_label_normalized):
                chloride_multiplier = 2
            else:
                chloride_multiplier = 1
            
            chloride_count = conc_value * chloride_multiplier * 100  # Scale to reasonable numbers
            return chloride_count
        
        # Collect coordination number data
        comparison_data = {}
        all_cn_values = {}  # For histogram: {salt_key: [list of all CN values]}
        
        log_lines.append("COORDINATION NUMBER DATA")
        log_lines.append("-"*80)
        if x_axis_type == 'chloride_count':
            log_lines.append(f"{'Salt':<15} {'Conc (M)':<10} {'Cl- Count':<12} {'CN':<8}")
        else:
            log_lines.append(f"{'Salt':<15} {'Conc (M)':<10} {'CN':<8}")
        log_lines.append("-"*80)
        
        for salt_key in self.loaded_data.keys():
            comparison_data[salt_key] = {}
            all_cn_values[salt_key] = []
            
            for conc_key in concentrations:
                if conc_key not in self.loaded_data[salt_key]:
                    continue
                
                conc_data = self.loaded_data[salt_key][conc_key]
                
                # Get coordination number data
                cn = None
                cn_distribution = None
                
                if 'shell_coordination_numbers' in conc_data:
                    cn_dict = conc_data['shell_coordination_numbers']
                    
                    # Try exact ion name and variations
                    for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
                        if ion_variant in cn_dict:
                            cn_data = cn_dict[ion_variant]
                            if 'shells' in cn_data and shell in cn_data['shells']:
                                shell_data = cn_data['shells'][shell]
                                cn = shell_data.get('mean_cn', shell_data.get('mean', None))
                                
                                # Try to get individual CN values for histogram
                                if plot_type == 'histogram':
                                    cn_distribution = shell_data.get('cn_values', 
                                                    shell_data.get('coordination_numbers', 
                                                    shell_data.get('values', None)))
                                break
                
                if cn is not None:
                    # Store both concentration and chloride count
                    chloride_count = get_chloride_count(salt_key, conc_key)
                    
                    comparison_data[salt_key][conc_key] = {
                        'cn': cn,
                        'concentration': float(conc_key.replace('M', '')),
                        'chloride_count': chloride_count
                    }
                    
                    # Log output
                    if x_axis_type == 'chloride_count':
                        log_lines.append(f"{self.salt_data[salt_key]['label']:<15} {conc_key:<10} {chloride_count:<12.0f} {cn:.2f}")
                    else:
                        log_lines.append(f"{self.salt_data[salt_key]['label']:<15} {conc_key:<10} {cn:.2f}")
                    
                    # Collect CN values for histogram
                    if plot_type == 'histogram' and cn_distribution is not None:
                        if isinstance(cn_distribution, (list, np.ndarray)):
                            all_cn_values[salt_key].extend(list(cn_distribution))
                        else:
                            # If we only have mean, create synthetic distribution (for demo)
                            synthetic_values = np.random.normal(cn, cn*0.1, 100)  # 10% std
                            all_cn_values[salt_key].extend(list(synthetic_values))
        
        log_lines.append("")
        
        # Create plots based on plot_type
        if plot_type == 'line':
            # Line plot functionality with x-axis choice
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for salt_key in sorted(comparison_data.keys()):
                if not comparison_data[salt_key]:
                    continue
                
                salt_label = self.salt_data[salt_key]['label']
                color = self.salt_data[salt_key].get('color', None)
                
                x_values = []
                cn_values = []
                
                for conc_key in sorted(comparison_data[salt_key].keys(), 
                                    key=lambda x: float(x.replace('M', ''))):
                    data_point = comparison_data[salt_key][conc_key]
                    
                    if x_axis_type == 'chloride_count':
                        x_values.append(data_point['chloride_count'])
                    else:  # concentration
                        x_values.append(data_point['concentration'])
                    
                    cn_values.append(data_point['cn'])
                
                ax.plot(x_values, cn_values, marker='o', markersize=8,
                    label=salt_label, linewidth=2, color=color)
            
            # Customize line plot
            if x_axis_type == 'chloride_count':
                if chloride_count_groups is not None:
                    ax.set_xlabel('Chloride Ion Count (Cl⁻)', fontsize=xlabel_fontsize)
                    title_suffix = '(vs Cl⁻ count - custom mapping)'
                    
                    # Add custom x-axis ticks and labels if using custom mapping
                    unique_cl_counts = sorted(set(float(cl.replace('_Cl', '')) for cl in chloride_count_groups.keys()))
                    ax.set_xticks(unique_cl_counts)
                    ax.set_xticklabels([f"{int(cl)}" for cl in unique_cl_counts])
                else:
                    ax.set_xlabel('Chloride Ion Count', fontsize=xlabel_fontsize)
                    title_suffix = '(vs Cl⁻ count - auto)'
            else:
                ax.set_xlabel('Concentration (M)', fontsize=xlabel_fontsize)
                title_suffix = '(vs concentration)'
            
            ax.set_ylabel('Coordination Number', fontsize=ylabel_fontsize)
            ax.set_title(f'Coordination Number: {ion_type} ({shell}) {title_suffix}',
                        fontsize=title_fontsize, fontweight='bold')
            ax.legend(fontsize=legend_fontsize, frameon=False)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        elif plot_type == 'histogram':
            # Histogram functionality (unchanged)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare histogram data
            hist_data = []
            hist_labels = []
            hist_colors = []
            
            for salt_key in sorted(all_cn_values.keys()):
                if len(all_cn_values[salt_key]) > 0:
                    hist_data.append(all_cn_values[salt_key])
                    hist_labels.append(self.salt_data[salt_key]['label'])
                    hist_colors.append(self.salt_data[salt_key].get('color', None))
            
            if hist_data:
                # Create histogram
                if hist_type == 'bar':
                    ax.hist(hist_data, bins=bins, alpha=alpha_hist, label=hist_labels,
                        color=hist_colors, density=density, cumulative=cumulative)
                elif hist_type == 'step':
                    for i, data in enumerate(hist_data):
                        ax.hist(data, bins=bins, alpha=alpha_hist, label=hist_labels[i],
                            color=hist_colors[i], histtype='step', linewidth=2,
                            density=density, cumulative=cumulative)
                elif hist_type == 'stepfilled':
                    for i, data in enumerate(hist_data):
                        ax.hist(data, bins=bins, alpha=alpha_hist, label=hist_labels[i],
                            color=hist_colors[i], histtype='stepfilled',
                            density=density, cumulative=cumulative)
                
                # Add statistics to log
                log_lines.append("HISTOGRAM STATISTICS")
                log_lines.append("-"*80)
                log_lines.append(f"{'Salt':<15} {'Mean CN':<15} {'Std CN':<15} {'N Values':<15}")
                log_lines.append("-"*80)
                
                for i, salt_key in enumerate(sorted(all_cn_values.keys())):
                    if len(all_cn_values[salt_key]) > 0:
                        values = np.array(all_cn_values[salt_key])
                        mean_cn = np.mean(values)
                        std_cn = np.std(values)
                        n_values = len(values)
                        
                        log_lines.append(f"{hist_labels[i]:<15} {mean_cn:<15.3f} {std_cn:<15.3f} {n_values:<15}")
            
            # Customize histogram plot
            ax.set_xlabel('Coordination Number', fontsize=xlabel_fontsize)
            
            if density:
                if cumulative:
                    ax.set_ylabel('Cumulative Probability Density', fontsize=ylabel_fontsize)
                else:
                    ax.set_ylabel('Probability Density', fontsize=ylabel_fontsize)
            else:
                if cumulative:
                    ax.set_ylabel('Cumulative Count', fontsize=ylabel_fontsize)
                else:
                    ax.set_ylabel('Count', fontsize=ylabel_fontsize)
            
            title_parts = [f'Coordination Number Distribution: {ion_type} ({shell})']
            if density:
                title_parts.append('(Normalized)')
            if cumulative:
                title_parts.append('(Cumulative)')
            
            ax.set_title(' '.join(title_parts), fontsize=title_fontsize, fontweight='bold')
            ax.legend(fontsize=legend_fontsize, frameon=False)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'line' or 'histogram'")
        
        plt.tight_layout()
        
        # Save files
        if save_plot:
            custom_suffix = '_custom' if chloride_count_groups is not None else ''
            filename = f'salt_comparison_cn_{ion_type}_{shell}_{plot_type}_{x_axis_type}{custom_suffix}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            log_lines.append(f"PLOT SAVED: {filename}")
            print(f"Plot saved: {filename}")
        
        if save_log:
            custom_suffix = '_custom' if chloride_count_groups is not None else ''
            log_filename = f'salt_comparison_cn_{ion_type}_{shell}_{plot_type}_{x_axis_type}{custom_suffix}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Log saved: {log_filename}")
        
        plt.show()
        
        return comparison_data

    
    # def compare_residence_times(self, ion_type, residence_type='water',
    #                         shell='shell_1', salt_pairs=None, concentrations=None,
    #                         save_plots=True, save_log=True,
    #                         figsize=(12, 8), alpha=0.7,
    #                         title_fontsize=14, label_fontsize=12, 
    #                         legend_fontsize=10, tick_fontsize=10,
    #                         x_axis_type='concentration',  # 'concentration' or 'chloride_count'
    #                         chloride_count_groups=None,
    #                         bar_width=0.35, capsize=5,
    #                         y_min=None, y_max=None,
    #                         shells=None,  # Multiple shells support
    #                         plot_type='line'):  # 'line' or 'histogram'
    #     '''
    #     Compare residence times across salts and concentrations with enhanced functionality.
    #     Shows discrete concentration groups on x-axis with side-by-side bars for each salt.
    #     When multiple shells are specified, plots all in single plot using color variations.
        
    #     Parameters
    #     ----------
    #     ion_type : str
    #         Ion to compare (e.g., 'Na', 'K', 'Mg', 'Ca', 'Cl')
    #     residence_type : str, default='water'
    #         Type of residence time: 'water' or 'ion_pairing'
    #     shell : str, default='shell_1'
    #         Shell to analyze ('shell_1', 'shell_2', 'shell_3'). Used when shells=None.
    #     salt_pairs : list of tuples, optional
    #         Pairs of salts to compare, e.g., [('NaCl', 'KCl'), ('CaCl2', 'MgCl2')]
    #         If None, will compare monovalent salts and divalent salts separately
    #     concentrations : list of str, optional
    #         Concentrations to include. If None, uses all available.
    #     save_plots : bool
    #         Whether to save the plots
    #     save_log : bool
    #         Whether to save analysis log
    #     figsize : tuple
    #         Figure size (width, height)
    #     alpha : float
    #         Transparency for histogram bars
    #     title_fontsize : int
    #         Font size for titles
    #     label_fontsize : int
    #         Font size for axis labels
    #     legend_fontsize : int
    #         Font size for legend
    #     tick_fontsize : int
    #         Font size for tick labels
    #     x_axis_type : str, default='concentration'
    #         X-axis type: 'concentration' (M) or 'chloride_count' (number of Cl- ions)
    #     chloride_count_groups : dict, optional
    #         Custom mapping of chloride counts to salt concentrations.
    #     bar_width : float, default=0.35
    #         Width of histogram bars
    #     capsize : float, default=5
    #         Size of error bar caps
    #     y_min : float, optional
    #         Minimum y-axis value. If None, uses automatic scaling.
    #     y_max : float, optional
    #         Maximum y-axis value. If None, uses automatic scaling with padding.
    #     shells : list of str, optional
    #         List of shells to compare (e.g., ['shell_1', 'shell_2', 'shell_3']).
    #         If provided, plots all shells in single plot using color variations for each salt.
    #         If None, uses single shell specified by 'shell' parameter.
    #     plot_type : str, default='line'
    #         Type of plot: 'line' for line plot or 'histogram' for bar chart
        
    #     Returns
    #     -------
    #     comparison_data : dict
    #         Dictionary with residence time data including means and standard deviations
    #     '''
        
    #     if not self.loaded_data:
    #         print("ERROR: No data loaded. Run load_all_salts() first.")
    #         return None
        
    #     # Determine which shells/regions to analyze
    #     if shells is not None:
    #         regions_to_analyze = shells
    #         multi_region = True
    #     else:
    #         regions_to_analyze = [shell]
    #         multi_region = False
        
    #     # FIXED: Define consistent ion pairing colors
    #     if residence_type == 'ion_pairing':
    #         # Use correct ion pairing colors to match coordination analysis
    #         pairing_colors = {
    #             'CIP': 'lightcoral',      # Light red
    #             'SIP': 'lightblue',       # Light blue  
    #             'DSIP': 'lightgreen',     # Light green
    #             'FI': 'lightyellow'       # Light yellow
    #         }
    #     else:
    #         # Water residence - use shell-based colors
    #         pairing_colors = {
    #             'shell_1': '#1f77b4',    # Blue
    #             'shell_2': '#ff7f0e',    # Orange  
    #             'shell_3': '#2ca02c'     # Green
    #         }
        
    #     # Initialize log
    #     log_lines = []
    #     log_lines.append("="*80)
    #     if multi_region:
    #         log_lines.append(f"RESIDENCE TIME COMPARISON: {ion_type} ({residence_type}) - Multiple Regions")
    #         log_lines.append(f"Regions: {', '.join(regions_to_analyze)}")
    #     else:
    #         log_lines.append(f"RESIDENCE TIME COMPARISON: {ion_type} ({residence_type}) - {shell}")
    #     log_lines.append("="*80)
    #     log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    #     log_lines.append(f"Ion Type: {ion_type}")
    #     log_lines.append(f"Residence Type: {residence_type}")
    #     log_lines.append(f"Plot Type: {plot_type}")
    #     log_lines.append(f"X-axis Type: {x_axis_type}")
        
    #     if y_min is not None:
    #         log_lines.append(f"Custom Y-axis minimum: {y_min}")
    #     else:
    #         log_lines.append("Y-axis minimum: Automatic scaling")
        
    #     if y_max is not None:
    #         log_lines.append(f"Custom Y-axis maximum: {y_max}")
    #     else:
    #         log_lines.append("Y-axis maximum: Automatic scaling with padding")
    #     log_lines.append("")
        
    #     # Determine salt pairs if not provided
    #     if salt_pairs is None:
    #         available_salts = list(self.loaded_data.keys())
            
    #         # EXCLUDE No_Salt and other non-salt systems
    #         excluded_salts = {'No_Salt', 'Pure_Water', 'Water', 'no_salt', 'pure_water'}
    #         available_salts = [s for s in available_salts if s not in excluded_salts and 
    #                         self.salt_data[s]['label'] not in excluded_salts]
            
    #         print(f"Available salts for comparison: {[self.salt_data[s]['label'] for s in available_salts]}")
            
    #         # Function to normalize salt labels for comparison
    #         def normalize_salt_label(label):
    #             '''Normalize salt label to handle different naming conventions'''
    #             normalized = label.upper()
    #             # Replace subscript/superscript characters with regular numbers
    #             normalized = normalized.replace('₂', '2').replace('²', '2')
    #             normalized = normalized.replace('₃', '3').replace('³', '3')
    #             normalized = normalized.replace('₄', '4').replace('⁴', '4')
    #             # Remove spaces and common separators
    #             normalized = normalized.replace(' ', '').replace('-', '').replace('_', '')
    #             return normalized
            
    #         # Separate monovalent and divalent based on labels
    #         monovalent = []
    #         divalent = []
            
    #         for salt_key in available_salts:
    #             label_normalized = normalize_salt_label(self.salt_data[salt_key]['label'])
    #             if ('CL2' in label_normalized or 'CACL2' in label_normalized or 
    #                 'MGCL2' in label_normalized or 'BACL2' in label_normalized or
    #                 'SRCL2' in label_normalized):
    #                 divalent.append(salt_key)
    #             else:
    #                 monovalent.append(salt_key)
            
    #         print(f"Monovalent salts: {[self.salt_data[s]['label'] for s in monovalent]}")
    #         print(f"Divalent salts: {[self.salt_data[s]['label'] for s in divalent]}")
            
    #         salt_pairs = []
    #         if len(monovalent) >= 2:
    #             salt_pairs.append((monovalent[0], monovalent[1]))  # e.g., NaCl vs KCl
    #         if len(divalent) >= 2:
    #             salt_pairs.append((divalent[0], divalent[1]))      # e.g., CaCl2 vs MgCl2
            
    #         if not salt_pairs:
    #             print("ERROR: Not enough salt pairs found for comparison")
    #             return None
            
    #         print(f"Salt pairs to compare: {[(self.salt_data[p[0]]['label'], self.salt_data[p[1]]['label']) for p in salt_pairs]}")
        
    #     log_lines.append(f"Salt pairs to compare:")
    #     for pair in salt_pairs:
    #         salt1_label = self.salt_data[pair[0]]['label']
    #         salt2_label = self.salt_data[pair[1]]['label']
    #         log_lines.append(f"  {salt1_label} vs {salt2_label}")
    #     log_lines.append("")
        
    #     # Determine concentrations
    #     if concentrations is None:
    #         if chloride_count_groups is not None:
    #             # Get all unique concentrations from the custom mapping
    #             all_concs = set()
    #             for cl_group, salt_concs in chloride_count_groups.items():
    #                 all_concs.update(salt_concs.values())
    #             concentrations = sorted(list(all_concs), key=lambda x: float(x.replace('M', '')))
    #         else:
    #             # Get all unique concentrations from loaded data
    #             all_concs = set()
    #             for salt_key in [salt for pair in salt_pairs for salt in pair]:
    #                 if salt_key in self.loaded_data:
    #                     all_concs.update(self.loaded_data[salt_key].keys())
    #             concentrations = sorted(list(all_concs), key=lambda x: float(x.replace('M', '')))
        
    #     log_lines.append(f"Concentrations: {', '.join(concentrations)}")
    #     log_lines.append("")
        
    #     # Function to normalize salt labels for comparison
    #     def normalize_salt_label(label):
    #         '''Normalize salt label to handle different naming conventions'''
    #         normalized = label.upper()
    #         normalized = normalized.replace('₂', '2').replace('²', '2')
    #         normalized = normalized.replace('₃', '3').replace('³', '3')
    #         normalized = normalized.replace('₄', '4').replace('⁴', '4')
    #         normalized = normalized.replace(' ', '').replace('-', '').replace('_', '')
    #         return normalized
        
    #     # Function to convert concentration to chloride count
    #     def get_chloride_count(salt_key, conc_str):
    #         '''Convert salt concentration to chloride ion count'''
            
    #         if chloride_count_groups is not None:
    #             # Use custom mapping - find which chloride group this salt-concentration belongs to
    #             salt_label = self.salt_data[salt_key]['label']
    #             salt_label_normalized = normalize_salt_label(salt_label)
                
    #             for cl_group, salt_concs in chloride_count_groups.items():
    #                 for mapping_salt_name, mapping_conc in salt_concs.items():
    #                     mapping_salt_normalized = normalize_salt_label(mapping_salt_name)
                        
    #                     # Check if normalized labels match and concentrations match
    #                     if (salt_label_normalized == mapping_salt_normalized and 
    #                         mapping_conc == conc_str):
    #                         # Extract numeric value from chloride group name (e.g., '11_Cl' -> 11)
    #                         chloride_count = float(cl_group.replace('_Cl', ''))
    #                         return chloride_count
                
    #             # Fallback to automatic calculation if not found in custom mapping
    #             print(f"WARNING: {salt_label} ({salt_label_normalized}) {conc_str} not found in chloride_count_groups, using automatic calculation")
            
    #         # Automatic calculation
    #         conc_value = float(conc_str.replace('M', ''))
    #         salt_label_normalized = normalize_salt_label(self.salt_data[salt_key]['label'])
            
    #         # Check for divalent salts using normalized labels
    #         if ('CL2' in salt_label_normalized or 'CACL2' in salt_label_normalized or 
    #             'MGCL2' in salt_label_normalized or 'BACL2' in salt_label_normalized or
    #             'SRCL2' in salt_label_normalized):
    #             chloride_multiplier = 2
    #         else:
    #             chloride_multiplier = 1
            
    #         chloride_count = conc_value * chloride_multiplier * 100  # Scale to reasonable numbers
    #         return chloride_count
        
    #     # Function to generate region-specific colors for each salt
    #     def get_region_colors(base_color, n_regions):
    #         '''
    #         Generate different shades of base_color for each region.
    #         Region 1 = darker, Region 2 = medium, Region 3 = lighter
    #         '''
    #         import matplotlib.colors as mcolors
            
    #         # Convert base color to RGB if it's a named color
    #         if isinstance(base_color, str):
    #             if base_color.startswith('#'):
    #                 base_rgb = mcolors.hex2color(base_color)
    #             else:
    #                 base_rgb = mcolors.to_rgb(base_color)
    #         else:
    #             base_rgb = base_color
            
    #         # Convert to HSV for easier manipulation
    #         base_hsv = mcolors.rgb_to_hsv(base_rgb)
    #         base_hue, base_saturation, base_value = base_hsv[0], base_hsv[1], base_hsv[2]
            
    #         region_colors = []
            
    #         if n_regions == 1:
    #             # Just return the base color
    #             region_colors = [mcolors.to_hex(base_rgb)]
    #         elif n_regions == 2:
    #             # Darker and lighter versions
    #             darker_value = max(0.3, base_value * 0.7)  # 70% of brightness, min 0.3
    #             lighter_value = min(1.0, base_value * 1.3)  # 130% of brightness, max 1.0
                
    #             region_colors = [
    #                 mcolors.to_hex(mcolors.hsv_to_rgb((base_hue, base_saturation, darker_value))),   # Region 1: darker
    #                 mcolors.to_hex(mcolors.hsv_to_rgb((base_hue, base_saturation, lighter_value)))   # Region 2: lighter
    #             ]
    #         elif n_regions == 3:
    #             # Darker, medium (original), lighter
    #             darker_value = max(0.2, base_value * 0.6)   # 60% of brightness, min 0.2
    #             medium_value = base_value                     # Original brightness
    #             lighter_value = min(1.0, base_value * 1.4)  # 140% of brightness, max 1.0
                
    #             region_colors = [
    #                 mcolors.to_hex(mcolors.hsv_to_rgb((base_hue, base_saturation, darker_value))),   # Region 1: darker
    #                 mcolors.to_hex(mcolors.hsv_to_rgb((base_hue, base_saturation, medium_value))),   # Region 2: medium
    #                 mcolors.to_hex(mcolors.hsv_to_rgb((base_hue, base_saturation, lighter_value)))   # Region 3: lighter
    #             ]
    #         else:
    #             # For more than 3 regions, create a gradient
    #             for i in range(n_regions):
    #                 # Create gradient from darker (region 1) to lighter (region n)
    #                 value_factor = 0.4 + (0.8 * i / (n_regions - 1))  # Range from 0.4 to 1.2
    #                 new_value = max(0.1, min(1.0, base_value * value_factor))
    #                 region_colors.append(mcolors.to_hex(mcolors.hsv_to_rgb((base_hue, base_saturation, new_value))))
            
    #         return region_colors
        
    #     # Collect residence time data for all relevant salts and regions
    #     comparison_data = {}
        
    #     log_lines.append("RESIDENCE TIME DATA")
    #     log_lines.append("-"*80)
    #     if x_axis_type == 'chloride_count':
    #         log_lines.append(f"{'Salt':<15} {'Region':<10} {'Conc (M)':<10} {'Cl- Count':<12} {'Mean (ps)':<12} {'Std (ps)':<12}")
    #     else:
    #         log_lines.append(f"{'Salt':<15} {'Region':<10} {'Conc (M)':<10} {'Mean (ps)':<12} {'Std (ps)':<12}")
    #     log_lines.append("-"*80)

    #     # Handle ion pairing with region-specific analysis
    #     if residence_type == 'ion_pairing':
    #         # For ion pairing, regions refer to pairing regions (CIP, SIP, DSIP)
    #         pairing_regions_to_analyze = regions_to_analyze  # CIP, SIP, DSIP, etc.
            
    #         for salt_key in [salt for pair in salt_pairs for salt in pair]:
    #             if salt_key not in self.loaded_data:
    #                 continue
                    
    #             comparison_data[salt_key] = {}
                
    #             for region_name in pairing_regions_to_analyze:
    #                 comparison_data[salt_key][region_name] = {}
                    
    #                 for conc_key in concentrations:
    #                     if conc_key not in self.loaded_data[salt_key]:
    #                         continue
                        
    #                     conc_data = self.loaded_data[salt_key][conc_key]
                        
    #                     # Get ion pairing residence time data for specific region
    #                     res_mean = None
    #                     res_std = None
                        
    #                     if 'ion_pairing_residence_times' in conc_data:
    #                         res_dict = conc_data['ion_pairing_residence_times']
                            
    #                         # Try exact ion name and variations
    #                         for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
    #                             if ion_variant in res_dict:
    #                                 res_data = res_dict[ion_variant]
    #                                 if isinstance(res_data, dict) and 'regions' in res_data:
    #                                     regions = res_data['regions']
                                        
    #                                     # Get data for specific region
    #                                     if region_name in regions:
    #                                         region_data = regions[region_name]
    #                                         if isinstance(region_data, dict):
    #                                             # Look for mean residence time in various possible keys
    #                                             for mean_key in ['mean_residence_time', 'mean', 'avg_residence_time', 'average']:
    #                                                 if mean_key in region_data:
    #                                                     res_mean = region_data[mean_key]
    #                                                     break
                                                
    #                                             for std_key in ['std_residence_time', 'std', 'stdev', 'standard_deviation']:
    #                                                 if std_key in region_data:
    #                                                     res_std = region_data[std_key]
    #                                                     break
                                                
    #                                             # If std not found, set to 0
    #                                             if res_std is None:
    #                                                 res_std = 0.0
                                    
    #                                 if res_mean is not None:
    #                                     break
                        
    #                     if res_mean is not None:
    #                         # Store region-specific data
    #                         chloride_count = get_chloride_count(salt_key, conc_key)
                            
    #                         comparison_data[salt_key][region_name][conc_key] = {
    #                             'res_mean': res_mean,
    #                             'res_std': res_std if res_std is not None else 0.0,
    #                             'concentration': float(conc_key.replace('M', '')),
    #                             'chloride_count': chloride_count
    #                         }
                            
    #                         # Log output
    #                         if x_axis_type == 'chloride_count':
    #                             log_lines.append(f"{self.salt_data[salt_key]['label']:<15} {region_name:<10} {conc_key:<10} {chloride_count:<12.0f} {res_mean:<12.2f} {res_std:<12.2f}")
    #                         else:
    #                             log_lines.append(f"{self.salt_data[salt_key]['label']:<15} {region_name:<10} {conc_key:<10} {res_mean:<12.2f} {res_std:<12.2f}")

    #     else:  # water residence times
    #         for salt_key in [salt for pair in salt_pairs for salt in pair]:
    #             if salt_key not in self.loaded_data:
    #                 continue
                    
    #             comparison_data[salt_key] = {}
                
    #             for shell_name in regions_to_analyze:
    #                 comparison_data[salt_key][shell_name] = {}
                    
    #                 for conc_key in concentrations:
    #                     if conc_key not in self.loaded_data[salt_key]:
    #                         continue
                        
    #                     conc_data = self.loaded_data[salt_key][conc_key]
                        
    #                     # Get water residence time data with statistics
    #                     res_mean = None
    #                     res_std = None
                        
    #                     if 'water_residence_times' in conc_data:
    #                         res_dict = conc_data['water_residence_times']
                            
    #                         # Try exact ion name and variations
    #                         for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
    #                             if ion_variant in res_dict:
    #                                 res_data = res_dict[ion_variant]
    #                                 if 'shells' in res_data and shell_name in res_data['shells']:
    #                                     shell_data = res_data['shells'][shell_name]
    #                                     res_mean = shell_data.get('mean_residence_time', shell_data.get('mean', None))
    #                                     res_std = shell_data.get('std_residence_time', shell_data.get('std', 0.0))
    #                                     break
                        
    #                     if res_mean is not None:
    #                         # Store concentration, chloride count, mean, and std
    #                         chloride_count = get_chloride_count(salt_key, conc_key)
                            
    #                         comparison_data[salt_key][shell_name][conc_key] = {
    #                             'res_mean': res_mean,
    #                             'res_std': res_std if res_std is not None else 0.0,
    #                             'concentration': float(conc_key.replace('M', '')),
    #                             'chloride_count': chloride_count
    #                         }
                            
    #                         # Log output
    #                         if x_axis_type == 'chloride_count':
    #                             log_lines.append(f"{self.salt_data[salt_key]['label']:<15} {shell_name:<10} {conc_key:<10} {chloride_count:<12.0f} {res_mean:<12.2f} {res_std:<12.2f}")
    #                         else:
    #                             log_lines.append(f"{self.salt_data[salt_key]['label']:<15} {shell_name:<10} {conc_key:<10} {res_mean:<12.2f} {res_std:<12.2f}")

    #     log_lines.append("")
        
    #     # Create plots based on plot_type
    #     if plot_type == 'line':
    #         # Line plot for each salt pair
    #         for i, (salt1_key, salt2_key) in enumerate(salt_pairs):
                
    #             if salt1_key not in comparison_data or salt2_key not in comparison_data:
    #                 log_lines.append(f"WARNING: Skipping {salt1_key} vs {salt2_key} - missing data")
    #                 continue
                
    #             salt1_label = self.salt_data[salt1_key]['label']
    #             salt2_label = self.salt_data[salt2_key]['label']
    #             salt1_base_color = self.salt_data[salt1_key].get('color', 'blue')
    #             salt2_base_color = self.salt_data[salt2_key].get('color', 'red')
                
    #             # FIXED: Use appropriate colors based on residence type
    #             if residence_type == 'ion_pairing':
    #                 # Use fixed pairing colors for each region, distinguish salts by line style
    #                 salt1_region_colors = [pairing_colors.get(region, '#1f77b4') for region in regions_to_analyze]
    #                 salt2_region_colors = [pairing_colors.get(region, '#1f77b4') for region in regions_to_analyze]
    #             else:
    #                 # Generate shell-specific colors for each salt
    #                 n_regions = len(regions_to_analyze)
    #                 salt1_region_colors = get_region_colors(salt1_base_color, n_regions)
    #                 salt2_region_colors = get_region_colors(salt2_base_color, n_regions)
                
    #             # Create figure
    #             fig, ax = plt.subplots(figsize=figsize, facecolor='white')
                
    #             # Find common concentrations across all regions for both salts
    #             common_concentrations = []
    #             for conc_key in concentrations:
    #                 if residence_type == 'ion_pairing':
    #                     # For ion pairing: include if both salts have the concentration (even if not in all regions)
    #                     salt1_has_conc = any(conc_key in comparison_data[salt1_key].get(region, {}) 
    #                                     for region in regions_to_analyze)
    #                     salt2_has_conc = any(conc_key in comparison_data[salt2_key].get(region, {}) 
    #                                     for region in regions_to_analyze)
                        
    #                     if salt1_has_conc and salt2_has_conc:
    #                         common_concentrations.append(conc_key)
    #                 else:
    #                     # Water residence: require all shells
    #                     exists_in_all = True
    #                     for region_name in regions_to_analyze:
    #                         if (conc_key not in comparison_data[salt1_key].get(region_name, {}) or 
    #                             conc_key not in comparison_data[salt2_key].get(region_name, {})):
    #                             exists_in_all = False
    #                             break
    #                     if exists_in_all:
    #                         common_concentrations.append(conc_key)
                
    #             if not common_concentrations:
    #                 print(f"No common concentrations found for {salt1_label} vs {salt2_label}")
    #                 continue
                
    #             # Sort concentrations properly
    #             common_concentrations.sort(key=lambda x: float(x.replace('M', '')))
                
    #             # Plot each region as a separate line
    #             for region_idx, region_name in enumerate(regions_to_analyze):
                    
    #                 # Collect data for this region
    #                 salt1_x_values = []
    #                 salt1_res_means = []
    #                 salt1_res_stds = []
    #                 salt2_x_values = []
    #                 salt2_res_means = []
    #                 salt2_res_stds = []
                    
    #                 for conc_key in common_concentrations:
    #                     # Salt 1 data for this region
    #                     if conc_key in comparison_data[salt1_key].get(region_name, {}):
    #                         salt1_data = comparison_data[salt1_key][region_name][conc_key]
    #                         if x_axis_type == 'chloride_count':
    #                             salt1_x_values.append(salt1_data['chloride_count'])
    #                         else:
    #                             salt1_x_values.append(salt1_data['concentration'])
    #                         salt1_res_means.append(salt1_data['res_mean'])
    #                         salt1_res_stds.append(salt1_data['res_std'])
                        
    #                     # Salt 2 data for this region
    #                     if conc_key in comparison_data[salt2_key].get(region_name, {}):
    #                         salt2_data = comparison_data[salt2_key][region_name][conc_key]
    #                         if x_axis_type == 'chloride_count':
    #                             salt2_x_values.append(salt2_data['chloride_count'])
    #                         else:
    #                             salt2_x_values.append(salt2_data['concentration'])
    #                         salt2_res_means.append(salt2_data['res_mean'])
    #                         salt2_res_stds.append(salt2_data['res_std'])
                    
    #                 # Get colors for this region
    #                 salt1_color = salt1_region_colors[region_idx]
    #                 salt2_color = salt2_region_colors[region_idx]
                    
    #                 # FIXED: Create labels for legend with proper formatting
    #                 if residence_type == 'ion_pairing':
    #                     # Keep ion pairing regions in FULL CAPITALS
    #                     salt1_label_text = f"{salt1_label} {region_name}"
    #                     salt2_label_text = f"{salt2_label} {region_name}"
    #                 else:
    #                     # Water residence - convert shell names to title case
    #                     salt1_label_text = f"{salt1_label} {region_name.replace('_', ' ').title()}"
    #                     salt2_label_text = f"{salt2_label} {region_name.replace('_', ' ').title()}"
                    
    #                 # FIXED: Plot lines with distinct markers and line styles
    #                 if len(salt1_x_values) > 0:
    #                     ax.errorbar(salt1_x_values, salt1_res_means, yerr=salt1_res_stds,
    #                             marker='o', markersize=6, capsize=capsize, linewidth=2,
    #                             linestyle='-', color=salt1_color, label=salt1_label_text)
                    
    #                 if len(salt2_x_values) > 0:
    #                     ax.errorbar(salt2_x_values, salt2_res_means, yerr=salt2_res_stds,
    #                             marker='s', markersize=6, capsize=capsize, linewidth=2,
    #                             linestyle='--', color=salt2_color, label=salt2_label_text)  # Dashed line for salt 2
                
    #             # Customize plot
    #             if x_axis_type == 'chloride_count':
    #                 ax.set_xlabel('Chloride Ion Count (Cl⁻)', fontsize=label_fontsize)
    #                 title_suffix = '(vs Cl⁻ count)'
    #             else:
    #                 ax.set_xlabel('Concentration (M)', fontsize=label_fontsize)
    #                 title_suffix = '(vs concentration)'
                
    #             ax.set_ylabel('Mean Residence Time (ps)', fontsize=label_fontsize)
                
    #             if multi_region:
    #                 regions_str = ', '.join([s.replace('_', ' ').title() if residence_type != 'ion_pairing' else s for s in regions_to_analyze])
    #                 ax.set_title(f'Residence Time Comparison: {ion_type} ({residence_type}) {title_suffix}\n'
    #                             f'{salt1_label} vs {salt2_label} - {regions_str}',
    #                             fontsize=title_fontsize, fontweight='bold')
    #             else:
    #                 region_display = shell.replace('_', ' ').title() if residence_type != 'ion_pairing' else shell
    #                 ax.set_title(f'Residence Time Comparison: {ion_type} ({residence_type}) ({region_display}) {title_suffix}\n'
    #                             f'{salt1_label} vs {salt2_label}',
    #                             fontsize=title_fontsize, fontweight='bold')
                
    #             # Add legend and styling
    #             ax.legend(fontsize=legend_fontsize, frameon=False, loc='best')
    #             ax.tick_params(axis='both', labelsize=tick_fontsize)
    #             ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
                
    #             # Apply y-axis limits
    #             if y_min is not None or y_max is not None:
    #                 current_y_min, current_y_max = ax.get_ylim()
                    
    #                 y_min_actual = y_min if y_min is not None else max(0, current_y_min - current_y_max*0.1)
    #                 y_max_actual = y_max if y_max is not None else current_y_max + current_y_max*0.1
                    
    #                 ax.set_ylim(y_min_actual, y_max_actual)
    #             else:
    #                 ax.set_ylim(bottom=0)  # Residence times should start from 0
                
    #             plt.tight_layout()
                
    #             # Save plot
    #             if save_plots:
    #                 custom_suffix = '_custom' if chloride_count_groups is not None else ''
    #                 y_range_suffix = ''
    #                 if y_min is not None or y_max is not None:
    #                     y_min_str = f"{y_min}" if y_min is not None else "auto"
    #                     y_max_str = f"{y_max}" if y_max is not None else "auto"
    #                     y_range_suffix = f'_y{y_min_str}to{y_max_str}'
                    
    #                 if multi_region:
    #                     regions_str = '_'.join(regions_to_analyze)
    #                     filename = f'salt_residence_comparison_{ion_type}_{residence_type}_{regions_str}_{salt1_key}_vs_{salt2_key}_{x_axis_type}_line{custom_suffix}{y_range_suffix}.png'
    #                 else:
    #                     filename = f'salt_residence_comparison_{ion_type}_{residence_type}_{shell}_{salt1_key}_vs_{salt2_key}_{x_axis_type}_line{custom_suffix}{y_range_suffix}.png'
                    
    #                 plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    #                 log_lines.append(f"LINE PLOT SAVED: {filename}")
    #                 print(f"Line plot saved: {filename}")
                
    #             plt.show()
        
    #     elif plot_type == 'histogram':
    #         # Histogram plots for each salt pair
    #         for i, (salt1_key, salt2_key) in enumerate(salt_pairs):
                
    #             if salt1_key not in comparison_data or salt2_key not in comparison_data:
    #                 log_lines.append(f"WARNING: Skipping {salt1_key} vs {salt2_key} - missing data")
    #                 continue
                
    #             salt1_label = self.salt_data[salt1_key]['label']
    #             salt2_label = self.salt_data[salt2_key]['label']
    #             salt1_base_color = self.salt_data[salt1_key].get('color', 'blue')
    #             salt2_base_color = self.salt_data[salt2_key].get('color', 'red')
                
    #             # FIXED: Use appropriate colors based on residence type
    #             if residence_type == 'ion_pairing':
    #                 # Use fixed pairing colors for each region
    #                 salt1_region_colors = [pairing_colors.get(region, '#1f77b4') for region in regions_to_analyze]
    #                 salt2_region_colors = [pairing_colors.get(region, '#1f77b4') for region in regions_to_analyze]
    #             else:
    #                 # Generate shell-specific colors for each salt
    #                 n_regions = len(regions_to_analyze)
    #                 salt1_region_colors = get_region_colors(salt1_base_color, n_regions)
    #                 salt2_region_colors = get_region_colors(salt2_base_color, n_regions)
                
    #             # Create single figure for all regions
    #             fig, ax = plt.subplots(figsize=figsize, facecolor='white')
                
    #             # Find common concentrations
    #             common_concentrations = []
    #             for conc_key in concentrations:
    #                 if residence_type == 'ion_pairing':
    #                     # For ion pairing: include if both salts have the concentration
    #                     salt1_has_conc = any(conc_key in comparison_data[salt1_key].get(region, {}) 
    #                                     for region in regions_to_analyze)
    #                     salt2_has_conc = any(conc_key in comparison_data[salt2_key].get(region, {}) 
    #                                     for region in regions_to_analyze)
                        
    #                     if salt1_has_conc and salt2_has_conc:
    #                         common_concentrations.append(conc_key)
    #                 else:
    #                     # Water residence: require all shells
    #                     exists_in_all = True
    #                     for region_name in regions_to_analyze:
    #                         if (conc_key not in comparison_data[salt1_key].get(region_name, {}) or 
    #                             conc_key not in comparison_data[salt2_key].get(region_name, {})):
    #                             exists_in_all = False
    #                             break
    #                     if exists_in_all:
    #                         common_concentrations.append(conc_key)
                
    #             if not common_concentrations:
    #                 print(f"No common concentrations found for {salt1_label} vs {salt2_label}")
    #                 continue
                
    #             log_lines.append(f"Common concentrations for {salt1_label} vs {salt2_label}: {common_concentrations}")
                
    #             # Sort concentrations properly
    #             common_concentrations.sort(key=lambda x: float(x.replace('M', '')))
                
    #             # Prepare data for grouped bars
    #             n_groups = len(common_concentrations)
    #             x_pos = np.arange(n_groups)
                
    #             # Create x-axis labels
    #             x_labels = []
    #             for conc_key in common_concentrations:
    #                 if x_axis_type == 'chloride_count':
    #                     # Find any available data point for this concentration to get chloride count
    #                     cl_count = None
    #                     for region_name in regions_to_analyze:
    #                         if conc_key in comparison_data[salt1_key].get(region_name, {}):
    #                             cl_count = comparison_data[salt1_key][region_name][conc_key]['chloride_count']
    #                             break
                        
    #                     if cl_count is not None:
    #                         if chloride_count_groups is not None:
    #                             x_labels.append(f"{int(cl_count)} Cl⁻")
    #                         else:
    #                             x_labels.append(f"{int(cl_count)}")
    #                     else:
    #                         x_labels.append(conc_key)  # Fallback
    #                 else:
    #                     x_labels.append(conc_key)
                
    #             # Plot bars for each region
    #             for region_idx, region_name in enumerate(regions_to_analyze):
                    
    #                 # Collect data for this region
    #                 salt1_res_means = []
    #                 salt1_res_stds = []
    #                 salt2_res_means = []
    #                 salt2_res_stds = []
                    
    #                 for conc_key in common_concentrations:
    #                     # Salt 1 data for this region
    #                     if conc_key in comparison_data[salt1_key].get(region_name, {}):
    #                         salt1_data = comparison_data[salt1_key][region_name][conc_key]
    #                         salt1_res_means.append(salt1_data['res_mean'])
    #                         salt1_res_stds.append(salt1_data['res_std'])
    #                     else:
    #                         # Missing data - use 0
    #                         salt1_res_means.append(0.0)
    #                         salt1_res_stds.append(0.0)
    #                         log_lines.append(f"WARNING: Missing {salt1_label} data for {region_name} at {conc_key}")
                        
    #                     # Salt 2 data for this region
    #                     if conc_key in comparison_data[salt2_key].get(region_name, {}):
    #                         salt2_data = comparison_data[salt2_key][region_name][conc_key]
    #                         salt2_res_means.append(salt2_data['res_mean'])
    #                         salt2_res_stds.append(salt2_data['res_std'])
    #                     else:
    #                         # Missing data - use 0
    #                         salt2_res_means.append(0.0)
    #                         salt2_res_stds.append(0.0)
    #                         log_lines.append(f"WARNING: Missing {salt2_label} data for {region_name} at {conc_key}")
                    
    #                 # Convert to numpy arrays
    #                 salt1_res_means = np.array(salt1_res_means)
    #                 salt1_res_stds = np.array(salt1_res_stds)
    #                 salt2_res_means = np.array(salt2_res_means)
    #                 salt2_res_stds = np.array(salt2_res_stds)
                    
    #                 # Calculate x positions for this region's bars
    #                 total_region_width = bar_width
    #                 single_region_width = total_region_width / len(regions_to_analyze)
    #                 salt_bar_width = single_region_width / 2
                    
    #                 region_start = -total_region_width / 2
    #                 region_center = region_start + (region_idx + 0.5) * single_region_width
                    
    #                 salt1_offset = region_center - salt_bar_width
    #                 salt2_offset = region_center
                    
    #                 x1 = x_pos + salt1_offset
    #                 x2 = x_pos + salt2_offset
                    
    #                 # Get colors for this region
    #                 salt1_color = salt1_region_colors[region_idx]
    #                 salt2_color = salt2_region_colors[region_idx]
                    
    #                 # FIXED: Create labels for legend with proper formatting
    #                 if residence_type == 'ion_pairing':
    #                     # Keep ion pairing regions in FULL CAPITALS
    #                     salt1_label_text = f"{salt1_label} {region_name}"
    #                     salt2_label_text = f"{salt2_label} {region_name}"
    #                 else:
    #                     # Water residence - convert shell names to title case
    #                     salt1_label_text = f"{salt1_label} {region_name.replace('_', ' ').title()}"
    #                     salt2_label_text = f"{salt2_label} {region_name.replace('_', ' ').title()}"
                    
    #                 # Plot bars (only plot non-zero values)
    #                 mask1 = salt1_res_means > 0
    #                 mask2 = salt2_res_means > 0
                    
    #                 if np.any(mask1):
    #                     # FIXED: Salt 1 - solid bars, no hatching
    #                     bars1 = ax.bar(x1[mask1], salt1_res_means[mask1], salt_bar_width, alpha=alpha, 
    #                                 color=salt1_color, label=salt1_label_text,
    #                                 edgecolor='black', linewidth=0.5)
                        
    #                     # Add error bars only for non-zero values
    #                     ax.errorbar(x1[mask1], salt1_res_means[mask1], yerr=salt1_res_stds[mask1],
    #                             fmt='none', color='black', capsize=capsize, capthick=1.5, 
    #                             elinewidth=1.5, alpha=0.8)
                    
    #                 if np.any(mask2):
    #                     # FIXED: Salt 2 - hatched bars for distinction
    #                     bars2 = ax.bar(x2[mask2], salt2_res_means[mask2], salt_bar_width, alpha=alpha, 
    #                                 color=salt2_color, label=salt2_label_text,
    #                                 hatch='///', edgecolor='black', linewidth=0.5)  # Hatching for salt 2
                        
    #                     # Add error bars only for non-zero values
    #                     ax.errorbar(x2[mask2], salt2_res_means[mask2], yerr=salt2_res_stds[mask2],
    #                             fmt='none', color='black', capsize=capsize, capthick=1.5, 
    #                             elinewidth=1.5, alpha=0.8)

    #             # Customize plot
    #             if x_axis_type == 'chloride_count':
    #                 ax.set_xlabel('Chloride Ion Count (Cl⁻)', fontsize=label_fontsize)
    #                 title_suffix = '(vs Cl⁻ count)'
    #             else:
    #                 ax.set_xlabel('Concentration (M)', fontsize=label_fontsize)
    #                 title_suffix = '(vs concentration)'
                
    #             ax.set_ylabel('Mean Residence Time (ps)', fontsize=label_fontsize)
                
    #             if multi_region:
    #                 regions_str = ', '.join([s.replace('_', ' ').title() if residence_type != 'ion_pairing' else s for s in regions_to_analyze])
    #                 ax.set_title(f'Residence Time Histogram: {ion_type} ({residence_type}) {title_suffix}\n'
    #                             f'{salt1_label} vs {salt2_label} - {regions_str}',
    #                             fontsize=title_fontsize, fontweight='bold')
    #             else:
    #                 region_display = shell.replace('_', ' ').title() if residence_type != 'ion_pairing' else shell
    #                 ax.set_title(f'Residence Time Histogram: {ion_type} ({residence_type}) ({region_display}) {title_suffix}\n'
    #                             f'{salt1_label} vs {salt2_label}',
    #                             fontsize=title_fontsize, fontweight='bold')
                
    #             # Set x-axis ticks and labels
    #             ax.set_xticks(x_pos)
    #             ax.set_xticklabels(x_labels)
                
    #             # Add legend and styling
    #             ax.legend(fontsize=legend_fontsize, frameon=False, loc='best')
    #             ax.tick_params(axis='both', labelsize=tick_fontsize)
    #             ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
                
    #             # Apply y-axis limits
    #             if y_min is not None or y_max is not None:
    #                 current_y_min, current_y_max = ax.get_ylim()
                    
    #                 y_min_actual = y_min if y_min is not None else max(0, current_y_min - current_y_max*0.1)
    #                 y_max_actual = y_max if y_max is not None else current_y_max + current_y_max*0.1
                    
    #                 ax.set_ylim(y_min_actual, y_max_actual)
    #             else:
    #                 ax.set_ylim(bottom=0)
                
    #             # Add padding around bars
    #             ax.set_xlim(-0.5, n_groups - 0.5)
                
    #             plt.tight_layout()
                
    #             # Save plot
    #             if save_plots:
    #                 custom_suffix = '_custom' if chloride_count_groups is not None else ''
    #                 y_range_suffix = ''
    #                 if y_min is not None or y_max is not None:
    #                     y_min_str = f"{y_min}" if y_min is not None else "auto"
    #                     y_max_str = f"{y_max}" if y_max is not None else "auto"
    #                     y_range_suffix = f'_y{y_min_str}to{y_max_str}'
                    
    #                 if multi_region:
    #                     regions_str = '_'.join(regions_to_analyze)
    #                     filename = f'salt_residence_comparison_{ion_type}_{residence_type}_{regions_str}_{salt1_key}_vs_{salt2_key}_{x_axis_type}_hist{custom_suffix}{y_range_suffix}.png'
    #                 else:
    #                     filename = f'salt_residence_comparison_{ion_type}_{residence_type}_{shell}_{salt1_key}_vs_{salt2_key}_{x_axis_type}_hist{custom_suffix}{y_range_suffix}.png'
                    
    #                 plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    #                 log_lines.append(f"HISTOGRAM PLOT SAVED: {filename}")
    #                 print(f"Histogram plot saved: {filename}")
                
    #             plt.show()
        
    #     else:
    #         raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'line' or 'histogram'")
        
    #     # Save log
    #     if save_log:
    #         custom_suffix = '_custom' if chloride_count_groups is not None else ''
    #         y_range_suffix = ''
    #         if y_min is not None or y_max is not None:
    #             y_min_str = f"{y_min}" if y_min is not None else "auto"
    #             y_max_str = f"{y_max}" if y_max is not None else "auto"
    #             y_range_suffix = f'_y{y_min_str}to{y_max_str}'
            
    #         if multi_region:
    #             regions_str = '_'.join(regions_to_analyze)
    #             log_filename = f'salt_residence_comparison_{ion_type}_{residence_type}_{regions_str}_{x_axis_type}_{plot_type}{custom_suffix}{y_range_suffix}.log'
    #         else:
    #             log_filename = f'salt_residence_comparison_{ion_type}_{residence_type}_{shell}_{x_axis_type}_{plot_type}{custom_suffix}{y_range_suffix}.log'
            
    #         with open(log_filename, 'w') as f:
    #             f.write('\n'.join(log_lines))
    #         print(f"Log saved: {log_filename}")
        
    #     return comparison_data

    def compare_residence_times(self, ion_type, residence_type='water',
                            shell='shell_1', salt_pairs=None, concentrations=None,
                            save_plots=True, save_log=True,
                            figsize=(12, 8), alpha=0.7,
                            title_fontsize=14, label_fontsize=12, 
                            legend_fontsize=10, tick_fontsize=10,
                            x_axis_type='concentration',  # 'concentration' or 'chloride_count'
                            chloride_count_groups=None,
                            bar_width=0.35, capsize=5,
                            y_min=None, y_max=None,
                            shells=None,  # Multiple shells support
                            plot_type='line'):  # 'line' or 'histogram'
        '''
        [Previous docstring remains the same]
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_salts() first.")
            return None
        
        # Determine which shells/regions to analyze
        if shells is not None:
            regions_to_analyze = shells
            multi_region = True
        else:
            regions_to_analyze = [shell]
            multi_region = False
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        if multi_region:
            log_lines.append(f"RESIDENCE TIME COMPARISON: {ion_type} ({residence_type}) - Multiple Regions")
            log_lines.append(f"Regions: {', '.join(regions_to_analyze)}")
        else:
            log_lines.append(f"RESIDENCE TIME COMPARISON: {ion_type} ({residence_type}) - {shell}")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Residence Type: {residence_type}")
        log_lines.append(f"Plot Type: {plot_type}")
        log_lines.append(f"X-axis Type: {x_axis_type}")
        
        if y_min is not None:
            log_lines.append(f"Custom Y-axis minimum: {y_min}")
        else:
            log_lines.append("Y-axis minimum: Automatic scaling")
        
        if y_max is not None:
            log_lines.append(f"Custom Y-axis maximum: {y_max}")
        else:
            log_lines.append("Y-axis maximum: Automatic scaling with padding")
        log_lines.append("")
        
        # Determine salt pairs if not provided
        if salt_pairs is None:
            available_salts = list(self.loaded_data.keys())
            
            # EXCLUDE No_Salt and other non-salt systems
            excluded_salts = {'No_Salt', 'Pure_Water', 'Water', 'no_salt', 'pure_water'}
            available_salts = [s for s in available_salts if s not in excluded_salts and 
                            self.salt_data[s]['label'] not in excluded_salts]
            
            print(f"Available salts for comparison: {[self.salt_data[s]['label'] for s in available_salts]}")
            
            # Function to normalize salt labels for comparison
            def normalize_salt_label(label):
                '''Normalize salt label to handle different naming conventions'''
                normalized = label.upper()
                # Replace subscript/superscript characters with regular numbers
                normalized = normalized.replace('₂', '2').replace('²', '2')
                normalized = normalized.replace('₃', '3').replace('³', '3')
                normalized = normalized.replace('₄', '4').replace('⁴', '4')
                # Remove spaces and common separators
                normalized = normalized.replace(' ', '').replace('-', '').replace('_', '')
                return normalized
            
            # Separate monovalent and divalent based on labels
            monovalent = []
            divalent = []
            
            for salt_key in available_salts:
                label_normalized = normalize_salt_label(self.salt_data[salt_key]['label'])
                if ('CL2' in label_normalized or 'CACL2' in label_normalized or 
                    'MGCL2' in label_normalized or 'BACL2' in label_normalized or
                    'SRCL2' in label_normalized):
                    divalent.append(salt_key)
                else:
                    monovalent.append(salt_key)
            
            print(f"Monovalent salts: {[self.salt_data[s]['label'] for s in monovalent]}")
            print(f"Divalent salts: {[self.salt_data[s]['label'] for s in divalent]}")
            
            salt_pairs = []
            if len(monovalent) >= 2:
                salt_pairs.append((monovalent[0], monovalent[1]))  # e.g., NaCl vs KCl
            if len(divalent) >= 2:
                salt_pairs.append((divalent[0], divalent[1]))      # e.g., CaCl2 vs MgCl2
            
            if not salt_pairs:
                print("ERROR: Not enough salt pairs found for comparison")
                return None
            
            print(f"Salt pairs to compare: {[(self.salt_data[p[0]]['label'], self.salt_data[p[1]]['label']) for p in salt_pairs]}")
        
        log_lines.append(f"Salt pairs to compare:")
        for pair in salt_pairs:
            salt1_label = self.salt_data[pair[0]]['label']
            salt2_label = self.salt_data[pair[1]]['label']
            log_lines.append(f"  {salt1_label} vs {salt2_label}")
        log_lines.append("")
        
        # Determine concentrations
        if concentrations is None:
            if chloride_count_groups is not None:
                # Get all unique concentrations from the custom mapping
                all_concs = set()
                for cl_group, salt_concs in chloride_count_groups.items():
                    all_concs.update(salt_concs.values())
                concentrations = sorted(list(all_concs), key=lambda x: float(x.replace('M', '')))
            else:
                # Get all unique concentrations from loaded data
                all_concs = set()
                for salt_key in [salt for pair in salt_pairs for salt in pair]:
                    if salt_key in self.loaded_data:
                        all_concs.update(self.loaded_data[salt_key].keys())
                concentrations = sorted(list(all_concs), key=lambda x: float(x.replace('M', '')))
        
        log_lines.append(f"Concentrations: {', '.join(concentrations)}")
        log_lines.append("")
        
        # Function to normalize salt labels for comparison
        def normalize_salt_label(label):
            '''Normalize salt label to handle different naming conventions'''
            normalized = label.upper()
            normalized = normalized.replace('₂', '2').replace('²', '2')
            normalized = normalized.replace('₃', '3').replace('³', '3')
            normalized = normalized.replace('₄', '4').replace('⁴', '4')
            normalized = normalized.replace(' ', '').replace('-', '').replace('_', '')
            return normalized
        
        # Function to convert concentration to chloride count
        def get_chloride_count(salt_key, conc_str):
            '''Convert salt concentration to chloride ion count'''
            
            if chloride_count_groups is not None:
                # Use custom mapping - find which chloride group this salt-concentration belongs to
                salt_label = self.salt_data[salt_key]['label']
                salt_label_normalized = normalize_salt_label(salt_label)
                
                for cl_group, salt_concs in chloride_count_groups.items():
                    for mapping_salt_name, mapping_conc in salt_concs.items():
                        mapping_salt_normalized = normalize_salt_label(mapping_salt_name)
                        
                        # Check if normalized labels match and concentrations match
                        if (salt_label_normalized == mapping_salt_normalized and 
                            mapping_conc == conc_str):
                            # Extract numeric value from chloride group name (e.g., '11_Cl' -> 11)
                            chloride_count = float(cl_group.replace('_Cl', ''))
                            return chloride_count
                
                # Fallback to automatic calculation if not found in custom mapping
                print(f"WARNING: {salt_label} ({salt_label_normalized}) {conc_str} not found in chloride_count_groups, using automatic calculation")
            
            # Automatic calculation
            conc_value = float(conc_str.replace('M', ''))
            salt_label_normalized = normalize_salt_label(self.salt_data[salt_key]['label'])
            
            # Check for divalent salts using normalized labels
            if ('CL2' in salt_label_normalized or 'CACL2' in salt_label_normalized or 
                'MGCL2' in salt_label_normalized or 'BACL2' in salt_label_normalized or
                'SRCL2' in salt_label_normalized):
                chloride_multiplier = 2
            else:
                chloride_multiplier = 1
            
            chloride_count = conc_value * chloride_multiplier * 100  # Scale to reasonable numbers
            return chloride_count
        
        # FIXED: Function to get region colors based on residence type
        def get_region_colors_for_residence_type(salt_base_color, regions, residence_type):
            '''
            Get region colors based on residence type.
            For ion_pairing: use fixed pairing colors
            For water: use blue saturation gradient from EquilibriumAnalysisOptimized
            '''
            import matplotlib.colors as mcolors
            
            if residence_type == 'ion_pairing':
                # Use fixed ion pairing colors
                pairing_color_map = {
                    'CIP': 'lightcoral',      # Light red
                    'SIP': 'lightblue',       # Light blue  
                    'DSIP': 'lightgreen',     # Light green
                    'FI': 'lightyellow'       # Light yellow
                }
                return [pairing_color_map.get(region, '#808080') for region in regions]
            
            else:  # water residence
                # Use the blue saturation gradient from EquilibriumAnalysisOptimized
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
                
                # Generate shell colors based on the number of regions
                shell_color_list = get_blue_saturation_colors_from_00c5ff(len(regions))
                return shell_color_list[:len(regions)]  # Take only the needed number of colors
        
        # Collect residence time data for all relevant salts and regions
        comparison_data = {}
        
        log_lines.append("RESIDENCE TIME DATA")
        log_lines.append("-"*80)
        if x_axis_type == 'chloride_count':
            log_lines.append(f"{'Salt':<15} {'Region':<10} {'Conc (M)':<10} {'Cl- Count':<12} {'Mean (ps)':<12} {'Std (ps)':<12}")
        else:
            log_lines.append(f"{'Salt':<15} {'Region':<10} {'Conc (M)':<10} {'Mean (ps)':<12} {'Std (ps)':<12}")
        log_lines.append("-"*80)

        # Handle ion pairing with region-specific analysis
        if residence_type == 'ion_pairing':
            # For ion pairing, regions refer to pairing regions (CIP, SIP, DSIP)
            pairing_regions_to_analyze = regions_to_analyze  # CIP, SIP, DSIP, etc.
            
            for salt_key in [salt for pair in salt_pairs for salt in pair]:
                if salt_key not in self.loaded_data:
                    continue
                    
                comparison_data[salt_key] = {}
                
                for region_name in pairing_regions_to_analyze:
                    comparison_data[salt_key][region_name] = {}
                    
                    for conc_key in concentrations:
                        if conc_key not in self.loaded_data[salt_key]:
                            continue
                        
                        conc_data = self.loaded_data[salt_key][conc_key]
                        
                        # Get ion pairing residence time data for specific region
                        res_mean = None
                        res_std = None
                        
                        if 'ion_pairing_residence_times' in conc_data:
                            res_dict = conc_data['ion_pairing_residence_times']
                            
                            # Try exact ion name and variations
                            for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
                                if ion_variant in res_dict:
                                    res_data = res_dict[ion_variant]
                                    if isinstance(res_data, dict) and 'regions' in res_data:
                                        regions = res_data['regions']
                                        
                                        # Get data for specific region
                                        if region_name in regions:
                                            region_data = regions[region_name]
                                            if isinstance(region_data, dict):
                                                # Look for mean residence time in various possible keys
                                                for mean_key in ['mean_residence_time', 'mean', 'avg_residence_time', 'average']:
                                                    if mean_key in region_data:
                                                        res_mean = region_data[mean_key]
                                                        break
                                                
                                                for std_key in ['std_residence_time', 'std', 'stdev', 'standard_deviation']:
                                                    if std_key in region_data:
                                                        res_std = region_data[std_key]
                                                        break
                                                
                                                # If std not found, set to 0
                                                if res_std is None:
                                                    res_std = 0.0
                                    
                                    if res_mean is not None:
                                        break
                        
                        if res_mean is not None:
                            # Store region-specific data
                            chloride_count = get_chloride_count(salt_key, conc_key)
                            
                            comparison_data[salt_key][region_name][conc_key] = {
                                'res_mean': res_mean,
                                'res_std': res_std if res_std is not None else 0.0,
                                'concentration': float(conc_key.replace('M', '')),
                                'chloride_count': chloride_count
                            }
                            
                            # Log output
                            if x_axis_type == 'chloride_count':
                                log_lines.append(f"{self.salt_data[salt_key]['label']:<15} {region_name:<10} {conc_key:<10} {chloride_count:<12.0f} {res_mean:<12.2f} {res_std:<12.2f}")
                            else:
                                log_lines.append(f"{self.salt_data[salt_key]['label']:<15} {region_name:<10} {conc_key:<10} {res_mean:<12.2f} {res_std:<12.2f}")

        else:  # water residence times
            for salt_key in [salt for pair in salt_pairs for salt in pair]:
                if salt_key not in self.loaded_data:
                    continue
                    
                comparison_data[salt_key] = {}
                
                for shell_name in regions_to_analyze:
                    comparison_data[salt_key][shell_name] = {}
                    
                    for conc_key in concentrations:
                        if conc_key not in self.loaded_data[salt_key]:
                            continue
                        
                        conc_data = self.loaded_data[salt_key][conc_key]
                        
                        # Get water residence time data with statistics
                        res_mean = None
                        res_std = None
                        
                        if 'water_residence_times' in conc_data:
                            res_dict = conc_data['water_residence_times']
                            
                            # Try exact ion name and variations
                            for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
                                if ion_variant in res_dict:
                                    res_data = res_dict[ion_variant]
                                    if 'shells' in res_data and shell_name in res_data['shells']:
                                        shell_data = res_data['shells'][shell_name]
                                        res_mean = shell_data.get('mean_residence_time', shell_data.get('mean', None))
                                        res_std = shell_data.get('std_residence_time', shell_data.get('std', 0.0))
                                        break
                        
                        if res_mean is not None:
                            # Store concentration, chloride count, mean, and std
                            chloride_count = get_chloride_count(salt_key, conc_key)
                            
                            comparison_data[salt_key][shell_name][conc_key] = {
                                'res_mean': res_mean,
                                'res_std': res_std if res_std is not None else 0.0,
                                'concentration': float(conc_key.replace('M', '')),
                                'chloride_count': chloride_count
                            }
                            
                            # Log output
                            if x_axis_type == 'chloride_count':
                                log_lines.append(f"{self.salt_data[salt_key]['label']:<15} {shell_name:<10} {conc_key:<10} {chloride_count:<12.0f} {res_mean:<12.2f} {res_std:<12.2f}")
                            else:
                                log_lines.append(f"{self.salt_data[salt_key]['label']:<15} {shell_name:<10} {conc_key:<10} {res_mean:<12.2f} {res_std:<12.2f}")

        log_lines.append("")
        
        # Create plots based on plot_type
        if plot_type == 'line':
            # Line plot for each salt pair
            for i, (salt1_key, salt2_key) in enumerate(salt_pairs):
                
                if salt1_key not in comparison_data or salt2_key not in comparison_data:
                    log_lines.append(f"WARNING: Skipping {salt1_key} vs {salt2_key} - missing data")
                    continue
                
                salt1_label = self.salt_data[salt1_key]['label']
                salt2_label = self.salt_data[salt2_key]['label']
                salt1_base_color = self.salt_data[salt1_key].get('color', 'blue')
                salt2_base_color = self.salt_data[salt2_key].get('color', 'red')
                
                # FIXED: Use the correct color function
                salt1_region_colors = get_region_colors_for_residence_type(salt1_base_color, regions_to_analyze, residence_type)
                salt2_region_colors = get_region_colors_for_residence_type(salt2_base_color, regions_to_analyze, residence_type)
                
                # Create figure
                fig, ax = plt.subplots(figsize=figsize, facecolor='white')
                
                # Find common concentrations across all regions for both salts
                common_concentrations = []
                for conc_key in concentrations:
                    if residence_type == 'ion_pairing':
                        # For ion pairing: include if both salts have the concentration (even if not in all regions)
                        salt1_has_conc = any(conc_key in comparison_data[salt1_key].get(region, {}) 
                                        for region in regions_to_analyze)
                        salt2_has_conc = any(conc_key in comparison_data[salt2_key].get(region, {}) 
                                        for region in regions_to_analyze)
                        
                        if salt1_has_conc and salt2_has_conc:
                            common_concentrations.append(conc_key)
                    else:
                        # Water residence: require all shells
                        exists_in_all = True
                        for region_name in regions_to_analyze:
                            if (conc_key not in comparison_data[salt1_key].get(region_name, {}) or 
                                conc_key not in comparison_data[salt2_key].get(region_name, {})):
                                exists_in_all = False
                                break
                        if exists_in_all:
                            common_concentrations.append(conc_key)
                
                if not common_concentrations:
                    print(f"No common concentrations found for {salt1_label} vs {salt2_label}")
                    continue
                
                # Sort concentrations properly
                common_concentrations.sort(key=lambda x: float(x.replace('M', '')))
                
                # Plot each region as a separate line
                for region_idx, region_name in enumerate(regions_to_analyze):
                    
                    # Collect data for this region
                    salt1_x_values = []
                    salt1_res_means = []
                    salt1_res_stds = []
                    salt2_x_values = []
                    salt2_res_means = []
                    salt2_res_stds = []
                    
                    for conc_key in common_concentrations:
                        # Salt 1 data for this region
                        if conc_key in comparison_data[salt1_key].get(region_name, {}):
                            salt1_data = comparison_data[salt1_key][region_name][conc_key]
                            if x_axis_type == 'chloride_count':
                                salt1_x_values.append(salt1_data['chloride_count'])
                            else:
                                salt1_x_values.append(salt1_data['concentration'])
                            salt1_res_means.append(salt1_data['res_mean'])
                            salt1_res_stds.append(salt1_data['res_std'])
                        
                        # Salt 2 data for this region
                        if conc_key in comparison_data[salt2_key].get(region_name, {}):
                            salt2_data = comparison_data[salt2_key][region_name][conc_key]
                            if x_axis_type == 'chloride_count':
                                salt2_x_values.append(salt2_data['chloride_count'])
                            else:
                                salt2_x_values.append(salt2_data['concentration'])
                            salt2_res_means.append(salt2_data['res_mean'])
                            salt2_res_stds.append(salt2_data['res_std'])
                    
                    # Get colors for this region
                    salt1_color = salt1_region_colors[region_idx]
                    salt2_color = salt2_region_colors[region_idx]
                    
                    # FIXED: Create labels for legend with proper formatting
                    if residence_type == 'ion_pairing':
                        # Keep ion pairing regions in FULL CAPITALS
                        salt1_label_text = f"{salt1_label} {region_name}"
                        salt2_label_text = f"{salt2_label} {region_name}"
                    else:
                        # Water residence - convert shell names to title case
                        salt1_label_text = f"{salt1_label} {region_name.replace('_', ' ').title()}"
                        salt2_label_text = f"{salt2_label} {region_name.replace('_', ' ').title()}"
                    
                    # FIXED: Plot lines with distinct markers and line styles
                    if len(salt1_x_values) > 0:
                        ax.errorbar(salt1_x_values, salt1_res_means, yerr=salt1_res_stds,
                                marker='o', markersize=6, capsize=capsize, linewidth=2,
                                linestyle='-', color=salt1_color, label=salt1_label_text)
                    
                    if len(salt2_x_values) > 0:
                        ax.errorbar(salt2_x_values, salt2_res_means, yerr=salt2_res_stds,
                                marker='s', markersize=6, capsize=capsize, linewidth=2,
                                linestyle='--', color=salt2_color, label=salt2_label_text)  # Dashed line for salt 2
                
                # Customize plot
                if x_axis_type == 'chloride_count':
                    ax.set_xlabel('Chloride Ion Count (Cl⁻)', fontsize=label_fontsize)
                    title_suffix = '(vs Cl⁻ count)'
                else:
                    ax.set_xlabel('Concentration (M)', fontsize=label_fontsize)
                    title_suffix = '(vs concentration)'
                
                ax.set_ylabel('Mean Residence Time (ps)', fontsize=label_fontsize)
                
                if multi_region:
                    regions_str = ', '.join([s.replace('_', ' ').title() if residence_type != 'ion_pairing' else s for s in regions_to_analyze])
                    ax.set_title(f'Residence Time Comparison: {ion_type} ({residence_type}) {title_suffix}\n'
                                f'{salt1_label} vs {salt2_label} - {regions_str}',
                                fontsize=title_fontsize, fontweight='bold')
                else:
                    region_display = shell.replace('_', ' ').title() if residence_type != 'ion_pairing' else shell
                    ax.set_title(f'Residence Time Comparison: {ion_type} ({residence_type}) ({region_display}) {title_suffix}\n'
                                f'{salt1_label} vs {salt2_label}',
                                fontsize=title_fontsize, fontweight='bold')
                
                # Add legend and styling
                ax.legend(fontsize=legend_fontsize, frameon=False, loc='best')
                ax.tick_params(axis='both', labelsize=tick_fontsize)
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
                
                # Apply y-axis limits
                if y_min is not None or y_max is not None:
                    current_y_min, current_y_max = ax.get_ylim()
                    
                    y_min_actual = y_min if y_min is not None else max(0, current_y_min - current_y_max*0.1)
                    y_max_actual = y_max if y_max is not None else current_y_max + current_y_max*0.1
                    
                    ax.set_ylim(y_min_actual, y_max_actual)
                else:
                    ax.set_ylim(bottom=0)  # Residence times should start from 0
                
                plt.tight_layout()
                
                # Save plot
                if save_plots:
                    custom_suffix = '_custom' if chloride_count_groups is not None else ''
                    y_range_suffix = ''
                    if y_min is not None or y_max is not None:
                        y_min_str = f"{y_min}" if y_min is not None else "auto"
                        y_max_str = f"{y_max}" if y_max is not None else "auto"
                        y_range_suffix = f'_y{y_min_str}to{y_max_str}'
                    
                    if multi_region:
                        regions_str = '_'.join(regions_to_analyze)
                        filename = f'salt_residence_comparison_{ion_type}_{residence_type}_{regions_str}_{salt1_key}_vs_{salt2_key}_{x_axis_type}_line{custom_suffix}{y_range_suffix}.png'
                    else:
                        filename = f'salt_residence_comparison_{ion_type}_{residence_type}_{shell}_{salt1_key}_vs_{salt2_key}_{x_axis_type}_line{custom_suffix}{y_range_suffix}.png'
                    
                    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                    log_lines.append(f"LINE PLOT SAVED: {filename}")
                    print(f"Line plot saved: {filename}")
                
                plt.show()
        
        elif plot_type == 'histogram':
            # Histogram plots for each salt pair
            for i, (salt1_key, salt2_key) in enumerate(salt_pairs):
                
                if salt1_key not in comparison_data or salt2_key not in comparison_data:
                    log_lines.append(f"WARNING: Skipping {salt1_key} vs {salt2_key} - missing data")
                    continue
                
                salt1_label = self.salt_data[salt1_key]['label']
                salt2_label = self.salt_data[salt2_key]['label']
                salt1_base_color = self.salt_data[salt1_key].get('color', 'blue')
                salt2_base_color = self.salt_data[salt2_key].get('color', 'red')
                
                # FIXED: Use the correct color function
                salt1_region_colors = get_region_colors_for_residence_type(salt1_base_color, regions_to_analyze, residence_type)
                salt2_region_colors = get_region_colors_for_residence_type(salt2_base_color, regions_to_analyze, residence_type)
                
                # Create single figure for all regions
                fig, ax = plt.subplots(figsize=figsize, facecolor='white')
                
                # Find common concentrations
                common_concentrations = []
                for conc_key in concentrations:
                    if residence_type == 'ion_pairing':
                        # For ion pairing: include if both salts have the concentration
                        salt1_has_conc = any(conc_key in comparison_data[salt1_key].get(region, {}) 
                                        for region in regions_to_analyze)
                        salt2_has_conc = any(conc_key in comparison_data[salt2_key].get(region, {}) 
                                        for region in regions_to_analyze)
                        
                        if salt1_has_conc and salt2_has_conc:
                            common_concentrations.append(conc_key)
                    else:
                        # Water residence: require all shells
                        exists_in_all = True
                        for region_name in regions_to_analyze:
                            if (conc_key not in comparison_data[salt1_key].get(region_name, {}) or 
                                conc_key not in comparison_data[salt2_key].get(region_name, {})):
                                exists_in_all = False
                                break
                        if exists_in_all:
                            common_concentrations.append(conc_key)
                
                if not common_concentrations:
                    print(f"No common concentrations found for {salt1_label} vs {salt2_label}")
                    continue
                
                log_lines.append(f"Common concentrations for {salt1_label} vs {salt2_label}: {common_concentrations}")
                
                # Sort concentrations properly
                common_concentrations.sort(key=lambda x: float(x.replace('M', '')))
                
                # Prepare data for grouped bars
                n_groups = len(common_concentrations)
                x_pos = np.arange(n_groups)
                
                # Create x-axis labels
                x_labels = []
                for conc_key in common_concentrations:
                    if x_axis_type == 'chloride_count':
                        # Find any available data point for this concentration to get chloride count
                        cl_count = None
                        for region_name in regions_to_analyze:
                            if conc_key in comparison_data[salt1_key].get(region_name, {}):
                                cl_count = comparison_data[salt1_key][region_name][conc_key]['chloride_count']
                                break
                        
                        if cl_count is not None:
                            if chloride_count_groups is not None:
                                x_labels.append(f"{int(cl_count)} Cl⁻")
                            else:
                                x_labels.append(f"{int(cl_count)}")
                        else:
                            x_labels.append(conc_key)  # Fallback
                    else:
                        x_labels.append(conc_key)
                
                # Plot bars for each region
                for region_idx, region_name in enumerate(regions_to_analyze):
                    
                    # Collect data for this region
                    salt1_res_means = []
                    salt1_res_stds = []
                    salt2_res_means = []
                    salt2_res_stds = []
                    
                    for conc_key in common_concentrations:
                        # Salt 1 data for this region
                        if conc_key in comparison_data[salt1_key].get(region_name, {}):
                            salt1_data = comparison_data[salt1_key][region_name][conc_key]
                            salt1_res_means.append(salt1_data['res_mean'])
                            salt1_res_stds.append(salt1_data['res_std'])
                        else:
                            # Missing data - use 0
                            salt1_res_means.append(0.0)
                            salt1_res_stds.append(0.0)
                            log_lines.append(f"WARNING: Missing {salt1_label} data for {region_name} at {conc_key}")
                        
                        # Salt 2 data for this region
                        if conc_key in comparison_data[salt2_key].get(region_name, {}):
                            salt2_data = comparison_data[salt2_key][region_name][conc_key]
                            salt2_res_means.append(salt2_data['res_mean'])
                            salt2_res_stds.append(salt2_data['res_std'])
                        else:
                            # Missing data - use 0
                            salt2_res_means.append(0.0)
                            salt2_res_stds.append(0.0)
                            log_lines.append(f"WARNING: Missing {salt2_label} data for {region_name} at {conc_key}")
                    
                    # Convert to numpy arrays
                    salt1_res_means = np.array(salt1_res_means)
                    salt1_res_stds = np.array(salt1_res_stds)
                    salt2_res_means = np.array(salt2_res_means)
                    salt2_res_stds = np.array(salt2_res_stds)
                    
                    # Calculate x positions for this region's bars
                    total_region_width = bar_width
                    single_region_width = total_region_width / len(regions_to_analyze)
                    salt_bar_width = single_region_width / 2
                    
                    region_start = -total_region_width / 2
                    region_center = region_start + (region_idx + 0.5) * single_region_width
                    
                    salt1_offset = region_center - salt_bar_width
                    salt2_offset = region_center
                    
                    x1 = x_pos + salt1_offset
                    x2 = x_pos + salt2_offset
                    
                    # Get colors for this region
                    salt1_color = salt1_region_colors[region_idx]
                    salt2_color = salt2_region_colors[region_idx]
                    
                    # FIXED: Create labels for legend with proper formatting
                    if residence_type == 'ion_pairing':
                        # Keep ion pairing regions in FULL CAPITALS
                        salt1_label_text = f"{salt1_label} {region_name}"
                        salt2_label_text = f"{salt2_label} {region_name}"
                    else:
                        # Water residence - convert shell names to title case
                        salt1_label_text = f"{salt1_label} {region_name.replace('_', ' ').title()}"
                        salt2_label_text = f"{salt2_label} {region_name.replace('_', ' ').title()}"
                    
                    # Plot bars (only plot non-zero values)
                    mask1 = salt1_res_means > 0
                    mask2 = salt2_res_means > 0
                    
                    if np.any(mask1):
                        # FIXED: Salt 1 - solid bars, no hatching
                        bars1 = ax.bar(x1[mask1], salt1_res_means[mask1], salt_bar_width, alpha=alpha, 
                                    color=salt1_color, label=salt1_label_text,
                                    edgecolor='black', linewidth=0.5)
                        
                        # Add error bars only for non-zero values
                        ax.errorbar(x1[mask1], salt1_res_means[mask1], yerr=salt1_res_stds[mask1],
                                fmt='none', color='black', capsize=capsize, capthick=1.5, 
                                elinewidth=1.5, alpha=0.8)
                    
                    if np.any(mask2):
                        # FIXED: Salt 2 - hatched bars for distinction
                        bars2 = ax.bar(x2[mask2], salt2_res_means[mask2], salt_bar_width, alpha=alpha, 
                                    color=salt2_color, label=salt2_label_text,
                                    hatch='///', edgecolor='black', linewidth=0.5)  # Hatching for salt 2
                        
                        # Add error bars only for non-zero values
                        ax.errorbar(x2[mask2], salt2_res_means[mask2], yerr=salt2_res_stds[mask2],
                                fmt='none', color='black', capsize=capsize, capthick=1.5, 
                                elinewidth=1.5, alpha=0.8)

                # Customize plot
                if x_axis_type == 'chloride_count':
                    ax.set_xlabel('Chloride Ion Count (Cl⁻)', fontsize=label_fontsize)
                    title_suffix = '(vs Cl⁻ count)'
                else:
                    ax.set_xlabel('Concentration (M)', fontsize=label_fontsize)
                    title_suffix = '(vs concentration)'
                
                ax.set_ylabel('Mean Residence Time (ps)', fontsize=label_fontsize)
                
                if multi_region:
                    regions_str = ', '.join([s.replace('_', ' ').title() if residence_type != 'ion_pairing' else s for s in regions_to_analyze])
                    ax.set_title(f'Residence Time Histogram: {ion_type} ({residence_type}) {title_suffix}\n'
                                f'{salt1_label} vs {salt2_label} - {regions_str}',
                                fontsize=title_fontsize, fontweight='bold')
                else:
                    region_display = shell.replace('_', ' ').title() if residence_type != 'ion_pairing' else shell
                    ax.set_title(f'Residence Time Histogram: {ion_type} ({residence_type}) ({region_display}) {title_suffix}\n'
                                f'{salt1_label} vs {salt2_label}',
                                fontsize=title_fontsize, fontweight='bold')
                
                # Set x-axis ticks and labels
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels)
                
                # Add legend and styling
                ax.legend(fontsize=legend_fontsize, frameon=False, loc='best')
                ax.tick_params(axis='both', labelsize=tick_fontsize)
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
                
                # Apply y-axis limits
                if y_min is not None or y_max is not None:
                    current_y_min, current_y_max = ax.get_ylim()
                    
                    y_min_actual = y_min if y_min is not None else max(0, current_y_min - current_y_max*0.1)
                    y_max_actual = y_max if y_max is not None else current_y_max + current_y_max*0.1
                    
                    ax.set_ylim(y_min_actual, y_max_actual)
                else:
                    ax.set_ylim(bottom=0)
                
                # Add padding around bars
                ax.set_xlim(-0.5, n_groups - 0.5)
                
                plt.tight_layout()
                
                # Save plot
                if save_plots:
                    custom_suffix = '_custom' if chloride_count_groups is not None else ''
                    y_range_suffix = ''
                    if y_min is not None or y_max is not None:
                        y_min_str = f"{y_min}" if y_min is not None else "auto"
                        y_max_str = f"{y_max}" if y_max is not None else "auto"
                        y_range_suffix = f'_y{y_min_str}to{y_max_str}'
                    
                    if multi_region:
                        regions_str = '_'.join(regions_to_analyze)
                        filename = f'salt_residence_comparison_{ion_type}_{residence_type}_{regions_str}_{salt1_key}_vs_{salt2_key}_{x_axis_type}_hist{custom_suffix}{y_range_suffix}.png'
                    else:
                        filename = f'salt_residence_comparison_{ion_type}_{residence_type}_{shell}_{salt1_key}_vs_{salt2_key}_{x_axis_type}_hist{custom_suffix}{y_range_suffix}.png'
                    
                    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                    log_lines.append(f"HISTOGRAM PLOT SAVED: {filename}")
                    print(f"Histogram plot saved: {filename}")
                
                plt.show()
        
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'line' or 'histogram'")
        
        # Save log
        if save_log:
            custom_suffix = '_custom' if chloride_count_groups is not None else ''
            y_range_suffix = ''
            if y_min is not None or y_max is not None:
                y_min_str = f"{y_min}" if y_min is not None else "auto"
                y_max_str = f"{y_max}" if y_max is not None else "auto"
                y_range_suffix = f'_y{y_min_str}to{y_max_str}'
            
            if multi_region:
                regions_str = '_'.join(regions_to_analyze)
                log_filename = f'salt_residence_comparison_{ion_type}_{residence_type}_{regions_str}_{x_axis_type}_{plot_type}{custom_suffix}{y_range_suffix}.log'
            else:
                log_filename = f'salt_residence_comparison_{ion_type}_{residence_type}_{shell}_{x_axis_type}_{plot_type}{custom_suffix}{y_range_suffix}.log'
            
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Log saved: {log_filename}")
        
        return comparison_data


    def compare_shell_coordination_probabilities_by_salt(self, ion_type, concentration='0.11M', 
                                                    salt_pairs=None, shell='shell_1',
                                                    save_plots=True, save_log=True,
                                                    figsize=(12, 8), alpha=0.7,
                                                    title_fontsize=14, label_fontsize=12,
                                                    legend_fontsize=10, tick_fontsize=10,
                                                    bar_width=0.35, group_spacing=1.0,
                                                    min_probability_threshold=0.5):  # NEW PARAMETER
        '''
        Compare shell coordination probabilities across different salts at a specific concentration.
        Shows coordination environments (shell types like "0-6", "1-5", etc.) on x-axis with probability on y-axis.
        
        Parameters
        ----------
        ion_type : str
            Ion to analyze (e.g., 'Na', 'K', 'Mg', 'Ca', 'Cl')
        concentration : str
            Concentration to analyze (e.g., '0.11M', '0.2M')
        salt_pairs : list of tuples, optional
            Pairs of salts to compare, e.g., [('NaCl', 'KCl'), ('CaCl2', 'MgCl2')]
            If None, compares first available pair based on ion_type
        shell : str
            Shell to analyze ('shell_1', 'shell_2', 'shell_3'), default='shell_1'
        save_plots : bool
            Whether to save the plots
        save_log : bool
            Whether to save analysis log
        figsize : tuple
            Figure size (width, height)
        alpha : float
            Transparency for bars
        title_fontsize : int
            Font size for titles
        label_fontsize : int
            Font size for axis labels
        legend_fontsize : int
            Font size for legend
        tick_fontsize : int
            Font size for tick labels
        bar_width : float
            Width of individual bars
        group_spacing : float
            Spacing between coordination environment groups
        min_probability_threshold : float, default=0.5
            Minimum probability threshold (%). Coordination environments with probabilities
            below this value will be filtered out from the plot and analysis.
            
        Returns
        -------
        comparison_data : dict
            Dictionary with shell coordination probability data for each salt
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_salts() first.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"SHELL COORDINATION PROBABILITIES COMPARISON: {ion_type} ({shell}) at {concentration}")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Shell: {shell}")
        log_lines.append(f"Concentration: {concentration}")
        log_lines.append(f"Minimum Probability Threshold: {min_probability_threshold}%")  # NEW LOG LINE
        log_lines.append("")
        
        # Determine salt pairs if not provided
        if salt_pairs is None:
            available_salts = list(self.loaded_data.keys())
            excluded_salts = {'No_Salt', 'Pure_Water', 'Water', 'no_salt', 'pure_water'}
            available_salts = [s for s in available_salts if s not in excluded_salts]
            
            # Auto-determine appropriate salt pairs based on ion_type
            if ion_type in ['Na', 'K', 'Li', 'Rb']:  # Monovalent cations
                # Look for monovalent salt pairs
                monovalent_salts = []
                for salt_key in available_salts:
                    label_normalized = self.salt_data[salt_key]['label'].upper().replace('₂', '2')
                    if 'CL' in label_normalized and '2' not in label_normalized:
                        monovalent_salts.append(salt_key)
                
                if len(monovalent_salts) >= 2:
                    salt_pairs = [(monovalent_salts[0], monovalent_salts[1])]
                else:
                    print(f"ERROR: Need at least 2 monovalent salts for {ion_type} comparison")
                    return None
                    
            elif ion_type in ['Mg', 'Ca', 'Sr', 'Ba']:  # Divalent cations
                # Look for divalent salt pairs
                divalent_salts = []
                for salt_key in available_salts:
                    label_normalized = self.salt_data[salt_key]['label'].upper().replace('₂', '2')
                    if 'CL2' in label_normalized or 'CACL2' in label_normalized or 'MGCL2' in label_normalized:
                        divalent_salts.append(salt_key)
                
                if len(divalent_salts) >= 2:
                    salt_pairs = [(divalent_salts[0], divalent_salts[1])]
                else:
                    print(f"ERROR: Need at least 2 divalent salts for {ion_type} comparison")
                    return None
                    
            elif ion_type in ['Cl', 'Br', 'F', 'I']:  # Anions
                # For anions, can compare across any salt types
                if len(available_salts) >= 2:
                    salt_pairs = [(available_salts[0], available_salts[1])]
                else:
                    print(f"ERROR: Need at least 2 salts for {ion_type} comparison")
                    return None
            else:
                print(f"ERROR: Unknown ion type {ion_type} or insufficient salts for comparison")
                return None
        
        log_lines.append(f"Salt pairs to compare:")
        for pair in salt_pairs:
            salt1_label = self.salt_data[pair[0]]['label']
            salt2_label = self.salt_data[pair[1]]['label']
            log_lines.append(f"  {salt1_label} vs {salt2_label}")
        log_lines.append("")
        
        # Collect shell coordination probability data
        comparison_data = {}
        
        log_lines.append("SHELL COORDINATION PROBABILITY DATA (RAW)")
        log_lines.append("-"*80)
        log_lines.append(f"{'Salt':<20} {'Environment':<15} {'Probability (%)':<15} {'Status':<10}")
        log_lines.append("-"*80)
        
        for salt_key in [salt for pair in salt_pairs for salt in pair]:
            if salt_key not in self.loaded_data:
                continue
            
            if concentration not in self.loaded_data[salt_key]:
                print(f"WARNING: {concentration} not found for {self.salt_data[salt_key]['label']}")
                continue
            
            conc_data = self.loaded_data[salt_key][concentration]
            comparison_data[salt_key] = {}
            
            # Look for shell region coordination probabilities (preferred)
            env_probs = None

            # Method 1: Look for shell_region_coordination_probabilities
            if 'shell_region_coordination_probabilities' in conc_data:
                region_prob_data = conc_data['shell_region_coordination_probabilities']
                
                # Try to find ion-specific data
                for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
                    if ion_variant in region_prob_data:
                        ion_data = region_prob_data[ion_variant]
                        
                        if 'shell_regions' in ion_data and shell in ion_data['shell_regions']:
                            shell_data = ion_data['shell_regions'][shell]
                            
                            if 'coordination_environments' in shell_data:
                                coord_envs = shell_data['coordination_environments']
                                
                                # FIXED: These are raw counts, not probabilities
                                # First, calculate total count
                                total_count = sum(coord_envs.values())
                                
                                env_probs = {}
                                
                                for env_name, env_value in coord_envs.items():
                                    if isinstance(env_value, (int, float)):
                                        # Convert count to percentage
                                        probability = (float(env_value) / total_count) * 100
                                        env_probs[env_name] = probability
                                break
                        break
            
            # Method 2: Look for shell_probabilities_by_ion_type (from Solute speciation)
            if env_probs is None and 'shell_probabilities_by_ion_type' in conc_data:
                shell_prob_data = conc_data['shell_probabilities_by_ion_type']
                
                # Try to find the specific ion type
                for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
                    if ion_variant in shell_prob_data:
                        ion_data = shell_prob_data[ion_variant]
                        
                        if 'data' in ion_data and hasattr(ion_data['data'], 'iterrows'):
                            # This is a DataFrame with shell types and fractions
                            df = ion_data['data']
                            env_probs = {}
                            
                            for _, row in df.iterrows():
                                shell_type = row['shell']
                                fraction = row['fraction']
                                
                                # FIXED: Check if fraction is already a percentage
                                if fraction > 1:
                                    # Already a percentage
                                    probability = fraction
                                else:
                                    # Fraction, convert to percentage
                                    probability = fraction * 100
                                
                                # Store the shell type as the environment name
                                env_probs[shell_type] = probability
                            break
            
            # Method 3: Fallback to shell_coordination_probabilities
            if env_probs is None and 'shell_coordination_probabilities' in conc_data:
                shell_prob_data = conc_data['shell_coordination_probabilities']
                
                for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
                    if ion_variant in shell_prob_data:
                        ion_prob_data = shell_prob_data[ion_variant]
                        
                        if shell in ion_prob_data and isinstance(ion_prob_data[shell], dict):
                            shell_data = ion_prob_data[shell]
                            env_probs = {}
                            
                            # Extract environment probabilities
                            for env_name, prob in shell_data.items():
                                if isinstance(prob, (int, float)):
                                    # FIXED: Check if probability is already a percentage
                                    if prob > 1:
                                        env_probs[env_name] = prob
                                    else:
                                        env_probs[env_name] = prob * 100  # Convert to percentage
                            break
            
            # Store the data if found
            if env_probs:
                # NEW: Apply threshold filtering and log all data
                salt_label = self.salt_data[salt_key]['label']
                filtered_env_probs = {}
                
                for env_name, prob in sorted(env_probs.items()):
                    if prob >= min_probability_threshold:
                        filtered_env_probs[env_name] = prob
                        status = "INCLUDED"
                    else:
                        status = "FILTERED"
                    
                    log_lines.append(f"{salt_label:<20} {env_name:<15} {prob:<15.2f} {status:<10}")
                
                comparison_data[salt_key] = filtered_env_probs
                
                # Log filtering summary
                n_total = len(env_probs)
                n_filtered = len(filtered_env_probs)
                n_removed = n_total - n_filtered
                
                log_lines.append(f"{salt_label} SUMMARY: {n_filtered}/{n_total} environments kept, {n_removed} filtered (threshold: {min_probability_threshold}%)")
                
            else:
                print(f"WARNING: No shell coordination environment data found for {self.salt_data[salt_key]['label']}")
                comparison_data[salt_key] = {}
        
        log_lines.append("")
        
        # NEW: Add filtered data summary
        log_lines.append("FILTERED COORDINATION ENVIRONMENT DATA")
        log_lines.append("-"*80)
        log_lines.append(f"{'Salt':<20} {'Environment':<15} {'Probability (%)':<15}")
        log_lines.append("-"*80)
        
        for salt_key in [salt for pair in salt_pairs for salt in pair]:
            if salt_key in comparison_data and comparison_data[salt_key]:
                salt_label = self.salt_data[salt_key]['label']
                for env_name, prob in sorted(comparison_data[salt_key].items()):
                    log_lines.append(f"{salt_label:<20} {env_name:<15} {prob:<15.2f}")
        
        log_lines.append("")
        
        # Check if we have valid data to plot
        valid_salts = [salt_key for salt_key in comparison_data.keys() if comparison_data[salt_key]]
        
        if len(valid_salts) < 2:
            print("ERROR: Need at least 2 salts with valid coordination environment data")
            print(f"Available data: {[self.salt_data[k]['label'] for k in valid_salts]}")
            
            # NEW: Suggest lowering threshold if no data passes
            if min_probability_threshold > 0:
                print(f"SUGGESTION: Try lowering min_probability_threshold from {min_probability_threshold}% to include more environments")
            
            return comparison_data
        
        # Create comparison plot for each salt pair
        for i, (salt1_key, salt2_key) in enumerate(salt_pairs):
            
            if salt1_key not in valid_salts or salt2_key not in valid_salts:
                log_lines.append(f"WARNING: Skipping {salt1_key} vs {salt2_key} - missing data")
                continue
            
            salt1_label = self.salt_data[salt1_key]['label']
            salt2_label = self.salt_data[salt2_key]['label']
            salt1_color = self.salt_data[salt1_key].get('color', 'blue')
            salt2_color = self.salt_data[salt2_key].get('color', 'red')
            
            # Get all unique coordination environments from both salts (after filtering)
            all_environments = set()
            all_environments.update(comparison_data[salt1_key].keys())
            all_environments.update(comparison_data[salt2_key].keys())
            
            # Sort environments naturally (handle different formats)
            def sort_env_key(env_name):
                # Handle formats like "0-6", "1-5", "Shell_1", etc.
                import re
                
                # Look for patterns like "number-number" (coions-waters format)
                match = re.search(r'(\d+)-(\d+)', env_name)
                if match:
                    coions = int(match.group(1))
                    waters = int(match.group(2))
                    return (coions, waters)
                
                # Look for patterns like "Shell_number"
                match = re.search(r'Shell_(\d+)', env_name)
                if match:
                    return (0, int(match.group(1)))
                
                # Fallback to string sorting
                return (999, 999, env_name)
            
            sorted_environments = sorted(all_environments, key=sort_env_key)
            
            if not sorted_environments:
                print(f"No coordination environments found for {salt1_label} vs {salt2_label} after filtering (threshold: {min_probability_threshold}%)")
                print(f"SUGGESTION: Try lowering min_probability_threshold to include more environments")
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize, facecolor='white')
            
            # Prepare data for plotting
            n_envs = len(sorted_environments)
            x_positions = np.arange(n_envs) * group_spacing  # Spacing between environment groups
            
            salt1_probs = []
            salt2_probs = []
            x_labels = []
            
            for env_name in sorted_environments:
                # Get probabilities for each salt (0 if environment not present)
                salt1_prob = comparison_data[salt1_key].get(env_name, 0.0)
                salt2_prob = comparison_data[salt2_key].get(env_name, 0.0)
                
                salt1_probs.append(salt1_prob)
                salt2_probs.append(salt2_prob)
                
                # Clean up environment name for x-axis label
                x_labels.append(env_name)
            
            # Convert to numpy arrays
            salt1_probs = np.array(salt1_probs)
            salt2_probs = np.array(salt2_probs)
            
            # Calculate bar positions (side by side within each environment group)
            width = bar_width
            x1 = x_positions - width/2  # Salt 1 positions (left)
            x2 = x_positions + width/2  # Salt 2 positions (right)
            
            # Plot bars
            bars1 = ax.bar(x1, salt1_probs, width, alpha=alpha, color=salt1_color, 
                        label=salt1_label, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x2, salt2_probs, width, alpha=alpha, color=salt2_color, 
                        label=salt2_label, edgecolor='black', linewidth=0.5)
            
            # Customize plot
            ax.set_xlabel('Shell Type (Coordination Environment)', fontsize=label_fontsize)
            ax.set_ylabel('Probability (%)', fontsize=label_fontsize)
            
            # NEW: Include threshold in title
            title_text = f'{ion_type} Shell Coordination Environments ({shell})\n'
            title_text += f'{salt1_label} vs {salt2_label} at {concentration}'
            if min_probability_threshold > 0:
                title_text += f' (≥{min_probability_threshold}% threshold)'
            
            ax.set_title(title_text, fontsize=title_fontsize, fontweight='bold')
            
            # Set x-axis ticks and labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=tick_fontsize)
            
            # Add legend and styling
            ax.legend(fontsize=legend_fontsize, frameon=False, loc='best')
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            
            # Set y-axis to show percentages
            max_prob = max(np.max(salt1_probs), np.max(salt2_probs))
            if max_prob > 0:
                ax.set_ylim(0, max_prob * 1.1)
            else:
                ax.set_ylim(0, 100)
            
            # Add value labels on top of bars
            def add_value_labels(bars, values):
                for bar, value in zip(bars, values):
                    if value > 0.1:  # Only show labels for non-zero values
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max_prob*0.01,
                            f'{value:.1f}%', ha='center', va='bottom', 
                            fontsize=tick_fontsize-1, fontweight='bold')
            
            add_value_labels(bars1, salt1_probs)
            add_value_labels(bars2, salt2_probs)
            
            plt.tight_layout()
            
            # Save plot
            if save_plots:
                # NEW: Include threshold in filename
                threshold_suffix = f'_thresh{min_probability_threshold}p' if min_probability_threshold > 0 else ''
                filename = f'shell_coord_env_comparison_{ion_type}_{shell}_{salt1_key}_vs_{salt2_key}_{concentration.replace(".", "p")}{threshold_suffix}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                log_lines.append(f"PLOT SAVED: {filename}")
                print(f"Shell coordination environment comparison plot saved: {filename}")
            
            plt.show()
            
            # Print verification - check if probabilities sum to ~100%
            print(f"\nProbability verification for {ion_type} at {concentration} (after {min_probability_threshold}% threshold):")
            print(f"  {salt1_label}: Total = {sum(salt1_probs):.1f}% (filtered environments only)")
            print(f"  {salt2_label}: Total = {sum(salt2_probs):.1f}% (filtered environments only)")
            
            # NEW: Calculate and show percentage of total probability retained
            # Get original totals before filtering
            orig_salt1_total = sum(env_probs.get(env, 0) for env in comparison_data[salt1_key].keys()) if salt1_key in comparison_data else 0
            orig_salt2_total = sum(env_probs.get(env, 0) for env in comparison_data[salt2_key].keys()) if salt2_key in comparison_data else 0
            
            if orig_salt1_total > 0:
                retention1 = (sum(salt1_probs) / orig_salt1_total) * 100
                print(f"  {salt1_label}: Retains {retention1:.1f}% of original probability mass")
            
            if orig_salt2_total > 0:
                retention2 = (sum(salt2_probs) / orig_salt2_total) * 100
                print(f"  {salt2_label}: Retains {retention2:.1f}% of original probability mass")
            
            # Warning if low retention
            if min_probability_threshold > 0 and (
                (orig_salt1_total > 0 and (sum(salt1_probs) / orig_salt1_total) < 0.8) or
                (orig_salt2_total > 0 and (sum(salt2_probs) / orig_salt2_total) < 0.8)
            ):
                print(f"  ⚠️  WARNING: High threshold ({min_probability_threshold}%) removed significant probability mass")
                print(f"     Consider lowering threshold to retain more coordination environments")
        
        # Save log
        if save_log:
            # NEW: Include threshold in log filename
            threshold_suffix = f'_thresh{min_probability_threshold}p' if min_probability_threshold > 0 else ''
            log_filename = f'shell_coord_env_comparison_{ion_type}_{shell}_{concentration.replace(".", "p")}{threshold_suffix}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Log saved: {log_filename}")
        
        return comparison_data
    
    def compare_polyhedron_sizes_by_salt(self, ion_type, concentration='0.11M', 
                                    salt_pairs=None, save_plots=True, save_log=True,
                                    figsize=(14, 8), alpha=0.7,
                                    title_fontsize=14, label_fontsize=12,
                                    legend_fontsize=10, tick_fontsize=10,
                                    bar_width=0.35, group_spacing=1.0):
        '''
        Compare polyhedron sizes (volume and area) across different salts at a specific concentration.
        Shows mean polyhedron volumes and areas for a specific ion type across different salt environments.
        
        Parameters
        ----------
        ion_type : str
            Ion to analyze (e.g., 'Na', 'K', 'Mg', 'Ca', 'Cl')
        concentration : str
            Concentration to analyze (e.g., '0.11M', '0.2M')
        salt_pairs : list of tuples, optional
            Pairs of salts to compare, e.g., [('NaCl', 'KCl'), ('CaCl2', 'MgCl2')]
            If None, compares first available pair based on ion_type
        save_plots : bool
            Whether to save the plots
        save_log : bool
            Whether to save analysis log
        figsize : tuple
            Figure size (width, height)
        alpha : float
            Transparency for bars
        title_fontsize : int
            Font size for titles
        label_fontsize : int
            Font size for axis labels
        legend_fontsize : int
            Font size for legend
        tick_fontsize : int
            Font size for tick labels
        bar_width : float
            Width of individual bars
        group_spacing : float
            Spacing between salt groups
            
        Returns
        -------
        comparison_data : dict
            Dictionary with polyhedron size data for each salt
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_salts() first.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"POLYHEDRON SIZE COMPARISON: {ion_type} at {concentration}")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Concentration: {concentration}")
        log_lines.append("")
        
        # Determine salt pairs if not provided
        if salt_pairs is None:
            available_salts = list(self.loaded_data.keys())
            excluded_salts = {'No_Salt', 'Pure_Water', 'Water', 'no_salt', 'pure_water'}
            available_salts = [s for s in available_salts if s not in excluded_salts]
            
            # Auto-determine appropriate salt pairs based on ion_type
            if ion_type in ['Na', 'K', 'Li', 'Rb']:  # Monovalent cations
                # Look for monovalent salt pairs
                monovalent_salts = []
                for salt_key in available_salts:
                    label_normalized = self.salt_data[salt_key]['label'].upper().replace('₂', '2')
                    if 'CL' in label_normalized and '2' not in label_normalized:
                        monovalent_salts.append(salt_key)
                
                if len(monovalent_salts) >= 2:
                    salt_pairs = [(monovalent_salts[0], monovalent_salts[1])]
                else:
                    print(f"ERROR: Need at least 2 monovalent salts for {ion_type} comparison")
                    return None
                    
            elif ion_type in ['Mg', 'Ca', 'Sr', 'Ba']:  # Divalent cations
                # Look for divalent salt pairs
                divalent_salts = []
                for salt_key in available_salts:
                    label_normalized = self.salt_data[salt_key]['label'].upper().replace('₂', '2')
                    if 'CL2' in label_normalized or 'CACL2' in label_normalized or 'MGCL2' in label_normalized:
                        divalent_salts.append(salt_key)
                
                if len(divalent_salts) >= 2:
                    salt_pairs = [(divalent_salts[0], divalent_salts[1])]
                else:
                    print(f"ERROR: Need at least 2 divalent salts for {ion_type} comparison")
                    return None
                    
            elif ion_type in ['Cl', 'Br', 'F', 'I']:  # Anions
                # For anions, can compare across any salt types
                if len(available_salts) >= 2:
                    salt_pairs = [(available_salts[0], available_salts[1])]
                else:
                    print(f"ERROR: Need at least 2 salts for {ion_type} comparison")
                    return None
            else:
                print(f"ERROR: Unknown ion type {ion_type} or insufficient salts for comparison")
                return None
        
        log_lines.append(f"Salt pairs to compare:")
        for pair in salt_pairs:
            salt1_label = self.salt_data[pair[0]]['label']
            salt2_label = self.salt_data[pair[1]]['label']
            log_lines.append(f"  {salt1_label} vs {salt2_label}")
        log_lines.append("")
        
        # Collect polyhedron size data
        comparison_data = {}
        
        log_lines.append("POLYHEDRON SIZE DATA")
        log_lines.append("-"*80)
        log_lines.append(f"{'Salt':<20} {'Volume (Å³)':<15} {'Area (Å²)':<15} {'N_ions':<10}")
        log_lines.append("-"*80)
        
        for salt_key in [salt for pair in salt_pairs for salt in pair]:
            if salt_key not in self.loaded_data:
                continue
            
            if concentration not in self.loaded_data[salt_key]:
                print(f"WARNING: {concentration} not found for {self.salt_data[salt_key]['label']}")
                continue
            
            conc_data = self.loaded_data[salt_key][concentration]
            comparison_data[salt_key] = {}
            
            # Look for polyhedron size data
            polyhedron_data = None
            
            # Check for polyhedron_results_by_type (preferred - ion-type specific)
            if 'polyhedron_results_by_type' in conc_data:
                polyhedron_by_type = conc_data['polyhedron_results_by_type']
                
                if ion_type in polyhedron_by_type:
                    poly_results = polyhedron_by_type[ion_type]
                    if poly_results is not None:
                        polyhedron_data = {
                            'mean_volume': poly_results.get('overall_mean_volume', 0),
                            'std_volume': poly_results.get('overall_std_volume', 0),
                            'mean_area': poly_results.get('overall_mean_area', 0),
                            'std_area': poly_results.get('overall_std_area', 0),
                            'n_ions': poly_results.get('n_ions', 0)
                        }
            
            # Fallback to general polyhedron_sizes
            if polyhedron_data is None and 'polyhedron_sizes' in conc_data:
                poly_sizes = conc_data['polyhedron_sizes']
                
                # Look for ion-type specific data in general polyhedron results
                if ion_type in poly_sizes:
                    poly_info = poly_sizes[ion_type]
                    polyhedron_data = {
                        'mean_volume': poly_info.get('overall_mean_volume', poly_info.get('mean_volume', 0)),
                        'std_volume': poly_info.get('overall_std_volume', poly_info.get('std_volume', 0)),
                        'mean_area': poly_info.get('overall_mean_area', poly_info.get('mean_area', 0)),
                        'std_area': poly_info.get('overall_std_area', poly_info.get('std_area', 0)),
                        'n_ions': poly_info.get('n_ions', 1)
                    }
            
            # Store the data if found
            if polyhedron_data:
                comparison_data[salt_key] = polyhedron_data
                
                # Log the data
                salt_label = self.salt_data[salt_key]['label']
                log_lines.append(f"{salt_label:<20} {polyhedron_data['mean_volume']:<15.2f} {polyhedron_data['mean_area']:<15.2f} {polyhedron_data['n_ions']:<10}")
            else:
                print(f"WARNING: No polyhedron size data found for {ion_type} in {self.salt_data[salt_key]['label']}")
                comparison_data[salt_key] = {}
        
        log_lines.append("")
        
        # Check if we have valid data to plot
        valid_salts = [salt_key for salt_key in comparison_data.keys() if comparison_data[salt_key]]
        
        if len(valid_salts) < 2:
            print("ERROR: Need at least 2 salts with valid polyhedron data")
            print(f"Available data: {[self.salt_data[k]['label'] for k in valid_salts]}")
            return comparison_data
        
        # Create comparison plot for each salt pair
        for i, (salt1_key, salt2_key) in enumerate(salt_pairs):
            
            if salt1_key not in valid_salts or salt2_key not in valid_salts:
                log_lines.append(f"WARNING: Skipping {salt1_key} vs {salt2_key} - missing data")
                continue
            
            salt1_label = self.salt_data[salt1_key]['label']
            salt2_label = self.salt_data[salt2_key]['label']
            salt1_color = self.salt_data[salt1_key].get('color', 'blue')
            salt2_color = self.salt_data[salt2_key].get('color', 'red')
            
            # Get polyhedron data
            salt1_data = comparison_data[salt1_key]
            salt2_data = comparison_data[salt2_key]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize, facecolor='white')
            
            # Prepare data for plotting
            categories = ['Volume (Å³)', 'Area (Å²)']
            salt1_values = [salt1_data['mean_volume'], salt1_data['mean_area']]
            salt2_values = [salt2_data['mean_volume'], salt2_data['mean_area']]
            salt1_errors = [salt1_data['std_volume'], salt1_data['std_area']]
            salt2_errors = [salt2_data['std_volume'], salt2_data['std_area']]
            
            # Set up bar positions
            x = np.arange(len(categories))
            width = bar_width
            
            # Plot bars
            bars1 = ax.bar(x - width/2, salt1_values, width, yerr=salt1_errors,
                        alpha=alpha, color=salt1_color, label=salt1_label, 
                        capsize=5, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, salt2_values, width, yerr=salt2_errors,
                        alpha=alpha, color=salt2_color, label=salt2_label, 
                        capsize=5, edgecolor='black', linewidth=0.5)
            
            # Customize plot
            ax.set_xlabel('Polyhedron Property', fontsize=label_fontsize)
            ax.set_ylabel('Size', fontsize=label_fontsize)
            ax.set_title(f'{ion_type} Polyhedron Sizes\n'
                        f'{salt1_label} vs {salt2_label} at {concentration}',
                        fontsize=title_fontsize, fontweight='bold')
            
            # Set x-axis ticks and labels
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=tick_fontsize)
            
            # Add legend and styling
            ax.legend(fontsize=legend_fontsize, frameon=False, loc='best')
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            
            # Set y-axis to start from 0
            max_val = max(max(salt1_values), max(salt2_values))
            max_err = max(max(salt1_errors), max(salt2_errors))
            ax.set_ylim(0, (max_val + max_err) * 1.15)
            
            # Add value labels on top of bars
            def add_value_labels(bars, values, errors):
                for bar, value, error in zip(bars, values, errors):
                    if value > 0.1:  # Only show labels for non-zero values
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + error + max_val*0.01,
                            f'{value:.1f}', ha='center', va='bottom', 
                            fontsize=tick_fontsize-1, fontweight='bold')
            
            add_value_labels(bars1, salt1_values, salt1_errors)
            add_value_labels(bars2, salt2_values, salt2_errors)
            
            plt.tight_layout()
            
            # Save plot
            if save_plots:
                filename = f'polyhedron_size_comparison_{ion_type}_{salt1_key}_vs_{salt2_key}_{concentration.replace(".", "p")}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                log_lines.append(f"PLOT SAVED: {filename}")
                print(f"Polyhedron size comparison plot saved: {filename}")
            
            plt.show()
            
            # Print statistical comparison
            print(f"\nPolyhedron size comparison for {ion_type} at {concentration}:")
            print(f"  {salt1_label}:")
            print(f"    Volume: {salt1_data['mean_volume']:.1f} ± {salt1_data['std_volume']:.1f} Å³")
            print(f"    Area: {salt1_data['mean_area']:.1f} ± {salt1_data['std_area']:.1f} Å²")
            print(f"  {salt2_label}:")
            print(f"    Volume: {salt2_data['mean_volume']:.1f} ± {salt2_data['std_volume']:.1f} Å³")
            print(f"    Area: {salt2_data['mean_area']:.1f} ± {salt2_data['std_area']:.1f} Å²")
            
            # Calculate percentage differences
            vol_diff = ((salt2_data['mean_volume'] - salt1_data['mean_volume']) / salt1_data['mean_volume']) * 100
            area_diff = ((salt2_data['mean_area'] - salt1_data['mean_area']) / salt1_data['mean_area']) * 100
            
            print(f"  Relative differences ({salt2_label} vs {salt1_label}):")
            print(f"    Volume: {vol_diff:+.1f}%")
            print(f"    Area: {area_diff:+.1f}%")
        
        # Save log
        if save_log:
            log_filename = f'polyhedron_size_comparison_{ion_type}_{concentration.replace(".", "p")}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Log saved: {log_filename}")
        
        return comparison_data    
    
    # def compare_ion_pairing_probabilities_by_coordination_by_salt(self, ion_type, concentration='0.11M',
    #                                                         salt_pairs=None, save_plots=True, save_log=True,
    #                                                         figsize=(14, 8), alpha=0.8,
    #                                                         title_fontsize=14, label_fontsize=12,
    #                                                         legend_fontsize=10, tick_fontsize=10,
    #                                                         plot_type='stacked_bar',  # 'stacked_bar', 'grouped_bar', 'heatmap'
    #                                                         min_coordination_threshold=0,
    #                                                         max_coordination_threshold=None,
    #                                                         min_probability_threshold=0.5):
    #     '''
    #     Compare ion pairing probabilities by coordination number across different salts.
    #     Shows how coordination state affects pairing behavior for different salt environments.
        
    #     Parameters
    #     ----------
    #     ion_type : str
    #         Ion to analyze (e.g., 'Na', 'K', 'Mg', 'Ca', 'Cl')
    #     concentration : str
    #         Concentration to analyze (e.g., '0.11M', '0.2M')
    #     salt_pairs : list of tuples, optional
    #         Pairs of salts to compare, e.g., [('NaCl', 'KCl')]
    #     save_plots : bool
    #         Whether to save the plots
    #     save_log : bool
    #         Whether to save analysis log
    #     figsize : tuple
    #         Figure size (width, height)
    #     alpha : float
    #         Transparency for bars
    #     title_fontsize : int
    #         Font size for titles
    #     label_fontsize : int
    #         Font size for axis labels
    #     legend_fontsize : int
    #         Font size for legend
    #     tick_fontsize : int
    #         Font size for tick labels
    #     plot_type : str, default='stacked_bar'
    #         Type of visualization:
    #         - 'stacked_bar': Stacked bars showing pairing state distribution for each CN
    #         - 'grouped_bar': Side-by-side bars for each pairing state
    #         - 'heatmap': 2D heatmap of coordination vs pairing probabilities
    #     min_coordination_threshold : int, default=0
    #         Minimum coordination number to include in analysis
    #     max_coordination_threshold : int, optional
    #         Maximum coordination number to include in analysis (None = no limit)
    #     min_probability_threshold : float, default=0.5
    #         Minimum probability threshold (%) for coordination states to include
            
    #     Returns
    #     -------
    #     comparison_data : dict
    #         Dictionary with coordination-pairing data for each salt
    #     '''
        
    #     if not self.loaded_data:
    #         print("ERROR: No data loaded. Run load_all_salts() first.")
    #         return None
        
    #     # Initialize log
    #     log_lines = []
    #     log_lines.append("="*80)
    #     log_lines.append(f"ION PAIRING BY COORDINATION COMPARISON: {ion_type} at {concentration}")
    #     log_lines.append("="*80)
    #     log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    #     log_lines.append(f"Ion Type: {ion_type}")
    #     log_lines.append(f"Concentration: {concentration}")
    #     log_lines.append(f"Plot Type: {plot_type}")
    #     log_lines.append(f"Min Coordination Threshold: {min_coordination_threshold}")
    #     log_lines.append(f"Max Coordination Threshold: {max_coordination_threshold if max_coordination_threshold else 'None'}")
    #     log_lines.append(f"Min Probability Threshold: {min_probability_threshold}%")
    #     log_lines.append("")
        
    #     # Determine salt pairs if not provided
    #     if salt_pairs is None:
    #         available_salts = list(self.loaded_data.keys())
    #         excluded_salts = {'No_Salt', 'Pure_Water', 'Water', 'no_salt', 'pure_water'}
    #         available_salts = [s for s in available_salts if s not in excluded_salts]
            
    #         # Auto-determine appropriate salt pairs based on ion_type
    #         if ion_type in ['Na', 'K', 'Li', 'Rb']:  # Monovalent cations
    #             monovalent_salts = []
    #             for salt_key in available_salts:
    #                 label_normalized = self.salt_data[salt_key]['label'].upper().replace('₂', '2')
    #                 if 'CL' in label_normalized and '2' not in label_normalized:
    #                     monovalent_salts.append(salt_key)
                
    #             if len(monovalent_salts) >= 2:
    #                 salt_pairs = [(monovalent_salts[0], monovalent_salts[1])]
    #             else:
    #                 print(f"ERROR: Need at least 2 monovalent salts for {ion_type} comparison")
    #                 return None
                    
    #         elif ion_type in ['Mg', 'Ca', 'Sr', 'Ba']:  # Divalent cations
    #             divalent_salts = []
    #             for salt_key in available_salts:
    #                 label_normalized = self.salt_data[salt_key]['label'].upper().replace('₂', '2')
    #                 if 'CL2' in label_normalized or 'CACL2' in label_normalized or 'MGCL2' in label_normalized:
    #                     divalent_salts.append(salt_key)
                
    #             if len(divalent_salts) >= 2:
    #                 salt_pairs = [(divalent_salts[0], divalent_salts[1])]
    #             else:
    #                 print(f"ERROR: Need at least 2 divalent salts for {ion_type} comparison")
    #                 return None
                    
    #         elif ion_type in ['Cl', 'Br', 'F', 'I']:  # Anions
    #             if len(available_salts) >= 2:
    #                 salt_pairs = [(available_salts[0], available_salts[1])]
    #             else:
    #                 print(f"ERROR: Need at least 2 salts for {ion_type} comparison")
    #                 return None
    #         else:
    #             print(f"ERROR: Unknown ion type {ion_type} or insufficient salts for comparison")
    #             return None
        
    #     log_lines.append(f"Salt pairs to compare:")
    #     for pair in salt_pairs:
    #         salt1_label = self.salt_data[pair[0]]['label']
    #         salt2_label = self.salt_data[pair[1]]['label']
    #         log_lines.append(f"  {salt1_label} vs {salt2_label}")
    #     log_lines.append("")
        
    #     # Collect ion pairing by coordination data
    #     comparison_data = {}
        
    #     log_lines.append("ION PAIRING BY COORDINATION DATA")
    #     log_lines.append("-"*80)
    #     log_lines.append(f"{'Salt':<15} {'CN':<5} {'CIP (%)':<10} {'SIP (%)':<10} {'DSIP (%)':<10} {'FI (%)':<10} {'CN_Prob (%)':<12}")
    #     log_lines.append("-"*80)
        

    #     for salt_key in [salt for pair in salt_pairs for salt in pair]:
    #         if salt_key not in self.loaded_data:
    #             continue
            
    #         if concentration not in self.loaded_data[salt_key]:
    #             print(f"WARNING: {concentration} not found for {self.salt_data[salt_key]['label']}")
    #             continue
            
    #         conc_data = self.loaded_data[salt_key][concentration]
    #         comparison_data[salt_key] = {}
            
    #         # FIXED: Method 1: Look for ion_pairing_probabilities_by_coordination
    #         if 'ion_pairing_probabilities_by_coordination' in conc_data:
    #             pairing_coord_data = conc_data['ion_pairing_probabilities_by_coordination']
                
    #             # Try to find ion-specific data
    #             for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
    #                 if ion_variant in pairing_coord_data:
    #                     ion_data = pairing_coord_data[ion_variant]
                        
    #                     # FIXED: Direct extraction from the actual data structure
    #                     if isinstance(ion_data, dict):
    #                         processed_data = {}
                            
    #                         for cn_key, cn_data in ion_data.items():
    #                             # cn_key is the coordination number (as integer)
    #                             if isinstance(cn_key, (int, float)):
    #                                 cn_number = int(cn_key)
                                    
    #                                 # Apply coordination thresholds
    #                                 if cn_number < min_coordination_threshold:
    #                                     continue
                                    
    #                                 # NEW: Apply max coordination threshold
    #                                 if max_coordination_threshold is not None and cn_number > max_coordination_threshold:
    #                                     continue
                                    
    #                                 # Extract from the nested structure: cn_data = {'probabilities': {...}, 'total_observations': ...}
    #                                 if isinstance(cn_data, dict) and 'probabilities' in cn_data:
    #                                     pairing_probs = cn_data['probabilities']
    #                                     total_obs = cn_data.get('total_observations', 1)
                                        
    #                                     # Extract pairing probabilities (already as fractions 0-1, convert to percentages)
    #                                     cip_prob = pairing_probs.get('CIP', 0) * 100
    #                                     sip_prob = pairing_probs.get('SIP', 0) * 100
    #                                     dsip_prob = pairing_probs.get('DSIP', 0) * 100
    #                                     fi_prob = pairing_probs.get('FI', 0) * 100
                                        
    #                                     # For now, set CN probability to 100% (you could calculate this from shell_coordination_numbers if needed)
    #                                     cn_prob = 100.0
                                        
    #                                     # Apply probability threshold to coordination number
    #                                     if cn_prob >= min_probability_threshold:
    #                                         processed_data[cn_number] = {
    #                                             'CIP': cip_prob,
    #                                             'SIP': sip_prob,
    #                                             'DSIP': dsip_prob,
    #                                             'FI': fi_prob,
    #                                             'CN_probability': cn_prob,
    #                                             'total_observations': total_obs
    #                                         }
                            
    #                         if processed_data:
    #                             # FIXED: Store directly in comparison_data
    #                             salt_label = self.salt_data[salt_key]['label']
                                
    #                             for cn_number, data in processed_data.items():
    #                                 # Log the data
    #                                 log_lines.append(f"{salt_label:<15} {cn_number:<5} {data['CIP']:<10.1f} {data['SIP']:<10.1f} {data['DSIP']:<10.1f} {data['FI']:<10.1f} {data['CN_probability']:<12.1f}")
                                
    #                             comparison_data[salt_key] = processed_data
    #                             break  # Exit the ion_variant loop
                
    #             # If we found data, skip to next salt
    #             if salt_key in comparison_data and comparison_data[salt_key]:
    #                 continue
            
    #         # Method 2: Try to construct from separate coordination and pairing data
    #         # (This is a fallback - try to combine shell_coordination_numbers and ion_pairing_probabilities)
    #         coord_data = None
    #         pairing_data = None
            
    #         # Get coordination number distribution
    #         if 'shell_coordination_numbers' in conc_data:
    #             cn_dict = conc_data['shell_coordination_numbers']
    #             for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
    #                 if ion_variant in cn_dict:
    #                     coord_data = cn_dict[ion_variant]
    #                     break
            
    #         # Get pairing probabilities
    #         if 'ion_pairing_probabilities' in conc_data:
    #             pair_dict = conc_data['ion_pairing_probabilities']
    #             for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
    #                 if ion_variant in pair_dict:
    #                     pairing_data = pair_dict[ion_variant]
    #                     break
            
    #         # If we have both, try to create synthetic coordination-pairing matrix
    #         if coord_data is not None and pairing_data is not None:
    #             print(f"INFO: Constructing coordination-pairing matrix from separate data for {self.salt_data[salt_key]['label']}")
    #             # This would require more complex logic to correlate CN and pairing states
    #             # For now, we'll skip this salt
    #             print(f"WARNING: Cannot construct coordination-pairing matrix for {self.salt_data[salt_key]['label']}")
            
    #         # If no data found by this point, comparison_data[salt_key] remains empty
    #         if salt_key not in comparison_data or not comparison_data[salt_key]:
    #             print(f"WARNING: No coordination-pairing data found for {self.salt_data[salt_key]['label']}")
    #             comparison_data[salt_key] = {}
        
    #     log_lines.append("")
        
    #     # Check if we have valid data to plot
    #     valid_salts = [salt_key for salt_key in comparison_data.keys() if comparison_data[salt_key]]
        
    #     if len(valid_salts) < 2:
    #         print("ERROR: Need at least 2 salts with valid coordination-pairing data")
    #         print(f"Available data: {[self.salt_data[k]['label'] for k in valid_salts]}")
    #         return comparison_data
        
    #     # Create comparison plots for each salt pair
    #     for i, (salt1_key, salt2_key) in enumerate(salt_pairs):
            
    #         if salt1_key not in valid_salts or salt2_key not in valid_salts:
    #             log_lines.append(f"WARNING: Skipping {salt1_key} vs {salt2_key} - missing data")
    #             continue
            
    #         salt1_label = self.salt_data[salt1_key]['label']
    #         salt2_label = self.salt_data[salt2_key]['label']
    #         salt1_color = self.salt_data[salt1_key].get('color', 'blue')
    #         salt2_color = self.salt_data[salt2_key].get('color', 'red')
            
    #         # Get all coordination numbers from both salts
    #         all_cn = set()
    #         all_cn.update(comparison_data[salt1_key].keys())
    #         all_cn.update(comparison_data[salt2_key].keys())
    #         sorted_cn = sorted(all_cn)
            
    #         if not sorted_cn:
    #             print(f"No coordination numbers found for {salt1_label} vs {salt2_label}")
    #             continue
            
    #         # Use the simple pairing colors you specified
    #         colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
    #         pair_types = ['CIP', 'SIP', 'DSIP', 'FI']
    #         pairing_colors = dict(zip(pair_types, colors))
            
    #         if plot_type == 'stacked_bar':
    #             # FIXED: Single figure with side-by-side stacked bars for both salts
    #             fig, ax = plt.subplots(figsize=figsize, facecolor='white')
                
    #             # Get data for both salts
    #             salt1_data = comparison_data[salt1_key]
    #             salt2_data = comparison_data[salt2_key]
                
    #             # Prepare data for side-by-side stacked bars
    #             n_cn = len(sorted_cn)
    #             x_positions = np.arange(n_cn)
    #             bar_width = 0.35  # Width of each stacked bar
                
    #             # Prepare data arrays for both salts
    #             salt1_cip = []
    #             salt1_sip = []
    #             salt1_dsip = []
    #             salt1_fi = []
    #             salt2_cip = []
    #             salt2_sip = []
    #             salt2_dsip = []
    #             salt2_fi = []
                
    #             for cn in sorted_cn:
    #                 # Salt 1 data
    #                 if cn in salt1_data:
    #                     salt1_cip.append(salt1_data[cn]['CIP'])
    #                     salt1_sip.append(salt1_data[cn]['SIP'])
    #                     salt1_dsip.append(salt1_data[cn]['DSIP'])
    #                     salt1_fi.append(salt1_data[cn]['FI'])
    #                 else:
    #                     salt1_cip.append(0)
    #                     salt1_sip.append(0)
    #                     salt1_dsip.append(0)
    #                     salt1_fi.append(0)
                    
    #                 # Salt 2 data
    #                 if cn in salt2_data:
    #                     salt2_cip.append(salt2_data[cn]['CIP'])
    #                     salt2_sip.append(salt2_data[cn]['SIP'])
    #                     salt2_dsip.append(salt2_data[cn]['DSIP'])
    #                     salt2_fi.append(salt2_data[cn]['FI'])
    #                 else:
    #                     salt2_cip.append(0)
    #                     salt2_sip.append(0)
    #                     salt2_dsip.append(0)
    #                     salt2_fi.append(0)
                
    #             # Convert to numpy arrays
    #             salt1_cip = np.array(salt1_cip)
    #             salt1_sip = np.array(salt1_sip)
    #             salt1_dsip = np.array(salt1_dsip)
    #             salt1_fi = np.array(salt1_fi)
    #             salt2_cip = np.array(salt2_cip)
    #             salt2_sip = np.array(salt2_sip)
    #             salt2_dsip = np.array(salt2_dsip)
    #             salt2_fi = np.array(salt2_fi)
                
    #             # Calculate x positions for side-by-side bars
    #             x1 = x_positions - bar_width/2  # Salt 1 positions (left)
    #             x2 = x_positions + bar_width/2  # Salt 2 positions (right)
                
    #             # Plot stacked bars for Salt 1 (no hatching)
    #             p1_1 = ax.bar(x1, salt1_cip, bar_width, label=f'{salt1_label} CIP', 
    #                         color=pairing_colors['CIP'], alpha=alpha, edgecolor='black', linewidth=0.5)
    #             p1_2 = ax.bar(x1, salt1_sip, bar_width, bottom=salt1_cip, label=f'{salt1_label} SIP',
    #                         color=pairing_colors['SIP'], alpha=alpha, edgecolor='black', linewidth=0.5)
    #             p1_3 = ax.bar(x1, salt1_dsip, bar_width, bottom=salt1_cip + salt1_sip, label=f'{salt1_label} DSIP',
    #                         color=pairing_colors['DSIP'], alpha=alpha, edgecolor='black', linewidth=0.5)
    #             p1_4 = ax.bar(x1, salt1_fi, bar_width, bottom=salt1_cip + salt1_sip + salt1_dsip, label=f'{salt1_label} FI',
    #                         color=pairing_colors['FI'], alpha=alpha, edgecolor='black', linewidth=0.5)
                
    #             # Plot stacked bars for Salt 2 (with hatching)
    #             p2_1 = ax.bar(x2, salt2_cip, bar_width, label=f'{salt2_label} CIP',
    #                         color=pairing_colors['CIP'], alpha=alpha, hatch='///', edgecolor='black', linewidth=0.5)
    #             p2_2 = ax.bar(x2, salt2_sip, bar_width, bottom=salt2_cip, label=f'{salt2_label} SIP',
    #                         color=pairing_colors['SIP'], alpha=alpha, hatch='///', edgecolor='black', linewidth=0.5)
    #             p2_3 = ax.bar(x2, salt2_dsip, bar_width, bottom=salt2_cip + salt2_sip, label=f'{salt2_label} DSIP',
    #                         color=pairing_colors['DSIP'], alpha=alpha, hatch='///', edgecolor='black', linewidth=0.5)
    #             p2_4 = ax.bar(x2, salt2_fi, bar_width, bottom=salt2_cip + salt2_sip + salt2_dsip, label=f'{salt2_label} FI',
    #                         color=pairing_colors['FI'], alpha=alpha, hatch='///', edgecolor='black', linewidth=0.5)
                
    #             # Customize plot
    #             ax.set_xlabel('Coordination Number', fontsize=label_fontsize)
    #             ax.set_ylabel('Probability (%)', fontsize=label_fontsize)
    #             ax.set_title(f'Ion Pairing by Coordination Number: {ion_type} at {concentration}\n'
    #                         f'{salt1_label} vs {salt2_label}',
    #                         fontsize=title_fontsize, fontweight='bold')
                
    #             # Set x-axis ticks and labels
    #             ax.set_xticks(x_positions)
    #             ax.set_xticklabels([f'CN {cn}' for cn in sorted_cn], fontsize=tick_fontsize)
    #             ax.tick_params(axis='both', labelsize=tick_fontsize)
    #             ax.set_ylim(0, 100)
    #             ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
                
    #             # FIXED: Better legend organization - group by pairing state
    #             from matplotlib.patches import Patch
    #             from matplotlib.lines import Line2D
                
    #             # Create custom legend elements
    #             legend_elements = []
                
    #             # Add pairing state colors
    #             for state in ['CIP', 'SIP', 'DSIP', 'FI']:
    #                 legend_elements.append(Patch(facecolor=pairing_colors[state], alpha=alpha, 
    #                                         edgecolor='black', linewidth=0.5, label=state))
                
    #             # Add separator
    #             legend_elements.append(Line2D([0], [0], color='white', linewidth=0, label=''))
                
    #             # Add salt distinction (solid vs hatched)
    #             legend_elements.append(Patch(facecolor='gray', alpha=alpha, 
    #                                     edgecolor='black', linewidth=0.5, label=f'{salt1_label} (solid)'))
    #             legend_elements.append(Patch(facecolor='gray', alpha=alpha, hatch='///',
    #                                     edgecolor='black', linewidth=0.5, label=f'{salt2_label} (hatched)'))
                
    #             ax.legend(handles=legend_elements, fontsize=legend_fontsize, frameon=False, loc='best', ncol=2)
                
    #             plt.tight_layout()
            
    #         elif plot_type == 'grouped_bar':
    #             # FIXED: Grouped bar chart with NO OVERLAPPING bars
    #             fig, ax = plt.subplots(figsize=figsize, facecolor='white')
                
    #             # Prepare grouped bar data
    #             pairing_states = ['CIP', 'SIP', 'DSIP', 'FI']
    #             x = np.arange(len(sorted_cn))
                
    #             # FIXED: Calculate proper bar width and spacing to avoid overlap
    #             n_salts = 2  # Number of salts being compared
    #             n_pairing_states = len(pairing_states)
    #             total_bars_per_cn = n_salts * n_pairing_states  # 2 salts × 4 pairing states = 8 bars per CN
                
    #             # Make bars narrower to fit all without overlap
    #             individual_bar_width = 0.8 / total_bars_per_cn  # 0.8 is the total space available per CN
    #             spacing = individual_bar_width * 0.1  # Small spacing between bars
                
    #             # Plot bars for each pairing state
    #             for ps_idx, pairing_state in enumerate(pairing_states):
    #                 salt1_values = []
    #                 salt2_values = []
                    
    #                 for cn in sorted_cn:
    #                     salt1_values.append(comparison_data[salt1_key].get(cn, {}).get(pairing_state, 0))
    #                     salt2_values.append(comparison_data[salt2_key].get(cn, {}).get(pairing_state, 0))
                    
    #                 # FIXED: Calculate x positions to avoid overlap
    #                 # Each pairing state gets 2 bars (one for each salt) side by side
    #                 bar_group_start = -0.4 + ps_idx * (2 * individual_bar_width + spacing * 2)
                    
    #                 x1 = x + bar_group_start  # Salt 1 position
    #                 x2 = x + bar_group_start + individual_bar_width + spacing  # Salt 2 position (next to salt 1)
                    
    #                 # Plot bars with no overlap
    #                 ax.bar(x1, salt1_values, individual_bar_width, alpha=alpha, 
    #                     color=pairing_colors[pairing_state], 
    #                     label=f'{salt1_label} {pairing_state}',
    #                     edgecolor='black', linewidth=0.5)
    #                 ax.bar(x2, salt2_values, individual_bar_width, alpha=alpha,
    #                     color=pairing_colors[pairing_state], 
    #                     label=f'{salt2_label} {pairing_state}',
    #                     hatch='///', edgecolor='black', linewidth=0.5)
                
    #             # Customize plot
    #             ax.set_xlabel('Coordination Number', fontsize=label_fontsize)
    #             ax.set_ylabel('Probability (%)', fontsize=label_fontsize)
    #             ax.set_title(f'Ion Pairing by Coordination Number: {ion_type}\n'
    #                         f'{salt1_label} vs {salt2_label} at {concentration}',
    #                         fontsize=title_fontsize, fontweight='bold')
    #             ax.set_xticks(x)
    #             ax.set_xticklabels([f'CN {cn}' for cn in sorted_cn], fontsize=tick_fontsize)
    #             ax.tick_params(axis='both', labelsize=tick_fontsize)
    #             ax.legend(fontsize=legend_fontsize-2, frameon=False, loc='best', ncol=2)
    #             ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
                
    #             plt.tight_layout()
            
    #         elif plot_type == 'heatmap':
    #             # FIXED: Heatmap showing coordination vs pairing probabilities with 'Blues' colormap
    #             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] + 4, figsize[1]), 
    #                                         facecolor='white')
                
    #             pairing_states = ['CIP', 'SIP', 'DSIP', 'FI']
                
    #             for ax_idx, (salt_key, salt_label, ax) in enumerate([(salt1_key, salt1_label, ax1), 
    #                                                             (salt2_key, salt2_label, ax2)]):
    #                 salt_data = comparison_data[salt_key]
                    
    #                 # Prepare heatmap data matrix
    #                 heatmap_data = np.zeros((len(sorted_cn), len(pairing_states)))
                    
    #                 for cn_idx, cn in enumerate(sorted_cn):
    #                     if cn in salt_data:
    #                         for ps_idx, pairing_state in enumerate(pairing_states):
    #                             heatmap_data[cn_idx, ps_idx] = salt_data[cn][pairing_state]
                    
    #                 # FIXED: Create heatmap with 'Blues' colormap like in EquilibriumAnalysisOptimized
    #                 im = ax.imshow(heatmap_data, cmap='Blues', aspect='auto', 
    #                             vmin=0, vmax=100, interpolation='nearest')
                    
    #                 # Add text annotations
    #                 for cn_idx in range(len(sorted_cn)):
    #                     for ps_idx in range(len(pairing_states)):
    #                         value = heatmap_data[cn_idx, ps_idx]
    #                         text = ax.text(ps_idx, cn_idx, f'{value:.1f}%',
    #                                     ha="center", va="center", 
    #                                     color="white" if value > 50 else "black",
    #                                     fontsize=tick_fontsize-2, fontweight='bold')
                    
    #                 # Customize heatmap
    #                 ax.set_xticks(np.arange(len(pairing_states)))
    #                 ax.set_xticklabels(pairing_states, fontsize=tick_fontsize)
    #                 ax.set_yticks(np.arange(len(sorted_cn)))
    #                 ax.set_yticklabels([f'CN {cn}' for cn in sorted_cn], fontsize=tick_fontsize)
    #                 ax.set_xlabel('Pairing State', fontsize=label_fontsize)
    #                 ax.set_ylabel('Coordination Number', fontsize=label_fontsize)
    #                 ax.set_title(f'{salt_label}', fontsize=title_fontsize, fontweight='bold')
                
    #             # Add colorbar
    #             cbar = plt.colorbar(im, ax=[ax1, ax2], shrink=0.8, pad=0.1)
    #             cbar.set_label('Probability (%)', fontsize=label_fontsize)
    #             cbar.ax.tick_params(labelsize=tick_fontsize)
                
    #             plt.suptitle(f'Ion Pairing by Coordination Heatmap: {ion_type} at {concentration}',
    #                         fontsize=title_fontsize + 2, fontweight='bold')
    #             plt.tight_layout()
            
    #         else:
    #             raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'stacked_bar', 'grouped_bar', or 'heatmap'")
            
    #         # Save plot
    #         if save_plots:
    #             threshold_suffix = f'_cn{min_coordination_threshold}'
    #             if max_coordination_threshold is not None:
    #                 threshold_suffix += f'to{max_coordination_threshold}'
    #             threshold_suffix += f'_prob{min_probability_threshold}p' if min_probability_threshold > 0 else ''
                
    #             filename = f'ion_pairing_coordination_comparison_{ion_type}_{salt1_key}_vs_{salt2_key}_{concentration.replace(".", "p")}_{plot_type}{threshold_suffix}.png'
    #             plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    #             log_lines.append(f"PLOT SAVED: {filename}")
    #             print(f"Ion pairing by coordination comparison plot saved: {filename}")
            
    #         plt.show()
            
    #         # Print statistical summary
    #         print(f"\nCoordination-Pairing Analysis for {ion_type} at {concentration}:")
    #         for salt_key, salt_label in [(salt1_key, salt1_label), (salt2_key, salt2_label)]:
    #             if salt_key in comparison_data and comparison_data[salt_key]:
    #                 print(f"\n  {salt_label}:")
    #                 salt_data = comparison_data[salt_key]
                    
    #                 # Calculate average pairing probabilities weighted by coordination probability
    #                 weighted_cip = 0
    #                 weighted_sip = 0
    #                 weighted_dsip = 0
    #                 weighted_fi = 0
    #                 total_weight = 0
                    
    #                 for cn, data in salt_data.items():
    #                     weight = data['CN_probability'] / 100
    #                     weighted_cip += data['CIP'] * weight
    #                     weighted_sip += data['SIP'] * weight
    #                     weighted_dsip += data['DSIP'] * weight
    #                     weighted_fi += data['FI'] * weight
    #                     total_weight += weight
                    
    #                 if total_weight > 0:
    #                     weighted_cip /= total_weight
    #                     weighted_sip /= total_weight
    #                     weighted_dsip /= total_weight
    #                     weighted_fi /= total_weight
                        
    #                     print(f"    Weighted Average Pairing Probabilities:")
    #                     print(f"      CIP: {weighted_cip:.1f}%")
    #                     print(f"      SIP: {weighted_sip:.1f}%") 
    #                     print(f"      DSIP: {weighted_dsip:.1f}%")
    #                     print(f"      FI: {weighted_fi:.1f}%")
                    
    #                 # Show coordination number range
    #                 cn_range = f"CN {min(salt_data.keys())}-{max(salt_data.keys())}"
    #                 print(f"    Coordination Range: {cn_range}")
    #                 print(f"    Number of CN States: {len(salt_data)}")
        
    #     # Save log
    #     if save_log:
    #         threshold_suffix = f'_cn{min_coordination_threshold}'
    #         if max_coordination_threshold is not None:
    #             threshold_suffix += f'to{max_coordination_threshold}'
    #         threshold_suffix += f'_prob{min_probability_threshold}p' if min_probability_threshold > 0 else ''
            
    #         log_filename = f'ion_pairing_coordination_comparison_{ion_type}_{concentration.replace(".", "p")}_{plot_type}{threshold_suffix}.log'
    #         with open(log_filename, 'w') as f:
    #             f.write('\n'.join(log_lines))
    #         print(f"Log saved: {log_filename}")
        
    #     return comparison_data

    def compare_ion_pairing_probabilities_by_coordination_by_salt(self, ion_type, concentration='0.11M',
                                                                salt_pairs=None, save_plots=True, save_log=True,
                                                                figsize=(14, 8), alpha=0.8,
                                                                title_fontsize=14, label_fontsize=12,
                                                                legend_fontsize=10, tick_fontsize=10,
                                                                plot_type='stacked_bar',  # 'stacked_bar', 'grouped_bar', 'heatmap'
                                                                min_coordination_threshold=0,
                                                                max_coordination_threshold=None,
                                                                min_probability_threshold=0.5,
                                                                # Enhanced broken axis parameters
                                                                enable_broken_axis=True,
                                                                broken_axis_threshold=15,  
                                                                broken_axis_min_gap=20,    
                                                                break_ratio=0.3,             # Default ratio (ignored if auto-calculated)
                                                                max_y_axis=None,              # Manual max y-axis limit
                                                                manual_break_point=None,     # Manual break point
                                                                manual_top_start=None,       # Manual top start
                                                                manual_break_ratio=None):    # NEW: Manual break ratio override
        '''
        Compare ion pairing probabilities by coordination number across different salts.
        Shows how coordination state affects pairing behavior for different salt environments.
        
        Parameters
        ----------
        ion_type : str
            Ion to analyze (e.g., 'Na', 'K', 'Mg', 'Ca', 'Cl')
        concentration : str
            Concentration to analyze (e.g., '0.11M', '0.2M')
        salt_pairs : list of tuples, optional
            Pairs of salts to compare, e.g., [('NaCl', 'KCl')]
        save_plots : bool
            Whether to save the plots
        save_log : bool
            Whether to save analysis log
        figsize : tuple
            Figure size (width, height)
        alpha : float
            Transparency for bars
        title_fontsize : int
            Font size for titles
        label_fontsize : int
            Font size for axis labels
        legend_fontsize : int
            Font size for legend
        tick_fontsize : int
            Font size for tick labels
        plot_type : str, default='stacked_bar'
            Type of visualization:
            - 'stacked_bar': Stacked bars showing pairing state distribution for each CN
            - 'grouped_bar': Side-by-side bars for each pairing state
            - 'heatmap': 2D heatmap of coordination vs pairing probabilities
        min_coordination_threshold : int, default=0
            Minimum coordination number to include in analysis
        max_coordination_threshold : int, optional
            Maximum coordination number to include in analysis (None = no limit)
        min_probability_threshold : float, default=0.5
            Minimum probability threshold (%) for coordination states to include
        enable_broken_axis : bool, default=True
            Whether to enable automatic broken y-axis when large gaps detected
        broken_axis_threshold : float, default=15
            Percentage gap threshold to trigger broken axis
        broken_axis_min_gap : float, default=20
            Minimum absolute gap (percentage points) required to apply broken axis
        break_ratio : float, default=0.3
            Default height ratio for bottom section (used only when auto-calculation fails)
        max_y_axis : float, optional
            Manual maximum y-axis limit. If provided, overrides automatic scaling.
            For broken axis: applies to top section maximum.
        manual_break_point : float, optional
            Manual break point position (%). If provided, overrides automatic detection.
            Only used when enable_broken_axis=True.
        manual_top_start : float, optional
            Manual top section start position (%). If provided, overrides automatic calculation.
            Only used when enable_broken_axis=True and manual_break_point is set.
        manual_break_ratio : float, optional
            Manual break ratio override (0.0-1.0). If provided, overrides automatic calculation.
            0.3 means bottom section gets 30% of figure height, top gets 70%.
            Only used when enable_broken_axis=True.
            
        Returns
        -------
        comparison_data : dict
            Dictionary with coordination-pairing data for each salt
        '''
        
        if not self.loaded_data:
            print("ERROR: No data loaded. Run load_all_salts() first.")
            return None
        
        # Initialize log
        log_lines = []
        log_lines.append("="*80)
        log_lines.append(f"ION PAIRING BY COORDINATION COMPARISON: {ion_type} at {concentration}")
        log_lines.append("="*80)
        log_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Ion Type: {ion_type}")
        log_lines.append(f"Concentration: {concentration}")
        log_lines.append(f"Plot Type: {plot_type}")
        log_lines.append(f"Min Coordination Threshold: {min_coordination_threshold}")
        log_lines.append(f"Max Coordination Threshold: {max_coordination_threshold if max_coordination_threshold else 'None'}")
        log_lines.append(f"Min Probability Threshold: {min_probability_threshold}%")
        log_lines.append(f"Max Y-Axis: {max_y_axis if max_y_axis else 'Auto'}")
        log_lines.append(f"Broken Axis Enabled: {enable_broken_axis}")
        if enable_broken_axis:
            log_lines.append(f"Broken Axis Gap Threshold: {broken_axis_threshold}%")
            log_lines.append(f"Broken Axis Min Gap: {broken_axis_min_gap}%")
            log_lines.append(f"Default Break Ratio: {break_ratio}")
            log_lines.append(f"Manual Break Point: {manual_break_point if manual_break_point else 'Auto'}")
            log_lines.append(f"Manual Top Start: {manual_top_start if manual_top_start else 'Auto'}")
            log_lines.append(f"Manual Break Ratio: {manual_break_ratio if manual_break_ratio else 'Auto'}")
        log_lines.append("")
        
        # Determine salt pairs if not provided
        if salt_pairs is None:
            available_salts = list(self.loaded_data.keys())
            excluded_salts = {'No_Salt', 'Pure_Water', 'Water', 'no_salt', 'pure_water'}
            available_salts = [s for s in available_salts if s not in excluded_salts]
            
            # Auto-determine appropriate salt pairs based on ion_type
            if ion_type in ['Na', 'K', 'Li', 'Rb']:  # Monovalent cations
                monovalent_salts = []
                for salt_key in available_salts:
                    label_normalized = self.salt_data[salt_key]['label'].upper().replace('₂', '2')
                    if 'CL' in label_normalized and '2' not in label_normalized:
                        monovalent_salts.append(salt_key)
                
                if len(monovalent_salts) >= 2:
                    salt_pairs = [(monovalent_salts[0], monovalent_salts[1])]
                else:
                    print(f"ERROR: Need at least 2 monovalent salts for {ion_type} comparison")
                    return None
                    
            elif ion_type in ['Mg', 'Ca', 'Sr', 'Ba']:  # Divalent cations
                divalent_salts = []
                for salt_key in available_salts:
                    label_normalized = self.salt_data[salt_key]['label'].upper().replace('₂', '2')
                    if 'CL2' in label_normalized or 'CACL2' in label_normalized or 'MGCL2' in label_normalized:
                        divalent_salts.append(salt_key)
                
                if len(divalent_salts) >= 2:
                    salt_pairs = [(divalent_salts[0], divalent_salts[1])]
                else:
                    print(f"ERROR: Need at least 2 divalent salts for {ion_type} comparison")
                    return None
                    
            elif ion_type in ['Cl', 'Br', 'F', 'I']:  # Anions
                if len(available_salts) >= 2:
                    salt_pairs = [(available_salts[0], available_salts[1])]
                else:
                    print(f"ERROR: Need at least 2 salts for {ion_type} comparison")
                    return None
            else:
                print(f"ERROR: Unknown ion type {ion_type} or insufficient salts for comparison")
                return None
        
        log_lines.append(f"Salt pairs to compare:")
        for pair in salt_pairs:
            salt1_label = self.salt_data[pair[0]]['label']
            salt2_label = self.salt_data[pair[1]]['label']
            log_lines.append(f"  {salt1_label} vs {salt2_label}")
        log_lines.append("")
        
        # Collect ion pairing by coordination data
        comparison_data = {}
        
        log_lines.append("ION PAIRING BY COORDINATION DATA")
        log_lines.append("-"*80)
        log_lines.append(f"{'Salt':<15} {'CN':<5} {'CIP (%)':<10} {'SIP (%)':<10} {'DSIP (%)':<10} {'FI (%)':<10} {'CN_Prob (%)':<12}")
        log_lines.append("-"*80)
        

        for salt_key in [salt for pair in salt_pairs for salt in pair]:
            if salt_key not in self.loaded_data:
                continue
            
            if concentration not in self.loaded_data[salt_key]:
                print(f"WARNING: {concentration} not found for {self.salt_data[salt_key]['label']}")
                continue
            
            conc_data = self.loaded_data[salt_key][concentration]
            comparison_data[salt_key] = {}
            
            # Method 1: Look for ion_pairing_probabilities_by_coordination
            if 'ion_pairing_probabilities_by_coordination' in conc_data:
                pairing_coord_data = conc_data['ion_pairing_probabilities_by_coordination']
                
                # Try to find ion-specific data
                for ion_variant in [ion_type, f'{ion_type}+', f'{ion_type}-', ion_type.upper()]:
                    if ion_variant in pairing_coord_data:
                        ion_data = pairing_coord_data[ion_variant]
                        
                        # Direct extraction from the actual data structure
                        if isinstance(ion_data, dict):
                            processed_data = {}
                            
                            for cn_key, cn_data in ion_data.items():
                                # cn_key is the coordination number (as integer)
                                if isinstance(cn_key, (int, float)):
                                    cn_number = int(cn_key)
                                    
                                    # Apply coordination thresholds
                                    if cn_number < min_coordination_threshold:
                                        continue
                                    
                                    if max_coordination_threshold is not None and cn_number > max_coordination_threshold:
                                        continue
                                    
                                    # Extract from the nested structure
                                    if isinstance(cn_data, dict) and 'probabilities' in cn_data:
                                        pairing_probs = cn_data['probabilities']
                                        total_obs = cn_data.get('total_observations', 1)
                                        
                                        # Extract pairing probabilities (convert to percentages)
                                        cip_prob = pairing_probs.get('CIP', 0) * 100
                                        sip_prob = pairing_probs.get('SIP', 0) * 100
                                        dsip_prob = pairing_probs.get('DSIP', 0) * 100
                                        fi_prob = pairing_probs.get('FI', 0) * 100
                                        
                                        cn_prob = 100.0
                                        
                                        # Apply probability threshold
                                        if cn_prob >= min_probability_threshold:
                                            processed_data[cn_number] = {
                                                'CIP': cip_prob,
                                                'SIP': sip_prob,
                                                'DSIP': dsip_prob,
                                                'FI': fi_prob,
                                                'CN_probability': cn_prob,
                                                'total_observations': total_obs
                                            }
                            
                            if processed_data:
                                salt_label = self.salt_data[salt_key]['label']
                                
                                for cn_number, data in processed_data.items():
                                    log_lines.append(f"{salt_label:<15} {cn_number:<5} {data['CIP']:<10.1f} {data['SIP']:<10.1f} {data['DSIP']:<10.1f} {data['FI']:<10.1f} {data['CN_probability']:<12.1f}")
                                
                                comparison_data[salt_key] = processed_data
                                break
                
                if salt_key in comparison_data and comparison_data[salt_key]:
                    continue
            
            # Fallback
            if salt_key not in comparison_data or not comparison_data[salt_key]:
                print(f"WARNING: No coordination-pairing data found for {self.salt_data[salt_key]['label']}")
                comparison_data[salt_key] = {}
        
        log_lines.append("")
        
        # Check if we have valid data to plot
        valid_salts = [salt_key for salt_key in comparison_data.keys() if comparison_data[salt_key]]
        
        if len(valid_salts) < 2:
            print("ERROR: Need at least 2 salts with valid coordination-pairing data")
            print(f"Available data: {[self.salt_data[k]['label'] for k in valid_salts]}")
            return comparison_data
        
        # ENHANCED Helper function to analyze bar heights and determine if broken axis is needed
        def analyze_bar_heights(bar_heights, debug=True):
            '''
            Analyze bar heights to determine if broken axis should be applied.
            ENHANCED: Now handles manual override parameters.
            '''
            # FIXED: Manual override takes precedence
            if manual_break_point is not None:
                if manual_top_start is not None:
                    break_point = manual_break_point
                    top_start = manual_top_start
                    
                    if debug:
                        print(f"DEBUG: Using MANUAL broken axis settings:")
                        print(f"  Manual Break point: {break_point:.1f}%")
                        print(f"  Manual Top start: {top_start:.1f}%")
                    
                    log_lines.append("BROKEN AXIS ANALYSIS - MANUAL OVERRIDE")
                    log_lines.append("-"*40)
                    log_lines.append(f"Manual break point: {break_point:.1f}%")
                    log_lines.append(f"Manual top start: {top_start:.1f}%")
                    log_lines.append("")
                    
                    return True, break_point, top_start
                else:
                    print(f"WARNING: manual_break_point provided ({manual_break_point}) but manual_top_start is None. Using automatic calculation for top_start.")
            
            if not enable_broken_axis or len(bar_heights) < 2:
                if debug:
                    print(f"DEBUG: Broken axis disabled or insufficient data points ({len(bar_heights)})")
                return False, None, None
            
            # Remove zeros and sort heights to find top values
            non_zero_heights = [h for h in bar_heights if h > 0.1]  # Ignore very small values
            
            if len(non_zero_heights) < 2:
                if debug:
                    print(f"DEBUG: Insufficient non-zero heights ({len(non_zero_heights)})")
                return False, None, None
            
            sorted_heights = sorted(non_zero_heights, reverse=True)
            
            if debug:
                print(f"DEBUG: Sorted heights (top 10): {sorted_heights[:10]}")
            
            # Find the largest gap between consecutive values
            gaps = []
            for i in range(len(sorted_heights) - 1):
                gap = sorted_heights[i] - sorted_heights[i + 1]
                gaps.append((gap, i, sorted_heights[i], sorted_heights[i + 1]))
            
            # Sort gaps by size (largest first)
            gaps.sort(reverse=True, key=lambda x: x[0])
            
            if debug:
                print(f"DEBUG: Top 5 gaps: {gaps[:5]}")
            
            if gaps:
                largest_gap, gap_index, higher_val, lower_val = gaps[0]
                
                # Check if this gap meets our criteria
                relative_gap = (largest_gap / higher_val) * 100 if higher_val > 0 else 0
                
                if debug:
                    print(f"DEBUG: Largest gap analysis:")
                    print(f"  Gap size: {largest_gap:.1f}%")
                    print(f"  Between: {higher_val:.1f}% and {lower_val:.1f}%")
                    print(f"  Relative to higher: {relative_gap:.1f}%")
                    print(f"  Absolute threshold: {broken_axis_min_gap}%")
                    print(f"  Relative threshold: {broken_axis_threshold}%")
                
                should_break = (relative_gap >= broken_axis_threshold and largest_gap >= broken_axis_min_gap)
                
                if should_break:
                    # FIXED: Use manual_break_point if provided
                    if manual_break_point is not None:
                        break_point = manual_break_point
                        # Calculate top_start relative to manual break point
                        if manual_top_start is not None:
                            top_start = manual_top_start
                        else:
                            # Auto-calculate top_start based on manual break point
                            top_start = max(break_point + 5, higher_val * 0.90)
                    else:
                        # Automatic calculation
                        break_point = lower_val * 1.05  # 5% above the lower value
                        top_start = higher_val * 0.90   # Start at 90% of highest value
                    
                    if debug:
                        print(f"DEBUG: APPLYING broken axis:")
                        print(f"  Break point: {break_point:.1f}% ({'manual' if manual_break_point else 'auto'})")
                        print(f"  Top start: {top_start:.1f}% ({'manual' if manual_top_start else 'auto'})")
                    
                    log_lines.append("BROKEN AXIS ANALYSIS")
                    log_lines.append("-"*40)
                    log_lines.append(f"Largest gap: {largest_gap:.1f}% between {higher_val:.1f}% and {lower_val:.1f}%")
                    log_lines.append(f"Relative gap: {relative_gap:.1f}%")
                    log_lines.append(f"Absolute gap: {largest_gap:.1f}%")
                    log_lines.append(f"Break point: {break_point:.1f}% ({'manual' if manual_break_point else 'auto'})")
                    log_lines.append(f"Top start: {top_start:.1f}% ({'manual' if manual_top_start else 'auto'})")
                    log_lines.append("")
                    
                    return True, break_point, top_start
                else:
                    if debug:
                        print(f"DEBUG: NOT applying broken axis (gap thresholds not met)")
                    
                    log_lines.append(f"BROKEN AXIS ANALYSIS: Not applied")
                    log_lines.append(f"  Largest gap: {relative_gap:.1f}% (threshold: {broken_axis_threshold}%)")
                    log_lines.append(f"  Absolute gap: {largest_gap:.1f}% (threshold: {broken_axis_min_gap}%)")
                    log_lines.append("")
                    return False, None, None
            else:
                if debug:
                    print(f"DEBUG: No gaps found")
                return False, None, None
        
        # Helper function to create broken axis subplots with automatic height ratios
        def create_broken_axis_subplots(fig, break_point, top_start):
            '''Create broken axis subplot configuration with automatic or manual height ratios'''
            print(f"Creating broken axis subplots: break_point={break_point:.1f}, top_start={top_start:.1f}")
            
            # ENHANCED: Calculate height ratios based on actual data ranges or use manual override
            if manual_break_ratio is not None:
                # Manual override
                bottom_ratio = manual_break_ratio
                top_ratio = 1 - manual_break_ratio
                
                print(f"Using MANUAL break ratio: bottom={bottom_ratio:.2f} ({bottom_ratio*100:.1f}%), top={top_ratio:.2f} ({top_ratio*100:.1f}%)")
                
                log_lines.append(f"MANUAL HEIGHT RATIOS:")
                log_lines.append(f"  Manual break ratio: {manual_break_ratio:.2f}")
                log_lines.append(f"  Bottom ratio: {bottom_ratio:.2f} ({bottom_ratio*100:.1f}% of figure height)")
                log_lines.append(f"  Top ratio: {top_ratio:.2f} ({top_ratio*100:.1f}% of figure height)")
                log_lines.append("")
                
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
                
                print(f"Using AUTOMATIC break ratio based on data ranges:")
                print(f"  Bottom range: 0 to {break_point:.1f}% = {bottom_range:.1f} percentage points")
                print(f"  Top range: {top_start:.1f}% to {max_y_axis if max_y_axis else 105}% = {top_range:.1f} percentage points")
                print(f"  Bottom ratio: {bottom_ratio:.2f} ({bottom_ratio*100:.1f}% of figure height)")
                print(f"  Top ratio: {top_ratio:.2f} ({top_ratio*100:.1f}% of figure height)")
                
                log_lines.append(f"AUTOMATIC HEIGHT RATIOS:")
                log_lines.append(f"  Bottom range: 0 to {break_point:.1f}% = {bottom_range:.1f} percentage points")
                log_lines.append(f"  Top range: {top_start:.1f}% to {max_y_axis if max_y_axis else 105}% = {top_range:.1f} percentage points")
                log_lines.append(f"  Bottom ratio: {bottom_ratio:.2f} ({bottom_ratio*100:.1f}% of figure height)")
                log_lines.append(f"  Top ratio: {top_ratio:.2f} ({top_ratio*100:.1f}% of figure height)")
                log_lines.append("")
            
            # Clear the figure and create subplots with calculated height ratios  
            fig.clear()
            gs = fig.add_gridspec(2, 1, height_ratios=[top_ratio, bottom_ratio], hspace=0.05)
            ax_top = fig.add_subplot(gs[0])
            ax_bottom = fig.add_subplot(gs[1])
            
            # Set y-limits
            if max_y_axis is not None:
                ax_top.set_ylim(top_start, max_y_axis)
                ax_bottom.set_ylim(0, break_point)
            else:
                ax_bottom.set_ylim(0, break_point)
                ax_top.set_ylim(top_start, 105)  # 105 to give some headroom
            
            # Hide spines between subplots
            ax_top.spines['bottom'].set_visible(False)
            ax_bottom.spines['top'].set_visible(False)
            ax_top.xaxis.tick_top()
            ax_top.tick_params(labeltop=False, top=False)  # Don't show x-axis labels on top
            
            # Add break lines
            d = 0.015  # Size of break lines
            kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=2)
            ax_top.plot((-d, +d), (-d, +d), **kwargs)        # Top-left diagonal
            ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal
            
            kwargs.update(transform=ax_bottom.transAxes)  # Switch to bottom axes
            ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
            ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal
            
            return ax_bottom, ax_top
        
        # Create comparison plots for each salt pair
        for i, (salt1_key, salt2_key) in enumerate(salt_pairs):
            
            if salt1_key not in valid_salts or salt2_key not in valid_salts:
                log_lines.append(f"WARNING: Skipping {salt1_key} vs {salt2_key} - missing data")
                continue
            
            salt1_label = self.salt_data[salt1_key]['label']
            salt2_label = self.salt_data[salt2_key]['label']
            salt1_color = self.salt_data[salt1_key].get('color', 'blue')
            salt2_color = self.salt_data[salt2_key].get('color', 'red')
            
            # Get all coordination numbers from both salts
            all_cn = set()
            all_cn.update(comparison_data[salt1_key].keys())
            all_cn.update(comparison_data[salt2_key].keys())
            sorted_cn = sorted(all_cn)
            
            if not sorted_cn:
                print(f"No coordination numbers found for {salt1_label} vs {salt2_label}")
                continue
            
            # Use the simple pairing colors
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
            pair_types = ['CIP', 'SIP', 'DSIP', 'FI']
            pairing_colors = dict(zip(pair_types, colors))
            
            if plot_type == 'stacked_bar':
                print(f"\n=== PROCESSING STACKED BAR PLOT ===")
                
                # Get data for both salts
                salt1_data = comparison_data[salt1_key]
                salt2_data = comparison_data[salt2_key]
                
                # Prepare data for side-by-side stacked bars
                n_cn = len(sorted_cn)
                x_positions = np.arange(n_cn)
                bar_width = 0.35
                
                # Collect individual pairing state heights for analysis
                all_pairing_heights = []
                
                # Prepare data arrays for both salts
                salt1_cip = []
                salt1_sip = []
                salt1_dsip = []
                salt1_fi = []
                salt2_cip = []
                salt2_sip = []
                salt2_dsip = []
                salt2_fi = []
                
                for cn in sorted_cn:
                    # Salt 1 data
                    if cn in salt1_data:
                        cip = salt1_data[cn]['CIP']
                        sip = salt1_data[cn]['SIP'] 
                        dsip = salt1_data[cn]['DSIP']
                        fi = salt1_data[cn]['FI']
                        
                        salt1_cip.append(cip)
                        salt1_sip.append(sip)
                        salt1_dsip.append(dsip)
                        salt1_fi.append(fi)
                        
                        # Add individual heights to analysis
                        all_pairing_heights.extend([cip, sip, dsip, fi])
                        # Also add total stacked height
                        all_pairing_heights.append(cip + sip + dsip + fi)
                    else:
                        salt1_cip.append(0)
                        salt1_sip.append(0)
                        salt1_dsip.append(0)
                        salt1_fi.append(0)
                    
                    # Salt 2 data
                    if cn in salt2_data:
                        cip = salt2_data[cn]['CIP']
                        sip = salt2_data[cn]['SIP']
                        dsip = salt2_data[cn]['DSIP']
                        fi = salt2_data[cn]['FI']
                        
                        salt2_cip.append(cip)
                        salt2_sip.append(sip)
                        salt2_dsip.append(dsip)
                        salt2_fi.append(fi)
                        
                        # Add individual heights to analysis
                        all_pairing_heights.extend([cip, sip, dsip, fi])
                        # Also add total stacked height
                        all_pairing_heights.append(cip + sip + dsip + fi)
                    else:
                        salt2_cip.append(0)
                        salt2_sip.append(0)
                        salt2_dsip.append(0)
                        salt2_fi.append(0)
                
                # Convert to numpy arrays
                salt1_cip = np.array(salt1_cip)
                salt1_sip = np.array(salt1_sip)
                salt1_dsip = np.array(salt1_dsip)
                salt1_fi = np.array(salt1_fi)
                salt2_cip = np.array(salt2_cip)
                salt2_sip = np.array(salt2_sip)
                salt2_dsip = np.array(salt2_dsip)
                salt2_fi = np.array(salt2_fi)
                
                # Analyze if broken axis is needed
                print(f"Analyzing {len(all_pairing_heights)} pairing heights for broken axis...")
                should_break, break_point, top_start = analyze_bar_heights(all_pairing_heights, debug=True)
                
                # Create figure
                fig = plt.figure(figsize=figsize, facecolor='white')
                
                if should_break:
                    print(f"APPLYING BROKEN AXIS to stacked bar plot!")
                    
                    # Create broken axis
                    ax_bottom, ax_top = create_broken_axis_subplots(fig, break_point, top_start)
                    axes = [ax_bottom, ax_top]
                    
                    # Plot on both axes
                    for ax in axes:
                        # Calculate x positions for side-by-side bars
                        x1 = x_positions - bar_width/2  # Salt 1 positions (left)
                        x2 = x_positions + bar_width/2  # Salt 2 positions (right)
                        
                        # Only show labels on bottom axis
                        show_labels = (ax == ax_bottom)
                        
                        # Plot stacked bars for Salt 1 (no hatching)
                        ax.bar(x1, salt1_cip, bar_width, 
                            label=f'{salt1_label} CIP' if show_labels else "", 
                            color=pairing_colors['CIP'], alpha=alpha, edgecolor='black', linewidth=0.5)
                        ax.bar(x1, salt1_sip, bar_width, bottom=salt1_cip, 
                            label=f'{salt1_label} SIP' if show_labels else "",
                            color=pairing_colors['SIP'], alpha=alpha, edgecolor='black', linewidth=0.5)
                        ax.bar(x1, salt1_dsip, bar_width, bottom=salt1_cip + salt1_sip, 
                            label=f'{salt1_label} DSIP' if show_labels else "",
                            color=pairing_colors['DSIP'], alpha=alpha, edgecolor='black', linewidth=0.5)
                        ax.bar(x1, salt1_fi, bar_width, bottom=salt1_cip + salt1_sip + salt1_dsip, 
                            label=f'{salt1_label} FI' if show_labels else "",
                            color=pairing_colors['FI'], alpha=alpha, edgecolor='black', linewidth=0.5)
                        
                        # Plot stacked bars for Salt 2 (with hatching)
                        ax.bar(x2, salt2_cip, bar_width, 
                            label=f'{salt2_label} CIP' if show_labels else "",
                            color=pairing_colors['CIP'], alpha=alpha, hatch='///', edgecolor='black', linewidth=0.5)
                        ax.bar(x2, salt2_sip, bar_width, bottom=salt2_cip, 
                            label=f'{salt2_label} SIP' if show_labels else "",
                            color=pairing_colors['SIP'], alpha=alpha, hatch='///', edgecolor='black', linewidth=0.5)
                        ax.bar(x2, salt2_dsip, bar_width, bottom=salt2_cip + salt2_sip, 
                            label=f'{salt2_label} DSIP' if show_labels else "",
                            color=pairing_colors['DSIP'], alpha=alpha, hatch='///', edgecolor='black', linewidth=0.5)
                        ax.bar(x2, salt2_fi, bar_width, bottom=salt2_cip + salt2_sip + salt2_dsip, 
                            label=f'{salt2_label} FI' if show_labels else "",
                            color=pairing_colors['FI'], alpha=alpha, hatch='///', edgecolor='black', linewidth=0.5)
                        
                        # Set x-axis ticks and labels (only on bottom)
                        ax.set_xticks(x_positions)
                        if ax == ax_bottom:
                            ax.set_xticklabels([f'CN {cn}' for cn in sorted_cn], fontsize=tick_fontsize)
                        else:
                            ax.set_xticklabels([])
                        
                        ax.tick_params(axis='both', labelsize=tick_fontsize)
                        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
                    
                    # Add labels and title
                    ax_bottom.set_xlabel('Coordination Number', fontsize=label_fontsize)
                    fig.text(0.04, 0.5, 'Probability (%)', va='center', rotation='vertical', fontsize=label_fontsize)
                    
                    # Enhanced title with manual settings info
                    title_text = f'Ion Pairing by Coordination Number: {ion_type} at {concentration}\n'
                    title_text += f'{salt1_label} vs {salt2_label} (Broken Axis'
                    if manual_break_point is not None or manual_top_start is not None or manual_break_ratio is not None:
                        title_text += ' - Manual'
                    title_text += ')'
                    fig.suptitle(title_text, fontsize=title_fontsize, fontweight='bold')
                    
                    # Create custom legend
                    from matplotlib.patches import Patch
                    from matplotlib.lines import Line2D
                    
                    legend_elements = []
                    # Add pairing state colors
                    for state in ['CIP', 'SIP', 'DSIP', 'FI']:
                        legend_elements.append(Patch(facecolor=pairing_colors[state], alpha=alpha, 
                                                    edgecolor='black', linewidth=0.5, label=state))
                    
                    # Add separator
                    legend_elements.append(Line2D([0], [0], color='white', linewidth=0, label=''))
                    
                    # Add salt distinction
                    legend_elements.append(Patch(facecolor='gray', alpha=alpha, 
                                                edgecolor='black', linewidth=0.5, label=f'{salt1_label} (solid)'))
                    legend_elements.append(Patch(facecolor='gray', alpha=alpha, hatch='///',
                                                edgecolor='black', linewidth=0.5, label=f'{salt2_label} (hatched)'))
                    
                    ax_top.legend(handles=legend_elements, fontsize=legend_fontsize, frameon=False, loc='best', ncol=2)
                
                else:
                    print(f"NOT applying broken axis to stacked bar plot (thresholds not met)")
                    
                    # Standard single axis
                    ax = fig.add_subplot(111)
                    
                    # Calculate x positions for side-by-side bars
                    x1 = x_positions - bar_width/2  # Salt 1 positions (left)
                    x2 = x_positions + bar_width/2  # Salt 2 positions (right)
                    
                    # Plot stacked bars (same as original)
                    ax.bar(x1, salt1_cip, bar_width, label=f'{salt1_label} CIP', 
                        color=pairing_colors['CIP'], alpha=alpha, edgecolor='black', linewidth=0.5)
                    ax.bar(x1, salt1_sip, bar_width, bottom=salt1_cip, label=f'{salt1_label} SIP',
                        color=pairing_colors['SIP'], alpha=alpha, edgecolor='black', linewidth=0.5)
                    ax.bar(x1, salt1_dsip, bar_width, bottom=salt1_cip + salt1_sip, label=f'{salt1_label} DSIP',
                        color=pairing_colors['DSIP'], alpha=alpha, edgecolor='black', linewidth=0.5)
                    ax.bar(x1, salt1_fi, bar_width, bottom=salt1_cip + salt1_sip + salt1_dsip, label=f'{salt1_label} FI',
                        color=pairing_colors['FI'], alpha=alpha, edgecolor='black', linewidth=0.5)
                    
                    ax.bar(x2, salt2_cip, bar_width, label=f'{salt2_label} CIP',
                        color=pairing_colors['CIP'], alpha=alpha, hatch='///', edgecolor='black', linewidth=0.5)
                    ax.bar(x2, salt2_sip, bar_width, bottom=salt2_cip, label=f'{salt2_label} SIP',
                        color=pairing_colors['SIP'], alpha=alpha, hatch='///', edgecolor='black', linewidth=0.5)
                    ax.bar(x2, salt2_dsip, bar_width, bottom=salt2_cip + salt2_sip, label=f'{salt2_label} DSIP',
                        color=pairing_colors['DSIP'], alpha=alpha, hatch='///', edgecolor='black', linewidth=0.5)
                    ax.bar(x2, salt2_fi, bar_width, bottom=salt2_cip + salt2_sip + salt2_dsip, label=f'{salt2_label} FI',
                        color=pairing_colors['FI'], alpha=alpha, hatch='///', edgecolor='black', linewidth=0.5)
                    
                    # Customize plot
                    ax.set_xlabel('Coordination Number', fontsize=label_fontsize)
                    ax.set_ylabel('Probability (%)', fontsize=label_fontsize)
                    
                    # Enhanced title with max_y_axis info
                    title_text = f'Ion Pairing by Coordination Number: {ion_type} at {concentration}\n'
                    title_text += f'{salt1_label} vs {salt2_label}'
                    if max_y_axis is not None:
                        title_text += f' (Max Y: {max_y_axis}%)'
                    ax.set_title(title_text, fontsize=title_fontsize, fontweight='bold')
                    
                    # Set x-axis ticks and labels
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels([f'CN {cn}' for cn in sorted_cn], fontsize=tick_fontsize)
                    ax.tick_params(axis='both', labelsize=tick_fontsize)
                    
                    # Apply max_y_axis
                    if max_y_axis is not None:
                        ax.set_ylim(0, max_y_axis)
                    else:
                        ax.set_ylim(0, 100)
                    
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
                    
                    # Create custom legend (same as original)
                    from matplotlib.patches import Patch
                    from matplotlib.lines import Line2D
                    
                    legend_elements = []
                    for state in ['CIP', 'SIP', 'DSIP', 'FI']:
                        legend_elements.append(Patch(facecolor=pairing_colors[state], alpha=alpha, 
                                                    edgecolor='black', linewidth=0.5, label=state))
                    
                    legend_elements.append(Line2D([0], [0], color='white', linewidth=0, label=''))
                    
                    legend_elements.append(Patch(facecolor='gray', alpha=alpha, 
                                                edgecolor='black', linewidth=0.5, label=f'{salt1_label} (solid)'))
                    legend_elements.append(Patch(facecolor='gray', alpha=alpha, hatch='///',
                                                edgecolor='black', linewidth=0.5, label=f'{salt2_label} (hatched)'))
                    
                    ax.legend(handles=legend_elements, fontsize=legend_fontsize, frameon=False, loc='best', ncol=2)
                
                plt.tight_layout()
            
            elif plot_type == 'grouped_bar':
                print(f"\n=== PROCESSING GROUPED BAR PLOT ===")
                
                # Grouped bar chart with broken axis support
                pairing_states = ['CIP', 'SIP', 'DSIP', 'FI']
                x = np.arange(len(sorted_cn))
                
                # Calculate proper bar width and spacing
                n_salts = 2
                n_pairing_states = len(pairing_states)
                total_bars_per_cn = n_salts * n_pairing_states
                
                individual_bar_width = 0.8 / total_bars_per_cn
                spacing = individual_bar_width * 0.1
                
                # Collect all bar heights for analysis
                all_bar_heights = []
                for ps_idx, pairing_state in enumerate(pairing_states):
                    for cn in sorted_cn:
                        salt1_val = comparison_data[salt1_key].get(cn, {}).get(pairing_state, 0)
                        salt2_val = comparison_data[salt2_key].get(cn, {}).get(pairing_state, 0)
                        all_bar_heights.extend([salt1_val, salt2_val])
                
                # Analyze if broken axis is needed
                print(f"Analyzing {len(all_bar_heights)} individual bar heights for broken axis...")
                should_break, break_point, top_start = analyze_bar_heights(all_bar_heights, debug=True)
                
                # Create figure
                fig = plt.figure(figsize=figsize, facecolor='white')
                
                if should_break:
                    print(f"APPLYING BROKEN AXIS to grouped bar plot!")
                    
                    # Create broken axis
                    ax_bottom, ax_top = create_broken_axis_subplots(fig, break_point, top_start)
                    axes = [ax_bottom, ax_top]
                else:
                    print(f"NOT applying broken axis to grouped bar plot (thresholds not met)")
                    
                    # Standard single axis
                    ax = fig.add_subplot(111)
                    axes = [ax]
                
                # Plot bars for each pairing state
                for ps_idx, pairing_state in enumerate(pairing_states):
                    salt1_values = []
                    salt2_values = []
                    
                    for cn in sorted_cn:
                        salt1_values.append(comparison_data[salt1_key].get(cn, {}).get(pairing_state, 0))
                        salt2_values.append(comparison_data[salt2_key].get(cn, {}).get(pairing_state, 0))
                    
                    # Calculate x positions to avoid overlap
                    bar_group_start = -0.4 + ps_idx * (2 * individual_bar_width + spacing * 2)
                    
                    x1 = x + bar_group_start
                    x2 = x + bar_group_start + individual_bar_width + spacing
                    
                    for ax_idx, ax in enumerate(axes):
                        # Only show labels on first axis (or bottom axis if broken)
                        show_labels = (ax_idx == 0) if should_break else True
                        
                        ax.bar(x1, salt1_values, individual_bar_width, alpha=alpha, 
                            color=pairing_colors[pairing_state], 
                            label=f'{salt1_label} {pairing_state}' if show_labels else "",
                            edgecolor='black', linewidth=0.5)
                        ax.bar(x2, salt2_values, individual_bar_width, alpha=alpha,
                            color=pairing_colors[pairing_state], 
                            label=f'{salt2_label} {pairing_state}' if show_labels else "",
                            hatch='///', edgecolor='black', linewidth=0.5)
                        
                        # Set x-axis ticks and labels
                        ax.set_xticks(x)
                        if should_break:
                            if ax == ax_bottom:
                                ax.set_xticklabels([f'CN {cn}' for cn in sorted_cn], fontsize=tick_fontsize)
                            else:
                                ax.set_xticklabels([])
                        else:
                            ax.set_xticklabels([f'CN {cn}' for cn in sorted_cn], fontsize=tick_fontsize)
                        
                        ax.tick_params(axis='both', labelsize=tick_fontsize)
                        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
                
                # Add labels and title
                if should_break:
                    ax_bottom.set_xlabel('Coordination Number', fontsize=label_fontsize)
                    fig.text(0.04, 0.5, 'Probability (%)', va='center', rotation='vertical', fontsize=label_fontsize)
                    
                    # Enhanced title with manual settings info
                    title_text = f'Ion Pairing by Coordination Number: {ion_type}\n'
                    title_text += f'{salt1_label} vs {salt2_label} at {concentration} (Broken Axis'
                    if manual_break_point is not None or manual_top_start is not None or manual_break_ratio is not None:
                        title_text += ' - Manual'
                    title_text += ')'
                    fig.suptitle(title_text, fontsize=title_fontsize, fontweight='bold')
                    
                    ax_top.legend(fontsize=legend_fontsize-2, frameon=False, loc='best', ncol=2)
                else:
                    ax.set_xlabel('Coordination Number', fontsize=label_fontsize)
                    ax.set_ylabel('Probability (%)', fontsize=label_fontsize)
                    
                    # Enhanced title with max_y_axis info
                    title_text = f'Ion Pairing by Coordination Number: {ion_type}\n'
                    title_text += f'{salt1_label} vs {salt2_label} at {concentration}'
                    if max_y_axis is not None:
                        title_text += f' (Max Y: {max_y_axis}%)'
                    ax.set_title(title_text, fontsize=title_fontsize, fontweight='bold')
                    
                    # Apply max_y_axis for single axis
                    if max_y_axis is not None:
                        ax.set_ylim(0, max_y_axis)
                    
                    ax.legend(fontsize=legend_fontsize-2, frameon=False, loc='best', ncol=2)
                
                plt.tight_layout()
            
            elif plot_type == 'heatmap':
                # Heatmap with proper colorbar positioning
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] + 2, figsize[1]), 
                                            facecolor='white')
                
                pairing_states = ['CIP', 'SIP', 'DSIP', 'FI']
                
                for ax_idx, (salt_key, salt_label, ax) in enumerate([(salt1_key, salt1_label, ax1), 
                                                                (salt2_key, salt2_label, ax2)]):
                    salt_data = comparison_data[salt_key]
                    
                    # Prepare heatmap data matrix
                    heatmap_data = np.zeros((len(sorted_cn), len(pairing_states)))
                    
                    for cn_idx, cn in enumerate(sorted_cn):
                        if cn in salt_data:
                            for ps_idx, pairing_state in enumerate(pairing_states):
                                heatmap_data[cn_idx, ps_idx] = salt_data[cn][pairing_state]
                    
                    # Apply max_y_axis as vmax for heatmaps
                    vmax_value = max_y_axis if max_y_axis is not None else 100
                    
                    # Create heatmap with 'Blues' colormap
                    im = ax.imshow(heatmap_data, cmap='Blues', aspect='auto', 
                                vmin=0, vmax=vmax_value, interpolation='nearest')
                    
                    # Add text annotations
                    for cn_idx in range(len(sorted_cn)):
                        for ps_idx in range(len(pairing_states)):
                            value = heatmap_data[cn_idx, ps_idx]
                            text = ax.text(ps_idx, cn_idx, f'{value:.1f}%',
                                        ha="center", va="center", 
                                        color="white" if value > vmax_value/2 else "black",
                                        fontsize=tick_fontsize-2, fontweight='bold')
                    
                    # Customize heatmap
                    ax.set_xticks(np.arange(len(pairing_states)))
                    ax.set_xticklabels(pairing_states, fontsize=tick_fontsize)
                    ax.set_yticks(np.arange(len(sorted_cn)))
                    ax.set_yticklabels([f'CN {cn}' for cn in sorted_cn], fontsize=tick_fontsize)
                    ax.set_xlabel('Pairing State', fontsize=label_fontsize)
                    ax.set_ylabel('Coordination Number', fontsize=label_fontsize)
                    ax.set_title(f'{salt_label}', fontsize=title_fontsize, fontweight='bold')
                
                # Add colorbar with proper positioning to avoid overlap
                # Create space for colorbar by adjusting subplot layout
                plt.subplots_adjust(right=0.85)
                
                # Add colorbar to the right of both subplots
                cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label('Probability (%)', fontsize=label_fontsize)
                cbar.ax.tick_params(labelsize=tick_fontsize)
                
                # Enhanced title with max_y_axis info
                title_text = f'Ion Pairing by Coordination Heatmap: {ion_type} at {concentration}'
                if max_y_axis is not None:
                    title_text += f' (Scale: 0-{max_y_axis}%)'
                plt.suptitle(title_text, fontsize=title_fontsize + 2, fontweight='bold')
            
            else:
                raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'stacked_bar', 'grouped_bar', or 'heatmap'")
            
            # Save plot
            if save_plots:
                threshold_suffix = f'_cn{min_coordination_threshold}'
                if max_coordination_threshold is not None:
                    threshold_suffix += f'to{max_coordination_threshold}'
                threshold_suffix += f'_prob{min_probability_threshold}p' if min_probability_threshold > 0 else ''
                
                # Enhanced filename with new parameters
                broken_suffix = ''
                if plot_type in ['stacked_bar', 'grouped_bar'] and enable_broken_axis and 'should_break' in locals() and should_break:
                    broken_suffix = '_broken'
                    if manual_break_point is not None or manual_break_ratio is not None:
                        broken_suffix += '_manual'
                
                max_y_suffix = f'_maxY{max_y_axis}' if max_y_axis is not None else ''
                
                filename = f'ion_pairing_coordination_comparison_{ion_type}_{salt1_key}_vs_{salt2_key}_{concentration.replace(".", "p")}_{plot_type}{threshold_suffix}{broken_suffix}{max_y_suffix}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                log_lines.append(f"PLOT SAVED: {filename}")
                print(f"Ion pairing by coordination comparison plot saved: {filename}")
            
            plt.show()
            
            # Print statistical summary
            print(f"\nCoordination-Pairing Analysis for {ion_type} at {concentration}:")
            for salt_key, salt_label in [(salt1_key, salt1_label), (salt2_key, salt2_label)]:
                if salt_key in comparison_data and comparison_data[salt_key]:
                    print(f"\n  {salt_label}:")
                    salt_data = comparison_data[salt_key]
                    
                    # Calculate average pairing probabilities weighted by coordination probability
                    weighted_cip = 0
                    weighted_sip = 0
                    weighted_dsip = 0
                    weighted_fi = 0
                    total_weight = 0
                    
                    for cn, data in salt_data.items():
                        weight = data['CN_probability'] / 100
                        weighted_cip += data['CIP'] * weight
                        weighted_sip += data['SIP'] * weight
                        weighted_dsip += data['DSIP'] * weight
                        weighted_fi += data['FI'] * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        weighted_cip /= total_weight
                        weighted_sip /= total_weight
                        weighted_dsip /= total_weight
                        weighted_fi /= total_weight
                        
                        print(f"    Weighted Average Pairing Probabilities:")
                        print(f"      CIP: {weighted_cip:.1f}%")
                        print(f"      SIP: {weighted_sip:.1f}%") 
                        print(f"      DSIP: {weighted_dsip:.1f}%")
                        print(f"      FI: {weighted_fi:.1f}%")
                    
                    # Show coordination number range
                    cn_range = f"CN {min(salt_data.keys())}-{max(salt_data.keys())}"
                    print(f"    Coordination Range: {cn_range}")
                    print(f"    Number of CN States: {len(salt_data)}")
        
        # Save log
        if save_log:
            threshold_suffix = f'_cn{min_coordination_threshold}'
            if max_coordination_threshold is not None:
                threshold_suffix += f'to{max_coordination_threshold}'
            threshold_suffix += f'_prob{min_probability_threshold}p' if min_probability_threshold > 0 else ''
            
            # Enhanced log filename
            max_y_suffix = f'_maxY{max_y_axis}' if max_y_axis is not None else ''
            manual_suffix = '_manual' if (manual_break_point is not None or manual_top_start is not None or manual_break_ratio is not None) else ''
            
            log_filename = f'ion_pairing_coordination_comparison_{ion_type}_{concentration.replace(".", "p")}_{plot_type}{threshold_suffix}{max_y_suffix}{manual_suffix}.log'
            with open(log_filename, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Log saved: {log_filename}")
        
        return comparison_data

    # ─────────────────────────────────────────────────────────────────────────
    # Z-DIRECTION DENSITY PROFILE COMPARISON
    # ─────────────────────────────────────────────────────────────────────────

    def compare_cation_density_profiles(
        self,
        mode='by_concentration',
        salt_key=None,
        concentration=None,
        ion_types=None,
        salts=None,
        concentrations=None,
        cache_filename='z_analysis_cache_100bins_centered.pkl',
        # Clay boundary visual params
        show_clay_fill=True,
        clay_fill_color='yellow',
        clay_fill_alpha=0.15,
        show_boundary_lines=True,
        boundary_linewidth=1.0,
        boundary_alpha=0.7,
        si_color='orange',
        mgo_color='green',
        clay_avg_color='gray',
        si_linestyle='--',
        mgo_linestyle=':',
        clay_avg_linestyle='-',
        # Density curve params
        linewidth=2,
        linewidth_range=None,
        line_alpha=0.85,
        line_alpha_range=None,
        fill_curves=False,
        fill_alpha=0.15,
        # Symmetrization
        symmetrize=True,
        symmetrize_method='average',
        symmetry_center=0.0,
        # Figure / axes params
        figsize=(8, 5),
        dpi=300,
        save_plot=True,
        label_fontsize=14,
        tick_fontsize=12,
        legend_fontsize=11,
        show_legend=True,
        legend_location='best',
        legend_ncol=1,
        show_grid=False,
        grid_alpha=0.3,
        show_zero_line=True,
        zero_line_color='black',
        zero_line_style=':',
        zero_line_alpha=0.5,
        xlabel_text='z (Å)',
        ylabel_text='Ion Density (ions/Å³)',
        title_text=None,
        show_title=False,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontweight='normal',
        tick_fontweight='normal',
        legend_fontweight='normal',
        legend_framealpha=0.85,
        legend_bbox=None,
        xlim=None,
        ylim=None,
        conc_colors=None,
        ion_type_linestyles=None,
        ion_type_colors=None,
        ion_type_labels=None,
        clay_boundaries=None,
        show_clay_boundaries_in_legend=False,
        legend_split_by_ion_type=False,
        ion_legend_bbox=None,
        ion_legend_location='upper left',
        print_summary=True,
        save_cache=True,
        force_rerun=False,
        concentration_map=None,
        plot_individual=False,
        individual_legend_location='best',
        show_half='both',
        x_origin='center',
        # Water density overlay
        show_water_density=False,
        show_water_line=True,       # False = fill only, hide right y-axis
        water_density_color='blue',
        water_linewidth=2,
        water_density_alpha=0.7,
        fill_water_curve=True,
        water_fill_alpha=0.15,
        # Ion-specific fill
        fill_ion_types=None,
        ion_fill_alpha=0.12,
        water_density_ylabel='Water Density (molecules/Å³)',
        # Ion z-order control
        ion_type_zorder=None,
    ):
        """
        Compare z-direction cation density profiles across concentrations or salts.

        Loads z-analysis cache files (z_analysis_cache_*.pkl) directly from the
        folder paths defined in salt_data and plots overlaid density curves with
        clay boundary shading.

        Parameters
        ----------
        mode : str
            'by_concentration'  — one salt, all concentrations overlaid.
                                  Requires salt_key.
            'by_salt'           — one concentration, all salts overlaid.
                                  Requires concentration.
        salt_key : str
            Salt system key (e.g. 'MgCl2').  Required for mode='by_concentration'.
        concentration : str
            Concentration key (e.g. '0.11M').  Required for mode='by_salt'.
        ion_types : list of str, optional
            Ion types to plot (e.g. ['MG', 'NA']).  If None, all ion types
            present in the first loaded cache are plotted.
        salts : list of str, optional
            Subset of salt keys to include (mode='by_salt' only).
            If None, all salts that have the requested concentration are used.
        concentrations : list of str, optional
            Subset of concentration keys to include (mode='by_concentration' only).
            If None, all concentrations for the chosen salt are used.
        cache_filename : str
            Pickle filename to load from each folder.
            Default: 'z_analysis_cache_100bins_centered.pkl'.
        show_clay_fill : bool
            Fill clay regions with color.
        clay_fill_color : str
            Fill colour for clay region.
        clay_fill_alpha : float
            Alpha for clay fill.
        show_boundary_lines : bool
            Draw vertical lines at Si / Mgo / clay-average boundaries.
        boundary_linewidth : float
            Width of boundary lines.
        boundary_alpha : float
            Alpha of boundary lines.
        si_color, mgo_color, clay_avg_color : str
            Colours for boundary lines.
        si_linestyle, mgo_linestyle, clay_avg_linestyle : str
            Line styles for boundary lines.
        linewidth : float
            Width of density profile curves.
        line_alpha : float
            Alpha of density curves.
        fill_curves : bool
            Fill area under each density curve.
        fill_alpha : float
            Alpha for fill under curves.
        symmetrize : bool
            Symmetrize density profiles and boundary positions about symmetry_center.
        symmetrize_method : str
            'average' | 'mirror_positive' | 'mirror_negative'.
        symmetry_center : float
            Z-coordinate of symmetry centre (default 0.0).
        figsize : tuple
            Figure size in inches.
        dpi : int
            Resolution for saved figure.
        save_plot : bool
            Save the figure to a file.
        label_fontsize, tick_fontsize, legend_fontsize : int
            Font sizes.
        title_fontweight : str
            Font weight for title ('normal', 'bold', 'heavy', 'light', etc.).
        label_fontweight : str
            Font weight for axis labels ('normal', 'bold', etc.).
        tick_fontweight : str
            Font weight for tick labels ('normal', 'bold', etc.).
        legend_fontweight : str
            Font weight for legend text ('normal', 'bold', etc.).
        legend_framealpha : float
            Alpha (transparency) for legend background frame (0.0-1.0).
        legend_bbox : tuple of (x, y), optional
            If provided, positions legend at exact (x, y) axes coordinates
            using bbox_to_anchor. If None, uses legend_location string.
            Example: (0.02, 0.98) for top-left, (0.98, 0.98) for top-right.
        show_legend : bool
        legend_location : str
        legend_ncol : int
        show_grid : bool
        grid_alpha : float
        show_zero_line : bool
            Draw a dotted vertical line at z=0.
        xlabel_text, ylabel_text : str
        title_text : str, optional
            Override the auto-generated title.
        show_title : bool
        title_fontsize : int
        xlim, ylim : tuple, optional
            Axis limits.
        conc_colors : list, optional
            Explicit colours per concentration (mode='by_concentration').
            If None, a sequential colour gradient derived from the salt colour
            is generated automatically.
        ion_type_linestyles : dict, optional
            Mapping of ion_type → linestyle.  E.g. {'MG': '-', 'NA': '--'}.
            If None, cycles through ['-', '--', '-.', ':'].
        print_summary : bool
            Print a summary of loaded systems and boundary positions.

        Returns
        -------
        dict
            Nested dict of the raw arrays that were plotted:
            {'label': {'z': ..., 'ion_type': array, ...}, ...}
        """
        from scipy.interpolate import interp1d as _interp1d

        # ── helper: symmetrize a 1-D density profile ────────────────────────
        def _sym(z, data, method):
            pos_mask = z >= symmetry_center
            neg_mask = z < symmetry_center
            if method == 'average':
                pos_z = z[pos_mask];  pos_d = data[pos_mask]
                neg_z = np.abs(z[neg_mask]); neg_d = data[neg_mask]
                if len(pos_z) == 0 or len(neg_z) == 0:
                    return z, data
                f_neg = _interp1d(neg_z, neg_d, kind='linear',
                                  bounds_error=False, fill_value=np.nan)
                neg_interp = f_neg(pos_z)
                averaged = np.where(~np.isnan(neg_interp),
                                    (pos_d + neg_interp) / 2.0, pos_d)
                sym_z = np.concatenate([-pos_z[::-1], pos_z])
                sym_d = np.concatenate([averaged[::-1], averaged])
                return sym_z, sym_d
            elif method == 'mirror_positive':
                pos_z = z[pos_mask]; pos_d = data[pos_mask]
                return np.concatenate([-pos_z[::-1], pos_z]), \
                       np.concatenate([pos_d[::-1], pos_d])
            elif method == 'mirror_negative':
                neg_z = z[neg_mask]; neg_d = data[neg_mask]
                return np.concatenate([neg_z, -neg_z[::-1]]), \
                       np.concatenate([neg_d, neg_d[::-1]])
            return z, data

        # ── helper: symmetrize a single boundary value ───────────────────────
        def _sym_boundary(pos_val, neg_val):
            """Return symmetric (positive) boundary position."""
            vals = [v for v in [pos_val, neg_val] if v is not None]
            if not vals:
                return None
            return float(np.mean(np.abs(vals)))

        # ── helper: load one cache file ───────────────────────────────────────
        def _load_cache(folder):
            import sys
            import time
            # In-memory cache: skip disk read if already loaded this session
            # (guard with getattr so instances created before _loaded_caches
            # was added to __init__ still work)
            _mem = self._loaded_caches
            if not force_rerun and folder in _mem:
                print(f"    ✓ INSTANT: Loading from MEMORY cache")
                sys.stdout.flush()
                return _mem[folder]

            p = Path(folder) / cache_filename
            if not p.exists():
                print(f"    ⚠ Cache not found: {p}")
                sys.stdout.flush()
                return None
            
            print(f"    ⟳ DISK LOAD STARTING: {p.name}...", end='', flush=True)
            t0 = time.time()

            # The cache stores self.cation_types / self.anion_types as dicts of
            # MDAnalysis AtomGroup objects.  Their __reduce__ chain is:
            #   AtomGroup → _unpickle(universe, ix)
            #             → Universe._unpickle_U(top, traj)
            #             → TRRReader.__init__(nvt.trr)  ← OSError: file missing
            #
            # We intercept each link in the chain with a purpose-built stub so
            # that only the numpy arrays / plain-dict data we actually need are
            # restored.

            class _TrajectoryStub:
                """Stub for XDR/DCD/TRR file-reader objects.
                Absorbs all args; never opens a file."""
                def __init__(self, *a, **kw):   pass
                def close(self):                pass
                def __len__(self):              return 0
                def __iter__(self):             return iter([])
                def __del__(self):              pass
                def __getstate__(self):         return {}
                def __setstate__(self, s):      pass
                def __reduce__(self):           return (self.__class__, ())

            class _UniverseStub:
                """Stub for MDAnalysis Universe.
                Must expose _unpickle_U so pickle can finish reconstructing
                the Universe object stored inside the AtomGroups."""
                def __init__(self, *a, **kw):   pass
                @classmethod
                def _unpickle_U(cls, *args, **kwargs):
                    # Accept any signature variant across MDAnalysis versions
                    return cls()
                def __reduce__(self):           return (self.__class__, ())

            class _AtomGroupStub:
                """Stub for AtomGroup / ResidueGroup / SegmentGroup."""
                def __init__(self, *a, **kw):   pass
                def __len__(self):              return 0
                def __iter__(self):             return iter([])
                def __reduce__(self):           return (self.__class__, ())

            def _unpickle_stub(*a, **kw):
                """Stub for MDAnalysis.core.groups._unpickle function."""
                return _AtomGroupStub()

            # Modules whose classes are file-reader stubs
            _TRAJ_PREFIXES = (
                'MDAnalysis.lib.formats.libmdaxdr',
                'MDAnalysis.lib.formats.libdcd',
                'MDAnalysis.lib.formats.cython_util',
                'MDAnalysis.coordinates.XDR',
                'MDAnalysis.coordinates.TRR',
                'MDAnalysis.coordinates.XTC',
                'MDAnalysis.coordinates.DCD',
                'MDAnalysis.coordinates.base',
                'MDAnalysis.coordinates.core',
            )

            class _SafeUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # 1. File-reader / trajectory classes → _TrajectoryStub
                    if any(module.startswith(pfx) for pfx in _TRAJ_PREFIXES):
                        return _TrajectoryStub
                    # 2. Universe class (needs _unpickle_U classmethod)
                    if module == 'MDAnalysis.core.universe' and name == 'Universe':
                        return _UniverseStub
                    # 3. AtomGroup module-level _unpickle reconstruction fn
                    if module == 'MDAnalysis.core.groups' and name == '_unpickle':
                        return _unpickle_stub
                    # 4. Other group classes (AtomGroup, ResidueGroup, etc.)
                    if module == 'MDAnalysis.core.groups':
                        return _AtomGroupStub
                    return super().find_class(module, name)

            try:
                with open(p, 'rb') as fh:
                    data = _SafeUnpickler(fh).load()
                elapsed = time.time() - t0
                print(f" DONE in {elapsed:.1f}s")
                sys.stdout.flush()
                if save_cache:
                    _mem[folder] = data
                return data
            except Exception as e:
                print(f"\n    ✗ Failed to load cache: {e}")
                sys.stdout.flush()
                return None

        # ── helper: colour gradient from a base hex/named colour ─────────────
        def _gradient_colors(base_color, n):
            """n colours from 35 % to 100 % of base_color lightness."""
            import matplotlib.colors as mc
            rgba = np.array(mc.to_rgba(base_color))
            white = np.array([1.0, 1.0, 1.0, 1.0])
            fracs = np.linspace(0.35, 1.0, n)
            return [tuple(frac * rgba + (1 - frac) * white) for frac in fracs]

        # ═══════════════════════════════════════════════════════════════════
        # 1. Collect (label, color, cache_data) tuples
        # ═══════════════════════════════════════════════════════════════════
        import sys
        # Lazy-init: handles instances created before _loaded_caches was added
        if not hasattr(self, '_loaded_caches'):
            self._loaded_caches = {}
        print(f"\n═══ compare_cation_density_profiles START ═══")
        print(f"In-memory cache contains {len(self._loaded_caches)} folders")
        print(f"force_rerun={force_rerun}")
        sys.stdout.flush()
        
        datasets = []   # list of dicts: {label, color, cache}

        # ── Ion-type canonical colours (user-overridable) ─────────────────
        _DEFAULT_ION_COLORS = {
            # Monovalent cations
            'LI': 'dodgerblue',
            'NA': 'blue',
            'K':  'purple',
            'RB': 'darkviolet',
            'CS': 'indigo',
            # Divalent cations
            'MG': 'green',
            'CA': 'orange',
            'SR': 'darkorange',
            'BA': 'saddlebrown',
            'ZN': 'teal',
            'FE': 'sienna',
            'CU': 'peru',
            'MN': 'orchid',
            'CO': 'mediumseagreen',
            'NI': 'darkcyan',
            # Trivalent cations
            'AL': 'slategray',
            # Anions
            'CL': 'red',
            'BR': 'firebrick',
            'F':  'limegreen',
            'I':  'darkmagenta',
        }
        _ion_colors = {**_DEFAULT_ION_COLORS, **(ion_type_colors or {})}

        # ── Ion-type display names (pretty labels for legend) ─────────────
        _DEFAULT_ION_LABELS = {
            # Monovalent cations
            'LI': 'Li', 'NA': 'Na', 'K': 'K', 'RB': 'Rb', 'CS': 'Cs',
            # Divalent cations
            'MG': 'Mg', 'CA': 'Ca', 'SR': 'Sr', 'BA': 'Ba',
            'ZN': 'Zn', 'FE': 'Fe', 'CU': 'Cu', 'MN': 'Mn',
            'CO': 'Co', 'NI': 'Ni',
            # Trivalent
            'AL': 'Al',
            # Anions
            'CL': 'Cl', 'BR': 'Br', 'F': 'F', 'I': 'I',
        }
        _ion_display = {**_DEFAULT_ION_LABELS, **(ion_type_labels or {})}

        if mode == 'by_concentration':
            if salt_key is None:
                raise ValueError("mode='by_concentration' requires salt_key.")
            if salt_key not in self.salt_data:
                raise KeyError(f"salt_key '{salt_key}' not found in salt_data.")

            salt_info = self.salt_data[salt_key]
            conc_keys = list(salt_info['concentrations'].keys())
            if concentrations is not None:
                conc_keys = [c for c in conc_keys if c in concentrations]
            # Sort concentrations numerically (e.g. '0.11M' < '0.42M' < '0.81M')
            import re as _re
            def _parse_conc(c):
                m = _re.search(r'[\d.]+', c)
                return float(m.group()) if m else 0.0
            conc_keys = sorted(conc_keys, key=_parse_conc)

            base_color = salt_info.get('color', 'steelblue')
            if not conc_colors and ion_types is not None and len(ion_types) == 1:
                base_color = _ion_colors.get(ion_types[0].upper(), base_color)
            colors = conc_colors if conc_colors else _gradient_colors(base_color, len(conc_keys))

            for i, ck in enumerate(conc_keys):
                conc_info = salt_info['concentrations'][ck]
                folder = conc_info['folder']
                label = conc_info.get('label', ck)
                print(f"  → Checking {label}: {folder}")
                sys.stdout.flush()
                cache = _load_cache(folder)
                if cache is None:
                    continue
                datasets.append({'label': label, 'color': colors[i], 'cache': cache})

        elif mode == 'by_salt':
            if concentration is None:
                raise ValueError("mode='by_salt' requires concentration.")

            # Slash notation: 'XM/YM' → charge-2 (divalent) salts use X,
            # charge-1 (monovalent) salts use Y.
            # Requires 'cation_charge' key in each salt_data entry.
            # Example: '0.06M/0.11M', '0.42M/0.81M'
            _slash_concs = None
            if '/' in concentration:
                _parts = [p.strip() for p in concentration.split('/', 1)]
                _slash_concs = {2: _parts[0], 1: _parts[1]}

            salt_keys = list(self.salt_data.keys())
            if salts is not None:
                salt_keys = [s for s in salt_keys if s in salts]

            # Slash notation requires 'cation_charge' on every salt — validate
            # up-front so a missing key is a loud error, never a silent wrong result.
            if _slash_concs is not None and concentration_map is None:
                missing_charge = [sk for sk in salt_keys
                                  if 'cation_charge' not in self.salt_data[sk]]
                if missing_charge:
                    raise ValueError(
                        f"Slash notation ('{concentration}') requires 'cation_charge' "
                        f"in every salt_data entry.\n"
                        f"  Missing for: {missing_charge}\n"
                        f"  Add  'cation_charge': 1  or  'cation_charge': 2  to those "
                        f"entries, or use  concentration_map={{...}}  instead."
                    )

            for sk in salt_keys:
                salt_info = self.salt_data[sk]
                # Priority: (1) concentration_map per-salt override,
                #           (2) slash notation via cation_charge,
                #           (3) plain concentration string
                if concentration_map is not None:
                    actual_conc = concentration_map.get(sk, concentration)
                elif _slash_concs is not None:
                    charge = salt_info['cation_charge']   # guaranteed present (validated above)
                    actual_conc = _slash_concs.get(charge, _parts[-1])
                else:
                    actual_conc = concentration
                if actual_conc not in salt_info['concentrations']:
                    print(f"  ⚠  '{sk}' has no concentration '{actual_conc}', skipping.")
                    continue
                conc_info = salt_info['concentrations'][actual_conc]
                folder = conc_info['folder']
                label = salt_info.get('label', sk)
                color = salt_info.get('color', None)
                print(f"  → Checking {label}: {folder}")
                sys.stdout.flush()
                cache = _load_cache(folder)
                if cache is None:
                    continue
                datasets.append({'label': label, 'color': color, 'cache': cache})
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'by_concentration' or 'by_salt'.")

        if not datasets:
            print("No data loaded. Check folder paths and cache filenames.")
            return {}

        print(f"\n✓ Loaded {len(datasets)} datasets")
        print(f"  Memory cache now contains {len(self._loaded_caches)} folders")
        sys.stdout.flush()

        if print_summary:
            print(f"\n── compare_cation_density_profiles (mode={mode}) ──")
            for d in datasets:
                n = d['cache'].get('n_frames_analyzed', '?')
                print(f"  {d['label']:25s}  frames={n}")

        # ═══════════════════════════════════════════════════════════════════
        # 2. Determine ion_types to plot
        # ═══════════════════════════════════════════════════════════════════
        if ion_types is None:
            first_results = datasets[0]['cache']['results']
            ion_types = list(first_results.ion_densities_by_type.keys())

        default_linestyles = ['-', '--', '-.', ':']
        if ion_type_linestyles is None:
            ion_type_linestyles = {
                it: default_linestyles[i % len(default_linestyles)]
                for i, it in enumerate(ion_types)
            }

        # ═══════════════════════════════════════════════════════════════════
        # 3. Collect boundary positions (average across all loaded datasets)
        #    Priority: (1) clay_boundaries param, (2) self._clay_boundaries,
        #              (3) read from each dataset's cached results
        # ═══════════════════════════════════════════════════════════════════
        _cb_override = clay_boundaries if clay_boundaries is not None \
                       else getattr(self, '_clay_boundaries', None)

        if _cb_override is not None:
            si_bound   = _sym_boundary(_cb_override.get('si_average_z_positive'),
                                       _cb_override.get('si_average_z_negative'))
            mgo_bound  = _sym_boundary(_cb_override.get('mgo_average_z_positive'),
                                       _cb_override.get('mgo_average_z_negative'))
            clay_bound = _sym_boundary(_cb_override.get('clay_average_z_positive'),
                                       _cb_override.get('clay_average_z_negative'))
        else:
            si_positions   = []
            mgo_positions  = []
            clay_positions = []

            for d in datasets:
                results = d['cache']['results']
                if not hasattr(results, 'clay_interface_boundaries'):
                    continue
                cb = results.clay_interface_boundaries
                if cb is None:
                    continue

                si_pos  = _sym_boundary(cb.get('si_average_z_positive'),
                                        cb.get('si_average_z_negative'))
                mgo_pos = _sym_boundary(cb.get('mgo_average_z_positive'),
                                        cb.get('mgo_average_z_negative'))
                clay_pos = _sym_boundary(cb.get('clay_average_z_positive'),
                                         cb.get('clay_average_z_negative'))

                if si_pos is not None:   si_positions.append(si_pos)
                if mgo_pos is not None:  mgo_positions.append(mgo_pos)
                if clay_pos is not None: clay_positions.append(clay_pos)

            # Consensus boundary (mean across loaded systems)
            si_bound   = float(np.mean(si_positions))   if si_positions   else None
            mgo_bound  = float(np.mean(mgo_positions))  if mgo_positions  else None
            clay_bound = float(np.mean(clay_positions)) if clay_positions else None

        if print_summary and (si_bound or mgo_bound or clay_bound):
            print(f"  Clay boundaries (consensus, |z|):  "
                  f"Si={si_bound:.2f} Å,  Mgo={mgo_bound:.2f} Å,  "
                  f"Clay avg={clay_bound:.2f} Å")

        # ── x-axis origin shift ───────────────────────────────────────────
        # x_origin='center' (default): no shift — z=0 stays at clay centre.
        # x_origin='si'/'mgo': Si/MgO surface becomes x = 0.
        #
        # Convention (both halves):  positive label = towards water,
        #                            negative label = into clay.
        #
        # For show_half='positive':
        #   label = si_bound - z  (si_bound - si_bound = 0;  z<si → positive water
        #                          z>si → negative clay)
        #   xlim reversed to (z_max, 0) so axis reads left=clay, right=water.
        # For show_half='negative':
        #   label = z - (-si_bound) = z + si_bound
        #   (z = -si_bound → 0;  z > -si_bound towards water → positive;
        #    z < -si_bound into clay → negative)
        #   xlim = (-z_max, 0)  — already increases left-to-right naturally.
        _x_flip  = False   # True only for positive half with non-center origin
        if x_origin == 'center':
            _x_shift = 0.0
        elif x_origin == 'si':
            if si_bound is None:
                raise ValueError("x_origin='si' requires a valid Si boundary position.")
            if show_half == 'both':
                _x_shift = 0.0        # symmetric plot: keep origin at clay centre
            elif show_half == 'negative':
                _x_shift = -si_bound  # label = z + si_bound  (correct already)
            else:  # 'positive'
                _x_shift = si_bound   # label = si_bound - z  (via _x_flip)
                _x_flip  = True
        elif x_origin == 'mgo':
            if mgo_bound is None:
                raise ValueError("x_origin='mgo' requires a valid MgO boundary position.")
            if show_half == 'both':
                _x_shift = 0.0
            elif show_half == 'negative':
                _x_shift = -mgo_bound
            else:  # 'positive'
                _x_shift = mgo_bound
                _x_flip  = True
        else:
            raise ValueError(f"x_origin must be 'center', 'si', or 'mgo'. Got '{x_origin}'.")

        # ═══════════════════════════════════════════════════════════════════
        # 4. Build figure
        # ═══════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=figsize)

        # ── clay region shading ───────────────────────────────────────────
        # Fill between Si (inner clay surface, facing water) and MgO (octahedral
        # centre), mirroring ZDirectionalPlotter.plot_clay_interface_boundaries.
        # This correctly starts at the Si boundary and has the true clay-slab
        # thickness (MgO − Si) rather than filling to the plot edge.
        if show_clay_fill and si_bound is not None and mgo_bound is not None:
            ax.axvspan( si_bound,  mgo_bound,
                        color=clay_fill_color, alpha=clay_fill_alpha, zorder=0)
            ax.axvspan(-mgo_bound, -si_bound,
                        color=clay_fill_color, alpha=clay_fill_alpha, zorder=0)

        # ── clay boundary lines ───────────────────────────────────────────
        if show_boundary_lines:
            for sign in (+1, -1):
                if si_bound is not None:
                    ax.axvline(sign * si_bound,
                               color=si_color, lw=boundary_linewidth,
                               ls=si_linestyle, alpha=boundary_alpha,
                               label=('Si boundary' if sign == +1 else None) if show_clay_boundaries_in_legend else '_nolegend_',
                               zorder=1)
                if mgo_bound is not None:
                    ax.axvline(sign * mgo_bound,
                               color=mgo_color, lw=boundary_linewidth,
                               ls=mgo_linestyle, alpha=boundary_alpha,
                               label=('MgO boundary' if sign == +1 else None) if show_clay_boundaries_in_legend else '_nolegend_',
                               zorder=1)
                if clay_bound is not None:
                    ax.axvline(sign * clay_bound,
                               color=clay_avg_color, lw=boundary_linewidth,
                               ls=clay_avg_linestyle, alpha=boundary_alpha,
                               label='_nolegend_',
                               zorder=1)

        # ── z=0 reference ────────────────────────────────────────────────
        if show_zero_line:
            ax.axvline(0.0, color=zero_line_color, ls=zero_line_style,
                       alpha=zero_line_alpha, lw=1.0, zorder=1)

        # ═══════════════════════════════════════════════════════════════════
        # 5. Plot density curves
        # ═══════════════════════════════════════════════════════════════════
        # ── Per-dataset linewidth / alpha gradient ────────────────────────
        _n_ds = len(datasets)
        _lw_vals = (list(np.linspace(linewidth_range[0], linewidth_range[1], _n_ds))
                    if linewidth_range is not None and _n_ds > 1
                    else [linewidth] * _n_ds)
        _alpha_vals = (list(np.linspace(line_alpha_range[0], line_alpha_range[1], _n_ds))
                       if line_alpha_range is not None and _n_ds > 1
                       else [line_alpha] * _n_ds)
        for _gi, _gd in enumerate(datasets):
            _gd['_lw']    = _lw_vals[_gi]
            _gd['_alpha'] = _alpha_vals[_gi]

        plot_data = {}

        for d in datasets:
            cache   = d['cache']
            results = cache['results']
            z_raw   = cache['z_centers'].copy()
            label   = d['label']
            color   = d['color']

            entry = {'z': z_raw}

            for ion_type in ion_types:
                if ion_type not in results.ion_densities_by_type:
                    continue

                dens = results.ion_densities_by_type[ion_type].copy()

                if symmetrize:
                    z_plot, dens_plot = _sym(z_raw, dens, symmetrize_method)
                else:
                    z_plot, dens_plot = z_raw, dens

                ls = ion_type_linestyles.get(ion_type, '-')

                # label: include ion_type only when multiple ion types are requested
                _it_display = _ion_display.get(ion_type.upper(), ion_type)
                if len(ion_types) > 1:
                    curve_label = f"{label} ({_it_display})"
                else:
                    curve_label = label

                # Color logic:
                # - by_salt: always use canonical ion colour (each salt already
                #   has a distinct colour so ion type needs to be the differentiator)
                # - by_concentration, multiple ion types: use canonical ion colour
                #   (concentration is differentiated by linewidth/alpha/linestyle)
                # - by_concentration, single ion type: use gradient colour
                #   (concentration is differentiated by colour shade)
                if mode == 'by_salt' or len(ion_types) > 1:
                    plot_color = _ion_colors.get(ion_type.upper(), color)
                else:
                    plot_color = color

                if ion_type_zorder is None:
                    _zline = 3
                elif isinstance(ion_type_zorder, dict):
                    _zline = ion_type_zorder.get(ion_type.upper(),
                             ion_type_zorder.get(ion_type, 3))
                else:
                    # List of names — priority order: first = on top.
                    # Listed ions get zorder 3+N down to 3+1; unlisted get 3.
                    # e.g. ['K', 'NA', 'CL'] → K=6, NA=5, CL=4, others=3
                    _priority = {it.upper(): len(ion_type_zorder) - i
                                 for i, it in enumerate(ion_type_zorder)}
                    _zline = 3 + _priority.get(ion_type.upper(), 0)
                _zfill = max(1, _zline - 1)

                ax.plot(z_plot, dens_plot,
                        color=plot_color, lw=d['_lw'],
                        ls=ls, alpha=d['_alpha'],
                        label=curve_label, zorder=_zline)

                if fill_curves:
                    ax.fill_between(z_plot, dens_plot,
                                    color=plot_color, alpha=fill_alpha, zorder=_zfill)

                if fill_ion_types is not None and ion_type.upper() in [
                        it.upper() for it in fill_ion_types]:
                    ax.fill_between(z_plot, 0, dens_plot,
                                    color=plot_color, alpha=ion_fill_alpha, zorder=_zfill)

                entry[ion_type] = dens_plot

            plot_data[label] = entry

        # ═══════════════════════════════════════════════════════════════════
        # 5b. Water density overlay (secondary y-axis)
        # ═══════════════════════════════════════════════════════════════════
        _ax2 = None
        if show_water_density:
            _w_arrays = []
            _w_z_ref  = None
            for _wd in datasets:
                _wc  = _wd['cache']
                _wz  = _wc['z_centers'].copy()
                _wda = _wc.get('water_density', None)
                if _wda is None:
                    _res = _wc.get('results', None)
                    if _res is not None and hasattr(_res, 'water_density'):
                        _wda = _res.water_density
                if _wda is None:
                    continue
                _wda = _wda.copy()
                if symmetrize:
                    _wz, _wda = _sym(_wz, _wda, symmetrize_method)
                _w_arrays.append(_wda)
                if _w_z_ref is None:
                    _w_z_ref = _wz
            if _w_arrays and _w_z_ref is not None:
                _w_mean = np.mean(np.vstack(_w_arrays), axis=0)
                _ax2 = ax.twinx()
                # Draw primary axis ON TOP of secondary so ion curves are
                # above the water fill, but make its background transparent
                # so the water fill beneath is visible.
                ax.set_zorder(_ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
                if show_water_line:
                    _ax2.plot(_w_z_ref, _w_mean,
                              color=water_density_color, lw=water_linewidth,
                              alpha=water_density_alpha, zorder=3, label='Water')
                if fill_water_curve:
                    _ax2.fill_between(_w_z_ref, 0, _w_mean,
                                      color=water_density_color,
                                      alpha=water_fill_alpha, zorder=2)
                if show_water_line:
                    _ax2.set_ylabel(water_density_ylabel,
                                    fontsize=label_fontsize,
                                    fontweight=label_fontweight,
                                    color=water_density_color)
                    _ax2.tick_params(axis='y', labelsize=tick_fontsize,
                                     labelcolor=water_density_color)
                    for _tl in _ax2.get_yticklabels():
                        _tl.set_fontweight(tick_fontweight)
                else:
                    # Fill-only mode: hide right y-axis entirely
                    _ax2.set_ylabel('')
                    _ax2.tick_params(axis='y', which='both',
                                     left=False, right=False,
                                     labelleft=False, labelright=False)
                    _ax2.set_yticks([])
                _ax2.set_ylim(bottom=0)

        # ═══════════════════════════════════════════════════════════════════
        # 6. Axes formatting
        # ═══════════════════════════════════════════════════════════════════
        ax.set_xlabel(xlabel_text, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(ylabel_text, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        # Set tick label font weight (tick_params doesn't support labelweight)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight(tick_fontweight)

        if xlim is not None:
            ax.set_xlim(xlim)
        elif show_half != 'both':
            _z_max = float(np.max(np.abs(datasets[0]['cache']['z_centers'])))
            if show_half == 'positive':
                ax.set_xlim(0.0, _z_max)
            else:  # 'negative'
                ax.set_xlim(-_z_max, 0.0)

        # ── x-axis tick labels: integers aligned to the Si/MgO origin ─────────
        # Ticks are placed at data-space positions whose labels are exact
        # multiples of _tick_step (e.g. 0, 10, 20 Å from the boundary).
        # Only active when x_origin != 'center' (_x_shift != 0).
        if _x_shift != 0.0:
            import math as _math
            _tick_step = 10
            _xlo, _xhi = ax.get_xlim()
            # label range visible in plot
            if _x_flip:               # positive half: label = _x_shift - x
                _lab_lo = _x_shift - _xhi
                _lab_hi = _x_shift - _xlo
            else:                     # negative half: label = x - _x_shift
                _lab_lo = _xlo - _x_shift
                _lab_hi = _xhi - _x_shift
            _t1   = int(_math.ceil (_lab_lo / _tick_step)) * _tick_step
            _t2   = int(_math.floor(_lab_hi / _tick_step)) * _tick_step
            _labs = list(range(_t1, _t2 + 1, _tick_step))
            _pos  = [(_x_shift - L if _x_flip else L + _x_shift) for L in _labs]
            ax.set_xticks(_pos)
            ax.set_xticklabels([str(L) for L in _labs])
            for _tl in ax.get_xticklabels():
                _tl.set_fontsize(tick_fontsize)
                _tl.set_fontweight(tick_fontweight)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(bottom=0)

        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        if show_title:
            if title_text is None:
                if mode == 'by_concentration':
                    title_text = f"{salt_key} — ion density profiles by concentration"
                else:
                    title_text = f"Ion density profiles — {concentration}"
            ax.set_title(title_text, fontsize=title_fontsize, fontweight=title_fontweight)

        # tight_layout BEFORE legends so bbox_to_anchor positions are respected
        plt.tight_layout()

        if show_legend:
            from matplotlib.lines import Line2D as _L2D
            legend_kwargs = {
                'ncol': legend_ncol,
                'frameon': True,
                'framealpha': legend_framealpha,
                'prop': {'weight': legend_fontweight, 'size': legend_fontsize}
            }
            if legend_bbox is not None:
                legend_kwargs['loc'] = legend_location
                legend_kwargs['bbox_to_anchor'] = legend_bbox
                legend_kwargs['bbox_transform'] = ax.transAxes
            else:
                legend_kwargs['loc'] = legend_location

            if legend_split_by_ion_type and len(ion_types) > 1:
                # ── Concentration legend (gradient width/alpha, gray) ────
                conc_handles = [
                    _L2D([], [], color='gray', ls='-',
                         lw=d['_lw'], alpha=d['_alpha'],
                         label=d['label'])
                    for d in datasets
                ]
                # ── Ion-type legend (canonical color + linestyle) ────────
                ion_handles = [
                    _L2D([], [],
                         color=_ion_colors.get(it.upper(), 'black'),
                         ls=ion_type_linestyles.get(it, '-'),
                         lw=2, label=_ion_display.get(it.upper(), it))
                    for it in ion_types
                ]
                # Draw concentration legend first
                leg_conc = ax.legend(handles=conc_handles, **legend_kwargs)
                ax.add_artist(leg_conc)   # keep it when we add second legend
                # Ion-type legend at its own position
                ion_legend_kwargs = {
                    'frameon': True,
                    'framealpha': legend_framealpha,
                    'prop': {'weight': legend_fontweight, 'size': legend_fontsize},
                }
                if ion_legend_bbox is not None:
                    ion_legend_kwargs['loc'] = ion_legend_location
                    ion_legend_kwargs['bbox_to_anchor'] = ion_legend_bbox
                    ion_legend_kwargs['bbox_transform'] = ax.transAxes
                else:
                    ion_legend_kwargs['loc'] = ion_legend_location
                ax.legend(handles=ion_handles, **ion_legend_kwargs)
            else:
                ax.legend(**legend_kwargs)

        # ═══════════════════════════════════════════════════════════════════
        # 7. Save
        # ═══════════════════════════════════════════════════════════════════
        if save_plot:
            if mode == 'by_concentration':
                fname = (f"z_density_comparison_{salt_key}_"
                         f"{'_'.join(ion_types)}_by_conc.png")
            else:
                _conc_str = concentration.replace('/', '_').replace(' ', '').replace('.', 'p')
                fname = (f"z_density_comparison_{_conc_str}_"
                         f"{'_'.join(ion_types)}_by_salt.png")
            plt.savefig(fname, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved: {fname}")

        plt.show()

        # ═══════════════════════════════════════════════════════════════════
        # 8. Individual per-ion plots
        # ═══════════════════════════════════════════════════════════════════
        if plot_individual and len(ion_types) > 1:
            for _solo_ion in ion_types:
                _solo_color = _ion_colors.get(_solo_ion.upper(), 'steelblue')
                _solo_display = _ion_display.get(_solo_ion.upper(), _solo_ion)
                _solo_ls = ion_type_linestyles.get(_solo_ion, '-')

                fig_s, ax_s = plt.subplots(figsize=figsize)

                # clay fill
                if show_clay_fill and si_bound is not None and mgo_bound is not None:
                    ax_s.axvspan( si_bound,  mgo_bound,
                                  color=clay_fill_color, alpha=clay_fill_alpha, zorder=0)
                    ax_s.axvspan(-mgo_bound, -si_bound,
                                  color=clay_fill_color, alpha=clay_fill_alpha, zorder=0)

                # boundary lines
                if show_boundary_lines:
                    for sign in (+1, -1):
                        if si_bound is not None:
                            ax_s.axvline(sign * si_bound,
                                         color=si_color, lw=boundary_linewidth,
                                         ls=si_linestyle, alpha=boundary_alpha,
                                         label=('Si boundary' if sign == +1 else None) if show_clay_boundaries_in_legend else '_nolegend_',
                                         zorder=1)
                        if mgo_bound is not None:
                            ax_s.axvline(sign * mgo_bound,
                                         color=mgo_color, lw=boundary_linewidth,
                                         ls=mgo_linestyle, alpha=boundary_alpha,
                                         label=('MgO boundary' if sign == +1 else None) if show_clay_boundaries_in_legend else '_nolegend_',
                                         zorder=1)
                        if clay_bound is not None:
                            ax_s.axvline(sign * clay_bound,
                                         color=clay_avg_color, lw=boundary_linewidth,
                                         ls=clay_avg_linestyle, alpha=boundary_alpha,
                                         label='_nolegend_', zorder=1)

                # z=0 line
                if show_zero_line:
                    ax_s.axvline(0.0, color=zero_line_color, ls=zero_line_style,
                                 alpha=zero_line_alpha, lw=1.0, zorder=1)

                # density curves — fixed ion color, lw/alpha gradient by concentration
                for d in datasets:
                    _cache_s   = d['cache']
                    _results_s = _cache_s['results']
                    _z_raw_s   = _cache_s['z_centers'].copy()
                    if _solo_ion not in _results_s.ion_densities_by_type:
                        continue
                    _dens_s = _results_s.ion_densities_by_type[_solo_ion].copy()
                    if symmetrize:
                        _z_s, _d_s = _sym(_z_raw_s, _dens_s, symmetrize_method)
                    else:
                        _z_s, _d_s = _z_raw_s, _dens_s
                    ax_s.plot(_z_s, _d_s,
                              color=_solo_color, lw=d['_lw'],
                              ls=_solo_ls, alpha=d['_alpha'],
                              label=d['label'], zorder=3)
                    if fill_curves:
                        ax_s.fill_between(_z_s, _d_s,
                                          color=_solo_color, alpha=fill_alpha, zorder=2)

                # axes formatting
                ax_s.set_xlabel(xlabel_text, fontsize=label_fontsize, fontweight=label_fontweight)
                ax_s.set_ylabel(ylabel_text, fontsize=label_fontsize, fontweight=label_fontweight)
                ax_s.tick_params(axis='both', labelsize=tick_fontsize)
                for _tl in ax_s.get_xticklabels() + ax_s.get_yticklabels():
                    _tl.set_fontweight(tick_fontweight)
                if xlim is not None:
                    ax_s.set_xlim(xlim)
                elif show_half != 'both':
                    _z_max_s = float(np.max(np.abs(datasets[0]['cache']['z_centers'])))
                    if show_half == 'positive':
                        ax_s.set_xlim(0.0, _z_max_s)
                    else:  # 'negative'
                        ax_s.set_xlim(-_z_max_s, 0.0)

                # ── x-axis tick labels: integers aligned to the Si/MgO origin ────
                if _x_shift != 0.0:
                    import math as _math
                    _tick_step = 10
                    _xlo_s, _xhi_s = ax_s.get_xlim()
                    if _x_flip:
                        _lab_lo_s = _x_shift - _xhi_s
                        _lab_hi_s = _x_shift - _xlo_s
                    else:
                        _lab_lo_s = _xlo_s - _x_shift
                        _lab_hi_s = _xhi_s - _x_shift
                    _t1_s   = int(_math.ceil (_lab_lo_s / _tick_step)) * _tick_step
                    _t2_s   = int(_math.floor(_lab_hi_s / _tick_step)) * _tick_step
                    _labs_s = list(range(_t1_s, _t2_s + 1, _tick_step))
                    _pos_s  = [(_x_shift - L if _x_flip else L + _x_shift)
                                for L in _labs_s]
                    ax_s.set_xticks(_pos_s)
                    ax_s.set_xticklabels([str(L) for L in _labs_s])
                    for _tl in ax_s.get_xticklabels():
                        _tl.set_fontsize(tick_fontsize)
                        _tl.set_fontweight(tick_fontweight)
                if ylim is not None:
                    ax_s.set_ylim(ylim)
                else:
                    ax_s.set_ylim(bottom=0)
                if show_grid:
                    ax_s.grid(True, alpha=grid_alpha)
                if show_title:
                    _solo_title = (title_text or
                                   (f"{salt_key} — {_solo_display} density by concentration"
                                    if mode == 'by_concentration'
                                    else f"{_solo_display} density — {concentration}"))
                    ax_s.set_title(_solo_title, fontsize=title_fontsize,
                                   fontweight=title_fontweight)

                plt.tight_layout()

                if show_legend:
                    _solo_legend_kwargs = {
                        'loc': individual_legend_location,
                        'ncol': legend_ncol,
                        'frameon': True,
                        'framealpha': legend_framealpha,
                        'prop': {'weight': legend_fontweight, 'size': legend_fontsize},
                    }
                    ax_s.legend(**_solo_legend_kwargs)

                if save_plot:
                    if mode == 'by_concentration':
                        _fname_s = f"z_density_comparison_{salt_key}_{_solo_ion}_by_conc.png"
                    else:
                        _cstr_s = concentration.replace('/', '_').replace(' ', '').replace('.', 'p')
                        _fname_s = f"z_density_comparison_{_cstr_s}_{_solo_ion}_by_salt.png"
                    plt.savefig(_fname_s, dpi=dpi, bbox_inches='tight')
                    print(f"✓ Saved: {_fname_s}")

                plt.show()

        return plot_data

    # ─────────────────────────────────────────────────────────────────────────
    def compare_water_dipole_distributions(
        self,
        ion_type,
        mode='by_concentration',
        salt_key=None,
        concentration=None,
        salts=None,
        concentrations=None,
        # per-CN breakdown overlay
        show_per_cn=False,
        cn_alpha=0.4,
        cn_linewidth=1.2,
        # KDE
        kde_bandwidth=None,
        kde_npoints=500,
        # Line style
        linewidth=2.0,
        linewidth_range=None,
        line_alpha=0.85,
        line_alpha_range=None,
        fill_curves=False,
        fill_alpha=0.15,
        # Figure
        figsize=(8, 5),
        dpi=300,
        save_plot=True,
        # Text / fonts
        xlabel_text='Dipole Angle (°)',
        ylabel_text='Probability Density',
        title_text=None,
        show_title=True,
        title_fontsize=14,
        title_fontweight='bold',
        label_fontsize=14,
        label_fontweight='normal',
        tick_fontsize=12,
        tick_fontweight='normal',
        legend_fontsize=11,
        legend_fontweight='normal',
        legend_location='best',
        legend_bbox=None,
        legend_ncol=1,
        legend_framealpha=0.85,
        show_legend=True,
        show_grid=False,
        grid_alpha=0.3,
        xlim=(0, 180),
        ylim=None,
        xticks=None,
        conc_colors=None,
        # Cache
        save_cache=True,
        force_rerun=False,
        print_summary=True,
        concentration_map=None,
    ):
        """
        Compare water-dipole-angle distributions (pooled over all CNs) across
        concentrations or salts.

        For each folder the method loads
        ``dipole_by_coordination_{ion_type}.pkl``, pools the angle arrays over
        all coordination numbers, and draws a Gaussian-KDE curve per dataset
        on a single axes (x = dipole angle 0–180°, y = probability density).

        Parameters
        ----------
        ion_type : str
            GROMACS residue/atom name of the ion (e.g. ``'MG'``, ``'NA'``).
        mode : {'by_concentration', 'by_salt'}
            'by_concentration' — same salt, overlay all concentrations.
            'by_salt'          — same concentration, overlay all salts.
        salt_key : str
            Required for mode='by_concentration'.
        concentration : str
            Required for mode='by_salt'.
        salts : list of str, optional
            Restrict which salts appear in mode='by_salt'.
        concentrations : list of str, optional
            Restrict which concentrations appear in mode='by_concentration'.
        show_per_cn : bool
            If True, also draw thin per-CN KDE lines (dashed, semi-transparent)
            behind the pooled line.
        cn_alpha : float
            Alpha for per-CN lines when show_per_cn=True.
        cn_linewidth : float
            Width of per-CN lines.
        kde_bandwidth : float or str, optional
            Bandwidth for scipy gaussian_kde.  None = Scott's rule.
        kde_npoints : int
            Number of evaluation points on the KDE x-grid.
        linewidth : float
            Width of the pooled-KDE line.
        linewidth_range : tuple (min, max), optional
            If given, linewidths are linearly interpolated light→dark.
        line_alpha : float
            Alpha of the pooled-KDE line.
        line_alpha_range : tuple (min, max), optional
            If given, alphas are linearly interpolated similarly.
        fill_curves : bool
            Fill area under each KDE curve.
        fill_alpha : float
            Alpha for the fill.
        figsize, dpi, save_plot : standard matplotlib parameters.
        xlabel_text, ylabel_text : str
            Axis labels.
        title_text : str, optional
            Override auto-generated title.
        show_title, show_legend : bool
        legend_location : str
        legend_bbox : tuple (x, y), optional
            Positional anchor for the legend (axes fraction coords).
        xlim : tuple
            X-axis limits.  Default (0, 180).
        ylim : tuple, optional
            Y-axis limits.
        xticks : list, optional
            Custom x-tick positions.  Default [0, 30, 60, 90, 120, 150, 180].
        conc_colors : list, optional
            Explicit colours per concentration (mode='by_concentration').
        save_cache : bool
            Cache loaded pkl payloads in memory between calls.
        force_rerun : bool
            Ignore in-memory cache and reload from disk.
        print_summary : bool
            Print a loading summary.
        concentration_map : dict, optional
            Per-salt concentration override for mode='by_salt'.

        Returns
        -------
        dict
            ``{label: {'angles': pooled_array, 'kde_x': ..., 'kde_y': ...,
                       'per_cn': {cn: array}, 'color': ...}}``
        """
        import sys
        import pickle
        import re as _re
        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mc
        from scipy.stats import gaussian_kde

        ion_type_up    = ion_type.upper()
        cache_filename = f'dipole_by_coordination_{ion_type_up}.pkl'

        # ── lazy-init per-ion in-memory cache ────────────────────────────────
        if not hasattr(self, '_loaded_dipole_caches'):
            self._loaded_dipole_caches = {}

        # ── helper: load one dipole pkl ───────────────────────────────────────
        def _load_dipole(folder):
            cache_key = (folder, ion_type_up)
            if not force_rerun and cache_key in self._loaded_dipole_caches:
                return self._loaded_dipole_caches[cache_key]
            p = Path(folder) / cache_filename
            if not p.exists():
                print(f"    ⚠  Dipole cache not found: {p}")
                sys.stdout.flush()
                return None
            try:
                with open(p, 'rb') as fh:
                    payload = pickle.load(fh)
                # Payload format: {'dipole_by_coordination': {ion_type: {cn: array}}}
                # Fall back gracefully if the outer wrapper is absent.
                data = payload.get('dipole_by_coordination', payload)
                if ion_type_up not in data:
                    print(f"    ⚠  ion_type '{ion_type_up}' not found in {p.name}. "
                          f"Available keys: {list(data.keys())}")
                    sys.stdout.flush()
                    return None
                result = data[ion_type_up]   # {cn: np.ndarray of angles}
                if save_cache:
                    self._loaded_dipole_caches[cache_key] = result
                return result
            except Exception as e:
                print(f"    ✗  Failed to load {p}: {e}")
                sys.stdout.flush()
                return None

        # ── helper: colour gradient from a base hex/named colour ─────────────
        def _gradient_colors(base_color, n):
            rgba  = np.array(mc.to_rgba(base_color))
            white = np.array([1.0, 1.0, 1.0, 1.0])
            fracs = np.linspace(0.35, 1.0, n)
            return [tuple(frac * rgba + (1 - frac) * white) for frac in fracs]

        # ── helper: sort concentration strings numerically ────────────────────
        def _parse_conc(c):
            m = _re.search(r'[\d.]+', c)
            return float(m.group()) if m else 0.0

        # ═════════════════════════════════════════════════════════════════════
        # 1. Collect datasets
        # ═════════════════════════════════════════════════════════════════════
        datasets = []   # [{label, color, dipole: {cn: array}}]

        if mode == 'by_concentration':
            if salt_key is None:
                raise ValueError("mode='by_concentration' requires salt_key.")
            if salt_key not in self.salt_data:
                raise KeyError(f"'{salt_key}' not found in salt_data.")
            salt_info  = self.salt_data[salt_key]
            conc_keys  = sorted(salt_info['concentrations'].keys(), key=_parse_conc)
            if concentrations is not None:
                conc_keys = [c for c in conc_keys if c in concentrations]
            base_color = salt_info.get('color', 'steelblue')
            colors     = conc_colors if conc_colors else _gradient_colors(base_color, len(conc_keys))
            for i, ck in enumerate(conc_keys):
                ci     = salt_info['concentrations'][ck]
                folder = ci['folder']
                label  = ci.get('label', ck)
                print(f"  → {label}: {folder}")
                sys.stdout.flush()
                dipole = _load_dipole(folder)
                if dipole is None:
                    continue
                datasets.append({'label': label, 'color': colors[i], 'dipole': dipole})

        elif mode == 'by_salt':
            if concentration is None:
                raise ValueError("mode='by_salt' requires concentration.")
            _slash_concs = None
            if '/' in concentration:
                _parts       = [p.strip() for p in concentration.split('/', 1)]
                _slash_concs = {2: _parts[0], 1: _parts[1]}
            salt_keys = list(self.salt_data.keys())
            if salts is not None:
                salt_keys = [s for s in salt_keys if s in salts]
            for sk in salt_keys:
                salt_info = self.salt_data[sk]
                if concentration_map is not None:
                    actual_conc = concentration_map.get(sk, concentration)
                elif _slash_concs is not None:
                    charge      = salt_info.get('cation_charge', 1)
                    actual_conc = _slash_concs.get(charge, _parts[-1])
                else:
                    actual_conc = concentration
                if actual_conc not in salt_info['concentrations']:
                    print(f"  ⚠  '{sk}' has no concentration '{actual_conc}', skipping.")
                    continue
                ci     = salt_info['concentrations'][actual_conc]
                folder = ci['folder']
                label  = salt_info.get('label', sk)
                color  = salt_info.get('color', None)
                print(f"  → {label}: {folder}")
                sys.stdout.flush()
                dipole = _load_dipole(folder)
                if dipole is None:
                    continue
                datasets.append({'label': label, 'color': color, 'dipole': dipole})
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'by_concentration' or 'by_salt'.")

        if not datasets:
            print("No dipole data loaded. Check folder paths and pkl filenames.")
            return {}

        print(f"\n✓ Loaded {len(datasets)} datasets")
        if print_summary:
            for d in datasets:
                n_pts = sum(len(v) for v in d['dipole'].values())
                cns   = sorted(d['dipole'].keys())
                print(f"  {d['label']:30s}  CNs={cns}  total_angles={n_pts:,}")
        sys.stdout.flush()

        # ═════════════════════════════════════════════════════════════════════
        # 2. Build per-dataset linewidth / alpha arrays
        # ═════════════════════════════════════════════════════════════════════
        n = len(datasets)
        lws    = (np.linspace(linewidth_range[0],   linewidth_range[1],   n).tolist()
                  if linewidth_range   is not None else [linewidth]   * n)
        alphas = (np.linspace(line_alpha_range[0],  line_alpha_range[1],  n).tolist()
                  if line_alpha_range  is not None else [line_alpha]  * n)

        # ═════════════════════════════════════════════════════════════════════
        # 3. Plot
        # ═════════════════════════════════════════════════════════════════════
        x_grid   = np.linspace(0, 180, kde_npoints)
        fig, ax  = plt.subplots(figsize=figsize)
        plot_data = {}

        kde_kw = {} if kde_bandwidth is None else {'bw_method': kde_bandwidth}
        cn_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

        for i, d in enumerate(datasets):
            dipole_data = d['dipole']   # {cn: np.ndarray}
            cns    = sorted(dipole_data.keys())
            arrays = [dipole_data[cn] for cn in cns if len(dipole_data[cn]) > 1]
            if not arrays:
                print(f"  ⚠  No angle data for {d['label']}, skipping.")
                continue
            pooled = np.concatenate(arrays)
            kde    = gaussian_kde(pooled, **kde_kw)
            y_grid = kde(x_grid)

            ax.plot(x_grid, y_grid,
                    color=d['color'], lw=lws[i], alpha=alphas[i],
                    label=d['label'])
            if fill_curves:
                ax.fill_between(x_grid, y_grid, alpha=fill_alpha, color=d['color'])

            # Optional per-CN overlay
            if show_per_cn:
                for j, cn in enumerate(cns):
                    arr = dipole_data[cn]
                    if len(arr) < 2:
                        continue
                    y_cn = gaussian_kde(arr, **kde_kw)(x_grid)
                    ax.plot(x_grid, y_cn,
                            color=d['color'], lw=cn_linewidth, alpha=cn_alpha,
                            linestyle=cn_linestyles[j % len(cn_linestyles)],
                            label=f"{d['label']} CN={cn}")

            plot_data[d['label']] = {
                'angles': pooled,
                'kde_x':  x_grid,
                'kde_y':  y_grid,
                'per_cn': dipole_data,
                'color':  d['color'],
            }

        # ═════════════════════════════════════════════════════════════════════
        # 4. Axes decoration
        # ═════════════════════════════════════════════════════════════════════
        ax.set_xlabel(xlabel_text, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel(ylabel_text, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight(tick_fontweight)

        ax.set_xticks(xticks if xticks is not None else [0, 30, 60, 90, 120, 150, 180])
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if show_grid:
            ax.grid(True, alpha=grid_alpha)

        if show_title:
            _ion_display = {
                'MG': 'Mg²⁺', 'CA': 'Ca²⁺', 'NA': 'Na⁺', 'K': 'K⁺',
                'CL': 'Cl⁻',  'LI': 'Li⁺',  'SR': 'Sr²⁺', 'BA': 'Ba²⁺',
            }.get(ion_type_up, ion_type_up)
            if title_text:
                _title = title_text
            elif mode == 'by_concentration':
                _title = f"{salt_key} — Water Dipole near {_ion_display}"
            else:
                _title = f"Water Dipole near {_ion_display} — {concentration}"
            ax.set_title(_title, fontsize=title_fontsize, fontweight=title_fontweight)

        if show_legend:
            _leg_kw = dict(
                loc=legend_location,
                ncol=legend_ncol,
                frameon=True,
                framealpha=legend_framealpha,
                prop={'weight': legend_fontweight, 'size': legend_fontsize},
            )
            if legend_bbox is not None:
                _leg_kw['loc'] = 'upper left'
                _leg_kw['bbox_to_anchor'] = legend_bbox
            ax.legend(**_leg_kw)

        plt.tight_layout()

        if save_plot:
            if mode == 'by_concentration':
                _fname = f"dipole_comparison_{salt_key}_{ion_type_up}_by_conc.png"
            else:
                _cstr  = (concentration.replace('/', '_')
                                       .replace(' ', '')
                                       .replace('.', 'p'))
                _fname = f"dipole_comparison_{_cstr}_{ion_type_up}_by_salt.png"
            plt.savefig(_fname, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved: {_fname}")

        plt.show()
        return plot_data
