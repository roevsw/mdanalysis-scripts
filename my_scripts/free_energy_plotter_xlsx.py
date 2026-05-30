"""
free_energy_plotter_xlsx.py
===========================
Class for loading and plotting solvation free energy (FEP) results from Excel
files.  Designed for CIP-/CIP+/--/CIP+ systems but fully configurable for any
grouped-bar FEP dataset.

Usage
-----
from free_energy_plotter_xlsx import FreeEnergyPlotterXLSX

plotter = FreeEnergyPlotterXLSX(
    fep_path='/path/to/FEP_results.xlsx',
    font_family='Times New Roman',
)
fig = plotter.plot_solvation_barplot(
    save_path='FEP_solvation_barchart.png',
)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Default styling constants (mirrors rmsd_plotter.py conventions)
# ---------------------------------------------------------------------------
_DEFAULT_COLORS  = ['#E07B7B', '#7BB3E0', '#7BE09A', '#E0D07B', '#C07BE0',
                    '#E0A87B', '#7BE0D8', '#B07BE0', '#E07BAE', '#A0A0A0']
_DEFAULT_HATCHES = ['', '///', 'xxx', '\\\\\\', '...', '|||', '+++', 'ooo',
                    '**', 'oo']


class FreeEnergyPlotterXLSX:
    """
    Load solvation free energies from an Excel file and produce publication-
    quality grouped bar charts.

    Parameters
    ----------
    fep_path : str
        Absolute path to the FEP results Excel file.
    font_family : str
        Matplotlib font family used for all text (default: 'Times New Roman').
    mathtext_fontset : str
        Mathtext renderer font set (default: 'stix', best for Times-like fonts).
    """

    def __init__(
        self,
        fep_path: Optional[str] = None,
        font_family: str = 'Times New Roman',
        mathtext_fontset: str = 'stix',
    ) -> None:
        self.fep_path = fep_path
        self.font_family = font_family
        self.mathtext_fontset = mathtext_fontset

        # Apply font globally on construction
        matplotlib.rcParams['font.family'] = self.font_family
        matplotlib.rcParams['mathtext.fontset'] = self.mathtext_fontset

    # ------------------------------------------------------------------ #
    # Static helpers for column auto-detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _col(raw: pd.DataFrame, spec: Union[int, str]) -> pd.Series:
        """Return a column from *raw* by integer index or string name."""
        if isinstance(spec, int):
            return raw.iloc[:, spec]
        return raw[spec]

    @staticmethod
    def _find_header_row(raw: pd.DataFrame) -> int:
        """
        Scan every row of *raw* and score it by how many of its cells contain
        string values that match known FEP column-label keywords.

        Returns the 0-based row index of the best candidate header row.
        Raises ``ValueError`` if no row scores >= 2 matches.
        """
        _KEYWORDS = [
            'conc', 'condition', 'group', 'system',
            'specie', 'species', 'ion', 'solute',
            'e (', 'energy', 'kj', 'kcal', 'dg', 'ddg',
            'err', 'error', '±', '+/-', 'std', 'sem',
            'name', 'label', 'type', 'mol',
        ]
        best_row, best_score = 0, 0
        for row_idx in range(len(raw)):
            cells = [str(v).strip().lower() for v in raw.iloc[row_idx] if pd.notna(v) and str(v).strip()]
            score = sum(any(kw in cell for kw in _KEYWORDS) for cell in cells)
            if score > best_score:
                best_score = score
                best_row   = row_idx
        if best_score < 2:
            raise ValueError(
                f"Could not find a header row. Best candidate was row {best_row} "
                f"with only {best_score} keyword match(es).\n"
                f"Row {best_row} values: {list(raw.iloc[best_row])}\n"
                "Pass header_row= explicitly to load_data()."
            )
        return best_row

    @staticmethod
    def _find_col(raw: pd.DataFrame, patterns: List[str], label: str) -> int:
        """
        Search column names of *raw* (case-insensitive) for the first pattern
        match.  Returns the 0-based integer column index.

        Raises ``ValueError`` with the full column list when nothing matches.
        """
        col_names = [str(c).strip().lower() for c in raw.columns]
        for pat in patterns:
            pat_l = pat.lower()
            for idx, name in enumerate(col_names):
                if pat_l in name:
                    return idx
        raise ValueError(
            f"Could not auto-detect the '{label}' column.\n"
            f"Tried patterns : {patterns}\n"
            f"Found columns  : {list(raw.columns)}\n"
            f"Pass '{label.replace(' ', '_')}_col=<index or name>' explicitly."
        )

    # ------------------------------------------------------------------ #

    def load_data(
        self,
        condition_col: Union[int, str, None] = None,
        species_col:   Union[int, str, None] = None,
        energy_col:    Union[int, str, None] = None,
        error_col:     Union[int, str, None] = None,
        header_row:    Optional[int]         = None,
        ffill_condition: bool                = True,
        sheet_name:    Union[int, str]       = 0,
    ) -> pd.DataFrame:
        """
        Load FEP data from :attr:`fep_path` and return a tidy DataFrame with
        columns ``['condition', 'species', 'E', 'err']``.

        The method starts completely blind — it reads the file exactly as-is
        (``header=None``) and then uses heuristics to locate the header row,
        identify data columns, and clean the data.  No assumptions are made
        about the number of rows, columns, blank rows, or column positions.

        Parameters
        ----------
        condition_col : int, str, or None
            Column for the experiment condition / group label.
            Auto-detected via keywords: 'conc', 'condition', 'group', 'system'.
        species_col : int, str, or None
            Column for the species / ion label.
            Auto-detected via keywords: 'specie', 'species', 'ion', 'solute'.
        energy_col : int, str, or None
            Column for the free energy values.
            Auto-detected via keywords: 'e (', 'energy', 'kj', 'kcal', 'dg', 'ddg'.
        error_col : int, str, or None
            Column for the uncertainty / error.
            Auto-detected via keywords: 'err', 'error', '±', '+/-', 'std', 'sem'.
        header_row : int, optional
            Override the auto-detected header row (0-based index in the raw
            file).  Use this only if auto-detection picks the wrong row.
        ffill_condition : bool
            Forward-fill the condition column to propagate group labels across
            rows that leave it blank (default: ``True``).
        sheet_name : int or str
            Sheet to read (default: first sheet, index 0).

        Returns
        -------
        pd.DataFrame
            Columns: ``['condition', 'species', 'E', 'err']``.
        """
        if self.fep_path is None:
            raise ValueError(
                'fep_path is None — set it in the constructor before calling load_data().'
            )

        # ── Step 1: read the whole file with zero assumptions ─────────────
        raw_full = pd.read_excel(self.fep_path, header=None, sheet_name=sheet_name)

        # ── Step 2: locate the header row ─────────────────────────────────
        hdr = header_row if header_row is not None else self._find_header_row(raw_full)

        # ── Step 3: promote that row as column names; keep only rows below ─
        raw = raw_full.iloc[hdr + 1:].copy()
        raw.columns = [str(v).strip() if pd.notna(v) else f'_col{i}'
                       for i, v in enumerate(raw_full.iloc[hdr])]
        raw = raw.reset_index(drop=True)

        # ── Step 4: drop completely empty rows ────────────────────────────
        raw = raw.dropna(how='all').reset_index(drop=True)

        # ── Step 5: resolve each data column ──────────────────────────────
        if condition_col is None:
            condition_col = self._find_col(raw, ['conc', 'condition', 'group', 'system'], 'condition')
        if species_col is None:
            species_col = self._find_col(raw, ['specie', 'species', 'ion', 'solute'], 'species')
        if energy_col is None:
            energy_col = self._find_col(raw, ['e (', 'energy', 'kj', 'kcal', 'dg', 'ddg'], 'energy')
        if error_col is None:
            error_col = self._find_col(raw, ['err', 'error', '±', '+/-', 'std', 'sem'], 'error')

        # ── Step 6: build tidy output ─────────────────────────────────────
        condition_vals = self._col(raw, condition_col).copy()
        if ffill_condition:
            condition_vals = condition_vals.ffill()

        df = pd.DataFrame({
            'condition': condition_vals,
            'species':   self._col(raw, species_col),
            'E':         pd.to_numeric(self._col(raw, energy_col),  errors='coerce'),
            'err':       pd.to_numeric(self._col(raw, error_col),   errors='coerce'),
        }).dropna(subset=['E', 'species'])
        df = df.reset_index(drop=True)
        return df

    # ---------------------------------------------------------------------- #
    # Main plotting method
    # ---------------------------------------------------------------------- #

    def plot_solvation_barplot(
        self,
        # ── Data ────────────────────────────────────────────────────────────
        df: Optional[pd.DataFrame]         = None,
        conditions: Optional[List[str]]    = None,
        species_list: Optional[List[str]]  = None,
        # ── Figure layout ───────────────────────────────────────────────────
        figsize: Tuple[float, float]       = (10, 6),
        # ── Bar geometry ────────────────────────────────────────────────────
        bar_width: float                   = 0.22,
        group_spacing: float               = 1.0,
        bar_alpha: float                   = 0.85,
        edgecolor: str                     = 'black',
        edgewidth: float                   = 1.0,
        # ── Grouping ────────────────────────────────────────────────────────
        group_by: str                      = 'condition',   # 'condition' | 'species'
        # ── Colors & hatches (one entry per inner-group item) ───────────────
        colors: Optional[List[str]]        = None,
        hatches: Optional[List[str]]       = None,
        # ── Error bars ──────────────────────────────────────────────────────
        show_error_bars: bool              = True,
        error_capsize: float               = 4.0,
        error_linewidth: float             = 1.5,
        error_color: str                   = 'black',
        # ── Axis labels & title ─────────────────────────────────────────────
        ylabel: str                        = 'Hydration free energy (kJ/mol)',
        xlabel: Optional[str]              = None,
        xticks_top: bool                   = False,
        show_title: bool                   = False,
        title: str                         = 'FEP Hydration free energies',
        # ── Y-axis limits ───────────────────────────────────────────────────
        ymin: Optional[float]              = None,
        ymax: Optional[float]              = None,
        # ── Font sizes ──────────────────────────────────────────────────────
        label_fontsize: int                = 18,
        label_fontweight: str              = 'bold',
        tick_fontsize: int                 = 16,
        title_fontsize: int                = 16,
        title_fontweight: str              = 'bold',
        section_label_fontsize: int        = 18,
        section_label_fontweight: str      = 'bold',
        # ── Grid ────────────────────────────────────────────────────────────
        show_grid: bool                    = True,
        grid_alpha: float                  = 0.3,
        grid_linestyle: str                = '--',
        # ── Minor ticks ─────────────────────────────────────────────────────
        show_minor_ticks: bool             = False,
        # ── Spines ──────────────────────────────────────────────────────────
        hide_top_right_spines: bool        = False,
        # ── Legend ──────────────────────────────────────────────────────────
        show_legend: bool                  = True,
        legend_title: str                  = 'Species',
        legend_loc: str                    = 'upper right',
        legend_bbox: Optional[Tuple[float, float]] = None,
        legend_fontsize: int               = 16,
        legend_fontweight: str             = 'normal',
        legend_title_fontsize: vmd run  vmd runvmint         = 16,
        legend_title_fontweight: str       = 'bold',
        legend_ncol: int                   = 1,
        legend_framealpha: float           = 0.9,
        legend_edgecolor: str              = 'black',
        legend_handletextpad: float        = 0.5,
        # ── Value labels above bars ──────────────────────────────────────────
        show_value_labels: bool            = False,
        value_label_fmt: str               = '.1f',
        value_label_fontsize: int          = 12,
        value_label_fontweight: str        = 'normal',
        value_label_offset: float          = 2.0,   # data units away from bar top
        # ── Save ────────────────────────────────────────────────────────────
        save_fig: bool                     = True,
        save_path: str                     = 'FEP_solvation_barchart.png',
        dpi: int                           = 300,
        bbox_inches: str                   = 'tight',
        transparent_bg: bool               = False,
    ) -> plt.Figure:
        """
        Plot a grouped bar chart of solvation free energies.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Tidy DataFrame with columns ``['condition', 'species', 'E', 'err']``.
            When *None* the built-in CIP-/CIP+/--/CIP+ dataset is used.
        conditions : list of str, optional
            Ordered list of condition group names (x-axis groups).
            Inferred from *df* when not supplied.
        species_list : list of str, optional
            Ordered list of species names within each group (determines bar order
            and color/hatch assignment).  Inferred from *df* when not supplied.
        bar_width : float
            Width of individual bars in data-unit width.
        group_spacing : float
            Center-to-center distance between condition groups.
        colors : list of str, optional
            One color per species.  Falls back to the internal default palette.
        hatches : list of str, optional
            One hatch pattern per species (e.g. ``['', '///', 'xxx']``).
            Falls back to a built-in default list.
        save_fig : bool
            Whether to save the figure to *save_path*.
        save_path : str
            Output file path (absolute or relative).  Relative paths are written
            to the current working directory.

        Returns
        -------
        matplotlib.figure.Figure
        """

        # ── Ensure font settings are applied ─────────────────────────────────
        matplotlib.rcParams['font.family'] = self.font_family
        matplotlib.rcParams['mathtext.fontset'] = self.mathtext_fontset

        # ── Data ─────────────────────────────────────────────────────────────
        if df is None:
            if self.fep_path is None:
                raise ValueError(
                    'No DataFrame supplied and fep_path is None. '
                    'Either pass df= to this method or set fep_path in the constructor.'
                )
            df = self.load_data()

        if conditions is None:
            # Preserve order of first occurrence
            seen: Dict[str, None] = {}
            for v in df['condition']:
                seen[v] = None
            conditions = list(seen.keys())

        if species_list is None:
            seen_sp: Dict[str, None] = {}
            for v in df['species']:
                seen_sp[v] = None
            species_list = list(seen_sp.keys())

        n_species = len(species_list)
        n_conditions = len(conditions)

        # ── Resolve groups / items based on group_by ─────────────────────────
        if group_by == 'species':
            groups = species_list          # x-axis tick labels
            items  = conditions            # bars within each group
            def _lookup(grp: str, itm: str):
                mask = (df['species'] == grp) & (df['condition'] == itm)
                row  = df.loc[mask]
                return (row['E'].values[0]  if len(row) else 0.0,
                        row['err'].values[0] if len(row) else 0.0)
        else:  # 'condition'
            groups = conditions
            items  = species_list
            def _lookup(grp: str, itm: str):
                mask = (df['condition'] == grp) & (df['species'] == itm)
                row  = df.loc[mask]
                return (row['E'].values[0]  if len(row) else 0.0,
                        row['err'].values[0] if len(row) else 0.0)

        n_groups = len(groups)
        n_items  = len(items)

        # ── Colors & hatches ─────────────────────────────────────────────────
        # group_by='condition': colors per species (items), hatches per species
        # group_by='species':   colors per species (groups), hatches per condition (items)
        if group_by == 'species':
            # colors: one per species group; hatches: one per condition item
            if colors is None:
                colors = _DEFAULT_COLORS[:n_groups]
            if hatches is None:
                hatches = _DEFAULT_HATCHES[:n_items]
        else:
            # colors: one per species item; hatches: one per species item
            if colors is None:
                colors = _DEFAULT_COLORS[:n_items]
            if hatches is None:
                hatches = _DEFAULT_HATCHES[:n_items]

        # ── X-positions ──────────────────────────────────────────────────────
        offsets = (
            np.linspace(-(n_items - 1) / 2, (n_items - 1) / 2, n_items)
            * bar_width
        )
        group_centers = np.arange(n_groups) * group_spacing

        # ── Figure ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize)

        for grp_idx, grp in enumerate(groups):
            for itm_idx, itm in enumerate(items):
                v, e = _lookup(grp, itm)
                x    = group_centers[grp_idx] + offsets[itm_idx]

                if group_by == 'species':
                    bar_color = colors[grp_idx]   # species color
                    bar_hatch = hatches[itm_idx]  # condition hatch
                else:
                    bar_color = colors[itm_idx]   # species color
                    bar_hatch = hatches[itm_idx]  # species hatch

                ax.bar(
                    x, v,
                    width=bar_width,
                    color=bar_color,
                    alpha=bar_alpha,
                    hatch=bar_hatch,
                    edgecolor=edgecolor,
                    linewidth=edgewidth,
                    zorder=2,
                )

                if show_error_bars:
                    ax.errorbar(
                        x, v, yerr=e,
                        fmt='none',
                        capsize=error_capsize,
                        linewidth=error_linewidth,
                        ecolor=error_color,
                        capthick=error_linewidth,
                        zorder=3,
                    )

                if show_value_labels:
                    label_y = v + (value_label_offset if v >= 0 else -value_label_offset)
                    va = 'bottom' if v >= 0 else 'top'
                    ax.annotate(
                        format(v, value_label_fmt),
                        xy=(x, label_y),
                        ha='center', va=va,
                        fontsize=value_label_fontsize,
                        fontweight=value_label_fontweight,
                        zorder=4,
                    )

        # ── X-axis ───────────────────────────────────────────────────────────
        ax.set_xticks(group_centers)
        ax.set_xticklabels(
            groups,
            fontsize=section_label_fontsize,
            fontweight=section_label_fontweight,
        )
        if xticks_top:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')

        # ── Y-axis ───────────────────────────────────────────────────────────
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight=label_fontweight)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        ax.tick_params(axis='x', labelsize=section_label_fontsize)

        if ymin is not None or ymax is not None:
            ax.set_ylim(
                bottom=ymin if ymin is not None else ax.get_ylim()[0],
                top=ymax   if ymax is not None else ax.get_ylim()[1],
            )

        if show_minor_ticks:
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=label_fontsize,
                          fontweight=label_fontweight)

        # ── Title ────────────────────────────────────────────────────────────
        if show_title:
            ax.set_title(title, fontsize=title_fontsize,
                         fontweight=title_fontweight)

        # ── Grid ─────────────────────────────────────────────────────────────
        if show_grid:
            ax.grid(True, axis='y', alpha=grid_alpha,
                    linestyle=grid_linestyle, zorder=0)
        ax.set_axisbelow(True)

        # ── Spines ───────────────────────────────────────────────────────────
        if hide_top_right_spines:
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        # ── Legend ───────────────────────────────────────────────────────────
        if show_legend:
            if group_by == 'species':
                # Legend shows conditions (items) differentiated by hatch;
                # species color shown via a striped neutral patch per condition.
                handles = [
                    Patch(
                        facecolor='white',
                        hatch=hatches[i],
                        edgecolor=edgecolor,
                        alpha=1.0,
                        label=items[i],
                    )
                    for i in range(n_items)
                ]
            else:
                # Legend shows species (items) with their colors and hatches.
                handles = [
                    Patch(
                        facecolor=colors[i],
                        hatch=hatches[i],
                        edgecolor=edgecolor,
                        alpha=bar_alpha,
                        label=items[i],
                    )
                    for i in range(n_items)
                ]
            legend_kw: dict = dict(
                handles=handles,
                title=legend_title,
                fontsize=legend_fontsize,
                title_fontsize=legend_title_fontsize,
                loc=legend_loc,
                framealpha=legend_framealpha,
                edgecolor=legend_edgecolor,
                ncol=legend_ncol,
                handletextpad=legend_handletextpad,
            )
            if legend_bbox is not None:
                legend_kw['bbox_to_anchor'] = legend_bbox
            leg = ax.legend(**legend_kw)
            leg.get_title().set_fontweight(legend_title_fontweight)
            for _t in leg.get_texts():
                _t.set_fontweight(legend_fontweight)

        plt.tight_layout()

        # ── Save ─────────────────────────────────────────────────────────────
        if save_fig:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches,
                        transparent=transparent_bg)
            print(f"✓ Saved: {save_path}")

        return fig
