"""
Plotly-based plotting functions for EIS data visualization
"""

import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Tuple

# Color palette for multiple files
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


def subscript_composition(composition: str) -> str:
    """
    Convert composition formula to HTML with subscript numbers.
    Example: Li7La3Zr2O12 -> Li<sub>7</sub>La<sub>3</sub>Zr<sub>2</sub>O<sub>12</sub>

    Args:
        composition: Chemical composition string like 'Li7La3Zr2O12'

    Returns:
        HTML formatted string with subscript numbers
    """
    if not composition:
        return composition

    # Split into elements and numbers
    elements = re.split(r'[^a-zA-Z]+', composition)
    numbers = re.split(r'[^0-9]+', composition)

    # Handle trailing empty string from split
    elements = [e for e in elements if e]
    numbers = [n for n in numbers if n]

    if not elements:
        return composition

    # Build HTML string
    result = ''
    for i, elem in enumerate(elements):
        result += elem
        if i < len(numbers):
            result += f'<sub>{numbers[i]}</sub>'

    return result


def get_unit_divisor(unit: str) -> float:
    """Get divisor for unit conversion"""
    unit_map = {
        'Ω': 1,
        'kΩ': 1e3,
        'MΩ': 1e6,
        'GΩ': 1e9,
    }
    return unit_map.get(unit, 1)


def get_axis_range(data_min: float, data_max: float) -> Tuple[float, float]:
    """
    Calculate axis range with 5% margin.
    """
    range_val = data_max - data_min
    if range_val == 0:
        range_val = abs(data_max) * 0.1 if data_max != 0 else 1
    margin = range_val * 0.05
    return data_min - margin, data_max + margin


def common_layout(settings: dict = None) -> dict:
    """Common layout settings for all plots"""
    if settings is None:
        settings = {}
    return {
        'font': {'family': 'Arial', 'color': 'black'},
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'margin': {'l': 60, 'r': 10, 't': 25, 'b': 50},
    }


def get_legend_name(file_name: str, data: dict, settings: dict, is_fit: bool = False) -> str:
    """
    Get legend name based on settings.

    For data: Filename, Temperature (int), or Manual
    For fit: Filename, Manual, or Hide (returns None)

    Returns None if legend should be hidden.
    """
    if is_fit:
        mode = settings.get('fit_legend_mode', 'Filename')
        if mode == 'Hide':
            return None
        elif mode == 'Manual':
            return settings.get('fit_legend_manual', 'fitted')
        else:  # Filename
            return f'{file_name} (fit)'
    else:
        mode = settings.get('plot_legend_mode', 'Filename')
        if mode == 'Manual':
            return settings.get('plot_legend_manual', 'measured')
        elif mode == 'Temperature':
            temp = data.get('temperature')
            if temp is not None:
                return f'{int(temp)} K'
            else:
                return file_name
        else:  # Filename
            return file_name


def common_axis_settings(settings: dict = None) -> dict:
    """Common axis settings with ticks inside"""
    if settings is None:
        settings = {}
    tick_size = settings.get('tick_font_size', 10)
    label_size = settings.get('axis_label_font_size', 12)

    return {
        'showgrid': False,
        'showline': True,
        'linewidth': 1,
        'linecolor': 'black',
        'tickcolor': 'black',
        'tickfont': {'family': 'Arial', 'color': 'black', 'size': tick_size},
        'title_font': {'family': 'Arial', 'color': 'black', 'size': label_size},
        'mirror': True,
        'ticks': 'inside',  # Ticks inside
        'ticklen': 5,
        'title_standoff': 8,
        'zeroline': True,
        'zerolinecolor': 'gray',
        'zerolinewidth': 1,
    }


def find_decade_frequencies(freq: np.ndarray) -> List[int]:
    """Find indices of frequencies closest to 10^n Hz"""
    decades = []
    for exp in range(-2, 8):  # 0.01 Hz to 10 MHz
        target = 10 ** exp
        if freq.min() <= target <= freq.max():
            idx = np.argmin(np.abs(freq - target))
            if idx not in decades:
                decades.append(idx)
    return decades


def create_nyquist_plot(
    data_dict: Dict,
    selected_files: Optional[List[str]] = None,
    show_fit: bool = False,
    show_legend: bool = True,
    highlight_freq: bool = False,
    settings: dict = None,
    freq_range: Optional[Tuple[int, int]] = None,
    deleted_points: Optional[List[int]] = None
) -> go.Figure:
    """
    Create interactive Nyquist plot using Plotly
    """
    if settings is None:
        settings = {}
    if deleted_points is None:
        deleted_points = []

    fig = go.Figure()

    if selected_files is None:
        selected_files = list(data_dict.keys())

    # Get unit settings
    z_unit = settings.get('z_unit', 'Ω')
    divisor = get_unit_divisor(z_unit)

    # Marker settings
    marker_size = settings.get('marker_size', 6)
    marker_color = settings.get('marker_color', '#1f77b4')
    marker_symbol = settings.get('marker_symbol', 'circle')
    marker_alpha = settings.get('marker_alpha', 1.0)
    marker_line_color = settings.get('marker_line_color', '#1f77b4')
    marker_line_width = settings.get('marker_line_width', 0)

    # Fit line settings
    fit_line_color = settings.get('fit_line_color', '#ff7f0e')
    fit_line_width = settings.get('fit_line_width', 2)

    # Zero line settings
    show_zeroline = settings.get('show_zeroline', True)

    # Collect all data for range calculation
    all_real = []
    all_imag = []

    for idx, file_name in enumerate(selected_files):
        if file_name not in data_dict:
            continue

        data = data_dict[file_name]
        Z_full = data['Z']
        freq_full = data['freq']
        n_points = len(freq_full)

        # Create index array
        indices = np.arange(n_points)

        # Apply freq_range filter
        if freq_range:
            start_idx, end_idx = freq_range
            start_idx = max(0, min(start_idx, n_points - 1))
            end_idx = max(0, min(end_idx, n_points - 1))
            mask = (indices >= start_idx) & (indices <= end_idx)
        else:
            mask = np.ones(n_points, dtype=bool)

        # Apply deleted_points filter
        for dp in deleted_points:
            if 0 <= dp < n_points:
                mask[dp] = False

        # Filter data
        Z = Z_full[mask]
        freq = freq_full[mask]
        filtered_indices = indices[mask]

        color = marker_color if len(selected_files) == 1 else COLORS[idx % len(COLORS)]

        z_real = np.real(Z) / divisor
        z_imag = -np.imag(Z) / divisor

        all_real.extend(z_real)
        all_imag.extend(z_imag)

        # Create hover text with f and index
        hover_texts = [f"idx={filtered_indices[i]}<br><i>f</i>={freq[i]:.2e} Hz<br><i>Z'</i>={z_real[i]:.2f} {z_unit}<br>–<i>Z''</i>={z_imag[i]:.2f} {z_unit}"
                       for i in range(len(z_real))]

        # Get legend name based on settings
        data_legend_name = get_legend_name(file_name, data, settings, is_fit=False)

        # Add measured data
        fig.add_trace(go.Scatter(
            x=z_real,
            y=z_imag,
            mode='markers',
            name=data_legend_name,
            marker=dict(
                size=marker_size,
                color=color,
                symbol=marker_symbol,
                opacity=marker_alpha,
                line=dict(color=marker_line_color, width=marker_line_width)
            ),
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
            showlegend=show_legend
        ))

        # Highlight decade frequencies (same size as data markers)
        if highlight_freq:
            decade_indices = find_decade_frequencies(freq)
            if decade_indices:
                fig.add_trace(go.Scatter(
                    x=z_real[decade_indices],
                    y=z_imag[decade_indices],
                    mode='markers',
                    name=f'{file_name} (decades)',
                    marker=dict(
                        size=marker_size,
                        color='red',
                        symbol='circle'
                    ),
                    hovertemplate="<i>f</i> = %{text} Hz<extra></extra>",
                    text=[f"{freq[i]:.0e}" for i in decade_indices],
                    showlegend=False
                ))

        # Add fitted data if available (limited to freq_range)
        if show_fit and data.get('Z_fit') is not None:
            Z_fit_full = data['Z_fit']
            # Limit to fitting range
            if freq_range:
                start_idx, end_idx = freq_range
                start_idx = max(0, min(start_idx, n_points - 1))
                end_idx = max(0, min(end_idx, n_points - 1))
                Z_fit = Z_fit_full[start_idx:end_idx + 1]
            else:
                Z_fit = Z_fit_full

            # Get fit legend name based on settings (None means hide)
            fit_legend_name = get_legend_name(file_name, data, settings, is_fit=True)
            fit_show_legend = show_legend and fit_legend_name is not None

            fig.add_trace(go.Scatter(
                x=np.real(Z_fit) / divisor,
                y=-np.imag(Z_fit) / divisor,
                mode='lines',
                name=fit_legend_name if fit_legend_name else '',
                line=dict(width=fit_line_width, color=fit_line_color),
                showlegend=fit_show_legend
            ))

    # Calculate range with 5% margin
    if all_real and all_imag:
        all_vals = all_real + all_imag
        data_min = min(all_vals)
        data_max = max(all_vals)
        axis_min, axis_max = get_axis_range(data_min, data_max)
    else:
        axis_min, axis_max = 0, 1

    # Update layout
    axis_settings = common_axis_settings(settings)
    # Override zeroline based on settings
    axis_settings['zeroline'] = show_zeroline
    label_size = settings.get('axis_label_font_size', 12)

    # Get legend font size from settings
    legend_font_size = settings.get('legend_font_size', 10)

    fig.update_layout(
        **common_layout(settings),
        height=350,
        uirevision='nyquist',  # Preserve zoom state on rerun
        xaxis=dict(
            **axis_settings,
            title=dict(text=f"<i>Z'</i> / {z_unit}", font=dict(family='Arial', color='black', size=label_size)),
            range=[axis_min, axis_max],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            **axis_settings,
            title=dict(text=f"–<i>Z''</i> / {z_unit}", font=dict(family='Arial', color='black', size=label_size)),
            range=[axis_min, axis_max],
        ),
        showlegend=show_legend,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig


def create_bode_plot(
    data_dict: Dict,
    selected_files: Optional[List[str]] = None,
    show_fit: bool = False,
    show_legend: bool = True,
    freq_range: Optional[Tuple[int, int]] = None,
    settings: dict = None,
    deleted_points: Optional[List[int]] = None
) -> go.Figure:
    """
    Create interactive Bode plot using Plotly with shared x-axis
    """
    if settings is None:
        settings = {}
    if deleted_points is None:
        deleted_points = []

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5]
    )

    if selected_files is None:
        selected_files = list(data_dict.keys())

    # Get unit settings
    z_unit = settings.get('z_unit', 'Ω')
    divisor = get_unit_divisor(z_unit)

    # Marker settings
    marker_size = settings.get('marker_size', 6)
    marker_color = settings.get('marker_color', '#1f77b4')
    marker_symbol = settings.get('marker_symbol', 'circle')
    marker_alpha = settings.get('marker_alpha', 1.0)
    marker_line_color = settings.get('marker_line_color', '#1f77b4')
    marker_line_width = settings.get('marker_line_width', 0)

    # Fit line settings
    fit_line_color = settings.get('fit_line_color', '#ff7f0e')
    fit_line_width = settings.get('fit_line_width', 2)

    for idx, file_name in enumerate(selected_files):
        if file_name not in data_dict:
            continue

        data = data_dict[file_name]
        freq_full = data['freq']
        Z_full = data['Z']
        n_points = len(freq_full)

        # Create index array
        indices = np.arange(n_points)

        # Apply freq_range filter
        if freq_range:
            start_idx, end_idx = freq_range
            start_idx = max(0, min(start_idx, n_points - 1))
            end_idx = max(0, min(end_idx, n_points - 1))
            mask = (indices >= start_idx) & (indices <= end_idx)
        else:
            mask = np.ones(n_points, dtype=bool)

        # Apply deleted_points filter
        for dp in deleted_points:
            if 0 <= dp < n_points:
                mask[dp] = False

        # Filter data
        freq = freq_full[mask]
        Z = Z_full[mask]
        filtered_indices = indices[mask]

        color = marker_color if len(selected_files) == 1 else COLORS[idx % len(COLORS)]

        log_freq = np.log10(freq)
        log_magnitude = np.log10(np.abs(Z) / divisor)
        phase = -np.angle(Z, deg=True)

        # Create hover text with f and index
        hover_texts_mag = [f"idx={filtered_indices[i]}<br><i>f</i>={freq[i]:.2e} Hz<br>log(|<i>Z</i>|/{z_unit})={log_magnitude[i]:.2f}"
                          for i in range(len(log_freq))]
        hover_texts_phase = [f"idx={filtered_indices[i]}<br><i>f</i>={freq[i]:.2e} Hz<br><i>θ</i>={phase[i]:.1f}°"
                            for i in range(len(log_freq))]

        # Get legend name based on settings
        data_legend_name = get_legend_name(file_name, data, settings, is_fit=False)

        # Magnitude plot (log |Z|)
        fig.add_trace(go.Scatter(
            x=log_freq,
            y=log_magnitude,
            mode='markers',
            name=data_legend_name,
            marker=dict(
                size=marker_size,
                color=color,
                symbol=marker_symbol,
                opacity=marker_alpha,
                line=dict(color=marker_line_color, width=marker_line_width)
            ),
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts_mag,
            showlegend=show_legend
        ), row=1, col=1)

        # Phase plot
        fig.add_trace(go.Scatter(
            x=log_freq,
            y=phase,
            mode='markers',
            name=data_legend_name,
            marker=dict(
                size=marker_size,
                color=color,
                symbol=marker_symbol,
                opacity=marker_alpha,
                line=dict(color=marker_line_color, width=marker_line_width)
            ),
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts_phase,
            showlegend=False
        ), row=2, col=1)

        # Add fitted data if available (only within freq_range)
        if show_fit and data.get('Z_fit') is not None:
            Z_fit = data['Z_fit']

            # Limit to fitting range
            if freq_range:
                start_idx, end_idx = freq_range
                freq_fit = freq_full[start_idx:end_idx + 1]
                Z_fit_range = Z_fit[start_idx:end_idx + 1]
            else:
                freq_fit = freq_full
                Z_fit_range = Z_fit

            log_freq_fit = np.log10(freq_fit)

            # Get fit legend name based on settings (None means hide)
            fit_legend_name = get_legend_name(file_name, data, settings, is_fit=True)
            fit_show_legend = show_legend and fit_legend_name is not None

            fig.add_trace(go.Scatter(
                x=log_freq_fit,
                y=np.log10(np.abs(Z_fit_range) / divisor),
                mode='lines',
                name=fit_legend_name if fit_legend_name else '',
                line=dict(width=fit_line_width, color=fit_line_color),
                showlegend=fit_show_legend
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=log_freq_fit,
                y=-np.angle(Z_fit_range, deg=True),
                mode='lines',
                name=fit_legend_name if fit_legend_name else '',
                line=dict(width=fit_line_width, color=fit_line_color),
                showlegend=False
            ), row=2, col=1)

    # Update layout
    axis_settings = common_axis_settings(settings)
    # Remove zeroline for Bode plot (not needed for log scale)
    bode_axis_settings = {k: v for k, v in axis_settings.items() if k not in ['zeroline', 'zerolinecolor', 'zerolinewidth']}
    label_size = settings.get('axis_label_font_size', 12)
    label_font = dict(family='Arial', color='black', size=label_size)

    # Get legend font size from settings
    legend_font_size = settings.get('legend_font_size', 10)

    fig.update_layout(
        **common_layout(settings),
        height=350,
        uirevision='bode',  # Preserve zoom state on rerun
        showlegend=show_legend,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
        )
    )

    # X-axis (shared, tick labels only on bottom)
    fig.update_xaxes(
        **bode_axis_settings,
        showticklabels=False,
        title='',
        row=1, col=1
    )
    fig.update_xaxes(
        **bode_axis_settings,
        title=dict(text="log(<i>f</i> / Hz)", font=label_font),
        showticklabels=True,
        row=2, col=1
    )

    # Y-axes
    fig.update_yaxes(
        **bode_axis_settings,
        title=dict(text=f"log(|<i>Z</i>| / {z_unit})", font=label_font),
        row=1, col=1
    )
    fig.update_yaxes(
        **bode_axis_settings,
        title=dict(text="<i>θ</i> / °", font=label_font),
        row=2, col=1
    )

    return fig


def arrhenius_fit(x_1000_T: np.ndarray, y_log_sigma_T: np.ndarray):
    """
    Perform linear fit on Arrhenius data.
    log(σT) = -Ea/(R * ln10) * (1000/T) + log(A)

    Returns: slope, slope_err, intercept, Ea (kJ/mol), Ea_err (kJ/mol), Ea (eV), Ea_err (eV)
    """
    if len(x_1000_T) < 2:
        return None, None, None, None, None, None, None

    # Linear fit: y = slope * x + intercept
    # Use np.polyfit with cov=True to get covariance matrix
    coeffs, cov = np.polyfit(x_1000_T, y_log_sigma_T, 1, cov=True)
    slope = coeffs[0]
    intercept = coeffs[1]
    slope_err = np.sqrt(cov[0, 0])

    # Calculate Ea
    # slope = -Ea / (R * ln10 * 1000)  where R = 8.314 J/(mol·K)
    # Ea = -slope * R * ln10 * 1000 (in J/mol)
    R = 8.314  # J/(mol·K)
    conversion_factor = R * np.log(10) * 1000  # J/mol per slope unit

    Ea_J_mol = -slope * conversion_factor  # J/mol
    Ea_J_mol_err = slope_err * conversion_factor  # Error propagation

    Ea_kJ_mol = Ea_J_mol / 1000  # kJ/mol
    Ea_kJ_mol_err = Ea_J_mol_err / 1000

    Ea_eV = Ea_J_mol / 96485  # eV (1 eV = 96485 J/mol)
    Ea_eV_err = Ea_J_mol_err / 96485

    return slope, slope_err, intercept, Ea_kJ_mol, Ea_kJ_mol_err, Ea_eV, Ea_eV_err


def generate_temp_ticks(x_min, x_max):
    """
    Auto-generate temperature ticks based on 1000/T range.
    Returns tickvals and ticktext for the temperature axis.
    Aims for 4-6 ticks for readability.
    """
    # Convert x range to temperature range
    t_max = 1000 / x_min if x_min > 0 else 2000
    t_min = 1000 / x_max if x_max > 0 else 100

    # Determine appropriate tick spacing to get 4-6 ticks
    t_range = t_max - t_min
    target_n_ticks = 5

    # Find step size that gives approximately target_n_ticks
    raw_step = t_range / target_n_ticks

    # Round to nice values: 10, 25, 50, 100, 200, 250, 500, ...
    nice_steps = [10, 20, 25, 50, 100, 150, 200, 250, 500, 1000]
    step = min(nice_steps, key=lambda x: abs(x - raw_step))

    # Ensure step is not too small (causing too many ticks)
    while t_range / step > 8:
        idx = nice_steps.index(step) if step in nice_steps else 0
        if idx < len(nice_steps) - 1:
            step = nice_steps[idx + 1]
        else:
            step = step * 2

    # Generate ticks
    start_t = int(np.ceil(t_min / step) * step)
    end_t = int(np.floor(t_max / step) * step)

    ticks = list(range(start_t, end_t + 1, step))

    tickvals = [1000 / t for t in ticks if t > 0]
    ticktext = [str(t) for t in ticks if t > 0]

    return tickvals, ticktext


def create_arrhenius_plots(
    multipoint_data: List[Dict],
    conductivity_types: List[str] = None,
    type_labels: Dict[str, str] = None,
    show_legend: bool = True,
    plot_settings: Dict = None,
    fit_range: Tuple[int, int] = None,
    show_fit: bool = False,
    fit_targets: List[str] = None,
    show_ranges: Dict[str, Tuple[int, int]] = None,
    show_fit_legend: bool = True
) -> Tuple[go.Figure, go.Figure, Dict]:
    """
    Create two Arrhenius plots: log(σT) vs 1000/T and log(σ) vs 1000/T.

    Parameters:
    - multipoint_data: List of dicts with temperature and conductivity data
    - conductivity_types: List of conductivity types to plot (e.g., ['total', 'R1', 'R2'])
    - type_labels: Dict mapping type names to display labels (e.g., {'R1': 'bulk', 'R2': 'gb'})
    - show_legend: Whether to show legend
    - plot_settings: Plot style settings
    - fit_range: Tuple of (start_idx, end_idx) for fitting
    - show_fit: Whether to show fit lines
    - fit_targets: List of conductivity types to fit (subset of conductivity_types)
    - show_ranges: Dict of {cond_type: (start_idx, end_idx)} for per-type display ranges
    - show_fit_legend: Whether to show fit line legend

    Returns: (fig_sigma_T, fig_sigma, fit_results)
    fit_results is a dict: {type_name: {Ea_kJ_mol, Ea_eV, ...}}
    """
    from plotly.subplots import make_subplots

    if conductivity_types is None:
        conductivity_types = ['total']
    if type_labels is None:
        type_labels = {}
    if fit_targets is None:
        fit_targets = conductivity_types
    if show_ranges is None:
        show_ranges = {}

    # Get settings with defaults
    if plot_settings is None:
        plot_settings = {}

    tick_font_size = plot_settings.get('tick_font_size', 18)
    label_font_size = plot_settings.get('axis_label_font_size', 18)
    marker_size = plot_settings.get('arr_marker_size', 10)
    marker_edge_color = plot_settings.get('arr_marker_edge_color', '#000000')
    marker_edge_width = plot_settings.get('arr_marker_edge_width', 1)
    line_width = plot_settings.get('arr_line_width', 2)
    show_line = plot_settings.get('arr_show_line', False)
    legend_font_size = plot_settings.get('arr_legend_font_size', 12)

    tick_font = dict(size=tick_font_size, family='Arial', color='black')
    label_font = dict(size=label_font_size, family='Arial', color='black')

    axis_settings = common_axis_settings(plot_settings)
    axis_settings['zeroline'] = False

    # Colors and markers for different conductivity types
    # Map by label: total=black, bulk=red, gb=blue, others cycle through remaining colors
    color_map = {
        'total': '#000000',  # black
        'bulk': '#E63946',   # red
        'gb': '#2196F3',     # blue
    }
    marker_map = {
        'total': 'circle',
        'bulk': 'square',
        'gb': 'triangle-up',
    }
    # Fallback colors for other types
    fallback_colors = ['#2A9D8F', '#E9C46A', '#F4A261', '#9B5DE5']
    fallback_markers = ['diamond', 'cross', 'x', 'star']

    fit_results = {}

    # Create figures
    fig_sigma_T = go.Figure()
    fig_sigma = go.Figure()

    # Check if cycle mode is active (data has 'cycle' key)
    has_cycles = any('cycle' in data for data in multipoint_data)

    # Cycle styles: different fill for different cycles
    cycle_styles = {
        '1st': {'fill': True, 'dash': None},
        '2nd': {'fill': False, 'dash': None},
        '3rd': {'fill': True, 'dash': 'dot'},
        '4th': {'fill': False, 'dash': 'dot'},
        'heating': {'fill': True, 'dash': None},
        'cooling': {'fill': False, 'dash': None},
    }

    # Collect all x values to determine axis range
    all_x = []
    fallback_idx = 0

    for idx, cond_type in enumerate(conductivity_types):
        label = type_labels.get(cond_type, cond_type)

        # Get color and marker based on label (total, bulk, gb)
        if label in color_map:
            color = color_map[label]
            marker_symbol = marker_map[label]
        else:
            color = fallback_colors[fallback_idx % len(fallback_colors)]
            marker_symbol = fallback_markers[fallback_idx % len(fallback_markers)]
            fallback_idx += 1

        # Get show_range for this conductivity type
        show_range = show_ranges.get(cond_type)

        # Filter data for this cond_type based on show_range
        if show_range is not None:
            filtered_data = multipoint_data[show_range[0]:show_range[1] + 1]
        else:
            filtered_data = multipoint_data

        if has_cycles:
            # Group data by cycle
            cycle_data = {}
            for data in filtered_data:
                if 'temperature' not in data:
                    continue
                T = data['temperature']
                if T <= 0:
                    continue

                # Get sigma value
                if cond_type == 'total':
                    sigma = data.get('total_sigma')
                else:
                    r_sigmas = data.get('r_sigmas', {})
                    sigma = r_sigmas.get(cond_type)

                if sigma is None or sigma <= 0:
                    continue

                cycle = data.get('cycle', '1st')
                if cycle not in cycle_data:
                    cycle_data[cycle] = {'x': [], 'y_T': [], 'y_s': [], 'hover': []}

                x_val = 1000 / T
                cycle_data[cycle]['x'].append(x_val)
                cycle_data[cycle]['y_T'].append(np.log10(sigma * T))
                cycle_data[cycle]['y_s'].append(np.log10(sigma))
                cycle_data[cycle]['hover'].append({
                    'filename': data.get('filename', ''),
                    'T': T,
                    'sigma': sigma,
                    'sigma_T': sigma * T,
                    'cycle': cycle
                })

            # Plot each cycle separately
            for cycle_name, cdata in cycle_data.items():
                if len(cdata['x']) == 0:
                    continue

                x_arr = np.array(cdata['x'])
                y_T_arr = np.array(cdata['y_T'])
                y_s_arr = np.array(cdata['y_s'])
                all_x.extend(x_arr)

                # Get cycle style
                style = cycle_styles.get(cycle_name, {'fill': True, 'dash': None})

                # Create hover text
                hover_T = []
                hover_s = []
                for i, hd in enumerate(cdata['hover']):
                    base_text = f"<b>{hd['filename']}</b><br>" if hd['filename'] else ""
                    base_text += f"<b>{label}</b> ({cycle_name})<br>"
                    base_text += f"<i>T</i> = {hd['T']:.1f} K<br>"
                    base_text += f"<i>σ</i> = {hd['sigma']:.2e} S/cm<br>"
                    hover_T.append(base_text + f"log(<i>σT</i>) = {y_T_arr[i]:.4f}")
                    hover_s.append(base_text + f"log(<i>σ</i>) = {y_s_arr[i]:.4f}")

                mode = 'markers+lines' if show_line else 'markers'

                # Marker style based on cycle (fill or not)
                if style['fill']:
                    marker_color = color
                else:
                    marker_color = 'rgba(255,255,255,0)'  # transparent fill

                fig_sigma_T.add_trace(go.Scatter(
                    x=x_arr,
                    y=y_T_arr,
                    mode=mode,
                    name=f'σ<sub>{label}</sub> ({cycle_name})',
                    marker=dict(
                        size=marker_size,
                        color=marker_color,
                        symbol=marker_symbol,
                        line=dict(color=color, width=marker_edge_width + 1)
                    ),
                    line=dict(width=line_width, color=color, dash=style['dash']) if show_line else None,
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_T,
                    showlegend=show_legend
                ))

                fig_sigma.add_trace(go.Scatter(
                    x=x_arr,
                    y=y_s_arr,
                    mode=mode,
                    name=f'σ<sub>{label}</sub> ({cycle_name})',
                    marker=dict(
                        size=marker_size,
                        color=marker_color,
                        symbol=marker_symbol,
                        line=dict(color=color, width=marker_edge_width + 1)
                    ),
                    line=dict(width=line_width, color=color, dash=style['dash']) if show_line else None,
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_s,
                    showlegend=show_legend
                ))

        else:
            # Original non-cycle behavior
            # Extract data for this conductivity type
            x_data = []
            y_sigma_T = []
            y_sigma = []
            hover_data = []

            for data in filtered_data:
                if 'temperature' not in data:
                    continue
                T = data['temperature']
                if T <= 0:
                    continue

                # Get sigma value
                if cond_type == 'total':
                    sigma = data.get('total_sigma')
                else:
                    r_sigmas = data.get('r_sigmas', {})
                    sigma = r_sigmas.get(cond_type)

                if sigma is None or sigma <= 0:
                    continue

                x_val = 1000 / T
                x_data.append(x_val)
                y_sigma_T.append(np.log10(sigma * T))
                y_sigma.append(np.log10(sigma))
                hover_data.append({
                    'filename': data.get('filename', ''),
                    'T': T,
                    'sigma': sigma,
                    'sigma_T': sigma * T
                })

            if len(x_data) == 0:
                continue

            x_data = np.array(x_data)
            y_sigma_T = np.array(y_sigma_T)
            y_sigma = np.array(y_sigma)
            all_x.extend(x_data)

            # Create hover text
            hover_texts_T = []
            hover_texts_s = []
            for i, hd in enumerate(hover_data):
                base_text = f"<b>{hd['filename']}</b><br>" if hd['filename'] else ""
                base_text += f"<b>{label}</b><br>"
                base_text += f"<i>T</i> = {hd['T']:.1f} K<br>"
                base_text += f"<i>σ</i> = {hd['sigma']:.2e} S/cm<br>"

                text_T = base_text + f"log(<i>σT</i>) = {y_sigma_T[i]:.4f}"
                text_s = base_text + f"log(<i>σ</i>) = {y_sigma[i]:.4f}"

                hover_texts_T.append(text_T)
                hover_texts_s.append(text_s)

            # Determine mode
            mode = 'markers+lines' if show_line else 'markers'

            # Add traces to both figures
            fig_sigma_T.add_trace(go.Scatter(
                x=x_data,
                y=y_sigma_T,
                mode=mode,
                name=f'σ<sub>{label}</sub>',
                marker=dict(
                    size=marker_size,
                    color=color,
                    symbol=marker_symbol,
                    line=dict(color=marker_edge_color, width=marker_edge_width)
                ),
                line=dict(width=line_width, color=color) if show_line else None,
                hovertemplate="%{text}<extra></extra>",
                text=hover_texts_T,
                showlegend=show_legend
            ))

            fig_sigma.add_trace(go.Scatter(
                x=x_data,
                y=y_sigma,
                mode=mode,
                name=f'σ<sub>{label}</sub>',
                marker=dict(
                    size=marker_size,
                    color=color,
                    symbol=marker_symbol,
                    line=dict(color=marker_edge_color, width=marker_edge_width)
                ),
                line=dict(width=line_width, color=color) if show_line else None,
                hovertemplate="%{text}<extra></extra>",
                text=hover_texts_s,
                showlegend=show_legend
            ))

        # Perform fitting if requested and this type is in fit_targets
        # For fitting, we need all data regardless of cycle mode
        if has_cycles:
            # Collect all data for this conductivity type (regardless of cycle)
            x_data_all = []
            y_sigma_T_all = []
            y_sigma_all = []
            for data in multipoint_data:
                if 'temperature' not in data:
                    continue
                T = data['temperature']
                if T <= 0:
                    continue
                if cond_type == 'total':
                    sigma = data.get('total_sigma')
                else:
                    r_sigmas = data.get('r_sigmas', {})
                    sigma = r_sigmas.get(cond_type)
                if sigma is None or sigma <= 0:
                    continue
                x_data_all.append(1000 / T)
                y_sigma_T_all.append(np.log10(sigma * T))
                y_sigma_all.append(np.log10(sigma))
            x_data_for_fit = np.array(x_data_all)
            y_sigma_T_for_fit = np.array(y_sigma_T_all)
            y_sigma_for_fit = np.array(y_sigma_all)
        else:
            x_data_for_fit = x_data
            y_sigma_T_for_fit = y_sigma_T
            y_sigma_for_fit = y_sigma

        if show_fit and cond_type in fit_targets and len(x_data_for_fit) >= 2:
            # Apply fit range
            if fit_range is not None:
                start_idx, end_idx = fit_range
                start_idx = max(0, min(start_idx, len(x_data_for_fit) - 1))
                end_idx = max(0, min(end_idx, len(x_data_for_fit) - 1))
                x_fit = x_data_for_fit[start_idx:end_idx + 1]
                y_fit_T = y_sigma_T_for_fit[start_idx:end_idx + 1]
                y_fit_s = y_sigma_for_fit[start_idx:end_idx + 1]
            else:
                x_fit = x_data_for_fit
                y_fit_T = y_sigma_T_for_fit
                y_fit_s = y_sigma_for_fit

            if len(x_fit) >= 2:
                # Fit log(σT) - for activation energy
                slope, slope_err, intercept, Ea_kJ, Ea_kJ_err, Ea_eV, Ea_eV_err = arrhenius_fit(x_fit, y_fit_T)

                if slope is not None:
                    fit_results[cond_type] = {
                        'label': label,
                        'Ea_kJ_mol': Ea_kJ,
                        'Ea_kJ_mol_err': Ea_kJ_err,
                        'Ea_eV': Ea_eV,
                        'Ea_eV_err': Ea_eV_err,
                        'slope': slope,
                        'slope_err': slope_err,
                        'intercept': intercept
                    }

                    # Generate fit line for σT plot
                    x_line = np.linspace(min(x_fit), max(x_fit), 100)
                    y_line_T = slope * x_line + intercept

                    fig_sigma_T.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line_T,
                        mode='lines',
                        name=f'Fit {label}',
                        line=dict(width=2, color=color, dash='dash'),
                        showlegend=show_fit_legend
                    ))

                    # Fit log(σ) separately for its own line (same Ea but different intercept)
                    slope_s, _, intercept_s, _, _, _, _ = arrhenius_fit(x_fit, y_fit_s)
                    if slope_s is not None:
                        y_line_s = slope_s * x_line + intercept_s
                        fig_sigma.add_trace(go.Scatter(
                            x=x_line,
                            y=y_line_s,
                            mode='lines',
                            name=f'Fit {label}',
                            line=dict(width=2, color=color, dash='dash'),
                            showlegend=show_fit_legend
                        ))

    # Calculate axis ranges
    if len(all_x) > 0:
        x_min, x_max = min(all_x), max(all_x)
        x_range = x_max - x_min
        x_min_plot = x_min - x_range * 0.05
        x_max_plot = x_max + x_range * 0.05
    else:
        x_min_plot, x_max_plot = 1, 4

    # Generate temperature ticks
    tickvals, ticktext = generate_temp_ticks(x_min_plot, x_max_plot)

    # Add invisible trace for secondary x-axis
    for fig in [fig_sigma_T, fig_sigma]:
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            xaxis='x2',
            showlegend=False,
            hoverinfo='skip'
        ))

    # Common layout settings
    layout = common_layout()
    layout['margin'] = {'l': 60, 'r': 10, 't': 60, 'b': 50}

    # Update layout for σT plot
    fig_sigma_T.update_layout(
        **layout,
        height=400,
        uirevision='arrhenius_sigma_T',
        xaxis=dict(
            **axis_settings,
            title=dict(text="1000 <i>T</i><sup>–1</sup> / K<sup>–1</sup>", font=label_font),
            range=[x_min_plot, x_max_plot]
        ),
        yaxis=dict(**axis_settings, title=dict(text="log(<i>σT</i> / S K cm<sup>–1</sup>)", font=label_font)),
        showlegend=show_legend,
        legend=dict(
            yanchor="top", y=0.98, xanchor="right", x=0.98,
            font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
        ),
        xaxis2=dict(
            title=dict(text="<i>T</i> / K", font=label_font, standoff=5),
            overlaying='x',
            side='top',
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            tickcolor='black',
            tickfont=tick_font,
            ticks='inside',
            ticklen=5,
            mirror=False,
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            range=[x_min_plot, x_max_plot]
        )
    )

    # Update layout for σ plot
    fig_sigma.update_layout(
        **layout,
        height=400,
        uirevision='arrhenius_sigma',
        xaxis=dict(
            **axis_settings,
            title=dict(text="1000 <i>T</i><sup>–1</sup> / K<sup>–1</sup>", font=label_font),
            range=[x_min_plot, x_max_plot]
        ),
        yaxis=dict(**axis_settings, title=dict(text="log(<i>σ</i> / S cm<sup>–1</sup>)", font=label_font)),
        showlegend=show_legend,
        legend=dict(
            yanchor="top", y=0.98, xanchor="right", x=0.98,
            font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
        ),
        xaxis2=dict(
            title=dict(text="<i>T</i> / K", font=label_font, standoff=5),
            overlaying='x',
            side='top',
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            tickcolor='black',
            tickfont=tick_font,
            ticks='inside',
            ticklen=5,
            mirror=False,
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            range=[x_min_plot, x_max_plot]
        )
    )

    return fig_sigma_T, fig_sigma, fit_results


# Keep old function for backward compatibility
def create_arrhenius_plot(
    multipoint_data: List[Dict],
    conductivity_type: str = 'total',
    show_legend: bool = True,
    plot_settings: Dict = None,
    fit_range: Tuple[int, int] = None,
    show_fit: bool = False
) -> Tuple[go.Figure, Dict]:
    """
    Create Arrhenius plot (log(σT) vs 1000/T) with optional fitting.
    Backward compatible wrapper for create_arrhenius_plots.
    """
    fig_sigma_T, _, fit_results = create_arrhenius_plots(
        multipoint_data,
        conductivity_types=[conductivity_type],
        show_legend=show_legend,
        plot_settings=plot_settings,
        fit_range=fit_range,
        show_fit=show_fit,
        fit_targets=[conductivity_type]
    )

    fit_result = fit_results.get(conductivity_type, {
        'Ea_kJ_mol': None, 'Ea_eV': None, 'slope': None, 'intercept': None
    })

    return fig_sigma_T, fit_result


def create_mapping_plot_1d(
    mapping_data: List[Dict],
    plot_settings: Dict = None,
    value_type: str = 'total_sigma',
    x_label: str = 'Position',
    y_label: str = None,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    show_zeroline: bool = True,
    use_log_scale: bool = False
) -> go.Figure:
    """
    Create 1D mapping plot (conductivity vs position).

    Args:
        mapping_data: List of dicts with keys: 'filename', 'x', 'sigma' (conductivity values)
        plot_settings: Dict with colorscale, marker_size, font_size, show_colorbar
        value_type: Type of conductivity to display
        x_label: Label for x-axis
        y_label: Label for y-axis (defaults to σ with proper formatting)
        x_range: Tuple of (min, max) for x-axis range
        y_range: Tuple of (min, max) for y-axis range
        show_zeroline: Whether to show zero lines
        use_log_scale: Whether to use log scale for y-axis (conductivity)

    Returns:
        Plotly Figure
    """
    if plot_settings is None:
        plot_settings = {}

    # Get settings matching other plot styles
    tick_size = plot_settings.get('tick_font_size', 20)
    label_size = plot_settings.get('axis_label_font_size', 20)
    marker_size = plot_settings.get('map_marker_size', 10)
    marker_color = plot_settings.get('map_marker_color', '#1f77b4')
    marker_edge_color = plot_settings.get('map_marker_edge_color', '#000000')
    marker_edge_width = plot_settings.get('map_marker_edge_width', 1)
    line_color = plot_settings.get('map_line_color', '#1f77b4')
    line_width = plot_settings.get('map_line_width', 2)
    show_line = plot_settings.get('map_show_line', True)

    # Default y-axis label
    if y_label is None:
        y_label = 'log(σ / S cm<sup>-1</sup>)' if use_log_scale else 'σ / S cm<sup>-1</sup>'

    # Extract data and sort by x
    data_points = []
    for item in mapping_data:
        if item.get('x') is not None and item.get('sigma') is not None:
            sigma = item['sigma']
            if use_log_scale and sigma > 0:
                sigma = np.log10(sigma)
            data_points.append({
                'x': item['x'],
                'sigma': sigma,
                'sigma_orig': item['sigma'],
                'filename': item.get('filename', '')
            })

    if not data_points:
        fig = go.Figure()
        fig.add_annotation(
            text="No mapping data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='black')
        )
        fig.update_layout(**common_layout(plot_settings))
        return fig

    # Sort by x value
    data_points.sort(key=lambda d: d['x'])
    x_vals = [d['x'] for d in data_points]
    sigma_vals = [d['sigma'] for d in data_points]
    sigma_orig = [d['sigma_orig'] for d in data_points]
    labels = [d['filename'] for d in data_points]

    fig = go.Figure()

    # Add line first (behind markers)
    if show_line:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=sigma_vals,
            mode='lines',
            line=dict(color=line_color, width=line_width),
            hoverinfo='skip',
            showlegend=False
        ))

    # Add markers
    hover_template = '<b>%{text}</b><br>' + x_label + ': %{x}<br>σ: %{customdata:.2e} S/cm<extra></extra>'
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=sigma_vals,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=marker_color,
            line=dict(color=marker_edge_color, width=marker_edge_width)
        ),
        text=labels,
        customdata=sigma_orig,
        hovertemplate=hover_template,
        showlegend=False
    ))

    # Apply axis settings matching Nyquist/Arrhenius style
    axis_settings = common_axis_settings(plot_settings)

    # Handle zeroline setting
    if not show_zeroline:
        axis_settings = {k: v for k, v in axis_settings.items()
                        if k not in ['zeroline', 'zerolinecolor', 'zerolinewidth']}
        axis_settings['zeroline'] = False

    # Build axis configs
    xaxis_config = dict(
        **axis_settings,
        title=dict(text=x_label, font=dict(size=label_size, color='black')),
    )
    if x_range is not None:
        xaxis_config['range'] = x_range

    yaxis_config = dict(
        **axis_settings,
        title=dict(text=y_label, font=dict(size=label_size, color='black')),
    )
    if not use_log_scale:
        yaxis_config['exponentformat'] = 'e'
    if y_range is not None:
        yaxis_config['range'] = y_range

    fig.update_layout(
        **common_layout(plot_settings),
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        showlegend=False
    )

    return fig


def create_mapping_plot_2d(
    mapping_data: List[Dict],
    plot_settings: Dict = None,
    value_type: str = 'total_sigma',
    interpolate: bool = False,
    interpolate_resolution: int = 50,
    x_label: str = 'X',
    y_label: str = 'Y',
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    show_zeroline: bool = False,
    use_log_scale: bool = False
) -> go.Figure:
    """
    Create 2D heatmap/contour plot for spatial conductivity distribution.

    Args:
        mapping_data: List of dicts with keys: 'filename', 'x', 'y', 'sigma' (conductivity)
        plot_settings: Dict with colorscale, marker_size, font_size, show_colorbar
        value_type: Type of conductivity to display
        interpolate: If True, interpolate data to create filled contour plot
        interpolate_resolution: Grid resolution for interpolation
        x_label: Label for x-axis
        y_label: Label for y-axis
        x_range: Tuple of (min, max) for x-axis range
        y_range: Tuple of (min, max) for y-axis range
        show_zeroline: Whether to show zero lines
        use_log_scale: Whether to use log scale for colorbar

    Returns:
        Plotly Figure
    """
    if plot_settings is None:
        plot_settings = {}

    colorscale = plot_settings.get('map_colorscale', 'Jet')
    tick_size = plot_settings.get('tick_font_size', 20)
    label_size = plot_settings.get('axis_label_font_size', 20)
    marker_size = plot_settings.get('map_marker_size', 10)
    marker_edge_color = plot_settings.get('map_marker_edge_color', '#000000')
    marker_edge_width = plot_settings.get('map_marker_edge_width', 1)
    show_colorbar = plot_settings.get('map_show_colorbar', True)
    marker_alpha = plot_settings.get('map_marker_alpha', 1.0)

    # Extract data
    x_vals = []
    y_vals = []
    sigma_vals = []
    labels = []

    for item in mapping_data:
        if (item.get('x') is not None and
            item.get('y') is not None and
            item.get('sigma') is not None):
            x_vals.append(item['x'])
            y_vals.append(item['y'])
            sigma_vals.append(item['sigma'])
            labels.append(item.get('filename', ''))

    if not x_vals:
        fig = go.Figure()
        fig.add_annotation(
            text="No mapping data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='black')
        )
        fig.update_layout(**common_layout(plot_settings))
        return fig

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    sigma_vals = np.array(sigma_vals)
    sigma_orig = sigma_vals.copy()

    # Apply log scale if requested
    if use_log_scale:
        sigma_vals = np.log10(np.where(sigma_vals > 0, sigma_vals, np.nan))

    fig = go.Figure()

    # Colorbar settings
    colorbar_label = 'log(σ / S cm<sup>-1</sup>)' if use_log_scale else 'σ / S cm<sup>-1</sup>'
    colorbar_settings = dict(
        title=dict(
            text=colorbar_label,
            font=dict(size=label_size, color='black')
        ),
        tickfont=dict(size=tick_size, color='black'),
    )
    if not use_log_scale:
        colorbar_settings['exponentformat'] = 'e'

    if interpolate and len(x_vals) >= 4:
        # Interpolate using scipy
        try:
            from scipy.interpolate import griddata

            # Create grid
            xi = np.linspace(x_vals.min(), x_vals.max(), interpolate_resolution)
            yi = np.linspace(y_vals.min(), y_vals.max(), interpolate_resolution)
            xi_grid, yi_grid = np.meshgrid(xi, yi)

            # Interpolate
            zi = griddata((x_vals, y_vals), sigma_vals, (xi_grid, yi_grid), method='cubic')

            # Fill NaN with nearest neighbor
            zi_nearest = griddata((x_vals, y_vals), sigma_vals, (xi_grid, yi_grid), method='nearest')
            zi = np.where(np.isnan(zi), zi_nearest, zi)

            # Add contour plot
            hover_sigma = 'log(σ): %{z:.2f}' if use_log_scale else 'σ: %{z:.2e} S/cm'
            fig.add_trace(go.Contour(
                x=xi,
                y=yi,
                z=zi,
                colorscale=colorscale,
                showscale=show_colorbar,
                colorbar=colorbar_settings if show_colorbar else None,
                contours=dict(
                    coloring='heatmap',
                    showlabels=False
                ),
                hovertemplate=f'X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>{hover_sigma}<extra></extra>'
            ))

            # Add scatter points on top with color based on sigma
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=sigma_vals,
                    colorscale=colorscale,
                    opacity=marker_alpha,
                    showscale=False,
                    line=dict(color=marker_edge_color, width=marker_edge_width)
                ),
                text=labels,
                customdata=sigma_orig,
                hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<br>σ: %{customdata:.2e} S/cm<extra></extra>',
                showlegend=False
            ))

        except ImportError:
            # Fall back to scatter if scipy not available
            interpolate = False

    if not interpolate:
        # Check if data forms a regular grid
        unique_x = sorted(set(x_vals))
        unique_y = sorted(set(y_vals))

        use_heatmap = False
        if len(unique_x) > 1 and len(unique_y) > 1:
            z_matrix = np.full((len(unique_y), len(unique_x)), np.nan)
            for x, y, sigma in zip(x_vals, y_vals, sigma_vals):
                xi = unique_x.index(x)
                yi = unique_y.index(y)
                z_matrix[yi, xi] = sigma

            fill_ratio = np.count_nonzero(~np.isnan(z_matrix)) / z_matrix.size
            if fill_ratio > 0.5:
                use_heatmap = True
                hover_sigma = 'log(σ): %{z:.2f}' if use_log_scale else 'σ: %{z:.2e} S/cm'
                fig.add_trace(go.Heatmap(
                    x=unique_x,
                    y=unique_y,
                    z=z_matrix,
                    colorscale=colorscale,
                    showscale=show_colorbar,
                    colorbar=colorbar_settings if show_colorbar else None,
                    hovertemplate=f'X: %{{x}}<br>Y: %{{y}}<br>{hover_sigma}<extra></extra>'
                ))

        if not use_heatmap:
            # Scatter plot for irregular data
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=sigma_vals,
                    colorscale=colorscale,
                    showscale=show_colorbar,
                    colorbar=colorbar_settings if show_colorbar else None,
                    opacity=marker_alpha,
                    line=dict(color=marker_edge_color, width=marker_edge_width)
                ),
                text=labels,
                customdata=sigma_orig,
                hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<br>σ: %{customdata:.2e} S/cm<extra></extra>',
                showlegend=False
            ))

    # Apply axis settings
    axis_settings = common_axis_settings(plot_settings)

    # Handle zeroline setting
    axis_settings_2d = {k: v for k, v in axis_settings.items()
                        if k not in ['zeroline', 'zerolinecolor', 'zerolinewidth']}
    if show_zeroline:
        axis_settings_2d['zeroline'] = True
        axis_settings_2d['zerolinecolor'] = 'gray'
        axis_settings_2d['zerolinewidth'] = 1
    else:
        axis_settings_2d['zeroline'] = False

    # Build axis configs
    xaxis_config = dict(
        **axis_settings_2d,
        title=dict(text=x_label, font=dict(size=label_size, color='black')),
    )
    if x_range is not None:
        xaxis_config['range'] = x_range

    yaxis_config = dict(
        **axis_settings_2d,
        title=dict(text=y_label, font=dict(size=label_size, color='black')),
        scaleanchor='x',
        scaleratio=1
    )
    if y_range is not None:
        yaxis_config['range'] = y_range

    fig.update_layout(
        **common_layout(plot_settings),
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        showlegend=False
    )

    return fig


def create_mapping_plot_ternary(
    mapping_data: List[Dict],
    plot_settings: Dict = None,
    value_type: str = 'total_sigma',
    interpolate: bool = False,
    interpolate_resolution: int = 50,
    axis_labels: Tuple[str, str, str] = ('A', 'B', 'C'),
    use_subscript: bool = True,
    use_log_scale: bool = False
) -> go.Figure:
    """
    Create ternary plot for 3-component composition mapping.

    Args:
        mapping_data: List of dicts with keys: 'filename', 'a', 'b', 'c', 'sigma'
                      where a, b, c are composition fractions (should sum to 1 or 100)
        plot_settings: Dict with colorscale, marker_size, font_size, show_colorbar
        value_type: Type of conductivity to display
        interpolate: If True, interpolate data to create filled contour
        interpolate_resolution: Resolution for interpolation grid
        axis_labels: Tuple of labels for (A, B, C) axes
        use_subscript: Whether to apply subscript formatting to axis labels
        use_log_scale: Whether to use log scale for colorbar

    Returns:
        Plotly Figure
    """
    if plot_settings is None:
        plot_settings = {}

    colorscale = plot_settings.get('map_colorscale', 'Jet')
    tick_size = plot_settings.get('tick_font_size', 20)
    label_size = plot_settings.get('axis_label_font_size', 20)
    marker_size = plot_settings.get('map_marker_size', 10)
    marker_edge_color = plot_settings.get('map_marker_edge_color', '#000000')
    marker_edge_width = plot_settings.get('map_marker_edge_width', 1)
    show_colorbar = plot_settings.get('map_show_colorbar', True)
    marker_alpha = plot_settings.get('map_marker_alpha', 1.0)
    # Interpolation fill marker size (larger for filled look)
    fill_marker_size = plot_settings.get('map_ternary_fill_size', 8)

    # Apply subscript formatting to axis labels if requested
    if use_subscript:
        display_labels = tuple(subscript_composition(label) for label in axis_labels)
    else:
        display_labels = axis_labels

    # Extract data
    a_vals = []
    b_vals = []
    c_vals = []
    sigma_vals = []
    labels = []

    for item in mapping_data:
        if (item.get('a') is not None and
            item.get('b') is not None and
            item.get('c') is not None and
            item.get('sigma') is not None):
            a_vals.append(item['a'])
            b_vals.append(item['b'])
            c_vals.append(item['c'])
            sigma_vals.append(item['sigma'])
            labels.append(item.get('filename', ''))

    if not a_vals:
        fig = go.Figure()
        fig.add_annotation(
            text="No ternary data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='black')
        )
        fig.update_layout(**common_layout(plot_settings))
        return fig

    a_vals = np.array(a_vals)
    b_vals = np.array(b_vals)
    c_vals = np.array(c_vals)
    sigma_vals = np.array(sigma_vals)
    sigma_orig = sigma_vals.copy()

    # Normalize to sum to 1 if needed
    totals = a_vals + b_vals + c_vals
    if np.mean(totals) > 10:  # Likely in percentage
        a_vals = a_vals / totals
        b_vals = b_vals / totals
        c_vals = c_vals / totals

    # Apply log scale if requested
    if use_log_scale:
        sigma_vals = np.log10(np.where(sigma_vals > 0, sigma_vals, np.nan))

    # Colorbar settings
    colorbar_label = 'log(σ / S cm<sup>-1</sup>)' if use_log_scale else 'σ / S cm<sup>-1</sup>'
    colorbar_settings = dict(
        title=dict(
            text=colorbar_label,
            font=dict(size=label_size, color='black')
        ),
        tickfont=dict(size=tick_size, color='black'),
        len=0.8,
        x=1.02
    )
    if not use_log_scale:
        colorbar_settings['exponentformat'] = 'e'

    fig = go.Figure()

    if interpolate and len(a_vals) >= 4:
        try:
            from scipy.interpolate import griddata

            # Create ternary grid
            n = interpolate_resolution
            a_grid = []
            b_grid = []
            c_grid = []

            for i in range(n + 1):
                for j in range(n + 1 - i):
                    a = i / n
                    b = j / n
                    c = 1 - a - b
                    if c >= 0:
                        a_grid.append(a)
                        b_grid.append(b)
                        c_grid.append(c)

            a_grid = np.array(a_grid)
            b_grid = np.array(b_grid)
            c_grid = np.array(c_grid)

            # Interpolate sigma values
            sigma_grid = griddata(
                (a_vals, b_vals),
                sigma_vals,
                (a_grid, b_grid),
                method='cubic'
            )

            # Fill NaN with nearest
            sigma_nearest = griddata(
                (a_vals, b_vals),
                sigma_vals,
                (a_grid, b_grid),
                method='nearest'
            )
            sigma_grid = np.where(np.isnan(sigma_grid), sigma_nearest, sigma_grid)

            # Add filled contour ternary with larger markers for filled look
            fig.add_trace(go.Scatterternary(
                a=a_grid,
                b=b_grid,
                c=c_grid,
                mode='markers',
                marker=dict(
                    size=fill_marker_size,
                    color=sigma_grid,
                    colorscale=colorscale,
                    showscale=show_colorbar,
                    colorbar=colorbar_settings if show_colorbar else None,
                    opacity=marker_alpha
                ),
                hoverinfo='skip',
                showlegend=False
            ))

        except ImportError:
            interpolate = False

    # Add scatter points (always show original data points)
    fig.add_trace(go.Scatterternary(
        a=a_vals,
        b=b_vals,
        c=c_vals,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=sigma_vals if not interpolate else sigma_vals,
            colorscale=colorscale,
            showscale=show_colorbar if not interpolate else False,
            colorbar=colorbar_settings if (not interpolate and show_colorbar) else None,
            opacity=marker_alpha,
            line=dict(color=marker_edge_color, width=marker_edge_width)
        ),
        text=labels,
        customdata=sigma_orig,
        hovertemplate='<b>%{text}</b><br>' +
                     f'{axis_labels[0]}' + ': %{a:.2f}<br>' +
                     f'{axis_labels[1]}' + ': %{b:.2f}<br>' +
                     f'{axis_labels[2]}' + ': %{c:.2f}<br>' +
                     'σ: %{customdata:.2e} S/cm<extra></extra>',
        showlegend=False
    ))

    # Update layout for ternary
    # Hide tick values (0.2, 0.4, 0.6, etc.) and only show axis labels
    fig.update_layout(
        **common_layout(plot_settings),
        ternary=dict(
            sum=1,
            aaxis=dict(
                title=dict(
                    text=display_labels[0],
                    font=dict(size=label_size, color='black')
                ),
                showticklabels=False,  # Hide tick labels
                linecolor='black',
                linewidth=1,
                gridcolor='lightgray',
                ticks='',  # No ticks
            ),
            baxis=dict(
                title=dict(
                    text=display_labels[1],
                    font=dict(size=label_size, color='black')
                ),
                showticklabels=False,  # Hide tick labels
                linecolor='black',
                linewidth=1,
                gridcolor='lightgray',
                ticks='',  # No ticks
            ),
            caxis=dict(
                title=dict(
                    text=display_labels[2],
                    font=dict(size=label_size, color='black')
                ),
                showticklabels=False,  # Hide tick labels
                linecolor='black',
                linewidth=1,
                gridcolor='lightgray',
                ticks='',  # No ticks
            ),
            bgcolor='white'
        ),
        showlegend=False,
        # Add margin at top to prevent label clipping
        margin=dict(t=80, l=60, r=60, b=60)
    )

    return fig
