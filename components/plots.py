"""
Plotly-based plotting functions for EIS data visualization
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Tuple

# Color palette for multiple files
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


def get_unit_scale(max_value: float) -> Tuple[float, str]:
    """
    Determine appropriate unit scale based on max value.
    Returns (divisor, unit_prefix)
    """
    abs_max = abs(max_value)
    if abs_max >= 1e9:
        return 1e9, 'G'
    elif abs_max >= 1e6:
        return 1e6, 'M'
    elif abs_max >= 1e3:
        return 1e3, 'k'
    else:
        return 1, ''


def get_axis_range(data_min: float, data_max: float) -> Tuple[float, float]:
    """
    Calculate axis range with 5% margin.
    """
    range_val = data_max - data_min
    if range_val == 0:
        range_val = abs(data_max) * 0.1 if data_max != 0 else 1
    margin = range_val * 0.05
    return data_min - margin, data_max + margin


def common_layout() -> dict:
    """Common layout settings for all plots"""
    return {
        'font': {'family': 'Arial', 'color': 'black'},
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'margin': {'l': 50, 'r': 10, 't': 25, 'b': 35},
    }


def common_axis_settings() -> dict:
    """Common axis settings"""
    return {
        'showgrid': False,
        'showline': True,
        'linewidth': 1,
        'linecolor': 'black',
        'tickcolor': 'black',
        'tickfont': {'family': 'Arial', 'color': 'black', 'size': 10},
        'title_font': {'family': 'Arial', 'color': 'black', 'size': 11},
        'mirror': True,
        'ticks': 'outside',
        'minor': {'ticks': 'outside', 'ticklen': 3, 'showgrid': False},
        'ticklen': 5,
        'title_standoff': 3,
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
    highlight_freq: bool = False
) -> go.Figure:
    """
    Create interactive Nyquist plot using Plotly
    """
    fig = go.Figure()

    if selected_files is None:
        selected_files = list(data_dict.keys())

    # Collect all data for range calculation
    all_real = []
    all_imag = []

    for idx, file_name in enumerate(selected_files):
        if file_name not in data_dict:
            continue

        data = data_dict[file_name]
        Z = data['Z']
        freq = data['freq']
        color = COLORS[idx % len(COLORS)]

        z_real = np.real(Z)
        z_imag = -np.imag(Z)  # -Z'' for Nyquist

        all_real.extend(z_real)
        all_imag.extend(z_imag)

        # Add measured data
        fig.add_trace(go.Scatter(
            x=z_real,
            y=z_imag,
            mode='markers',
            name=file_name,
            marker=dict(size=5, color=color),
            hovertemplate="<i>Z'</i> = %{x:.2f}<br>-<i>Z''</i> = %{y:.2f}<extra></extra>",
            showlegend=show_legend
        ))

        # Highlight decade frequencies
        if highlight_freq:
            decade_indices = find_decade_frequencies(freq)
            if decade_indices:
                fig.add_trace(go.Scatter(
                    x=z_real[decade_indices],
                    y=z_imag[decade_indices],
                    mode='markers',
                    name=f'{file_name} (decades)',
                    marker=dict(size=10, color='red', symbol='circle-open', line=dict(width=2)),
                    hovertemplate="<i>f</i> = %{text} Hz<extra></extra>",
                    text=[f"{freq[i]:.0e}" for i in decade_indices],
                    showlegend=False
                ))

        # Add fitted data if available
        if show_fit and data.get('Z_fit') is not None:
            Z_fit = data['Z_fit']
            fig.add_trace(go.Scatter(
                x=np.real(Z_fit),
                y=-np.imag(Z_fit),
                mode='lines',
                name=f'{file_name} (fit)',
                line=dict(width=2, color=color, dash='solid'),
                showlegend=show_legend
            ))

    # Calculate range with 5% margin
    if all_real and all_imag:
        all_vals = all_real + all_imag
        data_min = min(all_vals)
        data_max = max(all_vals)
        axis_min, axis_max = get_axis_range(data_min, data_max)

        # Determine unit scale
        scale, prefix = get_unit_scale(data_max)
    else:
        axis_min, axis_max = 0, 1
        scale, prefix = 1, ''

    # Update layout
    axis_settings = common_axis_settings()

    fig.update_layout(
        **common_layout(),
        height=320,
        xaxis=dict(
            **axis_settings,
            title=f"<i>Z'</i> / {prefix}Ω" if prefix else "<i>Z'</i> / Ω",
            range=[axis_min / scale, axis_max / scale],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            **axis_settings,
            title=f"-<i>Z''</i> / {prefix}Ω" if prefix else "-<i>Z''</i> / Ω",
            range=[axis_min / scale, axis_max / scale],
        ),
        showlegend=show_legend,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=8), bgcolor='rgba(255,255,255,0.8)'
        )
    )

    # Scale the data if using k, M, G units
    if scale != 1:
        for trace in fig.data:
            if hasattr(trace, 'x') and trace.x is not None:
                trace.x = tuple(np.array(trace.x) / scale)
            if hasattr(trace, 'y') and trace.y is not None:
                trace.y = tuple(np.array(trace.y) / scale)

    return fig


def create_bode_plot(
    data_dict: Dict,
    selected_files: Optional[List[str]] = None,
    show_fit: bool = False,
    show_legend: bool = True,
    freq_range: Optional[Tuple[int, int]] = None
) -> go.Figure:
    """
    Create interactive Bode plot using Plotly with shared x-axis
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5]
    )

    if selected_files is None:
        selected_files = list(data_dict.keys())

    # Collect max |Z| for unit scaling
    all_mag = []

    for idx, file_name in enumerate(selected_files):
        if file_name not in data_dict:
            continue

        data = data_dict[file_name]
        freq = data['freq']
        Z = data['Z']
        color = COLORS[idx % len(COLORS)]

        log_freq = np.log10(freq)
        magnitude = np.abs(Z)
        phase = -np.angle(Z, deg=True)

        all_mag.extend(magnitude)

        # Magnitude plot
        fig.add_trace(go.Scatter(
            x=log_freq,
            y=magnitude,
            mode='markers',
            name=file_name,
            marker=dict(size=4, color=color),
            hovertemplate="log(<i>f</i>) = %{x:.2f}<br>|<i>Z</i>| = %{y:.2e}<extra></extra>",
            showlegend=show_legend
        ), row=1, col=1)

        # Phase plot
        fig.add_trace(go.Scatter(
            x=log_freq,
            y=phase,
            mode='markers',
            name=file_name,
            marker=dict(size=4, color=color),
            hovertemplate="log(<i>f</i>) = %{x:.2f}<br><i>θ</i> = %{y:.1f}°<extra></extra>",
            showlegend=False
        ), row=2, col=1)

        # Add fitted data if available (only within freq_range)
        if show_fit and data.get('Z_fit') is not None:
            Z_fit = data['Z_fit']

            # Limit to fitting range
            if freq_range:
                start_idx, end_idx = freq_range
                freq_fit = freq[start_idx:end_idx + 1]
                Z_fit_range = Z_fit[start_idx:end_idx + 1]
            else:
                freq_fit = freq
                Z_fit_range = Z_fit

            log_freq_fit = np.log10(freq_fit)

            fig.add_trace(go.Scatter(
                x=log_freq_fit,
                y=np.abs(Z_fit_range),
                mode='lines',
                name=f'{file_name} (fit)',
                line=dict(width=2, color=color),
                showlegend=show_legend
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=log_freq_fit,
                y=-np.angle(Z_fit_range, deg=True),
                mode='lines',
                name=f'{file_name} (fit)',
                line=dict(width=2, color=color),
                showlegend=False
            ), row=2, col=1)

    # Determine unit scale for magnitude
    if all_mag:
        max_mag = max(all_mag)
        scale, prefix = get_unit_scale(max_mag)
    else:
        scale, prefix = 1, ''

    # Update layout
    axis_settings = common_axis_settings()

    fig.update_layout(
        **common_layout(),
        height=320,
        title=dict(text='Bode', x=0.5, font=dict(size=11, family='Arial', color='black')),
        showlegend=show_legend,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=8), bgcolor='rgba(255,255,255,0.8)'
        )
    )

    # X-axis (shared, no tick labels on top)
    fig.update_xaxes(
        **axis_settings,
        showticklabels=False,
        title='',
        row=1, col=1
    )
    fig.update_xaxes(
        **axis_settings,
        title="log(<i>f</i> / Hz)",
        showticklabels=False,
        row=2, col=1
    )

    # Y-axes
    fig.update_yaxes(
        **axis_settings,
        title=f"|<i>Z</i>| / {prefix}Ω" if prefix else "|<i>Z</i>| / Ω",
        type="log",
        row=1, col=1
    )
    fig.update_yaxes(
        **axis_settings,
        title="<i>θ</i> / °",
        row=2, col=1
    )

    return fig


def create_arrhenius_plot(
    multipoint_data: List[Dict],
    conductivity_type: str = 'total',
    show_legend: bool = True
) -> go.Figure:
    """
    Create Arrhenius plot (log(σT) vs 1000/T)
    """
    fig = go.Figure()

    axis_settings = common_axis_settings()

    if len(multipoint_data) == 0:
        fig.update_layout(
            **common_layout(),
            height=320,
            title=dict(text='Arrhenius', x=0.5, font=dict(size=11, family='Arial', color='black')),
            xaxis=dict(**axis_settings, title="1000/<i>T</i> / K⁻¹"),
            yaxis=dict(**axis_settings, title="log(<i>σT</i>) / S cm⁻¹ K"),
        )
        return fig

    # Extract data
    temperatures = []
    sigma_T_values = []

    for data in multipoint_data:
        if 'temperature' not in data:
            continue

        T = data['temperature']
        if T <= 0:
            continue

        sigma_key = f'{conductivity_type}_sigma'
        if sigma_key not in data or data[sigma_key] is None:
            continue

        sigma = data[sigma_key]
        if sigma <= 0:
            continue

        temperatures.append(1000 / T)
        sigma_T_values.append(np.log10(sigma * T))

    if len(temperatures) > 0:
        fig.add_trace(go.Scatter(
            x=temperatures,
            y=sigma_T_values,
            mode='markers+lines',
            name=f'{conductivity_type.capitalize()}',
            marker=dict(size=6, color=COLORS[0]),
            line=dict(width=2, color=COLORS[0]),
            hovertemplate="1000/<i>T</i> = %{x:.4f}<br>log(<i>σT</i>) = %{y:.4f}<extra></extra>",
            showlegend=show_legend
        ))

    fig.update_layout(
        **common_layout(),
        height=320,
        title=dict(text='Arrhenius', x=0.5, font=dict(size=11, family='Arial', color='black')),
        xaxis=dict(**axis_settings, title="1000/<i>T</i> / K⁻¹"),
        yaxis=dict(**axis_settings, title="log(<i>σT</i>) / S cm⁻¹ K"),
        showlegend=show_legend,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=8), bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig
