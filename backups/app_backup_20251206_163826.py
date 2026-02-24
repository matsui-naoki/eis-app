"""
EIS Analyzer - Streamlit Web Application
Electrochemical Impedance Spectroscopy Analysis Tool
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
from datetime import datetime
import io
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from tools.data_loader import load_uploaded_file, load_uploaded_file_with_loops, validate_impedance_data
from tools.fitting import circuit_fit, calc_rmspe, r2sigma, r2logsigma, effective_capacitance, sort_ecm_by_cap, BlackBoxOptEIS
from components.plots import create_nyquist_plot, create_bode_plot, create_arrhenius_plot
from utils.help_texts import (
    CIRCUIT_MODEL_HELP, WEIGHT_METHOD_HELP, RMSPE_HELP, SUMMARY_TABLE_HELP,
    FIT_SETTINGS_HELP, AUTO_FIT_SETTINGS_HELP, BATCH_FIT_SETTINGS_HELP,
    SAMPLE_INFO_HELP, VF_TOGGLE_HELP, FITTING_RANGE_HELP,
    TEMPERATURE_PATTERN_HELP, FILENAME_PATTERN_HELP,
    ARRHENIUS_PLOT_HELP, FILE_FORMAT_HELP
)
from utils.helpers import format_param_name, parse_temperature_pattern, extract_temp_from_filename

# Use impedance library for circuit model
from impedance.models.circuits import CustomCircuit


# Page configuration
st.set_page_config(
    page_title="EIS Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Global font - Arial */
    body, p, span, div, input, button, label, h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stText, .stTextInput, .stButton, .stSelectbox {
        font-family: Arial, Helvetica, sans-serif;
        color: #000000;
    }

    /* Pure white background everywhere */
    .stApp, .main, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    .block-container {
        background-color: #ffffff !important;
    }

    /* Input fields with light gray background, no border */
    input, .stTextInput input, .stNumberInput input {
        border: none !important;
        border-radius: 4px !important;
        padding: 0.3rem 0.5rem !important;
        background-color: #f0f0f0 !important;
    }

    /* Button styling - light blue */
    .stButton > button {
        width: 100%;
        background-color: #e3f2fd !important;
        color: #000000 !important;
        border: 1px solid #90caf9 !important;
    }

    .stButton > button:hover {
        background-color: #bbdefb !important;
        border-color: #64b5f6 !important;
    }

    /* Primary button - darker blue */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #1976d2 !important;
        color: #ffffff !important;
        border-color: #1976d2 !important;
    }


    /* Comments/help text - light gray */
    .stCaption, small, .stTooltipIcon {
        color: #888888 !important;
    }

    /* Hide Streamlit footer only, keep header/toolbar visible */
    footer {
        display: none !important;
    }

    /* Sidebar width when expanded */
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 300px !important;
        min-width: 300px !important;
    }

    /* Sidebar border - right edge */
    section[data-testid="stSidebar"] {
        border-right: 2px solid #e0e0e0 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.2rem !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0 !important;
    }

    /* Sidebar general text - smaller font */
    section[data-testid="stSidebar"] .stMarkdown p {
        font-size: 0.75rem !important;
    }

    section[data-testid="stSidebar"] h3 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.1rem !important;
        font-size: 0.8rem !important;
        color: #000000 !important;
    }

    /* Sidebar file list buttons - smaller text */
    section[data-testid="stSidebar"] .stButton > button {
        font-size: 0.6rem !important;
        padding: 0.1rem 0.25rem !important;
    }

    /* Sidebar labels and captions */
    section[data-testid="stSidebar"] label {
        font-size: 0.7rem !important;
    }

    section[data-testid="stSidebar"] .stCaption {
        font-size: 0.65rem !important;
    }

    /* Sidebar selectbox, text input, number input */
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stTextInput,
    section[data-testid="stSidebar"] .stNumberInput {
        font-size: 0.7rem !important;
    }

    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stTextInput > div > div > input,
    section[data-testid="stSidebar"] .stNumberInput > div > div > input {
        font-size: 0.7rem !important;
    }

    /* Metric value - same size as label */
    [data-testid="stMetricValue"] {
        font-size: 1rem !important;
    }

    /* Reduce main content gaps and padding */
    [data-testid="stVerticalBlock"] {
        gap: 0.3rem !important;
    }

    .block-container {
        padding-top: 4rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }

    [data-testid="stAppViewContainer"] > div:first-child {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    /* Sidebar header title */
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1976d2;
        margin-bottom: 0.3rem;
    }

    /* Selectbox styling - light gray background */
    .stSelectbox > div > div {
        background-color: #f0f0f0 !important;
        border: none !important;
        border-radius: 4px !important;
    }

    .stSelectbox [data-baseweb="select"] > div {
        background-color: #f0f0f0 !important;
        border: none !important;
    }

    /* Selectbox dropdown arrow area */
    .stSelectbox [data-baseweb="select"] > div > div {
        background-color: #f0f0f0 !important;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'files' not in st.session_state:
        st.session_state.files = {}  # {filename: {freq, Z, Z_fit, circuit_params, ...}}
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = None
    if 'sample_info' not in st.session_state:
        st.session_state.sample_info = {
            'name': '',
            'thickness': 0.1,  # cm
            'diameter': 1.0,   # cm
            'area': 0.785      # cm^2 (default from diameter=1.0)
        }
    if 'area_input_mode' not in st.session_state:
        st.session_state.area_input_mode = 'Diameter'  # 'Diameter' or 'Area'
    if 'arrhenius_mode' not in st.session_state:
        st.session_state.arrhenius_mode = False
    if 'multipoint_data' not in st.session_state:
        st.session_state.multipoint_data = []
    if 'show_fit' not in st.session_state:
        st.session_state.show_fit = True
    if 'show_all_data' not in st.session_state:
        st.session_state.show_all_data = False
    if 'freq_range' not in st.session_state:
        st.session_state.freq_range = (0, 70)  # Default range 0-70
    if 'deleted_points' not in st.session_state:
        st.session_state.deleted_points = []  # List of deleted indices
    if 'show_legend' not in st.session_state:
        st.session_state.show_legend = True
    if 'highlight_freq' not in st.session_state:
        st.session_state.highlight_freq = False
    if 'plot_settings' not in st.session_state:
        st.session_state.plot_settings = {
            # Unit settings
            'z_unit': 'Ω',  # 'Ω', 'kΩ', 'MΩ', 'GΩ'
            # Font settings
            'tick_font_size': 18,
            'axis_label_font_size': 18,
            # Marker settings
            'marker_color': '#FFFFFF',
            'marker_symbol': 'circle',
            'marker_size': 10,
            'marker_alpha': 0.8,
            'marker_line_color': '#000000',
            'marker_line_width': 2,
            # Fit line settings
            'fit_line_color': '#FF0000',
            'fit_line_width': 2,
            # Zero line
            'show_zeroline': True,
            # Legend settings
            'legend_font_size': 10,
            'plot_legend_mode': 'Filename',  # 'Filename', 'Temperature', 'Manual'
            'plot_legend_manual': 'measured',
            'fit_legend_mode': 'Filename',  # 'Filename', 'Manual'
            'fit_legend_manual': 'fitted',
            # Arrhenius plot settings
            'arr_marker_color': '#5AA4E6',
            'arr_marker_symbol': 'circle',
            'arr_marker_size': 10,
            'arr_marker_edge_color': '#000000',
            'arr_marker_edge_width': 1,
            'arr_line_color': '#5AA4E6',
            'arr_line_width': 2,
            'arr_show_line': False,
        }
    if 'r_labels' not in st.session_state:
        # Labels for R elements (R1, R2, ...) - default: bulk, gb
        st.session_state.r_labels = {'R1': 'bulk', 'R2': 'gb', 'R3': 'R3'}
    if 'arr_fit_targets' not in st.session_state:
        # Which conductivities to show/fit in Arrhenius plot
        st.session_state.arr_fit_targets = ['total']


def sidebar_header():
    """Render sidebar header with title"""
    st.markdown('<div class="sidebar-title">EIS Analyzer</div>', unsafe_allow_html=True)


def process_uploaded_files(uploaded_files):
    """Process uploaded EIS files and add to session state"""
    if not uploaded_files:
        return
        
    for uploaded_file in uploaded_files:
        original_filename = uploaded_file.name
        base_name = os.path.splitext(original_filename)[0]
        file_ext = os.path.splitext(original_filename)[1]

        # Check if this file has already been processed
        already_loaded = any(
            name == base_name or name.startswith(f"{base_name}_")
            for name in st.session_state.files.keys()
        )

        if not already_loaded:
            try:
                # Load with loop detection
                datasets, error = load_uploaded_file_with_loops(
                    uploaded_file, file_ext, base_name, rtol=0.01
                )

                if error:
                    st.error(f"Error loading {original_filename}: {error}")
                else:
                    loaded_count = 0
                    for dataset in datasets:
                        name = dataset['name']
                        freq = dataset['freq']
                        Z = dataset['Z']

                        is_valid, msg = validate_impedance_data(freq, Z)
                        if is_valid:
                            st.session_state.files[name] = {
                                'freq': freq,
                                'Z': Z,
                                'Z_fit': None,
                                'circuit_model': None,
                                'circuit_params': None,
                                'circuit_conf': None,
                                'rmspe': None,
                                'temperature': None
                            }
                            loaded_count += 1
                        else:
                            st.warning(f"Invalid data in {name}: {msg}")

                    if loaded_count > 0:
                        if len(datasets) > 1:
                            st.success(f"Loaded {loaded_count} datasets from {original_filename}")
                        else:
                            st.success(f"Loaded: {datasets[0]['name']}")
            except Exception as e:
                st.error(f"Unexpected error loading {original_filename}: {str(e)}")


def sidebar_file_upload():
    """File upload section in sidebar - EIS files and session files"""
    st.markdown("### Upload Files")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=['mpt', 'z', 'dta', 'csv', 'txt', 'par', 'json'],
        accept_multiple_files=True,
        help=FILE_FORMAT_HELP,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith('.json'):
                # Load as session file
                load_session(uploaded_file)
            else:
                # Process as EIS file
                process_uploaded_files([uploaded_file])


def sidebar_sample_info():
    """Render sample information input section in sidebar"""
    st.markdown("### Sample Information")

    # Sample Name
    st.session_state.sample_info['name'] = st.text_input(
        "Sample name",
        value=st.session_state.sample_info.get('name', ''),
        placeholder="Sample name",
        help=SAMPLE_INFO_HELP['label']
    )

    # Thickness
    thickness_value = st.text_input(
        "Thickness (cm)",
        value=str(st.session_state.sample_info.get('thickness', 0.1)),
        placeholder="0.1",
        help=SAMPLE_INFO_HELP['thickness']
    )
    try:
        thickness_float = float(thickness_value)
        if thickness_float > 0:
            st.session_state.sample_info['thickness'] = thickness_float
        else:
            st.error("Thickness must be positive")
            st.session_state.sample_info['thickness'] = 0.1
    except ValueError:
        st.error("Invalid thickness value")
        st.session_state.sample_info['thickness'] = 0.1

    # Diameter or Area selection
    st.session_state.area_input_mode = st.radio(
        "Input mode",
        ["Diameter", "Area"],
        index=0 if st.session_state.area_input_mode == 'Diameter' else 1,
        horizontal=True,
        label_visibility="collapsed"
    )

    if st.session_state.area_input_mode == 'Diameter':
        diameter_value = st.text_input(
            "Diameter (cm)",
            value=str(st.session_state.sample_info.get('diameter', 1.0)),
            placeholder="1.0",
            help=SAMPLE_INFO_HELP['diameter']
        )
        try:
            new_diameter = float(diameter_value)
            if new_diameter <= 0:
                st.error("Diameter must be positive")
                new_diameter = 1.0
        except ValueError:
            st.error("Invalid diameter value")
            new_diameter = 1.0
        # Calculate area from diameter
        new_area = np.pi * (new_diameter / 2) ** 2
        st.session_state.sample_info['diameter'] = new_diameter
        st.session_state.sample_info['area'] = new_area
        st.caption(f"Area = {new_area:.4f} cm²")
    else:
        area_value = st.text_input(
            "Area (cm²)",
            value=str(st.session_state.sample_info.get('area', 0.785)),
            placeholder="0.785",
            help=SAMPLE_INFO_HELP['area']
        )
        try:
            new_area = float(area_value)
            if new_area <= 0:
                st.error("Area must be positive")
                new_area = 0.785
        except ValueError:
            st.error("Invalid area value")
            new_area = 0.785
        # Calculate diameter from area
        new_diameter = 2 * np.sqrt(new_area / np.pi)
        st.session_state.sample_info['area'] = new_area
        st.session_state.sample_info['diameter'] = new_diameter
        st.caption(f"Diameter = {new_diameter:.4f} cm")

    # Recalculate sigma for all files when area/thickness changes
    recalculate_sigma_for_all_files()


def recalculate_sigma_for_all_files():
    """Recalculate ionic conductivity for all files based on current sample info."""
    S = st.session_state.sample_info.get('area', 1.0)
    L = st.session_state.sample_info.get('thickness', 0.1)

    for filename, data in st.session_state.files.items():
        if data.get('total_R') is not None:
            R_total = data['total_R']
            data['total_sigma'] = r2sigma(R_total, S, L)


def sidebar_file_manager():
    """File management tab in sidebar"""
    # Arrhenius mode toggle - use key for immediate update
    arrhenius_mode = st.checkbox(
        "Arrhenius Mode",
        value=st.session_state.arrhenius_mode,
        help="Temperature-dependent analysis",
        key="arrhenius_mode_checkbox"
    )
    if arrhenius_mode != st.session_state.arrhenius_mode:
        st.session_state.arrhenius_mode = arrhenius_mode
        st.rerun()

    # Show all data checkbox - use key for immediate update
    show_all_data = st.checkbox(
        "Show all data",
        value=st.session_state.show_all_data,
        help="Display all loaded files on plots",
        key="show_all_data_checkbox"
    )
    if show_all_data != st.session_state.show_all_data:
        st.session_state.show_all_data = show_all_data
        st.rerun()

    # File list
    if len(st.session_state.files) > 0:
        for i, filename in enumerate(list(st.session_state.files.keys())):
            col1, col2 = st.columns([4, 1])

            is_selected = (filename == st.session_state.selected_file)

            with col1:
                if st.button(filename, key=f"select_{i}", width="stretch", type="primary" if is_selected else "secondary"):
                    st.session_state.selected_file = filename

            with col2:
                if st.button("Del", key=f"delete_{i}", help="Delete file"):
                    del st.session_state.files[filename]
                    if st.session_state.selected_file == filename:
                        st.session_state.selected_file = None
                    st.rerun()

        # Save session button
        st.markdown("---")
        if st.button("Save Session", width="stretch"):
            save_session()

    # Reset button at bottom
    st.markdown("---")
    if st.button("Reset", width="stretch"):
        reset_session()
        st.rerun()


def sidebar_data_view():
    """Data view tab in sidebar"""
    if st.session_state.selected_file and st.session_state.selected_file in st.session_state.files:
        filename = st.session_state.selected_file
        data = st.session_state.files[filename]

        st.markdown(f"**{filename}**")

        # Show data table
        freq = data['freq']
        Z = data['Z']

        df = pd.DataFrame({
            'index': np.arange(len(freq)),
            'Frequency (Hz)': freq,
            "Z' (Ohm)": np.real(Z),
            "Z'' (Ohm)": np.imag(Z)
        })

        st.dataframe(df, height=250, width="stretch")

        # Temperature input for Arrhenius mode
        if st.session_state.arrhenius_mode:
            temp_value = st.text_input(
                "Temperature (K)",
                value=str(data.get('temperature') or 298.15),
                placeholder="298.15",
                key=f"temp_{filename}"
            )
            try:
                temp_float = float(temp_value)
                if temp_float > 0:
                    st.session_state.files[filename]['temperature'] = temp_float
                else:
                    st.error("Temperature must be positive")
                    st.session_state.files[filename]['temperature'] = 298.15
            except ValueError:
                st.error("Invalid temperature value")
                st.session_state.files[filename]['temperature'] = 298.15
    else:
        st.info("Select a file to view data")


def sidebar_settings():
    """Settings tab in sidebar for plot customization"""
    settings = st.session_state.plot_settings

    # Unit settings
    st.markdown("### Units")
    unit_options = ['Ω', 'kΩ', 'MΩ', 'GΩ']
    current_unit_idx = unit_options.index(settings.get('z_unit', 'Ω'))
    settings['z_unit'] = st.selectbox(
        "Impedance unit",
        unit_options,
        index=current_unit_idx
    )

    st.markdown("---")

    # Font settings
    st.markdown("### Font Size")
    col1, col2 = st.columns(2)
    with col1:
        settings['tick_font_size'] = st.number_input(
            "Tick",
            min_value=6, max_value=20,
            value=settings.get('tick_font_size', 10)
        )
    with col2:
        settings['axis_label_font_size'] = st.number_input(
            "Label",
            min_value=8, max_value=24,
            value=settings.get('axis_label_font_size', 12)
        )

    st.markdown("---")

    # Marker settings
    st.markdown("### Marker")
    col1, col2 = st.columns(2)
    with col1:
        settings['marker_color'] = st.color_picker(
            "Color",
            value=settings.get('marker_color', '#1f77b4')
        )
    with col2:
        symbol_options = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up']
        current_symbol_idx = symbol_options.index(settings.get('marker_symbol', 'circle')) if settings.get('marker_symbol', 'circle') in symbol_options else 0
        settings['marker_symbol'] = st.selectbox(
            "Symbol",
            symbol_options,
            index=current_symbol_idx
        )

    col1, col2 = st.columns(2)
    with col1:
        settings['marker_size'] = st.number_input(
            "Size",
            min_value=1, max_value=20,
            value=settings.get('marker_size', 6)
        )
    with col2:
        settings['marker_alpha'] = st.slider(
            "Alpha",
            min_value=0.0, max_value=1.0,
            value=float(settings.get('marker_alpha', 1.0)),
            step=0.1
        )

    col1, col2 = st.columns(2)
    with col1:
        settings['marker_line_color'] = st.color_picker(
            "Edge color",
            value=settings.get('marker_line_color', '#1f77b4')
        )
    with col2:
        settings['marker_line_width'] = st.number_input(
            "Edge width",
            min_value=0, max_value=5,
            value=settings.get('marker_line_width', 0)
        )

    st.markdown("---")

    # Fit line settings
    st.markdown("### Fit Line")
    col1, col2 = st.columns(2)
    with col1:
        settings['fit_line_color'] = st.color_picker(
            "Color",
            value=settings.get('fit_line_color', '#ff7f0e'),
            key="fit_color"
        )
    with col2:
        settings['fit_line_width'] = st.number_input(
            "Width",
            min_value=1, max_value=5,
            value=settings.get('fit_line_width', 2),
            key="fit_width"
        )

    st.markdown("---")

    # Display settings
    st.markdown("### Display")
    settings['show_zeroline'] = st.checkbox(
        "Show Zero Line",
        value=settings.get('show_zeroline', True)
    )

    st.markdown("---")

    # Legend settings
    st.markdown("### Legend")
    settings['legend_font_size'] = st.number_input(
        "Font Size",
        min_value=6, max_value=20,
        value=settings.get('legend_font_size', 10),
        key="legend_font_size"
    )

    # Plot legend name mode
    plot_legend_options = ['Filename', 'Temperature', 'Manual']
    plot_legend_idx = plot_legend_options.index(settings.get('plot_legend_mode', 'Filename')) if settings.get('plot_legend_mode', 'Filename') in plot_legend_options else 0
    settings['plot_legend_mode'] = st.selectbox(
        "Data Legend",
        plot_legend_options,
        index=plot_legend_idx,
        help="Legend name for measured data: Filename, Temperature (from table, as integer), or Manual input"
    )

    if settings['plot_legend_mode'] == 'Manual':
        settings['plot_legend_manual'] = st.text_input(
            "Data Legend Name",
            value=settings.get('plot_legend_manual', 'measured'),
            placeholder="measured",
            label_visibility="collapsed"
        )

    # Fit legend name mode
    fit_legend_options = ['Filename', 'Manual', 'Hide']
    fit_legend_idx = fit_legend_options.index(settings.get('fit_legend_mode', 'Filename')) if settings.get('fit_legend_mode', 'Filename') in fit_legend_options else 0
    settings['fit_legend_mode'] = st.selectbox(
        "Fit Legend",
        fit_legend_options,
        index=fit_legend_idx,
        help="Legend name for fitted curve: Filename, Manual input, or Hide (no legend)"
    )

    if settings['fit_legend_mode'] == 'Manual':
        settings['fit_legend_manual'] = st.text_input(
            "Fit Legend Name",
            value=settings.get('fit_legend_manual', 'fitted'),
            placeholder="fitted",
            label_visibility="collapsed"
        )

    st.markdown("---")

    # Arrhenius plot settings
    st.markdown("### Arrhenius Plot")
    col1, col2 = st.columns(2)
    with col1:
        settings['arr_marker_color'] = st.color_picker(
            "Marker Color",
            value=settings.get('arr_marker_color', '#1f77b4'),
            key="arr_marker_color"
        )
    with col2:
        arr_symbol_options = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up']
        arr_current_symbol_idx = arr_symbol_options.index(settings.get('arr_marker_symbol', 'circle')) if settings.get('arr_marker_symbol', 'circle') in arr_symbol_options else 0
        settings['arr_marker_symbol'] = st.selectbox(
            "Symbol",
            arr_symbol_options,
            index=arr_current_symbol_idx,
            key="arr_symbol"
        )

    col1, col2 = st.columns(2)
    with col1:
        settings['arr_marker_size'] = st.number_input(
            "Marker Size",
            min_value=1, max_value=20,
            value=settings.get('arr_marker_size', 10),
            key="arr_marker_size"
        )
    with col2:
        settings['arr_line_width'] = st.number_input(
            "Line Width",
            min_value=0, max_value=5,
            value=settings.get('arr_line_width', 2),
            key="arr_line_width"
        )

    col1, col2 = st.columns(2)
    with col1:
        settings['arr_marker_edge_color'] = st.color_picker(
            "Edge Color",
            value=settings.get('arr_marker_edge_color', '#000000'),
            key="arr_marker_edge_color"
        )
    with col2:
        settings['arr_marker_edge_width'] = st.number_input(
            "Edge Width",
            min_value=0, max_value=5,
            value=settings.get('arr_marker_edge_width', 1),
            key="arr_marker_edge_width"
        )

    col1, col2 = st.columns(2)
    with col1:
        settings['arr_line_color'] = st.color_picker(
            "Line Color",
            value=settings.get('arr_line_color', '#5AA4E6'),
            key="arr_line_color"
        )
    with col2:
        settings['arr_show_line'] = st.checkbox(
            "Show Line",
            value=settings.get('arr_show_line', False),
            key="arr_show_line"
        )

    col1, col2 = st.columns(2)
    with col1:
        settings['arr_legend_font_size'] = st.number_input(
            "Legend Font Size",
            min_value=6, max_value=24,
            value=settings.get('arr_legend_font_size', 12),
            key="arr_legend_font_size"
        )

    st.session_state.plot_settings = settings


def main_panel_plots():
    """Main panel for plots - Nyquist, Bode, Arrhenius side by side"""
    if len(st.session_state.files) == 0:
        st.info("Upload EIS data files to begin analysis")
        return

    # Determine which files to plot
    if st.session_state.show_all_data:
        selected_for_plot = list(st.session_state.files.keys())
    elif st.session_state.selected_file:
        selected_for_plot = [st.session_state.selected_file]
    else:
        selected_for_plot = []

    # Get plot settings
    plot_settings = st.session_state.get('plot_settings', {})

    if st.session_state.arrhenius_mode:
        # Arrhenius mode: Show Arrhenius plots
        from components.plots import create_arrhenius_plots

        # Initialize Arrhenius-specific session state
        if 'arr_show_fit' not in st.session_state:
            st.session_state.arr_show_fit = False
        if 'arr_show_fit_legend' not in st.session_state:
            st.session_state.arr_show_fit_legend = True
        if 'arr_fit_range' not in st.session_state:
            st.session_state.arr_fit_range = None
        if 'arr_cycle_mode' not in st.session_state:
            st.session_state.arr_cycle_mode = False
        if 'arr_file_cycles' not in st.session_state:
            st.session_state.arr_file_cycles = {}  # {filename: cycle_name}

        # Build multipoint data with filenames and individual R conductivities
        multipoint_data = []
        all_r_names = set()

        for filename in st.session_state.files:
            data = st.session_state.files[filename]
            if data.get('temperature') and data.get('circuit_params') is not None:
                entry = {
                    'filename': filename,
                    'temperature': data['temperature'],
                    'total_sigma': data.get('total_sigma'),
                    'r_sigmas': data.get('r_sigmas', {})
                }
                multipoint_data.append(entry)
                all_r_names.update(data.get('r_sigmas', {}).keys())

        # Determine available conductivity types
        conductivity_types = ['total'] + sorted(list(all_r_names))

        # Build type labels from r_labels
        type_labels = {'total': 'total'}
        for r_name in all_r_names:
            type_labels[r_name] = st.session_state.r_labels.get(r_name, r_name)

        # Controls BEFORE plot
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1.5, 1.5, 1])

        with ctrl_col1:
            show_legend = st.checkbox(
                "Show Legend",
                value=st.session_state.show_legend,
                key="arr_show_legend_checkbox"
            )
            st.session_state.show_legend = show_legend

            show_fit_legend = st.checkbox(
                "Show Fit Legend",
                value=st.session_state.arr_show_fit_legend,
                key="arr_show_fit_legend_checkbox"
            )
            st.session_state.arr_show_fit_legend = show_fit_legend

        with ctrl_col2:
            # Select which conductivities to plot
            st.caption("Plot")
            selected_types = st.multiselect(
                "Conductivity types to plot",
                conductivity_types,
                default=conductivity_types,
                format_func=lambda x: type_labels.get(x, x),
                label_visibility="collapsed",
                key="arr_plot_types"
            )

        with ctrl_col3:
            # Select which conductivities to fit
            st.caption("Fit targets")
            fit_targets = st.multiselect(
                "Conductivity types to fit",
                selected_types,
                default=[],
                format_func=lambda x: type_labels.get(x, x),
                label_visibility="collapsed",
                key="arr_fit_targets"
            )

        with ctrl_col4:
            # Fit button with help text
            fit_help = """**Arrhenius Fitting**

Performs linear regression on log(σT) vs 1000/T data.

**Formula:**
log(σT) = –Ea / (R · ln10) · (1000/T) + log(A)

**Calculation:**
Ea = –slope × R × ln(10) × 1000

**Unit Conversion:**
- 1 eV = 96,485 J/mol"""

            if st.button("Fit", key="arr_fit_button", help=fit_help, disabled=len(fit_targets) == 0):
                st.session_state.arr_show_fit = True

            if st.session_state.arr_show_fit:
                if st.button("Clear Fit", key="arr_clear_fit_button"):
                    st.session_state.arr_show_fit = False
                    st.rerun()

        # Fitting range slider
        if len(multipoint_data) > 1:
            range_col1, range_col2 = st.columns([1, 4])
            with range_col1:
                st.caption("Fit Range")
            with range_col2:
                n_points = len(multipoint_data)
                current_range = st.session_state.arr_fit_range or (0, n_points - 1)
                current_range = (
                    max(0, min(current_range[0], n_points - 1)),
                    max(0, min(current_range[1], n_points - 1))
                )
                fit_range = st.slider(
                    "Arrhenius Fit Range",
                    min_value=0,
                    max_value=n_points - 1,
                    value=current_range,
                    label_visibility="collapsed",
                    key="arr_fit_range_slider"
                )
                st.session_state.arr_fit_range = fit_range

            # Show Range sliders for each conductivity type
            # Initialize show ranges dict if not exists
            if 'arr_show_ranges' not in st.session_state:
                st.session_state.arr_show_ranges = {}

            show_ranges = {}
            for ctype in selected_types if selected_types else ['total']:
                clabel = type_labels.get(ctype, ctype)
                show_range_col1, show_range_col2 = st.columns([1, 4])
                with show_range_col1:
                    st.caption(f"Show ({clabel})")
                with show_range_col2:
                    # Get current range for this type
                    current_show_range = st.session_state.arr_show_ranges.get(ctype, (0, n_points - 1))
                    current_show_range = (
                        max(0, min(current_show_range[0], n_points - 1)),
                        max(0, min(current_show_range[1], n_points - 1))
                    )
                    type_show_range = st.slider(
                        f"Show Range ({clabel})",
                        min_value=0,
                        max_value=n_points - 1,
                        value=current_show_range,
                        label_visibility="collapsed",
                        key=f"arr_show_range_{ctype}"
                    )
                    st.session_state.arr_show_ranges[ctype] = type_show_range
                    show_ranges[ctype] = type_show_range
        else:
            fit_range = None
            show_ranges = {}

        # Cycle mode settings
        cycle_col1, cycle_col2 = st.columns([1, 4])
        with cycle_col1:
            cycle_mode = st.checkbox(
                "Cycle Mode",
                value=st.session_state.arr_cycle_mode,
                key="arr_cycle_mode_checkbox",
                help="Separate data into heating/cooling or 1st/2nd/3rd cycles"
            )
            st.session_state.arr_cycle_mode = cycle_mode

        if cycle_mode and len(multipoint_data) > 0:
            with cycle_col2:
                # Define available cycle options
                cycle_options = ['1st', '2nd', '3rd', '4th', 'heating', 'cooling']

                # Create expander for cycle assignments
                with st.expander("Cycle Assignments", expanded=False):
                    # Batch assignment buttons
                    batch_col1, batch_col2, batch_col3 = st.columns(3)
                    with batch_col1:
                        selected_cycle_for_all = st.selectbox(
                            "Set all to:",
                            cycle_options,
                            key="arr_cycle_batch_select"
                        )
                    with batch_col2:
                        if st.button("Apply to All", key="arr_cycle_apply_all"):
                            for entry in multipoint_data:
                                st.session_state.arr_file_cycles[entry['filename']] = selected_cycle_for_all
                            st.rerun()
                    with batch_col3:
                        if st.button("Auto Assign", key="arr_cycle_auto_assign",
                                    help="Auto-assign based on temperature order (ascending=heating, descending=cooling)"):
                            # Sort by temperature to detect heating/cooling
                            temps = [(i, entry['temperature']) for i, entry in enumerate(multipoint_data)]
                            if len(temps) > 1:
                                # Check if overall trend is increasing or decreasing
                                is_increasing = temps[-1][1] > temps[0][1]
                                for i, entry in enumerate(multipoint_data):
                                    st.session_state.arr_file_cycles[entry['filename']] = 'heating' if is_increasing else 'cooling'
                            st.rerun()

                    st.markdown("---")

                    # Individual file cycle assignments
                    for i, entry in enumerate(multipoint_data):
                        filename = entry['filename']
                        temp = entry['temperature']
                        current_cycle = st.session_state.arr_file_cycles.get(filename, '1st')

                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.caption(f"{filename} ({temp:.1f} K)")
                        with col_b:
                            new_cycle = st.selectbox(
                                f"Cycle for {filename}",
                                cycle_options,
                                index=cycle_options.index(current_cycle) if current_cycle in cycle_options else 0,
                                label_visibility="collapsed",
                                key=f"arr_cycle_{i}"
                            )
                            st.session_state.arr_file_cycles[filename] = new_cycle

            # Add cycle info to multipoint_data
            for entry in multipoint_data:
                entry['cycle'] = st.session_state.arr_file_cycles.get(entry['filename'], '1st')

        # Create plots with fitting
        fig_sigma_T, fig_sigma, fit_results = create_arrhenius_plots(
            multipoint_data,
            conductivity_types=selected_types if selected_types else ['total'],
            type_labels=type_labels,
            show_legend=st.session_state.show_legend,
            plot_settings=plot_settings,
            fit_range=fit_range if st.session_state.arr_show_fit else None,
            show_fit=st.session_state.arr_show_fit,
            fit_targets=fit_targets if st.session_state.arr_show_fit else [],
            show_ranges=show_ranges,
            show_fit_legend=st.session_state.arr_show_fit_legend
        )

        # Display two plots side by side
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            st.plotly_chart(fig_sigma_T, use_container_width=True, key="arrhenius_sigma_T")
        with plot_col2:
            st.plotly_chart(fig_sigma, use_container_width=True, key="arrhenius_sigma")

        # Display Ea results as table below plots
        if st.session_state.arr_show_fit and len(fit_results) > 0:
            # Get temperature range from fit
            fit_range_used = fit_range if fit_range else (0, len(multipoint_data) - 1)
            temps_in_range = []
            for idx, data in enumerate(multipoint_data):
                if fit_range_used[0] <= idx <= fit_range_used[1]:
                    temps_in_range.append(data.get('temperature', 0))

            if temps_in_range:
                t_min = min(temps_in_range)
                t_max = max(temps_in_range)
                t_range_str = f"{t_min:.0f}–{t_max:.0f}"
            else:
                t_range_str = "–"

            # Get sample name
            sample_name = st.session_state.sample_info.get('name', '') or 'Sample'

            # Format Ea with error using parenthesis notation
            def format_value_with_error(value, error):
                if error is None or error == 0:
                    return f"{value:.1f}"

                if error >= 1:
                    err_rounded = round(error)
                    first_digit = int(str(err_rounded)[0])
                    if first_digit == 1 and error >= 10:
                        err_str = str(err_rounded)
                        val_str = f"{value:.0f}"
                    elif first_digit == 1:
                        err_rounded = round(error, 1)
                        err_str = f"{err_rounded:.1f}".replace('.', '').lstrip('0') or '0'
                        val_str = f"{value:.1f}"
                    else:
                        err_str = str(err_rounded)
                        val_str = f"{value:.0f}"
                else:
                    import math
                    decimal_places = -int(math.floor(math.log10(error)))
                    first_sig = int(error * (10 ** decimal_places))
                    if first_sig == 1:
                        decimal_places += 1
                    err_rounded = round(error, decimal_places)
                    val_rounded = round(value, decimal_places)
                    err_str = str(int(round(err_rounded * (10 ** decimal_places))))
                    val_str = f"{val_rounded:.{decimal_places}f}"

                return f"{val_str}({err_str})"

            # Create Ea table with rows for each conductivity type
            ea_rows = []
            for cond_type, result in fit_results.items():
                label = result.get('label', cond_type)
                ea_kj_str = format_value_with_error(
                    result['Ea_kJ_mol'],
                    result.get('Ea_kJ_mol_err')
                )
                ea_ev_str = format_value_with_error(
                    result['Ea_eV'],
                    result.get('Ea_eV_err')
                )
                ea_rows.append({
                    'Sample': sample_name,
                    'Type': label,
                    'Ea / kJ mol⁻¹': ea_kj_str,
                    'Ea / eV': ea_ev_str,
                    'T range / K': t_range_str
                })

            if ea_rows:
                ea_table = pd.DataFrame(ea_rows)
                st.dataframe(ea_table, hide_index=True, use_container_width=True)
    else:
        # Normal mode: Show Nyquist and Bode plots

        # Controls BEFORE plots (so values are updated before plotting)
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 2, 1.5, 1])

        with ctrl_col1:
            show_fit = st.checkbox(
                "Show Fitted",
                value=st.session_state.show_fit,
                key="show_fit_checkbox"
            )
            st.session_state.show_fit = show_fit

            show_legend = st.checkbox(
                "Show Legend",
                value=st.session_state.show_legend,
                key="show_legend_checkbox"
            )
            st.session_state.show_legend = show_legend

            highlight_freq = st.checkbox(
                "Highlight Freq",
                value=st.session_state.highlight_freq,
                help="Highlight 10^n Hz points",
                key="highlight_freq_checkbox"
            )
            st.session_state.highlight_freq = highlight_freq

        # Fitting range slider
        with ctrl_col2:
            st.caption("Fitting Range (index)")
            freq_range = st.session_state.freq_range
            if st.session_state.selected_file and st.session_state.selected_file in st.session_state.files:
                data = st.session_state.files[st.session_state.selected_file]
                freq_data = data['freq']
                n_points = len(freq_data)

                if n_points > 1:
                    # Get current range or default to 0-70
                    current_range = st.session_state.freq_range or (0, min(70, n_points - 1))
                    # Clamp to valid range
                    current_range = (
                        max(0, min(current_range[0], n_points - 1)),
                        max(0, min(current_range[1], n_points - 1))
                    )

                    freq_range = st.slider(
                        "range",
                        min_value=0,
                        max_value=n_points - 1,
                        value=current_range,
                        label_visibility="collapsed",
                        key="freq_range_slider"
                    )
                    st.session_state.freq_range = freq_range

        # Delete points
        with ctrl_col3:
            st.caption("Delete Points (index)")
            delete_input = st.text_input(
                "delete_points",
                value=",".join(map(str, st.session_state.deleted_points)) if st.session_state.deleted_points else "",
                placeholder="e.g., 0,5,10",
                label_visibility="collapsed",
                key="delete_points_input"
            )
            # Parse delete points
            deleted_points = []
            if delete_input:
                try:
                    deleted_points = [int(x.strip()) for x in delete_input.split(",") if x.strip()]
                    st.session_state.deleted_points = deleted_points
                except ValueError:
                    deleted_points = st.session_state.deleted_points
            else:
                st.session_state.deleted_points = []

        with ctrl_col4:
            st.caption("")  # Spacer
            if st.button("Reset Delete", width="stretch"):
                st.session_state.deleted_points = []
                # Clear the text input widget value
                st.session_state["delete_points_input"] = ""
                st.rerun()

        # Get current values for plotting
        deleted_points = st.session_state.get('deleted_points', [])

        # Now render plots with updated values
        col1, col2 = st.columns(2)

        with col1:
            fig_nyquist = create_nyquist_plot(
                st.session_state.files, selected_for_plot,
                show_fit,
                show_legend,
                highlight_freq,
                plot_settings,
                freq_range,
                deleted_points
            )
            st.plotly_chart(fig_nyquist, use_container_width=True, key="nyquist")

        with col2:
            fig_bode = create_bode_plot(
                st.session_state.files, selected_for_plot,
                show_fit,
                show_legend,
                freq_range,
                plot_settings,
                deleted_points
            )
            st.plotly_chart(fig_bode, use_container_width=True, key="bode")


def circuit_analysis_panel():
    """Circuit analysis panel with new layout"""
    if not st.session_state.selected_file or st.session_state.selected_file not in st.session_state.files:
        st.info("Select a file from the sidebar to perform circuit analysis")
        return

    filename = st.session_state.selected_file
    data = st.session_state.files[filename]

    # Preset circuit models with display name -> (circuit_string, initial_guess)
    # Order: Custom, common solid electrolyte models, then others
    from collections import OrderedDict
    preset_circuits = OrderedDict([
        # Primary choices
        ("Custom", ("", None)),
        ("(R/CPE)-CPE", ("p(R1,CPE1)-CPE2", [1e5, 1e-9, 0.9, 1e-7, 0.7])),
        ("(R/CPE)-(R/CPE)-CPE", ("p(R1,CPE1)-p(R2,CPE2)-CPE3", [1e4, 1e-11, 0.95, 1e5, 1e-9, 0.9, 1e-6, 0.5])),
        ("(R/CPE)-(R/CPE)-(R/CPE)", ("p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)", [1e3, 1e-12, 0.98, 1e4, 1e-11, 0.95, 1e5, 1e-9, 0.9])),
        ("R-CPE", ("R1-CPE1", [1e4, 1e-9, 0.9])),
        ("R-L-CPE", ("R1-L1-CPE1", [1e2, 1e-6, 1e-9, 0.9])),
        # --- Other models ---
        ("(R/CPE)", ("p(R1,CPE1)", [1e4, 1e-9, 0.9])),
        ("(R/CPE)-(R/CPE)", ("p(R1,CPE1)-p(R2,CPE2)", [1e4, 1e-11, 0.95, 1e5, 1e-9, 0.9])),
        ("R-(R/CPE)", ("R1-p(R2,CPE1)", [1e2, 1e5, 1e-9, 0.9])),
        ("R-(R/CPE)-CPE", ("R1-p(R2,CPE1)-CPE2", [1e2, 1e5, 1e-9, 0.9, 1e-7, 0.7])),
        ("R-(R/CPE)-(R/CPE)", ("R1-p(R2,CPE1)-p(R3,CPE2)", [1e2, 1e4, 1e-11, 0.95, 1e5, 1e-9, 0.9])),
        ("R-(R/CPE)-(R/CPE)-CPE", ("R1-p(R2,CPE1)-p(R3,CPE2)-CPE3", [1e2, 1e4, 1e-11, 0.95, 1e5, 1e-9, 0.9, 1e-6, 0.5])),
        # Capacitor models
        ("R", ("R1", [1e4])),
        ("R-C", ("R1-C1", [1e4, 1e-9])),
        ("(R/C)", ("p(R1,C1)", [1e4, 1e-9])),
        ("R-(R/C)", ("R1-p(R2,C1)", [1e2, 1e5, 1e-10])),
        ("(R/C)-(R/C)", ("p(R1,C1)-p(R2,C2)", [1e4, 1e-11, 1e5, 1e-9])),
        # Warburg models
        ("R-W", ("R1-W1", [1e4, 1e-3])),
        ("(R/CPE)-W", ("p(R1,CPE1)-W1", [1e5, 1e-9, 0.9, 1e-3])),
        ("R-(R/CPE)-W", ("R1-p(R2,CPE1)-W1", [1e2, 1e5, 1e-9, 0.9, 1e-3])),
        ("(R/CPE)-(R/CPE)-W", ("p(R1,CPE1)-p(R2,CPE2)-W1", [1e4, 1e-11, 0.95, 1e5, 1e-9, 0.9, 1e-3])),
        # Randles circuit: Rs - ((Rct - W) // CPE)
        ("Randles: R-((R-W)/CPE)", ("R1-p(R2-W1,CPE1)", [1e2, 1e5, 1e-3, 1e-9, 0.9])),
    ])
    # Extract just circuit strings for backward compatibility
    preset_circuit_strings = {k: v[0] for k, v in preset_circuits.items()}
    preset_initial_guesses = {k: v[1] for k, v in preset_circuits.items()}

    # Top row: Circuit preset, Circuit String, and Weight Method
    col1, col2, col3 = st.columns([1, 2, 1])

    # Track previous preset to detect changes
    if 'prev_circuit_preset' not in st.session_state:
        st.session_state.prev_circuit_preset = "Custom"

    with col1:
        ec_help = """**Preset Equivalent Circuits:**

**Simple models:**
- **R**, **R-C**, **R-CPE**: Series elements
- **(R/C)**, **(R/CPE)**: Parallel elements

**Two-element models:**
- **R-(R/CPE)**: Bulk + grain boundary
- **(R/CPE)-(R/CPE)**: Two semicircles

**Three-element (solid electrolytes):**
- **(R/CPE)-(R/CPE)-CPE**: Common for ionic conductors
- **R-(R/CPE)-(R/CPE)-CPE**: With contact resistance

**Warburg (diffusion):**
- **R-W**, **(R/CPE)-W**, **Randles**

Select preset or "Custom" for manual input."""
        preset_choice = st.selectbox(
            "Equivalent Circuit",
            list(preset_circuits.keys()),
            index=0,
            key="circuit_preset",
            help=ec_help
        )

    # If preset changed, update the text input widget value and initial guess
    if preset_choice != st.session_state.prev_circuit_preset:
        st.session_state.prev_circuit_preset = preset_choice
        if preset_choice != "Custom" and preset_circuit_strings[preset_choice]:
            st.session_state["circuit_model_input"] = preset_circuit_strings[preset_choice]
            # Store initial guess for this preset
            if preset_initial_guesses[preset_choice]:
                st.session_state["preset_initial_guess"] = preset_initial_guesses[preset_choice]
            # Reset param_fixed state when circuit changes
            st.session_state.param_fixed = {}
            st.rerun()

    with col2:
        # Get default circuit model
        default_circuit = data.get('circuit_model') or 'p(R1,CPE1)-p(R2,CPE2)-CPE3'

        circuit_model = st.text_input(
            "Manual Input",
            value=default_circuit,
            help=CIRCUIT_MODEL_HELP,
            placeholder="e.g., R1-p(R2,CPE1)-CPE2",
            key="circuit_model_input"
        )

    with col3:
        weight_options = [None, "proportional", "modulus", "squared_modulus"]
        weight_labels = ["None", "Proportional", "Modulus", "Squared Modulus"]
        current_idx = 0  # default to None
        weight_method = st.selectbox(
            "Weighting",
            weight_options,
            index=current_idx,
            format_func=lambda x: weight_labels[weight_options.index(x)] if x in weight_options else str(x),
            help=WEIGHT_METHOD_HELP
        )

    # Button row: Fit Circuit, Auto Fit, Batch Fit, Auto-Batch Fit, Reset
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)

    with btn_col1:
        fit_clicked = st.button("Fit Circuit", width="stretch", type="primary")

    with btn_col2:
        auto_fit_help = """**Auto Fit**
Uses black-box optimization (Optuna) to automatically find the best circuit parameters.

- Tries multiple initial guesses
- Stops early when convergence criterion is met

**Note:** Requires 'optuna' package."""
        auto_fit_clicked = st.button("Auto Fit", width="stretch", help=auto_fit_help)

    with btn_col3:
        batch_fit_help = """**Batch Fit**
Fits selected files using current initial values.

- Select files in Batch Fit Settings
- Uses current parameters as initial guess
- Propagates fit results to next file"""
        batch_fit_clicked = st.button("Batch Fit", width="stretch", help=batch_fit_help)

    with btn_col4:
        auto_batch_help = """**Auto-Batch Fit**
Combines Auto Fit + Batch Fit.

- Uses Optuna optimization for each file
- Select files in Batch Fit Settings
- Best for initial fitting of many files"""
        auto_batch_clicked = st.button("Auto-Batch", width="stretch", help=auto_batch_help)

    with btn_col5:
        # Reset to initial button - restores default initial values
        reset_clicked = st.button("Reset", width="stretch",
                                   help="Reset parameters to default initial values")

    # Handle reset button - restore default initial values
    if reset_clicked:
        # Clear current initial guess to force regeneration of defaults
        if 'global_initial_guess' in st.session_state:
            del st.session_state['global_initial_guess']
        st.session_state.files[filename]['initial_guess'] = None
        # Clear widget values
        from impedance.models.circuits.fitting import calculateCircuitLength
        try:
            n_params = calculateCircuitLength(circuit_model)
            for i in range(n_params):
                widget_key = f"init_{i}"
                if widget_key in st.session_state:
                    del st.session_state[widget_key]
        except:
            pass
        st.rerun()

    # Fit Settings (expandable)
    with st.expander("Fit Settings", expanded=False):
        # Initialize fit settings in session state
        if 'fit_settings' not in st.session_state:
            st.session_state.fit_settings = {
                'maxfev': 10000,
                'ftol': 1e-10,
                'xtol': 1e-10,
                'keep_better': True,
                'global_opt': False
            }

        fit_settings = st.session_state.fit_settings

        # Row 1: Convergence settings
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            fit_settings['maxfev'] = st.number_input(
                "Max evaluations",
                min_value=1000, max_value=100000, value=fit_settings['maxfev'], step=1000,
                help=FIT_SETTINGS_HELP['maxfev']
            )
        with col_f2:
            ftol_exp = st.number_input(
                "ftol (10^x)",
                min_value=-15, max_value=-5, value=int(np.log10(fit_settings['ftol'])), step=1,
                help=FIT_SETTINGS_HELP['ftol']
            )
            fit_settings['ftol'] = 10 ** ftol_exp
        with col_f3:
            xtol_exp = st.number_input(
                "xtol (10^x)",
                min_value=-15, max_value=-5, value=int(np.log10(fit_settings['xtol'])), step=1,
                help=FIT_SETTINGS_HELP['xtol']
            )
            fit_settings['xtol'] = 10 ** xtol_exp

        # Row 2: Options
        col_f4, col_f5 = st.columns(2)
        with col_f4:
            fit_settings['keep_better'] = st.checkbox(
                "Keep better result",
                value=fit_settings['keep_better'],
                help=FIT_SETTINGS_HELP['keep_better']
            )
        with col_f5:
            fit_settings['global_opt'] = st.checkbox(
                "Global optimization",
                value=fit_settings['global_opt'],
                help=FIT_SETTINGS_HELP['global_opt']
            )

        st.session_state.fit_settings = fit_settings

    # Auto Fit Settings (expandable)
    with st.expander("Auto Fit Settings", expanded=False):
        # Initialize auto_fit settings in session state
        if 'auto_fit_settings' not in st.session_state:
            st.session_state.auto_fit_settings = {
                'n_trials': 30,
                'timeout': 2,
                'early_stop_rmspe': 3.0,
                'log_step': 0.5,
                'r_min': 1e0,
                'r_max': 1e8,
                'cpe_q_min': 1e-12,
                'cpe_q_max': 1e-4,
                'use_current_model': True,
                'model_list': ['p(R1,CPE1)-CPE2', 'p(R1,CPE1)-p(R2,CPE2)-CPE3'],
                # Individual R ranges (exponent values)
                'r1_range': (0, 8),
                'r2_range': (0, 8),
                'r3_range': (0, 8),
                # Individual CPE Q ranges (exponent values)
                'cpe1_q_range': (-12, -9),
                'cpe2_q_range': (-10, -8),
                'cpe3_q_range': (-7, -5)
            }

        settings = st.session_state.auto_fit_settings

        # Row 1: Trials, Timeout, Convergence, Log Step
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            settings['n_trials'] = st.number_input(
                "Max trials",
                min_value=10, max_value=1000, value=settings['n_trials'], step=10,
                help=AUTO_FIT_SETTINGS_HELP['n_trials']
            )
        with col_s2:
            settings['timeout'] = st.number_input(
                "Timeout (sec)",
                min_value=5, max_value=300, value=settings['timeout'], step=5,
                help=AUTO_FIT_SETTINGS_HELP['timeout']
            )
        with col_s3:
            settings['early_stop_rmspe'] = st.number_input(
                "Convergence (%)",
                min_value=0.1, max_value=20.0, value=settings['early_stop_rmspe'], step=0.5,
                help=AUTO_FIT_SETTINGS_HELP['early_stop_rmspe']
            )
        with col_s4:
            settings['log_step'] = st.number_input(
                "Log step",
                min_value=0.0, max_value=2.0, value=settings.get('log_step', 0.5), step=0.1,
                format="%.1f",
                help=AUTO_FIT_SETTINGS_HELP['log_step']
            )

        # Determine which elements are in the current model
        def get_model_elements(model_str):
            """Extract R and CPE element names from circuit model string."""
            import re
            r_elements = set(re.findall(r'R\d+', model_str))
            cpe_elements = set(re.findall(r'CPE\d+', model_str))
            return r_elements, cpe_elements

        # Get elements from current model or all models if using multiple
        if settings.get('use_current_model', True):
            models_to_check = [circuit_model] if circuit_model else ['p(R1,CPE1)-p(R2,CPE2)-CPE3']
        else:
            models_to_check = settings.get('model_list', ['p(R1,CPE1)-p(R2,CPE2)-CPE3'])

        all_r_elements = set()
        all_cpe_elements = set()
        for model in models_to_check:
            r_elems, cpe_elems = get_model_elements(model)
            all_r_elements.update(r_elems)
            all_cpe_elements.update(cpe_elems)

        # Sort elements for consistent display
        r_elements_sorted = sorted(all_r_elements, key=lambda x: int(x[1:]))
        cpe_elements_sorted = sorted(all_cpe_elements, key=lambda x: int(x[3:]))

        # Individual R range sliders (only show relevant elements)
        if r_elements_sorted:
            st.markdown("**R range (10^x Ω)**")
            for r_elem in r_elements_sorted:
                r_key = f'{r_elem.lower()}_range'
                # Initialize if not exists
                if r_key not in settings:
                    settings[r_key] = (0, 8)
                current_range = settings[r_key]
                settings[r_key] = st.slider(
                    f"{r_elem}",
                    min_value=-3, max_value=12,
                    value=(int(current_range[0]), int(current_range[1])),
                    step=1,
                    key=f"auto_fit_{r_key}",
                    help=f"Search range for {r_elem} (10^min to 10^max Ω)"
                )

        # Individual CPE Q range sliders (only show relevant elements)
        if cpe_elements_sorted:
            st.markdown("**CPE Q range (10^x F·s^(α-1))**")
            for cpe_elem in cpe_elements_sorted:
                cpe_key = f'{cpe_elem.lower()}_q_range'
                # Initialize with default values based on element number
                if cpe_key not in settings:
                    elem_num = int(cpe_elem[3:])
                    if elem_num == 1:
                        settings[cpe_key] = (-12, -9)
                    elif elem_num == 2:
                        settings[cpe_key] = (-10, -8)
                    else:  # CPE3 and beyond
                        settings[cpe_key] = (-7, -5)
                current_range = settings[cpe_key]
                settings[cpe_key] = st.slider(
                    f"{cpe_elem}_Q",
                    min_value=-15, max_value=0,
                    value=(int(current_range[0]), int(current_range[1])),
                    step=1,
                    key=f"auto_fit_{cpe_key}",
                    help=f"Search range for {cpe_elem} Q (10^min to 10^max F·s^(α-1))"
                )

        # Row 4: Circuit model selection
        st.markdown("**Circuit model**")
        settings['use_current_model'] = st.checkbox(
            "Use current circuit model only",
            value=settings['use_current_model'],
            help=AUTO_FIT_SETTINGS_HELP['use_current_model']
        )

        if not settings['use_current_model']:
            # Multi-select for circuit models
            available_models = [
                'p(R1,CPE1)-CPE2',
                'p(R1,CPE1)-p(R2,CPE2)-CPE3',
                'R1-p(R2,CPE1)-CPE2',
                'R1-p(R2,CPE1)-p(R3,CPE2)-CPE3',
                'p(R1,CPE1)',
                'p(R1,CPE1)-p(R2,CPE2)'
            ]
            settings['model_list'] = st.multiselect(
                "Models to try",
                available_models,
                default=settings['model_list'],
                help=AUTO_FIT_SETTINGS_HELP['model_list']
            )

        st.session_state.auto_fit_settings = settings

    # Batch Fit Settings (expandable)
    with st.expander("Batch Fit Settings", expanded=False):
        # Initialize batch_fit settings in session state
        if 'batch_fit_settings' not in st.session_state:
            st.session_state.batch_fit_settings = {
                'use_previous_result': True,
                'stop_on_error': False,
                'rmspe_threshold': 10.0
            }

        # Initialize batch file selection
        if 'batch_selected_files' not in st.session_state:
            st.session_state.batch_selected_files = set(st.session_state.files.keys())

        batch_settings = st.session_state.batch_fit_settings

        # Row 1: Options
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            batch_settings['use_previous_result'] = st.checkbox(
                "Use previous fit result as initial guess",
                value=batch_settings['use_previous_result'],
                help=BATCH_FIT_SETTINGS_HELP['use_previous_result']
            )
        with col_b2:
            batch_settings['stop_on_error'] = st.checkbox(
                "Stop on fitting error",
                value=batch_settings['stop_on_error'],
                help=BATCH_FIT_SETTINGS_HELP['stop_on_error']
            )

        # Row 2: RMSPE threshold
        batch_settings['rmspe_threshold'] = st.number_input(
            "RMSPE threshold (%)",
            min_value=1.0, max_value=50.0, value=batch_settings['rmspe_threshold'], step=1.0,
            help=BATCH_FIT_SETTINGS_HELP['rmspe_threshold']
        )

        st.session_state.batch_fit_settings = batch_settings

        # File selection section
        st.markdown("---")
        st.markdown("**File Selection**")

        # Select All / Deselect All / Select Bad Fits buttons
        sel_col1, sel_col2, sel_col3, sel_col4 = st.columns([1, 1, 1, 1])
        with sel_col1:
            if st.button("Select All", key="batch_select_all"):
                st.session_state.batch_selected_files = set(st.session_state.files.keys())
                st.rerun()
        with sel_col2:
            if st.button("Deselect All", key="batch_deselect_all"):
                st.session_state.batch_selected_files = set()
                st.rerun()
        with sel_col3:
            # Select files with RMSPE >= threshold or not fitted
            if st.button("Select Bad Fits", key="batch_select_bad"):
                threshold = batch_settings.get('rmspe_threshold', 10.0) / 100
                bad_files = set()
                for fname, fdata in st.session_state.files.items():
                    rmspe = fdata.get('rmspe')
                    if rmspe is None or rmspe >= threshold:
                        bad_files.add(fname)
                st.session_state.batch_selected_files = bad_files
                st.rerun()
        with sel_col4:
            n_selected = len(st.session_state.batch_selected_files)
            n_total = len(st.session_state.files)
            st.caption(f"Selected: {n_selected}/{n_total}")

        # File list with checkboxes and RMSPE
        all_files = list(st.session_state.files.keys())

        # Sync checkbox widget keys with session state
        for fname in all_files:
            checkbox_key = f"batch_file_{fname}"
            is_selected = fname in st.session_state.batch_selected_files
            # Initialize or update the widget key state
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = is_selected
            elif st.session_state[checkbox_key] != is_selected:
                # Select All / Deselect All was clicked, update widget state
                st.session_state[checkbox_key] = is_selected

        for fname in all_files:
            fdata = st.session_state.files[fname]
            rmspe = fdata.get('rmspe')

            # Build label with RMSPE if available
            if rmspe is not None:
                rmspe_pct = rmspe * 100
                if rmspe_pct < 3:
                    rmspe_label = f" (RMSPE: {rmspe_pct:.1f}%)"
                elif rmspe_pct < 10:
                    rmspe_label = f" (RMSPE: {rmspe_pct:.1f}%)"
                else:
                    rmspe_label = f" (RMSPE: {rmspe_pct:.1f}% ⚠)"
            else:
                rmspe_label = " (not fitted)"

            # Checkbox for file selection (uses on_change callback)
            checkbox_key = f"batch_file_{fname}"

            def on_checkbox_change(file_name=fname):
                key = f"batch_file_{file_name}"
                if st.session_state[key]:
                    st.session_state.batch_selected_files.add(file_name)
                else:
                    st.session_state.batch_selected_files.discard(file_name)

            st.checkbox(
                f"{fname}{rmspe_label}",
                key=checkbox_key,
                on_change=on_checkbox_change
            )

    # Handle fit button
    if fit_clicked:
        with st.spinner("Fitting..."):
            try:
                freq = data['freq']
                Z = data['Z']

                # Get fit settings
                fit_settings = st.session_state.get('fit_settings', {})
                maxfev = fit_settings.get('maxfev', 10000)
                ftol = fit_settings.get('ftol', 1e-10)
                xtol = fit_settings.get('xtol', 1e-10)
                keep_better = fit_settings.get('keep_better', True)
                global_opt = fit_settings.get('global_opt', False)

                # Store existing RMSPE for keep_better comparison
                existing_rmspe = data.get('rmspe')

                # Apply fitting range if set
                freq_range = st.session_state.freq_range
                if freq_range:
                    start_idx, end_idx = freq_range
                    freq_fit = freq[start_idx:end_idx + 1]
                    Z_fit_data = Z[start_idx:end_idx + 1]
                else:
                    freq_fit = freq
                    Z_fit_data = Z

                # Calculate required number of parameters for circuit
                from impedance.models.circuits.fitting import calculateCircuitLength
                n_params = calculateCircuitLength(circuit_model)

                # Get initial guess from table or generate default
                initial_guess = data.get('initial_guess')

                # If initial guess doesn't match circuit length, use preset or generate new default
                if initial_guess is None or len(initial_guess) != n_params:
                    # Check if we have preset initial guess stored
                    preset_guess = st.session_state.get('preset_initial_guess')
                    if preset_guess and len(preset_guess) == n_params:
                        initial_guess = preset_guess
                    else:
                        # Generate default values based on circuit element names
                        initial_guess = []
                        try:
                            temp_c = CustomCircuit(circuit_model, initial_guess=[1.0] * n_params)
                            pnames, _ = temp_c.get_param_names()
                            for pname in pnames:
                                if 'CPE' in pname and '_1' in pname:  # CPE alpha
                                    initial_guess.append(0.9)
                                elif 'CPE' in pname and '_0' in pname:  # CPE Q
                                    initial_guess.append(1e-9)
                                elif 'W' in pname:  # Warburg
                                    initial_guess.append(1e-3)
                                elif 'C' in pname and 'CPE' not in pname:  # Capacitor
                                    initial_guess.append(1e-9)
                                else:  # R element
                                    initial_guess.append(1e4)
                        except:
                            # Fallback to simple heuristic
                            for i in range(n_params):
                                if i % 3 == 0:
                                    initial_guess.append(1e4)
                                elif i % 3 == 1:
                                    initial_guess.append(1e-9)
                                else:
                                    initial_guess.append(0.9)

                # Build constants dict for fixed parameters
                constants = {}
                variable_initial_guess = []
                fixed_indices = []

                # Get param names for this circuit
                try:
                    temp_circuit = CustomCircuit(circuit_model, initial_guess=initial_guess)
                    temp_param_names, _ = temp_circuit.get_param_names()

                    # Check param_fixed state
                    param_fixed = st.session_state.get('param_fixed', {})

                    for i, pname in enumerate(temp_param_names):
                        is_fixed = param_fixed.get(f"fixed_{i}", False)
                        if is_fixed:
                            # Fixed parameter - add to constants
                            constants[pname] = initial_guess[i]
                            fixed_indices.append(i)
                        else:
                            # Variable parameter - add to initial guess for fitting
                            variable_initial_guess.append(initial_guess[i])

                except Exception as e:
                    variable_initial_guess = initial_guess

                # Check if all parameters are fixed (prediction only mode)
                if len(variable_initial_guess) == 0:
                    # All parameters fixed - prediction only, no fitting
                    popt = np.array(initial_guess)
                    perror = np.zeros(n_params)
                    st.info("All parameters are fixed. Prediction only (no fitting).")
                elif len(constants) == 0:
                    # No fixed parameters - normal fitting
                    popt_variable, perror_variable = circuit_fit(
                        freq_fit, Z_fit_data,
                        circuit_model,
                        initial_guess,
                        constants={},
                        weight_method=weight_method,
                        global_opt=global_opt,
                        maxfev=maxfev,
                        ftol=ftol,
                        xtol=xtol
                    )
                    popt = popt_variable
                    perror = perror_variable
                else:
                    # Some fixed, some variable - fit with constants
                    popt_variable, perror_variable = circuit_fit(
                        freq_fit, Z_fit_data,
                        circuit_model,
                        variable_initial_guess,
                        constants=constants,
                        weight_method=weight_method,
                        global_opt=global_opt,
                        maxfev=maxfev,
                        ftol=ftol,
                        xtol=xtol
                    )

                    # Reconstruct full parameter arrays (including fixed parameters)
                    popt = []
                    perror = []
                    var_idx = 0
                    for i in range(n_params):
                        if i in fixed_indices:
                            # Fixed parameter - use initial value
                            popt.append(initial_guess[i])
                            perror.append(0.0)  # No error for fixed params
                        else:
                            # Variable parameter - use fitted value
                            popt.append(popt_variable[var_idx])
                            perror.append(perror_variable[var_idx] if perror_variable is not None else 0.0)
                            var_idx += 1
                    popt = np.array(popt)
                    perror = np.array(perror)

                # Create CustomCircuit for prediction
                # For prediction, we need full parameters but pass constants separately
                circuit = CustomCircuit(circuit_model, initial_guess=list(popt), constants={})
                circuit.parameters_ = popt
                circuit.conf_ = perror

                # Predict fitted impedance for full range
                Z_fit = circuit.predict(freq)

                # Calculate RMSPE
                rmspe = calc_rmspe(Z_fit_data, circuit.predict(freq_fit))

                # Keep Better logic: compare with existing result
                should_update = True
                if keep_better and existing_rmspe is not None:
                    if rmspe >= existing_rmspe:
                        # New fit is worse or equal, keep existing result
                        st.warning(f"New fit (RMSPE: {rmspe*100:.2f}%) is not better than existing (RMSPE: {existing_rmspe*100:.2f}%). Keeping existing result.")
                        should_update = False
                    else:
                        st.info(f"New fit (RMSPE: {rmspe*100:.2f}%) is better than existing (RMSPE: {existing_rmspe*100:.2f}%). Updating result.")

                if should_update:
                    # Store results
                    st.session_state.files[filename]['Z_fit'] = Z_fit
                    st.session_state.files[filename]['circuit_model'] = circuit_model
                    st.session_state.files[filename]['circuit_params'] = popt
                    st.session_state.files[filename]['circuit_conf'] = perror
                    st.session_state.files[filename]['circuit_object'] = circuit
                    st.session_state.files[filename]['rmspe'] = rmspe

                    # Get param names for sorting
                    param_names, _ = circuit.get_param_names()

                    # Sort by effective capacitance (R1=smallest Ceff, R2=larger Ceff, etc.)
                    sorted_result = sort_ecm_by_cap(popt, perror, param_names)
                    effective_caps = sorted_result.get('effective_caps', {})

                    # Build sorted parameter arrays from sorted_result
                    sorted_popt = []
                    sorted_perror = []
                    for name in param_names:
                        if name in sorted_result:
                            sorted_popt.append(sorted_result[name])
                            sorted_perror.append(sorted_result.get(f'{name}_error', 0.0))
                        else:
                            # Fallback to original order if not in sorted result
                            idx = param_names.index(name)
                            sorted_popt.append(popt[idx])
                            sorted_perror.append(perror[idx] if perror is not None else 0.0)

                    sorted_popt = np.array(sorted_popt)
                    sorted_perror = np.array(sorted_perror)

                    # Update circuit with sorted parameters
                    circuit.parameters_ = sorted_popt
                    circuit.conf_ = sorted_perror

                    # Update stored results with sorted values
                    st.session_state.files[filename]['circuit_params'] = sorted_popt
                    st.session_state.files[filename]['circuit_conf'] = sorted_perror

                    # Update initial guess to sorted fitted values (for next fitting)
                    st.session_state.files[filename]['initial_guess'] = list(sorted_popt)
                    # Also store globally for use when switching files
                    st.session_state['global_initial_guess'] = list(sorted_popt)
                    # Update widget values
                    for i, val in enumerate(sorted_popt):
                        widget_key = f"init_{i}"
                        st.session_state[widget_key] = f"{val:.2e}"

                    # Calculate conductivity for each R element (using sorted values)
                    S = st.session_state.sample_info.get('area', 1.0)
                    L = st.session_state.sample_info.get('thickness', 0.1)

                    # Find all R elements and calculate individual conductivities
                    r_values = {}  # {'R1': value, 'R2': value, ...}
                    r_sigmas = {}  # {'R1': sigma, 'R2': sigma, ...}

                    for i, name in enumerate(param_names):
                        if 'R' in name and 'CPE' not in name:
                            # Extract R name (e.g., 'R1', 'R2')
                            r_name = name.split('_')[0] if '_' in name else name
                            r_values[r_name] = sorted_popt[i]
                            r_sigmas[r_name] = r2sigma(sorted_popt[i], S, L)

                    # Calculate total R and sigma
                    R_total = sum(r_values.values())
                    sigma_total = r2sigma(R_total, S, L)

                    # Store effective caps and sorted result
                    st.session_state.files[filename]['effective_caps'] = effective_caps
                    st.session_state.files[filename]['sorted_params'] = sorted_result

                    # Store all results
                    st.session_state.files[filename]['total_sigma'] = sigma_total
                    st.session_state.files[filename]['total_R'] = R_total
                    st.session_state.files[filename]['r_values'] = r_values
                    st.session_state.files[filename]['r_sigmas'] = r_sigmas

                st.rerun()  # Rerun to update plots immediately

            except Exception as e:
                st.error(f"Fitting failed: {str(e)}")

    # Handle Auto Fit button
    if auto_fit_clicked:
        # Create progress elements first (outside spinner)
        status_text = st.empty()
        progress_bar = st.progress(0)
        status_text.text("Auto Fitting (Optuna optimization)...")

        try:
            freq = data['freq']
            Z = data['Z']

            # Apply fitting range if set
            freq_range = st.session_state.freq_range
            if freq_range:
                start_idx, end_idx = freq_range
                freq_fit = freq[start_idx:end_idx + 1]
                Z_fit_data = Z[start_idx:end_idx + 1]
            else:
                freq_fit = freq
                Z_fit_data = Z

            # Get Auto Fit settings
            auto_settings = st.session_state.get('auto_fit_settings', {})
            n_trials = auto_settings.get('n_trials', 30)
            timeout = auto_settings.get('timeout', 2)
            early_stop_rmspe = auto_settings.get('early_stop_rmspe', 3.0) / 100  # Convert % to decimal
            log_step = auto_settings.get('log_step', 0.5)
            r_range = (auto_settings.get('r_min', 1e0), auto_settings.get('r_max', 1e8))
            cpe_q_range = (auto_settings.get('cpe_q_min', 1e-12), auto_settings.get('cpe_q_max', 1e-4))

            # Build individual R ranges from settings
            r_ranges = {}
            for r_name in ['R1', 'R2', 'R3']:
                r_key = f'{r_name.lower()}_range'
                if r_key in auto_settings:
                    r_min_exp, r_max_exp = auto_settings[r_key]
                    r_ranges[r_name] = (10 ** r_min_exp, 10 ** r_max_exp)

            # Build individual CPE Q ranges from settings
            cpe_q_ranges = {}
            for cpe_name in ['CPE1', 'CPE2', 'CPE3']:
                cpe_key = f'{cpe_name.lower()}_q_range'
                if cpe_key in auto_settings:
                    q_min_exp, q_max_exp = auto_settings[cpe_key]
                    cpe_q_ranges[cpe_name] = (10 ** q_min_exp, 10 ** q_max_exp)

            # Determine model list
            use_current_model = auto_settings.get('use_current_model', True)
            if use_current_model:
                model_list = [circuit_model] if circuit_model else None
            else:
                model_list = auto_settings.get('model_list', None)
                if not model_list:
                    model_list = [circuit_model] if circuit_model else None

            # Use BlackBoxOptEIS for optimization
            # Use the selected weighting method only
            optimizer = BlackBoxOptEIS(
                freq_fit, Z_fit_data,
                model_list=model_list,
                weight_list=[weight_method],  # Use selected weighting method
                n_trials=n_trials,
                timeout=timeout,
                early_stop_rmspe=early_stop_rmspe,
                log_step=log_step,
                r_range=r_range,
                cpe_q_range=cpe_q_range,
                r_ranges=r_ranges,
                cpe_q_ranges=cpe_q_ranges
            )

            # Run optimization with progress callback
            def progress_callback(trial_num, total_trials, best_rmspe):
                progress = min(trial_num / total_trials, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Trial {trial_num}/{total_trials}, Best RMSPE: {best_rmspe*100:.1f}%")

            best_params = optimizer.optimize(progress_callback=progress_callback)
            progress_bar.empty()
            status_text.empty()

            # Fit with best parameters
            popt, perror, Z_fit_result, rmspe, best_model, param_names = optimizer.fit_best()

            if popt is not None and len(popt) > 0:
                # Create circuit for full prediction
                circuit = CustomCircuit(best_model, initial_guess=list(popt))
                circuit.parameters_ = popt
                circuit.conf_ = perror

                # Predict fitted impedance for full range
                Z_fit = circuit.predict(freq)

                # Get param names for sorting
                param_names_list, _ = circuit.get_param_names()

                # Sort by effective capacitance
                sorted_result = sort_ecm_by_cap(popt, perror, param_names_list)
                effective_caps = sorted_result.get('effective_caps', {})

                # Build sorted parameter arrays
                sorted_popt = []
                sorted_perror = []
                for name in param_names_list:
                    if name in sorted_result:
                        sorted_popt.append(sorted_result[name])
                        sorted_perror.append(sorted_result.get(f'{name}_error', 0.0))
                    else:
                        idx = param_names_list.index(name)
                        sorted_popt.append(popt[idx])
                        sorted_perror.append(perror[idx] if perror is not None else 0.0)

                sorted_popt = np.array(sorted_popt)
                sorted_perror = np.array(sorted_perror)

                # Update circuit with sorted parameters
                circuit.parameters_ = sorted_popt
                circuit.conf_ = sorted_perror

                # Store results with sorted values
                st.session_state.files[filename]['Z_fit'] = Z_fit
                st.session_state.files[filename]['circuit_model'] = best_model
                st.session_state.files[filename]['circuit_params'] = sorted_popt
                st.session_state.files[filename]['circuit_conf'] = sorted_perror
                st.session_state.files[filename]['circuit_object'] = circuit
                st.session_state.files[filename]['rmspe'] = rmspe

                # Update initial guess with sorted values
                st.session_state.files[filename]['initial_guess'] = list(sorted_popt)
                st.session_state['global_initial_guess'] = list(sorted_popt)

                # Update widget values
                for i, val in enumerate(sorted_popt):
                    widget_key = f"init_{i}"
                    st.session_state[widget_key] = f"{val:.2e}"

                # Calculate conductivity for each R element (using sorted values)
                S = st.session_state.sample_info.get('area', 1.0)
                L = st.session_state.sample_info.get('thickness', 0.1)

                # Find all R elements and calculate individual conductivities
                r_values = {}
                r_sigmas = {}

                for i, name in enumerate(param_names_list):
                    if 'R' in name and 'CPE' not in name:
                        r_name = name.split('_')[0] if '_' in name else name
                        r_values[r_name] = sorted_popt[i]
                        r_sigmas[r_name] = r2sigma(sorted_popt[i], S, L)

                # Calculate total R and sigma
                R_total = sum(r_values.values())
                sigma_total = r2sigma(R_total, S, L)

                # Store effective caps and sorted result
                st.session_state.files[filename]['effective_caps'] = effective_caps
                st.session_state.files[filename]['sorted_params'] = sorted_result

                # Store all results
                st.session_state.files[filename]['total_sigma'] = sigma_total
                st.session_state.files[filename]['total_R'] = R_total
                st.session_state.files[filename]['r_values'] = r_values
                st.session_state.files[filename]['r_sigmas'] = r_sigmas

                st.success(f"Auto Fit completed! RMSPE: {rmspe*100:.1f}%")
                st.rerun()
            else:
                st.error("Auto Fit failed to find a valid solution")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Auto Fit failed: {str(e)}")

    # Handle Batch Fit button
    if batch_fit_clicked:
        # Get selected files from batch settings
        selected_files = list(st.session_state.get('batch_selected_files', set()))
        if len(selected_files) == 0:
            st.warning("No files selected for batch fitting. Select files in Batch Fit Settings.")
        else:
            st.info(f"Batch fitting {len(selected_files)} files...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            successful = 0
            failed = 0
            high_rmspe_count = 0
            current_initial_guess = None

            # Get batch fit settings
            batch_settings = st.session_state.get('batch_fit_settings', {})
            use_previous_result = batch_settings.get('use_previous_result', True)
            stop_on_error = batch_settings.get('stop_on_error', False)
            rmspe_threshold = batch_settings.get('rmspe_threshold', 10.0) / 100  # Convert to decimal

            # Get initial guess from UI input (for first cycle)
            from impedance.models.circuits.fitting import calculateCircuitLength
            n_params = calculateCircuitLength(circuit_model)
            ui_initial_guess = []
            for i in range(n_params):
                widget_key = f"init_{i}"
                if widget_key in st.session_state:
                    try:
                        ui_initial_guess.append(float(st.session_state[widget_key]))
                    except ValueError:
                        ui_initial_guess.append(1.0)
                else:
                    # Fallback default values
                    if i % 3 == 0:
                        ui_initial_guess.append(1e3)
                    elif i % 3 == 1:
                        ui_initial_guess.append(1e-9)
                    else:
                        ui_initial_guess.append(0.9)

            batch_stopped = False
            for idx, fname in enumerate(selected_files):
                if batch_stopped:
                    break

                status_text.text(f"Processing {fname} ({idx + 1}/{len(selected_files)})")
                progress_bar.progress((idx + 1) / len(selected_files))

                fdata = st.session_state.files[fname]
                freq = fdata['freq']
                Z = fdata['Z']

                try:
                    # Apply fitting range if set
                    freq_range = st.session_state.freq_range
                    if freq_range:
                        start_idx, end_idx = freq_range
                        freq_fit = freq[start_idx:end_idx + 1]
                        Z_fit_data = Z[start_idx:end_idx + 1]
                    else:
                        freq_fit = freq
                        Z_fit_data = Z

                    # Determine initial guess based on settings
                    if use_previous_result and current_initial_guess is not None and len(current_initial_guess) == n_params:
                        initial_guess = current_initial_guess
                    else:
                        # Use UI initial guess
                        initial_guess = ui_initial_guess

                    # Fit circuit
                    popt, perror = circuit_fit(
                        freq_fit, Z_fit_data,
                        circuit_model,
                        initial_guess,
                        weight_method=weight_method
                    )

                    # Create CustomCircuit for prediction
                    circuit = CustomCircuit(circuit_model, initial_guess=list(popt))
                    circuit.parameters_ = popt
                    circuit.conf_ = perror

                    # Predict fitted impedance for full range
                    Z_fit = circuit.predict(freq)

                    # Calculate RMSPE
                    rmspe = calc_rmspe(Z_fit_data, circuit.predict(freq_fit))

                    # Get param names for sorting
                    param_names, _ = circuit.get_param_names()

                    # Sort by effective capacitance
                    sorted_result = sort_ecm_by_cap(popt, perror, param_names)
                    effective_caps = sorted_result.get('effective_caps', {})

                    # Build sorted parameter arrays
                    sorted_popt = []
                    sorted_perror = []
                    for name in param_names:
                        if name in sorted_result:
                            sorted_popt.append(sorted_result[name])
                            sorted_perror.append(sorted_result.get(f'{name}_error', 0.0))
                        else:
                            idx = param_names.index(name)
                            sorted_popt.append(popt[idx])
                            sorted_perror.append(perror[idx] if perror is not None else 0.0)

                    sorted_popt = np.array(sorted_popt)
                    sorted_perror = np.array(sorted_perror)

                    # Update circuit with sorted parameters
                    circuit.parameters_ = sorted_popt
                    circuit.conf_ = sorted_perror

                    # Store results with sorted values
                    st.session_state.files[fname]['Z_fit'] = Z_fit
                    st.session_state.files[fname]['circuit_model'] = circuit_model
                    st.session_state.files[fname]['circuit_params'] = sorted_popt
                    st.session_state.files[fname]['circuit_conf'] = sorted_perror
                    st.session_state.files[fname]['circuit_object'] = circuit
                    st.session_state.files[fname]['rmspe'] = rmspe

                    # Update initial guess for next iteration (use sorted values)
                    current_initial_guess = list(sorted_popt)

                    # Calculate conductivity for each R element (using sorted values)
                    S = st.session_state.sample_info.get('area', 1.0)
                    L = st.session_state.sample_info.get('thickness', 0.1)

                    r_values = {}
                    r_sigmas = {}

                    for i, name in enumerate(param_names):
                        if 'R' in name and 'CPE' not in name:
                            r_name = name.split('_')[0] if '_' in name else name
                            r_values[r_name] = sorted_popt[i]
                            r_sigmas[r_name] = r2sigma(sorted_popt[i], S, L)

                    R_total = sum(r_values.values())
                    sigma_total = r2sigma(R_total, S, L)

                    # Store effective caps and sorted result
                    st.session_state.files[fname]['effective_caps'] = effective_caps
                    st.session_state.files[fname]['sorted_params'] = sorted_result

                    st.session_state.files[fname]['total_sigma'] = sigma_total
                    st.session_state.files[fname]['total_R'] = R_total
                    st.session_state.files[fname]['r_values'] = r_values
                    st.session_state.files[fname]['r_sigmas'] = r_sigmas

                    successful += 1

                    # Check RMSPE threshold
                    if rmspe > rmspe_threshold:
                        high_rmspe_count += 1

                except Exception as e:
                    failed += 1
                    st.warning(f"Failed to fit {fname}: {str(e)}")
                    if stop_on_error:
                        st.error("Batch fitting stopped due to error.")
                        batch_stopped = True

            progress_bar.empty()
            status_text.empty()

            # Update global initial guess
            if current_initial_guess is not None:
                st.session_state['global_initial_guess'] = current_initial_guess
                st.session_state.files[filename]['initial_guess'] = current_initial_guess
                for i, val in enumerate(current_initial_guess):
                    widget_key = f"init_{i}"
                    st.session_state[widget_key] = f"{val:.2e}"

            # Show summary
            result_msg = f"Batch fitting completed! Success: {successful}, Failed: {failed}"
            if high_rmspe_count > 0:
                result_msg += f", High RMSPE (>{rmspe_threshold*100:.0f}%): {high_rmspe_count}"
            st.success(result_msg)
            st.rerun()

    # Handle Auto-Batch Fit button
    if auto_batch_clicked:
        # Get selected files from batch settings
        selected_files = list(st.session_state.get('batch_selected_files', set()))
        if len(selected_files) == 0:
            st.warning("No files selected for Auto-Batch fitting. Select files in Batch Fit Settings.")
        else:
            st.info(f"Auto-Batch fitting {len(selected_files)} files...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            successful = 0
            failed = 0
            high_rmspe_count = 0

            # Get batch fit settings
            batch_settings = st.session_state.get('batch_fit_settings', {})
            use_previous_result = batch_settings.get('use_previous_result', True)
            stop_on_error = batch_settings.get('stop_on_error', False)
            rmspe_threshold = batch_settings.get('rmspe_threshold', 10.0) / 100

            # Get Auto Fit settings
            auto_settings = st.session_state.get('auto_fit_settings', {})
            n_trials = auto_settings.get('n_trials', 30)
            timeout = auto_settings.get('timeout', 2)
            early_stop_rmspe = auto_settings.get('early_stop_rmspe', 3.0) / 100
            log_step = auto_settings.get('log_step', 0.5)
            r_range = (auto_settings.get('r_min', 1e0), auto_settings.get('r_max', 1e8))
            cpe_q_range = (auto_settings.get('cpe_q_min', 1e-12), auto_settings.get('cpe_q_max', 1e-4))

            # Build individual R ranges from settings
            r_ranges = {}
            for r_name in ['R1', 'R2', 'R3']:
                r_key = f'{r_name.lower()}_range'
                if r_key in auto_settings:
                    r_min_exp, r_max_exp = auto_settings[r_key]
                    r_ranges[r_name] = (10 ** r_min_exp, 10 ** r_max_exp)

            # Build individual CPE Q ranges from settings
            cpe_q_ranges = {}
            for cpe_name in ['CPE1', 'CPE2', 'CPE3']:
                cpe_key = f'{cpe_name.lower()}_q_range'
                if cpe_key in auto_settings:
                    q_min_exp, q_max_exp = auto_settings[cpe_key]
                    cpe_q_ranges[cpe_name] = (10 ** q_min_exp, 10 ** q_max_exp)

            # Determine model list
            use_current_model = auto_settings.get('use_current_model', True)
            if use_current_model:
                model_list = [circuit_model] if circuit_model else None
            else:
                model_list = auto_settings.get('model_list', None)
                if not model_list:
                    model_list = [circuit_model] if circuit_model else None

            current_initial_guess = None
            batch_stopped = False

            for idx, fname in enumerate(selected_files):
                if batch_stopped:
                    break

                status_text.text(f"Auto-fitting {fname} ({idx + 1}/{len(selected_files)})")
                progress_bar.progress((idx + 1) / len(selected_files))

                fdata = st.session_state.files[fname]
                freq = fdata['freq']
                Z = fdata['Z']

                try:
                    # Apply fitting range if set
                    freq_range = st.session_state.freq_range
                    if freq_range:
                        start_idx, end_idx = freq_range
                        freq_fit = freq[start_idx:end_idx + 1]
                        Z_fit_data = Z[start_idx:end_idx + 1]
                    else:
                        freq_fit = freq
                        Z_fit_data = Z

                    # Use BlackBoxOptEIS for optimization
                    optimizer = BlackBoxOptEIS(
                        freq_fit, Z_fit_data,
                        model_list=model_list,
                        weight_list=[weight_method],
                        n_trials=n_trials,
                        timeout=timeout,
                        early_stop_rmspe=early_stop_rmspe,
                        log_step=log_step,
                        r_range=r_range,
                        cpe_q_range=cpe_q_range,
                        r_ranges=r_ranges,
                        cpe_q_ranges=cpe_q_ranges
                    )

                    # Run optimization (no progress callback for batch mode)
                    best_params = optimizer.optimize()

                    # Fit with best parameters
                    popt, perror, Z_fit_result, rmspe, best_model, param_names_fit = optimizer.fit_best()

                    if popt is not None and len(popt) > 0:
                        # Create circuit for full prediction
                        circuit = CustomCircuit(best_model, initial_guess=list(popt))
                        circuit.parameters_ = popt
                        circuit.conf_ = perror

                        # Predict fitted impedance for full range
                        Z_fit = circuit.predict(freq)

                        # Get param names for sorting
                        param_names_list, _ = circuit.get_param_names()

                        # Sort by effective capacitance
                        sorted_result = sort_ecm_by_cap(popt, perror, param_names_list)
                        effective_caps = sorted_result.get('effective_caps', {})

                        # Build sorted parameter arrays
                        sorted_popt = []
                        sorted_perror = []
                        for name in param_names_list:
                            if name in sorted_result:
                                sorted_popt.append(sorted_result[name])
                                sorted_perror.append(sorted_result.get(f'{name}_error', 0.0))
                            else:
                                pidx = param_names_list.index(name)
                                sorted_popt.append(popt[pidx])
                                sorted_perror.append(perror[pidx] if perror is not None else 0.0)

                        sorted_popt = np.array(sorted_popt)
                        sorted_perror = np.array(sorted_perror)

                        # Update circuit with sorted parameters
                        circuit.parameters_ = sorted_popt
                        circuit.conf_ = sorted_perror

                        # Store results
                        st.session_state.files[fname]['Z_fit'] = Z_fit
                        st.session_state.files[fname]['circuit_model'] = best_model
                        st.session_state.files[fname]['circuit_params'] = sorted_popt
                        st.session_state.files[fname]['circuit_conf'] = sorted_perror
                        st.session_state.files[fname]['circuit_object'] = circuit
                        st.session_state.files[fname]['rmspe'] = rmspe
                        st.session_state.files[fname]['initial_guess'] = list(sorted_popt)

                        # Store effective caps and sorted result
                        st.session_state.files[fname]['effective_caps'] = effective_caps
                        st.session_state.files[fname]['sorted_params'] = sorted_result

                        # Calculate conductivity
                        S = st.session_state.sample_info.get('area', 1.0)
                        L = st.session_state.sample_info.get('thickness', 0.1)

                        r_values = {}
                        r_sigmas = {}
                        for i, name in enumerate(param_names_list):
                            if 'R' in name and 'CPE' not in name:
                                r_name = name.split('_')[0] if '_' in name else name
                                r_values[r_name] = sorted_popt[i]
                                r_sigmas[r_name] = r2sigma(sorted_popt[i], S, L)

                        R_total = sum(r_values.values())
                        sigma_total = r2sigma(R_total, S, L)

                        st.session_state.files[fname]['total_sigma'] = sigma_total
                        st.session_state.files[fname]['total_R'] = R_total
                        st.session_state.files[fname]['r_values'] = r_values
                        st.session_state.files[fname]['r_sigmas'] = r_sigmas

                        successful += 1

                        # Check RMSPE threshold
                        if rmspe > rmspe_threshold:
                            high_rmspe_count += 1

                        # Update current_initial_guess for next file if enabled
                        if use_previous_result:
                            current_initial_guess = list(sorted_popt)
                    else:
                        failed += 1
                        st.warning(f"Auto-fit failed for {fname}: No valid solution found")
                        if stop_on_error:
                            st.error("Auto-Batch fitting stopped due to error.")
                            batch_stopped = True

                except Exception as e:
                    failed += 1
                    st.warning(f"Failed to auto-fit {fname}: {str(e)}")
                    if stop_on_error:
                        st.error("Auto-Batch fitting stopped due to error.")
                        batch_stopped = True

            progress_bar.empty()
            status_text.empty()

            # Update global initial guess
            if current_initial_guess is not None:
                st.session_state['global_initial_guess'] = current_initial_guess
                st.session_state.files[filename]['initial_guess'] = current_initial_guess
                for i, val in enumerate(current_initial_guess):
                    widget_key = f"init_{i}"
                    st.session_state[widget_key] = f"{val:.2e}"

            # Show summary
            result_msg = f"Auto-Batch fitting completed! Success: {successful}, Failed: {failed}"
            if high_rmspe_count > 0:
                result_msg += f", High RMSPE (>{rmspe_threshold*100:.0f}%): {high_rmspe_count}"
            st.success(result_msg)
            st.rerun()

    # Parameter table with editable initial values
    st.markdown("**Parameters**")

    # Always get param names from current circuit_model input (not from saved circuit_object)
    # This ensures parameters update immediately when circuit model is changed
    try:
        from impedance.models.circuits.fitting import calculateCircuitLength
        n_params = calculateCircuitLength(circuit_model)
        dummy_guess = [1.0] * n_params
        temp_circuit = CustomCircuit(circuit_model, initial_guess=dummy_guess)
        param_names, units = temp_circuit.get_param_names()
    except Exception as e:
        st.error(f"Invalid circuit model: {str(e)}")
        param_names = []
        units = []

    if len(param_names) > 0:
        # Format parameter names
        display_names = [format_param_name(name) for name in param_names]

        # Initialize param_fixed state if needed
        if 'param_fixed' not in st.session_state:
            st.session_state.param_fixed = {}

        # Ensure param_fixed has entries for all parameters
        for i in range(len(param_names)):
            key = f"fixed_{i}"
            if key not in st.session_state.param_fixed:
                st.session_state.param_fixed[key] = False  # Default: Variable

        # Get initial guess with priority:
        # 1. Current file's initial_guess (from previous fit)
        # 2. Global initial guess (from last fit on any file)
        # 3. Generate defaults
        initial_guess = data.get('initial_guess')

        if initial_guess is None or len(initial_guess) != len(param_names):
            # Try global initial guess
            global_guess = st.session_state.get('global_initial_guess')
            if global_guess is not None and len(global_guess) == len(param_names):
                initial_guess = global_guess
            else:
                # Generate appropriate defaults based on parameter names
                initial_guess = []
                for name in param_names:
                    if 'R' in name and 'CPE' not in name:
                        initial_guess.append(1e3)  # Resistance
                    elif '_0' in name or '_Q' in name.upper():
                        initial_guess.append(1e-9)  # CPE Q parameter
                    elif '_1' in name or 'alpha' in name.lower():
                        initial_guess.append(0.9)  # CPE alpha
                    else:
                        initial_guess.append(1.0)  # Default

        params = data.get('circuit_params')
        confs = data.get('circuit_conf')

        # Header row with "Initial values" and Set all V/F buttons
        header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
        with header_col1:
            st.caption("Initial values (editable)")
        with header_col2:
            # Set all V button - use HTML button for custom styling
            if st.button("Set all V", key="set_all_variable", help="Set all parameters to Variable"):
                for i in range(len(param_names)):
                    st.session_state.param_fixed[f"fixed_{i}"] = False
                st.rerun()
        with header_col3:
            if st.button("Set all F", key="set_all_fixed", help="Set all parameters to Fixed"):
                for i in range(len(param_names)):
                    st.session_state.param_fixed[f"fixed_{i}"] = True
                st.rerun()

        # Build CSS for V/F buttons - target buttons by their text content
        v_button_indices = []
        f_button_indices = []
        for i in range(len(param_names)):
            fixed_key = f"fixed_{i}"
            is_fixed = st.session_state.param_fixed.get(fixed_key, False)
            if is_fixed:
                f_button_indices.append(i)
            else:
                v_button_indices.append(i)

        # Inject CSS to style V buttons (red) and F buttons (black/default)
        st.markdown("""
        <style>
        /* V button - red text */
        .vf-row button p {
            font-weight: bold !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # V/F toggle row
        st.markdown('<div class="vf-row">', unsafe_allow_html=True)
        vf_cols = st.columns(len(param_names))
        for i, col in enumerate(vf_cols):
            with col:
                fixed_key = f"fixed_{i}"
                is_fixed = st.session_state.param_fixed.get(fixed_key, False)

                # V = filled circle, F = empty circle (using symbols)
                if is_fixed:
                    btn_label = "○ F"  # empty circle + F
                else:
                    btn_label = "● V"  # filled circle + V

                if st.button(btn_label, key=f"toggle_vf_{i}", help=VF_TOGGLE_HELP):
                    st.session_state.param_fixed[fixed_key] = not is_fixed
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Initial values input row
        init_cols = st.columns(len(param_names))
        new_initial_guess = []
        for i, (name, col) in enumerate(zip(display_names, init_cols)):
            with col:
                init_val = initial_guess[i] if i < len(initial_guess) else 1.0
                # Use shared key (not per-file) so values persist when switching files
                widget_key = f"init_{i}"
                # Set default value if not already set
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = f"{init_val:.2e}"

                # Check if this parameter is fixed
                is_fixed = st.session_state.param_fixed.get(f"fixed_{i}", False)

                new_val = st.text_input(
                    name,
                    key=widget_key,
                    label_visibility="visible",
                    disabled=False  # Keep editable but mark visually
                )
                try:
                    new_initial_guess.append(float(new_val))
                except ValueError:
                    new_initial_guess.append(init_val)

        # Update initial guess in session state
        st.session_state.files[filename]['initial_guess'] = new_initial_guess
        # Also update global
        st.session_state['global_initial_guess'] = new_initial_guess

        # Show fitted values table (read-only)
        if params is not None:
            table_data = {'': ['Value', 'Error', 'Error %']}
            for i, name in enumerate(display_names):
                if i < len(params):
                    val = params[i]
                    err = confs[i] if confs is not None and i < len(confs) else 0.0
                    err_pct = (err / val * 100) if val != 0 else 0.0
                    table_data[name] = [
                        f"{val:.3e}",
                        f"{err:.2e}",
                        f"{err_pct:.1f}%"
                    ]

            df = pd.DataFrame(table_data)
            st.dataframe(df, hide_index=True, width="stretch")

        # Show R labels input if R elements exist
        r_values = data.get('r_values', {})
        if len(r_values) >= 1:
            st.markdown("**R Element Labels**")
            r_label_cols = st.columns(max(len(r_values), 2))
            for i, (r_name, col) in enumerate(zip(sorted(r_values.keys()), r_label_cols)):
                with col:
                    # Default labels depend on number of R elements
                    if len(r_values) == 1:
                        default_labels = {'R1': 'total'}
                    else:
                        default_labels = {'R1': 'bulk', 'R2': 'gb', 'R3': 'electrode', 'R4': 'R4'}
                    current_label = st.session_state.r_labels.get(r_name, default_labels.get(r_name, r_name))
                    new_label = st.text_input(
                        r_name,
                        value=current_label,
                        key=f"r_label_{r_name}",
                        label_visibility="visible"
                    )
                    st.session_state.r_labels[r_name] = new_label

        # Show RMSPE and conductivity below table
        if data.get('rmspe') is not None:
            rmspe_pct = data['rmspe'] * 100  # Convert to percentage
            st.metric("RMSPE", f"{rmspe_pct:.1f} %", help=RMSPE_HELP)

            # Build conductivity table with all R values
            def format_sigma(sigma):
                if sigma is None or sigma == 0:
                    return "–"
                exp = int(np.floor(np.log10(abs(sigma))))
                mantissa = sigma / (10 ** exp)
                mantissa = round(mantissa, 2)
                return f"{mantissa:.2f}e{exp}"

            # Get effective capacitance data
            effective_caps = data.get('effective_caps', {})

            # Summary table label with help
            st.metric("Summary table", "", help=SUMMARY_TABLE_HELP, label_visibility="visible")

            # Create summary table with conductivity and effective capacitance
            cond_rows = []
            temp = data.get('temperature')

            # Total conductivity
            if data.get('total_sigma'):
                sigma = data['total_sigma']
                log_sigma = np.log10(sigma) if sigma > 0 else None
                log_sigma_T = np.log10(sigma * temp) if sigma > 0 and temp else None
                cond_rows.append({
                    'Type': 'total',
                    'R / Ω': f"{data.get('total_R', 0):.2e}",
                    'σ / S cm⁻¹': format_sigma(sigma),
                    'log(σ)': f"{log_sigma:.3f}" if log_sigma else "–",
                    'log(σT)': f"{log_sigma_T:.3f}" if log_sigma_T else "–",
                    'Ceff / F': "–"
                })

            # Individual R conductivities with effective capacitance
            r_sigmas = data.get('r_sigmas', {})
            for r_name in sorted(r_values.keys()):
                r_label = st.session_state.r_labels.get(r_name, r_name)
                r_val = r_values.get(r_name, 0)
                sigma = r_sigmas.get(r_name, 0)
                log_sigma = np.log10(sigma) if sigma > 0 else None
                log_sigma_T = np.log10(sigma * temp) if sigma > 0 and temp else None

                # Get effective capacitance for this R element
                ceff = effective_caps.get(r_name, None)
                ceff_str = f"{ceff:.2e}" if ceff and ceff > 0 else "–"

                cond_rows.append({
                    'Type': r_label,
                    'R / Ω': f"{r_val:.2e}",
                    'σ / S cm⁻¹': format_sigma(sigma),
                    'log(σ)': f"{log_sigma:.3f}" if log_sigma else "–",
                    'log(σT)': f"{log_sigma_T:.3f}" if log_sigma_T else "–",
                    'Ceff / F': ceff_str
                })

            if cond_rows:
                cond_df = pd.DataFrame(cond_rows)
                st.dataframe(cond_df, hide_index=True, use_container_width=True)


def multipoint_analysis_table():
    """Multipoint analysis table"""
    if len(st.session_state.files) == 0:
        st.info("No data available")
        return

    # Get files with circuit fitting results
    fitted_files = [f for f, d in st.session_state.files.items() if d.get('circuit_params') is not None]

    if len(fitted_files) == 0:
        st.info("Perform circuit fitting to see multipoint analysis results")
        return

    # Temperature input methods (first)
    temp_input_help = """**Temperature Input Methods:**

**1. Direct** - Enter comma-separated values
- Example: `25, 50, 100, 150` (°C) or `298, 350, 400` (K)

**2. Pattern** - Single values and `[T0,STEP,NUM]` can be mixed
- `25,[50,50,4]` → 25, 50, 100, 150, 200
- `[300,50,3]` → 300, 350, 400
- `25,50,[100,50,3]` → 25, 50, 100, 150, 200
- `[300,50,3],[450,-50,2]` → 300, 350, 400, 450, 400

**3. From Filename** - Extract from filenames with `[separator, index]`
- File: `sample_350K_01.csv` → Pattern: `[_,1]` → 350
- File: `EIS_25C_data.csv` → Pattern: `[_,1],[C,0]` → 25
- Splits filename by separator, takes index-th part

Select unit (K or °C) below. °C values are auto-converted to K."""

    input_method = st.radio(
        "Temperature Input",
        ["Direct", "Pattern", "From Filename"],
        horizontal=True,
        help=temp_input_help
    )

    # Temperature unit selection (after input method)
    # Default: °C for Direct/Pattern, K for From Filename
    default_unit_index = 0 if input_method == "From Filename" else 1  # 0=K, 1=°C
    temp_unit_col, _ = st.columns([1, 3])
    with temp_unit_col:
        temp_unit = st.selectbox(
            "Temperature Unit",
            ["K", "°C"],
            index=default_unit_index,
            key=f"temp_unit_select_{input_method}"  # Different key per input method
        )

    if input_method == "Direct":
        # Direct input - comma separated values
        # Display temperatures in selected unit
        current_temps = []
        for filename in fitted_files:
            temp = st.session_state.files[filename].get('temperature')
            if temp:
                # Convert stored K to display unit
                display_temp = temp - 273.15 if temp_unit == "°C" else temp
                current_temps.append(f"{display_temp:.1f}")
            else:
                current_temps.append("25" if temp_unit == "°C" else "298.15")
        default_temp_str = ", ".join(current_temps)

        direct_help = f"""Enter {len(fitted_files)} temperature values separated by commas.
Example: `298, 350, 400, 450`"""

        temp_input = st.text_input(
            "Temperature values",
            value=default_temp_str,
            placeholder="298, 350, 400" if temp_unit == "K" else "25, 50, 100",
            help=direct_help,
            label_visibility="collapsed"
        )

        # Apply button to confirm temperature changes
        if st.button("Apply Temperatures", key="apply_direct_temps"):
            try:
                temp_values = [float(t.strip()) for t in temp_input.split(",") if t.strip()]
                # Convert from °C to K if needed (input is in selected unit)
                if temp_unit == "°C":
                    temp_values = [t + 273.15 for t in temp_values]
                for i, filename in enumerate(fitted_files):
                    if i < len(temp_values) and temp_values[i] > 0:
                        st.session_state.files[filename]['temperature'] = temp_values[i]
                st.success(f"Applied {len(temp_values)} temperature values")
                st.rerun()
            except ValueError:
                st.warning("Invalid format. Use comma-separated numbers.")

    elif input_method == "Pattern":
        pattern_help = """**Pattern Format:** Single values and `[T0,STEP,NUM]` can be mixed.

- **T0**: Starting temperature
- **STEP**: Temperature increment
- **NUM**: Number of points

**Examples:**

| Pattern | Generated Temperatures |
|---------|----------------------|
| `25,[50,50,4]` | 25, 50, 100, 150, 200 |
| `[300,50,3]` | 300, 350, 400 |
| `25,50,[100,50,3]` | 25, 50, 100, 150, 200 |
| `[300,50,3],[450,-50,2]` | 300, 350, 400, 450, 400 |
| `[400,-50,5]` | 400, 350, 300, 250, 200 |

Each `[T0,STEP,NUM]` generates NUM values: T0, T0+STEP, T0+2×STEP, ..."""

        pattern_input = st.text_input(
            "Temperature Pattern",
            value="",
            placeholder="25,[50,50,4]",
            help=pattern_help,
            label_visibility="collapsed"
        )

        if pattern_input:
            temp_values = parse_temperature_pattern(pattern_input)
            if temp_values:
                # Show preview (convert for display if needed)
                display_values = temp_values.copy()
                st.caption(f"Generated: {', '.join([str(int(t)) if t == int(t) else str(t) for t in display_values[:10]])}{'...' if len(display_values) > 10 else ''} ({len(display_values)} values)")

                # Apply button to confirm temperature changes
                if st.button("Apply Pattern", key="apply_pattern_temps"):
                    # Convert from °C to K if needed
                    if temp_unit == "°C":
                        temp_values = [t + 273.15 for t in temp_values]
                    for i, filename in enumerate(fitted_files):
                        if i < len(temp_values) and temp_values[i] > 0:
                            st.session_state.files[filename]['temperature'] = temp_values[i]
                    st.success(f"Applied {len(temp_values)} temperature values")
                    st.rerun()
            else:
                st.warning("Invalid pattern format.")

    elif input_method == "From Filename":
        filename_help = """**Pattern Format:** `[separator,index],[separator,index],...`

- **separator**: Character(s) to split by
- **index**: Which part to take (0-based)

**Examples:**

| Filename | Pattern | Result |
|----------|---------|--------|
| `sample_350K_01.csv` | `[_,1]` | 350K → **350** |
| `sample_350K_01.csv` | `[_,1],[K,0]` | 350K → 350 → **350** |
| `EIS_25C_data.csv` | `[_,1],[C,0]` | 25C → 25 → **25** |
| `300K-sample-01.csv` | `[K,0]` | 300 → **300** |
| `data_T473_run1.csv` | `[T,1],[_,0]` | 473_run1 → 473 → **473** |

Patterns are applied sequentially. The first number in the final result is used as temperature."""

        filename_pattern = st.text_input(
            "Filename Pattern",
            value="",
            placeholder="[_,1],[K,0]",
            help=filename_help,
            label_visibility="collapsed"
        )

        if filename_pattern:
            extracted_temps = []
            for filename in fitted_files:
                temp = extract_temp_from_filename(filename, filename_pattern)
                extracted_temps.append(temp)

            # Show preview
            preview_items = []
            for fname, temp in zip(fitted_files[:5], extracted_temps[:5]):
                short_name = fname[:20] + "..." if len(fname) > 20 else fname
                preview_items.append(f"{short_name} → {temp if temp else '?'}")
            st.caption("Preview: " + ", ".join(preview_items))

            # Apply button to confirm temperature extraction
            if st.button("Apply from Filename", key="apply_filename_temps"):
                applied_count = 0
                for i, filename in enumerate(fitted_files):
                    if i < len(extracted_temps) and extracted_temps[i] is not None:
                        temp_val = extracted_temps[i]
                        # Convert from °C to K if needed
                        if temp_unit == "°C":
                            temp_val = temp_val + 273.15
                        if temp_val > 0:
                            st.session_state.files[filename]['temperature'] = temp_val
                            applied_count += 1
                st.success(f"Applied {applied_count} temperature values")
                st.rerun()

    # Collect all analyzed data
    rows = []
    # Check which R elements exist across all files
    all_r_keys = set()
    for filename, data in st.session_state.files.items():
        if data.get('circuit_params') is None:
            continue
        r_values = data.get('r_values', {})
        all_r_keys.update(r_values.keys())

    # Sort R keys (R1, R2, R3, ...)
    sorted_r_keys = sorted(all_r_keys, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)

    for filename, data in st.session_state.files.items():
        if data.get('circuit_params') is None:
            continue

        temp = data.get('temperature')
        row = {
            'File': filename,
            'T (K)': f"{temp:.1f}" if temp else '-',
            '1000/T': f"{1000/temp:.4f}" if temp else '-',
        }

        # Add circuit parameters
        if data.get('circuit_object'):
            circuit = data['circuit_object']
            param_names, _ = circuit.get_param_names()
            for i, name in enumerate(param_names):
                row[name] = f"{data['circuit_params'][i]:.4e}"

        row['RMSPE'] = f"{data.get('rmspe', 0):.6f}"

        # Get r_values and r_sigmas for individual R and σ columns
        r_values = data.get('r_values', {})
        r_sigmas = data.get('r_sigmas', {})

        # Add individual R columns (Rbulk, Rgb, Rtotal)
        for r_key in sorted_r_keys:
            r_label = st.session_state.r_labels.get(r_key, r_key)
            col_name = f"R{r_label}" if r_label != r_key else r_key
            if r_key in r_values:
                row[col_name] = f"{r_values[r_key]:.4e}"
            else:
                row[col_name] = '-'

        # Rtotal
        row['Rtotal'] = f"{data.get('total_R', 0):.4e}" if data.get('total_R') else '-'

        # Add individual σ columns
        for r_key in sorted_r_keys:
            r_label = st.session_state.r_labels.get(r_key, r_key)
            col_name = f"σ{r_label}"
            if r_key in r_sigmas:
                row[col_name] = f"{r_sigmas[r_key]:.4e}"
            else:
                row[col_name] = '-'

        # σtotal
        row['σtotal'] = f"{data.get('total_sigma', 0):.4e}" if data.get('total_sigma') else '-'

        # Add log(σ) columns
        for r_key in sorted_r_keys:
            r_label = st.session_state.r_labels.get(r_key, r_key)
            col_name = f"log(σ{r_label})"
            if r_key in r_sigmas and r_sigmas[r_key] > 0:
                row[col_name] = f"{np.log10(r_sigmas[r_key]):.4f}"
            else:
                row[col_name] = '-'

        # log(σtotal)
        row['log(σtotal)'] = f"{np.log10(data['total_sigma']):.4f}" if data.get('total_sigma') else '-'

        # Add log(σT) columns
        for r_key in sorted_r_keys:
            r_label = st.session_state.r_labels.get(r_key, r_key)
            col_name = f"log(σ{r_label}T)"
            if r_key in r_sigmas and r_sigmas[r_key] > 0 and temp:
                row[col_name] = f"{np.log10(r_sigmas[r_key] * temp):.4f}"
            else:
                row[col_name] = '-'

        # log(σtotalT)
        if data.get('total_sigma') and temp:
            log_sigma_T = np.log10(data['total_sigma'] * temp)
            row['log(σtotalT)'] = f"{log_sigma_T:.4f}"
        else:
            row['log(σtotalT)'] = '-'

        rows.append(row)

    if len(rows) > 0:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

        # CSV export with full column names
        export_rows = []
        for filename, data in st.session_state.files.items():
            if data.get('circuit_params') is None:
                continue
            temp = data.get('temperature')
            row = {
                'File': filename,
                'Temperature (K)': f"{temp:.2f}" if temp else '-',
                '1000/T (K-1)': f"{1000/temp:.4f}" if temp else '-',
            }
            if data.get('circuit_object'):
                circuit = data['circuit_object']
                param_names, _ = circuit.get_param_names()
                for i, name in enumerate(param_names):
                    row[name] = f"{data['circuit_params'][i]:.4e}"
            row['RMSPE'] = f"{data.get('rmspe', 0):.6f}"

            # Get r_values and r_sigmas for individual R and σ columns
            r_values = data.get('r_values', {})
            r_sigmas = data.get('r_sigmas', {})

            # Add individual R columns
            for r_key in sorted_r_keys:
                r_label = st.session_state.r_labels.get(r_key, r_key)
                col_name = f"R_{r_label} (Ohm)"
                if r_key in r_values:
                    row[col_name] = f"{r_values[r_key]:.4e}"
                else:
                    row[col_name] = '-'

            # Rtotal
            row['R_total (Ohm)'] = f"{data.get('total_R', 0):.4e}" if data.get('total_R') else '-'

            # Add individual σ columns
            for r_key in sorted_r_keys:
                r_label = st.session_state.r_labels.get(r_key, r_key)
                col_name = f"sigma_{r_label} (S/cm)"
                if r_key in r_sigmas:
                    row[col_name] = f"{r_sigmas[r_key]:.4e}"
                else:
                    row[col_name] = '-'

            # σtotal
            row['sigma_total (S/cm)'] = f"{data.get('total_sigma', 0):.4e}" if data.get('total_sigma') else '-'

            # Add log(σ) columns
            for r_key in sorted_r_keys:
                r_label = st.session_state.r_labels.get(r_key, r_key)
                col_name = f"log(sigma_{r_label})"
                if r_key in r_sigmas and r_sigmas[r_key] > 0:
                    row[col_name] = f"{np.log10(r_sigmas[r_key]):.4f}"
                else:
                    row[col_name] = '-'

            # log(σtotal)
            row['log(sigma_total)'] = f"{np.log10(data['total_sigma']):.4f}" if data.get('total_sigma') else '-'

            # Add log(σT) columns
            for r_key in sorted_r_keys:
                r_label = st.session_state.r_labels.get(r_key, r_key)
                col_name = f"log(sigma_{r_label}*T)"
                if r_key in r_sigmas and r_sigmas[r_key] > 0 and temp:
                    row[col_name] = f"{np.log10(r_sigmas[r_key] * temp):.4f}"
                else:
                    row[col_name] = '-'

            # log(σtotalT)
            if data.get('total_sigma') and temp:
                row['log(sigma_total*T)'] = f"{np.log10(data['total_sigma'] * temp):.4f}"
            else:
                row['log(sigma_total*T)'] = '-'

            export_rows.append(row)

        csv = pd.DataFrame(export_rows).to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"eis_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def load_session(uploaded_file):
    """Load session from JSON file"""
    try:
        session_data = json.loads(uploaded_file.read().decode())
        
        # Clear current session
        reset_session()
        
        # Load sample info
        if 'sample_info' in session_data:
            st.session_state.sample_info.update(session_data['sample_info'])
        
        # Load files
        if 'files' in session_data:
            for filename, data in session_data['files'].items():
                # Reconstruct complex impedance data
                Z = None
                Z_fit = None
                
                if data.get('Z_real') and data.get('Z_imag'):
                    Z_real = np.array(data['Z_real'])
                    Z_imag = np.array(data['Z_imag'])
                    Z = Z_real + 1j * Z_imag
                
                if data.get('Z_fit_real') and data.get('Z_fit_imag'):
                    Z_fit_real = np.array(data['Z_fit_real'])
                    Z_fit_imag = np.array(data['Z_fit_imag'])
                    Z_fit = Z_fit_real + 1j * Z_fit_imag
                
                freq = np.array(data['freq']) if data.get('freq') else None
                circuit_params = np.array(data['circuit_params']) if data.get('circuit_params') else None
                
                st.session_state.files[filename] = {
                    'freq': freq,
                    'Z': Z,
                    'Z_fit': Z_fit,
                    'circuit_model': data.get('circuit_model'),
                    'circuit_params': circuit_params,
                    'circuit_conf': None,  # Not saved in session
                    'rmspe': data.get('rmspe'),
                    'temperature': data.get('temperature'),
                    'total_sigma': data.get('total_sigma'),
                    'total_R': data.get('total_R'),
                    'bulk_sigma': data.get('bulk_sigma'),
                    'gb_sigma': data.get('gb_sigma'),
                    'initial_guess': data.get('initial_guess')
                }
        
        st.success("Session loaded successfully!")

    except Exception as e:
        st.error(f"Failed to load session: {str(e)}")


def save_session():
    """Save current session to JSON"""
    session_data = {
        'sample_info': st.session_state.sample_info,
        'files': {},
        'timestamp': datetime.now().isoformat()
    }

    # Convert numpy arrays to lists for JSON serialization
    for filename, data in st.session_state.files.items():
        session_data['files'][filename] = {
            'freq': data['freq'].tolist() if data.get('freq') is not None else None,
            'Z_real': np.real(data['Z']).tolist() if data.get('Z') is not None else None,
            'Z_imag': np.imag(data['Z']).tolist() if data.get('Z') is not None else None,
            'Z_fit_real': np.real(data['Z_fit']).tolist() if data.get('Z_fit') is not None else None,
            'Z_fit_imag': np.imag(data['Z_fit']).tolist() if data.get('Z_fit') is not None else None,
            'circuit_model': data.get('circuit_model'),
            'circuit_params': data.get('circuit_params').tolist() if data.get('circuit_params') is not None else None,
            'initial_guess': data.get('initial_guess'),
            'rmspe': data.get('rmspe'),
            'temperature': data.get('temperature'),
            'total_sigma': data.get('total_sigma'),
            'total_R': data.get('total_R'),
            'bulk_sigma': data.get('bulk_sigma'),
            'gb_sigma': data.get('gb_sigma')
        }

    json_str = json.dumps(session_data, indent=2)

    st.download_button(
        label="Download Session (JSON)",
        data=json_str,
        file_name=f"eis_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

    st.success("Session saved!")


def reset_session():
    """Reset all session data"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()


def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        # Header with title and control buttons
        sidebar_header()

        st.markdown("---")

        # File upload section
        sidebar_file_upload()

        # Sample info section
        sidebar_sample_info()

        st.markdown("---")

        # Tabs for different sidebar sections
        tab_selected = st.radio(
            "Navigation",
            ["Files", "Data", "Settings"],
            label_visibility="collapsed",
            horizontal=True
        )

        if tab_selected == "Files":
            sidebar_file_manager()
        elif tab_selected == "Data":
            sidebar_data_view()
        elif tab_selected == "Settings":
            sidebar_settings()

    # Main content - Single page layout
    # Plots section
    main_panel_plots()

    # Circuit Analysis (normal mode only) and Multipoint Table (always)
    if not st.session_state.arrhenius_mode:
        st.markdown("---")
        # Normal mode: Circuit analysis panel full width, then table below
        circuit_analysis_panel()
        # Separator before multipoint table
        st.markdown("---")

    # Multipoint table always shown
    multipoint_analysis_table()


if __name__ == "__main__":
    main()
