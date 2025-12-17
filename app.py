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
from tools.fitting import circuit_fit, calc_rmspe, r2sigma, r2logsigma, effective_capacitance, sort_ecm_by_cap, BlackBoxOptEIS, FittingTimeoutError
from components.plots import (
    create_nyquist_plot, create_bode_plot, create_arrhenius_plot,
    create_mapping_plot_1d, create_mapping_plot_2d, create_mapping_plot_ternary
)
from utils.help_texts import (
    CIRCUIT_MODEL_HELP, WEIGHT_METHOD_HELP, RMSPE_HELP, SUMMARY_TABLE_HELP,
    FIT_SETTINGS_HELP, BAYESIAN_FIT_SETTINGS_HELP, BATCH_FIT_SETTINGS_HELP,
    SAMPLE_INFO_HELP, VF_TOGGLE_HELP, FITTING_RANGE_HELP,
    TEMPERATURE_PATTERN_HELP, FILENAME_PATTERN_HELP,
    ARRHENIUS_PLOT_HELP, FILE_FORMAT_HELP,
    PLOT_SETTINGS_HELP, ARRHENIUS_SETTINGS_HELP, MODE_HELP,
    FREQ_RANGE_HELP, SHOW_RANGE_HELP, DELETE_POINTS_HELP,
    CYCLE_MODE_HELP, R_LABELS_HELP
)
from utils.helpers import format_param_name, parse_temperature_pattern, extract_temp_from_filename
from utils.circuit_presets import (
    PRESET_CIRCUITS, get_preset_circuit_strings, get_preset_initial_guesses,
    get_preset_names, PRESET_CIRCUITS_HELP
)
from utils.igor_export import generate_igor_file, generate_igor_procedure_file
from components.styles import inject_custom_css

# Use impedance library for circuit model
from impedance.models.circuits import CustomCircuit


# Page configuration
st.set_page_config(
    page_title="EIS Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS from external file
inject_custom_css(st)


def parse_delete_points(input_str):
    """Parse delete points input string.

    Supports:
    - Single values: "5"
    - Comma-separated: "1,3,5"
    - Range with hyphen: "5-10" (includes both 5 and 10)
    - Range with colon: "5:10" (includes both 5 and 10)
    - Mixed: "1,3,5-10,15,20:25"

    Returns sorted list of unique indices.
    """
    if not input_str or not input_str.strip():
        return []

    result = set()
    # Split by comma
    parts = input_str.split(',')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check for range notation (- or :)
        if '-' in part or ':' in part:
            # Use regex to split by - or :
            import re
            range_parts = re.split(r'[-:]', part)
            if len(range_parts) == 2:
                start = int(range_parts[0].strip())
                end = int(range_parts[1].strip())
                # Ensure start <= end
                if start > end:
                    start, end = end, start
                result.update(range(start, end + 1))
            else:
                # Invalid format, try as single number
                result.add(int(part))
        else:
            # Single number
            result.add(int(part))

    return sorted(list(result))


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
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = 'Nyquist'  # 'Nyquist', 'Arrhenius', 'Mapping'
    # Keep backward compatibility
    if 'arrhenius_mode' not in st.session_state:
        st.session_state.arrhenius_mode = False
    if 'mapping_mode' not in st.session_state:
        st.session_state.mapping_mode = False
    if 'mapping_dimension' not in st.session_state:
        st.session_state.mapping_dimension = '1D'  # '1D', '2D', or 'Ternary'
    if 'mapping_coords' not in st.session_state:
        st.session_state.mapping_coords = {}  # {filename: {'x': val, 'y': val, 'a': val, 'b': val, 'c': val}}
    if 'mapping_interpolate' not in st.session_state:
        st.session_state.mapping_interpolate = False
    if 'mapping_x_label' not in st.session_state:
        st.session_state.mapping_x_label = 'Position'
    if 'mapping_ternary_labels' not in st.session_state:
        st.session_state.mapping_ternary_labels = ('A', 'B', 'C')
    if 'multipoint_data' not in st.session_state:
        st.session_state.multipoint_data = []
    if 'show_fit' not in st.session_state:
        st.session_state.show_fit = True
    if 'show_all_data' not in st.session_state:
        st.session_state.show_all_data = False
    if 'display_range' not in st.session_state:
        st.session_state.display_range = (0, 70)  # Default display range 0-70
    if 'fitting_range' not in st.session_state:
        st.session_state.fitting_range = (0, 70)  # Default fitting range 0-70
    if 'range_apply_mode' not in st.session_state:
        st.session_state.range_apply_mode = 'global'  # 'global' or 'individual'
    if 'file_display_ranges' not in st.session_state:
        st.session_state.file_display_ranges = {}  # {filename: (start, end)} for individual mode
    if 'file_fitting_ranges' not in st.session_state:
        st.session_state.file_fitting_ranges = {}  # {filename: (start, end)} for individual mode
    # Keep backward compatibility with old freq_range
    if 'freq_range' not in st.session_state:
        st.session_state.freq_range = (0, 70)  # Legacy, will be removed
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
            'tick_font_size': 20,
            'axis_label_font_size': 20,
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
            'arr_legend_font_size': 20,
            # Mapping plot settings
            'map_colorscale': 'Viridis',  # Plotly colorscale
            'map_value_type': 'total_sigma',  # 'total_sigma', 'R1_sigma', 'R2_sigma', etc.
            'map_show_colorbar': True,
            'map_marker_size': 10,
            'map_marker_color': '#1f77b4',
            'map_marker_edge_color': '#000000',
            'map_marker_edge_width': 1,
            'map_line_color': '#1f77b4',
            'map_line_width': 2,
            'map_show_line': True,
        }
    if 'r_labels' not in st.session_state:
        # Labels for R elements (R1, R2, ...) - default: bulk, gb
        st.session_state.r_labels = {'R1': 'bulk', 'R2': 'gb', 'R3': 'R3'}
    if 'arr_fit_targets' not in st.session_state:
        # Which conductivities to show/fit in Arrhenius plot
        st.session_state.arr_fit_targets = ['total']
    if 'hidden_files' not in st.session_state:
        # Set of filenames that are hidden from display
        st.session_state.hidden_files = set()


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
    # Show all data checkbox - use key for immediate update
    show_all_data = st.checkbox(
        "Show all data",
        value=st.session_state.show_all_data,
        help=MODE_HELP['show_all_data'],
        key="show_all_data_checkbox"
    )
    if show_all_data != st.session_state.show_all_data:
        st.session_state.show_all_data = show_all_data
        st.rerun()

    # File list
    if len(st.session_state.files) > 0:
        for i, filename in enumerate(list(st.session_state.files.keys())):
            is_hidden = filename in st.session_state.hidden_files
            is_selected = (filename == st.session_state.selected_file)

            col1, col2, col3 = st.columns([4, 1, 1])

            with col1:
                # Gray out hidden files
                button_type = "primary" if is_selected else "secondary"
                display_name = f"- {filename}" if is_hidden else filename
                if st.button(display_name, key=f"select_{i}", width="stretch", type=button_type):
                    st.session_state.selected_file = filename

            with col2:
                # Show/Hide toggle button
                btn_label = "S" if is_hidden else "H"
                btn_help = "Show on plot" if is_hidden else "Hide from plot"
                if st.button(btn_label, key=f"toggle_vis_{i}", help=btn_help):
                    if is_hidden:
                        st.session_state.hidden_files.discard(filename)
                    else:
                        st.session_state.hidden_files.add(filename)
                    st.rerun()

            with col3:
                if st.button("D", key=f"delete_{i}", help="Delete file"):
                    del st.session_state.files[filename]
                    st.session_state.hidden_files.discard(filename)
                    if st.session_state.selected_file == filename:
                        st.session_state.selected_file = None
                    st.rerun()

        # Save session button
        st.markdown("---")
        if st.button("Save Session", key="save_session_btn", use_container_width=True):
            save_session()

        # Export Igor buttons
        if len(st.session_state.files) > 0:
            igor_str = generate_igor_file(
                files_data=st.session_state.files,
                sample_info=st.session_state.sample_info,
                r_labels=st.session_state.get('r_labels', {'R1': 'bulk', 'R2': 'gb'}),
                export_rows=None
            )
            sample_name = st.session_state.sample_info.get('name', 'eis').strip()
            safe_sample_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in sample_name) if sample_name else 'eis'

            igor_col1, igor_col2 = st.columns(2)
            with igor_col1:
                st.download_button(
                    label="ITX",
                    data=igor_str,
                    file_name=f"eis_{safe_sample_name}_{datetime.now().strftime('%Y%m%d')}.itx",
                    mime="text/plain",
                    use_container_width=True,
                    help="Igor Text File (data + plots)"
                )
            with igor_col2:
                st.download_button(
                    label="IPF",
                    data=generate_igor_procedure_file(),
                    file_name="eis_procedures.ipf",
                    mime="text/plain",
                    use_container_width=True,
                    help="Igor Procedure File (for top axis)"
                )

    # Reset button at bottom
    st.markdown("---")
    if st.button("Reset", key="reset_btn", use_container_width=True):
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
    """Settings tab in sidebar for plot customization - shows settings for current mode only"""
    settings = st.session_state.plot_settings
    analysis_mode = st.session_state.get('analysis_mode', 'Nyquist')

    if analysis_mode == 'Nyquist':
        # ===== Nyquist Mode Settings =====

        # Unit settings
        st.markdown("### Units")
        unit_options = ['Ω', 'kΩ', 'MΩ', 'GΩ']
        current_unit_idx = unit_options.index(settings.get('z_unit', 'Ω'))
        settings['z_unit'] = st.selectbox(
            "Impedance unit",
            unit_options,
            index=current_unit_idx,
            help=PLOT_SETTINGS_HELP['z_unit']
        )

        st.markdown("---")

        # Font settings
        st.markdown("### Font Size")
        col1, col2 = st.columns(2)
        with col1:
            settings['tick_font_size'] = st.number_input(
                "Tick",
                min_value=6, max_value=30,
                value=settings.get('tick_font_size', 20),
                help=PLOT_SETTINGS_HELP['tick_font_size']
            )
        with col2:
            settings['axis_label_font_size'] = st.number_input(
                "Label",
                min_value=8, max_value=30,
                value=settings.get('axis_label_font_size', 20),
                help=PLOT_SETTINGS_HELP['axis_label_font_size']
            )

        st.markdown("---")

        # Marker settings
        st.markdown("### Marker")
        col1, col2 = st.columns(2)
        with col1:
            settings['marker_color'] = st.color_picker(
                "Color",
                value=settings.get('marker_color', '#1f77b4'),
                help=PLOT_SETTINGS_HELP['marker_color']
            )
        with col2:
            symbol_options = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up']
            current_symbol_idx = symbol_options.index(settings.get('marker_symbol', 'circle')) if settings.get('marker_symbol', 'circle') in symbol_options else 0
            settings['marker_symbol'] = st.selectbox(
                "Symbol",
                symbol_options,
                index=current_symbol_idx,
                help=PLOT_SETTINGS_HELP['marker_symbol']
            )

        col1, col2 = st.columns(2)
        with col1:
            settings['marker_size'] = st.number_input(
                "Size",
                min_value=1, max_value=20,
                value=settings.get('marker_size', 6),
                help=PLOT_SETTINGS_HELP['marker_size']
            )
        with col2:
            settings['marker_alpha'] = st.slider(
                "Alpha",
                min_value=0.0, max_value=1.0,
                value=float(settings.get('marker_alpha', 1.0)),
                step=0.1,
                help=PLOT_SETTINGS_HELP['marker_alpha']
            )

        col1, col2 = st.columns(2)
        with col1:
            settings['marker_line_color'] = st.color_picker(
                "Edge color",
                value=settings.get('marker_line_color', '#1f77b4'),
                help=PLOT_SETTINGS_HELP['marker_edge_color']
            )
        with col2:
            settings['marker_line_width'] = st.number_input(
                "Edge width",
                min_value=0, max_value=5,
                value=settings.get('marker_line_width', 0),
                help=PLOT_SETTINGS_HELP['marker_edge_width']
            )

        st.markdown("---")

        # Fit line settings
        st.markdown("### Fit Line")
        col1, col2 = st.columns(2)
        with col1:
            settings['fit_line_color'] = st.color_picker(
                "Color",
                value=settings.get('fit_line_color', '#ff7f0e'),
                key="fit_color",
                help=PLOT_SETTINGS_HELP['fit_line_color']
            )
        with col2:
            settings['fit_line_width'] = st.number_input(
                "Width",
                min_value=1, max_value=5,
                value=settings.get('fit_line_width', 2),
                key="fit_width",
                help=PLOT_SETTINGS_HELP['fit_line_width']
            )

        st.markdown("---")

        # Display settings
        st.markdown("### Display")
        settings['show_zeroline'] = st.checkbox(
            "Show Zero Line",
            value=settings.get('show_zeroline', True),
            help=PLOT_SETTINGS_HELP['show_zeroline']
        )

        st.markdown("---")

        # Legend settings
        st.markdown("### Legend")
        settings['legend_font_size'] = st.number_input(
            "Font Size",
            min_value=6, max_value=20,
            value=settings.get('legend_font_size', 10),
            key="legend_font_size",
            help=PLOT_SETTINGS_HELP['legend_font_size']
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

    elif analysis_mode == 'Arrhenius':
        # ===== Arrhenius Mode Settings =====

        st.markdown("### Font Size")
        col1, col2 = st.columns(2)
        with col1:
            settings['tick_font_size'] = st.number_input(
                "Tick",
                min_value=6, max_value=30,
                value=settings.get('tick_font_size', 20),
                key="arr_tick_font",
                help=PLOT_SETTINGS_HELP['tick_font_size']
            )
        with col2:
            settings['axis_label_font_size'] = st.number_input(
                "Label",
                min_value=8, max_value=30,
                value=settings.get('axis_label_font_size', 20),
                key="arr_label_font",
                help=PLOT_SETTINGS_HELP['axis_label_font_size']
            )

        st.markdown("---")

        st.markdown("### Marker")
        col1, col2 = st.columns(2)
        with col1:
            settings['arr_marker_color'] = st.color_picker(
                "Color",
                value=settings.get('arr_marker_color', '#1f77b4'),
                key="arr_marker_color",
                help=ARRHENIUS_SETTINGS_HELP['arr_marker_color']
            )
        with col2:
            arr_symbol_options = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up']
            arr_current_symbol_idx = arr_symbol_options.index(settings.get('arr_marker_symbol', 'circle')) if settings.get('arr_marker_symbol', 'circle') in arr_symbol_options else 0
            settings['arr_marker_symbol'] = st.selectbox(
                "Symbol",
                arr_symbol_options,
                index=arr_current_symbol_idx,
                key="arr_symbol",
                help=ARRHENIUS_SETTINGS_HELP['arr_marker_symbol']
            )

        col1, col2 = st.columns(2)
        with col1:
            settings['arr_marker_size'] = st.number_input(
                "Size",
                min_value=1, max_value=20,
                value=settings.get('arr_marker_size', 10),
                key="arr_marker_size",
                help=ARRHENIUS_SETTINGS_HELP['arr_marker_size']
            )
        with col2:
            settings['arr_marker_edge_width'] = st.number_input(
                "Edge Width",
                min_value=0, max_value=5,
                value=settings.get('arr_marker_edge_width', 1),
                key="arr_marker_edge_width",
                help=ARRHENIUS_SETTINGS_HELP['arr_marker_edge_width']
            )

        col1, col2 = st.columns(2)
        with col1:
            settings['arr_marker_edge_color'] = st.color_picker(
                "Edge Color",
                value=settings.get('arr_marker_edge_color', '#000000'),
                key="arr_marker_edge_color",
                help=ARRHENIUS_SETTINGS_HELP['arr_marker_edge_color']
            )

        st.markdown("---")

        st.markdown("### Line")
        col1, col2 = st.columns(2)
        with col1:
            settings['arr_line_color'] = st.color_picker(
                "Color",
                value=settings.get('arr_line_color', '#5AA4E6'),
                key="arr_line_color",
                help=ARRHENIUS_SETTINGS_HELP['arr_line_color']
            )
        with col2:
            settings['arr_line_width'] = st.number_input(
                "Width",
                min_value=0, max_value=5,
                value=settings.get('arr_line_width', 2),
                key="arr_line_width",
                help=ARRHENIUS_SETTINGS_HELP['arr_line_width']
            )

        settings['arr_show_line'] = st.checkbox(
            "Show Line",
            value=settings.get('arr_show_line', False),
            key="arr_show_line",
            help=ARRHENIUS_SETTINGS_HELP['arr_show_line']
        )

        st.markdown("---")

        st.markdown("### Legend")
        settings['arr_legend_font_size'] = st.number_input(
            "Font Size",
            min_value=6, max_value=30,
            value=settings.get('arr_legend_font_size', 20),
            key="arr_legend_font_size",
            help=ARRHENIUS_SETTINGS_HELP['arr_legend_font_size']
        )

    elif analysis_mode == 'Mapping':
        # ===== Mapping Mode Settings =====

        st.markdown("### Font Size")
        col1, col2 = st.columns(2)
        with col1:
            settings['tick_font_size'] = st.number_input(
                "Tick",
                min_value=6, max_value=30,
                value=settings.get('tick_font_size', 20),
                key="map_tick_font",
                help="Font size for axis ticks"
            )
        with col2:
            settings['axis_label_font_size'] = st.number_input(
                "Label",
                min_value=8, max_value=30,
                value=settings.get('axis_label_font_size', 20),
                key="map_label_font",
                help="Font size for axis labels"
            )

        st.markdown("---")

        st.markdown("### Scale Options")
        settings['map_use_log_scale'] = st.checkbox(
            "Log Scale (σ)",
            value=settings.get('map_use_log_scale', False),
            key="map_use_log_scale",
            help="Use logarithmic scale for conductivity"
        )

        settings['map_show_zeroline'] = st.checkbox(
            "Show Zero Line",
            value=settings.get('map_show_zeroline', True),
            key="map_show_zeroline",
            help="Display zero lines on axes"
        )

        st.markdown("---")

        st.markdown("### Axis Range")
        st.caption("Leave empty for auto range")
        col1, col2 = st.columns(2)
        with col1:
            x_min = st.text_input("X min", value="", key="map_x_min", help="Minimum X value")
            y_min = st.text_input("Y min", value="", key="map_y_min", help="Minimum Y value")
        with col2:
            x_max = st.text_input("X max", value="", key="map_x_max", help="Maximum X value")
            y_max = st.text_input("Y max", value="", key="map_y_max", help="Maximum Y value")

        # Store axis ranges
        try:
            settings['map_x_range'] = (float(x_min), float(x_max)) if x_min and x_max else None
        except ValueError:
            settings['map_x_range'] = None
        try:
            settings['map_y_range'] = (float(y_min), float(y_max)) if y_min and y_max else None
        except ValueError:
            settings['map_y_range'] = None

        st.markdown("---")

        st.markdown("### Axis Labels")
        col1, col2 = st.columns(2)
        with col1:
            settings['map_x_label'] = st.text_input(
                "X Label",
                value=settings.get('map_x_label', 'X'),
                key="map_x_label",
                help="Label for X axis"
            )
        with col2:
            settings['map_y_label'] = st.text_input(
                "Y Label",
                value=settings.get('map_y_label', 'Y'),
                key="map_y_label",
                help="Label for Y axis"
            )

        st.markdown("---")

        st.markdown("### Marker (1D/Scatter)")
        col1, col2 = st.columns(2)
        with col1:
            settings['map_marker_color'] = st.color_picker(
                "Color",
                value=settings.get('map_marker_color', '#1f77b4'),
                key="map_marker_color",
                help="Marker fill color for 1D plot"
            )
        with col2:
            settings['map_marker_size'] = st.number_input(
                "Size",
                min_value=1, max_value=30,
                value=settings.get('map_marker_size', 10),
                key="map_marker_size",
                help="Marker size"
            )

        col1, col2 = st.columns(2)
        with col1:
            settings['map_marker_edge_color'] = st.color_picker(
                "Edge Color",
                value=settings.get('map_marker_edge_color', '#000000'),
                key="map_marker_edge_color",
                help="Marker edge color"
            )
        with col2:
            settings['map_marker_edge_width'] = st.number_input(
                "Edge Width",
                min_value=0, max_value=5,
                value=settings.get('map_marker_edge_width', 1),
                key="map_marker_edge_width",
                help="Marker edge width"
            )

        settings['map_marker_alpha'] = st.slider(
            "Marker Alpha",
            min_value=0.0, max_value=1.0,
            value=settings.get('map_marker_alpha', 1.0),
            step=0.1,
            key="map_marker_alpha",
            help="Marker transparency"
        )

        st.markdown("---")

        st.markdown("### Line (1D)")
        col1, col2 = st.columns(2)
        with col1:
            settings['map_line_color'] = st.color_picker(
                "Color",
                value=settings.get('map_line_color', '#1f77b4'),
                key="map_line_color",
                help="Line color for 1D plot"
            )
        with col2:
            settings['map_line_width'] = st.number_input(
                "Width",
                min_value=0, max_value=5,
                value=settings.get('map_line_width', 2),
                key="map_line_width",
                help="Line width"
            )

        settings['map_show_line'] = st.checkbox(
            "Show Line",
            value=settings.get('map_show_line', True),
            key="map_show_line",
            help="Show connecting line in 1D plot"
        )

        st.markdown("---")

        st.markdown("### Colorscale (2D/Ternary)")
        # Jet first, then others
        colorscales = ['Jet', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
                      'Blues', 'Reds', 'Greens', 'YlOrRd', 'RdBu', 'Spectral']
        current_colorscale = settings.get('map_colorscale', 'Jet')
        settings['map_colorscale'] = st.selectbox(
            "Colorscale",
            colorscales,
            index=colorscales.index(current_colorscale) if current_colorscale in colorscales else 0,
            key="map_colorscale_setting",
            label_visibility="collapsed",
            help="Color scheme for heatmap/ternary visualization"
        )

        settings['map_show_colorbar'] = st.checkbox(
            "Show Colorbar",
            value=settings.get('map_show_colorbar', True),
            key="map_show_colorbar",
            help="Display the colorbar for value reference"
        )

        st.markdown("---")

        st.markdown("### Ternary Settings")
        settings['map_ternary_fill_size'] = st.number_input(
            "Fill Marker Size",
            min_value=2, max_value=20,
            value=settings.get('map_ternary_fill_size', 8),
            key="map_ternary_fill_size",
            help="Marker size for interpolation fill (larger = more filled)"
        )

        settings['map_use_subscript'] = st.checkbox(
            "Subscript Labels",
            value=settings.get('map_use_subscript', True),
            key="map_use_subscript",
            help="Convert numbers in axis labels to subscript (e.g., Li7 → Li₇)"
        )

    st.session_state.plot_settings = settings


def main_panel_plots():
    """Main panel for plots - Nyquist, Bode, Arrhenius side by side"""
    if len(st.session_state.files) == 0:
        st.info("Upload EIS data files to begin analysis")
        return

    # Determine which files to plot (exclude hidden files)
    hidden_files = st.session_state.get('hidden_files', set())
    if st.session_state.show_all_data:
        selected_for_plot = [f for f in st.session_state.files.keys() if f not in hidden_files]
    elif st.session_state.selected_file and st.session_state.selected_file not in hidden_files:
        selected_for_plot = [st.session_state.selected_file]
    else:
        selected_for_plot = []

    # Get plot settings
    plot_settings = st.session_state.get('plot_settings', {})

    # Get current analysis mode
    analysis_mode = st.session_state.get('analysis_mode', 'Nyquist')

    if analysis_mode == 'Arrhenius':
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
        # Exclude hidden files from Arrhenius plot
        multipoint_data = []
        all_r_names = set()

        for filename in st.session_state.files:
            if filename in hidden_files:
                continue
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
                help=CYCLE_MODE_HELP
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

            # Helper function to get T range for a specific conductivity type
            def get_t_range_for_type(cond_type):
                # Get fit_range and show_range, use intersection (smaller range)
                fit_range_used = fit_range if fit_range else (0, len(multipoint_data) - 1)
                show_range_for_type = show_ranges.get(cond_type, fit_range_used)

                # Calculate the intersection (actual range used)
                actual_start = max(fit_range_used[0], show_range_for_type[0])
                actual_end = min(fit_range_used[1], show_range_for_type[1])

                temps_in_range = []
                for idx, data in enumerate(multipoint_data):
                    if actual_start <= idx <= actual_end:
                        temps_in_range.append(data.get('temperature', 0))

                if temps_in_range:
                    t_min = min(temps_in_range)
                    t_max = max(temps_in_range)
                    return f"{t_min:.0f}–{t_max:.0f}"
                else:
                    return "–"

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
                # Get T range specific to this conductivity type
                t_range_str = get_t_range_for_type(cond_type)
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

    elif analysis_mode == 'Mapping':
        # Mapping mode: Show spatial distribution heatmap
        hidden_files = st.session_state.get('hidden_files', set())

        # Get value type setting
        value_type = plot_settings.get('map_value_type', 'total_sigma')

        # Controls row 1: Dimension, Value type, Colorscale
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.5, 1.5, 1])

        with ctrl_col1:
            dim_options = ['1D', '2D', 'Ternary']
            dim_index = dim_options.index(st.session_state.mapping_dimension) if st.session_state.mapping_dimension in dim_options else 0
            dimension = st.radio(
                "Plot Type",
                dim_options,
                horizontal=True,
                index=dim_index,
                key="mapping_dimension_radio"
            )
            if dimension != st.session_state.mapping_dimension:
                st.session_state.mapping_dimension = dimension
                st.rerun()

        with ctrl_col2:
            # Value type selection
            available_types = ['total_sigma']
            all_r_names = set()
            for fname, fdata in st.session_state.files.items():
                all_r_names.update(fdata.get('r_sigmas', {}).keys())
            for r_name in sorted(all_r_names):
                available_types.append(f'{r_name}_sigma')

            type_labels = {'total_sigma': 'Total σ'}
            for r_name in all_r_names:
                label = st.session_state.r_labels.get(r_name, r_name)
                type_labels[f'{r_name}_sigma'] = f'σ ({label})'

            current_type = plot_settings.get('map_value_type', 'total_sigma')
            if current_type not in available_types:
                current_type = 'total_sigma'

            selected_type = st.selectbox(
                "Value",
                available_types,
                index=available_types.index(current_type),
                format_func=lambda x: type_labels.get(x, x),
                key="mapping_value_type"
            )
            if selected_type != plot_settings.get('map_value_type'):
                st.session_state.plot_settings['map_value_type'] = selected_type
                st.rerun()

        with ctrl_col3:
            colorscales = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
                          'Blues', 'Reds', 'Greens', 'YlOrRd', 'RdBu', 'Spectral']
            current_colorscale = plot_settings.get('map_colorscale', 'Viridis')
            selected_colorscale = st.selectbox(
                "Colorscale",
                colorscales,
                index=colorscales.index(current_colorscale) if current_colorscale in colorscales else 0,
                key="mapping_colorscale"
            )
            if selected_colorscale != plot_settings.get('map_colorscale'):
                st.session_state.plot_settings['map_colorscale'] = selected_colorscale
                st.rerun()

        with ctrl_col4:
            # Interpolation option (for 2D and Ternary)
            if st.session_state.mapping_dimension in ['2D', 'Ternary']:
                interpolate = st.checkbox(
                    "Interpolate",
                    value=st.session_state.mapping_interpolate,
                    key="mapping_interpolate_cb",
                    help="Fill area with interpolated values"
                )
                if interpolate != st.session_state.mapping_interpolate:
                    st.session_state.mapping_interpolate = interpolate

        # Controls row 2: X-axis label (1D) or Ternary axis labels
        if st.session_state.mapping_dimension == '1D':
            label_col1, label_col2 = st.columns([1, 3])
            with label_col1:
                x_label = st.text_input(
                    "X-axis label",
                    value=st.session_state.mapping_x_label,
                    key="mapping_x_label_input"
                )
                if x_label != st.session_state.mapping_x_label:
                    st.session_state.mapping_x_label = x_label

        elif st.session_state.mapping_dimension == 'Ternary':
            tern_col1, tern_col2, tern_col3 = st.columns(3)
            current_labels = st.session_state.mapping_ternary_labels
            with tern_col1:
                label_a = st.text_input("A-axis", value=current_labels[0], key="tern_label_a")
            with tern_col2:
                label_b = st.text_input("B-axis", value=current_labels[1], key="tern_label_b")
            with tern_col3:
                label_c = st.text_input("C-axis", value=current_labels[2], key="tern_label_c")
            new_labels = (label_a, label_b, label_c)
            if new_labels != st.session_state.mapping_ternary_labels:
                st.session_state.mapping_ternary_labels = new_labels

        # Coordinate input section
        st.markdown("---")

        coord_method = st.radio(
            "Coordinate Input",
            ["Manual", "From Pattern", "From File"],
            horizontal=True,
            key="coord_input_method"
        )

        # Manual coordinate input
        if coord_method == "Manual":
            with st.expander("Edit Coordinates", expanded=True):
                if st.session_state.mapping_dimension == '1D':
                    coord_cols = st.columns([3, 1])
                    with coord_cols[0]:
                        st.caption("Filename")
                    with coord_cols[1]:
                        st.caption("X")

                    for i, filename in enumerate(st.session_state.files.keys()):
                        if filename in hidden_files:
                            continue
                        col_fname, col_x = st.columns([3, 1])
                        with col_fname:
                            st.text(filename[:30] + "..." if len(filename) > 30 else filename)
                        with col_x:
                            current_x = st.session_state.mapping_coords.get(filename, {}).get('x', i)
                            new_x = st.number_input(
                                f"X_{filename}",
                                value=float(current_x) if current_x is not None else float(i),
                                label_visibility="collapsed",
                                key=f"map_x_{i}"
                            )
                            if filename not in st.session_state.mapping_coords:
                                st.session_state.mapping_coords[filename] = {}
                            st.session_state.mapping_coords[filename]['x'] = new_x

                elif st.session_state.mapping_dimension == '2D':
                    coord_cols = st.columns([3, 1, 1])
                    with coord_cols[0]:
                        st.caption("Filename")
                    with coord_cols[1]:
                        st.caption("X")
                    with coord_cols[2]:
                        st.caption("Y")

                    for i, filename in enumerate(st.session_state.files.keys()):
                        if filename in hidden_files:
                            continue
                        col_fname, col_x, col_y = st.columns([3, 1, 1])
                        with col_fname:
                            st.text(filename[:25] + "..." if len(filename) > 25 else filename)
                        with col_x:
                            current_x = st.session_state.mapping_coords.get(filename, {}).get('x', 0)
                            new_x = st.number_input(
                                f"X_{filename}",
                                value=float(current_x) if current_x is not None else 0.0,
                                label_visibility="collapsed",
                                key=f"map_x_{i}"
                            )
                            if filename not in st.session_state.mapping_coords:
                                st.session_state.mapping_coords[filename] = {}
                            st.session_state.mapping_coords[filename]['x'] = new_x
                        with col_y:
                            current_y = st.session_state.mapping_coords.get(filename, {}).get('y', 0)
                            new_y = st.number_input(
                                f"Y_{filename}",
                                value=float(current_y) if current_y is not None else 0.0,
                                label_visibility="collapsed",
                                key=f"map_y_{i}"
                            )
                            st.session_state.mapping_coords[filename]['y'] = new_y

                else:  # Ternary
                    coord_cols = st.columns([2.5, 1, 1, 1])
                    with coord_cols[0]:
                        st.caption("Filename")
                    with coord_cols[1]:
                        st.caption("A")
                    with coord_cols[2]:
                        st.caption("B")
                    with coord_cols[3]:
                        st.caption("C")

                    for i, filename in enumerate(st.session_state.files.keys()):
                        if filename in hidden_files:
                            continue
                        col_fname, col_a, col_b, col_c = st.columns([2.5, 1, 1, 1])
                        with col_fname:
                            st.text(filename[:20] + "..." if len(filename) > 20 else filename)
                        with col_a:
                            current_a = st.session_state.mapping_coords.get(filename, {}).get('a', 0.33)
                            new_a = st.number_input(
                                f"A_{filename}",
                                value=float(current_a) if current_a is not None else 0.33,
                                label_visibility="collapsed",
                                key=f"map_a_{i}",
                                step=0.01
                            )
                            if filename not in st.session_state.mapping_coords:
                                st.session_state.mapping_coords[filename] = {}
                            st.session_state.mapping_coords[filename]['a'] = new_a
                        with col_b:
                            current_b = st.session_state.mapping_coords.get(filename, {}).get('b', 0.33)
                            new_b = st.number_input(
                                f"B_{filename}",
                                value=float(current_b) if current_b is not None else 0.33,
                                label_visibility="collapsed",
                                key=f"map_b_{i}",
                                step=0.01
                            )
                            st.session_state.mapping_coords[filename]['b'] = new_b
                        with col_c:
                            current_c = st.session_state.mapping_coords.get(filename, {}).get('c', 0.34)
                            new_c = st.number_input(
                                f"C_{filename}",
                                value=float(current_c) if current_c is not None else 0.34,
                                label_visibility="collapsed",
                                key=f"map_c_{i}",
                                step=0.01
                            )
                            st.session_state.mapping_coords[filename]['c'] = new_c

        elif coord_method == "From Pattern":
            with st.expander("Pattern Settings", expanded=True):
                if st.session_state.mapping_dimension == '1D':
                    x_pattern = st.text_input(
                        "X Pattern",
                        value="[_,0]",
                        help="[separator, index]: Extract X from filename. Example: [_,0] extracts first part split by '_'",
                        key="map_x_pattern"
                    )
                elif st.session_state.mapping_dimension == '2D':
                    pattern_col1, pattern_col2 = st.columns(2)
                    with pattern_col1:
                        x_pattern = st.text_input(
                            "X Pattern",
                            value="[_,0]",
                            help="[separator, index]: Extract X from filename",
                            key="map_x_pattern"
                        )
                    with pattern_col2:
                        y_pattern = st.text_input(
                            "Y Pattern",
                            value="[_,1]",
                            help="[separator, index]: Extract Y from filename",
                            key="map_y_pattern"
                        )
                else:  # Ternary
                    pattern_col1, pattern_col2, pattern_col3 = st.columns(3)
                    with pattern_col1:
                        a_pattern = st.text_input("A Pattern", value="[_,0]", key="map_a_pattern")
                    with pattern_col2:
                        b_pattern = st.text_input("B Pattern", value="[_,1]", key="map_b_pattern")
                    with pattern_col3:
                        c_pattern = st.text_input("C Pattern", value="[_,2]", key="map_c_pattern")

                if st.button("Apply Pattern", key="apply_coord_pattern"):
                    import re
                    try:
                        def parse_pattern(pattern):
                            match = re.match(r'\[(.+),\s*(\d+)\]', pattern)
                            if match:
                                return match.groups()[0], int(match.groups()[1])
                            return '_', 0

                        def extract_value(filename, sep, idx):
                            name_no_ext = filename.rsplit('.', 1)[0]
                            parts = name_no_ext.split(sep)
                            if len(parts) > idx:
                                try:
                                    return float(parts[idx])
                                except ValueError:
                                    nums = re.findall(r'-?\d+\.?\d*', parts[idx])
                                    if nums:
                                        return float(nums[0])
                            return 0

                        for filename in st.session_state.files.keys():
                            if filename in hidden_files:
                                continue
                            if filename not in st.session_state.mapping_coords:
                                st.session_state.mapping_coords[filename] = {}

                            if st.session_state.mapping_dimension == '1D':
                                x_sep, x_idx = parse_pattern(x_pattern)
                                st.session_state.mapping_coords[filename]['x'] = extract_value(filename, x_sep, x_idx)

                            elif st.session_state.mapping_dimension == '2D':
                                x_sep, x_idx = parse_pattern(x_pattern)
                                y_sep, y_idx = parse_pattern(y_pattern)
                                st.session_state.mapping_coords[filename]['x'] = extract_value(filename, x_sep, x_idx)
                                st.session_state.mapping_coords[filename]['y'] = extract_value(filename, y_sep, y_idx)

                            else:  # Ternary
                                a_sep, a_idx = parse_pattern(a_pattern)
                                b_sep, b_idx = parse_pattern(b_pattern)
                                c_sep, c_idx = parse_pattern(c_pattern)
                                st.session_state.mapping_coords[filename]['a'] = extract_value(filename, a_sep, a_idx)
                                st.session_state.mapping_coords[filename]['b'] = extract_value(filename, b_sep, b_idx)
                                st.session_state.mapping_coords[filename]['c'] = extract_value(filename, c_sep, c_idx)

                        st.success("Coordinates extracted from filenames")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error parsing pattern: {e}")

        elif coord_method == "From File":
            with st.expander("Upload Coordinate File", expanded=True):
                if st.session_state.mapping_dimension == 'Ternary':
                    st.caption("CSV format: filename, a, b, c (header row required)")
                else:
                    st.caption("CSV format: filename, x, y (header row required)")

                coord_file = st.file_uploader(
                    "Upload CSV",
                    type=['csv'],
                    key="coord_file_upload"
                )
                if coord_file is not None:
                    try:
                        coord_df = pd.read_csv(coord_file)
                        if 'filename' in coord_df.columns:
                            for _, row in coord_df.iterrows():
                                fname = str(row['filename'])
                                if fname in st.session_state.files:
                                    if fname not in st.session_state.mapping_coords:
                                        st.session_state.mapping_coords[fname] = {}

                                    if 'x' in coord_df.columns:
                                        st.session_state.mapping_coords[fname]['x'] = float(row['x'])
                                    if 'y' in coord_df.columns:
                                        st.session_state.mapping_coords[fname]['y'] = float(row['y'])
                                    if 'a' in coord_df.columns:
                                        st.session_state.mapping_coords[fname]['a'] = float(row['a'])
                                    if 'b' in coord_df.columns:
                                        st.session_state.mapping_coords[fname]['b'] = float(row['b'])
                                    if 'c' in coord_df.columns:
                                        st.session_state.mapping_coords[fname]['c'] = float(row['c'])

                            st.success(f"Loaded coordinates for {len(coord_df)} files")
                            st.rerun()
                        else:
                            st.error("CSV must have 'filename' column")
                    except Exception as e:
                        st.error(f"Error loading CSV: {e}")

        # Build mapping_data with coordinates
        mapping_data = []
        for filename in st.session_state.files:
            if filename in hidden_files:
                continue
            data = st.session_state.files[filename]
            coords = st.session_state.mapping_coords.get(filename, {})

            if value_type == 'total_sigma':
                sigma = data.get('total_sigma')
            else:
                r_name = value_type.replace('_sigma', '')
                sigma = data.get('r_sigmas', {}).get(r_name)

            if sigma is not None:
                entry = {
                    'filename': filename,
                    'x': coords.get('x'),
                    'y': coords.get('y'),
                    'a': coords.get('a'),
                    'b': coords.get('b'),
                    'c': coords.get('c'),
                    'sigma': sigma
                }
                mapping_data.append(entry)

        # Plot
        st.markdown("---")

        # Get settings from plot_settings
        use_log_scale = plot_settings.get('map_use_log_scale', False)
        show_zeroline = plot_settings.get('map_show_zeroline', True)
        x_range = plot_settings.get('map_x_range', None)
        y_range = plot_settings.get('map_y_range', None)
        x_label = plot_settings.get('map_x_label', 'X')
        y_label = plot_settings.get('map_y_label', 'Y')
        use_subscript = plot_settings.get('map_use_subscript', True)

        if len(mapping_data) == 0:
            st.info("No data with conductivity values. Please run circuit fitting first.")
        elif st.session_state.mapping_dimension == '1D':
            has_coords = any(d.get('x') is not None for d in mapping_data)
            if not has_coords:
                st.warning("Please set X coordinates for the data points.")
            else:
                fig = create_mapping_plot_1d(
                    mapping_data, plot_settings, value_type,
                    x_label=st.session_state.mapping_x_label or x_label,
                    x_range=x_range,
                    y_range=y_range,
                    show_zeroline=show_zeroline,
                    use_log_scale=use_log_scale
                )
                st.plotly_chart(fig, use_container_width=True, key="mapping_1d")

        elif st.session_state.mapping_dimension == '2D':
            has_coords = any(d.get('x') is not None and d.get('y') is not None for d in mapping_data)
            if not has_coords:
                st.warning("Please set X and Y coordinates for the data points.")
            else:
                fig = create_mapping_plot_2d(
                    mapping_data, plot_settings, value_type,
                    interpolate=st.session_state.mapping_interpolate,
                    x_label=x_label,
                    y_label=y_label,
                    x_range=x_range,
                    y_range=y_range,
                    show_zeroline=show_zeroline,
                    use_log_scale=use_log_scale
                )
                st.plotly_chart(fig, use_container_width=True, key="mapping_2d")

        else:  # Ternary
            has_coords = any(
                d.get('a') is not None and d.get('b') is not None and d.get('c') is not None
                for d in mapping_data
            )
            if not has_coords:
                st.warning("Please set A, B, C coordinates for the data points.")
            else:
                fig = create_mapping_plot_ternary(
                    mapping_data, plot_settings, value_type,
                    interpolate=st.session_state.mapping_interpolate,
                    axis_labels=st.session_state.mapping_ternary_labels,
                    use_subscript=use_subscript,
                    use_log_scale=use_log_scale
                )
                st.plotly_chart(fig, use_container_width=True, key="mapping_ternary")

        # Data summary table
        if mapping_data:
            st.markdown("---")
            st.caption("Mapping Data Summary")
            summary_rows = []
            for d in mapping_data:
                row = {'Filename': d['filename']}

                if st.session_state.mapping_dimension == '1D':
                    row['X'] = f"{d.get('x', '-'):.2f}" if isinstance(d.get('x'), (int, float)) else '-'
                elif st.session_state.mapping_dimension == '2D':
                    row['X'] = f"{d.get('x', '-'):.2f}" if isinstance(d.get('x'), (int, float)) else '-'
                    row['Y'] = f"{d.get('y', '-'):.2f}" if isinstance(d.get('y'), (int, float)) else '-'
                else:  # Ternary
                    row['A'] = f"{d.get('a', '-'):.3f}" if isinstance(d.get('a'), (int, float)) else '-'
                    row['B'] = f"{d.get('b', '-'):.3f}" if isinstance(d.get('b'), (int, float)) else '-'
                    row['C'] = f"{d.get('c', '-'):.3f}" if isinstance(d.get('c'), (int, float)) else '-'

                row['σ (S/cm)'] = f"{d['sigma']:.2e}" if d['sigma'] else '-'
                summary_rows.append(row)

            if st.session_state.mapping_dimension == '1D':
                cols_order = ['Filename', 'X', 'σ (S/cm)']
            elif st.session_state.mapping_dimension == '2D':
                cols_order = ['Filename', 'X', 'Y', 'σ (S/cm)']
            else:
                cols_order = ['Filename', 'A', 'B', 'C', 'σ (S/cm)']

            summary_df = pd.DataFrame(summary_rows)
            if len(summary_df) > 0:
                summary_df = summary_df[cols_order]
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

    else:  # analysis_mode == 'Nyquist'
        # Nyquist mode: Show Nyquist and Bode plots

        # Controls BEFORE plots (so values are updated before plotting)
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 2, 1.5, 1])

        with ctrl_col1:
            show_fit = st.checkbox(
                "Show Fitted",
                value=st.session_state.show_fit,
                key="show_fit_checkbox",
                help=MODE_HELP['show_fit']
            )
            st.session_state.show_fit = show_fit

            show_legend = st.checkbox(
                "Show Legend",
                value=st.session_state.show_legend,
                key="show_legend_checkbox",
                help=MODE_HELP['show_legend']
            )
            st.session_state.show_legend = show_legend

            highlight_freq = st.checkbox(
                "Highlight Freq",
                value=st.session_state.highlight_freq,
                help=MODE_HELP['highlight_freq'],
                key="highlight_freq_checkbox"
            )
            st.session_state.highlight_freq = highlight_freq

        # Range sliders (Display Range and Fitting Range)
        with ctrl_col2:
            # Get selected file data for range limits
            display_range = st.session_state.display_range
            fitting_range = st.session_state.fitting_range
            n_points = 71  # Default

            if st.session_state.selected_file and st.session_state.selected_file in st.session_state.files:
                data = st.session_state.files[st.session_state.selected_file]
                freq_data = data['freq']
                n_points = len(freq_data)

                # For individual mode, get file-specific ranges
                if st.session_state.range_apply_mode == 'individual':
                    file_key = st.session_state.selected_file
                    if file_key in st.session_state.file_display_ranges:
                        display_range = st.session_state.file_display_ranges[file_key]
                    if file_key in st.session_state.file_fitting_ranges:
                        fitting_range = st.session_state.file_fitting_ranges[file_key]

            if n_points > 1:
                # Fitting range apply mode selector (Global vs Individual)
                apply_mode = st.radio(
                    "Fitting range apply mode",
                    options=['global', 'individual'],
                    format_func=lambda x: 'Global (all files)' if x == 'global' else 'Individual (per file)',
                    index=0 if st.session_state.range_apply_mode == 'global' else 1,
                    horizontal=True,
                    key="range_apply_mode_radio",
                    label_visibility="visible"
                )
                st.session_state.range_apply_mode = apply_mode

                # Display Range slider
                st.caption("Display Range (index)")
                current_display = display_range or (0, min(70, n_points - 1))
                current_display = (
                    max(0, min(current_display[0], n_points - 1)),
                    max(0, min(current_display[1], n_points - 1))
                )
                display_range = st.slider(
                    "display_range",
                    min_value=0,
                    max_value=n_points - 1,
                    value=current_display,
                    label_visibility="collapsed",
                    key="display_range_slider"
                )

                # Fitting Range slider
                st.caption("Fitting Range (index)")
                current_fitting = fitting_range or (0, min(70, n_points - 1))
                current_fitting = (
                    max(0, min(current_fitting[0], n_points - 1)),
                    max(0, min(current_fitting[1], n_points - 1))
                )
                fitting_range = st.slider(
                    "fitting_range",
                    min_value=0,
                    max_value=n_points - 1,
                    value=current_fitting,
                    label_visibility="collapsed",
                    key="fitting_range_slider"
                )

                # Store ranges based on apply mode
                if apply_mode == 'global':
                    st.session_state.display_range = display_range
                    st.session_state.fitting_range = fitting_range
                    # Also update legacy freq_range for backward compatibility
                    st.session_state.freq_range = fitting_range
                else:
                    # Individual mode: store per-file
                    if st.session_state.selected_file:
                        st.session_state.file_display_ranges[st.session_state.selected_file] = display_range
                        st.session_state.file_fitting_ranges[st.session_state.selected_file] = fitting_range

        # Delete points
        with ctrl_col3:
            st.caption("Delete Points (index)")
            delete_input = st.text_input(
                "delete_points",
                value=",".join(map(str, st.session_state.deleted_points)) if st.session_state.deleted_points else "",
                placeholder="e.g., 0,5,10 or 5:10",
                label_visibility="collapsed",
                key="delete_points_input",
                help=DELETE_POINTS_HELP
            )
            # Parse delete points (supports comma-separated, ranges with - or :)
            deleted_points = []
            if delete_input:
                try:
                    deleted_points = parse_delete_points(delete_input)
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

        # Get range settings for plotting
        # For plotting, use display_range; fitting_range is used only for fitting
        if st.session_state.range_apply_mode == 'global':
            current_display_range = st.session_state.display_range
            current_fitting_range = st.session_state.fitting_range
        else:
            # Individual mode
            file_key = st.session_state.selected_file
            current_display_range = st.session_state.file_display_ranges.get(file_key, st.session_state.display_range)
            current_fitting_range = st.session_state.file_fitting_ranges.get(file_key, st.session_state.fitting_range)

        # Now render plots with updated values
        col1, col2 = st.columns(2)

        with col1:
            fig_nyquist = create_nyquist_plot(
                st.session_state.files, selected_for_plot,
                show_fit,
                show_legend,
                highlight_freq,
                plot_settings,
                current_display_range,  # Use display_range for plotting
                deleted_points,
                fitting_range=current_fitting_range  # Pass fitting_range for fit curve display
            )
            st.plotly_chart(fig_nyquist, use_container_width=True, key="nyquist")

        with col2:
            fig_bode = create_bode_plot(
                st.session_state.files, selected_for_plot,
                show_fit,
                show_legend,
                current_display_range,  # Use display_range for plotting
                plot_settings,
                deleted_points,
                fitting_range=current_fitting_range  # Pass fitting_range for fit curve display
            )
            st.plotly_chart(fig_bode, use_container_width=True, key="bode")


def circuit_analysis_panel():
    """Circuit analysis panel with new layout"""
    if not st.session_state.selected_file or st.session_state.selected_file not in st.session_state.files:
        st.info("Select a file from the sidebar to perform circuit analysis")
        return

    filename = st.session_state.selected_file
    data = st.session_state.files[filename]

    # Use preset circuits from module
    preset_circuit_strings = get_preset_circuit_strings()
    preset_initial_guesses = get_preset_initial_guesses()

    # Top row: Circuit preset, Circuit String, and Weight Method
    col1, col2, col3 = st.columns([1, 2, 1])

    # Track previous preset to detect changes
    if 'prev_circuit_preset' not in st.session_state:
        st.session_state.prev_circuit_preset = "Custom"

    with col1:
        preset_choice = st.selectbox(
            "Equivalent Circuit",
            get_preset_names(),
            index=0,
            key="circuit_preset",
            help=PRESET_CIRCUITS_HELP
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
            # Clear previous initial guesses to avoid parameter count mismatch
            if 'global_initial_guess' in st.session_state:
                del st.session_state['global_initial_guess']
            if filename in st.session_state.files:
                st.session_state.files[filename]['initial_guess'] = None
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

    # Track circuit model changes and reset initial_guess if model changed
    if 'prev_circuit_model' not in st.session_state:
        st.session_state.prev_circuit_model = circuit_model
    elif st.session_state.prev_circuit_model != circuit_model:
        st.session_state.prev_circuit_model = circuit_model
        # Clear previous initial guesses to avoid parameter count mismatch
        if 'global_initial_guess' in st.session_state:
            del st.session_state['global_initial_guess']
        if filename in st.session_state.files:
            st.session_state.files[filename]['initial_guess'] = None
        st.session_state.param_fixed = {}

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

    # Button row: Fit, MC Fit, Bayesian, Auto Fit, Batch, Auto-Batch
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5, btn_col6, btn_col7 = st.columns(7)

    with btn_col1:
        fit_clicked = st.button("Fit", width="stretch", type="primary", help="Fit circuit with current initial values")

    with btn_col2:
        mc_fit_help = """**Monte Carlo Fit**
Repeats Add Noise → Fit multiple times to escape local minima.

- Uses current initial values as starting point
- Adds random noise and fits repeatedly
- Keeps the best result (lowest RMSPE)
- Set iterations in Fit Settings"""
        mc_fit_clicked = st.button("MC Fit", width="stretch", help=mc_fit_help)

    with btn_col3:
        bayesian_fit_help = """**Bayesian Fit**
Uses Bayesian optimization (Optuna) to find optimal parameters.

- Explores parameter space efficiently
- Good for finding global optimum
- Set trials/timeout in Bayesian Fit Settings

**Note:** Requires 'optuna' package."""
        bayesian_fit_clicked = st.button("Bayesian", width="stretch", help=bayesian_fit_help)

    with btn_col4:
        auto_fit_help = """**Auto Fit**
Combines Bayesian + Monte Carlo fitting.

1. First runs Bayesian optimization
2. Then refines with Monte Carlo iterations
- Best for comprehensive parameter search"""
        auto_fit_clicked = st.button("Auto Fit", width="stretch", help=auto_fit_help)

    with btn_col5:
        batch_fit_help = """**Batch Fit**
Fits selected files using current initial values.

- Select files in Batch Fit Settings
- Uses current parameters as initial guess
- Propagates fit results to next file"""
        batch_fit_clicked = st.button("Batch", width="stretch", help=batch_fit_help)

    with btn_col6:
        mc_batch_help = """**MC-Batch Fit**
Monte Carlo Batch fitting for selected files.

- Uses existing fit result as initial guess if available
- Otherwise uses previous file's result
- Good for refining batch fits"""
        mc_batch_clicked = st.button("MC-Batch", width="stretch", help=mc_batch_help)

    with btn_col7:
        auto_batch_help = """**Auto-Batch Fit**
Combines Auto Fit (Bayesian + MC) for each file.

- Uses Bayesian + MC for each file
- Select files in Batch Fit Settings
- Best for initial fitting of many files"""
        auto_batch_clicked = st.button("Auto-Batch", width="stretch", help=auto_batch_help)

    # Fit Settings (expandable)
    with st.expander("Fit Settings", expanded=False):
        # Initialize fit settings in session state
        if 'fit_settings' not in st.session_state:
            st.session_state.fit_settings = {
                'maxfev': 10000,
                'ftol': 1e-10,
                'xtol': 1e-10,
                'timeout': 5,  # seconds
                'noise_percent': 10,  # ±N% noise for Add Noise button
                'mc_iterations': 100,  # Monte Carlo iterations
                'keep_better': True,
                'global_opt': False
            }

        fit_settings = st.session_state.fit_settings

        # Row 1: Convergence settings
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
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
        with col_f4:
            fit_settings['timeout'] = st.number_input(
                "Timeout (sec)",
                min_value=1, max_value=120, value=fit_settings.get('timeout', 5), step=1,
                help=FIT_SETTINGS_HELP['timeout']
            )

        # Row 2: Options
        col_f5, col_f6, col_f7, col_f8 = st.columns(4)
        with col_f5:
            fit_settings['noise_percent'] = st.number_input(
                "Noise (±%)",
                min_value=1, max_value=100, value=fit_settings.get('noise_percent', 10), step=1,
                help=FIT_SETTINGS_HELP['noise_percent']
            )
        with col_f6:
            fit_settings['mc_iterations'] = st.number_input(
                "MC Iterations",
                min_value=5, max_value=200, value=fit_settings.get('mc_iterations', 20), step=5,
                help=FIT_SETTINGS_HELP['mc_iterations']
            )
        with col_f7:
            fit_settings['keep_better'] = st.checkbox(
                "Keep better result",
                value=fit_settings['keep_better'],
                help=FIT_SETTINGS_HELP['keep_better']
            )
        with col_f8:
            fit_settings['global_opt'] = st.checkbox(
                "Global optimization",
                value=fit_settings['global_opt'],
                help=FIT_SETTINGS_HELP['global_opt']
            )

        st.session_state.fit_settings = fit_settings

    # Bayesian Fit Settings (expandable)
    with st.expander("Bayesian Fit Settings", expanded=False):
        # Initialize bayesian_fit settings in session state
        if 'bayesian_fit_settings' not in st.session_state:
            st.session_state.bayesian_fit_settings = {
                'n_trials': 100,
                'timeout': 30,
                'early_stop_rmspe': 3.0,
                'early_stop_patience': 20,  # Stop if no improvement for N trials
                'log_step': 0.5,
                'r_min': 1e0,
                'r_max': 1e8,
                'cpe_q_min': 1e-12,
                'cpe_q_max': 1e-4,
                'use_current_model': False,  # Default: optimize model selection
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

        settings = st.session_state.bayesian_fit_settings

        # Row 1: Trials, Timeout, Convergence, Early Stop, Log Step
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        with col_s1:
            settings['n_trials'] = st.number_input(
                "Max trials",
                min_value=10, max_value=1000, value=settings['n_trials'], step=10,
                help=BAYESIAN_FIT_SETTINGS_HELP['n_trials']
            )
        with col_s2:
            settings['timeout'] = st.number_input(
                "Timeout (sec)",
                min_value=1, max_value=300, value=settings['timeout'], step=1,
                help=BAYESIAN_FIT_SETTINGS_HELP['timeout']
            )
        with col_s3:
            settings['early_stop_rmspe'] = st.number_input(
                "Convergence (%)",
                min_value=0.1, max_value=20.0, value=settings['early_stop_rmspe'], step=0.5,
                help=BAYESIAN_FIT_SETTINGS_HELP['early_stop_rmspe']
            )
        with col_s4:
            settings['early_stop_patience'] = st.number_input(
                "Early Stop",
                min_value=5, max_value=200, value=settings.get('early_stop_patience', 20), step=5,
                help=BAYESIAN_FIT_SETTINGS_HELP['early_stop_patience']
            )
        with col_s5:
            settings['log_step'] = st.number_input(
                "Log step",
                min_value=0.0, max_value=2.0, value=settings.get('log_step', 0.5), step=0.1,
                format="%.1f",
                help=BAYESIAN_FIT_SETTINGS_HELP['log_step']
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
            # Calculate max Z.real from current data for R range upper limit
            max_z_real = None
            if filename and filename in st.session_state.files:
                data = st.session_state.files[filename]
                Z = data.get('Z')
                if Z is not None:
                    # Apply freq_range if set
                    freq_range = st.session_state.get('freq_range')
                    if freq_range:
                        start_idx, end_idx = freq_range
                        Z_fit = Z[start_idx:end_idx + 1]
                    else:
                        Z_fit = Z
                    max_z_real = np.max(np.real(Z_fit))

            if max_z_real is not None:
                max_r_limit = max_z_real * 1.5
                max_r_exp = int(np.ceil(np.log10(max_r_limit))) if max_r_limit > 0 else 8
                st.markdown(f"**R range (10^x Ω)** — *max(Z') = {max_z_real:.1e} → R上限 ≈ 10^{max_r_exp}*")
            else:
                max_r_exp = None
                st.markdown("**R range (10^x Ω)**")

            for r_elem in r_elements_sorted:
                r_key = f'{r_elem.lower()}_range'
                # Initialize if not exists
                if r_key not in settings:
                    settings[r_key] = (0, 8)
                current_range = settings[r_key]

                # Cap upper limit based on data if available
                if max_r_exp is not None:
                    capped_max = min(int(current_range[1]), max_r_exp)
                    if capped_max < current_range[0]:
                        capped_max = int(current_range[1])
                    display_range = (int(current_range[0]), capped_max)
                else:
                    display_range = (int(current_range[0]), int(current_range[1]))

                settings[r_key] = st.slider(
                    f"{r_elem}",
                    min_value=-3, max_value=12,
                    value=display_range,
                    step=1,
                    key=f"auto_fit_{r_key}",
                    help=f"Search range for {r_elem} (10^min to 10^max Ω). Upper limit capped by max(Z.real) × 1.5"
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
        st.markdown("**Circuit model optimization**")
        settings['use_current_model'] = st.checkbox(
            "Fix to current model (uncheck to optimize model selection)",
            value=settings['use_current_model'],
            help=BAYESIAN_FIT_SETTINGS_HELP['use_current_model']
        )

        if not settings['use_current_model']:
            # Multi-select for circuit models
            available_models = [
                # Without series R0
                'p(R1,CPE1)-CPE2',
                'p(R1,CPE1)-p(R2,CPE2)-CPE3',
                'p(R1,CPE1)',
                'p(R1,CPE1)-p(R2,CPE2)',
                'p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)',
                # With series R0
                'R0-p(R1,CPE1)-CPE2',
                'R0-p(R1,CPE1)-p(R2,CPE2)-CPE3',
                'R0-p(R1,CPE1)',
                'R0-p(R1,CPE1)-p(R2,CPE2)',
                'R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)',
                # Warburg element
                'p(R1,CPE1)-Wo1',
                'p(R1,CPE1)-p(R2,CPE2)-Wo1',
                'R0-p(R1,CPE1)-Wo1',
            ]
            settings['model_list'] = st.multiselect(
                "Circuit models to optimize",
                available_models,
                default=settings['model_list'],
                help=BAYESIAN_FIT_SETTINGS_HELP['model_list']
            )

        st.session_state.bayesian_fit_settings = settings

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
                timeout = fit_settings.get('timeout', 5)
                keep_better = fit_settings.get('keep_better', True)
                global_opt = fit_settings.get('global_opt', False)

                # Store existing RMSPE for keep_better comparison
                existing_rmspe = data.get('rmspe')

                # Apply fitting range if set
                # Use fitting_range for fitting (not display_range)
                if st.session_state.range_apply_mode == 'global':
                    fitting_range = st.session_state.fitting_range
                else:
                    # Individual mode: get per-file fitting range
                    fitting_range = st.session_state.file_fitting_ranges.get(
                        filename, st.session_state.fitting_range
                    )

                if fitting_range:
                    start_idx, end_idx = fitting_range
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
                        timeout=timeout,
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
                        timeout=timeout,
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
                # Only compare if circuit model is the same
                should_update = True
                existing_circuit_model = data.get('circuit_model')
                if keep_better and existing_rmspe is not None and existing_circuit_model == circuit_model:
                    if rmspe >= existing_rmspe:
                        # New fit is worse or equal, keep existing result
                        st.info(f"Previous better fitting is preserved (RMSPE: {existing_rmspe*100:.2f}% < {rmspe*100:.2f}%)")
                        should_update = False

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

            except FittingTimeoutError as e:
                st.warning(f"Fitting timed out after {timeout} seconds. Try increasing timeout or adjusting initial values.")
            except Exception as e:
                st.error(f"Fitting failed: {str(e)}")

    # Handle Bayesian Fit button
    if bayesian_fit_clicked:
        # Create progress elements first (outside spinner)
        status_text = st.empty()
        progress_bar = st.progress(0)
        status_text.text("Bayesian Fit (Optuna optimization)...")

        try:
            freq = data['freq']
            Z = data['Z']

            # Apply fitting range if set
            # Use fitting_range for fitting (not display_range)
            if st.session_state.range_apply_mode == 'global':
                fitting_range = st.session_state.fitting_range
            else:
                # Individual mode: get per-file fitting range
                fitting_range = st.session_state.file_fitting_ranges.get(
                    filename, st.session_state.fitting_range
                )

            if fitting_range:
                start_idx, end_idx = fitting_range
                freq_fit = freq[start_idx:end_idx + 1]
                Z_fit_data = Z[start_idx:end_idx + 1]
            else:
                freq_fit = freq
                Z_fit_data = Z

            # Get Bayesian Fit settings
            bayesian_settings = st.session_state.get('bayesian_fit_settings', {})
            n_trials = bayesian_settings.get('n_trials', 100)
            timeout = bayesian_settings.get('timeout', 30)
            early_stop_rmspe = bayesian_settings.get('early_stop_rmspe', 3.0) / 100  # Convert % to decimal
            early_stop_patience = bayesian_settings.get('early_stop_patience', 20)
            log_step = bayesian_settings.get('log_step', 0.5)
            r_range = (bayesian_settings.get('r_min', 1e0), bayesian_settings.get('r_max', 1e8))
            cpe_q_range = (bayesian_settings.get('cpe_q_min', 1e-12), bayesian_settings.get('cpe_q_max', 1e-4))

            # Get fit settings
            fit_settings = st.session_state.get('fit_settings', {})
            fit_timeout = fit_settings.get('timeout', 5)
            fit_maxfev = fit_settings.get('maxfev', 10000)

            # Build individual R ranges from settings
            r_ranges = {}
            for r_name in ['R1', 'R2', 'R3']:
                r_key = f'{r_name.lower()}_range'
                if r_key in bayesian_settings:
                    r_min_exp, r_max_exp = bayesian_settings[r_key]
                    r_ranges[r_name] = (10 ** r_min_exp, 10 ** r_max_exp)

            # Build individual CPE Q ranges from settings
            cpe_q_ranges = {}
            for cpe_name in ['CPE1', 'CPE2', 'CPE3']:
                cpe_key = f'{cpe_name.lower()}_q_range'
                if cpe_key in bayesian_settings:
                    q_min_exp, q_max_exp = bayesian_settings[cpe_key]
                    cpe_q_ranges[cpe_name] = (10 ** q_min_exp, 10 ** q_max_exp)

            # Determine model list
            use_current_model = bayesian_settings.get('use_current_model', True)
            if use_current_model:
                model_list = [circuit_model] if circuit_model else None
            else:
                model_list = bayesian_settings.get('model_list', None)
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
                early_stop_patience=early_stop_patience,
                log_step=log_step,
                r_range=r_range,
                cpe_q_range=cpe_q_range,
                r_ranges=r_ranges,
                cpe_q_ranges=cpe_q_ranges,
                fit_timeout=fit_timeout,
                maxfev=fit_maxfev
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

            # Debug: check what fit_best returned
            if popt is None:
                st.error(f"Debug: popt is None. best_model={best_model}, best_params={optimizer.best_params}")
            elif len(popt) == 0:
                st.error(f"Debug: popt is empty. best_model={best_model}, best_params={optimizer.best_params}")

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

                st.success(f"Bayesian Fit completed! RMSPE: {rmspe*100:.1f}%")
                st.rerun()
            else:
                st.error("Bayesian Fit failed to find a valid solution")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Bayesian Fit failed: {str(e)}")

    # Handle Monte Carlo Fit button
    if mc_fit_clicked:
        # Create progress elements
        status_text = st.empty()
        progress_bar = st.progress(0)
        status_text.text("Monte Carlo Fit...")

        try:
            freq = data['freq']
            Z = data['Z']

            # Get fit settings
            fit_settings = st.session_state.get('fit_settings', {})
            maxfev = fit_settings.get('maxfev', 10000)
            ftol = fit_settings.get('ftol', 1e-10)
            xtol = fit_settings.get('xtol', 1e-10)
            timeout = fit_settings.get('timeout', 5)
            noise_percent = fit_settings.get('noise_percent', 10) / 100.0
            mc_iterations = fit_settings.get('mc_iterations', 20)

            # Apply fitting range if set
            # Use fitting_range for fitting (not display_range)
            if st.session_state.range_apply_mode == 'global':
                fitting_range = st.session_state.fitting_range
            else:
                # Individual mode: get per-file fitting range
                fitting_range = st.session_state.file_fitting_ranges.get(
                    filename, st.session_state.fitting_range
                )

            if fitting_range:
                start_idx, end_idx = fitting_range
                freq_fit = freq[start_idx:end_idx + 1]
                Z_fit_data = Z[start_idx:end_idx + 1]
            else:
                freq_fit = freq
                Z_fit_data = Z

            # Get current initial guess
            from impedance.models.circuits.fitting import calculateCircuitLength
            n_params = calculateCircuitLength(circuit_model)
            initial_guess = data.get('initial_guess')

            if initial_guess is None or len(initial_guess) != n_params:
                st.error("Please set initial values first before running Monte Carlo Fit")
            else:
                # Get param names
                temp_circuit = CustomCircuit(circuit_model, initial_guess=initial_guess)
                temp_param_names, _ = temp_circuit.get_param_names()
                param_fixed = st.session_state.get('param_fixed', {})

                best_rmspe = float('inf')
                best_popt = None
                best_perror = None

                for iteration in range(mc_iterations):
                    progress = (iteration + 1) / mc_iterations
                    progress_bar.progress(progress)
                    status_text.text(f"MC Fit iteration {iteration + 1}/{mc_iterations}, Best RMSPE: {best_rmspe*100:.2f}%")

                    # Add noise to variable parameters
                    noisy_guess = list(initial_guess) if iteration == 0 else list(best_popt) if best_popt is not None else list(initial_guess)

                    if iteration > 0:  # Don't add noise on first iteration
                        for i, pname in enumerate(temp_param_names):
                            is_fixed = param_fixed.get(f"fixed_{i}", False)
                            if not is_fixed:
                                # Add random noise
                                noise_factor = 1 + np.random.uniform(-noise_percent, noise_percent)
                                new_val = noisy_guess[i] * noise_factor
                                # Clamp alpha to max 1.0
                                if 'CPE' in pname or 'La' in pname:
                                    if new_val > 1.0:
                                        new_val = 1.0
                                noisy_guess[i] = new_val

                    # Build constants dict for fixed parameters
                    constants = {}
                    variable_initial_guess = []
                    fixed_indices = []

                    for i, pname in enumerate(temp_param_names):
                        is_fixed = param_fixed.get(f"fixed_{i}", False)
                        if is_fixed:
                            constants[pname] = noisy_guess[i]
                            fixed_indices.append(i)
                        else:
                            variable_initial_guess.append(noisy_guess[i])

                    try:
                        if len(variable_initial_guess) == 0:
                            popt = np.array(noisy_guess)
                            perror = np.zeros(n_params)
                        elif len(constants) == 0:
                            popt, perror = circuit_fit(
                                freq_fit, Z_fit_data,
                                circuit_model,
                                noisy_guess,
                                constants={},
                                weight_method=weight_method,
                                timeout=timeout,
                                maxfev=maxfev,
                                ftol=ftol,
                                xtol=xtol
                            )
                        else:
                            popt_variable, perror_variable = circuit_fit(
                                freq_fit, Z_fit_data,
                                circuit_model,
                                variable_initial_guess,
                                constants=constants,
                                weight_method=weight_method,
                                timeout=timeout,
                                maxfev=maxfev,
                                ftol=ftol,
                                xtol=xtol
                            )

                            # Reconstruct full parameter arrays
                            popt = []
                            perror = []
                            var_idx = 0
                            for i in range(n_params):
                                if i in fixed_indices:
                                    popt.append(noisy_guess[i])
                                    perror.append(0.0)
                                else:
                                    popt.append(popt_variable[var_idx])
                                    perror.append(perror_variable[var_idx] if perror_variable is not None else 0.0)
                                    var_idx += 1
                            popt = np.array(popt)
                            perror = np.array(perror)

                        # Calculate RMSPE
                        circuit = CustomCircuit(circuit_model, initial_guess=list(popt))
                        circuit.parameters_ = popt
                        Z_fit_iter = circuit.predict(freq_fit)
                        rmspe = calc_rmspe(Z_fit_data, Z_fit_iter)

                        if rmspe < best_rmspe:
                            best_rmspe = rmspe
                            best_popt = popt
                            best_perror = perror

                    except (FittingTimeoutError, Exception):
                        continue  # Skip failed iterations

                progress_bar.empty()
                status_text.empty()

                if best_popt is not None:
                    # Create circuit for full prediction
                    circuit = CustomCircuit(circuit_model, initial_guess=list(best_popt))
                    circuit.parameters_ = best_popt
                    circuit.conf_ = best_perror
                    Z_fit = circuit.predict(freq)

                    # Sort by effective capacitance
                    param_names, _ = circuit.get_param_names()
                    sorted_result = sort_ecm_by_cap(best_popt, best_perror, param_names)
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
                            sorted_popt.append(best_popt[idx])
                            sorted_perror.append(best_perror[idx] if best_perror is not None else 0.0)

                    sorted_popt = np.array(sorted_popt)
                    sorted_perror = np.array(sorted_perror)

                    # Update circuit with sorted parameters
                    circuit.parameters_ = sorted_popt
                    circuit.conf_ = sorted_perror

                    # Store results
                    st.session_state.files[filename]['Z_fit'] = Z_fit
                    st.session_state.files[filename]['circuit_model'] = circuit_model
                    st.session_state.files[filename]['circuit_params'] = sorted_popt
                    st.session_state.files[filename]['circuit_conf'] = sorted_perror
                    st.session_state.files[filename]['circuit_object'] = circuit
                    st.session_state.files[filename]['rmspe'] = best_rmspe

                    # Update initial guess
                    st.session_state.files[filename]['initial_guess'] = list(sorted_popt)
                    st.session_state['global_initial_guess'] = list(sorted_popt)

                    # Update widget values
                    for i, val in enumerate(sorted_popt):
                        widget_key = f"init_{i}"
                        st.session_state[widget_key] = f"{val:.2e}"

                    # Calculate conductivity
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

                    st.session_state.files[filename]['effective_caps'] = effective_caps
                    st.session_state.files[filename]['sorted_params'] = sorted_result
                    st.session_state.files[filename]['total_sigma'] = sigma_total
                    st.session_state.files[filename]['total_R'] = R_total
                    st.session_state.files[filename]['r_values'] = r_values
                    st.session_state.files[filename]['r_sigmas'] = r_sigmas

                    st.success(f"Monte Carlo Fit completed! RMSPE: {best_rmspe*100:.2f}%")
                    st.rerun()
                else:
                    st.error("Monte Carlo Fit failed to find a valid solution")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Monte Carlo Fit failed: {str(e)}")

    # Handle Auto Fit button (Bayesian + Monte Carlo)
    if auto_fit_clicked:
        # Create progress elements
        status_text = st.empty()
        progress_bar = st.progress(0)
        status_text.text("Auto Fit: Running Bayesian optimization...")

        try:
            freq = data['freq']
            Z = data['Z']

            # Apply fitting range if set
            # Use fitting_range for fitting (not display_range)
            if st.session_state.range_apply_mode == 'global':
                fitting_range = st.session_state.fitting_range
            else:
                # Individual mode: get per-file fitting range
                fitting_range = st.session_state.file_fitting_ranges.get(
                    filename, st.session_state.fitting_range
                )

            if fitting_range:
                start_idx, end_idx = fitting_range
                freq_fit = freq[start_idx:end_idx + 1]
                Z_fit_data = Z[start_idx:end_idx + 1]
            else:
                freq_fit = freq
                Z_fit_data = Z

            # Get Bayesian Fit settings
            bayesian_settings = st.session_state.get('bayesian_fit_settings', {})
            n_trials = bayesian_settings.get('n_trials', 100)
            bayesian_timeout = bayesian_settings.get('timeout', 30)
            early_stop_rmspe = bayesian_settings.get('early_stop_rmspe', 3.0) / 100
            early_stop_patience = bayesian_settings.get('early_stop_patience', 20)
            log_step = bayesian_settings.get('log_step', 0.5)
            r_range = (bayesian_settings.get('r_min', 1e0), bayesian_settings.get('r_max', 1e8))
            cpe_q_range = (bayesian_settings.get('cpe_q_min', 1e-12), bayesian_settings.get('cpe_q_max', 1e-4))

            # Get fit settings
            fit_settings = st.session_state.get('fit_settings', {})
            fit_timeout = fit_settings.get('timeout', 5)
            fit_maxfev = fit_settings.get('maxfev', 10000)

            # Build individual R ranges
            r_ranges = {}
            for r_name in ['R1', 'R2', 'R3']:
                r_key = f'{r_name.lower()}_range'
                if r_key in bayesian_settings:
                    r_min_exp, r_max_exp = bayesian_settings[r_key]
                    r_ranges[r_name] = (10 ** r_min_exp, 10 ** r_max_exp)

            # Build individual CPE Q ranges
            cpe_q_ranges = {}
            for cpe_name in ['CPE1', 'CPE2', 'CPE3']:
                cpe_key = f'{cpe_name.lower()}_q_range'
                if cpe_key in bayesian_settings:
                    q_min_exp, q_max_exp = bayesian_settings[cpe_key]
                    cpe_q_ranges[cpe_name] = (10 ** q_min_exp, 10 ** q_max_exp)

            # Determine model list
            use_current_model = bayesian_settings.get('use_current_model', True)
            if use_current_model:
                model_list = [circuit_model] if circuit_model else None
            else:
                model_list = bayesian_settings.get('model_list', None)
                if not model_list:
                    model_list = [circuit_model] if circuit_model else None

            # Phase 1: Bayesian optimization
            optimizer = BlackBoxOptEIS(
                freq_fit, Z_fit_data,
                model_list=model_list,
                weight_list=[weight_method],
                n_trials=n_trials,
                timeout=bayesian_timeout,
                early_stop_rmspe=early_stop_rmspe,
                early_stop_patience=early_stop_patience,
                log_step=log_step,
                r_range=r_range,
                cpe_q_range=cpe_q_range,
                r_ranges=r_ranges,
                cpe_q_ranges=cpe_q_ranges,
                fit_timeout=fit_timeout,
                maxfev=fit_maxfev
            )

            def progress_callback_phase1(trial_num, total_trials, best_rmspe):
                progress = min(trial_num / total_trials, 1.0) * 0.5  # First half for Bayesian
                progress_bar.progress(progress)
                status_text.text(f"Bayesian: Trial {trial_num}/{total_trials}, Best RMSPE: {best_rmspe*100:.1f}%")

            best_params = optimizer.optimize(progress_callback=progress_callback_phase1)
            popt, perror, Z_fit_result, rmspe, best_model, param_names = optimizer.fit_best()

            if popt is None or len(popt) == 0:
                progress_bar.empty()
                status_text.empty()
                st.error("Bayesian phase failed to find a valid solution")
            else:
                # Phase 2: Monte Carlo refinement
                status_text.text("Auto Fit: Running Monte Carlo refinement...")

                fit_settings = st.session_state.get('fit_settings', {})
                maxfev = fit_settings.get('maxfev', 10000)
                ftol = fit_settings.get('ftol', 1e-10)
                xtol = fit_settings.get('xtol', 1e-10)
                timeout = fit_settings.get('timeout', 5)
                noise_percent = fit_settings.get('noise_percent', 10) / 100.0
                mc_iterations = fit_settings.get('mc_iterations', 20)

                # Get param names for the best model
                temp_circuit = CustomCircuit(best_model, initial_guess=list(popt))
                temp_param_names, _ = temp_circuit.get_param_names()
                n_params = len(popt)

                best_rmspe = rmspe
                best_popt = popt
                best_perror = perror

                for iteration in range(mc_iterations):
                    progress = 0.5 + (iteration + 1) / mc_iterations * 0.5  # Second half for MC
                    progress_bar.progress(progress)
                    status_text.text(f"MC Refine: {iteration + 1}/{mc_iterations}, Best RMSPE: {best_rmspe*100:.2f}%")

                    # Add noise to parameters
                    noisy_guess = list(best_popt)

                    if iteration > 0:
                        for i, pname in enumerate(temp_param_names):
                            noise_factor = 1 + np.random.uniform(-noise_percent, noise_percent)
                            new_val = noisy_guess[i] * noise_factor
                            if 'CPE' in pname or 'La' in pname:
                                if new_val > 1.0:
                                    new_val = 1.0
                            noisy_guess[i] = new_val

                    try:
                        popt_iter, perror_iter = circuit_fit(
                            freq_fit, Z_fit_data,
                            best_model,
                            noisy_guess,
                            constants={},
                            weight_method=weight_method,
                            timeout=timeout,
                            maxfev=maxfev,
                            ftol=ftol,
                            xtol=xtol
                        )

                        circuit_iter = CustomCircuit(best_model, initial_guess=list(popt_iter))
                        circuit_iter.parameters_ = popt_iter
                        Z_fit_iter = circuit_iter.predict(freq_fit)
                        rmspe_iter = calc_rmspe(Z_fit_data, Z_fit_iter)

                        if rmspe_iter < best_rmspe:
                            best_rmspe = rmspe_iter
                            best_popt = popt_iter
                            best_perror = perror_iter

                    except (FittingTimeoutError, Exception):
                        continue

                progress_bar.empty()
                status_text.empty()

                # Create circuit for full prediction
                circuit = CustomCircuit(best_model, initial_guess=list(best_popt))
                circuit.parameters_ = best_popt
                circuit.conf_ = best_perror
                Z_fit = circuit.predict(freq)

                # Sort by effective capacitance
                param_names_list, _ = circuit.get_param_names()
                sorted_result = sort_ecm_by_cap(best_popt, best_perror, param_names_list)
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
                        sorted_popt.append(best_popt[idx])
                        sorted_perror.append(best_perror[idx] if best_perror is not None else 0.0)

                sorted_popt = np.array(sorted_popt)
                sorted_perror = np.array(sorted_perror)

                circuit.parameters_ = sorted_popt
                circuit.conf_ = sorted_perror

                # Store results
                st.session_state.files[filename]['Z_fit'] = Z_fit
                st.session_state.files[filename]['circuit_model'] = best_model
                st.session_state.files[filename]['circuit_params'] = sorted_popt
                st.session_state.files[filename]['circuit_conf'] = sorted_perror
                st.session_state.files[filename]['circuit_object'] = circuit
                st.session_state.files[filename]['rmspe'] = best_rmspe

                # Update initial guess
                st.session_state.files[filename]['initial_guess'] = list(sorted_popt)
                st.session_state['global_initial_guess'] = list(sorted_popt)

                # Update widget values
                for i, val in enumerate(sorted_popt):
                    widget_key = f"init_{i}"
                    st.session_state[widget_key] = f"{val:.2e}"

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

                st.session_state.files[filename]['effective_caps'] = effective_caps
                st.session_state.files[filename]['sorted_params'] = sorted_result
                st.session_state.files[filename]['total_sigma'] = sigma_total
                st.session_state.files[filename]['total_R'] = R_total
                st.session_state.files[filename]['r_values'] = r_values
                st.session_state.files[filename]['r_sigmas'] = r_sigmas

                st.success(f"Auto Fit completed! RMSPE: {best_rmspe*100:.2f}%")
                st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Auto Fit failed: {str(e)}")

    # Handle Batch Fit button
    if batch_fit_clicked:
        # Get selected files from batch settings (preserve load order)
        batch_selected = st.session_state.get('batch_selected_files', set())
        selected_files = [f for f in st.session_state.files.keys() if f in batch_selected]
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

            # Get fit settings for timeout
            fit_settings = st.session_state.get('fit_settings', {})
            timeout = fit_settings.get('timeout', 5)

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
                    # Use fitting_range for fitting (not display_range)
                    if st.session_state.range_apply_mode == 'global':
                        fitting_range = st.session_state.fitting_range
                    else:
                        # Individual mode: get per-file fitting range
                        fitting_range = st.session_state.file_fitting_ranges.get(
                            fname, st.session_state.fitting_range
                        )

                    if fitting_range:
                        start_idx, end_idx = fitting_range
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
                        weight_method=weight_method,
                        timeout=timeout
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

                except FittingTimeoutError:
                    failed += 1
                    st.warning(f"Timeout fitting {fname} (>{timeout}s)")
                    if stop_on_error:
                        st.error("Batch fitting stopped due to timeout.")
                        batch_stopped = True
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

    # Handle MC-Batch Fit button
    if mc_batch_clicked:
        # Get selected files from batch settings (preserve load order)
        batch_selected = st.session_state.get('batch_selected_files', set())
        selected_files = [f for f in st.session_state.files.keys() if f in batch_selected]
        if len(selected_files) == 0:
            st.warning("No files selected for MC-Batch fitting. Select files in Batch Fit Settings.")
        else:
            st.info(f"MC-Batch fitting {len(selected_files)} files...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            successful = 0
            failed = 0
            high_rmspe_count = 0

            # Get batch fit settings
            batch_settings = st.session_state.get('batch_fit_settings', {})
            stop_on_error = batch_settings.get('stop_on_error', False)
            rmspe_threshold = batch_settings.get('rmspe_threshold', 10.0) / 100

            # Get fit settings for MC
            fit_settings = st.session_state.get('fit_settings', {})
            maxfev = fit_settings.get('maxfev', 10000)
            ftol = fit_settings.get('ftol', 1e-10)
            xtol = fit_settings.get('xtol', 1e-10)
            timeout = fit_settings.get('timeout', 5)
            noise_percent = fit_settings.get('noise_percent', 10) / 100.0
            mc_iterations = fit_settings.get('mc_iterations', 20)

            previous_result = None
            batch_stopped = False

            for idx, fname in enumerate(selected_files):
                if batch_stopped:
                    break

                status_text.text(f"MC-fitting {fname} ({idx + 1}/{len(selected_files)})")
                progress_bar.progress((idx + 1) / len(selected_files))

                fdata = st.session_state.files[fname]
                freq = fdata['freq']
                Z = fdata['Z']

                try:
                    # Apply fitting range if set
                    # Use fitting_range for fitting (not display_range)
                    if st.session_state.range_apply_mode == 'global':
                        fitting_range = st.session_state.fitting_range
                    else:
                        # Individual mode: get per-file fitting range
                        fitting_range = st.session_state.file_fitting_ranges.get(
                            fname, st.session_state.fitting_range
                        )

                    if fitting_range:
                        start_idx, end_idx = fitting_range
                        freq_fit = freq[start_idx:end_idx + 1]
                        Z_fit_data = Z[start_idx:end_idx + 1]
                    else:
                        freq_fit = freq
                        Z_fit_data = Z

                    # Determine initial guess:
                    # 1. Use existing fit result if available
                    # 2. Otherwise use previous file's result
                    from impedance.models.circuits.fitting import calculateCircuitLength
                    n_params = calculateCircuitLength(circuit_model)

                    existing_params = fdata.get('circuit_params')
                    existing_model = fdata.get('circuit_model')

                    if existing_params is not None and existing_model == circuit_model and len(existing_params) == n_params:
                        # Use existing fit result
                        initial_guess = list(existing_params)
                    elif previous_result is not None and len(previous_result) == n_params:
                        # Use previous file's result
                        initial_guess = list(previous_result)
                    else:
                        # Generate default values
                        initial_guess = fdata.get('initial_guess')
                        if initial_guess is None or len(initial_guess) != n_params:
                            # Generate default based on circuit elements
                            initial_guess = []
                            try:
                                temp_c = CustomCircuit(circuit_model, initial_guess=[1.0] * n_params)
                                pnames, _ = temp_c.get_param_names()
                                for pname in pnames:
                                    if 'CPE' in pname and '_1' in pname:
                                        initial_guess.append(0.9)
                                    elif 'CPE' in pname and '_0' in pname:
                                        initial_guess.append(1e-9)
                                    elif 'W' in pname:
                                        initial_guess.append(1e-3)
                                    elif 'C' in pname and 'CPE' not in pname:
                                        initial_guess.append(1e-9)
                                    else:
                                        initial_guess.append(1e4)
                            except:
                                for i in range(n_params):
                                    if i % 3 == 0:
                                        initial_guess.append(1e4)
                                    elif i % 3 == 1:
                                        initial_guess.append(1e-9)
                                    else:
                                        initial_guess.append(0.9)

                    # Get param names
                    temp_circuit = CustomCircuit(circuit_model, initial_guess=initial_guess)
                    temp_param_names, _ = temp_circuit.get_param_names()

                    best_rmspe = float('inf')
                    best_popt = None
                    best_perror = None

                    for iteration in range(mc_iterations):
                        # Add noise to parameters
                        noisy_guess = list(initial_guess) if iteration == 0 else list(best_popt) if best_popt is not None else list(initial_guess)

                        if iteration > 0:
                            for i, pname in enumerate(temp_param_names):
                                noise_factor = 1 + np.random.uniform(-noise_percent, noise_percent)
                                new_val = noisy_guess[i] * noise_factor
                                if 'CPE' in pname or 'La' in pname:
                                    if new_val > 1.0:
                                        new_val = 1.0
                                noisy_guess[i] = new_val

                        try:
                            popt, perror = circuit_fit(
                                freq_fit, Z_fit_data,
                                circuit_model,
                                noisy_guess,
                                constants={},
                                weight_method=weight_method,
                                timeout=timeout,
                                maxfev=maxfev,
                                ftol=ftol,
                                xtol=xtol
                            )

                            circuit_iter = CustomCircuit(circuit_model, initial_guess=list(popt))
                            circuit_iter.parameters_ = popt
                            Z_fit_iter = circuit_iter.predict(freq_fit)
                            rmspe = calc_rmspe(Z_fit_data, Z_fit_iter)

                            if rmspe < best_rmspe:
                                best_rmspe = rmspe
                                best_popt = popt
                                best_perror = perror

                        except (FittingTimeoutError, Exception):
                            continue

                    if best_popt is not None:
                        # Create circuit for full prediction
                        circuit = CustomCircuit(circuit_model, initial_guess=list(best_popt))
                        circuit.parameters_ = best_popt
                        circuit.conf_ = best_perror
                        Z_fit = circuit.predict(freq)

                        # Sort by effective capacitance
                        param_names, _ = circuit.get_param_names()
                        sorted_result = sort_ecm_by_cap(best_popt, best_perror, param_names)
                        effective_caps = sorted_result.get('effective_caps', {})

                        # Build sorted parameter arrays
                        sorted_popt = []
                        sorted_perror = []
                        for name in param_names:
                            if name in sorted_result:
                                sorted_popt.append(sorted_result[name])
                                sorted_perror.append(sorted_result.get(f'{name}_error', 0.0))
                            else:
                                pidx = param_names.index(name)
                                sorted_popt.append(best_popt[pidx])
                                sorted_perror.append(best_perror[pidx] if best_perror is not None else 0.0)

                        sorted_popt = np.array(sorted_popt)
                        sorted_perror = np.array(sorted_perror)

                        circuit.parameters_ = sorted_popt
                        circuit.conf_ = sorted_perror

                        # Store results
                        st.session_state.files[fname]['Z_fit'] = Z_fit
                        st.session_state.files[fname]['circuit_model'] = circuit_model
                        st.session_state.files[fname]['circuit_params'] = sorted_popt
                        st.session_state.files[fname]['circuit_conf'] = sorted_perror
                        st.session_state.files[fname]['circuit_object'] = circuit
                        st.session_state.files[fname]['rmspe'] = best_rmspe

                        # Update initial guess
                        st.session_state.files[fname]['initial_guess'] = list(sorted_popt)

                        # Calculate conductivity
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

                        st.session_state.files[fname]['effective_caps'] = effective_caps
                        st.session_state.files[fname]['sorted_params'] = sorted_result
                        st.session_state.files[fname]['total_sigma'] = sigma_total
                        st.session_state.files[fname]['total_R'] = R_total
                        st.session_state.files[fname]['r_values'] = r_values
                        st.session_state.files[fname]['r_sigmas'] = r_sigmas

                        # Save for next file
                        previous_result = list(sorted_popt)

                        successful += 1
                        if best_rmspe > rmspe_threshold:
                            high_rmspe_count += 1
                    else:
                        failed += 1
                        if stop_on_error:
                            st.error(f"MC-Batch fitting stopped: failed to fit {fname}")
                            batch_stopped = True

                except Exception as e:
                    failed += 1
                    st.warning(f"Failed to MC-fit {fname}: {str(e)}")
                    if stop_on_error:
                        st.error("MC-Batch fitting stopped due to error.")
                        batch_stopped = True

            progress_bar.empty()
            status_text.empty()

            # Update global initial guess
            if previous_result is not None:
                st.session_state['global_initial_guess'] = previous_result
                st.session_state.files[filename]['initial_guess'] = previous_result
                for i, val in enumerate(previous_result):
                    widget_key = f"init_{i}"
                    st.session_state[widget_key] = f"{val:.2e}"

            # Show summary
            result_msg = f"MC-Batch fitting completed! Success: {successful}, Failed: {failed}"
            if high_rmspe_count > 0:
                result_msg += f", High RMSPE (>{rmspe_threshold*100:.0f}%): {high_rmspe_count}"
            st.success(result_msg)
            st.rerun()

    # Handle Auto-Batch Fit button
    if auto_batch_clicked:
        # Get selected files from batch settings (preserve load order)
        batch_selected = st.session_state.get('batch_selected_files', set())
        selected_files = [f for f in st.session_state.files.keys() if f in batch_selected]
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

            # Get Bayesian Fit settings
            bayesian_settings = st.session_state.get('bayesian_fit_settings', {})
            n_trials = bayesian_settings.get('n_trials', 100)
            timeout = bayesian_settings.get('timeout', 30)
            early_stop_rmspe = bayesian_settings.get('early_stop_rmspe', 3.0) / 100
            early_stop_patience = bayesian_settings.get('early_stop_patience', 20)
            log_step = bayesian_settings.get('log_step', 0.5)
            r_range = (bayesian_settings.get('r_min', 1e0), bayesian_settings.get('r_max', 1e8))
            cpe_q_range = (bayesian_settings.get('cpe_q_min', 1e-12), bayesian_settings.get('cpe_q_max', 1e-4))

            # Get fit settings
            fit_settings = st.session_state.get('fit_settings', {})
            fit_timeout = fit_settings.get('timeout', 5)
            fit_maxfev = fit_settings.get('maxfev', 10000)

            # Build individual R ranges from settings
            r_ranges = {}
            for r_name in ['R1', 'R2', 'R3']:
                r_key = f'{r_name.lower()}_range'
                if r_key in bayesian_settings:
                    r_min_exp, r_max_exp = bayesian_settings[r_key]
                    r_ranges[r_name] = (10 ** r_min_exp, 10 ** r_max_exp)

            # Build individual CPE Q ranges from settings
            cpe_q_ranges = {}
            for cpe_name in ['CPE1', 'CPE2', 'CPE3']:
                cpe_key = f'{cpe_name.lower()}_q_range'
                if cpe_key in bayesian_settings:
                    q_min_exp, q_max_exp = bayesian_settings[cpe_key]
                    cpe_q_ranges[cpe_name] = (10 ** q_min_exp, 10 ** q_max_exp)

            # Determine model list
            use_current_model = bayesian_settings.get('use_current_model', True)
            if use_current_model:
                model_list = [circuit_model] if circuit_model else None
            else:
                model_list = bayesian_settings.get('model_list', None)
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
                    # Use fitting_range for fitting (not display_range)
                    if st.session_state.range_apply_mode == 'global':
                        fitting_range = st.session_state.fitting_range
                    else:
                        # Individual mode: get per-file fitting range
                        fitting_range = st.session_state.file_fitting_ranges.get(
                            fname, st.session_state.fitting_range
                        )

                    if fitting_range:
                        start_idx, end_idx = fitting_range
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
                        early_stop_patience=early_stop_patience,
                        log_step=log_step,
                        r_range=r_range,
                        cpe_q_range=cpe_q_range,
                        r_ranges=r_ranges,
                        cpe_q_ranges=cpe_q_ranges,
                        fit_timeout=fit_timeout,
                        maxfev=fit_maxfev
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

        # Header row with "Initial values" and buttons
        header_col1, header_col2, header_col3, header_col4, header_col5, header_col6 = st.columns([2, 1, 1, 1, 1, 1])
        with header_col1:
            st.caption("Initial values (editable)")
        with header_col2:
            # Set all V button
            if st.button("Set all V", key="set_all_variable", help="Set all parameters to Variable"):
                for i in range(len(param_names)):
                    st.session_state.param_fixed[f"fixed_{i}"] = False
                st.rerun()
        with header_col3:
            if st.button("Set all F", key="set_all_fixed", help="Set all parameters to Fixed"):
                for i in range(len(param_names)):
                    st.session_state.param_fixed[f"fixed_{i}"] = True
                st.rerun()
        with header_col4:
            # Add Noise button
            add_noise_help = "Add ±N% random noise to Variable parameters (set N in Fit Settings)"
            if st.button("Add Noise", key="add_noise_btn", help=add_noise_help):
                fit_settings = st.session_state.get('fit_settings', {})
                noise_percent = fit_settings.get('noise_percent', 10) / 100.0
                param_fixed = st.session_state.get('param_fixed', {})
                for i in range(len(param_names)):
                    fixed_key = f"fixed_{i}"
                    is_fixed = param_fixed.get(fixed_key, False)
                    if not is_fixed:
                        widget_key = f"init_{i}"
                        if widget_key in st.session_state:
                            try:
                                current_val = float(st.session_state[widget_key])
                                noise_factor = 1 + np.random.uniform(-noise_percent, noise_percent)
                                new_val = current_val * noise_factor
                                # Alpha parameters should not exceed 1
                                pname = param_names[i] if i < len(param_names) else ""
                                if 'CPE' in pname or 'La' in pname:
                                    if new_val > 1.0:
                                        new_val = 1.0
                                st.session_state[widget_key] = f"{new_val:.2e}"
                            except ValueError:
                                pass
                st.rerun()
        with header_col5:
            # Reset button
            if st.button("Reset", key="reset_params_btn", help="Reset parameters to default initial values"):
                if 'global_initial_guess' in st.session_state:
                    del st.session_state['global_initial_guess']
                st.session_state.files[filename]['initial_guess'] = None
                for i in range(len(param_names)):
                    widget_key = f"init_{i}"
                    if widget_key in st.session_state:
                        del st.session_state[widget_key]
                st.rerun()
        with header_col6:
            # Copy from Table button
            copy_help = "Copy fitted parameters from table to initial values"
            if st.button("Copy Table", key="copy_from_table_btn", help=copy_help):
                if params is not None and len(params) == len(param_names):
                    for i in range(len(param_names)):
                        widget_key = f"init_{i}"
                        st.session_state[widget_key] = f"{params[i]:.2e}"
                    st.rerun()
                else:
                    st.warning("No fitted parameters available to copy")

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
        # Get sample name for filename
        sample_name = st.session_state.sample_info.get('name', '').strip()
        if not sample_name:
            sample_name = 'sample'
        # Sanitize sample name for filename
        safe_sample_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in sample_name)

        # Export buttons in columns
        export_col1, export_col2, export_col3 = st.columns(3)
        with export_col1:
            st.download_button(
                label="CSV",
                data=csv,
                file_name=f"eis_{safe_sample_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with export_col2:
            # Generate Igor file
            igor_str = generate_igor_file(
                files_data=st.session_state.files,
                sample_info=st.session_state.sample_info,
                r_labels=st.session_state.get('r_labels', {'R1': 'bulk', 'R2': 'gb'}),
                export_rows=export_rows
            )
            st.download_button(
                label="ITX",
                data=igor_str,
                file_name=f"eis_{safe_sample_name}_{datetime.now().strftime('%Y%m%d')}.itx",
                mime="text/plain",
                use_container_width=True,
                help="Igor Text File (data + plots)"
            )
        with export_col3:
            st.download_button(
                label="IPF",
                data=generate_igor_procedure_file(),
                file_name="eis_procedures.ipf",
                mime="text/plain",
                use_container_width=True,
                help="Igor Procedure File (for top axis)"
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

        # Load application state
        if 'app_state' in session_data:
            app_state = session_data['app_state']

            # Restore state variables
            if app_state.get('selected_file') is not None:
                st.session_state.selected_file = app_state['selected_file']
            if 'area_input_mode' in app_state:
                st.session_state.area_input_mode = app_state['area_input_mode']
            if 'arrhenius_mode' in app_state:
                st.session_state.arrhenius_mode = app_state['arrhenius_mode']
            if 'show_fit' in app_state:
                st.session_state.show_fit = app_state['show_fit']
            if 'show_all_data' in app_state:
                st.session_state.show_all_data = app_state['show_all_data']
            if 'freq_range' in app_state:
                st.session_state.freq_range = tuple(app_state['freq_range'])
            if 'display_range' in app_state:
                st.session_state.display_range = tuple(app_state['display_range'])
            if 'fitting_range' in app_state:
                st.session_state.fitting_range = tuple(app_state['fitting_range'])
            if 'range_apply_mode' in app_state:
                st.session_state.range_apply_mode = app_state['range_apply_mode']
            if 'file_display_ranges' in app_state:
                st.session_state.file_display_ranges = {k: tuple(v) for k, v in app_state['file_display_ranges'].items()}
            if 'file_fitting_ranges' in app_state:
                st.session_state.file_fitting_ranges = {k: tuple(v) for k, v in app_state['file_fitting_ranges'].items()}
            if 'deleted_points' in app_state:
                st.session_state.deleted_points = app_state['deleted_points']
            if 'show_legend' in app_state:
                st.session_state.show_legend = app_state['show_legend']
            if 'highlight_freq' in app_state:
                st.session_state.highlight_freq = app_state['highlight_freq']
            if 'plot_settings' in app_state:
                st.session_state.plot_settings.update(app_state['plot_settings'])
            if 'r_labels' in app_state:
                st.session_state.r_labels = app_state['r_labels']
            if 'arr_fit_targets' in app_state:
                st.session_state.arr_fit_targets = app_state['arr_fit_targets']
            if app_state.get('arr_fit_range') is not None:
                st.session_state.arr_fit_range = tuple(app_state['arr_fit_range'])
            if 'arr_show_ranges' in app_state:
                st.session_state.arr_show_ranges = {k: tuple(v) for k, v in app_state['arr_show_ranges'].items()}
            if 'param_fixed' in app_state:
                st.session_state.param_fixed = app_state['param_fixed']
            if 'prev_circuit_preset' in app_state:
                st.session_state.prev_circuit_preset = app_state['prev_circuit_preset']
            if 'prev_circuit_model' in app_state:
                st.session_state.prev_circuit_model = app_state['prev_circuit_model']
            if 'hidden_files' in app_state:
                st.session_state.hidden_files = set(app_state['hidden_files'])

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
                    'initial_guess': data.get('initial_guess'),
                    'r_sigmas': data.get('r_sigmas'),
                    'c_effs': data.get('c_effs')
                }

        st.success("Session loaded successfully!")

    except Exception as e:
        st.error(f"Failed to load session: {str(e)}")


def save_session():
    """Save current session to JSON"""
    # Get sample name for filename
    sample_name = st.session_state.sample_info.get('name', '').strip()
    if not sample_name:
        sample_name = 'sample'
    # Sanitize sample name for filename (remove special characters)
    safe_sample_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in sample_name)

    session_data = {
        'sample_info': st.session_state.sample_info,
        'files': {},
        'timestamp': datetime.now().isoformat(),
        # Save application state
        'app_state': {
            'selected_file': st.session_state.selected_file,
            'area_input_mode': st.session_state.get('area_input_mode', 'Diameter'),
            'arrhenius_mode': st.session_state.arrhenius_mode,
            'show_fit': st.session_state.show_fit,
            'show_all_data': st.session_state.show_all_data,
            'freq_range': list(st.session_state.freq_range),
            'display_range': list(st.session_state.display_range),
            'fitting_range': list(st.session_state.fitting_range),
            'range_apply_mode': st.session_state.range_apply_mode,
            'file_display_ranges': {k: list(v) for k, v in st.session_state.file_display_ranges.items()},
            'file_fitting_ranges': {k: list(v) for k, v in st.session_state.file_fitting_ranges.items()},
            'deleted_points': st.session_state.deleted_points,
            'show_legend': st.session_state.show_legend,
            'highlight_freq': st.session_state.highlight_freq,
            'plot_settings': st.session_state.plot_settings,
            'r_labels': st.session_state.r_labels,
            'arr_fit_targets': st.session_state.arr_fit_targets,
            'arr_fit_range': list(st.session_state.arr_fit_range) if st.session_state.get('arr_fit_range') else None,
            'arr_show_ranges': {k: list(v) for k, v in st.session_state.get('arr_show_ranges', {}).items()},
            'param_fixed': st.session_state.get('param_fixed', {}),
            'prev_circuit_preset': st.session_state.get('prev_circuit_preset', 'Custom'),
            'prev_circuit_model': st.session_state.get('prev_circuit_model', ''),
            'hidden_files': list(st.session_state.get('hidden_files', set())),
        }
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
            'gb_sigma': data.get('gb_sigma'),
            'r_sigmas': data.get('r_sigmas'),
            'c_effs': data.get('c_effs')
        }

    json_str = json.dumps(session_data, indent=2)

    st.download_button(
        label="Download Session (JSON)",
        data=json_str,
        file_name=f"eis_{safe_sample_name}_{datetime.now().strftime('%Y%m%d')}.json",
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

        st.markdown("---")

        # Sample info section
        sidebar_sample_info()

        st.markdown("---")

        # Analysis mode selector - single selection from 3 modes
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Nyquist", "Arrhenius", "Mapping"],
            index=["Nyquist", "Arrhenius", "Mapping"].index(st.session_state.analysis_mode),
            horizontal=True,
            key="analysis_mode_radio",
            help="Nyquist: Impedance plots | Arrhenius: Temperature dependence | Mapping: Spatial distribution"
        )
        if analysis_mode != st.session_state.analysis_mode:
            st.session_state.analysis_mode = analysis_mode
            # Update legacy flags for backward compatibility
            st.session_state.arrhenius_mode = (analysis_mode == 'Arrhenius')
            st.session_state.mapping_mode = (analysis_mode == 'Mapping')
            st.rerun()

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

    # Circuit Analysis (Nyquist mode only) and Multipoint Table
    analysis_mode = st.session_state.get('analysis_mode', 'Nyquist')
    if analysis_mode == 'Nyquist':
        st.markdown("---")
        # Nyquist mode: Circuit analysis panel full width, then table below
        circuit_analysis_panel()
        # Separator before multipoint table
        st.markdown("---")

    # Multipoint table shown in Nyquist and Arrhenius modes (Mapping has its own summary)
    if analysis_mode != 'Mapping':
        multipoint_analysis_table()


if __name__ == "__main__":
    main()
