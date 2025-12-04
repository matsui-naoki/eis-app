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
from tools.fitting import circuit_fit, calc_rmspe, r2sigma, r2logsigma, effective_capacitance
from components.plots import create_nyquist_plot, create_bode_plot, create_arrhenius_plot

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
    .block-container, [data-testid="stSidebar"] > div {
        background-color: #ffffff !important;
    }

    /* Input fields with black border and reduced padding */
    input, .stTextInput input, .stNumberInput input {
        border: 1px solid #000000 !important;
        border-radius: 4px !important;
        padding: 0.3rem 0.5rem !important;
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

    /* Selected file button */
    .stButton > button[data-testid="stBaseButton-primary"]:not(:disabled) {
        background-color: #1976d2 !important;
        color: #ffffff !important;
    }

    /* Comments/help text - light gray */
    .stCaption, small, .stTooltipIcon {
        color: #888888 !important;
    }

    /* Hide Streamlit header, menu, footer */
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none !important;
    }

    /* Hide ALL sidebar collapse/expand buttons */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="baseButton-header"],
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"] button[kind="header"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }

    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.2rem !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0 !important;
    }

    section[data-testid="stSidebar"] h3 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.1rem !important;
        font-size: 0.85rem !important;
        color: #000000 !important;
    }

    /* Reduce main content gaps and padding */
    [data-testid="stVerticalBlock"] {
        gap: 0.3rem !important;
    }

    .block-container {
        padding-top: 0.3rem !important;
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
        padding-bottom: 0.2rem;
        border-bottom: 2px solid #1976d2;
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
    if 'arrhenius_mode' not in st.session_state:
        st.session_state.arrhenius_mode = False
    if 'multipoint_data' not in st.session_state:
        st.session_state.multipoint_data = []
    if 'show_fit' not in st.session_state:
        st.session_state.show_fit = True
    if 'show_all_data' not in st.session_state:
        st.session_state.show_all_data = False
    if 'freq_range' not in st.session_state:
        st.session_state.freq_range = None  # (min_idx, max_idx)
    if 'show_legend' not in st.session_state:
        st.session_state.show_legend = True
    if 'highlight_freq' not in st.session_state:
        st.session_state.highlight_freq = False


def sidebar_sample_info():
    """Render sample information input section in sidebar"""
    # Title
    st.sidebar.markdown('<div class="sidebar-title">EIS Analyzer</div>', unsafe_allow_html=True)

    # Reset and Save buttons at top
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Reset", width='stretch'):
            reset_session()
            st.rerun()
    with col2:
        if st.button("Save", width='stretch'):
            save_session()

    st.sidebar.markdown("---")

    # Sample Name
    st.session_state.sample_info['name'] = st.sidebar.text_input(
        "Sample Name",
        value=st.session_state.sample_info.get('name', ''),
        placeholder="Sample name",
        label_visibility="collapsed"
    )

    # Thickness
    st.session_state.sample_info['thickness'] = st.sidebar.number_input(
        "Thickness (cm)",
        min_value=0.0001,
        max_value=100.0,
        value=st.session_state.sample_info.get('thickness', 0.1),
        format="%.4f"
    )

    # Diameter and Area side by side with auto-calculation
    col1, col2 = st.sidebar.columns(2)
    with col1:
        new_diameter = st.number_input(
            "Diameter (cm)",
            min_value=0.001,
            max_value=10.0,
            value=st.session_state.sample_info.get('diameter', 1.128),
            format="%.4f",
            key="diameter_input"
        )
    with col2:
        # Calculate expected area from current diameter
        expected_area = np.pi * (st.session_state.sample_info.get('diameter', 1.0) / 2) ** 2
        new_area = st.number_input(
            "Area (cm²)",
            min_value=0.0001,
            max_value=100.0,
            value=st.session_state.sample_info.get('area', expected_area),
            format="%.4f",
            key="area_input"
        )

    # Auto-calculate: if diameter changed, update area; if area changed, update diameter
    old_diameter = st.session_state.sample_info.get('diameter', 1.0)
    old_area = st.session_state.sample_info.get('area', 0.785)

    if abs(new_diameter - old_diameter) > 1e-6:
        # Diameter changed - calculate area
        st.session_state.sample_info['diameter'] = new_diameter
        st.session_state.sample_info['area'] = np.pi * (new_diameter / 2) ** 2
    elif abs(new_area - old_area) > 1e-6:
        # Area changed - calculate diameter
        st.session_state.sample_info['area'] = new_area
        st.session_state.sample_info['diameter'] = 2 * np.sqrt(new_area / np.pi)
    else:
        st.session_state.sample_info['diameter'] = new_diameter
        st.session_state.sample_info['area'] = new_area


def sidebar_file_manager():
    """File management tab in sidebar"""
    st.sidebar.markdown("### Files")

    # File uploader - supports multiple formats
    uploaded_files = st.sidebar.file_uploader(
        "Upload",
        type=['mpt', 'z', 'dta', 'csv', 'txt', 'par'],
        accept_multiple_files=True,
        help=".mpt, .z, .DTA, .csv, .txt, .PAR",
        label_visibility="collapsed"
    )

    # Arrhenius mode toggle
    st.session_state.arrhenius_mode = st.sidebar.checkbox(
        "Arrhenius Mode",
        value=st.session_state.arrhenius_mode,
        help="Temperature-dependent analysis"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            original_filename = uploaded_file.name
            base_name = os.path.splitext(original_filename)[0]
            file_ext = os.path.splitext(original_filename)[1]

            # Check if this file has already been processed
            # (check if any dataset name starts with base_name)
            already_loaded = any(
                name == base_name or name.startswith(f"{base_name}_")
                for name in st.session_state.files.keys()
            )

            if not already_loaded:
                # Load with loop detection
                datasets, error = load_uploaded_file_with_loops(
                    uploaded_file, file_ext, base_name, rtol=0.01
                )

                if error:
                    st.sidebar.error(f"Error loading {original_filename}: {error}")
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
                            st.sidebar.warning(f"Invalid data in {name}: {msg}")

                    if loaded_count > 0:
                        if len(datasets) > 1:
                            st.sidebar.success(f"Loaded {loaded_count} datasets from {original_filename}")
                        else:
                            st.sidebar.success(f"Loaded: {datasets[0]['name']}")

    # File list
    if len(st.session_state.files) > 0:
        st.sidebar.markdown("### Loaded Files")

        # Show all data checkbox
        st.session_state.show_all_data = st.sidebar.checkbox(
            "Show all data",
            value=st.session_state.show_all_data,
            help="Display all loaded files on plots"
        )

        for i, filename in enumerate(list(st.session_state.files.keys())):
            col1, col2 = st.sidebar.columns([4, 1])

            is_selected = (filename == st.session_state.selected_file)

            with col1:
                # Use different button type for selected file
                if is_selected:
                    st.markdown('<div class="selected-file-btn">', unsafe_allow_html=True)
                if st.button(filename, key=f"select_{i}", width='stretch', type="primary" if is_selected else "secondary"):
                    st.session_state.selected_file = filename
                    st.rerun()
                if is_selected:
                    st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                if st.button("Del", key=f"delete_{i}", help="Delete file"):
                    del st.session_state.files[filename]
                    if st.session_state.selected_file == filename:
                        st.session_state.selected_file = None
                    st.rerun()


def sidebar_data_view():
    """Data view tab in sidebar"""
    st.sidebar.markdown("### Data View")

    if st.session_state.selected_file and st.session_state.selected_file in st.session_state.files:
        filename = st.session_state.selected_file
        data = st.session_state.files[filename]

        st.sidebar.markdown(f"**Current File:** {filename}")

        # Show data table
        freq = data['freq']
        Z = data['Z']

        df = pd.DataFrame({
            'Frequency (Hz)': freq,
            "Z' (Ohm)": np.real(Z),
            "Z'' (Ohm)": np.imag(Z)
        })

        st.sidebar.dataframe(df, height=300, width='stretch')

        # Temperature input for Arrhenius mode
        if st.session_state.arrhenius_mode:
            temp = st.sidebar.number_input(
                "Temperature (K)",
                min_value=0.0,
                max_value=1000.0,
                value=data.get('temperature') or 298.15,
                format="%.2f",
                key=f"temp_{filename}"
            )
            st.session_state.files[filename]['temperature'] = temp
    else:
        st.sidebar.info("Select a file to view data")


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

    # Get fitting range for Bode plot fitting curve display
    freq_range = st.session_state.freq_range

    # Three plots side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_nyquist = create_nyquist_plot(
            st.session_state.files, selected_for_plot,
            st.session_state.show_fit,
            st.session_state.show_legend,
            st.session_state.highlight_freq
        )
        st.plotly_chart(fig_nyquist, width='stretch')

    with col2:
        fig_bode = create_bode_plot(
            st.session_state.files, selected_for_plot,
            st.session_state.show_fit,
            st.session_state.show_legend,
            freq_range
        )
        st.plotly_chart(fig_bode, width='stretch')

    with col3:
        # Arrhenius plot
        if st.session_state.arrhenius_mode:
            multipoint_data = []
            for filename in st.session_state.files:
                data = st.session_state.files[filename]
                if data.get('temperature') and data.get('circuit_params') is not None:
                    multipoint_data.append({
                        'temperature': data['temperature'],
                        'total_sigma': data.get('total_sigma'),
                        'bulk_sigma': data.get('bulk_sigma'),
                        'gb_sigma': data.get('gb_sigma')
                    })
            fig_arrhenius = create_arrhenius_plot(multipoint_data, 'total',
                                                   st.session_state.show_legend)
            st.plotly_chart(fig_arrhenius, width='stretch')
        else:
            st.caption("Enable Arrhenius Mode")

    # Controls below plots
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])

    with ctrl_col1:
        st.session_state.show_fit = st.checkbox(
            "Show Fitted",
            value=st.session_state.show_fit
        )
        st.session_state.show_legend = st.checkbox(
            "Show Legend",
            value=st.session_state.show_legend
        )
        st.session_state.highlight_freq = st.checkbox(
            "Highlight Freq",
            value=st.session_state.highlight_freq,
            help="Highlight 10^n Hz points"
        )

    # Fitting range slider
    with ctrl_col2:
        st.caption("Fitting Range")

    with ctrl_col3:
        if st.session_state.selected_file and st.session_state.selected_file in st.session_state.files:
            data = st.session_state.files[st.session_state.selected_file]
            freq = data['freq']
            n_points = len(freq)

            if n_points > 1:
                # Get current range or default to full range
                current_range = st.session_state.freq_range or (0, n_points - 1)

                freq_range = st.slider(
                    "range",
                    min_value=0,
                    max_value=n_points - 1,
                    value=current_range,
                    label_visibility="collapsed"
                )
                st.session_state.freq_range = freq_range


def format_param_name(name: str) -> str:
    """Convert parameter names: CPE1_0 -> CPE1_Q, CPE1_1 -> CPE1_α"""
    if '_0' in name and 'CPE' in name:
        return name.replace('_0', '_Q')
    elif '_1' in name and 'CPE' in name:
        return name.replace('_1', '_α')
    return name


def circuit_analysis_panel():
    """Circuit analysis panel with new layout"""
    if not st.session_state.selected_file or st.session_state.selected_file not in st.session_state.files:
        st.info("Select a file from the sidebar to perform circuit analysis")
        return

    filename = st.session_state.selected_file
    data = st.session_state.files[filename]

    # Top row: Circuit String (left) and Weight Method (right)
    col1, col2 = st.columns([2, 1])

    with col1:
        circuit_model = st.text_input(
            "Circuit String",
            value=data.get('circuit_model') or 'p(R1,CPE1)-CPE2',
            help="Example: p(R1,CPE1)-p(R2,CPE2)-CPE3",
            label_visibility="collapsed",
            placeholder="Circuit String (e.g., p(R1,CPE1)-CPE2)"
        )

    with col2:
        weight_options = [None, "modulus", "squared_modulus", "proportional"]
        weight_labels = ["None", "Modulus", "Squared Modulus", "Proportional"]
        current_idx = 1  # default to modulus
        weight_method = st.selectbox(
            "Weight Method",
            weight_options,
            index=current_idx,
            format_func=lambda x: weight_labels[weight_options.index(x)] if x in weight_options else str(x),
            label_visibility="collapsed"
        )

    # Button row: Fit Circuit (left) and Copy to Initial (right)
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        fit_clicked = st.button("Fit Circuit", width='stretch', type="primary")

    with btn_col2:
        copy_clicked = st.button("Copy to Initial", width='stretch',
                                  disabled=(data.get('circuit_params') is None))

    # Handle copy button
    if copy_clicked and data.get('circuit_params') is not None:
        st.session_state.files[filename]['initial_guess'] = list(data['circuit_params'])
        st.rerun()

    # Handle fit button
    if fit_clicked:
        with st.spinner("Fitting..."):
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

                # Get initial guess from table or default
                initial_guess = data.get('initial_guess') or [1e6, 1e-9, 0.9, 1e-6, 0.9]

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

                # Store results
                st.session_state.files[filename]['Z_fit'] = Z_fit
                st.session_state.files[filename]['circuit_model'] = circuit_model
                st.session_state.files[filename]['circuit_params'] = popt
                st.session_state.files[filename]['circuit_conf'] = perror
                st.session_state.files[filename]['circuit_object'] = circuit
                st.session_state.files[filename]['rmspe'] = rmspe

                # Calculate conductivity
                S = st.session_state.sample_info.get('area', 1.0)
                L = st.session_state.sample_info.get('thickness', 0.1)

                param_names, _ = circuit.get_param_names()
                R_total = sum([popt[i] for i, name in enumerate(param_names) if 'R' in name and 'CPE' not in name])

                sigma_total = r2sigma(R_total, S, L)
                st.session_state.files[filename]['total_sigma'] = sigma_total
                st.session_state.files[filename]['total_R'] = R_total

                st.rerun()

            except Exception as e:
                st.error(f"Fitting failed: {str(e)}")

    # Parameter table (transposed: columns = parameters, rows = Initial/Value/Error/Error%)
    st.markdown("**Parameters**")

    # Try to get param names from circuit object or parse from circuit_model
    if data.get('circuit_object'):
        circuit = data['circuit_object']
        param_names, units = circuit.get_param_names()
    else:
        # Create temporary circuit to get param names
        try:
            temp_circuit = CustomCircuit(circuit_model)
            param_names, units = temp_circuit.get_param_names()
        except:
            param_names = []
            units = []

    if len(param_names) > 0:
        # Format parameter names
        display_names = [format_param_name(name) for name in param_names]

        # Build table data
        initial_guess = data.get('initial_guess') or [0.0] * len(param_names)
        # Pad initial_guess if needed
        while len(initial_guess) < len(param_names):
            initial_guess.append(0.0)

        params = data.get('circuit_params')
        confs = data.get('circuit_conf')

        table_data = {'': ['Initial', 'Value', 'Error', 'Error %']}

        for i, name in enumerate(display_names):
            init_val = initial_guess[i] if i < len(initial_guess) else 0.0
            if params is not None and i < len(params):
                val = params[i]
                err = confs[i] if confs is not None and i < len(confs) else 0.0
                err_pct = (err / val * 100) if val != 0 else 0.0
                table_data[name] = [
                    f"{init_val:.3e}",
                    f"{val:.3e}",
                    f"{err:.2e}",
                    f"{err_pct:.1f}%"
                ]
            else:
                table_data[name] = [f"{init_val:.3e}", '-', '-', '-']

        df = pd.DataFrame(table_data)
        st.dataframe(df, hide_index=True, width='stretch')

        # Show RMSPE and conductivity below table
        if data.get('rmspe') is not None:
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("RMSPE", f"{data['rmspe']:.4f}")
            with info_col2:
                if data.get('total_R'):
                    st.metric("Total R", f"{data['total_R']:.2e} Ω")
            with info_col3:
                if data.get('total_sigma'):
                    st.metric("σ", f"{data['total_sigma']:.2e} S/cm")


def multipoint_analysis_table():
    """Multipoint analysis table"""
    if len(st.session_state.files) == 0:
        st.info("No data available")
        return

    # Collect all analyzed data
    rows = []
    for filename, data in st.session_state.files.items():
        if data.get('circuit_params') is None:
            continue

        # All values as strings for consistent typing
        temp = data.get('temperature')
        row = {
            'File': filename,
            'Temperature (K)': f"{temp:.2f}" if temp else '-',
            '1000/T (K-1)': f"{1000/temp:.4f}" if temp else '-',
        }

        # Add circuit parameters
        if data.get('circuit_object'):
            circuit = data['circuit_object']
            param_names, _ = circuit.get_param_names()
            for i, name in enumerate(param_names):
                row[name] = f"{data['circuit_params'][i]:.4e}"

        row['RMSPE'] = f"{data.get('rmspe', 0):.6f}"
        row['Total R (Ohm)'] = f"{data.get('total_R', 0):.4e}" if data.get('total_R') else '-'
        row['sigma (S/cm)'] = f"{data.get('total_sigma', 0):.4e}" if data.get('total_sigma') else '-'
        row['log(sigma)'] = f"{np.log10(data['total_sigma']):.4f}" if data.get('total_sigma') else '-'

        if data.get('total_sigma') and temp:
            log_sigma_T = np.log10(data['total_sigma'] * temp)
            row['log(sigma*T)'] = f"{log_sigma_T:.4f}"
        else:
            row['log(sigma*T)'] = '-'

        rows.append(row)

    if len(rows) > 0:
        df = pd.DataFrame(rows)
        st.dataframe(df, width='stretch')

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"eis_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Perform circuit fitting to see multipoint analysis results")


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

    st.sidebar.download_button(
        label="Download Session (JSON)",
        data=json_str,
        file_name=f"eis_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

    st.sidebar.success("Session saved!")


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
        # Sample info (always visible)
        sidebar_sample_info()

        st.sidebar.markdown("---")

        # Tabs for different sidebar sections
        tab_selected = st.sidebar.radio(
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
            #st.sidebar.markdown("### Settings")
            st.sidebar.info("Settings panel (to be implemented)")

    # Main content - Single page layout
    # Plots section
    main_panel_plots()

    st.markdown("---")

    # Circuit Analysis and Multipoint Table side by side
    col_left, col_right = st.columns([1, 1])

    with col_left:
        circuit_analysis_panel()

    with col_right:
        multipoint_analysis_table()


if __name__ == "__main__":
    main()
