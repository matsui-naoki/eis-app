"""
Help texts and documentation for EIS Analyzer UI elements.
Centralized location for all hover tooltips and help documentation.
"""

# =============================================================================
# Circuit Model Help
# =============================================================================

CIRCUIT_MODEL_HELP = """**Circuit Model Notation:**

Use impedance.py notation for circuit elements:
- **R**: Resistor (unit: Ω)
- **C**: Capacitor (unit: F)
- **CPE**: Constant Phase Element (Q in F·s^(α-1), α dimensionless)
- **W**: Warburg element (semi-infinite diffusion)
- **Ws**: Warburg short element (finite length diffusion)

**Connection operators:**
- **-** : Series connection (e.g., R1-R2)
- **p(A,B)** : Parallel connection (e.g., p(R1,CPE1))

**Examples:**
- `R1-p(R2,CPE1)` : R in series with parallel R//CPE
- `p(R1,CPE1)-p(R2,CPE2)-CPE3` : Two R//CPE in series with CPE
- `R1-p(R2,CPE1)-p(R3,CPE2)-CPE3` : R followed by two R//CPE and CPE
"""

CIRCUIT_ELEMENT_HELP = {
    'R': "Resistance in Ohms (Ω). Typical range: 1 - 10⁸ Ω",
    'CPE_Q': "CPE pseudo-capacitance Q in F·s^(α-1). Typical range: 10⁻¹² - 10⁻⁴",
    'CPE_alpha': "CPE exponent α (0-1). α=1: ideal capacitor, α=0.5: Warburg-like, α<0.9: non-ideal surface",
    'C': "Capacitance in Farads (F). Typical range: 10⁻¹² - 10⁻⁶ F",
}

# =============================================================================
# Weighting Method Help
# =============================================================================

WEIGHT_METHOD_HELP = """**Weighting Methods for Fitting:**

- **None**: No weighting (equal weights for all points) ★recommended for general use
- **Proportional**: w = |Z| (emphasizes high impedance / low frequency region) ★recommended for solid electrolytes
- **Modulus**: w = 1/|Z| (emphasizes low impedance / high frequency region)
- **Squared Modulus**: w = 1/|Z|² (stronger emphasis on low impedance / high frequency)

**Selection Guide:**
- Use **None** or **Proportional** for most EIS data
- Use **Modulus** when high-frequency features are important
- Use **Proportional** when low-frequency features (grain boundary, electrode) are more important
"""

# =============================================================================
# Fitting Metrics Help
# =============================================================================

RMSPE_HELP = """**RMSPE (Root Mean Square Percentage Error)**

Measures the quality of fit as a percentage:
- RMSPE = √(mean((|Z_measured - Z_fit| / |Z_measured|)²)) × 100%

**Interpretation:**
- < 1%: Excellent fit
- 1-3%: Good fit
- 3-5%: Acceptable fit
- > 5%: Poor fit (consider different circuit model or check data)
"""

# =============================================================================
# Summary Table Help
# =============================================================================

SUMMARY_TABLE_HELP = """**Summary Table Contents:**

**Resistance (R)**: Fitted resistance values in Ω
- R1: Often bulk resistance (grain interior)
- R2: Often grain boundary resistance
- R_total: Sum of all resistances

**Conductivity (σ)**: Calculated from σ = L / (R × S)
- Where L = sample thickness (cm), S = electrode area (cm²)
- Unit: S/cm

**Effective Capacitance (C_eff)**: Calculated from CPE parameters
- Formula: C_eff = Q^(1/n) × R^((1-n)/n)
- Where Q = CPE pseudo-capacitance, n = CPE exponent (α)

**Typical C_eff values:**
- Bulk: 10⁻¹² - 10⁻¹¹ F/cm (or ~pF/cm)
- Grain boundary: 10⁻¹¹ - 10⁻⁸ F/cm
- Electrode interface: 10⁻⁷ - 10⁻⁵ F/cm

**Note:** Elements are automatically sorted by C_eff (R1 = smallest C_eff = bulk)
"""

# =============================================================================
# Fit Settings Help
# =============================================================================

FIT_SETTINGS_HELP = {
    'maxfev': "Maximum number of function evaluations (scipy curve_fit). Higher values allow more iterations for convergence. Default: 10000",
    'ftol': "Function tolerance. Fitting stops when relative change in residuals is below this value. Lower = stricter. Default: 1e-10",
    'xtol': "Parameter tolerance. Fitting stops when relative change in parameters is below this value. Lower = stricter. Default: 1e-10",
    'timeout': "Maximum time (seconds) for each fitting attempt. If timeout is reached, fitting is interrupted and skipped. Prevents long stalls. Default: 5 sec",
    'noise_percent': "Random noise range (±%) for Add Noise button. Only Variable (V) parameters are modified. Default: 10%",
    'mc_iterations': "Number of Monte Carlo fit iterations. Each iteration adds random noise and refits. Higher = more chance to escape local minima. Default: 100",
    'keep_better': "Compare new fit result with existing result (if any) and keep the one with lower RMSPE. Prevents accidental overwriting of good fits.",
    'global_opt': "Use global optimization (basin-hopping) instead of local optimization. Slower but may find better solutions for complex landscapes."
}

# =============================================================================
# Bayesian Fit Settings Help
# =============================================================================

BAYESIAN_FIT_SETTINGS_HELP = {
    'n_trials': "Maximum number of optimization trials. More trials = better chance of finding optimal solution, but slower. Recommended: 50-200",
    'timeout': "Maximum time (seconds) for optimization. Stops early if timeout reached.",
    'early_stop_rmspe': "Target RMSPE (%). Optimization stops early if this target is reached. Lower = stricter.",
    'early_stop_patience': "Stop optimization if best result is not improved for N consecutive trials. Default: 20",
    'r_min': "Minimum resistance value (10^x Ω) to search. Adjust based on expected sample resistance.",
    'r_max': "Maximum resistance value (10^x Ω) to search. Adjust based on expected sample resistance.",
    'cpe_q_min': "Minimum CPE Q value (10^x F·s^(α-1)) to search.",
    'cpe_q_max': "Maximum CPE Q value (10^x F·s^(α-1)) to search.",
    'log_step': "Search step size in log scale (orders of magnitude). Smaller = finer mesh, more precise but slower. 0 = continuous (Optuna default).",
    'use_current_model': "If checked, use only the circuit model specified above. Otherwise, try multiple models to find the best fit.",
    'model_list': "Select which circuit models to try during optimization."
}

# =============================================================================
# Batch Fit Settings Help
# =============================================================================

BATCH_FIT_SETTINGS_HELP = {
    'use_previous_result': "Use the fitted result from the previous file as the initial guess for the next file. Helps with sequential fitting of similar samples.",
    'stop_on_error': "Stop batch fitting if any file fails to fit. Otherwise, continue with remaining files.",
    'rmspe_threshold': "Maximum acceptable RMSPE (%). Files with RMSPE above this threshold will be marked as potentially bad fits."
}

# =============================================================================
# Sample Info Help
# =============================================================================

SAMPLE_INFO_HELP = {
    'area': "Electrode area (S) in cm². Used to calculate conductivity σ = L/(R×S)",
    'thickness': "Sample thickness (L) in cm. Used to calculate conductivity σ = L/(R×S)",
    'diameter': "Sample diameter in mm (optional). For reference only.",
    'label': "Sample label/name for identification in plots and exports."
}

# =============================================================================
# V/F Toggle Help
# =============================================================================

VF_TOGGLE_HELP = """**V/F (Variable/Fixed) Toggle:**

- **● V (Variable)**: Parameter will be optimized during fitting
- **○ F (Fixed)**: Parameter will be held constant at the input value

**Use cases:**
- Fix α close to 1 if expecting ideal capacitor behavior
- Fix known bulk resistance from other measurements
- When fitting fails, try fixing some parameters to help convergence
"""

# =============================================================================
# Fitting Range Help
# =============================================================================

FITTING_RANGE_HELP = """**Fitting Range:**

Select the frequency range for fitting:
- Left slider: Start index (low frequency)
- Right slider: End index (high frequency)

**Tips:**
- Exclude noisy low-frequency data points
- Exclude incomplete high-frequency semicircles
- Focus on frequency region with reliable data
"""

# =============================================================================
# Temperature Pattern Help
# =============================================================================

TEMPERATURE_PATTERN_HELP = """**Temperature Pattern Format:**

**Single values:**
`25, 50, 100, 150`

**Pattern format [T0, STEP, NUM]:**
`[300, 50, 4]` → 300, 350, 400, 450

**Mixed:**
`25, [100, 50, 3]` → 25, 100, 150, 200

**Examples:**
- Heating: `[300, 50, 5]` → 300, 350, 400, 450, 500
- Cooling: `[500, -50, 5]` → 500, 450, 400, 350, 300
"""

FILENAME_PATTERN_HELP = """**Filename Pattern for Temperature Extraction:**

**Format:** [separator, index], [separator, index], ...

**How it works:**
1. Split filename by separator
2. Take the element at specified index
3. Repeat for additional patterns

**Example:**
Filename: `sample_300C_data.txt`
Pattern: `[_, 1], [C, 0]`
- Split by `_` → ["sample", "300C", "data.txt"]
- Take index 1 → "300C"
- Split by `C` → ["300", ""]
- Take index 0 → "300"

**Common patterns:**
- `[_, 1]` : Second part after underscore
- `[C, 0]` : Part before 'C'
- `[-, 1]` : Second part after hyphen
"""

# =============================================================================
# Arrhenius Plot Help
# =============================================================================

ARRHENIUS_PLOT_HELP = """**Arrhenius Plot:**

Displays log(σT) vs 1000/T for activation energy analysis.

**Axes:**
- X-axis: 1000/T (K⁻¹)
- Y-axis: log₁₀(σT) where σ = conductivity (S/cm), T = temperature (K)

**Linear fit:**
- Slope = -Ea/(1000×k×ln(10))
- Where Ea = activation energy, k = Boltzmann constant

**Activation Energy:**
- Ea = -slope × 1000 × k × ln(10)
- Typical values: 0.2-1.0 eV for solid electrolytes
"""

# =============================================================================
# Data Export Help
# =============================================================================

EXPORT_HELP = {
    'csv': "Export data as CSV file. Includes frequency, Z_real, Z_imag, fitted values if available.",
    'json_session': "Save complete session state including all loaded files, fit results, and settings. Can be loaded later to continue analysis.",
    'excel': "Export summary table as Excel file with multiple sheets."
}

# =============================================================================
# File Format Help
# =============================================================================

FILE_FORMAT_HELP = """**Supported File Formats:**

**Tab/Space separated (.txt, .dat, .csv):**
- Columns: Frequency, Z_real, Z_imag (or -Z_imag)
- Header row optional
- Comments starting with '#' are ignored

**Biologic (.mpt):**
- Exported from EC-Lab software
- Multiple data sections supported

**Gamry (.DTA):**
- Exported from Gamry Instruments software
- Automatic header detection

**Note:** Frequency should be in Hz, impedance in Ohms.
"""

# =============================================================================
# Plot Settings Help
# =============================================================================

PLOT_SETTINGS_HELP = {
    'z_unit': "Unit for impedance display. Choose from Ω, kΩ, MΩ, GΩ.",
    'tick_font_size': "Font size for axis tick labels (6-20 pt).",
    'axis_label_font_size': "Font size for axis labels (8-24 pt).",
    'marker_color': "Fill color for data point markers.",
    'marker_symbol': "Shape of data point markers (circle, square, diamond, etc.).",
    'marker_size': "Size of data point markers (1-20 pt).",
    'marker_alpha': "Transparency of markers (0=transparent, 1=opaque).",
    'marker_edge_color': "Color of marker edge/outline.",
    'marker_edge_width': "Width of marker edge line (0=no edge).",
    'fit_line_color': "Color of the fitted curve line.",
    'fit_line_width': "Width of the fitted curve line.",
    'show_zeroline': "Show horizontal zero line on plots.",
    'legend_font_size': "Font size for plot legend text.",
}

ARRHENIUS_SETTINGS_HELP = {
    'arr_marker_color': "Fill color for Arrhenius plot markers.",
    'arr_marker_symbol': "Shape of Arrhenius plot markers.",
    'arr_marker_size': "Size of Arrhenius plot markers.",
    'arr_marker_edge_color': "Edge color for Arrhenius plot markers.",
    'arr_marker_edge_width': "Edge width for Arrhenius plot markers.",
    'arr_line_color': "Color for connecting lines between points.",
    'arr_line_width': "Width of connecting lines.",
    'arr_show_line': "Show connecting lines between data points.",
    'arr_legend_font_size': "Font size for Arrhenius plot legend.",
}

# =============================================================================
# Mode and Display Help
# =============================================================================

MODE_HELP = {
    'arrhenius_mode': "Enable temperature-dependent Arrhenius analysis. Allows fitting activation energy from multiple temperature measurements.",
    'show_all_data': "Display all loaded files simultaneously on plots. Useful for comparing multiple measurements.",
    'show_fit': "Show the fitted curve on the Nyquist and Bode plots.",
    'show_legend': "Display legend on plots showing data labels.",
    'show_fit_legend': "Display legend for fitted lines (requires Show Legend enabled).",
    'highlight_freq': "Mark specific frequency points (10^n Hz) on the Nyquist plot.",
    'cycle_mode': "Separate data into heating/cooling or numbered cycles for Arrhenius analysis.",
}

# =============================================================================
# Frequency Range Help
# =============================================================================

FREQ_RANGE_HELP = """**Frequency Range Slider:**

Select the range of data points to display and fit:
- Drag left handle: Exclude low-frequency (noisy) data
- Drag right handle: Exclude high-frequency data

**Tips:**
- Low-frequency data often shows electrode effects
- High-frequency data may be incomplete
- Adjust range to focus on grain/bulk features
"""

SHOW_RANGE_HELP = """**Show Range Slider:**

Control which data points are displayed on the Arrhenius plot for each conductivity type:
- Useful for excluding outliers or temperature regions with different behavior
- Separate sliders for bulk, grain boundary, and total conductivity
"""

# =============================================================================
# Delete Points Help
# =============================================================================

DELETE_POINTS_HELP = """**Delete Points:**

Enter point indices to exclude from fitting:
- Single index: `5`
- Multiple indices: `1,3,5`
- Range (hyphen): `5-10`
- Range (colon): `5:10`
- Mixed: `1,3,5-10,15,20:25`

Excluded points will be marked on the plot but not used in fitting.
"""

# =============================================================================
# Cycle Mode Help
# =============================================================================

CYCLE_MODE_HELP = """**Cycle Mode:**

Separate temperature data into different cycles:
- **heating**: Temperature increasing
- **cooling**: Temperature decreasing
- **1st, 2nd, 3rd, 4th**: Numbered measurement cycles

Each cycle is displayed with distinct markers/lines.
Use "Auto Assign" to detect heating/cooling automatically.
"""

# =============================================================================
# R Element Labels Help
# =============================================================================

R_LABELS_HELP = """**R Element Labels:**

Assign physical meaning to fitted resistance values:
- **bulk**: Bulk (grain interior) resistance
- **gb**: Grain boundary resistance
- **electrode**: Electrode interface resistance
- **total**: Total resistance (for single R element)

Labels are used in conductivity calculations and Arrhenius plots.
"""
