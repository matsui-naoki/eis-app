"""
Igor Pro export functionality for EIS Analyzer.
Generates Igor Text Files (.itx) and Procedure Files (.ipf) with publication-quality plot settings.
"""
import numpy as np


def generate_igor_procedure_file() -> str:
    """
    Generate Igor Procedure File (.ipf) for Arrhenius plot top axis.
    This file should be included or opened in Igor before loading the ITX.
    """
    return '''#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3
#include <TransformAxis1.2>

// EIS Analyzer - Axis transform function for Arrhenius plot
// Load and compile this procedure file before loading the ITX data file.

// Transform function: 1000/T (K^-1) -> t (°C)
Function TransAx_ArrTempToTemp(w, x)
    Wave/Z w
    Variable x
    return 1000/x - 273.15
End
'''


def make_wave_name(base_name: str, suffix: str, common_prefix: str = "", max_len: int = 31) -> str:
    """Create Igor-compatible wave name (max 31 chars, alphanumeric + underscore)"""
    # Remove common prefix from filename
    name = base_name
    if common_prefix and name.startswith(common_prefix):
        name = name[len(common_prefix):]

    # Remove file extension
    name = name.rsplit('.', 1)[0] if '.' in name else name

    # Sanitize: only alphanumeric and underscore
    name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)

    # Remove leading underscores
    name = name.lstrip('_')

    # If starts with digit, add prefix
    if name and name[0].isdigit():
        name = 'w' + name

    # Combine with suffix
    full_name = f"{name}_{suffix}" if name else suffix

    # Truncate to max length
    if len(full_name) > max_len:
        # Keep suffix, truncate name part
        suffix_len = len(suffix) + 1  # +1 for underscore
        name_len = max_len - suffix_len
        if name_len > 0:
            full_name = f"{name[:name_len]}_{suffix}"
        else:
            full_name = suffix[:max_len]

    return full_name


def find_common_prefix(filenames: list) -> str:
    """Find common prefix among filenames"""
    if len(filenames) <= 1:
        return ""

    common_prefix = ""
    min_len = min(len(f) for f in filenames)
    for i in range(min_len):
        chars = set(f[i] for f in filenames)
        if len(chars) == 1:
            common_prefix += filenames[0][i]
        else:
            break
    return common_prefix


def generate_igor_procedures() -> str:
    """Generate Igor procedure code as comments (ITX cannot define procedures directly)"""
    # ITX files cannot define procedures - return empty string
    # Users need to add functions to their procedure file manually
    return ""


def generate_nyquist_style(axis_max: float) -> list:
    """Generate Nyquist plot style commands for combined plot"""
    return [
        "X ModifyGraph mode=3,marker=19,msize=4",
        "X ModifyGraph mrkThick=0.5,useMrkStrokeRGB=1",
        "X ModifyGraph tick=2,minor=0,btLen=5",
        "X ModifyGraph mirror=2,gfSize=14,gFont=\"Arial\"",
        "X ModifyGraph width=198,height=198",
        "X ModifyGraph margin(left)=50,margin(bottom)=45,margin(top)=10,margin(right)=10",
        "X Label left \"-\\\\f02Z\\\\f00'' / Ω\"",
        "X Label bottom \"\\\\f02Z\\\\f00' / Ω\"",
        f"X SetAxis left 0,{axis_max:.6E}",
        f"X SetAxis bottom 0,{axis_max:.6E}",
        "X ModifyGraph width={Aspect,1}"
    ]


def generate_nyquist_individual_style(axis_max: float, wave_Zi: str, wave_Zi_fit: str = None) -> list:
    """Generate Nyquist plot style commands for individual plot"""
    style_lines = [
        "X ModifyGraph mode=3,msize=4",
        "X ModifyGraph mrkThick=0.5,useMrkStrokeRGB=1",
        "X ModifyGraph tick=2,minor=0,btLen=5",
        "X ModifyGraph mirror=2,gfSize=14,gFont=\"Arial\"",
        "X ModifyGraph width=198,height=198",
        "X ModifyGraph margin(left)=50,margin(bottom)=45,margin(top)=10,margin(right)=10",
        "X Label left \"-\\\\f02Z\\\\f00'' / Ω\"",
        "X Label bottom \"\\\\f02Z\\\\f00' / Ω\"",
        f"X SetAxis left 0,{axis_max:.6E}",
        f"X SetAxis bottom 0,{axis_max:.6E}",
        "X ModifyGraph width={Aspect,1}",
        # White filled marker with opaque
        f"X ModifyGraph marker({wave_Zi})=19,opaque({wave_Zi})=1,rgb({wave_Zi})=(65535,65535,65535)",
    ]
    if wave_Zi_fit:
        style_lines.append(f"X ModifyGraph mode({wave_Zi_fit})=0,lsize({wave_Zi_fit})=0.5,rgb({wave_Zi_fit})=(65535,0,0)")
    return style_lines


def generate_arrhenius_style_log_sigma(wave_total: str, wave_bulk: str, wave_gb: str,
                                        inv_T_min: float, inv_T_max: float,
                                        log_sigma_min: float, log_sigma_max: float) -> list:
    """Generate Arrhenius plot style commands for log(sigma) vs 1000/T.

    Requires TransAx_ArrTempToTemp function to be available.
    """
    # Add margin to axis range (10% on each side)
    x_range = inv_T_max - inv_T_min
    x_margin = x_range * 0.1 if x_range > 0 else 0.1
    bottom_min = inv_T_min - x_margin
    bottom_max = inv_T_max + x_margin

    y_range = log_sigma_max - log_sigma_min
    y_margin = y_range * 0.1 if y_range > 0 else 0.5
    left_min = log_sigma_min - y_margin
    left_max = log_sigma_max + y_margin

    style_lines = [
        # Marker style
        "X ModifyGraph mode=3,msize=4",
        "X ModifyGraph mrkThick=0.5,useMrkStrokeRGB=1",
        f"X ModifyGraph rgb({wave_total})=(0,0,0),rgb({wave_bulk})=(65535,0,0),rgb({wave_gb})=(0,0,65535)",
        f"X ModifyGraph marker({wave_total})=19,marker({wave_bulk})=16,marker({wave_gb})=17",
        # Publication style
        "X ModifyGraph tick=2,minor=0,btLen=5",
        "X ModifyGraph mirror=2",
        "X ModifyGraph gFont=\"Arial\",gfSize=14",
        "X ModifyGraph width=198,height=170",
        "X ModifyGraph margin(left)=55,margin(bottom)=45,margin(top)=39,margin(right)=10",
        "X ModifyGraph standoff(bottom)=0",
        "X ModifyGraph tlOffset=0",
        "X ModifyGraph lblMargin(bottom)=5",
        # Labels
        "X Label left \"log(\\\\f02σ\\\\f00 / S cm\\\\S–1\\\\M)\"",
        "X Label bottom \"1000\\\\f02T\\\\f00\\\\S–1\\\\M / K\\\\S–1\\\\M\"",
        "X Legend/C/N=text1/F=0/B=1",
        # Axis range
        f"X SetAxis bottom {bottom_min:.2f}, {bottom_max:.2f}",
        f"X SetAxis left {left_min:.2f}, {left_max:.2f}",
        # Top axis with transform
        "X NewDataFolder/O root:Packages",
        "X SetupTransformMirrorAxis(\"\", \"bottom\", \"TransAx_ArrTempToTemp\", $\"\", 3, 0, 5, 0)",
        "X ModifyGraph standoff(MT_bottom)=0",
        "X ModifyGraph tlOffset(MT_bottom)=-1",
        "X ModifyGraph btLen(MT_bottom)=0,stLen(MT_bottom)=0",
        "X ModifyGraph lblPos(MT_bottom)=30",
        "X Label MT_bottom \"\\\\f02t\\\\f00 / ºC\"",
    ]

    return style_lines


def generate_arrhenius_style(wave_total: str, wave_bulk: str, wave_gb: str,
                              inv_T_min: float, inv_T_max: float,
                              log_sigma_T_min: float, log_sigma_T_max: float) -> list:
    """Generate Arrhenius plot style commands with top axis showing temperature in Celsius.

    Requires TransAx_ArrTempToTemp function to be available.

    Args:
        wave_total, wave_bulk, wave_gb: Wave names for traces
        inv_T_min, inv_T_max: Data range for 1000/T axis
        log_sigma_T_min, log_sigma_T_max: Data range for log(sigma*T) axis
    """
    # Add margin to axis range (10% on each side)
    x_range = inv_T_max - inv_T_min
    x_margin = x_range * 0.1 if x_range > 0 else 0.1
    bottom_min = inv_T_min - x_margin
    bottom_max = inv_T_max + x_margin

    y_range = log_sigma_T_max - log_sigma_T_min
    y_margin = y_range * 0.1 if y_range > 0 else 0.5
    left_min = log_sigma_T_min - y_margin
    left_max = log_sigma_T_max + y_margin

    style_lines = [
        # Marker style: mode=3 (markers), different markers and colors for each trace
        "X ModifyGraph mode=3,msize=4",
        "X ModifyGraph mrkThick=0.5,useMrkStrokeRGB=1",
        f"X ModifyGraph rgb({wave_total})=(0,0,0),rgb({wave_bulk})=(65535,0,0),rgb({wave_gb})=(0,0,65535)",
        f"X ModifyGraph marker({wave_total})=19,marker({wave_bulk})=16,marker({wave_gb})=17",
        # Publication style
        "X ModifyGraph tick=2,minor=0,btLen=5",
        "X ModifyGraph mirror=2",
        "X ModifyGraph gFont=\"Arial\",gfSize=14",
        "X ModifyGraph width=198,height=170",
        "X ModifyGraph margin(left)=55,margin(bottom)=45,margin(top)=39,margin(right)=10",
        "X ModifyGraph standoff(bottom)=0",
        "X ModifyGraph tlOffset=0",
        "X ModifyGraph lblMargin(bottom)=5",
        # Labels
        "X Label left \"log(\\\\f02σT\\\\f00 / S K cm\\\\S–1\\\\M)\"",
        "X Label bottom \"1000\\\\f02T\\\\f00\\\\S–1\\\\M / K\\\\S–1\\\\M\"",
        "X Legend/C/N=text1/F=0/B=1",
        # Axis range
        f"X SetAxis bottom {bottom_min:.2f}, {bottom_max:.2f}",
        f"X SetAxis left {left_min:.2f}, {left_max:.2f}",
        # Top axis with transform (requires eis_procedures.ipf with TransAx_ArrTempToTemp)
        "X NewDataFolder/O root:Packages",
        "X SetupTransformMirrorAxis(\"\", \"bottom\", \"TransAx_ArrTempToTemp\", $\"\", 3, 0, 5, 0)",
        "X ModifyGraph standoff(MT_bottom)=0",
        "X ModifyGraph tlOffset(MT_bottom)=-1",
        "X ModifyGraph btLen(MT_bottom)=0,stLen(MT_bottom)=0",
        "X ModifyGraph lblPos(MT_bottom)=30",
        "X Label MT_bottom \"\\\\f02t\\\\f00 / ºC\"",
    ]

    return style_lines


def generate_igor_file(files_data: dict, sample_info: dict, r_labels: dict, export_rows: list = None) -> str:
    """
    Generate Igor Text File (.itx) with:
    - Raw data (freq, Z_real, Z_imag) for each file
    - Fit curves for each file
    - Nyquist plot with all data
    - Individual Nyquist plots with fit
    - Arrhenius plot data with top axis

    Args:
        files_data: Dictionary of {filename: file_data} from session state
        sample_info: Sample information dictionary
        r_labels: R element labels dictionary
        export_rows: Optional pre-computed export rows for Arrhenius data

    Returns:
        Igor Text File content as string
    """
    lines = ["IGOR"]

    # Add procedure section for transform functions
    lines.append(generate_igor_procedures())

    # Get sample name for wave prefix
    sample_name = sample_info.get('name', '').strip()

    # Get all filenames and find common prefix to remove
    filenames = list(files_data.keys())
    if not filenames:
        return "IGOR\n"

    common_prefix = find_common_prefix(filenames)

    # Collect data for all files
    nyquist_data_waves = []  # For combined Nyquist plot

    # Track max values for unified axis range
    global_max_Zr = 0
    global_max_Zi = 0

    for fname in filenames:
        fdata = files_data[fname]
        freq = fdata.get('freq')
        Z = fdata.get('Z')

        if freq is None or Z is None:
            continue

        Z_real = np.real(Z)
        Z_imag = -np.imag(Z)  # -Z'' for Nyquist plot

        # Calculate max for this file (for individual plot axis range)
        file_max_Zr = np.max(Z_real)
        file_max_Zi = np.max(Z_imag)
        file_axis_max = max(file_max_Zr, file_max_Zi) * 1.05

        # Update global max
        global_max_Zr = max(global_max_Zr, file_max_Zr)
        global_max_Zi = max(global_max_Zi, file_max_Zi)

        # Create wave names
        wave_freq = make_wave_name(fname, 'freq', common_prefix)
        wave_Zr = make_wave_name(fname, 'Zr', common_prefix)
        wave_Zi = make_wave_name(fname, 'Zi', common_prefix)

        nyquist_data_waves.append((wave_Zr, wave_Zi, fname, file_axis_max))

        # Write raw data waves
        lines.append(f"WAVES/O {wave_freq}, {wave_Zr}, {wave_Zi}")
        lines.append("BEGIN")
        for i in range(len(freq)):
            lines.append(f"  {freq[i]:.6E}  {Z_real[i]:.6E}  {Z_imag[i]:.6E}")
        lines.append("END")
        lines.append("")

        # Write fit data if available
        Z_fit = fdata.get('Z_fit')
        if Z_fit is not None:
            Z_fit_real = np.real(Z_fit)
            Z_fit_imag = -np.imag(Z_fit)  # -Z'' for Nyquist

            wave_Zr_fit = make_wave_name(fname, 'Zr_fit', common_prefix)
            wave_Zi_fit = make_wave_name(fname, 'Zi_fit', common_prefix)

            lines.append(f"WAVES/O {wave_Zr_fit}, {wave_Zi_fit}")
            lines.append("BEGIN")
            for i in range(len(Z_fit)):
                lines.append(f"  {Z_fit_real[i]:.6E}  {Z_fit_imag[i]:.6E}")
            lines.append("END")
            lines.append("")

    # =========================================
    # Arrhenius plot data
    # =========================================
    inv_T = []
    log_sigma_bulk_T = []
    log_sigma_gb_T = []
    log_sigma_total_T = []
    log_sigma_bulk = []
    log_sigma_gb = []
    log_sigma_total = []
    wave_inv_T = None
    wave_bulk = None
    wave_gb = None
    wave_total = None
    wave_ls_bulk = None
    wave_ls_gb = None
    wave_ls_total = None

    if export_rows:
        # Use provided export_rows
        for row in export_rows:
            inv_t_val = row.get('1000/T (K-1)', '-')
            if inv_t_val != '-':
                inv_t = float(inv_t_val)
                inv_T.append(inv_t)
                temp_K = 1000.0 / inv_t  # Calculate T from 1000/T

                bulk_val = row.get('log(sigma_bulk*T)', '-')
                if bulk_val != '-':
                    log_sigma_bulk_T.append(float(bulk_val))
                    log_sigma_bulk.append(float(bulk_val) - np.log10(temp_K))
                else:
                    log_sigma_bulk_T.append(float('nan'))
                    log_sigma_bulk.append(float('nan'))

                gb_val = row.get('log(sigma_gb*T)', '-')
                if gb_val != '-':
                    log_sigma_gb_T.append(float(gb_val))
                    log_sigma_gb.append(float(gb_val) - np.log10(temp_K))
                else:
                    log_sigma_gb_T.append(float('nan'))
                    log_sigma_gb.append(float('nan'))

                total_val = row.get('log(sigma_total*T)', '-')
                if total_val != '-':
                    log_sigma_total_T.append(float(total_val))
                    log_sigma_total.append(float(total_val) - np.log10(temp_K))
                else:
                    log_sigma_total_T.append(float('nan'))
                    log_sigma_total.append(float('nan'))
    else:
        # Generate from file data directly
        for fname in filenames:
            fdata = files_data[fname]
            temp = fdata.get('temperature')
            if temp is None or temp <= 0:
                continue

            temp_K = temp
            inv_t = 1000.0 / temp_K
            inv_T.append(inv_t)

            # Get conductivities
            r_sigmas = fdata.get('r_sigmas', {})
            total_sigma = fdata.get('total_sigma')

            # Find bulk and gb from r_labels
            bulk_sigma = None
            gb_sigma = None
            for r_name, label in r_labels.items():
                if label == 'bulk' and r_name in r_sigmas:
                    bulk_sigma = r_sigmas[r_name]
                elif label == 'gb' and r_name in r_sigmas:
                    gb_sigma = r_sigmas[r_name]

            # Calculate log(sigma*T) and log(sigma)
            if bulk_sigma and bulk_sigma > 0:
                log_sigma_bulk_T.append(np.log10(bulk_sigma * temp_K))
                log_sigma_bulk.append(np.log10(bulk_sigma))
            else:
                log_sigma_bulk_T.append(float('nan'))
                log_sigma_bulk.append(float('nan'))

            if gb_sigma and gb_sigma > 0:
                log_sigma_gb_T.append(np.log10(gb_sigma * temp_K))
                log_sigma_gb.append(np.log10(gb_sigma))
            else:
                log_sigma_gb_T.append(float('nan'))
                log_sigma_gb.append(float('nan'))

            if total_sigma and total_sigma > 0:
                log_sigma_total_T.append(np.log10(total_sigma * temp_K))
                log_sigma_total.append(np.log10(total_sigma))
            else:
                log_sigma_total_T.append(float('nan'))
                log_sigma_total.append(float('nan'))

    if inv_T:
        # Create Arrhenius wave names
        arr_prefix = "".join(c if c.isalnum() or c == '_' else '_' for c in sample_name) if sample_name else "arr"
        if arr_prefix and arr_prefix[0].isdigit():
            arr_prefix = 'w' + arr_prefix

        wave_inv_T = f"{arr_prefix}_invT"[:31]
        # log(sigma*T) waves
        wave_bulk = f"{arr_prefix}_lsT_bulk"[:31]
        wave_gb = f"{arr_prefix}_lsT_gb"[:31]
        wave_total = f"{arr_prefix}_lsT_total"[:31]
        # log(sigma) waves
        wave_ls_bulk = f"{arr_prefix}_ls_bulk"[:31]
        wave_ls_gb = f"{arr_prefix}_ls_gb"[:31]
        wave_ls_total = f"{arr_prefix}_ls_total"[:31]

        # log(sigma*T) data waves
        lines.append(f"WAVES/O {wave_inv_T}, {wave_bulk}, {wave_gb}, {wave_total}")
        lines.append("BEGIN")
        for i in range(len(inv_T)):
            bulk_str = f"{log_sigma_bulk_T[i]:.5E}" if not np.isnan(log_sigma_bulk_T[i]) else "NaN"
            gb_str = f"{log_sigma_gb_T[i]:.5E}" if not np.isnan(log_sigma_gb_T[i]) else "NaN"
            total_str = f"{log_sigma_total_T[i]:.5E}" if not np.isnan(log_sigma_total_T[i]) else "NaN"
            lines.append(f"  {inv_T[i]:.5E}  {bulk_str}  {gb_str}  {total_str}")
        lines.append("END")
        lines.append("")

        # log(sigma) data waves
        lines.append(f"WAVES/O {wave_ls_bulk}, {wave_ls_gb}, {wave_ls_total}")
        lines.append("BEGIN")
        for i in range(len(inv_T)):
            bulk_str = f"{log_sigma_bulk[i]:.5E}" if not np.isnan(log_sigma_bulk[i]) else "NaN"
            gb_str = f"{log_sigma_gb[i]:.5E}" if not np.isnan(log_sigma_gb[i]) else "NaN"
            total_str = f"{log_sigma_total[i]:.5E}" if not np.isnan(log_sigma_total[i]) else "NaN"
            lines.append(f"  {bulk_str}  {gb_str}  {total_str}")
        lines.append("END")
        lines.append("")

    # =========================================
    # Igor commands for plots
    # =========================================

    # Calculate unified axis range (max of Z' and -Z'', min = 0)
    axis_max = max(global_max_Zr, global_max_Zi) * 1.05 if global_max_Zr > 0 or global_max_Zi > 0 else 1.0

    nyquist_style_lines = generate_nyquist_style(axis_max)

    # Color palette for Nyquist All plot (10 colors cycle) with alpha=32750
    nyquist_colors = [
        (0, 0, 0),           # black
        (65535, 0, 0),       # red
        (0, 0, 65535),       # blue
        (0, 39321, 0),       # green
        (39321, 39321, 39321), # gray
        (65535, 32768, 0),   # orange
        (0, 65535, 65535),   # cyan
        (65535, 0, 65535),   # magenta
        (32768, 0, 65535),   # purple
        (0, 32768, 0),       # dark green
    ]

    # Combined Nyquist plot (all data) with different colors
    if nyquist_data_waves:
        first_wave = nyquist_data_waves[0]
        lines.append(f"X Display {first_wave[1]} vs {first_wave[0]} as \"Nyquist_All\"")

        for wave_Zr, wave_Zi, _, _ in nyquist_data_waves[1:]:
            lines.append(f"X AppendToGraph {wave_Zi} vs {wave_Zr}")

        lines.extend(nyquist_style_lines)
        # Apply different colors with alpha to each trace
        for i, (wave_Zr, wave_Zi, _, _) in enumerate(nyquist_data_waves):
            color = nyquist_colors[i % len(nyquist_colors)]
            lines.append(f"X ModifyGraph rgb({wave_Zi})=({color[0]},{color[1]},{color[2]},32750)")
        lines.append("X Legend/C/N=text0/F=0/B=1")
        lines.append("")

    # Individual Nyquist plots with fit curves
    for wave_Zr, wave_Zi, fname, file_axis_max in nyquist_data_waves:
        fdata = files_data[fname]
        Z_fit = fdata.get('Z_fit')

        # Short name for window title
        short_name = fname.rsplit('.', 1)[0] if '.' in fname else fname
        if len(short_name) > 20:
            short_name = short_name[:20]

        lines.append(f"X Display {wave_Zi} vs {wave_Zr} as \"Nyquist_{short_name}\"")

        wave_Zi_fit = None
        if Z_fit is not None:
            wave_Zr_fit = make_wave_name(fname, 'Zr_fit', common_prefix)
            wave_Zi_fit = make_wave_name(fname, 'Zi_fit', common_prefix)
            lines.append(f"X AppendToGraph {wave_Zi_fit} vs {wave_Zr_fit}")

        lines.extend(generate_nyquist_individual_style(file_axis_max, wave_Zi, wave_Zi_fit))
        lines.append("")

    # Arrhenius plots
    if inv_T and wave_inv_T:
        # Get data range for axes
        inv_T_min = min(inv_T)
        inv_T_max = max(inv_T)

        # log(sigma*T) vs 1000/T plot
        lines.append(f"X Display {wave_total} vs {wave_inv_T} as \"Arrhenius_sigmaT\"")
        lines.append(f"X AppendToGraph {wave_bulk} vs {wave_inv_T}")
        lines.append(f"X AppendToGraph {wave_gb} vs {wave_inv_T}")
        # Combine all log(sigma*T) values, excluding NaN
        all_log_sigma_T = [v for v in log_sigma_bulk_T + log_sigma_gb_T + log_sigma_total_T if not np.isnan(v)]
        if all_log_sigma_T:
            log_sigma_T_min = min(all_log_sigma_T)
            log_sigma_T_max = max(all_log_sigma_T)
        else:
            log_sigma_T_min = -5
            log_sigma_T_max = 5
        lines.extend(generate_arrhenius_style(wave_total, wave_bulk, wave_gb, inv_T_min, inv_T_max, log_sigma_T_min, log_sigma_T_max))
        lines.append("")

        # log(sigma) vs 1000/T plot
        lines.append(f"X Display {wave_ls_total} vs {wave_inv_T} as \"Arrhenius_sigma\"")
        lines.append(f"X AppendToGraph {wave_ls_bulk} vs {wave_inv_T}")
        lines.append(f"X AppendToGraph {wave_ls_gb} vs {wave_inv_T}")
        # Combine all log(sigma) values, excluding NaN
        all_log_sigma = [v for v in log_sigma_bulk + log_sigma_gb + log_sigma_total if not np.isnan(v)]
        if all_log_sigma:
            log_sigma_min = min(all_log_sigma)
            log_sigma_max = max(all_log_sigma)
        else:
            log_sigma_min = -5
            log_sigma_max = 5
        lines.extend(generate_arrhenius_style_log_sigma(wave_ls_total, wave_ls_bulk, wave_ls_gb, inv_T_min, inv_T_max, log_sigma_min, log_sigma_max))
        lines.append("")

    return "\n".join(lines) + "\n"
