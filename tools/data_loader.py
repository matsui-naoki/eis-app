"""
Data loader module for EIS impedance files
Uses impedance library for file parsing
Supports multiple formats: .mpt (BioLogic), .z (ZPlot), .DTA (Gamry), etc.
"""

import numpy as np
import io
import os
import tempfile
from typing import Tuple, Optional, List, Dict

# Use impedance library for file parsing
from impedance import preprocessing


def split_loop_data(frequencies: np.ndarray, Z: np.ndarray,
                    rtol: float = 0.01) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split loop data into separate datasets based on repeated first frequency.

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequency values (Hz)
    Z : np.ndarray
        Array of complex impedance values (Ohm)
    rtol : float
        Relative tolerance for frequency comparison (default: 1%)

    Returns
    -------
    datasets : list of tuple
        List of (freq, Z) tuples, one per loop
    """
    if len(frequencies) == 0:
        return []

    first_freq = frequencies[0]

    # Find indices where frequency matches the first frequency (with tolerance)
    # Skip index 0 since that's the start
    split_indices = [0]

    for i in range(1, len(frequencies)):
        if np.isclose(frequencies[i], first_freq, rtol=rtol):
            split_indices.append(i)

    # If no splits found (only one dataset), return as-is
    if len(split_indices) == 1:
        return [(frequencies, Z)]

    # Split the data
    datasets = []
    for i in range(len(split_indices)):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1] if i + 1 < len(split_indices) else len(frequencies)

        freq_segment = frequencies[start_idx:end_idx]
        Z_segment = Z[start_idx:end_idx]

        if len(freq_segment) > 0:
            datasets.append((freq_segment, Z_segment))

    return datasets


def load_uploaded_file(uploaded_file, file_extension: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Load impedance data from an uploaded file in Streamlit

    Parameters
    ----------
    uploaded_file : UploadedFile
        File uploaded via Streamlit file_uploader
    file_extension : str
        File extension ('.mpt', '.z', '.DTA', etc.)

    Returns
    -------
    frequencies : np.ndarray or None
        Array of frequency values (Hz)
    Z : np.ndarray or None
        Array of complex impedance values (Ohm)
    error_message : str or None
        Error message if loading failed
    """
    try:
        # Read file content
        bytes_data = uploaded_file.read()

        # Create a temporary file for impedance library to read
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(bytes_data)
            tmp_path = tmp_file.name

        try:
            # Determine instrument type from extension
            instrument = get_instrument_type(file_extension)

            # Use impedance library to read file
            frequencies, Z = preprocessing.readFile(tmp_path, instrument=instrument)

            return frequencies, Z, None

        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

    except Exception as e:
        return None, None, f"Error loading file: {str(e)}"


def load_uploaded_file_with_loops(uploaded_file, file_extension: str,
                                   base_filename: str,
                                   rtol: float = 0.01) -> Tuple[List[Dict], Optional[str]]:
    """
    Load impedance data from an uploaded file, splitting loop data into separate datasets.

    Parameters
    ----------
    uploaded_file : UploadedFile
        File uploaded via Streamlit file_uploader
    file_extension : str
        File extension ('.mpt', '.z', '.DTA', etc.)
    base_filename : str
        Base filename for naming split datasets
    rtol : float
        Relative tolerance for frequency comparison when detecting loops

    Returns
    -------
    datasets : list of dict
        List of dictionaries with keys: 'name', 'freq', 'Z'
        Names are formatted as 'filename' for single data, 'filename_1', 'filename_2' for loops
    error_message : str or None
        Error message if loading failed
    """
    # Load the raw data
    freq, Z, error = load_uploaded_file(uploaded_file, file_extension)

    if error:
        return [], error

    if freq is None or Z is None:
        return [], "Failed to load data"

    # Split loop data
    split_data = split_loop_data(freq, Z, rtol=rtol)

    if len(split_data) == 0:
        return [], "No valid data found"

    # Create named datasets
    datasets = []

    if len(split_data) == 1:
        # Single dataset - use original filename
        datasets.append({
            'name': base_filename,
            'freq': split_data[0][0],
            'Z': split_data[0][1]
        })
    else:
        # Multiple datasets - add numbered suffix
        for i, (f, z) in enumerate(split_data, start=1):
            datasets.append({
                'name': f"{base_filename}_{i}",
                'freq': f,
                'Z': z
            })

    return datasets, None


def get_instrument_type(file_extension: str) -> Optional[str]:
    """
    Map file extension to instrument type for impedance library

    Parameters
    ----------
    file_extension : str
        File extension

    Returns
    -------
    str or None
        Instrument type string or None for CSV
    """
    extension_map = {
        '.mpt': 'biologic',
        '.z': 'zplot',
        '.dta': 'gamry',
        '.par': 'versastudio',
        '.txt': None,  # Could be parstat, powersuite, or chinstruments - try CSV
        '.csv': None,
    }
    return extension_map.get(file_extension.lower())


def load_from_path(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load impedance data from a file path

    Parameters
    ----------
    file_path : str
        Path to the impedance data file

    Returns
    -------
    frequencies : np.ndarray
        Array of frequency values
    Z : np.ndarray
        Array of complex impedance values
    """
    file_extension = os.path.splitext(file_path)[1]
    instrument = get_instrument_type(file_extension)
    return preprocessing.readFile(file_path, instrument=instrument)


def validate_impedance_data(frequencies: np.ndarray, Z: np.ndarray) -> Tuple[bool, str]:
    """
    Validate impedance data

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequency values
    Z : np.ndarray
        Array of complex impedance values

    Returns
    -------
    is_valid : bool
        True if data is valid
    message : str
        Validation message
    """
    if len(frequencies) == 0 or len(Z) == 0:
        return False, "Empty data"

    if len(frequencies) != len(Z):
        return False, "Frequency and impedance arrays have different lengths"

    if np.any(frequencies <= 0):
        return False, "Frequency values must be positive"

    if np.any(~np.isfinite(Z)):
        return False, "Impedance contains invalid values (NaN or Inf)"

    return True, "Data is valid"


def crop_frequencies(frequencies: np.ndarray, Z: np.ndarray,
                     freq_min: float = 0, freq_max: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop data to specified frequency range

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequency values
    Z : np.ndarray
        Array of complex impedance values
    freq_min : float
        Minimum frequency (default: 0)
    freq_max : float
        Maximum frequency (default: None, no limit)

    Returns
    -------
    frequencies : np.ndarray
        Cropped frequency array
    Z : np.ndarray
        Cropped impedance array
    """
    return preprocessing.cropFrequencies(frequencies, Z, freqmin=freq_min, freqmax=freq_max)


def ignore_below_x(frequencies: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove data points below the X-axis (positive imaginary part)

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequency values
    Z : np.ndarray
        Array of complex impedance values

    Returns
    -------
    frequencies : np.ndarray
        Filtered frequency array
    Z : np.ndarray
        Filtered impedance array
    """
    return preprocessing.ignoreBelowX(frequencies, Z)


def save_csv(filename: str, frequencies: np.ndarray, impedances: np.ndarray, **kwargs):
    """
    Save frequencies and impedances to a CSV file

    Parameters
    ----------
    filename : str
        Output filename
    frequencies : np.ndarray
        Array of frequencies
    impedances : np.ndarray
        Array of complex impedances
    kwargs :
        Keyword arguments passed to np.savetxt
    """
    preprocessing.saveCSV(filename, frequencies, impedances, **kwargs)
