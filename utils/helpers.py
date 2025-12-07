"""
Helper functions for EIS Analyzer
"""
import re


def format_param_name(name: str) -> str:
    """
    Convert parameter names to human-readable format.

    Examples:
        CPE1_0 -> CPE1_Q
        CPE1_1 -> CPE1_α

    Parameters
    ----------
    name : str
        Parameter name from circuit model

    Returns
    -------
    str
        Formatted parameter name
    """
    if '_0' in name and 'CPE' in name:
        return name.replace('_0', '_Q')
    elif '_1' in name and 'CPE' in name:
        return name.replace('_1', '_α')
    return name


def parse_temperature_pattern(pattern_str: str) -> list:
    """
    Parse temperature pattern string.
    Supports mixed format: single values and patterns.

    Examples:
        "25,[50,50,4]" -> [25, 50, 100, 150, 200]
        "[300,50,3]" -> [300, 350, 400]
        "25,50,[100,50,3]" -> [25, 50, 100, 150, 200]
        "[300,50,3],[450,-50,2]" -> [300, 350, 400, 450, 400]

    [T0,STEP,NUM] means: start at T0, add STEP, repeat NUM times

    Parameters
    ----------
    pattern_str : str
        Pattern string to parse

    Returns
    -------
    list
        List of temperature values
    """
    temperatures = []
    try:
        pattern_str = pattern_str.strip().replace(' ', '')
        if not pattern_str:
            return []

        # Parse the string character by character to handle mixed format
        i = 0
        current_token = ""

        while i < len(pattern_str):
            char = pattern_str[i]

            if char == '[':
                # Start of a pattern - find the matching ]
                j = i + 1
                bracket_content = ""
                while j < len(pattern_str) and pattern_str[j] != ']':
                    bracket_content += pattern_str[j]
                    j += 1

                # Parse bracket content as [T0, STEP, NUM]
                parts = bracket_content.split(',')
                if len(parts) == 3:
                    t0 = float(parts[0])
                    step = float(parts[1])
                    num = int(parts[2])
                    for k in range(num):
                        temperatures.append(t0 + step * k)

                i = j + 1  # Skip past ]
            elif char == ',':
                # End of a single value token
                if current_token:
                    temperatures.append(float(current_token))
                    current_token = ""
                i += 1
            else:
                # Part of a single value
                current_token += char
                i += 1

        # Don't forget the last token
        if current_token:
            temperatures.append(float(current_token))

    except (ValueError, IndexError):
        return []

    return temperatures


def extract_temp_from_filename(filename: str, pattern_str: str) -> float:
    """
    Extract temperature from filename using pattern.

    Pattern format: [separator, index], [separator, index], ...
    Each [separator, index] means: split by separator, take index-th element

    Examples:
        filename: "sample_300C_data.txt"
        pattern: "[_,1],[C,0]"
        result: splits by '_' -> takes index 1 -> "300C" -> splits by 'C' -> takes index 0 -> "300"

    Parameters
    ----------
    filename : str
        Filename to extract temperature from
    pattern_str : str
        Pattern string for extraction

    Returns
    -------
    float or None
        Extracted temperature value, or None if extraction failed
    """
    try:
        pattern_str = pattern_str.strip()
        if not pattern_str:
            return None

        result = filename
        patterns = pattern_str.replace(' ', '').split('],[')

        for p in patterns:
            p = p.strip('[]')
            parts = p.split(',')
            if len(parts) != 2:
                continue

            separator = parts[0]
            idx = int(parts[1])

            split_parts = result.split(separator)
            if idx < len(split_parts):
                result = split_parts[idx]
            else:
                return None

        # Try to extract number from result
        numbers = re.findall(r'[-+]?\d*\.?\d+', result)
        if numbers:
            return float(numbers[0])

    except (ValueError, IndexError):
        return None

    return None


def format_scientific(value: float, precision: int = 2) -> str:
    """
    Format a number in scientific notation.

    Parameters
    ----------
    value : float
        Number to format
    precision : int
        Decimal places

    Returns
    -------
    str
        Formatted string (e.g., "1.23e+05")
    """
    return f"{value:.{precision}e}"


def format_with_error(value: float, error: float, precision: int = 2) -> str:
    """
    Format a value with its error.

    Parameters
    ----------
    value : float
        Main value
    error : float
        Error value
    precision : int
        Decimal places

    Returns
    -------
    str
        Formatted string (e.g., "1.23e+05 ± 1.2e+03")
    """
    return f"{value:.{precision}e} ± {error:.{precision}e}"
