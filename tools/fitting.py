"""
Custom fitting functions for EIS circuit analysis
Supports multiple weight methods: modulus, squared_modulus, proportional
"""

import warnings
import numpy as np
from scipy.linalg import inv
from scipy.optimize import curve_fit, basinhopping
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Use elements from impedance library
from impedance.models.circuits.elements import circuit_elements, get_element_from_name
from impedance.models.circuits.fitting import (
    buildCircuit, extract_circuit_elements, calculateCircuitLength
)


class FittingTimeoutError(Exception):
    """Exception raised when fitting times out."""
    pass


def _run_with_timeout(func, timeout_sec, *args, **kwargs):
    """
    Run a function with a timeout using ThreadPoolExecutor.

    Parameters
    ----------
    func : callable
        Function to execute
    timeout_sec : float
        Timeout in seconds
    *args, **kwargs
        Arguments to pass to the function

    Returns
    -------
    Result of func(*args, **kwargs)

    Raises
    ------
    FittingTimeoutError
        If the function does not complete within timeout_sec

    Note
    ----
    On timeout, the background thread continues running but its result is
    discarded. This prevents blocking the main thread while still allowing
    the fitting to complete in the background (which will be garbage collected).
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        result = future.result(timeout=timeout_sec)
        executor.shutdown(wait=False)
        return result
    except FuturesTimeoutError:
        # Don't wait for the thread to complete - just abandon it
        # The thread will continue in background but result is discarded
        executor.shutdown(wait=False)
        raise FittingTimeoutError(f"Fitting timed out after {timeout_sec} seconds")


def rmse(a, b):
    """
    Calculate root mean squared error between two vectors.
    """
    n = len(a)
    return np.linalg.norm(a - b) / np.sqrt(n)


def set_default_bounds(circuit, constants={}):
    """
    Set default bounds for optimization.
    CPE and La alphas have upper bound of 1, others have np.inf.
    """
    extracted_elements = extract_circuit_elements(circuit)

    lower_bounds, upper_bounds = [], []
    for elem in extracted_elements:
        raw_element = get_element_from_name(elem)
        for i in range(check_and_eval(raw_element).num_params):
            if elem in constants or elem + f'_{i}' in constants:
                continue
            if raw_element in ['CPE', 'La'] and i == 1:
                upper_bounds.append(1)
            else:
                upper_bounds.append(np.inf)
            lower_bounds.append(0)

    bounds = ((lower_bounds), (upper_bounds))
    return bounds


def check_and_eval(element):
    """
    Check if an element is valid, then evaluate it.
    """
    allowed_elements = circuit_elements.keys()
    if element not in allowed_elements:
        raise ValueError(f'{element} not in allowed elements ({allowed_elements})')
    else:
        return eval(element, circuit_elements)


def wrapCircuit(circuit, constants):
    """
    Wrap circuit function so we can pass the circuit string.
    """
    def wrappedCircuit(frequencies, *parameters):
        x = eval(buildCircuit(circuit, frequencies, *parameters,
                              constants=constants, eval_string='',
                              index=0)[0],
                 circuit_elements)
        y_real = np.real(x)
        y_imag = np.imag(x)
        return np.hstack([y_real, y_imag])
    return wrappedCircuit


def _do_curve_fit_core(circuit, constants, f, Z, initial_guess, bounds, weight_method, kwargs):
    """Core curve_fit implementation without timeout."""
    fit_kwargs = kwargs.copy()

    if 'maxfev' not in fit_kwargs:
        fit_kwargs['maxfev'] = 10000
    if 'ftol' not in fit_kwargs:
        fit_kwargs['ftol'] = 1e-10
    if 'xtol' not in fit_kwargs:
        fit_kwargs['xtol'] = 1e-10

    # Calculate weights based on the chosen method
    if weight_method == 'squared_modulus':
        sigma = np.abs(Z) ** 2
        fit_kwargs['sigma'] = np.hstack([sigma, sigma])
    elif weight_method == 'modulus':
        sigma = np.abs(Z)
        fit_kwargs['sigma'] = np.hstack([sigma, sigma])
    elif weight_method == 'proportional':
        weights = np.abs(Z)
        sigma = 1 / weights
        fit_kwargs['sigma'] = np.hstack([sigma, sigma])

    popt, pcov = curve_fit(wrapCircuit(circuit, constants), f,
                           np.hstack([Z.real, Z.imag]),
                           p0=initial_guess, bounds=bounds, **fit_kwargs)

    perror = np.sqrt(np.diag(pcov))
    return popt, perror


def _do_curve_fit(circuit, constants, f, Z, initial_guess, bounds, weight_method, kwargs, timeout_sec=None):
    """
    Internal function to perform curve_fit with optional timeout.

    Parameters
    ----------
    timeout_sec : float, optional
        Timeout in seconds. If None, no timeout is applied.
    """
    if timeout_sec is not None and timeout_sec > 0:
        # Use ThreadPoolExecutor-based timeout (works in Streamlit)
        return _run_with_timeout(
            _do_curve_fit_core,
            timeout_sec,
            circuit, constants, f, Z, initial_guess, bounds, weight_method, kwargs
        )
    else:
        # No timeout
        return _do_curve_fit_core(circuit, constants, f, Z, initial_guess, bounds, weight_method, kwargs)


def _do_basinhopping_core(circuit, constants, f, Z, initial_guess, bounds, kwargs):
    """Core basinhopping implementation without timeout."""
    fit_kwargs = kwargs.copy()

    if 'seed' not in fit_kwargs:
        fit_kwargs['seed'] = 0

    def opt_function(x):
        return rmse(wrapCircuit(circuit, constants)(f, *x),
                    np.hstack([Z.real, Z.imag]))

    class BasinhoppingBounds(object):
        def __init__(self, xmin, xmax):
            self.xmin = np.array(xmin)
            self.xmax = np.array(xmax)

        def __call__(self, **kwargs):
            x = kwargs['x_new']
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin

    basinhopping_bounds = BasinhoppingBounds(xmin=bounds[0], xmax=bounds[1])
    results = basinhopping(opt_function, x0=initial_guess,
                           accept_test=basinhopping_bounds, **fit_kwargs)
    popt = results.x

    jac = results.lowest_optimization_result['jac'][np.newaxis]
    try:
        pcov = inv(np.dot(jac.T, jac)) * opt_function(popt) ** 2
        perror = np.sqrt(np.diag(pcov))
    except (ValueError, np.linalg.LinAlgError):
        warnings.warn('Failed to compute perror')
        perror = None

    return popt, perror


def _do_basinhopping(circuit, constants, f, Z, initial_guess, bounds, kwargs, timeout_sec=None):
    """
    Internal function to perform basinhopping with optional timeout.

    Parameters
    ----------
    timeout_sec : float, optional
        Timeout in seconds. If None, no timeout is applied.
    """
    if timeout_sec is not None and timeout_sec > 0:
        # Use ThreadPoolExecutor-based timeout (works in Streamlit)
        return _run_with_timeout(
            _do_basinhopping_core,
            timeout_sec,
            circuit, constants, f, Z, initial_guess, bounds, kwargs
        )
    else:
        # No timeout
        return _do_basinhopping_core(circuit, constants, f, Z, initial_guess, bounds, kwargs)


def circuit_fit(frequencies, impedances, circuit, initial_guess, constants={},
                bounds=None, weight_method=None, global_opt=False,
                timeout=None, **kwargs):
    """
    Main function for fitting an equivalent circuit to data.

    Parameters
    ----------
    frequencies : numpy array
        Frequencies
    impedances : numpy array of dtype 'complex128'
        Impedances
    circuit : string
        String defining the equivalent circuit to be fit
    initial_guess : list of floats
        Initial guesses for the fit parameters
    constants : dictionary, optional
        Parameters and their values to hold constant during fitting
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters
    weight_method : str, optional
        Weighting method for fitting:
        - None: no weighting
        - 'modulus': weights are 1/|Z|
        - 'squared_modulus': weights are 1/|Z|^2
        - 'proportional': weights are |Z|
    global_opt : bool, optional
        If True, use basinhopping global optimization
    timeout : float, optional
        Timeout in seconds for the fitting operation.
        If fitting exceeds this time, FittingTimeoutError is raised.
        Default is None (no timeout).
        Requires timeout_decorator package to be installed.

    Returns
    -------
    p_values : list of floats
        Best fit parameters
    p_errors : list of floats
        One standard deviation error estimates

    Raises
    ------
    FittingTimeoutError
        If fitting exceeds the specified timeout.
    """
    f = np.array(frequencies, dtype=float)
    Z = np.array(impedances, dtype=complex)

    if bounds is None:
        bounds = set_default_bounds(circuit, constants=constants)

    # Limit maxfev to prevent very long fits
    if 'maxfev' not in kwargs:
        kwargs['maxfev'] = 5000  # Reduced default for faster timeout behavior

    if not global_opt:
        popt, perror = _do_curve_fit(
            circuit, constants, f, Z, initial_guess, bounds, weight_method, kwargs,
            timeout_sec=timeout
        )
    else:
        popt, perror = _do_basinhopping(
            circuit, constants, f, Z, initial_guess, bounds, kwargs,
            timeout_sec=timeout
        )

    return popt, perror


def calc_rmspe(Z, Z_fit):
    """
    Calculate Root Mean Square Percentage Error between measured and fitted impedance.

    Parameters
    ----------
    Z : np.ndarray
        Measured complex impedance
    Z_fit : np.ndarray
        Fitted complex impedance

    Returns
    -------
    float
        RMSPE value
    """
    Z_real = np.real(Z)
    Z_fit_real = np.real(Z_fit)
    Z_imag = -np.imag(Z)
    Z_fit_imag = -np.imag(Z_fit)

    percentage_errors_real = ((Z_fit_real - Z_real) / Z_real)
    percentage_errors_imag = ((Z_fit_imag - Z_imag) / Z_imag)

    all_errors = np.concatenate([percentage_errors_real**2, percentage_errors_imag**2])
    rmspe = np.sqrt(np.mean(all_errors))

    return rmspe


def r2sigma(R, S, L):
    """
    Calculate ionic conductivity from resistance, area, and thickness.

    Parameters
    ----------
    R : float
        Resistance (Ohm)
    S : float
        Area (cm^2)
    L : float
        Thickness (cm)

    Returns
    -------
    float
        Ionic conductivity (S/cm)
    """
    return L / (R * S)


def r2logsigma(R, S, L):
    """
    Calculate log10 of ionic conductivity.

    Returns
    -------
    float
        log10(sigma)
    """
    return np.log10(L / (R * S))


def effective_capacitance(R, Q, n):
    """
    Calculate effective capacitance of a CPE in parallel with a resistor.

    Parameters
    ----------
    R : float
        Resistance (Ohm)
    Q : float
        CPE pseudo-capacitance (F*s^(n-1))
    n : float
        CPE exponent (0 < n <= 1)

    Returns
    -------
    float
        Effective capacitance (F)
    """
    return Q**(1/n) * R**((1-n)/n)


def sort_ecm_by_cap(circuit_params, circuit_conf, param_names):
    """
    Sort the elements of an Equivalent Circuit Model (ECM) by their effective capacitance.
    CPE elements with smaller effective capacitance come first (bulk before grain boundary).

    Parameters
    ----------
    circuit_params : array-like
        Fitted parameter values
    circuit_conf : array-like
        Parameter confidence intervals (errors)
    param_names : list
        List of parameter names

    Returns
    -------
    dict
        Dictionary with sorted parameter names as keys and values/errors.
        Format: {'R1': value, 'R1_error': error, 'CPE1_0': value, ...}
        Also includes 'effective_caps': {element_name: ceff_value}
    """
    elements = []  # List of R-CPE parallel elements
    elements_woR = []  # List of standalone CPE elements
    cpes = []  # Effective capacitances for sorting

    circuit_params = np.array(circuit_params)
    circuit_conf = np.array(circuit_conf) if circuit_conf is not None else np.zeros_like(circuit_params)

    # Find CPE elements (assumed max of 4)
    for i in range(1, 5):
        cpe_name = f'CPE{i}'
        if any(cpe_name in name for name in param_names):
            my_elements = []
            my_values = []
            my_errors = []

            # Check if there's a parallel resistor with same number
            r_name = f'R{i}'
            if r_name in param_names:
                # R-CPE parallel element
                r_idx = param_names.index(r_name)
                R = circuit_params[r_idx]
                R_error = circuit_conf[r_idx]
                my_elements.append(r_name)
                my_values.append(R)
                my_errors.append(R_error)

                q_name = f'CPE{i}_0'
                q_idx = param_names.index(q_name)
                Q = circuit_params[q_idx]
                Q_error = circuit_conf[q_idx]
                my_elements.append(q_name)
                my_values.append(Q)
                my_errors.append(Q_error)

                n_name = f'CPE{i}_1'
                n_idx = param_names.index(n_name)
                n = circuit_params[n_idx]
                n_error = circuit_conf[n_idx]
                my_elements.append(n_name)
                my_values.append(n)
                my_errors.append(n_error)

                # Calculate effective capacitance
                ceff = effective_capacitance(R=R, Q=Q, n=n)
                cpes.append(ceff)
                elements.append((my_elements, my_values, my_errors))
            else:
                # Standalone CPE (no parallel resistor)
                q_name = f'CPE{i}_0'
                if q_name in param_names:
                    q_idx = param_names.index(q_name)
                    my_elements.append(q_name)
                    my_values.append(circuit_params[q_idx])
                    my_errors.append(circuit_conf[q_idx])

                    n_name = f'CPE{i}_1'
                    n_idx = param_names.index(n_name)
                    my_elements.append(n_name)
                    my_values.append(circuit_params[n_idx])
                    my_errors.append(circuit_conf[n_idx])

                    elements_woR.append((my_elements, my_values, my_errors))

    # Sort R-CPE elements by effective capacitance (ascending: bulk first, then gb)
    if len(cpes) > 0 and len(elements) > 0:
        sorted_data = sorted(zip(cpes, elements), key=lambda x: x[0])
        cpes_sorted, elements_sorted = zip(*sorted_data)
        elements = list(elements_sorted)
        cpes = list(cpes_sorted)

    # Build result dictionary with new names (R1, R2, ... in order of increasing Ceff)
    result = {}
    effective_caps = {}

    # Add sorted R-CPE elements with new numbering
    for new_idx, (elem_names, elem_values, elem_errors) in enumerate(elements, start=1):
        # elem_names = ['R2', 'CPE2_0', 'CPE2_1'] (original names)
        # new_idx = 1, 2, ... (new order by Ceff)

        # R element
        new_r_name = f'R{new_idx}'
        result[new_r_name] = elem_values[0]
        result[f'{new_r_name}_error'] = elem_errors[0]

        # CPE Q element
        new_q_name = f'CPE{new_idx}_0'
        result[new_q_name] = elem_values[1]
        result[f'{new_q_name}_error'] = elem_errors[1]

        # CPE n element
        new_n_name = f'CPE{new_idx}_1'
        result[new_n_name] = elem_values[2]
        result[f'{new_n_name}_error'] = elem_errors[2]

        # Store effective capacitance
        if new_idx - 1 < len(cpes):
            effective_caps[new_r_name] = cpes[new_idx - 1]

    # Add standalone CPE elements (keep their original order after R-CPE elements)
    cpe_offset = len(elements) + 1
    for idx, (elem_names, elem_values, elem_errors) in enumerate(elements_woR):
        new_cpe_num = cpe_offset + idx

        new_q_name = f'CPE{new_cpe_num}_0'
        result[new_q_name] = elem_values[0]
        result[f'{new_q_name}_error'] = elem_errors[0]

        new_n_name = f'CPE{new_cpe_num}_1'
        result[new_n_name] = elem_values[1]
        result[f'{new_n_name}_error'] = elem_errors[1]

    result['effective_caps'] = effective_caps

    return result


def suggest_circuit_model(freq, Z, is_spike=True):
    """
    Suggest an equivalent circuit model and initial guesses based on impedance data.

    Parameters
    ----------
    freq : array-like
        Frequency values
    Z : array-like
        Complex impedance values
    is_spike : bool
        Whether the data shows a blocking electrode spike

    Returns
    -------
    tuple
        (circuit_model, initial_guess)
    """
    from scipy.signal import savgol_filter, find_peaks

    def find_min_peaks(Y):
        """Find indices of local minima in phase angle."""
        if len(Y) < 8:
            return []
        try:
            Y_smooth = savgol_filter(Y, window_length=min(8, len(Y)//2*2+1), polyorder=3)
            Y_inv = [-i for i in Y_smooth]
            peaks, _ = find_peaks(np.array(Y_inv), distance=max(10, len(Y)//10))
            return list(peaks)
        except:
            return []

    # Calculate phase angle
    theta = -np.angle(Z, deg=True)

    # Find minima in phase angle
    minima_indices = find_min_peaks(theta)
    minima_indices = sorted(minima_indices)

    # Calculate frequencies, resistances, and capacitances at minima
    freqs = [freq[i] for i in minima_indices if i < len(freq)]
    resists = [np.abs(Z)[i] for i in minima_indices if i < len(Z)]
    caps = []
    for f, r in zip(freqs, resists):
        if f > 0 and r > 0:
            caps.append(1 / (2 * np.pi * f * r))

    # Suggest circuit based on number of semicircles detected
    if len(minima_indices) == 0:
        if is_spike:
            circuit = 'p(R1,CPE1)-CPE2'
            initial_guess = [1e6, 1e-9, 0.9, 1e-6, 0.9]
        else:
            circuit = 'p(R1,CPE1)-p(R2,CPE2)'
            initial_guess = [1e6, 1e-9, 0.9, 1e10, 1e-6, 0.9]
    elif len(minima_indices) == 1:
        if is_spike:
            circuit = 'p(R1,CPE1)-CPE2'
            initial_guess = [resists[0], caps[0] if caps else 1e-9, 0.9, 1e-6, 0.9]
        else:
            circuit = 'p(R1,CPE1)-p(R2,CPE2)'
            initial_guess = [resists[0], caps[0] if caps else 1e-9, 0.9, 1e10, 1e-6, 0.9]
    elif len(minima_indices) >= 2:
        if is_spike:
            circuit = 'p(R1,CPE1)-p(R2,CPE2)-CPE3'
            initial_guess = [
                resists[0], caps[0] if len(caps) > 0 else 1e-9, 0.9,
                resists[1] if len(resists) > 1 else 1e6,
                caps[1] if len(caps) > 1 else 1e-6, 0.9,
                1e-6, 0.9
            ]
        else:
            circuit = 'p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)'
            initial_guess = [
                resists[0], caps[0] if len(caps) > 0 else 1e-9, 0.9,
                resists[1] if len(resists) > 1 else 1e6,
                caps[1] if len(caps) > 1 else 1e-6, 0.9,
                1e10, 1e-6, 0.9
            ]
    else:
        circuit = 'p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)'
        initial_guess = [1e6, 1e-9, 0.9, 1e6, 1e-6, 0.9, 1e10, 1e-6, 0.9]

    return circuit, initial_guess


def auto_fit(freq, Z, circuit_model=None, is_spike=True, constants={}):
    """
    Automatically fit an equivalent circuit model to impedance data.

    Parameters
    ----------
    freq : array-like
        Frequency values
    Z : array-like
        Complex impedance values
    circuit_model : str, optional
        Circuit model to use. If None, will be auto-suggested.
    is_spike : bool
        Whether the data shows a blocking electrode spike
    constants : dict
        Constant parameters in the model

    Returns
    -------
    tuple
        (circuit_params, circuit_conf, Z_fit, rmspe, circuit_model, param_names)
    """
    from impedance.models.circuits import CustomCircuit

    # Suggest circuit if not provided
    if circuit_model is None:
        circuit_model, initial_guess = suggest_circuit_model(freq, Z, is_spike=is_spike)
    else:
        # Generate initial guess for provided model
        n_params = calculateCircuitLength(circuit_model)
        initial_guess = []
        # Parse circuit to generate appropriate initial guesses
        elements = extract_circuit_elements(circuit_model)
        for elem in elements:
            raw_elem = get_element_from_name(elem)
            if raw_elem == 'R':
                initial_guess.append(1e3)
            elif raw_elem == 'CPE':
                initial_guess.append(1e-9)  # Q
                initial_guess.append(0.9)   # n
            elif raw_elem == 'C':
                initial_guess.append(1e-9)
            elif raw_elem == 'W':
                initial_guess.append(1e-3)
            else:
                initial_guess.append(1.0)

    # Multi-pass fitting with different weight methods
    weight_methods = [None, 'modulus', None]

    circuit = CustomCircuit(circuit_model, initial_guess=initial_guess, constants=constants)

    for weight_method in weight_methods:
        try:
            circuit.fit(freq, Z, weight_method=weight_method)
            # Update initial guess for next iteration
            circuit = CustomCircuit(circuit_model, initial_guess=circuit.parameters_, constants=constants)
        except Exception as e:
            continue

    Z_fit = circuit.predict(freq)
    rmspe = calc_rmspe(Z, Z_fit)

    param_names, _ = circuit.get_param_names()

    return circuit.parameters_, circuit.conf_, Z_fit, rmspe, circuit_model, param_names


class BlackBoxOptEIS:
    """
    Black-box optimization for EIS circuit fitting using Optuna.
    """

    def __init__(self, freq, Z, constants={},
                 model_list=None,
                 weight_list=['modulus', None],
                 r_range=(1e0, 1e8),
                 cpe_q_range=(1e-12, 1e-4),
                 r_ranges=None,
                 cpe_q_ranges=None,
                 n_trials=200,
                 timeout=30,
                 early_stop_rmspe=0.03,
                 early_stop_patience=20,
                 log_step=0.5,
                 fit_timeout=5,
                 maxfev=5000):
        """
        Initialize the optimizer.

        Parameters
        ----------
        freq : array-like
            Frequency values
        Z : array-like
            Complex impedance values
        constants : dict
            Constant parameters
        model_list : list
            List of circuit models to try
        weight_list : list
            List of weight methods to try
        r_range : tuple
            Default range for resistance values (min, max)
        cpe_q_range : tuple
            Default range for CPE Q values (min, max)
        r_ranges : dict, optional
            Individual R ranges: {'R1': (min, max), 'R2': (min, max), ...}
        cpe_q_ranges : dict, optional
            Individual CPE Q ranges: {'CPE1': (min, max), 'CPE2': (min, max), ...}
        n_trials : int
            Maximum number of optimization trials
        timeout : int
            Maximum optimization time in seconds
        early_stop_rmspe : float
            Stop optimization if RMSPE falls below this value
        early_stop_patience : int
            Stop optimization if best result is not improved for N consecutive trials
        log_step : float
            Step size in log scale for parameter search (0 = continuous)
        fit_timeout : int
            Timeout in seconds for each individual fitting attempt (not enforced)
        maxfev : int
            Maximum number of function evaluations for curve_fit
        """
        self.freq = np.array(freq)
        self.Z = np.array(Z)
        self.constants = constants

        # Calculate max Z.real for R range upper limit
        self.max_z_real = np.max(np.real(self.Z))

        if model_list is None:
            self.model_list = [
                'p(R1,CPE1)-CPE2',
                'p(R1,CPE1)-p(R2,CPE2)-CPE3'
            ]
        else:
            self.model_list = model_list

        self.weight_list = weight_list
        self.r_range = r_range
        self.cpe_q_range = cpe_q_range

        # Individual ranges (use default if not specified)
        self.r_ranges = r_ranges or {}
        self.cpe_q_ranges = cpe_q_ranges or {}

        self.n_trials = n_trials
        self.timeout = timeout
        self.early_stop_rmspe = early_stop_rmspe
        self.early_stop_patience = early_stop_patience
        self.log_step = log_step
        self.fit_timeout = fit_timeout
        self.maxfev = maxfev

        self.best_params = None
        self.best_rmspe = float('inf')

    def _get_r_range(self, r_name):
        """Get range for a specific R element, capped by max(Z.real)."""
        base_range = self.r_ranges.get(r_name, self.r_range)
        # Cap upper limit at max(Z.real) * 1.5 (with some margin)
        max_r = self.max_z_real * 1.5
        capped_max = min(base_range[1], max_r)
        # Ensure min < max
        if capped_max <= base_range[0]:
            capped_max = base_range[1]  # Fall back to original if capping would be invalid
        return (base_range[0], capped_max)

    def _get_cpe_q_range(self, cpe_name):
        """Get Q range for a specific CPE element."""
        return self.cpe_q_ranges.get(cpe_name, self.cpe_q_range)

    def _suggest_log_float(self, trial, name, low, high):
        """
        Suggest a float value in log scale with optional step.

        If log_step > 0, the search is discretized in log space.
        """
        if self.log_step > 0:
            # Convert to log space, apply step, convert back
            log_low = np.log10(low)
            log_high = np.log10(high)
            log_val = trial.suggest_float(name, log_low, log_high, step=self.log_step)
            return 10 ** log_val
        else:
            # Continuous log-scale search
            return trial.suggest_float(name, low, high, log=True)

    def _fit_circuit(self, model, initial_guess, weight_method):
        """Fit a single circuit configuration."""
        from impedance.models.circuits import CustomCircuit

        try:
            # Use our custom circuit_fit function which supports weight_method and timeout
            popt, perror = circuit_fit(
                self.freq, self.Z,
                model,
                initial_guess,
                constants=self.constants,
                weight_method=weight_method,
                timeout=self.fit_timeout,
                maxfev=self.maxfev
            )

            # Create circuit with fitted parameters
            circuit = CustomCircuit(model, initial_guess=list(popt), constants=self.constants)
            circuit.parameters_ = popt
            circuit.conf_ = perror

            Z_fit = circuit.predict(self.freq)
            rmspe = calc_rmspe(self.Z, Z_fit)
            return circuit, Z_fit, rmspe
        except FittingTimeoutError:
            # Timeout occurred - return inf RMSPE
            return None, None, float('inf')
        except Exception:
            # Other errors - return inf RMSPE
            return None, None, float('inf')

    def _objective(self, trial):
        """Optuna objective function."""
        # Select model
        model = trial.suggest_categorical('model', self.model_list)

        # Select weight method
        weight = trial.suggest_categorical('weight', self.weight_list)

        # Generate initial guesses based on model using individual ranges
        if model == 'p(R1,CPE1)-CPE2':
            r1_range = self._get_r_range('R1')
            cpe1_range = self._get_cpe_q_range('CPE1')
            cpe2_range = self._get_cpe_q_range('CPE2')
            r1 = self._suggest_log_float(trial, 'r1', r1_range[0], r1_range[1])
            cpe1 = self._suggest_log_float(trial, 'cpe1', cpe1_range[0], cpe1_range[1])
            cpe2 = self._suggest_log_float(trial, 'cpe2', cpe2_range[0], cpe2_range[1])
            initial_guess = [r1, cpe1, 0.9, cpe2, 0.9]
        elif model == 'p(R1,CPE1)-p(R2,CPE2)-CPE3':
            r1_range = self._get_r_range('R1')
            r2_range = self._get_r_range('R2')
            cpe1_range = self._get_cpe_q_range('CPE1')
            cpe2_range = self._get_cpe_q_range('CPE2')
            cpe3_range = self._get_cpe_q_range('CPE3')
            r1 = self._suggest_log_float(trial, 'r1', r1_range[0], r1_range[1])
            cpe1 = self._suggest_log_float(trial, 'cpe1', cpe1_range[0], cpe1_range[1])
            r2 = self._suggest_log_float(trial, 'r2', r2_range[0], r2_range[1])
            cpe2 = self._suggest_log_float(trial, 'cpe2', cpe2_range[0], cpe2_range[1])
            cpe3 = self._suggest_log_float(trial, 'cpe3', cpe3_range[0], cpe3_range[1])
            initial_guess = [r1, cpe1, 0.9, r2, cpe2, 0.9, cpe3, 0.9]
        elif model == 'R1-p(R2,CPE1)-CPE2':
            r1_range = self._get_r_range('R1')
            r2_range = self._get_r_range('R2')
            cpe1_range = self._get_cpe_q_range('CPE1')
            cpe2_range = self._get_cpe_q_range('CPE2')
            r1 = self._suggest_log_float(trial, 'r1', r1_range[0], r1_range[1])
            r2 = self._suggest_log_float(trial, 'r2', r2_range[0], r2_range[1])
            cpe1 = self._suggest_log_float(trial, 'cpe1', cpe1_range[0], cpe1_range[1])
            cpe2 = self._suggest_log_float(trial, 'cpe2', cpe2_range[0], cpe2_range[1])
            initial_guess = [r1, r2, cpe1, 0.9, cpe2, 0.9]
        elif model == 'R1-p(R2,CPE1)-p(R3,CPE2)-CPE3':
            r1_range = self._get_r_range('R1')
            r2_range = self._get_r_range('R2')
            r3_range = self._get_r_range('R3')
            cpe1_range = self._get_cpe_q_range('CPE1')
            cpe2_range = self._get_cpe_q_range('CPE2')
            cpe3_range = self._get_cpe_q_range('CPE3')
            r1 = self._suggest_log_float(trial, 'r1', r1_range[0], r1_range[1])
            r2 = self._suggest_log_float(trial, 'r2', r2_range[0], r2_range[1])
            r3 = self._suggest_log_float(trial, 'r3', r3_range[0], r3_range[1])
            cpe1 = self._suggest_log_float(trial, 'cpe1', cpe1_range[0], cpe1_range[1])
            cpe2 = self._suggest_log_float(trial, 'cpe2', cpe2_range[0], cpe2_range[1])
            cpe3 = self._suggest_log_float(trial, 'cpe3', cpe3_range[0], cpe3_range[1])
            initial_guess = [r1, r2, cpe1, 0.9, r3, cpe2, 0.9, cpe3, 0.9]
        elif model == 'p(R1,CPE1)':
            r1_range = self._get_r_range('R1')
            cpe1_range = self._get_cpe_q_range('CPE1')
            r1 = self._suggest_log_float(trial, 'r1', r1_range[0], r1_range[1])
            cpe1 = self._suggest_log_float(trial, 'cpe1', cpe1_range[0], cpe1_range[1])
            initial_guess = [r1, cpe1, 0.9]
        elif model == 'p(R1,CPE1)-p(R2,CPE2)':
            r1_range = self._get_r_range('R1')
            r2_range = self._get_r_range('R2')
            cpe1_range = self._get_cpe_q_range('CPE1')
            cpe2_range = self._get_cpe_q_range('CPE2')
            r1 = self._suggest_log_float(trial, 'r1', r1_range[0], r1_range[1])
            cpe1 = self._suggest_log_float(trial, 'cpe1', cpe1_range[0], cpe1_range[1])
            r2 = self._suggest_log_float(trial, 'r2', r2_range[0], r2_range[1])
            cpe2 = self._suggest_log_float(trial, 'cpe2', cpe2_range[0], cpe2_range[1])
            initial_guess = [r1, cpe1, 0.9, r2, cpe2, 0.9]
        else:
            # Generic handling for other models using parameter names
            try:
                from impedance.models.circuits import CustomCircuit
                n_params = calculateCircuitLength(model)
                temp_circuit = CustomCircuit(model, initial_guess=[1.0] * n_params)
                param_names, _ = temp_circuit.get_param_names()

                initial_guess = []
                for pname in param_names:
                    if 'CPE' in pname and '_1' in pname:  # CPE alpha
                        initial_guess.append(0.9)
                    elif 'CPE' in pname and '_0' in pname:  # CPE Q
                        # Extract CPE name (e.g., CPE1, CPE2)
                        cpe_name = pname.split('_')[0]
                        cpe_range = self._get_cpe_q_range(cpe_name)
                        initial_guess.append(self._suggest_log_float(trial, pname, cpe_range[0], cpe_range[1]))
                    elif 'W' in pname:  # Warburg
                        initial_guess.append(self._suggest_log_float(trial, pname, 1e-6, 1e-1))
                    elif 'L' in pname:  # Inductor
                        initial_guess.append(self._suggest_log_float(trial, pname, 1e-9, 1e-3))
                    elif 'C' in pname and 'CPE' not in pname:  # Capacitor
                        initial_guess.append(self._suggest_log_float(trial, pname, 1e-12, 1e-6))
                    elif 'R' in pname:  # Resistor
                        # Extract R name (e.g., R1, R2)
                        r_name = pname.split('_')[0] if '_' in pname else pname
                        r_range = self._get_r_range(r_name)
                        initial_guess.append(self._suggest_log_float(trial, pname, r_range[0], r_range[1]))
                    else:
                        initial_guess.append(self._suggest_log_float(trial, pname, 1e-6, 1e6))
            except Exception:
                # Fallback
                n_params = calculateCircuitLength(model)
                initial_guess = []
                for i in range(n_params):
                    initial_guess.append(self._suggest_log_float(trial, f'p{i}', 1e-6, 1e6))

        circuit, Z_fit, rmspe = self._fit_circuit(model, initial_guess, weight)
        return rmspe

    def optimize(self, progress_callback=None):
        """
        Run the optimization.

        Parameters
        ----------
        progress_callback : callable, optional
            Function to call with progress updates

        Returns
        -------
        dict
            Best parameters found
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            sampler = optuna.samplers.TPESampler(n_startup_trials=min(50, self.n_trials // 4))
            study = optuna.create_study(direction='minimize', sampler=sampler)

            # Track trials without improvement for early stopping
            trials_without_improvement = [0]
            last_best_value = [float('inf')]

            def callback(study, trial):
                if progress_callback:
                    best_val = study.best_value if study.best_value is not None else 1.0
                    progress_callback(trial.number + 1, self.n_trials, best_val)

                # Check RMSPE early stop
                if study.best_value is not None and study.best_value <= self.early_stop_rmspe:
                    study.stop()
                    return

                # Check patience early stop (no improvement for N trials)
                if study.best_value is not None:
                    if study.best_value < last_best_value[0]:
                        # Improvement found
                        last_best_value[0] = study.best_value
                        trials_without_improvement[0] = 0
                    else:
                        # No improvement
                        trials_without_improvement[0] += 1

                    if trials_without_improvement[0] >= self.early_stop_patience:
                        study.stop()

            study.optimize(
                self._objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                callbacks=[callback],
                show_progress_bar=False
            )

            self.best_params = study.best_params
            self.best_rmspe = study.best_value

            return study.best_params

        except ImportError:
            # Fallback if optuna not installed - simple grid search
            best_rmspe = float('inf')
            best_params = None

            for model in self.model_list:
                for weight in self.weight_list:
                    # Try a few random initial guesses
                    for _ in range(20):
                        if model == 'p(R1,CPE1)-CPE2':
                            r1 = np.random.uniform(np.log10(self.r_range[0]), np.log10(self.r_range[1]))
                            cpe1 = np.random.uniform(np.log10(self.cpe_q_range[0]), np.log10(self.cpe_q_range[1]))
                            cpe2 = np.random.uniform(np.log10(self.cpe_q_range[0]), np.log10(self.cpe_q_range[1]))
                            initial_guess = [10**r1, 10**cpe1, 0.9, 10**cpe2, 0.9]
                            params = {'model': model, 'weight': weight, 'r1': r1, 'cpe1': cpe1, 'cpe2': cpe2}
                        elif model == 'p(R1,CPE1)-p(R2,CPE2)-CPE3':
                            r1 = np.random.uniform(np.log10(self.r_range[0]), np.log10(self.r_range[1]))
                            r2 = np.random.uniform(np.log10(self.r_range[0]), np.log10(self.r_range[1]))
                            cpe1 = np.random.uniform(np.log10(self.cpe_q_range[0]), np.log10(self.cpe_q_range[1]))
                            cpe2 = np.random.uniform(np.log10(self.cpe_q_range[0]), np.log10(self.cpe_q_range[1]))
                            cpe3 = np.random.uniform(np.log10(self.cpe_q_range[0]), np.log10(self.cpe_q_range[1]))
                            initial_guess = [10**r1, 10**cpe1, 0.9, 10**r2, 10**cpe2, 0.9, 10**cpe3, 0.9]
                            params = {'model': model, 'weight': weight, 'r1': r1, 'r2': r2, 'cpe1': cpe1, 'cpe2': cpe2, 'cpe3': cpe3}
                        else:
                            # Generic model - skip for fallback
                            continue

                        try:
                            circuit, Z_fit, rmspe = self._fit_circuit(model, initial_guess, weight)
                            if rmspe < best_rmspe:
                                best_rmspe = rmspe
                                best_params = params
                        except Exception:
                            continue

            self.best_params = best_params
            self.best_rmspe = best_rmspe
            return best_params

    def fit_best(self):
        """
        Fit the circuit using the best parameters found.

        Returns
        -------
        tuple
            (circuit_params, circuit_conf, Z_fit, rmspe, model, param_names)
        """
        if self.best_params is None:
            self.optimize()

        if self.best_params is None:
            return None, None, None, float('inf'), self.model_list[0], []

        model = self.best_params.get('model', self.model_list[0])
        weight = self.best_params.get('weight', None)

        # Build initial guess from best params
        # Note: values in best_params are log10 values, need to convert with 10**x
        if model == 'p(R1,CPE1)-CPE2':
            initial_guess = [
                10 ** self.best_params.get('r1', 3),      # R1
                10 ** self.best_params.get('cpe1', -9),   # CPE1_Q
                0.9,                                       # CPE1_alpha
                10 ** self.best_params.get('cpe2', -6),   # CPE2_Q
                0.9                                        # CPE2_alpha
            ]
        elif model == 'p(R1,CPE1)-p(R2,CPE2)-CPE3':
            initial_guess = [
                10 ** self.best_params.get('r1', 3),      # R1
                10 ** self.best_params.get('cpe1', -9),   # CPE1_Q
                0.9,                                       # CPE1_alpha
                10 ** self.best_params.get('r2', 4),      # R2
                10 ** self.best_params.get('cpe2', -8),   # CPE2_Q
                0.9,                                       # CPE2_alpha
                10 ** self.best_params.get('cpe3', -6),   # CPE3_Q
                0.9                                        # CPE3_alpha
            ]
        else:
            # Generic model - build initial guess from best_params using parameter names
            # Note: In _objective, suggest_log_float returns actual values (not log10)
            # So best_params values are actual parameter values, not log10
            try:
                from impedance.models.circuits import CustomCircuit
                n_params = calculateCircuitLength(model)
                temp_circuit = CustomCircuit(model, initial_guess=[1.0] * n_params)
                param_names, _ = temp_circuit.get_param_names()

                initial_guess = []
                for pname in param_names:
                    # Check if parameter is in best_params
                    # Values from suggest_log_float are actual values (not log10)
                    if pname in self.best_params:
                        initial_guess.append(self.best_params[pname])
                    elif 'CPE' in pname and '_1' in pname:  # CPE alpha
                        initial_guess.append(0.9)
                    elif 'CPE' in pname and '_0' in pname:  # CPE Q
                        initial_guess.append(1e-9)
                    elif 'W' in pname:  # Warburg
                        initial_guess.append(1e-3)
                    elif 'L' in pname:  # Inductor
                        initial_guess.append(1e-6)
                    elif 'C' in pname and 'CPE' not in pname:  # Capacitor
                        initial_guess.append(1e-9)
                    elif 'R' in pname:  # Resistor
                        initial_guess.append(1e4)
                    else:
                        initial_guess.append(1.0)
            except Exception:
                # Fallback
                n_params = calculateCircuitLength(model)
                initial_guess = [1e4 if i % 3 == 0 else (1e-9 if i % 3 == 1 else 0.9) for i in range(n_params)]

        circuit, Z_fit, rmspe = self._fit_circuit(model, initial_guess, weight)

        if circuit is not None:
            param_names, _ = circuit.get_param_names()
            return circuit.parameters_, circuit.conf_, Z_fit, rmspe, model, param_names
        else:
            return None, None, None, float('inf'), model, []
