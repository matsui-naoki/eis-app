"""
Custom fitting functions for EIS circuit analysis
Supports multiple weight methods: modulus, squared_modulus, proportional
"""

import warnings
import numpy as np
from scipy.linalg import inv
from scipy.optimize import curve_fit, basinhopping

# Use elements from impedance library
from impedance.models.circuits.elements import circuit_elements, get_element_from_name
from impedance.models.circuits.fitting import (
    buildCircuit, extract_circuit_elements, calculateCircuitLength
)


ints = '0123456789'


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


def circuit_fit(frequencies, impedances, circuit, initial_guess, constants={},
                bounds=None, weight_method=None, global_opt=False,
                **kwargs):
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

    Returns
    -------
    p_values : list of floats
        Best fit parameters
    p_errors : list of floats
        One standard deviation error estimates
    """
    f = np.array(frequencies, dtype=float)
    Z = np.array(impedances, dtype=complex)

    if bounds is None:
        bounds = set_default_bounds(circuit, constants=constants)

    if not global_opt:
        if 'maxfev' not in kwargs:
            kwargs['maxfev'] = int(1e5)
        if 'ftol' not in kwargs:
            kwargs['ftol'] = 1e-13

        # Calculate weights based on the chosen method
        if weight_method == 'squared_modulus':
            # Squared modulus weighting: weights are 1/|Z|^2
            sigma = np.abs(Z) ** 2
            kwargs['sigma'] = np.hstack([sigma, sigma])
        elif weight_method == 'modulus':
            # Modulus weighting: weights are 1/|Z|
            sigma = np.abs(Z)
            kwargs['sigma'] = np.hstack([sigma, sigma])
        elif weight_method == 'proportional':
            # Proportional weighting: weights are |Z|
            weights = np.abs(Z)
            sigma = 1 / weights
            kwargs['sigma'] = np.hstack([sigma, sigma])
        # else: no weighting (weight_method is None or unknown)

        popt, pcov = curve_fit(wrapCircuit(circuit, constants), f,
                               np.hstack([Z.real, Z.imag]),
                               p0=initial_guess, bounds=bounds, **kwargs)

        perror = np.sqrt(np.diag(pcov))

    else:
        if 'seed' not in kwargs:
            kwargs['seed'] = 0

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
                               accept_test=basinhopping_bounds, **kwargs)
        popt = results.x

        jac = results.lowest_optimization_result['jac'][np.newaxis]
        try:
            pcov = inv(np.dot(jac.T, jac)) * opt_function(popt) ** 2
            perror = np.sqrt(np.diag(pcov))
        except (ValueError, np.linalg.LinAlgError):
            warnings.warn('Failed to compute perror')
            perror = None

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
