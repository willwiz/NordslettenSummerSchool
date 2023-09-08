from typing import Tuple, Union
import numpy as np
from numpy import float64 as f64, int32 as i32
from numpy.typing import NDArray as Arr
from scipy.special import gamma
from mittag_leffler.mittag_leffler import ml


def _polynomial_function(pars: Arr[f64], time: Arr[f64]) -> Arr[f64]:
    t = time
    res = pars[0] * t
    for p in pars[1:]:
        t = t * time
        res = res + p * t
    return res


def _polynomial_function_with_exponents(
    pars: Arr[f64], exponents: Arr[i32], time: Arr[f64]
) -> Arr[f64]:
    res = np.zeros_like(time)
    for p, e in zip(pars, exponents):
        res = res + p * time**e
    return res


def polynomial_function(
    pars: Union[Arr[f64], Tuple[Arr[f64], Arr[i32]]], time: Arr[f64]
) -> Arr[f64]:
    if isinstance(pars, tuple):
        pars, exponents = pars
        return _polynomial_function_with_exponents(pars, exponents, time)
    else:
        return _polynomial_function(pars, time)


def _analytical_polynomial_fractionalderiv_function(
    alpha: float, pars: Arr[f64], time: Arr[f64]
) -> Arr[f64]:
    t = time ** (1.0 - alpha)
    res = pars[0] * t / gamma(2.0 - alpha)
    for k, p in enumerate(pars[1:], start=2):
        t = t * time
        res = res + p * t * gamma(k + 1.0) / gamma(k + 1.0 - alpha)
    return res


def _analytical_polynomial_fractionalderiv_function_with_exponent(
    alpha: float, pars: Arr[f64], exponent: Arr[i32], time: Arr[f64]
) -> Arr[f64]:
    res = np.zeros_like(time)
    for p, e in zip(pars, exponent):
        res = res + p * time ** (e - alpha) * gamma(e + 1.0) / gamma(e + 1.0 - alpha)
    return res


def analytical_polynomial_fractionalderiv_function(
    alpha: float, pars: Union[Arr[f64], Tuple[Arr[f64], Arr[i32]]], time: Arr[f64]
) -> Arr[f64]:
    if isinstance(pars, tuple):
        pars, exponents = pars
        return _analytical_polynomial_fractionalderiv_function_with_exponent(
            alpha, pars, exponents, time
        )
    else:
        return _analytical_polynomial_fractionalderiv_function(alpha, pars, time)


def _analytical_polynomial_fractionaldiffeq_function(
    alpha: float, delta: float, pars: Arr[f64], time: Arr[f64]
) -> Arr[f64]:
    tk = np.ones_like(time)
    res = np.zeros_like(time)
    t_alpha_delta = -(time**alpha) / delta
    for i, p in enumerate(pars, start=1):
        tk = tk * time
        res = res + p * tk * gamma(1 + i) / delta * ml(t_alpha_delta, alpha, i + 1.0)
    return res


def _analytical_polynomial_fractionaldiffeq_function_with_exponent(
    alpha: float, delta: float, pars: Arr[f64], exponent: Arr[i32], time: Arr[f64]
) -> Arr[f64]:
    tk = np.ones_like(time)
    res = np.zeros_like(time)
    t_alpha_delta = -(time**alpha) / delta
    for p, e in zip(pars, exponent):
        tk = tk * time
        res = res + p * tk * gamma(1 + e) / delta * ml(t_alpha_delta, alpha, e + 1.0)
    return res


def analytical_polynomial_fractionaldiffeq_function(
    alpha: float,
    delta: float,
    pars: Union[Arr[f64], Tuple[Arr[f64]], Arr[i32]],
    time: Arr[f64],
) -> Arr[f64]:
    if isinstance(pars, tuple):
        pars, exponents = pars
        return _analytical_polynomial_fractionaldiffeq_function_with_exponent(
            alpha, delta, pars, exponents, time
        )
    else:
        return _analytical_polynomial_fractionaldiffeq_function(
            alpha, delta, pars, time
        )
