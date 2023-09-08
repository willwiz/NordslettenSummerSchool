from dataclasses import dataclass
import os
import numpy as np
from scipy.io import loadmat
from numpy import float64 as f64, interp, exp, pi, ndarray, zeros
from numpy.typing import NDArray as Arr

pack_dir = os.path.dirname(os.path.abspath(__file__))


@dataclass(init=False)
class CaputoInitialize:
    __slots__ = ["Np", "beta0", "betas", "taus"]
    Np: int
    beta0: float
    betas: Arr[f64]
    taus: Arr[f64]

    def __init__(
        self,
        alpha: float,
        Tf: float,
        Np: int = 9,
    ) -> None:
        freq = 2 * pi / (Tf)
        par_file = f"coeffs-opt-refined-100steps-{Np}-500.mat"
        carp = loadmat(os.path.join(pack_dir, "caputo_parm", par_file))
        betam = carp["betam"]
        taum = carp["taum"]
        x = np.linspace(0.01, 1.0, num=100, endpoint=True)
        self.Np = Np
        self.beta0 = interp(alpha, x, betam[-1]) * (freq ** (alpha - 1.0))
        self.betas = np.array(
            [interp(alpha, x, i) * (freq**alpha) for i in betam[:-1]]
        )
        self.taus = np.array([interp(alpha, x, i) / freq for i in taum])


def _caputo_derivative_body(
    Np: int, K0: Arr[f64], bek: Arr[f64], e2: Arr[f64], S: Arr[f64]
) -> Arr[f64]:
    nt, *dim = S.shape
    df = np.diff(S, prepend=[S[0]], axis=0)
    beta_part = np.einsum("mk,m...->mk...", bek, df)
    Qk = np.zeros((nt, Np, *dim), dtype=f64)
    for i in range(1, nt):
        Qk[i] = np.einsum("k,k...->k...", e2[i], Qk[i - 1]) + beta_part[i]
    return np.einsum("m,m...->m...", K0, df) + np.einsum("mk...->m...", Qk)


def caputo_derivative_linear(carp: CaputoInitialize, S: Arr[f64], dt: Arr[64]):
    K0 = carp.beta0 / dt
    ek = carp.taus / (dt[:, np.newaxis] + carp.taus)
    bek = np.einsum("k,mk->mk", carp.betas, ek)
    return _caputo_derivative_body(carp.Np, K0, bek, ek, S)


def caputo_derivative_quadratic(carp: CaputoInitialize, S: Arr[f64], dt: Arr[64]):
    K0 = carp.beta0 / dt
    ek = exp(-0.5 * dt[:, np.newaxis] / carp.taus)
    bek = np.einsum("k,mk->mk", carp.betas, ek)
    return _caputo_derivative_body(carp.Np, K0, bek, ek * ek, S)


def _check_array_shape(Arr1: Arr, Arr2: Arr) -> None:
    nt1, *dim1 = Arr1.shape
    nt2, *dim2 = Arr2.shape
    if (nt1, *dim1) != (nt2, *dim2):
        raise ValueError


def _caputo_diffeq_body(
    Np: int, delta: float, K1: Arr[f64], bek: Arr[f64], e2: Arr[f64], S: Arr[f64]
) -> Arr[f64]:
    nt, *dim = S.shape
    Qk = np.zeros((Np, *dim), dtype=f64)
    LHS = np.zeros_like(S)
    for i in range(1, nt):
        v = S[i] - delta * np.einsum("k,k...->...", e2[i], Qk)
        LHS[i] = (v + (K1[i] * LHS[i - 1])) / (1.0 + K1[i])
        # Updates
        beta_part = np.einsum("k,...->k...", bek[i], LHS[i] - LHS[i - 1])
        Qk = np.einsum("k,k...->k...", e2[i], Qk) + beta_part
    return LHS


def caputo_diffeq_linear(
    delta: float, carp: CaputoInitialize, S_HE: Arr[f64], S_VE: Arr[f64], dt: Arr[64]
):
    # Precomputes
    _check_array_shape(S_HE, S_VE)
    K0 = carp.beta0 / dt
    ek = carp.taus / (dt[:, np.newaxis] + carp.taus)
    bek = np.einsum("k,mk->mk", carp.betas, ek)
    K1 = delta * (K0 + np.einsum("mk->m", bek))
    # Solve
    RHS = S_HE + _caputo_derivative_body(carp.Np, K0, bek, ek, S_VE)
    return _caputo_diffeq_body(carp.Np, delta, K1, bek, ek, RHS)


def caputo_diffeq_quadratic(
    delta: float, carp: CaputoInitialize, S_HE: Arr[f64], S_VE: Arr[f64], dt: Arr[64]
):
    # Precomputes
    _check_array_shape(S_HE, S_VE)
    K0 = carp.beta0 / dt
    ek = exp(-0.5 * dt[:, np.newaxis] / carp.taus)
    bek = np.einsum("k,mk->mk", carp.betas, ek)
    e2 = ek * ek
    K1 = delta * (K0 + np.einsum("mk->m", bek))
    # Solve
    RHS = S_HE + _caputo_derivative_body(carp.Np, K0, bek, e2, S_VE)
    return _caputo_diffeq_body(carp.Np, delta, K1, bek, e2, RHS)
