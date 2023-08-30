import os
import numpy as np
from typing import Tuple, Optional
from scipy.io import loadmat
from numpy import float64 as f64, interp, exp, pi, ndarray, zeros
from numpy.typing import NDArray as Arr

pack_dir = os.path.dirname(os.path.abspath(__file__))


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


def caputo_derivative_linear(
    carp: CaputoInitialize, df: ndarray, dt: float, OldK: Arr[f64], betas: Arr[f64]
):
    ek = carp.tau / (carp.tau + dt)
    Qk = np.einsum("k,kij->kij", ek, OldK + np.einsum("k,ij->kij", betas, df))
    v = (carp.beta0 / dt) * df + np.einsum("ki->i", carp.Q)
    return v, Qk


def caputo_derivative1_iter(
    fn: ndarray, dt: float, carp: CaputoInitialize
) -> Tuple[ndarray, CaputoInitialize]:
    df = fn - carp.f_prev
    ek = carp.tau / (carp.tau + dt)
    for k in range(carp.N):
        carp.Q[k, :] = ek[k] * (carp.Q[k, :] + carp.beta[k] * df)
    # v = (carp.beta0 / dt) * df + einsum('ki->i', carp.Q)
    v = np.einsum("ki->i", carp.Q)
    carp.f_prev = fn
    return v, carp


def caputo_derivative2_iter(
    fn: ndarray, dt: float, carp: CaputoInitialize
) -> Tuple[ndarray, CaputoInitialize]:
    df = fn - carp.f_prev
    ek = exp(-0.5 * dt / carp.tau)
    e2 = ek * ek
    for k in range(carp.N):
        carp.Q[k, :] = e2[k] * carp.Q[k, :] + carp.beta[k] * ek[k] * df
    v = (carp.beta0 / dt) * df + np.einsum("ki->i", carp.Q)
    # v = (carp.beta0 / dt) * df
    carp.f_prev = fn
    return v, carp


def diffeq_approx1_iter(
    fn: ndarray, dt: float, carp: CaputoInitialize
) -> Tuple[ndarray, CaputoInitialize]:
    K0 = carp.beta0 / dt
    ek = carp.tau / (carp.tau + dt)
    K0 = carp.delta * (K0 + np.einsum("k,k->", carp.beta, ek))
    v = fn - carp.delta * np.einsum("k,ki->k", ek, carp.Q)
    v = (v + K0 * carp.f_prev) / (1.0 + K0)
    # Updates
    df = v - carp.f_prev
    carp.f_prev = v
    for k in range(carp.N):
        carp.Q[k, :] = ek[k] * (carp.Q[k, :] + carp.beta[k] * df)
    return v, carp


def diffeq_approx2_iter(
    fn: ndarray, dt: float, carp: CaputoInitialize
) -> Tuple[ndarray, CaputoInitialize]:
    K0 = carp.beta0 / dt
    ek = exp(-0.5 * dt / carp.tau)
    e2 = ek * ek
    K0 = carp.delta * (K0 + np.einsum("k,k->", carp.beta, ek))
    v = fn - carp.delta * np.einsum("k,ki->k", e2, carp.Q)
    v = (v + K0 * carp.f_prev) / (1.0 + K0)
    # Updates
    df = v - carp.f_prev
    carp.f_prev = v
    for k in range(carp.N):
        carp.Q[k, :] = e2[k] * carp.Q[k, :] + carp.beta[k] * ek[k] * df
    return v, carp
