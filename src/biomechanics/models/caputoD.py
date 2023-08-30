from dataclasses import dataclass
import os
import numpy as np
from scipy.io import loadmat
from numpy import float64 as f64, interp, exp, pi, ndarray, zeros
from numpy.typing import NDArray as Arr

pack_dir = os.path.dirname(os.path.abspath(__file__))


@dataclass(slots=True, init=False)
class CaputoInitialize:
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


def caputo_derivative_linear(carp: CaputoInitialize, S: Arr[f64], dt: Arr[64]):
    nt, *dim = S.shape
    ek = carp.taus / (dt[:, np.newaxis] + carp.taus)
    df = np.diff(S, prepend=[S[0, :]], axis=0)
    beta_part = np.einsum("k,mi...->mki...", carp.betas, df)
    Qk = np.zeros((nt, 9, *dim), dtype=f64)
    for i in range(1, nt):
        Qk[i] = np.einsum("k,ki...->ki...", ek[i], Qk[i - 1] + beta_part[i])
    return np.einsum("m,mi...->mi...", (carp.beta0 / dt), df) + np.einsum(
        "mki...->mi...", Qk
    )


def caputo_derivative_quadratic(carp: CaputoInitialize, S: Arr[f64], dt: Arr[64]):
    nt, *dim = S.shape
    ek = exp(-0.5 * dt[:, np.newaxis] / carp.taus)
    e2 = ek * ek
    df = np.diff(S, prepend=[S[0, :]], axis=0)
    beta_part = np.einsum("k,mk,mi...->mki...", carp.betas, ek, df)
    Qk = np.zeros((nt, 9, *dim), dtype=f64)
    for i in range(1, nt):
        Qk[i] = np.einsum("k,ki...->ki...", e2[i], Qk[i - 1]) + beta_part[i]
    return np.einsum("m,mi...->mi...", (carp.beta0 / dt), df) + np.einsum(
        "mki...->mi...", Qk
    )


def caputo_derivative1_iter(
    fn: ndarray, dt: float, carp: CaputoInitialize
) -> tuple[ndarray, CaputoInitialize]:
    df = fn - carp.f_prev
    ek = carp.tau / (carp.tau + dt)
    for k in range(carp.N):
        carp.Q[k, :] = ek[k] * (carp.Q[k, :] + carp.beta[k] * df)
    v = (carp.beta0 / dt) * df + np.einsum("ki->i", carp.Q)
    # v = np.einsum("ki->i", carp.Q)
    # carp.f_prev = fn
    return v, carp


def caputo_derivative2_iter(
    fn: ndarray, dt: float, carp: CaputoInitialize
) -> tuple[ndarray, CaputoInitialize]:
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
) -> tuple[ndarray, CaputoInitialize]:
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


def caputo_diffeq_linear(carp: CaputoInitialize, S: Arr[f64], dt: Arr[64]):
    nt, *dim = S.shape
    ek = carp.taus / (dt[:, np.newaxis] + carp.taus)
    df = np.diff(S, prepend=[S[0, :]], axis=0)
    beta_part = np.einsum("k,mi...->mki...", carp.betas, df)
    Qk = np.zeros((nt, 9, *dim), dtype=f64)
    for i in range(1, nt):
        Qk[i] = np.einsum("k,ki...->ki...", ek[i], Qk[i - 1] + beta_part[i])
    return np.einsum("m,mi...->mi...", (carp.beta0 / dt), df) + np.einsum(
        "mki...->mi...", Qk
    )


def diffeq_approx2_iter(
    fn: ndarray, dt: float, carp: CaputoInitialize
) -> tuple[ndarray, CaputoInitialize]:
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
