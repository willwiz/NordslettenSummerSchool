from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray as Arr
from numpy import float64 as f64
from numpy.linalg import norm
from biomechanics._interfaces import HyperelasticModel
from biomechanics.kinematics.biaxial import (
    compute_right_cauchy_green,
    compute_green_lagrange_strain,
)


_IDENTITY_MATRIX: Arr[f64] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


def get_change_of_basis_tensor(v_f: Arr[f64], v_s: Arr[f64]):
    v_n = np.cross(v_f, v_s)
    v_n_norm = norm(v_n)
    if v_n_norm > 0.0:
        v_n = v_n / v_n_norm
    else:
        raise ValueError("The cross product of v_f and v_s is zero.")
    return np.array([v_f, v_s, v_n], dtype=float)


@dataclass(slots=True)
class NeoHookeanModel(HyperelasticModel):
    """2D Neo Hookean Model
    Attributes:
        mu (float): Bulk Modulus.
    """

    mu: float  # Bulk Modulus

    def pk2(self, F: Arr[f64]) -> Arr[f64]:
        C = compute_right_cauchy_green(F)
        res = np.zeros_like(C)
        res[:, [0, 1, 2], [0, 1, 2]] = self.mu
        return res


class GuccioneModel(HyperelasticModel):
    __slots__ = ["mu", "b1", "b2", "H", "fiber"]
    mu: float  # Bulk Modulus
    b1: float  # Isotropic Exponent
    b2: Arr[f64]  # Array of fiber Exponents
    H: Arr[f64]  # Structural tensor for material orientation

    def __init__(
        self,
        mu: float,
        b_1: float,
        b_ff: float,
        b_fs: float,
        b_sn: float,
        v_f: Arr[f64],
        v_s: Arr[f64],
    ) -> None:
        self.mu = mu / 2.0
        self.b1 = 2.0 * b_1
        self.b2 = np.array([[b_ff, b_fs, b_fs], [b_fs, b_sn, b_sn], [b_fs, b_sn, b_sn]])
        self.H = 2.0 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        self.fiber = get_change_of_basis_tensor(v_f, v_s)

    def pk2(self, F: Arr[f64]) -> Arr[f64]:
        E = compute_green_lagrange_strain(F)
        E_material = np.einsum("ij,mjk,lk->mil", self.fiber, E, self.fiber)
        E_material2 = E_material * E_material
        b_iso = self.b1 * np.einsum("mii->m", E_material)
        b_fiber = np.einsum("ij,mij->m", self.b2, E_material2)
        mxm = np.einsum("ij,mij->mij", self.b2, E_material) + self.H
        pk2_material = self.mu * np.einsum("m,mij->mij", np.exp(b_iso + b_fiber), mxm)
        return np.einsum("ji,mjk,kl->mil", self.fiber, pk2_material, self.fiber)


class CostaModel(HyperelasticModel):
    __slots__ = ["mu", "b", "fiber"]
    mu: float  # Bulk Modulus
    b: Arr[f64]  # Array of fiber Exponents

    def __init__(
        self,
        mu: float,
        b_ff: float,
        b_ss: float,
        b_nn: float,
        b_fs: float,
        b_fn: float,
        b_sn: float,
    ) -> None:
        self.mu = mu / 2.0
        self.b = np.array(
            [[b_ff, b_fs, b_fn], [b_fs, b_ss, b_sn], [b_fn, b_sn, b_nn]], dtype=float
        )
        self.fiber = get_change_of_basis_tensor(v_f, v_s)

    def pk2(self, F: Arr[f64]) -> Arr[f64]:
        E = compute_green_lagrange_strain(F)
        E_material = np.einsum("ij,mjk,lk->mil", self.fiber, E, self.fiber)
        E_material2 = E_material * E_material
        b_fiber = np.einsum("ij,mij->m", self.b, E_material2)
        mxm = np.einsum("ij,mij->mij", self.b, E_material)
        pk2_material = self.mu * np.einsum("m,mij->mij", np.exp(b_fiber), mxm)
        return np.einsum("ji,mjk,kl->mil", self.fiber, pk2_material, self.fiber)


class HolzapfelOgdenModel(HyperelasticModel):
    __slots__ = ["k_iso", "b_iso", "k_fiber", "b_fiber", "fiber"]
    k_iso: float
    b_iso: float
    k_fiber: Arr[f64]
    b_fiber: Arr[f64]

    def __init__(
        self,
        k_iso: float,
        b_iso: float,
        k_ff: float,
        b_ff: float,
        k_fs: float,
        b_fs: float,
        k_ss: float,
        b_ss: float,
    ) -> None:
        self.k_iso = k_iso
        self.b_iso = b_iso
        self.k_fiber = np.array([[k_ff, k_fs, 0], [k_fs, k_ss, 0], [0, 0, 0]])
        self.b_fiber = np.array([[b_ff, b_fs, 0], [b_fs, b_ss, 0], [0, 0, 0]])

    def pk2(
        self,
        F: Arr[f64],
    ) -> Arr[f64]:
        E = compute_green_lagrange_strain(F)
        E_material = np.einsum("ij,mjk,lk->mil", self.fiber, E, self.fiber)
        E_material2 = E_material * E_material
        I_iso = np.einsum("mii->m", E)
        W_iso = self.k_iso * np.exp(self.b_iso * I_iso)
        S_iso = np.einsum("m,ij->mij", W_iso, _IDENTITY_MATRIX)
        b_fiber = np.einsum("ij,mij->mij", self.b_fiber, E_material2)
        W_fiber = np.einsum("ij,mij->mij", self.k_fiber, np.exp(b_fiber))
        S_fiber = W_fiber * E_material
        return S_iso + np.einsum("ji,mjk,kl->mil", self.fiber, S_fiber, self.fiber)
