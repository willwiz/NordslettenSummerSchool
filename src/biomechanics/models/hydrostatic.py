from numpy.typing import NDArray as Arr
from numpy import float64 as f64, einsum
from biomechanics.kinematics.mapping import compute_right_cauchy_green
from biomechanics.kinematics.operations import inverse_fastest


def add_hydrostatic_pressure(S: Arr[f64], F: Arr[f64]) -> Arr[f64]:
    C = compute_right_cauchy_green(F)
    Cinv = inverse_fastest(C)
    p = C[:, 2, 2] * S[:, 2, 2]
    return S - einsum("n,nij->nij", p, Cinv)
