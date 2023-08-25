from numpy.typing import NDArray as Arr
from numpy import float64 as f64, einsum


def calc_I_1(C: Arr[f64]) -> float:
    if C.shape == (3, 3):
        return einsum("ii->", C)
    elif C.shape[1:] == (3, 3):
        return einsum("nii->n", C)
    pass


def calc_I_2():
    pass


def calc_I_3():
    pass


def calc_I_k():
    pass
