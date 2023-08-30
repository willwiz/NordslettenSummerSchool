from numpy.typing import NDArray as Arr
from numpy import float64 as f64, einsum, array


def compute_right_cauchy_green(F: Arr[f64]) -> Arr[f64]:
    if F.shape == (3, 3):
        return F.T @ F
    elif F.shape[1:] == (3, 3):
        return einsum("nji,njk->nik", F, F)
    else:
        print(F, F.shape)


def compute_left_cauchy_green(F: Arr[f64]) -> Arr[f64]:
    if F.shape == (3, 3):
        return F @ F.T
    elif F.shape[1:] == (3, 3):
        return einsum("nij,nkj->nik", F, F)
    else:
        print(F, F.shape)


def compute_green_lagrange_strain(F: Arr[f64]) -> Arr[f64]:
    return compute_right_cauchy_green(F) - array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )


def compute_pk1_from_pk2(S: Arr[f64], F: Arr[f64]) -> Arr[f64]:
    if not S.shape == F.shape:
        raise ValueError
    if F.shape == (3, 3):
        return F @ S
    elif F.shape[1:] == (3, 3):
        return einsum("nij,njk->nik", F, S)
    else:
        print(F, F.shape)


def compute_cauchy_from_pk2(S: Arr[f64], F: Arr[f64]) -> Arr[f64]:
    if not S.shape == F.shape:
        raise ValueError
    if F.shape == (3, 3):
        return F @ S @ F.T
    elif F.shape[1:] == (3, 3):
        return einsum("nij,njk,nlk->nil", F, S, F)
    else:
        print(F, F.shape)
