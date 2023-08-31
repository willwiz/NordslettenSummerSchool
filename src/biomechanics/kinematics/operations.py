from numpy.typing import NDArray as Arr
from numpy import float64 as f64, einsum


def determinate2Dpart(a: Arr[f64]) -> Arr[f64]:
    return a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]


def inverse_fastest(a: Arr[f64]) -> Arr[f64]:
    if a.shape[1:] != (3, 3):
        raise ValueError
    det = 1.0 / determinate2Dpart(a)
    b = -a
    b[:, 0, 0] = a[:, 1, 1]
    b[:, 1, 1] = a[:, 0, 0]

    res = einsum("n,nij->nij", det, b)
    res[:, 2, 2] = 1.0 / a[:, 2, 2]
    return res
