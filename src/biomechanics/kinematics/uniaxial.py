from dataclasses import dataclass
from multiprocessing import Value
import numpy as np
from numpy.typing import NDArray as Arr
from numpy import float64 as f64


def construct_tensor_uniaxial(stretch: Arr[f64]) -> Arr[f64]:
    n = stretch.shape[0]
    res = np.zeros((n, 3, 3), dtype=f64)
    res[:, 0, 0] = stretch
    res[:, 1, 1] = res[:, 2, 2] = 1.0 / np.sqrt(stretch)
    return res


def construct_tensor_biaxial(
    stretch1: Arr[f64] | float = 1.0,
    stretch2: Arr[f64] | float = 1.0,
    shear12: Arr[f64] | float = 0.0,
    shear21: Arr[f64] | float = 0.0,
) -> Arr[f64]:
    # Validating Input
    n1 = stretch1.shape[0] if isinstance(stretch1, np.ndarray) else 1
    n2 = stretch2.shape[0] if isinstance(stretch2, np.ndarray) else 1
    n3 = shear12.shape[0] if isinstance(shear12, np.ndarray) else 1
    n4 = shear21.shape[0] if isinstance(shear21, np.ndarray) else 1
    n = max([n1, n2, n3, n4])
    for dim in [n1, n2, n3, n4]:
        if not (dim == n or dim == 1):
            raise ValueError
    # Construct Array
    res = np.zeros((n, 3, 3), dtype=f64)
    res[:, 0, 0] = stretch1
    res[:, 0, 1] = shear12
    res[:, 1, 0] = shear21
    res[:, 1, 1] = stretch2
    det = res[:, 0, 0] * res[:, 1, 1] - res[:, 0, 1] * res[:, 1, 0]
    res[:, 2, 2] = 1 / det
    return res
