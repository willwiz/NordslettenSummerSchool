from typing import Union
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
    F11: Union[Arr[f64], float] = 1.0,
    F12: Union[Arr[f64], float] = 0.0,
    F21: Union[Arr[f64], float] = 0.0,
    F22: Union[Arr[f64], float] = 1.0,
) -> Arr[f64]:
    # Validating Input
    n1 = F11.shape[0] if isinstance(F11, np.ndarray) else 1
    n2 = F12.shape[0] if isinstance(F12, np.ndarray) else 1
    n3 = F21.shape[0] if isinstance(F21, np.ndarray) else 1
    n4 = F22.shape[0] if isinstance(F22, np.ndarray) else 1
    n = max([n1, n2, n3, n4])
    for dim in [n1, n2, n3, n4]:
        if not (dim == n or dim == 1):
            raise ValueError
    # Construct Array
    res = np.zeros((n, 3, 3), dtype=f64)
    res[:, 0, 0] = F11
    res[:, 0, 1] = F12
    res[:, 1, 0] = F21
    res[:, 1, 1] = F22
    det = res[:, 0, 0] * res[:, 1, 1] - res[:, 0, 1] * res[:, 1, 0]
    res[:, 2, 2] = 1 / det
    return res
