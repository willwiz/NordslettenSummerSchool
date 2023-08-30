from dataclasses import dataclass
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray as Arr
from biomechanics.models.caputoD import CaputoInitialize
from biomechanics._interfaces import HyperelasticModel, ViscoelaticModel


@dataclass(slots=True)
class CompositeViscoelasticModel(ViscoelaticModel):
    hyperelastic_models: list[HyperelasticModel]
    viscoelastic_models: list[ViscoelaticModel]

    def pk2(self, F: Arr[f64], time: Arr[f64]) -> Arr[f64]:
        res = np.zeros_like(F)
        for law in self.hyperelastic_models:
            res = res + law.pk2(F)
        for law in self.viscoelastic_models:
            res = res + law.pk2(F, time)
        return res


class FractionalVEModel(ViscoelaticModel):
    __slots__ = ["Np", "laws", "carp"]
    carp: CaputoInitialize
    Np: int | tuple[int, int]
    laws: list[HyperelasticModel]

    def __init__(
        self,
        laws: list[HyperelasticModel],
        alpha: float,
        Tf: float,
        Np: int = 9,
        dim: int | tuple[int, int] = (3, 3),
    ) -> None:
        pass

    def pk2(self, F: Arr[f64], dt: Arr[f64]) -> Arr[f64]:
        pk2_hyperelastic = np.zeros_like(F)
        for law in self.laws:
            pk2_hyperelastic = pk2_hyperelastic + law.pk2(F)
        df = np.zeros_like(F)
        df[1:] = pk2_hyperelastic[1:] - pk2_hyperelastic[:-1]
        Qk = np.einsum("", ek)
        return super().pk2(F, dt)
