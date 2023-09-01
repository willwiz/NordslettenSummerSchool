from dataclasses import dataclass
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray as Arr
from biomechanics.models.caputoD import (
    CaputoInitialize,
    caputo_derivative_linear,
    caputo_diffeq_linear,
)
from biomechanics._interfaces import HyperelasticModel, ViscoelasticModel


@dataclass(slots=True)
class CompositeViscoelasticModel(ViscoelasticModel):
    hyperelastic_models: list[HyperelasticModel] | None = None
    viscoelastic_models: list[ViscoelasticModel] | None = None

    def pk2(self, F: Arr[f64], time: Arr[f64]) -> Arr[f64]:
        res = np.zeros_like(F)
        if self.hyperelastic_models:
            for law in self.hyperelastic_models:
                res = res + law.pk2(F)
        if self.viscoelastic_models:
            for law in self.viscoelastic_models:
                res = res + law.pk2(F, time)
        return res


class FractionalVEModel(ViscoelasticModel):
    __slots__ = ["Np", "laws", "carp"]
    carp: CaputoInitialize
    laws: list[HyperelasticModel]

    def __init__(
        self,
        alpha: float,
        Tf: float,
        Np: int = 9,
        models: list[HyperelasticModel] | None = None,
    ) -> None:
        self.carp = CaputoInitialize(alpha, Tf, Np)
        self.laws = models

    def pk2(self, F: Arr[f64], time: Arr[f64]) -> Arr[f64]:
        pk2_hyperelastic = np.zeros_like(F)
        for law in self.laws:
            pk2_hyperelastic = pk2_hyperelastic + law.pk2(F)
        dt = np.diff(time, prepend=-1)
        return caputo_derivative_linear(self.carp, pk2_hyperelastic, dt)


class FractionalDiffEqModel(ViscoelasticModel):
    __slots__ = ["delta", "Np", "laws", "carp"]
    delta: float
    carp: CaputoInitialize
    hlaws: list[HyperelasticModel] | None
    vlaws: list[ViscoelasticModel] | None

    def __init__(
        self,
        alpha: float,
        delta: float,
        Tf: float,
        Np: int = 9,
        hyperelastic_models: list[HyperelasticModel] | None = None,
        viscoelastic_models: list[ViscoelasticModel] | None = None,
    ) -> None:
        self.delta = delta
        self.carp = CaputoInitialize(alpha, Tf, Np)
        self.hlaws = hyperelastic_models
        self.vlaws = viscoelastic_models

    def pk2(self, F: Arr[f64], time: Arr[f64]) -> Arr[f64]:
        pk2_hyperelastic = np.zeros_like(F)
        if self.hlaws:
            for law in self.hlaws:
                pk2_hyperelastic = pk2_hyperelastic + law.pk2(F)
        pk2_viscoelastic = np.zeros_like(F)
        if self.vlaws:
            for law in self.vlaws:
                pk2_viscoelastic = pk2_viscoelastic + law.pk2(F, time)
        dt = np.diff(time, prepend=-1)
        return caputo_diffeq_linear(
            self.delta, self.carp, pk2_hyperelastic, pk2_viscoelastic, dt
        )
