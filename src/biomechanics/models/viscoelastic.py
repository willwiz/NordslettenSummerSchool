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


def array_time_derivative(arr: Arr[f64], dt: Arr[f64]) -> Arr[f64]:
    df = np.diff(arr, prepend=[arr[0]], axis=0)
    return np.einsum("mi...,m->mi...", df, 1.0 / dt)


class KelvinVoigtModel(ViscoelasticModel):
    __slots__ = ["w", "hlaws", "vlaws"]
    w: float
    laws: list[HyperelasticModel] | None

    def __init__(
        self,
        weight: float = 1.0,
        models: list[HyperelasticModel] | None = None,
    ) -> None:
        self.w = weight
        self.laws = models if models else list()

    def pk2(self, F: Arr[f64], time: Arr[f64]) -> Arr[f64]:
        pk2_hyperelastic = np.zeros_like(F)
        for law in self.laws:
            pk2_hyperelastic = pk2_hyperelastic + law.pk2(F)
        dt = np.diff(time, prepend=-1)
        return self.w * array_time_derivative(pk2_hyperelastic, dt)


def _solve_maxwell_diffeq(w: float, dt: Arr[f64], stresses: Arr[f64]):
    nt, *_ = stresses.shape
    r_dt = 1.0 / dt
    weights = w * r_dt
    res = np.zeros_like(stresses)
    for i in range(1, nt):
        res[i] = (stresses[i] + res[i - 1]) / (1.0 + weights[i])
    return res


class MaxwellModel(ViscoelasticModel):
    __slots__ = ["w", "hlaws"]
    w: float
    hlaws: list[HyperelasticModel]

    def __init__(
        self,
        weight: float = 0.0,
        models: list[HyperelasticModel] | None = None,
    ) -> None:
        self.w = weight
        self.hlaws = models

    def pk2(self, F: Arr[f64], time: Arr[f64]) -> Arr[f64]:
        dt = np.diff(time, prepend=-1)
        stress_laws = np.zeros_like(F)
        for law in self.hlaws:
            stress_laws = stress_laws + law.pk2(F)
        return _solve_maxwell_diffeq(self.w, dt, stress_laws)


class ZenerModel(ViscoelasticModel):
    __slots__ = ["w_right", "w_left", "hlaws", "vlaws"]
    w_left: float
    w_right: float
    hlaws: list[HyperelasticModel] | None
    vlaws: list[ViscoelasticModel] | None

    def __init__(
        self,
        weight_LHS: float = 0.0,
        weight_RHS: float = 1.0,
        hyperelastic_models: list[HyperelasticModel] | None = None,
        viscoelastic_models: list[ViscoelasticModel] | None = None,
    ) -> None:
        self.w_right = weight_RHS
        self.w_left = weight_LHS
        self.hlaws = hyperelastic_models if hyperelastic_models else list()
        self.vlaws = viscoelastic_models if viscoelastic_models else list()

    def pk2(self, F: Arr[f64], time: Arr[f64]) -> Arr[f64]:
        dt = np.diff(time, prepend=-1)
        pk2_hyperelastic = np.zeros_like(F)
        if self.hlaws:
            for law in self.hlaws:
                pk2_hyperelastic = pk2_hyperelastic + law.pk2(F)
        pk2_viscoelastic = np.zeros_like(F)
        if self.vlaws:
            for law in self.vlaws:
                pk2_viscoelastic = pk2_viscoelastic + law.pk2(F)

        rhs = pk2_hyperelastic + self.w_right * array_time_derivative(
            pk2_viscoelastic, dt
        )
        return _solve_maxwell_diffeq(self.w_left, dt, rhs)


class FractionalVEModel(ViscoelasticModel):
    __slots__ = ["carp", "laws"]
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
    __slots__ = ["delta", "carp", "hlaws", "vlaws"]
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
        self.hlaws = hyperelastic_models if hyperelastic_models else list()
        self.vlaws = viscoelastic_models if viscoelastic_models else list()

    def pk2(self, F: Arr[f64], time: Arr[f64]) -> Arr[f64]:
        pk2_hyperelastic = np.zeros_like(F)
        for law in self.hlaws:
            pk2_hyperelastic = pk2_hyperelastic + law.pk2(F)
        pk2_viscoelastic = np.zeros_like(F)
        for law in self.vlaws:
            pk2_viscoelastic = pk2_viscoelastic + law.pk2(F, time)
        dt = np.diff(time, prepend=-1)
        return caputo_diffeq_linear(
            self.delta, self.carp, pk2_hyperelastic, pk2_viscoelastic, dt
        )
