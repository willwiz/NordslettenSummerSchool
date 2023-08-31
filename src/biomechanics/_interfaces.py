import abc
from numpy.typing import NDArray as Arr
from numpy import float64 as f64


class HyperelasticModel(abc.ABC):
    """Hyperelastic Models

    Methods:
        pk2(Arr[f64, (2,2)]) : return the 2nd Piola Kirchhoff tensor
        pk1(Arr[f64, (2,2)]) : return the 1nd Piola Kirchhoff tensor
        cauchy(Arr[f64, (2,2)]) : return the Cauch stress tensor

    """

    @abc.abstractmethod
    def pk2(self, F: Arr[f64]) -> Arr[f64]:
        """Take the right cauchy green strain tensor and returns the 2nd Piola Kirchhoff stress tensor"""
        pass

    # @abc.abstractmethod
    # def pk1(self, x: Arr[f64]) -> Arr[f64]:
    #     """Take the right cauchy green strain tensor and returns the 2nd Piola Kirchhoff stress tensor"""
    #     pass

    # @abc.abstractmethod
    # def cauchy(self, x: Arr[f64]) -> Arr[f64]:
    #     """Take the right cauchy green strain tensor and returns the 2nd Piola Kirchhoff stress tensor"""
    #     pass


class ViscoelasticModel(abc.ABC):
    """Hyperelastic Models

    Methods:
        pk2(Arr[f64, (2,2)]) : return the 2nd Piola Kirchhoff tensor
        pk1(Arr[f64, (2,2)]) : return the 1nd Piola Kirchhoff tensor
        cauchy(Arr[f64, (2,2)]) : return the Cauch stress tensor

    """

    @abc.abstractmethod
    def pk2(self, F: Arr[f64], time: Arr[f64]) -> Arr[f64]:
        """Take the right cauchy green strain tensor and returns the 2nd Piola Kirchhoff stress tensor"""
        pass
