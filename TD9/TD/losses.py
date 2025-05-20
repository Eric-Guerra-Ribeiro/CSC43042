#! /usr/bin/env -S python3 
"""
losses module
"""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Loss(ABC):
    def __new__(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        return self.__call__(y_hat, y)

    @staticmethod
    @abstractmethod
    def __call__(y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        :param numpy.ndarray y_hat: predicted target(s) (can be multi-dimensional, i.e.
            several targets), possibly for several targets (thus is always 2D)
        :param numpy.ndarray y: target(s) values per sample
        """
        pass

    @staticmethod
    @abstractmethod
    def prime(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        (Element-wise) Derivative of the loss function.
        """
        pass

    @classmethod
    @abstractmethod
    def plot(cls) -> None:
        pass


class Quadratic(Loss):
    """Quadratic loss function"""
    @staticmethod
    def __call__(y_hat: np.ndarray, y: np.ndarray) -> float:
        pass
    
    @staticmethod
    def prime(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def plot(cls) -> None:
        x = np.arange(-5., 5., .01)
        plt.plot(x, [cls.__call__(x_i, np.zeros(1)) for x_i in x])
        plt.title(cls.__name__)
        plt.show()


class CrossEntropy(Loss):
    """Cross Entropy loss function"""
    @staticmethod
    def __call__(y_hat: np.ndarray, y: np.ndarray) -> float:
        pass

    @staticmethod
    def prime(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def plot(cls) -> None:
        x = np.arange(1e-9, 1-1e-9, .01)
        plt.plot(x, [cls.__call__(x_i, np.zeros(1)) for x_i in x])
        plt.title(cls.__name__)
        plt.show()
