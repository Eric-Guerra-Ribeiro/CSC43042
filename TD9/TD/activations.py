#! /usr/bin/env -S python3
"""
activations module
"""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Activation(ABC):
    def __new__(self, x: np.ndarray) -> np.ndarray:
        """
        What to do when instantiating an object: this basically says Activation(x)
        will be equivalent to Activation()(x).
        """
        return self.__call__(x)

    @staticmethod
    @abstractmethod
    def __call__(x: np.ndarray) -> np.ndarray:
        """
        :param numpy.ndarray x: (x_i^j) for 1..i..n, 1..j..p
        """
        pass

    @staticmethod
    @abstractmethod
    def prime(x: np.ndarray) -> np.ndarray:
        """
        (Element-wise) Derivative of the loss function.

        :param numpy.ndarray y: y for each sample
        :return: derivative of activation function w.r.t. x
        :rtype: numpy.ndarray
        """
        pass

    @classmethod
    def plot(cls, x: np.ndarray):
        x = np.arange(-5, 5, 0.01)
        plt.plot(x, cls.__call__(x))
        plt.plot(x, cls.prime(x))
        plt.legend((cls.__name__, cls.__name__ + " prime"))
        plt.show()


class Identity(Activation):
    """
    Identity activation node
    """
    @staticmethod
    def __call__(x: np.ndarray) -> np.ndarray:
        """
        :return: identity of input
        """
        pass

    @staticmethod
    def prime(x: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(Activation):    
    """
    Threshold activation node
    """
    @staticmethod
    def __call__(x: np.ndarray) -> np.ndarray:
        """
        :return: sigmoid transform of input
        """
        pass

    @staticmethod
    def prime(x: np.ndarray) -> np.ndarray:
        """
        Implements the derivative of the sigmoid activation function
        (you can make good use of Sigmoid(...) but you should probably
        only call it once!)
        """
        pass
