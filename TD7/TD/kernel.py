import numpy as np
from enum import Enum

# Enumeration for different kernel types
class KernelType(Enum):
    LINEAR = 0
    POLY = 1
    RBF = 2
    SIGMOID = 3
    RATQUAD = 4

class Kernel:
    def __init__(self, kernel_type: KernelType, degree: int = 0, gamma: float = 0.0, coef0: float = 0.0) -> None:
        """
        Initializes a Kernel object with specific parameters.

        :param kernel_type: Type of the kernel (from KernelType enum)
        :param degree: Degree of the polynomial kernel
        :param gamma: Scale parameter for RBF, POLY, and SIGMOID kernels
        :param coef0: Independent term for POLY, SIGMOID, and RATQUAD kernels
        """
        self.kernel_type = kernel_type
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0


        if kernel_type == KernelType.LINEAR:
            self._kernel_fun = self._kernel_linear
        elif kernel_type == KernelType.POLY:
            self._kernel_fun = self._kernel_poly
        elif kernel_type == KernelType.RBF:
            self._kernel_fun = self._kernel_rbf
        elif kernel_type == KernelType.SIGMOID:
            self._kernel_fun = self._kernel_sigmoid
        elif kernel_type == KernelType.RATQUAD:
            self._kernel_fun = self._kernel_ratquad
        else:
            raise ValueError("Invalid kernel type")


    def k(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Computes the kernel function between x1 and x2.
        """
        return self._kernel_fun(x1, x2)

    def _dot(self, x1: np.ndarray | list, x2: np.ndarray | list) -> float:
        """
        Compute dot product between two vectors.
        """
        x1 = np.asarray(x1, dtype=float).ravel()
        x2 = np.asarray(x2, dtype=float).ravel()
        if x1.size == 0 or x2.size == 0:
            return 0.0
        return float(np.dot(x1, x2))

    def _kernel_linear(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes the linear kernel."""
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.dot(x1, x2)

    def _kernel_poly(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes the polynomial kernel."""
        x1 = np.array(x1)
        x2 = np.array(x2)
        return (self.gamma*np.dot(x1, x2) + self.coef0)**self.degree

    def _kernel_rbf(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes the RBF (Gaussian) kernel."""
        x1 = np.array(x1)
        x2 = np.array(x2)
        norm_squared = lambda x: np.dot(x, x)
        return np.exp(-self.gamma*norm_squared(x1 - x2))

    def _kernel_sigmoid(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes the sigmoid kernel."""
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.tanh(self.gamma*np.dot(x1, x2) + self.coef0)

    def _kernel_ratquad(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes the rational quadratic kernel."""
        x1 = np.array(x1)
        x2 = np.array(x2)
        norm_squared = lambda x: np.dot(x, x)
        return self.coef0/(norm_squared(x1 - x2) + self.coef0)

    def get_kernel_type(self) -> KernelType:
        """Returns the kernel type."""
        return self.kernel_type

