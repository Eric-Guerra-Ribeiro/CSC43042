import math
from typing import Self
from abc import abstractmethod

import numpy as np
from sklearn.neighbors import NearestNeighbors

class Kernel:
    """A class for kernel density estimation, which also stores the cloud of points
    Attributes:
        d: int                 -- dimension of the ambient space
        data: list[np.ndarray] -- list of coordinates of the points (each of dimension self.d)
    """
    def __init__(self: Self, d: int, data: list[np.ndarray]):
        self.data = data
        try:
            self.data_np = np.vstack(data)
        except:
            pass
        self.d = d

    @abstractmethod
    def density(self: Self, x: np.ndarray) -> float:
        pass


class Radial(Kernel):
    """A class for radial kernel density estimation
    Attributes:
    d: int                 -- dimension of the ambient space
    data: list[np.ndarray] -- list of coordinates of the points (each of dimension self.d)
    bandwidth: float       -- 
    """
    def __init__(self, d, data, bandwidth):
        super().__init__(d, data)
        self.bandwidth = bandwidth
    
    @abstractmethod
    def volume(self) -> float:
        pass

    @abstractmethod
    def profile(self: Self, t: float) -> float:
        pass

    def density(self, x):
        return (
            np.sum(self.profile(np.sum(((self.data_np - x)/self.bandwidth)**2, axis=1)))
            /(len(self.data)*(self.bandwidth**self.d)*self.volume())
        )


class Flat(Radial):
    def volume(self):
        return np.pi**(0.5*self.d)/math.gamma(0.5*self.d + 1)
    
    def profile(self, t):
        return t <= 1.


class Gaussian(Radial):
    def __init__(self, d, data, bandwidth, c=0.328):
        super().__init__(d, data, bandwidth)

        mean = np.mean(self.data_np, axis=0)
        self.sample_std = np.sqrt(np.sum((self.data_np - mean)**2)/(len(self.data) - 1))
        self._volume = (2*np.pi)**(0.5*self.d)
        self.c = c

    def volume(self):
        return self._volume

    def profile(self, t):
        return np.exp(-0.5*t)

    def guess_bandwidth(self: Self)-> None:
        self.bandwidth = (0.25*len(self.data)*(self.d + 2))**(-1/(self.d + 4)) * self.sample_std

    def guess_bandwidth_challenge(self: Self)-> None:
        self.bandwidth = self.c * (0.25*len(self.data)*(self.d + 2))**(-1/(self.d + 4)) * self.sample_std


class Knn(Kernel):
    """A class for kernel density estimation with k-Nearest Neighbors
       derived from Kernel
    Attributes not already in Kernel:
        k: int      -- parameter for k-NN
        V: float    -- "volume" constant appearing in density
        neigh:    sklearn.neighbors.NearestNeighbors   -- data structure and methods for efficient k-NN computations
    """
    def __init__(self: Self, d: int, data: list[np.ndarray], k: int, V: float):
        super().__init__(d,data)
        self.k, self.V = k, V
        self.neigh = NearestNeighbors(n_neighbors=self.k)
        self.fit_knn()

    def fit_knn(self):
        """Computes the inner data structure acccording to the data points."""
        self.neigh.fit(np.array(self.data))

    def knn(self, x: np.ndarray, vk:int):
        """The vk nearest-neighbors (vk can be different from self.k)."""
        return [np.array(self.data[i]) for i in self.neigh.kneighbors([x], n_neighbors=vk)[1][0] ]

    def k_dist_knn(self, x: np.ndarray, vk: int) -> float:
        """The distance to vk-th nearest-neighbor."""
        return self.neigh.kneighbors([x], n_neighbors=vk)[0][0][vk-1]

    def density(self, x):
        return 0.5*self.k/(len(self.data)*self.V*self.k_dist_knn(x, self.k))
    
    def meanshift(self: Self, k: int) -> None:
        self.data = [np.mean(neighbours, axis=0) for x in self.data for neighbours in [self.knn(x, k)]]
        self.data_np = np.vstack(self.data)
        self.fit_knn()
