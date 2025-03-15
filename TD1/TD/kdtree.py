"""
kdtree module
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from TD.nearest_neighbor import NearestNeighborSearch


def median(X: np.ndarray, start: int, stop: int, c: int) -> float:
    """
    Returns median of array X between indices start and stop for coordinate c
    """
    assert stop - start >= 0, "Requested stop > start"
    assert X.ndim == 2, "2D required"
    sliced_X = X[start:stop]
    sorted_array = sliced_X[sliced_X[:, c].argsort()]
    med = sorted_array[(stop - start)//2, c]
    return med


def swap(X: np.ndarray, idx1, idx2) -> None:
    """Swaps two rows of a 2D numpy array"""
    X[idx1, :], X[idx2, :] = X[idx2, :], X[idx1, :]


def partition(X: np.ndarray, start: int, stop: int, c: int) -> int:
    """
    Partitions the array X between start and stop wrt to its median along a coordinate c
    """
    idx = (start + stop)//2
    X[start:stop] = X[start:stop][X[start:stop, c].argsort()]
    return idx


@dataclass
class Node:
    idx: int
    med: float = np.inf
    c: int = 0
    left: Self = (
        None  # Self denotes that left (and right) is of same type as self == Node
    )
    right: Self = None


class KDTree(NearestNeighborSearch):
    def __init__(self, X):
        """
        Contrary to LinearScan, KDTree's constructor must build the tree
        To that end, we will loop through the coordinates of X,
        hence the need for the `dim` attribute below.
        """
        super().__init__(X)
        self.dim = X.shape[1]
        self.build()

    def _build(self, start: int, stop: int, c: int) -> Node | None:
        """
        Builds a node with a correct index by partitioning X along c between start and stop,
        including left and right children nodes
        """
        assert stop >= start, "Indices issue"
        if stop == start:
            return
        if stop == start + 1:
            return Node(start)
        
        median_idx = partition(self.X, start, stop, c)
        next_c = (c + 1) % self.dim

        return Node(
            median_idx, self.X[median_idx, c], c,
            self._build(start, median_idx, next_c),
            self._build(median_idx + 1, stop, next_c)
        )

    def reset(self):
        """
        Resets current estimation of distance to and index of nearest neighbor
        """
        self._current_dist = np.inf
        self._current_idx = -1

    def build(self):
        """
        Builds the kdtree
        """
        self.reset()
        self.root = self._build(0, len(self.X), 0)

    def _defeatist(self, node: Node | None, x: np.ndarray):
        """
        Defeatist search of nearest neighbor of x in node
        """
        if node is None:
            return

        dist = self.metric(x, self.X[node.idx])
        if self._current_dist > dist:
            self._current_dist = dist
            self._current_idx = node.idx
    
        if x[node.c] < node.med:
            self._defeatist(node.left, x)
        else:
            self._defeatist(node.right, x)

    def _backtracking(self, node: Node | None, x: np.ndarray):
        """
        Backtracking search of nearest neighbor of x in node
        """
        if node is None:
            return

        dist = self.metric(x, self.X[node.idx])
        if self._current_dist > dist:
            self._current_dist = dist
            self._current_idx = node.idx
    
        if x[node.c] < node.med:
            self._backtracking(node.left, x)
            if x[node.c] + self._current_dist > node.med:
                self._backtracking(node.right, x)
        else:
            self._backtracking(node.right, x)
            if x[node.c] - self._current_dist < node.med:
                self._backtracking(node.left, x)

    def query(self, x, mode: str = "backtracking"):
        """
        Queries given mode 'backtracking' or 'defeatist'
        """
        super().query(x)
        self.reset()
        if mode == "defeatist":
            self._defeatist(self.root, x)
        elif mode == "backtracking":
            self._backtracking(self.root, x)
        else:
            raise ValueError("Incorrect mode!")
        return self._current_dist, self._current_idx

    def set_xaggle_config(self):
        self.mode = "defeatist"
