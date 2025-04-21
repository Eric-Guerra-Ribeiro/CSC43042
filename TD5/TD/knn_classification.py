from sklearn.neighbors import KDTree
import numpy as np

from TD.classification import Classification
from TD.dataset import Dataset


class KnnClassification(Classification):
    """A k-NN classifier.

    Attributes:
        k: int          -- the number of nearest neighbors to use for classification
        kd_tree: KDTree -- the kd-tree used for computing shortest distances quickly
    """

    def __init__(self, k: int, dataset: Dataset, col_class: int):
        super().__init__(dataset, col_class)
        self.kd_tree = KDTree(np.delete(dataset.instances, col_class, 1))
        self.k = k

    def estimate(self, x: np.ndarray, threshold: float = 0.5) -> int:
        """Classify data point x for the given threshold."""
        probability = lambda x: self.uniform_probability(x.reshape(1, -1))
        # probability = lambda x: self.weighted_probability(x.reshape(1, -1))
        return 1 if probability(x) > threshold else 0

    def uniform_probability(self, x: np.ndarray) -> float:
        _, indeces = self.kd_tree.query(x, k=self.k)
        return np.mean(self.dataset.instances[indeces, self.col_class])
    
    def weighted_probability(self, x: np.ndarray) -> float:
        distances, indeces = self.kd_tree.query(x, k=self.k)
        return np.sum(self.dataset.instances[indeces, self.col_class]*distances)/np.sum(distances)
