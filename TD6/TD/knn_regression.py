import numpy as np
from scipy.spatial import KDTree

from TD.regression import Regression
from TD.dataset import Dataset

class KnnRegression(Regression):
    """
    K-Nearest Neighbors Regression class.
    
    Implements the k-nearest neighbors approach for regression,
    using a KDTree for efficient neighbor search.
    """

    def __init__(self, k: int, dataset: Dataset, col_regr: int) -> None:
        """
        Initialize the KnnRegression instance.
        
        Parameters:
            k (int): Number of neighbors to consider.
            dataset (Dataset): The dataset containing the training samples.
            col_regr (int): The index of the target (regression) column.
        """
        super().__init__(dataset, col_regr)  # Initialize parent Regression class
        pass

    def _build_kd_tree(self) -> KDTree:
        """
        Build a KDTree from the dataset's features (excluding the target column).
        
        Returns:
            KDTree: A KDTree built from the feature vectors.
        """
        n_samples: int = self.m_dataset.get_nbr_samples()
        n_features: int = self.m_dataset.get_dim() - 1  # Exclude target column

        # Initialize an array to store feature vectors.
        data_pts = np.zeros((n_samples, n_features))
        
        # Populate data_pts by iterating through all samples.
        for i in range(n_samples):
            j2 = 0  # Tracks original column indices while skipping target column
            for j in range(n_features):
                if j2 == self.m_col_regr:
                    j2 += 1  # Skip the target column in the original dataset
                data_pts[i, j] = self.m_dataset.get_instance(i)[j2]
                j2 += 1

        return KDTree(data_pts)

    def estimate(self, x: np.ndarray) -> float:
        """
        Estimate the target value for a given input feature vector.

        Parameters:
            x (np.ndarray): Feature vector (excluding the target column).

        Returns:
            float: The estimated target value.
        """

        # Verify input dimension matches dataset feature dimension.
        assert x.size == self.m_dataset.get_dim() - 1, "Input dimension does not match expected feature size."

        # Query KDTree for k nearest neighbors.
        distances, indices = self.m_kdTree.query(x, k=self.m_k)
        pass

    def get_k(self) -> int:
        """
        Get the number of neighbors used in regression.
        
        Returns:
            int: The number of neighbors (k).
        """
        return self.m_k

    def get_kdTree(self) -> KDTree:
        """
        Get the KDTree used for neighbor searches.
        
        Returns:
            KDTree: The KDTree instance.
        """
        return self.m_kdTree


