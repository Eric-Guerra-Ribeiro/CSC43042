import numpy as np
from math import sqrt

from sklearn.random_projection import GaussianRandomProjection

from TD.dataset import Dataset


class RandomProjection:
    """A random projection for downsampling the data.

    Attributes:
        original_dimension: int -- the dimension of the dataset before the projection
        col_class: int          -- the index of the column to classify
        projection_dim: int     -- the dimension of the dataset after the projection
        type_sample: str        -- the type of the projection (either "Gaussian" or something else)
        projection: np.ndarray  -- the matrix of the projection itself
    """

    def __init__(
        self,
        original_dimension: int,
        col_class: int,
        projection_dim: int,
        type_sample: str,
    ):
        self.original_dimension = original_dimension
        self.col_class = col_class
        self.projection_dim = projection_dim
        self.type_sample = type_sample
        if type_sample == "Gaussian":
            self.projection = RandomProjection.random_gaussian_matrix(original_dimension, projection_dim)
        else:
            self.projection = RandomProjection.random_rademacher_matrix(original_dimension, projection_dim)

    @staticmethod
    def random_gaussian_matrix(d: int, projection_dim: int) -> np.ndarray:
        """Creates a random Gaussian matrix."""
        return GaussianRandomProjection(n_components=projection_dim).fit_transform(np.eye(d))

    @staticmethod
    def random_rademacher_matrix(d: int, projection_dim: int) -> np.ndarray:
        """Creates a random Rademacher matrix."""
        return np.random.choice(
            a=[sqrt(3.0 / projection_dim) * v for v in [-1.0, 0.0, 1.0]],
            size=(d, projection_dim),
            replace=True,
            p=[1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        )

    @staticmethod
    def mean_dist(mat: np.ndarray) -> float:
        """Computes the mean distance of a matrix."""
        rows = np.shape(mat)[0]
        mean_dist = 0.0
        for i in range(rows - 1):
            for j in range(i + 1, rows):
                mean_dist += np.linalg.norm(mat[i, :] - mat[j, :])
        mean_dist /= rows * (rows - 1) / 2.0
        return mean_dist

    def projection_quality(self, dataset: Dataset) -> tuple[float, float]:
        """Computes the quality of the projection."""
        return self.mean_dist(dataset.instances), self.mean_dist(self.project(dataset).instances)

    def project(self, dataset: Dataset) -> Dataset:
        """Projects a dataset to a lower dimension."""
        assert (
            dataset.dim - 1 >= self.projection_dim
        ), "Impossible to project to higher dimensions!"

        ds_wo_col_class = np.delete(dataset.instances, [self.col_class], axis=1)
        minor_projected_data = ds_wo_col_class.dot(self.projection)
        # Append the column to predict to the end
        projected_data = np.c_[
            minor_projected_data, dataset.instances[:, self.col_class]
        ]

        return Dataset(dataset=projected_data)
