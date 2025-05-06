import numpy as np
from typing import Optional, Tuple
from TD.regression import Regression
from TD.dataset import Dataset

class LinearRegression(Regression):
    """
    Linear Regression class using the least squares method.
    """

    def __init__(self, dataset: Dataset, col_regr: int, fit_intercept: bool = True) -> None:
        """
        Initialize the LinearRegression instance.
        
        Parameters:
            dataset (Dataset): The dataset containing the samples.
            col_regr (int): The index of the target (regression) column.
            fit_intercept (bool): Whether to include an intercept term.
        """
        super().__init__(dataset, col_regr)
        self.m_beta = None  # Regression coefficients
        self.fit_intercept = fit_intercept   # Flag for intercept term
        # Compute the coefficients using the implemented methods.
        self.set_coefficients()
        

    def construct_matrix(self) -> np.ndarray:
        """
        Construct the design matrix (feature matrix) from the dataset.
        
        Returns:
            np.ndarray: The design matrix X.
        """
        # Ensure the dataset has more than one dimension.
        dim = self.m_dataset.get_dim()
        assert dim > 1

        if self.fit_intercept:
            design_mtx = np.ones((self.m_dataset.get_nbr_samples(), dim))
            design_mtx[:,1:] = np.delete(self.m_dataset.data, self.m_col_regr, 1)
            return design_mtx
        return np.delete(self.m_dataset.data, self.m_col_regr, 1)

        

    def construct_y(self) -> np.ndarray:
        """
        Construct the target vector from the dataset.
        
        Returns:
            np.ndarray: The target vector y.
        """
        # Ensure the dataset has more than one dimension.
            # Ensure the dataset has more than one dimension.
        dim = self.m_dataset.get_dim()
        assert dim > 1
        
        return self.m_dataset.data[:, self.m_col_regr]

    def set_coefficients(self) -> None:
        """
        Compute and set the regression coefficients using the least squares solution.
        """
        # Ensure the dataset has more than one dimension.
                # Ensure the dataset has more than one dimension.
        dim = self.m_dataset.get_dim()
        assert dim > 1
        # Compute the least squares solution for the coefficients.
        self.m_beta = np.linalg.lstsq(self.construct_matrix(), self.construct_y(), rcond=None)[0]

    def get_coefficients(self) -> Optional[np.ndarray]:
        """
        Get a copy of the computed regression coefficients.
        
        Returns:
            Optional[np.ndarray]: The regression coefficients if computed, else None.
        """
        return self.m_beta.copy() if self.m_beta is not None else None

    def show_coefficients(self) -> None:
        """
        Print the regression coefficients.
        """
        if self.m_beta is None:
            print("Coefficients not computed.")
            return

        expected_size= (self.m_dataset.get_dim() if self.fit_intercept 
                              else self.m_dataset.get_dim() - 1)
        if self.m_beta.size != expected_size:
            print(f"Warning: Coefficients size mismatch. Expected {expected_size}, got {self.m_beta.size}")

        print("beta = (" + " ".join(map(str, self.m_beta)) + ")")

    def print_raw_coefficients(self) -> None:
        """
        Print the regression coefficients in raw format.
        """
        if self.m_beta is None:
            print("{}")
            return
        print("{ " + ", ".join(map(str, self.m_beta)) + " }")

    def sum_of_squares(self, dataset: Dataset) -> Tuple[float, float, float]:
        """
        Compute the Total Sum of Squares (TSS), Explained Sum of Squares (ESS), 
        and Residual Sum of Squares (RSS) for the given dataset.
        
        Parameters:
            dataset (Dataset): The dataset to compute the sums of squares.
        
        Returns:
            Tuple[float, float, float]: A tuple containing (TSS, ESS, RSS).
        """
        # Ensure the dataset has the same dimensions as the training dataset.
        assert dataset.get_dim() == self.m_dataset.get_dim(), "Datasets must have the same dimensions."
        n: int = dataset.get_nbr_samples()
        dim: int = dataset.get_dim()
        
        assert dim > 1

        if self.fit_intercept:
            design_mtx = np.ones((dataset.get_nbr_samples(), dim))
            design_mtx[:,1:] = np.delete(dataset.data, self.m_col_regr, 1)
        else:
            design_mtx = np.delete(dataset.data, self.m_col_regr, 1)
        
        y_vector = dataset.data[:, self.m_col_regr]

        avg_y = np.mean(y_vector)

        y_estimated = design_mtx@self.m_beta

        norm_squared = lambda x: np.dot(x, x)

        tss = norm_squared(y_vector - avg_y)
        ess = norm_squared(y_estimated - avg_y)
        rss = norm_squared(y_vector - y_estimated)

        return tss, ess, rss

    def estimate(self, x: np.ndarray) -> float:
        """
        Estimate the target value for a given input feature vector.
        
        Parameters:
            x (np.ndarray): Feature vector (excluding the target column).
        
        Returns:
            float: The predicted target value.
        """
        # Verify that the input vector x has the correct dimension (i.e., x.size should equal dataset.get_dim() - 1).
        assert x.size == self.m_dataset.get_dim() - 1
        # Check that the regression coefficients have been computed (i.e., self.m_beta is not None).
        assert self.m_dataset is not None

        x_copy = x[:]

        if self.fit_intercept:
            x_copy = np.concatenate((np.ones(1), x_copy))
        return np.dot(x_copy, self.m_beta)

