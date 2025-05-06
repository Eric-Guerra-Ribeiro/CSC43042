import numpy as np
from qpsolvers import solve_qp
from TD.confusion_matrix import ConfusionMatrix

class SVM:
    """Support Vector Machine classifier using quadratic programming."""

    def __init__(self, dataset, col_class: int, kernel: 'Kernel') -> None:
        """Initialize the SVM with dataset, target column, and kernel."""
        
        self.dataset = dataset
        self.col_class = col_class
        self.kernel = kernel
        self.alphas = None
        self.beta_0 = 0.0
        self.train_labels = self.__get_labels()
        self.train_features = self.__get_features()
        self.computed_kernel = self.__compute_kernel_matrix()
        
    def __get_labels(self) -> np.ndarray:
        """Extract and transform labels into -1/+1."""
        pass

    def __get_features(self) -> np.ndarray:
        """Extract features, removing the label column."""
        
        X = np.asarray([list(inst) for inst in self.dataset.get_instances()], dtype=float)
        return np.delete(X, self.col_class, axis=1)
        

    def __compute_kernel_matrix(self) -> np.ndarray:
        """Compute the full kernel matrix for the training data."""
        
        pass

    def train(self, C: float) -> None:
        """Train the SVM by solving the dual optimization problem."""
        pass

    def __compute_beta_0(self) -> None:
        """Compute the bias term (beta_0) from support vectors."""
        pass

    def predict(self, x: list[float]) -> int:
        """Predict the class (+1 or -1) for a given input vector."""
        pass

    def test(self, test_dataset) -> ConfusionMatrix:
        """Evaluate the SVM on a test set and produce a confusion matrix."""
        pass


    @property
    def alphas_(self) -> np.ndarray:
        return self.alphas

    @property
    def beta_0_(self) -> float:
        return self.beta_0

    @property
    def kernel_type(self):
        return self.kernel.kernel_type

