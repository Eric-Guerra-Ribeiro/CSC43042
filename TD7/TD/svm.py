import numpy as np
from qpsolvers import solve_qp
from TD.confusion_matrix import ConfusionMatrix
from TD.kernel import Kernel

class SVM:
    """Support Vector Machine classifier using quadratic programming."""

    def __init__(self, dataset, col_class: int, kernel: Kernel) -> None:
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
        return 2*self.dataset.data[:, self.col_class] - 1

    def __get_features(self) -> np.ndarray:
        """Extract features, removing the label column."""
        
        X = np.asarray([list(inst) for inst in self.dataset.get_instances()], dtype=float)
        return np.delete(X, self.col_class, axis=1)
        

    def __compute_kernel_matrix(self) -> np.ndarray:
        """Compute the full kernel matrix for the training data."""
        n = self.dataset.get_nbr_samples()
        return np.array([[self.kernel.k(x_i, x_j) for x_j in self.train_features] for x_i in self.train_features])

    def train(self, C: float) -> None:
        """Train the SVM by solving the dual optimization problem."""
        epsilon = 1e-8

        n_samples = self.dataset.get_nbr_samples()

        P = np.outer(self.train_labels, self.train_labels)*self.computed_kernel

        P = 0.5*(P + P.T) + epsilon*np.eye(n_samples)

        self.alphas = solve_qp(
            P, -np.ones(n_samples),
            G=np.vstack((-np.eye(n_samples), np.eye(n_samples))), h=np.hstack((np.zeros(n_samples), np.ones(n_samples)*C)),
            A=self.train_labels, b=np.zeros(1),
            solver='cvxopt'
        )

    def __compute_beta_0(self) -> None:
        """Compute the bias term (beta_0) from support vectors."""
        indexes = np.where(self.alphas > 0.)[0]
        return np.mean(self.train_features[indexes] - self.alphas[indexes]*(self.computed_kernel[indexes, indexes]@self.train_labels[indexes]))

    def predict(self, x: list[float]) -> int:
        """Predict the class (+1 or -1) for a given input vector."""
        x = np.array(x)

        return int(np.sign(sum(alpha_i*y_i*self.kernel.k(x_i, x) for alpha_i, y_i, x_i in zip(self.alphas, self.train_labels, self.train_features)) + self.beta_0))

    def test(self, test_dataset) -> ConfusionMatrix:
        """Evaluate the SVM on a test set and produce a confusion matrix."""
        conf_mtx = ConfusionMatrix()
        
        for instance in test_dataset.get_instances():
            true_label = int(instance[self.col_class])
            predicted_label = (self.predict(np.delete(instance, self.col_class, 0)) + 1)//2
            conf_mtx.add_prediction(true_label, predicted_label)
        
        return conf_mtx


    @property
    def alphas_(self) -> np.ndarray:
        return self.alphas

    @property
    def beta_0_(self) -> float:
        return self.beta_0

    @property
    def kernel_type(self):
        return self.kernel.kernel_type

