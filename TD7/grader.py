#! /usr/bin/env python3
import sys
import unittest
import numpy as np
from TD.dataset import Dataset
from TD.kernel import Kernel, KernelType
from TD.confusion_matrix import ConfusionMatrix
from TD.svm import SVM
from TD.mail_selector import MailSelector

"""
Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 6,
  "names" : [
      "kernel.py::test_kernels",
      "svm.py::test_svm_constructor",
      "svm.py::test_svm_train",
      "svm.py::test_svm_predict",
      "svm.py::test_svm_test",
      "mail_selector.py::test_mail_selector"
      ],
  "points" : [15, 15, 15, 15, 20, 20]
}
[END-AUTOGRADER-ANNOTATION]
"""


def print_help():
    print(
        "./grader script. Usage: ./grader.py test_number, e.g., ./grader.py 1 for the 1st exercise."
    )
    print("N.B.: ./grader.py 0 runs all tests.")
    print(f"You provided {sys.argv}.")
    exit(1)

# Tolerance for numerical comparisons
deps = 0.001

def rel_error(a: float | np.ndarray, b: float | np.ndarray) -> float:
    """Scalar relative error; returns 0 for identical arrays/lists."""
    if isinstance(a, (list, np.ndarray)):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        # rms‑style relative error
        num   = np.linalg.norm(a - b)
        denom = np.linalg.norm(a) if np.linalg.norm(a) else 1.0
        return num / denom
    return abs(a - b) / abs(a) if a != 0 else 0.0

def test_rel_error(out, fn_name: str, result, expected, delta) -> bool:
    """Return True/False; optionally log to any *stream‑like* object."""
    if isinstance(result, (list, np.ndarray)):
        success = np.allclose(result, expected, rtol=delta, atol=delta)
    else:
        success = rel_error(result, expected) <= delta


    return success


class Grader(unittest.TestCase):

    def test_kernels(self):
        entity_name = "Ex1_Testing_kernels"
        x1 = [1.0, 0.0]
        x2 = [2.0, 1.0]
        EPS = 0.00001

        # Linear Kernel
        kernel_linear = Kernel(KernelType.LINEAR)
        self.assertTrue(test_rel_error(self, "test linear 1", kernel_linear.k(x1, x2), 2.0, EPS))

        # Polynomial Kernel (degree 2, gamma=1, coef0=0)
        kernel_poly = Kernel(KernelType.POLY, degree=2, gamma=1.0, coef0=0.0)
        expected_poly = (np.dot(x1, x2)) ** 2
        self.assertTrue(test_rel_error(self, "test poly 1", kernel_poly.k(x1, x2), expected_poly, EPS))

        # RBF Kernel (gamma=1.0)
        kernel_rbf = Kernel(KernelType.RBF, gamma=1.0)
        expected_rbf = np.exp(-1.0 * np.linalg.norm(np.subtract(x1, x2)) ** 2)
        self.assertTrue(test_rel_error(self, "test rbf 1", kernel_rbf.k(x1, x2), expected_rbf, EPS))

        # Sigmoid Kernel (gamma=1.0, coef0=0)
        kernel_sigmoid = Kernel(KernelType.SIGMOID, gamma=1.0, coef0=0.0)
        expected_sigmoid = np.tanh(np.dot(x1, x2))
        self.assertTrue(test_rel_error(self, "test sigmoid 1", kernel_sigmoid.k(x1, x2), expected_sigmoid, EPS))

        # Rational Quadratic Kernel
        kernel_ratquad = Kernel(KernelType.RATQUAD)
        diff = np.subtract(x1, x2)
        expected_ratquad = 1.0 - (np.dot(diff, diff) / (np.dot(diff, diff) + 1))
        self.assertTrue(test_rel_error(self, "test ratquad 1", kernel_ratquad.k(x1, x2), expected_ratquad, EPS))

    def test_svm_constructor(self):
        entity_name = "Ex2_Testing_constructor"
        train_file = "csv/tests1.csv"
        col_class = 0
        train_dataset = Dataset(train_file)
        kernel = Kernel(KernelType.LINEAR)
        svm = SVM(train_dataset, col_class, kernel)
        train_labels = [1, 1, -1, -1]
        train_features = [[0.5], [1], [-1], [-0.5]]
        EPS = 0.00001
        computed = svm._SVM__compute_kernel_matrix()  # re-compute manually
        stored = svm.computed_kernel  # should match
        self.assertTrue(np.allclose(computed, stored, atol=EPS),"Compute the full kernel matrix for the training data mismatch.")
        # Test SVM constructor here
        self.assertTrue(test_rel_error(self, "test train_labels", svm.train_labels, train_labels, EPS))
        expected_labels = [1, 1, -1, -1]
        expected_features = [[0.5], [1], [-1], [-0.5]]
        expected_kernel = np.array([
            [ 0.25,  0.5 , -0.5 , -0.25],
            [ 0.5 ,  1.0 , -1.0 , -0.5 ],
            [-0.5 , -1.0 ,  1.0 ,  0.5 ],
            [-0.25, -0.5 ,  0.5 ,  0.25]
        ])

        EPS = 1e-5
        self.assertTrue(
            np.allclose(svm._SVM__compute_kernel_matrix(), svm.computed_kernel, atol=EPS),
            f"[{entity_name}] Precomputed kernel matrix mismatch."
        )
        self.assertTrue(
            np.allclose(svm.train_labels, expected_labels, atol=EPS),
            f"[{entity_name}] Train labels mismatch."
        )

    def test_svm_train(self):
        train_file = "csv/tests1.csv"
        col_class = 0
        train_dataset = Dataset(train_file)
        kernel = Kernel(KernelType.LINEAR)
        svm = SVM(train_dataset, col_class, kernel)
        svm.train(1.0)
        # Test SVM training here
        self.assertTrue(test_rel_error(self, "test alpha[0]", svm.alphas[0], 1.0, 0.0001))

    def test_svm_predict(self):
        train_file = "csv/tests1.csv"
        col_class = 0
        train_dataset = Dataset(train_file)
        kernel = Kernel(KernelType.LINEAR)
        svm = SVM(train_dataset, col_class, kernel)
        svm.train(1.0)
        # Test SVM prediction here
        self.assertTrue(test_rel_error(self, "test f_hat", svm.predict([2]), 1, 0.0001))

    def test_svm_test(self):
        train_file = "csv/tests1.csv"
        test_file = "csv/tests2.csv"
        col_class = 0
        train_dataset = Dataset(train_file)
        test_dataset = Dataset(test_file)
        kernel = Kernel(KernelType.LINEAR)
        svm = SVM(train_dataset, col_class, kernel)
        svm.train(1.0)
        cm = svm.test(test_dataset)
        # Test confusion matrix metrics here
        self.assertTrue(test_rel_error(self, "TP", cm.get_tp(), 2, 0.0001))
    
    def test_mail_selector(self):
   
        # @OFF
        selector = MailSelector(
            train_file="csv/mail_train_server.csv",
            test_file="csv/mail_test_server.csv",
            kernel_type=KernelType.RBF,
            gamma=0.5,
            C=1.0,
        )
        detection_rate, false_alarm_rate, precision, f_score = selector.run()

        EPS = 0.02  # 2% tolerance
        self.assertTrue(test_rel_error(self, "Detection rate", detection_rate, 0.11290322580645161, EPS))
        self.assertTrue(test_rel_error(self, "False alarm rate", false_alarm_rate, 0.0000, EPS))
        self.assertTrue(test_rel_error(self, "Precision", precision, 1.0000, EPS))
        self.assertTrue(test_rel_error(self, "F-score", f_score, 0.202898, EPS))

        

def suite(test_nb):
    suite = unittest.TestSuite()
    test_name = [
        "test_kernels",
        "test_svm_constructor",
        "test_svm_train",
        "test_svm_predict",
        "test_svm_test",
        "test_mail_selector"
    ]
    if test_nb > 0:
        suite.addTest(Grader(test_name[test_nb - 1]))
    else:
        for name in test_name:
            suite.addTest(Grader(name))

    return suite

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
    try:
        test_nb = int(sys.argv[1])
    except ValueError as e:
        print(
            f"You probably didn't pass an int to ./grader.py: passed {sys.argv[1]}; error: {e}"
        )
        exit(1)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite(test_nb))