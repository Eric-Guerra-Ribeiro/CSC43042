#! /usr/bin/env python3
from math import sqrt
import sys
import time
import unittest
import numpy as np

from TD.dataset import Dataset
from TD.knn_classification import KnnClassification
from TD.confusion_matrix import ConfusionMatrix
from TD.random_projection import RandomProjection
from TD.knn_application import *


"""
Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 6,
  "names" : [
      "knn_classification.py::test_constructor",
      "knn_classification.py::test_estimate",
      "confusion_matrix.py::test_confusion_matrix",
      "random_projection.py::test_gaussian",
      "random_projection.py::test_projection",
      "knn_application.py::kaggle_exercise"
      ],
  "points" : [10, 15, 10, 10, 25, 30]
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


class Grader(unittest.TestCase):
    def test_constructor(self):
        file_path = "csv/mail_train.csv"
        k = 3
        col_class = 0

        ds = Dataset(file_path)
        knnc = KnnClassification(k, ds, col_class)
        a = knnc.kd_tree.get_arrays()[0]

        self.assertEqual(a[25, 1898], 1, msg="knn constructor -- data loaded")
        self.assertEqual(a[28, 4], 1, msg="knn constructor -- data loaded")
        self.assertEqual(a[2250, 1], 0, msg="knn constructor -- data loaded")
        self.assertEqual(a[2250, 5], 0, msg="knn constructor -- data loaded")
        self.assertEqual(knnc.k, k, msg="knn constructor -- k")
        self.assertEqual(np.shape(a)[1], 1899, msg="knn constructor -- dim")
        self.assertEqual(np.shape(a)[0], 2251, msg="knn constructor -- n_pts")

    def test_estimate(self):
        train_path = "csv/mail_train.csv"
        test_path = "csv/mail_test.csv"
        k = 3
        col_class = 0

        ds_train = Dataset(train_path)
        ds_test = Dataset(test_path)
        knnc = KnnClassification(k, ds_train, col_class)

        mer = 0.0
        for i in range(ds_test.nsamples):
            sample = ds_test.instances[i]
            query = np.delete(sample, [col_class], axis=0)
            estim = knnc.estimate(query)
            mer += abs(estim - sample[col_class]) / ds_test.nsamples

        self.assertLessEqual(mer, 0.2, msg="knn estimate")

    def test_confusion_matrix(self):
        cm = ConfusionMatrix()

        self.assertEqual(cm.fn, 0, "confusion matrix -- fn")
        self.assertEqual(cm.fp, 0, "confusion matrix -- fp")
        self.assertEqual(cm.tn, 0, "confusion matrix -- tn")
        self.assertEqual(cm.tp, 0, "confusion matrix -- tp")

        cm.add_prediction(0, 0)
        self.assertEqual(cm.fn, 0, "confusion matrix -- fn")
        self.assertEqual(cm.fp, 0, "confusion matrix -- fp")
        self.assertEqual(cm.tn, 1, "confusion matrix -- tn")
        self.assertEqual(cm.tp, 0, "confusion matrix -- tp")

        cm.add_prediction(1, 1)
        self.assertEqual(cm.fn, 0, "confusion matrix -- fn")
        self.assertEqual(cm.fp, 0, "confusion matrix -- fp")
        self.assertEqual(cm.tn, 1, "confusion matrix -- tn")
        self.assertEqual(cm.tp, 1, "confusion matrix -- tp")

        cm.add_prediction(0, 1)
        self.assertEqual(cm.fn, 0, "confusion matrix -- fn")
        self.assertEqual(cm.fp, 1, "confusion matrix -- fp")
        self.assertEqual(cm.tn, 1, "confusion matrix -- tn")
        self.assertEqual(cm.tp, 1, "confusion matrix -- tp")

        cm.add_prediction(1, 0)
        self.assertEqual(cm.fn, 1, "confusion matrix -- fn")
        self.assertEqual(cm.fp, 1, "confusion matrix -- fp")
        self.assertEqual(cm.tn, 1, "confusion matrix -- tn")
        self.assertEqual(cm.tp, 1, "confusion matrix -- tp")

        self.assertEqual(cm.precision(), 0.5, "confusion matrix -- precision")
        self.assertEqual(cm.f_score(), 0.5, "confusion matrix -- f-score")
        self.assertEqual(
            cm.false_alarm_rate(), 0.5, "confusion matrix -- false-alarm rate"
        )
        self.assertEqual(cm.detection_rate(), 0.5, "confusion matrix -- detection rate")
        self.assertEqual(cm.error_rate(), 0.5, "confusion matrix -- error rate")

    def test_gaussian(self):
        sizes = [(40, 6), (30, 5), (60, 25)]

        for d, l in sizes:
            g = RandomProjection.random_gaussian_matrix(d, l)
            shp = np.shape(g)
            self.assertEqual(shp[0], d, "Gaussian matrix -- rows")
            self.assertEqual(shp[1], l, "Gaussian matrix -- columns")
            self.assertAlmostEqual(
                np.mean(g), 0.0, msg="Gaussian matrix -- mean", delta=0.1
            )
            self.assertAlmostEqual(
                np.var(g), 1 / l, msg="Gaussian matrix -- variance", delta=0.1
            )
            rdm_values = [sqrt(3.0 / l) * v for v in [0.0, -1.0, 1.0]]
            self.assertEqual(
                np.all(np.isin(g, rdm_values)),
                False,
                msg="Gaussian matrix -- not Rademacher",
            )

    def test_projection(self):
        cases = [
            (20, "csv/mail_train.csv", "Gaussian", 0),
            (20, "csv/mail_train.csv", "Rademacher", 0),
        ]

        for projection_dim, file_path, type_sample, col_class in cases:
            train_ds = Dataset(file_path)
            rnd_proj = RandomProjection(
                train_ds.dim - 1, col_class, projection_dim, type_sample
            )

            self.assertEqual(
                rnd_proj.original_dimension,
                train_ds.dim - 1,
                "random projection -- constructor, original_dimension",
            )
            self.assertEqual(
                rnd_proj.col_class,
                col_class,
                "random projection -- constructor, col_class",
            )
            self.assertEqual(
                rnd_proj.projection_dim,
                projection_dim,
                "random projection -- constructor, projection_dimension",
            )
            self.assertEqual(
                rnd_proj.type_sample,
                type_sample,
                "random projection -- constructor, type_sample",
            )

            (rows, cols) = np.shape(rnd_proj.projection)
            self.assertEqual(
                rows,
                train_ds.dim - 1,
                "random projection -- constructor, projection rows",
            )
            self.assertEqual(
                cols,
                projection_dim,
                "random projection -- constructor, projection columns",
            )
            self.assertAlmostEqual(
                np.mean(rnd_proj.projection),
                0.0,
                msg="random projection -- constructor, mean",
                delta=0.15,
            )
            self.assertAlmostEqual(
                np.var(rnd_proj.projection),
                1.0 / projection_dim,
                msg="random projection -- constructor, projection variance",
                delta=0.1,
            )

            (orig, proj) = rnd_proj.projection_quality(train_ds)
            self.assertAlmostEqual(
                orig,
                12.0,
                msg="random projection -- projection quality, original",
                delta=0.5,
            )
            self.assertAlmostEqual(
                proj,
                12.0,
                msg="random projection -- projection quality, projection",
                delta=1.0,
            )

    def kaggle_exercise(self):
        # Read data
        train_ds = Dataset(
            "../samples_train_full.csv"
        )  # Path needs to be like this on the grader!
        test_ds = Dataset(
            "../samples_test_full.csv"
        )  # Path needs to be like this on the grader!
        col_class = train_ds.dim - 1

        # Your data
        (k, projection_dimension, type_sample, threshold, seed) = my_parametrization(
            train_ds.dim, train_ds.nsamples
        )

        # Set the seed
        np.random.seed(seed)

        # Create projection
        rnd_proj = RandomProjection(
            train_ds.dim - 1, col_class, projection_dimension, type_sample
        )
        proj_ds = rnd_proj.project(train_ds)
        proj_test_ds = rnd_proj.project(test_ds)

        # Create k-NN baseline classifier
        knn = KnnClassification(k, train_ds, col_class)

        # Compute the quality of the result
        cm = ConfusionMatrix()
        start_time_test_orig = time.time()
        for i in range(test_ds.nsamples):
            sample = test_ds.instances[i]
            query = np.delete(sample, [col_class], axis=0)
            predicted_label = knn.estimate(query, threshold)
            true_label = int(sample[col_class])
            cm.add_prediction(true_label, predicted_label)
        duration_test_orig = time.time() - start_time_test_orig

        # Create k-NN classifier
        knn_proj = KnnClassification(k, proj_ds, projection_dimension)

        # Compute the quality of the result
        cm = ConfusionMatrix()
        start_time_test_proj = time.time()
        for i in range(proj_test_ds.nsamples):
            sample = proj_test_ds.instances[i]
            query = np.delete(sample, [projection_dimension], axis=0)
            predicted_label = knn_proj.estimate(query, threshold)
            true_label = int(sample[projection_dimension])
            cm.add_prediction(true_label, predicted_label)
        duration_test_proj = time.time() - start_time_test_proj

        # Calculate results
        speed_up = duration_test_orig / duration_test_proj
        f_score = cm.f_score()

        # Grade the outcome
        score = speed_up + 1.0 - (1.0 - f_score) * 100
        print(f"Score: {score}")
        match score:
            case c if c < 3.0:
                print("D")
            case c if 3.0 <= c < 3.5:
                print("C")
            case c if 3.5 <= c < 4.0:
                print("B")
            case c if 4.0 <= c < 4.2:
                print("A")
            case c if c >= 4.2:
                print("A+")


def suite(test_nb):
    suite = unittest.TestSuite()
    test_name = [
        "test_constructor",
        "test_estimate",
        "test_confusion_matrix",
        "test_gaussian",
        "test_projection",
        "kaggle_exercise",
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
            f"You probably didn't pass an int to ./grader.py: passed {sys.argv[1]}; error {e}"
        )
        exit(1)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite(test_nb))
