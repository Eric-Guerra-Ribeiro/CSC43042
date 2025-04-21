#! /usr/bin/env python3
import unittest
import numpy as np
import sys

from TD.dataset import Dataset
from TD.linear_regression import LinearRegression
from TD.knn_regression import KnnRegression

"""
Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 7,
  "names" : [
      "linear_regression.py::test_construct_Linear_regression",
      "linear_regression.py::test_construct_y_Linear_regression",
      "linear_regression.py::test_coefficients_linear_regression",
      "linear_regression.py::test_estimate_linear_regression",
      "linear_regression.py::test_sum_squares_linear_regression",
      "knn_regression.py::test_knn_kdtree",
      "knn_regression.py::test_estimate_knn"
      ],
  "points" : [10, 15, 15, 10, 10, 20, 20]
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

# Expected norm values for X^T * y for each regression column in the Boston dataset
Boston_XTy = [
    590180.829446,
    2840723.51761,
    2156285.03048,
    19039.9221446,
    115274.278109,
    1337892.85004,
    14001845.827,
    853578.915551,
    1673479.21376,
    55239953.4765,
    3854910.41925,
    55264613.4912,
    2525971.40407,
    5019095.96678,
]

# Expected beta coefficients for Linear Regression on the Boston dataset
Boston = [
    [
        -1.43778,
        0.0173683,
        0.0334321,
        -1.43041,
        -6.76887,
        2.05081,
        -0.022268,
        -0.531224,
        0.608631,
        0.000705713,
        -0.217417,
        -0.00288819,
        0.19694,
        -0.180487,
    ],
    [
        -8.42039,
        0.230507,
        -0.349575,
        -0.76017,
        -8.7658,
        2.78919,
        -0.133201,
        6.51874,
        -0.601777,
        0.0728535,
        -2.58498,
        -0.00013384,
        0.516997,
        0.542295,
    ],
    [
        -6.31744,
        0.0210511,
        -0.0165854,
        1.55026,
        21.8612,
        -0.830543,
        -0.00588946,
        -0.661229,
        -0.325042,
        0.0210537,
        0.365561,
        -0.00272097,
        0.100265,
        0.0304447,
    ],
    [
        -0.118513,
        -0.00483676,
        -0.000193678,
        0.00832503,
        0.187571,
        0.0088827,
        0.000602191,
        0.00711229,
        0.0124081,
        -0.000420896,
        -0.0071473,
        -0.000112789,
        0.000573135,
        0.00592023,
    ],
    [
        0.827624,
        -0.000859078,
        -8.38266e-05,
        0.00440634,
        0.00704024,
        -0.000141677,
        0.000798438,
        -0.0151188,
        0.00326109,
        0.000100344,
        -0.0122335,
        -0.000310831,
        5.49852e-05,
        -0.00175903,
    ],
    [
        5.10504,
        0.0192189,
        0.00196949,
        -0.0123609,
        0.0246179,
        -0.0104613,
        0.00526083,
        -0.00151918,
        -0.0110603,
        3.47998e-05,
        -0.00967672,
        0.00150729,
        -0.0338487,
        0.0413719,
    ],
    [
        -43.2548,
        -0.294112,
        -0.13256,
        -0.123537,
        2.35219,
        83.0916,
        7.41456,
        -4.16143,
        -0.300658,
        0.0142501,
        0.921566,
        -0.00116721,
        1.52991,
        0.032128,
    ],
    [
        11.4053,
        -0.0293058,
        0.0270965,
        -0.0579315,
        0.116035,
        -6.57167,
        -0.00894299,
        -0.0173815,
        0.0070763,
        -2.69616e-05,
        0.00235338,
        -0.00147648,
        -0.0229319,
        -0.0665445,
    ],
    [
        -22.2919,
        0.203405,
        -0.0151536,
        -0.172518,
        1.22636,
        8.58722,
        -0.394431,
        -0.00760759,
        0.0428684,
        0.037719,
        0.405174,
        0.01018,
        0.0658516,
        0.142617,
    ],
    [
        231.944,
        0.0937454,
        0.729198,
        4.44159,
        -16.5349,
        105.026,
        0.493283,
        0.14332,
        -0.0649218,
        14.9925,
        0.407642,
        -0.1126,
        -1.61171,
        -1.91816,
    ],
    [
        25.0653,
        -0.0276685,
        -0.024787,
        0.0738822,
        -0.268992,
        -12.2666,
        -0.131407,
        0.00887946,
        0.00542887,
        0.154286,
        0.000390526,
        0.00258834,
        -0.0357258,
        -0.095727,
    ],
    [
        386.959,
        -0.18408,
        -0.000642746,
        -0.275418,
        -2.12595,
        -156.095,
        10.2512,
        -0.00563243,
        -1.70581,
        1.94142,
        -0.0540253,
        1.29631,
        0.639737,
        -0.128924,
    ],
    [
        31.0254,
        0.126716,
        0.0250646,
        0.102456,
        0.109059,
        0.278758,
        -2.32401,
        0.07453,
        -0.267462,
        0.126782,
        -0.00780663,
        -0.180629,
        0.00645831,
        -0.317104,
    ],
    [
        30.1835,
        -0.194652,
        0.0440677,
        0.0521448,
        1.88823,
        -14.9475,
        4.76119,
        0.00262339,
        -1.30091,
        0.46023,
        -0.0155731,
        -0.811248,
        -0.00218155,
        -0.531514,
    ],
]

# Expected beta coefficients for the Wine datasets are defined similarly
RedWine = [
    [
        -660.373,
        -0.296525,
        1.39186,
        -0.238212,
        -4.45294,
        0.00673006,
        -0.0076202,
        685.417,
        -5.49347,
        -0.600104,
        0.474934,
        0.0309372,
    ],
    [
        -33.884,
        -0.0138812,
        -0.560518,
        -0.0139507,
        0.441621,
        -0.00172112,
        0.000806976,
        35.2681,
        -0.194332,
        -0.163726,
        0.0498344,
        -0.0437634,
    ],
    [
        -8.54648,
        0.0521218,
        -0.448383,
        -0.00434468,
        0.731694,
        -0.00254118,
        0.00153703,
        8.67994,
        -0.138677,
        -0.0489741,
        0.0420753,
        -0.0095979,
    ],
    [
        -845.654,
        -0.659516,
        -0.82507,
        -0.321214,
        -2.02094,
        0.00924564,
        0.00325395,
        862.06,
        -3.59827,
        -0.978333,
        0.679856,
        0.0318215,
    ],
    [
        -9.29178,
        -0.0235173,
        0.0498226,
        0.103192,
        -0.00385509,
        0.000692777,
        -0.000436164,
        10.0651,
        -0.156542,
        0.10312,
        0.00019447,
        -0.00772977,
    ],
    [
        -395.822,
        0.78131,
        -4.26826,
        -7.87801,
        0.387687,
        15.2285,
        0.234759,
        353.028,
        10.4742,
        -0.488434,
        0.407861,
        0.718078,
    ],
    [
        193.745,
        -8.25316,
        18.6702,
        44.4543,
        1.27293,
        -89.4462,
        2.19013,
        89.5289,
        -54.4989,
        12.6418,
        -0.701476,
        -6.00554,
    ],
    [
        0.97798,
        0.000927994,
        0.00102002,
        0.000313822,
        0.000421568,
        0.00258029,
        4.11713e-06,
        1.11918e-07,
        0.00473951,
        0.00100506,
        -0.000677029,
        -1.4988e-05,
    ],
    [
        -56.8251,
        -0.0953355,
        -0.072042,
        -0.0642669,
        -0.0225549,
        -0.514395,
        0.00156576,
        -0.000873256,
        60.7506,
        -0.105931,
        0.0621358,
        -0.00326131,
    ],
    [
        -50.028,
        -0.0419544,
        -0.244514,
        -0.091431,
        -0.0247046,
        1.36506,
        -0.000294139,
        0.000816029,
        51.8983,
        -0.426744,
        0.0527067,
        0.0438874,
    ],
    [
        555.507,
        0.540446,
        1.21139,
        1.27856,
        0.279432,
        0.0419017,
        0.00399785,
        -0.000737019,
        -569.031,
        4.07431,
        0.857895,
        0.306936,
    ],
    [
        15.4472,
        0.0335424,
        -1.01358,
        -0.277886,
        0.0124616,
        -1.58686,
        0.00670627,
        -0.0060119,
        -12.0024,
        -0.203751,
        0.680616,
        0.292444,
    ],
]

WhiteWine = [
    [
        -710.944,
        -0.484234,
        0.468924,
        -0.257376,
        -4.30806,
        0.00168227,
        -0.00234046,
        727.343,
        -3.53076,
        -1.05186,
        0.799428,
        0.0543195,
    ],
    [
        -6.58946,
        -0.0185883,
        -0.127927,
        0.000752747,
        0.287754,
        -0.00117442,
        0.000432157,
        7.28862,
        -0.121039,
        -0.0224521,
        0.0313838,
        -0.0279967,
    ],
    [
        -12.9908,
        0.0309728,
        -0.220118,
        -0.000974285,
        0.493541,
        0.000585825,
        -7.61557e-05,
        13.3544,
        -0.130398,
        0.0520595,
        0.0275065,
        -0.000760758,
    ],
    [
        -2475.04,
        -1.91299,
        0.145751,
        -0.109636,
        -13.1609,
        0.020899,
        -0.00733121,
        2511.52,
        -8.53586,
        -4.19374,
        2.48714,
        0.304874,
    ],
    [
        -7.93796,
        -0.00869786,
        0.0151345,
        0.0150861,
        -0.00357495,
        9.9206e-05,
        2.16721e-06,
        8.20058,
        -0.0370071,
        -0.0106866,
        0.00243882,
        -0.000302472,
    ],
    [
        3616.31,
        1.00007,
        -18.1876,
        5.27259,
        1.67153,
        29.2106,
        0.242093,
        -3653.2,
        12.3275,
        -5.87694,
        -4.2075,
        1.37346,
    ],
    [
        -13970.1,
        -8.45229,
        40.6567,
        -4.16388,
        -3.56208,
        3.87653,
        1.47069,
        14238.1,
        -45.5499,
        31.1846,
        10.3219,
        -0.0507728,
    ],
    [
        0.986425,
        0.000764929,
        0.000199684,
        0.000212632,
        0.000355364,
        0.00427165,
        -6.46281e-06,
        4.1463e-06,
        0.00329204,
        0.00154683,
        -0.00106636,
        -9.69113e-05,
    ],
    [
        -124.084,
        -0.144269,
        -0.128839,
        -0.0806677,
        -0.0469253,
        -0.74896,
        0.000847317,
        -0.000515369,
        127.905,
        -0.0732896,
        0.142751,
        0.0187976,
    ],
    [
        -67.0824,
        -0.0485394,
        -0.0269903,
        0.0363712,
        -0.026037,
        -0.244256,
        -0.000456196,
        0.000398476,
        67.8729,
        -0.0827698,
        0.0691344,
        0.0153725,
    ],
    [
        705.784,
        0.562836,
        0.575604,
        0.293197,
        0.23559,
        0.850454,
        -0.00498302,
        0.00201229,
        -713.881,
        2.45966,
        1.05478,
        0.0339177,
    ],
    [
        238.253,
        0.141813,
        -1.90407,
        -0.0300696,
        0.107086,
        -0.391123,
        0.00603171,
        -3.67043e-05,
        -240.576,
        1.20104,
        0.869698,
        0.125772,
    ],
]


###############################################################################
# DummyDataset for testing k-NN Regression (Exercises 5 & 6)
###############################################################################
class DummyDataset:
    def __init__(self):
        # Create a simple dataset with 5 samples, 2 features, and 1 target (last column)
        self.data = np.array(
            [
                [1.0, 2.0, 10.0],
                [2.0, 1.0, 20.0],
                [1.5, 1.5, 15.0],
                [2.0, 2.0, 25.0],
                [0.5, 1.0, 5.0],
            ]
        )

    def get_nbr_samples(self):
        return self.data.shape[0]

    def get_dim(self):
        return self.data.shape[1]

    def get_instance(self, i):
        return self.data[i]


# Define a simple dummy dataset with 3 samples and 3 columns (last column is target).
class DummyDataset2:
    def __init__(self):
        self.data = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]])

    def get_nbr_samples(self):
        return self.data.shape[0]

    def get_dim(self):
        return self.data.shape[1]

    def get_instance(self, i):
        return self.data[i]


###############################################################################
# Grader class with tests for Exercises 1 to 6
###############################################################################
class Grader(unittest.TestCase):
    def test_construct_linear_regression(self):
        ds = DummyDataset2()
        n_samples = ds.get_nbr_samples()
        d = ds.get_dim()

        # --- Test with intercept ---
        lr_with_intercept = LinearRegression(ds, col_regr=2, fit_intercept=True)
        X_int = lr_with_intercept.construct_matrix()

        
        self.assertEqual(
            X_int.shape,
            (n_samples, d),
            "With intercept, expected matrix shape to be (n_samples, d)",
        )

        
        self.assertTrue(
            np.allclose(X_int[:, 0], np.ones(n_samples)),
            "With intercept, the first column of X should be all ones",
        )

        
        for i in range(n_samples):
            instance = ds.get_instance(i)
            expected_features = [instance[j] for j in range(d) if j != 2]  # exclude target
            np.testing.assert_allclose(
                X_int[i, 1:], expected_features,
                err_msg=f"Mismatch in features at row {i} with intercept"
            )

        # --- Test without intercept ---
        lr_no_intercept = LinearRegression(ds, col_regr=2, fit_intercept=False)
        X_no_int = lr_no_intercept.construct_matrix()

        
        self.assertEqual(
            X_no_int.shape,
            (n_samples, d - 1),
            "Without intercept, expected matrix shape to be (n_samples, d-1)",
        )

        
        for i in range(n_samples):
            instance = ds.get_instance(i)
            expected_features = [instance[j] for j in range(d) if j != 2]  # exclude target
            np.testing.assert_allclose(
                X_no_int[i, :], expected_features,
                err_msg=f"Mismatch in features at row {i} without intercept"
            )


    def test_construct_y_linear_regression(self):
        """Exercise 1: Test construction of X and y for Linear Regression."""
        # @OFF
        fname = "private_set_1.csv"
        # @ON
        fname = "./csv/train_boston_housing.csv"
        ds = Dataset(fname)
        n_samples = ds.get_nbr_samples()
        dim = ds.get_dim()

        for col in range(len(Boston_XTy)):
            with self.subTest(column=col):
                lr = LinearRegression(ds, col)
                X = lr.construct_matrix()
                y = lr.construct_y()
                # Check intercept term
                self.assertAlmostEqual(
                    X[0, 0], 1.0, msg=f"Column {col}: X[0,0] should be 1.0 (intercept)"
                )
                # Check dimensions
                self.assertEqual(
                    X.shape[0],
                    n_samples,
                    msg=f"Column {col}: Incorrect number of rows in X",
                )
                self.assertEqual(
                    X.shape[1],
                    dim,
                    msg=f"Column {col}: Incorrect number of columns in X",
                )
                self.assertEqual(
                    y.size, n_samples, msg=f"Column {col}: Incorrect size of y"
                )
                # Check norm of X^T * y
                n1 = np.linalg.norm(X.T @ y) if X.size and y.size else 0.0
                self.assertAlmostEqual(
                    n1,
                    Boston_XTy[col],
                    delta=deps,
                    msg=f"Column {col}: Norm of X^T*y differs",
                )

    def test_coefficients_linear_regression(self):
        """Exercise 2: Test computed beta coefficients for Linear Regression."""
        # @OFF
        test_cases = [
            ("private_set_1.csv", Boston),
            ("private_set_2.csv", RedWine),
            ("private_set_3.csv", WhiteWine),
        ]
        # @ON
        test_cases = [
            ("./csv/train_boston_housing.csv", Boston),
            ("./csv/train_winequality-red.csv", RedWine),
            ("./csv/train_winequality-white.csv", WhiteWine),
        ]

        for fname, expected in test_cases:
            with self.subTest(dataset=fname):
                ds = Dataset(fname)
                for col in range(len(expected)):
                    with self.subTest(column=col):
                        lr = LinearRegression(ds, col)
                        beta_true = np.array(expected[col])
                        beta_computed = lr.get_coefficients()
                        RE = np.linalg.norm(beta_true - beta_computed) / np.linalg.norm(
                            beta_true
                        )
                        self.assertLessEqual(
                            RE,
                            deps,
                            msg=f"Dataset {fname}, Column {col}: Relative error {RE} exceeds tolerance {deps}",
                        )

    def test_estimate_linear_regression(self):
        """Exercise 3: Test the estimate function of Linear Regression."""
        # @OFF
        fname = "private_set_4.csv"
        # @ON
        fname = "./csv/train_boston_housing.csv"
        ds = Dataset(fname)
        # For each regression column (target), test predictions for every sample.
        for col in range(ds.get_dim() - 1):
            with self.subTest(column=col):
                lr = LinearRegression(ds, col)
                beta = lr.get_coefficients()
                n_samples = ds.get_nbr_samples()
                for i in range(n_samples):
                    with self.subTest(sample=i):
                        instance = ds.get_instance(i)
                        # Extract features by removing the target column (m_col_regr)
                        features = np.delete(instance, lr.m_col_regr)
                        # Expected prediction computed manually
                        if lr.fit_intercept:
                            expected_est = beta[0] + features @ beta[1:]
                        else:
                            expected_est = features @ beta
                        computed_est = lr.estimate(features)
                        self.assertAlmostEqual(
                            computed_est,
                            expected_est,
                            delta=deps,
                            msg=f"Sample {i}, Column {col}: estimate mismatch",
                        )

    def test_sum_squares_linear_regression(self):
        """Exercise 4: Test computation of TSS, ESS, and RSS in Linear Regression."""
        # @OFF
        fname = "private_set_4.csv"
        # @ON
        fname = "./csv/train_boston_housing.csv"
        ds = Dataset(fname)
        lr = LinearRegression(ds, 0)  # test on the first target column
        tss, ess, rss = lr.sum_of_squares(ds)
        self.assertAlmostEqual(tss, ess + rss, delta=deps, msg="TSS != ESS + RSS")

    def test_knn_kdtree(self):
        """Exercise 5: Test the constructor of KnnRegression (KDTree construction)."""
        dummy_ds = DummyDataset()
        # Use target column index 2 for the dummy dataset
        knn = KnnRegression(3, dummy_ds, 2)
        # Expected feature matrix: all columns except the target
        expected_features = np.delete(dummy_ds.data, 2, axis=1)
        kd_data = knn.get_kdTree().data
        self.assertTrue(
            np.allclose(kd_data, expected_features),
            msg="KDTree data does not match expected features",
        )

    def test_estimate_knn(self):
        """Exercise 6: Test the estimate function of KnnRegression."""
        # @OFF
        filename = "private_set_5.csv"
        # @ON
        filename = "./csv/dummy_data.csv"  # Fixed file name.
        dummy_ds = DummyDataset()
        # For simplicity, use k=1 so the nearest neighbor of a point is itself.
        knn = KnnRegression(1, dummy_ds, 2)
        n_samples = dummy_ds.get_nbr_samples()
        for i in range(n_samples):
            with self.subTest(sample=i):
                instance = dummy_ds.get_instance(i)
                features = np.delete(instance, 2)
                # For k=1, the estimated target should equal the target of the sample itself.
                estimated = knn.estimate(features)
                expected = instance[2]
                self.assertAlmostEqual(
                    estimated,
                    expected,
                    delta=deps,
                    msg=f"Sample {i}: knn estimate mismatch",
                )

        ds = Dataset(filename)

        # For k=1, the nearest neighbor should be the sample itself.
        knn = KnnRegression(1, ds, 2)
        n_samples = ds.get_nbr_samples()
        for i in range(n_samples):
            with self.subTest(sample=i):
                instance = ds.get_instance(i)
                # Remove the target column (index 2) to get the feature vector.
                features = np.delete(instance, 2)
                expected = instance[2]
                estimated = knn.estimate(features)
                self.assertAlmostEqual(
                    estimated,
                    expected,
                    delta=deps,
                    msg=f"Sample {i}: knn estimate mismatch (expected {expected}, got {estimated})",
                )


def suite(test_nb):
    suite = unittest.TestSuite()
    test_name = [
        "test_construct_linear_regression",
        "test_construct_y_linear_regression",
        "test_coefficients_linear_regression",
        "test_estimate_linear_regression",
        "test_sum_squares_linear_regression",
        "test_knn_kdtree",
        "test_estimate_knn",
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
