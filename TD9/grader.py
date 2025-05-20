#! /usr/bin/env python3 
import sys
import unittest
from pathlib import Path

import numpy as np

from TD import activations, losses, network
from TD.mnist import get_raw_data

"""
Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 7,
  "names" : [
      "activations.py::activations", 
      "network.py::network_init",
      "network.py::network_feed_forward",
      "losses.py::losses", 
      "network.py::network_backpropagation",
      "network.py::network_update_step",
      "network.py::xaggle"
      ],
  "points" : [20, 20, 10, 10, 15, 15, 10]
}
[END-AUTOGRADER-ANNOTATION]
"""


def print_help():
    print("./grader script. Usage: ./grader.py test_number, e.g. ./grader.py 1 for 1st exercise.")
    print("N.B.: ./grader.py 0 will run all tests.}")
    print(f"You provided {sys.argv}")
    exit(1)


def test_identity():
    assert activations.Identity(np.array([1])) == np.array([1])
    np.testing.assert_array_equal(activations.Identity(np.array([2, 2])), np.array([2, 2]))
    np.testing.assert_array_equal(activations.Identity.prime(np.array([2, 2])), np.array([1, 1]))


def test_sigmoid():
    assert activations.Sigmoid(np.array([0])) == np.array([0.5])
    np.testing.assert_array_equal(activations.Sigmoid(np.array([0, 0])), np.array([0.5, 0.5]))
    np.testing.assert_array_almost_equal(activations.Sigmoid(np.array([1, 1])), np.array([0.73105858, 0.73105858]))
    np.testing.assert_array_almost_equal(activations.Sigmoid.prime(np.array([2, 2])), np.array([0.104994, 0.104994]))


def test_quadratic():
    assert losses.Quadratic(np.array([0]), np.array([0])) == 0
    assert losses.Quadratic.prime(np.array([0]), np.array([0])) == 0
    assert losses.Quadratic(np.array([0, 0]), np.array([0, 0])) == 0
    assert losses.Quadratic(np.array([1]), np.array([0])) == 1
    assert losses.Quadratic.prime(np.array([1]), np.array([0])) == 2
    assert losses.Quadratic(np.array([1, 0]), np.array([0, 0])) == 1
    assert np.all(losses.Quadratic.prime(np.array([1, 0]), np.array([0, 0])) == np.array([2, 0]))


def test_cross_entropy():
    assert losses.CrossEntropy(np.array([0.5]), np.array([0])) == - np.log(0.5)
    assert losses.CrossEntropy.prime(np.array([0.5]), np.array([0])) == 2.0
    assert int(losses.CrossEntropy(np.array([0.5, 0.5]), np.array([0, 1]))) == 1  # noqa
    assert losses.CrossEntropy.prime(np.array([0.5]), np.array([0])) == 2.0
    assert losses.CrossEntropy(np.array([0.7]), np.array([1])) == - np.log(0.7)
    assert losses.CrossEntropy.prime(np.array([0.7]), np.array([1])) == - 1 / 0.7
    assert np.all(losses.CrossEntropy.prime(np.array([0.5, 0.5]), np.array([1, 0])) == np.array([-2, 2]))


class Grader(unittest.TestCase):
    def activations(self):
        test_identity()
        test_sigmoid()

    def losses(self):
        test_quadratic()
        test_cross_entropy()
    
    def network_init(self):
        assert network.Network([1], [activations.Sigmoid]).sizes == [1]
        assert network.Network([1], [activations.Sigmoid]).sigmas == [activations.Sigmoid]
        assert len(network.Network([1], [activations.Sigmoid]).weights) == 0
        assert len(network.Network([1], [activations.Sigmoid]).biases) == 0
        assert network.Network(
            [1, 2, 2, 1],
            3 * [activations.Sigmoid]).num_params == 13
        assert network.Network(
            [1, 3, 3, 1],
            3 * [activations.Sigmoid]).num_params == 22
        assert network.Network(
            [4, 2, 5, 1],
            3 * [activations.Sigmoid]).num_params == 31
        net = network.Network([4, 5, 5, 1], 3 * [activations.Sigmoid])
        assert net.weights[0].shape == (4, 5)
        assert net.weights[1].shape == (5, 5)
        assert net.weights[2].shape == (5, 1)
        assert net.biases[0].shape == (1, 5)
        assert net.biases[1].shape == (1, 5)
        assert net.biases[2].shape == (1, 1)

    def network_feed_forward(self):
        output, _, _ = network.Network(
            [1, 2, 2, 2],
            3 * [activations.Sigmoid]).feed_forward(np.array([[1]]))
        assert output.shape[0] == 1
        assert output.shape[1] == 2

    def network_backpropagation(self):
        x = np.array([1, -1]).reshape(1, 2)
        y = np.array([0])
        net = network.Network([2, 5, 1], [activations.Sigmoid, activations.Identity])
        w_grad, b_grad = net.backpropagation(x, y, losses.Quadratic)
        assert len(w_grad) == 2  # 2 layers (1 hidden, 1 output)
        assert len(b_grad) == 2  # 2 layers (1 hidden, 1 output)
        assert w_grad[0].shape == (2, 5)
        assert w_grad[1].shape == (5, 1)
        assert b_grad[0].shape == (1, 5)
        assert b_grad[1].shape == (1, 1)

    def network_update_step(self):
        np.random.seed(0)
        x = np.random.uniform(-1, 1, (100, 1))
        y = np.cos(np.pi * x) ** 2
        net = network.Network([1, 100, 1], [activations.Sigmoid, activations.Identity])
        # Store weights and biases before the update to see if something changed
        weights_before = net.weights[0].copy()
        biases_before = net.biases[0].copy()
        weights_before_bis = net.weights[1].copy()
        biases_before_bis = net.biases[1].copy()
        net.update_step(x, y, losses.Quadratic, 0.1)
        # Assert that something did change
        assert np.any(np.not_equal(net.weights[0], weights_before))
        assert np.any(np.not_equal(net.biases[0], biases_before))
        assert np.any(np.not_equal(net.weights[1], weights_before_bis))
        assert np.any(np.not_equal(net.biases[1], biases_before_bis))

    def xaggle(self):
        your_champion = network.Network.get_xaggle_champion(verbose=False)
        # Get the test data
        x_test, y_test = get_private_test_data()
        # Predict
        y_pred, _, _ = your_champion.feed_forward(x_test)
        # Compute accuracy
        accuracy = np.sum(np.equal(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1))) / len(y_test)
        print(f"Test accuracy: {accuracy}")
        if accuracy > 0.65:
            print("A+")
        elif accuracy > 0.6:
            print("A")
        elif accuracy > 0.4:
            print("B")
        elif accuracy > 0.2:
            print("C")
        else:
            print("D")  # Not (much) better than random


def get_private_test_data():
    x_test, y_test, _, _ = get_raw_data("test")
    try:
        with open(Path(__file__).parent / "seed", "r") as f:
            private_seed = int(f.read())
    except Exception as e:
        print("This only works on server grader")
        raise e
    private_test_indices = np.random.default_rng(private_seed).choice(range(len(x_test)), size=1000, replace=False, shuffle=False)
    return x_test[private_test_indices, :], y_test[private_test_indices, :]


def suite(test_nb):
    suite = unittest.TestSuite()
    test_name = [
        "activations",
        "network_init",
        "network_feed_forward",
        "losses",
        "network_backpropagation",
        "network_update_step",
        "xaggle",
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
        print(f"You probably didn't pass an int to ./grader.py: passed {sys.argv[1]}; error {e}")
        exit(1)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite(test_nb))
