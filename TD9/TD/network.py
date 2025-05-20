#! /usr/bin/env -S python3 
"""
Networks module
"""
import random
from typing import List, Self
from functools import cached_property
import numpy as np
import matplotlib.pyplot as plt

from TD import activations
from TD.activations import Activation
from TD.losses import Loss, CrossEntropy
from TD.mnist import get_raw_data


class Network:
    """
    A simple implementation of a feed-forward artificial neural network.
    Work on this class throughout the entire exercise, rerunning this cell after each update.
    """
    def __init__(self, sizes: List[int], activations: List[Activation]):
        """
        Construct a network given a list of the number of neurons in each layer and
        a list of activations which will be applied to each layer.

        :param list sizes: A list of integers representing the number of nodes in each layer,
        including the input and output layers.
        :param list activations: A list of callable objects representing the activation functions.
        Its size should be one less than sizes.
        """
        self.sizes = sizes
        self.sigmas = activations
        self.loss = []

        self.biases = None  # TODO Exercise
        self.weights = None  # TODO Exercise
        pass

    @cached_property
    def num_params(self) -> int:
        """Returns the total number of trainable parameters in this network."""
        pass

    def __len__(self):
        """Magical method called when using len(obj)"""
        return len(self.sizes)

    def feed_forward(self, x: np.ndarray) -> (np.ndarray, list, list):
        """
        Evaluates the network for the given input and returns the intermediary results.

        :param numpy.ndarray x: A numpy 2D-array where the columns represent input variables
            and rows represent independent samples.
        :return: output of the network
        :rtype: np.ndarray, list, list
        """
        activations = [x]
        zs = []

        # Forward pass, storing z and a as we go
        pass
        return activations[-1], activations, zs

    def backpropagation(self, x: np.ndarray, y: np.ndarray, loss: Loss):
        """
        Compute gradients of loss function w.r.t network parameters **possibly for multiple samples**.

        :param numpy.ndarray x: a single input in a 2D array (1, #features), or possibly
            several inputs in a 2D array (#samples, #features)
        :param numpy.ndarray y: a single output in a 2D array (1, #targets), or possibly
            several outputs in a 2D array (#samples, #targets)
        :param Loss loss: a cost class
        :return: list of gradients of weights, list of gradients of biases
        :rtype: list, list
        """
        _, list_a, list_z = self.feed_forward(x)
        L = len(self)
        w_grad, b_grad = [None] * (L - 1), [None] * (L - 1)
        pass
        return w_grad, b_grad

    def update_step(self, x_train: np.ndarray, y_train: np.ndarray, loss: Loss, learning_rate: float):
        """
        Compute the **average** parameter gradients over the whole training set and do a GD step.
        N.B.: this does not return anything! This should update weights and biases attributes.

        :param numpy.ndarray x_train: A design matrix (2D array (#samples, #features))
        :param numpy.ndarray y_train: A vector / array of responses
        :param Loss loss: A callable object with signature loss(y_hat, y) for computing the cost of a single
            training example.
        :param float learning_rate: The learning rate to use in the GD step.
        """
        w_grad, b_grad = self.backpropagation(x_train, y_train, loss)
        pass

    def _fit_mini_batch(self, x_train, y_train, loss, learning_rate, batch_size: int = 1):
        losses = []
        for batch in range(0, x_train.shape[0], batch_size):
            x_train_batch, y_train_batch = x_train[batch : batch + batch_size], y_train[batch : batch + batch_size]
            # Update the weights and biases
            self.update_step(x_train, y_train, loss, learning_rate)
            # Compute the new prediction on the batch
            y_hat_batch, _, _ = self.feed_forward(x_train)
            # Compute the loss on the batch
            losses.append(loss(y_hat_batch, y_train_batch))
        # Sum the batches' losses to obtain the epoch's loss
        self.loss.append(sum(losses))

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, loss: Loss, learning_rate: float,
              epochs: int = 100, verbose: bool = True):
        """
        Trains this network on the training data, i.e. iterate the update step.
        N.B.: this does not return anything! This should

        :param np.array x_train: A design matrix
        :param np.array y_train: A vector of responses
        :param Loss loss: A callable object with signature cost(y_hat, y) for computing the cost of a single
        training example.
        :param float learning_rate: The learning rate to use in the GD step.
        :param int epochs: The number of epochs to train with.
        :param bool verbose: Whether to print the loss at each epoch.
        """
        assert not self.loss, "Loss not empty, network was presumably already trained."
        for i in range(epochs):
            # N.B. you can play around with batch_size
            # x_train.shape[0] means average all gradients over the whole training set
            # This might not be feasible with lots of data, hence the use of mini batches, typically of sizes
            # 2^5, ..., 2^10, which acts kind of a regularization parameter (a tradeoff between more weights' and
            # biases' updates with high variance and one per epoch with high bias).
            self._fit_mini_batch(x_train, y_train, loss, learning_rate, batch_size=x_train.shape[0])
            if verbose:
                print('epoch:', i, '\n\tloss: ', self.loss[-1])
        # Similar to most APIs, e.g. sklearn, we return self, which is important for get_xaggle_champion to work
        return self

    def plot(self):
        assert self.loss, "Loss empty, network needs to be trained first."
        plt.plot(range(len(self.loss)), self.loss, label="Training loss")
        plt.legend()
        plt.show()

    @staticmethod
    def hyperparameters(input_size):
        """
        You may modify the number of iterations, learning rate, sizes, and activations arguments to MLP
        Once you have satisfying values, make sure to copy this function to
        """
        n_iterations = 10
        lr = 0.00001
        sizes = [input_size, 10]
        act = [activations.Sigmoid]
        pass
        return n_iterations, lr, sizes, act

    @classmethod
    def get_xaggle_champion(cls, verbose: bool = True) -> Self:
        """Change anything you want, but return a trained Network instance"""
        # Get the training data
        x_train, y_train, _, _ = get_raw_data()
        # Get your hyperparameters
        n_iterations, lr, sizes, activations = cls.hyperparameters(input_size=x_train.shape[1])
        return cls(sizes, activations).fit(x_train, y_train, CrossEntropy, lr, n_iterations, verbose=verbose)


def sine():
    from TD.activations import Identity, Sigmoid
    from TD.losses import Quadratic

    # Generate sinusoidal training points
    np.random.seed(0)
    x = np.random.uniform(-1, 1, (100, 1))
    y = np.cos(np.pi * x) ** 2

    # Hyperparameters to tune
    n_iterations = 10000
    lr = 0.0015
    verbose = False
    sizes = [1, 32, 16, 1]
    activations = [Sigmoid, Sigmoid, Identity]

    # Fit the network
    net = Network(sizes, activations)
    net.fit(x_train=x, y_train=y, loss=Quadratic, learning_rate=lr, epochs=n_iterations, verbose=verbose)

    # Some test samples
    x_pred = np.arange(-1, 1, 0.01).reshape(200, 1)
    y_pred, _, _ = net.feed_forward(x_pred)

    # Plot train samples, truth, test samples and predictions
    # Watch the Universal Approximation theorem unfold before your eyes
    plt.plot(x, y, 'k.', label="Truth")
    plt.plot(x_pred, y_pred, 'k-', label="Prediction")
    plt.legend()
    plt.show()
    net.plot()
    plt.plot(range(int(n_iterations / 2), n_iterations),
             net.loss[int(n_iterations / 2):],
             label=f"Training loss, last {int(n_iterations / 2)} observations")
    plt.legend()
    plt.show()


def xaggle():
    """
    Go back and forth in `hyperparameters` to optimize for validation accuracy...
    ... and hopefully achieve a great private test accuracy.
    """
    your_champion = Network.get_xaggle_champion()
    # Get the validation data
    _, _, x_val, y_val = get_raw_data()
    # Predict
    y_pred, _, _ = your_champion.feed_forward(x_val)
    # Compute accuracy
    accuracy = np.sum(np.equal(np.argmax(y_pred, axis=1), np.argmax(y_val, axis=1))) / len(y_val)
    print("Accuracy:", accuracy)
    # Plot loss and a random validation image
    your_champion.plot()
    image = random.choice(range(len(x_val)))
    plt.imshow(x_val[image, :].reshape(28, 28), cmap='gray')
    plt.show()
    print("The probabilities for each class for a random validation point:\n", y_pred[image].reshape(-1, 1))
    print(f"The actual class of the first validation point ({np.where(y_val[image])[0][0]}):\n", y_val[image].reshape(-1, 1))
