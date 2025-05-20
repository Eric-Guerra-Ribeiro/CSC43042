from __future__ import annotations
from typing import Tuple
from multiprocessing import Pool

import numpy as np

class DecisionTree:

    def __init__(self: DecisionTree, max_depth:int):
        self.max_depth: int = max_depth
        self.feature_index: int = None
        self.threshold: float = None
        self.value: int = None
        self.left: DecisionTree = None
        self.right: DecisionTree = None

    @staticmethod
    def entropy(y:np.array)->float:
        _, counts = np.unique(y, return_counts=True)
        proportions = counts/y.size
        return -np.sum(proportions*np.log2(proportions))
    
    @staticmethod
    def best_split(X:np.ndarray, y:np.ndarray, feature_index: int) -> Tuple[float, float]:

        def split_entropy_calc(i):
            return -(i*DecisionTree.entropy(y_sorted[:i]) + (length - i)*DecisionTree.entropy(y_sorted[i:]))/length

        length = y.size

        if length == 1:
            return 0, X[0, feature_index]

        x_uniques, x_count = np.unique(X[:, feature_index], return_counts=True)

        i_list = np.cumsum(x_count)

        idx_sorted = np.argsort(X[:,feature_index])
        y_sorted = y[idx_sorted]

        total_entropy = DecisionTree.entropy(y_sorted)
        return max(
            (
                (total_entropy + split_entropy_calc(i), threshold)
                for i, threshold in zip(i_list, x_uniques)
            ),
            key= lambda x: x[0]
        )

    @staticmethod
    def best_split_pair(X:np.ndarray, y:np.ndarray) -> Tuple[float, int, float]:
        return max(
            (
                (gain, feature_index, threshold) for feature_index in range(X.shape[1])
                for gain, threshold in [DecisionTree.best_split(X, y, feature_index)]
            ),
            key=lambda x: x[0]
        )
    
    def fit(self:DecisionTree, X:np.ndarray, y:np.ndarray, depth:int=0)->None:
        values, count = np.unique(y, return_counts=True)
        if depth == self.max_depth or values.size == 1:
            self.value = values[np.argmax(count)]
            return

        _, feature_index, threshold = DecisionTree.best_split_pair(X, y)
        self.feature_index = feature_index
        self.threshold = threshold

        feature_column = X[:, feature_index]

        right_idx = feature_column > threshold
        left_idx = feature_column <= threshold

        self.right = DecisionTree(self.max_depth)
        self.right.fit(X[right_idx, :], y[right_idx], depth + 1)

        self.left = DecisionTree(self.max_depth)
        self.left.fit(X[left_idx, :], y[left_idx], depth + 1)

    def predict(self:DecisionTree, X:np.ndarray) -> int:
        if self.value is not None:
            return self.value
    
        if X[self.feature_index] > self.threshold:
            return self.right.predict(X)
    
        return self.left.predict(X)


def best_split(split_data):
    return


class RandomForest:
    """
    Attributes:
    - trees: np.array
    """

    NUM_PROCESSORS = 16
    USE_MULTIPROCESSING = True

    def __init__(self: RandomForest, nbtrees:int=1, max_depth:int=1):
        self.nbtrees = nbtrees
        self.trees = np.array([DecisionTree(max_depth) for _ in range(nbtrees)])

    def fit(self: RandomForest, X: np.array, y: np.array, ratio=0.3) -> None:
        """Build the decision trees in the `trees` array,
        each using a proportion `ratio` of the data.
        """
        data_size = int(ratio*y.size)
        if RandomForest.USE_MULTIPROCESSING:
            with Pool() as pool:
                self.trees = pool.map(
                    fit_tree,
                    [(tree, X, y, ratio) for tree in self.trees]
                )
        else:
            for tree in self.trees:
                sample_idx = np.random.choice(y.size, size=data_size, replace=False)
                tree.fit(X[sample_idx, :], y[sample_idx])
    
    def predict(self:RandomForest, X:np.ndarray) -> int:
        predictions = np.fromiter((tree.predict(X) for tree in self.trees), dtype=int)
        values, count = np.unique(predictions, return_counts=True)
        return values[np.argmax(count)]

    @staticmethod
    def my_MNIST_Fashion_parameters()->Tuple[int, int, float]:
        nb_trees, max_depth, ratio = 16, 50, 0.25
        return (nb_trees, max_depth, ratio)


def predict_tree(tree_data):
    return DecisionTree.predict(*tree_data)


def fit_tree(tree_data):
    tree, X, y, ratio = tree_data
    data_size = int(ratio * y.size)
    sample_idx = np.random.choice(y.size, size=data_size, replace=False)
    tree.fit(X[sample_idx], y[sample_idx])
    return tree
