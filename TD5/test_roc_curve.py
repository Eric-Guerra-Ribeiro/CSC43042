import sys
import numpy as np
import matplotlib.pyplot as plt

from TD.dataset import Dataset
from TD.knn_classification import KnnClassification
from TD.confusion_matrix import ConfusionMatrix


def how_to_use() -> str:
    print(
        "Usage: test_roc_curve.py <k> <train_file> <test_file> <num_points> [ <column_for_classification> ]"
    )


def create_roc_plot() -> None:
    if len(sys.argv) < 4:
        how_to_use()
        return

    k = int(sys.argv[1])
    if k < 1:
        print("k needs to be at least 1")
        return

    train_ds = Dataset(sys.argv[2])
    test_ds = Dataset(sys.argv[3])

    num_points = int(sys.argv[4])
    if num_points < 0:
        print("The number of points needs to be a non-negative integer.")
        return

    col_class = 0
    print()
    if len(sys.argv) == 6:
        col_class = int(sys.argv[5])

    assert train_ds.dim == test_ds.dim, "Different dimensions from test and train."

    thresholds = [x / (num_points + 1.0) for x in range(1, num_points + 1)]
    conf_matrices = [ConfusionMatrix() for _ in range(len(thresholds))]

    # Add the predictions here.


if __name__ == "__main__":
    create_roc_plot()
