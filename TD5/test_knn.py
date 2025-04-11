import sys
import time
import numpy as np

from TD.dataset import Dataset
from TD.knn_classification import KnnClassification
from TD.confusion_matrix import ConfusionMatrix


def how_to_use() -> str:
    print(
        "Usage: test_knn.py <k> <train_file> <test_file> [ <column_for_classification> ]"
    )


def run_test() -> None:
    if len(sys.argv) < 4:
        how_to_use()
        return

    k = int(sys.argv[1])
    if k < 1:
        print("k needs to be at least 1")
        return

    train_ds = Dataset(sys.argv[2])
    test_ds = Dataset(sys.argv[3])

    col_class = 0
    print()
    if len(sys.argv) == 5:
        col_class = int(sys.argv[4])
    else:
        print(
            f"No column specified for classification, assuming first column of dataset ({col_class})."
        )
    print(f"Dataset with {train_ds.nsamples} samples and {train_ds.dim} dimensions.")
    assert train_ds.dim == test_ds.dim, "Different dimensions from test and train."

    print(
        f"Computing k-NN classification (k = {k}, classification over column {col_class}) ..."
    )
    knn = KnnClassification(k, train_ds, col_class)
    cm = ConfusionMatrix()

    print("Prediction and Confusion Matrix filling")
    start_time = time.time()
    for i in range(test_ds.nsamples):
        sample = test_ds.instances[i]
        query = np.delete(sample, [col_class], axis=0)
        predicted_label = knn.estimate(query)
        true_label = int(sample[col_class])
        cm.add_prediction(true_label, predicted_label)

    duration = time.time() - start_time
    print(f"\nExecution time: {round(1000 * duration)} ms\n")
    cm.print_evaluation()


if __name__ == "__main__":
    run_test()
