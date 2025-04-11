import sys
import time
import numpy as np

from TD.dataset import Dataset
from TD.knn_classification import KnnClassification
from TD.confusion_matrix import ConfusionMatrix
from TD.random_projection import RandomProjection


def how_to_use() -> str:
    print(
        "Usage: test_random_projection.py <k> <projection_dim> <train_file> <test_file> <sampling> [ <column_for_classification> ]"
    )


def run_test() -> None:
    if len(sys.argv) < 6:
        how_to_use()
        return

    k = int(sys.argv[1])
    if k < 1:
        print("k needs to be at least 1")
        return

    train_ds = Dataset(sys.argv[3])
    test_ds = Dataset(sys.argv[4])

    projection_dimension = int(sys.argv[2])
    if projection_dimension < 1 or projection_dimension >= train_ds.dim - 1:
        print("The projection dimension is either less than 1 or too large")
        return

    type_sample = str(sys.argv[5])
    if type_sample != "Gaussian" and type_sample != "Rademacher":
        print("The sampling type needs to be 'Gaussian' or 'Rademacher'")
        return

    col_class = 0
    print()
    if len(sys.argv) == 7:
        col_class = sys.argv[6]
    else:
        print(
            f"No column specified for classification, assuming first column of dataset ({col_class})."
        )
    print(f"Dataset with {train_ds.nsamples} samples and {train_ds.dim} dimensions.")
    assert train_ds.dim == test_ds.dim, "Different dimensions from test and train."

    print("\nCreate a random projection.")
    start_time = time.time()
    rnd_proj = RandomProjection(
        train_ds.dim - 1, col_class, projection_dimension, type_sample
    )
    duration = time.time() - start_time
    print(f"Execution time: {round(1000 * duration)} ms\n")

    print(f"Training k-NN classification on the original data.")
    start_time_train_orig = time.time()
    knn_orig = KnnClassification(k, train_ds, col_class)
    duration_train_orig = time.time() - start_time_train_orig
    print(f"Execution time: {round(1000 * duration_train_orig)} ms\n")

    print(f"Training k-NN classification on the projected data.")
    start_time_train_proj = time.time()
    proj_ds = rnd_proj.project(train_ds)
    knn_proj = KnnClassification(k, proj_ds, projection_dimension)
    duration_train_proj = time.time() - start_time_train_proj
    print(f"Execution time: {round(1000 * duration_train_proj)} ms\n")

    print("Predicting k-NN on original data.")
    cm = ConfusionMatrix()
    start_time_test_orig = time.time()
    for i in range(test_ds.nsamples):
        sample = test_ds.instances[i]
        query = np.delete(sample, [col_class], axis=0)
        predicted_label = knn_orig.estimate(query)
        true_label = int(sample[col_class])
        cm.add_prediction(true_label, predicted_label)
    duration_test_orig = time.time() - start_time_test_orig
    print(f"Execution time: {round(1000 * duration_test_orig)} ms\n")
    cm.print_evaluation()

    print("\n\nPredicting k-NN on projected data.")
    proj_cm = ConfusionMatrix()
    proj_test_ds = rnd_proj.project(test_ds)
    start_time_test_proj = time.time()
    for i in range(proj_test_ds.nsamples):
        sample = proj_test_ds.instances[i]
        query = np.delete(sample, [projection_dimension], axis=0)
        predicted_label = knn_proj.estimate(query)
        true_label = int(sample[projection_dimension])
        proj_cm.add_prediction(true_label, predicted_label)
    duration_test_proj = time.time() - start_time_test_proj
    print(f"Execution time: {round(1000 * duration_test_proj)} ms\n")
    proj_cm.print_evaluation()

    print(
        f"\n\nSpeed-up on training: {round(duration_train_orig / duration_train_proj)}"
    )
    print(f"Speed-up on testing: {round(duration_test_orig / duration_test_proj)}")


if __name__ == "__main__":
    run_test()
