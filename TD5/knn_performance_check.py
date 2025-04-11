import numpy as np
import time

from TD.dataset import Dataset
from TD.knn_classification import KnnClassification
from TD.confusion_matrix import ConfusionMatrix
from TD.random_projection import RandomProjection
from TD.knn_application import *


def main():
    # Read data
    train_ds = Dataset("../csv/samples_train.csv")
    test_ds = Dataset("../csv/samples_test_excerpt.csv")
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

    # Print results
    print(f"Time unprojected: {duration_test_orig} s")
    print(f"Time projected:   {duration_test_proj} s")
    print(f"Speed-up:         {speed_up}\n")
    print(f"F-score: {f_score}\n\n")

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


if __name__ == "__main__":
    main()
