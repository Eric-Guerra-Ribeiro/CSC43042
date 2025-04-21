from typing import Tuple

def my_parametrization(d: int, nb_train_data: int) -> Tuple[int, int, str, float, int]:
    # Define good values for the variables that are returned.
    k = 300
    projection_dim = 267
    type_sample = ""
    threshold = 0.2
    seed = 0

# threshold = 0.4, k = 25, dim = 152, type = Gaussian

    return (k, projection_dim, type_sample, threshold, seed)


if __name__ == "__main__":
    from pathlib import Path
    import time

    import numpy as np

    from TD.dataset import Dataset
    from TD.confusion_matrix import ConfusionMatrix
    from TD.knn_classification import KnnClassification
    from TD.random_projection import RandomProjection

    def score(threshold, knn_base, knn_proj, proj_test_dataset, projection_dimension):
        # Compute the quality of the result of the base result
        cm = ConfusionMatrix()
        start_time_test_orig = time.time()
        for i in range(test_dataset.nsamples):
            sample = test_dataset.instances[i]
            query = np.delete(sample, [col_class], axis=0)
            predicted_label = knn_base.estimate(query, threshold)
            true_label = int(sample[col_class])
            cm.add_prediction(true_label, predicted_label)
        duration_test_orig = time.time() - start_time_test_orig
        # Compute the quality of the result
        cm = ConfusionMatrix()
        start_time_test_proj = time.time()
        for i in range(proj_test_dataset.nsamples):
            sample = proj_test_dataset.instances[i]
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
        return score

    train_data_file = "./csv/samples_train.csv"
    test_data_file = "./csv/samples_test_excerpt.csv"

    train_dataset = Dataset(train_data_file)
    test_dataset = Dataset(test_data_file)
    col_class = train_dataset.dim - 1
    n_dim = train_dataset.dim
    n_trains = train_dataset.nsamples

    max_score = -np.inf

    thresholds = [0.1*i for i in range(1, 10)]
    type_samples = ["Gaussian", "Rademacher"]
    ks = [int(n_trains*0.03*i) for i in range(1, 5)]
    projection_dims = [int(n_dim*0.1*i) for i in range(4, 8)]

    for type_sample in type_samples:
        for k in ks:
            for projection_dim in projection_dims:
                for threshold in thresholds:
                    np.random.seed(0)
                    rnd_proj = RandomProjection(
                        train_dataset.dim - 1, col_class, projection_dim, type_sample
                    )
                    proj_ds = rnd_proj.project(train_dataset)
                    proj_test_ds = rnd_proj.project(train_dataset)

                    knn_base = KnnClassification(k, train_dataset, col_class)
                    knn_proj = KnnClassification(k, proj_ds, projection_dim)

                    new_score = score(threshold, knn_base, knn_proj, proj_test_ds, projection_dim)
                    if new_score > max_score:
                        max_score = new_score
                        print(f"threshold = {threshold}, k = {k}, dim = {projection_dim}, type = {type_sample}")
