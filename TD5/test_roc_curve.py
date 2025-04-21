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
    if len(sys.argv) == 6:
        col_class = int(sys.argv[5])

    assert train_ds.dim == test_ds.dim, "Different dimensions from test and train."

    thresholds = [x / (num_points + 1.0) for x in range(1, num_points + 1)]
    conf_matrices = [ConfusionMatrix() for _ in range(num_points)]

    # Add the predictions here.
    for i, threshold in enumerate(thresholds):
        knn_classifier = KnnClassification(k, train_ds, col_class)
        for x in test_ds.instances:
            true_label = int(x[col_class])
            predicted_label = knn_classifier.estimate(np.delete(x, col_class), threshold)
            conf_matrices[i].add_prediction(true_label, predicted_label)

    recalls = np.concatenate((np.zeros(1), np.fromiter((mtx.detection_rate() for mtx in reversed(conf_matrices)), dtype=float), np.ones(1)))
    false_alarms = np.concatenate((np.zeros(1), np.fromiter((mtx.false_alarm_rate() for mtx in reversed(conf_matrices)), dtype=float), np.ones(1)))

    f_scores = np.fromiter((mtx.f_score() for mtx in reversed(conf_matrices)), dtype=float)

    area_under_curve = np.trapz(recalls, false_alarms)
    print(f"k: {k}")
    print(f"AUC Score: {area_under_curve:.10f}")
    print(f"F-Score: {np.max(f_scores):.10f}")

    plt.plot(false_alarms, recalls, color='crimson', linewidth=2)
    plt.plot([0, 0, 1], [0, 1, 1], color='blue', linewidth=1, linestyle='--')
    plt.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
    plt.title("Receiver Operating Characteristic Curve")
    plt.xlabel("False-alarm rate")
    plt.ylabel("Recall rate")
    plt.show()

if __name__ == "__main__":
    create_roc_plot()
