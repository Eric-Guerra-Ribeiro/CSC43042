import numpy as np
import sys
from Dataset import Dataset
from KnnRegression import KnnRegression

def main() -> None:
    """
    Main function to run the k-nearest neighbors regression test.
    
    Usage:
        python test_k_knn.py <train_file> <test_file> [<column_for_Regression>]
    """
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <train_file> <test_file> [ <column_for_Regression> ]")
        sys.exit(1)

    train_dataset = Dataset(sys.argv[1])
    test_dataset = Dataset(sys.argv[2])

    if len(sys.argv) == 4:
        col_regr = int(sys.argv[3])
    else:
        col_regr = train_dataset.get_dim() - 1
        print(f"No column specified for Regression, assuming last column of dataset ({col_regr}).")

    assert train_dataset.get_dim() == test_dataset.get_dim(), "Training and test datasets must have the same dimensions."

    dim: int = train_dataset.get_dim()

    with open("output.txt", "w") as fp:
        for k in range(1, 101):
            total_error: float = 0.0

            knn_reg = KnnRegression(k, train_dataset, col_regr)
            for i in range(test_dataset.get_nbr_samples()):
                sample = test_dataset.get_instance(i)
                query = np.zeros(dim - 1)
                j2 = 0
                # Build the query vector by skipping the target column.
                for j in range(dim - 1):
                    if j2 == col_regr:
                        j2 += 1
                    query[j] = sample[j2]
                    j2 += 1

                estimate = knn_reg.estimate(query)
                total_error += ((estimate - sample[col_regr]) ** 2) / test_dataset.get_nbr_samples()

            fp.write(f"{k:2d} {total_error:9.6f}\n")

if __name__ == "__main__":
    main()

