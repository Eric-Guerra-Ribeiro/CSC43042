import sys
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import os

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        readfrom_train = sys.argv[1]
        readfrom_test = sys.argv[2]
    else:
        print(f"Syntax: python {sys.argv[0]} <dataset_train> <dataset_test> [<label_column>] [<rest_of_cols>] ['<char_sep>']")
        print("Example: python svm_with_sklearn.py ramen-train.csv ramen-test.csv 0 - ','")
        exit(1)

    if len(sys.argv) >= 5:
        label = int(sys.argv[3])
        rest = sys.argv[4]
        if len(sys.argv) >= 6:
            char_sep = sys.argv[5]
        else:
            char_sep = ','
    else:
        label = 0
        rest = "-"
        char_sep = ','

    if rest != "-":
        id_cols = list(map(int, rest.split(',')))
        some_cols = True
    else:
        some_cols = False

    print(f"Character separation: [{char_sep}]")

    # Detect filenames
    train_filename = os.path.basename(readfrom_train)
    test_filename = os.path.basename(readfrom_test)

    if 'scooter' in train_filename and 'scooter' in test_filename:
        dataset_train = pd.read_csv(readfrom_train, sep=char_sep, header=None, comment='#')
        dataset_test = pd.read_csv(readfrom_test, sep=char_sep, header=None, comment='#')
    else:
        dataset_train = pd.read_csv(readfrom_train, sep=char_sep, header=0, comment='#')
        dataset_test = pd.read_csv(readfrom_test, sep=char_sep, header=0, comment='#')

    if some_cols:
        X_train = dataset_train.iloc[:, id_cols]
        X_test = dataset_test.iloc[:, id_cols]
    else:
        X_train = dataset_train.drop(columns=dataset_train.columns[label])
        X_test = dataset_test.drop(columns=dataset_test.columns[label])

    y_train = dataset_train.iloc[:, label]
    y_test = dataset_test.iloc[:, label]

    # New line to fix labels:
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # ---- Students: modify these parameters if needed ----
    C_param = 1
    gamma_param = 'auto'
    kernel_type = 'rbf'
    # -----------------------------------------------------

    svm_model = sk.svm.SVC(C=C_param, gamma=gamma_param, kernel=kernel_type)

    # 1. Without scaling
    svm_model.fit(X_train, y_train)
    predictions = svm_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy (no scaling): {acc:.4f} ({np.sum(predictions == y_test)}/{len(y_test)})")

    # 2. With MinMax scaling
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_model.fit(X_train_scaled, y_train)
    predictions_scaled = svm_model.predict(X_test_scaled)
    acc_scaled = accuracy_score(y_test, predictions_scaled)
    print(f"Accuracy after scaling: {acc_scaled:.4f} ({np.sum(predictions_scaled == y_test)}/{len(y_test)})")

    # ---- EXTRA: Show data and simple inference for ebike dataset ----
    if 'ebike' in train_filename and 'ebike' in test_filename:
        print("\nPrediction summary for E-Bike dataset (after scaling):")
        print(f"{'avg_speed':>10} {'battery_usage':>15} {'ride_duration':>15} {'elevation_gain':>15} {'braking_events':>15} {'Predicted':>12} {'TrueLabel':>12} | Inference")
        print("-" * 120)

        # Re-read unscaled
        X_test_unscaled = pd.read_csv(readfrom_test, sep=char_sep, header=0, comment='#')

        for idx in range(len(X_test_unscaled)):
            avg_speed = X_test_unscaled.iloc[idx]['avg_speed']
            battery_usage = X_test_unscaled.iloc[idx]['battery_usage']
            ride_duration = X_test_unscaled.iloc[idx]['ride_duration']
            elevation_gain = X_test_unscaled.iloc[idx]['elevation_gain']
            braking_events = X_test_unscaled.iloc[idx]['braking_events']
            pred = predictions_scaled[idx]
            true_label = y_test.iloc[idx]

            if avg_speed > 25 and braking_events > 10:
                ride_type = "Aggressive ride (likely)"
            else:
                ride_type = "Normal ride (likely)"

            print(f"{avg_speed:10.2f} {battery_usage:15.2f} {ride_duration:15.2f} {elevation_gain:15.2f} {braking_events:15.0f} {pred:12d} {true_label:12d} | {ride_type}")

    # ---- EXTRA: Show data and simple inference for hilly dataset ----
    if 'hilly' in train_filename and 'hilly' in test_filename:
        print("\nPrediction summary for Hilly dataset (after scaling):")
        print(f"{'trip_distance':>12} {'avg_slope':>10} {'energy_consumed':>17} {'max_speed':>10} {'stop_count':>10} {'Predicted':>12} {'TrueLabel':>12} | Inference")
        print("-" * 120)

        # Re-read unscaled
        X_test_unscaled = pd.read_csv(readfrom_test, sep=char_sep, header=0, comment='#')

        for idx in range(len(X_test_unscaled)):
            trip_distance = X_test_unscaled.iloc[idx]['trip_distance']
            avg_slope = X_test_unscaled.iloc[idx]['avg_slope']
            energy_consumed = X_test_unscaled.iloc[idx]['energy_consumed']
            max_speed = X_test_unscaled.iloc[idx]['max_speed']
            stop_count = int(X_test_unscaled.iloc[idx]['stop_count'])
            pred = predictions_scaled[idx]
            true_label = y_test.iloc[idx]

            if avg_slope > 5 and energy_consumed > 300:
                trip_type = "Difficult trip (likely)"
            else:
                trip_type = "Easy trip (likely)"

            print(f"{trip_distance:12.2f} {avg_slope:10.2f} {energy_consumed:17.2f} {max_speed:10.2f} {stop_count:10d} {pred:12d} {true_label:12d} | {trip_type}")

