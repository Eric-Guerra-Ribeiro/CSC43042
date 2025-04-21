#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# Mapping from dataset name to target variable
file_to_target = {
    "salary": "salary",
    "maisons": "price"
}

def normalize(train_data, test_data, col_regr, method='mean_std'):
    # Remove the target column so that it is not scaled
    no_class_train = train_data.drop(col_regr, axis=1)
    no_class_test = test_data.drop(col_regr, axis=1)

    if method == 'mean_std':
        normalized_train = (no_class_train - no_class_train.mean()) / no_class_train.std()
        normalized_test = (no_class_test - no_class_train.mean()) / no_class_train.std()
    elif method == 'maxmin':
        normalized_train = (no_class_train - no_class_train.min()) / (no_class_train.max() - no_class_train.min())
        normalized_test = (no_class_test - no_class_train.min()) / (no_class_train.max() - no_class_train.min())
    else:
        raise ValueError(f"Unknown method {method}")

    # Reattach the target column and return
    return pd.concat([train_data[col_regr], normalized_train], axis=1), pd.concat([test_data[col_regr], normalized_test], axis=1)

def get_data(base_path="../csv", file_prefix="maisons", feature_cols=None, target=None, norm=False):
    # Validate dataset choice and set target column
    assert file_prefix in ["salary", "maisons"], "Unknown file"
    if target is None:
        target = file_to_target[file_prefix]
    train_path = os.path.join(base_path, f"{file_prefix}_train.csv")
    test_path = os.path.join(base_path, f"{file_prefix}_test.csv")

    train_dataset = pd.read_csv(train_path, header=0)
    print_summary(train_dataset, label="Training Data")
    test_dataset = pd.read_csv(test_path, header=0)
    print_summary(test_dataset, label="Test Data")

    if norm:
        train_dataset, test_dataset = normalize(train_dataset, test_dataset, target)
    
    # Use all columns except the target if not provided
    if feature_cols is None:
        feature_cols = train_dataset.columns.to_list()
        feature_cols.remove(target)
    
    X_train = train_dataset[feature_cols]
    y_train = train_dataset[target]
    X_test = test_dataset[feature_cols]
    y_test = test_dataset[target]

    return train_dataset, test_dataset, X_train, y_train, X_test, y_test

def print_summary(dataset, label="Dataset"):
    # Print a formatted summary of the DataFrame
    print("\n" + "=" * 60)
    print(f"{label} Summary:")
    print("=" * 60)
    print("Shape:", dataset.shape)
    # Using to_string for a tabulated look of the first few rows
    print("\nHead:\n", dataset.head().to_string(index=False))
    print("\nDescriptive Statistics:\n", dataset.describe().to_string())
    print("=" * 60 + "\n")

def fit_and_predict(X_train, y_train, X_test, y_test, regressor, verbose=False):
    regressor.fit(X_train, y_train)
    if isinstance(regressor, LinearRegression):
        print("\nLinear Regression Details:")
        print("-" * 40)
        print(f"Intercept: {regressor.intercept_:.2f}")
        print("Coefficients:")
        # Print coefficient names with values for clarity
        for col, coef in zip(X_train.columns, regressor.coef_):
            print(f"  {col:>10}: {coef:.2f}")
        print("-" * 40)
    y_pred = regressor.predict(X_test)
    if verbose:
        print("\nPredictions:")
        for true_val, pred in zip(y_test, y_pred):
            print(f"  True value: {true_val:10.2f}  |  Predicted: {pred:10.2f}")
    return y_pred

def evaluate_performance(y_test, y_pred):
    # Compute the performance metrics
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    # Print a nicely formatted summary of the evaluation metrics
    print("\nEvaluation Metrics:")
    print("=" * 40)
    print(f"Mean Absolute Error (MAE): {mae:10.2f}")
    print(f"Mean Squared Error (MSE): {mse:10.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:10.2f}")
    print("=" * 40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run regression models with optional data filtering and improved output.')
    parser.add_argument('dataset', choices=['salary', 'maisons'], help='Dataset to use: salary or maisons')
    parser.add_argument('--filter', type=str, default=None,
                        help='Filter condition using pandas query syntax (e.g. "yr >= 5").')
    parser.add_argument('--regressor', choices=['linear', 'knn'], default='linear',
                        help='Choose which regression model to run: linear or knn')
    args = parser.parse_args()

    # Load and print the data summaries
    train_ds, test_ds, X_train, y_train, X_test, y_test = get_data(file_prefix=args.dataset)
    
    # Apply data filtering if provided
    if args.filter:
        print(f"\nApplying filter: {args.filter}")
        target_col = file_to_target[args.dataset]
        # Apply the filter on both training and test sets
        train_ds = train_ds.query(args.filter)
        test_ds = test_ds.query(args.filter)
        # Recompute feature matrices and targets
        feature_cols = train_ds.columns.to_list()
        feature_cols.remove(target_col)
        X_train = train_ds[feature_cols]
        y_train = train_ds[target_col]
        X_test = test_ds[feature_cols]
        y_test = test_ds[target_col]
        print(f"After filtering, training set shape: {X_train.shape}")
        print(f"After filtering, test set shape: {X_test.shape}")
    
    # Select the regression model based on command-line argument
    if args.regressor == 'linear':
        reg = LinearRegression()
    else:
        reg = KNeighborsRegressor(n_neighbors=10)

    # Fit the model, make predictions, and evaluate the performance
    y_pred = fit_and_predict(X_train, y_train, X_test, y_test, reg, verbose=True)
    evaluate_performance(y_test, y_pred)

