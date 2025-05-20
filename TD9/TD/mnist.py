#! /usr/bin/env -S python3 --relpath-append ..
import os
import urllib.request
from pathlib import Path

import numpy as np

FILENAME = "mnist.npz"
URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
REMOTE_PATH = URL + FILENAME
LOCALPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), FILENAME)


def download_mnist():
    urllib.request.urlretrieve(REMOTE_PATH, LOCALPATH)
    print("Download complete.")


def cleanup():
    print("Removing original files...")
    os.remove(LOCALPATH)
    print("All cleaned up.")


def load(set, path: str = LOCALPATH):
    with np.load(path, allow_pickle=True) as f:
        x, y = f[f"x_{set}"], f[f"y_{set}"]
    return x, y


def mnist(set: str = "train"):
    try:
        print("Trying to load from local...")
        return load(set)
    except FileNotFoundError:
        print("... failed. Trying to load from grader server...")
        try:
            return load(set, next(Path(__file__).parent.parent.parent.rglob("mnist.npz")))
        except Exception:
            print("Failed. Downloading to local and loading.")
            download_mnist()
            return load(set)


def get_raw_data(set: str = "train"):
    """
    This is the input public data
    """
    x, y = mnist(set)
    # Flatten images
    x = x.reshape((x.shape[0], -1))
    # One-hot encode labels
    y = np.eye(10)[y]
    # You **may** want to use a subset of the data (unless it's winter, and you need a backup heater)
    x_train, y_train = x[:8000, :], y[:8000, :]
    x_val, y_val = x[10000:14000, :], y[10000:14000, :]
    return x_train, y_train, x_val, y_val
