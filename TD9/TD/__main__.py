#! /usr/bin/env -S python3
"""
__main__ is run when the module is "executed"
"""
import argparse
import numpy as np
import TD
from TD.network import sine, xaggle


def activations():
    x = np.arange(-5, 5, 0.01)
    TD.activations.Identity.plot(x)
    TD.activations.Sigmoid.plot(x)


def mnist():
    import matplotlib.pyplot as plt
    x, y = TD.mnist.mnist()
    f, axes = plt.subplots(5, 10, figsize=(11, 8))
    for i in range(10):
        count = 0
        j = 0
        while count < 5 and j < x.shape[0]:
            if y[j] == i:
                axes[count, i].imshow(x[j, :].reshape(28, 28))
                axes[count, i].axis('off')
                if count == 0:
                    axes[count, i].set_title(i)
                count += 1
            j += 1
    plt.show()


def losses():
    TD.losses.Quadratic.plot()
    TD.losses.CrossEntropy.plot()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["activations", "losses", "mnist", "sine", "xaggle"])
    args = parser.parse_args()
    plot = str(args.command)
    globals().get(plot)()


if __name__ == "__main__":
    main()
