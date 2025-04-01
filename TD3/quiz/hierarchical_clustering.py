import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import math

def analyse_bluered():
    fname = './csv/bluered.csv'
    data = pd.read_csv(fname, header = 0)

    tree = linkage(data[['x', 'y']])
    D = dendrogram(tree, labels = data['name'].to_numpy(), orientation = 'left', color_threshold=2.0)
    plt.show()


def analyse_test6():
    fname = './csv/test6.csv'
    data = pd.read_csv(fname, header = 0)

    print("Test 6:")
    print("     A      B      C      D      E     F")
    for x0, y0, letter in zip(data.x, data.y, ["A", "B", "C", "D", "E", "F"]):
        print(f"{letter}", end=" ")
        for x1, y1 in zip(data.x, data.y):
            print(f"{math.sqrt((x0-x1)**2 + (y0-y1)**2): .3f}", end=" ")
        print()

    tree = linkage(data[['x', 'y']])
    D = dendrogram(tree, labels = data['name'].to_numpy(), orientation = 'left')
    plt.show()


def analyse_iris():
    fname = './csv/iris.csv'
    data = pd.read_csv(fname, header = 0)

    tree = linkage(data[['a', 'b', 'c', 'd']])
    plt.figure(figsize=(10, 10))
    D = dendrogram(tree, labels = data['name'].to_numpy(), orientation = 'left')
    plt.show()


def analyse_languages():
    def load_distance_matrix(fname):
        """
        Takes as input a name of a file containing the information about a graph:
        - first line: number of vertices
        - then labels, one per line
        - then the distance matrix, one row per line, the entries separated by commas
        Returns a tuple containing a distance matrix in the condensed 1-D format and a list of labels
        """
        with open(fname, 'r') as infile:
            n = int(infile.readline())
            labels = [infile.readline().strip() for _ in range(n)]
            dist_matrix = [[float(x) for x in line.split(',')] for line in infile]
            return (squareform(dist_matrix), labels)

    fname = "./csv/languages.csv"
    dist_matrix, labels = load_distance_matrix(fname)
    tree = linkage(dist_matrix)
    plt.figure(figsize=(10, 10))
    D = dendrogram(tree, labels = labels, orientation = 'left')
    plt.show()

if __name__ == "__main__":
    analyse_bluered()
    analyse_test6()
    analyse_iris()
    analyse_languages()
