import random

import numpy as np
from matplotlib import pyplot as plt

CENTERS = np.array([
    [0.03354999971005405, 0.910624352227072, 1.1065662160469756, -0.06373332692739535,
      0.15445959336410092, 0.02526618982037847, 0.8173431310499057, -0.20388635910063638,
      -0.03887584231642263, 0.09352659890359048, 0.9002932639110137, -0.02961179177250613,
      1.0917575654804765, 0.04524275176260947, 1.0153839282806827, 0.9494511794392126,
      -0.09105943803794389, 1.0690023908150579, 0.010479536021934723, 1.0053602592583737],
    [0.03812821569380051, -0.030865402583636417, 0.9462673725240922, 1.0075572672750144,
      -0.005185311884068647, 1.0874731129396789, 0.9783762439293723, -0.1298156228590049,
      1.0327135873480289, 0.9936624156481715, -0.0764165264719577, -0.06433901259738932,
      0.031394116495359385, -0.004635864670913432, -0.16734970305099892, 0.9776763471288614,
      0.027289340712123644, 0.992646454337251, -0.09135085425924867, -0.04229499642262872],
    [0.984816516731726, 0.8878524474132734, 0.02266074698381395, 0.06549981298213167,
      0.9757100801186352, 1.0540947925465123, 1.1606710132326825, 0.10243189954902514,
      -0.05852878147072556, -0.06311244497500267, -0.10796776893505858, 0.09047275876604775,
      1.062924841974187, 0.05035138591892108, -0.02572141567436182, -0.11179838011385902,
      1.0715613034153728, 0.15591607220687767, 0.9146960964260246, 0.932348370853073],
    [0.9495463976626809, 0.11768266288827224, 0.027335352785492723, -0.08566613999343939,
      -0.09713068495083091, 1.1222010811180367, 0.0906197313523648, 0.7809992116208779,
      1.0218592715953, -0.20079423774623226, 0.9554090913408686, 0.9501189763231526,
      0.05800522593222039, 1.0627604103095925, 0.9422958066065025, 0.9250964557352578,
      1.0108765221781288, 0.9944065252273451, 0.8754257429629712, -0.04918322506061405]
])

def my_parametrization(d: int, nb_samples: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Set empirically determined initial parameters for $k$-means."""
    nb_clusters = 4
    my_labels = np.random.randint(0, nb_clusters - 1, nb_samples, dtype=int)
    my_center_coords = CENTERS


    return (nb_clusters, my_labels, my_center_coords)

def main():
    from kmeans import Point, Cloud
    # Read data
    data = np.genfromtxt("./csv/samples_excerpt.csv", delimiter=",")
    (nmax, d) = data.shape

    plot_silhouette = False
    print_values = False

    k_min = 2
    k_max = 10
    mean_silhouettes = np.zeros(k_max - k_min + 1)

    for k in range(k_min, k_max + 1):
        cloud = Cloud(d, k)
        for i in range(nmax):
            p = Point(d)
            p.update_coords(data[i])
            cloud.add_point(p, random.randrange(k))

        cloud.init_plusplus()
        cloud.lloyd()
        mean_silhouettes[k - k_min] = cloud.mean_silhouette()
    
    if plot_silhouette:
        plt.plot(range(k_min, k_max + 1), mean_silhouettes)
        plt.title("Mean Silhouette")
        plt.xlabel("$k$")
        plt.ylabel("$\\frac{1}{n}\\sum_{p \\in P} s(p)$")

        plt.show()



    nb_clusters = np.argmax(mean_silhouettes) + k_min

    

    # Build cloud with optimal number of clusters

    cloud = Cloud(d, nb_clusters)
    for i in range(nmax):
        p = Point(d)
        p.update_coords(data[i])
        cloud.add_point(p, random.randrange(k))

    cloud.init_plusplus()
    cloud.lloyd()

    if print_values:
        print(f"Best number of clusters: {nb_clusters}")
        print("Centers:")
        print(f"{[[coord for coord in center.coords] for center in cloud.centers]}")

if __name__ == "__main__":
    main()
