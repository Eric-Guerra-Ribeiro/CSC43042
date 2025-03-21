from __future__ import annotations
import random

import numpy as np


class Point:
    """A point in a dataset.

    Attributes:
        d: int             -- dimension of the ambient space
        coords: np.ndarray -- coordinates of the point
        label: int = 0     -- label of the cluster the point is assigned to
    """
    def __init__(self, d: int):
        assert d > 0, "The dimension needs to be positive."
        self.d = d
        self.coords = np.zeros(d)
        self.label = 0
    
    def update_coords(self, new_coords: np.ndarray) -> None:
        """Copy the values of new_coords to coords."""
        self.coords = new_coords.copy()

    def squared_dist(self, other: Point) -> float:
        """The square of the Euclidean distance to other."""
        difference = self.coords - other.coords
        return np.dot(difference, difference)


class Cloud:
    """A cloud of points for the k-means algorithm.

    Data attributes:
    - d: int              -- dimension of the ambient space
    - points: list[Point] -- list of points
    - k: int              -- number of centers
    - centers: np.ndarray -- array of centers
    """

    def __init__(self, d: int, k: int):
        self.d = d
        self.k = k
        self.points = []
        self.centers = np.array([Point(d) for _ in range(self.k)])

    def add_point(self, p: Point, label: int) -> None:
        """Copy p to the cloud, in cluster label."""
        new_point = Point(self.d)
        new_point.update_coords(p.coords)
        self.points.append(new_point)
        self.points[-1].label = label

    def intracluster_variance(self) -> float:
        """Calculates the intracluster variance."""
        return sum(point.squared_dist(self.centers[point.label]) for point in self.points)/len(self.points)

    def set_voronoi_labels(self) -> int:
        """Sets the labels to each point as the closest center (Voronoi Partition)."""
        changes = 0
        for point in self.points:
            new_label = np.argmin(np.fromiter((point.squared_dist(center) for center in self.centers), dtype=float))
            if new_label != point.label:
                changes += 1
                point.label = new_label
        return changes

    def set_centroid_centers(self) -> None:
        """Updates the centers to be the centroid of the partitions"""
        number_in_clusters = np.zeros(self.k, dtype=int)
        centers_coords = np.zeros((self.k, self.d), dtype=float)

        for point in self.points:
            number_in_clusters[point.label] += 1
            centers_coords[point.label] += point.coords

        for i in range(self.k):
            if number_in_clusters[i] != 0:
                self.centers[i].update_coords(centers_coords[i]/number_in_clusters[i])

    def lloyd(self) -> None:
        """Lloyd's algorithm.
        Assumes the clusters have already been initialized somehow.
        """
        while self.set_voronoi_labels() != 0:
            self.set_centroid_centers()

    def init_random_partition(self) -> None:
        """Assigns a random uniform parition to the points."""
        for point in self.points:
            point.label = random.randrange(self.k)
        
        self.set_centroid_centers()

    def init_forgy(self) -> None:
        """Forgy's initialization: distinct centers are sampled
        uniformly at random from the points of the cloud.
        """
        choosen_points = random.sample(range(len(self.points)), self.k)
        for i in range(self.k):
            self.centers[i].update_coords(self.points[choosen_points[i]].coords)

    def init_plusplus(self) -> None:
        """Implements k-means++ implementation"""
        def min_squared_dist_to_center(point:Point, num_centers:int) -> float:
            return min(point.squared_dist(self.centers[i]) for i in range(num_centers))
        self.centers[0].update_coords(random.choice(self.points).coords)

        for i in range(1, self.k):
            probabilities = np.fromiter((min_squared_dist_to_center(point, i) for point in self.points), dtype=float)
            probabilities = probabilities/np.sum(probabilities)

            uniform = random.uniform(0, 1)

            cum_sum = 0.
            j = 0
            while cum_sum < uniform:
                cum_sum += probabilities[j]
                j += 1
            self.centers[i].update_coords(self.points[j - 1].coords)

    def silhouette(self, point:Point) -> float:
        """Calculates the silhouette of a point"""
        number_in_clusters = np.zeros(self.k, dtype=int)
        mean_distance = np.zeros(self.k, dtype=float)

        for other_point in self.points:
            number_in_clusters[other_point.label] += 1
            mean_distance[other_point.label] += np.sqrt(point.squared_dist(other_point))


        # If point i is alone in cluster, we define s(i) = 0
        if number_in_clusters[point.label] == 1:
            return 0.

        # To avoid problems, if all points are in the same cluster, we'll assume s(i) = 1
        if number_in_clusters[point.label] == len(self.points):
            return 1.

        # a(i) = 1/(|C_I| - 1)* \sum_{j \in C_I, i != j} d(i, j)
        a_point = mean_distance[point.label]/(number_in_clusters[point.label] - 1)

        for i in range(self.k):
            if number_in_clusters[i] == 0 or i == point.label:
                mean_distance[i] = np.inf
                continue
            mean_distance[i] = mean_distance[i]/number_in_clusters[i]
        
        # b(i) = min J != I: 1/(C_J) * \sum_{j \in C_J} d(i, j)
        b_point = np.min(mean_distance)

        return (b_point - a_point)/max(a_point, b_point)
            
    def mean_silhouette(self):
        return sum(self.silhouette(point) for point in self.points)/len(self.points)
