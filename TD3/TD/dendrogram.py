"""
A basic dendrogram for single-linkage hierarchical clustering.
"""

from TD.graph import Graph, Edge


class Dendrogram:
    """
    Attributes
    ----------

    g : Graph -- the underlying graph
    parent : [int] -- parents for union-find
    rank : [int] -- ranks for union-find
    left : [int] -- binary tree lefts
    down : [int] -- binary tree downs
    height : [float]
        heights at which represented cluster is merged into parent
    cut_height : float -- current cut height
    cluster : [int] -- list of cluster representatives
    total_clusters : int -- number of clusters currently identified
    ns_clusters : int -- number of non-singleton clusters
    significant_heights : [float] -- List of significant heights (Ex. 5)
    """

    def __init__(self, g: Graph):
        self.g = g
        n = g.node_count()

        self.parent = [-1] * n
        self.rank = [0] * n
        self.left = [-1] * n
        self.down = [-1] * n
        self.height = [-1] * n
        self.cluster = [-1] * n

        self.cut_height = -1
        self.total_clusters = 0
        self.ns_clusters = 0
        self.significant_heights = []

    def __str__(self):
        lines = ["node\tparent\trank\tleft\tdown\theight\tcluster"]
        for i, vals in enumerate(
            zip(self.parent, self.rank, self.left, self.down, self.height, self.cluster)
        ):
            s = f"{i}\t"
            s += "\t".join("-" if v == -1 else str(v) for v in vals)
            lines.append(s)
        return "\n".join(lines)

    def get_n(self) -> int:
        return self.g.node_count()

    def get_dendrogram_height(self) -> float:
        """Height of the dendrogram."""
        return self.height[self.down[self.find_rep(0)]]

    def find_rep(self, i: int) -> int:
        """Find the representative in the union-find structure
        for the cluster containing node i.
        """
        assert 0 <= i < self.g.node_count()
        parent = self.parent[i]
        while parent != -1:
            i = parent
            parent = self.parent[i]
        return i

    def merge(self, e: Edge):
        """Merge the clusters connected by the edge."""
        # Plan:
        # 1. Find the representatives
        # 2. Choose the highest
        # 3. Adjust parent, left, and down
        # 4. Update ranks
        # 5. Update heights
        rep1 = self.find_rep(e.p1)
        rep2 = self.find_rep(e.p2)
        if self.rank[rep2] > self.rank[rep1]:
            rep1, rep2 = rep2, rep1

        self.rank[rep1] = max(self.rank[rep2] + 1, self.rank[rep1])
        self.parent[rep2] = rep1
        self.left[rep2] = self.down[rep1]
        self.down[rep1] = rep2
        self.height[rep2] = 0.5*e.length

    def build(self):
        """Merge along each edge in non-decreasing length order
        to build the dendrogram.
        """
        for edge in self.g:
            if self.find_rep(edge.p1) != self.find_rep(edge.p2):
                self.merge(edge)

    def find_heights(self, eps: float):
        """Put all heights <= eps into list of significant heights."""
        assert eps != 0.0
        h: float = self.get_dendrogram_height()
        slots = int(1 / eps + 1)
        buckets = [0.0] * slots
        for i, h_i in enumerate(self.height):
            if h_i == -1:
                continue
            q = int(h_i / eps / h)
            if h_i > buckets[q]:
                buckets[q] = h_i
        self.significant_heights = [x for x in buckets if x > 0]

    def _set_clusters(self, i: int, h: float):
        assert 0 <= i < self.get_n()

        # Root of Cluster
        if self.parent[i] == -1 or self.height[i] > h:
            self.cluster[i] = i
            self.total_clusters += 1
        # Not root of cluster
        else:
            self.cluster[i] = self.cluster[self.parent[i]]
        
        # Exploring to the left and down
        for node in [self.left[i], self.down[i]]:
            if node == -1:
                continue
            self._set_clusters(node, h)

    def set_clusters(self, h: float):
        """(Re)set clusters with cut height h."""
        if self.cut_height is h:
            return  # Already done!  Do nothing.
        self.cut_height = h
        self._set_clusters(self.find_rep(0), h)

    def _count_ns_clusters(self):
        """Count non-singleton clusters from scratch"""
        count = 0
        seen_at_least_once = set()
        seen_more_than_once = set()

        for cluster in self.cluster:
            if cluster not in seen_at_least_once:
                seen_at_least_once.add(cluster)
            elif cluster not in seen_more_than_once:
                seen_more_than_once.add(cluster)
                count += 1

        return count

    def count_ns_clusters(self) -> int:
        """The number of non-singleton clusters."""
        if self.ns_clusters == 0:
            self.ns_clusters = self._count_ns_clusters()
        return self.ns_clusters

    def clear_clusters(self):
        """Discard clustering data."""
        for i in range(len(self.cluster)):
            self.cluster[i] = -1
        self.ns_clusters = 0
        self.total_clusters = 0
        self.cut_height = -1

    def get_cluster_height(self, cluster: int) -> float:
        """Compute the height of the cluster c.
        Assumes build() has been called and the clusters of height <= h
        have been computed using set_clusters(h).
        """
        assert 0 <= cluster < self.get_n()
        
        nodes_in_cluster = [i for i, cluster_ in enumerate(self.cluster) if cluster_ == cluster]

        if len(nodes_in_cluster) <= 1:
            return 0.
    
        height = -1

        for node in nodes_in_cluster:
            if self.height[node] >= self.cut_height:
                return self.cut_height
            
            height = max(height, self.height[node])
        
        return height


    # /*** GETTERS ***/

    def get_name(self, i: int) -> str:
        """The name of the i-th node in the underlying graph."""
        assert 0 <= i < self.get_n()
        return self.g.get_name(i)

    # // For testing only

    def print_node(self, i: int):
        """For testing: print the i-th node."""
        print(
            f"Node (parent = {self.parent[i]},"
            f" down = {self.down[i]},"
            f" left = {self.left[i]},"
            f" rank = {self.rank[i]},"
            f" height = {self.height[i]},"
            f" cluster = {self.cluster[i]})"
        )

    def print_clusters(self):
        """For testing: print all clusters."""
        print_after = [-1] * self.get_n()
        for i, c_i in enumerate(self.cluster):
            if c_i != i:
                print_after[i] = print_after[c_i]
                print_after[c_i] = i
        for i, c_i in enumerate(self.cluster):
            if c_i == i:  # Cluster rep
                print(
                    f'Cluster "{c_i}" (node: {i}; height: {self.get_cluster_height(i)})'
                )
                next_c = print_after[i]
                while next_c != -1:
                    print(self.get_name(next_c))
                    next_c = print_after[next_c]
