class MicroCluster(object):

    def __init__(self,
                 label,  # the class the microcluster belongs to
                 instances=None,
                 timestamp=0, 
                 keep_instances=True #Required True for MINAS
                ):

        # TODO: remove instances entirely so it doesn't need to be stored in memory; Might not be possible because of _best_threshold used by MINAS which needs instances
        super(MicroCluster, self).__init__()
        self.label = label
        self.instances = instances

        self.n = len(instances)
        self.linear_sum = instances.sum(axis=0)
        # Sum of the squared l2 norms of all samples belonging to a microcluster:
        self.squared_sum = np.square(np.linalg.norm(self.instances, axis=1)).sum()
        # self.squared_sum = np.square(instances).sum(axis=0)  # From CluSTREAM paper

        self.centroid = self.linear_sum / self.n
        self.max_distance = np.max(self.distance_to_centroid(instances))
        self.mean_distance = np.mean(self.distance_to_centroid(instances))
        self.timestamp = timestamp

        self.update_properties()

        if not keep_instances:
            self.instances = None

    def __str__(self):
        return f"""Target class {self.label}
                # of instances: {self.n}
                Linear sum: {self.linear_sum}
                Squared sum: {self.squared_sum}
                Centroid: {self.centroid}
                Radius: {self.radius}
                Timestamp of last change: {self.timestamp}"""
    
    def small_str(self):
        return f"""Target class {self.label}
                # of instances: {self.n}
                Timestamp of last change: {self.timestamp}"""

    def get_radius(self):
        """Return radius of the subcluster"""
        factor = 1.5
        # from BIRCH Wikipedia
        #diff = (self.squared_sum / self.n) - np.dot(self.centroid, self.centroid)
        #if diff > 1e-15:
            #return factor * np.sqrt(diff)
        #else:  # in this case diff should be zero, but sometimes it's an infinitesimal difference
            #return 0
        # from MINAS paper:
        return factor*np.std(self.distance_to_centroid(self.instances))

    def distance_to_centroid(self, X):
        """Returns distance from X to centroid of this cluster.
        Parameters
        ----------
        X : numpy.ndarray
        """
        if len(X.shape) == 1:  # X is only one point
            return np.linalg.norm(X - self.centroid)
        else:  # X contains several points
            return np.linalg.norm(X - self.centroid, axis=1)

    def encompasses(self, X):
        """Check if points in X are inside this microcluster.
        Parameters
        ----------
        X : numpy.ndarray
        Returns
        -------
        numpy.bool_"""
        return np.less(self.distance_to_centroid(X), self.radius)

    def find_closest_cluster(self, clusters):
        """Finds closest cluster to this one among passed clusters.
        Parameters
        ----------
        clusters : List[minas.MicroCluster]
        Returns
        -------
        minas.MicroCluster
        """
        return min(clusters, key=lambda cl: cl.distance_to_centroid(self.centroid))

    def update_cluster(self, X, y, timestamp, update_summary):
        """
        Parameters
        ----------
        self : minas.MicroCluster
        X : numpy.ndarray
            X is one point.
        y : numpy.int64
        Returns
        -------
        """
        assert len(X.shape) == 1  # it's just one point
        self.timestamp = timestamp
        
        if self.instances is not None:
            self.instances = np.append(self.instances, [X],
                                       axis=0)  # TODO: remove later when dropping instances from class
        if update_summary:
            self.mean_distance = (self.n * self.mean_distance + self.distance_to_centroid(X)) / (self.n + 1)
            self.n += 1
            self.linear_sum = np.sum([self.linear_sum, X], axis=0)
            self.squared_sum = np.sum([self.squared_sum, np.square(X)], axis=0)
            self.update_properties()

    def update_properties(self):
        """
        Update centroid and radius based on current cluster properties.
        Returns
        -------
        None
        """
        self.centroid = self.linear_sum / self.n
        
        if self.instances is not None:
            self.radius = self.get_radius()

            if np.max(self.distance_to_centroid(self.instances)) > self.max_distance:
                self.max_distance = np.max(self.distance_to_centroid(self.instances))

    def is_cohesive(self, clusters):
        """Verifies if this cluster is cohesive for novelty detection purposes.
        A new micro-cluster is cohesive if its silhouette coefficient is larger than 0.
        b represents the Euclidean distance between the centroid of the new micro-cluster and the centroid of its
        closest micro-cluster, and a represents the standard deviation of the distances between the examples of the
        new micro-cluster and the centroid of the new micro-cluster.
        Parameters
        ----------
        clusters : List[minas.MicroCluster]
        Returns
        -------
        """
        b = self.distance_to_centroid(self.find_closest_cluster(clusters).centroid)
        a = np.std(self.distance_to_centroid(self.instances))
        silhouette = (b - a) / max(a, b)  # hm, this is always positive if b > a
        return silhouette > 0

    def is_representative(self, min_examples):
        """Verifies if this cluster is representative for novelty detection purposes.
        A new micro-cluster is representative if it contains a minimal number of examples,
        where this number is a user-defined parameter.
        Parameters
        ----------
        min_examples : int
        Returns
        -------
        bool
        """
        return self.n >= min_examples
    

class ShortMemInstance:
    def __init__(self, point, timestamp):
        self.point = point
        self.timestamp = timestamp

    def __eq__(self, other):
        """
        I'm considering elements equal if they have the same values for all variables.
        This currently does not consider the timestamp.
        """
        if type(other) == np.ndarray:
            return np.all(self.point == other)