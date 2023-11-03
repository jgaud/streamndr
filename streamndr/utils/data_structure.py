import numpy as np
import math

__all__ = ["MicroCluster", "ShortMemInstance", "ImpurityBasedCluster"]

class MicroCluster(object):
    """A representation of a cluster with compressed information.

    Parameters
    ----------
    label : int
        Label associated with this microcluster
    instances : numpy.ndarray
        Instances in this microcluster, preferably these would not be stored if not needed using keep_instances=False
    timestamp : int
        Timestamp this microcluster was last updated, used for forgetting mechanisms  
    keep_instances : bool
        Whether or not to store the instances within the microcluster. Should preferably set to false, but some implementations require
        access to the instances

    Attributes
    ----------
    n : int
        Number of instances stored in this microcluster
    linear_sum : numpy.ndarray
        Linear sum of the points belonging to this microcluster
    squared_sum : numpy.ndarray
        Sum of the squared l2 norms of all samples belonging to this microcluster
    centroid : numpy.ndarray
        Centroid coordinates of the microcluster
    max_distance : numpy.ndarray
        Maximum distance between a point belonging to the microcluster and its centroid
    mean_distance : numpy.ndarray
        Mean distance of the distances between the cluster's points and its centroid
    """

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
        """Returns string representation of a microcluster.

        Returns
        -------
        str
            String representation of microcluster
        """

        return f"""Target class {self.label}
                # of instances: {self.n}
                Linear sum: {self.linear_sum}
                Squared sum: {self.squared_sum}
                Centroid: {self.centroid}
                Radius: {self.radius}
                Timestamp of last change: {self.timestamp}"""
    
    def small_str(self):
        """Returns string representation of a microcluster.

        Returns
        -------
        str
            Small string representation of microcluster
        """

        return f"""Target class {self.label}
                # of instances: {self.n}
                Timestamp of last change: {self.timestamp}"""

    def get_radius(self):
        """Returns radius of the microcluster.

        Returns
        -------
        float
            Radius of the microcluster
        """

        factor = 1.5
        # from BIRCH Wikipedia
        diff = (self.squared_sum / self.n) - np.dot(self.centroid, self.centroid)
        if diff > 1e-15:
            return factor * np.sqrt(diff)
        else:  # in this case diff should be zero, but sometimes it's an infinitesimal difference
            return 0
        # from MINAS paper:
        #return factor*np.std(self.distance_to_centroid(self.instances))

    def distance_to_centroid(self, X):
        """Returns distance from X to centroid of this cluster.

        Parameters
        ----------
        X : numpy.ndarray
            Point or multiple points

        Returns
        -------
        numpy.ndarray
            Distance from X to the microcluster's centroid
        """

        if len(X.shape) == 1:  # X is only one point
            return np.linalg.norm(X - self.centroid)
        else:  # X contains several points
            return np.linalg.norm(X - self.centroid, axis=1)

    def encompasses(self, X):
        """Checks if point X is inside this microcluster. The point X is considered within the microcluster if the distance 
        between the point and the microcluster's centroid is less than the radius of the microcluster.

        Parameters
        ----------
        X : numpy.ndarray
            One point

        Returns
        -------
        bool
            If the point distance to centroid is contained within the microcluster or not
        """

        return np.less(self.distance_to_centroid(X), self.radius)

    def find_closest_cluster(self, clusters):
        """Finds closest microcluster to this one among passed microclusters.

        Parameters
        ----------
        clusters : list of MicroCluster

        Returns
        -------
        MicroCluster
            Closest microcluster
        """

        return min(clusters, key=lambda cl: cl.distance_to_centroid(self.centroid))

    def update_cluster(self, X, timestamp, update_summary):
        """Adds point received in parameter to the cluster and update cluster's centroid if wanted.

        Parameters
        ----------
        X : numpy.ndarray
            One point
        timestamp : int
            Timestamp when this point was added to this microcluster
        update_summary : bool
            Whether or not to update the microcluster properties with this new point
        """

        assert len(X.shape) == 1  # it's just one point
        self.timestamp = timestamp
        
        if self.instances is not None:
            self.instances = np.append(self.instances, [X],
                                       axis=0)
        if update_summary:
            self.mean_distance = (self.n * self.mean_distance + self.distance_to_centroid(X)) / (self.n + 1)
            self.n += 1
            self.linear_sum = np.sum([self.linear_sum, X], axis=0)
            self.squared_sum = np.sum([self.squared_sum, np.square(X).sum()], axis=0)
            self.update_properties()

    def update_properties(self):
        """Updates centroid and radius based on current cluster properties."""
        self.centroid = self.linear_sum / self.n
        
        if self.instances is not None:
            self.radius = self.get_radius()

            if np.max(self.distance_to_centroid(self.instances)) > self.max_distance:
                self.max_distance = np.max(self.distance_to_centroid(self.instances))

    def is_cohesive(self, clusters):
        """Verifies if this cluster is cohesive for novelty detection purposes.
        A new micro-cluster is cohesive if its silhouette coefficient is larger than 0.
        'b' represents the Euclidean distance between the centroid of the new micro-cluster and the centroid of its
        closest micro-cluster, and 'a' represents the standard deviation of the distances between the examples of the
        new micro-cluster and the centroid of the new micro-cluster.

        Parameters
        ----------
        clusters : List of MicroCluster
            Existing known micro-clusters

        Returns
        -------
        bool
            If the cluster is cohesive (silhouette>0) or not
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
            The number of samples the microcluster needs to have to be considered representative.

        Returns
        -------
        bool
            If the cluster is representative or not
        """
        return self.n >= min_examples

class ImpurityBasedCluster:

    def __init__(self, id, centroid):

        self.id = id
        self.centroid = centroid

        self.number_of_labeled_samples = 0
        self.samples_by_label = {}
        self.unlabeled_samples = []

        self.entropy = 0

    def size(self):
        return self.number_of_labeled_samples + len(self.unlabeled_samples)

    def add_sample(self, sample):
        if sample.y_true is not None:
            if sample.y_true not in self.samples_by_label:
                self.samples_by_label[sample.y_true] = []
            self.samples_by_label[sample.y_true].append(sample)
            self.number_of_labeled_samples += 1
        else:
            self.unlabeled_samples.append(sample)

    def remove_sample(self, sample):
        if sample.y_true is not None:
            if sample.y_true in self.samples_by_label:
                self.samples_by_label[sample.y_true].remove(sample)
            self.number_of_labeled_samples -= 1
        else:
            self.unlabeled_samples.remove(sample)

    def update_entropy(self):
        label_probabilities = [self.calculate_label_probability(label) for label in self.samples_by_label]
        self.entropy = -sum(p * math.log(p) for p in label_probabilities if p > 0)

    def calculate_label_probability(self, label):
        return len(self.samples_by_label[label]) / self.number_of_labeled_samples

    def update_centroid(self):
        samples = []
        for label_samples in self.samples_by_label.values():
            samples.extend([sample.point for sample in label_samples])
        
        samples.extend([sample.point for sample in self.unlabeled_samples])

        linear_sum = np.sum(samples, axis=0)
        self.centroid = linear_sum / len(samples)

    def distance_to_centroid(self, X):
        """Returns distance from X to centroid of this cluster.

        Parameters
        ----------
        X : numpy.ndarray
            Point or multiple points

        Returns
        -------
        numpy.ndarray
            Distance from X to the microcluster's centroid
        """

        if len(X.shape) == 1:  # X is only one point
            return np.linalg.norm(X - self.centroid)
        else:  # X contains several points
            return np.linalg.norm(X - self.centroid, axis=1)

    def calculate_standard_deviation(self):
        samples = []
        for label_samples in self.samples_by_label.values():
            samples.extend(label_samples)
        samples.extend(self.unlabeled_samples)

        centroid = self.get_centroid()
        sum_of_squared_distances = sum((sample.distance(centroid) ** 2) for sample in samples)
        return math.sqrt(sum_of_squared_distances / len(samples))

    def calculate_radius(self):
        samples = []
        for label_samples in self.samples_by_label.values():
            samples.extend(label_samples)
        samples.extend(self.unlabeled_samples)

        max_distance = max(samples, key=lambda sample: self.centroid.distance(sample))
        return self.centroid.distance(max_distance)

    def get_samples(self):
        samples = []
        for label_samples in self.samples_by_label.values():
            samples.extend(label_samples)
        samples.extend(self.unlabeled_samples)
        return samples

    def get_most_frequent_label(self):
        return max(self.samples_by_label, key=lambda label: len(self.samples_by_label[label]))

    def dissimilarity_count(self, labeled_sample):
        if labeled_sample.y_true not in self.samples_by_label:
            return self.number_of_labeled_samples
        return self.number_of_labeled_samples - len(self.samples_by_label[labeled_sample.y_true])


class ShortMemInstance:
    """Instance of a point associated with a timestamp. Used for the buffer memory which stores the unknown samples.
    
    Attributes
    ----------
    point : numpy.ndarray
        The coordinates of the point
    timestamp : int
        The timestamp the point was added/treated
    y_true : int
        The true value of the class
    """
    def __init__(self, point, timestamp, y_true=None):
        self.point = point
        self.timestamp = timestamp
        self.y_true = y_true

    def __eq__(self, other):
        """Elements are equal if they have the same values for all variables.
        This currently does not consider the timestamp.

        Parameters
        ----------
        other : ShortMemInstance
            Other instance to compared to

        Returns
        -------
        bool
            If the instances are equals or not
        """
        if type(other) == np.ndarray:
            return np.all(self.point == other)