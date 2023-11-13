import numpy as np
import copy
import random

from streamndr.utils.data_structure import MicroCluster, ShortMemInstance, ImpurityBasedCluster
from streamndr.utils.cluster_utils import get_closest_clusters

__all__ = ["MicroCluster", "ShortMemInstance"]

class MCIKMeans():
    """Implementation of K-Means with Minimization of Cluster Impurity (MCI-Kmeans), as described in [1].

    This algorithm implements a semi-supervised version of K-Means, that aims to minimize the intra-cluster dispersion while also minimizing the impurity of each cluster.

    [1] Masud, Mohammad M., et al. "A practical approach to classify evolving data streams: Training with limited amount of labeled data." 
    2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to generate
    max_iter : int
        Maximum number of iterations of the M-Step
    conditional_mode_max_iter : int
        Maximum number of iterations of the E-Step
    random_state : int
        Seed for the random number generation. Makes the algorithm deterministic if a number is provided.

    Attributes
    ----------
    clusters : dict
        Dictionary containing each cluster with their label as key
    cluster_centers_ : numpy.ndarray
        Array containing the coordinates of the cluster centers
    labels_ : numpy.ndarray
        Labels of each point
    """
    def __init__(self,
                 n_clusters=8,
                 max_iter=300,
                 conditional_mode_max_iter=300,
                 random_state=None):
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.conditional_mode_max_iter = conditional_mode_max_iter
        self.random_state = random_state

        if random_state != None:
            random.seed(random_state)

        self.clusters = []

    def fit(self, X, y):
        """Compute MCI-Kmeans clustering.

        Parameters
        ----------
        X : numpy.ndarray
            Samples
        y : list of int
            Labels of the samples, expects -1 if the label is not known

        Returns
        -------
        MCIKmeans
            Fitted estimator
        """
        y = np.array(y)

        samples_per_class = {}
        number_of_centroids = {}
        unlabeled_samples = []
        
        nb_labeled_samples = len(X[y!=-1])
        remaining = 0
        for label in np.unique(y):
            if label == -1:
                unlabeled_samples = X[y==-1]
                continue

            samples_per_class[label] = X[y==label]

            weight = self.n_clusters * len(samples_per_class[label]) / nb_labeled_samples
            remaining += weight - round(weight)
            number_of_centroids[label] = round(weight)

        while(remaining > 0):
            #Find the label with the smallest number of centroids and add the remainder to it
            key_with_min_value = min(number_of_centroids, key=lambda i: number_of_centroids[i])
            number_of_centroids[key_with_min_value] += 1

            remaining -= 1

        centroids = []
        for label in samples_per_class:
            centroids.extend(self._init_centroids(samples_per_class[label], number_of_centroids[label]))

            if (len(centroids) < number_of_centroids[label]) and (len(unlabeled_samples) > 0):
                filling_samples = copy.deepcopy(unlabeled_samples)

                while ((len(centroids) < number_of_centroids[label]) and (len(filling_samples) > 0)):
                    choice = filling_samples.pop(random.randrange(len(filling_samples)))
                    centroids.append(choice)

        for i in range(len(centroids)):
            self.clusters.append(ImpurityBasedCluster(i, centroids[i]))

        iterations = 0
        changing = True

        while changing and iterations < self.max_iter:
            changing = self._iterative_conditional_mode(samples_per_class, unlabeled_samples)

            for cluster in self.clusters:
                if cluster.n > 0:
                    cluster.update_properties()

            iterations += 1

        self.cluster_centers_ = np.array([cluster.centroid for cluster in self.clusters])
        self.labels_ = self.predict(X)

        return self


    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : numpy.ndarray
            Samples to predict

        Returns
        -------
        numpy.ndarray
            Index of the cluster each sample belongs to
        """
        labels, _ = get_closest_clusters(X, [cluster.centroid for cluster in self.clusters])

        return labels
    
    def fit_predict(self, X, y):
        """Compute cluster centers and predict cluster index for each sample. Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        X : numpy.ndarray
            Samples
        y : list of int
            Labels of the samples, expects -1 if the label is not known

        Returns
        -------
        numpy.ndarray
            Index of the cluster each sample belongs to
        """
        return self.fit(X, y).labels_

    def _init_centroids(self, samples, numbers_of_centroids):
        centroids = []

        if len(samples) <= numbers_of_centroids:
            centroids.extend(samples)
            return centroids

        candidates = copy.deepcopy(samples).tolist()

        for i in range(numbers_of_centroids):
            selected = candidates.pop(random.randrange(len(candidates)))

            centroids.append(selected)

        return centroids

    def _iterative_conditional_mode(self, samples_per_class, unlabeled_samples):
        
        _labeled_samples = []
        for key, value in samples_per_class.items():
            _labeled_samples.extend([ShortMemInstance(x, None, key) for x in value])

        _unlabeled_samples = [ShortMemInstance(x, None, -1) for x in unlabeled_samples]

        iterations = 0
        changed = True
        no_change = True

        while iterations < self.conditional_mode_max_iter and changed:
            total_nb_samples = len(_labeled_samples) + len(_unlabeled_samples)

            iterations += 1
            changed = False

            for i in range(total_nb_samples):
                sample = None

                if (len(_labeled_samples) > 0) and ((len(unlabeled_samples) == 0) or bool(random.getrandbits(1))):
                    sample = _labeled_samples.pop(random.randrange(len(_labeled_samples)))

                else:
                    sample = _unlabeled_samples.pop(random.randrange(len(_unlabeled_samples)))

                previous_cluster_id = sample.timestamp

                if previous_cluster_id is not None:
                    self.clusters[previous_cluster_id].remove_sample(sample)
                    sample.timestamp = None

                distances = np.linalg.norm([cluster.centroid for cluster in self.clusters] - sample.point, axis=1)

                if sample.y_true != -1:
                    entropies = np.array([cluster.entropy for cluster in self.clusters])
                    dissimilarities = np.array([cluster.dissimilarity_count(sample) for cluster in self.clusters])
                    distances = distances * (1 + entropies * dissimilarities)
                
                chosen_cluster = np.argmin(distances)


                self.clusters[chosen_cluster].add_sample(sample)
                sample.timestamp = chosen_cluster

                self.clusters[chosen_cluster].update_entropy()

                if self.clusters[chosen_cluster].label != previous_cluster_id:
                    changed = True
                    no_change = False

        return not no_change