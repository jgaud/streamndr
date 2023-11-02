import numpy as np
import copy
import random

from streamndr.utils.data_structure import MicroCluster, ShortMemInstance, ImpurityBasedCluster
from streamndr.utils.cluster_utils import get_closest_clusters

__all__ = ["MicroCluster", "ShortMemInstance"]

class MCIKMeans():
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

        self.centroids = []
        self.clusters = {}

    def fit(self, X, y):
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

        for label in samples_per_class:
            self._init_centroids(samples_per_class[label], number_of_centroids[label])

            if (len(self.centroids) < number_of_centroids[label]) and (len(unlabeled_samples) > 0):
                filling_samples = copy.deepcopy(unlabeled_samples)

                while ((len(self.centroids) < number_of_centroids[label]) and (len(filling_samples) > 0)):
                    choice = random.choice(filling_samples)
                    self.centroids.append(choice)
                    filling_samples.remove(choice)

        for i in range(len(self.centroids)):
            self.clusters[i] = ImpurityBasedCluster(i, self.centroids[i])

        iterations = 0
        changing = True

        while changing and iterations < self.max_iter:
            changing = self._iterative_conditional_mode(samples_per_class, unlabeled_samples)

            for cluster in self.clusters.values():
                if cluster.size() > 0:
                    cluster.update_centroid()

            iterations += 1

        to_delete = []
        for id, cluster in self.clusters.items():
            if cluster.size() < 2:
                to_delete.append(id)

        for id in to_delete:
            del self.clusters[id]


    def predict(self, X):
        labels, _ = get_closest_clusters(X, [cluster.centroid for cluster in self.clusters.values()])

        return labels


    def _init_centroids(self, samples, numbers_of_centroids):
        if len(samples) <= numbers_of_centroids:
            self.centroids.extend(samples)
            return

        candidates = copy.deepcopy(samples).tolist()

        for i in range(numbers_of_centroids):
            selected = random.choice(candidates)
            candidates.remove(selected)

            self.centroids.append(selected)

    def _iterative_conditional_mode(self, samples_per_class, unlabeled_samples):
        
        _labeled_samples = []
        for key, value in samples_per_class.items():
            _labeled_samples.extend([ShortMemInstance(x, None, key) for x in value])

        _unlabeled_samples = [ShortMemInstance(x, None, None) for x in unlabeled_samples]

        iterations = 0
        changed = True
        no_change = True

        while iterations < self.conditional_mode_max_iter and changed:
            total_nb_samples = len(_labeled_samples) + len(_unlabeled_samples)

            iterations += 1
            changed = False

            for i in range(total_nb_samples):
                sample = None

                if (len(unlabeled_samples) == 0) or bool(random.getrandbits(1)):
                    sample = random.choice(_labeled_samples)
                    _labeled_samples.remove(sample)

                else:
                    sample = random.choice(_unlabeled_samples)
                    _unlabeled_samples.remove(sample)

                previous_cluster_id = sample.timestamp

                if previous_cluster_id is not None:
                    self.clusters[previous_cluster_id].remove_sample(sample)
                    sample.timestamp = None

                
                chosen_cluster = 0
                min_dist = self._get_distance_value(sample, self.clusters[0], sample.y_true != None)

                for i in range(1, len(self.clusters)):
                    if (self._get_distance_value(sample, self.clusters[i], sample.y_true != None)) < min_dist:
                        chosen_cluster = i

                self.clusters[chosen_cluster].add_sample(sample)
                sample.timestamp = self.clusters[chosen_cluster].id

                self.clusters[chosen_cluster].update_entropy()

                if self.clusters[chosen_cluster].id != previous_cluster_id:
                    changed = True
                    no_change = False

        return not no_change
    
    def _get_distance_value(self, sample, cluster, is_labeled):
        if is_labeled:
            return cluster.distance_to_centroid(sample.point) * (1 + cluster.entropy * cluster.dissimilarity_count(sample))
        
        else:
            return cluster.distance_to_centroid(sample.point)