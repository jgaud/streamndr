import numpy as np
import pandas as pd
import math

from river import base

from clusopt_core.cluster import CluStream
from sklearn.cluster import KMeans

from streamndr.utils.data_structure import MicroCluster, ShortMemInstance


class Minas(base.MiniBatchClassifier):
    
    MAX_MEMORY_SIZE = 50000

    def __init__(self,
                 kini=3,
                 cluster_algorithm='kmeans',
                 random_state=None,
                 min_short_mem_trigger=10,
                 min_examples_cluster=10,
                 threshold_strategy=1,
                 threshold_factor=1.1,
                 window_size=100,
                 update_summary=False,
                 verbose=0):
        super().__init__()
        self.kini = kini
        self.random_state = random_state

        accepted_algos = ['kmeans','clustream']
        if cluster_algorithm not in accepted_algos:
            print('Available algorithms: {}'.format(', '.join(accepted_algos)))
        else:
            self.cluster_algorithm = cluster_algorithm

        self.microclusters = []  # list of microclusters
        self.before_offline_phase = True

        self.short_mem = []
        self.sleep_mem = []
        self.min_short_mem_trigger = min_short_mem_trigger
        self.min_examples_cluster = min_examples_cluster
        self.threshold_strategy = threshold_strategy
        self.threshold_factor = threshold_factor
        self.window_size = window_size
        self.update_summary = update_summary
        self.verbose = verbose
        self.sample_counter = 0  # to be used with window_size
    
    def learn_one(self, x, y, w=1.0):
        """X is a Dictionary"""
        # Not applicable
        pass
        

    def learn_many(self, X, y, w=1.0):
        """X is a pandas DataFrame or Numpy Array"""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        self.microclusters = self._offline(X, y)
        self.before_offline_phase = False

        return self
    
    def predict_one(self, X):
        """X is a Dictionary""" 
        return self.predict_many(np.array(list(X.values()))[None,:])

    def predict_many(self, X):
        """X is a pandas DataFrame or Numpy Array"""
        if self.before_offline_phase:
            raise Exception("Model must be fitted first")
        
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy() #Converting DataFrame to numpy array
        
        # Finding closest clusters for received samples
        closest_clusters = self._get_closest_clusters(X, [microcluster.centroid for microcluster in self.microclusters])
        
        pred_label = []
        
        for i in range(len(closest_clusters)):
            self.sample_counter += 1
            closest_cluster = self.microclusters[closest_clusters[i]]

            if closest_cluster.encompasses(X[i]):  # classify in this cluster
                pred_label.append(closest_cluster.label)

                closest_cluster.update_cluster(X[i], closest_cluster.label, self.sample_counter, self.update_summary)

            else:  # classify as unknown
                pred_label.append(-1)
                self.short_mem.append(ShortMemInstance(X[i], self.sample_counter))
                
                if self.verbose > 1:
                    print('Memory length: ', len(self.short_mem))
                elif self.verbose > 0:
                    if len(self.short_mem) % 100 == 0: print('Memory length: ', len(self.short_mem))
                    
                if len(self.short_mem) >= self.min_short_mem_trigger:
                    self._novelty_detect()
   
        # forgetting mechanism
        if self.sample_counter % self.window_size == 0:
            self._trigger_forget()

        
        return np.array(pred_label)

    def predict_proba_one(self,X):
        # Not applicable
        pass
    
    def predict_proba_many(self, X):
        # Not applicable
        pass

    def _offline(self, X_train, y_train):

        microclusters = []
        # in offline phase, consider all instances arriving at the same time in the microclusters:
        timestamp = len(X_train)
        
        for y_class in np.unique(y_train):
            # subset with instances from each class
            X_class = X_train[y_train == y_class]

            if self.cluster_algorithm == 'kmeans':
                class_cluster_clf = KMeans(n_clusters=self.kini, n_init='auto',
                                            random_state=self.random_state)
                class_cluster_clf.fit(X_class)
                labels = class_cluster_clf.labels_

            else:
                class_cluster_clf = CluStream(m=self.kini)
                class_cluster_clf.init_offline(X_class, seed=self.random_state)
                
                cluster_centers = class_cluster_clf.get_partial_cluster_centers()

                labels = self._get_closest_clusters(X_class, cluster_centers)

            for class_cluster in np.unique(labels):
                # get instances in cluster
                cluster_instances = X_class[labels == class_cluster]

                microclusters.append(
                    MicroCluster(y_class, cluster_instances, timestamp)
                )

        return microclusters

    def _novelty_detect(self):
        if self.verbose > 0: print("Novelty detection started")
        possible_clusters = []
        X = np.array([instance.point for instance in self.short_mem])

        if self.cluster_algorithm == 'kmeans':
            cluster_clf = KMeans(n_clusters=self.kini, n_init='auto',
                                 random_state=self.random_state)
            cluster_clf.fit(X)
            labels = cluster_clf.labels_

        else:
            cluster_clf = CluStream(m=self.kini)
            cluster_clf.init_offline(X, seed=self.random_state)
            
            cluster_centers = cluster_clf.get_partial_cluster_centers()

            labels = self._get_closest_clusters(X, cluster_centers)

            

        for cluster_label in np.unique(labels):
            cluster_instances = X[labels == cluster_label]
            possible_clusters.append(
                MicroCluster(-1, cluster_instances, self.sample_counter))
        
        for cluster in possible_clusters:
            if cluster.is_cohesive(self.microclusters) and cluster.is_representative(self.min_examples_cluster):
                closest_cluster = cluster.find_closest_cluster(self.microclusters)
                closest_distance = cluster.distance_to_centroid(closest_cluster.centroid)

                threshold = self._best_threshold(cluster, closest_cluster,
                                                self.threshold_strategy)

                # TODO make these ifs elifs cleaner
                if closest_distance <= threshold:  # the new microcluster is an extension
                    if self.verbose > 1:
                            print("Extension of cluster: ", closest_cluster)
                    elif self.verbose > 0:
                        print("Extension of cluster: ", closest_cluster.small_str())
                    
                    cluster.label = closest_cluster.label
                    
                elif self.sleep_mem:  # look in the sleep memory, if not empty
                    closest_cluster = cluster.find_closest_cluster(self.sleep_mem)
                    closest_distance = cluster.distance_to_centroid(closest_cluster.centroid)
                    
                    if closest_distance <= threshold:  # check again: the new microcluster is an extension
                        if self.verbose > 1:
                            print("Waking cluster: ", closest_cluster)
                        elif self.verbose > 0:
                            print("Waking cluster: ", closest_cluster.small_str())
                            
                        cluster.label = closest_cluster.label
                        # awake old cluster
                        self.sleep_mem.remove(closest_cluster)
                        closest_cluster.timestamp = self.sample_counter
                        self.microclusters.append(closest_cluster)
                        
                    else:  # the new microcluster is a novelty pattern
                        cluster.label = max([cluster.label for cluster in self.microclusters]) + 1
                        if self.verbose > 1:
                            print("Novel cluster: ", cluster)
                        elif self.verbose > 0:
                            print("Novel cluster: ", cluster.small_str())
                            
                else:  # the new microcluster is a novelty pattern
                    cluster.label = max([cluster.label for cluster in self.microclusters]) + 1
                    if self.verbose > 1:
                            print("Novel cluster: ", cluster)
                    elif self.verbose > 0:
                        print("Novel cluster: ", cluster.small_str())
                        
                # add the new cluster to the model
                self.microclusters.append(cluster)

                # remove these examples from short term memory
                for instance in cluster.instances:
                    self.short_mem.remove(instance)

    def _best_threshold(self, new_cluster, closest_cluster, strategy):
        def run_strategy_1():
            factor_1 = self.threshold_factor
            # factor_1 = 5  # good for artificial, separated data sets
            return factor_1 * np.std(closest_cluster.distance_to_centroid(closest_cluster.instances))

        if strategy == 1:
            return run_strategy_1()
        else:
            factor_2 = factor_3 = self.threshold_factor
            # factor_2 = factor_3 = 1.2 # good for artificial, separated data sets
            clusters_same_class = self._get_clusters_in_class(closest_cluster.label)
            if len(clusters_same_class) == 1:
                return run_strategy_1()
            else:
                class_centroids = np.array([cluster.centroid for cluster in clusters_same_class])
                distances = closest_cluster.distance_to_centroid(class_centroids)
                if strategy == 2:
                    return factor_2 * np.max(distances)
                elif strategy == 3:
                    return factor_3 * np.mean(distances)
    
    def _get_closest_clusters(self, X, centroids):   
        
        if len(centroids) == 0:
            print("No clusters")
            return
            
        centroids = np.array(centroids)
        norm_dists = np.zeros((X.shape[0],centroids.shape[0]))

        # Cut into batches if there are too many samples to save on memory
        for idx in range(math.ceil(X.shape[0]/Minas.MAX_MEMORY_SIZE)):
            sl = slice(idx*Minas.MAX_MEMORY_SIZE, (idx+1)*Minas.MAX_MEMORY_SIZE)
            norm_dists[sl] = np.linalg.norm(np.subtract(X[sl, :, None], np.transpose(centroids)), axis=1)

        return np.argmin(norm_dists, axis=1)

    def _get_clusters_in_class(self, label):
        return [cluster for cluster in self.microclusters if cluster.label == label]

    def _trigger_forget(self):
        for cluster in self.microclusters:
            if cluster.timestamp < self.sample_counter - self.window_size:
                if self.verbose > 1:
                    print("Forgetting cluster: ", cluster)
                elif self.verbose > 0:
                    print("Forgetting cluster: ", cluster.small_str())
                    
                self.sleep_mem.append(cluster)
                self.microclusters.remove(cluster)
        for instance in self.short_mem:
            if instance.timestamp < self.sample_counter - self.window_size:
                self.short_mem.remove(instance)

    def confusion_matrix(self, X_test, y_test):
        """Creates a confusion matrix.

        It must be run on a fitted classifier that has already seen the examples in the test set.

        Parameters
        ----------
        X_test : numpy.ndarray
            The set of data samples to predict the class labels for.
        y_test : numpy.ndarray
            The set of class labels for the data samples.

        Returns
        -------
        river.metrics.ConfusionMatrix

        """
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()

        closest_clusters = self._get_closest_clusters(X_test, [microcluster.centroid for microcluster in self.microclusters])

        conf_matrix = metrics.ConfusionMatrix()
        
        for i in range(len(closest_clusters)):
            closest_cluster = self.microclusters[closest_clusters[i]]

            if closest_cluster.encompasses(X_test[i]):  # classify in this cluster
                conf_matrix = conf_matrix.update(y_test[i], closest_cluster.label)

            else:  # classify as unknown
                conf_matrix = conf_matrix.update(y_test[i], -1)

        return conf_matrix