import numpy as np
import pandas as pd
import math

from river import base

from clusopt_core.cluster import CluStream
from sklearn.cluster import KMeans

from streamndr.utils.data_structure import MicroCluster, ShortMemInstance

__all__ = ["Minas"]

class Minas(base.MiniBatchClassifier):
    """Implementation of the MINAS algorithm for novelty detection.

    Parameters
    ----------
    kini : int
        Number of K clusters for the clustering (KMeans or Clustream) algorithm
    cluster_algorithm : str
        String containing the clustering algorithm to use, supports 'kmeans' and 'clustream'
    random_state : int
        Seed for the random number generation. Makes the algorithm deterministic if a number is provided.
    min_short_mem_trigger : int
        Minimum number of samples in the short term memory to trigger the novelty detection process
    min_examples_cluster : int
        Minimum number of samples to from a cluster
    threshold_strategy : int
        Strategy to use to compute the threshold. Can be '1', '2', or '3' as described in the MINAS paper.
    threshold_factor : float
        Factor for the threshold computation
    window_size : int
        Number of samples used by the forgetting mechanism
    update_summary : bool
        Whether or not the microcluster's properties are updated when a new point is added to it
    verbose : int
        Controls the level of verbosity, the higher, the more messages are displayed. Can be '1', '2', or '3'.

    Attributes
    ----------
    MAX_MEMORY_SIZE : int
        Constant used to determine the maximum number of rows used by numpy for the computation of the closest clusters. A higher number is faster but takes more memory.
    before_offline_phase : bool
        Whether or not the algorithm was initialized (offline phase). The algorithm needs to first be initialized to be used in an online fashion.
    short_mem : list of ShortMemInstance
        Buffer memory containing the samples labeled as unknown temporarily for the novelty detection process
    sleep_mem : list of MicroCluster
        Microclusters that have not have any new points added from the strem for a period of time are temporarily moved to a sleep memory
    sample_counter : int
        Number of samples treated, used by the forgetting mechanism
    """

    
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
        """Function used by river algorithms to learn one sample. It is not applicable to this algorithm since the offline phase requires all samples
        to arrive at once. It is only added as to follow River's API.

        Parameters
        ----------
        x : dict
            Sample
        y : int
            Label of the given sample
        w : float, optional
            Weight, not used, by default 1.0
        """
        # Not applicable
        pass
        

    def learn_many(self, X, y, w=1.0):
        """Represents the offline phase of the algorithm. Receives a number of samples and their given labels and learns all of the known classes.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Samples to be learned by the model
        y : list of int
            Labels corresponding to the given samples, must be the same length as the number of samples
        w : float, optional
            Weights, not used, by default 1.0

        Returns
        -------
        Minas
            Itself
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        self.microclusters = self._offline(X, y)
        self.before_offline_phase = False

        return self
    
    def predict_one(self, X):
        """Represents the online phase. Equivalent to predict_many() with only one sample. Receives only one sample, predict its label and adds 
        it to the cluster if it is a known class. Otherwise, if it's unknown, it is added to the short term memory and novelty detection is 
        performed once the trigger has been reached (min_short_mem_trigger).

        Parameters
        ----------
        X : dict
            Sample

        Returns
        -------
        numpy.ndarray
            Label predicted for the given sample, predicts -1 if labeled as unknown
        """
        return self.predict_many(np.array(list(X.values()))[None,:])

    def predict_many(self, X):
        """Represents the online phase. Receives multiple samples, for each sample predict its label and adds it to the cluster if it is a known class. 
        Otherwise, if it's unknown, it is added to the short term memory and novelty detection is performed once the trigger has been reached (min_short_mem_trigger).

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Samples

        Returns
        -------
        numpy.ndarray
            Array of length len(X) containing the predicted labels, predicts -1 if the corresponding sample labeled as unknown

        Raises
        ------
        Exception
            If the model has not been trained first with learn_many() (offline phase)
        """
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

                closest_cluster.update_cluster(X[i], self.sample_counter, self.update_summary)

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
        #Function used by river algorithms to get the probability of the prediction. It is not applicable to this algorithm since it only predicts labels. 
        #It is only added as to follow River's API.
        pass
    
    def predict_proba_many(self, X):
        #Function used by river algorithms to get the probability of the predictions. It is not applicable to this algorithm since it only predicts labels. 
        #It is only added as to follow River's API.
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