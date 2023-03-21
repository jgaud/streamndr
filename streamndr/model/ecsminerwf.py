import numpy as np
import pandas as pd
import math
from collections import Counter

from river import base

from sklearn.cluster import KMeans

from streamndr.utils.data_structure import MicroCluster, ShortMemInstance

__all__ = ["ECSMinerWF"]

class ECSMinerWF(base.MiniBatchClassifier):
    """Implementation of the ECSMinerWF (ECSMiner without feedback) algorithm for novelty detection.

    Parameters
    ----------
    K : int
        Number of pseudopoints per classifier. In other words, it is the number of K cluster for the clustering algorithm.
    min_examples_cluster : int
        Minimum number of examples to declare a novel class 
    ensemble_size : int
        Number of classifiers to use to create the ensemble
    verbose : int
        Controls the level of verbosity, the higher, the more messages are displayed. Can be '1', '2', or '3'.
    random_state : int
        Seed for the random number generation. Makes the algorithm deterministic if a number is provided.

    Attributes
    ----------
    MAX_MEMORY_SIZE : int
        Constant used to determine the maximum number of rows used by numpy for the computation of the closest clusters. A higher number is faster but takes more memory.
    models : list of list of MicroCluster
        List containing the models created in the offline phase. In other words, it contains multiple lists of MicroClusters.
    novel_models : list of MicroCluster
        Contains the clusters representing novel classes, added during the online phase
    sample_counter : int
        Number of samples treated, used by the forgetting mechanism
    short_mem : list of ShortMemInstance
        Buffer memory containing the samples labeled as unknown temporarily for the novelty detection process
    last_nd : int
        Timestamp when the last novelty detection was performed. Used to determine if a new novelty detection should be performed.
    before_offline_phase : bool
        Whether or not the algorithm was initialized (offline phase). The algorithm needs to first be initialized to be used in an online fashion.
    """

    MAX_MEMORY_SIZE = 50000
    
    def __init__(self,
                 K=50, 
                 min_examples_cluster=50, #Number of instances requried to declare a novel class 
                 ensemble_size=6, 
                 verbose=0,
                 random_state=None):
        
        super().__init__()
        self.K = K
        self.min_examples_cluster = min_examples_cluster
        self.ensemble_size = ensemble_size
        self.verbose = verbose
        self.random_state = random_state

        self.models = []
        self.novel_models = []
        self.sample_counter = 0
        self.short_mem = [] #Potential novel class instances
        self.last_nd = -self.min_examples_cluster #No novelty detection performed yet
        self.before_offline_phase = True
        
    def learn_one(self, x, y, w=1.0):
        #Function used by river algorithms to learn one sample. It is not applicable to this algorithm since the offline phase requires all samples
        #to arrive at once. It is only added as to follow River's API.
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
        ECSMinerWF
            Itself
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        self.chunk_size = math.ceil(len(X)/self.ensemble_size)
        
        # in offline phase, consider all instances arriving at the same time in the microclusters:
        timestamp = len(X)
        
        #Separate data into (ensemble_size) chunks
        for i in range(0, self.ensemble_size):
            X_chunk = X[i:i+self.chunk_size]
            y_chunk = y[i:i+self.chunk_size]
            
            self.models.append(self._generate_microclusters(X_chunk, y_chunk, timestamp, self.K, min_samples=3)) #As per ECSMiner paper, any microcluster with less than 3 instances is discarded
                    
        self.before_offline_phase = False
        
        return self
    
    def predict_one(self, X):
        """Represents the online phase. Equivalent to predict_many() with only one sample. Receives only one sample, predict its label and adds 
        it to the cluster if it is a known class. Otherwise, if it's unknown, it is added to the short term memory and novelty detection is 
        performed once the trigger has been reached (min_examples_cluster).

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
        Otherwise, if it's unknown, it is added to the short term memory and novelty detection is performed once the trigger has been reached (min_examples_cluster).

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
         
        closest_model_cluster, y_preds = self._majority_voting(X)
        
        if len(self.novel_models) > 0: #We have novel clusters in our list
            novel_closest_clusters, _ = self._get_closest_clusters(X, [microcluster.centroid for microcluster in self.novel_models])
        
        pred_label = []
        for i in range(len(X)):
            self.sample_counter += 1
            
            closest_cluster = self.models[closest_model_cluster[i][0]][closest_model_cluster[i][1]]
            
            self._filter_buffer()
            
            if closest_cluster.distance_to_centroid(X[i]) <= closest_cluster.max_distance: # classify with the label from majority voting
                pred_label.append(y_preds[i])
                closest_cluster.update_cluster(X[i], self.sample_counter, False)
                
            elif (len(self.novel_models) > 0) and (self.novel_models[novel_closest_clusters[i]].distance_to_centroid(X[i]) <= closest_cluster.max_distance): #One of our novel cluster can explain our sample
                pred_label.append(self.novel_models[novel_closest_clusters[i]].label)
                self.novel_models[novel_closest_clusters[i]].update_cluster(X[i], self.sample_counter, False)
                
            else: #Classify as unknown
                pred_label.append(-1)
                self.short_mem.append(ShortMemInstance(X[i], self.sample_counter))

                if len(self.short_mem) > self.min_examples_cluster and (self.last_nd + self.min_examples_cluster) <= self.sample_counter:
                    self.last_nd = self.sample_counter

                    novel_clusters = self._novelty_detect()

                    if novel_clusters is not None: #We have novelty clusters
                        for novel_cluster in novel_clusters:
                            max_label_ensemble = max([cluster.label for model in self.models for cluster in model])
                            
                            max_label_novel = max([cluster.label for cluster in self.novel_models]) if len(self.novel_models) > 0 else -1
                            
                            novel_cluster.label = max(max_label_ensemble, max_label_novel) + 1
                            
                            if self.verbose > 0: print("Novel cluster detected: ", novel_cluster.small_str())

                            #Add novel cluster to our novel models list
                            self.novel_models.append(novel_cluster)

                            #Remove instances from the buffer
                            for instance in novel_cluster.instances:
                                self.short_mem.remove(instance)
                    
        return np.array(pred_label)
    
    def predict_proba_one(self,X):
        #Function used by river algorithms to get the probability of the prediction. It is not applicable to this algorithm since it only predicts labels. 
        #It is only added as to follow River's API.
        pass
    
    def predict_proba_many(self, X):
        #Function used by river algorithms to get the probability of the predictions. It is not applicable to this algorithm since it only predicts labels. 
        #It is only added as to follow River's API.
        pass
    
    def _generate_microclusters(self, X, y, timestamp, K, keep_instances=False, min_samples=0):
        clf = KMeans(n_clusters=K, n_init='auto', random_state=self.random_state).fit(X)
        labels = clf.labels_

        microclusters = []
        for microcluster in np.unique(labels):
            cluster_instances = X[labels == microcluster]
            y_cluster_instances = y[labels == microcluster]
            
            values, counts = np.unique(y_cluster_instances, return_counts=True)
            most_common_y = values[np.argmax(counts)]

            if len(cluster_instances) >= min_samples:
                mc = MicroCluster(most_common_y, instances=cluster_instances, timestamp=timestamp, keep_instances=keep_instances)
                microclusters.append(mc)
        
        return microclusters
    
    def _majority_voting(self, X):
        closest_clusters = []
        labels = []
        dists = []
        
        for model in self.models:
            closest_clusters_model, dist = self._get_closest_clusters(X, [microcluster.centroid for microcluster in model])
            closest_clusters.append(closest_clusters_model)
            labels.append([model[closest_cluster].label for closest_cluster in closest_clusters_model])
            dists.append(dist) 
        
        best_models = np.argmin(dists, axis=0)
        
        closest_model_cluster = []
        for i in range(len(X)):
            closest_model_cluster.append((best_models[i], closest_clusters[best_models[i]][i]))
            
        return closest_model_cluster, [Counter(col).most_common(1)[0][0] for col in zip(*labels)]
        
        
    def _get_closest_clusters(self, X, centroids):   
        
        if len(centroids) == 0:
            print("No clusters")
            return
            
        centroids = np.array(centroids)
        norm_dists = np.zeros((X.shape[0],centroids.shape[0]))

        # Cut into batches if there are too many samples to save on memory
        for idx in range(math.ceil(X.shape[0]/ECSMinerWF.MAX_MEMORY_SIZE)):
            sl = slice(idx*ECSMinerWF.MAX_MEMORY_SIZE, (idx+1)*ECSMinerWF.MAX_MEMORY_SIZE)
            norm_dists[sl] = np.linalg.norm(np.subtract(X[sl, :, None], np.transpose(centroids)), axis=1)

        return np.argmin(norm_dists, axis=1), np.amin(norm_dists, axis=1)
        
    def _novelty_detect(self):
        if self.verbose > 0: print("Novelty detection started")
        
        X = np.array([instance.point for instance in self.short_mem])
        new_class_vote = 0
        
        #Creating F-pseudopoints
        K0 = math.ceil(self.K * (len(X) / self.chunk_size))
        K0 = max(K0, self.K)
        
        f_microclusters = self._generate_microclusters(X, np.array([-1] * len(X)), self.sample_counter, K0, keep_instances=True)
        f_microclusters_centroids = np.array([cl.centroid for cl in f_microclusters])
    
        potential_novel_clusters_idx = []
        #Computing qNSC for each model in our ensemble
        for model in self.models:
            qnscs = self._qnsc(f_microclusters_centroids, model)
            
            potential_clusters = []
            total_instances = 0
            for i, f_microcluster in enumerate(f_microclusters):
                if qnscs[i] > 0:
                    potential_clusters.append(f_microcluster)
                    total_instances += f_microcluster.n
                    potential_novel_clusters_idx.append(i)
            
            if total_instances > self.min_examples_cluster: new_class_vote += 1
                    
        
        if new_class_vote == len(self.models):
            #Get the indices of all clusters which had a positive qnsc for all models
            novel_clusters_idx = [item for item, count in Counter(potential_novel_clusters_idx).items() if count == len(self.models)]
            novel_clusters = [f_microclusters[i] for i in novel_clusters_idx]
            
            return novel_clusters
        
        else:
            return None
        
        
    def _qnsc(self, pseudopoints, model):
        #Calculate mean distance of all points between themselves
        dists = np.linalg.norm(pseudopoints - pseudopoints[:,None], axis=-1)
        dists[np.arange(dists.shape[0]), np.arange(dists.shape[0])] = np.nan
        mean_distances_between_points = np.nanmean(dists, axis=0)
        
        #Calculate minimum distance between points known cluster
        all_centroids = [microcluster.centroid for microcluster in model]
        _, minimum_distances_to_class = self._get_closest_clusters(pseudopoints, all_centroids)
        
        qnscs = (minimum_distances_to_class - mean_distances_between_points) / np.maximum(minimum_distances_to_class, mean_distances_between_points)
        
        return qnscs
    
    def _filter_buffer(self):
        for instance in self.short_mem:
            if (self.sample_counter - instance.timestamp > self.chunk_size): #We remove samples that have an age greater than the chunk size
                self.short_mem.remove(instance)
            else: #No need to iterate over the whole buffer since older elements are at the beginning
                break;
        
        