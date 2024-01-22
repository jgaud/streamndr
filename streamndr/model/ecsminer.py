import numpy as np
import pandas as pd
import math
from collections import Counter

from sklearn.metrics import accuracy_score
from streamndr.model.noveltydetectionclassifier import NoveltyDetectionClassifier

from streamndr.utils.data_structure import ShortMemInstance, ClusterModel, ShortMem
from streamndr.utils.cluster_utils import *

__all__ = ["ECSMiner"]

class ECSMiner(NoveltyDetectionClassifier):
    """Implementation of the ECSMiner algorithm for novelty detection [1].

    [1] Masud, Mohammad, et al. "Classification and novel class detection in concept-drifting data streams under time constraints." 
    IEEE Transactions on knowledge and data engineering 23.6 (2010): 859-874.

    Parameters
    ----------
    K : int
        Number of pseudopoints per classifier. In other words, it is the number of K cluster for the clustering algorithm.
    min_examples_cluster : int
        Minimum number of examples to declare a novel class 
    ensemble_size : int
        Number of classifiers to use to create the ensemble
    T_l : int
        Labeling time constraint
    verbose : int
        Controls the level of verbosity, the higher, the more messages are displayed. Can be '1', '2', or '3'.
    random_state : int
        Seed for the random number generation. Makes the algorithm deterministic if a number is provided.
    init_algorithm : string
        String containing the clustering algorithm to use to initialize the clusters, supports 'kmeans' and 'mcikmeans'

    Attributes
    ----------
    models : list of ClusterModel
        List containing the models of the ensemble.
    nb_class_unknown : dict
        Tracks the number of samples of each true class value currently in the unknown buffer (short_mem). Used to compute the unknown rate.
    class_sample_counter : dict
        Tracks the total number of samples of each true class value seen in the stream. Used to compute the unknown rate.
    sample_counter : int
        Number of samples treated, used by the forgetting mechanism
    short_mem : list of ShortMemInstance
        Buffer memory containing the samples labeled as unknown temporarily for the novelty detection process
    unlabeled_buffer : list of ShortMemInstance
        Buffer memory containing the unlabeled data points until they are labeled
    labeled_buffer : list of ShortMemInstance
        Buffer memory containing the labeled data points until they are used for training
    last_nd : int
        Timestamp when the last novelty detection was performed. Used to determine if a new novelty detection should be performed.
    before_offline_phase : bool
        Whether or not the algorithm was initialized (offline phase). The algorithm needs to first be initialized to be used in an online fashion.
    """

    def __init__(self,
                 K=50, 
                 min_examples_cluster=50, #Number of instances requried to declare a novel class 
                 ensemble_size=6,
                 T_l=1000,
                 verbose=0,
                 random_state=None,
                 init_algorithm="mcikmeans"):
        
        super().__init__(verbose, random_state)
        self.K = K
        self.min_examples_cluster = min_examples_cluster
        self.ensemble_size = ensemble_size
        self.T_l = T_l

        accepted_algos = ['kmeans','mcikmeans']
        if init_algorithm not in accepted_algos:
            print('Available algorithms: {}'.format(', '.join(accepted_algos)))
        else:
            self.init_algorithm = init_algorithm

        self.models = []
        self.novel_models = []
        self.short_mem = ShortMem() #Potential novel class instances
        self.unlabeled_buffer = [] #Unlabeled data points
        self.labeled_buffer = [] #Labeled data points for training
        self.last_nd = -self.min_examples_cluster #No novelty detection performed yet

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
        ECSMiner
            Fitted estimator
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        #The chunk size is equal to the number of samples given to learn divided by the size of the ensemble as one model is trained per chunk
        self.chunk_size = math.ceil(len(X)/self.ensemble_size)
        
        # in offline phase, consider all instances arriving at the same time in the microclusters:
        timestamp = len(X)
        
        #Separate data into {ensemble_size} chunks
        for i in range(0, self.ensemble_size):
            X_chunk = X[i:i+self.chunk_size]
            y_chunk = y[i:i+self.chunk_size]
            
            microclusters = generate_microclusters(X_chunk, y_chunk, timestamp, self.K, min_samples=3, algorithm=self.init_algorithm, random_state=self.random_state) #As per ECSMiner paper, any microcluster with less than 3 instances is discarded
            
            model = ClusterModel(microclusters, np.unique(y_chunk))
            
            if len(microclusters) > 0:
                self.models.append(model)
                    
        self.before_offline_phase = False
        
        return self
    
    def predict_one(self, X, y=None):
        """Represents the online phase. Equivalent to predict_many() with only one sample. Receives only one sample, predict its label and adds 
        it to the cluster if it is a known class. Otherwise, if it's unknown, it is added to the short term memory and novelty detection is 
        performed once the trigger has been reached (min_examples_cluster).

        Parameters
        ----------
        X : dict
            Sample
        y : int
            True y value of the sample.

        Returns
        -------
        numpy.ndarray
            Label predicted for the given sample, predicts -1 if labeled as unknown
        """
        return self.predict_many(np.array(list(X.values()))[None,:], [y])
            

    def predict_many(self, X, y=None):
        """Represents the online phase. Receives multiple samples, for each sample predict its label and adds it to the cluster if it is a known class. 
        Otherwise, if it's unknown, it is added to the short term memory and novelty detection is performed once the trigger has been reached (min_examples_cluster).

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Samples
        y : list of int
            True y values of the samples.

        Returns
        -------
        numpy.ndarray
            Array of length len(X) containing the predicted labels, predicts -1 if the corresponding sample is labeled as unknown

        Raises
        ------
        Exception
            If the model has not been trained first with learn_many() (offline phase)
        """
        if self.before_offline_phase:
            raise Exception("Model must be fitted first")
        
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy() #Converting DataFrame to numpy array
        
        f_outliers = check_f_outlier(X, self.models)
        closest_model_cluster, y_preds = majority_voting(X, self.models)
        
        pred_label = []
        for i in range(len(X)):
            self.sample_counter += 1
            if y is not None:
                if y[i] not in self.class_sample_counter:
                    self.class_sample_counter[y[i]] = 1
                else:
                    self.class_sample_counter[y[i]] += 1
            
            #Get the closest microcluster with our list of tuples self.models[closest_model_index][closest_cluster_index]
            closest_cluster = self.models[closest_model_cluster[i][0]].microclusters[closest_model_cluster[i][1]]
            
            self._filter_buffer()
            
            #If X is not an F-outlier (inside the closest cluster radius), then we classify it with the label from the majority voting
            if not f_outliers[i]:
                pred_label.append(y_preds[i])
                closest_cluster.update_cluster(X[i], self.sample_counter, False)
                
            else: #X is an F-outlier (outside the boundary of all classifiers)
                pred_label.append(-1)

                if y is not None:
                    self.short_mem.append(ShortMemInstance(X[i], self.sample_counter, y[i]))
                    if y[i] not in self.nb_class_unknown:
                        self.nb_class_unknown[y[i]] = 1
                    else:
                        self.nb_class_unknown[y[i]] += 1
                else:
                    self.short_mem.append(ShortMemInstance(X[i], self.sample_counter))

                #Check if the length of the buffer is at least {min_examples_cluster} and that the last check was at least {min_examples_cluster} samples ago
                if (len(self.short_mem) > self.min_examples_cluster) and ((self.last_nd + self.min_examples_cluster) <= self.sample_counter):
                    self.last_nd = self.sample_counter

                    #Find the list of novel clusters, if any
                    novel_clusters = self._novelty_detect()

                    if novel_clusters is not None: #We have novelty clusters
                        for novel_cluster in novel_clusters:
                            if self.verbose > 0: print("Novel cluster detected: ", novel_cluster.small_str())
                            

                            #Remove instances from the buffer
                            for instance in novel_cluster.instances:
                                self._remove_sample_from_short_mem(self.short_mem.index(np.array(instance)))

            #Enqueue X in the unlabeled buffer
            self.unlabeled_buffer.append(ShortMemInstance(X[i], self.sample_counter, y[i]))

            #If we have reached the labeling time constraint, we need to label the oldest instance
            if len(self.unlabeled_buffer) > self.T_l:
                instance_to_label = self.unlabeled_buffer.pop(0)
                self.labeled_buffer.append(instance_to_label)

                #Remove from short_term_memory if it was in there and is a known class
                if any(instance_to_label.y_true in model.labels for model in self.models) and (instance_to_label in self.short_mem.get_all_instances()):
                    self.short_mem.remove(instance_to_label)

                if len(self.labeled_buffer) == self.chunk_size:
                    if self.verbose > 0: 
                            print("Labeled buffer reached chunk size, creating new model...")
                    #Create a new model on the labeled buffer
                    points = np.vstack([inst.point for inst in self.labeled_buffer])
                    true_labels = np.array([inst.y_true for inst in self.labeled_buffer])
                    new_microclusters = generate_microclusters(points, true_labels, self.sample_counter, self.K, min_samples=3, algorithm=self.init_algorithm, random_state=self.random_state)
                    new_model = ClusterModel(new_microclusters, np.unique(true_labels))

                    #Update the existing ensemble
                    self.models.append(new_model)

                    #Check if the oldest classifier has a class not included in any of the new models
                    labels_of_oldest_model = set(self.models[0].labels)
                    labels_of_new_models = set(element for sublist in self.models[1:] for element in sublist.labels)

                    if labels_of_oldest_model - labels_of_new_models:
                        if self.verbose > 0: 
                            print("Oldest model includes label not in new models, forgetting...")
                        #Remove oldest model
                        self.models.pop(0)


                    if len(self.models) > self.ensemble_size:
                        accuracies = []
                        #Iterate over all of the models in the ensemble and compute the accuracy of each model on the labeled buffer
                        for model in self.models:
                            #Get the model's closest microcluster
                            closest_clusters_model, _ = get_closest_clusters(points, [microcluster.centroid for microcluster in model.microclusters])
                            accuracies.append(accuracy_score(true_labels, [model.microclusters[closest_cluster].label for closest_cluster in closest_clusters_model]))
                        
                        #Remove the model with the lowest accuracy
                        self.models.pop(np.argmin(accuracies))
                    
                    #Clear the labeled buffer
                    self.labeled_buffer.clear()
                    
        return np.array(pred_label)
      
    def _novelty_detect(self):
        if self.verbose > 1: print("Novelty detection started")
        X = self.short_mem.get_all_points()
        new_class_vote = 0
        
        #Creating F-pseudopoints representing all F-outliers to speedup computation of qnsc
        K0 = round(self.K * (len(X) / self.chunk_size))
        K0 = max(K0, self.K)
        K0 = min(K0, len(X)) #Can't create K clusters if K is higher than the number of samples

        f_microclusters = generate_microclusters(X, np.array([-1] * len(X)), self.sample_counter, K0, keep_instances=True, min_samples=0, algorithm="kmeans", random_state=self.random_state)
        f_microclusters_centroids = np.array([cl.centroid for cl in f_microclusters])
    
        potential_novel_clusters_idx = []
        #Computing qNSC for each model in our ensemble
        for model in self.models:
            qnscs = qnsc(f_microclusters_centroids, model.microclusters, self.min_examples_cluster)
            potential_clusters = []
            total_instances = 0
            for i, f_microcluster in enumerate(f_microclusters):
                if qnscs[i] > 0:
                    potential_clusters.append(f_microcluster)
                    total_instances += f_microcluster.n
                    potential_novel_clusters_idx.append(i)
            if total_instances > 0 and self.verbose > 1:
                print(f"Total instances in F-outliers: {total_instances}")
            
            if total_instances > self.min_examples_cluster: 
                new_class_vote += 1

        if new_class_vote == len(self.models):
            #Get the indices of all clusters which had a positive qnsc for all models
            novel_clusters_idx = [item for item, count in Counter(potential_novel_clusters_idx).items() if count == len(self.models)]
            novel_clusters = [f_microclusters[i] for i in novel_clusters_idx]
            
            return novel_clusters
        
        else:
            return None
        
    def _filter_buffer(self):
        closest_model_cluster = majority_voting(self.short_mem.get_all_points(), self.models, False)


        for i, instance in enumerate(self.short_mem.get_all_instances()):
            closest_cluster = self.models[closest_model_cluster[i][0]].microclusters[closest_model_cluster[i][1]]

            if ((self.sample_counter - instance.timestamp > self.chunk_size) #The instance has an age greater than the chunk size
                or (closest_cluster.distance_to_centroid(instance.point) <= closest_cluster.max_distance)): #The instance is no longer an F-outlier 

                self._remove_sample_from_short_mem(self.short_mem.index(instance))
