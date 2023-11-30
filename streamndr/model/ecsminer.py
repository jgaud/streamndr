import numpy as np
import pandas as pd
import math
from collections import Counter

from river import base

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from streamndr.utils.mcikmeans import MCIKMeans
from streamndr.utils.data_structure import MicroCluster, ShortMemInstance, ClusterModel, ShortMem
from streamndr.utils.cluster_utils import *

__all__ = ["ECSMiner"]

class ECSMiner(base.MiniBatchClassifier):
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
        
        super().__init__()
        self.K = K
        self.min_examples_cluster = min_examples_cluster
        self.ensemble_size = ensemble_size
        self.T_l = T_l
        self.verbose = verbose
        self.random_state = random_state

        accepted_algos = ['kmeans','mcikmeans']
        if init_algorithm not in accepted_algos:
            print('Available algorithms: {}'.format(', '.join(accepted_algos)))
        else:
            self.init_algorithm = init_algorithm

        self.models = []
        self.novel_models = []
        self.nb_class_unknown = dict()
        self.class_sample_counter = dict()
        self.sample_counter = 0
        self.short_mem = ShortMem() #Potential novel class instances
        self.unlabeled_buffer = [] #Unlabeled data points
        self.labeled_buffer = [] #Labeled data points for training
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
            
            microclusters = self._generate_microclusters(X_chunk, y_chunk, timestamp, self.K, min_samples=3, algorithm=self.init_algorithm) #As per ECSMiner paper, any microcluster with less than 3 instances is discarded
            
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
        
        f_outliers = self._check_f_outlier(X, self.models)
        closest_model_cluster, y_preds = self._majority_voting(X)
        
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
                    new_microclusters = self._generate_microclusters(points, true_labels, self.sample_counter, self.K, min_samples=3, algorithm=self.init_algorithm)
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
    
    def get_unknown_rate(self):
        """Returns the unknown rate, represents the percentage of unknown samples on the total number of samples classified in the online phase.

        Returns
        -------
        float
            Unknown rate
        """
        return len(self.short_mem) / self.sample_counter
    
    def get_class_unknown_rate(self):
        """Returns the unknown rate per class. Represents the percentage of unknown samples on the total number of samples of that class seen during the stream.

        Returns
        -------
        dict
            Dictionary containing the unknown rate of each class
        """
        return {key: val / self.class_sample_counter[key] for key, val in self.nb_class_unknown.items()}
    
    def predict_proba_one(self,X):
        #Function used by river algorithms to get the probability of the prediction. It is not applicable to this algorithm since it only predicts labels. 
        #It is only added as to follow River's API.
        pass
    
    def predict_proba_many(self, X):
        #Function used by river algorithms to get the probability of the predictions. It is not applicable to this algorithm since it only predicts labels. 
        #It is only added as to follow River's API.
        pass
    
    def _generate_microclusters(self, X, y, timestamp, K, keep_instances=False, min_samples=0, algorithm="kmeans"):
        if K == 0:
            return []
        
        #Create K clusters
        if algorithm == "kmeans":
            clf = KMeans(n_clusters=K, n_init="auto", random_state=self.random_state).fit(X)
        else:
            clf = MCIKMeans(n_clusters=K, random_state=self.random_state).fit(X, y)
            
        cluster_labels = clf.labels_

        #For each cluster, create a microcluster (cluster summary) and discard the data points
        microclusters = []
        for microcluster in np.unique(cluster_labels):
            cluster_instances = X[cluster_labels == microcluster]
            y_cluster_instances = y[cluster_labels == microcluster]
            
            #Assign the label of the cluster (the class label with the highest frequency in the cluster)
            values, counts = np.unique(y_cluster_instances, return_counts=True)
            most_common_y = values[np.argmax(counts)]

            if len(cluster_instances) >= min_samples:
                mc = MicroCluster(most_common_y, instances=cluster_instances, timestamp=timestamp, keep_instances=keep_instances)
                microclusters.append(mc)
        
        return microclusters
    
    def _check_f_outlier(self, X, models):
        
        #TODO: Parallelize these for loops through Numpy arrays
        f_outliers = []
        for point in X:
            f_outlier = True
            for model in models:
                #X is an F-outlier if it is outside the decision boundary of all models
                for microcluster in model.microclusters:
                    if microcluster.distance_to_centroid(point) <= microcluster.max_distance:
                        f_outlier = False
                        break
                else:
                    #If the inner condition was not triggered, we continue checking for the next model
                    continue
                break #Otherwise, we know X it not an F-outlier so we pass to the next point


            f_outliers.append(f_outlier)

        return f_outliers
    
    def _majority_voting(self, X, return_labels=True):
        closest_clusters = []
        labels = []
        dists = []
        
        #Iterate over all of the models in the ensemble
        for model in self.models:
            #Get the model's closest microcluster and its corresponding distance for each X
            closest_clusters_model, dist = get_closest_clusters(X, [microcluster.centroid for microcluster in model.microclusters])
            closest_clusters.append(closest_clusters_model)
            labels.append([model.microclusters[closest_cluster].label for closest_cluster in closest_clusters_model])
            dists.append(dist)
        
        #From all the closest microclusters of each model, get the index of the closest model for each X
        best_models = np.argmin(dists, axis=0)

        #Check if younger classifier classifies as new class C and older classifier wasn't trained on C
        for k in range(len(X)):
            for i in range(len(self.models)-1, 0, -1):
                for j in range(0, i):
                    if not labels[i][k] in self.models[j].labels: #If the label predicted by new classifier i was not in training sample of older classifier j
                        #Check if the point is an outlier of model j
                        outlier = True
                        for microcluster in self.models[j].microclusters:
                            if microcluster.distance_to_centroid(X[k]) <= microcluster.max_distance:
                                outlier = False
                                break
                        #If the point is an outlier of classifier j, don't consider its label
                        if outlier:
                            labels[j][k] = -1
        
        #Finally, create a list of tuples, which contain the index of the closest model and the index of the closest microcluster within that model for each X
        closest_model_cluster = []
        for i in range(len(X)):
            closest_model_cluster.append((best_models[i], closest_clusters[best_models[i]][i]))

        #Return the list of tuples (index of closest model, index of closest microcluster within that model), 
        # and a list containing the label Y with the most occurence between all of the models (majority voting) for each X. 
        if return_labels:
            return closest_model_cluster, self._get_most_occuring_by_column(labels)
        else:
            return closest_model_cluster
        
    def _novelty_detect(self):
        if self.verbose > 1: print("Novelty detection started")
        X = self.short_mem.get_all_points()
        new_class_vote = 0
        
        #Creating F-pseudopoints representing all F-outliers to speedup computation of qnsc
        K0 = round(self.K * (len(X) / self.chunk_size))
        K0 = max(K0, self.K)
        K0 = min(K0, len(X)) #Can't create K clusters if K is higher than the number of samples

        f_microclusters = self._generate_microclusters(X, np.array([-1] * len(X)), self.sample_counter, K0, keep_instances=True, min_samples=0, algorithm="kmeans")
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
        closest_model_cluster = self._majority_voting(self.short_mem.get_all_points(), False)


        for i, instance in enumerate(self.short_mem.get_all_instances()):
            closest_cluster = self.models[closest_model_cluster[i][0]].microclusters[closest_model_cluster[i][1]]

            if ((self.sample_counter - instance.timestamp > self.chunk_size) #The instance has an age greater than the chunk size
                or (closest_cluster.distance_to_centroid(instance.point) <= closest_cluster.max_distance)): #The instance is no longer an F-outlier 

                self._remove_sample_from_short_mem(self.short_mem.index(instance))

    def _get_most_occuring_by_column(self, l):
        most_common_values = {}
        for col in zip(*l):
            #Use a Counter to count the occurrences of each value in the column while ignoring -1 since it is a label we want to ignore
            counts = Counter(val for val in col if val != -1)
            
            #Find the most common value in the Counter
            most_common_value = counts.most_common(1)
            
            #If there are no valid values in the column, set the most_common_value to -1
            if not most_common_value:
                most_common_value = -1
            else:
                most_common_value = most_common_value[0][0]

            most_common_values[len(most_common_values)] = most_common_value

        return [most_common_values[i] for i in range(len(most_common_values))]

    def _remove_sample_from_short_mem(self, index):
        y_true = self.short_mem.get_instance(index).y_true
        if y_true is not None:
            self.nb_class_unknown[y_true] -= 1
        self.short_mem.remove(index)