from collections import Counter
from random import random
import pandas as pd
import numpy as np

from streamndr.model.noveltydetectionclassifier import NoveltyDetectionClassifier
from streamndr.utils.cluster_utils import *
from streamndr.utils.data_structure import ClusterModel, MicroCluster, ShortMem, ShortMemInstance

__all__ = ["Echo"]

class Echo(NoveltyDetectionClassifier):
    def __init__(self, 
                 K,
                 min_examples_cluster,
                 ensemble_size,
                 verbose=0,
                 random_state=None,
                 init_algorithm="mcikmeans"):
        
        super().__init__(verbose, random_state)
        self.K = K
        self.min_examples_cluster = min_examples_cluster
        self.ensemble_size = ensemble_size

        accepted_algos = ['kmeans','mcikmeans']
        if init_algorithm not in accepted_algos:
            print('Available algorithms: {}'.format(', '.join(accepted_algos)))
        else:
            self.init_algorithm = init_algorithm

        self.models = []
        self.short_mem = ShortMem() #Potential novel class instances

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
        Echo
            Fitted estimator
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # in offline phase, consider all instances arriving at the same time in the microclusters:
        timestamp = len(X)

        microclusters = generate_microclusters(X, y, timestamp, self.K, min_samples=0, algorithm=self.init_algorithm, random_state=self.random_state)

        model = ClusterModel(microclusters, np.unique(y))

        if len(microclusters) > 0:
            self.models.append(model)

        self.before_offline_phase = False
        
        return self


    def predict_one(self, X, y=None):
        """Represents the online phase. Equivalent to predict_many() with only one sample. Receives only one sample, predict its label if it's 
        within the decision boundary of the ensemble. Otherwise, if it's unknown, it is added to the short term memory and novelty detection is 
        performed.

        Parameters
        ----------
        X : dict
            Sample
        y : int, optional
            True y value of the sample, by default None
        """
        return self.predict_many(np.array(list(X.values()))[None,:], [y])

    def predict_many(self, X, y=None):
        """Represents the online phase. Receives multiple samples, for each sample predict its label predict its label if it's within the decision 
        boundary of the ensemble. Otherwise, if it's unknown, it is added to the short term memory and novelty detection is performed once the trigger has been reached.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Samples
        y : list of int, optional
            True y values of the samples, by default None

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
        closest_model_cluster, y_preds = self._majority_voting(X, True)

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
            
            #If X is not an F-outlier (inside the closest cluster radius), then we classify it with the label from the majority voting
            if not f_outliers[i]:
                pred_label.append(y_preds[i])
                #closest_cluster.update_cluster(X[i], self.sample_counter, False)
                
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

                if (len(self.short_mem) > self.min_examples_cluster):
                    #Find the novel cluster, if any
                    novel_cluster = self._novelty_detect()
                    if novel_cluster is not None:
                        if self.verbose > 1:
                            print("Novel cluster: ", novel_cluster)
                        elif self.verbose > 0:
                            print("Novel cluster: ", novel_cluster.small_str())
                        
                        #Change the predicted label for the new class label
                        pred_label[-1] = novel_cluster.label

                    #Remove all instances from the buffer since if they were not detected as a novel classes, they are classified as per ECHO paper
                    for i in range(len(self.short_mem)):
                        self._remove_sample_from_short_mem(i)
                    
        return np.array(pred_label)
    
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
        
        #Finally, create a list of tuples, which contain the index of the closest model and the index of the closest microcluster within that model for each X
        closest_model_cluster = []
        for i in range(len(X)):
            closest_model_cluster.append((best_models[i], closest_clusters[best_models[i]][i]))

        #Return the list of tuples (index of closest model, index of closest microcluster within that model), 
        # and a list containing the label Y with the most occurence between all of the models (majority voting) for each X. 
        if return_labels:
            return closest_model_cluster, get_most_occuring_by_column(labels)
        else:
            return closest_model_cluster
    
    def _novelty_detect(self):
        if self.verbose > 1: print("Novelty detection started")
        X = self.short_mem.get_all_points()
        new_class_vote = 0

        potential_novel_points_idx = []
        #Computing qNSC for each model in our ensemble
        for model in self.models:
            qnscs = qnsc(X, model.microclusters, self.min_examples_cluster)
            potential_points = []
            total_instances = 0
            for i, point in enumerate(X):
                if qnscs[i] > 0:
                    potential_points.append(point)
                    total_instances += 1
                    potential_novel_points_idx.append(i)
            if total_instances > 0 and self.verbose > 1:
                print(f"Total instances in F-outliers: {total_instances}")
            
            if total_instances > self.min_examples_cluster: 
                new_class_vote += 1

        if new_class_vote == len(self.models):
            #Get the indices of all points which had a positive qnsc for all models
            novel_points_idx = [item for item, count in Counter(potential_novel_points_idx).items() if count == len(self.models)]
            novel_points = [X[i] for i in novel_points_idx]
            
            label = max(set(element for sublist in self.models for element in sublist.labels)) + 1
            return MicroCluster(label, instances=novel_points, timestamp=self.sample_counter, keep_instances=False)
        
        else:
            return None