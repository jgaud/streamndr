import numpy as np
import pandas as pd

from streamndr.model import ECSMiner
from streamndr.utils.data_structure import ShortMemInstance, ClusterModel
from streamndr.utils.cluster_utils import get_closest_clusters

__all__ = ["ECSMinerWF"]

class ECSMinerWF(ECSMiner):
    """Implementation of the ECSMinerWF (ECSMiner without feedback) algorithm for novelty detection. [1]

    [1] de Faria, Elaine Ribeiro, André Carlos Ponce de Leon Ferreira Carvalho, and Joao Gama. "MINAS: multiclass learning algorithm for novelty detection in data streams." 
    Data mining and knowledge discovery 30 (2016): 640-680.

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
    init_algorithm : string
        String containing the clustering algorithm to use to initialize the clusters, supports 'kmeans' and 'mcikmeans'

    Attributes
    ----------
    novel_models : list of ClusterModel
        List containing the models representing novel classes, added during the online phase.
    """
    
    def __init__(self,
                 K=50, 
                 min_examples_cluster=50, #Number of instances requried to declare a novel class 
                 ensemble_size=6, 
                 verbose=0,
                 random_state=None,
                 init_algorithm="mcikmeans"):
        
        super().__init__(K=K, 
                         min_examples_cluster=min_examples_cluster, 
                         ensemble_size=ensemble_size, 
                         T_l=0,
                         verbose=verbose, 
                         random_state=random_state, 
                         init_algorithm=init_algorithm)
        
        self.novel_models = []
        
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

        #If we have novel models, get the closest ones for all Xs
        if len(self.novel_models) > 0:
            closest_novel_clusters, novel_dists = get_closest_clusters(X, [microcluster.centroid for model in self.novel_models for microcluster in model.microclusters])
            novel_microclusters = [microcluster for model in self.novel_models for microcluster in model.microclusters]

        
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
                #Check if X can be explained by our list of novel models
                if len(self.novel_models) > 0 and novel_dists[i] <= novel_microclusters[closest_novel_clusters[i]].max_distance:
                    pred_label.append(novel_microclusters[closest_novel_clusters[i]].label)
                    novel_microclusters[closest_novel_clusters[i]].update_cluster(X[i], self.sample_counter, False)

                else:
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
                            #Find the next available sequential label
                            max_label_ensemble = max([label for model in self.models for label in model.labels])
                            max_label_novel = max([label for model in self.novel_models for label in model.labels]) if len(self.novel_models) > 0 else -1
                            new_label = max(max_label_ensemble, max_label_novel) + 1

                            for novel_cluster in novel_clusters:
                                #Set the label for each microcluster
                                novel_cluster.label = new_label

                                if self.verbose > 0: print("Novel cluster detected: ", novel_cluster.small_str())

                                #Remove instances from the buffer
                                for instance in novel_cluster.instances:
                                    self._remove_sample_from_short_mem(self.short_mem.index(np.array(instance)))

                            pred_label[-1] = new_label

                            #Add the clusters to our novel models list
                            self.novel_models.append(ClusterModel(novel_clusters, [new_label]))
                    
        return np.array(pred_label)