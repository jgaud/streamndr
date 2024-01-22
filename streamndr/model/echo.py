from random import random
import pandas as pd
import numpy as np

from streamndr.model.noveltydetectionclassifier import NoveltyDetectionClassifier
from streamndr.utils.cluster_utils import *
from streamndr.utils.data_structure import ClusterModel, ShortMem

__all__ = ["Echo"]

class Echo(NoveltyDetectionClassifier):
    def __init__(self, 
                 K, 
                 ensemble_size,
                 random_state=None,
                 init_algorithm="mcikmeans"):
        super().__init__()
        self.K = K
        self.ensemble_size = ensemble_size
        self.random_state = random_state

        accepted_algos = ['kmeans','mcikmeans']
        if init_algorithm not in accepted_algos:
            print('Available algorithms: {}'.format(', '.join(accepted_algos)))
        else:
            self.init_algorithm = init_algorithm

        self.models = []
        self.short_mem = ShortMem() #Potential novel class instances
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
        closest_model_cluster, y_preds = self._majority_voting(X)

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