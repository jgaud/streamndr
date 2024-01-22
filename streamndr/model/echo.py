from random import random
import pandas as pd

from river import base

from streamndr.utils.mcikmeans import MCIKMeans

__all__ = ["Echo"]

class Echo(base.MiniBatchClassifier):
    def __init__(self, 
                 K, 
                 ensemble_size):
        super().__init__()
        self.K = K
        self.ensemble_size = ensemble_size

        self.models = []
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


    def predict_one(self, X, y=None):
        pass

    def predict_many(self, X, y=None):
        pass

    def get_unknown_rate(self):
        pass

    def get_class_unknown_rate(self):
        pass

    def predict_proba_one(self,X):
        #Function used by river algorithms to get the probability of the prediction. It is not applicable to this algorithm since it only predicts labels. 
        #It is only added as to follow River's API.
        pass

    def predict_proba_many(self, X):
        #Function used by river algorithms to get the probability of the predictions. It is not applicable to this algorithm since it only predicts labels. 
        #It is only added as to follow River's API.
        pass