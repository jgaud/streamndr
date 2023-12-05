import pandas as pd

from river import base

__all__ = ["Echo"]

class Echo(base.MiniBatchClassifier):
    def __init__(self, K, ensemble_size):
        self.K = K
        self.ensemble_size = ensemble_size

        self.models = []
        self.before_offline_phase = True

    def learn_one(self, x, y, w=1.0):
        pass

    def learn_many(self, X, y, w=1.0):
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