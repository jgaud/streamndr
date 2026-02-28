__all__ = ["NoveltyDetectionClassifier"]

from river import base

class NoveltyDetectionClassifier(base.MiniBatchClassifier):

    def __init__(self,
                 verbose=0,
                 random_state=None):
        super().__init__()
        self.nb_class_unknown = dict()
        self.class_sample_counter = dict()
        self.sample_counter = 0
        self.verbose = verbose
        self.random_state = random_state
        self.before_offline_phase = True

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

    def _remove_sample_from_short_mem(self, index):
        y_true = self.short_mem.get_instance(index).y_true
        if y_true is not None:
            self.nb_class_unknown[y_true] -= 1
        self.short_mem.remove(index)