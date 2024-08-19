from river import metrics
from streamndr.metrics.confusion import ConfusionMatrixNovelty
import pprint

__all__ = ["TTD"]

class TTD(metrics.base.MultiClassMetric):
    """Time To Detection (TTD), represents the amount of samples needed for a novel class to be classified as a novel concept, as defined in [1].

    [1] Gaudreault, JG., Branco, P. (2023). Toward Streamlining the Evaluation of Novelty Detection in Data Streams. 
        In: Bifet, A., Lorena, A.C., Ribeiro, R.P., Gama, J., Abreu, P.H. (eds) Discovery Science. DS 2023. Lecture Notes in Computer Science(), vol 14276. Springer, Cham. 

    Parameters
    ----------
    known_classes : list of int
        List of known labels, the labels the algorithm knows prior to the online phase
    cm : ConfusionMatrixNovelty
        Optional, can specify an already existing confusion matrix instead of creating a new one for the metric

    Attributes
    ----------
    cm : ConfusionMatrixNovelty
        Confusion matrix
    """
    def __init__(self, known_classes, cm: ConfusionMatrixNovelty = None):
        if cm is None:
            cm = ConfusionMatrixNovelty(known_classes)
            
        super(metrics.base.MultiClassMetric, self).__init__(cm)
        
        self.time_since_first_seen = dict()
        self.time_to_detection = dict()

    def update(self, y_true, y_pred, w=1.0):
        super().update(y_true, y_pred, w)
        
        if (not y_true in self.time_since_first_seen) and (not y_true in self.cm._init_classes):
            self.time_since_first_seen[y_true] = 0
        
        elif (y_true in self.time_since_first_seen) and (not y_true in self.time_to_detection) and (not y_pred in self.cm._init_classes.union({-1})):
            self.time_to_detection[y_true] = self.time_since_first_seen[y_true]

        elif (y_true in self.time_since_first_seen) and (not y_true in self.time_to_detection) and (y_pred in self.cm._init_classes.union({-1})):
            self.time_since_first_seen[y_true] += 1
    
    def get(self):
        tmp = dict()

        for key in self.time_since_first_seen:
            if key in self.time_to_detection:
                tmp[key] = self.time_to_detection[key]
            else:
                tmp[key] = -1

        return tmp
    
    def __repr__(self):
        return "TTD: " + pprint.pformat(self.get())