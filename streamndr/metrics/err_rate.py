from river import metrics
from streamndr.metrics.confusion import ConfusionMatrixNovelty

__all__ = ["ErrRate"]

class ErrRate(metrics.base.MultiClassMetric):
    """Error rate, represents the total misclassification error percentage.

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
    
    def get(self):
        fp = self.cm.false_positives_novelty() #Number of known class samples wrongly classified as novelties
        fn = self.cm.false_negatives_novelty() #Number of novelties wrongly classified as known
        
        try:
            return (fp + fn + self.cm.fe) / self.cm.n_samples
        except ZeroDivisionError:
            return 0.0