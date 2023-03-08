from river import metrics
from streamndr.metrics.confusion import ConfusionMatrixNovelty

__all__ = ["UnkRate"]

class UnkRate(metrics.base.MultiClassMetric):
    """Unknown rate, represents the percentage of unknown samples.

    Parameters
    ----------
    known_classes : list of int
        List of known labels, the labels the algorithm knows prior to the online phase

    Attributes
    ----------
    cm : ConfusionMatrixNovelty
        Confusion matrix
    """
    def __init__(self, known_classes):
        cm = ConfusionMatrixNovelty(known_classes)
        super(metrics.base.MultiClassMetric, self).__init__(cm)
    
    def get(self):
        try:
            return self.cm.sum_col[-1] / self.cm.n_samples
        except ZeroDivisionError:
            return 0.0