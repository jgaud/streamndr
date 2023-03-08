from river import metrics

class ConfusionMatrixNovelty(metrics.confusion.ConfusionMatrix):
    """Confusion Matrix for novelty detection in data streams.

    Parameters
    ----------
    known_classes : list of int
        List of known labels, the labels the algorithm knows prior to the online phase

    Attributes
    ----------
    novel_cm : river.metrics.Confusion.ConfusionMatrix
        Binary confusion matrix representing the problem in a binary manner, including class 0 (known) and class 1 (novelty)
    nc_samples : int
        Number of samples representing a novelty
    fe : int
        Known samples that have been classified as a known class other than its ground truth
    """
    def __init__(self, known_classes):
        super().__init__(known_classes)
        self.novel_cm = metrics.confusion.ConfusionMatrix()
        self.nc_samples = 0
        self.fe = 0
        
    def update(self, y_true, y_pred, sample_weight=1.0):
        super().update(y_true, y_pred, sample_weight)
        
        known_class = int(y_true in self._init_classes)
        pred_known_class = int(y_pred in self._init_classes)
        
        if known_class == 0:
            self.nc_samples += 1
        elif known_class == pred_known_class == 1 and y_true != y_pred: #If the prediction is not a novelty, but we predicted the wrong class
            self.fe += 1
        
        self.novel_cm.update(1-known_class, 1-pred_known_class)
        
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        super.revert(self, y_true, y_pred, sample_weight)
        
        known_class = int(y_true in self._init_classes)
        pred_known_class = int(y_pred in self._init_classes)
        
        if known_class == 1:
            self.nc_samples -= 1
            if pred_known_class == 1 and y_true != y_pred:
                self.fe -= 1
        
        self.novel_cm.revert(1-known_class, 1-pred_known_class)
        
        return self
    
    def true_positives_novelty(self):
        return self.novel_cm.true_positives(1)

    def true_negatives_novelty(self):
        return self.novel_cm.true_negatives(1)

    def false_positives_novelty(self):
        return self.novel_cm.false_positives(1)

    def false_negatives_novelty(self):
        return self.novel_cm.false_negatives(1)
