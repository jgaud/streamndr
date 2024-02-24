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

    def get_associated_classes(self):
        """Computes the associated known class to each novelty pattern discovered, as described in [1], by using the real class most represented in each novelty pattern.
        Ignores the unknown samples (label -1).

        [1] E. R. Faria, I. J. C. R. Gonçalves, J. Gama and A. C. P. L. F. Carvalho, "Evaluation Methodology for Multiclass Novelty Detection Algorithms," 
        2013 Brazilian Conference on Intelligent Systems, Fortaleza, Brazil, 2013, pp. 19-25, doi: 10.1109/BRACIS.2013.12.
        
        Returns
        -------
        ConfusionMatrixNovelty
            The confusion matrix using the most represented known class for each of the novelty pattern reported
        """
        associated_classes_conf_matrix = ConfusionMatrixNovelty(self._init_classes)
        unknown_classes = [x for x in self.classes if x not in self._init_classes.union({-1})]

        for cl in self.classes:
            col = [int(self.data[row][cl]) for row in self.classes]
            
            #If the class is a novelty pattern, we select the real class most represented within the novelty pattern
            if cl in unknown_classes:
                index_max = col.index(max(col))
                pred = self.classes[index_max]

                for row in self.classes:
                    associated_classes_conf_matrix.update(row, pred, self.data[row][cl])
            elif cl != -1:
                for row in self.classes:
                    associated_classes_conf_matrix.update(row, cl, self.data[row][cl])

        return associated_classes_conf_matrix
    
        
    def update(self, y_true, y_pred, w=1.0):
        super().update(y_true, y_pred, w)
        
        known_class = int(y_true in self._init_classes)
        pred_known_class = int(y_pred in self._init_classes)
        
        if known_class == 0:
            self.nc_samples += 1
        elif known_class == pred_known_class == 1 and y_true != y_pred: #If the prediction is not a novelty, but we predicted the wrong class
            self.fe += 1
        
        self.novel_cm.update(1-known_class, 1-pred_known_class)

    def revert(self, y_true, y_pred, w=1.0):
        super.revert(self, y_true, y_pred, w)
        
        known_class = int(y_true in self._init_classes)
        pred_known_class = int(y_pred in self._init_classes)
        
        if known_class == 1:
            self.nc_samples -= 1
            if pred_known_class == 1 and y_true != y_pred:
                self.fe -= 1
        
        self.novel_cm.revert(1-known_class, 1-pred_known_class)
    
    def true_positives_novelty(self):
        return self.novel_cm.true_positives(1)

    def true_negatives_novelty(self):
        return self.novel_cm.true_negatives(1)

    def false_positives_novelty(self):
        return self.novel_cm.false_positives(1)

    def false_negatives_novelty(self):
        return self.novel_cm.false_negatives(1)
