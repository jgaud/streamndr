import math
import numpy as np
from river import metrics
from streamndr.metrics.confusion import ConfusionMatrixNovelty

__all__ = ["CER"]

class CER(metrics.base.MultiClassMetric):
    """Combined Error Rate (CER), defined as the average of the weighted rate of false positive and false negative per class [1].

    [1] E. R. Faria, I. J. C. R. Gonçalves, J. Gama and A. C. P. L. F. Carvalho, "Evaluation Methodology for Multiclass Novelty Detection Algorithms," 
        2013 Brazilian Conference on Intelligent Systems, Fortaleza, Brazil, 2013, pp. 19-25, doi: 10.1109/BRACIS.2013.12.

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
        associated_classes_conf_matrix = self.cm.get_associated_classes()
        total = 0
        
        for c in self.cm.classes:
            try:
                fp = associated_classes_conf_matrix.false_positives(c)
                tn = associated_classes_conf_matrix.true_negatives(c)
                fn = associated_classes_conf_matrix.false_negatives(c)
                tp = associated_classes_conf_matrix.true_positives(c)

                fpr = fp / (fp + tn)
                fnr = fn / (fn + tp)

                w = associated_classes_conf_matrix.support(c) / associated_classes_conf_matrix.total_weight

                total += (w * fpr) + (w * fnr)

            except ZeroDivisionError:
                continue


        return total / 2

    def get_aic(self):
        """Computes the Akaike Information Criterion (AIC) as defined in [1], which measures the complexity of the model using the CER and the number of classes detected.

        Returns
        -------
        float
            The Akaike Information Criterion (AIC)
        """
        all_classes = self.cm.classes
        
        if -1 in all_classes:
            all_classes.remove(-1)
        
        num_classes_detected = 0
        N = 0

        for cl in all_classes:
            N += self.cm.sum_col[cl]
            if self.cm.sum_col[cl] > 0:
                num_classes_detected += 1

        try:
            return -2 * math.log(1-self.get()) + 2 * num_classes_detected / math.log(N)
        except:
            return 0.0