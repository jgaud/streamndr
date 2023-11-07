from streamndr.model import ECSMiner

__all__ = ["ECSMinerWF"]

class ECSMinerWF(ECSMiner):
    """Implementation of the ECSMinerWF (ECSMiner without feedback) algorithm for novelty detection.

    Parameters
    ----------
    K : int
        Number of pseudopoints per classifier. In other words, it is the number of K cluster for the clustering algorithm.
    min_examples_cluster : int
        Minimum number of examples to declare a novel class 
    ensemble_size : int
        Number of classifiers to use to create the ensemble
    verbose : int
        Controls the level of verbosity, the higher, the more messages are displayed. Can be '1', '2', or '3'.
    random_state : int
        Seed for the random number generation. Makes the algorithm deterministic if a number is provided.
    init_algorithm : string
        String containing the clustering algorithm to use to initialize the clusters, supports 'kmeans' and 'mcikmeans'
    """
    
    def __init__(self,
                 K=50, 
                 min_examples_cluster=50, #Number of instances requried to declare a novel class 
                 ensemble_size=6, 
                 verbose=0,
                 random_state=None,
                 init_algorithm="mcikmeans"):
        
        super().__init__(K, min_examples_cluster, ensemble_size, verbose, random_state, init_algorithm)        