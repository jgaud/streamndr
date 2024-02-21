import numpy as np
import math

__all__ = ["get_closest_clusters", "qnsc"]

#Constant used to determine the maximum number of rows used by numpy for the computation of the closest clusters. A higher number is faster but takes more memory.
MAX_MEMORY_SIZE = 50000

def get_closest_clusters(X, centroids):
    """Function returning the closest centroid and distance for each given point.

    Parameters
    ----------
    X : numpy.ndarray
        Array of points
    centroids : numpy.ndarray
        Array of centroids

    Returns
    -------
    numpy.ndarray
        Index of the closest cluster for each point
    numpy.ndarray
        Distance to the closest cluster for each point
    """
    if len(centroids) == 0:
        print("No clusters")
        return np.full(len(X), -1), -1
        
    centroids = np.array(centroids)
    norm_dists = np.zeros((X.shape[0],centroids.shape[0]))

    # Cut into batches if there are too many samples to save on memory
    for idx in range(math.ceil(X.shape[0]/MAX_MEMORY_SIZE)):
        sl = slice(idx*MAX_MEMORY_SIZE, (idx+1)*MAX_MEMORY_SIZE)
        norm_dists[sl] = np.linalg.norm(np.subtract(X[sl, :, None], np.transpose(centroids)), axis=1)

    return np.argmin(norm_dists, axis=1), np.amin(norm_dists, axis=1)

def qnsc(pseudopoints, model, q_p=5):
    """Computes the q-neighborhood silhouette coefficient, as described in [1].

    [1] Masud, Mohammad, et al. "Classification and novel class detection in concept-drifting data streams under time constraints." 
    IEEE Transactions on knowledge and data engineering 23.6 (2010): 859-874.

    Parameters
    ----------
    pseudopoints : numpy.ndarray
        List of points
    model : list of MicroCluster
        Microclusters representing a model
    q_p : int
        Number of neighboring points to consider

    Returns
    -------
    numpy.ndarray
        List of computed qnscs for each point
    """
    qnscs = []
    cluster_by_label = {}

    for cluster in model:
        if cluster.label not in cluster_by_label:
            cluster_by_label[cluster.label] = []

        cluster_by_label[cluster.label].append(cluster.centroid)

    
    #1 - Find the Q closest distances for each point
    distances = np.linalg.norm(pseudopoints[:, np.newaxis] - pseudopoints, axis=2)

    # Set the diagonal elements to a large value to avoid selecting the same point
    np.fill_diagonal(distances, np.inf)
    q = min(q_p, len(pseudopoints)-2) #-2 because we can't select the same point
    indices = np.argpartition(distances, q, axis=1)[:, :q] 

    # Retrieve the Q minimum distances for each point
    min_distances = np.take_along_axis(distances, indices, axis=1)
    dc_outs = np.mean(min_distances, axis=1)
    
    for i, point in enumerate(pseudopoints):
        dc_out = dc_outs[i]
        dc_q = []
        for _, clusters in cluster_by_label.items():
            dists = []
            for centroid in clusters:
                dists.append(np.linalg.norm(point-centroid))
            
            q = min(q_p, len(dists))
            q_closest = np.partition(dists, q-1)[:q]
            dc_q.append(np.sum(q_closest)/q)

        dcmin_q = np.min(dc_q)
        qnsc = (dcmin_q - dc_out) / max(dcmin_q, dc_out)
        
        qnscs.append(qnsc)

        
    return qnscs