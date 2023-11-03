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
        return
        
    centroids = np.array(centroids)
    norm_dists = np.zeros((X.shape[0],centroids.shape[0]))

    # Cut into batches if there are too many samples to save on memory
    for idx in range(math.ceil(X.shape[0]/MAX_MEMORY_SIZE)):
        sl = slice(idx*MAX_MEMORY_SIZE, (idx+1)*MAX_MEMORY_SIZE)
        norm_dists[sl] = np.linalg.norm(np.subtract(X[sl, :, None], np.transpose(centroids)), axis=1)

    return np.argmin(norm_dists, axis=1), np.amin(norm_dists, axis=1)

def qnsc(pseudopoints, model, q_p=5):
    qnscs = []
    cluster_by_label = {}

    for cluster in model:
        if cluster.label not in cluster_by_label:
            cluster_by_label[cluster.label] = []

        cluster_by_label[cluster.label].append(cluster.centroid)

    for i, point in enumerate(pseudopoints):
        dists = []
        for j, point2 in enumerate(pseudopoints):
            if i != j:
                dists.append(np.linalg.norm(point-point2))

        q = min(q_p, len(dists))
        q_closest = np.partition(dists, q-1)[:q]

        dc_out = np.sum(q_closest)/q

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