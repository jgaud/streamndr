import numpy as np
import math

__all__ = ["get_closest_clusters", "qnsc"]

#Constant used to determine the maximum number of rows used by numpy for the computation of the closest clusters. A higher number is faster but takes more memory.
MAX_MEMORY_SIZE = 50000

def get_closest_clusters(X, centroids):   
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

def qnsc(pseudopoints, model):
        #Calculate mean distance of all points between themselves
        dists = np.linalg.norm(pseudopoints - pseudopoints[:,None], axis=-1)
        dists[np.arange(dists.shape[0]), np.arange(dists.shape[0])] = np.nan
        mean_distances_between_points = np.nanmean(dists, axis=0)
        
        #Calculate minimum distance between points known cluster
        all_centroids = [microcluster.centroid for microcluster in model]
        _, minimum_distances_to_class = get_closest_clusters(pseudopoints, all_centroids)
        
        qnscs = (minimum_distances_to_class - mean_distances_between_points) / np.maximum(minimum_distances_to_class, mean_distances_between_points)
        
        return qnscs