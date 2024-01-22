from collections import Counter
import numpy as np
import math

from sklearn.cluster import KMeans

from streamndr.utils.data_structure import MicroCluster
from streamndr.utils.mcikmeans import MCIKMeans

__all__ = ["get_closest_clusters", "qnsc", "generate_microclusters", "check_f_outlier", "get_most_occuring_by_column", "majority_voting"]

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
        return np.full(len(X), -1)
        
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

def generate_microclusters(X, y, timestamp, K, keep_instances=False, min_samples=0, algorithm="kmeans", random_state=None):
    if K == 0:
        return []
    
    #Create K clusters
    if algorithm == "kmeans":
        clf = KMeans(n_clusters=K, n_init="auto", random_state=random_state).fit(X)
    else:
        clf = MCIKMeans(n_clusters=K, random_state=random_state).fit(X, y)
        
    cluster_labels = clf.labels_

    #For each cluster, create a microcluster (cluster summary) and discard the data points
    microclusters = []
    for microcluster in np.unique(cluster_labels):
        cluster_instances = X[cluster_labels == microcluster]
        y_cluster_instances = y[cluster_labels == microcluster]
        
        #Assign the label of the cluster (the class label with the highest frequency in the cluster)
        values, counts = np.unique(y_cluster_instances, return_counts=True)
        most_common_y = values[np.argmax(counts)]

        if len(cluster_instances) >= min_samples:
            mc = MicroCluster(most_common_y, instances=cluster_instances, timestamp=timestamp, keep_instances=keep_instances)
            microclusters.append(mc)
    
    return microclusters

def check_f_outlier(X, models):
        
    #TODO: Parallelize these for loops through Numpy arrays
    f_outliers = []
    for point in X:
        f_outlier = True
        for model in models:
            #X is an F-outlier if it is outside the decision boundary of all models
            for microcluster in model.microclusters:
                if microcluster.distance_to_centroid(point) <= microcluster.max_distance:
                    f_outlier = False
                    break
            else:
                #If the inner condition was not triggered, we continue checking for the next model
                continue
            break #Otherwise, we know X it not an F-outlier so we pass to the next point


        f_outliers.append(f_outlier)

    return f_outliers

def get_most_occuring_by_column(l):
    most_common_values = {}
    for col in zip(*l):
        #Use a Counter to count the occurrences of each value in the column while ignoring -1 since it is a label we want to ignore
        counts = Counter(val for val in col if val != -1)
        
        #Find the most common value in the Counter
        most_common_value = counts.most_common(1)
        
        #If there are no valid values in the column, set the most_common_value to -1
        if not most_common_value:
            most_common_value = -1
        else:
            most_common_value = most_common_value[0][0]

        most_common_values[len(most_common_values)] = most_common_value

    return [most_common_values[i] for i in range(len(most_common_values))]

def majority_voting(X, models, return_labels=True, ignore_old_votes=True):
    closest_clusters = []
    labels = []
    dists = []
    
    #Iterate over all of the models in the ensemble
    for model in models:
        #Get the model's closest microcluster and its corresponding distance for each X
        closest_clusters_model, dist = get_closest_clusters(X, [microcluster.centroid for microcluster in model.microclusters])
        closest_clusters.append(closest_clusters_model)
        labels.append([model.microclusters[closest_cluster].label for closest_cluster in closest_clusters_model])
        dists.append(dist)
    
    #From all the closest microclusters of each model, get the index of the closest model for each X
    best_models = np.argmin(dists, axis=0)

    #Check if younger classifier classifies as new class C and older classifier wasn't trained on C
    if ignore_old_votes:
        for k in range(len(X)):
            for i in range(len(models)-1, 0, -1):
                for j in range(0, i):
                    if not labels[i][k] in models[j].labels: #If the label predicted by new classifier i was not in training sample of older classifier j
                        #Check if the point is an outlier of model j
                        outlier = True
                        for microcluster in models[j].microclusters:
                            if microcluster.distance_to_centroid(X[k]) <= microcluster.max_distance:
                                outlier = False
                                break
                        #If the point is an outlier of classifier j, don't consider its label
                        if outlier:
                            labels[j][k] = -1
    
    #Finally, create a list of tuples, which contain the index of the closest model and the index of the closest microcluster within that model for each X
    closest_model_cluster = []
    for i in range(len(X)):
        closest_model_cluster.append((best_models[i], closest_clusters[best_models[i]][i]))

    #Return the list of tuples (index of closest model, index of closest microcluster within that model), 
    # and a list containing the label Y with the most occurence between all of the models (majority voting) for each X. 
    if return_labels:
        return closest_model_cluster, get_most_occuring_by_column(labels)
    else:
        return closest_model_cluster