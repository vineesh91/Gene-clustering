# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 10:22:05 2018

@author: S.Dejbord
"""

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plot

def DBSCAN(D, eps, MinPts):
    C = 0
    mark = [0]*len(D) # 0 - unvisited; 0.5 - visited; -1 - noise; 1,2,... - cluster
    for P in range(0, len(D)):
        if (mark[P] == 0): # if unvisited
            mark[P] = 0.5 # mark as visited
            neighbors = regionQuery(D, P, eps)
            if len(neighbors) < MinPts:
                mark[P] = -1 # mark as noise
            else:
                C += 1 # new cluster
                expandCluster(D, P, mark, neighbors, C, eps, MinPts)
    return mark
        
def regionQuery(D, P, eps):
    neighbor = []
    for p in range(0, len(D)):
        if np.linalg.norm(D[P] - D[p]) <= eps: # if distance of P and p <= eps
            neighbor.append(p) # then p is P's neighbor in radius of eps, add p to P's neighbors
    return neighbor

def expandCluster(D, P, mark, neighbors, C, eps, MinPts):
    mark[P] = C 
    n = 0
    while n < len(neighbors):
        if (mark[neighbors[n]] == 0): 
               mark[neighbors[n]] = 0.5 
               neighbor = regionQuery(D, neighbors[n], eps) 
               if len(neighbor) >= MinPts:
                   neighbors = neighbors + neighbor # expand P's neighbors
        if (mark[neighbors[n]] < 1): 
            mark[neighbors[n]] = C 
        n += 1


def get_jaccard_similarity(clustered_feature_matrix, classes_list, ground_truth_classes_list):
    obtained_same_cluster_matrix = np.zeros((len(clustered_feature_matrix), len(clustered_feature_matrix)))
    ground_truth_same_cluster_matrix = np.zeros((len(clustered_feature_matrix), len(clustered_feature_matrix)))

    # populate the same cluster matrices
    for i in range(obtained_same_cluster_matrix.shape[0]):
        obtained_same_cluster_matrix[i][i] = 1
        ground_truth_same_cluster_matrix[i][i] = 1
        for j in range(i + 1, obtained_same_cluster_matrix.shape[1]):
            if classes_list[i] == classes_list[j]:
                obtained_same_cluster_matrix[i][j] = 1
                obtained_same_cluster_matrix[j][i] = 1
            if ground_truth_classes_list[i] == ground_truth_classes_list[j]:
                ground_truth_same_cluster_matrix[i][j] = 1
                ground_truth_same_cluster_matrix[j][i] = 1

    # calculate the jaccard similarity
    numerator = np.sum(np.logical_and(obtained_same_cluster_matrix, ground_truth_same_cluster_matrix))
    denominator = np.sum(np.logical_or(obtained_same_cluster_matrix, ground_truth_same_cluster_matrix))
    return float(numerator) / float(denominator)


###################################################################################################################################################
if __name__ == "__main__":
    
    # Reading data
    data = pd.read_csv('cho.txt', header=None, sep='\t')
    #data = pd.read_csv('iyer.txt', header=None, sep='\t')
    #data = pd.read_csv('new_dataset_1.txt', header=None, sep='\t')
    
    # Extracting features and ground truth
    data = data.values
    data_ground_truth = data[:, 1]
    data_features = data[:, 2:]
    
    
    # Setting the parameters and run function DBSCAN
    """
    Before running here, determine parameters using 'params.py';
    
    To run directly, the followings are the parameters for the given datasets:
    'cho.txt'  eps: 1.3 MinPts: 16               'iyer.txt'   eps: 1.1 MinPts: 7                'new_dataset_1.txt' eps: 0.5 MinPts: 11
    
    You can manually change eps and MinPts for different datasets.
    """
    eps = 1.3
    MinPts = 16
    
    data_id = DBSCAN(data_features, eps, MinPts)
    
    
    # Calculating jaccard_similarity
   
    jaccard_similarity = get_jaccard_similarity(data_features, data_id, data_ground_truth)
    print("Jaccard similarity: " + str(jaccard_similarity))
   
    
    # visualization
    pca = PCA(n_components=2)
        
    #############################################################################################################################################################################################################
    
    
    principalComponents = pca.fit_transform(data_features)
    dim2_dbscan = pd.DataFrame(data = principalComponents
                 , index = data_id)
    
    dim2_ground_truth = pd.DataFrame(data = principalComponents, index = data_ground_truth)
    
    
    unique_label = np.unique(data_id)
    unique_label_gt = np.unique(data_ground_truth)
    
    
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(25)
    a = fig.add_subplot(1, 2, 1)
    img_dbscan = plot.D2_pca_plot(dim2_dbscan, unique_label)
    a.set_title('cho.txt Clusters from DBSCAN')
    a = fig.add_subplot(1, 2, 2)
    img_ground = plot.D2_pca_plot(dim2_ground_truth, unique_label_gt)
    a.set_title('cho.txt Clusters from Ground Truth')
    plt.show()    
    