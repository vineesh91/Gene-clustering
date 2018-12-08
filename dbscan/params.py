# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 10:21:25 2018

@author: S.Dejbord
"""


import numpy
from matplotlib import pyplot as plt
import pandas as pd

def besteps(i, D):
    iDistance = [0]*len(D) # build a vector to stores all points' distance to their ith nearest neighbor
    matrix = distance_Mat(D) # build a distance matrix to store all pair-distances and sort distances for each point 
    for a in range(0, len(D)):
        iDistance[a] = matrix[i, a] # return each point's ith nearest distance by extracting row-i col-point in dist matrix
    iDis = numpy.sort(iDistance) # sort all the ith nearest distances
    # plot sorted distance of every point to its kth nearest neighbor
    px =list(range(len(iDis))) 
    py = iDis
    plt.scatter(px, py)
    plt.xlabel('Points')
    plt.ylabel (str(i) + 'th nearest distance')
    plt.show()
    

def distance_Mat(D): 
    distance = numpy.zeros((len(D), len(D)))
    for p in range(0, len(D)):
        for q in range(0, len(D)):
            distance[p, q] = numpy.linalg.norm(D[p] - D[q])
    return numpy.sort(distance, axis = 0)


####################################################################################################################################################
# Reading and extracting data
data = pd.read_csv('cho.txt', header=None, sep='\t')
#data = pd.read_csv('iyer.txt', header=None, sep='\t')
#data = pd.read_csv('new_dataset_1.txt', header=None, sep='\t')


data = data.values
data_ground_truth = data[:, 1]
data_features = data[:, 2:]

# Determining eps
for i in range(3, 20, 1): # i - MinPts, we consider an representative range 3 to 20 all the time
    besteps(i-1, data_features) # obtain sorted distance plot by running besteps
# determing eps by taking the average of best eps for each MinPts from 3 to 20 by the plot,
import dbscan

# Determining MinPts by iteration
# after determining eps by first running, subsititute eps in the following DBSCAN function and then run again
for j in range(3, 20, 1): 
        data_id = dbscan.DBSCAN(data_features, 1.3, j)
        jaccard_similarity = dbscan.get_jaccard_similarity(data_features, data_id, data_ground_truth)
        print('The jaccard_similarity of eps {} MinPts {} is {}'.format(1.3, j, jaccard_similarity)) 
       
#choose the MinPts-eps pair with the largest jaccard_similarity
    