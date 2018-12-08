import math
import numpy as np
from sklearn.decomposition import PCA

class Hierarchical:
    def __init__(self, cluster_count, input_data):
        self.k = cluster_count
        self.data = input_data
        self.distance_matrix = self.calc_distance()
        self.intermediate_cluster = {}
        self.final_cluster = self.cluster()

    def cluster(self):
        cluster_list = {}
        for gn in self.data.gene_list:
            cluster_list[gn.id] = [gn]
        self.intermediate_cluster = cluster_list
        # min_dis_per_data = self.distance_matrix.min(axis=1)
        # fltrd_dat = np.argwhere(min_dis_per_data > 2.3)
        # fltrd_dat2 = np.asarray([i + 1 for i in fltrd_dat])
        while(len(self.intermediate_cluster) != self.k):
            self.merge_cluters(self.intermediate_cluster)
        return self.intermediate_cluster

    def merge_cluters(self,clstr_list):
        new_cluster_list = {}
        res = np.argwhere(self.distance_matrix == np.min(self.distance_matrix))[0]


        lst1_clstr = clstr_list[res[0]+1].copy()
        lst2_clstr = clstr_list[res[1]+1].copy()
        lst1_clstr.extend(lst2_clstr)
        del clstr_list[res[1]+1]
        clstr_list[res[0]+1] = lst1_clstr
        # id = 1
        # for clstrs_key in clstr_list:
        #     new_cluster_list[id] = clstr_list[clstrs_key]
        #     id += 1
        # N = len(clstr_list)
        self.distance_matrix = self.distance_mat_from_prev(res[0],res[1])
        self.intermediate_cluster = clstr_list

    def distance_mat_from_prev(self,c1,c2):
        N = len(self.distance_matrix)
        for i in range(0,N):
            lower_val = min(self.distance_matrix[i][c1],self.distance_matrix[i][c2])
            self.distance_matrix[i][c1] = lower_val
            self.distance_matrix[c1][i] = lower_val
            self.distance_matrix[i][c2] = math.inf
            self.distance_matrix[c2][i] = math.inf
        self.distance_matrix[c1][c1] = math.inf
        # mat = np.delete(self.distance_matrix, c2, 0)
        # np.delete(mat, c2, 1)
        return self.distance_matrix
        # dist_mat = np.ones((N, N)) * np.inf


    def get_distance_mat(self,cluster_list):
        dist_mat = np.ones((len(cluster_list),len(cluster_list))) * np.inf
        N = len(cluster_list)
        for clstr_id1 in  range(1,N):
            for clstr_id2 in range(clstr_id1 + 1,N + 1):
                if clstr_id1 != clstr_id2:
                    dis = self.cluster_dist(cluster_list[clstr_id1],cluster_list[clstr_id2])
                    dist_mat[clstr_id1-1][clstr_id2-1] = dis
                    dist_mat[clstr_id2-1][clstr_id1-1] = dis
        return dist_mat

    def cluster_dist(self,cluster1,cluster2):
        min_dist = math.inf
        for elem1 in cluster1:
            for elem2 in cluster2:
                dist_gns = elem1.gene_distance(elem2)
                if dist_gns < min_dist:
                    min_dist = dist_gns
        return min_dist

    def calc_distance(self):
        dist_mat = np.ones((len(self.data.gene_list),len(self.data.gene_list))) * np.inf
        for i in range(0,len(self.data.gene_list)):
            for j in range(i+1,len(self.data.gene_list)):
                gn_dist = self.data.gene_list[i].gene_distance(self.data.gene_list[j])
                dist_mat[self.data.gene_list[i].id - 1][self.data.gene_list[j].id - 1] = gn_dist
                dist_mat[self.data.gene_list[j].id - 1][self.data.gene_list[i].id - 1] = gn_dist

        return dist_mat



