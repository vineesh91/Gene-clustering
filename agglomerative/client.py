from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_similarity_score
import jaccard
import pylab as pl
import pcaplot
import hierarchical
import cho

clstr_cnt = 5
# data = cho.Cho('C:\\Users\\vinee\\Documents\\Fall2018\\DataMining\\Proj2\\new_dataset_2.txt')
data = cho.Cho('C:\\Users\\vinee\\Documents\\Fall2018\\DataMining\\Proj2\\cho.txt')
hclstr = hierarchical.Hierarchical(clstr_cnt,data).final_cluster
# a = np.empty((0,data.feature_length+1),float)
final_data = []
i = 0

final_cluster_list = []

for cluster_ids in hclstr:
    for vals in hclstr[cluster_ids]:
        final_cluster_list.append(cluster_ids)
        vals.attributes.append(i)
        final_data.append(vals.attributes)
        # a = np.append(a,np.asarray(vals.attributes),axis=0)
    i += 1

scr = jaccard_similarity_score(data.original_labels,final_cluster_list)

with open('hierarichal_out.txt', 'w+') as f:
    for _list in final_data:
        for _string in _list:
            #f.seek(0)
            f.write(str(_string) + '\n')
    f.close()
clstr_list = [x for x in range(0,clstr_cnt)]
# pcaplot.plot_pca('hierarichal_out.txt',clstr_list,final_data)


pca = PCA(n_components=2).fit(final_data)
pca_2d = pca.transform(final_data)

#reference from pca tutorial
for i in range(0, pca_2d.shape[0]):
    if final_data[i][-1] == 0:
        c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r')
    elif final_data[i][-1] == 1:
        c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g')
    elif final_data[i][-1] == 2:
        c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='y')
    elif final_data[i][-1] == 3:
        c4 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b')
    elif final_data[i][-1] == 4:
        c5 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='m')
pl.legend([c1, c2,c3, c4, c5], ['Class0', 'Class1','Class2','Class3','Class4'])
pl.title('Cho')
pl.show()
i = 1

# for i in range(0, pca_2d.shape[0]):
#     if final_data[i][-1] == 0:
#         c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r')
#     elif final_data[i][-1] == 1:
#         c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g')
# pl.legend([c1, c2], ['Class0', 'Class1'])
# pl.title('New dataset')
# pl.show()