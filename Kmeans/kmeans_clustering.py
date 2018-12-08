import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def plot_pca(filename1, classes_list, feat_matrix):
    # get unique classes
    class_uniq = list(set(classes_list))

    # obtain the principle components matrix
    pca_object = PCA(n_components=2, svd_solver='full')
    pca_object.fit(feat_matrix)
    prin_comp_mat = pca_object.transform(feat_matrix)

    # scatter plot

    colors_list = [plt.cm.jet(float(i) / max(class_uniq)) for i in class_uniq]
    for i, u in enumerate(class_uniq):
        xi = [p for (j,p) in enumerate(prin_comp_mat[:,0]) if classes_list[j] == u]
        yi = [p for (j,p) in enumerate(prin_comp_mat[:,1]) if classes_list[j] == u]
        plt.scatter(xi, yi, c = colors_list[i], label=str(int(u)))

    plt.title("scatter plot of " + filename1)
    plt.xlabel("Principle component 1")
    plt.ylabel("Principle component 2")
    plt.legend()
    plt.show()


def dict_to_df(data,l,datfr):

    df = pd.DataFrame()
    ls =[]
    for keys, vals in data.items():
        temp = pd.DataFrame.from_dict(datfr[data[keys],:])
        temp['label'] = keys
        df = df.append(temp)
        ls.append(vals)

    ls = [item for sublist in ls for item in sublist]
    df['ind'] = ls
    df = df.set_index('ind')
    #df = df.sort_values(by=['ind'])
    return df


def kmeanscluster(df,cluster_cent,cluster_data,k):
    for row in range(0,len(df)):
        count = 0
        diff_array = np.zeros(k)
        for keys in cluster_cent.keys():
            count += 1
            diff = np.linalg.norm(df[row,:] - cluster_cent[keys])
            diff_array[keys] = diff

        ind = diff_array.argmin()
        if ind not in cluster_data:
            cluster_data[ind] = []
        cluster_data[ind].append(row)

    for key1,val1 in cluster_data.items():
        new_cent = np.mean(df[val1,:], axis = 0)
        cluster_cent[key1] = []
        cluster_cent[key1].append(new_cent)

    return cluster_cent,cluster_data


def jaccard(feat, ground_class, pred_class):
    ground_inc_mat  = np.zeros((len(feat),len(feat)))
    pred_inc_mat = np.zeros((len(feat), len(feat)))

    for row in range(0,len(feat)):
        ground_inc_mat[row][row] = 1
        pred_inc_mat[row][row] = 1
        for col in range(row+1, len(feat)):

            if(ground_class[row] == ground_class[col]):
                ground_inc_mat[row][col] = ground_inc_mat[col][row] = 1

            if (pred_class[row] == pred_class[col]):
                pred_inc_mat[row][col] = pred_inc_mat[col][row] = 1

    #find jaccard similarity
    num = np.sum(np.logical_and(ground_inc_mat, pred_inc_mat))
    den = np.sum(np.logical_or(ground_inc_mat, pred_inc_mat))

    jac = (float(num))/(float(den))
    print("jaccard similarity : " + str(jac))




filename = "new_dataset_1.txt"
#reading the dataset
dataf = pd.read_csv(filename, header=None, sep="\t")
#taking only the attributes
df = np.asarray(dataf.iloc[:,2:])

choice = 2
k = 3
iter_val = 10
#randomly selecting means from dataset for initialization (Forgy method)
#cluster_centers = np.random.choice(len(df),k)
cluster_centers = [3,20,9]

if choice == 1:
    cluster_class = dataf.iloc[:,1]
    plot_pca(filename, list(cluster_class),df)
else:


    cluster_class = dataf.iloc[:, 1]
    cluster_cent = {}
    cluster_data = {}

    ind = 0
    for i in cluster_centers:
        cluster_cent[ind] = df[i,:]
        ind += 1

    count1 = 0
    for iter in range(0,iter_val):
        prev_cent,prev_data = cluster_cent,cluster_data
        cluster_data = {}
        cluster_cent,cluster_data = kmeanscluster(df,cluster_cent,cluster_data,k)


        for i in range(0, k):
            elems1 = cluster_cent[i]
            elems2 = prev_cent[i]
            result = all(elems in elems1 for elems in elems2)

            if result:
                count1 += 1
                if(count1 == 3):

                    df1 = dict_to_df(cluster_data,len(df),df)
                    features = df1.iloc[:, :-1]
                    cluster_class1 = df1.iloc[:, -1]
                    plot_pca(filename, list(cluster_class1), features)
                    jaccard(df, cluster_class, cluster_class1)
                    break
            else:

                continue




