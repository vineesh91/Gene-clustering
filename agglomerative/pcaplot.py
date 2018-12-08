from sklearn.decomposition import PCA
import matplotlib as plt
def plot_pca(filename, classes_list, feat_matrix):
    # get the unique list of classes
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

    plt.title("scatter plot of " + filename)
    plt.xlabel("Principle component 1")
    plt.ylabel("Principle component 2")
    plt.legend()
    plt.show()
