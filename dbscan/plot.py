# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 10:22:40 2018

@author: S.Dejbord
"""

from matplotlib import pyplot as plt
   
def D2_pca_plot(X_pca, labels):
    colors = ['red', 'blue', 'green', 'black', 'purple', 'lime',\
              'cyan', 'orange', 'yellow', 'brown', 'olive', 'gray', 'pink']

    for i in range(len(labels)):
        try:
            x = X_pca.loc[labels[i]][0]
            y = X_pca.loc[labels[i]][1]

            if x.size > 1:
                px = X_pca.loc[labels[i]][0].values
            else:
                px = x
            if y.size > 1:
                py = X_pca.loc[labels[i]][1].values
            else:
                py = y
            plt.scatter(px, py, c = colors[i])
        except :
            print ""
        
    plt.legend(labels)
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    
