"""
============================================ K-means Decision Boundaries ============================================
        
All credit to Tiep Vu (the author of 'Machine Learning co ban') and Aurélien Geron (the author of Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow)
I am replicating their code (with a few changes) and adding comments along the way for a studying purpose only

Content of this code:
1. Functions to plot the decision boundaries 
   * plot_data()
   * plot_centroids()
   * plot_decision_boundaries()

2. Main method

""" 

from Kmeans_numpy import X_ageron, X_tiepvu, original_label_ageron, original_label_tiepvu, means as centroids_tiepvu, blob_centers as centroids_ageron 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

centroids_tiepvu = np.asarray(centroids_tiepvu)

# ========================================= Functions to plot the decision boundaries =========================================
# plot the data as black dot
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

# plot centroid
def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    # this means if the weight of the centroid is too small (smaller than max/10), ignore those centroids
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True, show_xlabels=True, show_ylabels=True):
    # print("This is X: {0} with shape {1} \n".format( X_ageron, X_ageron.shape ) )
    mins = X_ageron.min(axis=0) - 0.1
    maxs = X_ageron.max(axis=0) + 0.1
    # print("This is maxs: ", maxs, "\n")
    # print("This is mins: ", mins, "\n\n")
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], 1000),
                         np.linspace(mins[1], maxs[1], 1000))
    # print("This is xx: {0} with the shape {1} \n".format(xx, xx.shape), "\n")
    # print("This is xx.ravel(): {0} with shape {1} \n\n".format(xx.ravel(), xx.ravel().shape))
    # print(np.c_[xx.ravel(), yy.ravel()], np.c_[xx.ravel(), yy.ravel()].shape)

    # ---------------------- important idea for this whole function ----------------------
    # we are predicting the class label for each pixel in the rectangle of space
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # ------------------------------------------------------------------------------------

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidths=1, colors='k')
    plot_data(X)
    
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


if __name__ == "__main__":
    # fit KMeans
    kmeans_ageron = KMeans(n_clusters=5, random_state=0).fit(X_ageron)
    
    # plot the decision boundaries
    plt.figure()
    plot_decision_boundaries(kmeans_ageron, X_ageron)
    plt.show()









