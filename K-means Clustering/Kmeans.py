"""
================================================== K-means Clustering ==================================================
        
All credit to Tiep Vu (the author of 'Machine Learning co ban') and Aurélien Geron (the author of Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow)
I am replicating their code (with a few changes) and adding comments along the way for a studying purpose only

Content of this code:
1. Create toy data
    1.1 Tiep Vu's toy data
    1.2 Aurélien Geron's toy data
2. 

""" 
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11) 

# =========================================================== Toy data ===========================================================
# -------------------------- Tiep Vu's toy data --------------------------
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
print(X0)

X_tiepvu = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

# plot
def kmeans_display(X, label):
    # K = np.amax(label) + 1
    # get all X0, X1, X2
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
    
kmeans_display(X_tiepvu, original_label)


# -------------------------- Aurélien Geron's toy data --------------------------
from sklearn.datasets import make_blobs

blob_centers = np.array(
    [[ 2 , 2.3],
     [ 0 , 2.3],
     [-1.6, 1.8],
     [-4.2, 2.2],
     [ -3 , 1.3]])
blob_std = np.array([0.4, 0.3, 0.3, 0.25, 0.25])

X_ageron, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)

# plot
def plot_clusters(X):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)

plt.figure(figsize=(12, 6))
plot_clusters(X_ageron)
plt.show()


