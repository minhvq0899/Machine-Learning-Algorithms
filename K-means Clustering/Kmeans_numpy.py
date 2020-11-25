"""
================================================== K-means Clustering ==================================================
        
All credit to Tiep Vu (the author of 'Machine Learning co ban') and Aurélien Geron (the author of Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow)
I am replicating their code (with a few changes) and adding comments along the way for a studying purpose only

Content of this code:
1. Create toy data
    1.1. Tiep Vu's toy data
    1.2. Aurélien Geron's toy data
2. Build K-mean algorithms using only numpy 
3. Testing
    3.1. Test on X_tiepvu using functions above
    3.2. Test on X_tiepvu using Scikit-learn Kmeans 

""" 
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
np.random.seed(11) 
from sklearn.cluster import KMeans

# =========================================================== Toy data ===========================================================
# -------------------------- Tiep Vu's toy data --------------------------
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
# print(X0)

X_tiepvu = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label_tiepvu = np.asarray([0]*N + [1]*N + [2]*N).T

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

    # plt.axis('equal')
    plt.plot()
    plt.show()


# -------------------------- Aurélien Geron's toy data --------------------------
from sklearn.datasets import make_blobs

blob_centers = np.array(
    [[ 2 , 2.3],
     [ 0 , 2.3],
     [-1.6, 1.8],
     [-4.2, 2.2],
     [ -3 , 1.3]])
blob_std = np.array([0.4, 0.3, 0.3, 0.25, 0.25])

X_ageron, original_label_ageron = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)

# plot
def plot_clusters(X, ylabel = None):
    plt.figure(figsize=(12, 6))
    plt.scatter(X[:, 0], X[:, 1], c=ylabel, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    plt.show()




# ================================ Build K-mean algorithms using only numpy - Tiep Vu ================================
from scipy.spatial.distance import cdist

# this algorithm will pick initial centers at random
# in more advanced K-mean, there will be a way to better initialize centers -> coverge faster and more accurately
def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]

# go through every point in X and assign each of them a label 
def kmeans_assign_labels(X, centers):
    # calculate pairwise distances between data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis = 1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster 
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))


# imput will be X and K - number of clusters
def kmeans(X, K):
    # initialize centers randomly 
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0 
    # keep updating until coverage
    while True:
        # assign label to each instance in X
        new_labels = kmeans_assign_labels(X, centers[-1])
        labels.append(new_labels)
        # update centers
        new_centers = kmeans_update_centers(X, labels[-1], K)
        # check for coverage 
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    
    return (centers, labels, it)




# ========================================================== Testing ==========================================================

if __name__ == "__main__":
    # plot data
    kmeans_display(X_tiepvu, original_label_tiepvu)
    plot_clusters(X_ageron, original_label_ageron)

    # ------------------------- Test on X_tiepvu using functions above -------------------------
    (centers_tiepvu, labels_tiepvu, it_tiepvu) = kmeans(X_tiepvu, K) # K = 3
    # kmeans_display(X_tiepvu, labels_tiepvu[-1])

    # -------------------------- Test on X_tiepvu using Scikit-learn Kmeans --------------------------
    kmeans_tiepvu = KMeans(n_clusters=3, random_state=0).fit(X_tiepvu)

    print("Result for Tiep Vu\n")
    print('Centers initialized: ', means, '\n')
    print('Centers found by our algorithm: ', centers_tiepvu[-1], '\n')
    print('Centers found by scikit-learn:', kmeans_tiepvu.cluster_centers_, '\n')
    
    # pred_label = kmeans.predict(X)
    # kmeans_display(X, pred_label)




    # ------------------------- Test on X_ageron using functions above -------------------------
    print('-----------------------------------------------------------------------------------------')
    (centers_ageron, labels_ageron, it_ageron) = kmeans(X_ageron, 5) 
    # plot_clusters(X_ageron, labels_ageron[-1])

    # -------------------------- Test on X_ageron using Scikit-learn Kmeans --------------------------
    kmeans_ageron = KMeans(n_clusters=5, random_state=0).fit(X_ageron)

    print("Result for Geron\n")
    print('Centers initialized: ', blob_centers, '\n')
    print('Centers found by our algorithm: ', centers_ageron[-1], '\n')
    print('Centers found by scikit-learn:', kmeans_ageron.cluster_centers_, '\n')
    print('Label found by our algorithm:', labels_ageron[-1], '\n')
    
    # pred_label = kmeans.predict(X)
    # kmeans_display(X, pred_label)









