"""
================================================== Find the optimal number of clusters ==================================================
        
All credit to Tiep Vu (the author of 'Machine Learning co ban') and Aurélien Geron (the author of Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow)
I am replicating their code (with a few changes) and adding comments along the way for a studying purpose only

Content of this code:
# Try to fit X_ageron with 3 and 8 clusters 
# Plot them
# Print inertia
# Elbow technique

"""

from Kmeans_numpy import X_ageron, X_tiepvu, original_label_ageron, original_label_tiepvu, means as centroids_tiepvu, blob_centers as centroids_ageron, kmeans as kmeans_np
from Kmean_decision_boundaries import plot_decision_boundaries 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans as kmeans_scikit


# Try to fit X_ageron with 3 and 8 clusters 
kmeans_ageron_3 = kmeans_scikit(n_clusters=3, random_state=0).fit(X_ageron)
kmeans_ageron_8 = kmeans_scikit(n_clusters=8, random_state=0).fit(X_ageron)

# Plot them
plt.figure(figsize=(26,8))

plt.subplot(121)
plot_decision_boundaries(kmeans_ageron_3, X_ageron)
plt.title("k = 3", fontsize=18)

plt.subplot(122)
plot_decision_boundaries(kmeans_ageron_8, X_ageron)
plt.title("k = 8", fontsize=18)

plt.show()

# Print inertia
print("The inertia of k = 3 is: ", kmeans_ageron_3.inertia_, "\n")
print("The inertia of k = 8 is: ", kmeans_ageron_8.inertia_, "\n")

print("We cannot simply take the value of k that minimizes the inertia, since it keeps getting lower as we increase k.")
print("Indeed, the more clusters there are, the closer each instance will be to its closest centroid, and therefore the")
print("lower the inertia will be. However, we can plot the inertia as a function of k and analyze the resulting curve")


# Elbow technique
kmeans_per_k = [kmeans_scikit(n_clusters=k, random_state=42).fit(X_ageron) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

# Plot Elbow technique
plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow',
             xy=(2, inertias[1]),
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1)
            )
# plt.axis([1, 8.5, 0, 1300])
plt.show()

# try k = 2 
plt.figure()
plot_decision_boundaries(kmeans_per_k[1], X_ageron)
plt.show()

