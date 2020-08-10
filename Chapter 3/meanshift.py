import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

#Load data
X = np.loadtxt('data_clustering.txt', delimiter=',')

#Estimating bandwidth
bandwidth_X = estimate_bandwidth(X, quantile=.1, n_samples=len(X))
#Cluster data with mean shift
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

#Extract the centers of clusters
cluster_centers= meanshift_model.cluster_centers_
print('\nCenters of clusters: ', cluster_centers)

#Extract the number of clusters
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters: ", num_clusters)

#Ploting the points and clusters centers
plt.figure()
markers = 'o*xvs'
for i, marker in zip(range(num_clusters), markers):
    #Plot points that belong to the scatter
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='black')

    #Ploting the means-centroids
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o', markerfacecolor='black', markeredgecolor='black', markersize=15)
plt.title('Clusters')
plt.show()
