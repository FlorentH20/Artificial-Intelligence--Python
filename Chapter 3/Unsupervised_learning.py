import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

#import the data
X = np.loadtxt('data_clustering.txt', delimiter=',')
num_clusters = 5

#Plot input
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolor = 'none', edgecolors='black', s=80)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() +1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title("Input data")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
#Kmeans object
k_means = KMeans(init='k-means++', n_clusters = num_clusters, n_init = 10)
k_means.fit(X)

#Making a mesh
step_size = 0.01

#Defining grid points to plot boundaries
x_min, x_max  = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max,step_size), np.arange(y_min, y_max,step_size))

#Predict output labels for all points on the grid
output = k_means.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

#Plot different regions
plt.figure()
plt.clf()
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolor = 'none', edgecolors='black', s=80)
#Plot the centers of regions
clusters_center = k_means.cluster_centers_
plt.scatter(clusters_center[:, 0], clusters_center[:, 1], marker='o', s=210, linewidths=4, color='black', zorder=12, facecolor='black')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Boundaries of clusters')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
