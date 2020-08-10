import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

#Load data
X = np.loadtxt('data_quality.txt', delimiter=',')

#Initialize variables
scores = []
values = np.arange(2, 10)

#Iterate through the defined range
for num_clusters in values:
    #Train the KMeans clusters:
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init = 10)
    kmeans.fit(X)

#Estimate the silhouette score for the current clustering model using Euclidean distance metric
    score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean', sample_size=len(X))
    print('Num of clusters: ', num_clusters)
    print('Silhouette score: ', score)
    scores.append(score)

#Plot silhouette scores
plt.figure()
plt.bar(values, scores, width=.7, color='black', align='center')
plt.title('Silhouette score vs Num of clusters')

# Extract best score and optimal number of clusters
num_clusters = np.argmax(scores) + values[0]
print("\nOptimal number of clusters: ", num_clusters)

#Plot the data
plt.figure()
plt.scatter(X[:,0], X[:,1], color='black', s=80, marker='o', facecolor='none')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() -1, X[:, 1].max() + 1
plt.title("Input data")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
