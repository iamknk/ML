# hierarical Clustering

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[: , [3,4]].values

# Using dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#Fitting hierarhical clustering to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


#Visualizing Clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s= 100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s= 100, c = 'green', label = 'Right')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s= 100, c = 'blue', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s= 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s= 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual Income in US DOLLARS')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


