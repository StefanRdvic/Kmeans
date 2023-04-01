# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 00:16:48 2023
 
@author: Stefan Radovanovic & Shaina Bakili
"""
# %%
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# %% class
class Kmeans(object):
   
    def __init__(self,n_clusters = 4,ite_nb = 100):
        self.n_clusters = n_clusters
        self.ite_nb = ite_nb
        
    def fit(self, data):
       self.centroids = pd.DataFrame([np.random.uniform(data.min(), data.max()) for _ in range(self.n_clusters)])
       i = 0
       prev_centroids = pd.DataFrame()
       while not prev_centroids.equals(self.centroids) or i < self.ite_nb:
           distances = pd.DataFrame([np.linalg.norm(data - centroid, axis=1) for centroid in self.centroids.to_numpy()]).T
           cluster = distances.idxmin(axis=1)
           new_centroids = data.groupby(cluster).mean()
           prev_centroids = self.centroids.copy()
           self.centroids.update(new_centroids)
           i += 1
       intraclass_dist = distances.groupby(cluster).sum()
       return self.centroids, cluster, np.diag(intraclass_dist)
   
# %% TO TEST WITH ANY CSV :
df = pd.read_csv('irisV1.csv', header = None)
x = df.drop(df.columns[-1], axis=1) # suppression de la colonne attribut "class"
model = Kmeans(n_clusters=int(input("nombre de cluster souhaité : ")))
centroids, cluster, min_intraclass_dist = model.fit(x.copy())

print('\nCentroids :')
print(centroids.to_string())
print('\nNombre d\'individu par cluster :')
print(cluster.value_counts().to_string())
print('\nDistance intra classe des clusters/centroids:')
print('\n'.join(f'{i} : {j}' for i, j in enumerate(min_intraclass_dist)))


# %% TO TEST WITH sklearn
iris = datasets.load_iris()
x=pd.DataFrame(iris.data)

columns_name = ['Sepal_Length','Sepal_width','Petal_Length','Petal_width']

model = Kmeans(n_clusters=3)
centroids, cluster, min_intraclass_dist = model.fit(x.copy())
x.columns = columns_name
centroids.columns = columns_name
 
colormap = np.array(['purple', 'orange', 'green'])
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 7))
ax1.scatter(x.Petal_Length, x.Petal_width,c=colormap[cluster], s=20)
ax1.scatter(centroids['Petal_Length'], centroids['Petal_width'],s=160, c='black', marker='x')
ax1.set(xlabel='Petal_Length', ylabel='Petal_width')

ax2.scatter(x.Sepal_Length, x.Sepal_width,c=colormap[cluster], s=20)
ax2.scatter(centroids['Sepal_Length'], centroids['Sepal_width'],s=160, c='black', marker='x')
ax2.set(xlabel='Sepal_Length', ylabel='Sepal_width')

plt.show()

print('\nCentroids :')
print(centroids.to_string())
print('\nNombre d\'individu par cluster :')
print(cluster.value_counts().to_string())
print('\nDistance intra classe des clusters/centroids:')
print('\n'.join(f'{i} : {j}' for i, j in enumerate(min_intraclass_dist)))












































































