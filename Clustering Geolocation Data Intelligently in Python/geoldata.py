import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

from ipywidgets import interactive

from collections import defaultdict

import hdbscan
import folium
import re

cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363b8', '#f58231', '#911eb4',
        '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
        '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
        '#000075', '#808080']*10

df = pd.read_csv("taxi_data.csv")
# print(df.head())

# print(df.duplicated(subset=['LON', 'LAT']).values.any())
# print(df.isna().values.any())

# print(f'Before dropping NaNs and dupes\t:\tdf.shape = {df.shape}')
# print(df.dropna(inplace=True))
# print(df.drop_duplicates(subset=['LON', 'LAT'], keep='first', inplace=True))
# print(f'After dropping NaNs and dupes\t:\tdf.shape = {df.shape}')

# print(df.head())

X = np.array(df[['LON', 'LAT']], dtype='float64')
#  print(plt.scatter(X[:, 0], X[:, 1], alpha=0.2, s=50))

m = folium.Map(location=[df.LAT.mean(), df.LON.mean()], zoom_start=9, tiles='Stamen Toner')
#for _, row in df.iterrows():
#        folium.CircleMarker(
#                location=[row.LON, row.LAT],
#                radius=5,
#                popup=re.sub(r'[^a-zA-Z ]+', '', row.NAME),
#                color='#1787FE',
#                fill=True,
#                fill_colour='#1787FE'
#        ).add_to(m)
#print(m)

X_blobs, _ = make_blobs(n_samples=1000, centers=10, n_features=2,
                       cluster_std=0.5, random_state=4)

plt.scatter(X_blobs[:, 0], X_blobs[:, 1], alpha=0.2)

class_predictions = np.load('sample_clusters.npy')
#print(class_predictions)


unique_clusters = np.unique(class_predictions)
for unique_cluster in unique_clusters:
    X = X_blobs[class_predictions == unique_cluster]
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2, c=cols[unique_cluster])

#print(silhouette_score(X_blobs, class_predictions))

class_predictions = np.load('sample_clusters_improved.npy')
unique_clusters = np.unique(class_predictions)
for unique_cluster in unique_clusters:
    X = X_blobs[class_predictions==unique_cluster]
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2, c=cols[unique_cluster])

#print(silhouette_score(X_blobs, class_predictions))

X_blobs, _ = make_blobs(n_samples=1000, centers=50,
                        n_features=2, cluster_std=1, random_state=4)

data = defaultdict(dict)
for x in range(1, 21):
        model = KMeans(n_clusters=3, random_state=17,
                       max_iter=x, n_init=1).fit(X_blobs)

        data[x]['class_predictions'] = model.predict(X_blobs)
        data[x]['centroids'] = model.cluster_centers_
        data[x]['unique_classes'] = np.unique(class_predictions)

def f(x):
    class_predictions = data[x]['class_predictions']
    centroids = data[x]['centroids']
    unique_classes = data[x]['unique_classes']

    for unique_class in unique_classes:
            plt.scatter(X_blobs[class_predictions==unique_class][:,0],
                        X_blobs[class_predictions==unique_class][:,1],
                        alpha=0.3, c=cols[unique_class])
    plt.scatter(centroids[:,0], centroids[:,1], s=200, c='#000000', marker='v')
    plt.ylim([-15,15]); plt.xlim([-15,15])
    plt.title('How K-Means Clusters')

interactive_plot = interactive(f, x=(1, 20))
output = interactive_plot.children[-1]
output.layout.height = '350px'
#print(interactive_plot)

X = np.array(df[['LON', 'LAT']], dtype='float64')
k = 70
model = KMeans(n_clusters=k, random_state=17).fit(X)
class_predictions = model.predict(X)
df[f'CLUSTER_kmeans{k}'] = class_predictions


def create_map(df, cluster_column):
        m = folium.Map(location=[df.LAT.mean(), df.LON.mean()], zoom_start=9, tiles='Stamen Toner')

        for _, row in df.iterrows():
                cluster_colour = cols[row[cluster_column]]
                folium.CircleMarker(
                        location=[[row.LAT, row.LON]],
                        radius=5,
                        popup=row[cluster_column],
                        color=cluster_colour,
                        fill=True,
                        fill_color=cluster_colour
                ).add_to(m)
        return m


#print(f'K={k}')
#print(f'Silhouette Score: {silhouette_score(X, class_predictions)}')

m.save('kmeans_70.html')

best_silhouette, best_k = -1, 0

for k in tqdm(range(2, 100)):
        model = KMeans(n_clusters=k, random_state=1).fit(X)
        class_predictions = model.predict(X)

        curr_silhouette = silhouette_score(X, class_predictions)
        if curr_silhouette > best_silhouette:
                best_k = k
                best_silhouette = curr_silhouette

#print(f'K={best_k}')
#print(f'Silhouette Score: {best_silhouette}')


# code for indexing out certain values
dummy = np.array([-1, -1, -1, 2, 3, 4, 5, -1])

new = np.array([ (counter+2)*x if x == -1 else x for counter, x in enumerate(dummy)])

#print(new)
