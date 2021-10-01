# File that tests the elbow point in order to choose K for K-Means algorithm

import seaborn as sns
import matplotlib.pyplot as plt

from organizer_kmeans import features
from sklearn.cluster import KMeans

wcss = list()
for i in range(1, 50):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1,50), wcss)
plt.title('Selecting the Numbeer of Clusters using the Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()