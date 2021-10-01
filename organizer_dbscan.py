import numpy as np
import os

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler 
from organizer import features, dataframe  

db = DBSCAN(eps=0.3, min_samples=10).fit(features)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_ 

dataframe['cluster'] = labels
clusters = dataframe.groupby('cluster')

if os.listdir('output_dbscan/'):
    os.system('rm -r output_dbscan/*')
for cluster in clusters.groups:
    f = open('output_dbscan/'+'cluster'+str(cluster)+'.csv', 'w')
    data = clusters.get_group(cluster)[['name', 'price', 'brand']]
    f.write(data.to_csv(index_label='id'))
    f.close()