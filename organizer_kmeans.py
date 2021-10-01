
import pandas as pd
import os
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Import original data
original_dataframe = pd.read_csv("products.csv")

## Define Portuguese as the language to remove stopwords
# Includ some irrelevant portuguese words and numbers
stopwords_pt = stopwords.words('portuguese')
colors = ['azul', 'verde', 'amarelo', 'roxo', 'marrom', 'branco']
for color in colors:
    stopwords_pt.append(color)
for number in list(range(1000)):
    stopwords_pt.append(number)


## Start Preprocessing: Get first 'n' words from each product name
n=3
initial_names = list() 
initial_names_list = list()
# Split product name string in single words, and get first n words
for name, index in enumerate(original_dataframe['name']):
    initial_names.append(index.split()[0:n+1])
# Reconstruct the new version of product name into a list again
# Removing stopwords from the dataframe
for index, item in enumerate(initial_names):
    for i, word in enumerate(item):
        # Pop out of the list if item is in stopwords list
        if word.lower() in stopwords_pt:
            item.pop(i)
    initial_names_list.append(' '.join(initial_names[index]))


## Construct a Pandas Dataframe from the trimmed product names
dataframe = pd.DataFrame({
    'name': initial_names_list,
    'price': original_dataframe['price'].tolist(),
    'brand': original_dataframe['brand'].tolist(),
})

# Get the name column as Unicode
documents = dataframe['name'].values.astype('U')
# Vectorize (convert words into numbers) not considering stopwords 
vectorizer = TfidfVectorizer(stop_words=stopwords_pt)
features = vectorizer.fit_transform(documents)

# Create K-Means model and run it, with n_clusters = k
k=14
model = KMeans(n_clusters=k, init='k-means++', algorithm='auto', max_iter=1000, n_init=1, tol=0.0001)
model.fit(features)


# Create a new Datafrane column to keep the cluster label  
# And group Dataframe by cluster
dataframe['cluster'] = model.labels_
clusters = dataframe.groupby('cluster')

# Clean output/ folder usin Linux os command
if os.listdir('output-kmeans/'):
    os.system('rm -r output-kmeans/*')
# Generate a set of files, one file for each cluster, into output folder
for cluster in clusters.groups:
    f = open('output-kmeans/'+'cluster'+str(cluster)+'.csv', 'w')
    data = clusters.get_group(cluster)[['name', 'price', 'brand']]
    f.write(data.to_csv(index_label='id'))
    f.close()


print('Clusters Centroids: \n')
order_centroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names()

for i in range(k):
    print(f'CLUSTER {i} \n')
    for j in order_centroids[i, :10]:
        print(f'{terms[j]}')
    print('-----------------------')




