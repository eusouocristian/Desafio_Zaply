import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB


def remove_stopwords(list_input, stopwords_pt):
    # Reconstruct the new version of product name into a list again
    # Removing stopwords from the dataframe
    cleaned_list = list()
    for index, item in enumerate(list_input):
        for i, word in enumerate(item):
            # Pop out of the list if item is in stopwords list
            if word.lower() in stopwords_pt:
                item.pop(i)
        cleaned_list.append(' '.join(list_input[index]))
    return cleaned_list


# Defined categories
categories = [
    'legumes', 'verduras', 'frutas', 'temperos', 
    'congelados', 'frutas-secas', 'massas', 'embutidos',
    'bebidas', 'casa', 'papelaria', 'higiene', 'padaria',
    'carnes', 'doces-biscoitos-guloseimas', 'light-diet'
]

# load a dataset of grocey and convert it into pandas series using squeeze method
# Source: https://www.scribd.com/doc/85871765/lista-de-compras-supermercado-acougue-e-sacolao
train_orig_data = pd.read_excel('train.xlsx', sheet_name=categories)
for index, item in enumerate(categories):
    train_orig_data[categories[index]] = train_orig_data[categories[index]].squeeze()


# Create a list of products related to each category
products_list = [content for i, _ in enumerate(categories) for content in train_orig_data[categories[i]]]
class_list = [item for i, item in enumerate(categories) for _ in train_orig_data[categories[i]]]

# Create a pandas dataframe with known grocery categories dataset
train_dataframe = pd.DataFrame({
    'produto': products_list,
    'categoria': class_list
})


# Import original data
original_dataframe = pd.read_csv("products.csv")

## Define Portuguese as the language to remove stopwords
# Includ some irrelevant portuguese words and numbers
stopwords_pt = stopwords.words('portuguese')

## Start Preprocessing: Get first 'n' words from each product name
n=2
# Split product name string in single words, and get first n words
initial_names = [index.split()[0:n+1] for index in original_dataframe['name']]

initial_names_list = remove_stopwords(initial_names, stopwords_pt=stopwords_pt)

## Construct a Pandas Dataframe from the trimmed product names
dataframe = pd.DataFrame({
    'name': initial_names_list,
    'price': original_dataframe['price'].tolist(),
    'brand': original_dataframe['brand'].tolist(),
})


# Vectorize dataframe
count_vect= CountVectorizer()
X_train_tf = count_vect.fit_transform(train_dataframe['produto'])
# use TF-IDF (term-frequency inverse) to define the most significant words
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)

# Vectorize dataframe 
count_vect2= CountVectorizer()
X_test_tf = count_vect2.fit_transform(dataframe['name'])
# use TF-IDF (term-frequency inverse) to define the most significant words
tfidf_transformer2 = TfidfTransformer()
X_test_tfidf = tfidf_transformer2.fit_transform(X_test_tf.toarray())


model = MultinomialNB().fit(X_train_tfidf, train_dataframe['categoria'])
predict = model.predict(X_test_tfidf)
print(predict)