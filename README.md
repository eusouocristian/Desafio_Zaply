# **Desafio Zaply: Desenvolvimento**
##### Link das instruções [aqui](https://ruddy-radius-5a5.notion.site/Desafio-Zaply-Machine-Learning-09-2021-a4783deab5534d5a89db06e530fcd690)
## **Candidato**: Cristian Figueiredo dos Santos


Foram realizadas algumas estratégias diferentes buscando categorizar os items de *products.csv* :  


### 1) Clustering usando K-Means
No código [organizer_kmeans.py](organizer_kmeans.py) foram utilizadas a biblioteca para *Natural Language* [NLTK](https://www.nltk.org/), [SkLearn](https://scikit-learn.org/stable/) e [Pandas](https://pandas.pydata.org/) para trabalhar com as séries de dados.
Inicialmente foi realizado um pré-processamento dos dados da coluna 'name' em  *products.csv*  incluindo na lista de stop-words a lista de cores do português e uma série de números sequenciais de 0 a 100. 

Tendo em vista que as palavras iniciais do nome de cada produto tem um maior grau de importância no contexto da categorização, foram mantidas as 'n' palavras iniciais e removidas as demais, do nome de cada produto.

Foi utilizado o critério 'elbow-method' para definição do número k do algoritmo de clustering. O código [elbow_method.py](elbol_method.py) utiliza a biblioteca Matplotlib para visualização os dados, resultando em um 'elbow' aproximado de 18.

Os resultados de cada 'cluster' resultante do ajuste de dados no modelo k-means é inserido na pasta [output-kmeans/](output-kmeans/). Nela estão contidos os arquivos com o conteúdo de cada 'cluster' resultando em um agrupamento de dados por similaridade de texto. Observando o resultado se pode observar que a técnica não é a ideal para esse tipo de problema.

#
### 2) Clustering usando DBScan

O código [organizer_dbscan.py](organizer_dbscan.py) recebe os dados já criados em [organizer_kmeans.py](organizer_kmeans.py) e processa o ajuste do modelo usando outra técnica. Essa estratégia se mostrou ainda pior, pois é esperado que existam features que não tenham similaridade entre outras, resultando em um cluster de arquivos não classificados. No caso de classificação de itens de supermercado, a solução é menos adequada que k-means. 

O resultado do agrupamento de itens similares pelo método DBScan é inserido na pasta [output-dbscan/](output-dbscan/), sendo um arquivo para grupo de itens.

#
### 3) Classificação de dados usando Multinomial Naive Bayes

Segundo a documentação da biblioteca Scikit-Learn, na seção [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB) é provável que o modelo supervisionado contendo dados de treinamento resolva o problema de maneira muito mais eficaz.
No código [organizer_NB.py](organizer_NB.py) foram definidas uma série de categorias desejadas para os dados de supermercado, e uma pequena série de dados [train.xlsx](train.xlsx) foi importada com a biblioteca Pandas.

Os dados de treinamento foram pré-processados bem como os dados de teste contidos na coluna 'name' de [products.csv](products.csv). 

Como resultado da execução do código houve um ValueError por conta da diferença de dimensionalidade entre os dados treinados e os dados de teste. As alternativas que foram buscadas até o término do tempo para conclusão desse desafio foram baseadas na busca por séries de dados com uma dimensão muito maior que a dimensão dos dados de teste, de forma que não ocorra nenhum erro no ajuste dos dados pelo modelo.









