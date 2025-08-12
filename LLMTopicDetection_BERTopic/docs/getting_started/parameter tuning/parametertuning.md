# Hyperparameter Tuning

Although BERTopic works quite well out of the box, there are a number of hyperparameters to tune according to your use case.
This section will focus on important parameters directly accessible in BERTopic but also hyperparameter optimization in sub-models
such as HDBSCAN and UMAP.

## **BERTopic**

When instantiating BERTopic, there are several hyperparameters that you can directly adjust that could significantly improve the performance of your topic model. In this section, we will go through the most impactful parameters in BERTopic and directions on how to optimize them.

### **language**
The `language` parameter is used to simplify the selection of models for those who are not familiar with sentence-transformers models.

In essence, there are two options to choose from:

* `language = "english"` or
* `language = "multilingual"`

The English model is "all-MiniLM-L6-v2" and can be found [here](https://www.sbert.net/docs/pretrained_models.html). It is the default model that is used in BERTopic and works great for English documents.

The multilingual model is "paraphrase-multilingual-MiniLM-L12-v2" and supports over 50+ languages which can be found [here](https://www.sbert.net/docs/pretrained_models.html). The model is very similar to the base model but is trained on many languages and has a slightly different architecture.

### **top_n_words**

`top_n_words` refers to the number of words per topic that you want to be extracted. In practice, I would advise you to keep this value below 30 and preferably between 10 and 20. The reasoning for this is that the more words you put in a topic the less coherent it can become. The top words are the most representative of the topic and should be focused on.

### **n_gram_range**
The `n_gram_range` parameter refers to the CountVectorizer used when creating the topic representation. It relates to the number of words you want in your topic representation. For example, "New" and "York" are two separate words but are often used as "New York" which represents an n-gram of 2. Thus, the `n_gram_range` should be set to (1, 2) if you want "New York" in your topic representation.

### **min_topic_size**
`min_topic_size` is an important parameter! It is used to specify what the minimum size of a topic can be. The lower this value the more topics are created. If you set this value too high, then it is possible that simply no topics will be created! Set this value too low and you will get many microclusters.

It is advised to play around with this value depending on the size of your dataset. If it nears a million documents, then it is advised to set it much higher than the default of 10, for example, 100 or even 500.

### **nr_topics**
`nr_topics` can be a tricky parameter. It specifies, after training the topic model, the number of topics that will be reduced. For example, if your topic model results in 100 topics but you have set `nr_topics` to 20 then the topic model will try to reduce the number of topics from 100 to 20.

This reduction can take a while as each reduction in topics activates a c-TF-IDF calculation. If this is set to None, no reduction is applied. Use "auto" to automatically reduce topics using HDBSCAN.

### **low_memory**
`low_memory` sets UMAP's `low_memory` to True to make sure that less memory is used in the computation. This slows down computation but allows UMAP to be run on low-memory machines.

### **calculate_probabilities**
`calculate_probabilities` lets you calculate the probabilities of each topic in each document. This is computationally quite expensive and is turned off by default.

## **UMAP**

UMAP is an amazing technique for dimensionality reduction. In BERTopic, it is used to reduce the dimensionality of document embedding into something easier to use with HDBSCAN to create good clusters.

However, it does has a significant number of parameters you could take into account. As exposing all parameters in BERTopic would be difficult to manage, we can instantiate our UMAP model and pass it to BERTopic:

```python
from umap import UMAP

umap_model = UMAP(n_neighbors=15, n_components=10, metric='cosine', low_memory=False)
topic_model = BERTopic(umap_model=umap_model).fit(docs)
```

### **n_neighbors**
`n_neighbors` is the number of neighboring sample points used when making the manifold approximation. Increasing this value typically results in a
more global view of the embedding structure whilst smaller values result in a more local view. Increasing this value often results in larger clusters
being created.

### **n_components**
`n_components` refers to the dimensionality of the embeddings after reducing them. This is set as a default to `5` to reduce dimensionality
as much as possible whilst trying to maximize the information kept in the resulting embeddings. Although lowering or increasing this value influences the quality of embeddings, its effect is largest on the performance of HDBSCAN. Increasing this value too much and HDBSCAN will have a
hard time clustering the high-dimensional embeddings. Lower this value too much and too little information in the resulting embeddings are available
to create proper clusters. If you want to increase this value, I would advise setting using a metric for HDBSCAN that works well in high dimensional data.

### **metric**
`metric` refers to the method used to compute the distances in high dimensional space. The default is `cosine` as we are dealing with high dimensional data. However, BERTopic is also able to use any input, even regular tabular data, to cluster the documents. Thus, you might want to change the metric
to something that fits your use case.

### **low_memory**
`low_memory` is used when datasets may consume a lot of memory. Using millions of documents can lead to memory issues and setting this value to `True`
might alleviate some of the issues.

## **HDBSCAN**
After reducing the embeddings with UMAP, we use HDBSCAN to cluster our documents into clusters of similar documents. Similar to UMAP, HDBSCAN has many parameters that could be tweaked to improve the cluster's quality.

```python
from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', prediction_data=True)
topic_model = BERTopic(hdbscan_model=hdbscan_model).fit(docs)
```

### **min_cluster_size**
`min_cluster_size` is arguably the most important parameter in HDBSCAN. It controls the minimum size of a cluster and thereby the number of clusters
that will be generated. It is set to `10` as a default. Increasing this value results in fewer clusters but of larger size whereas decreasing this value
results in more micro clusters being generated. Typically, I would advise increasing this value rather than decreasing it.

### **min_samples**
`min_samples` is automatically set to `min_cluster_size` and controls the number of outliers generated. Setting this value significantly lower than
`min_cluster_size` might help you reduce the amount of noise you will get. Do note that outliers are to be expected and forcing the output
to have no outliers may not properly represent the data.

### **metric**
`metric`, like with HDBSCAN is used to calculate the distances. Here, we went with `euclidean` as, after reducing the dimensionality, we have
low dimensional data and not much optimization is necessary. However, if you increase `n_components` in UMAP, then it would be advised to look into
metrics that work with high dimensional data.

### **prediction_data**
Make sure you always set this value to `True` as it is needed to predict new points later on. You can set this to False if you do not wish to predict
any unseen data points.
