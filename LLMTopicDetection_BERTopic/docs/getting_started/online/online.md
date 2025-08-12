Online topic modeling (sometimes called "incremental topic modeling") is the ability to learn incrementally from a mini-batch of instances. Essentially, it is a way to update your topic model with data on which it was not trained before. In Scikit-Learn, this technique is often modeled through a `.partial_fit` function, which is also used in BERTopic.

!!! Tip
    Another method for online topic modeling can be found with the [**.merge_models**](https://maartengr.github.io/BERTopic/getting_started/merge/merge.html) functionality of BERTopic. It allows for merging multiple BERTopic models to create a single new one. This method can be used to discover new topics by training a new model and exploring whether that new model added new topics to the original model when merging. A major benefit, compared to `.partial_fit` is that you can keep using the original UMAP and HDBSCAN models which tends result in improved performance and gives you significant more flexibility.

In BERTopic, there are three main goals for using this technique.

* To reduce the memory necessary for training a topic model.
* To continuously update the topic model as new data comes in.
* To continuously find new topics as new data comes in.

In BERTopic, online topic modeling can be a bit tricky as there are several steps involved in which online learning needs to be made available. To recap, BERTopic consists of the following 6 steps:

1. Extract embeddings
2. Reduce dimensionality
3. Cluster reduced embeddings
4. Tokenize topics
5. Extract topic words
6. (Optional) Fine-tune topic words

For some steps, an online variant is more important than others. Typically, in step 1 we use pre-trained language models that are in less need of continuous updates. This means that we can use an embedding model like Sentence-Transformers for extracting the embeddings and still use it in an online setting. Similarly, steps 5 and 6 do not necessarily need online variants since they are built upon step 4, tokenization. If that tokenization is by itself incremental, then so will steps 5 and 6.

<br>
<div class="svg_image">
--8<-- "docs/getting_started/online/online.svg"
</div>
<br>

This means that we will need online variants for steps 2 through 4. Steps 2 and 3, dimensionality reduction and clustering, can be modeled through the use of Scikit-Learn's `.partial_fit` function. In other words, it supports any algorithm that can be trained using `.partial_fit` since these algorithms can be trained incrementally. For example, incremental dimensionality reduction can be achieved using Scikit-Learn's `IncrementalPCA` and incremental clustering with `MiniBatchKMeans`.

Lastly, we need to develop an online variant for step 5, tokenization. In this step, a Bag-of-words representation is created through the `CountVectorizer`. However, as new data comes in, its vocabulary will need to be updated. For that purpose, `bertopic.vectorizers.OnlineCountVectorizer` was created that not only updates out-of-vocabulary words but also implements decay and cleaning functions to prevent the sparse bag-of-words matrix to become too large. Most notably, the `decay` parameter is a value between 0 and 1 to weigh the percentage of frequencies that the previous bag-of-words matrix should be reduced to. For example, a value of `.1` will decrease the frequencies in the bag-of-words matrix by 10% at each iteration. This will make sure that recent data has more weight than previous iterations. Similarly, `delete_min_df` will remove certain words from its vocabulary if their frequency is lower than a set value. This ties together with the `decay` parameter as some words will decay over time if not used. For more information regarding the `OnlineCountVectorizer`, please see the [vectorizers documentation](https://maartengr.github.io/BERTopic/getting_started/vectorizers/vectorizers.html#onlinecountvectorizer).



## **Example**

Online topic modeling in BERTopic is rather straightforward. We first need to have our documents split into chunks such that we can train and update our topic model incrementally.

```python
from sklearn.datasets import fetch_20newsgroups

# Prepare documents
all_docs = fetch_20newsgroups(subset=subset,  remove=('headers', 'footers', 'quotes'))["data"]
doc_chunks = [all_docs[i:i+1000] for i in range(0, len(all_docs), 1000)]
```

Here, we created chunks of 1000 documents to be fed in BERTopic. Then, we will need to define several sub-models that support online learning. Specifically, we are going to be using `IncrementalPCA`, `MiniBatchKMeans`, and the `OnlineCountVectorizer`:

```python
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer

# Prepare sub-models that support online learning
umap_model = IncrementalPCA(n_components=5)
cluster_model = MiniBatchKMeans(n_clusters=50, random_state=0)
vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=.01)
```

After having defined our sub-models, we can start training our topic model incrementally by looping over our document chunks:

```python
from bertopic import BERTopic

topic_model = BERTopic(umap_model=umap_model,
                       hdbscan_model=cluster_model,
                       vectorizer_model=vectorizer_model)

# Incrementally fit the topic model by training on 1000 documents at a time
for docs in doc_chunks:
    topic_model.partial_fit(docs)
```

And that is it! During each iteration, you can access the predicted topics through the `.topics_` attribute.

!!! note
    Do note that in BERTopic it is not possible to use `.partial_fit` after the `.fit` as they work quite differently concerning internally updating topics, frequencies, representations, etc.

!!! tip Tip
    You can use any other dimensionality reduction and clustering algorithm as long as they have a `.partial_fit` function. Moreover, you can use dimensionality reduction algorithms that do not support `.partial_fit` functions but do have a `.fit` function to first train it on a large amount of data and then continuously  add documents. The dimensionality reduction will not be updated but may be trained sufficiently to properly reduce the embeddings without the need to continuously add documents.

!!! warning
    Only the most recent batch of documents is tracked. If you want to be using online topic modeling for low-memory use cases, then it is advised to also update the `.topics_` attribute. Otherwise, variations such as **hierarchical topic modeling** will not work.

    ```python
    # Incrementally fit the topic model by training on 1000 documents at a time and track the topics in each iteration
    topics = []
    for docs in doc_chunks:
        topic_model.partial_fit(docs)
        topics.extend(topic_model.topics_)

    topic_model.topics_ = topics
    ```


## **River**

To continuously find new topics as they come in, we can use the package [river](https://github.com/online-ml/river). It contains several clustering models that can create new clusters as new data comes in. To make sure we can use their models, we first need to create a class that has a `.partial_fit` function and the option to extract labels through `.labels_`:

```python
from river import stream
from river import cluster

class River:
    def __init__(self, model):
        self.model = model

    def partial_fit(self, umap_embeddings):
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            self.model.learn_one(umap_embedding)

        labels = []
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            label = self.model.predict_one(umap_embedding)
            labels.append(label)

        self.labels_ = labels
        return self
```

Then, we can choose any `river.cluster` model that we are interested in and pass it to the `River` class before using it in BERTopic:

```python
# Using DBSTREAM to detect new topics as they come in
cluster_model = River(cluster.DBSTREAM())
vectorizer_model = OnlineCountVectorizer(stop_words="english")
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

# Prepare model
topic_model = BERTopic(
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
)


# Incrementally fit the topic model by training on 1000 documents at a time
for docs in doc_chunks:
    topic_model.partial_fit(docs)
```
