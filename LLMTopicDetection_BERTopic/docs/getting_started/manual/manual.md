Although topic modeling is typically done by discovering topics in an unsupervised manner, there might be times when you already have a bunch of clusters or classes from which you want to model the topics. For example, the often used [20 NewsGroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) is already split up into 20 classes. Here, we might want to see how we can transform those 20 classes into 20 topics. Instead of using BERTopic to discover previously unknown topics, we are now going to manually pass them to BERTopic without actually learning them.

We can view this as a manual topic modeling approach. There is no underlying algorithm for detecting these topics since you already have done that before. Whether that is simply because they are already available, like with the 20 NewsGroups dataset, or maybe because you have created clusters of documents before using packages like [human-learn](https://github.com/koaning/human-learn), [bulk](https://github.com/koaning/bulk), [thisnotthat](https://github.com/TutteInstitute/thisnotthat) or something entirely different.

In other words, we can pass our labels to BERTopic and it will try to transform those labels into topics by running the c-TF-IDF representations on the set of documents within each label. This process allows us to model the topics themselves and similarly gives us the option to use everything BERTopic has to offer.

<br>
<div class="svg_image">
--8<-- "docs/getting_started/manual/pipeline.svg"
</div>
<br>

To do so, we need to skip over the dimensionality reduction and clustering steps since we already know the labels for our documents. We can use the documents and labels from the 20 NewsGroups dataset to create topics from those 20 labels:


```python
from sklearn.datasets import fetch_20newsgroups

# Get labeled data
data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
docs = data['data']
y = data['target']
```

Then, we make sure to create empty instances of the dimensionality reduction and clustering steps. We pass those to BERTopic to simply skip over them and go to the topic representation process:


```python
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.cluster import BaseCluster
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction

# Prepare our empty sub-models and reduce frequent words while we are at it.
empty_embedding_model = BaseEmbedder()
empty_dimensionality_model = BaseDimensionalityReduction()
empty_cluster_model = BaseCluster()
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# Fit BERTopic without actually performing any clustering
topic_model= BERTopic(
        embedding_model=empty_embedding_model,
        umap_model=empty_dimensionality_model,
        hdbscan_model=empty_cluster_model,
        ctfidf_model=ctfidf_model
)
topics, probs = topic_model.fit_transform(docs, y=y)
```

Let's take a look at a few topics that we get out of training this way by running `topic_model.get_topic_info()`:

<br>
<div class="svg_image">
--8<-- "docs/getting_started/manual/table.svg"
</div>
<br>

We can see several interesting topics appearing here. They seem to relate to the 20 classes we had as input. Now, let's map those topics to our original classes to view their relationship:

```python
# Map input `y` to topics
mappings = topic_model.topic_mapper_.get_mappings()
mappings = {value: data["target_names"][key] for key, value in mappings.items()}

# Assign original classes to our topics
df = topic_model.get_topic_info()
df["Class"] = df.Topic.map(mappings)
df
```

<br>
<div class="svg_image">
--8<-- "docs/getting_started/manual/table_classes.svg"
</div>
<br>

We can see that the c-TF-IDF representations nicely extract the words that give a nice representation of our input classes. This is all done without actually embedding and clustering the data.

As a result, the entire "training" process only takes a couple of seconds. Moreover, we can still perform BERTopic-specific features like dynamic topic modeling, topics per class, hierarchical topic modeling, modeling topic distributions, etc.

!!! note
    The resulting `topics` may be a different mapping from the `y` labels. To map `y` to `topics`, we can run the following:


    ```python
    mappings = topic_model.topic_mapper_.get_mappings()
    y_mapped = [mappings[val] for val in y]
    ```
