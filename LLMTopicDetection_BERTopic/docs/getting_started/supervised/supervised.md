Although topic modeling is typically done by discovering topics in an unsupervised manner, there might be times when you already have a bunch of clusters or classes from which you want to model the topics. For example, the often used [20 NewsGroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) is already split up into 20 classes. Similarly, you might already have created some labels yourself through packages like [human-learn](https://github.com/koaning/human-learn), [bulk](https://github.com/koaning/bulk), [thisnotthat](https://github.com/TutteInstitute/thisnotthat) or something entirely different.

Instead of using BERTopic to discover previously unknown topics, we are now going to manually pass them to BERTopic and try to learn the relationship between those topics and the input documents.

> In other words, we are going to be performing classification instead!

We can view this as a supervised topic modeling approach. Instead of using a clustering algorithm, we are going to be using a classification algorithm instead.

Generally, we have the following pipeline:

<br>
<div class="svg_image">
--8<-- "docs/getting_started/supervised/default_pipeline.svg"
</div>
<br>

Instead, we are now going to skip over the dimensionality reduction step and replace the clustering step with a classification model:

<br>
<div class="svg_image">
--8<-- "docs/getting_started/supervised/classification_pipeline.svg"
</div>
<br>

In other words, we can pass our labels to BERTopic and it will not only learn how to predict labels for new instances, but it also transforms those labels into topics by running the c-TF-IDF representations on the set of documents within each label. This process allows us to model the topics themselves and similarly gives us the option to use everything BERTopic has to offer.

To do so, we need to skip over the dimensionality reduction step and replace the clustering step with a classification algorithm. We can use the documents and labels from the 20 NewsGroups dataset to create topics from those 20 labels:


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
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression

# Get labeled data
data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
docs = data['data']
y = data['target']

# Skip over dimensionality reduction, replace cluster model with classifier,
# and reduce frequent words while we are at it.
empty_dimensionality_model = BaseDimensionalityReduction()
clf = LogisticRegression()
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# Create a fully supervised BERTopic instance
topic_model= BERTopic(
        umap_model=empty_dimensionality_model,
        hdbscan_model=clf,
        ctfidf_model=ctfidf_model
)
topics, probs = topic_model.fit_transform(docs, y=y)
```

Let's take a look at a few topics that we get out of training this way by running `topic_model.get_topic_info()`:

<br>
<div class="svg_image">
--8<-- "docs/getting_started/supervised/table.svg"
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
<div class="svg_image">
--8<-- "docs/getting_started/supervised/table_classes.svg"
</div>

<br>

We can see that the c-TF-IDF representations extract the words that give a good representation of our input classes. This is all done directly from the labeling. A welcome side-effect is that we now have a classification algorithm that allows us to predict the topics of unseen data:

```python
>>> topic, _ = topic_model.transform("this is a document about cars")
>>> topic_model.get_topic(topic)
[('car', 0.4407600315538472),
 ('cars', 0.32348015696446325),
 ('engine', 0.28032518444946686),
 ('ford', 0.2500224508115155),
 ('oil', 0.2325984913598611),
 ('dealer', 0.2310723968585826),
 ('my', 0.22045777551991935),
 ('it', 0.21327993649430219),
 ('tires', 0.20420842634292657),
 ('brake', 0.20246902481367085)]
```

 Moreover, we can still perform BERTopic-specific features like dynamic topic modeling, topics per class, hierarchical topic modeling, modeling topic distributions, etc.

!!! note
    The resulting `topics` may be a different mapping from the `y` labels. To map `y` to `topics`, we can run the following:


    ```python
    mappings = topic_model.topic_mapper_.get_mappings()
    y_mapped = [mappings[val] for val in y]
    ```
