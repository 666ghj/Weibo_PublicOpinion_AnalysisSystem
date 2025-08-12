In some cases, you might be interested in how certain topics are represented over certain categories. Perhaps
there are specific groups of users for which you want to see how they talk about certain topics.

Instead of running the topic model per class, we can simply create a topic model and then extract, for each topic, its representation per class. This allows you to see how certain topics, calculated over all documents, are represented for certain subgroups.

<br>
<div class="svg_image">
--8<-- "docs/getting_started/topicsperclass/class_modeling.svg"
</div>
<br>


To do so, we use the 20 Newsgroups dataset to see how the topics that we uncover are represented in the 20 categories of documents.

First, let's prepare the data:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
docs = data["data"]
targets = data["target"]
target_names = data["target_names"]
classes = [data["target_names"][i] for i in data["target"]]
```

Next, we want to extract the topics across all documents without taking the categories into account:

```python
topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(docs)
```

Now that we have created our global topic model, let us calculate the topic representations across each category:

```python
topics_per_class = topic_model.topics_per_class(docs, classes=classes)
```

The `classes` variable contains the class for each document. Then, we simply visualize these topics per class:

```python
topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10)
```
<iframe src="topics_per_class.html" style="width:1000px; height: 1100px; border: 0px;""></iframe>

You can hover over the bars to see the topic representation per class.

As you can see in the visualization above, the topics `93_homosexual_homosexuality_sex` and `58_bike_bikes_motorcycle`
are somewhat distributed over all classes.

You can see that the topic representation between rec.motorcycles and rec.autos in `58_bike_bikes_motorcycle` clearly
differs from one another. It seems that BERTopic has tried to combine those two categories into a single topic. However,
since they do contain two separate topics, the topic representation in those two categories differs.

We see something similar for `93_homosexual_homosexuality_sex`, where the topic is distributed among several categories
and is represented slightly differently.

Thus, you can see that although in certain categories the topic is similar, the way the topic is represented can differ.
