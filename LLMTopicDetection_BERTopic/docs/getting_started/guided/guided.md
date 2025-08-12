Guided Topic Modeling or Seeded Topic Modeling is a collection of techniques that guides the topic modeling approach by setting several seed topics to which the model will converge to. These techniques allow the user to set a predefined number of topic representations that are sure to be in documents. For example, take an IT business that has a ticket system for the software their clients use. Those tickets may typically contain information about a specific bug regarding login issues that the IT business is aware of.

To model that bug, we can create a seed topic representation containing the words `bug`, `login`, `password`,
and `username`. By defining those words, a Guided Topic Modeling approach will try to converge at least one topic to those words.

<br>
<div class="svg_image">
--8<-- "docs/getting_started/guided/guided.svg"
</div>
<br>

Guided BERTopic has two main steps:

First, we create embeddings for each seeded topic by joining them and passing them through the document embedder. These embeddings will be compared with the existing document embeddings through cosine similarity and assigned a label. If the document is most similar to a seeded topic, then it will get that topic's label.
If it is most similar to the average document embedding, it will get the -1 label.
These labels are then passed through UMAP to create a semi-supervised approach that should nudge
the topic creation to the seeded topics.

Second, we take all words in seed_topic_list and assign them a multiplier larger than 1.
Those multipliers will be used to increase the IDF values of the words across all topics thereby increasing
the likelihood that a seeded topic word will appear in a topic. This does, however, also increase the chance of an irrelevant topic having unrelated words. In practice, this should not be an issue since the IDF value is likely to remain low regardless of the multiplier. The multiplier is now a fixed value but may change to something more elegant, like taking the distribution of IDF values and its position into account when defining the multiplier.

### **Example**
To demonstrate Guided BERTopic, we use the 20 Newsgroups dataset as our example. We have frequently used this
dataset in BERTopic examples and we sometimes see a topic generated about health with words such as `drug` and `cancer`
being important. However, due to the stochastic nature of UMAP, this topic is not always found.

In order to guide BERTopic to that topic, we create a seed topic list that we pass through our model. However,
there may be several other topics that we know should be in the documents. Let's also initialize those:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))["data"]

seed_topic_list = [["drug", "cancer", "drugs", "doctor"],
                   ["windows", "drive", "dos", "file"],
                   ["space", "launch", "orbit", "lunar"]]

topic_model = BERTopic(seed_topic_list=seed_topic_list)
topics, probs = topic_model.fit_transform(docs)
```

As you can see above, the `seed_topic_list` contains a list of topic representations. By defining the above topics
BERTopic is more likely to model the defined seeded topics. However, BERTopic is merely nudged towards creating those
topics. In practice, if the seeded topics do not exist or might be divided into smaller topics, then they will
not be modeled. Thus, seed topics need to be accurate to accurately converge towards them.
