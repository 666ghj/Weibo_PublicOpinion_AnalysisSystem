Visualizing BERTopic and its derivatives is important in understanding the model, how it works, and more importantly, where it works.
Since topic modeling can be quite a subjective field it is difficult for users to validate their models. Looking at the topics and seeing
if they make sense is an important factor in alleviating this issue.

## **Visualize Topics**
After having trained our `BERTopic` model, we can iteratively go through hundreds of topics to get a good
understanding of the topics that were extracted. However, that takes quite some time and lacks a global representation.
Instead, we can visualize the topics that were generated in a way very similar to
[LDAvis](https://github.com/cpsievert/LDAvis).

We embed our c-TF-IDF representation of the topics in 2D using Umap and then visualize the two dimensions using
plotly such that we can create an interactive view.

First, we need to train our model:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
```

Then, we can call `.visualize_topics` to create a 2D representation of your topics. The resulting graph is a
plotly interactive graph which can be converted to HTML:

```python
topic_model.visualize_topics()
```

<iframe src="viz.html" style="width:1000px; height: 680px; border: 0px;""></iframe>

You can use the slider to select the topic which then lights up red. If you hover over a topic, then general
information is given about the topic, including the size of the topic and its corresponding words.


## **Visualize Topic Similarity**
Having generated topic embeddings, through both c-TF-IDF and embeddings, we can create a similarity
matrix by simply applying cosine similarities through those topic embeddings. The result will be a
matrix indicating how similar certain topics are to each other.
To visualize the heatmap, run the following:

```python
topic_model.visualize_heatmap()
```

<iframe src="heatmap.html" style="width:1000px; height: 720px; border: 0px;""></iframe>


!!! note
    You can set `n_clusters` in `visualize_heatmap` to order the topics by their similarity.
    This will result in blocks being formed in the heatmap indicating which clusters of topics are
    similar to each other. This step is very much recommended as it will make reading the heatmap easier.

## **Visualize Topics over Time**
After creating topics over time with Dynamic Topic Modeling, we can visualize these topics by
leveraging the interactive abilities of Plotly. Plotly allows us to show the frequency
of topics over time whilst giving the option of hovering over the points to show the time-specific topic representations.
Simply call `.visualize_topics_over_time` with the newly created topics over time:


```python
import re
import pandas as pd
from bertopic import BERTopic

# Prepare data
trump = pd.read_csv('https://drive.google.com/uc?export=download&id=1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6')
trump.text = trump.apply(lambda row: re.sub(r"http\S+", "", row.text).lower(), 1)
trump.text = trump.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.text.split())), 1)
trump.text = trump.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.text).split()), 1)
trump = trump.loc[(trump.isRetweet == "f") & (trump.text != ""), :]
timestamps = trump.date.to_list()
tweets = trump.text.to_list()

# Create topics over time
model = BERTopic(verbose=True)
topics, probs = model.fit_transform(tweets)
topics_over_time = model.topics_over_time(tweets, timestamps)
```

Then, we visualize some interesting topics:

```python
model.visualize_topics_over_time(topics_over_time, topics=[9, 10, 72, 83, 87, 91])
```
<iframe src="trump.html" style="width:1000px; height: 680px; border: 0px;""></iframe>

## **Visualize Topics per Class**
You might want to extract and visualize the topic representation per class. For example, if you have
specific groups of users that might approach topics differently, then extracting them would help understanding
how these users talk about certain topics. In other words, this is simply creating a topic representation for
certain classes that you might have in your data.

First, we need to train our model:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Prepare data and classes
data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
docs = data["data"]
classes = [data["target_names"][i] for i in data["target"]]

# Create topic model and calculate topics per class
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
topics_per_class = topic_model.topics_per_class(docs, classes=classes)
```

Then, we visualize the topic representation of major topics per class:

```python
topic_model.visualize_topics_per_class(topics_per_class)
```

<iframe src="topics_per_class.html" style="width:1400px; height: 1000px; border: 0px;""></iframe>
