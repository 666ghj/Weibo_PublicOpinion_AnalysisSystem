After having created a BERTopic model, you might end up with over a hundred topics. Searching through those
can be quite cumbersome especially if you are searching for a specific topic. Fortunately, BERTopic allows you
to search for topics using search terms. First, let's create and train a BERTopic model:


```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Create topics
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
```

After having trained our model, we can use `find_topics` to search for topics that are similar
to an input search_term. Here, we are going to be searching for topics that closely relate the
search term "motor". Then, we extract the most similar topic and check the results:

```python
>>> similar_topics, similarity = topic_model.find_topics("motor", top_n=5)
>>> topic_model.get_topic(similar_topics[0])
[('bike', 0.02275997701645559),
 ('motorcycle', 0.011391202866080292),
 ('bikes', 0.00981187573649205),
 ('dod', 0.009614623748226669),
 ('honda', 0.008247663662558535),
 ('ride', 0.0064683227888861945),
 ('harley', 0.006355502638631013),
 ('riding', 0.005766601561614182),
 ('motorcycles', 0.005596372493714447),
 ('advice', 0.005534544418830091)]
```

It definitely seems that a topic was found that closely matches "motor". The topic seems to be motorcycle
related and therefore matches our "motor" input. You can use the `similarity` variable to see how similar
the extracted topics are to the search term.

!!! note
    You can only use this method if an embedding model was supplied to BERTopic using `embedding_model`.
