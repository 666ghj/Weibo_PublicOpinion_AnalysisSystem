# Merge Multiple Fitted Models

After you have trained a new BERTopic model on your data, new data might still be coming in. Although you can use [online BERTopic](https://maartengr.github.io/BERTopic/getting_started/online/online.html), you might prefer to use the default HDBSCAN and UMAP models since they do not support incremental learning out of the box.

Instead, we you can train a new BERTopic on incoming data and merge it with your base model to detect whether new topics have appeared in the unseen documents. This is a great way of detecting whether your new model contains information that was not previously found in your base topic model.

Similarly, you might want to train multiple BERTopic models using different sets of settings, even though they might all be using the same underlying embedding model. Merging these models would also allow for a single model that you can use throughout your use cases.

Lastly, this methods also allows for a degree of `federated learning` where each node trains a topic model that are aggregated in a central server.

## **Example**
To demonstrate merging different topic models with BERTopic, we use the ArXiv paper abstracts to see which topics they generally contain.

First, we train three separate models on different parts of the data:

```python
from umap import UMAP
from bertopic import BERTopic
from datasets import load_dataset

dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]

# Extract abstracts to train on and corresponding titles
abstracts_1 = dataset["abstract"][:5_000]
abstracts_2 = dataset["abstract"][5_000:10_000]
abstracts_3 = dataset["abstract"][10_000:15_000]

# Create topic models
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
topic_model_1 = BERTopic(umap_model=umap_model, min_topic_size=20).fit(abstracts_1)
topic_model_2 = BERTopic(umap_model=umap_model, min_topic_size=20).fit(abstracts_2)
topic_model_3 = BERTopic(umap_model=umap_model, min_topic_size=20).fit(abstracts_3)
```

Then, we can combine all three models into one with `.merge_models`:

```python
# Combine all models into one
merged_model = BERTopic.merge_models([topic_model_1, topic_model_2, topic_model_3])
```

When we inspect the first model, we can see it has 52 topics:

```python
>>> len(topic_model_1.get_topic_info())
52
```

Now, we inspect the merged model, we can see it has 57 topics:

```python
>>> len(merged_model.get_topic_info())
57
```

It seems that by merging these three models, there were 6 undiscovered topics that we could add to the very first model.

!!! Note
    Note that the models are merged sequentially. This means that the comparison starts with `topic_model_1` and that
    each new topic from `topic_model_2` and `topic_model_3` will be added to `topic_model_1`.

We can check the newly added topics in the `merged_model` by simply looking at the 6 latest topics that were added. The order of topics from `topic_model_1`
remains the same. All new topics are simply added on top of them.

Let's inspect them:

```python
>>> merged_model.get_topic_info().tail(5)
```

|    |   Topic |   Count | Name                                   | Representation                                                                                                         |   Representative_Docs |
|---:|--------:|--------:|:---------------------------------------|:-----------------------------------------------------------------------------------------------------------------------|----------------------:|
| 52 |      51 |      47 | 50_activity_mobile_wearable_sensors    | ['activity', 'mobile', 'wearable', 'sensors', 'falls', 'human', 'phone', 'recognition', 'activities', 'accelerometer'] |                   nan |
| 53 |      52 |      48 | 25_music_musical_audio_chord           | ['music', 'musical', 'audio', 'chord', 'and', 'we', 'to', 'that', 'of', 'for']                                         |                   nan |
| 54 |      53 |      32 | 36_fairness_discrimination_fair_groups | ['fairness', 'discrimination', 'fair', 'groups', 'protected', 'decision', 'we', 'of', 'classifier', 'to']              |                   nan |
| 55 |      54 |      30 | 38_traffic_driver_prediction_flow      | ['traffic', 'driver', 'prediction', 'flow', 'trajectory', 'the', 'and', 'congestion', 'of', 'transportation']          |                   nan |
| 56 |      55 |      22 | 50_spiking_neurons_networks_learning   | ['spiking', 'neurons', 'networks', 'learning', 'neural', 'snn', 'dynamics', 'plasticity', 'snns', 'of']                |                   nan |


It seems that topics about activity, music, fairness, traffic, and spiking networks were added to the base topic model! Two things that you might have noticed. First,
the representative documents were not added to the model. This is because of privacy reasons, you might want to combine models that were trained on different stations which
would allow for a degree of `federated learning`. Second, the names of the new topics contain topic ids that refer to one of the old models. They were purposefully left this way
so that the user can identify which topics were newly added which you could inspect in the original models.


## **min_similarity**

The way the models are merged is through comparison of their topic embeddings. If topics between models are similar enough, then they will be regarded as the same topics
and the topic of the first model in the list will be chosen. However, if topics between models are dissimilar enough, then the topic of the latter model will be added to the former.

This (dis)similarity is can be tweaked using the `min_similarity` parameter. Increasing this value will increase the chance of adding new topics. In contrast, decreasing this value
will make it more strict and threfore decrease the chance of adding new topics. The value is set to `0.7` by default, so let's see what happens if we were to increase this value to
`0.9``:

```python
# Combine all models into one
merged_model = BERTopic.merge_models([topic_model_1, topic_model_2, topic_model_3], min_similarity=0.9)
```

When we inspect the number of topics in our new model, we can see that they have increased quite a bit:

```python
>>> len(merged_model.get_topic_info())
102
```

This demonstrates the influence of `min_similarity` on the number of new topics that are added to the base model.
