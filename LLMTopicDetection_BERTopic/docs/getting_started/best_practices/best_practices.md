[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2?usp=sharing) - Overview of Best Practices

Through the nature of BERTopic, its modularity, many variations of the topic modeling technique is possible. However, during the development and through the usage of the package, a set of best practices have been developed that generally lead to great results.

The following are a number of steps, parameters, and settings that you can use that will generally improve the quality of the resulting topics. In other words, after going through the quick start and getting a feeling for the API these steps should get you to the next level of performance.

!!! Note
    Although these are called *best practices*, it does not necessarily mean that they work across all use cases perfectly. The underlying modular nature of BERTopic is meant to take different use cases into account. After going through these practices it is advised to fine-tune wherever necessary.


To showcase how these "best practices" work, we will go through an example dataset and apply all practices to it.

## **Data**

For this example, we will use a dataset containing abstracts and metadata from [ArXiv articles](https://huggingface.co/datasets/arxiv_dataset).


```python
from datasets import load_dataset

dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]

# Extract abstracts to train on and corresponding titles
abstracts = dataset["abstract"]
titles = dataset["title"]
```

!!! Tip "Sentence Splitter"

    Whenever you have large documents, you typically want to split them up into either paragraphs or sentences. A nice way to do so is by using NLTK's sentence splitter which is nothing more than:

    ```python
    from nltk.tokenize import sent_tokenize, word_tokenize
    sentences = [sent_tokenize(abstract) for abstract in abstracts]
    sentences = [sentence for doc in sentences for sentence in doc]
    ```

## **Pre-calculate Embeddings**
After having created our data, namely `abstracts`, we can dive into the very first best practice, **pre-calculating embeddings**.

BERTopic works by converting documents into numerical values, called embeddings. This process can be very costly, especially if we want to iterate over parameters. Instead, we can calculate those embeddings once and feed them to BERTopic to skip calculating embeddings each time.

```python
from sentence_transformers import SentenceTransformer

# Pre-calculate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
```

!!! Tip
    New embedding models are released frequently and their performance keeps getting better. To keep track of the best embedding models out there, you can visit the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). It is an excellent place for selecting the embedding that works best for you. For example, if you want the best of the best, then the top 5 models might the place to look.


## **Preventing Stochastic Behavior**
In BERTopic, we generally use a dimensionality reduction algorithm to reduce the size of the embeddings. This is done to prevent the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) to a certain degree.

As a default, this is done with [UMAP](https://github.com/lmcinnes/umap) which is an incredible algorithm for reducing dimensional space. However, by default, it shows stochastic behavior which creates different results each time you run it. To prevent that, we will need to set a `random_state` of the model before passing it to BERTopic.

As a result, we can now fully reproduce the results each time we run the model.

```python
from umap import UMAP

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
```

## **Controlling Number of Topics**
There is a parameter to control the number of topics, namely `nr_topics`. This parameter, however, merges topics **after** they have been created. It is a parameter that supports creating a fixed number of topics.

However, it is advised to control the number of topics through the cluster model which is by default HDBSCAN. HDBSCAN has a parameter, namely `min_cluster_size` that indirectly controls the number of topics that will be created.

A higher `min_cluster_size` will generate fewer topics and a lower `min_cluster_size` will generate more topics.

Here, we will go with `min_cluster_size=150` to prevent too many micro-clusters from being created:

```python
from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
```

## **Improving Default Representation**
The default representation of topics is calculated through [c-TF-IDF](https://maartengr.github.io/BERTopic/algorithm/algorithm.html#5-topic-representation). However, c-TF-IDF is powered by the [CountVectorizer](https://maartengr.github.io/BERTopic/getting_started/vectorizers/vectorizers.html) which converts text into tokens. Using the CountVectorizer, we can do a number of things:

* Remove stopwords
* Ignore infrequent words
* Increase the n-gram range

In other words, we can preprocess the topic representations **after** documents are assigned to topics. This will not influence the clustering process in any way.

Here, we will ignore English stopwords and infrequent words. Moreover, by increasing the n-gram range we will consider topic representations that are made up of one or two words.

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
```

## **Additional Representations**
Previously, we have tuned the default representation but there are quite a number of [other topic representations](https://maartengr.github.io/BERTopic/getting_started/representation/representation.html) in BERTopic that we can choose from. From [KeyBERTInspired](https://maartengr.github.io/BERTopic/getting_started/representation/representation.html#keybertinspired) and [PartOfSpeech](https://maartengr.github.io/BERTopic/getting_started/representation/representation.html#partofspeech), to [OpenAI's ChatGPT](https://maartengr.github.io/BERTopic/getting_started/representation/llm.html#chatgpt) and [open-source](https://maartengr.github.io/BERTopic/getting_started/representation/llm.html#langchain) alternatives, many representations are possible.

In BERTopic, you can model many different topic representations simultaneously to test them out and get different perspectives of topic descriptions. This is called [multi-aspect](https://maartengr.github.io/BERTopic/getting_started/multiaspect/multiaspect.html) topic modeling.

Here, we will demonstrate a number of interesting and useful representations in BERTopic:

* KeyBERTInspired
  * A method that derives inspiration from how KeyBERT works
* PartOfSpeech
  * Using SpaCy's POS tagging to extract words
* MaximalMarginalRelevance
  * Diversify the topic words
* OpenAI
  * Use ChatGPT to label our topics


```python
import openai
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech

# KeyBERT
keybert_model = KeyBERTInspired()

# Part-of-Speech
pos_model = PartOfSpeech("en_core_web_sm")

# MMR
mmr_model = MaximalMarginalRelevance(diversity=0.3)

# GPT-3.5
client = openai.OpenAI(api_key="sk-...")
prompt = """
I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
topic: <topic label>
"""
openai_model = OpenAI(client, model="gpt-4o-mini", exponential_backoff=True, prompt=prompt)

# All representation models
representation_model = {
    "KeyBERT": keybert_model,
    # "OpenAI": openai_model,  # Uncomment if you will use OpenAI
    "MMR": mmr_model,
    "POS": pos_model
}
```

## **Training**
Now that we have a set of best practices, we can use them in our training loop. Here, several different representations, keywords and labels for our topics will be created. If you want to iterate over the topic model it is advised to use the pre-calculated embeddings as that significantly speeds up training.

```python
from bertopic import BERTopic

topic_model = BERTopic(

  # Pipeline models
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  representation_model=representation_model,

  # Hyperparameters
  top_n_words=10,
  verbose=True
)

# Train model
topics, probs = topic_model.fit_transform(abstracts, embeddings)

# Show topics
topic_model.get_topic_info()
```

To get all representations for a single topic, we simply run the following:

```python
>>> topic_model.get_topic(1, full=True)
{'Main': [('adversarial', 0.028838938990764302),
  ('attacks', 0.021726302042463556),
  ('attack', 0.016803574415028524),
  ('robustness', 0.013046135743326167),
  ('adversarial examples', 0.01151254557995679),
  ('examples', 0.009920962487998853),
  ('perturbations', 0.009053305826870773),
  ('adversarial attacks', 0.008747627064844006),
  ('malware', 0.007675131707700338),
  ('defense', 0.007365955840313783)],
 'KeyBERT': [('adversarial training', 0.76427937),
  ('adversarial attack', 0.74271905),
  ('vulnerable adversarial', 0.73302543),
  ('adversarial', 0.7311052),
  ('adversarial examples', 0.7179245),
  ('adversarial attacks', 0.7082),
  ('adversarially', 0.7005141),
  ('adversarial robustness', 0.69911957),
  ('adversarial perturbations', 0.6588783),
  ('adversary', 0.4467769)],
 'OpenAI': [('Adversarial attacks and defense', 1)],
 'MMR': [('adversarial', 0.028838938990764302),
  ('attacks', 0.021726302042463556),
  ('attack', 0.016803574415028524),
  ('robustness', 0.013046135743326167),
  ('adversarial examples', 0.01151254557995679),
  ('examples', 0.009920962487998853),
  ('perturbations', 0.009053305826870773),
  ('adversarial attacks', 0.008747627064844006),
  ('malware', 0.007675131707700338),
  ('defense', 0.007365955840313783)],
 'POS': [('adversarial', 0.028838938990764302),
  ('attacks', 0.021726302042463556),
  ('attack', 0.016803574415028524),
  ('robustness', 0.013046135743326167),
  ('adversarial examples', 0.01151254557995679),
  ('examples', 0.009920962487998853),
  ('perturbations', 0.009053305826870773),
  ('adversarial attacks', 0.008747627064844006),
  ('malware', 0.007675131707700338),
  ('defense', 0.007365955840313783)]}
```

**NOTE**: The labels generated by OpenAI's **ChatGPT** are especially interesting to use throughout your model. Below, we will go into more detail how to set that as a custom label.

!!! Tip "Parameters"
    If you would like to return the topic-document probability matrix, then it is advised to use `calculate_probabilities=True`. Do note that this can significantly slow down training. To speed it up, use [cuML's HDBSCAN](https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html#cuml-hdbscan) instead. You could also approximate the topic-document probability matrix with `.approximate_distribution` which will be discussed later.


## **(Custom) Labels**
The default label of each topic are the top 3 words in each topic combined with an underscore between them.

This, of course, might not be the best label that you can think of for a certain topic. Instead, we can use `.set_topic_labels` to manually label all or certain topics.

We can also use `.set_topic_labels` to use one of the other topic representations that we had before, like `KeyBERTInspired` or even `OpenAI`.

```python
# Label the topics yourself
topic_model.set_topic_labels({1: "Space Travel", 7: "Religion"})

# or use one of the other topic representations, like KeyBERTInspired
keybert_topic_labels = {topic: " | ".join(list(zip(*values))[0][:3]) for topic, values in topic_model.topic_aspects_["KeyBERT"].items()}
topic_model.set_topic_labels(keybert_topic_labels)

# or ChatGPT's labels
chatgpt_topic_labels = {topic: " | ".join(list(zip(*values))[0]) for topic, values in topic_model.topic_aspects_["OpenAI"].items()}
chatgpt_topic_labels[-1] = "Outlier Topic"
topic_model.set_topic_labels(chatgpt_topic_labels)
```

Now that we have set the updated topic labels, we can access them with the many functions used throughout BERTopic. Most notably, you can show the updated labels in visualizations with the `custom_labels=True` parameters.

If we were to run `topic_model.get_topic_info()` it will now include the column `CustomName`. That is the custom label that we just created for each topic.

## **Topic-Document Distribution**
If using `calculate_probabilities=True` is not possible, then you can [approximate the topic-document distributions](https://maartengr.github.io/BERTopic/getting_started/distribution/distribution.html) using `.approximate_distribution`. It is a fast and flexible method for creating different topic-document distributions.

```python
# `topic_distr` contains the distribution of topics in each document
topic_distr, _ = topic_model.approximate_distribution(abstracts, window=8, stride=4)
```

Next, lets take a look at a specific abstract and see how the topic distribution was extracted:

```python
# Visualize the topic-document distribution for a single document
topic_model.visualize_distribution(topic_distr[abstract_id], custom_labels=True)
```

It seems to have extracted a number of topics that are relevant and shows the distributions of these topics across the abstract. We can go one step further and visualize them on a token-level:

```python
# Calculate the topic distributions on a token-level
topic_distr, topic_token_distr = topic_model.approximate_distribution(abstracts[abstract_id], calculate_tokens=True)

# Visualize the token-level distributions
df = topic_model.visualize_approximate_distribution(abstracts[abstract_id], topic_token_distr[0])
df
```

!!! Tip "use_embedding_model"
    As a default, we compare the c-TF-IDF calculations between the token sets and all topics. Due to its bag-of-word representation, this is quite fast. However, you might want to use the selected embedding_model instead to do this comparison. Do note that due to the many token sets, it is often computationally quite a bit slower:

    ```python
    topic_distr, _ = topic_model.approximate_distribution(docs, use_embedding_model=True)
    ```
## **Outlier Reduction**
By default, HDBSCAN generates outliers which is a helpful mechanic in creating accurate topic representations. However, you might want to assign every single document to a topic. We can use `.reduce_outliers` to map some or all outliers to a topic:

```python
# Reduce outliers
new_topics = topic_model.reduce_outliers(abstracts, topics)

# Reduce outliers with pre-calculate embeddings instead
new_topics = topic_model.reduce_outliers(abstracts, topics, strategy="embeddings", embeddings=embeddings)
```

!!! Note "Update Topics with Outlier Reduction"
    After having generated updated topic assignments, we can pass them to BERTopic in order to update the topic representations:

    ```python
    topic_model.update_topics(docs, topics=new_topics)
    ```

    It is important to realize that updating the topics this way may lead to errors if topic reduction or topic merging techniques are used afterwards. The reason for this is that when you assign a -1 document to topic 1 and another -1 document to topic 2, it is unclear how you map the -1 documents. Is it matched to topic 1 or 2.


## **Visualize Topics**

With visualizations, we are closing into the realm of subjective "best practices". These are things that I generally do because I like the representations but your experience might differ.

Having said that, there are two visualizations that are my go-to when visualizing the topics themselves:

* `topic_model.visualize_topics()`
* `topic_model.visualize_hierarchy()`

```python
# Visualize topics with custom labels
topic_model.visualize_topics(custom_labels=True)

# Visualize hierarchy with custom labels
topic_model.visualize_hierarchy(custom_labels=True)
```

## **Visualize Documents**

When visualizing documents, it helps to have embedded the documents beforehand to speed up computation. Fortunately, we have already done that as a "best practice".

Visualizing documents in 2-dimensional space helps in understanding the underlying structure of the documents and topics.

```python
# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
```

The following plot is **interactive** which means that you can zoom in, double click on a label to only see that one and generally interact with the plot:

```python
# Visualize the documents in 2-dimensional space and show the titles on hover instead of the abstracts
# NOTE: You can hide the hover with `hide_document_hover=True` which is especially helpful if you have a large dataset
# NOTE: You can also hide the annotations with `hide_annotations=True` which is helpful to see the larger structure
topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, custom_labels=True)
```

!!! Note "2-dimensional space"
    Although visualizing the documents in 2-dimensional gives an idea of their underlying structure, there is a risk involved.

    Visualizing the documents in 2-dimensional space means that we have lost significant information since the original embeddings were more than 384 dimensions. Condensing all that information in 2 dimensions is simply not possible. In other words, it is merely an **approximation**, albeit quite an accurate one.

## **Serialization**

When saving a BERTopic model, there are several ways in doing so. You can either save the entire model with `pickle`, `pytorch`, or `safetensors`.

Personally, I would advise going with `safetensors` whenever possible. The reason for this is that the format allows for a very small topic model to be saved and shared.

When saving a model with `safetensors`, it skips over saving the dimensionality reduction and clustering models. The `.transform` function will still work without these models but instead assign topics based on the similarity between document embeddings and the topic embeddings.

As a result, the `.transform` step might give different results but it is generally worth it considering the smaller and significantly faster model.

```python
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
topic_model.save("my_model_dir", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
```

!!! Note "Embedding Model"
    Using `safetensors`, we are not saving the underlying embedding model but merely a pointer to the model. For example, in the above example we are saving the string `"sentence-transformers/all-MiniLM-L6-v2"` so that we can load in the embedding model alongside the topic model.

    This currently only works if you are using a sentence transformer model. If you are using a different model, you can load it in when loading the topic model like this:

    ```python
    from sentence_transformers import SentenceTransformer

    # Define embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load model and add embedding model
    loaded_model = BERTopic.load("my_model_dir", embedding_model=embedding_model)
    ```

## **Inference**

To speed up the inference, we can leverage a "best practice" that we used before, namely serialization. When you save a model as `safetensors` and then load it in, we are removing the dimensionality reduction and clustering steps from the pipeline.

Instead, the assignment of topics is done through cosine similarity of document embeddings and topic embeddings. This speeds up inferences significantly.

To show its effect, let's start by disabling the logger:

```python
from bertopic._utils import MyLogger
logger = MyLogger()
logger.configure("ERROR")
loaded_model.verbose = False
topic_model.verbose = False
```

Then, we run inference on both the loaded model and the non-loaded model:

```python
>>> %timeit loaded_model.transform(abstracts[:100])
343 ms ± 31.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %timeit topic_model.transform(abstracts[:100])
1.37 s ± 166 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Based on the above, the `loaded_model` seems to be quite a bit faster for inference than the original `topic_model`.
