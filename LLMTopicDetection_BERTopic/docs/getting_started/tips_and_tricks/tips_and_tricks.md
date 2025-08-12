# Tips & Tricks


## **Document length**
As a default, we are using sentence-transformers to embed our documents. However, as the name implies, the embedding model works best for either sentences or paragraphs. This means that whenever you have a set of documents, where each documents contains several paragraphs, the document is truncated and the topic model is only trained on a small part of the data.

One way to solve this issue is by splitting up longer documents into either sentences or paragraphs before embedding them. Another solution is to approximate the [topic distributions](https://maartengr.github.io/BERTopic/getting_started/distribution/distribution.html) of topics after having trained your topic model.


## **Removing stop words**
At times, stop words might end up in our topic representations. This is something we typically want to avoid as they contribute little to the interpretation of the topics. However, removing stop words as a preprocessing step is not advised as the transformer-based embedding models that we use need the full context in order to create accurate embeddings.

Instead, we can use the `CountVectorizer` to preprocess our documents **after** having generated embeddings and clustered
our documents. Personally, I have found almost no disadvantages to using the `CountVectorizer` to remove stopwords and
it is something I would strongly advise to try out:

```python
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words="english")
topic_model = BERTopic(vectorizer_model=vectorizer_model)
```

We can also use the `ClassTfidfTransformer` to reduce the impact of frequent words. The end result is very similar to explicitly removing stopwords but this process does this automatically:

```python
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
topic_model = BERTopic(ctfidf_model=ctfidf_model)
```

Lastly, we can use a KeyBERT-Inspired model to reduce the appearance of stop words. This also often improves the topic representation:

```python
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic

# Create your representation model
representation_model = KeyBERTInspired()

# Use the representation model in BERTopic on top of the default pipeline
topic_model = BERTopic(representation_model=representation_model)
```

## **Diversify topic representation**
After having calculated our top *n* words per topic there might be many words that essentially
mean the same thing. As a little bonus, we can use `bertopic.representation.MaximalMarginalRelevance` in BERTopic to
diversify words in each topic such that we limit the number of duplicate words we find in each topic.
This is done using an algorithm called Maximal Marginal Relevance which compares word embeddings
with the topic embedding.

We do this by specifying a value between 0 and 1, with 0 being not at all diverse and 1 being completely diverse:

```python
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance

representation_model = MaximalMarginalRelevance(diversity=0.2)
topic_model = BERTopic(representation_model=representation_model)
```

Since MMR is using word embeddings to diversify the topic representations, it is necessary to pass the embedding model to BERTopic if you are using pre-computed embeddings:

```python
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=False)
representation_model = MaximalMarginalRelevance(diversity=0.2)
topic_model = BERTopic(embedding_model=sentence_model, representation_model=representation_model)
```


## **Topic-term matrix**
Although BERTopic focuses on clustering our documents, the end result does contain a topic-term matrix.
This topic-term matrix is calculated using c-TF-IDF, a TF-IDF procedure optimized for class-based analyses.

To extract the topic-term matrix (or c-TF-IDF matrix) with the corresponding words, we can simply do the following:

```python
topic_term_matrix = topic_model.c_tf_idf_
words = topic_model.vectorizer_model.get_feature_names()
```


## **Pre-compute embeddings**
Typically, we want to iterate fast over different versions of our BERTopic model whilst we are trying to optimize it to a specific use case. To speed up this process, we can pre-compute the embeddings, save them,
and pass them to BERTopic so it does not need to calculate the embeddings each time:

```python
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

# Prepare embeddings
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=False)

# Train our topic model using our pre-trained sentence-transformers embeddings
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs, embeddings)
```

## **Speed up UMAP**
At times, UMAP may take a while to fit on the embeddings that you have. This often happens when you have
the embeddings millions of documents that you want to reduce in dimensionality. There is a trick that
can speed up this process somewhat: Initializing UMAP with rescaled PCA embeddings.

Without going in too much detail (look [here](https://github.com/lmcinnes/umap/issues/771#issuecomment-931886015) for more information), you can reduce the embeddings using PCA
and use that as a starting point. This can speed up the dimensionality reduction a bit:


```python
import numpy as np
from umap import UMAP
from bertopic import BERTopic
from sklearn.decomposition import PCA


def rescale(x, inplace=False):
    """ Rescale an embedding so optimization will not have convergence issues.
    """
    if not inplace:
        x = np.array(x, copy=True)

    x /= np.std(x[:, 0]) * 10000

    return x


# Initialize and rescale PCA embeddings
pca_embeddings = rescale(PCA(n_components=5).fit_transform(embeddings))

# Start UMAP from PCA embeddings
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    init=pca_embeddings,
)

# Pass the model to BERTopic:
topic_model = BERTopic(umap_model=umap_model)
```

## **GPU acceleration**

You can use [cuML](https://rapids.ai/start.html#rapids-release-selector) to speed up both
UMAP and HDBSCAN through GPU acceleration:

```python
from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP

# Create instances of GPU-accelerated UMAP and HDBSCAN
umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)

# Pass the above models to be used in BERTopic
topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)
topics, probs = topic_model.fit_transform(docs)
```

Depending on the embeddings you are using, you might want to normalize them first in order to
force a cosine-related distance metric in UMAP:

```python
from cuml.preprocessing import normalize
embeddings = normalize(embeddings)
```

!!! note
    As of the v0.13 release, it is not yet possible to calculate the topic-document probability matrix for unseen data (i.e., `.transform`) using cuML's HDBSCAN.
    However, it is still possible to calculate the topic-document probability matrix for the data on which the model was trained (i.e., `.fit` and `.fit_transform`).

!!! note
    If you want to install cuML together with BERTopic using Google Colab, you can run the following code:

    ```bash
    !pip install bertopic
    !pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com
    !pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
    !pip install cugraph-cu11 --extra-index-url=https://pypi.nvidia.com
    !pip install --upgrade cupy-cuda11x -f https://pip.cupy.dev/aarch64
    ```


## **Lightweight installation**

The default embedding model in BERTopic is one of the amazing sentence-transformers models, namely `"all-MiniLM-L6-v2"`. Although this model performs well out of the box, it typically needs a GPU to transform the documents into embeddings in a reasonable time. Moreover, the installation requires `pytorch` which often results in a rather large environment, memory-wise.

Fortunately, it is possible to install BERTopic without `sentence-transformers`, `UMAP`, and/or `HDBSCAN`. This can be to reduce your docker images for inference or when you do not use `pytorch` but for instance [Model2Vec](https://github.com/MinishLab/model2vec) instead. The installation can be done as follows:

```bash
pip install --no-deps bertopic
pip install --upgrade numpy pandas scikit-learn tqdm plotly pyyaml
```

This installs a bare-bones version of BERTopic. If you want to use UMAP and Model2Vec for instance, you'll need to first install them:

`pip install model2vec umap-learn`

Then, you can BERTopic without needing to have a GPU:

```python
from bertopic import BERTopic
from model2vec import StaticModel

# Model2Vec
embedding_model = StaticModel.from_pretrained("minishlab/potion-base-8M")

# BERTopic
topic_model = BERTopic(embedding_model=embedding_model)
```

As a result, the entire package and resulting model can be run quickly on the CPU and no GPU is necessary!

!!! Note
    If you have an alternative embedding model, you can use that instead of Model2Vec. Likewise, if you have a different method for dimensionality reduction that you want to use, you can use that instead of UMAP.


## **WordCloud**
To minimize the number of dependencies in BERTopic, it is not possible to generate wordclouds out-of-the-box. However,
there is a minimal script that you can use to generate wordclouds in BERTopic. First, you will need to install
the [wordcloud](https://github.com/amueller/word_cloud) package with `pip install wordcloud`. Then, run the following code
to generate the wordcloud for a specific topic:

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_wordcloud(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Show wordcloud
create_wordcloud(topic_model, topic=1)
```

![](wordcloud.jpg)


!!! tip Tip
    To increase the number of words shown in the wordcloud, you can increase the `top_n_words`
    parameter when instantiating BERTopic. You can also increase the number of words in a topic
    after training the model using `.update_topics()`.


## **Finding similar topics between models**

Whenever you have trained separate BERTopic models on different datasets, it might
be worthful to find the similarities among these models. Is there overlap between
topics in model A and topic in model B? In other words, can we find topics in model A that are similar to those in model B?

We can compare the topic representations of several models in two ways. First, by comparing the topic embeddings that are created when using the same embedding model across both fitted BERTopic instances. Second, we can compare the c-TF-IDF representations instead assuming we have fixed the vocabulary in both instances.

This example will go into the former, using the same embedding model across two BERTopic instances. To do this comparison, let's first create an example where I trained two models, one on an English dataset and one on a Dutch dataset:

```python
from datasets import load_dataset
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP

# The same embedding model needs to be used for both topic models
# and since we are dealing with multiple languages, the model needs to be multi-lingual
sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# To make this example reproducible
umap_model = UMAP(n_neighbors=15, n_components=5,
                  min_dist=0.0, metric='cosine', random_state=42)

# English
en_dataset = load_dataset("stsb_multi_mt", name="en", split="train").to_pandas().sentence1.tolist()
en_model = BERTopic(embedding_model=sentence_model, umap_model=umap_model)
en_model.fit(en_dataset)

# Dutch
nl_dataset = load_dataset("stsb_multi_mt", name="nl", split="train").to_pandas().sentence1.tolist()
nl_model = BERTopic(embedding_model=sentence_model, umap_model=umap_model)
nl_model.fit(nl_dataset)
```

In the code above, there is one important thing to note and that is the `sentence_model`. This model needs to be exactly the same in all BERTopic models, otherwise, it is not possible to compare topic models.

Next, we can calculate the similarity between topics in the English topic model `en_model` and the Dutch model `nl_model`. To do so, we can simply calculate the cosine similarity between the `topic_embedding` of both models:

```python
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(en_model.topic_embeddings_, nl_model.topic_embeddings_)
```

Now that we know which topics are similar to each other, we can extract the most similar topics. Let's say that we have topic 10 in the `en_model` which represents a topic related to trains:

```python
>>> topic = 10
>>> en_model.get_topic(topic)
[('train', 0.2588080580844999),
 ('tracks', 0.1392140438801078),
 ('station', 0.12126454635946024),
 ('passenger', 0.058057876475695866),
 ('engine', 0.05123717127783682),
 ('railroad', 0.048142847325312044),
 ('waiting', 0.04098973702226946),
 ('track', 0.03978248702913929),
 ('subway', 0.03834661195748458),
 ('steam', 0.03834661195748458)]
```

To find the matching topic, we extract the most similar topic in the `sim_matrix`:

```python
>>> most_similar_topic = np.argmax(sim_matrix[topic + 1])-1
>>> nl_model.get_topic(most_similar_topic)
[('trein', 0.24186603209316418),
 ('spoor', 0.1338118418551581),
 ('sporen', 0.07683661859111401),
 ('station', 0.056990389779394225),
 ('stoommachine', 0.04905829711711234),
 ('zilveren', 0.04083879598477808),
 ('treinen', 0.03534099197032758),
 ('treinsporen', 0.03534099197032758),
 ('staat', 0.03481332997324445),
 ('zwarte', 0.03179591746822408)]
```

It seems to be working as, for example, `trein` is a translation of `train` and `sporen` a translation of `tracks`! You can do this for every single topic to find out which topic in the `en_model` might belong to a model in the `nl_model`.

## **Multimodal data**
[Concept](https://github.com/MaartenGr/Concept) is a variation
of BERTopic for multimodal data, such as images with captions. Although we can use that
package for multimodal data, we can perform a small trick with BERTopic to have a similar feature.

BERTopic is a relatively modular approach that attempts to isolate steps from one another. This means,
for example, that you can use k-Means instead of HDBSCAN or PCA instead of UMAP as it does not make
any assumptions with respect to the nature of the clustering.

Similarly, you can pass pre-calculated embeddings to BERTopic that represent the documents that you have.
However, it does not make any assumption with respect to the relationship between those embeddings and
the documents. This means that we could pass any metadata to BERTopic to cluster on instead of document
embeddings. In this example, we can separate our embeddings from our documents so that the embeddings
are generated from images instead of their corresponding images. Thus, we will cluster image embeddings but
create the topic representation from the related captions.

In this example, we first need to fetch our data, namely the Flickr 8k dataset that contains images
with captions:

```python
import os
import glob
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# Flickr 8k images
img_folder = 'photos/'
caps_folder = 'captions/'
if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
    os.makedirs(img_folder, exist_ok=True)

    if not os.path.exists('Flickr8k_Dataset.zip'):   #Download dataset if does not exist
        util.http_get('https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip', 'Flickr8k_Dataset.zip')
        util.http_get('https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip', 'Flickr8k_text.zip')

    for folder, file in [(img_folder, 'Flickr8k_Dataset.zip'), (caps_folder, 'Flickr8k_text.zip')]:
        with zipfile.ZipFile(file, 'r') as zf:
            for member in tqdm(zf.infolist(), desc='Extracting'):
                zf.extract(member, folder)
images = list(glob.glob('photos/Flicker8k_Dataset/*.jpg'))

# Prepare dataframe
captions = pd.read_csv("captions/Flickr8k.lemma.token.txt",sep='\t',names=["img_id","img_caption"])
captions.img_id = captions.apply(lambda row: "photos/Flicker8k_Dataset/" + row.img_id.split(".jpg")[0] + ".jpg", 1)
captions = captions.groupby(["img_id"])["img_caption"].apply(','.join).reset_index()
captions = pd.merge(captions, pd.Series(images, name="img_id"), on="img_id")

# Extract images together with their documents/captions
images = captions.img_id.to_list()
docs = captions.img_caption.to_list()
```

Now that we have our images and captions, we need to generate our image embeddings:

```python
model = SentenceTransformer('clip-ViT-B-32')

# Prepare images
batch_size = 32
nr_iterations = int(np.ceil(len(images) / batch_size))

# Embed images per batch
embeddings = []
for i in tqdm(range(nr_iterations)):
    start_index = i * batch_size
    end_index = (i * batch_size) + batch_size

    images_to_embed = [Image.open(filepath) for filepath in images[start_index:end_index]]
    img_emb = model.encode(images_to_embed, show_progress_bar=False)
    embeddings.extend(img_emb.tolist())

    # Close images
    for image in images_to_embed:
        image.close()
embeddings = np.array(embeddings)
```

Finally, we can fit BERTopic the way we are used to, with documents and embeddings:

```python
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words="english")
topic_model = BERTopic(vectorizer_model=vectorizer_model)
topics, probs = topic_model.fit_transform(docs, embeddings)
captions["Topic"] = topics
```

After fitting our model, let's inspect a topic about skateboarders:

```python
>>> topic_model.get_topic(2)
[('skateboard', 0.09592033177340711),
 ('skateboarder', 0.07792520092546491),
 ('trick', 0.07481578896400298),
 ('ramp', 0.056952605147927216),
 ('skate', 0.03745127816149923),
 ('perform', 0.036546213623432654),
 ('bicycle', 0.03453483070441857),
 ('bike', 0.033233021253898994),
 ('jump', 0.026709362981948037),
 ('air', 0.025422798170830936)]
```

Based on the above output, we can take an image to see if the representation makes sense:

```python
image = captions.loc[captions.Topic == 2, "img_id"].values.tolist()[0]
Image.open(image)
```

![](skateboarders.jpg)

## **KeyBERT** & **BERTopic**

Although BERTopic focuses on topic extraction methods that does not assume specific structures for the generated clusters, it is possible to do this on a more local level. More specifically, we can use KeyBERT to generate a number of keywords for each document and then build a vocabulary on top of that as the input for BERTopic. This way, we can select words that we know have meaning to a topic, without focusing on the centroid of that cluster. This also allows more frequent words to pop-up regardless of the structure and density of a cluster.

To do this, we first need to run [KeyBERT](https://github.com/MaartenGr/KeyBERT) on our data and create our vocabulary:

```python
from sklearn.datasets import fetch_20newsgroups
from keybert import KeyBERT

# Prepare documents
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

# Extract keywords
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(docs)

# Create our vocabulary
vocabulary = [k[0] for keyword in keywords for k in keyword]
vocabulary = list(set(vocabulary))
```

Then, we pass our `vocabulary` to BERTopic and train the model:

```python
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model= CountVectorizer(vocabulary=vocabulary)
topic_model = BERTopic(vectorizer_model=vectorizer_model)
topics, probs = topic_model.fit_transform(docs)
```
