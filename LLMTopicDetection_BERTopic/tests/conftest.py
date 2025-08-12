import copy
import pytest
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from bertopic.vectorizers import OnlineCountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression


@pytest.fixture(scope="session")
def embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


@pytest.fixture(scope="session")
def document_embeddings(documents, embedding_model):
    embeddings = embedding_model.encode(documents)
    return embeddings


@pytest.fixture(scope="session")
def reduced_embeddings(document_embeddings):
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine").fit_transform(
        document_embeddings
    )
    return reduced_embeddings


@pytest.fixture(scope="session")
def documents():
    newsgroup_docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"][:1000]
    return newsgroup_docs


@pytest.fixture(scope="session")
def targets():
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    y = data["target"][:1000]
    return y


@pytest.fixture(scope="session")
def base_topic_model(documents, document_embeddings, embedding_model):
    model = BERTopic(embedding_model=embedding_model, calculate_probabilities=True)
    model.umap_model.random_state = 42
    model.hdbscan_model.min_cluster_size = 3
    model.fit(documents, document_embeddings)
    return model


@pytest.fixture(scope="session")
def zeroshot_topic_model(documents, document_embeddings, embedding_model):
    zeroshot_topic_list = ["religion", "cars", "electronics"]
    model = BERTopic(
        embedding_model=embedding_model,
        calculate_probabilities=True,
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=0.3,
    )
    model.umap_model.random_state = 42
    model.hdbscan_model.min_cluster_size = 2
    model.fit(documents, document_embeddings)
    return model


@pytest.fixture(scope="session")
def custom_topic_model(documents, document_embeddings, embedding_model):
    umap_model = UMAP(n_neighbors=15, n_components=6, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        calculate_probabilities=True,
    ).fit(documents, document_embeddings)
    return model


@pytest.fixture(scope="session")
def representation_topic_model(documents, document_embeddings, embedding_model):
    umap_model = UMAP(n_neighbors=15, n_components=6, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    representation_model = {
        "Main": KeyBERTInspired(),
        "MMR": [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance()],
    }
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        representation_model=representation_model,
        calculate_probabilities=True,
    ).fit(documents, document_embeddings)
    return model


@pytest.fixture(scope="session")
def reduced_topic_model(custom_topic_model, documents):
    model = copy.deepcopy(custom_topic_model)
    model.reduce_topics(documents, nr_topics="auto")
    return model


@pytest.fixture(scope="session")
def merged_topic_model(custom_topic_model, documents):
    model = copy.deepcopy(custom_topic_model)

    # Merge once
    topics_to_merge = [[1, 2], [3, 4]]
    model.merge_topics(documents, topics_to_merge)

    # Merge second time
    topics_to_merge = [[5, 6, 7]]
    model.merge_topics(documents, topics_to_merge)
    return model


@pytest.fixture(scope="session")
def kmeans_pca_topic_model(documents, document_embeddings):
    hdbscan_model = KMeans(n_clusters=15, random_state=42)
    dim_model = PCA(n_components=5)
    model = BERTopic(
        hdbscan_model=hdbscan_model,
        umap_model=dim_model,
        embedding_model=embedding_model,
    ).fit(documents, document_embeddings)
    return model


@pytest.fixture(scope="session")
def supervised_topic_model(documents, document_embeddings, embedding_model, targets):
    empty_dimensionality_model = BaseDimensionalityReduction()
    clf = LogisticRegression()

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=empty_dimensionality_model,
        hdbscan_model=clf,
    ).fit(documents, embeddings=document_embeddings, y=targets)
    return model


@pytest.fixture(scope="session")
def online_topic_model(documents, document_embeddings, embedding_model):
    umap_model = PCA(n_components=5)
    cluster_model = MiniBatchKMeans(n_clusters=50, random_state=0)
    vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=0.01)
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        embedding_model=embedding_model,
    )

    topics = []
    for index in range(0, len(documents), 50):
        model.partial_fit(documents[index : index + 50], document_embeddings[index : index + 50])
        topics.extend(model.topics_)
    model.topics_ = topics
    return model


@pytest.fixture(scope="session")
def cuml_base_topic_model(documents, document_embeddings, embedding_model):
    from cuml.cluster import HDBSCAN as cuml_hdbscan
    from cuml.manifold import UMAP as cuml_umap

    model = BERTopic(
        embedding_model=embedding_model,
        calculate_probabilities=True,
        umap_model=cuml_umap(n_components=5, n_neighbors=5, random_state=42),
        hdbscan_model=cuml_hdbscan(min_cluster_size=3, prediction_data=True),
    )
    model.fit(documents, document_embeddings)
    return model
