import copy
import pytest
import numpy as np
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("base_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
        ("online_topic_model"),
    ],
)
def test_extract_embeddings(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    single_embedding = topic_model._extract_embeddings("a document")
    multiple_embeddings = topic_model._extract_embeddings(["something different", "another document"])
    sim_matrix = cosine_similarity(single_embedding, multiple_embeddings)[0]

    assert single_embedding.shape[0] == 1
    assert single_embedding.shape[1] == 384
    assert np.min(single_embedding) > -5
    assert np.max(single_embedding) < 5

    assert multiple_embeddings.shape[0] == 2
    assert multiple_embeddings.shape[1] == 384
    assert np.min(multiple_embeddings) > -5
    assert np.max(multiple_embeddings) < 5

    assert sim_matrix[0] < 0.5
    assert sim_matrix[1] > 0.5


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("base_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
        ("online_topic_model"),
    ],
)
def test_extract_embeddings_compare(model, embedding_model, request):
    docs = ["some document"]
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    bertopic_embeddings = topic_model._extract_embeddings(docs)

    assert isinstance(bertopic_embeddings, np.ndarray)
    assert bertopic_embeddings.shape == (1, 384)

    sentence_embeddings = embedding_model.encode(docs, show_progress_bar=False)
    assert np.array_equal(bertopic_embeddings, sentence_embeddings)


def test_extract_incorrect_embeddings():
    with pytest.raises(ValueError):
        model = BERTopic(language="Unknown language")
        model.fit(["some document"])
