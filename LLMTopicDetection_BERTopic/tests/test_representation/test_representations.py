import copy
import pytest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("base_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
    ],
)
def test_update_topics(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    old_ctfidf = topic_model.c_tf_idf_
    old_topics = topic_model.topics_

    topic_model.update_topics(documents, n_gram_range=(1, 3))

    assert old_ctfidf.shape[1] < topic_model.c_tf_idf_.shape[1]
    assert old_topics == topic_model.topics_

    updated_topics = [topic if topic != 1 else 0 for topic in old_topics]
    topic_model.update_topics(documents, topics=updated_topics, n_gram_range=(1, 3))

    assert len(set(old_topics)) - 1 == len(set(topic_model.topics_))

    old_topics = topic_model.topics_
    updated_topics = [topic if topic != 2 else 0 for topic in old_topics]
    topic_model.update_topics(documents, topics=updated_topics, n_gram_range=(1, 3))

    assert len(set(old_topics)) - 1 == len(set(topic_model.topics_))


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
def test_extract_topics(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    nr_topics = 5
    documents = pd.DataFrame(
        {
            "Document": documents,
            "ID": range(len(documents)),
            "Topic": np.random.randint(-1, nr_topics - 1, len(documents)),
        }
    )
    topic_model._update_topic_size(documents)
    topic_model._extract_topics(documents)
    freq = topic_model.get_topic_freq()

    assert topic_model.c_tf_idf_.shape[0] == 5
    assert topic_model.c_tf_idf_.shape[1] > 100
    assert isinstance(freq, pd.DataFrame)
    assert nr_topics == len(freq.Topic.unique())
    assert freq.Count.sum() == len(documents)
    assert len(freq.Topic.unique()) == len(freq)


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
def test_extract_topics_custom_cv(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    nr_topics = 5
    documents = pd.DataFrame(
        {
            "Document": documents,
            "ID": range(len(documents)),
            "Topic": np.random.randint(-1, nr_topics - 1, len(documents)),
        }
    )

    cv = CountVectorizer(ngram_range=(1, 2))
    topic_model.vectorizer_model = cv
    topic_model._update_topic_size(documents)
    topic_model._extract_topics(documents)
    freq = topic_model.get_topic_freq()

    assert topic_model.c_tf_idf_.shape[0] == 5
    assert topic_model.c_tf_idf_.shape[1] > 100
    assert isinstance(freq, pd.DataFrame)
    assert nr_topics == len(freq.Topic.unique())
    assert freq.Count.sum() == len(documents)
    assert len(freq.Topic.unique()) == len(freq)


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
@pytest.mark.parametrize("reduced_topics", [2, 4, 10])
def test_topic_reduction(model, reduced_topics, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    old_topics = copy.deepcopy(topic_model.topics_)
    old_freq = topic_model.get_topic_freq()

    topic_model.reduce_topics(documents, nr_topics=reduced_topics)

    new_freq = topic_model.get_topic_freq()

    if model != "online_topic_model":
        assert old_freq.Count.sum() == new_freq.Count.sum()
    assert len(old_freq.Topic.unique()) == len(old_freq)
    assert len(new_freq.Topic.unique()) == len(new_freq)
    assert len(topic_model.topics_) == len(old_topics)
    assert topic_model.topics_ != old_topics


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
def test_topic_reduction_edge_cases(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topic_model.nr_topics = 100
    nr_topics = 5
    topics = np.random.randint(-1, nr_topics - 1, len(documents))
    old_documents = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": topics})
    topic_model._update_topic_size(old_documents)
    old_documents = topic_model._sort_mappings_by_frequency(old_documents)
    topic_model._extract_topics(old_documents)
    old_freq = topic_model.get_topic_freq()

    new_documents = topic_model._reduce_topics(old_documents)
    new_freq = topic_model.get_topic_freq()

    assert not set(old_documents.Topic).difference(set(new_documents.Topic))
    pd.testing.assert_frame_equal(old_documents, new_documents)
    pd.testing.assert_frame_equal(old_freq, new_freq)


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
def test_find_topics(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    similar_topics, similarity = topic_model.find_topics("car")

    assert np.mean(similarity) > 0.1
    assert len(similar_topics) > 0
