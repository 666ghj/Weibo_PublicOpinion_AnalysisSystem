import copy
import pytest
import numpy as np
import pandas as pd
from packaging import version
from scipy.sparse import csr_matrix
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer


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
def test_ctfidf(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topics = topic_model.topics_
    documents = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": topics})
    documents_per_topic = documents.groupby(["Topic"], as_index=False).agg({"Document": " ".join})
    documents = topic_model._preprocess_text(documents_per_topic.Document.values)
    count = topic_model.vectorizer_model.fit(documents)

    # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
    # and will be removed in 1.2. Please use get_feature_names_out instead.
    if version.parse(sklearn_version) >= version.parse("1.0.0"):
        words = count.get_feature_names_out()
    else:
        words = count.get_feature_names()

    X = count.transform(documents)
    transformer = ClassTfidfTransformer().fit(X)
    c_tf_idf = transformer.transform(X)

    assert len(words) > 1000
    assert all([isinstance(x, str) for x in words])

    assert isinstance(X, csr_matrix)
    assert isinstance(c_tf_idf, csr_matrix)

    assert X.shape[0] == len(set(topics))
    assert X.shape[1] == len(words)

    assert c_tf_idf.shape[0] == len(set(topics))
    assert c_tf_idf.shape[1] == len(words)

    assert np.min(X) == 0


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
def test_ctfidf_custom_cv(model, documents, request):
    cv = CountVectorizer(ngram_range=(1, 3), stop_words="english")
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topic_model.vectorizer_model = cv
    topics = topic_model.topics_
    documents = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": topics})
    documents_per_topic = documents.groupby(["Topic"], as_index=False).agg({"Document": " ".join})
    documents = topic_model._preprocess_text(documents_per_topic.Document.values)
    count = topic_model.vectorizer_model.fit(documents)

    # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
    # and will be removed in 1.2. Please use get_feature_names_out instead.
    if version.parse(sklearn_version) >= version.parse("1.0.0"):
        words = count.get_feature_names_out()
    else:
        words = count.get_feature_names()

    X = count.transform(documents)
    transformer = ClassTfidfTransformer().fit(X)
    c_tf_idf = transformer.transform(X)

    assert len(words) > 1000
    assert all([isinstance(x, str) for x in words])

    assert isinstance(X, csr_matrix)
    assert isinstance(c_tf_idf, csr_matrix)

    assert X.shape[0] == len(set(topics))
    assert X.shape[1] == len(words)

    assert c_tf_idf.shape[0] == len(set(topics))
    assert c_tf_idf.shape[1] == len(words)

    assert np.min(X) == 0
