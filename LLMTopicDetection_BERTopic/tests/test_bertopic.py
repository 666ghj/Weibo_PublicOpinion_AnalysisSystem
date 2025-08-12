import copy
import pytest
from bertopic import BERTopic
import importlib.util


def cuml_available():
    try:
        return importlib.util.find_spec("cuml") is not None
    except ImportError:
        return False


@pytest.mark.parametrize(
    "model",
    [
        ("base_topic_model"),
        ("kmeans_pca_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
        ("online_topic_model"),
        ("supervised_topic_model"),
        ("representation_topic_model"),
        ("zeroshot_topic_model"),
        pytest.param(
            "cuml_base_topic_model",
            marks=pytest.mark.skipif(not cuml_available(), reason="cuML not available"),
        ),
    ],
)
def test_full_model(model, documents, request):
    """Tests the entire pipeline in one go. This serves as a sanity check to see if the default
    settings result in a good separation of topics.

    NOTE: This does not cover all cases but merely combines it all together
    """
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    if model == "base_topic_model":
        topic_model.save(
            "model_dir",
            serialization="pytorch",
            save_ctfidf=True,
            save_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        topic_model = BERTopic.load("model_dir")

    if model == "cuml_base_topic_model":
        assert "cuml" in str(type(topic_model.umap_model)).lower()
        assert "cuml" in str(type(topic_model.hdbscan_model)).lower()

    topics = topic_model.topics_

    for topic in set(topics):
        words = topic_model.get_topic(topic)[:10]
        assert len(words) == 10

    for topic in topic_model.get_topic_freq().Topic:
        words = topic_model.get_topic(topic)[:10]
        assert len(words) == 10

    assert len(topic_model.get_topic_freq()) > 2
    assert len(topic_model.get_topics()) == len(topic_model.get_topic_freq())

    # Test extraction of document info
    document_info = topic_model.get_document_info(documents)
    assert len(document_info) == len(documents)

    # Test transform
    doc = "This is a new document to predict."
    topics_test, probs_test = topic_model.transform([doc, doc])

    assert len(topics_test) == 2

    # Test zero-shot topic modeling
    if topic_model._is_zeroshot():
        if topic_model._outliers:
            assert set(topic_model.topic_labels_.keys()) == set(range(-1, len(topic_model.topic_labels_) - 1))
        else:
            assert set(topic_model.topic_labels_.keys()) == set(range(len(topic_model.topic_labels_)))

    # Test topics over time
    timestamps = [i % 10 for i in range(len(documents))]
    topics_over_time = topic_model.topics_over_time(documents, timestamps)

    assert topics_over_time.Frequency.sum() == len(documents)
    assert len(topics_over_time.Topic.unique()) == len(set(topics))

    # Test hierarchical topics
    hier_topics = topic_model.hierarchical_topics(documents)

    assert len(hier_topics) > 0
    assert hier_topics.Parent_ID.astype(int).min() > max(topics)

    # Test creation of topic tree
    tree = topic_model.get_topic_tree(hier_topics, tight_layout=False)
    assert isinstance(tree, str)
    assert len(tree) > 10

    # Test find topic
    similar_topics, similarity = topic_model.find_topics("query", top_n=2)
    assert len(similar_topics) == 2
    assert len(similarity) == 2
    assert max(similarity) <= 1

    # Test topic reduction
    nr_topics = len(set(topics))
    nr_topics = 2 if nr_topics < 2 else nr_topics - 1
    topic_model.reduce_topics(documents, nr_topics=nr_topics)

    assert len(topic_model.get_topic_freq()) == nr_topics
    assert len(topic_model.topics_) == len(topics)

    # Test update topics
    topic = topic_model.get_topic(1)[:10]
    vectorizer_model = topic_model.vectorizer_model
    topic_model.update_topics(documents, n_gram_range=(2, 2))

    updated_topic = topic_model.get_topic(1)[:10]

    topic_model.update_topics(documents, vectorizer_model=vectorizer_model)
    original_topic = topic_model.get_topic(1)[:10]

    assert topic != updated_topic
    if topic_model.representation_model is not None:
        assert topic != original_topic

    # Test updating topic labels
    topic_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, word_length=10, separator=", ")
    assert len(topic_labels) == len(set(topic_model.topics_))

    # Test setting topic labels
    topic_model.set_topic_labels(topic_labels)
    assert topic_model.custom_labels_ == topic_labels

    # Test merging topics
    freq = topic_model.get_topic_freq(0)
    topics_to_merge = [0, 1]
    topic_model.merge_topics(documents, topics_to_merge)
    assert freq < topic_model.get_topic_freq(0)

    # Test reduction of outliers
    if -1 in topics:
        new_topics = topic_model.reduce_outliers(documents, topics, threshold=0.0)
        nr_outliers_topic_model = sum([1 for topic in topic_model.topics_ if topic == -1])
        nr_outliers_new_topics = sum([1 for topic in new_topics if topic == -1])

        if topic_model._outliers == 1:
            assert nr_outliers_topic_model > nr_outliers_new_topics

    # Combine models
    topic_model1 = BERTopic.load("model_dir")
    merged_model = BERTopic.merge_models([topic_model, topic_model1])

    assert len(merged_model.get_topic_info()) > len(topic_model.get_topic_info())
