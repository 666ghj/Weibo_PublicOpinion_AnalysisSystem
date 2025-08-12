import copy
import pytest


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
def test_merge(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    nr_topics = len(set(topic_model.topics_))

    topics_to_merge = [1, 2]
    topic_model.merge_topics(documents, topics_to_merge)
    mappings = topic_model.topic_mapper_.get_mappings(list(topic_model.hdbscan_model.labels_))
    mapped_labels = [mappings[label] for label in topic_model.hdbscan_model.labels_]

    assert nr_topics == len(set(topic_model.topics_)) + 1
    assert topic_model.get_topic_info().Count.sum() == len(documents)
    if model == "online_topic_model":
        assert mapped_labels == topic_model.topics_[950:]
    else:
        assert mapped_labels == topic_model.topics_

    topics_to_merge = [1, 2]
    topic_model.merge_topics(documents, topics_to_merge)
    mappings = topic_model.topic_mapper_.get_mappings(list(topic_model.hdbscan_model.labels_))
    mapped_labels = [mappings[label] for label in topic_model.hdbscan_model.labels_]

    assert nr_topics == len(set(topic_model.topics_)) + 2
    assert topic_model.get_topic_info().Count.sum() == len(documents)
    if model == "online_topic_model":
        assert mapped_labels == topic_model.topics_[950:]
    else:
        assert mapped_labels == topic_model.topics_
