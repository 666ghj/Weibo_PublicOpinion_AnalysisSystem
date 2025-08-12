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
def test_delete(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    nr_topics = len(set(topic_model.topics_))
    length_documents = len(topic_model.topics_)

    # First deletion
    topics_to_delete = [1, 2]
    topic_model.delete_topics(topics_to_delete)
    mappings = topic_model.topic_mapper_.get_mappings(list(topic_model.hdbscan_model.labels_))
    mapped_labels = [mappings[label] for label in topic_model.hdbscan_model.labels_]

    if model == "online_topic_model" or model == "kmeans_pca_topic_model":
        assert nr_topics == len(set(topic_model.topics_)) + 1
        assert topic_model.get_topic_info().Count.sum() == length_documents
    else:
        assert nr_topics == len(set(topic_model.topics_)) + 2
        assert topic_model.get_topic_info().Count.sum() == length_documents

    if model == "online_topic_model":
        assert mapped_labels == topic_model.topics_[950:]
    else:
        assert mapped_labels == topic_model.topics_

    # Find two existing topics for second deletion
    remaining_topics = sorted(list(set(topic_model.topics_)))
    remaining_topics = [t for t in remaining_topics if t != -1]  # Exclude outlier topic
    topics_to_delete = remaining_topics[:2]  # Take first two remaining topics

    # Second deletion
    topic_model.delete_topics(topics_to_delete)
    mappings = topic_model.topic_mapper_.get_mappings(list(topic_model.hdbscan_model.labels_))
    mapped_labels = [mappings[label] for label in topic_model.hdbscan_model.labels_]

    if model == "online_topic_model" or model == "kmeans_pca_topic_model":
        assert nr_topics == len(set(topic_model.topics_)) + 3
        assert topic_model.get_topic_info().Count.sum() == length_documents
    else:
        assert nr_topics == len(set(topic_model.topics_)) + 4
        assert topic_model.get_topic_info().Count.sum() == length_documents

    if model == "online_topic_model":
        assert mapped_labels == topic_model.topics_[950:]
    else:
        assert mapped_labels == topic_model.topics_
