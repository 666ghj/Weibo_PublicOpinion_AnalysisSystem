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
def test_generate_topic_labels(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    labels = topic_model.generate_topic_labels(topic_prefix=False)

    assert sum([label[0].isdigit() for label in labels[1:]]) / len(labels) < 0.2

    labels = [int(label.split("_")[0]) for label in topic_model.generate_topic_labels()]
    assert labels == sorted(list(set(topic_model.topics_)))

    labels = topic_model.generate_topic_labels(nr_words=1, topic_prefix=False)
    assert all([True if len(label) < 15 else False for label in labels])


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
def test_set_labels(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))

    labels = topic_model.generate_topic_labels()
    topic_model.set_topic_labels(labels)
    assert topic_model.custom_labels_ == labels

    if model != "online_topic_model":
        labels = {1: "My label", 2: "Another label"}
        topic_model.set_topic_labels(labels)
        assert topic_model.custom_labels_[1 + topic_model._outliers] == "My label"
        assert topic_model.custom_labels_[2 + topic_model._outliers] == "Another label"

        labels = {1: "Change label", 3: "New label"}
        topic_model.set_topic_labels(labels)
        assert topic_model.custom_labels_[1 + topic_model._outliers] == "Change label"
        assert topic_model.custom_labels_[3 + topic_model._outliers] == "New label"
    else:
        labels = {
            sorted(set(topic_model.topics_))[0]: "My label",
            sorted(set(topic_model.topics_))[1]: "Another label",
        }
        topic_model.set_topic_labels(labels)
        assert topic_model.custom_labels_[0] == "My label"
        assert topic_model.custom_labels_[1] == "Another label"

        labels = {
            sorted(set(topic_model.topics_))[0]: "Change label",
            sorted(set(topic_model.topics_))[2]: "New label",
        }
        topic_model.set_topic_labels(labels)
        assert topic_model.custom_labels_[0 + topic_model._outliers] == "Change label"
        assert topic_model.custom_labels_[2 + topic_model._outliers] == "New label"
