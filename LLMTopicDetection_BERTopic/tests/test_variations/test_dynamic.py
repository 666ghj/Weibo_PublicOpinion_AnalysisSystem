import copy
import pytest


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
        ("online_topic_model"),
    ],
)
def test_dynamic(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    timestamps = [i % 10 for i in range(len(documents))]
    topics_over_time = topic_model.topics_over_time(documents, timestamps)

    assert topics_over_time.Frequency.sum() == len(documents)
    assert set(topics_over_time.Topic.unique()) == set(topic_model.topics_)
    assert len(topics_over_time.Timestamp.unique()) == len(set(timestamps))
