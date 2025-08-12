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
def test_dynamic(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    timestamps = [i % 10 for i in range(len(documents))]
    topics_over_time = topic_model.topics_over_time(documents, timestamps)
    fig = topic_model.visualize_topics_over_time(topics_over_time)

    assert len(fig.to_dict()["data"]) == len(set(topic_model.topics_)) - topic_model._outliers
