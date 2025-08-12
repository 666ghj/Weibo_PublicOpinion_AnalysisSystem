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
    ],
)
def test_heatmap(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topics = set(topic_model.topics_)
    if -1 in topics:
        topics.remove(-1)
    fig = topic_model.visualize_heatmap()
    fig_topics = [int(topic.split("_")[0]) for topic in fig.to_dict()["data"][0]["x"]]

    assert set(fig_topics) == topics
