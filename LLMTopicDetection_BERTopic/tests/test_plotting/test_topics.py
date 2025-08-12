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
def test_topics(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    fig = topic_model.visualize_topics()
    for slider in fig.to_dict()["layout"]["sliders"]:
        for step in slider["steps"]:
            assert int(step["label"].split(" ")[-1]) != -1

    fig = topic_model.visualize_topics(top_n_topics=5)
    for slider in fig.to_dict()["layout"]["sliders"]:
        for step in slider["steps"]:
            assert int(step["label"].split(" ")[-1]) != -1


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
def test_topics_outlier(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topic_model.topic_sizes_[-1] = 4
    fig = topic_model.visualize_topics()

    for slider in fig.to_dict()["layout"]["sliders"]:
        for step in slider["steps"]:
            assert int(step["label"].split(" ")[-1]) != -1

    fig = topic_model.visualize_topics(top_n_topics=5)
    for slider in fig.to_dict()["layout"]["sliders"]:
        for step in slider["steps"]:
            assert int(step["label"].split(" ")[-1]) != -1
