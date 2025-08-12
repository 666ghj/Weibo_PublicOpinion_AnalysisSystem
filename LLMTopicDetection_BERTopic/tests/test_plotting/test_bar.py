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
def test_barchart(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    fig = topic_model.visualize_barchart()

    assert len(fig.to_dict()["layout"]["annotations"]) == 8
    for annotation in fig.to_dict()["layout"]["annotations"]:
        assert int(annotation["text"].split(" ")[-1]) != -1

    fig = topic_model.visualize_barchart(top_n_topics=5)

    assert len(fig.to_dict()["layout"]["annotations"]) == 5
    for annotation in fig.to_dict()["layout"]["annotations"]:
        assert int(annotation["text"].split(" ")[-1]) != -1


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
def test_barchart_outlier(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topic_model.topic_sizes_[-1] = 4
    fig = topic_model.visualize_barchart()

    assert len(fig.to_dict()["layout"]["annotations"]) == 8
    for annotation in fig.to_dict()["layout"]["annotations"]:
        assert int(annotation["text"].split(" ")[-1]) != -1

    fig = topic_model.visualize_barchart(top_n_topics=5)

    assert len(fig.to_dict()["layout"]["annotations"]) == 5
    for annotation in fig.to_dict()["layout"]["annotations"]:
        assert int(annotation["text"].split(" ")[-1]) != -1
