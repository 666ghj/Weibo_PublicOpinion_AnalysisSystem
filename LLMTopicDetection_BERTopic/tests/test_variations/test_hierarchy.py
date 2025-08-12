import copy
import pytest
from scipy.cluster import hierarchy as sch


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
def test_hierarchy(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    hierarchical_topics = topic_model.hierarchical_topics(documents)

    merged_topics = set([v for vals in hierarchical_topics.Topics.values for v in vals])

    assert len(hierarchical_topics) > 0
    assert merged_topics == set(topic_model.topics_).difference({-1})


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
def test_linkage(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    linkage_function = lambda x: sch.linkage(x, "single", optimal_ordering=True)
    hierarchical_topics = topic_model.hierarchical_topics(documents, linkage_function=linkage_function)
    merged_topics = set([v for vals in hierarchical_topics.Topics.values for v in vals])
    tree = topic_model.get_topic_tree(hierarchical_topics)

    assert len(hierarchical_topics) > 0
    assert len(tree) > 50
    assert len(tree.split("\n")) <= 2 * len(set(topic_model.topics_))
    assert merged_topics == set(topic_model.topics_).difference({-1})


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
def test_tree(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    linkage_function = lambda x: sch.linkage(x, "single", optimal_ordering=True)
    hierarchical_topics = topic_model.hierarchical_topics(documents, linkage_function=linkage_function)
    merged_topics = set([v for vals in hierarchical_topics.Topics.values for v in vals])
    tree = topic_model.get_topic_tree(hierarchical_topics)

    assert len(hierarchical_topics) > 0
    assert len(tree) > 50
    assert len(tree.split("\n")) <= 2 * len(set(topic_model.topics_))
    assert merged_topics == set(topic_model.topics_).difference({-1})
