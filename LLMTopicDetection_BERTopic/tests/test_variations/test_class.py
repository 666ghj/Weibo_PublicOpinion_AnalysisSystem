import copy
import pytest
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
classes = [data["target_names"][i] for i in data["target"]][:1000]


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
def test_class(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topics_per_class_global = topic_model.topics_per_class(documents, classes=classes, global_tuning=True)
    topics_per_class_local = topic_model.topics_per_class(documents, classes=classes, global_tuning=False)

    assert topics_per_class_global.Frequency.sum() == len(documents)
    assert topics_per_class_local.Frequency.sum() == len(documents)
    assert set(topics_per_class_global.Topic.unique()) == set(topic_model.topics_)
    assert set(topics_per_class_local.Topic.unique()) == set(topic_model.topics_)
    assert len(topics_per_class_global.Class.unique()) == len(set(classes))
    assert len(topics_per_class_local.Class.unique()) == len(set(classes))
