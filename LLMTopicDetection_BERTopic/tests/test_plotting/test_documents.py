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
def test_documents(model, reduced_embeddings, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topics = set(topic_model.topics_)
    if -1 in topics:
        topics.remove(-1)
    fig = topic_model.visualize_documents(documents, embeddings=reduced_embeddings, hide_document_hover=True)
    fig_topics = [int(data["name"].split("_")[0]) for data in fig.to_dict()["data"][1:]]
    assert set(fig_topics) == topics
