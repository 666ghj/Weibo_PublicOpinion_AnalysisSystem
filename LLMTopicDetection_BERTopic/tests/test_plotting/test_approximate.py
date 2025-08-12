import copy
import pytest


@pytest.mark.parametrize("batch_size", [50, None])
@pytest.mark.parametrize("padding", [True, False])
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
def test_approximate_distribution(batch_size, padding, model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))

    # Calculate only on a document-level based on tokensets
    topic_distr, _ = topic_model.approximate_distribution(documents, padding=padding, batch_size=batch_size)
    assert topic_distr.shape[1] == len(topic_model.topic_labels_) - topic_model._outliers

    # Use the distribution visualization
    for i in range(3):
        topic_model.visualize_distribution(topic_distr[i])

    # Calculate distribution on a token-level
    topic_distr, topic_token_distr = topic_model.approximate_distribution(documents[:100], calculate_tokens=True)
    assert topic_distr.shape[1] == len(topic_model.topic_labels_) - topic_model._outliers
    assert len(topic_token_distr) == len(documents[:100])

    for token_distr in topic_token_distr:
        assert token_distr.shape[1] == len(topic_model.topic_labels_) - topic_model._outliers
