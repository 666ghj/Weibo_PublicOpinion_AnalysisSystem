import copy
import pytest


@pytest.mark.parametrize("model", [("kmeans_pca_topic_model"), ("base_topic_model"), ("custom_topic_model")])
def test_term_rank(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topic_model.visualize_term_rank()
