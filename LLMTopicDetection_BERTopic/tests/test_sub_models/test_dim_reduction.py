import copy
import pytest
import numpy as np
from umap import UMAP
from sklearn.decomposition import PCA

from bertopic import BERTopic


@pytest.mark.parametrize("dim_model", [UMAP, PCA])
@pytest.mark.parametrize(
    "embeddings,shape,n_components",
    [
        (np.random.rand(100, 128), 100, 5),
        (np.random.rand(10, 256), 10, 5),
        (np.random.rand(50, 15), 50, 10),
    ],
)
def test_reduce_dimensionality(dim_model, embeddings, shape, n_components):
    model = BERTopic(umap_model=dim_model(n_components=n_components))
    umap_embeddings = model._reduce_dimensionality(embeddings)
    assert umap_embeddings.shape == (shape, n_components)


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
def test_custom_reduce_dimensionality(model, request):
    embeddings = np.random.rand(500, 128)
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    umap_embeddings = topic_model._reduce_dimensionality(embeddings)
    assert umap_embeddings.shape[1] < embeddings.shape[1]
