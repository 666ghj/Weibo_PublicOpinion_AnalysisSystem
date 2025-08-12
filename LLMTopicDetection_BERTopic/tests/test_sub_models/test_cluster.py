import pytest
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN

from bertopic import BERTopic


@pytest.mark.parametrize("cluster_model", ["hdbscan", "kmeans"])
@pytest.mark.parametrize(
    "samples,features,centers",
    [
        (200, 500, 1),
        (500, 200, 1),
        (200, 500, 2),
        (500, 200, 2),
        (200, 500, 4),
        (500, 200, 4),
    ],
)
def test_hdbscan_cluster_embeddings(cluster_model, samples, features, centers):
    embeddings, _ = make_blobs(n_samples=samples, centers=centers, n_features=features, random_state=42)
    documents = [str(i + 1) for i in range(embeddings.shape[0])]
    old_df = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": None})

    if cluster_model == "kmeans":
        cluster_model = KMeans(n_clusters=centers)
    else:
        cluster_model = HDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

    model = BERTopic(hdbscan_model=cluster_model)
    new_df, _ = model._cluster_embeddings(embeddings, old_df)

    assert len(new_df.Topic.unique()) == centers
    assert "Topic" in new_df.columns
    pd.testing.assert_frame_equal(old_df.drop("Topic", axis=1), new_df.drop("Topic", axis=1))


@pytest.mark.parametrize("cluster_model", ["hdbscan", "kmeans"])
@pytest.mark.parametrize(
    "samples,features,centers",
    [
        (200, 500, 1),
        (500, 200, 1),
        (200, 500, 2),
        (500, 200, 2),
        (200, 500, 4),
        (500, 200, 4),
    ],
)
def test_custom_hdbscan_cluster_embeddings(cluster_model, samples, features, centers):
    embeddings, _ = make_blobs(n_samples=samples, centers=centers, n_features=features, random_state=42)
    documents = [str(i + 1) for i in range(embeddings.shape[0])]
    old_df = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": None})
    if cluster_model == "kmeans":
        cluster_model = KMeans(n_clusters=centers)
    else:
        cluster_model = HDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

    model = BERTopic(hdbscan_model=cluster_model)
    new_df, _ = model._cluster_embeddings(embeddings, old_df)

    assert len(new_df.Topic.unique()) == centers
    assert "Topic" in new_df.columns
    pd.testing.assert_frame_equal(old_df.drop("Topic", axis=1), new_df.drop("Topic", axis=1))
