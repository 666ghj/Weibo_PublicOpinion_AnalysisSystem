import numpy as np


class BaseCluster:
    """The Base Cluster class.

    Using this class directly in BERTopic will make it skip
    over the cluster step. As a result, topics need to be passed
    to BERTopic in the form of its `y` parameter in order to create
    topic representations.

    Examples:
    This will skip over the cluster step in BERTopic:

    ```python
    from bertopic import BERTopic
    from bertopic.cluster import BaseCluster

    empty_cluster_model = BaseCluster()

    topic_model = BERTopic(hdbscan_model=empty_cluster_model)
    ```

    Then, this class can be used to perform manual topic modeling.
    That is, topic modeling on a topics that were already generated before
    without the need to learn them:

    ```python
    topic_model.fit(docs, y=y)
    ```
    """

    def fit(self, X, y=None):
        if y is not None:
            self.labels_ = y
        else:
            self.labels_ = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X
