import numpy as np


class BaseDimensionalityReduction:
    """The Base Dimensionality Reduction class.

    You can use this to skip over the dimensionality reduction step in BERTopic.

    Examples:
    This will skip over the reduction step in BERTopic:

    ```python
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction

    empty_reduction_model = BaseDimensionalityReduction()

    topic_model = BERTopic(umap_model=empty_reduction_model)
    ```
    """

    def fit(self, X: np.ndarray = None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X
