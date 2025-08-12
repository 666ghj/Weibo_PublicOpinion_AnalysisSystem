import numpy as np
from tqdm import tqdm
from typing import List

from bertopic.backend import BaseEmbedder


class USEBackend(BaseEmbedder):
    """Universal Sentence Encoder.

    USE encodes text into high-dimensional vectors that
    are used for semantic similarity in BERTopic.

    Arguments:
        embedding_model: An USE embedding model

    Examples:
    ```python
    import tensorflow_hub
    from bertopic.backend import USEBackend

    embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    use_embedder = USEBackend(embedding_model)
    ```
    """

    def __init__(self, embedding_model):
        super().__init__()

        try:
            embedding_model(["test sentence"])
            self.embedding_model = embedding_model
        except TypeError:
            raise ValueError(
                "Please select a correct USE model: \n"
                "`import tensorflow_hub` \n"
                "`embedding_model = tensorflow_hub.load(path_to_model)`"
            )

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n documents/words into an n-dimensional
        matrix of embeddings.

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        embeddings = np.array(
            [self.embedding_model([doc]).cpu().numpy()[0] for doc in tqdm(documents, disable=not verbose)]
        )
        return embeddings
