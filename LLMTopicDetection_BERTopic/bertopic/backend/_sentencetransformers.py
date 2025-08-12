import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

from bertopic.backend import BaseEmbedder


class SentenceTransformerBackend(BaseEmbedder):
    """Sentence-transformers embedding model.

    The sentence-transformers embedding model used for generating document and
    word embeddings.

    Arguments:
        embedding_model: A sentence-transformers embedding model
        model2vec: Indicates whether `embedding_model` is a model2vec model.
                   NOTE: Only works if `embedding_model` is a string.
                   Otherwise, you can pass the model2vec model directly to `embedding_model`.

    Examples:
    To create a model, you can load in a string pointing to a
    sentence-transformers model:

    ```python
    from bertopic.backend import SentenceTransformerBackend

    sentence_model = SentenceTransformerBackend("all-MiniLM-L6-v2")
    ```

    or  you can instantiate a model yourself:

    ```python
    from bertopic.backend import SentenceTransformerBackend
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_model = SentenceTransformerBackend(embedding_model)
    ```

    If you want to use a model2vec model without having to install model2vec,
    you can pass the model2vec model as a string:

    ```python
    from bertopic.backend import SentenceTransformerBackend
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("minishlab/potion-base-8M", model2vec=True)
    sentence_model = SentenceTransformerBackend(embedding_model)
    ```
    """

    def __init__(self, embedding_model: Union[str, SentenceTransformer], model2vec: bool = False):
        super().__init__()

        self._hf_model = None
        if model2vec and isinstance(embedding_model, str):
            static_embedding = StaticEmbedding.from_model2vec(embedding_model)
            self.embedding_model = SentenceTransformer(modules=[static_embedding])
        elif isinstance(embedding_model, SentenceTransformer):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
            self._hf_model = embedding_model
        else:
            raise ValueError(
                "Please select a correct SentenceTransformers model: \n"
                "`from sentence_transformers import SentenceTransformer` \n"
                "`model = SentenceTransformer('all-MiniLM-L6-v2')`"
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
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings
