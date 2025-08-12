import numpy as np
from typing import List
from bertopic.backend._base import BaseEmbedder
from bertopic.backend._utils import select_backend


class WordDocEmbedder(BaseEmbedder):
    """Combine a document- and word-level embedder."""

    def __init__(self, embedding_model, word_embedding_model):
        super().__init__()

        self.embedding_model = select_backend(embedding_model)
        self.word_embedding_model = select_backend(word_embedding_model)

    def embed_words(self, words: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n words into an n-dimensional
        matrix of embeddings.

        Arguments:
            words: A list of words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Word embeddings with shape (n, m) with `n` words
            that each have an embeddings size of `m`

        """
        return self.word_embedding_model.embed(words, verbose)

    def embed_documents(self, document: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n words into an n-dimensional
        matrix of embeddings.

        Arguments:
            document: A list of documents to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document embeddings with shape (n, m) with `n` documents
            that each have an embeddings size of `m`
        """
        return self.embedding_model.embed(document, verbose)
