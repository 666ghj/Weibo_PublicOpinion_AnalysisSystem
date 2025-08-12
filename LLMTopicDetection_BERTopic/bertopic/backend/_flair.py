import numpy as np
from tqdm import tqdm
from typing import Union, List
from flair.data import Sentence
from flair.embeddings import DocumentEmbeddings, TokenEmbeddings, DocumentPoolEmbeddings

from bertopic.backend import BaseEmbedder


class FlairBackend(BaseEmbedder):
    """Flair Embedding Model.

    The Flair embedding model used for generating document and
    word embeddings.

    Arguments:
        embedding_model: A Flair embedding model

    Examples:
    ```python
    from bertopic.backend import FlairBackend
    from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings

    # Create a Flair Embedding model
    glove_embedding = WordEmbeddings('crawl')
    document_glove_embeddings = DocumentPoolEmbeddings([glove_embedding])

    # Pass the Flair model to create a new backend
    flair_embedder = FlairBackend(document_glove_embeddings)
    ```
    """

    def __init__(self, embedding_model: Union[TokenEmbeddings, DocumentEmbeddings]):
        super().__init__()

        # Flair word embeddings
        if isinstance(embedding_model, TokenEmbeddings):
            self.embedding_model = DocumentPoolEmbeddings([embedding_model])

        # Flair document embeddings + disable fine tune to prevent CUDA OOM
        # https://github.com/flairNLP/flair/issues/1719
        elif isinstance(embedding_model, DocumentEmbeddings):
            if "fine_tune" in embedding_model.__dict__:
                embedding_model.fine_tune = False
            self.embedding_model = embedding_model

        else:
            raise ValueError(
                "Please select a correct Flair model by either using preparing a token or document "
                "embedding model: \n"
                "`from flair.embeddings import TransformerDocumentEmbeddings` \n"
                "`roberta = TransformerDocumentEmbeddings('roberta-base')`"
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
        embeddings = []
        for document in tqdm(documents, disable=not verbose):
            try:
                sentence = Sentence(document) if document else Sentence("an empty document")
                self.embedding_model.embed(sentence)
            except RuntimeError:
                sentence = Sentence("an empty document")
                self.embedding_model.embed(sentence)
            embedding = sentence.embedding.detach().cpu().numpy()
            embeddings.append(embedding)
        embeddings = np.asarray(embeddings)
        return embeddings
