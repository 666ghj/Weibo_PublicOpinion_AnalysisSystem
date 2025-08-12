import numpy as np
from typing import List, Union
from model2vec import StaticModel
from sklearn.feature_extraction.text import CountVectorizer

from bertopic.backend import BaseEmbedder


class Model2VecBackend(BaseEmbedder):
    """Model2Vec embedding model.

    Arguments:
        embedding_model: Either a model2vec model or a
                         string pointing to a model2vec model
        distill: Indicates whether to distill a sentence-transformers compatible model.
                 The distillation will happen during fitting of the topic model.
                 NOTE: Only works if `embedding_model` is a string.
        distill_kwargs: Keyword arguments to pass to the distillation process
                        of `model2vec.distill.distill`
        distill_vectorizer: A CountVectorizer used for creating a custom vocabulary
                            based on the same documents used for topic modeling.
                            NOTE: If "vocabulary" is in `distill_kwargs`, this will be ignored.

    Examples:
    To create a model, you can load in a string pointing to a
    model2vec model:

    ```python
    from bertopic.backend import Model2VecBackend

    sentence_model = Model2VecBackend("minishlab/potion-base-8M")
    ```

    or  you can instantiate a model yourself:

    ```python
    from bertopic.backend import Model2VecBackend
    from model2vec import StaticModel

    embedding_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    sentence_model = Model2VecBackend(embedding_model)
    ```

    If you want to distill a sentence-transformers model with the vocabulary of the documents,
    run the following:

    ```python
    from bertopic.backend import Model2VecBackend

    sentence_model = Model2VecBackend("sentence-transformers/all-MiniLM-L6-v2", distill=True)
    ```
    """

    def __init__(
        self,
        embedding_model: Union[str, StaticModel],
        distill: bool = False,
        distill_kwargs: dict = {},
        distill_vectorizer: str = None,
    ):
        super().__init__()

        self.distill = distill
        self.distill_kwargs = distill_kwargs
        self.distill_vectorizer = distill_vectorizer
        self._has_distilled = False

        # When we distill, we need a string pointing to a sentence-transformer model
        if self.distill:
            self._check_model2vec_installation()
            if not self.distill_vectorizer:
                self.distill_vectorizer = CountVectorizer()
            if isinstance(embedding_model, str):
                self.embedding_model = embedding_model
            else:
                raise ValueError("Please pass a string pointing to a sentence-transformer model when distilling.")

        # If we don't distill, we can pass a model2vec model directly or load from a string
        elif isinstance(embedding_model, StaticModel):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = StaticModel.from_pretrained(embedding_model)
        else:
            raise ValueError(
                "Please select a correct Model2Vec model: \n"
                "`from model2vec import StaticModel` \n"
                "`model = StaticModel.from_pretrained('minishlab/potion-base-8M')`"
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
        # Distill the model
        if self.distill and not self._has_distilled:
            from model2vec.distill import distill

            # Distill with the vocabulary of the documents
            if not self.distill_kwargs.get("vocabulary"):
                X = self.distill_vectorizer.fit_transform(documents)
                word_counts = np.array(X.sum(axis=0)).flatten()
                words = self.distill_vectorizer.get_feature_names_out()
                vocabulary = [word for word, _ in sorted(zip(words, word_counts), key=lambda x: x[1], reverse=True)]
                self.distill_kwargs["vocabulary"] = vocabulary

            # Distill the model
            self.embedding_model = distill(self.embedding_model, **self.distill_kwargs)

            # Distillation should happen only once and not for every embed call
            # The distillation should only happen the first time on the entire vocabulary
            self._has_distilled = True

        # Embed the documents
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings

    def _check_model2vec_installation(self):
        try:
            from model2vec.distill import distill  # noqa: F401
        except ImportError:
            raise ImportError("To distill a model using model2vec, you need to run `pip install model2vec[distill]`")
