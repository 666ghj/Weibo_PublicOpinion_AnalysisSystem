from bertopic.backend import BaseEmbedder
from sklearn.utils.validation import check_is_fitted, NotFittedError


class SklearnEmbedder(BaseEmbedder):
    """Scikit-Learn based embedding model.

    This component allows the usage of scikit-learn pipelines for generating document and
    word embeddings.

    Arguments:
        pipe: A scikit-learn pipeline that can `.transform()` text.

    Examples:
    Scikit-Learn is very flexible and it allows for many representations.
    A relatively simple pipeline is shown below.

    ```python
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    from bertopic.backend import SklearnEmbedder

    pipe = make_pipeline(
        TfidfVectorizer(),
        TruncatedSVD(100)
    )

    sklearn_embedder = SklearnEmbedder(pipe)
    topic_model = BERTopic(embedding_model=sklearn_embedder)
    ```

    This pipeline first constructs a sparse representation based on TF/idf and then
    makes it dense by applying SVD. Alternatively, you might also construct something
    more elaborate. As long as you construct a scikit-learn compatible pipeline, you
    should be able to pass it to Bertopic.

    !!! Warning
        One caveat to be aware of is that scikit-learns base `Pipeline` class does not
        support the `.partial_fit()`-API. If you have a pipeline that theoretically should
        be able to support online learning then you might want to explore
        the [scikit-partial](https://github.com/koaning/scikit-partial) project.
    """

    def __init__(self, pipe):
        super().__init__()
        self.pipe = pipe

    def embed(self, documents, verbose=False):
        """Embed a list of n documents/words into an n-dimensional
        matrix of embeddings.

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: No-op variable that's kept around to keep the API consistent. If you want to get feedback on training times, you should use the sklearn API.

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        try:
            check_is_fitted(self.pipe)
            embeddings = self.pipe.transform(documents)
        except NotFittedError:
            embeddings = self.pipe.fit_transform(documents)

        return embeddings
