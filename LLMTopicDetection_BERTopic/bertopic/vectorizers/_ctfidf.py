from typing import List
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.utils import check_array
import numpy as np
import scipy.sparse as sp


class ClassTfidfTransformer(TfidfTransformer):
    """A Class-based TF-IDF procedure using scikit-learns TfidfTransformer as a base.

    ![](../algorithm/c-TF-IDF.svg)

    c-TF-IDF can best be explained as a TF-IDF formula adopted for multiple classes
    by joining all documents per class. Thus, each class is converted to a single document
    instead of set of documents. The frequency of each word **x** is extracted
    for each class **c** and is **l1** normalized. This constitutes the term frequency.

    Then, the term frequency is multiplied with IDF which is the logarithm of 1 plus
    the average number of words per class **A** divided by the frequency of word **x**
    across all classes.

    Arguments:
        bm25_weighting: Uses BM25-inspired idf-weighting procedure instead of the procedure
                        as defined in the c-TF-IDF formula. It uses the following weighting scheme:
                        `log(1+((avg_nr_samples - df + 0.5) / (df+0.5)))`
        reduce_frequent_words: Takes the square root of the bag-of-words after normalizing the matrix.
                               Helps to reduce the impact of words that appear too frequently.
        seed_words: Specific words that will have their idf value increased by
                    the value of `seed_multiplier`.
                    NOTE: This will only increase the value of words that have an exact match.
        seed_multiplier: The value with which the idf values of the words in `seed_words`
                         are multiplied.

    Examples:
    ```python
    transformer = ClassTfidfTransformer()
    ```
    """

    def __init__(
        self,
        bm25_weighting: bool = False,
        reduce_frequent_words: bool = False,
        seed_words: List[str] = None,
        seed_multiplier: float = 2,
    ):
        self.bm25_weighting = bm25_weighting
        self.reduce_frequent_words = reduce_frequent_words
        self.seed_words = seed_words
        self.seed_multiplier = seed_multiplier
        super(ClassTfidfTransformer, self).__init__()

    def fit(self, X: sp.csr_matrix, multiplier: np.ndarray = None):
        """Learn the idf vector (global term weights).

        Arguments:
            X: A matrix of term/token counts.
            multiplier: A multiplier for increasing/decreasing certain IDF scores
        """
        X = check_array(X, accept_sparse=("csr", "csc"))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = np.float64

        if self.use_idf:
            _, n_features = X.shape

            # Calculate the frequency of words across all classes
            df = np.squeeze(np.asarray(X.sum(axis=0)))

            # Calculate the average number of samples as regularization
            avg_nr_samples = int(X.sum(axis=1).mean())

            # BM25-inspired weighting procedure
            if self.bm25_weighting:
                idf = np.log(1 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))

            # Divide the average number of samples by the word frequency
            # +1 is added to force values to be positive
            else:
                idf = np.log((avg_nr_samples / df) + 1)

            # Multiplier to increase/decrease certain idf scores
            if multiplier is not None:
                idf = idf * multiplier

            self._idf_diag = sp.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )

        return self

    def transform(self, X: sp.csr_matrix):
        """Transform a count-based matrix to c-TF-IDF.

        Arguments:
            X (sparse matrix): A matrix of term/token counts.

        Returns:
            X (sparse matrix): A c-TF-IDF matrix
        """
        if self.use_idf:
            X = normalize(X, axis=1, norm="l1", copy=False)

            if self.reduce_frequent_words:
                X.data = np.sqrt(X.data)

            X = X * self._idf_diag

        return X
