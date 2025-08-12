import numpy as np
from itertools import chain
from typing import List

from scipy import sparse
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import CountVectorizer


class OnlineCountVectorizer(CountVectorizer):
    """An online variant of the CountVectorizer with updating vocabulary.

    At each `.partial_fit`, its vocabulary is updated based on any OOV words
    it might find. Then, `.update_bow` can be used to track and update
    the Bag-of-Words representation. These functions are separated such that
    the vectorizer can be used in iteration without updating the Bag-of-Words
    representation can might speed up the fitting process. However, the
    `.update_bow` function is used in BERTopic to track changes in the
    topic representations and allow for decay.

    This class inherits its parameters and attributes from:
        `sklearn.feature_extraction.text.CountVectorizer`

    Arguments:
        decay: A value between [0, 1] to weight the percentage of frequencies
               the previous bag-of-words should be decreased. For example,
               a value of `.1` will decrease the frequencies in the bag-of-words
               matrix with 10% at each iteration.
        delete_min_df: Delete words at each iteration from its vocabulary
                       that are below a minimum frequency.
                       This will keep the resulting bag-of-words matrix small
                       such that it does not explode in size with increasing
                       vocabulary. If `decay` is None then this equals `min_df`.
        **kwargs: Set of parameters inherited from:
                  `sklearn.feature_extraction.text.CountVectorizer`
                  In practice, this means that you can still use parameters
                  from the original CountVectorizer, like `stop_words` and
                  `ngram_range`.

    Attributes:
        X_ (scipy.sparse.csr_matrix) : The Bag-of-Words representation

    Examples:
    ```python
    from bertopic.vectorizers import OnlineCountVectorizer
    vectorizer = OnlineCountVectorizer(stop_words="english")

    for index, doc in enumerate(my_docs):
        vectorizer.partial_fit(doc)

        # Update and clean the bow every 100 iterations:
        if index % 100 == 0:
            X = vectorizer.update_bow()
    ```

    To use the model in BERTopic:

    ```python
    from bertopic import BERTopic
    from bertopic.vectorizers import OnlineCountVectorizer

    vectorizer_model = OnlineCountVectorizer(stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    ```

    References:
        Adapted from: https://github.com/idoshlomo/online_vectorizers
    """

    def __init__(self, decay: float = None, delete_min_df: float = None, **kwargs):
        self.decay = decay
        self.delete_min_df = delete_min_df
        super(OnlineCountVectorizer, self).__init__(**kwargs)

    def partial_fit(self, raw_documents: List[str]) -> None:
        """Perform a partial fit and update vocabulary with OOV tokens.

        Arguments:
            raw_documents: A list of documents
        """
        if not hasattr(self, "vocabulary_"):
            return self.fit(raw_documents)

        analyzer = self.build_analyzer()
        analyzed_documents = [analyzer(doc) for doc in raw_documents]
        new_tokens = set(chain.from_iterable(analyzed_documents))
        oov_tokens = new_tokens.difference(set(self.vocabulary_.keys()))

        if oov_tokens:
            max_index = max(self.vocabulary_.values())
            oov_vocabulary = dict(
                zip(
                    oov_tokens,
                    list(range(max_index + 1, max_index + 1 + len(oov_tokens), 1)),
                )
            )
            self.vocabulary_.update(oov_vocabulary)

        return self

    def update_bow(self, raw_documents: List[str]) -> csr_matrix:
        """Create or update the bag-of-words matrix.

        Update the bag-of-words matrix by adding the newly transformed
        documents. This may add empty columns if new words are found and/or
        add empty rows if new topics are found.

        During this process, the previous bag-of-words matrix might be
        decayed if `self.decay` has been set during init. Similarly, words
        that do not exceed `self.delete_min_df` are removed from its
        vocabulary and bag-of-words matrix.

        Arguments:
            raw_documents: A list of documents

        Returns:
            X_: Bag-of-words matrix
        """
        if hasattr(self, "X_"):
            X = self.transform(raw_documents)

            # Add empty columns if new words are found
            columns = csr_matrix((self.X_.shape[0], X.shape[1] - self.X_.shape[1]), dtype=int)
            self.X_ = sparse.hstack([self.X_, columns])

            # Add empty rows if new topics are found
            rows = csr_matrix((X.shape[0] - self.X_.shape[0], self.X_.shape[1]), dtype=int)
            self.X_ = sparse.vstack([self.X_, rows])

            # Decay of BoW matrix
            if self.decay is not None:
                self.X_ = self.X_ * (1 - self.decay)

            self.X_ += X
        else:
            self.X_ = self.transform(raw_documents)

        if self.delete_min_df is not None:
            self._clean_bow()

        return self.X_

    def _clean_bow(self) -> None:
        """Remove words that do not exceed `self.delete_min_df`."""
        # Only keep words with a minimum frequency
        indices = np.where(self.X_.sum(0) >= self.delete_min_df)[1]
        indices_dict = {index: index for index in indices}
        self.X_ = self.X_[:, indices]

        # Update vocabulary with new words
        new_vocab = {}
        vocabulary_dict = {v: k for k, v in self.vocabulary_.items()}
        for i, index in enumerate(indices):
            if indices_dict.get(index) is not None:
                new_vocab[vocabulary_dict[index]] = i

        self.vocabulary_ = new_vocab
