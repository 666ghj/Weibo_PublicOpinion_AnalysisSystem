import numpy as np
import pandas as pd
import logging
from collections.abc import Iterable
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from typing import Optional, Union, Tuple


class MyLogger:
    def __init__(self):
        self.logger = logging.getLogger("BERTopic")

    def configure(self, level):
        self.set_level(level)
        self._add_handler()
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(f"{message}")

    def warning(self, message):
        self.logger.warning(f"WARNING: {message}")

    def set_level(self, level):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level in levels:
            self.logger.setLevel(level)

    def _add_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
        self.logger.addHandler(sh)

        # Remove duplicate handlers
        if len(self.logger.handlers) > 1:
            self.logger.handlers = [self.logger.handlers[0]]


def check_documents_type(documents):
    """Check whether the input documents are indeed a list of strings."""
    if isinstance(documents, pd.DataFrame):
        raise TypeError("Make sure to supply a list of strings, not a dataframe.")
    elif isinstance(documents, Iterable) and not isinstance(documents, str):
        if not any([isinstance(doc, str) for doc in documents]):
            raise TypeError("Make sure that the iterable only contains strings.")
    else:
        raise TypeError("Make sure that the documents variable is an iterable containing strings only.")


def check_embeddings_shape(embeddings, docs):
    """Check if the embeddings have the correct shape."""
    if embeddings is not None:
        if not any([isinstance(embeddings, np.ndarray), isinstance(embeddings, csr_matrix)]):
            raise ValueError("Make sure to input embeddings as a numpy array or scipy.sparse.csr.csr_matrix. ")
        else:
            if embeddings.shape[0] != len(docs):
                raise ValueError(
                    "Make sure that the embeddings are a numpy array with shape: "
                    "(len(docs), vector_dim) where vector_dim is the dimensionality "
                    "of the vector embeddings. "
                )


def check_is_fitted(topic_model):
    """Checks if the model was fitted by verifying the presence of self.matches.

    Arguments:
        topic_model: BERTopic instance for which the check is performed.

    Returns:
        None

    Raises:
        ValueError: If the matches were not found.
    """
    msg = "This %(name)s instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."

    if topic_model.topics_ is None:
        raise ValueError(msg % {"name": type(topic_model).__name__})


class NotInstalled:
    """This object is used to notify the user that additional dependencies need to be
    installed in order to use the string matching model.
    """

    def __init__(self, tool, dep, custom_msg=None):
        self.tool = tool
        self.dep = dep

        msg = f"In order to use {self.tool} you will need to install via;\n\n"
        if custom_msg is not None:
            msg += custom_msg
        else:
            msg += f"pip install bertopic[{self.dep}]\n\n"
        self.msg = msg

    def __getattr__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.msg)

    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.msg)


def validate_distance_matrix(X, n_samples):
    """Validate the distance matrix and convert it to a condensed distance matrix
    if necessary.

    A valid distance matrix is either a square matrix of shape (n_samples, n_samples)
    with zeros on the diagonal and non-negative values or condensed distance matrix
    of shape (n_samples * (n_samples - 1) / 2,) containing the upper triangular of the
    distance matrix.

    Arguments:
        X: Distance matrix to validate.
        n_samples: Number of samples in the dataset.

    Returns:
        X: Validated distance matrix.

    Raises:
        ValueError: If the distance matrix is not valid.
    """
    # Make sure it is the 1-D condensed distance matrix with zeros on the diagonal
    s = X.shape
    if len(s) == 1:
        # check it has correct size
        n = s[0]
        if n != (n_samples * (n_samples - 1) / 2):
            raise ValueError("The condensed distance matrix must have shape (n*(n-1)/2,).")
    elif len(s) == 2:
        # check it has correct size
        if (s[0] != n_samples) or (s[1] != n_samples):
            raise ValueError("The distance matrix must be of shape (n, n) where n is the number of samples.")
        # force zero diagonal and convert to condensed
        np.fill_diagonal(X, 0)
        X = squareform(X)
    else:
        raise ValueError(
            "The distance matrix must be either a 1-D condensed "
            "distance matrix of shape (n*(n-1)/2,) or a "
            "2-D square distance matrix of shape (n, n)."
            "where n is the number of documents."
            "Got a distance matrix of shape %s" % str(s)
        )

    # Make sure its entries are non-negative
    if np.any(X < 0):
        raise ValueError("Distance matrix cannot contain negative values.")

    return X


def get_unique_distances(dists: np.array, noise_max=1e-7) -> np.array:
    """Check if the consecutive elements in the distance array are the same. If so, a small noise
    is added to one of the elements to make sure that the array does not contain duplicates.

    Arguments:
        dists: distance array sorted in the increasing order.
        noise_max: the maximal magnitude of noise to be added.

    Returns:
         Unique distances sorted in the preserved increasing order.
    """
    dists_cp = dists.copy()

    for i in range(dists.shape[0] - 1):
        if dists[i] == dists[i + 1]:
            # returns the next unique distance or the current distance with the added noise
            next_unique_dist = next((d for d in dists[i + 1 :] if d != dists[i]), dists[i] + noise_max)

            # the noise can never be large then the difference between the next unique distance and the current one
            curr_max_noise = min(noise_max, next_unique_dist - dists_cp[i])
            dists_cp[i + 1] = np.random.uniform(low=dists_cp[i] + curr_max_noise / 2, high=dists_cp[i] + curr_max_noise)
    return dists_cp


def select_topic_representation(
    ctfidf_embeddings: Optional[Union[np.ndarray, csr_matrix]] = None,
    embeddings: Optional[Union[np.ndarray, csr_matrix]] = None,
    use_ctfidf: bool = True,
    output_ndarray: bool = False,
) -> Tuple[np.ndarray, bool]:
    """Select the topic representation.

    Arguments:
        ctfidf_embeddings: The c-TF-IDF embedding matrix
        embeddings: The topic embedding matrix
        use_ctfidf: Whether to use the c-TF-IDF representation. If False, topics embedding representation is used, if it
                    exists. Default is True.
        output_ndarray: Whether to convert the selected representation into ndarray
    Raises
        ValueError:
            - If no topic representation was found
            - If c-TF-IDF embeddings are not a numpy array or a scipy.sparse.csr_matrix

    Returns:
        The selected topic representation and a boolean indicating whether it is c-TF-IDF.
    """

    def to_ndarray(array: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        if isinstance(array, csr_matrix):
            return array.toarray()
        return array

    logger = MyLogger()

    if use_ctfidf:
        if ctfidf_embeddings is None:
            logger.warning(
                "No c-TF-IDF matrix was found despite it is supposed to be used (`use_ctfidf` is True). "
                "Defaulting to semantic embeddings."
            )
            repr_, ctfidf_used = embeddings, False
        else:
            repr_, ctfidf_used = ctfidf_embeddings, True
    else:
        if embeddings is None:
            logger.warning(
                "No topic embeddings were found despite they are supposed to be used (`use_ctfidf` is False). "
                "Defaulting to c-TF-IDF representation."
            )
            repr_, ctfidf_used = ctfidf_embeddings, True
        else:
            repr_, ctfidf_used = embeddings, False

    return to_ndarray(repr_) if output_ndarray else repr_, ctfidf_used
