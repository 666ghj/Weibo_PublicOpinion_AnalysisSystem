import pytest
import logging
import numpy as np
from typing import List
from bertopic._utils import (
    check_documents_type,
    check_embeddings_shape,
    MyLogger,
    select_topic_representation,
    get_unique_distances,
)
from scipy.sparse import csr_matrix


def test_logger():
    logger = MyLogger()
    logger.configure("DEBUG")
    assert isinstance(logger.logger, logging.Logger)
    assert logger.logger.level == 10

    logger = MyLogger()
    logger.configure("WARNING")
    assert isinstance(logger.logger, logging.Logger)
    assert logger.logger.level == 30


@pytest.mark.parametrize(
    "docs",
    ["A document not in an iterable", [None], 5],
)
def test_check_documents_type(docs):
    with pytest.raises(TypeError):
        check_documents_type(docs)


def test_check_embeddings_shape():
    docs = ["doc_one", "doc_two"]
    embeddings = np.array([[1, 2, 3], [2, 3, 4]])
    check_embeddings_shape(embeddings, docs)


def test_make_unique_distances():
    def check_dists(dists: List[float], noise_max: float):
        unique_dists = get_unique_distances(np.array(dists, dtype=float), noise_max=noise_max)
        assert len(unique_dists) == len(dists), "The number of elements must be the same"
        assert len(dists) == len(np.unique(unique_dists)), "The distances must be unique"

    check_dists([0, 0, 0.5, 0.75, 1, 1], noise_max=1e-7)

    # testing whether the distances are sorted in ascending order when if the noise is extremely high
    check_dists([0, 0, 0, 0.5, 0.75, 1, 1], noise_max=20)

    # test whether the distances are sorted in ascending order when the distances are all the same
    check_dists([0, 0, 0, 0, 0, 0, 0], noise_max=1e-7)


def test_select_topic_representation():
    ctfidf_embeddings = np.array([[1, 1, 1]])
    ctfidf_embeddings_sparse = csr_matrix(
        (ctfidf_embeddings.reshape(-1).tolist(), ([0, 0, 0], [0, 1, 2])),
        shape=ctfidf_embeddings.shape,
    )
    topic_embeddings = np.array([[2, 2, 2]])

    # Use topic embeddings
    repr_, ctfidf_used = select_topic_representation(ctfidf_embeddings, topic_embeddings, use_ctfidf=False)
    np.testing.assert_array_equal(topic_embeddings, repr_)
    assert not ctfidf_used

    # Fallback to c-TF-IDF
    repr_, ctfidf_used = select_topic_representation(ctfidf_embeddings, None, use_ctfidf=False)
    np.testing.assert_array_equal(ctfidf_embeddings, repr_)
    assert ctfidf_used

    # Use c-TF-IDF
    repr_, ctfidf_used = select_topic_representation(ctfidf_embeddings, topic_embeddings, use_ctfidf=True)
    np.testing.assert_array_equal(ctfidf_embeddings, repr_)
    assert ctfidf_used

    # Fallback to topic embeddings
    repr_, ctfidf_used = select_topic_representation(None, topic_embeddings, use_ctfidf=True)
    np.testing.assert_array_equal(topic_embeddings, repr_)
    assert not ctfidf_used

    # `scipy.sparse.csr_matrix` can be used as c-TF-IDF embeddings
    np.testing.assert_array_equal(
        ctfidf_embeddings,
        select_topic_representation(ctfidf_embeddings_sparse, None, use_ctfidf=True, output_ndarray=True)[0],
    )

    # check that `csr_matrix` is not casted to `np.ndarray` when `ctfidf_as_ndarray` is False
    repr_ = select_topic_representation(ctfidf_embeddings_sparse, None, output_ndarray=False)[0]

    assert isinstance(repr_, csr_matrix)
