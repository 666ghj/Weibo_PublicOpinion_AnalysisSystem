import numpy as np
import pandas as pd

from packaging import version
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.representation._base import BaseRepresentation
from sklearn import __version__ as sklearn_version


class KeyBERTInspired(BaseRepresentation):
    def __init__(
        self,
        top_n_words: int = 10,
        nr_repr_docs: int = 5,
        nr_samples: int = 500,
        nr_candidate_words: int = 100,
        random_state: int = 42,
    ):
        """Use a KeyBERT-like model to fine-tune the topic representations.

        The algorithm follows KeyBERT but does some optimization in
        order to speed up inference.

        The steps are as follows. First, we extract the top n representative
        documents per topic. To extract the representative documents, we
        randomly sample a number of candidate documents per cluster
        which is controlled by the `nr_samples` parameter. Then,
        the top n representative documents  are extracted by calculating
        the c-TF-IDF representation for the  candidate documents and finding,
        through cosine similarity, which are closest to the topic c-TF-IDF representation.
        Next, the top n words per topic are extracted based on their
        c-TF-IDF representation, which is controlled by the `nr_repr_docs`
        parameter.

        Then, we extract the embeddings for words and representative documents
        and create topic embeddings by averaging the representative documents.
        Finally, the most similar words to each topic are extracted by
        calculating the cosine similarity between word and topic embeddings.

        Arguments:
            top_n_words: The top n words to extract per topic.
            nr_repr_docs: The number of representative documents to extract per cluster.
            nr_samples: The number of candidate documents to extract per cluster.
            nr_candidate_words: The number of candidate words per cluster.
            random_state: The random state for randomly sampling candidate documents.

        Usage:

        ```python
        from bertopic.representation import KeyBERTInspired
        from bertopic import BERTopic

        # Create your representation model
        representation_model = KeyBERTInspired()

        # Use the representation model in BERTopic on top of the default pipeline
        topic_model = BERTopic(representation_model=representation_model)
        ```
        """
        self.top_n_words = top_n_words
        self.nr_repr_docs = nr_repr_docs
        self.nr_samples = nr_samples
        self.nr_candidate_words = nr_candidate_words
        self.random_state = random_state

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
        embeddings: np.ndarray = None,
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics.

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF
            embeddings: Pre-trained document embeddings. These can be used
                        instead of an embedding model

        Returns:
            updated_topics: Updated topic representations
        """
        # We extract the top n representative documents per class
        _, representative_docs, repr_doc_indices, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, self.nr_samples, self.nr_repr_docs
        )

        # If document embeddings are precomputed, extract the embeddings of the representative documents based on repr_doc_indices
        repr_embeddings = None
        if embeddings is not None:
            repr_embeddings = [embeddings[index] for index in np.concatenate(repr_doc_indices)]

        # We extract the top n words per class
        topics = self._extract_candidate_words(topic_model, c_tf_idf, topics)

        # We calculate the similarity between word and document embeddings and create
        # topic embeddings from the representative document embeddings
        sim_matrix, words = self._extract_embeddings(
            topic_model, topics, representative_docs, repr_doc_indices, repr_embeddings
        )
        # Find the best matching words based on the similarity matrix for each topic
        updated_topics = self._extract_top_words(words, topics, sim_matrix)

        return updated_topics

    def _extract_candidate_words(
        self,
        topic_model,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """For each topic, extract candidate words based on the c-TF-IDF
        representation.

        Arguments:
            topic_model: A BERTopic model
            c_tf_idf: The topic c-TF-IDF representation
            topics: The top words per topic

        Returns:
            topics: The `self.top_n_words` per topic
        """
        labels = [int(label) for label in sorted(list(topics.keys()))]

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = topic_model.vectorizer_model.get_feature_names_out()
        else:
            words = topic_model.vectorizer_model.get_feature_names()

        indices = topic_model._top_n_idx_sparse(c_tf_idf, self.nr_candidate_words)
        scores = topic_model._top_n_values_sparse(c_tf_idf, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        # Get top 30 words per topic based on c-TF-IDF score
        topics = {
            label: [
                (words[word_index], score) if word_index is not None and score > 0 else ("", 0.00001)
                for word_index, score in zip(indices[index][::-1], scores[index][::-1])
            ]
            for index, label in enumerate(labels)
        }
        topics = {label: list(zip(*values[: self.nr_candidate_words]))[0] for label, values in topics.items()}

        return topics

    def _extract_embeddings(
        self,
        topic_model,
        topics: Mapping[str, List[Tuple[str, float]]],
        representative_docs: List[str],
        repr_doc_indices: List[List[int]],
        repr_embeddings: np.ndarray = None,
    ) -> Union[np.ndarray, List[str]]:
        """Extract the representative document embeddings and create topic embeddings.
        Then extract word embeddings and calculate the cosine similarity between topic
        embeddings and the word embeddings. Topic embeddings are the average of
        representative document embeddings.

        Arguments:
            topic_model: A BERTopic model
            topics: The top words per topic
            representative_docs: A flat list of representative documents
            repr_doc_indices: The indices of representative documents
                              that belong to each topic
            repr_embeddings: Embeddings of respective representative_docs

        Returns:
            sim: The similarity matrix between word and topic embeddings
            vocab: The complete vocabulary of input documents
        """
        # Calculate representative document embeddings if there are no precomputed embeddings.
        if repr_embeddings is None:
            repr_embeddings = topic_model._extract_embeddings(representative_docs, method="document", verbose=False)

        topic_embeddings = [np.mean(repr_embeddings[i[0] : i[-1] + 1], axis=0) for i in repr_doc_indices]

        # Calculate word embeddings and extract best matching with updated topic_embeddings
        vocab = list(set([word for words in topics.values() for word in words]))
        word_embeddings = topic_model._extract_embeddings(vocab, method="document", verbose=False)
        sim = cosine_similarity(topic_embeddings, word_embeddings)

        return sim, vocab

    def _extract_top_words(
        self,
        vocab: List[str],
        topics: Mapping[str, List[Tuple[str, float]]],
        sim: np.ndarray,
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract the top n words per topic based on the
        similarity matrix between topics and words.

        Arguments:
            vocab: The complete vocabulary of input documents
            labels: All topic labels
            topics: The top words per topic
            sim: The similarity matrix between word and topic embeddings

        Returns:
            updated_topics: The updated topic representations
        """
        labels = [int(label) for label in sorted(list(topics.keys()))]
        updated_topics = {}
        for i, topic in enumerate(labels):
            indices = [vocab.index(word) for word in topics[topic]]
            values = sim[:, indices][i]
            word_indices = [indices[index] for index in np.argsort(values)[-self.top_n_words :]]
            updated_topics[topic] = [
                (vocab[index], val) for val, index in zip(np.sort(values)[-self.top_n_words :], word_indices)
            ][::-1]

        return updated_topics
