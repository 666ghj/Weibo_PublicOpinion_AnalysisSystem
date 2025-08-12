import warnings
import numpy as np
import pandas as pd
from typing import List, Mapping, Tuple
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.representation._base import BaseRepresentation


class MaximalMarginalRelevance(BaseRepresentation):
    """Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.

    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.

    Arguments:
        diversity: How diverse the select keywords/keyphrases are.
                    Values range between 0 and 1 with 0 being not diverse at all
                    and 1 being most diverse.
        top_n_words: The number of keywords/keyhprases to return

    Usage:

    ```python
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic import BERTopic

    # Create your representation model
    representation_model = MaximalMarginalRelevance(diversity=0.3)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```
    """

    def __init__(self, diversity: float = 0.1, top_n_words: int = 10):
        self.diversity = diversity
        self.top_n_words = top_n_words

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topic representations.

        Arguments:
            topic_model: The BERTopic model
            documents: Not used
            c_tf_idf: Not used
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        if topic_model.embedding_model is None:
            warnings.warn(
                "MaximalMarginalRelevance can only be used BERTopic was instantiated"
                "with the `embedding_model` parameter."
            )
            return topics

        updated_topics = {}
        for topic, topic_words in topics.items():
            words = [word[0] for word in topic_words]
            word_embeddings = topic_model._extract_embeddings(words, method="word", verbose=False)
            topic_embedding = topic_model._extract_embeddings(" ".join(words), method="word", verbose=False).reshape(
                1, -1
            )
            topic_words = mmr(
                topic_embedding,
                word_embeddings,
                words,
                self.diversity,
                self.top_n_words,
            )
            updated_topics[topic] = [(word, value) for word, value in topics[topic] if word in topic_words]
        return updated_topics


def mmr(
    doc_embedding: np.ndarray,
    word_embeddings: np.ndarray,
    words: List[str],
    diversity: float = 0.1,
    top_n: int = 10,
) -> List[str]:
    """Maximal Marginal Relevance.

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        diversity: The diversity of the selected embeddings.
                   Values between 0 and 1.
        top_n: The top n items to return

    Returns:
            List[str]: The selected keywords/keyphrases
    """
    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]
