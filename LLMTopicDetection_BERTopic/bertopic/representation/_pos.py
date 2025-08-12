import numpy as np
import pandas as pd

import spacy
from spacy.matcher import Matcher
from spacy.language import Language

from packaging import version
from scipy.sparse import csr_matrix
from typing import List, Mapping, Tuple, Union
from sklearn import __version__ as sklearn_version
from bertopic.representation._base import BaseRepresentation


class PartOfSpeech(BaseRepresentation):
    """Extract Topic Keywords based on their Part-of-Speech.

    DEFAULT_PATTERNS = [
                [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
                [{'POS': 'NOUN'}],
                [{'POS': 'ADJ'}]
    ]

    From candidate topics, as extracted with c-TF-IDF,
    find documents that contain keywords found in the
    candidate topics. These candidate documents then
    serve as the representative set of documents from
    which the Spacy model can extract a set of candidate
    keywords for each topic.

    These candidate keywords are first judged by whether
    they fall within the DEFAULT_PATTERNS or the user-defined
    pattern. Then, the resulting keywords are sorted by
    their respective c-TF-IDF values.

    Arguments:
        model: The Spacy model to use
        top_n_words: The top n words to extract
        pos_patterns: Patterns for Spacy to use.
                      See https://spacy.io/usage/rule-based-matching

    Usage:

    ```python
    from bertopic.representation import PartOfSpeech
    from bertopic import BERTopic

    # Create your representation model
    representation_model = PartOfSpeech("en_core_web_sm")

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can define custom POS patterns to be extracted:

    ```python
    pos_patterns = [
                [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
                [{'POS': 'NOUN'}], [{'POS': 'ADJ'}]
    ]
    representation_model = PartOfSpeech("en_core_web_sm", pos_patterns=pos_patterns)
    ```
    """

    def __init__(
        self,
        model: Union[str, Language] = "en_core_web_sm",
        top_n_words: int = 10,
        pos_patterns: List[str] = None,
    ):
        if isinstance(model, str):
            self.model = spacy.load(model)
        elif isinstance(model, Language):
            self.model = model
        else:
            raise ValueError(
                "Make sure that the Spacy model that you"
                "pass is either a string referring to a"
                "Spacy model or a Spacy nlp object."
            )

        self.top_n_words = top_n_words

        if pos_patterns is None:
            self.pos_patterns = [
                [{"POS": "ADJ"}, {"POS": "NOUN"}],
                [{"POS": "NOUN"}],
                [{"POS": "ADJ"}],
            ]
        else:
            self.pos_patterns = pos_patterns

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics.

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: Not used
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        matcher = Matcher(self.model.vocab)
        matcher.add("Pattern", self.pos_patterns)

        candidate_topics = {}
        for topic, values in topics.items():
            keywords = list(zip(*values))[0]

            # Extract candidate documents
            candidate_documents = []
            for keyword in keywords:
                selection = documents.loc[documents.Topic == topic, :]
                selection = selection.loc[selection.Document.str.contains(keyword, regex=False), "Document"]
                if len(selection) > 0:
                    for document in selection[:2]:
                        candidate_documents.append(document)
            candidate_documents = list(set(candidate_documents))

            # Extract keywords
            docs_pipeline = self.model.pipe(candidate_documents)
            updated_keywords = []
            for doc in docs_pipeline:
                matches = matcher(doc)
                for _, start, end in matches:
                    updated_keywords.append(doc[start:end].text)
            candidate_topics[topic] = list(set(updated_keywords))

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = list(topic_model.vectorizer_model.get_feature_names_out())
        else:
            words = list(topic_model.vectorizer_model.get_feature_names())

        # Match updated keywords with c-TF-IDF values
        words_lookup = dict(zip(words, range(len(words))))
        updated_topics = {topic: [] for topic in topics.keys()}

        for topic, candidate_keywords in candidate_topics.items():
            word_indices = np.sort(
                [words_lookup.get(keyword) for keyword in candidate_keywords if keyword in words_lookup]
            )
            vals = topic_model.c_tf_idf_[:, word_indices][topic + topic_model._outliers]
            indices = np.argsort(np.array(vals.todense().reshape(1, -1))[0])[-self.top_n_words :][::-1]
            vals = np.sort(np.array(vals.todense().reshape(1, -1))[0])[-self.top_n_words :][::-1]
            topic_words = [(words[word_indices[index]], val) for index, val in zip(indices, vals)]
            updated_topics[topic] = topic_words
            if len(updated_topics[topic]) < self.top_n_words:
                updated_topics[topic] += [("", 0) for _ in range(self.top_n_words - len(updated_topics[topic]))]

        return updated_topics
