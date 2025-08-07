import numpy as np
import umap
import sys
import os
import inspect
from tqdm import tqdm
import umap
import json

# make sure the import works even if the package has not been installed and just the files are used

from topicgpt.Clustering import Clustering_and_DimRed
from topicgpt.ExtractTopWords import ExtractTopWords
from topicgpt.TopwordEnhancement import TopwordEnhancement

class Topic:
    """
    class to represent a topic and all its attributes
    """

    def __init__(self, 
             topic_idx: str, 
             documents: list[str], 
             words: dict[str, int],
             centroid_hd: np.ndarray = None, 
             centroid_ld: np.ndarray = None,
             document_embeddings_hd: np.ndarray = None,
             document_embeddings_ld: np.ndarray = None,
             document_embedding_similarity: np.ndarray = None,
             umap_mapper: umap.UMAP = None,
             top_words: dict[str, list[str]] = None,
             top_word_scores: dict[str, list[float]] = None
             ) -> None:
        """
        Represents a topic and all its attributes.

        Args:
            topic_idx (str): Index or name of the topic.
            documents (list[str]): List of documents in the topic.
            words (dict[str, int]): Dictionary of words and their counts in the topic.
            centroid_hd (np.ndarray, optional): Centroid of the topic in high-dimensional space.
            centroid_ld (np.ndarray, optional): Centroid of the topic in low-dimensional space.
            document_embeddings_hd (np.ndarray, optional): Embeddings of documents in high-dimensional space that belong to this topic.
            document_embeddings_ld (np.ndarray, optional): Embeddings of documents in low-dimensional space that belong to this topic.
            document_embedding_similarity (np.ndarray, optional): Similarity array of document embeddings to the centroid in low-dimensional space.
            umap_mapper (umap.UMAP, optional): UMAP mapper object to map from high-dimensional space to low-dimensional space.
            top_words (dict[str, list[str]], optional): Dictionary of top words in the topic according to different metrics.
            top_word_scores (dict[str, list[float]], optional): Dictionary of how representative the top words are according to different metrics.
        """

        # do some checks on the input

        assert len(documents) == len(document_embeddings_hd) == len(document_embeddings_ld) == len(document_embedding_similarity), "documents, document_embeddings_hd, document_embeddings_ld and document_embedding_similarity must have the same length"
        assert len(documents) > 0, "documents must not be empty"
        assert len(words) > 0, "words must not be empty"


        self.topic_idx = topic_idx
        self.documents = documents
        self.words = words
        self.centroid_hd = centroid_hd
        self.centroid_ld = centroid_ld
        self.document_embeddings_hd = document_embeddings_hd
        self.document_embeddings_ld = document_embeddings_ld
        self.document_embedding_similarity = document_embedding_similarity
        self.umap_mapper = umap_mapper
        self.top_words = top_words
        self.top_word_scores = top_word_scores

        self.topic_name = None # initialize the name of the topic as none

    def __str__(self) -> str:

        if self.topic_idx and self.topic_name is None:
            repr = f"Topic {hash(self)}\n"
        if self.topic_name is None:
            repr = f"Topic: {self.topic_idx}\n"
        else: 
            repr = f"Topic {self.topic_idx}: {self.topic_name}\n"
        
        return repr
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_json(self) -> str:
        """
        return a json representation of the topic
        """
        repr_dict = {
            "topic_idx": self.topic_idx,
            "topic_name": self.topic_name,
            "topic_description": self.topic_description
        }

        json_object = json.dumps(repr_dict, indent = 4)
        return json_object
    
    def to_dict(self) -> dict:
        """
        return a dict representation of the topic
        """
        repr_dict = {
            "topic_idx": int(self.topic_idx),
            "topic_name": self.topic_name,
            "topic_description": self.topic_description
        }
        return repr_dict
    
    def set_topic_name(self, name:str):
        """
        add a name to the topic
        params:
            name: name of the topic
        """
        self.topic_name = name

    def set_topic_description(self, text: str):
        """
        add a text description to the topic
        params:
            text: text description of the topic
        """
        self.topic_description = text

def topic_to_json(topic: Topic) -> str:
    """
    Return a JSON representation of the topic.

    Args:
        topic (Topic): The topic object to convert to JSON.

    Returns:
        str: A JSON string representing the topic.
    """
    repr_dict = {
        "topic_idx": topic.topic_idx,
        "topic_name": topic.topic_name,
        "topic_description": topic.topic_description
    }

    json_object = json.dumps(repr_dict, indent = 4)
    return json_object

def topic_lis_to_json(topics: list[Topic]) -> str:
    """
    Return a JSON representation of a list of topics.

    Args:
        topics (list[Topic]): The list of topic objects to convert to JSON.

    Returns:
        str: A JSON string representing the list of topics.
    """
    repr_dict = {}
    for topic in topics:
        repr_dict[topic.topic_idx] = {
            "topic_name": topic.topic_name,
            "topic_description": topic.topic_description
        }

    json_object = json.dumps(repr_dict, indent = 4)
    return json_object

@staticmethod
def extract_topics(corpus: list[str], document_embeddings: np.ndarray, clusterer: Clustering_and_DimRed, vocab_embeddings: np.ndarray, n_topwords: int = 2000, topword_extraction_methods: list[str] = ["tfidf", "cosine_similarity"], compute_vocab_hyperparams: dict = {}) -> list[Topic]:
    """
    Extracts topics from the given corpus using the provided clusterer object on the document embeddings.

    Args:
        corpus (list[str]): List of documents.
        document_embeddings (np.ndarray): Embeddings of the documents.
        clusterer (Clustering_and_DimRed): Clustering and dimensionality reduction object to cluster the documents.
        vocab_embeddings (np.ndarray): Embeddings of the vocabulary.
        n_topwords (int, optional): Number of top-words to extract from the topics (default is 2000).
        topword_extraction_methods (list[str], optional): List of methods to extract top-words from the topics. 
            Can contain "tfidf" and "cosine_similarity" (default is ["tfidf", "cosine_similarity"]).
        compute_vocab_hyperparams (dict, optional): Hyperparameters for the top-word extraction methods.

    Returns:
        list[Topic]: List of Topic objects representing the extracted topics.
    """

    for elem in topword_extraction_methods:
        if elem not in ["tfidf", "cosine_similarity"]:
            raise ValueError("topword_extraction_methods can only contain 'tfidf' and 'cosine_similarity'")
    if topword_extraction_methods == []:
        raise ValueError("topword_extraction_methods cannot be empty")

    dim_red_embeddings, labels, umap_mapper = clusterer.cluster_and_reduce(document_embeddings)  # get dimensionality reduced embeddings, their labels and the umap mapper object

    unique_labels = np.unique(labels)  # In case the cluster labels are not consecutive numbers, we need to map them to consecutive 
    label_mapping = {label: i for i, label in enumerate(unique_labels[unique_labels != -1])}
    label_mapping[-1] = -1
    labels = np.array([label_mapping[label] for label in labels])

    extractor = ExtractTopWords()
    centroid_dict = extractor.extract_centroids(document_embeddings, labels)  # get the centroids of the clusters
    centroid_arr = np.array(list(centroid_dict.values()))
    if centroid_arr.ndim == 1:
        centroid_arr = centroid_arr.reshape(-1, 1)
    dim_red_centroids = umap_mapper.transform(np.array(list(centroid_dict.values())))  # map the centroids to low dimensional space
    
    dim_red_centroid_dict = {label: centroid for label, centroid in zip(centroid_dict.keys(), dim_red_centroids)}

    vocab = extractor.compute_corpus_vocab(corpus, **compute_vocab_hyperparams)  # compute the vocabulary of the corpus

    word_topic_mat = extractor.compute_word_topic_mat(corpus, vocab, labels, consider_outliers = False)  # compute the word-topic matrix of the corpus
    if "tfidf" in topword_extraction_methods:
        tfidf_topwords, tfidf_dict = extractor.extract_topwords_tfidf(word_topic_mat = word_topic_mat, vocab = vocab, labels = labels, top_n_words = n_topwords)  # extract the top-words according to tfidf
    if "cosine_similarity" in topword_extraction_methods:
        cosine_topwords, cosine_dict = extractor.extract_topwords_centroid_similarity(word_topic_mat = word_topic_mat, vocab = vocab, vocab_embedding_dict = vocab_embeddings, centroid_dict= dim_red_centroid_dict, umap_mapper = umap_mapper, top_n_words = n_topwords, reduce_vocab_embeddings = True, reduce_centroid_embeddings = False, consider_outliers = False)
                                                                                     
    topics = []
    for i, label in enumerate(np.unique(labels)):
        if label < -0.5: # dont include outliers
            continue
        topic_idx = f"{label}"
        documents = [doc for j, doc in enumerate(corpus) if labels[j] == label]
        embeddings_hd = document_embeddings[labels == label]
        embeddings_ld = dim_red_embeddings[labels == label]
        centroid_hd = centroid_dict[label]
        centroid_ld = dim_red_centroids[label]
        
        centroid_similarity = np.dot(embeddings_ld, centroid_ld)/(np.linalg.norm(embeddings_ld, axis = 1)*np.linalg.norm(centroid_ld))
        similarity_sorting = np.argsort(centroid_similarity)[::-1]
        documents = [documents[i] for i in similarity_sorting]
        embeddings_hd = embeddings_hd[similarity_sorting]
        embeddings_ld = embeddings_ld[similarity_sorting]

        if type(cosine_topwords[label]) == dict:
            cosine_topwords[label] = cosine_topwords[label][0]

        top_words = {
            "tfidf": tfidf_topwords[label] if "tfidf" in topword_extraction_methods else None,
            "cosine_similarity": cosine_topwords[label] if "cosine_similarity" in topword_extraction_methods else None
        }
        top_word_scores = {
            "tfidf": tfidf_dict[label] if "tfidf" in topword_extraction_methods else None,
            "cosine_similarity": cosine_dict[label] if "cosine_similarity" in topword_extraction_methods else None
        }

        topic = Topic(topic_idx = topic_idx,
                        documents = documents,
                        words = vocab,
                        centroid_hd = centroid_hd,
                        centroid_ld = centroid_ld,
                        document_embeddings_hd = embeddings_hd,
                        document_embeddings_ld = embeddings_ld,
                        document_embedding_similarity = centroid_similarity,
                        umap_mapper = umap_mapper,
                        top_words = top_words, 
                        top_word_scores = top_word_scores
                        )
                      
        topics.append(topic)
    
    return topics

@staticmethod
def extract_topics_no_new_vocab_computation(corpus: list[str], vocab: list[str], document_embeddings: np.ndarray, clusterer: Clustering_and_DimRed, vocab_embeddings: np.ndarray, n_topwords: int = 2000, topword_extraction_methods: list[str] = ["tfidf", "cosine_similarity"], consider_outliers: bool = False) -> list[Topic]:
    """
    Extracts topics from the given corpus using the provided clusterer object on the document embeddings. 
    This version does not compute the vocabulary of the corpus and instead uses the provided vocabulary.

    Args:
        corpus (list[str]): List of documents.
        vocab (list[str]): Vocabulary of the corpus.
        document_embeddings (np.ndarray): Embeddings of the documents.
        clusterer (Clustering_and_DimRed): Clustering and dimensionality reduction object to cluster the documents.
        vocab_embeddings (np.ndarray): Embeddings of the vocabulary.
        n_topwords (int, optional): Number of top-words to extract from the topics (default is 2000).
        topword_extraction_methods (list[str], optional): List of methods to extract top-words from the topics. 
            Can contain "tfidf" and "cosine_similarity" (default is ["tfidf", "cosine_similarity"]).
        consider_outliers (bool, optional): Whether to consider outliers during topic extraction (default is False).

    Returns:
        list[Topic]: List of Topic objects representing the extracted topics.
    """


    for elem in topword_extraction_methods:
        if elem not in ["tfidf", "cosine_similarity"]:
            raise ValueError("topword_extraction_methods can only contain 'tfidf' and 'cosine_similarity'")
    if topword_extraction_methods == []:
        raise ValueError("topword_extraction_methods cannot be empty")

    dim_red_embeddings, labels, umap_mapper = clusterer.cluster_and_reduce(document_embeddings)  # get dimensionality reduced embeddings, their labels and the umap mapper object

    unique_labels = np.unique(labels)  # In case the cluster labels are not consecutive numbers, we need to map them to consecutive 
    label_mapping = {label: i for i, label in enumerate(unique_labels[unique_labels != -1])}
    label_mapping[-1] = -1
    labels = np.array([label_mapping[label] for label in labels])

    extractor = ExtractTopWords()
    centroid_dict = extractor.extract_centroids(document_embeddings, labels)  # get the centroids of the clusters

    centroid_arr = np.array(list(centroid_dict.values()))
    if centroid_arr.ndim == 1:
        centroid_arr = centroid_arr.reshape(-1, 1)
    dim_red_centroids = umap_mapper.transform(np.array(list(centroid_dict.values())))  # map the centroids to low dimensional space

    dim_red_centroid_dict = {label: centroid for label, centroid in zip(centroid_dict.keys(), dim_red_centroids)}

    word_topic_mat = extractor.compute_word_topic_mat(corpus, vocab, labels, consider_outliers = consider_outliers)  # compute the word-topic matrix of the corpus
    if "tfidf" in topword_extraction_methods:
        tfidf_topwords, tfidf_dict = extractor.extract_topwords_tfidf(word_topic_mat = word_topic_mat, vocab = vocab, labels = labels, top_n_words = n_topwords)  # extract the top-words according to tfidf
    if "cosine_similarity" in topword_extraction_methods:
        cosine_topwords, cosine_dict = extractor.extract_topwords_centroid_similarity(word_topic_mat = word_topic_mat, vocab = vocab, vocab_embedding_dict = vocab_embeddings, centroid_dict= dim_red_centroid_dict, umap_mapper = umap_mapper, top_n_words = n_topwords, reduce_vocab_embeddings = True, reduce_centroid_embeddings = False, consider_outliers = True)
                                                                                           
    topics = []
    for i, label in enumerate(np.unique(labels)):
        if label < -0.5: # dont include outliers
            continue
        topic_idx = f"{label}"
        documents = [doc for j, doc in enumerate(corpus) if labels[j] == label]
        embeddings_hd = document_embeddings[labels == label]
        embeddings_ld = dim_red_embeddings[labels == label]
        centroid_hd = centroid_dict[label]
        centroid_ld = dim_red_centroids[label]
        
        centroid_similarity = np.dot(embeddings_ld, centroid_ld)/(np.linalg.norm(embeddings_ld, axis = 1)*np.linalg.norm(centroid_ld))
        similarity_sorting = np.argsort(centroid_similarity)[::-1]
        documents = [documents[i] for i in similarity_sorting]
        embeddings_hd = embeddings_hd[similarity_sorting]
        embeddings_ld = embeddings_ld[similarity_sorting]

        try:
            if type(cosine_topwords[label]) == dict:
                cosine_topwords[label] = cosine_topwords[label][0]
        except:
            pass

        top_words = {
            "tfidf": tfidf_topwords[label] if "tfidf" in topword_extraction_methods else None,
            "cosine_similarity": cosine_topwords[label] if "cosine_similarity" in topword_extraction_methods else None
        }
        top_word_scores = {
            "tfidf": tfidf_dict[label] if "tfidf" in topword_extraction_methods else None,
            "cosine_similarity": cosine_dict[label] if "cosine_similarity" in topword_extraction_methods else None
        }

        topic = Topic(topic_idx = topic_idx,
                        documents = documents,
                        words = vocab,
                        centroid_hd = centroid_hd,
                        centroid_ld = centroid_ld,
                        document_embeddings_hd = embeddings_hd,
                        document_embeddings_ld = embeddings_ld,
                        document_embedding_similarity = centroid_similarity,
                        umap_mapper = umap_mapper,
                        top_words = top_words, 
                        top_word_scores = top_word_scores
                        )
                      
        topics.append(topic)
    
    return topics

@staticmethod
def extract_and_describe_topics(corpus: list[str], document_embeddings: np.ndarray, clusterer: Clustering_and_DimRed, vocab_embeddings: np.ndarray, enhancer: TopwordEnhancement, n_topwords: int = 2000, n_topwords_description: int = 500, topword_extraction_methods: list[str] = ["tfidf", "cosine_similarity"], compute_vocab_hyperparams: dict = {}, topword_description_method: str = "cosine_similarity") -> list[Topic]:
    """
    Extracts topics from the given corpus using the provided clusterer object on the document embeddings and describes/names them using the given enhancer object.

    Args:
        corpus (list[str]): List of documents.
        document_embeddings (np.ndarray): Embeddings of the documents.
        clusterer (Clustering_and_DimRed): Clustering and dimensionality reduction object to cluster the documents.
        vocab_embeddings (np.ndarray): Embeddings of the vocabulary.
        enhancer (TopwordEnhancement): Enhancer object for enhancing top-words and generating descriptions/names for topics.
        n_topwords (int, optional): Number of top-words to extract from the topics (default is 2000).
        n_topwords_description (int, optional): Number of top-words to use from the extracted topics for description and naming (default is 500).
        topword_extraction_methods (list[str], optional): List of methods to extract top-words from the topics. 
            Can contain "tfidf" and "cosine_similarity" (default is ["tfidf", "cosine_similarity"]).
        compute_vocab_hyperparams (dict, optional): Hyperparameters for the top-word extraction methods.
        topword_description_method (str, optional): Method to use for top-word extraction for description/naming. 
            Can be "tfidf" or "cosine_similarity" (default is "cosine_similarity").

    Returns:
        list[Topic]: List of Topic objects representing the extracted and described topics.
    """

    print("Extracting topics...")
    topics = extract_topics(corpus, document_embeddings, clusterer, vocab_embeddings, n_topwords, topword_extraction_methods, compute_vocab_hyperparams)
    print("Describing topics...")
    topics = describe_and_name_topics(topics, enhancer, topword_description_method, n_topwords_description)
    return topics

@staticmethod
def extract_topics_labels_vocab(corpus: list[str], document_embeddings_hd: np.ndarray, document_embeddings_ld: np.ndarray, labels: np.ndarray, umap_mapper: umap.UMAP, vocab_embeddings: np.ndarray, vocab: list[str] = None, n_topwords: int = 2000, topword_extraction_methods: list[str] = ["tfidf", "cosine_similarity"]) -> list[Topic]:
    """
    Extracts topics from the given corpus using the provided labels that indicate the topics (no -1 for outliers). Vocabulary is already computed.

    Args:
        corpus (list[str]): List of documents.
        document_embeddings_hd (np.ndarray): Embeddings of the documents in high-dimensional space.
        document_embeddings_ld (np.ndarray): Embeddings of the documents in low-dimensional space.
        labels (np.ndarray): Labels indicating the topics.
        umap_mapper (umap.UMAP): UMAP mapper object to map from high-dimensional space to low-dimensional space.
        vocab_embeddings (np.ndarray): Embeddings of the vocabulary.
        vocab (list[str], optional): Vocabulary of the corpus (default is None).
        n_topwords (int, optional): Number of top-words to extract from the topics (default is 2000).
        topword_extraction_methods (list[str], optional): List of methods to extract top-words from the topics. 
            Can contain "tfidf" and "cosine_similarity" (default is ["tfidf", "cosine_similarity"]).

    Returns:
        list[Topic]: List of Topic objects representing the extracted topics.
    """

    for elem in topword_extraction_methods:
        if elem not in ["tfidf", "cosine_similarity"]:
            raise ValueError("topword_extraction_methods can only contain 'tfidf' and 'cosine_similarity'")
    if topword_extraction_methods == []:
        raise ValueError("topword_extraction_methods cannot be empty")
    
    if vocab is None:
        extractor = ExtractTopWords()
        vocab = extractor.compute_corpus_vocab(corpus)  # compute the vocabulary of the corpus
    
    extractor = ExtractTopWords()
    centroid_dict = extractor.extract_centroids(document_embeddings_hd, labels)  # get the centroids of the clusters
    
    centroid_arr = np.array(list(centroid_dict.values()))
    if centroid_arr.ndim == 1:
        centroid_arr = centroid_arr.reshape(-1, 1)
    dim_red_centroids = umap_mapper.transform(np.array(list(centroid_dict.values())))  # map the centroids to low dimensional space

    word_topic_mat = extractor.compute_word_topic_mat(corpus, vocab, labels, consider_outliers = False)  # compute the word-topic matrix of the corpus

    dim_red_centroid_dict = {label: centroid for label, centroid in zip(centroid_dict.keys(), dim_red_centroids)}

    if "tfidf" in topword_extraction_methods:
        tfidf_topwords, tfidf_dict = extractor.extract_topwords_tfidf(word_topic_mat = word_topic_mat, vocab = vocab, labels = labels, top_n_words = n_topwords)  # extract the top-words according to tfidf
    if "cosine_similarity" in topword_extraction_methods:
        cosine_topwords, cosine_dict = extractor.extract_topwords_centroid_similarity(word_topic_mat = word_topic_mat, vocab = vocab, vocab_embedding_dict = vocab_embeddings, centroid_dict= dim_red_centroid_dict, umap_mapper = umap_mapper, top_n_words = n_topwords, reduce_vocab_embeddings = True, reduce_centroid_embeddings = False, consider_outliers = False)
                                                                                                 
    topics = []
    for i, label in enumerate(np.unique(labels)):
        if label < -0.5: # dont include outliers
            continue
        topic_idx = f"{label}"
        documents = [doc for j, doc in enumerate(corpus) if labels[j] == label]
        embeddings_hd = document_embeddings_hd[labels == label]
        embeddings_ld = document_embeddings_ld[labels == label]
        centroid_hd = centroid_dict[label]
        centroid_ld = dim_red_centroids[label]
        
        centroid_similarity = np.dot(embeddings_ld, centroid_ld)/(np.linalg.norm(embeddings_ld, axis = 1)*np.linalg.norm(centroid_ld))
        similarity_sorting = np.argsort(centroid_similarity)[::-1]
        documents = [documents[i] for i in similarity_sorting]
        embeddings_hd = embeddings_hd[similarity_sorting]
        embeddings_ld = embeddings_ld[similarity_sorting]

        if type(cosine_topwords[label]) == dict:
            cosine_topwords[label] = cosine_topwords[label][0]
        top_words = {
            "tfidf": tfidf_topwords[label] if "tfidf" in topword_extraction_methods else None,
            "cosine_similarity": cosine_topwords[label] if "cosine_similarity" in topword_extraction_methods else None
        }
        top_word_scores = {
            "tfidf": tfidf_dict[label] if "tfidf" in topword_extraction_methods else None,
            "cosine_similarity": cosine_dict[label] if "cosine_similarity" in topword_extraction_methods else None
        }

        topic = Topic(topic_idx = topic_idx,
                        documents = documents,
                        words = vocab,
                        centroid_hd = centroid_hd,
                        centroid_ld = centroid_ld,
                        document_embeddings_hd = embeddings_hd,
                        document_embeddings_ld = embeddings_ld,
                        document_embedding_similarity = centroid_similarity,
                        umap_mapper = umap_mapper,
                        top_words = top_words, 
                        top_word_scores = top_word_scores
                        )
                      
        topics.append(topic)
    
    return topics

@staticmethod
def extract_describe_topics_labels_vocab(
    corpus: list[str],
    document_embeddings_hd: np.ndarray,
    document_embeddings_ld: np.ndarray,
    labels: np.ndarray,
    umap_mapper: umap.UMAP,
    vocab_embeddings: np.ndarray,
    enhancer: TopwordEnhancement,
    vocab: list[str] = None,
    n_topwords: int = 2000,
    n_topwords_description: int = 500,
    topword_extraction_methods: list[str] = ["tfidf", "cosine_similarity"],
    topword_description_method: str = "cosine_similarity"
) -> list[Topic]:
    """
    Extracts topics from the given corpus using the provided labels that indicate the topics (no -1 for outliers). Vocabulary is already computed.
    Describe and name the topics with the given enhancer object.

    Args:
        corpus (list[str]): List of documents.
        document_embeddings_hd (np.ndarray): Embeddings of the documents in high-dimensional space.
        document_embeddings_ld (np.ndarray): Embeddings of the documents in low-dimensional space.
        labels (np.ndarray): Labels indicating the topics.
        umap_mapper (umap.UMAP): UMAP mapper object to map from high-dimensional space to low-dimensional space.
        vocab_embeddings (np.ndarray): Embeddings of the vocabulary.
        enhancer (TopwordEnhancement): Enhancer object to enhance the top-words and generate the description.
        vocab (list[str], optional): Vocabulary of the corpus (default is None).
        n_topwords (int, optional): Number of top-words to extract from the topics (default is 2000).
        n_topwords_description (int, optional): Number of top-words to use from the extracted topics for the description and the name (default is 500).
        topword_extraction_methods (list[str], optional): List of methods to extract top-words from the topics. 
            Can contain "tfidf" and "cosine_similarity" (default is ["tfidf", "cosine_similarity"]).
        topword_description_method (str, optional): Method to use for top-word extraction. Can be "tfidf" or "cosine_similarity" (default is "cosine_similarity").

    Returns:
        list[Topic]: List of Topic objects representing the extracted topics.
    """

    topics = extract_topics_labels_vocab(corpus, document_embeddings_hd, document_embeddings_ld, labels, umap_mapper, vocab_embeddings, vocab, n_topwords, topword_extraction_methods)
    topics = describe_and_name_topics(topics, enhancer, topword_description_method, n_topwords_description)
    return topics

@staticmethod
def extract_topic_cos_sim(
    documents_topic: list[str],
    document_embeddings_topic: np.ndarray,
    words_topic: list[str],
    vocab_embeddings: dict,
    umap_mapper: umap.UMAP,
    n_topwords: int = 2000
) -> Topic:
    """
    Create a Topic object from the given documents and embeddings by computing the centroid and the top-words.
    Only uses cosine-similarity for top-word extraction.

    Args:
        documents_topic (list[str]): List of documents in the topic.
        document_embeddings_topic (np.ndarray): High-dimensional embeddings of the documents in the topic.
        words_topic (list[str]): List of words in the topic.
        vocab_embeddings (dict): Embeddings of the vocabulary.
        umap_mapper (umap.UMAP): UMAP mapper object to map from high-dimensional space to low-dimensional space.
        n_topwords (int, optional): Number of top-words to extract from the topics (default is 2000).

    Returns:
        Topic: Topic object representing the extracted topic.
    """

    topword_extraction_methods = ["cosine_similarity"]
    extractor = ExtractTopWords()
    centroid_hd = extractor.extract_centroid(document_embeddings_topic)
    centroid_ld = umap_mapper.transform(centroid_hd.reshape(1, -1))[0]

    labels = np.zeros(len(documents_topic), dtype = int) #everything has label 0   

    word_topic_mat = extractor.compute_word_topic_mat(documents_topic, words_topic, labels, consider_outliers = False)  # compute the word-topic matrix of the corpus
    if "cosine_similarity" in topword_extraction_methods:
        cosine_topwords, cosine_dict = extractor.extract_topwords_centroid_similarity(word_topic_mat = word_topic_mat, vocab = words_topic, vocab_embedding_dict = vocab_embeddings, centroid_dict= {0: centroid_ld}, umap_mapper = umap_mapper, top_n_words = n_topwords, reduce_vocab_embeddings = True, reduce_centroid_embeddings = False, consider_outliers = False)

    

    top_words = {
        "cosine_similarity": cosine_topwords if "cosine_similarity" in topword_extraction_methods else None
    }
    top_word_scores = {
        "cosine_similarity": cosine_dict if "cosine_similarity" in topword_extraction_methods else None
    }

    document_embeddings_hd = document_embeddings_topic
    document_embeddings_ld = umap_mapper.transform(document_embeddings_hd)
    document_embedding_similarity = np.dot(document_embeddings_ld, centroid_ld)/(np.linalg.norm(document_embeddings_ld, axis = 1)*np.linalg.norm(centroid_ld)) # is this correct???

    topic = Topic(topic_idx = None,
                documents = documents_topic,	
                words = words_topic,
                centroid_hd = centroid_hd,
                centroid_ld = centroid_ld,
                document_embeddings_hd = document_embeddings_hd,
                document_embeddings_ld = document_embeddings_ld,
                document_embedding_similarity = document_embedding_similarity,
                umap_mapper = umap_mapper,
                top_words = top_words,
                top_word_scores = top_word_scores
                )
    
    return topic

@staticmethod
def extract_and_describe_topic_cos_sim(
    documents_topic: list[str],
    document_embeddings_topic: np.ndarray,
    words_topic: list[str],
    vocab_embeddings: dict,
    umap_mapper: umap.UMAP,
    enhancer: TopwordEnhancement,
    n_topwords: int = 2000,
    n_topwords_description=500
) -> Topic:
    """
    Create a Topic object from the given documents and embeddings by computing the centroid and the top-words.
    Only use cosine-similarity for top-word extraction.
    Describe and name the topic with the given enhancer object.

    Args:
        documents_topic (list[str]): List of documents in the topic.
        document_embeddings_topic (np.ndarray): High-dimensional embeddings of the documents in the topic.
        words_topic (list[str]): List of words in the topic.
        vocab_embeddings (dict): Embeddings of the vocabulary.
        umap_mapper (umap.UMAP): UMAP mapper object to map from high-dimensional space to low-dimensional space.
        enhancer (TopwordEnhancement): Enhancer object to enhance the top-words and generate the description.
        n_topwords (int, optional): Number of top-words to extract from the topics (default is 2000).
        n_topwords_description (int, optional): Number of top-words to use from the extracted topics for the description and the name (default is 500).

    Returns:
        Topic: Topic object representing the extracted and described topic.
    """
    topic = extract_topic_cos_sim(documents_topic, document_embeddings_topic, words_topic, vocab_embeddings, umap_mapper, n_topwords)
    topic = describe_and_name_topics([topic], enhancer, "cosine_similarity", n_topwords_description)[0]
    return topic

    topic = extract_topic_cos_sim(documents_topic, document_embeddings_topic, words_topic, vocab_embeddings, umap_mapper, n_topwords)
    topic = describe_and_name_topics([topic], enhancer, "cosine_similarity", n_topwords_description)[0]
    return topic

@staticmethod
def describe_and_name_topics(
    topics: list[Topic],
    enhancer: TopwordEnhancement,
    topword_method="tfidf",
    n_words=500
) -> list[Topic]:
    """
    Describe and name the topics using the OpenAI API with the given enhancer object.

    Args:
        topics (list[Topic]): List of Topic objects.
        enhancer (TopwordEnhancement): Enhancer object to enhance the top-words and generate the description.
        topword_method (str, optional): Method to use for top-word extraction. Can be "tfidf" or "cosine_similarity" (default is "tfidf").
        n_words (int, optional): Number of topwords to extract for the description and the name (default is 500).

    Returns:
        list[Topic]: List of Topic objects with the description and name added.
    """

    if topword_method not in ["tfidf", "cosine_similarity"]:
        raise ValueError("topword_method can only be 'tfidf' or 'cosine_similarity'")
   
    for topic in tqdm(topics):
        tws = topic.top_words[topword_method]
        try: 
            topic_name = enhancer.generate_topic_name_str(tws, n_words = n_words)
            topic_description = enhancer.describe_topic_topwords_str(tws, n_words = n_words)
        except Exception as e:
            print(f"Error in topic {topic.topic_idx}: {e}")
            print("Trying again...")
            topic_name = enhancer.generate_topic_name_str(tws, n_words = n_words)
            topic_description = enhancer.describe_topic_topwords_str(tws, n_words = n_words)


        topic.set_topic_name(topic_name)
        topic.set_topic_description(topic_description)
        
    return topics

