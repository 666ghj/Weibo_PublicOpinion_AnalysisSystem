import nltk
import string
import collections
from tqdm import tqdm
from typing import List
import numpy as np
import re  
from nltk.tokenize import word_tokenize
import umap
from collections import Counter
import warnings

from typing import List

# make sure the import works even if the package has not been installed and just the files are used
try:
    from topicgpt.GetEmbeddingsOpenAI import GetEmbeddingsOpenAI
except:
    from GetEmbeddingsOpenAI import GetEmbeddingsOpenAI

nltk.download('stopwords', quiet=True)  # download stopwords
nltk.download('punkt', quiet=True) # download tokenizer

class ExtractTopWords:
    
    def extract_centroids(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """
        Extract centroids of clusters.

        Args:
            embeddings (np.ndarray): Embeddings to cluster and reduce.
            labels (np.ndarray): Cluster labels. -1 means outlier.

        Returns:
            dict: Dictionary of cluster labels and their centroids.
        """

        centroid_dict = {}
        for label in np.unique(labels):
            if label != -1:
                centroid_dict[label] = np.mean(embeddings[labels == label], axis = 0)

        return centroid_dict
    
    def extract_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Extract the single centroid of a cluster.

        Args:
            embeddings (np.ndarray): Embeddings to extract the centroid from.

        Returns:
            np.ndarray: The centroid of the cluster.
        """

        return np.mean(embeddings, axis = 0)
    
    def compute_centroid_similarity(self, embeddings: np.ndarray, centroid_dict: dict, cluster_label: int) -> np.ndarray:
        """
        Compute the similarity of the document embeddings to the centroid of the cluster via cosine similarity.

        Args:
            embeddings (np.ndarray): Embeddings to cluster and reduce.
            centroid_dict (dict): Dictionary of cluster labels and their centroids.
            cluster_label (int): Cluster label for which to compute the similarity.

        Returns:
            np.ndarray: Cosine similarity of the document embeddings to the centroid of the cluster.
        """

        centroid = centroid_dict[cluster_label]
        similarity = np.dot(embeddings, centroid) / (np.linalg.norm(embeddings) * np.linalg.norm(centroid))
        return similarity
    
    def get_most_similar_docs(self, corpus: list[str], embeddings: np.ndarray, labels: np.ndarray, centroid_dict: dict, cluster_label: int, top_n: int = 10) -> List[str]:
        """
        Get the most similar documents to the centroid of a cluster.

        Args:
            corpus (list[str]): List of documents.
            embeddings (np.ndarray): Embeddings to cluster and reduce.
            labels (np.ndarray): Cluster labels. -1 means outlier.
            centroid_dict (dict): Dictionary of cluster labels and their centroids.
            cluster_label (int): Cluster label for which to compute the similarity.
            top_n (int, optional): Number of top documents to extract.

        Returns:
            List[str]: List of the most similar documents to the centroid of a cluster.
        """

        similarity = self.compute_centroid_similarity(embeddings, centroid_dict, cluster_label)
        most_similar_docs = [corpus[i] for i in np.argsort(similarity)[-top_n:][::-1]]
        return most_similar_docs
    
    def compute_corpus_vocab(self, 
                        corpus: list[str],
                        remove_stopwords: bool = True, 
                        remove_punction: bool = True, 
                        min_word_length: int = 3,
                        max_word_length: int = 20, 
                        remove_short_words: bool = True, 
                        remove_numbers: bool = True, 
                        verbose: bool = True,
                        min_doc_frequency: int = 3,
                        min_freq: float = 0.1,
                        max_freq: float = 0.9) -> list[str]:
        """
        Compute the vocabulary of the corpus and perform preprocessing of the corpus.

        Args:
            corpus (list[str]): List of documents.
            remove_stopwords (bool, optional): Whether to remove stopwords.
            remove_punction (bool, optional): Whether to remove punctuation.
            min_word_length (int, optional): Minimum word length to retain.
            max_word_length (int, optional): Maximum word length to retain.
            remove_short_words (bool, optional): Whether to remove short words.
            remove_numbers (bool, optional): Whether to remove numbers.
            verbose (bool, optional): Whether to print progress and describe what is happening.
            min_doc_frequency (int, optional): Minimum number of documents a word should appear in to be considered in the vocabulary.
            min_freq (float, optional): Minimum frequency percentile of words to be considered in the vocabulary.
            max_freq (float, optional): Maximum frequency percentile of words to be considered in the vocabulary.

        Returns:
            list[str]: List of words in the corpus sorted alphabetically.
        """

        stopwords = set(nltk.corpus.stopwords.words('english'))
        
        word_counter = collections.Counter()
        doc_frequency = collections.defaultdict(set)

        for doc_id, doc in enumerate(tqdm(corpus, disable=not verbose, desc="Processing corpus")):
            words = nltk.word_tokenize(doc)
            for word in words:
                if remove_punction and word in string.punctuation:
                    continue
                if remove_stopwords and word.lower() in stopwords:
                    continue
                if remove_numbers and re.search(r'\d', word):  # use a regular expression to check for digits
                    continue
                if not re.search('[a-zA-Z]', word):  # checks if word contains at least one alphabetic character
                    continue
                # remove words that do not begin with an alphabetic character
                if not word[0].isalpha():
                    continue
                if len(word) > max_word_length or (remove_short_words and len(word) < min_word_length):
                    continue
                
                word_lower = word.lower()
                word_counter[word_lower] += 1
                doc_frequency[word_lower].add(doc_id)

        total_words = sum(word_counter.values())
        freq_counter = {word: count / total_words for word, count in word_counter.items()}

        # print most common words and their frequencies
        if verbose:
            print("Most common words in the vocabulary:")
            for word, count in word_counter.most_common(10):
                print(f"{word}: {count}")

        freq_arr = np.array(list(freq_counter.values()))

        min_freq_value = np.quantile(freq_arr, min_freq, method="lower")
        max_freq_value = np.quantile(freq_arr, max_freq, method="higher")
        

        vocab = {}

        for word in freq_counter.keys():
            if min_freq_value <= freq_counter[word] <= max_freq_value and len(doc_frequency[word]) >= min_doc_frequency:
                vocab[word] = freq_counter[word]

        vocab = {word for word in freq_counter.keys() 
                if min_freq_value <= freq_counter[word] <= max_freq_value 
                and len(doc_frequency[word]) >= min_doc_frequency}

        # Sorting the vocabulary alphabetically
        vocab = sorted(list(vocab))
        
        return vocab

    def compute_words_topics(self, corpus: list[str], vocab: list[str], labels: np.ndarray) -> dict:
        """
        Compute the words per topic.

        Args:
            corpus (list[str]): List of documents.
            vocab (list[str]): List of words in the corpus sorted alphabetically.
            labels (np.ndarray): Cluster labels. -1 means outlier.

        Returns:
            dict: Dictionary of topics and their words.
        """


        # Download NLTK resources (only required once)
        nltk.download("punkt")
        vocab = set(vocab)

        words_per_topic = {label: [] for label in np.unique(labels) if label != -1}

        for doc, label in tqdm(zip(corpus, labels), desc="Computing words per topic", total=len(corpus)):
            if label != -1:
                words = word_tokenize(doc)
                for word in words:
                    if word.lower() in vocab:
                        words_per_topic[label].append(word.lower())

        return words_per_topic
                    
    def embed_vocab_openAI(self, client, vocab: list[str], embedder: GetEmbeddingsOpenAI = None) -> dict[str, np.ndarray]:
        """
        Embed the vocabulary using the OpenAI embedding API.

        Args:
            client: Client.
            vocab (list[str]): List of words in the corpus sorted alphabetically.
            embedder (GetEmbeddingsOpenAI, optional): Embedding object.

        Returns:
            dict[str, np.ndarray]: Dictionary of words and their embeddings.
        """

        vocab = sorted(list(set(vocab)))
        if embedder is None: 
            embedder = GetEmbeddingsOpenAI.GetEmbeddingsOpenAI(client)
        result = embedder.get_embeddings(vocab)

        res_dict = {}
        for word, emb in zip(vocab, result["embeddings"]):
            res_dict[word] = emb
        return res_dict
    
    def compute_bow_representation(self, document: str, vocab: list[str], vocab_set: set[str]) -> np.ndarray:
        """
        Compute the bag-of-words representation of a document.

        Args:
            document (str): Document to compute the bag-of-words representation of.
            vocab (list[str]): List of words in the corpus sorted alphabetically.
            vocab_set (set[str]): Set of words in the corpus sorted alphabetically.

        Returns:
            np.ndarray: Bag-of-words representation of the document.
        """

        bow = np.zeros(len(vocab))
        words = word_tokenize(document)
        if vocab_set is None:
            vocab_set = set(vocab)
        for word in words:
            if word.lower() in vocab_set:
                bow[vocab.index(word.lower())] += 1
        return bow   
    
    def compute_word_topic_mat_old(self, corpus: list[str], vocab: list[str], labels: np.ndarray, consider_outliers: bool = False) -> np.ndarray:
        """
        Compute the word-topic matrix.

        Args:
            corpus (list[str]): List of documents.
            vocab (list[str]): List of words in the corpus sorted alphabetically.
            labels (np.ndarray): Cluster labels. -1 means outlier.
            consider_outliers (bool, optional): Whether to consider outliers when computing the top words. I.e. whether the labels contain -1 to indicate outliers.

        Returns:
            np.ndarray: Word-topic matrix.
        """

        if consider_outliers:
            word_topic_mat = np.zeros(len(vocab), len((np.unique(labels))))
        else:
            word_topic_mat = np.zeros((len(vocab), len((np.unique(labels)) - 1)))

        vocab_set = set(vocab)
        for i, doc in tqdm(enumerate(corpus), desc="Computing word-topic matrix", total=len(corpus)):
            if labels[i] > - 0.5:
                bow = self.compute_bow_representation(doc, vocab, vocab_set)
                idx_to_add = labels[i]
                word_topic_mat[:, idx_to_add] += bow

        return word_topic_mat
    
    def compute_word_topic_mat(self, corpus: list[str], vocab: list[str], labels: np.ndarray, consider_outliers=False) -> np.ndarray:
        """
        Compute the word-topic matrix efficiently.

        Args:
            corpus (list[str]): List of documents.
            vocab (list[str]): List of words in the corpus, sorted alphabetically.
            labels (np.ndarray): Cluster labels. -1 indicates outliers.
            consider_outliers (bool, optional): Whether to consider outliers when computing the top words. Defaults to False.

        Returns:
            np.ndarray: Word-topic matrix.
        """


        corpus_arr = np.array(corpus) 

        if consider_outliers:
            word_topic_mat = np.zeros((len(vocab), len((np.unique(labels)))))
        else:
            word_topic_mat = np.zeros((len(vocab), len((np.unique(labels)))))
        
        for i, label in tqdm(enumerate(np.unique(labels)), desc="Computing word-topic matrix", total=len(np.unique(labels))):
            topic_docs = corpus_arr[labels == label]
            topic_doc_string = " ".join(topic_docs)
            topic_doc_words = word_tokenize(topic_doc_string)
            topic_doc_counter = Counter(topic_doc_words)

            word_topic_mat[:, i] = np.array([topic_doc_counter.get(word, 0) for word in vocab])
        
        return word_topic_mat

    def extract_topwords_tfidf(self, word_topic_mat: np.ndarray, vocab: list[str], labels: np.ndarray, top_n_words: int = 10) -> dict:
        """
        Extract the top words for each topic using a class-based tf-idf score.

        Args:
            word_topic_mat (np.ndarray): Word-topic matrix.
            vocab (list[str]): List of words in the corpus sorted alphabetically.
            labels (np.ndarray): Cluster labels. -1 means outlier.
            top_n_words (int, optional): Number of top words to extract per topic.

        Returns:
            dict: Dictionary of topics and their top words.
        """


        if min(labels) == -1:
            word_topic_mat = word_topic_mat[:, 1:]


        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            tf = word_topic_mat / np.sum(word_topic_mat, axis=0)
            idf = np.log(1 + (word_topic_mat.shape[1] / np.sum(word_topic_mat > 0, axis=1)))

            tfidf = tf * idf[:, np.newaxis]
        
            # set tfidf to zero if tf is nan (happens if word does not occur in any document or topic does not have any words)
            tfidf[np.isnan(tf)] = 0

        # extract top words for each topic
        top_words = {}
        top_word_scores = {}
        for topic in np.unique(labels):
            if topic != -1:
                indices = np.argsort(-tfidf[:, topic])[:top_n_words]
                top_words[topic] = [vocab[word_idx] for word_idx in indices]
                top_word_scores[topic] = [tfidf[word_idx, topic] for word_idx in indices]


        return top_words, top_word_scores
    
    def compute_embedding_similarity_centroids(self, vocab: list[str], vocab_embedding_dict: dict, umap_mapper: umap.UMAP, centroid_dict: dict, reduce_vocab_embeddings: bool = False, reduce_centroid_embeddings: bool = False) -> np.ndarray:
        """
        Compute the cosine similarity of each word in the vocabulary to each centroid.

        Args:
            vocab (list[str]): List of words in the corpus sorted alphabetically.
            vocab_embedding_dict (dict): Dictionary of words and their embeddings.
            umap_mapper (umap.UMAP): UMAP mapper to transform new embeddings in the same way as the document embeddings.
            centroid_dict (dict): Dictionary of cluster labels and their centroids. -1 means outlier.
            reduce_vocab_embeddings (bool, optional): Whether to reduce the vocab embeddings with the UMAP mapper.
            reduce_centroid_embeddings (bool, optional): Whether to reduce the centroid embeddings with the UMAP mapper.

        Returns:
            np.ndarray: Cosine similarity of each word in the vocab to each centroid. Has shape (len(vocab), len(centroid_dict) - 1).
        """

        embedding_dim = umap_mapper.n_components
        centroid_arr = np.zeros((len(centroid_dict), embedding_dim))
        for i, centroid in enumerate(centroid_dict.values()):
            centroid_arr[i] = centroid
        if reduce_centroid_embeddings:
            centroid_arr = umap_mapper.transform(centroid_arr)
        
        centroid_arr = centroid_arr / np.linalg.norm(centroid_arr, axis=1).reshape(-1,1)
        

        org_embedding_dim = list(vocab_embedding_dict.values())[0].shape[0]
        vocab_arr = np.zeros((len(vocab), org_embedding_dim))
        for i, word in enumerate(vocab):
            vocab_arr[i] = vocab_embedding_dict[word]
        if reduce_vocab_embeddings:
            vocab_arr = umap_mapper.transform(vocab_arr)

        vocab_arr = vocab_arr / np.linalg.norm(vocab_arr, axis=1).reshape(-1,1)
        
        similarity = vocab_arr @ centroid_arr.T # cosine similarity
        return similarity
    
    def extract_topwords_centroid_similarity(self, word_topic_mat: np.ndarray, vocab: list[str], vocab_embedding_dict: dict, centroid_dict: dict, umap_mapper: umap.UMAP, top_n_words: int = 10, reduce_vocab_embeddings: bool = True, reduce_centroid_embeddings: bool = False, consider_outliers: bool = False) -> tuple[dict, np.ndarray]:
        """
        Extract the top words for each cluster by computing the cosine similarity of the words that occur in the corpus to the centroid of the cluster.

        Args:
            word_topic_mat (np.ndarray): Word-topic matrix.
            vocab (list[str]): List of words in the corpus sorted alphabetically.
            vocab_embedding_dict (dict): Dictionary of words and their embeddings.
            centroid_dict (dict): Dictionary of cluster labels and their centroids. -1 means outlier.
            umap_mapper (umap.UMAP): UMAP mapper to transform new embeddings in the same way as the document embeddings.
            top_n_words (int, optional): Number of top words to extract per topic.
            reduce_vocab_embeddings (bool, optional): Whether to reduce the vocab embeddings with the UMAP mapper.
            reduce_centroid_embeddings (bool, optional): Whether to reduce the centroid embeddings with the UMAP mapper.
            consider_outliers (bool, optional): Whether to consider outliers when computing the top words. I.e., whether the labels contain -1 to indicate outliers.

        Returns:
            dict: Dictionary of topics and their top words.
            np.ndarray: Cosine similarity of each word in the vocab to each centroid. Has shape (len(vocab), len(centroid_dict) - 1).
        """

        similarity_mat = self.compute_embedding_similarity_centroids(vocab, vocab_embedding_dict, umap_mapper, centroid_dict, reduce_vocab_embeddings, reduce_centroid_embeddings)
        top_words = {}
        top_word_scores = {}
        
        if word_topic_mat.shape[1] > len(np.unique(list(centroid_dict.keys()))):	
            word_topic_mat = word_topic_mat[:, 1:] #ignore outliers

        for i, topic in enumerate(np.unique(list(centroid_dict.keys()))):
            if topic != -1:
                topic_similarity_mat = similarity_mat[:, topic] * word_topic_mat[:, topic]
                top_words[topic] = [vocab[word_idx] for word_idx in np.argsort(-topic_similarity_mat)[:top_n_words]]
                top_word_scores[topic] = [similarity_mat[word_idx, topic] for word_idx in np.argsort(-similarity_mat[:, topic])[:top_n_words]]

        return top_words, top_word_scores