import numpy as np
import os
import pickle
# make sure the import works even if the package has not been installed and just the files are used
from topicgpt.Clustering import Clustering_and_DimRed
from topicgpt.ExtractTopWords import ExtractTopWords
from topicgpt.TopwordEnhancement import TopwordEnhancement
from topicgpt.GetEmbeddingsOpenAI import GetEmbeddingsOpenAI
from topicgpt.TopicPrompting import TopicPrompting
from topicgpt.TopicRepresentation import Topic
from topicgpt.Client import Client
import topicgpt.TopicRepresentation as TopicRepresentation


embeddings_path= "SavedEmbeddings/embeddings.pkl" #global variable for the path to the embeddings

class TopicGPT:
    """
    This is the main class for doing topic modelling with TopicGPT. 
    """

    def __init__(self,
             api_key: str = "",
             azure_endpoint: dict = {},
             n_topics: int = None,
             openai_prompting_model: str = "gpt-3.5-turbo-16k",
             max_number_of_tokens: int = 16384,
             corpus_instruction: str = "",
             document_embeddings: np.ndarray = None,
             vocab_embeddings: dict[str, np.ndarray] = None,
             embedding_model: str = "text-embedding-ada-002",
             max_number_of_tokens_embedding: int = 8191,
             use_saved_embeddings: bool = True,
             path_saved_embeddings: str = embeddings_path,
             clusterer: Clustering_and_DimRed = None,
             n_topwords: int = 2000,
             n_topwords_description: int = 500,
             topword_extraction_methods: list[str] = ["tfidf", "cosine_similarity"],
             compute_vocab_hyperparams: dict = {},
             enhancer: TopwordEnhancement = None,
             topic_prompting: TopicPrompting = None,
             verbose: bool = True) -> None:
        
        """
        Initializes the main class for conducting topic modeling with TopicGPT.

        Args:
            api_key (str): Your OpenAI API key. Obtain this key from https://beta.openai.com/account/api-keys.
            n_topics (int, optional): Number of topics to discover. If None, the Hdbscan algorithm (https://pypi.org/project/hdbscan/) is used to determine the number of topics automatically. Otherwise, agglomerative clustering is used. Note that with insufficient data, fewer topics may be found than specified.
            openai_prompting_model (str, optional): Model provided by OpenAI for topic description and prompts. Refer to https://platform.openai.com/docs/models for available models.
            max_number_of_tokens (int, optional): Maximum number of tokens to use for the OpenAI API.
            corpus_instruction (str, optional): Additional information about the corpus, if available, to benefit the model.
            document_embeddings (np.ndarray, optional): Document embeddings for the corpus. If None, they will be computed using the OpenAI API.
            vocab_embeddings (dict[str, np.ndarray], optional): Vocabulary embeddings for the corpus in a dictionary format where keys are words and values are embeddings. If None, they will be computed using the OpenAI API.
            embedding_model (str, optional): Name of the embedding model to use. See https://beta.openai.com/docs/api-reference/text-embedding for available models.
            max_number_of_tokens_embedding (int, optional): Maximum number of tokens to use for the OpenAI API when computing embeddings.
            use_saved_embeddings (bool, optional): Whether to use saved embeddings. If True, embeddings are loaded from the file 'SavedEmbeddings/embeddings.pkl' or path_saved_embeddings if different. If False, embeddings are computed using the OpenAI API and saved to the file.
            path_saved_embeddings (str, optional): Path to the saved embeddings file.
            clusterer (Clustering_and_DimRed, optional): Clustering and dimensionality reduction object. Find the class in the "Clustering/Clustering" folder. If None, a clustering object with default parameters is used. Note that providing document and vocab embeddings and an embedding object at the same time is not sensible; the number of topics specified in the clusterer will overwrite the n_topics argument.
            n_topwords (int, optional): Number of top words to extract and save for each topic. Note that fewer top words might be used later.
            n_topwords_description (int, optional): Number of top words to provide to the LLM (Language Model) to describe the topic.
            topword_extraction_methods (list[str], optional): List of methods for extracting top words. Available methods include "tfidf", "cosine_similarity", and "topword_enhancement". Refer to the file 'ExtractTopWords/ExtractTopWords.py' for more details.
            compute_vocab_hyperparams (dict, optional): Hyperparameters for computing vocabulary embeddings. Refer to the file 'ExtractTopWords/ExtractTopWords.py' for more details.
            enhancer (TopwordEnhancement, optional): Topword enhancement object. Used for describing topics. Find the class in the "TopwordEnhancement/TopwordEnhancement.py" folder. If None, a topword enhancement object with default parameters is used. If an openai model is specified here, it will overwrite the openai_prompting_model argument for topic description.
            topic_prompting (TopicPrompting, optional): Topic prompting object for formulating prompts. Find the class in the "TopicPrompting/TopicPrompting.py" folder. If None, a topic prompting object with default parameters is used. If an openai model is specified here, it will overwrite the openai_prompting_model argument for topic description.
            verbose (bool, optional): Whether to print detailed information about the process. This can be overridden by arguments in passed objects.
        """
        


        # Do some checks on the input arguments
        assert api_key is not None, "You need to provide an OpenAI API key."
        assert n_topics is None or n_topics > 0, "The number of topics needs to be a positive integer."
        assert max_number_of_tokens > 0, "The maximum number of tokens needs to be a positive integer."
        assert max_number_of_tokens_embedding > 0, "The maximum number of tokens for the embedding model needs to be a positive integer."
        assert n_topwords > 0, "The number of top words needs to be a positive integer."
        assert n_topwords_description > 0, "The number of top words for the topic description needs to be a positive integer."
        assert len(topword_extraction_methods) > 0, "You need to provide at least one topword extraction method."
        assert n_topwords_description <= n_topwords, "The number of top words for the topic description needs to be smaller or equal to the number of top words."

        self.client = Client(api_key = api_key, azure_endpoint = azure_endpoint)


        self.n_topics = n_topics
        self.openai_prompting_model = openai_prompting_model
        self.max_number_of_tokens = max_number_of_tokens
        self.corpus_instruction = corpus_instruction
        self.document_embeddings = document_embeddings
        self.vocab_embeddings = vocab_embeddings
        self.embedding_model = embedding_model
        self.max_number_of_tokens_embedding = max_number_of_tokens_embedding
        self.embedder = GetEmbeddingsOpenAI(client = self.client, embedding_model = self.embedding_model, max_tokens = self.max_number_of_tokens_embedding)
        self.clusterer = clusterer
        self.n_topwords = n_topwords
        self.n_topwords_description = n_topwords_description
        self.topword_extraction_methods = topword_extraction_methods
        self.compute_vocab_hyperparams = compute_vocab_hyperparams
        self.enhancer = enhancer
        self.topic_prompting = topic_prompting	
        self.use_saved_embeddings = use_saved_embeddings
        self.verbose = verbose

        self.compute_vocab_hyperparams["verbose"] = self.verbose
        
        # if embeddings have already been downloaded to the folder SavedEmbeddings, then load them
        if self.use_saved_embeddings and os.path.exists(path_saved_embeddings):
            with open(path_saved_embeddings, "rb") as f:
                self.document_embeddings, self.vocab_embeddings = pickle.load(f)

        for elem in topword_extraction_methods:
            assert elem in ["tfidf", "cosine_similarity", "topword_enhancement"], "Invalid topword extraction method. Valid methods are 'tfidf', 'cosine_similarity', and 'topword_enhancement'."
        
        if clusterer is None:
            self.clusterer = Clustering_and_DimRed(number_clusters_hdbscan = self.n_topics, verbose = self.verbose)
        else:
            self.n_topics = clusterer.number_clusters_hdbscan
        
        if enhancer is None:
            self.enhancer = TopwordEnhancement(client = self.client, openai_model = self.openai_prompting_model, max_context_length = self.max_number_of_tokens, corpus_instruction = self.corpus_instruction)

        if topic_prompting is None:
            self.topic_prompting = TopicPrompting(topic_lis = [], client = self.client, openai_prompting_model = self.openai_prompting_model,  max_context_length_promting = 16000, enhancer = self.enhancer, openai_embedding_model = self.embedding_model, max_context_length_embedding = self.max_number_of_tokens_embedding, corpus_instruction = corpus_instruction)
        
        self.extractor = ExtractTopWords()
    
    def __repr__(self) -> str:
        repr = "TopicGPT object with the following parameters:\n"
        repr += "-"*150 + "\n"
        repr += "n_topics: " + str(self.n_topics) + "\n"
        repr += "openai_prompting_model: " + self.openai_prompting_model + "\n"
        repr += "max_number_of_tokens: " + str(self.max_number_of_tokens) + "\n"
        repr += "corpus_instruction: " + self.corpus_instruction + "\n"
        repr += "embedding_model: " + self.embedding_model + "\n"
        repr += "clusterer: " + str(self.clusterer) + "\n"
        repr += "n_topwords: " + str(self.n_topwords) + "\n"
        repr += "n_topwords_description: " + str(self.n_topwords_description) + "\n"
        repr += "topword_extraction_methods: " + str(self.topword_extraction_methods) + "\n"
        repr += "compute_vocab_hyperparams: " + str(self.compute_vocab_hyperparams) + "\n"
        repr += "enhancer: " + str(self.enhancer) + "\n"
        repr += "topic_prompting: " + str(self.topic_prompting) + "\n"

        return repr

    def compute_embeddings(self, corpus: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Computes document and vocabulary embeddings for the given corpus.

        Args:
            corpus (list[str]): List of strings to embed, where each element is a document.

        Returns:
            tuple: A tuple containing two items:
                - document_embeddings (np.ndarray): Document embeddings for the corpus, with shape (len(corpus), n_embedding_dimensions).
                - vocab_embeddings (dict[str, np.ndarray]): Vocabulary embeddings for the corpus, provided as a dictionary where keys are words and values are embeddings.
        """

        
        self.document_embeddings = self.embedder.get_embeddings(corpus)["embeddings"]

        self.vocab_embeddings = self.extractor.embed_vocab_openAI(self.client, self.vocab, embedder = self.embedder)

        return self.document_embeddings, self.vocab_embeddings
    
    def extract_topics(self, corpus: list[str]) -> list[Topic]:
        """
        Extracts topics from the given corpus.

        Args:
            corpus (list[str]): List of strings to process, where each element represents a document.

        Returns:
            list[Topic]: A list of Topic objects representing the extracted topics.
        """

        assert self.document_embeddings is not None and self.vocab_embeddings is not None, "You need to compute the embeddings first."

        if self.vocab is None: 
            self.vocab = self.extractor.compute_corpus_vocab(self.corpus, **self.compute_vocab_hyperparams)
        
        self.topic_lis = TopicRepresentation.extract_topics_no_new_vocab_computation(
            corpus = corpus,
            vocab = self.vocab,
            document_embeddings = self.document_embeddings,
            clusterer = self.clusterer,
            vocab_embeddings = self.vocab_embeddings,
            n_topwords = self.n_topwords,
            topword_extraction_methods = self.topword_extraction_methods,
            consider_outliers = True
        )

        return self.topic_lis
    
    def describe_topics(self, topics: list[Topic]) -> list[Topic]:
        """
        Names and describes the provided topics using the OpenAI API.

        Args:
            topics (list[Topic]): List of Topic objects to be named and described.

        Returns:
            list[Topic]: A list of Topic objects with names and descriptions.
        """


        assert self.topic_lis is not None, "You need to extract the topics first."

        if "cosine_similarity" in self.topword_extraction_methods:
            topword_method = "cosine_similarity"
        elif "tfidf" in self.topword_extraction_methods:
            topword_method = "tfidf"
        else:
            raise ValueError("You need to use either 'cosine_similarity' or 'tfidf' as topword extraction method.")

        self.topic_lis = TopicRepresentation.describe_and_name_topics(
            topics = topics,
            enhancer = self.enhancer,
            topword_method= topword_method,
            n_words = self.n_topwords_description
        )

        return self.topic_lis
    
    def fit(self, corpus: list[str], verbose: bool = True):
        """
        Compute embeddings if necessary, extract topics, and describe them.

        Args:
            corpus (list[str]): List of strings to embed, where each element represents a document.
            verbose (bool, optional): Whether to print the progress and details of the process.
        """

        self.corpus = corpus 
        
        # remove empty documents
        len_before_removing = len(self.corpus)
        while '' in self.corpus:
            corpus.remove('')
        len_after_removing = len(self.corpus)
        if verbose:
            print("Removed " + str(len_before_removing - len_after_removing) + " empty documents.")

        if self.vocab_embeddings is None:
            if verbose:
                print("Computing vocabulary...")

            self.vocab = self.extractor.compute_corpus_vocab(self.corpus, **self.compute_vocab_hyperparams)
        else:
            print('Vocab already computed')
            self.vocab = list(self.vocab_embeddings.keys())

        if self.vocab_embeddings is None or self.document_embeddings is None:  
            if verbose:
                print("Computing embeddings...")
            self.compute_embeddings(corpus = self.corpus)
        else:
            print('Embeddings already computed')
        if verbose: 
            print("Extracting topics...")
        self.topic_lis = self.extract_topics(corpus = self.corpus)

        if verbose:
            print("Describing topics...")
        self.topic_lis = self.describe_topics(topics = self.topic_lis)

        self.topic_prompting.topic_lis = self.topic_lis
        self.topic_prompting.vocab_embeddings = self.vocab_embeddings
        self.topic_prompting.vocab = self.vocab

    def visualize_clusters(self):
        """
        Visualizes the identified clusters representing the topics in a scatterplot.
        """

        assert self.topic_lis is not None, "You need to extract the topics first."

        all_document_embeddings = np.concatenate([topic.document_embeddings_hd for topic in self.topic_lis], axis = 0)
        all_texts = np.concatenate([topic.documents for topic in self.topic_lis], axis = 0)
        all_document_indices = np.concatenate([np.repeat(i, topic.document_embeddings_hd.shape[0]) for i, topic in enumerate(self.topic_lis)], axis = 0)
        class_names = [str(topic) for topic in self.topic_lis]

        self.clusterer.visualize_clusters_dynamic(all_document_embeddings, all_document_indices, all_texts, class_names)
    
    def repr_topics(self) -> str:
        """
        Returns a string explanation of the topics.
        """

        assert self.topic_lis is not None, "You need to extract the topics first."

        if "cosine_similarity" in self.topword_extraction_methods:
            topword_method = "cosine_similarity"
        elif "tfidf" in self.topword_extraction_methods:
            topword_method = "tfidf"
        else:
            raise ValueError("You need to use either 'cosine_similarity' or 'tfidf' as topword extraction method.")

        repr = ""
        for topic in self.topic_lis:
            repr += str(topic) + "\n"
            repr += "Topic_description: " + topic.topic_description + "\n"
            repr += "Top words: " + str(topic.top_words[topword_method][:10]) + "\n"
            repr += "\n"
            repr += "-"*150 + "\n"

        return repr

    def print_topics(self):
        """
        Prints a string explanation of the topics.
        """
   
        print(self.repr_topics())

    def prompt(self, query: str) -> tuple[str, object]:
        """
        Prompts the model with the given query.

        Args:
            query (str): The query to prompt the model with.

        Returns:
            tuple: A tuple containing two items:
                - answer (str): The answer from the model.
                - function_result (object): The result of the function call.
        
        Note:
            Please refer to the TopicPrompting class for more details on available functions for prompting the model.
        """


        result = self.topic_prompting.general_prompt(query)

        answer = result[0][-1].choices[0].message.content
        function_result = result[1]
        self.topic_prompting._fix_dictionary_topwords()
        self.topic_lis = self.topic_prompting.topic_lis

        return answer, function_result
    
    def pprompt(self, query: str, return_function_result: bool = True) -> object:
        """
        Prompts the model with the given query and prints the answer.

        Args:
            query (str): The query to prompt the model with.
            return_function_result (bool, optional): Whether to return the result of the function call by the Language Model (LLM).

        Returns:
            object: The result of the function call if return_function_result is True, otherwise None.
        """


        answer, function_result = self.prompt(query)

        print(answer)

        if return_function_result:
            return function_result
        
    def save_embeddings(self, path: str = embeddings_path) -> None:
        """
        Saves the document and vocabulary embeddings to a pickle file for later re-use.

        Args:
            path (str, optional): The path to save the embeddings to. Defaults to embeddings_path.
        """


        assert self.document_embeddings is not None and self.vocab_embeddings is not None, "You need to compute the embeddings first."

        # create dictionary if it doesn't exist yet 
        if not os.path.exists("SavedEmbeddings"):
            os.makedirs("SavedEmbeddings")


        with open(path, "wb") as f:
            pickle.dump([self.document_embeddings, self.vocab_embeddings], f)

