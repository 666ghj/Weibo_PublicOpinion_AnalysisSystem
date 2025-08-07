import openai
from openai import OpenAI
import numpy as np
import json
import tiktoken
import openai
from openai import OpenAI
import re
import sklearn
import hdbscan
from copy import deepcopy

# make sure the import works even if the package has not been installed and just the files are used
try: 
    from topicgpt.TopicRepresentation import Topic
    from topicgpt.TopicRepresentation import extract_and_describe_topic_cos_sim
    from topicgpt.TopicRepresentation import extract_describe_topics_labels_vocab
    from topicgpt.TopwordEnhancement import TopwordEnhancement
except:
    from TopicRepresentation import Topic
    from TopicRepresentation import extract_and_describe_topic_cos_sim
    from TopicRepresentation import extract_describe_topics_labels_vocab
    from TopwordEnhancement import TopwordEnhancement


basic_model_instruction = """You are a helpful assistant. 
You are excellent at inferring information about topics discovered via topic modelling using information retrieval. 
You summarize information intelligently. 
You use the functions you are provided with if applicable.
You make sure that everything you output is strictly based on the provided text. If you cite documents, give their indices. 
You always explicitly say if you don't find any useful information!
You only say that something is contained in the corpus if you are very sure about it!"""


class TopicPrompting:
    """
    This class allows to formulate prompts and queries against the identified topics to get more information about them
    """

    def __init__(self, 
             topic_lis: list[Topic], 
             client,
             openai_prompting_model: str = "gpt-3.5-turbo-16k", 
             max_context_length_promting: int = 16000, 
             openai_model_temperature_prompting: float = 0.5,
             openai_embedding_model: str = "text-embedding-ada-002",
             max_context_length_embedding: int = 8191, 
             basic_model_instruction: str = basic_model_instruction,
             corpus_instruction: str = "",
             enhancer: TopwordEnhancement = None,
             vocab: list = None,
             vocab_embeddings: dict = None,
             random_state: int = 42):
        """
        Initialize the object.

        Args:
            topic_list (list[Topic]): List of Topic objects.
            client: Client.
            openai_prompting_model (str, optional): OpenAI model to use for prompting (default is "gpt-3.5-turbo-16k").
            max_context_length_prompting (int, optional): Maximum context length for the prompting model (default is 16000).
            openai_model_temperature_prompting (float, optional): Temperature for the prompting model (default is 0.5).
            openai_embedding_model (str, optional): OpenAI model to use for computing embeddings for similarity search (default is "text-embedding-ada-002").
            max_context_length_embedding (int, optional): Maximum context length for the embedding model (default is 8191).
            basic_model_instruction (str, optional): Basic instruction for the prompting model.
            corpus_instruction (str, optional): Instruction for the prompting model to use the corpus.
            enhancer (TopwordEnhancement, optional): TopwordEnhancement object for naming and describing the topics (default is None).
            vocab (list, optional): Vocabulary of the corpus (default is None).
            vocab_embeddings (dict, optional): Dictionary mapping words to their embeddings (default is None).
            random_state (int, optional): Random state for reproducibility (default is 42).
        """

        self.topic_lis = topic_lis
        self.client = client
        self.openai_prompting_model = openai_prompting_model
        self.max_context_length_promting = max_context_length_promting
        self.openai_model_temperature_prompting = openai_model_temperature_prompting
        self.openai_embedding_model = openai_embedding_model
        self.max_context_length_embedding = max_context_length_embedding    
        self.basic_model_instruction = basic_model_instruction
        self.corpus_instruction = f" The following information is available about the corpus used to identify the topics: {corpus_instruction}.\n"
        self.enhancer = enhancer
        self.vocab = vocab
        self.vocab_embeddings = vocab_embeddings
        self.random_state = random_state


        self.function_descriptions = {
                "knn_search": {
                    "name": "knn_search",
                    "description": "This function is the best choice to find out if a topic is about a specific subject or keyword or aspects or contains information about it. It should also be used to infer the subtopics of a given topic. Note that it is possible that just useless documents are returned.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic_index": {
                                "type": "integer",
                                "description": "index of the topic to search in."
                            },
                            "query": {
                                "type": "string",
                                "description": "query string. Can be a single word or a sentence. Used to create an embedding and search a vector database for the k nearest neighbors."
                            },
                            "k": {
                                "type": "integer",
                                "description": "number of neighbors to return. Use more neighbors to get a more diverse and comprehensive set of results."
                            }
                        },
                        "required": ["topic_index", "query"]

                    }
                },
                "identify_topic_idx": {
                    "name": "identify_topic_idx",
                    "description": "This function can be used to identify the index of the topic that the query is most likely about. This is useful if the topic index is needed for other functions. It should NOT be used to find more detailed information on topics. Note that it is possible that the model does not find any topic that fits the query. In this case, the function returns None.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "query string. Can be a single word or a sentence. Used to find the index of the topic that is most likely about the query."
                            }
                        },
                        "required": ["query"]

                    }
                },
                "split_topic_kmeans": {
                    "name": "split_topic_kmeans",
                    "description": "This function can be used to split a topic into several subtopics using kmeans clustering. Only use this function to actually split topics. The subtopics do not need to be specified and are found automatically via clustering. It returns the topics the original topic has been split into.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic_idx": {
                                "type": "integer",
                                "description": "index of the topic to split."
                            },
                            "n_clusters": {
                                "type": "integer",
                                "description": "number of clusters to split the topic into. The more clusters, the more fine-grained the splitting. Typically 2 clusters are used.",
                                "default": 2
                            },
                            "inplace": {
                                "type": "boolean",
                                "description": "if True, the topic is split inplace. Otherwise, a new list of topics is created and returned. ALWAYS set inplace to False unless something else is explicitly requested!",
                                "default": False
                            }
                        },
                        "required": ["topic_idx"]
                    }
                },
                "split_topic_keywords": {
                    "name": "split_topic_keywords",
                    "description": "This function can be used to split a topic into subtopics according to the keywords. I.e. a topic about 'machine learning' can be split into a topic about 'supervised learning' and a topic about 'unsupervised learning'. This is achieved by computing the cosine similarity between the keywords and the documents in the topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic_idx": {
                                "type": "integer",
                                "description": "index of the topic to split."
                            },
                            "keywords": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "minItems": 2,
                                "description": "keywords to form new subtopics to replace old topic. Needs to be a list of at least two keywords."
                            },
                            "inplace": {
                                "type": "boolean",
                                "description": "if True, the topic is split inplace. Otherwise, a new list of topics is created and returned. ALWAYS set inplace to False unless something else is explicitly requested!",
                                "default": False
                            }
                        },
                        "required": ["topic_idx", "keywords"]
                    }
                },
                "split_topic_single_keyword": {
                    "name": "split_topic_single_keyword",
                    "description": "This function can be used to split a topic into the main topic and an additional subtopic. I.e. a topic about 'machine learning' can be split into a topic about 'machine learning' and a topic about 'supervised learning.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic_idx": {
                                "type": "integer",
                                "description": "index of the topic to split."
                            },
                            "keyword": {
                                "type": "string",
                                "description": "keyword to form new subtopic besides old main topic. Needs to be a single keyword."
                            },
                            "inplace": {
                                "type": "boolean",
                                "description": "if True, the topic is split inplace. Otherwise, a new list of topics is created and returned. ALWAYS set inplace to False unless something else is explicitly requested!",
                                "default": False
                            }
                        },
                        "required": ["topic_idx", "keyword"]
                    }
                },
                "combine_topics": {
                    "name": "combine_topics",
                    "description": "This function can be used to combine several topics into one topic. It returns the newly formed topic and removes the old topics from the list of topics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic_idx_lis": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                },
                                "minItems": 2,
                                "description": "list of topic indices to combine."
                            },
                            "inplace": {
                                "type": "boolean",
                                "description": "if True, the topic is split inplace. Otherwise, a new list of topics is created and returned. ALWAYS set inplace to False unless something else is explicitly requested!",
                                "default": False
                            }
                        },
                        "required": ["topic_idx_lis"]
                    }
                },
                "add_new_topic_keyword": {
                    "name": "add_new_topic_keyword",
                    "description": "This function can be used to globally create a new topic based on a keyword. This is useful if the keyword is not contained in any of the topics. The new topic is created by finding the documents that are closest to the keyword and then taking away those documents from the other topics. Note that this method is computationally expensive and should only be used if splitting another topic is unavoidable.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {
                                "type": "string",
                                "description": "keyword to form new topic. Needs to be a single keyword."
                            },
                            "inplace": {
                                "type": "boolean",
                                "description": "if True, the topic is split inplace. Otherwise, a new list of topics is created and returned. ALWAYS set inplace to False unless something else is explicitly requested!",
                                "default": False
                            }

                        },
                        "required": ["keyword"]
                    }
                },
                "delete_topic": {
                    "name": "delete_topic",
                    "description": "This function can be used to delete a topic and assign the documents of this topic to the other topics based on centroid similarity. This is useful if the topic is not needed anymore. Note that this method is computationally expensive.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic_idx": {
                                "type": "integer",
                                "description": "index of the topic to delete."
                            },
                            "inplace": {
                                "type": "boolean",
                                "description": "if True, the topic is split inplace. Otherwise, a new list of topics is created and returned. ALWAYS set inplace to False unless something else is explicitly requested!",
                                "default": False
                            }

                        },
                        "required": ["topic_idx"]
                    }
                },
                "get_topic_information": {
                    "name": "get_topic_information",
                    "description": "This function can be used to get information about several topics. This function can be used to COMPARE topics or to get an overview over them. It returns a list of dictionaries containing the topic index and information about the topics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic_idx_lis": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                },
                                "minItems": 1,
                                "description": "list of topic indices to get information about."
                            }
                        },
                        "required": ["topic_idx_lis"]
                    }
                },
                "split_topic_hdbscan": {
                    "name": "split_topic_hdbscan",
                    "description": "This function can be used to split a topic into several subtopics using hdbscan clustering. This method should be used if the number of clusters to split the topic into is not known.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic_idx": {
                                "type": "integer",
                                "description": "index of the topic to split."
                            },
                            "min_cluster_size": {
                                "type": "integer",
                                "description": "minimum number of documents in a cluster. The higher the number, the more fine-grained the splitting.",
                                "default": 10
                            },
                            "inplace": {
                                "type": "boolean",
                                "description": "if True, the topic is split inplace. Otherwise, a new list of topics is created and returned. ALWAYS set inplace to False unless something else is explicitly requested!",
                                "default": False
                            }
                        },
                        "required": ["topic_idx"]
                    }
                }
        }

        self.functionNames2Functions = {
            "knn_search": self._knn_search_openai,
            "identify_topic_idx": self._identify_topic_idx_openai,
            "split_topic_kmeans": self._split_topics_kmeans_openai,
            "split_topic_keywords": self._split_topic_keywords_openai,
            "split_topic_single_keyword": self._split_topic_single_keyword_openai,
            "combine_topics": self._combine_topics_openai,
            "add_new_topic_keyword": self._add_new_topic_keyword_openai,
            "delete_topic": self._delete_topic_openai,
            "get_topic_information": self._get_topic_information_openai,
            "split_topic_hdbscan": self._split_topic_hdbscan_openai
        }

    def reindex_topics(self) -> None:
        """
        Reindexes the topics in self.topic_list to assign correct new indices.

        This method updates the indices of topics within the instance's topic list to ensure they are correctly ordered.

        Returns:
            None
        """

        for idx, topic in enumerate(self.topic_lis):
            topic.topic_idx = idx

    def reindex_topic_lis(self, topic_list: list[Topic]) -> list[Topic]:
        """
        Reindexes the topics in the provided topic list to assign correct new indices.

        This method updates the indices of topics within the given topic list to ensure they are correctly ordered.

        Args:
            topic_list (list[Topic]): The list of Topic objects to reindex.

        Returns:
            list[Topic]: The reindexed list of Topic objects.
        """

        for idx, topic in enumerate(topic_list):
            topic.topic_idx = idx
        return topic_list

    def show_topic_lis(self) -> str:
        """
        Returns a string representation of the list of topics.

        This method generates a human-readable string representation of the topics in the instance's topic list.

        Returns:
            str: A string containing the representation of the list of topics.
        """

        self.reindex_topics()
        res = ""
        for idx, topic in enumerate(self.topic_lis):
            res += str(topic)

        print(res)

    def get_topic_lis(self) -> list[Topic]:
        """
        Returns the list of topics stored in the instance.

        This method retrieves and returns the list of topics associated with the instance.

        Returns:
            list[Topic]: The list of Topic objects.
        """

        self.reindex_topics()
        return self.topic_lis

    def set_topic_lis(self, topic_list: list[Topic]) -> None:
        """
        Sets the list of topics for the instance.

        This method updates the list of topics associated with the instance to the provided list.

        Args:
            topic_list (list[Topic]): The list of Topic objects to set.

        Returns:
            None
        """

        self.topic_lis = topic_list
        self.reindex_topics()

    def knn_search(self, topic_index: int, query: str, k: int = 20, doc_cutoff_threshold: int = 1000) -> tuple[list[str], list[int]]:
        """
        Finds the k nearest neighbors of the query in the given topic based on cosine similarity in the original embedding space.

        Args:
            topic_index (int): Index of the topic to search within.
            query (str): Query string.
            k (int, optional): Number of neighbors to return (default is 20).
            doc_cutoff_threshold (int, optional): Maximum number of tokens per document. Afterwards, the document is cut off (default is 1000).

        Returns:
            tuple: A tuple containing two lists -
                - A list of top k documents (as strings).
                - A list of indices corresponding to the top k documents in the topic.
        """

        topic = self.topic_lis[topic_index]

        query_embedding = self.client.embeddings.create(input = [query], model = self.openai_embedding_model)["data"][0]["embedding"]

        query_similarities = topic.document_embeddings_hd @ query_embedding / (np.linalg.norm(topic.document_embeddings_hd, axis = 1) * np.linalg.norm(query_embedding))

        topk_doc_indices = np.argsort(query_similarities)[::-1][:k]
        topk_docs = [topic.documents[i] for i in topk_doc_indices]

        # cut off documents that are too long
        max_number_tokens = self.max_context_length_promting - len(tiktoken.encoding_for_model(self.openai_prompting_model).encode(self.basic_model_instruction + " " + self.corpus_instruction)) - 100
        n_tokens = 0
        for i, doc in enumerate(topk_docs):
            encoded_doc = tiktoken.encoding_for_model(self.openai_prompting_model).encode(doc)
            n_tokens += len(encoded_doc[:doc_cutoff_threshold])
            if n_tokens > max_number_tokens:
                topk_docs = topk_docs[:i]
                topk_doc_indices = topk_doc_indices[:i]
                break
            if len(encoded_doc) > doc_cutoff_threshold:
                encoded_doc = encoded_doc[:doc_cutoff_threshold]
                topk_docs[i] = tiktoken.encoding_for_model(self.openai_prompting_model).decode(encoded_doc)




        return topk_docs, [int(elem) for elem in topk_doc_indices]

    def prompt_knn_search(self, llm_query: str, topic_index: int = None, n_tries: int = 3) -> tuple[str, tuple[list[str], list[int]]]:
        """
        Uses the Language Model (LLM) to answer the llm_query based on the documents belonging to the topic.

        Args:
            llm_query (str): Query string for the Language Model (LLM).
            topic_index (int, optional): Index of the topic object. If None, the topic is inferred from the query.
            n_tries (int, optional): Number of tries to get a valid response from the LLM (default is 3).

        Returns:
            tuple: A tuple containing two elements -
                - A string representing the answer from the LLM.
                - A tuple containing two lists -
                    - A list of top k documents (as strings).
                    - A list of indices corresponding to the top k documents in the topic.
        """

        messages = [
            {
                "role": "system",
                "content": self.basic_model_instruction + " " + self.corpus_instruction
            },
            {
                "role": "user",
                "content": llm_query
            }
            ]
        for _ in range(n_tries):
            try: 
                response_message = self.client.chat.completions.create(model = self.openai_prompting_model,
                messages = messages,
                functions = [self.function_descriptions["knn_search"]],
                function_call = "auto")["choices"][0]["message"]

                # Step 2: check if GPT wanted to call a function
                function_call = response_message.get("function_call")
                if function_call is not None:
                    #print("GPT wants to the call the function: ", function_call)
                    # Step 3: call the function
                    # Note: the JSON response may not always be valid; be sure to handle errors

                    function_name = function_call["name"]
                    function_to_call = self.functionNames2Functions[function_name]
                    function_args = json.loads(function_call["arguments"])
                    if topic_index is not None:
                        function_args["topic_index"] = topic_index
                    function_response = function_to_call(**function_args)
                    function_response_json = function_response[0]
                    function_response_return_output = function_response[1]



                    # Step 4: send the info on the function call and function response to GPT
                    messages.append(response_message)  # extend conversation with assistant's reply


                    messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": function_response_json,
                        }
                    )  # extend conversation with function response

                    #print(messages)
                    second_response = self.client.chat.completions.create(model=self.openai_prompting_model,
                    messages=messages)  # get a new response from GPT where it can see the function response
            except (TypeError, ValueError, openai.APIError, openai.APIConnectionError) as error:
                print("Error occured: ", error)
                print("Trying again...")

            return second_response, function_response_return_output

    def identify_topic_idx(self, query: str, n_tries: int = 3) -> int:
        """
        Identifies the index of the topic that the query is most likely about.

        This method uses a Language Model (LLM) to determine which topic best fits the query description. If the LLM does not find any topic that fits the query, None is returned.

        Args:
            query (str): Query string.
            n_tries (int, optional): Number of tries to get a valid response from the LLM (default is 3).

        Returns:
            int: The index of the topic that the query is most likely about. If no suitable topic is found, None is returned.
        """


        topic_descriptions_str = ""
        for i, topic in enumerate(self.topic_lis):
            description = topic.topic_description
            description = f"""Topic index: {i}: \n {description} \n \n"""
            topic_descriptions_str += description

        system_prompt = f"""You are a helpful assistant."""

        user_prompt = f""" Please find the index of the topic that is about the following query: {query}. 
        Those are the given topics: '''{topic_descriptions_str}'''.
        Please make sure to reply ONLY with an integer number between 0 and {len(self.topic_lis) - 1}!
        Reply with -1 if you don't find any topic that fits the query!
        Always explicitly say if you don't find any useful information by replying with -1! If in doubt, say that you did not find any useful information!
        Reply in the following format: "The topic index is: <index>"""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
            ]
        for _ in range(n_tries):
            try: 
                response_message = self.client.chat.completions.create(model = self.openai_prompting_model,
                messages = messages)["choices"][0]["message"]

            except (TypeError, ValueError, openai.APIError, openai.APIConnectionError) as error:
                print("Error occured: ", error)
                print("Trying again...")



        response_text = response_message["content"]
        # find integer number in response text
        try:
            match = re.search(r'(-?\d+)', response_text)
            topic_index = int(match.group(1))
        except:  # in case the LLM does not find any topic that fits the query, return None
            topic_index = None


        if topic_index is None:
            raise ValueError("No integer number found in response text! The model gave the following response: ", response_text)

        if topic_index == -1: 
            return None
        else:
            return topic_index

    def split_topic_new_assignments(self, topic_idx: int, new_topic_assignments: np.ndarray, inplace: bool = False) -> list[Topic]:
        """
        Splits a topic into new topics based on new topic assignments.

        Note that this method only computes topwords based on the cosine-similarity method because tf-idf topwords need expensive computation on the entire corpus. 
        The topwords of the old topic are also just split among the new ones. No new topwords are computed in this step.

        Args:
            topic_idx (int): Index of the topic to split.
            new_topic_assignments (np.ndarray): New topic assignments for the documents in the topic.
            inplace (bool, optional): If True, the topic is split in place. Otherwise, a new list of topics is created and returned (default is False).

        Returns:
            list of Topic: A list of new topics resulting from the split.
        """


        if self.vocab_embeddings is None:
            raise(ValueError("Need to provide vocab_embeddings to Topic prompting class to split a topic!"))
        if self.enhancer is None:
            raise(ValueError("Need to provide enhancer to Topic prompting class to split a topic!"))

        vocab_embedding_dict = self.vocab_embeddings
        enhancer = self.enhancer

        old_topic = self.topic_lis[topic_idx]

        assert len(new_topic_assignments) == len(old_topic.documents), "new_topic_assignments must have the same length as the number of documents in the topic!"

        # create new topics
        new_topics = []
        for i in np.unique(new_topic_assignments):
            docs = [old_topic.documents[j] for j in range(len(old_topic.documents)) if new_topic_assignments[j] == i]
            docs_embeddings = old_topic.document_embeddings_hd[new_topic_assignments == i]
            words_raw = []
            for doc in docs:
                words_raw += doc.split(" ")
            words_raw = set(words_raw)
            words = [word for word in old_topic.words if word in words_raw]

            new_topic = extract_and_describe_topic_cos_sim(
                documents_topic = docs,
                document_embeddings_topic = docs_embeddings,
                words_topic = words,
                vocab_embeddings = vocab_embedding_dict,
                umap_mapper = old_topic.umap_mapper,
                enhancer=enhancer,
                n_topwords = 2000
            )
            new_topic.topic_idx = len(self.topic_lis) + i + 1
            new_topics.append(new_topic)

        new_topic_lis = self.topic_lis.copy()
        new_topic_lis.pop(topic_idx)
        new_topic_lis += new_topics
        new_topic_lis = self.reindex_topic_lis(new_topic_lis)

        if inplace:
            self.topic_lis = new_topic_lis

        return new_topic_lis

    def split_topic_kmeans(self, topic_idx: int, n_clusters: int = 2, inplace: bool = False) -> list[Topic]:
        """
        Splits an existing topic into several subtopics using k-means clustering on the document embeddings of the topic.

        Note that no new topwords are computed in this step, and the topwords of the old topic are just split among the new ones. Additionally, only the cosine-similarity method for topwords extraction is used.

        Args:
            topic_idx (int): Index of the topic to split.
            n_clusters (int, optional): Number of clusters to split the topic into (default is 2).
            inplace (bool, optional): If True, the topic is split in place. Otherwise, a new list of topics is created and returned (default is False).

        Returns:
            list of Topic: A list of new topics resulting from the split.
        """


        old_topic = self.topic_lis[topic_idx]
        embeddings = old_topic.document_embeddings_ld  # embeddings to split into clusters

        kmeans_res = sklearn.cluster.KMeans(n_clusters = n_clusters, random_state = self.random_state, n_init = "auto").fit(embeddings)
        cluster_labels = kmeans_res.labels_
        new_topics = self.split_topic_new_assignments(topic_idx, cluster_labels, inplace)

        return new_topics

    def split_topic_hdbscan(self, topic_idx: int, min_cluster_size: int = 100, inplace: bool = False) -> list[Topic]:
        """
        Splits an existing topic into several subtopics using HDBSCAN clustering on the document embeddings of the topic.

        This method does not require specifying the number of clusters to split. Note that no new topwords are computed in this step, and the topwords of the old topic are just split among the new ones. Additionally, only the cosine-similarity method for topwords extraction is used.

        Args:
            topic_idx (int): Index of the topic to split.
            min_cluster_size (int, optional): Minimum cluster size to split the topic into (default is 100).
            inplace (bool, optional): If True, the topic is split in place. Otherwise, a new list of topics is created and returned (default is False).

        Returns:
            list of Topic: A list of new topics resulting from the split.
        """


        old_topic = self.topic_lis[topic_idx]
        embeddings = old_topic.document_embeddings_ld

        clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, prediction_data = True)
        clusterer.fit(embeddings)
        cluster_labels = clusterer.labels_
        new_topics = self.split_topic_new_assignments(topic_idx, cluster_labels, inplace)

        new_topics = self.reindex_topic_lis(new_topics)

        if inplace:
            self.topic_lis = new_topics

        return new_topics

    def split_topic_keywords(self, topic_idx: int, keywords: str, inplace: bool = False) -> list[Topic]:
        """
        Splits the topic into subtopics according to the provided keywords.

        This is achieved by computing the cosine similarity between the keywords and the documents in the topic. Note that no new topwords are computed in this step, and the topwords of the old topic are just split among the new ones. Additionally, only the cosine-similarity method for topwords extraction is used.

        Args:
            topic_idx (int): Index of the topic to split.
            keywords (str): Keywords to split the topic into. Needs to be a list of at least two keywords.
            inplace (bool, optional): If True, the topic is split in place. Otherwise, a new list of topics is created and returned (default is False).

        Returns:
            list of Topic: A list of new topics resulting from the split.
        """

        assert len(keywords) > 1, "Need at least two keywords to split the topic! Otherwise use the split_topic_single_keyword function!"
        keyword_embeddings = []
        for keyword in keywords:
            keyword_embeddings.append(self.client.embeddings.create(input = [keyword], model = self.openai_embedding_model)["data"][0]["embedding"])
        keyword_embeddings = np.array(keyword_embeddings)

        old_topic = self.topic_lis[topic_idx]
        document_embeddings = old_topic.document_embeddings_hd

        document_embeddings = document_embeddings / np.linalg.norm(document_embeddings, axis = 1)[:, np.newaxis]
        keyword_embeddings = keyword_embeddings / np.linalg.norm(keyword_embeddings, axis = 1)[:, np.newaxis]
        similarities = document_embeddings @ keyword_embeddings.T
        new_topic_assignments = np.argmax(similarities, axis = 1)

        # if the topic cannot be split, i.e. all documents are assigned the same label, raise an error
        if len(np.unique(new_topic_assignments)) == 1:
            raise ValueError(f"The topic cannot be split into the subtopics {keywords}. All documents are assigned the same label!")

        new_topics = self.split_topic_new_assignments(topic_idx, new_topic_assignments, inplace = inplace)

        new_topics = self.reindex_topic_lis(new_topics)

        if inplace:
            self.topic_lis = new_topics

        return new_topics

    def split_topic_single_keyword(self, topic_idx: int, keyword: str, inplace: bool = False) -> list[Topic]:
        """
        Splits the topic with a single keyword.

        This method splits the topic such that all documents closer to the original topic name stay in the old topic, while all documents closer to the keyword are moved to the new topic. Note that no new topwords are computed in this step, and the topwords of the old topic are just split among the new ones. Additionally, only the cosine-similarity method for topwords extraction is used.

        Args:
            topic_idx (int): Index of the topic to split.
            keyword (str): Keyword to split the topic into.
            inplace (bool, optional): If True, the topic is split in place. Otherwise, a new list of topics is created and returned (default is False).

        Returns:
            list of Topic: A list of new topics resulting from the split.
        """

        keywords = [self.topic_lis[topic_idx].topic_name, keyword]

        res = self.split_topic_keywords(topic_idx, keywords, inplace)

        return res

    def combine_topics(self, topic_idx_lis: list[int], inplace: bool = False) -> list[Topic]:
        """
        Combines several topics into one topic.

        This method combines the specified topics into a single topic. Note that no new topwords are computed in this step, and the topwords of the old topics are just combined. Additionally, only the cosine-similarity method for topwords extraction is used.

        Args:
            topic_idx_list (list[int]): List of topic indices to combine.
            inplace (bool, optional): If True, the topics are combined in place. Otherwise, a new list of topics is created and returned (default is False).

        Returns:
            list of Topic: A list of new topics resulting from the combination.
        """

        new_topic_docs = []
        new_topic_words = []
        new_topic_document_embeddings_hd = []

        for topic_idx in topic_idx_lis:
            topic = self.topic_lis[topic_idx]
            new_topic_docs += topic.documents
            new_topic_words += topic.words
            new_topic_document_embeddings_hd.append(topic.document_embeddings_hd)

        new_topic_document_embeddings_hd = np.concatenate(new_topic_document_embeddings_hd, axis = 0)

        new_topic = extract_and_describe_topic_cos_sim(
            documents_topic = new_topic_docs,
            document_embeddings_topic = new_topic_document_embeddings_hd,
            words_topic = new_topic_words,
            vocab_embeddings = self.vocab_embeddings,
            umap_mapper = self.topic_lis[0].umap_mapper,
            enhancer=self.enhancer,
            n_topwords = 2000
        )

        new_topic.topic_idx = len(self.topic_lis) + 1
        new_topic_lis = self.topic_lis.copy()

        for topic_idx in sorted(topic_idx_lis, reverse = True):
            new_topic_lis.pop(topic_idx)
        new_topic_lis.append(new_topic)
        new_topic_lis = self.reindex_topic_lis(new_topic_lis)


        if inplace:
            self.topic_lis = new_topic_lis
            self.reindex_topics()

        return new_topic_lis

    def add_new_topic_keyword(self, keyword: str, inplace: bool = False, rename_new_topic: bool = False) -> list[Topic]:
        """
        Create a new topic based on a keyword and recompute topic topwords.

        This method removes all documents belonging to other topics from them and adds them to the new topic. It computes new topwords using both the tf-idf and the cosine-similarity method.

        Args:
            keyword (str): Keyword to create the new topic from.
            inplace (bool, optional): If True, the topic is updated in place. Otherwise, a new list of topics is created and returned (default is False).
            rename_new_topic (bool, optional): If True, the new topic is renamed to the keyword (default is False).

        Returns:
            list of Topic: A list of new topics, including the newly created topic and the modified old ones.
        """

        umap_mapper = self.topic_lis[0].umap_mapper

        keyword_embedding_hd = self.client.embeddings.create(input = [keyword], model = self.openai_embedding_model)["data"][0]["embedding"]
        keyword_embedding_hd = np.array(keyword_embedding_hd).reshape(1, -1)
        keyword_embedding_ld = umap_mapper.transform(keyword_embedding_hd)[0]

        old_centroids_ld = []
        for topic in self.topic_lis:
            old_centroids_ld.append(topic.centroid_ld)
        old_centroids_ld = np.array(old_centroids_ld)

        # assign documents to new centroid (keyword_embedding_ld) iff they are closer to the new centroid than to their old centroid

        new_doc_topic_assignments = []
        doc_lis = []

        new_topic_idx = len(self.topic_lis)
        for i, topic in enumerate(self.topic_lis):
            doc_lis += topic.documents
            document_embeddings = topic.document_embeddings_ld
            cos_sim_old_centroid = document_embeddings @ old_centroids_ld[i] / (np.linalg.norm(document_embeddings, axis = 1) * np.linalg.norm(old_centroids_ld[i]))
            cos_sim_new_centroid = document_embeddings @ keyword_embedding_ld / (np.linalg.norm(document_embeddings, axis = 1) * np.linalg.norm(keyword_embedding_ld))
            new_centroid_is_closer = cos_sim_new_centroid > cos_sim_old_centroid

            new_document_assignments = np.where(new_centroid_is_closer, new_topic_idx, i)
            new_doc_topic_assignments.append(new_document_assignments)

        new_doc_topic_assignments = np.concatenate(new_doc_topic_assignments, axis = 0)

        assert len(doc_lis) == len(new_doc_topic_assignments), "Number of documents must be equal to the number of document assignments!"

        new_embeddings_hd = []
        new_embeddings_ld = []

        for topic in self.topic_lis:
            new_embeddings_hd.append(topic.document_embeddings_hd)
            new_embeddings_ld.append(topic.document_embeddings_ld)

        new_embeddings_hd = np.concatenate(new_embeddings_hd, axis = 0)
        new_embeddings_ld = np.concatenate(new_embeddings_ld, axis = 0)

        new_topics = extract_describe_topics_labels_vocab(
            corpus = doc_lis,
            document_embeddings_hd = new_embeddings_hd,
            document_embeddings_ld = new_embeddings_ld,
            labels = new_doc_topic_assignments,
            vocab = self.vocab,
            umap_mapper = umap_mapper,
            vocab_embeddings = self.vocab_embeddings, 
            enhancer = self.enhancer
        )

        if rename_new_topic:
            new_topics[-1].topic_name = keyword

        new_topics = self.reindex_topic_lis(new_topics)

        if inplace:
            self.topic_lis = new_topics

        return new_topics

    def delete_topic(self, topic_idx: int, inplace: bool = False) -> list[Topic]:
        """
        Deletes a topic with the given index from the list of topics and recomputes topwords and representations of the remaining topics.

        This method assigns the documents of the deleted topic to the remaining topics.

        Args:
            topic_idx (int): Index of the topic to delete.
            inplace (bool, optional): If True, the topic is deleted in place. Otherwise, a new list of topics is created and returned (default is False).

        Returns:
            list of Topic: A list of new topics resulting from the deletion.
        """


        topic_lis_new = deepcopy(self.topic_lis)
        topic_lis_new.pop(topic_idx)

        old_centroids_ld = []
        for topic in topic_lis_new:
            old_centroids_ld.append(topic.centroid_ld)

        old_centroids_ld = np.array(old_centroids_ld)

        document_embeddings_ld = []

        for topic in self.topic_lis:
            document_embeddings_ld.append(topic.document_embeddings_ld)

        document_embeddings_ld = np.concatenate(document_embeddings_ld, axis = 0) # has shape (n_documents, n_topics)

        centroid_similarities = document_embeddings_ld @ old_centroids_ld.T / (np.linalg.norm(document_embeddings_ld, axis = 1)[:, np.newaxis] * np.linalg.norm(old_centroids_ld, axis = 1))
        new_topic_assignments = np.argmax(centroid_similarities, axis = 1)

        new_embeddings_hd = []
        new_embeddings_ld = []

        for topic in self.topic_lis:
            new_embeddings_hd.append(topic.document_embeddings_hd)
            new_embeddings_ld.append(topic.document_embeddings_ld)

        new_embeddings_hd = np.concatenate(new_embeddings_hd, axis = 0)
        new_embeddings_ld = np.concatenate(new_embeddings_ld, axis = 0)

        doc_lis = []
        for topic in self.topic_lis:
            doc_lis += topic.documents



        new_topics = extract_describe_topics_labels_vocab(
            corpus = doc_lis,
            document_embeddings_hd = new_embeddings_hd,
            document_embeddings_ld = new_embeddings_ld,
            labels = new_topic_assignments,
            vocab = self.vocab,
            umap_mapper = self.topic_lis[0].umap_mapper,
            vocab_embeddings = self.vocab_embeddings,
            enhancer = self.enhancer
        )

        new_topics = self.reindex_topic_lis(new_topics)

        if inplace:
            self.topic_lis = new_topics

        return new_topics

    def get_topic_information(self, topic_idx_lis: list[int], max_number_topwords: int = 500) -> dict:
        """
        Get detailed information on topics by their indices.

        This function returns a dictionary where the keys are the topic indices, and the values are strings describing the topics. The description includes a maximum of max_number_topwords topwords.

        Args:
            topic_idx_list (list[int]): List of topic indices to compare.
            max_number_topwords (int, optional): Maximum number of topwords to include in the description of the topics (default is 500).

        Returns:
            dict: A dictionary with topic indices as keys and their descriptions as values.
        """

        max_number_tokens = self.max_context_length_promting - len(tiktoken.encoding_for_model(self.openai_prompting_model).encode(self.basic_model_instruction + " " + self.corpus_instruction)) - 100

        topic_info = {} # dictionary with the topic indices as keys and the topic descriptions as values

        for topic_idx in topic_idx_lis:
            topic = self.topic_lis[topic_idx]
            topic_info[topic_idx] = topic.topic_description

            topic_str = f"""
            Topic index: {topic_idx}
            Topic name: {topic.topic_name}
            Topic description: {topic.topic_description}
            Topic topwords: {topic.top_words["cosine_similarity"][:max_number_topwords]}"""

            topic_info[topic_idx] = topic_str

        # prune all topic descriptions to the maximum number of tokens by taking away the last word until the description fits

        max_number_tokens_per_topic = max_number_tokens // len(topic_idx_lis)
        tiktoken_encodings = {idx: tiktoken.encoding_for_model(self.openai_prompting_model).encode(topic_info[idx]) for idx in topic_idx_lis}
        pruned_encodings = {idx: tiktoken_encodings[idx][:max_number_tokens_per_topic] for idx in topic_idx_lis}

        topic_info = {idx: tiktoken.encoding_for_model(self.openai_prompting_model).decode(pruned_encodings[idx]) for idx in topic_idx_lis}

        return topic_info

    def _knn_search_openai(self, topic_index: int, query: str, k: int = 20) -> tuple[str, (list[str], list[int])]:
        """
        A version of the knn_search function that returns a JSON file to be used with the OpenAI API.

        Args:
            topic_index (int): Index of the topic to search in.
            query (str): Query string.
            k (int, optional): Number of neighbors to return (default is 20).

        Returns:
            json: JSON object to be used with the OpenAI API.
            tuple: A tuple containing two lists -
                - A list of top k documents (as strings).
                - A list of indices corresponding to the top k documents in the topic.
        """

        topk_docs, topk_doc_indices = self.knn_search(topic_index, query, k)
        json_obj = json.dumps({
            "top-k documents": topk_docs,
            "indices of top-k documents": list(topk_doc_indices)
        })
        return json_obj, (topk_docs, topk_doc_indices)

    def _identify_topic_idx_openai(self, query: str, n_tries: int = 3) -> tuple[str, int]:
        """
        A version of the identify_topic_idx function that returns a JSON file to be used with the OpenAI API.

        Args:
            query (str): Query string.
            n_tries (int, optional): Number of tries to get a valid response from the LLM (default is 3).

        Returns:
            json: JSON object to be used with the OpenAI API.
            int: The topic index.
        """

        topic_index = self.identify_topic_idx(query, n_tries)
        json_obj = json.dumps({
            "topic index": topic_index
        })
        return json_obj, topic_index

    def _split_topic_hdbscan_openai(self, topic_idx: int, min_cluster_size: int = 10, inplace: bool = False) -> tuple[str, list[Topic]]:
        """
        A version of the split_topic_hdbscan function that returns a JSON file to be used with the OpenAI API.

        Args:
            topic_idx (int): Index of the topic to split.
            min_cluster_size (int, optional): Minimum cluster size to split the topic into (default is 10).
            inplace (bool, optional): If True, the topic is split in place. Otherwise, a new list of topics is created and returned (default is False).

        Returns:
            json: JSON object to be used with the OpenAI API.
            list of Topic: A list of new topics resulting from the split.
        """

        new_topics = self.split_topic_hdbscan(topic_idx, min_cluster_size, inplace)
        json_obj = json.dumps({
            "new topics": [topic.to_dict() for topic in new_topics][-len(new_topics):]
        })
        return json_obj, new_topics

    def _split_topics_kmeans_openai(self, topic_idx: list[int], n_clusters: int = 2, inplace: bool = False) -> tuple[str, list[Topic]]:
        """
        A version of the split_topic_kmeans function that returns a JSON file to be used with the OpenAI API.

        Args:
            topic_idx (list[int]): List of indices of the topics to split.
            n_clusters (int, optional): Number of clusters to split each topic into (default is 2).
            inplace (bool, optional): If True, the topics are split in place. Otherwise, new lists of topics are created and returned (default is False).

        Returns:
            json: JSON object to be used with the OpenAI API.
            list of Topic: A list of new topics resulting from the split.
        """

        new_topics = self.split_topic_kmeans(topic_idx, n_clusters, inplace)
        json_obj = json.dumps({
            "new topics": [topic.to_dict() for topic in new_topics][-n_clusters:]
        })
        return json_obj, new_topics

    def _split_topic_keywords_openai(self, topic_idx: int, keywords: str, inplace: bool = False) -> tuple[str, list[Topic]]:
        """
        A version of the split_topic_keywords function that returns a JSON file to be used with the OpenAI API.

        Args:
            topic_idx (int): Index of the topic to split.
            keywords (str): Keywords to split the topic into. Needs to be a list of at least two keywords.
            inplace (bool, optional): If True, the topic is split in place. Otherwise, a new list of topics is created and returned (default is False).

        Returns:
            json: JSON object to be used with the OpenAI API.
            list of Topic: A list of new topics resulting from the split.
        """

        new_topics = self.split_topic_keywords(topic_idx, keywords, inplace)
        json_obj = json.dumps({
            "new topics": [topic.to_dict() for topic in new_topics][-len(keywords):]
        })
        return json_obj, new_topics

    def _split_topic_single_keyword_openai(self, topic_idx: int, keyword: str, inplace: bool = False) -> tuple[str, list[Topic]]:
        """
        A version of the split_topic_single_keyword function that returns a JSON file to be used with the OpenAI API.

        Args:
            topic_idx (int): Index of the topic to split.
            keyword (str): Keyword to split the topic into.
            inplace (bool, optional): If True, the topic is split in place. Otherwise, a new list of topics is created and returned (default is False).

        Returns:
            json: JSON object to be used with the OpenAI API.
            list of Topic: A list of new topics resulting from the split.
        """

        new_topics = self.split_topic_single_keyword(topic_idx, keyword, inplace)
        json_obj = json.dumps({
            "new topics": [topic.to_dict() for topic in new_topics][-2:]
        })
        return json_obj, new_topics

    def _combine_topics_openai(self, topic_idx_lis: list[int], inplace: bool = False) -> tuple[str, list[Topic]]:
        """
        A version of the combine_topics function that returns a JSON file to be used with the OpenAI API.

        Args:
            topic_idx_lis (list[int]): List of topic indices to combine.
            inplace (bool, optional): If True, the topics are combined in place. Otherwise, a new list of topics is created and returned (default is False).

        Returns:
            json: JSON object to be used with the OpenAI API.
            list of Topic: A list of new topics resulting from the combination.
        """

        new_topics = self.combine_topics(topic_idx_lis, inplace)
        json_obj = json.dumps({
            "new topics": [topic.to_dict() for topic in new_topics][-1]
        })
        return json_obj, new_topics

    def _add_new_topic_keyword_openai(self, keyword: str, inplace: bool = False, rename_new_topic: bool = False) -> tuple[str, list[Topic]]:
        """
        A version of the add_new_topic_keyword function that returns a JSON file to be used with the OpenAI API.

        Args:
            keyword (str): Keyword to create the new topic from.
            inplace (bool, optional): If True, the topic is split in place. Otherwise, a new list of topics is created and returned (default is False).
            rename_new_topic (bool, optional): If True, the new topic is renamed to the keyword (default is False).

        Returns:
            json: JSON object to be used with the OpenAI API.
            list of Topic: A list of new topics resulting from the operation.
        """

        new_topics = self.add_new_topic_keyword(keyword, inplace, rename_new_topic)
        json_obj = json.dumps({
            "new topics": [topic.to_dict() for topic in new_topics][-1]
        })
        return json_obj, new_topics

    def _delete_topic_openai(self, topic_idx: int, inplace: bool = False) -> tuple[str, list[Topic]]:
        """
        A version of the delete_topic function that returns a JSON file to be used with the OpenAI API.

        Args:
            topic_idx (int): Index of the topic to delete.
            inplace (bool, optional): If True, the topic is deleted in place. Otherwise, a new list of topics is created and returned (default is False).

        Returns:
            json: JSON object to be used with the OpenAI API.
            list of Topic: A list of topics after the deletion operation.
        """

        new_topics = self.delete_topic(topic_idx, inplace)
        json_obj = json.dumps({
            f"Topics after deleting the one with index {topic_idx}": [topic.to_dict() for topic in new_topics]
        })
        return json_obj, new_topics

    def _get_topic_information_openai(self, topic_idx_lis: list[int]) -> tuple[str, dict]:
        """
        A version of the get_topic_information function that returns a JSON file suitable for use with the OpenAI API.

        Args:
            topic_idx_lis (list[int]): List of topic indices to compare.

        Returns:
            json: JSON object to be used with the OpenAI API.
            dict: A dictionary containing detailed information about the specified topics.
        """

        topic_info = self.get_topic_information(topic_idx_lis)
        json_obj = json.dumps({
            "topic info": topic_info
        })
        return json_obj, topic_info

    def _fix_dictionary_topwords(self):
        """
        Fix an issue with the topic representation where the topwords are nested within another dictionary in the actual dictionary defining them.
        """

        for topic in self.topic_lis:
            if type(topic.top_words["cosine_similarity"]) == dict:
                topic.top_words["cosine_similarity"] = topic.top_words["cosine_similarity"][0]

    def general_prompt(self, prompt: str, n_tries: int = 2) -> tuple[list[str], object]:
        """
        Prompt the Language Model (LLM) with a general prompt and return the response. Allow the LLM to call any function defined in the class.

        Use n_tries in case the LLM does not provide a valid response.

        Args:
            prompt (str): Prompt string.
            n_tries (int, optional): Number of tries to get a valid response from the LLM (default is 2).

        Returns:
            list of str: Response messages from the LLM.
            object: Response of the invoked function.
        """

        messages = [
            {
                "role": "system",
                "content": self.basic_model_instruction + " " + self.corpus_instruction
            },
            {
                "role": "user",
                "content": prompt
            }
            ]

        functions = [self.function_descriptions[key] for key in self.function_descriptions.keys()]
        for _ in range(n_tries):
            try: 
                response_message = self.client.chat.completions.create(model = self.openai_prompting_model,
                messages = messages,
                functions = functions,
                function_call = "auto").choices[0].message

                # Step 2: check if GPT wanted to call a function
                function_call = response_message.function_call
                if function_call is not None:
                    print("GPT wants to the call the function: ", function_call)
                    # Step 3: call the function
                    # Note: the JSON response may not always be valid; be sure to handle errors

                    function_name = function_call.name
                    function_to_call = self.functionNames2Functions[function_name]
                    function_args = json.loads(function_call.arguments)
                    function_response = function_to_call(**function_args)
                    function_response_json = function_response[0]
                    function_response_return_output = function_response[1]

                    # Step 4: send the info on the function call and function response to GPT
                    messages.append(response_message)  # extend conversation with assistant's reply

                    messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": function_response_json,
                        }
                    )  # extend conversation with function response

                    second_response = self.client.chat.completions.create(model=self.openai_prompting_model,
                    messages=messages)  # get a new response from GPT where it can see the function response
            except (TypeError, ValueError, openai.APIError, openai.APIConnectionError) as error:
                print("Error occured: ", error)
                print("Trying again...")

        return [response_message, second_response], function_response_return_output