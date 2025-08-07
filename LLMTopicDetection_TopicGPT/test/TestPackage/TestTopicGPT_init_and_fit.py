"""
This class tests the init and fit functions of the TopicGPT module. 
"""

import os 
import sys
import inspect
import openai
import pickle

import unittest

from topicgpt.TopicRepresentation import Topic

from topicgpt.Clustering import Clustering_and_DimRed
from topicgpt.TopwordEnhancement import TopwordEnhancement
from topicgpt.TopicPrompting import TopicPrompting
from topicgpt.TopicGPT import TopicGPT

class TestTopicGPT_init_and_fit(unittest.TestCase):
    """
    Test the init and fit functions of the TopicGPT class
    """

    @classmethod
    def setUpClass(cls, sample_size = 0.5):
        """
        load the necessary data and only keep a sample of it 
        """
        print("Setting up class...")
        cls.api_key_openai = os.environ.get('api_key')
        # TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization=os.environ.get('OPENAI_ORG'))'
        # openai.organization = os.environ.get('OPENAI_ORG')

        with open("../../Data/Emebeddings/embeddings_20ng_raw.pkl", "rb")  as f:
            data_raw = pickle.load(f)

        corpus = data_raw["corpus"]
        doc_embeddings = data_raw["embeddings"]

        n_docs = int(len(corpus) * sample_size)
        cls.corpus = corpus[:n_docs]
        cls.doc_embeddings = doc_embeddings[:n_docs]

        print("Using {} out of {} documents".format(n_docs, len(data_raw["corpus"])))

        with open("../../Data/Emebeddings/embeddings_20ng_vocab.pkl", "rb") as f:
            cls.embeddings_vocab = pickle.load(f)

    def test_init(self):
        """
        test the init function of the TopicGPT class
        """
        print("Testing init...")
        topicgpt = TopicGPT(api_key = self.api_key_openai)
        self.assertTrue(isinstance(topicgpt, TopicGPT))

        topicgpt = TopicGPT(api_key = self.api_key_openai, 
                            n_topics= 20)
        self.assertTrue(isinstance(topicgpt, TopicGPT))

        topicgpt = TopicGPT(api_key = self.api_key_openai, 
                            n_topics= 20,
                            corpus_instruction="This is a corpus instruction", 
                            document_embeddings = self.doc_embeddings,
                            vocab_embeddings= self.embeddings_vocab)
        self.assertTrue(isinstance(topicgpt, TopicGPT))

        # check if assertions are triggered

        with self.assertRaises(AssertionError):
            topicgpt = TopicGPT(api_key = None, 
                                n_topics= 32,
                                openai_prompting_model="gpt-4",
                                max_number_of_tokens=8000,
                                corpus_instruction="This is a corpus instruction")

        with self.assertRaises(AssertionError):
            topicgpt = TopicGPT(api_key = self.api_key_openai, 
                                n_topics= 0,
                                max_number_of_tokens=8000,
                                corpus_instruction="This is a corpus instruction")

        with self.assertRaises(AssertionError):
            topicgpt = TopicGPT(api_key = self.api_key_openai, 
                                n_topics= 20,
                                max_number_of_tokens=0,
                                corpus_instruction="This is a corpus instruction")

    def test_fit(self):
        """
        test the fit function of the TopicGPT class
        """
        print("Testing fit...")

        def instance_test(topicgpt):
            topicgpt.fit(self.corpus)

            self.assertTrue(hasattr(topicgpt, "vocab"))
            self.assertTrue(hasattr(topicgpt, "topic_lis"))

            self.assertTrue(isinstance(topicgpt.vocab, list))
            self.assertTrue(isinstance(topicgpt.vocab[0], str))

            self.assertTrue(isinstance(topicgpt.topic_lis, list))
            self.assertTrue(type(topicgpt.topic_lis[0]) == Topic)

            if topicgpt.n_topics is not None:
                self.assertTrue(len(topicgpt.topic_lis) == topicgpt.n_topics)

            self.assertTrue(topicgpt.topic_lis == topicgpt.topic_prompting.topic_lis)
            self.assertTrue(topicgpt.vocab == topicgpt.topic_prompting.vocab)
            self.assertTrue(topicgpt.vocab_embeddings == topicgpt.topic_prompting.vocab_embeddings)


        topicgpt1 = TopicGPT(api_key = self.api_key_openai, 
                            n_topics= 20,
                            document_embeddings = self.doc_embeddings,
                            vocab_embeddings = self.embeddings_vocab)

        topicgpt2 = TopicGPT(api_key = self.api_key_openai,
                             n_topics= None,
                                document_embeddings = self.doc_embeddings, 
                                vocab_embeddings = self.embeddings_vocab)

        topicgpt3 = TopicGPT(api_key=self.api_key_openai, 
                              n_topics = 1,
                                document_embeddings = self.doc_embeddings,
                                vocab_embeddings = self.embeddings_vocab,
                                n_topwords=10,
                                n_topwords_description=10,
                                topword_extraction_methods=["cosine_similarity"])

        clusterer4 = Clustering_and_DimRed(
            n_dims_umap = 10,
            n_neighbors_umap = 20,
            min_cluster_size_hdbscan = 10,
            number_clusters_hdbscan= 10 # use only 10 clusters
        )

        topword_enhancement4 = TopwordEnhancement(api_key = self.api_key_openai)
        topic_prompting4 = TopicPrompting(
            api_key = self.api_key_openai,
            enhancer = topword_enhancement4,
            topic_lis = None
        )

        topicgpt4 = TopicGPT(api_key=self.api_key_openai,
                                n_topics= None,
                                    document_embeddings = self.doc_embeddings, 
                                    vocab_embeddings = self.embeddings_vocab,
                                    topic_prompting = topic_prompting4,
                                    clusterer = clusterer4,
                                    topword_extraction_methods=["tfidf"])


        topic_gpt_list = [topicgpt1, topicgpt2, topicgpt3, topicgpt4]

        for topic_gpt in topic_gpt_list:
            instance_test(topic_gpt)




if __name__ == "__main__":
    unittest.main()