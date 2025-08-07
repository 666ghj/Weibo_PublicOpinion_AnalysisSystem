from topicgpt.TopicRepresentation import Topic

import unittest
from sklearn.datasets import fetch_20newsgroups 

from topicgpt.TopicGPT import TopicGPT

    
import sys


class QuickestTopicGPT_prompting(unittest.TestCase):
    """
    This class is used to mainly test the prompting functionality of the TopicGPT class.
    """


    @classmethod
    def setUpClass(cls, sample_size:int = 500):
        """
        download the necessary data and only keep a sample of it 
        params: 
            client: Client.
            sample_size: the number of documents to use for the test
        """

        data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes')) #download the 20 Newsgroups dataset
        corpus = data['data']# just select the first 1000 documents for this example
        corpus = [doc for doc in corpus if doc != ""]
        corpus = corpus[:sample_size]

        cls.corpus = corpus

        cls.tm = TopicGPT(client = client, n_topics = 1)
        cls.tm.fit(cls.corpus)

    def test_repr_topics(self):
        """
        test the repr_topics function of the TopicGPT class
        """
        print("Testing repr_topics...")
        self.assertTrue(type(self.tm.repr_topics()) == str)

    def test_promt_knn_search(self):
        """
        test the ppromt function that calls knn_search of the TopicPrompting class
        """
        print("Testing ppromt_knn_search...")
        
        prompt_lis = ["Is topic 0 about Bananas? Use knn Search",
                      "Is topic 0 about Space? Use knn Search"]
        
        for prompt in prompt_lis:

            answer, function_result = self.tm.prompt(prompt)

            print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")

            self.assertTrue(type(answer) == str)
            self.assertTrue(type(function_result[0]) == list)
            self.assertTrue(type(function_result[1]) == list)
            self.assertTrue(type(function_result[0][0]) == str)
            self.assertTrue(type(function_result[1][0]) == int)


    def test_prompt_split_topic_kmeans_inplace(self):
        """
        test the ppromt function that calls split_topic_kmeans of the TopicPrompting class
        """

        print("Testing ppromt_split_topic_kmeans...")

        prompt_lis = ["Split topic 0 into 2 subtopics using kmeans. Do this inplace"]
        added_topic_lis_len  = [2]

        old_number_of_topics = len(self.tm.topic_lis)

        for prompt, added_topic_len in zip(prompt_lis, added_topic_lis_len):
                
                answer, function_result = self.tm.prompt(prompt)
    
                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)
    
                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                self.assertTrue(type(function_result[0]) == Topic)

                self.assertTrue(len(self.tm.topic_lis) == old_number_of_topics + added_topic_len -1 )
                self.assertTrue(self.tm.topic_lis == function_result)

   
    def test_prompt_combine_topics_inplace(self):
        """
        test the prompt function that calls combine_topics of the TopicPrompting class
        """

        print("Testing ppromt_combine_topics...")

        prompt_lis = ["Combine topic 0 and topic 1 into one topic. Do this inplace"]

        # split topic first
        self.tm.prompt("Please split topic 0 into two subtopic. Do this inplace.")

        old_number_topics = len(self.tm.topic_lis)



        for prompt in prompt_lis:
                
                answer, function_result = self.tm.prompt(prompt)
    
                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)
                print("topic_gpt_topic_list: ", self.tm.topic_lis)
    
                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                self.assertTrue(type(function_result[0]) == Topic)
                self.assertTrue(self.tm.topic_lis == function_result)
                self.assertTrue(len(self.tm.topic_lis) == old_number_topics -1)


if __name__ == "__main__":
    
    for i, arg in enumerate(sys.argv):
        if arg == "--api-key":
            api_key = sys.argv.pop(i + 1)
            sys.argv.pop(i)
            break

    if api_key is None:
        print("API key must be provided with --api-key")
        sys.exit(1)

    
    unittest.main()