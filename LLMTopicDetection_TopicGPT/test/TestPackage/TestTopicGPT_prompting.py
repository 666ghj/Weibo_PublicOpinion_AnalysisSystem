"""
This class is used to mainly test the prompting functionality of the TopicGPT package. 
"""

import os 
import sys
import inspect

import openai
import pickle
import unittest

from topicgpt.TopicGPT import TopicGPT
from topicgpt.TopicRepresentation import Topic
from topicgpt.Clustering import Clustering_and_DimRed
from topicgpt.TopwordEnhancement import TopwordEnhancement
from topicgpt.TopicPrompting import TopicPrompting


# TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization=os.environ.get('OPENAI_ORG'))'
# openai.organization = os.environ.get('OPENAI_ORG')

class TestTopicGPT_prompting(unittest.TestCase):
    """
    This class is used to mainly test the prompting functionality of the TopicGPT class.
    """

    @classmethod
    def setUp(self):
        """
        load the necessary topic prompting object
        """

        print("Setting up class...")
        try:
            with open("Data/SavedTopicRepresentations/TopicGpt_20ng.pkl", "rb")  as f:
                self.topicgpt = pickle.load(f)
        except FileNotFoundError:
              with open("../../Data/SavedTopicRepresentations/TopicGpt_20ng.pkl", "rb")  as f:
                self.topicgpt = pickle.load(f)

        print(f"The topic list of this object is: \n {self.topicgpt.topic_lis} \n\n")

    def test_visualize_clusters(self):
        """
        test the visualize_clusters function of the TopicGPT class
        """
        print("Testing visualize_clusters...")
        self.topicgpt.visualize_clusters()

    def test_repr_topics(self):
        """
        test the repr_topics function of the TopicGPT class
        """
        print("Testing repr_topics...")
        self.assertTrue(type(self.topicgpt.repr_topics()) == str)

    def test_promt_knn_search(self):
        """
        test the ppromt function that calls knn_search of the TopicPrompting class
        """
        print("Testing ppromt_knn_search...")

        prompt_lis = ["Is topic 0 about Bananas? Use knn Search",
                      "Is topic 0 about Space? Use knn Search",
                      "Is topic 13 about Space exploration? Use knn Search"]

        for prompt in prompt_lis:

            answer, function_result = self.topicgpt.prompt(prompt)

            print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")

            self.assertTrue(type(answer) == str)
            self.assertTrue(type(function_result[0]) == list)
            self.assertTrue(type(function_result[1]) == list)
            self.assertTrue(type(function_result[0][0]) == str)
            self.assertTrue(type(function_result[1][0]) == int)

    def test_promt_identify_topic_idx(self):
        """
        test the ppromt function that calls identify_topic_idx of the TopicPrompting class
        """

        print("Testing ppromt_identify_topic_idx...")
        prompt_lis = ["What is the index of the topic about Space?",
                      "What is the index of the topic about cars?",
                      "What is the index of the topic about gun control?"]
        correct_indices = [13, 9, 2]

        for prompt, correct_idx in zip(prompt_lis, correct_indices):

            answer, function_result = self.topicgpt.prompt(prompt)

            print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
            print("function_result: ", function_result)
            self.assertTrue(type(answer) == str)
            self.assertTrue(type(function_result) == int)
            self.assertTrue(function_result == correct_idx) # topic 14 is about space

    def test_prompt_identify_topc_idx_no_index_prompt(self):
        """
        test the ppromt function that calls identify_topic_idx of the TopicPrompting class
        """

        print("Testing ppromt_identify_topic_idx...")
        no_index_prompt = "What is the index of the topic about bananas?"

        answer, function_result = self.topicgpt.prompt(no_index_prompt)

        print(f"Answer to the prompt '{no_index_prompt}' \n is \n '{answer}'")
        self.assertTrue(type(answer) == str)
        self.assertTrue(function_result == None)

    def test_prompt_split_topic_kmeans(self):
        """
        test the ppromt function that calls split_topic_kmeans of the TopicPrompting class
        """

        print("Testing ppromt_split_topic_kmeans...")

        prompt_lis = ["Split topic 0 into 2 subtopics using kmeans",
                        "Split topic 1 into 3 subtopics using kmeans",
                        "Split topic 2 into 4 subtopics using kmeans"]
        added_topic_lis_len  = [2, 3, 4]

        for prompt, added_topic_len in zip(prompt_lis, added_topic_lis_len):

                answer, function_result = self.topicgpt.prompt(prompt)

                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)

                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                self.assertTrue(type(function_result[0]) == Topic)
                self.assertTrue(len(function_result) == added_topic_len + len(self.topicgpt.topic_lis) -1 )

    def test_prompt_split_topic_kmeans_inplace(self):
        """
        test the ppromt function that calls split_topic_kmeans of the TopicPrompting class
        """

        print("Testing ppromt_split_topic_kmeans...")

        prompt_lis = ["Split topic 0 into 2 subtopics using kmeans. Do this inplace"]
        added_topic_lis_len  = [2]

        old_number_of_topics = len(self.topicgpt.topic_lis)

        for prompt, added_topic_len in zip(prompt_lis, added_topic_lis_len):

                answer, function_result = self.topicgpt.prompt(prompt)

                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)

                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                self.assertTrue(type(function_result[0]) == Topic)

                self.assertTrue(len(self.topicgpt.topic_lis) == old_number_of_topics + added_topic_len -1 )
                self.assertTrue(self.topicgpt.topic_lis == function_result)

    def test_prompt_split_topic_hdbscan(self):
        """
        test the ppromt function that calls split_topic_hdbscan of the TopicPrompting class
        """

        print("Testing ppromt_split_topic_hdbscan...")

        prompt_lis = ["Split topic 0 into subtopics using hdbscan",
                        "Split topic 1 into subtopics using hdbscan",
                        "Split topic 2 into subtopics using hdbscan"]

        for prompt in prompt_lis:

                answer, function_result = self.topicgpt.prompt(prompt)

                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)

                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                self.assertTrue(type(function_result[0]) == Topic)

    def test_prompt_split_topic_hdbscan_inplace(self):
        """
        test the ppromt function that calls split_topic_hdbscan of the TopicPrompting class
        """

        print("Testing ppromt_split_topic_hdbscan...")

        prompt_lis = ["Split topic 4 into subtopics using hdbscan. Do this inplace"]

        for prompt in prompt_lis:

                answer, function_result = self.topicgpt.prompt(prompt)

                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)

                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                self.assertTrue(type(function_result[0]) == Topic)

                print("topic_gpt_topic_list: ", self.topicgpt.topic_lis)
                print("function_result: ", function_result)
                self.assertTrue(self.topicgpt.topic_lis == function_result)

    def test_prompt_split_topic_keywords(self):
         """
         test the prompt function that calls split_topic_keywords of the TopicPrompting class. This test works almost the same as the test_prompt_split_topic_kmeans
         """

         print("Testing ppromt_split_topic_keywords...")

         prompt_lis = ["Split topic 0 into 2 subtopics based on the keywords Technology and Computers",
                        "Split topic 14 into two subbtopics based on the keywords Space and Exploration"]

         added_topic_lis_len  = [2, 2]   
         for prompt, added_topic_len in zip(prompt_lis, added_topic_lis_len):

                answer, function_result = self.topicgpt.prompt(prompt)

                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)

                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                print(type(function_result[0]))
                self.assertTrue(type(function_result[0]) == Topic)
                self.assertTrue(len(function_result) == added_topic_len + len(self.topicgpt.topic_lis) -1 )

    def test_prompt_split_topic_keywords_inplace(self):
            """
            test the prompt function that calls split_topic_keywords of the TopicPrompting class. This test works almost the same as the test_prompt_split_topic_kmeans
            """

            print("Testing ppromt_split_topic_keywords...")

            prompt_lis = ["Split topic 13 into 2 subtopics based on the keywords 'Rocket and 'Milky Way'. Do this inplace"]

            added_topic_lis_len  = [2]   
            old_number_of_topics = len(self.topicgpt.topic_lis)
            for prompt, added_topic_len in zip(prompt_lis, added_topic_lis_len):

                    answer, function_result = self.topicgpt.prompt(prompt)

                    print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                    print("function_result: ", function_result)

                    self.assertTrue(type(answer) == str)
                    self.assertTrue(type(function_result) == list)
                    self.assertTrue(type(function_result[0]) == Topic)

                    self.assertTrue(len(self.topicgpt.topic_lis) == old_number_of_topics + added_topic_len - 1)
                    self.assertTrue(self.topicgpt.topic_lis == function_result)

    def test_prompt_split_topic_single_keyword(self):
         """
         test the prompt function that calls split_topic_keywords of the TopicPrompting class. This test works almost the same as the test_prompt_split_topic_kmeans
         """

         print("Testing ppromt_split_topic_keywords...")

         prompt_lis = ["Split topic into two topics using the additional keyword 'Technology'",
                        "Split topic into two topics using the additional keyword 'Space'"]

         added_topic_lis_len  = [2, 2]

         for prompt, added_topic_len in zip(prompt_lis, added_topic_lis_len):
                    answer, function_result = self.topicgpt.prompt(prompt)

                    print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                    print("function_result: ", function_result)

                    self.assertTrue(type(answer) == str)
                    self.assertTrue(type(function_result) == list)
                    self.assertTrue(type(function_result[0]) == Topic)
                    self.assertTrue(len(function_result) == added_topic_len + len(self.topicgpt.topic_lis) -1 )

    def test_prompt_split_topic_single_keyword_inplace(self):
            """
            test the prompt function that calls split_topic_keywords of the TopicPrompting class. This test works almost the same as the test_prompt_split_topic_kmeans
            """

            print("Testing ppromt_split_topic_keywords...")

            prompt_lis = ["Split topic 0 into 2 subtopics based on the keywords Technology and Computers. Do this inplace"]

            added_topic_lis_len  = [2]   
            old_number_of_topics = len(self.topicgpt.topic_lis)
            for prompt, added_topic_len in zip(prompt_lis, added_topic_lis_len):

                    answer, function_result = self.topicgpt.prompt(prompt)

                    print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                    print("function_result: ", function_result)

                    self.assertTrue(type(answer) == str)
                    self.assertTrue(type(function_result) == list)
                    self.assertTrue(type(function_result[0]) == Topic)

                    self.assertTrue(len(self.topicgpt.topic_lis) == old_number_of_topics + added_topic_len -1 )
                    self.assertTrue(self.topicgpt.topic_lis == function_result)

    def test_prompt_combine_topics(self):
        """
        test the prompt function that calls combine_topics of the TopicPrompting class
        """

        print("Testing ppromt_combine_topics...")

        prompt_lis = ["Combine topic 0 and topic 1 into one topic",
                        "Combine topic 1 and topic 2 into one topic",
                        "Combine topic 2 and topic 3 into one topic"]

        for prompt in prompt_lis:

                answer, function_result = self.topicgpt.prompt(prompt)

                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)

                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                self.assertTrue(type(function_result[0]) == Topic)
                self.assertTrue(len(function_result) == len(self.topicgpt.topic_lis) -1)

    def test_prompt_combine_topics_inplace(self):
        """
        test the prompt function that calls combine_topics of the TopicPrompting class
        """

        print("Testing ppromt_combine_topics...")

        prompt_lis = ["Combine topic 0 and topic 1 into one topic. Do this inplace"]
        old_number_topics = len(self.topicgpt.topic_lis)

        for prompt in prompt_lis:

                answer, function_result = self.topicgpt.prompt(prompt)

                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)
                print("topic_gpt_topic_list: ", self.topicgpt.topic_lis)

                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                self.assertTrue(type(function_result[0]) == Topic)
                self.assertTrue(self.topicgpt.topic_lis == function_result)
                self.assertTrue(len(self.topicgpt.topic_lis) == old_number_topics -1)

    def test_prompt_add_new_topic_keyword(self):
         """
         test the prompt function that calls add_new_topic_keyword of the TopicPrompting class
         """

         print("Testing ppromt_add_new_topic_keyword...")

         prompt_lis = ["Add a new topic with the keyword 'Politics'",
                    "Add a new topic with the keyword 'Climate Change'",
                    "Add a new topic with the keyword 'Computers'"]

         for prompt in prompt_lis:

                answer, function_result = self.topicgpt.prompt(prompt)

                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)

                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                print(type(function_result[0]))
                self.assertTrue(type(function_result[0]) == Topic)
                self.assertTrue(len(function_result) == len(self.topicgpt.topic_lis) +1)

    def test_prompt_add_new_topic_keyword_inplace(self):
        """
        test the prompt function that calls add_new_topic_keyword of the TopicPrompting class
        """

        print("Testing ppromt_add_new_topic_keyword...")

        prompt_lis = ["Add a new topic with the keyword 'Politics'. Do this inplace"]
        old_number_topics = len(self.topicgpt.topic_lis)

        for prompt in prompt_lis:

                answer, function_result = self.topicgpt.prompt(prompt)

                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)

                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                self.assertTrue(type(function_result[0]) == Topic)
                self.assertTrue(self.topicgpt.topic_lis == function_result)
                self.assertTrue(len(self.topicgpt.topic_lis) == old_number_topics +1)

    def test_prompt_delete_topic(self):
        """
        test the prompt function that calls delete_topic of the TopicPrompting class
        """

        print("Testing ppromt_delete_topic...")

        prompt_lis = ["Delete topic 0",
                        "Delete topic 1",
                        "Delete topic 2"]

        for prompt in prompt_lis:

                answer, function_result = self.topicgpt.prompt(prompt)

                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)

                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                self.assertTrue(type(function_result[0]) == Topic)
                self.assertTrue(len(function_result) == len(self.topicgpt.topic_lis) -1)

    def test_prompt_delete_topic_inplace(self):
        """
        test the prompt function that calls delete_topic of the TopicPrompting class
        """

        print("Testing ppromt_delete_topic...")

        prompt_lis = ["Delete topic 0. Do this inplace"]
        old_number_topics = len(self.topicgpt.topic_lis)

        for prompt in prompt_lis:

                answer, function_result = self.topicgpt.prompt(prompt)

                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)

                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == list)
                self.assertTrue(type(function_result[0]) == Topic)
                self.assertTrue(self.topicgpt.topic_lis == function_result)
                self.assertTrue(len(self.topicgpt.topic_lis) == old_number_topics -1)

    def test_prompt_get_topic_information(self):
         """
         test the get_topic_information function of the TopicGPT class
         """

         print("Testing get_topic_information...")

         prompt_lis = ["Please compare topic 0 and topic 1",
                    "Please compare topic 3,4,5"]

         for prompt in prompt_lis:

                answer, function_result = self.topicgpt.prompt(prompt)

                print(f"Answer to the prompt '{prompt}' \n is \n '{answer}'")
                print("function_result: ", function_result)

                self.assertTrue(type(answer) == str)
                self.assertTrue(type(function_result) == dict) 

if __name__ == "__main__":
   unittest.main()