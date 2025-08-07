==============
TopicGPT
==============

TopicGPT integrates the remarkable capabilities of current LLMs such as GPT-3.5 and GPT-4 into topic modeling.

While traditional topic models extract topics as simple lists of top-words, such as ["Lion", "Leopard", "Rhino", "Elephant", "Buffalo"], TopicGPT offers rich and dynamic topic representations that can be intuitively understood, extensively investigated and modified in various ways via simple text commands.

More specifically, it provides the following core functionalities:

- Identification of clusters within document-embeddings and top-word extraction
- Generation of informative topic descriptions
- Extraction of detailed information about topics via Retrieval-Augmented-Generation (RAG)
- Comparison of topics
- Splitting and combining of identified topics
- Addition of new topics based on keywords
- Deletion of topics

It is further possible to directly interact with TopicGPT via prompting and without explicitly calling functions - an LLM autonomously decides which functionality to use.

Installation Guide
------------------

To install TopicGPT, simply use PyPI:

.. code-block:: bash

    pip install topicgpt

GitHub Repository
-----------------

For more details, usage examples, source code, and testing procedures, please visit the TopicGPT GitHub repository: https://github.com/LMU-Seminar-LLMs/TopicGPT

