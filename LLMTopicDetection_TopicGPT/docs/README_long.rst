TopicGPT
========

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


GitHub Repository
----------------

You can find the source code and related materials for this project in the GitHub repository:

- [TopicGPT](https://github.com/LMU-Seminar-LLMs/TopicGPT/tree/dev)





Installation
------------

You can install topicgpt via `PyPI <https://pypi.org/project/topicgpt/>`

::

    pip install topicgpt


Example
=======

The following example demonstrates how TopicGPT can be used on a real-world dataset. The Twenty Newsgroups corpus (`Twenty Newsgroups Corpus Documentation <https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html>`_) is used for this purpose.

Load the data
-------------

.. code-block:: python

    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes')) #download the 20 Newsgroups dataset
    corpus = data['data']

    corpus = [doc for doc in corpus if doc != ""] #remove empty documents

Initialize the model
--------------------

Note that an OpenAi API-Key is needed to compute the embeddings and execute the prompts. See `OpenAI API Keys Documentation <https://platform.openai.com/account/api-keys>`_ for more details. We select 20 topics in this case since the Twenty Newsgroups corpus comprises documents from 20 different newsgroups. It is also possible to let Hdbscan determine the number of topics automatically.

.. code-block:: python

    from topicgpt.TopicGPT import TopicGPT

    tm = TopicGPT(
        openai_api_key = <your-openai-api-key>,
        n_topics = 20 # select 20 topics since the true number of topics is 20
    )


Fit the model
------------

The fit-method fits the model. This can take, depending on the size of the dataset and whether embeddings have been provided, from a few minutes to several hours. Especially the computation of the embeddings can take some time.

.. code-block:: python

    tm.fit(corpus) # the corpus argument should be of type list[str] where each string represents one document

Inspect the found topics
------------------------

Obtain an overview of the identified topics.

.. code-block:: python

    print(tm.topic_lis)

    Output:

    .. code-block:: plaintext

        [Topic 0: Electronics Equipment Sales,
         Topic 1: Image Processing,
         Topic 2: Gun control,
         Topic 3: Online Privacy and Anonymity,
         Topic 4: Conflict and Violence.,
         Topic 5: Computer Hardware,
         Topic 6: Belief and Atheism,
         Topic 7: Online Discussions,
         Topic 8: Computer Software,
         Topic 9: Car Features and Performance,
         Topic 10: Encryption and Government,
         Topic 11: Technology and Computing.,
         Topic 12: Technology and Computing,
         Topic 13: Space Exploration,
         Topic 14: Motorcycle Riding Techniques,
         Topic 15: Technology,
         Topic 16: Hockey Games,
         Topic 17: Health and Medicine.,
         Topic 18: Baseball games and teams.,
         Topic 19: Beliefs about Homosexuality.]

To obtain more detailed information on each topic, we can call the "print_topics" method:

.. code-block:: python

    tm.print_topics()

    Output:

    .. code-block:: plaintext

        Topic 0: Electronics Equipment Sales

        Topic_description: The common topic of the given words appears to be "electronics and technology".

        Various aspects and sub-topics of this topic include:
        1. Buying and selling: "offer", "sale", "sell", "price", "buy"
        2. Device usage and features: "use", "get", "new", "used", "condition"
        3. Technical specifications: "wire", "ground", "power", "circuit", "voltage"
        4. Communication and connectivity: "phone", "email", "modem", "wireless", "connection"
        5. Accessories and peripherals: "battery", "cable", "manuals", "disk", "monitor"
        Top words: ["n't", 'one', 'would', 'use', 'like', 'get', 'new', 'used', 'offer', 'sale']

        [...]

We can also visualize the resulting clusters to get an overview of the shape and size of the clusters.

.. code-block:: plaintext

    tm.visualize_clusters()

Find out more detailed information about the identified topics
------------------------------------------------------------

First, we might be interested in knowing what information the space topic (topic 13) contains on the moon landing.

.. code-block:: python

    tm.pprompt("Which information related to the keyword 'moon landing' does topic 13 have?")

    Output:

    .. code-block:: plaintext

        GPT wants to call the function: {
          "name": "knn_search",
          "arguments": "{\n  \"topic_index\": 13,\n  \"query\": \"moon landing\",\n  \"k\": 5\n}"
        }
        Topic 13, which is related to the keyword "moon landing," has the following information:

        1. Document index 258: This document provides an introduction to the solar system and mentions that advancements in rocketry after World War II enabled machines to travel to the Moon and other planets. It highlights that the United States has sent both automated spacecraft and human-crewed expeditions to explore the Moon.

        2. Document index 535: This document discusses a $65 million program called the Back to the Moon bill, which aims to encourage private companies to develop lunar orbiters. It mentions that there is a chance of making a lunar mission happen in this decade through this program.

        3. Document index 357: This document is a request for more information on a recent newspaper article about the Japanese crashing or crash-landing a package on the Moon. It indicates that the article was vague and unclear.

        4. Document index 321: This document speculates about what would have happened if the Soviets had beaten the United States in the Moon race. It suggests that the US would have still performed Moon landings and potentially set up a lunar base. The focus on Mars exploration would have depended on the Soviets' actions.

        5. Document index 102: This document mentions the Hiten engineering-test mission, which spent time in a highly eccentric Earth orbit and performed lunar flybys before being inserted into lunar orbit using gravity-assist-like maneuvers. It states that the mission was expected to crash on the Moon eventually.

        Please note that the above summaries are based on the content of the documents and may not capture all the information contained within them.

From this output, we see that an instance of a GPT decided to call the function "knn_search" from the class "TopicPrompting." Indeed, some documents on the topic "moon landing" have been found, and the model summarizes the relevant information accordingly.

If we want to check, for instance, the document with index 102 in topic 13 to learn more about the Hiten engineering-test mission, we can simply do the following:

.. code-block:: python

    print(tm.topic_lis[13].documents[535])

    Output:

    .. code-block:: plaintext

        Their Hiten engineering-test mission spent a while in a highly eccentric Earth orbit doing lunar flybys, and then was inserted into lunar orbit using some very tricky gravity-assist-like maneuvering. This meant that it would crash on the Moon eventually, since there is no such thing as a stable lunar orbit (as far as anyone knows), and I believe I recall hearing recently that it was about to happen.


Split Topics
------------

We find that topic 6 about Belief and Atheism is a bit general and would thus like to split it into subtopics. Let's see what 5 potential subtopics it contains:

.. code-block:: python

    tm.pprompt("What are 5 potential subtopics of topic 6")

    Output:

    .. code-block:: plaintext

        GPT wants to call the function:  {
          "name": "split_topic_kmeans",
          "arguments": "{\n  \"topic_idx\": 6,\n  \"n_clusters\": 5\n}"
        }

        Here are five potential subtopics of topic 6:

        1. Existence of God: This subtopic explores arguments, evidence, proofs, and the existence of God.

        2. Atheism vs Theism: This subtopic delves into the concepts of atheism, theism, atheists, and theists, and the debates surrounding these belief systems.

        3. Belief and Faith: This subtopic focuses on beliefs, faith, believers, and the roles they play in religious and philosophical contexts.

        4. Rationality and Reasoning: This subtopic involves discussions on reason, rationality, logic, and argumentation as they relate to beliefs and debates about religion.

        5. Morality and Ethics: This subtopic explores moral values, ethics, morals, and how they influence and are influenced by religious beliefs and debates.

        These subtopics were inferred from the words provided and the descriptions of the topics in the corpus used for topic modeling.

Based on those results, we decide to not actually split the topic into 5 subtopics but rather just 3 - One on "religious faith," one on "atheism," and one on "ethics and philosophy." We also want to actually make those changes to the existing topics, so we tell TopicGPT to do the changes inplace.

.. code-block:: python

    tm.pprompt("Please split topic 6 into subtopics based on the keywords 'religious faith', 'atheism' and 'ethics and philosophy'. Do this inplace.")

```

Output:

    .. code-block:: plaintext


        I have split topic 6 into three subtopics based on the provided keywords: "religious faith", "atheism", and "ethics and philosophy". Here are the new subtopics:

        1. Topic 21: Religious Beliefs
          - This topic encompasses the general theme of "Religion and Belief".
          - Subtopics and aspects of this topic include:
            - Faith and Beliefs: Believing, belief, true, evidence, truth.
            - Religious Practices: Praying, heretic, bible, church.
            - Religious Communities: Christian, Muslim, atheist.
            - Interpretation and Understanding: Interpretations, interpretation, understanding, context.
            - Morality and Ethics: Good, wrong, punishment, violence.

        2. Topic 22: Existence of God
          - This topic focuses on "Atheism and Belief".
          - Subtopics and aspects of this topic include:
            - Arguments and Evidence: Atheists provide evidence for their beliefs.
            - Existence of God: Atheists question the existence of God.
            - Religious Beliefs: Atheists challenge religious beliefs.
            - Science and Reason: Atheists rely on science and reason.
            - Morality and Ethics: Atheists discuss morality and ethics.

        3. Topic 23: Ethics and Morality
          - This topic revolves around "Morality and Ethics".
          - Subtopics and aspects of this topic include:
            - Moral reasoning: "Argument," "reasoning," "logic," "fallacy," "premises."
            - Ethical principles: "Right," "wrong," "immoral," "morals," "ethical."
            - Moral judgments: "Moral," "ethical," "morality," "judgments," "values."
            - Human behavior: "Homosexual," "sexual," "promiscuous," "behavior," "sex."
            - Scientific perspective: "Science," "scientific," "evidence," "empirical," "studies."
        Please note that these subtopics are based on the keywords provided and may not cover all aspects of the original topic.


We see that TopicGPT performed the splitting as intended. However, the names and descriptions of the topics got changed slightly in order to optimally fit to the documents that actually constitute the topics based on the keywords. (If you like you can rename them at any time). 

Combining topics
===============

Since topics 15 ("Hockey Games") and 17 ("Baseball games and teams") are both about sports, we want to combine them into a single topic.

.. code-block:: python

    tm.pprompt("Please combine topics 15 and 17. Do this inplace.")

Output
------

GPT wants to the call the function:

.. code-block:: json

    {
      "name": "combine_topics",
      "arguments": "{\n  \"topic_idx_lis\": [15, 17],\n  \"inplace\": true\n}"
    }

The topics 15 and 17 have been combined into a new topic called "Sports". This topic includes aspects and sub-topics related to sports such as team and players, games and seasons, performance and skills, fans and audience, and statistics and records. Some of the common words found in this topic include "team," "players," "hockey," "baseball," "game," "games," "season," "playoffs," "good," "better," "win," "hit," "score," "fans," "series," "watch," "fan," "stats," "record," "pts," and "career".

Tips and tricks for prompting TopicGPT
---------------------------------------

When using the "pprompt" or "prompt" function, TopicGPT can behave differently than intended. To alleviate those issues some simple tricks can help:

- Explicitly tell the model which function it should use and which parameters to select. (Sometimes the model simply cannot know what you except it to do.) For example, instead of using ``tm.pprompt("What are the subtopic of topic 13?")``, use something like ``tm.pprompt("What are the subtopic of topic 13? Please use the function that uses the k-means algorithm to split the topic. Use a parameter of k = 5 and do this inplace")``.

- Just ask the same prompt again. Since TopicGPT is a stochastic system, calling the same function with the same argument again might yield a different functionality to be used or a different result.

- If this doesn't help, you can also directly call the function you want to use from the TopicPrompting class. In the example above you could do ``tm.topic_prompting.split_topic_kmeans(topic_idx=13, n_clusters=5, inplace=True)``. Note that all functions the model can call can also be called directly.

- In case of hallucination of facts it may help to use GPT-4 for TopicGPT



How TopicGPT works
==================

TopicGPT is centrally built on top of text embeddings and the prompting mechanisms obtained via LLMs and provided by the OpenAI API. Please also see the section `References <#references_>`_ for more details on the models and ideas used in TopicGPT.

Embeddings
----------

When no embeddings are provided, TopicGPT automatically computes the embeddings of the documents of the provided corpus and also of the vocabulary that is extracted from the corpus. This happens after the fit-method is called.

The class ``GetEmbeddingsOpenAI`` is used for this purpose.

Clustering
----------

In order to identify topics among the documents, TopicGPT reduces the dimensionality of the document embeddings via UMAP and then uses Hdbscan to identify the clusters. Dimensionality reduction is necessary since the document embeddings are of very high dimensionality, and thus the curse of dimensionality would make it very difficult, if not impossible, to identify the clusters.

When not specifying the number of topics in the ``Topic GPT`` class, Hdbscan is used to automatically determine the number of topics. If the number of topics is specified, agglomerative clustering is used on top of the clusters identified by HDBSCAN.

The class ``Clustering`` is used for this purpose.

Extraction of Top-Words
------------------------

After the clusters have been identified, TopicGPT extracts the top-words of each topic. This is done via two different methods:

- **Tf-idf**: The tf-idf method is based on the idea that words that occur frequently in a topic but rarely in other topics are good indicators for the topic. The top-words are thus the words with the highest tf-idf scores.

- **Centroid similarity**: The centroid similarity method is based on the idea that the words that are closest to the centroid of a topic are good indicators for the topic. The top-words are thus the words that are closest to the centroid of the topic.

Note that the Tf-idf heuristic was introduced for the BerTopic Model (Grootendorst, Maarten. "BERTopic: Neural topic modeling with a class-based TF-IDF procedure." arXiv preprint arXiv:2203.05794 (2022)) and a similar idea to the centroid similarity method is used in Top2Vec (Angelov, Dimo. "Top2vec: Distributed representations of topics." arXiv preprint arXiv:2008.09470 (2020)).

Topword extraction is performed with help of the class ``ExtractTopWords``.

Describing and naming topics
------------------------------

In the next step, all topics are provided with a short name and a description. This is done via prompting an LLM provided by OpenAI with around 500 top-words of each topic. The LLM then generates a short name and a description for each topic.

The class ``TopwordEnhancement`` is used for this purpose.

Note that computation of Embeddings, Extraction of Top-Words, and Describing and Naming Topics are all performed when calling the ``fit`` method of the ``TopicGPT`` class.

#### Describing and naming topics

In the next step, all topics are provided with a short name and a description. This is done via prompting an LLM provided by OpenAI with around 500 top-words of each topic. The LLM then generates a short name and a description for each topic.

The class ```TopwordEnhancement``` is used for this purpose.


Note that computation of Embeddings, Extraction of Top-Words and Describing and Naming Topics are all performed when calling the ```fit``` method of the ```TopicGPT``` class.	

Prompting
---------

The main way to interact with TopicGPT is via direct textual prompts. Those prompts are augmented with basic information about desired behavior and potentially useful information. Additionally, information on available functions and their parameters is provided. Then this information is used to prompt an LLM via the OpenAI API. The LLM then decides if it should call a function of the ones provided and if so, which parameters to use. The respective function call is executed, and part of the result is returned to the LLM, which uses the original prompt together with the function call and the result to generate a response.

Functions available for prompting
---------------------------------

The following functions are available for the LLM to use:

- ``knn_search``: This function is used to find documents that are related to a certain keyword. The LLM can specify the number of documents to be found and the number of keywords to be used. The result is retrieved by performing retrieval-augmented-generation (RAG) where the query is embedded, and the most similar documents are retrieved.

- ``identify_topic_idx``: This function is used to identify the topic that is most related to a certain keyword. This is simply done by providing all topic descriptions to the LLM and then asking for the index of the topic that is most related to the keyword.

- ``get_topic_information``: This function is used to obtain information on certain topics. This can be useful to compare similar topics.

- ``split_topic_kmeans``: This function is used to split a topic into subtopics. The LLM can specify the number of subtopics to be created. The result is retrieved by performing k-means clustering on the document embeddings of the documents in the topic. Note that when splitting a topic, the top-words are not completely recomputed, but rather the top-words of the "super"-topic are distributed among the subtopics.

- ``split_topic_hdbscan``: Works analogously to ``split_topic_kmeans`` but uses Hdbscan instead of k-means clustering. This means that the number of subtopics is not specified by the user but rather automatically determined by Hdbscan.

- ``split_topic_keywords``: This function is used to split a topic into subtopics based on provided keywords. Each keyword is embedded, and the topic is split according to cosine similarity of the document embeddings within the "super"-topic. This means that documents among the "super"-topic that are most similar to a certain keyword are assigned to the corresponding subtopic.

- ``add_new_topic_keyword``: This function is used to add a new topic based on a keyword. The documents belonging to this new topic are computed as the documents from all other topics that are more similar to the embedding of the new keyword than the centroid of the original topic. Then all topwords and the topic description are recomputed.

- ``delete_topic``: This function is used to delete a topic. The LLM can specify the topic to be deleted. The result is retrieved by simply removing the topic from the list of topics and assigning the documents of the deleted topic to the topic with the most similar centroid. Then all topwords and the topic description are recomputed.

- ``combine_topics``: This function is used to combine two topics into a single topic. The LLM can specify the two topics to be combined. The result is retrieved by simply combining the documents of the two topics and re-computing the embeddings and top-words of the new topic.



Limitations and Caveats
------------------------

It is important to note that, as a model built on top of inherently stochastic LLMs and all their shortcomings, TopicGPT has several limitations and shortcomings as well. The following list is not aimed at being complete but could provide useful information on what may go wrong when using TopicGPT:

- **Hallucination**: LLMs are well known for yielding incorrect but coherent and plausible answers that seem convincing but are actually just made up. Although we tried to minimize this undesired behavior through carefully designing the used prompts, we found that TopicGPT may hallucinate (especially) with respect to the following aspects:

  - Making up, distorting, or misinterpreting content of documents retrieved via knn-search.
  - Incorrectly naming and describing topics based on top-words. Specifically, the model can identify topics that seem coherent and reasonable, although the corresponding documents are not actually related.

- **Undesired Behavior**: When using the "prompt" or "pprompt" function, TopicGPT may not call the function you intended it to call. This can be alleviated by explicitly telling the model which function to use or directly calling the function yourself.

- **Stochasticity**: The behavior of TopicGPT is not deterministic and exhibits some randomness. There is always some probability that certain actions do not work as intended at the first try because some components of the LLM do not function as desired. Simply trying again should mostly help with those issues.

  - On the other hand, TopicGPT may also be overly cautious and report that no relevant information has been found or no topic exists that matches a certain keyword, even though it does. This could be caused by designing prompts to prevent the massive occurrence of falsely positive results.

  Note that using GPT-4 in TopicGPT can help to significantly alleviate issues with hallucination.

- **Erroneous Embeddings**: The document- and word-embeddings used in TopicGPT may not always reflect the actual semantics of the texts correctly. More specifically, the embeddings sometimes reflect, for instance, grammatical or orthographical aspects such that clusters based on those aspects may be identified.

References
----------

The following models, software packages, and ideas are central for TopicGPT:

- **UMAP**: The Uniform Manifold Approximation and Projection for Dimension Reduction algorithm is used for reducing the dimensionality of document- and word embeddings (McInnes, Leland, John Healy, and James Melville. "Umap: Uniform manifold approximation and projection for dimension reduction." arXiv preprint arXiv:1802.03426 (2018)).

- **HDBSCAN**: Hierarchical density-based clustering is used to identify the clusters among the dimensionality reduced topics (McInnes, Leland, John Healy, and Steve Astels. "hdbscan: Hierarchical density-based clustering." J. Open Source Softw. 2.11 (2017): 205).

- **Agglomerative Clustering**: The agglomerative clustering functionality from sklearn is used to combine topics in case the identified number of clusters exceeds the number of topics specified by the user (Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." the Journal of machine Learning research 12 (2011): 2825-2830., https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html).

- **Topword extraction**: Even though the corresponding packages are not directly used, the topword extraction methods used for this package are based on very similar ideas as found in the BerTopic Model (Grootendorst, Maarten. "BERTopic: Neural topic modeling with a class-based TF-IDF procedure." arXiv preprint arXiv:2203.05794 (2022)) in the case of the tf-idf method and in Top2Vec for the centroid-similarity method (Angelov, Dimo. "Top2vec: Distributed representations of topics." arXiv preprint arXiv:2008.09470 (2020)).

- **LLMs from the GPT family**: Some references for the models for computing embeddings and answering the prompts include:

  - Brown, Tom B., et al. “Language Models are Few-Shot Learners.” Advances in Neural Information Processing Systems 33 (2020).

  - Radford, Alec, et al. “GPT-4: Generative Pre-training of Transformers with Discrete Latent Variables.” arXiv preprint arXiv:2302.07413 (2023).

  - Radford, Alec, et al. “Improving Language Understanding by Generative Pre-Training.” URL: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf. [6]

  - Radford, Alec, et al. “Language Models are Unsupervised Multitask Learners.” OpenAI Blog 1.8 (2019): 9. [7]

