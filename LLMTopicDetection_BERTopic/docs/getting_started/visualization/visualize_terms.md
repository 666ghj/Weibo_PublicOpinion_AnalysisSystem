We can visualize the selected terms for a few topics by creating bar charts out of the c-TF-IDF scores
for each topic representation. Insights can be gained from the relative c-TF-IDF scores between and within
topics. Moreover, you can easily compare topic representations to each other.
To visualize this hierarchy, run the following:

```python
topic_model.visualize_barchart()
```

<iframe src="bar_chart.html" style="width:1100px; height: 660px; border: 0px;""></iframe>


## **Visualize Term Score Decline**
Topics are represented by a number of words starting with the best representative word.
Each word is represented by a c-TF-IDF score. The higher the score, the more representative a word
to the topic is. Since the topic words are sorted by their c-TF-IDF score, the scores slowly decline
with each word that is added. At some point adding words to the topic representation only marginally
increases the total c-TF-IDF score and would not be beneficial for its representation.

To visualize this effect, we can plot the c-TF-IDF scores for each topic by the term rank of each word.
In other words, the position of the words (term rank), where the words with
the highest c-TF-IDF score will have a rank of 1, will be put on the x-axis. Whereas the y-axis
will be populated by the c-TF-IDF scores. The result is a visualization that shows you the decline
of c-TF-IDF score when adding words to the topic representation. It allows you, using the elbow method,
the select the best number of words in a topic.

To visualize the c-TF-IDF score decline, run the following:

```python
topic_model.visualize_term_rank()
```

<iframe src="term_rank.html" style="width:1000px; height: 530px; border: 0px;""></iframe>

To enable the log scale on the y-axis for a better view of individual topics, run the following:

```python
topic_model.visualize_term_rank(log_scale=True)
```

<iframe src="term_rank_log.html" style="width:1000px; height: 530px; border: 0px;""></iframe>

This visualization was heavily inspired by the "Term Probability Decline" visualization found in an
analysis by the amazing [tmtoolkit](https://tmtoolkit.readthedocs.io/).
Reference to that specific analysis can be found
[here](https://wzbsocialsciencecenter.github.io/tm_corona/tm_analysis.html).
