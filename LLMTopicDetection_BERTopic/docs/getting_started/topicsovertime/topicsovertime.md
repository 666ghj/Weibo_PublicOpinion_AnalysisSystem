Dynamic topic modeling (DTM) is a collection of techniques aimed at analyzing the evolution of topics
over time. These methods allow you to understand how a topic is represented across different times.
For example, in 1995 people may talk differently about environmental awareness than those in 2015. Although the
topic itself remains the same, environmental awareness, the exact representation of that topic might differ.

BERTopic allows for DTM by calculating the topic representation at each timestep without the need to
run the entire model several times. To do this, we first need to fit BERTopic as if there were no temporal
aspect in the data. Thus, a general topic model will be created. We use the global representation as to the main topics that can be found at, most likely, different timesteps. For each topic and timestep, we calculate the c-TF-IDF representation. This will result in a specific topic representation at each timestep without the need to create clusters from embeddings as they were already created.

<br>
<div class="svg_image">
--8<-- "docs/getting_started/topicsovertime/topicsovertime.svg"
</div>
<br>

Next, there are two main ways to further fine-tune these specific topic representations,
namely **globally** and **evolutionary**.

A topic representation at timestep *t* can be fine-tuned **globally** by averaging its c-TF-IDF representation with that of the global representation. This allows each topic representation to move slightly towards the global representation whilst still keeping some of its specific words.

A topic representation at timestep *t* can be fine-tuned **evolutionary** by averaging its c-TF-IDF representation with that of the c-TF-IDF representation at timestep *t-1*. This is done for each topic representation allowing for the representations to evolve over time.

Both fine-tuning methods are set to `True` as a default and allow for interesting representations to be created.

## **Example**
To demonstrate DTM in BERTopic, we first need to prepare our data. A good example of where DTM is useful is topic
modeling on Twitter data. We can analyze how certain people have talked about certain topics in the years
they have been on Twitter. Due to the controversial nature of his tweets, we are going to be using all
tweets by Donald Trump.

First, we need to load the data and do some very basic cleaning. For example, I am not interested in his
re-tweets for this use-case:

```python
import re
import pandas as pd

# Prepare data
trump = pd.read_csv('https://drive.google.com/uc?export=download&id=1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6')
trump.text = trump.apply(lambda row: re.sub(r"http\S+", "", row.text).lower(), 1)
trump.text = trump.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.text.split())), 1)
trump.text = trump.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.text).split()), 1)
trump = trump.loc[(trump.isRetweet == "f") & (trump.text != ""), :]
timestamps = trump.date.to_list()
tweets = trump.text.to_list()
```

Then, we need to extract the global topic representations by simply creating and training a BERTopic model:

```python
from bertopic import BERTopic

topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(tweets)
```

From these topics, we are going to generate the topic representations at each timestamp for each topic. We do this
by simply calling `topics_over_time` and passing the tweets, the corresponding timestamps, and the related topics:

```python
topics_over_time = topic_model.topics_over_time(tweets, timestamps, nr_bins=20)
```

And that is it! Aside from what you always need for BERTopic, you now only need to add `timestamps`
to quickly calculate the topics over time.

## **Parameters**
There are a few parameters that are of interest which will be discussed below.

### **Tuning**
Both `global_tuning` and `evolutionary_tuning` are set to True as a default, but can easily be changed. Perhaps
you do not want the representations to be influenced by the global representation and merely see how they
evolved over time:

```python
topics_over_time = topic_model.topics_over_time(tweets, timestamps,
                                                global_tuning=True, evolution_tuning=True, nr_bins=20)
```

### **Bins**
If you have more than 100 unique timestamps, then there will be topic representations created for each of those
timestamps which can negatively affect the topic representations. It is advised to keep the number of unique
timestamps below 50. To do this, you can simply set the number of bins that are created when calculating the
topic representations. The timestamps will be taken and put into equal-sized bins:

```python
topics_over_time = topic_model.topics_over_time(tweets, timestamps, nr_bins=20)
```

### **Datetime format**
If you are passing strings (dates) instead of integers, then BERTopic will try to automatically detect
which datetime format your strings have. Unfortunately, this will not always work if they are in an unexpected format.
We can use `datetime_format` to pass the format the timestamps have:

```python
topics_over_time = topic_model.topics_over_time(tweets, timestamps, datetime_format="%b%M", nr_bins=20)
```

## **Visualization**
To me, DTM becomes truly interesting when you have a good way of visualizing how topics have changed over time.
A nice way of doing so is by leveraging the interactive abilities of Plotly. Plotly allows us to show the frequency
of topics over time whilst giving the option of hovering over the points to show the time-specific topic representations.
Simply call `visualize_topics_over_time` with the newly created topics over time:

```python
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
```

I used `top_n_topics` to only show the top 20 most frequent topics. If I were to visualize all topics, which is possible by
leaving `top_n_topics` empty, there is a chance that hundreds of lines will fill the plot.

You can also use `topics` to show specific topics:

```python
topic_model.visualize_topics_over_time(topics_over_time, topics=[9, 10, 72, 83, 87, 91])
```

<iframe src="trump.html" style="width:1000px; height: 680px; border: 0px;""></iframe>
