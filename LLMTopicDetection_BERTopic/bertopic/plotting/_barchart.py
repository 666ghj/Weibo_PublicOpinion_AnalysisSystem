import itertools
import numpy as np
from typing import List, Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_barchart(
    topic_model,
    topics: List[int] = None,
    top_n_topics: int = 8,
    n_words: int = 5,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Topic Word Scores</b>",
    width: int = 250,
    height: int = 250,
    autoscale: bool = False,
) -> go.Figure:
    """Visualize a barchart of selected topics.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most frequent topics.
        n_words: Number of words to show in a topic
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of each figure.
        height: The height of each figure.
        autoscale: Whether to automatically calculate the height of the figures to fit the whole bar text

    Returns:
        fig: A plotly figure

    Examples:
    To visualize the barchart of selected topics
    simply run:

    ```python
    topic_model.visualize_barchart()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_barchart()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/bar_chart.html"
    style="width:1100px; height: 660px; border: 0px;""></iframe>
    """
    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list()[0:6])

    # Initialize figure
    if isinstance(custom_labels, str):
        subplot_titles = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        subplot_titles = ["_".join([label[0] for label in labels[:4]]) for labels in subplot_titles]
        subplot_titles = [label if len(label) < 30 else label[:27] + "..." for label in subplot_titles]
    elif topic_model.custom_labels_ is not None and custom_labels:
        subplot_titles = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topics]
    else:
        subplot_titles = [f"Topic {topic}" for topic in topics]
    columns = 4
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(
        rows=rows,
        cols=columns,
        shared_xaxes=False,
        horizontal_spacing=0.1,
        vertical_spacing=0.4 / rows if rows > 1 else 0,
        subplot_titles=subplot_titles,
    )

    # Add barchart for each topic
    row = 1
    column = 1
    for topic in topics:
        words = [word + "  " for word, _ in topic_model.get_topic(topic)][:n_words][::-1]
        scores = [score for _, score in topic_model.get_topic(topic)][:n_words][::-1]

        fig.add_trace(
            go.Bar(x=scores, y=words, orientation="h", marker_color=next(colors)),
            row=row,
            col=column,
        )

        if autoscale:
            if len(words) > 12:
                height = 250 + (len(words) - 12) * 11

            if len(words) > 9:
                fig.update_yaxes(tickfont=dict(size=(height - 140) // len(words)))

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            "text": f"{title}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        width=width * 4,
        height=height * rows if rows > 1 else height * 1.3,
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig
