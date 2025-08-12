import numpy as np
from typing import List, Union
import plotly.graph_objects as go


def visualize_term_rank(
    topic_model,
    topics: List[int] = None,
    log_scale: bool = False,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Term score decline per Topic</b>",
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """Visualize the ranks of all terms across all topics.

    Each topic is represented by a set of words. These words, however,
    do not all equally represent the topic. This visualization shows
    how many words are needed to represent a topic and at which point
    the beneficial effect of adding words starts to decline.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize. These will be colored
                red where all others will be colored black.
        log_scale: Whether to represent the ranking on a log scale
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Returns:
        fig: A plotly figure

    Examples:
    To visualize the ranks of all words across
    all topics simply run:

    ```python
    topic_model.visualize_term_rank()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_term_rank()
    fig.write_html("path/to/file.html")
    ```

    <iframe src="../../getting_started/visualization/term_rank.html"
    style="width:1000px; height: 530px; border: 0px;""></iframe>

    <iframe src="../../getting_started/visualization/term_rank_log.html"
    style="width:1000px; height: 530px; border: 0px;""></iframe>

    Reference:

    This visualization was heavily inspired by the
    "Term Probability Decline" visualization found in an
    analysis by the amazing [tmtoolkit](https://tmtoolkit.readthedocs.io/).
    Reference to that specific analysis can be found
    [here](https://wzbsocialsciencecenter.github.io/tm_corona/tm_analysis.html).
    """
    topics = [] if topics is None else topics

    topic_ids = topic_model.get_topic_info().Topic.unique().tolist()
    topic_words = [topic_model.get_topic(topic) for topic in topic_ids]

    values = np.array([[value[1] for value in values] for values in topic_words])
    indices = np.array([[value + 1 for value in range(len(values))] for values in topic_words])

    # Create figure
    lines = []
    for topic, x, y in zip(topic_ids, indices, values):
        if not any(y > 1.5):
            # labels
            if isinstance(custom_labels, str):
                label = f"{topic}_" + "_".join(list(zip(*topic_model.topic_aspects_[custom_labels][topic]))[0][:3])
            elif topic_model.custom_labels_ is not None and custom_labels:
                label = topic_model.custom_labels_[topic + topic_model._outliers]
            else:
                label = f"<b>Topic {topic}</b>:" + "_".join([word[0] for word in topic_model.get_topic(topic)])
                label = label[:50]

            # line parameters
            color = "red" if topic in topics else "black"
            opacity = 1 if topic in topics else 0.1
            if any(y == 0):
                y[y == 0] = min(values[values > 0])
            y = np.log10(y, out=y, where=y > 0) if log_scale else y

            line = go.Scatter(
                x=x,
                y=y,
                name="",
                hovertext=label,
                mode="lines+lines",
                opacity=opacity,
                line=dict(color=color, width=1.5),
            )
            lines.append(line)

    fig = go.Figure(data=lines)

    # Stylize layout
    fig.update_xaxes(range=[0, len(indices[0])], tick0=1, dtick=2)
    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        title={
            "text": f"{title}",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        width=width,
        height=height,
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
    )

    fig.update_xaxes(title_text="Term Rank")
    if log_scale:
        fig.update_yaxes(title_text="c-TF-IDF score (log scale)")
    else:
        fig.update_yaxes(title_text="c-TF-IDF score")

    return fig
