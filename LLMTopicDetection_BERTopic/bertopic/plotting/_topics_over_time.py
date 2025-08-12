import pandas as pd
from typing import List, Union
import plotly.graph_objects as go
from sklearn.preprocessing import normalize


def visualize_topics_over_time(
    topic_model,
    topics_over_time: pd.DataFrame,
    top_n_topics: int = None,
    topics: List[int] = None,
    normalize_frequency: bool = False,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Topics over Time</b>",
    width: int = 1250,
    height: int = 450,
) -> go.Figure:
    """Visualize topics over time.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics_over_time: The topics you would like to be visualized with the
                          corresponding topic representation
        top_n_topics: To visualize the most frequent topics instead of all
        topics: Select which topics you would like to be visualized
        normalize_frequency: Whether to normalize each topic's frequency individually
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Returns:
        A plotly.graph_objects.Figure including all traces

    Examples:
    To visualize the topics over time, simply run:

    ```python
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    topic_model.visualize_topics_over_time(topics_over_time)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/trump.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """
    colors = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#D55E00",
        "#0072B2",
        "#CC79A7",
    ]

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    if isinstance(custom_labels, str):
        topic_names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        topic_names = ["_".join([label[0] for label in labels[:4]]) for labels in topic_names]
        topic_names = [label if len(label) < 30 else label[:27] + "..." for label in topic_names]
        topic_names = {key: topic_names[index] for index, key in enumerate(topic_model.topic_labels_.keys())}
    elif topic_model.custom_labels_ is not None and custom_labels:
        topic_names = {
            key: topic_model.custom_labels_[key + topic_model._outliers] for key, _ in topic_model.topic_labels_.items()
        }
    else:
        topic_names = {
            key: value[:40] + "..." if len(value) > 40 else value for key, value in topic_model.topic_labels_.items()
        }
    topics_over_time["Name"] = topics_over_time.Topic.map(topic_names)
    data = topics_over_time.loc[topics_over_time.Topic.isin(selected_topics), :].sort_values(["Topic", "Timestamp"])

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(data.Topic.unique()):
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(
            go.Scatter(
                x=trace_data.Timestamp,
                y=y,
                mode="lines",
                marker_color=colors[index % 7],
                hoverinfo="text",
                name=topic_name,
                hovertext=[f"<b>Topic {topic}</b><br>Words: {word}" for word in words],
            )
        )

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
        title={
            "text": f"{title}",
            "y": 0.95,
            "x": 0.40,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        legend=dict(
            title="<b>Global Topic Representation",
        ),
    )
    return fig
