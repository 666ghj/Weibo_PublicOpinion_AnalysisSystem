import pandas as pd
from typing import List, Union
import plotly.graph_objects as go
from sklearn.preprocessing import normalize


def visualize_topics_per_class(
    topic_model,
    topics_per_class: pd.DataFrame,
    top_n_topics: int = 10,
    topics: List[int] = None,
    normalize_frequency: bool = False,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Topics per Class</b>",
    width: int = 1250,
    height: int = 900,
) -> go.Figure:
    """Visualize topics per class.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics_per_class: The topics you would like to be visualized with the
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
    To visualize the topics per class, simply run:

    ```python
    topics_per_class = topic_model.topics_per_class(docs, classes)
    topic_model.visualize_topics_per_class(topics_per_class)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_topics_per_class(topics_per_class)
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/topics_per_class.html"
    style="width:1400px; height: 1000px; border: 0px;""></iframe>
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
    topics_per_class["Name"] = topics_per_class.Topic.map(topic_names)
    data = topics_per_class.loc[topics_per_class.Topic.isin(selected_topics), :]

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(selected_topics):
        if index == 0:
            visible = True
        else:
            visible = "legendonly"
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            x = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            x = trace_data.Frequency
        fig.add_trace(
            go.Bar(
                y=trace_data.Class,
                x=x,
                visible=visible,
                marker_color=colors[index % 7],
                hoverinfo="text",
                name=topic_name,
                orientation="h",
                hovertext=[f"<b>Topic {topic}</b><br>Words: {word}" for word in words],
            )
        )

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        xaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
        yaxis_title="Class",
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
