import numpy as np
from typing import List, Union
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics.pairwise import cosine_similarity
from bertopic._utils import select_topic_representation

import plotly.express as px
import plotly.graph_objects as go


def visualize_heatmap(
    topic_model,
    topics: List[int] = None,
    top_n_topics: int = None,
    n_clusters: int = None,
    use_ctfidf: bool = False,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Similarity Matrix</b>",
    width: int = 800,
    height: int = 800,
) -> go.Figure:
    """Visualize a heatmap of the topic's similarity matrix.

    Based on the cosine similarity matrix between topic embeddings (either c-TF-IDF or the embeddings from the embedding
    model), a heatmap is created showing the similarity between topics.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most frequent topics.
        n_clusters: Create n clusters and order the similarity
                    matrix by those clusters.
        use_ctfidf: Whether to calculate distances between topics based on c-TF-IDF embeddings. If False, the embeddings
                    from the embedding model are used.
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Returns:
        fig: A plotly figure

    Examples:
    To visualize the similarity matrix of
    topics simply run:

    ```python
    topic_model.visualize_heatmap()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_heatmap()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/heatmap.html"
    style="width:1000px; height: 720px; border: 0px;""></iframe>
    """
    embeddings = select_topic_representation(topic_model.c_tf_idf_, topic_model.topic_embeddings_, use_ctfidf)[0][
        topic_model._outliers :
    ]

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Order heatmap by similar clusters of topics
    sorted_topics = topics
    if n_clusters:
        if n_clusters >= len(set(topics)):
            raise ValueError("Make sure to set `n_clusters` lower than the total number of unique topics.")

        distance_matrix = cosine_similarity(embeddings[topics])
        Z = linkage(distance_matrix, "ward")
        clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

        # Extract new order of topics
        mapping = {cluster: [] for cluster in clusters}
        for topic, cluster in zip(topics, clusters):
            mapping[cluster].append(topic)
        mapping = [cluster for cluster in mapping.values()]
        sorted_topics = [topic for cluster in mapping for topic in cluster]

    # Select embeddings
    indices = np.array([topics.index(topic) for topic in sorted_topics])
    embeddings = embeddings[indices]
    distance_matrix = cosine_similarity(embeddings)

    # Create labels
    if isinstance(custom_labels, str):
        new_labels = [
            [[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in sorted_topics
        ]
        new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]
    elif topic_model.custom_labels_ is not None and custom_labels:
        new_labels = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in sorted_topics]
    else:
        new_labels = [[[str(topic), None]] + topic_model.get_topic(topic) for topic in sorted_topics]
        new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]

    fig = px.imshow(
        distance_matrix,
        labels=dict(color="Similarity Score"),
        x=new_labels,
        y=new_labels,
        color_continuous_scale="GnBu",
    )

    fig.update_layout(
        title={
            "text": f"{title}",
            "y": 0.95,
            "x": 0.55,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        width=width,
        height=height,
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
    )
    fig.update_layout(showlegend=True)
    fig.update_layout(legend_title_text="Trend")

    return fig
