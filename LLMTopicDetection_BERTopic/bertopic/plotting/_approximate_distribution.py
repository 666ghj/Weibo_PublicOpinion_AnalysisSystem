import numpy as np
import pandas as pd

try:
    from pandas.io.formats.style import Styler  # noqa: F401

    HAS_JINJA = True
except (ModuleNotFoundError, ImportError):
    HAS_JINJA = False


def visualize_approximate_distribution(
    topic_model,
    document: str,
    topic_token_distribution: np.ndarray,
    normalize: bool = False,
):
    """Visualize the topic distribution calculated by `.approximate_topic_distribution`
    on a token level. Thereby indicating the extend to which a certain word or phrases belong
    to a specific topic. The assumption here is that a single word can belong to multiple
    similar topics and as such give information about the broader set of topics within
    a single document.

    Note:
    This function will return a stylized pandas dataframe if Jinja2 is installed. If not,
    it will only return a pandas dataframe without color highlighting. To install jinja:

    `pip install jinja2`

    Arguments:
        topic_model: A fitted BERTopic instance.
        document: The document for which you want to visualize
                  the approximated topic distribution.
        topic_token_distribution: The topic-token distribution of the document as
                                  extracted by `.approximate_topic_distribution`
        normalize: Whether to normalize, between 0 and 1 (summing to 1), the
                   topic distribution values.

    Returns:
        df: A stylized dataframe indicating the best fitting topics
            for each token.

    Examples:
    ```python
    # Calculate the topic distributions on a token level
    # Note that we need to have `calculate_token_level=True`
    topic_distr, topic_token_distr = topic_model.approximate_distribution(
            docs, calculate_token_level=True
    )

    # Visualize the approximated topic distributions
    df = topic_model.visualize_approximate_distribution(docs[0], topic_token_distr[0])
    df
    ```

    To revert this stylized dataframe back to a regular dataframe,
    you can run the following:

    ```python
    df.data.columns = [column.strip() for column in df.data.columns]
    df = df.data
    ```
    """
    # Tokenize document
    analyzer = topic_model.vectorizer_model.build_tokenizer()
    tokens = analyzer(document)

    if len(tokens) == 0:
        raise ValueError("Make sure that your document contains at least 1 token.")

    # Prepare dataframe with results
    if normalize:
        df = pd.DataFrame(topic_token_distribution / topic_token_distribution.sum()).T
    else:
        df = pd.DataFrame(topic_token_distribution).T

    df.columns = [f"{token}_{i}" for i, token in enumerate(tokens)]
    df.columns = [f"{token}{' ' * i}" for i, token in enumerate(tokens)]
    df.index = list(topic_model.topic_labels_.values())[topic_model._outliers :]
    df = df.loc[(df.sum(axis=1) != 0), :]

    # Style the resulting dataframe
    def text_color(val):
        color = "white" if val == 0 else "black"
        return "color: %s" % color

    def highligh_color(data, color="white"):
        attr = "background-color: {}".format(color)
        return pd.DataFrame(np.where(data == 0, attr, ""), index=data.index, columns=data.columns)

    if len(df) == 0:
        return df
    elif HAS_JINJA:
        df = (
            df.style.format("{:.3f}")
            .background_gradient(cmap="Blues", axis=None)
            .applymap(lambda x: text_color(x))
            .apply(highligh_color, axis=None)
        )
    return df
