from ._base import BaseEmbedder

# Imports for light-weight variant of BERTopic
from bertopic.backend._sklearn import SklearnEmbedder
from bertopic._utils import MyLogger
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as ScikitPipeline

logger = MyLogger()
logger.configure("WARNING")

languages = [
    "arabic",
    "bulgarian",
    "catalan",
    "czech",
    "danish",
    "german",
    "greek",
    "english",
    "spanish",
    "estonian",
    "persian",
    "finnish",
    "french",
    "canadian french",
    "galician",
    "gujarati",
    "hebrew",
    "hindi",
    "croatian",
    "hungarian",
    "armenian",
    "indonesian",
    "italian",
    "japanese",
    "georgian",
    "korean",
    "kurdish",
    "lithuanian",
    "latvian",
    "macedonian",
    "mongolian",
    "marathi",
    "malay",
    "burmese",
    "norwegian bokmal",
    "dutch",
    "polish",
    "portuguese",
    "brazilian portuguese",
    "romanian",
    "russian",
    "slovak",
    "slovenian",
    "albanian",
    "serbian",
    "swedish",
    "thai",
    "turkish",
    "ukrainian",
    "urdu",
    "vietnamese",
    "chinese (simplified)",
    "chinese (traditional)",
]


def select_backend(embedding_model, language: str = None, verbose: bool = False) -> BaseEmbedder:
    """Select an embedding model based on language or a specific provided model.
    When selecting a language, we choose all-MiniLM-L6-v2 for English and
    paraphrase-multilingual-MiniLM-L12-v2 for all other languages as it support 100+ languages.
    If sentence-transformers is not installed, in the case of a lightweight installation,
    a scikit-learn backend is default.

    Returns:
        model: The selected model backend.
    """
    logger.set_level("INFO" if verbose else "WARNING")

    # BERTopic language backend
    if isinstance(embedding_model, BaseEmbedder):
        return embedding_model

    # Scikit-learn backend
    if isinstance(embedding_model, ScikitPipeline):
        return SklearnEmbedder(embedding_model)

    # Flair word embeddings
    if "flair" in str(type(embedding_model)):
        from bertopic.backend._flair import FlairBackend

        return FlairBackend(embedding_model)

    # Spacy embeddings
    if "spacy" in str(type(embedding_model)):
        from bertopic.backend._spacy import SpacyBackend

        return SpacyBackend(embedding_model)

    # Gensim embeddings
    if "gensim" in str(type(embedding_model)):
        from bertopic.backend._gensim import GensimBackend

        return GensimBackend(embedding_model)

    # USE embeddings
    if "tensorflow" and "saved_model" in str(type(embedding_model)):
        from bertopic.backend._use import USEBackend

        return USEBackend(embedding_model)

    # Sentence Transformer embeddings
    if "sentence_transformers" in str(type(embedding_model)) or isinstance(embedding_model, str):
        from ._sentencetransformers import SentenceTransformerBackend

        return SentenceTransformerBackend(embedding_model)

    # Hugging Face embeddings
    if "transformers" and "pipeline" in str(type(embedding_model)):
        from ._hftransformers import HFTransformerBackend

        return HFTransformerBackend(embedding_model)

    # Model2Vec embeddings
    if "model2vec" in str(type(embedding_model)):
        from ._model2vec import Model2VecBackend

        return Model2VecBackend(embedding_model)

    # FastEmbed word embeddings
    if "fastembed" in str(type(embedding_model)):
        from bertopic.backend._fastembed import FastEmbedBackend

        return FastEmbedBackend(embedding_model)

    # Select embedding model based on language
    if language:
        try:
            from ._sentencetransformers import SentenceTransformerBackend

            if language.lower() in ["English", "english", "en"]:
                return SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")
            elif language.lower() in languages or language == "multilingual":
                return SentenceTransformerBackend("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            else:
                raise ValueError(
                    f"{language} is currently not supported. However, you can "
                    f"create any embeddings yourself and pass it through fit_transform(docs, embeddings)\n"
                    "Else, please select a language from the following list:\n"
                    f"{languages}"
                )

        # A ModuleNotFoundError might be a lightweight installation
        except ModuleNotFoundError as e:
            if e.name != "sentence_transformers":
                # Error occurred in a downstream module, probably not a lightweight install
                raise e
            # Whole sentence_transformers module is missing, probably a lightweight install
            if verbose:
                logger.info(
                    "Automatically selecting lightweight scikit-learn embedding backend as sentence-transformers appears to not be installed."
                )
            pipe = make_pipeline(TfidfVectorizer(), TruncatedSVD(100))
            return SklearnEmbedder(pipe)

    from ._sentencetransformers import SentenceTransformerBackend

    return SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")
