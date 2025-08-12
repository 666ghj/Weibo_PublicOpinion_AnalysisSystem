from ._base import BaseEmbedder
from ._word_doc import WordDocEmbedder
from ._utils import languages
from bertopic._utils import NotInstalled

# OpenAI Embeddings
try:
    from bertopic.backend._openai import OpenAIBackend
except ModuleNotFoundError:
    msg = "`pip install openai` \n\n"
    OpenAIBackend = NotInstalled("OpenAI", "OpenAI", custom_msg=msg)

# Cohere Embeddings
try:
    from bertopic.backend._cohere import CohereBackend
except ModuleNotFoundError:
    msg = "`pip install cohere` \n\n"
    CohereBackend = NotInstalled("Cohere", "Cohere", custom_msg=msg)

# Multimodal Embeddings
try:
    from bertopic.backend._multimodal import MultiModalBackend
except ModuleNotFoundError:
    msg = "`pip install bertopic[vision]` \n\n"
    MultiModalBackend = NotInstalled("Vision", "Vision", custom_msg=msg)

# Model2Vec Embeddings
try:
    from bertopic.backend._model2vec import Model2VecBackend
except ModuleNotFoundError:
    msg = "`pip install model2vec` \n\n"
    Model2VecBackend = NotInstalled("Model2Vec", "Model2Vec", custom_msg=msg)

# FasteEmbed Embeddings
try:
    from bertopic.backend._fastembed import FastEmbedBackend
except ModuleNotFoundError:
    msg = "`pip install fastembed` \n\n"
    FastEmbedBackend = NotInstalled("FastEmbed", "FastEmbed", custom_msg=msg)


# Langchain Embedddings
try:
    from bertopic.backend._langchain import LangChainBackend
except ModuleNotFoundError:
    msg = "`pip install langchain` \n\n"
    LangChainBackend = NotInstalled("LangChain", "LangChain", custom_msg=msg)


__all__ = [
    "BaseEmbedder",
    "WordDocEmbedder",
    "OpenAIBackend",
    "CohereBackend",
    "Model2VecBackend",
    "MultiModalBackend",
    "FastEmbedBackend",
    "LangChainBackend",
    "languages",
]
