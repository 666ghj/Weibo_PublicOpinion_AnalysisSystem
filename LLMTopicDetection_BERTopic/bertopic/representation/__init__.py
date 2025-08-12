from bertopic._utils import NotInstalled
from bertopic.representation._cohere import Cohere
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._keybert import KeyBERTInspired
from bertopic.representation._mmr import MaximalMarginalRelevance


# Llama CPP Generator
try:
    from bertopic.representation._llamacpp import LlamaCPP
except ModuleNotFoundError:
    msg = "`pip install llama-cpp-python` \n\n"
    LlamaCPP = NotInstalled("llama.cpp", "llama-cpp-python", custom_msg=msg)

# Text Generation using transformers
try:
    from bertopic.representation._textgeneration import TextGeneration
except ModuleNotFoundError:
    msg = "`pip install bertopic` without `--no-deps` \n\n"
    TextGeneration = NotInstalled("TextGeneration", "transformers", custom_msg=msg)

# Zero-shot classification using transformers
try:
    from bertopic.representation._zeroshot import ZeroShotClassification
except ModuleNotFoundError:
    msg = "`pip install bertopic` without `--no-deps` \n\n"
    ZeroShotClassification = NotInstalled("ZeroShotClassification", "transformers", custom_msg=msg)

# OpenAI Generator
try:
    from bertopic.representation._openai import OpenAI
except ModuleNotFoundError:
    msg = "`pip install openai` \n\n"
    OpenAI = NotInstalled("OpenAI", "openai", custom_msg=msg)

# LiteLLM Generator
try:
    from bertopic.representation._litellm import LiteLLM
except ModuleNotFoundError:
    msg = "`pip install litellm` \n\n"
    LiteLLM = NotInstalled("LiteLLM", "litellm", custom_msg=msg)

# LangChain Generator
try:
    from bertopic.representation._langchain import LangChain
except ModuleNotFoundError:
    msg = "`pip install langchain` \n\n"
    LangChain = NotInstalled("langchain", "langchain", custom_msg=msg)

# POS using Spacy
try:
    from bertopic.representation._pos import PartOfSpeech
except ModuleNotFoundError:
    PartOfSpeech = NotInstalled("Part of Speech with Spacy", "spacy")

# Multimodal
try:
    from bertopic.representation._visual import VisualRepresentation
except ModuleNotFoundError:
    VisualRepresentation = NotInstalled("a visual representation model", "vision")


__all__ = [
    "BaseRepresentation",
    "TextGeneration",
    "ZeroShotClassification",
    "KeyBERTInspired",
    "PartOfSpeech",
    "MaximalMarginalRelevance",
    "Cohere",
    "OpenAI",
    "LangChain",
    "LiteLLM",
    "LlamaCPP",
    "VisualRepresentation",
]
