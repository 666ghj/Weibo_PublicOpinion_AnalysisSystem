from importlib.metadata import version

from bertopic._bertopic import BERTopic

__version__ = version("bertopic")

__all__ = [
    "BERTopic",
]
