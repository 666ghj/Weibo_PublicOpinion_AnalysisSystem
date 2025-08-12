import numpy as np

from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from transformers.pipelines import Pipeline

from bertopic.backend import BaseEmbedder


class HFTransformerBackend(BaseEmbedder):
    """Hugging Face transformers model.

    This uses the `transformers.pipelines.pipeline` to define and create
    a feature generation pipeline from which embeddings can be extracted.

    Arguments:
        embedding_model: A Hugging Face feature extraction pipeline

    Examples:
    To use a Hugging Face transformers model, load in a pipeline and point
    to any model found on their model hub (https://huggingface.co/models):

    ```python
    from bertopic.backend import HFTransformerBackend
    from transformers.pipelines import pipeline

    hf_model = pipeline("feature-extraction", model="distilbert-base-cased")
    embedding_model = HFTransformerBackend(hf_model)
    ```
    """

    def __init__(self, embedding_model: Pipeline):
        super().__init__()

        if isinstance(embedding_model, Pipeline):
            self.embedding_model = embedding_model
        else:
            raise ValueError(
                "Please select a correct transformers pipeline. For example: "
                "pipeline('feature-extraction', model='distilbert-base-cased', device=0)"
            )

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n documents/words into an n-dimensional
        matrix of embeddings.

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        dataset = MyDataset(documents)

        embeddings = []
        for document, features in tqdm(
            zip(documents, self.embedding_model(dataset, truncation=True, padding=True)),
            total=len(dataset),
            disable=not verbose,
        ):
            embeddings.append(self._embed(document, features))

        return np.array(embeddings)

    def _embed(self, document: str, features: np.ndarray) -> np.ndarray:
        """Mean pooling.

        Arguments:
            document: The document for which to extract the attention mask
            features: The embeddings for each token

        Adopted from:
        https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2#usage-huggingface-transformers
        """
        token_embeddings = np.array(features)
        attention_mask = self.embedding_model.tokenizer(document, truncation=True, padding=True, return_tensors="np")[
            "attention_mask"
        ]
        input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), token_embeddings.shape)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = np.clip(
            input_mask_expanded.sum(1),
            a_min=1e-9,
            a_max=input_mask_expanded.sum(1).max(),
        )
        embedding = normalize(sum_embeddings / sum_mask)[0]
        return embedding


class MyDataset(Dataset):
    """Dataset to pass to `transformers.pipelines.pipeline`."""

    def __init__(self, docs):
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx]
