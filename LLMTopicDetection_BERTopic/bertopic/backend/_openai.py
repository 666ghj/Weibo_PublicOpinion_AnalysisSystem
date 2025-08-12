import time
import openai
import numpy as np
from tqdm import tqdm
from typing import List, Mapping, Any
from bertopic.backend import BaseEmbedder


class OpenAIBackend(BaseEmbedder):
    """OpenAI Embedding Model.

    Arguments:
        client: A `openai.OpenAI` client.
        embedding_model: An OpenAI model. Default is
                         For an overview of models see:
                         https://platform.openai.com/docs/models/embeddings
        delay_in_seconds: If a `batch_size` is given, use this set
                          the delay in seconds between batches.
        batch_size: The size of each batch.
        generator_kwargs: Kwargs passed to `openai.Embedding.create`.
                          Can be used to define custom engines or
                          deployment_ids.

    Examples:
    ```python
    import openai
    from bertopic.backend import OpenAIBackend

    client = openai.OpenAI(api_key="sk-...")
    openai_embedder = OpenAIBackend(client, "text-embedding-ada-002")
    ```
    """

    def __init__(
        self,
        client: openai.OpenAI,
        embedding_model: str = "text-embedding-ada-002",
        delay_in_seconds: float = None,
        batch_size: int = None,
        generator_kwargs: Mapping[str, Any] = {},
    ):
        super().__init__()
        self.client = client
        self.embedding_model = embedding_model
        self.delay_in_seconds = delay_in_seconds
        self.batch_size = batch_size
        self.generator_kwargs = generator_kwargs

        if self.generator_kwargs.get("model"):
            self.embedding_model = generator_kwargs.get("model")
        elif not self.generator_kwargs.get("engine"):
            self.generator_kwargs["model"] = self.embedding_model

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
        # Prepare documents, replacing empty strings with a single space
        prepared_documents = [" " if doc == "" else doc for doc in documents]

        # Batch-wise embedding extraction
        if self.batch_size is not None:
            embeddings = []
            for batch in tqdm(self._chunks(prepared_documents), disable=not verbose):
                response = self.client.embeddings.create(input=batch, **self.generator_kwargs)
                embeddings.extend([r.embedding for r in response.data])

                # Delay subsequent calls
                if self.delay_in_seconds:
                    time.sleep(self.delay_in_seconds)

        # Extract embeddings all at once
        else:
            response = self.client.embeddings.create(input=prepared_documents, **self.generator_kwargs)
            embeddings = [r.embedding for r in response.data]
        return np.array(embeddings)

    def _chunks(self, documents):
        for i in range(0, len(documents), self.batch_size):
            yield documents[i : i + self.batch_size]
