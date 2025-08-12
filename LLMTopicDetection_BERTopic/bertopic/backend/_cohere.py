import time
import numpy as np
from tqdm import tqdm
from typing import Any, List, Mapping
from bertopic.backend import BaseEmbedder


class CohereBackend(BaseEmbedder):
    """Cohere Embedding Model.

    Arguments:
        client: A `cohere` client.
        embedding_model: A Cohere model. Default is "large".
                         For an overview of models see:
                         https://docs.cohere.ai/docs/generation-card
        delay_in_seconds: If a `batch_size` is given, use this set
                          the delay in seconds between batches.
        batch_size: The size of each batch.
        embed_kwargs: Kwargs passed to `cohere.Client.embed`.
                            Can be used to define additional parameters
                            such as `input_type`

    Examples:
    ```python
    import cohere
    from bertopic.backend import CohereBackend

    client = cohere.Client("APIKEY")
    cohere_model = CohereBackend(client)
    ```

    If you want to specify `input_type`:

    ```python
    cohere_model = CohereBackend(
        client,
        embedding_model="embed-english-v3.0",
        embed_kwargs={"input_type": "clustering"}
    )
    ```
    """

    def __init__(
        self,
        client,
        embedding_model: str = "large",
        delay_in_seconds: float = None,
        batch_size: int = None,
        embed_kwargs: Mapping[str, Any] = {},
    ):
        super().__init__()
        self.client = client
        self.embedding_model = embedding_model
        self.delay_in_seconds = delay_in_seconds
        self.batch_size = batch_size
        self.embed_kwargs = embed_kwargs

        if self.embed_kwargs.get("model"):
            self.embedding_model = embed_kwargs.get("model")
        else:
            self.embed_kwargs["model"] = self.embedding_model

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
        # Batch-wise embedding extraction
        if self.batch_size is not None:
            embeddings = []
            for batch in tqdm(self._chunks(documents), disable=not verbose):
                response = self.client.embed(texts=batch, **self.embed_kwargs)
                embeddings.extend(response.embeddings)

                # Delay subsequent calls
                if self.delay_in_seconds:
                    time.sleep(self.delay_in_seconds)

        # Extract embeddings all at once
        else:
            response = self.client.embed(texts=documents, **self.embed_kwargs)
            embeddings = response.embeddings
        return np.array(embeddings)

    def _chunks(self, documents):
        for i in range(0, len(documents), self.batch_size):
            yield documents[i : i + self.batch_size]
