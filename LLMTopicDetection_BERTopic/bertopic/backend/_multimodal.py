import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Union
from sentence_transformers import SentenceTransformer

from bertopic.backend import BaseEmbedder


class MultiModalBackend(BaseEmbedder):
    """Multimodal backend using Sentence-transformers.

    The sentence-transformers embedding model used for
    generating word, document, and image embeddings.

    Arguments:
        embedding_model: A sentence-transformers embedding model that
                         can either embed both images and text or only text.
                         If it only embeds text, then `image_model` needs
                         to be used to embed the images.
        image_model: A sentence-transformers embedding model that is used
                     to embed only images.
        batch_size: The sizes of image batches to pass

    Examples:
    To create a model, you can load in a string pointing to a
    sentence-transformers model:

    ```python
    from bertopic.backend import MultiModalBackend

    sentence_model = MultiModalBackend("clip-ViT-B-32")
    ```

    or  you can instantiate a model yourself:
    ```python
    from bertopic.backend import MultiModalBackend
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("clip-ViT-B-32")
    sentence_model = MultiModalBackend(embedding_model)
    ```
    """

    def __init__(
        self,
        embedding_model: Union[str, SentenceTransformer],
        image_model: Union[str, SentenceTransformer] = None,
        batch_size: int = 32,
    ):
        super().__init__()
        self.batch_size = batch_size

        # Text or Text+Image model
        if isinstance(embedding_model, SentenceTransformer):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            raise ValueError(
                "Please select a correct SentenceTransformers model: \n"
                "`from sentence_transformers import SentenceTransformer` \n"
                "`model = SentenceTransformer('clip-ViT-B-32')`"
            )

        # Image Model
        self.image_model = None
        if image_model is not None:
            if isinstance(image_model, SentenceTransformer):
                self.image_model = image_model
            elif isinstance(image_model, str):
                self.image_model = SentenceTransformer(image_model)
            else:
                raise ValueError(
                    "Please select a correct SentenceTransformers model: \n"
                    "`from sentence_transformers import SentenceTransformer` \n"
                    "`model = SentenceTransformer('clip-ViT-B-32')`"
                )

        try:
            self.tokenizer = self.embedding_model._first_module().processor.tokenizer
        except AttributeError:
            self.tokenizer = self.embedding_model.tokenizer
        except:  # noqa: E722
            self.tokenizer = None

    def embed(self, documents: List[str], images: List[str] = None, verbose: bool = False) -> np.ndarray:
        """Embed a list of n documents/words or images into an n-dimensional
        matrix of embeddings.

        Either documents, images, or both can be provided. If both are provided,
        then the embeddings are averaged.

        Arguments:
            documents: A list of documents or words to be embedded
            images: A list of image paths to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        # Embed documents
        doc_embeddings = None
        if documents[0] is not None:
            doc_embeddings = self.embed_documents(documents)

        # Embed images
        image_embeddings = None
        if isinstance(images, list):
            image_embeddings = self.embed_images(images, verbose)

        # Average embeddings
        averaged_embeddings = None
        if doc_embeddings is not None and image_embeddings is not None:
            averaged_embeddings = np.mean([doc_embeddings, image_embeddings], axis=0)

        if averaged_embeddings is not None:
            return averaged_embeddings
        elif doc_embeddings is not None:
            return doc_embeddings
        elif image_embeddings is not None:
            return image_embeddings

    def embed_documents(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n documents/words into an n-dimensional
        matrix of embeddings.

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        truncated_docs = [self._truncate_document(doc) for doc in documents]
        embeddings = self.embedding_model.encode(truncated_docs, show_progress_bar=verbose)
        return embeddings

    def embed_words(self, words: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n words into an n-dimensional
        matrix of embeddings.

        Arguments:
            words: A list of words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        embeddings = self.embedding_model.encode(words, show_progress_bar=verbose)
        return embeddings

    def embed_images(self, images, verbose):
        if self.batch_size:
            nr_iterations = int(np.ceil(len(images) / self.batch_size))

            # Embed images per batch
            embeddings = []
            for i in tqdm(range(nr_iterations), disable=not verbose):
                start_index = i * self.batch_size
                end_index = (i * self.batch_size) + self.batch_size

                images_to_embed = [
                    Image.open(image) if isinstance(image, str) else image for image in images[start_index:end_index]
                ]
                if self.image_model is not None:
                    img_emb = self.image_model.encode(images_to_embed)
                else:
                    img_emb = self.embedding_model.encode(images_to_embed, show_progress_bar=False)
                embeddings.extend(img_emb.tolist())

                # Close images
                if isinstance(images[0], str):
                    for image in images_to_embed:
                        image.close()
            embeddings = np.array(embeddings)
        else:
            images_to_embed = [Image.open(filepath) for filepath in images]
            if self.image_model is not None:
                embeddings = self.image_model.encode(images_to_embed)
            else:
                embeddings = self.embedding_model.encode(images_to_embed, show_progress_bar=False)
        return embeddings

    def _truncate_document(self, document):
        if self.tokenizer:
            tokens = self.tokenizer.encode(document)

            if len(tokens) > 77:
                # Skip the starting token, only include 75 tokens
                truncated_tokens = tokens[1:76]
                document = self.tokenizer.decode(truncated_tokens)

                # Recursive call here, because the encode(decode()) can have different result
                return self._truncate_document(document)

        return document
