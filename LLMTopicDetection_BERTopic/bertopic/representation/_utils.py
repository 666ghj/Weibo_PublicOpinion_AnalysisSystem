import random
import time
from typing import Union


def truncate_document(topic_model, doc_length: Union[int, None], tokenizer: Union[str, callable], document: str) -> str:
    """Truncate a document to a certain length.

    If you want to add a custom tokenizer, then it will need to have a `decode` and
    `encode` method. An example would be the following custom tokenizer:

    ```python
    class Tokenizer:
        'A custom tokenizer that splits on commas'
        def encode(self, doc):
            return doc.split(",")

        def decode(self, doc_chunks):
            return ",".join(doc_chunks)
    ```

    You can use this tokenizer by passing it to the `tokenizer` parameter.

    Arguments:
        topic_model: A BERTopic model
        doc_length: The maximum length of each document. If a document is longer,
                    it will be truncated. If None, the entire document is passed.
        tokenizer: The tokenizer used to calculate to split the document into segments
                   used to count the length of a document.
                       * If tokenizer is 'char', then the document is split up
                         into characters which are counted to adhere to `doc_length`
                       * If tokenizer is 'whitespace', the document is split up
                         into words separated by whitespaces. These words are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is 'vectorizer', then the internal CountVectorizer
                         is used to tokenize the document. These tokens are counted
                         and truncated depending on `doc_length`. They are decoded with
                         whitespaces.
                       * If tokenizer is a callable, then that callable is used to tokenize
                         the document. These tokens are counted and truncated depending
                         on `doc_length`
        document: A single document

    Returns:
        truncated_document: A truncated document
    """
    if doc_length is not None:
        if tokenizer == "char":
            truncated_document = document[:doc_length]
        elif tokenizer == "whitespace":
            truncated_document = " ".join(document.split()[:doc_length])
        elif tokenizer == "vectorizer":
            tokenizer = topic_model.vectorizer_model.build_tokenizer()
            truncated_document = " ".join(tokenizer(document)[:doc_length])
        elif hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
            encoded_document = tokenizer.encode(document)
            truncated_document = tokenizer.decode(encoded_document[:doc_length])
        return truncated_document
    return document


def validate_truncate_document_parameters(tokenizer, doc_length) -> Union[None, ValueError]:
    """Validates parameters that are used in the function `truncate_document`."""
    if tokenizer is None and doc_length is not None:
        raise ValueError(
            "Please select from one of the valid options for the `tokenizer` parameter: \n"
            "{'char', 'whitespace', 'vectorizer'} \n"
            "If `tokenizer` is of type callable ensure it has methods to encode and decode a document \n"
        )
    elif tokenizer is not None and doc_length is None:
        raise ValueError("If `tokenizer` is provided, `doc_length` of type int must be provided as well.")


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = None,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper
