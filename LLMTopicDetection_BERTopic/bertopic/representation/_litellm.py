import time
from litellm import completion
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import retry_with_exponential_backoff


DEFAULT_PROMPT = """
I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]
Based on the information above, extract a short topic label in the following format:
topic: <topic label>
"""


class LiteLLM(BaseRepresentation):
    """Using the LiteLLM API to generate topic labels.

    For an overview of models see:
    https://docs.litellm.ai/docs/providers

    Arguments:
        model: Model to use. Defaults to OpenAI's "gpt-3.5-turbo".
        generator_kwargs: Kwargs passed to `litellm.completion`.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        delay_in_seconds: The delay in seconds between consecutive prompts
                          in order to prevent RateLimitErrors.
        exponential_backoff: Retry requests with a random exponential backoff.
                             A short sleep is used when a rate limit error is hit,
                             then the requests is retried. Increase the sleep length
                             if errors are hit until 10 unsuccesfull requests.
                             If True, overrides `delay_in_seconds`.
        nr_docs: The number of documents to pass to LiteLLM if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        diversity: The diversity of documents to pass to LiteLLM.
                   Accepts values between 0 and 1. A higher
                   values results in passing more diverse documents
                   whereas lower values passes more similar documents.

    Usage:

    To use this, you will need to install the litellm package first:

    `pip install litellm`

    Then, get yourself an API key of any provider (for instance OpenAI) and use it as follows:

    ```python
    import os
    from bertopic.representation import LiteLLM
    from bertopic import BERTopic

    # set ENV variables
    os.environ["OPENAI_API_KEY"] = "your-openai-key"

    # Create your representation model
    representation_model = LiteLLM(model="gpt-3.5-turbo")

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS] \nThese documents are about the following topic: '"
    representation_model = LiteLLM(model="gpt", prompt=prompt)
    ```
    """  # noqa: D301

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        prompt: str = None,
        generator_kwargs: Mapping[str, Any] = {},
        delay_in_seconds: float = None,
        exponential_backoff: bool = False,
        nr_docs: int = 4,
        diversity: float = None,
    ):
        self.model = model
        self.prompt = prompt if prompt else DEFAULT_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
        self.delay_in_seconds = delay_in_seconds
        self.exponential_backoff = exponential_backoff
        self.nr_docs = nr_docs
        self.diversity = diversity

        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = generator_kwargs.get("model")
        if self.generator_kwargs.get("prompt"):
            del self.generator_kwargs["prompt"]

    def extract_topics(
        self, topic_model, documents: pd.DataFrame, c_tf_idf: csr_matrix, topics: Mapping[str, List[Tuple[str, float]]]
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics.

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # Extract the top n representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        # Generate using a (Large) Language Model
        updated_topics = {}
        for topic, docs in repr_docs_mappings.items():
            prompt = self._create_prompt(docs, topic, topics)

            # Delay
            if self.delay_in_seconds:
                time.sleep(self.delay_in_seconds)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            kwargs = {"model": self.model, "messages": messages, **self.generator_kwargs}
            if self.exponential_backoff:
                response = chat_completions_with_backoff(**kwargs)
            else:
                response = completion(**kwargs)
            label = response["choices"][0]["message"]["content"].strip().replace("topic: ", "")

            updated_topics[topic] = [(label, 1)]

        return updated_topics

    def _create_prompt(self, docs, topic, topics):
        keywords = list(zip(*topics[topic]))[0]

        # Use the Default Chat Prompt
        if self.prompt == DEFAULT_PROMPT:
            prompt = self.prompt.replace("[KEYWORDS]", " ".join(keywords))
            prompt = self._replace_documents(prompt, docs)

        # Use a custom prompt that leverages keywords, documents or both using
        # custom tags, namely [KEYWORDS] and [DOCUMENTS] respectively
        else:
            prompt = self.prompt
            if "[KEYWORDS]" in prompt:
                prompt = prompt.replace("[KEYWORDS]", " ".join(keywords))
            if "[DOCUMENTS]" in prompt:
                prompt = self._replace_documents(prompt, docs)

        return prompt

    @staticmethod
    def _replace_documents(prompt, docs):
        to_replace = ""
        for doc in docs:
            to_replace += f"- {doc[:255]}\n"
        prompt = prompt.replace("[DOCUMENTS]", to_replace)
        return prompt


def chat_completions_with_backoff(**kwargs):
    return retry_with_exponential_backoff(
        completion,
    )(**kwargs)
