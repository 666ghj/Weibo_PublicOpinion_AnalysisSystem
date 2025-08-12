import time
import openai
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any, Union, Callable
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import (
    retry_with_exponential_backoff,
    truncate_document,
    validate_truncate_document_parameters,
)


DEFAULT_CHAT_PROMPT = """You will extract a short topic label from given documents and keywords.
Here are two examples of topics you created before:

# Example 1
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the worst food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
topic: Environmental impacts of eating meat

# Example 2
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
topic: Shipping and delivery issues

# Your task
Sample texts from this topic:
[DOCUMENTS]

Keywords: [KEYWORDS]

Based on the information above, extract a short topic label (three words at most) in the following format:
topic: <topic_label>
"""

DEFAULT_SYSTEM_PROMPT = "You are an assistant that extracts high-level topics from texts."


class OpenAI(BaseRepresentation):
    r"""Using the OpenAI API to generate topic labels based
    on one of their Completion of ChatCompletion models.

    For an overview see:
    https://platform.openai.com/docs/models

    Arguments:
        client: A `openai.OpenAI` client
        model: Model to use within OpenAI, defaults to `"gpt-4o-mini"`.
        generator_kwargs: Kwargs passed to `openai.Completion.create`
                          for fine-tuning the output.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `self.default_system_prompt_` is used instead.
        delay_in_seconds: The delay in seconds between consecutive prompts
                          in order to prevent RateLimitErrors.
        exponential_backoff: Retry requests with a random exponential backoff.
                             A short sleep is used when a rate limit error is hit,
                             then the requests is retried. Increase the sleep length
                             if errors are hit until 10 unsuccessful requests.
                             If True, overrides `delay_in_seconds`.
        nr_docs: The number of documents to pass to OpenAI if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        diversity: The diversity of documents to pass to OpenAI.
                   Accepts values between 0 and 1. A higher
                   values results in passing more diverse documents
                   whereas lower values passes more similar documents.
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
                         and truncated depending on `doc_length`
                       * If tokenizer is a callable, then that callable is used to tokenize
                         the document. These tokens are counted and truncated depending
                         on `doc_length`

    Usage:

    To use this, you will need to install the openai package first:

    `pip install openai`

    Then, get yourself an API key and use OpenAI's API as follows:

    ```python
    import openai
    from bertopic.representation import OpenAI
    from bertopic import BERTopic

    # Create your representation model
    client = openai.OpenAI(api_key=MY_API_KEY)
    representation_model = OpenAI(client, delay_in_seconds=5)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS] \nThese documents are about the following topic: '"
    representation_model = OpenAI(client, prompt=prompt, delay_in_seconds=5)
    ```

    To choose a model:

    ```python
    representation_model = OpenAI(client, model="gpt-4o-mini", delay_in_seconds=10)
    ```
    """

    def __init__(
        self,
        client,
        model: str = "gpt-4o-mini",
        prompt: str = None,
        system_prompt: str = None,
        generator_kwargs: Mapping[str, Any] = {},
        delay_in_seconds: float = None,
        exponential_backoff: bool = False,
        nr_docs: int = 4,
        diversity: float = None,
        doc_length: int = None,
        tokenizer: Union[str, Callable] = None,
        **kwargs,
    ):
        self.client = client
        self.model = model

        if prompt is None:
            self.prompt = DEFAULT_CHAT_PROMPT
        else:
            self.prompt = prompt

        if system_prompt is None:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
        else:
            self.system_prompt = system_prompt

        self.default_prompt_ = DEFAULT_CHAT_PROMPT
        self.default_system_prompt_ = DEFAULT_SYSTEM_PROMPT
        self.delay_in_seconds = delay_in_seconds
        self.exponential_backoff = exponential_backoff
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer
        validate_truncate_document_parameters(self.tokenizer, self.doc_length)

        self.prompts_ = []

        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = generator_kwargs.get("model")
            del self.generator_kwargs["model"]
        if self.generator_kwargs.get("prompt"):
            del self.generator_kwargs["prompt"]
        if not self.generator_kwargs.get("stop"):
            self.generator_kwargs["stop"] = "\n"

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
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

        # Generate using OpenAI's Language Model
        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            truncated_docs = [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
            prompt = self._create_prompt(truncated_docs, topic, topics)
            self.prompts_.append(prompt)

            # Delay
            if self.delay_in_seconds:
                time.sleep(self.delay_in_seconds)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            kwargs = {
                "model": self.model,
                "messages": messages,
                **self.generator_kwargs,
            }
            if self.exponential_backoff:
                response = chat_completions_with_backoff(self.client, **kwargs)
            else:
                response = self.client.chat.completions.create(**kwargs)

            # Check whether content was actually generated
            # Addresses #1570 for potential issues with OpenAI's content filter
            # Addresses #2176 for potential issues when openAI returns a None type object
            if response and hasattr(response.choices[0].message, "content"):
                label = response.choices[0].message.content.strip().replace("topic: ", "")
            else:
                label = "No label returned"

            updated_topics[topic] = [(label, 1)]

        return updated_topics

    def _create_prompt(self, docs, topic, topics):
        keywords = list(zip(*topics[topic]))[0]

        # Use the Default Chat Prompt
        if self.prompt == DEFAULT_CHAT_PROMPT:
            prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
            prompt = self._replace_documents(prompt, docs)

        # Use a custom prompt that leverages keywords, documents or both using
        # custom tags, namely [KEYWORDS] and [DOCUMENTS] respectively
        else:
            prompt = self.prompt
            if "[KEYWORDS]" in prompt:
                prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
            if "[DOCUMENTS]" in prompt:
                prompt = self._replace_documents(prompt, docs)

        return prompt

    @staticmethod
    def _replace_documents(prompt, docs):
        to_replace = ""
        for doc in docs:
            to_replace += f"- {doc}\n"
        prompt = prompt.replace("[DOCUMENTS]", to_replace)
        return prompt


def chat_completions_with_backoff(client, **kwargs):
    return retry_with_exponential_backoff(
        client.chat.completions.create,
        errors=(openai.RateLimitError,),
    )(**kwargs)
