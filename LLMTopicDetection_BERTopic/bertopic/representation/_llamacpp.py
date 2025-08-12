import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from llama_cpp import Llama
from typing import Mapping, List, Tuple, Any, Union, Callable
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document, validate_truncate_document_parameters


DEFAULT_PROMPT = """
This is a list of texts where each collection of texts describe a topic. After each collection of texts, the name of the topic they represent is mentioned as a short-highly-descriptive title
---
Topic:
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
Topic name: Environmental impacts of eating meat
---
Topic:
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
Topic name: Shipping and delivery issues
---
Topic:
Sample texts from this topic:
[DOCUMENTS]
Keywords: [KEYWORDS]
Topic name:"""

DEFAULT_SYSTEM_PROMPT = "You are an assistant that extracts high-level topics from texts."


class LlamaCPP(BaseRepresentation):
    """A llama.cpp implementation to use as a representation model.

    Arguments:
        model: Either a string pointing towards a local LLM or a
                `llama_cpp.Llama` object.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `self.default_system_prompt_` is used instead.
        pipeline_kwargs: Kwargs that you can pass to the `llama_cpp.Llama`
                         when it is called such as `max_tokens` to be generated.
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
                       * If tokenizer is 'whitespace', the the document is split up
                         into words separated by whitespaces. These words are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is 'vectorizer', then the internal CountVectorizer
                         is used to tokenize the document. These tokens are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is a callable, then that callable is used to tokenize
                         the document. These tokens are counted and truncated depending
                         on `doc_length`

    Usage:

    To use a llama.cpp, first download the LLM:

    ```bash
    wget https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/resolve/main/zephyr-7b-alpha.Q4_K_M.gguf
    ```

    Then, we can now use the model the model with BERTopic in just a couple of lines:

    ```python
    from bertopic import BERTopic
    from bertopic.representation import LlamaCPP

    # Use llama.cpp to load in a 4-bit quantized version of Zephyr 7B Alpha
    representation_model = LlamaCPP("zephyr-7b-alpha.Q4_K_M.gguf")

    # Create our BERTopic model
    topic_model = BERTopic(representation_model=representation_model, verbose=True)
    ```

    If you want to have more control over the LLMs parameters, you can run it like so:

    ```python
    from bertopic import BERTopic
    from bertopic.representation import LlamaCPP
    from llama_cpp import Llama

    # Use llama.cpp to load in a 4-bit quantized version of Zephyr 7B Alpha
    llm = Llama(model_path="zephyr-7b-alpha.Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=4096, stop="Q:")
    representation_model = LlamaCPP(llm)

    # Create our BERTopic model
    topic_model = BERTopic(representation_model=representation_model, verbose=True)
    ```
    """

    def __init__(
        self,
        model: Union[str, Llama],
        prompt: str = None,
        system_prompt: str = None,
        pipeline_kwargs: Mapping[str, Any] = {},
        nr_docs: int = 4,
        diversity: float = None,
        doc_length: int = None,
        tokenizer: Union[str, Callable] = None,
    ):
        if isinstance(model, str):
            self.model = Llama(model_path=model, n_gpu_layers=-1, stop="\n", chat_format="ChatML")
        elif isinstance(model, Llama):
            self.model = model
        else:
            raise ValueError(
                "Make sure that the model that you"
                "pass is either a string referring to a"
                "local LLM or a ` llama_cpp.Llama` object."
            )
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
        self.default_system_prompt_ = DEFAULT_SYSTEM_PROMPT
        self.pipeline_kwargs = pipeline_kwargs
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer
        validate_truncate_document_parameters(self.tokenizer, self.doc_length)

        self.prompts_ = []

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topic representations and return a single label.

        Arguments:
            topic_model: A BERTopic model
            documents: Not used
            c_tf_idf: Not used
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # Extract the top 4 representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            # Prepare prompt
            truncated_docs = [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
            prompt = self._create_prompt(truncated_docs, topic, topics)
            self.prompts_.append(prompt)

            # Extract result from generator and use that as label
            # topic_description = self.model(prompt, **self.pipeline_kwargs)["choices"]
            topic_description = self.model.create_chat_completion(
                messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                **self.pipeline_kwargs,
            )
            label = topic_description["choices"][0]["message"]["content"].strip()
            updated_topics[topic] = [(label, 1)] + [("", 0) for _ in range(9)]

        return updated_topics

    def _create_prompt(self, docs, topic, topics):
        keywords = list(zip(*topics[topic]))[0]

        # Use the Default Chat Prompt
        if self.prompt == DEFAULT_PROMPT:
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
