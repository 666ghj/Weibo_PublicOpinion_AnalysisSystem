import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from transformers import pipeline, set_seed
from transformers.pipelines.base import Pipeline
from typing import Mapping, List, Tuple, Any, Union, Callable
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document, validate_truncate_document_parameters


DEFAULT_PROMPT = """
I have a topic described by the following keywords: [KEYWORDS].
The name of this topic is:
"""


class TextGeneration(BaseRepresentation):
    """Text2Text or text generation with transformers.

    Arguments:
        model: A transformers pipeline that should be initialized as "text-generation"
               for gpt-like models or "text2text-generation" for T5-like models.
               For example, `pipeline('text-generation', model='gpt2')`. If a string
               is passed, "text-generation" will be selected by default.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        pipeline_kwargs: Kwargs that you can pass to the transformers.pipeline
                         when it is called.
        random_state: A random state to be passed to `transformers.set_seed`
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

    To use a gpt-like model:

    ```python
    from bertopic.representation import TextGeneration
    from bertopic import BERTopic

    # Create your representation model
    generator = pipeline('text-generation', model='gpt2')
    representation_model = TextGeneration(generator)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTo pic(representation_model=representation_model)
    ```

    You can use a custom prompt and decide where the keywords should
    be inserted by using the `[KEYWORDS]` or documents with thte `[DOCUMENTS]` tag:

    ```python
    from bertopic.representation import TextGeneration

    prompt = "I have a topic described by the following keywords: [KEYWORDS]. Based on the previous keywords, what is this topic about?""

    # Create your representation model
    generator = pipeline('text2text-generation', model='google/flan-t5-base')
    representation_model = TextGeneration(generator)
    ```
    """

    def __init__(
        self,
        model: Union[str, pipeline],
        prompt: str = None,
        pipeline_kwargs: Mapping[str, Any] = {},
        random_state: int = 42,
        nr_docs: int = 4,
        diversity: float = None,
        doc_length: int = None,
        tokenizer: Union[str, Callable] = None,
    ):
        self.random_state = random_state
        set_seed(random_state)
        if isinstance(model, str):
            self.model = pipeline("text-generation", model=model)
        elif isinstance(model, Pipeline):
            self.model = model
        else:
            raise ValueError(
                "Make sure that the HF model that you"
                "pass is either a string referring to a"
                "HF model or a `transformers.pipeline` object."
            )
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
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
        if self.prompt != DEFAULT_PROMPT and "[DOCUMENTS]" in self.prompt:
            repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
                c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
            )
        else:
            repr_docs_mappings = {topic: None for topic in topics.keys()}

        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            # Prepare prompt
            truncated_docs = (
                [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
                if docs is not None
                else docs
            )
            prompt = self._create_prompt(truncated_docs, topic, topics)
            self.prompts_.append(prompt)

            # Extract result from generator and use that as label
            topic_description = self.model(prompt, **self.pipeline_kwargs)
            topic_description = [
                (description["generated_text"].replace(prompt, ""), 1) for description in topic_description
            ]

            if len(topic_description) < 10:
                topic_description += [("", 0) for _ in range(10 - len(topic_description))]

            updated_topics[topic] = topic_description

        return updated_topics

    def _create_prompt(self, docs, topic, topics):
        keywords = ", ".join(list(zip(*topics[topic]))[0])

        # Use the default prompt and replace keywords
        if self.prompt == DEFAULT_PROMPT:
            prompt = self.prompt.replace("[KEYWORDS]", keywords)

        # Use a prompt that leverages either keywords or documents in
        # a custom location
        else:
            prompt = self.prompt
            if "[KEYWORDS]" in prompt:
                prompt = prompt.replace("[KEYWORDS]", keywords)
            if "[DOCUMENTS]" in prompt:
                to_replace = ""
                for doc in docs:
                    to_replace += f"- {doc}\n"
                prompt = prompt.replace("[DOCUMENTS]", to_replace)

        return prompt
