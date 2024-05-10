# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""This module implements a Mech tool for binary predictions."""

from io import BytesIO
import PyPDF2
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Generator, List, Optional, Tuple, Callable
from pydantic import BaseModel, Field
from docstring_parser import parse
import tiktoken
from openai import OpenAI
import numpy as np
import faiss
import requests
from readability import Document as ReadabilityDocument
from markdownify import markdownify as md
from googleapiclient.discovery import build
from tiktoken import encoding_for_model

client: Optional[OpenAI] = None


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


class OpenAIClientManager:
    """Client context manager for OpenAI."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def __enter__(self) -> OpenAI:
        global client
        if client is None:
            client = OpenAI(api_key=self.api_key)
        return client

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global client
        if client is not None:
            client.close()
            client = None


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0,
}
MAX_TOKENS = {
    "gpt-3.5-turbo-0125": 4096,
    "gpt-4-0125-preview": 8192,
}
ALLOWED_TOOLS = [
    "resolve-market-reasoning-gpt-3.5-turbo",
    "resolve-market-reasoning-gpt-4",
]
TOOL_TO_ENGINE = {
    "resolve-market-reasoning-gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "resolve-market-reasoning-gpt-4": "gpt-4-0125-preview",
}
DEFAULT_NUM_WORDS: Dict[str, Optional[int]] = defaultdict(lambda: 300)
NUM_QUERIES = 3
NUM_URLS_PER_QUERY = 3
SPLITTER_CHUNK_SIZE = 1800
SPLITTER_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_BATCH_SIZE = 1000
EMBEDDING_SIZE = 3072
NUM_NEIGHBORS = 4
BUFFER_TOKENS = 250


class OpenAISchema(BaseModel):  # type: ignore[misc]
    @classmethod  # type: ignore[misc]
    @property
    def openai_schema(cls) -> Dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema
        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.
        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """
        schema = cls.model_json_schema()
        docstring = parse(cls.__doc__ or "")
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        for param in docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (
                    description := param.description
            ):
                if "description" not in parameters["properties"][name]:
                    parameters["properties"][name]["description"] = description

        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if "default" not in v
        )

        if "description" not in schema:
            if docstring.short_description:
                schema["description"] = docstring.short_description
            else:
                schema["description"] = (
                    f"Correctly extracted `{cls.__name__}` with all "
                    f"the required parameters with correct types"
                )

        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }

    @classmethod
    def from_response(cls, completion: Dict[str, Any]) -> "OpenAISchema":
        """
        Convert the response from OpenAI into the class instance
        Args:
            completion (dict): The response from OpenAI
        Returns:
            OpenAISchema: The instance of the class
        """

        message = completion.choices[0].message

        return cls.model_validate_json(
            message.function_call.arguments,
        )


class Queries(OpenAISchema):
    queries: List[str]


class Date(OpenAISchema):
    date_available: bool = Field(..., description="Whether the date is available")
    year: Optional[int] = Field(..., description="The year the article was published")
    month: Optional[str] = Field(..., description="The month the article was published")
    day: Optional[int] = Field(..., description="The day the article was published")


class Results(OpenAISchema):
    has_occurred: bool = Field(..., description="Whether the event has occurred.")


class Valid(OpenAISchema):
    is_valid: bool = Field(..., description="Whether the question is valid.")
    reason: Optional[str] = Field(..., description="Reason that the question is invalid.")


class Determinable(OpenAISchema):
    is_determinable: bool = Field(...,
                                  description="Whether it is possible to answer the question based on the information provided and reasoning.")


class Document(BaseModel):
    text: str
    date: str
    url: str
    embedding: Optional[List[float]] = None


URL_QUERY_PROMPT = """
 You are an expert fact checker in a team tasked with determining whether an event happened before a given date in the past. 
* Your role in the team to come up with search queries to be used to find relevant news articles that may help in determining whether the event occured. 
* You are provided with the input question about the event under the label "USER_PROMPT". 
* You must follow the instructions under the label "INSTRUCTIONS". 

INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" is a question about whether an event happened.
* The "USER_PROMPT" will contain a date which in the past.
* The event will only have has two possible outcomes: either the event has happened or the event has not happened.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You should come up with {num_queries} diverse queries to search for relevant news articles that may help in determining whether the event occured. 
* Focus on capturing different aspects and interpretations of the question to ensure comprehensive coverage of the topic.
* Make sure the queries are in past tense and are in the form of a question.
* ONLY function calls are allowed in the response.

USER_PROMPT:
```
{user_prompt}
```
"""

GET_DATE_PROMPT = """
INSTRUCTIONS
* You are an expert data analyst that takes in extracted text from a web search result. 
* You are provided with text extracted from a relevant web page under the label "EXTRACTED_TEXT" delimited by three backticks.
* Your task is to extract the date that the web page was published. 
* If there is no date information available, you should not try to guess. Instead indicate that it is not available.
* Your response should only be a function call with the extracted date information as arguments.

EXTRACTED_TEXT:
```
{extracted_text}
```
"""

PREDICTION_PROMPT = """
INSTRUCTIONS
* You are an expert data analyst. 
* You are provided with the input question about the event under the label "USER_PROMPT". 
* You are provided with a colleague's reasoning as to whether the event occurred based on online research under the label "REASONING" delimited by three backticks.
* Your task is to parse the decision on whether an event occurred.
* The answer that you give should match the answer that you come to in the reasoning field
* ONLY function calls are allowed in the response.

USER_PROMPT:
```
{user_prompt}
```

REASONING:
```
{reasoning}
```
"""

REASONING_PROMPT = """
You are an expert fact checker that takes in a question asking whether an event will happen before a given date. 
That date has now passed and your role is to determine whether the event actually happened before the date.
You are provided with the input question about the event under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS".

INSTRUCTIONS
* Read the input question under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" specifies a question about whether an event happened before a certain date.
* The date will has already passed, so you need to determine whether the event did or did not happen. There are only two
possible answers: either the event did happen or it did not happen.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION" delimited by three backticks.
* The items in "ADDITIONAL_INFORMATION" "ARTICLE (N), DATE: (MONTH/YEAR), URL: (URL), CONTENT: (CONTENT)"
* You can use any item in "ADDITIONAL_INFORMATION" in addition to your training data.
* If an item in "ADDITIONAL_INFORMATION" is not relevant, you must ignore that item for the estimation.
* Ideally, these will be news articles about the event in question.
* Pay special attention to the date of the article if it is available.
* You should show your process of thinking through the problem step by step, taking the date and information of the various articles into consideration, and explain your reasoning for your decision as to whether an event occurred by the specified date. 
* The articles will not always explicitly contain all the information needed to determine the answer. In this case, you may need to make an educated guess based on certain assumptions. If you need to do this, please provide your assumptions in your explanation.

Here are some examples of how you can figure out whether an event occurred by the date:
* If an article says that the event did happen and the date of the article is before the question date, then it is likely that the event did occur before the question date.
* If an article is talking about whether an event will happen and the date of the article is after the question date, then it is likely that the event did not happen before the question date.

USER_PROMPT:
```
{user_prompt}
```

ADDITIONAL_INFORMATION:
```
{formatted_docs}
```
"""

VALID_PROMPT = """
* You are an expert data analyst. 
* You are provided with a question about an event (submitted to a prediction market) under "USER_PROMPT" delimited by three backticks.
* Your task is to determine whether the question is valid.
* You are provided with rules that determine whether a question is invalid (as well as examples) under the label "RULES".
* Your response should only be a function call with the information about whether a question is valid and reason as arguments.

RULES
* Questions with relative dates should be marked as invalid. E.g. Invalid: Who will be the president of the United States in 6 months? (“in 6 months depends on the current time”).
* Questions about moral values and not facts should be marked as invalid. E.g. Invalid: “Is it ethical to eat meat?”.
* Questions in which none of the answers are valid should be marked as invalid. E.g. Invalid: “What is the result of 1+1?” with the outcomes “0” and “1”.
* Questions in which multiple answers are valid should be marked as invalid. E.g. Invalid: “Who will be the Time person of the year 1937?” with answers “Chiang Kai-shek” and “Soong Mei-ling” (they got the prize jointly).

USER_PROMPT:
```
{user_prompt}
```
"""

DETERMINABLE_PROMPT = """
* You are an expert data analyst. 
* You are provided with a question about an event (submitted to a prediction market) under "USER_PROMPT" delimited by three backticks.
* You are provided with a colleague's reasoning as to whether the event occurred based on online research under the label "REASONING" delimited by three backticks.
* Your task is to determine whether it is possible to answer whether the event actually happened before the date based on the content of this reasoning. 
* The answer that you give should reflect the opinion in the reasoning field.
* Your response should only be a function call with the information about whether a question is valid and reason as arguments.

USER_PROMPT:
```
{user_prompt}
```

REASONING:
```
{reasoning}
```

"""

SYSTEM_PROMPT = """You are a world class algorithm for generating structured output from a given input."""


def multi_queries(
        client: OpenAI,
        prompt: str,
        engine: str,
        num_queries: int,
        counter_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[str]:
    """Generate multiple queries for fetching information from the web."""

    url_query_prompt = URL_QUERY_PROMPT.format(
        user_prompt=prompt, num_queries=num_queries
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url_query_prompt},
    ]

    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=DEFAULT_OPENAI_SETTINGS["temperature"],
        max_tokens=DEFAULT_OPENAI_SETTINGS["max_tokens"],
        n=1,
        timeout=150,
        stop=None,
        functions=[Queries.openai_schema],
        function_call={'name': 'Queries'}
    )
    queries = Queries.from_response(response)

    # append the user's question to the list of queries
    queries.queries.append(prompt)

    if counter_callback:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=engine,
            token_counter=count_tokens,
        )
    return queries.queries, counter_callback


def search_google(query: str, api_key: str, engine: str, num: int) -> List[str]:
    service = build("customsearch", "v1", developerKey=api_key)
    search = (
        service.cse()
        .list(
            q=query,
            cx=engine,
            num=num,
        )
        .execute()
    )
    return [result["link"] for result in search["items"]]


def get_urls_from_queries(
        queries: List[str], api_key: str, engine: str, num: int
) -> List[str]:
    """Get URLs from search engine queries"""
    results = []
    for query in queries:
        try:
            for url in search_google(
                query=query,
                api_key=api_key,
                engine=engine,
                num=num,
            ):
                results.append(url)
        except Exception as e:
            print(f"An error occurred: {e}")
    unique_results = list(set(results))
    return unique_results


def get_dates(
        client: OpenAI,
        text: str,
        counter_callback: Optional[Callable[[int, int, str], None]] = None,
):
    """Get the date from the extracted text"""
    adjusted_text = adjust_additional_information(
        prompt=GET_DATE_PROMPT, additional_information=text, model="gpt-3.5-turbo-0125"
    )
    get_date_prompt = GET_DATE_PROMPT.format(extracted_text=adjusted_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_date_prompt},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        temperature=0,
        n=1,
        timeout=90,
        stop=None,
        functions=[Date.openai_schema],
        function_call={'name': 'Date'}
    )
    date = Date.from_response(response)
    if date.date_available:
        if counter_callback:
            counter_callback(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model="gpt-3.5-turbo-0125",
                token_counter=count_tokens,
            )
            return f"{date.year}-{date.month}-{date.day}", counter_callback
        return f"{date.year}-{date.month}-{date.day}", None
    return "Date not available", None


def extract_text_from_pdf(url: str, num_words: Optional[int] = None) -> str:
    """Extract text from a PDF document at the given URL."""
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        if "application/pdf" not in response.headers.get("Content-Type", ""):
            return ValueError("URL does not point to a PDF document")

        with BytesIO(response.content) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        doc = Document(text=text[:num_words] if num_words else text, date="", url=url)

        return doc
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_text(
        client: OpenAI,
        html: str,
        num_words: Optional[int] = None,
        counter_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """Extract text from a single HTML document"""
    text = ReadabilityDocument(html).summary()
    text = text = md(text, heading_style="ATX")
    date, counter_callback = get_dates(
        client=client, text=text, counter_callback=counter_callback
    )
    doc = Document(text=text[:num_words] if num_words else text, date=date, url="")
    return doc, counter_callback


def extract_texts(
        urls: List[str],
        client: OpenAI,
        counter_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Extract texts from URLs"""
    extracted_texts = []
    for batch in process_in_batches(urls=urls):
        for future, url in batch:
            try:
                if url.lower().endswith(".pdf"):
                    result = extract_text_from_pdf(url)
                    if result:
                        extracted_texts.append(result)
                    continue
                result = future.result()
                if result.status_code != 200:
                    continue
                # first 4 bytes is pdf
                if result.content[:4] == b"%PDF":
                    result = extract_text_from_pdf(url)
                    if result:
                        extracted_texts.append(result)
                    continue
                doc, counter_callback = extract_text(
                    html=result.text, client=client, counter_callback=counter_callback
                )
                doc.url = url
                extracted_texts.append(doc)
            except requests.exceptions.ReadTimeout:
                print(f"Request timed out: {url}.")
            except Exception as e:
                print(f"An error occurred: {e}")
    return extracted_texts, counter_callback


def process_in_batches(
        urls: List[str], window: int = 5, timeout: int = 50
) -> Generator[None, None, List[Tuple[Future, str]]]:
    """Iter URLs in batches."""
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(urls), window):
            batch = urls[i: i + window]
            futures = [
                (executor.submit(requests.get, url, timeout=timeout), url)
                for url in batch
            ]
            yield futures


def recursive_character_text_splitter(text, max_tokens, overlap):
    if len(text) <= max_tokens:
        return [text]
    else:
        return [
            text[i: i + max_tokens] for i in range(0, len(text), max_tokens - overlap)
        ]


def get_embeddings(split_docs: List[Document]) -> List[Document]:
    """Get embeddings for the split documents."""
    for batch_start in range(0, len(split_docs), EMBEDDING_BATCH_SIZE):
        batch_end = batch_start + EMBEDDING_BATCH_SIZE
        batch = [doc.text for doc in split_docs[batch_start:batch_end]]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        for i, be in enumerate(response.data):
            assert i == be.index
        batch_embeddings = [e.embedding for e in response.data]
        for i, doc in enumerate(split_docs[batch_start:batch_end]):
            doc.embedding = batch_embeddings[i]
    return split_docs


def find_similar_chunks(
        query: str, docs_with_embeddings: List[Document], k: int = 4
) -> List:
    """Similarity search to find similar chunks to a query"""

    query_embedding = (
        client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        .data[0]
        .embedding
    )

    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    index.add(np.array([doc.embedding for doc in docs_with_embeddings]))
    D, I = index.search(np.array([query_embedding]), k)

    return [docs_with_embeddings[i] for i in I[0]]


def fetch_additional_information(
        client: OpenAI,
        prompt: str,
        engine: str,
        google_api_key: Optional[str],
        google_engine_id: Optional[str],
        counter_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple:
    """Fetch additional information from the web."""

    # generate multiple queries for fetching information from the web
    queries, counter_callback = multi_queries(
        client=client,
        prompt=prompt,
        engine=engine,
        num_queries=NUM_QUERIES,
        counter_callback=counter_callback,
    )
    print(f"Queries: {queries}")

    # get the top URLs for the queries
    urls = get_urls_from_queries(
        queries=queries,
        api_key=google_api_key,
        engine=google_engine_id,
        num=NUM_URLS_PER_QUERY,
    )
    print(f"URLs: {urls}")

    # Extract text and dates from the URLs
    docs, counter_callback = extract_texts(
        urls=urls, client=client, counter_callback=counter_callback
    )

    # Remove None values from the list
    docs = [doc for doc in docs if doc]

    # remove doc with ""
    docs = [doc for doc in docs if hasattr(doc, "text") and doc.text != ""]

    # Chunk the documents
    split_docs = []
    for doc in docs:
        t = recursive_character_text_splitter(
            doc.text, SPLITTER_CHUNK_SIZE, SPLITTER_OVERLAP
        )
        split_docs.extend(
            [Document(text=chunk, date=doc.date, url=doc.url) for chunk in t]
        )
    print(f"Split Docs: {len(split_docs)}")

    # Remove None values from the list
    split_docs = [doc for doc in split_docs if doc]

    # Embed the documents
    docs_with_embeddings = get_embeddings(split_docs)
    print(f"Docs with embeddings: {len(docs_with_embeddings)}")

    # Find similar chunks
    similar_chunks = find_similar_chunks(
        query=prompt,
        docs_with_embeddings=docs_with_embeddings,
        k=NUM_NEIGHBORS,
    )
    print(f"Similar Chunks: {len(similar_chunks)}")

    # Format the additional information
    additional_information = "\n".join(
        [
            f"ARTICLE {i}, URL: {doc.url}, DATE: {doc.date}, CONTENT: {doc.text}\n"
            for i, doc in enumerate(similar_chunks)
        ]
    )

    return additional_information, queries, counter_callback


def adjust_additional_information(
        prompt: str, additional_information: str, model: str
) -> str:
    """Adjust the additional_information to fit within the token budget"""

    # Initialize tiktoken encoder for the specified model
    enc = tiktoken.encoding_for_model(model)

    # Encode the user prompt to calculate its token count
    prompt_tokens = len(enc.encode(prompt))

    # Calculate available tokens for additional_information
    MAX_PREDICTION_PROMPT_TOKENS = (
            MAX_TOKENS[model] - DEFAULT_OPENAI_SETTINGS["max_tokens"]
    )
    available_tokens = MAX_PREDICTION_PROMPT_TOKENS - prompt_tokens - BUFFER_TOKENS

    # Encode the additional_information
    additional_info_tokens = enc.encode(additional_information)

    # If additional_information exceeds available tokens, truncate it
    if len(additional_info_tokens) > available_tokens:
        truncated_info_tokens = additional_info_tokens[:available_tokens]
        # Decode tokens back to text, ensuring the output fits within the budget
        additional_information = enc.decode(truncated_info_tokens)

    return additional_information


def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        tool = kwargs["tool"]
        prompt = kwargs["prompt"]
        counter_callback = kwargs.get("counter_callback", None)
        api_keys = kwargs.get("api_keys", {})
        google_api_key = api_keys.get("google_api_key", None)
        google_engine_id = api_keys.get("google_engine_id", None)

        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"Tool {tool} is not supported.")

        engine = kwargs.get("model", TOOL_TO_ENGINE[tool])
        print(f"ENGINE: {engine}")

        # Check if question is valid
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": VALID_PROMPT.format(
                    user_prompt=prompt
                ),
            },
        ]

        response_valid = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=DEFAULT_OPENAI_SETTINGS["temperature"],
            max_tokens=DEFAULT_OPENAI_SETTINGS["max_tokens"],
            n=1,
            timeout=150,
            stop=None,
            functions=[Valid.openai_schema],
            function_call={'name': 'Valid'}
        )

        valid_results = Valid.from_response(response_valid)
        print(f"Valid: {valid_results}")

        if not valid_results.is_valid:
            return valid_results.json(), None, None, None

        (
            additional_information,
            queries,
            counter_callback,
        ) = fetch_additional_information(
            client=client,
            prompt=prompt,
            engine=engine,
            google_api_key=google_api_key,
            google_engine_id=google_engine_id,
            counter_callback=counter_callback,
        )

        # Adjust the additional_information to fit within the token budget
        adjusted_info = adjust_additional_information(
            prompt=PREDICTION_PROMPT,
            additional_information=additional_information,
            model=engine,
        )

        # Do reasoning
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": REASONING_PROMPT.format(
                    user_prompt=prompt, formatted_docs=adjusted_info
                ),
            },
        ]

        response_reasoning = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=DEFAULT_OPENAI_SETTINGS["temperature"],
            max_tokens=DEFAULT_OPENAI_SETTINGS["max_tokens"],
            n=1,
            timeout=150,
            stop=None,
        )

        reasoning = response_reasoning.choices[0].message.content

        # Check if question is determinable
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": DETERMINABLE_PROMPT.format(
                    user_prompt=prompt, reasoning=reasoning
                ),
            },
        ]

        response_determinable = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=DEFAULT_OPENAI_SETTINGS["temperature"],
            max_tokens=DEFAULT_OPENAI_SETTINGS["max_tokens"],
            n=1,
            timeout=150,
            stop=None,
            functions=[Determinable.openai_schema],
            function_call={'name': 'Determinable'}
        )

        determinable_results = Determinable.from_response(response_determinable)
        print(f"Determinable: {determinable_results}")

        if not determinable_results.is_determinable:
            return determinable_results.json(), reasoning, None, None

        # Make the prediction
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PREDICTION_PROMPT.format(
                    user_prompt=prompt, reasoning=reasoning
                ),
            },
        ]

        response_prediction = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=DEFAULT_OPENAI_SETTINGS["temperature"],
            max_tokens=DEFAULT_OPENAI_SETTINGS["max_tokens"],
            n=1,
            timeout=150,
            stop=None,
            functions=[Results.openai_schema],
            function_call={'name': 'Results'}
        )

        results = Results.from_response(response_prediction)
        print(f"Results: {results}")

        # Make the prediction
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PREDICTION_PROMPT.format(
                    user_prompt=prompt, reasoning=reasoning
                ),
            },
        ]

        response_prediction = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=DEFAULT_OPENAI_SETTINGS["temperature"],
            max_tokens=DEFAULT_OPENAI_SETTINGS["max_tokens"],
            n=1,
            timeout=150,
            stop=None,
            functions=[Results.openai_schema],
        )

        results = Results.from_response(response_prediction)
        print(f"Results: {results}")

        if counter_callback is not None:
            counter_callback(
                input_tokens=response_reasoning.usage.prompt_tokens
                             + response_prediction.usage.prompt_tokens,
                output_tokens=response_reasoning.usage.completion_tokens
                              + response_prediction.usage.completion_tokens,
                model=engine,
                token_counter=count_tokens,
                )
        return results.json(), reasoning, None, counter_callback

