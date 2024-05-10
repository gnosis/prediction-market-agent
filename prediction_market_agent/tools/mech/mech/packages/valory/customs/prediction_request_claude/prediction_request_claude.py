# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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

import json
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Iterator, Callable
from itertools import islice

import anthropic
import requests
from readability import Document
from tiktoken import encoding_for_model
from markdownify import markdownify as md
from googleapiclient.discovery import build

NUM_URLS_EXTRACT = 5
DEFAULT_NUM_WORDS = 300
DEFAULT_CLAUDE_SETTINGS = {
    "max_tokens": 1000,
    "temperature": 0,
}
MAX_TOKENS = {
    'claude-2': 200_0000,
    'claude-2.1': 200_0000,
    'claude-3-haiku-20240307': 200_0000,
    'claude-3-sonnet-20240229': 200_0000,
    'claude-3-opus-20240229': 200_0000,
}
ALLOWED_TOOLS = [
    "claude-prediction-offline",
    "claude-prediction-online",
]
ALLOWED_MODELS = [
    "claude-2",
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
]
TOOL_TO_ENGINE = {
    "claude-prediction-offline": "claude-3-sonnet-20240229",
    "claude-prediction-online": "claude-3-sonnet-20240229",
}

PREDICTION_PROMPT = """
You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
for a given event. You are provided with an input under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" specifies an event.
* The event will only have two possible outcomes: either the event will happen or the event will not happen.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You must provide a probability estimation of the event happening, based on your training data.
* You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION" delimited by three backticks.
* You can use any item in "ADDITIONAL_INFORMATION" in addition to your training data.
* If an item in "ADDITIONAL_INFORMATION" is not relevant, you must ignore that item for the estimation.
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.

USER_PROMPT:
```
{user_prompt}
```

ADDITIONAL_INFORMATION:
```
{additional_information}
```

OUTPUT_FORMAT
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain four fields: "p_yes", "p_no", "confidence", and "info_utility".
* Each item in the JSON must have a value between 0 and 1.
   - "p_yes": Estimated probability that the event in the "USER_PROMPT" occurs.
   - "p_no": Estimated probability that the event in the "USER_PROMPT" does not occur.
   - "confidence": A value between 0 and 1 indicating the confidence in the prediction. 0 indicates lowest
     confidence value; 1 maximum confidence value.
   - "info_utility": Utility of the information provided in "ADDITIONAL_INFORMATION" to help you make the prediction.
     0 indicates lowest utility; 1 maximum utility.
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object. Do not include any other contents in your response.
* Never use Markdown syntax highlighting, such as ```json``` to surround the output. Only output the raw json string.
* Do not output the json string surrounded by quotation marks
* This is incorrect:"```json{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}```"
* This is incorrect:```json"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"```
* This is incorrect:"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"
* This is correct:"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"
"""

URL_QUERY_PROMPT = """
You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
for a given event. You are provided with an input under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" specifies an event.
* The event will only have two possible outcomes: either the event will happen or the event will not happen.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.

USER_PROMPT:
```
{user_prompt}
```

OUTPUT_FORMAT
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain two fields: "queries", and "urls".
   - "queries": An array of strings of size between 1 and 5. Each string must be a search engine query that can help obtain relevant information to estimate
     the probability that the event in "USER_PROMPT" occurs. You must provide original information in each query, and they should not overlap
     or lead to obtain the same set of results.
* Output only the JSON object to be parsed by Python's "json.loads()". Do not include any other contents in your response.
* Never use Markdown syntax highlighting, such as ```json```. Only output the raw json string.
* This is incorrect:"```json{{\n  \"queries\": [\"term1\", \"term2\"]}}```"
* This is incorrect:```json"{{\n  \"queries\": [\"term1\", \"term2\"]}}"```
* This is correct:"{{\n  \"queries\": [\"term1\", \"term2\"]}}"
"""

SYSTEM_PROMPT = """You are a world class algorithm for generating structured output from a given input."""

ASSISTANT_TEXT = "```json"
STOP_SEQUENCES = ["```"]


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))

def search_google(query: str, api_key: str, engine: str, num: int = 3) -> List[str]:
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
    items = search.get('items')
    if items is not None:
        return [result["link"] for result in items]
    else:
        return []


def get_urls_from_queries(queries: List[str], api_key: str, engine: str) -> List[str]:
    """Get URLs from search engine queries"""
    results = []
    for query in queries:
        for url in search_google(
            query=query,
            api_key=api_key,
            engine=engine,
            num=3,  # Number of returned results
        ):
            results.append(url)
    unique_results = list(set(results))
    return unique_results


def extract_text(
    html: str,
    num_words: Optional[int] = None,
) -> str:
    """Extract text from a single HTML document"""
    text = Document(html).summary()
    text = md(text, heading_style="ATX")
    if text is None:
        return ""

    if num_words:
        return " ".join(text.split()[:num_words])
    
    # remove newlines and extra spaces
    text = " ".join(text.split())
    
    return text


def process_in_batches(
    urls: List[str], window: int = 5, timeout: int = 10
) -> Iterator[List[Tuple[Future, str]]]:
    """Iter URLs in batches."""
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(urls), window):
            batch = urls[i : i + window]
            futures = [
                (executor.submit(requests.get, url, timeout=timeout), url)
                for url in batch
            ]
            yield futures


def extract_texts(urls: List[str], num_words: int = 300) -> List[str]:
    """Extract texts from URLs"""
    max_allowed = 5
    extracted_texts = []
    count = 0
    stop = False
    for batch in process_in_batches(urls=urls):
        for future, url in batch:
            try:
                result = future.result()
                if result.status_code != 200:
                    continue
                doc = {}
                doc['text'] = extract_text(html=result.text, num_words=num_words)
                doc['url'] = url
                extracted_texts.append(doc)
                count += 1
                if count >= max_allowed:
                    stop = True
                    break
            except requests.exceptions.ReadTimeout:
                print(f"Request timed out: {url}.")
            except Exception as e:
                print(f"An error occurred: {e}")
        if stop:
            break
    return extracted_texts


def fetch_additional_information(
    client: anthropic.Anthropic,
    prompt: str,
    engine: str,
    google_api_key: Optional[str],
    google_engine: Optional[str],
    num_urls: Optional[int],
    num_words: Optional[int],
    counter_callback: Optional[Callable] = None,
    temperature: Optional[float] = DEFAULT_CLAUDE_SETTINGS["temperature"],
    max_tokens: Optional[int] = DEFAULT_CLAUDE_SETTINGS["max_tokens"],
    source_links: Optional[List[str]] = None,
) -> str:
    """Fetch additional information."""
    url_query_prompt = URL_QUERY_PROMPT.format(user_prompt=prompt)
    messages = [
        {"role": "user", "content": url_query_prompt},
    ]
    response = client.messages.create(
        model=engine,
        messages=messages,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        json_data = json.loads(response.content[0].text)
    except json.JSONDecodeError:
        json_data = {}

    if "queries" not in json_data:
        json_data["queries"] = [prompt]

    if not source_links:
        urls = get_urls_from_queries(
            json_data["queries"],
            api_key=google_api_key,
            engine=google_engine,
        )
        texts = extract_texts(urls)
    else:
        texts = []
        for url, content in islice(source_links.items(), num_urls):
            doc = {}
            doc['text'], doc['url'] = extract_text(html=content, num_words=num_words), url
            texts.append(doc)
    # Format the additional information
    additional_information = "\n".join(
        [
            f"ARTICLE {i}, URL: {doc['url']}, CONTENT: {doc['text']}\n"
            for i, doc in enumerate(texts)
        ]
    )
    if counter_callback:
        counter_callback(
            model=engine,
            input_prompt=url_query_prompt,
            output_tokens=40,
            token_counter=count_tokens,
        )
    return additional_information, counter_callback


def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    tool = kwargs["tool"]
    model = kwargs.get("model", TOOL_TO_ENGINE[tool])
    prompt = kwargs["prompt"]
    max_tokens = kwargs.get("max_tokens", DEFAULT_CLAUDE_SETTINGS["max_tokens"])
    temperature = kwargs.get("temperature", DEFAULT_CLAUDE_SETTINGS["temperature"])
    num_urls = kwargs.get("num_urls", NUM_URLS_EXTRACT)
    num_words = kwargs.get("num_words", DEFAULT_NUM_WORDS)
    counter_callback = kwargs.get("counter_callback", None)
    api_keys = kwargs.get("api_keys", {})
    google_api_key = api_keys.get("google_api_key", None)
    google_engine_id = api_keys.get("google_engine_id", None)
    client = anthropic.Anthropic(api_key=api_keys["anthropic"])

    # Make sure the model is supported
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Model {model} not supported.")

    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    engine = kwargs.get("model", TOOL_TO_ENGINE[tool])
    print(f"ENGINE: {engine}")

    if tool == "claude-prediction-online":
        additional_information, counter_callback = fetch_additional_information(
            client=client,
            prompt=prompt,
            engine=engine,
            google_api_key=google_api_key,
            google_engine=google_engine_id,
            num_urls=num_urls,
            num_words=num_words,
            counter_callback=counter_callback,
            source_links=kwargs.get("source_links", None),
        )
    else:
        additional_information = ""

    # Make the prediction
    prediction_prompt = PREDICTION_PROMPT.format(
        user_prompt=prompt, additional_information=additional_information
    )

    messages = [
        {
            "role": "user",
            "content": prediction_prompt,
        },
    ]

    response_prediction = client.messages.create(
        model=engine,
        messages=messages,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    prediction = response_prediction.content[0].text

    if counter_callback:
        counter_callback(
            input_tokens=response_prediction.usage.input_tokens,
            output_tokens=response_prediction.usage.output_tokens,
            model=engine,
            token_counter=count_tokens,
        )

    return prediction, prediction_prompt, None, counter_callback
