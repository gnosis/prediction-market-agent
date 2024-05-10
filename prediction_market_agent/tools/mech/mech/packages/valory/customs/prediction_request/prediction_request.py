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
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from heapq import nlargest
from itertools import islice
from string import punctuation
from typing import Any, Dict, Generator, List, Optional, Tuple, Callable

from openai import OpenAI

import requests
import spacy
import tiktoken
from markdownify import markdownify as md
from readability import Document
from googleapiclient.discovery import build
from spacy import Language
from spacy.cli import download
from spacy.lang.en import STOP_WORDS
from spacy.tokens import Doc, Span
from tiktoken import encoding_for_model

client: Optional[OpenAI] = None


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

def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


FrequenciesType = Dict[str, float]
ScoresType = Dict[Span, float]


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.,
}
ALLOWED_TOOLS = [
    "prediction-offline",
    "prediction-online",
    "prediction-online-summarized-info",
]
MAX_TOKENS = {
    "gpt-3.5-turbo-0125": 4096,
    "gpt-4-0125-preview": 8192,
}
TOOL_TO_ENGINE = {tool: "gpt-4-0125-preview" for tool in ALLOWED_TOOLS}
# the default number of URLs to fetch online information for
DEFAULT_NUM_URLS = defaultdict(lambda: 3)
DEFAULT_NUM_URLS["prediction-online-summarized-info"] = 7
# the default number of words to fetch online information for
DEFAULT_NUM_WORDS: Dict[str, Optional[int]] = defaultdict(lambda: 300)
DEFAULT_NUM_WORDS["prediction-online-summarized-info"] = None
# how much of the initial content will be kept during summarization
DEFAULT_COMPRESSION_FACTOR = 0.05
# the vocabulary to use for the summarization
DEFAULT_VOCAB = "en_core_web_sm"

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
* This is incorrect:"```json{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}```"
* This is incorrect:```json"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"```
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
* Output only the JSON object. Do not include any other contents in your response.
* Never use Markdown syntax highlighting, such as ```json``` to surround the output. Only output the raw json string.
* This is incorrect: "```json{{"queries": []}}```"
* This is incorrect: "```json"{{"queries": []}}"```"
* This is correct: "{{"queries": []}}"
"""


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
        for url in search_google(
            query=query,
            api_key=api_key,
            engine=engine,
            num=num,
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
) -> Generator[None, None, List[Tuple[Future, str]]]:
    """Iter URLs in batches."""
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(urls), window):
            batch = urls[i : i + window]
            futures = [
                (executor.submit(requests.get, url, timeout=timeout), url)
                for url in batch
            ]
            yield futures


def extract_texts(urls: List[str], num_words: Optional[int]) -> List[str]:
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
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    google_api_key: Optional[str],
    google_engine: Optional[str],
    num_urls: Optional[int],
    num_words: Optional[int],
    counter_callback: Optional[Callable] = None,
    source_links: Optional[List[str]] = None,
) -> Tuple[str, Any]:
    """Fetch additional information."""
    url_query_prompt = URL_QUERY_PROMPT.format(user_prompt=prompt)
    moderation_result = client.moderations.create(input=url_query_prompt)
    if moderation_result.results[0].flagged:
        return ""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": url_query_prompt},
    ]
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        timeout=90,
        stop=None,
    )
    json_data = json.loads(response.choices[0].message.content)
    if not source_links:
        urls = get_urls_from_queries(
            json_data["queries"],
            google_api_key,
            google_engine,
            num_urls,
        )
        texts = extract_texts(urls, num_words)
    else:
        texts = []
        for url, content in islice(source_links.items(), 3):
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
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=engine,
            token_counter=count_tokens,
        )
    return additional_information, counter_callback


def load_model(vocab: str) -> Language:
    """Utilize spaCy to load the model and download it if it is not already available."""
    try:
        return spacy.load(vocab)
    except OSError:
        print("Downloading language model...")
        download(vocab)
        return spacy.load(vocab)


def calc_word_frequencies(doc: Doc) -> FrequenciesType:
    """Get the frequency of each word in the given text, excluding stop words and punctuations."""
    word_frequencies = defaultdict(lambda: 0)
    for token in doc:
        word = token.text
        lower = word.lower()
        if lower not in STOP_WORDS.union(punctuation):
            word_frequencies[lower] += 1

    max_frequency = max(word_frequencies.values())
    normalized_frequencies = defaultdict(
        lambda: 0,
        {
            word: frequency / max_frequency
            for word, frequency in word_frequencies.items()
        },
    )
    return normalized_frequencies


def calc_sentence_scores(
    sentence_tokens: List[Span], word_frequencies: FrequenciesType
) -> ScoresType:
    """Calculate the sentence scores."""
    sentence_scores = defaultdict(lambda: 0)
    for sentence in sentence_tokens:
        for token in sentence:
            sentence_scores[sentence] += word_frequencies[token.text.lower()]

    return sentence_scores


def summarize(text: str, compression_factor: float, vocab: str) -> str:
    """Summarize the given text, retaining the given compression factor."""
    if not text:
        raise ValueError("Cannot summarize empty text!")

    nlp = load_model(vocab)
    doc = nlp(text)
    word_frequencies = calc_word_frequencies(doc)
    sentence_tokens = list(doc.sents)
    sentence_scores = calc_sentence_scores(sentence_tokens, word_frequencies)
    n = int(len(sentence_tokens) * compression_factor)
    summary = nlargest(n, sentence_scores, key=sentence_scores.get)
    summary_words = [word.text for word in summary]
    summary_text = "".join(summary_words)
    return summary_text


def adjust_additional_information(
    prompt: str, 
    prompt_template:str, 
    additional_information: str, 
    model: str
) -> str:
    """Adjust the additional_information to fit within the token budget"""

    # Initialize tiktoken encoder for the specified model
    enc = tiktoken.encoding_for_model(model)
    
    # Encode the user prompt to calculate its token count
    prompt = prompt_template.format(user_prompt=prompt, additional_information="")
    prompt_tokens = len(enc.encode(prompt))
    
    # Calculate available tokens for additional_information
    MAX_PREDICTION_PROMPT_TOKENS = MAX_TOKENS[model] - DEFAULT_OPENAI_SETTINGS["max_tokens"]
    available_tokens = MAX_PREDICTION_PROMPT_TOKENS - prompt_tokens
    
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
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        num_urls = kwargs.get("num_urls", DEFAULT_NUM_URLS[tool])
        num_words = kwargs.get("num_words", DEFAULT_NUM_WORDS[tool])
        compression_factor = kwargs.get("compression_factor", DEFAULT_COMPRESSION_FACTOR)
        vocab = kwargs.get("vocab", DEFAULT_VOCAB)
        counter_callback = kwargs.get("counter_callback", None)
        api_keys = kwargs.get("api_keys", {})
        google_api_key = api_keys.get("google_api_key", None)
        google_engine_id = api_keys.get("google_engine_id", None)

        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"Tool {tool} is not supported.")

        engine = kwargs.get("model", TOOL_TO_ENGINE[tool])
        print(f"ENGINE: {engine}")

        if tool.startswith("prediction-online"):
            additional_information, counter_callback = fetch_additional_information(
                prompt,
                engine,
                temperature,
                max_tokens,
                google_api_key,
                google_engine_id,
                num_urls,
                num_words,
                counter_callback=counter_callback,
                source_links=kwargs.get("source_links", None),
            )
        else:
            additional_information = ""

        if additional_information and tool == "prediction-online-summarized-info":
            additional_information = summarize(
                additional_information, compression_factor, vocab
            )
        additional_information = adjust_additional_information(
            prompt, PREDICTION_PROMPT, additional_information, engine
        )
        prediction_prompt = PREDICTION_PROMPT.format(
            user_prompt=prompt, additional_information=additional_information
        )
        moderation_result = client.moderations.create(input=prediction_prompt)
        if moderation_result.results[0].flagged:
            return "Moderation flagged the prompt as in violation of terms.", None, None, None
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prediction_prompt},
        ]
        response = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            timeout=150,
            stop=None,
        )
        if counter_callback is not None:
            counter_callback(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=engine,
                token_counter=count_tokens,
            )
        return response.choices[0].message.content, prediction_prompt, None, counter_callback
