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

import re
import json
import faiss
import PyPDF2
import requests
import anthropic
import numpy as np
from io import BytesIO
from openai import OpenAI
from itertools import islice
from pydantic import BaseModel
from collections import defaultdict
from googleapiclient.discovery import build
from concurrent.futures import Future, ThreadPoolExecutor
from readability import Document as ReadabilityDocument
from requests.exceptions import RequestException, TooManyRedirects
from markdownify import markdownify as md
from typing import Any, Dict, Generator, List, Optional, Tuple, Callable
from tiktoken import encoding_for_model


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
    "prediction-request-rag-claude",
]
ALLOWED_MODELS = [
    "claude-2",
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
]
TOOL_TO_ENGINE = {tool: "claude-3-haiku-20240307" for tool in ALLOWED_TOOLS}

DEFAULT_NUM_URLS = defaultdict(lambda: 3)
DEFAULT_NUM_QUERIES = defaultdict(lambda: 3)
NUM_URLS_PER_QUERY = 5
SPLITTER_CHUNK_SIZE = 1800
SPLITTER_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_BATCH_SIZE = 1000
EMBEDDING_SIZE = 1536
SPLITTER_MAX_TOKENS = 1800
SPLITTER_OVERLAP = 50
NUM_NEIGHBORS = 4
HTTP_TIMEOUT = 20
HTTP_MAX_REDIRECTS = 5
HTTP_MAX_RETIES = 2


PREDICTION_PROMPT = """
Here is some additional background information that may be relevant to the question:
<additional_information> {ADDITIONAL_INFORMATION} </additional_information>

A user has asked the following:

<user_prompt> {USER_PROMPT} </user_prompt>

Carefully consider the user's question and the additional information provided. Think through the likelihood of the event the user asked about actually happening in the future, based on the details given. Write out your reasoning and analysis in a section.

Now, based on your analysis above, provide a prediction of the probability the event will happen, as p_yes between 0 and 1. Also provide the probability it will not happen, as p_no between 0 and 1. The two probabilities should sum to 1.

p_yes: p_no:

How useful was the additional information in allowing you to make a prediction? Provide your rating as info_utility, a number between 0 and 1.

info_utility:

Finally, considering everything, what is your overall confidence in your prediction? Provide your confidence as a number between 0 and 1.

confidence:

Make sure the values you provide are between 0 and 1. And p_yes and p_no should sum to 1.

Your response should be structured as follows:
<p_yes></p_yes>
<p_no></p_no>
<info_utility></info_utility>
<confidence></confidence>
<analysis></analysis>
"""

URL_QUERY_PROMPT = """
Here is the user prompt: {USER_PROMPT}

Please read the prompt carefully and identify the key pieces of information that need to be searched for in order to comprehensively address the topic.

Brainstorm a list of {NUM_QUERIES} different search queries that cover various aspects of the user prompt. Each query should be focused on a specific sub-topic or question related to the overarching prompt.

Please write each search query inside its own tags, like this: example search query here

The queries should be concise while still containing enough information to return relevant search results. Focus the queries on gathering factual information to address the prompt rather than opinions.

After you have written all {NUM_QUERIES} search queries, please submit your final response.

<queries></queries>
"""

SYSTEM_PROMPT = """You are a world class algorithm for generating structured output from a given input."""

class Document(BaseModel):
    text: str
    url: str
    embedding: Optional[List[float]] = None


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


def multi_queries(
    client: anthropic.Anthropic,
    prompt: str,
    engine: str,
    num_queries: int,
    counter_callback: Optional[Callable[[int, int, str], None]] = None,
    temperature: Optional[float] = DEFAULT_CLAUDE_SETTINGS["temperature"],
    max_tokens: Optional[int] = DEFAULT_CLAUDE_SETTINGS["max_tokens"],
) -> List[str]:
    """Generate multiple queries for fetching information from the web."""
    url_query_prompt = URL_QUERY_PROMPT.format(
        USER_PROMPT=prompt, NUM_QUERIES=num_queries
    )

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
    if counter_callback:
        counter_callback(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=engine,
            token_counter=count_tokens,
        )
    queries = parser_query_response(response.content[0].text, num_queries=num_queries)
    queries.append(prompt)

    return queries, counter_callback


def parser_query_response(response: str, num_queries: int = 5) -> List[str]:
    """Parse the response from the query generation model with optional enhancements."""
    queries = response.split("<queries>")[1].split("</queries>")[0].split("\n")
    parsed_queries = [query.strip() for query in queries if query.strip()]
    enhanced_queries = []

    for query in parsed_queries:
        if query[0].isdigit():
            query = ". ".join(query.split(". ")[1:])
        query = query.replace('"', '')
        enhanced_queries.append(query)

    if len(enhanced_queries) == num_queries * 2:
        enhanced_queries = enhanced_queries[::2]

    # Remove doubel quotes from the queries
    final_queries = [query.replace('"', '') for query in enhanced_queries]

    # if there are any xml tags in the queries, remove them
    final_queries = [re.sub(r'<[^>]*>', '', query) for query in final_queries]
    
    return final_queries


def search_google(
    query: str, 
    api_key: str, 
    engine: str, 
    num: int
) -> List[str]:
    """Search Google for the given query."""
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
        except Exception:
            pass
    unique_results = list(set(results))
    return unique_results


def extract_text(
    html: str,
    num_words: Optional[int] = None,
) -> str:
    """Extract text from a single HTML document"""
    text = ReadabilityDocument(html).summary()

    # use html2text to convert HTML to markdown
    text = md(text, heading_style="ATX")

    if text is None:
        return None

    if num_words:
        text = " ".join(text.split()[:num_words])
    else:
        text = " ".join(text.split())

    doc = Document(text=text, url="")
    return doc


def extract_text_from_pdf(url: str, num_words: Optional[int] = None) -> str:
    """Extract text from a PDF document at the given URL."""
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()

        if "application/pdf" not in response.headers.get("Content-Type", ""):
            return ValueError("URL does not point to a PDF document")

        with BytesIO(response.content) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        doc = Document(text=text[:num_words] if num_words else text, date="", url=url)
        print(f"Using PDF: {url}: {doc.text[:300]}...")
        return doc
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def process_in_batches(
    urls: List[str], 
    window: int = 5, 
    timeout: int = HTTP_TIMEOUT,
    max_redirects: int = HTTP_MAX_REDIRECTS,
    retries: int = HTTP_MAX_RETIES,
) -> Generator[None, None, List[Tuple[Optional[Future], str]]]:
    """Iter URLs in batches with improved error handling and retry mechanism."""
    with ThreadPoolExecutor() as executor, requests.Session() as session:
        session.max_redirects = max_redirects
        for i in range(0, len(urls), window):
            batch = urls[i : i + window]
            futures = []
            for url in batch:
                future = None
                attempt = 0
                while attempt < retries:
                    try:
                        future = executor.submit(session.get, url, timeout=timeout)
                        break  
                    except (TooManyRedirects, RequestException) as e:
                        print(f"Attempt {attempt + 1} failed for {url}: {e}")
                        attempt += 1
                        if attempt == retries:
                            print(f"Max retries reached for {url}. Moving to next URL.")
                futures.append((future, url))
            yield futures


def extract_texts(urls: List[str], num_words: Optional[int] = None) -> List[Document]:
    """Extract texts from URLs with improved error handling, excluding failed URLs."""
    extracted_texts = []
    for batch in process_in_batches(urls=urls):
        for future, url in batch:
            if future is None:
                continue
            try:
                result = future.result()
                if result.status_code == 200:
                    # Check if URL ends with .pdf or content starts with %PDF
                    if url.endswith('.pdf') or result.content[:4] == b'%PDF':
                        doc = extract_text_from_pdf(url, num_words=num_words)
                    else:
                        doc = extract_text(html=result.text, num_words=num_words)
                    doc.url = url
                    extracted_texts.append(doc)
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
    return extracted_texts


def find_similar_chunks(
    client: OpenAI,
    query: str, 
    docs_with_embeddings: List[Document], 
    k: int = 4
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


def get_embeddings(
    client: OpenAI,
    split_docs: List[Document]
) -> List[Document]:
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


def recursive_character_text_splitter(text, max_tokens, overlap):
    if len(text) <= max_tokens:
        return [text]
    else:
        return [text[i:i+max_tokens] for i in range(0, len(text), max_tokens - overlap)]
    

def fetch_additional_information(
    client: anthropic.Anthropic,
    client_openai: OpenAI,
    prompt: str,
    engine: str,
    google_api_key: Optional[str],
    google_engine_id: Optional[str],
    counter_callback: Optional[Callable[[int, int, str], None]] = None,
    source_links: Optional[List[str]] = None,
    num_urls: Optional[int] = DEFAULT_NUM_URLS,
    num_queries: Optional[int] = DEFAULT_NUM_QUERIES,
    temperature: Optional[float] = DEFAULT_CLAUDE_SETTINGS["temperature"],
    max_tokens: Optional[int] = DEFAULT_CLAUDE_SETTINGS["max_tokens"]
) -> Tuple[str, Callable[[int, int, str], None]]:
    
    """Fetch additional information to help answer the user prompt."""

    # generate multiple queries for fetching information from the web
    
    try:
        queries, counter_callback = multi_queries(
            client=client,
            prompt=prompt,
            engine=engine,
            num_queries=num_queries,
            counter_callback=counter_callback,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print(f"Queries: {queries}")
    except Exception as e:
        print(f"Error generating queries: {e}")
        queries = [prompt]

    # get the top URLs for the queries
    if not source_links:
        urls = get_urls_from_queries(
            queries=queries,
            api_key=google_api_key,
            engine=google_engine_id,
            num=NUM_URLS_PER_QUERY,
        )
        print(f"URLs: {urls}")

        urls = list(set(urls))

        # Extract text and dates from the URLs
        docs = extract_texts(
            urls=urls,
        )
    else:
        docs = []
        for url, content in islice(source_links.items(), num_urls or len(source_links)):
            doc = extract_text(html=content)
            doc.url = url
            docs.append(doc)

    # Remove None values from the list
    docs = [doc for doc in docs if doc]

    # remove empty documents ""
    filtered_docs = [doc for doc in docs if hasattr(doc, 'text') and doc.text != ""]

    # Chunk the documents
    split_docs = []
    for doc in filtered_docs:
        try:
            t = recursive_character_text_splitter(
                doc.text, SPLITTER_CHUNK_SIZE, SPLITTER_OVERLAP
            )
            split_docs.extend(
                [Document(text=chunk, url=doc.url) for chunk in t]
            )
        except Exception as e:
            print(f"Error splitting document: {e}")
            continue
    print(f"Split Docs: {len(split_docs)}")

    # Remove None values from the list
    split_docs = [doc for doc in split_docs if doc]

    # Embed the documents
    docs_with_embeddings = get_embeddings(client_openai, split_docs)
    print(f"Docs with embeddings: {len(docs_with_embeddings)}")
    
    # Find similar chunks
    similar_chunks = find_similar_chunks(
        client_openai,
        query=prompt,
        docs_with_embeddings=docs_with_embeddings,
        k=NUM_NEIGHBORS,
    )
    print(f"Similar Chunks: {len(similar_chunks)}")

    # Format the additional information
    additional_information = "\n".join(
        [
            f"ARTICLE {i}, URL: {doc.url}, CONTENT: {doc.text}\n"
            for i, doc in enumerate(similar_chunks)
        ]
    )

    return additional_information, counter_callback


def extract_question(prompt: str) -> str:
    pattern = r'\"(.*?)\"'
    try:
        question = re.findall(pattern, prompt)[0]
    except Exception as e:
        print(f"Error extracting question: {e}")
        question = prompt

    return question


def parser_prediction_response(response: str) -> str:
    """Parse the response from the prediction model."""
    results = {}
    for key in ["p_yes", "p_no", "info_utility", "confidence"]:
        try:
            value = response.split(f"<{key}>")[1].split(f"</{key}>")[0].strip()
            if key in ["p_yes", "p_no", "info_utility", "confidence"]:
                value = float(value)
            results[key] = value
        except Exception:
            raise ValueError(f"Error parsing {key}")

    results = json.dumps(results)
    return results


def run(**kwargs) -> Tuple[Optional[str], Any, Optional[Dict[str, Any]], Any]:
    """Run the task"""

    tool = kwargs["tool"]
    model = kwargs.get("model", TOOL_TO_ENGINE[tool])
    prompt = extract_question(kwargs["prompt"])
    max_tokens = kwargs.get("max_tokens", DEFAULT_CLAUDE_SETTINGS["max_tokens"])
    temperature = kwargs.get("temperature", DEFAULT_CLAUDE_SETTINGS["temperature"])
    num_urls = kwargs.get("num_urls", DEFAULT_NUM_URLS[tool])
    num_queries = kwargs.get("num_queries", DEFAULT_NUM_QUERIES[tool])
    counter_callback = kwargs.get("counter_callback", None)
    api_keys = kwargs.get("api_keys", {})
    google_api_key = api_keys.get("google_api_key", None)
    google_engine_id = api_keys.get("google_engine_id", None)
    client = anthropic.Anthropic(api_key=api_keys["anthropic"])
    client_openai = OpenAI(api_key=api_keys["openai"])

    # Make sure the model is supported
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Model {model} not supported.")
    
    # make sure the tool is supported
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} not supported.")
    
    engine = kwargs.get("model", TOOL_TO_ENGINE[tool])
    print(f"ENGINE: {engine}")
    
    additional_information, counter_callback = fetch_additional_information(
        client=client,
        client_openai=client_openai,
        prompt=prompt,
        engine=engine,
        google_api_key=google_api_key,
        google_engine_id=google_engine_id,
        counter_callback=counter_callback,
        source_links=kwargs.get("source_links", None),
        num_urls=num_urls,
        num_queries=num_queries,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Generate the prediction prompt
    prediction_prompt = PREDICTION_PROMPT.format(
        ADDITIONAL_INFORMATION=additional_information,
        USER_PROMPT=prompt,
    )

    # Generate the prediction
    messages = [
        {"role": "user", "content": prediction_prompt},
    ]

    response = client.messages.create(
        model=engine,
        messages=messages,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if counter_callback:
        counter_callback(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=engine,
            token_counter=count_tokens,
        )   

    results = parser_prediction_response(response.content[0].text)
    
    return results, prediction_prompt, None, counter_callback
