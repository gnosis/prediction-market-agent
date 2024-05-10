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

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient
import logging
from markdownify import markdownify
import requests
from bs4 import BeautifulSoup
from requests import Response
from chromadb import Collection, EphemeralClient, Documents, Embeddings
import chromadb.utils.embedding_functions as embedding_functions
from tiktoken import encoding_for_model


# CONFIGURATION

DEFAULT_OPENAI_SETTINGS = {
    "temperature": 0,
    "max_compl_tokens": 500,
}

ALLOWED_TOOLS = [
    "prediction-with-research-conservative",
    "prediction-with-research-bold",
]
TOOL_TO_ENGINE = {
    "prediction-with-research-conservative": "gpt-3.5-turbo-1106",
    "prediction-with-research-bold": "gpt-4-1106-preview",
}

DEFAULT_RESEARCH_SETTINGS = {
    "initial_subqueries_limit": 20,
    "subqueries_limit": 4,
    "scrape_content_split_chunk_size": 800,
    "scrape_content_split_chunk_overlap": 225,
    "top_k_per_query": 8,
}

# PROMPTS

PREDICTION_PROMPT = """
INTRODUCTION:
You are a Large Language Model (LLM) within a multi-agent system. Your primary task is to accurately estimate the probabilities for the outcome of a 'market question', \
found in 'USER_PROMPT'. The market question is part of a prediction market, where users can place bets on the outcomes of market questions and earn rewards if the selected outcome occurrs. The 'market question' \
in this scenario has only two possible outcomes: `Yes` or `No`. Each market has a closing date at which the outcome is evaluated. This date is typically stated within the market question.  \
The closing date is considered to be 23:59:59 of the date provided in the market question. If the event specified in the market question has not occurred before the closing date, the market question's outcome is `No`. \
If the event has happened before the closing date, the market question's outcome is `Yes`. You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION", which is \
sourced from a Google search engine query performed a few seconds ago and is meant to assist you in your probability estimation. You must adhere to the following 'INSTRUCTIONS'.  


INSTRUCTIONS:
* Examine the user's input labeled 'USER_PROMPT'. Focus on the part enclosed in double quotes, which contains the 'market question'.
* If the 'market question' implies more than two outcomes, output the response "Error" and halt further processing.
* When the current time {timestamp} has passed the closing date of the market and the event specified in the market question has not happened, the market question's outcome is `No` and the user who placed a bet on `No` will receive a reward.
* When the current time {timestamp} has passed the closing date of the market and the event has happened before, the market question's final outcome is `Yes` and the user who placed a bet on `yes` will receive a reward.
* Consider the prediction market with the market question, the closing date and the outcomes in an isolated context that has no influence on the protagonists that are involved in the event in the real world, specified in the market question. The closing date is always arbitrarily set by the market creator and has no influence on the real world. So it is likely that the protagonists of the event in the real world are not even aware of the prediction market and do not care about the market's closing date.
* The probability estimations of the market question outcomes must be as accurate as possible, as an inaccurate estimation will lead to financial loss for the user.
* Utilize your training data and the information provided under "ADDITIONAL_INFORMATION" to generate probability estimations for the outcomes of the 'market question'.
* Examine the itemized list under "ADDITIONAL_INFORMATION" thoroughly and use all the relevant information for your probability estimation. This data is sourced from a Google search engine query done a few seconds ago. 
* Use any relevant item in "ADDITIONAL_INFORMATION" in addition to your training data to make the probability estimation. You can assume that you have been provided with the most current and relevant information available on the internet. Still pay close attention on the release and modification timestamps provided in parentheses right before each information item. Some information might be outdated and not relevant anymore.
* More recent information indicated by the timestamps provided in parentheses right before each information item overrides older information within ADDITIONAL_INFORMATION and holds more weight for your probability estimation.
* If there exist contradicting information, evaluate the release and modification dates of those information and prioritize the information that is more recent and adjust your confidence in the probability estimation accordingly.
* Even if not all information might not be released today, you can assume that there haven't been publicly available updates in the meantime except for those inside ADDITIONAL_INFORMATION.
* If the information in "ADDITIONAL_INFORMATION" indicate without a doubt that the event has already happened, it is very likely that the outcome of the market question will be `Yes`.
* If the information in "ADDITIONAL_INFORMATION" indicate that the event will happen after the closing date, it is very likely that the outcome of the market question will be `No`.
* The closer the current time `{timestamp}` is to the closing time the higher the likelyhood that the outcome of the market question will be `No`, if recent information do not clearly indicate that the event will occur before the closing date.
* If there exist recent information indicating that the event will happen after the closing date, it is very likely that the outcome of the market question will be `No`.
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

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain four fields: "p_yes", "p_no", "confidence", and "info_utility", each ranging from 0 to 1.
   - "p_yes": Probability that the market question's outcome will be `Yes`.
   - "p_no": Probability that the market questions outcome will be `No`.
   - "confidence": Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.
   - "info_utility": Utility of the information provided in "ADDITIONAL_INFORMATION" to help you make the probability estimation ranging from 0 (lowest utility) to 1 (maximum utility).
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object in your response. Do not include any other contents in your response.
* Never use Markdown syntax highlighting, such as ```json``` to surround the output. Only output the raw json string.
* This is incorrect:"```json{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}```"
* This is incorrect:```json"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"```
* This is correct:"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"
"""


SUBQUERIES_PROMPT_TEMPLATE = """
    Your goal is to prepare a research plan for {query}.

    The plan will consist of multiple web searches separated by commas.
    Return ONLY the web searches, separated by commas and without quotes.

    Limit your searches to {search_limit}.
"""


QUERY_RERANKING_PROMPT_TEMPLATE = """
    I will present you with a list of queries to search the web for, for answers to the question: {goal}.

    The queries are divided by '---query---'

    Evaluate the queries in order that will provide the best data to answer the question. Do not modify the queries.
    Return them, in order of relevance, as a comma separated list of strings with no quotes.

    Queries: {queries}
"""

REPORT_PROMPT_TEMPLATE = """
    Your goal is to provide a relevant information report
    in order to make an informed prediction for the question: '{goal}'.
    
    Here are the results of relevant web searches:
    
    {search_results}
    
    Prepare a full comprehensive report that provides relevant information to answer the aforementioned question.
    If that is not possible, state why.
    You will structure your report in the following sections:
    
    - Introduction
    - Background
    - Findings and Analysis
    - Conclusion
    - Caveats
    
    Don't limit yourself to just stating each finding; provide a thorough, full and comprehensive analysis of each finding.
    Use markdown syntax. Include as much relevant information as possible and try not to summarize.
    """

class CustomOpenAIEmbeddingFunction(embedding_functions.OpenAIEmbeddingFunction):
    """Custom OpenAI embedding function"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize custom OpenAI embedding function"""
        super().__init__(*args, **kwargs)
        # OpenAI@v1 compatible
        self._client = openai.embeddings

    def __call__(self, texts: Documents) -> Embeddings:
        """Return embedding"""
        # replace newlines, which can negatively affect performance.
        texts = [t.replace("\n", " ") for t in texts]

        # Call the OpenAI Embedding API
        embeddings = self._client.create(input=texts, model=self._model_name).data

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e.index)  # type: ignore

        # Return just the embeddings
        return [result.embedding for result in sorted_embeddings]
# MODELS

MAX_TEXT_LENGTH = 7500


class WebSearchResult(BaseModel):
    title: str
    url: str
    description: str
    relevancy: float
    query: str
    
    def __getitem__(self, item):
        return getattr(self, item)
    
class WebScrapeResult(BaseModel):
    query: str
    url: str
    title: str
    content: str

    def __getitem__(self, item):
        return getattr(self, item)


# FUNCTIONS


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


def fetch_html(url: str, timeout: int) -> Response:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0"
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    return response


def web_scrape(url: str, timeout: int = 10000) -> tuple[str, str]:
    try:
        response = fetch_html(url=url, timeout=timeout)

        if 'text/html' in response.headers.get('Content-Type', ''):
            soup = BeautifulSoup(response.content, "html.parser")
            
            [x.extract() for x in soup.findAll('script')]
            [x.extract() for x in soup.findAll('style')]
            [x.extract() for x in soup.findAll('noscript')]
            [x.extract() for x in soup.findAll('link')]
            [x.extract() for x in soup.findAll('head')]
            [x.extract() for x in soup.findAll('image')]
            [x.extract() for x in soup.findAll('img')]
            
            text = soup.get_text()
            text = markdownify(text)
            text = "  ".join([x.strip() for x in text.split("\n")])
            text = " ".join([x.strip() for x in text.split("  ")])
            
            return (text, url)
        else:
            logging.warning("Non-HTML content received")
            return ("", url)

    except requests.RequestException as e:
        logging.error(f"HTTP request failed: {e}")
        return ("", url)


def scrape_results(results: list[WebSearchResult]) -> list[WebScrapeResult]:
    scraped: list[WebScrapeResult] = []
    results_by_url = {result.url: result for result in results}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(web_scrape, result.url) for result in results}
        for future in as_completed(futures):
            (scraped_content, url) = future.result()
            websearch_result = results_by_url[url]
            result = WebScrapeResult(
                query=websearch_result.query,
                url=websearch_result.url,
                title=websearch_result.title,
                content=scraped_content
            )
            
            scraped.append(result)

    return scraped


def web_search(query: str, api_key: str, max_results=5) -> list[WebSearchResult]:
    tavily = TavilyClient(api_key=api_key)
    response = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
    )

    transformed_results = [
        WebSearchResult(
            title=result['title'],
            url=result['url'],
            description=result['content'],
            relevancy=result['score'],
            query=query
        )
        for result in response['results']
    ]

    return transformed_results


def search(queries: list[str], api_key: str, filter = lambda x: True) -> list[tuple[str, WebSearchResult]]:
    results: list[list[WebSearchResult]] = []
    results_with_queries: list[tuple[str, WebSearchResult]] = []

    # Each result will have a query associated with it
    # We only want to keep the results that are unique
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(web_search, query, api_key) for query in queries}
        for future in as_completed(futures):
            results.append(future.result())

    for i in range(len(results)):
        for result in results[i]:
            if result.url not in [existing_result.url for (_,existing_result) in results_with_queries]:
                if filter(result):
                  results_with_queries.append((queries[i], result))

    return results_with_queries


def create_embeddings_from_results(results: list[WebScrapeResult], text_splitter, api_key: str) -> Collection:
    client = EphemeralClient()
    openai_ef = CustomOpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-ada-002"
            )
    collection = client.create_collection(
        name=f"web_search_results_{random.randint(1, 100)}",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    texts = []
    metadatas = []

    for scrape_result in results:
        text_splits = text_splitter.split_text(scrape_result.content)
        if not len(texts + text_splits) > MAX_TEXT_LENGTH:
            texts += text_splits
        metadatas += [scrape_result.dict() for _ in text_splits]   

    collection.add(
        documents=texts,
        metadatas=metadatas,  # type: ignore
        ids=[f'id{i}' for i in range(len(texts))]
    )
    return collection


def generate_subqueries(
    query: str,
    limit: int,
    api_key: str,
    model: str,
    counter_callback: Optional[Callable] = None,
) -> tuple[list[str], Any]:
    client = OpenAI(api_key=api_key)
    subquery_generation_prompt = SUBQUERIES_PROMPT_TEMPLATE.format(query=query, search_limit=limit)
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional researcher"
            },
            {
                "role": "user",
                "content": subquery_generation_prompt,
            }
        ],
        model=model,
        n=1,
        timeout=90,
        stop=None,
    )

    subqueries_str = str(response.choices[0].message.content)
    
    if counter_callback:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=model,
            token_counter=count_tokens,
        )
        return [query] + [subquery.strip('\"').strip() for subquery in subqueries_str.split(',')], counter_callback
    return [query] + [subquery.strip('\"').strip() for subquery in subqueries_str.split(',')], None


def rerank_subqueries(
    queries: list[str],
    goal: str,
    api_key: str,
    model: str,
    counter_callback: Optional[Callable] = None
) -> tuple[list[str], Any]:
    client = OpenAI(api_key=api_key)
    rerank_results_prompt = QUERY_RERANKING_PROMPT_TEMPLATE.format(
        goal=goal,
        queries="\n---query---\n".join(queries)
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional researcher"
            },
            {
                "role": "user",
                "content": rerank_results_prompt,
            }
        ],
        model=model,
        n=1,
        timeout=90,
        stop=None,
    )

    subqueries_str = str(response.choices[0].message.content)
    
    if counter_callback:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=model,
            token_counter=count_tokens,
        )
        return [subquery.strip('\"').strip() for subquery in subqueries_str.split(',')], counter_callback
    return [subquery.strip('\"').strip() for subquery in subqueries_str.split(',')], None


def prepare_report(
    goal: str,
    scraped: list[str],
    model: str,
    api_key: str,
    counter_callback: Optional[Callable] = None
) -> tuple[str, Any]:
    evaluation_prompt = REPORT_PROMPT_TEMPLATE.format(search_results=scraped, goal=goal)
    
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional researcher"
            },
            {
                "role": "user",
                "content": evaluation_prompt,
            }
        ],
        model=model,
        n=1,
        timeout=90,
        stop=None,
    )
    
    if counter_callback:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=model,
            token_counter=count_tokens,
        )
        return str(response.choices[0].message.content), counter_callback
    return str(response.choices[0].message.content), None


def make_prediction(
    prompt: str,
    model: str,
    temperature: float,
    max_compl_tokens: int,
    openai_api_key: str,
    counter_callback: Optional[Callable] = None
) -> tuple[str, Any]:
    client = OpenAI(api_key=openai_api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_compl_tokens,
        n=1,
        timeout=150,
        stop=None
    )
    
    if counter_callback:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=model,
            token_counter=count_tokens,
        )
        return str(response.choices[0].message.content), counter_callback
    return str(response.choices[0].message.content), None


def run(**kwargs) -> Tuple[Optional[str], Any, Optional[Dict[str, Any]], Any]:
    """Run the task"""
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]
    temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
    max_compl_tokens = kwargs.get(
        "max_tokens", DEFAULT_OPENAI_SETTINGS["max_compl_tokens"]
    )
    
    openai_api_key = kwargs["api_keys"]["openai"]
    tavily_api_key = kwargs["api_keys"]["tavily"]
    
    initial_subqueries_limit = kwargs.get('initial_subqueries_limit', DEFAULT_RESEARCH_SETTINGS["initial_subqueries_limit"])
    subqueries_limit = kwargs.get('subqueries_limit', DEFAULT_RESEARCH_SETTINGS["subqueries_limit"])
    scrape_content_split_chunk_size = kwargs.get('scrape_content_split_chunk_size', DEFAULT_RESEARCH_SETTINGS["scrape_content_split_chunk_size"])
    scrape_content_split_chunk_overlap = kwargs.get('scrape_content_split_chunk_overlap', DEFAULT_RESEARCH_SETTINGS["scrape_content_split_chunk_overlap"])
    top_k_per_query = kwargs.get('top_k_per_query', DEFAULT_RESEARCH_SETTINGS["top_k_per_query"])
    counter_callback = kwargs.get("counter_callback", None)
    
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"TOOL {tool} is not supported.")
    
    model = kwargs.get("model", TOOL_TO_ENGINE[tool])
    print(f"ENGINE: {model}")
    queries, counter_callback = generate_subqueries(query=prompt, limit=initial_subqueries_limit, api_key=openai_api_key, model=model, counter_callback=counter_callback)
    queries, counter_callback = rerank_subqueries(queries=queries, goal=prompt, api_key=openai_api_key, model=model, counter_callback=counter_callback)
    queries = queries[:subqueries_limit]

    search_results_with_queries = search(queries, tavily_api_key, lambda result: not result["url"].startswith("https://www.youtube"))

    scrape_args = [result for (_, result) in search_results_with_queries]
    scraped = scrape_results(scrape_args)
    scraped = [result for result in scraped if result.content != ""]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "  "],
        chunk_size=scrape_content_split_chunk_size,
        chunk_overlap=scrape_content_split_chunk_overlap
    )
    collection = create_embeddings_from_results(scraped, text_splitter, api_key=openai_api_key)

    query_results = collection.query(query_texts=queries, n_results=top_k_per_query)
    vector_result_texts: list[str] = []
    
    for documents_list in query_results['documents']: # type: ignore
        vector_result_texts += [x for x in documents_list]

    research_report, counter_callback = prepare_report(prompt, vector_result_texts, api_key=openai_api_key, model=model, counter_callback=counter_callback)
    
    current_time_utc = datetime.now(timezone.utc)
    formatted_time_utc = current_time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-6] + "Z"

    prediction_prompt = PREDICTION_PROMPT.format(
        user_prompt=prompt,
        additional_information=research_report,
        timestamp=formatted_time_utc
    )
    prediction, counter_callback = make_prediction(
        prompt=prediction_prompt,
        model=model,
        temperature=temperature,
        max_compl_tokens=max_compl_tokens,
        openai_api_key=openai_api_key,
        counter_callback=counter_callback
    )
    
    if counter_callback is not None:
        return prediction, prediction_prompt, None, counter_callback
    return prediction, prediction_prompt, None, None