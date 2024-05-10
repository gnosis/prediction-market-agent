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

"""This module implements a Mech tool for binary predictions.

This module tries to mimic the current logic on the market-creator service
(https://github.com/valory-xyz/market-creator) for resolving closed markets.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI


client: Optional[OpenAI] = None


class OpenAIClientManager:
    """Client context manager for OpenAI."""

    def __init__(self, api_key: str):
        """Init OpenAIClientManager"""
        self.api_key = api_key

    def __enter__(self) -> OpenAI:
        """Enter"""
        global client
        if client is None:
            client = OpenAI(api_key=self.api_key)
        return client

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        """Exit"""
        global client
        if client is not None:
            client.close()
            client = None


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 700,
    "temperature": 0.7,
}
ALLOWED_TOOLS = [
    "close_market",
]
TOOL_TO_ENGINE = {tool: "gpt-4-0125-preview" for tool in ALLOWED_TOOLS}

NEWSAPI_ENDPOINT = "https://newsapi.org/v2"
TOP_HEADLINES = "top-headlines"
EVERYTHING = "everything"

ARTICLE_LIMIT = 1_000
ADDITIONAL_INFO_LIMIT = 5_000
HTTP_OK = 200

URL_QUERY_PROMPT_TEMPLATE = """
You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a binary outcome
for a given event. You are provided with an input under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" specifies an event.
* The "USER_PROMPT" will contain a date which in the past.
* The event will only have has two possible outcomes: either the event has happened or the event has not happened.
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
   - "queries": An array of strings of size between 1 and 4. Each string must be a search engine query that can help
     obtain relevant information to check that the event in "USER_PROMPT" occurs.
     You must provide original information in each query, and they should not overlap.
     or lead to obtain the same set of results.
* Output only the JSON object. Do not include any other contents in your response.
"""

OUTCOME_PROMPT_TEMPLATE = """
You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
for a given event. You are provided with an input under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" specifies an event.
* The "USER_PROMPT" will contain a date which in the past.
* The event will only have two possible outcomes: either the event has happened or the event has not happened.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You must provide a decision whether the event in "USER_PROMPT" has occurred or not.
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
* The JSON must contain one field: "has_occurred". When the event in "USER_PROMPT" has occurred, the value of
"has_occurred" must be true if it has occurred, and false if it has not.
* Output only the JSON object. Do not include any other contents in your response.
"""

logging.basicConfig(level=logging.INFO)


class Object(object):
    """Object"""


class CloseMarketBehaviourMock:
    """CloseMarketBehaviourMock"""

    params: Object
    context: Object
    kwargs: Dict[str, Any]

    def __init__(self, **kwargs):
        """Init the object."""
        self.kwargs = kwargs
        self.context = Object()
        self.context.logger = logging.getLogger(__name__)
        self.params = Object()
        self.params.market_closing_newsapi_api_key = kwargs.get("api_keys", {})[
            "newsapi"
        ]
        self.params.newsapi_endpoint = NEWSAPI_ENDPOINT

    def get_http_response(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        parameters: Dict[str, Any],
    ) -> requests.Response:
        """Make an HTTP request and yield the response."""
        if method == "GET":
            response = requests.get(url, headers=headers, params=parameters)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=parameters)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        return response

    def _parse_llm_output(
        self, output: str, required_fields: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse the llm output to json."""
        try:
            output = output.replace("`", "")
            json_data = json.loads(output)
            if required_fields is not None:
                for field in required_fields:
                    if field not in json_data:
                        self.context.logger.error(
                            f"Field {field} not in json_data {json_data}"
                        )
                        return None
            return json_data
        except json.JSONDecodeError as e:
            self.context.logger.error(f"Error decoding JSON response. {e}")
            return None

    def _append_articles_to_input(
        self, news_list: List[dict], input_string: str
    ) -> str:
        """Append articles to input."""
        for article in news_list:
            title = article["title"]
            content = article["content"][:ARTICLE_LIMIT]
            date = article["publishedAt"]
            current_article = f"- ({date}) {title}\n  {content}\n\n"
            if len(input_string) + len(current_article) > ADDITIONAL_INFO_LIMIT:
                break
            input_string += current_article
        return input_string

    def do_llm_request(self, **kwargs) -> str:
        """Do LLM request."""
        with OpenAIClientManager(kwargs["api_keys"]["openai"]):
            max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
            temperature = kwargs.get(
                "temperature", DEFAULT_OPENAI_SETTINGS["temperature"]
            )
            prompt = kwargs.get("prompt")
            engine = kwargs.get("model", TOOL_TO_ENGINE[tool])
            print(f"ENGINE: {engine}")
            moderation_result = client.moderations.create(input=prompt)
            if moderation_result.results[0].flagged:
                return (
                    "Moderation flagged the prompt as in violation of terms.",
                    None,
                    None,
                )

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                timeout=120,
                stop=None,
            )
            res = Object()
            res.value = response.choices[0].message.content

            return res

    def _get_answer(self, question: str) -> Optional[str]:
        """Get an answer for the provided questions"""

        # An initial query is made to Newsapi to detect ratelimit issue
        # This query is also included in the input_news passed to the LLM,
        # if the call succeeds. Newsapi returns 0 output if included the
        # question mark ? sign.
        input_news = ""
        initial_news_articles = self._get_news(
            question.replace("will", "").replace("?", "")
        )
        if initial_news_articles is None:
            self.context.logger.info(
                f"Could not get news articles for query {question} (initial)"
            )
            return None
        input_news = self._append_articles_to_input(initial_news_articles, input_news)

        prompt_values = {
            "user_prompt": question,
        }

        prompt = URL_QUERY_PROMPT_TEMPLATE.format(**prompt_values)
        kwargs1 = {"prompt": prompt}
        kwargs1.update(self.kwargs)
        llm_response_message = self.do_llm_request(**kwargs1)

        result_str = llm_response_message.value.replace("OUTPUT:", "").rstrip().lstrip()
        self.context.logger.info(f"Got LLM response: {result_str}")
        result = self._parse_llm_output(result_str, required_fields=["queries"])
        if result is None:
            self.context.logger.info(f"Could not parse LLM response: {result}")
            return None

        queries = result["queries"]
        self.context.logger.info(f"Got queries: {queries}")
        if len(queries) == 0:
            self.context.logger.info(f"No queries found in LLM response: {result}")
            return None

        # query newsapi
        for query in queries:
            news_articles = self._get_news(query)
            if news_articles is None:
                self.context.logger.info(
                    f"Could not get news articles for query {query}"
                )
                return None
            input_news = self._append_articles_to_input(news_articles, input_news)

        if len(input_news) == 0:
            self.context.logger.info(f"No news articles found for queries {queries}")
            return None

        prompt_values["additional_information"] = input_news

        # llm request message
        prompt = OUTCOME_PROMPT_TEMPLATE.format(**prompt_values)
        kwargs2 = {"prompt": prompt}
        kwargs2.update(self.kwargs)
        llm_response_message = self.do_llm_request(**kwargs2)

        result_str = llm_response_message.value.replace("OUTPUT:", "").rstrip().lstrip()
        self.context.logger.info(f"Got LLM response: {result_str}")
        json_data = self._parse_llm_output(result_str, required_fields=["has_occurred"])
        if json_data is None:
            self.context.logger.info(f"Could not parse LLM response: {json_data}")
            return None

        has_occurred = bool(json_data["has_occurred"])
        self.context.logger.info(f'Has "{question!r}" occurred?: {has_occurred}')

        json_data["question"] = question
        json_data["utc_timestamp"] = int(datetime.utcnow().timestamp())

        return json_data

    def _get_news(self, query: str) -> List[Dict[str, Any]]:
        """Auxiliary method to collect data from endpoint."""

        headers = {"X-Api-Key": self.params.market_closing_newsapi_api_key}

        parameters = {
            "q": query,
            "pageSize": "100",
        }
        # search through all articles everything
        url = f"{self.params.newsapi_endpoint}/{EVERYTHING}"
        response = self.get_http_response(
            method="GET",
            url=url,
            headers=headers,
            parameters=parameters,
        )
        if response.status_code != HTTP_OK:
            self.context.logger.error(
                f"Could not retrieve response from {self.params.newsapi_endpoint}."
                f"Received status code {response.status_code}.\n{response}"
            )
            return None

        response_data = json.loads(response.text)
        self.context.logger.info(
            f"Response received from {self.params.newsapi_endpoint}:\n {response_data}"
        )
        return response_data["articles"]


def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    tool = kwargs["tool"]

    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    market_behavior = CloseMarketBehaviourMock(**kwargs)
    question = kwargs.pop("prompt", None)
    result = market_behavior._get_answer(question)
    return result
