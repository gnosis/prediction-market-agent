import json
from datetime import datetime
from enum import Enum

import requests
import tenacity
from bs4 import BeautifulSoup
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from markdown_it import MarkdownIt
from prediction_market_agent_tooling.tools.cache import persistent_inmemory_cache
from prediction_market_agent_tooling.tools.utils import secret_str_from_env
from pydantic import BaseModel
from requests import Response
from tavily import TavilyClient

from prediction_market_agent.tools.web_scrape_structured import web_scrape_structured


class Result(str, Enum):
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"

    def to_p_yes(self) -> float:
        if self is Result.YES:
            return 1.0
        elif self is Result.NO:
            return 0.0
        else:
            raise ValueError("Unexpected result")


class WebSearchResult(BaseModel):
    url: str
    query: str


class Answer(BaseModel):
    result: Result
    reasoning: str


GENERATE_SEARCH_QUERY_PROMPT = """
The current date is {date_str}. You are trying to determine whether the answer
to the following question has a definite answer. Generate a web search query
based on the question to find relevant information.

"{question}"

For example, if the question is:

"Will Arsenal reach the Champions League semi-finals on 19 March 2025?"

You might generate the following search query:

"Champions League semi-finals draw 2024"

Answer with the single prompt only, and nothing else.
"""


ANSWER_FROM_WEBSCRAPE_PROMPT = """
The current date is {date_str}. You are an expert researcher trying to answer a
question based on information scraped from the web. The question is:

```
{question}
```

The information you have scraped from the web is:

```
{scraped_content}
```

Answer in json format with the following fields:
- "result": "<RESULT>",
- "reasoning": "<REASONING>",

where <REASONING> is a free text field containing your reasoning, and <RESULT>
is a multiple-choice field containing only one of 'YES' (if the answer to the
question is yes), 'NO' (if the answer to the question is no), or 'UNKNOWN' if
you are unable to answer the question with a reasonable degree of certainty from
the web-scraped information. Your answer should only contain this json string,
and nothing else.

If the question is of the format: "Will X happen by Y?" then the result should
be as follows:
- If X has already happened, the result is 'YES'.
- If not-X has already happened, the result is 'NO'.
- If X has been announced to happen after Y, result 'NO'.
- Otherwise, the result is 'UNKNOWN'.

If the question is of the format: "Will X happen on Y?"
- If something has happened that necessarily prevents X from happening on Y, the result is 'NO'.
- Otherwise, the result is 'UNKNOWN'.
"""


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(1), reraise=True
)
@persistent_inmemory_cache
def web_search(query: str, max_results: int) -> list[WebSearchResult]:
    """
    Web search using Tavily API.

    TODO replace with GoogleSearchTool?
    """
    tavily_api_key = secret_str_from_env("TAVILY_API_KEY")
    tavily = TavilyClient(api_key=tavily_api_key.get_secret_value())
    response = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=True,
    )

    results = [
        WebSearchResult(
            url=result["url"],
            query=query,
        )
        for result in response["results"]
    ]

    return results


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(1), reraise=True
)
@persistent_inmemory_cache
def fetch_html(url: str, timeout: int) -> Response:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0"
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    return response


def web_scrape(url: str, timeout: int = 10) -> str:
    """
    Taken from evo.prophet
    """
    try:
        response = fetch_html(url=url, timeout=timeout)

        if "text/html" in response.headers.get("Content-Type", ""):
            soup = BeautifulSoup(response.content, "html.parser")

            [x.extract() for x in soup.findAll("script")]
            [x.extract() for x in soup.findAll("style")]
            [x.extract() for x in soup.findAll("noscript")]
            [x.extract() for x in soup.findAll("link")]
            [x.extract() for x in soup.findAll("head")]
            [x.extract() for x in soup.findAll("image")]
            [x.extract() for x in soup.findAll("img")]

            text: str = soup.get_text()
            text = MarkdownIt(text)
            text = "  ".join([x.strip() for x in text.split("\n")])
            text = " ".join([x.strip() for x in text.split("  ")])

            return text
        else:
            print("Non-HTML content received")
            return ""

    except requests.RequestException as e:
        print(f"HTTP request failed: {e}")
        return ""


def completion_str_to_json(completion: str) -> dict:
    """
    Cleans completion JSON in form of a string:

    ```json
    {
        ...
    }
    ```

    into just { ... }
    ```
    """
    start_index = completion.find("{")
    end_index = completion.rfind("}")
    completion = completion[start_index : end_index + 1]
    return json.loads(completion)


def get_known_outcome(model: str, question: str, max_tries: int) -> Answer:
    tries = 0
    date_str = datetime.now().strftime("%d %B %Y")
    previous_urls = []
    llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.4)
    while tries < max_tries:
        search_prompt = ChatPromptTemplate.from_template(
            template=GENERATE_SEARCH_QUERY_PROMPT
        ).format_messages(date_str=date_str, question=question)
        search_query = llm.invoke(search_prompt).content.strip('"')
        search_results = web_search(query=search_query, max_results=5)
        if not search_results:
            raise ValueError("No search results found.")

        for result in search_results:
            if result.url in previous_urls:
                continue
            previous_urls.append(result.url)
            # scraped_content = web_scrape(url=result.url)
            scraped_content = web_scrape_structured(url=result.url)
            prompt = ChatPromptTemplate.from_template(
                template=ANSWER_FROM_WEBSCRAPE_PROMPT
            ).format_messages(
                date_str=date_str,
                question=question,
                scraped_content=scraped_content,
            )
            try:
                answer = llm.invoke(prompt).content
                parsed_answer = Answer.model_validate(completion_str_to_json(answer))
            except ValueError as e:
                breakpoint()
            if parsed_answer.result is not Result.UNKNOWN:
                return parsed_answer

        tries += 1

    return Answer(result=Result.UNKNOWN, reasoning="Max tries exceeded.")
