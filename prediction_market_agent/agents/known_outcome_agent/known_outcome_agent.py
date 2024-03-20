import json
import typing as t
from datetime import datetime
from enum import Enum

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from prediction_market_agent.tools.web_scrape.basic_summary import _summary
from prediction_market_agent.tools.web_scrape.markdown import web_scrape
from prediction_market_agent.tools.web_search.tavily import web_search


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

    def to_boolean(self) -> bool:
        if self is Result.YES:
            return True
        elif self is Result.NO:
            return False
        else:
            raise ValueError("Unexpected result")


class Answer(BaseModel):
    result: Result
    reasoning: str

    def has_known_outcome(self) -> bool:
        return self.result is not Result.UNKNOWN


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

You goal is to determine whether the answer can be inferred with a reasonable
degree of certainty from the scraped web content. Answer in json format with the
following fields:
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


def completion_str_to_json(completion: str) -> dict[str, t.Any]:
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
    completion_dict: dict[str, t.Any] = json.loads(completion)
    return completion_dict


def summarize_if_required(content: str, model: str, question: str) -> str:
    """
    If the content is too long to fit in the model's context, summarize it.
    """
    if model == "gpt-3.5-turbo-0125":  # 16k context length
        max_length = 10000
    elif model == "gpt-4-1106-preview":  # 128k context length
        max_length = 100000
    else:
        raise ValueError(f"Unknown model: {model}")

    if len(content) > max_length:
        breakpoint()
        return _summary(content=content, objective=question, separators=["  "])
    else:
        return content


def get_known_outcome(model: str, question: str, max_tries: int) -> Answer:
    """
    In a loop, perform web search and scrape to find if the answer to the
    question is known. Break if the answer is found, or after a certain number
    of tries, and no definite answer is found, return an 'unknown' answer.
    """
    tries = 0
    date_str = datetime.now().strftime("%d %B %Y")
    previous_urls = []
    llm = ChatOpenAI(model=model, temperature=0.4)
    while tries < max_tries:
        search_prompt = ChatPromptTemplate.from_template(
            template=GENERATE_SEARCH_QUERY_PROMPT
        ).format_messages(date_str=date_str, question=question)
        search_query = str(llm.invoke(search_prompt).content).strip('"')
        search_results = web_search(query=search_query, max_results=5)
        if not search_results:
            raise ValueError("No search results found.")

        for result in search_results:
            if result.url in previous_urls:
                continue
            previous_urls.append(result.url)

            scraped_content = web_scrape(url=result.url)
            scraped_content = summarize_if_required(
                content=scraped_content, model=model, question=question
            )

            prompt = ChatPromptTemplate.from_template(
                template=ANSWER_FROM_WEBSCRAPE_PROMPT
            ).format_messages(
                date_str=date_str,
                question=question,
                scraped_content=scraped_content,
            )
            answer = str(llm.invoke(prompt).content)
            parsed_answer = Answer.model_validate(completion_str_to_json(answer))

            if parsed_answer.result is not Result.UNKNOWN:
                return parsed_answer

        tries += 1

    return Answer(result=Result.UNKNOWN, reasoning="Max tries exceeded.")
