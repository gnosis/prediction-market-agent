from datetime import datetime
from enum import Enum

from langchain.prompts import ChatPromptTemplate
from langchain_core.callbacks import Callbacks
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic import BaseModel

from prediction_market_agent.tools.web_scrape.basic_summary import _summary
from prediction_market_agent.tools.web_scrape.markdown import web_scrape
from prediction_market_agent.tools.web_search.tavily import web_search
from prediction_market_agent.utils import APIKeys, completion_str_to_json


class Result(str, Enum):
    """
    With perfect information, a binary question should have one of three answers:
    YES, NO, or KNOWN_UNKNOWABLE:

    - Will the sun rise tomorrow? YES
    - Will the Bradley Cooper win best actor at the 2024 oscars by 01/04/2024? NO (because the event has already happened, and Cillian Murphy won)
    - Will over 360 pople have died while climbing Mount Everest by 1st Jan 2028? KNOWN_UNKNOWABLE (because the closing date has not happened yet, and we cannot predict the outcome with reasoanable certainty)

    but since the agent's information is based on web scraping, and is therefore
    imperfect, we allow it to defer answering definitively via the UNKNOWN result.
    """

    YES = "YES"
    NO = "NO"
    KNOWN_UNKNOWABLE = "KNOWN_UNKNOWABLE"
    UNKNOWN = "UNKNOWN"

    def to_p_yes(self) -> Probability:
        if self is Result.YES:
            return Probability(1.0)
        elif self is Result.NO:
            return Probability(0.0)
        else:
            raise ValueError("Unexpected result")

    def to_boolean(self) -> bool:
        if self is Result.YES:
            return True
        elif self is Result.NO:
            return False
        else:
            raise ValueError("Unexpected result")

    @property
    def is_known(self) -> bool:
        return self in [Result.YES, Result.NO]


class KnownOutcomeOutput(BaseModel):
    result: Result
    reasoning: str

    def has_known_result(self) -> bool:
        return self.result.is_known


HAS_QUESTION_HAPPENED_IN_THE_PAST_PROMPT = """
The current date is {date_str}. Your goal is to assert if a QUESTION references an event that is already finished (according to the current date and time) or if it will still take place in a later date. 

For example, you should return 1 if given the event "Will Bitcoin have reached the price of $100 by 30 March 2023?", since the event ends on a data prior to the current date.

Your answer MUST be an integer and follow the logic below:
- If the event is already finished, return 1
- If the event has not yet finished, return 0
- If you are not sure, return -1

Answer with the single 1, 0 or -1 only, and nothing else.

[QUESTION]
"{question}"
"""

GENERATE_SEARCH_QUERY_PROMPT = """
The current date is {date_str}. You are trying to determine whether the answer
to the following question has a definite answer. Generate a web search query
based on the question to find relevant information.

"{question}"

For example, if the question is:

"Will Arsenal reach the Champions League semi-finals on 19 March 2025?"

You might generate the following search query:

"Champions League semi-finals draw 2025"

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
question is yes), 'NO' (if the answer to the question is no), 'KNOWN_UNKNOWABLE'
(if we can answer with a reasonable degree of certainty from the web-scraped
information that the question cannot be answered either way for the time being),
or 'UNKNOWN' if you are unable to give one of the above answers with a
reasonable degree of certainty from the web-scraped information. Your answer
should only contain this json string, and nothing else.

If the question is of the format: "Will X happen by Y?" then the result should
be as follows:
- If X has already happened, the result is 'YES'.
- If not-X has already happened, the result is 'NO'.
- If X has been announced to happen after Y, result 'NO'.
- If you are confident that none of the above are the case, and the result will only be knowable in the future, or not at all, the result is 'KNOWN_UNKNOWABLE'.
- Otherwise, the result is 'UNKNOWN'.

If the question is of the format: "Will X happen on Y?" then the result should
be as follows:
- If something has happened that necessarily prevents X from happening on Y, the result is 'NO'.
- If you are confident that nothing has happened that necessarily prevents X from happening on Y, the result is 'KNOWN_UNKNOWABLE'.
- Otherwise, the result is 'UNKNOWN'.
"""


def summarize_if_required(content: str, model: str, question: str) -> str:
    """
    If the content is too long to fit in the model's context, summarize it.
    """
    if model == "gpt-3.5-turbo-0125":  # 16k context length
        max_length = 10000
    elif model == "gpt-4-1106-preview":  # 128k context length
        max_length = 100000
    else:
        raise ValueError(
            f"Unknown model `{model}`, please add him to the `summarize_if_required` function."
        )

    if len(content) > max_length:
        return _summary(content=content, objective=question, separators=["  "])
    else:
        return content


def has_question_event_happened_in_the_past(
    model: str, question: str, callbacks: Callbacks
) -> bool:
    """Asks the model if the event referenced by the question has finished (given the
    current date) (returning 1), if the event has not yet finished (returning 0) or
     if it cannot be sure (returning -1)."""
    date_str = utcnow().strftime("%Y-%m-%d %H:%M:%S %Z")
    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
        api_key=APIKeys().openai_api_key.get_secret_value(),
        callbacks=callbacks,
    )
    prompt = ChatPromptTemplate.from_template(
        template=HAS_QUESTION_HAPPENED_IN_THE_PAST_PROMPT
    ).format_messages(
        date_str=date_str,
        question=question,
    )
    answer = str(llm.invoke(prompt).content)
    try:
        parsed_answer = int(answer)
        if parsed_answer == 1:
            return True
    except Exception as e:
        logger.error(
            f"Exception occured, cannot assert if title happened in the past because of '{e}'."
        )

    return False


def get_known_outcome(
    model: str, question: str, max_tries: int, callbacks: Callbacks = None
) -> KnownOutcomeOutput:
    """
    In a loop, perform web search and scrape to find if the answer to the
    question is known. Break if the answer is found, or after a certain number
    of tries, and no definite answer is found, return an 'unknown' answer.
    """
    tries = 0
    date_str = datetime.now().strftime("%d %B %Y")
    previous_urls = []
    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
        api_key=APIKeys().openai_api_key.get_secret_value(),
        callbacks=callbacks,
    )
    while tries < max_tries:
        search_prompt = ChatPromptTemplate.from_template(
            template=GENERATE_SEARCH_QUERY_PROMPT
        ).format_messages(date_str=date_str, question=question)
        logger.debug(f"Invoking LLM for the prompt '{search_prompt[0]}'")
        search_query = str(llm.invoke(search_prompt).content).strip('"')
        logger.debug(f"Searching web for the search query '{search_query}'")
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
            parsed_answer = KnownOutcomeOutput.model_validate(
                completion_str_to_json(answer)
            )

            if parsed_answer.result is not Result.UNKNOWN:
                return parsed_answer

        tries += 1

    return KnownOutcomeOutput(result=Result.UNKNOWN, reasoning="Max tries exceeded.")
