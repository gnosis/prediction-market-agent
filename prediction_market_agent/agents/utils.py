from string import Template

from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.tools.langfuse_ import (
    get_langfuse_langchain_config,
    observe,
)
from prediction_market_agent_tooling.tools.utils import DatetimeUTC

from prediction_market_agent.agents.microchain_agent.memory import (
    DatedChatMessage,
    SimpleMemoryThinkThoroughly,
)
from prediction_market_agent.utils import DEFAULT_OPENAI_MODEL, APIKeys

STREAMLIT_TAG = "streamlit"


MEMORIES_TO_LEARNINGS_TEMPLATE = """
You are an agent that does actions on its own. You are aiming to improve
your strategy over time. You have a collection of memories that record your
actions, and your reasoning behind them.

Analyse the pattern of actions, and very concisely summarise this into a list of
'past learnings'. Consider whether you made good decisions, and if you have made
any mistakes, and what you have learned from them.

Each memory comes with a timestamp. If the memories are clustered into
different times, then make a separate list for each cluster. Refer to each
cluster as a 'Session', and display the range of timestamps for each.

{additional_prompt}

MEMORIES:
{{memories}}
"""

EXTRACT_REASONINGS_TEMPLATE = """
You are an agent that trades in prediction markets. You have a collection of memories that record your
actions, and your reasoning behind them.

Below you can find a TWEET related to previous trades that you placed. Produce a summary of the reasonings, extracted from the memories, that can explain why the trade was placed.

TWEET: $TWEET

MEMORIES:
{memories}
"""


def market_is_saturated(market: AgentMarket) -> bool:
    return market.current_p_yes > 0.95 or market.current_p_no > 0.95


def _summarize_learnings(
    memories: list[str],
    prompt_template: PromptTemplate,
    model: str = DEFAULT_OPENAI_MODEL,
) -> str:
    model = model if model else DEFAULT_OPENAI_MODEL
    llm = ChatOpenAI(
        temperature=0,
        model=model,
        api_key=APIKeys().openai_api_key_secretstr_v1,
    )
    summary_chain = load_summarize_chain(
        llm=llm,
        prompt=prompt_template,
        document_variable_name="memories",
        chain_type="stuff",
        verbose=False,
    )

    memory_docs = [Document(page_content=m) for m in memories]
    summary: str = summary_chain.run(input_documents=memory_docs)
    return summary


def extract_reasonings_to_learnings(
    memories: list[SimpleMemoryThinkThoroughly], tweet: str
) -> str:
    prompt_with_tweet = Template(EXTRACT_REASONINGS_TEMPLATE).substitute(TWEET=tweet)
    return _summarize_learnings(
        [str(m) for m in memories],
        PromptTemplate.from_template(prompt_with_tweet),
    )


def memories_to_learnings(
    memories: list[DatedChatMessage],
    model: str = DEFAULT_OPENAI_MODEL,
    question: str | None = None,
) -> str:
    """
    Synthesize the memories into an intelligible summary that represents the
    past learnings.
    """
    prompt = PromptTemplate.from_template(
        MEMORIES_TO_LEARNINGS_TEMPLATE.format(
            additional_prompt=(
                f"Process memories to answer the following question: {question}"
                if question
                else ""
            )
        )
    )
    return _summarize_learnings(
        memories=[str(m) for m in memories],
        prompt_template=prompt,
        model=model,
    )


@observe()
def get_event_date_from_question(question: str) -> DatetimeUTC | None:
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0.0,
        api_key=APIKeys().openai_api_key_secretstr_v1,
    )
    event_date_str = str(
        llm.invoke(
            f"Extract the event date in the format `%m-%d-%Y` from the following question, don't write anything else, only the event date in the given format: `{question}`",
            config=get_langfuse_langchain_config(),
        ).content
    ).strip("'`\"")

    try:
        event_date = DatetimeUTC.to_datetime_utc(event_date_str)
    except ValueError:
        logger.warning(
            f"Could not extract event date from question `{question}`, got `{event_date_str}`."
        )
        return None

    return event_date


def get_maximum_possible_bet_amount(
    min_: float, max_: float, trading_balance: float
) -> float:
    trading_balance *= 0.95  # Allow to use only most of the trading balance, to keep something to pay for fees on markets where it's necessary.
    # Require bet size of at least `min_` and maximum `max_`, use available trading balance if its between.
    return min(max(min_, trading_balance), max_)
