from datetime import datetime
from enum import Enum
from string import Template

from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.memory import (
    DatedChatMessage,
    SimpleMemoryThinkThoroughly,
)
from prediction_market_agent.utils import APIKeys


class AgentIdentifier(str, Enum):
    THINK_THOROUGHLY = "think-thoroughly-agent"
    MICROCHAIN_AGENT_OMEN = "microchain-agent-deployment-omen"
    MICROCHAIN_AGENT_STREAMLIT = "microchain-streamlit-app"

    @staticmethod
    def microchain_task_from_market(
        market_type: MarketType,
    ) -> "AgentIdentifier":
        if market_type == MarketType.OMEN:
            return AgentIdentifier.MICROCHAIN_AGENT_OMEN
        else:
            raise ValueError(f"Market {market_type} not supported.")


MEMORIES_TO_LEARNINGS_TEMPLATE = """
You are an agent that trades in prediction markets. You are aiming to improve
your strategy over time. You have a collection of memories that record your
actions, and your reasoning behind them.

Analyse the pattern of actions, and very concisely summarise this into a list of
'past learnings'. Consider whether you made good decisions, and if you have made
any mistakes, and what you have learned from them.

Each memory comes with a timestamp. If the memories are clustered into
different times, then make a separate list for each cluster. Refer to each
cluster as a 'Trading Session', and display the range of timestamps for each.

MEMORIES:
{memories}
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
    model: str = "gpt-4o-2024-05-13",
) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model=model,
        api_key=APIKeys().openai_api_key,
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


def memories_to_learnings(memories: list[DatedChatMessage], model: str) -> str:
    """
    Synthesize the memories into an intelligible summary that represents the
    past learnings.
    """
    prompt = PromptTemplate.from_template(MEMORIES_TO_LEARNINGS_TEMPLATE)
    return _summarize_learnings(
        memories=[str(m) for m in memories],
        prompt_template=prompt,
        model=model,
    )


def get_event_date_from_question(question: str) -> datetime | None:
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0.0,
        api_key=APIKeys().openai_api_key,
    )
    event_date_str = str(
        llm.invoke(
            f"Extract the event date in the format `%m-%d-%Y` from the following question, don't write anything else, only the event date in the given format: `{question}`"
        ).content
    ).strip("'`\"")

    try:
        event_date = datetime.strptime(event_date_str, "%m-%d-%Y")
    except ValueError:
        logger.error(
            f"Could not extract event date from question `{question}`, got `{event_date_str}`."
        )
        return None

    return event_date
