from enum import Enum

from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.memory import (
    MemoryContainer,
)
from prediction_market_agent.utils import APIKeys


class LongTermMemoryTaskIdentifier(str, Enum):
    THINK_THOROUGHLY = "think-thoroughly-agent"
    MICROCHAIN_AGENT_OMEN = "microchain-agent-deployment-omen"
    MICROCHAIN_AGENT_STREAMLIT = "microchain-streamlit-app"

    @staticmethod
    def microchain_task_from_market(market_type: MarketType):
        if market_type == MarketType.OMEN:
            return LongTermMemoryTaskIdentifier.MICROCHAIN_AGENT_OMEN
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


def market_is_saturated(market: AgentMarket) -> bool:
    return market.current_p_yes > 0.95 or market.current_p_no > 0.95


def memories_to_learnings(memories: list[MemoryContainer], model: str) -> str:
    """
    Synthesize the memories into an intelligible summary that represents the
    past learnings.
    """
    llm = ChatOpenAI(
        temperature=0,
        model=model,
        api_key=APIKeys().openai_api_key.get_secret_value(),
    )
    summary_chain = load_summarize_chain(
        llm=llm,
        prompt=PromptTemplate.from_template(MEMORIES_TO_LEARNINGS_TEMPLATE),
        document_variable_name="memories",
        chain_type="stuff",
        verbose=False,
    )

    memory_docs = [Document(page_content=str(m)) for m in memories]
    summary: str = summary_chain.run(input_documents=memory_docs)
    return summary
