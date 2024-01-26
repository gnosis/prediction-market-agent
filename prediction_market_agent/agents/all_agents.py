import typing as t
from enum import Enum
from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.agents.langchain_agent import LangChainAgent
from prediction_market_agent.agents.autogen_agent import AutoGenAgent
from prediction_market_agent.agents.always_yes import AlwaysYesAgent
from prediction_market_agent.agents.llamaindex_agent import LlamaIndexAgent
from prediction_market_agent.agents.metagpt_agent import MetaGPTAgent
from prediction_market_agent.agents.crewai_agent import CrewAIAgent
from prediction_market_agent.agents.custom_agent import (
    CustomAgentOpenAi,
    CustomAgentLlama,
)


class AgentType(str, Enum):
    LANGCHAIN = "langchain"
    AUTOGEN = "autogen"
    ALWAYS_YES = "always_yes"
    LLAMAINDEX = "llamaindex"
    METAGPT = "metagpt"
    CREWAI = "crewai"
    CUSTOM_OPENAI = "custom_openai"
    CUSTOM_LLAMA = "custom_llama"


AGENT_MAPPING: dict[AgentType, t.Type[AbstractAgent]] = {
    AgentType.LANGCHAIN: LangChainAgent,
    AgentType.AUTOGEN: AutoGenAgent,
    AgentType.ALWAYS_YES: AlwaysYesAgent,
    AgentType.LLAMAINDEX: LlamaIndexAgent,
    AgentType.METAGPT: MetaGPTAgent,
    AgentType.CREWAI: CrewAIAgent,
    AgentType.CUSTOM_OPENAI: CustomAgentOpenAi,
    AgentType.CUSTOM_LLAMA: CustomAgentLlama,
}


def get_agent(agent_type: AgentType) -> AbstractAgent:
    return AGENT_MAPPING[agent_type]()
