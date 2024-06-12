import json
import typing as t

import pytest
from microchain import LLM, Agent, Engine, Function, OpenAIChatGenerator
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType
from pydantic import BaseModel

from prediction_market_agent.agents.microchain_agent.functions_from_tools import (
    microchain_function_from_tool,
)
from prediction_market_agent.agents.microchain_agent.market_functions import (
    GetMarketProbability,
)
from prediction_market_agent.tools.prediction_market import GetMarkets
from prediction_market_agent.utils import APIKeys
from tests.utils import RUN_PAID_TESTS


@pytest.fixture
def generator() -> OpenAIChatGenerator:
    return OpenAIChatGenerator(
        model="gpt-4-turbo-preview",
        api_key=APIKeys().openai_api_key.get_secret_value(),
        api_base="https://api.openai.com/v1",
        temperature=0.0,
    )


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_get_probability(
    market_type: MarketType, generator: OpenAIChatGenerator
) -> None:
    """
    Test the the agent's ability to use the GetMarkets and GetMarketProbability
    functions.

    The agent should be able to find a market and its probability, and return
    it in valid json format.
    """

    class MarketIDAndProbability(BaseModel):
        market_id: str
        probability: float

    class Jsonify(Function):
        @property
        def description(self) -> str:
            return "Use this function to jsonify the market id and probability"

        @property
        def example_args(self) -> list[t.Any]:
            return ["0x1234", 0.5]

        def __call__(self, market_id: str, p_yes: float) -> str:
            return MarketIDAndProbability(
                market_id=market_id, probability=p_yes
            ).model_dump_json()

    engine = Engine()
    engine.register(Reasoning())
    engine.register(Stop())
    get_markets = microchain_function_from_tool(
        GetMarkets(market_type=market_type), example_args=[]
    )
    engine.register(get_markets)
    engine.register(GetMarketProbability(market_type=market_type))
    engine.register(Jsonify())
    agent = Agent(llm=LLM(generator=generator), engine=engine)
    agent.system_prompt = f"""Act as a agent to find any market and its probability, and return it in valid json format.
    
    You can use the following functions:

    {engine.help}
    
    Only output valid Python function calls. When you have a market's id and
    probability, return it in valid json format, with fields 'market_id' and 'probability'.

    Once you have output the valid json, then stop.
    """

    agent.bootstrap = ['Reasoning("I need to reason step-by-step")']
    agent.run(iterations=5)

    # history[-1] is 'user' stop message
    # history[-2] is 'assistant' stop function call
    m_json = json.loads(agent.history[-3]["content"])
    m = MarketIDAndProbability.model_validate(m_json)
    market: AgentMarket = market_type.market_class.get_binary_market(m.market_id)
    assert market.current_p_yes == m.probability
