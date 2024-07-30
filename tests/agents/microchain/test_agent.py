import json
import typing as t

import pytest
from microchain import LLM, Agent, Engine, Function, OpenAIChatGenerator
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType
from pydantic import BaseModel

from prediction_market_agent.agents.microchain_agent.blockchain.contract_class_converter import (
    ContractClassConverter,
)
from prediction_market_agent.agents.microchain_agent.blockchain.models import (
    AbiItemStateMutabilityEnum,
)
from prediction_market_agent.agents.microchain_agent.market_functions import (
    GetMarketProbability,
    GetMarkets,
)
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

    engine = Engine()
    engine.register(Reasoning())
    engine.register(Stop())
    engine.register(GetMarkets(market_type=market_type, keys=APIKeys()))
    engine.register(GetMarketProbability(market_type=market_type, keys=APIKeys()))
    engine.register(Jsonify())
    agent = Agent(llm=LLM(generator=generator), engine=engine)
    agent.system_prompt = f"""Act as a agent to find any market and its probability, and return it in valid json format.
    
    You can use the following functions:

    {engine.help}
    
    Only output valid Python function calls. When you have a market's id and
    probability, return it in valid json format, with fields 'market_id' and 'probability'.

    Once you have output the valid json, then stop.
    """

    agent.run(iterations=5)

    # history[-1] is 'user' stop message
    # history[-2] is 'assistant' stop function call
    m_json = json.loads(agent.history[-3]["content"])
    m = MarketIDAndProbability.model_validate(m_json)
    market: AgentMarket = market_type.market_class.get_binary_market(m.market_id)
    assert market.current_p_yes == m.probability


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_get_decimals(
    generator: OpenAIChatGenerator,
    wxdai_contract_class_converter: ContractClassConverter,
) -> None:
    engine = Engine()
    engine.register(Reasoning())
    engine.register(Stop())

    function_types_to_classes = (
        wxdai_contract_class_converter.create_classes_from_smart_contract()
    )

    view_classes = function_types_to_classes[AbiItemStateMutabilityEnum.VIEW]
    for clz in view_classes:
        engine.register(clz())

    agent = Agent(llm=LLM(generator=generator), engine=engine)
    agent.system_prompt = f"""Act as a agent to query the number of decimals from a smart contract.

        You can use the following functions:

        {engine.help}

        Only output valid Python function calls. When you have requested the number of decimals, return it as an integer, then stop.
        """

    agent.run(iterations=5)

    # history[-1] is 'user' stop message
    # history[-2] is 'assistant' stop function call
    result = agent.history[-3]["content"]
    assert int(result) == 18
