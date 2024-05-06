import typing as t

from microchain import Function
from prediction_market_agent_tooling.config import PrivateCredentials
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import Currency, TokenAmount
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.utils import (
    MicroMarket,
    get_balance,
    get_binary_markets,
    get_boolean_outcome,
    get_example_market_id,
    get_no_outcome,
    get_yes_outcome,
)
from prediction_market_agent.tools.mech.utils import (
    MechResponse,
    MechTool,
    mech_request,
    mech_request_local,
)
from prediction_market_agent.utils import APIKeys

from pydantic import BaseModel, Field

Operator = t.Literal["+", "-", "*", "/"]


class SumInput(BaseModel):
    a: t.Annotated[float, Field(description="The first number.")]
    b: t.Annotated[float, Field(description="The second number.")]


def sum_function(input: t.Annotated[SumInput, "Input to the calculator."]) -> float:
    return input.a + input.b


# assistant.register_for_llm(name="calculator", description="A calculator tool that accepts nested expression as input")(
#     calculator
# )
# user_proxy.register_for_execution(name="calculator")(calculator)
