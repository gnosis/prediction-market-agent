from typing import Type

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from prediction_market_agent_tooling.markets.data_models import Currency
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.utils import (
    MicroMarket,
    get_binary_markets,
)


class MarketTool(BaseTool):
    market_type: MarketType

    @property
    def currency(self) -> Currency:
        return self.market_type.market_class.currency


class GetMarkets(MarketTool):
    name: str = "get_markets"
    description: str = (
        "Use this function to get a list of predction market questions, and "
        "the corresponding market IDs"
    )

    def _run(self) -> list[str]:
        return [
            str(MicroMarket.from_agent_market(m))
            for m in get_binary_markets(market_type=self.market_type)
        ]


class GetMarketProbability(MarketTool):
    name: str = "get_market_probability"
    description = (
        f"Use this function to get the probability of a 'Yes' outcome for "
        f"a binary prediction market. This is equivalent to the price of "
        f"the 'Yes' token in the market currency. Pass in the market id as a "
        f"string."
    )

    class Input(BaseModel):
        market_id: str = Field(
            description="The ID of the market to get the probability of."
        )

    args_schema: Type[BaseModel] = Input

    def _run(self, market_id: str) -> str:
        return str(
            self.market_type.market_class.get_binary_market(id=market_id).current_p_yes
        )
