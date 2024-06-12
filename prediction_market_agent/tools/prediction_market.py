# from typing import Type

# from langchain.pydantic_v1 import BaseModel, Field
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

    # class Input(BaseModel):
    #     foo: str = Field(description="bar")
    # args_schema: Type[BaseModel] = Input

    def _run(self) -> list[str]:
        return [
            str(MicroMarket.from_agent_market(m))
            for m in get_binary_markets(market_type=self.market_type)
        ]
