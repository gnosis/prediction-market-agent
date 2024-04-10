from decimal import Decimal

from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.markets.agent_market import (
    AgentMarket,
    FilterBy,
    SortBy,
)
from prediction_market_agent_tooling.markets.data_models import BetAmount
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.data_models import (
    OMEN_FALSE_OUTCOME,
    OMEN_TRUE_OUTCOME,
)
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    OmenConditionalTokenContract,
)
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from pydantic import BaseModel
from web3.types import Wei

from prediction_market_agent.utils import APIKeys


class MicroMarket(BaseModel):
    question: str
    p_yes: float

    @staticmethod
    def from_agent_market(market: AgentMarket) -> "MicroMarket":
        return MicroMarket(
            question=market.question,
            p_yes=float(market.p_yes),
        )

    def __str__(self) -> str:
        return f"'{self.question}' with probability of yes: {self.p_yes:.2%}"


def get_binary_markets(market_type: MarketType) -> list[AgentMarket]:
    # Get the 5 markets that are closing soonest
    cls = market_type.market_class
    markets: list[AgentMarket] = cls.get_binary_markets(
        filter_by=FilterBy.OPEN,
        sort_by=(
            SortBy.NONE
            if market_type == MarketType.POLYMARKET
            else SortBy.CLOSING_SOONEST
        ),
        limit=5,
    )
    return markets


def get_balance(market_type: MarketType) -> BetAmount:
    currency = market_type.market_class.currency
    if market_type == MarketType.OMEN:
        # We focus solely on xDAI balance for now to avoid the agent having to wrap/unwrap xDAI.
        return BetAmount(
            amount=Decimal(get_balances(APIKeys().bet_from_address).xdai),
            currency=currency,
        )
    else:
        raise ValueError(f"Market type '{market_type}' not supported")


def get_binary_market_from_question(
    market: str, market_type: MarketType
) -> AgentMarket:
    markets = get_binary_markets(market_type=market_type)
    for m in markets:
        if m.question == market:
            return m
    raise ValueError(f"Market '{market}' not found")


def get_market_token_balance(
    user_address: ChecksumAddress, market_condition_id: HexBytes, market_index_set: int
) -> Wei:
    # We get the multiple positions for each market
    positions = OmenSubgraphHandler().get_positions(market_condition_id)
    # Find position matching market_outcome
    position_for_index_set = next(
        p for p in positions if market_index_set in p.indexSets
    )
    position_as_int = int(position_for_index_set.id.hex(), 16)
    balance = OmenConditionalTokenContract().balanceOf(user_address, position_as_int)
    return balance


def get_yes_outcome(market_type: MarketType) -> str:
    if market_type == MarketType.OMEN:
        return OMEN_TRUE_OUTCOME
    else:
        raise ValueError(f"Market type '{market_type}' not supported")


def get_no_outcome(market_type: MarketType) -> str:
    if market_type == MarketType.OMEN:
        return OMEN_FALSE_OUTCOME
    else:
        raise ValueError(f"Market type '{market_type}' not supported")