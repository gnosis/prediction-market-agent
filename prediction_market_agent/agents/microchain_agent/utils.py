import typing as t

from microchain import Agent
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
from prediction_market_agent_tooling.markets.omen.data_models import (
    get_boolean_outcome as get_omen_boolean_outcome,
)
from prediction_market_agent_tooling.tools.balances import get_balances
from pydantic import BaseModel

from prediction_market_agent.utils import APIKeys


class MicroMarket(BaseModel):
    question: str
    id: str

    @staticmethod
    def from_agent_market(market: AgentMarket) -> "MicroMarket":
        return MicroMarket(
            question=market.question,
            id=market.id,
        )

    def __str__(self) -> str:
        return f"'{self.question}', id: {self.id}"


def get_binary_markets(market_type: MarketType) -> list[AgentMarket]:
    # Get the 5 markets that are closing soonest
    cls = market_type.market_class
    markets: t.Sequence[AgentMarket] = cls.get_binary_markets(
        filter_by=FilterBy.OPEN,
        sort_by=(
            SortBy.NONE
            if market_type == MarketType.POLYMARKET
            else SortBy.CLOSING_SOONEST
        ),
        limit=5,
    )
    return list(markets)


def get_balance(market_type: MarketType) -> BetAmount:
    currency = market_type.market_class.currency
    if market_type == MarketType.OMEN:
        # We focus solely on xDAI balance for now to avoid the agent having to wrap/unwrap xDAI.
        return BetAmount(
            amount=get_balances(APIKeys().bet_from_address).xdai,
            currency=currency,
        )
    else:
        raise ValueError(f"Market type '{market_type}' not supported")


def get_boolean_outcome(market_type: MarketType, outcome: str) -> bool:
    if market_type == MarketType.OMEN:
        return get_omen_boolean_outcome(outcome)
    else:
        raise ValueError(f"Market type '{market_type}' not supported")


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


def get_example_market_id(market_type: MarketType) -> str:
    if market_type == MarketType.OMEN:
        return "0x0020d13c89140b47e10db54cbd53852b90bc1391"
    else:
        raise ValueError(f"Market type '{market_type}' not supported")


def get_initial_history_length(agent: Agent) -> int:
    initialized_history_length = 1
    if agent.bootstrap:
        initialized_history_length += len(agent.bootstrap) * 2
    return initialized_history_length


def has_been_run_past_initialization(agent: Agent) -> bool:
    if not hasattr(agent, "history"):
        return False

    return len(agent.history) > get_initial_history_length(agent)
