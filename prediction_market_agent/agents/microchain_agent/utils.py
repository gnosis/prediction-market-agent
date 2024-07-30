import typing as t

import pandas as pd
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

from prediction_market_agent.agents.microchain_agent.memory import ChatHistory
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
    # Get the 15 markets that are closing soonest
    cls = market_type.market_class
    markets: t.Sequence[AgentMarket] = cls.get_binary_markets(
        filter_by=FilterBy.OPEN,
        sort_by=(
            SortBy.NONE
            if market_type == MarketType.POLYMARKET
            else SortBy.CLOSING_SOONEST
        ),
        limit=15,
    )
    return list(markets)


def get_balance(api_keys: APIKeys, market_type: MarketType) -> BetAmount:
    currency = market_type.market_class.currency
    if market_type == MarketType.OMEN:
        balances = get_balances(api_keys.bet_from_address)
        total_balance = balances.xdai + balances.wxdai
        return BetAmount(
            amount=total_balance,
            currency=currency,
        )
    else:
        raise ValueError(f"Market type '{market_type}' not supported")


def get_total_asset_value(api_keys: APIKeys, market_type: MarketType) -> BetAmount:
    balance = get_balance(api_keys, market_type)
    positions = market_type.market_class.get_positions(api_keys.bet_from_address)
    positions_value = market_type.market_class.get_positions_value(positions)

    return BetAmount(
        amount=balance.amount + positions_value.amount,
        currency=market_type.market_class.currency,
    )


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


def get_function_useage_from_history(
    chat_history: ChatHistory, agent: Agent
) -> pd.DataFrame:
    """
    Get the number of times each function is used in the chat history.

    Returns a DataFrame, indexed by the function names, with a column for the
    usage count.
    """
    function_names = [function for function in agent.engine.functions]
    function_useage = {function: 0 for function in function_names}
    for message in chat_history.chat_messages:
        for function in function_names:
            if message.content.startswith(f"{function}("):
                function_useage[function] += 1
                break

    return pd.DataFrame(
        data={"Usage Count": list(function_useage.values())},
        index=function_names,
    )
