from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.markets.agent_market import (
    AgentMarket,
    FilterBy,
    SortBy,
)
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    OmenConditionalTokenContract,
)
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from web3.types import Wei


def get_omen_binary_markets() -> list[OmenAgentMarket]:
    # Get the 5 markets that are closing soonest
    markets: list[AgentMarket] = OmenAgentMarket.get_binary_markets(
        filter_by=FilterBy.OPEN,
        sort_by=SortBy.CLOSING_SOONEST,
        limit=5,
    )


def get_omen_binary_market_from_question(market: str) -> OmenAgentMarket:
    markets = get_omen_binary_markets()
    for m in markets:
        if m.question == market:
            return m
    raise ValueError(f"Market '{market}' not found")


def get_omen_market_token_balance(
    user_address: ChecksumAddress, market_condition_id: HexBytes, market_index_set: int
) -> Wei:
    # We get the multiple positions for each market
    positions = OmenSubgraphHandler().get_positions(market_condition_id)
    # Find position matching market_outcome
    position_for_index_set = next(
        p for p in positions if p.indexSets.__contains__(market_index_set)
    )
    position_as_int = int(position_for_index_set.id.hex(), 16)
    balance = OmenConditionalTokenContract().balanceOf(user_address, position_as_int)
    return balance
