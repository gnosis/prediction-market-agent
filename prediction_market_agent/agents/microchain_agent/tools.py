from eth_typing import HexStr, HexAddress, ChecksumAddress
from prediction_market_agent_tooling.markets.agent_market import (
    AgentMarket,
    FilterBy,
    SortBy,
)
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_contracts import OmenConditionalTokenContract, \
    OmenFixedProductMarketMakerContract
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import OmenSubgraphHandler
from web3 import Web3
from web3.types import Wei


def address_to_checksum_address(address: str) -> ChecksumAddress:
    return ChecksumAddress(HexAddress(HexStr(address)))

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


def find_index_set_for_market_outcome(market: OmenAgentMarket, market_outcome: str):
    try:
        market_outcome_match = market.outcomes.index(market_outcome)
        # Index_sets start at 1
        return market_outcome_match + 1
    except ValueError as e:
        print (f"Market outcome {market_outcome} not present in market. Available outcomes: {market.outcomes}")
        raise e


def get_omen_market_token_balance(user_address: ChecksumAddress, market: OmenAgentMarket, market_outcome: str) -> Wei:
    # We get the multiple positions for each market
    positions = OmenSubgraphHandler().get_positions(market.condition.id)
    # Find position matching market_outcome
    index_set = find_index_set_for_market_outcome(market, market_outcome)
    position_for_index_set = next(p for p in positions if p.indexSets.__contains__(index_set))
    position_as_int = int(position_for_index_set.id.hex(), 16)
    balance = OmenConditionalTokenContract().balanceOf(user_address, position_as_int)
    return balance
