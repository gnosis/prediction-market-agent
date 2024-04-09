import os
from typing import List, cast

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
from prediction_market_agent_tooling.tools.web3_utils import private_key_to_public_key
from pydantic import BaseModel, SecretStr
from web3.types import Wei


class MicroMarket(BaseModel):
    question: str
    p_yes: float

    @staticmethod
    def from_agent_market(market: OmenAgentMarket) -> "MicroMarket":
        return MicroMarket(
            question=market.question,
            p_yes=float(market.p_yes),
        )

    def __str__(self) -> str:
        return f"'{self.question}' with probability of yes: {self.p_yes:.2%}"


def get_omen_binary_markets() -> list[OmenAgentMarket]:
    # Get the 5 markets that are closing soonest
    markets: list[AgentMarket] = OmenAgentMarket.get_binary_markets(
        filter_by=FilterBy.OPEN,
        sort_by=SortBy.CLOSING_SOONEST,
        limit=5,
    )
    return cast(List[OmenAgentMarket], markets)


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
        p for p in positions if market_index_set in p.indexSets
    )
    position_as_int = int(position_for_index_set.id.hex(), 16)
    balance = OmenConditionalTokenContract().balanceOf(user_address, position_as_int)
    return balance


def fetch_public_key_from_env() -> ChecksumAddress:
    private_key = os.environ.get("BET_FROM_PRIVATE_KEY")
    if private_key is None:
        raise EnvironmentError("Could not load private key using env var")
    return private_key_to_public_key(SecretStr(private_key))
