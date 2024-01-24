import typing as t
from enum import Enum
from prediction_market_agent.markets import manifold, omen


class MarketType(str, Enum):
    MANIFOLD = "manifold"
    OMEN = "omen"


@t.overload
def get_binary_markets(
    market_type: t.Literal[MarketType.MANIFOLD],
) -> list[manifold.ManifoldMarket]:
    ...


@t.overload
def get_binary_markets(
    market_type: t.Literal[MarketType.OMEN],
) -> list[omen.OmenMarket]:
    ...


@t.overload
def get_binary_markets(
    market_type: MarketType,
) -> t.Union[list[manifold.ManifoldMarket], list[omen.OmenMarket]]:
    ...


def get_binary_markets(
    market_type: MarketType,
) -> t.Union[list[manifold.ManifoldMarket], list[omen.OmenMarket]]:
    if market_type == MarketType.MANIFOLD:
        return manifold.get_manifold_binary_markets(limit=10)
    elif market_type == MarketType.OMEN:
        return omen.get_omen_binary_markets(limit=10)
    else:
        raise ValueError(f"Unknown market type: {market_type}")
