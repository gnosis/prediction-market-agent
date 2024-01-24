import typing as t
from enum import Enum
from prediction_market_agent.markets import manifold, omen
from prediction_market_agent.tools.utils import should_not_happen, check_not_none
from prediction_market_agent.utils import APIKeys


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


def place_bet(
    market: t.Union[omen.OmenMarket, manifold.ManifoldMarket],
    outcome: bool,
    keys: APIKeys,
    omen_auto_deposit: bool,
    amount_mana: t.Optional[manifold.Mana] = None,
    amount_xdai: t.Optional[omen.xDai] = None,
) -> None:
    manifold.place_bet(
        amount=check_not_none(amount_mana),
        market_id=market.id,
        outcome=outcome,
        api_key=check_not_none(keys.manifold),
    ) if isinstance(
        market, manifold.ManifoldMarket
    ) else omen.binary_omen_buy_outcome_tx(
        amount=check_not_none(amount_xdai),
        from_address=check_not_none(keys.bet_from_address),
        from_private_key=check_not_none(keys.bet_from_private_key),
        market=market,
        binary_outcome=outcome,
        auto_deposit=omen_auto_deposit,
    ) if isinstance(
        market, omen.OmenMarket
    ) else should_not_happen(
        f"Unknown market {market}."
    )
