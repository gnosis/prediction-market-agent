from prediction_market_agent import utils
from prediction_market_agent.markets import manifold
from prediction_market_agent.tools.gtypes import mana_type


def test_manifold() -> None:
    market = manifold.pick_binary_market()
    print("Placing bet on market:", market.question)
    manifold.place_bet(mana_type(0.01), market.id, True, utils.get_keys().manifold)
