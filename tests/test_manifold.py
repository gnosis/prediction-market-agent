from prediction_market_agent import utils
from prediction_market_agent.markets import manifold
from prediction_market_agent.tools.types import mana_type


def test_manifold() -> None:
    market = manifold.pick_binary_market()
    print(market.question)
    print("Placing bet on market:", market.question)
    manifold.place_bet(mana_type(2), market.id, True, utils.get_manifold_api_key())
