from prediction_market_agent import utils
from prediction_market_agent import manifold


def test_manifold():
    market = manifold.pick_binary_market()
    print(market.question)
    print("Placing bet on market:", market.question)
    manifold.place_bet(2, market.id, True, utils.get_manifold_api_key())
