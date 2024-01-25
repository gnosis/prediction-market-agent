import pytest
from prediction_market_agent import utils
from prediction_market_agent.markets import manifold
from prediction_market_agent.tools.gtypes import mana_type
from prediction_market_agent.tools.utils import check_not_none


@pytest.mark.skipif(utils.get_manifold_api_key() is None, reason="No Manifold API key")
def test_manifold() -> None:
    market = manifold.pick_binary_market()
    print(market.question)
    print("Placing bet on market:", market.question)
    manifold.place_bet(
        mana_type(2), market.id, True, check_not_none(utils.get_manifold_api_key())
    )
