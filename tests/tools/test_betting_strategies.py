import pytest
import numpy as np
from prediction_market_agent_tooling.markets.markets import omen
from prediction_market_agent_tooling.gtypes import (
    Probability,
    xdai_type,
    wei_type,
    usd_type,
    HexAddress,
    OmenOutcomeToken,
    HexStr,
    Wei,
    xDai,
)
from prediction_market_agent.tools.betting_strategies import (
    get_kelly_criterion_bet,
    get_market_moving_bet,
)

TEST_OMEN_MARKET = omen.OmenMarket(
    id=HexAddress(HexStr("0x76a7a3487f85390dc568f3fce01e0a649cb39651")),
    title="Will Plex launch a store for movies and TV shows by 26 January 2024?",
    collateralVolume=Wei(4369016776639073062),
    usdVolume=usd_type("4.369023756584789670441178585394842"),
    collateralToken=HexAddress(HexStr("0xe91d153e0b41518a2ce8dd3d7944fa863463a97d")),
    outcomes=["Yes", "No"],
    outcomeTokenAmounts=[
        OmenOutcomeToken(7277347438897016099),
        OmenOutcomeToken(13741270543921756242),
    ],
    outcomeTokenMarginalPrices=[
        xdai_type("0.6537666061181695741160552853310822"),
        xdai_type("0.3462333938818304258839447146689178"),
    ],
    fee=wei_type(20000000000000000),
)


@pytest.mark.parametrize(
    "wanted_p_yes_on_the_market, expected_buying_xdai_amount, expected_buying_outcome",
    [
        (Probability(0.1), xdai_type(25.32), "No"),
        (Probability(0.9), xdai_type(18.1), "Yes"),
    ],
)
def test_get_market_moving_bet_0(
    wanted_p_yes_on_the_market: Probability,
    expected_buying_xdai_amount: xDai,
    expected_buying_outcome: str,
) -> None:
    xdai_amount, outcome_index = get_market_moving_bet(
        market=TEST_OMEN_MARKET,
        target_p_yes=wanted_p_yes_on_the_market,
        verbose=True,
    )
    assert np.isclose(
        float(xdai_amount),
        float(expected_buying_xdai_amount),
        atol=2.0,  # We don't expect it to be 100% accurate, but close enough.
    ), f"To move this martket to ~{wanted_p_yes_on_the_market}% for yes, the amount should be {expected_buying_xdai_amount}xDai, according to aiomen website."
    assert outcome_index == TEST_OMEN_MARKET.get_outcome_index(
        expected_buying_outcome
    ), f"The buying outcome index should `{expected_buying_outcome}`."


@pytest.mark.parametrize(
    "est_p_yes, expected_outcome",
    [
        (Probability(0.1), "No"),
        (Probability(0.9), "Yes"),
    ],
)
def test_kelly_criterion_bet_0(est_p_yes: Probability, expected_outcome: str) -> None:
    xdai_amount, outcome_index = get_kelly_criterion_bet(
        market=TEST_OMEN_MARKET,
        estimated_p_yes=est_p_yes,
        max_bet=xdai_type(10),  # This significantly changes the outcome.
    )
    # Kelly estimates the best bet for maximizing the expected value of the logarithm of the wealth.
    # We don't know the real best xdai_amount, but at least we know which outcome index makes sense.
    assert outcome_index == TEST_OMEN_MARKET.get_outcome_index(expected_outcome)
