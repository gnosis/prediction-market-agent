import numpy as np
import pytest
from microchain import Engine
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.functions import (
    MARKET_FUNCTIONS,
    MISC_FUNCTIONS,
    BuyNo,
    BuyYes,
    GetBalance,
    GetMarketProbability,
    GetMarkets,
    MarketFunction,
    PredictProbabilityForQuestionLocal,
    PredictProbabilityForQuestionRemote,
    SellNo,
    SellYes,
)
from prediction_market_agent.agents.microchain_agent.utils import (
    get_balance,
    get_binary_markets,
    get_no_outcome,
    get_yes_outcome,
)
from prediction_market_agent.utils import APIKeys
from tests.utils import RUN_PAID_TESTS

REPLICATOR_ADDRESS = "0x993DFcE14768e4dE4c366654bE57C21D9ba54748"
AGENT_0_ADDRESS = "0x2DD9f5678484C1F59F97eD334725858b938B4102"


# TODO investigate why this fails for polymarket https://github.com/gnosis/prediction-market-agent/issues/62
@pytest.mark.parametrize("market_type", [MarketType.OMEN, MarketType.MANIFOLD])
def test_get_markets(market_type: MarketType) -> None:
    get_markets = GetMarkets(market_type=market_type)
    assert len(get_markets()) > 0


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_buy_yes(market_type: MarketType) -> None:
    market = get_binary_markets(market_type=market_type)[0]
    buy_yes = BuyYes(market_type=market_type)
    print(buy_yes(market.question, 0.0001))


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_buy_no(market_type: MarketType) -> None:
    market = get_binary_markets(market_type=market_type)[0]
    buy_yes = BuyNo(market_type=market_type)
    print(buy_yes(market.question, 0.0001))


@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_replicator_has_balance_gt_0(market_type: MarketType) -> None:
    balance = GetBalance(market_type=market_type)()
    assert balance > 0


@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_engine_help(market_type: MarketType) -> None:
    engine = Engine()
    engine.register(Reasoning())
    engine.register(Stop())
    for function in MISC_FUNCTIONS:
        engine.register(function())
    for function in MARKET_FUNCTIONS:
        engine.register(function(market_type=market_type))

    print(engine.help)


@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_get_probability(market_type: MarketType) -> None:
    market_id = "0x0020d13c89140b47e10db54cbd53852b90bc1391"
    get_market_probability = GetMarketProbability(market_type=market_type)
    assert float(get_market_probability(market_id)[0]) == 0.0
    market: AgentMarket = market_type.market_class.get_binary_market(market_id)
    assert market.is_resolved()  # Probability wont change after resolution


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_buy_sell_tokens(market_type: MarketType) -> None:
    """
    Test buying and selling tokens for a market
    """
    market = get_binary_markets(market_type=market_type)[0]
    from_address = APIKeys().bet_from_address
    outcomes_functions = {
        get_yes_outcome(market_type=market_type): [
            BuyYes(market_type=market_type),
            SellYes(market_type=market_type),
        ],
        get_no_outcome(market_type=market_type): [
            BuyNo(market_type=market_type),
            SellNo(market_type=market_type),
        ],
    }

    # Needs to be big enough below for fees to be relatively small enough
    # for checks to pass
    buy_sell_amount = 0.1

    def get_balances() -> tuple[float, float]:
        wallet_balance = get_balance(market_type=market_type).amount
        token_balance = market.get_token_balance(
            user_id=from_address,
            outcome=outcome,
        ).amount
        return float(wallet_balance), float(token_balance)

    for outcome, functions in outcomes_functions.items():
        buy_tokens, sell_tokens = functions

        before_wallet_balance, before_tokens = get_balances()
        buy_tokens(market.id, buy_sell_amount)
        after_wallet_balance, after_tokens = get_balances()

        # Check that the wallet balance has decreased by the amount bought
        assert np.isclose(
            before_wallet_balance - after_wallet_balance,
            buy_sell_amount,
            rtol=0.01,
        )

        # Can't sell the exact amount bought due to fees
        buy_sell_amount *= 0.96
        sell_tokens(market.id, buy_sell_amount)
        final_wallet_balance, final_tokens = get_balances()

        # Check that the wallet balance has increased by the amount sold
        assert np.isclose(
            final_wallet_balance - after_wallet_balance,
            buy_sell_amount,
            rtol=0.01,
        )

        # Check that the number of tokens bought and sold is approximately equal
        n_tokens_bought = after_tokens - before_tokens
        n_tokens_sold = after_tokens - final_tokens
        assert np.isclose(n_tokens_bought, n_tokens_sold, rtol=0.02)


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
@pytest.mark.parametrize(
    "prediction_method",
    [
        PredictProbabilityForQuestionRemote,
        PredictProbabilityForQuestionLocal,
    ],
)
@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_predict_probability(
    market_type: MarketType, prediction_method: MarketFunction
) -> None:
    """
    Test calling a mech to predict the probability of a market
    """
    predict_probability = prediction_method(market_type=market_type)
    market = get_binary_markets(market_type=market_type)[0]
    p_yes = predict_probability(market.id)
    assert 0.0 <= float(p_yes) <= 1.0
