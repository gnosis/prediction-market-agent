import numpy as np
import pytest
from microchain import Engine
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.market_functions import (
    MARKET_FUNCTIONS,
    BuyNo,
    BuyYes,
    GetBalance,
    GetKellyBet,
    GetMarketProbability,
    GetMarkets,
    PredictProbabilityForQuestion,
    SellNo,
    SellYes,
)
from prediction_market_agent.agents.microchain_agent.memory_functions import (
    CheckAllPastActionsGivenContext,
    LookAtPastActionsFromLastDay,
)
from prediction_market_agent.agents.microchain_agent.utils import (
    get_balance,
    get_binary_markets,
    get_no_outcome,
    get_yes_outcome,
)
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.utils import APIKeys
from tests.utils import RUN_PAID_TESTS


# TODO investigate why this fails for polymarket https://github.com/gnosis/prediction-market-agent/issues/62
@pytest.mark.parametrize("market_type", [MarketType.OMEN, MarketType.MANIFOLD])
def test_get_markets(market_type: MarketType) -> None:
    get_markets = GetMarkets(market_type=market_type, keys=APIKeys())
    assert len(get_markets()) > 0


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_buy_yes(market_type: MarketType) -> None:
    market = get_binary_markets(market_type=market_type)[0]
    buy_yes = BuyYes(market_type=market_type, keys=APIKeys())
    print(buy_yes(market.id, 0.0001))


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_buy_no(market_type: MarketType) -> None:
    market = get_binary_markets(market_type=market_type)[0]
    buy_no = BuyNo(market_type=market_type, keys=APIKeys())
    print(buy_no(market.id, 0.0001))


@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_replicator_has_balance_gt_0(market_type: MarketType) -> None:
    balance = GetBalance(market_type=market_type, keys=APIKeys())()
    assert balance > 0


@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_engine_help(market_type: MarketType) -> None:
    engine = Engine()
    engine.register(Reasoning())
    engine.register(Stop())
    for function in MARKET_FUNCTIONS:
        engine.register(function(market_type=market_type, keys=APIKeys()))

    print(engine.help)


@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_get_probability(market_type: MarketType) -> None:
    market_id = "0x0020d13c89140b47e10db54cbd53852b90bc1391"
    get_market_probability = GetMarketProbability(
        market_type=market_type, keys=APIKeys()
    )
    assert float(get_market_probability(market_id)[0]) == 0.0
    market: AgentMarket = market_type.market_class.get_binary_market(market_id)
    assert market.is_resolved()  # Probability wont change after resolution


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_buy_sell_tokens(market_type: MarketType) -> None:
    """
    Test buying and selling tokens for a market
    """
    keys = APIKeys()
    market = get_binary_markets(market_type=market_type)[0]
    from_address = keys.bet_from_address
    outcomes_functions = {
        get_yes_outcome(market_type=market_type): [
            BuyYes(market_type=market_type, keys=APIKeys()),
            SellYes(market_type=market_type, keys=APIKeys()),
        ],
        get_no_outcome(market_type=market_type): [
            BuyNo(market_type=market_type, keys=APIKeys()),
            SellNo(market_type=market_type, keys=APIKeys()),
        ],
    }

    # Needs to be big enough below for fees to be relatively small enough
    # for checks to pass
    buy_amount = 0.1

    def get_balances() -> tuple[float, float]:
        wallet_balance = get_balance(keys, market_type=market_type)
        token_balance = market.get_token_balance(
            user_id=from_address,
            outcome=outcome,
        )
        return float(wallet_balance), float(token_balance)

    for outcome, functions in outcomes_functions.items():
        buy_tokens, sell_tokens = functions

        before_wallet_balance, before_tokens = get_balances()
        buy_tokens(market.id, buy_amount)
        after_wallet_balance, after_tokens = get_balances()

        # Check that the wallet balance has decreased by the amount bought
        assert np.isclose(
            before_wallet_balance - after_wallet_balance,
            buy_amount,
            rtol=0.01,
        )

        # Sell all the tokens you just bought
        sell_token_amount = after_tokens - before_tokens
        sell_tokens(market.id, sell_token_amount)
        final_wallet_balance, final_tokens = get_balances()

        # Check that the wallet balance has increased by the amount sold
        # Can't sell the exact amount bought due to fees
        new_buy_amount = buy_amount * 0.96
        assert np.isclose(
            final_wallet_balance - after_wallet_balance, new_buy_amount, rtol=0.01
        )

        # Check that the number of tokens bought and sold is approximately equal
        n_tokens_bought = after_tokens - before_tokens
        n_tokens_sold = after_tokens - final_tokens
        assert np.isclose(n_tokens_bought, n_tokens_sold, rtol=0.01)


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_predict_probability(market_type: MarketType) -> None:
    """
    Test calling a mech to predict the probability of a market
    """
    predict_probability = PredictProbabilityForQuestion(
        market_type=market_type, keys=APIKeys()
    )
    market = get_binary_markets(market_type=market_type)[0]
    p_yes = predict_probability(market.id)
    assert 0.0 <= float(p_yes) <= 1.0


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_look_at_past_actions(
    long_term_memory_table_handler: LongTermMemoryTableHandler,
) -> None:
    long_term_memory_table_handler.save_history(
        history=[
            {"role": "user", "content": "I went to the park and saw a dog."},
            {"role": "user", "content": "I went to the park and saw a cat."},
            {"role": "user", "content": "I went to the park and saw a bird."},
        ]
    )
    ## Uncomment below to test with the memories accrued from use of https://autonomous-trader-agent.streamlit.app/
    # long_term_memory = LongTermMemoryTableHandler(task_description="microchain-streamlit-app")
    past_actions = LookAtPastActionsFromLastDay(
        long_term_memory=long_term_memory_table_handler
    )
    print(past_actions())


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_check_past_actions_given_context(
    long_term_memory_table_handler: LongTermMemoryTableHandler,
) -> None:
    long_term_memory_table_handler.save_history(
        history=[
            {
                "role": "user",
                "content": "Agent X sent me a message asking for a coalition.",
            },
            {
                "role": "user",
                "content": "I agreed with agent X to form a coalition, I'll send him my NFT key if he sends me 5 xDai",
            },
            {"role": "user", "content": "I went to the park and saw a bird."},
        ]
    )
    ## Uncomment below to test with the memories accrued from use of https://autonomous-trader-agent.streamlit.app/
    # long_term_memory = LongTermMemoryTableHandler(task_description="microchain-streamlit-app")
    past_actions = CheckAllPastActionsGivenContext(
        long_term_memory=long_term_memory_table_handler
    )
    print(past_actions(context="What coalitions did I form?"))


@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_kelly_bet(market_type: MarketType) -> None:
    market = get_binary_markets(market_type=market_type)[0]
    get_kelly_bet = GetKellyBet(market_type=market_type, keys=APIKeys())
    bet = get_kelly_bet(market_id=market.id, estimated_p_yes=market.current_p_yes)
    assert "Bet size: 0.0" in bet  # No 'edge', so no bet size
