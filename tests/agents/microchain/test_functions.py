from typing import Generator

import numpy as np
import pytest
from langchain.tools import StructuredTool
from microchain import Engine
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.functions_from_tools import (
    microchain_function_from_tool,
)
from prediction_market_agent.agents.microchain_agent.market_functions import (
    BuyNo,
    BuyYes,
    GetBalance,
    MarketFunction,
    PredictProbabilityForQuestionLocal,
    PredictProbabilityForQuestionRemote,
    SellNo,
    SellYes,
    build_market_functions,
)
from prediction_market_agent.agents.microchain_agent.memory import LongTermMemory
from prediction_market_agent.agents.microchain_agent.memory_functions import (
    RememberPastActions,
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


@pytest.fixture(scope="session")
def long_term_memory() -> Generator[LongTermMemory, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    long_term_memory = LongTermMemory(
        task_description="test", sqlalchemy_db_url="sqlite://"
    )
    long_term_memory.storage._initialize_db()
    yield long_term_memory


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
    for function in build_market_functions(market_type=market_type):
        engine.register(function)

    print(engine.help)


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
    buy_amount = 0.1

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


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_remember_past_learnings(long_term_memory: LongTermMemory) -> None:
    long_term_memory.save_history(
        history=[
            {"content": "I went to the park and saw a dog."},
            {"content": "I went to the park and saw a cat."},
            {"content": "I went to the park and saw a bird."},
        ]
    )
    ## Uncomment below to test with the memories accrued from use of https://autonomous-trader-agent.streamlit.app/
    # long_term_memory = LongTermMemory(task_description="microchain-streamlit-app")
    remember_past_learnings = RememberPastActions(
        long_term_memory=long_term_memory,
        model="gpt-4o-2024-05-13",
    )
    print(remember_past_learnings())


def test_build_market_functions() -> None:
    functions = build_market_functions(market_type=MarketType.OMEN)
    assert len(functions) > 0


def test_microchain_function_from_tool() -> None:
    # @tool
    def add(a: int, b: int) -> int:
        """
        Add two numbers
        """
        return a + b

    add_tool = StructuredTool.from_function(
        func=add,
        name="AddTool",
        description="For adding two numbers",
    )
    microchain_add = microchain_function_from_tool(
        tool=add_tool, example_args=["1", "2"]
    )

    # Check microchain function has inherited the expected properties
    assert microchain_add.example_args == ["1", "2"]
    assert microchain_add.description == add_tool.description
    assert microchain_add.name == add_tool.name

    # Run both and check they return the same value
    assert microchain_add(1, 2) == add_tool._run(1, 2)
