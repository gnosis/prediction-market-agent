import pytest
from eth_typing import HexAddress, HexStr
from microchain import Engine
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.functions import (
    MARKET_FUNCTIONS,
    MISC_FUNCTIONS,
    BuyNo,
    BuyYes,
    GetMarkets,
    GetUserPositions,
    GetWalletBalance,
)
from prediction_market_agent.agents.microchain_agent.utils import (
    get_binary_markets,
    get_market_token_balance,
)
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
    balance = GetWalletBalance(market_type=market_type)()
    assert balance > 0


@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_agent_0_has_bet_on_market(market_type: MarketType) -> None:
    user_positions = GetUserPositions(market_type=market_type)(AGENT_0_ADDRESS)
    # Assert 3 conditionIds are included
    expected_condition_ids = [
        HexBytes("0x9c7711bee0902cc8e6838179058726a7ba769cc97d4d0ea47b31370d2d7a117b"),
        HexBytes("0xe2bf80af2a936cdabeef4f511620a2eec46f1caf8e75eb5dc189372367a9154c"),
        HexBytes("0x3f8153364001b26b983dd92191a084de8230f199b5ad0b045e9e1df61089b30d"),
    ]
    unique_condition_ids: list[HexBytes] = sum(
        [u.position.conditionIds for u in user_positions], []
    )
    assert set(expected_condition_ids).issubset(unique_condition_ids)


def test_balance_for_user_in_market() -> None:
    user_address = AGENT_0_ADDRESS
    subgraph_handler = OmenSubgraphHandler()
    market_id = HexAddress(
        HexStr("0x59975b067b0716fef6f561e1e30e44f606b08803")
    )  # yes/no
    market = subgraph_handler.get_omen_market(market_id)
    omen_agent_market = OmenAgentMarket.from_data_model(market)
    balance_yes = get_market_token_balance(
        user_address=Web3.to_checksum_address(user_address),
        market_condition_id=omen_agent_market.condition.id,
        market_index_set=market.condition.index_sets[0],
    )

    assert balance_yes == 1959903969410997

    balance_no = get_market_token_balance(
        user_address=Web3.to_checksum_address(user_address),
        market_condition_id=omen_agent_market.condition.id,
        market_index_set=market.condition.index_sets[1],
    )
    assert balance_no == 0


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
