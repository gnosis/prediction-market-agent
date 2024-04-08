import pytest
from dotenv import load_dotenv
from eth_typing import HexStr, HexAddress
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import OmenSubgraphHandler
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.functions import GetWalletBalance, GetUserPositions
from prediction_market_agent.agents.microchain_agent.tools import get_omen_market_token_balance

REPLICATOR_ADDRESS = "0x993DFcE14768e4dE4c366654bE57C21D9ba54748"
AGENT_0_ADDRESS = "0x2DD9f5678484C1F59F97eD334725858b938B4102"


@pytest.fixture(scope="session", autouse=True)
def do_something(request):
    load_dotenv()
    yield


@pytest.fixture()
def get_wallet_balance():
    return GetWalletBalance()


def test_replicator_has_balance_lt_0():
    balance = GetWalletBalance().__call__(REPLICATOR_ADDRESS)
    assert balance > 0


def test_agent_0_has_bet_on_market():
    user_positions = GetUserPositions().__call__(AGENT_0_ADDRESS)
    # Assert 3 conditionIds are included
    expected_condition_ids = [
        HexBytes("0x9c7711bee0902cc8e6838179058726a7ba769cc97d4d0ea47b31370d2d7a117b"),
        HexBytes("0xe2bf80af2a936cdabeef4f511620a2eec46f1caf8e75eb5dc189372367a9154c"),
        HexBytes("0x3f8153364001b26b983dd92191a084de8230f199b5ad0b045e9e1df61089b30d"),
    ]
    unique_condition_ids = sum([u.position.conditionIds for u in user_positions], [])
    assert set(expected_condition_ids).issubset(unique_condition_ids)

def test_balance_for_user_in_market() -> None:
    user_address = '0x2DD9f5678484C1F59F97eD334725858b938B4102'
    subgraph_handler = OmenSubgraphHandler()
    market_id = HexAddress(HexStr('0x59975b067b0716fef6f561e1e30e44f606b08803')) # yes/no
    market = subgraph_handler.get_omen_market(market_id)
    omen_agent_market = OmenAgentMarket.from_data_model(market)
    outcomes = omen_agent_market.outcomes
    balance_yes = get_omen_market_token_balance(user_address=Web3.to_checksum_address(user_address),
                                            market=omen_agent_market,
                                            market_outcome=outcomes[0])
    assert balance_yes == 1959903969410997

    balance_no = get_omen_market_token_balance(user_address=Web3.to_checksum_address(user_address),
                                                market=omen_agent_market,
                                                market_outcome=outcomes[1])
    assert balance_no == 0
