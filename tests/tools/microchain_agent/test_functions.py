import pytest
from dotenv import load_dotenv

from prediction_market_agent.agents.microchain_agent.functions import GetWalletBalance


@pytest.fixture()
def get_wallet_balance():
    return GetWalletBalance()

def test_replicator_has_balance_lt_0(get_wallet_balance: GetWalletBalance):
    load_dotenv()
    user_address = '0x993DFcE14768e4dE4c366654bE57C21D9ba54748'
    balance = get_wallet_balance.__call__(user_address)
    assert balance > 0
