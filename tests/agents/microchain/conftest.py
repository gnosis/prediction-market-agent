from typing import Generator

import pytest
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    WrappedxDaiContract,
)
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.blockchain.contract_class_converter import (
    ContractClassConverter,
)


@pytest.fixture(scope="module")
def wxdai_contract_class_converter() -> Generator[ContractClassConverter, None, None]:
    wxdai = WrappedxDaiContract()
    contract_address = Web3.to_checksum_address(wxdai.address)
    yield ContractClassConverter(contract_address)
