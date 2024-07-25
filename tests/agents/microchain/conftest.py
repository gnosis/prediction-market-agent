from typing import Generator
from unittest.mock import Mock, patch

import pytest
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    WrappedxDaiContract,
)
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.blockchain.code_interpreter import (
    FunctionSummary,
    Summaries,
)
from prediction_market_agent.agents.microchain_agent.blockchain.contract_class_converter import (
    ContractClassConverter,
)


def mock_summaries(function_names: list[str]) -> Summaries:
    return Summaries(
        summaries=[FunctionSummary(function_name=i, summary=i) for i in function_names]
    )


@pytest.fixture(scope="module")
def wxdai_contract_class_converter() -> Generator[ContractClassConverter, None, None]:
    wxdai = WrappedxDaiContract()
    contract_address = Web3.to_checksum_address(wxdai.address)
    yield ContractClassConverter(
        contract_address=contract_address, contract_name=wxdai.__class__.__name__
    )


class PatcherManager:
    """Class for patching R"""

    def __init__(self) -> None:
        self.patchers = [
            patch(
                "prediction_market_agent.agents.microchain_agent.blockchain.code_interpreter.CodeInterpreter.generate_summary",
                Mock(side_effect=mock_summaries),
            ),
            patch(
                "prediction_market_agent.agents.microchain_agent.blockchain.code_interpreter.CodeInterpreter.build_chain",
                Mock(return_value=None),
            ),
        ]

    def start(self) -> None:
        for i in self.patchers:
            i.start()

    def stop(self) -> None:
        for i in self.patchers:
            i.stop()


@pytest.fixture(scope="module")
def patcher_manager() -> Generator[PatcherManager, None, None]:
    pm = PatcherManager()
    pm.start()
    yield pm
    pm.stop()


@pytest.fixture(scope="module")
def sdai_contract_mocked_rag(
    patcher_manager: PatcherManager,
) -> Generator[ContractClassConverter, None, None]:
    contract_address = Web3.to_checksum_address(
        "0xaf204776c7245bF4147c2612BF6e5972Ee483701"
    )
    c = ContractClassConverter(contract_address=contract_address, contract_name="sDAI")
    yield c


@pytest.fixture(scope="module")
def wxdai_contract_mocked_rag(
    patcher_manager: PatcherManager,
) -> Generator[ContractClassConverter, None, None]:
    wxdai = WrappedxDaiContract()
    contract_address = Web3.to_checksum_address(wxdai.address)
    yield ContractClassConverter(
        contract_address=contract_address, contract_name=wxdai.__class__.__name__
    )
