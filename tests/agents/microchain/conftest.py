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
def wxdai_contract_mocked_rag() -> Generator[ContractClassConverter, None, None]:
    with patch(
        "prediction_market_agent.agents.microchain_agent.blockchain.code_interpreter.CodeInterpreter.generate_summary",
        Mock(side_effect=mock_summaries),
    ), patch(
        "prediction_market_agent.agents.microchain_agent.blockchain.code_interpreter.CodeInterpreter.build_retriever",
        Mock(return_value=None),
    ), patch(
        "prediction_market_agent.agents.microchain_agent.blockchain.code_interpreter.CodeInterpreter.build_rag_chain",
        Mock(return_value=None),
    ):
        wxdai = WrappedxDaiContract()
        contract_address = Web3.to_checksum_address(wxdai.address)
        c = ContractClassConverter(contract_address, wxdai.__class__.__name__)
        yield c


@pytest.fixture(scope="module")
def wxdai_contract_class_converter() -> Generator[ContractClassConverter, None, None]:
    wxdai = WrappedxDaiContract()
    contract_address = Web3.to_checksum_address(wxdai.address)
    yield ContractClassConverter(contract_address, wxdai.__class__.__name__)
