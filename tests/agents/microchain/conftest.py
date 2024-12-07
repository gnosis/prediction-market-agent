from typing import Generator
from unittest.mock import Mock, patch

import pytest
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    WrappedxDaiContract,
)
from pydantic import SecretStr
from pytest_postgresql.executor import PostgreSQLExecutor
from pytest_postgresql.janitor import DatabaseJanitor
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.blockchain.code_interpreter import (
    FunctionSummary,
    Summaries,
)
from prediction_market_agent.agents.microchain_agent.blockchain.contract_class_converter import (
    ContractClassConverter,
)
from prediction_market_agent.utils import DBKeys


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


@pytest.fixture(scope="session")
def session_keys_with_postgresql_proc_and_enabled_cache(
    postgresql_proc: PostgreSQLExecutor,
) -> Generator[DBKeys, None, None]:
    with DatabaseJanitor(
        user=postgresql_proc.user,
        host=postgresql_proc.host,
        port=postgresql_proc.port,
        dbname=postgresql_proc.dbname,
        version=postgresql_proc.version,
    ):
        sqlalchemy_db_url = f"postgresql+psycopg2://{postgresql_proc.user}:@{postgresql_proc.host}:{postgresql_proc.port}/{postgresql_proc.dbname}"
        yield DBKeys(SQLALCHEMY_DB_URL=SecretStr(sqlalchemy_db_url))
