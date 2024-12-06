import pytest

from prediction_market_agent.agents.microchain_agent.blockchain.code_interpreter import (
    CodeInterpreter,
)
from prediction_market_agent.agents.microchain_agent.blockchain.contract_class_converter import (
    ContractClassConverter,
)
from tests.utils import RUN_PAID_TESTS


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_generate_summaries(
    wxdai_contract_class_converter: ContractClassConverter,
) -> None:
    abi_items = wxdai_contract_class_converter.get_abi()
    source_code = wxdai_contract_class_converter.get_source_code()

    code_interpreter = CodeInterpreter(source_code=source_code)
    summaries = code_interpreter.generate_summary(
        function_names=[i.name for i in abi_items]
    )
    assert len(summaries.summaries) == len(abi_items)
