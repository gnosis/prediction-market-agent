import web3.constants

from prediction_market_agent.agents.microchain_agent.blockchain.contract_class_converter import (
    ContractClassConverter,
)


def test_decimals(wxdai_contract_mocked_rag: ContractClassConverter) -> None:
    function_types_to_classes = (
        wxdai_contract_mocked_rag.create_classes_from_smart_contract()
    )
    classes = sum(function_types_to_classes.values(), [])
    assert len(classes) == 11
    decimals = next((clz for clz in classes if clz.__name__ == "Decimals"), None)
    assert decimals
    result_decimals = decimals().__call__()
    assert result_decimals == 18


def test_balance_of(wxdai_contract_mocked_rag: ContractClassConverter) -> None:
    function_types_to_classes = (
        wxdai_contract_mocked_rag.create_classes_from_smart_contract()
    )
    classes = sum(function_types_to_classes.values(), [])
    balance_of = next((clz for clz in classes if clz.__name__ == "Balanceof"), None)
    assert balance_of
    result_balance_of = balance_of().__call__(web3.constants.ADDRESS_ZERO)
    assert result_balance_of > 0
