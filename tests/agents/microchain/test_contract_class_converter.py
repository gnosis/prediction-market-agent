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
    class_name = wxdai_contract_mocked_rag.build_class_name("decimals")
    decimals = next((clz for clz in classes if clz.__name__ == class_name), None)
    assert decimals
    result_decimals = decimals().__call__()
    assert result_decimals == 18


def test_balance_of(wxdai_contract_mocked_rag: ContractClassConverter) -> None:
    function_types_to_classes = (
        wxdai_contract_mocked_rag.create_classes_from_smart_contract()
    )
    classes = sum(function_types_to_classes.values(), [])
    class_name = wxdai_contract_mocked_rag.build_class_name("balanceOf")
    balance_of = next((clz for clz in classes if clz.__name__ == class_name), None)
    assert balance_of
    result_balance_of = balance_of().__call__(web3.constants.ADDRESS_ZERO)
    assert result_balance_of > 0


def test_sdai(sdai_contract_mocked_rag: ContractClassConverter) -> None:
    classes = sdai_contract_mocked_rag.create_classes_from_smart_contract()
    assert classes
