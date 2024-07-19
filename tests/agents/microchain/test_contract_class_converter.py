import web3.constants

from prediction_market_agent.agents.microchain_agent.blockchain.contract_class_converter import (
    ContractClassConverter,
)


def test_decimals(wxdai_contract_mocked_rag: ContractClassConverter) -> None:
    classes = wxdai_contract_mocked_rag.create_classes_from_smart_contract()
    # ToDo - Register with engine on separate test
    # only views
    assert len(classes) == 11
    decimals = next((clz for clz in classes if clz.__name__ == "Decimals"), None)

    result_decimals = decimals().__call__()
    assert result_decimals == 18

    balance_of = next((clz for clz in classes if clz.__name__ == "Balanceof"), None)
    result_balance_of = balance_of().__call__(web3.constants.ADDRESS_ZERO)
    assert result_balance_of > 0
