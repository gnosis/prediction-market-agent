from prediction_market_agent.agents.microchain_agent.blockchain.contract_class_converter import (
    ContractClassConverter,
)


def test_decimals(wxdai_contract_class_converter: ContractClassConverter) -> None:
    classes = wxdai_contract_class_converter.create_classes_from_smart_contract()
    # ToDo - Register with engine on separate test
    # only views
    assert len(classes) == 9
    decimals = next((clz for clz in classes if clz.__name__ == "Decimals"), None)

    result_decimals = decimals().__call__()
    assert result_decimals == 18
    # ToDo - Test totalSupply > 7e18
