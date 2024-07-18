from web3 import Web3

from create_class_for_general_agent import create_classes_from_smart_contract

if __name__ == "__main__":
    contract_address = Web3.to_checksum_address(
        "0xe91d153e0b41518a2ce8dd3d7944fa863463a97d"
    )
    classes = create_classes_from_smart_contract(contract_address)
    print(classes)
