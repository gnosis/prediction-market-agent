import json
from typing import Any

import requests_cache
from eth_typing import ChecksumAddress
from microchain import Function
from prediction_market_agent_tooling.tools.contract import ContractOnGnosisChain
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.blockchain.models import (
    AbiItemTypeEnum,
    ABIMetadata,
)

TYPE_MAPPING: dict[str, (str, Any)] = {
    "address": (
        "str",
        Web3.to_checksum_address("0xe91d153e0b41518a2ce8dd3d7944fa863463a97d"),
    ),  # Solidity address to Python string
    "uint": ("int", 1),  # Solidity unsigned integer to Python int
    "uint8": ("int", 1),  # Solidity unsigned integer to Python int
    "uint256": ("int", 1),  # Solidity unsigned integer (256-bit) to Python int
    "int": ("int", 1),  # Solidity integer to Python int
    "bool": ("bool", True),  # Solidity boolean to Python bool
    "string": ("str", "test string"),
    # Add more mappings as needed
}


class ClassFactory:
    def create_class(self, class_name, base_classes=(), attributes=None):
        if attributes is None:
            attributes = {}

        new_class = type(class_name, base_classes, attributes)
        return new_class


class ContractClassConverter:
    """Class responsible for reading a smart contract on Gnosis Chain and converting its functionalities into Python classes."""

    def __init__(self, contract_address: ChecksumAddress):
        # For caching requests of the same contract
        self.session = requests_cache.CachedSession(backend="sqlite")
        self.contract_address = contract_address

    def get_abi(self) -> list[ABIMetadata]:
        r = self.session.get(
            f"https://gnosis.blockscout.com/api/v2/smart-contracts/{self.contract_address}"
        )
        r.raise_for_status()
        contract_abi = r.json()["abi"]
        # We extract only functions, not events
        return [
            ABIMetadata.parse_obj(abi_item)
            for abi_item in contract_abi
            if abi_item["type"] == "function"
        ]

    def generate_microchain_class_from_abi_item(
        self,
        abi_item: ABIMetadata,
        contract: ContractOnGnosisChain,
    ):
        if abi_item.type != AbiItemTypeEnum.function:
            return None

        input_args = ","
        for input in abi_item.inputs:
            # add type mapping
            input_args += f", {input.name}: {TYPE_MAPPING[input.type][0]}"

        all_input_args = [
            f"{i.name}: {TYPE_MAPPING[i.type][0]}" for i in abi_item.inputs
        ]
        input_args = f"{','.join(all_input_args)}"

        input_as_list = f"{','.join([i.name for i in abi_item.inputs])}"

        # add output
        all_args = [TYPE_MAPPING[i.type][0] for i in abi_item.outputs]
        output_args = f"{','.join(all_args)}"
        if not output_args:
            output_args = "None"

        # ToDo - process case mapping (name == "") - dict
        if abi_item.inputs and abi_item.inputs[0].name == "":
            return
        # ToDo - separate in case payable, nonpayable

        function_code = f"def {abi_item.name}(self, {input_args}) -> {output_args}: return contract.call('{abi_item.name}', [{input_as_list}])"

        print(function_code)
        namespace = {"contract": contract}
        exec(function_code, namespace)
        dynamic_function = namespace[abi_item.name]

        # Example usage
        factory = ClassFactory()

        example_args = [TYPE_MAPPING[i.type][1] for i in abi_item.inputs]
        attributes = {
            "__call__": dynamic_function,
            "description": abi_item.name,
            "example_args": example_args,
        }

        DynamicClass = factory.create_class(
            abi_item.name.title(), (Function,), attributes
        )
        return DynamicClass

    def create_classes_from_smart_contract(self):
        # Get ABI from contract
        abi_items = self.get_abi()
        # For each method, generate 1 class
        classes = []
        abi_str = json.dumps([i.model_dump() for i in abi_items])
        contract = ContractOnGnosisChain(abi=abi_str, address=self.contract_address)
        for abi_item in abi_items:
            generated_class = self.generate_microchain_class_from_abi_item(
                abi_item, contract
            )
            if generated_class:
                classes.append(generated_class)

        return classes
