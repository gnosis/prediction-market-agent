import json
from typing import Any

import requests_cache
from eth_typing import ChecksumAddress
from microchain import Function
from prediction_market_agent_tooling.tools.contract import ContractOnGnosisChain
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.blockchain.code_interpreter import (
    CodeInterpreter,
)
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

    def fetch_from_blockscout(self) -> dict[str, Any]:
        r = self.session.get(
            f"https://gnosis.blockscout.com/api/v2/smart-contracts/{self.contract_address}"
        )
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        return data

    def get_abi(self) -> list[ABIMetadata]:
        data = self.fetch_from_blockscout()
        contract_abi = data["abi"]
        # We extract only functions, not events
        return [
            ABIMetadata.parse_obj(abi_item)
            for abi_item in contract_abi
            if abi_item["type"] == "function"
        ]

    def get_source_code(self) -> str:
        data = self.fetch_from_blockscout()
        source_code = data["source_code"]
        return str(source_code)

    def generate_microchain_class_from_abi_item(
        self,
        abi_item: ABIMetadata,
        contract: ContractOnGnosisChain,
        code_interpreter: CodeInterpreter,
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

        namespace = {"contract": contract}
        exec(function_code, namespace)
        dynamic_function = namespace[abi_item.name]

        # Microchain specific attributes
        example_args = [TYPE_MAPPING[i.type][1] for i in abi_item.inputs]
        # ToDo - Call generate_summary once instead of per function.
        summary = code_interpreter.generate_summary(function_name=abi_item.name)

        attributes = {
            "__call__": dynamic_function,
            "description": summary,
            "example_args": example_args,
        }

        dynamic_class = ClassFactory().create_class(
            abi_item.name.title(), (Function,), attributes
        )
        return dynamic_class

    def create_classes_from_smart_contract(self):
        # Get ABI from contract
        abi_items = self.get_abi()
        source_code = self.get_source_code()
        # For each method, generate 1 class
        classes = []
        abi_str = json.dumps([i.model_dump() for i in abi_items])
        contract = ContractOnGnosisChain(abi=abi_str, address=self.contract_address)
        code_interpreter = CodeInterpreter(source_code=source_code)
        for abi_item in abi_items:
            generated_class = self.generate_microchain_class_from_abi_item(
                abi_item, contract, code_interpreter
            )
            if generated_class:
                classes.append(generated_class)

        return classes
