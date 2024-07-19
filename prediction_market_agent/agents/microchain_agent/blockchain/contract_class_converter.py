import json
from collections import defaultdict
from typing import Any, Tuple

import requests
from eth_typing import ChecksumAddress
from loguru import logger
from microchain import Function
from prediction_market_agent_tooling.gtypes import ABI
from prediction_market_agent_tooling.tools.contract import ContractOnGnosisChain
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.blockchain.code_interpreter import (
    CodeInterpreter,
    FunctionSummary,
    Summaries,
)
from prediction_market_agent.agents.microchain_agent.blockchain.models import (
    AbiItemStateMutabilityEnum,
    AbiItemTypeEnum,
    ABIMetadata,
)
from prediction_market_agent.utils import APIKeys

TYPE_MAPPING: dict[str, Tuple[str, Any]] = {
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


class FunctionWithKeys(Function):
    def __init__(self, keys: APIKeys) -> None:
        self.keys = keys
        super().__init__()


class ClassFactory:
    @staticmethod
    def create_class(
        class_name: str,
        base_classes: Tuple[type],
        attributes: dict[str, Any] | None = None,
    ) -> type:
        if attributes is None:
            attributes = {}

        new_class = type(class_name, base_classes, attributes)
        return new_class


class ContractClassConverter:
    """Class responsible for reading a smart contract on Gnosis Chain and converting its functionalities into Python classes."""

    def __init__(self, contract_address: ChecksumAddress):
        # For caching requests of the same contract
        self.contract_address = contract_address

    def fetch_from_blockscout(self) -> dict[str, Any]:
        r = requests.get(
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
        summaries: Summaries,
    ) -> Tuple[AbiItemStateMutabilityEnum | None, type | None]:
        if abi_item.type != AbiItemTypeEnum.function:
            return None, None

        # If type mapping fails, we log and fail gracefully. Note that structs as input- or output args are not supported.
        for input in abi_item.inputs:
            if not TYPE_MAPPING.get(input.type, None):
                logger.info(f"Type mapping has failed. Check inputs {abi_item.inputs}")

        for output in abi_item.outputs:
            if not TYPE_MAPPING.get(output.type, None):
                logger.info(
                    f"Type mapping has failed. Check outputs {abi_item.outputs}"
                )

        input_to_types = {}
        for idx, input in enumerate(abi_item.inputs):
            input_name = input.name if input.name else f"idx_{idx}"
            input_to_types[input_name] = TYPE_MAPPING[input.type][0]

        all_input_args = [
            f"{input_name}: {v}" for input_name, v in input_to_types.items()
        ]
        input_args = f"{','.join(all_input_args)}"
        input_as_list = ",".join(input_to_types.keys())

        # add output
        all_args = [TYPE_MAPPING[i.type][0] for i in abi_item.outputs]
        output_args = f"{','.join(all_args)}"
        if not output_args:
            output_args = "None"

        namespace = {"contract": contract}

        base = Function
        if abi_item.stateMutability == AbiItemStateMutabilityEnum.VIEW:
            function_code = f"def {abi_item.name}(self, {input_args}) -> {output_args}: return contract.call('{abi_item.name}', [{input_as_list}])"

        elif abi_item.stateMutability in [
            AbiItemStateMutabilityEnum.PAYABLE,
            AbiItemStateMutabilityEnum.NON_PAYABLE,
        ]:
            function_code = f"def {abi_item.name}(self, {input_args}) -> {output_args}: return contract.send(self.keys,'{abi_item.name}', [{input_as_list}])"
            base = FunctionWithKeys

        exec(function_code, namespace)
        dynamic_function = namespace[abi_item.name]

        # Microchain specific attributes
        example_args = [TYPE_MAPPING[i.type][1] for i in abi_item.inputs]

        summary = next(
            (s for s in summaries.summaries if s.function_name == abi_item.name),
            FunctionSummary(function_name="", summary=""),
        )

        attributes = {
            "__call__": dynamic_function,
            "description": summary.summary,
            "example_args": example_args,
        }

        dynamic_class = ClassFactory().create_class(
            abi_item.name.title(), (base,), attributes
        )
        return abi_item.stateMutability, dynamic_class

    def create_classes_from_smart_contract(
        self,
    ) -> defaultdict[AbiItemStateMutabilityEnum | None, list[type]]:
        # Get ABI from contract
        abi_items = self.get_abi()
        source_code = self.get_source_code()
        abi_str = json.dumps([i.model_dump() for i in abi_items])
        contract = ContractOnGnosisChain(
            abi=ABI(abi_str), address=self.contract_address
        )
        code_interpreter = CodeInterpreter(source_code=source_code)
        summaries = code_interpreter.generate_summary(
            function_names=[i.name for i in abi_items]
        )
        function_types_to_classes = defaultdict(list)
        for abi_item in abi_items:
            class_type, generated_class = self.generate_microchain_class_from_abi_item(
                abi_item, contract, summaries
            )
            if generated_class:
                function_types_to_classes[class_type].append(generated_class)

        return function_types_to_classes
