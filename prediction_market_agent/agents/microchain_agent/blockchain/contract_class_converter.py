import json
from collections import defaultdict
from typing import Any, Tuple

import requests
from eth_typing import ChecksumAddress
from loguru import logger
from microchain import Function
from prediction_market_agent_tooling.gtypes import ABI
from prediction_market_agent_tooling.tools.contract import ContractOnGnosisChain

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
from prediction_market_agent.agents.microchain_agent.blockchain.type_mapping import (
    TYPE_MAPPING,
    get_example_args_from_solidity_type,
    get_python_type_from_solidity_type,
)
from prediction_market_agent.utils import APIKeys


class FunctionWithKeys(Function):
    def __init__(self, keys: APIKeys) -> None:
        self.keys = keys
        super().__init__()


class ClassFactory:
    @staticmethod
    def create_class(
        class_name: str,
        base_classes: Tuple[type, ...],
        attributes: dict[str, Any] | None = None,
    ) -> type:
        if attributes is None:
            attributes = {}

        new_class = type(class_name, base_classes, attributes)
        return new_class


class ContractClassConverter:
    """Class responsible for reading a smart contract on Gnosis Chain and converting its functionalities into Python
    classes."""

    def __init__(self, contract_address: ChecksumAddress, contract_name: str):
        self.contract_address = contract_address
        self.contract_name = contract_name

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
            ABIMetadata.model_validate(abi_item)
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
    ) -> Tuple[AbiItemStateMutabilityEnum | None, type[Function] | None]:
        if abi_item.type != AbiItemTypeEnum.function:
            return None, None

        # If type mapping fails, we exit. Note that structs as input- or output args are not supported.
        for input in abi_item.inputs:
            if not TYPE_MAPPING.get(input.type, None):
                logger.warning(
                    f"Type mapping for {abi_item.name} has failed. Check inputs {abi_item.inputs}"
                )
                return None, None

        for output in abi_item.outputs:
            if not TYPE_MAPPING.get(output.type, None):
                logger.warning(
                    f"Type mapping for {abi_item.name} has failed. Check outputs {abi_item.outputs}"
                )
                return None, None

        input_to_types = {}
        for idx, input in enumerate(abi_item.inputs):
            input_name = input.name if input.name else f"idx_{idx}"
            input_to_types[input_name] = get_python_type_from_solidity_type(input.type)

        all_input_args = [
            f"{input_name}: {v}" for input_name, v in input_to_types.items()
        ]
        input_args = f"{','.join(all_input_args)}"
        input_as_list = ",".join(input_to_types.keys())

        # add output
        all_args = [
            get_python_type_from_solidity_type(i.type) for i in abi_item.outputs
        ]
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

        # We fix "from" as argument since it's a reserved keyword in Python
        if "from" in function_code:
            function_code = function_code.replace("from", "sender")
        exec(function_code, namespace)
        dynamic_function = namespace[abi_item.name]

        # Microchain specific attributes
        example_args = [
            get_example_args_from_solidity_type(i.type) for i in abi_item.inputs
        ]

        summary = next(
            (s for s in summaries.summaries if s.function_name == abi_item.name),
            FunctionSummary(function_name="", summary=""),
        )

        class_name = self.build_class_name(abi_item.name)
        attributes = {
            "__name__": class_name,
            "__call__": dynamic_function,
            "description": summary.summary,
            "example_args": example_args,
        }

        dynamic_class = ClassFactory().create_class(class_name, (base,), attributes)
        return abi_item.stateMutability, dynamic_class

    def build_class_name(self, abi_item_name: str) -> str:
        return f"{self.contract_name.title()}_{abi_item_name.title()}"

    def create_classes_from_smart_contract(
        self,
    ) -> defaultdict[AbiItemStateMutabilityEnum | None, list[type[Function]]]:
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
