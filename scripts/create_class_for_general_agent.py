import json
from enum import Enum

import requests
from eth_typing import ChecksumAddress
from microchain import Function
from prediction_market_agent_tooling.tools.contract import ContractOnGnosisChain
from pydantic import BaseModel


class BaseClass:
    def base_method(self):
        return "This is a method from the BaseClass"


class ClassFactory:
    def create_class(self, class_name, base_classes=(), attributes=None):
        if attributes is None:
            attributes = {}

        # Dynamically create the new class using the type function
        new_class = type(class_name, base_classes, attributes)
        return new_class


class ArgMetadata(BaseModel):
    name: str
    type: str


class AbiItemTypeEnum(str, Enum):
    function = "function"
    event = "event"


class AbiItemStateMutabilityEnum(str, Enum):
    view = "view"
    nonpayable = "nonpayable"
    payable = "payable"


class ABIMetadata(BaseModel):
    constant: bool
    inputs: list[ArgMetadata]
    name: str
    outputs: list[ArgMetadata]
    stateMutability: AbiItemStateMutabilityEnum
    type: AbiItemTypeEnum


def get_abi(contract_address: ChecksumAddress) -> list[ABIMetadata]:
    r = requests.get(
        f"https://gnosis.blockscout.com/api/v2/smart-contracts/{contract_address}"
    )
    r.raise_for_status()
    contract_abi = r.json()["abi"]
    # We extract only functions, not events
    return [
        ABIMetadata.parse_obj(abi_item)
        for abi_item in contract_abi
        if abi_item["type"] == "function"
    ]


type_mapping: dict[str, str] = {
    "address": "str",  # Solidity address to Python string
    "uint": "int",  # Solidity unsigned integer to Python int
    "uint8": "int",  # Solidity unsigned integer to Python int
    "uint256": "int",  # Solidity unsigned integer (256-bit) to Python int
    "int": "int",  # Solidity integer to Python int
    "bool": "bool",  # Solidity boolean to Python bool
    "string": "str"
    # Add more mappings as needed
}


def generate_microchain_class_from_abi_item(
    abi_item: ABIMetadata,
    contract: ContractOnGnosisChain,
):
    if abi_item.type != AbiItemTypeEnum.function:
        return None

    input_args = ","
    for input in abi_item.inputs:
        # add type mapping
        input_args += f", {input.name}: {type_mapping[input.type]}"

    all_input_args = [f"{i.name}: {type_mapping[i.type]}" for i in abi_item.inputs]
    input_args = f"{','.join(all_input_args)}"

    input_as_list = f"{','.join([i.name for i in abi_item.inputs])}"

    # add output
    all_args = [type_mapping[i.type] for i in abi_item.outputs]
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
    attributes = {
        "run": dynamic_function,
        "description": abi_item.name,
        "example_args": [],
    }

    DynamicClass = factory.create_class(abi_item.name.title(), (Function,), attributes)
    return DynamicClass


def create_classes_from_smart_contract(contract_address: ChecksumAddress):
    pass
    # ToDo
    # Get ABI from contract
    abi_items = get_abi(contract_address)
    # For each method, generate 1 class
    classes = []
    abi_str = json.dumps([i.model_dump() for i in abi_items])
    contract = ContractOnGnosisChain(abi=abi_str, address=contract_address)
    for abi_item in abi_items:
        generated_class = generate_microchain_class_from_abi_item(abi_item, contract)
        if generated_class:
            classes.append(generated_class)

    return classes
    # ToDo - Write test where engine register functions and calls name of wxDAI
