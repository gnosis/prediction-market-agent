from typing import Any, Type, TypeAlias

from pydantic import BaseModel, model_validator
from typing_extensions import Self
from web3 import Web3


class PythonType(BaseModel):
    type: Type[Any]
    example_value: Any

    @model_validator(mode="after")
    def validate_example_value(self) -> Self:
        if not isinstance(self.example_value, self.type):
            raise ValueError(f"The example_value must be of type {self.type.__name__}")
        return self


SolidityType: TypeAlias = str


TYPE_MAPPING: dict[SolidityType, PythonType] = {
    "address": PythonType(
        type=str,
        example_value=Web3.to_checksum_address(
            "0xe91d153e0b41518a2ce8dd3d7944fa863463a97d"
        ),
    ),
    "uint": PythonType(type=int, example_value=1),
    "uint8": PythonType(type=int, example_value=1),
    "uint256": PythonType(type=int, example_value=1),
    "int": PythonType(type=int, example_value=1),
    "bool": PythonType(type=bool, example_value=True),
    "string": PythonType(type=str, example_value="test string"),
    # Add more mappings as needed
}


def raise_if_unknown_type(solidity_type: str) -> PythonType:
    python_type = TYPE_MAPPING.get(solidity_type, None)
    if python_type is None:
        raise ValueError(f"Unknown solidity type: {solidity_type}")
    return python_type


def get_python_type_from_solidity_type(solidity_type: str) -> str:
    python_type = raise_if_unknown_type(solidity_type)
    return python_type.type.__name__


def get_example_args_from_solidity_type(solidity_type: str) -> Any:
    python_type = raise_if_unknown_type(solidity_type)
    return python_type.example_value
