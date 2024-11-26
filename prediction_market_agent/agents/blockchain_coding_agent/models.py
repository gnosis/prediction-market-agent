import typing as t

from pydantic import BaseModel


class SmartContractImplementation(BaseModel):
    address: str
    name: str


class SmartContractResponse(BaseModel):
    abi: list[dict[str, t.Any]]
    source_code: str
    implementations: list[SmartContractImplementation]


class SourceCodeContainer(BaseModel):
    abi: list[dict[str, t.Any]]
    source_code: str
