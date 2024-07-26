from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ArgMetadata(BaseModel):
    name: str
    type: str


class AbiItemTypeEnum(str, Enum):
    function = "function"
    event = "event"


class AbiItemStateMutabilityEnum(str, Enum):
    VIEW = "view"
    NON_PAYABLE = "nonpayable"
    PAYABLE = "payable"


class ABIMetadata(BaseModel):
    constant: Optional[bool] = None
    inputs: list[ArgMetadata]
    name: str
    outputs: list[ArgMetadata]
    stateMutability: AbiItemStateMutabilityEnum
    type: AbiItemTypeEnum
