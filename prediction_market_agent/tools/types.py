from decimal import Decimal
from typing import NewType
from web3.types import Wei
from eth_typing.evm import (
    Address,
    HexAddress,
    ChecksumAddress,
)  # noqa: F401  # Import for the sake of easy importing with others from here.

Wad = Wei  # Wei tends to be referred to as `wad` variable in contracts.
USD = NewType(
    "USD", Decimal
)  # Decimals are more precise than float, good for finances.
PrivateKey = NewType("PrivateKey", str)
xDai = NewType("xDai", Decimal)
GNO = NewType("GNO", Decimal)
ABI = NewType("Abi", str)
