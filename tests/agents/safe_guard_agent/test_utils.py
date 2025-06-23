import pytest
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    SDAI_CONTRACT_ADDRESS,
    WRAPPED_XDAI_CONTRACT_ADDRESS,
)
from web3 import Web3

from prediction_market_agent.agents.safe_watch_agent.utils import (
    ChecksumAddress,
    is_erc20_contract,
    is_nft_contract,
)


@pytest.mark.parametrize(
    "addr, is_erc20",
    [
        (WRAPPED_XDAI_CONTRACT_ADDRESS, True),
        (SDAI_CONTRACT_ADDRESS, True),
        (Web3.to_checksum_address("0x0D7C0Bd4169D090038c6F41CFd066958fe7619D0"), False),
    ],
)
def test_is_erc20_contract(addr: ChecksumAddress, is_erc20: bool) -> None:
    assert is_erc20_contract(addr) == is_erc20


@pytest.mark.parametrize(
    "addr, is_nft",
    [
        (WRAPPED_XDAI_CONTRACT_ADDRESS, False),
        (SDAI_CONTRACT_ADDRESS, False),
        (Web3.to_checksum_address("0x0D7C0Bd4169D090038c6F41CFd066958fe7619D0"), True),
    ],
)
def test_is_nft_contract(addr: ChecksumAddress, is_nft: bool) -> None:
    assert is_nft_contract(addr) == is_nft
