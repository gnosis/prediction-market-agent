from unittest.mock import patch

import pytest
from prediction_market_agent_tooling.chains import GNOSIS_CHAIN_ID
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import (
    ChecksumAddress,
    HexBytes,
    private_key_type,
)
from web3 import Web3

from prediction_market_agent.agents.safe_watch_agent.safe_utils import (
    find_addresses_in_nested_structure,
    get_safe,
    get_safes,
    post_message,
)

TESTING_CHAIN_ID = GNOSIS_CHAIN_ID
TESTING_OWNER_EOA = Web3.to_checksum_address(
    "0xB3896D2751Fe1229B49d74820FA9205e54550d3B"
)
TESTING_OWNER_SAFE = Web3.to_checksum_address(
    "0x464f1A446bCF278A2d4E04b7eC1889b4d16D8f3f"
)
TESTING_SAFE_ADDRESS = Web3.to_checksum_address(
    "0xeb95D75e178e6521ff18dFEf4927893D18D1C4E9"
)
DUMMY_PRIVATE_KEY = private_key_type("0x" + "1" * 64)  # web3-private-key-ok


def test_get_safe() -> None:
    safe = get_safe(TESTING_SAFE_ADDRESS, TESTING_CHAIN_ID)
    assert safe.address == TESTING_SAFE_ADDRESS


def test_get_safes() -> None:
    safes = get_safes(Web3.to_checksum_address(TESTING_OWNER_SAFE), TESTING_CHAIN_ID)
    assert len(safes) > 0
    assert safes[0] == TESTING_SAFE_ADDRESS


@pytest.mark.parametrize(
    "owner_safe_address, expected_signature",
    [
        (
            None,
            HexBytes(
                "0x32b03003edd77d63ea36e8937ca96cf16d265804b039a419b3b0ee536958f0de266c8fca3cb3391c8bb38e2026e70aa6eeb0f67557d22685b39bb631dd952a671c"
            ),
        ),
        (
            TESTING_OWNER_SAFE,
            HexBytes(
                "0x000000000000000000000000464f1a446bcf278a2d4e04b7ec1889b4d16d8f3f000000000000000000000000000000000000000000000000000000000000004100000000000000000000000000000000000000000000000000000000000000004144274701e0491a1068f85d85b19f0791639eb8195ed8d6edaea74962437efbdf03e0c5f3d6642a7a33458272d3a58c77fe9ca96c6b0917475a4603c85adc155b1b"
            ),
        ),
    ],
)
def test_post_message(
    owner_safe_address: ChecksumAddress, expected_signature: HexBytes
) -> None:
    """
    Due to mock later on, this test covers only process of correctly signing the message.
    """
    safe = get_safe(TESTING_SAFE_ADDRESS, TESTING_CHAIN_ID)
    message = "Test message"
    api_keys = APIKeys(
        BET_FROM_PRIVATE_KEY=DUMMY_PRIVATE_KEY,
        SAFE_ADDRESS=owner_safe_address,
    )
    # Patch so we don't actually call the API and spam the messages.
    with patch(
        "prediction_market_agent.agents.safe_watch_agent.safe_utils.TransactionServiceApi.post_message"
    ) as mock_post_message:
        post_message(safe, message, api_keys, TESTING_CHAIN_ID)
        mock_post_message.assert_called_once_with(
            safe.address, message, expected_signature
        )


def test_find_addresses_in_nested_structure() -> None:
    # Test with a nested dictionary containing valid and invalid Ethereum addresses
    nested = [
        {
            "level1": {
                "level2": {
                    "valid_address": "0x5Aeda56215b167893e80B4fE645BA6d5Bab767DE",
                    "invalid_address": "0xInvalidAddress",
                    "level3": [
                        "0xInvalidShouldNotBeFound",
                        DUMMY_PRIVATE_KEY,
                        DUMMY_PRIVATE_KEY.get_secret_value(),
                    ],
                },
                "another_valid_address": "0xAb8483F64d9C6d1EcF9b849Ae677dD3315835Cb2",
            },
            "level1_list": [
                "0x0000000000000000000000000000000000000000",  # Null address, should be ignored
                "0x4B0897b0513fdc7C541B6d9D7E929C4e5364D2dB",
            ],
        }
    ]
    expected_addresses = {
        Web3.to_checksum_address("0x5Aeda56215b167893e80B4fE645BA6d5Bab767DE"),
        Web3.to_checksum_address("0xAb8483F64d9C6d1EcF9b849Ae677dD3315835Cb2"),
        Web3.to_checksum_address("0x4B0897b0513fdc7C541B6d9D7E929C4e5364D2dB"),
    }
    found_addresses = find_addresses_in_nested_structure(nested)
    assert found_addresses == expected_addresses
