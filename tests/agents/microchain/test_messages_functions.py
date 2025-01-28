from typing import Generator
from unittest.mock import PropertyMock, patch

import pytest
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.gtypes import wei_type
from prediction_market_agent_tooling.tools.contract import AgentCommunicationContract
from prediction_market_agent_tooling.tools.data_models import MessageContainer
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from pydantic import SecretStr
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.messages_functions import (
    ReceiveMessage,
)
from prediction_market_agent.db.agent_communication import (
    fetch_count_unprocessed_transactions,
)


@pytest.fixture(scope="session")
def account2_address() -> ChecksumAddress:
    # anvil account # 2
    return Web3.to_checksum_address("0x70997970C51812dc3A010C7d01b50e0d17dc79C8")


@pytest.fixture(scope="session")
def account2_private_key() -> SecretStr:
    "Anvil test account private key. It's public already."
    return SecretStr(
        "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
    )


# Random transactions found on Gnosisscan.
MOCK_HASH_1 = "0x5ba6dd51d3660f98f02683e032daa35644d3f7f975975da3c2628a5b4b1f5cb6"
MOCK_HASH_2 = "0x429f61ea3e1afdd104fdd0a6f3b88432ec4c7b298fd126378e53a63bc60fed6a"
MOCK_SENDER = Web3.to_checksum_address(
    "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
)  # anvil account 1
MOCK_COUNT_UNPROCESSED_TXS = 1


@pytest.fixture(scope="module")
def patch_count_unseen_messages() -> Generator[PropertyMock, None, None]:
    with patch.object(
        AgentCommunicationContract,
        "count_unseen_messages",
        return_value=MOCK_COUNT_UNPROCESSED_TXS,
    ) as mock:
        yield mock


@pytest.fixture
def patch_public_key(
    account2_address: ChecksumAddress, account2_private_key: SecretStr
) -> Generator[PropertyMock, None, None]:
    with patch(
        "prediction_market_agent.agents.microchain_agent.microchain_agent_keys.MicrochainAgentKeys.public_key",
        new_callable=PropertyMock,
    ) as mock_public_key, patch(
        "prediction_market_agent.agents.microchain_agent.microchain_agent_keys.MicrochainAgentKeys.bet_from_private_key",
        new_callable=PropertyMock,
    ) as mock_private_key:
        mock_public_key.return_value = account2_address
        mock_private_key.return_value = account2_private_key
        yield mock_public_key


def test_receive_message_description(
    patch_public_key: PropertyMock,
    patch_count_unseen_messages: PropertyMock,
) -> None:
    r = ReceiveMessage()
    description = r.description
    count_unseen_messages = fetch_count_unprocessed_transactions(
        patch_public_key.return_value
    )
    assert str(count_unseen_messages) in description


def test_receive_message_call(
    patch_public_key: PropertyMock,
    patch_count_unseen_messages: PropertyMock,
) -> None:
    mock_log_message = MessageContainer(
        sender=MOCK_SENDER,
        recipient=MOCK_SENDER,
        message=HexBytes("0x123"),  # dummy message
        value=wei_type(10000),
    )
    with patch.object(
        AgentCommunicationContract,
        "pop_message",
        return_value=mock_log_message,
    ):
        r = ReceiveMessage()

        blockchain_message = r(minimum_fee=0)
        assert blockchain_message is not None
