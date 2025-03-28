from typing import Generator
from unittest.mock import PropertyMock, patch

import pytest
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.gtypes import xDaiWei
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from pydantic import SecretStr
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    AgentCommunicationContract,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.data_models import (
    MessageContainer,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.nft_game_messages_functions import (
    GetUnseenMessagesInformation,
    ReceiveMessagesAndPayments,
    SleepUntil,
)


@pytest.fixture(scope="session")
def account2_address() -> ChecksumAddress:
    # anvil account # 2
    return Web3.to_checksum_address("0x70997970C51812dc3A010C7d01b50e0d17dc79C8")


@pytest.fixture(scope="session")
def account2_private_key() -> SecretStr:
    "Anvil test account private key. It's public already."
    return SecretStr(
        "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"  # web3-private-key-ok
    )


# Random transactions found on Gnosisscan.
MOCK_HASH_1 = "0x5ba6dd51d3660f98f02683e032daa35644d3f7f975975da3c2628a5b4b1f5cb6"  # web3-private-key-ok
MOCK_HASH_2 = "0x429f61ea3e1afdd104fdd0a6f3b88432ec4c7b298fd126378e53a63bc60fed6a"  # web3-private-key-ok
MOCK_SENDER = Web3.to_checksum_address(
    "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
)  # anvil account 1


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


def test_message_statistics(patch_public_key: PropertyMock) -> None:
    with patch(
        "prediction_market_agent.db.agent_communication.fetch_unseen_transactions",
        return_value=[
            MessageContainer(
                sender=MOCK_SENDER,
                recipient=MOCK_SENDER,
                message=HexBytes("0x123"),  # dummy message
                value=xDaiWei(10000),
            ),
            MessageContainer(
                sender=MOCK_SENDER,
                recipient=MOCK_SENDER,
                message=HexBytes("0x123"),  # dummy message
                value=xDaiWei(100000),
            ),
        ],
    ):
        r = GetUnseenMessagesInformation()
        statistics = r("0x1Ca11b2520345993e78312b00441050d2d57065f")
        assert statistics == (
            "Unseen messages statistics for 0x1Ca11b2520345993e78312b00441050d2d57065f:\n"
            "Minimum fee: 1e-14 xDai\n"
            "Maximum fee: 1e-13 xDai\n"
            "Average fee: 5.5e-14 xDai\n"
            "Number of unique senders: 1\n"
            "Total number of messages: 2"
        )


def test_receive_message_call(patch_public_key: PropertyMock) -> None:
    mock_log_message = MessageContainer(
        sender=MOCK_SENDER,
        recipient=MOCK_SENDER,
        message=HexBytes("0x123"),  # dummy message
        value=xDaiWei(10000),
    )
    with patch.object(
        AgentCommunicationContract,
        "pop_message",
        return_value=mock_log_message,
    ), patch(
        "prediction_market_agent.db.agent_communication.fetch_unseen_transactions",
        return_value=[mock_log_message],
    ):
        r = ReceiveMessagesAndPayments()

        blockchain_messages = r(n=2, minimum_fee=0)
        assert blockchain_messages is not None


@pytest.mark.parametrize(
    "call_code",
    [
        "SleepUntil(sleep_until='2025-03-26 19:00:00+00:00', reason='Game over, sold my NFT.')",
        "SleepUntil(sleep_until='2025-03-26 18:00:45.710960+00:00', reason='Sleeping for an hour.')",
        "SleepUntil('2025-03-26 19:00:00+00:00', 'Game over, sold my NFT.')",
    ],
)
def test_sleep_until_parsing(call_code: str) -> None:
    SleepUntil.execute_calling_of_this_function(call_code)
