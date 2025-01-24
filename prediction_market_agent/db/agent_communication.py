from functools import cache

from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.config import APIKeys as APIKeys_PMAT
from prediction_market_agent_tooling.gtypes import HexBytes, Wei, xDai
from prediction_market_agent_tooling.tools.contract import (
    AgentCommunicationContract,
)
from prediction_market_agent_tooling.tools.data_models import MessageContainer
from web3 import Web3
from web3.types import TxReceipt


def fetch_unseen_transactions(
    consumer_address: ChecksumAddress, n: int | None = None, web3: Web3 | None = None
) -> list[MessageContainer]:
    agent_comm_contract = AgentCommunicationContract()

    count_unseen_messages = fetch_count_unprocessed_transactions(
        consumer_address, web3=web3
    )

    message_containers = [
        agent_comm_contract.get_at_index(
            agent_address=consumer_address, idx=idx, web3=web3
        )
        for idx in range(
            min(n, count_unseen_messages) if n is not None else count_unseen_messages
        )
    ]

    return message_containers


def fetch_count_unprocessed_transactions(
    consumer_address: ChecksumAddress, web3: Web3 | None = None
) -> int:
    agent_comm_contract = AgentCommunicationContract()

    count_unseen_messages = agent_comm_contract.count_unseen_messages(
        consumer_address, web3=web3
    )
    return count_unseen_messages


def pop_message(api_keys: APIKeys_PMAT) -> MessageContainer:
    agent_comm_contract = AgentCommunicationContract()
    popped_message = agent_comm_contract.pop_message(
        api_keys=api_keys,
        agent_address=api_keys.bet_from_address,
    )
    return popped_message


def send_message(
    api_keys: APIKeys_PMAT,
    recipient: ChecksumAddress,
    message: HexBytes,
    amount_wei: Wei,
    web3: Web3 | None = None,
) -> TxReceipt:
    agent_comm_contract = AgentCommunicationContract()
    return agent_comm_contract.send_message(
        api_keys=api_keys,
        agent_address=recipient,
        message=message,
        amount_wei=amount_wei,
        web3=web3,
    )


@cache
def get_message_minimum_value() -> xDai:
    return AgentCommunicationContract().minimum_message_value()


@cache
def get_treasury_tax_ratio() -> float:
    return AgentCommunicationContract().ratio_given_to_treasury()
