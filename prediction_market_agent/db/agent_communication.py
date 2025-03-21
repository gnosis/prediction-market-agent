from functools import cache

from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.config import APIKeys as APIKeys_PMAT
from prediction_market_agent_tooling.gtypes import HexBytes, xDai, xDaiWei
from pydantic import BaseModel
from web3 import Web3
from web3.types import TxReceipt

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    AgentCommunicationContract,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.data_models import (
    MessageContainer,
)


class MessagesStatistics(BaseModel):
    min_fee: xDai | None
    max_fee: xDai | None
    avg_fee: xDai | None
    n_unique_senders: int
    n_messages: int


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


def get_unseen_messages_statistics(
    consumer_address: ChecksumAddress,
) -> MessagesStatistics:
    messages = fetch_unseen_transactions(consumer_address)

    min_fee = min(messages, key=lambda m: m.value).value if messages else None
    max_fee = max(messages, key=lambda m: m.value).value if messages else None
    avg_fee = (
        sum((m.value for m in messages), start=xDaiWei(0)) / len(messages)
        if messages
        else None
    )
    n_unique_senders = len(set(m.sender for m in messages))

    return MessagesStatistics(
        min_fee=min_fee.as_xdai if min_fee is not None else None,
        max_fee=max_fee.as_xdai if max_fee is not None else None,
        avg_fee=avg_fee.as_xdai if avg_fee is not None else None,
        n_unique_senders=n_unique_senders,
        n_messages=len(messages),
    )


def pop_message(minimum_fee: xDai, api_keys: APIKeys_PMAT) -> MessageContainer | None:
    agent_comm_contract = AgentCommunicationContract()
    all_messages = fetch_unseen_transactions(api_keys.bet_from_address)
    filtered_indices_and_messages = [
        (i, m) for i, m in enumerate(all_messages) if m.value.as_xdai >= minimum_fee
    ]
    return (
        agent_comm_contract.pop_message(
            api_keys=api_keys,
            index=filtered_indices_and_messages[0][0],
        )
        if filtered_indices_and_messages
        else None
    )


def send_message(
    api_keys: APIKeys_PMAT,
    recipient: ChecksumAddress,
    message: HexBytes,
    amount_wei: xDaiWei,
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
