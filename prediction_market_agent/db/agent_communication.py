from functools import cache

from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.config import APIKeys as APIKeys_PMAT
from prediction_market_agent_tooling.gtypes import HexBytes, Wei, wei_type, xDai
from prediction_market_agent_tooling.tools.contract import (
    AgentCommunicationContract,
    ContractOnGnosisChain,
)
from prediction_market_agent_tooling.tools.data_models import MessageContainer
from prediction_market_agent_tooling.tools.parallelism import par_map
from prediction_market_agent_tooling.tools.web3_utils import wei_to_xdai
from pydantic import BaseModel
from web3.types import TxReceipt


class MessagesStatistics(BaseModel):
    min_fee: xDai | None
    max_fee: xDai | None
    avg_fee: xDai | None
    n_unique_senders: int
    n_messages: int


def fetch_unseen_transactions(
    consumer_address: ChecksumAddress,
    n: int | None = None,
) -> list[MessageContainer]:
    agent_comm_contract = AgentCommunicationContract()

    count_unseen_messages = fetch_count_unprocessed_transactions(consumer_address)

    message_containers = par_map(
        items=list(
            range(
                min(n, count_unseen_messages)
                if n is not None
                else count_unseen_messages
            )
        ),
        func=lambda idx: agent_comm_contract.get_at_index(
            agent_address=consumer_address, idx=idx
        ),
    )

    return message_containers


def fetch_count_unprocessed_transactions(consumer_address: ChecksumAddress) -> int:
    agent_comm_contract = AgentCommunicationContract()

    count_unseen_messages = agent_comm_contract.count_unseen_messages(consumer_address)
    return count_unseen_messages


def get_unseen_messages_statistics(
    consumer_address: ChecksumAddress,
) -> MessagesStatistics:
    messages = fetch_unseen_transactions(consumer_address)

    min_fee = wei_type(min(messages, key=lambda m: m.value).value) if messages else None
    max_fee = wei_type(max(messages, key=lambda m: m.value).value) if messages else None
    avg_fee = (
        wei_type(sum(m.value for m in messages) / len(messages)) if messages else None
    )
    n_unique_senders = len(set(m.sender for m in messages))

    return MessagesStatistics(
        min_fee=wei_to_xdai(min_fee) if min_fee is not None else None,
        max_fee=wei_to_xdai(max_fee) if max_fee is not None else None,
        avg_fee=wei_to_xdai(avg_fee) if avg_fee is not None else None,
        n_unique_senders=n_unique_senders,
        n_messages=len(messages),
    )


def pop_message(minimum_fee: xDai, api_keys: APIKeys_PMAT) -> MessageContainer | None:
    agent_comm_contract = AgentCommunicationContract()
    all_messages = fetch_unseen_transactions(api_keys.bet_from_address)
    filtered_indices = [i for i, m in enumerate(all_messages) if m.value >= minimum_fee]
    return (
        agent_comm_contract.pop_message(
            api_keys=api_keys,
            agent_address=api_keys.bet_from_address,
            index=filtered_indices[0],
        )
        if filtered_indices
        else None
    )


def send_message(
    api_keys: APIKeys_PMAT,
    recipient: ChecksumAddress,
    message: HexBytes,
    amount_wei: Wei,
) -> TxReceipt:
    agent_comm_contract = AgentCommunicationContract()
    return agent_comm_contract.send_message(
        api_keys=api_keys,
        agent_address=recipient,
        message=message,
        amount_wei=amount_wei,
        web3=ContractOnGnosisChain.get_web3(),
    )


@cache
def get_message_minimum_value() -> xDai:
    return AgentCommunicationContract().minimum_message_value()


@cache
def get_treasury_tax_ratio() -> float:
    return AgentCommunicationContract().ratio_given_to_treasury()
