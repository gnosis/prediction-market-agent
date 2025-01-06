from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.config import APIKeys as APIKeys_PMAT
from prediction_market_agent_tooling.gtypes import HexBytes, Wei
from prediction_market_agent_tooling.tools.contract import (
    AgentCommunicationContract,
    ContractOnGnosisChain,
)
from prediction_market_agent_tooling.tools.data_models import MessageContainer
from prediction_market_agent_tooling.tools.parallelism import par_map
from web3.types import TxReceipt


def fetch_unseen_transactions(
    consumer_address: ChecksumAddress,
    n: int | None = None,
) -> list[MessageContainer]:
    agent_comm_contract = AgentCommunicationContract()

    count_unseen_messages = fetch_count_unprocessed_transactions(consumer_address)

    message_containers = par_map(
        items=list(range(n or count_unseen_messages)),
        func=lambda idx: agent_comm_contract.get_at_index(
            agent_address=consumer_address, idx=idx
        ),
    )

    return message_containers


def fetch_count_unprocessed_transactions(consumer_address: ChecksumAddress) -> int:
    agent_comm_contract = AgentCommunicationContract()

    count_unseen_messages = agent_comm_contract.count_unseen_messages(consumer_address)
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
) -> TxReceipt:
    agent_comm_contract = AgentCommunicationContract()
    return agent_comm_contract.send_message(
        api_keys=api_keys,
        agent_address=recipient,
        message=message,
        amount_wei=amount_wei,
        web3=ContractOnGnosisChain.get_web3(),
    )
