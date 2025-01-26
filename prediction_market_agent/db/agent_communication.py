from functools import cache

from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.config import APIKeys as APIKeys_PMAT
from prediction_market_agent_tooling.gtypes import HexBytes, Wei, xDai
from prediction_market_agent_tooling.tools.contract import (
    AgentCommunicationContract,
    ContractOnGnosisChain,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    ENABLE_GET_MESSAGES_BY_HIGHEST_FEE,
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


def pop_message(api_keys: APIKeys_PMAT) -> MessageContainer:
    agent_comm_contract = AgentCommunicationContract()
    if not ENABLE_GET_MESSAGES_BY_HIGHEST_FEE:
        return agent_comm_contract.pop_message(
            api_keys=api_keys,
            agent_address=api_keys.bet_from_address,
        )
    else:
        all_messages = fetch_unseen_transactions(api_keys.bet_from_address)
        index, _ = max(
            [(i, m) for i, m in enumerate(all_messages)], key=lambda m: m[1].value
        )
        return agent_comm_contract.pop_message(
            api_keys=api_keys,
            agent_address=api_keys.bet_from_address,
            index=index,
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
