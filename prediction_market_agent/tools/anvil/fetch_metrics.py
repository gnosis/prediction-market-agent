import time

from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.contract import (
    AgentCommunicationContract,
    ContractOwnableERC721BaseClass,
)
from prediction_market_agent_tooling.tools.parallelism import par_map
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    TREASURY_ADDRESS,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DEPLOYED_NFT_AGENTS,
)
from prediction_market_agent.tools.anvil.anvil_requests import get_balance
from prediction_market_agent.tools.anvil.models import (
    ERC721Transfer,
    AgentCommunicationMessage,
    BalanceData,
)


def fetch_nft_transfers(
    web3: Web3,
    nft_contract_address: ChecksumAddress,
    from_block: int = 37341108,
    to_block: int | None = None,
) -> list[ERC721Transfer]:
    s = ContractOwnableERC721BaseClass(address=nft_contract_address)
    nft_c = s.get_web3_contract(web3=web3)

    # fetch transfer events in the last block
    start = time.time()
    logs = nft_c.events.Transfer().get_logs(fromBlock=from_block, toBlock=to_block)
    logger.debug(f"elapsed {time.time() - start}")
    logger.debug(f"fetched {len(logs)} NFT transfers")
    events = [ERC721Transfer.from_event_log(log) for log in logs]
    return events


def extract_messages_exchanged(
    web3: Web3,
    from_block: int = 37341108,
    to_block: int | None = None,
) -> list[AgentCommunicationMessage]:
    agent_communication_contract = AgentCommunicationContract()
    agent_communication_c = agent_communication_contract.get_web3_contract(web3=web3)

    start = time.time()
    logs = agent_communication_c.events.LogMessage().get_logs(
        fromBlock=from_block, toBlock=to_block
    )
    logger.debug(f"elapsed {time.time() - start}")
    logger.debug(f"fetched {len(logs)} events from AgentCommunication contract")
    # ToDo - make sure this works
    messages = [AgentCommunicationMessage.from_event_log(log) for log in logs]
    return messages


def extract_balances_per_block(
    rpc_url: str,
    from_block: int = 37341108,
    to_block: int | None = None,
) -> list[BalanceData]:
    blocks = list(range(from_block, to_block + 1))  # include end block

    balance_data: list[BalanceData] = []
    agent_addresses = [a.wallet_address for a in DEPLOYED_NFT_AGENTS]

    for address in agent_addresses + [TREASURY_ADDRESS]:
        balances = par_map(
            items=blocks,
            func=lambda block: get_balance(
                rpc_url=rpc_url, address=address, block=block
            ),
        )
        balance_data.extend(
            [
                BalanceData(address=address, block=block, balance_wei=balance)
                for block, balance in balances
            ]
        )
    return balance_data
