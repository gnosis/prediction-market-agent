import time

import tqdm
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.contract import (
    ContractOwnableERC721BaseClass,
)
from web3 import Web3

from prediction_market_agent.tools.anvil.models import (
    ERC721Transfer,
    TransactionDict,
)


def fetch_nft_transfers(
    web3: Web3,
    nft_contract_address: ChecksumAddress,
    from_block: int,
    to_block: int | None = None,
) -> list[ERC721Transfer]:
    s = ContractOwnableERC721BaseClass(address=nft_contract_address)
    nft_c = s.get_web3_contract(web3=web3)

    # fetch transfer events in the last block
    start = time.time()
    logs = nft_c.events.Transfer().get_logs(fromBlock=from_block, toBlock=to_block)  # type: ignore[attr-defined]
    logger.debug(f"elapsed {time.time() - start}")
    logger.debug(f"fetched {len(logs)} NFT transfers")
    events = [ERC721Transfer.from_event_log(log) for log in logs]
    return events


def extract_transactions_involving_agents_and_treasuries(
    web3: Web3,
    from_block: int,
    to_block: int | None = None,
) -> list[TransactionDict]:
    to_block = web3.eth.get_block_number() if to_block is None else to_block
    blocks = list(range(from_block, to_block + 1))  # include end block

    txs = []
    for block in tqdm.tqdm(blocks):
        block = web3.eth.get_block(block, full_transactions=True)
        for tx in block.transactions:
            transaction = TransactionDict.model_validate(tx)
            if transaction.relevant_to_nft_game():
                txs.append(transaction)

    return txs
