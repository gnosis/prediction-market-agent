from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.contract import (
    SimpleTreasuryContract,
)
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    TREASURY_ADDRESS,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DEPLOYED_NFT_AGENTS,
)
from prediction_market_agent.tools.anvil.anvil_requests import (
    impersonate_account,
    set_balance,
)


def reset_balances(
    rpc_url: str, new_balance_agents_xdai: int, new_balance_treasury_xdai: int
) -> None:
    for agent in DEPLOYED_NFT_AGENTS:
        set_balance(
            rpc_url=rpc_url,
            address=agent.wallet_address,
            balance=new_balance_agents_xdai,
        )

    set_balance(
        rpc_url=rpc_url,
        address=TREASURY_ADDRESS,
        balance=new_balance_treasury_xdai,
    )


def get_token_owner(token_id: int, web3: Web3) -> ChecksumAddress:
    nft_contract = SimpleTreasuryContract().nft_contract(web3=web3)
    return Web3.to_checksum_address(nft_contract.owner_of(token_id=token_id, web3=web3))


def redistribute_nft_keys(rpc_url: str, count_nft_keys: int = 5) -> None:
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    for token_id in range(count_nft_keys):
        token_owner = get_token_owner(token_id=token_id, web3=w3)
        if token_owner == DEPLOYED_NFT_AGENTS[token_id].wallet_address:
            logger.info(
                f"Token {token_id} already owned by agent {DEPLOYED_NFT_AGENTS[token_id].identifier}"
            )
            continue
        else:
            with impersonate_account(w3, token_owner):
                # We need to build tx ourselves since no private key available from
                # impersonated accounts.
                nft_contract = SimpleTreasuryContract().nft_contract(web3=w3)
                recipient = DEPLOYED_NFT_AGENTS[token_id].wallet_address
                tx_hash = (
                    nft_contract.get_web3_contract(web3=w3)
                    .functions.safeTransferFrom(token_owner, recipient, token_id)
                    .transact({"from": token_owner})
                )
                w3.eth.wait_for_transaction_receipt(transaction_hash=tx_hash)
                logger.info(
                    f"Token {token_id} transferred to agent {DEPLOYED_NFT_AGENTS[token_id].identifier}"
                )
