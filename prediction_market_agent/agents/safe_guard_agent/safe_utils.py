import tenacity
from eth_account.messages import defunct_hash_message
from prediction_market_agent_tooling.config import APIKeys, RPCConfig
from prediction_market_agent_tooling.gtypes import ChecksumAddress, HexBytes
from prediction_market_agent_tooling.loggers import logger
from safe_eth.eth import EthereumClient, EthereumNetwork
from safe_eth.safe import Safe
from safe_eth.safe.api.transaction_service_api import TransactionServiceApi
from safe_eth.safe.api.transaction_service_api.transaction_service_api import (
    EthereumNetwork,
    TransactionServiceApi,
)
from safe_eth.safe.safe import Safe, SafeTx
from web3 import Web3


def get_safe(safe_address: ChecksumAddress) -> Safe:
    safe = Safe(safe_address, EthereumClient(RPCConfig().gnosis_rpc_url))  # type: ignore
    return safe


def check_if_owner(safe_address: ChecksumAddress, maybe_owner: ChecksumAddress) -> bool:
    safe = get_safe(safe_address)
    return safe.retrieve_is_owner(maybe_owner)


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(1))
def get_safes(owner: ChecksumAddress) -> list[ChecksumAddress]:
    api = TransactionServiceApi(EthereumNetwork(RPCConfig().chain_id))
    safes = api.get_safes_for_owner(owner)
    return safes


def post_message(safe: Safe, message: str, api_keys: APIKeys) -> None:
    logger.info(f"Posting message to Safe {safe.address}.")

    message_hash = defunct_hash_message(text=message)
    safe_message_hash = safe.get_message_hash(message_hash)  # type: ignore # type bug, it's iffed to work correctly inside the function.
    owner_signature = api_keys.get_account().signHash(safe_message_hash)

    api = TransactionServiceApi(network=EthereumNetwork(RPCConfig().chain_id))
    api.post_message(safe.address, message, owner_signature.signature)


def reject_transaction(safe: Safe, tx: SafeTx, api_keys: APIKeys) -> None:
    # To reject transaction on Safe, you create a new transaction with the same nonce, but empty data.
    rejection_tx = SafeTx(
        ethereum_client=tx.ethereum_client,
        safe_address=tx.safe_address,
        to=tx.safe_address,
        value=0,
        data=HexBytes("0x"),
        operation=0,
        safe_tx_gas=0,
        base_gas=0,
        gas_price=0,
        gas_token=Web3.to_checksum_address(
            "0x0000000000000000000000000000000000000000"
        ),
        refund_receiver=Web3.to_checksum_address(
            "0x0000000000000000000000000000000000000000"
        ),
        signatures=None,
        safe_nonce=tx.safe_nonce,
        safe_version=tx.safe_version,
        chain_id=tx.chain_id,
    )
    post_or_execute(safe, rejection_tx, api_keys)


def sign_or_execute(safe: Safe, tx: SafeTx, api_keys: APIKeys) -> None:
    """
    Use this function to sign an existing transaction and automatically either execute it (if threshold is met), or to post your signature into the transaction in the queue.
    """
    tx.sign(api_keys.bet_from_private_key.get_secret_value())

    if safe.retrieve_threshold() > len(tx.signatures):
        logger.info("Threshold not met yet, just adding a sign.")
        api = TransactionServiceApi(EthereumNetwork(RPCConfig().chain_id))
        api.post_signatures(tx.safe_tx_hash, tx.signatures)
    else:
        logger.info("Threshold met, executing.")
        tx.call()
        tx.execute(api_keys.bet_from_private_key.get_secret_value())


def post_or_execute(safe: Safe, tx: SafeTx, api_keys: APIKeys) -> None:
    """
    Use this function to automatically sign and execute your new transaction (in case only 1 signer is required),
    or to post it to the queue (in case more than 1 signer is required).
    """
    if len(tx.signatures):
        raise ValueError(
            "Should be a fresh transaction. See `sign_or_execute` function for signing existing transaction."
        )

    # Sign by our account.
    tx.sign(api_keys.bet_from_private_key.get_secret_value())

    if safe.retrieve_threshold() > 1:
        logger.info(f"Safe requires multiple signers, posting to the queue.")
        api = TransactionServiceApi(EthereumNetwork(RPCConfig().chain_id))
        api.post_transaction(tx)
    else:
        logger.info("Safe requires only 1 signer, executing.")
        tx.call()
        tx.execute(api_keys.bet_from_private_key.get_secret_value())
