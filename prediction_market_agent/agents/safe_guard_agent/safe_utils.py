from prediction_market_agent_tooling.config import APIKeys, RPCConfig
from prediction_market_agent_tooling.gtypes import HexBytes
from prediction_market_agent_tooling.loggers import logger
from safe_eth.safe.api.transaction_service_api.transaction_service_api import (
    EthereumNetwork,
    TransactionServiceApi,
)
from safe_eth.safe.safe import Safe, SafeTx
from web3 import Web3


def post_message(safe: Safe, message: str, api_keys: APIKeys) -> None:
    # TODO: Doesn't work atm!
    return
    web3 = Web3(Web3.HTTPProvider(RPCConfig().gnosis_rpc_url))
    message_hash = web3.keccak(text=message)
    signed_message = web3.eth.account.signHash(
        message_hash, private_key=api_keys.bet_from_private_key.get_secret_value()
    )
    signature = signed_message.signature

    api = TransactionServiceApi(EthereumNetwork.GNOSIS)
    api.post_message(
        safe.address,
        message,
        signature=signature,
    )


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
        api = TransactionServiceApi(EthereumNetwork.GNOSIS)
        api.post_signatures(tx.safe_tx_hash, tx.signatures)
    else:
        logger.info("Threshold met, executing the transaction.")
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
        logger.info(
            f"Safe required multiple signers, posting the transaction to the queue."
        )
        api = TransactionServiceApi(EthereumNetwork.GNOSIS)
        api.post_transaction(tx)
    else:
        logger.info("Safe requires only 1 signer, executing the transaction.")
        tx.call()
        tx.execute(api_keys.bet_from_private_key.get_secret_value())
