from datetime import timedelta
from typing import Any

import tenacity
from eth_account import Account
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
from safe_eth.safe.safe_signature import SafeSignature, SafeSignatureContract
from web3 import Web3
from web3.types import TxParams

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)

NULL_ADDRESS = Web3.to_checksum_address("0x0000000000000000000000000000000000000000")


def get_safe(safe_address: ChecksumAddress) -> Safe:
    safe = Safe(safe_address, EthereumClient(RPCConfig().gnosis_rpc_url))  # type: ignore
    return safe


@tenacity.retry(
    stop=tenacity.stop_after_attempt(10),
    wait=tenacity.wait_exponential(max=timedelta(minutes=5)),
    after=lambda x: logger.debug(
        f"get_safes failed, {x.attempt_number=}, {x.upcoming_sleep=}."
    ),
)
def get_safes(owner: ChecksumAddress) -> list[ChecksumAddress]:
    api = TransactionServiceApi(EthereumNetwork(RPCConfig().chain_id))
    safes = api.get_safes_for_owner(owner)
    return safes


def post_message(safe: Safe, message: str, api_keys: APIKeys) -> None:
    logger.info(f"Posting a message to Safe {safe.address}.", streamlit=True)

    message_hash = defunct_hash_message(text=message)
    target_safe_message_hash = safe.get_message_hash(message_hash)  # type: ignore # type bug, it's iffed to work correctly inside the function.

    if api_keys.safe_address_checksum is not None:
        # In the case we are posting message from another Safe.
        # Based on https://github.com/safe-global/safe-eth-py/blob/v6.4.0/safe_eth/safe/tests/test_safe_signature.py#L184.
        owner_safe_message_hash = get_safe(api_keys.safe_address_checksum).get_message_hash(message_hash)  # type: ignore # type bug, it's iffed to work correctly inside the function.
        owner_safe_eoa_signature = api_keys.get_account().signHash(
            owner_safe_message_hash
        )["signature"]

        owner_safe_signature = SafeSignatureContract.from_values(
            api_keys.safe_address_checksum,
            target_safe_message_hash,
            message_hash,
            owner_safe_eoa_signature,
        )
        signature = SafeSignature.export_signatures([owner_safe_signature])
    else:
        # Otherwise normal signature directly using EOA.
        signature = api_keys.get_account().signHash(target_safe_message_hash)[
            "signature"
        ]

    api = TransactionServiceApi(network=EthereumNetwork(RPCConfig().chain_id))
    api.post_message(safe.address, message, signature)


def reject_transaction(safe: Safe, tx: SafeTx, api_keys: APIKeys) -> None:
    # To reject transaction on Safe, you create a new transaction with the same nonce, but empty data.
    logger.info("Creating a rejection transaction.", streamlit=True)
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
        gas_token=NULL_ADDRESS,
        refund_receiver=NULL_ADDRESS,
        signatures=None,
        safe_nonce=tx.safe_nonce,
        safe_version=tx.safe_version,
        chain_id=tx.chain_id,
    )
    post_or_execute(safe, rejection_tx, api_keys)


def sign_or_execute(safe: Safe, tx: SafeTx, api_keys: APIKeys) -> TxParams | None:
    """
    Use this function to sign an existing transaction and automatically either execute it (if threshold is met), or to post your signature into the transaction in the queue.
    """
    if api_keys.safe_address_checksum is not None:
        _safe_sign(
            tx,
            api_keys.safe_address_checksum,
            api_keys.bet_from_private_key.get_secret_value(),
        )
    else:
        tx.sign(api_keys.bet_from_private_key.get_secret_value())

    threshold = safe.retrieve_threshold()

    if threshold > len(tx.signers):
        logger.info("Threshold not met yet, posting a signature.", streamlit=True)
        api = TransactionServiceApi(EthereumNetwork(RPCConfig().chain_id))
        api.post_signatures(tx.safe_tx_hash, tx.signatures)
        return None
    else:
        logger.info(
            f"Threshold {threshold} met with {len(tx.signers)} signs, executing.",
            streamlit=True,
        )
        tx.call()
        _, tx_params = tx.execute(api_keys.bet_from_private_key.get_secret_value())
        return tx_params


def post_or_execute(safe: Safe, tx: SafeTx, api_keys: APIKeys) -> TxParams | None:
    """
    Use this function to automatically sign and execute your new transaction (in case only 1 signer is required),
    or to post it to the queue (in case more than 1 signer is required).
    """
    if api_keys.safe_address_checksum is not None:
        _safe_sign(
            tx,
            api_keys.safe_address_checksum,
            api_keys.bet_from_private_key.get_secret_value(),
        )
    else:
        tx.sign(api_keys.bet_from_private_key.get_secret_value())

    threshold = safe.retrieve_threshold()

    if threshold > len(tx.signers):
        logger.info(f"Safe requires multiple signers, posting to the queue.")
        api = TransactionServiceApi(EthereumNetwork(RPCConfig().chain_id))
        api.post_transaction(tx)
        return None
    else:
        logger.info(
            f"Threshold {threshold} met with {len(tx.signers)} signs, executing."
        )
        tx.call()
        _, tx_params = tx.execute(api_keys.bet_from_private_key.get_secret_value())
        return tx_params


def extract_all_addresses_or_raise(
    tx: DetailedTransactionResponse,
) -> list[ChecksumAddress]:
    """
    Extracts all addresses from the transaction data.
    Useful for example when dealing with Multi-Send transaction, which is built from multiple transactions inside it.
    Raises ValueError if no addresses are found.
    """
    addresses = extract_all_addresses(tx)
    if not addresses:
        raise ValueError("No addresses found in the transaction.")
    return addresses


def extract_all_addresses(tx: DetailedTransactionResponse) -> list[ChecksumAddress]:
    """
    Extracts all addresses from the transaction data.
    Useful for example when dealing with Multi-Send transaction, which is built from multiple transactions inside it.
    """
    # Automatically remove null address as that one isn't interesting.
    found_addresses = find_addresses_in_nested_structure(tx.model_dump()) - {
        NULL_ADDRESS
    }
    return sorted(found_addresses)


def find_addresses_in_nested_structure(value: Any) -> set[ChecksumAddress]:
    addresses = set()
    if isinstance(value, dict):
        for v in value.values():
            addresses.update(find_addresses_in_nested_structure(v))
    elif isinstance(value, list):
        for item in value:
            addresses.update(find_addresses_in_nested_structure(item))
    elif isinstance(value, str):
        try:
            addresses.add(Web3.to_checksum_address(value))
        except ValueError:
            # Ignore if it's not a valid address.
            pass
    # Automatically remove null address as that one isn't interesting.
    return addresses - {NULL_ADDRESS}


def _safe_sign(
    tx: SafeTx, owner_safe_address: ChecksumAddress, private_key: str
) -> bytes:
    """
    TODO: This could be proposed into safe-eth-py as a method of SafeTx.
    Based on https://github.com/safe-global/safe-eth-py/blob/v6.4.0/safe_eth/safe/tests/test_safe_signature.py#L210.

    :param tx:
    :param owner_safe_address:
    :param private_key:
    :return: Signature
    """
    account = Account.from_key(private_key)

    owner_safe_message_hash = get_safe(owner_safe_address).get_message_hash(
        tx.safe_tx_hash_preimage  # type: ignore # type bug, this is correct.
    )
    owner_safe_eoa_signature = account.signHash(owner_safe_message_hash)["signature"]
    owner_safe_signature = SafeSignatureContract.from_values(
        owner_safe_address,
        tx.safe_tx_hash,
        tx.safe_tx_hash_preimage,
        owner_safe_eoa_signature,
    )

    # Insert signature sorted
    if owner_safe_address.lower() not in [x.lower() for x in tx.signers]:
        existing_signatures = SafeSignature.parse_signature(
            tx.signatures,
            tx.safe_tx_hash,
            tx.safe_tx_hash_preimage,
        )
        tx.signatures = SafeSignature.export_signatures(
            [owner_safe_signature, *existing_signatures]
        )

    return tx.signatures
