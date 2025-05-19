import requests
import tenacity
from prediction_market_agent_tooling.gtypes import ChainID, ChecksumAddress, HexBytes
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import check_not_none
from pydantic import ValidationError
from safe_eth.safe.safe import NULL_ADDRESS, Safe, SafeTx
from web3 import Web3

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.balances import (
    Balances,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_models.transactions import (
    CustomTxInfo,
    MultisigExecutionInfo,
    Transaction,
    TransactionResponse,
    TransactionWithMultiSig,
)


def signer_is_missing(
    signer: ChecksumAddress,
    tx: TransactionWithMultiSig,
) -> bool | None:
    """
    Check if the given signer is still missing from the given transaction.
    """
    missing_signers = (
        [Web3.to_checksum_address(x.value) for x in tx.executionInfo.missingSigners]
        if tx.executionInfo.missingSigners is not None
        else None
    )
    return signer in missing_signers if missing_signers is not None else None


def is_already_canceled(
    tx: TransactionWithMultiSig,
    *,
    all_queued_txs: list[TransactionWithMultiSig],
) -> bool:
    """
    Check if the given transaction has already been canceled (given the all other transactions for the Safe).
    """
    cancelled_nonces = [
        tx.executionInfo.nonce
        for tx in all_queued_txs
        if isinstance(tx.txInfo, CustomTxInfo) and tx.txInfo.isCancellation
    ]
    return tx.executionInfo.nonce in cancelled_nonces


def maybe_multisig_tx(
    tx: Transaction,
) -> TransactionWithMultiSig | None:
    return (
        TransactionWithMultiSig.from_tx(tx)
        if tx.executionInfo is not None
        and isinstance(tx.executionInfo, MultisigExecutionInfo)
        else None
    )


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(max=60),
    retry=tenacity.retry_if_not_exception_type(ValidationError),
)
def get_safe_queue(
    safe_address: ChecksumAddress,
    chain_id: ChainID,
) -> list[Transaction]:
    """
    TODO: This isn't great as we would need to call Safe's API for each guarded Safe non-stop.
    Can we somehow listen to creation of queued transactions? Are they emited as events or something? And ideally without relying on Safe's APIs? Project Zero maybe?
    """
    response = requests.get(
        f"https://safe-client.safe.global/v1/chains/{chain_id}/safes/{safe_address}/transactions/queued"
    )
    response.raise_for_status()
    response_parsed = TransactionResponse.model_validate(response.json())
    transactions = [
        r.transaction for r in response_parsed.results if r.transaction is not None
    ]
    # This is standard as API returns it, sorting explicitly just to be sure.
    # The first transaction is the oldest one, as they need to be processed in order of nonces ASC.
    transactions.sort(key=lambda x: x.timestamp, reverse=False)
    return transactions


def get_safe_queue_multisig(
    safe_address: ChecksumAddress,
    chain_id: ChainID,
) -> list[TransactionWithMultiSig]:
    transactions = get_safe_queue(safe_address, chain_id=chain_id)
    multisig_transactions = [
        multisig_tx
        for item in transactions
        if (multisig_tx := maybe_multisig_tx(item)) is not None
    ]
    return multisig_transactions


@observe()
def gather_safe_detailed_transaction_info(
    transaction_ids: list[str],
    chain_id: ChainID,
) -> list[DetailedTransactionResponse]:
    return [
        get_safe_detailed_transaction_info(transaction_id, chain_id=chain_id)
        for transaction_id in transaction_ids
    ]


@observe()
@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(max=60),
    retry=tenacity.retry_if_not_exception_type(ValidationError),
)
def get_safe_detailed_transaction_info(
    transaction_id: str,
    chain_id: ChainID,
) -> DetailedTransactionResponse:
    """
    TODO: Can we retrieve this without relying on Safe's APIs?
    """
    response = requests.get(
        f"https://safe-client.safe.global/v1/chains/{chain_id}/transactions/{transaction_id}"
    )
    response.raise_for_status()
    response_parsed = DetailedTransactionResponse.model_validate(response.json())
    return response_parsed


@observe()
@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(max=60),
    retry=tenacity.retry_if_not_exception_type(ValidationError),
)
def get_safe_history(
    safe_address: ChecksumAddress,
    chain_id: ChainID,
) -> list[Transaction]:
    """
    TODO: Can we get this without relying on Safe's APIs?
    """
    response = requests.get(
        f"https://safe-client.safe.global/v1/chains/{chain_id}/safes/{safe_address}/transactions/history"
    )
    response.raise_for_status()
    response_parsed = TransactionResponse.model_validate(response.json())
    transactions = [
        r.transaction for r in response_parsed.results if r.transaction is not None
    ]
    # This is standard as API returns it, sorting explicitly just to be sure.
    # The first transaction is the newest one, as it's more practical to see latest transactions than always the oldest one.
    transactions.sort(key=lambda x: x.timestamp, reverse=True)
    return transactions


def get_safe_history_multisig(
    safe_address: ChecksumAddress,
    chain_id: ChainID,
) -> list[TransactionWithMultiSig]:
    """
    TODO: Can we get this without relying on Safe's APIs?
    """
    transactions = get_safe_history(safe_address, chain_id=chain_id)
    multisig_transactions = [
        multisig_tx
        for item in transactions
        if (multisig_tx := maybe_multisig_tx(item)) is not None
    ]
    return multisig_transactions


def safe_tx_from_detailed_transaction(
    safe: Safe,
    transaction_details: DetailedTransactionResponse,
) -> SafeTx:
    tx_data = check_not_none(
        transaction_details.txData,
        f"txData is None for {safe.address=}, {transaction_details=}",
    )
    exec_info = transaction_details.detailedExecutionInfo
    return safe.build_multisig_tx(
        to=tx_data.to.value,
        value=int(tx_data.value),
        data=HexBytes(tx_data.hexData or "0x"),
        operation=tx_data.operation,
        safe_tx_gas=(
            int(exec_info.safeTxGas) if exec_info and exec_info.safeTxGas else 0
        ),
        base_gas=int(exec_info.baseGas) if exec_info and exec_info.baseGas else 0,
        gas_price=int(exec_info.gasPrice) if exec_info and exec_info.gasPrice else 0,
        gas_token=(
            exec_info.gasToken if exec_info and exec_info.gasToken else NULL_ADDRESS
        ),
        refund_receiver=(
            exec_info.refundReceiver.value
            if exec_info and exec_info.refundReceiver
            else NULL_ADDRESS
        ),
        signatures=(
            b"".join(
                [
                    HexBytes(confirmation.signature)
                    for confirmation in exec_info.confirmations
                ]
            )
            if exec_info and exec_info.confirmations
            else b""
        ),
        safe_nonce=exec_info.nonce if exec_info else None,
    )


@observe()
@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(max=60),
    retry=tenacity.retry_if_not_exception_type(ValidationError),
)
def get_balances_usd(safe_address: ChecksumAddress, chain_id: ChainID) -> Balances:
    """
    TODO: Can we get this without relying on Safe's APIs?
    """
    response = requests.get(
        f"https://safe-client.safe.global/v1/chains/{chain_id}/safes/{safe_address}/balances/usd?trusted=true"
    )
    response.raise_for_status()
    response_model = Balances.model_validate(response.json())
    return response_model
