import requests
import tenacity
from prediction_market_agent_tooling.config import RPCConfig
from prediction_market_agent_tooling.gtypes import ChecksumAddress, HexBytes
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import check_not_none
from pydantic import ValidationError
from safe_eth.safe.safe import NULL_ADDRESS, Safe, SafeTx

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.balances import (
    Balances,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_models.transactions import (
    CreationTxInfo,
    CustomTxInfo,
    ModuleExecutionInfo,
    MultisigExecutionInfo,
    Transaction,
    TransactionResponse,
    TransactionResult,
)


def is_valued_transaction_result(
    tx: TransactionResult, all_txs: list[TransactionResult]
) -> bool:
    """
    Filter out creation transactions (nothing to validate there) and transactions that have been already cancelled.
    """
    cancelled_nonces = [
        tx.transaction.executionInfo.nonce
        for tx in all_txs
        if tx.transaction is not None
        and tx.transaction.executionInfo is not None
        and isinstance(tx.transaction.executionInfo, MultisigExecutionInfo)
        and isinstance(tx.transaction.txInfo, CustomTxInfo)
        and tx.transaction.txInfo.isCancellation
    ]
    return (
        tx.type == "TRANSACTION"
        and tx.transaction is not None
        and not isinstance(
            tx.transaction.txInfo,
            CreationTxInfo,
        )
        and (
            tx.transaction.executionInfo is None
            or isinstance(tx.transaction.executionInfo, ModuleExecutionInfo)
            or (
                isinstance(tx.transaction.executionInfo, MultisigExecutionInfo)
                and tx.transaction.executionInfo.nonce not in cancelled_nonces
            )
        )
    )


@observe()
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(1),
    retry=tenacity.retry_if_not_exception_type(ValidationError),
)
def get_safe_queued_transactions(
    safe_address: ChecksumAddress,
) -> list[Transaction]:
    """
    TODO: This isn't great as we would need to call Safe's API for each guarded Safe non-stop.
    Can we somehow listen to creation of queued transactions? Are they emited as events or something? And ideally without relying on Safe's APIs? Project Zero maybe?
    """
    response = requests.get(
        f"https://safe-client.safe.global/v1/chains/{RPCConfig().chain_id}/safes/{safe_address}/transactions/queued"
    ).json()
    response_parsed = TransactionResponse.model_validate(response)
    transactions = [
        check_not_none(item.transaction)
        for item in response_parsed.results
        if is_valued_transaction_result(item, response_parsed.results)
    ]
    return transactions


@observe()
def gather_safe_detailed_transaction_info(
    transaction_ids: list[str],
) -> list[DetailedTransactionResponse]:
    return [
        get_safe_detailed_transaction_info(transaction_id)
        for transaction_id in transaction_ids
    ]


@observe()
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(1),
    retry=tenacity.retry_if_not_exception_type(ValidationError),
)
def get_safe_detailed_transaction_info(
    transaction_id: str,
) -> DetailedTransactionResponse:
    """
    TODO: Can we retrieve this without relying on Safe's APIs?
    """
    response = requests.get(
        f"https://safe-client.safe.global/v1/chains/{RPCConfig().chain_id}/transactions/{transaction_id}"
    ).json()
    response_parsed = DetailedTransactionResponse.model_validate(response)
    return response_parsed


@observe()
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(1),
    retry=tenacity.retry_if_not_exception_type(ValidationError),
)
def get_safe_history(
    safe_address: ChecksumAddress,
) -> list[Transaction]:
    """
    TODO: Can we get this without relying on Safe's APIs?
    """
    response = requests.get(
        f"https://safe-client.safe.global/v1/chains/{RPCConfig().chain_id}/safes/{safe_address}/transactions/history"
    ).json()
    response_parsed = TransactionResponse.model_validate(response)
    transactions = [
        check_not_none(item.transaction)
        for item in response_parsed.results
        if is_valued_transaction_result(item, response_parsed.results)
    ]
    return transactions


def safe_tx_from_detailed_transaction(
    safe: Safe,
    transaction_details: DetailedTransactionResponse,
) -> SafeTx:
    tx_data = check_not_none(transaction_details.txData)
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
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(1),
    retry=tenacity.retry_if_not_exception_type(ValidationError),
)
def get_balances_usd(safe_address: ChecksumAddress) -> Balances:
    """
    TODO: Can we get this without relying on Safe's APIs?
    """
    response = requests.get(
        f"https://safe-client.safe.global/v1/chains/{RPCConfig().chain_id}/safes/{safe_address}/balances/usd?trusted=true"
    ).json()
    response_model = Balances.model_validate(response)
    return response_model
