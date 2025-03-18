from safe_eth.safe import SafeTx

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_guard_agent.validation_result import (
    ValidationResult,
)


def validate_safe_transaction_hash(
    new_transaction: DetailedTransactionResponse,
    new_transaction_safetx: SafeTx,
    history: list[DetailedTransactionResponse],
) -> ValidationResult:
    """Function that checks if the safe_tx_hashes match."""

    if not new_transaction.detailedExecutionInfo:
        return ValidationResult(ok=False, reason="Detailed execution info is missing.")

    # Here we compare the safe_tx_hash delivered by the Safe client (see get_safe_detailed_transaction_info) and the
    # safe constructed by safe-eth-py library (see the usage of safe.build_multisig_tx inside safe_tx_from_detailed_transaction).
    # Note that the Safe Transaction Service offers a function (https://github.com/safe-global/safe-eth-py/blob/main/safe_eth/safe/api/transaction_service_api/transaction_service_api.py#L129)
    # that also does a safe_tx_hash assertion. However, we choose to do it like this for more granular control.
    safe_tx_hash_from_new_transaction = new_transaction.detailedExecutionInfo.safeTxHash
    safe_tx_hash_from_built_tx = new_transaction_safetx.safe_tx_hash
    if safe_tx_hash_from_new_transaction != safe_tx_hash_from_built_tx:
        return ValidationResult(ok=False, reason="Safe tx hashes do not match.")

    return ValidationResult(ok=True, reason="Hashes match.")
