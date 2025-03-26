from typing import Callable

from prediction_market_agent_tooling.config import APIKeys, RPCConfig
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from safe_eth.safe.api.transaction_service_api.transaction_service_api import (
    EthereumNetwork,
    TransactionServiceApi,
)
from safe_eth.safe.safe import Safe, SafeTx

from prediction_market_agent.agents.safe_guard_agent import safe_api_utils
from prediction_market_agent.agents.safe_guard_agent.guards import (
    blacklist,
    goplus_,
    llm,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_models.transactions import (
    Transaction,
)
from prediction_market_agent.agents.safe_guard_agent.safe_utils import (
    get_safe,
    post_message,
    reject_transaction,
    sign_or_execute,
)
from prediction_market_agent.agents.safe_guard_agent.validation_result import (
    ValidationResult,
)

SAFE_GUARDS: list[
    Callable[
        [DetailedTransactionResponse, SafeTx, list[DetailedTransactionResponse]],
        ValidationResult,
    ]
] = [
    # Keep ordered from cheapest/fastest to most expensive/slowest.
    blacklist.validate_safe_transaction_blacklist,
    # hash_checker.validate_safe_transaction_hash,
    llm.validate_safe_transaction_llm,
    goplus_.validate_safe_transaction_goplus_address_security,
    goplus_.validate_safe_transaction_goplus_token_security,
    goplus_.validate_safe_transaction_goplus_nft_security,
]


def validate_all(
    do_sign_or_execution: bool,
    do_reject: bool,
    do_message: bool,
) -> None:
    api = TransactionServiceApi(EthereumNetwork(RPCConfig().chain_id))
    api_keys = APIKeys()

    safes_to_verify = api.get_safes_for_owner(api_keys.bet_from_address)
    logger.info(
        f"For owner {api_keys.bet_from_address}, retrieved {safes_to_verify} safes to verify transactions for."
    )

    for safe_address in safes_to_verify:
        validate_safe(
            safe_address,
            do_sign_or_execution,
            do_reject,
            do_message,
        )


def validate_safe(
    safe_address: ChecksumAddress,
    do_sign_or_execution: bool,
    do_reject: bool,
    do_message: bool,
) -> None:
    queued_transactions = safe_api_utils.get_safe_queued_transactions(safe_address)
    logger.info(
        f"Retrieved {len(queued_transactions)} queued transactions to verify for {safe_address}."
    )

    for queued_transaction in queued_transactions:
        validate_safe_transaction(
            safe_address,
            queued_transaction,
            do_sign_or_execution,
            do_reject,
            do_message,
        )


def validate_safe_transaction(
    safe_address: ChecksumAddress,
    transaction: Transaction,
    do_sign_or_execution: bool,
    do_reject: bool,
    do_message: bool,
    ignore_historical_transaction_ids: set[str] | None = None,
) -> list[ValidationResult] | None:
    api_keys = APIKeys()
    safe = get_safe(safe_address)
    is_owner = safe.retrieve_is_owner(api_keys.bet_from_address)

    if not is_owner:
        logger.warning(
            f"{api_keys.bet_from_address} is not owner of Safe {safe_address}, some functionality may be limited."
        )

    logger.info(f"Processing transaction {transaction.id}.")

    historical_transactions = safe_api_utils.get_safe_history(safe_address)
    detailed_historical_transactions = [
        safe_api_utils.get_safe_detailed_transaction_info(transaction_id=tx.id)
        for tx in historical_transactions
        if ignore_historical_transaction_ids is None
        or tx.id not in ignore_historical_transaction_ids
    ]
    logger.info(
        f"Retrieved {len(detailed_historical_transactions)} historical transactions."
    )

    detailed_transaction_info = safe_api_utils.get_safe_detailed_transaction_info(
        transaction_id=transaction.id
    )
    safe_tx = safe_api_utils.safe_tx_from_detailed_transaction(
        safe, detailed_transaction_info
    )

    validation_results: list[ValidationResult] = []

    logger.info("Running the transaction validation...")
    for safe_guard_fn in SAFE_GUARDS:
        logger.info(f"Running {safe_guard_fn.__name__}...")
        validation_result = safe_guard_fn(
            detailed_transaction_info,
            safe_tx,
            detailed_historical_transactions,
        )
        (logger.success if validation_result.ok else logger.warning)(
            f"Validation result: {validation_result}"
        )
        validation_results.append(validation_result)

        if not validation_result.ok:
            logger.error(
                f"Validation using {safe_guard_fn.__name__} reported malicious activity."
            )
            break

    if all(result.ok for result in validation_results):
        logger.success("All validations successful.")
        if do_sign_or_execution:
            sign_or_execute(safe, safe_tx, api_keys)

    else:
        logger.error("At least one validation failed.")
        if do_reject:
            reject_transaction(safe, safe_tx, api_keys)

    logger.info("Done.")
    if do_message:
        send_message(safe, transaction.id, validation_results, api_keys)
    return validation_results


def send_message(
    safe: Safe,
    transaction_id: str,
    validation_results: list[ValidationResult],
    api_keys: APIKeys,
) -> None:
    ok = all(result.ok for result in validation_results)
    reasons_formatted = "\n".join(
        f"- {'OK' if result.ok else 'Failed'} -- {result.reason}"
        for result in validation_results
    )
    message = f"""Your transaction with id `{transaction_id}` was {'approved' if ok else 'rejected'}.

Reasons:
{reasons_formatted}
"""
    post_message(safe, message, api_keys)
