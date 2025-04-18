from typing import Callable

from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.langfuse_ import observe
from safe_eth.safe.safe import Safe, SafeTx

from prediction_market_agent.agents.safe_guard_agent import safe_api_utils
from prediction_market_agent.agents.safe_guard_agent.guards import (
    blacklist,
    goplus_,
    hash_checker,
    llm,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    CreationTxInfo,
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_utils import (
    is_already_canceled,
    signer_is_missing,
)
from prediction_market_agent.agents.safe_guard_agent.safe_utils import (
    extract_all_addresses_or_raise,
    get_safe,
    get_safes,
    post_message,
    reject_transaction,
    sign_or_execute,
)
from prediction_market_agent.agents.safe_guard_agent.validation_result import (
    ValidationConclusion,
    ValidationResult,
    ValidationResultWithName,
)

SAFE_GUARDS: list[
    Callable[
        [
            DetailedTransactionResponse,
            SafeTx,
            list[ChecksumAddress],
            list[DetailedTransactionResponse],
        ],
        ValidationResult | None,
    ]
] = [
    # Keep ordered from cheapest/fastest to most expensive/slowest.
    blacklist.validate_safe_transaction_blacklist,
    hash_checker.validate_safe_transaction_hash,
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
    api_keys = APIKeys()

    safes_to_verify = get_safes(api_keys.bet_from_address)
    logger.info(
        f"For owner {api_keys.bet_from_address}, retrieved {safes_to_verify} safes to verify transactions for."
    )

    for safe_address in safes_to_verify:
        validate_safe(
            safe_address,
            do_sign_or_execution,
            do_reject,
            do_message,
            api_keys,
        )


def validate_safe(
    safe_address: ChecksumAddress,
    do_sign_or_execution: bool,
    do_reject: bool,
    do_message: bool,
    api_keys: APIKeys,
) -> None:
    # Load only multisig transactions here, others are not relevant for the agent to check.
    queued_transactions = safe_api_utils.get_safe_queue_multisig(safe_address)
    logger.info(
        f"Retrieved {len(queued_transactions)} queued transactions to verify for {safe_address}."
    )

    for queued_transaction in queued_transactions:
        if is_already_canceled(queued_transaction, all_queued_txs=queued_transactions):
            logger.info(
                f"Skipping {queued_transaction.id=} because it already been canceled."
            )
            continue

        if not signer_is_missing(api_keys.bet_from_address, queued_transaction):
            logger.info(
                f"Skipping {queued_transaction.id=} because it has already been signed by the agent {api_keys.bet_from_address} or no signer is required."
            )
            continue

        validate_safe_transaction(
            queued_transaction.id,
            do_sign_or_execution,
            do_reject,
            do_message,
        )


@observe()
def validate_safe_transaction(
    transaction_id: str,
    do_sign_or_execution: bool,
    do_reject: bool,
    do_message: bool,
    ignore_historical_transaction_ids: set[str] | None = None,
) -> ValidationConclusion:
    api_keys = APIKeys()

    detailed_transaction_info = safe_api_utils.get_safe_detailed_transaction_info(
        transaction_id=transaction_id
    )
    all_addresses_from_tx = extract_all_addresses_or_raise(detailed_transaction_info)

    safe_address = detailed_transaction_info.safeAddress
    safe = get_safe(safe_address)
    is_owner = safe.retrieve_is_owner(api_keys.bet_from_address)

    if not is_owner:
        logger.warning(
            f"{api_keys.bet_from_address} is not owner of Safe {safe_address}, some functionality may be limited."
        )

    logger.info(f"Processing transaction {transaction_id}.")
    logger.info(
        f"Transaction interacts with the following addresses: {all_addresses_from_tx}"
    )

    # Load all historical transactions here (include non-multisig transactions, so the agent has overall overview of the Safe).
    historical_transactions = safe_api_utils.get_safe_history(safe_address)
    detailed_historical_transactions = safe_api_utils.gather_safe_detailed_transaction_info(
        [
            tx.id
            for tx in historical_transactions
            if (
                ignore_historical_transaction_ids is None
                or tx.id not in ignore_historical_transaction_ids
            )
            # Creation tx can not be converted to detailed transaction info and isn't needed for validation anyway.
            and not isinstance(tx.txInfo, CreationTxInfo)
        ]
    )

    logger.info(
        f"Retrieved {len(detailed_historical_transactions)} historical transactions."
    )

    safe_tx = safe_api_utils.safe_tx_from_detailed_transaction(
        safe, detailed_transaction_info
    )

    validation_results = run_safe_guards(
        safe_tx,
        detailed_transaction_info,
        all_addresses_from_tx,
        detailed_historical_transactions,
    )
    ok = all(result.ok for result in validation_results)

    if ok:
        logger.success("All validations successful.")
        if do_sign_or_execution:
            sign_or_execute(safe, safe_tx, api_keys)

    else:
        logger.error("At least one validation failed.")
        if do_reject:
            reject_transaction(safe, safe_tx, api_keys)

    logger.info("Done.")
    if do_message:
        send_message(safe, transaction_id, validation_results, api_keys)

    return ValidationConclusion(
        ok=ok,
        results=validation_results,
    )


@observe()
def run_safe_guards(
    safe_tx: SafeTx,
    detailed_transaction_info: DetailedTransactionResponse,
    all_addresses_from_tx: list[ChecksumAddress],
    detailed_historical_transactions: list[DetailedTransactionResponse],
) -> list[ValidationResultWithName]:
    validation_results: list[ValidationResultWithName] = []
    logger.info("Running the transaction validation...")
    for safe_guard_fn in SAFE_GUARDS:
        logger.info(f"Running {safe_guard_fn.__name__}...")
        validation_result = safe_guard_fn(
            detailed_transaction_info,
            safe_tx,
            all_addresses_from_tx,
            detailed_historical_transactions,
        )
        if validation_result is None:
            logger.info(
                f"Skipping {safe_guard_fn.__name__} because it isn't supported for the given SafeTX."
            )
            continue
        validation_result_with_name = ValidationResultWithName.from_result(
            validation_result, safe_guard_fn.__name__
        )
        (logger.success if validation_result.ok else logger.warning)(
            f"Validation result: {validation_result}"
        )
        validation_results.append(validation_result_with_name)

        if not validation_result.ok:
            logger.error(
                f"Validation using {validation_result_with_name.name} reported malicious activity."
            )

    return validation_results


def send_message(
    safe: Safe,
    transaction_id: str,
    validation_results: list[ValidationResultWithName],
    api_keys: APIKeys,
) -> None:
    ok = all(result.ok for result in validation_results)
    reasons_formatted = "\n".join(
        f"- {result.name}: {'OK' if result.ok else 'Failed'} -- {result.reason}"
        for result in validation_results
    )
    message = f"""Your transaction with id `{transaction_id}` was {'approved' if ok else 'rejected'}.

Reasons:
{reasons_formatted}
"""
    post_message(safe, message, api_keys)
