from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import ChainID, ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.langfuse_ import observe
from safe_eth.safe.safe import Safe, SafeTx

from prediction_market_agent.agents.safe_guard_agent import safe_api_utils
from prediction_market_agent.agents.safe_guard_agent.guards import (
    agent,
    blacklist,
    goplus_,
    hash_checker,
    llm,
)
from prediction_market_agent.agents.safe_guard_agent.guards.abstract_guard import (
    AbstractGuard,
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
)
from prediction_market_agent.agents.safe_guard_agent.validation_summary import (
    create_validation_summary,
)
from prediction_market_agent.agents.safe_guard_agent.whitelist import is_whitelisted

SAFE_GUARDS: list[type[AbstractGuard]] = [
    agent.DoNotRemoveAgent,
    blacklist.Blacklist,
    hash_checker.HashCheck,
    llm.LLM,
    goplus_.GoPlusTokenSecurity,
    goplus_.GoPlusAddressSecurity,
    goplus_.GoPlusNftSecurity,
]


def validate_all(
    do_sign: bool,
    do_execution: bool,
    do_reject: bool,
    do_message: bool,
    chain_id: ChainID,
) -> None:
    api_keys = APIKeys()

    safes_to_verify = get_safes(api_keys.bet_from_address, chain_id)
    logger.info(
        f"For owner {api_keys.bet_from_address}, retrieved {safes_to_verify} safes to verify transactions for."
    )

    for safe_address in safes_to_verify:
        validate_safe(
            safe_address,
            do_sign=do_sign,
            do_execution=do_execution,
            do_reject=do_reject,
            do_message=do_message,
            api_keys=api_keys,
            chain_id=chain_id,
        )


def validate_safe(
    safe_address: ChecksumAddress,
    do_sign: bool,
    do_execution: bool,
    do_reject: bool,
    do_message: bool,
    api_keys: APIKeys,
    chain_id: ChainID,
) -> None:
    # Load only multisig transactions here, others are not relevant for the agent to check.
    queued_transactions = safe_api_utils.get_safe_queue_multisig(
        safe_address, chain_id=chain_id
    )
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
            do_sign=do_sign,
            do_execution=do_execution,
            do_reject=do_reject,
            do_message=do_message,
            chain_id=chain_id,
        )


@observe()
def validate_safe_transaction(
    transaction_id: str,
    do_sign: bool,
    do_execution: bool,
    do_reject: bool,
    do_message: bool,
    chain_id: ChainID,
    ignore_historical_transaction_ids: set[str] | None = None,
) -> ValidationConclusion:
    detailed_transaction_info = safe_api_utils.get_safe_detailed_transaction_info(
        transaction_id=transaction_id,
        chain_id=chain_id,
    )
    return validate_safe_transaction_obj(
        detailed_transaction_info=detailed_transaction_info,
        do_sign=do_sign,
        do_execution=do_execution,
        do_reject=do_reject,
        do_message=do_message,
        ignore_historical_transaction_ids=ignore_historical_transaction_ids,
        chain_id=chain_id,
    )


@observe()
def validate_safe_transaction_obj(
    detailed_transaction_info: DetailedTransactionResponse,
    do_sign: bool,
    do_execution: bool,
    do_reject: bool,
    do_message: bool,
    chain_id: ChainID,
    ignore_historical_transaction_ids: set[str] | None = None,
) -> ValidationConclusion:
    api_keys = APIKeys()

    all_addresses_from_tx_raw = extract_all_addresses_or_raise(
        detailed_transaction_info
    )

    safe_address = detailed_transaction_info.safeAddress
    safe = get_safe(safe_address, chain_id)
    is_owner = safe.retrieve_is_owner(api_keys.bet_from_address)

    if not is_owner:
        logger.warning(
            f"{api_keys.bet_from_address} is not owner of Safe {safe_address}, some functionality may be limited."
        )

    logger.info(
        f"Processing transaction {detailed_transaction_info.txId}.", streamlit=True
    )
    logger.info(
        f"The transaction interacts with the following addresses ({len(all_addresses_from_tx_raw)}): {all_addresses_from_tx_raw}"
    )

    all_addresses_from_tx_not_whitelisted = [
        addr for addr in all_addresses_from_tx_raw if not is_whitelisted(addr)
    ]
    logger.info(
        f"The transaction interacts with the following addresses that will be verified ({len(all_addresses_from_tx_not_whitelisted)}): {all_addresses_from_tx_not_whitelisted}",
        streamlit=True,
    )

    # Load all historical transactions here (include non-multisig transactions, so the agent has overall overview of the Safe).
    historical_transactions = safe_api_utils.get_safe_history(
        safe_address, chain_id=chain_id
    )
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
        ],
        chain_id=chain_id,
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
        all_addresses_from_tx_not_whitelisted,
        detailed_historical_transactions,
        chain_id=chain_id,
    )
    ok = all(result.ok for result in validation_results)

    if ok:
        logger.success("All validations successful.")
        if do_sign or do_execution:
            sign_or_execute(safe, safe_tx, api_keys, allow_exec=do_execution)

    else:
        logger.warning("At least one validation failed.")
        if do_reject:
            reject_transaction(safe, safe_tx, api_keys)

    summary = create_validation_summary(detailed_transaction_info, validation_results)

    logger.info("Done.")

    conclusion = ValidationConclusion(
        txId=detailed_transaction_info.txId,
        all_ok=ok,
        summary=summary,
        results=validation_results,
    )

    if do_message:
        send_message(safe, conclusion, api_keys)

    return conclusion


@observe()
def run_safe_guards(
    safe_tx: SafeTx,
    detailed_transaction_info: DetailedTransactionResponse,
    all_addresses_from_tx: list[ChecksumAddress],
    detailed_historical_transactions: list[DetailedTransactionResponse],
    chain_id: ChainID,
) -> list[ValidationResult]:
    validation_results: list[ValidationResult] = []
    logger.info("Running the transaction validation...")
    for safe_guard_class in SAFE_GUARDS:
        safe_guard = safe_guard_class()
        logger.info(
            f"Running guard {safe_guard.name} -- {safe_guard.description}",
            streamlit=True,
        )
        validation_result = safe_guard.validate(
            detailed_transaction_info,
            safe_tx,
            all_addresses_from_tx,
            detailed_historical_transactions,
            chain_id=chain_id,
        )
        if validation_result is None:
            logger.info(
                f"Skipping {safe_guard.name} because it isn't supported for the given SafeTX.",
                streamlit=True,
            )
            continue
        (logger.success if validation_result.ok else logger.warning)(
            f"Guard {safe_guard.name} validation result: {validation_result.reason}",
            streamlit=True,
        )
        validation_results.append(validation_result)

        if not validation_result.ok:
            logger.warning(
                f"Validation using {validation_result.name} reported malicious activity."
            )

    return validation_results


def send_message(
    safe: Safe,
    conclusion: ValidationConclusion,
    api_keys: APIKeys,
) -> None:
    message = f"""Your transaction with id `{conclusion.txId}` was {'approved' if conclusion.all_ok else 'rejected'}.

{conclusion.summary}
"""
    post_message(safe, message, api_keys)
