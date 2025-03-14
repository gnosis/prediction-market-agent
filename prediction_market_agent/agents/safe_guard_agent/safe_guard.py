from typing import Callable

from eth_typing import URI
from prediction_market_agent_tooling.config import APIKeys, RPCConfig
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from safe_eth.eth import EthereumClient
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
from prediction_market_agent.agents.safe_guard_agent.safe_utils import (
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
    quened_transactions = safe_api_utils.get_safe_quened_transactions(safe_address)
    logger.info(
        f"Retrieved {len(quened_transactions)} quened transactions to verify for {safe_address}."
    )

    for quened_transaction in quened_transactions:
        validate_safe_transaction(
            safe_address,
            quened_transaction.id,
            do_sign_or_execution,
            do_reject,
            do_message,
        )


def validate_safe_transaction(
    safe_address: ChecksumAddress,
    transaction_id: str,
    do_sign_or_execution: bool,
    do_reject: bool,
    do_message: bool,
) -> list[ValidationResult] | None:
    api_keys = APIKeys()
    safe = Safe(safe_address, EthereumClient(URI(RPCConfig().gnosis_rpc_url)))  # type: ignore

    if not safe.retrieve_is_owner(api_keys.bet_from_address):
        logger.error(
            f"{api_keys.bet_from_address} is not owner of Safe {safe_address}. Please add him as a signer and then try again."
        )
        return None

    quened_transactions = safe_api_utils.get_safe_quened_transactions(safe_address)
    quened_transaction_to_process = next(
        (
            quened_transaction
            for quened_transaction in quened_transactions
            if quened_transaction.id == transaction_id
        ),
        None,
    )

    if quened_transaction_to_process is None:
        raise ValueError(
            f"Transaction {transaction_id} not found in quened transactions for {safe_address}."
        )

    logger.info(f"Processing quened transaction {quened_transaction_to_process.id}.")

    historical_transactions = safe_api_utils.get_safe_history(safe_address)
    detailed_historical_transactions = [
        safe_api_utils.get_safe_detailed_transaction_info(transaction_id=tx.id)
        for tx in historical_transactions
    ]
    logger.info(
        f"Retrieved {len(detailed_historical_transactions)} historical transactions."
    )

    detailed_quened_transaction_info = (
        safe_api_utils.get_safe_detailed_transaction_info(
            transaction_id=quened_transaction_to_process.id
        )
    )
    quened_safe_tx = safe_api_utils.safe_tx_from_detailed_transaction(
        safe, detailed_quened_transaction_info
    )

    validation_results: list[ValidationResult] = []

    logger.info("Running the transaction validation...")
    for safe_guard_fn in SAFE_GUARDS:
        logger.info(f"Running {safe_guard_fn.__name__}...")
        validation_result = safe_guard_fn(
            detailed_quened_transaction_info,
            quened_safe_tx,
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
            sign_or_execute(safe, quened_safe_tx, api_keys)

    else:
        logger.error("At least one validation failed.")
        if do_reject:
            reject_transaction(safe, quened_safe_tx, api_keys)

    logger.info("Done.")
    if do_message:
        send_message(safe, transaction_id, validation_results, api_keys)
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
