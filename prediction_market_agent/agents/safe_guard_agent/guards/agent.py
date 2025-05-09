from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.tools.langfuse_ import observe
from safe_eth.safe.safe import SafeTx

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_guard_agent.validation_result import (
    ValidationResult,
)


@observe()
def validate_do_not_remove_agent(
    new_transaction: DetailedTransactionResponse,
    new_transaction_safetx: SafeTx,
    all_addresses_from_tx: list[ChecksumAddress],
    history: list[DetailedTransactionResponse],
) -> ValidationResult:
    ok_result = ValidationResult(
        ok=True, reason="Doesn't remove agent from the owner list."
    )

    if not new_transaction.txData:
        return ok_result

    if not new_transaction.txData.dataDecoded:
        return ok_result

    if new_transaction.txData.dataDecoded.get("method") != "removeOwner":
        return ok_result

    if (
        # Based on https://github.com/safe-global/safe-smart-account/blob/main/contracts/base/OwnerManager.sol#L73.
        new_transaction.txData.dataDecoded["parameters"][1]["value"].lower()
        != APIKeys().bet_from_address.lower()
    ):
        return ok_result

    return ValidationResult(
        ok=False, reason="Agent can not accept removal of itself from the owners."
    )
