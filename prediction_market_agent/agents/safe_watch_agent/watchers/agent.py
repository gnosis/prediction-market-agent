from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import ChainID, ChecksumAddress
from prediction_market_agent_tooling.tools.langfuse_ import observe
from safe_eth.safe.safe import SafeTx

from prediction_market_agent.agents.safe_watch_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_watch_agent.validation_result import (
    ValidationResult,
)
from prediction_market_agent.agents.safe_watch_agent.watchers.abstract_watch import (
    AbstractWatch,
)


class DoNotRemoveAgent(AbstractWatch):
    name = "Agent remains owner"
    description = "This watch ensures that the transaction doesn't remove the agent itself from the owners of the Safe."

    @observe(name="validate_do_not_remove_agent")
    def validate(
        self,
        new_transaction: DetailedTransactionResponse,
        new_transaction_safetx: SafeTx,
        all_addresses_from_tx: list[ChecksumAddress],
        history: list[DetailedTransactionResponse],
        chain_id: ChainID,
    ) -> ValidationResult:
        if (
            not new_transaction.txData
            or not new_transaction.txData.dataDecoded
            or new_transaction.txData.dataDecoded.get("method") != "removeOwner"
            or (
                # Based on https://github.com/safe-global/safe-smart-account/blob/main/contracts/base/OwnerManager.sol#L73.
                new_transaction.txData.dataDecoded["parameters"][1]["value"].lower()
                != APIKeys().bet_from_address.lower()
            )
        ):
            return ValidationResult(
                name=self.name,
                description=self.description,
                ok=True,
                reason="The transaction does not remove the agent from owners.",
            )

        return ValidationResult(
            name=self.name,
            description=self.description,
            ok=False,
            reason="Agent will not confirm a transaction that removes itself from owners.",
        )
