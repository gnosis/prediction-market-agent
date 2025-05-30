from prediction_market_agent_tooling.gtypes import ChainID, ChecksumAddress
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import check_not_none
from safe_eth.safe import SafeTx

from prediction_market_agent.agents.safe_watch_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_watch_agent.validation_result import (
    ValidationResult,
)
from prediction_market_agent.agents.safe_watch_agent.watchers.abstract_watch import (
    AbstractWatch,
)


class HashCheck(AbstractWatch):
    name = "Hash check"
    description = "This watch ensures that the hash delivered from Safe API and the hash calculated by the agent match."

    @observe(name="validate_safe_transaction_hash")
    def validate(
        self,
        new_transaction: DetailedTransactionResponse,
        new_transaction_safetx: SafeTx,
        all_addresses_from_tx: list[ChecksumAddress],
        history: list[DetailedTransactionResponse],
        chain_id: ChainID,
    ) -> ValidationResult:
        """Function that checks if the safe_tx_hashes match."""

        detailedExecutionInfo = check_not_none(new_transaction.detailedExecutionInfo)

        # Here we compare the safe_tx_hash delivered by the Safe client (see get_safe_detailed_transaction_info) and the
        # safe constructed by safe-eth-py library (see the usage of safe.build_multisig_tx inside safe_tx_from_detailed_transaction).
        # This essentially verifies that the TransactionService computed the hash correctly. Ideally a third-party integration
        # is also used, that checks that the safe_tx_hash from the original intent of the user matches the calculated safe_tx_hash here.
        safe_tx_hash_from_new_transaction = detailedExecutionInfo.safeTxHash
        safe_tx_hash_from_built_tx = new_transaction_safetx.safe_tx_hash
        if safe_tx_hash_from_new_transaction != safe_tx_hash_from_built_tx.hex():
            return ValidationResult(
                name=self.name,
                description=self.description,
                ok=False,
                reason="Safe tx hashes do not match.",
            )

        return ValidationResult(
            name=self.name,
            description=self.description,
            ok=True,
            reason="Hashes match.",
        )
