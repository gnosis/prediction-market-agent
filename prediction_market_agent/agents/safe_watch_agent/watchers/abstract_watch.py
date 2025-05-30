from abc import ABC, abstractmethod

from prediction_market_agent_tooling.gtypes import ChainID, ChecksumAddress
from safe_eth.safe.safe import SafeTx

from prediction_market_agent.agents.safe_watch_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_watch_agent.validation_result import (
    ValidationResult,
)


class AbstractWatch(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the watch."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the watch."""

    @abstractmethod
    def validate(
        self,
        new_transaction: DetailedTransactionResponse,
        new_transaction_safetx: SafeTx,
        all_addresses_from_tx: list[ChecksumAddress],
        history: list[DetailedTransactionResponse],
        chain_id: ChainID,
    ) -> ValidationResult | None:
        raise NotImplementedError("Subclasses should implement this method.")
