from goplus.address import Address
from goplus.nft import Nft
from goplus.token import Token
from prediction_market_agent_tooling.config import RPCConfig
from prediction_market_agent_tooling.tools.langfuse_ import observe
from safe_eth.safe.safe import SafeTx

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_guard_agent.validation_result import (
    ValidationResult,
)


@observe()
def validate_safe_transaction_goplus_token_security(
    new_transaction: DetailedTransactionResponse,
    new_transaction_safetx: SafeTx,
    history: list[DetailedTransactionResponse],
) -> ValidationResult:
    addr = new_transaction_safetx.to
    data = Token().token_security(chain_id=f"{RPCConfig().chain_id}", addresses=[addr])
    return ValidationResult(
        ok=data.message.lower() == "ok",
        reason=str(data.result),
    )


@observe()
def validate_safe_transaction_goplus_address_security(
    new_transaction: DetailedTransactionResponse,
    new_transaction_safetx: SafeTx,
    history: list[DetailedTransactionResponse],
) -> ValidationResult:
    addr = new_transaction_safetx.to
    data = Address().address_security(chain_id=f"{RPCConfig().chain_id}", address=addr)
    return ValidationResult(
        ok=data.message.lower() == "ok",
        reason=str(data.result),
    )


@observe()
def validate_safe_transaction_goplus_nft_security(
    new_transaction: DetailedTransactionResponse,
    new_transaction_safetx: SafeTx,
    history: list[DetailedTransactionResponse],
) -> ValidationResult:
    addr = new_transaction_safetx.to
    data = Nft().nft_security(chain_id=f"{RPCConfig().chain_id}", address=addr)
    return ValidationResult(
        ok=data.message.lower() in ("ok", "non-contract address"),
        reason=str(data.result),
    )
