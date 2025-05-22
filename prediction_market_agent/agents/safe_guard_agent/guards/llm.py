from prediction_market_agent_tooling.gtypes import ChainID, ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.datetime_utc import DatetimeUTC
from prediction_market_agent_tooling.tools.langfuse_ import observe
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from safe_eth.safe.safe import SafeTx

from prediction_market_agent.agents.safe_guard_agent.guards.abstract_guard import (
    AbstractGuard,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    DetailedExecutionInfo,
    DetailedTransactionResponse,
    TxData,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_utils import (
    get_balances_usd,
)
from prediction_market_agent.agents.safe_guard_agent.validation_result import (
    ValidationResult,
)
from prediction_market_agent.tools.openai_utils import get_openai_provider
from prediction_market_agent.utils import APIKeys

HISTORY_LIMIT = 25


class LLMValidationResult(BaseModel):
    reason: str
    ok: bool


class LLM(AbstractGuard):
    name = "LLM"
    description = "This guard uses a large language model to analyze the transaction and determine if it is malicious."

    @observe(name="validate_safe_transaction_llm")
    def validate(
        self,
        new_transaction: DetailedTransactionResponse,
        new_transaction_safetx: SafeTx,
        all_addresses_from_tx: list[ChecksumAddress],
        history: list[DetailedTransactionResponse],
        chain_id: ChainID,
    ) -> ValidationResult:
        # Get the latest ones.
        history = sorted(
            history, key=lambda x: x.executedAt or float("-inf"), reverse=True
        )[:HISTORY_LIMIT]

        agent = Agent(
            OpenAIModel(
                "gpt-4o",
                provider=get_openai_provider(api_key=APIKeys().openai_api_key),
            ),
            system_prompt="""You are fraud detection agent. 
You need to determine if the new transaction is malicious or not.
Take a look at the new transaction details and also consider the previous transactions already made.
You don't know if previous transactions were malicious or not.
The number of current confirmations do not matter, don't take them into account and do not describe them in your answer.
Consider money and wallet balance in absolute terms, not in relative terms.
Keep in mind, better be safe than sorry.
""",
            result_type=LLMValidationResult,
        )

        balances_formatted = format_balances(
            new_transaction.safeAddress, chain_id=chain_id
        )
        new_formatted = format_transaction(new_transaction)
        history_formatted = (
            "\n\n".join(format_transaction(tx) for tx in history)
            if history
            else "No historical transactions found."
        )

        prompt = f"""Current status of the wallet is:

{balances_formatted}

---
        
The new transaction that is being made is as follows:

{new_formatted}

---

The history of transactions made by this Safe is as follows:

{history_formatted}

---

Is the new transaction malicious or not? Why? Output your answer in the JSON format with the following structure:

{{"reason": string (under 500 characters), "ok": bool}}
"""
        logger.info(f"Prompting LLM agent with:\n\n\n{prompt}")

        result = agent.run_sync(prompt).output
        return ValidationResult(
            name=self.name,
            description=self.description,
            reason=result.reason,
            ok=result.ok,
        )


def format_transaction(tx: DetailedTransactionResponse) -> str:
    formatted = (
        f"Transaction ID: {tx.txId} | "
        + (
            f"Executed at: {DatetimeUTC.to_datetime_utc(tx.executedAt)} | "
            if tx.executedAt
            else ""
        )
        + format_tx_data(tx.txData)
        if tx.txData
        else ""
    )
    if tx.detailedExecutionInfo is not None:
        formatted += format_detailed_execution_info(tx.detailedExecutionInfo)
    formatted += tx.txInfo.format_llm()
    return formatted


def format_tx_data(tx_data: TxData) -> str:
    return (
        (f"Decoded TX data: {tx_data.dataDecoded} | " if tx_data.dataDecoded else "")
        + f"To: {tx_data.to.value} | "
        + f"Value: {tx_data.value} | "
        + f"Operation: {tx_data.operation} | "
        + (
            f"Trusted delegate call target: {tx_data.trustedDelegateCallTarget} | "
            if tx_data.trustedDelegateCallTarget
            else ""
        )
    )


def format_detailed_execution_info(exec_info: DetailedExecutionInfo) -> str:
    return (
        f"Type: {exec_info.type} | "
        + (f"Address: {exec_info.address.value} | " if exec_info.address else "")
        + (
            f"Submitted at: {DatetimeUTC.to_datetime_utc(exec_info.submittedAt)} | "
            if exec_info.submittedAt
            else ""
        )
        + (f"Nonce: {exec_info.nonce} | " if exec_info.nonce else "")
        + (f"Safe TX gas: {exec_info.safeTxGas} | " if exec_info.safeTxGas else "")
        + (f"Base gas: {exec_info.baseGas} | " if exec_info.baseGas else "")
        + (f"Gas price: {exec_info.gasPrice} | " if exec_info.gasPrice else "")
        + (f"Gas token: {exec_info.gasToken} | " if exec_info.gasToken else "")
        + (
            f"Refund receiver: {exec_info.refundReceiver.value} | "
            if exec_info.refundReceiver
            else ""
        )
        + (f"Executor: {exec_info.executor.value} | " if exec_info.executor else "")
        + (
            f"Signers: {[s.value for s in exec_info.signers]} | "
            if exec_info.signers
            else ""
        )
        + (
            f"Confirmations required: {exec_info.confirmationsRequired} | "
            if exec_info.confirmationsRequired
            else ""
        )
        + (
            f"Confirmations: {[c.signer.value for c in exec_info.confirmations]} | "
            if exec_info.confirmations
            else ""
        )
        + (f"Rejectors: {exec_info.rejectors} | " if exec_info.rejectors else "")
        + (f"Trusted: {exec_info.trusted} | " if exec_info.trusted is not None else "")
        + (f"Proposer: {exec_info.proposer.value} | " if exec_info.proposer else "")
        + (
            f"Proposed by delegate: {exec_info.proposedByDelegate} | "
            if exec_info.proposedByDelegate
            else ""
        )
    )


def format_balances(safe_address: ChecksumAddress, chain_id: ChainID) -> str:
    balances = get_balances_usd(safe_address, chain_id)
    formatted_items = "\n\n".join(
        f"Token: {item.tokenInfo.name} ({item.tokenInfo.symbol}) | "
        f"Address: {item.tokenInfo.address} | "
        f"Balance: {item.balance} | "
        f"Fiat Balance: {item.fiatBalance} USD | "
        f"Fiat Conversion Rate: {item.fiatConversion} USD | "
        for item in balances.items
    )
    return f"Safe address: {safe_address}\n\nTotal Fiat Balance: {balances.fiatTotal} USD\n\nToken Balances:\n\n{formatted_items}"
