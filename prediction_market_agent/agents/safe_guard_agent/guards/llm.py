from langfuse.openai import AsyncOpenAI
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.datetime_utc import DatetimeUTC
from prediction_market_agent_tooling.tools.langfuse_ import observe
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from safe_eth.safe.safe import SafeTx

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_models.transactions import (
    SwapOrderTxInfo,
    TransferTxInfo,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_utils import (
    get_balances_usd,
)
from prediction_market_agent.agents.safe_guard_agent.validation_result import (
    ValidationResult,
)
from prediction_market_agent.utils import APIKeys


@observe()
def validate_safe_transaction_llm(
    new_transaction: DetailedTransactionResponse,
    new_transaction_safetx: SafeTx,
    history: list[DetailedTransactionResponse],
) -> ValidationResult:
    agent = Agent(
        OpenAIModel(
            "o1",
            openai_client=AsyncOpenAI(
                api_key=APIKeys().openai_api_key.get_secret_value()
            ),
        ),
        system_prompt="""You are fraud detection agent. 
You need to determine if the new transaction is malicious or not.
Take a look at the new transaction details and also consider the previous transactions already made.
You don't know if previous transactions were malicious or not.
Keep in mind, better be safe than sorry.
""",
        result_type=ValidationResult,
    )

    balances_formatted = format_balances(new_transaction.safeAddress)
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

{{"ok": bool, "reason": string}}
"""
    logger.info(f"Prompting LLM agent with:\n\n\n{prompt}")

    result = agent.run_sync(prompt)
    return result.data


def format_transaction(tx: DetailedTransactionResponse) -> str:
    tx_info = tx.txInfo

    sender_value = (
        tx_info.sender.value
        if isinstance(tx_info, TransferTxInfo)
        else tx_info.owner
        if isinstance(tx_info, SwapOrderTxInfo)
        else "N/A"
    )
    recipient_value = (
        tx_info.recipient.value
        if isinstance(tx_info, TransferTxInfo)
        else tx_info.receiver
        if isinstance(tx_info, SwapOrderTxInfo)
        else "N/A"
    )
    transfer_value = (
        tx_info.transferInfo.value
        if isinstance(tx_info, TransferTxInfo)
        else tx_info.sellAmount
        if isinstance(tx_info, SwapOrderTxInfo)
        else "N/A"
    )

    return (
        f"Transaction ID: {tx.txId} | "
        + f"Transaction type: {tx.txInfo.type} | "
        + f"Human description: {tx.txInfo.humanDescription} | "
        + f"Time: {DatetimeUTC.to_datetime_utc(tx.detailedExecutionInfo.submittedAt) if tx.detailedExecutionInfo and tx.detailedExecutionInfo.submittedAt else 'N/A'} | "
        + f"Sender: {sender_value} | "
        + f"Recipient: {recipient_value} | "
        + (
            f"Sell token: {tx.txInfo.sellToken.symbol} | "
            if isinstance(tx.txInfo, SwapOrderTxInfo)
            else ""
        )
        + (
            f"Buy token: {tx.txInfo.buyToken.symbol} | "
            if isinstance(tx.txInfo, SwapOrderTxInfo)
            else ""
        )
        + f"Transfer value: {transfer_value} ;"
    )


def format_balances(safe_address: ChecksumAddress) -> str:
    balances = get_balances_usd(safe_address)
    formatted_items = "\n\n".join(
        f"Token: {item.tokenInfo.name} ({item.tokenInfo.symbol}) | "
        f"Address: {item.tokenInfo.address} | "
        f"Balance: {item.balance} | "
        f"Fiat Balance: {item.fiatBalance} USD | "
        f"Fiat Conversion Rate: {item.fiatConversion} USD ;"
        for item in balances.items
    )
    return f"Safe address: {safe_address}\n\nTotal Fiat Balance: {balances.fiatTotal} USD\n\nToken Balances:\n\n{formatted_items}"
