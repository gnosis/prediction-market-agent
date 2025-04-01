from langfuse.openai import AsyncOpenAI
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.datetime_utc import DatetimeUTC
from prediction_market_agent_tooling.tools.langfuse_ import observe
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from safe_eth.safe.safe import SafeTx

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    CreationTxInfo,
    CustomTxInfo,
    DetailedExecutionInfo,
    DetailedTransactionResponse,
    SettingsChangeTxInfo,
    SwapOrderTxInfo,
    TransferTxInfo,
    TxData,
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

    tx_info = tx.txInfo

    if isinstance(tx_info, CreationTxInfo):
        formatted += format_creation_info(tx_info)

    elif isinstance(tx_info, SettingsChangeTxInfo):
        formatted += format_settings_change_info(tx_info)

    elif isinstance(tx_info, TransferTxInfo):
        formatted += format_transfer_info(tx_info)

    elif isinstance(tx_info, SwapOrderTxInfo):
        formatted += format_swap_order_info(tx_info)

    elif isinstance(tx_info, CustomTxInfo):
        formatted += format_custom_info(tx_info)

    else:
        raise ValueError(f"Unknown transaction type: {tx_info}")

    return formatted


def format_creation_info(tx_info: CreationTxInfo) -> str:
    return f"Creator address: {tx_info.creator.value} | " + (
        f"Human description: {tx_info.humanDescription} | "
        if tx_info.humanDescription
        else ""
    )


def format_settings_change_info(tx_info: SettingsChangeTxInfo) -> str:
    return f"Transaction type: {tx_info.type} | " + (
        f"Human description: {tx_info.humanDescription} | "
        if tx_info.humanDescription
        else ""
    )


def format_transfer_info(tx_info: TransferTxInfo) -> str:
    return (
        f"Transaction type: {tx_info.type} | "
        + (
            f"Human description: {tx_info.humanDescription} | "
            if tx_info.humanDescription
            else ""
        )
        + f"Sender: {tx_info.sender.value} | "
        + f"Recipient: {tx_info.recipient.value} | "
        + f"Direction: {tx_info.direction} | "
        + f"Transfer token type: {tx_info.transferInfo.type} | "
        + (
            f"Transfer token address: {tx_info.transferInfo.tokenAddress} | "
            if tx_info.transferInfo.tokenAddress
            else ""
        )
        + (
            f"Transfer token symbol: {tx_info.transferInfo.tokenSymbol} | "
            if tx_info.transferInfo.tokenSymbol
            else ""
        )
        + f"Transfer value: {tx_info.transferInfo.value} | "
    )


def format_swap_order_info(tx_info: SwapOrderTxInfo) -> str:
    return (
        f"Transaction type: {tx_info.type} | "
        + (
            f"Human description: {tx_info.humanDescription} | "
            if tx_info.humanDescription
            else ""
        )
        + f"Sender: {tx_info.owner} | "
        + f"Recipient: {tx_info.receiver} | "
        + f"Sell token address: {tx_info.sellToken.address} | "
        + f"Sell token symbol: {tx_info.sellToken.symbol} | "
        + f"Buy token address: {tx_info.buyToken.address} | "
        + f"Buy token symbol: {tx_info.buyToken.symbol} | "
        + f"Transfer value: {tx_info.sellAmount} | "
    )


def format_custom_info(tx_info: CustomTxInfo) -> str:
    return (
        (
            f"Human description: {tx_info.humanDescription} | "
            if tx_info.humanDescription
            else ""
        )
        + f"To address: {tx_info.to.value} | "
        + f"Value: {tx_info.value} | "
        + (f"Method name: {tx_info.methodName} | " if tx_info.methodName else "")
        + (
            f"Action count: {tx_info.actionCount} | "
            if tx_info.actionCount is not None
            else ""
        )
        + f"Is cancellation tx: {tx_info.isCancellation} | "
    )


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


def format_balances(safe_address: ChecksumAddress) -> str:
    balances = get_balances_usd(safe_address)
    formatted_items = "\n\n".join(
        f"Token: {item.tokenInfo.name} ({item.tokenInfo.symbol}) | "
        f"Address: {item.tokenInfo.address} | "
        f"Balance: {item.balance} | "
        f"Fiat Balance: {item.fiatBalance} USD | "
        f"Fiat Conversion Rate: {item.fiatConversion} USD | "
        for item in balances.items
    )
    return f"Safe address: {safe_address}\n\nTotal Fiat Balance: {balances.fiatTotal} USD\n\nToken Balances:\n\n{formatted_items}"
