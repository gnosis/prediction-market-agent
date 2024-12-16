from prediction_market_agent_tooling.gtypes import ChecksumAddress, HexBytes, xDai
from prediction_market_agent_tooling.tools.contract import ContractOnGnosisChain
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from web3 import Web3


def get_transactions_to_address_trace(
    w3: Web3,
    target_address: ChecksumAddress,
    start_block: int,
    end_block: int | None = None,
    min_value: xDai | None = None,
) -> list[HexBytes]:
    if end_block is None:
        end_block = w3.eth.block_number

    trace_filter_params = {
        "fromBlock": hex(start_block),
        "toBlock": hex(end_block),
        "toAddress": [target_address],
    }
    min_value_wei = xdai_to_wei(min_value) if min_value is not None else 0

    traces = w3.manager.request_blocking("trace_filter", [trace_filter_params])

    transactions = []
    for trace in traces:
        tx_hash = trace.get("transactionHash")
        if tx_hash:
            if min_value_wei is not None:
                action = trace.get("action")
                if action and "value" in action:
                    value = int(action["value"], 16)
                    if value < min_value_wei:
                        continue
                else:
                    # If value is missing in the trace, skip this transaction
                    continue
            transactions.append(HexBytes(tx_hash))

    return transactions


w3 = ContractOnGnosisChain.get_web3()

print(
    get_transactions_to_address_trace(
        w3, "0xd845A24014B3BD96212A21f602a4F16A7dA518A4", 37353329
    )
)
