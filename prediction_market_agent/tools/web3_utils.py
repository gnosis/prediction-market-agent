import os
from typing import Optional, Any
from web3 import Web3
from web3.types import Wei, TxReceipt, TxParams, Nonce
from prediction_market_agent.tools.types import (
    ABI,
    xDai,
    HexAddress,
    PrivateKey,
    ChecksumAddress,
)

ONE_NONCE = Nonce(1)

with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../abis/wxdai.abi.json")
) as f:
    # File content taken from https://gnosisscan.io/address/0xe91d153e0b41518a2ce8dd3d7944fa863463a97d#code.
    WXDAI_ABI = ABI(f.read())


def wei_to_xdai(wei: Wei) -> xDai:
    return xDai(Web3.from_wei(wei, "ether"))


def xdai_to_wei(native: xDai) -> Wei:
    return Web3.to_wei(native, "ether")


def remove_fraction_wei(amount: Wei, fraction: float) -> Wei:
    """Removes the given fraction from the given integer amount and returns the value as an integer."""
    if 0 <= fraction <= 1:
        keep_percentage = 1 - fraction
        return Wei(int(amount * keep_percentage))
    raise ValueError(f"The given fraction {fraction!r} is not in the range [0, 1].")


def check_tx_receipt(receipt: TxReceipt) -> None:
    if receipt["status"] != 1:
        raise ValueError(
            f"Transaction failed with status code {receipt['status']}. Receipt: {receipt}"
        )


def call_function_on_contract(
    web3: Web3,
    contract_address: ChecksumAddress,
    contract_abi: ABI,
    function_name: str,
    function_params: Optional[list[Any]] = None,
) -> Any:
    contract = web3.eth.contract(address=contract_address, abi=contract_abi)
    output = contract.functions[function_name](*(function_params or [])).call()
    return output


def call_function_on_contract_with_tx(
    web3: Web3,
    *,
    contract_address: ChecksumAddress,
    contract_abi: ABI,
    from_address: HexAddress,
    from_private_key: PrivateKey,
    function_name: str,
    function_params: Optional[list[Any]] = None,
    tx_params: Optional[TxParams] = None,
) -> TxReceipt:
    contract = web3.eth.contract(address=contract_address, abi=contract_abi)

    # Fill in required defaults, if not provided.
    tx_params = tx_params or {}
    tx_params["nonce"] = tx_params.get(
        "nonce", web3.eth.get_transaction_count(from_address)
    )
    tx_params["from"] = tx_params.get("from", from_address)

    # Build the transaction.
    tx = contract.functions[function_name](*(function_params or [])).build_transaction(
        tx_params
    )
    # Sign with the private key.
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=from_private_key)
    # Send the signed transaction.
    send_tx = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    # And wait for the receipt.
    receipt_tx = web3.eth.wait_for_transaction_receipt(send_tx)
    return receipt_tx
