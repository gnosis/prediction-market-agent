import requests
from prediction_market_agent_tooling.gtypes import xdai_type, HexBytes
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from web3 import Web3


def get_snapshot(rpc_url: str):
    data = {
        "jsonrpc": "2.0",
        "method": "evm_snapshot",
    }

    response = requests.post(rpc_url, json=data)
    response.raise_for_status()
    print(response.json())


def get_balance(rpc_url: str, address: str, block: int) -> tuple[int, int]:
    data = {
        "jsonrpc": "2.0",
        "method": "eth_getBalance",
        "params": [
            Web3.to_checksum_address(address),
            {"blockNumber": hex(block)},
        ],
        "id": 1,
    }

    response = requests.post(rpc_url, json=data)
    try:
        # response is always 200, hence not raising status
        balance = HexBytes(response.json()["result"])
        return block, balance.as_int()
    except:
        logger.error(
            f"Error occurred while fetching balance for {address} block {block}"
        )
        return -1


def set_balance(rpc_url: str, address: str, balance: int) -> None:
    balance_wei = xdai_to_wei(xdai_type(balance))
    data = {
        "jsonrpc": "2.0",
        "method": "anvil_setBalance",
        "params": [
            Web3.to_checksum_address(address),
            str(balance_wei),
        ],
        "id": 1,
    }

    response = requests.post(rpc_url, json=data)
    response.raise_for_status()
    print(f"Set balance {balance} xDAI for address {address}")
