import contextlib
from typing import Generator

import requests
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.gtypes import xDai
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from tenacity import retry, stop_after_attempt, wait_fixed
from web3 import Web3
from web3.types import RPCEndpoint


@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def set_balance(rpc_url: str, address: str, balance: xDai) -> None:
    balance_wei = xdai_to_wei(balance)
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
    if "error" in response.json():
        raise ValueError(f"error occurred: {response.json()}")
    print(f"Set balance {balance} xDAI for address {address}")


@contextlib.contextmanager
def impersonate_account(
    w3: Web3, account: ChecksumAddress
) -> Generator[None, None, None]:
    w3.provider.make_request(RPCEndpoint("anvil_impersonateAccount"), [account])
    try:
        yield
    finally:
        w3.provider.make_request(
            RPCEndpoint("anvil_stopImpersonatingAccount"), [account]
        )
