import contextlib
from typing import Generator

import requests
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.gtypes import xDai
from prediction_market_agent_tooling.loggers import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from web3 import Web3
from web3.types import RPCEndpoint


@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(5),
    after=lambda x: logger.debug(
        f"set_balance failed, {x.attempt_number=}, {x.upcoming_sleep=}."
    ),
)
def set_balance(rpc_url: str, address: str, balance: xDai) -> None:
    balance_wei = balance.as_xdai_wei
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
