import os
import typing as t

import pytest
import requests
from ape import accounts as ape_accounts
from ape.managers import ChainManager
from ape_test import TestAccount
from dotenv import load_dotenv
from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_typing import URI, ChecksumAddress
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import (
    ABI,
    HexAddress,
    PrivateKey,
    private_key_type,
    xDai,
)
from prediction_market_agent_tooling.tools.web3_utils import prepare_tx, send_xdai_to
from safe_eth.eth import EthereumClient
from web3 import Web3
from web3.types import TxReceipt


def set_env_rpc_url(url: str) -> None:
    os.environ["GNOSIS_RPC_URL"] = url


@pytest.fixture(autouse=True, scope="session")
def load_env() -> None:
    load_dotenv()


@pytest.fixture(scope="session")
def local_web3(
    load_env: None, chain: ChainManager, accounts: list[TestAccount]
) -> t.Generator[Web3, None, None]:
    print("entering fixture local_web3")

    if (tenderly_fork_rpc := APIKeys().TENDERLY_FORK_RPC) is not None:
        print("using tenderly rpc")
        w3 = Web3(Web3.HTTPProvider(tenderly_fork_rpc))
        print("funding test accounts on tenderly")
        fund_account_on_tenderly(
            tenderly_fork_rpc, [a.address for a in accounts], xDai(1000)
        )
        set_env_rpc_url(tenderly_fork_rpc)
        yield w3
    else:
        print("using foundry")
        with chain.network_manager.parse_network_choice(
            "gnosis:mainnet_fork:foundry"
        ) as provider:
            set_env_rpc_url(provider.http_uri)
            w3 = Web3(Web3.HTTPProvider(provider.http_uri))
            yield w3

    print("exiting fixture local_web3")


@pytest.fixture(scope="module")
def local_ethereum_client(local_web3: Web3) -> EthereumClient:
    return EthereumClient(URI(local_web3.provider.endpoint_uri))  # type: ignore


@pytest.fixture(scope="session")
def many_test_keys(local_web3: Web3) -> list[APIKeys]:
    # Create fresh EOA accounts and fund them from one of the ape's testing accounts.
    fund_from_account = ape_accounts.test_accounts[0]

    accounts = [
        create_and_fund_random_account(
            local_web3, private_key_type(fund_from_account.private_key)
        )
        for _ in range(10)
    ]

    return [
        APIKeys(
            BET_FROM_PRIVATE_KEY=private_key_type(account.key.hex()), SAFE_ADDRESS=None
        )
        for account in accounts
    ]


@pytest.fixture(scope="session")
def test_keys(many_test_keys: list[APIKeys]) -> APIKeys:
    return many_test_keys[0]


def fund_account_on_tenderly(
    fork_rpc: str, addresses: list[HexAddress], balance: xDai
) -> None:
    payload = {
        "jsonrpc": "2.0",
        "method": "tenderly_setBalance",
        "params": [addresses, f"0x{balance.as_xdai_wei.value:X}"],
    }
    response = requests.post(fork_rpc, json=payload)
    response.raise_for_status()


def create_and_fund_random_account(
    web3: Web3, private_key: PrivateKey, deposit_amount: xDai = xDai(10)
) -> LocalAccount:
    fresh_account: LocalAccount = Account.create()
    send_xdai_to(
        web3=web3,
        from_private_key=private_key,
        to_address=fresh_account.address,
        value=deposit_amount.as_xdai_wei,
    )
    return fresh_account


def execute_tx_from_impersonated_account(
    web3: Web3,
    impersonated_account: LocalAccount,
    contract_address: ChecksumAddress,
    contract_abi: ABI,
    function_name: str,
    function_params: t.Optional[list[t.Any] | dict[str, t.Any]] = None,
) -> TxReceipt:
    with ape_accounts.use_sender(impersonated_account.address) as s:
        tx_params = prepare_tx(
            web3=web3,
            contract_address=contract_address,
            contract_abi=contract_abi,
            from_address=s.address,
            function_name=function_name,
            function_params=function_params,
        )

        send_tx = web3.eth.send_transaction(tx_params)
        # And wait for the receipt.
        tx_receipt = web3.eth.wait_for_transaction_receipt(send_tx)
        return tx_receipt
