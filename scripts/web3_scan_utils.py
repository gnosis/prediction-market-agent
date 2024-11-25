from typing import List

import requests_cache
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    OmenConditionalTokenContract,
)
from prediction_market_agent_tooling.tools.contract import ContractOnGnosisChain
from web3 import Web3


def fetch_contract_source_code(contract_address: str) -> str:
    url = f"https://gnosis.blockscout.com/api/v2/smart-contracts/{contract_address}/source-code"
    url = f"
  'https://gnosis.blockscout.com/api/v2/smart-contracts/0xf8D1677c8a0c961938bf2f9aDc3F3CFDA759A9d9"\
  -H 'accept: application/json'


def fetch_read_methods_from_blockscout(contract_address: str) -> str:
    w3 = OmenConditionalTokenContract().get_web3()
    if not is_contract(w3, Web3.to_checksum_address(contract_address)):
        raise ValueError(f"{contract_address=} is not a contract on Gnosis Chain.")
    read_not_proxy = fetch_read_methods(contract_address)
    read_proxy = fetch_read_methods_proxy(contract_address)
    # ToDo - Fetch write methods
    return read_not_proxy + read_proxy


def fetch_read_methods(contract_address: str) -> str:
    url = f"https://gnosis.blockscout.com/api/v2/smart-contracts/{contract_address}/methods-read?is_custom_abi=false"
    session = requests_cache.CachedSession("demo_cache")
    r = session.get(url)
    return r.json()


def is_contract(web3: Web3, contract_address: ChecksumAddress) -> bool:
    return bool(web3.eth.get_code(contract_address))


def fetch_read_methods_proxy(contract_address: str) -> str:
    url = f"https://gnosis.blockscout.com/api/v2/smart-contracts/{contract_address}/methods-read-proxy?is_custom_abi=false"
    session = requests_cache.CachedSession("demo_cache")
    r = session.get(url)
    return r.json()


def get_rpc_endpoint() -> str:
    return "https://rpc.gnosischain.com"


def checksum_address(address: str) -> ChecksumAddress:
    from web3 import Web3

    return Web3.to_checksum_address(address)


def execute_read_function(
    contract_address: str,
    abi: str,
    function_name: str,
    function_parameters: List[str] = [],
) -> str:
    """
    Purpose:
        Executes a read function on a smart contract using the specified contract address, ABI, function name, and function parameters.

    Args:
        contract_address (str): The address of the smart contract on which to execute the function.
        abi (str): The ABI (Application Binary Interface) of the smart contract.
        function_name (str): The name of the function to execute on the smart contract.
        function_parameters (list): A list of parameters to pass to the function.

    Returns:
        Any: The result of calling the specified function on the smart contract.

    """
    from web3 import Web3
    from prediction_market_agent_tooling.tools.contract import abi_field_validator

    c = ContractOnGnosisChain(
        abi=abi_field_validator(abi), address=Web3.to_checksum_address(contract_address)
    )
    return c.call(function_name, function_parameters)


# def fetch_abi_from_verified_contract_on_gnosis_chain(contract_address: str) -> str:
#     session = requests_cache.CachedSession("demo_cache")
#     keys = APIKeys()
#     checksummed_contract_address = Web3.to_checksum_address(contract_address)
#     r = session.get(
#         f"""https://api.gnosisscan.io/api?module=contract
#    &action=getabi&address={checksummed_contract_address}&apikey={keys.gnosisscan_api_key.get_secret_value()}"""
#     )
#     # Invalid contracts also returned with status code 200.
#     message = r.json()["message"]
#     if message != "OK":
#         raise ValueError(
#             f"Could not fetch ABI from contract {checksummed_contract_address}. Error: {json.dumps(r.json())}"
#         )
#     return r.json()["result"]
