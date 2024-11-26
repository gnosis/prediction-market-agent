import typing as t

import requests_cache
import tenacity
from autogen import ConversableAgent, register_function
from tavily import TavilyClient

from prediction_market_agent.agents.blockchain_coding_agent.models import (
    SourceCodeContainer,
    SmartContractResponse,
)
from prediction_market_agent.utils import APIKeys


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(1))
def tavily_search(
    query: str,
) -> dict[str, t.Any]:
    """
    Internal minimalistic wrapper around Tavily's search method, that will retry if the call fails.
    """
    tavily = TavilyClient(api_key=(APIKeys()).tavily_api_key.get_secret_value())
    response: dict[str, t.Any] = tavily.search(query=query)
    return response


from eth_typing import ChecksumAddress
from web3 import Web3


# def fetch_read_methods_from_blockscout(contract_address: str) -> t.Any:
#     w3 = OmenConditionalTokenContract().get_web3()
#     if not is_contract(w3, Web3.to_checksum_address(contract_address)):
#         raise ValueError(f"{contract_address=} is not a contract on Gnosis Chain.")
#     read_not_proxy = fetch_read_methods(contract_address)
#     read_proxy = fetch_read_methods_proxy(contract_address)
#     # ToDo - Fetch write methods
#     return read_not_proxy + read_proxy


# def fetch_read_methods(contract_address: str) -> t.Any:
#     url = f"https://gnosis.blockscout.com/api/v2/smart-contracts/{contract_address}/methods-read?is_custom_abi=false"
#     r = requests.get(url)
#     return r.json()


def fetch_source_code_and_abi_from_contract(
    contract_address: str,
) -> SourceCodeContainer:
    url = f"https://gnosis.blockscout.com/api/v2/smart-contracts/{contract_address}"
    session = requests_cache.CachedSession("demo_cache")
    r = session.get(url)
    r.raise_for_status()
    data_parsed = SmartContractResponse.model_validate(r.json())
    source_code = data_parsed.source_code
    abi = data_parsed.abi
    # If proxy, expand abi and source code with implementation data.
    if data_parsed.implementations:
        for proxy in data_parsed.implementations:
            proxy_container = fetch_source_code_and_abi_from_contract(proxy.address)
            abi.extend(proxy_container.abi)
            source_code += "\n" + proxy_container.source_code

    return SourceCodeContainer(source_code=source_code, abi=abi)


def is_contract(web3: Web3, contract_address: ChecksumAddress) -> bool:
    return bool(web3.eth.get_code(contract_address))


# def fetch_read_methods_proxy(contract_address: str) -> t.Any:
#     url = f"https://gnosis.blockscout.com/api/v2/smart-contracts/{contract_address}/methods-read-proxy?is_custom_abi=false"
#     r = requests.get(url)
#     return r.json()


def get_rpc_endpoint() -> str:
    return "https://rpc.gnosischain.com"


def checksum_address(address: str) -> ChecksumAddress:
    from web3 import Web3

    return Web3.to_checksum_address(address)


def fetch_web3_instance() -> Web3:
    from web3 import Web3

    return Web3(Web3.HTTPProvider(get_rpc_endpoint()))


def execute_read_function(
    contract_address: str,
    abi: str,
    function_name: str,
    function_params: list[str],
    w3: Web3,
) -> str:
    """
    Purpose:
        Executes a read function on a smart contract using the specified contract address, ABI, function name, and function parameters.

    Args:
        contract_address (str): The address of the smart contract on which to execute the function.
        abi (str): The ABI (Application Binary Interface) of the smart contract.
        function_name (str): The name of the function to execute on the smart contract.
        function_params (list): A list of parameters to pass to the function.
        w3 (Web3): A Web3 instance.

    Returns:
        Any: The result of calling the specified function on the smart contract.

    """
    from web3 import Web3
    from prediction_market_agent_tooling.tools.contract import abi_field_validator
    from prediction_market_agent_tooling.tools.web3_utils import parse_function_params

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address), abi=abi_field_validator(abi)
    )

    output = contract.functions[function_name](
        *parse_function_params(function_params)
    ).call()  # type: ignore # TODO: Fix Mypy, as this works just OK.
    return output


def register_all_functions(
    caller_agent: ConversableAgent, executor_agent: ConversableAgent
) -> None:
    register_function(
        tavily_search,
        caller=caller_agent,
        executor=executor_agent,
        description="Search the web for the given query",
    )

    register_function(
        fetch_source_code_and_abi_from_contract,
        caller=caller_agent,
        executor=executor_agent,
        description="Function for fetching the ABI and source code from a verified smart contract on Gnosis Chain",
    )

    register_function(
        fetch_web3_instance,
        caller=caller_agent,
        executor=executor_agent,
        description="Returns the Web3 provider instance to be used for interacting with Gnosis Chain when calling functions.",
    )

    register_function(
        execute_read_function,
        caller=caller_agent,
        executor=executor_agent,
        description="Executes a function on a smart contract deployed on the Gnosis Chain and returns the result of the function execution.",
    )

    register_function(
        checksum_address,
        caller=caller_agent,
        executor=executor_agent,
        name="checksum_address",
        description="Extracts the checksummed address of an address.",
    )
