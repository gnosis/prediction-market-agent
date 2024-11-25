import typing as t
from typing import Callable

from autogen import register_function, ConversableAgent
from pydantic import BaseModel
from tavily import TavilyClient

from prediction_market_agent.utils import APIKeys


class FunctionDefinition(BaseModel):
    function_name: str
    function_callable: Callable


def search_tool(
    query: t.Annotated[str, "The search query"]
) -> t.Annotated[str, "The search results"]:
    tavily = TavilyClient(api_key=APIKeys().tavily_api_key.get_secret_value())
    return tavily.get_search_context(query=query, search_depth="advanced")


from typing import List

import requests_cache
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    OmenConditionalTokenContract,
)
from prediction_market_agent_tooling.tools.contract import ContractOnGnosisChain
from web3 import Web3


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


def register_all_functions(
    caller_agent: ConversableAgent, executor_agent: ConversableAgent
):
    # Register the search tool.
    register_function(
        search_tool,
        caller=caller_agent,
        executor=executor_agent,
        name="search_tool",
        description="Search the web for the given query",
    )

    # Register the calculator function to the two agents.
    register_function(
        fetch_read_methods_from_blockscout,
        caller=caller_agent,  # The assistant agent can suggest calls to the calculator.
        executor=executor_agent,  # The user proxy agent can execute the calculator calls.
        name="fetch_read_methods_from_blockscout",  # By default, the function name is used as the tool name.
        description="Function for fetching the ABI from a verified smart contract on Gnosis Chain",
        # A description of the tool.
    )

    register_function(
        get_rpc_endpoint,
        caller=caller_agent,  # The assistant agent can suggest calls to the calculator.
        executor=executor_agent,  # The user proxy agent can execute the calculator calls.
        name="get_rpc_endpoint",  # By default, the function name is used as the tool name.
        description="Returns the RPC endpoint to be used for interacting with Gnosis Chain when instantiating a provider",
        # A description of the tool.
    )

    register_function(
        execute_read_function,
        caller=caller_agent,
        executor=executor_agent,
        name="execute_read_function",
        description="Executes a function on a smart contract deployed on the Gnosis Chain and returns the result of the function execution.",
    )

    register_function(
        checksum_address,
        caller=caller_agent,
        executor=executor_agent,
        name="checksum_address",
        description="Extracts the checksummed address of an address.",
    )
