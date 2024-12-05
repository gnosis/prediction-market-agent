import typing as t

import requests
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.config import RPCConfig
from prediction_market_agent_tooling.gtypes import ABI
from prediction_market_agent_tooling.tools.caches.inmemory_cache import (
    persistent_inmemory_cache,
)
from prediction_market_agent_tooling.tools.contract import (
    abi_field_validator,
)
from prediction_market_agent_tooling.tools.tavily.tavily_models import TavilyResponse
from prediction_market_agent_tooling.tools.tavily.tavily_search import (
    tavily_search as tavily_search_pmat,
)
from prediction_market_agent_tooling.tools.web3_utils import (
    send_function_on_contract_tx,
    call_function_on_contract,
)
from web3 import Web3
from web3.types import TxReceipt

from prediction_market_agent.agents.blockchain_coding_agent.models import (
    SmartContractResponse,
    SourceCodeContainer,
)
from prediction_market_agent.utils import APIKeys


def tavily_search(
    query: str,
) -> TavilyResponse:
    return tavily_search_pmat(query=query)


def checksum_address(address: str) -> ChecksumAddress:
    return Web3.to_checksum_address(address)


def is_contract(web3: Web3, contract_address: ChecksumAddress) -> bool:
    return bool(web3.eth.get_code(contract_address))


@persistent_inmemory_cache
def fetch_source_code_and_abi_from_contract(
    contract_address: str,
) -> SourceCodeContainer:
    url = f"https://gnosis.blockscout.com/api/v2/smart-contracts/{contract_address}"
    r = requests.get(url)
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


def execute_read_function(
    contract_address: str,
    abi: str,
    function_name: str,
    function_params: t.List[t.Any],
) -> t.Any:
    """
    Purpose:
        Executes a read function on a smart contract using the specified contract address, ABI, function name, and function parameters.
        It uses PMAT under-the-hood.

    Args:
        contract_address (str): The address of the smart contract on which to execute the function.
        abi (str): The ABI (Application Binary Interface) of the smart contract.
        function_name (str): The name of the function to execute on the smart contract.
        function_params (list): A list of parameters to pass to the function.

    Returns:
        Any: The result of calling the specified function on the smart contract.

    """
    w3 = Web3(Web3.HTTPProvider(RPCConfig().gnosis_rpc_url))
    return call_function_on_contract(
        web3=w3,
        contract_address=Web3.to_checksum_address(contract_address),
        contract_abi=abi_field_validator(abi),
        function_name=function_name,
        function_params=function_params,
    )


def execute_write_function(
    contract_address: str,
    abi: str,
    function_name: str,
    function_params: t.List[t.Any],
) -> TxReceipt:
    w3 = Web3(Web3.HTTPProvider(RPCConfig().gnosis_rpc_url))
    return send_function_on_contract_tx(
        web3=w3,
        contract_address=Web3.to_checksum_address(contract_address),
        contract_abi=ABI(abi),
        from_private_key=APIKeys().bet_from_private_key,
        function_name=function_name,
        function_params=function_params,
    )
