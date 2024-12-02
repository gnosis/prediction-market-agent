import typing as t

import requests_cache
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.gtypes import PrivateKey, ABI
from prediction_market_agent_tooling.tools.contract import abi_field_validator
from prediction_market_agent_tooling.tools.tavily.tavily_models import TavilyResponse
from prediction_market_agent_tooling.tools.tavily.tavily_search import (
    tavily_search as tavily_search_pmat,
)
from prediction_market_agent_tooling.tools.web3_utils import (
    parse_function_params,
    send_function_on_contract_tx,
)
from web3 import Web3
from web3.types import TxReceipt

from prediction_market_agent.agents.blockchain_coding_agent.models import (
    SourceCodeContainer,
    SmartContractResponse,
)
from prediction_market_agent.utils import APIKeys


def tavily_search(
    query: str,
) -> TavilyResponse:
    return tavily_search_pmat(query=query)


def get_rpc_endpoint() -> str:
    # return "https://rpc.gnosischain.com"
    return "http://localhost:8545"


def checksum_address(address: str) -> ChecksumAddress:
    return Web3.to_checksum_address(address)


def is_contract(web3: Web3, contract_address: ChecksumAddress) -> bool:
    return bool(web3.eth.get_code(contract_address))


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


def execute_read_function(
    contract_address: str,
    abi: str,
    function_name: str,
    function_params: t.List[t.Any],
) -> str:
    """
    Purpose:
        Executes a read function on a smart contract using the specified contract address, ABI, function name, and function parameters.

    Args:
        contract_address (str): The address of the smart contract on which to execute the function.
        abi (str): The ABI (Application Binary Interface) of the smart contract.
        function_name (str): The name of the function to execute on the smart contract.
        function_params (list): A list of parameters to pass to the function.

    Returns:
        Any: The result of calling the specified function on the smart contract.

    """
    rpc_endpoint = get_rpc_endpoint()
    w3 = Web3(Web3.HTTPProvider(rpc_endpoint))
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address), abi=abi_field_validator(abi)
    )

    output = contract.functions[function_name](
        *parse_function_params(function_params)
    ).call()  # type: ignore # TODO: Fix Mypy, as this works just OK.
    return output


##### Write


def get_private_key() -> PrivateKey:
    return APIKeys().bet_from_private_key


def execute_write_function(
    contract_address: str,
    abi: str,
    function_name: str,
    function_params: t.List[t.Any],
) -> TxReceipt:
    # ToDo - Do not use PMAT if possible.
    w3 = Web3(Web3.HTTPProvider(get_rpc_endpoint()))
    private_key = get_private_key()
    return send_function_on_contract_tx(
        web3=w3,
        contract_address=Web3.to_checksum_address(contract_address),
        contract_abi=ABI(abi),
        from_private_key=private_key,
        function_name=function_name,
        function_params=function_params,
    )
