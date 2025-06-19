from web3 import Web3
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.config import RPCConfig
from prediction_market_agent_tooling.tools.contract import contract_implements_function


def is_erc20_contract(address: ChecksumAddress, web3: Web3 | None = None) -> bool:
    """
    Checks if the given address is an ERC20-compatible contract.

    It estimates it by looking if it implements symbol, name, decimals, totalSupply, balanceOf.
    """
    web3 = web3 or RPCConfig().get_web3()
    return (
        contract_implements_function(address, "symbol", web3)
        and contract_implements_function(address, "name", web3)
        and contract_implements_function(address, "totalSupply", web3)
        and contract_implements_function(
            address, "balanceOf", web3, function_arg_types=["address"]
        )
    )


def is_nft_contract(address: ChecksumAddress, web3: Web3 | None = None) -> bool:
    """
    Checks if the given address is an NFT-compatible contract (ERC721 or ERC1155).

    For ERC721, checks for: ownerOf, balanceOf, transferFrom.
    For ERC1155, checks for: balanceOf, safeTransferFrom.

    Returns True if either ERC721 or ERC1155 interface is detected.
    """
    web3 = web3 or RPCConfig().get_web3()
    is_erc721 = (
        contract_implements_function(
            address, "ownerOf", web3, function_arg_types=["uint256"]
        )
        and contract_implements_function(
            address, "balanceOf", web3, function_arg_types=["address"]
        )
        and contract_implements_function(
            address,
            "transferFrom",
            web3,
            function_arg_types=["address", "address", "uint256"],
        )
    )
    if is_erc721:
        return True
    is_erc1155 = contract_implements_function(
        address, "balanceOf", web3, function_arg_types=["address", "uint256"]
    ) and contract_implements_function(
        address,
        "safeTransferFrom",
        web3,
        function_arg_types=["address", "address", "uint256", "uint256", "bytes"],
    )
    if is_erc1155:
        return True
    return False
