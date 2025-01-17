import requests
import typer
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    TREASURY_ADDRESS,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DEPLOYED_NFT_AGENTS,
)


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


def main(
    rpc_url: str, new_balance_agents_xdai: int, new_balance_treasury_xdai: int
) -> None:
    for agent in DEPLOYED_NFT_AGENTS:
        set_balance(
            rpc_url=rpc_url,
            address=agent.wallet_address,
            balance=new_balance_agents_xdai,
        )

    set_balance(
        rpc_url=rpc_url,
        address=TREASURY_ADDRESS,
        balance=new_balance_treasury_xdai,
    )


if __name__ == "__main__":
    typer.run(main)
