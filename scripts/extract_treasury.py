import os

from dotenv import load_dotenv
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from safe_eth.safe import SafeOperationEnum
from web3 import Web3

load_dotenv()

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DeployableAgentNFTGameAbstract,
)

if __name__ == "__main__":
    print("start")
    # Create web3, fork at block where agent has 3 keys

    # Try creating safe tx for transferring treasury

    safe = DeployableAgentNFTGameAbstract.build_treasury_safe()
    nft_owner_address = Web3.to_checksum_address(
        "0x84690A78d74e90608fc3e73cA79A06ee4F261A06"
    )  # agent 5 with 3 keys
    nft_owner_private_key = os.getenv("PRIVATE_KEY_WITH_NFTS", "")

    safe_tx = safe.build_multisig_tx(
        to=nft_owner_address,
        value=xdai_to_wei(xdai_type(5)),
        data=b"",
        operation=SafeOperationEnum.CALL.value,  # from default args
    )
    # ToDo
    # safe_tx.sign(owner_1_private_key)
    # safe_tx.sign(owner_2_private_key)

    safe_tx.call()  # Check it works
    safe_tx.execute(tx_sender_private_key=nft_owner_private_key)

    # Sign transaction with all 3 mechs
    # Execute transaction
    # Assert that treasury has balance 0.
    print("end")
