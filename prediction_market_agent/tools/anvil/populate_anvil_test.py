from eth_account import Account
from prediction_market_agent_tooling.config import APIKeys as APIKeys_PMAT
from prediction_market_agent_tooling.gtypes import (
    private_key_type,
    xdai_type,
)
from prediction_market_agent_tooling.tools.contract import (
    ContractOwnableERC721OnGnosisChain,
)
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.web3_utils import send_xdai_to, xdai_to_wei
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    NFT_TOKEN_FACTORY,
)
from prediction_market_agent.db.agent_communication import send_message
from prediction_market_agent.tools.message_utils import compress_message

if __name__ == "__main__":
    agent1 = Account.from_key(
        "0x57d188552ae3933dfb4bbbff7f9e6a31c396a2b0d74a6c464e6035396bee4626"
    )
    agent2 = Account.from_key(
        "0x97413e5bf20d192c74c7e81abc910fc9a19c3f37382168db370572ca63117a95"
    )
    agent3 = Account.from_key(
        "0x8ee04bf552704c9fc2e95007789ff61afe886f8a84355ad6cee2d1bdf96ab903"
    )
    agent5 = Account.from_key(
        "0x44cf73596e3686621803194f61d5e97fdb9354d10ea8caf0fa93a2b24a89ff5c"
    )
    #     - [ ]  Send money from agent 1 to agent 2
    RPC_URL = "http://localhost:8545"
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    api_keys = APIKeys_PMAT(BET_FROM_PRIVATE_KEY=private_key_type(agent5.key.hex()))
    ###############################################################################################

    # - [ ]  NFT transfer from agent 1 to agent 3
    contract = ContractOwnableERC721OnGnosisChain(
        address=Web3.to_checksum_address(NFT_TOKEN_FACTORY)
    )
    # 0xcac3ebc867d11a41e5bbed7493476ebea7cd76a0212e47f1e02f3660a2d14b9e
    contract.safeTransferFrom(
        api_keys=api_keys,
        from_address=agent5.address,
        to_address=agent3.address,
        tokenId=0,
    )
    # 0xbe852179c16b1c2582b046bb66b03a702c5d8a5221f32458aaa5d31e6bf14943
    send_xdai_to(
        web3=w3,
        from_private_key=private_key_type(agent2.key.hex()),
        to_address=agent3.address,
        value=xdai_to_wei(xdai_type(0.1)),
    )
    # - [ ]  Send money from agent 2 to agent 5
    # 0x1edd58db6c2131cee78284eed2492fe91ef9ab60a0d2b651d67db70c2ebfc4b7
    send_xdai_to(
        web3=w3,
        from_private_key=private_key_type(agent2.key.hex()),
        to_address=agent5.address,
        value=xdai_to_wei(xdai_type(0.1)),
    )
    # - [ ]  Send message from agent 1 to agent 2 (no zlib compress)
    # 0x8f078a08836164edee84741bd711e49e891c6d0b5514b838af7c84911839ec2b
    send_message(
        api_keys=api_keys,
        recipient=agent1.address,
        message=HexBytes("hello agent1, this is agent 5".encode()),
        amount_wei=xdai_to_wei(xdai_type(0.1)),
    )
    # - [ ]  Send message from agent 5 to agent 2 (zlib compress)
    compressed_message = compress_message("hello agent1, this is agent 5")
    # 0x03a04e250ab4aa113356b618d108eed064243f69f7d9e38d43763ec15f7d70fc
    send_message(
        api_keys=api_keys,
        recipient=agent1.address,
        message=HexBytes(compressed_message),
        amount_wei=xdai_to_wei(xdai_type(0.1)),
    )
