import os

from dotenv import load_dotenv
from eth_account import Account
from eth_typing import URI
from prediction_market_agent_tooling.config import RPCConfig
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from safe_eth.eth import EthereumClient
from safe_eth.safe import SafeOperationEnum
from safe_eth.safe.safe import SafeV141
from safe_eth.safe.safe_signature import SafeSignature
from web3 import Web3

load_dotenv()

from eth_abi import encode
from eth_utils import to_hex


def sign_with_mech(mech_address: str, signature_data: bytes) -> str:
    """
    Produce a signature as bytes of the form:
    {bytes32 r = mech address}{bytes32 s = 65 (offset to signature data)}{unpadded uint8 v = 0}{bytes32 signature data length}{bytes signature data}

    :param mech_address: The address of the Mech (Ethereum address as a hex string)
    :param signature_data: The data to be signed (in bytes)
    :return: The resulting signature as a hex string
    """
    offset = 65  # 32 bytes for r + 32 bytes for s data + 1 byte for v

    # Encode r (the Mech address) as bytes32
    r = encode(["address"], [mech_address]).hex()[2:]

    # Encode s (the offset) as bytes32
    s = encode(["uint8"], [offset]).hex()[2:]

    # Encode v (unpadded uint8, which is 0 for contract signatures)
    v = "00"

    # Hex-encode the signature data
    data = to_hex(signature_data)[2:]

    # Encode the length of the signature data as bytes32
    data_length = encode(["uint256"], [len(signature_data)]).hex()[2:]

    # Concatenate the components into the final signature
    return f"0x{r}{s}{v}{data_length}{data}"


if __name__ == "__main__":
    print("start")
    # Create web3, fork at block where agent has 3 keys

    # Try creating safe tx for transferring treasury

    # safe = DeployableAgentNFTGameAbstract.build_treasury_safe()
    # Test safe
    TEST_SAFE = Web3.to_checksum_address("0x8E30a20550343b22BE738Ab62114453f2a1427A7")
    client = EthereumClient(URI(RPCConfig().gnosis_rpc_url))
    safe = SafeV141(TEST_SAFE, client)

    nft_owner_private_key = os.getenv("PRIVATE_KEY_WITH_NFTS", "")
    nft_owner_address = Web3.to_checksum_address(
        Account.from_key(nft_owner_private_key).address
    )  # agent 5 with 3 keys

    safe_tx = safe.build_multisig_tx(
        to=nft_owner_address,
        value=xdai_to_wei(xdai_type(0.5)),
        data=b"",
        operation=SafeOperationEnum.CALL.value,  # from default args
    )
    # ToDo
    mech1_address = Web3.to_checksum_address(
        "0xDDe0780F744B84b505E344931F37cEDEaD8B6163"
    )  # mech 1
    # Example usage
    signature_data = safe_tx.sign(private_key=nft_owner_private_key)
    result = safe_tx.unsign(address=nft_owner_address)
    mech_signature = sign_with_mech(mech1_address, signature_data)
    # Remove owner, only mech has valid signature
    # ToDo - Add mech signature to transaction

    signatures = SafeSignature.parse_signature(
        signatures=mech_signature,
        safe_hash=safe_tx.safe_tx_hash,
        safe_hash_preimage=safe_tx.safe_tx_hash_preimage,
    )
    safe_tx.signatures = SafeSignature.export_signatures(signatures)

    # safe_tx.sign(owner_1_private_key)
    # safe_tx.sign(owner_2_private_key)

    safe_tx.call()  # Check it works
    safe_tx.execute(tx_sender_private_key=nft_owner_private_key)

    # Sign transaction with all 3 mechs
    # Execute transaction
    # Assert that treasury has balance 0.

    print("end")
