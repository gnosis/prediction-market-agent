import os
import typing as t

import requests
import tqdm
from prediction_market_agent_tooling.gtypes import ABI
from prediction_market_agent_tooling.tools.contract import (
    ContractOnGnosisChain,
    abi_field_validator,
)
from prediction_market_agent_tooling.tools.gnosis_rpc import GNOSIS_RPC_URL
from pydantic import BaseModel
from web3 import Web3

SERVICE_REGISTRY_ADDRESS = Web3.to_checksum_address(
    "0x9338b5153AE39BB89f50468E608eD9d764B755fD"
)


class MechIPFSDescription(BaseModel):
    name: str
    description: str
    code_uri: str
    image: str
    attributes: list[dict[str, t.Any]]


def fetch_service_by_service_id(abi: ABI, service_id: int) -> MechIPFSDescription:
    services = ContractOnGnosisChain(address=SERVICE_REGISTRY_ADDRESS, abi=abi)
    result = services.call(
        "getService", [service_id], web3=Web3(Web3.HTTPProvider(GNOSIS_RPC_URL))
    )
    if int(result[1], 16) == 0:
        raise ValueError("Invalid service")
    address_as_bytes = result[2]
    decoded = int.from_bytes(address_as_bytes, byteorder="big")
    address_for_ipfs_hash = str(hex(decoded))[2:]
    prefix_ipfs_hash = "f01701220"
    ipfs_hash = f"{prefix_ipfs_hash}{address_for_ipfs_hash}"
    # 0x54cf2a8899eabf88f39bbe395b446592e8d3969054ece9b57ce6d197affb2ba0
    # https://gateway.autonolas.tech/ipfs/f0170122054cf2a8899eabf88f39bbe395b446592e8d3969054ece9b57ce6d197affb2ba0
    r = requests.get(f"https://gateway.autonolas.tech/ipfs/{ipfs_hash}")
    # print (r.json())
    return MechIPFSDescription.model_validate(r.json())


def main():
    abi: ABI = abi_field_validator(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "service_registry_abi.json",
        )
    )
    service_descrs = {}
    for service_id in tqdm.trange(1, 10):
        try:
            mech_ipfs_description = fetch_service_by_service_id(
                abi=abi, service_id=service_id
            )
            service_descrs[service_id] = mech_ipfs_description
        except:
            print(f"service {service_id} not available.")

    print(service_descrs)


if __name__ == "__main__":
    main()
