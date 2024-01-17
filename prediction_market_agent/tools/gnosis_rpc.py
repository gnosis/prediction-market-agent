import requests
from typing import NewType

GNOSIS_RPC_URL = "https://rpc.gnosischain.com/"

# Useful tool https://gnosisscan.io/unitconverter
Wei = NewType("Wei", int)


def get_balance(address: str) -> Wei:
    response = requests.post(
        GNOSIS_RPC_URL,
        json={
            "jsonrpc": "2.0",
            "method": "eth_getBalance",
            "params": [address, "latest"],
            "id": 1,
        },
        headers={"content-type": "application/json"},
    ).json()
    balance = Wei(int(response["result"], 16))  # Convert hex value to int.
    return balance


if __name__ == "__main__":
    print(get_balance("0xf3318C420e5e30C12786C4001D600e9EE1A7eBb1"))
