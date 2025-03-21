from microchain import Function
from prediction_market_agent_tooling.gtypes import xDai
from prediction_market_agent_tooling.tools.contract import ContractOnGnosisChain
from prediction_market_agent_tooling.tools.web3_utils import send_xdai_to
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)


class SendXDAI(Function):
    @property
    def description(self) -> str:
        return f"Use {SendXDAI.__class__} to send xDai (stable coin, where 1 xDai equal $1) to a specified blockchain address on Gnosis Chain."

    @property
    def example_args(self) -> list[str]:
        return ["0x9D0260500ba7b068b5b0f4AfA9F8864eBc0B059a", "0.01"]

    def __call__(
        self,
        address: str,
        amount_xdai: float,
    ) -> str:
        keys = MicrochainAgentKeys()
        web3 = ContractOnGnosisChain.get_web3()
        address_checksum = Web3.to_checksum_address(address)
        send_xdai_to(
            web3,
            keys.bet_from_private_key,
            address_checksum,
            keys.cap_sending_xdai(xDai(amount_xdai)).as_xdai_wei,
        )
        return f"Sent {amount_xdai} xDAI to {address_checksum}."


SENDING_FUNCTIONS: list[type[Function]] = [
    SendXDAI,
]
