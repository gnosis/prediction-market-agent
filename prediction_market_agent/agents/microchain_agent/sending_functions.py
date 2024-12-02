from microchain import Function
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.tools.contract import ContractOnGnosisChain
from prediction_market_agent_tooling.tools.web3_utils import send_xdai_to, xdai_to_wei
from web3 import Web3

from prediction_market_agent.utils import APIKeys


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
        amount: float,
    ) -> str:
        keys = APIKeys()
        web3 = ContractOnGnosisChain.get_web3()
        address_checksum = Web3.to_checksum_address(address)
        send_xdai_to(
            web3,
            keys.bet_from_private_key,
            address_checksum,
            xdai_to_wei(xdai_type(amount)),
        )
        return f"Sent {amount} xDAI to {address_checksum}."


SENDING_FUNCTIONS: list[type[Function]] = [
    SendXDAI,
]
