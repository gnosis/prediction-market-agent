from microchain import Function
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.tools.contract import ContractOnGnosisChain
from prediction_market_agent_tooling.tools.web3_utils import send_xdai_to, xdai_to_wei
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.agents.microchain_agent.utils import compress_message
from prediction_market_agent.db.blockchain_transaction_fetcher import (
    BlockchainTransactionFetcher,
)
from prediction_market_agent.db.models import BlockchainMessage


class BroadcastPublicMessageToHumans(Function):
    @property
    def description(self) -> str:
        return f"""Use {BroadcastPublicMessageToHumans.__name__} to send a message that humans can see. Use this to communicate with users that send you messages."""

    @property
    def example_args(self) -> list[str]:
        return ["Hello!"]

    def __call__(self, message: str) -> str:
        # TODO: Implement as needed in https://github.com/gnosis/prediction-market-agent/issues/570.
        print(message)
        return f"Message broadcasted to humans."


class SendPaidMessageToAnotherAgent(Function):
    @property
    def description(self) -> str:
        return f"""Use {SendPaidMessageToAnotherAgent.__name__} to send a message to an another agent, given his wallet address. 
You can also specify the fee for the message, which will be deducted from your account. Higher the fee, bigger the chance that agent will read the message and act accordingly."""

    @property
    def example_args(self) -> list[str]:
        return ["0x123", "Hello!", "0.001"]

    def __call__(
        self,
        address: str,
        message: str,
        fee: float,
    ) -> str:
        keys = MicrochainAgentKeys()
        send_xdai_to(
            web3=ContractOnGnosisChain.get_web3(),
            from_private_key=keys.bet_from_private_key,
            to_address=Web3.to_checksum_address(address),
            value=xdai_to_wei(keys.cap_sending_xdai(xdai_type(fee))),
            data_text=compress_message(message),
        )
        return "Message sent to the agent."


class ReceiveMessage(Function):
    @property
    def description(self) -> str:
        # TODO: Add number of unseen messages to the description.
        return f"Use {ReceiveMessage.__name__} to receive last unseen message from the users. Afterwards, process each message sequentially."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> list[BlockchainMessage]:
        keys = MicrochainAgentKeys()
        fetcher = BlockchainTransactionFetcher()
        # Txs were retrieved here, hence they are stored in the DB and won't be fetched again.
        txs_to_process = fetcher.update_unprocessed_transactions_sent_to_address(
            keys.public_key
        )
        return txs_to_process


MESSAGES_FUNCTIONS: list[type[Function]] = [
    BroadcastPublicMessageToHumans,
    SendPaidMessageToAnotherAgent,
    ReceiveMessage,
]
