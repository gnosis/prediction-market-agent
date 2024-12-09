from microchain import Function
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.loggers import logger
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

TRANSACTION_MESSAGE_FEE = xdai_type(0.01)


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
Fee for sending the message is {TRANSACTION_MESSAGE_FEE} xDai."""

    @property
    def example_args(self) -> list[str]:
        return ["0x123", "Hello!"]

    def __call__(self, address: str, message: str) -> str:
        keys = MicrochainAgentKeys()
        send_xdai_to(
            web3=ContractOnGnosisChain.get_web3(),
            from_private_key=keys.bet_from_private_key,
            to_address=Web3.to_checksum_address(address),
            value=xdai_to_wei(keys.cap_sending_xdai(TRANSACTION_MESSAGE_FEE)),
            data_text=compress_message(message),
        )
        return "Message sent to the agent."


class ReceiveMessage(Function):
    @staticmethod
    def get_count_unseen_messages() -> int:
        return BlockchainTransactionFetcher().fetch_count_unprocessed_transactions(
            consumer_address=MicrochainAgentKeys().bet_from_address
        )

    @property
    def description(self) -> str:
        count_unseen_messages = self.get_count_unseen_messages()
        return f"Use {ReceiveMessage.__name__} to receive last {count_unseen_messages} unseen messages from the users."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> BlockchainMessage | None:
        keys = MicrochainAgentKeys()
        fetcher = BlockchainTransactionFetcher()
        # Txs were retrieved here, hence they are stored in the DB and won't be fetched again.
        message_to_process = (
            fetcher.fetch_one_unprocessed_blockchain_message_and_store_as_processed(
                keys.public_key
            )
        )
        # ToDo - Fund the treasury with xDai.
        if not message_to_process:
            logger.info("No messages to process.")
        return message_to_process


MESSAGES_FUNCTIONS: list[type[Function]] = [
    BroadcastPublicMessageToHumans,
    SendPaidMessageToAnotherAgent,
    ReceiveMessage,
]
