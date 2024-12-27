from microchain import Function
from prediction_market_agent_tooling.config import APIKeys as APIKeys_PMAT
from prediction_market_agent_tooling.gtypes import wei_type, xdai_type
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.contract import ContractOnGnosisChain
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.web3_utils import send_xdai_to, xdai_to_wei

from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    TREASURY_SAFE_ADDRESS,
)
from prediction_market_agent.db.agent_communication import (
    fetch_count_unprocessed_transactions,
    pop_message,
    send_message,
)
from prediction_market_agent.tools.message_utils import (
    compress_message,
    parse_message_for_agent,
)


class BroadcastPublicMessageToHumans(Function):
    OUTPUT_TEXT = "Message broadcasted to humans."

    @property
    def description(self) -> str:
        return f"""Use {BroadcastPublicMessageToHumans.__name__} to send a message that humans can see. Use this to communicate with users that send you messages."""

    @property
    def example_args(self) -> list[str]:
        return ["Hello!"]

    def __call__(self, message: str) -> str:
        return self.OUTPUT_TEXT


class SendPaidMessageToAnotherAgent(Function):
    OUTPUT_TEXT = "Message sent to the agent."

    @property
    def description(self) -> str:
        return f"""Use {SendPaidMessageToAnotherAgent.__name__} to send a message to an another agent, given his wallet address.
You need to send a fee of at least {MicrochainAgentKeys().RECEIVER_MINIMUM_AMOUNT} xDai for other agent to read the message."""

    @property
    def example_args(self) -> list[str]:
        return ["0x123", "Hello!", f"{MicrochainAgentKeys().RECEIVER_MINIMUM_AMOUNT}"]

    def __call__(self, address: str, message: str, fee: float) -> str:
        keys = MicrochainAgentKeys()
        api_keys = APIKeys_PMAT(BET_FROM_PRIVATE_KEY=keys.bet_from_private_key)
        send_message(
            api_keys=api_keys,
            message=HexBytes(compress_message(message)),
            amount_wei=xdai_to_wei(keys.cap_sending_xdai(xdai_type(fee))),
        )
        return self.OUTPUT_TEXT


class ReceiveMessage(Function):
    # Percentage of message value that goes to the treasury.
    TREASURY_ACCUMULATION_PERCENTAGE = 0.7

    @staticmethod
    def get_count_unseen_messages() -> int:
        keys = MicrochainAgentKeys()
        return fetch_count_unprocessed_transactions(
            consumer_address=keys.bet_from_address
        )

    @property
    def description(self) -> str:
        count_unseen_messages = self.get_count_unseen_messages()
        return f"Use {ReceiveMessage.__name__} to receive last unseen message from the users or other agents. Currently, you have {count_unseen_messages} unseen messages."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        keys = MicrochainAgentKeys()

        count_unseen_messages = self.get_count_unseen_messages()

        if count_unseen_messages == 0:
            logger.info("No messages to process.")
            return "No new messages"

        popped_message = pop_message(
            api_keys=APIKeys_PMAT(BET_FROM_PRIVATE_KEY=keys.bet_from_private_key),
        )

        # Accumulate a percentage of the message value in the treasury.
        tx_receipt = send_xdai_to(
            web3=ContractOnGnosisChain.get_web3(),
            from_private_key=keys.bet_from_private_key,
            to_address=TREASURY_SAFE_ADDRESS,
            value=wei_type(
                self.TREASURY_ACCUMULATION_PERCENTAGE * popped_message.value
            ),
        )
        logger.info(
            f"Funded the treasury with xDai, tx_hash: {HexBytes(tx_receipt['transactionHash']).hex()}"
        )
        return parse_message_for_agent(message=popped_message)


MESSAGES_FUNCTIONS: list[type[Function]] = [
    BroadcastPublicMessageToHumans,
    SendPaidMessageToAnotherAgent,
    ReceiveMessage,
]
