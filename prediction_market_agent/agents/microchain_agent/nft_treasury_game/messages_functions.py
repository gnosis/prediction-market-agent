import time

from microchain import Function
from prediction_market_agent_tooling.config import APIKeys as APIKeys_PMAT
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.db.agent_communication import (
    fetch_count_unprocessed_transactions,
    get_message_minimum_value,
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
You need to send a fee of at least {get_message_minimum_value()} xDai for other agent to read the message."""

    @property
    def example_args(self) -> list[str]:
        return ["0x123", "Hello!", f"{get_message_minimum_value()}"]

    def __call__(self, address: str, message: str, fee: float) -> str:
        keys = MicrochainAgentKeys()
        api_keys = APIKeys_PMAT(BET_FROM_PRIVATE_KEY=keys.bet_from_private_key)
        send_message(
            api_keys=api_keys,
            recipient=Web3.to_checksum_address(address),
            message=HexBytes(compress_message(message)),
            amount_wei=xdai_to_wei(keys.cap_sending_xdai(xdai_type(fee))),
        )
        return self.OUTPUT_TEXT


class ReceiveMessage(Function):
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

        return parse_message_for_agent(message=popped_message)


class Wait(Function):
    @property
    def description(self) -> str:
        return f"""Use {Wait.__name__} to wait for given amount of time in seconds and the reason for it. You can use this for example to wait for a while before checking for new messages."""

    @property
    def example_args(self) -> list[str]:
        return ["10", "Waiting for responses."]

    def __call__(self, wait: int, reason: str) -> str:
        time.sleep(wait)
        return f"Waited for {wait} seconds to {reason}."


class GameRoundEnd(Function):
    GAME_ROUND_END_OUTPUT = "Agent decided to stop playing."

    @property
    def description(self) -> str:
        return f"""Use {GameRoundEnd.__name__} to indicate that you are done playing this game."""

    @property
    def example_args(self) -> list[str]:
        return ["The game has finished and I did all necessary steps."]

    def __call__(self, reasoning: str) -> str:
        return self.GAME_ROUND_END_OUTPUT


MESSAGES_FUNCTIONS: list[type[Function]] = [
    BroadcastPublicMessageToHumans,
    SendPaidMessageToAnotherAgent,
    ReceiveMessage,
    Wait,
    GameRoundEnd,
]
