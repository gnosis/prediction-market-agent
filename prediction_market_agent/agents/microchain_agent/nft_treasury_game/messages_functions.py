import time

from microchain import Function
from prediction_market_agent_tooling.config import APIKeys as APIKeys_PMAT
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.db.agent_communication import (
    get_message_minimum_value,
    get_unseen_messages_statistics,
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
You need to send a fee of at least {get_message_minimum_value()} xDai to send the message.
However, other agents, same as you, can decide to ignore messages with low fees."""

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


class GetUnseenMessagesInformation(Function):
    @property
    def description(self) -> str:
        return f"""Use {GetUnseenMessagesInformation.__name__} to get information about the unseen messages that you have received. Use this message to decice what message you want to process next."""

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        keys = MicrochainAgentKeys()
        messages_statistics = get_unseen_messages_statistics(
            consumer_address=keys.bet_from_address
        )

        return (
            f"Minimum fee: {messages_statistics.min_fee} xDai\n"
            f"Maximum fee: {messages_statistics.max_fee} xDai\n"
            f"Average fee: {messages_statistics.avg_fee} xDai\n"
            f"Number of unique senders: {messages_statistics.n_unique_senders}\n"
            f"Total number of messages: {messages_statistics.n_messages}"
        )


class ReceiveMessage(Function):
    @property
    def description(self) -> str:
        return f"Use {ReceiveMessage.__name__} to get a message from the unseen messages that you have received. You have to also specify a minimum fee of the message you are willing to read. Before receiving messages, always check with {GetUnseenMessagesInformation.__name__} for the up to date statistics, so you can decide which message to read next."

    @property
    def example_args(self) -> list[str]:
        return ["0.0"]

    def __call__(self, minimum_fee: float) -> str:
        keys = MicrochainAgentKeys()
        popped_message = pop_message(
            minimum_fee=xdai_type(minimum_fee),
            api_keys=APIKeys_PMAT(BET_FROM_PRIVATE_KEY=keys.bet_from_private_key),
        )
        return (
            parse_message_for_agent(message=popped_message)
            if popped_message
            else "No new messages"
        )


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
