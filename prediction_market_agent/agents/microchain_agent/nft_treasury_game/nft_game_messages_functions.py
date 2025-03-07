import time
from datetime import timedelta

from microchain import Function
from prediction_market_agent_tooling.config import APIKeys as APIKeys_PMAT
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.datetime_utc import DatetimeUTC
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.utils import utcnow
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    AgentRegisterContract,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.tools_nft_treasury_game import (
    purge_all_messages,
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


class SendPaidMessageToAnotherAgent(Function):
    OUTPUT_TEXT = "Message sent to the agent."

    @property
    def description(self) -> str:
        return f"""Use {SendPaidMessageToAnotherAgent.__name__} to send a message to an another agent, given his wallet address.
You need to send a fee of at least {get_message_minimum_value()} xDai to send the message.
However, other agents, same as you, can decide to ignore messages with low fees.
You can also specify a payment to the agent."""

    @property
    def example_args(self) -> list[str | float]:
        return ["0x123", "Hello!", get_message_minimum_value(), 0.0]

    def __call__(self, address: str, message: str, fee: float, payment: float) -> str:
        recipient = Web3.to_checksum_address(address)

        registered_addresses = AgentRegisterContract().get_all_registered_agents()
        if recipient not in registered_addresses:
            return f"Agent with address {address} is not currently receiving messages, please check later."

        keys = MicrochainAgentKeys()
        api_keys = APIKeys_PMAT(BET_FROM_PRIVATE_KEY=keys.bet_from_private_key)
        send_message(
            api_keys=api_keys,
            recipient=recipient,
            message=HexBytes(compress_message(message)),
            # We don't differ between fees and payments, but it helps to accept it as separate arguments,
            # for LLM to understand how to pay for NFT keys or stuff.
            amount_wei=xdai_to_wei(keys.cap_sending_xdai(xdai_type(fee + payment))),
        )
        return self.OUTPUT_TEXT


class GetUnseenMessagesInformation(Function):
    @property
    def description(self) -> str:
        return f"""Use {GetUnseenMessagesInformation.__name__} to get information about the unseen messages that some address have received.
Use this function to decice what message you want to process next.
But also to check out what other agents (their addresses) are going to receive, so you can send a message with higher fee, to attract them to read your message first."""

    @property
    def example_args(self) -> list[str]:
        return ["0x123"]

    def __call__(self, for_public_key: str) -> str:
        messages_statistics = get_unseen_messages_statistics(
            consumer_address=Web3.to_checksum_address(for_public_key)
        )

        return (
            f"Unseen messages statistics for {for_public_key}:\n"
            f"Minimum fee: {messages_statistics.min_fee} xDai\n"
            f"Maximum fee: {messages_statistics.max_fee} xDai\n"
            f"Average fee: {messages_statistics.avg_fee} xDai\n"
            f"Number of unique senders: {messages_statistics.n_unique_senders}\n"
            f"Total number of messages: {messages_statistics.n_messages}"
        )


class ReceiveMessagesAndPayments(Function):
    @property
    def description(self) -> str:
        return f"""Use {ReceiveMessagesAndPayments.__name__} to get N messages from the unseen messages that you have received.
You have to also specify a minimum fee of the message you are willing to read.
Before receiving messages, you can check with {GetUnseenMessagesInformation.__name__} for the up to date statistics of the messages."""

    @property
    def example_args(self) -> list[str | float | int]:
        return [10, 0.0]

    def __call__(self, n: int, minimum_fee: float) -> str:
        keys = MicrochainAgentKeys()
        popped_messages = [
            message
            for _ in range(n)
            if (
                message := pop_message(
                    minimum_fee=xdai_type(minimum_fee),
                    api_keys=APIKeys_PMAT(
                        BET_FROM_PRIVATE_KEY=keys.bet_from_private_key
                    ),
                )
            )
            is not None
        ]
        messages_statistics = get_unseen_messages_statistics(
            consumer_address=keys.bet_from_address
        )
        footer_message = (
            f"\n\n---\n\nYou have another {messages_statistics.n_messages} unseen messages. Use {GetUnseenMessagesInformation.__name__} to get more information."
            if messages_statistics.n_messages
            else "\n\n---\n\nNo more unseen messages."
        )
        return (
            "\n\n".join(
                parse_message_for_agent(message=message) for message in popped_messages
            )
            + footer_message
            if popped_messages
            else "No new messages"
        )


class RemoveAllUnreadMessages(Function):
    @property
    def description(self) -> str:
        return f"Use {RemoveAllUnreadMessages.__name__} to remove all unread messages from your inbox."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        keys = MicrochainAgentKeys()
        purge_all_messages(keys)
        return "All unread messages have been removed."


class SleepUntil(Function):
    """
    This function is special, as it can not be executed in it __call___ method (because it wouldn't be shown in the UI and it wouldn't be continued if the server restarts randomly).
    Therefore, the logic itself is implemented in `execute_calling_of_this_function` and used in the iteration callback of the agent.
    """

    OK_OUTPUT = "Sleeping. Don't forget to check the status of the game and your messages once you wake up!"
    SLEEP_THRESHOLD = timedelta(hours=1)

    def __init__(self) -> None:
        super().__init__()
        self.last_sleep_until: str | None = None

    @property
    def description(self) -> str:
        return f"""Use {SleepUntil.__name__} to sleep until the specified time.
Before using this function, you need to know the exact time you want to sleep until.
You can use this for example to wait for a while before checking for new messages."""

    @property
    def example_args(self) -> list[str]:
        return [f"{utcnow()}", "Waiting for responses."]

    def __call__(self, sleep_until: str, reason: str) -> str:
        sleep_until_datetime = DatetimeUTC.to_datetime_utc(sleep_until)

        if sleep_until_datetime < utcnow():
            output = f"You can not sleep in the past. Current time is {utcnow()}."

        elif (sleep_time := sleep_until_datetime - utcnow()) > self.SLEEP_THRESHOLD:
            # TODO: Testing so the agents won't cut themselves out of the game.
            output = f"You can not sleep for more than {self.SLEEP_THRESHOLD.seconds / 3600} hours. Current time is {utcnow()}."
            # if self.last_sleep_until == sleep_until:
            #     output = self.OK_OUTPUT
            # else:
            #     output = f"You would sleep for {sleep_time}, are you sure you want to do that? Current time is {utcnow()}. To confirm, call this function again with the exact same sleep_until argument."
        else:
            output = self.OK_OUTPUT

        self.last_sleep_until = sleep_until
        return output

    @staticmethod
    def execute_calling_of_this_function(call_code: str) -> None:
        """
        Parse the calling of this function and execute the logic.
        """
        # Handle both positional and keyword arguments
        if "=" in call_code:
            # Keyword arguments
            args = dict(
                item.strip().split("=")
                for item in call_code.split("(")[1][:-1].split(",")
            )
            sleep_until = args.get("sleep_until", "").strip()
            reason = args.get("reason", "").strip()
        else:
            # Positional arguments
            sleep_until = call_code.split(",")[0].split("(")[1].strip()
            reason = call_code.split(",")[1].strip()[:-1]

        while utcnow() < DatetimeUTC.to_datetime_utc(sleep_until):
            logger.info(f"Sleeping until {sleep_until} because {reason}.")
            time.sleep(1.0)


MESSAGES_FUNCTIONS: list[type[Function]] = [
    SendPaidMessageToAnotherAgent,
    ReceiveMessagesAndPayments,
    GetUnseenMessagesInformation,
    SleepUntil,
]
