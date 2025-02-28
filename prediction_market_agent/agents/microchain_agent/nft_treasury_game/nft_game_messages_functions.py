import time
from datetime import datetime

from microchain import Function
from prediction_market_agent_tooling.config import APIKeys as APIKeys_PMAT
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.datetime_utc import DatetimeUTC
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    AgentRegisterContract,
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
However, other agents, same as you, can decide to ignore messages with low fees."""

    @property
    def example_args(self) -> list[str]:
        return ["0x123", "Hello!", f"{get_message_minimum_value()}"]

    def __call__(self, address: str, message: str, fee: float) -> str:
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
            amount_wei=xdai_to_wei(keys.cap_sending_xdai(xdai_type(fee))),
        )
        return self.OUTPUT_TEXT


class GetUnseenMessagesInformation(Function):
    @property
    def description(self) -> str:
        return f"""Use {GetUnseenMessagesInformation.__name__} to get information about the unseen messages that you have received. Use this function to decice what message you want to process next."""

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
        return f"Use {ReceiveMessage.__name__} to get a message from the unseen messages that you have received. You have to also specify a minimum fee of the message you are willing to read. Before receiving messages, you can check with {GetUnseenMessagesInformation.__name__} for the up to date statistics of the messages."

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


class SleepUntil(Function):
    """
    This function is special, as it can not be executed in it __call___ method (because it wouldn't be shown in the UI and it wouldn't be continued if the server restarts randomly).
    Therefore, the logic itself is implemented in `execute_calling_of_this_function` and used in the iteration callback of the agent.
    """

    @property
    def description(self) -> str:
        return f"""Use {SleepUntil.__name__} to wait for given amount of time in seconds and the reason for it. You can use this for example to wait for a while before checking for new messages."""

    @property
    def example_args(self) -> list[str]:
        return ["10", "Waiting for responses."]

    def __call__(self, sleep_until: str, reason: str) -> str:
        sleep_until_datetime = DatetimeUTC.to_datetime_utc(sleep_until)
        return f"Sleeping until {sleep_until_datetime.strftime('%Y-%m-%d %H:%M:%S')} (UTC), because {reason}."

    @staticmethod
    def execute_calling_of_this_function(call_code: str) -> None:
        """
        Parse the calling of this function and execute the logic.
        """
        sleep_until = call_code.split(",")[0].split("(")[1].strip()
        reason = call_code.split(",")[1].strip()[:-1]
        while datetime.utcnow() < DatetimeUTC.to_datetime_utc(sleep_until):
            logger.info(f"Sleeping until {sleep_until} because {reason}.")
            time.sleep(1.0)


MESSAGES_FUNCTIONS: list[type[Function]] = [
    SendPaidMessageToAnotherAgent,
    ReceiveMessage,
    GetUnseenMessagesInformation,
    SleepUntil,
]
