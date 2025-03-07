import zlib

from prediction_market_agent_tooling.gtypes import HexBytes
from prediction_market_agent_tooling.tools.web3_utils import wei_to_xdai

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.data_models import (
    MessageContainer,
)


def compress_message(message: str) -> bytes:
    """Used to reduce size of the message before sending it to reduce gas costs."""
    return zlib.compress(message.encode(), level=zlib.Z_BEST_COMPRESSION)


def decompress_message(message: bytes) -> str:
    return zlib.decompress(message).decode()


def unzip_message_else_do_nothing(data_field: str) -> str:
    """We try decompressing the message, else we return the original data field."""
    try:
        return decompress_message(HexBytes(data_field))
    except Exception:
        return data_field


def parse_message_for_agent(message: MessageContainer) -> str:
    parsed = f"Sender's wallet address: {message.sender}"
    if (
        xdai_value := wei_to_xdai(message.value)
    ) > 1.0:  # Random threshold to not consider tiny fees as payments.
        parsed += f"\n\nSender paid you: {xdai_value} xDai"
    parsed += f"\n\nMessage: {unzip_message_else_do_nothing(message.message.hex())}"
    return parsed
