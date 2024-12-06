from prediction_market_agent.agents.microchain_agent.utils import (
    compress_message,
    decompress_message,
)


def test_message_compression() -> None:
    message = "Hello!"
    encoded = compress_message(message)
    assert message == decompress_message(encoded)
