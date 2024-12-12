from prediction_market_agent.agents.microchain_agent.prompts import FunctionsConfig
from prediction_market_agent.agents.microchain_agent.utils import (
    compress_message,
    decompress_message,
)


def test_message_compression() -> None:
    message = "Hello!"
    encoded = compress_message(message)
    assert message == decompress_message(encoded)


def test_combine_functions_config() -> None:
    a = FunctionsConfig(include_agent_functions=True)
    b = FunctionsConfig(include_trading_functions=True)
    assert a.combine(b) == FunctionsConfig(
        include_agent_functions=True, include_trading_functions=True
    )
