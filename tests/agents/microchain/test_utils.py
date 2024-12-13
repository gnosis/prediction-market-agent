from prediction_market_agent.agents.microchain_agent.prompts import FunctionsConfig


def test_combine_functions_config() -> None:
    a = FunctionsConfig(include_agent_functions=True)
    b = FunctionsConfig(include_trading_functions=True)
    assert a.combine(b) == FunctionsConfig(
        include_agent_functions=True, include_trading_functions=True
    )
