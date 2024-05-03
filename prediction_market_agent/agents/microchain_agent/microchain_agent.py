import typer
from functions import MARKET_FUNCTIONS, MISC_FUNCTIONS
from microchain import LLM, Agent, Engine, OpenAIChatGenerator
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.omen_functions import (
    OMEN_FUNCTIONS,
)
from prediction_market_agent.utils import APIKeys

SYSTEM_PROMPT = """Act as a agent to maximise your profit.

You can use the following functions:

{engine_help}

Only output valid Python function calls.
"""


def get_agent(
    market_type: MarketType,
    model: str,
    api_base: str = "https://api.openai.com/v1",
) -> Agent:
    engine = Engine()
    engine.register(Reasoning())
    engine.register(Stop())
    for function in MISC_FUNCTIONS:
        engine.register(function())
    for function in MARKET_FUNCTIONS:
        engine.register(function(market_type=market_type))
    for function in OMEN_FUNCTIONS:
        engine.register(function())
    generator = OpenAIChatGenerator(
        model=model,
        api_key=APIKeys().openai_api_key.get_secret_value(),
        api_base=api_base,
        temperature=0.7,
    )
    agent = Agent(llm=LLM(generator=generator), engine=engine)
    agent.prompt = SYSTEM_PROMPT.format(engine_help=engine.help)
    agent.bootstrap = [
        'Reasoning("I need to reason step by step. Start by assessing my '
        'current position and balance.")'
    ]
    return agent


def main(
    market_type: MarketType = MarketType.OMEN,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4-turbo-preview",
    iterations: int = 10,
    seed_prompt: str | None = None,
) -> None:
    agent = get_agent(
        market_type=market_type,
        api_base=api_base,
        model=model,
    )
    if seed_prompt:
        agent.bootstrap = [f'Reasoning("{seed_prompt}")']
    agent.run(iterations=iterations)
    # generator.print_usage() # Waiting for microchain release


if __name__ == "__main__":
    typer.run(main)
