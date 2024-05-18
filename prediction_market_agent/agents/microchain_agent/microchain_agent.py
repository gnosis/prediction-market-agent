import typer
from microchain import LLM, Agent, Engine, OpenAIChatGenerator
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.functions import (
    MARKET_FUNCTIONS,
    MISC_FUNCTIONS,
    RememberPastLearnings,
)
from prediction_market_agent.agents.microchain_agent.memory import LongTermMemory
from prediction_market_agent.agents.microchain_agent.omen_functions import (
    OMEN_FUNCTIONS,
)
from prediction_market_agent.utils import APIKeys

SYSTEM_PROMPT = """
Act as a trader agent in prediction markets to maximise your profit.

Research markets, buy tokens you consider undervalued, and sell tokens that you
hold and consider overvalued.

You can use the following functions:

{engine_help}

Only output valid Python function calls.
Make 'Reasoning' calls frequently - at least every other call.
"""


def build_agent(
    market_type: MarketType,
    model: str,
    api_base: str = "https://api.openai.com/v1",
    long_term_memory: LongTermMemory | None = None,
    allow_stop: bool = True,
) -> Agent:
    engine = Engine()
    engine.register(Reasoning())
    if allow_stop:
        engine.register(Stop())
    for function in MISC_FUNCTIONS:
        engine.register(function())
    for function in MARKET_FUNCTIONS:
        engine.register(function(market_type=market_type))
    for function in OMEN_FUNCTIONS:
        engine.register(function())

    if long_term_memory:
        engine.register(
            RememberPastLearnings(long_term_memory=long_term_memory, model=model)
        )

    generator = OpenAIChatGenerator(
        model=model,
        api_key=APIKeys().openai_api_key.get_secret_value(),
        api_base=api_base,
        temperature=0.7,
    )
    agent = Agent(llm=LLM(generator=generator), engine=engine)
    agent.max_tries = 3
    agent.prompt = SYSTEM_PROMPT.format(engine_help=engine.help)
    agent.bootstrap = [
        'Reasoning("I need to reason step by step. Start by assessing my '
        "current positions and balance. Do I have any positions in the markets "
        "returned from GetMarkets? Consider selling overvalued tokens AND "
        'buying undervalued tokens.")'
    ]
    return agent


def main(
    market_type: MarketType = MarketType.OMEN,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4-turbo-preview",
    iterations: int = 10,
    seed_prompt: str | None = None,
) -> None:
    # This description below serves to unique identify agent entries on the LTM, and should be
    # unique across instances (i.e. markets).
    unique_task_description = f"microchain-agent-demo-{market_type}"
    long_term_memory = LongTermMemory(unique_task_description)

    agent = build_agent(
        market_type=market_type,
        api_base=api_base,
        model=model,
        long_term_memory=long_term_memory,
        allow_stop=False,  # Prevent the agent from stopping itself
    )
    if seed_prompt:
        agent.bootstrap = [f'Reasoning("{seed_prompt}")']
    agent.run(iterations=iterations)
    # generator.print_usage() # Waiting for microchain release
    long_term_memory.save_history(agent.history)


if __name__ == "__main__":
    typer.run(main)
