import typer
from loguru import logger
from microchain import LLM, Agent, Engine, Function, OpenAIChatGenerator
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.agent_functions import (
    AGENT_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.learning_functions import (
    LEARNING_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.market_functions import (
    MARKET_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.memory import LongTermMemory
from prediction_market_agent.agents.microchain_agent.memory_functions import (
    RememberPastActions,
)
from prediction_market_agent.agents.microchain_agent.omen_functions import (
    OMEN_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.prompt_handler import PromptHandler
from prediction_market_agent.agents.microchain_agent.prompts import (
    TRADING_AGENT_BOOTSTRAP,
    TRADING_AGENT_SYSTEM_PROMPT,
)
from prediction_market_agent.agents.utils import LongTermMemoryTaskIdentifier
from prediction_market_agent.utils import APIKeys


def build_agent_functions(
    agent: Agent,
    market_type: MarketType,
    allow_stop: bool,
    long_term_memory: LongTermMemory | None,
    model: str,
) -> list[Function]:
    functions = []

    functions.append(Reasoning())
    if allow_stop:
        functions.append(Stop())

    functions.extend([f() for f in LEARNING_FUNCTIONS])
    functions.extend([f(agent=agent) for f in AGENT_FUNCTIONS])
    functions.extend([f(market_type=market_type) for f in MARKET_FUNCTIONS])
    if market_type == MarketType.OMEN:
        functions.extend([f() for f in OMEN_FUNCTIONS])
    if long_term_memory:
        functions.append(
            RememberPastActions(long_term_memory=long_term_memory, model=model)
        )
    return functions


def build_agent(
    market_type: MarketType,
    model: str,
    system_prompt: str,
    bootstrap: str,
    api_base: str = "https://api.openai.com/v1",
    long_term_memory: LongTermMemory | None = None,
    allow_stop: bool = True,
    prompt_handler: PromptHandler | None = None,
) -> Agent:
    engine = Engine()
    generator = OpenAIChatGenerator(
        model=model,
        api_key=APIKeys().openai_api_key.get_secret_value(),
        api_base=api_base,
        temperature=0.7,
    )
    agent = Agent(llm=LLM(generator=generator), engine=engine)

    for f in build_agent_functions(
        agent=agent,
        market_type=market_type,
        allow_stop=allow_stop,
        long_term_memory=long_term_memory,
        model=model,
    ):
        engine.register(f)

    agent.max_tries = 3

    # We restore the prompt from a historical session.
    if prompt_handler:
        if historical_prompt := prompt_handler.fetch_latest_prompt():
            system_prompt = historical_prompt.prompt

    # if {engine_help} not in prompt, we expect the functions to have been already loaded,
    # thus no need to load them again. Otherwise we can simply not use the historical prompt.
    agent.system_prompt = system_prompt
    if "{engine_help}" not in system_prompt:
        logger.info("Agent's functions were not loaded into prompt")
        # We simply call help here otherwise microchain throws exception if not called.
        engine.help
    else:
        agent.system_prompt = system_prompt.format(engine_help=engine.help)
    agent.bootstrap = [bootstrap]
    return agent


def main(
    market_type: MarketType = MarketType.OMEN,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4-turbo-preview",
    iterations: int = 10,
    seed_prompt: str | None = None,
    load_historical_prompt: bool = False,
) -> None:
    # This description below serves to unique identify agent entries on the LTM, and should be
    # unique across instances (i.e. markets).
    unique_task_description = LongTermMemoryTaskIdentifier.microchain_task_from_market(
        market_type
    )
    long_term_memory = LongTermMemory(unique_task_description)
    prompt_handler = PromptHandler()

    agent = build_agent(
        market_type=market_type,
        api_base=api_base,
        model=model,
        system_prompt=TRADING_AGENT_SYSTEM_PROMPT,  # Use pre-learned system prompt until the prompt-fetching from DB is implemented.
        bootstrap=TRADING_AGENT_BOOTSTRAP,  # Same here.
        long_term_memory=long_term_memory,
        allow_stop=False,  # Prevent the agent from stopping itself
        prompt_handler=prompt_handler if load_historical_prompt else None,
    )
    if seed_prompt:
        agent.bootstrap = [f'Reasoning("{seed_prompt}")']
    agent.run(iterations=iterations)
    # generator.print_usage() # Waiting for microchain release
    long_term_memory.save_history(agent.history)
    prompt_handler.save_prompt(agent.system_prompt)


if __name__ == "__main__":
    typer.run(main)
