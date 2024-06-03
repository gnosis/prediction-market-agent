import typer
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
    RememberPastLearnings,
)
from prediction_market_agent.agents.microchain_agent.omen_functions import (
    OMEN_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.prompts import (
    TRADING_AGENT_BOOTSTRAP,
    TRADING_AGENT_SYSTEM_PROMPT,
)
from prediction_market_agent.agents.utils import LongTermMemoryTaskIdentifier
from prediction_market_agent.utils import APIKeys


class CustomizedAgent(Agent):
    """
    Subclass microchain's agent for any customizations that are needed right away.
    Anything generally useful should be moved upstream to the microchain repository.
    """

    def build_initial_messages(self) -> None:
        # Use self.prompt in the system prompt instead of the user prompt.
        # TODO: This should be moved upstream to the microchain repository, there should be both "prompt" and "system_prompt" argument in __init__.
        self.history = [
            dict(role="system", content=self.prompt),
        ]
        for command in self.bootstrap:
            self.execute_command(command)


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
            RememberPastLearnings(long_term_memory=long_term_memory, model=model)
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
) -> Agent:
    engine = Engine()
    generator = OpenAIChatGenerator(
        model=model,
        api_key=APIKeys().openai_api_key.get_secret_value(),
        api_base=api_base,
        temperature=0.7,
    )
    agent = CustomizedAgent(llm=LLM(generator=generator), engine=engine)

    for f in build_agent_functions(
        agent=agent,
        market_type=market_type,
        allow_stop=allow_stop,
        long_term_memory=long_term_memory,
        model=model,
    ):
        engine.register(f)

    agent.max_tries = 3
    print(system_prompt)
    agent.prompt = system_prompt.format(engine_help=engine.help)
    agent.bootstrap = [bootstrap]
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
    unique_task_description = LongTermMemoryTaskIdentifier.microchain_task_from_market(
        market_type
    )
    long_term_memory = LongTermMemory(unique_task_description)

    agent = build_agent(
        market_type=market_type,
        api_base=api_base,
        model=model,
        system_prompt=TRADING_AGENT_SYSTEM_PROMPT,  # Use pre-learned system prompt until the prompt-fetching from DB is implemented.
        bootstrap=TRADING_AGENT_BOOTSTRAP,  # Same here.
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
