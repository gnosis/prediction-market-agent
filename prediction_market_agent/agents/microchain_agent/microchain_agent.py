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
from prediction_market_agent.agents.microchain_agent.memory_functions import (
    RememberPastActions,
)
from prediction_market_agent.agents.microchain_agent.omen_functions import (
    OMEN_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.prompts import (
    TRADING_AGENT_SYSTEM_PROMPT,
    build_full_system_prompt,
    extract_updatable_system_prompt,
)
from prediction_market_agent.agents.utils import AgentIdentifier
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.prompt_table_handler import PromptTableHandler
from prediction_market_agent.utils import APIKeys


def build_agent_functions(
    agent: Agent,
    market_type: MarketType,
    allow_stop: bool,
    long_term_memory: LongTermMemoryTableHandler | None,
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
    api_base: str = "https://api.openai.com/v1",
    long_term_memory: LongTermMemoryTableHandler | None = None,
    allow_stop: bool = True,
    bootstrap: str | None = None,
    prompt_handler: PromptTableHandler | None = None,
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

    # Restore the prompt from a historical session, replacing the editable part with it.
    if prompt_handler:
        if historical_prompt := prompt_handler.fetch_latest_prompt():
            system_prompt = build_full_system_prompt(historical_prompt.prompt)

    agent.system_prompt = system_prompt.format(engine_help=agent.engine.help)
    if bootstrap:
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
    unique_task_description = (
        AgentIdentifier.microchain_task_from_market(market_type) + "_test"
    )
    long_term_memory = LongTermMemoryTableHandler(unique_task_description)

    # We only use microchain on Omen currently, hence no need for prompt handler for other markets.
    prompt_handler = (
        PromptTableHandler(session_identifier=AgentIdentifier.MICROCHAIN_AGENT_OMEN)
        if market_type == MarketType.OMEN and load_historical_prompt
        else None
    )

    agent = build_agent(
        market_type=market_type,
        api_base=api_base,
        model=model,
        system_prompt=TRADING_AGENT_SYSTEM_PROMPT,
        long_term_memory=long_term_memory,
        allow_stop=False,  # Prevent the agent from stopping itself
        prompt_handler=prompt_handler,
    )
    if seed_prompt:
        agent.bootstrap = [f'Reasoning("{seed_prompt}")']
    agent.run(iterations=iterations)
    # generator.print_usage() # Waiting for microchain release
    long_term_memory.save_history(agent.history)
    editable_prompt = get_editable_prompt_from_agent(agent)
    if prompt_handler:
        prompt_handler.save_prompt(editable_prompt)


def get_editable_prompt_from_agent(agent: Agent) -> str:
    return extract_updatable_system_prompt(str(agent.system_prompt))
