from microchain import LLM, Agent, Engine, Function, OpenAIChatGenerator
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.agent_functions import (
    AGENT_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.call_api import API_FUNCTIONS
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
    build_full_unformatted_system_prompt,
)
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.prompt_table_handler import PromptTableHandler
from prediction_market_agent.utils import APIKeys


def build_agent_functions(
    agent: Agent,
    market_type: MarketType,
    keys: APIKeys,
    allow_stop: bool,
    long_term_memory: LongTermMemoryTableHandler | None,
    model: str,
) -> list[Function]:
    functions = []

    functions.append(Reasoning())
    if allow_stop:
        functions.append(Stop())

    functions.extend([f() for f in API_FUNCTIONS])
    functions.extend([f() for f in LEARNING_FUNCTIONS])
    functions.extend([f(agent=agent) for f in AGENT_FUNCTIONS])
    functions.extend([f(market_type=market_type, keys=keys) for f in MARKET_FUNCTIONS])
    if market_type == MarketType.OMEN:
        functions.extend([f() for f in OMEN_FUNCTIONS])
    if long_term_memory:
        functions.append(
            RememberPastActions(long_term_memory=long_term_memory, model=model)
        )
    return functions


def build_agent(
    keys: APIKeys,
    market_type: MarketType,
    model: str,
    unformatted_system_prompt: str,
    api_base: str = "https://api.openai.com/v1",
    long_term_memory: LongTermMemoryTableHandler | None = None,
    allow_stop: bool = True,
    bootstrap: str | None = None,
) -> Agent:
    engine = Engine()
    generator = OpenAIChatGenerator(
        model=model,
        api_key=keys.openai_api_key.get_secret_value(),
        api_base=api_base,
        temperature=0.7,
    )
    agent = Agent(llm=LLM(generator=generator), engine=engine)

    for f in build_agent_functions(
        agent=agent,
        market_type=market_type,
        keys=keys,
        allow_stop=allow_stop,
        long_term_memory=long_term_memory,
        model=model,
    ):
        engine.register(f)

    agent.max_tries = 3

    agent.system_prompt = unformatted_system_prompt.format(
        engine_help=agent.engine.help
    )
    if bootstrap:
        agent.bootstrap = [bootstrap]
    return agent


def get_unformatted_system_prompt(
    unformatted_prompt: str, prompt_table_handler: PromptTableHandler | None
) -> str:
    # Restore the prompt from a historical session, replacing the editable part with it.
    if prompt_table_handler:
        if historical_prompt := prompt_table_handler.fetch_latest_prompt():
            return build_full_unformatted_system_prompt(historical_prompt.prompt)

    # If no historical prompt is found, return the original prompt.
    return unformatted_prompt


def save_agent_history(
    long_term_memory: LongTermMemoryTableHandler,
    agent: Agent,
    initial_system_prompt: str,
) -> None:
    """
    Save the agent's history to the long-term memory. But first, restore the
    system prompt to its initial state. This is necessary because the some
    functions may have changed the system prompt during the agent's run.
    """
    agent.history[0] = dict(role="system", content=initial_system_prompt)
    long_term_memory.save_history(agent.history)
