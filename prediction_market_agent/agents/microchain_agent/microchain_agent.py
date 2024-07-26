from enum import Enum

from eth_typing import ChecksumAddress
from loguru import logger
from microchain import LLM, Agent, Engine, Function, OpenAIChatGenerator
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.utils import should_not_happen

from prediction_market_agent.agents.microchain_agent.agent_functions import (
    AGENT_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.blockchain.contract_class_converter import (
    ContractClassConverter,
)
from prediction_market_agent.agents.microchain_agent.blockchain.models import (
    AbiItemStateMutabilityEnum,
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
from prediction_market_agent.agents.microchain_agent.microchain_generators import (
    ReplicateLlama31,
)
from prediction_market_agent.agents.microchain_agent.omen_functions import (
    OMEN_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.prompts import (
    build_full_unformatted_system_prompt,
    extract_updatable_system_prompt,
)
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.prompt_table_handler import PromptTableHandler
from prediction_market_agent.utils import APIKeys


class SupportedModel(str, Enum):
    gpt_4_turbo = "gpt-4-turbo"
    gpt_35_turbo = "gpt-3.5-turbo-0125"
    gpt_4o = "gpt-4o-2024-05-13"
    llama_31_instruct = "meta/meta-llama-3.1-405b-instruct"

    @property
    def is_openai(self) -> bool:
        return self in [
            SupportedModel.gpt_4_turbo,
            SupportedModel.gpt_35_turbo,
            SupportedModel.gpt_4o,
        ]

    @property
    def is_replicate(self) -> bool:
        return self in [SupportedModel.llama_31_instruct]

      
def build_functions_from_smart_contract(
    keys: APIKeys, contract_address: ChecksumAddress, contract_name: str
) -> list[Function]:
    functions = []

    contract_class_converter = ContractClassConverter(
        contract_address=contract_address, contract_name=contract_name
    )
    function_types_to_classes = (
        contract_class_converter.create_classes_from_smart_contract()
    )

    view_classes = function_types_to_classes[AbiItemStateMutabilityEnum.VIEW]
    functions.extend([clz() for clz in view_classes])

    payable_classes = function_types_to_classes[AbiItemStateMutabilityEnum.PAYABLE]
    non_payable_classes = function_types_to_classes[
        AbiItemStateMutabilityEnum.NON_PAYABLE
    ]
    for clz in payable_classes + non_payable_classes:
        functions.append(clz(keys=keys))

    return functions


def build_agent_functions(
    agent: Agent,
    market_type: MarketType,
    keys: APIKeys,
    allow_stop: bool,
    long_term_memory: LongTermMemoryTableHandler | None,
    model: str,
) -> list[Function]:
    logger.error("entered build agent functions")
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
    model: SupportedModel,
    unformatted_system_prompt: str,
    api_base: str = "https://api.openai.com/v1",
    long_term_memory: LongTermMemoryTableHandler | None = None,
    allow_stop: bool = True,
    bootstrap: str | None = None,
) -> Agent:
    engine = Engine()
    generator = (
        OpenAIChatGenerator(
            model=model.value,
            api_key=keys.openai_api_key.get_secret_value(),
            api_base=api_base,
            temperature=0.7,
        )
        if model.is_openai
        else (
            ReplicateLlama31(
                model=model.value,
                api_key=keys.replicate_api_key.get_secret_value(),
            )
            if model.is_replicate
            else should_not_happen()
        )
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
    # Save off the most up-to-date, or 'head' system prompt
    head_system_prompt = agent.history[0]

    # Restore the system prompt to its initial state
    agent.history[0] = dict(role="system", content=initial_system_prompt)
    long_term_memory.save_history(agent.history)

    # Restore the head system prompt
    agent.history[0] = head_system_prompt


def get_editable_prompt_from_agent(agent: Agent) -> str:
    return extract_updatable_system_prompt(str(agent.system_prompt))
