from enum import Enum

from eth_typing import ChecksumAddress
from microchain import (
    LLM,
    Agent,
    Engine,
    Function,
    FunctionResult,
    OpenAIChatGenerator,
    ReplicateLlama31ChatGenerator,
    StepOutput,
)
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.utils import should_not_happen

from prediction_market_agent.agents.microchain_agent.agent_functions import (
    AGENT_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.balance_functions import (
    BALANCE_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.blockchain.contract_class_converter import (
    ContractClassConverter,
)
from prediction_market_agent.agents.microchain_agent.blockchain.models import (
    AbiItemStateMutabilityEnum,
)
from prediction_market_agent.agents.microchain_agent.call_api import API_FUNCTIONS
from prediction_market_agent.agents.microchain_agent.code_functions import (
    CODE_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.common_functions import (
    COMMON_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.jobs_functions import JOB_FUNCTIONS
from prediction_market_agent.agents.microchain_agent.learning_functions import (
    LEARNING_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.market_functions import (
    MARKET_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.memory_functions import (
    MEMORY_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.nft_functions import NFT_FUNCTIONS
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.nft_game_functions import (
    NFT_GAME_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.nft_game_messages_functions import (
    MESSAGES_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.omen_functions import (
    OMEN_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.prompts import (
    FunctionsConfig,
    build_full_unformatted_system_prompt,
    extract_updatable_system_prompt,
)
from prediction_market_agent.agents.microchain_agent.search_functions import (
    SEARCH_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.sending_functions import (
    SENDING_FUNCTIONS,
)
from prediction_market_agent.agents.microchain_agent.twitter_functions import (
    TWITTER_FUNCTIONS,
)
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.prompt_table_handler import PromptTableHandler
from prediction_market_agent.utils import OPENROUTER_BASE_URL, APIKeys


class SupportedModel(str, Enum):
    gpt_4o = "gpt-4o-2024-08-06"
    gpt_4o_mini = "gpt-4o-mini-2024-07-18"
    gpt_4_turbo = "gpt-4-turbo"
    llama_31_instruct = "meta/meta-llama-3.1-405b-instruct"
    deepseek_chat = "deepseek/deepseek-chat"
    deepseek_r1 = "deepseek/deepseek-r1"
    gemini_20_flash = "google/gemini-2.0-flash-001"

    @property
    def is_openai(self) -> bool:
        return self.value.startswith("gpt-") or self.value.startswith("o1-")

    @property
    def is_replicate(self) -> bool:
        return self in [SupportedModel.llama_31_instruct]

    @property
    def is_openrouter(self) -> bool:
        return self in [
            SupportedModel.deepseek_chat,
            SupportedModel.deepseek_r1,
            SupportedModel.gemini_20_flash,
        ]


def replicate_model_to_tokenizer(model: SupportedModel) -> str:
    if model == SupportedModel.llama_31_instruct:
        return "tokenizers/replicate_llama_31_405b"
    else:
        raise ValueError(f"Unsupported model: {model}")


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
    functions_config: FunctionsConfig,
) -> list[Function]:
    functions = []

    functions.append(Reasoning())
    if allow_stop:
        functions.append(Stop())

    if functions_config.common_functions:
        functions.extend(f() for f in COMMON_FUNCTIONS)

    if functions_config.include_agent_functions:
        functions.extend([f(agent=agent) for f in AGENT_FUNCTIONS])

    if functions_config.include_universal_functions:
        functions.extend([f() for f in API_FUNCTIONS])
        functions.extend([f() for f in CODE_FUNCTIONS])
        functions.extend([f() for f in SEARCH_FUNCTIONS])

    if functions_config.include_job_functions:
        functions.extend([f(market_type=market_type, keys=keys) for f in JOB_FUNCTIONS])

    if functions_config.include_learning_functions:
        functions.extend([f() for f in LEARNING_FUNCTIONS])

    if functions_config.include_trading_functions:
        functions.extend(
            [f(market_type=market_type, keys=keys) for f in MARKET_FUNCTIONS]
        )
        if market_type == MarketType.OMEN:
            functions.extend([f() for f in OMEN_FUNCTIONS])

    if functions_config.include_sending_functions:
        functions.extend(f() for f in SENDING_FUNCTIONS)

    if functions_config.include_twitter_functions:
        functions.extend(f() for f in TWITTER_FUNCTIONS)

    if functions_config.include_messages_functions:
        functions.extend(f() for f in MESSAGES_FUNCTIONS)

    if functions_config.include_nft_functions:
        functions.extend(f() for f in NFT_FUNCTIONS)

    if functions_config.balance_functions:
        functions.extend(f() for f in BALANCE_FUNCTIONS)

    if functions_config.nft_game_functions:
        functions.extend(f() for f in NFT_GAME_FUNCTIONS)

    if long_term_memory:
        functions.extend(f(long_term_memory=long_term_memory) for f in MEMORY_FUNCTIONS)

    return functions


def build_agent(
    keys: APIKeys,
    market_type: MarketType,
    model: SupportedModel,
    unformatted_system_prompt: str,
    functions_config: FunctionsConfig,
    enable_langfuse: bool,
    long_term_memory: LongTermMemoryTableHandler | None = None,
    max_tokens: int = 8196,
    allow_stop: bool = True,
    bootstrap: str | None = None,
    raise_on_error: bool = True,
) -> Agent:
    engine = Engine()
    generator = (
        OpenAIChatGenerator(
            model=model.value,
            api_key=(
                keys.openai_api_key if model.is_openai else keys.openrouter_api_key
            ).get_secret_value(),
            api_base=OPENROUTER_BASE_URL if model.is_openrouter else None,
            temperature=0.7,
            enable_langfuse=enable_langfuse,
            max_tokens=max_tokens,
        )
        if model.is_openai or model.is_openrouter
        else (
            ReplicateLlama31ChatGenerator(
                model=model.value,
                tokenizer_pretrained_model_name_or_path=replicate_model_to_tokenizer(
                    model
                ),
                api_key=keys.replicate_api_key.get_secret_value(),
                enable_langfuse=enable_langfuse,
                max_tokens=max_tokens,
            )
            if model.is_replicate
            else should_not_happen()
        )
    )

    if raise_on_error:
        # Define a callback that raises an if an iteration of `agent.run` fails
        def step_end_callback(agent: Agent, step_output: StepOutput) -> None:
            if step_output.result == FunctionResult.ERROR:
                raise Exception(step_output.output)

        on_iteration_step = step_end_callback
    else:
        on_iteration_step = None

    agent = Agent(
        llm=LLM(generator=generator),
        engine=engine,
        on_iteration_step=on_iteration_step,
        enable_langfuse=enable_langfuse,
    )

    for f in build_agent_functions(
        agent=agent,
        market_type=market_type,
        keys=keys,
        allow_stop=allow_stop,
        long_term_memory=long_term_memory,
        model=model,
        functions_config=functions_config,
    ):
        engine.register(f)

    agent.max_tries = 3

    # Use replace instead of `format`, because otherwise it fails if they put into the prompt {values_in_brackets}.
    agent.system_prompt = unformatted_system_prompt.replace(
        "{engine_help}", agent.engine.help
    )
    if bootstrap:
        agent.bootstrap.append(bootstrap)
    return agent


def get_unformatted_system_prompt(
    unformatted_prompt: str, prompt_table_handler: PromptTableHandler | None
) -> str:
    # Restore the prompt from a historical session, replacing the editable part with it.
    if prompt_table_handler:
        if historical_prompt := prompt_table_handler.fetch_latest_prompt():
            return build_full_unformatted_system_prompt(historical_prompt.prompt)

    # If no historical prompt is found, return the original prompt.
    return build_full_unformatted_system_prompt(unformatted_prompt)


def save_agent_history(
    long_term_memory: LongTermMemoryTableHandler,
    agent: Agent,
    initial_system_prompt: str,
    save_last_n: int | None = None,
) -> None:
    """
    Save the agent's history to the long-term memory. But first, restore the
    system prompt to its initial state. This is necessary because the some
    functions may have changed the system prompt during the agent's run.
    """
    # Save off the most up-to-date, or 'head' system prompt
    head_system_prompt = agent.history[0]
    if head_system_prompt["role"] != "system":
        raise ValueError(
            f"Expected the first message in the history to be a system message, but got: {head_system_prompt}"
        )

    # Restore the system prompt to its initial state
    agent.history[0] = dict(role="system", content=initial_system_prompt)
    long_term_memory.save_history(
        agent.history[-save_last_n if save_last_n else len(agent.history) :]
    )

    # Restore the head system prompt
    agent.history[0] = head_system_prompt


def get_editable_prompt_from_agent(agent: Agent) -> str:
    return extract_updatable_system_prompt(str(agent.system_prompt))


def get_functions_summary_list(engine: Engine) -> str:
    return "\n".join(
        [f"- {fname}: {f.description}" for fname, f in engine.functions.items()]
    )
