"""
PYTHONPATH=. streamlit run prediction_market_agent/agents/microchain_agent/app.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, instead of just this one.
"""

from prediction_market_agent_tooling.tools.streamlit_utils import (
    streamlit_asyncio_event_loop_hack,
)

from prediction_market_agent.db.prompt_table_handler import PromptTableHandler

# Imports using asyncio (in this case mech_client) cause issues with Streamlit
from prediction_market_agent.tools.streamlit_utils import (  # isort:skip
    display_chat_history,
)

streamlit_asyncio_event_loop_hack()

# Fix "Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0" error
from prediction_market_agent.utils import patch_sqlite3  # isort:skip

patch_sqlite3()

import langfuse
import streamlit as st
from microchain import Agent
from prediction_market_agent_tooling.deploy.agent import initialize_langfuse
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.streamlit_user_login import streamlit_login
from prediction_market_agent_tooling.tools.streamlit_utils import (
    check_required_api_keys,
)
from prediction_market_agent_tooling.tools.utils import utcnow
from streamlit_extras.bottom_container import bottom

from prediction_market_agent.agents.identifiers import MICROCHAIN_AGENT_STREAMLIT
from prediction_market_agent.agents.microchain_agent.deploy import GENERAL_AGENT_TAG
from prediction_market_agent.agents.microchain_agent.memory import ChatHistory
from prediction_market_agent.agents.microchain_agent.microchain_agent import (
    SupportedModel,
    build_agent,
    get_unformatted_system_prompt,
)
from prediction_market_agent.agents.microchain_agent.prompts import (
    JOB_AGENT_SYSTEM_PROMPT_CONFIG,
    JUST_BORN_SYSTEM_PROMPT_CONFIG,
    TRADING_AGENT_SYSTEM_PROMPT_CONFIG,
    TRADING_AGENT_SYSTEM_PROMPT_MINIMAL_CONFIG,
    extract_updatable_system_prompt,
)
from prediction_market_agent.agents.microchain_agent.utils import (
    get_balance,
    get_initial_history_length,
    has_been_run_past_initialization,
)
from prediction_market_agent.agents.utils import STREAMLIT_TAG
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.utils import APIKeys

MARKET_TYPE = MarketType.OMEN
AGENT_IDENTIFIER = MICROCHAIN_AGENT_STREAMLIT
ALLOW_STOP = False
ALLOWED_CONFIGS = {
    x.name: x
    for x in (
        TRADING_AGENT_SYSTEM_PROMPT_CONFIG,
        TRADING_AGENT_SYSTEM_PROMPT_MINIMAL_CONFIG,
        JOB_AGENT_SYSTEM_PROMPT_CONFIG,
        JUST_BORN_SYSTEM_PROMPT_CONFIG,
    )
}

st.session_state.session_id = st.session_state.get(
    "session_id", "StrealitGeneralAgent - " + utcnow().strftime("%Y-%m-%d %H:%M:%S")
)


@observe()
def run_general_agent_streamlit(
    agent: Agent, iterations: int, model: SupportedModel
) -> None:
    langfuse.get_client().update_current_trace(
        tags=[GENERAL_AGENT_TAG, STREAMLIT_TAG], session_id=st.session_state.session_id
    )
    maybe_initialize_long_term_memory()
    with st.spinner("Agent is running..."):
        for _ in range(iterations):
            agent.run(iterations=1, resume=st.session_state.total_iterations > 0)
            st.session_state.total_iterations += 1
    st.session_state.running_cost = agent.llm.generator.token_tracker.get_total_cost(
        model.value
    )


def execute_reasoning(agent: Agent, reasoning: str, model: SupportedModel) -> None:
    agent.execute_command(f'Reasoning("{reasoning}")')
    display_new_history_callback(agent)  # Run manually after `execute_command`
    st.session_state.running_cost = agent.llm.generator.token_tracker.get_total_cost(
        model.value
    )


def display_agent_history(agent: Agent) -> None:
    """
    Display the agent's history in the Streamlit app.
    """
    # Skip the initial messages
    history = agent.history[get_initial_history_length(agent) :]
    display_chat_history(ChatHistory.from_list_of_dicts(history))


def get_history_from_last_turn(agent: Agent) -> ChatHistory:
    if len(agent.history) < 2:
        raise ValueError(
            "Agent history is too short. You must call `agent.run` before calling this function."
        )

    history_depth = 2  # One for the user input, one for the agent's reply
    history = agent.history[-history_depth:]
    return ChatHistory.from_list_of_dicts(history)


def display_new_history_callback(agent: Agent) -> None:
    """
    A callback to display the agent's history in the Streamlit app after a run
    with a single interation.
    """
    display_chat_history(chat_history=get_history_from_last_turn(agent))


def long_term_memory_is_initialized() -> bool:
    return "long_term_memory" in st.session_state


def maybe_initialize_long_term_memory() -> None:
    # Initialize the db storage
    if not long_term_memory_is_initialized():
        st.session_state.long_term_memory = (
            LongTermMemoryTableHandler.from_agent_identifier(AGENT_IDENTIFIER)
        )


def agent_is_initialized() -> bool:
    return "agent" in st.session_state


def save_last_turn_history_to_memory(agent: Agent) -> None:
    get_history_from_last_turn(agent).save_to(st.session_state.long_term_memory)


def maybe_initialize_agent(
    model: SupportedModel, unformatted_system_prompt: str
) -> None:
    # Set the unformatted system prompt
    prompt_table_handler = (
        PromptTableHandler.from_agent_identifier(AGENT_IDENTIFIER)
        if st.session_state.get("load_historical_prompt")
        else None
    )
    unformatted_system_prompt = get_unformatted_system_prompt(
        unformatted_prompt=unformatted_system_prompt,
        prompt_table_handler=prompt_table_handler,
    )

    # Initialize the agent
    if not agent_is_initialized():
        initialize_langfuse(ENABLE_LANGFUSE)
        st.session_state.agent = build_agent(
            market_type=MARKET_TYPE,
            model=model,
            unformatted_system_prompt=unformatted_system_prompt,
            allow_stop=ALLOW_STOP,
            long_term_memory=st.session_state.long_term_memory,
            keys=KEYS,
            functions_config=st.session_state.selected_config.functions_config,
            enable_langfuse=ENABLE_LANGFUSE,
        )
        st.session_state.total_iterations = 0
        st.session_state.running_cost = 0.0
        # Add a callback to display the agent's history after each run
        st.session_state.agent.on_iteration_end = display_new_history_callback


def get_function_bullet_point_list(agent: Agent) -> str:
    if len(agent.engine.functions) == 0:
        raise ValueError("Agent must be initialized with registered functions.")
    bullet_points = ""
    for function in agent.engine.functions:
        bullet_points += f"  - {function}\n"
    return bullet_points


##########
# Layout #
##########

st.set_page_config(
    layout="wide",
    page_title="Gnosis AI: Agents Playground",
    page_icon=":owl:",
)
st.title("Agent")
with st.sidebar:
    streamlit_login()
KEYS = APIKeys()
check_required_api_keys(KEYS, ["OPENAI_API_KEY", "BET_FROM_PRIVATE_KEY"])
ENABLE_LANGFUSE = KEYS.default_enable_langfuse
maybe_initialize_long_term_memory()

with st.sidebar:
    st.subheader("Agent Info:")
    with st.container(border=True):
        # Placeholder for the agent's balance
        balance_container = st.container()
    st.write(
        f"To see the agent's transaction history, click [here]({MARKET_TYPE.market_class.get_user_url(keys=KEYS)})."
    )

    st.divider()
    st.subheader("Configure:")
    st.session_state.model = SupportedModel(
        st.selectbox(
            "Model",
            [m.value for m in SupportedModel],
            index=0,
            disabled=agent_is_initialized(),
        )
    )

    st.session_state.selected_config = ALLOWED_CONFIGS[
        st.selectbox(
            "Initial memory config",
            list(ALLOWED_CONFIGS.keys()),
            index=0,
            disabled=agent_is_initialized(),
        )
    ]

    st.toggle(
        "Load historical prompt",
        key="load_historical_prompt",
        disabled=agent_is_initialized(),
    )

    system_prompt = st.session_state.selected_config.system_prompt

    st.divider()
    st.subheader("Built by Gnosis AI")
    st.image(
        "https://assets-global.website-files.com/63692bf32544bee8b1836ea6/63bea29e26bbff45e81efe4a_gnosis-owl-cream.png",
        width=30,
    )

    st.caption(
        "View the source code on our [github](https://github.com/gnosis/prediction-market-agent/tree/main/prediction_market_agent/agents/microchain_agent)"
    )

intro_expander = st.expander(
    "Interact with an autonomous agent that uses its own balance to "
    "live its live on blockchain. More info..."
)
system_prompt_expander = st.expander("Agent's current system prompt")

# Placeholder for the agent's history
history_container = st.container()

# Interactive elements
with bottom():
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 8])
        with col1:
            run_agent_button = st.button("Run the agent")
        with col2:
            iterations = st.number_input(
                "Iterations",
                value=1,
                step=1,
                min_value=1,
                max_value=100,
            )
        with col3:
            st.caption(" \- OR -")
        with col4:
            user_reasoning = st.chat_input("Add reasoning and run the agent")

#############
# Execution #
#############

# Run the agent and display its history
with history_container:
    if agent_is_initialized():
        display_agent_history(st.session_state.agent)
    if user_reasoning:
        maybe_initialize_agent(st.session_state.model, system_prompt)
        execute_reasoning(
            agent=st.session_state.agent,
            reasoning=user_reasoning,
            model=st.session_state.model,
        )
        # Run the agent after the user's reasoning
        run_general_agent_streamlit(
            agent=st.session_state.agent,
            iterations=int(iterations),
            model=st.session_state.model,
        )
        save_last_turn_history_to_memory(st.session_state.agent)
    if run_agent_button:
        maybe_initialize_agent(st.session_state.model, system_prompt)
        run_general_agent_streamlit(
            agent=st.session_state.agent,
            iterations=int(iterations),
            model=st.session_state.model,
        )
        save_last_turn_history_to_memory(st.session_state.agent)
    if (
        agent_is_initialized()
        and has_been_run_past_initialization(st.session_state.agent)
        and st.session_state.running_cost > 0.0
    ):
        # Display running cost
        st.info(f"Running LLM credits cost: ${st.session_state.running_cost:.2f}")

# Once the agent has run...

# Display its updated balance
with balance_container:
    st.metric(
        label=f"Current balance (USD)",
        value=f"{get_balance(KEYS, MARKET_TYPE).value:.2f}",
    )

# Display its updated function list, system prompt
with intro_expander:
    st.markdown(
        "To start, click 'Run' to see the agent in action, or execute the "
        "agent with your own reasoning."
    )
    st.markdown("It is equipped with the following tools:")

    st.markdown(
        get_function_bullet_point_list(agent=st.session_state.agent)
        if agent_is_initialized()
        else "The agent is not initialized yet."
    )

with system_prompt_expander:
    st.markdown(
        extract_updatable_system_prompt(st.session_state.agent.system_prompt)
        if agent_is_initialized()
        else "The agent is not initialized yet."
    )
