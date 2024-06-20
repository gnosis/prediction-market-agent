"""
PYTHONPATH=. streamlit run prediction_market_agent/agents/microchain_agent/app.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, instead of just this one.
"""

# Imports using asyncio (in this case mech_client) cause issues with Streamlit
from prediction_market_agent.tools.streamlit_utils import (  # isort:skip
    display_chat_history,
    streamlit_asyncio_event_loop_hack,
)


streamlit_asyncio_event_loop_hack()

# Fix "Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0" error
from prediction_market_agent.utils import patch_sqlite3  # isort:skip
from prediction_market_agent.agents.microchain_agent.prompt_handler import PromptHandler

patch_sqlite3()


import streamlit as st
from microchain import Agent
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.costs import openai_costs
from prediction_market_agent_tooling.tools.streamlit_user_login import streamlit_login
from streamlit_extras.bottom_container import bottom

from prediction_market_agent.agents.microchain_agent.memory import (
    ChatHistory,
    LongTermMemory,
)
from prediction_market_agent.agents.microchain_agent.microchain_agent import (
    build_agent,
    build_agent_functions,
)
from prediction_market_agent.agents.microchain_agent.prompts import (
    SYSTEM_PROMPTS,
    SystemPromptChoice,
)
from prediction_market_agent.agents.microchain_agent.utils import (
    get_balance,
    get_initial_history_length,
    has_been_run_past_initialization,
)
from prediction_market_agent.agents.utils import AgentIdentifier
from prediction_market_agent.tools.streamlit_utils import check_required_api_keys
from prediction_market_agent.utils import APIKeys

MARKET_TYPE = MarketType.OMEN
ALLOW_STOP = False


def run_agent(agent: Agent, iterations: int, model: str) -> None:
    maybe_initialize_long_term_memory()
    with openai_costs(model) as costs:
        with st.spinner("Agent is running..."):
            for _ in range(iterations):
                agent.run(iterations=1, resume=True)
        st.session_state.running_cost += costs.cost


def execute_reasoning(agent: Agent, reasoning: str, model: str) -> None:
    with openai_costs(model) as costs:
        agent.execute_command(f'Reasoning("{reasoning}")')
        display_new_history_callback(agent)  # Run manually after `execute_command`
        st.session_state.running_cost += costs.cost


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
        st.session_state.long_term_memory = LongTermMemory(
            AgentIdentifier.MICROCHAIN_AGENT_STREAMLIT
        )


def agent_is_initialized() -> bool:
    return "agent" in st.session_state


def save_last_turn_history_to_memory(agent: Agent) -> None:
    last_turn_history = get_history_from_last_turn(agent)
    st.session_state.long_term_memory.save_history(last_turn_history)


def maybe_initialize_agent(model: str, system_prompt: str, bootstrap: str) -> None:
    # Initialize the agent
    if not agent_is_initialized():
        st.session_state.agent = build_agent(
            market_type=MARKET_TYPE,
            model=model,
            system_prompt=system_prompt,
            bootstrap=bootstrap,
            allow_stop=ALLOW_STOP,
            long_term_memory=st.session_state.long_term_memory,
            prompt_handler=PromptHandler(
                session_identifier=AgentIdentifier.MICROCHAIN_AGENT_STREAMLIT
            )
            if st.session_state.get("load_historical_prompt")
            else None,
        )
        st.session_state.agent.reset()
        st.session_state.agent.build_initial_messages()
        st.session_state.running_cost = 0.0

        # Add a callback to display the agent's history after each run
        st.session_state.agent.on_iteration_end = display_new_history_callback


def get_function_bullet_point_list(agent: Agent, model: str) -> str:
    bullet_points = ""
    for function in build_agent_functions(
        agent=agent,
        market_type=MARKET_TYPE,
        long_term_memory=st.session_state.long_term_memory,
        allow_stop=ALLOW_STOP,
        model=model,
    ):
        bullet_points += f"  - {function.__class__.__name__}\n"
    return bullet_points


##########
# Layout #
##########

st.set_page_config(
    layout="wide",
    page_title="Gnosis AI: Prediction Market Trader Agent",
    page_icon=":owl:",
)
st.title("Prediction Market Trader Agent")
with st.sidebar:
    streamlit_login()
check_required_api_keys(["OPENAI_API_KEY", "BET_FROM_PRIVATE_KEY"])
keys = APIKeys()
maybe_initialize_long_term_memory()

with st.sidebar:
    st.subheader("Agent Info:")
    with st.container(border=True):
        # Placeholder for the agent's balance
        balance_container = st.container()
    st.write(
        f"To see the agent's transaction history, click [here]({MARKET_TYPE.market_class.get_user_url(keys=keys)})."
    )

    st.divider()
    st.subheader("Configure:")
    st.selectbox(
        "Model",
        [
            "gpt-4-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-4o-2024-05-13",
        ],
        index=0,
        disabled=agent_is_initialized(),
        key="model",
    )

    st.selectbox(
        "Initial memory",
        [p.value for p in SystemPromptChoice],
        index=0,
        key="system_prompt_select",
        disabled=agent_is_initialized(),
    )
    st.toggle(
        "Load historical prompt",
        key="load_historical_prompt",
        disabled=agent_is_initialized(),
    )

    system_prompt, bootstrap = SYSTEM_PROMPTS[
        SystemPromptChoice(st.session_state.system_prompt_select)
    ]

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
    "participate in prediction markets. More info..."
)
system_prompt_expander = st.expander("Agent's current system prompt")
bootstrap_expander = st.expander("Agent's current bootstrap")

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
        maybe_initialize_agent(st.session_state.model, system_prompt, bootstrap)
        execute_reasoning(
            agent=st.session_state.agent,
            reasoning=user_reasoning,
            model=st.session_state.model,
        )
        # Run the agent after the user's reasoning
        run_agent(
            agent=st.session_state.agent,
            iterations=int(iterations),
            model=st.session_state.model,
        )
        save_last_turn_history_to_memory(st.session_state.agent)
    if run_agent_button:
        maybe_initialize_agent(st.session_state.model, system_prompt, bootstrap)
        run_agent(
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
        st.info(
            f"Running OpenAPI credits cost: ${st.session_state.running_cost:.2f}"
        )  # TODO debug why always == 0.0

# Once the agent has run...

# Display its updated balance
with balance_container:
    st.metric(
        label=f"Current balance ({MARKET_TYPE.market_class.currency})",
        value=f"{get_balance(MARKET_TYPE).amount:.2f}",
    )

# Display its updated function list, system prompt and bootstrap
with intro_expander:
    st.markdown(
        "To start, click 'Run' to see the agent in action, or bootstrap the "
        "agent with your own reasoning."
    )
    st.markdown("It is equipped with the following tools:")

    st.markdown(
        get_function_bullet_point_list(
            agent=st.session_state.agent, model=st.session_state.model
        )
        if agent_is_initialized()
        else "The agent is not initialized yet."
    )

with system_prompt_expander:
    st.markdown(
        st.session_state.agent.system_prompt
        if agent_is_initialized()
        else "The agent is not initialized yet."
    )


with bootstrap_expander:
    st.markdown(
        st.session_state.agent.bootstrap
        if agent_is_initialized()
        else "The agent is not initialized yet."
    )
