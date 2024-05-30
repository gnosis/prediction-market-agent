# Imports using asyncio (in this case mech_client) cause issues with Streamlit
from prediction_market_agent.agents.microchain_agent.memory import LongTermMemory
from prediction_market_agent.agents.utils import LongTermMemoryTaskIdentifier

from prediction_market_agent.tools.streamlit_utils import (  # isort:skip
    streamlit_asyncio_event_loop_hack,
)

streamlit_asyncio_event_loop_hack()

# Fix "Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0" error
from prediction_market_agent.utils import patch_sqlite3  # isort:skip

patch_sqlite3()


import streamlit as st
from microchain import Agent
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.costs import openai_costs
from prediction_market_agent_tooling.tools.streamlit_logging import streamlit_login
from prediction_market_agent_tooling.tools.utils import check_not_none
from streamlit_extras.bottom_container import bottom

from prediction_market_agent.agents.microchain_agent.microchain_agent import (
    build_agent,
    build_agent_functions,
)
from prediction_market_agent.agents.microchain_agent.utils import (
    get_balance,
    get_initial_history_length,
    has_been_run_past_initialization,
)
from prediction_market_agent.tools.streamlit_utils import check_required_api_keys
from prediction_market_agent.utils import APIKeys

MARKET_TYPE = MarketType.OMEN
ALLOW_STOP = False


def run_agent(agent: Agent, iterations: int, model: str) -> None:
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


def display_all_history(agent: Agent) -> None:
    """
    Display the agent's history in the Streamlit app.
    """
    # Skip the initial messages
    history = agent.history[get_initial_history_length(agent) :]

    for h in history:
        st.chat_message(h["role"]).write(h["content"])


def get_history_from_last_turn(agent: Agent) -> list[dict[str, str]]:
    if len(agent.history) < 2:
        raise ValueError(
            "Agent history is too short. You must call `agent.run` before calling this function."
        )

    history_depth = 2  # One for the user input, one for the agent's reply
    history = agent.history[-history_depth:]
    str_history: list[dict[str, str]] = [
        {str(key): str(value) for key, value in h.items()} for h in history
    ]  # Enforce string typing
    return str_history


def display_new_history_callback(agent: Agent) -> None:
    """
    A callback to display the agent's history in the Streamlit app after a run
    with a single interation.
    """
    for h in get_history_from_last_turn(agent):
        st.chat_message(h["role"]).write(h["content"])


def long_term_memory_is_initialized() -> bool:
    return "long_term_memory" in st.session_state


def maybe_initialize_long_term_memory() -> None:
    # Initialize the db storage
    if not long_term_memory_is_initialized():
        st.session_state.long_term_memory = LongTermMemory(
            LongTermMemoryTaskIdentifier.MICROCHAIN_AGENT_STREAMLIT
        )


def agent_is_initialized() -> bool:
    return "agent" in st.session_state


def save_last_turn_history_to_memory(agent: Agent) -> None:
    last_turn_history = get_history_from_last_turn(agent)
    st.session_state.long_term_memory.save_history(last_turn_history)


def maybe_initialize_agent(model: str) -> None:
    # Initialize the agent
    if not agent_is_initialized():
        st.session_state.agent = build_agent(
            market_type=MARKET_TYPE,
            model=model,
            allow_stop=ALLOW_STOP,
            long_term_memory=st.session_state.long_term_memory,
        )
        st.session_state.agent.reset()
        st.session_state.agent.build_initial_messages()
        st.session_state.running_cost = 0.0

        # Add a callback to display the agent's history after each run
        st.session_state.agent.on_iteration_end = display_new_history_callback


def get_function_bullet_point_list(model: str) -> str:
    bullet_points = ""
    for function in build_agent_functions(
        market_type=MARKET_TYPE,
        long_term_memory=st.session_state.long_term_memory,
        allow_stop=ALLOW_STOP,
        model=model,
    ):
        bullet_points += f"  - {function.__class__.__name__}\n"
    return bullet_points


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
        st.metric(
            label=f"Current balance ({MARKET_TYPE.market_class.currency})",
            value=f"{get_balance(MARKET_TYPE).amount:.2f}",
        )
    st.write(
        f"To see the agent's transaction history, click [here]({MARKET_TYPE.market_class.get_user_url(keys=keys)})."
    )

    st.divider()
    st.subheader("Configure:")
    if not agent_is_initialized():
        model = st.selectbox(
            "Model",
            ["gpt-3.5-turbo-0125", "gpt-4o-2024-05-13"],
            index=0,
        )
        if model is None:
            st.error("Please select a model.")
    else:
        model = st.selectbox(
            "Model",
            [st.session_state.agent.llm.generator.model],
            index=0,
            disabled=True,
        )
    model = check_not_none(model)

    st.divider()
    st.subheader("Built by Gnosis AI")
    st.image(
        "https://assets-global.website-files.com/63692bf32544bee8b1836ea6/63bea29e26bbff45e81efe4a_gnosis-owl-cream.png",
        width=30,
    )

    st.caption(
        "View the source code on our [github](https://github.com/gnosis/prediction-market-agent/tree/main/prediction_market_agent/agents/microchain_agent)"
    )

with st.expander(
    "Interact with an autonomous agent that uses its own balance to "
    "participate in prediction markets. More info..."
):
    st.markdown(
        "To start, click 'Run' to see the agent in action, or bootstrap the "
        "agent with your own reasoning."
    )
    st.markdown(
        f"It is equipped with the following tools to access the "
        f"[{MARKET_TYPE}]({MARKET_TYPE.market_class.base_url}) prediction "
        f"market APIs:"
    )
    st.markdown(get_function_bullet_point_list(model=model))

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
            user_reasoning = st.chat_input("Add reasoning...")

# Execution
with history_container:
    if agent_is_initialized():
        display_all_history(st.session_state.agent)
    if user_reasoning:
        maybe_initialize_agent(model)
        execute_reasoning(
            agent=st.session_state.agent,
            reasoning=user_reasoning,
            model=model,
        )
        save_last_turn_history_to_memory(st.session_state.agent)
        # Run the agent after the user's reasoning
        run_agent(
            agent=st.session_state.agent,
            iterations=int(iterations),
            model=model,
        )
    if run_agent_button:
        maybe_initialize_agent(model)
        run_agent(
            agent=st.session_state.agent,
            iterations=int(iterations),
            model=model,
        )
        save_last_turn_history_to_memory(st.session_state.agent)
    if agent_is_initialized() and has_been_run_past_initialization(
        st.session_state.agent
    ):
        # Display running cost
        if st.session_state.running_cost > 0.0:
            st.info(
                f"Running OpenAPI credits cost: ${st.session_state.running_cost:.2f}"
            )  # TODO debug why always == 0.0
