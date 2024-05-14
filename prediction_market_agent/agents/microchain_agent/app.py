# Imports using asyncio (in this case mech_client) cause issues with Streamlit
from prediction_market_agent.streamlit_utils import (  # isort:skip
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
from prediction_market_agent_tooling.tools.utils import check_not_none
from streamlit_extras.bottom_container import bottom

from prediction_market_agent.agents.microchain_agent.functions import MARKET_FUNCTIONS
from prediction_market_agent.agents.microchain_agent.microchain_agent import build_agent
from prediction_market_agent.agents.microchain_agent.utils import (
    get_initial_history_length,
    has_been_run_past_initialization,
)
from prediction_market_agent.streamlit_utils import check_required_api_keys


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


def display_new_history_callback(agent: Agent) -> None:
    """
    A callback to display the agent's history in the Streamlit app after a run
    with a single interation.
    """
    history_depth = 2  # One for the user input, one for the agent's reply
    history = agent.history[-history_depth:]
    for h in history:
        st.chat_message(h["role"]).write(h["content"])


def agent_is_initialized() -> bool:
    return "agent" in st.session_state


def maybe_initialize_agent(model: str) -> None:
    # Initialize the agent
    if not agent_is_initialized():
        st.session_state.agent = build_agent(
            market_type=MarketType.OMEN, model=model, allow_stop=False
        )
        st.session_state.agent.reset()
        st.session_state.agent.build_initial_messages()
        st.session_state.running_cost = 0.0

        # Add a callback to display the agent's history after each run
        st.session_state.agent.on_iteration_end = display_new_history_callback


def get_market_function_bullet_point_list() -> str:
    bullet_points = ""
    for function in MARKET_FUNCTIONS:
        bullet_points += f"  - {function.__name__}\n"
    return bullet_points


st.set_page_config(
    layout="wide",
    page_title="Gnosis AI: Prediction Market Trader Agent",
    page_icon=":owl:",
)
st.title("Prediction Market Trader Agent")
check_required_api_keys(["OPENAI_API_KEY", "BET_FROM_PRIVATE_KEY"])

with st.sidebar:
    st.subheader("Configure:")
    if not agent_is_initialized():
        model = st.selectbox(
            "Model",
            ["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09"],
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
    "Interact with an autonomous agent that participates in prediction "
    "markets. More info..."
):
    st.markdown(
        "To start, click 'Run' to see the agent in action, or bootstrap the "
        "agent with your own reasoning."
    )
    st.markdown(
        "It is equipped with the following tools to access the "
        "[AIOmen](https://aiomen.eth.limo/) prediction market APIs:"
    )
    st.markdown(get_market_function_bullet_point_list())

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
    if run_agent_button:
        maybe_initialize_agent(model)
        run_agent(
            agent=st.session_state.agent,
            iterations=int(iterations),
            model=model,
        )
    if agent_is_initialized() and has_been_run_past_initialization(
        st.session_state.agent
    ):
        # Display running cost
        if st.session_state.running_cost > 0.0:
            st.info(
                f"Running OpenAPI credits cost: ${st.session_state.running_cost:.2f}"
            )  # TODO debug why always == 0.0
