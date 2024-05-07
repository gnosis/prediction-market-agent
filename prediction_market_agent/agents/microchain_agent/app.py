# Imports using asyncio (in this case mech_client) cause issues with Streamlit
from prediction_market_agent.streamlit_utils import (  # isort:skip
    streamlit_asyncio_event_loop_hack,
)

streamlit_asyncio_event_loop_hack()

import streamlit as st
from microchain import Agent
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.costs import openai_costs
from prediction_market_agent_tooling.tools.utils import check_not_none

from prediction_market_agent.agents.microchain_agent.microchain_agent import get_agent
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
        st.session_state.agent = get_agent(market_type=MarketType.OMEN, model=model)
        st.session_state.agent.reset()
        st.session_state.agent.build_initial_messages()
        st.session_state.running_cost = 0.0

        # Add a callback to display the agent's history after each run
        st.session_state.agent.on_iteration_end = display_new_history_callback


st.set_page_config(layout="wide")
st.title("Microchain Agent")
st.write(
    "This agent participates in prediction markets with a high degree of "
    "autonomy. It is equipped with tools to access the prediction market APIs, "
    "and can use its own reasoning to guide its betting strategy."
)

check_required_api_keys(["OPENAI_API_KEY", "BET_FROM_PRIVATE_KEY"])

# Ask the user to choose a model
if not agent_is_initialized():
    model = st.selectbox(
        "Model",
        ["gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-0125"],
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

# Interactive settings
user_reasoning = st.text_input("Reasoning")
add_reasoning_button = st.button("Intervene by adding reasoning")
iterations = st.number_input(
    "Run iterations",
    value=1,
    step=1,
    min_value=1,
    max_value=100,
)
run_agent_button = st.button("Run the agent")

# Execution
if agent_is_initialized():
    display_all_history(st.session_state.agent)
if add_reasoning_button:
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
if agent_is_initialized() and has_been_run_past_initialization(st.session_state.agent):
    st.info(
        "Run complete. Click 'Run' to allow the agent to continue, or add your "
        "own reasoning."
    )
    # Display running cost
    # st.info(f"Running OpenAPI credits cost: ${st.session_state.running_cost:.2f}")  # TODO debug why always == 0.0
else:
    st.info(
        "Start by clicking 'Run' to see the agent in action. Alternatively "
        "bootstrap the agent with your own reasoning before running."
    )
