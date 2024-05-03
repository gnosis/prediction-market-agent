"""
PYTHONPATH=. streamlit run prediction_market_agent/agents/microchain_agent/app.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, isntead of just this one.
"""
import streamlit as st

from prediction_market_agent.utils import APIKeys, streamlit_asyncio_event_loop_hack

# Imports using asyncio (in this case mech_client) cause issues with Streamlit
streamlit_asyncio_event_loop_hack()

from microchain import Agent
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.costs import openai_costs

from prediction_market_agent.agents.microchain_agent.microchain_agent import get_agent


def check_api_keys() -> None:
    keys = APIKeys()
    if not keys.OPENAI_API_KEY:
        st.error("No OpenAI API Key provided via env var/secret.")
        st.stop()
    elif not keys.BET_FROM_PRIVATE_KEY:
        st.error("No wallet private key provided via env var/secret.")
        st.stop()


def run_agent(agent: Agent, iterations: int, model: str) -> None:
    with openai_costs(model) as costs:
        with st.spinner("Agent is running..."):
            for _ in range(iterations):
                agent.run(iterations=1, resume=True)
        st.session_state.running_cost += costs.cost


def execute_reasoning(agent: Agent, reasoning: str, model: str) -> None:
    with openai_costs(model) as costs:
        agent.execute_command(f'Reasoning("{reasoning}")')
        st.session_state.running_cost += costs.cost


st.set_page_config(layout="wide")
st.title("Microchain Agent")
st.write(
    "This agent participates in prediction markets with a high degree of autonomy. It is equipped with tools to access the prediction market APIs, and can use its own reasoning to guide its betting strategy."
)

# Ask the user to choose a model
model = st.selectbox(
    "Model",
    ["gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-0125"],
    index=0,
)
check_api_keys()

# Initialize the agent
if "agent" not in st.session_state:
    st.session_state.agent = get_agent(market_type=MarketType.OMEN, model=model)
    st.session_state.agent.bootstrap = [
        'Reasoning("I need to reason step by step. Start by assessing my current position and balance.")'
    ]
    st.session_state.agent.reset()
    st.session_state.agent.build_initial_messages()
    st.session_state.running_cost = 0.0
    st.info(
        "Start by clicking 'Run' to see the agent in action. Alternatively bootstrap the agent with your own reasoning before running."
    )


user_reasoning = st.text_input("Reasoning")
if st.button("Intervene by adding reasoning"):
    execute_reasoning(
        agent=st.session_state.agent,
        reasoning=user_reasoning,
        model=model,
    )

# Allow the user to run the
iterations = st.number_input(
    "Run iterations",
    value=1,
    step=1,
    min_value=1,
    max_value=100,
)
if st.button("Run the agent"):
    run_agent(
        agent=st.session_state.agent,
        iterations=int(iterations),
        model=model,
    )

# Display the agent's history
history = st.session_state.agent.history[3:]  # Skip the initial messages
for h in history:
    st.chat_message(h["role"]).write(h["content"])

# Display running cost
# st.info(f"Running OpenAPI credits cost: ${st.session_state.running_cost:.2f}")  # TODO debug why always == 0.0
