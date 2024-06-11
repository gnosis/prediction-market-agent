"""
PYTHONPATH=. streamlit run prediction_market_agent/agents/microchain_agent/app.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, instead of just this one.
"""
import random
from streamlit.delta_generator import DeltaGenerator

# Imports using asyncio (in this case mech_client) cause issues with Streamlit
from prediction_market_agent.agents.microchain_agent.memory import LongTermMemory
from prediction_market_agent.agents.microchain_agent.prompt_handler import PromptHandler
from prediction_market_agent.agents.microchain_agent.prompts import (
    SYSTEM_PROMPTS,
    SystemPromptChoice,
)
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
    maybe_init()
    with openai_costs(model) as costs:
        with st.spinner("Agent is running..."):
            for _ in range(iterations):
                agent.run(iterations=1, resume=True)
                st.session_state.system_prompt_test = random.randint(0, 100)
        st.session_state.running_cost += costs.cost


def on_click_user_reasoning() -> None:
    maybe_initialize_agent(model, system_prompt, bootstrap)

    with openai_costs(model) as costs:
        st.session_state.agent.execute_command(f'Reasoning("{user_reasoning}")')
        st.session_state.running_cost += costs.cost

    save_last_turn_history_to_memory(st.session_state.agent)
    # Run the agent after the user's reasoning
    run_agent(
        agent=st.session_state.agent,
        iterations=int(iterations),
        model=model,
    )


def chat_message(role: str, content: str) -> None:
    # Return of functions is stringified, so we need to check for "None" string.

    if content != "None":
        st.chat_message(role).write(content)


def display_all_history(agent: Agent) -> None:
    """
    Display the agent's history in the Streamlit app.
    """
    # Skip the initial messages
    history = agent.history[get_initial_history_length(agent) :]

    for h in history:
        chat_message(h["role"], h["content"])


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
        chat_message(h["role"], h["content"])
        chat_message(h["role"], h["content"])


def maybe_initialize_long_term_memory() -> None:
    # Initialize the db storage
    if not "long_term_memory" in st.session_state:
        st.session_state.long_term_memory = LongTermMemory(
            LongTermMemoryTaskIdentifier.MICROCHAIN_AGENT_STREAMLIT
        )


def maybe_initialize_prompt_handler() -> None:
    if not "prompt_handler" in st.session_state:
        st.session_state.prompt_handler = PromptHandler()


def agent_is_initialized() -> bool:
    return "agent" in st.session_state


def save_last_turn_history_to_memory(agent: Agent) -> None:
    last_turn_history = get_history_from_last_turn(agent)
    st.session_state.long_term_memory.save_history(last_turn_history)


def fetch_latest_prompt():
    if st.session_state.get("load_historical_prompt"):
        prompt_handler: PromptHandler = st.session_state.prompt_handler
        historical_prompt = prompt_handler.fetch_latest_prompt()
        return historical_prompt


def maybe_initialize_agent(model: str, system_prompt: str, bootstrap: str) -> None:
    # Initialize the agent
    maybe_init()
    if not agent_is_initialized():
        historical_prompt = fetch_latest_prompt()
        system_prompt = (
            historical_prompt.prompt if historical_prompt is not None else system_prompt
        )

        st.session_state.agent = build_agent(
            market_type=MARKET_TYPE,
            model=model,
            system_prompt=system_prompt,
            bootstrap=bootstrap,
            allow_stop=ALLOW_STOP,
            long_term_memory=st.session_state.long_term_memory,
            load_historical_prompt=st.session_state.get("load_historical_prompt"),
        )
        st.session_state.agent.reset()
        st.session_state.agent.build_initial_messages()
        st.session_state.running_cost = 0.0

        # Add a callback to display the agent's history after each run
        # st.session_state.agent.on_iteration_end = display_new_history_callback


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


st.set_page_config(
    layout="wide",
    page_title="Gnosis AI: Prediction Market Trader Agent",
    page_icon=":owl:",
)
st.title("Prediction Market Trader Agent")
# with st.sidebar:
#    streamlit_login()
check_required_api_keys(["OPENAI_API_KEY", "BET_FROM_PRIVATE_KEY"])


def maybe_init():
    maybe_initialize_long_term_memory()


with st.sidebar:
    st.subheader("Agent Info:")
    with st.container(border=True):
        st.metric(
            label=f"Current balance ({MARKET_TYPE.market_class.currency})",
            value=f"{get_balance(MARKET_TYPE).amount:.2f}",
        )
    st.write(
        f"To see the agent's transaction history, click [here]({MARKET_TYPE.market_class.get_user_url(keys=APIKeys())})."
    )

    st.divider()
    st.subheader("Configure:")
    if not agent_is_initialized():
        model = st.selectbox(
            "Model",
            [
                "gpt-4-turbo",
                "gpt-3.5-turbo-0125",
                "gpt-4o-2024-05-13",
            ],
            index=0,
        )
        if model is None:
            st.error("Please select a model.")
        st.session_state.system_prompt_select = st.selectbox(
            "Initial memory",
            [p.value for p in SystemPromptChoice],
            index=0,
        )
        st.session_state.load_historical_prompt = st.toggle("Load historical prompt")

    else:
        model = st.selectbox(
            "Model",
            [st.session_state.agent.llm.generator.model],
            index=0,
            disabled=True,
        )
        st.selectbox(
            "Initial memory",
            [st.session_state.system_prompt_select],
            index=0,
            disabled=True,
        )

    model = check_not_none(model)

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

with st.expander(
    "Interact with an autonomous agent that uses its own balance to "
    "participate in prediction markets. More info..."
):
    st.markdown(
        "To start, click 'Run' to see the agent in action, or bootstrap the "
        "agent with your own reasoning."
    )
    st.markdown(f"It is equipped with the following tools:")
    if agent_is_initialized():
        st.markdown(
            get_function_bullet_point_list(agent=st.session_state.agent, model=model)
        )
    else:
        st.markdown("The agent is not initialized yet.")

with st.expander("Agent's current system prompt"):
    if agent_is_initialized():
        st.markdown(st.session_state.agent.system_prompt)
    else:
        st.markdown("The agent is not initialized yet.")

    st.markdown(f"check {st.session_state.get('system_prompt_test')}")

with st.expander("Agent's current bootstrap"):
    if agent_is_initialized():
        st.markdown(st.session_state.agent.bootstrap)
    else:
        st.markdown("The agent is not initialized yet.")


def on_click_run_agent():
    # st.session_state.system_prompt_test = random.randint(1000, 2000)
    maybe_initialize_agent(model, system_prompt, bootstrap)

    run_agent(
        agent=st.session_state.agent,
        iterations=int(iterations),
        model=model,
    )
    save_last_turn_history_to_memory(st.session_state.agent)


# Placeholder for the agent's history
with st.container() as history_container:
    if "agent" in st.session_state:
        display_all_history(st.session_state.agent)

    # costs
    if agent_is_initialized() and has_been_run_past_initialization(
        st.session_state.agent
    ):
        # Display running cost
        if st.session_state.running_cost > 0.0:
            st.info(
                f"Running OpenAPI credits cost: ${st.session_state.running_cost:.2f}"
            )  # TODO debug why always == 0.0

# Interactive elements
with bottom():
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 8])
        with col1:
            st.button("Run the agent", on_click=on_click_run_agent)
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
            user_reasoning = st.chat_input(
                "Add reasoning...", on_submit=on_click_user_reasoning
            )
