"""
PYTHONPATH=. streamlit run deployed_agent_viewer.py

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

patch_sqlite3()


from datetime import datetime

import streamlit as st
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.memory import (
    DatedChatHistory,
    LongTermMemory,
)
from prediction_market_agent.agents.microchain_agent.utils import get_total_asset_value
from prediction_market_agent.agents.utils import AgentIdentifier
from prediction_market_agent.tools.streamlit_utils import check_required_api_keys
from prediction_market_agent.utils import APIKeys

START_TIME = datetime(2024, 5, 29, 14, 8, 45)  # TODO make configurable
STARTING_BALANCE = 5  # xDai # TODO make configurable
MARKET_TYPE = MarketType.OMEN
currency = MARKET_TYPE.market_class.currency

st.set_page_config(
    layout="wide",
    page_title="Gnosis AI: Deployed Prediction Market Trader Agent",
    page_icon=":owl:",
)
st.title("Deployed Trader Agent Viewer")

check_required_api_keys(["BET_FROM_PRIVATE_KEY"])
keys = APIKeys()

with st.sidebar:
    st.subheader("App info:")
    st.write(
        "This is a viewer for monitoring a deployed 'general agent' that aims "
        "to profit from trading on prediction markets. Observe its behavior "
        "as it tries to learn over time."
    )
    with st.container(border=True):
        st.metric(
            label=f"Agent Wallet Address",
            value=f"{keys.bet_from_address}",
        )
    st.write(
        f"To see the agent's transaction history, click [here]({MARKET_TYPE.market_class.get_user_url(keys=keys)})."
    )

    st.divider()
    st.subheader("Built by Gnosis AI")
    st.image(
        "https://assets-global.website-files.com/63692bf32544bee8b1836ea6/63bea29e26bbff45e81efe4a_gnosis-owl-cream.png",
        width=30,
    )

    st.caption(
        "View the source code on our [github](https://github.com/gnosis/prediction-market-agent/tree/main/scrips/deployed_agent_viewer.py)."
    )


task_description = AgentIdentifier.microchain_task_from_market(MARKET_TYPE)
long_term_memory = LongTermMemory(task_description=task_description)
chat_history = DatedChatHistory.from_long_term_memory(
    long_term_memory=long_term_memory,
    from_=START_TIME,
)
sessions = chat_history.cluster_by_datetime(max_minutes_between_messages=30)

total_asset_value = get_total_asset_value(MARKET_TYPE).amount
roi = (total_asset_value - STARTING_BALANCE) * 100 / STARTING_BALANCE

with st.container(border=True):
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col1.metric("Starting Time", START_TIME.strftime("%Y-%m-%d %H:%M:%S"))
    col2.metric("Number of iterations", len(chat_history.chat_messages))
    col3.metric("Starting Balance", f"{STARTING_BALANCE:.2f} {currency}")
    col4.metric(
        "Total Asset Value",
        f"{total_asset_value:.2f} {currency}",
        delta=f"{roi:.2f}% ROI",
    )

st.subheader("Agent Logs")
for session in sessions:
    with st.expander(f"{session.start_time} - {session.end_time}"):
        display_chat_history(session)

# TODO add tool call frequency stats