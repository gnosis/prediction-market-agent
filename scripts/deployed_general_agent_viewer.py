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

import pandas as pd
import plotly.express as px
import streamlit as st
from prediction_market_agent_tooling.gtypes import PrivateKey
from prediction_market_agent_tooling.markets.markets import MarketType
from pydantic_settings import BaseSettings, SettingsConfigDict

from prediction_market_agent.agents.microchain_agent.memory import DatedChatHistory
from prediction_market_agent.agents.microchain_agent.microchain_agent import (
    SupportedModel,
    build_agent,
)
from prediction_market_agent.agents.microchain_agent.prompts import FunctionsConfig
from prediction_market_agent.agents.microchain_agent.utils import (
    get_function_useage_from_history,
    get_total_asset_value,
)
from prediction_market_agent.agents.utils import AgentIdentifier
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.utils import APIKeys


class DeployedGeneralAgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    STARTING_BALANCE: float
    AGENT_IDENTIFIER_TO_PRIVATE_KEY: dict[AgentIdentifier, PrivateKey]

    @property
    def starting_balance(self) -> float:
        return self.STARTING_BALANCE

    @property
    def agent_identifier_to_private_key(self) -> dict[AgentIdentifier, PrivateKey]:
        return self.AGENT_IDENTIFIER_TO_PRIVATE_KEY

    @property
    def available_agents(self) -> list[AgentIdentifier]:
        return list(self.agent_identifier_to_private_key.keys())

    def to_api_keys(self, identifier: AgentIdentifier) -> APIKeys:
        return APIKeys(
            BET_FROM_PRIVATE_KEY=self.agent_identifier_to_private_key[identifier]
        )


MARKET_TYPE = MarketType.OMEN
currency = MARKET_TYPE.market_class.currency

st.set_page_config(
    layout="wide",
    page_title="Gnosis AI: Deployed Prediction Market Trader Agent",
    page_icon=":owl:",
)
st.title("Deployed Trader Agent Viewer")

settings = DeployedGeneralAgentSettings()

with st.sidebar:
    task_description = AgentIdentifier(
        st.selectbox(
            label="Select the agent",
            options=[x.value for x in settings.available_agents],
            index=0,
        )
    )
    if task_description is None:
        st.error("Please select an agent.")
        st.stop()

keys = settings.to_api_keys(task_description)
starting_balance = settings.starting_balance

with st.sidebar:
    st.subheader("App info:")
    st.write(
        "This is a viewer for monitoring a deployed 'general agent' that aims "
        "to profit from trading on prediction markets. Observe its behavior "
        "as it tries to learn over time."
    )
    with st.container(border=True):
        st.metric(label=f"Agent Wallet Address", value=keys.bet_from_address)
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
        "View the source code on our [github](https://github.com/gnosis/prediction-market-agent/tree/main/scripts/deployed_agent_viewer.py)."
    )

long_term_memory = LongTermMemoryTableHandler(task_description=task_description)
chat_history = DatedChatHistory.from_long_term_memory(long_term_memory=long_term_memory)
sessions = chat_history.cluster_by_session()

total_asset_value = get_total_asset_value(keys, MARKET_TYPE).amount
roi = (total_asset_value - starting_balance) * 100 / starting_balance

with st.container(border=True):
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.metric(
        "Starting Time",
        (
            "N/A"
            if chat_history.is_empty
            else chat_history.start_time.strftime("%Y-%m-%d %H:%M:%S")
        ),
    )
    col2.metric("Number of iterations", chat_history.iterations)
    col3.metric("Starting Balance", f"{starting_balance:.2f} {currency}")
    col4.metric(
        "Total Asset Value",
        f"{total_asset_value:.2f} {currency}",
        delta=f"{roi:.2f}% ROI",
    )

st.subheader("Agent Logs")
for session in sessions:
    expander_str = (
        f"{session.start_time.strftime('%Y-%m-%d %H:%M:%S')} - "
        f"{session.end_time.strftime('%Y-%m-%d %H:%M:%S')}    |    "
        f":blue[Iterations: {session.iterations}]"
    )
    with st.expander(expander_str):
        display_chat_history(session)

st.subheader("Tool Usage")
agent = build_agent(
    market_type=MARKET_TYPE,
    model=SupportedModel.gpt_4o,  # placeholder, not used
    keys=keys,  # placeholder, not used
    unformatted_system_prompt="foo",  # placeholder, not used
    allow_stop=True,
    long_term_memory=long_term_memory,
    functions_config=FunctionsConfig(
        include_trading_functions=True,  # placeholder, not used
        include_learning_functions=True,  # placeholder, not used
        include_universal_functions=True,  # placeholder, not used
    ),
)
tab1, tab2 = st.tabs(["Overall", "Per-Session"])
usage_count_col_name = "Usage Count"
tool_name_col_name = "Tool Name"
with tab1:
    function_use = get_function_useage_from_history(
        chat_history=chat_history.to_undated_chat_history(),
        agent=agent,
    )
    st.bar_chart(
        function_use,
        horizontal=True,
        y_label=tool_name_col_name,
        x_label=usage_count_col_name,
    )
with tab2:
    session_start_times = [session.start_time for session in sessions]
    session_function_use: dict[datetime, pd.DataFrame] = {}
    for session in sessions:
        session_function_use[session.start_time] = get_function_useage_from_history(
            chat_history=session,
            agent=agent,
        )

    # Concatenate the per-session DataFrames by date for displaying in a heatmap
    heatmap_data = {}
    for date, df in session_function_use.items():
        date_str = date.strftime("%Y-%m-%d %H:%M:%S")
        heatmap_data[date_str] = df[usage_count_col_name]
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.index.name = tool_name_col_name
    dates = heatmap_df.columns.tolist()
    tool_names = heatmap_df.index.tolist()
    if not heatmap_df.empty:
        fig = px.imshow(
            heatmap_df.values.tolist(),
            labels=dict(x="Session", y=tool_name_col_name, color=usage_count_col_name),
            x=dates,
            y=tool_names,
            aspect="auto",
        )
        fig.update_xaxes(side="top")
        fig.update_yaxes(tickvals=list(range(len(tool_names))), ticktext=tool_names)
        st.plotly_chart(fig, theme="streamlit")
