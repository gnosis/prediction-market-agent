"""
PYTHONPATH=. streamlit run scripts/monitor_app.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, instead of just this one.
"""

import streamlit as st
from prediction_market_agent_tooling.monitor.monitor_app import MarketType, monitor_app
from prediction_market_agent_tooling.tools.streamlit_user_login import streamlit_login

if __name__ == "__main__":
    st.set_page_config(layout="wide")  # Best viewed with a wide screen
    with st.sidebar:
        streamlit_login()
    st.title("Monitoring")
    monitor_app(enabled_market_types=[MarketType.MANIFOLD, MarketType.OMEN])
