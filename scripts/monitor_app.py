import streamlit as st

from prediction_market_agent_tooling.monitor.monitor_app import MarketType, monitor_app

if __name__ == "__main__":
    st.set_page_config(layout="wide")  # Best viewed with a wide screen
    st.title(f"Monitoring")
    monitor_app(enabled_market_types=[MarketType.MANIFOLD, MarketType.OMEN])
