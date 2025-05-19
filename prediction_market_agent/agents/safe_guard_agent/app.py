import nest_asyncio
import streamlit as st
from prediction_market_agent_tooling.gtypes import ChainID

from prediction_market_agent.agents.safe_guard_agent.streamlit_pages import (
    agents_page,
    api_page,
    demo_page,
)
from prediction_market_agent.tools.streamlit_utils import add_sink_to_logger

nest_asyncio.apply()  # Required for streamlit to work with asyncio.
st.set_page_config(layout="wide")
add_sink_to_logger(explicit=True)

st.markdown(
    f"""# Safe Guard -- Fraud Detection Agent for Safe 

This app is entry point for the Safe Guard. 

On the left panel, you can select pages where you can:

- Test out validation in this demo, without adding agent as a signer or doing any real transactions
- Call the validation via endpoint and see the documentation for it
- See the list of deployed agents that can be added as signers to your Safe
"""
)

chain_name, chain_id = st.selectbox(
    "Select chain",
    [("Ethereum", ChainID(1)), ("Gnosis", ChainID(100))],
    index=1,
    format_func=lambda x: x[0],
)

pages = [
    st.Page(demo_page.get_demo_page(chain_id=chain_id), title="Demo", url_path="demo"),
    st.Page(api_page.get_api_page(chain_id=chain_id), title="API", url_path="api"),
    st.Page(
        agents_page.get_agents_page(chain_id=chain_id),
        title="Agents",
        url_path="agents",
    ),
]
pg = st.navigation(pages)
pg.run()
