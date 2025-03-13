import nest_asyncio
import streamlit as st
from prediction_market_agent_tooling.config import APIKeys
from web3 import Web3

from prediction_market_agent.agents.safe_guard_agent import safe_api_utils
from prediction_market_agent.agents.safe_guard_agent.safe_guard import (
    validate_safe_transaction,
)
from prediction_market_agent.tools.streamlit_utils import add_sink_to_logger

nest_asyncio.apply()  # Required for streamlit to work with asyncio.
st.set_page_config(layout="wide")
add_sink_to_logger()

st.markdown(
    f"""# Safe Guard -- Fraud Detection Agent for Safe

This app is a PoC of a fraud detection system for Safe wallets.
"""
)
with st.expander("How it works"):
    st.markdown(
        f"""1. In your Safe wallet, add the following address as a signer: {APIKeys().bet_from_address}
2. Create a new transaction with at least 2 signers required.
3. Copy the address of your Safe wallet and paste it in the field below.
4. Click the "Run validation" button.
5. The system will determine if the new transaction is malicious or not. 
    - If not, it will sign the transaction and execute it.
    - If yes, it will just warn you here with its reasoning. (imagine that it would have power to stop you from executing that transaction and pinging you about it!)
"""
    )
with st.expander("Examples"):
    st.markdown(
        f"""- Try to send most of your funds from the Safe to a completely new address. (powered by LLM)
- Try to send transaction with an unusually high fee. (powered by LLM)
- Try to send funds to a known blacklisted address. (powered by our own blacklist)
- Try to use a malicious address, nft, token. (powered by https://gopluslabs.io)
"""
    )

safe_address = st.text_input(
    "Your safe address (or keep the default one provided)",
    value="gno:0x1275EE969cd187Fd7B9c6Aa8339d41985eFd1EF1",
)
do_execute = st.checkbox("Execute transaction if validated", value=False)
do_reject = st.checkbox("Reject transaction if not validated", value=False)
do_message = st.checkbox("Send a message about the outcome", value=False)

if not safe_address:
    st.stop()

# Get rid of potential token id
*_, safe_address = safe_address.split(":")
safe_address_checksum = Web3.to_checksum_address(safe_address)

with st.spinner("Loading queued transactions..."):
    quened_transactions = safe_api_utils.get_safe_quened_transactions(
        safe_address_checksum
    )

transaction_id = st.selectbox(
    "Choose one of the queued transactions",
    [tx.id for tx in quened_transactions],
)


def run_validation() -> None:
    validate_safe_transaction(
        safe_address_checksum, transaction_id, do_execute, do_reject, do_message
    )


st.button(
    "Run validation",
    on_click=run_validation,
)
