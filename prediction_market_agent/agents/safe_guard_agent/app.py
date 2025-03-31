import nest_asyncio
import streamlit as st
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.tools.utils import check_not_none
from web3 import Web3

from prediction_market_agent.agents.safe_guard_agent import safe_api_utils
from prediction_market_agent.agents.safe_guard_agent.safe_guard import (
    validate_safe_transaction,
)
from prediction_market_agent.agents.safe_guard_agent.safe_utils import get_safe
from prediction_market_agent.tools.streamlit_utils import add_sink_to_logger

nest_asyncio.apply()  # Required for streamlit to work with asyncio.
st.set_page_config(layout="wide")
add_sink_to_logger()

keys = APIKeys()

st.markdown(
    f"""# Safe Guard -- Fraud Detection Agent for Safe

This app is a PoC of a fraud detection system for Safe wallets.
"""
)
with st.expander("How it works"):
    st.markdown(
        f"""1. [Optional] In your Safe wallet, add the following address as a signer: {keys.bet_from_address}
    - This step is optional, but some functionality is limited without it.
2. [Optional] Create a new transaction with at least 2 signers required.
    - You can also run the validation on your historical transactions as a simulation.
3. Copy the address of your Safe wallet and paste it in the field below.
    - Or try out with the one filled by default.
4. Choose one of the pending transactions.
5. Click the "Run validation" button.
6. Agent will determine if the new transaction is malicious or not. 
    - If not, it will sign the transaction and maybe execute it. (if threshold is met)
    - If yes, it will create a rejection transaction.
7. Agent will also send you a signed message into Safe, with all the available details.
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

if not safe_address:
    st.stop()

# Get rid of potential token id
*_, safe_address = safe_address.split(":")
safe_address_checksum = Web3.to_checksum_address(safe_address)
safe = get_safe(safe_address_checksum)

is_owner = safe.retrieve_is_owner(keys.bet_from_address)
threshold = safe.retrieve_threshold()

if threshold == 1:
    st.warning("This Safe has threshold set to 1.")

do_execute = st.checkbox(
    "Execute transaction if validated (possible only for Safe's owners)",
    value=False,
    disabled=not is_owner,
)
do_reject = st.checkbox(
    "Reject transaction if not validated (possible only for Safe's owners)",
    value=False,
    disabled=not is_owner,
)
do_message = st.checkbox(
    "Send a message about the outcome (possible only for Safe's owners)",
    value=False,
    disabled=not is_owner,
)


@st.cache_data(ttl=60)
def get_safe_queue_multisig_cached(
    safe_address_checksum: ChecksumAddress,
) -> list[safe_api_utils.TransactionWithMultiSig]:
    return safe_api_utils.get_safe_queue_multisig(safe_address_checksum)


@st.cache_data(ttl=60)
def get_safe_history_multisig_cached(
    safe_address_checksum: ChecksumAddress,
) -> list[safe_api_utils.TransactionWithMultiSig]:
    return safe_api_utils.get_safe_history_multisig(safe_address_checksum)


# Load only multisig transactions here, others are not relevant for the agent to check.
queued_transactions = get_safe_queue_multisig_cached(safe_address_checksum)
historical_transactions = get_safe_history_multisig_cached(safe_address_checksum)

if not queued_transactions and not historical_transactions:
    st.error("No multi-sig transactions found for the Safe address.")
    st.stop()

queue_col, hist_col = st.columns(2)
with queue_col:
    queue_transaction_id = st.selectbox(
        f"Choose one of the queued transactions ({len(queued_transactions)} available)",
        [tx.id for tx in queued_transactions],
        index=None,
    )
with hist_col:
    hist_transaction_id = st.selectbox(
        f"Or historical transaction for simulation (available if no queue transaction is selected, {len(historical_transactions)} available)",
        [tx.id for tx in historical_transactions],
        index=None,
        disabled=queue_transaction_id is not None,
    )

transaction_id = queue_transaction_id or hist_transaction_id
if transaction_id is None:
    st.stop()

transaction = check_not_none(
    next(
        (
            tx
            for tx in queued_transactions + historical_transactions
            if tx.id == transaction_id
        ),
        None,
    )
)


def run_validation() -> None:
    validate_safe_transaction(
        safe_address_checksum,
        transaction,
        do_execute,
        do_reject,
        do_message,
        # In the case user selected historical transaction, we want to ignore it in the history for better simulation.
        ignore_historical_transaction_ids={transaction.id},
    )


st.button(
    "Run validation",
    on_click=run_validation,
)
