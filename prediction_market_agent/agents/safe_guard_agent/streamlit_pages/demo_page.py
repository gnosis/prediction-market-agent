from typing import Callable

import streamlit as st
import streamlit.components.v1 as components
from prediction_market_agent_tooling.gtypes import ChainID, ChecksumAddress
from prediction_market_agent_tooling.tools.utils import check_not_none
from web3 import Web3

from prediction_market_agent.agents.safe_guard_agent import safe_api_utils
from prediction_market_agent.agents.safe_guard_agent.safe_guard import (
    validate_safe_transaction,
)

CHAIN_ID_TO_DUNE_QUERY: dict[ChainID, str] = {
    ChainID(1): "https://dune.com/embeds/5144835/8476673",
    ChainID(100): "https://dune.com/embeds/4933998/8164678",
}


def get_demo_page(chain_id: ChainID) -> Callable[[], None]:
    def demo_page() -> None:
        st.markdown(
            """## Demo Preview

On this page, you can test out Safe Guard and see how it works in practice step by step.           
    """
        )

        with st.expander("How it works"):
            st.markdown(
                f"""1. [Optional] Create a new transaction with at least 2 signers required.
    - You can also run the validation on your (or others) historical transactions as a simulation.
2. Copy the address of your Safe wallet and paste it in the field below.
    - Or try out with the one from the Dune query.
3. Choose one of the transactions.
4. Agent will determine if the new transaction is malicious or not.
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

        if dune_query_embed := CHAIN_ID_TO_DUNE_QUERY.get(chain_id):
            with st.expander("Find Safe addresses to test on"):
                components.iframe(dune_query_embed, height=500)

        safe_address = st.text_input("Safe's address")

        if not safe_address:
            st.stop()

        # Get rid of potential token id
        *_, safe_address = safe_address.split(":")
        safe_address_checksum = Web3.to_checksum_address(safe_address)

        @st.cache_data(ttl=60)
        def get_safe_queue_multisig_cached(
            safe_address_checksum: ChecksumAddress,
        ) -> list[safe_api_utils.TransactionWithMultiSig]:
            return safe_api_utils.get_safe_queue_multisig(
                safe_address_checksum, chain_id=chain_id
            )

        @st.cache_data(ttl=60)
        def get_safe_history_multisig_cached(
            safe_address_checksum: ChecksumAddress,
        ) -> list[safe_api_utils.TransactionWithMultiSig]:
            return safe_api_utils.get_safe_history_multisig(
                safe_address_checksum, chain_id=chain_id
            )

        # Load only multisig transactions here, others are not relevant for the agent to check.
        queued_transactions = get_safe_queue_multisig_cached(safe_address_checksum)
        historical_transactions = get_safe_history_multisig_cached(
            safe_address_checksum
        )

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

        st.markdown("---")

        st.markdown("### Summary")

        with st.spinner("Validating transaction..."):
            with st.expander("Show details...", expanded=False):
                conclusion = validate_safe_transaction(
                    check_not_none(transaction_id),
                    do_sign=False,
                    do_execution=False,
                    do_reject=False,
                    do_message=False,
                    chain_id=chain_id,
                    # In the case user selected historical transaction, we want to ignore it in the history for better simulation.
                    ignore_historical_transaction_ids={check_not_none(transaction_id)},
                )

        (st.success if conclusion.all_ok else st.warning)(conclusion.summary)

    return demo_page
