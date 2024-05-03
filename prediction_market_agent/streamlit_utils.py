import asyncio

import streamlit as st

from prediction_market_agent.utils import APIKeys


def streamlit_asyncio_event_loop_hack() -> None:
    """
    This function is a hack to make Streamlit work with asyncio event loop.
    See https://github.com/streamlit/streamlit/issues/744
    """

    def get_or_create_eventloop() -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_event_loop()
        except RuntimeError as ex:
            if "There is no current event loop in thread" in str(ex):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return asyncio.get_event_loop()
            else:
                raise ex

    loop = get_or_create_eventloop()
    asyncio.set_event_loop(loop)


def check_required_api_keys(required_keys: list[str]) -> None:
    keys = APIKeys()
    has_missing_keys = False
    for key in required_keys:
        if not getattr(keys, key):
            st.error(f"Environment variable for key {key} has not been set.")
            has_missing_keys = True
    if has_missing_keys:
        st.stop()
