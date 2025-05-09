import asyncio
import typing as t

import streamlit as st
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.caches.db_cache import DB_CACHE_LOG_PREFIX

from prediction_market_agent.agents.microchain_agent.memory import (
    ChatHistory,
    ChatMessage,
)
from prediction_market_agent.utils import APIKeys

if t.TYPE_CHECKING:
    from loguru import Message


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


def loguru_streamlit_sink(
    log: "Message", expander_if_longer_than: int = 500, include_in_expander: int = 50
) -> None:
    record = log.record
    level = record["level"].name

    message = streamlit_escape(record["message"])

    # Ignore certain messages that aren't interesting for Streamlit user, but are in the production logs.
    if any(x in message for x in [DB_CACHE_LOG_PREFIX]):
        return

    if level == "ERROR":
        st_func = st.error
        st_icon = "❌"

    elif level == "WARNING":
        st_func = st.warning
        st_icon = "⚠️"

    elif level == "SUCCESS":
        st_func = st.success
        st_icon = "✅"

    elif level == "DEBUG":
        st_func = None
        st_icon = None

    else:
        st_func = st.info
        st_icon = "ℹ️"

    if st_func is None:
        pass

    elif len(message) > expander_if_longer_than:
        with st.expander(f"[Expand to see more] {message[:include_in_expander]}..."):
            st_func(message, icon=st_icon)

    else:
        st_func(message, icon=st_icon)


@st.cache_resource
def add_sink_to_logger() -> None:
    """
    Adds streamlit as a sink to the loguru, so any loguru logs will be shown in the streamlit app.

    Needs to be behind a cache decorator, so it only runs once per streamlit session (otherwise we would see duplicated messages).
    """
    logger.add(loguru_streamlit_sink)


def streamlit_escape(message: str) -> str:
    """
    Escapes the string for streamlit writes.
    """
    # Replace escaped newlines with actual newlines.
    message = message.replace("\\n", "\n")
    # Fix malformed dollar signs in the messages.
    message = message.replace("$", "\$")

    return message


def display_chat_message(chat_message: ChatMessage) -> None:
    # Return of functions is stringified, so we need to check for "None" string.
    if chat_message.content != "None":
        st.chat_message(chat_message.role).write(chat_message.content)


def display_chat_history(chat_history: ChatHistory) -> None:
    for m in chat_history.chat_messages:
        display_chat_message(m)
