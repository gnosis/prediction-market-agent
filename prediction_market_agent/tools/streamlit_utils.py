import typing as t

import streamlit as st
from prediction_market_agent_tooling.loggers import logger

if t.TYPE_CHECKING:
    from loguru import Message


def loguru_streamlit_sink(log: "Message") -> None:
    record = log.record
    level = record["level"].name

    message = streamlit_escape(record["message"])

    if level == "ERROR":
        st.error(message, icon="❌")

    elif level == "WARNING":
        st.warning(message, icon="⚠️")

    else:
        st.info(message, icon="ℹ️")


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
