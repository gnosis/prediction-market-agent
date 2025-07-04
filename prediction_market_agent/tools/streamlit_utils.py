import asyncio
import os
import typing as t

import streamlit
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

STREAMLIT_SINK_EXPLICIT_FLAG = "streamlit"


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


def get_loguru_streamlit_sink(
    explicit: bool,
    expander_if_longer_than: int | None,
    include_in_expander: int = 50,
) -> t.Callable[["Message"], None]:
    def loguru_streamlit_sink(log: "Message") -> None:
        record = log.record
        level = record["level"].name

        message = streamlit_escape(record["message"])

        # Ignore certain messages that aren't interesting for Streamlit user, but are in the production logs.
        if any(x in message for x in [DB_CACHE_LOG_PREFIX]):
            return

        if explicit and not record["extra"].get(STREAMLIT_SINK_EXPLICIT_FLAG, False):
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

        elif (
            expander_if_longer_than is not None
            and len(message) > expander_if_longer_than
        ):
            with st.expander(
                f"[Expand to see more] {message[:include_in_expander]}..."
            ):
                st_func(message, icon=st_icon)

        else:
            st_func(message, icon=st_icon)

    return loguru_streamlit_sink


@st.cache_resource
def add_sink_to_logger(
    explicit: bool = False, expander_if_longer_than: int | None = 500
) -> None:
    """
    Adds streamlit as a sink to the loguru, so any loguru logs will be shown in the streamlit app.

    Needs to be behind a cache decorator, so it only runs once per streamlit session (otherwise we would see duplicated messages).

    If `explicit` is set to True, only logged messages with extra attribute `streamlit` will be shown in the streamlit app.
    """
    logger.add(
        get_loguru_streamlit_sink(
            explicit=explicit, expander_if_longer_than=expander_if_longer_than
        )
    )


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


def dict_to_point_list(d: dict[str, t.Any], indent: int = 0) -> str:
    """
    Helper method to convert nested dicts to a bullet point list.
    """
    lines = []
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}- {k}:")
            lines.append(dict_to_point_list(v, indent + 1))
        elif isinstance(v, list):
            lines.append(f"{prefix}- {k}:")
            for idx, item in enumerate(v):
                if isinstance(item, dict):
                    lines.append(f"{prefix}  - item {idx}:")
                    lines.append(dict_to_point_list(item, indent + 2))
                else:
                    lines.append(f"{prefix}  - item {idx}: {item}")
        else:
            lines.append(f"{prefix}- {k}: {v}")
    return "\n".join(lines)


def customize_index_html(head_content: str) -> None:
    """
    Unfortunatelly, Streamlit doesn't allow to update HTML content of the main index.html file, any component that allows passing of HTML will render it in iframe.
    That's unusable for analytics tools like Posthog.

    This is workaround that patches their index.html file directly in their package, found in https://stackoverflow.com/questions/70520191/how-to-add-the-google-analytics-tag-to-website-developed-with-streamlit/78992559#78992559.

    There is also an issue that tracks this feature (open since 2023): https://github.com/streamlit/streamlit/issues/6140
    """
    streamlit_package_dir = os.path.dirname(streamlit.__file__)
    index_path = os.path.join(streamlit_package_dir, "static", "index.html")

    with open(index_path, "r") as f:
        index_html = f.read()

    if head_content not in index_html:
        # Add the custom content to the head
        index_html = index_html.replace("</head>", f"{head_content}</head>")

        # Replace the <title> tag
        index_html = index_html.replace(
            "<title>Streamlit</title>", "<title>Savantly is cool</title>"
        )

        with open(index_path, "w") as f:
            f.write(index_html)

        logger.info("index.html injected with custom code.")

    else:
        logger.info("index.html injection skipped because it's already present.")
