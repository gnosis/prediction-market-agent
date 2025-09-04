import streamlit
import streamlit as st

from prediction_market_agent.agents.microchain_agent.memory import (
    ChatHistory,
    ChatMessage,
)


def display_chat_message(chat_message: ChatMessage) -> None:
    # Return of functions is stringified, so we need to check for "None" string.
    if chat_message.content != "None":
        st.chat_message(chat_message.role).write(chat_message.content)


def display_chat_history(chat_history: ChatHistory) -> None:
    for m in chat_history.chat_messages:
        display_chat_message(m)
