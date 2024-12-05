import asyncio
from enum import Enum

import streamlit as st

from prediction_market_agent.agents.blockchain_coding_agent.agents import get_agent_team
from prediction_market_agent.agents.blockchain_coding_agent.prompts import (
    message_read,
    message_write,
)
from prediction_market_agent.agents.blockchain_coding_agent.streamlit_console import (
    streamlit_console,
)


def is_user_prompt_not_null(prompt: str) -> bool:
    return prompt != ""


class UserPrompt(str, Enum):
    READ_CONDITIONAL_TOKENS_BALANCE = message_read
    WRITE_USDC_APPROVAL = message_write


agent_team = get_agent_team()

st.title("Blockchain agent")


with st.container():
    st.subheader("How to interact with the agent")
    st.markdown(
        """You have 2 options:
        
    1. Enter a custom prompt in the text input OR
    2. Load one of the predefined prompts (Read, Write).
    """
    )
    st.divider()
    user_input = st.text_input(label="Enter a new prompt if so desired", key="prompt")
    prompt_option = st.selectbox(
        "Select a prompt option",
        (UserPrompt.READ_CONDITIONAL_TOKENS_BALANCE, UserPrompt.WRITE_USDC_APPROVAL),
        placeholder="Select an option",
        index=0,
        key="prompt_option",
        disabled=is_user_prompt_not_null(st.session_state.prompt),
    )

    submit_button = st.button("Submit", key="button")

    if submit_button:
        message = st.session_state.prompt_option
        if st.session_state.prompt is not None and st.session_state.prompt != "":
            message = user_input
        stream = agent_team.run_stream(task=message)
        asyncio.run(streamlit_console(stream))
