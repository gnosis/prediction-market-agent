import sys
from typing import AsyncGenerator, TypeVar

import streamlit as st
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import AgentMessage
from autogen_agentchat.task._console import Console

T = TypeVar("T", bound=TaskResult | Response)


class PrintRedirector:
    @staticmethod
    def write(text):
        st.info(text)


# Context manager to temporarily redirect sys.stdout
class RedirectStdoutToPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = PrintRedirector()

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout


async def streamlit_console(stream: AsyncGenerator[AgentMessage | T, None]) -> None:
    """
    Displays stream of messages as Streamlit st.info elements.
    """
    with RedirectStdoutToPrint():
        await Console(stream)
