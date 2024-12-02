import sys
from contextlib import contextmanager
from typing import AsyncGenerator, Iterator, TypeVar

import streamlit as st
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import AgentMessage
from autogen_agentchat.task._console import Console

T = TypeVar("T", bound=TaskResult | Response)


class PrintRedirector:
    @staticmethod
    def write(text: str) -> None:
        st.info(text)


# Context manager to temporarily redirect sys.stdout
@contextmanager
def redirect_stdout_to_print() -> Iterator[None]:
    original_stdout = sys.stdout
    sys.stdout = PrintRedirector()
    try:
        yield
    finally:
        sys.stdout = original_stdout


async def streamlit_console(stream: AsyncGenerator[AgentMessage | T, None]) -> None:
    """
    Displays stream of messages as Streamlit st.info elements.
    """
    with redirect_stdout_to_print():
        await Console(stream)
