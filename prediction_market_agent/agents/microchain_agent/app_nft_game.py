"""
PYTHONPATH=. streamlit run prediction_market_agent/agents/microchain_agent/app_nft_game.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, instead of just this one.
"""

import typing as t
from datetime import timedelta

import streamlit as st
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.datetime_utc import DatetimeUTC

from prediction_market_agent.agents.identifiers import AgentIdentifier
from prediction_market_agent.agents.microchain_agent.agents_nft_game import (
    AGENTS_IN_THE_GAME,
    TREASURY_SAFE_ADDRESS,
    NFTAgent,
)
from prediction_market_agent.agents.microchain_agent.messages_functions import (
    BroadcastPublicMessageToHumans,
    ReceiveMessage,
    SendPaidMessageToAnotherAgent,
)
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.prompt_table_handler import PromptTableHandler

st.set_page_config(
    page_title="Agent's NFT-locked Treasury Game", page_icon="🎮", layout="wide"
)


@st.cache_resource
def long_term_memory_table_handler(
    identifier: AgentIdentifier,
) -> LongTermMemoryTableHandler:
    return LongTermMemoryTableHandler.from_agent_identifier(identifier)


@st.cache_resource
def prompt_table_handler(identifier: AgentIdentifier) -> PromptTableHandler:
    return PromptTableHandler.from_agent_identifier(identifier)


def send_message_part(nft_agent: NFTAgent) -> None:
    message = st.text_area("Write a message to the agent")

    if st.button("Send message", disabled=not message):
        # TODO: Don't do this manually with deployment private key, use the user's wallet!
        SendPaidMessageToAnotherAgent()(nft_agent.wallet_address, message)
        st.success("Message sent and will be processed soon!")


def parse_function_and_body(
    role: t.Literal["user", "assistant", "system"], message: str
) -> t.Tuple[str | None, str | None]:
    message = message.strip()

    if role == "assistant":
        # Microchain agent is a function calling agent, his outputs are in the form of `SendPaidMessageToAnotherAgent(address='...',message='...')`.
        parsed_function = message.split("(")[0]
        parsed_body = message.split("(")[1].rsplit(")")[0]
    elif role == "user":
        # Responses from the individual functions are stored under `user` role.
        parsed_function = "Response"
        parsed_body = message
    elif role == "system":
        # System message isn't shown in the chat history, so ignore.
        parsed_function = None
        parsed_body = None
    else:
        raise ValueError(f"Unknown role: {role}")

    return parsed_function, parsed_body


def customized_chat_message(
    role: t.Literal["user", "assistant", "system"],
    message: str,
    created_at: DatetimeUTC,
) -> None:
    parsed_function, parsed_body = parse_function_and_body(role, message)
    if parsed_function is None:
        return

    match parsed_function:
        case "Response":
            icon = "✔️"
        case ReceiveMessage.__name__:
            icon = "👤"
        case BroadcastPublicMessageToHumans.__name__:
            icon = "📣"
        case SendPaidMessageToAnotherAgent.__name__:
            icon = "💸"
        case _:
            icon = "🤖"

    with st.chat_message(icon):
        if parsed_function:
            st.markdown(f"**{parsed_function}**")
        st.write(created_at.strftime("%Y-%m-%d %H:%M:%S"))
        if message:
            st.markdown(parsed_body)


@st.fragment(run_every=timedelta(seconds=5))
def show_function_calls_part(nft_agent: NFTAgent) -> None:
    st.markdown(f"""### Agent's actions""")

    with st.spinner("Loading agent's actions..."):
        calls = long_term_memory_table_handler(nft_agent.identifier).search()

    if not calls:
        st.markdown("No actions yet.")
        return

    for item in calls:
        if item.metadata_dict is None:
            continue
        customized_chat_message(
            item.metadata_dict["role"],
            item.metadata_dict["content"],
            item.datetime_,
        )


@st.fragment(run_every=timedelta(seconds=5))
def show_about_agent_part(nft_agent: NFTAgent) -> None:
    system_prompt = prompt_table_handler(nft_agent.identifier).fetch_latest_prompt()
    xdai_balance = get_balances(nft_agent.wallet_address).xdai
    st.markdown(
        f"""### {nft_agent.name}

Currently holds <span style='font-size: 1.1em;'><strong>{xdai_balance:.2f} xDAI</strong></span>.

---
""",
        unsafe_allow_html=True,
    )
    st.text_area(
        f"{nft_agent.name}'s system prompt",
        value=system_prompt.prompt if system_prompt else "No system prompt yet.",
        disabled=True,
    )


@st.fragment(run_every=timedelta(seconds=5))
def show_treasury_part() -> None:
    treasury_xdai_balance = get_balances(TREASURY_SAFE_ADDRESS).xdai
    st.markdown(
        f"""### Treasury
Currently holds <span style='font-size: 1.1em;'><strong>{treasury_xdai_balance:.2f} xDAI</strong></span>.""",
        unsafe_allow_html=True,
    )


def get_agent_page(nft_agent: NFTAgent) -> t.Callable[[], None]:
    def page() -> None:
        left, _, right = st.columns([0.3, 0.05, 0.65])

        with left:
            show_about_agent_part(nft_agent)

        with right:
            send_message_part(nft_agent)
            show_function_calls_part(nft_agent)

    return page


with st.sidebar:
    show_treasury_part()

pg = st.navigation(
    [
        st.Page(get_agent_page(agent), title=agent.name, url_path=agent.url)
        for agent in AGENTS_IN_THE_GAME
    ]
)
pg.run()