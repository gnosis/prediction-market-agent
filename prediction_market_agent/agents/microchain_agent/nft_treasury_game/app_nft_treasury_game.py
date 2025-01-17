"""
PYTHONPATH=. streamlit run prediction_market_agent/agents/microchain_agent/nft_treasury_game/app_nft_treasury_game.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, instead of just this one.
"""

import typing as t
from datetime import timedelta
from enum import Enum

import streamlit as st
from eth_typing import ChecksumAddress
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.utils import check_not_none
from prediction_market_agent_tooling.tools.web3_utils import wei_to_xdai
from python_web3_wallet import wallet_component
from streamlit_extras.stylable_container import stylable_container

from prediction_market_agent.agents.identifiers import AgentIdentifier
from prediction_market_agent.agents.microchain_agent.agent_functions import (
    UpdateMySystemPrompt,
)
from prediction_market_agent.agents.microchain_agent.nft_functions import BalanceOfNFT
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    NFT_TOKEN_FACTORY,
    TREASURY_ADDRESS,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts_nft_treasury_game import (
    get_nft_token_factory_max_supply,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DEPLOYED_NFT_AGENTS,
    DeployableAgentNFTGameAbstract,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.messages_functions import (
    BroadcastPublicMessageToHumans,
    GameRoundEnd,
    ReceiveMessage,
    SendPaidMessageToAnotherAgent,
    Wait,
)
from prediction_market_agent.db.agent_communication import (
    fetch_count_unprocessed_transactions,
    fetch_unseen_transactions,
    get_message_minimum_value,
)
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemories,
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.prompt_table_handler import PromptTableHandler
from prediction_market_agent.tools.message_utils import (
    compress_message,
    unzip_message_else_do_nothing,
)

st.set_page_config(
    page_title="Agent's NFT-locked Treasury Game", page_icon="🎮", layout="wide"
)


class DummyFunctionName(str, Enum):
    # Respones from Microchain's functions don't have a function name to show, so use this dummy one.
    RESPONSE_FUNCTION_NAME = "Response"


@st.cache_resource
def long_term_memory_table_handler(
    identifier: AgentIdentifier,
) -> LongTermMemoryTableHandler:
    return LongTermMemoryTableHandler.from_agent_identifier(identifier)


@st.cache_resource
def prompt_table_handler(identifier: AgentIdentifier) -> PromptTableHandler:
    return PromptTableHandler.from_agent_identifier(identifier)


@st.dialog("Send message to agent")
def send_message_via_wallet(
    recipient: ChecksumAddress, message: str, amount_to_send: float
) -> None:
    wallet_component(
        recipient=recipient,
        amount_in_ether=f"{amount_to_send:.10f}",  # formatting number as 0.0001000 instead of scientific notation
        data=message,
    )


def send_message_part(nft_agent: type[DeployableAgentNFTGameAbstract]) -> None:
    message = st.text_area("Write a message to the agent")
    default_value = get_message_minimum_value()
    amount_to_send = st.number_input(
        "Value in xDai",
        min_value=default_value,
        value=default_value,
        format="%.5f",
    )
    message_compressed = HexBytes(compress_message(message)).hex() if message else ""

    if st.button("Send message", disabled=not message):
        send_message_via_wallet(
            recipient=nft_agent.wallet_address,
            message=message_compressed,
            amount_to_send=amount_to_send,
        )


def parse_function_and_body(
    role: t.Literal["user", "assistant"], message: str
) -> t.Tuple[str, str]:
    message = message.strip()

    if role == "assistant":
        # Microchain agent is a function calling agent, his outputs are in the form of `SendPaidMessageToAnotherAgent(address='...',message='...')`.
        parsed_function = message.split("(")[0]
        parsed_body = message.split("(")[1].rsplit(")")[0]
    elif role == "user":
        # Responses from the individual functions are stored under `user` role.
        parsed_function = DummyFunctionName.RESPONSE_FUNCTION_NAME
        parsed_body = message
    else:
        raise ValueError(f"Unknown role: {role}")

    return parsed_function, parsed_body


def customized_chat_message(
    function_call: LongTermMemories,
    function_output: LongTermMemories,
) -> None:
    created_at = function_output.datetime_

    parsed_function_call_name, parsed_function_call_body = parse_function_and_body(
        check_not_none(function_call.metadata_dict)["role"],
        check_not_none(function_call.metadata_dict)["content"],
    )
    parsed_function_output_name, parsed_function_output_body = parse_function_and_body(
        check_not_none(function_output.metadata_dict)["role"],
        check_not_none(function_output.metadata_dict)["content"],
    )

    match parsed_function_call_name:
        case Reasoning.__name__:
            icon = "🧠"
        case Stop.__name__:
            icon = "😴"
        case Wait.__name__:
            icon = "⏳"
        case UpdateMySystemPrompt.__name__:
            icon = "📝"
        case GameRoundEnd.__name__:
            icon = "🏁"
        case ReceiveMessage.__name__:
            icon = "👤"
        case BroadcastPublicMessageToHumans.__name__:
            icon = "📣"
        case SendPaidMessageToAnotherAgent.__name__:
            icon = "💸"
        case _:
            icon = "🤖"

    with st.chat_message(icon):
        if parsed_function_call_name == Reasoning.__name__:
            # Don't show reasoning as function call, to make it a bit nicer.
            st.markdown(
                parsed_function_call_body.replace("reasoning='", "")
                .replace("')", "")
                .replace('reasoning="', "")
                .replace('")', "")
            )
        elif parsed_function_call_name == Stop.__name__:
            # If the agent decided to stop, show it as a break, as it will be started soon again.
            st.markdown("Taking a break.")
        else:
            # Otherwise, show it as a normal function-response call, e.g. `ReceiveMessages() -> ...`.
            st.markdown(
                f"**{parsed_function_call_name}**({parsed_function_call_body}) *{created_at.strftime('%Y-%m-%d %H:%M:%S')}*"
            )

        # Only show the output if it's supposed to be interesting.
        if parsed_function_call_name not in (
            Reasoning.__name__,
            Stop.__name__,
            BroadcastPublicMessageToHumans.__name__,
            SendPaidMessageToAnotherAgent.__name__,
            Wait.__name__,
            GameRoundEnd.__name__,
            UpdateMySystemPrompt.__name__,
        ):
            st.markdown(parsed_function_output_body)


def show_function_calls_part(nft_agent: type[DeployableAgentNFTGameAbstract]) -> None:
    st.markdown(f"""### Agent's actions""")

    n_total_messages = long_term_memory_table_handler(nft_agent.identifier).count()
    messages_per_page = 50
    if "page_number" not in st.session_state:
        st.session_state.page_number = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Previous page", disabled=st.session_state.page_number == 0):
            st.session_state.page_number -= 1
    with col2:
        if st.button(
            "Next page",
            disabled=st.session_state.page_number
            == n_total_messages // messages_per_page,
        ):
            st.session_state.page_number += 1
    with col3:
        st.write(f"Current page {st.session_state.page_number + 1}")

    show_function_calls_part_messages(
        nft_agent, messages_per_page, st.session_state.page_number
    )


@st.fragment(run_every=timedelta(seconds=10))
def show_function_calls_part_messages(
    nft_agent: type[DeployableAgentNFTGameAbstract],
    messages_per_page: int,
    page_number: int,
) -> None:
    with st.spinner("Loading agent's actions..."):
        calls = long_term_memory_table_handler(nft_agent.identifier).search(
            offset=page_number * messages_per_page,
            limit=messages_per_page,
        )

    if not calls:
        st.markdown("No actions yet.")
        return

    # Filter out system calls, because they aren't supposed to be shown in the chat history itself.
    calls = [
        call for call in calls if check_not_none(call.metadata_dict)["role"] != "system"
    ]

    # Microchain works on `function call` - `funtion response` pairs, so we will process them together.
    for index, (function_output, function_call) in enumerate(
        zip(calls[::2], calls[1::2])
    ):
        with stylable_container(
            key=f"function_call_{index}",
            css_styles=f"{{background-color: {'#f0f0f0' if (index % 2 == 0) else 'white'}; border-radius: 5px;}}",
        ):
            customized_chat_message(function_call, function_output)


@st.fragment(run_every=timedelta(seconds=10))
def show_about_agent_part(nft_agent: type[DeployableAgentNFTGameAbstract]) -> None:
    system_prompt = (
        system_prompt_from_db.prompt
        if (
            system_prompt_from_db := prompt_table_handler(
                nft_agent.identifier
            ).fetch_latest_prompt()
        )
        is not None
        else nft_agent.get_initial_system_prompt()
    )
    xdai_balance = get_balances(nft_agent.wallet_address).xdai
    n_nft = BalanceOfNFT()(NFT_TOKEN_FACTORY, nft_agent.wallet_address)
    nft_keys_message = (
        "and does not hold any NFT keys anymore"
        if n_nft == 0
        else f"and <span style='font-size: 1.1em;'><strong>{n_nft} NFT key{'s' if n_nft > 1 else ''}</strong></span>"
    )
    st.markdown(
        f"""### {nft_agent.name}

Currently holds <span style='font-size: 1.1em;'><strong>{xdai_balance:.2f} xDAI</strong></span> {nft_keys_message}.

---
""",
        unsafe_allow_html=True,
    )
    st.text_area(
        f"{nft_agent.name}'s system prompt",
        value=system_prompt,
        disabled=True,
    )
    st.markdown("---")
    with st.popover("Show unprocessed incoming messages"):
        show_n = 10
        n_messages = fetch_count_unprocessed_transactions(nft_agent.wallet_address)
        messages = fetch_unseen_transactions(nft_agent.wallet_address, n=show_n)

        if not messages:
            st.info("No unprocessed messages")
        else:
            for message in messages:
                st.markdown(
                    f"""
                    **From:** {message.sender}  
                    **Message:** {unzip_message_else_do_nothing(message.message.hex())}  
                    **Value:** {wei_to_xdai(message.value)} xDai
                    """
                )
                st.divider()

            if n_messages > show_n:
                st.write(f"... and another {n_messages - show_n} unprocessed messages.")


@st.fragment(run_every=timedelta(seconds=10))
def show_treasury_part() -> None:
    treasury_xdai_balance = get_balances(TREASURY_ADDRESS).xdai
    st.markdown(
        f"""### Treasury
Currently holds <span style='font-size: 1.1em;'><strong>{treasury_xdai_balance:.2f} xDAI</strong></span>. There are {get_nft_token_factory_max_supply()} NFT keys.""",
        unsafe_allow_html=True,
    )


def get_agent_page(
    nft_agent: type[DeployableAgentNFTGameAbstract],
) -> t.Callable[[], None]:
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
        st.Page(get_agent_page(agent), title=agent.name, url_path=agent.get_url())
        for agent in DEPLOYED_NFT_AGENTS
    ]
)
pg.run()
