"""
PYTHONPATH=. streamlit run prediction_market_agent/agents/microchain_agent/nft_treasury_game/app_nft_treasury_game.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, instead of just this one.
"""

import typing as t
from datetime import timedelta
from enum import Enum
from math import ceil

import streamlit as st
from eth_typing import ChecksumAddress
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.config import RPCConfig
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.utils import check_not_none
from prediction_market_agent_tooling.tools.web3_utils import (
    generate_private_key,
    private_key_to_public_key,
    wei_to_xdai,
)
from pydantic import SecretStr
from python_web3_wallet import wallet_component
from streamlit_extras.stylable_container import stylable_container

from prediction_market_agent.agents.identifiers import AgentIdentifier
from prediction_market_agent.agents.microchain_agent.agent_functions import (
    UpdateMySystemPrompt,
)
from prediction_market_agent.agents.microchain_agent.nft_functions import BalanceOfNFT
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.agent_db import (
    AgentDB,
    AgentTableHandler,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.agent_prompt_inject import (
    PromptInjectHandler,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    STARTING_AGENT_BALANCE,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    NFTKeysContract,
    SimpleTreasuryContract,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DeployableAgentNFTGameAbstract,
    get_all_nft_agents,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.nft_game_messages_functions import (
    ReceiveMessagesAndPayments,
    RemoveAllUnreadMessages,
    SendPaidMessageToAnotherAgent,
    SleepUntil,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.prompts import (
    nft_treasury_game_base_prompt,
    nft_treasury_game_buyer_prompt,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.tools_nft_treasury_game import (
    get_end_datetime_of_current_round,
    get_start_datetime_of_next_round,
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
from prediction_market_agent.db.report_table_handler import (
    ReportNFTGame,
    ReportNFTGameTableHandler,
)
from prediction_market_agent.tools.anvil.anvil_requests import set_balance
from prediction_market_agent.tools.message_utils import (
    compress_message,
    unzip_message_else_do_nothing,
)

st.set_page_config(
    page_title="Agent's NFT-locked Treasury Game", page_icon="🎮", layout="wide"
)


AgentInputType: t.TypeAlias = (
    type[DeployableAgentNFTGameAbstract] | DeployableAgentNFTGameAbstract | AgentDB
)


class DummyFunctionName(str, Enum):
    # Respones from Microchain's functions don't have a function name to show, so use this dummy one.
    RESPONSE_FUNCTION_NAME = "Response"


@st.cache_resource
def prompt_inject_handler(identifier: AgentIdentifier) -> PromptInjectHandler:
    return PromptInjectHandler.from_agent_identifier(identifier)


@st.cache_resource
def long_term_memory_table_handler(
    identifier: AgentIdentifier,
) -> LongTermMemoryTableHandler:
    return LongTermMemoryTableHandler.from_agent_identifier(identifier)


@st.cache_resource
def prompt_table_handler(identifier: AgentIdentifier) -> PromptTableHandler:
    return PromptTableHandler.from_agent_identifier(identifier)


@st.cache_resource
def report_table_handler() -> ReportNFTGameTableHandler:
    return ReportNFTGameTableHandler()


@st.cache_resource
def agent_table_handler() -> AgentTableHandler:
    return AgentTableHandler()


@st.dialog("Send message to agent")
def send_message_via_wallet(
    recipient: ChecksumAddress, message: str, amount_to_send: float
) -> None:
    wallet_component(
        recipient=recipient,
        amount_in_ether=f"{amount_to_send:.10f}",  # formatting number as 0.0001000 instead of scientific notation
        data=message,
    )


def send_message_part(
    nft_agent: AgentInputType,
) -> None:
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


def prompt_inject_part(
    nft_agent: AgentInputType,
) -> None:
    handler = prompt_inject_handler(nft_agent.identifier)
    message = st.text_area("Prompt to inject (in the form of agent's own reasoning)")
    if st.button("Inject", disabled=not message):
        handler.add(message)
        st.success("Prompt injected successfully!")


def parse_function_and_body(
    role: t.Literal["user", "assistant"], message: str
) -> t.Tuple[str, str]:
    message = message.strip()

    if role == "assistant":
        # Microchain agent is a function calling agent, his outputs are in the form of `SendPaidMessageToAnotherAgent(address='...',message='...')`.
        parsed_function = message.split("(", 1)[0]
        parsed_body = message.split("(", 1)[1].rsplit(")", 1)[0]
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
        case SleepUntil.__name__:
            icon = "⏳"
        case UpdateMySystemPrompt.__name__:
            icon = "📝"
        case ReceiveMessagesAndPayments.__name__:
            icon = "👤"
        case SendPaidMessageToAnotherAgent.__name__:
            icon = "💸"
        case RemoveAllUnreadMessages.__name__:
            icon = "🗑️"
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
            SendPaidMessageToAnotherAgent.__name__,
            RemoveAllUnreadMessages.__name__,
            UpdateMySystemPrompt.__name__,
        ):
            st.markdown(parsed_function_output_body)


def show_function_calls_part(
    nft_agent: AgentInputType,
) -> None:
    st.markdown(f"### Agent's actions")

    n_total_messages = long_term_memory_table_handler(nft_agent.identifier).count()
    messages_per_page = 50
    if "page_number" not in st.session_state:
        st.session_state.page_number = 0

    max_page_number = max(ceil(n_total_messages / messages_per_page) - 1, 0)

    # Define as function callbacks, because otherwise Streamlit web updates logic with 1 step delay.
    def go_to_first_page() -> None:
        st.session_state.page_number = 0

    def go_to_prev_page() -> None:
        st.session_state.page_number = max(0, st.session_state.page_number - 1)

    def go_to_next_page() -> None:
        st.session_state.page_number = min(
            max_page_number, st.session_state.page_number + 1
        )

    def go_to_last_page() -> None:
        st.session_state.page_number = max_page_number

    # Compute disabled statuses based on the current page number
    disable_first_prev = st.session_state.page_number == 0
    disable_next_last = st.session_state.page_number == max_page_number

    # Build the columns and buttons with updated disabled statuses
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.button("First page", disabled=disable_first_prev, on_click=go_to_first_page)
    with col2:
        st.button(
            "Previous page", disabled=disable_first_prev, on_click=go_to_prev_page
        )
    with col3:
        st.write(f"Page {st.session_state.page_number + 1} of {max_page_number + 1}")
    with col4:
        st.button("Next page", disabled=disable_next_last, on_click=go_to_next_page)
    with col5:
        st.button("Last page", disabled=disable_next_last, on_click=go_to_last_page)

    show_function_calls_part_messages(
        nft_agent, messages_per_page, st.session_state.page_number
    )


@st.fragment(run_every=timedelta(seconds=10))
def show_function_calls_part_messages(
    nft_agent: AgentInputType,
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
def show_about_agent_part(
    nft_agent: AgentInputType,
) -> None:
    system_prompt = (
        system_prompt_from_db.prompt
        if (
            system_prompt_from_db := prompt_table_handler(
                nft_agent.identifier
            ).fetch_latest_prompt()
        )
        is not None
        else nft_agent.initial_system_prompt
    )
    xdai_balance = get_balances(nft_agent.wallet_address).xdai
    n_nft = BalanceOfNFT()(NFTKeysContract().address, nft_agent.wallet_address)
    nft_keys_message = (
        "and does not hold any NFT keys"
        if n_nft == 0
        else f"and <span style='font-size: 1.1em;'><strong>{n_nft} NFT key{'s' if n_nft > 1 else ''}</strong></span>"
    )
    st.markdown(
        f"""### {nft_agent.name}

Currently holds <span style='font-size: 1.1em;'><strong>{xdai_balance:.2f} xDAI</strong></span> {nft_keys_message}.

Wallet address: [{nft_agent.wallet_address}](https://gnosisscan.io/address/{nft_agent.wallet_address})

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
    treasury_xdai_balance = SimpleTreasuryContract().balances().xdai
    end_datetime = get_end_datetime_of_current_round()
    start_datetime_next_round = get_start_datetime_of_next_round()
    st.markdown(
        f"""### Treasury
Currently holds <span style='font-size: 1.1em;'><strong>{treasury_xdai_balance:.2f} xDAI</strong></span>. There are {DeployableAgentNFTGameAbstract.retrieve_total_number_of_keys()} NFT keys.

- The current round ends at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}
- The next round starts at: {start_datetime_next_round.strftime('%Y-%m-%d %H:%M:%S') if start_datetime_next_round else 'No next round planned yet.'}
""",
        unsafe_allow_html=True,
    )


def reports_page() -> None:
    handler = report_table_handler()
    reports: t.Sequence[ReportNFTGame] = handler.sql_handler.get_all()
    overall_reports = [report for report in reports if report.is_overall_report]
    overall_reports.sort(key=lambda r: r.datetime_, reverse=True)

    for i, report in enumerate(overall_reports):
        game_number = len(overall_reports) - i
        st.markdown(
            f"""### Game {game_number}, from {report.datetime_.strftime('%Y-%m-%d %H:%M:%S')}
{report.learnings}

---
"""
        )


def add_new_agent() -> None:
    table_handler = agent_table_handler()
    prefill_prompt = st.checkbox("Prefill with default system prompt")

    with st.form("add_agent_form", clear_on_submit=True):
        name = st.text_input("Agent Name")
        initial_system_prompt = st.text_area(
            "Initial System Prompt",
            value=(
                nft_treasury_game_base_prompt() + nft_treasury_game_buyer_prompt()
                if prefill_prompt
                else ""
            ),
        )
        private_key_str = st.text_input(
            "Private Key (optional, will be generated if not provided)", type="password"
        )
        submitted = st.form_submit_button("Add Agent")
        if submitted:
            if not name or not initial_system_prompt:
                st.error("Please fill in all required fields.")

            else:
                private_key = (
                    SecretStr(private_key_str)
                    if private_key_str
                    else generate_private_key()
                )
                public_key = private_key_to_public_key(private_key)
                table_handler.add_agent(
                    AgentDB(
                        name=name,
                        initial_system_prompt=initial_system_prompt,
                        private_key=private_key.get_secret_value(),
                    )
                )
                st.success(f"Agent '{name}' added successfully!")
                set_balance(
                    rpc_url=RPCConfig().gnosis_rpc_url,
                    address=public_key,
                    balance=STARTING_AGENT_BALANCE,
                )
                st.success(
                    f"Agent '{name}' balance set to {STARTING_AGENT_BALANCE} xDai."
                )


def get_agent_page(
    nft_agent: AgentInputType,
) -> t.Callable[[], None]:
    def page() -> None:
        left, _, right = st.columns([0.3, 0.05, 0.65])

        with left:
            show_about_agent_part(nft_agent)

        with right:
            with st.expander("Write message to agent"):
                send_message_part(nft_agent)
            with st.expander("Inject prompt to agent"):
                prompt_inject_part(nft_agent)
            show_function_calls_part(nft_agent)

    return page


def build_url(
    agent: (
        AgentDB | DeployableAgentNFTGameAbstract | type[DeployableAgentNFTGameAbstract]
    ),
) -> str:
    return agent.identifier.lower().replace("_", "-").replace(" ", "-")


with st.sidebar:
    show_treasury_part()


all_agents = get_all_nft_agents()
pages = [
    st.Page(
        get_agent_page(agent), title=f"Agent {agent.name}", url_path=build_url(agent)
    )
    for agent in all_agents
] + [
    st.Page(reports_page, title="Game Reports", url_path="game-reports"),
    st.Page(add_new_agent, title="Add Agent", url_path="add-agent"),
]

pg = st.navigation(pages)  # type: ignore[arg-type] # This is just fine, all items in the pages are from st.Page!
pg.run()
