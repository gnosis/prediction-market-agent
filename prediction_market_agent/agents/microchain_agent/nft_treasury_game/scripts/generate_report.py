import typing as t

from langchain_core.prompts import PromptTemplate
from prediction_market_agent_tooling.gtypes import ChecksumAddress, xDai
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.parallelism import par_map
from prediction_market_agent_tooling.tools.utils import check_not_none, utcnow
from tenacity import retry, stop_after_attempt, wait_fixed
from web3 import Web3

from prediction_market_agent.agents.identifiers import AgentIdentifier
from prediction_market_agent.agents.microchain_agent.memory import DatedChatMessage
from prediction_market_agent.agents.microchain_agent.memory_functions import (
    fetch_memories_from_game_round,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    NFTKeysContract,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    get_all_nft_agents,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.game_history import (
    NFTGameRound,
)
from prediction_market_agent.agents.utils import _summarize_learnings
from prediction_market_agent.db.models import LongTermMemories, ReportNFTGame
from prediction_market_agent.db.report_table_handler import ReportNFTGameTableHandler

SUMMARY_PROMPT_TEMPLATE = """
Summarize the memories below. They represent the actions taken by AI agents competing on an NFT game.
You must include the key transfers in the summary, and also the messages which led to these keys being transferred.

Memories:
{memories}"""

FINAL_SUMMARY_PROMPT_TEMPLATE = """
Make a final summary of a collection of memories from each agent. Describe the main activities that took place on the game, specially which keys were transferred and which messages yielded best results.

Memories:
{memories}
"""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def get_nft_balance(owner_address: ChecksumAddress, web3: Web3) -> int:
    contract = NFTKeysContract()
    balance = contract.balanceOf(Web3.to_checksum_address(owner_address), web3=web3)
    return balance.value


def summarize_past_actions_from_agent(agent_memories: list[LongTermMemories]) -> str:
    dated_chat_messages = [
        DatedChatMessage.from_long_term_memory(ltm) for ltm in agent_memories
    ]
    prompt = PromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
    learnings_per_agent = _summarize_learnings(
        memories=[str(m) for m in dated_chat_messages], prompt_template=prompt
    )
    return learnings_per_agent


def store_all_learnings_in_db(
    last_round: NFTGameRound,
    final_summary: str,
    learnings_per_agent: dict[AgentIdentifier, str],
) -> None:
    table_handler = ReportNFTGameTableHandler()
    for agent_id, learnings in learnings_per_agent.items():
        report = ReportNFTGame(
            game_round_id=check_not_none(last_round.id),
            agent_id=agent_id,
            learnings=learnings,
            datetime_=utcnow(),
        )
        logger.info(f"Saving report from {agent_id}")
        table_handler.save_report(report)

    final_report = ReportNFTGame(
        game_round_id=check_not_none(last_round.id),
        agent_id=None,
        learnings=final_summary,
        datetime_=utcnow(),
    )

    logger.info(f"Saving final summary")
    table_handler.save_report(final_report)


def summarize_prompts_from_all_agents(
    round_: NFTGameRound,
) -> tuple[dict[AgentIdentifier, str], str]:
    nft_agents = get_all_nft_agents()

    memories = fetch_memories_from_game_round(
        round_, agent_identifiers=[i.identifier for i in nft_agents]
    )
    # We generate the learnings from each agent's memories.
    learnings: list[str] = par_map(
        items=list(memories.values()), func=summarize_past_actions_from_agent
    )

    # We combine each agent's memories into a final summary.
    final_summary = _summarize_learnings(
        memories=learnings,
        prompt_template=PromptTemplate.from_template(FINAL_SUMMARY_PROMPT_TEMPLATE),
    )

    learnings_per_agent = {
        agent.identifier: learnings_from_agent
        for agent, learnings_from_agent in zip(nft_agents, learnings)
    }
    return learnings_per_agent, final_summary


def calculate_nft_and_xdai_balances_diff(
    rpc_url: str, initial_balance: xDai
) -> list[dict[str, t.Any]]:
    nft_agents = get_all_nft_agents()

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    lookup = {agent.wallet_address: agent.name for agent in nft_agents}

    balances_diff = []
    for agent_address, agent_name in lookup.items():
        balance = get_balances(address=Web3.to_checksum_address(agent_address), web3=w3)
        # how much each agent won/lost during the game.
        diff_xdai_balance = balance.xdai - initial_balance
        # How many NFTs the agents ended the game with.
        nft_balance = get_nft_balance(owner_address=agent_address, web3=w3)
        logger.info(f"{agent_name} {diff_xdai_balance.value=:.2f} {nft_balance=}")
        balances_diff.append(
            {
                "agent_name": agent_name,
                "xdai_difference": f"{diff_xdai_balance.value:.2f}",
                "nft_balance_end": nft_balance,
            }
        )
    return balances_diff


def format_markdown_table(data: list[dict[str, t.Any]]) -> str:
    """Format data as a markdown table string."""
    headers = list(data[0].keys())
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---" for _ in headers]) + " |"

    data_rows = []
    for row in data:
        data_rows.append("| " + " | ".join(str(x) for x in row.values()) + " |")

    return "\n".join([header_row, separator_row] + data_rows)


def generate_report(
    last_round: NFTGameRound,
    rpc_url: str,
    initial_xdai_balance_per_agent: xDai,
) -> None:
    balances_diff = calculate_nft_and_xdai_balances_diff(
        rpc_url=rpc_url, initial_balance=initial_xdai_balance_per_agent
    )

    (
        learnings_per_agent,
        final_summary,
    ) = summarize_prompts_from_all_agents(last_round)

    balances_data = format_markdown_table(balances_diff)
    final_summary = balances_data + "\n\n---\n" + final_summary
    store_all_learnings_in_db(
        last_round=last_round,
        final_summary=final_summary,
        learnings_per_agent=learnings_per_agent,
    )
