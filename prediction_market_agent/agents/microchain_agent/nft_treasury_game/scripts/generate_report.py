from langchain_core.prompts import PromptTemplate
from prediction_market_agent_tooling.gtypes import ChecksumAddress, xdai_type
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.contract import (
    ContractOwnableERC721OnGnosisChain,
)
from prediction_market_agent_tooling.tools.datetime_utc import DatetimeUTC
from prediction_market_agent_tooling.tools.parallelism import par_map
from tabulate import tabulate
from tenacity import retry, stop_after_attempt, wait_fixed
from web3 import Web3

from prediction_market_agent.agents.identifiers import AgentIdentifier
from prediction_market_agent.agents.microchain_agent.memory import DatedChatMessage
from prediction_market_agent.agents.microchain_agent.memory_functions import (
    fetch_memories_from_last_run,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    NFT_TOKEN_FACTORY,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DEPLOYED_NFT_AGENTS,
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


def get_nft_balance(owner_address: ChecksumAddress, web3: Web3) -> int:
    contract = ContractOwnableERC721OnGnosisChain(
        address=Web3.to_checksum_address(NFT_TOKEN_FACTORY)
    )
    balance: int = contract.balanceOf(
        Web3.to_checksum_address(owner_address), web3=web3
    )
    return balance


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
    final_summary: str, learnings_per_agent: dict[AgentIdentifier, str]
) -> None:
    table_handler = ReportNFTGameTableHandler()
    for agent_id, learnings in learnings_per_agent.items():
        report = build_report(agent_id=agent_id, learnings=learnings)
        logger.info(f"Saving report from {agent_id}")
        table_handler.save_report(report)

    final_report = build_report(agent_id=None, learnings=final_summary)
    logger.info(f"Saving final summary")
    table_handler.save_report(final_report)
    pass


def summarize_prompts_from_all_agents() -> tuple[dict[AgentIdentifier, str], str]:
    memories_last_run = fetch_memories_from_last_run(
        agent_identifiers=[i.identifier for i in DEPLOYED_NFT_AGENTS]
    )
    # We generate the learnings from each agent's memories.
    learnings: list[str] = par_map(
        items=list(memories_last_run.values()), func=summarize_past_actions_from_agent
    )

    # We combine each agent's memories into a final summary.
    final_summary = _summarize_learnings(
        memories=learnings,
        prompt_template=PromptTemplate.from_template(FINAL_SUMMARY_PROMPT_TEMPLATE),
    )

    learnings_per_agent = {
        agent.identifier: learnings_from_agent
        for agent, learnings_from_agent in zip(DEPLOYED_NFT_AGENTS, learnings)
    }
    return learnings_per_agent, final_summary


def calculate_nft_and_xdai_balances_diff(lookup, w3, initial_balance):
    # We retry the functions below due to RPC errors that can occur.
    get_balances_retry = retry(stop=stop_after_attempt(3), wait=wait_fixed(1))(
        get_balances
    )
    get_nft_balance_retry = retry(stop=stop_after_attempt(3), wait=wait_fixed(1))(
        get_nft_balance
    )

    balances_diff = []
    for agent_address, agent_id in lookup.items():
        balance = get_balances_retry(
            address=Web3.to_checksum_address(agent_address), web3=w3
        )
        # how much each agent won/lost during the game.
        diff_xdai_balance = balance.xdai - initial_balance
        # How many NFTs the agents ended the game with.
        nft_balance = get_nft_balance_retry(owner_address=agent_address, web3=w3)
        logger.info(f"{agent_id} {diff_xdai_balance=:.2f} {nft_balance=}")
        balances_diff.append(
            {
                "agent_id": agent_id.value,
                "xdai_difference": f"{diff_xdai_balance:.2f}",
                "nft_balance_end": nft_balance,
            }
        )
    return balances_diff


def build_report(learnings: str, agent_id: str | None = None) -> ReportNFTGame:
    return ReportNFTGame(
        agent_id=agent_id, learnings=learnings, datetime_=DatetimeUTC.now()
    )


def generate_report(rpc_url: str, initial_xdai_balance_per_agent: int = 200) -> None:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    lookup = {agent.wallet_address: agent.identifier for agent in DEPLOYED_NFT_AGENTS}
    # Initial balance of each agent at the beginning of the game.
    initial_balance = xdai_type(initial_xdai_balance_per_agent)

    balances_diff = calculate_nft_and_xdai_balances_diff(lookup, w3, initial_balance)

    (
        learnings_per_agent,
        final_summary,
    ) = summarize_prompts_from_all_agents()
    balances_data = tabulate(
        [x.values() for x in balances_diff], list(balances_diff[0].keys())
    )
    final_summary = balances_data + "\n\n---\n" + final_summary
    store_all_learnings_in_db(
        final_summary=final_summary, learnings_per_agent=learnings_per_agent
    )
