import datetime
import textwrap

from langchain_core.prompts import PromptTemplate
from prediction_market_agent_tooling.gtypes import xdai_type, ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.contract import (
    ContractOwnableERC721OnGnosisChain,
)
from prediction_market_agent_tooling.tools.parallelism import par_map
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.memory import DatedChatMessage
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    NFT_TOKEN_FACTORY,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DEPLOYED_NFT_AGENTS,
)
from prediction_market_agent.agents.utils import (
    _summarize_learnings,
)
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.models import LongTermMemories

SUMMARY_PROMPT_TEMPLATE = """
Summarize the memories below. They contain summarizations of the actions of AI agents competing on an NFT game.

Memories:
{memories}"""

FINAL_SUMMARY_PROMPT_TEMPLATE = """
Make a final summary of a collection of summaries from each agent. Encompass the main activities that took place involving the agents.

Memories:
{memories}"""


def get_nft_balance(owner_address: ChecksumAddress, web3: Web3) -> int:
    contract = ContractOwnableERC721OnGnosisChain(
        address=Web3.to_checksum_address(NFT_TOKEN_FACTORY)
    )
    balance: int = contract.balanceOf(
        Web3.to_checksum_address(owner_address), web3=web3
    )
    return balance


def fetch_memories_from_last_run() -> dict[str, list[LongTermMemories]]:
    entries_from_latest_run: dict[str, list[LongTermMemories]] = {}
    for agent in DEPLOYED_NFT_AGENTS:
        logger.info(f"Fetching memories from {agent.identifier}")
        ltm = LongTermMemoryTableHandler.from_agent_identifier(agent.identifier)
        entries_for_agent = ltm.search()
        # We need the 2nd GameRoundEnd. If there is only one, then start_date should be None.
        game_round_occurances = [
            i
            for i in entries_for_agent
            if "GameRoundEnd(" in i.metadata_dict["content"]
        ]
        # We initially assume all memories should be processed.
        # If there are at least 2 end_game occurrances, at least 1 run was completed. Hence
        # we fetch the memories from the latest run.
        if len(game_round_occurances) > 1:
            # 2nd entry indicates the beginning of the latest run.
            last_entry_before_latest_run = game_round_occurances[1]
            start_date = last_entry_before_latest_run.datetime_ + datetime.timedelta(
                seconds=1
            )
            entries_for_agent = ltm.search(from_=start_date)

        logger.info(
            f"Fetched {len(entries_for_agent)} memories from {agent.identifier} latest run"
        )
        entries_from_latest_run[agent.identifier] = entries_for_agent

    return entries_from_latest_run


def summarize_past_actions_from_agent(agent_memories: list[LongTermMemories]) -> str:
    dated_chat_messages = [
        DatedChatMessage.from_long_term_memory(ltm) for ltm in agent_memories
    ]
    prompt = PromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
    learnings_per_agent = _summarize_learnings(
        memories=[str(m) for m in dated_chat_messages], prompt_template=prompt
    )
    return learnings_per_agent


def summarize_prompts_from_all_agents() -> str:
    memories_last_run = fetch_memories_from_last_run()
    learnings: list[str] = par_map(
        items=list(memories_last_run.values()), func=summarize_past_actions_from_agent
    )

    prompt = PromptTemplate.from_template(FINAL_SUMMARY_PROMPT_TEMPLATE)
    final_summary = _summarize_learnings(memories=learnings, prompt_template=prompt)
    return final_summary


def print_key_movements(rpc_url: str) -> None:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    for agent in DEPLOYED_NFT_AGENTS:
        print(agent.identifier)
        balance = get_balances(
            address=Web3.to_checksum_address(agent.wallet_address), web3=w3
        )
        print(balance)
        print(get_nft_balance(owner_address=agent.wallet_address, web3=w3))


def generate_report() -> None:
    rpc_url = "https://remote-anvil-2.ai.gnosisdev.com"
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    lookup = {agent.wallet_address: agent.identifier for agent in DEPLOYED_NFT_AGENTS}
    initial_balance = xdai_type(200)
    for agent_address, agent_id in lookup.items():
        balance = get_balances(address=Web3.to_checksum_address(agent_address), web3=w3)
        diff_xdai_balance = balance.xdai - initial_balance
        nft_balance = get_nft_balance(owner_address=agent_address, web3=w3)
        print(f"{agent_id} {diff_xdai_balance=:.2f} {nft_balance=}")

    learnings = summarize_prompts_from_all_agents()
    wrapped_text = textwrap.fill(learnings, width=80)
    with open("report.txt", "w") as file:
        file.write(wrapped_text)


if __name__ == "__main__":
    generate_report()
