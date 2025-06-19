import multiprocessing
import os
import time

import typer
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.utils import check_not_none

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.agent_db import (
    AgentDB,
    AgentTableHandler,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DeployableAgentNFTGameAbstract,
)


def run_agent_in_process(agent_name: str) -> None:
    try:
        db_agent = AgentTableHandler().get(agent_name)

        # This is run in a new process and we need to set the environment variables for the agent.
        os.environ["BET_FROM_PRIVATE_KEY"] = db_agent.private_key

        if db_agent.safe_address:
            os.environ["SAFE_ADDRESS"] = str(db_agent.safe_address)
        elif os.environ.get("SAFE_ADDRESS"):
            # If agent doesn't specify Safe, but we have some in the env, remove it from the subprocess.
            del os.environ["SAFE_ADDRESS"]

        deployable_agent = DeployableAgentNFTGameAbstract.from_db(db_agent)
        deployable_agent.run(market_type=MarketType.OMEN)
    except Exception:
        logger.exception(f"Agent {agent_name=} encountered an error.")


def spawn_process(agent_name: str) -> multiprocessing.Process:
    return multiprocessing.Process(target=run_agent_in_process, args=(agent_name,))


def monitor_processes(
    processes: dict[str, multiprocessing.Process],
    agent_table_handler: AgentTableHandler,
    agent_group_idx: int | None = None,
    n_groups: int | None = None,
) -> None:
    while True:
        # Check for new agents
        current_agents: list[AgentDB] = list(agent_table_handler.sql_handler.get_all())

        if agent_group_idx is not None and n_groups is not None:
            # Only run agents whose belong to the specified group.
            logger.info(
                f"Filtering for agents in the group {agent_group_idx} / {n_groups}."
            )
            current_agents = [
                agent
                for agent in current_agents
                if check_not_none(agent.id) % n_groups == agent_group_idx
            ]

        logger.info(f"Running agents with ids {[a.id for a in current_agents]}")

        # Identify new agents
        new_agents = {agent.name for agent in current_agents} - set(processes.keys())
        for new_agent in new_agents:
            logger.info(f"New agent {new_agent} detected. Starting process...")
            new_process = spawn_process(new_agent)
            new_process.start()
            processes[new_agent] = new_process

        # Monitor existing processes
        for agent_name, process in processes.items():
            if not process.is_alive():
                logger.warning(f"Agent {agent_name} has stopped. Restarting...")
                new_process = spawn_process(agent_name)
                new_process.start()
                processes[agent_name] = new_process

        time.sleep(5)


def stop_all_processes(processes: dict[str, multiprocessing.Process]) -> None:
    for process in processes.values():
        process.terminate()
    logger.info("All agents have been stopped.")


def main(agent_group_idx: int | None = None, n_groups: int | None = None) -> None:
    """
    If `agent_group_idx` and `n_groups` are specified, script will run agents belonging only to that group, based on their id.
    """
    if (agent_group_idx is None) != (n_groups is None):
        raise ValueError(
            "Both agent_group_idx and n_groups must be set, or neither of them."
        )

    # Spawn a whole new Python interpreter process for each agent,
    # otherwise DB connection freaks. (spawn is default on Mac, but not on Linux, so Kube job fails without this)
    multiprocessing.set_start_method("spawn")

    agent_table_handler = AgentTableHandler()
    processes: dict[str, multiprocessing.Process] = {}

    try:
        monitor_processes(
            processes,
            agent_table_handler,
            agent_group_idx=agent_group_idx,
            n_groups=n_groups,
        )
    except KeyboardInterrupt:
        logger.info("Stopping all agents...")
        stop_all_processes(processes)


if __name__ == "__main__":
    typer.run(main)
