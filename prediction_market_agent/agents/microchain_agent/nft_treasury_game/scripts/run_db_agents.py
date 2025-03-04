import multiprocessing
import os
import time

import typer
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.agent_db import (
    AgentDB,
    AgentIdentifier,
    AgentTableHandler,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DeployableAgentNFTGameAbstract,
)


def run_agent_in_process(agent_identifier: AgentIdentifier) -> None:
    try:
        db_agent = AgentTableHandler().get(agent_identifier)

        # This is run in a new process and we need to set the environment variables for the agent.
        os.environ["BET_FROM_PRIVATE_KEY"] = db_agent.private_key

        if db_agent.safe_address:
            os.environ["SAFE_ADDRESS"] = str(db_agent.safe_address)
        elif os.environ.get("SAFE_ADDRESS"):
            del os.environ["SAFE_ADDRESS"]

        deployable_agent = DeployableAgentNFTGameAbstract.from_db(db_agent)
        deployable_agent.run(market_type=MarketType.OMEN)
    except Exception:
        logger.exception(f"Agent {agent_identifier} encountered an error.")


def spawn_process(agent_identifier: AgentIdentifier) -> multiprocessing.Process:
    return multiprocessing.Process(
        target=run_agent_in_process, args=(agent_identifier,)
    )


def monitor_processes(
    processes: dict[AgentIdentifier, multiprocessing.Process]
) -> None:
    while True:
        for agent_identifier, process in processes.items():
            if not process.is_alive():
                logger.warning(f"Agent {agent_identifier} has stopped. Restarting...")
                new_process = spawn_process(agent_identifier)
                new_process.start()
                processes[agent_identifier] = new_process

        time.sleep(5)


def stop_all_processes(
    processes: dict[AgentIdentifier, multiprocessing.Process]
) -> None:
    for process in processes.values():
        process.terminate()
    logger.info("All agents have been stopped.")


def main() -> None:
    agent_table_handler = AgentTableHandler()
    all_agents: list[AgentDB] = list(agent_table_handler.sql_handler.get_all())

    processes: dict[AgentIdentifier, multiprocessing.Process] = {}
    for agent in all_agents:
        processes[agent.identifier] = spawn_process(agent.identifier)
        processes[agent.identifier].start()

    try:
        monitor_processes(processes)
    except KeyboardInterrupt:
        logger.info("Stopping all agents...")
        stop_all_processes(processes)


if __name__ == "__main__":
    typer.run(main)
